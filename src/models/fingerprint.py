"""
TinyShip-Fingerprint embedding model.

Same MobileNetV3-Small backbone as the ShipNN reproduction (keeps the on-device
story), but the classifier head is replaced by an L2-normalized embedding plus a
margin head used ONLY at training time. At evaluation we discard the head and use
the embeddings directly for enrollment/retrieval on UNSEEN ships.

Heads / pooling (train-time unless noted):
  ArcFace           -- angular-margin softmax over TRAIN identities
  SubCenterArcFace  -- K sub-centers per identity (absorbs intra-class modes)
  GatedAttentionPool -- set-level pooling over clip embeddings (eval + train)
  StatisticsPool    -- mean+std concat over clip set; eval; output dim 2*emb_dim
  AttentiveStatsPool -- ECAPA-style ASP over clip embeddings (eval + train)
  TemporalGRUPool   -- biGRU over clip sequence (E1b)
  TemporalTCNPool   -- tiny 1-D TCN over clip sequence (E1b)
  MultiHeadAttentionPool -- learned-query MHA pooling (E1b)
  Triplet           -- ablation; embeddings + triplet loss

Verified separately: 25%-width MobileNetV3-Small forwards (B,2,95,T) fine.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf, MobileNetV3


class Encoder(nn.Module):
    """Spectrogram -> L2-normalized embedding of dim `emb_dim`."""
    def __init__(self, in_channels: int = 2, emb_dim: int = 128, width_mult: float = 0.25):
        super().__init__()
        conf, last = _mobilenet_v3_conf("mobilenet_v3_small", width_mult=width_mult)
        net = MobileNetV3(conf, last, num_classes=emb_dim)     # reuse its head slot
        if in_channels != 3:
            stem = net.features[0][0]
            net.features[0][0] = nn.Conv2d(in_channels, stem.out_channels,
                                           kernel_size=stem.kernel_size, stride=stem.stride,
                                           padding=stem.padding, bias=False)
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = net.classifier[0].in_features
        self.proj = nn.Linear(feat_dim, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x, normalize: bool = True):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return F.normalize(x, dim=1) if normalize else x


class ArcFaceHead(nn.Module):
    """Additive angular margin head (Deng et al. 2019). Train-time only."""
    def __init__(self, emb_dim: int, n_classes: int, s: float = 30.0, m: float = 0.30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, emb_dim))
        nn.init.xavier_normal_(self.W)
        self.s, self.m = s, m

    def forward(self, emb, labels):
        Wn = F.normalize(self.W, dim=1)
        cos = emb @ Wn.t()                                   # emb already normalized
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(labels, num_classes=self.W.size(0)).float()
        logits = self.s * (onehot * target + (1 - onehot) * cos)
        return logits


class SubCenterArcFaceHead(nn.Module):
    """
    Sub-center ArcFace (Deng et al. CVPR 2020): K sub-centers per identity.
    Softmax uses the *max* cosine among the K sub-centers of the true class,
    which absorbs intra-class modes (different SNRs / ranges) without collapsing
    them into one noisy prototype.
    """
    def __init__(self, emb_dim: int, n_classes: int, K: int = 3,
                 s: float = 16.0, m: float = 0.20):
        super().__init__()
        self.K = int(K)
        self.W = nn.Parameter(torch.randn(n_classes * self.K, emb_dim))
        nn.init.xavier_normal_(self.W)
        self.s, self.m = s, m
        self.n_classes = n_classes

    def forward(self, emb, labels):
        # emb: (B, D) L2-normalized
        Wn = F.normalize(self.W, dim=1)                       # (C*K, D)
        cos_all = emb @ Wn.t()                                # (B, C*K)
        cos_all = cos_all.clamp(-1 + 1e-7, 1 - 1e-7)
        cos = cos_all.view(-1, self.n_classes, self.K).max(dim=2).values  # (B, C)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(labels, num_classes=self.n_classes).float()
        logits = self.s * (onehot * target + (1 - onehot) * cos)
        return logits


class GatedAttentionPool(nn.Module):
    """
    Gated attention over a set of clip embeddings (Ilse et al. 2018 style).
    Input:  (B, W, D) or (W, D)  — already L2-normalized clip embeddings
    Output: (B, D) pooled embedding + attention weights (B, W)
    Tiny add-on: runs at enrollment/probe time; encoder stays 1-second MCU-sized.
    """
    def __init__(self, emb_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.V = nn.Linear(emb_dim, hidden, bias=False)
        self.U = nn.Linear(emb_dim, hidden, bias=False)
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, H, mask=None):
        # H: (B, W, D)
        if H.dim() == 2:
            H = H.unsqueeze(0)
        gated = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))   # (B, W, h)
        scores = self.w(gated).squeeze(-1)                         # (B, W)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)                       # (B, W)
        pooled = (attn.unsqueeze(-1) * H).sum(dim=1)               # (B, D)
        pooled = F.normalize(pooled, dim=-1)
        return pooled, attn


class StatisticsPool(nn.Module):
    """
    Parameter-free mean+std statistics pooling over a clip set (eval).
    Input:  (B, W, D) or (W, D) — clip embeddings
    Output: (B, 2D) L2-normalized. Output dim is 2*emb_dim.
    """
    def forward(self, H):
        if H.dim() == 2:
            H = H.unsqueeze(0)
        mu = H.mean(dim=1)
        # population std (unbiased=False), same as numpy ddof=0
        sigma = H.std(dim=1, unbiased=False).clamp(min=1e-9)
        pooled = torch.cat([mu, sigma], dim=-1)
        return F.normalize(pooled, dim=-1)


class AttentiveStatsPool(nn.Module):
    """
    ECAPA-style attentive statistics pooling (Desplanques et al., Interspeech 2020).
    Learnable attention over clip embeddings (not spectrogram frames; not Ilse gated pool).
    Standard ASP: alpha = softmax(scores over W); mu = sum alpha*h;
    sigma = sqrt(sum alpha*h^2 - mu^2).clamp; concat [mu, sigma];
    Linear(2D -> D); L2-normalize.
    Input: (B, W, D) or (W, D). Output: (B, D).
    """
    def __init__(self, emb_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.attn_hidden = nn.Linear(emb_dim, hidden)
        self.attn_out = nn.Linear(hidden, 1, bias=False)
        self.proj = nn.Linear(2 * emb_dim, emb_dim)

    def forward(self, H, mask=None):
        if H.dim() == 2:
            H = H.unsqueeze(0)
        scores = self.attn_out(torch.tanh(self.attn_hidden(H))).squeeze(-1)  # (B, W)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=-1)  # (B, W)
        a = alpha.unsqueeze(-1)
        mu = (a * H).sum(dim=1)
        # clamp variance before sqrt (ECAPA ASP numerical form)
        var = (a * H.pow(2)).sum(dim=1) - mu.pow(2)
        sigma = torch.sqrt(var.clamp(min=1e-9))
        pooled = self.proj(torch.cat([mu, sigma], dim=-1))
        return F.normalize(pooled, dim=-1)


class TemporalGRUPool(nn.Module):
    """
    Bidirectional 1-layer GRU over clip embeddings, then Linear(2D -> D).
    Bidirectional (vs unidirectional last-state) treats the set as weakly ordered
    clip tokens without privileging one temporal direction — appropriate for
    mixed-session set pooling.
    Input: (B, W, D) or (W, D). Output: (B, D) L2-normalized.
    """
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.gru = nn.GRU(
            emb_dim, emb_dim, num_layers=1, batch_first=True, bidirectional=True,
        )
        self.proj = nn.Linear(2 * emb_dim, emb_dim)

    def forward(self, H, mask=None):
        if H.dim() == 2:
            H = H.unsqueeze(0)
        # h_n: (2, B, D) — last forward + last backward states
        _, h_n = self.gru(H)
        pooled = self.proj(torch.cat([h_n[0], h_n[1]], dim=-1))
        return F.normalize(pooled, dim=-1)


class TemporalTCNPool(nn.Module):
    """
    Tiny 1-D TCN over the time (clip) axis, then mean-pool + Linear.
    Conv1d(D->D, k=3, pad=1) -> ReLU -> Conv1d(D->D, k=3, dil=2, pad=2) -> ReLU
    -> mean over time -> Linear(D->D) -> L2-norm.
    Input: (B, W, D) or (W, D). Output: (B, D) L2-normalized.
    """
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.conv1 = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            emb_dim, emb_dim, kernel_size=3, padding=2, dilation=2,
        )
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, H, mask=None):
        if H.dim() == 2:
            H = H.unsqueeze(0)
        x = H.transpose(1, 2)  # (B, D, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=-1)  # (B, D)
        return F.normalize(self.proj(x), dim=-1)


class MultiHeadAttentionPool(nn.Module):
    """
    Learnable query attending to clip tokens via multi-head attention
    (Ilse-style set pooling with MHA; cf. Set Transformer / Ilse et al. 2018).
    nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True) with a
    learned (1, D) query over H as K/V -> (B, D) -> L2-norm.
    Input: (B, W, D) or (W, D). Output: (B, D) L2-normalized.
    """
    def __init__(self, emb_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.mha = nn.MultiheadAttention(
            emb_dim, num_heads=num_heads, batch_first=True,
        )

    def forward(self, H, mask=None):
        if H.dim() == 2:
            H = H.unsqueeze(0)
        B = H.size(0)
        q = self.query.expand(B, -1, -1)  # (B, 1, D)
        key_padding_mask = None
        if mask is not None:
            # mask True = valid; MHA key_padding_mask True = ignore
            key_padding_mask = ~mask
        out, _ = self.mha(q, H, H, key_padding_mask=key_padding_mask)
        return F.normalize(out.squeeze(1), dim=-1)


def build_temporal_aggregator(name: str, emb_dim: int = 128):
    """Factory: name in {gru, tcn, mha}."""
    key = str(name).lower().strip()
    if key == "gru":
        return TemporalGRUPool(emb_dim)
    if key == "tcn":
        return TemporalTCNPool(emb_dim)
    if key in ("mha", "attention", "multihead"):
        return MultiHeadAttentionPool(emb_dim, num_heads=4)
    raise ValueError(f"unknown temporal aggregator {name!r}; expected gru|tcn|mha")


def batch_hard_triplet(emb, labels, margin: float = 0.2):
    """Batch-hard triplet loss (Hermans et al. 2017) on normalized embeddings."""
    d = torch.cdist(emb, emb)                                # euclidean on unit sphere
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    pos = d.masked_fill(~same | eye, -1).max(1).values      # hardest positive
    neg = d.masked_fill(same, float("inf")).min(1).values   # hardest negative
    return F.relu(pos - neg + margin).mean()


def build_encoder(in_channels=2, emb_dim=128, width_mult=0.25):
    return Encoder(in_channels, emb_dim, width_mult)
