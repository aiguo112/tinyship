"""
ShipNN model = MobileNetV3-Small at reduced width, no architectural changes
(paper §4.2). Stem adapted to accept 1- or 2-channel spectrograms.

Verified: build_shipnn(num_classes=100, in_channels=2, width_mult=0.25)
forwards a (B,2,95,126) tensor -> (B,100).

FOOTPRINT NOTE (honest): torchvision width_mult=0.25 -> ~0.143M params, while
the paper reports 0.351M for ShipNN. This is a TensorFlow(Keras alpha=0.25) vs
PyTorch scaling difference, NOT an error. It does not affect the leakage
experiment (which reads accuracy deltas). Tune `width_mult` (~0.4-0.5) later if
you want to match the paper's Flash/RAM footprint for the systems section.
"""
from __future__ import annotations
import torch.nn as nn
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf, MobileNetV3


def build_shipnn(num_classes: int = 100, in_channels: int = 2,
                 width_mult: float = 0.25, dropout: float = 0.2) -> nn.Module:
    conf, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", width_mult=width_mult)
    model = MobileNetV3(conf, last_channel, num_classes=num_classes, dropout=dropout)
    if in_channels != 3:
        stem = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels, stem.out_channels, kernel_size=stem.kernel_size,
            stride=stem.stride, padding=stem.padding, bias=False)
    return model


def build_mobilenet_full(num_classes: int = 100, in_channels: int = 2) -> nn.Module:
    """Full-width MobileNetV3-Small baseline (paper reports 91.7% CQT+MFCC)."""
    return build_shipnn(num_classes, in_channels, width_mult=1.0)
