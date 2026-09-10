#!/usr/bin/env python
"""
Remove Cursor Agent from git history (GitHub Contributors graph).

GitHub lists anyone named in Co-authored-by trailers, even if they are not
the commit Author. This repo had:

  Co-authored-by: Cursor <cursoragent@cursor.com>

This script rewrites the current branch with git commit-tree (bypasses hooks
so Cursor cannot re-inject the trailer). Author/committer stay yours unless
they were Cursor, in which case they are remapped to --name/--email.

Usage (repo root):
  python scripts/remove_cursor_contributor.py
  git push --force origin main

WARNING: this rewrites commit SHAs. You must force-push afterwards.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

CURSOR_MARKERS = ("cursoragent", "cursor agent", "cursor@cursor", "<cursor>")

TRAILER_RE = re.compile(
    r"^(Co-authored-by|Signed-off-by|Reviewed-by|Acked-by)\s*:.*$",
    re.IGNORECASE,
)


def run(args: list[str], *, env: dict | None = None, input_text: str | None = None) -> str:
    r = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        input=input_text,
    )
    return r.stdout


def is_cursor_identity(name: str, email: str) -> bool:
    blob = f"{name} {email}".lower()
    return any(m in blob for m in CURSOR_MARKERS) or "cursoragent@" in email.lower()


def is_cursor_trailer(line: str) -> bool:
    if not TRAILER_RE.match(line.strip()):
        return False
    low = line.lower()
    return any(m in low for m in CURSOR_MARKERS) or "cursoragent@" in low or "cursor <cursor" in low


def strip_cursor_from_message(msg: str) -> tuple[str, bool]:
    """Return (message, changed_because_cursor_was_removed)."""
    nl = "\r\n" if "\r\n" in msg else "\n"
    lines = msg.splitlines()
    kept = [line for line in lines if not is_cursor_trailer(line)]
    removed = len(kept) != len(lines)
    if not removed:
        return msg, False
    text = nl.join(kept)
    if msg.endswith(("\n", "\r\n")) and not text.endswith(("\n", "\r\n")):
        text += nl
    return text, True


def field(rev: str, fmt: str) -> str:
    return run(["git", "log", "-1", f"--format={fmt}", rev]).rstrip("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="", help="Replacement author name (default: most common non-Cursor author)")
    ap.add_argument("--email", default="", help="Replacement author email")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(run(["git", "rev-parse", "--show-toplevel"]).strip())
    os.chdir(root)

    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    if branch == "HEAD":
        print("Detached HEAD; check out a branch first.", file=sys.stderr)
        return 1

    hashes = [h for h in run(["git", "rev-list", "--reverse", "HEAD"]).splitlines() if h]
    if not hashes:
        print("No commits.", file=sys.stderr)
        return 1

    name = args.name
    email = args.email
    if not name or not email:
        authors = run(["git", "log", "--format=%an<%ae>"]).splitlines()
        counts: dict[str, int] = {}
        for a in authors:
            if is_cursor_identity(a, a):
                continue
            counts[a] = counts.get(a, 0) + 1
        if not counts:
            print("Could not infer your git identity; pass --name and --email.", file=sys.stderr)
            return 1
        best = max(counts, key=counts.get)
        inferred_name, inferred_email = best.split("<", 1)
        inferred_name = inferred_name.strip()
        inferred_email = inferred_email.rstrip(">").strip()
        name = name or inferred_name
        email = email or inferred_email

    print(f"Branch: {branch}")
    print(f"Rewrite identity: {name} <{email}>")
    print(f"Commits to scan: {len(hashes)}")

    dirty = run(["git", "status", "--porcelain"]).strip()
    if dirty and not args.dry_run:
        print("Note: working tree has local changes; they will be kept (--soft reset).")

    mapping: dict[str, str] = {}
    parent: str | None = None
    changed = 0

    for old in hashes:
        msg = field(old, "%B")
        new_msg, msg_changed = strip_cursor_from_message(msg)
        an, ae, ad = field(old, "%an"), field(old, "%ae"), field(old, "%ad")
        cn, ce, cd = field(old, "%cn"), field(old, "%ce"), field(old, "%cd")
        tree = field(old, "%T")

        if is_cursor_identity(an, ae):
            an, ae = name, email
        if is_cursor_identity(cn, ce):
            cn, ce = name, email

        msg_changed = new_msg != msg
        ident_changed = (an, ae) != (field(old, "%an"), field(old, "%ae")) or (
            cn,
            ce,
        ) != (field(old, "%cn"), field(old, "%ce"))
        # parent may already have been rewritten
        old_parents = run(["git", "rev-list", "--parents", "-n", "1", old]).split()
        old_parent_list = old_parents[1:]
        new_parents = [mapping.get(p, p) for p in old_parent_list]
        parent_changed = new_parents != old_parent_list

        if not (msg_changed or ident_changed or parent_changed):
            mapping[old] = old
            parent = old
            continue

        changed += 1
        if args.dry_run:
            print(f"  would rewrite {old[:10]}  {field(old, '%s')}")
            if msg_changed:
                print("    strip Cursor trailer from message")
            mapping[old] = old
            parent = old
            continue

        env = os.environ.copy()
        env.update(
            {
                "GIT_AUTHOR_NAME": an,
                "GIT_AUTHOR_EMAIL": ae,
                "GIT_AUTHOR_DATE": ad,
                "GIT_COMMITTER_NAME": cn,
                "GIT_COMMITTER_EMAIL": ce,
                "GIT_COMMITTER_DATE": cd,
            }
        )
        cmd = ["git", "commit-tree", tree]
        for p in new_parents:
            cmd.extend(["-p", p])
        new_sha = run(cmd, env=env, input_text=new_msg).strip()
        mapping[old] = new_sha
        parent = new_sha
        print(f"  {old[:10]} -> {new_sha[:10]}  {field(old, '%s')}")

    if args.dry_run:
        print(f"Dry run: {changed} commit(s) would be rewritten. Re-run without --dry-run.")
        return 0

    if changed == 0:
        print("Nothing to rewrite. No Cursor trailers or Cursor authors found.")
        return 0

    new_head = mapping[hashes[-1]]
    # --soft keeps unstaged local files (same tree; only metadata changed).
    run(["git", "reset", "--soft", new_head])

    print()
    print(f"Rewrote {changed} commit(s). New HEAD: {new_head[:10]}")
    print("GitHub Contributors will drop Cursor Agent after you replace the remote history:")
    print(f"  git push --force origin {branch}")
    print("The graph can take a few minutes to refresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
