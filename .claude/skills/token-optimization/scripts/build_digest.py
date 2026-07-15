#!/usr/bin/env python3
"""Regenerate the compressed repo digest (.claude/context/BCIR_DIGEST.md) skeleton.

Token-optimization scaffolding: an agent reads ONE small digest instead of
re-exploring the tree. This script rebuilds the *mechanical* half (inventory,
doc heading maps, generated counts) and preserves the *curated* half (the
subsystem knowledge block between the KNOWLEDGE markers), which agents update
by hand when they learn something durable.

    python .claude/skills/token-optimization/scripts/build_digest.py          # rewrite
    python .claude/skills/token-optimization/scripts/build_digest.py --check  # drift gate

Stdlib-only, deterministic. Inspired by: reversible compression (originals stay
on disk; the digest is the retrieval key), hierarchical modular encoding (one
module = one summary line), and docs/STATUS.md's generated-count discipline.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
# When installed at .claude/skills/token-optimization/scripts/, ROOT is the repo root.
if not os.path.isdir(os.path.join(ROOT, "bcir")):
    ROOT = os.getcwd()
DIGEST = os.path.join(ROOT, ".claude", "context", "BCIR_DIGEST.md")
K_BEGIN, K_END = "<!-- KNOWLEDGE:BEGIN -->", "<!-- KNOWLEDGE:END -->"


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def doc_map() -> str:
    out = []
    docs_root = os.path.join(ROOT, "docs")
    docs = []
    for dirpath, dirnames, filenames in os.walk(docs_root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for filename in filenames:
            if filename.endswith(".md"):
                docs.append(os.path.join(dirpath, filename))
    for p in sorted(docs):
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        lines = open(p, encoding="utf-8").read().splitlines()
        heads = [l.lstrip("# ").strip() for l in lines if re.match(r"^##? ", l)][:14]
        out.append(f"- **{rel}** ({len(lines)}L): " + " · ".join(h[:60] for h in heads[1:] or heads))
    return "\n".join(out)


def mechanical() -> str:
    status = ""
    sp = os.path.join(ROOT, "docs", "STATUS.md")
    if os.path.exists(sp):
        status = "\n".join(l for l in open(sp, encoding="utf-8").read().splitlines() if l.startswith("|"))[:1200]
    tree = sh("git ls-tree -d --name-only HEAD | grep -v '^\\.' | sed 's#^#./#' | sort | tr '\\n' ' '")
    return (
        "## Generated inventory (do not edit — rebuild with build_digest.py)\n\n"
        f"Top-level: {tree}\n\n"
        f"### STATUS.md counts (generated source of truth)\n{status}\n\n"
        f"### docs/ heading map\n{doc_map()}\n"
    )


def main() -> int:
    check = "--check" in sys.argv
    cur = open(DIGEST, encoding="utf-8").read() if os.path.exists(DIGEST) else ""
    if K_BEGIN in cur and K_END in cur:
        knowledge = cur[cur.index(K_BEGIN): cur.index(K_END) + len(K_END)]
    else:
        knowledge = f"{K_BEGIN}\n(curated subsystem knowledge goes here)\n{K_END}"
    new = (
        "# BCIR repo digest (compressed knowledge base — read this INSTEAD of exploring)\n\n"
        + knowledge + "\n\n" + mechanical()
    )
    if check:
        ok = cur == new
        print("digest is current." if ok else "digest is STALE — rerun build_digest.py")
        return 0 if ok else 1
    os.makedirs(os.path.dirname(DIGEST), exist_ok=True)
    open(DIGEST, "w", encoding="utf-8").write(new)
    print(f"wrote {os.path.relpath(DIGEST, ROOT)} ({len(new)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
