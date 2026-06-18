#!/usr/bin/env python3
"""Fail CI on broken relative Markdown links. Code-aware (skips fenced + inline code) so C++/IR
snippets that look like `foo[i](Type t)` are never mistaken for links.

    python tools/docs/check_links.py            # scan the whole tree
    python tools/docs/check_links.py path/...    # scan specific files/dirs
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SKIP_DIRS = (".git", "__pycache__", "node_modules", ".mypy_cache")
_LINK = re.compile(r"\[[^\]]*\]\(\s*<?([^)>\s]+)>?\s*(?:\"[^\"]*\")?\)")


def _strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)   # fenced blocks
    text = re.sub(r"~~~.*?~~~", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", "", text)                    # inline code
    return text


def _md_files(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        ap = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if os.path.isfile(ap) and ap.endswith(".md"):
            out.append(ap)
        elif os.path.isdir(ap):
            for dp, dn, fn in os.walk(ap):
                dn[:] = [d for d in dn if d not in _SKIP_DIRS]
                out += [os.path.join(dp, f) for f in fn if f.endswith(".md")]
    return sorted(set(out))


def broken_links(paths: list[str]) -> list[tuple[str, str]]:
    bad: list[tuple[str, str]] = []
    for md in _md_files(paths):
        d = os.path.dirname(md)
        try:
            text = _strip_code(open(md, encoding="utf-8").read())
        except OSError:
            continue
        for m in _LINK.finditer(text):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "tel:", "#")):
                continue
            path = target.split("#", 1)[0].split("?", 1)[0].strip()
            if not path:
                continue
            if not os.path.exists(os.path.normpath(os.path.join(d, path))):
                bad.append((os.path.relpath(md, ROOT), target))
    return bad


def main() -> int:
    paths = sys.argv[1:] or ["."]
    bad = broken_links(paths)
    if bad:
        sys.stderr.write(f"[check_links] {len(bad)} broken relative link(s):\n")
        for f, t in bad:
            sys.stderr.write(f"  {f} -> {t}\n")
        return 1
    sys.stderr.write("[check_links] no broken relative links.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
