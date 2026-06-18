"""CLI demo for the C frontend: print the six artifacts for a C file (or the bundled fixtures).

    python -m bcir.frontends.cfront bcir/frontends/cfront/fixtures/L2_struct.c
    python -m bcir.frontends.cfront --explain <file.c>
"""
from __future__ import annotations

import sys

from .pipeline import compile_unit


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    show_explain = "--explain" in args
    files = [a for a in args if not a.startswith("-")]
    if not files:
        sys.stderr.write(__doc__)
        return 2
    rc = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            r = compile_unit(f.read())
        print(f"\n=== {path} ===")
        print(f"functions: {list(r.lowered.functions)}")
        for name, lf in r.lowered.functions.items():
            print(f"\n-- {name}: {len(lf.claims)} claims, plan cost "
                  f"{r.plans[name].score if hasattr(r.plans[name], 'score') else '?'} --")
            print(r.emitted[name])
            if show_explain:
                print(r.explain[name])
        status = "CLEAN" if r.is_clean else "DIRTY"
        print(f"\nR1–R18: {status} (r18_ok={r.r18_ok})  |  Clang behaviour: {r.equivalence}")
        if not r.is_clean:
            for d in r.diagnostics:
                print(f"  {d.law}: {d.message}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
