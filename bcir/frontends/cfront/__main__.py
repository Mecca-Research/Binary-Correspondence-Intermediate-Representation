"""bcir-cfront — the C-frontend driver CLI (a step toward a `cc`-like compiler driver).

Compiles a C file through the plug-in C frontend and prints the artifacts. The file's own
directory is always on the include search path, so a driver with sibling headers
(`#include "regs.h"`) compiles directly from the CLI — no test-harness include map needed.

    python -m bcir.frontends.cfront [options] file.c ...

Options (a cc-compatible subset):
    -I <dir>        add <dir> to the #include search path (repeatable; -I<dir> also works)
    -D name[=val]   predefine an object macro (val defaults to 1; repeatable)
    -U name         undefine a predefined/-D macro (repeatable)
    -std=<std>      language standard: c23/c2x (default), c17, c11  (sets __STDC_VERSION__)
    -E              preprocess only — print the expanded translation unit, then stop
    -o <file>       write output to <file> instead of stdout
    --explain       also print the per-function explain record
    --selfcheck     print the generated self-check harness
"""
from __future__ import annotations

import os
import sys

from .pipeline import compile_unit, emit_selfcheck

_STD_VERSION = {"c23": "202311L", "c2x": "202311L", "c17": "201710L", "c18": "201710L",
                "c11": "201112L", "gnu23": "202311L", "gnu17": "201710L", "gnu11": "201112L"}


def _add_define(defines: dict, spec: str) -> None:
    name, _eq, val = spec.partition("=")
    defines[name] = val if _eq else ""               # "" -> defined as 1 by the preprocessor


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    inc_dirs: list[str] = []
    defines: dict[str, str] = {}
    undefs: list[str] = []
    files: list[str] = []
    std = "c23"
    pp_only = show_explain = selfcheck = False
    out_path: str | None = None

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--explain":
            show_explain = True
        elif a == "--selfcheck":
            selfcheck = True
        elif a == "-E":
            pp_only = True
        elif a == "-o":
            i += 1; out_path = args[i]
        elif a.startswith("-o"):
            out_path = a[2:]
        elif a == "-I":
            i += 1; inc_dirs.append(args[i])
        elif a.startswith("-I"):
            inc_dirs.append(a[2:])
        elif a == "-D":
            i += 1; _add_define(defines, args[i])
        elif a.startswith("-D"):
            _add_define(defines, a[2:])
        elif a == "-U":
            i += 1; undefs.append(args[i])
        elif a.startswith("-U"):
            undefs.append(a[2:])
        elif a.startswith("-std="):
            std = a[5:]
        elif a in ("-h", "--help"):
            sys.stderr.write(__doc__)
            return 0
        elif a.startswith("-"):
            sys.stderr.write(f"bcir-cfront: unknown option {a!r} (see --help)\n")
            return 2
        else:
            files.append(a)
        i += 1

    if not files:
        sys.stderr.write(__doc__)
        return 2
    if std in _STD_VERSION:
        defines.setdefault("__STDC_VERSION__", _STD_VERSION[std])
    for u in undefs:
        defines.pop(u, None)

    out: list[str] = []
    rc = 0
    for path in files:
        src_dir = os.path.dirname(os.path.abspath(path))
        search = [src_dir, *inc_dirs]                # quoted include: the source dir, then -I dirs
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            sys.stderr.write(f"bcir-cfront: cannot open {path!r}: {e}\n")
            rc = 2
            continue

        if pp_only:                                  # -E: just the preprocessed translation unit
            from .cpp import CPPError, preprocess  # noqa: PLC0415
            try:
                out.append(preprocess(text, search_paths=search, defines=defines))
            except CPPError as e:
                sys.stderr.write(f"{path}: preprocessor error: {e}\n")
                rc = 1
            continue

        try:
            r = compile_unit(text, search_paths=search, defines=defines, check_clang=False)
        except Exception as e:  # noqa: BLE001 -- the CLI surfaces any frontend error as a diagnostic
            sys.stderr.write(f"{path}: error: {e}\n")
            rc = 1
            continue

        if selfcheck:
            out.append(emit_selfcheck(r))
            continue
        out.append(f"=== {path} ===")
        out.append(f"functions: {list(r.lowered.functions)}")
        for name, lf in r.lowered.functions.items():
            out.append(f"\n-- {name}: {len(lf.claims)} claims --")
            out.append(r.emitted[name])
            if show_explain:
                out.append(r.explain[name])
        status = "CLEAN" if r.is_clean else "DIRTY"
        out.append(f"\nR1-R18: {status} (r18_ok={r.r18_ok})  |  Clang behaviour: {r.equivalence}")
        if not r.is_clean:
            for d in r.diagnostics:
                out.append(f"  {d.law}: {d.message}")
            rc = 1

    text_out = "\n".join(out) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_out)
    else:
        sys.stdout.write(text_out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
