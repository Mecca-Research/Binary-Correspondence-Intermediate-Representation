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
    --target <abi>  the target data model the unit is laid out for (default x86_64-linux); one of
                    x86_64-linux, aarch64-linux, riscv64-linux, x86_64-windows, i386-linux
    -fsyntax-only   parse + check only; print Clang-style diagnostics, emit no compiled output
    --emit-json     print the diagnostics as a machine-readable JSON array, then stop
    --fallback      graceful degradation: a construct outside the supported subset reports a
                    fallback-to-LLVM signal (exit 2) instead of a hard error
    --r21 <policy>  how a detected use-after-free / double-free (R21, §5.12) gates the compile:
                    advisory (default; surfaced, never gates), fallback (route to LLVM, exit 2),
                    or reject (hard verify error, exit 1)
    -o <file>       write output to <file> instead of stdout
    --explain       also print the per-function explain record
    --selfcheck     print the generated self-check harness
"""
from __future__ import annotations

import os
import sys

from .abi import TARGETS
from .pipeline import compile_unit, compile_with_fallback, diagnose, emit_selfcheck

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
    pp_only = show_explain = selfcheck = syntax_only = emit_json = fallback = False
    r21_policy = "advisory"
    target: str | None = None
    out_path: str | None = None

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--explain":
            show_explain = True
        elif a == "--selfcheck":
            selfcheck = True
        elif a == "-fsyntax-only":
            syntax_only = True
        elif a == "--emit-json":
            emit_json = True
        elif a == "--fallback":
            fallback = True
        elif a == "--r21":
            i += 1; r21_policy = args[i]
        elif a.startswith("--r21="):
            r21_policy = a[len("--r21="):]
        elif a == "--target":
            i += 1; target = args[i]
        elif a.startswith("--target="):
            target = a[len("--target="):]
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
    if target is not None and target not in TARGETS:
        sys.stderr.write(f"bcir-cfront: unknown --target {target!r}; choose from "
                         f"{', '.join(sorted(TARGETS))}\n")
        return 2
    if r21_policy not in ("advisory", "fallback", "reject"):
        sys.stderr.write(f"bcir-cfront: unknown --r21 policy {r21_policy!r} "
                         f"(advisory|fallback|reject)\n")
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
                out.append(preprocess(text, search_paths=search, defines=defines, name=path))
            except CPPError as e:
                sys.stderr.write(f"{path}: preprocessor error: {e}\n")
                rc = 1
            continue

        if syntax_only or emit_json:                 # check only: Clang-style or JSON diagnostics
            rep = diagnose(text, search_paths=search, defines=defines, filename=path)
            if emit_json:
                out.append(rep.to_json())
            elif rep.diagnostics:
                sys.stderr.write(rep.render() + "\n")
            if not rep.ok:
                rc = 1
            continue

        try:
            if fallback:                             # the LLVM-backend fallback contract (never raises)
                r = compile_with_fallback(text, search_paths=search, defines=defines,
                                          check_clang=False, filename=path, target=target)
                if r.needs_fallback:
                    sys.stderr.write(f"{path}: fallback to LLVM backend: {r.fallback}\n")
                    rc = 2
                    continue
            else:
                r = compile_unit(text, search_paths=search, defines=defines, check_clang=False,
                                 filename=path, target=target)
        except Exception as e:  # noqa: BLE001 -- render the front-end error with a Clang-style caret
            rep = diagnose(text, search_paths=search, defines=defines, filename=path)
            sys.stderr.write((rep.render() + "\n") if rep.diagnostics else f"{path}: error: {e}\n")
            rc = 1
            continue

        # R21 lifetime policy (§5.12): a detected use-after-free / double-free routes the unit to the
        # LLVM backend (fallback, rc 2) or hard-rejects it (rc 1) under a non-advisory policy. The
        # detection is the advisory `lifetime_diagnostics`; only the verdict changes. Parity: the C
        # twin driver runtime/c/bcir_cc.c applies the identical policy + exit codes.
        if r21_policy != "advisory" and r.lifetime_diagnostics:
            d0 = r.lifetime_diagnostics[0]
            if r21_policy == "fallback":
                sys.stderr.write(f"{path}: fallback to LLVM backend: lifetime: {d0.law} {d0.message}\n")
                rc = 2
            else:                                        # reject
                sys.stderr.write(f"{path}: lifetime error: {d0.law} {d0.message}\n")
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
        out.append(f"\nR1-R18: {status} (r18_ok={r.r18_ok})  |  target: {r.target}  |  "
                   f"Clang behaviour: {r.equivalence}")
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
