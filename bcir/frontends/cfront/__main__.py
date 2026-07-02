"""bcir-cfront — the C-frontend driver CLI (a step toward a `cc`-like compiler driver).

Compiles a C file — or a multi-file PROJECT — through the plug-in C frontend and prints the
artifacts. The file's own directory is always on the include search path, so a driver with sibling
headers (`#include "regs.h"`) compiles directly from the CLI — no test-harness include map needed.

    python -m bcir.frontends.cfront [options] file.c ...

Options (a cc-compatible subset):
    -I <dir>        add <dir> to the #include search path (repeatable; -I<dir> also works)
    -D name[=val]   predefine an object macro (val defaults to 1; repeatable)
    -U name         undefine a predefined/-D macro (repeatable)
    -std=<std>      language standard: c23/c2x (default), c17, c11  (sets __STDC_VERSION__)
    -E              preprocess only — print the expanded translation unit, then stop
    -M / -MM        dependency output only: print a make rule `unit.o: unit.c hdr...` per file
                    (the two spellings agree here — <system> headers are modeled intrinsically,
                    so there are no system-header dependencies to omit)
    -MF <file>      write the dependency rules to <file> instead of stdout (implies -M)
    -MT <target>    override the rule target (default: the unit basename with `.o`)
    -p <path>       compile-database mode: read `compile_commands.json` (a directory containing
                    it, or the file itself) and compile every entry with ITS OWN -I/-D/-U/-std/
                    --target flags (Phase 3 project orchestration)
    --project       print the per-PROJECT verdict line over the compiled file set — CLEAN /
                    PARTIAL-FALLBACK (some units routed to LLVM under --fallback) / DIRTY —
                    printed automatically for a multi-file or compile-database invocation
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
    --emit-link-flags  print just the linker flags this unit's external-call edges need (B1), one
                    space-separated line (e.g. `-lm`); empty for a pure-integer unit. For build-system
                    consumption. Mirrors bcir-cc --emit-link-flags byte-for-byte (dual-rail parity).
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


def _parse_entry_flags(arguments: list) -> tuple[list, dict, list, str | None, str | None]:
    """The cc-compatible subset OF A COMPILE-DATABASE ENTRY's argv: -I/-D/-U/-std/--target.
    Everything else in the entry (the compiler name, -c, -o, warning flags, ...) is ignored —
    the database records how ANOTHER compiler was invoked; this driver takes only what changes
    the translation unit's meaning."""
    inc: list = []
    defs: dict = {}
    undefs: list = []
    std: str | None = None
    target: str | None = None
    i = 0
    while i < len(arguments):
        a = str(arguments[i])
        if a == "-I":
            i += 1; inc.append(str(arguments[i]))
        elif a.startswith("-I"):
            inc.append(a[2:])
        elif a == "-D":
            i += 1; _add_define(defs, str(arguments[i]))
        elif a.startswith("-D"):
            _add_define(defs, a[2:])
        elif a == "-U":
            i += 1; undefs.append(str(arguments[i]))
        elif a.startswith("-U"):
            undefs.append(a[2:])
        elif a.startswith("-std="):
            std = a[5:]
        elif a == "--target":
            i += 1; target = str(arguments[i])
        elif a.startswith("--target="):
            target = a[len("--target="):]
        i += 1
    return inc, defs, undefs, std, target


def _load_compile_db(path: str) -> list[dict]:
    """`compile_commands.json` entries as job dicts: {path, inc_dirs, defines, undefs, std, target}.
    `path` may be the JSON file or a directory containing it (the clang -p convention)."""
    import json  # noqa: PLC0415
    import shlex  # noqa: PLC0415
    db = os.path.join(path, "compile_commands.json") if os.path.isdir(path) else path
    with open(db, encoding="utf-8") as f:
        entries = json.load(f)
    jobs: list[dict] = []
    for e in entries:
        args = e.get("arguments")
        if args is None:
            args = shlex.split(e.get("command", ""))
        inc, defs, undefs, std, target = _parse_entry_flags(list(args))
        directory = e.get("directory", ".")
        fpath = e["file"]
        if not os.path.isabs(fpath):
            fpath = os.path.join(directory, fpath)
        # entry -I dirs resolve relative to the entry's directory, per the compile-database spec.
        inc = [d if os.path.isabs(d) else os.path.join(directory, d) for d in inc]
        jobs.append({"path": fpath, "inc_dirs": inc, "defines": defs, "undefs": undefs,
                     "std": std, "target": target})
    return jobs


def _dep_rule(path: str, deps: list[str], mt: str | None) -> str:
    tgt = mt or (os.path.splitext(os.path.basename(path))[0] + ".o")
    return f"{tgt}: {' '.join([path, *deps])}"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    inc_dirs: list[str] = []
    defines: dict[str, str] = {}
    undefs: list[str] = []
    files: list[str] = []
    std = "c23"
    pp_only = show_explain = selfcheck = syntax_only = emit_json = fallback = False
    emit_link_flags = dep_only = project_verdict = False
    dep_file: str | None = None
    dep_target: str | None = None
    compdb: str | None = None
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
        elif a == "--emit-link-flags":
            emit_link_flags = True
        elif a == "--project":
            project_verdict = True
        elif a == "-fsyntax-only":
            syntax_only = True
        elif a == "--emit-json":
            emit_json = True
        elif a == "--fallback":
            fallback = True
        elif a in ("-M", "-MM"):
            dep_only = True
        elif a == "-MF":
            i += 1; dep_file = args[i]; dep_only = True
        elif a == "-MT":
            i += 1; dep_target = args[i]
        elif a == "-p":
            i += 1; compdb = args[i]
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

    # The job list (Phase 3 project orchestration): plain CLI files carry the global flags;
    # compile-database entries carry their own. Both may be combined in one invocation.
    jobs: list[dict] = [{"path": p, "inc_dirs": [], "defines": {}, "undefs": [],
                         "std": None, "target": None} for p in files]
    if compdb is not None:
        try:
            jobs += _load_compile_db(compdb)
        except (OSError, ValueError, KeyError) as e:
            sys.stderr.write(f"bcir-cfront: cannot load compile database {compdb!r}: {e}\n")
            return 2
        project_verdict = True

    if not jobs:
        sys.stderr.write(__doc__)
        return 2
    if len(jobs) > 1:
        project_verdict = True
    if target is not None and target not in TARGETS:
        sys.stderr.write(f"bcir-cfront: unknown --target {target!r}; choose from "
                         f"{', '.join(sorted(TARGETS))}\n")
        return 2
    if r21_policy not in ("advisory", "fallback", "reject"):
        sys.stderr.write(f"bcir-cfront: unknown --r21 policy {r21_policy!r} "
                         f"(advisory|fallback|reject)\n")
        return 2
    for u in undefs:
        defines.pop(u, None)

    out: list[str] = []
    dep_rules: list[str] = []
    outcomes: list[str] = []                         # per job: clean | fallback | dirty
    rc = 0
    for job in jobs:
        path = job["path"]
        src_dir = os.path.dirname(os.path.abspath(path))
        search = [src_dir, *inc_dirs, *job["inc_dirs"]]  # quoted include: source dir, -I, entry -I
        job_defines = dict(defines)
        job_defines.update(job["defines"])
        job_std = job["std"] or std
        if job_std in _STD_VERSION:
            job_defines.setdefault("__STDC_VERSION__", _STD_VERSION[job_std])
        for u in job["undefs"]:
            job_defines.pop(u, None)
        job_target = job["target"] or target
        if job_target is not None and job_target not in TARGETS:
            sys.stderr.write(f"{path}: unknown --target {job_target!r} (compile-database entry)\n")
            outcomes.append("dirty")
            rc = 1
            continue
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            sys.stderr.write(f"bcir-cfront: cannot open {path!r}: {e}\n")
            outcomes.append("dirty")
            rc = 2 if rc != 1 else rc
            continue

        if dep_only:                                 # -M/-MM/-MF: the make dependency rule per unit
            from .cpp import CPPError, Preprocessor  # noqa: PLC0415
            try:
                pp = Preprocessor(None, None, search, job_defines)
                pp.process(text, path)
                dep_rules.append(_dep_rule(path, pp.dep_paths, dep_target))
                outcomes.append("clean")
            except CPPError as e:
                sys.stderr.write(f"{path}: preprocessor error: {e}\n")
                outcomes.append("dirty")
                rc = 1
            continue

        if pp_only:                                  # -E: just the preprocessed translation unit
            from .cpp import CPPError, preprocess  # noqa: PLC0415
            try:
                out.append(preprocess(text, search_paths=search, defines=job_defines, name=path))
                outcomes.append("clean")
            except CPPError as e:
                sys.stderr.write(f"{path}: preprocessor error: {e}\n")
                outcomes.append("dirty")
                rc = 1
            continue

        if syntax_only or emit_json:                 # check only: Clang-style or JSON diagnostics
            rep = diagnose(text, search_paths=search, defines=job_defines, filename=path)
            if emit_json:
                out.append(rep.to_json())
            elif rep.diagnostics:
                sys.stderr.write(rep.render() + "\n")
            if not rep.ok:
                outcomes.append("dirty")
                rc = 1
            else:
                outcomes.append("clean")
            continue

        try:
            if fallback:                             # the LLVM-backend fallback contract (never raises)
                r = compile_with_fallback(text, search_paths=search, defines=job_defines,
                                          check_clang=False, filename=path, target=job_target)
                if r.needs_fallback:
                    sys.stderr.write(f"{path}: fallback to LLVM backend: {r.fallback}\n")
                    outcomes.append("fallback")
                    rc = 2 if rc != 1 else rc
                    continue
            else:
                r = compile_unit(text, search_paths=search, defines=job_defines, check_clang=False,
                                 filename=path, target=job_target)
        except Exception as e:  # noqa: BLE001 -- render the front-end error with a Clang-style caret
            rep = diagnose(text, search_paths=search, defines=job_defines, filename=path)
            sys.stderr.write((rep.render() + "\n") if rep.diagnostics else f"{path}: error: {e}\n")
            outcomes.append("dirty")
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
                outcomes.append("fallback")
                rc = 2 if rc != 1 else rc
            else:                                        # reject
                sys.stderr.write(f"{path}: lifetime error: {d0.law} {d0.message}\n")
                outcomes.append("dirty")
                rc = 1
            continue

        if emit_link_flags:                          # B1: just the derived linker flags, one line
            from .linkflags import format_link_flags  # noqa: PLC0415
            out.append(format_link_flags(r.link_flags))
            outcomes.append("clean" if r.is_clean else "dirty")
            continue
        if selfcheck:
            out.append(emit_selfcheck(r))
            outcomes.append("clean" if r.is_clean else "dirty")
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
            outcomes.append("dirty")
            rc = 1
        else:
            outcomes.append("clean")

    if dep_rules:                                    # -M/-MM output (its own artifact stream)
        if dep_file:
            with open(dep_file, "w", encoding="utf-8") as f:
                f.write("\n".join(dep_rules) + "\n")
        else:
            out.extend(dep_rules)

    # The per-PROJECT verdict (Phase 3): one line over the whole compiled file set. CLEAN = every
    # unit compiled clean; PARTIAL-FALLBACK = no failure but >=1 unit routed to LLVM (--fallback /
    # --r21=fallback); DIRTY = >=1 unit failed. The exit code keeps the per-unit rules (1 dominates 2).
    if project_verdict and outcomes:
        n = len(outcomes)
        dirty = outcomes.count("dirty")
        fell = outcomes.count("fallback")
        if dirty:
            verdict = f"DIRTY ({dirty}/{n} failed" + (f", {fell} fell back" if fell else "") + ")"
        elif fell:
            verdict = f"PARTIAL-FALLBACK ({fell}/{n} routed to the LLVM backend)"
        else:
            verdict = f"CLEAN ({n} file{'s' if n != 1 else ''})"
        out.append(f"project: {verdict}")

    text_out = "\n".join(out) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_out)
    else:
        sys.stdout.write(text_out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
