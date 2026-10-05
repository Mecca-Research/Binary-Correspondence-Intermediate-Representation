#!/usr/bin/env python3
"""Write the C rails' build as a BCIRfile (BUILD-6, docs/BCIR_BUILD_ROADMAP.md §8).

The manifest is the first BCIRfile's input: every library, tool, harness, seam library and seam test,
every variant this host can build and every kernel of runtime/manifest.json becomes a target --
named as the manifest names it -- over targets that compile each unit to an object, archive the
libraries, write the mutants (tools/build/mutate.py) and emit the kernels (tools/build/emit_kernel.py).
The compilers, the archiver and the interpreter are declared by path and by the sha256 of their
bytes (R1: registry-first), so the plan names what built it.

    bcirfile.py --cc /usr/bin/gcc --cxx /usr/bin/g++ [-o build/bcir-make/BCIRfile]

Compile lines follow the CMake build's policy (cmake/BCIRCompilerFlags.cmake): C23 as the compiler
spells it (c23, else c2x), -O2 with the asserts kept, -Wall -Wextra -Werror (not on a mutant, which
CMake builds without the warning policy), the manifest's per-unit options (`gcc:`/`clang:` scoped),
and the seam as C++17 -Wpedantic. Each object reads its source and every header that source reaches
through a quoted #include, so an edit to a header makes exactly its includers stale. The fuzzers are
not planned: they are Clang-only libFuzzer builds (the `fuzzer` preset). A sanitizer variant is
planned only where tools/build/sanitizer.py finds its runtime working, as CMake decides it.

Exit 0 with the BCIRfile written, 2 when it cannot be written (no such compiler, an unreadable
manifest): every exit is a verdict (docs/security/laws.md L1).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C_DIR = "runtime/c"
CPP_DIR = "runtime/cpp"
OUT = "build/bcir-make"
# What a section task reads beyond its script and its binaries: the trees a section script reaches
# (the oracle it runs, the fixtures and sources it compiles, the gate's own tools, the channel
# examples), claimed whole -- an edit anywhere in them re-runs the sections, never too few.
SECTION_TREES = ("bcir", "channels", "runtime/c", "tools")
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"bcir_build_{name}", ROOT / "tools" / "build" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tool_path(value: str) -> str | None:
    """A tool's path as a BCIRfile declares it: absolute, where PATH finds ``value``, its symlinks
    kept. A multi-call driver picks its mode from the name it is run by -- `clang++` is a link to
    `clang`, and run by the resolved name it links C, not C++ (found by the first BCIR Make build
    with Clang, whose C++ seam did not link) -- and the identity hashes the bytes the name
    reaches either way. None when PATH finds nothing."""
    found = shutil.which(value)
    return os.path.abspath(found) if found else None


def path_compilers() -> dict[str, str]:
    """The gcc and clang on PATH: what a compiler-only section runs besides CC."""
    found = {}
    for name in ("gcc", "clang"):
        path = tool_path(name)
        if path:
            found[name] = path
    return found


def identity(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def compiler_family(cc: str) -> str:
    """`clang` or `gcc`, from what the compiler says it is (the manifest's option scopes)."""
    try:
        out = subprocess.run([cc, "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    return "clang" if "clang" in out.lower() else "gcc"


def c23_spelling(cc: str) -> str:
    """`c23` when the compiler knows it, else `c2x` (GCC 13), as CMake maps C_STANDARD 23."""
    try:
        ok = subprocess.run(
            [cc, "-std=c23", "-x", "c", "-fsyntax-only", "-"],
            input=b"int main(void){return 0;}\n",
            capture_output=True,
            timeout=60,
        ).returncode
    except (OSError, subprocess.TimeoutExpired):
        ok = 1
    return "c23" if ok == 0 else "c2x"


def header_closure(source: str, include_dirs: tuple[str, ...]) -> list[str]:
    """Every header a source reaches through quoted includes, resolved as the compiler does: the
    including file's directory first, then the include path; repo paths, sorted."""
    seen: set[str] = set()
    todo = [source]
    while todo:
        current = todo.pop()
        try:
            text = (ROOT / current).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # a repo path on every host: the host's own spelling of the directory gave a Windows runner
        # `runtime\c/bcir_runtime.h`, which no BCIRfile can name
        here = posixpath.dirname(current) or "."
        for name in INCLUDE.findall(text):
            for base in (here, *include_dirs):
                candidate = f"{base}/{name}"
                if (ROOT / candidate).is_file():
                    if candidate not in seen:
                        seen.add(candidate)
                        todo.append(candidate)
                    break
    return sorted(seen)


def scoped(options: list[str], family: str) -> list[str]:
    out = []
    for opt in options:
        if opt.startswith("gcc:"):
            if family == "gcc":
                out.append(opt[4:])
        elif opt.startswith("clang:"):
            if family == "clang":
                out.append(opt[6:])
        else:
            out.append(opt)
    return out


class Writer:
    def __init__(self) -> None:
        self.lines: list[str] = ["bcirfile 1"]
        self.names: set[str] = set()

    def comment(self, text: str) -> None:
        self.lines.append(f"# {text}")

    def tool(self, name: str, path: str) -> None:
        self.lines.append(f"tool {name} {path} {identity(path)}")

    def target(
        self,
        name: str,
        reads: list[str],
        writes: list[str],
        runs: list[list[str]],
        trees=(),
        uses=(),
    ) -> None:
        assert name not in self.names, name
        self.names.add(name)
        self.lines.append(f"target {name}")
        if reads:
            self.lines.append("  reads " + " ".join(dict.fromkeys(reads)))
        for tree in trees:
            self.lines.append(f"  reads-tree {tree}")
        if uses:
            self.lines.append("  uses " + " ".join(uses))
        self.lines.append("  writes " + " ".join(writes))
        for argv in runs:
            self.lines.append("  run " + " ".join(argv))

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def generate(
    cc: str,
    cxx: str | None,
    ar: str,
    python: str,
    tsan: bool,
    *,
    out: str = OUT,
    bash: str | None = None,
    path_compilers: dict[str, str] | None = None,
) -> tuple[str, dict]:
    """The BCIRfile text, and what it planned: {kind: [manifest names]}. Its outputs go under
    ``out``; with ``bash``, every manifest section whose binaries it plans is a task too: the
    section's script run over those binaries by tools/build/run_section.sh, its verdict the
    task's outputs (BUILD-7). ``path_compilers`` ({"gcc": path, "clang": path}, each where PATH has
    one) are the compilers a compiler-only section runs besides CC: declared by identity and used
    by those sections, so a different gcc or clang on PATH is a different tag."""
    manifest_tool = _load("manifest")
    manifest = manifest_tool.load()
    family = compiler_family(cc)
    std23 = c23_spelling(cc)
    w = Writer()
    w.comment("The C rails' build, written by tools/build/bcirfile.py from runtime/manifest.json.")
    w.tool("cc", cc)
    w.tool("ar", ar)
    w.tool("python", python)
    if cxx:
        w.tool("cxx", cxx)
    if bash:
        w.tool("bash", bash)
        for name, path in sorted((path_compilers or {}).items()):
            w.tool(name, path)
    planned: dict[str, list[str]] = {
        "sections": [],
        "libraries": [],
        "tools": [],
        "harnesses": [],
        "seam_libraries": [],
        "seam_tests": [],
        "variants": [],
        "kernels": [],
    }
    warnings = ["-Wall", "-Wextra", "-Werror"]

    def compile_c(owner, source_path, flags, extra_reads=(), header_of=None, cxx_unit=False):
        stem = Path(source_path).name.rsplit(".", 1)[0]
        # a directory of its own: BCIR Make never runs two targets that write into one directory
        # (bcir/make/run.py), so objects in a shared one would compile one at a time
        obj = f"{out}/obj/{owner}/{stem}/{stem}.o"
        tool = "cxx" if cxx_unit else "cc"
        dirs = (C_DIR, CPP_DIR) if cxx_unit else (C_DIR,)
        reads = [source_path, *header_closure(header_of or source_path, dirs), *extra_reads]
        includes = [f"-I{d}" for d in dirs]
        w.target(
            f"{owner}.{stem}.o",
            reads,
            [obj],
            [[tool, *flags, *includes, "-c", source_path, "-o", obj]],
        )
        return obj

    def c_flags(std: str, options: list[str], strict: bool = True) -> list[str]:
        return [f"-std={std}", "-O2", *(warnings if strict else []), *scoped(options, family)]

    def link_flags(links: list[str]) -> list[str]:
        out = []
        if "m" in links:
            out.append("-lm")
        if "pthread" in links:
            out.append("-pthread")
        return out

    def archive_path(lib: str) -> str:
        return f"{out}/lib/lib{lib}.a"

    # --- libraries: objects, then the archive ---
    for lib, unit in manifest["libraries"].items():
        objs = [
            compile_c(lib, f"{C_DIR}/{s}", c_flags(std23, unit.get("options", [])))
            for s in unit["sources"]
        ]
        w.target(lib, objs, [archive_path(lib)], [["ar", "rcs", archive_path(lib), *objs]])
        planned["libraries"].append(lib)

    def closure(kind: str, name: str) -> dict:
        return manifest_tool.closure(manifest, kind, name)

    def link_order(libs: list[str]) -> list[str]:
        """The archives in link order: every library before the libraries it uses (a static
        linker resolves left to right), whatever order the closure found them in."""
        units = {**manifest["libraries"], **manifest.get("seam_libraries", {})}
        order: list[str] = []
        seen: set[str] = set()

        def visit(lib: str) -> None:
            if lib in seen:
                return
            seen.add(lib)
            for dep in units[lib].get("libraries", []):
                visit(dep)
            order.append(lib)

        for lib in libs:
            visit(lib)
        return list(reversed(order))

    def link_exe(name, kind, unit, objs, cxx_link=False):
        libs = link_order(closure(kind, name)["libraries"])
        archives = [archive_path(lib) for lib in libs]
        links = sorted(set(closure(kind, name).get("link", [])) | set(unit.get("link", [])))
        exe = f"{out}/bin/{name}"
        tool = "cxx" if cxx_link else "cc"
        w.target(
            name,
            [*objs, *archives],
            [exe],
            [[tool, *objs, *archives, *link_flags(links), "-o", exe]],
        )

    # --- tools and harnesses: their mains, linked over their library closure ---
    for kind in ("tools", "harnesses"):
        for name, unit in manifest[kind].items():
            objs = [
                compile_c(name, f"{C_DIR}/{s}", c_flags(std23, unit.get("options", [])))
                for s in unit["sources"]
            ]
            link_exe(name, kind, unit, objs)
            planned[kind].append(name)

    # --- the C++ seam ---
    if cxx:
        cxx_flags = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
        for kind in ("seam_libraries", "seam_tests"):
            for name, unit in manifest[kind].items():
                objs = [
                    compile_c(name, f"{CPP_DIR}/{s}", cxx_flags, cxx_unit=True)
                    for s in unit["sources"]
                ]
                if kind == "seam_libraries":
                    w.target(
                        name, objs, [archive_path(name)], [["ar", "rcs", archive_path(name), *objs]]
                    )
                else:
                    link_exe(name, kind, unit, objs, cxx_link=True)
                planned[kind].append(name)

    # --- variants: the harness's whole closure rebuilt with the variant's change ---
    sanitizer_ok = {"thread": tsan}
    for name, variant in manifest["variants"].items():
        san = variant.get("sanitizer")
        if san and not sanitizer_ok.get(san, False):
            w.comment(f"variant {name} is not planned: no working {san} sanitizer runtime here")
            continue
        harness = manifest["harnesses"][variant["of"]]
        std = {11: "c11", 17: "c17"}.get(variant.get("standard"), std23)
        mutation = variant.get("mutation")
        strict = mutation is None
        extra = list(variant.get("options", []))
        if san:
            extra += [f"-fsanitize={san}"]
        lib_closure = closure("harnesses", variant["of"])
        units: list[tuple[str, list[str]]] = [
            (s, harness.get("options", [])) for s in harness["sources"]
        ]
        for lib in link_order(lib_closure["libraries"]):
            unit = manifest["libraries"][lib]
            units += [(s, unit.get("options", [])) for s in unit["sources"]]
        objs = []
        for source, options in units:
            path = f"{C_DIR}/{source}"
            reads: list[str] = []
            header_of = None
            if mutation and source == mutation["file"]:
                mutant = f"{out}/src/{name}/{source}"
                w.target(
                    f"{name}.mutant",
                    [f"{C_DIR}/{source}", "tools/build/mutate.py", "runtime/manifest.json"],
                    [mutant],
                    [["python", "tools/build/mutate.py", "--variant", name, "--out", mutant]],
                )
                header_of, path = path, mutant
            flags = [*c_flags(std, options, strict), *extra]
            objs.append(compile_c(name, path, flags, reads, header_of))
        links = sorted(set(lib_closure.get("link", [])) | set(harness.get("link", [])))
        exe = f"{out}/bin/{name}"
        sanflag = [f"-fsanitize={san}"] if san else []
        w.target(name, objs, [exe], [["cc", *objs, *sanflag, *link_flags(links), "-o", exe]])
        planned["variants"].append(name)

    # --- kernels: the oracle's emit with its driver appended, built in C11 ---
    for name, kernel in manifest["kernels"].items():
        unit_c = f"{out}/kernels/{name}/{name}.c"
        driver = f"{C_DIR}/kernels/{kernel['main']}"
        w.target(
            f"{name}.unit",
            [
                "tools/build/emit_kernel.py",
                "tools/build/manifest.py",
                "runtime/manifest.json",
                driver,
            ],
            [unit_c],
            [["python", "tools/build/emit_kernel.py", name, "-o", unit_c]],
            trees=["bcir"],
        )
        obj = f"{out}/obj/{name}/{name}.o"
        w.target(
            f"{name}.o",
            [unit_c],
            [obj],
            [["cc", "-std=c11", "-O2", *warnings, "-c", unit_c, "-o", obj]],
        )
        exe = f"{out}/bin/{name}"
        w.target(name, [obj], [exe], [["cc", obj, *link_flags(kernel.get("link", [])), "-o", exe]])
        planned["kernels"].append(name)

    # --- sections: the gate's sections as tasks, over the binaries planned above ---
    if bash:
        built = {n for kind in planned for n in planned[kind]}
        for name, section in manifest["sections"].items():
            binaries = section["harnesses"]
            missing = [b for b in binaries if b not in built]
            if missing:
                w.comment(f"section {name} is not planned: {', '.join(missing)} is not built here")
                continue
            paths = [f"{out}/bin/{b}" for b in binaries]
            verdict = f"{out}/verdicts/{name}"
            writes = [f"{verdict}/stdout", f"{verdict}/stderr", f"{verdict}/status"]
            w.target(
                f"section.{name}",
                ["tools/build/run_section.sh", section["script"], *paths],
                writes,
                [["bash", "tools/build/run_section.sh", *writes, section["script"], *paths]],
                trees=SECTION_TREES,
                uses=(
                    "cc",
                    "python",
                    *(sorted(path_compilers or {}) if section.get("compiler_only") else ()),
                ),
            )
            planned["sections"].append(name)
    return w.text(), planned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cc", default="cc", help="the C compiler")
    parser.add_argument("--cxx", help="the C++ compiler (omit to leave the seam out)")
    parser.add_argument("--ar", default="ar", help="the archiver")
    parser.add_argument("--python", default=sys.executable, help="the interpreter for the helpers")
    parser.add_argument("-o", "--out", type=Path, help="write here (default: stdout)")
    parser.add_argument("--planned", type=Path, help="also write what was planned, as JSON")
    parser.add_argument("--out-dir", default=OUT, help=f"where the outputs go (default {OUT})")
    parser.add_argument(
        "--sections", action="store_true", help="plan the gate's sections as tasks (needs bash)"
    )
    args = parser.parse_args(argv)
    resolved = {}
    args.bash = "bash" if args.sections else None
    for key in ("cc", "cxx", "ar", "python", "bash"):
        value = getattr(args, key)
        if value is None:
            resolved[key] = None
            continue
        found = tool_path(value)
        if not found:
            print(f"bcirfile: UNUSABLE: no {key} {value!r}", file=sys.stderr)
            return 2
        resolved[key] = found
    sanitizer = _load("sanitizer")
    tsan, _why = sanitizer.probe(resolved["cc"], "thread")
    try:
        text, planned = generate(
            resolved["cc"],
            resolved["cxx"],
            resolved["ar"],
            resolved["python"],
            tsan,
            out=args.out_dir,
            bash=resolved["bash"],
            path_compilers=path_compilers() if resolved["bash"] else None,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"bcirfile: UNUSABLE: {exc}", file=sys.stderr)
        return 2
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="ascii", newline="\n")
    if args.planned is not None:
        args.planned.write_text(
            json.dumps(planned, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
