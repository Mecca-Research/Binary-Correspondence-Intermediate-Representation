"""The C rails' own build as a BCIRfile (BUILD-6's gate): tools/build/bcirfile.py writes it from
runtime/manifest.json, and it must pass every BCIR Make law and plan exactly the targets the
manifest lists -- every library, tool, harness, seam unit, kernel, and every variant this host can
build -- each named as the manifest names it. A unit the plan dropped would be a binary nothing
builds; a target the manifest lacks would be a second source list (docs/security/laws.md L14).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from bcir.make import check, lower, parse, schedule

_ROOT = Path(__file__).resolve().parents[2]


def _posix() -> bool:
    """BCIR Make names a tool by its POSIX path -- the grammar has no other spelling of an absolute
    path -- and runs POSIX programs, so on another host (a Windows runner) there is nothing here to
    judge, and each test that builds a BCIRfile over a host path says so by returning."""
    return os.name == "posix"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"bcir_build_{name}_under_test", _ROOT / "tools" / "build" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_rails_build_plans_the_manifests_targets():
    if not _posix():
        return
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    ar = shutil.which("ar")
    if cc is None or ar is None:
        return  # no compiler visible here (a test tier that hides the toolchain)
    generator = _load("bcirfile")
    manifest = json.loads((_ROOT / "runtime" / "manifest.json").read_text(encoding="utf-8"))
    cxx = shutil.which("g++") or shutil.which("clang++")
    for tsan in (True, False):
        text, planned = generator.generate(cc, cxx, ar, shutil.which("python3") or "python3", tsan)
        bf = parse(text.encode("ascii"))
        assert check(bf, _ROOT, check_tools=True) == [], "the rails' BCIRfile breaks a law"
        names = {t.name for t in bf.targets}
        for kind in ("libraries", "tools", "harnesses", "kernels"):
            assert sorted(planned[kind]) == sorted(manifest[kind]), kind
            assert set(manifest[kind]) <= names, kind
        if cxx:
            for kind in ("seam_libraries", "seam_tests"):
                assert sorted(planned[kind]) == sorted(manifest[kind]), kind
        sanitized = {n for n, v in manifest["variants"].items() if v.get("sanitizer")}
        expected = set(manifest["variants"]) - (set() if tsan else sanitized)
        assert set(planned["variants"]) == expected, (
            tsan,
            sorted(set(planned["variants"]) ^ expected),
        )
        assert not set(manifest["fuzzers"]) & names, "the fuzzers are the fuzzer preset's"
        waves = schedule(bf, lower(bf), 2)
        assert all(len(w) <= 2 for w in waves) and sum(map(len, waves)) == len(bf.targets)


def test_a_tool_keeps_the_name_it_is_run_by():
    """A BCIRfile declares a tool where PATH finds it, its symlinks kept: Clang's driver picks C or
    C++ from the name it is run by, and `clang++` is a link to `clang`, so a tool declared by its
    resolved path linked the C++ seam as C (found by the first BCIR Make build with Clang). Here a
    compiler reached through a link of its own name is declared by that link, and a C++ unit
    links through the declared path of `clang++` where the host has one."""
    if not _posix():
        return
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    if cc is None:
        return  # no compiler visible here (a test tier that hides the toolchain)
    generator = _load("bcirfile")
    with tempfile.TemporaryDirectory() as tmp:
        link = Path(tmp) / "bin" / "bcir-cc-link"
        link.parent.mkdir()
        link.symlink_to(cc)
        assert generator.tool_path(str(link)) == str(link)
        assert generator.tool_path("no-such-tool-bcir") is None
        clangxx = shutil.which("clang++")
        if clangxx is not None and Path(clangxx).resolve().name != Path(clangxx).name:
            unit = Path(tmp) / "seam.cpp"
            unit.write_text(
                '#include <string>\n#include <vector>\nint main() { std::vector<std::string> v{"a"};'
                " return static_cast<int>(v.size()) - 1; }\n",
                encoding="utf-8",
            )
            declared = generator.tool_path("clang++")
            done = subprocess.run(
                [declared, str(unit), "-o", str(Path(tmp) / "seam")], capture_output=True
            )
            assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-600:]


def test_an_object_reads_the_headers_its_source_reaches():
    """A header edit must make exactly its includers stale, so each object claims the headers its
    source reaches through quoted includes, transitively."""
    generator = _load("bcirfile")
    closure = generator.header_closure("runtime/c/test_runtime.c", ("runtime/c",))
    assert "runtime/c/bcir_runtime.h" in closure, closure
    with tempfile.TemporaryDirectory() as tmp:
        assert generator.header_closure(str(Path(tmp) / "absent.c"), ("runtime/c",)) == []
