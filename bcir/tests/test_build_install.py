"""The install gate (BUILD-3): one version, the package's layout held to the manifest, and a consumer
that cannot see the source tree.

tools/build/install_consumer.py runs the real install and the out-of-tree build (CTest's
`build-install`, the `cmake-build` CI job); what is held here is the gate's own logic, with no CMake:
each of its refusals is driven by a fabricated prefix or text, so a gate that stopped refusing would
fail here first (docs/security/laws.md L2, L11).
"""

from __future__ import annotations

import importlib.util
import os
import re
import tempfile
from pathlib import Path

import bcir

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"bcir_build_{name}_under_test", _ROOT / "tools" / "build" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


install_tool = _load("install_consumer")
manifest_tool = _load("manifest")


def test_the_repository_has_one_version():
    """pyproject.toml's version is the package's; CMake reads it instead of spelling a second one
    (CMake had said 0.1.0 while the package said 0.2.0), and bcir.__version__ agrees."""
    version = install_tool.package_version((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert bcir.__version__ == version, (bcir.__version__, version)
    cmake = (_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "VERSION ${BCIR_PACKAGE_VERSION}" in cmake, "the CMake project spells its own version"
    assert not re.search(r"project\(BCIR\s+VERSION\s+[0-9]", cmake), "a literal project version"
    assert "pyproject.toml" in cmake and "CMAKE_CONFIGURE_DEPENDS" in cmake
    template = (_ROOT / "cmake" / "bcir_version.h.in").read_text(encoding="utf-8")
    for field in ("MAJOR", "MINOR", "PATCH"):
        assert f"BCIR_VERSION_{field} @PROJECT_VERSION_{field}@" in template, field


def test_a_version_line_that_is_not_one_is_refused():
    for text in ("", 'version = "0.2"\n', 'version = "0.2.0"\nversion = "0.3.0"\n'):
        try:
            install_tool.package_version(text)
        except install_tool.Unusable:
            continue
        raise AssertionError(f"{text!r} read as a version")
    assert install_tool.package_version('name = "bcir"\nversion = "1.20.3"\n') == "1.20.3"


def _prefix(prefix: Path, manifest: dict, headers: list[str], with_seam: bool) -> Path:
    (prefix / "lib" / "cmake" / "BCIR").mkdir(parents=True)
    (prefix / "bin").mkdir()
    (prefix / "include" / "bcir").mkdir(parents=True)
    libraries = list(manifest["libraries"]) + (
        list(manifest["seam_libraries"]) if with_seam else []
    )
    for name in libraries:
        (prefix / "lib" / f"lib{name}.a").write_bytes(b"!<arch>\n")
    for name in manifest["tools"]:
        tool = prefix / "bin" / name
        tool.write_text("#!/bin/sh\n", encoding="utf-8")
        tool.chmod(0o755)
    for name in headers:
        (prefix / "include" / "bcir" / name).write_text("/* */\n", encoding="utf-8")
    for name in install_tool.PACKAGE_FILES:
        (prefix / "lib" / "cmake" / "BCIR" / name).write_text("#\n", encoding="utf-8")
    return prefix


def test_the_installed_layout_is_held_to_the_manifest():
    """A complete prefix is clean; each missing piece, and a header the tree does not publish, is a
    finding that names it."""
    manifest = manifest_tool.load()
    dirs = {"lib": "lib", "bin": "bin", "include": "include"}
    for with_seam in (True, False):
        headers = install_tool.public_headers(_ROOT, with_seam)
        assert "bcir_version.h" in headers and "bcir_runtime.h" in headers
        assert ("bcir_orchestrator.hpp" in headers) is with_seam
        assert not any(h.startswith(("cfront_", "test_", "uart_", "cmsis_")) for h in headers)
        faults = {
            "header bcir_runtime.h": lambda p: (p / "include/bcir/bcir_runtime.h").unlink(),
            "header uart_regs.h": lambda p: (p / "include/bcir/uart_regs.h").write_text(""),
            "library bcir_base": lambda p: (p / "lib/libbcir_base.a").unlink(),
            "tool bcir-cc": lambda p: (p / "bin/bcir-cc").chmod(0o644),
            "package file BCIRConfigVersion.cmake": lambda p: (
                p / "lib/cmake/BCIR/BCIRConfigVersion.cmake"
            ).unlink(),
        }
        if os.name != "posix":  # an executable bit is a POSIX file's; os.access says yes elsewhere
            del faults["tool bcir-cc"]
        with tempfile.TemporaryDirectory() as tmp:
            clean = _prefix(Path(tmp) / "clean", manifest, headers, with_seam)
            assert install_tool.layout_problems(clean, manifest, headers, dirs, with_seam) == []
            for n, (what, fault) in enumerate(faults.items()):
                prefix = _prefix(Path(tmp) / f"fault{n}", manifest, headers, with_seam)
                fault(prefix)
                problems = install_tool.layout_problems(prefix, manifest, headers, dirs, with_seam)
                assert any(p.startswith(what) for p in problems), (what, problems)


def test_the_consumer_cannot_see_the_source_tree():
    """The consumer's directory holds its CMakeLists.txt and a copy of the harness, nothing else, so
    a quoted #include resolves through the installed include directory alone; and the consumer's
    project names no path of the tree."""
    consumer = install_tool.CONSUMER.read_text(encoding="utf-8")
    assert "runtime/" not in consumer and "CMAKE_SOURCE_DIR}/.." not in consumer
    assert "find_package(BCIR ${BCIR_CONSUMER_VERSION} REQUIRED CONFIG)" in consumer
    with tempfile.TemporaryDirectory() as tmp:
        src = install_tool.stage_consumer(Path(tmp), _ROOT / "runtime" / "c" / "test_runtime.c")
        assert sorted(p.name for p in src.iterdir()) == ["CMakeLists.txt", "test_runtime.c"]


def test_the_cache_is_read_as_the_build_wrote_it():
    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp)
        (build / "CMakeCache.txt").write_text(
            "# comment\nCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc\nCMAKE_GENERATOR:INTERNAL=Ninja\n"
            "BCIR_BUILD_CPP:BOOL=OFF\n//help\nCMAKE_INSTALL_LIBDIR:PATH=lib64\n",
            encoding="utf-8",
        )
        cache = install_tool.read_cache(build)
        assert (
            cache["CMAKE_C_COMPILER"] == "/usr/bin/gcc" and cache["CMAKE_INSTALL_LIBDIR"] == "lib64"
        )
        assert cache["BCIR_BUILD_CPP"] == "OFF" and cache["CMAKE_GENERATOR"] == "Ninja"
        try:
            install_tool.read_cache(build / "absent")
        except install_tool.Unusable:
            pass
        else:
            raise AssertionError("an unconfigured directory read as a build")


def test_the_install_rules_take_their_lists_from_the_manifest():
    """cmake/BCIRInstall.cmake installs and exports the manifest's libraries and tools by their
    manifest lists, finds Threads for the package when a library publishes it, and the libraries'
    include directory is the source tree's only inside the build (L12)."""
    install = (_ROOT / "cmake" / "BCIRInstall.cmake").read_text(encoding="utf-8")
    assert "set(BCIR_EXPORTED_LIBRARIES ${BCIR_MANIFEST_libraries})" in install
    assert "${BCIR_MANIFEST_seam_libraries}" in install
    assert "set(BCIR_EXPORTED_TOOLS ${BCIR_MANIFEST_tools})" in install
    assert "EXPORT BCIRTargets" in install and "NAMESPACE BCIR::" in install
    assert '"${BCIR_C_DIR}/bcir_*.h"' in install, "the public-header rule moved"
    config = (_ROOT / "cmake" / "BCIRConfig.cmake.in").read_text(encoding="utf-8")
    assert "find_dependency(Threads)" in config and "BCIRTargets.cmake" in config
    runtime = (_ROOT / "runtime" / "c" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert '"$<BUILD_INTERFACE:${BCIR_C_DIR}>"' in runtime
    assert '"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/bcir>"' in runtime
    tests = (_ROOT / "cmake" / "BCIRTests.cmake").read_text(encoding="utf-8")
    assert "NAME build-install" in tests and "install_consumer.py" in tests


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    for name in tests:
        getattr(mod, name)()
        print(f"PASS {name}")


def test_only_find_packages_own_refusal_counts_as_the_version_files():
    """The consumer refuses a version it did not ask for on its own, so a version file that accepts
    any request would still see the next major refused -- by the wrong check (L11). The gate counts
    the refusal only when CMake's find_package says it, wrapped as CMake wraps it."""
    cmake = (
        "CMake Error at CMakeLists.txt:15 (find_package):\n  Could not find a configuration file for "
        'package "BCIR" that is compatible\n  with requested version "1.0.0".\n\n  The following '
        "configuration files were considered but not accepted:\n\n    /p/lib/cmake/BCIR/"
        "BCIRConfig.cmake, version: 0.2.0\n"
    )
    assert install_tool.refused_by_version(cmake, "1.0.0", "0.2.0")
    assert not install_tool.refused_by_version(cmake, "2.0.0", "0.2.0")
    ours = "CMake Error at CMakeLists.txt:17 (message):\n  consumer: found BCIR 0.2.0, asked for 1.0.0\n"
    assert not install_tool.refused_by_version(ours, "1.0.0", "0.2.0")
