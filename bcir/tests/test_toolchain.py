"""Resident-host link adaptation and coherent versioned LLVM discovery."""

import os
from pathlib import Path
import stat
import tempfile
from unittest import mock

from bcir.toolchain import host_bash, host_link_args, resolve_llvm_tools


_ROOT = Path(__file__).resolve().parents[2]


def _exe(directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\nexit 0\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return os.path.abspath(path)


def _filesystem_which(command: str, path: str | None = None) -> str | None:
    """Small ungated which used only for synthetic resolver fixtures."""
    candidates = [command] if os.path.dirname(command) else [
        os.path.join(directory, command) for directory in (path or "").split(os.pathsep) if directory]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _resolve(*names: str, **kwargs):
    # Quick-tier gating wraps the process-wide shutil.which. These unit tests exercise
    # resolver policy over isolated fake executables, not host capability visibility.
    with mock.patch("bcir.toolchain.shutil.which", _filesystem_which):
        return resolve_llvm_tools(*names, **kwargs)


def test_host_link_args_only_adapts_libm_on_windows():
    flags = ["-lcblas", "-lm", "-Wl,--as-needed", "-lm"]
    assert host_link_args(flags, "win32") == ["-lcblas", "-Wl,--as-needed"]
    assert host_link_args(flags, "windows") == ["-lcblas", "-Wl,--as-needed"]
    assert host_link_args(flags, "linux") == flags
    assert host_link_args(flags, "darwin") == flags


def test_host_bash_prefers_git_for_windows_over_the_wsl_launcher():
    with tempfile.TemporaryDirectory() as d:
        git = os.path.join(d, "cmd", "git.exe")
        git_bash = os.path.join(d, "bin", "bash.exe")
        wsl_bash = os.path.join(d, "Windows", "System32", "bash.exe")
        files = {git_bash}

        def which(name):
            return {"git": git, "bash": wsl_bash}.get(name)

        assert host_bash("win32", which=which, is_file=lambda path: path in files,
                         env={}) == git_bash


def test_host_bash_rejects_a_lone_windows_wsl_launcher():
    wsl_bash = os.path.join(os.sep, "Windows", "System32", "bash.exe")
    assert host_bash("win32", which=lambda name: wsl_bash if name == "bash" else None,
                     is_file=lambda path: False, env={}) is None


def test_host_bash_uses_path_normally_on_posix():
    assert host_bash("linux", which=lambda name: "/bin/bash" if name == "bash" else None,
                     env={}) == "/bin/bash"


def test_versioned_llvm_resolves_highest_complete_major():
    with tempfile.TemporaryDirectory() as d:
        for name in ("clang-19", "llvm-link-19", "lli-19", "clang-20", "llvm-link-20", "lli-20"):
            _exe(d, name)
        r = _resolve("clang", "llvm-link", "lli", pipeline="test", env={"PATH": d})
        assert r.ok and r.major == 20, r.message
        assert all(path.endswith("-20") for path in r.paths.values())


def test_versioned_llvm_accepts_windows_executable_suffixes():
    with tempfile.TemporaryDirectory() as d:
        for name in ("llc-20.exe", "llvm-objdump-20.exe"):
            _exe(d, name)
        r = _resolve("llc", "llvm-objdump", pipeline="windows names", env={"PATH": d})
        assert r.ok and r.major == 20, r.message
        assert all(path.endswith("-20.exe") for path in r.paths.values())


def test_versioned_llvm_never_mixes_majors():
    with tempfile.TemporaryDirectory() as d:
        _exe(d, "clang-20")
        _exe(d, "llvm-link-19")
        _exe(d, "lli-20")
        r = _resolve("clang", "llvm-link", "lli", pipeline="mixed", env={"PATH": d})
        assert not r.ok
        assert "missing tool(s) for mixed" in r.message
        assert "clang-20" in r.attempted and "llvm-link-19" in r.attempted


def test_llvm_suffix_and_bin_take_precedence():
    with tempfile.TemporaryDirectory() as selected, tempfile.TemporaryDirectory() as other:
        expected = {}
        for name in ("llc", "llvm-objdump"):
            expected[name] = _exe(selected, name + "-18")
            _exe(other, name)
            _exe(other, name + "-22")
        env = {"LLVM_BIN": selected, "LLVM_SUFFIX": "18", "PATH": other}
        r = _resolve("llc", "llvm-objdump", pipeline="suffix", env=env)
        assert r.ok and r.major == 18
        assert dict(r.paths) == expected


def test_mlir22_environment_pins_backend_tools_to_its_distribution():
    env_script = (_ROOT / "tools" / "local" / "env_mlir22.sh").read_text(
        encoding="utf-8")
    assert 'export LLVM_BIN="${_ENV}/bin"' in env_script


def test_unversioned_complete_toolchain_precedes_versioned():
    with tempfile.TemporaryDirectory() as d:
        expected = {}
        for name in ("clang", "wasm-ld"):
            expected[name] = _exe(d, name)
            _exe(d, name + "-22")
        r = _resolve("clang", "wasm-ld", pipeline="wasm", env={"PATH": d})
        assert r.ok and r.major is None
        assert dict(r.paths) == expected


def test_detectably_old_unversioned_toolchain_does_not_shadow_supported_versioned_set():
    with tempfile.TemporaryDirectory() as root:
        old = os.path.join(root, "llvm-14", "bin")
        new = os.path.join(root, "llvm-22", "bin")
        os.makedirs(old); os.makedirs(new)
        for name in ("clang", "llvm-as", "llc"):
            _exe(old, name)
            _exe(new, name + "-22")
        r = _resolve("clang", "llvm-as", "llc", pipeline="minimum",
                     env={"PATH": os.pathsep.join((old, new))})
        assert r.ok and r.major == 22, r.message
        assert all(path.endswith("-22") for path in r.paths.values())


def test_detectably_old_unversioned_toolchain_is_not_reported_supported():
    with tempfile.TemporaryDirectory() as root:
        old = os.path.join(root, "llvm-14", "bin")
        os.makedirs(old)
        for name in ("clang", "llvm-as", "llc"):
            _exe(old, name)
        r = _resolve("clang", "llvm-as", "llc", pipeline="minimum", env={"PATH": old})
        assert not r.ok


def test_llvm_resolution_reports_attempted_names():
    with tempfile.TemporaryDirectory() as d:
        r = _resolve("llc", pipeline="codegen", env={"PATH": d})
        assert not r.ok and r.missing == ("llc",)
        assert "attempted llc" in r.message
