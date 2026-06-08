"""Per-target codegen (Phase 9): real ARM/RISC-V/GPU/eBPF artifacts + C fallback."""

import subprocess
from shutil import which

from bcir.codegen import CODEGEN_TARGETS, codegen, codegen_c, emit_c_source
from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta

# target name -> the keyword that appears in `llc --version` when its backend is built in.
_ARCH_KW = {"x86_64": "x86-64", "aarch64": "aarch64", "riscv64": "riscv64",
            "nvptx64": "nvptx64", "bpf": "bpf"}


def _result():
    m = vector_add(1024)
    return m, optimize(m, TargetProfile.x86_avx512(), Theta.cool())


def _llc_version_text():
    llc = which("llc") or which("llc-18")
    if not llc:
        return None
    return subprocess.run([llc, "--version"], capture_output=True, text=True).stdout


def test_machine_targets_produce_artifacts():
    text = _llc_version_text()
    if text is None:
        return  # skip cleanly without llc
    tested = 0
    for name, kw in _ARCH_KW.items():
        if kw not in text:
            continue  # this llvm build lacks the backend
        r = codegen(*_result(), target_name=name)
        assert r.ok, f"{name}: {r.message}"
        assert r.artifact, f"{name}: empty artifact"
        tested += 1
    assert tested >= 2  # at least x86 + one cross-target on a normal llvm build


def test_spirv_is_graceful_when_absent():
    if _llc_version_text() is None:
        return
    r = codegen(*_result(), target_name="spirv64")
    # No crash: either it works (SPIR-V backend present) or fails cleanly.
    assert r.target == "spirv64"
    assert isinstance(r.message, str) and r.message


def test_c_source_fallback_compiles():
    if not (which("clang") or which("cc") or which("gcc")):
        return
    m, res = _result()
    assert "bcir_kernel" in emit_c_source(m, res)
    r = codegen_c(m, res)
    assert r.ok, r.message
    assert r.artifact


def test_every_target_is_a_lower_contract():
    # Each codegen target is a data descriptor (the bcir.target.lower_contract).
    assert {"aarch64", "riscv64", "nvptx64", "bpf"}.issubset(CODEGEN_TARGETS)
    assert CODEGEN_TARGETS["bpf"].elem == "i32"  # eBPF has no FP
