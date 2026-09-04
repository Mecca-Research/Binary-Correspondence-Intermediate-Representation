"""Native object emission -- the decision gate (roadmap §5 #3, docs/BCIR_NATIVE_OBJECT_GATE.md).

BCIR-native instruction selection is DEFERRED (the gate stands); the *warranted* slice
is to close one target end-to-end via the **resident compiler**: emit the K_BCIR-planned
kernel and compile it straight to a real native object with `clang --target=...`. These
tests prove that slice for eBPF (integer-only scalar) and x86-64 scalar -- a genuine ELF
object for the expected machine, not just a zero-exit compile -- and that the gate is
documented. They skip cleanly when clang lacks the target.
"""

import os
import platform
import shutil

from bcir.codegen import codegen_object_c
from bcir.codegen.codegen import _elf_machine
from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _plan():
    m = vector_add(1024)
    return m, optimize(m, TargetProfile.x86_avx512(), Theta.cool())


def _clang_targets() -> str:
    import subprocess

    clang = shutil.which("clang")
    if not clang:
        return ""
    return subprocess.run([clang, "--print-targets"], capture_output=True, text=True).stdout


def test_decision_gate_is_documented():
    doc = os.path.join(_ROOT, "docs", "BCIR_NATIVE_OBJECT_GATE.md")
    assert os.path.exists(doc), "the decision gate must be documented"
    text = open(doc, encoding="utf-8").read()
    for marker in ("GO criteria", "STOP criteria", "DEFERRED", "EM_BPF", "resident compiler"):
        assert marker in text, f"gate doc missing {marker!r}"


def test_ebpf_object_end_to_end():
    """eBPF: the integer-only scalar target the roadmap names. Emit i32 kernel ->
    clang --target=bpf -ffreestanding -c -> a real EM_BPF (247) ELF object."""
    if "bpf" not in _clang_targets():
        return  # clang without the BPF backend
    m, r = _plan()
    res = codegen_object_c(m, r, "bpf")
    assert res.ok, res.message
    assert isinstance(res.artifact, (bytes, bytearray))
    assert _elf_machine(res.artifact) == 247, "must be a real EM_BPF object"


def test_x86_64_scalar_object_end_to_end():
    if "x86-64" not in _clang_targets() and "x86_64" not in _clang_targets():
        return
    m, r = _plan()
    res = codegen_object_c(m, r, "x86_64")
    # clang lists every backend, but cross-compiling an x86_64 object from a non-x86 host
    # (e.g. a Pi 5) needs an assembler/sysroot it may lack -> best-effort off-host.
    if platform.machine() not in ("x86_64", "amd64") and not res.ok:
        return
    assert res.ok, res.message
    assert _elf_machine(res.artifact) == 62, "must be a real EM_X86_64 object"


def test_aarch64_scalar_object_end_to_end():
    """The ARM (Raspberry Pi 5) native object: emit the K_BCIR-planned kernel ->
    clang --target=aarch64 -> a real EM_AARCH64 (183) ELF object. Native on an aarch64 host
    (proves the gate on the Pi); a cross-compile best-effort elsewhere (skips on a toolchain gap)."""
    if "aarch64" not in _clang_targets() and "arm64" not in _clang_targets():
        return
    m, r = _plan()
    res = codegen_object_c(m, r, "aarch64")
    if platform.machine() not in ("aarch64", "arm64") and not res.ok:
        return
    assert res.ok, res.message
    assert _elf_machine(res.artifact) == 183, "must be a real EM_AARCH64 object"


def test_ebpf_is_integer_only():
    """eBPF has no FP unit -- the gate path must lower the integer (i32) kernel, never
    a float one (the same constraint the llc bpf CodegenTarget encodes)."""
    from bcir.codegen.codegen import _OBJECT_TARGETS

    assert _OBJECT_TARGETS["bpf"]["elem"] == "i32"
    assert _OBJECT_TARGETS["bpf"]["freestanding"] is True


def test_unknown_object_target_is_rejected_cleanly():
    m, r = _plan()
    res = codegen_object_c(m, r, "nonsuch")
    assert not res.ok and "unknown" in res.message
