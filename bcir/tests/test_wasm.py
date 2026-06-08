"""WASM deployment target (Phase 7) tests."""

from shutil import which

from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.lower.wasm import compile_to_wasm, is_valid_wasm, run_wasm_node


def _result():
    m = vector_add(1024)
    return m, optimize(m, TargetProfile.x86_avx512(), Theta.cool())


def test_is_valid_wasm_header():
    assert is_valid_wasm(b"\x00asm\x01\x00\x00\x00")
    assert not is_valid_wasm(b"\x00asm\x02\x00\x00\x00")
    assert not is_valid_wasm(b"not wasm")


def test_compiles_to_valid_wasm():
    if not which("clang") or not (which("wasm-ld") or which("wasm-ld-18")):
        return  # skip cleanly without the wasm toolchain
    m, res = _result()
    ok, data, msg = compile_to_wasm(m, res, fn_name="bcir_kernel")
    assert ok, msg
    assert is_valid_wasm(data)


def test_wasm_runs_and_self_checks_via_node():
    if not which("node") or not which("clang") or not (which("wasm-ld") or which("wasm-ld-18")):
        return  # skip cleanly without node + the wasm toolchain
    m, res = _result()
    ok, out = run_wasm_node(m, res, fn_name="bcir_kernel")
    assert ok, out
    assert "OK" in out
