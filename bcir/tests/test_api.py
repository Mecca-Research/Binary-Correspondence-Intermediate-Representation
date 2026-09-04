"""The library API facade (library-first): plan -> a deployable kernel artifact
(C source + ABI header + metadata + R12 attestation), usable AOT or embedded."""

import shutil

from bcir.api import build_artifact, compile_kernel


def test_artifact_carries_metadata_and_r12_attestation():
    a = build_artifact("vector_add", target="x86_avx512", theta="cool")
    assert a.program == "vector_add" and a.width == 16 and a.op == "+"
    assert a.score == 7808 and a.manifest_digest != 0
    assert a.attested and a.diagnostics == ()  # R12 clean
    # width 16 == the AVX-512 widest lane -> the shipped kernel is the go-fast
    # idiomatic loop (the resident compiler vectorizes to >= the lane), not blocked.
    assert "full hardware lane" in a.kernel_c and "16u" not in a.kernel_c


def test_throttled_artifact_caps_to_honor_the_thermal_lane():
    a = build_artifact("vector_add", target="x86_avx512", theta="hot")  # vec8 < lane 16
    assert a.width == 8 and a.attested
    assert "throttled lane" in a.kernel_c and "< 8u" in a.kernel_c  # cap honored


def test_header_declares_the_kernel_abi():
    a = build_artifact("vector_add", target="x86_avx512")
    assert "#ifndef BCIR_BCIR_KERNEL_H" in a.header_c
    assert "void bcir_kernel(const float *restrict A" in a.header_c
    assert "size_t n);" in a.header_c


def test_kernel_is_the_c23_backend_output():
    a = build_artifact("vector_add", target="x86_avx512", theta="hot")  # hot -> vec8
    assert a.width == 8 and "width=8" in a.kernel_c
    assert "restrict" in a.kernel_c and "_Static_assert" in a.kernel_c


def test_metadata_json_excludes_source():
    a = build_artifact("vector_add", target="x86_avx512")
    md = a.metadata()
    assert "kernel_c" not in md and "header_c" not in md
    assert md["attested"] is True and md["width"] == 16


def test_to_files_writes_kernel_and_header(tmp_path=None):
    import tempfile, os

    a = build_artifact("vector_add", target="x86_avx512")
    d = tempfile.mkdtemp(prefix="bcir-api-")
    try:
        paths = a.to_files(d)
        assert os.path.exists(paths["kernel"]) and os.path.exists(paths["header"])
        assert "bcir_kernel" in open(paths["kernel"]).read()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_compile_kernel_aot_self_checks():
    if shutil.which("clang") is None and shutil.which("cc") is None and shutil.which("gcc") is None:
        return  # skip cleanly without a C compiler
    art, (ok, out) = compile_kernel("vector_add", target="x86_avx512", run=True, theta="cool")
    assert art.attested and ok, out
    assert "OK" in out


# --- budget feasibility: the correctness win a budget-unaware compiler can't make -


def test_budget_makes_bcir_emit_the_feasible_kernel():
    # Unconstrained, BCIR (like a max-width compiler) picks vec16. Under a 700
    # thermal/power cap, vec16 is infeasible -- BCIR picks the feasible vec8.
    un = build_artifact("vector_add", target="x86_avx512")
    assert un.width == 16 and un.feasible and un.budget == ()
    cap = build_artifact("vector_add", target="x86_avx512", budget="thermal=700,power=700")
    assert cap.width == 8 and cap.feasible and cap.score == 9472
    assert cap.budget == ((5, 700), (6, 700))


def test_the_naive_max_width_plan_is_infeasible_under_the_cap():
    # The correctness contrast: the unconstrained vec16 plan violates the same cap
    # BCIR's vec8 satisfies (a budget-unaware compiler would emit the infeasible one).
    from bcir.examples import vector_add
    from bcir.kbcir import Budget, TARGETS, feasible, optimize
    from bcir.kbcir.cost import Theta

    un = optimize(vector_add(1024), TARGETS["x86_avx512"], Theta.cool())
    cap = Budget.of(thermal=700, power=700)
    assert not feasible(un, Theta.cool(), cap)  # vec16 violates the cap


def test_an_overtight_budget_is_infeasible():
    from bcir.kbcir import Infeasible

    try:
        build_artifact("vector_add", target="x86_avx512", budget="thermal=500,power=500")
        assert False, "expected Infeasible"
    except Infeasible:
        pass
