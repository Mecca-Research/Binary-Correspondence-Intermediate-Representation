"""G2 -- autonomous matmul+activation epilogue fusion, priced by the EXISTING tropical optimizer.

The canonical "matmul -> activation" epilogue (VISION_ALIGNMENT_AUDIT Pillar 3c): a ``gem.matmul`` whose
output feeds a ``gem.activation`` is FUSED into a single pass that applies the activation INLINE as each
matmul output element is produced (no intermediate buffer materialized). The fusion is the OPTIMIZER'S
PRICED CHOICE, not a hardcoded rewrite -- it falls out of the same deforestation discount
(``realize._DEFOREST_FACTOR``) that prices elementwise producer->consumer fusion: the fused plan
(producer + consumer in one phase, intermediate deforested) beats the unfused plan (a phase barrier
materializes the intermediate). Every adopted fusion emits a proof-carrying ``FusionCertificate`` (modeled
on ``bundle.BundleCertificate``).

Covers: the optimizer CHOOSES fusion (fused score < unfused, a certificate emitted) on matmul->relu and
matmul->gelu; the pricing is the existing deforestation discount (not a bespoke one); fusion is a clean
NO-OP when illegal (the intermediate has another consumer) or non-fusible (softmax, a row reduction);
detection is precise (sole-consumer rule); and -- when a C toolchain is visible -- the fused emitted kernel
compiles and EQUALS the unfused reference (matmul then activation), bit-exact for relu and to float
round-off for the transcendentals. All deterministic + pure-Python on the oracle rail."""

import os
import random
import shutil
import subprocess
import tempfile

from bcir.examples import matmul_activation
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.kbcir.fusion import (FUSIBLE_ACTIVATIONS, NON_FUSIBLE_ACTIVATIONS, FusionCertificate,
                               MatmulActivationEpilogue, find_matmul_activation_epilogues,
                               is_fusible_activation, optimize_fused)
from bcir.kbcir.activation import activation_reference
from bcir.kbcir.matmul import matmul_reference
from bcir.toolchain import host_link_args
from bcir.kbcir.realize import _DEFOREST_FACTOR, fused_candidates
from bcir.lower.c_kernel import (emit_activation_kernel_c, emit_blas_gemm_c, emit_matmul_activation_c)
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def _two_reader_module(kind: str = "relu", n: int = 1024) -> Module:
    """An ILLEGAL epilogue: a second claim also reads the matmul output (rid 3), so the activation is NOT
    the sole consumer -- the raw product is needed elsewhere, the intermediate must materialize, fusion is
    unsound. The detector must reject it."""
    m = Module(name="two_reader")
    for rid, nm in ((1, "A"), (2, "B"), (3, "CPRE"), (4, "C"), (5, "X")):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(n,), name=nm))
    mm = Claim(id=1, opcode=Opcode.T_MACC, lane=Lane.T, stride_class=StrideClass.TILE, count=n,
               rd=(1, 2), wr=(3,), op="gem.matmul", domain=Domain.RAM)
    act = Claim(id=2, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT, count=n,
                rd=(3,), wr=(4,), op=f"gem.activation:{kind}", domain=Domain.RAM)
    other = Claim(id=3, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=n,
                  rd=(3, 5), wr=(5,), op="vector.add", domain=Domain.RAM)   # also reads the raw product
    m.add_phase(Phase(phase_id=0, deps=(), claims=[mm, act, other]))
    return m


# --- detection: the sole-consumer epilogue + the softmax scope-out -------------------------------

def test_detects_the_matmul_activation_epilogue():
    epis = find_matmul_activation_epilogues(matmul_activation(kind="relu"))
    assert len(epis) == 1
    e = epis[0]
    assert isinstance(e, MatmulActivationEpilogue)
    assert e.matmul_id == 1 and e.activation_id == 2          # the producer / consumer pair
    assert e.inter_rid == 3 and e.out_rid == 4                # the elided intermediate + the final output
    assert e.act_kind == "relu" and e.fusible


def test_detection_rejects_a_non_sole_consumer():
    # the legality boundary: if anything else reads the un-activated product, it must materialize -> no
    # epilogue is detected (the producer->consumer single-reader rule).
    assert find_matmul_activation_epilogues(_two_reader_module()) == []


def test_softmax_is_detected_but_flagged_non_fusible():
    # softmax is a last-axis ROW REDUCTION: its normalizer needs the whole row, so no output element is
    # final as it drops out of the matmul -- it can NOT be an inline epilogue. Detected, flagged non-fusible.
    epis = find_matmul_activation_epilogues(matmul_activation(kind="softmax"))
    assert len(epis) == 1 and epis[0].act_kind == "softmax"
    assert not epis[0].fusible
    assert set(FUSIBLE_ACTIVATIONS) == {"relu", "sigmoid", "tanh", "gelu"}
    assert set(NON_FUSIBLE_ACTIVATIONS) == {"softmax"}
    assert is_fusible_activation("relu") and not is_fusible_activation("softmax")


# --- the optimizer CHOOSES fusion: fused score < unfused, a certificate emitted -------------------

def test_optimizer_chooses_fusion_on_matmul_relu_and_gelu():
    for kind in ("relu", "gelu"):
        m = matmul_activation(kind=kind)
        res, certs = optimize_fused(m, AVX, COOL)
        assert len(certs) == 1, kind
        c = certs[0]
        assert isinstance(c, FusionCertificate)
        assert c.act_kind == kind
        assert c.fused_score < c.unfused_score                # the priced win -- fusion is the cheaper plan
        assert c.gain == c.unfused_score - c.fused_score > 0
        assert c.matmul_id == 1 and c.activation_id == 2
        # the fused score IS the plan score of the module as given (producer+consumer same phase).
        assert c.fused_score == res.score == optimize(m, AVX, COOL).score


def test_fusion_pricing_is_the_existing_deforestation_discount_not_a_bespoke_one():
    # The win is exactly the deforestation memory discount realize.py already applies: the activation claim
    # reads the operand the matmul produced IN THE SAME PHASE, so fused_candidates couples its cost by
    # _DEFOREST_FACTOR. Prove the fusion module introduces NO new discount -- it reuses this one.
    from bcir.kbcir.cost import MEMORY
    # (1) _DEFOREST_FACTOR is the existing constant: discounts ONLY the memory axis, identity elsewhere.
    assert _DEFOREST_FACTOR[MEMORY] < 256 and all(
        f == 256 for i, f in enumerate(_DEFOREST_FACTOR) if i != MEMORY)
    # (2) the deforestation discount is what fused_candidates bakes into the activation: re-pricing the
    #     activation as if it did NOT read the matmul's output (no producer in the phase) costs MORE on the
    #     memory axis. Build the producer-less variant and compare the activation claim's candidate base.
    m = matmul_activation(kind="relu")
    fused_base = fused_candidates(m, AVX)[2][0].base                    # claim 2 = the activation, deforested
    standalone = Module(name="standalone_act")
    for rid, nm in ((3, "CPRE"), (4, "C")):
        standalone.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(1024,), name=nm))
    standalone.add_phase(Phase(phase_id=0, deps=(), claims=[
        Claim(id=2, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT, count=1024,
              rd=(3,), wr=(4,), op="gem.activation:relu", domain=Domain.RAM)]))
    full_base = fused_candidates(standalone, AVX)[2][0].base            # no producer -> NOT deforested
    assert fused_base.v[MEMORY] < full_base.v[MEMORY]                   # the deforestation memory discount
    # (3) the certificate's gain is positive (the priced win materializes in the plan score).
    _, certs = optimize_fused(m, AVX, COOL)
    assert certs and certs[0].gain > 0


# --- fusion is a clean NO-OP when illegal / non-fusible / unprofitable ----------------------------

def test_fusion_is_a_noop_when_the_intermediate_has_another_consumer():
    m = _two_reader_module()
    base = optimize(m, AVX, COOL).score
    res, certs = optimize_fused(m, AVX, COOL)
    assert certs == []                                        # illegal -> no fusion adopted
    assert res.score == base                                  # the base plan is untouched


def test_fusion_is_a_noop_on_softmax():
    m = matmul_activation(kind="softmax")
    base = optimize(m, AVX, COOL).score
    res, certs = optimize_fused(m, AVX, COOL)
    assert certs == []                                        # a row reduction is not an inline epilogue
    assert res.score == base


def test_fusion_is_a_noop_on_modules_with_no_epilogue():
    from bcir.examples import vector_add, fused_chain, scan_chain, matmul_tiled
    for mod in (vector_add(), fused_chain(), scan_chain(), matmul_tiled()):
        base = optimize(mod, AVX, COOL).score
        res, certs = optimize_fused(mod, AVX, COOL)
        assert certs == []
        assert res.score == base


def test_is_deterministic():
    a, ca = optimize_fused(matmul_activation(kind="gelu"), AVX, COOL)
    b, cb = optimize_fused(matmul_activation(kind="gelu"), AVX, COOL)
    assert a.score == b.score
    assert [c.gain for c in ca] == [c.gain for c in cb]


# --- the fused emitted kernel compiles + EQUALS the unfused reference -----------------------------

def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _run_fused_kernel(kind, M, N, K, a, b):
    """Compile the fused matmul+activation kernel + a tiny main that prints C[i], run it, return floats."""
    cc = _cc()
    fn = "fused"
    kernel = emit_matmul_activation_c(M, N, K, kind, fn)
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float A[{M * K}] = {{{', '.join(f'{x:.6f}f' for x in a)}}};\n"
            f"  float B[{K * N}] = {{{', '.join(f'{x:.6f}f' for x in b)}}};\n"
            f"  float C[{M * N}];\n  {fn}(A, B, C);\n"
            f'  for (int i=0;i<{M * N};++i) printf("%.7f ", C[i]);\n  return 0;\n}}\n')
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "f.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "f")
        # -lm: the trusted libm edge (expf/tanhf) for the transcendental epilogues, exactly as G1 links it.
        bld = subprocess.run(host_link_args([cc, "-std=c11", "-O2", src, "-lm", "-o", exe]),
                             capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        return [float(t) for t in out.stdout.split()]


def test_fused_relu_kernel_is_bit_exact_to_the_unfused_reference():
    cc = _cc()
    if not cc:
        return                                                # quick tier hides the toolchain -> self-skip
    rng = random.Random(0x6E20)
    M, N, K = 3, 4, 5
    a = [float(rng.randint(-5, 5)) for _ in range(M * K)]     # integer-valued -> matmul sums are exact
    b = [float(rng.randint(-5, 5)) for _ in range(K * N)]
    # the UNFUSED reference: matmul, THEN relu applied separately (the two-pass plan fusion replaces).
    unfused = activation_reference("relu", matmul_reference(a, b, M, N, K))
    got = _run_fused_kernel("relu", M, N, K, a, b)
    assert got == unfused                                     # 0 ULP -- the clean-rail guarantee carries over


def test_fused_transcendental_kernels_match_the_unfused_reference():
    cc = _cc()
    if not cc:
        return
    rng = random.Random(0x713D)
    M, N, K = 4, 3, 6
    a = [rng.uniform(-1.5, 1.5) for _ in range(M * K)]
    b = [rng.uniform(-1.5, 1.5) for _ in range(K * N)]
    prod = matmul_reference(a, b, M, N, K)
    for kind in ("sigmoid", "tanh", "gelu"):
        unfused = activation_reference(kind, prod)            # matmul then activation, separately
        got = _run_fused_kernel(kind, M, N, K, a, b)
        # the fused kernel calls the SAME libm; fused == unfused reference to float round-off only.
        maxerr = max(abs(g - r) for g, r in zip(got, unfused))
        assert maxerr < 1e-4, (kind, maxerr)


def test_fused_kernel_equals_the_two_pass_emitted_path():
    # The strongest correctness statement: the FUSED kernel output equals what the actual UNFUSED emitted
    # path produces -- emit_blas_gemm_c (matmul) into an intermediate, then emit_activation_kernel_c on it.
    # Same answer, one fewer buffer (the fusion's whole point).
    cc = _cc()
    if not cc:
        return
    rng = random.Random(0x2A55)
    M, N, K = 2, 3, 4
    a = [float(rng.randint(-4, 4)) for _ in range(M * K)]
    b = [float(rng.randint(-4, 4)) for _ in range(K * N)]
    fused = _run_fused_kernel("relu", M, N, K, a, b)
    # build the two-pass program: gemm into CPRE, then relu CPRE -> C.
    gemm = emit_blas_gemm_c(M, N, K, "gemm")
    act = emit_activation_kernel_c("relu", M * N, "act")
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float A[{M * K}] = {{{', '.join(f'{x:.1f}f' for x in a)}}};\n"
            f"  float B[{K * N}] = {{{', '.join(f'{x:.1f}f' for x in b)}}};\n"
            f"  float CPRE[{M * N}], C[{M * N}];\n"
            f"  gemm(A, B, CPRE);\n  act(CPRE, C, {M * N});\n"     # the materialized two-pass path
            f'  for (int i=0;i<{M * N};++i) printf("%.7f ", C[i]);\n  return 0;\n}}\n')
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "tp.c")
        open(src, "w").write(gemm + act + main)
        exe = os.path.join(d, "tp")
        bld = subprocess.run(host_link_args([cc, "-std=c11", "-O2", src, "-lm", "-o", exe]),
                             capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        two_pass = [float(t) for t in out.stdout.split()]
    assert fused == two_pass                                  # fused == the materialized two-pass, exactly


def test_emit_rejects_softmax_and_bad_dims():
    # softmax is non-fusible (a row reduction) -> the inline-epilogue emit rejects it.
    try:
        emit_matmul_activation_c(2, 2, 2, "softmax")
        assert False, "softmax should be rejected by the inline-epilogue emit"
    except ValueError:
        pass
    for bad in [(0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        try:
            emit_matmul_activation_c(*bad, "relu")
            assert False, f"{bad} should raise"
        except ValueError:
            pass
    # the emitted relu kernel carries NO libm (the clean rail); gelu DOES pull in math.h.
    assert "math.h" not in emit_matmul_activation_c(2, 2, 2, "relu")
    src_g = emit_matmul_activation_c(2, 2, 2, "gelu")
    assert "math.h" in src_g and "tanhf" in src_g
    # the fused kernel materializes NO intermediate: it writes C inline, no separate activation pass.
    assert "C[i * 2 + j] =" in emit_matmul_activation_c(2, 2, 2, "relu")
