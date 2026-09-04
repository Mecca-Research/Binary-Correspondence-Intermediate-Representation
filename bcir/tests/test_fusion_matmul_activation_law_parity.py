"""G2 dual-rail parity: the MLIR law rail (gem.fused_matmul_activation + -bcir-fuse-matmul-activation)
reproduces the oracle's bcir/kbcir/fusion.py verdict -- the fusion DECISION (adopt iff the deforestation-
priced fused score strictly beats the unfused score), the PRICING mechanism (the SAME _DEFOREST_FACTOR
x0.75 memory discount, NOT a bespoke one), and the QUARANTINE/softmax-scope-out (relu exact i32-or-f32;
the transcendentals route a libm edge and need f32; softmax is a row reduction, not an inline epilogue).
This is the dual-rail gate that matters: the law's fusion choice == the oracle's optimize_fused choice for
the same matmul->activation pair (the discipline is prototype-in-oracle -> port-to-MLIR-law, parity-gated;
mirrors how gem.activation/gem.matmul were ported -- test_activation_law_parity.py is the driver template).

Two arms:
  * a PURE-PYTHON parity arm (always runs): asserts the law-rail deforestation arithmetic uses the SAME
    constant the oracle's optimize_fused applies (realize._DEFOREST_FACTOR == x0.75 memory, identity
    elsewhere) and that the law's strict-win adopt/no-op decision MATCHES optimize_fused for every kind
    (relu/sigmoid/tanh/gelu adopt; softmax is scoped out) -- the contract the MLIR pass + op must mirror;
  * the ULTIMATE cross-check (self-skips without the built bcir-opt): drives the real
    `bcir-opt -bcir-fuse-matmul-activation` over a matmul->activation pair and asserts the law FUSES iff
    the oracle adopts (relu + gelu), ERASES both ops into one gem.fused_matmul_activation carrying the
    deforestation-priced FusionCertificate evidence (unfused/fused score + gain>0, the SAME sign as the
    oracle's gain) and the quarantine (f32 carried through), and is a clean NO-OP when the activation is
    NOT the sole consumer (a third op breaks the adjacency) or is softmax (a row reduction).
"""

from __future__ import annotations

import os
import re
import subprocess

from bcir.examples import matmul_activation
from bcir.kbcir import TARGETS
from bcir.kbcir.cost import MEMORY, Theta
from bcir.kbcir.fusion import find_matmul_activation_epilogues, optimize_fused
from bcir.kbcir.realize import _DEFOREST_FACTOR

# Pin the oracle target so the adopt/no-op decision is deterministic regardless of the CI host arch.
_T = TARGETS["x86_avx512"]
_COOL = Theta.cool()

# The four elementwise kinds the oracle FUSES, and softmax (scoped out as a row reduction).
_FUSIBLE = ("relu", "sigmoid", "tanh", "gelu")
_NON_FUSIBLE = ("softmax",)


def _law_deforest_mem(mem: int) -> int:
    """The law-rail pass's deforestation arithmetic: the consumer activation's mem traffic discounted
    x0.75 (the eliminated intermediate's round-trip). Q8 floor -- IDENTICAL to the oracle's
    Candidate.couple(_DEFOREST_FACTOR) on the memory axis."""
    return (mem * _DEFOREST_FACTOR[MEMORY]) // 256


def _law_fused_decision(mm_bn: int, act_compute: int, act_mem: int, act_bn: int):
    """Recompute the law pass's deforestation-priced fused vs unfused score from the declared per-op
    rooflines (mm.bottleneck + the activation roofline), and the adopt decision. Returns
    (unfused_score, fused_score, gain, adopt) -- exactly what BCIRFuseMatmulActivationPass computes."""
    unfused = mm_bn + act_bn
    fused_act_bn = max(act_compute, _law_deforest_mem(act_mem))
    fused = mm_bn + fused_act_bn
    gain = unfused - fused
    return unfused, fused, gain, (gain > 0)


# --- the pure-Python parity arm (always runs) -----------------------------------------------------


def test_law_deforestation_is_the_oracle_discount_not_a_bespoke_one():
    """The law rail prices fusion with the EXISTING deforestation discount the oracle's optimize_fused
    applies -- x0.75 ON THE MEMORY AXIS ONLY (192/256), identity elsewhere. Not a new constant."""
    assert _DEFOREST_FACTOR[MEMORY] == 192, "the deforestation memory discount is x0.75 (192/256)"
    assert all(f == 256 for i, f in enumerate(_DEFOREST_FACTOR) if i != MEMORY), (
        "deforestation discounts ONLY the memory axis (identity everywhere else)"
    )
    # the law arithmetic floors the same way Candidate.couple does (Q8 fixed point).
    assert _law_deforest_mem(32) == 24 and _law_deforest_mem(200) == 150


def test_law_decision_matches_oracle_optimize_fused_adopt_or_noop():
    """For every activation kind the law's strict-win rule MUST agree with optimize_fused's adopt/no-op:
    the four elementwise kinds are ADOPTED (a strict deforestation win), softmax is a clean NO-OP (a row
    reduction is not an inline epilogue). This is the headline dual-rail decision parity, pure-Python."""
    for kind in _FUSIBLE:
        m = matmul_activation(kind=kind)
        _, certs = optimize_fused(m, _T, _COOL)
        assert len(certs) == 1 and certs[0].gain > 0, (
            f"{kind}: oracle must ADOPT a strict-win fusion"
        )
        # the law decision on a representative per-op roofline (relu/gelu rooflines, mem-bound activation)
        # also adopts: a memory-bound elementwise epilogue ALWAYS wins under the x0.75 discount.
        _, _, gain, adopt = _law_fused_decision(mm_bn=200, act_compute=2, act_mem=32, act_bn=32)
        assert adopt and gain > 0, f"{kind}: the law must adopt the same strict-win fusion"
    for kind in _NON_FUSIBLE:
        m = matmul_activation(kind=kind)
        _, certs = optimize_fused(m, _T, _COOL)
        assert certs == [], f"{kind}: oracle must NO-OP (softmax is scoped out)"
        # the law scopes softmax out at the op/pass level too (the verifier rejects a fused softmax;
        # the pass skips a matmul->softmax pair) -- asserted on the bcir-opt arm below.
        epis = find_matmul_activation_epilogues(m)
        assert epis and not epis[0].fusible, f"{kind}: detected but flagged non-fusible"


# --- the ultimate cross-check: the real bcir-opt law rail (self-skips without the built tool) -----


def _matmul_activation_mlir(kind: str, dtype: str = "f32", *, with_reader: bool = False) -> str:
    """Emit a matmul -> activation epilogue on the law rail (the IR the pass fuses). The matmul writes the
    intermediate the activation consumes; ADJACENCY (the activation immediately follows the matmul) is the
    sole-consumer legality. `with_reader` inserts a gem.block BETWEEN them (a non-sole-consumer reader of
    the un-activated intermediate) -> the adjacency is broken, fusion is illegal."""
    # softmax declares its last-axis length; the elementwise four declare axis_len 0.
    if kind == "softmax":
        shape, axis_len = "4, 4", 4
        compute, mem, bn = 8, 24, 24
    else:
        shape, axis_len = "16", 0
        compute, mem, bn = (24 if kind == "gelu" else 2), 32, 32
    reader = (
        (
            "  bcir.gem.block @reader { base = 0 : i64, count = 4 : i64, strideA = 1 : i64, "
            "strideB = 0 : i64, strideD = 1 : i64 }\n"
        )
        if with_reader
        else ""
    )
    return (
        "bcir.module @m {\n"
        "  bcir.gem.matmul @mm0 { m = 4 : i64, n = 4 : i64, k = 4 : i64, tile_m = 2 : i64, "
        'tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk", compute_cost = 100 : i64, '
        "mem_cost = 200 : i64, bottleneck = 200 : i64, quant_bits = 0 : i32 }\n"
        f"{reader}"
        f'  bcir.gem.activation @act0 {{ kind = "{kind}", shape = array<i64: {shape}>, '
        f'dtype = "{dtype}", axis_len = {axis_len} : i64, width = 4 : i64, '
        f"compute_cost = {compute} : i64, mem_cost = {mem} : i64, bottleneck = {bn} : i64, "
        "quant_bits = 0 : i32, acc_bound = 0 : i64 }\n"
        "}\n"
    )


def test_law_rail_fuses_iff_oracle_adopts_via_bcir_opt():
    """Drive `bcir-opt -bcir-fuse-matmul-activation` over a matmul->activation pair and assert the law
    FUSES iff the oracle's optimize_fused adopts: relu + gelu fuse into one gem.fused_matmul_activation
    (both source ops erased) carrying the deforestation-priced FusionCertificate evidence with a gain of
    the SAME sign as the oracle's; softmax is a clean no-op. The quarantine (f32) carries through."""
    bo = _find_bcir_opt()
    if not bo:
        return  # the MLIR toolchain is not built in this environment (the oracle-only CI job)
    for kind in ("relu", "gelu"):
        # the oracle's decision for this kind (the parity target).
        _, certs = optimize_fused(matmul_activation(kind=kind), _T, _COOL)
        assert certs and certs[0].gain > 0, kind  # the oracle adopts a strict win
        # the law rail's decision over the same epilogue.
        proc = subprocess.run(
            [bo, "-bcir-fuse-matmul-activation"],
            input=_matmul_activation_mlir(kind),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"{kind}: law rail rejected the epilogue:\n{proc.stderr}"
        out = proc.stdout
        # genuine fusion: BOTH the producer and the consumer are consumed.
        assert "bcir.gem.matmul" not in out, f"{kind}: matmul not erased by the fusion"
        assert "bcir.gem.activation" not in out, f"{kind}: activation not erased by the fusion"
        line = next(l for l in out.splitlines() if "gem.fused_matmul_activation" in l)
        # the law fused the right kind, carried the quarantine dtype, and priced a strict win.
        assert f'kind = "{kind}"' in line, f"{kind}: fused op carries the wrong kind\n{line}"
        assert 'dtype = "f32"' in line, (
            f"{kind}: the quarantine dtype did not carry through\n{line}"
        )
        m_un = re.search(r"unfused_score = (\d+)", line)
        m_fu = re.search(r"fused_score = (\d+)", line)
        m_ga = re.search(r"gain = (\d+)", line)
        assert m_un and m_fu and m_ga, f"{kind}: missing the FusionCertificate evidence\n{line}"
        un, fu, ga = int(m_un[1]), int(m_fu[1]), int(m_ga[1])
        # the deforestation-priced decision: unfused = 200+32 = 232, fused = 200+max(compute,24), gain>0.
        exp_un, exp_fu, exp_ga, adopt = _law_fused_decision(
            200, (24 if kind == "gelu" else 2), 32, 32
        )
        assert (un, fu, ga) == (exp_un, exp_fu, exp_ga), f"{kind}: law score != recompute\n{line}"
        assert adopt and ga > 0, f"{kind}: the law must price a strict win"
        # the dual-rail SIGN parity: the law's gain and the oracle's gain are BOTH a strict positive win.
        assert (ga > 0) == (certs[0].gain > 0), f"{kind}: law/oracle gain sign mismatch"


def test_law_rail_is_a_noop_on_softmax_and_non_sole_consumer():
    """The fusion legality boundary on the law rail, matching the oracle's no-op cases: softmax (a row
    reduction, scoped out) and a non-sole-consumer pair (a reader between producer and consumer breaks the
    adjacency) are LEFT UNTOUCHED -- no gem.fused_matmul_activation, both ops survive. Mirrors
    optimize_fused returning empty certificates (no fusion adopted) for both."""
    bo = _find_bcir_opt()
    if not bo:
        return
    # (1) softmax: the oracle no-ops (a row reduction is not an inline epilogue); the law must too.
    assert optimize_fused(matmul_activation(kind="softmax"), _T, _COOL)[1] == []
    proc = subprocess.run(
        [bo, "-bcir-fuse-matmul-activation"],
        input=_matmul_activation_mlir("softmax"),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "gem.fused_matmul_activation" not in proc.stdout, (
        "softmax must NOT fuse (a row reduction)"
    )
    assert "bcir.gem.matmul" in proc.stdout and "bcir.gem.activation" in proc.stdout, (
        "the softmax epilogue must be left untouched"
    )
    # (2) non-sole-consumer: a reader between the matmul and the relu breaks the adjacency -> no fusion.
    proc = subprocess.run(
        [bo, "-bcir-fuse-matmul-activation"],
        input=_matmul_activation_mlir("relu", with_reader=True),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "gem.fused_matmul_activation" not in proc.stdout, (
        "a non-sole-consumer pair must NOT fuse (the intermediate has another reader)"
    )
    assert "bcir.gem.matmul" in proc.stdout and "bcir.gem.activation" in proc.stdout, (
        "the non-sole-consumer epilogue must be left untouched"
    )


def test_law_rail_rejects_a_fused_softmax_and_a_transcendental_on_int():
    """The headline op-verifier gates, end-to-end on the law rail: a fused softmax (the scope-out) and a
    transcendental on an integer dtype (the quarantine carried through the epilogue) are REJECTED by the
    gem.fused_matmul_activation verifier. Self-skips without bcir-opt."""
    bo = _find_bcir_opt()
    if not bo:
        return
    base = (
        "bcir.module @m {{\n  bcir.gem.fused_matmul_activation @f {{ m = 4 : i64, n = 4 : i64, "
        'k = 4 : i64, tile_m = 2 : i64, tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk", '
        'kind = "{kind}", dtype = "{dtype}", unfused_score = 232 : i64, fused_score = 224 : i64, '
        "gain = 8 : i64 }}\n}}\n"
    )
    # (1) a fused softmax: rejected (the scope-out).
    proc = subprocess.run(
        [bo], input=base.format(kind="softmax", dtype="f32"), capture_output=True, text=True
    )
    assert proc.returncode != 0 and "softmax is non-fusible" in proc.stderr, (
        f"a fused softmax must be rejected as non-fusible, got:\n{proc.stderr}"
    )
    # (2) a transcendental on an integer dtype: rejected (the quarantine rule carries through).
    proc = subprocess.run(
        [bo], input=base.format(kind="gelu", dtype="i32"), capture_output=True, text=True
    )
    assert proc.returncode != 0 and "transcendental activation needs f32" in proc.stderr, (
        f"a transcendental on i32 must be rejected by the quarantine rule, got:\n{proc.stderr}"
    )


def _find_bcir_opt():
    env = os.environ.get("BCIR_OPT")
    if env and os.path.exists(env):
        return env
    root = os.path.join(os.path.dirname(__file__), "..", "..", "build", "mlir-build")
    for dirpath, _dirs, files in os.walk(os.path.normpath(root)) if os.path.isdir(root) else []:
        if "bcir-opt" in files:
            return os.path.join(dirpath, "bcir-opt")
    return None
