"""G7 dual-rail parity: the MLIR law rail (gem.attention + -bcir-lower-gem-attention) reproduces the
oracle's bcir/kbcir/attention.py verdict -- the chosen two-matmul realization (the scores Q@K^T tile + the
context A@V tile), the roofline COST (the SUMMED compute/memory + the R17 accuracy bound), all PRICED
THROUGH THE EXISTING matmul roofline for BOTH matmuls (attention is a COMPOSITION of two gem.matmuls -- no
bespoke roofline term), and the softmax QUARANTINE (the expf libm edge, the SOLE transcendental; the
matmuls are exact). This is the dual-rail gate that matters: the law's plan == the oracle's plan_attention
for the same attention (the discipline is prototype-in-oracle -> port-to-MLIR-law, parity-gated; mirrors
how gem.matmul/gem.activation/gem.conv were ported -- test_conv_law_parity.py is the bcir-opt driver
template).

Two arms:
  * a PURE-PYTHON parity arm (always runs): for representative attention shapes it asserts the oracle's plan
    is internally consistent (bottleneck == max(compute, mem), the dual-semiring invariant the law verifier
    checks), that the attention is priced as the SUM of two matmul.cost_of costs (NOT a bespoke term), and
    that the two derived gemm dims are the decomposition (scores seq,seq,d_k; context seq,d_k,seq) -- the
    contract the MLIR op + verifier must mirror;
  * the ULTIMATE cross-check (self-skips without the built bcir-opt): drives the real
    `bcir-opt -bcir-lower-gem-attention` over that attention's oracle-pinned gem.attention IR and asserts the
    law lowers it clean, ERASES the plan, emits the decomposition (the two gem.matmul tile sequences + the
    softmax stripe), and annotates the SAME summed compute/mem/bottleneck/acc_bound + per-gemm costs + the
    softmax quarantine verdict the oracle computed. The oracle target is PINNED to x86-avx512 so the
    plan/cost are host-independent (CI-agnostic), exactly like the conv parity.
"""

from __future__ import annotations

import os
import subprocess

from bcir.kbcir import matmul
from bcir.kbcir.attention import cost_of, cost_vector, plan_attention
from bcir.kbcir.cost import ACCURACY, TargetProfile

# Pin the oracle target so the chosen tiles / cost are deterministic regardless of the CI host arch
# (TargetProfile.for_host() would price the roofline differently per arch -- the parity claim must not
# depend on which runner executes it). avx512 is the reference, exactly as test_conv_law_parity.
_T = TargetProfile.x86_avx512()

# (seq_len, d_k, quant_bits) -- a small dense attention and a (wider-head) quantized one (the R17 bound).
# Both single-head, single-sequence scaled-dot-product attentions, the G7 deliverable.
_CASES = [
    (8, 8, 0),  # a small dense attention (both gemms 8x8x8)
    (
        4,
        16,
        8,
    ),  # a wider head, QUANTIZED -> the R17 Q8-bridge bound; the asymmetric A@V gemm 4x16x4
]


def _oracle_plan(seq_len, d_k, quant_bits):
    """The oracle's planned attention realization: the two matmul tiles + the cost_of roofline (the SUM of
    the two priced matmul costs, via matmul.cost_of) + the cost_vector accuracy axis. Returns the dict of
    fields the law-rail op carries (so the two rails are compared field-by-field)."""
    spec = plan_attention(seq_len, d_k, target=_T)
    sm, sn, sk = spec.scores_dims
    cm, cn, ck = spec.context_dims
    st, ct = spec.scores_tile, spec.context_tile
    # each underlying matmul priced through the EXISTING matmul roofline (no bespoke term).
    sc, smm, _ = matmul.cost_of(sm, sn, sk, st.tile_m, st.tile_n, st.tile_k, _T)
    cc, cmm, _ = matmul.cost_of(cm, cn, ck, ct.tile_m, ct.tile_n, ct.tile_k, _T)
    compute, mem, _ = cost_of(spec, _T, st, ct)  # the SUM: c1+c2, m1+m2
    cv = cost_vector(spec, quant_bits=quant_bits)
    return {
        "seq_len": seq_len,
        "d_k": d_k,
        "dtype": "f32",
        "scores_m": sm,
        "scores_n": sn,
        "scores_k": sk,
        "context_m": cm,
        "context_n": cn,
        "context_k": ck,
        "scores_tile_m": st.tile_m,
        "scores_tile_n": st.tile_n,
        "scores_tile_k": st.tile_k,
        "scores_loop_order": st.loop_order,
        "context_tile_m": ct.tile_m,
        "context_tile_n": ct.tile_n,
        "context_tile_k": ct.tile_k,
        "context_loop_order": ct.loop_order,
        "scores_compute": sc,
        "scores_mem": smm,
        "context_compute": cc,
        "context_mem": cmm,
        "compute": compute,
        "mem": mem,
        "bottleneck": max(compute, mem),
        "quant_bits": quant_bits,
        "acc_bound": cv.v[ACCURACY],
    }


def _attn_mlir(p) -> str:
    """Emit the gem.attention plan record MLIR for an oracle plan dict (the law-rail op the oracle pins)."""
    return (
        "bcir.module @m {\n"
        f"  bcir.gem.attention @at {{ seq_len = {p['seq_len']} : i64, d_k = {p['d_k']} : i64, "
        f'dtype = "{p["dtype"]}", scores_m = {p["scores_m"]} : i64, scores_n = {p["scores_n"]} : i64, '
        f"scores_k = {p['scores_k']} : i64, context_m = {p['context_m']} : i64, "
        f"context_n = {p['context_n']} : i64, context_k = {p['context_k']} : i64, "
        f"scores_tile_m = {p['scores_tile_m']} : i64, scores_tile_n = {p['scores_tile_n']} : i64, "
        f'scores_tile_k = {p["scores_tile_k"]} : i64, scores_loop_order = "{p["scores_loop_order"]}", '
        f"context_tile_m = {p['context_tile_m']} : i64, context_tile_n = {p['context_tile_n']} : i64, "
        f'context_tile_k = {p["context_tile_k"]} : i64, context_loop_order = "{p["context_loop_order"]}", '
        f"scores_compute = {p['scores_compute']} : i64, scores_mem = {p['scores_mem']} : i64, "
        f"context_compute = {p['context_compute']} : i64, context_mem = {p['context_mem']} : i64, "
        f"compute_cost = {p['compute']} : i64, mem_cost = {p['mem']} : i64, "
        f"bottleneck = {p['bottleneck']} : i64, quant_bits = {p['quant_bits']} : i32, "
        f"acc_bound = {p['acc_bound']} : i64 }}\n"
        "}\n"
    )


def _ceil_div(a, b):
    return (a + b - 1) // b


# --- the pure-Python parity arm (always runs) -----------------------------------------------------


def test_oracle_attention_plan_is_self_consistent_and_priced_as_two_matmuls():
    """The oracle's attention plan is internally consistent (bottleneck == max(compute, mem), the dual-
    semiring invariant the law verifier checks), its two gemm dims ARE the decomposition (scores Q@K^T =
    seq,seq,d_k; context A@V = seq,d_k,seq), and -- the load-bearing G7 property -- its (compute, mem) is
    EXACTLY the SUM of the two matmul.cost_of costs (attention is a COMPOSITION of two gem.matmuls; NO
    bespoke roofline term). This is the contract the MLIR op + verifier mirror."""
    for case in _CASES:
        p = _oracle_plan(*case)
        assert p["bottleneck"] == max(p["compute"], p["mem"]), f"{case}: bottleneck != max"
        # the two derived gemm dims are the decomposition.
        assert (p["scores_m"], p["scores_n"], p["scores_k"]) == (
            p["seq_len"],
            p["seq_len"],
            p["d_k"],
        )
        assert (p["context_m"], p["context_n"], p["context_k"]) == (
            p["seq_len"],
            p["d_k"],
            p["seq_len"],
        )
        # the SUMMED roofline == the two priced matmuls (no bespoke term).
        assert p["compute"] == p["scores_compute"] + p["context_compute"], f"{case}: compute != sum"
        assert p["mem"] == p["scores_mem"] + p["context_mem"], f"{case}: mem != sum"
        # each per-gemm cost == matmul.cost_of at its chosen tile.
        sc, smm, _ = matmul.cost_of(
            p["scores_m"],
            p["scores_n"],
            p["scores_k"],
            p["scores_tile_m"],
            p["scores_tile_n"],
            p["scores_tile_k"],
            _T,
        )
        cc, cmm, _ = matmul.cost_of(
            p["context_m"],
            p["context_n"],
            p["context_k"],
            p["context_tile_m"],
            p["context_tile_n"],
            p["context_tile_k"],
            _T,
        )
        assert (p["scores_compute"], p["scores_mem"]) == (sc, smm), (
            f"{case}: scores cost != matmul roofline"
        )
        assert (p["context_compute"], p["context_mem"]) == (cc, cmm), (
            f"{case}: context cost != matmul roofline"
        )
        # the R17 accuracy axis: the Q8-bridge 1-ULP bound iff quantized, else 0 (matmuls exact, libm trusted).
        assert p["acc_bound"] == (1 if p["quant_bits"] > 0 else 0), (
            f"{case}: R17 acc bound mismatch"
        )


# --- the ultimate cross-check: the real bcir-opt law rail (self-skips without the built tool) -----


def test_law_rail_reproduces_oracle_attention_plan_via_bcir_opt():
    """Drive `bcir-opt -bcir-lower-gem-attention` over the oracle-pinned gem.attention IR and assert the law
    reproduces the oracle's verdict: the plan is ERASED, the decomposition (the two gem.matmul tile
    sequences + the softmax stripe) is emitted, and the recomputed kbcir.compute_cost/mem_cost/bottleneck/
    acc_bound + the per-gemm costs + the softmax quarantine verdict (kbcir.lane_kind transcendental,
    kbcir.libm_edge expf) EQUAL the oracle's plan_attention/cost_of/cost_vector. The plan/cost are
    host-independent because the oracle target is pinned to avx512."""
    bo = _find_bcir_opt()
    if not bo:
        return  # the MLIR toolchain is not built in this environment (the oracle-only CI job)
    for case in _CASES:
        p = _oracle_plan(*case)
        proc = subprocess.run(
            [bo, "-bcir-lower-gem-attention"], input=_attn_mlir(p), capture_output=True, text=True
        )
        assert proc.returncode == 0, f"{case}: law rail rejected the oracle plan:\n{proc.stderr}"
        out = proc.stdout
        # genuine lowering: the plan record is consumed.
        assert "bcir.gem.attention" not in out, f"{case}: plan record not erased by the lowering"
        # the realization: the scores Q@K^T gemm tiled -> ceil(M/tm)*ceil(N/tn)*ceil(K/tk) tile blocks @at_s_t*,
        # then the softmax stripe @at_sm_s0, then the context A@V gemm tiles @at_c_t*.
        n_scores = (
            _ceil_div(p["scores_m"], p["scores_tile_m"])
            * _ceil_div(p["scores_n"], p["scores_tile_n"])
            * _ceil_div(p["scores_k"], p["scores_tile_k"])
        )
        n_context = (
            _ceil_div(p["context_m"], p["context_tile_m"])
            * _ceil_div(p["context_n"], p["context_tile_n"])
            * _ceil_div(p["context_k"], p["context_tile_k"])
        )
        for t in range(n_scores):
            assert f"@at_s_t{t} " in out, f"{case}: missing scores tile @at_s_t{t}"
        assert f"@at_s_t{n_scores} " not in out, (
            f"{case}: emitted too many scores tiles (> {n_scores})"
        )
        assert "@at_sm_s0 " in out, f"{case}: missing softmax stripe @at_sm_s0"
        for t in range(n_context):
            assert f"@at_c_t{t} " in out, f"{case}: missing context tile @at_c_t{t}"
        assert f"@at_c_t{n_context} " not in out, (
            f"{case}: emitted too many context tiles (> {n_context})"
        )
        # the dual-rail parity: the law's recomputed plan EQUALS the oracle's, field by field, on @at_s_t0.
        t0 = next(l for l in out.splitlines() if "@at_s_t0 " in l)
        assert f"kbcir.seq_len = {p['seq_len']} : i64" in t0, f"{case}: seq_len != oracle\n{t0}"
        assert f"kbcir.d_k = {p['d_k']} : i64" in t0, f"{case}: d_k != oracle\n{t0}"
        assert f"kbcir.scores_compute = {p['scores_compute']} : i64" in t0, (
            f"{case}: scores_compute != oracle\n{t0}"
        )
        assert f"kbcir.scores_mem = {p['scores_mem']} : i64" in t0, (
            f"{case}: scores_mem != oracle\n{t0}"
        )
        assert f"kbcir.context_compute = {p['context_compute']} : i64" in t0, (
            f"{case}: context_compute != oracle\n{t0}"
        )
        assert f"kbcir.context_mem = {p['context_mem']} : i64" in t0, (
            f"{case}: context_mem != oracle\n{t0}"
        )
        assert f"kbcir.compute_cost = {p['compute']} : i64" in t0, (
            f"{case}: compute (sum) != oracle\n{t0}"
        )
        assert f"kbcir.mem_cost = {p['mem']} : i64" in t0, f"{case}: mem (sum) != oracle\n{t0}"
        assert f"kbcir.bottleneck = {p['bottleneck']} : i64" in t0, (
            f"{case}: bottleneck != oracle\n{t0}"
        )
        assert f"kbcir.acc_bound = {p['acc_bound']} : i64" in t0, f"{case}: R17 acc != oracle\n{t0}"
        assert f'kbcir.scores_loop_order = "{p["scores_loop_order"]}"' in t0, (
            f"{case}: scores loop_order != oracle\n{t0}"
        )
        assert f'kbcir.context_loop_order = "{p["context_loop_order"]}"' in t0, (
            f"{case}: context loop_order != oracle\n{t0}"
        )
        # the softmax quarantine verdict on the stripe: the SOLE transcendental, expf via the libm edge.
        sm = next(l for l in out.splitlines() if "@at_sm_s0 " in l)
        assert 'kbcir.lane_kind = "transcendental"' in sm, (
            f"{case}: softmax quarantine side != oracle\n{sm}"
        )
        assert 'kbcir.libm_edge = "expf"' in sm, f"{case}: softmax libm edge != oracle\n{sm}"


def test_law_rail_rejects_an_inconsistent_attention_plan():
    """The headline op-verifier gate, end-to-end on the law rail: a gem.attention whose declared scores
    Q@K^T gemm dims do not match its (seq_len, d_k) is REJECTED by the verifier (the two-matmul
    decomposition consistency the oracle's check_attention enforces). Self-skips without bcir-opt."""
    bo = _find_bcir_opt()
    if not bo:
        return
    bad = (
        "bcir.module @m {\n"
        '  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32", '
        "scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 7 : i64, "
        "context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64, "
        'scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 7 : i64, scores_loop_order = "ijk", '
        'context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk", '
        "scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64, "
        "compute_cost = 32 : i64, mem_cost = 128 : i64, bottleneck = 128 : i64, quant_bits = 0 : i32, "
        "acc_bound = 0 : i64 }\n}\n"
    )
    proc = subprocess.run(
        [bo, "-bcir-lower-gem-attention"], input=bad, capture_output=True, text=True
    )
    assert proc.returncode != 0, "the law rail must reject an inconsistent scores gemm dim"
    assert "declared scores Q@K^T gemm dims" in proc.stderr, (
        f"the rejection must cite the two-matmul decomposition consistency, got:\n{proc.stderr}"
    )


# --- the -bcir-gem-attention-cost cost-producer parity arm (mirrors test_conv_law_parity) ---------
# The SIBLING cost pass to the lowering above: -bcir-gem-attention-cost RECOMPUTES the attention
# roofline (port of attention.py::cost_of -- attention is a COMPOSITION of two gem.matmuls priced
# through matmul.cost_of and SUMMED, NO bespoke term) and parity-checks the declared per-gemm +
# summed compute/mem/bottleneck, annotating kbcir.* INFORMS-ONLY -- a cost PRODUCER (the op is NOT
# erased), not a lowering. This is the EXACT analog of -bcir-gem-matmul-cost / -bcir-gem-activation-
# cost / -bcir-gem-conv-cost (their *_law_parity.py); the four cost passes are a tight family. The
# same oracle-pinned IR (_attn_mlir / _oracle_plan) drives both the lowering rail and this cost rail.


def test_law_rail_reproduces_oracle_attention_roofline_via_bcir_opt():
    """Drive `bcir-opt -bcir-gem-attention-cost` over the oracle-pinned gem.attention IR and assert the
    law recomputes the SAME per-gemm (scores/context) + summed compute/mem/bottleneck the oracle's
    cost_of computed (attention.py, == matmul.cost_of on each gemm, SUMMED), and annotates the
    INFORMS-ONLY quarantine record (kbcir.informs_only = true). The recomputed roofline is host-
    independent because the oracle target is pinned to avx512 (the C++ pass pins the same x86-avx512
    reference constants). A cost producer, NOT a lowering: the carrier op is NOT erased (cf.
    -bcir-lower-gem-attention above, which ERASES the plan into its two gemm tile seqs + softmax)."""
    bo = _find_bcir_opt()
    if not bo:
        return  # the MLIR toolchain is not built in this environment (the oracle-only CI job)
    for case in _CASES:
        p = _oracle_plan(*case)
        proc = subprocess.run(
            [bo, "-bcir-gem-attention-cost"], input=_attn_mlir(p), capture_output=True, text=True
        )
        assert proc.returncode == 0, (
            f"{case}: cost rail rejected the oracle roofline:\n{proc.stderr}"
        )
        out = proc.stdout
        line = next(l for l in out.splitlines() if "@at " in l)
        # the SUMMED roofline (compute = the two computes summed; mem = the two mems summed).
        assert f"kbcir.compute_cost = {p['compute']} : i64" in line, (
            f"{case}: compute (sum) != oracle\n{line}"
        )
        assert f"kbcir.mem_cost = {p['mem']} : i64" in line, f"{case}: mem (sum) != oracle\n{line}"
        assert f"kbcir.bottleneck = {p['bottleneck']} : i64" in line, (
            f"{case}: bottleneck != oracle\n{line}"
        )
        # the PER-GEMM priced matmul costs (each == matmul.cost_of at its chosen tile).
        assert f"kbcir.scores_compute = {p['scores_compute']} : i64" in line, (
            f"{case}: scores_compute != oracle\n{line}"
        )
        assert f"kbcir.scores_mem = {p['scores_mem']} : i64" in line, (
            f"{case}: scores_mem != oracle\n{line}"
        )
        assert f"kbcir.context_compute = {p['context_compute']} : i64" in line, (
            f"{case}: context_compute != oracle\n{line}"
        )
        assert f"kbcir.context_mem = {p['context_mem']} : i64" in line, (
            f"{case}: context_mem != oracle\n{line}"
        )
        # the INFORMS-ONLY quarantine record: the recomputed roofline informs the plan, never gates legality.
        assert "kbcir.informs_only = true" in line, (
            f"{case}: missing the informs-only quarantine flag\n{line}"
        )
        # a cost producer, NOT a lowering: the carrier op is NOT erased.
        assert "bcir.gem.attention" in out, (
            f"{case}: the attention op must NOT be erased (it is a producer)"
        )


def test_cost_rail_rejects_an_inconsistent_attention_roofline():
    """The headline parity gate, end-to-end on the cost rail: a gem.attention whose declared
    scores_compute is NOT the recomputed per-gemm roofline (the analytic cost the op verifier
    deliberately omits, but the declared SUM kept consistent with the per-gemm terms keeps it op-level
    well-formed) is REJECTED by -bcir-gem-attention-cost (the dual-rail R13 parity the oracle's cost_of
    enforces). Self-skips without bcir-opt."""
    bo = _find_bcir_opt()
    if not bo:
        return
    p = _oracle_plan(*_CASES[0])  # the small dense attention (both gemms 8x8x8)
    assert p["scores_compute"] == 16  # sanity: the oracle's recomputed scores compute (gemm 8x8x8)
    bad = dict(p)
    bad["scores_compute"] = 9999  # a wrong scores per-gemm compute term...
    # ...keep the SUM + bottleneck op-level consistent so the OP verifier still admits it (the cost pass
    # catches the per-gemm term before the sum): compute_cost = 9999 + context_compute; bottleneck = max.
    bad["compute"] = 9999 + p["context_compute"]
    bad["bottleneck"] = max(bad["compute"], p["mem"])
    proc = subprocess.run(
        [bo, "-bcir-gem-attention-cost"], input=_attn_mlir(bad), capture_output=True, text=True
    )
    assert proc.returncode != 0, "the cost rail must reject a non-cost_of per-gemm compute term"
    assert (
        "scores_compute 9999 != the recomputed scores Q@K^T roofline compute 16" in proc.stderr
    ), f"the rejection must cite the recomputed per-gemm roofline compute, got:\n{proc.stderr}"


def _find_bcir_opt():
    env = os.environ.get("BCIR_OPT")
    if env and os.path.exists(env):
        return env
    root = os.path.join(os.path.dirname(__file__), "..", "..", "build", "mlir-build")
    for dirpath, _dirs, files in os.walk(os.path.normpath(root)) if os.path.isdir(root) else []:
        if "bcir-opt" in files:
            return os.path.join(dirpath, "bcir-opt")
    return None
