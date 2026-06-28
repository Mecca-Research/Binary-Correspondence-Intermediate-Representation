"""G4 dual-rail parity: the MLIR law rail (gem.contention + -bcir-cache-contention) reproduces the oracle's
bcir/kbcir/cache_predict.py CONTENTION signal -- the cache-line-utilization WASTE + the bank-conflict
serialization degree (frozen Q8) and the integer CONTENTION cost (the per-element Q8 scaled by count), all
INFORMS-ONLY (the signal rides the CONTENTION cost axis and NEVER gates legality -- the two-truth quarantine).
This is the dual-rail gate that matters: the law's contention == cache_predict.contention_q8 for the same
access shape (the discipline is prototype-in-oracle -> port-to-MLIR-law, parity-gated; mirrors how
gem.matmul/activation/conv/attention were ported -- test_attention_law_parity.py is the bcir-opt driver
template).

Two arms:
  * a PURE-PYTHON parity arm (always runs): for representative strides (a unit sweep, a bank-colliding
    power-of-2, a strided gather) it asserts the oracle's frozen analytic model is self-consistent
    (line_waste_q8 = (min(stride, cacheline/elem)-1)*256, bank_conflict_q8 = (gcd(stride, banks)-1)*256,
    contention_q8 = the sum, contention_cost = (contention_q8*count)>>8), AND -- the load-bearing G4
    quarantine -- that the CONTENTION axis is the only one the signal moves and it never gates a legality
    verdict (verify_plan reads only R8/R9, never CONTENTION);
  * the ULTIMATE cross-check (self-skips without the built bcir-opt): drives the real
    `bcir-opt -bcir-cache-contention` over the oracle-pinned gem.contention IR and asserts the law recomputes
    the SAME line_waste_q8/bank_conflict_q8/contention_q8/contention_cost the oracle computed AND annotates
    the informs-only quarantine flag (kbcir.informs_only = true, kbcir.contention_axis = "CONTENTION").
"""

from __future__ import annotations

import os
import subprocess

from bcir.kbcir.cache_predict import AccessShape, CachePredictor

# The frozen analytic predictor (gen 0 seeded constants): a 64-byte cache line over 4 interleaved banks.
# Host-independent (the parity claim must not depend on the CI runner's TargetProfile.for_host()).
_PRED = CachePredictor(cacheline=64, banks=4)

# (stride, elem_bytes, count) -- the three representative access shapes the task names:
#   * a UNIT sweep (stride 1): full cache-line utilization, all banks reached -> the clean NO-OP (cost 0).
#   * a bank-colliding power-of-2 (stride 4 == #banks): one bank reached -> maximal serialization.
#   * a strided gather (stride 16 == cacheline/elem -> a fresh line per element): the worst-case waste.
_CASES = [
    (1, 4, 64),     # unit sweep: waste 0, conflict 0, cost 0 (a contiguous access has 0 contention)
    (4, 4, 64),     # bank-colliding stride (== banks): gcd(4,4)=4 -> conflict (4-1)*256; reach min(4,16)=4
    (16, 4, 64),    # strided gather (stride == cacheline/elem 16): reach 16 -> max waste; gcd(16,4)=4 conflict
]


def _oracle_signal(stride, elem_bytes, count):
    """The oracle's CONTENTION signal for one access shape: the Q8 waste/conflict terms + the integer cost.
    Returns the dict of fields the law-rail op carries (so the two rails are compared field-by-field)."""
    s = AccessShape(stride=stride, elem_bytes=elem_bytes, count=count)
    return {
        "stride": stride, "elem_bytes": elem_bytes, "count": count,
        "cacheline": _PRED.cacheline, "banks": _PRED.banks,
        "line_waste_q8": _PRED.line_waste_q8(s),
        "bank_conflict_q8": _PRED.bank_conflict_q8(s),
        "contention_q8": _PRED.contention_q8(s),
        "contention_cost": _PRED.contention_cost(s),
    }


def _contention_mlir(p) -> str:
    """Emit the gem.contention carrier-op MLIR for an oracle signal dict (the law-rail op the oracle pins)."""
    return (
        'bcir.module @m {\n'
        f'  bcir.gem.contention @c {{ stride = {p["stride"]} : i64, elem_bytes = {p["elem_bytes"]} : i64, '
        f'count = {p["count"]} : i64, cacheline = {p["cacheline"]} : i64, banks = {p["banks"]} : i64, '
        f'line_waste_q8 = {p["line_waste_q8"]} : i64, bank_conflict_q8 = {p["bank_conflict_q8"]} : i64, '
        f'contention_q8 = {p["contention_q8"]} : i64, contention_cost = {p["contention_cost"]} : i64 }}\n'
        '}\n'
    )


# --- the pure-Python parity arm (always runs) -----------------------------------------------------

def test_oracle_contention_signal_is_self_consistent_and_quarantined():
    """The oracle's CONTENTION signal is internally consistent with the frozen analytic model
    (contention_q8 == waste + conflict; contention_cost == (contention_q8*count)>>8), a unit sweep is the
    clean NO-OP (cost 0), and -- the load-bearing G4 quarantine -- it touches ONLY the CONTENTION axis (every
    other axis is 0), so adding it can never perturb a legality verdict. This is the contract the MLIR op +
    verifier mirror."""
    from math import gcd

    from bcir.kbcir.cost import CONTENTION, N
    for stride, eb, count in _CASES:
        p = _oracle_signal(stride, eb, count)
        epl = max(1, p["cacheline"] // eb)
        assert p["line_waste_q8"] == (min(max(1, stride), epl) - 1) * 256, f"{stride}: waste model"
        assert p["bank_conflict_q8"] == (gcd(max(1, stride), p["banks"]) - 1) * 256, f"{stride}: conflict model"
        assert p["contention_q8"] == p["line_waste_q8"] + p["bank_conflict_q8"], f"{stride}: q8 sum"
        assert p["contention_cost"] == (p["contention_q8"] * count) >> 8, f"{stride}: cost scale"
    # the unit sweep is the clean no-op: 0 waste, 0 conflict, 0 cost.
    u = _oracle_signal(1, 4, 64)
    assert (u["line_waste_q8"], u["bank_conflict_q8"], u["contention_cost"]) == (0, 0, 0)
    # the QUARANTINE: the cost vector touches ONLY the CONTENTION axis (every other dim is 0).
    cv = CachePredictor(cacheline=64, banks=4).cost_vector(AccessShape(stride=16, elem_bytes=4, count=64))
    for axis in range(N):
        if axis != CONTENTION:
            assert cv.v[axis] == 0, f"contention signal moved axis {axis} (must be CONTENTION-only)"
    assert cv.v[CONTENTION] > 0, "the contention signal must move the CONTENTION axis"


def test_contention_never_gates_legality_verify_plan_ignores_it():
    """The two-truth quarantine end-to-end on the oracle rail: verify_plan (R8/R9) checks 12-d cost
    completeness/non-negativity + lane legality but NEVER reads the CONTENTION axis as a verdict, so a
    conflict-prone access and a conflict-free one are EXACTLY as legal. We assert the verifier source never
    branches on CONTENTION (the structural quarantine the MLIR -bcir-verify rail also preserves -- it never
    reads the kbcir.contention_* annotations)."""
    import inspect

    import bcir.verify as verify_mod
    src = inspect.getsource(verify_mod)
    assert "CONTENTION" not in src, "the legality verifier must NEVER read the CONTENTION axis (the quarantine)"


# --- the ultimate cross-check: the real bcir-opt law rail (self-skips without the built bcir-opt) -

def test_law_rail_reproduces_oracle_contention_signal_via_bcir_opt():
    """Drive `bcir-opt -bcir-cache-contention` over the oracle-pinned gem.contention IR and assert the law
    recomputes the SAME line_waste_q8/bank_conflict_q8/contention_q8/contention_cost the oracle computed
    (cache_predict.py), and annotates the INFORMS-ONLY quarantine record (kbcir.informs_only = true,
    kbcir.contention_axis = "CONTENTION"). The signal is host-independent (the frozen predictor is pinned)."""
    bo = _find_bcir_opt()
    if not bo:
        return  # the MLIR toolchain is not built in this environment (the oracle-only CI job)
    for case in _CASES:
        p = _oracle_signal(*case)
        proc = subprocess.run([bo, "-bcir-cache-contention"], input=_contention_mlir(p),
                              capture_output=True, text=True)
        assert proc.returncode == 0, f"{case}: law rail rejected the oracle signal:\n{proc.stderr}"
        out = proc.stdout
        line = next(l for l in out.splitlines() if "@c " in l)
        assert f'kbcir.line_waste_q8 = {p["line_waste_q8"]} : i64' in line, f"{case}: waste != oracle\n{line}"
        assert f'kbcir.bank_conflict_q8 = {p["bank_conflict_q8"]} : i64' in line, f"{case}: conflict != oracle\n{line}"
        assert f'kbcir.contention_q8 = {p["contention_q8"]} : i64' in line, f"{case}: q8 != oracle\n{line}"
        assert f'kbcir.contention_cost = {p["contention_cost"]} : i64' in line, f"{case}: cost != oracle\n{line}"
        # the INFORMS-ONLY quarantine record: the signal rides the CONTENTION axis and never gates legality.
        assert "kbcir.informs_only = true" in line, f"{case}: missing the informs-only quarantine flag\n{line}"
        assert 'kbcir.contention_axis = "CONTENTION"' in line, f"{case}: missing the CONTENTION axis tag\n{line}"
        # a cost producer, NOT a lowering: the carrier op is NOT erased.
        assert "bcir.gem.contention" in out, f"{case}: the contention op must NOT be erased (it is a producer)"


def test_law_rail_rejects_an_inconsistent_contention_signal():
    """The headline op-verifier gate, end-to-end on the law rail: a gem.contention whose declared
    contention_q8 does not equal line_waste_q8 + bank_conflict_q8 is REJECTED by the verifier (the frozen
    analytic-model consistency the oracle's CachePredictor enforces). Self-skips without bcir-opt."""
    bo = _find_bcir_opt()
    if not bo:
        return
    bad = ('bcir.module @m {\n'
           '  bcir.gem.contention @bad { stride = 16 : i64, elem_bytes = 4 : i64, count = 64 : i64, '
           'cacheline = 64 : i64, banks = 4 : i64, line_waste_q8 = 3840 : i64, bank_conflict_q8 = 768 : i64, '
           'contention_q8 = 9999 : i64, contention_cost = 1152 : i64 }\n}\n')
    proc = subprocess.run([bo, "-bcir-cache-contention"], input=bad, capture_output=True, text=True)
    assert proc.returncode != 0, "the law rail must reject an inconsistent contention_q8"
    assert "contention_q8 must equal line_waste_q8 + bank_conflict_q8" in proc.stderr, \
        f"the rejection must cite the analytic-model consistency, got:\n{proc.stderr}"


def _find_bcir_opt():
    env = os.environ.get("BCIR_OPT")
    if env and os.path.exists(env):
        return env
    root = os.path.join(os.path.dirname(__file__), "..", "..", "build", "mlir-build")
    for dirpath, _dirs, files in os.walk(os.path.normpath(root)) if os.path.isdir(root) else []:
        if "bcir-opt" in files:
            return os.path.join(dirpath, "bcir-opt")
    return None
