"""G3 dual-rail parity: the MLIR law rail (gem.layout_pivot + -bcir-layout-pivot) reproduces the oracle's
bcir/kbcir/layout.py PRICING DECISION -- the SoA-vs-AoS workload memory cost (priced through the SAME
stride-penalty terms realize.candidates_for prices every claim by -- a single-field sweep is UNIT under SoA /
STRIDED(F) under AoS, a whole-record access the mirror), the chosen MIN-cost layout (ties keep the declared
default SoA), and the modeled gain (a LayoutCertificate-equivalent), all INFORMS-ONLY (the priced choice
informs the plan's memory cost but NEVER gates a legality verdict -- the two-truth quarantine; the hard
"layout changes addresses, never values" invariant is the oracle's). This is the dual-rail gate that matters:
the law's layout choice/cost == plan_layout for the same workload (the discipline is prototype-in-oracle ->
port-to-MLIR-law, parity-gated; mirrors how gem.matmul/activation/conv/attention were ported).

Two arms:
  * a PURE-PYTHON parity arm (always runs): for representative workloads (a single-field-dominated SoA no-op,
    a whole-record-dominated AoS pivot, a one-field resource, a balanced tie) it asserts the oracle's
    plan_layout is internally consistent (soa/aos costs priced through candidates_for's memory axis, the
    min-cost layout chosen with ties keeping SoA, gain = declared-minus-chosen >= 0) -- the contract the MLIR
    op + verifier mirror;
  * the ULTIMATE cross-check (self-skips without the built bcir-opt): drives the real
    `bcir-opt -bcir-layout-pivot` over the oracle-pinned gem.layout_pivot IR and asserts the law recomputes
    the SAME soa_cost/aos_cost/layout/gain the oracle computed AND annotates the informs-only quarantine flag.
    The oracle target is PINNED to x86-avx512 so the priced costs are host-independent (CI-agnostic).
"""

from __future__ import annotations

import os
import subprocess

from bcir.kbcir.cost import TargetProfile
from bcir.kbcir.layout import (
    AOS,
    SOA,
    FieldAccess,
    SINGLE_FIELD,
    WHOLE_RECORD,
    certify_layout,
    plan_layout,
)

# Pin the oracle target so the priced costs are deterministic regardless of the CI host arch
# (TargetProfile.for_host() would price the roofline differently per arch). avx512 is the reference,
# exactly as test_attention_law_parity / test_conv_law_parity.
_T = TargetProfile.x86_avx512()

# The cost-model `streams` for the layout pivot's claim shape: layout.py builds Claim(rd=(1,2), wr=(3,)),
# so streams = len(rd)+len(wr) = 3 -- the constant the op carries (the pricing is over that fixed shape).
_STREAMS = 3

# (fields F, single_field_records, whole_record_records) -- the four representative workloads:
#   * a single-field-dominated workload  -> SoA <= AoS  -> keep SoA (the clean no-op).
#   * a whole-record-dominated workload   -> AoS strictly cheaper -> pivot to AoS (a positive gain).
#   * a one-field resource (F=1)          -> no SoA/AoS distinction (a clean no-op).
#   * a balanced workload (equal records) -> a tie keeps the declared default SoA.
_CASES = [
    (4, 64, 8),  # single-field-dominated: SoA wins, no-op
    (4, 8, 64),  # whole-record-dominated: AoS wins, gain > 0
    (1, 64, 0),  # one field: no distinction, no-op
    (4, 32, 32),  # balanced: a tie keeps SoA
]


def _oracle_plan(fields, sfr, wrr):
    """The oracle's priced SoA/AoS layout plan for a (fields, single_field_records, whole_record_records)
    workload. Returns the dict of fields the law-rail op carries (so the two rails are compared field-by-field).
    The workload is reduced to its two mirrored shapes (one FieldAccess per non-empty shape)."""
    accesses = []
    if sfr > 0:
        accesses.append(FieldAccess(claim_id=1, shape=SINGLE_FIELD, records=sfr, fields=fields))
    if wrr > 0:
        accesses.append(FieldAccess(claim_id=2, shape=WHOLE_RECORD, records=wrr, fields=fields))
    plan = plan_layout(99, fields, tuple(accesses), _T)
    cert = certify_layout(plan)
    return {
        "rid": 99,
        "fields": fields,
        "single_field_records": sfr,
        "whole_record_records": wrr,
        "cacheline": _T.cacheline,
        "elem_bytes": _T.elem_bytes,
        "base_overhead": _T.base_overhead,
        "streams": _STREAMS,
        "vector_width": _T.vector_width,
        "gather_penalty": _T.gather_penalty,
        "soa_cost": plan.soa_cost,
        "aos_cost": plan.aos_cost,
        "layout": plan.layout,
        "gain": cert.gain,
    }


def _layout_mlir(p) -> str:
    """Emit the gem.layout_pivot carrier-op MLIR for an oracle plan dict (the law-rail op the oracle pins)."""
    return (
        "bcir.module @m {\n"
        f"  bcir.gem.layout_pivot @lp {{ rid = {p['rid']} : i64, fields = {p['fields']} : i64, "
        f"single_field_records = {p['single_field_records']} : i64, "
        f"whole_record_records = {p['whole_record_records']} : i64, "
        f"cacheline = {p['cacheline']} : i64, elem_bytes = {p['elem_bytes']} : i64, "
        f"base_overhead = {p['base_overhead']} : i64, streams = {p['streams']} : i64, "
        f"vector_width = {p['vector_width']} : i64, gather_penalty = {p['gather_penalty']} : i64, "
        f"soa_cost = {p['soa_cost']} : i64, aos_cost = {p['aos_cost']} : i64, "
        f'layout = "{p["layout"]}", gain = {p["gain"]} : i64 }}\n'
        "}\n"
    )


# --- the pure-Python parity arm (always runs) -----------------------------------------------------


def test_oracle_layout_plan_is_self_consistent_and_priced_through_the_cost_model():
    """The oracle's layout plan is internally consistent: the chosen layout is the MIN-cost one (ties keep
    the declared default SoA), the gain = the declared-layout (SoA) cost minus the chosen-layout cost (>= 0),
    a single-field-dominated / one-field / balanced workload keeps SoA (a clean no-op), and a whole-record-
    dominated one pivots to AoS with a positive gain. The cost is priced THROUGH THE EXISTING cost model (the
    stride-penalty memory terms), not a bespoke discount. This is the contract the MLIR op + verifier mirror."""
    for fields, sfr, wrr in _CASES:
        p = _oracle_plan(fields, sfr, wrr)
        # the min,+ selection: AoS only on a STRICT win, else SoA.
        want = AOS if p["aos_cost"] < p["soa_cost"] else SOA
        assert p["layout"] == want, f"{(fields, sfr, wrr)}: layout != min-cost"
        chosen = p["aos_cost"] if p["layout"] == AOS else p["soa_cost"]
        assert p["gain"] == p["soa_cost"] - chosen, f"{(fields, sfr, wrr)}: gain != soa - chosen"
        assert p["gain"] >= 0, f"{(fields, sfr, wrr)}: negative gain"
    # the single-field-dominated / one-field / balanced workloads are SoA no-ops; the whole-record one pivots.
    assert _oracle_plan(4, 64, 8)["layout"] == SOA
    assert _oracle_plan(4, 8, 64)["layout"] == AOS and _oracle_plan(4, 8, 64)["gain"] > 0
    assert _oracle_plan(1, 64, 0)["layout"] == SOA and _oracle_plan(1, 64, 0)["gain"] == 0
    assert _oracle_plan(4, 32, 32)["layout"] == SOA  # the tie keeps the default


def test_layout_pivot_never_gates_legality_verify_plan_ignores_it():
    """The two-truth quarantine: the priced layout choice informs the plan's memory cost but is a cost lever,
    NOT a legality verdict -- the legality verifier never reads the layout decision. The hard 'layout changes
    addresses, never values' invariant is the oracle's (lower.layout_kernel proves SoA result == AoS result,
    bit-exact). We assert the legality verifier source never branches on a layout cost (the structural
    quarantine the MLIR -bcir-verify rail also preserves -- it never reads the kbcir.layout/soa_cost/aos_cost
    annotations)."""
    import inspect

    import bcir.verify as verify_mod

    src = inspect.getsource(verify_mod)
    assert "soa_cost" not in src and "aos_cost" not in src, (
        "the legality verifier must NEVER read the layout cost (the informs-only quarantine)"
    )


# --- the ultimate cross-check: the real bcir-opt law rail (self-skips without the built bcir-opt) -


def test_law_rail_reproduces_oracle_layout_plan_via_bcir_opt():
    """Drive `bcir-opt -bcir-layout-pivot` over the oracle-pinned gem.layout_pivot IR and assert the law
    recomputes the SAME soa_cost/aos_cost/layout/gain the oracle's plan_layout computed (priced through the
    same stride-penalty memory terms), and annotates the INFORMS-ONLY quarantine record (kbcir.informs_only =
    true). The priced costs are host-independent because the oracle target is pinned to avx512."""
    bo = _find_bcir_opt()
    if not bo:
        return  # the MLIR toolchain is not built in this environment (the oracle-only CI job)
    for case in _CASES:
        p = _oracle_plan(*case)
        proc = subprocess.run(
            [bo, "-bcir-layout-pivot"], input=_layout_mlir(p), capture_output=True, text=True
        )
        assert proc.returncode == 0, f"{case}: law rail rejected the oracle plan:\n{proc.stderr}"
        out = proc.stdout
        line = next(l for l in out.splitlines() if "@lp " in l)
        assert f"kbcir.soa_cost = {p['soa_cost']} : i64" in line, (
            f"{case}: soa_cost != oracle\n{line}"
        )
        assert f"kbcir.aos_cost = {p['aos_cost']} : i64" in line, (
            f"{case}: aos_cost != oracle\n{line}"
        )
        assert f'kbcir.layout = "{p["layout"]}"' in line, f"{case}: layout != oracle\n{line}"
        assert f"kbcir.gain = {p['gain']} : i64" in line, f"{case}: gain != oracle\n{line}"
        # the INFORMS-ONLY quarantine record: the priced choice informs the plan, never gates legality.
        assert "kbcir.informs_only = true" in line, (
            f"{case}: missing the informs-only quarantine flag\n{line}"
        )
        # a cost producer, NOT a lowering: the carrier op is NOT erased.
        assert "bcir.gem.layout_pivot" in out, (
            f"{case}: the layout_pivot op must NOT be erased (it is a producer)"
        )


def test_law_rail_rejects_an_inconsistent_layout_choice():
    """The headline op-verifier gate, end-to-end on the law rail: a gem.layout_pivot whose declared layout is
    NOT the min-cost one (claiming SoA when AoS prices strictly cheaper) is REJECTED by the verifier (the
    min,+ selection the oracle's plan_layout enforces). Self-skips without bcir-opt."""
    bo = _find_bcir_opt()
    if not bo:
        return
    # the whole-record-dominated workload prices AoS strictly cheaper, but this op declares "soa" -> rejected.
    p = _oracle_plan(4, 8, 64)
    assert p["layout"] == AOS  # sanity: the oracle does pick AoS here
    bad = p.copy()
    bad["layout"] = "soa"
    bad["gain"] = (
        0  # a wrong gain to match the wrong layout (the verifier still rejects the layout first)
    )
    proc = subprocess.run(
        [bo, "-bcir-layout-pivot"], input=_layout_mlir(bad), capture_output=True, text=True
    )
    assert proc.returncode != 0, "the law rail must reject a non-min-cost layout choice"
    assert "must be the min-cost choice" in proc.stderr, (
        f"the rejection must cite the min,+ layout selection, got:\n{proc.stderr}"
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
