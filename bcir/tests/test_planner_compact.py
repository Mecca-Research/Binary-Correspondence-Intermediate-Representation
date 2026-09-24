"""GEM+ G17 (S4-A): the compact planner, its records and its native twin.

The compact planner (`realize.fused_offer` + `realize.optimize`) must be the pre-G17 planner
(`realize_reference`) byte for byte, R9 must re-derive the chosen realization from the same offer
without building the rest, and the native planner (`runtime/c/bcir_kplan.c`) must write the
compact planner's plan byte for byte -- or refuse it for the same reason. Each law is held to a
negative witness: a planner that disagrees, a forgery R9 must refuse, a record each law must
refuse, a fixture that makes the 128-bit carry observable.
"""

from __future__ import annotations

import os
import random
import re
import tempfile

from bcir.abi import planner_abi as pa
from bcir.kbcir import realize, realize_reference
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.weights import PERF
from bcir.tests import planner_fixtures as pf
from bcir.verify import verify_plan

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _steps(result):
    return [(s.claim_id, s.phase_id, s.candidate, s.cost) for s in result.steps]


def test_the_compact_planner_is_the_reference_planner_over_a_corpus_sample():
    cases = pf.corpus_cases()
    rng = random.Random(1709)
    sample = [c for c in cases if c.fixed][::7] + rng.sample([c for c in cases if not c.fixed], 120)
    for case in sample:
        args = (case.module, case.h, case.theta, case.policy)
        new, old = realize.optimize(*args), realize_reference.optimize(*args)
        assert new == old and _steps(new) == _steps(old), case.label
        assert pf.realization_bytes(new) == pf.realization_bytes(old), case.label


def test_the_candidate_map_is_one_derivation_and_a_faithful_view():
    for case in pf.corpus_cases()[::41]:
        fused = realize.fused_candidates(case.module, case.h)
        want = realize_reference.fused_candidates(case.module, case.h)
        assert fused == want and list(fused) == list(want), case.label
        view = realize.optimize(case.module, case.h, case.theta, case.policy).cand_map
        if view is None:
            continue
        assert list(view) == list(want) and len(view) == len(want)
        assert dict(view) == want
        first = next(iter(want))
        assert view.get(first) is view[first]  # built once, then kept
        assert view.get(object()) is None and object() not in view


def test_a_duplicate_claim_id_plans_as_the_id_keyed_map_always_did():
    """Outside the native domain, the compact planner keeps the pre-G17 behavior exactly: every
    occurrence of a repeated id plans with the last one's realizations."""
    from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass

    m = Module(name="dup")
    m.add_resource(Resource(rid=1, shape=(64,)))
    m.add_resource(Resource(rid=2, shape=(64,)))
    m.add_phase(
        Phase(
            0,
            (),
            [
                Claim(
                    id=7,
                    opcode=Opcode.ADD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=64,
                    rd=(1,),
                    wr=(2,),
                    op="a",
                ),  # fmt: skip
                Claim(
                    id=7,
                    opcode=Opcode.T_MACC,
                    lane=Lane.T,
                    stride_class=StrideClass.TILE,
                    count=64,
                    rd=(2,),
                    wr=(1,),
                    op="b",
                ),  # fmt: skip
            ],
        )
    )
    h = TargetProfile.x86_avx512()
    for theta in (Theta.cool(), Theta.hot()):
        new, old = realize.optimize(m, h, theta), realize_reference.optimize(m, h, theta)
        assert new == old and _steps(new) == _steps(old)
    try:
        pa.encode_input(m, h, Theta.cool())
    except pa.PlannerAbiError as exc:
        assert exc.status == "BCIR_ERR_PLANNER"
    else:
        raise AssertionError("the native domain admitted a duplicate claim id")


def test_the_planner_makes_a_stated_factor_fewer_calls():
    """planner.calls: the call count is per interpreter, so the gate compares the two planners in
    one process. The stated factor is 5: 5.8x at scale 8 and 5.76x at scale 4 on CPython 3.11."""
    reference = pf.call_count(realize_reference.optimize, scale=4)
    compact = pf.call_count(realize.optimize, scale=4)
    assert reference >= 5 * compact, (reference, compact)


def _forged(result, index, **fields):
    from dataclasses import replace

    steps = list(result.steps)
    s = steps[index]
    steps[index] = replace(s, candidate=replace(s.candidate, **fields))
    return replace(result, steps=steps)


def _old_admissibility(module, h, result):
    """The pre-G17 R9 admissibility, spelled over the reference offer: the diagnostics
    `verify_plan` must still produce."""
    offered = realize_reference.fused_candidates(module, h)
    out = []
    for step in result.steps:
        cands = offered.get(step.claim_id, ())
        c = step.candidate
        if not any(
            c.lane is o.lane and c.width == o.width and c.name == o.name and c.base.v == o.base.v
            for o in cands
        ):
            lane = c.lane.name if hasattr(c.lane, "name") else repr(c.lane)
            out.append(
                f"claim {step.claim_id}: realization {c.name!r}({lane}, width {c.width}) "
                f"is not among the {len(cands)} candidate(s) this target admits: "
                f"{sorted(o.name for o in cands)}"
            )
    return out


def test_r9_rederives_the_chosen_realization_and_refuses_every_forgery():
    from bcir.examples import PROGRAMS
    from bcir.kbcir.cost import CostVector
    from bcir.model import Lane

    h, theta = TargetProfile.x86_avx512(), Theta.cool()
    for name in ("fused_chain", "histogram_gather", "tiled_matmul", "vector_add"):
        module = PROGRAMS[name]()
        plan = realize.optimize(module, h, theta)
        assert verify_plan(module, plan, h, theta=theta, policy=PERF) == []
        c = plan.steps[0].candidate
        forgeries = [
            _forged(plan, 0, name="forged"),
            _forged(plan, 0, width=c.width * 2 + 1),
            _forged(plan, 0, base=CostVector(tuple(x + 1 for x in c.base.v))),
            _forged(plan, 0, lane=int(c.lane)),  # an int, not the Lane: `is` refuses it
            _forged(plan, 0, lane=Lane.A if c.lane is not Lane.A else Lane.U),
        ]
        for forged in forgeries:
            got = [d.message for d in verify_plan(module, forged, h) if d.law == "R9"]
            want = _old_admissibility(module, h, forged)
            assert want and all(w in got for w in want), (name, got, want)
        # a forged cost is refused by the scope re-derivation, not the offer
        from dataclasses import replace

        steps = list(plan.steps)
        steps[0] = replace(steps[0], cost=steps[0].cost + 1)
        cost_forged = replace(plan, steps=steps, score=plan.score + 1)
        assert any(
            "does not re-derive" in d.message
            for d in verify_plan(module, cost_forged, h, theta=theta, policy=PERF)
        )


def test_r9_is_total_over_an_unhashable_forgery():
    from dataclasses import replace

    from bcir.examples import PROGRAMS

    module = PROGRAMS["vector_add"]()
    h, theta = TargetProfile.x86_avx512(), Theta.cool()
    plan = realize.optimize(module, h, theta)
    forged = _forged(plan, 0, name=["not", "hashable"])
    diags = verify_plan(module, forged, h)  # a verdict, never a traceback (L1)
    assert any(d.law == "R9" for d in diags)
    steps = list(plan.steps)
    steps[0] = replace(steps[0], phase_id=[0])
    diags = verify_plan(module, replace(plan, steps=steps), h, theta=theta, policy=PERF)
    assert isinstance(diags, list)


def test_the_records_round_trip_and_every_law_refuses_its_violation():
    for case in pf.corpus_cases()[::53]:
        data = pa.encode_input(case.module, case.h, case.theta, case.policy)
        value = pa.decode_input(data)
        pa.check_input(value)
        assert pa._pack_input(value) == data  # one spelling
        status, plan = pf.realization_bytes(
            realize.optimize(case.module, case.h, case.theta, case.policy)
        )
        if status == "BCIR_OK":
            assert pa.encode_realization(realize.optimize(
                case.module, case.h, case.theta, case.policy)) == plan  # fmt: skip
            assert pa.decode_realization(plan).score >= 0
    for label, data, status in pf.malformed_inputs():
        assert pf.python_input_status(data) == status, label
    for label, data, status in pf.malformed_realizations():
        assert pf.python_realization_status(data) == status, label


def test_every_law_has_a_malformed_witness():
    """Every refusal the decoder and `check_input` can make is reached by some malformed variant --
    each `_refuse(...)` message, its placeholders matching anything. A law nothing reaches is a law
    nothing tests (L22); two were missing when this test was written."""
    import inspect

    reached = []
    for _label, data, _status in pf.malformed_inputs():
        try:
            pa.check_input(pa.decode_input(data))
        except pa.PlannerAbiError as exc:
            reached.append(str(exc))
    source = inspect.getsource(pa.decode_input) + inspect.getsource(pa.check_input)
    laws = re.findall(r'_refuse\(\s*f?"((?:[^"\\]|\\.)*)"', source)
    assert len(laws) >= 20, laws
    for law in laws:
        pattern = "^" + ".*".join(re.escape(part) for part in re.split(r"\{[^}]*\}", law)) + "$"
        assert any(re.match(pattern, r, re.S) for r in reached), (
            f"no malformed variant reaches {law!r}"
        )


def test_the_carry_and_the_wide_path_fixtures_bite():
    """The two cases that make the native planner's 128-bit arithmetic observable (L11): the carry
    case must be refused (its score is past 2**63 - 1 only because terms that each fit a u64 are
    added), and the wide-path case must carry a losing path heavier than 2**64."""
    carry = pf.carry_case()
    status, _ = pf.realization_bytes(
        realize.optimize(carry.module, carry.h, carry.theta, carry.policy)
    )
    assert status == "BCIR_ERR_OVERFLOW"
    wide = pf.wide_path_case()
    offer = realize.fused_offer(wide.module, wide.h)
    from bcir.kbcir.weights import weights

    w = weights(wide.h, wide.theta, 0, wide.policy)
    heaviest = max(
        realize._edge_cost_pair(row[3], wide.theta.thermal >= 60 and row[1] >= 16, w)[0]
        for row in offer.rows[0]
    )
    assert heaviest >= 1 << 64
    status, _ = pf.realization_bytes(realize.optimize(wide.module, wide.h, wide.theta, wide.policy))
    assert status == "BCIR_OK"


def test_the_gate_and_the_harness_link_the_same_sources():
    """The #719 trap: tools/c/check_runtime.sh's planner section and planner_fixtures build the
    harness from one source list, read out of both files."""
    with open(os.path.join(_ROOT, "tools", "c", "check_runtime.sh"), encoding="utf-8") as f:
        script = f.read()
    line = re.search(r"^kplan_sources=\(([^)]*)\)", script, re.M)
    assert line, "check_runtime.sh has no kplan_sources"
    gate = sorted(re.findall(r"\$\{C\}/([\w.]+)", line.group(1)))
    assert gate == sorted((*pf.C_UNITS, pf.HARNESS))


def test_the_native_planner_is_the_compact_planner_byte_for_byte():
    """planner.parity and planner.malformed.accepted at zero on both rails. The quick tier hides the
    compiler on purpose; where one is visible the rows must be measured (L2)."""
    with tempfile.TemporaryDirectory(prefix="bcir-kplan-") as tmp:
        exe = pf.build_harness(tmp)
        if exe is None:
            return
        assert pf.measure(exe, tmp) == {row: 0.0 for row in pf.ROWS}
