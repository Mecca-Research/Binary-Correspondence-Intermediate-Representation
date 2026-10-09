"""G27 fixtures, shared by `test_value_dispatch` and the GEM+ harness so both grade the same
way: the declared value the corpora are priced at, the overspend counts over the memory and the
six-job schedule corpora, the generated straight-line modules, and the delegated-output
comparison."""

from __future__ import annotations

import random
from fractions import Fraction

from bcir.gem.dispatch import (
    DispatchRecord,
    DispatchRequest,
    OptimizationValue,
    delegation,
    solve_memory,
    solve_schedule,
)
from bcir.gem.exact import exact_schedule
from bcir.kbcir.static_memory import first_fit_layout
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.tests.exact_fixtures import six_job_corpus, six_job_module, six_job_target
from bcir.tests.memory_fixtures import CORPUS_SIZE, fixture, items_of

#: The declared value the corpora are priced at: one execution, one cost unit per work unit.
ONCE = OptimizationValue(1)


def overspent(record: DispatchRecord, gap: int, value: OptimizationValue) -> bool:
    """Whether a run spent more work than its gap repays over every execution."""
    return record.spent * Fraction(value.price) > value.executions * gap


def memory_overspend(solve, value: OptimizationValue = ONCE) -> int:
    """How many of the memory corpus's instances `solve(request, items, value)` searches with
    more work (candidate placements) than the first-fit gap repays."""
    count = 0
    for seed in range(CORPUS_SIZE):
        items = items_of(fixture(seed))
        fast = first_fit_layout(items)
        gap = fast.extent - fast.lower_bound
        _layout, record = solve(DispatchRequest("memory", len(items), "TMSAO-1"), items, value)
        if record.decision.rail == "proof" and overspent(record, gap, value):
            count += 1
    return count


def schedule_overspend(solve, value: OptimizationValue = ONCE) -> int:
    """How many of the six-job corpus's schedules `solve(request, module, durations, value,
    target)` searches with more work (node expansions) than the root gap repays."""
    module, target = six_job_module(), six_job_target(2)
    request = DispatchRequest("schedule", 6, "TMSAO-1")
    count = 0
    for durs in six_job_corpus():
        durations = {i + 1: d for i, d in enumerate(durs)}
        root = exact_schedule(module, durations, target, budget=0)
        _result, record = solve(request, module, durations, value, target)
        count += overspent(record, root.heuristic - root.lower_bound, value)
    return count


def parent_memory(request, items, value):
    """The parent's law on a memory region: the caller's budget, whatever the gap."""
    return solve_memory(request, items)


def parent_schedule(request, module, durations, value, target):
    """The parent's law on a schedule region: the caller's budget, whatever the gap."""
    return solve_schedule(request, module, durations, target)


def straight_line_modules(count: int = 40) -> list[Module]:
    """Generated one-claim modules: add, sub and mul; separate, in-place and doubled operands;
    every hazard; trip counts on and off a vector width."""
    out = []
    for seed in range(count):
        r = random.Random(seed)
        m = Module(name=f"line{seed}")
        for rid in (10, 11, 12):
            m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(4096,), name=f"r{rid}"))
        shape = r.choice(((10, 11, 12), (10, 11, 10), (10, 10, 11), (11, 10, 11)))
        claim = Claim(
            id=1000 + seed,
            opcode=r.choice((Opcode.ADD, Opcode.SUB, Opcode.MUL)),
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=r.choice((1, 7, 64, 1000, 1024)),
            rd=shape[:2],
            wr=shape[2:],
            op="vector.op",
            domain=Domain.RAM,
            hazard=r.choice(("unique", "unique", "barriered", "atomic")),
        )
        m.add_phase(Phase(phase_id=0, deps=(), claims=[claim]))
        out.append(m)
    return out


def delegated_modules() -> list[Module]:
    """The corpus's straight-line programs and the generated ones the law delegates."""
    from bcir.examples import PROGRAMS

    modules = [build() for build in PROGRAMS.values()]
    modules += straight_line_modules(16)
    return [m for m in modules if delegation(m).delegated]


def delegation_mismatches() -> int | None:
    """How many delegated modules' outputs differ from the planned kernel's under clang -O2
    (`lower.llvm.compare_delegated`); None when no coherent clang is available."""
    from bcir.kbcir import TARGETS, optimize
    from bcir.kbcir.cost import Theta
    from bcir.lower.llvm import compare_delegated

    mismatches = 0
    for module in delegated_modules():
        result = optimize(module, TARGETS["x86_avx512"], Theta.cool())
        verdict, _width, _out = compare_delegated(module, result)
        if verdict.startswith("skip:"):
            return None
        mismatches += verdict != "match"
    return mismatches


def measure() -> dict[str, float]:
    """The G27 harness rows: the priced memory rail's overspend, and -- where clang is -- the
    delegated modules whose output differs from the planned kernel's."""
    from bcir.gem.dispatch import solve_memory_by_value, solve_schedule_by_value

    overspend = memory_overspend(solve_memory_by_value) + schedule_overspend(
        solve_schedule_by_value
    )
    out = {"dispatch.value.overspend": float(overspend)}
    mismatches = delegation_mismatches()
    if mismatches is not None:
        out["dispatch.delegation.mismatch"] = float(mismatches)
    return out
