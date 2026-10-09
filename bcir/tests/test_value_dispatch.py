"""G27: the depth of optimization priced by its value; straight-line modules delegated to LLVM.

The parent's dispatch law granted whatever budget the caller named: a search could spend
more work than the gap it was closing could ever repay, and nothing routed simple code to
LLVM because deep optimization would not pay (the audit's item 14). These witnesses hold the
value law to its own arithmetic (never more work than `executions x gap / price`, none at a
closed gap, monotone in each input), show the priced runners spend within it on the corpora
the exact rail is proved on -- where the parent's law overspends -- and hold every delegated
module's kernel to the planned one: the same text at width one, and the same bytes out of
clang at every trip count.
"""

from __future__ import annotations

import random
from fractions import Fraction

from bcir.examples import PROGRAMS
from bcir.gem.dispatch import (
    DispatchRequest,
    OptimizationValue,
    delegation,
    dispatch,
    dispatch_by_value,
    priced_budget,
    solve_memory_by_value,
    solve_schedule_by_value,
)
from bcir.gem.exact import exact_schedule
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.kbcir.static_memory import exact_layout, first_fit_layout
from bcir.lower import llvm
from bcir.tests.exact_fixtures import six_job_corpus, six_job_module, six_job_target
from bcir.tests.memory_fixtures import fixture, items_of
from bcir.tests.value_fixtures import (
    ONCE,
    delegated_modules,
    delegation_mismatches,
    memory_overspend,
    overspent,
    parent_memory,
    parent_schedule,
    schedule_overspend,
    straight_line_modules,
)


def test_the_value_law_is_its_own_arithmetic():
    """Over generated requests: never above the caller's budget, never above what the gap
    repays, nothing at a closed gap; monotone -- more executions or a wider gap never grant
    less, a dearer work unit never more -- and an unsound bound or a malformed value refused."""
    rng = random.Random(27)
    for _ in range(3000):
        requested = rng.randint(0, 5000)
        bound = rng.randint(0, 400)
        incumbent = bound + rng.choice((0, 0, rng.randint(1, 300)))
        value = OptimizationValue(
            rng.randint(0, 50), rng.choice((1, 3, Fraction(1, 4), Fraction(7, 2)))
        )
        granted = priced_budget(requested, incumbent, bound, value)
        gap = incumbent - bound
        assert 0 <= granted <= requested
        assert granted * Fraction(value.price) <= value.executions * gap
        if gap == 0:
            assert granted == 0
        more = OptimizationValue(value.executions + rng.randint(1, 9), value.price)
        assert priced_budget(requested, incumbent, bound, more) >= granted
        assert priced_budget(requested, incumbent + 5, bound, value) >= granted
        dearer = OptimizationValue(value.executions, Fraction(value.price) * 2)
        assert priced_budget(requested, incumbent, bound, dearer) <= granted
    for bad in (
        lambda: priced_budget(10, 5, 6, ONCE),
        lambda: OptimizationValue(-1),
        lambda: OptimizationValue(True),
        lambda: OptimizationValue(3, 0),
        lambda: OptimizationValue(3, 0.5),
        lambda: OptimizationValue(3, True),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("a malformed value was priced")


def test_the_decision_names_what_the_value_bought():
    proof = DispatchRequest("memory", 7, "TMSAO-1", budget=1000)
    closed = dispatch_by_value(proof, 64, 64, ONCE)
    assert (closed.rail, closed.expected, closed.budget) == ("fast", "TMSAO-1", 0)
    cheap = dispatch_by_value(proof, 64, 60, OptimizationValue(1, 5))
    assert (cheap.rail, cheap.expected, cheap.budget) == ("fast", "TMSAO-2", 0)
    assert "less than one work unit" in cheap.reason
    priced = dispatch_by_value(proof, 64, 60, OptimizationValue(30))
    assert (priced.rail, priced.budget) == ("proof", 120) and "priced by value" in priced.reason
    rich = dispatch_by_value(proof, 64, 60, OptimizationValue(10**6))
    assert (rich.rail, rich.budget, rich.solver) == ("proof", proof.budget, dispatch(proof).solver)
    # a heuristic or a measured request is the table's answer, whatever the value
    for asked in ("TMSAO-4", "TMSAO-3"):
        request = DispatchRequest("memory", 7, asked, budget=1000)
        assert dispatch_by_value(request, 64, 0, OptimizationValue(10**6)) == dispatch(request)


def test_the_priced_memory_rail_never_spends_more_than_the_gap_repays():
    """On the 500-instance memory corpus the exact rail is proved on, the parent's law (the
    caller's budget, whatever the gap) searches past what one execution repays on 25
    instances; the priced runner on none -- and with a value that repays the whole budget it is the parent's
    answer, layout for layout."""
    parent = memory_overspend(parent_memory)
    assert parent == 25, parent
    assert memory_overspend(solve_memory_by_value) == 0
    rich = OptimizationValue(10**9)
    for seed in range(60):
        items = items_of(fixture(seed))
        request = DispatchRequest("memory", len(items), "TMSAO-1")
        layout, record = solve_memory_by_value(request, items, rich)
        fast = first_fit_layout(items)
        if fast.extent == fast.lower_bound:
            assert record.stop_reason == "optimal" and record.spent == 0
            assert layout.extent == fast.extent
            continue
        reference = exact_layout(items, request.budget)
        assert (layout.extent, layout.offsets) == (reference.extent, reference.offsets)


def test_the_priced_schedule_rail_resumes_the_root_on_the_priced_budget():
    """The six-job corpus: the root (budget zero) prices the search; the run that resumes
    from it on the priced budget is the uninterrupted run at that budget, and never spends
    more than the gap repays."""
    module, target = six_job_module(), six_job_target(2)
    request = DispatchRequest("schedule", 6, "TMSAO-1")
    checked = priced_out = 0
    for durs in six_job_corpus()[::41]:
        durations = {i + 1: d for i, d in enumerate(durs)}
        root = exact_schedule(module, durations, target, budget=0)
        gap = root.heuristic - root.lower_bound
        for value in (ONCE, OptimizationValue(40), OptimizationValue(1, 1000)):
            result, record = solve_schedule_by_value(request, module, durations, value, target)
            assert not overspent(record, gap, value)
            if record.decision.rail == "fast":
                priced_out += record.stop_reason == "value"
                assert result.incumbent == root.heuristic
                continue
            budget = record.decision.budget
            assert budget == priced_budget(request.budget, root.heuristic, root.lower_bound, value)
            direct = exact_schedule(module, durations, target, budget=budget)
            assert (result.incumbent, result.lower_bound, result.expansions) == (
                direct.incumbent,
                direct.lower_bound,
                direct.expansions,
            )
            checked += 1
    assert checked >= 20 and priced_out >= 5, (checked, priced_out)
    # over the whole corpus: the parent's law searches every open gap past what one
    # execution repays; the priced rail none
    assert schedule_overspend(parent_schedule) == 437
    assert schedule_overspend(solve_schedule_by_value) == 0


def _body(kernel: str) -> str:
    return kernel.split("\n", 1)[1]


def test_straight_line_modules_are_delegated_and_the_rest_planned():
    """The corpus: the two one-claim adds are delegated, every other program planned with the
    reason. The generated modules: delegated exactly when the lowering subset takes their
    claim (an atomic claim is planned, with the subset's refusal), and a delegated kernel is
    the planned kernel at width one, line for line below its provenance comment."""
    verdicts = {name: delegation(build()).delegated for name, build in PROGRAMS.items()}
    assert {name for name, d in verdicts.items() if d} == {"vector_add", "vector_add_hbm"}
    for name, build in PROGRAMS.items():
        reason = delegation(build()).reason
        assert reason.startswith("delegated" if verdicts[name] else "planned: not straight-line")
    h = TARGETS["x86_avx512"]
    delegated = refused = 0
    for module in straight_line_modules():
        verdict = delegation(module)
        claim = module.phases[0].claims[0]
        if claim.hazard == "atomic":
            assert not verdict.delegated and "atomic" in verdict.reason
            refused += 1
            continue
        assert verdict.delegated, verdict.reason
        result = optimize(module, h, Theta.cool())
        planned = llvm.emit_kernel_ll(module, result, "k", width_override=1)
        ours = llvm.emit_delegated_ll(module, "k")
        assert _body(ours) == _body(planned)
        assert "width=1" in ours.split("\n", 1)[0] and "delegated to LLVM" in ours
        delegated += 1
    assert delegated >= 20 and refused >= 5


def test_a_delegated_kernels_output_is_the_planned_kernels():
    """clang -O2 builds the planned kernel (at its K_BCIR-selected width) and the delegated
    one (scalar; LLVM's vectorizer picks) into one program, which runs both at every harness
    trip count on its own copy of every buffer and compares them byte for byte -- canaries
    included. Without a coherent clang this is a skip, never a pass; and the comparison is
    shown to fail on a delegated kernel computing another operation."""
    h = TARGETS["x86_avx512"]
    modules = delegated_modules()
    assert len(modules) >= 8
    first = optimize(modules[0], h, Theta.cool())
    verdict, _width, _out = llvm.compare_delegated(modules[0], first)
    if verdict.startswith("skip:"):
        assert delegation_mismatches() is None  # a skip is never counted as a match
        return  # the quick tier hides the toolchain; the thorough tier and CI run it
    vectorized = 0
    for module in modules:
        result = optimize(module, h, Theta.cool())
        verdict, width, out = llvm.compare_delegated(module, result)
        assert verdict == "match", (module.name, out)
        vectorized += bool(width and width > 1)
    assert vectorized >= 1  # LLVM made the width decision on at least one delegated loop
    assert delegation_mismatches() == 0
    original = llvm.emit_delegated_ll
    try:
        llvm.emit_delegated_ll = lambda m, fn="bcir_kernel", elem="f32": original(
            m, fn, elem
        ).replace("fadd", "fsub")
        verdict, _width, out = llvm.compare_delegated(modules[0], first)
        assert verdict == "MISMATCH", out
    finally:
        llvm.emit_delegated_ll = original
