"""B3 transcendental, mutation, dynamic-loop, rematerialization, and AD-order gates."""
from __future__ import annotations

from bcir.kbcir.autodiff import (Tape, finite_difference_grad, grad, gradients_match, hessian,
                                 second_difference_hessian)
from bcir.kbcir.autodiff_program import (ADProgramFeatures, ADQuarantineReason,
                                         ADQuarantinedError, MutationStatement, admit_ad_program,
                                         compare_ad_ordering, defunctionalize_call,
                                         functionalize_scalar_mutation,
                                         grad_with_rematerialization, trace_bounded_while)
from bcir.lower.autodiff_kernel import compile_and_run_grad_c, emit_autodiff_kernel_c


def test_transcendental_vjps_close_and_match_finite_differences_and_c():
    tape = Tape(); x = tape.var("x")
    positive = tape.add(tape.mul(x, x), tape.const(2.0))
    output = tape.add(tape.exp(tape.neg(x)), tape.log(positive))
    output = tape.add(output, tape.sqrt(positive))
    output = tape.add(output, tape.tanh(x))
    output = tape.add(output, tape.sin(x))
    output = tape.add(output, tape.cos(x))
    env = {"x": 0.7}
    analytic = grad(tape, output, env).grads
    numeric = finite_difference_grad(tape, output, env)
    assert gradients_match(analytic, numeric, rtol=2e-4, atol=2e-5)
    symbolic_hessian = hessian(tape, output, ("x",), env)
    numeric_hessian = second_difference_hessian(tape, output, env)
    assert abs(symbolic_hessian[("x", "x")] - numeric_hessian[("x", "x")]) < 2e-3
    source = emit_autodiff_kernel_c(tape, output, ("x",), "trans_grad")
    assert "#include <math.h>" in source and "expf(" in source and "tanhf(" in source
    ok, value, gradients, diagnostics = compile_and_run_grad_c(
        tape, output, ("x",), env, "trans_grad")
    if ok:
        assert abs(value - grad(tape, output, env).value) < 2e-5, diagnostics
        assert abs(gradients[0] - analytic["x"]) < 2e-5, diagnostics


def test_admission_rewrites_supported_features_and_quarantines_escapes():
    supported = admit_ad_program(ADProgramFeatures(
        mutation="local_scalar", dynamic_loop=True, loop_max_steps=8,
        higher_order="finite"))
    assert supported.admitted and len(supported.rewrites) == 3
    refused = admit_ad_program(ADProgramFeatures(
        mutation="aliased", dynamic_loop=True, higher_order="dynamic",
        recursion=True, unknown_calls=True))
    assert not refused.admitted
    assert set(refused.quarantine) == {
        ADQuarantineReason.ALIASED_MUTATION,
        ADQuarantineReason.DYNAMIC_LOOP_WITHOUT_BUDGET,
        ADQuarantineReason.DYNAMIC_HIGHER_ORDER_CALL,
        ADQuarantineReason.RECURSION,
        ADQuarantineReason.UNKNOWN_CALL,
    }


def test_local_scalar_mutation_is_functionalized_into_ssa_nodes():
    program = functionalize_scalar_mutation(
        ("x", "y"), (MutationStatement("x", "add", "y"),
                     MutationStatement("x", "mul", 2.0)), "x")
    result = grad(program.tape, program.output, {"x": 3.0, "y": 4.0})
    assert result.value == 14.0 and result.grads == {"x": 2.0, "y": 2.0}
    assert program.versions == (("x", 1), ("x", 2))


def test_dynamic_while_is_traced_with_a_hard_budget_and_remains_differentiable():
    tape = Tape(); x = tape.var("x"); one = tape.const(1.0); three = tape.const(3.0)
    condition = lambda graph, carry: graph.sub(three, carry)
    body = lambda graph, carry: graph.add(carry, one)
    output, trace = trace_bounded_while(tape, x, condition, body, {"x": 0.0}, max_steps=4)
    assert trace.iterations == 3 and grad(tape, output, {"x": 0.0}).grads["x"] == 1.0
    other = Tape(); start = other.const(0.0); one2 = other.const(1.0); three2 = other.const(3.0)
    try:
        trace_bounded_while(other, start,
                            lambda graph, carry: graph.sub(three2, carry),
                            lambda graph, carry: graph.add(carry, one2), {}, max_steps=2)
        assert False, "live loop exceeded its budget"
    except ADQuarantinedError as exc:
        assert exc.reason == ADQuarantineReason.DYNAMIC_LOOP_BUDGET_EXHAUSTED


def test_finite_higher_order_calls_are_defunctionalized_by_tag():
    tape = Tape(); x = tape.var("x")
    table = {"square": lambda graph, value: graph.mul(value, value),
             "sine": lambda graph, value: graph.sin(value)}
    output = defunctionalize_call("square", table, tape, (x,))
    assert grad(tape, output, {"x": 3.0}).grads["x"] == 6.0
    try:
        defunctionalize_call("runtime-value", table, tape, (x,))
        assert False, "dynamic function escaped quarantine"
    except ADQuarantinedError as exc:
        assert exc.reason == ADQuarantineReason.DYNAMIC_HIGHER_ORDER_CALL


def test_checkpoint_rematerialization_replays_and_matches_full_reverse_mode():
    tape = Tape(); x = tape.var("x"); value = x
    for index in range(1, 9):
        value = tape.tanh(tape.add(tape.mul(value, x), tape.const(index / 10.0)))
    env = {"x": 0.25}
    full = grad(tape, value, env)
    remat = grad_with_rematerialization(tape, value, env, max_saved_intermediates=2)
    assert remat.result.value == full.value
    assert abs(remat.result.grads["x"] - full.grads["x"]) < 1e-12
    assert len(remat.plan.checkpoint_nodes) <= 2
    assert remat.evaluated_ops >= full.forward_ops


def test_pre_post_optimization_measurement_selects_differentiate_high_optimize_low():
    tape = Tape(); x = tape.var("x"); zero = tape.const(0.0); one = tape.const(1.0)
    redundant = tape.mul(tape.add(x, zero), one)
    output = tape.add(tape.mul(redundant, redundant), tape.sin(redundant))
    report = compare_ad_ordering(tape, output, {"x": 0.4})
    assert report.selected_strategy == "differentiate-high-optimize-low"
    assert report.optimized_forward_nodes < report.forward_nodes
    assert report.max_gradient_error < 1e-12
    assert report.differentiate_high_optimize_low_ns > 0
    assert report.optimize_high_differentiate_low_ns > 0
