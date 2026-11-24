"""Precision-contract reference planner: exact bounds, refusals and DAG fanout.

These tests use Fractions rather than float approximations and do NOT claim to
benchmark BF16/FP8/MXFP4 hardware or validate model perplexity.
"""

from fractions import Fraction

import pytest

from bcir.kbcir.precision_contracts import (
    ErrorBound,
    NumericalRepresentation,
    PrecisionContractError,
    PrecisionNode,
    PrecisionOption,
    propagate_precision,
    quantization_error_bound_real,
    quantization_error_in_q8_ticks,
    signed_dot_accumulator_bits,
)


def rep(storage="Q8", **kwargs):
    return NumericalRepresentation(
        storage=storage, compute="FP32", accumulator="FP32", **kwargs
    )


def option(name, error, cost, assurance="bounded", evidence_id="test:local-proof"):
    return PrecisionOption(
        name=name,
        representation=rep(),
        local_error=None if error is None else ErrorBound(error),
        cost=cost,
        assurance=assurance,
        evidence_id=evidence_id,
    )


def node(name, inputs=(), gains=(), options=None):
    return PrecisionNode(
        name, tuple(inputs), tuple(gains),
        tuple(options) if options is not None else (option("exact", 0, 2, "exact"),)
    )


def test_explicit_error_units_and_no_float_budget_rounding():
    assert ErrorBound(Fraction(1, 3), "q8_tick").amount == Fraction(1, 3)
    with pytest.raises(PrecisionContractError):
        ErrorBound(0.1)
    with pytest.raises(PrecisionContractError):
        ErrorBound(-1)
    with pytest.raises(PrecisionContractError):
        ErrorBound(1, "")


def test_quantization_grid_is_not_q8_grid():
    bound = quantization_error_bound_real(-4, range_proved=True)
    assert bound.amount == Fraction(1, 32)
    assert quantization_error_in_q8_ticks(-4, range_proved=True) == 8
    assert quantization_error_in_q8_ticks(-4, rounding="truncate", range_proved=True) == 16
    assert quantization_error_in_q8_ticks(-10, range_proved=True) == 1
    assert quantization_error_in_q8_ticks(2, range_proved=True) == 512


def test_quantizer_refuses_unproven_range_or_unsupported_rounding():
    with pytest.raises(PrecisionContractError, match="no-clipping"):
        quantization_error_bound_real(-4)
    with pytest.raises(PrecisionContractError):
        quantization_error_bound_real(-4, rounding="stochastic", range_proved=True)
    with pytest.raises(PrecisionContractError):
        quantization_error_bound_real(400, range_proved=True)
    with pytest.raises(PrecisionContractError):
        quantization_error_bound_real(True, range_proved=True)


def test_symmetric_code_dot_uses_sufficient_signed_accumulator():
    for bits_a, bits_b, count in [(2, 2, 1), (4, 8, 37), (8, 8, 1024)]:
        width = signed_dot_accumulator_bits(bits_a, bits_b, count)
        biggest = count * ((1 << (bits_a - 1)) - 1) * ((1 << (bits_b - 1)) - 1)
        assert biggest <= (1 << (width - 1)) - 1
        assert -biggest >= -(1 << (width - 1))
    with pytest.raises(PrecisionContractError):
        signed_dot_accumulator_bits(8, 8, 0)


def test_mx_storage_requires_exact_ocp_v1_metadata():
    valid = NumericalRepresentation(
        storage="MXFP4", compute="FP32", accumulator="FP32",
        block_size=32, scale_format="E8M0"
    )
    assert valid.block_size == 32
    with pytest.raises(PrecisionContractError):
        NumericalRepresentation(storage="MXFP4", compute="FP32", accumulator="FP32")
    with pytest.raises(PrecisionContractError):
        NumericalRepresentation(
            storage="MXFP8", compute="FP32", accumulator="FP32",
            block_size=16, scale_format="E8M0"
        )


def test_stochastic_rounding_must_be_counter_addressed():
    with pytest.raises(PrecisionContractError, match="seed"):
        rep(rounding="stochastic")
    r = rep(rounding="stochastic", stochastic_seed=123, stochastic_counter="op42/step")
    assert r.stochastic_counter == "op42/step"
    with pytest.raises(PrecisionContractError):
        rep(stochastic_seed=123, stochastic_counter="op42")


def test_empirical_results_cannot_masquerade_as_proof():
    x = node("x", options=[option("FP32", 0, 5, "exact")])
    y = node(
        "y", inputs=["x"], gains=[1],
        options=[
            option("fast-empirical", Fraction(1, 100), 1, "empirical", "ppl:pass"),
            option("safe", Fraction(1, 100), 4)
        ]
    )
    p = propagate_precision([x, y], {"y": ErrorBound(1)})
    assert p.selected["y"].name == "safe"
    with pytest.raises(PrecisionContractError):
        propagate_precision(
            [x, node("y", ["x"], [1], [option("only-predicted", 0, 1, "predicted")])],
            {"y": ErrorBound(1)}
        )


def test_backward_budget_selects_cheapest_proven_implementation():
    x = node("x")
    y = node("y", ["x"], [1], [
        option("FP64", 0, 20, "exact"),
        option("FP32", Fraction(1, 8), 3),
        option("FP8", Fraction(3, 4), 1),
    ])
    p = propagate_precision([x, y], {"y": ErrorBound(1)})
    assert p.selected["y"].name == "FP32"
    assert p.output_bounds["y"].amount == Fraction(1, 8)
    assert p.local_budgets["y"].amount == Fraction(1, 2)
    assert p.total_cost == 5


def test_shared_dag_fanout_sums_sensitivities_not_min_of_paths():
    x = node("x", options=[
        option("cheap", Fraction(1, 8), 1),
        option("wide", 0, 9, "exact"),
    ])
    a = node("a", ["x"], [2])
    b = node("b", ["x"], [3])
    y = node("y", ["a", "b"], [1, 1])
    p = propagate_precision([x, a, b, y], {"y": ErrorBound(Fraction(1, 2))})
    assert p.selected["x"].name == "wide"
    assert p.local_budgets["x"].amount == Fraction(1, 40)
    assert p.output_bounds["y"].amount == 0


def test_multiple_outputs_keep_tighter_requirement():
    x = node("x", options=[
        option("fast", Fraction(1, 16), 1),
        option("exact", 0, 9, "exact"),
    ])
    a = node("a", ["x"], [1])
    b = node("b", ["x"], [4])
    p = propagate_precision(
        [x, a, b],
        {"a": ErrorBound(1), "b": ErrorBound(Fraction(1, 8))}
    )
    assert p.selected["x"].name == "exact"
    assert p.local_budgets["x"].amount == Fraction(1, 64)


def test_rejects_unordered_cycles_or_missing_dependencies():
    a = node("a", ["b"], [1])
    b = node("b")
    with pytest.raises(PrecisionContractError, match="topological"):
        propagate_precision([a, b], {"a": ErrorBound(1)})


def test_rejects_units_not_proven_commensurate():
    x = node("x")
    y = node("y", ["x"], [1])
    with pytest.raises(PrecisionContractError, match="common"):
        propagate_precision(
            [x, y], {"x": ErrorBound(1, "meters"), "y": ErrorBound(1, "seconds")}
        )
    with pytest.raises(PrecisionContractError):
        propagate_precision([x], {"missing": ErrorBound(1)})


def test_negative_gains_rejected_and_no_qualified_implementation_refused():
    with pytest.raises(PrecisionContractError):
        node("x", ["z"], [-1])
    with pytest.raises(PrecisionContractError):
        option("exact-with-error", 1, 5, "exact")
    with pytest.raises(PrecisionContractError):
        option("unproven", 0, 5, "bounded", evidence_id="")
    with pytest.raises(PrecisionContractError, match="no qualified"):
        propagate_precision(
            [node("x", options=[option("too-inaccurate", 2, 1)])],
            {"x": ErrorBound(1)}
        )
