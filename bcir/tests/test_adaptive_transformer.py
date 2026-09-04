"""Dependency-free gates for adaptive transformer contracts and lowering."""

from __future__ import annotations

import math

from bcir.kbcir.adaptive_transformer import (
    AdaptiveLanguageSpec,
    ExogenousAnchorSpec,
    MultiPatchSpec,
    ReferenceWindowSpec,
    WidthScheduleSpec,
    assess_adaptive_language,
    assess_multi_patch,
    fixed_residual_update,
    fixed_residual_view,
    lower_adaptive_language,
)


def _refuse(call, fragment: str = "") -> None:
    try:
        call()
        raise AssertionError("invalid adaptive-transformer input was accepted")
    except ValueError as exc:
        assert fragment in str(exc), (fragment, str(exc))


def _uniform(
    *,
    layers=3,
    width=16,
    context=16,
    repeats=1,
    residual_mode="pre_norm",
    window=None,
    anchor=None,
    gated=False,
):
    schedule = WidthScheduleSpec((width,) * layers, (2,) * layers, (2 * width,) * layers)
    return AdaptiveLanguageSpec(
        64,
        context,
        schedule,
        repeats=repeats,
        residual_mode=residual_mode,
        reference_window=window,
        anchor=anchor,
        gated_attention=gated,
    )


def test_geometric_width_schedule_is_symmetric_aligned_and_strict():
    schedule = WidthScheduleSpec.symmetric_u(
        layers=5, outer_width=32, inner_width=16, head_dim=8, ff_multiplier=3, quantum=8
    )
    assert schedule.widths == (32, 24, 16, 24, 32)
    assert schedule.heads == (4, 3, 2, 3, 4)
    assert schedule.ff_widths == (96, 72, 48, 72, 96)
    assert schedule.transition == "carry_forward"
    even = WidthScheduleSpec.symmetric_u(
        layers=6, outer_width=32, inner_width=16, head_dim=8, ff_multiplier=2, quantum=8
    )
    assert even.widths == (32, 24, 16, 16, 24, 32)
    _refuse(lambda: WidthScheduleSpec((16,), (3,), (32,)), "divisible")
    _refuse(lambda: WidthScheduleSpec((16,), (2,), (8,)), "at least")
    _refuse(lambda: WidthScheduleSpec((16,), (2,), (32,), "zero_pad"), "carry_forward")
    _refuse(
        lambda: WidthScheduleSpec.symmetric_u(layers=3, outer_width=30, inner_width=16, head_dim=8),
        "align",
    )
    _refuse(
        lambda: WidthScheduleSpec.symmetric_u(layers=2, outer_width=32, inner_width=16, head_dim=8),
        "fewer than three",
    )


def test_fixed_residual_carries_inactive_coordinates_across_a_bottleneck():
    initial = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    narrow = fixed_residual_view(initial, 1, 8, 4)
    assert narrow == [1.0, 2.0, 3.0, 4.0]
    residual = fixed_residual_update(initial, [9.0, 10.0, 11.0, 12.0], 1, 8, 4)
    assert residual == [9.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0, 8.0]
    assert fixed_residual_view(residual, 1, 8, 8) == residual
    _refuse(lambda: fixed_residual_view([1.0], 1, 8, 4), "length")
    _refuse(lambda: fixed_residual_update(initial, [1.0], 1, 8, 4), "active")


def test_reference_sliding_attention_has_global_prefix_and_bounded_tail():
    window = ReferenceWindowSpec(prefix_tokens=3, window_tokens=2)
    assert window.visible_keys(0, 8) == (0,)
    assert window.visible_keys(2, 8) == (0, 1, 2)
    assert window.visible_keys(3, 8) == (0, 1, 2, 3)
    assert window.visible_keys(7, 8) == (0, 1, 2, 6, 7)
    mask = window.additive_mask(8)
    assert len(mask) == 64 and mask[7 * 8 + 5] < 0 and mask[7 * 8 + 6] == 0.0
    assert window.visible_pairs(8) == sum(len(window.visible_keys(index, 8)) for index in range(8))
    assert window.visible_pairs(1 << 20) == (
        6 + 3 * ((1 << 20) - 3) + 3 + 2 * (((1 << 20) - 3) - 2)
    )
    assert window.cache_tokens(0) == 3
    assert window.cache_tokens(1_000) == 5
    _refuse(lambda: window.visible_keys(8, 8), "query")
    _refuse(lambda: window.additive_mask(8, masked=float("-inf")), "finite")
    _refuse(lambda: window.additive_mask(1025), "materialization")


def test_exogenous_anchor_contract_quarantines_unstudied_combinations():
    anchor = ExogenousAnchorSpec(mode="dynamic", granularity="elementwise", source_depth=1)
    spec = _uniform(layers=2, anchor=anchor, gated=True)
    assert spec.anchor.initial_base_coefficient == 1.0
    assert spec.anchor.coefficient_channels(16, 2) == 16
    assert spec.effective_layers == 2
    _refuse(lambda: _uniform(layers=2, anchor=anchor), "gated")
    _refuse(
        lambda: _uniform(
            layers=2, repeats=2, residual_mode="loop_deepnorm", anchor=anchor, gated=True
        ),
        "pre-norm",
    )
    variable = WidthScheduleSpec((16, 8, 16), (2, 1, 2), (32, 16, 32))
    _refuse(
        lambda: AdaptiveLanguageSpec(64, 16, variable, gated_attention=True, anchor=anchor),
        "fixed-width",
    )
    _refuse(
        lambda: _uniform(layers=1, anchor=ExogenousAnchorSpec(source_depth=1), gated=True),
        "precede",
    )


def test_loopdeepnorm_ties_parameters_but_prices_effective_depth():
    once = _uniform(layers=2, residual_mode="loop_deepnorm", repeats=1)
    thrice = _uniform(layers=2, residual_mode="loop_deepnorm", repeats=3)
    first = assess_adaptive_language(once, prefill_tokens=8)
    repeated = assess_adaptive_language(thrice, prefill_tokens=8)
    assert first.parameter_elements == repeated.parameter_elements
    assert repeated.effective_layers == 3 * first.effective_layers
    assert repeated.output_projection_flops == first.output_projection_flops
    assert repeated.prefill_flops - repeated.output_projection_flops == 3 * (
        first.prefill_flops - first.output_projection_flops
    )
    assert repeated.kv_cache_bytes == 3 * first.kv_cache_bytes
    assert math.isclose(thrice.loop_alpha, math.sqrt(12.0))
    assert math.isclose(thrice.loop_beta, 1.0 / math.sqrt(48.0))


def test_variable_width_and_reference_window_costs_are_honest_lower_bounds():
    schedule = WidthScheduleSpec.symmetric_u(
        layers=5, outer_width=32, inner_width=16, head_dim=8, quantum=8
    )
    variable = AdaptiveLanguageSpec(64, 32, schedule)
    full = assess_adaptive_language(variable, prefill_tokens=24, decode_context=24)
    assert full.classification == "analytic-lower-bound"
    assert full.prefill_flop_delta < 0
    assert full.width_transition_elements > 0

    dense = _uniform(layers=3, context=32)
    sliding = _uniform(layers=3, context=32, window=ReferenceWindowSpec(4, 6))
    dense_cost = assess_adaptive_language(dense, prefill_tokens=24, decode_context=24)
    sliding_cost = assess_adaptive_language(sliding, prefill_tokens=24, decode_context=24)
    assert sliding_cost.prefill_attention_pairs < dense_cost.prefill_attention_pairs
    assert sliding_cost.kv_cache_bytes < dense_cost.kv_cache_bytes
    assert sliding_cost.decode_flops_per_token < dense_cost.decode_flops_per_token


def test_adaptive_lowering_is_verified_complete_and_deterministic():
    schedule = WidthScheduleSpec.symmetric_u(
        layers=3, outer_width=24, inner_width=16, head_dim=8, quantum=8, ff_multiplier=2
    )
    spec = AdaptiveLanguageSpec(
        64,
        16,
        schedule,
        repeats=2,
        residual_mode="loop_deepnorm",
        reference_window=ReferenceWindowSpec(3, 4),
    )
    first = lower_adaptive_language(spec, sequence_length=10)
    second = lower_adaptive_language(spec, sequence_length=10)
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.streampack_data == second.streampack_data
    assert first.streampack.provenance_ok()
    assert len(first.realization.steps) == sum(len(phase.claims) for phase in first.module.phases)
    op_names = {claim.op for phase in first.module.phases for claim in phase.claims}
    assert "model.embedding.lookup" in op_names
    assert "model.head.logits" in op_names
    assert "model.width.carry_forward_view" in op_names
    assert "model.attention.reference_sliding" in op_names
    assert "model.loop.iteration_barrier" in op_names

    attention_projection = [
        claim
        for phase in first.module.phases
        for claim in phase.claims
        if claim.op == "model.attention.project"
    ]
    expected = []
    rows = 10
    for _repeat in range(spec.repeats):
        expected.extend(rows * 4 * width * width for width in spec.schedule.widths)
    assert [claim.count for claim in attention_projection] == expected


def test_multipatch_costs_coarse_attention_without_claiming_runtime_speed():
    spec = MultiPatchSpec(
        image_size=16,
        input_channels=3,
        hidden_width=16,
        heads=4,
        coarse_patch=4,
        fine_patch=2,
        coarse_layers=3,
        fine_layers=1,
        condition_tokens=2,
    )
    report = assess_multi_patch(spec)
    assert spec.coarse_spatial_tokens == 16 and spec.fine_spatial_tokens == 64
    assert report.attention_pairs < report.baseline_attention_pairs
    assert report.estimated_flops < report.baseline_flops
    assert 0 < report.flop_reduction_q20 < 1 << 20
    assert report.classification == "analytic-lower-bound"
    _refuse(lambda: MultiPatchSpec(16, 3, 18, 3, 4, 2, 1, 1), "divisible by four")
    _refuse(lambda: MultiPatchSpec(16, 3, 16, 4, 3, 2, 1, 1), "divisible")
