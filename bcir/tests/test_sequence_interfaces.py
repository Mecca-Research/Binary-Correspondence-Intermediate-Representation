"""Deterministic contracts for tokenizer adaptation and progressive model growth."""
from __future__ import annotations

import hashlib
import math
from dataclasses import replace

from bcir.kbcir.sequence_interfaces import (
    CausalSeriesCodecSpec,
    DecodedToken,
    FiniteScalarQuantizerSpec,
    FrozenTokenCodeSpec,
    ProgressiveGrowthSpec,
    SequenceInterfaceEvidence,
    UnigramPiece,
    align_decoded_tokens,
    assess_tokenizer_expansion,
    canonical_substring_candidate,
    decode_causal_series,
    encode_causal_series,
    lower_growth_stage,
    pareto_sequence_interfaces,
    plan_continued_bpe_expansion,
    project_aligned_log_probabilities,
    segment_unigram,
    thunder_candidate_score,
)

_HASH_A = hashlib.sha256(b"sequence-interface-a").hexdigest()
_HASH_B = hashlib.sha256(b"sequence-interface-b").hexdigest()
_Q20 = 1 << 20


def _refuse(call, fragment: str = "") -> None:
    try:
        call()
    except ValueError as exc:
        if fragment and fragment not in str(exc):
            raise AssertionError(f"unexpected refusal: {exc}") from exc
    else:
        raise AssertionError("invalid sequence-interface input was accepted")


def test_cross_tokenizer_alignment_is_byte_exact_complete_and_minimal():
    student = (
        DecodedToken(10, b"the"), DecodedToken(11, b" "),
        DecodedToken(12, b"cat"), DecodedToken(13, b"!"),
    )
    teacher = (
        DecodedToken(20, b"t"), DecodedToken(21, b"he "),
        DecodedToken(22, b"cat"), DecodedToken(23, b"!"),
    )
    chunks = align_decoded_tokens(student, teacher)
    assert [(row.student_start, row.student_end,
             row.teacher_start, row.teacher_end, row.byte_length) for row in chunks] == [
        (0, 2, 0, 2, 4), (2, 3, 2, 3, 3), (3, 4, 3, 4, 1)]
    assert chunks[0].payload_sha256 == hashlib.sha256(b"the ").hexdigest()
    assert chunks == align_decoded_tokens(student, teacher)

    _refuse(lambda: DecodedToken(0, b""), "nonempty")
    _refuse(lambda: align_decoded_tokens(
        (DecodedToken(0, b"a"),), (DecodedToken(0, b"b"),)), "disagree")
    _refuse(lambda: align_decoded_tokens(
        (DecodedToken(0, b"a"),), (DecodedToken(0, b"ab"),)), "lengths")
    _refuse(lambda: align_decoded_tokens(student, teacher, max_bytes=7), "max_bytes")


def test_cross_tokenizer_projection_conserves_every_teacher_chunk():
    chunks = align_decoded_tokens(
        (DecodedToken(0, b"ab"), DecodedToken(1, b"c")),
        (DecodedToken(2, b"a"), DecodedToken(3, b"bc")))
    projection = project_aligned_log_probabilities(
        chunks, (-1.0, -2.0), (-0.75, -0.75))
    assert len(chunks) == 1
    assert math.isclose(sum(projection.projected_log_probabilities), -1.5,
                        rel_tol=0.0, abs_tol=1e-12)
    assert projection.projected_log_probabilities == (-0.5, -1.0)
    assert projection.advantages == (0.5, 1.0)
    assert projection == project_aligned_log_probabilities(
        chunks, (-1.0, -2.0), (-0.75, -0.75))

    _refuse(lambda: project_aligned_log_probabilities(chunks, (0.0, 0.0), (-1.0, -1.0)),
            "zero student")
    _refuse(lambda: project_aligned_log_probabilities(chunks, (-1.0,), (-1.0, -1.0)),
            "cover")
    _refuse(lambda: project_aligned_log_probabilities(chunks, iter((-1.0, -2.0)),
                                                       (-1.0, -1.0)), "bounded")
    _refuse(lambda: project_aligned_log_probabilities(chunks, (-1.0, float("nan")),
                                                       (-1.0, -1.0)), "valid range")


def test_bounded_substring_candidates_and_unigram_segmentation_are_exact():
    assert canonical_substring_candidate(b"hello") == b"hello"
    assert canonical_substring_candidate(b"hello world") == b"hello "
    assert canonical_substring_candidate(b"\xff") == b"\xff"
    _refuse(lambda: canonical_substring_candidate(b"a\xff"), "UTF-8")
    _refuse(lambda: UnigramPiece(9, b"hello world", -0.1), "unfinished")
    assert thunder_candidate_score(-0.25, 0.125) == -0.375

    pieces = (
        UnigramPiece(0, b"a", -0.4), UnigramPiece(1, b"b", -0.4),
        UnigramPiece(2, b"c", -0.4), UnigramPiece(3, b"ab", -0.3),
        UnigramPiece(4, b"bc", -0.2),
    )
    assert segment_unigram(b"abc", pieces) == (0, 4)
    assert segment_unigram(b"a" * 8192, (UnigramPiece(8, b"a", -0.1),)) == (8,) * 8192
    _refuse(lambda: segment_unigram(b"abd", pieces), "cannot exactly")
    _refuse(lambda: segment_unigram(b"abc", pieces + (UnigramPiece(0, b"z", -1.0),)),
            "unique")
    _refuse(lambda: segment_unigram(b"abc", pieces, max_pieces="many"), "integer")


def test_continued_bpe_expansion_proves_rows_and_prices_decode_per_unit():
    source = ((0, 1, 4), (4, 2, 5))
    target = source + ((5, 3, 6), (6, 1, 7))
    plan = plan_continued_bpe_expansion(
        source, target, base_vocab_size=4, special_token_ids=(0,))
    assert plan.source_vocab_size == 6
    assert plan.target_vocab_size == 8
    assert plan.new_rows[0].source_token_ids == (5, 3)
    assert plan.new_rows[1].source_token_ids == (5, 3, 1)
    assert plan == plan_continued_bpe_expansion(
        source, target, base_vocab_size=4, special_token_ids=(0,))
    _refuse(lambda: replace(plan, plan_sha256=_HASH_A), "digest")

    cost = assess_tokenizer_expansion(
        plan, hidden_width=16, bytes_per_element=2, tied_head=False,
        source_tokens_per_unit_q20=2 * _Q20,
        target_tokens_per_unit_q20=_Q20)
    assert cost.added_parameter_bytes == 2 * 16 * 2 * 2
    assert cost.additional_head_macs_per_token == 32
    assert cost.decode_compute_improves
    _refuse(lambda: plan_continued_bpe_expansion(
        source, ((0, 1, 4), (4, 3, 5), (5, 2, 6)), base_vocab_size=4),
        "continuation")
    _refuse(lambda: plan_continued_bpe_expansion(
        source, source + ((9, 1, 6),), base_vocab_size=4), "future")


def _evidence(name: str, *, tokens: int, artifact: int, probe: int,
              error: int = 0, exact: bool = True) -> SequenceInterfaceEvidence:
    return SequenceInterfaceEvidence(
        name, hashlib.sha256(name.encode()).hexdigest(), _HASH_A,
        source_units=100, token_count=tokens, artifact_bytes=artifact,
        encode_ns=1000, decode_ns=1000, reconstruction_error_q20=error,
        linear_probe_q20=probe, nonlinear_probe_q20=probe,
        domain_probe_q20=probe, exact_roundtrip=exact)


def test_sequence_interface_evidence_keeps_a_multiobjective_pareto_front():
    compact = _evidence("compact", tokens=50, artifact=10, probe=600_000)
    semantic = _evidence("semantic", tokens=70, artifact=20, probe=800_000)
    dominated = _evidence("dominated", tokens=80, artifact=30, probe=500_000)
    frontier = pareto_sequence_interfaces((dominated, semantic, compact))
    assert tuple(row.name for row in frontier) == ("compact", "semantic")
    assert compact.tokens_per_unit_q20 == _Q20 // 2
    _refuse(lambda: pareto_sequence_interfaces((compact, compact)), "duplicate")
    _refuse(lambda: pareto_sequence_interfaces([compact] * 4097), "bounded")
    _refuse(lambda: SequenceInterfaceEvidence(
        "bad", _HASH_A, _HASH_B, 1, 1, 1, 1, 1, 0,
        _Q20 + 1, 0, 0, True), "integer")


def test_causal_fsq_series_codec_is_prefix_stable_and_preserves_float32_anchors():
    quantizer = FiniteScalarQuantizerSpec((17,))
    spec = CausalSeriesCodecSpec(quantizer, warmup_values=2,
                                 normalized_clip=4.0, scale_floor=1.0e-6)
    source = (1.25, -2.5, 0.75, 3.0, -1.0, 2.0)
    full = encode_causal_series(source, spec)
    assert full.reconstructed_values == decode_causal_series(full.tokens, spec)
    assert full.reconstructed_values[:2] == source[:2]
    for prefix in range(1, len(source) + 1):
        encoded = encode_causal_series(source[:prefix], spec)
        assert encoded.tokens == full.tokens[:full.token_ends[prefix - 1]]
        assert len(encoded.tokens) == spec.prefix_token_count(prefix)
    long_source = tuple(float(index % 17 - 8) for index in range(4096))
    long_encoding = encode_causal_series(long_source, spec)
    assert decode_causal_series(long_encoding.tokens, spec) == \
        long_encoding.reconstructed_values
    assert quantizer.decode(quantizer.encode((0.0,))) == (0.0,)
    assert quantizer.decode(quantizer.encode((-10.0,))) == (-1.0,)
    assert quantizer.decode(quantizer.encode((10.0,))) == (1.0,)

    _refuse(lambda: FiniteScalarQuantizerSpec((8,)), "odd")
    _refuse(lambda: decode_causal_series(full.tokens[:-1], spec), "truncated")
    corrupted = list(full.tokens); corrupted[0] = 1
    _refuse(lambda: decode_causal_series(corrupted, spec), "mistagged")
    _refuse(lambda: encode_causal_series((1.0, float("inf")), spec), "valid range")


def test_fixed_binary_interface_and_growth_schedule_are_bounded_and_verified():
    interface = FrozenTokenCodeSpec(vocab_size=32, code_bits=8, model_width=16)
    assert interface.code(3)[:8] == (1.0, 1.0, -1.0, -1.0,
                                           -1.0, -1.0, -1.0, -1.0)
    assert interface.code(3)[8:] == interface.code(3)[:8]
    assert interface.rank_upper_bound == 8
    _refuse(lambda: FrozenTokenCodeSpec(vocab_size=257, code_bits=8, model_width=16),
            "uniquely")

    growth = ProgressiveGrowthSpec(
        initial_layers=1, final_layers=3, growth_layers=1,
        block_parameter_elements=100, head_parameter_elements=20,
        fixed_interface_elements=512, interface_width=16,
        active_parameter_budget=160)
    stages = growth.stages()
    assert tuple(stage.total_layers for stage in stages) == (1, 2, 3)
    assert tuple(stage.trainable_blocks for stage in stages) == ((0,), (1,), (2,))
    assert all(stage.active_parameter_elements == 120 for stage in stages)
    assert growth.inference_elements == 832

    lowered = lower_growth_stage(growth, 2, sequence_length=4, batch=2)
    again = lower_growth_stage(growth, 2, sequence_length=4, batch=2)
    assert lowered.artifact_sha256 == again.artifact_sha256
    assert lowered.streampack_data == again.streampack_data
    assert lowered.streampack.provenance_ok()
    ops = [claim.op for phase in lowered.module.phases for claim in phase.claims]
    assert ops == [
        "model.interface.fixed_lookup", "model.growth.forward.frozen",
        "model.growth.forward.new_blocks", "model.growth.backward.active_only",
        "model.growth.optimizer.update", "model.growth.stage.freeze_boundary"]

    with_lora = ProgressiveGrowthSpec(
        initial_layers=1, final_layers=2, growth_layers=1,
        block_parameter_elements=100, head_parameter_elements=20,
        fixed_interface_elements=512, interface_width=16, active_parameter_budget=160,
        lora_rank=2, lora_elements_per_rank_per_layer=20)
    assert tuple(stage.kind for stage in with_lora.stages()) == (
        "dense_growth", "dense_growth", "lora_readjustment")
    assert with_lora.stages()[-1].active_parameter_elements == 80
    lora_lowered = lower_growth_stage(with_lora, 2, sequence_length=2)
    assert any(claim.op == "model.growth.forward.lora_adapters"
               for phase in lora_lowered.module.phases for claim in phase.claims)
    assert lora_lowered.module.resources[3].shape == (220,)
    assert lora_lowered.module.resources[4].shape == (80,)

    _refuse(lambda: ProgressiveGrowthSpec(
        1, 2, 1, 100, 20, 512, 16, 119), "budget")
    _refuse(lambda: ProgressiveGrowthSpec(
        1, 2, 1, 100, 20, 512, 16, 160, lora_rank=1), "together")
