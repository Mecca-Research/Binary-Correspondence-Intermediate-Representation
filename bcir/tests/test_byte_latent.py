"""Dependency-free gates for byte-native model contracts and lowering."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from bcir.kbcir.byte_latent import (
    ByteDiffusionSpec,
    ByteIngestProfile,
    ByteLatentSpec,
    BytePatchSpec,
    ByteSpeculationSpec,
    ByteVocabularySpec,
    LearnedPatchPolicySpec,
    MambaByteSpec,
    PatchLayout,
    TensorShape,
    assess_byte_latent,
    assess_mambabyte,
    choose_byte_ingest,
    decode_raw_bytes,
    discounted_batch_advantages,
    encode_raw_sequence,
    lower_byte_latent,
    plan_byteified_transplant,
    verify_byte_draft,
)
from bcir.kbcir.cost import TargetProfile


def _refuse(call, fragment: str = "") -> None:
    try:
        call()
        raise AssertionError("invalid byte-model input was accepted")
    except (UnicodeDecodeError, ValueError) as exc:
        assert fragment in str(exc), (fragment, str(exc))


def _spec(
    *,
    patcher=None,
    generation_mode="autoregressive",
    diffusion=None,
    speculation=None,
    backbone="transformer",
    learned_policy=None,
):
    return ByteLatentSpec(
        context_bytes=64,
        patcher=patcher or BytePatchSpec(stride=4, max_patch_bytes=8),
        local_width=16,
        global_width=24,
        local_heads=2,
        global_heads=3,
        local_encoder_layers=1,
        global_layers=1,
        local_decoder_layers=1,
        local_ff_width=32,
        global_ff_width=48,
        local_window_bytes=16,
        hash_ngram_sizes=(3, 4),
        hash_buckets=32,
        local_backbone=backbone,
        ssm_state_size=4,
        generation_mode=generation_mode,
        diffusion=diffusion,
        speculation=speculation,
        learned_policy=learned_policy,
    )


def test_raw_utf8_is_lossless_unormalized_and_references_exact_byte_spans():
    text = "Aée\u0301🙂"
    sequence = encode_raw_sequence(text)
    assert sequence.payload == text.encode("utf-8")
    assert decode_raw_bytes(sequence.payload) == text
    assert len(sequence.references) == len(text)
    assert tuple(item.byte_start for item in sequence.references) == (0, 1, 3, 4, 6)
    assert sequence.references[-1].utf8_hex == "f09f9982"
    assert sequence.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert encode_raw_sequence("é").payload != encode_raw_sequence("e\u0301").payload

    arbitrary = encode_raw_sequence(b"\xff\x00raw")
    assert arbitrary.references == ()
    assert decode_raw_bytes(arbitrary.payload, errors="replace").startswith("�")
    _refuse(lambda: decode_raw_bytes(arbitrary.payload), "decode")
    _refuse(lambda: decode_raw_bytes([256]), "[0, 255]")
    _refuse(lambda: ByteVocabularySpec(size=261), "exactly")
    oversized = b"x" * ((1 << 20) + 1)
    _refuse(lambda: encode_raw_sequence(oversized), "exceeds")
    _refuse(lambda: decode_raw_bytes(oversized), "exceeds")


def test_patch_layouts_are_bounded_incremental_and_content_addressed():
    stride = BytePatchSpec(stride=3, min_patch_bytes=2, max_patch_bytes=4)
    layout = stride.layout(10)
    assert layout.starts == (0, 3, 6, 9)
    assert layout.lengths == (3, 3, 3, 1)
    assert layout.byte_to_patch() == (0, 0, 0, 1, 1, 1, 2, 2, 2, 3)

    scores = (100, 200, 7000, 100, 200, 7100, 100, 200)
    entropy = BytePatchSpec(
        method="entropy_global",
        min_patch_bytes=2,
        max_patch_bytes=4,
        entropy_threshold_millibits=5000,
    )
    first = entropy.layout(len(scores), entropy_millibits=scores)
    assert first.starts == (0, 2, 5)
    assert entropy.layout(len(scores), entropy_millibits=scores).digest == first.digest
    prefix = entropy.layout(6, entropy_millibits=scores[:6])
    assert tuple(start for start in first.starts if start < 6) == prefix.starts

    rising = BytePatchSpec(
        method="entropy_monotonic",
        min_patch_bytes=1,
        max_patch_bytes=5,
        entropy_delta_millibits=500,
    )
    assert rising.layout(6, entropy_millibits=(100, 200, 900, 1000, 1700, 1800)).starts == (0, 2, 4)
    learned = BytePatchSpec(
        method="learned", min_patch_bytes=2, max_patch_bytes=4, learned_threshold_q20=1 << 19
    )
    assert learned.layout(7, learned_scores_q20=(0, 0, 1 << 20, 0, 0, 0, 0)).starts == (0, 2, 6)
    _refuse(lambda: entropy.layout(4, entropy_millibits=(1, 2)), "one causal")


def test_diffusion_corruption_and_selection_are_deterministic_and_progress_bounded():
    vocabulary = ByteVocabularySpec()
    confidence = ByteDiffusionSpec(
        block_size=4, max_steps=4, unmask_strategy="confidence", confidence_threshold_q20=900_000
    )
    first = confidence.corrupt((1, 2, 3, 4), vocabulary, mask_probability_q20=1, seed=17)
    assert first == confidence.corrupt((1, 2, 3, 4), vocabulary, mask_probability_q20=1, seed=17)
    assert vocabulary.mask_id in first
    assert confidence.select_positions(
        (0, 2), confidences_q20=(10, 0, 20, 0), entropies_millibits=(5, 0, 4, 0)
    ) == (2,)

    entropy = ByteDiffusionSpec(
        block_size=4, max_steps=2, unmask_strategy="entropy_bounded", entropy_budget_millibits=7
    )
    assert entropy.select_positions(
        (0, 1, 2), confidences_q20=(0, 0, 0), entropies_millibits=(5, 3, 4)
    ) == (1, 2)
    _refuse(lambda: confidence.corrupt((), vocabulary, mask_probability_q20=1, seed=1), "malformed")
    _refuse(
        lambda: confidence.corrupt(
            (1, vocabulary.mask_id), vocabulary, mask_probability_q20=1, seed=1
        ),
        "malformed",
    )
    _refuse(
        lambda: confidence.select_positions((0, 0), confidences_q20=(1,), entropies_millibits=(1,)),
        "distinct",
    )
    _refuse(
        lambda: confidence.select_positions((0,), confidences_q20=(1, 2), entropies_millibits=(1,)),
        "cover",
    )


def test_exact_byte_draft_verification_preserves_greedy_target_semantics():
    mismatch = verify_byte_draft((10, 11, 12), (10, 99, 12), free_prediction=7, vocabulary_size=260)
    assert mismatch.accepted == (10,)
    assert mismatch.emitted == (10, 99)
    assert mismatch.rejected == 2 and not mismatch.matched_all
    complete = verify_byte_draft((10, 11), (10, 11), free_prediction=12, vocabulary_size=260)
    assert complete.emitted == (10, 11, 12)
    assert complete.matched_all and complete.rejected == 0
    _refuse(lambda: verify_byte_draft((), (), free_prediction=0, vocabulary_size=260), "nonzero")


def test_learned_boundary_advantages_are_future_discounted_and_batch_centered():
    advantages = discounted_batch_advantages(((1.0, 2.0), (3.0, 4.0)), 0.5)
    assert advantages == ((-1.5, -1.0), (1.5, 1.0))
    assert all(abs(sum(row[index] for row in advantages)) < 1.0e-12 for index in range(2))
    _refuse(lambda: discounted_batch_advantages(((1.0,), (1.0, 2.0)), 0.9), "rectangular")
    _refuse(lambda: discounted_batch_advantages((range((1 << 20) + 1),), 0.9), "bounded")


def test_byte_model_and_mambabyte_costs_are_exact_sizes_with_honest_lower_bounds():
    transformer = _spec()
    report = assess_byte_latent(transformer, byte_count=12, patch_count=3, batch=2)
    assert report.parameter_elements > 0 and report.total_flops_lower_bound > 0
    assert report.recurrent_state_bytes == 0
    assert report.classification == "exact-size+analytic-lower-bound"

    hybrid = _spec(backbone="selective_ssm")
    hybrid_report = assess_byte_latent(hybrid, byte_count=12, patch_count=3)
    assert hybrid_report.recurrent_state_bytes > 0
    assert hybrid_report.local_cache_bytes == 0

    mamba = MambaByteSpec(
        context_bytes=64, width=16, layers=2, state_size=4, expansion=2, conv_width=4
    )
    mamba_report = assess_mambabyte(mamba, sequence_bytes=12)
    assert mamba_report.parameter_elements > 0
    assert mamba_report.recurrent_state_bytes == 2 * 32 * 4 * 2
    assert mamba_report.convolution_state_bytes == 2 * 32 * 3 * 2
    _refuse(lambda: assess_mambabyte(mamba, sequence_bytes=65), "sequence_bytes")
    _refuse(lambda: _spec(generation_mode="diffusion"), "require diffusion")
    # Joint AR + block-denoising training is valid even when deployment remains AR.
    assert _spec(diffusion=ByteDiffusionSpec(block_size=4, max_steps=2)).diffusion

    diffusion = ByteDiffusionSpec(block_size=4, max_steps=2)
    one_layer = _spec(diffusion=diffusion)
    two_layers = replace(one_layer, local_decoder_layers=2)
    one_report = assess_byte_latent(one_layer, byte_count=8, patch_count=2)
    two_report = assess_byte_latent(two_layers, byte_count=8, patch_count=2)
    local, ff, block, steps = 16, 32, 4, 2
    decoder_block = 2 * block * (4 * local * local + 3 * local * ff) + 4 * local * block * block
    vocabulary_projection = 2 * block * local * one_layer.vocabulary.size
    assert one_report.diffusion_flops_lower_bound == steps * (decoder_block + vocabulary_projection)
    assert two_report.diffusion_flops_lower_bound == steps * (
        2 * decoder_block + vocabulary_projection
    )


def test_byteified_global_transplant_is_shape_exact_and_generator_safe():
    spec = _spec()
    source = (TensorShape("layer.0.q", (24, 24)),)
    target = (
        TensorShape("global_blocks.0.attention.q_proj.weight", (24, 24)),
        TensorShape("embed_bytes.weight", (260, 16)),
    )
    source_sha = "a" * 64
    plan = plan_byteified_transplant(
        spec,
        source_sha256=source_sha,
        source_shapes=(item for item in source),
        target_shapes=(item for item in target),
        mappings={"global_blocks.0.attention.q_proj.weight": "layer.0.q"},
    )
    assert plan.reused_elements == 24 * 24
    assert plan.initialized_elements == 260 * 16
    assert plan.global_learning_rate_q20 == (1 << 20) // 10
    _refuse(
        lambda: plan_byteified_transplant(
            spec,
            source_sha256=source_sha,
            source_shapes=source,
            target_shapes=(TensorShape("global_blocks.0.attention.q_proj.weight", (24, 16)),),
            mappings={"global_blocks.0.attention.q_proj.weight": "layer.0.q"},
        ),
        "exactly",
    )
    _refuse(
        lambda: plan_byteified_transplant(
            spec,
            source_sha256=source_sha,
            source_shapes=source,
            target_shapes=target,
            mappings={"embed_bytes.weight": "layer.0.q"},
        ),
        "only global_blocks",
    )
    _refuse(
        lambda: plan_byteified_transplant(
            spec,
            source_sha256=source_sha,
            source_shapes=source,
            target_shapes=target,
            mappings={1: "layer.0.q"},
        ),
        "only global_blocks",
    )
    incomplete_target = target + (TensorShape("global_blocks.0.attention.k_proj.weight", (24, 24)),)
    _refuse(
        lambda: plan_byteified_transplant(
            spec,
            source_sha256=source_sha,
            source_shapes=source,
            target_shapes=incomplete_target,
            mappings={"global_blocks.0.attention.q_proj.weight": "layer.0.q"},
        ),
        "cover",
    )


def test_measured_ingest_selects_device_only_with_parity_capacity_and_advantage():
    digest = "b" * 64
    fast = ByteIngestProfile(
        digest,
        host_ns_per_byte_q20=10 << 20,
        device_ns_per_byte_q20=1 << 20,
        device_launch_ns=100,
        device_pool_bytes=4096,
        max_chunk_bytes=1024,
        exact_roundtrip=True,
    )
    assert choose_byte_ingest(fast, 10).provider == "host"
    selected = choose_byte_ingest(fast, 1024)
    assert selected.provider == "device" and selected.predicted_ns == 1124
    no_parity = ByteIngestProfile(digest, 10 << 20, 1 << 20, 1, 4096, 1024, False)
    assert choose_byte_ingest(no_parity, 1024).provider == "host"
    assert choose_byte_ingest(fast, 4097).provider == "host"
    huge_chunk = ByteIngestProfile(digest, 10 << 20, 1 << 20, 0, (1 << 63) - 1, (1 << 63) - 1, True)
    assert choose_byte_ingest(huge_chunk, 1024).provider == "device"


def test_byte_latent_lowering_is_complete_verified_and_deterministic():
    diffusion = ByteDiffusionSpec(block_size=4, max_steps=2)
    speculation = ByteSpeculationSpec(draft_bytes=3)
    spec = _spec(generation_mode="diffusion_verify", diffusion=diffusion, speculation=speculation)
    layout = spec.patcher.layout(12)
    target = TargetProfile.x86_avx2()
    first = lower_byte_latent(spec, layout, batch=2, target=target)
    second = lower_byte_latent(spec, layout, batch=2, target=target)
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.streampack_data == second.streampack_data
    assert first.streampack.provenance_ok()
    assert len(first.realization.steps) == len(first.module.phases)
    operations = {claim.op for phase in first.module.phases for claim in phase.claims}
    assert operations == {
        "model.byte.ingest.exact",
        "model.byte.patch.stride",
        "model.byte.encoder.local_attention",
        "model.byte.patch.cross_attention",
        "model.byte.global.causal_transformer",
        "model.byte.decoder.causal_cross_attention",
        "model.byte.decoder.block_diffusion",
        "model.byte.verify.exact_greedy",
    }
    verifier = next(
        claim
        for phase in first.module.phases
        for claim in phase.claims
        if claim.op == "model.byte.verify.exact_greedy"
    )
    assert verifier.rd == (1, 7)
    assert verifier.wr == (9,)
    assert verifier.count == 2 * (speculation.draft_bytes + 1)
    verified = first.module.resources[9]
    assert verified.name == "verified_bytes"
    assert verified.shape == (2 * (speculation.draft_bytes + 1),)
    full_precision = lower_byte_latent(spec, layout, batch=2, bytes_per_element=4, target=target)
    assert full_precision.module.resources[3].elem_bytes == 4
    assert full_precision.report.local_cache_bytes == 2 * first.report.local_cache_bytes
    malformed = PatchLayout(12, (0, 9, 11), "stride", "c" * 64)
    _refuse(lambda: lower_byte_latent(spec, malformed, target=target), "bounds")


def test_learned_and_selective_scan_lowering_paths_are_explicit():
    patcher = BytePatchSpec(
        method="learned",
        min_patch_bytes=2,
        max_patch_bytes=4,
        learned_threshold_q20=1 << 19,
        learned_window=1,
    )
    spec = _spec(patcher=patcher, backbone="selective_ssm", learned_policy=LearnedPatchPolicySpec())
    layout = patcher.layout(8, learned_scores_q20=(1 << 20, 0, 1 << 20, 0, 0, 0, 1 << 20, 0))
    lowered = lower_byte_latent(spec, layout, target=TargetProfile.x86_avx2())
    operations = {claim.op for phase in lowered.module.phases for claim in phase.claims}
    assert "model.byte.patch.learned" in operations
    assert "model.byte.encoder.selective_scan" in operations
