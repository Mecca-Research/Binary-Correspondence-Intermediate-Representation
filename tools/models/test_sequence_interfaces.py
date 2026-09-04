#!/usr/bin/env python3
"""Bounded CPU gate for sequence interfaces, tokenizer adaptation, and depth growth.

The gate is offline, one-threaded, and fixture-sized.  It imports no external source,
checkpoint, tokenizer, font, or dataset and performs no GPU work.  Two fresh runs must
produce the same timestamp-free report on the same host.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

import torch

from bcir.hosted.training import BytePairTokenizer
from bcir.hosted.training.contracts import StageTrainSpec
from bcir.hosted.training.sequence_interfaces import (
    ExpandedRowTable,
    HostedProgressiveLanguageModel,
    clipped_cross_tokenizer_opd_loss,
    train_progressive_growth_stage,
)
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
    decode_causal_series,
    encode_causal_series,
    lower_growth_stage,
    pareto_sequence_interfaces,
    plan_continued_bpe_expansion,
    project_aligned_log_probabilities,
    segment_unigram,
)

_Q20 = 1 << 20

RESEARCH_PROVENANCE = {
    "papers": {
        "breaking_the_tokenizer_barrier": {
            "arxiv": "2606.09456v1",
            "sha256": "d9d361add05fbeb9fd338966ecd0f99282033a3160eb7f4d35e6940402d2e18f",
        },
        "emergent_semantics": {
            "arxiv": "2507.04886v4",
            "sha256": "9676a04d619f09f35dace23906a7196c876153bffb447ad37a39a98923e74310",
        },
        "galaxys_guide": {
            "arxiv": "2606.25610v1",
            "sha256": "dccc20680ac120a8916cb1091433fb2ed02fda4eaaecc6d572a54ab577bfe87a",
        },
        "growing_transformers": {
            "arxiv": "2507.07129v3",
            "sha256": "b0aeb9afd97ad498e9f23a1c5970e1b493a5f442227c5956c4eae0590692e192",
        },
        "in_place_tokenizer_expansion": {
            "arxiv": "2607.15232v1",
            "sha256": "63f689a974302bbfc025b74b897d0b7aceb24a06be750e7117acc3cfa81eb918",
        },
        "less_is_more": {
            "arxiv": "2506.15138v2",
            "sha256": "a395f6aad4219a1447a32ef2edf6440f1d8b3d4d395f2deb2500a73ca2fa74e4",
        },
        "time_series_as_language": {
            "arxiv": "2606.09861v1",
            "sha256": "1777fcd9f81fce593a884e65d7b7bb7239777ccf410fadd69dc5598ba9f91143",
        },
    },
    "repositories": {
        "AVBochkov/Embeddings": {
            "commit": "4161521d9f88a9d7fe7aa146de673bdac3f4a1a5",
            "license": "Apache-2.0",
            "adoption": "audit only; no source, data, weights, or font assets imported",
        },
        "AVBochkov/PGT": {
            "commit": "4ed95fd272e126281352aea321d175cd00b9b381",
            "license": "Apache-2.0",
            "adoption": "paper-derived independent growth contract; no source or weights imported",
        },
    },
}


def _tokenizer_case() -> dict:
    corpus = (
        "hello world hello tokenizer world\n",
        "hello adaptive world hello tokenizer\n",
        "world tokenizer hello world adaptive\n",
    )
    source = BytePairTokenizer.train(corpus, vocab_size=266, min_frequency=2)
    target = BytePairTokenizer.train(corpus, vocab_size=270, min_frequency=2)
    if len(target.merges) <= len(source.merges):
        raise AssertionError("fixture corpus did not produce a strict tokenizer continuation")
    plan = plan_continued_bpe_expansion(
        source.merges, target.merges, base_vocab_size=260, special_token_ids=(0, 1, 2, 3)
    )
    source_tokens = sum(len(source.encode(text)) for text in corpus)
    target_tokens = sum(len(target.encode(text)) for text in corpus)
    units = sum(len(text) for text in corpus)
    cost = assess_tokenizer_expansion(
        plan,
        hidden_width=18,
        bytes_per_element=4,
        tied_head=False,
        source_tokens_per_unit_q20=source_tokens * _Q20 // units,
        target_tokens_per_unit_q20=target_tokens * _Q20 // units,
    )

    source_rows = torch.linspace(-1.0, 1.0, plan.source_vocab_size * 18, dtype=torch.float32).view(
        plan.source_vocab_size, 18
    )
    table = ExpandedRowTable(source_rows, plan)
    copied = table.source_rows.detach().clone()
    expected_first = source_rows[list(plan.new_rows[0].source_token_ids)].mean(dim=0)
    if not torch.equal(table.added_rows[0].detach(), expected_first):
        raise AssertionError("continued-BPE row was not initialized from its source mean")
    optimizer = torch.optim.SGD(table.parameters(), lr=0.1)
    selected = torch.tensor([plan.source_vocab_size], dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    table(selected).square().mean().backward()
    optimizer.step()
    if not torch.equal(table.source_rows.detach(), copied):
        raise AssertionError("stage-one tokenizer adaptation changed a copied source row")
    if table.added_rows.grad is None:
        raise AssertionError("stage-one tokenizer adaptation did not train the added row")
    table.begin_full_model_stage()
    if table.stage != 2 or not table.source_rows.requires_grad:
        raise AssertionError("tokenizer adaptation did not admit its explicit full-model stage")

    pieces = (
        UnigramPiece(0, b"h", -0.5),
        UnigramPiece(1, b"e", -0.5),
        UnigramPiece(2, b"l", -0.5),
        UnigramPiece(3, b"o", -0.5),
        UnigramPiece(4, b"he", -0.2),
        UnigramPiece(5, b"llo", -0.3),
    )
    if segment_unigram(b"hello", pieces) != (4, 5):
        raise AssertionError("bounded unigram fixture lost its maximum-likelihood segmentation")
    return {
        "source_tokenizer_sha256": plan.source_tokenizer_sha256,
        "target_tokenizer_sha256": plan.target_tokenizer_sha256,
        "expansion_plan_sha256": plan.plan_sha256,
        "source_vocab_size": plan.source_vocab_size,
        "target_vocab_size": plan.target_vocab_size,
        "new_rows": len(plan.new_rows),
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "added_parameter_bytes": cost.added_parameter_bytes,
        "source_head_macs_per_unit": cost.source_head_macs_per_unit,
        "target_head_macs_per_unit": cost.target_head_macs_per_unit,
        "copied_rows_frozen_in_stage_one": True,
    }


def _alignment_case() -> dict:
    student = (
        DecodedToken(10, b"the"),
        DecodedToken(11, b" "),
        DecodedToken(12, b"model"),
    )
    teacher = (
        DecodedToken(20, b"t"),
        DecodedToken(21, b"he "),
        DecodedToken(22, b"mo"),
        DecodedToken(23, b"del"),
    )
    chunks = align_decoded_tokens(student, teacher)
    projection = project_aligned_log_probabilities(
        chunks, (-0.8, -0.4, -1.2), (-0.3, -0.6, -0.5, -0.4)
    )
    current = torch.tensor((-0.75, -0.45, -1.1), dtype=torch.float32, requires_grad=True)
    old = torch.tensor((-0.8, -0.4, -1.2), dtype=torch.float32)
    loss = clipped_cross_tokenizer_opd_loss(current, old, projection)
    loss.backward()
    if current.grad is None or not torch.isfinite(current.grad).all():
        raise AssertionError("cross-tokenizer OPD fixture produced invalid gradients")
    for projected, target in zip(
        (
            sum(projection.projected_log_probabilities[row.student_start : row.student_end])
            for row in chunks
        ),
        projection.teacher_chunk_log_probabilities,
    ):
        if not math.isclose(projected, target, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("cross-tokenizer projection lost teacher chunk log mass")
    return {
        "chunks": len(chunks),
        "projection_sha256": projection.projection_sha256,
        "projected_log_probabilities": list(projection.projected_log_probabilities),
        "advantages": list(projection.advantages),
        "clipped_loss": float(loss.detach().item()),
        "finite_gradient": True,
    }


def _series_and_growth_case(seed: int) -> dict:
    codec = CausalSeriesCodecSpec(
        FiniteScalarQuantizerSpec((17,)),
        warmup_values=2,
        normalized_clip=4.0,
        scale_floor=1.0e-6,
        max_values=64,
    )
    series = (0.0, 0.5, 1.0, 0.25, -0.75, -1.0, -0.25, 0.75)
    encoding = encode_causal_series(series, codec)
    if decode_causal_series(encoding.tokens, codec) != encoding.reconstructed_values:
        raise AssertionError("causal series codec failed its exact token replay")
    for length in range(1, len(series) + 1):
        prefix = encode_causal_series(series[:length], codec)
        if prefix.tokens != encoding.tokens[: encoding.token_ends[length - 1]]:
            raise AssertionError("causal series tokenizer changed an earlier prefix")

    try:
        HostedProgressiveLanguageModel(
            FrozenTokenCodeSpec(vocab_size=2, code_bits=1, model_width=65536),
            heads=1,
            ff_width=65536,
            max_layers=64,
            context_length=1,
        )
    except ValueError as exc:
        if "parameter limit" not in str(exc):
            raise AssertionError(f"unexpected oversized-model refusal: {exc}") from exc
    else:
        raise AssertionError("oversized progressive model reached tensor allocation")

    torch.manual_seed(seed)
    interface = FrozenTokenCodeSpec(vocab_size=codec.vocabulary_size, code_bits=9, model_width=18)
    model = HostedProgressiveLanguageModel(
        interface, heads=3, ff_width=36, max_layers=2, context_length=32
    )
    active_budget = model.block_parameter_elements + model.head_parameter_elements
    growth = ProgressiveGrowthSpec(
        initial_layers=1,
        final_layers=2,
        growth_layers=1,
        block_parameter_elements=model.block_parameter_elements,
        head_parameter_elements=model.head_parameter_elements,
        fixed_interface_elements=model.fixed_interface_elements,
        interface_width=interface.model_width,
        active_parameter_budget=active_budget,
    )
    token_ids = torch.tensor([encoding.tokens[:-1]], dtype=torch.long)
    targets = torch.tensor([encoding.tokens[1:]], dtype=torch.long)
    events = []
    first = train_progressive_growth_stage(
        model,
        growth,
        0,
        token_ids,
        targets,
        StageTrainSpec("pretrain", steps=4, learning_rate=4.0e-3, weight_decay=0.0, seed=seed),
        events.append,
    )
    frozen_block = {
        name: value.detach().clone() for name, value in model.blocks[0].state_dict().items()
    }
    second = train_progressive_growth_stage(
        model,
        growth,
        1,
        token_ids,
        targets,
        StageTrainSpec("pretrain", steps=4, learning_rate=4.0e-3, weight_decay=0.0, seed=seed + 1),
        events.append,
    )
    if not first.final_loss < first.initial_loss or not second.final_loss < second.initial_loss:
        raise AssertionError("tiny progressive stages did not reduce repeated-batch loss")
    if any(
        not torch.equal(value, frozen_block[name])
        for name, value in model.blocks[0].state_dict().items()
    ):
        raise AssertionError("an earlier dense block changed during the next growth stage")
    if len(events) != 8 or [event.step for event in events] != [1, 2, 3, 4] * 2:
        raise AssertionError("progressive training telemetry is incomplete or unordered")
    lowered = lower_growth_stage(growth, 1, sequence_length=token_ids.shape[1])
    if not lowered.streampack.provenance_ok():
        raise AssertionError("progressive growth produced an untraced StreamPack")
    return {
        "codec_sha256": codec.digest,
        "encoding_sha256": encoding.encoding_sha256,
        "series_values": len(series),
        "series_tokens": len(encoding.tokens),
        "maximum_absolute_error": encoding.maximum_absolute_error,
        "interface_sha256": interface.digest,
        "interface_rank_upper_bound": interface.rank_upper_bound,
        "growth_sha256": growth.digest,
        "active_parameter_budget": growth.active_parameter_budget,
        "stage_losses": [
            [first.initial_loss, first.final_loss],
            [second.initial_loss, second.final_loss],
        ],
        "model_sha256": second.model_sha256,
        "frozen_block_unchanged": True,
        "plan_sha256": lowered.artifact_sha256,
        "streampack_bytes": len(lowered.streampack_data),
    }


def _evidence_case() -> dict:
    corpus = "0" * 64
    corpus_sha = __import__("hashlib").sha256(corpus.encode()).hexdigest()

    def row(name, tokens, artifact, reconstruction, probe, exact):
        return SequenceInterfaceEvidence(
            name,
            __import__("hashlib").sha256(name.encode()).hexdigest(),
            corpus_sha,
            64,
            tokens,
            artifact,
            1000,
            1000,
            reconstruction,
            probe,
            probe,
            probe,
            exact,
        )

    evidence = (
        row("affine", 64, 512, 2000, 700_000, False),
        row("discrete", 32, 1024, 4000, 800_000, False),
        row("dominated", 64, 2048, 8000, 600_000, False),
        row("lossless-byte", 64, 256, 0, 650_000, True),
    )
    frontier = pareto_sequence_interfaces(evidence)
    if "dominated" in {item.name for item in frontier}:
        raise AssertionError("multi-objective tokenizer gate retained a dominated trial")
    return {
        "evidence_sha256": [item.digest for item in evidence],
        "pareto_front": [item.name for item in frontier],
        "single_aggregate_score_used": False,
    }


def _run_once() -> dict:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return {
        "schema": "bcir.sequence_interface_gate.v1",
        "research_provenance": RESEARCH_PROVENANCE,
        "tokenizer_expansion": _tokenizer_case(),
        "cross_tokenizer_alignment": _alignment_case(),
        "series_and_growth": _series_and_growth_case(2607),
        "multiobjective_evidence": _evidence_case(),
        "boundaries": {
            "external_source_imported": False,
            "external_assets_imported": False,
            "gpu_used": False,
            "large_training_run": False,
            "learned_values_affect_bcir_legality": False,
            "glyph_rendering_and_pca": "hosted-roadmap-only",
            "lora_execution": "contract-only",
        },
    }


def run_gate(output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"sequence-interface gate output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    first = _run_once()
    # PyTorch permits setting inter-op threads only before parallel work starts.  The
    # second run reuses that fixed setting but reconstructs every model and optimizer.
    second = {
        "schema": "bcir.sequence_interface_gate.v1",
        "research_provenance": RESEARCH_PROVENANCE,
        "tokenizer_expansion": _tokenizer_case(),
        "cross_tokenizer_alignment": _alignment_case(),
        "series_and_growth": _series_and_growth_case(2607),
        "multiobjective_evidence": _evidence_case(),
        "boundaries": first["boundaries"],
    }
    if first != second:
        raise AssertionError("sequence-interface gate is not deterministic across fresh runs")
    text = json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    path = output_dir / "report.json"
    temporary = output_dir / "report.json.part"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return first


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "build" / "sequence-interface-gate"))
    arguments = parser.parse_args()
    output = Path(arguments.output_dir)
    if output.exists():
        import shutil

        shutil.rmtree(output)
    with tempfile.TemporaryDirectory(prefix="bcir-sequence-interface-"):
        report = run_gate(output)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
