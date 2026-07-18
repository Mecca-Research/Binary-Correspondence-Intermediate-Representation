"""Dependency-free gates for the hosted training pipeline contracts."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from bcir.hosted.training import (ArtifactFile, BytePairTokenizer, DataPreparationSpec,
                                  OfflineComputeAdapter, PPOExample, PreferenceExample,
                                  RawDocument, ReasoningCandidate, ReasoningExample,
                                  RecordedTeacherProvider, RemoteTrainingBundle,
                                  RemoteTrainingResult, SFTExample, SearchBudget,
                                  SmallModelSpec, StageTrainEvent, StageTrainSpec, TeacherRequest,
                                  TeacherResponse, TrainingPipelineLedger, PipelineStageRecord,
                                  Verification, read_pipeline_ledger,
                                  bounded_reasoning_search, prepare_corpus,
                                  relational_embedding_targets, token_source_from_corpus,
                                  write_pipeline_ledger, write_prepared_corpus)

_ROOT = Path(__file__).resolve().parents[2]
_HASH_A = hashlib.sha256(b"a").hexdigest()
_HASH_B = hashlib.sha256(b"b").hexdigest()


def _must_refuse(call, fragment: str = "") -> None:
    try:
        call()
        raise AssertionError("invalid training-pipeline input was accepted")
    except (ValueError, LookupError, RuntimeError, FileExistsError) as exc:
        assert fragment in str(exc), (fragment, str(exc))


def _documents():
    return [
        RawDocument("doc-b", "alpha beta alpha beta", "fixture", "CC0-1.0", _HASH_A),
        RawDocument("doc-a", "gamma\r\ndelta\x00", "fixture", "CC0-1.0", _HASH_A),
        RawDocument("doc-copy", "alpha beta alpha beta", "fixture", "CC0-1.0", _HASH_A),
        RawDocument("blocked", "must not enter", "fixture", "restricted", _HASH_B),
    ]


def test_data_preparation_is_deterministic_deduplicated_and_atomic():
    spec = DataPreparationSpec(("CC0-1.0",), validation_permyriad=5000)
    first = prepare_corpus(_documents(), spec)
    second = prepare_corpus(_documents(), spec)
    assert first == second and first.digest == second.digest
    assert first.report.accepted == 2
    assert first.report.exact_duplicates == 1
    assert first.report.rejected_license == 1
    assert all("\r" not in row.text and "\x00" not in row.text for row in first.documents)
    assert {row.split for row in first.documents} <= {"train", "validation"}

    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "corpus"
        write_prepared_corpus(first, target)
        manifest = json.loads((target / "manifest.json").read_text("utf-8"))
        assert manifest["schema"] == "bcir.prepared_corpus.v1"
        assert set(manifest["files"]) == {"train.jsonl", "validation.jsonl"}
        _must_refuse(lambda: write_prepared_corpus(first, target), "replace")


def test_data_preparation_rejects_duplicate_ids_and_hard_control_policy():
    duplicate = _documents()[:1] * 2
    _must_refuse(lambda: prepare_corpus(duplicate, DataPreparationSpec(("CC0-1.0",))),
                 "duplicate document_id")
    corpus = prepare_corpus(
        [_documents()[1]], DataPreparationSpec(("CC0-1.0",), reject_control_characters=True))
    assert corpus.report.accepted == 0 and corpus.report.rejected_control == 1


def test_byte_bpe_roundtrips_utf8_and_builds_indexed_token_source():
    corpus = prepare_corpus(_documents()[:2], DataPreparationSpec(
        ("CC0-1.0",), validation_permyriad=0))
    texts = [row.text for row in corpus.split("train")]
    tokenizer = BytePairTokenizer.train(texts, vocab_size=280, min_frequency=2)
    assert 260 < tokenizer.vocab_size <= 280
    for text in texts:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode(tokenizer.encode("e\u0301\r\nline")) == "\u00e9\nline"
    assert BytePairTokenizer.from_json(tokenizer.to_json()) == tokenizer
    duplicate = tokenizer.to_json().replace(
        '"schema":"bcir.byte_bpe.v1"',
        '"schema":"bcir.byte_bpe.v1","schema":"bcir.byte_bpe.v1"', 1)
    _must_refuse(lambda: BytePairTokenizer.from_json(duplicate), "duplicate")
    _must_refuse(lambda: BytePairTokenizer.from_json(tokenizer.to_json() + "\n"),
                 "canonical")
    source = token_source_from_corpus(corpus, tokenizer)
    assert source.manifest.vocab_size == tokenizer.vocab_size
    assert source.batch(0, 2, 4) == source.batch(0, 2, 4)
    _must_refuse(lambda: BytePairTokenizer(((4, 5, 999),)), "canonical")


def test_pipeline_ledger_is_append_only_strict_and_atomic():
    outputs = [hashlib.sha256(f"output-{index}".encode()).hexdigest() for index in range(4)]
    reports = [hashlib.sha256(f"report-{index}".encode()).hexdigest() for index in range(4)]
    records = (
        PipelineStageRecord("data", (), outputs[0], reports[0], "prepared-corpus"),
        PipelineStageRecord("tokenizer", (outputs[0],), outputs[1], reports[1], "byte-bpe"),
        PipelineStageRecord("pretrain", (outputs[1],), outputs[2], reports[2], "checkpoint"),
        PipelineStageRecord("sft", (outputs[2],), outputs[3], reports[3], "checkpoint"),
    )
    ledger = TrainingPipelineLedger(records)
    assert TrainingPipelineLedger.from_json(ledger.to_json()).records == records
    _must_refuse(lambda: ledger.append(records[-1]), "already exists")
    bad_parent = PipelineStageRecord("reward", (outputs[0],), _HASH_A, _HASH_B, "checkpoint")
    _must_refuse(lambda: ledger.append(bad_parent), "invalid parent")
    malformed = json.loads(ledger.to_json())
    del malformed["records"][0]["artifact_kind"]
    _must_refuse(lambda: TrainingPipelineLedger.from_json(json.dumps(malformed)), "record 0")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pipeline.json"
        write_pipeline_ledger(path, ledger)
        assert read_pipeline_ledger(path).digest == ledger.digest


def test_stage_examples_and_specs_refuse_ambiguous_values():
    spec = StageTrainSpec("dpo", 2, 1e-3)
    assert len(spec.digest) == 64 and json.loads(spec.to_json())["stage"] == "dpo"
    assert StageTrainSpec.from_json(spec.to_json()) == spec
    malformed = json.loads(spec.to_json()); malformed["unknown"] = 1
    _must_refuse(lambda: StageTrainSpec.from_json(json.dumps(malformed)), "unknown")
    _must_refuse(lambda: replace(spec, steps=True), "steps")
    _must_refuse(lambda: replace(spec, stage="unknown"), "stage")
    _must_refuse(lambda: SFTExample((1,), (2.5,), _HASH_A), "token")
    _must_refuse(lambda: PreferenceExample((1,), (2,), (2,), _HASH_A), "differ")
    _must_refuse(lambda: PPOExample((1,), (2,), float("nan"), _HASH_A), "reward")
    reasoning = ReasoningExample((1,), (2, 3), (4,), _HASH_A, "exact-answer")
    assert reasoning.as_sft().response_ids == (2, 3, 4)
    _must_refuse(lambda: replace(reasoning, verified=False).as_sft(), "unverified")
    event = StageTrainEvent("dpo", 1, 0.5, 0.25, 2)
    assert event.to_dict()["stage"] == "dpo"
    _must_refuse(lambda: replace(event, loss=float("nan")), "event loss")


def test_small_architecture_specs_cover_mlp_recurrent_and_encoder_families():
    specs = [SmallModelSpec("mlp", 4, 8, 2), SmallModelSpec("gru", 4, 8, 2, layers=2),
             SmallModelSpec("transformer_encoder", 4, 8, 2, heads=2)]
    assert len({spec.digest for spec in specs}) == 3
    _must_refuse(lambda: SmallModelSpec("transformer_encoder", 4, 7, 2, heads=2),
                 "divisible")


def test_recorded_teacher_is_content_addressed_and_offline():
    request = TeacherRequest("embedding", _HASH_A, {"texts": ["alpha", "beta"]})
    response = TeacherResponse(request.request_id, "fixture", "immutable-v1",
                               {"vectors": [[1.0, 0.0], [0.0, 1.0]]}, _HASH_B)
    provider = RecordedTeacherProvider([response])
    assert provider.complete(request) == response
    missing = TeacherRequest("score", _HASH_A, {"value": "x"})
    _must_refuse(lambda: provider.complete(missing), "no frozen")
    matrix = relational_embedding_targets([[3.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    assert matrix[0][0] == 1.0 and matrix[0][1] == 0.0
    assert abs(matrix[0][2] - 2 ** -0.5) < 1e-12
    _must_refuse(lambda: relational_embedding_targets([[0.0, 0.0]]), "zero")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "teacher.jsonl"
        serialized = json.dumps(asdict(response), sort_keys=True)
        path.write_text(serialized + "\n", "utf-8")
        assert RecordedTeacherProvider.from_jsonl(path).complete(request) == response
        duplicate = serialized.replace('"provider": "fixture"',
                                       '"provider": "fixture", "provider": "fixture"', 1)
        path.write_text(duplicate + "\n", "utf-8")
        _must_refuse(lambda: RecordedTeacherProvider.from_jsonl(path), "duplicate")


def test_remote_compute_bundle_requires_attested_result():
    output = ArtifactFile("model.safetensors", 16, _HASH_B)
    bundle = RemoteTrainingBundle("4a5ae9a", _HASH_A, _HASH_B, _HASH_A,
                                  _HASH_B, _HASH_A, 1729)

    def execute(value):
        return RemoteTrainingResult(value.digest, _HASH_A, (output,), "offline-fixture")

    assert OfflineComputeAdapter(execute).run(bundle).bundle_sha256 == bundle.digest
    bad = OfflineComputeAdapter(lambda value: RemoteTrainingResult(
        _HASH_A, _HASH_A, (output,), "bad-fixture"))
    _must_refuse(lambda: bad.run(bundle), "unattested")
    _must_refuse(lambda: RemoteTrainingBundle(
        "ABCDEF0", _HASH_A, _HASH_B, _HASH_A, _HASH_B, _HASH_A, 1729),
        "source commit")
    _must_refuse(lambda: RemoteTrainingResult(
        _HASH_A, _HASH_B, (output, output), "offline-fixture"), "duplicate")


def test_reasoning_search_is_bounded_verified_and_replayable():
    budget = SearchBudget(2, 3, 8, 20)

    def propose(_prompt, round_index, _limit):
        if round_index == 0:
            return [ReasoningCandidate((10,), (40,), "micro"),
                    ReasoningCandidate((11, 12), (42,), "micro")]
        return [ReasoningCandidate((11, 12), (42,), "micro"),
                ReasoningCandidate(tuple(range(9)), (43,), "too-large")]

    def verify(_prompt, candidate):
        return Verification(candidate.answer_ids == (42,),
                            1.0 if candidate.answer_ids == (42,) else 0.1, "exact")

    first = bounded_reasoning_search((1, 2), budget, propose, verify)
    second = bounded_reasoning_search((1, 2), budget, propose, verify)
    assert first == second and first.selected.answer_ids == (42,)
    assert first.duplicate_candidates == 1 and first.rejected_by_budget == 1


def test_hosted_training_namespace_keeps_torch_quarantined():
    code = ("import sys; import bcir; import bcir.hosted.training as training; "
            "assert 'torch' not in sys.modules; assert training.StageTrainSpec")
    env = dict(os.environ); env["PYTHONPATH"] = str(_ROOT)
    result = subprocess.run([sys.executable, "-I", "-c",
                             f"import sys; sys.path.insert(0, {str(_ROOT)!r}); {code}"],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
