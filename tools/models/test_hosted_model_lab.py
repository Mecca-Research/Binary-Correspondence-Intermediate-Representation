#!/usr/bin/env python3
"""Bounded hosted-model tests that intentionally require PyTorch and Safetensors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file, save_file

from bcir.frontends.models.decode import DecoderSpec, decoder_param_count, next_token_logits
from bcir.frontends.models.hf_ingest import ingest_checkpoint_with_report
from bcir.hosted.models.checkpoint import (
    load_safe_checkpoint,
    save_safe_checkpoint,
    tensor_mapping_digest,
)
from bcir.hosted.models.export import export_hf_checkpoint, sha256_file
from bcir.hosted.models.model import HostedLlama
from bcir.hosted.models.spec import CorpusManifest, HostedTrainSpec
from bcir.hosted.models.train import _device_and_dtype, train_hosted
from tools.models.run_hosted_model_gate import run_gate

_FAILURE_STAGES = (
    "after-model",
    "after-optimizer",
    "after-state",
    "after-manifest",
    "before-publish",
    "after-generation",
    "before-latest",
)


class _InjectedFailure(RuntimeError):
    pass


def _decoder(*, tied: bool = True) -> DecoderSpec:
    return DecoderSpec(
        vocab_size=32,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        activation="silu_gate",
        n_kv_heads=1,
        tied_embeddings=tied,
        rms_norm_eps=1e-6,
    )


def _spec(*, tied: bool = True) -> HostedTrainSpec:
    return HostedTrainSpec(
        decoder=_decoder(tied=tied),
        context_length=4,
        batch_size=1,
        gradient_accumulation=1,
        steps=2,
        checkpoint_every=1,
        learning_rate=1e-3,
        min_learning_rate=0.0,
        warmup_steps=0,
        weight_decay=0.0,
        seed=19,
    )


def _corpus(tokenizer: bytes = b"bcir-hosted-test-tokenizer-v1") -> CorpusManifest:
    return CorpusManifest(
        name="bcir-hosted-unit-corpus",
        revision="v1",
        split="train",
        license="LicenseRef-BCIR-NC-1.0",
        source_sha256=hashlib.sha256(bytes(range(32))).hexdigest(),
        tokenizer_sha256=hashlib.sha256(tokenizer).hexdigest(),
        token_count=32,
        vocab_size=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def _optimizer(model: HostedLlama, spec: HostedTrainSpec):
    return torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        betas=(spec.beta1, spec.beta2),
        eps=spec.epsilon,
        weight_decay=spec.weight_decay,
        amsgrad=False,
    )


def _one_step(model: HostedLlama, optimizer) -> float:
    inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    _logits, loss = model(inputs, targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def _fixture():
    spec, corpus = _spec(), _corpus()
    torch.manual_seed(spec.seed)
    model = HostedLlama(spec.decoder, context_length=spec.context_length)
    optimizer = _optimizer(model, spec)
    initial_loss = _one_step(model, optimizer)
    return spec, corpus, model, optimizer, initial_loss


def _fresh_loader(spec: HostedTrainSpec):
    torch.manual_seed(spec.seed + 100)
    model = HostedLlama(spec.decoder, context_length=spec.context_length)
    return model, _optimizer(model, spec)


def _state_digest(model: HostedLlama) -> str:
    values = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not (model.spec.tied_embeddings and name == "lm_head.weight")
    }
    return tensor_mapping_digest(values)


def _canonical_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _generation(root: Path) -> Path:
    pointer = json.loads((root / "latest.json").read_text("utf-8"))
    return root / pointer["generation"]


def _rehash_generation(root: Path, changed: tuple[str, ...]) -> None:
    generation = _generation(root)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    for name in changed:
        path = generation / name
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if "optimizer.safetensors" in changed:
        values = load_file(str(generation / "optimizer.safetensors"), device="cpu")
        manifest["optimizer_sha256"] = tensor_mapping_digest(
            {name: value for name, value in values.items() if not name.startswith("rng.")}
        )
    _canonical_json(manifest_path, manifest)
    pointer_path = root / "latest.json"
    pointer = json.loads(pointer_path.read_text("utf-8"))
    pointer["manifest_sha256"] = sha256_file(manifest_path)
    _canonical_json(pointer_path, pointer)


def _assert_rejected_before_mutation(
    root: Path, spec: HostedTrainSpec, corpus: CorpusManifest
) -> None:
    destination, optimizer = _fresh_loader(spec)
    model_before = _state_digest(destination)
    rng_before = torch.get_rng_state().clone()
    try:
        load_safe_checkpoint(
            root, destination, optimizer, spec=spec, corpus=corpus, device=torch.device("cpu")
        )
        raise AssertionError(f"malformed checkpoint was accepted: {root.name}")
    except ValueError:
        pass
    assert _state_digest(destination) == model_before
    assert not optimizer.state
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_dependency_versions_are_exactly_pinned() -> None:
    assert torch.__version__.split("+", 1)[0] == "2.13.0"
    assert importlib.metadata.version("safetensors") == "0.8.0"
    assert importlib.metadata.version("numpy") == "2.2.6"


def test_tied_and_untied_gqa_exports_match_bcir_float_semantics() -> None:
    tokenizer_bytes = b"bcir-hosted-test-tokenizer-v1"
    corpus = _corpus(tokenizer_bytes)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        tokenizer = root / "tokenizer.model"
        tokenizer.write_bytes(tokenizer_bytes)
        for tied in (True, False):
            spec = _decoder(tied=tied)
            torch.manual_seed(101 + int(tied))
            model = HostedLlama(spec, context_length=16)
            assert tuple(model.model.layers[0].self_attn.k_proj.weight.shape) == (
                spec.kv_dim,
                spec.d_model,
            )
            exported = export_hf_checkpoint(
                model, root / f"export-{tied}", tokenizer_path=tokenizer, corpus_manifest=corpus
            )
            ingested, weights, report = ingest_checkpoint_with_report(str(exported.directory))
            assert ingested == spec
            assert bool(weights.lm_head) is not tied
            assert report.auxiliary_element_count == 0
            assert (
                report.decoder_element_count
                == decoder_param_count(spec)
                == model.unique_parameter_count()
            )
            prompt = [1, 7, 3, 9]
            model.eval()
            with torch.no_grad():
                logits, _ = model(torch.tensor([prompt], dtype=torch.long))
            hosted = logits[0, -1].double().tolist()
            oracle = next_token_logits(prompt, ingested, weights)
            assert max(abs(left - right) for left, right in zip(hosted, oracle)) <= 1e-5


def test_requested_device_precision_and_activation_checkpointing_are_explicit() -> None:
    for device, dtype, error in (
        ("cpu", "float16", ValueError),
        ("cpu", "bfloat16", ValueError),
        ("metal", "float32", ValueError),
    ):
        try:
            _device_and_dtype(device, dtype)
            raise AssertionError(f"unsupported {device}/{dtype} request fell back")
        except error:
            pass
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        try:
            _device_and_dtype("cuda", "float32")
            raise AssertionError("unavailable CUDA request fell back")
        except RuntimeError:
            pass

    spec = _spec()
    torch.manual_seed(spec.seed)
    plain = HostedLlama(spec.decoder, context_length=spec.context_length)
    torch.manual_seed(spec.seed)
    rematerialized = HostedLlama(
        spec.decoder, activation_checkpointing=True, context_length=spec.context_length
    )
    plain_optimizer = _optimizer(plain, spec)
    rematerialized_optimizer = _optimizer(rematerialized, spec)
    plain_loss = _one_step(plain, plain_optimizer)
    rematerialized_loss = _one_step(rematerialized, rematerialized_optimizer)
    assert plain_loss == rematerialized_loss
    assert _state_digest(plain) == _state_digest(rematerialized)


def test_checkpoint_writes_are_deterministic() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec, corpus, model, optimizer, initial_loss = _fixture()
        left = save_safe_checkpoint(
            root / "left",
            model,
            optimizer,
            spec=spec,
            corpus=corpus,
            global_step=1,
            batch_index=1,
            initial_loss=initial_loss,
        )
        right = save_safe_checkpoint(
            root / "right",
            model,
            optimizer,
            spec=spec,
            corpus=corpus,
            global_step=1,
            batch_index=1,
            initial_loss=initial_loss,
        )
        assert left.manifest_sha256 == right.manifest_sha256
        assert (root / "left" / "latest.json").read_bytes() == (
            root / "right" / "latest.json"
        ).read_bytes()
        for name in (
            "config.json",
            "model.safetensors",
            "optimizer.safetensors",
            "state.json",
            "manifest.json",
        ):
            assert (left.directory / name).read_bytes() == (right.directory / name).read_bytes()


def test_every_checkpoint_write_stage_preserves_the_previous_generation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        for stage in _FAILURE_STAGES:
            spec, corpus, model, optimizer, initial_loss = _fixture()
            root = parent / stage
            baseline = save_safe_checkpoint(
                root,
                model,
                optimizer,
                spec=spec,
                corpus=corpus,
                global_step=1,
                batch_index=1,
                initial_loss=initial_loss,
            )
            _one_step(model, optimizer)
            pointer_before = (root / "latest.json").read_bytes()

            def inject(current: str, expected: str = stage) -> None:
                if current == expected:
                    raise _InjectedFailure(expected)

            try:
                save_safe_checkpoint(
                    root,
                    model,
                    optimizer,
                    spec=spec,
                    corpus=corpus,
                    global_step=2,
                    batch_index=2,
                    initial_loss=initial_loss,
                    failure_injector=inject,
                )
                raise AssertionError(f"failure stage {stage} was not reached")
            except _InjectedFailure as exc:
                assert str(exc) == stage
            assert (root / "latest.json").read_bytes() == pointer_before
            entries = sorted((root / "checkpoints").iterdir())
            assert entries == [baseline.directory]
            assert not any(path.name.startswith(".") for path in root.rglob("*"))
            loaded_model, loaded_optimizer = _fresh_loader(spec)
            loaded = load_safe_checkpoint(
                root,
                loaded_model,
                loaded_optimizer,
                spec=spec,
                corpus=corpus,
                device=torch.device("cpu"),
            )
            assert loaded.global_step == 1 and loaded.batch_index == 1


def test_loaded_checkpoint_releases_its_file_mappings() -> None:
    """A successful load must not keep the checkpoint files memory-mapped.

    ``safetensors.torch.load_file`` returns mmap-backed tensors.  If those
    tensors are committed into optimizer state or returned as RNG snapshots,
    the mapping outlives the load; Windows cannot delete a mapped file, so
    checkpoint pruning and every TemporaryDirectory cleanup then fails with
    PermissionError.  POSIX unlinks mapped files silently, which is why only
    the Windows lab job catches the regression.
    """
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec, corpus, model, optimizer, initial_loss = _fixture()
        saved = save_safe_checkpoint(
            root,
            model,
            optimizer,
            spec=spec,
            corpus=corpus,
            global_step=1,
            batch_index=1,
            initial_loss=initial_loss,
        )
        destination, destination_optimizer = _fresh_loader(spec)
        loaded = load_safe_checkpoint(
            root,
            destination,
            destination_optimizer,
            spec=spec,
            corpus=corpus,
            device=torch.device("cpu"),
        )
        # Hold the references exactly the way a resuming trainer does while
        # the generation directory is deleted out from under them.
        assert loaded.global_step == 1
        shutil.rmtree(saved.directory)
        assert not saved.directory.exists()
        assert destination_optimizer.state, "optimizer state must stay usable"
        assert loaded.cpu_rng.numel() > 0, "RNG snapshot must stay usable"


def test_checkpoint_corruption_and_malicious_indexes_fail_before_mutation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        spec, corpus, model, optimizer, initial_loss = _fixture()
        baseline_root = parent / "baseline"
        saved = save_safe_checkpoint(
            baseline_root,
            model,
            optimizer,
            spec=spec,
            corpus=corpus,
            global_step=1,
            batch_index=1,
            initial_loss=initial_loss,
        )

        for name in (
            "config.json",
            "model.safetensors",
            "optimizer.safetensors",
            "state.json",
            "manifest.json",
        ):
            root = parent / f"truncated-{name.replace('.', '-')}"
            shutil.copytree(baseline_root, root)
            pointer = json.loads((root / "latest.json").read_text("utf-8"))
            generation = root / pointer["generation"]
            path = generation / name
            raw = path.read_bytes()
            path.write_bytes(raw[:-1])
            destination, destination_optimizer = _fresh_loader(spec)
            before = _state_digest(destination)
            try:
                load_safe_checkpoint(
                    root,
                    destination,
                    destination_optimizer,
                    spec=spec,
                    corpus=corpus,
                    device=torch.device("cpu"),
                )
                raise AssertionError(f"truncated {name} was accepted")
            except (ValueError, OSError):
                pass
            assert _state_digest(destination) == before

        # Re-hash a structurally malicious optimizer index so the test reaches semantic
        # validation instead of stopping at the outer checksum fence.
        root = parent / "malicious-index"
        shutil.copytree(baseline_root, root)
        pointer_path = root / "latest.json"
        pointer = json.loads(pointer_path.read_text("utf-8"))
        generation = root / pointer["generation"]
        state_path = generation / "state.json"
        state = json.loads(state_path.read_text("utf-8"))
        state["optimizer_index"][0]["parameter"] = "not.a.model.parameter"
        _canonical_json(state_path, state)
        manifest_path = generation / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["files"]["state.json"] = {
            "bytes": state_path.stat().st_size,
            "sha256": sha256_file(state_path),
        }
        _canonical_json(manifest_path, manifest)
        pointer["manifest_sha256"] = sha256_file(manifest_path)
        _canonical_json(pointer_path, pointer)
        destination, destination_optimizer = _fresh_loader(spec)
        before = _state_digest(destination)
        try:
            load_safe_checkpoint(
                root,
                destination,
                destination_optimizer,
                spec=spec,
                corpus=corpus,
                device=torch.device("cpu"),
            )
            raise AssertionError("malicious optimizer index was accepted")
        except ValueError:
            pass
        assert _state_digest(destination) == before

        traversal = parent / "pointer-traversal"
        shutil.copytree(baseline_root, traversal)
        _canonical_json(
            traversal / "latest.json",
            {
                "schema": "bcir.hosted_checkpoint_pointer.v1",
                "generation": "../baseline",
                "manifest_sha256": "00" * 32,
            },
        )
        destination, destination_optimizer = _fresh_loader(spec)
        try:
            load_safe_checkpoint(
                traversal,
                destination,
                destination_optimizer,
                spec=spec,
                corpus=corpus,
                device=torch.device("cpu"),
            )
            raise AssertionError("checkpoint pointer traversal was accepted")
        except ValueError:
            pass


def test_checkpoint_semantic_state_is_rejected_before_mutation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        spec, corpus, model, optimizer, initial_loss = _fixture()
        baseline = parent / "baseline"
        save_safe_checkpoint(
            baseline,
            model,
            optimizer,
            spec=spec,
            corpus=corpus,
            global_step=1,
            batch_index=1,
            initial_loss=initial_loss,
        )

        bad_scaler = parent / "bad-scaler"
        shutil.copytree(baseline, bad_scaler)
        state_path = _generation(bad_scaler) / "state.json"
        state = json.loads(state_path.read_text("utf-8"))
        state["scaler_state"] = {"bogus": 1}
        _canonical_json(state_path, state)
        _rehash_generation(bad_scaler, ("state.json",))
        _assert_rejected_before_mutation(bad_scaler, spec, corpus)

        wrong_topology = parent / "wrong-rng-topology"
        shutil.copytree(baseline, wrong_topology)
        generation = _generation(wrong_topology)
        optimizer_path = generation / "optimizer.safetensors"
        values = load_file(str(optimizer_path), device="cpu")
        values["rng.cuda.0"] = values["rng.cpu"].clone()
        save_file({name: values[name] for name in sorted(values)}, str(optimizer_path))
        state_path = generation / "state.json"
        state = json.loads(state_path.read_text("utf-8"))
        state["cuda_rng_tensors"] = ["rng.cuda.0"]
        _canonical_json(state_path, state)
        _rehash_generation(wrong_topology, ("optimizer.safetensors", "state.json"))
        _assert_rejected_before_mutation(wrong_topology, spec, corpus)

        bad_rng = parent / "bad-cpu-rng"
        shutil.copytree(baseline, bad_rng)
        optimizer_path = _generation(bad_rng) / "optimizer.safetensors"
        values = load_file(str(optimizer_path), device="cpu")
        values["rng.cpu"] = torch.tensor([1, 2], dtype=torch.uint8)
        save_file({name: values[name] for name in sorted(values)}, str(optimizer_path))
        _rehash_generation(bad_rng, ("optimizer.safetensors",))
        _assert_rejected_before_mutation(bad_rng, spec, corpus)

        bad_step = parent / "bad-adam-step"
        shutil.copytree(baseline, bad_step)
        generation = _generation(bad_step)
        optimizer_path = generation / "optimizer.safetensors"
        values = load_file(str(optimizer_path), device="cpu")
        state = json.loads((generation / "state.json").read_text("utf-8"))
        step_name = next(
            row["tensor"] for row in state["optimizer_index"] if row["state"] == "step"
        )
        values[step_name] = torch.tensor(123.5, dtype=torch.float32)
        save_file({name: values[name] for name in sorted(values)}, str(optimizer_path))
        _rehash_generation(bad_step, ("optimizer.safetensors",))
        _assert_rejected_before_mutation(bad_step, spec, corpus)

        oversized = parent / "oversized-manifest"
        shutil.copytree(baseline, oversized)
        with (_generation(oversized) / "manifest.json").open("wb") as stream:
            stream.truncate(16 * 1024 * 1024 + 1)
        _assert_rejected_before_mutation(oversized, spec, corpus)


def test_token_ids_are_not_coerced_and_deterministic_mode_is_restored() -> None:
    spec, corpus = _spec(), _corpus()

    class FractionalSource:
        manifest = corpus

        @staticmethod
        def batch(_batch_index, batch_size, context_length):
            return [[1.5] * (context_length + 1) for _ in range(batch_size)]

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            try:
                train_hosted(spec, FractionalSource(), temporary)
                raise AssertionError("fractional token IDs were accepted")
            except ValueError as exc:
                assert "invalid ID" in str(exc)
        assert not torch.are_deterministic_algorithms_enabled()
        model = HostedLlama(spec.decoder, context_length=spec.context_length)
        try:
            model.greedy([1.5])
            raise AssertionError("greedy decode coerced a fractional token ID")
        except ValueError:
            pass
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_train_to_c_gate(output_dir: Path) -> None:
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        report = run_gate(output_dir)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    assert report["passed"] is True
    assert report["decode"]["expected_id"] == 100
    assert report["decode"]["q8_c_ids"] == [100]
    assert report["decode"]["max_abs_logit_error_c_vs_q8"] <= 1e-9
    assert report["training"]["resume_exact"] is True
    encoded = (output_dir / "parity-report.json").read_text("utf-8").lower()
    assert "timestamp" not in encoded and "created_at" not in encoded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "build" / "hosted-model-gate"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    tests = (
        test_dependency_versions_are_exactly_pinned,
        test_tied_and_untied_gqa_exports_match_bcir_float_semantics,
        test_requested_device_precision_and_activation_checkpointing_are_explicit,
        test_checkpoint_writes_are_deterministic,
        test_every_checkpoint_write_stage_preserves_the_previous_generation,
        test_loaded_checkpoint_releases_its_file_mappings,
        test_checkpoint_corruption_and_malicious_indexes_fail_before_mutation,
        test_checkpoint_semantic_state_is_rejected_before_mutation,
        test_token_ids_are_not_coerced_and_deterministic_mode_is_restored,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    test_train_to_c_gate(Path(args.output_dir))
    print("PASS test_train_to_c_gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
