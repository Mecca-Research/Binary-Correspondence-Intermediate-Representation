"""Pickle-free, generation-based checkpoints for hosted BCIR training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import torch
    from safetensors.torch import load_file, save_file
except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
    raise ModuleNotFoundError("safe hosted checkpoints require torch and safetensors") from exc

from .export import sha256_file
from .model import HostedLlama
from .spec import CorpusManifest, HostedTrainSpec, _unique_object

_CHECKPOINT_FILE_LIMITS = {
    "config.json": 1024 * 1024,
    "model.safetensors": 8 * 1024 * 1024 * 1024,
    "optimizer.safetensors": 8 * 1024 * 1024 * 1024,
    "state.json": 16 * 1024 * 1024,
}
_CHECKPOINT_MANIFEST_LIMIT = 16 * 1024 * 1024
_ADAMW_STATE_NAMES = frozenset({"step", "exp_avg", "exp_avg_sq"})
_GENERATION_RE = re.compile(r"step-[0-9]{8}-[0-9a-f]{12}-[0-9a-f]{12}")
_SCALER_STATE_NAMES = frozenset(
    {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
)


def _write_json(path: Path, value) -> None:
    raw = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path, *, limit: int = 16 * 1024 * 1024):
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size > limit:
            raise ValueError(f"{path.name} exceeds {limit} bytes")
        raw = stream.read(limit + 1)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"malformed {path.name}: {exc}") from exc


def _fsync(path: Path) -> None:
    # Windows' CRT rejects fsync() on a read-only descriptor.  Reopen the
    # completed tensor file without truncation but with write access so the
    # durability barrier has the same semantics on Windows and POSIX hosts.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _tensor_bytes(value: torch.Tensor) -> bytes:
    # Reshape first: PyTorch refuses a dtype-changing view of a rank-0 optimizer
    # ``step`` tensor even though its storage is perfectly byte-addressable.
    return value.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_mapping_digest(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().to("cpu").contiguous()
        encoded = name.encode("utf-8")
        dtype = str(tensor.dtype).encode("ascii")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(len(dtype).to_bytes(2, "little"))
        digest.update(dtype)
        digest.update(len(tensor.shape).to_bytes(2, "little"))
        for dim in tensor.shape:
            digest.update(int(dim).to_bytes(8, "little", signed=False))
        raw = _tensor_bytes(tensor)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _tensor_is_finite(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(
        torch.isfinite(value).all().item()
    )


def _validate_scaler_state(value, *, enabled: bool) -> dict:
    if not isinstance(value, dict):
        raise ValueError("checkpoint scaler state must be an object")
    if not enabled:
        if value:
            raise ValueError("checkpoint has scaler state for a non-FP16/CUDA run")
        return {}
    if set(value) != _SCALER_STATE_NAMES:
        raise ValueError("checkpoint FP16 scaler state has missing or unknown fields")
    for field in ("scale", "growth_factor", "backoff_factor"):
        item = value[field]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"checkpoint scaler {field} is not finite")
    if (
        float(value["scale"]) <= 0.0
        or float(value["growth_factor"]) <= 1.0
        or not 0.0 < float(value["backoff_factor"]) < 1.0
    ):
        raise ValueError("checkpoint scaler factors are outside their valid ranges")
    if (
        type(value["growth_interval"]) is not int
        or value["growth_interval"] < 1
        or type(value["_growth_tracker"]) is not int
        or value["_growth_tracker"] < 0
    ):
        raise ValueError("checkpoint scaler counters are outside their valid ranges")
    return dict(value)


def _validate_rng_state(value: torch.Tensor, device: str) -> None:
    if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"checkpoint {device} RNG tensor data is malformed")
    try:
        generator = torch.Generator(device=device)
        generator.set_state(value.to("cpu"))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint {device} RNG state is invalid") from exc


def _model_tensors(model: HostedLlama) -> dict[str, torch.Tensor]:
    out = {}
    for name, value in sorted(model.state_dict().items()):
        if model.spec.tied_embeddings and name == "lm_head.weight":
            continue
        tensor = value.detach().to("cpu").contiguous()
        if not _tensor_is_finite(tensor):
            raise ValueError(f"model tensor {name!r} contains non-finite data")
        out[name] = tensor
    return out


def _optimizer_tensors(model: HostedLlama, optimizer) -> tuple[dict, list]:
    out, index = {}, []
    for parameter_name, parameter in sorted(model.named_parameters()):
        state = optimizer.state.get(parameter, {})
        for state_name in sorted(state):
            value = state[state_name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"optimizer state {parameter_name}.{state_name} is not a tensor")
            tensor_name = f"optimizer.{len(index):08d}"
            tensor = value.detach().to("cpu").contiguous()
            if state_name not in _ADAMW_STATE_NAMES or not _tensor_is_finite(tensor):
                raise ValueError(f"optimizer state {parameter_name}.{state_name} is invalid")
            out[tensor_name] = tensor
            index.append({"parameter": parameter_name, "state": state_name, "tensor": tensor_name})
    return out, index


def _dependencies() -> dict:
    return {
        "numpy": importlib.metadata.version("numpy"),
        "python": platform.python_version(),
        "python_tensor_framework": torch.__version__,
        "safetensors": importlib.metadata.version("safetensors"),
    }


def _validate_optimizer(model: HostedLlama, optimizer, spec: HostedTrainSpec) -> None:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("hosted checkpoint optimizer must be torch.optim.AdamW")
    model_parameters = {id(parameter) for parameter in model.parameters()}
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group.get("params", ())
    ]
    if (
        len(optimizer_parameters) != len({id(parameter) for parameter in optimizer_parameters})
        or {id(parameter) for parameter in optimizer_parameters} != model_parameters
    ):
        raise ValueError("optimizer parameters do not exactly match the hosted model")
    for group in optimizer.param_groups:
        if (
            tuple(group.get("betas", ())) != (spec.beta1, spec.beta2)
            or group.get("eps") != spec.epsilon
            or group.get("weight_decay") != spec.weight_decay
            or group.get("amsgrad") is not False
        ):
            raise ValueError("optimizer configuration differs from HostedTrainSpec")


@dataclass(frozen=True)
class SavedCheckpoint:
    directory: Path
    manifest_sha256: str
    model_sha256: str
    optimizer_sha256: str


@dataclass(frozen=True)
class LoadedCheckpoint:
    directory: Path
    manifest_sha256: str
    global_step: int
    batch_index: int
    initial_loss: float
    scaler_state: dict
    cpu_rng: torch.Tensor
    cuda_rng: tuple[torch.Tensor, ...]
    model_sha256: str
    optimizer_sha256: str


def save_safe_checkpoint(
    root: os.PathLike | str,
    model: HostedLlama,
    optimizer,
    *,
    spec: HostedTrainSpec,
    corpus: CorpusManifest,
    global_step: int,
    batch_index: int,
    initial_loss: float,
    scaler_state: dict | None = None,
    failure_injector: Callable[[str], None] | None = None,
) -> SavedCheckpoint:
    """Publish one immutable checkpoint generation, then atomically advance ``latest``."""
    if (
        not isinstance(model, HostedLlama)
        or not isinstance(spec, HostedTrainSpec)
        or not isinstance(corpus, CorpusManifest)
    ):
        raise ValueError("checkpoint model/spec/corpus types are invalid")
    if model.spec != spec.decoder or corpus.vocab_size != model.spec.vocab_size:
        raise ValueError("checkpoint model/spec/corpus identities differ")
    if model.context_length != spec.context_length:
        raise ValueError("checkpoint model context differs from HostedTrainSpec")
    _validate_optimizer(model, optimizer, spec)
    if type(global_step) is not int or not 0 <= global_step <= spec.steps:
        raise ValueError("global_step must be in [0, spec.steps]")
    if type(batch_index) is not int or batch_index != global_step * spec.gradient_accumulation:
        raise ValueError("batch_index must equal global_step * gradient_accumulation")
    if (
        isinstance(initial_loss, bool)
        or not isinstance(initial_loss, (int, float))
        or not math.isfinite(float(initial_loss))
    ):
        raise ValueError("initial_loss must be a finite number")
    initial_loss = float(initial_loss)
    hook = failure_injector or (lambda _stage: None)
    model_device = next(model.parameters()).device
    scaler_payload = {} if scaler_state is None else scaler_state
    scaler_payload = _validate_scaler_state(
        scaler_payload, enabled=(model_device.type == "cuda" and spec.dtype == "float16")
    )
    root = Path(root)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint.tmp-", dir=checkpoints))
    published: Path | None = None
    published_by_this_call = False
    try:
        model_values = _model_tensors(model)
        optimizer_values, optimizer_index = _optimizer_tensors(model, optimizer)
        cpu_rng_name = "rng.cpu"
        optimizer_values[cpu_rng_name] = torch.get_rng_state().to("cpu")
        cuda_names = []
        if model_device.type == "cuda":
            for index, state in enumerate(torch.cuda.get_rng_state_all()):
                name = f"rng.cuda.{index}"
                optimizer_values[name] = state.to("cpu")
                cuda_names.append(name)
        if global_step and not optimizer_index:
            raise ValueError("optimizer checkpoint has no AdamW state after step zero")
        save_file(
            {name: model_values[name] for name in sorted(model_values)},
            str(temporary / "model.safetensors"),
        )
        _fsync(temporary / "model.safetensors")
        hook("after-model")
        save_file(
            {name: optimizer_values[name] for name in sorted(optimizer_values)},
            str(temporary / "optimizer.safetensors"),
        )
        _fsync(temporary / "optimizer.safetensors")
        hook("after-optimizer")
        _write_json(temporary / "config.json", json.loads(spec.to_json()))
        state = {
            "schema": "bcir.hosted_checkpoint_state.v1",
            "batch_index": batch_index,
            "corpus_sha256": corpus.digest,
            "cuda_rng_tensors": cuda_names,
            "dependencies": _dependencies(),
            "global_step": global_step,
            "initial_loss": initial_loss,
            "optimizer_index": optimizer_index,
            "rng_cpu_tensor": cpu_rng_name,
            "scaler_state": scaler_payload,
            "scheduler_state": {"kind": "warmup_cosine_v1", "last_completed_step": global_step},
            "spec_sha256": spec.digest,
            "tokenizer_sha256": corpus.tokenizer_sha256,
        }
        _write_json(temporary / "state.json", state)
        hook("after-state")
        model_digest = tensor_mapping_digest(model_values)
        optimizer_digest = tensor_mapping_digest(
            {name: value for name, value in optimizer_values.items() if not name.startswith("rng.")}
        )
        files = {}
        for name in ("config.json", "model.safetensors", "optimizer.safetensors", "state.json"):
            path = temporary / name
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        manifest = {
            "schema": "bcir.hosted_checkpoint.v1",
            "files": files,
            "model_sha256": model_digest,
            "optimizer_sha256": optimizer_digest,
        }
        _write_json(temporary / "manifest.json", manifest)
        hook("after-manifest")
        manifest_digest = sha256_file(temporary / "manifest.json")
        generation = f"step-{global_step:08d}-{model_digest[:12]}-{optimizer_digest[:12]}"
        published = checkpoints / generation
        hook("before-publish")
        if published.exists():
            if (
                published.is_symlink()
                or not published.is_dir()
                or sha256_file(published / "manifest.json") != manifest_digest
                or any(
                    not (published / name).is_file()
                    or (published / name).is_symlink()
                    or sha256_file(published / name) != sha256_file(temporary / name)
                    for name in (*files, "manifest.json")
                )
            ):
                raise FileExistsError(f"checkpoint generation collision at {published}")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, published)
            published_by_this_call = True
        hook("after-generation")
        latest = {
            "schema": "bcir.hosted_checkpoint_pointer.v1",
            "generation": f"checkpoints/{generation}",
            "manifest_sha256": manifest_digest,
        }
        fd, pointer_tmp = tempfile.mkstemp(prefix=".latest.tmp-", dir=root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(
                    json.dumps(latest, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    + b"\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            hook("before-latest")
            os.replace(pointer_tmp, root / "latest.json")
        finally:
            try:
                os.unlink(pointer_tmp)
            except FileNotFoundError:
                pass
        return SavedCheckpoint(published, manifest_digest, model_digest, optimizer_digest)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        # A generation is committed only when latest.json names it.  In particular, an
        # older valid pointer must not make a newly renamed-but-unreferenced generation
        # survive a failure between the two atomic renames.
        if published_by_this_call and published is not None:
            shutil.rmtree(published, ignore_errors=True)
        raise


def _resolve_checkpoint(path: os.PathLike | str) -> tuple[Path, str | None]:
    path = Path(path)
    pointer = path / "latest.json"
    if pointer.is_symlink():
        raise ValueError("hosted checkpoint pointer must not be a symbolic link")
    if pointer.is_file():
        value = _read_json(pointer, limit=1024 * 1024)
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "generation", "manifest_sha256"}
            or value["schema"] != "bcir.hosted_checkpoint_pointer.v1"
        ):
            raise ValueError("malformed hosted checkpoint pointer")
        if not _is_sha256(value["manifest_sha256"]):
            raise ValueError("checkpoint pointer has an invalid manifest digest")
        generation = value["generation"]
        candidate = Path(generation) if isinstance(generation, str) else None
        if (
            candidate is None
            or len(candidate.parts) != 2
            or candidate.parts[0] != "checkpoints"
            or not _GENERATION_RE.fullmatch(candidate.parts[1])
        ):
            raise ValueError("checkpoint pointer escapes its root")
        directory = path / candidate
        try:
            if os.path.commonpath((str(path.resolve()), str(directory.resolve()))) != str(
                path.resolve()
            ):
                raise ValueError("checkpoint pointer escapes its root")
        except (OSError, ValueError) as exc:
            raise ValueError("checkpoint pointer escapes its root") from exc
        return directory, value["manifest_sha256"]
    return path, None


def load_safe_checkpoint(
    path: os.PathLike | str,
    model: HostedLlama,
    optimizer,
    *,
    spec: HostedTrainSpec,
    corpus: CorpusManifest,
    device: torch.device,
) -> LoadedCheckpoint:
    if (
        not isinstance(model, HostedLlama)
        or not isinstance(spec, HostedTrainSpec)
        or not isinstance(corpus, CorpusManifest)
        or not isinstance(device, torch.device)
    ):
        raise ValueError("checkpoint load arguments are invalid")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("hosted checkpoints support CPU or CUDA destinations only")
    cuda_device_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if device.type == "cuda" and (
        cuda_device_count < 1
        or device.index is not None
        and not 0 <= device.index < cuda_device_count
    ):
        raise ValueError("checkpoint CUDA destination is unavailable")
    if model.spec != spec.decoder or corpus.vocab_size != model.spec.vocab_size:
        raise ValueError("checkpoint model/spec/corpus identities differ")
    if model.context_length not in (None, spec.context_length):
        raise ValueError("checkpoint model context differs from HostedTrainSpec")
    _validate_optimizer(model, optimizer, spec)
    directory, pointer_digest = _resolve_checkpoint(path)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint generation is not a regular directory")
    manifest_path = directory / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > _CHECKPOINT_MANIFEST_LIMIT
    ):
        raise ValueError("checkpoint manifest is not a bounded regular file")
    manifest_digest = sha256_file(manifest_path)
    if pointer_digest is not None and pointer_digest != manifest_digest:
        raise ValueError("checkpoint pointer/manifest digest mismatch")
    manifest = _read_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "files", "model_sha256", "optimizer_sha256"}
        or manifest["schema"] != "bcir.hosted_checkpoint.v1"
    ):
        raise ValueError("malformed hosted checkpoint manifest")
    if not _is_sha256(manifest["model_sha256"]) or not _is_sha256(manifest["optimizer_sha256"]):
        raise ValueError("checkpoint manifest has an invalid tensor digest")
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != {
        "config.json",
        "model.safetensors",
        "optimizer.safetensors",
        "state.json",
    }:
        raise ValueError("checkpoint manifest has an invalid file inventory")
    for name, record in manifest["files"].items():
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
            raise ValueError(f"checkpoint file record {name!r} is malformed")
        if (
            type(record["bytes"]) is not int
            or not 0 <= record["bytes"] <= _CHECKPOINT_FILE_LIMITS[name]
            or not _is_sha256(record["sha256"])
        ):
            raise ValueError(f"checkpoint file record {name!r} has invalid size/hash fields")
        file = directory / name
        if (
            file.is_symlink()
            or not file.is_file()
            or file.stat().st_size != record["bytes"]
            or sha256_file(file) != record["sha256"]
        ):
            raise ValueError(f"checkpoint file {name!r} failed size/hash verification")
    loaded_spec = HostedTrainSpec.from_json((directory / "config.json").read_text("utf-8"))
    if loaded_spec != spec:
        raise ValueError("checkpoint training specification mismatch")
    state = _read_json(directory / "state.json")
    required = {
        "schema",
        "batch_index",
        "corpus_sha256",
        "cuda_rng_tensors",
        "dependencies",
        "global_step",
        "initial_loss",
        "optimizer_index",
        "rng_cpu_tensor",
        "scaler_state",
        "scheduler_state",
        "spec_sha256",
        "tokenizer_sha256",
    }
    if (
        not isinstance(state, dict)
        or set(state) != required
        or state["schema"] != "bcir.hosted_checkpoint_state.v1"
    ):
        raise ValueError("malformed hosted checkpoint state")
    if (
        state["spec_sha256"] != spec.digest
        or state["corpus_sha256"] != corpus.digest
        or state["tokenizer_sha256"] != corpus.tokenizer_sha256
    ):
        raise ValueError("checkpoint input identity mismatch")
    if state["dependencies"] != _dependencies():
        raise ValueError("checkpoint dependency versions differ from the current runtime")
    global_step, batch_index = state["global_step"], state["batch_index"]
    if (
        type(global_step) is not int
        or not 0 <= global_step <= spec.steps
        or type(batch_index) is not int
        or batch_index < 0
        or batch_index != global_step * spec.gradient_accumulation
    ):
        raise ValueError("checkpoint step/data cursor is invalid")
    initial_loss = state["initial_loss"]
    if (
        isinstance(initial_loss, bool)
        or not isinstance(initial_loss, (int, float))
        or not math.isfinite(float(initial_loss))
    ):
        raise ValueError("checkpoint initial loss is invalid")
    if not isinstance(state["scaler_state"], dict) or state["scheduler_state"] != {
        "kind": "warmup_cosine_v1",
        "last_completed_step": global_step,
    }:
        raise ValueError("checkpoint scheduler/scaler state is malformed")
    if (
        not isinstance(state["optimizer_index"], list)
        or not isinstance(state["rng_cpu_tensor"], str)
        or not isinstance(state["cuda_rng_tensors"], list)
        or any(not isinstance(name, str) for name in state["cuda_rng_tensors"])
    ):
        raise ValueError("checkpoint optimizer/RNG index is malformed")

    model_values = load_file(str(directory / "model.safetensors"), device="cpu")
    if tensor_mapping_digest(model_values) != manifest["model_sha256"]:
        raise ValueError("checkpoint model tensor digest mismatch")
    expected_model = model.state_dict()
    expected_names = set(expected_model)
    if model.spec.tied_embeddings:
        expected_names.remove("lm_head.weight")
    if set(model_values) != expected_names:
        raise ValueError("checkpoint model tensor inventory mismatch")
    for name, value in model_values.items():
        expected = expected_model[name]
        if (
            value.shape != expected.shape
            or value.dtype != expected.dtype
            or not _tensor_is_finite(value)
        ):
            raise ValueError(f"checkpoint model tensor {name!r} has invalid shape/dtype/data")

    all_values = load_file(str(directory / "optimizer.safetensors"), device="cpu")
    optimizer_values = {
        name: value for name, value in all_values.items() if not name.startswith("rng.")
    }
    if tensor_mapping_digest(optimizer_values) != manifest["optimizer_sha256"]:
        raise ValueError("checkpoint optimizer tensor digest mismatch")
    by_name = dict(model.named_parameters())
    seen, indexed_tensors = set(), set()
    restored: list[tuple[torch.nn.Parameter, str, torch.Tensor]] = []
    states_by_parameter = {name: set() for name in by_name}
    for index, row in enumerate(state["optimizer_index"]):
        if not isinstance(row, dict) or set(row) != {"parameter", "state", "tensor"}:
            raise ValueError("malformed optimizer index row")
        parameter_name, state_name, tensor_name = row["parameter"], row["state"], row["tensor"]
        if (
            not all(isinstance(value, str) for value in (parameter_name, state_name, tensor_name))
            or parameter_name not in by_name
            or state_name not in _ADAMW_STATE_NAMES
            or tensor_name != f"optimizer.{index:08d}"
            or tensor_name not in optimizer_values
            or (parameter_name, state_name) in seen
        ):
            raise ValueError("optimizer index references an unknown or duplicate state")
        seen.add((parameter_name, state_name))
        indexed_tensors.add(tensor_name)
        parameter = by_name[parameter_name]
        value = optimizer_values[tensor_name]
        if state_name == "step":
            valid = (
                value.ndim == 0
                and value.dtype == torch.float32
                and float(value.item()) == float(global_step)
            )
        else:
            valid = value.shape == parameter.shape and value.dtype == parameter.dtype
        if not valid or not _tensor_is_finite(value):
            raise ValueError(f"optimizer tensor {tensor_name!r} has invalid shape/dtype/data")
        states_by_parameter[parameter_name].add(state_name)
        restored.append((parameter, state_name, value))
    if indexed_tensors != set(optimizer_values):
        raise ValueError("checkpoint has unindexed optimizer tensors")
    expected_states = _ADAMW_STATE_NAMES if global_step else frozenset()
    if any(names != expected_states for names in states_by_parameter.values()):
        raise ValueError("checkpoint AdamW state is incomplete")
    cpu_name = state["rng_cpu_tensor"]
    cuda_names = state["cuda_rng_tensors"]
    if (
        cpu_name != "rng.cpu"
        or cuda_names != [f"rng.cuda.{index}" for index in range(len(cuda_names))]
        or set(all_values) != indexed_tensors | {cpu_name, *cuda_names}
    ):
        raise ValueError("checkpoint RNG tensor inventory is malformed")
    rng_values = [all_values[cpu_name], *(all_values[name] for name in cuda_names)]
    expected_cuda_states = cuda_device_count
    if len(cuda_names) != expected_cuda_states:
        raise ValueError("checkpoint CUDA RNG topology differs from the destination")
    _validate_rng_state(rng_values[0], "cpu")
    for index, value in enumerate(rng_values[1:]):
        _validate_rng_state(value, f"cuda:{index}")
    scaler_state = _validate_scaler_state(
        state["scaler_state"], enabled=(device.type == "cuda" and spec.dtype == "float16")
    )

    # Commit only after every externally supplied byte and index has been validated.
    # Every committed tensor is COPIED off the safetensors file mapping:
    # ``load_file`` returns mmap-backed views, and a plain ``.to(target)`` is a
    # no-op alias when the tensor is already on ``target``.  A retained view
    # keeps the checkpoint file mapped; Windows cannot delete a mapped file, so
    # generation pruning and TemporaryDirectory cleanup fail with
    # PermissionError (POSIX unlinks mapped files silently, hiding the leak).
    missing, unexpected = model.load_state_dict(model_values, strict=False)
    allowed_missing = ["lm_head.weight"] if model.spec.tied_embeddings else []
    if list(missing) != allowed_missing or unexpected:  # defensive; prevalidation pins this
        raise ValueError(
            f"checkpoint model keys mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    model.context_length = spec.context_length
    optimizer.state.clear()
    for parameter, state_name, value in restored:
        target = torch.device("cpu") if state_name == "step" else device
        optimizer.state.setdefault(parameter, {})[state_name] = value.to(target, copy=True)
    return LoadedCheckpoint(
        directory,
        manifest_digest,
        global_step,
        batch_index,
        float(initial_loss),
        scaler_state,
        all_values[cpu_name].clone(),
        tuple(all_values[name].clone() for name in cuda_names),
        manifest["model_sha256"],
        manifest["optimizer_sha256"],
    )
