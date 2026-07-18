"""Optional PyTorch GNN/Transformer policy for bounded BCIR hardware-plan search."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
    raise ModuleNotFoundError(
        "hosted hardware-policy training requires bcir[model-lab]") from exc

from ..._artifact_json import strict_json_loads
from ...kbcir.hardware_rl import (
    Q20,
    CandidateFeature,
    HardwareEpisode,
    HardwareRewardPolicy,
    HardwareTopology,
    TelemetryToken,
    outcomes_to_preferences,
)
from ..models.checkpoint import tensor_mapping_digest
from .contracts import canonical_json, require_sha256, sha256_text
from .hardware_policy_spec import (
    HardwarePolicyArtifact,
    HardwarePolicyEvent,
    HardwarePolicyRunReport,
    HardwarePolicySpec,
    HardwarePolicyTrainSpec,
)

_ARTIFACT_FILE_LIMITS = {
    "config.json": 1024 * 1024,
    "hardware-policy.safetensors": 128 * 1024 * 1024,
    "report.json": 1024 * 1024,
}
_MAX_EPISODES = 1024
_MAX_TRAIN_TOKENS = 65_536
_MAX_LOGIT_CELLS = 262_144
_MAX_PREFERENCE_PAIRS = 262_144


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block: break
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush(); os.fsync(stream.fileno())


@contextmanager
def _deterministic(seed: int):
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


class HardwarePolicyNetwork(nn.Module):
    """GNN memory-topology encoder + temporal Transformer + plan heads."""

    TELEMETRY_DIM = 15
    TOPOLOGY_DIM = 7
    LINK_DIM = 2
    CANDIDATE_DIM = 13

    def __init__(self, spec: HardwarePolicySpec):
        super().__init__()
        if not isinstance(spec, HardwarePolicySpec):
            raise ValueError("hardware policy requires HardwarePolicySpec")
        self.spec = spec; hidden = spec.hidden_dim
        self.telemetry_projection = nn.Linear(self.TELEMETRY_DIM, hidden)
        self.position = nn.Parameter(torch.zeros(spec.max_telemetry_tokens, hidden))
        layer = nn.TransformerEncoderLayer(
            hidden, spec.heads, dim_feedforward=hidden * 2, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, spec.transformer_layers,
                                              enable_nested_tensor=False)
        self.topology_projection = nn.Linear(self.TOPOLOGY_DIM, hidden)
        self.link_projection = nn.Linear(self.LINK_DIM, hidden, bias=False)
        self.graph_updates = nn.ModuleList(
            nn.Linear(hidden * 2, hidden) for _ in range(spec.graph_layers))
        self.candidate_projection = nn.Linear(self.CANDIDATE_DIM, hidden)
        self.bank_role_fusion = nn.Linear(hidden * 3, hidden, bias=False)
        self.plan_fusion = nn.Linear(hidden * 3, hidden)
        self.context_norm = nn.LayerNorm(hidden)
        self.policy_head = nn.Linear(hidden, 1)
        self.reward_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def _graph(self, nodes, edge_index, edge_features):
        state = torch.tanh(self.topology_projection(nodes))
        for update in self.graph_updates:
            aggregate = torch.zeros_like(state)
            degree = torch.zeros((state.shape[0], 1), dtype=state.dtype, device=state.device)
            if edge_index.numel():
                source, destination = edge_index[0], edge_index[1]
                messages = state[source] + self.link_projection(edge_features)
                aggregate.index_add_(0, destination, messages)
                degree.index_add_(0, destination, torch.ones(
                    (destination.shape[0], 1), dtype=state.dtype, device=state.device))
            aggregate = aggregate / degree.clamp_min(1.0)
            state = torch.tanh(update(torch.cat((state, aggregate), dim=-1)))
        return state

    def forward(self, telemetry, telemetry_padding, nodes, edge_index,
                edge_features, candidates, candidate_banks):
        if telemetry.shape[-1] != self.TELEMETRY_DIM \
                or candidates.shape[-1] != self.CANDIDATE_DIM \
                or candidate_banks.shape != (candidates.shape[0], 3) \
                or candidate_banks.dtype != torch.long \
                or bool((candidate_banks < -1).any()) \
                or bool((candidate_banks >= nodes.shape[0]).any()):
            raise ValueError("hardware policy feature dimensions are incompatible")
        length = telemetry.shape[1]
        if length > self.spec.max_telemetry_tokens:
            raise ValueError("telemetry sequence exceeds policy context")
        temporal = self.telemetry_projection(telemetry) + self.position[:length]
        temporal = self.temporal(temporal, src_key_padding_mask=telemetry_padding)
        valid = (~telemetry_padding).to(temporal.dtype).unsqueeze(-1)
        temporal_context = (temporal * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        graph_nodes = self._graph(nodes, edge_index, edge_features)
        graph_context = graph_nodes.mean(dim=0)
        context = self.context_norm(temporal_context + graph_context.unsqueeze(0))
        role_mask = (candidate_banks >= 0).to(graph_nodes.dtype).unsqueeze(-1)
        role_nodes = graph_nodes[candidate_banks.clamp_min(0)] * role_mask
        bank_context = self.bank_role_fusion(role_nodes.flatten(start_dim=1))
        candidate_base = torch.tanh(
            self.candidate_projection(candidates) + bank_context).unsqueeze(0).expand(
            context.shape[0], -1, -1)
        context_base = context.unsqueeze(1).expand(-1, candidate_base.shape[1], -1)
        candidate_state = torch.tanh(self.plan_fusion(torch.cat(
            (candidate_base, context_base, candidate_base * context_base), dim=-1)))
        logits = self.policy_head(candidate_state).squeeze(-1)
        predicted_utility = self.reward_head(candidate_state).squeeze(-1)
        values = torch.sigmoid(self.value_head(context).squeeze(-1))
        return logits, values, predicted_utility


def _scaled_telemetry(token: TelemetryToken) -> list[float]:
    values = token.feature_vector
    return [values[0] / 63.0, values[1] / 63.0,
            *[value / 100.0 for value in values[2:]]]


def _scaled_node(row) -> list[float]:
    values = row.feature_vector
    return [values[0] / 16.0, values[1] / 63.0, values[2] / Q20,
            values[3] / 63.0, values[4] / 63.0,
            values[5] / 31.0, values[6] / 31.0]


def _scaled_candidate(row: CandidateFeature) -> list[float]:
    values = row.feature_vector
    return [values[0] / 2.0, float(values[1]), values[2] / 2.0,
            values[3] / 32.0, values[4] / Q20,
            values[5] / 63.0, values[6] / 63.0,
            values[7] / 63.0, values[8] / 63.0, values[9] / 4096.0,
            values[10] / 32.0, values[11] / 32.0, values[12] / 32.0]


def _inputs(topology: HardwareTopology, candidates, telemetry_rows):
    candidates = tuple(candidates); episodes = tuple(telemetry_rows)
    max_length = max(len(rows) for rows in episodes)
    telemetry = torch.zeros((len(episodes), max_length,
                             HardwarePolicyNetwork.TELEMETRY_DIM), dtype=torch.float32)
    padding = torch.ones((len(episodes), max_length), dtype=torch.bool)
    for index, rows in enumerate(episodes):
        for position, token in enumerate(rows):
            telemetry[index, position] = torch.tensor(_scaled_telemetry(token))
            padding[index, position] = False
    nodes = torch.tensor([_scaled_node(row) for row in topology.nodes], dtype=torch.float32)
    if topology.links:
        edge_index = torch.tensor([[row.source for row in topology.links],
                                   [row.destination for row in topology.links]], dtype=torch.long)
        edge_features = torch.tensor([[row.bandwidth_log2 / 63.0, row.latency_log2 / 63.0]
                                      for row in topology.links], dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_features = torch.empty((0, 2), dtype=torch.float32)
    candidate_tensor = torch.tensor([_scaled_candidate(row) for row in candidates],
                                    dtype=torch.float32)
    candidate_banks = torch.tensor([
        (row.compute_bank0 - 1,
         row.compute_bank1 - 1 if row.compute_bank1 else -1,
         row.backing_bank - 1) for row in candidates], dtype=torch.long)
    return (telemetry, padding, nodes, edge_index, edge_features,
            candidate_tensor, candidate_banks)


def _validate_candidate_topology(topology, candidates) -> None:
    node_count = len(topology.nodes)
    for row in candidates:
        bank_ids = (row.compute_bank0, row.backing_bank) + (
            (row.compute_bank1,) if row.compute_bank1 else ())
        if any(bank_id > node_count for bank_id in bank_ids):
            raise ValueError("candidate feature references a missing topology bank")


def _input_digest(topology, candidates, episodes, reward_policy) -> str:
    return sha256_text(canonical_json({
        "topology": {"hardware_digest": topology.hardware_digest,
                     "nodes": [asdict(row) for row in topology.nodes],
                     "links": [asdict(row) for row in topology.links]},
        "candidates": [asdict(row) for row in candidates],
        "episodes": [row.to_dict() for row in episodes],
        "reward_policy": asdict(reward_policy)}))


def _validate(topology, candidates, episodes, reward_policy, model_spec):
    if not isinstance(topology, HardwareTopology) \
            or not isinstance(reward_policy, HardwareRewardPolicy):
        raise ValueError("hardware-policy inputs have invalid contract types")
    try:
        candidates = tuple(candidates); episodes = tuple(episodes)
    except TypeError as exc:
        raise ValueError("hardware policy candidates/episodes must be bounded sequences") from exc
    if (len(candidates) < 2 or len(candidates) > 256
            or any(not isinstance(row, CandidateFeature) for row in candidates)
            or len({row.candidate_id for row in candidates}) != len(candidates)):
        raise ValueError("hardware policy needs 2..256 unique typed candidates")
    if (not episodes or len(episodes) > _MAX_EPISODES
            or any(not isinstance(row, HardwareEpisode) for row in episodes)):
        raise ValueError("hardware policy needs a bounded nonempty episode sequence")
    if sum(len(row.telemetry) for row in episodes) > _MAX_TRAIN_TOKENS \
            or len(episodes) * len(candidates) > _MAX_LOGIT_CELLS \
            or len(episodes) * len(candidates) * (len(candidates) - 1) // 2 \
            > _MAX_PREFERENCE_PAIRS:
        raise ValueError("hardware policy training inventory exceeds its aggregate bound")
    _validate_candidate_topology(topology, candidates)
    candidate_ids = {row.candidate_id for row in candidates}
    for episode in episodes:
        if episode.hardware_digest != topology.hardware_digest:
            raise ValueError("episode hardware does not match topology")
        if episode.reward_policy_digest != reward_policy.digest:
            raise ValueError("episode reward policy does not match the trainer")
        if {row.candidate_id for row in episode.outcomes} != candidate_ids:
            raise ValueError("every episode must cover the candidate portfolio exactly")
        if len(episode.telemetry) > model_spec.max_telemetry_tokens:
            raise ValueError("episode telemetry exceeds model context")
        scores = {reward_policy.score(row) for row in episode.outcomes}
        if len(scores) < 2:
            raise ValueError("an episode must contain at least one strict metric preference")
    return tuple(sorted(candidates, key=lambda row: row.candidate_id)), episodes


def _targets(episodes, candidates, reward_policy):
    index = {row.candidate_id: position for position, row in enumerate(candidates)}
    values = torch.zeros((len(episodes), len(candidates)), dtype=torch.float32)
    best = []
    preferences = []
    for episode_index, episode in enumerate(episodes):
        scores = {row.candidate_id: reward_policy.score(row) for row in episode.outcomes}
        low, high = min(scores.values()), max(scores.values())
        scale = max(1, high - low)
        for candidate_id, score in scores.items():
            values[episode_index, index[candidate_id]] = (high - score) / scale
        best.append(min(scores, key=lambda candidate_id: (scores[candidate_id], candidate_id)))
        for preference in outcomes_to_preferences(episode.outcomes, reward_policy):
            preferences.append((episode_index, index[preference.chosen_id],
                                index[preference.rejected_id]))
    return values, torch.tensor([index[row] for row in best], dtype=torch.long), preferences


def _accuracy(logits, best) -> int:
    return int((logits.argmax(dim=-1) == best).sum().item())


def _emit(sink, phase, step, loss, norm):
    event = HardwarePolicyEvent(phase, step, float(loss), float(norm))
    if sink is not None: sink(event)


@dataclass(frozen=True)
class HardwarePolicyTrainingResult:
    model: HardwarePolicyNetwork
    report: HardwarePolicyRunReport
    artifact: HardwarePolicyArtifact


def _export(model, model_spec, train_spec, input_digest, report, output_root) \
        -> HardwarePolicyArtifact:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".hardware-policy-", dir=root))
    try:
        config = canonical_json({"schema": "bcir.hardware_policy_artifact.v1",
                                 "model_spec": asdict(model_spec),
                                 "train_spec_sha256": train_spec.digest,
                                 "input_sha256": input_digest})
        _write_text(temporary / "config.json", config)
        state = {name: value.detach().cpu().contiguous()
                 for name, value in sorted(model.state_dict().items())}
        save_file(state, str(temporary / "hardware-policy.safetensors"))
        _write_text(temporary / "report.json", report.to_json())
        files = {}
        for name in ("config.json", "hardware-policy.safetensors", "report.json"):
            path = temporary / name; size = path.stat().st_size
            if not 1 <= size <= _ARTIFACT_FILE_LIMITS[name]:
                raise ValueError(f"hardware policy artifact file {name!r} is out of bounds")
            files[name] = {"bytes": size, "sha256": _sha_file(path)}
        manifest_text = canonical_json({
            "schema": "bcir.hardware_policy_manifest.v1", "files": files,
            "model_spec_sha256": model_spec.digest,
            "train_spec_sha256": train_spec.digest, "input_sha256": input_digest,
            "model_sha256": report.model_sha256})
        _write_text(temporary / "manifest.json", manifest_text)
        manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        destination = root / manifest_sha
        if destination.exists():
            loaded, _spec, artifact = load_hardware_policy(destination)
            if artifact.manifest_sha256 != manifest_sha \
                    or tensor_mapping_digest(loaded.state_dict()) != report.model_sha256:
                raise ValueError("existing content-addressed hardware policy is inconsistent")
            shutil.rmtree(temporary)
            return artifact
        os.replace(temporary, destination)
        return HardwarePolicyArtifact(str(destination), manifest_sha, report.model_sha256)
    except Exception:
        if temporary.exists(): shutil.rmtree(temporary)
        raise


def train_hardware_policy(model_spec: HardwarePolicySpec,
                          train_spec: HardwarePolicyTrainSpec,
                          topology: HardwareTopology, candidates, episodes,
                          reward_policy: HardwareRewardPolicy, output_root,
                          *, telemetry_sink=None) -> HardwarePolicyTrainingResult:
    """Train a tiny reward/DPO/PPO policy and atomically export a safe artifact."""
    if not isinstance(model_spec, HardwarePolicySpec) \
            or not isinstance(train_spec, HardwarePolicyTrainSpec):
        raise ValueError("hardware-policy trainer requires typed specifications")
    if telemetry_sink is not None and not callable(telemetry_sink):
        raise ValueError("telemetry_sink must be callable or None")
    candidates, episodes = _validate(
        topology, candidates, episodes, reward_policy, model_spec)
    input_digest = _input_digest(topology, candidates, episodes, reward_policy)
    tensors = _inputs(topology, candidates, [row.telemetry for row in episodes])
    targets, best, preferences = _targets(episodes, candidates, reward_policy)

    with _deterministic(train_spec.seed):
        model = HardwarePolicyNetwork(model_spec)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_spec.learning_rate,
            betas=(train_spec.beta1, train_spec.beta2), eps=train_spec.epsilon,
            weight_decay=train_spec.weight_decay, amsgrad=False)

        def forward(current=model): return current(*tensors)
        with torch.no_grad():
            initial_logits, _initial_values, initial_rewards = forward()
            initial_loss = float(F.mse_loss(torch.sigmoid(initial_rewards), targets).item())
            initial_accuracy = _accuracy(initial_logits, best)

        for step in range(train_spec.reward_steps):
            optimizer.zero_grad(set_to_none=True)
            _logits, _values, predicted = forward()
            loss = F.mse_loss(torch.sigmoid(predicted), targets)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_spec.grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError("hardware reward training produced non-finite state")
            optimizer.step(); _emit(telemetry_sink, "reward", step + 1,
                                    loss.detach().item(), norm.item())

        reference = copy.deepcopy(model).eval()
        with torch.no_grad(): reference_logits = reference(*tensors)[0]
        for step in range(train_spec.dpo_steps):
            optimizer.zero_grad(set_to_none=True)
            logits, _values, predicted = forward()
            terms = []
            for episode_index, chosen, rejected in preferences:
                delta = logits[episode_index, chosen] - logits[episode_index, rejected]
                reference_delta = (reference_logits[episode_index, chosen]
                                   - reference_logits[episode_index, rejected])
                terms.append(-F.logsigmoid(train_spec.dpo_beta * (delta - reference_delta)))
            preference_loss = torch.stack(terms).mean()
            reward_loss = F.mse_loss(torch.sigmoid(predicted), targets)
            loss = preference_loss + 0.1 * reward_loss
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_spec.grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError("hardware DPO training produced non-finite state")
            optimizer.step(); _emit(telemetry_sink, "dpo", step + 1,
                                    loss.detach().item(), norm.item())

        old_policy = copy.deepcopy(model).eval()
        with torch.no_grad(): old_probabilities = old_policy(*tensors)[0].softmax(dim=-1)
        for step in range(train_spec.ppo_steps):
            optimizer.zero_grad(set_to_none=True)
            logits, values, predicted = forward()
            probabilities = logits.softmax(dim=-1)
            selected = probabilities.gather(1, best[:, None]).squeeze(1)
            old_selected = old_probabilities.gather(1, best[:, None]).squeeze(1)
            ratio = selected / old_selected.clamp_min(1e-12)
            advantage = (torch.ones_like(values) - values).detach()
            unclipped = ratio * advantage
            clipped = ratio.clamp(1.0 - train_spec.ppo_clip,
                                  1.0 + train_spec.ppo_clip) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, torch.ones_like(values))
            reward_loss = F.mse_loss(torch.sigmoid(predicted), targets)
            loss = policy_loss + train_spec.value_coefficient * value_loss + 0.1 * reward_loss
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_spec.grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError("hardware PPO training produced non-finite state")
            optimizer.step(); _emit(telemetry_sink, "ppo", step + 1,
                                    loss.detach().item(), norm.item())

        model.eval()
        with torch.no_grad():
            final_logits, _values, final_rewards = forward()
            final_loss = float(F.mse_loss(torch.sigmoid(final_rewards), targets).item())
            final_accuracy = _accuracy(final_logits, best)
        if not all(math.isfinite(value) for value in (initial_loss, final_loss)):
            raise RuntimeError("hardware policy report contains non-finite loss")
        model_sha = tensor_mapping_digest(model.state_dict())
        report = HardwarePolicyRunReport(
            model_spec.digest, train_spec.digest, input_digest, initial_loss, final_loss,
            initial_accuracy, final_accuracy, len(episodes), len(candidates),
            train_spec.reward_steps + train_spec.dpo_steps + train_spec.ppo_steps,
            model_sha)
        artifact = _export(model, model_spec, train_spec, input_digest, report, output_root)
        return HardwarePolicyTrainingResult(model, report, artifact)


def hardware_policy_priors_q20(model: HardwarePolicyNetwork, topology: HardwareTopology,
                               candidates, telemetry) -> dict[str, int]:
    try:
        candidates = tuple(candidates)
    except TypeError as exc:
        raise ValueError("hardware policy candidates must be a bounded sequence") from exc
    try:
        rows = tuple(telemetry)
    except TypeError as exc:
        raise ValueError("hardware policy telemetry must be a bounded sequence") from exc
    if (not isinstance(model, HardwarePolicyNetwork)
            or not isinstance(topology, HardwareTopology)
            or not rows or len(rows) > model.spec.max_telemetry_tokens
            or any(not isinstance(row, TelemetryToken) for row in rows)
            or len(candidates) < 1 or len(candidates) > 256
            or any(not isinstance(row, CandidateFeature) for row in candidates)
            or len({row.candidate_id for row in candidates}) != len(candidates)):
        raise ValueError("hardware policy inference requires a model and telemetry")
    candidates = tuple(sorted(candidates, key=lambda row: row.candidate_id))
    sequences = tuple(row.sequence for row in rows)
    if any(right <= left for left, right in zip(sequences, sequences[1:])):
        raise ValueError("hardware policy inference telemetry must be strictly increasing")
    _validate_candidate_topology(topology, candidates)
    tensors = _inputs(topology, candidates, (rows,))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(): probabilities = model(*tensors)[0][0].softmax(dim=-1)
    finally:
        model.train(was_training)
    values = [max(1, int(round(float(value.item()) * Q20))) for value in probabilities]
    return {row.candidate_id: values[index] for index, row in enumerate(candidates)}


def load_hardware_policy(directory) -> tuple[HardwarePolicyNetwork, HardwarePolicySpec,
                                             HardwarePolicyArtifact]:
    directory = Path(directory)
    expected = {"config.json", "hardware-policy.safetensors", "report.json", "manifest.json"}
    if directory.is_symlink() or not directory.is_dir() \
            or {path.name for path in directory.iterdir()} != expected:
        raise ValueError("hardware policy artifact file inventory is incomplete or ambiguous")
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() \
            or not 1 <= manifest_path.stat().st_size <= 1024 * 1024:
        raise ValueError("hardware policy manifest is not a bounded regular file")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = strict_json_loads(manifest_text, "hardware policy manifest", max_bytes=1024 * 1024)
    fields = {"schema", "files", "model_spec_sha256", "train_spec_sha256",
              "input_sha256", "model_sha256"}
    if not isinstance(manifest, dict) or set(manifest) != fields \
            or manifest["schema"] != "bcir.hardware_policy_manifest.v1":
        raise ValueError("hardware policy manifest has missing or unknown fields")
    if not isinstance(manifest["files"], dict) \
            or set(manifest["files"]) != expected - {"manifest.json"}:
        raise ValueError("hardware policy manifest file census is invalid")
    for field in ("model_spec_sha256", "train_spec_sha256", "input_sha256",
                  "model_sha256"):
        require_sha256(manifest[field], f"hardware policy manifest {field}")
    for name, record in manifest["files"].items():
        path = directory / name
        if (not isinstance(record, dict) or set(record) != {"bytes", "sha256"}
                or type(record["bytes"]) is not int
                or not 1 <= record["bytes"] <= _ARTIFACT_FILE_LIMITS[name]
                or require_sha256(record["sha256"], f"hardware policy file {name} sha256")
                != record["sha256"]
                or path.is_symlink() or not path.is_file()
                or record["bytes"] != path.stat().st_size
                or record["sha256"] != _sha_file(path)):
            raise ValueError(f"hardware policy artifact file {name!r} failed integrity")
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    if directory.name != manifest_sha:
        raise ValueError("hardware policy directory is not content-addressed by its manifest")
    config = strict_json_loads((directory / "config.json").read_text(encoding="utf-8"),
                               "hardware policy config", max_bytes=1024 * 1024)
    if (not isinstance(config, dict)
            or set(config) != {"schema", "model_spec", "train_spec_sha256", "input_sha256"}
            or config["schema"] != "bcir.hardware_policy_artifact.v1"):
        raise ValueError("hardware policy config has missing or unknown fields")
    require_sha256(config["train_spec_sha256"], "hardware policy config train_spec_sha256")
    require_sha256(config["input_sha256"], "hardware policy config input_sha256")
    try: model_spec = HardwarePolicySpec(**config["model_spec"])
    except (TypeError, KeyError) as exc:
        raise ValueError(f"malformed hardware policy specification: {exc}") from exc
    if (model_spec.digest != manifest["model_spec_sha256"]
            or config["train_spec_sha256"] != manifest["train_spec_sha256"]
            or config["input_sha256"] != manifest["input_sha256"]):
        raise ValueError("hardware policy config and manifest disagree")
    report = HardwarePolicyRunReport.from_json(
        (directory / "report.json").read_text(encoding="utf-8"))
    if (report.model_spec_sha256 != model_spec.digest
            or report.train_spec_sha256 != manifest["train_spec_sha256"]
            or report.input_sha256 != manifest["input_sha256"]
            or report.model_sha256 != manifest["model_sha256"]):
        raise ValueError("hardware policy report and manifest disagree")

    cpu_rng = torch.get_rng_state()
    try:
        model = HardwarePolicyNetwork(model_spec)
    finally:
        torch.set_rng_state(cpu_rng)
    try:
        state = load_file(str(directory / "hardware-policy.safetensors"), device="cpu")
    except Exception as exc:
        raise ValueError(f"hardware policy tensor container is invalid: {exc}") from exc
    expected_state = model.state_dict()
    if set(state) != set(expected_state) or any(
            state[name].shape != expected_state[name].shape
            or state[name].dtype != expected_state[name].dtype for name in expected_state):
        raise ValueError("hardware policy tensor schema is invalid")
    model_sha = tensor_mapping_digest(state)
    if model_sha != manifest["model_sha256"]:
        raise ValueError("hardware policy tensor digest does not match manifest")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"hardware policy tensor schema is invalid: {exc}") from exc
    model.eval()
    artifact = HardwarePolicyArtifact(str(directory), manifest_sha, model_sha)
    return model, model_spec, artifact


__all__ = ["HardwarePolicyNetwork", "HardwarePolicyTrainingResult",
           "hardware_policy_priors_q20", "load_hardware_policy",
           "train_hardware_policy"]
