"""Bounded, dependency-free performance/correctness audit for BCIR data paths.

Theoretical maximum performance is a hardware-specific upper bound, not a property a
portable Python run can certify.  This rail therefore gives TMSAO an operational,
falsifiable meaning: exercise representative BCIR graph, GEM, StreamPack, memory,
telemetry, quantization, ML, and hardware-search organs; require deterministic results;
and record host-relative timings without making shared-CI latency a legality gate.

The report has no timestamp or hostname.  Its definition and correctness digests exclude
timings, so repeated runs can distinguish semantic drift from measurement noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_SCHEMA = "bcir.tmsao_audit.v1"
_MAX_SCALE = 4
_MAX_REPEATS = 9


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _q(value: float, places: int = 9) -> int:
    """Stable fixed-point witness for a finite numeric result."""
    value = float(value)
    if not math.isfinite(value):
        raise AssertionError("performance audit produced a non-finite numeric result")
    return int(round(value * (10 ** places)))


def _validate_result(value, field: str = "result"):
    """Return a JSON-safe immutable projection, rejecting opaque/non-finite output."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if type(value) is int:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssertionError(f"{field} is not finite")
        return value
    if isinstance(value, (tuple, list)):
        return [_validate_result(row, f"{field}[]") for row in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) or not key for key in value):
            raise AssertionError(f"{field} has a non-string or empty key")
        return {key: _validate_result(value[key], f"{field}.{key}")
                for key in sorted(value)}
    raise AssertionError(f"{field} has unsupported type {type(value).__name__}")


@dataclass(frozen=True)
class AuditSample:
    group: str
    name: str
    items: int
    repeats: int
    min_ns: int
    median_ns: int
    max_ns: int
    result_sha256: str
    metrics: dict


@dataclass(frozen=True)
class TMSAOAuditReport:
    host: dict
    scale: int
    repeats: int
    definition_sha256: str
    correctness_sha256: str
    samples: tuple[AuditSample, ...]
    schema: str = _SCHEMA
    timing_is_gate: bool = False

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "timing_is_gate": self.timing_is_gate,
            "host": self.host,
            "scale": self.scale,
            "repeats": self.repeats,
            "definition_sha256": self.definition_sha256,
            "correctness_sha256": self.correctness_sha256,
            "samples": [asdict(sample) for sample in self.samples],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)


def _values(count: int, *, modulus: int = 31, divisor: float = 64.0) -> list[float]:
    return [((index * 17 + 11) % modulus - modulus // 2) / divisor
            for index in range(count)]


def _case_phase_dag(scale: int) -> dict:
    from .gem.execute import execute
    from .model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
    from .verify import verify

    count = 512 * scale
    module = Module(name="tmsao_deep_phase_dag")
    module.add_resource(Resource(1, Domain.RAM, shape=(1,), name="state"))
    phases = []
    for index in range(count):
        claim = Claim(index + 1, Opcode.ADD, Lane.U, StrideClass.UNIT,
                      count=1, rd=(1,), wr=(1,), op="vector.add", domain=Domain.RAM)
        phases.append(Phase(index, () if index == 0 else (index - 1,), [claim]))
    module.phases.extend(reversed(phases))
    diagnostics = verify(module)
    result = execute(module)
    if diagnostics or result.executed != count or result.phase_order != list(range(count)):
        raise AssertionError("deep phase DAG failed verification or canonical execution order")
    return {"claims": count, "diagnostics": 0, "first_phase": result.phase_order[0],
            "last_phase": result.phase_order[-1], "order_sum": sum(result.order)}


def _case_gem_schedulers(scale: int) -> dict:
    from .gem.async_tokens import async_plan
    from .gem.concurrency import schedule_concurrent
    from .gem.schedule import schedule_eft
    from .kbcir.cost import TargetProfile
    from .model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

    count = 512 * scale
    module = Module(name="tmsao_mixed_scheduler")
    for rid in range(1, 65):
        module.add_resource(Resource(rid, Domain.RAM, shape=(64,), name=f"shared{rid}"))
    for index in range(count):
        module.add_resource(Resource(1000 + index, Domain.RAM, shape=(17 + index % 13,),
                                     name=f"out{index}"))
    claims = []
    for index in range(count):
        shared = 1 + (index * 17) % 64
        writes = (shared,) if index % 19 == 0 else (1000 + index,)
        claims.append(Claim(index + 1, Opcode.ADD, Lane.U, StrideClass.UNIT,
                            count=17 + index % 13, rd=(shared,), wr=writes,
                            op="vector.add", domain=Domain.RAM,
                            cost_class="compute" if index % 5 == 0 else "bandwidth"))
    module.add_phase(Phase(0, (), claims))
    target = TargetProfile(name="tmsao-8-domain", affinity_domains=8, mem_channels=4)
    waves = schedule_concurrent(module, target)
    tokens = async_plan(module)
    durations = {claim.id: 1 + (claim.id * 13) % 29 for claim in claims}
    eft = schedule_eft(module, durations, target)
    edge_count = sum(len(rows) for rows in tokens.awaits.values())
    if len(tokens.forks) != count or len(eft.slots) != count:
        raise AssertionError("GEM scheduler omitted claims")
    return {"claims": count, "waves": len(waves.waves),
            "max_parallelism": waves.max_parallelism(), "await_edges": edge_count,
            "makespan": eft.makespan,
            "slot_checksum": sum((slot.claim_id * 31 + slot.start * 7 + slot.finish)
                                 for slot in eft.slots)}


def _case_kbcir_streampack(scale: int) -> dict:
    from .examples import matmul_tiled
    from .gem.streampack import hydrate_pipelined
    from .kbcir.cost import TargetProfile, Theta
    from .kbcir.realize import optimize
    from .verify import verify, verify_pack, verify_plan

    grid = 4 * scale
    module = matmul_tiled(n=grid * 8, tile=8)
    target = TargetProfile.x86_avx2()
    result = optimize(module, target, Theta.mem_bound())
    pack = hydrate_pipelined(module, result, plan="tmsao", depth=2)
    diagnostics = verify(module) + verify_plan(module, result) + verify_pack(module, pack)
    if diagnostics or not pack.provenance_ok() or len(pack.segments) != len(result.steps):
        raise AssertionError("K_BCIR -> StreamPack chain failed its laws")
    return {"claims": len(result.steps), "score": result.score,
            "segments": len(pack.segments), "prefetches": len(pack.prefetches),
            "blocks": len(pack.blocks), "trace_notes": len(pack.trace_notes)}


@dataclass(frozen=True)
class _AuditBank:
    name: str = "ram"
    domain: str = "RAM"
    allocatable_bytes: int = 1 << 30
    alignment: int = 64


class _AuditHardware:
    digest = hashlib.sha256(b"bcir-tmsao-static-memory-v1").hexdigest()

    def __init__(self) -> None:
        self._bank = _AuditBank()

    def bank(self, name: str) -> _AuditBank:
        if name != self._bank.name:
            raise KeyError(name)
        return self._bank


def _case_static_memory(scale: int) -> dict:
    from .kbcir.static_memory import plan_static_memory, verify_static_memory_plan
    from .model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

    resources = 512 * scale
    phase_count = 128 * scale
    module = Module(name="tmsao_static_memory")
    phases = [Phase(index, () if index == 0 else (index - 1,), [])
              for index in range(phase_count)]
    claim_id = 1
    for index in range(resources):
        rid = index + 1
        module.add_resource(Resource(rid, Domain.RAM, elem_bytes=4,
                                     shape=(1 + (index * 11) % 63,), align=64,
                                     name=f"tensor{rid}"))
        first = (index * 37) % phase_count
        last = min(phase_count - 1, first + (index * 7) % 17)
        for phase_id in ((first,) if first == last else (first, last)):
            phases[phase_id].claims.append(Claim(
                claim_id, Opcode.ADD, Lane.U, StrideClass.UNIT,
                count=1, rd=(rid,), op="tensor.touch", domain=Domain.RAM))
            claim_id += 1
    module.phases.extend(reversed(phases))
    hardware = _AuditHardware()
    bindings = {rid: "ram" for rid in module.resources}
    plan = plan_static_memory(module, bindings, hardware)
    errors = verify_static_memory_plan(plan, module, bindings, hardware)
    if errors:
        raise AssertionError("static memory plan failed independent verification")
    bank = plan.banks[0]
    return {"resources": resources, "phases": phase_count,
            "extent_bytes": bank.extent_bytes, "naive_bytes": bank.naive_bytes,
            "saved_bytes": bank.saved_bytes,
            "offset_checksum": sum(row.offset for row in plan.allocations)}


def _case_telemetry_ring(scale: int) -> dict:
    from .telemetry import DataDNA, TelemetryRing

    capacity = 512 * scale
    writes = capacity * 2 + 17
    ring = TelemetryRing(capacity)
    for index in range(writes):
        ring.write(DataDNA("tmsao", index, cycles=1000 + index,
                           bytes=64 * (1 + index % 31), misses=index % 101,
                           thermal=(index * 3) % 101, voltage=(index * 5) % 101,
                           utilization=(index * 7) % 101, provenance="synthetic"))
    rows = ring.drain()
    if len(rows) != capacity or ring.stats.dropped != writes - capacity or ring.pending:
        raise AssertionError("telemetry overwrite/backpressure accounting is inconsistent")
    return {"capacity": capacity, "writes": writes, "reads": len(rows),
            "dropped": ring.stats.dropped, "first_claim": rows[0].claim_id,
            "cycle_checksum": sum(row.cycles for row in rows)}


def _case_lowbit(scale: int) -> dict:
    from .kbcir.lowbit import PackedQ4Tensor
    from .kbcir.quantize import dequantize, quantize_per_group

    count = 4096 * scale
    values = [((index * 7919) % 2048 - 1024) / 128.0 for index in range(count)]
    q8 = quantize_per_group(values, 32, 8)
    restored = dequantize(q8)
    source = hashlib.sha256(b"tmsao-source").hexdigest()
    calibration = hashlib.sha256(b"tmsao-calibration").hexdigest()
    q4 = PackedQ4Tensor.from_values(values, (count,), source_sha256=source,
                                    calibration_sha256=calibration)
    if len(restored) != count or q4.element_count != count:
        raise AssertionError("low-bit tensor census mismatch")
    return {"elements": count, "q8_groups": len(q8), "q4_bytes": len(q4.packed_codes),
            "q8_error_q9": _q(max(abs(left - right)
                                      for left, right in zip(values, restored))),
            "q4_code_checksum": sum(q4.codes)}


def _case_unsupervised(scale: int) -> dict:
    from .kbcir.classical import KnnModel, knn_neighbours
    from .kbcir.unsupervised import (EmbeddingTable, embedding_lookup, kmeans_fit,
                                     kmeans_inertia, standard_scaler_fit,
                                     standard_scaler_transform)

    samples = 256 * scale
    features = 8
    clusters = 8
    data = []
    for index in range(samples):
        cluster = index % clusters
        data.extend(cluster * 4.0 + ((index * 13 + feature * 7) % 17) / 17.0
                    for feature in range(features))
    centroids, labels = kmeans_fit(data, samples, features, clusters,
                                   list(range(clusters)), 6)
    inertia = kmeans_inertia(data, samples, features, centroids, labels)
    model = KnnModel(tuple(data), tuple(float(index % clusters)
                                       for index in range(samples)), features)
    neighbours = knn_neighbours(model, data[7 * features:8 * features], 7)
    scaler = standard_scaler_fit(data, samples, features)
    standardized = standard_scaler_transform(data[:features], scaler.mean, scaler.std)
    vocab = 512 * scale
    dim = 16
    table = EmbeddingTable(tuple(_values(vocab * dim)), vocab, dim)
    ids = [((index * 97) ^ (index >> 2)) % vocab for index in range(256 * scale)]
    embedded = embedding_lookup(table, ids, dim)
    return {"samples": samples, "features": features, "clusters": clusters,
            "inertia_q9": _q(inertia), "label_checksum": sum(labels),
            "neighbour_checksum": sum(neighbours),
            "standardized_sum_q9": _q(sum(standardized)),
            "embedding_values": len(embedded), "embedding_sum_q9": _q(sum(embedded))}


def _case_matmul(scale: int) -> dict:
    from .kbcir.matmul import matmul_reference, matmul_tiled, plan_matmul

    size = 24 * scale
    left = [float((index * 7) % 11 - 5) for index in range(size * size)]
    right = [float((index * 13) % 9 - 4) for index in range(size * size)]
    plan = plan_matmul(size, size, size)
    tiled = matmul_tiled(left, right, size, size, size, plan)
    reference = matmul_reference(left, right, size, size, size)
    if tiled != reference:
        raise AssertionError("integer-valued tiled matmul diverged from its reference")
    return {"dimension": size, "elements": len(tiled), "tile_m": plan.tile_m,
            "tile_n": plan.tile_n, "tile_k": plan.tile_k,
            "bottleneck": plan.bottleneck, "output_sum": int(sum(tiled))}


def _case_ols_pca(scale: int) -> dict:
    from .kbcir.ols import normal_equation_residual, ols_reference
    from .kbcir.pca import covariance_matrix, eigen_residual, pca_reference

    rows = 64 * scale
    columns = 6
    design = []
    for row in range(rows):
        design.extend((1.0 if row % columns == column else 0.0)
                      + ((row * 11 + column * 5) % 19) / 100.0
                      for column in range(columns))
    truth = [float(index + 1) for index in range(columns)]
    rhs = [sum(design[row * columns + column] * truth[column]
               for column in range(columns)) for row in range(rows)]
    estimate = ols_reference(design, rhs, rows, columns)
    ols_residual = normal_equation_residual(design, estimate, rhs, rows, columns)
    observations = [((row * 17 + column * 29) % 101) / 37.0
                    + column * (row % 7) / 50.0
                    for row in range(rows) for column in range(columns)]
    eigenvalues, eigenvectors = pca_reference(observations, rows, columns, 3)
    covariance = covariance_matrix(observations, rows, columns, ddof=1)
    pca_residual = eigen_residual(covariance, eigenvalues, eigenvectors, columns, 3)
    if ols_residual > 1e-8 or pca_residual > 1e-8:
        raise AssertionError("OLS/PCA independent residual exceeded its bounded fixture tolerance")
    return {"rows": rows, "columns": columns, "ols_residual_q15": _q(ols_residual, 15),
            "estimate_sum_q9": _q(sum(estimate)),
            "pca_residual_q15": _q(pca_residual, 15),
            "eigenvalue_sum_q9": _q(sum(eigenvalues))}


def _case_transformer(scale: int) -> dict:
    from .kbcir.transformer import (TransformerBlockParams, TransformerBlockSpec,
                                    transformer_block_reference)

    d_model = 16
    d_ff = 32
    sequence = 8 * scale
    spec = TransformerBlockSpec(d_model, 4, d_ff, sequence)
    matrix = lambda count, shift: [((index * 13 + shift) % 23 - 11) / 128.0
                                   for index in range(count)]
    params = TransformerBlockParams(
        matrix(d_model * d_model, 1), matrix(d_model * d_model, 2),
        matrix(d_model * d_model, 3), matrix(d_model * d_model, 4),
        matrix(d_model * d_ff, 5), [0.0] * d_ff,
        matrix(d_ff * d_model, 6), [0.0] * d_model,
        [1.0] * d_model, [0.0] * d_model, [1.0] * d_model, [0.0] * d_model)
    output = transformer_block_reference(_values(spec.token_count), spec, params,
                                         activation="gelu", pre_ln=True)
    if len(output) != spec.token_count or any(not math.isfinite(row) for row in output):
        raise AssertionError("transformer block produced malformed output")
    return {"tokens": sequence, "d_model": d_model, "output_values": len(output),
            "output_sum_q9": _q(sum(output)),
            "output_energy_q9": _q(sum(value * value for value in output))}


def _case_recurrent(scale: int) -> dict:
    from .kbcir.recurrent import GruParams, LstmParams, gru_unroll, lstm_unroll

    input_dim = hidden_dim = 8
    steps = 16 * scale
    input_weights = _values(hidden_dim * input_dim, modulus=37, divisor=96.0)
    recurrent_weights = _values(hidden_dim * hidden_dim, modulus=41, divisor=128.0)
    bias = [0.0] * hidden_dim
    lstm = LstmParams(*(list(input_weights), list(recurrent_weights), list(bias)) * 4,
                      input_dim, hidden_dim)
    gru = GruParams(*(list(input_weights), list(recurrent_weights), list(bias)) * 3,
                    input_dim, hidden_dim)
    sequence = _values(steps * input_dim, modulus=43, divisor=64.0)
    hidden, cell = lstm_unroll(sequence, [0.0] * hidden_dim,
                               [0.0] * hidden_dim, lstm)
    gru_hidden = gru_unroll(sequence, [0.0] * hidden_dim, gru)
    if any(not math.isfinite(row) for row in hidden + cell + gru_hidden):
        raise AssertionError("recurrent references produced non-finite state")
    return {"steps": steps, "hidden_dim": hidden_dim,
            "lstm_hidden_q9": _q(sum(hidden)), "lstm_cell_q9": _q(sum(cell)),
            "gru_hidden_q9": _q(sum(gru_hidden))}


def _case_training(scale: int) -> dict:
    from .kbcir.training import linear_model, make_linearly_separable, train

    dataset = make_linearly_separable(32 * scale, seed=1729)
    result = train(linear_model, [0.0, 0.0, 0.0], dataset, loss="bce",
                   optimizer="adam", epochs=8 * scale, batch_size=8, lr=0.05,
                   metrics=("accuracy",), seed=1729)
    if (not result.train_loss or result.train_loss[-1] >= result.train_loss[0]
            or any(not math.isfinite(row) for row in result.params)):
        raise AssertionError("bounded training did not reduce its deterministic fixture loss")
    return {"examples": len(dataset), "epochs": result.epochs_run,
            "initial_loss_q9": _q(result.train_loss[0]),
            "final_loss_q9": _q(result.train_loss[-1]),
            "accuracy_q9": _q(result.train_metrics["accuracy"][-1]),
            "parameter_sum_q9": _q(sum(result.params))}


def _case_hardware_search(scale: int) -> dict:
    from .kbcir.hardware_rl import Q20, bounded_mcts

    candidates = tuple(f"plan-{index:02d}" for index in range(12))
    priors = {candidate: Q20 + index * 97
              for index, candidate in enumerate(candidates)}
    utility = {candidate: Q20 - abs(index - 7) * (Q20 // 16)
               for index, candidate in enumerate(candidates)}
    result = bounded_mcts(candidates, priors, utility.__getitem__,
                          simulations=64 * scale)
    if result.candidate_id not in candidates or sum(row.visits for row in result.visits) != 64 * scale:
        raise AssertionError("bounded hardware search returned an invalid visit census")
    return {"candidates": len(candidates), "simulations": result.simulations,
            "selected": result.candidate_id,
            "visit_checksum": sum((index + 1) * row.visits
                                  for index, row in enumerate(result.visits))}


def _case_definitions(scale: int):
    return (
        ("graph", "iterative-phase-dag", 512 * scale, _case_phase_dag),
        ("gem", "mixed-wave-token-eft", 512 * scale, _case_gem_schedulers),
        ("gem", "kbcir-streampack", (4 * scale) ** 3, _case_kbcir_streampack),
        ("memory", "static-lifetime-planner", 512 * scale, _case_static_memory),
        ("telemetry", "bounded-overwrite-ring", 512 * scale, _case_telemetry_ring),
        ("quantization", "q8-q4-blocks", 4096 * scale, _case_lowbit),
        ("ml-unsupervised", "kmeans-knn-scaler-embedding", 256 * scale,
         _case_unsupervised),
        ("ml-linear-algebra", "tiled-matmul", 24 * scale, _case_matmul),
        ("ml-linear-algebra", "ols-pca", 64 * scale, _case_ols_pca),
        ("ml-sequence", "transformer-block", 8 * scale, _case_transformer),
        ("ml-sequence", "lstm-gru", 16 * scale, _case_recurrent),
        ("ml-training", "autodiff-adam", 32 * scale, _case_training),
        ("ml-hardware-policy", "bounded-mcts", 64 * scale, _case_hardware_search),
    )


def _measure(group: str, name: str, items: int, function, scale: int,
             repeats: int) -> AuditSample:
    # Warm imported modules and one-time caches outside the measured interval.
    warm = _validate_result(function(scale), name)
    expected = _sha(warm)
    timings: list[int] = []
    final = warm
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = function(scale)
        elapsed = max(1, time.perf_counter_ns() - started)
        final = _validate_result(result, name)
        if _sha(final) != expected:
            raise AssertionError(f"{name} result changed across repeated audit runs")
        timings.append(elapsed)
    ordered = sorted(timings)
    return AuditSample(group, name, items, repeats, ordered[0],
                       ordered[len(ordered) // 2], ordered[-1], expected, final)


def run_tmsao_audit(*, scale: int = 1, repeats: int = 3) -> TMSAOAuditReport:
    """Run the bounded audit and return its timestamp-free report."""
    if type(scale) is not int or not 1 <= scale <= _MAX_SCALE:
        raise ValueError(f"scale must be an integer in [1, {_MAX_SCALE}]")
    if type(repeats) is not int or not 1 <= repeats <= _MAX_REPEATS:
        raise ValueError(f"repeats must be an integer in [1, {_MAX_REPEATS}]")
    definitions = _case_definitions(scale)
    definition = [{"group": group, "name": name, "items": items}
                  for group, name, items, _function in definitions]
    samples = tuple(_measure(group, name, items, function, scale, repeats)
                    for group, name, items, function in definitions)
    correctness = [{"group": row.group, "name": row.name,
                    "result_sha256": row.result_sha256} for row in samples]
    host = {"system": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(),
            "implementation": platform.python_implementation()}
    return TMSAOAuditReport(host, scale, repeats, _sha(definition),
                            _sha(correctness), samples)


def write_tmsao_report(path: os.PathLike | str, report: TMSAOAuditReport) -> None:
    """Atomically write one report; a failed write never replaces the prior file."""
    if not isinstance(report, TMSAOAuditReport):
        raise ValueError("report must be a TMSAOAuditReport")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (report.to_json() + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bcir-tmsao-audit",
        description="Run the bounded BCIR performance/correctness audit")
    parser.add_argument("--output", default="build/performance/tmsao-report.json")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args(argv)
    report = run_tmsao_audit(scale=arguments.scale, repeats=arguments.repeats)
    write_tmsao_report(arguments.output, report)
    print(f"TMSAO audit: {len(report.samples)} cases; "
          f"correctness={report.correctness_sha256}; report={arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
