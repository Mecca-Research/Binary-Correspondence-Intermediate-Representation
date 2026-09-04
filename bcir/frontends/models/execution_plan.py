"""Lower model placement candidates into verified BCIR claims and StreamPack."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from ...abi.streampack_abi import encode
from ...gem.streampack import StreamPack, hydrate_pipelined
from ...kbcir.cost import TargetProfile, Theta
from ...kbcir.device_manifest import (
    DeviceManifest,
    MemoryBank,
    check_bank_moves,
    check_device_manifest,
    move_claim,
)
from ...kbcir.provenance import hash_module
from ...kbcir.realize import RealizationResult, optimize
from ...kbcir.static_memory import ResourceBankBinding
from ...model import Claim, Domain, Lane, Lifetime, Module, Opcode, Phase, Resource, StrideClass
from ...verify import verify_all, verify_smart_lowering
from .assessment import (
    HardwareEnvelope,
    ModelCostReport,
    ModelExecutionPlan,
    ModelWorkloadSpec,
    PlanAction,
    PlacementCandidate,
    format_tensor_sizes,
)
from .inventory import TensorInventory


@dataclass(frozen=True)
class LoweredModelExecution:
    artifact: ModelExecutionPlan
    module: Module
    realization: RealizationResult
    streampack: StreamPack
    streampack_data: bytes
    resource_banks: tuple[ResourceBankBinding, ...]


def _device_manifest(hardware: HardwareEnvelope) -> DeviceManifest:
    banks = tuple(
        MemoryBank(row.name, Domain[row.domain], row.capacity_bytes, row.native_tile, row.alignment)
        for row in hardware.banks
    )
    # DeviceManifest's current D-R3 metric is symmetric. The report retains exact
    # directed bandwidth/latency links; this law-only projection records a neutral
    # positive crossing distance and never substitutes for the measured link price.
    distance = tuple(
        tuple(0 if i == j else 256 for j in range(len(banks))) for i in range(len(banks))
    )
    manifest = DeviceManifest(hardware.name, banks, distance, hardware.architecture, 0)
    errors = check_device_manifest(manifest)
    if errors:
        raise ValueError("invalid device projection: " + "; ".join(errors))
    return manifest


def _target(hardware: HardwareEnvelope, candidate: PlacementCandidate) -> TargetProfile:
    channels = {hardware.bank(name).channel.lower() for name in candidate.compute_banks}
    arch = hardware.architecture.lower()
    if any("nvidia" in channel or "ptx" in channel for channel in channels):
        return TargetProfile.nvidia_ptx()
    if "aarch64" in arch or "arm64" in arch:
        return TargetProfile.arm64_neon()
    if "riscv" in arch:
        return TargetProfile.riscv_rvv()
    return TargetProfile.x86_avx2()


def _layer_sizes(inventory: TensorInventory, fmt: str) -> tuple[tuple[int, ...], int]:
    sizes, metadata = format_tensor_sizes(inventory, fmt)
    layers = tuple(
        sum(
            sizes[row.name] for row in inventory.tensors if row.layer == layer and row.name in sizes
        )
        for layer in range(inventory.num_hidden_layers)
    )
    globals_ = metadata + sum(
        sizes[row.name] for row in inventory.tensors if row.layer < 0 and row.name in sizes
    )
    return layers, globals_


class _Builder:
    def __init__(
        self,
        hardware: HardwareEnvelope,
        candidate: PlacementCandidate,
        inventory: TensorInventory,
        workload: ModelWorkloadSpec,
        report: ModelCostReport,
    ) -> None:
        self.hardware = hardware
        self.candidate = candidate
        self.inventory = inventory
        self.workload = workload
        self.report = report
        self.manifest = _device_manifest(hardware)
        self.module = Module(
            name=f"model-plan:{candidate.candidate_id}", target=hardware.architecture
        )
        self.actions: list[PlanAction] = []
        self.claim_channels: dict[int, str] = {}
        self.resource_banks: dict[int, str] = {}
        self.next_rid = 1
        self.next_cid = 1
        self.next_phase = 1
        self.previous_phase = 0

    def peak_component(self, bank_name: str, field: str) -> int:
        for peak in self.candidate.peaks:
            if peak.bank == bank_name:
                return getattr(peak, field)
        return 0

    def resource(self, bank_name: str, nbytes: int, name: str) -> int:
        bank = self.hardware.bank(bank_name)
        rid = self.next_rid
        self.next_rid += 1
        self.module.add_resource(
            Resource(
                rid, Domain[bank.domain], 1, (max(1, nbytes),), align=bank.alignment, name=name
            )
        )
        self.resource_banks[rid] = bank_name
        return rid

    def phase(self, claim: Claim, action: PlanAction) -> None:
        pid = self.next_phase
        self.next_phase += 1
        deps = (self.previous_phase,) if self.previous_phase else ()
        self.module.add_phase(Phase(pid, deps, [claim]))
        self.previous_phase = pid
        self.actions.append(replace(action, phase=pid))
        self.claim_channels[claim.id] = action.channel

    def cid(self) -> int:
        value = self.next_cid
        self.next_cid += 1
        return value

    def barrier(self, bank_name: str, layer: int, stage: str, repeats: int) -> None:
        bank = self.hardware.bank(bank_name)
        claim = Claim(
            self.cid(),
            Opcode.BARRIER,
            Lane.H,
            StrideClass.SCALAR,
            hazard="barriered",
            domain=Domain[bank.domain],
            bounds="assumed_safe",
            op=f"model.{stage}.barrier",
            cost_class="latency",
        )
        self.phase(
            claim,
            PlanAction(
                0, "barrier", stage, layer, bank_name, bank_name, 0, 0, repeats, bank.channel
            ),
        )

    def compute(
        self,
        bank_name: str,
        layer: int,
        weight_rid: int,
        activation_rid: int,
        kv_rid: int,
        weight_bytes: int,
        operations: int,
        stage: str,
        repeats: int,
    ) -> None:
        bank = self.hardware.bank(bank_name)
        # The claim is one stage template. ModelExecutionPlan.repeats applies to the
        # complete decode sequence, preserving token -> all-layers autoregressive order.
        count = max(1, operations)
        claim = Claim(
            self.cid(),
            Opcode.T_MACC,
            Lane.T,
            StrideClass.TILE,
            count=count,
            rd=(weight_rid, activation_rid, kv_rid),
            wr=(activation_rid, kv_rid),
            domain=Domain[bank.domain],
            bounds="assumed_safe",
            op=f"model.{stage}.compute.layer.{layer}",
            cost_class="compute",
            quantized_bits=8 if self.candidate.weight_format.startswith("bcirq8") else 0,
        )
        self.phase(
            claim,
            PlanAction(
                0,
                "compute",
                stage,
                layer,
                bank_name,
                bank_name,
                weight_bytes,
                operations,
                repeats,
                bank.channel,
            ),
        )

    def transfer(
        self,
        source: str,
        destination: str,
        source_rid: int,
        destination_rid: int,
        nbytes: int,
        layer: int,
        stage: str,
        repeats: int,
        *,
        allocate: bool = False,
    ) -> None:
        claim = move_claim(
            self.manifest,
            source,
            destination,
            source_rid,
            destination_rid,
            max(1, nbytes),
            self.cid(),
        )
        if allocate:
            claim = replace(claim, lifetime=Lifetime("alloc"))
        channel = self.hardware.bank(destination).channel
        self.phase(
            claim,
            PlanAction(0, "move", stage, layer, source, destination, nbytes, 0, repeats, channel),
        )

    def prefetch(
        self, bank_name: str, rid: int, nbytes: int, layer: int, stage: str, repeats: int
    ) -> None:
        bank = self.hardware.bank(bank_name)
        claim = Claim(
            self.cid(),
            Opcode.LOAD,
            Lane.U,
            StrideClass.UNIT,
            count=max(1, nbytes),
            rd=(rid,),
            domain=Domain[bank.domain],
            bounds="assumed_safe",
            op=f"model.{stage}.prefetch",
            cost_class="bandwidth",
        )
        self.phase(
            claim,
            PlanAction(
                0, "prefetch", stage, layer, bank_name, bank_name, nbytes, 0, repeats, bank.channel
            ),
        )

    def evict(
        self, bank_name: str, rid: int, nbytes: int, layer: int, stage: str, repeats: int
    ) -> None:
        bank = self.hardware.bank(bank_name)
        claim = Claim(
            self.cid(),
            Opcode.NOP,
            Lane.H,
            StrideClass.SCALAR,
            rd=(rid,),
            domain=Domain[bank.domain],
            bounds="assumed_safe",
            op=f"model.{stage}.evict",
            lifetime=Lifetime("free"),
        )
        self.phase(
            claim,
            PlanAction(
                0, "evict", stage, layer, bank_name, bank_name, nbytes, 0, repeats, bank.channel
            ),
        )


def _compute_counts(builder: _Builder) -> tuple[tuple[int, int], tuple[int, int]]:
    inventory = builder.inventory
    workload = builder.workload
    d = inventory.hidden_size
    f = inventory.intermediate_size
    kvd = d * inventory.num_key_value_heads // inventory.num_attention_heads
    projection_macs = 2 * d * d + 2 * d * kvd + 3 * d * f
    prefill_tokens = workload.batch_size * workload.prompt_tokens
    decode_tokens = workload.sessions
    prefill_attention = (
        2 * workload.batch_size * workload.prompt_tokens * workload.prompt_tokens * d
    )
    decode_attention = 2 * workload.sessions * workload.context_tokens * d
    prefill_layer = 2 * (prefill_tokens * projection_macs + prefill_attention)
    decode_layer = 2 * (decode_tokens * projection_macs + decode_attention)
    prefill_head = 2 * prefill_tokens * inventory.vocab_size * d
    decode_head = 2 * decode_tokens * inventory.vocab_size * d
    values = (prefill_layer, decode_layer, prefill_head, decode_head)
    if any(not 1 <= value < (1 << 63) for value in values):
        raise ValueError("model compute block exceeds signed 64-bit plan accounting")
    return (prefill_layer, prefill_head), (decode_layer, decode_head)


def _resident(builder: _Builder, layer_sizes: tuple[int, ...], global_bytes: int) -> None:
    bank_name = builder.candidate.compute_banks[0]
    activation = builder.resource(
        bank_name, builder.peak_component(bank_name, "activation_bytes"), "activation"
    )
    kv = builder.resource(bank_name, builder.peak_component(bank_name, "kv_bytes"), "kv-cache")
    weights = [
        builder.resource(bank_name, nbytes, f"layer-{layer}-weights")
        for layer, nbytes in enumerate(layer_sizes)
    ]
    final_weight = builder.resource(bank_name, global_bytes, "global-weights")
    counts = _compute_counts(builder)
    for stage, (layer_operations, head_operations), repeats in (
        ("prefill", counts[0], 1),
        ("decode", counts[1], builder.workload.max_new_tokens),
    ):
        for layer, (nbytes, weight) in enumerate(zip(layer_sizes, weights)):
            builder.compute(
                bank_name, layer, weight, activation, kv, nbytes, layer_operations, stage, repeats
            )
            builder.barrier(bank_name, layer, stage, repeats)
        builder.compute(
            bank_name,
            -1,
            final_weight,
            activation,
            kv,
            global_bytes,
            head_operations,
            stage,
            repeats,
        )
        builder.barrier(bank_name, -1, stage, repeats)


def _stream(builder: _Builder, layer_sizes: tuple[int, ...], global_bytes: int) -> None:
    compute_name = builder.candidate.compute_banks[0]
    backing_name = builder.candidate.backing_bank
    activation = builder.resource(
        compute_name, builder.peak_component(compute_name, "activation_bytes"), "activation"
    )
    kv = builder.resource(
        compute_name, builder.peak_component(compute_name, "kv_bytes"), "kv-cache"
    )
    # The cost report reserves two full layer buffers. Materialize both resources and
    # alternate them so the verified module agrees with the candidate's memory contract.
    stages = tuple(
        builder.resource(compute_name, max(layer_sizes, default=1), f"layer-stage-{index}")
        for index in range(2)
    )
    backing_weights = [
        builder.resource(backing_name, nbytes, f"layer-{layer}-backing")
        for layer, nbytes in enumerate(layer_sizes)
    ]
    final_weight = builder.resource(compute_name, global_bytes, "global-weights")
    counts = _compute_counts(builder)
    for stage_name, (layer_operations, head_operations), repeats in (
        ("prefill", counts[0], 1),
        ("decode", counts[1], builder.workload.max_new_tokens),
    ):
        for layer, (nbytes, backing) in enumerate(zip(layer_sizes, backing_weights)):
            stage = stages[layer % len(stages)]
            builder.transfer(
                backing_name,
                compute_name,
                backing,
                stage,
                nbytes,
                layer,
                stage_name,
                repeats,
                allocate=True,
            )
            builder.prefetch(compute_name, stage, nbytes, layer, stage_name, repeats)
            builder.barrier(compute_name, layer, stage_name, repeats)
            builder.compute(
                compute_name,
                layer,
                stage,
                activation,
                kv,
                nbytes,
                layer_operations,
                stage_name,
                repeats,
            )
            builder.barrier(compute_name, layer, stage_name, repeats)
            builder.evict(compute_name, stage, nbytes, layer, stage_name, repeats)
        builder.compute(
            compute_name,
            -1,
            final_weight,
            activation,
            kv,
            global_bytes,
            head_operations,
            stage_name,
            repeats,
        )
        builder.barrier(compute_name, -1, stage_name, repeats)


def _split(builder: _Builder, layer_sizes: tuple[int, ...], global_bytes: int) -> None:
    left_name, right_name = builder.candidate.compute_banks
    split = builder.candidate.split_layer
    left_activation = builder.resource(
        left_name, builder.peak_component(left_name, "activation_bytes"), "activation-left"
    )
    right_activation = builder.resource(
        right_name, builder.peak_component(right_name, "activation_bytes"), "activation-right"
    )
    left_kv = builder.resource(left_name, builder.peak_component(left_name, "kv_bytes"), "kv-left")
    right_kv = builder.resource(
        right_name, builder.peak_component(right_name, "kv_bytes"), "kv-right"
    )
    left_weights = [
        builder.resource(left_name, nbytes, f"layer-{layer}-weights")
        for layer, nbytes in enumerate(layer_sizes[:split])
    ]
    right_weights = [
        builder.resource(right_name, nbytes, f"layer-{layer}-weights")
        for layer, nbytes in enumerate(layer_sizes[split:], start=split)
    ]
    dtype_bytes = {"u8": 1, "i8": 1, "f16": 2, "bf16": 2, "f32": 4, "f64": 8}[
        builder.workload.activation_dtype
    ]
    final_weight = builder.resource(right_name, global_bytes, "global-weights")
    counts = _compute_counts(builder)
    for stage, (layer_operations, head_operations), repeats, token_count in (
        ("prefill", counts[0], 1, builder.workload.batch_size * builder.workload.prompt_tokens),
        ("decode", counts[1], builder.workload.max_new_tokens, builder.workload.sessions),
    ):
        for layer, (nbytes, weight) in enumerate(zip(layer_sizes[:split], left_weights)):
            builder.compute(
                left_name,
                layer,
                weight,
                left_activation,
                left_kv,
                nbytes,
                layer_operations,
                stage,
                repeats,
            )
            builder.barrier(left_name, layer, stage, repeats)
        boundary_bytes = max(1, builder.inventory.hidden_size * token_count * dtype_bytes)
        builder.transfer(
            left_name,
            right_name,
            left_activation,
            right_activation,
            boundary_bytes,
            split,
            stage,
            repeats,
            allocate=False,
        )
        builder.barrier(right_name, split, stage, repeats)
        for layer, (nbytes, weight) in enumerate(
            zip(layer_sizes[split:], right_weights), start=split
        ):
            builder.compute(
                right_name,
                layer,
                weight,
                right_activation,
                right_kv,
                nbytes,
                layer_operations,
                stage,
                repeats,
            )
            builder.barrier(right_name, layer, stage, repeats)
        builder.compute(
            right_name,
            -1,
            final_weight,
            right_activation,
            right_kv,
            global_bytes,
            head_operations,
            stage,
            repeats,
        )
        builder.barrier(right_name, -1, stage, repeats)


def lower_model_execution(
    report: ModelCostReport,
    candidate: PlacementCandidate,
    inventory: TensorInventory,
    hardware: HardwareEnvelope,
    workload: ModelWorkloadSpec,
) -> LoweredModelExecution:
    """Create an artifact only after every available deterministic law accepts it."""
    if (
        report.inventory_digest != inventory.header_digest
        or report.hardware_digest != hardware.digest
        or report.workload_digest != workload.digest
    ):
        raise ValueError("report inputs do not match inventory/hardware/workload")
    if candidate not in report.candidates or not candidate.feasible:
        raise ValueError("execution candidate is absent from the report or infeasible")
    if workload.mode != "inference":
        raise ValueError(
            "model execution-plan lowering currently supports inference workloads only"
        )
    builder = _Builder(hardware, candidate, inventory, workload, report)
    layer_sizes, global_bytes = _layer_sizes(inventory, candidate.weight_format)
    if candidate.kind == "resident":
        _resident(builder, layer_sizes, global_bytes)
    elif candidate.kind == "layer-stream":
        _stream(builder, layer_sizes, global_bytes)
    elif candidate.kind == "cpu-gpu-split":
        _split(builder, layer_sizes, global_bytes)
    else:
        raise ValueError(f"candidate kind {candidate.kind!r} has no lowering")

    target = _target(hardware, candidate)
    result = optimize(builder.module, target, Theta.cool())
    pack = hydrate_pipelined(
        builder.module, result, plan=f"model:{candidate.candidate_id}", depth=2
    )
    pack.segments = [
        replace(segment, channel=builder.claim_channels.get(segment.claim_id, "host"))
        for segment in pack.segments
    ]
    diagnostics = verify_all(builder.module, result=result, pack=pack, h=target, theta=Theta.cool())
    diagnostics += verify_smart_lowering(builder.module, pack=pack)
    bank_errors = check_bank_moves(builder.module)
    if diagnostics or bank_errors:
        messages = [f"{row.law}: {row.message}" for row in diagnostics] + bank_errors
        raise ValueError("model execution lowering failed verification: " + "; ".join(messages))
    pack_data = encode(pack)
    action_json = json.dumps(
        [action.__dict__ for action in builder.actions], sort_keys=True, separators=(",", ":")
    )
    module_digest = hashlib.sha256(
        f"{hash_module(builder.module)}:{action_json}".encode()
    ).hexdigest()
    artifact = ModelExecutionPlan(
        report.digest,
        candidate.candidate_id,
        candidate.classification,
        tuple(builder.actions),
        module_digest,
        hashlib.sha256(pack_data).hexdigest(),
        len(pack_data),
    )
    bindings = tuple(
        ResourceBankBinding(rid, builder.resource_banks[rid])
        for rid in sorted(builder.resource_banks)
    )
    return LoweredModelExecution(artifact, builder.module, result, pack, pack_data, bindings)


__all__ = ["LoweredModelExecution", "lower_model_execution"]
