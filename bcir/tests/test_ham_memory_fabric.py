"""HAM planning, context-shard, and dual-memory regression gates."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
from dataclasses import replace

from bcir.frontends.models.assessment import BankEnvelope, HardwareEnvelope, LinkEnvelope
from bcir.kbcir.context_shard import (
    ContextShardActivation,
    ContextShardCatalog,
    ContextShardEntry,
    ContextShardManifest,
    certify_context_activation,
    read_context_shard_catalog,
    read_context_shard_manifest,
    verify_context_payload,
    write_context_shard_catalog,
    write_context_shard_manifest,
)
from bcir.kbcir.ham import (
    HAMAccess,
    HAMBankPeak,
    HAMExecutionPlan,
    HAMResource,
    HAMWorkload,
    lower_ham_workload,
    verify_ham_plan,
)
from bcir.kbcir.ham_tool import main as ham_main
from bcir.kbcir.optimization_memory import (
    OptimizationFact,
    OptimizationMemory,
    OptimizationPattern,
    query_optimization_memory,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hardware(*, direct=True, vram_bytes=1024) -> HardwareEnvelope:
    banks = (
        BankEnvelope("ssd", "NVM", "storage", 1 << 20, 1 << 20,
                     2_000_000_000, 2_000_000_000),
        BankEnvelope("ram", "RAM", "host", 1 << 20, 1 << 20,
                     20_000_000_000, 20_000_000_000),
        BankEnvelope("vram", "VRAM", "gpu", vram_bytes, vram_bytes,
                     100_000_000_000, 100_000_000_000),
    )
    links = [
        LinkEnvelope("ssd", "ram", 1_000_000_000, 2_000),
        LinkEnvelope("ram", "ssd", 1_000_000_000, 2_000),
        LinkEnvelope("ram", "vram", 5_000_000_000, 500),
        LinkEnvelope("vram", "ram", 5_000_000_000, 500),
    ]
    if direct:
        links.extend((LinkEnvelope("ssd", "vram", 10_000_000_000, 100),
                      LinkEnvelope("vram", "ssd", 10_000_000_000, 100)))
    return HardwareEnvelope("ham-test", "x86_64", banks, tuple(links), ())


def _workload(hardware: HardwareEnvelope, *, mutable=False, prefetch=1) -> HAMWorkload:
    resources = (
        HAMResource(1, "base", "tensor", 400, "ssd", _hash("base"), (), False, 10),
        HAMResource(2, "context", "adapter", 300, "ssd", _hash("context"),
                    (1,), mutable, 20),
    )
    accesses = (HAMAccess(1, 1, "vram", "read", "gem.prefill"),
                HAMAccess(2, 2, "vram", "write" if mutable else "read",
                          "gem.context"),
                HAMAccess(3, 2, "vram", "read", "gem.decode"))
    return HAMWorkload(hardware.digest, resources, accesses, prefetch, 3)


def _raises(fragment, callback):
    try:
        callback()
        raise AssertionError(f"expected failure containing {fragment!r}")
    except (ValueError, LookupError) as exc:
        assert fragment in str(exc), str(exc)


def test_ham_workload_is_strict_canonical_and_dependency_acyclic():
    hardware = _hardware(); workload = _workload(hardware)
    assert HAMWorkload.from_json(workload.to_json()) == workload
    assert HAMWorkload.from_json(workload.to_json()).digest == workload.digest
    _raises("cycle", lambda: HAMWorkload(
        hardware.digest,
        (HAMResource(1, "a", "other", 1, "ssd", _hash("a"), (2,)),
         HAMResource(2, "b", "other", 1, "ssd", _hash("b"), (1,))),
        (HAMAccess(1, 1, "vram", "read", "read"),)))
    _raises("contiguous", lambda: HAMWorkload(
        hardware.digest, workload.resources,
        (HAMAccess(2, 1, "vram", "read", "read"),)))
    duplicate = workload.to_json().replace(
        '"schema":"bcir.ham_workload.v1"',
        '"schema":"bcir.ham_workload.v1","schema":"bcir.ham_workload.v1"')
    _raises("duplicate", lambda: HAMWorkload.from_json(duplicate))
    nested_array = json.loads(workload.to_json())
    nested_array["resources"][0] = list(nested_array["resources"][0].items())
    _raises("entries must be objects", lambda: HAMWorkload.from_json(
        json.dumps(nested_array)))
    _raises("immutable", lambda: lower_ham_workload(HAMWorkload(
        hardware.digest, workload.resources,
        (HAMAccess(1, 1, "vram", "write", "bad-write"),)), hardware))


def test_direct_peer_route_lowers_to_verified_claims_and_deterministic_streampack():
    hardware = _hardware(); workload = _workload(hardware)
    first = lower_ham_workload(workload, hardware)
    second = lower_ham_workload(workload, hardware)
    assert first.artifact == second.artifact
    assert first.streampack_data == second.streampack_data
    assert first.artifact.to_json() == second.artifact.to_json()
    assert not verify_ham_plan(first.artifact, workload, hardware)
    assert first.artifact.logical_transfer_bytes == first.artifact.direct_peer_bytes
    assert first.artifact.staged_transfer_bytes == first.artifact.host_bounce_bytes == 0
    transfers = [row for row in first.artifact.actions if row.kind == "transfer"]
    assert transfers and all(row.route == ("ssd", "vram") for row in transfers)
    assert all(row.route_class == "direct-peer" and not row.host_bounce
               for row in transfers)
    assert first.streampack.provenance_ok()
    assert {row.channel for row in first.streampack.segments} <= {"gpu"}
    assert all(resource.access == "ham" for resource in first.module.resources.values())
    assert HAMExecutionPlan.from_json(first.artifact.to_json()) == first.artifact


def test_staged_route_is_explicit_and_reports_host_bounce_without_claiming_p2p():
    hardware = _hardware(direct=False); workload = _workload(hardware)
    lowered = lower_ham_workload(workload, hardware)
    transfers = [row for row in lowered.artifact.actions if row.kind == "transfer"]
    assert transfers and all(row.route == ("ssd", "ram", "vram") for row in transfers)
    assert all(row.route_class == "staged" and row.host_bounce for row in transfers)
    assert lowered.artifact.direct_peer_bytes == 0
    assert lowered.artifact.staged_transfer_bytes == lowered.artifact.logical_transfer_bytes
    assert lowered.artifact.host_bounce_bytes == lowered.artifact.logical_transfer_bytes
    assert lowered.artifact.wire_transfer_bytes == 2 * lowered.artifact.logical_transfer_bytes

    no_path = HardwareEnvelope(hardware.name, hardware.architecture, hardware.banks,
                               tuple(link for link in hardware.links
                                     if link.destination != "vram"), ())
    broken = replace(workload, hardware_digest=no_path.digest)
    _raises("no current HAM copy", lambda: lower_ham_workload(broken, no_path))

    one_hop_only = replace(workload, max_route_hops=1)
    _raises("no current HAM copy", lambda: lower_ham_workload(one_hop_only, hardware))


def test_host_endpoint_route_is_direct_dma_not_peer_or_bounce():
    hardware = _hardware(direct=False)
    workload = HAMWorkload(
        hardware.digest,
        (HAMResource(1, "host-stage", "tensor", 256, "ssd", _hash("host-stage")),),
        (HAMAccess(1, 1, "ram", "read", "host-stage"),), 0, 1)
    lowered = lower_ham_workload(workload, hardware)
    transfer = next(row for row in lowered.artifact.actions if row.kind == "transfer")
    assert transfer.route == ("ssd", "ram")
    assert transfer.route_class == "direct-dma" and not transfer.host_bounce
    assert lowered.artifact.direct_peer_bytes == lowered.artifact.staged_transfer_bytes == 0
    assert lowered.artifact.host_bounce_bytes == 0


def test_capacity_uses_explicit_farthest_next_use_eviction_and_never_home_eviction():
    hardware = _hardware(vram_bytes=500)
    resources = (
        HAMResource(1, "a", "tensor", 400, "ssd", _hash("a")),
        HAMResource(2, "b", "tensor", 400, "ssd", _hash("b")),
    )
    accesses = (HAMAccess(1, 1, "vram", "read", "a0"),
                HAMAccess(2, 2, "vram", "read", "b0"),
                HAMAccess(3, 1, "vram", "read", "a1"))
    workload = HAMWorkload(hardware.digest, resources, accesses, 2, 3)
    lowered = lower_ham_workload(workload, hardware)
    assert not verify_ham_plan(lowered.artifact, workload, hardware)
    evictions = [row for row in lowered.artifact.actions if row.kind == "evict"]
    assert evictions and all(row.bank == "vram" for row in evictions)
    assert sum(row.kind == "transfer" and row.resource_id == 1
               for row in lowered.artifact.actions) == 2
    assert lowered.artifact.peaks[-1] == HAMBankPeak("vram", 400, 500)


def test_capacity_ties_evict_lower_priority_before_equal_next_use():
    hardware = _hardware(vram_bytes=400)
    resources = (
        HAMResource(1, "important", "tensor", 190, "ssd", _hash("important"),
                    (), False, 10),
        HAMResource(2, "disposable", "tensor", 190, "ssd", _hash("disposable"),
                    (), False, 1),
        HAMResource(3, "temporary", "tensor", 200, "ssd", _hash("temporary")),
        HAMResource(4, "join", "tensor", 1, "ssd", _hash("join"), (1, 2)),
    )
    accesses = (HAMAccess(1, 1, "vram", "read", "important-now"),
                HAMAccess(2, 2, "vram", "read", "disposable-now"),
                HAMAccess(3, 3, "vram", "read", "temporary"),
                HAMAccess(4, 4, "vram", "read", "equal-next-use-join"))
    lowered = lower_ham_workload(
        HAMWorkload(hardware.digest, resources, accesses, 0, 3), hardware)
    capacity_evictions = [row for row in lowered.artifact.actions
                          if row.kind == "evict"
                          and row.reason == "capacity:farthest-next-use"]
    assert capacity_evictions[0].resource_id == 2
    assert not verify_ham_plan(lowered.artifact,
                               HAMWorkload(hardware.digest, resources, accesses, 0, 3),
                               hardware)


def test_mutable_generation_invalidates_stale_copies_and_writes_back_home():
    hardware = _hardware(); workload = _workload(hardware, mutable=True)
    lowered = lower_ham_workload(workload, hardware)
    write = next(row for row in lowered.artifact.actions
                 if row.kind == "access" and row.mode == "write")
    assert (write.generation_before, write.generation_after) == (0, 1)
    assert any(row.kind == "invalidate" and row.bank == "ssd"
               for row in lowered.artifact.actions)
    assert any(row.kind == "transfer" and row.intent == "writeback"
               and row.route == ("vram", "ssd") for row in lowered.artifact.actions)
    assert not verify_ham_plan(lowered.artifact, workload, hardware)

    actions = list(lowered.artifact.actions)
    index = actions.index(write)
    actions[index] = replace(write, version_sha256=_hash("forged"))
    forged = replace(lowered.artifact, actions=tuple(actions))
    assert any("write lineage" in row for row in verify_ham_plan(forged, workload, hardware))


def test_independent_replay_rejects_route_size_and_peak_tampering():
    hardware = _hardware(direct=False); workload = _workload(hardware)
    lowered = lower_ham_workload(workload, hardware)
    actions = list(lowered.artifact.actions)
    index = next(index for index, row in enumerate(actions) if row.kind == "transfer")
    actions[index] = replace(actions[index], route=("ssd", "vram"),
                             route_class="direct-peer", host_bounce=False)
    tampered = replace(lowered.artifact, actions=tuple(actions))
    assert any("undeclared HAM route" in row for row in
               verify_ham_plan(tampered, workload, hardware))

    actions = list(lowered.artifact.actions)
    transfer = actions[index]
    actions[index] = replace(transfer, nbytes=transfer.nbytes - 1)
    tampered = replace(lowered.artifact, actions=tuple(actions))
    assert any("transfer size" in row for row in verify_ham_plan(tampered, workload, hardware))

    peaks = tuple(replace(row, peak_bytes=max(0, row.peak_bytes - 1))
                  if row.bank == "vram" else row for row in lowered.artifact.peaks)
    tampered = replace(lowered.artifact, peaks=peaks)
    assert any("peak inventory" in row for row in verify_ham_plan(tampered, workload, hardware))

    actions = [row for row in lowered.artifact.actions
               if not (row.kind == "barrier"
                       and row.reason == "residency:complete-before-access")]
    actions = [replace(row, order=index + 1) for index, row in enumerate(actions)]
    tampered = replace(lowered.artifact, actions=tuple(actions))
    assert any("lacks its residency barrier" in row
               for row in verify_ham_plan(tampered, workload, hardware))

    actions = list(lowered.artifact.actions)
    index = next(index for index, row in enumerate(actions) if row.kind == "prefetch")
    actions[index] = replace(actions[index], generation_before=1, generation_after=1)
    tampered = replace(lowered.artifact, actions=tuple(actions))
    assert any("prefetch lineage" in row
               for row in verify_ham_plan(tampered, workload, hardware))


def test_ham_cli_is_metadata_only_atomic_and_replays_its_outputs():
    hardware = _hardware(); workload = _workload(hardware)
    with tempfile.TemporaryDirectory() as directory:
        hardware_path = os.path.join(directory, "hardware.json")
        workload_path = os.path.join(directory, "workload.json")
        plan_path = os.path.join(directory, "plan.json")
        pack_path = os.path.join(directory, "plan.pack")
        with open(hardware_path, "w", encoding="utf-8") as stream:
            stream.write(hardware.to_json())
        with open(workload_path, "w", encoding="utf-8") as stream:
            stream.write(workload.to_json())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assert ham_main(["--hardware", hardware_path, "--workload", workload_path,
                             "--plan-out", plan_path, "--pack-out", pack_path]) == 0
        summary = json.loads(output.getvalue())
        plan = HAMExecutionPlan.from_json(open(plan_path, encoding="utf-8").read())
        assert summary["payload_bytes_read"] == 0 and summary["plan_digest"] == plan.digest
        assert os.path.getsize(pack_path) == plan.streampack_bytes
        assert not verify_ham_plan(plan, workload, hardware)

        with open(plan_path, "wb") as stream:
            stream.write(b"previous-valid-generation")
        bad = replace(workload, hardware_digest=_hash("wrong"))
        with open(workload_path, "w", encoding="utf-8") as stream:
            stream.write(bad.to_json())
        with contextlib.redirect_stderr(io.StringIO()):
            assert ham_main(["--hardware", hardware_path, "--workload", workload_path,
                             "--plan-out", plan_path]) == 2
        assert open(plan_path, "rb").read() == b"previous-valid-generation"


def _manifest(payload: bytes, hardware_digest: str) -> ContextShardManifest:
    return ContextShardManifest(
        "bcirq8", "bcirq8-v1", "quantized", hashlib.sha256(payload).hexdigest(),
        len(payload), _hash("base-model"), hardware_digest, _hash("selector"),
        _hash("provenance"), _hash("certificate"), "offline-test@1", 1)


def test_context_shard_is_content_addressed_payload_verified_and_ham_addressable():
    payload = b"small-bcirq8-context-shard"
    manifest = _manifest(payload, _hardware().digest)
    resource = manifest.to_ham_resource(9, "ssd", priority=7)
    assert resource.nbytes == len(payload) and resource.content_sha256 == manifest.payload_sha256
    with tempfile.TemporaryDirectory() as directory:
        manifest_path = os.path.join(directory, "manifest.json")
        payload_path = os.path.join(directory, "payload.bcirq8")
        with open(payload_path, "wb") as stream:
            stream.write(payload)
        write_context_shard_manifest(manifest_path, manifest)
        first = open(manifest_path, "rb").read()
        write_context_shard_manifest(manifest_path, manifest)
        assert open(manifest_path, "rb").read() == first
        assert read_context_shard_manifest(manifest_path) == manifest
        verify_context_payload(manifest, payload_path)
        with open(payload_path, "ab") as stream:
            stream.write(b"x")
        _raises("larger", lambda: verify_context_payload(manifest, payload_path))


def test_context_catalog_activation_is_quiescent_bound_and_rollback_capable():
    hardware = _hardware(); manifest = _manifest(b"artifact", hardware.digest)
    entry = ContextShardEntry(manifest.selector_sha256, manifest.digest, 10)
    catalog = ContextShardCatalog(manifest.base_model_sha256, hardware.digest, 2, (entry,))
    activation = certify_context_activation(
        catalog, manifest, manifest.selector_sha256,
        current_shard_sha256=_hash("old"), activation_generation=3, quiescent=True)
    assert activation.shard_sha256 == manifest.digest
    assert activation.previous_shard_sha256 == _hash("old")
    assert activation.rollback_token_sha256 == certify_context_activation(
        catalog, manifest, manifest.selector_sha256,
        current_shard_sha256=_hash("old"), activation_generation=3,
        quiescent=True).rollback_token_sha256
    assert ContextShardActivation.from_json(activation.to_json()) == activation
    assert len(activation.digest) == 64
    _raises("quiescent", lambda: certify_context_activation(
        catalog, manifest, manifest.selector_sha256,
        activation_generation=3, quiescent=False))
    _raises("advance", lambda: certify_context_activation(
        catalog, manifest, manifest.selector_sha256,
        activation_generation=2, quiescent=True))

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "catalog.json")
        write_context_shard_catalog(path, catalog)
        assert read_context_shard_catalog(path) == catalog
        duplicate = catalog.to_json().replace(
            '"generation":2', '"generation":2,"generation":2')
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(duplicate)
        _raises("duplicate", lambda: read_context_shard_catalog(path))


def test_dual_memory_similarity_can_rank_but_hard_facts_veto_candidates():
    enough_vram = OptimizationFact("vram", "capacity_bytes", "8192")
    p2p = OptimizationFact("ssd->vram", "direct_link", "true")
    shard_near = _hash("near-shard"); shard_far = _hash("far-shard")
    near = OptimizationPattern(_hash("near-pattern"), (10, 10, 10), shard_near,
                               _hash("near-evidence"), (p2p,))
    far = OptimizationPattern(_hash("far-pattern"), (100, 100, 100), shard_far,
                              _hash("far-evidence"), (enough_vram,))
    memory = OptimizationMemory(_hash("model"), _hash("embedder"),
                                tuple(sorted((near, far), key=lambda row: row.pattern_sha256)),
                                tuple(sorted((enough_vram, p2p))))
    assert OptimizationMemory.from_json(memory.to_json()) == memory
    nested_array = json.loads(memory.to_json())
    nested_array["patterns"][0] = list(nested_array["patterns"][0].items())
    _raises("entries must be objects", lambda: OptimizationMemory.from_json(
        json.dumps(nested_array)))
    matches = query_optimization_memory(memory, (11, 11, 11), top_k=2)
    assert matches[0].context_shard_sha256 == shard_near
    # Remove the direct-link capability from the active hard facts: the nearest fuzzy
    # match is vetoed, so the farther but legal pattern wins.
    matches = query_optimization_memory(memory, (11, 11, 11), top_k=2,
                                        active_facts=(enough_vram,))
    assert len(matches) == 1 and matches[0].context_shard_sha256 == shard_far
    _raises("outside hard memory", lambda: query_optimization_memory(
        memory, (11, 11, 11), active_facts=(OptimizationFact("x", "y", "z"),)))


def test_context_and_dual_memory_schema_combinations_fail_closed():
    hardware = _hardware(); manifest = _manifest(b"artifact", hardware.digest)
    _raises("requires tensor_format", lambda: replace(manifest, tensor_format="safetensors"))
    _raises("positive lora_rank", lambda: ContextShardManifest(
        "lora", "safetensors", "approximate", _hash("p"), 1,
        _hash("base"), hardware.digest, _hash("selector"), _hash("provenance"),
        _hash("certificate"), "test@1", 1, 0))
    malformed = manifest.to_json()[:-1]
    _raises("invalid", lambda: ContextShardManifest.from_json(malformed))


def run():
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()


if __name__ == "__main__":
    run()
