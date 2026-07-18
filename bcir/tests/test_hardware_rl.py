"""Bounded hardware-policy, exact static-address, and promotion-gate regressions."""

import json
import os
import tempfile
from dataclasses import replace

from bcir.frontends.models.assessment import assess_model
from bcir.frontends.models.execution_plan import lower_model_execution
from bcir.frontends.models.inventory import build_tensor_inventory
from bcir.kbcir.hardware_rl import (
    HardwareEpisode,
    HardwareOutcome,
    HardwareRewardPolicy,
    Q20,
    TelemetryToken,
    bounded_mcts,
    certify_hardware_promotion,
    encode_candidate_features,
    encode_hardware_topology,
    hardware_context_digest,
    outcomes_to_preferences,
)
from bcir.kbcir.static_memory import (
    ResourceBankBinding,
    StaticMemoryPlan,
    plan_static_memory,
    verify_static_memory_plan,
)
from bcir.model import Claim, Domain, Module, Opcode, Phase, Resource
from bcir.telemetry import DataDNA
from bcir.tests.test_model_assessment import _hardware, _toy, _workload


def _token(sequence=0):
    event = DataDNA("segment", sequence + 1, cycles=100 + sequence,
                    bytes=64, misses=20, thermal=30, utilization=40,
                    provenance="fixture")
    return TelemetryToken.from_data_dna(
        event, sequence=sequence, register_pressure=25 + sequence,
        bandwidth_pressure=35, throttled=False)


def _outcome(context, candidate, latency, energy, *, provenance="simulated"):
    return HardwareOutcome(context, candidate, latency, energy, 20, 15, 18, 25,
                           False, True, provenance, 5, "1" * 64)


def test_telemetry_availability_and_episode_roundtrip_are_strict():
    token = TelemetryToken.from_data_dna(
        DataDNA("s", 1, cycles=4, bytes=8, misses=1, thermal=2, utilization=3),
        sequence=0)
    assert token.register_pressure == 0
    assert len(token.feature_vector) == 15 and token.voltage_pressure == 0
    try:
        replace(token, register_pressure=1)
        raise AssertionError("an unavailable counter must not masquerade as a zero-based reading")
    except ValueError:
        pass
    try:
        TelemetryToken.from_data_dna(
            DataDNA("s", 1, cycles=4, bytes=8, misses=1, thermal=2, utilization=3),
            sequence=0, throttled=1)
        raise AssertionError("non-Boolean throttle evidence must fail closed")
    except ValueError:
        pass
    report, hardware, workload = ("a" * 64, "b" * 64, "c" * 64)
    context = hardware_context_digest(report, hardware, workload, (token,))
    policy = HardwareRewardPolicy()
    episode = HardwareEpisode(report, hardware, workload, (token,),
                              (_outcome(context, "a", 10, 4),), policy.digest)
    assert HardwareEpisode.from_json(episode.to_json()) == episode
    duplicate = json.loads(episode.to_json())
    duplicate["unknown"] = 1
    try:
        HardwareEpisode.from_json(json.dumps(duplicate))
        raise AssertionError("unknown episode fields must fail closed")
    except ValueError:
        pass
    try:
        hardware_context_digest(report, hardware, workload, (token, token))
        raise AssertionError("duplicate telemetry sequence numbers must fail closed")
    except ValueError:
        pass


def test_reward_preferences_follow_exact_kbcir_metrics_and_reject_bad_outputs():
    context = "d" * 64; policy = HardwareRewardPolicy()
    fast = _outcome(context, "fast", 500_000, 500)
    hot = _outcome(context, "hot", 400_000, 2_000)
    slow = _outcome(context, "slow", 2_000_000, 1_000)
    preferences = outcomes_to_preferences((fast, hot, slow), policy)
    pairs = {(row.chosen_id, row.rejected_id) for row in preferences}
    assert pairs and all(policy.score(next(row for row in (fast, hot, slow)
                                           if row.candidate_id == chosen))
                         < policy.score(next(row for row in (fast, hot, slow)
                                             if row.candidate_id == rejected))
                         for chosen, rejected in pairs)
    try:
        policy.score(replace(fast, correctness_passed=False))
        raise AssertionError("incorrect plans must not receive a reward")
    except ValueError:
        pass
    assert policy.score(replace(fast, cache_miss_pressure=0, bandwidth_pressure=90)) \
        > policy.score(replace(fast, cache_miss_pressure=0, bandwidth_pressure=0))
    try:
        HardwareRewardPolicy(latency_reference_ns=1).score(
            replace(fast, latency_ns=(1 << 63) - 1))
        raise AssertionError("unbounded fixed-point reward costs must fail closed")
    except ValueError:
        pass


def test_bounded_mcts_is_deterministic_and_never_leaves_the_candidate_set():
    priors = {"a": 3, "b": 2, "c": 1}
    values = {"a": 10 * Q20, "b": 50 * Q20, "c": -4 * Q20}
    first = bounded_mcts(values, priors, values.__getitem__, simulations=96)
    second = bounded_mcts(values, priors, values.__getitem__, simulations=96)
    assert first == second and first.candidate_id == "b"
    assert sum(row.visits for row in first.visits) == 96
    try:
        bounded_mcts(("a",), {"a": 1}, values.__getitem__, simulations=4097)
        raise AssertionError("unbounded search must fail")
    except ValueError:
        pass
    try:
        bounded_mcts(([],), {}, values.__getitem__)
        raise AssertionError("malformed candidate identifiers must fail as validation errors")
    except ValueError:
        pass


def _lifetime_module():
    module = Module("static")
    for rid, size in ((1, 64), (2, 128), (3, 32)):
        module.add_resource(Resource(rid, Domain.RAM, 1, (size,), align=64))
    module.add_phase(Phase(0, claims=[Claim(1, Opcode.LOAD, rd=(1,), domain=Domain.RAM)]))
    module.add_phase(Phase(1, deps=(0,), claims=[Claim(2, Opcode.LOAD, rd=(2,), domain=Domain.RAM)]))
    module.add_phase(Phase(2, deps=(1,), claims=[Claim(3, Opcode.ADD, rd=(2, 3),
                                                       wr=(3,), domain=Domain.RAM)]))
    return module


def test_static_memory_reuses_only_dead_ranges_and_detects_alias_tampering():
    module = _lifetime_module(); hardware = _hardware()
    bindings = tuple(ResourceBankBinding(rid, "ram") for rid in (1, 2, 3))
    plan = plan_static_memory(module, bindings, hardware)
    rows = {row.rid: row for row in plan.allocations}
    assert rows[1].offset == rows[2].offset  # resource 1 died before resource 2
    assert sum(row.saved_bytes for row in plan.banks) > 0
    assert rows[2].end <= rows[3].offset or rows[3].end <= rows[2].offset
    assert StaticMemoryPlan.from_json(plan.to_json()) == plan
    tampered = replace(plan, allocations=tuple(
        replace(row, offset=rows[2].offset) if row.rid == 3 else row
        for row in plan.allocations))
    assert any("alias" in error for error in
               verify_static_memory_plan(tampered, module, bindings, hardware))
    unknown = replace(plan, allocations=tuple(
        replace(row, bank="unknown") if row.rid == 1 else row
        for row in plan.allocations))
    assert verify_static_memory_plan(unknown, module, bindings, hardware)


def test_model_candidates_lower_to_streampack_static_addresses_and_measured_promotion():
    with tempfile.TemporaryDirectory() as directory:
        path, config = _toy(directory)
        inventory = build_tensor_inventory([path], config)
        hardware = _hardware(); workload = _workload(formats=("source",))
        report = assess_model(inventory, hardware, workload)
        features = encode_candidate_features(report)
        topology = encode_hardware_topology(hardware)
        assert features and len(topology.nodes) == 2 and topology.links
        reversed_topology = encode_hardware_topology(
            replace(hardware, banks=tuple(reversed(hardware.banks))))
        assert tuple(row.name for row in topology.nodes) == tuple(
            row.name for row in reversed_topology.nodes) == ("ram", "vram")
        try:
            replace(topology, nodes=tuple(reversed(topology.nodes)))
            raise AssertionError("noncanonical topology order must fail closed")
        except ValueError:
            pass
        candidate = next(row for row in report.candidates
                         if row.candidate_id == "source:resident:vram")
        lowered = lower_model_execution(report, candidate, inventory, hardware, workload)
        static = plan_static_memory(lowered.module, lowered.resource_banks, hardware)
        assert not verify_static_memory_plan(static, lowered.module,
                                             lowered.resource_banks, hardware)
        token = _token(); context = hardware_context_digest(
            report.digest, hardware.digest, workload.digest, (token,))
        selected = _outcome(context, candidate.candidate_id, 500_000, 500,
                            provenance="measured")
        baseline = _outcome(context, "source:resident:ram", 2_000_000, 2_000,
                            provenance="measured")
        policy = HardwareRewardPolicy()
        episode = HardwareEpisode(
            report.digest, hardware.digest, workload.digest, (token,),
            (selected, baseline), policy.digest)
        cert = certify_hardware_promotion(
            lowered, static, hardware, selected, baseline, policy,
            episode=episode, generation=2, quiescent=True)
        assert cert.generation == 2 and cert.improvement_q20 > 0 \
            and cert.episode_digest == episode.digest
        assert type(cert).from_json(cert.to_json()) == cert
        wrong_lowered = replace(
            lowered, artifact=replace(lowered.artifact, report_digest="e" * 64))
        try:
            certify_hardware_promotion(
                wrong_lowered, static, hardware, selected, baseline, policy,
                episode=episode, generation=2, quiescent=True)
            raise AssertionError("promotion evidence must bind to the lowered report")
        except ValueError:
            pass
        try:
            certify_hardware_promotion(
                lowered, static, hardware, replace(selected, provenance="simulated"),
                baseline, policy, episode=episode, generation=2, quiescent=True)
            raise AssertionError("simulated evidence must not activate a live plan")
        except ValueError:
            pass
