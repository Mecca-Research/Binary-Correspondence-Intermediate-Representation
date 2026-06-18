"""Hardware channels: the clean per-architecture separation for a unified heterogeneous tower.

The recent friction (x86 and aarch64 rules colliding) came from per-arch specifics scattered as
inline checks. These tests prove the fix: each HardwareChannel ISOLATES its backend's rules, two
channels coexist without collision, every channel plans through the SAME unified K_BCIR core, and
a tower of mixed backends (CPU + GPU + FPGA + NVMe + HBM) orchestrates to one binary-graph plan.
"""

from bcir import channels as ch
from bcir.channels import (
    CHANNELS,
    EM_AARCH64,
    EM_X86_64,
    HardwareChannel,
    RuntimeChannel,
    channels_of_kind,
    host_channel,
    host_perf_syscall_nr,
    orchestrate,
    register_channel,
)
from bcir.examples import gather_reduce, matmul_tiled, multi_histogram, vector_add
from bcir.kbcir.cost import Theta
from bcir.kbcir.realize import optimize
from dataclasses import replace


# --- the registry + isolation ----------------------------------------------------

def test_the_tower_spans_every_hardware_class():
    kinds = {c.kind for c in CHANNELS.values()}
    assert kinds == {"cpu", "gpu", "fpga", "storage", "memory"}
    # the six real arch backends + the three modeled future ones
    assert {"x86_avx512", "x86_avx2", "arm64_neon", "arm64_sve", "riscv_rvv", "nvidia_ptx"} <= set(CHANNELS)
    assert {"fpga_systolic", "nvme_stream", "hbm_pim"} <= set(CHANNELS)


def test_x86_and_arm_rules_do_not_collide():
    # The exact source of the recent friction: each channel owns its ABI/sensor rules in isolation.
    x = CHANNELS["x86_avx512"]
    a = CHANNELS["arm64_neon"]
    assert x.runtime.perf_syscall_nr == 298 and a.runtime.perf_syscall_nr == 241   # the #255 collision
    assert x.runtime.energy_source == "rapl" and a.runtime.energy_source == "hwmon"
    assert x.e_machine == EM_X86_64 and a.e_machine == EM_AARCH64
    # changing one channel cannot perturb another (frozen, independent records)
    assert x.runtime is not a.runtime


def test_channels_carry_real_llvm_triples_not_labels():
    # the codegen identity is a genuine triple (so -mtriple works), not the old "aarch64-neon" label
    assert CHANNELS["arm64_neon"].llvm_triple == "aarch64-unknown-linux-gnu"
    assert CHANNELS["x86_avx512"].llvm_triple == "x86_64-unknown-linux-gnu"
    assert CHANNELS["riscv_rvv"].llvm_triple == "riscv64-unknown-linux-gnu"


def test_cpu_channels_make_host_elf_objects_offhost_backends_do_not():
    assert CHANNELS["x86_avx512"].is_host_elf and CHANNELS["arm64_neon"].is_host_elf
    assert not CHANNELS["nvidia_ptx"].is_host_elf      # GPU: PTX/cubin, not a host ELF
    assert not CHANNELS["fpga_systolic"].is_host_elf   # FPGA: a bitstream
    assert not CHANNELS["nvme_stream"].is_host_elf


def test_host_channel_and_perf_syscall_are_consistent():
    h = host_channel()
    assert h.kind == "cpu" and h.name in CHANNELS
    assert host_perf_syscall_nr() == h.runtime.perf_syscall_nr


def test_modeled_channels_are_flagged_real_ones_are_not():
    assert CHANNELS["fpga_systolic"].modeled and CHANNELS["hbm_pim"].modeled
    assert not CHANNELS["x86_avx512"].modeled and not CHANNELS["arm64_neon"].modeled


# --- the unified K_BCIR core: every channel plans the same module the same way --------

def test_every_channel_plans_through_the_same_unified_optimizer():
    # The whole point: regardless of hardware origin, a module decomposes to a K_BCIR plan via the
    # one optimizer on the channel's profile -- a positive, deterministic score on every channel.
    m = vector_add(1024)
    for c in CHANNELS.values():
        r = optimize(m, c.profile, Theta.cool())
        assert r.score > 0 and r.steps, c.name
        assert optimize(m, c.profile, Theta.cool()).score == r.score   # deterministic


def test_realizable_widths_are_per_channel_no_unrealizable_carryover():
    # NEON tops out at vec4; the tile lane is realizable per channel (the B2 fix, now isolated).
    neon = optimize(matmul_tiled(), CHANNELS["arm64_neon"].profile, Theta.cool())
    assert max(s.candidate.width for s in neon.steps) <= 4
    avx = optimize(matmul_tiled(), CHANNELS["x86_avx512"].profile, Theta.cool())
    assert max(s.candidate.width for s in avx.steps) == 16


# --- heterogeneous orchestration: one binary graph across the tower -------------------

def _tower(*names):
    return [CHANNELS[n] for n in names]


def test_orchestration_routes_a_module_across_the_tower():
    tower = _tower("x86_avx512", "nvidia_ptx", "fpga_systolic", "hbm_pim", "nvme_stream")
    plan = orchestrate(multi_histogram(), tower, Theta.cool())
    # every claim is placed; the plan decomposes (total == sum of per-claim placements)
    assert plan.placements and plan.total_cost == sum(p.cost for p in plan.placements)
    assert plan.channels_used <= {c.name for c in tower}
    # a real heterogeneous mix (scatters/gathers offload off the CPU; not everything on one backend)
    assert len(plan.channels_used) >= 2


def test_orchestration_is_deterministic():
    tower = _tower("x86_avx512", "nvidia_ptx", "hbm_pim")
    a = orchestrate(gather_reduce(), tower, Theta.cool())
    b = orchestrate(gather_reduce(), tower, Theta.cool())
    assert [(p.claim_id, p.channel, p.cost) for p in a.placements] == \
           [(p.claim_id, p.channel, p.cost) for p in b.placements]


def test_a_cpu_only_tower_keeps_everything_on_the_cpu():
    # With no accelerators in the tower, the unified plan still works -- all on the CPU channel.
    plan = orchestrate(matmul_tiled(), _tower("x86_avx512"), Theta.cool())
    assert plan.channels_used == {"x86_avx512"}
    # and it equals the single-channel K_BCIR score (one binary graph, one backend)
    assert plan.total_cost == optimize(matmul_tiled(), CHANNELS["x86_avx512"].profile, Theta.cool()).score


def test_orchestration_tags_one_streampack_across_the_tower():
    # The unified binary graph: a single GEM StreamPack whose segments carry their channel dispatch,
    # so the executor runs one pack across mixed backends.
    from bcir.gem import hydrate
    from bcir.channels import apply_channels
    m = multi_histogram()
    tower = _tower("x86_avx512", "nvidia_ptx", "hbm_pim")
    plan = orchestrate(m, tower, Theta.cool())
    pack = apply_channels(hydrate(m, optimize(m, CHANNELS["x86_avx512"].profile, Theta.cool())), plan)
    placed = {p.claim_id: p.channel for p in plan.placements}
    # every segment names the channel its claim was placed on -> one graph, dispatched per segment
    for seg in pack.segments:
        assert seg.channel == placed[seg.claim_id]
    assert {seg.channel for seg in pack.segments} == plan.channels_used


def test_adding_a_channel_does_not_touch_the_others():
    # The single extension point: register a new backend; the core + other channels are untouched.
    before = dict(CHANNELS)
    try:
        accel = HardwareChannel("tpu_test", "accelerator", CHANNELS["nvidia_ptx"].profile,
                                "", 0, RuntimeChannel(), modeled=True)
        register_channel(accel)
        assert CHANNELS["tpu_test"].kind == "accelerator"
        # every pre-existing channel is byte-identical (no collision from the new one)
        for name, c in before.items():
            assert CHANNELS[name] is c
    finally:
        CHANNELS.pop("tpu_test", None)
