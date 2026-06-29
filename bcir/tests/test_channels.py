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
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify
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


def _offload_chain(n_in: int, n_out: int = 16) -> Module:
    """A producer->consumer chain that makes cross-device offload a real cost decision: a plain UNIT
    `vector.add` over `n_in` elements writes MID (the producer -- the host CPU is its only suitable
    channel, since the PIM/storage backends do not take elementwise UNIT work), then a RANDOM-access
    `histogram.scatter` over `n_out` elements reads MID and writes OUT (the consumer -- much cheaper on
    the gather-friendly HBM/PIM channel). The consumer offloads to the PIM channel only when MID is
    small enough that the host->device transfer (FABRIC ∝ bytes(MID) + a SYNC barrier) is cheaper than
    the compute it saves; a large MID keeps it on the host. The consumer sits in a second phase (the
    GGG tail's R5 hazard rule forbids it sharing the producer's phase)."""
    m = Module(name=f"offload_{n_in}_{n_out}")
    m.add_resource(Resource(rid=1, domain=Domain.RAM, shape=(n_in,), name="A"))
    m.add_resource(Resource(rid=2, domain=Domain.RAM, shape=(n_in,), name="B"))
    m.add_resource(Resource(rid=3, domain=Domain.RAM, shape=(n_in,), name="MID"))
    m.add_resource(Resource(rid=4, domain=Domain.RAM, shape=(n_out,), name="OUT"))
    prod = Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=n_in,
                 rd=(1, 2), wr=(3,), op="vector.add", domain=Domain.RAM)
    cons = Claim(id=2, opcode=Opcode.GGG_LOAD, lane=Lane.GGG, stride_class=StrideClass.RANDOM,
                 count=n_out, rd=(3,), wr=(4,), op="histogram.scatter", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[prod]))
    m.add_phase(Phase(phase_id=1, deps=(0,), claims=[cons]))
    return m


def test_orchestration_routes_a_module_across_the_tower():
    # A producer->consumer chain with a SMALL intermediate: the host produces MID, the gather consumer
    # offloads to the gather-friendly PIM channel (its compute savings beat the small MID transfer) --
    # a real heterogeneous mix governed by the cross-device cost, not per-claim isolation.
    tower = _tower("x86_avx512", "hbm_pim")
    m = _offload_chain(n_in=1024)
    assert not verify(m), verify(m)
    plan = orchestrate(m, tower, Theta.cool())
    # every claim is placed; the plan decomposes -- the BURDENED total is per-claim compute + transfer.
    assert plan.placements
    assert plan.total_cost == sum(p.cost + p.transfer_cost for p in plan.placements)
    assert plan.total_cost == plan.compute_cost + plan.transfer_total
    assert plan.channels_used <= {c.name for c in tower}
    # a real heterogeneous mix: the producer stays on the CPU, the gather offloads to the PIM channel.
    placed = {p.claim_id: p.channel for p in plan.placements}
    assert placed[1] == "x86_avx512" and placed[2] == "hbm_pim"
    assert len(plan.channels_used) >= 2
    assert plan.transfer_total > 0          # the cross-device edge paid a real (fabric+sync) transfer


def test_orchestration_is_deterministic():
    tower = _tower("x86_avx512", "nvidia_ptx", "hbm_pim")
    a = orchestrate(gather_reduce(), tower, Theta.cool())
    b = orchestrate(gather_reduce(), tower, Theta.cool())
    # placement, compute cost AND transfer cost are all identical across runs (the greedy forward pass
    # is order-deterministic with cost-then-name tie-breaks).
    assert [(p.claim_id, p.channel, p.cost, p.transfer_cost) for p in a.placements] == \
           [(p.claim_id, p.channel, p.cost, p.transfer_cost) for p in b.placements]


def test_a_cpu_only_tower_keeps_everything_on_the_cpu():
    # With no accelerators in the tower, the unified plan still works -- all on the CPU channel.
    plan = orchestrate(matmul_tiled(), _tower("x86_avx512"), Theta.cool())
    assert plan.channels_used == {"x86_avx512"}
    # A single-channel tower has NO cross-channel edges -> zero transfer, so the burdened total still
    # equals the oracle's single-channel score (the invariant the transfer model must preserve).
    assert plan.transfer_total == 0
    assert plan.total_cost == optimize(matmul_tiled(), CHANNELS["x86_avx512"].profile, Theta.cool()).score


def test_orchestration_tags_one_streampack_across_the_tower():
    # The unified binary graph: a single GEM StreamPack whose segments carry their channel dispatch,
    # so the executor runs one pack across mixed backends. Uses the offload chain so the placement is a
    # genuine CPU+PIM mix under the cross-device cost model.
    from bcir.gem import hydrate
    from bcir.channels import apply_channels
    m = _offload_chain(n_in=1024)
    tower = _tower("x86_avx512", "hbm_pim")
    plan = orchestrate(m, tower, Theta.cool())
    assert len(plan.channels_used) >= 2          # a real cross-device pack to dispatch per segment
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


# --- cross-device placement cost: offload is a real cost-governed decision (the S1 feature) -----------

def test_offload_wins_when_the_transferred_operand_is_small():
    # SMALL intermediate: the gather consumer's compute is far cheaper on the PIM channel, and moving
    # the small MID operand there costs little -> the consumer OFFLOADS (compute savings beat transfer).
    tower = _tower("x86_avx512", "hbm_pim")
    plan = orchestrate(_offload_chain(n_in=1024), tower, Theta.cool())
    placed = {p.claim_id: p.channel for p in plan.placements}
    assert placed[1] == "x86_avx512"          # the producer stays on the host (PIM does not suit it)
    assert placed[2] == "hbm_pim"             # the gather consumer offloads to the PIM channel
    cons = next(p for p in plan.placements if p.claim_id == 2)
    assert cons.transfer_cost > 0             # it paid a real cross-device transfer to get there


def test_offload_loses_when_the_transferred_operand_is_large():
    # SAME chain, LARGE intermediate: the consumer's per-channel compute is unchanged, but moving the
    # big MID operand now costs MORE than the compute it would save -> the consumer STAYS on the host.
    tower = _tower("x86_avx512", "hbm_pim")
    plan = orchestrate(_offload_chain(n_in=16384), tower, Theta.cool())
    placed = {p.claim_id: p.channel for p in plan.placements}
    assert placed[1] == "x86_avx512"
    assert placed[2] == "x86_avx512"          # offload no longer worth it -> consumer stays home
    assert plan.transfer_total == 0           # same-channel edge -> no transfer charged


def test_the_transfer_cost_alone_flips_the_placement():
    # The decision flips on the TRANSFER term alone: the consumer's compute cost on each channel is
    # IDENTICAL across the two modules (same op, same n_out), so only the bytes(MID) transfer differs.
    tower = _tower("x86_avx512", "hbm_pim")
    small = orchestrate(_offload_chain(n_in=1024), tower, Theta.cool())
    large = orchestrate(_offload_chain(n_in=16384), tower, Theta.cool())
    cons_small = next(p for p in small.placements if p.claim_id == 2)
    cons_large = next(p for p in large.placements if p.claim_id == 2)
    # the consumer's CHANNEL flips, and the realization arithmetic itself didn't change the decision --
    # the only difference between the two runs is the transferred-operand size.
    assert cons_small.channel == "hbm_pim" and cons_large.channel == "x86_avx512"


def test_same_channel_edges_charge_zero_transfer():
    # A single-channel tower: every producer->consumer edge stays on one backend, so NO transfer is
    # charged on any placement (the invariant that keeps a single-channel plan at the oracle score).
    plan = orchestrate(_offload_chain(n_in=1024), _tower("x86_avx512"), Theta.cool())
    assert plan.channels_used == {"x86_avx512"}
    assert all(p.transfer_cost == 0 for p in plan.placements)
    assert plan.transfer_total == 0


def test_transfer_total_equals_sum_of_placements_and_is_positive_cross_device():
    # The plan-level transfer total is exactly the sum of the per-placement transfer costs, and it is
    # strictly positive in the cross-device case (the offload chain pays one fabric+sync edge).
    plan = orchestrate(_offload_chain(n_in=1024), _tower("x86_avx512", "hbm_pim"), Theta.cool())
    assert plan.transfer_total == sum(p.transfer_cost for p in plan.placements)
    assert plan.transfer_total > 0
    assert plan.total_cost == plan.compute_cost + plan.transfer_total


def test_cross_device_placement_is_deterministic():
    # The greedy forward pass with cost-then-name tie-breaks is fully deterministic: identical
    # placements, compute costs and transfer costs across repeated runs.
    tower = _tower("x86_avx512", "hbm_pim", "nvidia_ptx")
    m = _offload_chain(n_in=1024)
    a = orchestrate(m, tower, Theta.cool())
    b = orchestrate(m, tower, Theta.cool())
    assert [(p.claim_id, p.channel, p.cost, p.transfer_cost) for p in a.placements] == \
           [(p.claim_id, p.channel, p.cost, p.transfer_cost) for p in b.placements]
