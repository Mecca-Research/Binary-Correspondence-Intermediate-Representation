"""Hardware channels -- the clean per-architecture separation for a unified heterogeneous tower.

BCIR is designed to orchestrate *any* mix of hardware (x86, aarch64, RISC-V, a GPU, an FPGA, an
NVMe near-storage engine, an HBM/PIM module) in a single system, yet decompose every backend to
the *same* K_BCIR plan and the *same* GEM binary-graph (StreamPack) execution. The friction we
kept hitting -- x86 and aarch64 rules getting in each other's way -- came from per-architecture
specifics (the ``perf_event_open`` syscall number, the energy/thermal sensor paths, the codegen
triple, the realizable lane widths) being *scattered* as inline ``platform.machine()`` checks.

A :class:`HardwareChannel` fixes that by isolating EVERYTHING hardware-specific about one backend
in one place, behind a uniform interface:

  * ``profile``      -- the K_BCIR cost model H (lane widths, ISA, penalties): the UNIFIED input
                        the optimizer reasons over (the only part the planner sees).
  * ``llvm_triple`` / ``e_machine`` -- the real codegen identity (a genuine LLVM triple + ELF
                        machine), so the AOT/native-object path targets the right backend.
  * ``runtime``      -- the host-runtime hooks (the perf syscall ABI number, how Joules are read,
                        which thermal zones name the CPU) -- the bits that collided when inline.
  * ``kind``         -- the hardware class (cpu / gpu / fpga / accelerator / storage / memory), so
                        the heterogeneous orchestrator can route a reduction to a PIM/GPU/FPGA
                        channel, a big resident scan to a storage channel, control to a CPU channel.

Two channels coexist in one tower without their rules colliding. The K_BCIR optimizer and the GEM
executor consume only the unified parts, so **every channel decomposes to the same plan + the same
StreamPack regardless of hardware origin** -- one binary-graph execution across the whole tower.

Adding a backend is adding a channel: the optimizer, the executor, and the other channels are
untouched. The GPU/FPGA/NVMe/HBM channels below carry *modeled* cost profiles (no resident driver
yet) so the architecture -- and the orchestrator -- already make room for them.
"""

from __future__ import annotations

import platform
import threading
from dataclasses import dataclass, field

from .kbcir.cost import TARGETS, TargetProfile, _host_cpu_features

# ELF e_machine ids (for the native-object channels; 0 == not a host-ELF backend, e.g. a GPU/FPGA).
EM_X86_64, EM_AARCH64, EM_RISCV, EM_NONE = 62, 183, 243, 0

# perf_event_open syscall numbers per ABI (NOT portable -- the source of the x86/ARM collision).
_PERF_SYSCALL = {"x86_64": 298, "amd64": 298, "aarch64": 241, "arm64": 241,
                 "riscv64": 241, "armv7l": 364, "ppc64le": 319, "s390x": 331}


@dataclass(frozen=True)
class RuntimeChannel:
    """The host-runtime specifics of one architecture, isolated so x86's rules never leak into
    ARM's. ``perf_syscall_nr`` is the ``perf_event_open`` ABI number (it differs per machine);
    ``energy_source`` is how real Joules are read (``rapl`` is Intel-only; ``hwmon`` is a shunt
    monitor; ``none`` is honest -- many parts expose no Joules); ``thermal_zone_types`` are the
    ``/sys/class/thermal`` zone types that name this backend's temperature."""
    perf_syscall_nr: int = 298
    energy_source: str = "none"                      # "rapl" | "hwmon" | "none"
    thermal_zone_types: tuple[str, ...] = ("cpu", "soc")


@dataclass(frozen=True)
class HardwareChannel:
    """One isolated execution channel: all of a backend's hardware-specific rules in one place.

    ``profile`` is the unified K_BCIR cost model the optimizer reasons over; ``kind`` classifies
    the hardware so the heterogeneous orchestrator can place suitable work on it; ``llvm_triple``
    + ``e_machine`` are the real codegen identity; ``runtime`` carries the host-side hooks. Every
    channel plans + executes through the same K_BCIR/GEM core."""
    name: str
    kind: str                                        # cpu | gpu | fpga | accelerator | storage | memory
    profile: TargetProfile
    llvm_triple: str
    e_machine: int = EM_NONE
    runtime: RuntimeChannel = field(default_factory=RuntimeChannel)
    arch_match: tuple[str, ...] = ()                 # platform.machine() values this channel serves
    modeled: bool = False                            # True == cost profile is modeled (no resident driver yet)
    capabilities: frozenset[str] = frozenset()       # declared StreamPack-execution capability (plugin
    #                                                  routing contract); empty == use kind-default routing

    @property
    def widest_lane(self) -> int:
        return self.profile.vector_width

    @property
    def is_host_elf(self) -> bool:
        """Produces a native host ELF object (a CPU channel), vs an off-host backend (GPU/FPGA)."""
        return self.e_machine != EM_NONE


# --- the StreamPack-execution capability vocabulary (the plugin routing contract) ---------------
# A channel plugin declares *what GEM work it executes well* as a set of these tags (in its
# channel.json), instead of the core hard-coding a per-kind rule. A claim is mapped to the tags it
# could use; a channel suits the claim if it declares an overlapping tag (or "universal").
CAPABILITY_VOCAB = frozenset({
    "universal",        # runs anything (the CPU fallback; a fully programmable backend)
    "data_parallel",    # elementwise / SIMD lanes
    "reduce",           # reductions / accumulation trees
    "gather",           # random / indexed access (scatter/gather)
    "tile",             # blocked tiles (systolic)
    "matmul",           # dense matmul
    "stream_unit",      # unit-stride sequential streaming
    "scalar_stream",    # scalar sequential streaming
})


def claim_required_caps(claim) -> frozenset:
    """The capability tags a claim could be served by (cumulative — op-derived + stride-derived).
    A channel suits the claim if its declared capabilities overlap this set."""
    from .model import StrideClass

    op = (claim.op or "")
    sc = getattr(claim, "stride_class", None)
    req: set = set()
    if op.startswith("reduce."):
        req.add("reduce")
    if "gather" in op or sc == StrideClass.RANDOM:
        req.add("gather")
    if "matmul" in op:
        req.add("matmul")
    if sc == StrideClass.TILE:
        req.add("tile")
    if sc == StrideClass.UNIT:
        req.add("stream_unit")
    if sc == StrideClass.SCALAR:
        req.add("scalar_stream")
    if not req:
        req.add("data_parallel")                     # default elementwise work
    return frozenset(req)


def channel_suits(claim, ch: HardwareChannel) -> bool:
    """Does channel ``ch`` suit ``claim``? If the channel *declares* capabilities (a plugin), route
    by the declared tags; otherwise fall back to the built-in per-kind rule (legacy, unchanged)."""
    if ch.capabilities:
        return "universal" in ch.capabilities or bool(ch.capabilities & claim_required_caps(claim))
    return _claim_suits_channel(claim, ch)


def route_claim(claim, channels):
    """The plan-time, *cost-free* backend pick for a claim: among the channels that suit it, prefer
    the most specialized match -- a plugin that declares one of the claim's concrete required
    capabilities, over a universal/legacy fallback -- tie-broken by channel name. Returns the chosen
    channel, or ``None`` if nothing suits.

    This is the decision a driver makes *before* (or without) measured costs; `orchestrate` refines
    it with the K_BCIR cost model. It is the seam ported to C (`runtime/c/bcir_channel.c`), so both
    rails route an identical (claim -> channel) map for a given channel set."""
    req = claim_required_caps(claim)
    eligible = [c for c in channels if channel_suits(claim, c)]
    if not eligible:
        return None

    def key(c):
        specialized = bool(c.capabilities & req) and "universal" not in c.capabilities
        return (0 if specialized else 1, c.name)

    return min(eligible, key=key)


def _rt(machine: str, energy: str, zones: tuple[str, ...]) -> RuntimeChannel:
    return RuntimeChannel(perf_syscall_nr=_PERF_SYSCALL.get(machine, 298),
                          energy_source=energy, thermal_zone_types=zones)


# --- the channel registry: the real arch backends + the modeled future ones ----------------------
# Each wraps an existing K_BCIR TargetProfile (unchanged -- the pinned cost model) and adds the
# previously-scattered runtime + codegen identity. CPU channels carry a real ELF machine; the
# GPU/FPGA/NVMe/HBM channels are modeled extension points (kind != cpu) the orchestrator can target.
CHANNELS: dict[str, HardwareChannel] = {
    "x86_avx512": HardwareChannel(
        "x86_avx512", "cpu", TARGETS["x86_avx512"], "x86_64-unknown-linux-gnu", EM_X86_64,
        _rt("x86_64", "rapl", ("x86_pkg_temp", "coretemp")), arch_match=("x86_64", "amd64")),
    "x86_avx2": HardwareChannel(
        "x86_avx2", "cpu", TARGETS["x86_avx2"], "x86_64-unknown-linux-gnu", EM_X86_64,
        _rt("x86_64", "rapl", ("x86_pkg_temp", "coretemp")), arch_match=("x86_64", "amd64")),
    "arm64_neon": HardwareChannel(
        "arm64_neon", "cpu", TARGETS["arm64_neon"], "aarch64-unknown-linux-gnu", EM_AARCH64,
        _rt("aarch64", "hwmon", ("cpu-thermal", "cpu", "soc")), arch_match=("aarch64", "arm64")),
    "arm64_sve": HardwareChannel(
        "arm64_sve", "cpu", TARGETS["arm64_sve"], "aarch64-unknown-linux-gnu", EM_AARCH64,
        _rt("aarch64", "hwmon", ("cpu-thermal", "cpu", "soc")), arch_match=("aarch64", "arm64")),
    "riscv_rvv": HardwareChannel(
        "riscv_rvv", "cpu", TARGETS["riscv_rvv"], "riscv64-unknown-linux-gnu", EM_RISCV,
        _rt("riscv64", "hwmon", ("cpu-thermal", "cpu", "soc")), arch_match=("riscv64",)),
    # GPU: a warp machine, off-host (PTX/cubin, no host ELF). Already a first-class K_BCIR target.
    "nvidia_ptx": HardwareChannel(
        "nvidia_ptx", "gpu", TARGETS["nvidia_ptx"], "nvptx64-nvidia-cuda", EM_NONE,
        RuntimeChannel(energy_source="nvml"), modeled=False),
}
_CHANNELS_LOCK = threading.RLock()


# --- modeled future backends (no resident driver yet; the architecture + orchestrator make room) -
# Constructed here (NOT added to the six-target K_BCIR registry) so the heterogeneous tower can
# include them today. Their cost profiles are deliberately shaped to their strength so a tower
# orchestration is sensible: a systolic FPGA streams wide tiles cheaply; near-storage NVMe runs
# huge sequential scans where the data already lives; an HBM/PIM module runs large reductions and
# gathers in-memory (no transfer). Calibrating these is each channel's own future task.
def _modeled(name, *, lane_widths, gather_penalty, mem_unit, base_overhead, affinity_domains,
             mem_channels, features) -> TargetProfile:
    return TargetProfile(name=name, triple=name, lane_widths=lane_widths,
                         gather_penalty=gather_penalty, mem_unit=mem_unit,
                         base_overhead=base_overhead, isa_features=frozenset(features),
                         affinity_domains=affinity_domains, mem_channels=mem_channels)


def _install_modeled_channels() -> None:
    CHANNELS["fpga_systolic"] = HardwareChannel(
        "fpga_systolic", "fpga",
        _modeled("fpga-systolic", lane_widths=(1, 64), gather_penalty=256, mem_unit=1,
                 base_overhead=32, affinity_domains=64, mem_channels=16, features=("systolic",)),
        "", EM_NONE, RuntimeChannel(energy_source="hwmon"), modeled=True)
    CHANNELS["nvme_stream"] = HardwareChannel(
        "nvme_stream", "storage",
        _modeled("nvme-stream", lane_widths=(1, 8), gather_penalty=1024, mem_unit=1,
                 base_overhead=2, affinity_domains=8, mem_channels=4, features=("near_storage",)),
        "", EM_NONE, RuntimeChannel(energy_source="none"), modeled=True)
    CHANNELS["hbm_pim"] = HardwareChannel(
        "hbm_pim", "memory",
        _modeled("hbm-pim", lane_widths=(1, 16), gather_penalty=8, mem_unit=1,
                 base_overhead=4, affinity_domains=128, mem_channels=32, features=("pim",)),
        "", EM_NONE, RuntimeChannel(energy_source="none"), modeled=True)


def register_channel(channel: HardwareChannel) -> None:
    """Add a backend to the tower. The optimizer, executor, and other channels are untouched --
    this is the single extension point for a new architecture (an FPGA bitstream, an NVMe engine)."""
    if not isinstance(channel, HardwareChannel) or not channel.name:
        raise ValueError("channel registration requires a named HardwareChannel")
    with _CHANNELS_LOCK:
        if channel.name in CHANNELS:
            raise ValueError(f"duplicate channel name {channel.name!r}")
        CHANNELS[channel.name] = channel


_install_modeled_channels()


def channel(name: str) -> HardwareChannel:
    with _CHANNELS_LOCK:
        return CHANNELS[name]


def channels_of_kind(kind: str) -> list[HardwareChannel]:
    """Every channel of a hardware class (e.g. all ``gpu`` channels in the tower)."""
    with _CHANNELS_LOCK:
        return [c for c in CHANNELS.values() if c.kind == kind]


def host_channel() -> HardwareChannel:
    """The channel that serves the host machine -- the safe default for a single-arch run. Reads
    ``platform.machine()`` + ``/proc/cpuinfo`` features (richest matching CPU channel for the arch);
    falls back to the AVX2 baseline. This replaces every scattered inline arch check."""
    mach = platform.machine().lower()
    feat = _host_cpu_features()
    with _CHANNELS_LOCK:
        if mach in ("aarch64", "arm64"):
            return CHANNELS["arm64_sve"] if "sve" in feat else CHANNELS["arm64_neon"]
        if mach in ("x86_64", "amd64"):
            return CHANNELS["x86_avx512"] if "avx512f" in feat else CHANNELS["x86_avx2"]
        if mach.startswith("riscv"):
            return CHANNELS["riscv_rvv"]
        return CHANNELS["x86_avx2"]


def host_perf_syscall_nr() -> int:
    """The host's ``perf_event_open`` syscall number (the silicon layer's single source of truth --
    no inline arch dispatch). 298 on x86_64, 241 on aarch64/riscv64, 364 on armv7."""
    return _PERF_SYSCALL.get(platform.machine(), 298)


# --- heterogeneous orchestration: one plan, one binary graph, across the whole tower ------------
from dataclasses import dataclass as _dc  # noqa: E402 (kept local to the orchestration section)


@_dc(frozen=True)
class ChannelPlacement:
    """One claim's assignment in a heterogeneous tower: the channel chosen, the K_BCIR realization
    on it, and that realization's cost. The same K_BCIR arithmetic on the channel's profile.

    ``cost`` is the claim's COMPUTE cost (the channel's K_BCIR realization score, scalarized under the
    plan policy). ``transfer_cost`` is the CROSS-DEVICE TRANSFER charged to LAND the claim's inputs on
    this channel when a producer wrote them on a *different* channel (FABRIC ∝ bytes moved + a SYNC
    barrier per cross-device edge, scalarized under the same policy weights). It is 0 for a same-channel
    edge or a live-in (see `_transfer_cost` / orchestrate). The placement's full burdened cost is
    ``cost + transfer_cost``."""
    claim_id: int
    op: str
    channel: str
    kind: str
    realization: str
    width: int
    cost: int
    transfer_cost: int = 0


@_dc(frozen=True)
class HeterogeneousPlan:
    """A module planned across a tower: per-claim channel placements that decompose to a single
    GEM StreamPack (each segment carrying its channel dispatch). One binary graph, many backends.

    ``total_cost`` is the burdened total: the sum of per-claim compute costs PLUS the cross-device
    transfer the placement pays to move inputs between backends (``transfer_total``). A single-channel
    tower has no cross-channel edges, so ``transfer_total == 0`` and the burdened total equals the
    oracle's single-channel score (the invariant the cpu-only test pins)."""
    placements: tuple

    @property
    def compute_cost(self) -> int:
        """The sum of per-claim K_BCIR realization (compute) costs, ignoring cross-device transfer."""
        return sum(p.cost for p in self.placements)

    @property
    def transfer_total(self) -> int:
        """The total cross-device transfer the plan pays (FABRIC bytes + SYNC barriers); 0 when every
        producer->consumer edge stays on one channel (single-channel tower / no offload)."""
        return sum(p.transfer_cost for p in self.placements)

    @property
    def total_cost(self) -> int:
        """The burdened total: per-claim compute + cross-device transfer."""
        return sum(p.cost + p.transfer_cost for p in self.placements)

    @property
    def channels_used(self) -> set:
        return {p.channel for p in self.placements}

    @property
    def by_kind(self) -> dict:
        out: dict = {}
        for p in self.placements:
            out.setdefault(p.kind, []).append(p.claim_id)
        return out


def _claim_suits_channel(claim, ch: HardwareChannel) -> bool:
    """Which hardware *classes* can run a claim well (the routing policy a heterogeneous tower uses,
    refined per-channel as real drivers land). A CPU channel is the universal fallback; a memory/PIM
    channel takes reductions + random gathers (resident, no transfer); a GPU takes data-parallel
    work; a systolic FPGA takes tiles/unit-stride streaming; near-storage takes big sequential
    streams. Cost then picks among the suitable channels."""
    from .model import StrideClass

    op = (claim.op or "")
    sc = getattr(claim, "stride_class", None)
    if ch.kind == "cpu":
        return True
    if ch.kind == "gpu":
        return True                                  # warp machine: most data-parallel work
    if ch.kind == "memory":                          # PIM: reductions + gathers over resident data
        return op.startswith("reduce.") or "gather" in op or sc == StrideClass.RANDOM
    if ch.kind == "fpga":                            # systolic: tiles / unit-stride streaming / matmul
        return sc in (StrideClass.TILE, StrideClass.UNIT) or "matmul" in op
    if ch.kind == "storage":                         # near-storage: big sequential streams (ETL/scan)
        return sc in (StrideClass.UNIT, StrideClass.SCALAR)
    return False


# --- cross-device transfer cost (the fabric/sync dimensions wired into placement) ----------------
# A claim placed on a channel different from the one that produced its inputs pays to MOVE that data
# host<->device. This is the first real producer of the FABRIC cost dim (SYNC was only the BARRIER
# at realize.py): a cross-channel producer->consumer edge costs
#
#     FABRIC = (bytes(R) * XFER_BW_Q8) >> 8        # bytes ∝ traffic, Q8-scaled to the memory scale
#     SYNC   = XFER_SYNC                           # one cross-device barrier per dependency
#
# where bytes(R) = R.count * elem_bytes. XFER_BW_Q8 = 64 (x0.25) makes the FABRIC term for moving a
# resource equal to ONE memory-stream-equivalent of that resource on the cost model's scale: realize's
# `mem_shared` charges (count * mem_unit * 256) >> 8 == count per stream, and bytes/4 == count for the
# default 4-byte element, so a cross-device operand transfer is the same order as one memory pass over
# it -- not a wildly different scale. XFER_SYNC = 16 mirrors the BARRIER `sync=16` at realize.py (one
# barrier per cross-device dependency). Same-channel edges and live-ins (resources no claim in the
# module writes) cost ZERO: a same-channel edge needs no move, and a live-in is assumed pre-resident
# on whatever channel first consumes it (a v1 scope limit -- placement does not yet model staging a
# live-in to a device). This keeps a single-channel tower at exactly the oracle's single-channel score.
XFER_BW_Q8 = 64        # Q8 fabric-bandwidth factor: bytes -> fabric units (x0.25 == 1 mem-stream/elem)
XFER_SYNC = 16         # cross-device barrier constant (matches the realize.py BARRIER sync=16)


def _transfer_cost(module, claim, consumer_channel, producer_of, weights_vec) -> int:
    """The cross-device transfer charged to land ``claim``'s inputs on ``consumer_channel``, scalarized
    under ``weights_vec`` (the same policy weights orchestrate scores compute costs with, so the term is
    in the SAME units as ``step.cost`` -- a scalarized int). For each read RID, if a prior claim wrote
    it on a *different* channel, charge FABRIC ∝ bytes(R) + one SYNC barrier; same-channel producers and
    live-ins (no producer in the module) cost zero. Summed over the claim's reads."""
    from .kbcir.cost import CostVector

    xfer = CostVector.zero()
    consumer_elem_bytes = consumer_channel.profile.elem_bytes
    for rid in claim.rd:
        src = producer_of.get(rid)
        if src is None or src == consumer_channel.name:
            continue                                 # live-in (assume resident) or same-channel: free
        res = module.resource(rid)
        count = res.count if res is not None else 1
        nbytes = count * consumer_elem_bytes
        fabric = (nbytes * XFER_BW_Q8) >> 8
        xfer = xfer + CostVector.of(fabric=fabric, sync=XFER_SYNC)
    return xfer.dot(weights_vec)


def orchestrate(module, channels, theta, policy=None) -> HeterogeneousPlan:
    """Plan a module across a TOWER of heterogeneous channels (x86 + ARM + GPU + FPGA + NVMe + HBM).

    A GREEDY FORWARD PASS in topological claim order. For each claim, among the channels whose *kind*
    suits it, pick the one whose K_BCIR compute cost PLUS the cross-device transfer to land its inputs
    is lowest -- so a large reduction may land on the HBM/PIM module, a tiled matmul on the FPGA, a
    gather on the GPU, and control on the CPU, but only when the compute savings beat the host<->device
    transfer. Every placement is the same K_BCIR realization arithmetic on that channel's profile, so
    the result decomposes to a single GEM StreamPack whose segments carry their channel dispatch: one
    binary graph, executed across the whole tower. ``channels`` is the tower (a list of HardwareChannel).

    The transfer term (FABRIC ∝ bytes moved + a SYNC barrier, see `_transfer_cost`) is the first real
    producer of the FABRIC cost dim and makes offload a genuine cost-governed decision: a claim only
    moves off its producer's channel when its compute is enough cheaper there to pay the move. Because a
    claim's transfer cost depends on where its producers already landed, the pass is order-dependent --
    the greedy forward placement IS the model. Tie-breaks stay deterministic (burdened cost, then
    channel name)."""
    from .kbcir.realize import optimize, _flatten
    from .kbcir.weights import PERF, weights as _weights

    policy = policy or PERF
    plans = {c.name: optimize(module, c.profile, theta, policy) for c in channels}
    step_of = {c.name: {s.claim_id: s for s in plans[c.name].steps} for c in channels}

    placements = []
    producer_of: dict[int, str] = {}     # rid -> name of the channel the claim that WROTE it landed on
    for phase_id, claim in _flatten(module):
        wv = _weights(None, theta, phase_id, policy)   # policy weights for this phase (target-neutral)
        cands = [(c, step_of[c.name][claim.id]) for c in channels
                 if channel_suits(claim, c) and claim.id in step_of[c.name]]
        if not cands:                                # nothing suits -> the host CPU channel
            hc = host_channel()
            if hc.name not in step_of:
                step_of[hc.name] = {s.claim_id: s for s in optimize(module, hc.profile, theta, policy).steps}
            cands = [(hc, step_of[hc.name][claim.id])]
        # burdened cost = the channel's compute score + the cross-device transfer to land inputs there.
        scored = [(c, step, _transfer_cost(module, claim, c, producer_of, wv)) for c, step in cands]
        ch, step, xfer = min(scored, key=lambda x: (x[1].cost + x[2], x[0].name))  # det. tie-break
        placements.append(ChannelPlacement(claim.id, claim.op, ch.name, ch.kind,
                                           step.candidate.name, step.candidate.width, step.cost, xfer))
        for rid in claim.wr:                         # record where this claim's outputs now live
            producer_of[rid] = ch.name
    return HeterogeneousPlan(tuple(placements))


def apply_channels(pack, plan: HeterogeneousPlan):
    """Tag a GEM StreamPack's lane segments with the channel each claim was placed on, yielding the
    *one* unified binary graph that runs across the whole tower (the executor dispatches per
    segment by `LaneSegment.channel`). Returns a new StreamPack (segments are frozen). This is the
    bridge from a heterogeneous plan to GEM execution."""
    from dataclasses import replace
    where = {p.claim_id: p.channel for p in plan.placements}
    segs = tuple(replace(s, channel=where.get(s.claim_id, s.channel)) for s in pack.segments)
    return replace(pack, segments=segs)
