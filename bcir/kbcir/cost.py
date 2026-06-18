"""Cost vectors, the substrate profile H, and live runtime state Theta.

Costs are 12-dimensional integer vectors (deterministic -- no floats in the hot
path). The dimensions are the resource axes a transition spends on: arithmetic
throughput, memory traffic, interconnect, etc.

The coupling operator (X) (`CostVector.couple`) scales a base cost by a
per-dimension *context factor* in Q8 fixed point (256 == x1.0). This is the
f_i(pi) of the K_BCIR equation: a transition's realized cost depends on where/when
it runs (placement, fusion, thermal), not just on the raw opcode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from ..model.lanes import Domain

# 12-dimensional K_BCIR cost (matches the BCIR LangRef v0.1 !kbcir.costvec).
# `verification` is the 12th axis: the cost of discharging a claim's verify
# contract (bounds/exact/hash). Currently modeled as 0 in the oracle until
# verify-cost modeling lands; the dimension exists in the algebra now for lockstep.
DIMS = (
    "compute", "memory", "fabric", "sync", "compile",
    "thermal", "power", "reliability", "security", "accuracy", "contention",
    "verification",
)
N = len(DIMS)
(COMPUTE, MEMORY, FABRIC, SYNC, COMPILE, THERMAL, POWER, RELIABILITY,
 SECURITY, ACCURACY, CONTENTION, VERIFICATION) = range(N)
_INDEX = {name: i for i, name in enumerate(DIMS)}
IDENTITY_FACTOR = (256,) * N  # x1.0 in Q8 per dimension


@dataclass(frozen=True)
class CostVector:
    v: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.v) != N:
            raise ValueError(f"CostVector needs {N} dims, got {len(self.v)}")

    @classmethod
    def zero(cls) -> "CostVector":
        return cls((0,) * N)

    @classmethod
    def of(cls, **named: int) -> "CostVector":
        vals = [0] * N
        for name, value in named.items():
            if name not in _INDEX:
                raise KeyError(f"unknown cost dim {name!r}")
            vals[_INDEX[name]] = int(value)
        return cls(tuple(vals))

    def __add__(self, other: "CostVector") -> "CostVector":
        return CostVector(tuple(a + b for a, b in zip(self.v, other.v)))

    def couple(self, factor: tuple[int, ...]) -> "CostVector":
        """(X) : element-wise scale by a Q8 context-factor vector."""
        return CostVector(tuple((c * f) >> 8 for c, f in zip(self.v, factor)))

    def dot(self, weights: tuple[int, ...]) -> int:
        """Scalarize under a weight vector: sum_d w_d * c_d."""
        return sum(c * w for c, w in zip(self.v, weights))

    def as_dict(self) -> dict[str, int]:
        return {name: self.v[i] for i, name in enumerate(DIMS)}


@dataclass(frozen=True)
class Theta:
    """Live runtime state. All fields are 0..100 normalized pressures (0 = idle/cool)."""

    thermal: int = 0       # 0 cool .. 100 throttling
    power: int = 0         # 0 plenty .. 100 capped
    mem_pressure: int = 0  # 0 free .. 100 bandwidth-starved
    contention: int = 0    # 0 idle .. 100 congested
    noise: int = 0
    wear: int = 0
    utilization: int = 0
    voltage: int = 0       # CT4 data-DNA axis (reserved; 0 = nominal)

    @staticmethod
    def cool() -> "Theta":
        return Theta()

    @staticmethod
    def hot() -> "Theta":
        return Theta(thermal=80, power=60)

    @staticmethod
    def mem_bound() -> "Theta":
        return Theta(mem_pressure=80, contention=50)


# --- memory hierarchy (HBM / HAM / CXL) -----------------------------------------
#
# Tier factors are Q8 relative to DRAM (256 == x1.0). bw_factor scales the
# bandwidth (traffic) term: lower = more bandwidth = cheaper (HBM). lat_factor
# scales the per-access latency term. This is the C_mem = sum_levels expansion: a
# resource's domain selects its tier, and the memory cost scales by that tier's
# factors. DRAM is the baseline, so RAM resources cost exactly as before.

class MemTier(IntEnum):
    """Memory hierarchy tier (CT1). Integer values are normative and MUST match
    ``BCIR_MemTier`` in ``mlir/include/BCIR/BCIRAttrs.td`` exactly (docs/PARITY.md)."""

    L1 = 0
    L2 = 1
    L3 = 2
    DRAM = 3
    HBM = 4
    CXL = 5
    SSD = 6


@dataclass(frozen=True)
class Tier:
    name: str
    latency_cyc: int
    bw_factor: int   # Q8 vs DRAM (lower = more bandwidth)
    lat_factor: int  # Q8 vs DRAM
    capacity: int = 0

    @property
    def mem_tier(self) -> MemTier:
        """The normative tier id (tier names mirror the MemTier enum)."""
        return MemTier[self.name]


# Domain -> memory tier. Domains already exist in the model; we map, not add.
_DOMAIN_TIER = {
    Domain.RAM: MemTier.DRAM,
    Domain.VRAM: MemTier.HBM,
    Domain.NVM: MemTier.SSD,
    Domain.MMIO: MemTier.DRAM,
    Domain.CXL: MemTier.CXL,
    Domain.HBM: MemTier.HBM,
}

_DRAM = Tier("DRAM", latency_cyc=200, bw_factor=256, lat_factor=256)


@dataclass(frozen=True)
class MemoryHierarchy:
    tiers: tuple[Tier, ...]

    def by_name(self, name: str) -> Tier:
        for t in self.tiers:
            if t.name == name:
                return t
        return _DRAM

    def tier_for(self, domain: Domain) -> Tier:
        return self.by_name(_DOMAIN_TIER.get(domain, MemTier.DRAM).name)

    @staticmethod
    def default() -> "MemoryHierarchy":
        return MemoryHierarchy((
            Tier("L1", 4, 16, 16),
            Tier("L2", 12, 32, 48),
            Tier("L3", 40, 96, 96),
            _DRAM,                                            # x1.0 baseline
            Tier("HBM", 160, bw_factor=64, lat_factor=192),  # ~4x bandwidth, lower latency
            Tier("CXL", 350, bw_factor=384, lat_factor=512),  # memory-semantic, costlier than DRAM
            Tier("SSD", 5000, bw_factor=1024, lat_factor=4096),  # semantic-swap backing store
        ))


# --- substrate / target descriptor H (the open container) -----------------------

@dataclass(frozen=True)
class TargetProfile:
    """Substrate descriptor H -- an *open container* for any target (x86/ARM/RISC-V/GPU).

    The cost model is target-neutral; this profile supplies the target-specific
    constants (lane geometry, cache line, gather penalty, heat/power density) and
    the memory hierarchy. New targets are added as factories, not optimizer code.
    """

    name: str = "x86-64-avx2"
    triple: str = "x86_64-avx2"
    cacheline: int = 64
    elem_bytes: int = 4
    lane_widths: tuple[int, ...] = (1, 8)   # element lanes per vector op
    warp: int = 0                           # GPU warp size (0 = not a warp machine)
    scalable: bool = False                  # vector-length-agnostic (SVE / RVV)
    gather_penalty: int = 32                # per-access overhead for a random DRAM-miss gather
    mem_unit: int = 1                       # bandwidth term per element per stream
    base_overhead: int = 4                  # latency term per memory access op (unit stride)
    thermal_density: int = 64               # heat ~ vector width
    power_density: int = 64                 # current draw ~ vector width
    per_op_heat: int = 1
    fma: bool = True
    isa_features: frozenset[str] = frozenset()
    affinity_domains: int = 1               # cores / SMs available for pinning (CT2)
    mem_channels: int = 4                   # concurrent bandwidth-bound streams sustained
    cal_gen: int = 0                        # cost-table generation (0 = seeded constants;
                                            # >=1 = a frozen CalibratedProfile was applied)
    mem: MemoryHierarchy = MemoryHierarchy.default()

    @property
    def vector_width(self) -> int:
        return max(self.lane_widths)

    def widths(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.lane_widths)))

    # -- target factories (the container stays open: add a factory, not optimizer code) --
    @staticmethod
    def x86_avx2() -> "TargetProfile":
        return TargetProfile(name="x86-64-avx2", triple="x86_64-avx2", lane_widths=(1, 8),
                             isa_features=frozenset({"avx2", "fma"}), affinity_domains=8)

    @staticmethod
    def x86_avx512() -> "TargetProfile":
        return TargetProfile(name="x86-64-avx512", triple="x86_64-avx512", lane_widths=(1, 8, 16),
                             isa_features=frozenset({"avx2", "avx512f", "fma"}), affinity_domains=8)

    @staticmethod
    def arm64_neon() -> "TargetProfile":
        return TargetProfile(name="aarch64-neon", triple="aarch64-neon", cacheline=64,
                             lane_widths=(1, 4), isa_features=frozenset({"neon"}), affinity_domains=8)

    @staticmethod
    def arm64_sve() -> "TargetProfile":
        # SVE is vector-length-agnostic; we model a representative width set.
        return TargetProfile(name="aarch64-sve", triple="aarch64-sve", cacheline=64,
                             lane_widths=(1, 8, 16), scalable=True,
                             isa_features=frozenset({"neon", "sve"}), affinity_domains=8)

    @staticmethod
    def nvidia_ptx() -> "TargetProfile":
        # Warp machine: wide lanes, big HBM bandwidth, coalesced gather cheaper than CPU.
        return TargetProfile(name="nvptx64-warp", triple="nvptx64-warp", cacheline=128,
                             lane_widths=(1, 32), warp=32, gather_penalty=16,
                             thermal_density=32, power_density=48,
                             isa_features=frozenset({"ptx", "warp"}), affinity_domains=128,
                             mem_channels=32)  # HBM: many concurrent streams

    @staticmethod
    def riscv_rvv() -> "TargetProfile":
        # Open-ISA anchor (documented; proves the container is extensible). RVV is length-agnostic.
        return TargetProfile(name="riscv64-rvv", triple="riscv64-rvv", lane_widths=(1, 8, 16),
                             scalable=True, isa_features=frozenset({"rvv"}), affinity_domains=4)

    @staticmethod
    def for_host() -> "TargetProfile":
        """The profile matching the *host* machine, so a plain invocation plans for the
        hardware it actually runs on -- a Raspberry Pi 5 gets NEON/SVE, not AVX-512. Reads
        platform.machine() + the CPU feature flags (/proc/cpuinfo): aarch64 -> SVE if present
        else NEON; x86_64 -> AVX-512 if avx512f else AVX2; riscv64 -> RVV; anything else -> AVX2
        (a conservative default). BCIR is hardware-agnostic: nothing should assume x86."""
        import platform
        mach = platform.machine().lower()
        feat = _host_cpu_features()
        if mach in ("aarch64", "arm64"):
            return TargetProfile.arm64_sve() if "sve" in feat else TargetProfile.arm64_neon()
        if mach in ("x86_64", "amd64"):
            return TargetProfile.x86_avx512() if "avx512f" in feat else TargetProfile.x86_avx2()
        if mach.startswith("riscv"):
            return TargetProfile.riscv_rvv()
        return TargetProfile.x86_avx2()


# Back-compat alias: existing code/tests use `HProfile`.
HProfile = TargetProfile


def _host_cpu_features() -> frozenset:
    """The host CPU feature flags from /proc/cpuinfo ('flags' on x86, 'Features' on ARM);
    empty if unreadable -- a portable best-effort that never raises."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith(("flags", "Features")):
                    return frozenset(line.split(":", 1)[1].split())
    except OSError:
        pass
    return frozenset()

# Named registry so the CLI/tests can select a target by name.
TARGETS = {
    "x86_avx2": TargetProfile.x86_avx2(),
    "x86_avx512": TargetProfile.x86_avx512(),
    "arm64_neon": TargetProfile.arm64_neon(),
    "arm64_sve": TargetProfile.arm64_sve(),
    "nvidia_ptx": TargetProfile.nvidia_ptx(),
    "riscv_rvv": TargetProfile.riscv_rvv(),
}


def default_target_name() -> str:
    """The TARGETS key matching the host machine (TargetProfile.for_host()) -- the arch-neutral
    default for the CLI / API / bench, so BCIR plans for the hardware it runs on rather than
    assuming x86. Falls back to x86_avx512 only if the host profile is unnamed in TARGETS."""
    host = TargetProfile.for_host()
    for key, prof in TARGETS.items():
        if prof.name == host.name:
            return key
    return "x86_avx512"
