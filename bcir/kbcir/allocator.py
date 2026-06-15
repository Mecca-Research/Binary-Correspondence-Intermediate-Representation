"""The RL allocator ("smart malloc"): intent-aware resource placement.

Standard allocators are blind to *why* memory is allocated. BCIR knows: a resource
is touched by a known set of claims across a known phase span, so its **lifespan**
(phase span), **access frequency** (touching claims) and **size** are all derivable
at plan time -- before a byte is written. A learned scorer predicts each resource's
"heat" from those static features and a placement pass puts **hot ephemeral**
resources in the fast tier (L1/L2 SRAM) and **cold durable** ones in DRAM/HBM.

Tied to `moegate`'s discipline: the scorer is a small linear model trained offline
(L2/L3, floats) and **frozen to Q8 integers** (`FrozenPlacer`) for a deterministic,
host-independent deployment forward -- exactly the `moegate.freeze` pattern. The
default `DEFAULT_PLACER` is a hand-set frozen scorer, so no training is required.

GAINS-ONLY (the deployment contract): `place()` relocates a resource off its
declared domain **only** when (a) it fits the fast tier's capacity and (b) the
modeled memory cost strictly drops. Otherwise the resource keeps its default tier.
A placement that helps nothing is `is_noop` -- the allocator never makes the simple
case slower, it only acts where there is a modeled win. Pure and deterministic; it
emits a *plan* (rid -> tier), it does not move memory itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Module
from .cost import MemTier, TargetProfile

Q8 = 256

# Fast (SRAM-like) tiers a hot resource may be promoted into, fastest first, with
# their byte capacities (the hierarchy's Tier.capacity is unset/0 in the seeded
# profile, so the allocator owns realistic SRAM budgets here). Hot resources win
# the scarce fast tiers by heat rank; cold/large resources stay in DRAM or go to
# the bandwidth tier (HBM) when they cannot be cached.
_FAST_TIERS = (MemTier.L1, MemTier.L2, MemTier.L3)
_FAST_CAPACITY = {MemTier.L1: 32 * 1024, MemTier.L2: 256 * 1024, MemTier.L3: 8 * 1024 * 1024}
_DEFAULT_TIER = MemTier.DRAM
_BANDWIDTH_TIER = MemTier.HBM


def _log2(n: int) -> int:
    return max(0, (max(1, n) - 1).bit_length())


@dataclass(frozen=True)
class ResourceProfile:
    """Static, plan-time features of a resource -- the allocator's 'intent'."""

    rid: int
    size_bytes: int
    lifespan: int        # phase span (1 = ephemeral, used within a single phase)
    frequency: int       # number of claims touching the resource
    declared_tier: MemTier

    @property
    def ephemeral(self) -> bool:
        return self.lifespan <= 1

    def features(self) -> tuple[int, int, int]:
        """(log2 size, inverse lifespan x16, frequency) -- the scorer's inputs."""
        return (_log2(self.size_bytes), 16 // max(1, self.lifespan), self.frequency)


def profile_resources(module: Module) -> dict[int, ResourceProfile]:
    """Derive every resource's lifespan / frequency / size from the claim graph
    (deterministic; no new model fields -- intent is read from how it is used)."""
    order = {ph.phase_id: i for i, ph in enumerate(module.phases)}
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    freq: dict[int, int] = {}
    for ph in module.phases:
        pos = order[ph.phase_id]
        for c in ph.claims:
            for rid in c.io_rids():
                freq[rid] = freq.get(rid, 0) + 1
                first[rid] = min(first.get(rid, pos), pos)
                last[rid] = max(last.get(rid, pos), pos)
    profiles: dict[int, ResourceProfile] = {}
    for rid, res in module.resources.items():
        if rid not in freq:
            continue  # untouched resource: nothing to place
        profiles[rid] = ResourceProfile(
            rid=rid, size_bytes=res.count * res.elem_bytes,
            lifespan=last[rid] - first[rid] + 1, frequency=freq[rid],
            declared_tier=MemTier[_tier_name(res)])
    return profiles


def _tier_name(res) -> str:
    from .cost import _DOMAIN_TIER
    return _DOMAIN_TIER.get(res.domain, MemTier.DRAM).name


# --- the learned heat scorer (frozen Q8, moegate discipline) ---------------------

@dataclass(frozen=True)
class FrozenPlacer:
    """A frozen Q8 heat predictor: heat = w . [1, log2size, inv_lifespan, freq].
    A *higher* heat => hotter => prefer the fast tier. Deterministic integer forward
    (no floats), so placement is identical across hosts. `gen` is R13 provenance."""

    wq: tuple[int, ...]      # Q8 weights: (bias, w_size, w_invlife, w_freq)
    gen: int = 1

    def heat(self, p: ResourceProfile) -> int:
        size, invlife, freq = p.features()
        x = (1, size, invlife, freq)
        return sum(w * xi for w, xi in zip(self.wq, x)) >> 8

    @property
    def fingerprint(self) -> int:
        h = 1469598103934665603
        for v in self.wq:
            h = ((h ^ (v & 0xFFFFFFFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h


# Hand-set default: small + frequent + short-lived scores hot; large + long-lived
# scores cold. (bias -, size penalises heat, inv_lifespan rewards ephemeral, freq
# rewards reuse.) Q8.
DEFAULT_PLACER = FrozenPlacer(wq=(0, -1 * Q8, 4 * Q8, 3 * Q8), gen=1)


def train_placer(dataset: list[tuple[ResourceProfile, int]], *, epochs: int = 400,
                 lr: float = 0.02, gen: int = 1) -> FrozenPlacer:
    """Fit the heat scorer to labelled (profile, hot?) examples via logistic SGD
    (offline, float), then freeze to Q8. `hot? in {0,1}`. Deterministic: fixed
    init/order/lr/epochs (the moegate/calibrate freeze recipe)."""
    import math
    w = [0.0, 0.0, 0.0, 0.0]
    for _ in range(epochs):
        for p, label in dataset:
            x = (1.0, *[float(v) for v in p.features()])
            z = sum(wi * xi for wi, xi in zip(w, x))
            pred = 1.0 / (1.0 + math.exp(-z))
            g = pred - label
            for i in range(4):
                w[i] -= lr * g * x[i]
    return FrozenPlacer(wq=tuple(int(round(wi * Q8)) for wi in w), gen=max(1, gen))


# --- the placement plan ----------------------------------------------------------

@dataclass(frozen=True)
class Placement:
    """An allocation plan: the chosen tier per resource, with the relocations the
    gains-only gate actually admitted."""

    tiers: dict[int, MemTier]
    moved: tuple[int, ...]                  # rids relocated off their declared tier
    rationale: dict[int, str] = field(default_factory=dict)

    def tier_of(self, rid: int) -> MemTier:
        return self.tiers.get(rid, _DEFAULT_TIER)

    @property
    def is_noop(self) -> bool:
        return not self.moved


def _mem_cost(p: ResourceProfile, tier: MemTier, h: TargetProfile) -> int:
    """Modeled memory cost of serving this resource from `tier`: traffic scaled by
    the tier's bandwidth factor + per-access latency scaled by the latency factor.
    Lower is better. Uses the existing tier factors -- the same model `realize`
    prices against, so the gate agrees with the optimizer."""
    t = h.mem.by_name(tier.name)
    traffic = (p.size_bytes * p.frequency * t.bw_factor) >> 8
    latency = (p.frequency * t.lat_factor) >> 8
    return traffic + latency


def place(module: Module, h: TargetProfile, placer: FrozenPlacer = DEFAULT_PLACER) -> Placement:
    """Plan resource placement: rank by predicted heat, fill the fast tiers by
    capacity, and admit a relocation only when it fits and strictly lowers modeled
    memory cost (gains-only). Cold/large resources stay in DRAM, or go to HBM when
    that is bandwidth-cheaper than their declared tier."""
    profiles = profile_resources(module)
    tiers: dict[int, MemTier] = {rid: p.declared_tier for rid, p in profiles.items()}
    moved: list[int] = []
    rationale: dict[int, str] = {}

    # Hottest first; ties broken by rid for determinism.
    ranked = sorted(profiles.values(), key=lambda p: (-placer.heat(p), p.rid))
    remaining = dict(_FAST_CAPACITY)

    for p in ranked:
        default_cost = _mem_cost(p, p.declared_tier, h)
        best_tier, best_cost = p.declared_tier, default_cost
        hot = placer.heat(p) > 0
        # HOT: contend for scarce SRAM -- the fastest tier that both fits (bytes)
        # and lowers modeled cost. (Cold data never evicts hot data from SRAM.)
        if hot:
            for t in _FAST_TIERS:
                if p.size_bytes <= _FAST_CAPACITY[t] and remaining[t] >= p.size_bytes:
                    cost = _mem_cost(p, t, h)
                    if cost < best_cost:
                        best_tier, best_cost = t, cost
                        break
        # COLD/large that cannot be cached: prefer the bandwidth tier when cheaper.
        if best_tier == p.declared_tier and p.size_bytes > _FAST_CAPACITY[MemTier.L3]:
            hbm = _mem_cost(p, _BANDWIDTH_TIER, h)
            if hbm < best_cost:
                best_tier, best_cost = _BANDWIDTH_TIER, hbm
        if best_tier != p.declared_tier:
            tiers[p.rid] = best_tier
            moved.append(p.rid)
            kind = "hot/ephemeral" if p.ephemeral else ("hot/reused" if hot else "cold/bandwidth")
            rationale[p.rid] = (f"{kind} -> {best_tier.name} "
                                f"(cost {default_cost}->{best_cost})")
            if best_tier in remaining:
                remaining[best_tier] -= p.size_bytes
    return Placement(tiers=tiers, moved=tuple(moved), rationale=rationale)
