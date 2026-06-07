"""Scalarization weights w(H, Theta, phase, policy).

K_BCIR selection is `score = w . (T_i (X) f_i(pi))`. The weight vector encodes the
policy (what we are optimizing for) and rises/falls with the live runtime state
Theta: a hot machine raises the thermal/power/reliability weights (so wide SIMD
becomes expensive and the optimizer replans to a narrower lane), a memory-bound
machine raises the memory/fabric/contention weights.

Weights are integers (deterministic). The dimension order matches
``cost.DIMS``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cost import (
    CONTENTION,
    FABRIC,
    MEMORY,
    N,
    POWER,
    RELIABILITY,
    THERMAL,
    Theta,
)


@dataclass(frozen=True)
class Policy:
    """A K_BCIR scalarization policy: base per-dimension weights + a name."""

    name: str
    base: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.base) != N:
            raise ValueError(f"policy needs {N} weights, got {len(self.base)}")


# Dimension order: compute, memory, fabric, sync, compile, thermal, power,
#                  reliability, security, accuracy, contention, verification.
# Matches the LangRef vec_add policy `array<i64: 2,2,1,1,0,0,0,1,0,0,1,1>`.
PERF = Policy("latency", (2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1))
THROUGHPUT = Policy("throughput", (2, 3, 1, 1, 0, 0, 0, 1, 0, 0, 2, 1))
ENERGY = Policy("energy", (1, 1, 1, 1, 0, 2, 3, 1, 0, 0, 1, 1))
SAFE = Policy("safe", (1, 1, 1, 2, 1, 1, 1, 3, 1, 0, 1, 2))

POLICIES = {p.name: p for p in (PERF, THROUGHPUT, ENERGY, SAFE)}


def weights(h, theta: Theta, phase_id: int, policy: Policy = PERF) -> tuple[int, ...]:
    """Return the live weight vector for a phase under H and Theta.

    `h` (the TargetProfile) is accepted for signature stability and future
    target-specific weighting; the base policy already encodes target-neutral
    intent. Theta modulates the heat/power/contention axes.
    """
    w = list(policy.base)
    # Hot machine -> penalize heat/power/reliability axes (AVX-512 downclock effect).
    w[THERMAL] += theta.thermal // 20
    w[POWER] += theta.power // 20
    w[RELIABILITY] += (theta.thermal + theta.wear) // 50
    # Memory-bound machine -> penalize traffic/interconnect/contention axes.
    w[MEMORY] += theta.mem_pressure // 40
    w[FABRIC] += theta.mem_pressure // 60
    w[CONTENTION] += theta.contention // 40
    return tuple(w)
