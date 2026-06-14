"""bcir.kbcir.throttle -- the L1 cost throttle: keep performance throttled where
it matters.

ML at every layer has latency and energy cost. The separation that keeps it
bounded is the amortization tiering (LangRef Sec. 13): a learned component may
live at a tier only if the improvement it buys outweighs the inference it spends
*at that tier's decision rate*. This module makes that placement a *checkable*
artifact instead of doctrine:

  * **L0 prohibition** -- the hot path runs no learned inference (decisions are
    compiled out). A component declared at tier L0 must have `inference_cost == 0`.
  * **Amortization** -- at any tier, a component pays for itself: `gain >=
    inference_cost` per decision, and `inference_cost <= budget` (the tier's
    declared inference budget -- the throttle). The accelerator is the ideal
    case: net-negative overhead (its inference is cheaper than the search it
    saves), so gain >= inference_cost trivially.

An `AmortizationCertificate` declares (component, tier, inference_cost, gain,
budget) and is *admitted* iff it satisfies the L0 prohibition + amortization +
budget. A deployed learned component must carry an admitting certificate,
witnessed by R13 (`bcir.kbcir.amortization`). Cross-references the provenance
manifest: the manifest records *which* components ran, the throttle bounds *what
they cost*.

`measure_inference_cost` times a component offline (floats, L2/L3) to fill the
certificate; the certificate itself is integer/deterministic. The L0/L1 hot path
never runs this -- it runs the compiled-out decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Per-tier inference budgets (Q-units per decision). L0 is zero by law: the hot
# path carries decisions, not models. L1+ budgets are generous (plan-time
# amortizes over the executions a plan governs) but finite -- the throttle.
TIER_BUDGET = {"L0": 0, "L1": 100_000, "L2": 10_000_000, "L3": 1_000_000_000}


@dataclass(frozen=True)
class AmortizationCertificate:
    """A learned component's plan-time cost declaration + its placement verdict."""

    component: str
    tier: str                 # "L0" | "L1" | "L2" | "L3"
    inference_cost: int       # cost per decision (Q-units; 0 == compiled-out)
    gain: int                 # improvement per decision (same units)
    budget: int = None        # tier inference budget (defaults to TIER_BUDGET[tier])

    def _budget(self) -> int:
        return self.budget if self.budget is not None else TIER_BUDGET.get(self.tier, 0)

    @property
    def within_budget(self) -> bool:
        return self.inference_cost <= self._budget()

    @property
    def amortizes(self) -> bool:
        return self.gain >= self.inference_cost

    @property
    def l0_ok(self) -> bool:
        return self.tier != "L0" or self.inference_cost == 0

    @property
    def admitted(self) -> bool:
        """The component belongs at its tier: L0 prohibition + it pays for itself
        + within the tier's inference budget."""
        return (self.tier in TIER_BUDGET and self.l0_ok and self.amortizes
                and self.within_budget)


def certify(component: str, tier: str, inference_cost: int, gain: int,
            budget: int = None) -> AmortizationCertificate:
    return AmortizationCertificate(component=component, tier=tier,
                                   inference_cost=int(inference_cost), gain=int(gain),
                                   budget=budget)


@dataclass
class ThrottleReport:
    """The dashboard: which learned components run, what they cost, and whether
    each pays for itself at its tier."""

    certs: list = field(default_factory=list)

    def admitted(self) -> bool:
        return all(c.admitted for c in self.certs)

    def offenders(self) -> list:
        return [c for c in self.certs if not c.admitted]

    def render(self) -> str:
        lines = [f"L1 cost throttle ({len(self.certs)} component(s))",
                 f"{'component':<16} {'tier':>4} {'infer':>10} {'gain':>10} "
                 f"{'budget':>12} {'verdict':>9}"]
        for c in self.certs:
            lines.append(f"{c.component:<16} {c.tier:>4} {c.inference_cost:>10} "
                         f"{c.gain:>10} {c._budget():>12} "
                         f"{'admit' if c.admitted else 'THROTTLE':>9}")
        return "\n".join(lines)


def measure_inference_cost(fn, reps: int = 5) -> int:
    """Median wall-time (ns) of `fn` over `reps` -- an offline (L2/L3) measurement
    to fill a certificate. Never run on the hot path; the value is advisory and
    the certificate it feeds is the integer/deterministic artifact."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        times.append(max(1, time.perf_counter_ns() - t0))
    return sorted(times)[len(times) // 2]
