"""L3 instrumentation: the regret ledger -- the dashboard for the boundary.

The third-order question ("where does learning start?") is itself a measured
question (LangRef Sec. 13). This module is the instrument, not the actuator:
it continuously computes, from logged episodes, the gap between what each
deployed decision rule achieved and the hindsight-best alternative -- and
renders the verdict that tells you where the heuristic/learned boundary
belongs. Actuation stays human: a flagged rule is a *candidate* for retuning,
and any swap still goes through the L2 replay gate and carries R13 provenance.

Regret is judged under one fixed, neutral yardstick (default: the nominal
latency policy's scheduled metric M(pi, Theta)) -- never under the deployed
rule's own preferences, under which every rule is trivially optimal. In
production the yardstick is measured cycles from the data-DNA log; the ledger
schema is the contract either way. Deterministic given the log; the ledger is
generation-tagged like every other decision artifact (law R13).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Module
from .cost import Theta
from .realize import optimize
from .weights import PERF, Policy


@dataclass(frozen=True)
class RegretMeasurement:
    """One episode's verdict: deployed outcome vs the hindsight best."""

    rule: str            # the deployed decision rule (policy name)
    deployed: int        # M(deployed plan) under the judge metric
    best: int            # min over candidate rules, same metric
    best_rule: str

    @property
    def regret(self) -> int:
        return self.deployed - self.best  # >= 0: deployed is in the candidate set


@dataclass
class RegretEntry:
    """The ledger's books for one decision rule."""

    rule: str
    episodes: int = 0
    total_regret: int = 0
    worst_regret: int = 0

    @property
    def regret_rate(self) -> int:
        return self.total_regret // self.episodes if self.episodes else 0


@dataclass
class RegretLedger:
    """Accumulated regret per deployed rule (generation-tagged, law R13)."""

    gen: int = 1
    entries: dict[str, RegretEntry] = field(default_factory=dict)

    def record(self, m: RegretMeasurement) -> None:
        e = self.entries.setdefault(m.rule, RegretEntry(rule=m.rule))
        e.episodes += 1
        e.total_regret += m.regret
        e.worst_regret = max(e.worst_regret, m.regret)

    def dashboard(self) -> str:
        lines = [f"regret ledger (gen {self.gen})",
                 f"{'rule':<12} {'episodes':>8} {'total':>8} {'worst':>8} {'rate':>8}"]
        for name in sorted(self.entries):
            e = self.entries[name]
            lines.append(f"{e.rule:<12} {e.episodes:>8} {e.total_regret:>8} "
                         f"{e.worst_regret:>8} {e.regret_rate:>8}")
        return "\n".join(lines)


@dataclass(frozen=True)
class BoundaryVerdict:
    """Where the boundary belongs for one rule: keep it, or retune through the
    gate. A verdict is a recommendation -- never an actuation."""

    rule: str
    verdict: str         # "keep" | "retune"
    regret_rate: int
    episodes: int


def measure_regret(module: Module, h, theta: Theta, deployed: Policy,
                   candidates: list[Policy], judge: Policy = PERF) -> RegretMeasurement:
    """Hindsight regret of the deployed rule against the candidate set.

    Every candidate plans; every plan is priced under the *judge* metric
    M(pi, Theta) -- one yardstick, all plans. The deployed rule is always in
    the comparison set, so regret is non-negative by construction.
    """
    from ..gem.overlap import price_scheduled

    pool: list[Policy] = list(candidates)
    if all(p.name != deployed.name for p in pool):
        pool.append(deployed)

    deployed_m = None
    best_m, best_rule = None, deployed.name
    for pol in pool:
        plan = optimize(module, h, theta, pol)
        m = price_scheduled(module, plan, h, theta, judge).makespan
        if pol.name == deployed.name:
            deployed_m = m
        if best_m is None or m < best_m:
            best_m, best_rule = m, pol.name
    assert deployed_m is not None
    return RegretMeasurement(rule=deployed.name, deployed=deployed_m,
                             best=best_m, best_rule=best_rule)


def ledger_from_episodes(module: Module, h, episodes: list[Theta], portfolio,
                         judge: Policy = PERF, gen: int = 1) -> RegretLedger:
    """Replay the log: for each episode, the portfolio's deployed selection is
    measured against every certified alternative."""
    ledger = RegretLedger(gen=max(1, gen))
    candidates = [e.policy for e in portfolio.entries.values() if e.certified]
    for theta in episodes:
        deployed = portfolio.select(theta)
        ledger.record(measure_regret(module, h, theta, deployed, candidates, judge))
    return ledger


def boundary_report(ledger: RegretLedger, retune_threshold: int = 0) -> list[BoundaryVerdict]:
    """The boundary dashboard: rules whose sustained regret rate exceeds the
    threshold are candidates for retuning (through the replay gate); the rest
    keep their slot. Deterministic, sorted by rule name."""
    out: list[BoundaryVerdict] = []
    for name in sorted(ledger.entries):
        e = ledger.entries[name]
        verdict = "retune" if e.regret_rate > retune_threshold else "keep"
        out.append(BoundaryVerdict(rule=name, verdict=verdict,
                                   regret_rate=e.regret_rate, episodes=e.episodes))
    return out
