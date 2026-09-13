"""L2 learning placement: the policy portfolio + the counterfactual replay gate.

The L2 law (LangRef Sec. 13): gain schedules (policy weight vectors, thresholds)
adapt only at checkpoints, only as members of a **portfolio of frozen,
generation-tagged policies**, and only through the **replay gate** -- a
counterfactual no-regression certificate computed on logged episodes. Plan-time
"adaptation" is a deterministic table lookup over the workload class; nothing
learns while anything executes.

The gate judges *outcomes*, not preferences: both policies' chosen plans are
priced under the incumbent's own scheduled metric M(pi, Theta), so a candidate
is admitted only if it never loses ground as currently measured
(shadow -> canary -> promote, in oracle form). Deterministic given the log.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Module
from .cost import Theta
from .realize import optimize
from .weights import ENERGY, PERF, SAFE, THROUGHPUT, Policy


# --- workload classes: the deterministic plan-time selector ----------------------


def classify(theta: Theta) -> str:
    """Bucket the live runtime state into a workload class (a pure table rule;
    thresholds are L2 parameters, adapted only through the gate)."""
    if theta.thermal >= 60 or theta.power >= 60 or theta.voltage >= 60:
        return "constrained"
    if theta.mem_pressure >= 60 or theta.contention >= 60:
        return "saturated"
    return "nominal"


_CLASS_POLICY = {"constrained": "energy", "saturated": "throughput", "nominal": "latency"}

#: The workload dimension of the table (G13): under a `nominal` runtime the declared workload
#: class decides -- an interactive latency budget wants the latency schedule, a batch or a
#: throughput target the throughput one. A runtime constraint outranks the workload: a hot
#: or saturated machine is one whatever the workload declares.
_WORKLOAD_POLICY = {"interactive": "latency", "batch": "throughput", "nominal": "latency"}


@dataclass(frozen=True)
class PortfolioEntry:
    """A frozen gain schedule with its certification state (generation-tagged)."""

    policy: Policy
    gen: int = 1
    certified: bool = False


@dataclass(frozen=True)
class ReplayCertificate:
    """The gate's verdict: counterfactual replay of candidate vs incumbent.

    When the episodes came from the measured corpus (`kbcir.measured`, G13), `corpus` names
    the corpus head the replay ran over and `logged` how many episodes the corpus logs for
    the key. A certificate over fewer episodes than the corpus logs is NOT admitting: the
    corpus decides which episodes are replayed, never the promoter, so a candidate cannot be
    promoted on the episodes it happens to win.
    """

    candidate: str
    incumbent: str
    episodes: int
    regressions: int
    corpus: str = ""
    logged: int = 0

    @property
    def covered(self) -> bool:
        """Every logged episode replayed (vacuously so for an analytic replay)."""
        return self.episodes == self.logged if self.corpus else True

    @property
    def admitted(self) -> bool:
        return self.episodes >= 1 and self.regressions == 0 and self.covered


@dataclass
class PolicyPortfolio:
    """The set of deployable gain schedules. Selection is a class-table lookup;
    mutation happens only through `promote` with an admitting certificate."""

    entries: dict[str, PortfolioEntry] = field(default_factory=dict)

    @staticmethod
    def default() -> "PolicyPortfolio":
        # The seeded incumbents are certified by construction (they ARE the
        # behavior every pinned worked example was measured under).
        return PolicyPortfolio(
            entries={
                p.name: PortfolioEntry(policy=p, gen=1, certified=True)
                for p in (PERF, THROUGHPUT, ENERGY, SAFE)
            }
        )

    def select(self, theta: Theta, workload=None) -> Policy:
        """Deterministic plan-time selection: (runtime class, workload class) -> certified
        entry. Without a declared workload the table is the runtime class alone (the
        historical rule); with one (`kbcir.workload.Workload`, G13) the workload class decides
        under a `nominal` runtime and a runtime constraint outranks it."""
        runtime = classify(theta)
        name = _CLASS_POLICY[runtime]
        if workload is not None and runtime == "nominal":
            name = _WORKLOAD_POLICY[workload.classify()]
        entry = self.entries.get(name)
        if entry is not None and entry.certified:
            return entry.policy
        return self.entries["latency"].policy

    def promote(
        self, name: str, candidate: Policy, certificate: ReplayCertificate
    ) -> PortfolioEntry:
        """Swap in a new gain schedule -- only behind an admitting certificate."""
        if not certificate.admitted:
            raise ValueError(
                f"replay gate rejected {certificate.candidate!r}: "
                f"{certificate.regressions} regression(s) over {certificate.episodes} episode(s)"
            )
        if certificate.candidate != candidate.name or certificate.incumbent != name:
            raise ValueError("certificate does not cover this promotion")
        incumbent = self.entries.get(name)
        entry = PortfolioEntry(
            policy=candidate, gen=(incumbent.gen + 1 if incumbent else 1), certified=True
        )
        self.entries[name] = entry
        return entry


# --- the replay gate --------------------------------------------------------------


def replay_gate(
    module: Module, h, candidate: Policy, incumbent: Policy, episodes: list[Theta]
) -> ReplayCertificate:
    """Counterfactual no-regression check on logged Theta episodes.

    For each episode, both policies plan; both plans are priced under the
    *incumbent's* scheduled metric M(pi, Theta) (`gem.overlap.price_scheduled`)
    -- one yardstick, two plans. A regression is any episode where the
    candidate's plan prices worse than the incumbent's.
    """
    from ..gem.overlap import price_scheduled

    regressions = 0
    for theta in episodes:
        plan_c = optimize(module, h, theta, candidate)
        plan_i = optimize(module, h, theta, incumbent)
        m_c = price_scheduled(module, plan_c, h, theta, incumbent).makespan
        m_i = price_scheduled(module, plan_i, h, theta, incumbent).makespan
        if m_c > m_i:
            regressions += 1
    return ReplayCertificate(
        candidate=candidate.name,
        incumbent=incumbent.name,
        episodes=len(episodes),
        regressions=regressions,
    )


def episodes_from(batches, calibrator, base: Theta = Theta.cool()) -> list[Theta]:
    """Fold logged telemetry batches into the Theta episodes the gate replays."""
    episodes: list[Theta] = []
    theta = base
    for events in batches:
        theta = calibrator.update(theta, events)
        episodes.append(theta)
    return episodes
