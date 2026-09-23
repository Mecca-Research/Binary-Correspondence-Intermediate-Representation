"""The measured-candidate corpus (G13 / S2-E): B1's measured evidence generalized to whole
plans and filed under the scope it was measured in.

`kbcir.schedule_artifact` (B1) records real counters around one matmul's tile plans. The
corpus here holds the same kind of evidence for any program's *plan* -- the realization
`optimize` selects under a policy, executed by a caller's `work` and sampled by the OS
counters and the hardware PMU B1 uses -- keyed by the four identities a measurement is of:
the program (its R13 digest), the target (B1's fingerprint), the workload (`W`, G13) and
Theta. Every entry carries the host's attestation (`kbcir.microbench.host_attestation`, the
S0-F rig's rule), so a reader can see whether the samples come from silicon or from a
hypervisor, and the raw samples themselves -- one per repeat, nothing discarded.

Three laws:

  * **append-only and content-addressed.** An entry's digest covers its content; the corpus
    chains the digests (h_i = sha256(h_{i-1} || d_i)), so its head is the identity of the
    whole history, and a corpus read back from JSON is refused if any entry was altered,
    removed or reordered. `extends` says whether one corpus is a prefix-extension of another
    -- the only relation two versions of a store may have.
  * **informs, never decides.** Nothing here changes what `verify` accepts or what `optimize`
    selects. The corpus orders *policies* (each plan being `optimize`'s under that policy)
    through the L2 replay gate, and a plan the evidence names must still re-derive from the
    planner: stale evidence (the planner moved since it was measured) is not used.
  * **the whole log, or nothing.** The measured replay gate replays EVERY logged episode of a
    key; a candidate without evidence on one of them is refused rather than judged on the
    rest, and the certificate names the corpus head and the number of episodes the corpus
    logs, so the portfolio can refuse a certificate over a subset.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .._artifact_json import read_bounded_text, strict_json_loads
from .portfolio import ReplayCertificate

SCHEMA = "bcir.measured_plan.v1"
CORPUS_SCHEMA = "bcir.measured_corpus.v1"

#: The tenancies the rig can attest (mirrors `kbcir.microbench.TENANCIES`).
TENANCIES = ("bare-metal", "virtualized", "containerized", "unproven")
_MAX_SAMPLES = 128  # the rig's repeat bound
_STR_MAX = 256


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hex_digest(value, what: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{what} must be a lowercase SHA-256 digest")
    return value


def _u63(value, what: str, lo: int = 0) -> int:
    if type(value) is not int or value < lo or value >= 1 << 63:
        raise ValueError(f"{what} must be an integer in [{lo}, 2^63)")
    return value


def _bounded_str(value, what: str, limit: int = _STR_MAX) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError(f"{what} must be a nonempty string of at most {limit} characters")
    return value


@dataclass(frozen=True)
class MeasuredSample:
    """One repeat: the OS counters around one execution of the plan (nothing discarded)."""

    wall_ns: int
    cpu_ns: int
    minor_faults: int
    major_faults: int
    voluntary_context_switches: int
    involuntary_context_switches: int

    def __post_init__(self) -> None:
        for name in (
            "cpu_ns",
            "minor_faults",
            "major_faults",
            "voluntary_context_switches",
            "involuntary_context_switches",
        ):
            _u63(getattr(self, name), name)
        _u63(self.wall_ns, "wall_ns", lo=1)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MeasuredPlan:
    """One measured candidate: the plan a policy selected for (program, target, workload,
    Theta), the raw samples of executing it, the PMU counters when the host has a PMU, and
    the host's attestation."""

    program: str  # the module's R13 digest (decimal), or B1's family:shape for an artifact
    target: str  # `schedule_artifact.target_fingerprint` (SHA-256)
    workload: str  # `Workload.digest()` (SHA-256)
    theta: str  # `provenance.hash_theta` (decimal)
    policy: str  # the admitted candidate: a portfolio policy name, or B1's tile plan name
    plan: str  # `assignment_digest` of what the policy selected when measured (SHA-256)
    samples: tuple[MeasuredSample, ...]
    cycles: int
    instructions: int
    cache_misses: int
    tenancy: str
    signals: str
    hardware_pmu: bool
    host_architecture: str
    source_commit: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("unsupported measured-plan schema")
        _bounded_str(self.program, "program")
        _hex_digest(self.target, "target")
        _hex_digest(self.workload, "workload")
        if not isinstance(self.theta, str) or not self.theta.isdigit():
            raise ValueError("theta must be the decimal Theta hash")
        _bounded_str(self.policy, "policy")
        _hex_digest(self.plan, "plan")
        if (
            not isinstance(self.samples, (tuple, list))
            or not self.samples
            or len(self.samples) > _MAX_SAMPLES
            or any(not isinstance(sample, MeasuredSample) for sample in self.samples)
        ):
            raise ValueError(f"samples must be 1..{_MAX_SAMPLES} measured repeats")
        object.__setattr__(self, "samples", tuple(self.samples))
        for name in ("cycles", "instructions", "cache_misses"):
            _u63(getattr(self, name), name)
        if self.tenancy not in TENANCIES:
            raise ValueError(f"tenancy must be one of {TENANCIES}")
        _bounded_str(self.signals, "signals")
        if type(self.hardware_pmu) is not bool:
            raise ValueError("hardware_pmu must be a bool")
        _bounded_str(self.host_architecture, "host_architecture")
        if (
            not isinstance(self.source_commit, str)
            or not 7 <= len(self.source_commit) <= 64
            or self.source_commit != self.source_commit.lower()
            or any(character not in "0123456789abcdef" for character in self.source_commit)
        ):
            raise ValueError("source_commit must be a hexadecimal commit ID")

    @property
    def median_wall_ns(self) -> int:
        return int(statistics.median(sample.wall_ns for sample in self.samples))

    def stats(self) -> tuple[int, int, int, int]:
        """(min, median, max, MAD) of the wall samples -- derived, never stored."""
        walls = sorted(sample.wall_ns for sample in self.samples)
        median = int(statistics.median(walls))
        mad = int(statistics.median(abs(wall - median) for wall in walls))
        return walls[0], median, walls[-1], mad

    @property
    def silicon(self) -> bool:
        """True only for the tenancy the evidence proved: bare metal with a PMU."""
        return self.tenancy == "bare-metal" and self.hardware_pmu

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "program": self.program,
            "target": self.target,
            "workload": self.workload,
            "theta": self.theta,
            "policy": self.policy,
            "plan": self.plan,
            "samples": [sample.to_dict() for sample in self.samples],
            "cycles": self.cycles,
            "instructions": self.instructions,
            "cache_misses": self.cache_misses,
            "tenancy": self.tenancy,
            "signals": self.signals,
            "hardware_pmu": self.hardware_pmu,
            "host_architecture": self.host_architecture,
            "source_commit": self.source_commit,
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return _sha256_of(self.to_json())

    @staticmethod
    def from_dict(value) -> "MeasuredPlan":
        expected = {
            "schema",
            "program",
            "target",
            "workload",
            "theta",
            "policy",
            "plan",
            "samples",
            "cycles",
            "instructions",
            "cache_misses",
            "tenancy",
            "signals",
            "hardware_pmu",
            "host_architecture",
            "source_commit",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or not isinstance(value["samples"], list)
        ):
            raise ValueError("measured plan has missing or unknown fields")
        try:
            samples = tuple(
                MeasuredSample(**row) for row in value["samples"] if isinstance(row, dict)
            )
            if len(samples) != len(value["samples"]):
                raise ValueError("malformed sample")
            fields = {key: value[key] for key in expected if key != "samples"}
            return MeasuredPlan(samples=samples, **fields)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed measured plan: {exc}") from exc


def assignment_digest(result) -> str:
    """The content address of what a plan selected: (claim, phase, candidate, width, lane)
    per step, in the plan's order. What the evidence was measured on, and what a re-derived
    plan must equal for the evidence to apply."""
    rows = [
        [
            step.claim_id,
            step.phase_id,
            step.candidate.name,
            step.candidate.width,
            getattr(step.candidate.lane, "name", str(step.candidate.lane)),
        ]
        for step in result.steps
    ]
    return _sha256_of(_canonical(rows))


def scope_key(module, target, workload) -> tuple[str, str, str]:
    """(program, target, workload) as the corpus files them."""
    from .provenance import module_identity
    from .schedule_artifact import target_fingerprint

    return str(module_identity(module).digest), target_fingerprint(target), workload.digest()


def theta_key(theta) -> str:
    from .provenance import hash_theta

    return str(hash_theta(theta))


class MeasuredCorpus:
    """The append-only, content-addressed store of measured plans."""

    def __init__(self, entries=()) -> None:
        self._entries: list[MeasuredPlan] = []
        self._chain: list[str] = []
        self._digests: set[str] = set()
        for entry in entries:
            self.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[MeasuredPlan, ...]:
        return tuple(self._entries)

    @property
    def chain(self) -> tuple[str, ...]:
        return tuple(self._chain)

    @property
    def head(self) -> str:
        """The identity of the whole history ("" for an empty corpus)."""
        return self._chain[-1] if self._chain else ""

    def append(self, entry: MeasuredPlan) -> str:
        """File one measured plan; returns the new head. Evidence already held is refused
        (the corpus counts evidence, so a duplicate would count twice)."""
        if not isinstance(entry, MeasuredPlan):
            raise ValueError("a corpus holds MeasuredPlan entries")
        digest = entry.digest
        if digest in self._digests:
            raise ValueError(f"the corpus already holds this evidence ({digest[:12]})")
        head = _sha256_of(self.head + digest)
        self._entries.append(entry)
        self._chain.append(head)
        self._digests.add(digest)
        return head

    def lookup(
        self, program: str, target: str, workload: str, *, theta: str | None = None, policy=None
    ) -> tuple[MeasuredPlan, ...]:
        name = policy if policy is None or isinstance(policy, str) else policy.name
        return tuple(
            entry
            for entry in self._entries
            if entry.program == program
            and entry.target == target
            and entry.workload == workload
            and (theta is None or entry.theta == theta)
            and (name is None or entry.policy == name)
        )

    def episodes(self, program: str, target: str, workload: str) -> tuple[str, ...]:
        """The Theta keys the corpus logs for a scope, in a fixed order."""
        keys = {entry.theta for entry in self.lookup(program, target, workload)}
        return tuple(sorted(keys, key=int))

    def census(
        self, program: str, target: str, workload: str, *, theta: str | None = None
    ) -> tuple[str, ...]:
        """The policies with evidence for a scope (and episode), in name order."""
        return tuple(
            sorted({entry.policy for entry in self.lookup(program, target, workload, theta=theta)})
        )

    def physical_targets(self) -> int:
        """Distinct targets whose evidence is attested silicon (bare metal with a PMU) -- the
        count the two-target rule reads."""
        return len({entry.target for entry in self._entries if entry.silicon})

    def extends(self, older: "MeasuredCorpus") -> bool:
        """True iff this corpus is `older` with entries appended: the only relation two
        versions of an append-only store may have."""
        return len(self._chain) >= len(older._chain) and self._chain[: len(older._chain)] == list(
            older._chain
        )

    def to_dict(self) -> dict:
        return {
            "schema": CORPUS_SCHEMA,
            "entries": [entry.to_dict() for entry in self._entries],
            "chain": list(self._chain),
            "head": self.head,
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @staticmethod
    def from_dict(value) -> "MeasuredCorpus":
        """Rebuild a corpus and re-derive its chain: an altered, removed or reordered entry,
        or a forged chain, is refused -- the head a reader trusts is the one it recomputed."""
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "entries", "chain", "head"}
            or value["schema"] != CORPUS_SCHEMA
            or not isinstance(value["entries"], list)
            or not isinstance(value["chain"], list)
        ):
            raise ValueError("measured corpus has missing or unknown fields")
        corpus = MeasuredCorpus()
        for row in value["entries"]:
            corpus.append(MeasuredPlan.from_dict(row))
        if list(corpus.chain) != value["chain"] or corpus.head != value["head"]:
            raise ValueError(
                "corpus chain does not match its entries: altered, removed or reordered evidence"
            )
        return corpus

    @staticmethod
    def from_json(text: str) -> "MeasuredCorpus":
        return MeasuredCorpus.from_dict(strict_json_loads(text, "measured corpus"))


def write_corpus(path: os.PathLike | str, corpus: MeasuredCorpus) -> None:
    if not isinstance(corpus, MeasuredCorpus):
        raise ValueError("corpus must be a MeasuredCorpus")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(corpus.to_json().encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_corpus(path: os.PathLike | str) -> MeasuredCorpus:
    return MeasuredCorpus.from_json(read_bounded_text(str(path), "measured corpus"))


# --- producing evidence ---------------------------------------------------------------------


def measure_plans(
    module,
    h,
    theta,
    policies,
    work,
    *,
    workload,
    source_commit: str,
    repeats: int = 3,
) -> tuple[MeasuredPlan, ...]:
    """B1's discipline for whole plans: for every policy, the plan `optimize` selects is run
    once unmeasured (warm-up, excluded), then `repeats` times under the OS counters, once
    more under the hardware PMU when the host exposes one; the entry carries the host's
    attestation. `work(result)` executes the plan; it is the caller's, as in B1."""
    from ..silicon import CounterSampler, read_hw_counters
    from .microbench import host_attestation
    from .realize import optimize

    if not callable(work):
        raise ValueError("work must be callable")
    _u63(repeats, "repeats", lo=1)
    if repeats > _MAX_SAMPLES:
        raise ValueError(f"repeats must be at most {_MAX_SAMPLES}")
    members = tuple(policies)
    if not members or len({policy.name for policy in members}) != len(members):
        raise ValueError("policies must be a nonempty set of distinct names")
    program, target, workload_digest = scope_key(module, h, workload)
    episode = theta_key(theta)
    attestation = host_attestation()
    architecture = platform.machine() or "unknown"
    out = []
    for policy in members:
        result = optimize(module, h, theta, policy)
        work(result)  # one bounded warm-up, excluded from evidence
        samples = []
        for _ in range(repeats):
            sampler = CounterSampler()
            work(result)
            lap = sampler.lap()
            samples.append(
                MeasuredSample(
                    max(1, lap.wall_ns),
                    lap.cpu_ns,
                    lap.minor_faults,
                    lap.major_faults,
                    lap.vol_ctx,
                    lap.invol_ctx,
                )
            )
        hardware = read_hw_counters(functools.partial(work, result))
        out.append(
            MeasuredPlan(
                program,
                target,
                workload_digest,
                episode,
                policy.name,
                assignment_digest(result),
                tuple(samples),
                hardware.cycles if hardware else 0,
                hardware.instructions if hardware else 0,
                hardware.cache_misses if hardware else 0,
                attestation["tenancy"],
                attestation["signals"],
                bool(attestation["hardware_pmu"]),
                architecture,
                source_commit,
            )
        )
    return tuple(out)


def from_schedule_artifact(artifact, workload, theta=None) -> tuple[MeasuredPlan, ...]:
    """B1's matmul evidence as corpus entries: one per measured tile plan. B1 recorded
    medians and no attestation, so each entry carries one sample and the tenancy
    `unproven` -- the corpus says what B1 knew, not more."""
    from .cost import Theta
    from .schedule_artifact import ScheduleArtifact, _plan_dict

    if not isinstance(artifact, ScheduleArtifact):
        raise ValueError("artifact must be a ScheduleArtifact")
    M, N, K = artifact.shape
    episode = theta_key(theta if theta is not None else Theta.cool())
    out = []
    for candidate in artifact.candidates:
        plan = candidate.plan
        out.append(
            MeasuredPlan(
                f"{artifact.workload}:{M}x{N}x{K}",
                artifact.target_sha256,
                workload.digest(),
                episode,
                f"tile:{plan.loop_order}:{plan.tile_m}x{plan.tile_n}x{plan.tile_k}",
                _sha256_of(_canonical(_plan_dict(plan))),
                (
                    MeasuredSample(
                        candidate.wall_ns,
                        candidate.cpu_ns,
                        candidate.minor_faults,
                        candidate.major_faults,
                        candidate.voluntary_context_switches,
                        candidate.involuntary_context_switches,
                    ),
                ),
                candidate.cycles,
                candidate.instructions,
                candidate.cache_misses,
                "unproven",
                "b1-artifact: no attestation recorded",
                candidate.cycles > 0,
                artifact.host_architecture,
                artifact.source_commit,
            )
        )
    return tuple(out)


# --- the measured replay gate ---------------------------------------------------------------


def pooled_median(entries) -> int:
    return int(statistics.median(sample.wall_ns for entry in entries for sample in entry.samples))


def replay_measured(
    corpus: MeasuredCorpus, program: str, target: str, workload: str, candidate, incumbent
) -> ReplayCertificate:
    """The L2 replay gate over measured evidence: for EVERY episode the corpus logs for the
    scope, the candidate's plan against the incumbent's, judged by the pooled median wall
    time of the samples. A regression is an episode the candidate measured slower on. A
    candidate or incumbent without evidence on a logged episode is refused: the corpus
    decides which episodes are replayed, not the promoter. The certificate names the corpus
    head and the number of logged episodes."""
    cand = candidate if isinstance(candidate, str) else candidate.name
    inc = incumbent if isinstance(incumbent, str) else incumbent.name
    episodes = corpus.episodes(program, target, workload)
    if not episodes:
        raise ValueError("the corpus logs no episode for this scope: nothing to replay")
    regressions = 0
    for episode in episodes:
        cand_rows = corpus.lookup(program, target, workload, theta=episode, policy=cand)
        inc_rows = corpus.lookup(program, target, workload, theta=episode, policy=inc)
        for name, rows in ((cand, cand_rows), (inc, inc_rows)):
            if not rows:
                raise ValueError(
                    f"the corpus has no evidence for {name!r} under episode {episode}: a "
                    "candidate is judged on every logged episode or not at all"
                )
        if pooled_median(cand_rows) > pooled_median(inc_rows):
            regressions += 1
    return ReplayCertificate(
        candidate=cand,
        incumbent=inc,
        episodes=len(episodes),
        regressions=regressions,
        corpus=corpus.head,
        logged=len(episodes),
    )


__all__ = [
    "CORPUS_SCHEMA",
    "SCHEMA",
    "TENANCIES",
    "MeasuredCorpus",
    "MeasuredPlan",
    "MeasuredSample",
    "assignment_digest",
    "from_schedule_artifact",
    "measure_plans",
    "pooled_median",
    "read_corpus",
    "replay_measured",
    "scope_key",
    "theta_key",
    "write_corpus",
]
