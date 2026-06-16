"""Proof-carrying optimization records: explain / replay / reduce.

R13 (`provenance`) records a plan's *commit hash* -- the inputs + artifact generations
chained into a digest, with the recorded optimal score and plan shape. This layer adds the
*rationale* on top: a replayable `DecisionRecord` that, for every claim, lists the
candidate realizations the optimizer weighed, the one it chose, and the per-candidate
scores that justify it -- plus any **rewrite certificates** (the bundle joint reorders,
`kbcir.bundle`). Three operations a deployed plan can carry and a reviewer can check:

  * ``explain(module, H, Theta, policy)`` -> ``DecisionRecord`` (+ ``explain_text``)
  * ``replay(record, module, H, Theta, policy)`` -> ``ReplayResult`` -- reproduces the
    record bit-for-bit from the same inputs (the R13 digest gate + the per-claim decisions),
    or reports exactly what diverged.
  * ``reduce(module, predicate)`` -> a minimal `Module` still satisfying `predicate`
    (the debugging reducer; `bcir-reduce`).

Deterministic and integer end to end; the record round-trips through JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from ..model import Module
from .cost import HProfile, Theta
from .weights import PERF, Policy


@dataclass(frozen=True)
class ClaimDecision:
    """Why one claim got its realization: the candidates weighed + the chosen one."""

    claim_id: int
    op: str
    chosen: str
    width: int
    score: int
    candidates: tuple        # ((candidate_name, candidate_score), ...) ascending name


@dataclass(frozen=True)
class RewriteCertificate:
    """A serializable record of one joint reorder (a `kbcir.bundle.BundleCertificate`)."""

    kind: str                # "bundle"
    claim_ids: tuple
    detail: str
    gain: int
    searched: int


@dataclass(frozen=True)
class DecisionRecord:
    """The full proof-carrying record of a plan: provenance digest + per-claim rationale
    + rewrite certificates. Serializes to JSON; `replay` reproduces it from the inputs."""

    module_name: str
    target: str
    theta: str
    policy: str
    digest: int
    total_score: int
    decisions: tuple         # (ClaimDecision, ...)
    certificates: tuple      # (RewriteCertificate, ...)

    def to_json(self) -> str:
        d = asdict(self)
        d["decisions"] = [
            {**asdict(dc), "candidates": [list(c) for c in dc.candidates]}
            for dc in self.decisions]
        d["certificates"] = [
            {**asdict(c), "claim_ids": list(c.claim_ids)} for c in self.certificates]
        return json.dumps(d, indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "DecisionRecord":
        d = json.loads(text)
        decisions = tuple(
            ClaimDecision(dc["claim_id"], dc["op"], dc["chosen"], dc["width"], dc["score"],
                          tuple(tuple(c) for c in dc["candidates"]))
            for dc in d["decisions"])
        certs = tuple(
            RewriteCertificate(c["kind"], tuple(c["claim_ids"]), c["detail"], c["gain"],
                               c["searched"])
            for c in d["certificates"])
        return DecisionRecord(d["module_name"], d["target"], d["theta"], d["policy"],
                              d["digest"], d["total_score"], decisions, certs)


def _theta_label(theta: Theta) -> str:
    for name, t in (("cool", Theta.cool()), ("hot", Theta.hot()), ("mem_bound", Theta.mem_bound())):
        if t == theta:
            return name
    return f"theta(thermal={theta.thermal},power={theta.power})"


def explain(module: Module, h: HProfile, theta: Theta, policy: Policy = PERF, *,
            target_name: str = "", joint: bool = False) -> DecisionRecord:
    """Build the proof-carrying record for `module`'s plan: the provenance digest, the
    per-claim decision (candidates weighed + chosen + per-candidate scores), and -- with
    `joint=True` -- the bundle rewrite certificates. `target_name` labels the record (the
    digest, not the label, is what `replay` checks)."""
    from ..lower.mlir import plan_view          # the IR-level candidate view (shared rail)
    from .bundle import optimize_bundled
    from .provenance import build_manifest

    pv = plan_view(module, h, theta, policy)
    weights = pv.weights
    decisions = []
    for cv in pv.claims:
        cands = tuple(sorted(
            (p.name, sum(c * w for c, w in zip(p.cost, weights))) for p in cv.paths))
        decisions.append(ClaimDecision(
            claim_id=cv.claim_id, op=cv.op, chosen=cv.selected, width=cv.width,
            score=cv.score, candidates=cands))

    certs: list[RewriteCertificate] = []
    if joint:
        _res, bcerts = optimize_bundled(module, h, theta, policy)
        for bc in bcerts:
            certs.append(RewriteCertificate(
                kind="bundle", claim_ids=bc.bundle.claim_ids,
                detail=f"reorder around shared rid {bc.bundle.shared_rid} -> {bc.order}",
                gain=bc.gain, searched=bc.searched))

    manifest = build_manifest(module, h, theta, policy)
    return DecisionRecord(
        module_name=module.name, target=target_name or getattr(h, "name", "?"),
        theta=_theta_label(theta), policy=policy.name, digest=manifest.digest,
        total_score=pv.total_score, decisions=tuple(decisions), certificates=tuple(certs))


@dataclass(frozen=True)
class ReplayResult:
    reproduced: bool
    mismatches: tuple

    def __bool__(self) -> bool:
        return self.reproduced


def replay(record: DecisionRecord, module: Module, h: HProfile, theta: Theta,
           policy: Policy = PERF, *, joint: bool = False) -> ReplayResult:
    """Reproduce `record` from the supplied inputs and diff. Reproduction holds iff the
    R13 provenance digest matches (same commit) AND every per-claim decision + the total
    score + the rewrite certificates are identical -- the proof the deployed plan is
    bit-for-bit replayable. Returns the (reproduced, mismatches) verdict."""
    fresh = explain(module, h, theta, policy, target_name=record.target, joint=joint)
    mismatches: list[str] = []
    if fresh.digest != record.digest:
        mismatches.append(f"provenance digest {fresh.digest} != recorded {record.digest}")
    if fresh.total_score != record.total_score:
        mismatches.append(f"total score {fresh.total_score} != recorded {record.total_score}")
    fresh_by_id = {d.claim_id: d for d in fresh.decisions}
    for rd in record.decisions:
        fd = fresh_by_id.get(rd.claim_id)
        if fd is None:
            mismatches.append(f"claim {rd.claim_id} absent on replay")
        elif (fd.chosen, fd.score, fd.width) != (rd.chosen, rd.score, rd.width):
            mismatches.append(
                f"claim {rd.claim_id}: replay ({fd.chosen}/{fd.score}/w{fd.width}) != "
                f"recorded ({rd.chosen}/{rd.score}/w{rd.width})")
    if fresh.certificates != record.certificates:
        mismatches.append("rewrite certificates diverged")
    return ReplayResult(reproduced=not mismatches, mismatches=tuple(mismatches))


def reduce(module: Module, predicate, max_passes: int = 60) -> Module:
    """Minimal witness: greedily shrink `module` (drop phases/claims, shrink counts) while
    `predicate(module)` stays True and the module stays legal+plannable -- `bcir-reduce`,
    built on the differential shrinker. Returns the minimal module satisfying `predicate`."""
    from .differential import _legal, shrink

    def fails(m: Module) -> bool:
        return _legal(m) and predicate(m)

    if not fails(module):
        raise ValueError("predicate does not hold on the input module")
    return shrink(module, fails, max_passes=max_passes)


def explain_text(record: DecisionRecord) -> str:
    """A human-readable rendering of the decision record (for `bcir-explain`)."""
    lines = [
        f"plan {record.module_name} on {record.target} ({record.theta}, {record.policy})",
        f"  provenance digest {record.digest}, total score {record.total_score}",
    ]
    for d in record.decisions:
        alts = ", ".join(f"{n}={s}" for n, s in d.candidates)
        lines.append(f"  claim {d.claim_id} [{d.op}] -> {d.chosen} (w{d.width}, score "
                     f"{d.score}); weighed: {alts}")
    for c in record.certificates:
        lines.append(f"  rewrite[{c.kind}] {list(c.claim_ids)}: {c.detail} "
                     f"(gain {c.gain}, searched {c.searched})")
    return "\n".join(lines)
