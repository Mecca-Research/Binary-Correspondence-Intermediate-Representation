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

from .._artifact_json import strict_json_loads
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
        # The 0.4b STABLE-SCHEMA envelope: kind + schema version wrap the payload, so a record
        # is self-describing on disk and a decoder can upgrade (or refuse) by version alone.
        return json.dumps({"kind": RECORD_KIND, "schema": SCHEMA_VERSION, "record": d},
                          indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "DecisionRecord":
        """Decode ANY known schema revision (the 0.4b decode/upgrade path): a bare legacy
        payload (v1, pre-envelope) is upgraded in place via the `_UPGRADES` chain; an
        envelope at the current version decodes directly; an UNKNOWN (newer) version fails
        LOUDLY -- a certificate must never be silently misread."""
        d = strict_json_loads(text, "decision record")
        if not isinstance(d, dict):
            raise ValueError("decision record JSON must be an object")
        if "schema" not in d:                          # v1: the bare, unversioned payload
            version, payload = 1, d
        else:
            if set(d) != {"kind", "schema", "record"}:
                raise ValueError("decision-record envelope has unknown or missing fields")
            if d["kind"] != RECORD_KIND:
                raise ValueError(f"not a {RECORD_KIND} document (kind={d.get('kind')!r})")
            version = d["schema"]
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                raise ValueError("decision-record schema must be a positive integer")
            payload = d["record"]
        if version > SCHEMA_VERSION:
            raise ValueError(f"decision-record schema v{version} is newer than this build's "
                             f"v{SCHEMA_VERSION}; upgrade BCIR to re-check this record")
        while version < SCHEMA_VERSION:                # chain upgrades one revision at a time
            upgrade = _UPGRADES.get(version)
            if upgrade is None:
                raise ValueError(f"decision-record schema v{version} is not supported")
            payload = upgrade(payload)
            version += 1
        fields = {"module_name", "target", "theta", "policy", "digest", "total_score",
                  "decisions", "certificates"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError(f"decision-record fields must be exactly {sorted(fields)}")

        def bounded_string(value, label):
            if (not isinstance(value, str) or not value or len(value) > 4096 or
                    any(ord(ch) < 0x20 for ch in value)):
                raise ValueError(f"{label} must be a bounded, non-control string")
            return value

        def nonnegative_i63(value, label):
            if (isinstance(value, bool) or not isinstance(value, int) or
                    not 0 <= value <= (1 << 63) - 1):
                raise ValueError(f"{label} must be a non-negative i63")
            return value

        labels = tuple(bounded_string(payload[key], f"decision-record {key}")
                       for key in ("module_name", "target", "theta", "policy"))
        digest = nonnegative_i63(payload["digest"], "decision-record digest")
        total_score = nonnegative_i63(payload["total_score"],
                                      "decision-record total_score")

        raw_decisions = payload["decisions"]
        if not isinstance(raw_decisions, list) or len(raw_decisions) > 65536:
            raise ValueError("decision-record decisions must be a bounded array")
        decisions = []
        seen_claims = set()
        decision_fields = {"claim_id", "op", "chosen", "width", "score", "candidates"}
        for index, dc in enumerate(raw_decisions):
            if not isinstance(dc, dict) or set(dc) != decision_fields:
                raise ValueError(f"decision[{index}] has unknown or missing fields")
            claim_id = nonnegative_i63(dc["claim_id"], f"decision[{index}].claim_id")
            if claim_id in seen_claims:
                raise ValueError(f"duplicate decision claim_id {claim_id}")
            seen_claims.add(claim_id)
            op = bounded_string(dc["op"], f"decision[{index}].op")
            chosen = bounded_string(dc["chosen"], f"decision[{index}].chosen")
            width = dc["width"]
            if (isinstance(width, bool) or not isinstance(width, int) or
                    not 1 <= width <= 0xFFFFFFFF):
                raise ValueError(f"decision[{index}].width must be a positive u32")
            score = nonnegative_i63(dc["score"], f"decision[{index}].score")
            raw_candidates = dc["candidates"]
            if (not isinstance(raw_candidates, list) or not raw_candidates or
                    len(raw_candidates) > 4096):
                raise ValueError(f"decision[{index}].candidates must be a bounded array")
            candidates = []
            seen_names = set()
            for candidate_index, candidate in enumerate(raw_candidates):
                if not isinstance(candidate, list) or len(candidate) != 2:
                    raise ValueError(
                        f"decision[{index}].candidates[{candidate_index}] must be [name, score]")
                name = bounded_string(candidate[0],
                                      f"decision[{index}].candidates[{candidate_index}].name")
                candidate_score = nonnegative_i63(
                    candidate[1], f"decision[{index}].candidates[{candidate_index}].score")
                if name in seen_names:
                    raise ValueError(f"decision[{index}] has duplicate candidate {name!r}")
                seen_names.add(name)
                candidates.append((name, candidate_score))
            if candidates != sorted(candidates):
                raise ValueError(f"decision[{index}] candidates must be sorted by name")
            chosen_scores = [candidate_score for name, candidate_score in candidates
                             if name == chosen]
            if chosen_scores != [score]:
                raise ValueError(
                    f"decision[{index}] chosen candidate must exist and match its score")
            decisions.append(ClaimDecision(claim_id, op, chosen, width, score,
                                           tuple(candidates)))
        raw_certs = payload["certificates"]
        if not isinstance(raw_certs, list) or len(raw_certs) > 65536:
            raise ValueError("decision-record certificates must be a bounded array")
        certs = []
        cert_fields = {"kind", "claim_ids", "detail", "gain", "searched"}
        for index, cert in enumerate(raw_certs):
            if not isinstance(cert, dict) or set(cert) != cert_fields:
                raise ValueError(f"certificate[{index}] has unknown or missing fields")
            kind = bounded_string(cert["kind"], f"certificate[{index}].kind")
            if kind != "bundle":
                raise ValueError(f"certificate[{index}] has unsupported kind {kind!r}")
            raw_ids = cert["claim_ids"]
            if (not isinstance(raw_ids, list) or len(raw_ids) < 2 or
                    len(raw_ids) > 65536):
                raise ValueError(f"certificate[{index}].claim_ids must be a bounded array")
            claim_ids = tuple(nonnegative_i63(value,
                                               f"certificate[{index}].claim_ids")
                              for value in raw_ids)
            if len(set(claim_ids)) != len(claim_ids):
                raise ValueError(f"certificate[{index}] has duplicate claim ids")
            if any(claim_id not in seen_claims for claim_id in claim_ids):
                raise ValueError(f"certificate[{index}] references an unknown claim id")
            detail = bounded_string(cert["detail"], f"certificate[{index}].detail")
            gain = nonnegative_i63(cert["gain"], f"certificate[{index}].gain")
            searched = nonnegative_i63(cert["searched"], f"certificate[{index}].searched")
            if searched == 0:
                raise ValueError(f"certificate[{index}].searched must be positive")
            certs.append(RewriteCertificate(kind, claim_ids, detail, gain, searched))

        return DecisionRecord(labels[0], labels[1], labels[2], labels[3], digest,
                              total_score, tuple(decisions), tuple(certs))


RECORD_KIND = "bcir.decision_record"
SCHEMA_VERSION = 2


def _upgrade_1_to_2(payload: dict) -> dict:
    """v1 -> v2: the payload FIELDS are unchanged -- v2 only added the self-describing
    envelope (kind + schema) around them. Every future revision adds its own upgrader here
    and bumps SCHEMA_VERSION; from_json chains them, so any old record stays decodable."""
    return payload


_UPGRADES = {1: _upgrade_1_to_2}


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
    # ``target`` is a user-facing label and may be a registry alias (for example
    # ``x86_avx512`` for profile name ``x86-64-avx512``).  The target contents are
    # bound by the provenance digest; the CLI separately checks the accepted alias.
    fresh = explain(module, h, theta, policy, target_name=record.target, joint=joint)
    mismatches: list[str] = []
    for field in ("module_name", "theta", "policy"):
        actual = getattr(fresh, field)
        recorded = getattr(record, field)
        if actual != recorded:
            mismatches.append(f"{field} {actual!r} != recorded {recorded!r}")
    if fresh.digest != record.digest:
        mismatches.append(f"provenance digest {fresh.digest} != recorded {record.digest}")
    if fresh.total_score != record.total_score:
        mismatches.append(f"total score {fresh.total_score} != recorded {record.total_score}")
    fresh_by_id = {d.claim_id: d for d in fresh.decisions}
    recorded_by_id = {d.claim_id: d for d in record.decisions}
    for claim_id in sorted(set(fresh_by_id) | set(recorded_by_id)):
        fd = fresh_by_id.get(claim_id)
        rd = recorded_by_id.get(claim_id)
        if fd is None:
            mismatches.append(f"recorded claim {claim_id} absent on replay")
        elif rd is None:
            mismatches.append(f"replay claim {claim_id} absent from record")
        elif fd != rd:
            mismatches.append(f"claim {claim_id}: replay {fd!r} != recorded {rd!r}")
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
