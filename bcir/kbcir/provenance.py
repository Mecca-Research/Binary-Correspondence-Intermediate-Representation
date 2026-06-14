"""Provenance manifest + deterministic replay: the version-DAG spine.

A plan is determined by its *inputs* (the goal graph G, the substrate H, the
runtime state Theta, the policy) and by the *decision rules in force* (the
calibration table generation, the policy/portfolio generations, the gate/ranker
fingerprints). A `ProvenanceManifest` is the content hash chaining all of those
-- the **commit hash of a plan**. Two consequences make the constantly-updating
computation DAG debuggable and reproducible:

  * **Manifest equality => identical plan.** The optimizer is deterministic given
    its inputs, so a recorded manifest reproduces its plan exactly (`replay`,
    `reproduces`). This is the "closed branch" made concrete: an immutable plan is
    a committed manifest; replaying from it yields the same result.
  * **Tamper-evidence + the version DAG.** The digest binds the component hashes
    (module / target / theta / policy) and the in-force artifact generations, so
    a changed input or a swapped decision rule changes the digest. `diff` reports
    *which* component moved between two runs -- the debugging view across
    generations ("the table went cal_gen 3 -> 4, so the plan changed").

LEARNING-PLACEMENT LAW (LangRef Sec. 13): every learned artifact is already
frozen + generation-tagged; the manifest is the keystone that chains them into a
single reproducible record, witnessed by R13 (`bcir.kbcir.provenance_manifest`,
`verify.verify_manifest`). Nothing is globally immutable -- but everything is
immutable *within its generation*, and the manifest is what pins that identity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from ..model import Module
from .cost import Theta
from .realize import RealizationResult, optimize
from .weights import PERF, Policy

_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211
_MASK = (1 << 64) - 1
_I63 = (1 << 63) - 1            # keep digests within signed-i64 range (MLIR parity)


def _fnv(*items) -> int:
    """Deterministic FNV-1a over a canonical flattened item sequence (i63-masked)."""
    h = _FNV_OFFSET
    for it in _flatten(items):
        for byte in str(it).encode("utf-8"):
            h = ((h ^ byte) * _FNV_PRIME) & _MASK
        h = ((h ^ 0xFF) * _FNV_PRIME) & _MASK          # field separator
    return h & _I63


def _flatten(xs):
    for x in xs:
        if isinstance(x, (list, tuple)):
            yield from _flatten(x)
        else:
            yield x


# --- component hashes (the canonical content of each input) ----------------------

def hash_module(module: Module) -> int:
    """Hash the goal graph G structure (resources + claims + phase DAG)."""
    res = [(r.rid, int(r.domain), tuple(r.shape), r.layout, r.align, r.access,
            r.priority, r.map_gen, r.data_gen)
           for r in sorted(module.resources.values(), key=lambda r: r.rid)]
    phases = []
    for ph in module.phases:
        claims = [(c.id, int(c.opcode), int(c.lane), int(c.stride_class), c.count,
                   c.stride_k, tuple(c.rd), tuple(c.wr), c.hazard, int(c.domain),
                   c.verify, c.bounds, c.op, c.offset, c.cost_class)
                  for c in sorted(ph.claims, key=lambda c: c.id)]
        phases.append((ph.phase_id, tuple(sorted(ph.deps)), tuple(claims)))
    return _fnv(module.name, module.cacheline, module.align, tuple(res), tuple(phases))


def hash_target(h) -> int:
    return _fnv(h.name, h.triple, h.cacheline, h.elem_bytes, tuple(sorted(h.lane_widths)),
                h.warp, h.scalable, h.gather_penalty, h.mem_unit, h.base_overhead,
                h.thermal_density, h.power_density, h.per_op_heat, h.affinity_domains,
                getattr(h, "mem_channels", 4), getattr(h, "cal_gen", 0))


def hash_theta(theta: Theta) -> int:
    return _fnv(theta.thermal, theta.power, theta.mem_pressure, theta.contention,
                theta.noise, theta.wear, theta.utilization, theta.voltage)


def hash_policy(policy: Policy) -> int:
    return _fnv(policy.name, tuple(policy.base))


# --- the manifest ----------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceManifest:
    """The commit hash of a plan: component hashes + in-force artifact generations,
    chained into a digest, with the recorded optimal score and plan shape."""

    digest: int
    score: int
    widths: tuple              # ((claim_id, width), ...) -- the recorded plan shape
    artifacts: tuple           # ((name, generation/fingerprint), ...) sorted
    m_module: int
    m_target: int
    m_theta: int
    m_policy: int

    def diff(self, other: "ProvenanceManifest") -> list:
        """Which components changed between two runs -- the cross-generation view."""
        out = []
        for name, a, b in (("module", self.m_module, other.m_module),
                           ("target", self.m_target, other.m_target),
                           ("theta", self.m_theta, other.m_theta),
                           ("policy", self.m_policy, other.m_policy),
                           ("artifacts", self.artifacts, other.artifacts)):
            if a != b:
                out.append(name)
        return out

    def to_json(self) -> str:
        d = asdict(self)
        d["widths"] = [list(w) for w in self.widths]
        d["artifacts"] = [list(a) for a in self.artifacts]
        return json.dumps(d, indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "ProvenanceManifest":
        d = json.loads(text)
        return ProvenanceManifest(
            digest=d["digest"], score=d["score"],
            widths=tuple(tuple(w) for w in d["widths"]),
            artifacts=tuple(tuple(a) for a in d["artifacts"]),
            m_module=d["m_module"], m_target=d["m_target"],
            m_theta=d["m_theta"], m_policy=d["m_policy"])


def _digest(m_module, m_target, m_theta, m_policy, artifacts) -> int:
    return _fnv(m_module, m_target, m_theta, m_policy, tuple(artifacts))


def _norm_artifacts(artifacts) -> tuple:
    return tuple(sorted((str(n), int(t)) for n, t in artifacts))


def _manifest_from_result(module: Module, h, theta: Theta, policy: Policy,
                          artifacts, result: RealizationResult) -> ProvenanceManifest:
    """Assemble a manifest from an *already-computed* plan -- the digest depends
    only on the inputs/artifacts, the score/shape on the result. Factored out so
    the planner is run exactly once per manifest (no recursive re-planning)."""
    arts = _norm_artifacts(artifacts)
    mm, mt, mth, mp = (hash_module(module), hash_target(h), hash_theta(theta),
                       hash_policy(policy))
    widths = tuple(sorted((cid, c.width) for cid, c in result.by_claim().items()))
    return ProvenanceManifest(digest=_digest(mm, mt, mth, mp, arts), score=result.score,
                              widths=widths, artifacts=arts,
                              m_module=mm, m_target=mt, m_theta=mth, m_policy=mp)


def build_manifest(module: Module, h, theta: Theta, policy: Policy = PERF,
                   artifacts=()) -> ProvenanceManifest:
    """Record a plan's provenance: run the optimizer and chain its inputs + the
    in-force decision-rule generations into a manifest (the flight-recorder entry)."""
    return _manifest_from_result(module, h, theta, policy, artifacts,
                                 optimize(module, h, theta, policy))


def manifest_for(module: Module, h, theta: Theta, policy: Policy = PERF, *,
                 table=None, gate=None, pack=None, memory=None,
                 extra=()) -> ProvenanceManifest:
    """Convenience: assemble the artifact tags from the actual learned objects (the
    calibration table generation, the gate fingerprint, the StreamPack generation
    tags, a frozen memory module's generation + fingerprint) and record the
    manifest. A `memory` module chains the e-graph fixpoint artifact (Phase 21)
    into the plan's commit hash (Phase 20)."""
    arts = list(extra)
    if table is not None:
        arts.append(("cal_fp", _fnv(table.gather_penalty, table.base_overhead,
                                    table.mem_unit)))
        arts.append(("cal_gen", table.cal_gen))
    if gate is not None:
        arts.append(("gate", gate.fingerprint))
    if pack is not None:
        arts.append(("topo_gen", pack.topo_gen))
        arts.append(("map_gen", pack.map_gen))
        arts.append(("data_gen", pack.data_gen))
    if memory is not None:
        arts.append(("mem_gen", memory.generation))
        arts.append(("mem_fp", memory.fingerprint))
    return build_manifest(module, h, theta, policy, arts)


# --- replay (deterministic reproduction from the manifest) -----------------------

class ProvenanceMismatch(Exception):
    """The inputs/artifacts do not match the manifest's recorded digest."""


def replay(manifest: ProvenanceManifest, module: Module, h, theta: Theta,
           policy: Policy = PERF, artifacts=()) -> RealizationResult:
    """Reproduce the plan from the manifest. Raises if the supplied inputs do not
    hash to the manifest's digest (you are replaying a different commit). The
    planner runs exactly once -- the digest check reuses the replayed plan instead
    of re-planning (no recursive optimize)."""
    result = optimize(module, h, theta, policy)
    fresh = _manifest_from_result(module, h, theta, policy, artifacts, result)
    if fresh.digest != manifest.digest:
        raise ProvenanceMismatch(
            f"inputs hash to {fresh.digest}, manifest records {manifest.digest} "
            f"(changed: {fresh.diff(manifest)})")
    return result


def reproduces(manifest: ProvenanceManifest, module: Module, h, theta: Theta,
               policy: Policy = PERF, artifacts=()) -> bool:
    """True iff the manifest's inputs reproduce its recorded score and plan shape."""
    fresh = build_manifest(module, h, theta, policy, artifacts)
    return (fresh.digest == manifest.digest and fresh.score == manifest.score
            and fresh.widths == manifest.widths)
