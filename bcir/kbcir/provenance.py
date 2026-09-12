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
from dataclasses import asdict, dataclass, field

from .._artifact_json import strict_json_loads
from ..model import Module
from .cost import Theta
from .realize import RealizationResult, optimize
from .weights import PERF, Policy

_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211
_MASK = (1 << 64) - 1
_I63 = (1 << 63) - 1  # keep digests within signed-i64 range (MLIR parity)

#: The canonical rendering of an item, memoized for exact ints and strs (the item kinds a
#: module stream is made of) with the field separator appended, so the chain below runs one
#: multiply per byte and nothing else. A bool is NOT an int here (`str(True)` is "True", and
#: the law rail renders `scalable` the same way), so only `type(it) is int` takes the memo.
_ENCODED: dict = {}

#: Diagnostics: how many FULL module digests this process has computed. The G3 gate
#: (`static_memory.digests.2048`) and the identity witnesses read it; it is never an input.
_STATS = {"hash_module": 0}


def _fnv_items(items) -> int:
    """FNV-1a over an already-flat item sequence, byte-identical to the historical
    per-byte chain: for each item, every UTF-8 byte of `str(item)` then the 0xFF field
    separator, each step `h = (h ^ byte) * PRIME (mod 2^64)`, the result i63-masked.

    The reduction mod 2^64 is applied once per item rather than per byte: XOR with a
    byte only touches bits 0..7 and multiplication commutes with reduction, so the low
    64 bits are the same and the accumulator stays small (an item is a few bytes).
    """
    h = _FNV_OFFSET
    prime = _FNV_PRIME
    mask = _MASK
    enc = _ENCODED
    for it in items:
        if type(it) is int or type(it) is str:
            e = enc.get(it)
            if e is None:
                e = enc[it] = str(it).encode("utf-8") + b"\xff"
        else:
            e = str(it).encode("utf-8") + b"\xff"
        for byte in e:
            h = (h ^ byte) * prime
        h &= mask
    return h & _I63


def _fnv(*items) -> int:
    """Deterministic FNV-1a over a canonical flattened item sequence (i63-masked)."""
    return _fnv_items(_flatten(items))


def _flatten(xs):
    for x in xs:
        if isinstance(x, (list, tuple)):
            yield from _flatten(x)
        else:
            yield x


# --- component hashes (the canonical content of each input) ----------------------


def canonical_stream(module: Module) -> tuple:
    """The canonical item sequence of a module -- exactly the items `hash_module` chains,
    in order, produced by ONE iterative walk (G3 / S1-B): the module header, the resources
    by rid (rid, domain, shape extents, layout, align, access, priority, map_gen,
    data_gen), then every phase in declared order (id, sorted deps, then its claims in
    DECLARED order: id, opcode, lane, stride class, count, stride_k, reads, writes, hazard,
    domain, verify, bounds, op, offset, cost_class). It is the R13 canonical content the
    law rail's `hashModuleFromIR` walks field for field, and it is what a `ModuleIdentity`
    is validated against: two modules with the same stream have the same digest."""
    out = [module.name, module.cacheline, module.align]
    ext = out.extend
    for r in sorted(module.resources.values(), key=lambda r: r.rid):
        ext((r.rid, int(r.domain)))
        ext(r.shape)
        ext((r.layout, r.align, r.access, r.priority, r.map_gen, r.data_gen))
    for ph in module.phases:
        ext((ph.phase_id,))
        ext(sorted(ph.deps))
        for c in ph.claims:  # declared order (S0-D); the deps stay a set
            ext((c.id, int(c.opcode), int(c.lane), int(c.stride_class), c.count, c.stride_k))
            ext(c.rd)
            ext(c.wr)
            ext((c.hazard, int(c.domain), c.verify, c.bounds, c.op, c.offset, c.cost_class))
    return tuple(out)


def hash_module(module: Module) -> int:
    """Hash the goal graph G structure: resources, the phase DAG, and the claims in their
    DECLARED order (S0-1 / staged plan S0-D). Declared order is plan-affecting -- two claims
    declared `a, b` and `b, a` plan to 9,216 and 10,496 on the same target -- and the old
    sort by claim id erased it, so the two plans shared one content address. The law rail's
    `hashModuleFromIR` walks the claims in textual order, which is the order the emitter
    writes them: declared.

    This is the RECOMPUTATION primitive -- what a verifier calls at a trust boundary. A
    producer that needs the digest of a module it holds calls `module_identity` (computed
    once per revision); a verifier handed that identity validates it against the module's
    content with `digest_of` (G3 / S1-B)."""
    _STATS["hash_module"] += 1
    return _fnv_items(canonical_stream(module))


def digest_stats() -> dict:
    """A copy of the digest counters (full module digests computed so far)."""
    return dict(_STATS)


class IdentityMismatch(ValueError):
    """A module identity presented for a module whose content it does not describe."""


@dataclass(frozen=True)
class ModuleIdentity:
    """A module's R13 digest, computed once, bound to the content it was computed from.

    `matches(module)` compares the module's canonical stream against the one the digest
    was computed over -- a complete content check at the cost of the walk alone (a few
    milliseconds at 2,048 resources, against tens for the digest) -- so an identity cannot
    survive any mutation, declared or not, and cannot be substituted across modules: a
    different module has a different stream. The `revision` is only the cache key on the
    module; the census (resource, phase and claim counts) is a cheap first witness the
    cache re-checks before trusting the revision."""

    digest: int
    revision: int
    n_resources: int
    n_phases: int
    n_claims: int
    stream: tuple = field(compare=False, repr=False)

    def census_of(self, module: Module) -> bool:
        return (
            self.n_resources == len(module.resources)
            and self.n_phases == len(module.phases)
            and self.n_claims == sum(len(ph.claims) for ph in module.phases)
        )

    def matches(self, module: Module) -> bool:
        """True iff `module`'s canonical content is exactly what this digest describes."""
        return self.census_of(module) and self.stream == canonical_stream(module)


def module_identity(module: Module) -> ModuleIdentity:
    """The module's identity, computed once per revision and cached on the module.

    The cache is dropped by every declared mutation (`Module.touch`, which `add_resource`
    and `add_phase` call) and re-checked against the module's census, so an appended claim,
    phase or resource is never served a stale digest. An in-place edit of a field is
    declared with `touch()`; a verifier does not depend on that declaration, because it
    validates the identity by content (`digest_of`) -- a stale identity is refused or
    recomputed there, never accepted."""
    cached = module._identity
    if (
        isinstance(cached, ModuleIdentity)
        and cached.revision == module.revision
        and cached.census_of(module)
    ):
        return cached
    stream = canonical_stream(module)
    _STATS["hash_module"] += 1
    identity = ModuleIdentity(
        digest=_fnv_items(stream),
        revision=module.revision,
        n_resources=len(module.resources),
        n_phases=len(module.phases),
        n_claims=sum(len(ph.claims) for ph in module.phases),
        stream=stream,
    )
    module._identity = identity
    return identity


def digest_of(module: Module, identity: "ModuleIdentity | None" = None, *, strict: bool = False):
    """The digest a VERIFIER uses: the identity's digest when the identity describes this
    module's current content exactly (validated by the stream, never by the revision),
    otherwise a fresh `hash_module` -- the verifier's right to recompute at a trust
    boundary. With `strict=True` a non-matching identity is refused (`IdentityMismatch`)
    instead of recomputed: mutation invalidates, cross-module substitution is refused."""
    if identity is not None:
        if identity.matches(module):
            return identity.digest
        if strict:
            raise IdentityMismatch(
                "module identity does not describe this module's content "
                "(mutated since it was computed, or minted from a different module)"
            )
    return hash_module(module)


def hash_target(h) -> int:
    """The target fields R13 hashes -- every plan-affecting field of the profile, the
    MEMORY HIERARCHY included (S0-1 / staged plan S0-D).

    Multiplying the DRAM tier's factors by 32 moves a `vector_add(4096)` score from 31,232 to
    983,552; before S0-D this hash did not move with it, and only `replay`'s plan comparison
    (and, since G0, the execution scope) stood between that collision and a wrong answer. The
    tiers now fold in as (name, latency_cyc, bw_factor, lat_factor, capacity) in the
    hierarchy's declared order, and the law rail recomputes them from `target.capability`'s
    `mem_tier_names` / `mem_tier_values` (`hashTargetFromIR`; absent = `MemoryHierarchy.default()`,
    the hierarchy every shipped profile carries, so IR written before the attributes hashes as
    it did). Both rails
    moved in one commit: a cross-rail content address that one rail widens alone is worse than
    the gap.
    """
    mem = getattr(h, "mem", None)
    tiers = tuple(
        (t.name, t.latency_cyc, t.bw_factor, t.lat_factor, t.capacity)
        for t in getattr(mem, "tiers", ())
    )
    return _fnv(
        h.name,
        h.triple,
        h.cacheline,
        h.elem_bytes,
        tuple(sorted(h.lane_widths)),
        h.warp,
        h.scalable,
        h.gather_penalty,
        h.mem_unit,
        h.base_overhead,
        h.thermal_density,
        h.power_density,
        h.per_op_heat,
        h.affinity_domains,
        getattr(h, "mem_channels", 4),
        getattr(h, "cal_gen", 0),
        tiers,
    )


def hash_theta(theta: Theta) -> int:
    return _fnv(
        theta.thermal,
        theta.power,
        theta.mem_pressure,
        theta.contention,
        theta.noise,
        theta.wear,
        theta.utilization,
        theta.voltage,
    )


def hash_policy(policy: Policy) -> int:
    return _fnv(policy.name, tuple(policy.base))


# --- the manifest ----------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceManifest:
    """The commit hash of a plan: component hashes + in-force artifact generations,
    chained into a digest, with the recorded optimal score and plan shape."""

    digest: int
    score: int
    widths: tuple  # ((claim_id, width), ...) -- the recorded plan shape
    artifacts: tuple  # ((name, generation/fingerprint), ...) sorted
    m_module: int
    m_target: int
    m_theta: int
    m_policy: int

    def diff(self, other: "ProvenanceManifest") -> list:
        """Which components changed between two runs -- the cross-generation view."""
        out = []
        for name, a, b in (
            ("module", self.m_module, other.m_module),
            ("target", self.m_target, other.m_target),
            ("theta", self.m_theta, other.m_theta),
            ("policy", self.m_policy, other.m_policy),
            ("artifacts", self.artifacts, other.artifacts),
        ):
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
        d = strict_json_loads(text, "provenance manifest")
        fields = {
            "digest",
            "score",
            "widths",
            "artifacts",
            "m_module",
            "m_target",
            "m_theta",
            "m_policy",
        }
        if not isinstance(d, dict) or set(d) != fields:
            raise ValueError(f"provenance manifest fields must be exactly {sorted(fields)}")

        def i63(key: str) -> int:
            value = d[key]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _I63:
                raise ValueError(f"provenance manifest {key} must be a non-negative i63")
            return value

        digest = i63("digest")
        score = i63("score")
        components = tuple(i63(key) for key in ("m_module", "m_target", "m_theta", "m_policy"))
        if not isinstance(d["widths"], list):
            raise ValueError("provenance manifest widths must be an array")
        widths = []
        for index, pair in enumerate(d["widths"]):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in pair)
            ):
                raise ValueError(f"provenance manifest width[{index}] must be [claim_id, width]")
            cid, width = pair
            if cid < 0 or cid > _I63 or width <= 0 or width > 0xFFFFFFFF:
                raise ValueError(f"provenance manifest width[{index}] is out of range")
            widths.append((cid, width))
        if widths != sorted(widths) or len({cid for cid, _ in widths}) != len(widths):
            raise ValueError("provenance manifest widths must be sorted with unique claim ids")
        if not isinstance(d["artifacts"], list):
            raise ValueError("provenance manifest artifacts must be an array")
        artifacts = []
        for index, pair in enumerate(d["artifacts"]):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"provenance manifest artifact[{index}] must be [name, value]")
            name, value = pair
            if (
                not isinstance(name, str)
                or not name
                or len(name) > 4096
                or any(ord(ch) < 0x20 for ch in name)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or not -(1 << 63) <= value <= _I63
            ):
                raise ValueError(f"provenance manifest artifact[{index}] is invalid")
            artifacts.append((name, value))
        if artifacts != sorted(artifacts) or len({name for name, _ in artifacts}) != len(artifacts):
            raise ValueError("provenance artifacts must be sorted with unique names")
        if _digest(*components, tuple(artifacts)) != digest:
            raise ValueError("provenance manifest digest does not match its components/artifacts")
        return ProvenanceManifest(
            digest=digest,
            score=score,
            widths=tuple(widths),
            artifacts=tuple(artifacts),
            m_module=components[0],
            m_target=components[1],
            m_theta=components[2],
            m_policy=components[3],
        )


def _digest(m_module, m_target, m_theta, m_policy, artifacts) -> int:
    return _fnv(m_module, m_target, m_theta, m_policy, tuple(artifacts))


def _norm_artifacts(artifacts) -> tuple:
    return tuple(sorted((str(n), int(t)) for n, t in artifacts))


def _manifest_from_result(
    module: Module,
    h,
    theta: Theta,
    policy: Policy,
    artifacts,
    result: RealizationResult,
    *,
    fresh: bool = False,
) -> ProvenanceManifest:
    """Assemble a manifest from an *already-computed* plan -- the digest depends
    only on the inputs/artifacts, the score/shape on the result. Factored out so
    the planner is run exactly once per manifest (no recursive re-planning). The
    module digest is the cached identity's (computed once per module revision) unless
    `fresh` -- the verifier's recomputation at a trust boundary (R13, replay)."""
    arts = _norm_artifacts(artifacts)
    mm = hash_module(module) if fresh else module_identity(module).digest
    mt, mth, mp = (hash_target(h), hash_theta(theta), hash_policy(policy))
    widths = tuple(sorted((cid, c.width) for cid, c in result.by_claim().items()))
    return ProvenanceManifest(
        digest=_digest(mm, mt, mth, mp, arts),
        score=result.score,
        widths=widths,
        artifacts=arts,
        m_module=mm,
        m_target=mt,
        m_theta=mth,
        m_policy=mp,
    )


def build_manifest(
    module: Module, h, theta: Theta, policy: Policy = PERF, artifacts=(), *, fresh: bool = False
) -> ProvenanceManifest:
    """Record a plan's provenance: run the optimizer and chain its inputs + the
    in-force decision-rule generations into a manifest (the flight-recorder entry).
    `fresh=True` recomputes the module digest instead of reading the cached identity --
    what R13's `verify_manifest` and `reproduces` do, since they judge an external record."""
    return _manifest_from_result(
        module, h, theta, policy, artifacts, optimize(module, h, theta, policy), fresh=fresh
    )


def manifest_for(
    module: Module,
    h,
    theta: Theta,
    policy: Policy = PERF,
    *,
    table=None,
    gate=None,
    pack=None,
    memory=None,
    extra=(),
) -> ProvenanceManifest:
    """Convenience: assemble the artifact tags from the actual learned objects (the
    calibration table generation, the gate fingerprint, the StreamPack generation
    tags, a frozen memory module's generation + fingerprint) and record the
    manifest. A `memory` module chains the e-graph fixpoint artifact (Phase 21)
    into the plan's commit hash (Phase 20)."""
    arts = list(extra)
    if table is not None:
        arts.append(("cal_fp", _fnv(table.gather_penalty, table.base_overhead, table.mem_unit)))
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


def replay(
    manifest: ProvenanceManifest,
    module: Module,
    h,
    theta: Theta,
    policy: Policy = PERF,
    artifacts=(),
) -> RealizationResult:
    """Reproduce the plan from the manifest. Raises if the supplied inputs do not
    hash to the manifest's digest (you are replaying a different commit). The
    planner runs exactly once -- the digest check reuses the replayed plan instead
    of re-planning (no recursive optimize)."""
    result = optimize(module, h, theta, policy)
    fresh = _manifest_from_result(module, h, theta, policy, artifacts, result, fresh=True)
    if fresh.digest != manifest.digest:
        raise ProvenanceMismatch(
            f"inputs hash to {fresh.digest}, manifest records {manifest.digest} "
            f"(changed: {fresh.diff(manifest)})"
        )
    # The OUTCOME too, not only the inputs' digest. A digest is a claim about the inputs;
    # replaying is a claim about the plan, and the two came apart whenever a plan-affecting
    # input was missing from the hash -- `replay` handed back a plan scoring 1,574,912 for
    # a manifest recording 51,200 and raised nothing. Checking here means a future gap in
    # the input hash surfaces as a loud mismatch rather than as a silently different plan.
    if fresh.score != manifest.score or fresh.widths != manifest.widths:
        raise ProvenanceMismatch(
            f"inputs hash to the manifest's digest but replay produced a DIFFERENT plan: "
            f"score {fresh.score} vs {manifest.score}, widths {fresh.widths} vs "
            f"{manifest.widths}. The digest is missing a plan-affecting input."
        )
    return result


def reproduces(
    manifest: ProvenanceManifest,
    module: Module,
    h,
    theta: Theta,
    policy: Policy = PERF,
    artifacts=(),
) -> bool:
    """True iff the manifest's inputs reproduce its recorded score and plan shape."""
    fresh = build_manifest(module, h, theta, policy, artifacts, fresh=True)
    return (
        fresh.digest == manifest.digest
        and fresh.score == manifest.score
        and fresh.widths == manifest.widths
    )
