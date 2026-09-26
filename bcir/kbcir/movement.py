"""Data movement as a first-class transformation (GEM+ roadmap G8, staged plan S5-C).

The optimizer chooses, for every claim, the bank its operands are read and written in -- its
*site* -- and transforms the goal graph M into M' = M plus explicit `mem.move.*` claims over
per-(resource, bank, episode) *copy* resources: D-R2's cast, one move claim per hop, the copy a
consumer reads made fresh first. M' is an ordinary module, so nothing downstream learns a second
language. `optimize` realizes it (a move claim is priced by the same K_BCIR arithmetic every rail
shares), `schedule_plan` places it (a move is a bandwidth-class claim: it overlaps compute
through the hazard DAG and contends for the bandwidth knee, the backpressure law the dispatch
already holds), `plan_static_memory` gives every copy its own address and interval (a resource
resident in two banks, or evicted and reloaded, is two copies), and the ExecutionPlan's `moves`
family carries one `MovementEdge` per move claim. Movement and compute are therefore chosen
**jointly**: a site assignment is priced by the realization and the placement of the module it
produces, moves included -- never by pricing transfers after placement.

What makes it safe is that the transform is **correctness-neutral** and checked, not trusted
(`verify_movement`, laws MV1-MV11 below): every original claim reads, through its copy, exactly
the version of each operand it reads in M; every mutable resource's home holds M's final version
at the end, through a generation-checked writeback; and the Semantic Swap family is explicit --

* an **immutable** resource (no claim writes it) may be dropped and reloaded freely;
* a **recomputable** resource (a temporary, not live-out) needs no writeback and may be
  **rematerialized** only from a replay-certified producer: a deterministic claim that reads,
  at the remat site, exactly the versions it read when it first ran (the certificate digest is
  re-derived by the verifier);
* a **mutable** resource (the default: nothing is assumed) is written back home, the edge
  carrying the version it lands;
* an **approximation** (a compressed transfer through a declared lossy codec) needs an accuracy
  certificate: every claim that reads the compressed copy declares a tolerance R17's
  `meets_tolerance` admits with the codec's quantization step;
* **deadlines and backpressure** stop an "optimal" plan that thrashes storage: a declared
  makespan budget is a hard bound, a move is a bandwidth-class claim the knee bounds, and every
  reload of a version a bank already held must be FORCED by that bank's capacity.

Laws (Python rail; the C twin holds the byte-level subset, `bcir_ep_check_moves`):

  MV1  declared   every M' resource is a logical resource of M or a copy of one (same shape,
                  element size, layout) in a declared bank of the copy's domain;
  MV2  preserved  M' holds every claim of M -- same id, phase, opcode, op, count and fields --
                  with its operands remapped to copies of ONE site (D-R2 clean);
  MV3  explicit   every other M' claim is a move (one read, one write, copies of one resource,
                  two banks joined by a declared link) or a remat (a clone of the producer);
  MV4  versions   every original claim reads, through its copy, the version it reads in M;
  MV5  writeback  every mutable resource's home holds M's final version at the end, landed by a
                  coherence=writeback edge that carries that version;
  MV6  classes    an immutable resource is written by no claim; a writeback is never compressed;
  MV7  remat      a remat's producer is replay-certified and its certificate re-derives;
  MV8  accuracy   a compressed copy is read only by original claims whose tolerance admits the
                  codec, and its certificate re-derives;
  MV9  capacity   the copies fit every bank's allocatable bytes, and every reload is forced;
  MV10 deadline   the placement's makespan is within the declared budget;
  MV11 binding    the plan carries one edge per move/remat claim, field for field, bound to M's
                  digest and the spec's; each window (after: the claim that last wrote the
                  source; before: the first claim that reads the destination) is ordered by the
                  placement.

Scope, declared: sites are banks; a claim that touches an isolated domain (MMIO) is pinned to
its home and its isolated operands never move; a claim's version is the whole resource's (a
partial write -- `count` below the resource's -- first makes its copy fresh); link bandwidths
and latencies select routes, and the K_BCIR price of a move is the realization's (the tier of
its source -- a far move priced like a near one is the recorded D-R3 finding, a four-rail
pricing change, not this slice).
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
from dataclasses import dataclass, field, replace

from ..model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from ..model.lanes import ISOLATED_DOMAINS
from ..model.opcodes import ATOMIC_OPCODES

#: The Semantic Swap classes; `mutable` is the default (nothing is assumed of a resource).
CLASSES = ("immutable", "recomputable", "mutable")
MOVE_PREFIX = "mem.move."
#: Opcodes a replay-certified producer may not have: control and synchronization.
_CONTROL = frozenset(
    {
        Opcode.NOP,
        Opcode.BARRIER,
        Opcode.PHASE_ENTER,
        Opcode.PHASE_LEAVE,
        Opcode.PROV_NOTE,
        Opcode.GEM_DISPATCH,
    }
)
_U63 = (1 << 63) - 1


def _u64_digest(value) -> int:
    """A nonzero u64 of the canonical JSON of `value` (the first eight SHA-256 bytes, little
    endian): the certificate and binding digests the plan carries. Zero is reserved for
    'absent' on the wire, so a zero digest is spelled 1."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    raw = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    return raw or 1


def source_digest(module: Module) -> int:
    """The v3 `source_hash` of the goal graph a movement transform started from: its canonical
    digest (`provenance.digest_of`), with zero -- reserved for 'absent' on the wire -- spelled
    1, as `_u64_digest` spells it. The producer and MV11 both read it here."""
    from .provenance import digest_of

    return digest_of(module) or 1


# --- the spec -------------------------------------------------------------------------------


@dataclass(frozen=True)
class MovementSpec:
    """What the planner may assume about the resources and the hardware, and nothing else.

    `hardware` is a `HardwareEnvelope` (duck-typed: `banks`, `links`, `bank(name)`,
    `link(src, dst)`, `digest`): banks carry their domain and allocatable bytes, links are
    DIRECTED. `homes` binds a logical resource to its home bank (default: the one bank of the
    resource's domain); `classes` gives the Semantic Swap class (default `mutable`); `codecs`
    names the lossy transfer codec a resource admits, in bits per element; `sites` restricts
    the banks a claim may run in; `deadline` is a makespan budget in plan ticks (0 = none);
    `remat` lists the recomputable resources the planner may rematerialize; `max_hops` bounds a
    staged route."""

    hardware: object
    homes: tuple[tuple[int, str], ...] = ()
    classes: tuple[tuple[int, str], ...] = ()
    codecs: tuple[tuple[int, int], ...] = ()
    sites: tuple[tuple[int, tuple[str, ...]], ...] = ()
    deadline: int = 0
    remat: tuple[int, ...] = ()
    max_hops: int = 3

    def __post_init__(self) -> None:
        for name in ("homes", "classes", "codecs", "sites", "remat"):
            object.__setattr__(self, name, tuple(sorted(getattr(self, name))))
        for rid, cls in self.classes:
            if cls not in CLASSES:
                raise ValueError(f"resource {rid} has unknown movement class {cls!r}")
        for rid, bits in self.codecs:
            if not isinstance(bits, int) or isinstance(bits, bool) or not 1 <= bits <= 31:
                raise ValueError(f"resource {rid} codec must be 1..31 bits, got {bits!r}")
        if not isinstance(self.deadline, int) or self.deadline < 0:
            raise ValueError("movement deadline must be a non-negative integer")
        if not isinstance(self.max_hops, int) or not 1 <= self.max_hops <= 16:
            raise ValueError("movement max_hops must be 1..16")
        banks = {b.name for b in self.hardware.banks}
        for rid, bank in self.homes:
            if bank not in banks:
                raise ValueError(f"resource {rid} home names unknown bank {bank!r}")
        for cid, allowed in self.sites:
            if not allowed or any(b not in banks for b in allowed):
                raise ValueError(f"claim {cid} site restriction names an unknown bank")

    # the lookups the transform and the verifier share
    def bank_domain(self, bank: str) -> Domain:
        return Domain[self.hardware.bank(bank).domain]

    def home_of(self, module: Module, rid: int) -> str:
        homes = dict(self.homes)
        if rid in homes:
            return homes[rid]
        res = module.resources[rid]
        banks = [b.name for b in self.hardware.banks if b.domain == res.domain.name]
        if len(banks) != 1:
            raise ValueError(
                f"resource {rid} ({res.domain.name}) needs a declared home: {len(banks)} banks "
                f"of its domain"
            )
        return banks[0]

    def class_of(self, rid: int) -> str:
        return dict(self.classes).get(rid, "mutable")

    def codec_of(self, rid: int) -> int:
        return dict(self.codecs).get(rid, 0)

    def sites_for(self, claim_id: int) -> tuple[str, ...] | None:
        return dict(self.sites).get(claim_id)

    def compute_banks(self) -> tuple[str, ...]:
        """The banks a claim may run in: every bank of a memory domain (an isolated domain is
        a device register file, never a site)."""
        return tuple(
            sorted(b.name for b in self.hardware.banks if Domain[b.domain] not in ISOLATED_DOMAINS)
        )

    def digest(self) -> int:
        """The spec's u64 binding digest (the v3 plan's `spec_hash`)."""
        return _u64_digest(
            {
                "hardware": str(self.hardware.digest),
                "homes": [list(x) for x in self.homes],
                "classes": [list(x) for x in self.classes],
                "codecs": [list(x) for x in self.codecs],
                "sites": [[cid, list(b)] for cid, b in self.sites],
                "deadline": self.deadline,
                "remat": list(self.remat),
                "max_hops": self.max_hops,
            }
        )

    def route(self, src: str, dst: str) -> tuple[str, ...] | None:
        """The route a move takes, as the banks it visits: the fewest hops over declared
        directed links (at most `max_hops`), ties broken by the summed link latency and then
        the bank names -- deterministic, and the same the verifier re-derives. None when no
        route exists."""
        if src == dst:
            return (src,)
        best: dict[str, tuple] = {src: (0, 0, (src,))}
        heap = [(0, 0, (src,))]
        while heap:
            hops, lat, path = heapq.heappop(heap)
            here = path[-1]
            if here == dst:
                return path
            if hops >= self.max_hops or best.get(here, (hops, lat, path)) < (hops, lat, path):
                continue
            for link in self.hardware.links:
                if link.source != here or link.destination in path:
                    continue
                key = (hops + 1, lat + int(link.latency_ns), path + (link.destination,))
                if link.destination not in best or key < best[link.destination]:
                    best[link.destination] = key
                    heapq.heappush(heap, key)
        return None


# --- orders and versions --------------------------------------------------------------------


def execution_order(module: Module) -> list[tuple[int, Claim]]:
    """(phase id, claim) in the canonical program order: phases in topological order, and inside
    a phase ascending claim id -- the order `schedule.phase_hazards` builds the hazard DAG in and
    the GEM executor dispatches by. The transform walks M in it and the verifier replays both
    modules in it; the race law (MV4) makes the replay independent of the linearization."""
    from ..model import topological_phase_ids

    pmap = module.phase_map()
    return [
        (pid, c)
        for pid in topological_phase_ids(module)
        for c in sorted(pmap[pid].claims, key=lambda c: c.id)
    ]


def _isolated(module: Module, rid: int) -> bool:
    res = module.resources.get(rid)
    return res is not None and res.domain in ISOLATED_DOMAINS


def _declared_domain(claim: Claim, resources: dict, site_domain: Domain) -> Domain:
    """The domain a remapped claim declares: its own while it still touches that domain (R3:
    a claim declares a domain it touches), else its site's -- so a claim that did not move keeps
    every field, and one that did declares where it now runs."""
    touched = {resources[r].domain for r in (*claim.rd, *claim.wr) if r in resources}
    if not touched or claim.domain in touched:
        return claim.domain
    return site_domain


def _whole_write(claim: Claim, res: Resource) -> bool:
    """A write that replaces the whole resource (so the old version need not be present)."""
    return claim.count >= res.count and not claim.dynamic


@dataclass(frozen=True)
class Versions:
    """M's version flow: the version of each operand each claim reads (`reads[(cid, rid)]`),
    the version it writes (`writes[(cid, rid)]`), each claim's read versions in order
    (`inputs[cid]`), the claim that produced each version (`producer[(rid, v)]`) and each
    resource's final version (`final[rid]`). Version 0 is the resource's content before the
    module runs."""

    reads: dict
    writes: dict
    inputs: dict
    producer: dict
    final: dict


def versions_of(module: Module) -> Versions:
    """Replay M in `execution_order`, counting writes per resource."""
    ver = {rid: 0 for rid in module.resources}
    reads, writes, inputs, producer = {}, {}, {}, {}
    for _pid, c in execution_order(module):
        inputs[c.id] = tuple((rid, ver.get(rid, 0)) for rid in c.rd)
        for rid in c.rd:
            reads[(c.id, rid)] = ver.get(rid, 0)
        for rid in c.wr:
            reads.setdefault((c.id, rid), ver.get(rid, 0))  # the pre-write version
            ver[rid] = ver.get(rid, 0) + 1
            writes[(c.id, rid)] = ver[rid]
            producer[(rid, ver[rid])] = c.id
    return Versions(reads, writes, inputs, producer, dict(ver))


def replay_safe(module: Module, claim: Claim) -> str:
    """Why `claim` cannot be replayed to rematerialize what it wrote ('' when it can): a
    deterministic, unfenced, non-atomic compute claim over memory-domain operands that does not
    read what it writes (an in-place update's old version is gone once it has run)."""
    if claim.opcode in _CONTROL:
        return "a control opcode"
    if claim.opcode in ATOMIC_OPCODES or claim.hazard != "unique":
        return "an atomic or ordered claim"
    if claim.volatile or claim.domain in ISOLATED_DOMAINS:
        return "a device access"
    if any(_isolated(module, r) for r in (*claim.rd, *claim.wr)):
        return "a device operand"
    if set(claim.rd) & set(claim.wr):
        return "an in-place update"
    if claim.op.startswith(MOVE_PREFIX):
        return "a move"
    if claim.dynamic:
        return "a dynamic count"
    return ""


def replay_certificate(
    module: Module, versions: Versions, producer: int, rid: int, site: str
) -> int:
    """The remat certificate: the producer, what it reads at which versions, the version it
    produces and where it is replayed -- the digest a remat edge carries (MV7)."""
    claim = next(c for ph in module.phases for c in ph.claims if c.id == producer)
    return _u64_digest(
        {
            "law": "MV7",
            "producer": producer,
            "opcode": int(claim.opcode),
            "op": claim.op,
            "count": claim.count,
            "inputs": [list(x) for x in versions.inputs[producer]],
            "output": [rid, versions.writes[(producer, rid)]],
            "site": site,
        }
    )


def accuracy_certificate(module: Module, rid: int, bits: int, readers) -> int:
    """The compression certificate: the resource, the codec's bits, and each reading claim's
    declared tolerance and its error bound with the codec's quantization step (MV8)."""
    from .precision import accuracy_bound

    rows = []
    for c in sorted(readers, key=lambda c: c.id):
        rows.append([c.id, c.tolerance_ulp, accuracy_bound(replace(c, quantized_bits=bits))])
    return _u64_digest({"law": "MV8", "rid": rid, "bits": bits, "readers": rows})


def codec_admits(claim: Claim, bits: int) -> bool:
    """Whether `claim` tolerates reading an operand through a `bits`-wide lossy codec: it
    declares an R17 tolerance and its error bound with the quantization step fits it."""
    from .precision import meets_tolerance

    if claim.tolerance_ulp <= 0:
        return False
    return meets_tolerance(replace(claim, quantized_bits=bits), claim.tolerance_ulp)


# --- the transform --------------------------------------------------------------------------


@dataclass(frozen=True)
class Copy:
    """A resource of M' that holds a logical resource's bytes in one bank. Episode 0 is the
    home copy -- the logical RID itself, in its home bank; every other episode is a new RID."""

    rid: int
    logical: int
    bank: str
    episode: int


@dataclass(frozen=True)
class MoveRecord:
    """One move (or remat) claim of M' and what it means: the logical resource, the banks, the
    copies it reads and writes, the route, the edge kind and coherence action, the version of
    the logical resource it lands, the replayed producer (remat), the codec bits (compressed)
    and the certificate digest (remat / compressed; 0 = none)."""

    claim_id: int
    rid: int
    src_bank: str
    dst_bank: str
    src_copy: int
    dst_copy: int
    route: tuple[str, ...]
    kind: str
    coherence: str
    version: int
    size_bytes: int
    producer: int = -1
    bits: int = 0
    cert: int = 0


@dataclass(frozen=True)
class Eviction:
    """A forced episode split: the copy of `rid` in `bank` ends before M's phase `phase` (after
    its last use; a dirty sole copy is written back first) and is reloaded when next needed."""

    rid: int
    bank: str
    phase: int


@dataclass
class MovementTransform:
    """M' and the relation that makes it checkable: the site of every original claim, every
    copy of M' (home copies included), every move/remat claim's meaning, and the choices the
    transform was run under."""

    source: Module
    module: Module
    spec: MovementSpec
    sites: dict[int, str]
    copies: dict[int, Copy]
    moves: list[MoveRecord]
    evictions: tuple[Eviction, ...] = ()
    remat: frozenset = frozenset()
    compress: frozenset = frozenset()
    #: M' phase id -> (role, M phase): "in" (the moves before it), "main" (M's own), "out" (the
    #: eviction writebacks after an M' phase), "final" (the closing writebacks)
    roles: dict = field(default_factory=dict)

    @property
    def move_of(self) -> dict[int, MoveRecord]:
        return {m.claim_id: m for m in self.moves}

    def bindings(self):
        """`ResourceBankBinding`s for every resource M' touches (the static memory input)."""
        from .static_memory import ResourceBankBinding

        touched = {r for ph in self.module.phases for c in ph.claims for r in (*c.rd, *c.wr)}
        return tuple(
            ResourceBankBinding(rid, self.copies[rid].bank)
            for rid in sorted(touched)
            if rid in self.copies
        )


def _bytes_of(res: Resource) -> int:
    return res.count * res.elem_bytes


def _move_kind(spec: MovementSpec, route: tuple[str, ...]) -> str:
    if len(route) > 2:
        return "staged"
    host = {
        b.name for b in spec.hardware.banks if Domain[b.domain] == Domain.RAM
    }  # the host's memory: a move that touches it is a DMA, one that does not is a peer copy
    return "direct" if route[0] in host or route[-1] in host else "peer"


class _Builder:
    """The single pass that builds M' (see `transform`).

    M is walked in program order. The moves and remats a phase needs land in an inbound phase
    before it (`in(p)`); the writeback a forced eviction owes lands in an outbound phase right
    after the copy's last use (`out(q)`); the final writebacks land in one closing phase. A
    claim keeps its id and its phase, so the canonical in-phase order (ascending id) is M's.
    Phase deps order every new phase after the phases that last wrote what it reads, a
    reload after the eviction that freed its room, and every phase that followed `q` after
    `out(q)`. A fetch never writes a home copy (it opens a new episode); only a writeback does,
    and it is ordered after every access of the older versions (MV4's race law checks it)."""

    def __init__(self, module, spec, sites, remat, compress, evictions):
        self.m = module
        self.spec = spec
        self.sites = sites
        self.remat = frozenset(remat)
        self.compress = frozenset(compress)
        self.evictions = tuple(sorted(set(evictions), key=lambda e: (e.phase, e.rid, e.bank)))
        self.versions = versions_of(module)
        self.order = execution_order(module)
        self.pid_of = {c.id: p for p, c in self.order}
        self.by_id = {c.id: c for _p, c in self.order}
        self.next_rid = max(module.resources, default=0) + 1
        self.next_cid = max((c.id for ph in module.phases for c in ph.claims), default=0) + 1
        self.next_pid = max((ph.phase_id for ph in module.phases), default=0) + 1
        self.copies: dict[int, Copy] = {}
        self.resources: dict[int, Resource] = dict(module.resources)
        self.episode_count: dict[tuple[int, str], int] = {}
        # (rid, bank) -> [copy rid, version held, exact?, last user claim, last user M' phase]
        self.state: dict[tuple[int, str], list] = {}
        self.writer_phase: dict[int, int] = {}  # copy rid -> M' phase of its last writer
        self.cur = {rid: 0 for rid in module.resources}
        self.moves: list[MoveRecord] = []
        self.claims: dict[int, list[Claim]] = {ph.phase_id: [] for ph in module.phases}
        self.extra: dict[int, list[Claim]] = {}  # new phase id -> its claims
        self.extra_deps: dict[int, set] = {}  # any phase id -> deps added to it
        self.in_pid: dict[int, int] = {}  # M phase -> its inbound phase
        self.out_pid: dict[int, int] = {}  # M' phase -> its outbound phase
        self.final_pid: int | None = None
        self.homes: dict[int, str] = {}
        for rid, res in module.resources.items():
            home = spec.home_of(module, rid)
            if spec.bank_domain(home) != res.domain:
                raise ValueError(f"resource {rid} home bank {home!r} is not of its domain")
            self.copies[rid] = Copy(rid, rid, home, 0)  # an isolated resource is bound, never moved
            if res.domain in ISOLATED_DOMAINS:
                continue
            self.homes[rid] = home
            self.state[(rid, home)] = [rid, 0, True, None, None]

    # phases
    def _new_phase(self) -> int:
        pid = self.next_pid
        self.next_pid += 1
        self.extra[pid] = []
        self.extra_deps.setdefault(pid, set())
        return pid

    def _inbound(self, pid: int) -> int:
        if pid not in self.in_pid:
            self.in_pid[pid] = self._new_phase()
        return self.in_pid[pid]

    def _outbound(self, pid: int) -> int:
        if pid not in self.out_pid:
            self.out_pid[pid] = self._new_phase()
            self.extra_deps[self.out_pid[pid]].add(pid)
        return self.out_pid[pid]

    def _final(self) -> int:
        if self.final_pid is None:
            self.final_pid = self._new_phase()
        return self.final_pid

    def _after(self, phase: int, dep: int | None) -> None:
        if dep is not None and dep != phase:
            self.extra_deps.setdefault(phase, set()).add(dep)

    # copies
    def _fresh_banks(self, rid: int, exact_only: bool = True) -> list[str]:
        v = self.cur[rid]
        return sorted(
            b
            for (r, b), st in self.state.items()
            if r == rid and st[1] == v and (st[2] or not exact_only)
        )

    def _new_copy(self, rid: int, bank: str) -> int:
        logical = self.m.resources[rid]
        episode = self.episode_count.get((rid, bank), 0) + 1
        self.episode_count[(rid, bank)] = episode
        crid = self.next_rid
        self.next_rid += 1
        self.resources[crid] = replace(
            logical,
            rid=crid,
            domain=self.spec.bank_domain(bank),
            name=f"{logical.name or f'r{rid}'}@{bank}#{episode}",
        )
        self.copies[crid] = Copy(crid, rid, bank, episode)
        return crid

    def _route_from(self, rid: int, dst: str) -> tuple[str, ...] | None:
        best = None
        for src in self._fresh_banks(rid):
            if src == dst:
                continue
            route = self.spec.route(src, dst)
            if route is None:
                continue
            key = (
                len(route),
                sum(
                    int(self.spec.hardware.link(x, y).latency_ns) for x, y in zip(route, route[1:])
                ),
                route,
            )
            if best is None or key < best:
                best = key
        return None if best is None else best[2]

    def _hops(self, phase, rid, route, kind, coherence, *, home=False, bits=0, cert=0):
        """One move claim per hop of `route` into `phase`, each landing a fresh copy; the last
        hop lands in the home copy itself when `home` (a writeback)."""
        res = self.m.resources[rid]
        v = self.cur[rid]
        hops = list(zip(route, route[1:]))
        for index, (a, b) in enumerate(hops):
            src = self.state[(rid, a)][0]
            final = index == len(hops) - 1
            hop_kind = kind if len(hops) == 1 or (final and kind == "evicted") else "staged"
            dst = rid if final and home else self._new_copy(rid, b)
            far = self.spec.bank_domain(a) != self.spec.bank_domain(b)
            count = res.count if not bits else max(1, -(-res.count * bits // (8 * res.elem_bytes)))
            claim = Claim(
                id=self.next_cid,
                opcode=Opcode.ADD,
                lane=Lane.U,
                stride_class=StrideClass.UNIT,
                count=count,
                rd=(src,),
                wr=(dst,),
                op=f"{MOVE_PREFIX}{'far' if far else 'near'}:{a}->{b}",
                domain=self.spec.bank_domain(b),
                bounds="assumed_safe",
            )
            self.next_cid += 1
            self.extra[phase].append(claim)
            self._after(phase, self.writer_phase.get(src))
            self.moves.append(
                MoveRecord(
                    claim.id,
                    rid,
                    a,
                    b,
                    src,
                    dst,
                    tuple(route),
                    hop_kind if not bits else "compressed",
                    coherence if final else "none",
                    v,
                    _bytes_of(res),
                    bits=bits,
                    cert=cert,
                )
            )
            st = self.state.get((rid, a))
            st[3], st[4] = claim.id, phase
            self.state[(rid, b)] = [dst, v, not bits, claim.id, phase]
            self.writer_phase[dst] = phase

    def _readers_until_write(self, index: int, rid: int, site: str) -> list[Claim]:
        """The original claims in `site` that read the version of `rid` current at `index`:
        from the claim at `index` on, up to (and including) the next claim that writes it."""
        out = []
        for _pid, c in self.order[index:]:
            if rid in c.rd and self.sites[c.id] == site:
                out.append(c)
            if rid in c.wr:
                break
        return out

    def _need(self, index: int, pid: int, claim: Claim, rid: int, site: str) -> int:
        """The RID of a copy of `rid` in `site` holding the current version, made fresh."""
        st = self.state.get((rid, site))
        v = self.cur[rid]
        if st is not None and st[1] == v:
            st[3], st[4] = claim.id, pid
            return st[0]
        if v > 0 and self.pid_of[self.versions.producer[(rid, v)]] == pid:
            raise ValueError(
                f"claim {claim.id} reads resource {rid} that its own phase {pid} produced in "
                f"another bank: a move cannot sit between two claims of one phase"
            )
        phase = self._inbound(pid)
        # rematerialize a recomputable temporary from a replay-certified producer
        if rid in self.remat and self.spec.class_of(rid) == "recomputable" and v > 0:
            producer = self.versions.producer[(rid, v)]
            pclaim = self.by_id[producer]
            inputs = self.versions.inputs[producer]
            if (
                not replay_safe(self.m, pclaim)
                and len(pclaim.wr) == 1
                and all(self._replayable_input(x, vx, site) for x, vx in inputs)
            ):
                for x, vx in inputs:  # land what the replay reads, at the versions it first read
                    held = self.state.get((x, site))
                    if held is None or held[1] != vx:
                        self._need(index, pid, claim, x, site)
                dst = self._new_copy(rid, site)
                clone = replace(
                    pclaim,
                    id=self.next_cid,
                    rd=tuple(self.state[(x, site)][0] for x in pclaim.rd),
                    wr=(dst,),
                )
                clone = replace(
                    clone,
                    domain=_declared_domain(clone, self.resources, self.spec.bank_domain(site)),
                )
                self.next_cid += 1
                self.extra[phase].append(clone)
                for x in pclaim.rd:
                    held = self.state[(x, site)]
                    self._after(phase, self.writer_phase.get(held[0]))
                    held[3], held[4] = clone.id, phase
                cert = replay_certificate(self.m, self.versions, producer, rid, site)
                self.moves.append(
                    MoveRecord(
                        clone.id,
                        rid,
                        site,
                        site,
                        0,
                        dst,
                        (site,),
                        "rematerialized",
                        "none",
                        v,
                        _bytes_of(self.m.resources[rid]),
                        producer=producer,
                        cert=cert,
                    )
                )
                self.state[(rid, site)] = [dst, v, True, claim.id, pid]
                self.writer_phase[dst] = phase
                return dst
        route = self._route_from(rid, site)
        if route is None:
            raise ValueError(f"no route lands resource {rid} in bank {site!r} for claim {claim.id}")
        bits = cert = 0
        if rid in self.compress and self.spec.codec_of(rid) and len(route) == 2:
            readers = self._readers_until_write(index, rid, site)
            b = self.spec.codec_of(rid)
            if readers and all(codec_admits(c, b) for c in readers):
                bits, cert = b, 1  # the certificate is bound to the actual readers after the build
        self._hops(phase, rid, route, _move_kind(self.spec, route), "none", bits=bits, cert=cert)
        st = self.state[(rid, site)]
        st[3], st[4] = claim.id, pid
        return st[0]

    def _replayable_input(self, rid: int, version: int, site: str) -> bool:
        """Whether a replay in `site` can read `version` of `rid`: a copy there holds it, or it
        is still the current version and a route lands it there."""
        held = self.state.get((rid, site))
        if held is not None and held[1] == version:
            return True
        if _isolated(self.m, rid) or self.cur[rid] != version:
            return False
        return self._route_from(rid, site) is not None

    def _writeback(self, phase: int, rid: int, kind: str | None = None) -> None:
        home = self.homes[rid]
        route = self._route_from(rid, home)
        if route is None:
            raise ValueError(f"no route writes resource {rid} back to {home!r}")
        self._hops(phase, rid, route, kind or _move_kind(self.spec, route), "writeback", home=True)

    def _apply_evictions(self, pid: int) -> None:
        for ev in self.evictions:
            if ev.phase != pid:
                continue
            st = self.state.get((ev.rid, ev.bank))
            if st is None or st[0] == ev.rid:
                continue  # nothing there, or the home copy (it is never evicted)
            last_phase = st[4]
            v = self.cur[ev.rid]
            only = st[1] == v and self._fresh_banks(ev.rid) == [ev.bank]
            cls = self.spec.class_of(ev.rid)
            freed_by = last_phase
            if (
                only
                and cls == "mutable"
                or (only and cls == "recomputable" and ev.rid not in self.remat)
            ):
                out = self._outbound(last_phase)
                self._writeback(out, ev.rid, "evicted")
                freed_by = out
            del self.state[(ev.rid, ev.bank)]
            # whatever lands after the eviction waits for the room it freed
            self._after(pid, freed_by)
            self._after(self._inbound(pid), freed_by)

    def build(self) -> MovementTransform:
        seen = None
        for index, (pid, c) in enumerate(self.order):
            if pid != seen:
                seen = pid
                self._apply_evictions(pid)
            site = self.sites[c.id]
            if all(_isolated(self.m, r) for r in (*c.rd, *c.wr)):
                self.claims[pid].append(c)
                continue
            rd = [r if _isolated(self.m, r) else self._need(index, pid, c, r, site) for r in c.rd]
            wr = []
            for rid in c.wr:
                if _isolated(self.m, rid):
                    wr.append(rid)
                    continue
                if self.spec.class_of(rid) == "immutable":
                    raise ValueError(f"claim {c.id} writes immutable resource {rid}")
                if rid in c.rd or not _whole_write(c, self.m.resources[rid]):
                    wr.append(self._need(index, pid, c, rid, site))
                else:
                    st = self.state.get((rid, site))
                    wr.append(st[0] if st is not None else self._new_copy(rid, site))
            moved = replace(c, rd=tuple(rd), wr=tuple(wr))
            domain = _declared_domain(moved, self.resources, self.spec.bank_domain(site))
            self.claims[pid].append(replace(moved, domain=domain) if domain != c.domain else moved)
            for rid, crid in zip(c.wr, wr):
                if _isolated(self.m, rid):
                    continue
                self.cur[rid] = self.versions.writes[(c.id, rid)]
                self.state[(rid, site)] = [crid, self.cur[rid], True, c.id, pid]
                self.writer_phase[crid] = pid
        # every mutable resource's home holds its final version (generation-checked writeback)
        for rid in sorted(self.homes):
            if self.spec.class_of(rid) != "mutable" or self.versions.final.get(rid, 0) == 0:
                continue
            st = self.state.get((rid, self.homes[rid]))
            if st is not None and st[0] == rid and st[1] == self.cur[rid]:
                continue
            self._writeback(self._final(), rid)
        return self._assemble()

    def _assemble(self) -> MovementTransform:
        # an outbound phase sits right after its anchor in every chain the anchor was in
        for anchor, out in self.out_pid.items():
            for ph in self.m.phases:
                if anchor in ph.deps:
                    self._after(ph.phase_id, out)
            for pid in self.extra:
                if pid != out and anchor in self.extra_deps.get(pid, ()):
                    self._after(pid, out)
        phases: list[Phase] = []
        for ph in self.m.phases:
            p = ph.phase_id
            if p in self.in_pid:
                i = self.in_pid[p]
                deps = set(ph.deps) | self.extra_deps.get(i, set())
                phases.append(Phase(i, tuple(sorted(deps - {i})), self.extra[i]))
                self._after(p, i)
            # M's claims in M's declared order, M's deps first: nothing moved, nothing changed
            mine = {c.id: c for c in self.claims[p]}
            added = sorted(self.extra_deps.get(p, set()) - set(ph.deps) - {p})
            phases.append(
                replace(
                    ph, deps=tuple(ph.deps) + tuple(added), claims=[mine[c.id] for c in ph.claims]
                )
            )
        for anchor, out in sorted(self.out_pid.items(), key=lambda x: x[1]):
            deps = self.extra_deps.get(out, set())
            phases.append(Phase(out, tuple(sorted(deps - {out})), self.extra[out]))
        if self.final_pid is not None:
            others = {ph.phase_id for ph in phases}
            phases.append(Phase(self.final_pid, tuple(sorted(others)), self.extra[self.final_pid]))
        resources = dict(self.m.resources)
        resources.update(sorted((r, v) for r, v in self.resources.items() if r not in resources))
        module = replace(self.m, resources=resources, phases=phases)
        # a compressed copy's certificate names the claims that actually read it
        readers: dict[int, list[Claim]] = {}
        for ph in phases:
            for c in ph.claims:
                if c.id in self.by_id:
                    for rid in c.rd:
                        readers.setdefault(rid, []).append(self.by_id[c.id])
        for i, mv in enumerate(self.moves):
            if mv.bits:
                cert = accuracy_certificate(self.m, mv.rid, mv.bits, readers.get(mv.dst_copy, ()))
                self.moves[i] = replace(mv, cert=cert)
        return MovementTransform(
            source=self.m,
            module=module,
            spec=self.spec,
            sites=dict(self.sites),
            copies=dict(sorted(self.copies.items())),
            moves=list(self.moves),
            evictions=self.evictions,
            remat=self.remat,
            compress=self.compress,
            roles=self._roles(),
        )

    def _roles(self) -> dict:
        roles = {ph.phase_id: ("main", ph.phase_id) for ph in self.m.phases}
        for p, i in self.in_pid.items():
            roles[i] = ("in", p)
        for anchor, o in self.out_pid.items():
            roles[o] = ("out", anchor)
        if self.final_pid is not None:
            roles[self.final_pid] = ("final", -1)
        return roles


def default_sites(module: Module, spec: MovementSpec) -> dict[int, str]:
    """The do-nothing assignment: every claim runs in the home of its first memory operand
    (a claim with none stays in the first compute bank)."""
    banks = spec.compute_banks()
    out = {}
    for _pid, c in execution_order(module):
        home = next(
            (spec.home_of(module, r) for r in (*c.rd, *c.wr) if not _isolated(module, r)), None
        )
        out[c.id] = home if home is not None else banks[0]
    return out


def site_choices(module: Module, spec: MovementSpec) -> dict[int, tuple[str, ...]]:
    """The banks each claim may run in: the spec's restriction, else every compute bank; a
    claim that touches an isolated domain is pinned to its default site."""
    base = default_sites(module, spec)
    banks = spec.compute_banks()
    out = {}
    for _pid, c in execution_order(module):
        if (
            c.domain in ISOLATED_DOMAINS
            or c.volatile
            or any(_isolated(module, r) for r in (*c.rd, *c.wr))
        ):
            out[c.id] = (base[c.id],)
            continue
        out[c.id] = spec.sites_for(c.id) or banks
    return out


def transform(
    module: Module,
    spec: MovementSpec,
    sites: dict[int, str] | None = None,
    *,
    remat=(),
    compress=(),
    evictions=(),
) -> MovementTransform:
    """Build M' for a site assignment (default: `default_sites`) -- deterministic in its
    inputs, so the verifier and the planner agree on what a choice means."""
    sites = dict(sites) if sites is not None else default_sites(module, spec)
    choices = site_choices(module, spec)
    for cid, allowed in choices.items():
        if cid not in sites:
            raise ValueError(f"claim {cid} has no site")
        if sites[cid] not in allowed:
            raise ValueError(f"claim {cid} may not run in {sites[cid]!r} (allowed {allowed})")
    return _Builder(module, spec, sites, remat, compress, evictions).build()


# --- pricing a choice: capacity, the realization, the placement -----------------------------

#: Candidate evaluations the exact planner may spend (a work-unit budget, G12's discipline):
#: within it the planner enumerates every site assignment and decision and its answer is the
#: optimum of the joint objective (TMSAO-1 on that row); beyond it the answer is a local search
#: from the best baseline and carries no optimality claim (TMSAO-4).
DEFAULT_BUDGET = 4096


def _positions(module: Module) -> dict[int, int]:
    from ..model import topological_phase_ids

    return {pid: i for i, pid in enumerate(topological_phase_ids(module))}


def _aligned_bytes(res: Resource, bank) -> int:
    alignment = max(res.align, bank.alignment)
    return -(-(res.count * res.elem_bytes) // alignment) * alignment


def copy_intervals(tr: MovementTransform) -> dict[int, tuple[int, int, frozenset]]:
    """copy RID -> (first, last, touched): the topological phase positions M' touches it in.
    Phase liveness, the static memory planner's default domain: a copy is live from the first
    phase that touches it through the last."""
    pos = _positions(tr.module)
    touched: dict[int, set] = {}
    for ph in tr.module.phases:
        for c in ph.claims:
            for rid in (*c.rd, *c.wr):
                if rid in tr.copies:
                    touched.setdefault(rid, set()).add(pos[ph.phase_id])
    return {rid: (min(p), max(p), frozenset(p)) for rid, p in touched.items()}


def bank_occupancy(tr: MovementTransform, intervals=None) -> dict[tuple[str, int], int]:
    """(bank, position) -> the aligned bytes of the copies live there."""
    intervals = copy_intervals(tr) if intervals is None else intervals
    out: dict[tuple[str, int], int] = {}
    for rid, (first, last, _touched) in intervals.items():
        cp = tr.copies[rid]
        size = _aligned_bytes(tr.module.resources[rid], tr.spec.hardware.bank(cp.bank))
        for p in range(first, last + 1):
            out[(cp.bank, p)] = out.get((cp.bank, p), 0) + size
    return out


def _overflow(tr: MovementTransform, intervals) -> tuple[str, int] | None:
    for (bank, p), used in sorted(
        bank_occupancy(tr, intervals).items(), key=lambda x: (x[0][1], x[0][0])
    ):
        if used > tr.spec.hardware.bank(bank).allocatable_bytes:
            return bank, p
    return None


def _victim(tr: MovementTransform, intervals, bank: str, position: int) -> Eviction | None:
    """The copy to end before the M phase whose room overflows at M' `position`: live across the
    overflow, idle in that phase and in its inbound phase, not a home copy; the one used again
    farthest in the future (Belady), then the larger, then the lower RID."""
    from ..model import topological_phase_ids

    topo = topological_phase_ids(tr.module)
    pos = {pid: i for i, pid in enumerate(topo)}
    target = None
    for pid in topo[position:]:
        role, anchor = tr.roles.get(pid, ("main", pid))
        if role in ("main", "in"):
            target = anchor
            break
    if target is None:
        return None
    busy = {pos[target]} | {
        pos[pid] for pid, (role, anchor) in tr.roles.items() if role == "in" and anchor == target
    }
    best = None
    for rid, (first, last, touched) in intervals.items():
        cp = tr.copies[rid]
        if cp.bank != bank or cp.episode == 0 or not first < position < last:
            continue
        if touched & busy or position in touched:
            continue
        nxt = min(p for p in touched if p > position)
        size = _aligned_bytes(tr.module.resources[rid], tr.spec.hardware.bank(bank))
        key = (-nxt, -size, rid)
        if best is None or key < best[0]:
            best = (key, Eviction(cp.logical, bank, target))
    return None if best is None else best[1]


@dataclass
class PricedMovement:
    """A choice made concrete: its transform, the realization and the scheduled price of M',
    and M''s static memory plan (every copy has an address that fits its bank)."""

    transform: MovementTransform
    result: object
    makespan: int
    serial: int
    static: object

    @property
    def key(self) -> tuple:
        """The joint objective, lexicographic: M(pi, Theta) (the scheduled makespan), then the
        serial Sigma, then the choice itself (a deterministic tie-break)."""
        tr = self.transform
        return (
            self.makespan,
            self.serial,
            tuple(sorted(tr.sites.items())),
            tuple(sorted(tr.remat)),
            tuple(sorted(tr.compress)),
        )


def price(
    module: Module,
    spec: MovementSpec,
    sites: dict[int, str],
    h,
    theta,
    policy=None,
    mode: str = "eft",
    *,
    remat=(),
    compress=(),
    max_evictions: int = 64,
) -> PricedMovement | None:
    """Price one choice jointly: build M', end the episodes the banks cannot hold (Belady, one
    forced eviction at a time), lay out every copy, realize M' and place it. None when the
    choice is infeasible: no route, a bank it cannot fit, or a makespan past the deadline."""
    from ..gem.overlap import price_scheduled
    from .realize import optimize
    from .static_memory import plan_static_memory
    from .weights import PERF

    policy = policy or PERF
    evictions: list[Eviction] = []
    for _ in range(max_evictions + 1):
        try:
            tr = transform(module, spec, sites, remat=remat, compress=compress, evictions=evictions)
        except ValueError:
            return None
        intervals = copy_intervals(tr)
        over = _overflow(tr, intervals)
        if over is None:
            break
        ev = _victim(tr, intervals, *over)
        if ev is None or ev in evictions:
            return None
        evictions.append(ev)
    else:
        return None
    static = None
    for layout in ("first-fit", "exact"):
        try:
            static = plan_static_memory(tr.module, tr.bindings(), spec.hardware, layout=layout)
            break
        except ValueError:
            continue
    if static is None:
        return None
    result = optimize(tr.module, h, theta, policy)
    sp = price_scheduled(tr.module, result, h, theta, policy, mode)
    if spec.deadline and sp.makespan > spec.deadline:
        return None
    return PricedMovement(tr, result, sp.makespan, sp.serial, static)


# --- the planners ---------------------------------------------------------------------------


def standalone_cost(
    claim: Claim, bank: str, spec: MovementSpec, h, theta, phase_id: int, policy=None
) -> int:
    """What `claim` costs by itself with its operands in `bank`'s tier: its cheapest
    candidate's realized step cost, no movement -- the compute-only view the sequential
    baseline places by."""
    from .realize import candidates_for, step_cost
    from .weights import PERF

    policy = policy or PERF
    probe = Resource(0, domain=spec.bank_domain(bank))
    return min(
        step_cost(None, cand, h, theta, phase_id, policy)
        for cand in candidates_for(claim, h, probe)
    )


def _transfer_units(res: Resource, theta, h, phase_id: int, policy) -> int:
    """channels.orchestrate's cross-device term for landing `res` elsewhere, scalarized: FABRIC
    one memory stream of its bytes, one SYNC barrier."""
    from ..channels import XFER_BW_Q8, XFER_SYNC
    from .cost import CostVector
    from .weights import weights

    vec = CostVector.of(fabric=(_bytes_of(res) * XFER_BW_Q8) >> 8, sync=XFER_SYNC)
    return vec.dot(weights(h, theta, phase_id, policy))


def sequential_sites(module: Module, spec: MovementSpec, h, theta, policy=None) -> dict[int, str]:
    """The disconnected step the roadmap rules out: every claim placed where it computes
    cheapest, movement priced afterwards."""
    choices = site_choices(module, spec)
    base = default_sites(module, spec)
    out = {}
    for pid, c in execution_order(module):
        out[c.id] = min(
            choices[c.id],
            key=lambda b: (standalone_cost(c, b, spec, h, theta, pid, policy), b != base[c.id], b),
        )
    return out


def greedy_sites(module: Module, spec: MovementSpec, h, theta, policy=None) -> dict[int, str]:
    """The pre-G8 transfer-aware placement (`channels.orchestrate`'s rule, as a site choice): a
    forward pass that puts each claim where its compute plus the transfer of the operands not
    already there is lowest, remembering where it left each resource. Order-dependent and
    myopic: it never pays a move now that later claims would amortize."""
    from .weights import PERF

    policy = policy or PERF
    choices = site_choices(module, spec)
    base = default_sites(module, spec)
    fresh = {
        rid: {spec.home_of(module, rid)} for rid in module.resources if not _isolated(module, rid)
    }
    out = {}
    for pid, c in execution_order(module):
        need = [r for r in c.rd if r in fresh] + [
            r
            for r in c.wr
            if r in fresh and (r in c.rd or not _whole_write(c, module.resources[r]))
        ]

        def cost(b, c=c, pid=pid, need=need):
            move = sum(
                _transfer_units(module.resources[r], theta, h, pid, policy)
                for r in dict.fromkeys(need)
                if b not in fresh[r]
            )
            return (standalone_cost(c, b, spec, h, theta, pid, policy) + move, b != base[c.id], b)

        site = min(choices[c.id], key=cost)
        out[c.id] = site
        for r in need:
            fresh[r].add(site)
        for r in c.wr:
            if r in fresh:
                fresh[r] = {site}
    return out


@dataclass
class MovementPlan:
    """The planner's answer and its evidence: the chosen priced choice, how it was found
    (`exact` enumerated every candidate within the budget -- the optimum of the joint objective
    -- or `greedy`, a local search from the best baseline with no optimality claim), the size of
    the candidate space, the evaluations spent, and the three baselines it was measured against
    (`home`: the do-nothing sites; `sequential`: compute-only sites, movement after; `greedy`:
    the pre-G8 transfer-aware forward pass)."""

    best: PricedMovement | None
    method: str
    optimum: bool
    evaluated: int
    budget: int
    space: int
    baselines: dict

    @property
    def tmsao(self) -> str:
        return "TMSAO-1" if self.optimum else "TMSAO-4"


def _subsets(items) -> list[frozenset]:
    items = sorted(items)
    return [
        frozenset(combo)
        for k in range(len(items) + 1)
        for combo in itertools.combinations(items, k)
    ]


def plan_movement(
    module: Module,
    spec: MovementSpec,
    h,
    theta,
    policy=None,
    mode: str = "eft",
    *,
    method: str = "auto",
    budget: int = DEFAULT_BUDGET,
) -> MovementPlan:
    """Choose sites, remats and compressed transfers jointly with the realization and the
    placement they produce. `method` "exact" enumerates (refusing a space over `budget`),
    "greedy" searches locally, "auto" enumerates when the space fits the budget."""
    if method not in ("auto", "exact", "greedy"):
        raise ValueError(f"unknown movement planning method {method!r}")
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("movement planning budget must be a positive integer")
    choices = site_choices(module, spec)
    ids = sorted(choices)
    remat_ok = sorted(
        r for r in spec.remat if r in module.resources and spec.class_of(r) == "recomputable"
    )
    compress_ok = sorted(r for r, _bits in spec.codecs if r in module.resources)
    space = 1
    for cid in ids:
        space *= len(choices[cid])
    space *= 2 ** (len(remat_ok) + len(compress_ok))
    baselines = {
        "home": price(module, spec, default_sites(module, spec), h, theta, policy, mode),
        "sequential": price(
            module, spec, sequential_sites(module, spec, h, theta, policy), h, theta, policy, mode
        ),
        "greedy": price(
            module, spec, greedy_sites(module, spec, h, theta, policy), h, theta, policy, mode
        ),
    }
    evaluated = 3
    if method == "exact" and space > budget:
        raise ValueError(f"movement candidate space {space} exceeds the budget {budget}")
    if method in ("auto", "exact") and space <= budget:
        best = None
        for combo in itertools.product(*(choices[c] for c in ids)):
            sites = dict(zip(ids, combo))
            for rm in _subsets(remat_ok):
                for cp in _subsets(compress_ok):
                    p = price(module, spec, sites, h, theta, policy, mode, remat=rm, compress=cp)
                    evaluated += 1
                    if p is not None and (best is None or p.key < best.key):
                        best = p
        return MovementPlan(best, "exact", True, evaluated, budget, space, baselines)
    feasible = [p for p in baselines.values() if p is not None]
    if not feasible:
        return MovementPlan(None, "greedy", False, evaluated, budget, space, baselines)
    best = min(feasible, key=lambda p: p.key)
    sites = dict(best.transform.sites)
    rm, cp = set(best.transform.remat), set(best.transform.compress)
    improved = True
    while improved and evaluated < budget:
        improved = False
        trials = [("site", cid, b) for cid in ids for b in choices[cid]]
        trials += [("remat", r, None) for r in remat_ok] + [
            ("compress", r, None) for r in compress_ok
        ]
        for what, key, value in trials:
            if evaluated >= budget:
                break
            t_sites, t_rm, t_cp = dict(sites), set(rm), set(cp)
            if what == "site":
                if value == sites[key]:
                    continue
                t_sites[key] = value
            elif what == "remat":
                t_rm ^= {key}
            else:
                t_cp ^= {key}
            p = price(module, spec, t_sites, h, theta, policy, mode, remat=t_rm, compress=t_cp)
            evaluated += 1
            if p is not None and p.key < best.key:
                best, sites, rm, cp, improved = p, t_sites, t_rm, t_cp, True
    return MovementPlan(best, "greedy", False, evaluated, budget, space, baselines)


# --- the plan: edges, windows, binding ------------------------------------------------------


def _accesses(module: Module):
    """(position, claim) in execution order, and per RID the positions that write and read it."""
    order = [c for _p, c in execution_order(module)]
    writes: dict[int, list[int]] = {}
    reads: dict[int, list[int]] = {}
    for i, c in enumerate(order):
        for rid in c.rd:
            reads.setdefault(rid, []).append(i)
        for rid in c.wr:
            writes.setdefault(rid, []).append(i)
    return order, writes, reads


def _window(
    order, writes, reads, index: int, src: int | None, dst: int
) -> tuple[int | None, int | None]:
    """(after, before) of the move at `index`: the last claim before it that wrote its source
    copy, and the first claim after it that reads its destination copy."""
    after = None
    if src is not None:
        prior = [i for i in writes.get(src, ()) if i < index]
        after = order[prior[-1]].id if prior else None
    later = [i for i in reads.get(dst, ()) if i > index]
    before = order[later[0]].id if later else None
    return after, before


def movement_edges(pm: PricedMovement) -> list:
    """One `MovementEdge` per move/remat claim of M', in execution order: the logical resource
    and its registry generations, the banks, the bytes, the route, the kind and coherence
    action, and the v3 tail (the executing claim, the version landed, the window and producer
    flags, the codec bits and the certificate)."""
    from ..gem.execution_plan import (
        MOVE_HAS_AFTER,
        MOVE_HAS_BEFORE,
        MOVE_HAS_CLAIM,
        MOVE_HAS_PRODUCER,
        MovementEdge,
    )

    tr = pm.transform
    order, writes, reads = _accesses(tr.module)
    index = {c.id: i for i, c in enumerate(order)}
    out = []
    for mv in sorted(tr.moves, key=lambda m: index[m.claim_id]):
        remat = mv.kind == "rematerialized"
        after, before = _window(
            order, writes, reads, index[mv.claim_id], None if remat else mv.src_copy, mv.dst_copy
        )
        flags = MOVE_HAS_CLAIM
        flags |= MOVE_HAS_AFTER if after is not None else 0
        flags |= MOVE_HAS_BEFORE if before is not None else 0
        flags |= MOVE_HAS_PRODUCER if remat else 0
        res = tr.source.resources[mv.rid]
        out.append(
            MovementEdge(
                rid=mv.rid,
                src_bank=mv.src_bank,
                dst_bank=mv.dst_bank,
                offset=0,
                size_bytes=mv.size_bytes,
                route=">".join(mv.route),
                kind=mv.kind,
                coherence=mv.coherence,
                map_gen=res.map_gen,
                data_gen=res.data_gen,
                after_claim=after or 0,
                before_claim=before or 0,
                claim=mv.claim_id,
                version=mv.version,
                flags=flags,
                producer=mv.producer if remat else 0,
                bits=mv.bits,
                cert=mv.cert,
            )
        )
    return out


def execution_plan_of(pm: PricedMovement, target, mode: str = "eft", *, plan: str = "plan0"):
    """Mint the ExecutionPlan of a priced choice: M''s realization placed by the canonical
    dispatch (`plan_from_realization`), its static memory plan's lifetimes (every copy has an
    address in its bank), one edge per move/remat claim, and the v3 binding to M and the spec."""
    from ..gem.execution_plan import plan_from_realization

    ep = plan_from_realization(
        pm.transform.module, pm.result, target, mode, plan=plan, static_plan=pm.static
    )
    if not pm.transform.moves and pm.transform.module == pm.transform.source:
        return ep  # nothing moved: the plan of M itself, byte for byte (no v3 binding)
    ep.moves = movement_edges(pm)
    ep.source_hash = source_digest(pm.transform.source)
    ep.spec_hash = pm.transform.spec.digest()
    return ep


# --- the laws -------------------------------------------------------------------------------


def verify_movement(source: Module, spec: MovementSpec, module: Module, plan) -> list:
    """MV1-MV11 (module docstring): is `module` (M') a correctness-neutral movement transform of
    `source` (M) under `spec`, and is `plan` its faithful, bound description? Everything is
    derived from the three artifacts -- the copy relation positionally (an original claim's
    operands, a move's edge, a remat's producer), the banks from the plan's lifetimes, the
    versions by replaying both modules -- so the planner's own bookkeeping is never trusted."""
    from ..verify import Diagnostic
    from .device_manifest import check_bank_moves
    from .provenance import digest_of

    diags: list = []

    def fail(law: str, msg: str) -> None:
        diags.append(Diagnostic(law, msg))

    src_claims = {c.id: (pid, c) for pid, c in execution_order(source)}
    versions = versions_of(source)
    edges: dict[int, list] = {}
    for mv in plan.moves:
        edges.setdefault(mv.claim, []).append(mv)
    home_bank = {lt.rid: lt.bank for lt in plan.lifetimes}

    # MV11 binding: a plan that moves data (or realizes a module other than M) names M and the
    # spec; the identity transform's plan is M's own plan and binds M through module_hash
    if plan.moves or plan.source_hash or plan.spec_hash or module != source:
        if plan.source_hash != source_digest(source):
            fail("MV11", "the plan's source_hash is not the source module's digest")
        if plan.spec_hash != spec.digest():
            fail("MV11", "the plan's spec_hash is not the movement spec's digest")
    elif plan.module_hash != digest_of(source):
        fail("MV11", "the plan does not realize the source module")

    # MV1 / MV2 / MV3: classify every claim of M' and derive the copy relation
    logical: dict[int, int] = {}  # M' rid -> the logical rid it holds

    def bind(rid: int, lrid: int, where: str) -> None:
        if rid in source.resources and rid != lrid:
            fail("MV1", f"{where}: resource {rid} is a logical resource, not a copy of {lrid}")
        elif logical.setdefault(rid, lrid) != lrid:
            fail("MV1", f"{where}: resource {rid} holds both {logical[rid]} and {lrid}")

    mprime = {c.id: (pid, c) for pid, c in execution_order(module)}
    # what the relation is judged over, checked before any lookup below could trip on it (a
    # malformed transform is answered with a verdict, never a traceback): every operand M' names
    # is a resource M' declares, and every edge moves a logical, non-isolated resource of M
    bound = len(diags)
    for cid, (_pid, c) in mprime.items():
        for r in (*c.rd, *c.wr):
            if r not in module.resources:
                fail(
                    "MV1", f"claim {cid} touches resource {r}, which the transform does not declare"
                )
    for mv in plan.moves:
        if mv.rid not in source.resources:
            fail("MV11", f"an edge moves resource {mv.rid}, which is not a resource of the source")
        elif _isolated(source, mv.rid):
            fail("MV3", f"an edge moves isolated resource {mv.rid}: a device resource never moves")
    if len(diags) > bound:
        return diags
    for cid in src_claims:
        if cid not in mprime:
            fail("MV2", f"claim {cid} of the source is missing from the transform")
    moves: dict[int, object] = {}
    for cid, (pid, c) in mprime.items():
        if cid in src_claims:
            spid, o = src_claims[cid]
            site_domain = c.domain
            same = replace(o, rd=c.rd, wr=c.wr)
            if c.domain != o.domain:
                same = replace(same, domain=_declared_domain(same, module.resources, site_domain))
            if pid != spid or same != c or len(c.rd) != len(o.rd) or len(c.wr) != len(o.wr):
                fail("MV2", f"claim {cid} is not the source claim with its operands remapped")
                continue
            for a, b in zip((*o.rd, *o.wr), (*c.rd, *c.wr)):
                if _isolated(source, a):
                    if a != b:
                        fail("MV2", f"claim {cid} moved isolated operand {a}")
                else:
                    bind(b, a, f"claim {cid}")
            continue
        rows = edges.get(cid, [])
        if len(rows) != 1:
            fail("MV11", f"claim {cid} is neither a source claim nor exactly one edge's claim")
            continue
        mv = rows[0]
        moves[cid] = mv
        if mv.kind == "rematerialized":
            p = src_claims.get(mv.producer)
            if p is None:
                fail("MV7", f"remat {cid} replays unknown producer {mv.producer}")
                continue
            o = p[1]
            same = replace(o, id=cid, rd=c.rd, wr=c.wr)
            if c.domain != o.domain:
                same = replace(same, domain=_declared_domain(same, module.resources, c.domain))
            if same != c or len(c.rd) != len(o.rd) or len(o.wr) != 1 or len(c.wr) != 1:
                fail("MV7", f"remat {cid} is not a clone of producer {mv.producer}")
                continue
            for a, b in zip(o.rd, c.rd):
                bind(b, a, f"remat {cid}")
            bind(c.wr[0], mv.rid, f"remat {cid}")
        else:
            if (
                not c.op.startswith(MOVE_PREFIX)
                or c.opcode != Opcode.ADD
                or len(c.rd) != 1
                or len(c.wr) != 1
            ):
                fail(
                    "MV3", f"claim {cid} is not an explicit move (mem.move.*, one read, one write)"
                )
                continue
            bind(c.rd[0], mv.rid, f"move {cid}")
            bind(c.wr[0], mv.rid, f"move {cid}")
    for rid, res in module.resources.items():
        if rid in source.resources:
            if res != source.resources[rid]:
                fail("MV1", f"logical resource {rid} was changed")
            continue
        lrid = logical.get(rid)
        o = source.resources.get(lrid) if lrid is not None else None
        if o is None:
            fail("MV1", f"resource {rid} is neither a logical resource nor a copy of one")
            continue
        if _isolated(module, rid):
            # the isolation exemption is for the device resources M declares, which never move;
            # a copy in an isolated domain would escape every bank law below
            fail("MV1", f"copy {rid} of resource {lrid} is declared in an isolated domain")
            continue
        if (res.shape, res.elem_bytes, res.layout, res.align) != (
            o.shape,
            o.elem_bytes,
            o.layout,
            o.align,
        ):
            fail("MV1", f"copy {rid} does not have the shape of resource {lrid}")
    bank_of: dict[int, str] = {}
    for rid in module.resources:
        if rid in source.resources and not _isolated(source, rid):
            try:
                bank_of[rid] = spec.home_of(source, rid)
            except ValueError as exc:
                fail("MV1", str(exc))
        if rid in home_bank:
            if rid in bank_of and bank_of[rid] != home_bank[rid]:
                fail(
                    "MV1",
                    f"resource {rid} lives in {home_bank[rid]!r}, not its home {bank_of[rid]!r}",
                )
            bank_of[rid] = home_bank[rid]
    touched = {r for _p, c in mprime.values() for r in (*c.rd, *c.wr)}
    for rid in sorted(touched):
        if rid not in bank_of:
            if _isolated(module, rid):
                continue  # a device resource the plan does not place
            fail(
                "MV9", f"resource {rid} has no lifetime: the plan does not say which bank holds it"
            )
            continue
        try:
            bank = spec.hardware.bank(bank_of[rid])
        except KeyError:
            fail("MV1", f"resource {rid} names undeclared bank {bank_of[rid]!r}")
            continue
        if Domain[bank.domain] != module.resources[rid].domain:
            fail("MV1", f"resource {rid} is not of its bank {bank.name!r}'s domain")
    if diags:
        return diags  # the relation itself is broken; nothing below would be meaningful

    # MV2: M's phases survive with their dependences; MV4: no two accesses of one resource race
    mphase = module.phase_map()
    for ph in source.phases:
        q = mphase.get(ph.phase_id)
        if q is None or not set(ph.deps) <= set(q.deps):
            fail("MV2", f"phase {ph.phase_id} is missing or lost a dependence")
    for msg in races(module):
        fail("MV4", msg)

    # MV2: one site per original claim, D-R2 clean
    for cid, (_pid, c) in mprime.items():
        if cid not in src_claims:
            continue
        banks = {bank_of[r] for r in (*c.rd, *c.wr) if not _isolated(module, r)}
        if len(banks) > 1:
            fail(
                "MV2", f"claim {cid} reads or writes copies in {sorted(banks)}: one site per claim"
            )
        allowed = spec.sites_for(cid)
        if allowed and banks and not banks <= set(allowed):
            fail("MV2", f"claim {cid} runs in {sorted(banks)}, outside its allowed sites")
    for msg in check_bank_moves(module):
        fail("MV2", f"implicit cross-tier access: {msg}")

    # MV3 / MV11: every move is one declared hop, faithfully described
    for cid, mv in moves.items():
        c = mprime[cid][1]
        if mv.kind == "rematerialized":
            if bank_of.get(c.wr[0]) != mv.dst_bank or mv.src_bank != mv.dst_bank:
                fail("MV11", f"remat {cid}'s edge does not name the bank it replays in")
            if any(bank_of.get(r) != mv.dst_bank for r in c.rd):
                fail("MV7", f"remat {cid} reads outside the bank it replays in")
            continue
        a, b = bank_of.get(c.rd[0]), bank_of.get(c.wr[0])
        if a is None or b is None:
            fail("MV3", f"move {cid} reads or writes a resource the plan places in no bank")
            continue
        if (a, b) != (mv.src_bank, mv.dst_bank):
            fail(
                "MV11",
                f"move {cid}'s edge says {mv.src_bank}->{mv.dst_bank}, the claim moves {a}->{b}",
            )
        if spec.hardware.link(a, b) is None:
            fail("MV3", f"move {cid} crosses {a}->{b}, which no declared link joins")
        far = spec.bank_domain(a) != spec.bank_domain(b)
        if c.op != f"{MOVE_PREFIX}{'far' if far else 'near'}:{a}->{b}":
            fail("MV3", f"move {cid} is spelled {c.op!r}")
        route = tuple(mv.route.split(">")) if mv.route else ()
        hops = list(zip(route, route[1:]))
        if (a, b) not in hops or any(spec.hardware.link(x, y) is None for x, y in hops):
            fail("MV11", f"move {cid}'s route {mv.route!r} is not a declared path through {a}->{b}")
        res = source.resources[mv.rid]
        if mv.size_bytes != _bytes_of(res) or mv.offset != 0:
            fail("MV11", f"move {cid}'s edge does not carry the whole of resource {mv.rid}")
        if (mv.map_gen, mv.data_gen) != (res.map_gen, res.data_gen):
            fail("MV11", f"move {cid}'s edge carries generations other than resource {mv.rid}'s")
        want_kind = (
            "compressed"
            if mv.bits
            else "evicted"
            if mv.kind == "evicted"
            else _move_kind(spec, route)
            if len(route) >= 2
            else "direct"
        )
        if mv.kind != want_kind:
            fail("MV11", f"move {cid} is a {want_kind} edge described as {mv.kind!r}")
        if mv.kind == "evicted" and mv.coherence != "writeback":
            fail("MV5", f"eviction {cid} drops its bytes without a writeback")

    # MV4 / MV5 / MV7 / MV8: replay the versions through M'
    held: dict[int, int | None] = {rid: 0 for rid in source.resources}
    approx: set[int] = set()
    readers_of: dict[int, list] = {}
    order, writes, reads = _accesses(module)
    index = {c.id: i for i, c in enumerate(order)}
    for i, c in enumerate(order):
        if c.id in src_claims:
            o = src_claims[c.id][1]
            for a, b in zip(o.rd, c.rd):
                if _isolated(source, a):
                    continue
                want = versions.reads[(o.id, a)]
                if held.get(b) != want:
                    fail(
                        "MV4",
                        f"claim {o.id} reads version {held.get(b)} of resource {a}, not {want}",
                    )
                if b in approx:
                    readers_of.setdefault(b, []).append(o)
            for a, b in zip(o.wr, c.wr):
                if _isolated(source, a):
                    continue
                if a not in o.rd and not _whole_write(o, source.resources[a]):
                    want = versions.reads[(o.id, a)]
                    if held.get(b) != want:
                        fail(
                            "MV4",
                            f"claim {o.id} partially writes a copy holding version {held.get(b)} of {a}, not {want}",
                        )
                if spec.class_of(a) == "immutable":
                    fail("MV6", f"claim {o.id} writes immutable resource {a}")
                held[b] = versions.writes[(o.id, a)]
                approx.discard(b)
            continue
        mv = moves[c.id]
        if mv.kind == "rematerialized":
            p = src_claims[mv.producer][1]
            why = replay_safe(source, p)
            if why:
                fail("MV7", f"remat {c.id} replays producer {p.id}, which is {why}")
            want = versions.writes.get((p.id, mv.rid))
            if (
                want is None
                or versions.producer.get((mv.rid, mv.version)) != p.id
                or want != mv.version
            ):
                fail(
                    "MV7",
                    f"remat {c.id}: producer {p.id} did not produce version {mv.version} of {mv.rid}",
                )
            for (x, vx), b in zip(versions.inputs[p.id], c.rd):
                if held.get(b) != vx or b in approx:
                    fail(
                        "MV7",
                        f"remat {c.id} reads version {held.get(b)} of {x}, not the {vx} its producer read",
                    )
            if mv.cert != replay_certificate(source, versions, p.id, mv.rid, mv.dst_bank):
                fail("MV7", f"remat {c.id}'s certificate does not re-derive")
            held[c.wr[0]] = mv.version
            approx.discard(c.wr[0])
            continue
        src, dst = c.rd[0], c.wr[0]
        if held.get(src) is None:
            fail("MV4", f"move {c.id} reads copy {src} before anything landed in it")
        if src in approx:
            fail("MV8", f"move {c.id} moves an approximate copy onward")
        if held.get(src) != mv.version:
            fail(
                "MV11",
                f"move {c.id}'s edge says version {mv.version}, it moves version {held.get(src)}",
            )
        held[dst] = held.get(src)
        if mv.bits:
            if spec.codec_of(mv.rid) != mv.bits:
                fail("MV8", f"move {c.id} compresses resource {mv.rid} through an undeclared codec")
            approx.add(dst)
            readers_of.setdefault(dst, [])
        else:
            approx.discard(dst)
        if mv.coherence == "writeback":
            if dst != mv.rid:
                fail("MV5", f"writeback {c.id} does not land in resource {mv.rid}'s home copy")
    for rid in sorted(r for r in source.resources if not _isolated(source, r)):
        final = versions.final.get(rid, 0)
        if spec.class_of(rid) == "mutable" and final and held.get(rid) != final:
            fail(
                "MV5",
                f"resource {rid}'s home ends at version {held.get(rid)}, not its final {final}",
            )
    for cid, mv in moves.items():
        if not mv.bits:
            continue
        dst = mprime[cid][1].wr[0]
        rows = readers_of.get(dst, [])
        onward = [
            order[j].id
            for j in reads.get(dst, ())
            if j > index[cid] and order[j].id not in src_claims
        ]
        if onward:
            fail(
                "MV8", f"compressed copy {dst} feeds claims {onward[:4]} that are not source claims"
            )
        for o in rows:
            if not codec_admits(o, mv.bits):
                fail(
                    "MV8",
                    f"claim {o.id} does not tolerate resource {mv.rid} through a {mv.bits}-bit codec",
                )
        if mv.cert != accuracy_certificate(source, mv.rid, mv.bits, rows):
            fail("MV8", f"compressed move {cid}'s accuracy certificate does not re-derive")

    # MV11: the windows are the writer of the source and the first reader of the destination
    from ..gem.execution_plan import MOVE_HAS_AFTER, MOVE_HAS_BEFORE

    for cid, mv in moves.items():
        c = mprime[cid][1]
        remat = mv.kind == "rematerialized"
        after, before = _window(
            order, writes, reads, index[cid], None if remat else c.rd[0], c.wr[0]
        )
        got_after = mv.after_claim if mv.flags & MOVE_HAS_AFTER else None
        got_before = mv.before_claim if mv.flags & MOVE_HAS_BEFORE else None
        if (got_after, got_before) != (after, before):
            fail(
                "MV11",
                f"move {cid}'s window ({got_after}, {got_before}) is not the source's writer and the "
                f"destination's first reader ({after}, {before})",
            )

    # MV9: every bank holds its copies, and every reload was forced
    _capacity_laws(source, spec, module, plan, bank_of, moves, fail)
    # MV10: the deadline
    if spec.deadline and plan.makespan > spec.deadline:
        fail("MV10", f"the plan's makespan {plan.makespan} is past the deadline {spec.deadline}")
    return diags


def races(module: Module) -> list[str]:
    """Every pair of accesses to one resource, at least one a write, that neither the phase
    dependences (transitively) nor the in-phase order (one phase, ascending id) order -- the
    pairs whose version would depend on how the dispatch interleaves them. Empty for a
    well-formed transform, so the version replay may take any linearization."""
    from ..model import topological_phase_ids

    pmap = module.phase_map()
    before: dict[int, set] = {}
    for pid in topological_phase_ids(module):
        seen: set = set()
        for d in pmap[pid].deps:
            seen |= {d} | before.get(d, set())
        before[pid] = seen
    accesses: dict[int, list] = {}
    for pid, c in execution_order(module):
        for rid in c.rd:
            accesses.setdefault(rid, []).append((pid, c.id, False))
        for rid in c.wr:
            accesses.setdefault(rid, []).append((pid, c.id, True))
    out = []
    for rid, acc in sorted(accesses.items()):
        for i, (pa, ca, wa) in enumerate(acc):
            for pb, cb, wb in acc[i + 1 :]:
                if ca == cb or not (wa or wb) or pa == pb:
                    continue
                if pa in before[pb] or pb in before[pa]:
                    continue
                out.append(
                    f"claims {ca} and {cb} access resource {rid} in unordered phases {pa}, {pb}"
                )
    return out


def _capacity_laws(source, spec, module, plan, bank_of, moves, fail) -> None:
    """MV9: every lifetime fits its bank's allocatable bytes, and every reload of a version a
    bank already held was forced -- keeping the earlier copy across the gap would have exceeded
    the bank at some phase of it (the planner's own occupancy measure)."""
    for lt in plan.lifetimes:
        try:
            bank = spec.hardware.bank(lt.bank)
        except KeyError:
            fail("MV9", f"lifetime of {lt.rid} names undeclared bank {lt.bank!r}")
            continue
        if lt.offset + lt.size_bytes > bank.allocatable_bytes:
            fail("MV9", f"resource {lt.rid} ends past bank {lt.bank!r}'s allocatable bytes")
    pos = _positions(module)
    declared = {b.name for b in spec.hardware.banks}
    touched: dict[int, set] = {}
    for ph in module.phases:
        for c in ph.claims:
            for rid in (*c.rd, *c.wr):
                if bank_of.get(rid) in declared:
                    touched.setdefault(rid, set()).add(pos[ph.phase_id])
    live = {rid: (min(p), max(p)) for rid, p in touched.items()}
    size = {
        rid: _aligned_bytes(module.resources[rid], spec.hardware.bank(bank_of[rid])) for rid in live
    }
    occupancy: dict[tuple[str, int], int] = {}
    for rid, (first, last) in live.items():
        for p in range(first, last + 1):
            occupancy[(bank_of[rid], p)] = occupancy.get((bank_of[rid], p), 0) + size[rid]
    # the episodes of each (logical, bank), in order, with the version each one was filled with
    filled: dict[int, int] = {}
    for cid, mv in moves.items():
        if mv.kind != "rematerialized":
            filled[module_claim_dst(module, cid)] = mv.version
    episodes: dict[tuple[int, str], list] = {}
    for rid in live:
        if rid in source.resources or rid not in filled:
            continue
        lrid = next(mv.rid for cid, mv in moves.items() if module_claim_dst(module, cid) == rid)
        episodes.setdefault((lrid, bank_of[rid]), []).append(rid)
    for (lrid, bank), rids in episodes.items():
        rids.sort(key=lambda r: live[r])
        cap = spec.hardware.bank(bank).allocatable_bytes
        for a, b in zip(rids, rids[1:]):
            if filled[a] != filled[b]:
                continue  # a new version: nothing was thrown away
            gap = range(live[a][1] + 1, live[b][0])
            if not any(occupancy.get((bank, p), 0) + size[a] > cap for p in gap):
                fail(
                    "MV9",
                    f"resource {lrid} is reloaded into {bank!r} at version {filled[b]} though keeping "
                    f"its copy would have fit: an unforced reload (thrash)",
                )


def module_claim_dst(module: Module, cid: int) -> int:
    for ph in module.phases:
        for c in ph.claims:
            if c.id == cid:
                return c.wr[0]
    raise KeyError(cid)
