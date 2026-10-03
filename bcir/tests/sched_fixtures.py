"""PERF-SCHED's graders: the GEM schedulers, the phase-DAG traversals, the executor and the
verifier's per-claim laws as they were before the slice, kept verbatim, and the random programs
the faster rails are held to them over.

The slice restructured the hot loops of `gem.concurrency` (the wave indices, the conflict and
frontier predecessors, the fence layering), `gem.schedule` (the dispatch, the domain pick, the
slot record), `gem.async_tokens`, `gem.execute`, `model.graph` (the two iterative traversals,
the resource count) and `verify` (`_phase_laws`, `_claim_laws`, `_bounds_laws`, `_pair_laws`).
Each rail's result must be the one the historical code computed, on every program -- waves,
tail, affinity and contention; forks and awaits; every slot, the makespan and the residency it
came from; the dispatch order and the telemetry; the canonical phase order and the cycle verdict;
the verdict, diagnostic for diagnostic -- and where the historical code refused (a duplicate
claim id, a cyclic dispatch graph, a predecessor outside the dispatch), the same refusal. The
row is `gem.schedule.parity`: mismatches over every program and every rail, 0 when the slice is
exact.

The programs (`random_program`) are built to reach every branch of every rail: shared and
private resources, duplicated and overlapping read and write lists, fences (`barriered` and
`volatile`), the sparse tail (GGG lanes and random strides), unpriced and zero-cost claims,
one to eight affinity domains with and without the locality tie-break, multi-phase DAGs with
dangling and cyclic dependencies and duplicate phase ids, undeclared RIDs, isolated domains,
illegal lanes, hazards, bounds, extents, cost classes and the mask/unmask sub-law.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass

from bcir.gem.async_tokens import AsyncPlan
from bcir.gem.concurrency import ConcurrentSchedule, _claim_accesses, _is_sparse, is_fence
from bcir.gem.execute import ExecResult, PhaseStat
from bcir.gem.schedule import TAIL_STREAM, GemSchedule, Slot, _Checkpoint, _Record, _streams
from bcir.kbcir.cost import TargetProfile
from bcir.model import (
    ISOLATED_DOMAINS,
    Claim,
    Domain,
    Lane,
    Module,
    Opcode,
    Phase,
    Resource,
    StrideClass,
)
from bcir.verify import (
    _ATOMIC_OPCODES,
    _BOUNDS,
    _COST_CLASSES,
    _DATA_DEPENDENT,
    _HAZARDS,
    _I64_MAX,
    _LEGAL_LANES,
    _S_BOUNDS,
    _S_CLAIM_IDS,
    _S_COST,
    _S_DOMAIN,
    _S_EVENTS,
    _S_HAZARD,
    _S_LANE,
    _S_MASK,
    _S_PHASES,
    _S_REGISTRY,
    _S_RES_DOMAIN,
    _S_RESOLVE,
    _SECTIONS,
    _VERIFY,
    Diagnostic,
    _claim_id_laws,
    _event_laws,
    _pair_law,
    _registry_laws,
    _resource_domain_laws,
)
from bcir.verify import _is_sparse as _verify_is_sparse

# --- model.graph, before the slice -----------------------------------------------------------------


def count_reference(shape: tuple[int, ...]) -> int:
    """`Resource.count` before the slice."""
    n = 1
    for d in shape:
        n *= d
    return n


def topological_phase_ids_reference(module: Module) -> list[int]:
    pmap = module.phase_map()
    color: dict[int, int] = {}
    order: list[int] = []
    for phase in module.phases:
        root = phase.phase_id
        if color.get(root, 0) != 0:
            continue
        color[root] = 1
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            pid, next_dep = stack[-1]
            deps = pmap[pid].deps
            while next_dep < len(deps):
                dep = deps[next_dep]
                next_dep += 1
                stack[-1] = (pid, next_dep)
                if dep in pmap and color.get(dep, 0) == 0:
                    color[dep] = 1
                    stack.append((dep, 0))
                    break
            else:
                stack.pop()
                if color.get(pid) != 2:
                    color[pid] = 2
                    order.append(pid)
    return order


def phase_graph_has_cycle_reference(module: Module) -> bool:
    pmap = module.phase_map()
    color: dict[int, int] = {}
    for phase in module.phases:
        root = phase.phase_id
        if color.get(root, 0) != 0:
            continue
        color[root] = 1
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            pid, next_dep = stack[-1]
            deps = pmap[pid].deps
            while next_dep < len(deps):
                dep = deps[next_dep]
                next_dep += 1
                stack[-1] = (pid, next_dep)
                if dep not in pmap:
                    continue
                state = color.get(dep, 0)
                if state == 1:
                    return True
                if state == 0:
                    color[dep] = 1
                    stack.append((dep, 0))
                    break
            else:
                stack.pop()
                color[pid] = 2
    return False


_topo_phase_ids = topological_phase_ids_reference


# --- gem.concurrency, before the slice -------------------------------------------------------------


def _conflict_predecessors(claims: list[Claim]) -> dict[int, list[int]]:
    if len({claim.id for claim in claims}) != len(claims):
        raise ValueError("GEM dependency planning requires unique claim ids")
    readers: dict[int, list[int]] = {}
    writers: dict[int, list[int]] = {}
    position: dict[int, int] = {}
    out: dict[int, list[int]] = {}
    for index, claim in enumerate(claims):
        rd, wr = _claim_accesses(claim)
        predecessors: set[int] = set()
        for rid in rd:
            predecessors.update(writers.get(rid, ()))
        for rid in wr:
            predecessors.update(writers.get(rid, ()))
            predecessors.update(readers.get(rid, ()))
        out[claim.id] = sorted(predecessors, key=position.__getitem__)
        position[claim.id] = index
        for rid in rd:
            readers.setdefault(rid, []).append(claim.id)
        for rid in wr:
            writers.setdefault(rid, []).append(claim.id)
    return out


def _frontier_predecessors(claims: list[Claim]) -> dict[int, list[int]]:
    if len({claim.id for claim in claims}) != len(claims):
        raise ValueError("GEM dependency planning requires unique claim ids")
    last_writer: dict[int, int] = {}
    readers_since: dict[int, list[int]] = {}  # per resource: readers since its last write
    position: dict[int, int] = {}
    out: dict[int, list[int]] = {}
    for index, claim in enumerate(claims):
        rd, wr = _claim_accesses(claim)
        predecessors: set[int] = set()
        for rid in rd | wr:
            writer = last_writer.get(rid)
            if writer is not None:
                predecessors.add(writer)
        for rid in wr:
            predecessors.update(readers_since.get(rid, ()))
        out[claim.id] = sorted(predecessors, key=position.__getitem__)
        position[claim.id] = index
        for rid in rd - wr:  # a claim that also writes the resource is its new last writer
            readers_since.setdefault(rid, []).append(claim.id)
        for rid in wr:
            last_writer[rid] = claim.id
            readers_since[rid] = []
    return out


def hazard_predecessors_reference(claims: list[Claim]) -> dict[int, list[int]]:
    return _with_fences(claims, _conflict_predecessors(claims))


def hazard_frontier_reference(claims: list[Claim]) -> dict[int, list[int]]:
    return _with_fences(claims, _frontier_predecessors(claims))


def _with_fences(claims: list[Claim], out: dict[int, list[int]]) -> dict[int, list[int]]:
    position = {claim.id: index for index, claim in enumerate(claims)}
    last_fence: int | None = None
    segment: list[int] = []  # claims since the last fence (non-fences)
    for claim in claims:
        predecessors = out[claim.id]
        if is_fence(claim):
            extra = segment if last_fence is None else [last_fence, *segment]
            if extra:
                out[claim.id] = sorted(set(predecessors) | set(extra), key=position.__getitem__)
            last_fence = claim.id
            segment = []
        else:
            if last_fence is not None and last_fence not in predecessors:
                out[claim.id] = sorted(set(predecessors) | {last_fence}, key=position.__getitem__)
            segment.append(claim.id)
    return out


def _wave_indices(claims: list[Claim]) -> tuple[dict[int, int], dict[int, list[Claim]]]:
    if len({claim.id for claim in claims}) != len(claims):
        raise ValueError("GEM wave planning requires unique claim ids")
    reader_wave: dict[int, int] = {}
    writer_wave: dict[int, int] = {}
    wave_of: dict[int, int] = {}
    members: dict[int, list[Claim]] = {}
    for claim in claims:
        rd, wr = _claim_accesses(claim)
        wave = 0
        for rid in rd:
            if rid in writer_wave:
                wave = max(wave, writer_wave[rid] + 1)
        for rid in wr:
            if rid in writer_wave:
                wave = max(wave, writer_wave[rid] + 1)
            if rid in reader_wave:
                wave = max(wave, reader_wave[rid] + 1)
        wave_of[claim.id] = wave
        members.setdefault(wave, []).append(claim)
        for rid in rd:
            reader_wave[rid] = max(reader_wave.get(rid, -1), wave)
        for rid in wr:
            writer_wave[rid] = max(writer_wave.get(rid, -1), wave)
    return wave_of, members


def schedule_concurrent_reference(module: Module, target=None) -> ConcurrentSchedule:
    domains = max(1, getattr(target, "affinity_domains", 1)) if target is not None else 1
    sched = ConcurrentSchedule()
    pmap = module.phase_map()
    seen_claim_ids: set[int] = set()

    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        claim_ids = [claim.id for claim in claims]
        if len(set(claim_ids)) != len(claim_ids) or seen_claim_ids & set(claim_ids):
            raise ValueError("GEM scheduling requires module-wide unique claim ids")
        seen_claim_ids.update(claim_ids)
        main = [c for c in claims if not _is_sparse(c)]
        sched.ggg_tail.extend(c.id for c in claims if _is_sparse(c))

        # Greedy wave assignment within the phase (phase boundary = implicit barrier).
        wave_of, wave_members = _wave_indices(main)

        nwaves = (max(wave_of.values()) + 1) if wave_of else 0
        for w in range(nwaves):
            members = [c.id for c in wave_members.get(w, ())]
            for slot, cid in enumerate(members):
                sched.affinity[cid] = slot % domains
            if len(members) > domains:
                sched.contention += len(members) - domains
            sched.waves.append(members)

    return sched


# --- gem.async_tokens, before the slice ------------------------------------------------------------


def async_plan_reference(module: Module) -> AsyncPlan:
    pmap = module.phase_map()
    flat = []
    for pid in _topo_phase_ids(module):
        for c in sorted(pmap[pid].claims, key=lambda c: c.id):
            flat.append(c)

    plan = AsyncPlan(forks=[c.id for c in flat])
    plan.awaits = hazard_predecessors_reference(flat)
    return plan


# --- gem.schedule, before the slice ----------------------------------------------------------------


def _rids(c: Claim) -> set[int]:
    return set(c.rd) | set(c.wr)


def _pick_domain(
    rids: set[int],
    ready_t: int,
    dur: int,
    domain_free: list[int],
    resident: list[set[int]],
    eligible: range,
    locality: bool,
) -> tuple[int, int]:
    best_d = eligible[0]
    best_start = max(domain_free[best_d], ready_t)
    for d in eligible:
        start = max(domain_free[d], ready_t)
        if start < best_start:
            best_d, best_start = d, start
    if locality:
        best_score = len(rids & resident[best_d])
        for d in eligible:
            if d != best_d and max(domain_free[d], ready_t) == best_start:
                score = len(rids & resident[d])
                if score > best_score or (score == best_score and d < best_d):
                    best_d, best_score = d, score
    return best_d, best_start


class _PhaseDispatch:
    __slots__ = (
        "claims",
        "claim_by_id",
        "ordinal",
        "preds_of",
        "successors",
        "indegree0",
        "roots",
        "eligible",
        "rids",
        "domains",
        "locality",
    )

    def __init__(
        self,
        claims: list[Claim],
        preds: dict[int, list[int]],
        domains: int,
        knee: int,
        locality: bool,
    ) -> None:
        claim_by_id = {claim.id: claim for claim in claims}
        if len(claim_by_id) != len(claims):
            raise ValueError("GEM dispatch requires unique claim ids")
        self.claims = claims
        self.claim_by_id = claim_by_id
        self.ordinal = {claim.id: index for index, claim in enumerate(claims)}
        self.preds_of = {claim.id: preds.get(claim.id, ()) for claim in claims}  # no copy
        self.indegree0 = {claim.id: 0 for claim in claims}
        self.successors: dict[int, list[int]] = {claim.id: [] for claim in claims}
        for claim in claims:
            for predecessor in self.preds_of[claim.id]:
                if predecessor in claim_by_id:
                    self.indegree0[claim.id] += 1
                    self.successors[predecessor].append(claim.id)
        self.roots = [claim.id for claim in claims if self.indegree0[claim.id] == 0]
        tail = range(domains, domains + 1)  # the decoupled tail: its own stream, the same edges
        wave = {True: range(knee), False: range(domains)}  # bandwidth: the knee; compute: all
        self.eligible = {
            claim.id: tail if _is_sparse(claim) else wave[claim.cost_class == "bandwidth"]
            for claim in claims
        }
        self.rids = {claim.id: _rids(claim) for claim in claims}
        self.domains = domains
        self.locality = locality

    def run(
        self,
        durations: dict[int, int],
        t0: int,
        domain_free: list[int],
        resident: list[set[int]],
        finish_of: dict[int, int],
        sched: GemSchedule | None,
        resume: "_Checkpoint | None" = None,
        record: "_Record | None" = None,
    ) -> int:
        if len(domain_free) != self.domains + 1 or len(resident) != self.domains + 1:
            raise ValueError("GEM dispatch needs one stream per affinity domain plus the tail")
        for claim in self.claims:
            for predecessor in self.preds_of[claim.id]:
                if predecessor not in self.claim_by_id and predecessor not in finish_of:
                    raise ValueError(
                        f"GEM dispatch predecessor {predecessor} for claim {claim.id} "
                        "is unavailable"
                    )

        def duration_of(claim_id: int) -> int:
            return max(0, durations.get(claim_id, 1))  # an unpriced claim takes a unit

        ordinal, successors, preds_of = self.ordinal, self.successors, self.preds_of
        eligible, rids_of, domains, locality = self.eligible, self.rids, self.domains, self.locality
        if resume is None:
            indegree = dict(self.indegree0)
            ready = [(-duration_of(cid), cid, ordinal[cid]) for cid in self.roots]
            heapq.heapify(ready)  # the key is total (ids are unique): the pop order is the same
            pops = 0
        else:
            indegree = dict(resume.indegree)
            ready = list(resume.ready)
            pops = resume.pops
        stride = record.stride if record is not None and record.checkpoints is not None else 0
        while ready:
            if stride and pops and pops % stride == 0:
                record.checkpoints.append(
                    _Checkpoint(
                        pops,
                        list(ready),
                        dict(indegree),
                        list(domain_free),
                        [set(s) for s in resident],
                        dict(finish_of),
                    )
                )
            key = heapq.heappop(ready)
            claim_id = key[1]
            if record is not None:
                record.pop_index[claim_id] = pops
                record.pop_keys.append(key)
            dur = duration_of(claim_id)
            ready_t = t0
            for p in preds_of[claim_id]:
                if finish_of[p] > ready_t:
                    ready_t = finish_of[p]
            rids = rids_of[claim_id]
            d, start = _pick_domain(
                rids, ready_t, dur, domain_free, resident, eligible[claim_id], locality
            )
            finish = start + dur
            domain_free[d] = finish
            resident[d] |= rids
            finish_of[claim_id] = finish
            if sched is not None:
                stream = TAIL_STREAM if d == domains else d
                sched.slots.append(Slot(claim_id, stream, start, finish))
                sched.affinity[claim_id] = stream
            pops += 1
            for successor in successors[claim_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    if record is not None:
                        record.push_index[successor] = pops
                    heapq.heappush(ready, (-duration_of(successor), successor, ordinal[successor]))
        if pops != len(self.claims):
            raise ValueError("GEM dispatch dependency graph is cyclic")
        return pops


def _dispatch(
    claims: list[Claim],
    preds: dict[int, list[int]],
    durations: dict[int, int],
    t0: int,
    domain_free: list[int],
    resident: list[set[int]],
    domains: int,
    knee: int,
    locality: bool,
    finish_of: dict[int, int],
    sched: GemSchedule,
) -> None:
    _PhaseDispatch(claims, preds, domains, knee, locality).run(
        durations, t0, domain_free, resident, finish_of, sched
    )


def phase_hazards_reference(
    module: Module, *, frontier: bool = False
) -> dict[int, dict[int, list[int]]]:
    build = hazard_frontier_reference if frontier else hazard_predecessors_reference
    pmap = module.phase_map()
    out: dict[int, dict[int, list[int]]] = {}
    seen_claim_ids: set[int] = set()
    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        claim_ids = [claim.id for claim in claims]
        if len(set(claim_ids)) != len(claim_ids) or seen_claim_ids & set(claim_ids):
            raise ValueError("GEM scheduling requires module-wide unique claim ids")
        seen_claim_ids.update(claim_ids)
        out[pid] = build(claims)
    return out


def schedule_eft_reference(
    module: Module,
    durations: dict[int, int],
    target=None,
    locality: bool = True,
    hazards: dict[int, dict[int, list[int]]] | None = None,
) -> GemSchedule:
    domains, knee = _streams(target)
    sched = GemSchedule(mode="eft", knee=knee)
    resident: list[set[int]] = [set() for _ in range(domains + 1)]
    pmap = module.phase_map()
    t0 = 0
    if hazards is None:
        hazards = phase_hazards_reference(module, frontier=True)

    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        # The intra-phase hazard DAG over main AND tail claims, before the stream split
        # (the lower claim id is the producer of a conflicting pair).
        preds = hazards[pid]
        domain_free = [t0] * (domains + 1)
        finish_of: dict[int, int] = {}
        _dispatch(
            claims,
            preds,
            durations,
            t0,
            domain_free,
            resident,
            domains,
            knee,
            locality,
            finish_of,
            sched,
        )
        t0 = max([t0] + list(finish_of.values()))
    sched.makespan = t0
    return sched


# --- gem.execute, before the slice -----------------------------------------------------------------


def execute_reference(module: Module, kernels=None, deterministic: bool = True) -> ExecResult:
    pmap = module.phase_map()
    result = ExecResult()
    for pid in _topo_phase_ids(module):
        phase = pmap[pid]
        claims = list(phase.claims)
        if deterministic:
            claims.sort(key=lambda c: c.id)
        stat = PhaseStat(phase_id=pid, scheduled=len(claims))
        result.phase_order.append(pid)
        for claim in claims:
            result.order.append(claim.id)
            if kernels and claim.id in kernels:
                kernels[claim.id]()
            stat.executed += 1
            result.executed += 1
        result.phases.append(stat)
    return result


# --- verify, before the slice ----------------------------------------------------------------------


def verify_reference(module: Module) -> list[Diagnostic]:
    from bcir.kbcir.events import mask_law

    sections: list[list] = [[] for _ in range(_SECTIONS)]
    sections[_S_REGISTRY] = _registry_laws(module)
    sections[_S_CLAIM_IDS] = _claim_id_laws(module)
    sections[_S_RES_DOMAIN] = _resource_domain_laws(module)
    sections[_S_PHASES] = _phase_laws(module)
    hazard = sections[_S_HAZARD]
    for ph in module.phases:
        for claim in ph.claims:
            for section, diag in _claim_laws(module, claim, mask_law):
                sections[section].append(diag)
        hazard += _pair_laws(ph)
    sections[_S_EVENTS] = _event_laws(module)
    return [diag for section in sections for diag in section]


def _phase_laws(module: Module) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    seen_pid: set[int] = set()
    for ph in module.phases:
        if ph.phase_id in seen_pid:
            diags.append(Diagnostic("R4", f"duplicate phase id {ph.phase_id}"))
        seen_pid.add(ph.phase_id)
    declared_pids = {ph.phase_id for ph in module.phases}
    for ph in module.phases:
        for dep in ph.deps:
            if dep not in declared_pids:
                diags.append(
                    Diagnostic("R4", f"phase {ph.phase_id} depends on undeclared phase {dep}")
                )
    # R4: phase DAG legality (acyclic).
    if phase_graph_has_cycle_reference(module):
        diags.append(Diagnostic("R4", "phase dependency graph contains a cycle"))
    return diags


def _claim_laws(module: Module, claim, mask_law) -> list[tuple[int, Diagnostic]]:
    out: list[tuple[int, Diagnostic]] = []
    resources = module.resources
    rd, wr = claim.rd, claim.wr
    reads = [(rid, resources[rid] if rid in resources else None) for rid in rd]
    writes = [(rid, resources[rid] if rid in resources else None) for rid in wr]

    # R2: registry resolution -- every claim resource reference resolves.
    for rid, res in reads + writes:
        if res is None:
            out.append(
                (_S_RESOLVE, Diagnostic("R2", f"claim {claim.id} references undeclared RID {rid}"))
            )

    # R3: domain legality -- claim domain contracts correspond to registry placement.
    touched = [res for _, res in reads + writes if res is not None]
    if touched and claim.domain not in {r.domain for r in touched}:
        out.append(
            (
                _S_DOMAIN,
                Diagnostic(
                    "R3",
                    f"claim {claim.id}: declares domain {claim.domain.name} but touches only "
                    f"{{{', '.join(sorted({r.domain.name for r in touched}))}}}",
                ),
            )
        )
    for kind, resolved in (("read", reads), ("write", writes)):
        for rid, res in resolved:
            if res is not None and res.domain in ISOLATED_DOMAINS and res.domain != claim.domain:
                out.append(
                    (
                        _S_DOMAIN,
                        Diagnostic(
                            "R3",
                            f"claim {claim.id}: {kind} of RID {rid} (domain {res.domain.name}) "
                            f"does not match the claim domain {claim.domain.name} -- an "
                            f"isolated resource may not be reached as another address space",
                        ),
                    )
                )
    for rid, res in writes:
        if res is not None and res.domain == Domain.MMIO and claim.hazard == "unique":
            out.append(
                (
                    _S_DOMAIN,
                    Diagnostic(
                        "R3",
                        f"claim {claim.id}: MMIO write to RID {rid} requires an "
                        f"atomic/barriered hazard contract",
                    ),
                )
            )

    # R5: hazard legality -- the hazard contract is sufficient for the declared semantics.
    if claim.hazard not in _HAZARDS:
        out.append(
            (
                _S_HAZARD,
                Diagnostic("R5", f"claim {claim.id}: unknown hazard contract {claim.hazard!r}"),
            )
        )
    else:
        if claim.opcode in _ATOMIC_OPCODES and claim.hazard == "unique":
            out.append(
                (
                    _S_HAZARD,
                    Diagnostic(
                        "R5",
                        f"claim {claim.id}: atomic opcode {claim.opcode.name} requires an "
                        f"atomic/barriered hazard contract",
                    ),
                )
            )
        if claim.lane == Lane.A and claim.hazard == "unique":
            out.append(
                (
                    _S_HAZARD,
                    Diagnostic(
                        "R5",
                        f"claim {claim.id}: atomic lane A requires an atomic/barriered "
                        f"hazard contract",
                    ),
                )
            )
        if claim.callee_sig and "(" not in claim.callee_sig:
            out.append(
                (
                    _S_HAZARD,
                    Diagnostic(
                        "R18",
                        f"claim {claim.id}: malformed indirect-callee signature "
                        f"{claim.callee_sig!r} (expected 'ret(params)')",
                    ),
                )
            )
        if claim.volatile and claim.hazard == "unique":
            out.append(
                (
                    _S_HAZARD,
                    Diagnostic(
                        "R5",
                        f"claim {claim.id}: volatile access requires an atomic/barriered "
                        f"hazard contract",
                    ),
                )
            )

    # R6: lane legality -- lane type matches the declared access pattern.
    legal = _LEGAL_LANES.get(claim.stride_class, set())
    if claim.lane not in legal:
        out.append(
            (
                _S_LANE,
                Diagnostic(
                    "R6",
                    f"claim {claim.id}: lane {claim.lane.name} illegal for "
                    f"stride_class {claim.stride_class.name}",
                ),
            )
        )

    # R7: bounds legality.
    for diag in _bounds_laws(claim, reads, writes):
        out.append((_S_BOUNDS, diag))

    # R8 (static half): cost completeness -- every claim names a known cost class.
    if claim.cost_class not in _COST_CLASSES:
        out.append(
            (
                _S_COST,
                Diagnostic("R8", f"claim {claim.id}: unknown cost class {claim.cost_class!r}"),
            )
        )

    # EV (driver roadmap A1/B1): the mask/unmask well-formedness sub-law.
    msg = mask_law(claim)
    if msg is not None:
        law, _, text = msg.partition(":")
        out.append((_S_MASK, Diagnostic(law.strip(), text.strip())))
    return out


def _bounds_laws(claim, reads, writes) -> list[Diagnostic]:
    if claim.bounds not in _BOUNDS:
        return [Diagnostic("R7", f"claim {claim.id}: unknown bounds mode {claim.bounds!r}")]
    if claim.verify not in _VERIFY:
        return [Diagnostic("R7", f"claim {claim.id}: unknown verify contract {claim.verify!r}")]
    diags: list[Diagnostic] = []
    if claim.count < 0:
        diags.append(
            Diagnostic("R7", f"claim {claim.id}: count must be non-negative (got {claim.count})")
        )
    if claim.offset < 0:
        diags.append(
            Diagnostic("R7", f"claim {claim.id}: offset must be non-negative (got {claim.offset})")
        )
    if claim.stride_k < 1:
        diags.append(
            Diagnostic("R7", f"claim {claim.id}: stride_k must be positive (got {claim.stride_k})")
        )
    if diags:
        return diags
    if claim.bounds == "masked" and claim.verify != "bounds":
        why = f" (extent provenance: {claim.bounds_provenance})" if claim.bounds_provenance else ""
        diags.append(
            Diagnostic(
                "R7",
                f"claim {claim.id}: masked (runtime-bounds-checked) access must carry a "
                f"'bounds' verify contract, not {claim.verify!r}{why}",
            )
        )
    if claim.bounds != "strict":
        return diags
    if claim.stride_class in _DATA_DEPENDENT:
        if claim.verify == "none":
            diags.append(
                Diagnostic(
                    "R7",
                    f"claim {claim.id}: data-dependent {claim.stride_class.name} access "
                    f"with strict bounds requires a runtime verify contract",
                )
            )
        return diags
    k = claim.stride_k
    read_extent = claim.offset + (claim.count - 1) * k + 1 if claim.count > 0 else 0
    is_reduction = claim.op.startswith("reduce.")
    write_extent = claim.offset + (1 if is_reduction else claim.count)
    if max(read_extent, write_extent) > _I64_MAX:
        diags.append(
            Diagnostic("R7", f"claim {claim.id}: affine access extent exceeds signed 64-bit range")
        )
        return diags
    for resolved, extent, kind in ((reads, read_extent, "read"), (writes, write_extent, "write")):
        for rid, res in resolved:
            if res is None or not res.shape:
                continue
            if extent > count_reference(res.shape):
                diags.append(
                    Diagnostic(
                        "R7",
                        f"claim {claim.id}: {kind} of RID {rid} overruns the resource "
                        f"(extent {extent} > {count_reference(res.shape)})",
                    )
                )
    return diags


def _pair_laws(ph) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    claims = ph.claims
    is_sp = [_verify_is_sparse(c) for c in claims]
    if any(is_sp):
        for i, a in enumerate(claims):
            a_sp = is_sp[i]
            for j in range(i + 1, len(claims)):
                if not (a_sp or is_sp[j]):
                    continue
                diags += _pair_law(ph.phase_id, a, claims[j])
    return diags


# --- the random programs ----------------------------------------------------------------------------


@dataclass
class Program:
    """One graded program: the module, the target (or None), the durations, the locality
    switch and the claim ids whose kernels the executor is handed."""

    module: Module
    target: TargetProfile | None
    durations: dict[int, int]
    locality: bool
    kernels: tuple[int, ...]


_OPCODES = (
    Opcode.ADD,
    Opcode.MUL,
    Opcode.LOAD,
    Opcode.STORE,
    Opcode.ATOMIC_ADD,
    Opcode.CMPXCHG,
    Opcode.NOP,
    Opcode.BARRIER,
)
_LANES = (Lane.U, Lane.U, Lane.U, Lane.UX, Lane.T, Lane.GGG, Lane.A, Lane.H)
_STRIDES = (
    StrideClass.UNIT,
    StrideClass.UNIT,
    StrideClass.SCALAR,
    StrideClass.STRIDED,
    StrideClass.CACHELINE,
    StrideClass.TILE,
    StrideClass.RANDOM,
)
_HAZARD_SPELLINGS = ("unique", "unique", "unique", "atomic", "barriered", "fenced")
_BOUNDS_SPELLINGS = ("strict", "strict", "strict", "masked", "assumed_safe", "loose")
_VERIFY_SPELLINGS = ("bounds", "bounds", "none", "exact", "hash", "crc")
_COST_SPELLINGS = ("bandwidth", "bandwidth", "compute", "latency", "memory")
_OPS = ("vector.add", "vector.add", "vector.mul", "reduce.sum", "irq.mask:pic", "c.call.indirect")
_DOMAINS = (Domain.RAM, Domain.RAM, Domain.RAM, Domain.HBM, Domain.MMIO)
_SHAPES = ((), (1,), (8,), (64,), (4, 4), (0,), (1 << 40, 1 << 24))


def _rid_list(rng: random.Random, rids: list[int], most: int) -> tuple[int, ...]:
    """Up to `most` RIDs with the odd repeat and the odd undeclared one."""
    out = [rng.choice(rids) for _ in range(rng.randint(0, most))]
    if out and rng.random() < 0.15:
        out.append(out[0])  # a repeated RID
    if rng.random() < 0.08:
        out.append(97)  # an undeclared RID
    return tuple(out)


def random_program(seed: int) -> Program:
    """The program of `seed`: see the module docstring for what it reaches."""
    rng = random.Random(seed)
    module = Module(name=f"sched{seed}")
    rids = list(range(1, rng.randint(2, 9)))
    for rid in rids:
        kw = {}
        if rng.random() < 0.05:
            kw["align"] = 48
        if rng.random() < 0.05:
            kw["access"] = "ham"
        module.add_resource(
            Resource(rid, rng.choice(_DOMAINS), shape=rng.choice(_SHAPES), name=f"r{rid}", **kw)
        )
    phase_count = rng.randint(1, 6)
    phase_ids = list(range(phase_count))
    if rng.random() < 0.1:
        phase_ids[-1] = phase_ids[0]  # a duplicate phase id
    cid = 1
    kernels: list[int] = []
    durations: dict[int, int] = {}
    for index, pid in enumerate(phase_ids):
        deps: list[int] = []
        for other in range(phase_count):
            if other != index and rng.random() < 0.3:
                deps.append(phase_ids[other])  # earlier or later: a cycle is possible
        if rng.random() < 0.08:
            deps.append(99)  # a dangling dependency
        claims = []
        for _ in range(rng.randint(0, 12)):
            fence = rng.random() < 0.06
            claim = Claim(
                cid,
                rng.choice(_OPCODES),
                rng.choice(_LANES),
                rng.choice(_STRIDES),
                count=rng.choice((-1, 0, 1, 8, 17, 64)),
                stride_k=rng.choice((0, 1, 1, 1, 2, 3)),
                rd=_rid_list(rng, rids, 3),
                wr=_rid_list(rng, rids, 2),
                hazard="barriered" if fence else rng.choice(_HAZARD_SPELLINGS),
                domain=rng.choice(_DOMAINS),
                verify=rng.choice(_VERIFY_SPELLINGS),
                bounds=rng.choice(_BOUNDS_SPELLINGS),
                op=rng.choice(_OPS),
                offset=rng.choice((-1, 0, 0, 0, 1, 4)),
                cost_class=rng.choice(_COST_SPELLINGS),
                volatile=rng.random() < 0.04,
                callee_sig=rng.choice(("", "", "", "i32(i32)", "bad")),
                bounds_provenance=rng.choice(("", "", "declared_extent")),
            )
            claims.append(claim)
            if rng.random() < 0.9:
                durations[cid] = rng.choice((0, 1, 1, 2, 3, 5, 8, 13, 21))
            if rng.random() < 0.3:
                kernels.append(cid)
            cid += 1
        if rng.random() < 0.03 and claims:
            claims.append(claims[0])  # a duplicate claim id: every scheduler must refuse
        if rng.random() < 0.5:
            rng.shuffle(claims)  # declared out of id order
        module.add_phase(Phase(pid, tuple(deps), claims))
    target = None
    if rng.random() < 0.85:
        target = TargetProfile(
            name="sched", affinity_domains=rng.randint(1, 8), mem_channels=rng.randint(1, 6)
        )
    return Program(module, target, durations, rng.random() < 0.7, tuple(kernels))


def outcome(fn, *args, **kwargs) -> tuple:
    """`fn(*args, **kwargs)` as a comparable value: the result, or the refusal's type and
    message."""
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 -- the grader compares every outcome, crashes included
        return ("raise", type(exc).__name__, str(exc))


def _waves(sched: ConcurrentSchedule) -> tuple:
    return (sched.waves, sched.ggg_tail, sched.affinity, sched.contention)


def _plan(plan: AsyncPlan) -> tuple:
    return (plan.forks, plan.awaits)


def _eft(sched: GemSchedule) -> tuple:
    slots = [(s.claim_id, s.domain, s.start, s.finish) for s in sched.slots]
    return (sched.mode, slots, sched.makespan, sched.knee, sched.affinity)


def _exec(result: ExecResult) -> tuple:
    stats = [(p.phase_id, p.scheduled, p.executed) for p in result.phases]
    return (result.order, result.phase_order, stats, result.executed)


def _executed(fn, module: Module, kernel_ids: tuple[int, ...], deterministic: bool) -> tuple:
    """The executor's result and the order its kernels were invoked in."""
    log: list[int] = []
    kernels = {cid: (lambda cid=cid: log.append(cid)) for cid in kernel_ids}
    out = outcome(fn, module, kernels, deterministic)
    if out[0] == "ok":
        return ("ok", _exec(out[1]), log)
    return out


def rails(program: Program) -> list[tuple[str, tuple, tuple]]:
    """Every (rail, reference outcome, outcome) of one program."""
    from bcir.gem.async_tokens import async_plan
    from bcir.gem.concurrency import hazard_frontier, hazard_predecessors, schedule_concurrent
    from bcir.gem.execute import execute
    from bcir.gem.schedule import EftPlacer, phase_hazards, schedule_eft
    from bcir.model import phase_graph_has_cycle, topological_phase_ids
    from bcir.verify import verify

    m, target, durations, locality = (
        program.module,
        program.target,
        program.durations,
        program.locality,
    )
    out: list[tuple[str, tuple, tuple]] = []
    out.append(
        ("order", outcome(topological_phase_ids_reference, m), outcome(topological_phase_ids, m))
    )
    out.append(
        ("cycle", outcome(phase_graph_has_cycle_reference, m), outcome(phase_graph_has_cycle, m))
    )
    out.append(("verify", outcome(verify_reference, m), outcome(verify, m)))
    for deterministic in (True, False):
        out.append(
            (
                f"execute[{deterministic}]",
                _executed(execute_reference, m, program.kernels, deterministic),
                _executed(execute, m, program.kernels, deterministic),
            )
        )
    ref = outcome(schedule_concurrent_reference, m, target)
    new = outcome(schedule_concurrent, m, target)
    out.append(
        (
            "waves",
            ref if ref[0] != "ok" else ("ok", _waves(ref[1])),
            new if new[0] != "ok" else ("ok", _waves(new[1])),
        )
    )
    ref = outcome(async_plan_reference, m)
    new = outcome(async_plan, m)
    out.append(
        (
            "tokens",
            ref if ref[0] != "ok" else ("ok", _plan(ref[1])),
            new if new[0] != "ok" else ("ok", _plan(new[1])),
        )
    )
    for pid, phase in m.phase_map().items():
        claims = sorted(phase.claims, key=lambda c: c.id)
        out.append(
            (
                f"dag[{pid}]",
                outcome(hazard_predecessors_reference, claims),
                outcome(hazard_predecessors, claims),
            )
        )
        out.append(
            (
                f"frontier[{pid}]",
                outcome(hazard_frontier_reference, claims),
                outcome(hazard_frontier, claims),
            )
        )
    for hazards in (None, "full"):
        label = "eft" if hazards is None else "eft[full]"
        ref = outcome(
            schedule_eft_reference,
            m,
            durations,
            target,
            locality,
            None if hazards is None else outcome(phase_hazards_reference, m)[1],
        )
        new = outcome(
            schedule_eft,
            m,
            durations,
            target,
            locality,
            None if hazards is None else outcome(phase_hazards, m)[1],
        )
        out.append(
            (
                label,
                ref if ref[0] != "ok" else ("ok", _eft(ref[1])),
                new if new[0] != "ok" else ("ok", _eft(new[1])),
            )
        )
    ref = outcome(schedule_eft_reference, m, durations, target, locality)
    new = outcome(lambda: EftPlacer(m, durations, target, locality).schedule())
    out.append(
        (
            "placer",
            ref if ref[0] != "ok" else ("ok", _eft(ref[1])),
            new if new[0] != "ok" else ("ok", _eft(new[1])),
        )
    )
    return out


def measure(seen: dict | None = None, seeds: int = 400) -> dict[str, float]:
    """The row (module docstring), over `seeds` programs and every rail of each."""
    mismatches = 0
    stats = {
        "programs": 0,
        "rails": 0,
        "refused": 0,
        "laws": set(),
        "fenced": 0,
        "sparse": 0,
        "cyclic": 0,
        "multi_phase": 0,
        "domains": set(),
    }
    for seed in range(seeds):
        program = random_program(seed)
        stats["programs"] += 1
        claims = [c for ph in program.module.phases for c in ph.claims]
        stats["fenced"] += any(is_fence(c) for c in claims)
        stats["sparse"] += any(_is_sparse(c) for c in claims)
        stats["cyclic"] += phase_graph_has_cycle_reference(program.module)
        stats["multi_phase"] += len(program.module.phases) > 1
        stats["domains"].add(_streams(program.target)[0])
        for rail, ref, new in rails(program):
            stats["rails"] += 1
            if ref[0] == "raise":
                stats["refused"] += 1
            if rail == "verify" and ref[0] == "ok":
                stats["laws"].update(d.law for d in ref[1])
            if new != ref:
                mismatches += 1
                if seen is not None:
                    seen.setdefault("mismatched", []).append((seed, rail, ref, new))
    if seen is not None:
        seen.update(stats)
    return {"gem.schedule.parity": float(mismatches)}


__all__ = [
    "Program",
    "async_plan_reference",
    "count_reference",
    "execute_reference",
    "hazard_frontier_reference",
    "hazard_predecessors_reference",
    "measure",
    "outcome",
    "phase_graph_has_cycle_reference",
    "phase_hazards_reference",
    "rails",
    "random_program",
    "schedule_concurrent_reference",
    "schedule_eft_reference",
    "topological_phase_ids_reference",
    "verify_reference",
]
