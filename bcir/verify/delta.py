"""GEM+ G18 (S4-B): incremental re-verification -- the chain's verdict, re-derived per unit.

The K_BCIR -> StreamPack chain's verdict is

    verify(module) + verify_plan(module, result, h, theta=theta, policy=policy)
                   + verify_pack(module, pack)

and all three verifiers assemble it from units (`verify._claim_laws` per claim, `_pair_law` per
same-phase pair, `_step_laws` and `_cost_law` per plan step, `_segment_laws` per segment,
`_prefetch_law` per prefetch, and a few module-, plan- and pack-wide laws). `VerifyState` keeps
every unit's verdict and advances by what changed, re-deriving only the units whose inputs moved
and re-assembling the verdict in the verifiers' own order:

  * a claim's laws read the claim and the registry entries it names -- re-derived for every
    claim object that changed and every claim naming a resource that changed;
  * a decoupled-tail pair reads its two claims -- re-derived for the pairs that touch a claim
    that changed;
  * a step's laws read the step, its claim, the phase declaring it and its claim's offer --
    re-derived for every step object that changed and every step whose claim or claim's offer
    rows moved. The offer is the verifier's OWN (`kbcir.delta.IncrementalOffer`, advanced from
    the module alone): the plan being verified never supplies the offer it is judged against;
  * a step's cost re-derivation also reads the step before it -- re-derived for each changed
    step and the next known step;
  * a segment's laws read the segment, the trace notes' claim ids, the prefetch targets by name
    and the registry -- re-derived for every changed segment and every segment whose claim's
    trace status or whose prefetch moved;
  * the module-, plan- and pack-wide laws (the registry, coverage, the score, the phase order,
    duplicate names, the header, the generation vector) are kept as counts and re-derived when
    what they read moved.

What changed is found by object IDENTITY between the old and the new module, plan and pack --
never by asking the planner or the emitter what they changed, so a forged or mis-reported change
is re-verified like any other (the instrument does not take its subject's word, L9).

`verify.delta.identity` holds `VerifyState.apply` to the full verdict over the corpus, including
every injected violation and every forgery.

Declared boundaries (v0): the chain's delta shape -- the same phases (ids, dependencies, events),
each claim id once and at the same position, the same RIDs, a plan whose steps keep their claim
ids and a pack whose segments and trace notes keep their count. Anything else, and a module with
event phases whose claims changed (EV3 reads the whole program flow), is verified from scratch
(`rebuilds` counts it). A claim or resource mutated IN PLACE is not a change an identity diff can
see; that is the S1-B boundary, stated in `kbcir.delta`.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from itertools import compress
from operator import is_not, not_

from ..model import Module
from . import (
    _DUPLICATE_PREFETCHES,
    _DUPLICATE_SEGMENTS,
    _DUPLICATE_TRACES,
    _S_BOUNDS,
    _S_COST,
    _S_DOMAIN,
    _S_HAZARD,
    _S_LANE,
    _S_MASK,
    _S_RESOLVE,
    Diagnostic,
    _claim_id_laws,
    _claim_laws,
    _cost_law,
    _CostScope,
    _event_laws,
    _is_sparse,
    _pack_generation_laws,
    _pack_header_laws,
    _pair_law,
    _phase_laws,
    _phase_order_diag,
    _phase_position,
    _prefetch_law,
    _registry_laws,
    _repeated_step_diag,
    _resource_domain_laws,
    _score_diag,
    _segment_laws,
    _step_laws,
    _topo_phase_ids,
    _uncovered_claim_diag,
    _uncovered_segment_diag,
    _unknown_step_diag,
)

__all__ = ["VerifyState"]


class _Sparse:
    """Payloads keyed by an orderable key, only the non-empty ones kept, read in key order."""

    __slots__ = ("items", "keys")

    def __init__(self):
        self.items: dict = {}
        self.keys: list = []

    def put(self, key, payload) -> None:
        if payload:
            if key not in self.items:
                insort(self.keys, key)
            self.items[key] = payload
        elif key in self.items:
            del self.items[key]
            del self.keys[bisect_left(self.keys, key)]

    def ordered(self):
        items = self.items
        return (items[k] for k in self.keys)

    def between(self, lo, hi):
        """The payloads of keys in [lo, hi), in key order."""
        keys, items = self.keys, self.items
        return (items[k] for k in keys[bisect_left(keys, lo) : bisect_left(keys, hi)])


class _Tally:
    """A multiset that knows, without a scan, whether any element occurs twice (the R10
    duplicate laws) and which elements occur at all."""

    __slots__ = ("counts", "repeated")

    def __init__(self, items=()):
        self.counts: dict = {}
        self.repeated = 0  # elements with count > 1
        for x in items:
            self.add(x)

    def add(self, x) -> None:
        n = self.counts.get(x, 0) + 1
        self.counts[x] = n
        if n == 2:
            self.repeated += 1

    def remove(self, x) -> None:
        n = self.counts[x] - 1
        if n == 1:
            self.repeated -= 1
        if n:
            self.counts[x] = n
        else:
            del self.counts[x]

    def __contains__(self, x) -> bool:
        return x in self.counts


class _Structure(Exception):
    """The new artifacts are not the old ones' delta shape: verify from scratch."""


class VerifyState:
    """The chain's verdict for (module, result, pack) under (h, theta, policy), kept per unit
    and advanced by `apply`. `diagnostics` is always the full verdict of the current three."""

    @classmethod
    def build(cls, module: Module, result, pack, h, theta, policy=None) -> "VerifyState":
        self = cls.__new__(cls)
        self.h, self.theta, self.policy = h, theta, policy
        self.rebuilds = 0
        self._build(module, result, pack)
        return self

    # --- construction ------------------------------------------------------------------------

    def _build(self, module: Module, result, pack) -> None:
        from ..kbcir.delta import IncrementalOffer, _unique_where
        from ..kbcir.events import mask_law

        self._mask_law = mask_law
        self.module, self.result, self.pack = module, result, pack
        self._scope = _CostScope(self.h, self.theta, self.policy)
        # -- the module --------------------------------------------------------------------
        phases = module.phases
        self._phases = list(phases)
        self._phase_start: list[int] = []
        self._phase_of: list[int] = []
        flat: list = []
        for pi, ph in enumerate(phases):
            self._phase_start.append(len(flat))
            flat.extend(ph.claims)
            self._phase_of.extend([pi] * len(ph.claims))
        self._phase_start.append(len(flat))
        self._flat = flat
        _unique_where(module)  # a delta addresses a claim by its id: verify such a module whole
        self._claims_by_id = {c.id: c for c in flat}
        self._declared_in = {c.id: ph.phase_id for ph in phases for c in ph.claims}
        self._events = any(getattr(ph, "event", "") for ph in phases)
        self._registry = _registry_laws(module)
        self._claim_ids = _claim_id_laws(module)
        self._res_domain = _resource_domain_laws(module)
        self._phase_diags = _phase_laws(module)
        self._event_diags = _event_laws(module)
        self._users: dict = {}  # RID -> claim indexes naming it
        self._units = _Sparse()  # claim index -> [(section, diag)]
        for i, claim in enumerate(flat):
            self._units.put(i, _claim_laws(module, claim, mask_law))
            for rid in claim.io_rids():
                self._users.setdefault(rid, set()).add(i)
        self._pairs: list[_Sparse] = []
        self._pair_keys: list[dict] = []
        self._sparse: list[list[int]] = []
        for pi, ph in enumerate(phases):
            self._pairs.append(_Sparse())
            self._pair_keys.append({})
            sparse = [i for i, c in enumerate(ph.claims) if _is_sparse(c)]
            self._sparse.append(sparse)
            if sparse:
                claims = ph.claims
                for i in range(len(claims)):
                    self._repair(pi, i, claims, sparse, lower_only=True)
        # -- the plan ----------------------------------------------------------------------
        self._offer = IncrementalOffer(module, self.h) if self.h is not None else None
        self._topo = {pid: i for i, pid in enumerate(_topo_phase_ids(module))}
        steps = result.steps
        self._steps = list(steps)
        occ: dict = {}
        for s, step in enumerate(steps):
            occ.setdefault(step.claim_id, []).append(s)
        self._occ = occ
        self._known = [step.claim_id in self._claims_by_id for step in steps]
        self._step_units = _Sparse()
        self._cost_units = _Sparse()
        self._total = 0
        for s in range(len(steps)):
            self._step_units.put(s, self._step_unit(s))
            if self._known[s]:
                self._total += steps[s].cost
        if self.theta is not None and self.h is not None:
            for s in range(len(steps)):
                self._cost_units.put(s, self._cost_unit(s))
        self._uncovered_plan = [
            _uncovered_claim_diag(cid) for cid in self._claims_by_id if cid not in occ
        ]
        self._ppos = [_phase_position(self._topo, step) for step in steps]
        self._descents = [s for s in range(1, len(steps)) if self._ppos[s] < self._ppos[s - 1]]
        # -- the pack ----------------------------------------------------------------------
        self._segments = list(pack.segments)
        self._prefetches = list(pack.prefetches)
        self._notes = list(pack.trace_notes)
        self._gens = list(getattr(pack, "generations", ()))
        self._seg_ids = _Tally(s.claim_id for s in self._segments)
        self._trace_ids = _Tally(t.claim_id for t in self._notes)
        self._pf_names = _Tally(p.name for p in self._prefetches)
        self._pf_targets = {p.name: set(p.targets) for p in self._prefetches}
        self._pf_ref: dict = {}
        self._cid_segs: dict = {}
        for s, seg in enumerate(self._segments):
            self._pf_ref.setdefault(seg.prefetch, set()).add(s)
            self._cid_segs.setdefault(seg.claim_id, set()).add(s)
        self._seg_units = _Sparse()
        for s in range(len(self._segments)):
            self._seg_units.put(s, self._segment_unit(s))
        self._bad_pf = [d for d in (_prefetch_law(pf) for pf in self._prefetches) if d is not None]
        self._uncovered_pack = self._pack_coverage()
        self._pack_header = _pack_header_laws(pack)
        self._pack_gens = _pack_generation_laws(module, pack)
        self._pack_key = (pack.topo_gen, pack.map_gen, pack.data_gen, pack.pipeline_depth)
        self.diagnostics = self._assemble()

    # --- units --------------------------------------------------------------------------------

    def _repair(self, pi: int, e: int, claims, sparse, lower_only: bool = False) -> None:
        """Re-derive the decoupled-tail pairs of phase pi that touch local claim e (only the
        pairs (e, x > e) when building, where every claim is visited)."""
        pairs, keys = self._pairs[pi], self._pair_keys[pi]
        if not lower_only:  # a delta: drop the pairs e was part of before re-deriving them
            for key in keys.pop(e, ()):
                pairs.put(key, None)
                other = key[0] if key[1] == e else key[1]
                if other in keys:
                    keys[other].discard(key)
        a = claims[e]
        partners = range(len(claims)) if _is_sparse(a) else sparse
        pid = self._phases[pi].phase_id
        for x in partners:
            if x == e or (lower_only and x < e):
                continue
            i, j = (x, e) if x < e else (e, x)
            diags = _pair_law(pid, claims[i], claims[j])
            if diags:
                key = (i, j)
                pairs.put(key, diags)
                keys.setdefault(i, set()).add(key)
                keys.setdefault(j, set()).add(key)

    def _step_unit(self, s: int) -> list:
        step = self._steps[s]
        cid = step.claim_id
        claims = self._claims_by_id
        claim = claims[cid] if cid in claims else None
        if claim is None:
            return [_unknown_step_diag(step)]
        out = []
        if self._occ[cid][0] != s:
            out.append(_repeated_step_diag(step))
        offer = self._offer
        if offer is None:
            out += _step_laws(step, claim, self._declared_in[cid], None, None)
        else:
            out += _step_laws(
                step, claim, self._declared_in[cid], _OfferView(offer), offer.position.get(cid)
            )
        return out

    def _cost_unit(self, s: int) -> list:
        if not self._known[s]:
            return []
        t = s - 1
        while t >= 0 and not self._known[t]:
            t -= 1
        prev = self._steps[t].candidate if t >= 0 else None
        diag = _cost_law(self._steps[s], prev, self._scope)
        return [] if diag is None else [diag]

    def _segment_unit(self, s: int) -> list:
        return _segment_laws(
            self.module, self._segments[s], self._trace_ids, self._claims_by_id, self._pf_targets
        )

    def _pack_coverage(self) -> list:
        """R10 coverage: the claims no segment realizes, in claim-id order, control claims aside."""
        seg_ids = self._seg_ids
        out = []
        for cid in sorted(c for c in self._claims_by_id if c not in seg_ids):
            diag = _uncovered_segment_diag(cid, self._claims_by_id[cid])
            if diag is not None:
                out.append(diag)
        return out

    # --- a delta -----------------------------------------------------------------------------

    def apply(self, module: Module, result, pack) -> list[Diagnostic]:
        """Advance to (module, result, pack) and return the full verdict of the three."""
        try:
            self._advance(module, result, pack)
        except _Structure:
            self.rebuilds += 1
            self._build(module, result, pack)
        return self.diagnostics

    def _advance(self, module: Module, result, pack) -> None:
        old_module = self.module
        # -- what changed (by identity) --------------------------------------------------
        if len(module.phases) != len(self._phases):
            raise _Structure
        changed_claims: list[int] = []
        for pi, (was, now) in enumerate(zip(self._phases, module.phases)):
            if was is now:
                continue
            if (
                was.phase_id != now.phase_id
                or tuple(was.deps) != tuple(now.deps)
                or getattr(was, "event", "") != getattr(now, "event", "")
                or len(was.claims) != len(now.claims)
            ):
                raise _Structure
            lo = self._phase_start[pi]
            for i in compress(range(len(now.claims)), map(is_not, was.claims, now.claims)):
                if now.claims[i].id != was.claims[i].id:
                    raise _Structure
                changed_claims.append(lo + i)
        changed_rids: list = []
        if module.resources is not old_module.resources:
            if module.resources.keys() != old_module.resources.keys():
                raise _Structure
            old_res = old_module.resources
            changed_rids = [rid for rid, r in module.resources.items() if old_res[rid] is not r]
        if changed_claims and self._events:
            raise _Structure  # EV3 walks the whole program flow
        steps = result.steps
        segments, notes = pack.segments, pack.trace_notes
        prefetches, gens = pack.prefetches, getattr(pack, "generations", ())
        if (
            len(steps) != len(self._steps)
            or len(segments) != len(self._segments)
            or len(notes) != len(self._notes)
        ):
            raise _Structure
        changed_steps = list(compress(range(len(steps)), map(is_not, self._steps, steps)))
        for s in changed_steps:
            if steps[s].claim_id != self._steps[s].claim_id:
                raise _Structure  # a step that moves claims moves the occurrences
        changed_segs = list(compress(range(len(segments)), map(is_not, self._segments, segments)))
        changed_notes = list(compress(range(len(notes)), map(is_not, self._notes, notes)))
        pf_moved = len(prefetches) != len(self._prefetches) or any(
            map(is_not, prefetches, self._prefetches)
        )
        # From here on nothing raises _Structure: the state advances.

        # -- the module's units --------------------------------------------------------------
        self.module = module
        flat = self._flat
        old_claims: dict[int, object] = {}
        for i in changed_claims:
            pi = self._phase_of[i]
            old_claims[i] = flat[i]
            flat[i] = module.phases[pi].claims[i - self._phase_start[pi]]
        dirty_claims = set(changed_claims)
        for rid in changed_rids:
            dirty_claims |= self._users.get(rid, set())
        if changed_rids:
            self._registry = _registry_laws(module)
            self._res_domain = _resource_domain_laws(module)
        for i, old in old_claims.items():
            for rid in set(old.io_rids()):
                users = self._users.get(rid)
                if users is not None:
                    users.discard(i)
            for rid in flat[i].io_rids():
                self._users.setdefault(rid, set()).add(i)
            self._claims_by_id[flat[i].id] = flat[i]
        mask_law = self._mask_law
        for i in dirty_claims:
            self._units.put(i, _claim_laws(module, flat[i], mask_law))
        by_phase: dict[int, list[int]] = {}
        for i in changed_claims:
            pi = self._phase_of[i]
            by_phase.setdefault(pi, []).append(i - self._phase_start[pi])
        for pi, locals_ in by_phase.items():
            claims = module.phases[pi].claims
            sparse = self._sparse[pi]
            for e in locals_:
                was_sparse = _is_sparse(old_claims[self._phase_start[pi] + e])
                now_sparse = _is_sparse(claims[e])
                if was_sparse and not now_sparse:
                    del sparse[bisect_left(sparse, e)]
                elif now_sparse and not was_sparse:
                    insort(sparse, e)
            for e in locals_:
                self._repair(pi, e, claims, sparse)
        self._phases = list(module.phases)

        # -- the plan's units: the verifier's own offer, advanced from the module alone ------
        dirty_steps = set(changed_steps)
        if self._offer is not None and (changed_claims or changed_rids):
            offer = self._offer
            position = offer.position
            edited = {position[flat[i].id]: flat[i] for i in changed_claims}
            moved = offer.apply(
                module, edited, {rid: module.resources[rid] for rid in changed_rids}
            )
            for p in moved:
                dirty_steps.update(self._occ.get(offer.claims[p].id, ()))
        for i in changed_claims:  # the step's claim changed
            dirty_steps.update(self._occ.get(flat[i].id, ()))
        known = self._known
        for s in changed_steps:
            if known[s]:
                self._total += steps[s].cost - self._steps[s].cost
            self._steps[s] = steps[s]
            self._ppos[s] = _phase_position(self._topo, steps[s])
        for s in sorted(dirty_steps):
            self._step_units.put(s, self._step_unit(s))
        if self.theta is not None and self.h is not None:
            cost_dirty = set()
            n = len(steps)
            for s in changed_steps:
                cost_dirty.add(s)
                t = s + 1
                while t < n and not known[t]:
                    t += 1
                if t < n:
                    cost_dirty.add(t)
            for s in sorted(cost_dirty):
                self._cost_units.put(s, self._cost_unit(s))
        if changed_steps:
            descents = set(self._descents)
            for s in changed_steps:
                for t in (s, s + 1):
                    if 1 <= t < len(steps):
                        if self._ppos[t] < self._ppos[t - 1]:
                            descents.add(t)
                        else:
                            descents.discard(t)
            self._descents = sorted(descents)
        # (The plan's coverage reads claim ids only, which the delta shape keeps.)
        self.result = result

        # -- the pack's units --------------------------------------------------------------
        dirty_segs = set(changed_segs)
        coverage_moved = False
        if pf_moved:
            old_ids = set(map(id, self._prefetches))
            new_ids = set(map(id, prefetches))
            gone = list(
                compress(
                    self._prefetches,
                    map(not_, map(new_ids.__contains__, map(id, self._prefetches))),
                )
            )
            came = list(
                compress(prefetches, map(not_, map(old_ids.__contains__, map(id, prefetches))))
            )
            for p in gone:
                self._pf_names.remove(p.name)
            for p in came:
                self._pf_names.add(p.name)
            names = {p.name for p in gone} | {p.name for p in came}
            if self._pf_names.repeated or any(self._pf_names.counts.get(n, 0) > 1 for n in names):
                targets = {p.name: set(p.targets) for p in prefetches}  # the last one wins
            else:
                targets = self._pf_targets
                for name in names:
                    targets.pop(name, None)
                for p in came:
                    targets[p.name] = set(p.targets)
            for name in names:
                dirty_segs |= self._pf_ref.get(name, set())
            self._pf_targets = targets
            if self._bad_pf or any(_prefetch_law(p) is not None for p in came):
                self._bad_pf = [
                    d for d in (_prefetch_law(pf) for pf in prefetches) if d is not None
                ]
            self._prefetches = list(prefetches)
        for t in changed_notes:
            old_cid, new_cid = self._notes[t].claim_id, notes[t].claim_id
            before = (old_cid in self._trace_ids, new_cid in self._trace_ids)
            self._trace_ids.remove(old_cid)
            self._trace_ids.add(new_cid)
            if before != (old_cid in self._trace_ids, new_cid in self._trace_ids):
                dirty_segs |= self._cid_segs.get(old_cid, set())
                dirty_segs |= self._cid_segs.get(new_cid, set())
            self._notes[t] = notes[t]
        for s in changed_segs:
            old, new = self._segments[s], segments[s]
            self._seg_ids.remove(old.claim_id)
            self._seg_ids.add(new.claim_id)
            if old.claim_id != new.claim_id and (
                old.claim_id not in self._seg_ids or self._seg_ids.counts[new.claim_id] == 1
            ):
                coverage_moved = True
            self._pf_ref[old.prefetch].discard(s)
            self._pf_ref.setdefault(new.prefetch, set()).add(s)
            self._cid_segs[old.claim_id].discard(s)
            self._cid_segs.setdefault(new.claim_id, set()).add(s)
            self._segments[s] = new
        # (A replaced resource keeps its RID, so no segment's RID resolution moves.)
        for s in sorted(dirty_segs):
            self._seg_units.put(s, self._segment_unit(s))
        if coverage_moved or any(flat[i].id not in self._seg_ids for i in changed_claims):
            self._uncovered_pack = self._pack_coverage()  # a claim no segment realizes moved
        key = (pack.topo_gen, pack.map_gen, pack.data_gen, pack.pipeline_depth)
        gens_moved = len(gens) != len(self._gens) or any(map(is_not, gens, self._gens))
        if changed_rids or gens_moved or key != self._pack_key:
            self._pack_gens = _pack_generation_laws(module, pack)
            self._pack_header = _pack_header_laws(pack)
            self._gens = list(gens)
            self._pack_key = key
        self.pack = pack
        self.diagnostics = self._assemble()

    # --- the verdict ------------------------------------------------------------------------

    def _claim_section(self, section: int) -> list:
        return [d for unit in self._units.ordered() for sec, d in unit if sec == section]

    def _assemble(self) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        # verify(module), section by section
        out += self._registry
        out += self._claim_ids
        out += self._claim_section(_S_RESOLVE)
        out += self._res_domain
        out += self._claim_section(_S_DOMAIN)
        out += self._phase_diags
        starts = self._phase_start
        units = self._units
        for pi in range(len(self._phases)):
            for unit in units.between(starts[pi], starts[pi + 1]):
                out += [d for sec, d in unit if sec == _S_HAZARD]
            for diags in self._pairs[pi].ordered():
                out += diags
        for section in (_S_LANE, _S_BOUNDS, _S_COST, _S_MASK):
            out += self._claim_section(section)
        out += self._event_diags
        # verify_plan(module, result, h, theta=theta, policy=policy)
        for diags in self._step_units.ordered():
            out += diags
        for diags in self._cost_units.ordered():
            out += diags
        out += self._uncovered_plan
        if self._steps and self._total != self.result.score:
            out.append(_score_diag(self.result.score, self._total))
        if self._descents:
            out.append(_phase_order_diag(self._steps[self._descents[0]]))
        # verify_pack(module, pack)
        out += self._uncovered_pack
        if self._seg_ids.repeated:
            out.append(_DUPLICATE_SEGMENTS)
        if self._trace_ids.repeated:
            out.append(_DUPLICATE_TRACES)
        if self._pf_names.repeated:
            out.append(_DUPLICATE_PREFETCHES)
        out += self._pack_header
        out += self._bad_pf
        for diags in self._seg_units.ordered():
            out += diags
        out += self._pack_gens
        return out


class _OfferView:
    """The verifier's incremental offer as `_step_laws` reads an offer: rows per entry, and the
    full rows for a diagnostic (the incremental offer already holds every realization)."""

    __slots__ = ("rows",)

    def __init__(self, offer):
        self.rows = offer.rows

    def full_rows(self, j):
        return self.rows[j]
