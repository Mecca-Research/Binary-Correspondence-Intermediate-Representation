"""GEM+ G18 (S4-B): the declared delta, the incremental offer and the incremental planner.

A `Delta` declares claim and resource REPLACEMENTS: each replacement is addressed by its own id
(a claim) or RID (a resource), so a replacement cannot move a claim to another id, phase or
position. `apply_delta` is the module a delta declares -- the reference, O(module) -- a new
`Module` that shares every unchanged claim, phase and resource with the old one and leaves the
old one as it was.

`IncrementalOffer` is the planner's offer (`realize.fused_offer`) kept with the position
indexes its discount facts are read from, advanced by the claims and resources a delta
replaced. It re-derives the edited claims, and every claim whose discount (`realize._DISCOUNT`)
the edit can move: the later readers, in the same phase, of an operand an edited claim writes
differently; the later claims sharing a value-numbered identity with a claim whose identity
moved; and the claims whose tier or CSE eligibility reads an edited resource. The rows come from
the one enumeration (`realize._offer_rows`). The planner and the verifier's delta
(`verify.delta`) each hold their own: one spelling, two derivations, neither trusting the other.

`IncrementalPlan` holds the planner's state for one module under (h, theta, policy) and advances
it by a delta:

  * **the min-plus DP** -- from the first column whose rows or `shares` changed, through the one
    relaxation (`realize._relax_column`), until a column leaves the state the next column reads
    as it was: the same cheapest narrow and wide slots with the same gap between their weights.
    Every later column is then the old one shifted by a constant, which is applied lazily (a
    Fenwick tree of column shifts), and the pass jumps to the next dirty column, if any;
  * **the path** -- walked back from the sink through the recomputed columns until it rejoins
    the old path, whose steps are kept as they were.

The result is `realize.optimize(apply_delta(module, delta), h, theta, policy)` -- the same steps,
costs and score (`planner.delta.parity`) -- while the work is proportional to the delta's
dependency cone plus a few C-level list copies, not to the module.

Declared boundaries (v0):
  * only replacements: adding or removing a claim or a resource, moving a claim, editing a
    phase, or changing (h, theta, policy) is a new plan (`IncrementalPlan.build`);
  * the module must declare each claim id once (a delta addresses a claim by its id; the
    native planner's domain draws the same line, and R1.1 refuses the module anyway);
  * the state trusts the declared history: a claim or resource mutated in place behind its back
    is outside what a delta declares. A mutation that followed the S1-B rule (`touch()`) is
    refused here (the revision moved); one that did not is found where S1-B finds it, by the
    content identity (`provenance.module_identity`), not by the incremental planner.
"""

from __future__ import annotations

import heapq
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass, replace

from ..model import Claim, Module, Resource
from .cost import HProfile, Theta
from .realize import (
    _DISCOUNT,
    ChosenStep,
    Offer,
    OfferMap,
    RealizationResult,
    _materialize,
    _offer_rows,
    _relax_column,
    _resource_factors,
    cse_eligible,
    cse_identity,
    fused_offer,
)
from .weights import PERF, Policy, weights

__all__ = ["Delta", "DeltaError", "IncrementalOffer", "IncrementalPlan", "apply_delta"]


class DeltaError(ValueError):
    """A delta the incremental chain does not admit (v0): a replacement for an id or RID the
    module does not declare, two replacements for one, a module that declares a claim id twice,
    or a module that changed outside a delta."""


@dataclass(frozen=True)
class Delta:
    """A declared edit: claim and resource replacements, each addressed by its own id / RID."""

    claims: tuple[Claim, ...] = ()
    resources: tuple[Resource, ...] = ()


def _replacements(delta: Delta) -> tuple[dict[int, Claim], dict[int, Resource]]:
    if not isinstance(delta, Delta):
        raise DeltaError(f"a delta is a Delta, not {type(delta).__name__}")
    if not isinstance(delta.claims, tuple) or not isinstance(delta.resources, tuple):
        raise DeltaError(
            "a delta's claims and resources are tuples of replacements, not "
            f"{type(delta.claims).__name__} and {type(delta.resources).__name__}"
        )
    claims: dict[int, Claim] = {}
    for claim in delta.claims:
        if not isinstance(claim, Claim):
            raise DeltaError(f"a claim replacement is a Claim, not {type(claim).__name__}")
        if claim.id in claims:
            raise DeltaError(f"claim {claim.id}: replaced twice by one delta")
        claims[claim.id] = claim
    resources: dict[int, Resource] = {}
    for resource in delta.resources:
        if not isinstance(resource, Resource):
            raise DeltaError(f"a resource replacement is a Resource, not {type(resource).__name__}")
        if resource.rid in resources:
            raise DeltaError(f"resource {resource.rid}: replaced twice by one delta")
        resources[resource.rid] = resource
    return claims, resources


def _rebuild(module: Module, where: dict[int, tuple[int, int]], claims, resources) -> Module:
    """The new module: the edited phases rebuilt with their replacements, every other phase,
    claim and resource shared, and the revision moved (a declared mutation, S1-B)."""
    phases = list(module.phases)
    edits: dict[int, list] = {}
    for cid, claim in claims.items():
        pi, ci = where[cid]
        edits.setdefault(pi, []).append((ci, claim))
    for pi, changes in edits.items():
        listed = list(phases[pi].claims)
        for ci, claim in changes:
            listed[ci] = claim
        phases[pi] = replace(phases[pi], claims=listed)
    registry = module.resources
    if resources:
        registry = dict(registry)
        registry.update(resources)
    out = replace(module, resources=registry, phases=phases)
    out.touch()
    return out


def _unique_where(module: Module) -> dict[int, tuple[int, int]]:
    """Claim id -> (phase index, index in the phase), refusing a module that declares an id
    twice: a delta addresses a claim by its id. The one spelling of that law -- `apply_delta`,
    both incremental states and the incremental verdict (`verify.delta`) refuse through it."""
    where: dict[int, tuple[int, int]] = {}
    for pi, phase in enumerate(module.phases):
        for ci, claim in enumerate(phase.claims):
            if claim.id in where:
                raise DeltaError(
                    f"claim {claim.id}: the module declares the id twice, so a delta cannot "
                    "address it"
                )
            where[claim.id] = (pi, ci)
    return where


def _addressed(claims: dict, resources: dict, where: dict, registry: dict) -> None:
    """Refuse a replacement addressed to a claim id or RID the module does not declare."""
    for cid in claims:
        if cid not in where:
            raise DeltaError(f"claim {cid}: not declared by the module")
    for rid in resources:
        if rid not in registry:
            raise DeltaError(f"resource {rid}: not declared by the module")


def apply_delta(module: Module, delta: Delta) -> Module:
    """The module `delta` declares, as a new `Module` (the reference; O(module)). Every claim,
    phase and resource the delta does not replace is shared with `module`, which is unchanged."""
    claims, resources = _replacements(delta)
    where = _unique_where(module)
    _addressed(claims, resources, where, module.resources)
    return _rebuild(module, where, claims, resources)


def _remove_all(positions: list, p: int) -> None:
    del positions[bisect_left(positions, p) : bisect_right(positions, p)]


class IncrementalOffer:
    """`realize.fused_offer` of one module on one target, kept with the position indexes its
    discount facts are read from, and advanced by the claims and resources a delta replaced.

    Positions are planning positions (phases in the canonical order, claims in phase order).
    `rows[p]`, `modes[p]` and `factors[p]` are `fused_offer`'s for the current module, always."""

    __slots__ = (
        "module",
        "h",
        "geometry",
        "claims",
        "phase_ids",
        "rows",
        "modes",
        "factors",
        "position",
        "where",
        "_block",
        "_writes",
        "_barred",
        "_readers",
        "_holders",
        "_cost_users",
        "_io_users",
        "_eligible",
        "_sig",
    )

    def __init__(self, module: Module, h: HProfile):
        self.where = _unique_where(module)
        offer = fused_offer(module, h)
        claims = offer.claims
        position = {claim.id: p for p, claim in enumerate(claims)}
        self.module, self.h = module, h
        self.geometry = offer.geometry
        self.claims = list(claims)
        self.phase_ids = list(offer.phase_ids)
        self.rows = list(offer.rows)
        self.modes = list(offer.modes)
        self.factors = list(offer.factors)
        self.position = position
        self._index(module)

    def snapshot(self) -> Offer:
        """The current offer as an `Offer` of its own (copies: the state goes on moving)."""
        n = len(self.claims)
        return Offer(
            list(self.claims),
            list(self.phase_ids),
            list(self.rows),
            range(n),
            list(self.modes),
            list(self.factors),
            self.geometry,
            self.h,
        )

    def _index(self, module: Module) -> None:
        """Per phase block, the ordered positions that write (with multiplicity), barrier-write
        and read each operand and hold each value-numbered identity; per resource, the positions
        whose tier or CSE eligibility reads it."""
        claims, phase_ids = self.claims, self.phase_ids
        n = len(claims)
        block = [0] * n
        writes: list[dict] = []
        barred: list[dict] = []
        readers: list[dict] = []
        b = -1
        for p in range(n):
            if p == 0 or phase_ids[p] != phase_ids[p - 1]:  # a phase's claims are contiguous
                b += 1
                writes.append({})
                barred.append({})
                readers.append({})
            block[p] = b
            claim = claims[p]
            wi, bi, ri = writes[b], barred[b], readers[b]
            for r in claim.wr:
                wi.setdefault(r, []).append(p)
                if claim.hazard == "barriered":
                    lst = bi.setdefault(r, [])
                    if not lst or lst[-1] != p:
                        lst.append(p)
            for r in set(claim.rd):
                ri.setdefault(r, []).append(p)
        self._block, self._writes, self._barred, self._readers = block, writes, barred, readers
        holders: list[dict] = [{} for _ in writes]
        eligible = [False] * n
        sig: list = [None] * n
        cost_users: dict = {}
        io_users: dict = {}
        for p in range(n):
            claim = claims[p]
            cost_rid = claim.rd[0] if claim.rd else claim.primary_rid
            if cost_rid is not None:
                cost_users.setdefault(cost_rid, set()).add(p)
            for r in claim.io_rids():
                io_users.setdefault(r, set()).add(p)
            if cse_eligible(claim, module):
                eligible[p] = True
                s = self._identity(p, claim)
                sig[p] = s
                holders[block[p]].setdefault(s, []).append(p)
        self._holders, self._eligible, self._sig = holders, eligible, sig
        self._cost_users, self._io_users = cost_users, io_users

    def _identity(self, p: int, claim: Claim) -> tuple:
        """`cse_identity` with each read's version read off the index: the writes to it earlier
        in the phase, counted with multiplicity -- the count `fused_offer`'s walk keeps."""
        wi = self._writes[self._block[p]]
        version = {}
        for r in claim.rd:
            positions = wi.get(r)
            if positions:
                version[r] = bisect_left(positions, p)
        return cse_identity(claim, version)

    def apply(self, module: Module, edited: dict, resources: dict) -> set:
        """Advance to `module`, which replaced the claims at the positions `edited` names
        (position -> the new claim) and the resources `resources` names (RID -> the new
        resource). Returns the positions whose rows changed."""
        claims = self.claims
        old = {p: claims[p] for p in edited}
        for p, claim in edited.items():
            claims[p] = claim
        touched: set = set()  # positions whose eligibility or tier an edited resource reads
        for rid, res in resources.items():
            before = self.module.resources[rid]
            if (res.domain, res.access) != (before.domain, before.access):
                touched |= self._cost_users.get(rid, set())
                touched |= self._io_users.get(rid, set())
        self.module = module
        work: list = []
        queued: set = set()

        def push(p: int) -> None:
            if p not in queued:
                queued.add(p)
                heapq.heappush(work, p)

        for p, claim in edited.items():
            was = old[p]
            self._reindex(p, was, claim)
            push(p)
            # The operands whose write history at p changed: their later readers in the phase
            # may see another version, another producer or another fence.
            barred_now = claim.hazard == "barriered"
            barred_was = was.hazard == "barriered"
            readers = self._readers[self._block[p]]
            for r in set(was.wr) | set(claim.wr):
                if was.wr.count(r) != claim.wr.count(r) or barred_was != barred_now:
                    for q in readers.get(r, ()):
                        if q > p:
                            push(q)
        for p in touched:
            push(p)
        dirty: set = set()
        h, geometry = self.h, self.geometry
        while work:
            p = heapq.heappop(work)
            queued.discard(p)
            claim = claims[p]
            b = self._block[p]
            if p in edited or p in touched:
                factors = _resource_factors(
                    h, module.resource(claim.rd[0] if claim.rd else claim.primary_rid)
                )
                eligible = cse_eligible(claim, module)
            else:
                factors = self.factors[p]
                eligible = self._eligible[p]
            sig = self._identity(p, claim) if eligible else None
            holders = self._holders[b]
            if eligible != self._eligible[p] or sig != self._sig[p]:
                if self._eligible[p]:
                    lst = holders[self._sig[p]]
                    lst.remove(p)
                    for q in lst:
                        if q > p:
                            push(q)
                    if not lst:
                        del holders[self._sig[p]]
                if eligible:
                    lst = holders.setdefault(sig, [])
                    insort(lst, p)
                    for q in lst:
                        if q > p:
                            push(q)
                self._eligible[p], self._sig[p] = eligible, sig
            if eligible and holders[sig][0] < p:
                mode = _DISCOUNT[True][False][False]
            else:
                wi, bi = self._writes[b], self._barred[b]
                consumes = fenced = False
                for r in claim.rd:
                    lst = wi.get(r)
                    if lst and lst[0] < p:
                        consumes = True
                        lst = bi.get(r)
                        if lst and lst[0] < p:
                            fenced = True
                mode = _DISCOUNT[False][consumes][fenced or claim.hazard == "barriered"]
            if p in edited or mode != self.modes[p] or factors != self.factors[p]:
                rows = _offer_rows(claim, h, geometry, *factors, None, mode)
                if rows != self.rows[p]:
                    self.rows[p] = rows
                    dirty.add(p)
                self.modes[p], self.factors[p] = mode, factors
        return dirty

    def _reindex(self, p: int, was: Claim, claim: Claim) -> None:
        """Move position p's entries in the operand indexes from its old claim to its new one
        (the identity index is moved by the re-derivation, which knows the new identity)."""
        b = self._block[p]
        wi, bi, ri = self._writes[b], self._barred[b], self._readers[b]
        for r in set(was.wr):
            _remove_all(wi[r], p)
            if not wi[r]:
                del wi[r]
            if r in bi:
                _remove_all(bi[r], p)
                if not bi[r]:
                    del bi[r]
        for r in set(was.rd):
            _remove_all(ri[r], p)
            if not ri[r]:
                del ri[r]
        for r in claim.wr:
            insort(wi.setdefault(r, []), p)
        if claim.hazard == "barriered":
            for r in set(claim.wr):
                insort(bi.setdefault(r, []), p)
        for r in set(claim.rd):
            insort(ri.setdefault(r, []), p)
        was_cost = was.rd[0] if was.rd else was.primary_rid
        cost = claim.rd[0] if claim.rd else claim.primary_rid
        if was_cost != cost:
            if was_cost is not None:
                self._cost_users[was_cost].discard(p)
            if cost is not None:
                self._cost_users.setdefault(cost, set()).add(p)
        for r in set(was.io_rids()) - set(claim.io_rids()):
            self._io_users[r].discard(p)
        for r in set(claim.io_rids()):
            self._io_users.setdefault(r, set()).add(p)


class _Shifts:
    """The DP's lazy column shifts: a constant added to every column from j on, read per column
    (a Fenwick tree over the column index)."""

    __slots__ = ("tree",)

    def __init__(self, n: int):
        self.tree = [0] * (n + 1)

    def add_from(self, j: int, value: int) -> None:
        tree = self.tree
        i = j + 1
        n = len(tree)
        while i < n:
            tree[i] += value
            i += i & -i

    def at(self, j: int) -> int:
        tree = self.tree
        s = 0
        i = j + 1
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s


class IncrementalPlan:
    """The planner's state for one module under (h, theta, policy), advanced by declared deltas.

    `result` is always `realize.optimize(module, h, theta, policy)` for the current `module`.
    `apply(delta)` advances both and returns the new result; `changed` then names the columns
    (planning positions, which are also the plan's step indexes) whose step is a new object,
    and `edited` the positions whose claim the delta replaced."""

    __slots__ = (
        "module",
        "h",
        "theta",
        "policy",
        "result",
        "changed",
        "edited",
        "revision",
        "_offer",
        "_hot",
        "_where",
        "_w",
        "_cd",
        "_cp",
        "_cn",
        "_cw",
        "_shifts",
        "_choice",
    )

    @property
    def claims(self) -> list:
        """The claims in planning order: `claims[n]` is the claim plan step n realizes. The
        state's own list, read by the delta StreamPack -- read it, never mutate it."""
        return self._offer.claims

    @classmethod
    def build(
        cls, module: Module, h: HProfile, theta: Theta, policy: Policy = PERF
    ) -> "IncrementalPlan":
        """Plan `module` from scratch and keep the state a delta re-derives from. The first
        result is `optimize`'s: the same offer (`fused_offer`), the same relaxation."""
        self = cls.__new__(cls)
        self.module, self.h, self.theta, self.policy = module, h, theta, policy
        self.revision = module.revision
        self._hot = theta.thermal >= 60
        self._offer = IncrementalOffer(module, h)
        self._where = self._offer.where
        self._w = {}
        n = len(self._offer.claims)
        self._cd, self._cp = [None] * n, [None] * n
        self._cn, self._cw = [-1] * n, [-1] * n
        self._shifts = _Shifts(n)
        self._choice = [0] * n
        if n:
            for j in range(n):
                self._relax(j)
            steps = [None] * n
            s = self._sink()
            for j in range(n - 1, -1, -1):
                self._choice[j] = s
                steps[j] = self._step(j, s)
                s = self._cp[j][s]
            self.result = self._result(steps)
        else:
            self.result = RealizationResult([], 0)
        self.changed = frozenset(range(n))
        self.edited = frozenset()
        return self

    # --- the DP ------------------------------------------------------------------------------

    def _weights(self, pid) -> tuple:
        w = self._w.get(pid)
        if w is None:
            w = self._w[pid] = weights(self.h, self.theta, pid, self.policy)
        return w

    def _relax(self, j: int) -> None:
        """Recompute column j from column j-1 (`_relax_column`), storing weights net of the
        column's lazy shift."""
        cd, cn, cw = self._cd, self._cn, self._cw
        claims = self._offer.claims
        sh = self._shifts.at(j)
        if j:
            pn, pw = cn[j - 1], cw[j - 1]
            prev = cd[j - 1]
            base = self._shifts.at(j - 1) - sh
            dn = prev[pn] + base if pn >= 0 else 0
            dw = prev[pw] + base if pw >= 0 else 0
            shares = not set(claims[j - 1].rd).isdisjoint(claims[j].rd)
        else:
            pn = pw = -1
            dn = dw = 0
            shares = False
        crow = self._offer.rows[j]
        k = len(crow)
        dist = [0] * k
        pred = [-1] * k
        cn[j], cw[j] = _relax_column(
            crow,
            k,
            self._weights(self._offer.phase_ids[j]),
            self._hot,
            shares,
            pn,
            dn,
            pw,
            dw,
            dist,
            pred,
            0,
        )
        cd[j], self._cp[j] = dist, pred

    def _sink(self) -> int:
        last = self._cd[-1]
        s = 0
        for i in range(1, len(last)):
            if last[i] < last[s]:
                s = i
        return s

    def _step(self, j: int, s: int) -> ChosenStep:
        cd = self._cd
        claim = self._offer.claims[j]
        p = self._cp[j][s]
        weight = cd[j][s] + self._shifts.at(j)
        if p >= 0:
            weight -= cd[j - 1][p] + self._shifts.at(j - 1)
        return ChosenStep(
            claim.id, self._offer.phase_ids[j], _materialize(claim, self._offer.rows[j][s]), weight
        )

    def _result(self, steps: list) -> RealizationResult:
        n = len(steps)
        score = self._cd[n - 1][self._choice[n - 1]] + self._shifts.at(n - 1) if n else 0
        offer = self._offer
        return RealizationResult(steps, score, cand_map=OfferMap(offer.snapshot(), offer.position))

    # --- a delta -----------------------------------------------------------------------------

    def apply(self, delta: Delta) -> RealizationResult:
        """Advance the state by `delta` and return the new plan: `optimize` of the new module,
        step for step."""
        if self.module.revision != self.revision:
            raise DeltaError(
                f"the module changed outside a delta (revision {self.module.revision}; the "
                f"state holds {self.revision}): plan it again"
            )
        claims, resources = _replacements(delta)
        _addressed(claims, resources, self._where, self.module.resources)
        position = self._offer.position
        module = _rebuild(self.module, self._where, claims, resources)
        edited = {position[cid]: claim for cid, claim in claims.items()}
        old_rd = {p: self._offer.claims[p].rd for p in edited}
        dirty = self._offer.apply(module, edited, resources)
        n = len(self._offer.claims)
        for p, claim in edited.items():
            if tuple(old_rd[p]) != tuple(claim.rd):  # `shares` of columns p and p + 1
                dirty.add(p)
                if p + 1 < n:
                    dirty.add(p + 1)
        self.module = module
        self.revision = module.revision
        self.edited = frozenset(edited)
        self.changed = self._replan(sorted(dirty), edited)
        return self.result

    def _replan(self, dirty: list, edited: dict) -> frozenset:
        """Re-relax from each dirty column until the state the next column reads is the old one
        up to a constant; then jump to the next dirty column. Returns the changed step columns."""
        n = len(self._offer.claims)
        cd, cn, cw = self._cd, self._cn, self._cw
        shifts = self._shifts
        recomputed: list[int] = []
        running = 0  # the shift this pass has added beyond the last column it recomputed
        di = 0
        m = len(dirty)
        j = dirty[0] if m else n
        while j < n:
            old, on, ow = cd[j], cn[j], cw[j]
            self._relax(j)
            recomputed.append(j)
            while di < m and dirty[di] <= j:
                di += 1
            if j == n - 1:
                break
            new, nn, nw = cd[j], cn[j], cw[j]
            if (
                nn == on
                and nw == ow
                and (nn < 0 or nw < 0 or new[nw] - new[nn] == old[ow] - old[on])
            ):
                ref = nn if nn >= 0 else nw
                delta = new[ref] - old[ref] + running
                if delta != running:
                    shifts.add_from(j + 1, delta - running)
                    running = delta
                j = dirty[di] if di < m else n
            else:
                j += 1
        choice = self._choice
        steps = list(self.result.steps)
        changed: dict[int, int] = {}
        if n:
            done = set(recomputed)
            s = self._sink() if recomputed and recomputed[-1] == n - 1 else choice[n - 1]
            j = n - 1
            while j >= 0:
                if j in done:
                    changed[j] = s
                    s = self._cp[j][s]
                    j -= 1
                elif s == choice[j]:
                    # On the old path: it holds down to the next recomputed column, which the
                    # clean column above it enters where the old path did.
                    i = bisect_left(recomputed, j) - 1
                    if i < 0:
                        break
                    j = recomputed[i]
                    s = choice[j]
                else:
                    changed[j] = s
                    s = self._cp[j][s]
                    j -= 1
            for j, s in changed.items():
                choice[j] = s
            for j in edited:
                changed.setdefault(j, choice[j])
            for j in changed:
                steps[j] = self._step(j, choice[j])
        self.result = self._result(steps)
        return frozenset(changed)
