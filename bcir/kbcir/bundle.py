"""Multi-claim bundle (joint) optimization -- the combinatorial step beyond pairwise.

The K_BCIR optimizer's shortest path couples claims **pairwise in their declared order**
(the shared-input fusion / deforestation discounts apply only between *adjacent* wide
claims that share a read). When claims that share an input operand are *interleaved* with
non-sharers, that pairwise model misses the discount a joint reordering would capture.

Bundle optimization is the genuine combinatorial joint step: find the clusters of claims
that share a read operand, and jointly search the intra-phase **order** (bounded,
exhaustive) that minimizes the plan score, re-pricing through the same `optimize`.
Reordering independent same-phase claims is *legal* -- intra-phase claim order is a free
deterministic choice; the schedule keys on the phase DAG, not the declared order
(`differential._respects_phase_dag`), and only mutually-independent claims (no RAW/WAR/WAW
between them) are reordered.

Bounded compile time + search certificates: each bundle is capped at `max_size` claims, so
its search is <= `max_size!` orderings; every adopted reorder emits a `BundleCertificate`
recording what was searched and the gain over the greedy (declared-order) plan -- a
proof-carrying record of the joint decision. On a module with no interleaved sharers it is
a clean no-op (the pinned scores are unchanged).
"""

from __future__ import annotations

import copy
import itertools
from collections import defaultdict
from dataclasses import dataclass

from ..model import Claim, Module
from .cost import HProfile, Theta
from .realize import RealizationResult, optimize
from .weights import PERF, Policy


def _conflict(a: Claim, b: Claim) -> bool:
    """RAW / WAR / WAW between two claims (mirrors verify / overlap `_conflict`)."""
    # ASM3b: a barriered claim is an ordering fence -- it conflicts with everything, so find_bundles /
    # _legal_reorder never reorder any claim across it (the first-class-ordering-edge contract).
    if a.hazard == "barriered" or b.hazard == "barriered":
        return True
    aw, ar = set(a.wr), set(a.rd)
    bw, br = set(b.wr), set(b.rd)
    return bool(aw & (br | bw) or bw & ar)


@dataclass(frozen=True)
class Bundle:
    """A cluster of mutually-independent same-phase claims that share a read operand."""

    phase_id: int
    claim_ids: tuple[int, ...]
    shared_rid: int


@dataclass(frozen=True)
class BundleCertificate:
    """The proof-carrying record of one joint reorder: what was searched and the win."""

    bundle: Bundle
    searched: int            # orderings evaluated (<= max_size!)
    greedy_score: int        # the plan score before this bundle's reorder
    joint_score: int         # ... and after
    order: tuple[int, ...]   # the adopted intra-bundle claim order

    @property
    def gain(self) -> int:
        return self.greedy_score - self.joint_score


def find_bundles(module: Module, max_size: int = 4) -> list[Bundle]:
    """The input-sharing clusters: per phase, for each read operand, the maximal set of
    mutually-independent claims that read it (size 2..`max_size`). Deduplicated by claim
    set; a claim may appear in more than one bundle (it shares different operands)."""
    bundles: list[Bundle] = []
    seen: set[tuple[int, ...]] = set()
    for ph in module.phases:
        readers: dict[int, list[Claim]] = defaultdict(list)
        for c in ph.claims:
            for rid in set(c.rd):
                readers[rid].append(c)
        for rid, group in sorted(readers.items()):
            if len(group) < 2:
                continue
            indep: list[Claim] = []
            for c in sorted(group, key=lambda x: x.id):
                if all(not _conflict(c, d) for d in indep):
                    indep.append(c)
                if len(indep) >= max_size:
                    break
            if len(indep) < 2:
                continue
            ids = tuple(sorted(c.id for c in indep))
            if ids in seen:
                continue
            seen.add(ids)
            bundles.append(Bundle(ph.phase_id, ids, rid))
    return bundles


def _phase_of(module: Module, phase_id: int):
    return next(p for p in module.phases if p.phase_id == phase_id)


def _legal_reorder(orig: list[Claim], new: list[Claim]) -> bool:
    """True iff `new` is a valid linear extension of `orig`'s intra-phase conflict order:
    every pair of *conflicting* claims keeps its original relative order (the declared
    order is the sequential semantics for a RAW/WAR/WAW pair, e.g. a matmul k-accumulation
    chain). Independent claims may be freely permuted; conflicting ones may not be inverted."""
    pos = {c.id: i for i, c in enumerate(new)}
    for i in range(len(orig)):
        for j in range(i + 1, len(orig)):
            if _conflict(orig[i], orig[j]) and pos[orig[i].id] > pos[orig[j].id]:
                return False
    return True


def _reordered(module: Module, phase_id: int, bundle_ids: tuple[int, ...],
               order: tuple[int, ...]) -> Module:
    """A deep copy of `module` with the bundle's claims placed contiguously, in `order`,
    at the bundle's anchor position (the earliest position any bundle claim held); the
    non-bundle claims keep their relative order. Only the intra-phase claim list changes."""
    m = copy.deepcopy(module)
    phase = _phase_of(m, phase_id)
    bset = set(bundle_ids)
    positions = [i for i, c in enumerate(phase.claims) if c.id in bset]
    anchor = sum(1 for i, c in enumerate(phase.claims) if c.id not in bset and i < positions[0])
    by_id = {c.id: c for c in phase.claims}
    non_bundle = [c for c in phase.claims if c.id not in bset]
    permuted = [by_id[i] for i in order]
    phase.claims = non_bundle[:anchor] + permuted + non_bundle[anchor:]
    return m


def optimize_bundled(module: Module, h: HProfile, theta: Theta, policy: Policy = PERF,
                     max_size: int = 4) -> tuple[RealizationResult, list[BundleCertificate]]:
    """Greedy + per-bundle exhaustive joint optimization. Starts from the declared-order
    plan, then for each input-sharing bundle searches every intra-bundle ordering
    (contiguous at the anchor), adopting the one that strictly lowers the plan score.
    Returns the best `RealizationResult` and one `BundleCertificate` per *improving*
    bundle. A no-op (empty certificates, base plan) when no reorder helps."""
    working = module
    result = optimize(working, h, theta, policy)
    certs: list[BundleCertificate] = []

    for bundle in find_bundles(module, max_size):
        before = result.score
        orig_claims = list(_phase_of(working, bundle.phase_id).claims)
        best_order = bundle.claim_ids
        best_score = before
        best_module = working
        searched = 0
        for order in itertools.permutations(bundle.claim_ids):
            cand_module = _reordered(working, bundle.phase_id, bundle.claim_ids, order)
            # Only consider dependency-preserving reorders (never invert a conflicting pair).
            if not _legal_reorder(orig_claims, _phase_of(cand_module, bundle.phase_id).claims):
                continue
            searched += 1
            score = optimize(cand_module, h, theta, policy).score
            if score < best_score:
                best_score, best_order, best_module = score, order, cand_module
        if best_score < before:
            working = best_module
            result = optimize(working, h, theta, policy)
            certs.append(BundleCertificate(bundle, searched, before, best_score, best_order))
    return result, certs
