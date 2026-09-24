"""GEM+ G18 (S4-B): the K_BCIR -> StreamPack chain advanced by declared deltas.

`gem.delta_chain.DeltaChain` holds the incremental plan (`kbcir.delta`), the delta StreamPack
(`gem.delta_pack`) and the incremental verdict (`verify.delta`) and advances them by a declared
`Delta`. Every link must be the chain run from scratch on the module the delta declares, while the
work follows the delta's dependency cone. Each law is held to a negative witness: a malformed delta
every rail must refuse before anything moves, a pack the wire cannot carry refused as `encode`
refuses it and re-emitted in full once repaired, a forged plan or pack the verdict re-verifies, a
forged offer the verdict never reads, and the corpus's own coverage -- every edit family, every
forgery, every law a replacement can raise.
"""

from __future__ import annotations

import functools
from dataclasses import replace

from bcir.tests import delta_fixtures as df


@functools.cache
def _graded() -> tuple[dict, dict]:
    seen: dict = {}
    return df.measure(1, seen), seen


def _scope():
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.kbcir.weights import PERF

    return TargetProfile.x86_avx2(), Theta.mem_bound(), PERF


def _refuses(fn, *args) -> str:
    from bcir.kbcir.delta import DeltaError

    try:
        fn(*args)
    except DeltaError as exc:
        return str(exc)
    raise AssertionError(f"{getattr(fn, '__qualname__', fn)} admitted what it must refuse")


def test_the_delta_chain_is_the_chain_from_scratch_over_the_corpus():
    """planner.delta.parity, pack.delta.identity, verify.delta.identity and
    delta.malformed.accepted at zero over the whole corpus (bcir/tests/delta_fixtures.py)."""
    rows, _seen = _graded()
    assert rows == {row: 0.0 for row in df.ROWS}, rows


def test_the_corpus_exercises_every_edit_family_forgery_and_reachable_law():
    """A construct absent from the corpus is untested: every declared edit family was applied
    (a degenerate one is named `empty` instead), every forgery was handed to the verdict, the
    full verdicts raised every law the chain checks that a replacement can reach -- all but R1.1
    (a delta cannot give two claims one id) and EV1 (nor move a phase's dependencies) -- the wire
    refused a pack in almost every case, and the honest chain re-derived its verdict from scratch
    only where the declared boundary says it must: a module with event phases."""
    _rows, seen = _graded()
    cases = len(df.corpus_cases())
    families = (
        {f"claim.{f}" for f in df.CLAIM_EDITS}
        | {f"cone.{f}" for f in df.CONE_EDITS if f != "empty"}
        | {f"resource.{f}" for f in df.RESOURCE_EDITS}
        | {f"unemittable.{f}" for f in df.UNEMITTABLE}
        | {f"event.{f}" for f in df.EVENT_EDITS}
        | {"repair", "mixture", "empty"}
    )
    assert seen["families"] == families, sorted(families ^ seen["families"])
    assert seen["forgeries"] == set(df.FORGERIES)
    laws = {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R18"}
    assert laws | {"EV", "EV2", "EV3"} <= seen["laws"], sorted(seen["laws"])
    assert seen["graded"] == cases and "unbuilt" not in seen
    assert seen["refused"] >= cases * 9 // 10
    assert seen["chain.rebuilds"] > 0 and seen["chain.rebuilds.without-events"] == 0


def test_every_malformed_delta_is_refused_on_every_rail_before_anything_moves():
    """Each delta the chain does not admit is a DeltaError from the reference application, the
    plan and the chain -- with the module, the plan and the link as they were -- and the next
    honest delta still gives the chain from scratch."""
    from bcir.gem.delta_chain import DeltaChain
    from bcir.kbcir.delta import Delta, IncrementalPlan, apply_delta

    h, theta, policy = _scope()
    base = df.malformed_base()
    corpus = df.malformed_deltas(Delta)
    assert len(corpus) == 15
    for label, delta in corpus:
        m = df.malformed_base()
        _refuses(apply_delta, m, delta)
        assert m == base and m.revision == base.revision, label
        plan = IncrementalPlan.build(m, h, theta, policy)
        result = plan.result
        _refuses(plan.apply, delta)
        assert plan.module is m and plan.result is result, label
        chain = DeltaChain.build(m, h, theta, policy)
        link = chain.link
        _refuses(chain.apply, delta)
        assert chain.link is link and chain.module is m, label
        claims, resources = df._honest(m)
        got = chain.apply(Delta(claims, resources))
        want = df.reference_chain(df.declared(m, claims, resources), h, theta, policy, 2)
        assert (got.result, got.pack, got.data, got.diagnostics) == (
            want.result,
            want.pack,
            want.data,
            want.diagnostics,
        ), label


def test_a_module_a_delta_cannot_address_is_refused_by_every_state():
    """A module declaring a claim id twice (a delta names claims by id), and a module changed
    outside a delta after the state was built (a declared mutation: the revision moved)."""
    from bcir.gem.delta_chain import DeltaChain
    from bcir.kbcir.delta import Delta, IncrementalPlan, apply_delta
    from bcir.kbcir.realize import optimize
    from bcir.verify.delta import VerifyState

    h, theta, policy = _scope()
    dup = df.duplicate_module()
    assert "twice" in _refuses(IncrementalPlan.build, dup, h, theta, policy)
    assert "twice" in _refuses(DeltaChain.build, dup, h, theta, policy)
    stand_in = df.reference_chain(df.malformed_base(), h, theta, policy, 2).pack
    result = optimize(dup, h, theta, policy)
    assert "twice" in _refuses(VerifyState.build, dup, result, stand_in, h, theta, policy)
    for claim in df._flat(dup)[:2]:  # the duplicated id, and one the module declares once
        assert "twice" in _refuses(apply_delta, dup, Delta((claim,), ()))

    m = df.malformed_base()
    chain = DeltaChain.build(m, h, theta, policy)
    plan = IncrementalPlan.build(m, h, theta, policy)
    m.touch()
    delta = Delta(*df._honest(m))
    assert "outside a delta" in _refuses(chain.apply, delta)
    assert "outside a delta" in _refuses(plan.apply, delta)
    assert chain.module is m and plan.module is m


def test_a_pack_the_wire_refuses_is_refused_as_encode_refuses_it_and_re_emitted_once_repaired():
    """The plan moves first, so `DeltaChain.result` is the new module's plan even when its pack
    is refused; the refusal is the error `encode` raises, message and all; the repair after it
    re-emits the pack in full and re-verifies everything that moved while there was no verdict."""
    from bcir.abi.streampack_abi import AbiError
    from bcir.gem.delta_chain import DeltaChain
    from bcir.kbcir.delta import Delta

    h, theta, policy = _scope()
    m = dict(df.delta_modules())["delta.pipeline"]
    chain = DeltaChain.build(m, h, theta, policy)
    flat = df._flat(m)
    for bad in (
        replace(flat[2], count=1 << 64),
        replace(flat[3], rd=(1 << 32,)),
        replace(flat[4], op="x" * 70_000),
    ):
        new = df.declared(chain.module, (bad,))
        want = df.reference_chain(new, h, theta, policy, 2)
        assert want.refusal is not None
        try:
            chain.apply(Delta((bad,)))
        except AbiError as exc:
            assert (type(exc), str(exc)) == (type(want.refusal), str(want.refusal))
        else:
            raise AssertionError("the delta pack carried what the wire cannot")
        assert chain.module == new and chain.result == want.result
        # a second edit while the pack is refused, then the repair
        other = replace(flat[0], count=flat[0].count + 1)
        repaired = df.declared(new, (other, flat[_position(flat, bad)]))
        link = chain.apply(Delta((other, flat[_position(flat, bad)])))
        want = df.reference_chain(repaired, h, theta, policy, 2)
        assert (link.pack, link.data, link.diagnostics) == (want.pack, want.data, want.diagnostics)
        flat = df._flat(chain.module)


def _position(flat, bad) -> int:
    return [c.id for c in flat].index(bad.id)


def test_the_verdict_never_reads_the_offer_the_plan_carries():
    """The instrument does not take its subject's word (L9): a plan whose step names a
    realization no module offers, carrying a candidate map that claims it does, draws R9 from the
    incremental verdict exactly as from the full one -- the verdict's offer is its own."""
    from bcir.gem.delta_chain import DeltaChain
    from bcir.kbcir.delta import Delta
    from bcir.verify.delta import VerifyState

    h, theta, policy = _scope()
    m = dict(df.delta_modules())["delta.identity"]
    chain = DeltaChain.build(m, h, theta, policy)
    state = VerifyState.build(m, chain.result, chain.link.pack, h, theta, policy)
    claims, resources = df._honest(m)
    link = chain.apply(Delta(claims, resources))
    step = link.result.steps[2]
    forged = replace(step, candidate=replace(step.candidate, name="forged"))
    steps = list(link.result.steps)
    steps[2] = forged
    lie = {cid: list(cands) for cid, cands in (link.result.cand_map or {}).items()}
    lie[forged.claim_id] = lie.get(forged.claim_id, []) + [forged.candidate]
    result = replace(link.result, steps=steps, cand_map=lie)
    got = state.apply(link.module, result, link.pack)
    want = df.full_verdict(
        df.reference_chain(link.module, h, theta, policy, 2).module_laws,
        link.module,
        result,
        link.pack,
        h,
        theta,
        policy,
    )
    assert got == want
    assert any(d.law == "R9" and "forged" in d.message for d in got), got


def test_a_one_claim_delta_relaxes_its_own_column_and_splices_the_rest():
    """The cutoff: a count edit changes one column's weights by a constant, so the pass relaxes
    that column alone and shifts the rest lazily; a read edit also moves the fused edge into the
    next column, so two. The plan is `optimize`'s each time."""
    from bcir.examples import matmul_tiled
    from bcir.kbcir.delta import Delta, IncrementalPlan
    from bcir.kbcir.realize import optimize

    h, theta, policy = _scope()
    plan = IncrementalPlan.build(matmul_tiled(n=64, tile=8), h, theta, policy)
    relaxed: list[int] = []
    original = IncrementalPlan._relax

    def counting(self, j):
        relaxed.append(j)
        return original(self, j)

    IncrementalPlan._relax = counting
    try:
        for k, p in enumerate((256, 0, 511, 17)):
            c = df._flat(plan.module)[p]
            edit = replace(c, count=c.count + 1 + k)
            relaxed.clear()
            got = plan.apply(Delta((edit,)))
            assert relaxed == [p] and plan.changed == {p}, (p, relaxed, plan.changed)
            assert got == optimize(plan.module, h, theta, policy)
        c = df._flat(plan.module)[300]
        relaxed.clear()
        got = plan.apply(Delta((replace(c, rd=()),)))
        assert relaxed == [300, 301]
        assert got == optimize(plan.module, h, theta, policy)
    finally:
        IncrementalPlan._relax = original


def test_the_reference_the_fixtures_spell_is_the_product_chain():
    """`delta_fixtures.reference_chain` is spelled with the parent's entry points so RED can run
    there; it must be `delta_chain.full_chain`, or the rows would grade against a third chain."""
    from bcir.abi.streampack_abi import AbiError
    from bcir.gem.delta_chain import full_chain

    cases = df.corpus_cases()[::9]
    assert len(cases) > 20
    for case in cases:
        args = (case.module, case.h, case.theta, case.policy)
        want = df.reference_chain(*args, case.depth)
        try:
            link = full_chain(*args, df.PLAN, case.depth)
        except AbiError as exc:
            assert str(exc) == str(want.refusal), case.label
            continue
        assert want.refusal is None, case.label
        assert (link.result, link.pack, link.data, link.diagnostics) == (
            want.result,
            want.pack,
            want.data,
            want.diagnostics,
        ), case.label


def test_apply_delta_is_the_declared_module_sharing_what_it_did_not_replace():
    from bcir.kbcir.delta import Delta, apply_delta

    for name, m in df.delta_modules():
        flat = df._flat(m)
        claims = tuple(replace(c, count=c.count + 2) for c in flat[::2])
        resources = tuple(
            replace(r, priority=r.priority + 1) for r in list(m.resources.values())[:1]
        )
        before = (m.revision, list(m.phases), dict(m.resources))
        new = apply_delta(m, Delta(claims, resources))
        assert new == df.declared(m, claims, resources), name
        assert (m.revision, list(m.phases), dict(m.resources)) == before, name
        replaced = {c.id for c in claims}
        for was, now in zip(flat, df._flat(new)):
            assert (now is was) == (was.id not in replaced), name
        for rid, r in m.resources.items():
            assert (new.resources[rid] is r) == (rid not in {x.rid for x in resources}), name
        assert new.revision != 0


def test_the_delta_chain_makes_a_stated_factor_fewer_calls():
    """A one-claim delta of the audit fixture at scale 4 (4,096 claims), both sides counted in
    this process: under 1,000 calls against the chain from scratch's million-odd -- the work is the
    delta's cone, not the module (`kbcir-streampack.delta.calls` at scale 8 is the gated row)."""
    step, full = df.delta_calls(4)
    assert step < 1_000 and full > 500 * step, (step, full)
