"""The verifier chain must not pass what it has not actually checked.

Six defects found in an independent audit of `main` at `3decf69`. Every one of them is the
same shape, and it is the shape that matters most in this repository: **a check returned
clean over something it never examined.** A law that is vacuous on the input it is meant to
constrain is worse than no law, because a certificate gets issued either way.

  * R9  — candidate generation dispatched on `stride_class` alone, so an ATOMIC_ADD was
          offered `U vec16` (SCALAR geometry) and `GGG gather` (RANDOM geometry). Both plans
          verified clean on both rails. A vectorized read-modify-write is not a fast atomic;
          it is a data race, and the ordering and synchronization cost went unpriced too.
  * R9  — the law checked lane against geometry and the score against the step sum, and
          never asked whether the chosen realization was one the planner could produce.
          `Candidate(Lane.U, width=3, name="forged", cost=0)` passed: a width the hardware
          cannot issue, a name denoting no realization, and zero cost.
  * R10 — the provenance law iterated the pack's segments. An EMPTY pack made every loop
          vacuous, so a pack realizing none of the module's claims verified clean.
  * R10 — `verify_all` checked plan and pack independently and never proved the pack was
          derived from that plan, so a pack hydrated from a `vec4` plan verified clean
          against a `scalar` one. The graph -> plan -> pack chain had no link.
  * api — `build_artifact` set `attested` from R12 alone while the field reads as "this
          artifact is legal", so a module violating R5 came back `attested=True` with an
          empty diagnostic tuple.
  * R12 — both lowerers emit `C[i] = A[i] op B[i]` unconditionally. A claim declaring
          `offset=8` or `stride_k=4` was accepted, lowered to that same body, and verified
          clean: a kernel computing a different function than the claim declares, carrying
          an attestation.

Each test drives the failure, not just the success. A test that only asserted the honest
path would have passed against every one of these.
"""

from __future__ import annotations

from dataclasses import replace

from bcir.api import build_artifact
from bcir.examples import vector_add
from bcir.gem import hydrate
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.kbcir.provenance import ProvenanceMismatch, build_manifest, replay
from bcir.kbcir.realize import Candidate, CostVector, candidates_for
from bcir.kbcir.weights import PERF
from bcir.model import (ATOMIC_OPCODES, Claim, Lane, Module, Opcode, Phase, Resource,
                        StrideClass)
from bcir.verify import verify, verify_all, verify_pack, verify_plan

_H = TARGETS[sorted(TARGETS)[0]]
_TH = Theta.cool()


def _laws(diags) -> set[str]:
    return {d.law for d in diags}


def _atomic_module(stride_class: StrideClass, opcode=Opcode.ATOMIC_ADD) -> Module:
    m = Module(name="atomics")
    m.add_resource(Resource(rid=1, shape=(64,)))
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=opcode, lane=Lane.A, stride_class=stride_class,
              count=64, rd=(1,), wr=(1,), hazard="atomic")]))
    return m


# --- R9: an atomic stays atomic -----------------------------------------------------------

def test_an_atomic_opcode_has_exactly_one_realization() -> None:
    """Every atomic opcode, under every geometry, is offered A-lane width 1 and nothing else.

    Swept over all four opcodes and all six stride classes rather than the two that were
    reported, because the defect was that the generator consulted the geometry INSTEAD of
    the opcode -- so any geometry could have carried it.
    """
    for opcode in sorted(ATOMIC_OPCODES, key=lambda o: int(o)):
        for stride_class in StrideClass:
            claim = _atomic_module(stride_class, opcode).phases[0].claims[0]
            offered = candidates_for(claim, _H)
            assert [(c.lane, c.width) for c in offered] == [(Lane.A, 1)], (
                f"{opcode.name}/{stride_class.name} offered "
                f"{[(c.lane.name, c.width, c.name) for c in offered]}")


def test_r9_rejects_a_vectorized_or_gathered_atomic() -> None:
    """The law, independent of the generator.

    Fixing only `candidates_for` would leave the law still unable to say the plan is
    illegal, and R9 is what a certificate rests on -- so the plan is forged by hand here,
    the way a tampered or third-party plan would arrive.
    """
    module = _atomic_module(StrideClass.SCALAR)
    honest = optimize(module, _H, _TH, PERF)
    assert not verify_plan(module, honest, _H), "the honest atomic plan must stay clean"

    for lane, width, name in ((Lane.U, 16, "vec16"), (Lane.U, 4, "vec4"),
                              (Lane.GGG, 1, "gather"), (Lane.A, 4, "atomic")):
        forged = replace(honest, steps=tuple(
            replace(s, candidate=replace(s.candidate, lane=lane, width=width, name=name))
            for s in honest.steps))
        diags = verify_plan(module, forged, _H)
        assert "R9" in _laws(diags), f"{lane.name} width {width} was accepted for an atomic"
        assert any("atomic" in d.message for d in diags), diags

    # A non-atomic claim with the same geometry is unaffected -- the rule keys on the
    # opcode, which is the whole point.
    plain = Module(name="plain")
    plain.add_resource(Resource(rid=1, shape=(64,)))
    plain.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.SCALAR,
              count=64, rd=(1,), wr=(1,))]))
    assert not verify_plan(plain, optimize(plain, _H, _TH, PERF), _H)


# --- R9: the realization has to be one the planner could produce ---------------------------

def test_r9_refuses_a_realization_the_target_does_not_admit() -> None:
    """A fabricated candidate is the plan a tampered or third-party artifact carries.

    The `cost=0` part is what makes it more than a typo: R9 also checks that the score is
    the sum of the step costs, and a forged plan satisfies that trivially by claiming the
    work is free.
    """
    module = vector_add()
    honest = optimize(module, _H, _TH, PERF)
    assert not verify_plan(module, honest, _H)

    forged = replace(honest, score=0, steps=tuple(
        replace(s, cost=0,
                candidate=Candidate(Lane.U, 3, "forged-does-not-exist", CostVector.zero()))
        for s in honest.steps))

    diags = verify_plan(module, forged, _H)
    assert "R9" in _laws(diags)
    assert any("not among" in d.message for d in diags), diags

    # Without a target the law cannot re-derive anything, and must not pretend otherwise:
    # this is the documented weaker mode, pinned so it stays a deliberate choice.
    assert "R9" not in _laws(verify_plan(module, forged))

    # A REAL candidate for the claim, chosen differently, is still legal -- the law
    # constrains the candidate set, not the planner's preference within it.
    offered = candidates_for(module.phases[0].claims[0], _H,
                             module.resource(module.phases[0].claims[0].rd[0]))
    other = next(c for c in offered if c.name != honest.steps[0].candidate.name)
    swapped = replace(honest, steps=(replace(honest.steps[0], candidate=other),))
    assert "R9" not in _laws(verify_plan(module, swapped, _H))


# --- R10: the pack realizes the module, and comes from the plan ----------------------------

def test_r10_refuses_a_pack_that_realizes_nothing() -> None:
    """The empty pack: every provenance loop was vacuous, so the law held over nothing."""
    module = vector_add()
    result = optimize(module, _H, _TH, PERF)
    pack = hydrate(module, result)
    assert not verify_pack(module, pack, result), "the honest pack must stay clean"

    empty = replace(pack, segments=(), trace_notes=(), prefetches=())
    diags = verify_pack(module, empty, result)
    assert "R10" in _laws(diags)
    assert any("no StreamPack segment" in d.message for d in diags), diags


def test_r10_binds_the_pack_to_the_plan_it_was_hydrated_from() -> None:
    """Segment-to-claim provenance is not the same as pack-to-plan provenance.

    The claim ids can all be right while the realization the pack encodes is one the plan
    never chose -- and the certificate prices the plan.
    """
    module = vector_add()
    wide = optimize(module, _H, _TH, PERF)
    narrow = optimize(module, replace(_H, lane_widths=(1,)), _TH, PERF)
    assert wide.steps[0].candidate.width != narrow.steps[0].candidate.width, (
        "the fixture needs two plans that differ")

    pack = hydrate(module, wide)
    assert not verify_all(module, wide, pack, h=_H)

    diags = verify_all(module, narrow, pack)
    assert "R10" in _laws(diags), "a pack from another plan was accepted"


# --- the public artifact API ---------------------------------------------------------------

def test_build_artifact_does_not_attest_an_illegal_module() -> None:
    """`attested` reads as "this artifact is legal"; it meant "the C matches the plan".

    A faithful lowering of an illegal plan is exactly the case a deployable-artifact API
    must refuse to bless, and R5 -- a volatile access without an ordered hazard -- is a
    legality question the emitted C cannot answer.
    """
    module = Module(name="volatile_add")
    for rid in (1, 2, 3):
        module.add_resource(Resource(rid=rid, shape=(64,)))
    module.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1000, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=64, rd=(1, 2), wr=(3,), op="vector.add",
              hazard="unique", volatile=True)]))
    assert "R5" in _laws(verify(module)), "the fixture must actually be illegal"

    artifact = build_artifact(module)
    assert artifact.attested is False
    assert "R5" in {law for law, _ in artifact.diagnostics}

    # The legal program still attests, and still carries no diagnostics.
    ok = build_artifact(vector_add())
    assert ok.attested is True and ok.diagnostics == ()


# --- the lowering subset says what it does not cover ----------------------------------------

def test_the_lowering_subset_refuses_addressing_it_does_not_emit() -> None:
    """`offset` and `stride_k` were accepted and dropped, and R12 called the result clean.

    Both emitters go through one selection gate, so one refusal covers the LLVM and C
    rails together -- which is also why the defect appeared in both.
    """
    def module(**kw) -> Module:
        m = Module(name="addr")
        for rid in (1, 2, 3):
            m.add_resource(Resource(rid=rid, shape=(64,)))
        fields = dict(id=1000, opcode=Opcode.ADD, lane=Lane.U,
                      stride_class=StrideClass.UNIT, count=64, rd=(1, 2), wr=(3,),
                      op="vector.add")
        fields.update(kw)
        m.add_phase(Phase(phase_id=0, claims=[Claim(**fields)]))
        return m

    baseline = build_artifact(module())
    assert baseline.attested is True

    for kw, why in (({"offset": 8}, "offset"),
                    ({"offset": 1}, "offset"),
                    ({"stride_k": 4, "stride_class": StrideClass.STRIDED}, "stride"),
                    ({"stride_k": 2, "stride_class": StrideClass.STRIDED}, "stride")):
        try:
            artifact = build_artifact(module(**kw))
        except NotImplementedError as exc:
            assert why in str(exc), (kw, str(exc))
            continue
        raise AssertionError(
            f"{kw} was lowered anyway, to:\n{artifact.kernel_c}")


# --- replay: a matching digest is not a matching plan ---------------------------------------

def test_replay_refuses_to_return_a_plan_the_manifest_does_not_record() -> None:
    """`hash_target` omits the memory hierarchy, and that gap is not closed here.

    Closing it means new ODS attributes on `TargetCapabilityOp` plus a matching walk in
    `BCIRVerifyPass.cpp`'s `hashTargetFromIR`, because R13 cross-checks this hash against
    one recomputed from the IR -- a two-rail change, tracked with the GEM+ scope work.

    What IS closed is the consequence: `replay` compared only the digest, so it returned a
    plan scoring 1,574,912 for a manifest recording 51,200 and raised nothing, while
    `reproduces()` on the same inputs correctly said False. A known-incomplete hash is
    survivable; a replay that silently answers with a different plan is not.
    """
    from bcir.kbcir.cost import MemoryHierarchy, Tier

    module = Module(name="dram")
    for rid in (1, 2, 3):
        module.add_resource(Resource(rid=rid, shape=(4096,)))
    module.add_phase(Phase(phase_id=0, claims=[
        Claim(id=10, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=4096, rd=(1, 2), wr=(3,), op="vector.add", cost_class="bandwidth")]))

    slower = replace(_H, mem=MemoryHierarchy(tuple(
        Tier(t.name, t.latency_cyc, t.bw_factor * 32, t.lat_factor * 32, t.capacity)
        if t.name == "DRAM" else t for t in _H.mem.tiers)))

    manifest = build_manifest(module, _H, _TH, PERF)
    baseline = optimize(module, _H, _TH, PERF)
    altered = optimize(module, slower, _TH, PERF)
    assert altered.score != baseline.score, "the fixture needs the change to move the plan"

    # Replaying against the inputs it was built from still works.
    assert replay(manifest, module, _H, _TH, PERF).score == baseline.score

    try:
        got = replay(manifest, module, slower, _TH, PERF)
    except ProvenanceMismatch as exc:
        assert "DIFFERENT plan" in str(exc), str(exc)
        return
    raise AssertionError(
        f"replay returned a plan scoring {got.score} for a manifest recording "
        f"{manifest.score}")


# --- the emitter must not assert an alias fact the claim contradicts -------------------------

def test_noalias_is_emitted_only_where_the_rids_prove_it() -> None:
    """`noalias` is an assertion the caller must honour, not a hint.

    LLVM reorders loads and stores across it and violating it is undefined behaviour, so a
    blanket `noalias` is not a missed optimization -- it is a false fact handed to the
    backend. The emitter wrote it on all three pointers unconditionally, which is wrong the
    moment a claim writes a resource it also reads: `Claim(rd=(1, 2), wr=(1,))` is
    `A[i] = A[i] + B[i]`, an ordinary in-place graph the subset gate accepts, and A and C
    were both declared not to alias while both being resource 1.

    BCIR never had to infer this. A claim DECLARES its RIDs, so disjointness is a fact here
    rather than an analysis result -- which is why the fix loses nothing: a pointer that
    really is unaliased still gets the attribute.
    """
    from bcir.lower.llvm import emit_kernel_ll

    def signature(reads, writes):
        module = Module(name="alias")
        for rid in sorted(set(reads) | set(writes)):
            module.add_resource(Resource(rid=rid, shape=(64,)))
        module.add_phase(Phase(phase_id=0, claims=[
            Claim(id=1000, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
                  count=64, rd=reads, wr=writes, op="vector.add")]))
        text = emit_kernel_ll(module, optimize(module, _H, _TH, PERF))
        return next(line for line in text.splitlines() if line.startswith("define"))

    # Disjoint: every pointer keeps the attribute, so the optimization is not lost.
    disjoint = signature((1, 2), (3,))
    assert disjoint.count("noalias") == 3, disjoint

    # In place: A and C are one resource, so neither may claim it. B still may.
    in_place = signature((1, 2), (1,))
    assert in_place.count("noalias") == 1, in_place
    assert "ptr %A" in in_place and "ptr %C" in in_place, in_place
    assert "ptr noalias %B" in in_place, in_place

    # Two reads of one resource: A and B alias each other; only the write is disjoint.
    shared_reads = signature((1, 1), (2,))
    assert shared_reads.count("noalias") == 1, shared_reads
    assert "ptr noalias %C" in shared_reads, shared_reads


# --- R9: the scope re-derives what the planner priced (S0-A) ---------------------------------


def _chain_module() -> Module:
    """A producer->consumer pair in one phase: the consumer earns the deforestation
    discount, so its planned candidate's base is NOT the plain `candidates_for` base."""
    m = Module(name="chain")
    for rid in (1, 2, 3):
        m.add_resource(Resource(rid=rid, shape=(64,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(
                    id=1,
                    opcode=Opcode.ADD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=64,
                    rd=(1,),
                    wr=(2,),
                ),
                Claim(
                    id=2,
                    opcode=Opcode.MUL,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=64,
                    rd=(2,),
                    wr=(3,),
                ),
            ],
        )
    )
    return m


def test_r9_admits_the_planners_own_fused_plan() -> None:
    """The planner draws from `fused_candidates`; R9 must re-derive THAT set. Re-deriving
    from `candidates_for` rejected every fused consumer -- the audit's matmul fixture drew
    3,840 false R9 diagnostics out of 4,096 claims -- and no real caller passed `h`."""
    m = _chain_module()
    res = optimize(m, _H, _TH, PERF)
    assert (
        res.steps[1].candidate.base.v
        != tuple(
            c
            for c in candidates_for(m.phases[0].claims[1], _H, m.resource(2))
            if c.name == res.steps[1].candidate.name
        )[0].base.v
    ), "the fixture must be fused"
    assert not verify_plan(m, res, _H), verify_plan(m, res, _H)
    assert not verify_plan(m, res, _H, theta=_TH, policy=PERF)
    assert not verify_all(m, result=res, h=_H, theta=_TH, policy=PERF)


def test_r9_refuses_a_forged_step_cost_only_the_scope_can_see() -> None:
    """A legitimate candidate carrying a forged realized cost: the score still sums, the
    candidate is admitted, every geometry rule holds. Only re-deriving the cost from
    (h, theta, policy) exposes it. Without theta the documented weaker mode passes it --
    pinned here so the gap stays a deliberate choice, never an accident."""
    m = _chain_module()
    honest = optimize(m, _H, _TH, PERF)
    cheap = replace(honest.steps[1], cost=honest.steps[1].cost // 2)
    forged = replace(
        honest, steps=[honest.steps[0], cheap], score=honest.steps[0].cost + cheap.cost
    )
    assert "R9" not in _laws(verify_plan(m, forged, _H))
    diags = verify_plan(m, forged, _H, theta=_TH, policy=PERF)
    assert "R9" in _laws(diags)
    assert any("does not re-derive" in d.message for d in diags), diags
    # the assessment's reproduction -- a zero-cost plan -- is refused on the same ground
    zero = replace(honest, score=0, steps=[replace(s, cost=0) for s in honest.steps])
    assert any(
        "does not re-derive" in d.message for d in verify_plan(m, zero, _H, theta=_TH, policy=PERF)
    )
    # and the policy is part of the scope: the same plan priced under another policy
    from bcir.kbcir.weights import ENERGY

    assert "R9" in _laws(verify_plan(m, honest, _H, theta=_TH, policy=ENERGY))


def test_r9_refuses_a_plan_that_exceeds_its_budget() -> None:
    """R(pi, Theta) <= B is part of legality (the LangRef central equation): a plan checked
    against caps it violates is refused with the dimension named, the constrained planner's
    plan under the same caps is admitted, and a budget without theta is a refused call, not
    a skipped check."""
    from bcir.kbcir.cost import DIMS
    from bcir.kbcir.rcsp import Budget, optimize_constrained, pareto_plans, plan_resources

    m = _chain_module()
    free = optimize(m, _H, _TH, PERF)
    thermal = DIMS.index("thermal")
    hot = plan_resources(free, _TH).v[thermal]
    cooler = [
        p
        for p in pareto_plans(m, _H, _TH, PERF, dims=("thermal",))
        if plan_resources(p, _TH).v[thermal] < hot
    ]
    assert cooler, "the fixture needs a thermal trade-off on its Pareto front"
    cap = plan_resources(cooler[0], _TH).v[thermal]
    tight = Budget.of(thermal=cap)
    diags = verify_plan(m, free, _H, theta=_TH, policy=PERF, budget=tight)
    assert "R9" in _laws(diags)
    assert any("exceeds its budget" in d.message and "thermal" in d.message for d in diags), diags
    fitted = optimize_constrained(m, _H, _TH, PERF, tight)
    assert not verify_plan(m, fitted, _H, theta=_TH, policy=PERF, budget=tight)
    try:
        verify_plan(m, free, _H, budget=tight)
    except ValueError:
        pass
    else:
        raise AssertionError("budget feasibility without theta must be refused, not skipped")


# --- EV1-EV3 are module laws, so the canonical verifier carries them ----------------------


def test_ev_laws_are_part_of_the_canonical_verifier() -> None:
    """EV1-EV3 lived in `kbcir.events` and `verify_all` never invoked them: an unarmed event
    phase reported EV2 there while the canonical chain called the module lawful. They are
    module laws; `verify(module)` carries them, vacuous over the eventless corpus."""
    from bcir.kbcir.events import mask_claim, unmask_claim

    def irq_module(armed: bool) -> Module:
        m = Module(name="irq")
        m.add_resource(Resource(rid=1, shape=(1,)))  # the controller (IER)
        m.add_resource(Resource(rid=2, shape=(16,)))  # the ring the handler fills
        m.add_resource(Resource(rid=3, shape=(16,)))  # program-only data
        program = [
            Claim(
                id=1,
                opcode=Opcode.ADD,
                lane=Lane.U,
                stride_class=StrideClass.UNIT,
                count=16,
                rd=(3,),
                wr=(3,),
            )
        ]
        if armed:
            program.append(unmask_claim("uart.rx", 1, 3))
        m.add_phase(Phase(phase_id=0, claims=program))
        m.add_phase(
            Phase(
                phase_id=1,
                event="uart.rx",
                claims=[
                    Claim(
                        id=2,
                        opcode=Opcode.STORE,
                        lane=Lane.U,
                        stride_class=StrideClass.SCALAR,
                        count=1,
                        wr=(2,),
                    )
                ],
            )
        )
        return m

    assert "EV2" in _laws(verify(irq_module(armed=False)))
    assert "EV2" in _laws(verify_all(irq_module(armed=False)))
    assert not {law for law in _laws(verify(irq_module(armed=True))) if law.startswith("EV")}
    # the well-formedness sub-law holds on an eventless module too
    bad = Module(name="mask")
    bad.add_resource(Resource(rid=1, shape=(1,)))
    bad.add_resource(Resource(rid=2, shape=(1,)))
    wide = replace(mask_claim("uart.rx", 1, 9), wr=(1, 2))
    bad.add_phase(Phase(phase_id=0, claims=[wide]))
    assert "EV" in _laws(verify(bad))
    # vacuous over the eventless corpus
    assert not {law for law in _laws(verify(vector_add())) if law.startswith("EV")}
