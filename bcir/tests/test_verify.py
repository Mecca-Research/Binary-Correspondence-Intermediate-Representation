"""Verifier law tests: R1-R7 (module), R8-R9 (plan), R10-R11 (stream), R12 (lowering)."""

from dataclasses import replace

from bcir.examples import PROGRAMS, vector_add
from bcir.gem import hydrate
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.kbcir.realize import Candidate, ChosenStep, CostVector, RealizationResult
from bcir.lower.llvm import emit_kernel_ll
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import is_legal, verify, verify_all, verify_lowering, verify_pack, verify_plan


def _laws(diags):
    return {d.law for d in diags}


# --- positive: the canonical corpus passes the whole chain -----------------------


def test_vector_add_is_legal():
    assert is_legal(vector_add(1024))


def test_all_canonical_programs_pass_module_laws():
    for name, build in PROGRAMS.items():
        diags = verify(build())
        assert not diags, f"{name}: {diags}"


def test_full_chain_R1_R12_clean_on_vector_add():
    m = vector_add(1024)
    res = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    pack = hydrate(m, res)
    ll = emit_kernel_ll(m, res)
    assert verify_plan(m, res) == []
    assert verify_pack(m, pack) == []
    assert verify_lowering(m, res, ll) == []
    assert verify_all(m, result=res, pack=pack, ll_text=ll) == []


# --- R2 / R4 / R6 (existing structural laws) -------------------------------------


def test_undeclared_rid_is_R2():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(4,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(
                    id=1,
                    opcode=Opcode.ADD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=4,
                    rd=(1, 99),
                    wr=(1,),
                )
            ],
        )
    )
    assert "R2" in _laws(verify(m))


def test_phase_cycle_is_R4():
    m = Module(name="cyclic")
    m.add_phase(Phase(phase_id=0, deps=(1,)))
    m.add_phase(Phase(phase_id=1, deps=(0,)))
    assert "R4" in _laws(verify(m))


def test_illegal_lane_for_stride_is_R6():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(4,)))
    # UNIT access on a GGG lane is illegal geometry.
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(
                    id=1,
                    opcode=Opcode.ADD,
                    lane=Lane.GGG,
                    stride_class=StrideClass.UNIT,
                    count=4,
                    rd=(1,),
                    wr=(1,),
                )
            ],
        )
    )
    assert "R6" in _laws(verify(m))


# --- R3: domain legality ----------------------------------------------------------


def test_claim_domain_unbacked_by_resources_is_R3():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, domain=Domain.RAM, shape=(4,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[Claim(id=1, opcode=Opcode.ADD, count=4, rd=(1,), wr=(1,), domain=Domain.HBM)],
        )
    )
    assert "R3" in _laws(verify(m))


def test_mmio_write_with_unique_hazard_is_R3():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, domain=Domain.MMIO, shape=(4,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(
                    id=1, opcode=Opcode.STORE, count=4, wr=(1,), domain=Domain.MMIO, hazard="unique"
                )
            ],
        )
    )
    assert "R3" in _laws(verify(m))


def test_ham_access_on_mmio_is_R3():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, domain=Domain.MMIO, access="ham", shape=(4,)))
    assert "R3" in _laws(verify(m))


# --- R5: hazard legality ----------------------------------------------------------


def test_atomic_opcode_with_unique_hazard_is_R5():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(64,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(
                    id=1,
                    opcode=Opcode.ATOMIC_ADD,
                    lane=Lane.A,
                    stride_class=StrideClass.RANDOM,
                    count=64,
                    rd=(1,),
                    wr=(1,),
                    hazard="unique",
                )
            ],
        )
    )
    assert "R5" in _laws(verify(m))


def test_atomic_hazard_makes_atomic_claim_legal():
    m = Module(name="ok")
    m.add_resource(Resource(rid=1, shape=(64,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(
                    id=1,
                    opcode=Opcode.ATOMIC_ADD,
                    lane=Lane.A,
                    stride_class=StrideClass.RANDOM,
                    count=64,
                    rd=(1,),
                    wr=(1,),
                    hazard="atomic",
                )
            ],
        )
    )
    assert "R5" not in _laws(verify(m))


def test_ggg_tail_conflict_without_ordered_hazard_is_R5():
    # CT2 decouples the GGG tail from wave order: a same-phase conflict through a
    # sparse claim needs an atomic/barriered contract on both ends.
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(1024,)))
    m.add_resource(Resource(rid=2, shape=(1024,)))
    scatter = Claim(
        id=1,
        opcode=Opcode.GGG_STORE,
        lane=Lane.GGG,
        stride_class=StrideClass.RANDOM,
        count=1024,
        wr=(1,),
        hazard="unique",
        verify="bounds",
    )
    reader = Claim(
        id=2,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=1024,
        rd=(1,),
        wr=(2,),
        hazard="unique",
    )
    m.add_phase(Phase(phase_id=0, claims=[scatter, reader]))
    assert "R5" in _laws(verify(m))
    # Ordered hazard contracts on both ends discharge the law.
    scatter.hazard = reader.hazard = "barriered"
    assert "R5" not in _laws(verify(m))


# --- R7: bounds legality ----------------------------------------------------------


def test_static_overrun_is_R7():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(4,)))
    m.add_phase(
        Phase(phase_id=0, claims=[Claim(id=1, opcode=Opcode.ADD, count=8, rd=(1,), wr=(1,))])
    )
    assert "R7" in _laws(verify(m))


def test_strided_read_extent_counts_stride():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(1024,)))
    m.add_resource(Resource(rid=2, shape=(1024,)))
    # 1024 reads at stride 4 touch an extent of 4093 > 1024.
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(
                    id=1,
                    opcode=Opcode.MUL,
                    lane=Lane.U,
                    stride_class=StrideClass.STRIDED,
                    count=1024,
                    stride_k=4,
                    rd=(1,),
                    wr=(2,),
                )
            ],
        )
    )
    assert "R7" in _laws(verify(m))


def test_masked_without_bounds_verify_is_R7():
    # §5.12 item 4: R7 now SEES the masked metadata it used to skip. A masked (runtime-bounds-checked)
    # access that does not declare the `bounds` verify contract is a promotion the backend would emit
    # without a guard -- a silent loss of the check.
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(8,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(id=1, opcode=Opcode.LOAD, rd=(1,), wr=(1,), bounds="masked", verify="none")
            ],
        )
    )
    assert "R7" in _laws(verify(m))


def test_masked_with_bounds_verify_is_legal():
    m = Module(name="ok")
    m.add_resource(Resource(rid=1, shape=(8,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(id=1, opcode=Opcode.LOAD, rd=(1,), wr=(1,), bounds="masked", verify="bounds")
            ],
        )
    )
    assert "R7" not in _laws(verify(m))


def test_random_strict_bounds_without_runtime_verify_is_R7():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(1024,)))
    m.add_resource(Resource(rid=2, shape=(1024,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(
                    id=1,
                    opcode=Opcode.GGG_LOAD,
                    lane=Lane.GGG,
                    stride_class=StrideClass.RANDOM,
                    count=1024,
                    rd=(1,),
                    wr=(2,),
                    bounds="strict",
                    verify="none",
                )
            ],
        )
    )
    assert "R7" in _laws(verify(m))


def test_unknown_contract_mnemonics_are_R5_R7_R8():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(4,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(
                    id=1,
                    opcode=Opcode.ADD,
                    count=4,
                    rd=(1,),
                    wr=(1,),
                    hazard="wild",
                    bounds="wild",
                    cost_class="wild",
                )
            ],
        )
    )
    laws = _laws(verify(m))
    assert {"R5", "R7", "R8"} <= laws


# --- R8 / R9: plan laws -----------------------------------------------------------


def test_plan_missing_claim_is_R9():
    m = vector_add(1024)
    assert "R9" in _laws(verify_plan(m, RealizationResult([], 0)))


def test_plan_illegal_chosen_lane_is_R9():
    m = vector_add(1024)
    cand = Candidate(Lane.UX, 8, "ux_bucket", CostVector.zero())  # UNIT claim: UX illegal
    res = RealizationResult([ChosenStep(1000, 0, cand, 0)], 0)
    assert "R9" in _laws(verify_plan(m, res))


def test_plan_negative_cost_is_R8_and_score_mismatch_is_R9():
    m = vector_add(1024)
    cand = Candidate(Lane.U, 16, "vec16", CostVector.zero())
    res = RealizationResult([ChosenStep(1000, 0, cand, -1)], 7808)
    laws = _laws(verify_plan(m, res))
    assert "R8" in laws and "R9" in laws


def test_optimizer_plans_satisfy_R8_R9_for_all_targets():
    for h in TARGETS.values():
        for build in PROGRAMS.values():
            m = build()
            assert verify_plan(m, optimize(m, h, Theta.cool())) == []


# --- R10 / R11: stream laws -------------------------------------------------------


def _hydrated(n=1024):
    m = vector_add(n)
    res = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    return m, hydrate(m, res)


def test_segment_without_trace_note_is_R10():
    m, pack = _hydrated()
    pack.trace_notes.clear()
    assert "R10" in _laws(verify_pack(m, pack))


def test_segment_with_undeclared_prefetch_is_R10():
    m, pack = _hydrated()
    pack.prefetches.clear()  # segments still name pf0
    assert "R10" in _laws(verify_pack(m, pack))


def test_duplicate_stream_bindings_are_R10():
    m, pack = _hydrated()
    pack.segments.append(pack.segments[0])
    assert any("duplicate segment" in d.message for d in verify_pack(m, pack))

    m, pack = _hydrated()
    pack.trace_notes.append(pack.trace_notes[0])
    assert any("duplicate trace" in d.message for d in verify_pack(m, pack))

    m, pack = _hydrated()
    pack.prefetches.append(pack.prefetches[0])
    assert any("duplicate prefetch" in d.message for d in verify_pack(m, pack))


def test_generation_drift_is_R11():
    m, pack = _hydrated()
    # The registry drifts after hydration: the pack is stale and must rehydrate.
    m.resources[10] = replace(m.resources[10], data_gen=5)
    assert "R11" in _laws(verify_pack(m, pack))


def _hydrated_two_generations():
    """vector_add with two resources at different generations, hydrated (StreamPack v4):
    the header maxima are (3, 2); resource 11 sits under them at (1, 0)."""
    m = vector_add(1024)
    m.resources[10] = replace(m.resources[10], map_gen=3, data_gen=2)
    m.resources[11] = replace(m.resources[11], map_gen=1, data_gen=0)
    pack = hydrate(m, optimize(m, TARGETS["x86_avx512"], Theta.cool()))
    assert (pack.map_gen, pack.data_gen) == (3, 2)
    assert [(g.rid, g.map_gen, g.data_gen) for g in pack.generations] == [
        (10, 3, 2),
        (11, 1, 0),
        (12, 0, 0),
    ]
    assert verify_pack(m, pack) == []
    return m, pack


def _messages(diags, law="R11"):
    return [d.message for d in diags if d.law == law]


def test_generation_drift_under_the_maxima_is_R11():
    # S0-2 (StreamPack v4, R11 per resource): resource 11 moves from (1, 0) to (2, 1) while
    # resource 10 still holds the maxima (3, 2). The maxima-only rule cannot see it (the
    # RED witness: the header tags still equal the registry maxima); the per-resource
    # vector names the moved resource and the repair.
    m, pack = _hydrated_two_generations()
    m.resources[11] = replace(m.resources[11], map_gen=2, data_gen=1)
    live_max = (
        max(r.map_gen for r in m.resources.values()),
        max(r.data_gen for r in m.resources.values()),
    )
    assert live_max == (pack.map_gen, pack.data_gen)  # invisible to the maxima
    messages = _messages(verify_pack(m, pack))
    assert "stale StreamPack: RID 11 map_gen 1 != registry 2 (rehydrate: repack)" in messages
    assert "stale StreamPack: RID 11 data_gen 0 != registry 1 (rehydrate: replan)" in messages
    assert not [msg for msg in messages if "RID 10 " in msg or "RID 12 " in msg]


def test_resource_declared_after_hydration_is_R11():
    m, pack = _hydrated_two_generations()
    m.resources[13] = replace(m.resources[12], rid=13, name="E")
    messages = _messages(verify_pack(m, pack))
    assert messages == [
        "stale StreamPack: RID 13 was declared after hydration "
        "(no generation vector entry; rehydrate: repack)"
    ]


def test_vector_naming_an_undeclared_rid_is_R11():
    from bcir.gem.streampack import Generation

    m, pack = _hydrated_two_generations()
    pack.generations.append(Generation(13, 0, 0))  # the registry declares no RID 13
    messages = _messages(verify_pack(m, pack))
    assert messages == [
        "generation vector names RID 13, which the registry does not declare (rehydrate: repack)"
    ]


def test_vector_less_pack_against_a_registry_is_R11():
    # A v1-v3 artifact (no vector) is stale against any registry that declares resources:
    # the maxima alone cannot see a resource that moved under them.
    m, pack = _hydrated_two_generations()
    pack.generations = []
    messages = _messages(verify_pack(m, pack))
    assert len(messages) == 1 and messages[0].startswith(
        "stale StreamPack: no per-resource generation vector for a registry of 3 resource(s)"
    )
    # ... and a vector-less pack over an empty registry is judged by the maxima alone:
    # (3, 2) over no resources is stale, (0, 0) is clean (R10 aside, which still sees
    # the segment's undeclared RIDs and claim on the empty module).
    empty = Module(name="empty")
    assert _messages(verify_pack(empty, pack)) == [
        "stale StreamPack: map_gen 3 != registry 0 (rehydrate: repack)",
        "stale StreamPack: data_gen 2 != registry 0 (rehydrate: replan)",
    ]
    pack.map_gen, pack.data_gen = 0, 0
    assert _messages(verify_pack(empty, pack)) == []


def test_header_and_vector_maxima_must_agree_for_R11():
    m, pack = _hydrated_two_generations()
    pack.map_gen = 9  # the header summary disagrees with the vector
    messages = _messages(verify_pack(m, pack))
    assert messages == [
        "header map_gen/data_gen (9, 2) are not the generation vector's maxima (3, 2)"
    ]


def test_unsorted_or_duplicate_vector_is_R11():
    m, pack = _hydrated_two_generations()
    pack.generations.reverse()
    assert any("not in RID order" in msg for msg in _messages(verify_pack(m, pack)))
    m, pack = _hydrated_two_generations()
    pack.generations.append(pack.generations[0])
    assert any("twice" in msg for msg in _messages(verify_pack(m, pack)))


# --- R12: lowering legality -------------------------------------------------------


def _lowered(n=1024):
    m = vector_add(n)
    res = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    return m, res, emit_kernel_ll(m, res)


def test_invented_instruction_is_R12():
    m, res, ll = _lowered()
    assert "R12" in _laws(verify_lowering(m, res, ll.replace("fadd", "fdiv")))


def test_width_drift_is_R12():
    m, res, ll = _lowered()
    assert "R12" in _laws(verify_lowering(m, res, ll.replace("width=16", "width=4")))


def test_missing_discharge_note_is_R12():
    m, res, ll = _lowered()
    body = "\n".join(ll.splitlines()[1:])
    assert "R12" in _laws(verify_lowering(m, res, body))


def test_dropped_bounds_guard_is_R12():
    m, res, ll = _lowered()
    tampered = ll.replace(", %n", ", 9999").replace("i64 %n)", "i64 %m)")
    assert "R12" in _laws(verify_lowering(m, res, tampered))


def test_ordered_hazard_without_fence_is_R12():
    m, res, ll = _lowered()
    # Tighten the claim's contract after lowering: the fenceless kernel no longer
    # discharges it.
    m.phases[0].claims[0].hazard = "barriered"
    assert "R12" in _laws(verify_lowering(m, res, ll))


def test_non_divisible_count_is_discharged_by_the_scalar_epilogue():
    # S0-8: n not divisible by the selected width keeps the width; the runtime-n tail
    # contract (the `n & -W` bound plus the scalar epilogue) is the discharge R12 accepts.
    m = vector_add(1000)  # 1000 % 16 != 0
    res = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    ll = emit_kernel_ll(m, res)
    assert "width=16" in ll.splitlines()[0] and "epilogue=scalar" in ll.splitlines()[0]
    assert verify_lowering(m, res, ll) == []


def _r12_messages(m, res, ll):
    return [d.message for d in verify_lowering(m, res, ll) if d.law == "R12"]


def test_missing_tail_contract_is_R12():
    # S0-8: a vector kernel that steps to n (no `n & -W` bound), one without the scalar
    # epilogue, and one that does not declare the epilogue are each refused. The
    # parent emitter's kernel -- the vector loop bounded by %n itself -- is the first.
    m, res, ll = _lowered()
    unbounded = ll.replace("%nvec = and i64 %n, -16", "%nvec = add i64 %n, 0")
    assert any("no `and i64 %n, -16` mask" in msg for msg in _r12_messages(m, res, unbounded))
    no_epilogue = ll.replace("%c = fadd float %a, %b", "%c = fadd <16 x float> %va, %vb")
    assert any("no scalar epilogue" in msg for msg in _r12_messages(m, res, no_epilogue))
    undeclared = ll.replace("epilogue=scalar", "epilogue=none")
    assert any("declare epilogue=scalar" in msg for msg in _r12_messages(m, res, undeclared))
    # a scalar kernel claiming an epilogue is a geometry lie the other way
    scalar = emit_kernel_ll(m, res, width_override=1)
    assert verify_lowering(m, res, scalar, width_override=1) == []
    lying = scalar.replace("epilogue=none", "epilogue=scalar")
    assert any("declares epilogue=scalar" in msg for msg in _r12_messages(m, res, lying)) or any(
        "epilogue" in msg
        for msg in [d.message for d in verify_lowering(m, res, lying, width_override=1)]
    )
