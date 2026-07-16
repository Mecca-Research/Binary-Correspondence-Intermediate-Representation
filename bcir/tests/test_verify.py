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
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=4, rd=(1, 99), wr=(1,))]))
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
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ADD, lane=Lane.GGG, stride_class=StrideClass.UNIT,
              count=4, rd=(1,), wr=(1,))]))
    assert "R6" in _laws(verify(m))


# --- R3: domain legality ----------------------------------------------------------

def test_claim_domain_unbacked_by_resources_is_R3():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, domain=Domain.RAM, shape=(4,)))
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ADD, count=4, rd=(1,), wr=(1,), domain=Domain.HBM)]))
    assert "R3" in _laws(verify(m))


def test_mmio_write_with_unique_hazard_is_R3():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, domain=Domain.MMIO, shape=(4,)))
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.STORE, count=4, wr=(1,), domain=Domain.MMIO,
              hazard="unique")]))
    assert "R3" in _laws(verify(m))


def test_ham_access_on_mmio_is_R3():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, domain=Domain.MMIO, access="ham", shape=(4,)))
    assert "R3" in _laws(verify(m))


# --- R5: hazard legality ----------------------------------------------------------

def test_atomic_opcode_with_unique_hazard_is_R5():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(64,)))
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ATOMIC_ADD, lane=Lane.A, stride_class=StrideClass.RANDOM,
              count=64, rd=(1,), wr=(1,), hazard="unique")]))
    assert "R5" in _laws(verify(m))


def test_atomic_hazard_makes_atomic_claim_legal():
    m = Module(name="ok")
    m.add_resource(Resource(rid=1, shape=(64,)))
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ATOMIC_ADD, lane=Lane.A, stride_class=StrideClass.RANDOM,
              count=64, rd=(1,), wr=(1,), hazard="atomic")]))
    assert "R5" not in _laws(verify(m))


def test_ggg_tail_conflict_without_ordered_hazard_is_R5():
    # CT2 decouples the GGG tail from wave order: a same-phase conflict through a
    # sparse claim needs an atomic/barriered contract on both ends.
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(1024,)))
    m.add_resource(Resource(rid=2, shape=(1024,)))
    scatter = Claim(id=1, opcode=Opcode.GGG_STORE, lane=Lane.GGG,
                    stride_class=StrideClass.RANDOM, count=1024, wr=(1,), hazard="unique",
                    verify="bounds")
    reader = Claim(id=2, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
                   count=1024, rd=(1,), wr=(2,), hazard="unique")
    m.add_phase(Phase(phase_id=0, claims=[scatter, reader]))
    assert "R5" in _laws(verify(m))
    # Ordered hazard contracts on both ends discharge the law.
    scatter.hazard = reader.hazard = "barriered"
    assert "R5" not in _laws(verify(m))


# --- R7: bounds legality ----------------------------------------------------------

def test_static_overrun_is_R7():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(4,)))
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ADD, count=8, rd=(1,), wr=(1,))]))
    assert "R7" in _laws(verify(m))


def test_strided_read_extent_counts_stride():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(1024,)))
    m.add_resource(Resource(rid=2, shape=(1024,)))
    # 1024 reads at stride 4 touch an extent of 4093 > 1024.
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.STRIDED,
              count=1024, stride_k=4, rd=(1,), wr=(2,))]))
    assert "R7" in _laws(verify(m))


def test_masked_without_bounds_verify_is_R7():
    # §5.12 item 4: R7 now SEES the masked metadata it used to skip. A masked (runtime-bounds-checked)
    # access that does not declare the `bounds` verify contract is a promotion the backend would emit
    # without a guard -- a silent loss of the check.
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(8,)))
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.LOAD, rd=(1,), wr=(1,), bounds="masked", verify="none")]))
    assert "R7" in _laws(verify(m))


def test_masked_with_bounds_verify_is_legal():
    m = Module(name="ok")
    m.add_resource(Resource(rid=1, shape=(8,)))
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.LOAD, rd=(1,), wr=(1,), bounds="masked", verify="bounds")]))
    assert "R7" not in _laws(verify(m))


def test_random_strict_bounds_without_runtime_verify_is_R7():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(1024,)))
    m.add_resource(Resource(rid=2, shape=(1024,)))
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.GGG_LOAD, lane=Lane.GGG, stride_class=StrideClass.RANDOM,
              count=1024, rd=(1,), wr=(2,), bounds="strict", verify="none")]))
    assert "R7" in _laws(verify(m))


def test_unknown_contract_mnemonics_are_R5_R7_R8():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1, shape=(4,)))
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ADD, count=4, rd=(1,), wr=(1,),
              hazard="wild", bounds="wild", cost_class="wild")]))
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


def test_scalar_legalization_is_a_legal_discharge():
    # n not divisible by the selected width: the emitter legalizes to scalar and
    # the discharge note records it -- R12 accepts exactly that fallback.
    m = vector_add(1000)  # 1000 % 16 != 0
    res = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    ll = emit_kernel_ll(m, res)
    assert verify_lowering(m, res, ll) == []
