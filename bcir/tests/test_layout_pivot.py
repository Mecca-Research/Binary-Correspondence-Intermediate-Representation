"""G3 -- cost-model-driven SoA<->AoS layout pivot for a multi-field tensor resource.

``Resource.layout`` defaulted to ``"soa"`` and was NEVER analyzed or transformed (the MLIR lowering
hard-coded ``#bcir.layout<soa>``). G3 builds the pivot: it analyzes a multi-field resource's access pattern,
PRICES SoA vs AoS through the EXISTING K_BCIR cost model (the SAME memory/stride terms ``candidates_for``
scores every claim by -- NOT a bespoke discount), chooses the minimum-cost layout as a priced plan, records
it in a proof-carrying ``LayoutCertificate``, and -- THE GATE THAT MATTERS -- the layout change computes the
IDENTICAL values under SoA and AoS (only the addressing/stride changes).

Covers:
  * access-pattern analysis: a single-field sweep is UNIT under SoA / STRIDED(F) under AoS, a whole-record
    access is the mirror; ``candidates_for`` prices each shape and the cheaper layout wins;
  * a single-field-sweep-dominated workload prices SoA cheaper (chosen, a clean no-op on the default);
  * a whole-record-dominated workload prices AoS cheaper (chosen, a positive priced gain);
  * the certificate records the resource + chosen layout + SoA-vs-AoS cost + win, with the clean one-field
    / SoA-optimal no-op;
  * REFERENCE-INVARIANCE (the load-bearing gate): the SAME multi-field computation under SoA and AoS
    addressing computes BIT-IDENTICAL results -- proved in pure Python (no compiler) AND compiled+run
    (f32 + i32), bit-exact;
  * the emit + the MLIR emitter HONOR the chosen layout (the addressing macro + ``#bcir.layout<...>``).

Distinct from B3 (test_calling_side_tuning.py): B3 tunes the CALLING side of a WRAPPED kernel (cblas
major-order / prefetch / channel). G3 is the GENERAL memory layout of a multi-field resource IN THE CLAIM
GRAPH -- no wrapped call. Deterministic + pure-Python on the oracle rail; no MLIR C++ change.
"""

import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import replace

from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.kbcir.cost import MEMORY, TargetProfile
from bcir.kbcir.layout import (AOS, SINGLE_FIELD, SOA, WHOLE_RECORD, FieldAccess, LayoutCertificate,
                               LayoutPlan, analyze_resource_accesses, certify_layout, plan_layout,
                               pivot_resource_layout)
from bcir.kbcir.realize import candidates_for
from bcir.lower.layout_kernel import (compile_and_run_layout_c, emit_layout_kernel_c,
                                      emit_layout_selfcheck_c, reference_layout_out)

AVX = TargetProfile.x86_avx512()
F, N = 4, 1024


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _sweeps(n_claims=4, fields=F, records=N):
    return tuple(FieldAccess(claim_id=i, shape=SINGLE_FIELD, records=records, fields=fields)
                 for i in range(n_claims))


def _records(n_claims=4, fields=F, records=N):
    return tuple(FieldAccess(claim_id=i, shape=WHOLE_RECORD, records=records, fields=fields)
                 for i in range(n_claims))


# --- access-pattern analysis: the stride shape is the cost-model lever, not a discount -------------

def test_single_field_sweep_is_unit_under_soa_and_strided_under_aos():
    a = FieldAccess(claim_id=1, shape=SINGLE_FIELD, records=N, fields=F)
    assert a.stride_class_under(SOA) == StrideClass.UNIT and a.stride_k_under(SOA) == 1
    assert a.stride_class_under(AOS) == StrideClass.STRIDED and a.stride_k_under(AOS) == F


def test_whole_record_is_the_exact_mirror():
    a = FieldAccess(claim_id=1, shape=WHOLE_RECORD, records=N, fields=F)
    assert a.stride_class_under(AOS) == StrideClass.UNIT and a.stride_k_under(AOS) == 1
    assert a.stride_class_under(SOA) == StrideClass.STRIDED and a.stride_k_under(SOA) == F


def test_layout_cost_is_the_existing_candidates_memory_term_not_a_bespoke_discount():
    # the per-access cost IS the cheapest candidate's memory axis from realize.candidates_for -- the SAME
    # term every claim is priced by. A UNIT claim vectorizes + pays base_overhead*1; a STRIDED(F) claim is
    # pinned scalar + pays base_overhead*min(F, cacheline/elem), so it is genuinely more expensive.
    def best_mem(sc, sk):
        c = Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=sc, stride_k=sk,
                  count=N, rd=(1, 2), wr=(3,))
        return min(cand.base.v[MEMORY] for cand in candidates_for(c, AVX))
    unit = best_mem(StrideClass.UNIT, 1)
    strided = best_mem(StrideClass.STRIDED, F)
    assert strided > unit                       # the stride penalty makes the unfavorable layout costlier
    # and the layout planner's per-access cost equals exactly that candidate min (no extra factor).
    plan = plan_layout(rid=1, fields=F, accesses=(FieldAccess(1, SINGLE_FIELD, N, F),), target=AVX)
    assert plan.soa_cost == unit and plan.aos_cost == strided


# --- the priced choice: single-field -> SoA, whole-record -> AoS ----------------------------------

def test_single_field_sweep_workload_prices_soa_cheaper_and_chooses_it():
    plan = plan_layout(rid=7, fields=F, accesses=_sweeps(), target=AVX)
    assert plan.soa_cost < plan.aos_cost            # single-field sweeps are unit under SoA
    assert plan.is_soa and plan.layout == SOA       # the minimum-cost layout is chosen


def test_whole_record_workload_prices_aos_cheaper_and_chooses_it():
    plan = plan_layout(rid=7, fields=F, accesses=_records(), target=AVX)
    assert plan.aos_cost < plan.soa_cost            # whole-record access is unit under AoS
    assert plan.is_aos and plan.layout == AOS       # the cost model pivots to AoS


def test_mixed_workload_picks_the_dominant_shape():
    # 5 whole-record vs 1 single-field -> AoS wins (whole-record dominates); flip the ratio -> SoA wins.
    aos_heavy = _records(5) + _sweeps(1)
    soa_heavy = _records(1) + _sweeps(5)
    assert plan_layout(7, F, aos_heavy, AVX).layout == AOS
    assert plan_layout(7, F, soa_heavy, AVX).layout == SOA


def test_the_choice_is_a_genuine_cost_comparison_across_field_counts():
    # the AoS win grows with the record width F (the stride penalty caps at cacheline/elem), and at every
    # F>=2 a whole-record workload still strictly prefers AoS -- the cost model's monotone, priced decision.
    prev_gain = -1
    for f in (2, 3, 4, 8, 16):
        plan = plan_layout(7, f, _records(4, fields=f), AVX)
        cert = certify_layout(plan)
        assert plan.layout == AOS and cert.gain > 0
        assert cert.gain >= prev_gain               # the win is non-decreasing in F (stride penalty grows)
        prev_gain = cert.gain


# --- the certificate: proof-carrying record + the clean no-op -------------------------------------

def test_certificate_records_resource_chosen_layout_costs_and_win():
    plan = plan_layout(rid=42, fields=F, accesses=_records(), target=AVX)
    cert = certify_layout(plan)
    assert isinstance(cert, LayoutCertificate)
    assert cert.rid == 42 and cert.chosen_layout == AOS
    assert cert.soa_cost == plan.soa_cost and cert.aos_cost == plan.aos_cost
    assert cert.declared_layout == SOA
    assert cert.gain == plan.soa_cost - plan.aos_cost and cert.gain > 0
    assert not cert.is_noop


def test_certificate_is_a_clean_noop_when_soa_is_already_optimal():
    plan = plan_layout(rid=7, fields=F, accesses=_sweeps(), target=AVX)
    cert = certify_layout(plan)
    assert plan.is_soa and cert.gain == 0 and cert.is_noop      # single-field sweep stays on default SoA


def test_one_field_resource_is_a_clean_noop():
    # a resource with one field has no SoA/AoS distinction -> SoA, equal costs, a clean no-op.
    plan = plan_layout(rid=7, fields=1, accesses=_records(), target=AVX)
    cert = certify_layout(plan)
    assert plan.fields == 1 and plan.is_soa
    assert plan.soa_cost == plan.aos_cost and cert.gain == 0 and cert.is_noop


def test_empty_workload_is_a_noop():
    plan = plan_layout(rid=7, fields=F, accesses=(), target=AVX)
    cert = certify_layout(plan)
    assert plan.is_soa and cert.is_noop and plan.soa_cost == 0


def test_ties_keep_the_declared_default_soa():
    # if AoS does not STRICTLY win, the declared default SoA is kept (off-default adopted only on a win).
    plan = plan_layout(rid=7, fields=F, accesses=_sweeps(2) + _records(2), target=AVX)
    # equal numbers of mirror-cost accesses -> a tie -> SoA kept.
    assert plan.soa_cost == plan.aos_cost and plan.layout == SOA


def test_is_deterministic():
    a = pivot_resource_layout(_aos_module(), rid=1, fields=F, target=AVX)
    b = pivot_resource_layout(_aos_module(), rid=1, fields=F, target=AVX)
    assert a[0] == b[0]
    assert (a[1].soa_cost, a[1].aos_cost, a[1].gain) == (b[1].soa_cost, b[1].aos_cost, b[1].gain)


# --- end-to-end on a Module: analyze + price + certify --------------------------------------------

def _aos_module(fields=F, records=N):
    """A module whose claims touch resource rid=1 as WHOLE-RECORD accesses (op tagged :whole_record)."""
    m = Module(name="g3_aos")
    m.add_resource(Resource(rid=1, domain=Domain.RAM, shape=(records * fields,), name="records"))
    m.add_resource(Resource(rid=2, domain=Domain.RAM, shape=(fields,), name="weight"))
    m.add_resource(Resource(rid=3, domain=Domain.RAM, shape=(records,), name="out"))
    claims = [Claim(id=i, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=records,
                    rd=(1, 2), wr=(3,), op="reduce.record:whole_record") for i in range(3)]
    m.add_phase(Phase(phase_id=0, claims=claims))
    return m


def _soa_module(fields=F, records=N):
    """A module whose claims sweep ONE field of resource rid=1 (no :whole_record tag) -> SINGLE_FIELD."""
    m = Module(name="g3_soa")
    m.add_resource(Resource(rid=1, domain=Domain.RAM, shape=(records * fields,), name="fields"))
    m.add_resource(Resource(rid=3, domain=Domain.RAM, shape=(records,), name="out"))
    claims = [Claim(id=i, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=records,
                    rd=(1,), wr=(3,), op="vector.scale") for i in range(3)]
    m.add_phase(Phase(phase_id=0, claims=claims))
    return m


def test_analyze_resource_accesses_classifies_the_shapes():
    accesses = analyze_resource_accesses(_aos_module(), rid=1, fields=F)
    assert len(accesses) == 3 and all(a.shape == WHOLE_RECORD for a in accesses)
    accesses2 = analyze_resource_accesses(_soa_module(), rid=1, fields=F)
    assert len(accesses2) == 3 and all(a.shape == SINGLE_FIELD for a in accesses2)


def test_pivot_resource_layout_picks_aos_for_whole_record_module():
    plan, cert = pivot_resource_layout(_aos_module(), rid=1, fields=F, target=AVX)
    assert plan.layout == AOS and cert.gain > 0 and not cert.is_noop


def test_pivot_resource_layout_keeps_soa_for_single_field_module():
    plan, cert = pivot_resource_layout(_soa_module(), rid=1, fields=F, target=AVX)
    assert plan.layout == SOA and cert.is_noop


# --- the emit honors the chosen layout (addressing macro + MLIR attribute) ------------------------

def test_emit_kernel_honors_the_chosen_layout_addressing():
    soa_plan = LayoutPlan(rid=0, fields=F, layout=SOA, soa_cost=0, aos_cost=0, accesses=())
    aos_plan = LayoutPlan(rid=0, fields=F, layout=AOS, soa_cost=0, aos_cost=0, accesses=())
    ks = emit_layout_kernel_c(soa_plan, "k", records=8)
    ka = emit_layout_kernel_c(aos_plan, "k", records=8)
    assert "(f) * 8u + (r)" in ks and "layout=soa" in ks         # SoA: field-contiguous f*N+r
    assert "(r) * 4u + (f)" in ka and "layout=aos" in ka         # AoS: record-contiguous r*F+f
    assert "data[DATA(r, f)] * weight[f]" in ks                  # the same computation in both


def test_mlir_emitter_honors_the_chosen_layout():
    # the chosen layout flows into the resource attribute AND the path attributes (no hard-coded soa).
    from bcir.kbcir.cost import Theta
    from bcir.lower.mlir import to_mlir
    m = _aos_module()
    plan, _ = pivot_resource_layout(m, rid=1, fields=F, target=AVX)
    m.resources[1] = replace(m.resources[1], layout=plan.layout)
    mlir = to_mlir(m, AVX, Theta.cool(), run_pipeline=False)
    assert "layout = #bcir.layout<aos>" in mlir                  # the resource carries the chosen aos
    assert mlir.count("#bcir.layout<aos>") >= 2                  # resource + at least one path


def test_default_soa_module_mlir_is_unchanged():
    # a default-layout (soa) module still emits #bcir.layout<soa> everywhere (no regression / drift).
    from bcir.kbcir.cost import Theta
    from bcir.lower.mlir import to_mlir
    mlir = to_mlir(_soa_module(), AVX, Theta.cool(), run_pipeline=False)
    assert "#bcir.layout<aos>" not in mlir and "#bcir.layout<soa>" in mlir


# --- REFERENCE-INVARIANCE (the gate that matters): SoA result == AoS result, bit-exact ------------

def test_python_reference_invariance_soa_equals_aos_bit_exact():
    # pure Python, NO compiler: the SAME logical data[r,f] addressed as SoA (f*N+r) and AoS (r*F+f) yields
    # IDENTICAL out[r] = sum_f data[r,f]*weight[f]. A layout change moves addresses, never values.
    rng = random.Random(0x6E3)
    for _ in range(200):
        fields = rng.randint(1, 8)
        records = rng.randint(1, 12)
        data_soa = [0] * (records * fields)
        data_aos = [0] * (records * fields)
        for r in range(records):
            for f in range(fields):
                v = rng.randint(-9, 9)
                data_soa[f * records + r] = v          # SoA address
                data_aos[r * fields + f] = v           # AoS address -- same value
        weight = [rng.randint(-4, 4) for _ in range(fields)]
        soa_out, aos_out = reference_layout_out(fields, records, data_soa, data_aos, weight)
        assert soa_out == aos_out, (fields, records)   # bit-exact, every time


def test_compiled_invariance_f32_is_bit_exact():
    if not _cc():
        return                                          # quick tier hides the toolchain -> self-skip
    ok, out = compile_and_run_layout_c(F, 256, elem="f32")
    assert ok, out                                      # the emitted SoA + AoS kernels agree bit-exact


def test_compiled_invariance_i32_is_bit_exact():
    if not _cc():
        return
    ok, out = compile_and_run_layout_c(F, 256, elem="i32")
    assert ok, out


def test_compiled_invariance_across_field_counts():
    if not _cc():
        return
    for fields in (1, 2, 3, 5, 8):
        ok, out = compile_and_run_layout_c(fields, 64, elem="f32")
        assert ok, (fields, out)


def test_emitted_selfcheck_runs_both_real_kernels():
    # the self-check exercises the REAL emit path (bcir_layout_soa + bcir_layout_aos), not hand-written code.
    src = emit_layout_selfcheck_c(F, 16, elem="f32")
    assert "bcir_layout_soa" in src and "bcir_layout_aos" in src
    assert "SoA out == AoS out" in src
    if not _cc():
        return
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.c")
        open(path, "w").write(src)
        exe = os.path.join(d, "s")
        bld = subprocess.run([_cc(), "-std=c23", "-O2", "-Wall", "-Wextra", path, "-o", exe],
                             capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        assert run.returncode == 0 and "OK" in run.stdout, run.stdout + run.stderr


def test_emit_rejects_bad_fields():
    bad = LayoutPlan(rid=0, fields=0, layout=SOA, soa_cost=0, aos_cost=0, accesses=())
    try:
        emit_layout_kernel_c(bad, "k", records=8)
        assert False, "fields=0 should raise"
    except ValueError:
        pass
