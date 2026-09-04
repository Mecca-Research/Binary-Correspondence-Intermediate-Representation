"""precision="compensated" C-kernel + the R17 accuracy-contract law (dual-rail).

Roadmap (BCIR_MASTER_ROADMAP.md §5.2 / §6): lower the compensated Q8 reduction the
precision module models, and add an accuracy-contract verifier law. `emit_compensated_
reduce_c` emits the residual-carry MAC (bit-identical to int64-exact; the naive form
drifts up to n ULP). `verify.verify_accuracy` (R17) is the contract: a claim that declares
`tolerance_ulp` must realize within it -- and a tight tolerance on a long reduction is
exactly the law that FORCES the compensated realization. (MLIR R17 parity is cross-checked
by mlir/test/passes/verify_laws_deep.mlir on the toolchain rail.)
"""

import shutil

from bcir.kbcir.precision import (
    accuracy_bound,
    compensated_reduce_q8,
    exact_reduce_q8,
    naive_reduce_q8,
)
from bcir.lower.c_kernel import compile_and_run_compensated_c, emit_compensated_reduce_c
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify_accuracy, verify_smart_lowering


def _no_clang():
    return not shutil.which("clang")


def _reduce_claim(count=1000, precision="", tol=0):
    return Claim(
        id=5000,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=count,
        rd=(50,),
        wr=(51,),
        op="reduce.add",
        domain=Domain.RAM,
        precision=precision,
        tolerance_ulp=tol,
    )


def _module(claim):
    m = Module(name="acc")
    m.add_resource(Resource(rid=50, domain=Domain.RAM, shape=(claim.count,)))
    m.add_resource(Resource(rid=51, domain=Domain.RAM, shape=(1,)))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[claim]))
    return m


# --- the compensated C kernel ----------------------------------------------------


def test_compensated_kernel_emits_residual_carry():
    c = emit_compensated_reduce_c()
    assert "residual-carry" in c.lower()
    assert "resid = (int32_t)(full & 0xFF)" in c  # carry the dropped low 8 bits
    assert "acc += (int32_t)(full >> 8)" in c  # floor-shift accumulate
    assert "restrict" in c
    assert "_Static_assert((-256 >> 8) == -1" in c  # requires arithmetic shift


def test_compensated_kernel_is_bit_identical_to_exact_c11_and_c23():
    if _no_clang():
        return  # _Static_assert + the kernel need a C11/C23 clang
    for std in ("c23", "c11"):
        ok, out = compile_and_run_compensated_c(std=std)
        assert ok, f"{std}: {out}"


def test_oracle_compensated_equals_exact_and_naive_drifts():
    vals = [1 + (i % 5) for i in range(1000)]
    w = 200
    assert compensated_reduce_q8(vals, w) == exact_reduce_q8(vals, w)
    drift = exact_reduce_q8(vals, w) - naive_reduce_q8(vals, w)
    assert 0 < drift <= len(vals)  # within the n-ULP bound, and real


# --- R17: the accuracy contract --------------------------------------------------


def test_accuracy_bound_naive_vs_compensated():
    c = _reduce_claim(1000)
    assert accuracy_bound(c, compensated=False) == 1000  # naive drifts O(count)
    assert accuracy_bound(c, compensated=True) == 1  # compensated: 1 ULP


def test_r17_fires_when_a_tight_tolerance_needs_compensation():
    # naive reduction over 1000 terms, tolerance 1 ULP -> bound 1000 > 1 -> R17.
    diags = verify_accuracy(_module(_reduce_claim(1000, precision="", tol=1)))
    assert [d.law for d in diags] == ["R17"]
    assert "exceeds tolerance" in diags[0].message


def test_r17_satisfied_by_the_compensated_realization():
    # the SAME claim with precision="compensated": bound 1 <= 1 -> legal.
    diags = verify_accuracy(_module(_reduce_claim(1000, precision="compensated", tol=1)))
    assert diags == []


def test_r17_is_a_noop_without_a_declared_tolerance():
    # the default (tolerance_ulp == 0) is unconstrained -- the corpus is unaffected.
    assert verify_accuracy(_module(_reduce_claim(1000, tol=0))) == []
    # an elementwise op truncates at most 1 ULP, so a 1-ULP tolerance is met.
    elt = Claim(
        id=1,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=4096,
        rd=(50,),
        wr=(51,),
        op="vector.add",
        domain=Domain.RAM,
        tolerance_ulp=1,
    )
    assert verify_accuracy(_module(elt)) == []


def test_verify_smart_lowering_runs_r17():
    diags = verify_smart_lowering(_module(_reduce_claim(1000, tol=1)))
    assert any(d.law == "R17" for d in diags)
