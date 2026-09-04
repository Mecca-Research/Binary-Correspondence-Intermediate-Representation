"""D2 (ML/AI roadmap §8.2): shape/dtype promoted to first-class laws R22/R23 -- the R19-R21
six-artifact pattern applied to the "structurally valid tensors" guarantee.

R22 (shape consistency) has a STRUCTURAL surface on the model rail: a `gem.*` producer->consumer
seam (a gem claim writing a resource a later gem claim reads) hands over one tensor, so both ends
must declare the same element extent -- exactly the adjacency contract the fusion optimizer prices
(`matmul_activation`). R23 (dtype compatibility) has no model-rail field (dtype rides the gem op
attrs on MLIR and the spec args on the oracle), so its oracle surface is the SPEC level:
`verify.verify_ml_spec` promotes the E3-E6 `check_*` validator messages to numbered diagnostics
(dtype rules -> R23, shape/extent/kind rules -> R22). The MLIR rail carries both laws structurally
(`verifyR22/R23` in `-bcir-verify` over the gem op shape/dtype attrs, `verify_shape_dtype.mlir`).
Both laws are vacuous by default (no gem seam / no spec checked), the non-disturbance invariant.
"""

from bcir.examples import gather_reduce, matmul_activation, vector_add
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify_ml_spec, verify_shape, verify_smart_lowering


def _gem_mod(counts, ops=("gem.matmul", "gem.activation:relu")):
    """Two claims in one phase: claim 1 writes RID 3 with counts[0]; claim 2 reads RID 3 with
    counts[1] -- the producer->consumer seam under test."""
    m = Module(name="seam")
    for rid in range(1, 6):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(max(counts),)))
    c1 = Claim(
        id=1,
        opcode=Opcode.T_MACC,
        lane=Lane.T,
        stride_class=StrideClass.TILE,
        count=counts[0],
        rd=(1, 2),
        wr=(3,),
        op=ops[0],
        domain=Domain.RAM,
    )
    c2 = Claim(
        id=2,
        opcode=Opcode.MUL,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=counts[1],
        rd=(3,),
        wr=(4,),
        op=ops[1],
        domain=Domain.RAM,
    )
    m.add_phase(Phase(phase_id=0, deps=(), claims=[c1, c2]))
    return m


def test_r22_is_vacuous_without_gem_seams():
    # Non-disturbance: the scalar corpus has no gem seam, so R22 emits nothing.
    for m in (vector_add(), gather_reduce()):
        assert verify_shape(m) == []
        assert [d for d in verify_smart_lowering(m) if d.law == "R22"] == []


def test_r22_accepts_the_consistent_fusion_seam():
    # The canonical G2 epilogue (matmul -> activation over the same n elements) is legal -- the law
    # runs NON-vacuously over it and stays clean.
    assert verify_shape(matmul_activation()) == []


def test_r22_flags_a_shape_inconsistent_gem_seam():
    d = verify_shape(_gem_mod((1024, 512)))
    assert [x.law for x in d] == ["R22"] and "shape-inconsistent" in d[0].message, d
    # and the law rides verify_smart_lowering, like R19-R21.
    assert [x.law for x in verify_smart_lowering(_gem_mod((1024, 512))) if x.law == "R22"] == [
        "R22"
    ]


def test_r22_seam_requires_gem_on_both_ends():
    # A non-gem consumer (or producer) is not a tensor seam -- no false positive.
    assert verify_shape(_gem_mod((1024, 512), ops=("gem.matmul", "vector.add"))) == []
    assert verify_shape(_gem_mod((1024, 512), ops=("vector.add", "gem.activation:relu"))) == []


def test_r22_a_non_gem_rewrite_breaks_the_seam():
    # gem writes RID 3, a scalar claim REWRITES RID 3, then a gem claim reads it: the tensor seam
    # is broken by the intermediate writer, so the extent handover law no longer applies.
    m = _gem_mod((1024, 512))
    mid = Claim(
        id=3,
        opcode=Opcode.STORE,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=1,
        rd=(5,),
        wr=(3,),
        op="c.store",
        domain=Domain.RAM,
    )
    claims = m.phases[0].claims
    m.phases[0].claims[:] = [claims[0], mid, claims[1]]
    assert verify_shape(m) == []


def test_r23_spec_promotion_classifies_dtype_vs_shape():
    # The spec-level promotion: dtype messages -> R23, shape/extent/kind messages -> R22, the
    # family name prefixed. (The checkers stay in kbcir -- the two-truth quarantine -- and the
    # caller hands their messages over, preserving the sanctioned import direction.)
    d = verify_ml_spec(
        "transformer",
        ["d_model 5 not divisible by n_heads 2", "softmax/layernorm need f32, got i32"],
    )
    assert [x.law for x in d] == ["R22", "R23"], d
    assert d[0].message.startswith("transformer: ") and d[1].message.startswith("transformer: ")
    assert verify_ml_spec("recurrent", []) == []


def test_r23_promotes_the_real_e3_checker():
    # End-to-end over the real E3 validator: a transcendental block spec fed i32 violates the
    # quarantine dtype rule -> R23; a head-divisibility violation -> R22.
    from bcir.kbcir.transformer import TransformerBlockSpec, check_transformer

    good = TransformerBlockSpec(4, 2, 8, 3, batch=1)
    rows = good.rows
    assert (
        verify_ml_spec("transformer", check_transformer(good, (rows, 4), "f32", (rows, 4), "f32"))
        == []
    )
    laws = {
        d.law
        for d in verify_ml_spec(
            "transformer", check_transformer(good, (rows, 4), "i32", (rows, 4), "i32")
        )
    }
    assert "R23" in laws, laws
    bad = TransformerBlockSpec(5, 2, 8, 3, batch=1)
    laws = {
        d.law
        for d in verify_ml_spec(
            "transformer", check_transformer(bad, (bad.rows, 5), "f32", (bad.rows, 5), "f32")
        )
    }
    assert "R22" in laws, laws


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
