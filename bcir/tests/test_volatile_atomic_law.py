"""§5.14 Phase 2 (volatile/atomic, first pair): the VOLATILE qualifier is a structural ordering law.

The C frontend has always known volatility (`domain=MMIO`, `hazard="barriered"`, the `access="volatile"`
resource string tag) but the law rail saw it only through those side signals. This slice makes it
first-class on the claim: `Claim.volatile` (digest-excluded from R13, so vacuous-by-default over the whole
corpus -- the non-disturbance invariant), the R5 extension (a volatile access must carry an ordered
hazard), the bundle fence (a volatile claim is never reordered/bundled across, exactly like `barriered`),
the cfront stamp (every MMIO access claim now carries `volatile=True`), and the `to_mlir` emission
(`is_volatile = true`, emitted ONLY when set so the existing corpus stays byte-identical). The MLIR rail
mirrors each piece: the `is_volatile` claim attr, the `verifyR5` extension in `-bcir-verify`, and the
`-bcir-bundle` fence (`atomic_ops*.mlir` / `verify_volatile.mlir` FileCheck tests).
"""

from bcir.examples import gather_reduce, vector_add
from bcir.kbcir.bundle import _conflict, find_bundles
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify


def _mod(claims):
    m = Module(name="vol")
    for rid in range(1, 9):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(1024,)))
    m.add_phase(Phase(phase_id=0, deps=(), claims=list(claims)))
    return m


def _claim(cid, *, volatile=False, hazard="unique", rd=(1, 2), wr=(3,)):
    return Claim(id=cid, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
                 count=1024, rd=rd, wr=wr, op="vector.add", domain=Domain.RAM,
                 hazard=hazard, volatile=volatile)


def test_r5_volatile_with_unique_hazard_is_flagged():
    # A volatile access with the default `unique` hazard has no ordering contract -- R5 flags it.
    diags = [d for d in verify(_mod([_claim(1, volatile=True)])) if d.law == "R5"]
    assert len(diags) == 1 and "volatile access requires" in diags[0].message, diags


def test_r5_volatile_with_ordered_hazard_is_legal():
    for hz in ("atomic", "barriered"):
        diags = [d for d in verify(_mod([_claim(1, volatile=True, hazard=hz)])) if d.law == "R5"]
        assert diags == [], (hz, diags)


def test_volatile_law_is_vacuous_over_the_existing_corpus():
    # Non-disturbance: no existing claim opts into `volatile`, so the R5 extension emits nothing and
    # the corpus verifies unchanged (the same invariant R19/R20/R21 established for timing/lifetime).
    for m in (vector_add(), gather_reduce()):
        assert [d for d in verify(m) if "volatile" in d.message] == []


def test_bundle_never_bundles_across_a_volatile_claim():
    # The fence: a volatile claim conflicts with everything (mirrors the ASM3b `barriered` guard), so
    # find_bundles never places it in a bundle and never reorders another claim across it.
    a, b = _claim(1, rd=(1, 2), wr=(3,)), _claim(3, rd=(1, 7), wr=(8,))
    v = _claim(2, volatile=True, hazard="barriered", rd=(1, 4), wr=(5,))
    assert _conflict(a, v) and _conflict(v, b)                   # fences both neighbours
    assert not _conflict(a, b)                                   # the A-sharers are independent
    for bd in find_bundles(_mod([a, v, b])):
        assert 2 not in bd.claim_ids                             # the volatile claim joins no bundle


def test_cfront_mmio_claims_carry_the_volatile_qualifier():
    # The frontend stamp: an MMIO register access (a `volatile`-qualified struct member) lowers to a
    # claim with `volatile=True` (plus the pre-existing MMIO signals: domain, H lane, barriered hazard);
    # a plain RAM access stays `volatile=False`. The unit is R1-R8 clean -- MMIO claims already carried
    # `hazard="barriered"`, so the new R5 clause is satisfied by construction (non-disturbance).
    from bcir.frontends.cfront import compile_unit
    src = ("typedef volatile unsigned int reg32;\n"
           "struct dev { reg32 r; };\n"
           "unsigned int rd(volatile struct dev *p, unsigned int x) { return p->r + x; }\n")
    r = compile_unit(src, check_clang=False)
    assert r.is_clean, r.diagnostics
    claims = [c for fn in r.lowered.functions.values() for ph in fn.module.phases for c in ph.claims]
    mmio = [c for c in claims if c.domain == Domain.MMIO]
    assert mmio and all(c.volatile for c in mmio), [(c.op, c.volatile) for c in mmio]
    ram_access = [c for c in claims if c.domain == Domain.RAM and c.op in ("c.load", "c.store")]
    assert all(not c.volatile for c in ram_access)


def test_to_mlir_emits_is_volatile_only_when_set():
    # The law-rail emission: `is_volatile = true` appears IFF a claim opts in, so the existing
    # generated corpus (never volatile) stays byte-identical.
    from bcir.kbcir import TARGETS
    from bcir.kbcir.cost import Theta
    from bcir.lower.mlir import to_mlir

    h, theta = TARGETS["x86_avx512"], Theta.cool()
    base = to_mlir(vector_add(), h, theta)
    assert "is_volatile" not in base
    m = vector_add()
    m.phases[0].claims[0].volatile = True
    m.phases[0].claims[0].hazard = "barriered"                   # keep R5 legal
    assert "is_volatile = true" in to_mlir(m, h, theta)


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
