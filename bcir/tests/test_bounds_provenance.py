"""§5.14 Phase 2: the pointer EXTENT-PROVENANCE signal -- why an access is checked vs trusted,
carried first-class on the claim (`Claim.bounds_provenance`, "" default, R13-digest-excluded).

The cfront lowering stamps every access with the reason `_access_bounds` decided its contract:
`declared_extent` (a local/static array of known shape -> masked), `recovered_count` (a
malloc/calloc pointer with a stable recovered count -> masked), `snapshot_extent` (a pure count
expression snapshotted at the alloc -> masked), `mmio_register` / `string_literal` /
`unknown_extent` (trusted, with the reason), or "" (a scalar/member/non-indexed access -- no
extent story). R7's masked-needs-guard clause now surfaces the provenance in its diagnostic on
BOTH rails (the MLIR clause is newly dual-railed -- it was oracle-only), and `to_mlir` emits the
attr only when set, so the existing generated corpus stays byte-identical.
"""

from bcir.examples import gather_reduce, vector_add
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify


def _unit(src):
    from bcir.frontends.cfront import compile_unit

    r = compile_unit(src, check_clang=False)
    assert r.is_clean, r.diagnostics
    return [c for fn in r.lowered.functions.values() for ph in fn.module.phases for c in ph.claims]


def _accesses(claims, prov):
    return [c for c in claims if c.bounds_provenance == prov and c.op in ("c.load", "c.store")]


def test_declared_extent_is_stamped_on_known_array_accesses():
    claims = _unit("unsigned f(unsigned i){ unsigned a[8]; a[i & 7u] = i; return a[i & 7u]; }")
    got = _accesses(claims, "declared_extent")
    assert got and all(c.bounds == "masked" for c in got), [(c.op, c.bounds) for c in got]


def test_recovered_count_is_stamped_on_stable_malloc_extents():
    claims = _unit(
        "unsigned f(unsigned n){ unsigned m = n & 7u; unsigned *p = malloc(m * 4u);"
        " p[0] = 1u; unsigned r = p[0]; free(p); return r; }"
    )
    got = _accesses(claims, "recovered_count")
    assert got and all(c.bounds == "masked" for c in got), [
        (c.bounds_provenance, c.bounds) for c in claims
    ]


def test_snapshot_extent_is_stamped_on_expression_counts():
    claims = _unit(
        "unsigned f(unsigned n){ unsigned *p = malloc(((n & 7u) + 1u) * 4u);"
        " p[0] = 2u; unsigned r = p[0]; free(p); return r; }"
    )
    got = _accesses(claims, "snapshot_extent")
    assert got and all(c.bounds == "masked" for c in got), [
        (c.bounds_provenance, c.bounds) for c in claims
    ]


def test_trusted_accesses_carry_their_reason():
    # an unknown-extent pointer parameter: indexed, not recoverable -> trusted, reason carried.
    claims = _unit("unsigned f(unsigned *p, unsigned i){ return p[i & 3u]; }")
    got = _accesses(claims, "unknown_extent")
    assert got and all(c.bounds == "assumed_safe" for c in got)
    # an MMIO register access: trusted with the mmio reason (and volatile+barriered, from Phase 2).
    claims = _unit(
        "typedef volatile unsigned int reg32;\nstruct dev { reg32 r; };\n"
        "unsigned int rd(volatile struct dev *p){ return p->r; }\n"
    )
    got = [c for c in claims if c.bounds_provenance == "mmio_register"]
    assert got and all(c.domain == Domain.MMIO for c in got)


def test_r7_diagnostic_surfaces_the_provenance():
    m = Module(name="p")
    m.add_resource(Resource(rid=1, domain=Domain.RAM, shape=(64,)))
    m.add_resource(Resource(rid=2, domain=Domain.RAM, shape=(64,)))
    c = Claim(
        id=1,
        opcode=Opcode.LOAD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=64,
        rd=(1,),
        wr=(2,),
        op="c.load",
        domain=Domain.RAM,
        verify="none",
        bounds="masked",
        bounds_provenance="declared_extent",
    )
    m.add_phase(Phase(phase_id=0, deps=(), claims=[c]))
    d = [x for x in verify(m) if x.law == "R7"]
    assert d and "extent provenance: declared_extent" in d[0].message, d


def test_provenance_is_vacuous_over_the_existing_corpus():
    # Non-disturbance: no generated-corpus claim carries a provenance, and to_mlir stays identical.
    from bcir.kbcir import TARGETS
    from bcir.kbcir.cost import Theta
    from bcir.lower.mlir import to_mlir

    for m in (vector_add(), gather_reduce()):
        assert all(not c.bounds_provenance for ph in m.phases for c in ph.claims)
    assert "bounds_provenance" not in to_mlir(vector_add(), TARGETS["x86_avx512"], Theta.cool())


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
