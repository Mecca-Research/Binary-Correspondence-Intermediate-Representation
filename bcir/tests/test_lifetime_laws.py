"""R21 (pointer-lifetime legality: use-after-free / double-free) as a bcir.verify function over the
OPTIONAL `claim.lifetime` annotation -- the naked-pointer safety track (§5.12), the lifetime half of the
first increment. Like R19/R20 (timing) this is governed by NON-DISTURBANCE: a claim with no lifetime (the
default for the entire scalar / C-frontend subset) places no constraint, so the existing corpus verifies
identically. It becomes load-bearing once a frontend annotates its malloc/free.
"""

from bcir.examples import gather_reduce, vector_add
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.model import Claim, Domain, Lifetime, Module, Opcode, Phase, Resource
from bcir.verify import verify_lifetime, verify_smart_lowering

COOL = Theta.cool()


def _mod(claims):
    # rid 9 is "the pointer" (a heap object); rids 0/1 scratch.
    res = {r: Resource(rid=r, domain=Domain.RAM, elem_bytes=4, shape=(16,)) for r in (0, 1, 9)}
    return Module(resources=res, phases=[Phase(phase_id=0, claims=claims)])


# --- NON-DISTURBANCE: the whole existing subset carries no lifetime -> R21 is a NO-OP --------------


def test_lifetime_is_vacuous_without_annotations():
    assert verify_lifetime(_mod([Claim(id=0, opcode=Opcode.LOAD, rd=(9,), wr=(0,))])) == []


def test_existing_corpus_verifies_identically_with_lifetime_law_present():
    """The non-disturbance proof: real optimizer modules (no lifetime) emit no R21, so wiring it into
    verify_smart_lowering cannot perturb any existing verdict."""
    for build in (vector_add, gather_reduce):
        m = build(1 << 16)
        optimize(m, TARGETS["x86_avx512"], COOL)
        assert verify_lifetime(m) == []
        assert [d.law for d in verify_smart_lowering(m) if d.law == "R21"] == []


# --- R21: use-after-free / double-free -------------------------------------------------------------


def test_r21_alloc_use_free_is_legal():
    claims = [
        Claim(
            id=0, opcode=Opcode.STORE, rd=(0,), wr=(9,), lifetime=Lifetime("alloc")
        ),  # p = malloc()
        Claim(id=1, opcode=Opcode.LOAD, rd=(9,), wr=(1,)),  # use *p
        Claim(id=2, opcode=Opcode.STORE, rd=(9,), wr=(), lifetime=Lifetime("free")),  # free(p)
    ]
    assert verify_lifetime(_mod(claims)) == []


def test_r21_use_after_free_is_rejected():
    claims = [
        Claim(id=0, opcode=Opcode.STORE, rd=(0,), wr=(9,), lifetime=Lifetime("alloc")),
        Claim(id=1, opcode=Opcode.STORE, rd=(9,), wr=(), lifetime=Lifetime("free")),  # free(p)
        Claim(id=2, opcode=Opcode.LOAD, rd=(9,), wr=(1,)),  # use *p AFTER free
    ]
    d = verify_lifetime(_mod(claims))
    assert [x.law for x in d] == ["R21"] and "use-after-free" in d[0].message


def test_r21_double_free_is_rejected():
    claims = [
        Claim(id=0, opcode=Opcode.STORE, rd=(0,), wr=(9,), lifetime=Lifetime("alloc")),
        Claim(id=1, opcode=Opcode.STORE, rd=(9,), wr=(), lifetime=Lifetime("free")),
        Claim(
            id=2, opcode=Opcode.STORE, rd=(9,), wr=(), lifetime=Lifetime("free")
        ),  # free(p) AGAIN
    ]
    d = verify_lifetime(_mod(claims))
    assert [x.law for x in d] == ["R21"] and "double-free" in d[0].message


def test_r21_realloc_after_free_revalidates():
    """A fresh allocation into the same RID (a re-alloc / new epoch) clears the freed state, so a later
    use is legal again."""
    claims = [
        Claim(id=0, opcode=Opcode.STORE, rd=(0,), wr=(9,), lifetime=Lifetime("alloc")),
        Claim(id=1, opcode=Opcode.STORE, rd=(9,), wr=(), lifetime=Lifetime("free")),
        Claim(
            id=2, opcode=Opcode.STORE, rd=(0,), wr=(9,), lifetime=Lifetime("alloc", epoch=1)
        ),  # re-alloc
        Claim(id=3, opcode=Opcode.LOAD, rd=(9,), wr=(1,)),  # use the new object
    ]
    assert verify_lifetime(_mod(claims)) == []


def test_r21_rejects_an_unknown_lifetime_event():
    claims = [Claim(id=0, opcode=Opcode.LOAD, rd=(9,), wr=(1,), lifetime=Lifetime("bogus"))]
    assert [x.law for x in verify_lifetime(_mod(claims))] == ["R21"]


# --- frontend integration: malloc/free annotation makes R21 load-bearing for the C compiler -----------


def _r21_over_unit(src: str):
    """Lower C via the oracle frontend and run R21 over every function's claim module. `compile_unit`'s
    own pass/fail is R1-R8 (R21 is a smart-lowering law run separately), so this never changes the
    frontend verdict -- it shows R21 CAN now catch the dangling access the annotated free() exposes."""
    from bcir.frontends.cfront import compile_unit

    r = compile_unit(src, check_clang=False)
    diags = []
    for name in r.lowered.functions:
        diags += verify_lifetime(r.lowered.functions[name].module)
    return r.is_clean, [d.law for d in diags]


def test_r21_catches_c_use_after_free():
    clean, laws = _r21_over_unit(
        "unsigned f(unsigned n){ unsigned *p = malloc(n * 4u); free(p); return p[0]; }"
    )
    assert clean and laws == ["R21"]  # frontend stays clean (R1-R8); R21 flags the deref


def test_r21_catches_c_double_free():
    clean, laws = _r21_over_unit(
        "void f(unsigned n){ unsigned *p = malloc(n * 4u); free(p); free(p); }"
    )
    assert clean and laws == ["R21"]


def test_r21_is_clean_on_a_well_formed_malloc_use_free():
    clean, laws = _r21_over_unit(
        "unsigned f(unsigned n){ unsigned *p = malloc(n * 4u); unsigned s = 0u;"
        " for(unsigned i=0u;i<(n&7u);i++){ p[i]=i; s+=p[i]; } free(p); return s; }"
    )
    assert clean and laws == []  # malloc -> use -> free at the end: no dangling access


def test_r21_allows_reassignment_after_free():
    """`free(p); p = malloc(...); use *p` is legal -- the WRITE re-validates the pointer."""
    clean, laws = _r21_over_unit(
        "unsigned f(void){ unsigned *p = malloc(4u); free(p); p = malloc(8u);"
        " p[0] = 1u; unsigned r = p[0]; free(p); return r; }"
    )
    assert clean and laws == []
