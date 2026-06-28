"""The count->structural parity proof for the cfront dual-rail gate.

The dual-rail parity gate used to reduce each emitted claim graph to a 9-INTEGER count summary
(`funcs=.. claims=.. mmio=.. bf=.. const=.. binop=.. call=.. repro=.. ok=..`) and compare the two
rails by string equality. The graphs were NEVER compared node-by-node, so ANY corruption that
PRESERVED the counts slipped through: swapping operands between two same-op claims, substituting one
`c.bin.*` op for another of the same arity, or redirecting a call `@foo`->`@bar` (both defined).

This module is the non-tautological demonstration that the new `digest=<hex>` field
(`bcir.verify.cfront_structural_digest`, byte-identical to the C twin's `bcir_cfront_digest`) closes
that gap: for each count-preserving mutation it asserts the 9 COUNTS stay identical (so the OLD gate
would miss it) while the STRUCTURAL DIGEST changes (so the NEW gate catches it) -- or, for an injected
duplicate claim id, that the new unit-wide R1.1 uniqueness law fires on both rails.

Mutations proven caught (counts unchanged, digest changes -- or R1.1 fires):
  - swap operands between two distinct-dataflow same-op claims;
  - REVERSE a NON-COMMUTATIVE op's two operands (a-b -> b-a, a/b -> b/a, a<b -> b<a) -- caught via the
    digest's POSITIONAL read order (commutative add/mul/and/or/xor/eq/ne stay order-insensitive, which
    is what keeps the cross-rail byte-identity);
  - substitute one `c.bin.*` op for another of the same arity;
  - redirect a call `@foo`->`@bar` (both defined);
  - tamper a c.const immediate (5 -> 999) -- caught via the c.const imm in its record;
  - change a c.cast width (uint8_t -> uint64_t) -- caught via the kept c.cast suffix;
  - read/write the WRONG struct MEMBER (s->x vs s->y) or take &s->x vs &s->y -- caught via the member
    byte OFFSET folded from the c.load / c.addrof / c.store imm;
  - read the WRONG BITFIELD (p->a vs p->b) or flip its SIGNEDNESS -- caught via the bit offset/width/
    sign folded from the c.bf.get / c.bf.set imm;
  - redirect a SINK claim's write target (return-temp / dead copy: `return t` -> `return u`) -- caught
    via the OBSERVABLE-OUTPUT anchor (the return value's last-writer value-number);
  - redirect a memory STORE's destination (*p=a -> *q=a) -- caught via the store's read operand and the
    stores anchor (dest-VN -> value-VN pairs);
  - inject a duplicate claim id, WITHIN and ACROSS functions -- caught by unit-wide R1.1 on both rails.

Honest limit: a pure INDEPENDENT-sibling reorder is a structural no-op (the two cfront rails legitimately
differ in sibling evaluation order on 11 fixtures, so claim position is not a cross-rail invariant); a
reorder that REWIRES a data dependency is caught. Commutative-op operand order is intentionally
order-insensitive (it cannot change the computed value). A non-member load's bounds metadata and a
store's _Bool/stride flags are dropped from the digest (rail-divergent), keeping byte-identity.

It also pins the foundational gate: the Python canonical serialization == the C twin's `--canon` dump
byte-for-byte on the whole fixture corpus (the byte-identity proof that makes the cross-rail digests
match). The corpus check self-skips when no C compiler is visible (the `quick` tier); the mutation
proof is pure Python and always runs.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront import compile_unit
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import (cfront_structural_canon, cfront_structural_digest,
                         cfront_unit_claim_ids_unique, verify)

_C = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "runtime", "c")


# A unit engineered to expose every mandated count-preserving corruption: two defined functions to
# redirect a call between (`foo`/`bar`), and two same-op `c.bin.add` claims whose operands have DISTINCT
# dataflow value-numbers (one adds a multiply result, the other a subtract result) so a swap between
# them is a real structural change -- not an isomorphic no-op.
_SRC = """
unsigned foo(unsigned x){ return x + 1u; }
unsigned bar(unsigned x){ return x + 2u; }
unsigned f(unsigned a, unsigned b){
  unsigned m = a * b;        /* a multiply */
  unsigned s = a - b;        /* a subtract */
  unsigned p = m + a;        /* add #1: reads the multiply result */
  unsigned q = s + b;        /* add #2: reads the subtract result */
  unsigned r = foo(p);       /* a direct call -- redirectable to bar */
  return p + q + r;
}
"""


def _counts(lowered) -> tuple:
    """The 9-count structural summary the OLD gate compared (entry function), as a tuple."""
    fns = lowered.functions
    lf = fns[next(reversed(fns))]
    cl = lf.claims
    return (
        len(fns), len(cl),
        sum(1 for c in cl if c.op == "c.load" and c.domain == Domain.MMIO),
        sum(1 for c in cl if c.op == "c.bf.get"),
        sum(1 for c in cl if c.op == "c.const"),
        sum(1 for c in cl if c.op.startswith("c.bin.")),
        sum(1 for c in cl if c.op.startswith("c.call")),
    )


def _lowered():
    return compile_unit(_SRC, check_clang=False).lowered


def _adds(lf):
    return [c for c in lf.claims if c.op == "c.bin.add"]


# --- the count-preserving-corruption proof (pure Python; always runs) -----------------------------

def test_swap_operands_between_same_op_claims_changes_digest_not_counts():
    """Swap the read operands between the two `c.bin.add` claims (rd permuted). The counts are
    untouched; the structural digest changes (the two adds now bind different value-number trees)."""
    base = _lowered()
    base_counts, base_digest = _counts(base), cfront_structural_digest(base)
    mut = copy.deepcopy(base)
    a = _adds(mut.functions["f"])
    assert len(a) >= 2, "fixture must have >=2 c.bin.add claims"
    a[0].rd, a[1].rd = a[1].rd, a[0].rd
    assert _counts(mut) == base_counts, "operand swap must preserve the 9 counts (old gate misses it)"
    assert cfront_structural_digest(mut) != base_digest, "operand swap must change the structural digest"


def test_substitute_binop_changes_digest_not_counts():
    """Substitute one `c.bin.add` for `c.bin.sub` (same arity, still a `c.bin.*` -> the binop count is
    unchanged). The structural digest changes (a claim's op-base differs)."""
    base = _lowered()
    base_counts, base_digest = _counts(base), cfront_structural_digest(base)
    mut = copy.deepcopy(base)
    a = _adds(mut.functions["f"])
    a[0].op, a[0].opcode = "c.bin.sub", Opcode.SUB
    assert _counts(mut) == base_counts, "op substitution must preserve the 9 counts (old gate misses it)"
    assert cfront_structural_digest(mut) != base_digest, "op substitution must change the structural digest"


def test_reverse_noncommutative_operands_changes_digest():
    """Reverse the two reads of a NON-COMMUTATIVE op (a-b -> b-a; a/b -> b/a; a<b -> b<a). The counts
    are untouched; the digest changes because non-commutative reads keep POSITIONAL order and the two
    params value-number distinctly (in:p0 vs in:p1) -- the false-negative the adversary found, closed.
    A commutative op's reversal is intentionally a no-op (it cannot change the computed value), which is
    also asserted (and is what preserves cross-rail byte-identity)."""
    for src, op in (("unsigned f(unsigned a,unsigned b){ return a-b; }", "c.bin.sub"),
                    ("unsigned f(unsigned a,unsigned b){ return a/b; }", "c.bin.div"),
                    ("unsigned f(unsigned a,unsigned b){ return a<b; }", "c.bin.lt")):
        base = compile_unit(src, check_clang=False).lowered
        base_counts, base_digest = _counts(base), cfront_structural_digest(base)
        mut = copy.deepcopy(base)
        cl = next(c for c in mut.functions["f"].claims if c.op == op)
        cl.rd = (cl.rd[1], cl.rd[0])
        assert _counts(mut) == base_counts, f"{op} reversal must preserve the 9 counts"
        assert cfront_structural_digest(mut) != base_digest, \
            f"reversing a non-commutative {op}'s operands must change the digest"

    # commutative add: reversal is a value-preserving no-op (order-insensitive -> cross-rail safe).
    add = compile_unit("unsigned f(unsigned a,unsigned b){ return a+b; }", check_clang=False).lowered
    add_digest = cfront_structural_digest(add)
    radd = copy.deepcopy(add)
    c = next(c for c in radd.functions["f"].claims if c.op == "c.bin.add")
    c.rd = (c.rd[1], c.rd[0])
    assert cfront_structural_digest(radd) == add_digest, "a commutative add reversal is a value no-op"


def test_constant_value_tamper_changes_digest_not_counts():
    """Tamper a c.const immediate (5 -> 999). The const-CLAIM count is unchanged (the 9-count summary
    counts claims, not values), but the digest changes because c.const's imm is folded into its
    record -- the imm-omission false-negative the adversary found, closed."""
    base = compile_unit("unsigned f(void){ return 5u; }", check_clang=False).lowered
    base_counts, base_digest = _counts(base), cfront_structural_digest(base)
    mut = copy.deepcopy(base)
    cst = next(c for c in mut.functions["f"].claims if c.op == "c.const")
    cst.imm = tuple(999 if int(v) == 5 else int(v) for v in cst.imm)
    assert _counts(mut) == base_counts, "a constant-value tamper preserves the 9 counts (old gate misses it)"
    assert cfront_structural_digest(mut) != base_digest, "a c.const imm tamper must change the digest"


def test_cast_width_change_changes_digest_not_counts():
    """Change a c.cast target width (uint8_t -> uint64_t). The counts are unchanged, but the digest
    changes because the c.cast `:WIDTH` suffix is kept in the record -- the cast-stripping false-negative
    the adversary found, closed (only `c.call.vaarg`'s rail-divergent suffix is stripped)."""
    base = compile_unit("unsigned f(unsigned a){ return (unsigned char)a; }", check_clang=False).lowered
    base_counts, base_digest = _counts(base), cfront_structural_digest(base)
    mut = copy.deepcopy(base)
    cast = next(c for c in mut.functions["f"].claims if c.op.startswith("c.cast:"))
    cast.op = "c.cast:uint64_t"
    assert _counts(mut) == base_counts, "a cast-width change preserves the 9 counts (old gate misses it)"
    assert cfront_structural_digest(mut) != base_digest, "a c.cast width change must change the digest"


def test_member_access_redirection_changes_digest():
    """Reading the WRONG struct member (`s->x` vs `s->y`) changes the digest -- the member byte OFFSET
    rides in the c.load (and c.addrof) imm, now folded into the record. The 9-count summary and R1.1 are
    blind to this (same op counts, same ids); the emitted C differs (`+0` vs `+4`) and runs differently.
    The imm-omission false-negative the adversary found, closed."""
    sx = cfront_structural_digest(compile_unit(
        "struct S{int x;int y;}; int f(struct S*s){return s->x;}", check_clang=False).lowered)
    sy = cfront_structural_digest(compile_unit(
        "struct S{int x;int y;}; int f(struct S*s){return s->y;}", check_clang=False).lowered)
    assert sx != sy, "reading s->x vs s->y must change the digest (member offset folded)"
    # &s->x vs &s->y (the c.addrof offset imm) likewise.
    ax = cfront_structural_digest(compile_unit(
        "struct S{int x;int y;}; int* f(struct S*s){return &s->x;}", check_clang=False).lowered)
    ay = cfront_structural_digest(compile_unit(
        "struct S{int x;int y;}; int* f(struct S*s){return &s->y;}", check_clang=False).lowered)
    assert ax != ay, "&s->x vs &s->y must change the digest (addrof offset folded)"


def test_bitfield_layout_and_signedness_change_digest():
    """A bitfield member redirection (`p->a` vs `p->b`: different bit-offset/width) and a
    signed-vs-unsigned bitfield (different sign-extension) change the digest -- the bit offset, width and
    signedness ride in the c.bf.get/c.bf.set imm, now folded. The 9-count summary is blind to all of it."""
    pa = cfront_structural_digest(compile_unit(
        "struct P{unsigned a:3; unsigned b:5;}; unsigned f(struct P*p){return p->a;}", check_clang=False).lowered)
    pb = cfront_structural_digest(compile_unit(
        "struct P{unsigned a:3; unsigned b:5;}; unsigned f(struct P*p){return p->b;}", check_clang=False).lowered)
    assert pa != pb, "bitfield p->a vs p->b must change the digest (bit-offset/width folded)"
    sgn = cfront_structural_digest(compile_unit(
        "struct P{signed a:4;}; int f(struct P*p){return p->a;}", check_clang=False).lowered)
    uns = cfront_structural_digest(compile_unit(
        "struct P{unsigned a:4;}; unsigned f(struct P*p){return p->a;}", check_clang=False).lowered)
    assert sgn != uns, "a signed-vs-unsigned bitfield must change the digest (sign bit folded)"


def test_sink_wr_redirect_changes_digest():
    """Redirecting a SINK claim's write target -- making the `u=a+b` copy write the RETURN temp instead
    of `u` -- turns `return t` into `return (a+b)` in the emitted C (12 -> 7, gcc-confirmed). No per-claim
    record changes (wr is rail-private and excluded), but the OBSERVABLE-OUTPUT anchor (the return
    value's LAST-writer value-number) changes -- the wr-absent-from-record false-negative the adversary
    found, closed."""
    base = compile_unit("int f(int a,int b){ int t=a*b; int u=a+b; return t; }", check_clang=False).lowered
    base_counts, base_digest = _counts(base), cfront_structural_digest(base)
    mut = copy.deepcopy(base)
    lf = mut.functions["f"]
    redirected = False
    for c in lf.claims:
        if c.op == "c.copy" and lf.return_rid not in c.wr and c.wr:
            # the copy whose result is a dead temp (u); point it at the return temp (t) -> return u
            consumed = any(int(c.wr[0]) in [int(r) for r in d.rd] for d in lf.claims)
            if not consumed:
                c.wr = (lf.return_rid,)
                redirected = True
                break
    assert redirected, "fixture must have a sink copy to redirect"
    assert _counts(mut) == base_counts, "a sink-wr redirect preserves the 9 counts (old gate misses it)"
    assert cfront_structural_digest(mut) != base_digest, \
        "a sink-wr redirect (return t -> return u) must change the digest via the observable-output anchor"


def test_store_target_redirect_changes_digest():
    """Redirecting a memory store's DESTINATION (`*p = a` -> `*q = a`) changes the digest -- the store's
    destination is a read operand AND appears in the observable-output stores anchor (dest-VN -> value-VN
    pairs). The node-redirection class, closed."""
    base = compile_unit("void g(int*p,int*q,int a){ *p=a; }", check_clang=False).lowered
    base_digest = cfront_structural_digest(base)
    mut = copy.deepcopy(base)
    lf = mut.functions["g"]
    store = next(c for c in lf.claims if c.op == "c.store")
    q_rid = lf.params[1][1]                              # redirect the store dest from p to q
    store.rd = (q_rid,) + tuple(store.rd[1:])
    assert cfront_structural_digest(mut) != base_digest, "a store-target redirect must change the digest"


def test_redirect_call_to_other_defined_function_changes_digest_not_counts():
    """Redirect the call `@foo`->`@bar` (both defined, so R18 stays clean and the call count is
    unchanged). The structural digest changes (the callee is part of the `c.call:NAME` op-base)."""
    base = _lowered()
    base_counts, base_digest = _counts(base), cfront_structural_digest(base)
    mut = copy.deepcopy(base)
    redirected = False
    for c in mut.functions["f"].claims:
        if c.op.startswith("c.call") and "foo" in c.op:
            c.op = c.op.replace("foo", "bar")
            redirected = True
            break
    assert redirected, "fixture must contain a direct call to foo"
    assert _counts(mut) == base_counts, "call redirect must preserve the 9 counts (old gate misses it)"
    assert cfront_structural_digest(mut) != base_digest, "call redirect must change the structural digest"


def test_reorder_independent_claims_is_a_structural_no_op_but_dependency_rewire_is_caught():
    """Reordering two INDEPENDENT sibling claims (the multiply and the subtract) is a genuine
    structural no-op: the two cfront rails themselves legitimately differ in sibling evaluation order
    on 11 corpus fixtures, so claim POSITION is not a cross-rail invariant and the order-invariant
    digest is (correctly) unchanged -- this is exactly what keeps the gate from false-positiving on
    those fixtures. A reorder that REWIRES a data dependency (so a consumer reads a different
    producer) DOES change the dataflow, and is caught."""
    base = _lowered()
    base_counts, base_digest = _counts(base), cfront_structural_digest(base)

    # (a) pure independent reorder -> structural no-op (counts AND digest unchanged).
    noop = copy.deepcopy(base)
    cl = noop.functions["f"].claims
    im = next(i for i, c in enumerate(cl) if c.op == "c.bin.mul")
    isb = next(i for i, c in enumerate(cl) if c.op == "c.bin.sub")
    cl[im], cl[isb] = cl[isb], cl[im]
    assert _counts(noop) == base_counts
    assert cfront_structural_digest(noop) == base_digest, "a benign independent reorder is a structural no-op"

    # (b) a dependency-rewiring corruption: point one add's operand at a different producer (a "reorder"
    # that changes which value flows where). The counts are unchanged; the digest changes.
    rewired = copy.deepcopy(base)
    a = _adds(rewired.functions["f"])
    a[0].rd = (a[1].rd[0],) + tuple(a[0].rd[1:])   # add #1 now reads add #2's first producer
    assert _counts(rewired) == base_counts
    assert cfront_structural_digest(rewired) != base_digest, "a dependency-rewiring reorder must be caught"


def test_duplicate_claim_id_is_caught_by_R1_1_uniqueness_law():
    """An INTRA-function duplicate/injected claim id is caught by the R1.1 uniqueness law (the Python
    per-module rail). The C twin enforces the same law -- see test_c_cfront's dual-rail R1.1 check. A
    clean unit with unique ids does not trip it."""
    dup = Module(name="dup")
    dup.add_resource(Resource(rid=1, shape=(4,)))
    dup.add_phase(Phase(phase_id=0, claims=[
        Claim(id=5, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=4, rd=(1,), wr=(1,)),
        Claim(id=5, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=4, rd=(1,), wr=(1,))]))
    assert "R1.1" in {d.law for d in verify(dup)}, "a duplicate claim id must raise R1.1"

    ok = Module(name="ok")
    ok.add_resource(Resource(rid=1, shape=(4,)))
    ok.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=4, rd=(1,), wr=(1,)),
        Claim(id=2, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=4, rd=(1,), wr=(1,))]))
    assert "R1.1" not in {d.law for d in verify(ok)}, "unique claim ids must NOT raise R1.1"


def test_cross_function_duplicate_claim_id_is_caught_unit_wide():
    """A CROSS-function duplicate claim id is caught by the unit-wide R1.1 pass on the Python rail
    (`cfront_unit_claim_ids_unique`) -- the rail-asymmetry the adversary found (the C twin checked
    unit-wide, the Python rail only per-function). A clean two-function unit does not trip it; injecting
    bar.claims[0].id := foo.claims[0].id makes R1.1 fire. The C twin's unit-wide R1.1 is the matching
    rail (test_c_cfront's R1.1 dual-rail check)."""
    res = compile_unit("unsigned foo(unsigned x){return x+1u;}\n"
                       "unsigned bar(unsigned x){return x+2u;}", check_clang=False)
    assert cfront_unit_claim_ids_unique(res.lowered) == [], "a clean unit must not raise R1.1"
    foo, bar = res.lowered.functions["foo"], res.lowered.functions["bar"]
    bar.claims[0].id = foo.claims[0].id                  # a cross-function duplicate id
    diags = cfront_unit_claim_ids_unique(res.lowered)
    assert any(d.law == "R1.1" for d in diags), \
        f"a cross-function duplicate claim id must raise unit-wide R1.1, got {diags}"


# --- the byte-identity proof (cross-rail; self-skips without a C compiler) -------------------------

def _cc():
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _fixture_includes(path: str) -> dict:
    src = open(path, encoding="utf-8").read()
    inc = {}
    for h in re.findall(r'#include\s+"([^"]+)"', src):
        p = os.path.join(_C, h)
        if os.path.exists(p):
            inc[h] = open(p, encoding="utf-8").read()
    return inc


def _corpus_fixtures() -> list[str]:
    return sorted(f for f in os.listdir(_C) if f.startswith("cfront_") and f.endswith(".c"))


def test_python_canon_equals_c_twin_canon_byte_for_byte_on_the_corpus():
    """The foundational gate: the Python `cfront_structural_canon` must equal the C twin's `--canon`
    dump byte-for-byte on every corpus fixture (so the FNV-1a digests are cross-rail identical). Builds
    the C frontend once, then diffs the canon of each fixture. Self-skips without a C compiler."""
    cc = _cc()
    if not cc:
        return  # quick tier: no C compiler visible -> the pure-Python mutation proof above still ran
    srcs = ["bcir_cfront.c", "bcir_cpp.c", "bcir_verify.c", "bcir_runtime.c", "test_cfront.c"]
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "tcf")
        built = False
        for std in ("c23", "c2x", "c11"):
            b = subprocess.run([cc, f"-std={std}", "-O2", "-I", _C,
                                *[os.path.join(_C, s) for s in srcs], "-o", exe],
                               capture_output=True, text=True)
            if b.returncode == 0:
                built = True
                break
        assert built, f"C frontend build failed:\n{b.stderr}"

        fixtures = _corpus_fixtures()
        assert len(fixtures) > 100, f"expected the full fixture corpus, found {len(fixtures)}"
        mismatches = []
        for fx in fixtures:
            path = os.path.join(_C, fx)
            c_out = subprocess.run([exe, "--canon", path], capture_output=True, text=True)
            if c_out.returncode != 0:
                continue  # a fixture the twin rejects (parse-err) has no claim graph to compare
            try:
                res = compile_unit(open(path, encoding="utf-8").read(), check_clang=False,
                                   includes=_fixture_includes(path) or None)
            except Exception:
                continue  # an oracle-unsupported fixture: not part of the dual-rail corpus
            py_canon = cfront_structural_canon(res.lowered)
            if c_out.stdout != py_canon:
                mismatches.append(fx)
        assert not mismatches, f"cross-rail canon diverged on: {mismatches}"


def test_digest_value_matches_c_twin_on_representative_fixtures():
    """The digest VALUE (not just the canon) is cross-rail identical: the C summary's `digest=<hex>`
    equals the Python `cfront_structural_digest`. Self-skips without a C compiler."""
    cc = _cc()
    if not cc:
        return
    srcs = ["bcir_cfront.c", "bcir_cpp.c", "bcir_verify.c", "bcir_runtime.c", "test_cfront.c"]
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "tcf")
        for std in ("c23", "c2x", "c11"):
            b = subprocess.run([cc, f"-std={std}", "-O2", "-I", _C,
                                *[os.path.join(_C, s) for s in srcs], "-o", exe],
                               capture_output=True, text=True)
            if b.returncode == 0:
                break
        else:
            raise AssertionError(f"C frontend build failed:\n{b.stderr}")
        for fx in ("cfront_callgraph.c", "cfront_regmap.c", "cfront_typeof.c", "cfront_extern.c"):
            path = os.path.join(_C, fx)
            out = subprocess.run([exe, path], capture_output=True, text=True).stdout
            m = re.search(r"digest=([0-9a-f]+)", out.splitlines()[0])
            assert m, f"C summary for {fx} carries no digest= field: {out.splitlines()[0]!r}"
            res = compile_unit(open(path, encoding="utf-8").read(), check_clang=False,
                               includes=_fixture_includes(path) or None)
            assert m.group(1) == f"{cfront_structural_digest(res.lowered):016x}", \
                f"{fx}: C digest {m.group(1)} != Python digest"
