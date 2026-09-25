"""G10 (S5-B): escape analysis, indirect-call narrowing and the effect footprint of a lowered C unit
(`bcir/frontends/cfront/escape.py`).

Every unit here is written into the test, so the module runs from the installed package; the
cfront corpus and the C twin's byte-identical reports are gated in `test_c_cfront.py`. Each test
drives a failure the parent shipped, or a rule the analysis could get wrong: a store recorded as a
read, a write through a pointer, a static local, a heap buffer, a pointer made from an integer, an
ops-table callback, a device access, a call with more operands than a C-twin claim holds -- and the
property the two rails' parity rests on, that the answer does not depend on the order the claims
are visited in.
"""

from __future__ import annotations

import dataclasses
import os
import random
import shutil

from bcir.frontends.cfront import compile_unit
from bcir.model.lanes import Domain
from bcir.frontends.cfront.escape import (
    UNKNOWN,
    Footprint,
    analyze,
    effects_report,
    escape_report,
)
from bcir.tests import escape_fixtures as ef


def _unit(src: str):
    return compile_unit(src, check_clang=False)


def _foot(res, fn):
    fp = res.escape.footprints[fn]
    return set(fp.reads), set(fp.writes)


# --- the footprint: every write is a write -------------------------------------------------------

_STORES = """
struct pair { unsigned a; unsigned b; };
unsigned g_arr[4];
struct pair g_st;
unsigned g_buf[4];
void arr_w(unsigned v) { g_arr[0] = v; }
unsigned arr_r(void) { return g_arr[0]; }
void st_w(unsigned v) { g_st.b = v; }
unsigned st_r(void) { return g_st.b; }
static void pw(unsigned *p) { p[0] = 1u; }
static unsigned pr(unsigned *q) { return q[0]; }
void via_w(void) { pw(g_buf); }
unsigned via_r(void) { return pr(g_buf); }
unsigned next_id(void) { static unsigned n; n = n + 1u; return n; }
unsigned two_ids(void) { return next_id() + next_id(); }
"""


def test_a_store_is_a_write_and_a_reader_of_the_same_memory_conflicts_with_it():
    """The parent took a write from a claim's `wr` alone, and a cfront store is `c.store
    rd=(base, [index,] value) wr=()`: every store read as a read, so each pair below was reported
    to commute. Each writes what the other reads."""
    r = _unit(_STORES)
    assert "g_arr" in _foot(r, "arr_w")[1]
    assert "g_st" in _foot(r, "st_w")[1]
    assert _foot(r, "pw")[1] == {"g_buf"}  # a write through a pointer parameter, bound to g_buf
    assert _foot(r, "next_id")[1] == {"next_id.n"}  # a static local, named `function.name`
    for writer, reader in (
        ("arr_w", "arr_r"),
        ("st_w", "st_r"),
        ("via_w", "via_r"),
        ("pw", "pr"),
        ("next_id", "two_ids"),
    ):
        assert not r.commute(writer, reader), (writer, reader)
        assert not r.commute(reader, writer), (reader, writer)
    assert r.commute("via_r", "pr")  # two readers of one array
    assert r.commute("arr_w", "st_w")  # disjoint globals


def test_a_pointer_parameter_of_an_exported_function_may_point_anywhere():
    """A non-static function can be called from another unit with any pointer: what it writes
    through one is unknown memory, `*`, which conflicts with any other function's memory."""
    r = _unit(
        "unsigned g;\n"
        "void w(unsigned *p) { p[0] = 1u; }\n"
        "unsigned rg(void) { return g; }\n"
        "unsigned pure(unsigned x) { return x + 1u; }\n"
    )
    assert _foot(r, "w") == (set(), {UNKNOWN})
    assert not r.commute("w", "rg")
    assert r.commute("w", "pure")  # a function touching no memory commutes with anything


def test_unknown_memory_conflicts_with_every_name_and_nothing_empty():
    star = Footprint(frozenset(), frozenset({UNKNOWN}))
    reader = Footprint(frozenset({"g"}), frozenset())
    other_star = Footprint(frozenset({UNKNOWN}), frozenset())
    pure = Footprint(frozenset(), frozenset())
    assert star.conflicts(reader) and reader.conflicts(star)
    assert star.conflicts(other_star) and star.conflicts(star)
    assert not star.conflicts(pure) and not pure.conflicts(star)
    assert not reader.conflicts(other_star)  # two readers, one of unknown memory, commute


# --- memory no declaration names ------------------------------------------------------------------


def test_a_heap_buffer_is_its_allocators_and_a_helper_writing_it_conflicts_with_a_reader():
    """The parent's library rule gave `malloc` a result pointing nowhere: a helper writing the
    buffer and another reading it were reported to commute. The heap object belongs to the
    function that allocated it: private to it, unknown (`*`) to anyone else."""
    r = _unit(
        "#include <stdlib.h>\n"
        "static void w(unsigned *p) { p[0] = 1u; }\n"
        "static unsigned rd(unsigned *p) { return p[0]; }\n"
        "unsigned drive(void) { unsigned *h = malloc(16); w(h); unsigned x = rd(h); free(h);"
        " return x; }\n"
    )
    assert _foot(r, "w")[1] == {UNKNOWN}
    assert _foot(r, "rd")[0] == {UNKNOWN}
    assert not r.commute("w", "rd")
    assert _foot(r, "drive") == (set(), set())  # its own heap, private to its activation


def test_a_pointer_made_from_an_integer_points_to_unknown_memory_and_null_to_nothing():
    """`(T *)0x1000` and `(T *)addr` may be any address -- a device register, another unit's
    object -- so what is written through them is `*`; the null constant points nowhere."""
    r = _unit(
        "#include <stdint.h>\n"
        "unsigned g;\n"
        "unsigned from_const(unsigned v) { unsigned *p = (unsigned *)0x1000u; *p = v; return v; }\n"
        "static void via_int(uintptr_t a, unsigned v) { unsigned *p = (unsigned *)a; *p = v; }\n"
        "void drive(unsigned v) { via_int(0x2000u, v); }\n"
        "unsigned nullp(unsigned v) { unsigned *p = 0; if (v) p = &g; return p ? *p : 0u; }\n"
    )
    assert UNKNOWN in _foot(r, "from_const")[1]
    assert UNKNOWN in _foot(r, "via_int")[1]
    assert UNKNOWN in _foot(r, "drive")[1]
    assert _foot(r, "nullp") == ({"g"}, set())  # the null constant adds no unknown pointee


def test_a_callback_in_a_file_scope_ops_table_has_callers_the_unit_cannot_see():
    """A static function's pointer parameters hold only what its callers pass -- unless a
    file-scope initializer hands it out: whoever reads `table` can call `handler` with anything."""
    r = _unit(
        "struct ops { unsigned (*fn)(unsigned *); };\n"
        "static unsigned handler(unsigned *p) { p[0] = 7u; return p[1]; }\n"
        "struct ops table = { handler };\n"
        "unsigned w(unsigned *q) { q[0] = 1u; return 0u; }\n"
    )
    assert "handler" in r.lowered.init_refs
    assert _foot(r, "handler") == ({UNKNOWN}, {UNKNOWN})
    assert not r.commute("handler", "w")


def test_a_device_access_is_an_effect_on_state_the_unit_cannot_name():
    """Two reads of a FIFO register do not commute: a volatile (MMIO-domain) access reads and
    writes unknown memory, while two plain readers of one pointer's memory still commute."""
    r = _unit(
        "#include <stdint.h>\n"
        "uint32_t vp(volatile uint32_t *p) { return p[0]; }\n"
        "uint32_t vq(volatile uint32_t *p) { return p[1]; }\n"
        "uint32_t pl(uint32_t *p) { return p[0]; }\n"
        "uint32_t pm(uint32_t *p) { return p[1]; }\n"
    )
    assert _foot(r, "vp") == ({UNKNOWN}, {UNKNOWN})
    assert not r.commute("vp", "vq")
    assert r.commute("pl", "pm")


def test_readers_of_a_volatile_global_member_or_register_pointer_do_not_commute():
    """Two reads of a FIFO register do not commute, however the program reaches it: a volatile
    file-scope variable, a file-scope pointer to volatile, a struct member pointing at volatile
    storage. Each read is a device access. The parent lowered all three as plain RAM accesses, so it
    reported that the two readers of `fifo` commute. Two readers of a plain global still do."""
    r = _unit(
        "#include <stdint.h>\n"
        "volatile uint32_t fifo;\n"
        "uint32_t g1(void) { return fifo; }\n"
        "uint32_t g2(void) { return fifo; }\n"
        "volatile uint32_t *gvp;\n"
        "uint32_t p1(uint32_t i) { return gvp[i & 3u]; }\n"
        "uint32_t p2(uint32_t i) { return *gvp + i; }\n"
        "struct D { volatile uint32_t *regs; };\n"
        "uint32_t m1(struct D *d) { return d->regs[0]; }\n"
        "uint32_t m2(struct D *d) { return d->regs[1]; }\n"
        "uint32_t plain;\n"
        "uint32_t q1(void) { return plain; }\n"
        "uint32_t q2(void) { return plain; }\n"
    )
    for a, b in (("g1", "g2"), ("p1", "p2"), ("m1", "m2")):
        for fn in (a, b):
            reads, writes = _foot(r, fn)
            assert UNKNOWN in reads and UNKNOWN in writes, (fn, reads, writes)
        assert not r.commute(a, b)
    assert _foot(r, "q1") == ({"plain"}, set())
    assert r.commute("q1", "q2")


def test_the_base_resource_decides_a_device_access_not_the_claims_spelling():
    """The C twin once lowered `p[i]` through a `volatile T *` as an ordinary load, where the oracle
    marked it MMIO; both rails mark it now. The device rule still reads the access's BASE resource,
    so the same load spelled RAM-domain is a device access -- one predicate, one answer on both rails
    (the twin's half is held by the forms' parity) -- and a load through a plain pointer is not."""
    r = _unit("unsigned vget(volatile unsigned *p, unsigned i) { return p[i & 7u]; }\n")
    lf = r.lowered.functions["vget"]
    assert [c.domain for c in lf.claims if c.op == "c.load"] == [Domain.MMIO]
    plain = [
        dataclasses.replace(c, domain=Domain.RAM, volatile=False) if c.op == "c.load" else c
        for c in lf.claims
    ]
    spelled = dataclasses.replace(
        r.lowered, functions={"vget": dataclasses.replace(lf, claims=plain)}
    )
    fp = analyze(spelled).footprints["vget"]
    assert (set(fp.reads), set(fp.writes)) == ({UNKNOWN}, {UNKNOWN})
    q = _unit("unsigned get(unsigned *p, unsigned i) { return p[i & 7u]; }\n")
    assert _foot(q, "get") == ({UNKNOWN}, set())


def test_a_file_scope_pointer_is_read_through_not_touched_in_place():
    """`gp[i] = v` writes what the global pointer holds -- unknown memory, since another unit can
    set `gp` -- not the pointer variable. The C twin models a file-scope pointer's slot as a scalar,
    and read it in place until the declaration was marked (`bcir_resource.is_pointer`)."""
    r = _unit(
        "unsigned *gp;\n"
        "void w(unsigned i) { gp[i & 7u] = i; }\n"
        "unsigned rd(unsigned i) { return gp[i & 7u]; }\n"
    )
    assert _foot(r, "w") == ({"gp"}, {UNKNOWN})
    assert _foot(r, "rd") == ({UNKNOWN, "gp"}, set())
    assert not r.commute("w", "rd")


def test_a_scalar_variable_written_in_place_is_private():
    """`a++` on a parameter is a store to the parameter itself, not through it."""
    r = _unit(
        "unsigned post_inc(unsigned a) { unsigned x = a++; return x * 100u + a; }\n"
        "unsigned pure(unsigned x) { return x; }\n"
    )
    assert _foot(r, "post_inc") == (set(), set())
    assert r.commute("post_inc", "pure")


# --- escape and narrowing --------------------------------------------------------------------------

_ESCAPE = """
unsigned *gp;
static unsigned reader(unsigned *p, unsigned i) { return p[i & 3u]; }
static unsigned capture(unsigned *p) { gp = p; return 0u; }
unsigned f(unsigned x) {
  unsigned priv[4]; unsigned lent[4]; unsigned esc[4];
  priv[x & 3u] = x; lent[x & 3u] = x; esc[x & 3u] = x;
  return priv[(x + 1u) & 3u] + reader(lent, x) + capture(esc);
}
unsigned *leak(unsigned x) { unsigned local[4]; local[0] = x; return local; }
"""


def test_each_local_array_gets_the_verdict_of_what_happens_to_its_address():
    r = _unit(_ESCAPE)
    assert r.escape.objects["f"] == {"priv": "nonescaping", "lent": "lent", "esc": "escaping"}
    assert r.escape.objects["leak"] == {"local": "escaping"}  # returned: outlives its frame
    counts = r.escape.counts()
    assert counts["candidates"] == counts["nonescaping"] + counts["lent"] + counts["escaping"]
    assert counts["nonescaping"] >= 1 and counts["escaping"] >= 1


def test_an_indirect_call_is_narrowed_to_the_functions_its_pointer_can_hold():
    r = _unit(
        "static unsigned add1(unsigned x) { return x + 1u; }\n"
        "static unsigned dbl(unsigned x) { return x + x; }\n"
        "unsigned one(unsigned x) { unsigned (*f)(unsigned) = add1; return f(x); }\n"
        "unsigned two(unsigned x) { unsigned (*f)(unsigned) = add1; if (x & 1u) f = dbl;"
        " return f(x); }\n"
        "unsigned any(unsigned (*f)(unsigned), unsigned x) { return f(x); }\n"
    )
    sites = {s.function: s for s in r.escape.sites}
    assert sites["one"].targets == ("add1",) and sites["one"].resolved
    assert sites["two"].targets == ("add1", "dbl") and not sites["two"].resolved
    assert sites["any"].targets is None  # an exported function's parameter: any function
    assert r.escape.edges["two"] == frozenset({"add1", "dbl"})
    assert _foot(r, "one") == (set(), set())  # the narrowed callee's footprint, not `*`
    assert _foot(r, "any") == ({UNKNOWN}, {UNKNOWN})


def test_a_call_with_more_operands_than_a_c_twin_claim_holds_refuses_the_unit():
    """The C twin keeps six operands per claim and drops the rest, so neither rail may reason
    about the unit: every footprint is `*`, no local is proved private, no site is narrowed. Six
    operands is the boundary: that unit is analyzed."""
    seven = (
        "static unsigned s7(unsigned a, unsigned b, unsigned c, unsigned d, unsigned e,"
        " unsigned f, unsigned *p) { p[0] = a + b + c + d + e + f; return p[1]; }\n"
        "unsigned caller(unsigned x) { unsigned t[4]; t[1] = x;"
        " return s7(x, x, x, x, x, x, t); }\n"
    )
    six = (
        "static unsigned s6(unsigned a, unsigned b, unsigned c, unsigned d, unsigned e,"
        " unsigned *p) { p[0] = a + b + c + d + e; return p[1]; }\n"
        "unsigned caller(unsigned x) { unsigned t[4]; t[1] = x;"
        " return s6(x, x, x, x, x, t); }\n"
    )
    everything = Footprint(frozenset({UNKNOWN}), frozenset({UNKNOWN}))
    refused = _unit(seven)
    assert refused.escape.truncated
    assert all(fp == everything for fp in refused.escape.footprints.values())
    assert not any(refused.escape.objects.values())
    assert escape_report(refused.lowered, refused.escape) == (
        "truncated=1\nfn=s7 refused\nfn=caller refused\n"
    )
    kept = _unit(six)
    assert not kept.escape.truncated
    assert kept.escape.objects["caller"] == {"t": "lent"}


def test_the_reports_the_c_twin_prints_have_the_documented_shape():
    r = _unit(_STORES)
    eff = effects_report(r.lowered, r.escape).splitlines()
    n = len(r.lowered.functions)
    assert len(eff) == n + n * (n - 1) // 2
    assert eff[0] == "fn=arr_w reads=g_arr writes=g_arr"
    assert "commute arr_w arr_r = 0" in eff and "commute via_r pr = 1" not in eff  # unit order
    e = _unit(_ESCAPE)
    esc = escape_report(e.lowered, e.escape).splitlines()
    assert "fn=f nonescaping=priv lent=lent escaping=esc icalls=-" in esc


# --- the property the rails' parity rests on ---------------------------------------------------------


def _shuffled(lowered, rnd):
    """The unit with every function's claims in a random order."""
    fns = {}
    for name, lf in lowered.functions.items():
        claims = list(lf.claims)
        rnd.shuffle(claims)
        fns[name] = dataclasses.replace(lf, claims=claims)
    return dataclasses.replace(lowered, functions=fns)


def _answer(result):
    return (
        result.footprints,
        result.objects,
        result.candidates,
        result.edges,
        sorted((s.function, s.report()) for s in result.sites),
    )


def test_the_answer_does_not_depend_on_the_order_the_claims_are_visited_in():
    """The two rails order sibling expressions differently, so byte-identical reports need an
    order-independent fixpoint: every rule is monotone in what a pointer may hold (an indirect call
    binds every known target and goes external as soon as the pointer may hold something unknown,
    rather than deciding once from a partial set)."""
    rnd = random.Random(0)
    sources = [_STORES, _ESCAPE] + [ef.generate(seed)[0] for seed in range(12)]
    for src in sources:
        lowered = _unit(src).lowered
        want = _answer(analyze(lowered))
        for _ in range(3):
            assert _answer(analyze(_shuffled(lowered, rnd))) == want


# --- the generated units: verdicts and targets known by construction ------------------------------


def test_the_generated_verdicts_and_targets_are_the_constructed_ones():
    esc_bad, esc_n, icall_bad, icall_n = ef.truth_mismatches(range(8))
    assert esc_n > 0 and icall_n > 0  # the grader examined something
    assert (esc_bad, icall_bad) == (0, 0)


def test_the_generator_is_deterministic_and_spans_every_verdict():
    seen = set()
    for seed in range(24):
        a, b = ef.generate(seed), ef.generate(seed)
        assert a == b
        seen |= set(a[1].values())
    assert seen == {"nonescaping", "lent", "escaping"}


def test_the_forms_span_every_kind_and_place_and_are_c_a_compiler_accepts():
    """The forms are pinned, so they must stay what they claim: every declared kind and access
    form used, every storage place present, one function per form that the oracle lowers, and C a
    compiler accepts with pointer/integer confusion an error (the casts to a pointer are the
    forged-pointer rule's own cases, a warning)."""
    assert set(ef.FORMS) == {"global", "static", "local", "param"}
    assert {k for per in ef.FORMS.values() for k in per} == set(ef.FORM_KINDS)
    used = {a for per in ef.FORMS.values() for v in per.values() for a in v.split()}
    assert used == set(ef.FORM_ACCESSES)
    units = ef.form_units()
    for place, source in units:
        forms = {f"f_{k}_{a}" for k, v in ef.FORMS[place].items() for a in v.split()}
        assert forms <= set(_unit(source).lowered.functions)
    cc = ef.find_cc()
    if not cc:
        assert not os.environ.get("BCIR_REQUIRE_LLVM"), (
            "BCIR_REQUIRE_LLVM is set and no C compiler checks the forms"
        )
        return
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as d:
        for place, source in units:
            path = os.path.join(d, f"forms_{place}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(source)
            p = subprocess.run(
                [cc, "-std=gnu11", "-fsyntax-only", "-Werror=incompatible-pointer-types",
                 "-Werror=int-conversion", "-Wno-unused", path],
                capture_output=True, text=True, timeout=120,
            )  # fmt: skip
            assert p.returncode == 0, (place, p.stderr[-2000:])


def test_the_witness_tells_apart_two_orders_only_a_shared_static_separates():
    """`s_a` and `s_b` differ only through the static counter of the helper they both call: the
    footprints say they do not commute (both write `tick.n`), and the witness shows the two orders
    diverge -- the pair that holds the static-local rule to the witness, not only to parity."""
    r = _unit(ef.generate(0)[0])
    assert _foot(r, "s_a") == ({"tick.n"}, {"tick.n"})
    assert not r.commute("s_a", "s_b")
    cc = ef.find_cc()
    if not ef.witness_available(cc):
        assert not os.environ.get("BCIR_REQUIRE_LLVM"), (
            "BCIR_REQUIRE_LLVM is set and no C compiler builds the commute witness"
        )
        return
    assert ef.witness(r, [("s_a", "s_b")], cc) == {("s_a", "s_b"): "conflict"}


def test_the_witness_finds_no_commuting_pair_that_diverges_and_can_find_one():
    """The dynamic witness runs each pair in both orders; a pair it shows diverging must not be
    reported to commute. It must also be able to show a divergence (a gate that cannot fire is
    not a gate): `gsw`-style writers and readers of one global diverge somewhere in the set."""
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not ef.witness_available(cc):
        # No compiler builds the witness (the quick tier hides it): nothing to run -- except in a
        # job that installed clang for this suite, where the absence is a failure (L2).
        assert not os.environ.get("BCIR_REQUIRE_LLVM"), (
            "BCIR_REQUIRE_LLVM is set and no C compiler builds the commute witness"
        )
        return
    results = [_unit(ef.generate(seed)[0]) for seed in range(6)]
    bad, decided = ef.commute_unsound(results, cc)
    assert decided > 0 and bad == 0
    conflicts = 0
    for res in results:
        fns = list(res.lowered.functions)
        pairs = [(a, b) for i, a in enumerate(fns) for b in fns[i + 1 :]]
        conflicts += sum(v == "conflict" for v in ef.witness(res, pairs, cc).values())
    assert conflicts > 0
