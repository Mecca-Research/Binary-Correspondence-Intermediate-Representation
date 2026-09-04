"""Phase 4 — segment 3 (optimizer correctness story): alias/effect analysis for the C frontend.

Each compiled function gets a read/write footprint over shared (module-scope) state, folded through
its callees, so the inter-procedural scheduler can tell which functions are independent (commute).
The read set is sound even for a bare `return g;` / `if (g)` that names a global without a load claim.
"""

from __future__ import annotations

from bcir.frontends.cfront import compile_unit

_SRC = """
int g1;
int g2;
int reads_g1(void){ return g1; }
int also_reads_g1(void){ return g1 + 1; }
void writes_g1(int v){ g1 = v; }
void writes_g2(int v){ g2 = v; }
int pure(int x){ return x * 2; }
int wrapper(void){ return reads_g1(); }
"""


def _foot(res, fn):
    names = {rid: r.name for rid, r in res.lowered.resources.items()}
    eff = res.effects[fn]
    return (
        {names[r] for r in eff.reads if r in names},
        {names[w] for w in eff.writes if w in names},
    )


def _compiled():
    return compile_unit(_SRC, check_clang=False)


def test_read_and_write_footprints_are_captured():
    r = _compiled()
    assert _foot(r, "also_reads_g1") == ({"g1"}, set())
    assert _foot(r, "writes_g1") == (set(), {"g1"})
    assert _foot(r, "pure") == (set(), set())


def test_bare_return_of_a_global_is_a_sound_read():
    # `return g1;` names the global without emitting a read claim -- the footprint must still see it,
    # else a reader would wrongly "commute" with a writer.
    r = _compiled()
    assert _foot(r, "reads_g1") == ({"g1"}, set())


def test_callee_effects_fold_into_the_caller():
    r = _compiled()
    assert _foot(r, "wrapper") == ({"g1"}, set())  # inherits reads_g1's read of g1


def test_commutation_follows_the_footprints():
    r = _compiled()
    assert r.commute("reads_g1", "also_reads_g1")  # two readers of g1 commute
    assert r.commute("writes_g1", "writes_g2")  # disjoint globals commute
    assert r.commute("pure", "writes_g1")  # a pure function commutes with anything
    assert not r.commute("reads_g1", "writes_g1")  # reader vs writer of g1: RAW/WAR conflict
    assert not r.commute("wrapper", "writes_g1")  # the conflict is inherited transitively
    assert not r.commute("writes_g1", "writes_g1")  # a writer conflicts with itself (WAW)


def test_commute_is_symmetric():
    r = _compiled()
    for a in ("reads_g1", "writes_g1", "writes_g2", "pure", "wrapper"):
        for b in ("reads_g1", "writes_g1", "writes_g2", "pure", "wrapper"):
            assert r.commute(a, b) == r.commute(b, a)


def test_attestation_records_the_effect_footprint():
    r = _compiled()
    assert r.attestation["writes_g1"]["effect_writes"] == "g1"
    assert r.attestation["writes_g1"]["effect_reads"] == "-"
    assert r.attestation["reads_g1"]["effect_reads"] == "g1"
