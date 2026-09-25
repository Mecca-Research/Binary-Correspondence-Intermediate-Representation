"""GEM+ G9 (S5-A): the declared alias facts carried the rest of the way to LLVM.

A claim declares its RIDs, its hazard and its volatility, and every resource its element size.
`lower.alias_facts` derives the kernel's facts from that declaration once, and every emitter of the
elementwise claim carries them: the LLVM kernel (`noalias`, one alias scope per resource, the TBAA
tag of the element type, `volatile`, a barrier's fences) and the C emitters clang lowers (`restrict`,
`volatile`, the fences). R12 holds each emitted fact to the declaration on both backends, LLVM's
own alias analysis judges what the metadata says, and every self-check runs the kernel with the
aliasing the claim declares. Each law is held to a negative witness: the modules the subset must
refuse, a kernel forged one fact at a time, and module pairs differing in one declared fact.
"""

from __future__ import annotations

import functools
import os
import re
import shutil

from bcir.tests import alias_fixtures as af


@functools.cache
def _graded() -> tuple[dict, dict]:
    seen: dict = {}
    return af.measure(1, seen), seen


def _refusal(emit, case) -> str:
    try:
        emit(case)
    except af.Refused as exc:
        return str(exc)
    raise AssertionError(f"{emit.__name__} lowered {case} instead of refusing it")


def _r12(case, text: str, backend: str = "ll") -> list[str]:
    from bcir.verify import verify_c_lowering, verify_lowering

    m, r = af.planned(case)
    law = verify_lowering if backend == "ll" else verify_c_lowering
    return [
        d.message for d in law(m, r, text, case.elem, width_override=case.width) if d.law == "R12"
    ]


def test_every_emitter_carries_the_declared_facts_over_the_corpus():
    """alias.* at zero over the whole corpus (bcir/tests/alias_fixtures.py): every partition,
    operation, element type, width and lawful contract on the LLVM kernel and the five C emitters,
    the refusals, the one-fact differential, the self-checks and R12's forgeries."""
    rows, _seen = _graded()
    assert rows == {row: 0.0 for row in af.ROWS}, rows


def test_the_corpus_exercises_every_emitter_harness_refusal_and_forgery():
    """A construct absent from the corpus is untested: the grader saw every kernel it declares
    on every emitter, every self-check, every refusal on every rail that owns it, every flip kind
    on every rail whose text can carry it, and every forgery kind -- so a zero row is a verdict,
    not a loop that iterated zero times (L2)."""
    _rows, seen = _graded()
    valid = len(af.valid_cases())
    plans = len(af.PARTITIONS) * len(af.CONTRACTS)
    assert seen["ll.kernels"] == valid and seen["c.kernel.kernels"] == valid, seen
    assert seen["ll.accesses"] >= 5 * valid, seen  # a vector loop and its epilogue per kernel
    for rail in ("c.gather", "c.specialist", "c.header"):
        assert seen[f"{rail}.kernels"] == 2 * plans, (rail, seen)
    assert seen["c.qfixed.kernels"] == plans, seen
    for harness in ("ll.harness", "qfixed.selfcheck", "wasm.node"):
        assert seen[f"{harness}.harnesses"] == plans, (harness, seen)
    assert seen["c.selfcheck.harnesses"] == 2 * plans, seen
    refusals = len(af.refusal_cases())
    hazard_only = sum(1 for case in af.refusal_cases() if not case.sizes and not case.undeclared)
    assert seen["refused"] == refusals * (1 + len(af.C_RAILS) - 1) + hazard_only, seen
    for rail in ("ll", "c.kernel", "c.gather", "c.specialist", "c.header", "c.qfixed"):
        assert seen[f"{rail}.flips"] > 0, (rail, seen)
    assert seen["forgeries"] == set(af.LL_FORGERIES) | set(af.C_FORGERIES), seen


def test_the_in_place_kernel_carries_its_partition_on_every_channel():
    """`Claim(rd=(1, 2), wr=(1,))` is `A[i] = A[i] + B[i]`: A and C are one resource. Only B's
    pointer is exclusive, A's and C's accesses share one scope, B's name the other in `!noalias`,
    every access carries clang's `float` TBAA tag -- and no fact the declaration contradicts."""
    case = af.Case((1, 2, 1), "ADD", "f32")
    text = af.emit_ll(case)
    k = af.parse_ll(text)
    assert "define void @bcir_kernel(ptr %A, ptr noalias %B, ptr %C, i64 %n)" in text, text
    assert af.scope_mismatches(k, af.declared(case)) == 0
    scopes = {
        acc.position: af._list(k, acc.scope) for acc in k.accesses if acc.position is not None
    }
    assert scopes[0] == scopes[2] != scopes[1], scopes
    assert {af.tbaa_name(k, acc.tbaa) for acc in k.accesses} == {"float"}
    assert _r12(case, text) == []
    ints = af.parse_ll(af.emit_ll(af.Case((1, 2, 1), "ADD", "i32")))
    assert {af.tbaa_name(ints, acc.tbaa) for acc in ints.accesses} == {"int"}


def test_a_volatile_barriered_claim_is_volatile_and_fenced_on_every_emitter():
    """The qualifier used to be dropped by every emitter and the hazard never realized: each
    ordered kernel failed its own R12, and a volatile claim -- which R5 requires to be ordered --
    could not produce a kernel the law accepts at all."""
    case = af.Case((1, 2, 3), "MUL", "f32", None, "barriered", True)
    text = af.emit_ll(case)
    k = af.parse_ll(text)
    assert k.accesses and all(acc.volatile for acc in k.accesses)
    assert k.entry_fence == "seq_cst" and k.exit_fences == ("seq_cst",)
    assert _r12(case, text) == []
    for _label, emit, body, _per_elem, _sizes in af.C_RAILS:
        c_text, fn = emit(case)
        ck = af.parse_c(c_text, fn)
        assert ck.volatile == frozenset({0, 1, 2}), (fn, c_text)
        if body:
            assert ck.entry_fence and ck.exit_fence and ck.fences == 2, (fn, c_text)
    plain = af.parse_ll(af.emit_ll(af.Case((1, 2, 3), "MUL", "f32")))
    assert not any(acc.volatile for acc in plain.accesses) and not plain.fences


def test_the_subset_refuses_what_it_does_not_generate_on_every_emitter():
    """An atomic hazard needs atomic element operations and an unknown one names no ordering:
    every emitter lowered both to plain accesses. A kernel addressing 4-byte elements over a
    resource declaring 8-byte ones reads the wrong bytes. All are refused now, by name."""
    atomic = af.Case((1, 2, 3), "ADD", "f32", None, "atomic", True)
    wild = af.Case((1, 2, 3), "ADD", "f32", None, "wild")
    wide = af.Case((1, 2, 3), "ADD", "f32", None, "unique", False, ((0, 8),))
    emitters = [af.emit_ll] + [emit for _label, emit, *_rest in af.C_RAILS]
    for emit in emitters:
        assert "atomic element operations" in _refusal(emit, atomic)
        assert "'wild' names no ordering" in _refusal(emit, wild)
    ghost = af.Case((1, 2, 3), "ADD", "f32", None, "unique", False, (), (1,))
    for emit in [af.emit_ll] + [emit for label, emit, *_rest in af.C_RAILS if label != "c.qfixed"]:
        assert "declares 8-byte elements" in _refusal(emit, wide)
        assert "RID 2, which the module does not declare" in _refusal(emit, ghost)
    # R12 says the same thing about a kernel handed to it for such a claim.
    honest = af.emit_ll(af.Case((1, 2, 3), "ADD", "f32"))
    assert any("atomic element operations" in msg for msg in _r12(atomic, honest))
    # The subset gate itself (`find_elementwise`) refuses before any fact is derived: every
    # emitter would refuse anyway (`kernel_facts` shares the predicate), so its witness is the
    # entry point that never derives facts (docs/security/laws.md L22, S5-A).
    from bcir.lower.llvm import harness_trip_counts

    m, r = af.planned(atomic)
    try:
        harness_trip_counts(m, r)
    except NotImplementedError as exc:
        assert "atomic element operations" in str(exc)
    else:
        raise AssertionError("the subset gate admitted an atomic claim")


def test_r12_names_each_forged_fact():
    """Each forgery raises its own finding, a false fact apart from a dropped one."""
    shared = af.Case((1, 2, 1), "ADD", "f32")
    honest = af.emit_ll(shared)
    decl = af.declared(shared)
    expected = {
        "noalias.add": "false alias fact: %A carries noalias",
        "noalias.drop": "alias fact dropped: %B is the only pointer to RID 2",
        "volatile.add": "volatile not declared",
        "scope.drop": "carry no !alias.scope",
        "scope.own": "name their own resource's scope in !noalias",
        "tbaa.drop": "TBAA not preserved",
        "tbaa.type": "TBAA not preserved",
        "fence.add": "hazard contract 'unique' declares no ordering",
    }
    for kind, needle in expected.items():
        forged = af.forge_ll(kind, honest, decl)
        assert forged is not None and forged != honest, kind
        assert any(needle in msg for msg in _r12(shared, forged)), (kind, _r12(shared, forged))
    ordered = af.Case((1, 2, 3), "ADD", "f32", None, "barriered", True)
    honest = af.emit_ll(ordered)
    decl = af.declared(ordered)
    for kind, needle in {
        "volatile.drop": "accesses of a volatile claim are not volatile",
        "volatile.bare": "1 accesses of a volatile claim are not volatile",
        "fence.drop": "hazard contract 'barriered' requires a fence",
        "fence.narrow": "hazard contract 'barriered' requires a fence",
    }.items():
        assert any(needle in msg for msg in _r12(ordered, af.forge_ll(kind, honest, decl))), kind
    # The C backend.
    c_honest = af.emit_c(shared)
    for kind, needle in {
        "c.restrict.add": "false alias fact: A is restrict-qualified",
        "c.restrict.drop": "B is the only pointer to RID 2",
        "c.fence.add": "hazard contract 'unique' declares no ordering",
    }.items():
        forged = af.forge_c(kind, c_honest, af.declared(shared))
        assert any(needle in msg for msg in _r12(shared, forged, "c")), (
            kind,
            _r12(shared, forged, "c"),
        )
    c_ordered = af.emit_c(ordered)
    for kind, needle in {
        "c.volatile.drop": "the claim is volatile",
        "c.volatile.cast": "A is used other than as a subscript",
        "c.volatile.address": "the kernel body takes an address",
        "c.fence.drop": "requires `atomic_thread_fence(memory_order_seq_cst);`",
        "c.fence.narrow": "requires `atomic_thread_fence(memory_order_seq_cst);`",
        "c.fence.return": "a `return` leaves the kernel before its last",
    }.items():
        forged = af.forge_c(kind, c_ordered, af.declared(ordered))
        assert any(needle in msg for msg in _r12(ordered, forged, "c")), kind


def test_r12_reads_the_language_not_a_subset():
    """L4, both halves: R12 attributes every spelling LLVM's parser accepts. The honest kernel
    respelled -- the system sync scope named, each alignment left to the parser (a scalar float's
    is 4) -- is still honest, and the same respellings carrying a false fact are still findings."""
    case = af.Case((1, 2, 1), "ADD", "f32", 1, "barriered", True)
    honest = af.emit_ll(case)
    assert _r12(case, honest) == []
    same = honest.replace("fence seq_cst", 'fence syncscope("") seq_cst').replace(", align 4", "")
    assert same != honest and _r12(case, same) == [], _r12(case, same)
    narrowed = same.replace('syncscope("")', 'syncscope("singlethread")')
    assert any("requires a fence" in msg for msg in _r12(case, narrowed)), _r12(case, narrowed)
    plain = re.sub(r"load volatile float, ptr (%\w+)[^\n]*", r"load float, ptr \1", same, count=1)
    assert plain != same
    assert any("are not volatile" in msg for msg in _r12(case, plain)), _r12(case, plain)


def test_r12_reads_any_text_without_raising():
    """R12 is total (L1): a kernel whose metadata dangles, loops or is not metadata at all is a
    finding in the law, never a traceback."""
    case = af.Case((1, 2, 1), "ADD", "f32")
    honest = af.emit_ll(case)
    cuts = [honest[: len(honest) * i // 8] for i in range(8)]
    mangled = [
        honest.replace("!alias.scope !", "!alias.scope !9"),
        honest.replace("distinct !{", "!{"),
        honest + '!0 = !{!0, !0, !0}\n!1 = !{!"x", !1, i64 0}\n',
        honest.replace("!tbaa !", "!tbaa !1"),
        "!0 = distinct !{!0\n",
        "",
    ]
    for text in cuts + mangled:
        messages = _r12(case, text)
        assert messages, text[-200:]
    c_honest = af.emit_c(case)
    for text in (c_honest[: len(c_honest) // 2], "void f(", "", c_honest.replace("{", "")):
        assert _r12(case, text, "c")


def test_two_modules_differing_in_one_alias_fact_emit_different_facts():
    """The differential the slice is gated on, spelled out once: volatility, the hazard, the
    partition and the element type each reach the emitted facts, and an element size the kernel
    does not address is refused rather than emitted identically."""
    base = af.Case((1, 2, 3), "ADD", "f32", None, "barriered", False)
    facts = af.ll_facts(af.parse_ll(af.emit_ll(base)))
    for kind, other in af.flips(base):
        if kind == "size":
            assert _refusal(af.emit_ll, other)
            continue
        assert af.ll_facts(af.parse_ll(af.emit_ll(other))) != facts, kind


def test_every_self_check_binds_one_buffer_per_declared_resource():
    """The harnesses passed three private buffers whatever the RIDs said, so a kernel's alias
    facts were never executed against the aliasing they describe."""
    from bcir.lower.c_kernel import emit_qfixed_selfcheck_c, emit_selfcheck_c
    from bcir.lower.llvm import emit_harness_c
    from bcir.lower.wasm import harness_binding

    m, r = af.planned(af.Case((7, 7, 4), "SUB", "f32"))
    assert "bcir_kernel(R0, R0, R1, n);" in emit_harness_c(m, r)
    assert "bcir_kernel(R0, R0, R1, n);" in emit_selfcheck_c(m, r)
    assert "bcir_qfixed(R0, R0, R1, n);" in emit_qfixed_selfcheck_c(m, r)
    assert harness_binding(m, r) == (0, 0, 1)
    m, r = af.planned(af.Case((1, 2, 1), "ADD", "f32"))
    assert "bcir_kernel(R0, R1, R0, n);" in emit_harness_c(m, r)
    assert harness_binding(m, r) == (0, 1, 0)


def test_the_library_artifact_publishes_the_declared_contract():
    """`bcir.api` hands out the kernel and its ABI header together: the header's qualifiers are
    the kernel's (a translation unit including both must compile), and it said A, B and C were
    non-overlapping for every claim."""
    from bcir.api import build_artifact

    m, _r = af.planned(af.Case((1, 2, 1), "ADD", "f32", None, "barriered", True))
    art = build_artifact(m)
    assert art.attested, art
    assert "const volatile float *A, const volatile float *restrict B," in art.header_c
    assert "non-overlapping" not in art.header_c
    ck = af.parse_c(art.kernel_c, art.fn_name)
    assert ck.restrict == frozenset({1}) and ck.volatile == frozenset({0, 1, 2}), art.kernel_c


def test_llvm_judges_the_facts_and_every_runner_executes_them():
    """Where a coherent LLVM toolset is present: LLVM's scoped-noalias analysis alone proves every
    declared-disjoint access pair NoAlias, its default analysis proves no pair sharing a resource
    NoAlias, clang's IR for the C kernel carries the LLVM kernel's facts, and every runner the
    host has executes each kernel with the declared aliasing bound. Without the tools the rows
    are NOT-MEASURED -- never zero by default -- and in a job that installed LLVM for this suite
    (`BCIR_REQUIRE_LLVM=1`, both CI oracle jobs) their absence is a failure, not a skip (L2)."""
    seen: dict = {}
    rows = af.measure_llvm(seen)
    if rows is None:
        assert not os.environ.get("BCIR_REQUIRE_LLVM"), (
            "BCIR_REQUIRE_LLVM is set and no coherent clang/llvm-link/opt resolved: the rows "
            "LLVM judges were not measured"
        )
        return
    assert rows == {row: 0.0 for row in af.LLVM_ROWS}, rows
    assert seen["aa.default.pairs"] > 0 and seen["aa.scoped.pairs"] > 0, seen
    assert seen["clang.kernels"] == len(af.core_cases()), seen
    assert seen["exec.runners"], seen
    if shutil.which("node") is not None:
        # The node self-check computed `+` whatever the claim declared: a SUB or MUL kernel
        # could not pass it. The rotation runs every operation under it.
        assert "wasm.node" in seen["exec.runners"] and seen["wasm.node.runs"] == len(
            af.exec_cases()
        )
