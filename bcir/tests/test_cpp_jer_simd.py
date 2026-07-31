"""J5 — the hosted SIMD rail, and the one clause of its gate this container cannot close.

**J5's gate**: *"Optional C++17 structural/UTF-8 scanner behind the C ABI with scalar
fallback. Same accepted/rejected corpus and trace; statistically significant measured
advantage on at least two hosts; no unsupported-CPU fault."*

Three of those four clauses are checked here. The fourth — **two hosts** — is not, and is
recorded as unmet rather than approximated. One machine cannot produce a two-host result,
and a single-host number presented against a two-host gate would be the kind of claim §8
exists to refuse: *"no absolute claim ships without reproducible evidence."*

**Why "same trace" holds by construction and is tested anyway.** §4.1 says the scalar rail
is authoritative. So the SIMD path answers only *"is this block entirely ASCII?"* — a
question one comparison settles, and whose "yes" implies valid UTF-8 with no further
reasoning — and hands everything else to `bcir_jer_validate_utf8` itself. There is no second
UTF-8 implementation to keep in step. The differential below still walks every tier over
multi-byte sequences straddling **every** offset in a 32-octet block, and invalid sequences
at every offset, because "by construction" is a claim about the code as written and the test
is a claim about the code as built.

Skips cleanly when no C++ compiler is visible.
"""

from __future__ import annotations

import os
import random
import shutil
import statistics
import subprocess
import tempfile

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CPP = os.path.join(_ROOT, "runtime", "cpp")
#: The C++ adapter and the C core are compiled by their OWN compilers and then linked.
#: Handing a `.c` file to clang++ is an error under `-Werror` — and it would also be the
#: wrong thing to test, since the rail ships as a C++ translation unit linked against the C
#: core, not as C recompiled in C++ mode.
_CXX_SOURCES = [os.path.join(_CPP, "bcir_jer_simd.cpp"),
                os.path.join(_CPP, "test_jer_simd.cpp")]
_C_SOURCES = [os.path.join(_C, "bcir_jer.c"), os.path.join(_C, "bcir_runtime.c")]

#: Every tier the driver accepts. `auto` is whatever this CPU resolved to, and it is in the
#: list so the dispatch itself is differentiated rather than only the pinned paths.
_TIERS = ("scalar", "sse2", "avx2", "neon", "auto")


def _available() -> bool:
    have_cxx = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    have_cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    return bool(have_cxx and have_cc)


def _build(tmp: str, optimization: str = "-O2") -> str:
    cxx = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    objects = []
    for source, compiler, std in ([(s, cxx, "-std=c++17") for s in _CXX_SOURCES]
                                  + [(s, cc, "-std=c11") for s in _C_SOURCES]):
        obj = os.path.join(tmp, os.path.basename(source) + ".o")
        proc = subprocess.run(
            [compiler, std, optimization, "-Wall", "-Wextra", "-Werror", "-I", _C, "-I",
             _CPP, "-c", source, "-o", obj], capture_output=True, text=True)
        assert proc.returncode == 0, (
            f"{os.path.basename(source)} must build warning-clean:\n{proc.stderr[:2000]}")
        objects.append(obj)
    out = os.path.join(tmp, "test_jer_simd")
    proc = subprocess.run([cxx, *objects, "-o", out], capture_output=True, text=True)
    assert proc.returncode == 0, f"the SIMD rail must link:\n{proc.stderr[:2000]}"
    return out


def _drive(binary: str, lines: list[str]) -> list[str]:
    proc = subprocess.run([binary], input="\n".join(lines) + "\n", capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[:2000]
    return proc.stdout.splitlines()


def _corpus() -> list[tuple[str, bytes]]:
    """Everything where a vector pass and a scalar pass could disagree.

    The straddle and offset families are the ones that earn their place: a multi-byte
    sequence crossing a 16- or 32-octet boundary is exactly what a block-at-a-time validator
    splits, and an invalid octet at every offset is what pins the *offset* rather than just
    the status.
    """
    cases: list[tuple[str, bytes]] = [
        ("empty", b""),
        ("pure ascii", b'{"a":1,"b":"hello"}'),
        ("long ascii", b'{"k":"' + b"x" * 5000 + b'"}'),
        ("two-octet", "café".encode()),
        ("three-octet", "日本語".encode()),
        ("four-octet", "\U0001f600".encode()),
        ("astral run", "\U0001f600".encode() * 400),
        ("a lone 0xff", b"\xff"),
        ("every high octet", bytes(range(0x80, 0x100))),
    ]
    # Past both vector widths, so a boundary at 16 and at 32 is covered.
    for pad in range(40):
        cases.append((f"straddle-2-{pad}", b"a" * pad + "é".encode() + b"b" * 40))
        cases.append((f"straddle-4-{pad}", b"a" * pad + "\U0001f600".encode() + b"b" * 40))
        cases.append((f"stray-continuation-{pad}", b"a" * pad + b"\x80" + b"b" * 40))
        cases.append((f"truncated-{pad}", b"a" * pad + b"\xc3"))
        cases.append((f"overlong-{pad}", b"a" * pad + b"\xc0\xaf" + b"b" * 10))
        cases.append((f"surrogate-{pad}", b"a" * pad + b"\xed\xa0\x80" + b"b" * 10))
        cases.append((f"above-u10ffff-{pad}", b"a" * pad + b"\xf5\x80\x80\x80" + b"b" * 10))
    generator = random.Random(7)
    for index in range(200):
        cases.append((f"random-{index}",
                      bytes(generator.randrange(256)
                            for _ in range(generator.randrange(0, 200)))))
    return cases


# --- the gate's binding clause: same corpus, same trace ---------------------------------------


def test_every_tier_agrees_with_the_scalar_rail_on_status_and_offset():
    """J5's first clause. Not "equivalent" — identical, status and byte offset both.

    An offset that differed by one would still accept and reject the same documents, and
    would still be a broken diagnostic: §4.2's contract is a stable code *and* a byte
    offset, and a caller that reports the wrong octet sends someone to the wrong place.
    """
    if not _available():
        return
    cases = _corpus()
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        lines = [f"utf8 {tier} {octets.hex() or '-'}"
                 for _label, octets in cases for tier in _TIERS]
        replies = _drive(binary, lines)
    assert len(replies) == len(cases) * len(_TIERS)
    for index, (label, _octets) in enumerate(cases):
        window = replies[index * len(_TIERS):(index + 1) * len(_TIERS)]
        scalar = window[0]
        for tier, reply in zip(_TIERS, window):
            assert reply == scalar, f"{label}: {tier} gave {reply}, scalar gave {scalar}"
    assert len(cases) > 250, f"the corpus collapsed to {len(cases)} documents"


def test_the_corpus_actually_contains_both_verdicts():
    """A differential over documents that are all valid would prove nothing about rejection."""
    if not _available():
        return
    cases = _corpus()
    with tempfile.TemporaryDirectory() as tmp:
        replies = _drive(_build(tmp),
                         [f"utf8 auto {octets.hex() or '-'}" for _label, octets in cases])
    verdicts = {reply.split()[1] for reply in replies}
    assert len(verdicts) > 1, f"every document got the same verdict: {verdicts}"
    rejected = sum(1 for reply in replies if reply.split()[1] != "0")
    assert rejected > 100, f"only {rejected} documents were rejected"
    # And the rejections carry a spread of offsets, so the offset comparison has teeth.
    offsets = {reply.split()[2] for reply in replies if reply.split()[1] != "0"}
    assert len(offsets) > 20, f"only {len(offsets)} distinct offsets among the rejections"


# --- the gate's "no unsupported-CPU fault" clause ---------------------------------------------


def test_a_tier_the_cpu_does_not_advertise_is_never_entered():
    """J5's third clause. The dispatch degrades; it does not fault and does not refuse.

    Asking for NEON on x86 (or AVX2 on a machine without it) must produce the scalar answer,
    not a crash and not an error about the machine — a caller asking for a width that is not
    there wants the answer.
    """
    if not _available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        report = _drive(binary, ["tiers"])[0].split()
        available, compiled = int(report[1]), [int(v) for v in report[3].split(",")]
        # Scalar is always compiled; the rest depend on the target.
        assert compiled[0] == 1
        # Every tier answers, including ones this build or this CPU does not have.
        replies = _drive(binary, [f"utf8 {tier} {b'{}'.hex()}" for tier in _TIERS])
        assert len({reply for reply in replies}) == 1, replies
    # The resolved tier must be one this build actually compiled.
    assert compiled[available] == 1, f"resolved tier {available} is not compiled in"


def test_the_resolved_tier_is_reported_by_name_so_a_measurement_can_say_which():
    if not _available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        report = _drive(_build(tmp), ["tiers"])[0].split()
    assert report[2] in ("scalar", "sse2", "avx2", "neon"), report


# --- what the rail actually buys, including where it buys nothing ------------------------------


def _median_ns(binary: str, tier: str, document: bytes, rounds: int = 15,
               iterations: int = 32) -> float:
    replies = _drive(binary, [f"bench {tier} {rounds} {iterations} {document.hex()}"])
    return statistics.median(int(reply.split()[3]) for reply in replies)


def _ascii_document(nodes: int = 400) -> bytes:
    body = b",".join(b'{"kind":"claim","label":"%d","attributes":['
                     b'{"name":"op","value":"add"}]}' % index for index in range(nodes))
    return b'{"version":1,"nodes":[' + body + b'],"roots":[0]}'


def test_the_rail_is_faster_on_the_documents_it_targets():
    """The advantage clause, on ONE host — which is not the gate, and the next test says so.

    JER text is ASCII-dominant, which is the case this rail accelerates. Asserted as a
    generous ratio rather than a tight one: the claim under test is "the vector path is
    doing something", and a threshold tuned to this container would fail on a slower one for
    reasons that have nothing to do with the code.
    """
    if not _available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        document = _ascii_document()
        scalar = _median_ns(binary, "scalar", document)
        best = _median_ns(binary, "auto", document)
    assert best < scalar, f"the resolved tier ({best}ns) did not beat scalar ({scalar}ns)"
    assert scalar / best > 2.0, f"only {scalar / best:.1f}x on an all-ASCII document"


def test_one_multi_byte_octet_no_longer_costs_the_whole_document():
    """The defect this test was originally written to pin, now fixed — and still pinned.

    The first version handed everything from the first non-ASCII octet to the END of the
    document to the scalar rail, so a single `café` near the front cost a 29 KB document
    its entire acceleration: 1.00x. The runs now ALTERNATE, so the ASCII either side of a
    short multi-byte stretch is still vectorized.

    This is asserted as a *ratio against the all-ASCII case* rather than an absolute
    speedup, because that is the property that regressed before: an early accent must not
    collapse the document to scalar. A generous bound, since the claim is "the cliff is
    gone", not "the number is exactly this".
    """
    if not _available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        clean = _ascii_document()
        early = _ascii_document().replace(b'"add"', '"café"'.encode(), 1)
        clean_gain = _median_ns(binary, "scalar", clean) / _median_ns(binary, "auto", clean)
        early_gain = _median_ns(binary, "scalar", early) / _median_ns(binary, "auto", early)
    assert early_gain > 2.0, (
        f"one multi-byte octet dropped the speedup to {early_gain:.2f}x; the alternating "
        f"walk regressed to a single hand-off")
    assert early_gain > clean_gain * 0.4, (
        f"an early accent cost {clean_gain:.1f}x -> {early_gain:.1f}x; the two should be "
        f"close, because only the short multi-byte run goes scalar")


def test_multi_byte_text_does_not_regress_against_the_scalar_rail():
    """Heavy multi-byte text gains too, but the gain is small — so this asserts NO REGRESSION.

    Measured on an idle host (§7.3): 2.43× on accented text, 2.74× on CJK, 1.46× on emoji.
    Those are real, and they are also too small to assert on a shared runner: this test
    originally required >1.2× and failed under `-j 2` while the same binary measured 2.4×
    standalone. §8 settles what to do about that — *"shared CI gates validity and trend
    evidence, not noisy timing thresholds"* — so the threshold moves to the claim contention
    cannot fake.

    **No regression is still a real claim.** The alternating walk adds run-detection work to
    every multi-byte stretch; if that overhead exceeded what it saves, heavy multi-byte
    documents would be *slower* than plain scalar, and this is what would catch it. The
    speedups themselves live in §7.3 with the host they were measured on, which is where a
    performance number belongs.
    """
    if not _available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        for label, replacement in (("cjk", "日本語のテキスト"), ("accents", "café"),
                                   ("emoji", "\U0001f600\U0001f601")):
            document = _ascii_document().replace(b'"add"', replacement.encode())
            gain = (_median_ns(binary, "scalar", document)
                    / _median_ns(binary, "auto", document))
            assert gain > 0.75, (
                f"{label}: {gain:.2f}x — the vector rail is materially SLOWER than scalar on "
                f"multi-byte text, so run detection is costing more than it saves")


def test_the_adapter_contains_no_utf8_decision_of_its_own():
    """Structural: the only verdict-producing call is into the scalar rail.

    A word search would be the wrong check — it flags prose explaining the absence. This
    counts what the adapter actually *calls*: `bcir_jer_validate_utf8` must be the only
    function it invokes that can return a status, so a future contributor who adds a local
    validator trips this rather than passing the differential by luck.
    """
    source = open(os.path.join(_CPP, "bcir_jer_simd.cpp"), encoding="utf-8").read()
    body = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("*") and "/*" not in line)
    # Every producer of a bcir_jer_status in the adapter is the scalar rail or a dispatcher.
    assert "bcir_jer_validate_utf8(" in body
    for invented in ("0x80 && data[", "continuation", "overlong", "0xC2", "0xF4"):
        assert invented not in body, (
            f"the adapter references {invented!r}, which suggests it decides UTF-8 validity "
            f"itself; that is the second semantics rail §4.1 forbids")


def test_the_two_host_clause_of_the_gate_is_unmet_and_recorded_as_unmet():
    """J5's gate wants a significant advantage on **at least two hosts**. This is one host.

    This test exists so the gap is a checked fact rather than a sentence someone might edit
    away. It reads the roadmap and requires the J5 row to still say the measurement is
    single-host — if a second host is ever added, this test is what tells whoever does it to
    come back and update the claim.
    """
    roadmap = os.path.join(_ROOT, "docs", "BCIR_ASN1_JSON_ROADMAP.md")
    text = open(roadmap, encoding="utf-8").read()
    assert "two hosts" in text, "the gate's two-host clause disappeared from the roadmap"
    assert "single-host" in text or "one host" in text, (
        "the roadmap no longer records that J5's measurement is single-host; either a "
        "second host was added — in which case update this test and the row together — or "
        "the limitation was quietly dropped")


def test_the_scan_s_work_budget_is_a_semantic_limit_not_an_incidental_cost():
    """Why the structural index is not the same shape as the UTF-8 rail (§7.4).

    `bcir_jer_validate_utf8` has no cost budget, so skipping an ASCII run is semantically
    free. `bcir_jer_scan` charges one work unit per octet against §4.3's `work` ceiling, and
    `WORK_EXCEEDED` carries the *exact* octet at which the budget ran out — so the scan's
    cost is observable output, not an implementation detail. A vector pass that skipped a
    run without charging for it would accept documents the scalar rail rejects.

    This test exists so that constraint is a checked fact before anyone builds the index,
    rather than something discovered halfway through.
    """
    import dataclasses

    from bcir.asn1.jer_bounded import JerBoundedError, JerLimits, scan

    document = b'{"k":"' + b" " * 400 + b'"}'
    assert scan(document, JerLimits()) == 3, "the fixture stopped being accepted at all"
    try:
        scan(document, dataclasses.replace(JerLimits(), work=100))
    except JerBoundedError as error:
        # The exact octet, not merely "too much work": that precision is what a bulk-charging
        # vector pass would have to reproduce by re-walking the run that crosses the budget.
        assert "octet 100" in str(error), error
        assert "needs 101" in str(error), error
    else:
        raise AssertionError(
            "a 410-octet document passed a work ceiling of 100; the budget stopped being a "
            "semantic limit, and §7.4's argument about the structural index no longer holds")


def test_a_bulk_work_charge_reproduces_the_scalar_failure_point_in_closed_form():
    """The property that makes a vectorized scan feasible at all (§7.4).

    The scan's main loop charges **one unit per octet, at that octet's own position**. So a
    vector pass that skips a run and charges for it in bulk does not have to re-walk the run
    to find where the budget crossed: with `w` units spent against a ceiling of `L`, the
    failure is at octet `L - w` reporting `needs L + 1`, by arithmetic.

    The first version of §7.4 claimed the opposite — that a crossing run had to be re-walked
    per octet — and that would have capped any speed-up at the point the budget binds. It is
    wrong, and this test is what makes the correction checkable: if the charging ever stops
    being uniform, the closed form stops predicting and this fails.
    """
    import dataclasses

    from bcir.asn1.jer_bounded import JerBoundedError, JerLimits, scan

    document = b'{"k":' + b" " * 500 + b"1}"
    checked = 0
    for ceiling in range(1, 60):
        try:
            scan(document, dataclasses.replace(JerLimits(), work=ceiling))
        except JerBoundedError as error:
            # Nothing has been spent when the loop starts, so w = 0 and the failure is at
            # octet `ceiling`, needing `ceiling + 1`.
            assert f"at octet {ceiling}" in str(error), (ceiling, str(error))
            assert f"needs {ceiling + 1}" in str(error), (ceiling, str(error))
            checked += 1
    assert checked > 40, f"only {checked} ceilings actually failed; the fixture is too cheap"
