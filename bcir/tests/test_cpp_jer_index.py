"""The hosted structural index, and the seam that keeps it from being a second scanner.

§1's pipeline lists an *"optional hosted SIMD structural index"* beside the UTF-8 scanner,
and §7.4 records why it is a different problem. `bcir_jer_validate_utf8` has no cost budget,
so skipping an ASCII run is semantically free. **`bcir_jer_scan` charges one work unit per
octet** against §4.3's ceiling, and `BCIR_JER_WORK_EXCEEDED` carries the exact octet at which
the budget ran out — so the scan's cost is *observable output*, not an implementation detail.

**The design that follows from that.** `bcir_jer_scan`'s loop is a dispatch — skip
whitespace, recognise a structural octet, or hand off to a token scanner — and only the
dispatch is vectorizable. The token scanners are where the semantics live: §4.3's
`string_bytes` and `number_digits` limits, escape validity, the exponent ceiling. So
`bcir_jer.h` exports them through `bcir_jer_scan_cursor`, and `bcir_jer_index_scan` rebuilds
**only the dispatch**. It is a second dispatch loop, not a second scanner — the difference
between differential-testing one loop and differential-testing a parser, and the reason §4.1's
"no second semantics rail" survives the optimization.

**The seam was proven scalar first, and the scalar tier is still swept.** A differential that
only begins to exist alongside the optimization cannot tell you which of the two broke it, so
the rebuilt loop was shown to reproduce `bcir_jer_scan`'s status, offset, `needed` and node
count over the whole corpus while there was still only one variable. The vector pass then
arrived under that harness, and **every tier this build compiled is compared** — a tier that
degraded to scalar would otherwise pass by never running.

Skips cleanly when no C++ compiler is visible.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import tempfile

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CPP = os.path.join(_ROOT, "runtime", "cpp")
_CXX_SOURCES = [os.path.join(_CPP, "bcir_jer_index.cpp"),
                os.path.join(_CPP, "bcir_jer_simd.cpp"),
                os.path.join(_CPP, "test_jer_index.cpp")]
_C_SOURCES = [os.path.join(_C, "bcir_jer.c"), os.path.join(_C, "bcir_runtime.c")]

#: Ceilings swept per document. The point is not any single value — it is walking the budget's
#: failure point across every position, including *inside* a whitespace run, which is the one
#: place the index charges differently from the scalar rail.
_WORK_CAPS = (0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 500, 5000)


def _available() -> bool:
    have_cxx = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    have_cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    return bool(have_cxx and have_cc)


def _build(tmp: str) -> str:
    cxx = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    objects = []
    for source, compiler, std in ([(s, cxx, "-std=c++17") for s in _CXX_SOURCES]
                                  + [(s, cc, "-std=c11") for s in _C_SOURCES]):
        obj = os.path.join(tmp, os.path.basename(source) + ".o")
        proc = subprocess.run(
            [compiler, std, "-O2", "-Wall", "-Wextra", "-Werror", "-I", _C, "-I", _CPP,
             "-c", source, "-o", obj], capture_output=True, text=True)
        assert proc.returncode == 0, (
            f"{os.path.basename(source)} must build warning-clean:\n{proc.stderr[:2000]}")
        objects.append(obj)
    out = os.path.join(tmp, "test_jer_index")
    proc = subprocess.run([cxx, *objects, "-o", out], capture_output=True, text=True)
    assert proc.returncode == 0, f"the index rail must link:\n{proc.stderr[:2000]}"
    return out


def _drive(binary: str, lines: list[str]) -> list[str]:
    proc = subprocess.run([binary], input="\n".join(lines) + "\n", capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[:2000]
    return proc.stdout.splitlines()


def _corpus() -> list[tuple[str, bytes]]:
    """Documents where a rebuilt dispatch could diverge from the original.

    The whitespace families earn their place twice over: they are what the index charges in
    bulk, and a long leading run is where a budget failure lands *inside* the one call the
    scalar rail would have made octet by octet.
    """
    cases: list[tuple[str, bytes]] = [
        ("empty", b""),
        ("only whitespace", b"   \t\n\r  "),
        ("flat object", b'{"a":1}'),
        ("whitespace everywhere", b'  {  "a" : 1 , "b" : [ 1 , 2 ] }  '),
        ("empty containers", b"[]"),
        ("empty object", b"{}"),
        ("literals", b'{"a":true,"b":false,"c":null}'),
        ("a non-JSON literal", b'{"bad":NaN}'),
        ("a number with exponent", b'{"n":-1.5e10}'),
        ("a bmp escape", b'{"u":"\\u0041"}'),
        ("a surrogate pair", b'{"u":"\\ud83d\\ude00"}'),
        ("multi-byte text", b'{"k":"caf\xc3\xa9"}'),
        ("unclosed object", b"{"),
        ("stray close", b"}"),
        ("trailing comma", b"[1,"),
        ("truncated member", b'{"a"'),
        ("missing value", b'{"a":}'),
        ("mismatched containers", b'{"a":[1}'),
        ("deep nesting", b'\t\n\r {"x":[[[[[1]]]]]}  \n'),
        ("a long string", b'{"s":"' + b"x" * 3000 + b'"}'),
        # A 5000-octet leading run: the budget's failure point lands mid-run for most of the
        # swept ceilings, which is precisely the case the bulk charge has to get right.
        ("a very long whitespace run", b" " * 5000 + b'{"a":1}'),
        ("mixed whitespace run", (b" \t\n\r" * 900) + b"[1]"),
        ("many elements", b"[" + b"1," * 500 + b"1]"),
        ("many members", b"{" + b'"k":1,' * 400 + b'"z":1}'),
    ]
    generator = random.Random(11)
    for index in range(300):
        cases.append((f"random-{index}",
                      bytes(generator.randrange(256)
                            for _ in range(generator.randrange(0, 140)))))
    # Whitespace-heavy randoms, so the bulk path is hit by generated input too rather than
    # only by the documents someone thought to write down.
    for index in range(200):
        pad = bytes(generator.choice(b" \t\n\r") for _ in range(generator.randrange(0, 80)))
        body = b'{"a":' + str(generator.randrange(-9999, 9999)).encode() + b"}"
        cases.append((f"padded-{index}", pad + body + pad))
    return cases


def test_the_rebuilt_dispatch_answers_exactly_as_the_scalar_scan_does_at_every_tier():
    """The claim the whole design rests on: same status, same offset, same `needed`, same nodes.

    **Identical, not equivalent.** §4.2's contract is a stable code *and* a byte offset, so an
    offset that differed by one would refuse the same documents and still send a caller to the
    wrong octet — the quietest way for an accelerated rail to be wrong.

    The ceiling is swept per document because the index charges a whitespace run in **one**
    call where the scalar rail charges one octet at a time. §7.4's closed form is what makes
    those identical — the charge is uniform and positional, so with `w` spent against ceiling
    `L` the first octet to exceed is at `L - w` with `needed = L + 1` — and sweeping walks
    that failure point through every position, including inside the bulk call.

    Every compiled tier answers, and all of them are checked against the scalar rail. A vector
    block scan that ran off the end of a run, or stopped a block early and reported the wrong
    octet, shows up here and nowhere else.
    """
    if not _available():
        return
    cases = _corpus()
    lines = ["tiers"] + [f"both {cap} {octets.hex() or '-'}"
                         for _label, octets in cases for cap in _WORK_CAPS]
    with tempfile.TemporaryDirectory() as tmp:
        replies = _drive(_build(tmp), lines)
    header, replies = replies[0], replies[1:]
    assert header.startswith("tiers "), header
    # `tiers <available> <name> <c0,c1,c2,c3>`. Every tier this build compiled *and* this CPU
    # advertises must answer on every line; deriving the count from the build rather than
    # hard-coding it keeps the assertion honest on a host with no SIMD at all.
    available = int(header.split()[1])
    compiled = [int(flag) for flag in header.split()[3].split(",")]
    expected_tiers = sum(1 for tier, flag in enumerate(compiled) if flag and tier <= available)
    assert expected_tiers >= 1, header
    assert len(replies) == len(cases) * len(_WORK_CAPS), (
        f"{len(replies)} replies for {len(cases) * len(_WORK_CAPS)} comparisons")
    index = 0
    tiers_seen = 0
    for label, _octets in cases:
        for cap in _WORK_CAPS:
            reply = replies[index]
            index += 1
            scalar, *rebuilt = reply.split("|")
            assert len(rebuilt) == expected_tiers, (
                f"{label} at work<={cap}: {len(rebuilt)} tier(s) answered, but the build "
                f"reports {expected_tiers} runnable ({header}); a tier that skips the corpus "
                f"is a tier nothing shows correct")
            tiers_seen = max(tiers_seen, len(rebuilt))
            for group in rebuilt:
                assert scalar.split()[1:] == group.split(), (
                    f"{label} at work<={cap} [{header}]: scalar {scalar.strip()!r} against "
                    f"index {group.strip()!r}")
    assert index * tiers_seen > 5000, (
        f"the sweep collapsed to {index} documents x {tiers_seen} tier(s)")


def test_the_sweep_actually_exercises_a_budget_failure_inside_a_bulk_charge():
    """A differential over documents that never exhaust the budget would prove nothing.

    This is the counterpart to the test above: it confirms the corpus really does drive the
    interesting case — `WORK_EXCEEDED` reported at an offset *inside* a whitespace run, which
    is the only place the two rails compute the answer differently rather than identically.
    """
    if not _available():
        return
    document = b" " * 5000 + b'{"a":1}'
    caps = (1, 2, 17, 100, 999, 4999)
    with tempfile.TemporaryDirectory() as tmp:
        replies = _drive(_build(tmp),
                         [f"both {cap} {document.hex()}" for cap in caps])
    exceeded = 0
    for cap, reply in zip(caps, replies):
        scalar, *rebuilt = reply.split("|")
        fields = scalar.split()
        status, offset, needed = int(fields[1]), int(fields[2]), int(fields[3])
        for group in rebuilt:
            assert scalar.split()[1:] == group.split(), f"work<={cap}: {reply}"
        if status == 0:
            continue
        exceeded += 1
        # §7.4's closed form, checked against the rail rather than assumed: entering the run
        # with nothing spent, the first octet to exceed a ceiling of `cap` is at `cap`, and
        # the budget it needed is `cap + 1`.
        assert offset == cap, f"work<={cap}: failed at octet {offset}, expected {cap}"
        assert needed == cap + 1, f"work<={cap}: needed {needed}, expected {cap + 1}"
    assert exceeded >= 5, (
        f"only {exceeded} of {len(caps)} ceilings exhausted the budget; the sweep is not "
        f"reaching the case it exists for")


def test_the_index_reuses_the_token_scanners_rather_than_reimplementing_them():
    """§4.1's rule, checked structurally rather than trusted.

    The index may own the dispatch. It must **not** own what a string token is, what a
    number's digit limit is, or what an octet costs — those are §4.3's semantics and a second
    copy of them is the second rail §4.1 forbids and §8's table names as the risk.

    So the source is read: it must call the exported cursor for every token, and must contain
    none of the tells of a private UTF-8 or escape decision.
    """
    source = open(os.path.join(_CPP, "bcir_jer_index.cpp"), encoding="utf-8").read()
    body = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("*") and "/*" not in line)
    for required in ("bcir_jer_scan_spend", "bcir_jer_scan_string_token",
                     "bcir_jer_scan_number_token", "bcir_jer_scan_literal_token"):
        assert required in body, f"the index does not go through {required}"
    for invented in ("0xD800", "0xDC00", "\\\\u", "string_bytes", "number_bytes",
                     "integer_digits", "exponent_magnitude", "0x80"):
        assert invented not in body, (
            f"the index references {invented!r}, which suggests it decides a §4.3 limit or a "
            f"UTF-8 question itself; that is the second semantics rail §4.1 forbids")
    # Tier resolution belongs to the SIMD rail. A second CPU probe here would be a second
    # thing that can be wrong about the machine, and J5's "no unsupported-CPU fault" clause
    # would then hold on one rail and not the other.
    for probed in ("__builtin_cpu_supports", "__builtin_cpu_init", "cpuid", "getauxval"):
        assert probed not in body, (
            f"the index calls {probed!r} rather than deferring to bcir_jer_simd_tier_available")
    assert "bcir_jer_simd_tier_available" in body, "the index does not defer tier resolution"


def test_the_vector_pass_and_the_scalar_predicate_share_one_whitespace_set():
    """The vector's four constants and `is_space` must be the *same* four constants.

    This is the drift §8's table actually warns about — not a slow vector pass, but one that
    quietly means something else. ECMA-404 clause 4 admits SPACE, TAB, LF and CR and nothing
    else; a vector pass that additionally matched FORM FEED would accept documents the scalar
    rail refuses, and would do it only for runs long enough to reach a wide block.

    So the source is checked for a *named* set used by both, rather than each spelling the
    octets out where they could diverge one at a time.
    """
    source = open(os.path.join(_CPP, "bcir_jer_index.cpp"), encoding="utf-8").read()
    body = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("*") and "/*" not in line)
    names = ("kSpace", "kTab", "kLineFeed", "kReturn")
    for name in names:
        # Once to define it, once in `is_space`, and once per vector width present.
        assert body.count(name) >= 3, (
            f"{name} appears {body.count(name)} time(s); the vector pass and the scalar "
            f"predicate are meant to share it rather than each spell the octet out")
    # The literal octets may appear only where the four names are bound.
    for literal in ("0x20", "0x09", "0x0A", "0x0D"):
        holders = [line for line in body.splitlines() if literal in line]
        assert len(holders) <= 1, (
            f"{literal} is written on {len(holders)} lines; a second spelling of the "
            f"whitespace set is exactly how the tiers come to disagree:\n" + "\n".join(holders))
