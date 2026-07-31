"""E2 — the C twin of the plan-driven encoder, against E1's proven reference.

E1 established four Python emitters byte-identical to the oracle. This is the differential
that says the freestanding C encoder reaches the *same* octets from the *same* descriptor and
the *same* neutral value stream — which is what a native encode timing has to rest on, since
a number produced by a second implementation nobody checked measures nothing.

**The corpus is shared with E1 on purpose.** Re-deriving expectations here would let the two
rails agree on a wrong answer; every expectation below comes from `emit()`, which is itself
pinned against `encode_der`/`encode_jer`/`encode_oer`.

**A defect this differential found**, pinned in its own test: the first C `put_int_decimal`
accumulated into a `uint64_t`, so `2**64 + 7` emitted as `7` — a perfectly well-formed JER
document of a different value. Python has arbitrary-precision integers and could never have
shown this; only the twin could.

Skips cleanly when no C compiler is visible, as the other native tests do.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from bcir.asn1.codec import NULL, Oid
from bcir.asn1.emit import EmitRules, emit, flatten
from bcir.asn1.encode_plan import PLAN_VERSION, compile_encode_plan
from bcir.asn1.schema import Choice, Component, Primitive, Sequence, SequenceOf
from bcir.asn1.tags import Universal

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_SOURCES = ["bcir_emit.c", "test_emit.c"]

_I = Primitive(Universal.INTEGER)
_S = Primitive(Universal.UTF8_STRING)
_B = Primitive(Universal.BOOLEAN)
_N = Primitive(Universal.NULL)
_O = Primitive(Universal.OCTET_STRING)
_OID = Primitive(Universal.OBJECT_IDENTIFIER)


def _seq(*components, name="X") -> Sequence:
    return Sequence(tuple(components), name=name)


_CHOICE = Choice((Component("num", _I, tag=0), Component("txt", _S, tag=1)), name="C")

#: Everything the two rails must agree on. The entries that earn their place are the ones
#: where a plausible implementation diverges: an integer past 64 bits, a preamble past one
#: octet, an implicit tag on a constructed type, and a length past the short form.
_CORPUS = (
    ("a negative integer", _seq(Component("v", _I)), {"v": -1}),
    ("a 70-bit negative", _seq(Component("v", _I)), {"v": -(2 ** 70)}),
    ("an integer past 64 bits", _seq(Component("v", _I)), {"v": 2 ** 64 + 7}),
    ("a 400-bit integer", _seq(Component("v", _I)), {"v": 2 ** 400 + 12345}),
    ("zero", _seq(Component("v", _I)), {"v": 0}),
    ("the 128 length boundary", _seq(Component("v", _I)), {"v": 128}),
    ("both booleans", _seq(Component("a", _B), Component("b", _B)),
     {"a": True, "b": False}),
    ("a bare NULL", _seq(Component("v", _N)), {"v": NULL}),
    ("a NULL between integers",
     _seq(Component("a", _I), Component("v", _N), Component("b", _I)),
     {"a": 1, "v": NULL, "b": 2}),
    ("an octet string", _seq(Component("v", _O)), {"v": b"\x00\xff\x10"}),
    ("an empty octet string", _seq(Component("v", _O)), {"v": b""}),
    ("an empty string", _seq(Component("v", _S)), {"v": ""}),
    ("a non-ASCII string", _seq(Component("v", _S)), {"v": "café \U0001f600"}),
    ("a string of JSON escapes", _seq(Component("v", _S)), {"v": "a\nb\tc\"d\\e\x01"}),
    ("a string past the short length form", _seq(Component("v", _S)), {"v": "x" * 300}),
    ("an object identifier", _seq(Component("v", _OID)),
     {"v": Oid((1, 3, 6, 1, 4, 1, 62596, 1))}),
    ("an OID with a large arc", _seq(Component("v", _OID)), {"v": Oid((2, 999, 1234567))}),
    ("an optional present", _seq(Component("a", _I), Component("b", _I, optional=True)),
     {"a": 1, "b": 2}),
    ("an optional absent", _seq(Component("a", _I), Component("b", _I, optional=True)),
     {"a": 1}),
    ("a default absent", _seq(Component("a", _I), Component("b", _B, default=False)),
     {"a": 1}),
    ("a default present", _seq(Component("a", _I), Component("b", _B, default=False)),
     {"a": 1, "b": True}),
    ("twelve optionals, alternate present",
     _seq(*[Component(f"c{i}", _I, optional=True) for i in range(12)]),
     {f"c{i}": i for i in range(0, 12, 2)}),
    ("twelve optionals, none present",
     _seq(*[Component(f"c{i}", _I, optional=True) for i in range(12)]), {}),
    ("an empty SEQUENCE OF", _seq(Component("v", SequenceOf(_I, "SEQ"))), {"v": []}),
    ("a short SEQUENCE OF", _seq(Component("v", SequenceOf(_I, "SEQ"))), {"v": [1, 2, 3]}),
    ("a SEQUENCE OF past 255", _seq(Component("v", SequenceOf(_I, "SEQ"))),
     {"v": list(range(300))}),
    ("a SEQUENCE OF SEQUENCE",
     _seq(Component("v", SequenceOf(_seq(Component("a", _I), name="E"), "SEQ"))),
     {"v": [{"a": 1}, {"a": 2}]}),
    ("a nested SEQUENCE", _seq(Component("in", _seq(Component("a", _I), name="In"))),
     {"in": {"a": 5}}),
    ("three levels of nesting",
     _seq(Component("a", _seq(Component("b", _seq(Component("c", _I), name="C")),
                              name="B"))), {"a": {"b": {"c": 7}}}),
    ("implicit context tags", _seq(Component("a", _I, tag=0), Component("b", _S, tag=1)),
     {"a": 1, "b": "x"}),
    ("an explicit context tag", _seq(Component("a", _I, tag=0, explicit=True)), {"a": 1}),
    ("an explicit tag past the low-tag form",
     _seq(Component("a", _I, tag=100, explicit=True)), {"a": 1}),
    ("an implicit tag past the low-tag form", _seq(Component("a", _I, tag=100)), {"a": 1}),
    ("a CHOICE on its integer arm",
     _seq(Component("v", _CHOICE, tag=5, explicit=True)), {"v": ("num", 7)}),
    ("a CHOICE on its string arm",
     _seq(Component("v", _CHOICE, tag=5, explicit=True)), {"v": ("txt", "hi")}),
    ("an implicit tag on a SEQUENCE",
     _seq(Component("s", _seq(Component("a", _I), name="In"), tag=3)), {"s": {"a": 9}}),
    ("an implicit tag on a SEQUENCE OF",
     _seq(Component("v", SequenceOf(_I, "SEQ"), tag=4)), {"v": [1, 2]}),
    ("content past the three-octet length", _seq(Component("v", _S)), {"v": "y" * 70000}),
)

_RULES = ((EmitRules.DER, "der"), (EmitRules.BER, "ber"), (EmitRules.JER, "jer"),
          (EmitRules.COER, "coer"))


def _available() -> bool:
    return (shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")) is not None


def _build(tmp: str, optimization: str = "-O2") -> str | None:
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        return None
    out = os.path.join(tmp, "test_emit")
    proc = None
    for std in ("c2x", "c11"):
        proc = subprocess.run(
            [cc, f"-std={std}", optimization, "-Wall", "-Wextra", "-Werror", "-I", _C,
             *[os.path.join(_C, name) for name in _SOURCES], "-o", out],
            capture_output=True, text=True)
        if proc.returncode == 0:
            return out
    raise AssertionError(f"the emit driver must build warning-clean:\n{proc.stderr[:2000]}")


def _drive(binary: str, lines: list[str]) -> list[str]:
    proc = subprocess.run([binary], input="\n".join(lines) + "\n", capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[:2000]
    return proc.stdout.splitlines()


def _script() -> tuple[list[str], list[tuple[str, str, bytes]]]:
    lines: list[str] = []
    expect: list[tuple[str, str, bytes]] = []
    for label, kind, value in _CORPUS:
        plan = compile_encode_plan(kind, module="Test", type_name=label)
        stream = flatten(plan, value)
        lines.append("plan " + plan.serialize().hex())
        for rules, name in _RULES:
            # `-` spells an empty stream: a SEQUENCE whose only member is NULL flattens to
            # zero octets, and that is a case worth driving rather than skipping.
            lines.append(f"emit {name} {stream.hex() or '-'}")
            expect.append((label, name, emit(plan, stream, rules=rules)))
    return lines, expect


def _run(binary: str) -> None:
    lines, expect = _script()
    replies = [line for line in _drive(binary, lines) if line != "ok"]
    assert len(replies) == len(expect), f"{len(replies)} replies for {len(expect)} cases"
    for (label, name, want), reply in zip(expect, replies):
        assert reply.startswith("ok "), f"{label} / {name}: {reply}"
        assert bytes.fromhex(reply[3:]) == want, f"{label} / {name}"


# --- the differential ------------------------------------------------------------------------


def test_the_c_encoder_matches_the_python_reference_octet_for_octet():
    """152 cases across four candidates, all four expectations taken from E1's emitters."""
    if not _available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        _run(_build(tmp, "-O2"))


def test_the_answer_is_the_same_at_O0_and_O3():
    """The repository's standard C gate. This file does long division on octet arrays and
    compares widths against attacker-supplied lengths, so a difference here would be real
    undefined behaviour rather than a flake."""
    if not _available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        lines, expect = _script()
        low = _drive(_build(os.path.join(tmp), "-O0"), lines)
        with tempfile.TemporaryDirectory() as tmp3:
            high = _drive(_build(tmp3, "-O3"), lines)
        assert low == high, "the -O0 and -O3 builds disagree"
        assert len([line for line in low if line != "ok"]) == len(expect)


def test_an_integer_past_64_bits_is_not_silently_truncated():
    """The defect this differential found, pinned so it cannot come back quietly.

    The first C `put_int_decimal` accumulated into a `uint64_t`. `2**64 + 7` emitted as `7`
    — well-formed JER, wrong value, and no error anywhere. Python's arbitrary-precision
    integers meant the reference could never have shown it; only the twin could.
    """
    if not _available():
        return
    kind = _seq(Component("v", _I))
    plan = compile_encode_plan(kind, module="Test", type_name="wide")
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        lines, expect = [], []
        for value in (2 ** 64 + 7, 2 ** 64, 2 ** 128 - 1, -(2 ** 64) - 7, 2 ** 400 + 12345):
            stream = flatten(plan, value := {"v": value} if not isinstance(value, dict)
                             else value)
            lines.append("plan " + plan.serialize().hex())
            lines.append(f"emit jer {stream.hex()}")
            expect.append(emit(plan, stream, rules=EmitRules.JER))
        replies = [line for line in _drive(binary, lines) if line != "ok"]
        for want, reply in zip(expect, replies):
            assert bytes.fromhex(reply[3:]) == want, f"{want!r} vs {reply}"
        # And the digits really are past what a 64-bit accumulator holds.
        assert b"18446744073709551623" in expect[0]


# --- the refusals ------------------------------------------------------------------------------


def test_a_short_output_buffer_reports_the_capacity_it_needed():
    """A short buffer is a retryable answer, not a corrupt one."""
    if not _available():
        return
    kind = _seq(Component("a", _I), Component("b", _S))
    plan = compile_encode_plan(kind, module="Test", type_name="cap")
    stream = flatten(plan, {"a": 1, "b": "hello"})
    full = emit(plan, stream, rules=EmitRules.DER)
    with tempfile.TemporaryDirectory() as tmp:
        replies = [line for line in _drive(_build(tmp), [
            "plan " + plan.serialize().hex(),
            f"emitcap der 2 {stream.hex()}",
            f"emitcap der {len(full)} {stream.hex()}",
        ]) if line != "ok"]
    assert replies[0].startswith("err 6 "), replies[0]      # BCIR_EMIT_OUT_SHORT
    assert replies[0].split()[3] == str(len(full)), replies[0]
    assert bytes.fromhex(replies[1][3:]) == full            # exactly enough is enough


def test_a_truncated_or_overlong_stream_is_refused():
    if not _available():
        return
    kind = _seq(Component("a", _I), Component("b", _S))
    plan = compile_encode_plan(kind, module="Test", type_name="stream")
    stream = flatten(plan, {"a": 1, "b": "hello"})
    with tempfile.TemporaryDirectory() as tmp:
        replies = [line for line in _drive(_build(tmp), [
            "plan " + plan.serialize().hex(),
            f"emit der {stream[:-3].hex()}",
            f"emit der {(stream + b'zz').hex()}",
            f"emit jer {stream[:-3].hex()}",
            f"emit coer {(stream + b'zz').hex()}",
        ]) if line != "ok"]
    assert replies[0].startswith("err 4 "), replies[0]      # STREAM_SHORT
    assert replies[1].startswith("err 5 "), replies[1]      # STREAM_LONG
    assert replies[2].startswith("err 4 "), replies[2]
    assert replies[3].startswith("err 5 "), replies[3]


def test_a_plan_of_another_version_is_refused_rather_than_read_hopefully():
    """The fields are positional, so a mismatched reader mis-assigns instead of failing.

    The version is read from `PLAN_VERSION` rather than spelled out: a literal here silently
    stopped substituting when the format moved to 3, so the test kept passing while checking
    that a *well-formed current* plan parses — which every other test already covers.
    """
    if not _available():
        return
    plan = compile_encode_plan(_seq(Component("a", _I)), module="Test", type_name="v")
    current = f"plan-version {PLAN_VERSION}".encode("ascii")
    assert plan.serialize().startswith(current), plan.serialize()[:32]
    text = plan.serialize().replace(current, b"plan-version 9999")
    with tempfile.TemporaryDirectory() as tmp:
        replies = _drive(_build(tmp), ["plan " + text.hex()])
    assert replies[0].startswith("err 2 "), replies[0]      # PLAN_VERSION_BAD


def test_a_plan_larger_than_the_caller_s_tables_is_refused():
    if not _available():
        return
    plan = compile_encode_plan(_seq(Component("a", _I)), module="Test", type_name="junk")
    for mangled, status in ((plan.serialize().replace(b"kind=integer", b"kind=nonsense"), 1),
                            (plan.serialize()[:40], 1)):
        with tempfile.TemporaryDirectory() as tmp:
            replies = _drive(_build(tmp), ["plan " + mangled.hex()])
        assert replies[0].startswith(f"err {status} "), replies[0]


def test_a_short_scratch_reports_the_visit_count_it_needed():
    """Only DER reads the scratch, and it says how many slots the value actually wanted."""
    if not _available():
        return
    kind = _seq(Component("v", SequenceOf(_I, "SEQ")))
    plan = compile_encode_plan(kind, module="Test", type_name="scratch")
    stream = flatten(plan, {"v": list(range(50))})
    with tempfile.TemporaryDirectory() as tmp:
        replies = [line for line in _drive(_build(tmp), [
            "plan " + plan.serialize().hex(),
            "scratchcap 4",
            f"emit der {stream.hex()}",
            f"emit jer {stream.hex()}",
        ]) if line != "ok"]
    assert replies[0].startswith("err 7 "), replies[0]      # SCRATCH_SHORT
    assert int(replies[0].split()[3]) > 4, replies[0]
    # JER never touches the scratch, so a tiny one is irrelevant to it.
    assert replies[1].startswith("ok "), replies[1]
    assert bytes.fromhex(replies[1][3:]) == emit(plan, stream, rules=EmitRules.JER)


# --- what the fuzzer found ---------------------------------------------------------------


def test_a_sequence_of_cannot_declare_more_elements_than_the_stream_could_hold():
    """The first hang the fuzzer found, and the amplification it implies.

    The element count is a 32-bit field in the value stream. An element that consumes NO
    stream octets — a NULL, or a SEQUENCE of only NULLs — turns four attacker-chosen bytes
    into four billion iterations that produce output and read nothing: a few octets in,
    gigabytes out.

    The bound comes from the PLAN, which is trusted, not from the stream, which is not.
    """
    if not _available():
        return
    counting = compile_encode_plan(
        _seq(Component("v", SequenceOf(_I, "SEQ"))), module="Test", type_name="bound")
    free = compile_encode_plan(
        _seq(Component("v", SequenceOf(_N, "SEQ"))), module="Test", type_name="free")
    # A count of 2^32-1 with a one-octet minimum element, and a three-octet stream.
    hostile = (0xFFFFFFFF).to_bytes(4, "big")
    with tempfile.TemporaryDirectory() as tmp:
        replies = [line for line in _drive(_build(tmp), [
            "plan " + counting.serialize().hex(),
            f"emit der {hostile.hex()}",
            f"emit jer {hostile.hex()}",
            f"emit coer {hostile.hex()}",
            f"emit ber {hostile.hex()}",
            "plan " + free.serialize().hex(),
            f"emit jer {hostile.hex()}",
        ]) if line != "ok"]
    # An element that costs at least one octet: refused as a short stream, exactly.
    for reply in replies[:4]:
        assert reply.startswith("err 4 "), reply          # STREAM_SHORT
    # A zero-cost element has no stream bound at all, so it meets an explicit ceiling.
    assert replies[4].startswith("err 9 "), replies[4]    # UNSUPPORTED


def test_a_nested_sequence_does_not_cost_exponential_time_in_oer():
    """The second hang. OER's preamble precedes components the stream interleaves with
    values, and the first version walked each SEQUENCE's members twice to collect the
    presence bits — so every nesting level re-walked its whole subtree.

    The preamble's SIZE comes from the plan and no value can change it, so the space is
    reserved and the bits are patched in during the single emit pass. This test is a timing
    assertion in disguise: at eight levels the old code did 2^8 subtree walks, and the
    corpus below would not have returned.
    """
    if not _available():
        return
    kind = _seq(Component("a", _I, optional=True))
    for _ in range(8):
        kind = Sequence((Component("inner", kind, optional=True),
                         Component("x", _I, optional=True)), name="N")
    plan = compile_encode_plan(kind, module="Test", type_name="deep")
    value: dict = {"a": 1}
    for _ in range(8):
        value = {"inner": value, "x": 2}
    stream = flatten(plan, value)
    with tempfile.TemporaryDirectory() as tmp:
        replies = [line for line in _drive(_build(tmp), [
            "plan " + plan.serialize().hex(),
            f"emit coer {stream.hex()}",
        ]) if line != "ok"]
    assert bytes.fromhex(replies[0][3:]) == emit(plan, stream, rules=EmitRules.COER)


# --- plan version 3: the constraints, across both rails ---------------------------------------

#: Constrained cases the shared `_CORPUS` cannot hold, because every entry there is
#: unconstrained. That is not an oversight in the corpus — it is the exact blindness that let
#: an OER emitter ignoring constraints pass every parity test it had.
_CONSTRAINED = (
    ("§10.3 a) one unsigned octet", "ValueRange(0, 255)", 42),
    ("§10.3 b) two unsigned octets", "ValueRange(0, 65535)", 42),
    ("§10.3 c) four unsigned octets", "ValueRange(0, 2 ** 32 - 1)", 70000),
    ("§10.3 d) eight unsigned octets", "ValueRange(0, 2 ** 64 - 1)", 2 ** 63),
    ("§10.4 a) one signed octet", "ValueRange(-128, 127)", -5),
    ("§10.4 b) two signed octets", "ValueRange(-32768, 32767)", -5),
    ("§10.4 d) eight signed octets", "ValueRange(-(2 ** 63), 2 ** 63 - 1)", -(2 ** 62)),
    ("§10.3 e) a lower bound alone", "ValueRange(0, None)", 300),
    ("§10.4 e) an upper bound alone", "ValueRange(None, 255)", -300),
    ("an extensible bound is invisible to OER", "Extensible(ValueRange(0, 255))", 42),
)


#: An enumeration whose identifiers exercise the descriptor's `name:number|...` field: a
#: hyphen (X.680 12.4 allows it), a negative number, and a number past the short form.
_ENUM = (("five", 5), ("two-hundred", 200), ("minus-one", -1))


def _constrained_cases():
    """Build the constrained corpus. Imported lazily so the module stays cheap to import."""
    from bcir.asn1.constraints import Extensible, PermittedAlphabet, Size, ValueRange

    scope = {"Extensible": Extensible, "PermittedAlphabet": PermittedAlphabet,
             "Size": Size, "ValueRange": ValueRange}
    cases = [(label, _seq(Component("v", Primitive(Universal.INTEGER, "I",
                                                   constraint=eval(spelling, scope)))),
              {"v": value})
             for label, spelling, value in _CONSTRAINED]
    cases += [
        ("§14.1 a fixed-size OCTET STRING",
         _seq(Component("v", Primitive(Universal.OCTET_STRING, "O",
                                       constraint=Size(ValueRange(3, 3))))), {"v": b"abc"}),
        ("§14.2 a size RANGE keeps its determinant",
         _seq(Component("v", Primitive(Universal.OCTET_STRING, "O",
                                       constraint=Size(ValueRange(1, 3))))), {"v": b"abc"}),
        ("§27.2 a fixed-size known-multiplier string",
         _seq(Component("v", Primitive(Universal.IA5_STRING, "A",
                                       constraint=Size(ValueRange(3, 3))))), {"v": "abc"}),
        ("§27.1 UTF8String is never known-multiplier",
         _seq(Component("v", Primitive(Universal.UTF8_STRING, "U",
                                       constraint=Size(ValueRange(3, 3))))), {"v": "abc"}),
        ("a permitted alphabet, which OER does not read",
         _seq(Component("v", Primitive(Universal.IA5_STRING, "A",
                                       constraint=PermittedAlphabet(
                                           ValueRange("0", "9"))))), {"v": "019"}),
        # Version 4: an ENUMERATED needs its enumeration, because X.697 22.2 spells the
        # value as the IDENTIFIER and X.691 14.1 indexes the root. The three rows below are
        # X.696 11.3's short form, 11.4's long form and 11.4's SIGNED body.
        ("an ENUMERATED, short form", _seq(Component("v", Primitive(
            Universal.ENUMERATED, "E", enumeration=_ENUM))), {"v": 5}),
        ("an ENUMERATED, long form", _seq(Component("v", Primitive(
            Universal.ENUMERATED, "E", enumeration=_ENUM))), {"v": 200}),
        ("a negative ENUMERATED", _seq(Component("v", Primitive(
            Universal.ENUMERATED, "E", enumeration=_ENUM))), {"v": -1}),
        ("an extensible ENUMERATED", _seq(Component("v", Primitive(
            Universal.ENUMERATED, "E", enumeration=_ENUM, enum_extensible=True))),
         {"v": 5}),
        ("an extensible SEQUENCE", Sequence(
            (Component("a", _I),), name="X", extensible=True), {"a": 1}),
        ("an extensible CHOICE", _seq(Component("v", Choice(
            (Component("num", _I, tag=0),), name="C", extensible=True), tag=5,
            explicit=True)), {"v": ("num", 7)}),
        ("a constrained integer inside a SEQUENCE OF",
         _seq(Component("v", SequenceOf(Primitive(Universal.INTEGER, "I",
                                                  constraint=ValueRange(0, 255)), "S"))),
         {"v": [1, 2, 250]}),
        ("an optional constrained integer, absent",
         _seq(Component("a", _I), Component("b", Primitive(
             Universal.INTEGER, "I", constraint=ValueRange(0, 255)), optional=True)),
         {"a": 1}),
    ]
    return cases


def test_the_twin_reaches_the_same_octets_for_a_constrained_type():
    """The differential over the cases version 2 could not express at all.

    Every rule is driven, not just OER: recording a constraint must move no byte for DER,
    BER or JER, and a twin that quietly started reading bounds in the X.690 path would be
    just as wrong as one that ignored them in the OER path.
    """
    if not _available():
        return
    lines: list[str] = []
    expect: list[tuple[str, str, bytes]] = []
    for label, kind, value in _constrained_cases():
        plan = compile_encode_plan(kind, module="Test", type_name=label)
        stream = flatten(plan, value)
        lines.append("plan " + plan.serialize().hex())
        for rules, name in _RULES:
            lines.append(f"emit {name} {stream.hex() or '-'}")
            expect.append((label, name, emit(plan, stream, rules=rules)))
    with tempfile.TemporaryDirectory() as tmp:
        replies = [line for line in _drive(_build(tmp), lines) if line != "ok"]
    assert len(replies) == len(expect), f"{len(replies)} replies for {len(expect)} cases"
    for (label, name, want), reply in zip(expect, replies):
        assert reply.startswith("ok "), f"{label} / {name}: {reply}"
        assert bytes.fromhex(reply[3:]) == want, (
            f"{label} / {name}: {reply[3:]} != {want.hex()}")


def test_the_c_reader_stores_every_constraint_field_the_compiler_wrote():
    """The fields no emitter reads YET are still checked, because a field nothing reads rots.

    Version 3 records the X.691 extension-root bounds and the permitted alphabet, which only
    PER will consume. Landing a parser for them and testing only the OER-visible half would
    leave the PER writer to discover its own reader's bugs later, on top of its own.

    So the driver reads the parsed table back and this compares it against the Python
    compiler that wrote it — field for field, including the sign of a negative bound and the
    exact octets of an alphabet.
    """
    if not _available():
        return
    from bcir.asn1.constraints import (
        Extensible, Intersection, PermittedAlphabet, Size, ValueRange,
    )

    cases = [
        ("plain bounds", ValueRange(0, 255)),
        ("a negative lower bound", ValueRange(-(2 ** 63), 2 ** 63 - 1)),
        ("the widest unsigned bound", ValueRange(0, 2 ** 64 - 1)),
        ("MIN..255", ValueRange(None, 255)),
        ("a size", Size(ValueRange(1, 4))),
        ("an alphabet", PermittedAlphabet(ValueRange("0", "9"))),
        # The case that proves the two bound pairs are separate facts rather than one: OER
        # reads 0..1000 here (§8.2.2 g) drops the extensible part) while PER's extension
        # root is 0..255. A reader that stored one and derived the other gets this wrong.
        ("an extensible bound met by a plain one",
         Intersection((Extensible(ValueRange(0, 255)), ValueRange(0, 1000)))),
    ]
    lines, expect = [], []
    for label, constraint in cases:
        kind = _seq(Component("v", Primitive(Universal.IA5_STRING, "A",
                                             constraint=constraint)))
        plan = compile_encode_plan(kind, module="Test", type_name=label)
        lines.append("plan " + plan.serialize().hex())
        lines.append("constraint 1")           # node 0 is the SEQUENCE, node 1 the member
        expect.append((label, plan.root.members[0].node.constraint))

    with tempfile.TemporaryDirectory() as tmp:
        replies = [line for line in _drive(_build(tmp), lines) if line != "ok"]
    assert len(replies) == len(expect), f"{len(replies)} replies for {len(expect)} cases"
    for (label, want), reply in zip(expect, replies):
        assert reply.startswith("ok "), f"{label}: {reply}"
        fields = reply[3:].split()
        assert want is not None, label
        bounds = [None if f == "-" else int(f) for f in fields[:8]]
        assert bounds == [want.value_low, want.value_high, want.size_low, want.size_high,
                          want.root_value_low, want.root_value_high,
                          want.root_size_low, want.root_size_high], label
        assert fields[8] == ("1" if want.value_extensible else "0"), label
        assert fields[9] == ("1" if want.size_extensible else "0"), label
        alphabet = "" if fields[10] == "-" else bytes.fromhex(fields[10]).decode("utf-8")
        assert alphabet == want.alphabet, label


def test_the_twin_agrees_with_the_oracle_and_not_merely_with_the_python_rail():
    """The differential's blind spot, closed.

    Every case above compares the C twin against E1's Python emitter, and E1's parity with
    the oracle is asserted separately over `_CORPUS` — which contains no constrained type and
    no ENUMERATED. So a construct present here and absent there could have both rails
    agreeing on a wrong answer, and one did: the JER emitter wrote an enumerated as a number
    on both rails, and the C differential passed.

    This closes the loop by taking the expectation from the ORACLE for every case the
    constrained corpus adds.
    """
    if not _available():
        return
    from bcir.asn1.codec import encode_tlv
    from bcir.asn1.jer import encode_jer
    from bcir.asn1.oer import encode_oer

    for label, kind, value in _constrained_cases():
        plan = compile_encode_plan(kind, module="Test", type_name=label)
        stream = flatten(plan, value)
        assert emit(plan, stream, rules=EmitRules.DER) == encode_tlv(kind.encode(value)), label
        assert emit(plan, stream, rules=EmitRules.JER) == encode_jer(kind, value), label
        assert emit(plan, stream, rules=EmitRules.COER) == encode_oer(kind, value), label


def test_the_c_reader_stores_the_enumeration_and_the_extension_marker():
    """Version 4's two fields, read back off the parsed table.

    The extension marker is PER's and nothing emits it yet, so the same argument applies as
    for the constraint's root bounds: a field nothing reads is a field nothing checks. The
    enumeration IS read — by the JER emitter — but its numbers are PER's, and a reader that
    stored the names and dropped the numbers would pass every JER test.
    """
    if not _available():
        return
    cases = [
        ("a plain enumeration", Primitive(Universal.ENUMERATED, "E", enumeration=_ENUM),
         False),
        ("an extensible enumeration",
         Primitive(Universal.ENUMERATED, "E", enumeration=_ENUM, enum_extensible=True),
         True),
    ]
    lines, expect = [], []
    for label, kind, extensible in cases:
        plan = compile_encode_plan(_seq(Component("v", kind)), module="Test",
                                   type_name=label)
        lines.append("plan " + plan.serialize().hex())
        lines.append("enum 1")
        expect.append((label, kind.enumeration, extensible))
    with tempfile.TemporaryDirectory() as tmp:
        replies = [line for line in _drive(_build(tmp), lines) if line != "ok"]
    assert len(replies) == len(expect), f"{len(replies)} replies for {len(expect)} cases"
    for (label, enumeration, extensible), reply in zip(expect, replies):
        assert reply.startswith("ok "), f"{label}: {reply}"
        fields = reply[3:].split()
        assert fields[0] == ("1" if extensible else "0"), label
        got = tuple((item.split(":")[0], int(item.split(":")[1])) for item in fields[1:])
        assert got == enumeration, f"{label}: {got} != {enumeration}"


def test_an_unconstrained_node_records_no_constraint_at_all():
    """The format's cost is proportional to what it says.

    A node with nothing an encoder would read emits no constraint line, so a plan for an
    unconstrained schema is what version 2 wrote apart from its version. Checked on both
    rails: the Python compiler must leave the line out, and the C reader must report the
    node as unconstrained rather than as one with all-absent bounds.
    """
    if not _available():
        return
    kind = _seq(Component("v", _I))
    plan = compile_encode_plan(kind, module="Test", type_name="free")
    text = plan.serialize().decode("utf-8")
    assert "constraint" not in text, text
    assert text.startswith(f"plan-version {PLAN_VERSION}\n")
    with tempfile.TemporaryDirectory() as tmp:
        replies = _drive(_build(tmp), ["plan " + plan.serialize().hex(), "constraint 1"])
    assert replies[1] == "ok none", replies


def test_a_value_outside_its_constraint_is_refused_rather_than_truncated():
    """A fixed-width OER field cannot hold every integer, and the twin must say so.

    X.696 §10.3 selects the width from the *constraint*, not from the value, so a value the
    constraint does not admit has nowhere to go. Writing its low octets would produce a
    perfectly well-formed document of a different value — the failure this whole plan
    version exists to stop — so the twin refuses, and the Python emitter raises.
    """
    if not _available():
        return
    from bcir.asn1.constraints import ValueRange

    kind = _seq(Component("v", Primitive(Universal.INTEGER, "I",
                                         constraint=ValueRange(0, 255))))
    plan = compile_encode_plan(kind, module="Test", type_name="narrow")
    # The stream is built against an UNCONSTRAINED plan of the same shape, so the value is
    # well-formed as a stream and only the constraint makes it impossible.
    free = compile_encode_plan(_seq(Component("v", _I)), module="Test", type_name="free")
    for value in (300, -1):
        stream = flatten(free, {"v": value})
        try:
            emit(plan, stream, rules=EmitRules.COER)
        except Exception as error:               # noqa: BLE001 - the type is the oracle's
            assert "10.3" in str(error) or "10.4" in str(error), error
        else:
            raise AssertionError(f"{value} encoded into a one-octet unsigned field")
        with tempfile.TemporaryDirectory() as tmp:
            replies = [line for line in _drive(_build(tmp), [
                "plan " + plan.serialize().hex(),
                f"emit coer {stream.hex()}",
            ]) if line != "ok"]
        assert replies[0].startswith("err 9 "), f"{value}: {replies[0]}"   # UNSUPPORTED
