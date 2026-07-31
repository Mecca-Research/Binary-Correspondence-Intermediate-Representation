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
from bcir.asn1.encode_plan import compile_encode_plan
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
    """The fields are positional, so a mismatched reader mis-assigns instead of failing."""
    if not _available():
        return
    plan = compile_encode_plan(_seq(Component("a", _I)), module="Test", type_name="v")
    text = plan.serialize().replace(b"plan-version 2", b"plan-version 9")
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
