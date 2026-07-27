"""Dual-rail parity for the X.693 XER lexical primitives.

`runtime/c/bcir_xer.c` is the C twin of the lexical layer of `bcir/asn1/xer.py`: the tag
scanner and the `xmlcstring` escaper of X.680 clause 12.15. These tests build the driver in
`runtime/c/test_xer.c` and push the SAME campaign through both rails.

The campaign is the whole point. XER is text, so a superficial "does it parse" check passes
on almost anything; what is compared here is the exact tokenization — the tag's *kind*, the
*bounds of its name*, and the *offset just past it* — because a scanner that classifies a
tag right but bounds it wrong desynchronizes everything after it, and a scanner that
accepts a construct X.693 §8.1.2 excludes hands back a value the sender did not mean. The
refusals are compared by *reason*, not merely by the fact of a refusal, which is why
`bcir_xer_excluded` exists at all.

The totality property over arbitrary bytes is the fuzzer's job
(`runtime/c/fuzz_xer.c`), not this file's.

Skips cleanly when no C compiler is visible, exactly as the other C-twin tests do.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import tempfile

from bcir.asn1.codec import Asn1Error
from bcir.asn1.xer import XerRules, _Reader, escape_xmlcstring

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_SOURCES = ["bcir_xer.c", "test_xer.c"]
_SEED = 20260727

#: `bcir_xer_status`, mirrored so a failure names the status rather than a number.
_STATUS = {0: "OK", 1: "TRUNCATED", 2: "MALFORMED", 3: "EXCLUDED",
           4: "UNREPRESENTABLE", 5: "OVERFLOW", 6: "INVALID"}
#: `bcir_xer_excluded`.
_EXCLUDED = {1: "COMMENT", 2: "PI", 3: "CDATA", 4: "DOCTYPE", 5: "ATTRIBUTE",
             6: "NAMESPACE", 7: "NUMERIC"}
#: `bcir_xer_tag_kind`, in the spelling the Python `_Reader` uses.
_KIND = {0: "start", 1: "end", 2: "empty"}


def _build(tmp: str) -> str | None:
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        return None
    out = os.path.join(tmp, "test_xer")
    proc = None
    for std in ("c23", "c2x", "c11"):
        proc = subprocess.run(
            [cc, f"-std={std}", "-O1", "-Wall", "-Wextra", "-Werror", "-I", _C,
             *[os.path.join(_C, name) for name in _SOURCES], "-o", out],
            capture_output=True, text=True)
        if proc.returncode == 0:
            return out
    raise AssertionError(f"the XER twin must build warning-clean:\n{proc.stderr[:3000]}")


def _run(binary: str, lines: list[str]) -> list[str]:
    proc = subprocess.run([binary], input="\n".join(lines) + "\n",
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"driver exited {proc.returncode}: {proc.stderr[:2000]}"
    return proc.stdout.strip().splitlines()


def _hex(data: bytes) -> str:
    return data.hex() if data else "-"


# --- the tag scanner ---------------------------------------------------------------------

#: Documents chosen so that every branch of the scanner is reached: the three tag kinds,
#: white-space before `>` and before `/>`, each excluded construct, and the truncations that
#: sit one octet short of each of them.
_TAG_CASES = [
    "<a>", "</a>", "<a/>", "<PersonnelRecord>", "</ChildInformation>", "<_XMLThing/>",
    "<a >", "<a\t/>", "<a\n>", "<nul/>", "<BIT_STRING>", "<x-y.z/>",
    "<!-- c -->", "<![CDATA[x]]>", "<!DOCTYPE a>", "<?xml?>", '<a b="1">', "<a:b>",
    "<", "</", "<a", "<a/", "<!", "<!-", "<![CDATA", "<?",
    "<1a>", "<>", "< a>", "a", "", "<a b>", "<aé>", "<aé/>",
]


def _python_tag(text: str, pos: int) -> str:
    """The Python `_Reader`'s answer, in the driver's output vocabulary.

    Offsets are in OCTETS on both rails: the C twin has no notion of a code point boundary
    in a tag, and `_Reader` indexes a `str`, so the two only agree when the test feeds ASCII
    or converts. The campaign below therefore reports the octet offsets it computed from the
    encoded form, and any case whose name would need a non-ASCII character is a refusal on
    both rails anyway.
    """
    reader = _Reader(text, XerRules.BASIC)
    reader.pos = pos
    try:
        kind, name = reader._scan_tag()
    except Asn1Error as error:
        return f"ERR {error}"
    start = reader.text.index(name, pos)
    return f"OK {kind} {start} {len(name)} {reader.pos}"


def test_the_tag_scanner_tokenizes_identically_on_both_rails():
    """The exact tokenization, not merely "did it parse": kind, name bounds, and end.

    A scanner that gets the kind right and the end wrong desynchronizes every element after
    it, which is a decode that silently returns the wrong value rather than an error.
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        cases = [(text, 0) for text in _TAG_CASES]
        cases += [(text, pos) for text in ("<a><b/></a>", "  <a>x</a>")
                  for pos in range(len(text.encode()) + 2)]
        lines = [f"tag {_hex(text.encode())} {pos}" for text, pos in cases]
        answers = _run(binary, lines)
        assert len(answers) == len(cases)
        for (text, pos), got in zip(cases, answers):
            want = _python_tag(text, pos)
            if got.startswith("OK"):
                _kind_word = _KIND[int(got.split()[1])]
                rebuilt = "OK " + " ".join([_kind_word, *got.split()[2:]])
                assert want == rebuilt, (
                    f"{text!r} at {pos}: C said {rebuilt}, Python said {want}")
            else:
                assert want.startswith("ERR"), (
                    f"{text!r} at {pos}: C refused ({got}), Python accepted ({want})")


def test_the_excluded_constructs_are_refused_for_the_same_reason_on_both_rails():
    """§8.1.2's NOTE and the EXTENDED-XER surface — by reason, not just by refusal.

    Both rails have to agree on *which* clause excludes a construct, because that is what a
    peer is told. A twin that refused a CDATA section as "malformed" would be conformant and
    useless.
    """
    expected = {
        "<!-- c -->": ("COMMENT", "comments"),
        "<![CDATA[x]]>": ("CDATA", "CDATA"),
        "<!DOCTYPE a>": ("DOCTYPE", "document type declaration"),
        "<?xml?>": ("PI", "processing instruction"),
        '<a b="1">': ("ATTRIBUTE", "ATTRIBUTE instruction"),
        "<a:b>": ("NAMESPACE", "NAMESPACE instruction"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        texts = list(expected)
        answers = _run(binary, [f"tag {_hex(t.encode())} 0" for t in texts])
        for text, got in zip(texts, answers):
            reason, needle = expected[text]
            parts = got.split()
            assert parts[0] == "ERR" and parts[1] == "3", f"{text!r}: {got}"
            assert _EXCLUDED[int(parts[2])] == reason, f"{text!r}: {got}"
            reader = _Reader(text, XerRules.BASIC)
            try:
                reader._scan_tag()
            except Asn1Error as error:
                assert needle in str(error), f"{text!r}: {error}"
            else:
                raise AssertionError(f"the Python rail accepted {text!r}")


# --- the xmlcstring escaper --------------------------------------------------------------

_ESCAPE_CASES = [
    "", "a", "a<b>&c", "\x00\x1f", "\t\n\r", "John", "&&&", "<<<", ">>>",
    "".join(chr(code) for code in range(32)),
    "\x0b\x0c\x0e\x1a", "é中\U0001f600", "퟿", "", "�",
]


def test_the_escaper_agrees_character_for_character():
    """X.680 §12.15.4/§12.15.5 — one escape table, two implementations."""
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        rng = random.Random(_SEED)
        cases = list(_ESCAPE_CASES)
        for _ in range(120):
            cases.append("".join(
                chr(rng.choice([rng.randint(0, 0x7F), rng.randint(0xA0, 0x2FF),
                                rng.randint(0x4E00, 0x4EFF), rng.randint(0x10000, 0x100FF)]))
                for _ in range(rng.randint(0, 24))))
        lines = [f"escape {_hex(text.encode())}" for text in cases]
        answers = _run(binary, lines)
        assert len(answers) == len(cases)
        for text, got in zip(cases, answers):
            want = escape_xmlcstring(text)
            assert got.startswith("OK "), f"{text!r}: {got}"
            payload = got.split(maxsplit=1)[1]
            octets = b"" if payload == "-" else bytes.fromhex(payload)
            assert octets == want.encode(), (
                f"{text!r}: C produced {octets!r}, Python produced {want.encode()!r}")


def test_a_character_with_no_xmlcstring_spelling_is_refused_on_both_rails():
    """X.680 §12.15.1 with §41.10's NOTE — U+FFFE and U+FFFF have no escape at all."""
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        # Encoded directly as UTF-8, since Python refuses to build the string another way.
        cases = ["￾", "￿", "a￾"]
        answers = _run(binary, [f"escape {_hex(t.encode())}" for t in cases])
        for text, got in zip(cases, answers):
            assert got == "ERR 4", f"{text!r}: {got}"      # UNREPRESENTABLE
            try:
                escape_xmlcstring(text)
            except Asn1Error as error:
                assert "12.15.1" in str(error)
            else:
                raise AssertionError(f"the Python rail escaped {text!r}")


def test_escape_and_unescape_are_inverse_on_the_c_rail():
    """The round trip the Python rail gets from `_Reader._unescape` and Table 3."""
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        cases = [text for text in _ESCAPE_CASES if text]
        escaped = [escape_xmlcstring(text) for text in cases]
        answers = _run(binary, [f"unescape 0 {_hex(t.encode())}" for t in escaped])
        for text, got in zip(cases, answers):
            assert got.startswith("OK "), f"{text!r}: {got}"
            payload = got.split(maxsplit=1)[1]
            octets = b"" if payload == "-" else bytes.fromhex(payload)
            assert octets == text.encode(), f"{text!r}: got {octets!r}"


def test_the_numeric_escape_is_gated_by_the_rule_set_on_both_rails():
    """§9.1.3 deletes X.680 §12.15.8 from CXER, and the twin carries the same switch."""
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        text = "a&#233;b&#xEE;c"
        allowed, refused = _run(binary, [f"unescape 1 {_hex(text.encode())}",
                                         f"unescape 0 {_hex(text.encode())}"])
        assert allowed == "OK " + "aébîc".encode().hex(), allowed
        assert refused == "ERR 3", refused                 # EXCLUDED, per §9.1.3
        reader = _Reader(text, XerRules.BASIC)
        assert reader._unescape(text) == "aébîc"
        strict = _Reader(text, XerRules.CANONICAL)
        try:
            strict._unescape(text)
        except Asn1Error as error:
            assert "9.1.3" in str(error)
        else:
            raise AssertionError("the Python rail admitted a numeric escape under CXER")


def test_the_utf8_decoder_refuses_what_is_not_a_character():
    """Overlong forms, surrogates and anything above U+10FFFF.

    Two decoders that disagree about what a byte sequence means is the classic
    validator/consumer split, so the twin refuses all three rather than passing them on —
    and Python's own UTF-8 decoder is the oracle it is checked against.
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        cases = [
            b"\xc0\x80",          # overlong NUL
            b"\xe0\x80\x80",      # overlong, three octets
            b"\xf0\x80\x80\x80",  # overlong, four octets
            b"\xed\xa0\x80",      # U+D800, a surrogate
            b"\xf5\x80\x80\x80",  # above U+10FFFF
            b"\x80",              # a bare continuation
            b"\xc3",              # truncated
            b"\xc3\xa9",          # the one valid case, as a control
        ]
        answers = _run(binary, [f"utf8 {_hex(data)} 0" for data in cases])
        for data, got in zip(cases, answers):
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                assert got.startswith("ERR"), f"{data.hex()}: C accepted it ({got})"
            else:
                assert got.startswith("OK"), f"{data.hex()}: C refused it ({got})"


def test_white_space_is_the_four_characters_clause_8_1_4_names():
    """§8.1.4 — narrower than XML's own S production, and both rails must agree."""
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        cases = ["   x", "\t\n\r x", "\x0bx", "\x0cx", "x", "", " x"]
        answers = _run(binary, [f"space {_hex(t.encode())} 0" for t in cases])
        for text, got in zip(cases, answers):
            reader = _Reader(text, XerRules.BASIC)
            reader.skip_space()
            assert got == f"OK {reader.pos}", f"{text!r}: C said {got}, Python {reader.pos}"
