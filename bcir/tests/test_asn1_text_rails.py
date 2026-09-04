"""The two TEXT transfer syntaxes read digits with Python's idea of a digit, not X.680's.

Found by sweeping the root cause named in the 2026-08-12 audit — "a wire format parsed with
a host-language parser" — into every place it could still be hiding. It was hiding in five,
all of them in XER and JER, and the reason those two and no others is worth stating: they
carry **UTF-8 by design**. The octet-based rails decode their contents as ASCII first, so a
non-ASCII digit never reaches `int()` there; `decode_utctime` refuses `١٢٣٤٥٦٧٨٩٠١٢Z` at
the ASCII decode and never consults its `\\d` regex. In a text rail there is no such earlier
gate, and `str.isdigit()`, `re`'s `\\d` and `int()` are all Unicode-aware.

X.680 does not leave this open to interpretation. §12.26 spells an arc out as "an
arbitrarily long sequence of ISO/IEC 10646 characters **in the range 0 (DIGIT ZERO) to 9
(DIGIT NINE)**", and adds that it "shall not commence with a 0 (DIGIT ZERO) character unless
it has only a single character".

  * X.680 §19.9  — `<S>٤٢</S>` decoded to the INTEGER 42 (`str.isdigit()`).
  * X.680 §12.9  — the same for a realnumber.
  * X.680 §9.8   — the same for an XER object identifier, which ALSO accepted `1.2.0840`,
                   a second spelling of `1.2.840`. Three functions earlier in the same file
                   `_parse_integer` already refused a leading zero citing §12.8; the arc
                   production needed it just as much and did not have it.
  * X.697 §32    — the same two defects in JER's object-identifier string.
  * X.680 §12.15.8 — the numeric character escape was `int(digits, 16 if x else 10)`, which
                   accepts PEP 515 underscores, a leading PLUS SIGN, surrounding whitespace,
                   every Unicode decimal digit, and — for the hex form — a SECOND `0x`
                   prefix, because `int(s, 16)` strips one. Nine accepted spellings of `A`.

The fix is one predicate the whole model shares rather than five local repairs, because five
local repairs is how there came to be five.
"""

from __future__ import annotations

from bcir.asn1.jer import decode_jer, encode_jer
from bcir.asn1.schema import Primitive
from bcir.asn1.tags import Universal
from bcir.asn1.values import Asn1Error, is_ascii_digits, is_number_form
from bcir.asn1.xer import XerRules, decode_xer, encode_xer

#: ARABIC-INDIC DIGIT ZERO..NINE (U+0660..U+0669) and FULLWIDTH DIGIT ZERO..NINE
#: (U+FF10..U+FF19). Both answer True to `str.isdigit()`; neither is in X.680's range.
_ARABIC = "٠١٢٣٤٥٦٧٨٩"
_FULLWIDTH = "０１２３４５６７８９"


def _foreign(ascii_digits: str, table: str = _ARABIC) -> str:
    return "".join(table[int(c)] for c in ascii_digits)


def _refused(fn, what: str) -> str:
    try:
        result = fn()
    except Asn1Error as exc:
        return str(exc)
    raise AssertionError(f"{what}: accepted, returning {result!r}")


def _xer(body: str, universal: int, name: str, rules: XerRules = XerRules.BASIC):
    return decode_xer(f"<S>{body}</S>", Primitive(universal, name), rules=rules, name="S")


# --- the shared predicate ------------------------------------------------------------------


def test_the_digit_predicate_means_what_x680_says() -> None:
    """One predicate, so a sixth site cannot quietly disagree with the other five."""
    for good in ("0", "9", "42", "1234567890"):
        assert is_ascii_digits(good), good
    for bad in ("", "٤٢", "４２", "4_2", "-4", "+4", " 4", "4 ", "٤", "四"):
        assert not is_ascii_digits(bad), bad

    # §12.8 / §12.26 add the leading-zero rule on top.
    for good in ("0", "9", "42", "840"):
        assert is_number_form(good), good
    for bad in ("00", "007", "0840", "٤٢", ""):
        assert not is_number_form(bad), bad


# --- X.693 XER ------------------------------------------------------------------------------


def test_xer_numbers_are_spelled_with_ascii_digits() -> None:
    """§19.9's XMLSignedNumber and §12.9's realnumber, in both foreign digit families."""
    assert _xer("42", Universal.INTEGER, "INTEGER") == 42
    assert _xer("-42", Universal.INTEGER, "INTEGER") == -42
    assert _xer("1.5", Universal.REAL, "REAL") == 1.5

    for table, family in ((_ARABIC, "ARABIC-INDIC"), (_FULLWIDTH, "FULLWIDTH")):
        assert "XMLSignedNumber" in _refused(
            lambda t=table: _xer(_foreign("42", t), Universal.INTEGER, "INTEGER"),
            f"{family} digits as an XER INTEGER",
        )
        assert "XMLSignedNumber" in _refused(
            lambda t=table: _xer("-" + _foreign("42", t), Universal.INTEGER, "INTEGER"),
            f"a negative {family} XER INTEGER",
        )
        assert "realnumber" in _refused(
            lambda t=table: _xer(_foreign("1", t) + "." + _foreign("5", t), Universal.REAL, "REAL"),
            f"{family} digits as an XER REAL",
        )

    # The leading-zero rule `_parse_integer` already had must not have regressed.
    assert "leading zero" in _refused(
        lambda: _xer("0042", Universal.INTEGER, "INTEGER"), "an XER INTEGER with a leading zero"
    )


def test_an_xer_object_identifier_arc_is_a_number_form() -> None:
    """§9.8 with §12.26: ASCII digits, and no leading zero unless the arc is one digit.

    `1.2.٨٤٠`, `1.2.0840` and `1.2.840` were three spellings of one object identifier.
    """
    oid = Universal.OBJECT_IDENTIFIER
    assert _xer("1.2.840", oid, "OID") == (1, 2, 840)
    assert _xer("2.0.1", oid, "OID") == (2, 0, 1), "a single-digit zero arc is legal"

    for body, why in (
        (f"1.2.{_foreign('840')}", "ARABIC-INDIC digits"),
        (f"1.2.{_foreign('840', _FULLWIDTH)}", "FULLWIDTH digits"),
        ("1.2.0840", "a leading zero"),
        ("1.2.00", "a two-digit zero"),
    ):
        assert "XMLObjectIdentifierValue" in _refused(
            lambda b=body: _xer(b, oid, "OID"), f"{why} in an XER arc"
        )


def test_the_xer_numeric_escape_is_x680s_escape_and_not_pythons_int() -> None:
    """§12.15.8. Nine byte strings decoded to the single character `A`.

    The `&#x0x41;` case is the one that shows this was never a near-miss: `int(s, 16)`
    strips a `0x` prefix, so the hex form accepted its own prefix twice.
    """
    text = Universal.UTF8_STRING
    assert _xer("&#65;", text, "UTF8String") == "A"
    assert _xer("&#x41;", text, "UTF8String") == "A"
    # §12.15.8 says "decimal digits" with no leading-zero rule, unlike `number`, so this
    # stays legal. Pinned so a future tightening is a deliberate decision, not a slip.
    assert _xer("&#0065;", text, "UTF8String") == "A"

    for escape, why in (
        ("&#6_5;", "a PEP 515 underscore"),
        ("&#x4_1;", "a PEP 515 underscore in the hex form"),
        ("&#+65;", "a PLUS SIGN"),
        ("&# 65 ;", "surrounding whitespace"),
        (f"&#{_foreign('65')};", "ARABIC-INDIC digits"),
        (f"&#{_foreign('65', _FULLWIDTH)};", "FULLWIDTH digits"),
        ("&#x0x41;", "a second 0x prefix, which int(s, 16) strips"),
        ("&#-65;", "a MINUS SIGN"),
    ):
        assert "numeric escape" in _refused(
            lambda e=escape: _xer(e, text, "UTF8String"), f"{why} in a numeric escape"
        )

    # The three named escapes are untouched, and so is a document that uses none.
    assert _xer("&amp;&lt;&gt;", text, "UTF8String") == "&<>"
    assert _xer("plain", text, "UTF8String") == "plain"


# --- X.697 JER -------------------------------------------------------------------------------


def test_a_jer_object_identifier_arc_is_a_number_form() -> None:
    """§32 borrows the same arc production, and had the same two defects.

    JER's own INTEGER path was already safe: its scanner enforces the JSON number grammar
    before converting, which is the pattern the rest of this now follows. An object
    identifier arrives as a JSON *string*, so no number grammar ever guarded it.
    """
    for universal, name in (
        (Universal.OBJECT_IDENTIFIER, "OBJECT IDENTIFIER"),
        (Universal.RELATIVE_OID, "RELATIVE-OID"),
    ):
        kind = Primitive(universal, name)
        assert decode_jer('"1.2.840"', kind) == (1, 2, 840)
        assert decode_jer('"2.0.1"', kind) == (2, 0, 1)

        for text, why in (
            (f'"1.2.{_foreign("840")}"', "ARABIC-INDIC digits"),
            (f'"1.2.{_foreign("840", _FULLWIDTH)}"', "FULLWIDTH digits"),
            ('"1.2.0840"', "a leading zero"),
            ('"1.2.8_40"', "a PEP 515 underscore"),
            ('"1.2.+840"', "a PLUS SIGN"),
        ):
            assert "XMLObjectIdentifierValue" in _refused(
                lambda t=text, k=kind: decode_jer(t, k), f"{why} in a JER arc"
            )


def test_the_text_rails_still_round_trip_everything_they_encode() -> None:
    """The other half: a guard that rejects the encoder's own output is the worse bug."""
    cases = [
        (Universal.INTEGER, "INTEGER", [0, 1, -1, 42, -42, 10**30, -(10**30)]),
        (Universal.REAL, "REAL", [0.0, 1.5, -1.5, 1e10, 1e-10]),
        (Universal.OBJECT_IDENTIFIER, "OID", [(1, 2, 840, 113549), (2, 0, 1), (0, 0)]),
        (Universal.UTF8_STRING, "UTF8String", ["", "plain", "&<>", "héllo ✓", "٤٢"]),
    ]
    for universal, name, values in cases:
        kind = Primitive(universal, name)
        for value in values:
            xml = encode_xer(kind, value, name="S")
            assert decode_xer(xml, kind, name="S") == value, (name, value, xml)
            js = encode_jer(kind, value)
            assert decode_jer(js, kind) == value, (name, value, js)

    # A string may of course CONTAIN foreign digits -- they are characters there, not a
    # number, and that distinction is the entire point of the fix.
    text = Primitive(Universal.UTF8_STRING, "UTF8String")
    assert decode_xer(encode_xer(text, "٤٢", name="S"), text, name="S") == "٤٢"
