"""C lexer for the frontend subset. Tokenizes identifiers/keywords, integer literals (decimal, hex
`0x`, binary `0b`, C23 digit separators `'`, `u/U/l/L` suffixes), floating literals (decimal and
C `0x1.8p3` hex floats, with `f`/`l` suffixes), the operators L1–L4 need, and
punctuation; skips whitespace, `//` + `/* */` comments, and (for now) preprocessor `#` lines —
`#include <stdint.h>` is recognized but its types are built in, so the L7 preprocessor is deferred.
"""
from __future__ import annotations

from dataclasses import dataclass


class CLexError(Exception):
    """A lexing error. `pos` is the source byte offset of the offending character (for the caret)."""
    def __init__(self, message: str, pos: int | None = None):
        super().__init__(message)
        self.pos = pos


@dataclass(frozen=True)
class Tok:
    kind: str           # IDENT | INT | CHAR | STRING | OP | PUNCT | EOF
    text: str
    pos: int


KEYWORDS = frozenset({
    "struct", "union", "return", "if", "else", "while", "for", "do", "break", "continue",
    "void", "_Bool", "bool", "char", "short", "int", "long", "unsigned", "signed",
    "const", "volatile", "static", "inline", "sizeof", "typedef", "enum",
})

# Multi-char operators first (longest-match), then single-char.
_OPS = ["<<=", ">>=", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
        "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
        "+", "-", "*", "/", "%", "&", "|", "^", "~", "!", "<", ">", "="]
_PUNCT = set("(){}[];,.:?")


def _scan_decimal_float(src: str, i: int, n: int) -> int | None:
    """If `src[i:]` begins a *decimal* floating literal (`1.5` / `.5` / `1.` / `1e10` / `1.5e-3` /
    `3.14f`), return its end index, else None. A bare integer (no `.` and no exponent) returns None
    so it lexes as an INT; hex floats (`0x1p4`) are handled separately by `_scan_hex_float`."""
    j = i
    has_digit = False
    while j < n and (src[j].isdigit() or src[j] == "'"):
        j += 1
        has_digit = True
    has_dot = False
    if j < n and src[j] == ".":
        has_dot = True
        j += 1
        while j < n and (src[j].isdigit() or src[j] == "'"):
            j += 1
            has_digit = True
    has_exp = False
    if has_digit and j < n and src[j] in "eE":                # an exponent: e[+-]?digits
        k = j + 1
        if k < n and src[k] in "+-":
            k += 1
        if k < n and src[k].isdigit():
            has_exp = True
            j = k
            while j < n and src[j].isdigit():
                j += 1
    if not has_digit or not (has_dot or has_exp):             # no fraction/exponent -> not a float
        return None
    if j < n and src[j] in "fFlL":                            # f/F (float) or l/L (long double) suffix
        j += 1
    return j


_HEXD = "0123456789abcdefABCDEF'"      # hex digits (with the C23 ' separator)


def _scan_hex_float(src: str, i: int, n: int) -> int | None:
    """If `src[i:]` begins a C hex floating literal (`0x1p4` / `0x1.8p3` / `0x.8p1` / `0xAp-2f`),
    return its end index, else None. A hex float needs the `0x` prefix, at least one significand hex
    digit, and a *mandatory* binary `p`/`P` exponent (decimal digits) — the `p` is what distinguishes
    it from a plain hex integer like `0x1f`. The significand may carry a `.` and C23 `'` separators."""
    if src[i:i + 2] not in ("0x", "0X"):
        return None
    j = i + 2
    start = j
    while j < n and src[j] in _HEXD:                          # the integer part of the significand
        j += 1
    has_sig = j > start
    if j < n and src[j] == ".":                               # an optional fractional part
        j += 1
        fstart = j
        while j < n and src[j] in _HEXD:
            j += 1
        has_sig = has_sig or j > fstart
    if not has_sig or j >= n or src[j] not in "pP":           # need digits AND the binary exponent
        return None
    j += 1
    if j < n and src[j] in "+-":                              # an optionally-signed exponent
        j += 1
    estart = j
    while j < n and (src[j].isdigit() or src[j] == "'"):
        j += 1
    if j == estart:                                           # the exponent needs at least one digit
        return None
    if j < n and src[j] in "fFlL":                            # f/F (float) or l/L (long double) suffix
        j += 1
    return j


def tokenize(src: str) -> list[Tok]:
    toks: list[Tok] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":          # line comment
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":          # block comment
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                i += 1
            i += 2
            continue
        if c == "#":                                              # preprocessor line (skipped)
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c in "LuU":                                            # wide/UTF literal prefix L/u/U/u8
            pfx = ""                                              # before a " or ' (else an identifier)
            if src[i:i + 2] == "u8" and i + 2 < n and src[i + 2] in "\"'":
                pfx = "u8"
            elif i + 1 < n and src[i + 1] in "\"'":
                pfx = c
            if pfx:
                q = i + len(pfx)
                quote = src[q]
                j = q + 1
                while j < n and src[j] != quote:
                    j += 2 if (src[j] == "\\" and j + 1 < n) else 1
                if j >= n:
                    raise CLexError(f"unterminated {'string' if quote == chr(34) else 'character'} "
                                    f"literal", pos=i)
                toks.append(Tok("STRING" if quote == '"' else "CHAR", src[i:j + 1], i))
                i = j + 1
                continue
        if c.isalpha() or c == "_":                               # identifier / keyword
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            toks.append(Tok("IDENT", src[i:j], i))
            i = j
            continue
        if ((c.isdigit() and src[i:i + 2] not in ("0x", "0X", "0b", "0B"))   # decimal float literal
                or (c == "." and i + 1 < n and src[i + 1].isdigit())):        # (.5 / 1.5 / 1e10 / 3.14f)
            end = _scan_decimal_float(src, i, n)
            if end is not None:
                toks.append(Tok("FLOAT", src[i:end], i))
                i = end
                continue
        if src[i:i + 2] in ("0x", "0X"):                          # hex float (0x1p4) vs hex int (0x1f)
            end = _scan_hex_float(src, i, n)                      # only matches when a `p` exponent is present
            if end is not None:
                toks.append(Tok("FLOAT", src[i:end], i))
                i = end
                continue
        if c.isdigit():                                           # integer literal
            j = i
            if src[j:j + 2] in ("0x", "0X", "0b", "0B"):
                j += 2
            while j < n and (src[j].isalnum() or src[j] == "'"):  # digits, suffix, C23 separators
                j += 1
            toks.append(Tok("INT", src[i:j], i))
            i = j
            continue
        if c == '"':                                              # string literal
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if (src[j] == "\\" and j + 1 < n) else 1   # skip an escaped char as a unit
            if j >= n:
                raise CLexError("unterminated string literal", pos=i)
            toks.append(Tok("STRING", src[i:j + 1], i))           # text includes the surrounding quotes
            i = j + 1
            continue
        if c == "'":                                              # character constant
            j = i + 1
            while j < n and src[j] != "'":
                j += 2 if (src[j] == "\\" and j + 1 < n) else 1   # skip an escaped char as a unit
            if j >= n:
                raise CLexError("unterminated character constant", pos=i)
            toks.append(Tok("CHAR", src[i:j + 1], i))             # text includes the surrounding quotes
            i = j + 1
            continue
        for op in _OPS:                                           # operators (longest match)
            if src.startswith(op, i):
                toks.append(Tok("OP", op, i))
                i += len(op)
                break
        else:
            if c in _PUNCT:
                toks.append(Tok("PUNCT", c, i))
                i += 1
            else:
                raise CLexError(f"unexpected character {c!r}", pos=i)
    toks.append(Tok("EOF", "", n))
    return toks


_SIMPLE_ESCAPE = {"n": 10, "t": 9, "r": 13, "\\": 92, "'": 39, '"': 34,
                  "a": 7, "b": 8, "f": 12, "v": 11, "?": 63}

_LIT_PREFIXES = ("u8", "L", "u", "U")           # wide/UTF string + character literal prefixes


def split_lit_prefix(text: str) -> tuple[str, str]:
    """Split an optional wide/UTF prefix (`L` / `u` / `U` / `u8`) off a string/character literal's
    spelling, returning ``(prefix, rest)`` where ``rest`` begins with the opening quote."""
    for p in _LIT_PREFIXES:
        if text.startswith(p) and len(text) > len(p) and text[len(p)] in "\"'":
            return p, text[len(p):]
    return "", text


def str_elem_size(prefix: str) -> int:
    """The element size of a string literal with this prefix on the Linux/Clang ABI: plain/`u8` = 1
    (`char`), `u` = 2 (`char16_t`), `L`/`U` = 4 (`wchar_t` / `char32_t`)."""
    return {"u": 2, "L": 4, "U": 4}.get(prefix, 1)


def decode_c_bytes(inner: str) -> list[int]:
    """Decode the *inner* text of a string/character literal (surrounding quotes already stripped)
    to its sequence of byte values, interpreting C escape sequences: the simple `\\c` escapes, an
    octal `\\NNN` (up to three digits), and a hex `\\xHH..` (all following hex digits)."""
    out: list[int] = []
    i, ln = 0, len(inner)
    while i < ln:
        ch = inner[i]
        if ch == "\\" and i + 1 < ln:
            e = inner[i + 1]
            if e == "x":                                       # \xHH.. -> all following hex digits
                i, val = i + 2, 0
                while i < ln and inner[i] in "0123456789abcdefABCDEF":
                    val, i = val * 16 + int(inner[i], 16), i + 1
                out.append(val & 0xFF)
            elif e in "01234567":                              # \NNN -> up to three octal digits
                i, val, k = i + 1, 0, 0
                while k < 3 and i < ln and inner[i] in "01234567":
                    val, i, k = val * 8 + int(inner[i], 8), i + 1, k + 1
                out.append(val & 0xFF)
            else:                                              # \n, \t, \\, \", \0-less simple escapes
                out.append(_SIMPLE_ESCAPE.get(e, ord(e)) & 0xFF)
                i += 2
        else:
            out.append(ord(ch) & 0xFF)
            i += 1
    return out


def parse_char_literal(text: str) -> int:
    """Decode a C character constant to its `int` value. A single character is its byte value
    sign-extended as a (signed) `char`; a multi-character constant `'AB'` packs big-endian
    (Clang/GCC: `('A'<<8)|'B'`), interpreted as a 32-bit `int`. An optional wide/UTF prefix
    (`L`/`u`/`U`) does not change the (ASCII) code-point value. `text` includes the quotes."""
    _pfx, text = split_lit_prefix(text)
    inner = text[1:-1] if len(text) >= 2 and text[0] == "'" else text
    bs = decode_c_bytes(inner)
    if not bs:
        return 0
    if len(bs) == 1:
        b = bs[0]
        return b - 256 if b >= 128 else b                      # a single char is a signed char
    v = 0
    for b in bs:
        v = ((v << 8) | b) & 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v              # an int32 multi-character constant


def parse_int_literal(text: str) -> int:
    """Decode a C integer literal: strip C23 digit separators + the u/U/l/L suffix, honor 0x / 0b.
    A malformed pp-number that the lexer tokenized as INT but is not a valid integer (e.g. `9a`) is a
    clean CLexError, not a bare ValueError -- so the diagnostics / fallback paths report, not crash."""
    t = text.replace("'", "")
    while t and t[-1] in "uUlL":
        t = t[:-1]
    try:
        if t[:2] in ("0x", "0X"):
            return int(t, 16)
        if t[:2] in ("0b", "0B"):
            return int(t[2:], 2)
        if len(t) > 1 and t[0] == "0":
            return int(t, 8)
        return int(t or "0", 10)
    except ValueError as e:
        raise CLexError(f"invalid integer literal {text!r}") from e
