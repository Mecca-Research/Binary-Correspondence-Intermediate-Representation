"""C lexer for the frontend subset. Tokenizes identifiers/keywords, integer literals (decimal, hex
`0x`, binary `0b`, C23 digit separators `'`, `u/U/l/L` suffixes), the operators L1–L4 need, and
punctuation; skips whitespace, `//` + `/* */` comments, and (for now) preprocessor `#` lines —
`#include <stdint.h>` is recognized but its types are built in, so the L7 preprocessor is deferred.
"""
from __future__ import annotations

from dataclasses import dataclass


class CLexError(Exception):
    pass


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
        if c.isalpha() or c == "_":                               # identifier / keyword
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            toks.append(Tok("IDENT", src[i:j], i))
            i = j
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
                raise CLexError(f"unterminated string literal at offset {i}")
            toks.append(Tok("STRING", src[i:j + 1], i))           # text includes the surrounding quotes
            i = j + 1
            continue
        if c == "'":                                              # character constant
            j = i + 1
            while j < n and src[j] != "'":
                j += 2 if (src[j] == "\\" and j + 1 < n) else 1   # skip an escaped char as a unit
            if j >= n:
                raise CLexError(f"unterminated character constant at offset {i}")
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
                raise CLexError(f"unexpected character {c!r} at offset {i}")
    toks.append(Tok("EOF", "", n))
    return toks


_SIMPLE_ESCAPE = {"n": 10, "t": 9, "r": 13, "\\": 92, "'": 39, '"': 34,
                  "a": 7, "b": 8, "f": 12, "v": 11, "?": 63}


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
    (Clang/GCC: `('A'<<8)|'B'`), interpreted as a 32-bit `int`. `text` includes the quotes."""
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
    """Decode a C integer literal: strip C23 digit separators + the u/U/l/L suffix, honor 0x / 0b."""
    t = text.replace("'", "")
    while t and t[-1] in "uUlL":
        t = t[:-1]
    if t[:2] in ("0x", "0X"):
        return int(t, 16)
    if t[:2] in ("0b", "0B"):
        return int(t[2:], 2)
    if len(t) > 1 and t[0] == "0":
        return int(t, 8)
    return int(t or "0", 10)
