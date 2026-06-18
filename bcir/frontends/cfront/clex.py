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
    kind: str           # IDENT | INT | OP | PUNCT | EOF
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
_PUNCT = set("(){}[];,.")


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
