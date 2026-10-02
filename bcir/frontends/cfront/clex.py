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
    kind: str  # IDENT | INT | CHAR | STRING | OP | PUNCT | EOF
    text: str
    pos: int


KEYWORDS = frozenset(
    {
        "struct",
        "union",
        "return",
        "if",
        "else",
        "while",
        "for",
        "do",
        "break",
        "continue",
        "void",
        "_Bool",
        "bool",
        "char",
        "short",
        "int",
        "long",
        "unsigned",
        "signed",
        "const",
        "volatile",
        "static",
        "inline",
        "sizeof",
        "typedef",
        "enum",
        "_BitInt",  # C23 bit-precise integer type `_BitInt(N)` (§6.2.5)
        "__asm__",
        "__volatile__",  # GNU inline assembly (ASM1 trusted opaque edge): only the
        #   DOUBLE-UNDERSCORE spellings are reserved keywords (benign;
        #   implementation-reserved). Plain `asm` is NOT a keyword -- it
        #   stays a usable ISO-C identifier under -std=c11; the parser
        #   recognizes the `asm` STATEMENT contextually (a following `(`
        #   or a volatile/__volatile__/goto qualifier), see cparse._stmt.
    }
)

# Multi-char operators first (longest-match), then single-char.
_OPS = [
    "<<=",
    ">>=",
    "->",
    "++",
    "--",
    "<<",
    ">>",
    "<=",
    ">=",
    "==",
    "!=",
    "&&",
    "||",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "+",
    "-",
    "*",
    "/",
    "%",
    "&",
    "|",
    "^",
    "~",
    "!",
    "<",
    ">",
    "=",
]
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
    if has_digit and j < n and src[j] in "eE":  # an exponent: e[+-]?digits
        k = j + 1
        if k < n and src[k] in "+-":
            k += 1
        if k < n and src[k].isdigit():
            has_exp = True
            j = k
            while j < n and src[j].isdigit():
                j += 1
    if not has_digit or not (has_dot or has_exp):  # no fraction/exponent -> not a float
        return None
    if j < n and src[j] in "fFlL":  # f/F (float) or l/L (long double) suffix
        j += 1
    return j


_HEXD = "0123456789abcdefABCDEF'"  # hex digits (with the C23 ' separator)


def _scan_hex_float(src: str, i: int, n: int) -> int | None:
    """If `src[i:]` begins a C hex floating literal (`0x1p4` / `0x1.8p3` / `0x.8p1` / `0xAp-2f`),
    return its end index, else None. A hex float needs the `0x` prefix, at least one significand hex
    digit, and a *mandatory* binary `p`/`P` exponent (decimal digits) — the `p` is what distinguishes
    it from a plain hex integer like `0x1f`. The significand may carry a `.` and C23 `'` separators."""
    if src[i : i + 2] not in ("0x", "0X"):
        return None
    j = i + 2
    start = j
    while j < n and src[j] in _HEXD:  # the integer part of the significand
        j += 1
    has_sig = j > start
    if j < n and src[j] == ".":  # an optional fractional part
        j += 1
        fstart = j
        while j < n and src[j] in _HEXD:
            j += 1
        has_sig = has_sig or j > fstart
    if not has_sig or j >= n or src[j] not in "pP":  # need digits AND the binary exponent
        return None
    j += 1
    if j < n and src[j] in "+-":  # an optionally-signed exponent
        j += 1
    estart = j
    while j < n and (src[j].isdigit() or src[j] == "'"):
        j += 1
    if j == estart:  # the exponent needs at least one digit
        return None
    if j < n and src[j] in "fFlL":  # f/F (float) or l/L (long double) suffix
        j += 1
    return j


#: The longest identifier or floating constant the frontend accepts (CF-BUF): C11 5.2.4.1's 63 significant
#: initial characters of an internal identifier. The C twin's claim graph holds a name, a tag, an op and an
#: alias of that many characters (`BCIR_CIR_IDENT_MAX`, runtime/c/bcir_cir.h) and a floating constant's
#: spelling in its op, so a longer one could only be truncated there -- another callee, member or constant.
#: The oracle has no such buffer; it refuses what the twin would truncate, with the twin's diagnostic.
IDENT_MAX = 63


def _too_long(kind: str, text: str, pos: int) -> None:
    if len(text) > IDENT_MAX:
        raise CLexError(f"{kind} longer than {IDENT_MAX} characters is not supported", pos=pos)


def tokenize(src: str) -> list[Tok]:
    toks: list[Tok] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":  # line comment
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":  # block comment
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                i += 1
            i += 2
            continue
        if c == "#":  # preprocessor line (skipped)
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c in "LuU":  # wide/UTF literal prefix L/u/U/u8
            pfx = ""  # before a " or ' (else an identifier)
            if src[i : i + 2] == "u8" and i + 2 < n and src[i + 2] in "\"'":
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
                    raise CLexError(
                        f"unterminated {'string' if quote == chr(34) else 'character'} literal",
                        pos=i,
                    )
                toks.append(Tok("STRING" if quote == '"' else "CHAR", src[i : j + 1], i))
                i = j + 1
                continue
        if c.isalpha() or c == "_":  # identifier / keyword
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            _too_long("an identifier", src[i:j], i)
            toks.append(Tok("IDENT", src[i:j], i))
            i = j
            continue
        if (
            (
                c.isdigit() and src[i : i + 2] not in ("0x", "0X", "0b", "0B")
            )  # decimal float literal
            or (c == "." and i + 1 < n and src[i + 1].isdigit())
        ):  # (.5 / 1.5 / 1e10 / 3.14f)
            end = _scan_decimal_float(src, i, n)
            if end is not None:
                _too_long("a floating constant", src[i:end], i)
                toks.append(Tok("FLOAT", src[i:end], i))
                i = end
                continue
        if src[i : i + 2] in ("0x", "0X"):  # hex float (0x1p4) vs hex int (0x1f)
            end = _scan_hex_float(src, i, n)  # only matches when a `p` exponent is present
            if end is not None:
                _too_long("a floating constant", src[i:end], i)
                toks.append(Tok("FLOAT", src[i:end], i))
                i = end
                continue
        if c.isdigit():  # integer literal
            j = i
            if src[j : j + 2] in ("0x", "0X", "0b", "0B"):
                j += 2
            while j < n and (src[j].isalnum() or src[j] == "'"):  # digits, suffix, C23 separators
                j += 1
            toks.append(Tok("INT", src[i:j], i))
            i = j
            continue
        if c == '"':  # string literal
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if (src[j] == "\\" and j + 1 < n) else 1  # skip an escaped char as a unit
            if j >= n:
                raise CLexError("unterminated string literal", pos=i)
            toks.append(Tok("STRING", src[i : j + 1], i))  # text includes the surrounding quotes
            i = j + 1
            continue
        if c == "'":  # character constant
            j = i + 1
            while j < n and src[j] != "'":
                j += 2 if (src[j] == "\\" and j + 1 < n) else 1  # skip an escaped char as a unit
            if j >= n:
                raise CLexError("unterminated character constant", pos=i)
            toks.append(Tok("CHAR", src[i : j + 1], i))  # text includes the surrounding quotes
            i = j + 1
            continue
        if src.startswith("...", i):  # the variadic ellipsis (one 3-char token)
            toks.append(Tok("PUNCT", "...", i))
            i += 3
            continue
        for op in _OPS:  # operators (longest match)
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


_SIMPLE_ESCAPE = {
    "n": 10,
    "t": 9,
    "r": 13,
    "\\": 92,
    "'": 39,
    '"': 34,
    "a": 7,
    "b": 8,
    "f": 12,
    "v": 11,
    "?": 63,
}

_LIT_PREFIXES = ("u8", "L", "u", "U")  # wide/UTF string + character literal prefixes


def split_lit_prefix(text: str) -> tuple[str, str]:
    """Split an optional wide/UTF prefix (`L` / `u` / `U` / `u8`) off a string/character literal's
    spelling, returning ``(prefix, rest)`` where ``rest`` begins with the opening quote."""
    for p in _LIT_PREFIXES:
        if text.startswith(p) and len(text) > len(p) and text[len(p)] in "\"'":
            return p, text[len(p) :]
    return "", text


def lit_prefix(spelling: str) -> str:
    """The encoding prefix of a (possibly concatenated) string literal -- its pieces joined by a space, as the
    parser keeps them: the one prefix its pieces carry. A piece without one takes the others' (C11 6.4.5p5:
    `"a" L"b"` is a wide literal); two different prefixes are a ValueError, as Clang refuses them. The twin's
    `str_prefix` (CF-STRELEM)."""
    found, i, n = "", 0, len(spelling)
    while i < n:
        if spelling[i] == " ":
            i += 1
            continue
        j = spelling.find('"', i)
        if j < 0:
            break
        p = spelling[i:j]
        if p:
            if found and found != p:
                raise ValueError(STR_PREFIXES)
            found = p
        k = j + 1
        while k < n and spelling[k] != '"':
            k += 2 if spelling[k] == "\\" else 1
        i = k + 1
    return found


#: two pieces of one string literal with different encoding prefixes (`L"a" u"b"`): Clang refuses them
STR_PREFIXES = "string literals with different encoding prefixes are concatenated"


def str_elem_size(prefix: str, abi=None) -> int:
    """The element size of a string literal with this prefix: plain/`u8` = 1 (`char`), `u` = 2
    (`char16_t`), `U` = 4 (`char32_t`), `L` the target's `wchar_t` (4, or 2 on Windows)."""
    if prefix == "L":
        return abi.wchar_size if abi is not None else 4
    return {"u": 2, "U": 4}.get(prefix, 1)


def str_units(spelling: str, unit_bytes) -> tuple[str, list[int]]:
    """The code units a (possibly concatenated) string literal holds, *excluding* the terminating NUL,
    for a string that initializes a character array (C11 6.7.9p14-15): its prefix and the unit values.
    `unit_bytes` maps the prefix to its code-unit width (the target's `wchar_t` for `L`). Adjacent
    pieces stay separate, as `_str_bytes` counts them, so an escape never merges with the next piece.
    Each unit is one source character or one escape (`\\c`, octal `\\NNN`, hex `\\xH..`). A spelling
    the twin cannot decode the same way is a ValueError the caller refuses: pieces with different
    prefixes, a non-ASCII source character or a universal character name (their code units depend on
    the source and execution encodings), or an escape too wide for the unit (Clang's error). The twin's
    `str_units` decodes the same units."""
    pieces, i, n = [], 0, len(spelling)
    while i < n:  # the pieces: (prefix, first inner index, closing-quote index)
        if spelling[i] == " ":
            i += 1
            continue
        j = spelling.find('"', i)
        if j < 0:
            break
        k = j + 1
        while k < n and spelling[k] != '"':
            k += 2 if spelling[k] == "\\" else 1
        pieces.append((spelling[i:j], j + 1, min(k, n)))
        i = k + 1
    prefixes = {p for p, _a, _b in pieces}
    if len(prefixes) > 1:
        raise ValueError("a string initializer concatenating literals of different prefixes")
    prefix = next(iter(prefixes)) if prefixes else ""
    width = unit_bytes(prefix)
    units: list[int] = []
    for _p, i, end in pieces:
        while i < end:
            ch = spelling[i]
            if ord(ch) >= 128:
                raise ValueError("a non-ASCII character in a string initializer is not supported")
            if ch != "\\" or i + 1 >= end:
                units.append(ord(ch))
                i += 1
                continue
            e = spelling[i + 1]
            if e == "x":  # \xH.. -> every following hex digit
                i, val, nd = i + 2, 0, 0
                while i < end and spelling[i] in "0123456789abcdefABCDEF":
                    val, i, nd = val * 16 + int(spelling[i], 16), i + 1, nd + 1
                if not nd:
                    raise ValueError("a \\x escape with no hex digit")
            elif e in "01234567":  # \NNN -> up to three octal digits
                i, val, k = i + 1, 0, 0
                while k < 3 and i < end and spelling[i] in "01234567":
                    val, i, k = val * 8 + int(spelling[i], 8), i + 1, k + 1
            elif e in "uU":
                raise ValueError(
                    "a universal character name in a string initializer is not supported"
                )
            else:
                val, i = _SIMPLE_ESCAPE.get(e, ord(e)), i + 2
            if val >= 1 << (8 * width):
                raise ValueError("an escape sequence out of range for its character type")
            units.append(val)
    return prefix, units


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
            if e == "x":  # \xHH.. -> all following hex digits
                i, val = i + 2, 0
                while i < ln and inner[i] in "0123456789abcdefABCDEF":
                    val, i = val * 16 + int(inner[i], 16), i + 1
                out.append(val & 0xFF)
            elif e in "01234567":  # \NNN -> up to three octal digits
                i, val, k = i + 1, 0, 0
                while k < 3 and i < ln and inner[i] in "01234567":
                    val, i, k = val * 8 + int(inner[i], 8), i + 1, k + 1
                out.append(val & 0xFF)
            else:  # \n, \t, \\, \", \0-less simple escapes
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
        return b - 256 if b >= 128 else b  # a single char is a signed char
    v = 0
    for b in bs:
        v = ((v << 8) | b) & 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v  # an int32 multi-character constant


_BASE_DIGITS = {16: "0123456789abcdefABCDEF", 10: "0123456789", 8: "01234567", 2: "01"}


def int_suffix_ok(suffix: str) -> bool:
    """Whether `suffix` -- an integer constant's trailing run of `u`/`U`/`l`/`L` -- is one C spells (C11 6.4.4.1p1):
    at most one `u` or `U`, first or last, around nothing, one `l` or `L`, or `ll` or `LL` -- never `lL`, `uu` or
    `lul`. Clang and GCC refuse any other (`invalid suffix 'lL' on integer constant`), and `int_literal_type` reads
    a long rank only up to `ll` (`1lll` had raised a bare `KeyError`) -- the twin's `int_suffix_ok` (CF-SUFFIX)."""
    rest = suffix
    if rest[:1] in ("u", "U"):
        rest = rest[1:]
    elif rest[-1:] in ("u", "U"):
        rest = rest[:-1]
    return rest in ("", "l", "L", "ll", "LL")


def int_literal_parts(text: str) -> "tuple[int, bool, str]":
    """An integer constant (C11 6.4.4.1) as (its value, whether it is decimal, its suffix): the C23 `'`
    digit separators dropped, the trailing run of `u`/`U`/`l`/`L` its suffix, then `0x` hex, `0b` binary,
    a leading `0` octal, else decimal. Each digit is checked against its base, so a malformed pp-number the
    lexer took as INT (`9a`, `08`, `0b2`, and `0o17`, which Python's `int` would read as octal) is a clean
    CLexError, never a value -- the twin's `int_literal` reads the constant the same way."""
    t = text.replace("'", "")
    end = len(t)
    while end and t[end - 1] in "uUlL":
        end -= 1
    body, suffix = t[:end], t[end:]
    if not int_suffix_ok(suffix):
        raise CLexError(f"invalid suffix '{suffix[:48]}' on integer constant")
    if body[:2] in ("0x", "0X"):
        base, digits = 16, body[2:]
    elif body[:2] in ("0b", "0B"):
        base, digits = 2, body[2:]
    elif len(body) > 1 and body[0] == "0":
        base, digits = 8, body[1:]
    else:
        base, digits = 10, body
    if not digits or any(ch not in _BASE_DIGITS[base] for ch in digits):
        raise CLexError(f"invalid integer literal {text!r}")
    return int(digits, base), base == 10, suffix


#: The one reason both rails give for an integer constant no type in its list can hold (C11 6.4.4p2): one
#: past `unsigned long long`, or a decimal one with no `u` past `long long` -- whose type C leaves to the
#: implementation (6.4.4.1p6: GCC gives it `__int128`, Clang `unsigned long long`). The twin had kept such a
#: constant as `LLONG_MAX`.
INT_TOO_LARGE = "an integer constant too large for every type its base and suffix allow"


def parse_int_literal(text: str, pos: int | None = None) -> int:
    """Decode a C integer literal's value (`int_literal_parts`), refusing one that is malformed or that no
    type can hold (`INT_TOO_LARGE`) -- where the parser reads it, so a malformed token in a declarator's place
    stays a parse error the recovering parser resumes after. The twin refuses both where it lexes them."""
    try:
        value, decimal, suffix = int_literal_parts(text)
    except CLexError as e:
        raise CLexError(str(e), pos=pos) from None
    if value >> (64 if ("u" in suffix.lower() or not decimal) else 63):
        raise CLexError(INT_TOO_LARGE, pos=pos)
    return value


#: The one reason both rails give for a character constant they do not read (CF-PPARITH): one with no character, more
#: than one character behind an encoding prefix, an escape past its type's code unit, a universal character name, an
#: escape C does not define (`\q`, GNU's `\e`), or a source character past ASCII. Clang refuses most of these itself
#: (`empty character constant`, `hex escape sequence out of range`, `character too large for enclosing character
#: literal type`); the rest it reads in ways of its own. The twin's `char_bad` (`bcir_intlit.h`).
CHAR_UNSUPPORTED = "unsupported character constant"

#: A simple escape sequence's value (C11 6.4.4.4p1).
_SIMPLE_ESCAPES = {
    "'": 39,
    '"': 34,
    "?": 63,
    "\\": 92,
    "a": 7,
    "b": 8,
    "f": 12,
    "n": 10,
    "r": 13,
    "t": 9,
    "v": 11,
}


def char_constant_units(text: str, wchar_bits: int = 32) -> "tuple[str, list[int]]":
    """A character constant (C11 6.4.4.4, C23's `u8`) as its encoding prefix -- "", "L", "u", "U" or "u8" -- and its
    code units, each an ASCII source character but `'` and `\\` (a space, a tab, a vertical tab or a form feed, or a
    printable one) or a simple, octal or hexadecimal escape, its value within its type's code unit: 8 bits for a plain
    and a `u8` one, 16 for `u`, 32 for `U`, `wchar_bits` (the target's `wchar_t`) for `L`. A plain one may hold
    several (a multi-character constant); a prefixed one holds one. A CLexError(CHAR_UNSUPPORTED) for any other
    (CF-PPARITH; the twin's `char_literal`)."""
    prefix = "u8" if text[:2] == "u8" else text[:1] if text[:1] in ("L", "u", "U") else ""
    body = text[len(prefix) :]
    if len(body) < 2 or body[0] != "'" or body[-1] != "'":
        raise CLexError(CHAR_UNSUPPORTED)
    body = body[1:-1]
    limit = 1 << {"": 8, "u8": 8, "u": 16, "U": 32, "L": wchar_bits}[prefix]
    units, i = [], 0
    while i < len(body):
        ch = body[i]
        if ch == "\\":
            e = body[i + 1 : i + 2]
            if e and e in _SIMPLE_ESCAPES:
                v, i = _SIMPLE_ESCAPES[e], i + 2
            elif e and e in "01234567":
                j = i + 1
                while j < len(body) and j < i + 4 and body[j] in "01234567":
                    j += 1
                v, i = int(body[i + 1 : j], 8), j
            elif e == "x":
                j = i + 2
                while j < len(body) and body[j] in _BASE_DIGITS[16]:
                    j += 1
                if j == i + 2:
                    raise CLexError(CHAR_UNSUPPORTED)
                v, i = int(body[i + 2 : j], 16), j
            else:
                raise CLexError(CHAR_UNSUPPORTED)
        elif ch in "\t\v\f" or (" " <= ch <= "~" and ch not in "'\\"):
            v, i = ord(ch), i + 1
        else:
            raise CLexError(CHAR_UNSUPPORTED)
        if v >= limit:
            raise CLexError(CHAR_UNSUPPORTED)
        units.append(v)
    if not units or (prefix and len(units) > 1):
        raise CLexError(CHAR_UNSUPPORTED)
    return prefix, units
