"""Lexical items of ASN.1 notation — Rec. ITU-T X.680 (02/2021) clause 12.

The lexer is where a hand-written ASN.1 front-end usually goes wrong, because three
of X.680's lexical rules are unlike most languages and each one silently changes what
a module means:

* **§12.6 comments.** A `--` comment ends at the next `--` *or* at the end of the line,
  whichever comes first — so `-- a -- b` leaves `b` as live text. `/* */` comments
  NEST (§12.6.4), unlike C. Getting either wrong makes a module parse as something
  its author did not write.
* **§12.2/§12.3 hyphens.** A typereference or identifier may contain a hyphen, but not
  two in a row (that would be a comment) and not as its last character. So
  `identified-organization` is one lexeme while `a-b` in `SIZE (a-b)` is not — the
  decision needs one character of lookahead past the hyphen.
* **§12.8 numbers.** A leading zero is not permitted (`0` alone is, `01` is not), which
  is what keeps `0` unambiguous as an OID arc.

The item names below are X.680's own (`typereference`, `identifier`, `bstring`, …) so
that the parser reads against the grammar in the standard rather than a paraphrase.
"""

from __future__ import annotations

from dataclasses import dataclass

#: X.680 §12.38 Table 3 — the reserved words. A word in this set is never a
#: typereference, which is the only thing that distinguishes `SET` from a type named
#: `SET` (the latter being illegal precisely because of this table).
RESERVED = frozenset("""
ABSENT ABSTRACT-SYNTAX ALL APPLICATION AUTOMATIC BEGIN BIT BMPString BOOLEAN BY
CHARACTER CHOICE CLASS COMPONENT COMPONENTS CONSTRAINED CONTAINING DATE
DATE-TIME DEFAULT DEFINITIONS DURATION EMBEDDED ENCODED ENCODING-CONTROL END
ENUMERATED EXCEPT EXPLICIT EXPORTS EXTENSIBILITY EXTERNAL FALSE FROM
GeneralizedTime GeneralString GraphicString IA5String IDENTIFIER IMPLICIT
IMPLIED IMPORTS INCLUDES INSTANCE INSTRUCTIONS INTEGER INTERSECTION ISO646String
MAX MIN MINUS-INFINITY NOT-A-NUMBER NULL NumericString OBJECT ObjectDescriptor
OCTET OF OID-IRI OPTIONAL PATTERN PDV PLUS-INFINITY PRESENT PrintableString
PRIVATE REAL RELATIVE-OID RELATIVE-OID-IRI SEQUENCE SET SETTINGS SIZE STRING
SYNTAX T61String TAGS TeletexString TIME TIME-OF-DAY TRUE TYPE-IDENTIFIER UNION
UNIQUE UNIVERSAL UniversalString UTCTime UTF8String VideotexString VisibleString
WITH
""".split())

#: §12.16–§12.37, longest first so `::=` is never read as `:` and `...` never as `..`.
_PUNCTUATION = ("::=", "...", "[[", "]]", "..", "{", "}", "<", ">", ",", ".", "/",
                "(", ")", "[", "]", "-", ":", "=", ";", "@", "|", "!", "^")


class Asn1SyntaxError(Exception):
    """A lexical or syntactic fault, carrying the source position that caused it."""

    def __init__(self, message: str, line: int, column: int, source: str = "<asn1>"):
        super().__init__(f"{source}:{line}:{column}: {message}")
        self.message, self.line, self.column, self.source = message, line, column, source


@dataclass(frozen=True)
class Token:
    kind: str          # typereference | identifier | number | bstring | hstring
                       # | cstring | reserved | punct | end
    text: str
    line: int
    column: int

    def __repr__(self) -> str:                            # pragma: no cover - debug aid
        return f"{self.kind}({self.text!r})@{self.line}:{self.column}"


def _is_alnum(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


class Lexer:
    def __init__(self, text: str, source: str = "<asn1>"):
        self.text, self.source = text, source
        self.pos, self.line, self.col = 0, 1, 1

    def _error(self, message: str) -> Asn1SyntaxError:
        return Asn1SyntaxError(message, self.line, self.col, self.source)

    def _advance(self, count: int = 1) -> str:
        chunk = self.text[self.pos:self.pos + count]
        for ch in chunk:
            if ch == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1
        self.pos += count
        return chunk

    def _skip_trivia(self) -> None:
        """Whitespace and both comment forms, until real text or end of input."""
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch in " \t\r\n\v\f":
                self._advance()
            elif self.text.startswith("--", self.pos):
                self._skip_line_comment()
            elif self.text.startswith("/*", self.pos):
                self._skip_block_comment()
            else:
                return

    def _skip_line_comment(self) -> None:
        """§12.6.3: a `--` comment ends at the next `--` OR at the end of the line."""
        self._advance(2)
        while self.pos < len(self.text):
            if self.text[self.pos] == "\n":
                return                                     # newline terminates, unconsumed
            if self.text.startswith("--", self.pos):
                self._advance(2)                           # explicit terminator
                return
            self._advance()

    def _skip_block_comment(self) -> None:
        """§12.6.4: `/* */` comments NEST, so track depth rather than scanning to `*/`."""
        start_line, start_col = self.line, self.col
        depth = 0
        while self.pos < len(self.text):
            if self.text.startswith("/*", self.pos):
                self._advance(2)
                depth += 1
            elif self.text.startswith("*/", self.pos):
                self._advance(2)
                depth -= 1
                if depth == 0:
                    return
            else:
                self._advance()
        raise Asn1SyntaxError("unterminated /* comment", start_line, start_col,
                              self.source)

    def _word(self) -> Token:
        """§12.2/§12.3/§12.38: typereference, identifier, or reserved word.

        A hyphen continues the word only when a letter or digit follows it; that single
        character of lookahead is what separates `identified-organization` (one lexeme)
        from `a-b` (three) and from `a--b` (a word, then a comment).
        """
        line, col, start = self.line, self.col, self.pos
        self._advance()
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if _is_alnum(ch) or ch == "_":
                self._advance()
            elif (ch == "-" and self.pos + 1 < len(self.text)
                    and _is_alnum(self.text[self.pos + 1])):
                self._advance(2)
            else:
                break
        text = self.text[start:self.pos]
        if text in RESERVED:
            return Token("reserved", text, line, col)
        kind = "typereference" if text[0].isupper() else "identifier"
        return Token(kind, text, line, col)

    def _number(self) -> Token:
        """§12.8: `0`, or a non-zero digit followed by digits. No leading zeros."""
        line, col, start = self.line, self.col, self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self._advance()
        text = self.text[start:self.pos]
        if len(text) > 1 and text[0] == "0":
            raise Asn1SyntaxError(
                f"number {text!r} has a leading zero (X.680 12.8)", line, col,
                self.source)
        return Token("number", text, line, col)

    def _quoted(self) -> Token:
        """§12.10/§12.12 bstring and hstring, §12.14 cstring."""
        line, col = self.line, self.col
        quote = self.text[self.pos]
        self._advance()
        chunk: list[str] = []
        while True:
            if self.pos >= len(self.text):
                raise Asn1SyntaxError("unterminated string", line, col, self.source)
            ch = self.text[self.pos]
            if ch == quote:
                self._advance()
                # §12.14: a doubled quote inside a cstring denotes one quote character.
                if quote == '"' and self.pos < len(self.text) and self.text[self.pos] == '"':
                    chunk.append('"')
                    self._advance()
                    continue
                break
            chunk.append(ch)
            self._advance()
        body = "".join(chunk)
        if quote == '"':
            return Token("cstring", body, line, col)
        # A `'...'` literal is bstring or hstring depending on the letter that follows.
        suffix = self.text[self.pos:self.pos + 1].upper()
        stripped = "".join(body.split())
        if suffix == "B":
            self._advance()
            if any(c not in "01" for c in stripped):
                raise Asn1SyntaxError("bstring must contain only 0 and 1 (X.680 12.10)",
                                      line, col, self.source)
            return Token("bstring", stripped, line, col)
        if suffix == "H":
            self._advance()
            if any(c not in "0123456789ABCDEFabcdef" for c in stripped):
                raise Asn1SyntaxError("hstring must be hexadecimal (X.680 12.12)",
                                      line, col, self.source)
            return Token("hstring", stripped.upper(), line, col)
        raise Asn1SyntaxError("a '...' literal must be followed by B or H "
                              "(X.680 12.10/12.12)", line, col, self.source)

    def tokens(self) -> list[Token]:
        out: list[Token] = []
        while True:
            self._skip_trivia()
            if self.pos >= len(self.text):
                out.append(Token("end", "", self.line, self.col))
                return out
            ch = self.text[self.pos]
            if ch.isascii() and ch.isalpha():
                out.append(self._word())
            elif ch.isdigit():
                out.append(self._number())
            elif ch in "\"'":
                out.append(self._quoted())
            elif ch == "&":
                # X.681 §7.4/§7.5 field references (`&Type`, `&value`). This front-end
                # does not implement information object classes, but it must LEX them:
                # dying here with "unexpected character '&'" would hide the real reason
                # behind a character-level complaint, and the parser can only name
                # X.681 if the file tokenizes far enough to reach the CLASS keyword.
                line, col = self.line, self.col
                self._advance()
                word = self._word() if (self.pos < len(self.text)
                                        and self.text[self.pos].isalpha()) else None
                out.append(Token("fieldreference",
                                 "&" + (word.text if word else ""), line, col))
            else:
                for punct in _PUNCTUATION:
                    if self.text.startswith(punct, self.pos):
                        line, col = self.line, self.col
                        self._advance(len(punct))
                        out.append(Token("punct", punct, line, col))
                        break
                else:
                    raise self._error(f"unexpected character {ch!r}")


def tokenize(text: str, source: str = "<asn1>") -> list[Token]:
    return Lexer(text, source).tokens()


__all__ = ["Asn1SyntaxError", "Lexer", "RESERVED", "Token", "tokenize"]
