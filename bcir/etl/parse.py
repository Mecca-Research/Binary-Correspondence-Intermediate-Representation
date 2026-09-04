"""Grammar -> tokens -> BCIR claims (the mouth of BCIR).

A real, runnable mini front-end: a regex lexer plus a recursive-descent parser
for a tiny ROP-claim grammar that emits a BCIR `Module`. The point is that the
parser produces *IR* (claims), not host-language code, so the same machinery can
later ingest ROP/MAP source, object files, and packets. The produced module feeds
`kbcir.optimize` directly, closing the loop text -> claims -> K_BCIR plan.

Grammar (EBNF):
    program := claim+
    claim   := "claim" IDENT "{" stmt* "}"
    stmt    := "reads"  idlist ";"
             | "writes" idlist ";"
             | "count"  INT ";"
             | "lane"   IDENT ";"
             | "stride" IDENT ";"
    idlist  := IDENT ("," IDENT)*
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int


@dataclass(frozen=True)
class TokenRule:
    name: str
    pattern: str
    skip: bool = False
    precedence: int = 0


@dataclass(frozen=True)
class Grammar:
    """Representational parity with `bcir.parse.grammar` (the law)."""

    name: str
    syntax: str
    start_symbol: str
    tokens: tuple[TokenRule, ...]


class LexError(Exception):
    pass


class ParseError(Exception):
    pass


# Punctuation before IDENT; INT before IDENT is unnecessary (disjoint) but explicit.
DEFAULT_TOKENS = (
    TokenRule("WS", r"[ \t\r\n]+", skip=True),
    TokenRule("LBRACE", r"\{"),
    TokenRule("RBRACE", r"\}"),
    TokenRule("SEMI", r";"),
    TokenRule("COMMA", r","),
    TokenRule("INT", r"[0-9]+"),
    TokenRule("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
)


class Lexer:
    def __init__(self, rules: tuple[TokenRule, ...] = DEFAULT_TOKENS):
        self.rules = tuple(rules)
        self._compiled = [(r, re.compile(r.pattern)) for r in self.rules]

    def tokenize(self, text: str) -> list[Token]:
        toks: list[Token] = []
        i = 0
        while i < len(text):
            for rule, rx in self._compiled:
                m = rx.match(text, i)
                if m and m.end() > i:
                    if not rule.skip:
                        toks.append(Token(rule.name, m.group(0), i))
                    i = m.end()
                    break
            else:
                raise LexError(f"unexpected character {text[i]!r} at offset {i}")
        toks.append(Token("EOF", "", len(text)))
        return toks


_LANES = {l.name.lower(): l for l in Lane}
_STRIDES = {s.name.lower(): s for s in StrideClass}


class _Parser:
    def __init__(self, tokens: list[Token]):
        self.toks = tokens
        self.i = 0

    def _peek(self) -> Token:
        return self.toks[self.i]

    def _next(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def _expect(self, kind: str, text: str | None = None) -> Token:
        t = self._peek()
        if t.kind != kind or (text is not None and t.text != text):
            want = text if text is not None else kind
            raise ParseError(f"expected {want!r}, got {t.kind} {t.text!r} at offset {t.pos}")
        return self._next()

    def parse_module(self, name: str = "rop") -> Module:
        m = Module(name=name)
        rid_of: dict[str, int] = {}
        claims: list[Claim] = []
        while self._peek().kind != "EOF":
            claims.append(self._claim(m, rid_of, claim_id=1000 + len(claims)))
        m.add_phase(Phase(phase_id=0, deps=(), claims=claims))
        return m

    def _rid(self, m: Module, rid_of: dict[str, int], sym: str) -> int:
        if sym not in rid_of:
            rid = 10 + len(rid_of)
            rid_of[sym] = rid
            m.add_resource(Resource(rid=rid, name=sym))
        return rid_of[sym]

    def _idlist(self) -> list[str]:
        names = [self._expect("IDENT").text]
        while self._peek().kind == "COMMA":
            self._next()
            names.append(self._expect("IDENT").text)
        return names

    def _claim(self, m: Module, rid_of: dict[str, int], claim_id: int) -> Claim:
        self._expect("IDENT", "claim")
        name = self._expect("IDENT").text
        self._expect("LBRACE")
        reads: tuple[int, ...] = ()
        writes: tuple[int, ...] = ()
        count = 1
        lane = Lane.U
        stride = StrideClass.UNIT
        while self._peek().kind != "RBRACE":
            kw = self._expect("IDENT").text
            if kw == "reads":
                reads = tuple(self._rid(m, rid_of, s) for s in self._idlist())
            elif kw == "writes":
                writes = tuple(self._rid(m, rid_of, s) for s in self._idlist())
            elif kw == "count":
                count = int(self._expect("INT").text)
            elif kw == "lane":
                lane = _LANES[self._expect("IDENT").text.lower()]
            elif kw == "stride":
                stride = _STRIDES[self._expect("IDENT").text.lower()]
            else:
                raise ParseError(f"unknown statement {kw!r} at offset {self._peek().pos}")
            # Statements are ';'-separated; the separator before '}' is optional.
            if self._peek().kind == "SEMI":
                self._next()
            elif self._peek().kind != "RBRACE":
                t = self._peek()
                raise ParseError(f"expected ';' or '}}', got {t.kind} {t.text!r} at offset {t.pos}")
        self._expect("RBRACE")
        return Claim(
            id=claim_id,
            opcode=Opcode.ADD,
            lane=lane,
            stride_class=stride,
            count=count,
            rd=reads,
            wr=writes,
            op=f"rop.{name}",
        )


def parse_rop(text: str, name: str = "rop") -> Module:
    """Parse mini-ROP source text into a BCIR `Module` (text -> claims)."""
    tokens = Lexer().tokenize(text)
    return _Parser(tokens).parse_module(name=name)
