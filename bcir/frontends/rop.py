"""ROP front-end: a registry-first declarative form -> a BCIR Module.

Grammar (brace-delimited, keyword-driven; whitespace-insensitive):

    module NAME {
      resource A { rid 10 domain ram count 1024 }
      resource C { rid 12 domain hbm count 1024 }
      phase 0 {
        claim add { op add reads A B writes C count 1024 lane u stride unit }
      }
    }

Reuses the ETL lexer; lists (reads/writes) run until the next field keyword.
"""

from __future__ import annotations

from ..etl.parse import Lexer, ParseError, Token
from ..model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

_DOMAINS = {d.name.lower(): d for d in Domain}
_LANES = {l.name.lower(): l for l in Lane}
_STRIDES = {s.name.lower(): s for s in StrideClass}
_OPCODES = {o.name.lower(): o for o in Opcode}
_CLAIM_FIELDS = {"op", "reads", "writes", "count", "lane", "stride"}


class _P:
    def __init__(self, toks: list[Token]):
        self.t = toks
        self.i = 0

    def peek(self) -> Token:
        return self.t[self.i]

    def nxt(self) -> Token:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, kind: str, text: str | None = None) -> Token:
        tok = self.peek()
        if tok.kind != kind or (text is not None and tok.text != text):
            raise ParseError(f"expected {text or kind!r}, got {tok.kind} {tok.text!r} at {tok.pos}")
        return self.nxt()

    def parse(self) -> Module:
        self.expect("IDENT", "module")
        name = self.expect("IDENT").text
        m = Module(name=name)
        self.expect("LBRACE")
        claims_by_phase: dict[int, list[Claim]] = {}
        phase_deps: dict[int, tuple[int, ...]] = {}
        cid = 1000
        while self.peek().kind != "RBRACE":
            kw = self.expect("IDENT").text
            if kw == "resource":
                self._resource(m)
            elif kw == "phase":
                pid, deps, claims = self._phase(cid)
                phase_deps[pid] = deps
                claims_by_phase.setdefault(pid, []).extend(claims)
                cid += len(claims)
            else:
                raise ParseError(f"unknown top-level keyword {kw!r}")
        self.expect("RBRACE")
        for pid, claims in claims_by_phase.items():
            m.add_phase(Phase(phase_id=pid, deps=phase_deps.get(pid, ()), claims=claims))
        return m

    def _resource(self, m: Module) -> None:
        name = self.expect("IDENT").text
        self.expect("LBRACE")
        rid = 0
        domain = Domain.RAM
        count = 1
        while self.peek().kind != "RBRACE":
            f = self.expect("IDENT").text
            if f == "rid":
                rid = int(self.expect("INT").text)
            elif f == "domain":
                domain = _DOMAINS[self.expect("IDENT").text.lower()]
            elif f == "count":
                count = int(self.expect("INT").text)
            else:
                raise ParseError(f"unknown resource field {f!r}")
        self.expect("RBRACE")
        m.add_resource(Resource(rid=rid, domain=domain, shape=(count,), name=name))

    def _phase(self, cid_start: int) -> tuple[int, tuple[int, ...], list[Claim]]:
        pid = int(self.expect("INT").text)
        self.expect("LBRACE")
        claims: list[Claim] = []
        cid = cid_start
        while self.peek().kind != "RBRACE":
            self.expect("IDENT", "claim")
            claims.append(self._claim(cid))
            cid += 1
        self.expect("RBRACE")
        return pid, (), claims

    def _claim(self, cid: int) -> Claim:
        name = self.expect("IDENT").text
        self.expect("LBRACE")
        names: dict[str, int] = {}  # not used for rids here; reads/writes are RIDs by name lookup

        op = Opcode.ADD
        reads: list[str] = []
        writes: list[str] = []
        count = 1
        lane = Lane.U
        stride = StrideClass.UNIT

        def idlist() -> list[str]:
            out: list[str] = []
            while self.peek().kind in ("IDENT", "COMMA"):
                if self.peek().kind == "COMMA":
                    self.nxt()
                    continue
                if self.peek().text in _CLAIM_FIELDS:
                    break
                out.append(self.nxt().text)
            return out

        while self.peek().kind != "RBRACE":
            f = self.expect("IDENT").text
            if f == "op":
                op = _OPCODES.get(self.expect("IDENT").text.lower(), Opcode.ADD)
            elif f == "reads":
                reads = idlist()
            elif f == "writes":
                writes = idlist()
            elif f == "count":
                count = int(self.expect("INT").text)
            elif f == "lane":
                lane = _LANES[self.expect("IDENT").text.lower()]
            elif f == "stride":
                stride = _STRIDES[self.expect("IDENT").text.lower()]
            else:
                raise ParseError(f"unknown claim field {f!r}")
        self.expect("RBRACE")

        # resolve resource names -> rids via the module is done by caller context;
        # here we trust earlier `resource` decls registered names. The caller passes
        # a name->rid resolver implicitly through the module; we look it up at build.
        return Claim(id=cid, opcode=op, lane=lane, stride_class=stride, count=count,
                     rd=tuple(_RESOLVER.get(n, 0) for n in reads),
                     wr=tuple(_RESOLVER.get(n, 0) for n in writes),
                     op=f"rop.{name}")


# Name->rid resolver populated per-parse (single-threaded oracle).
_RESOLVER: dict[str, int] = {}


def parse_rop_program(text: str) -> Module:
    """Parse a declarative ROP program into a verified-shape BCIR `Module`."""
    global _RESOLVER
    toks = Lexer().tokenize(text)
    # Pre-scan resource decls to build the name->rid map before claims resolve.
    _RESOLVER = {}
    i = 0
    while i < len(toks) - 1:
        if toks[i].kind == "IDENT" and toks[i].text == "resource" and toks[i + 1].kind == "IDENT":
            name = toks[i + 1].text
            # find rid token
            j = i + 2
            while j < len(toks) and not (toks[j].kind == "IDENT" and toks[j].text == "rid"):
                if toks[j].kind == "RBRACE":
                    break
                j += 1
            if j + 1 < len(toks) and toks[j].text == "rid" and toks[j + 1].kind == "INT":
                _RESOLVER[name] = int(toks[j + 1].text)
        i += 1
    return _P(toks).parse()
