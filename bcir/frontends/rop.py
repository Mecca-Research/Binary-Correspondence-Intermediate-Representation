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
_MAX_SOURCE_CHARS = 1 << 20
_MAX_TOKENS = 1 << 18
_MAX_RESOURCES = 1 << 16
_MAX_CLAIMS = 1 << 16
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1


class _P:
    def __init__(self, toks: list[Token], resolver: dict[str, int]):
        self.t = toks
        self.i = 0
        self.resolver = dict(resolver)
        self.resource_names: set[str] = set()
        self.resource_ids: set[int] = set()
        self.phase_ids: set[int] = set()
        self.claim_count = 0

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
        self.expect("EOF")
        for pid, claims in claims_by_phase.items():
            m.add_phase(Phase(phase_id=pid, deps=phase_deps.get(pid, ()), claims=claims))
        return m

    def _resource(self, m: Module) -> None:
        name = self.expect("IDENT").text
        if name in self.resource_names:
            raise ParseError(f"duplicate resource name {name!r}")
        if len(self.resource_names) >= _MAX_RESOURCES:
            raise ParseError(f"resource count exceeds {_MAX_RESOURCES}")
        self.expect("LBRACE")
        rid: int | None = None
        domain = Domain.RAM
        count = 1
        seen: set[str] = set()
        while self.peek().kind != "RBRACE":
            f = self.expect("IDENT").text
            if f in seen:
                raise ParseError(f"resource {name!r} repeats field {f!r}")
            seen.add(f)
            if f == "rid":
                rid = int(self.expect("INT").text)
            elif f == "domain":
                spelling = self.expect("IDENT").text.lower()
                if spelling not in _DOMAINS:
                    raise ParseError(f"resource {name!r} has unknown domain {spelling!r}")
                domain = _DOMAINS[spelling]
            elif f == "count":
                count = int(self.expect("INT").text)
            else:
                raise ParseError(f"unknown resource field {f!r}")
        self.expect("RBRACE")
        if rid is None or not 1 <= rid <= _MAX_U32:
            raise ParseError(f"resource {name!r} requires rid in [1, 2^32-1]")
        if rid in self.resource_ids:
            raise ParseError(f"duplicate resource rid {rid}")
        if not 1 <= count <= _MAX_U64:
            raise ParseError(f"resource {name!r} count must be in [1, 2^64-1]")
        if self.resolver.get(name) != rid:
            raise ParseError(f"resource pre-scan disagrees for {name!r}")
        self.resource_names.add(name)
        self.resource_ids.add(rid)
        m.add_resource(Resource(rid=rid, domain=domain, shape=(count,), name=name))

    def _phase(self, cid_start: int) -> tuple[int, tuple[int, ...], list[Claim]]:
        pid = int(self.expect("INT").text)
        if not 0 <= pid <= _MAX_U32:
            raise ParseError("phase id must fit uint32")
        if pid in self.phase_ids:
            raise ParseError(f"duplicate phase id {pid}")
        self.phase_ids.add(pid)
        self.expect("LBRACE")
        claims: list[Claim] = []
        cid = cid_start
        while self.peek().kind != "RBRACE":
            self.expect("IDENT", "claim")
            if self.claim_count >= _MAX_CLAIMS:
                raise ParseError(f"claim count exceeds {_MAX_CLAIMS}")
            claims.append(self._claim(cid))
            cid += 1
            self.claim_count += 1
        self.expect("RBRACE")
        return pid, (), claims

    def _claim(self, cid: int) -> Claim:
        name = self.expect("IDENT").text
        self.expect("LBRACE")
        op = Opcode.ADD
        reads: list[str] = []
        writes: list[str] = []
        count = 1
        lane = Lane.U
        stride = StrideClass.UNIT
        seen: set[str] = set()

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
            if f in seen:
                raise ParseError(f"claim {name!r} repeats field {f!r}")
            seen.add(f)
            if f == "op":
                spelling = self.expect("IDENT").text.lower()
                if spelling not in _OPCODES:
                    raise ParseError(f"claim {name!r} has unknown opcode {spelling!r}")
                op = _OPCODES[spelling]
            elif f == "reads":
                reads = idlist()
            elif f == "writes":
                writes = idlist()
            elif f == "count":
                count = int(self.expect("INT").text)
            elif f == "lane":
                spelling = self.expect("IDENT").text.lower()
                if spelling not in _LANES:
                    raise ParseError(f"claim {name!r} has unknown lane {spelling!r}")
                lane = _LANES[spelling]
            elif f == "stride":
                spelling = self.expect("IDENT").text.lower()
                if spelling not in _STRIDES:
                    raise ParseError(f"claim {name!r} has unknown stride {spelling!r}")
                stride = _STRIDES[spelling]
            else:
                raise ParseError(f"unknown claim field {f!r}")
        self.expect("RBRACE")

        if not 1 <= count <= _MAX_U64:
            raise ParseError(f"claim {name!r} count must be in [1, 2^64-1]")

        def resolve(resource: str) -> int:
            if resource not in self.resolver:
                raise ParseError(f"claim {name!r} names unknown resource {resource!r}")
            return self.resolver[resource]

        # resolve resource names -> rids via the module is done by caller context;
        # here we trust earlier `resource` decls registered names. The caller passes
        # a name->rid resolver implicitly through the module; we look it up at build.
        return Claim(
            id=cid,
            opcode=op,
            lane=lane,
            stride_class=stride,
            count=count,
            rd=tuple(resolve(n) for n in reads),
            wr=tuple(resolve(n) for n in writes),
            op=f"rop.{name}",
        )


def parse_rop_program(text: str) -> Module:
    """Parse a declarative ROP program into a verified-shape BCIR `Module`."""
    if not isinstance(text, str) or len(text) > _MAX_SOURCE_CHARS:
        raise ParseError(f"ROP source must be text of at most {_MAX_SOURCE_CHARS} characters")
    toks = Lexer().tokenize(text)
    if len(toks) > _MAX_TOKENS:
        raise ParseError(f"ROP token count exceeds {_MAX_TOKENS}")
    # Pre-scan resource decls to build the name->rid map before claims resolve.
    resolver: dict[str, int] = {}
    rid_names: dict[int, str] = {}
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
                rid = int(toks[j + 1].text)
                if name in resolver:
                    raise ParseError(f"duplicate resource name {name!r}")
                if rid in rid_names:
                    raise ParseError(
                        f"duplicate resource rid {rid} for {name!r} and {rid_names[rid]!r}"
                    )
                resolver[name] = rid
                rid_names[rid] = name
        i += 1
    return _P(toks, resolver).parse()
