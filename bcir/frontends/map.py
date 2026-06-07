"""MAP front-end: a terse macro-assembly form -> a BCIR Module.

One declaration or operation per line (the assembly-performance paradigm):

    ; resources
    res A rid 10 n 1024
    res B rid 11 n 1024
    res C rid 12 n 1024 domain hbm
    ; operations: OP DST <- SRC[, SRC...] [n N] [lane L] [stride S]
    add C <- A, B n 1024 lane u stride unit

Lines starting with ';' or '#' are comments. All operations land in phase 0.
"""

from __future__ import annotations

from ..model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

_DOMAINS = {d.name.lower(): d for d in Domain}
_LANES = {l.name.lower(): l for l in Lane}
_STRIDES = {s.name.lower(): s for s in StrideClass}
_OPCODES = {o.name.lower(): o for o in Opcode}


class MapError(Exception):
    pass


def parse_map(text: str) -> Module:
    """Parse a MAP macro-assembly program into a BCIR `Module`."""
    m = Module(name="map")
    rid_of: dict[str, int] = {}
    claims: list[Claim] = []
    cid = 1000

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line[0] in ";#":
            continue
        tok = line.split()
        head = tok[0].lower()

        if head == "res":
            name = tok[1]
            kv = _kv(tok[2:])
            rid = int(kv.get("rid", len(rid_of) + 10))
            n = int(kv.get("n", 1))
            domain = _DOMAINS.get(kv.get("domain", "ram"), Domain.RAM)
            rid_of[name] = rid
            m.add_resource(Resource(rid=rid, domain=domain, shape=(n,), name=name))
            continue

        # operation line: OP DST <- SRC[, SRC...] [n N] [lane L] [stride S]
        if head not in _OPCODES:
            raise MapError(f"line {lineno}: unknown opcode {tok[0]!r}")
        if len(tok) < 4 or tok[2] != "<-":
            raise MapError(f"line {lineno}: expected 'OP DST <- SRC...' ")
        op = _OPCODES[head]
        dst = tok[1]
        # operands run after '<-' until a field keyword (n/lane/stride).
        srcs: list[str] = []
        rest = tok[3:]
        fields_start = len(rest)
        for k, w in enumerate(rest):
            if w.lower() in ("n", "lane", "stride"):
                fields_start = k
                break
            srcs.append(w.strip(","))
        kv = _kv(rest[fields_start:])
        count = int(kv.get("n", 1))
        lane = _LANES.get(kv.get("lane", "u"), Lane.U)
        stride = _STRIDES.get(kv.get("stride", "unit"), StrideClass.UNIT)

        rd = tuple(rid_of[s] for s in srcs if s)
        if dst not in rid_of:
            raise MapError(f"line {lineno}: undeclared destination {dst!r}")
        claims.append(Claim(id=cid, opcode=op, lane=lane, stride_class=stride, count=count,
                            rd=rd, wr=(rid_of[dst],), op=f"map.{head}"))
        cid += 1

    if claims:
        m.add_phase(Phase(phase_id=0, deps=(), claims=claims))
    return m


def _kv(tokens: list[str]) -> dict[str, str]:
    """Fold ['rid','10','n','1024'] -> {'rid':'10','n':'1024'} (key value pairs)."""
    out: dict[str, str] = {}
    i = 0
    while i + 1 < len(tokens) + 1 and i < len(tokens):
        key = tokens[i].lower()
        if i + 1 < len(tokens):
            out[key] = tokens[i + 1]
            i += 2
        else:
            i += 1
    return out
