"""MAP front-end: a terse macro-assembly form -> a BCIR Module.

One declaration or operation per line (the assembly-performance paradigm):

    ; resources
    res A rid 10 n 1024
    res B rid 11 n 1024
    res C rid 12 n 1024 domain hbm
    ; operations: OP DST <- SRC[, SRC...] [n N] [lane L] [stride S] [hazard H]
    add C <- A, B n 1024 lane u stride unit

Lines starting with ';' or '#' are comments. All operations land in phase 0.

A claim's domain contract is DERIVED from the resources it touches (`model.derived_claim_domain`,
LangRef R3): an isolated domain (mmio) it touches is its domain, else its destination's.
A device-register access is therefore volatile, and needs `hazard atomic|barriered` -- the
verifier's R3/R5 say so when it is missing, the frontend never invents one.
"""

from __future__ import annotations

from ..model import (
    ISOLATED_DOMAINS,
    Claim,
    Domain,
    Lane,
    Module,
    Opcode,
    Phase,
    Resource,
    StrideClass,
    derived_claim_domain,
)

_DOMAINS = {d.name.lower(): d for d in Domain}
_LANES = {l.name.lower(): l for l in Lane}
_STRIDES = {s.name.lower(): s for s in StrideClass}
_OPCODES = {o.name.lower(): o for o in Opcode}
_HAZARDS = ("unique", "atomic", "barriered")
_CLAIM_KEYS = ("n", "lane", "stride", "hazard")
_MAX_SOURCE_CHARS = 1 << 20
_MAX_LINES = 1 << 16
_MAX_RESOURCES = 1 << 16
_MAX_CLAIMS = 1 << 16
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1


class MapError(Exception):
    pass


def parse_map(text: str) -> Module:
    """Parse a MAP macro-assembly program into a BCIR `Module`."""
    if not isinstance(text, str) or len(text) > _MAX_SOURCE_CHARS:
        raise MapError(f"MAP source must be text of at most {_MAX_SOURCE_CHARS} characters")
    lines = text.splitlines()
    if len(lines) > _MAX_LINES:
        raise MapError(f"MAP source exceeds {_MAX_LINES} lines")
    m = Module(name="map")
    rid_of: dict[str, int] = {}
    names_by_rid: dict[int, str] = {}
    claims: list[Claim] = []
    cid = 1000

    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line[0] in ";#":
            continue
        tok = line.split()
        head = tok[0].lower()

        if head == "res":
            if len(tok) < 2:
                raise MapError(f"line {lineno}: resource name is required")
            name = tok[1]
            if name in rid_of:
                raise MapError(f"line {lineno}: duplicate resource name {name!r}")
            if len(rid_of) >= _MAX_RESOURCES:
                raise MapError(f"line {lineno}: resource count exceeds {_MAX_RESOURCES}")
            kv = _kv(tok[2:], {"rid", "n", "domain"}, lineno)
            rid = _integer(kv.get("rid", str(len(rid_of) + 10)), "rid", lineno)
            n = _integer(kv.get("n", "1"), "n", lineno)
            domain_name = kv.get("domain", "ram").lower()
            if not 1 <= rid <= _MAX_U32:
                raise MapError(f"line {lineno}: rid must be in [1, 2^32-1]")
            if rid in names_by_rid:
                raise MapError(
                    f"line {lineno}: duplicate rid {rid} for {name!r} and {names_by_rid[rid]!r}"
                )
            if not 1 <= n <= _MAX_U64:
                raise MapError(f"line {lineno}: n must be in [1, 2^64-1]")
            if domain_name not in _DOMAINS:
                raise MapError(f"line {lineno}: unknown domain {domain_name!r}")
            domain = _DOMAINS[domain_name]
            rid_of[name] = rid
            names_by_rid[rid] = name
            m.add_resource(Resource(rid=rid, domain=domain, shape=(n,), name=name))
            continue

        # operation line: OP DST <- SRC[, SRC...] [n N] [lane L] [stride S] [hazard H]
        if head not in _OPCODES:
            raise MapError(f"line {lineno}: unknown opcode {tok[0]!r}")
        if len(tok) < 4 or tok[2] != "<-":
            raise MapError(f"line {lineno}: expected 'OP DST <- SRC...' ")
        op = _OPCODES[head]
        if len(claims) >= _MAX_CLAIMS:
            raise MapError(f"line {lineno}: claim count exceeds {_MAX_CLAIMS}")
        dst = tok[1]
        # operands run after '<-' until a field keyword (n/lane/stride).
        srcs: list[str] = []
        rest = tok[3:]
        fields_start = len(rest)
        for k, w in enumerate(rest):
            if w.lower() in _CLAIM_KEYS:
                fields_start = k
                break
            srcs.append(w.strip(","))
        if not srcs or any(not source for source in srcs):
            raise MapError(f"line {lineno}: at least one source is required")
        kv = _kv(rest[fields_start:], set(_CLAIM_KEYS), lineno)
        count = _integer(kv.get("n", "1"), "n", lineno)
        lane_name = kv.get("lane", "u").lower()
        stride_name = kv.get("stride", "unit").lower()
        hazard = kv.get("hazard", "unique").lower()
        if hazard not in _HAZARDS:
            raise MapError(f"line {lineno}: unknown hazard {hazard!r}")
        if not 1 <= count <= _MAX_U64:
            raise MapError(f"line {lineno}: n must be in [1, 2^64-1]")
        if lane_name not in _LANES:
            raise MapError(f"line {lineno}: unknown lane {lane_name!r}")
        if stride_name not in _STRIDES:
            raise MapError(f"line {lineno}: unknown stride {stride_name!r}")
        lane = _LANES[lane_name]
        stride = _STRIDES[stride_name]

        unknown = [source for source in srcs if source not in rid_of]
        if unknown:
            raise MapError(f"line {lineno}: undeclared source {unknown[0]!r}")
        rd = tuple(rid_of[s] for s in srcs)
        if dst not in rid_of:
            raise MapError(f"line {lineno}: undeclared destination {dst!r}")
        try:
            domain = derived_claim_domain([m.resource(rid_of[dst])], [m.resource(r) for r in rd])
        except ValueError as exc:
            raise MapError(f"line {lineno}: {exc}") from exc
        claims.append(
            Claim(
                id=cid,
                opcode=op,
                lane=lane,
                stride_class=stride,
                count=count,
                rd=rd,
                wr=(rid_of[dst],),
                hazard=hazard,
                domain=domain,
                volatile=domain in ISOLATED_DOMAINS,
                op=f"map.{head}",
            )
        )
        cid += 1

    if claims:
        m.add_phase(Phase(phase_id=0, deps=(), claims=claims))
    return m


def _kv(tokens: list[str], allowed: set[str], lineno: int) -> dict[str, str]:
    """Fold ['rid','10','n','1024'] -> {'rid':'10','n':'1024'} (key value pairs)."""
    if len(tokens) % 2:
        raise MapError(f"line {lineno}: metadata must be key/value pairs")
    out: dict[str, str] = {}
    for i in range(0, len(tokens), 2):
        key = tokens[i].lower()
        if key not in allowed:
            raise MapError(f"line {lineno}: unknown metadata field {key!r}")
        if key in out:
            raise MapError(f"line {lineno}: duplicate metadata field {key!r}")
        out[key] = tokens[i + 1]
    return out


def _integer(text: str, field: str, lineno: int) -> int:
    try:
        return int(text, 10)
    except (TypeError, ValueError) as exc:
        raise MapError(f"line {lineno}: {field} must be a decimal integer") from exc
