"""Lower the C AST (L1–L4) to the BCIR claim graph — the *same* `Resource`/`Claim`/`Phase` model the
oracle reasons over, so R1–R18 + the K_BCIR cost model apply unchanged (the dual-rail invariant).

Mapping:
  * a C **function** -> a single-phase `Module` (its straight-line claim graph) + a `compose.Function`
    (its region, for the inter-procedural call graph / R18);
  * **scalar variables** (params, locals, temporaries) -> scalar `Resource`s (one per SSA value);
  * **integer expressions** (L1) -> `ADD`/`SUB`/`MUL`-cost-class `Claim`s carrying the exact C
    operator in `op` (so the emitter reproduces the semantics) and constants in `imm`;
  * **struct/union member access** (L2) -> a `LOAD`/`STORE` claim at the member's byte `offset`;
  * **pointer/array indexing** (L3) -> a `LOAD`/`STORE` claim over the base resource (GEP-equivalent);
  * **calls** (L4) -> a `GEM_DISPATCH` claim + a `compose.Call`, so `plan_composite` enforces R18
    (callee resolution + no recursion).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...kbcir import compose
from ...model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from . import cast
from .ctype_model import AggregateBuilder, CType, array, is_scalar_name, pointer, scalar

# C operator -> (cost-class Opcode, op-suffix the emitter maps back to a C operator).
_BIN = {
    "+": (Opcode.ADD, "add"), "-": (Opcode.SUB, "sub"), "*": (Opcode.MUL, "mul"),
    "/": (Opcode.MUL, "div"), "%": (Opcode.MUL, "mod"),
    "&": (Opcode.ADD, "and"), "|": (Opcode.ADD, "or"), "^": (Opcode.ADD, "xor"),
    "<<": (Opcode.ADD, "shl"), ">>": (Opcode.ADD, "shr"),
    "==": (Opcode.SUB, "eq"), "!=": (Opcode.SUB, "ne"), "<": (Opcode.SUB, "lt"),
    ">": (Opcode.SUB, "gt"), "<=": (Opcode.SUB, "le"), ">=": (Opcode.SUB, "ge"),
    "&&": (Opcode.ADD, "land"), "||": (Opcode.ADD, "lor"),
}
_UN = {"-": (Opcode.SUB, "neg"), "~": (Opcode.ADD, "bnot"), "!": (Opcode.SUB, "lnot")}


class CLowerError(Exception):
    pass


@dataclass
class LoweredFunc:
    """One C function lowered: its claim-graph `Module`, the value/SSA bookkeeping the emitter needs,
    and the call sites (for the call graph / R18)."""
    name: str
    module: Module
    ret_type: CType
    params: list                          # (name, rid, CType) in C order
    return_rid: int | None
    claims: list                          # ordered claims (== the single phase), for emission
    resources: dict                       # rid -> Resource
    rid_types: dict = field(default_factory=dict)   # rid -> CType (for faithful C emission)
    calls: list = field(default_factory=list)   # (callee, (actual_rids...))
    region: object = None                 # compose.Region


@dataclass
class LoweredUnit:
    functions: dict                       # name -> LoweredFunc
    entry: str
    aggregates: dict                      # tag -> CType
    compose_functions: dict               # name -> compose.Function
    resources: dict                       # rid -> Resource (whole unit)


class _FuncLowerer:
    def __init__(self, func: cast.Func, aggregates: dict, base_rid: int, cid: list):
        self.func = func
        self.aggregates = aggregates
        self.rid = base_rid
        self.cid = cid                    # shared mutable [next_claim_id]
        self.env: dict[str, tuple[int, CType]] = {}    # name -> (current rid, type)
        self.resources: dict[int, Resource] = {}
        self.rtypes: dict[int, CType] = {}             # rid -> CType (for emission)
        self.claims: list[Claim] = []
        self.params: list = []
        self.calls: list = []

    # --- resource allocation ---
    def _new_rid(self) -> int:
        self.rid += 1
        return self.rid

    def _resolve_type(self, tref: cast.TypeRef) -> CType:
        if tref.aggregate:
            base = self.aggregates[tref.base]
        elif is_scalar_name(tref.base):
            base = scalar(tref.base)
        else:
            raise CLowerError(f"unknown type {tref.base!r}")
        t = base
        for _ in range(tref.ptr):
            t = pointer(t)
        for dim in reversed(tref.array):
            t = array(t, dim)
        return t

    def _resource(self, rid: int, ct: CType, name: str, *, mmio: bool = False) -> None:
        if ct.kind == "array":
            shape, elem = (ct.count or 1,), ct.of.size
        elif ct.kind == "pointer":
            shape, elem = (1 << 16,), ct.of.size if ct.of else 1     # symbolic pointee extent
        elif ct.is_aggregate:
            shape, elem = (1,), max(1, ct.size)
        else:
            shape, elem = (1,), max(1, ct.size)
        self.resources[rid] = Resource(
            rid=rid, domain=Domain.MMIO if mmio else Domain.RAM, elem_bytes=elem, shape=shape,
            access="volatile" if mmio else "rw", name=name)
        self.rtypes[rid] = ct

    def _emit(self, op: str, opcode: Opcode, rd: tuple, wr: tuple, *, imm: tuple = (),
              lane=Lane.U, stride=StrideClass.SCALAR, count: int = 1, domain=Domain.RAM,
              bounds: str = "strict") -> int:
        self.cid[0] += 1
        self.claims.append(Claim(id=self.cid[0], opcode=opcode, lane=lane, stride_class=stride,
                                 count=count, rd=tuple(rd), wr=tuple(wr), imm=tuple(imm),
                                 domain=domain, op=op, bounds=bounds))
        return wr[0] if wr else -1

    def _temp(self, ct: CType, name: str) -> int:
        rid = self._new_rid()
        self._resource(rid, ct, name)
        return rid

    # --- lvalue resolution (returns (base_rid, index_rid|None, offset, elem_type, mmio)) ---
    def _lvalue(self, node):
        if isinstance(node, cast.Name):
            rid, ct = self.env[node.ident]
            return ("var", rid, None, 0, ct, False)
        if isinstance(node, cast.Index):
            base_rid, base_ct = self._addr(node.base)
            idx = self._rvalue(node.index)
            elem = base_ct.of if base_ct.of else scalar("uint32_t")
            return ("mem", base_rid, idx, 0, elem, base_ct.kind == "pointer" and False)
        if isinstance(node, cast.Member):
            if node.arrow:
                base_rid, base_ct = self._addr(node.base)
                agg = base_ct.of
            else:
                base_rid, agg = self._addr(node.base)
                base_ct = agg
            ftype, off = agg.field(node.field)
            return ("mem", base_rid, None, off, ftype, False)
        if isinstance(node, cast.Unary) and node.op == "*":
            base_rid, base_ct = self._addr(node.operand)
            return ("mem", base_rid, None, 0, base_ct.of or scalar("uint32_t"), False)
        raise CLowerError(f"not an lvalue: {type(node).__name__}")

    def _addr(self, node):
        """The (rid, type) of an aggregate/pointer base used by member/index access."""
        if isinstance(node, cast.Name):
            return self.env[node.ident]
        if isinstance(node, cast.Member):
            base_rid, agg = (self._addr(node.base) if not node.arrow else self._addr(node.base))
            ftype, _ = (agg.of if node.arrow else agg).field(node.field)
            return base_rid, ftype
        raise CLowerError(f"unsupported base expression {type(node).__name__}")

    # --- rvalue lowering: returns the rid holding the value ---
    def _rvalue(self, node) -> int:
        if isinstance(node, cast.IntLit):
            t = self._temp(scalar("uint32_t"), f"k{node.value}")
            return self._emit("c.const", Opcode.LOAD, (), (t,), imm=(node.value,))
        if isinstance(node, cast.Name):
            return self.env[node.ident][0]
        if isinstance(node, cast.Binary):
            a, b = self._rvalue(node.lhs), self._rvalue(node.rhs)
            opcode, suf = _BIN[node.op]
            t = self._temp(scalar("uint32_t"), f"b_{suf}")
            return self._emit(f"c.bin.{suf}", opcode, (a, b), (t,))
        if isinstance(node, cast.Unary):
            if node.op == "*":
                kind, base, idx, off, et, _ = self._lvalue(node)
                return self._load(base, idx, off, et)
            if node.op == "&":
                return self._addr(node.operand)[0]
            v = self._rvalue(node.operand)
            opcode, suf = _UN[node.op]
            t = self._temp(scalar("uint32_t"), f"u_{suf}")
            return self._emit(f"c.un.{suf}", opcode, (v,), (t,))
        if isinstance(node, (cast.Index, cast.Member)):
            _kind, base, idx, off, et, _m = self._lvalue(node)
            return self._load(base, idx, off, et)
        if isinstance(node, cast.Assign):
            return self._assign(node)
        if isinstance(node, cast.CallExpr):
            return self._call(node)
        raise CLowerError(f"cannot lower expression {type(node).__name__}")

    def _load(self, base_rid: int, idx_rid, offset: int, et: CType) -> int:
        t = self._temp(et, "ld")
        rd = (base_rid,) if idx_rid is None else (base_rid, idx_rid)
        dom = self.resources[base_rid].domain
        # the field/index byte offset rides in imm (not the strict-bounds `offset` field), and the
        # access is `assumed_safe` — the frontend resolved the member/index statically, so R7's
        # affine extent check (element units) does not apply to this memory access.
        return self._emit("c.load", Opcode.LOAD, rd, (t,), imm=(offset,) if offset else (),
                          domain=dom, bounds="assumed_safe")

    def _assign(self, node: cast.Assign) -> int:
        v = self._rvalue(node.value)
        if isinstance(node.target, cast.Name) and node.target.ident in self.env:
            ct = self.env[node.target.ident][1]
            self.env[node.target.ident] = (v, ct)            # SSA rebind
            return v
        kind, base, idx, off, et, _m = self._lvalue(node.target)
        rd = (base, v) if idx is None else (base, idx, v)
        self._emit("c.store", Opcode.STORE, rd, (), imm=(off,) if off else (),
                   domain=self.resources[base].domain, bounds="assumed_safe")
        return v

    def _call(self, node: cast.CallExpr) -> int:
        actuals = tuple(self._rvalue(a) for a in node.args)
        t = self._temp(scalar("uint32_t"), f"call_{node.callee}")
        self.calls.append((node.callee, actuals))
        return self._emit(f"c.call:{node.callee}", Opcode.GEM_DISPATCH, actuals, (t,))

    # --- statements ---
    def _stmt(self, st):
        if isinstance(st, cast.Decl):
            ct = self._resolve_type(st.type)
            if st.init is not None:
                v = self._rvalue(st.init)
                self.env[st.name] = (v, ct)
            else:
                rid = self._temp(ct, st.name)
                self.env[st.name] = (rid, ct)
        elif isinstance(st, cast.ExprStmt):
            self._rvalue(st.expr)
        elif isinstance(st, cast.Return):
            return None if st.value is None else self._rvalue(st.value)
        else:
            raise CLowerError(f"statement {type(st).__name__} is beyond the L1–L4 subset")
        return None

    def lower(self) -> LoweredFunc:
        for p in self.func.params:                            # params -> input resources
            ct = self._resolve_type(p.type)
            rid = self._new_rid()
            self._resource(rid, ct, p.name)
            self.env[p.name] = (rid, ct)
            self.params.append((p.name, rid, ct))
        return_rid = None
        for st in self.func.body:
            r = self._stmt(st)
            if isinstance(st, cast.Return):
                return_rid = r
        m = Module(name=self.func.name)
        for rid in sorted(self.resources):
            m.add_resource(self.resources[rid])
        m.add_phase(Phase(phase_id=0, deps=(), claims=list(self.claims)))
        ret_ct = self._resolve_type(self.func.ret)
        return LoweredFunc(name=self.func.name, module=m, ret_type=ret_ct, params=self.params,
                           return_rid=return_rid, claims=list(self.claims),
                           resources=dict(self.resources), rid_types=dict(self.rtypes),
                           calls=list(self.calls), region=None)


def _region_for(lf: LoweredFunc, functions: dict) -> compose.Region:
    """The function's compose region: its non-call claims as a `Leaf`, each call as a `compose.Call`
    whose arg_map binds the *callee's actual param rids* to the caller's actuals — exactly what
    `plan_composite` substitutes (so R18's callee-resolution + recursion checks see the real graph).
    An undefined callee is left unmapped so `plan_composite` raises (an R18 violation)."""
    leaf = compose.Leaf(tuple(c for c in lf.claims if not c.op.startswith("c.call:")))
    parts: list = [leaf]
    for callee, actuals in lf.calls:
        callee_fn = functions.get(callee)
        formals = [rid for _n, rid, _ct in callee_fn.params] if callee_fn else []
        arg_map = tuple(zip(formals, actuals))
        parts.append(compose.Call(callee, arg_map))
    return compose.Seq(tuple(parts)) if len(parts) > 1 else leaf


def lower_unit(unit: cast.Unit) -> LoweredUnit:
    """Lower a whole translation unit. Aggregates are laid out first (Clang-compatible), then each
    function to its claim-graph Module + compose.Function."""
    aggregates: dict[str, CType] = {}
    for tag, agg in unit.aggregates.items():
        b = AggregateBuilder(agg.kind, tag)
        for tref, mname in agg.members:
            b.members.append((mname, _resolve_member_type(tref, aggregates)))
        aggregates[tag] = b.build()

    functions: dict[str, LoweredFunc] = {}
    compose_functions: dict[str, compose.Function] = {}
    resources: dict[int, Resource] = {}
    cid = [1000]
    for idx, fn in enumerate(unit.funcs):
        lf = _FuncLowerer(fn, aggregates, base_rid=100 + idx * 1000, cid=cid).lower()
        functions[fn.name] = lf
        resources.update(lf.resources)
    for lf in functions.values():                          # regions need every callee's param rids
        lf.region = _region_for(lf, functions)
        compose_functions[lf.name] = compose.Function(lf.name, lf.region)
    entry = unit.funcs[-1].name if unit.funcs else ""
    return LoweredUnit(functions=functions, entry=entry, aggregates=aggregates,
                       compose_functions=compose_functions, resources=resources)


def _resolve_member_type(tref: cast.TypeRef, aggregates: dict) -> CType:
    if tref.aggregate:
        base = aggregates[tref.base]
    else:
        base = scalar(tref.base)
    t = base
    for _ in range(tref.ptr):
        t = pointer(t)
    for dim in reversed(tref.array):
        t = array(t, dim)
    return t
