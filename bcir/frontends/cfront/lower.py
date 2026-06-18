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
from .ctype_model import (
    AggregateBuilder,
    CType,
    array,
    is_scalar_name,
    pointer,
    scalar,
    with_volatile,
)

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
class _LV:
    """A resolved lvalue: a scalar variable, or a memory access (member/index/deref) that may be a
    bitfield (`bit_width > 0`) and/or MMIO (decided from the base resource's domain)."""
    kind: str                              # "var" | "mem"
    rid: int                               # variable rid OR base rid
    ct: CType                              # variable type OR accessed element/field type
    idx: int | None = None                 # index rid for base[idx]
    byte_off: int = 0                      # member byte offset (in imm, not the strict-bounds field)
    bit_off: int = 0                       # bitfield bit offset within its storage unit
    bit_width: int = 0                     # bitfield width (0 == plain access)


# --- the structured body tree (L6): a block is a list mixing straight-line Claims with these. ---
@dataclass
class IfNode:
    cond: int                              # rid of the condition value
    then: list                             # block (list of Claim | IfNode | WhileNode)
    els: list


@dataclass
class WhileNode:
    cond_block: list                       # claims that recompute the condition each iteration
    cond: int                              # rid of the condition value
    body: list
    bound: int = 1024                      # static iteration upper bound (for the cost model)


@dataclass
class ReturnNode:
    rid: int | None                        # the returned value's rid (None == `return;`)


def _flatten_block(block: list) -> list:
    """All Claim objects in a body tree, in order (the flat single-phase view verify/plan use)."""
    out: list = []
    for node in block:
        if isinstance(node, IfNode):
            out += _flatten_block(node.then) + _flatten_block(node.els)
        elif isinstance(node, WhileNode):
            out += _flatten_block(node.cond_block) + _flatten_block(node.body)
        elif isinstance(node, (Claim,)):
            out.append(node)
    return out


@dataclass
class LoweredFunc:
    """One C function lowered: its claim-graph `Module`, the value/SSA bookkeeping the emitter needs,
    and the call sites (for the call graph / R18)."""
    name: str
    module: Module
    ret_type: CType
    params: list                          # (name, rid, CType) in C order
    return_rid: int | None
    claims: list                          # all claims, flat (== the single phase) for verify/plan
    resources: dict                       # rid -> Resource
    rid_types: dict = field(default_factory=dict)   # rid -> CType (for faithful C emission)
    calls: list = field(default_factory=list)   # (callee, (actual_rids...))
    region: object = None                 # compose.Region
    body: list = field(default_factory=list)        # the structured body tree (for emission)
    locals: list = field(default_factory=list)      # (rid, name, CType) mutable named locals
    globals_used: dict = field(default_factory=dict)   # rid -> name (file-scope globals referenced)


@dataclass
class LoweredUnit:
    functions: dict                       # name -> LoweredFunc
    entry: str
    aggregates: dict                      # tag -> CType
    compose_functions: dict               # name -> compose.Function
    resources: dict                       # rid -> Resource (whole unit)


class _FuncLowerer:
    def __init__(self, func: cast.Func, aggregates: dict, base_rid: int, cid: list,
                 genv: dict | None = None, gres: dict | None = None):
        self.func = func
        self.aggregates = aggregates
        self.rid = base_rid
        self.cid = cid                    # shared mutable [next_claim_id]
        self.genv = genv or {}            # file-scope globals: name -> (rid, CType)
        self.gres = gres or {}            # global rid -> Resource
        self.env: dict[str, tuple[int, CType]] = {}    # name -> (storage rid, type)
        self.resources: dict[int, Resource] = {}
        self.rtypes: dict[int, CType] = {}             # rid -> CType (for emission)
        self.params: list = []
        self.locals: list = []                         # (rid, name, CType) mutable named locals
        self.calls: list = []
        self.block_stack: list = [[]]                  # claims/control nodes append to the top block

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
        if "volatile" in tref.quals:                       # volatile pointee/object -> MMIO region
            base = with_volatile(base)
        t = base
        for _ in range(tref.ptr):
            t = pointer(t)
        for dim in reversed(tref.array):
            t = array(t, dim)
        return t

    def _resource(self, rid: int, ct: CType, name: str) -> None:
        mmio = ct.touches_mmio                              # a volatile object / pointer-to-volatile
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
              bounds: str = "strict", hazard: str = "unique") -> int:
        self.cid[0] += 1
        self.block_stack[-1].append(Claim(
            id=self.cid[0], opcode=opcode, lane=lane, stride_class=stride, count=count,
            rd=tuple(rd), wr=tuple(wr), imm=tuple(imm), domain=domain, op=op, bounds=bounds,
            hazard=hazard))
        return wr[0] if wr else -1

    def _storage(self, ct: CType, name: str) -> int:
        """A mutable named local — assignments write it, reads read it (so control-flow merges and
        loop accumulators work). The emitter declares it and refers to it by name."""
        rid = self._temp(ct, name)
        self.locals.append((rid, name, ct))
        return rid

    def _temp(self, ct: CType, name: str) -> int:
        rid = self._new_rid()
        self._resource(rid, ct, name)
        return rid

    # --- lvalue resolution ---
    def _lvalue(self, node) -> "_LV":
        if isinstance(node, cast.Name):
            rid, ct = self.env[node.ident]
            return _LV("var", rid, ct)
        if isinstance(node, cast.Index):
            base_rid, base_ct = self._addr(node.base)
            idx = self._rvalue(node.index)
            elem = base_ct.of if base_ct.of else scalar("uint32_t")
            return _LV("mem", base_rid, elem, idx=idx)
        if isinstance(node, cast.Member):
            base_rid, base_ct = self._addr(node.base)
            agg = base_ct.of if node.arrow else base_ct
            ftype, byte_off, bit_off, bit_w = agg.field(node.field)
            return _LV("mem", base_rid, ftype, byte_off=byte_off, bit_off=bit_off, bit_width=bit_w)
        if isinstance(node, cast.Unary) and node.op == "*":
            base_rid, base_ct = self._addr(node.operand)
            return _LV("mem", base_rid, base_ct.of or scalar("uint32_t"))
        raise CLowerError(f"not an lvalue: {type(node).__name__}")

    def _addr(self, node):
        """The (rid, type) of an aggregate/pointer base used by member/index access."""
        if isinstance(node, cast.Name):
            return self.env[node.ident]
        if isinstance(node, cast.Member):
            base_rid, base_ct = self._addr(node.base)
            agg = base_ct.of if node.arrow else base_ct
            ftype, _bo, _bf, _bw = agg.field(node.field)
            return base_rid, ftype
        raise CLowerError(f"unsupported base expression {type(node).__name__}")

    def _mmio(self, base_rid: int) -> bool:
        res = self.resources.get(base_rid) or self.gres.get(base_rid)
        return res is not None and res.domain == Domain.MMIO

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
                return self._read(self._lvalue(node))
            if node.op == "&":
                return self._addr(node.operand)[0]
            v = self._rvalue(node.operand)
            opcode, suf = _UN[node.op]
            t = self._temp(scalar("uint32_t"), f"u_{suf}")
            return self._emit(f"c.un.{suf}", opcode, (v,), (t,))
        if isinstance(node, (cast.Index, cast.Member)):
            return self._read(self._lvalue(node))
        if isinstance(node, cast.Assign):
            return self._assign(node)
        if isinstance(node, cast.Ternary):
            # A scalar select: both arms are evaluated (the straight-line subset has no branches),
            # then one is chosen. The emitter renders the real C `(cond ? a : b)` -- behaviour-exact
            # for the pure scalar arms the driver subset uses (no side effects to double-run).
            c = self._rvalue(node.cond)
            a = self._rvalue(node.then)
            b = self._rvalue(node.els)
            t = self._temp(scalar("uint32_t"), "sel")
            return self._emit("c.select", Opcode.ADD, (c, a, b), (t,))
        if isinstance(node, cast.CallExpr):
            return self._call(node)
        raise CLowerError(f"cannot lower expression {type(node).__name__}")

    # --- memory read/write, with bitfield (mask/shift) + MMIO (ordered) handling ---
    def _read(self, lv: "_LV") -> int:
        if lv.kind == "var":
            return lv.rid
        unit = self._load_unit(lv)
        if lv.bit_width:                                     # bitfield extract: (unit >> off) & mask
            t = self._temp(scalar("uint32_t"), "bf")
            return self._emit("c.bf.get", Opcode.ADD, (unit,), (t,),
                              imm=(lv.bit_off, lv.bit_width))
        return unit

    def _load_unit(self, lv: "_LV") -> int:
        t = self._temp(lv.ct if not lv.bit_width else scalar("uint32_t"), "ld")
        rd = (lv.rid,) if lv.idx is None else (lv.rid, lv.idx)
        mmio = self._mmio(lv.rid)
        # the byte offset rides in imm (not the strict-bounds `offset`), and the access is
        # assumed_safe (the frontend resolved the member/index). MMIO accesses are ordered.
        return self._emit("c.load", Opcode.LOAD, rd, (t,),
                          imm=(lv.byte_off,) if lv.byte_off else (),
                          domain=Domain.MMIO if mmio else Domain.RAM, bounds="assumed_safe",
                          lane=Lane.H if mmio else Lane.U,
                          hazard="barriered" if mmio else "unique")

    def _write(self, lv: "_LV", v: int) -> None:
        if lv.bit_width:                                     # read-modify-write the storage unit
            old = self._load_unit(lv)
            t = self._temp(scalar("uint32_t"), "bf")
            v = self._emit("c.bf.set", Opcode.ADD, (old, v), (t,), imm=(lv.bit_off, lv.bit_width))
        mmio = self._mmio(lv.rid)
        if lv.idx is None:                                    # member/deref: carry (offset, size)
            rd, imm = (lv.rid, v), (lv.byte_off, max(1, lv.ct.size))
        else:                                                 # base[idx]: a typed array store
            rd, imm = (lv.rid, lv.idx, v), ()
        self._emit("c.store", Opcode.STORE, rd, (), imm=imm,
                   domain=Domain.MMIO if mmio else Domain.RAM, bounds="assumed_safe",
                   lane=Lane.H if mmio else Lane.U,
                   hazard="barriered" if mmio else "unique")

    def _assign(self, node: cast.Assign) -> int:
        v = self._rvalue(node.value)
        if isinstance(node.target, cast.Name) and node.target.ident in self.env:
            rid, _ct = self.env[node.target.ident]           # copy into the mutable storage
            self._emit("c.copy", Opcode.ADD, (v,), (rid,))
            return rid
        self._write(self._lvalue(node.target), v)
        return v

    # GCC/Clang atomic + fence builtins -> the BCIR ATOMIC_*/BARRIER opcodes (§5.8).
    _ATOMIC = {"__atomic_fetch_add": ("c.atomic.add", Opcode.ATOMIC_ADD),
               "__atomic_fetch_sub": ("c.atomic.sub", Opcode.ATOMIC_SUB),
               "__atomic_fetch_xor": ("c.atomic.xor", Opcode.ATOMIC_XOR)}
    _FENCE = {"__atomic_thread_fence", "__sync_synchronize"}
    # Compare-and-swap -> the CMPXCHG opcode: a 3-read claim (ptr, expected, desired). The `val`
    # form returns the pre-swap value, the `bool` form returns whether the swap happened.
    _CMPXCHG = {"__sync_val_compare_and_swap": "c.cmpxchg.val",
                "__sync_bool_compare_and_swap": "c.cmpxchg.bool"}

    def _call(self, node: cast.CallExpr) -> int:
        actuals = tuple(self._rvalue(a) for a in node.args)
        # Atomics run on the A lane. A scalar atomic counter is a single-location RMW (not on
        # the decoupled GGG/scatter tail), so it stays SCALAR-shaped -- the lane law (R6) admits
        # lane A for SCALAR, and the atomic/barriered hazard discharges R5.
        if node.callee in self._FENCE:
            t = self._temp(scalar("uint32_t"), "fence")
            self._emit("c.fence", Opcode.BARRIER, (), (), lane=Lane.A, hazard="barriered")
            return t
        if node.callee in self._ATOMIC:
            op, oc = self._ATOMIC[node.callee]
            ptr = actuals[0]
            val = actuals[1] if len(actuals) > 1 else actuals[0]
            t = self._temp(scalar("uint32_t"), "atom")
            dom = self.resources[ptr].domain if ptr in self.resources else Domain.RAM
            return self._emit(op, oc, (ptr, val), (t,), lane=Lane.A, domain=dom, hazard="atomic")
        if node.callee in self._CMPXCHG:
            op = self._CMPXCHG[node.callee]
            ptr = actuals[0]
            exp = actuals[1] if len(actuals) > 1 else ptr
            des = actuals[2] if len(actuals) > 2 else exp
            t = self._temp(scalar("uint32_t"), "cas")
            dom = self.resources[ptr].domain if ptr in self.resources else Domain.RAM
            return self._emit(op, Opcode.CMPXCHG, (ptr, exp, des), (t,),
                              lane=Lane.A, domain=dom, hazard="atomic")
        t = self._temp(scalar("uint32_t"), f"call_{node.callee}")
        self.calls.append((node.callee, actuals))
        return self._emit(f"c.call:{node.callee}", Opcode.GEM_DISPATCH, actuals, (t,))

    # --- statements ---
    def _block(self, stmts) -> list:
        block: list = []
        self.block_stack.append(block)
        for s in stmts:
            self._stmt(s)
        self.block_stack.pop()
        return block

    def _stmt(self, st):
        if isinstance(st, cast.Decl):
            ct = self._resolve_type(st.type)
            rid = self._storage(ct, st.name)                  # a mutable named local
            self.env[st.name] = (rid, ct)
            if st.init is not None:
                self._emit("c.copy", Opcode.ADD, (self._rvalue(st.init),), (rid,))
        elif isinstance(st, cast.ExprStmt):
            self._rvalue(st.expr)
        elif isinstance(st, cast.Return):
            rid = None if st.value is None else self._rvalue(st.value)
            self.block_stack[-1].append(ReturnNode(rid))
            if rid is not None:
                self.last_return = rid
        elif isinstance(st, cast.If):
            cond = self._rvalue(st.cond)                       # condition claims -> current block
            node = IfNode(cond, self._block(st.then), self._block(st.els))
            self.block_stack[-1].append(node)
        elif isinstance(st, cast.While):
            cond_block: list = []
            self.block_stack.append(cond_block)
            cond = self._rvalue(st.cond)
            self.block_stack.pop()
            self.block_stack[-1].append(WhileNode(cond_block, cond, self._block(st.body)))
        elif isinstance(st, cast.For):
            if st.init is not None:                            # init -> the enclosing block, once
                self._stmt(st.init)
            cond_block2: list = []
            self.block_stack.append(cond_block2)
            cond = self._rvalue(st.cond)
            self.block_stack.pop()
            body = self._block(st.body)
            if st.step is not None:                            # the step runs at the end of each iter
                self.block_stack.append(body)
                self._stmt(st.step)
                self.block_stack.pop()
            self.block_stack[-1].append(WhileNode(cond_block2, cond, body))
        else:
            raise CLowerError(f"statement {type(st).__name__} is beyond the L1–L6 subset")
        return None

    def lower(self) -> LoweredFunc:
        self.last_return = None
        self.env.update(self.genv)                            # file-scope globals are in scope
        for rid, ct in self.genv.values():
            self.rtypes.setdefault(rid, ct)
        for p in self.func.params:                            # params -> input resources (shadowing)
            ct = self._resolve_type(p.type)
            rid = self._new_rid()
            self._resource(rid, ct, p.name)
            self.env[p.name] = (rid, ct)
            self.params.append((p.name, rid, ct))
        for st in self.func.body:
            self._stmt(st)
        body = self.block_stack[0]
        claims = _flatten_block(body)
        touched = {r for c in claims for r in (tuple(c.rd) + tuple(c.wr))}
        for rid in touched & set(self.gres):                  # pull in referenced global resources
            self.resources[rid] = self.gres[rid]
        gnames = {rid: nm for nm, (rid, _ct) in self.genv.items() if rid in touched}
        m = Module(name=self.func.name)
        for rid in sorted(self.resources):
            m.add_resource(self.resources[rid])
        m.add_phase(Phase(phase_id=0, deps=(), claims=list(claims)))
        ret_ct = self._resolve_type(self.func.ret)
        return LoweredFunc(name=self.func.name, module=m, ret_type=ret_ct, params=self.params,
                           return_rid=self.last_return, claims=list(claims),
                           resources=dict(self.resources), rid_types=dict(self.rtypes),
                           calls=list(self.calls), region=None, body=body, locals=list(self.locals),
                           globals_used=gnames)


def _block_region(block: list, functions: dict, calls_iter: list) -> "compose.Region":
    """Map a body block to a compose region: straight-line claim runs -> Leaf, an IfNode -> Cond,
    a WhileNode -> its body region (planned once — a bounded loop's per-iteration cost), and a call
    claim -> compose.Call (so plan_composite's R18 checks see the real call graph)."""
    parts: list = []
    run: list = []

    def flush_run():
        if run:
            parts.append(compose.Leaf(tuple(run)))
            run.clear()

    for node in block:
        if isinstance(node, IfNode):
            flush_run()
            parts.append(compose.Cond("c", _block_region(node.then, functions, calls_iter),
                                      _block_region(node.els, functions, calls_iter)))
        elif isinstance(node, WhileNode):
            flush_run()
            _block_region(node.cond_block, functions, calls_iter)   # cond claims (cost folded in body)
            parts.append(_block_region(node.body, functions, calls_iter))
        elif isinstance(node, ReturnNode):
            continue
        elif node.op.startswith("c.call:"):
            flush_run()
            callee, actuals = calls_iter.pop(0)
            callee_fn = functions.get(callee)
            formals = [rid for _n, rid, _ct in callee_fn.params] if callee_fn else []
            parts.append(compose.Call(callee, tuple(zip(formals, actuals))))
        else:
            run.append(node)
    flush_run()
    return parts[0] if len(parts) == 1 else compose.Seq(tuple(parts))


def _region_for(lf: LoweredFunc, functions: dict) -> compose.Region:
    """The function's compose region, built from its structured body so `plan_composite` sees the
    real control-flow + call graph (R18: callee resolution + no recursion). An undefined callee is
    left unmapped so `plan_composite` raises."""
    return _block_region(lf.body, functions, list(lf.calls))


def lower_unit(unit: cast.Unit) -> LoweredUnit:
    """Lower a whole translation unit. Aggregates are laid out first (Clang-compatible), then each
    function to its claim-graph Module + compose.Function."""
    aggregates: dict[str, CType] = {}
    for tag, agg in unit.aggregates.items():
        b = AggregateBuilder(agg.kind, tag, packed=agg.packed, force_align=agg.align)
        for tref, mname, width in agg.members:
            b.members.append((mname, _resolve_member_type(tref, aggregates), width))
        aggregates[tag] = b.build()

    genv: dict[str, tuple] = {}                            # file-scope globals: name -> (rid, CType)
    gres: dict[int, Resource] = {}
    for gi, g in enumerate(unit.globals):
        ct = _resolve_member_type(g.type, aggregates)
        if g.init and ct.kind == "array" and ct.count == 0:
            ct = array(ct.of, len(g.init))                # `T name[] = {...}` -> sized from the init
        elif g.init and ct.kind != "array":
            ct = array(ct, len(g.init))
        rid = 900000 + gi
        gres[rid] = Resource(rid=rid, domain=Domain.RAM, elem_bytes=(ct.of.size if ct.of else ct.size),
                             shape=(ct.count or len(g.init) or 1,), access="ro",
                             data_gen=1, name=g.name)
        genv[g.name] = (rid, ct)

    functions: dict[str, LoweredFunc] = {}
    compose_functions: dict[str, compose.Function] = {}
    resources: dict[int, Resource] = dict(gres)
    cid = [1000]
    for idx, fn in enumerate(unit.funcs):
        lf = _FuncLowerer(fn, aggregates, base_rid=100 + idx * 1000, cid=cid,
                          genv=genv, gres=gres).lower()
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
