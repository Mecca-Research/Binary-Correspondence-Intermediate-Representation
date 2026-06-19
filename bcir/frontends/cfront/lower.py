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

from dataclasses import dataclass, field, replace

from ...kbcir import compose
from ...model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from . import cast
from .ctype_model import (
    AggregateBuilder,
    CType,
    array,
    funcptr,
    is_scalar_name,
    pointer,
    scalar,
    with_atomic,
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

# A cast's target type, named by width so both rails emit the same (uintN_t) spelling. The C value
# model tracks everything as a 32-bit unit, so a cast renders by size (a pointer cast keeps its `*`).
_CAST_W = {1: "uint8_t", 2: "uint16_t", 4: "uint32_t", 8: "uint64_t"}


def _cast_name(ct: CType) -> str:
    if ct.kind == "pointer":
        return _cast_name(ct.of) + " *" if ct.of else "void *"
    return _CAST_W.get(ct.size, "uint32_t")


def _str_bytes(spelling: str) -> int:
    """The number of bytes a string literal's value occupies *excluding* the terminating NUL,
    decoding escape sequences (a simple `\\c`, an octal `\\NNN`, or a hex `\\xHH..` each count as one
    byte). `spelling` is the source text including the surrounding quotes."""
    s = spelling[1:-1] if len(spelling) >= 2 and spelling[0] == '"' else spelling
    i, n = 0, 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            c = s[i + 1]
            if c == "x":                                       # \xHH.. -> consume all hex digits
                i += 2
                while i < len(s) and s[i] in "0123456789abcdefABCDEF":
                    i += 1
            elif c in "01234567":                              # \NNN  -> up to three octal digits
                i += 1
                k = 0
                while k < 3 and i < len(s) and s[i] in "01234567":
                    i, k = i + 1, k + 1
            else:                                              # \n, \t, \\, \", \0, ...
                i += 2
        else:
            i += 1
        n += 1
    return n


def _fold_const(node) -> int:
    """A `static` local's initializer must be a constant expression (it is baked into the C
    declaration). Fold the integer-constant subset; a missing init zero-initializes."""
    if node is None:
        return 0
    if isinstance(node, cast.IntLit):
        return node.value
    if isinstance(node, cast.Unary) and node.op in ("-", "~"):
        v = _fold_const(node.operand)
        return -v if node.op == "-" else ~v
    if isinstance(node, cast.Binary):
        a, b = _fold_const(node.lhs), _fold_const(node.rhs)
        return {"+": a + b, "-": a - b, "*": a * b, "&": a & b, "|": a | b,
                "^": a ^ b, "<<": a << b, ">>": a >> b}.get(node.op, 0)
    raise CLowerError("static initializer is not a constant expression")


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
    test_at_end: bool = False              # do/while: run body first, then test the condition
    step: list = field(default_factory=list)   # for-loop step: runs after body, at the continue point
    loop_id: int = 0                       # unique id for the emitter's `__cont_<id>` label


@dataclass
class ReturnNode:
    rid: int | None                        # the returned value's rid (None == `return;`)


@dataclass
class BreakNode:
    pass                                   # `break;` -- exit the nearest enclosing loop (emit-only)


@dataclass
class ContinueNode:
    pass                                   # `continue;` -- jump to the loop's continue point (emit-only)


@dataclass
class GotoNode:
    label: str                             # `goto label;` -- an unconditional jump (emit-only)


@dataclass
class LabelNode:
    name: str                              # `name:` -- a jump target (emit-only)


def _flatten_block(block: list) -> list:
    """All Claim objects in a body tree, in order (the flat single-phase view verify/plan use)."""
    out: list = []
    for node in block:
        if isinstance(node, IfNode):
            out += _flatten_block(node.then) + _flatten_block(node.els)
        elif isinstance(node, WhileNode):
            out += _flatten_block(node.cond_block) + _flatten_block(node.body) + _flatten_block(node.step)
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
    statics: list = field(default_factory=list)     # (rid, name, CType, init) static-storage locals
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
        self.statics: list = []                        # (rid, name, CType, init) static-storage locals
        self.calls: list = []
        self.block_stack: list = [[]]                  # claims/control nodes append to the top block
        self.loop_ctr = 0                              # unique loop ids (the `continue` label numbering)

    def _next_loop_id(self) -> int:
        self.loop_ctr += 1
        return self.loop_ctr

    # --- resource allocation ---
    def _new_rid(self) -> int:
        self.rid += 1
        return self.rid

    def _resolve_type(self, tref: cast.TypeRef) -> CType:
        if tref.funcptr:                                   # a function-pointer alias (HAL dispatch)
            ret = self._resolve_type(tref.func_ret)
            params = tuple(self._resolve_type(p) for p in tref.func_params)
            return funcptr(tref.base, ret, params)
        if tref.aggregate:
            base = self.aggregates[tref.base]
        elif is_scalar_name(tref.base):
            base = scalar(tref.base)
        else:
            raise CLowerError(f"unknown type {tref.base!r}")
        if "volatile" in tref.quals:                       # volatile pointee/object -> MMIO region
            base = with_volatile(base)
        if "_Atomic" in tref.quals:                        # _Atomic-qualified object (C11/C23)
            base = with_atomic(base)
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

    def _static_storage(self, ct: CType, name: str, init: int) -> int:
        """A `static` local: persistent storage with a once-only constant initializer baked into the
        declaration (so the init is NOT a per-call assignment). Reads/writes hit it like any local."""
        rid = self._temp(ct, name)
        self.statics.append((rid, name, ct, init))
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
            # collect the (possibly nested) index chain down to the ultimate base, then flatten
            # row-major: m[i][j] on a `T m[A][B]` param -> the linear index i*B + j (Horner).
            idx_nodes, n = [], node
            while isinstance(n, cast.Index):
                idx_nodes.append(n.index)
                n = n.base
            idx_nodes.reverse()
            base_rid, base_ct = self._addr(n)
            idx_rids = [self._rvalue(ix) for ix in idx_nodes]
            shape = base_ct.shape
            lin = idx_rids[0]
            for d in range(1, len(idx_rids)):
                dim = shape[d] if d < len(shape) else 1
                k = self._temp(scalar("uint32_t"), f"k{dim}")
                self._emit("c.const", Opcode.LOAD, (), (k,), imm=(dim,))
                m1 = self._temp(scalar("uint32_t"), "b_mul")
                self._emit("c.bin.mul", Opcode.MUL, (lin, k), (m1,))
                a1 = self._temp(scalar("uint32_t"), "b_add")
                self._emit("c.bin.add", Opcode.ADD, (m1, idx_rids[d]), (a1,))
                lin = a1
            elem = base_ct.of if base_ct.of else scalar("uint32_t")
            return _LV("mem", base_rid, elem, idx=lin)
        if isinstance(node, cast.Member):
            base_rid, base_ct = self._addr(node.base)
            agg = base_ct.of if node.arrow else base_ct
            ftype, byte_off, bit_off, bit_w = agg.field(node.field)
            return _LV("mem", base_rid, ftype, byte_off=byte_off, bit_off=bit_off, bit_width=bit_w)
        if isinstance(node, cast.Unary) and node.op == "*":
            operand = node.operand
            if isinstance(operand, cast.Binary) and operand.op == "+":   # *(p + i) == p[i]
                return self._lvalue(cast.Index(operand.lhs, operand.rhs))
            base_rid, base_ct = self._addr(operand)
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
        if isinstance(node, cast.Cast):
            v = self._rvalue(node.operand)
            ct = self._resolve_type(node.type)
            # assigning the cast to a uint32 temp reproduces the integer-promotion semantics: a
            # narrowing cast masks (zero-extends back), so downstream arithmetic matches Clang.
            t = self._temp(ct if ct.is_integer else scalar("uint32_t"), "cast")
            return self._emit(f"c.cast:{_cast_name(ct)}", Opcode.ADD, (v,), (t,))
        if isinstance(node, (cast.Index, cast.Member)):
            return self._read(self._lvalue(node))
        if isinstance(node, cast.Assign):
            return self._assign(node)
        if isinstance(node, cast.SizeOf):
            # sizeof folds to a compile-time constant -- the operand is NOT evaluated.
            if node.type is not None:
                size = self._resolve_type(node.type).size
            elif isinstance(node.expr, cast.StringLit):
                size = _str_bytes(node.expr.value) + 1         # char[N]: the bytes + the NUL
            elif isinstance(node.expr, cast.Name):
                size = self.env[node.expr.ident][1].size       # the variable's declared type size
            else:
                size = 4                                       # an integer expression -> int
            t = self._temp(scalar("uint32_t"), "szof")
            return self._emit("c.const", Opcode.LOAD, (), (t,), imm=(size,))
        if isinstance(node, cast.AlignOf):
            # _Alignof folds to the target type's alignment (operand never evaluated, like sizeof).
            t = self._temp(scalar("uint32_t"), "alof")
            return self._emit("c.const", Opcode.LOAD, (), (t,), imm=(self._resolve_type(node.type).align,))
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
        if isinstance(node, cast.CallMember):
            return self._call_member(node)
        raise CLowerError(f"cannot lower expression {type(node).__name__}")

    def _call_member(self, node: cast.CallMember) -> int:
        """`o->fn(args)` / `o.fn(args)` — an indirect call through a function-pointer struct member
        (the dispatch-table pattern). Fused into one `c.call.imember:<field>` claim (reads: the struct
        base, then the actuals) emitted as `o->fn(args)`, so no 8-byte function-pointer value has to
        ride in the 4-byte value model. Not added to the call graph (R18: an opaque external edge)."""
        m = node.callee
        base_rid, _base_ct = self._addr(m.base)
        actuals = tuple(self._rvalue(a) for a in node.args)
        t = self._temp(scalar("uint32_t"), f"icall_{m.field}")
        return self._emit(f"c.call.imember:{m.field}", Opcode.GEM_DISPATCH,
                          (base_rid, *actuals), (t,), imm=(1 if m.arrow else 0,))

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
    # C11 <stdatomic.h> generics on _Atomic objects -> the same ATOMIC opcodes, but emitted as the C11
    # functions (which accept an _Atomic* -- the __atomic_* builtins do not).
    _C11_RMW = {"atomic_fetch_add": ("c.c11atom.fetch_add", Opcode.ATOMIC_ADD),
                "atomic_fetch_sub": ("c.c11atom.fetch_sub", Opcode.ATOMIC_SUB),
                "atomic_fetch_xor": ("c.c11atom.fetch_xor", Opcode.ATOMIC_XOR),
                "atomic_exchange":  ("c.c11atom.exchange", Opcode.ATOMIC_ADD)}  # swap: set + return old

    def _call(self, node: cast.CallExpr) -> int:
        actuals = tuple(self._rvalue(a) for a in node.args)
        # Indirect call through a function-pointer local/param (HAL dispatch): the target is dynamic,
        # so there is no named callee -- it lowers to a `c.call.indirect` claim (reads: the pointer
        # value then the actuals) and is *not* added to the call graph, leaving R18 to treat it as an
        # opaque external edge (no recursion / callee-resolution constraint can apply).
        if node.callee in self.env and self.env[node.callee][1].kind == "funcptr":
            fptr = self.env[node.callee][0]
            t = self._temp(scalar("uint32_t"), f"icall_{node.callee}")
            return self._emit("c.call.indirect", Opcode.GEM_DISPATCH, (fptr, *actuals), (t,))
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
        if node.callee in self._C11_RMW:               # atomic_fetch_add/sub/xor(p, v) -> old value
            op, oc = self._C11_RMW[node.callee]
            ptr = actuals[0]
            val = actuals[1] if len(actuals) > 1 else actuals[0]
            t = self._temp(scalar("uint32_t"), "c11")
            dom = self.resources[ptr].domain if ptr in self.resources else Domain.RAM
            return self._emit(op, oc, (ptr, val), (t,), lane=Lane.A, domain=dom, hazard="atomic")
        if node.callee == "atomic_load":               # atomic_load(p) -> *p (ordered)
            ptr = actuals[0]
            t = self._temp(scalar("uint32_t"), "c11ld")
            dom = self.resources[ptr].domain if ptr in self.resources else Domain.RAM
            return self._emit("c.c11atom.load", Opcode.LOAD, (ptr,), (t,),
                              lane=Lane.A, domain=dom, hazard="atomic")
        if node.callee == "atomic_store":              # atomic_store(p, v) (ordered, no value)
            ptr = actuals[0]
            val = actuals[1] if len(actuals) > 1 else actuals[0]
            dom = self.resources[ptr].domain if ptr in self.resources else Domain.RAM
            self._emit("c.c11atom.store", Opcode.STORE, (ptr, val), (),
                       lane=Lane.A, domain=dom, hazard="atomic")
            return ptr
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
            if st.static_storage:                             # static storage: init once, in the decl
                rid = self._static_storage(ct, st.name, _fold_const(st.init))
                self.env[st.name] = (rid, ct)
                return None
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
            self.block_stack[-1].append(
                WhileNode(cond_block, cond, self._block(st.body), loop_id=self._next_loop_id()))
        elif isinstance(st, cast.For):
            if st.init is not None:                            # init -> the enclosing block, once
                self._stmt(st.init)
            cond_block2: list = []
            self.block_stack.append(cond_block2)
            cond = self._rvalue(st.cond)
            self.block_stack.pop()
            body = self._block(st.body)
            step: list = []
            if st.step is not None:                            # the step runs after body (the continue point)
                self.block_stack.append(step)
                self._stmt(st.step)
                self.block_stack.pop()
            self.block_stack[-1].append(
                WhileNode(cond_block2, cond, body, step=step, loop_id=self._next_loop_id()))
        elif isinstance(st, cast.DoWhile):                     # body runs, then the cond is tested
            body = self._block(st.body)
            cond_block3: list = []
            self.block_stack.append(cond_block3)
            cond = self._rvalue(st.cond)
            self.block_stack.pop()
            self.block_stack[-1].append(
                WhileNode(cond_block3, cond, body, test_at_end=True, loop_id=self._next_loop_id()))
        elif isinstance(st, cast.Break):
            self.block_stack[-1].append(BreakNode())
        elif isinstance(st, cast.Continue):
            self.block_stack[-1].append(ContinueNode())
        elif isinstance(st, cast.Goto):
            self.block_stack[-1].append(GotoNode(st.label))
        elif isinstance(st, cast.Label):
            self.block_stack[-1].append(LabelNode(st.name))
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
            if ct.kind == "array":                            # an array param decays to a flat
                dims, elem = [], ct                           # element pointer + a recorded shape
                while elem.kind == "array":
                    dims.append(elem.count)
                    elem = elem.of
                ct = replace(pointer(elem), shape=tuple(dims))
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
                           statics=list(self.statics), globals_used=gnames)


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
            parts.append(_block_region(node.body + node.step, functions, calls_iter))
        elif isinstance(node, (ReturnNode, BreakNode, ContinueNode, GotoNode, LabelNode)):
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
    if tref.funcptr:                                       # a function-pointer member (dispatch table)
        ret = _resolve_member_type(tref.func_ret, aggregates)
        params = tuple(_resolve_member_type(p, aggregates) for p in tref.func_params)
        return funcptr(tref.base, ret, params)
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
