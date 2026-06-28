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

import dataclasses
from dataclasses import dataclass, field, replace

from ...kbcir import compose
from ...model import Claim, Domain, Lane, Lifetime, Module, Opcode, Phase, Resource, StrideClass
from . import cast
from .abi import HOST
from .clex import split_lit_prefix, str_elem_size
from .ctype_model import (
    AggregateBuilder,
    BitIntMix,
    CType,
    array,
    bitint,
    funcptr,
    is_scalar_name,
    pointer,
    promote_int,
    scalar,
    usual_arith_int,
    valist,
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
_CAST_W_SIGNED = {1: "int8_t", 2: "int16_t", 4: "int32_t", 8: "int64_t"}


def _cast_name(ct: CType) -> str:
    if ct.kind == "pointer":
        return _cast_name(ct.of) + " *" if ct.of else "void *"
    if ct.is_bitint:
        return ct.name                                # `_BitInt(N)` / `unsigned _BitInt(N)` -- faithful, exact width
    if ct.is_float:
        return ct.name                                # float / double / long double
    if ct.name in ("_Bool", "bool"):
        return "_Bool"                                # a bool cast normalizes any nonzero to 1 on the
                                                      # FULL value -- (uint8_t) would truncate first
                                                      # (so (_Bool)256 must be 1, not 0; (_Bool)0.5 is 1)
    return _CAST_W.get(ct.size, "uint32_t")


_FLOAT_RANK = {"float": 0, "double": 1, "long double": 2}
_RANK_FLOAT = {0: "float", 1: "double", 2: "long double"}


def _elem_float_name(t) -> str:
    """The element float spelling of a (possibly complex) float type: `double _Complex` -> `double`,
    a plain `double` -> `double`. Used to promote complex arithmetic by element rank."""
    return t.name[:-len(" _Complex")] if t.is_complex else t.name


def _float_lit_type(spelling: str) -> CType:
    """The type of a floating literal from its suffix: `f`/`F` -> float, `l`/`L` -> long double,
    else double (the C default)."""
    s = spelling.strip()
    if s and s[-1] in "fF":
        return scalar("float")
    if s and s[-1] in "lL":
        return scalar("long double")
    return scalar("double")


# <math.h> functions whose result is a floating type fixed by the name suffix (base -> double,
# +f -> float, +l -> long double). Most are real-valued; a few take a non-floating argument that is
# simply passed through -- ldexp/scalbn (an int exponent), scalbln (a long), nan (a tag string). They
# lower to an opaque external library edge: the emit calls the real libm function (the harness links
# -lm) so the IEEE-754 result is delegated to the backend, and the call graph sees no callee to
# resolve (R18 treats it like an indirect call).
_LIBM = frozenset({
    "acos", "asin", "atan", "atan2", "cos", "sin", "tan",
    "acosh", "asinh", "atanh", "cosh", "sinh", "tanh",
    "exp", "exp2", "expm1", "log", "log10", "log1p", "log2", "logb",
    "cbrt", "fabs", "hypot", "pow", "sqrt",
    "ceil", "floor", "round", "trunc", "nearbyint", "rint",
    "erf", "erfc", "lgamma", "tgamma",
    "copysign", "fdim", "fmax", "fmin", "fmod", "remainder", "fma", "nextafter",
    "ldexp", "scalbn", "scalbln", "nan",
    "frexp", "modf", "remquo",                            # a pointer out-param (rides c.addrof); double result
})
# <math.h> functions with a fixed *integer* result (the f/l suffix types only the argument): ilogb
# returns int; lround/lrint return long and llround/llrint return long long. The emitted result temp
# is declared at the rid type's true width, so the 8-byte returns are not truncated to the 4-byte
# value model. (frexp/modf/remquo, which take a pointer out-param, ride the address-of-argument path.)
_LIBM_INT = {"ilogb": "int",
             "lround": "long", "lrint": "long",
             "llround": "long long", "llrint": "long long"}

# <complex.h> functions, lowered like a libm call (opaque external edge, emitted verbatim against
# <complex.h>). The result type differs by function: creal/cimag/cabs/carg return the *real* element
# float; conj/cproj AND the C99 complex transcendentals return the *complex* type. The f/l suffix picks
# float / long double (base -> double).
# <complex.h> imaginary unit: `I` is the macro `_Complex_I`, a `const float _Complex` of value i (0+1i).
# (Clang doesn't implement `_Imaginary`, so `_Imaginary_I` is intentionally not recognized.) Lowered to a
# `c.cconst` that emits the token VERBATIM -- the re-emitted twin compiles against <complex.h> just like the
# original, so it resolves to the same imaginary unit. Honored only when NOT shadowed by a declared name.
_IMAG_UNIT = frozenset({"I", "_Complex_I"})

_CPLX_REAL = frozenset({"creal", "cimag", "cabs", "carg"})    # -> real element float
_CPLX_CPLX = frozenset({                                      # -> the complex type (same width)
    "conj", "cproj",                                          #   algebraic
    "cexp", "clog", "csqrt", "cpow",                          #   exp / log / sqrt / power
    "csin", "ccos", "ctan", "casin", "cacos", "catan",        #   circular + inverse
    "csinh", "ccosh", "ctanh", "casinh", "cacosh", "catanh"}) #   hyperbolic + inverse


def _complex_libm_type(name: str) -> "CType | None":
    """The result type of a <complex.h> call, or None. The FULL name is tested first so creal/cimag
    (which themselves end in l/g) are not misread as the long-double `*l` variant of a shorter base."""
    if name in _CPLX_REAL:
        return scalar("double")
    if name in _CPLX_CPLX:
        return scalar("double _Complex")
    base, suf = name[:-1], name[-1:]
    if suf in "fl" and base in _CPLX_REAL:
        return scalar("float" if suf == "f" else "long double")
    if suf in "fl" and base in _CPLX_CPLX:
        return scalar(f"{'float' if suf == 'f' else 'long double'} _Complex")
    return None

# the printf / scanf family of external variadic <stdio.h> functions -- not defined in the unit and not
# lowered, they emit verbatim (like a libm call, opaque to R18) and return int.
_EXTERN_VARIADIC = frozenset({
    "snprintf", "vsnprintf", "sprintf", "vsprintf", "printf", "fprintf", "vprintf",
    "vfprintf", "sscanf", "vsscanf", "scanf", "fscanf", "dprintf"})

# <stdlib.h> memory management -- external libc edges (emitted VERBATIM, opaque to R18, NOT bcir_-renamed),
# the seam the naked-pointer safety track (§5.12) hangs lifetime annotations on. The allocators return a
# `void *` (assignable to any object pointer); `free` returns void.
_STDLIB_ALLOC = frozenset({"malloc", "calloc", "realloc", "aligned_alloc"})

# GCC/Clang integer builtins -- emitted verbatim (no bcir_ twin, opaque to R18) with a fixed result type.
_BUILTIN_INT = frozenset({                            # the bit-count family + abs -> int
    "__builtin_popcount", "__builtin_popcountl", "__builtin_popcountll",
    "__builtin_clz", "__builtin_clzl", "__builtin_clzll", "__builtin_ctz", "__builtin_ctzl", "__builtin_ctzll",
    "__builtin_ffs", "__builtin_ffsl", "__builtin_ffsll", "__builtin_parity", "__builtin_parityl",
    "__builtin_parityll", "__builtin_clrsb", "__builtin_clrsbl", "__builtin_clrsbll", "__builtin_abs"})
_BUILTIN_OTHER = {"__builtin_labs": "long", "__builtin_llabs": "long long",
                  "__builtin_bswap16": "uint16_t", "__builtin_bswap32": "uint32_t",
                  "__builtin_bswap64": "uint64_t"}


def _builtin_type(name: str, abi) -> "CType | None":
    if name in _BUILTIN_INT:
        return scalar("int", abi)
    if name in _BUILTIN_OTHER:
        return scalar(_BUILTIN_OTHER[name], abi)
    return None


def _libm_type(name: str) -> CType | None:
    """The result type of a `<math.h>` call: real-valued ones are typed by the name suffix
    (base -> double, `f` -> float, `l` -> long double); a fixed-integer one (`ilogb`) returns its
    int type for any suffix. None if `name` is not a recognized libm function. The full name is
    tried first so a base that ends in `f` (`erf`) is not misread as the float variant of `er`."""
    cx = _complex_libm_type(name)                     # <complex.h> creal/cimag/conj/... (typed first)
    if cx is not None:
        return cx
    base = name[:-1] if name[-1:] in "fl" else name
    if name in _LIBM_INT:
        return scalar(_LIBM_INT[name])
    if base in _LIBM_INT:                             # a suffixed variant (ilogbf/ilogbl) -> same int
        return scalar(_LIBM_INT[base])
    if name in _LIBM:
        return scalar("double")
    if name[-1:] == "f" and name[:-1] in _LIBM:
        return scalar("float")
    if name[-1:] == "l" and name[:-1] in _LIBM:
        return scalar("long double")
    return None


def _str_bytes(spelling: str) -> int:
    """The number of bytes a (possibly concatenated) string literal's value occupies *excluding* the
    terminating NUL, decoding escape sequences (a simple `\\c`, an octal `\\NNN`, or a hex `\\xHH..`
    each count as one byte). `spelling` is the source text including the quotes; adjacent literals
    (`"a" "b"`) are walked quote-aware, so bytes are counted only *inside* quotes and the inter-piece
    whitespace is ignored -- the pieces stay separate, so a hex/octal escape can't merge with the next
    piece's leading digit."""
    i, n, inq = 0, 0, False
    while i < len(spelling):
        ch = spelling[i]
        if not inq:                                            # between pieces: only `"` opens one
            inq = ch == '"'
            i += 1
            continue
        if ch == '"':                                          # closing quote of this piece
            inq = False
            i += 1
            continue
        if ch == "\\" and i + 1 < len(spelling):
            c = spelling[i + 1]
            if c == "x":                                       # \xHH.. -> consume all hex digits
                i += 2
                while i < len(spelling) and spelling[i] in "0123456789abcdefABCDEF":
                    i += 1
            elif c in "01234567":                              # \NNN  -> up to three octal digits
                i += 1
                k = 0
                while k < 3 and i < len(spelling) and spelling[i] in "01234567":
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


def _is_pure(node) -> bool:
    """True if evaluating `node` has NO side effect -- safe to re-evaluate for the §5.12 extent snapshot:
    an integer literal, a name read, `sizeof`/`_Alignof` (unevaluated), or an arithmetic unary / binary /
    cast over pure operands. A call, assignment, `++`/`--`, deref/address-of, or anything else is NOT pure
    (conservative: an unrecognized node is impure, so it stays unmanaged rather than double-run)."""
    if isinstance(node, (cast.IntLit, cast.Name, cast.SizeOf, cast.AlignOf)):
        return True
    if isinstance(node, cast.Unary):
        return node.op in ("-", "~", "+") and _is_pure(node.operand)       # arithmetic only (NOT * / &)
    if isinstance(node, cast.Binary):
        return (node.op in {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"}
                and _is_pure(node.lhs) and _is_pure(node.rhs))
    if isinstance(node, cast.Cast):
        return _is_pure(node.operand)
    return False


def _scan_mutations(node, assigned: dict, body: dict, addr: set) -> None:
    """Walk a function-body AST collecting, per name: the TOTAL assignment count (`assigned`: a decl-init,
    `x = e`, compound `x OP= e`, and `x++`/`x--` -- which the parser desugars to an assign), the count of
    BODY (non-decl-init) assignments (`body`), and whether its address is ever taken (`addr`: `&x`). Drives
    §5.12 recoverable-extent soundness.

    The two tallies serve different gates. A bound POINTER is trusted when `assigned == 1` (defined exactly
    once -- at its allocation -- never reassigned/arith'd); its rid is used directly, so a decl-init binding
    is as good as a plain assignment. A recovered COUNT is re-emitted by NAME at every access, so it must be
    STABLE from the allocation onward -- it is trusted only when `body == 0`: a decl-init (`m = n & 31`,
    before the alloc) is fine, but ANY ordinary assignment (`n = n - 1`, `n--`) could mutate it AFTER the
    alloc and make the runtime extent disagree with the allocation -- a false trap. Order-independent and
    conservative: an over-approximation of mutation (more seen -> fewer promotions, never an unsound one)."""
    if isinstance(node, (list, tuple)):
        for x in node:
            _scan_mutations(x, assigned, body, addr)
        return
    if not dataclasses.is_dataclass(node):
        return
    if isinstance(node, cast.Assign):
        if isinstance(node.target, cast.Name):
            assigned[node.target.ident] = assigned.get(node.target.ident, 0) + 1
            body[node.target.ident] = body.get(node.target.ident, 0) + 1   # an ordinary (non-decl-init) write
    elif isinstance(node, cast.Decl):
        if node.init is not None:
            assigned[node.name] = assigned.get(node.name, 0) + 1           # a decl-init: counted, but NOT body
    elif isinstance(node, cast.Unary) and node.op == "&" and isinstance(node.operand, cast.Name):
        addr.add(node.operand.ident)
    for f in dataclasses.fields(node):
        _scan_mutations(getattr(node, f.name), assigned, body, addr)


class CLowerError(Exception):
    """A lowering error (an unsupported construct). `pos` is a source byte offset when known."""
    def __init__(self, message: str, pos: int | None = None):
        super().__init__(message)
        self.pos = pos


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
    member: bool = False                   # a member-array element `s.arr[i]` (carries offset+size even
                                           # at offset 0, distinct from a plain `base[idx]`)
    stride: int = 0                        # array-of-structs element stride (`s.arr[i].field`): the access
                                           # lands at &base + byte_off + idx*stride but copies ct.size bytes
                                           # (stride sizeof(element) != the field copy size). 0 == stride is ct.size.
    packed: bool = False                   # a bitfield in a PACKED struct: its storage unit spans only
                                           # ceil((bit_off+bit_width)/8) bytes, possibly straddling words

    def unit_bytes(self) -> int:
        """The bitfield storage-unit byte span: a PACKED field covers only the bytes it straddles; otherwise
        the declared type width (an MMIO register must be accessed at its natural width, so non-packed is
        unchanged)."""
        if self.packed and self.bit_width:
            return (self.bit_off + self.bit_width + 7) // 8
        return max(1, self.ct.size)


def _is_scalar_member_lv(target, lv: "_LV") -> bool:
    """A memory lvalue safe to store+reload as an assignment-EXPRESSION value: any SCALAR memory lvalue -- a
    direct/nested struct member `s.x` / `s.a.b`, an array element `a[i]`, a pointer deref `*p`, a BITFIELD
    member `s.bits`, OR a strided member (an array-of-structs element field `aos[i].f` / a member-array element
    `s.arr[i]`). The lvalue is resolved ONCE (its index/address captured), so the re-read returns the
    stored/converted value without re-running a side-effecting index. (A bitfield's value is the
    masked/sign-extended stored field via bf.get; an array-of-structs field re-reads the same strided element.
    MMIO/volatile lvalues are excluded at the `_assign` call sites -- a re-read would be an extra access.)"""
    return lv.kind == "mem" and lv.ct.kind == "scalar"


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


@dataclass
class ComputedGotoNode:
    target: int                            # `goto *<target>;` -- an indirect jump to a label address (GNU)


@dataclass
class SwitchNode:
    disc: int                              # the discriminant rid (lowered once)
    body: list                             # flat: CaseLabel | DefaultLabel | Claim | control nodes


@dataclass
class CaseLabel:
    value: int                             # `case <value>:` -- a folded integer constant


@dataclass
class DefaultLabel:
    pass                                   # `default:`


def _flatten_block(block: list) -> list:
    """All Claim objects in a body tree, in order (the flat single-phase view verify/plan use)."""
    out: list = []
    for node in block:
        if isinstance(node, IfNode):
            out += _flatten_block(node.then) + _flatten_block(node.els)
        elif isinstance(node, WhileNode):
            out += _flatten_block(node.cond_block) + _flatten_block(node.body) + _flatten_block(node.step)
        elif isinstance(node, SwitchNode):
            out += _flatten_block(node.body)
        elif isinstance(node, (Claim,)):
            out.append(node)
    return out


def own_footprint(lf, shared_min: int = 900000) -> tuple:
    """The function's *own* alias/effect footprint over shared (module-scope) resources -- the global
    rids it reads and writes directly (callee effects are folded in by the caller). Writes come from
    claim results (every store/copy of a global is a claim, so writes are exact). Reads come from
    claim operands *plus* the rids referenced by control conditions and `return`, which name a value
    without emitting a read claim -- so the read set stays sound for a bare `return g;` / `if (g)`."""
    reads = {r for c in lf.claims for r in c.rd if r >= shared_min}
    writes = {w for c in lf.claims for w in c.wr if w >= shared_min}
    if lf.return_rid is not None and lf.return_rid >= shared_min:
        reads.add(lf.return_rid)

    def walk(nodes):
        for n in nodes:
            if isinstance(n, ReturnNode):
                if n.rid is not None and n.rid >= shared_min:
                    reads.add(n.rid)
            elif isinstance(n, IfNode):
                if n.cond >= shared_min:
                    reads.add(n.cond)
                walk(n.then)
                walk(n.els)
            elif isinstance(n, WhileNode):
                if n.cond >= shared_min:
                    reads.add(n.cond)
                walk(n.cond_block)
                walk(n.body)
                walk(n.step)
            elif isinstance(n, SwitchNode):
                if n.disc >= shared_min:                    # the discriminant is a read
                    reads.add(n.disc)
                walk(n.body)

    walk(lf.body)
    return frozenset(reads), frozenset(writes)


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
    vla_locals: list = field(default_factory=list)  # (rid, name, CType) 1-D stack VLAs -- declared IN-BODY
    statics: list = field(default_factory=list)     # (rid, name, CType, init) static-storage locals
    globals_used: dict = field(default_factory=dict)   # rid -> name (file-scope globals referenced)
    zero_init_locals: set = field(default_factory=set)  # aggregate-local rids declared `= {0}`
    variadic: bool = False                # a trailing `...` after the named params (variadic function)
    reproducible: bool = False            # a C23 `[[reproducible]]`/`[[unsequenced]]` hint is on the
                                          #   definition -- a fusion-legality signal (the hint is value-
                                          #   neutral, so the emit drops it). Default False == every
                                          #   un-annotated function, undisturbed.
    ptr_extent: dict = field(default_factory=dict)      # §5.12: pointer rid -> recovered extent (count) variable rid


@dataclass
class LoweredUnit:
    functions: dict                       # name -> LoweredFunc
    entry: str
    aggregates: dict                      # tag -> CType
    compose_functions: dict               # name -> compose.Function
    resources: dict                       # rid -> Resource (whole unit)


# the rid `_call` returns for a void callee -- never read as a value (a void call is a statement); the
# Return lowering maps it back to a bare `return;`.
_VOID_RID = -1


class _FuncLowerer:
    def __init__(self, func: cast.Func, aggregates: dict, base_rid: int, cid: list,
                 genv: dict | None = None, gres: dict | None = None, strctr: list | None = None,
                 func_rets: dict | None = None, abi=None):
        self.func = func
        self.abi = abi or HOST            # the target data model for laying out user types (long/ptr)
        self.func_rets = func_rets or {}  # callee name -> return CType (for void / wide / float returns)
        self.aggregates = aggregates
        self.rid = base_rid
        self.cid = cid                    # shared mutable [next_claim_id]
        self.genv = genv or {}            # file-scope globals: name -> (rid, CType)
        self.gres = gres or {}            # global rid -> Resource
        self.strctr = strctr if strctr is not None else [0]   # shared string-literal counter (unique rids)
        self.str_globals: dict[int, str] = {}          # string-literal global rid -> the source spelling
        self.str_pool: dict[str, int] = {}             # spelling -> rid (dedup identical literals)
        self.func_globals: dict[int, str] = {}         # function-as-value rid -> the function name (emit verbatim)
        self.func_pool: dict[str, int] = {}            # function name -> rid (dedup)
        self.env: dict[str, tuple[int, CType]] = {}    # name -> (storage rid, type)
        self.resources: dict[int, Resource] = {}
        self.rtypes: dict[int, CType] = {}             # rid -> CType (for emission)
        self.params: list = []
        self.locals: list = []                         # (rid, name, CType) mutable named locals
        self.vla_locals: list = []                     # (rid, name, CType) 1-D stack VLAs -- declared IN-BODY
                                                       #   (`T a[__ext];`) at the source decl, NOT up front
        self.vla_strides: dict = {}                    # multi-dim VLA array rid -> per-dim snapshot rids (the
                                                       #   runtime Horner multipliers: m[i][j] -> i*dim1 + j)
        self.zero_init: set = set()                    # aggregate-local rids that emit a `= {0}` baseline
        self.statics: list = []                        # (rid, name, CType, init) static-storage locals
        self.calls: list = []
        self.block_stack: list = [[]]                  # claims/control nodes append to the top block
        self.loop_ctr = 0                              # unique loop ids (the `continue` label numbering)
        self.cl_ctr = 0                                # unique compound-literal ids (anonymous `_cl<N>` locals)
        # §5.12 recoverable extents: a pointer local bound to malloc(N*sizeof(T)) / calloc(N, sizeof(T)) carries
        # a RECOVERED element-count -> its `p[i]` accesses promote to `masked` (runtime-bounds-checked against N).
        self.ptr_extent: dict[int, int] = {}           # pointer rid -> the count VARIABLE's rid (re-emitted by name)
        self._mut_assigned: dict[str, int] = {}        # name -> TOTAL assignments incl. decl-init (pointer gate)
        self._mut_body: dict[str, int] = {}            # name -> NON-decl-init assignments only (count gate)
        self._mut_addr: set[str] = set()               # names whose address is taken anywhere (`&x`)
        self._ext_ctr = 0                              # unique hidden extent-snapshot locals (`__bcir_extK`)

    def _next_loop_id(self) -> int:
        self.loop_ctr += 1
        return self.loop_ctr

    # --- resource allocation ---
    def _new_rid(self) -> int:
        self.rid += 1
        return self.rid

    def _resolve_type(self, tref: cast.TypeRef) -> CType:
        if tref.vla is not None:                           # `T a[n]` (runtime size): native lowering is a
            raise CLowerError(                             # follow-on (the emitter's up-front local declaration
                "variable-length array native lowering is a follow-on")   # can't size it) -- route to fallback
        if tref.funcptr:                                   # a function-pointer alias (HAL dispatch)
            ret = self._resolve_type(tref.func_ret)
            params = tuple(self._resolve_type(p) for p in tref.func_params)
            return funcptr(tref.base, ret, params, self.abi)
        if tref.typeof_var:                                # `typeof(var)` -> the in-scope variable's type
            if tref.typeof_var not in self.env:
                raise CLowerError(f"typeof of unknown variable {tref.typeof_var!r}")
            base = self.env[tref.typeof_var][1]
        elif tref.typeof_expr is not None:                 # `typeof(expr)` -> the operand's static type
            base = self._type_of(tref.typeof_expr)
        elif tref.aggregate:
            base = self.aggregates[tref.base]
        elif tref.base == "va_list":                       # the <stdarg.h> variadic cursor (opaque)
            base = valist(self.abi)
        elif tref.bit_width:                               # C23 `_BitInt(N)`: an exact-width, non-promoting int
            base = bitint(tref.bit_width, signed="unsigned" not in tref.base)
        elif is_scalar_name(tref.base):
            base = scalar(tref.base, self.abi)
        else:
            raise CLowerError(f"unknown type {tref.base!r}")
        if "volatile" in tref.quals:                       # volatile pointee/object -> MMIO region
            base = with_volatile(base)
        if "_Atomic" in tref.quals:                        # _Atomic-qualified object (C11/C23)
            base = with_atomic(base)
        t = base
        for _ in range(tref.ptr):
            t = pointer(t, self.abi)
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
              bounds: str = "strict", hazard: str = "unique", lifetime=None) -> int:
        self.cid[0] += 1
        self.block_stack[-1].append(Claim(
            id=self.cid[0], opcode=opcode, lane=lane, stride_class=stride, count=count,
            rd=tuple(rd), wr=tuple(wr), imm=tuple(imm), domain=domain, op=op, bounds=bounds,
            hazard=hazard, lifetime=lifetime))
        return wr[0] if wr else -1

    def _storage(self, ct: CType, name: str) -> int:
        """A mutable named local — assignments write it, reads read it (so control-flow merges and
        loop accumulators work). The emitter declares it and refers to it by name."""
        rid = self._temp(ct, name)
        self.locals.append((rid, name, ct))
        return rid

    def _vla_storage(self, ct: CType, name: str, ext_rid: int) -> int:
        """A 1-D stack VLA `T a[n]`. Unlike a normal local it CANNOT be declared up front — its size is the
        runtime value already captured in `ext_rid` (`__bcir_extK`), known only once execution reaches the
        declaration. So it is registered in `self.vla_locals` (named, but NOT in the up-front decls) and a
        `c.vladecl` claim is emitted at THIS point in the body, which the emitter renders as the in-body
        `T a[__bcir_extK];` (a faithful stack VLA — no heap, no leak). Its `a[i]` bounds are masked against
        `ext_rid` via `ptr_extent` (§5.12), exactly like a recovered-extent pointer."""
        rid = self._temp(ct, name)
        self.vla_locals.append((rid, name, ct))
        self._emit("c.vladecl", Opcode.ADD, (ext_rid,), (rid,))
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

    def _lookup(self, name: str, pos: int | None = None):
        """The (rid, type) of a declared name (locals + the file-scope globals merged into `env`), or
        a CLowerError naming the undeclared identifier -- so the diagnostics entry reports `use of
        undeclared identifier 'x'` rather than crashing with a bare KeyError. `pos` (the identifier's
        source offset, carried on the Name node) anchors a caret under the offending name."""
        if name in self.env:
            return self.env[name]
        raise CLowerError(f"use of undeclared identifier {name!r}", pos=pos)

    # --- lvalue resolution ---
    def _lvalue(self, node) -> "_LV":
        if isinstance(node, cast.Name):
            rid, ct = self._lookup(node.ident, node.pos)
            return _LV("var", rid, ct)
        if isinstance(node, cast.CompoundLiteral):        # `&(int){v}`, `(struct P){...}.field` -- an lvalue
            rid, ct = self._compound_literal(node)
            return _LV("var", rid, ct)
        if isinstance(node, cast.Index):
            # collect the (possibly nested) index chain down to the ultimate base, then flatten
            # row-major: m[i][j] on a `T m[A][B]` param -> the linear index i*B + j (Horner).
            idx_nodes, n = [], node
            while isinstance(n, cast.Index):
                idx_nodes.append(n.index)
                n = n.base
            idx_nodes.reverse()
            byte_off = 0
            mem_shape = mem_elem = None
            ptr_member = False
            if isinstance(n, cast.Member):
                # `s.arr[i]` / `s.m[i][j]` -- indexing a struct *member* array. The base is the enclosing
                # struct; the member's byte offset rides in the access imm beside the element size, and
                # the row-major flattened index is scaled by the element, so the element lands at
                # `&s + member_off + lin*elem_size` -- never the enclosing struct's offset 0.
                base_rid, struct_ct, base_off = self._addr(n.base)
                agg = struct_ct.of if n.arrow else struct_ct
                ftype, byte_off, _bo, _bw = agg.field(n.field)
                byte_off += 0 if n.arrow else base_off    # ride the enclosing offset (a non-first member array)
                if ftype.kind == "pointer":
                    # a *pointer* member subscripted (`s->p[i]` == `*(s->p + i)`): load the full pointer
                    # field, then index the loaded pointer like a plain `base[idx]` (no member offset).
                    # (Pointer-value slice 2b -- the load is an 8-byte pointer, not a truncating uint32.)
                    base_rid = self._read(_LV("mem", base_rid, ftype, byte_off=byte_off))
                    base_ct, byte_off, ptr_member = ftype, 0, True
                elif ftype.kind == "array":
                    dims, t = [], ftype                       # descend the (nested) member array:
                    while t.kind == "array":                  # per-dim sizes + the ultimate scalar element
                        dims.append(t.count)
                        t = t.of if t.of is not None else scalar("uint32_t")
                    if len(dims) > 3:
                        # the dim table (oracle shape / twin field.adims) holds at most 3 dims; defer deeper.
                        raise CLowerError(
                            f"indexing a >3-dimensional struct member array ('{n.field}') is not yet supported")
                    if len(idx_nodes) != len(dims):
                        # partial indexing of a member array (a row pointer `s.m[i]`): defer to fallback.
                        raise CLowerError(
                            f"partial indexing of a struct member array ('{n.field}') is not yet supported")
                    base_ct, mem_shape, mem_elem = ftype, tuple(dims), t
                else:
                    raise CLowerError(
                        f"indexing a non-array struct member ('{n.field}[...]') is not yet supported")
            else:
                base_rid, base_ct, byte_off = self._addr(n)   # a non-Member base: its accumulated offset
            idx_rids = [self._rvalue(ix) for ix in idx_nodes]
            shape = mem_shape if mem_shape is not None else base_ct.shape
            lin = idx_rids[0]
            vla_str = self.vla_strides.get(base_rid)          # a multi-dim VLA -> runtime dim multipliers
            for d in range(1, len(idx_rids)):
                if vla_str is not None:                       # dim d's snapshot rid (NO c.const -- the runtime
                    k = vla_str[d]                            # extent), so `m[i][j]` -> i*dim1 + j masks the total
                else:
                    dim = shape[d] if d < len(shape) else 1
                    k = self._temp(scalar("uint32_t"), f"k{dim}")
                    self._emit("c.const", Opcode.LOAD, (), (k,), imm=(dim,))
                m1 = self._temp(scalar("uint32_t"), "b_mul")
                self._emit("c.bin.mul", Opcode.MUL, (lin, k), (m1,))
                a1 = self._temp(scalar("uint32_t"), "b_add")
                self._emit("c.bin.add", Opcode.ADD, (m1, idx_rids[d]), (a1,))
                lin = a1
            elem = mem_elem if mem_elem is not None else (base_ct.of if base_ct.of else scalar("uint32_t"))
            return _LV("mem", base_rid, elem, idx=lin, byte_off=byte_off,
                       member=isinstance(n, cast.Member) and not ptr_member)
        if isinstance(node, cast.Member):
            if isinstance(node.base, cast.Index):                 # `arr[i].field` -- index into an ARRAY-OF-STRUCTS
                el = self._lvalue(node.base)                      # member, then descend into the struct element.
                if el.kind == "mem" and el.idx is not None and el.ct.kind in ("struct", "union"):
                    agg = el.ct                                   # the element struct: add the field offset, keep
                    ftype, foff, fbo, fbw = agg.field(node.field) # the runtime index, and stride by the element size
                    if fbw or ftype.kind != "scalar":             # a bitfield / nested / array / pointer element
                        raise CLowerError(                        # field is a follow-on -- both rails fall back
                            f"array-of-structs non-scalar element field ('.{node.field}') is not yet supported")
                    return _LV("mem", el.rid, ftype, idx=el.idx, byte_off=el.byte_off + foff,
                               bit_off=fbo, bit_width=fbw, member=True, stride=agg.size, packed=agg.packed)
            base_rid, base_ct, base_off = self._addr(node.base)
            agg = base_ct.of if node.arrow else base_ct
            ftype, byte_off, bit_off, bit_w = agg.field(node.field)
            off = byte_off if node.arrow else base_off + byte_off   # `->` resets to *base; `.` accumulates
            return _LV("mem", base_rid, ftype, byte_off=off, bit_off=bit_off, bit_width=bit_w, packed=agg.packed)
        if isinstance(node, cast.Unary) and node.op == "*":
            operand = node.operand
            if isinstance(operand, cast.Binary) and operand.op == "+":   # *(p + i) == p[i]
                return self._lvalue(cast.Index(operand.lhs, operand.rhs))
            base_rid, base_ct, base_off = self._addr(operand)
            return _LV("mem", base_rid, base_ct.of or scalar("uint32_t"), byte_off=base_off)
        raise CLowerError(f"not an lvalue: {type(node).__name__}")

    def _string_ptr(self, spelling: str) -> int:
        """A string literal -> an anonymous read-only `char[]` global (NUL-terminated); the value is a
        pointer to it (decay). The emitter renders references to it as the inline literal, which is
        Clang-equivalent, so no synthesized global declaration is needed. A wide/UTF prefix
        (`L`/`u`/`U`/`u8`) sets the element width (`wchar_t`/`char16_t`/`char32_t`/`char`)."""
        existing = self.str_pool.get(spelling)                # dedup: identical literals share a global
        if existing is not None:
            return existing
        prefix, _ = split_lit_prefix(spelling)
        elem = str_elem_size(prefix)                          # 1 (char/u8) / 2 (u) / 4 (L/U)
        nunits = _str_bytes(spelling) + 1                     # the decoded code units + the NUL
        idx = self.strctr[0]
        self.strctr[0] += 1
        rid = 970000 + idx
        self.gres[rid] = Resource(rid=rid, domain=Domain.RAM, elem_bytes=elem, shape=(nunits,),
                                  access="ro", data_gen=1, name=f"__str{idx}")
        self.rtypes[rid] = array(scalar("char" if elem == 1 else f"uint{elem * 8}_t"), nunits)
        self.str_globals[rid] = spelling                      # rid -> the literal spelling (for emit)
        self.str_pool[spelling] = rid
        return rid

    def _func_ptr_value(self, name: str) -> int:
        """A defined function used as a VALUE (function-to-pointer decay, `o->fn = g`): an anonymous funcptr
        'global' whose value is the function's address. The emitter renders the rid as the bare function name
        (C decays it to a pointer), so -- like a string literal -- no claim is emitted (parity-critical: the
        bare name must cost 0 claims on both rails)."""
        existing = self.func_pool.get(name)
        if existing is not None:
            return existing
        idx = self.strctr[0]                              # share the literal-global rid space (unique rids)
        self.strctr[0] += 1
        rid = 970000 + idx
        self.gres[rid] = Resource(rid=rid, domain=Domain.RAM, elem_bytes=self.abi.pointer_size, shape=(1,),
                                  access="ro", data_gen=1, name=name)
        self.rtypes[rid] = funcptr(name, self.func_rets[name], (), self.abi)
        self.func_globals[rid] = name                     # rid -> the function name (rendered verbatim by emit)
        self.func_pool[name] = rid
        return rid

    def _addr(self, node):
        """The (rid, type, byte_offset) of an aggregate/pointer base used by member/index access. The
        offset ACCUMULATES through nested value-struct/union members (`t.q.a` -- q's byte offset must ride
        into a's access); `cfront_nestmember` only ever nested through a *first* member, so the drop went
        unseen. A `->` resets to the pointee (the access goes through the pointer base), and a
        pointer-valued field used as a base is loaded (offset reset to 0)."""
        if isinstance(node, cast.Name):
            rid, ct = self._lookup(node.ident, node.pos)
            return rid, ct, 0
        if isinstance(node, cast.CompoundLiteral):        # `(struct P){...}.field` / indexing the literal
            rid, ct = self._compound_literal(node)
            return rid, ct, 0
        if isinstance(node, cast.StringLit):                  # "abc"[i] -> index the anonymous global
            rid = self._string_ptr(node.value)
            return rid, self.rtypes[rid], 0
        if isinstance(node, cast.Member):
            base_rid, base_ct, base_off = self._addr(node.base)
            agg = base_ct.of if node.arrow else base_ct
            ftype, byte_off, bit_off, bit_w = agg.field(node.field)
            off = byte_off if node.arrow else base_off + byte_off   # `->` resets to *base; `.` accumulates
            if ftype.kind == "pointer":                       # a pointer-valued field used as a *base*
                # (`*(s->p)`, `s->t->a`): load the full pointer (at its field offset) and use the loaded
                # pointer as the new base -- the same move as `*q` below. Returning (struct_rid, T*) would
                # instead deref the struct's own address as if it held the pointee. (Pointer-value 2b.)
                lv = _LV("mem", base_rid, ftype, byte_off=off, bit_off=bit_off, bit_width=bit_w, packed=agg.packed)
                return self._read(lv), ftype, 0
            return base_rid, ftype, off
        if isinstance(node, cast.Unary) and node.op == "*":   # `*q` as a base (`**pp`): load the pointer it
            rid = self._read(self._lvalue(node))              # holds, and use that loaded pointer as the base
            return rid, self.rtypes.get(rid, pointer(scalar("uint32_t"))), 0
        if isinstance(node, cast.CallExpr):                   # `mk(x).field` -- a struct-returning call's result is
            rid = self._call(node)                            # a by-value struct temp; address it for member access
            return rid, self.rtypes.get(rid, scalar("uint32_t")), 0
        raise CLowerError(f"unsupported base expression {type(node).__name__}")

    def _mmio(self, base_rid: int) -> bool:
        res = self.resources.get(base_rid) or self.gres.get(base_rid)
        return res is not None and res.domain == Domain.MMIO

    # --- rvalue lowering: returns the rid holding the value ---
    def _bin_result_type(self, op: str, a: int, b: int) -> CType:
        """The result type of a binary op over the values in rids `a`, `b` (their types read from
        `rtypes`). Driving the emitted temp's true C type makes the backend do signed-vs-unsigned and
        width-correct arithmetic (the old flat uint32 model did not)."""
        ta, tb = self.rtypes.get(a), self.rtypes.get(b)
        # C23 `_BitInt(N)` (non-promoting, exact width): same-type stays `_BitInt(N)`; a `_BitInt` op an
        # integer CONSTANT also stays `_BitInt(N)` (the constant converts to the bit-int type). The
        # constant case needs the rid (is this operand a literal?), so it is resolved HERE, not in the
        # type-only `_bin_result_type_ct`. A relational/logical op stays `int` (handled below); everything
        # else over a `_BitInt` mixed with a non-constant standard int / different width routes to fallback.
        if (ta is not None and ta.is_bitint) or (tb is not None and tb.is_bitint):
            if op not in ("<", ">", "<=", ">=", "==", "!=", "&&", "||"):
                bi = self._bitint_binop_type(op, a, b, ta, tb)
                if bi is not None:
                    return bi
        return self._bin_result_type_ct(op, ta, tb)

    def _bitint_binop_type(self, op: str, a: int, b: int, ta: CType | None, tb: CType | None):
        """The result `_BitInt(N)` type for an arithmetic/bitwise/shift op with a `_BitInt` operand, or
        raise `CLowerError` (route to fallback) for a mix outside the first-class subset. SUPPORTED:
          * same-type `_BitInt(N)` on both sides (-> `_BitInt(N)`);
          * a `_BitInt(A)` mixed with a NARROWER `_BitInt(B)`, a NARROWER standard-int VARIABLE, or an
            integer CONSTANT whose (post-promotion) literal type is narrower, where the C23
            usual-arithmetic-conversions result is the WIDER `_BitInt` (its width strictly exceeds the other
            operand's post-promotion width, so it wins the bit-precise rank). The result is that `_BitInt(N)`
            with its own signedness -- verified == Clang via the `_Generic` differential. (So `bi64 + 5`
            stays `_BitInt(64)`, but `bi8 + 5` -- whose Clang result is `int`, the constant `5` being an
            `int` of greater rank -- falls back; the constant carries its REAL literal type, not a forced
            bit-int conversion.) An explicitly-cast constant like `(_BitInt(8))3` is already a `_BitInt`
            value, so it takes the same-type path.
          * a shift `bi << k` keeps the `_BitInt` LEFT operand (a `_BitInt` does not promote).
        UNSUPPORTED (fallback): a `_BitInt` mixed with a standard int / constant / different `_BitInt` whose
        C23 result is a STANDARD integer type (the bit-precise operand does NOT win the rank -- equal or
        lesser width). Those collapse long/long-long in the value model, so they stay fallback (correctness
        over coverage). `a`/`b` (the rids) are unused now that the constant flows through the real model."""
        del a, b                                              # the constant case uses ta/tb's real literal type
        ba = ta is not None and ta.is_bitint
        if op in ("<<", ">>"):                                # shift: the (non-promoting) `_BitInt` left operand;
            if ba:                                            # the right operand may be any integer (it is not
                return ta                                     # part of the result type)
            raise CLowerError("a `_BitInt` shift count without a `_BitInt` left operand is not supported")
        # Apply the C23 6.3.1.8 model to the operands' REAL types, keeping ONLY the cases whose result is a
        # `_BitInt(N)`. A bare integer constant carries its true literal type (`int`/`long`/...), so the
        # rank comparison matches Clang exactly (`bi64 + 5` -> `_BitInt(64)`, `bi8 + 5` -> fallback).
        try:
            r = usual_arith_int(ta, tb, self.abi)
        except BitIntMix as e:
            raise CLowerError(str(e)) from e
        if not r.is_bitint:                                   # defensive: the subset must yield a `_BitInt`
            raise CLowerError("`_BitInt` arithmetic whose C23 result is a standard integer type")
        return r

    def _bin_result_type_ct(self, op: str, ta: CType | None, tb: CType | None) -> CType:
        """The result type of a binary op from its operand *types*. `+ - * /` over a float operand yield
        the *wider* float (float usual arithmetic conversions); a relational/logical op is `int`; a shift
        takes the promoted *left* operand; everything else (`+ - * / %`, bitwise) takes the integer usual
        arithmetic conversions over both operands. Shared by `_rvalue` (emitting) and `_type_of` (typeof)."""
        if op in ("+", "-", "*", "/"):
            floats = [t for t in (ta, tb) if t is not None and t.is_float]
            if floats:
                # complex usual arithmetic conversions: if ANY operand is complex the result is complex with
                # the wider ELEMENT float (`float _Complex + double` -> `double _Complex`); otherwise the
                # wider real float. Comparing element ranks makes this order-independent.
                if any(t.is_complex for t in floats):
                    er = max(_FLOAT_RANK.get(_elem_float_name(t), 1) for t in floats)
                    return scalar(f"{_RANK_FLOAT[er]} _Complex")
                return scalar(max(floats, key=lambda t: _FLOAT_RANK.get(t.name, 1)).name)
        if op in ("+", "-"):                                  # pointer arithmetic: p + i / i + p / p - i ->
            pa = ta if (ta is not None and ta.kind == "pointer") else None   # a pointer carrying the pointee.
            pb = tb if (tb is not None and tb.kind == "pointer") else None   # (p - q, both pointers, is a
            if (pa is None) != (pb is None):                                 # ptrdiff and stays integer below.)
                return pa or pb
        if op in ("<", ">", "<=", ">=", "==", "!=", "&&", "||"):
            return scalar("int", self.abi)                    # a relational / logical result is int
        ia = ta if (ta is not None and ta.is_integer) else scalar("int", self.abi)
        if op in ("<<", ">>"):
            return promote_int(ia, self.abi)                  # the shift result carries the promoted LHS type
        ib = tb if (tb is not None and tb.is_integer) else scalar("int", self.abi)
        # `_BitInt(N)` does not canonicalize: a same-type pair stays itself; any other mix raises BitIntMix,
        # which is an unsupported form here (the rid-aware `_bin_result_type` already handled the legal
        # `_BitInt op constant` case) -> route the whole unit to fallback rather than mis-type the result.
        try:
            return usual_arith_int(ia, ib, self.abi)
        except BitIntMix as e:
            raise CLowerError(str(e)) from e

    def _type_of(self, node) -> CType:
        """The static C type of an expression, computed WITHOUT evaluating or emitting it -- the operand
        of `typeof` is unevaluated (like `sizeof`). Mirrors the result types `_rvalue` assigns its temps,
        so `typeof(expr)` resolves to exactly the type Clang gives the expression. Scoped to the forms
        whose type both rails infer identically (the twin reads it off a speculatively-lowered value);
        calls / address-of / ternary are a deferred follow-on (raised here rather than silently mistyped)."""
        if isinstance(node, cast.IntLit):
            return scalar(node.ctype, self.abi) if is_scalar_name(node.ctype) else scalar("int", self.abi)
        if isinstance(node, cast.FloatLit):
            return _float_lit_type(node.value)
        if isinstance(node, cast.Name):
            return self._lookup(node.ident, node.pos)[1]
        if isinstance(node, cast.StringLit):                  # a string literal has type `char[N]` (not char*)
            prefix, _ = split_lit_prefix(node.value)
            elem = str_elem_size(prefix)
            n = _str_bytes(node.value) + 1                    # decoded code units + the NUL
            return array(scalar("char" if elem == 1 else f"uint{elem * 8}_t"), n)
        if isinstance(node, cast.Binary):
            return self._bin_result_type_ct(node.op, self._type_of(node.lhs), self._type_of(node.rhs))
        if isinstance(node, cast.Unary):
            if node.op == "*":                                # deref -> the pointee / element type
                t = self._type_of(node.operand)
                return t.of if t.of is not None else scalar("uint32_t")
            if node.op == "!":                                # logical not -> int (0/1)
                return scalar("int", self.abi)
            t = self._type_of(node.operand)                   # `-` / `~`: the integer-promoted operand
            if t.is_float:                                    # (a float stays float -- floats don't promote)
                return t
            if t.is_integer:
                return promote_int(t, self.abi)
            return scalar("uint32_t")
        if isinstance(node, (cast.Cast, cast.CompoundLiteral)):
            return self._resolve_type(node.type)
        if isinstance(node, cast.Member):                     # `s.f` / `p->f` -> the field's type
            base_t = self._type_of(node.base)
            agg = base_t.of if node.arrow else base_t
            return agg.field(node.field)[0]
        if isinstance(node, cast.Index):                      # `a[i]` -> the element (pointee) type
            base_t = self._type_of(node.base)
            return base_t.of if base_t.of is not None else scalar("uint32_t")
        if isinstance(node, cast.Generic):                    # _Generic -> the selected association's type
            return self._type_of(self._generic_select(node))
        if isinstance(node, cast.StmtExpr):                   # `({ ...; e; })` -> the type of the last expr
            if node.stmts and isinstance(node.stmts[-1], cast.ExprStmt):
                return self._type_of(node.stmts[-1].expr)
            return scalar("void")
        raise CLowerError(f"typeof of this expression form ({type(node).__name__}) is not yet supported")

    def _type_key(self, t: CType):
        """A canonical, comparable identity for a type used by `_Generic` matching (C11 §6.5.1.1, after
        lvalue/array decay; qualifiers do not affect the key). int / int32_t collapse (same width + sign);
        plain `char` stays distinct from signed/unsigned char; floats key on width; aggregates on tag."""
        if t.kind == "array":
            t = pointer(t.of, self.abi) if t.of is not None else pointer(scalar("uint32_t"), self.abi)
        if t.kind in ("struct", "union"):
            return (t.kind, t.name)
        if t.kind == "pointer":
            return ("ptr", self._type_key(t.of) if t.of is not None else ("void",))
        if t.kind == "valist":
            return ("valist",)
        if t.name == "void":
            return ("void",)
        if t.name in ("_Bool", "bool"):
            return ("bool",)
        if t.name == "char":                              # plain char -- a distinct type from signed/unsigned char
            return ("char",)
        if t.is_float:
            return ("float", t.size)
        return ("int", t.size, t.signed)

    def _generic_select(self, node: "cast.Generic"):
        """The chosen association's expression for a `_Generic` selection: the first type-name whose type
        matches the controlling expression's static type, else `default`. The controlling expression and
        the unselected associations are NOT lowered (only the controlling type is inspected)."""
        key = self._type_key(self._type_of(node.controlling))
        default = None
        for tref, expr in node.assocs:
            if tref is None:
                default = expr
            elif self._type_key(self._resolve_type(tref)) == key:
                return expr
        if default is not None:
            return default
        raise CLowerError("no _Generic association matches the controlling expression's type")

    def _rvalue(self, node) -> int:
        if isinstance(node, cast.IntLit):
            ct = scalar(node.ctype, self.abi) if is_scalar_name(node.ctype) else scalar("int", self.abi)
            t = self._temp(ct, f"k{node.value}")
            r = self._emit("c.const", Opcode.LOAD, (), (t,), imm=(node.value,))
            return r
        if isinstance(node, cast.FloatLit):                   # a float constant -> a typed c.fconst
            ct = _float_lit_type(node.value)                  # f/F -> float, l/L -> long double, else double
            t = self._temp(ct, "fk")
            return self._emit(f"c.fconst:{node.value}", Opcode.LOAD, (), (t,))
        if isinstance(node, cast.Name):
            if node.ident not in self.env:
                if node.ident in self.func_rets:
                    return self._func_ptr_value(node.ident)   # a function NAME as a value -> its funcptr
                if node.ident in _IMAG_UNIT:                  # <complex.h> imaginary unit (unless shadowed)
                    t = self._temp(scalar("float _Complex"), "imag_unit")
                    return self._emit(f"c.cconst:{node.ident}", Opcode.LOAD, (), (t,))
            return self._lookup(node.ident, node.pos)[0]
        if isinstance(node, cast.StringLit):                  # a string value -> the global pointer
            return self._string_ptr(node.value)
        if isinstance(node, cast.Binary) and node.op == ",":
            self._rvalue(node.lhs)                            # the comma operator: evaluate the left operand for
            return self._rvalue(node.rhs)                     # its side effects, discard it, yield the right value
        if isinstance(node, cast.IncDec):
            return self._incdec_value(node)
        if isinstance(node, cast.LabelAddr):              # `&&L` -- a label's address as a `void *` value (GNU)
            t = self._temp(pointer(scalar("void")), f"labeladdr_{node.label}")
            return self._emit(f"c.labeladdr:{node.label}", Opcode.LOAD, (), (t,))
        if isinstance(node, cast.Binary):
            a, b = self._rvalue(node.lhs), self._rvalue(node.rhs)
            opcode, suf = _BIN[node.op]
            # float arithmetic propagates the (wider) float type; comparisons/bitwise stay int. The
            # actual IEEE-754 math is delegated to the emitted C / resident backend (never computed here).
            rt = self._bin_result_type(node.op, a, b)
            t = self._temp(rt, f"b_{suf}")
            return self._emit(f"c.bin.{suf}", opcode, (a, b), (t,))
        if isinstance(node, cast.Unary):
            if node.op in ("__real__", "__imag__"):       # GNU complex part -> the real element float
                v = self._rvalue(node.operand)
                vt = self.rtypes.get(v)
                if vt is not None and vt.is_complex:
                    rt = scalar({4: "float", 8: "double", 16: "long double"}.get(vt.size // 2, "double"))
                elif vt is not None and vt.is_float:       # __real__ of a real float is the value itself
                    rt = vt
                else:
                    rt = scalar("double")
                suf = "creal" if node.op == "__real__" else "cimag"
                t = self._temp(rt, f"u_{suf}")             # emitted `__real__ x` -- not integer-computed
                return self._emit(f"c.un.{suf}", Opcode.GEM_DISPATCH, (v,), (t,))
            if node.op == "*":
                return self._read(self._lvalue(node))
            if node.op == "&":
                # `&lvalue` as a *value* (a call argument, a pointer initializer, an out-param): a pointer
                # temp whose address is delegated to the emitted C, never computed here. Resolving through
                # `_lvalue` (the same path loads/stores use) makes it general AND correct for a base reached
                # THROUGH a pointer: `&s->m` takes the pointer's address (`(char*)s + off`), not `&s` -- the
                # bug the old `_addr`-with-hardcoded-`&` emit had (it computed `&s + off`, the address of the
                # pointer VARIABLE). The emit uses `_base_ptr` (decays a pointer/array base, addresses a value
                # base), so every form lands the right byte address.
                operand = node.operand
                if isinstance(operand, cast.Unary) and operand.op == "*":  # &*p == p (the pointer itself;
                    return self._rvalue(operand.operand)                   # &*(p+i) == p+i) -- 0 claims, both rails
                lv = self._lvalue(operand)
                if lv.bit_width:                              # &bitfield is illegal in C (no addressable unit)
                    raise CLowerError("cannot take the address of a bit-field")
                if lv.kind == "mem" and lv.idx is None and lv.ct.kind == "array":
                    raise CLowerError(                        # &s.arr -- the result is a pointer-TO-ARRAY
                        "address-of an array member is not yet supported")  # (needs (*)[] declarators; follow-on)
                t = self._temp(pointer(lv.ct), "addr")
                if lv.kind == "var":                          # &name / &(compound literal) -> the object's address
                    return self._emit("c.addrof", Opcode.ADD, (lv.rid,), (t,))
                if lv.idx is not None:                        # &arr[i] / &s.arr[i] / &arr[i].field
                    stride = lv.stride or (lv.ct.size or 4)   # array-of-structs strides by the STRUCT, else the elem
                    return self._emit("c.addrof", Opcode.ADD, (lv.rid, lv.idx), (t,),
                                      imm=(lv.byte_off, stride))
                return self._emit("c.addrof", Opcode.ADD, (lv.rid,), (t,), imm=(lv.byte_off,))  # &base.member
            v = self._rvalue(node.operand)
            opcode, suf = _UN[node.op]
            if node.op == "!":                                # logical not -> int (0/1)
                rt = scalar("int", self.abi)
            else:                                             # `-` / `~`: the promoted operand type, so
                vt = self.rtypes.get(v)                       # negating a `long` stays 64-bit, and `-x` on
                if vt is not None and vt.is_float:            # a float stays float (floats don't promote --
                    rt = vt                                   # was forced to uint32, truncating -2.5 -> 4.29e9)
                elif vt is not None and vt.is_integer:        # integer promotion: a sub-int operand (e.g.
                    rt = promote_int(vt, self.abi)            # `unsigned char`) becomes signed int -> ~c is -1
                else:
                    rt = scalar("uint32_t")
            t = self._temp(rt, f"u_{suf}")
            return self._emit(f"c.un.{suf}", opcode, (v,), (t,))
        if isinstance(node, cast.Cast):
            v = self._rvalue(node.operand)
            ct = self._resolve_type(node.type)
            # a float cast target types the temp float (so downstream arithmetic is float, and the emit
            # declares it float); an integer cast to a uint32 temp reproduces integer-promotion (a
            # narrowing cast masks/zero-extends back), so either way the result matches Clang.
            t = self._temp(ct if (ct.is_integer or ct.is_float) else scalar("uint32_t"), "cast")
            # A float -> signed-integer conversion needs a SIGNED cast operator: the canonical unsigned
            # name (uint32_t / uint8_t) makes it float -> unsigned, which is UB for a negative value and
            # diverges by target (x86 wraps, aarch64 saturates to 0) -- and even on x86 a sub-int signed
            # target loses the sign. Emit the signed fixed-width operator so `(int)(-5.0f)` is -5.
            vt = self.rtypes.get(v)
            cname = (_CAST_W_SIGNED.get(ct.size, "int32_t")
                     if (vt is not None and vt.is_float and ct.is_integer and ct.signed and not ct.is_bitint)
                     else _cast_name(ct))   # a `_BitInt` target keeps its exact spelling (no width-named fallback)
            return self._emit(f"c.cast:{cname}", Opcode.ADD, (v,), (t,))
        if isinstance(node, (cast.Index, cast.Member)):
            return self._read(self._lvalue(node))
        if isinstance(node, cast.CompoundLiteral):        # `(struct P){a,b}` by value / `(int){v}` as a value
            return self._compound_literal(node)[0]
        if isinstance(node, cast.VaArg):                  # va_arg(ap, T) -> the next variadic argument (type T)
            ap = self._rvalue(node.ap)                    # the va_list cursor (by name)
            vt = self._resolve_type(node.type)            # the result has type T (drives the temp + emit)
            t = self._temp(vt, "vaarg")
            return self._emit("c.call.vaarg", Opcode.GEM_DISPATCH, (ap,), (t,))
        if isinstance(node, cast.Generic):                # _Generic(ctrl, T: e, ..., default: e) -- select on
            return self._rvalue(self._generic_select(node))   # ctrl's static type; only the chosen e is lowered
        if isinstance(node, cast.StmtExpr):               # `({ s1; ...; e; })` -> lower the prefix stmts inline
            if not node.stmts:                            # (its own scope), then yield the last expr's value
                raise CLowerError("empty statement expression")
            saved_env = dict(self.env)
            for s in node.stmts[:-1]:
                self._stmt(s)
            last = node.stmts[-1]
            if isinstance(last, cast.ExprStmt) and isinstance(last.expr, cast.Assign):
                # the last item being an assignment (incl. the `i++`/`++i` desugar) is outside the subset
                # as a *value*: the twin's value-expression grammar has no assignment, and a trailing `i++`
                # discarded its post/pre distinction. Defer to the backend rather than diverge / guess wrong.
                raise CLowerError("assignment as a statement-expression value")
            if isinstance(last, cast.ExprStmt):
                val = self._rvalue(last.expr)
            else:                                         # a non-expression last stmt -> void (rarely used)
                self._stmt(last)
                val = _VOID_RID
            self.env = saved_env
            return val
        if isinstance(node, cast.Assign):
            return self._assign(node)
        if isinstance(node, cast.SizeOf):
            # sizeof folds to a compile-time constant -- the operand is NOT evaluated.
            if node.type is not None:
                size = self._resolve_type(node.type).size
            elif isinstance(node.expr, cast.StringLit):
                prefix, _ = split_lit_prefix(node.expr.value)
                size = (_str_bytes(node.expr.value) + 1) * str_elem_size(prefix)   # units incl. NUL × width
            elif isinstance(node.expr, cast.Name):
                rid, ct = self._lookup(node.expr.ident, node.expr.pos)
                if ct.kind == "array" and ct.count == 0 and rid in self.ptr_extent:
                    # `sizeof a` of a 1-D stack VLA is a RUNTIME value: the snapshot extent × sizeof(element).
                    # (Opcode.ADD: a cost hint only — the emit string carries the multiply, like c.copy/c.vladecl.)
                    ext = self.ptr_extent[rid]
                    t = self._temp(scalar("size_t", self.abi), "szof")
                    return self._emit("c.sizeof.vla", Opcode.ADD, (ext,), (t,), imm=(ct.of.size,))
                size = ct.size                                  # the variable's declared (static) type size
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
            # a select over ARITHMETIC arms carries their common type (usual arithmetic conversions), NOT a
            # blanket uint32_t -- otherwise a signed arm loses its sign (a downstream `>>`/compare on the
            # select goes unsigned) and a FLOAT arm is truncated to int (`(c?x:y)` of doubles becomes an
            # int, dropping the value / mis-converting a nan). (pointer arms keep the 4-byte unit.)
            ta, tb = self.rtypes.get(a), self.rtypes.get(b)
            if (ta is not None and tb is not None and (ta.is_integer or ta.is_float)
                    and (tb.is_integer or tb.is_float)):
                rt = self._bin_result_type_ct("+", ta, tb)
            else:
                rt = scalar("uint32_t")
            t = self._temp(rt, "sel")
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
        base_rid, base_ct, _base_off = self._addr(m.base)    # a dispatch base is a pointer (offset 0)
        actuals = tuple(self._rvalue(a) for a in node.args)
        agg = base_ct.of if m.arrow else base_ct             # the struct carrying the funcptr field
        try:                                                 # the member's funcptr CType -> its return type
            fct = agg.field(m.field)[0] if agg is not None else None
        except KeyError:
            fct = None
        ret_ct = fct.of if (fct is not None and fct.kind == "funcptr") else None
        t = self._temp(self._call_result_ct(ret_ct), f"icall_{m.field}")   # a signed member return reads back signed
        return self._emit(f"c.call.imember:{m.field}", Opcode.GEM_DISPATCH,
                          (base_rid, *actuals), (t,), imm=(1 if m.arrow else 0,))

    # --- memory read/write, with bitfield (mask/shift) + MMIO (ordered) handling ---
    def _read(self, lv: "_LV") -> int:
        if lv.kind == "var":
            return lv.rid
        unit = self._load_unit(lv)
        if lv.bit_width:                                     # bitfield extract: (unit >> off) & mask
            signed = lv.ct.is_integer and lv.ct.signed       # a signed bitfield read sign-extends from bit w-1
            # integer promotion (6.3.1.1): a bitfield narrower than int promotes to int (int holds all its
            # values), so an UNSIGNED sub-int bitfield reads as a SIGNED int -- `bf < x` is a signed compare,
            # not an unsigned one. A full-width (== 32) unsigned bitfield stays unsigned; a WIDE bitfield
            # (> 32 bits, in a long-long unit) keeps its declared 64-bit type -- int/unsigned can't hold it.
            rt = (lv.ct if lv.bit_width > 32
                  else scalar("int" if (signed or lv.bit_width < 32) else "uint32_t", self.abi))
            t = self._temp(rt, "bf")
            return self._emit("c.bf.get", Opcode.ADD, (unit,), (t,),
                              imm=(lv.bit_off, lv.bit_width, int(signed)))
        return unit

    def _access_bounds(self, lv: "_LV") -> str:
        """The bounds contract for a load/store (§5.12 bounds-promotion). An INDEXED access into a
        LOCAL/STATIC array OBJECT -- whose extent is statically RECOVERABLE from the resource shape -- is
        promoted from `assumed_safe` (trusted) to `masked` (runtime-bounds-checked): the access declares that
        its index is checked against the known extent, the contract the quarantine handler discharges. A
        pointer/MMIO base (extent unknown / a device register) and a non-indexed access stay `assumed_safe`.
        A string-LITERAL base stays `assumed_safe` too: it is anonymous read-only data (no named object the
        debugger/ML-layer would surface), and the twin does not promote it -- so excluding it keeps the rails
        in lockstep. A naked POINTER with a RECOVERED extent (malloc/calloc'd, `self.ptr_extent`) also promotes
        -- the runtime-checked extent is the allocation's element count. This is metadata only -- no
        emit/behaviour change; `verify` already defaults to `bounds`."""
        if (lv.kind == "mem" and lv.idx is not None and not lv.member
                and not self._mmio(lv.rid) and lv.rid not in self.str_globals):
            rt = self.rtypes.get(lv.rid)
            if rt is not None and rt.kind == "array" and rt.count:
                return "masked"                       # a known-extent local/static array -> runtime-bounds-checked
            if lv.rid in self.ptr_extent:
                return "masked"                       # a malloc/calloc pointer with a recovered element count
        return "assumed_safe"

    def _alloc_count_node(self, call, pointee_size: int):
        """The element-COUNT AST node of a recoverable `malloc(N*sizeof(T))` / `malloc(sizeof(T)*N)` /
        `calloc(N, sizeof(T))` / `malloc(N)` (1-byte pointee) -- N, where the size operand is `sizeof(T)`
        (T the pointee) or its byte literal. Else None. Recognition only; the binding decides how to capture
        N (a stable Name re-emitted by name, or an expression SNAPSHOTTED). (§5.12)"""
        if not isinstance(call, cast.CallExpr) or pointee_size <= 0:
            return None

        def size_bytes(e):                                   # a per-element size operand -> its byte width
            if isinstance(e, cast.SizeOf) and e.type is not None and e.expr is None:
                try:
                    return self._resolve_type(e.type).size
                except CLowerError:
                    return None
            return e.value if isinstance(e, cast.IntLit) else None

        if call.callee == "calloc" and len(call.args) == 2:
            return call.args[0] if size_bytes(call.args[1]) == pointee_size else None
        if call.callee == "malloc" and len(call.args) == 1:
            a = call.args[0]
            if isinstance(a, cast.Binary) and a.op == "*":
                if size_bytes(a.rhs) == pointee_size:
                    return a.lhs
                if size_bytes(a.lhs) == pointee_size:
                    return a.rhs
                return None
            if pointee_size == 1:                            # a byte buffer: malloc(N) bytes == N elements
                return a
        return None

    def _bind_extent(self, p_rid: int, p_ct: "CType", p_name: str, init) -> None:
        """Bind a recovered element-count to a malloc/calloc'd pointer local, so its `p[i]` accesses promote
        to `masked`. Only when the POINTER itself is stable -- assigned exactly once (this binding) and never
        address-taken (a `p = realloc(...)` reassigns it, count 2, and is left unmanaged). The COUNT is then
        captured one of two ways:
          * a STABLE integer-count Name (no ordinary post-alloc write, not aliased) -- re-emitted BY NAME at
            the access, byte-identical to the pre-snapshot path (the common `malloc(m*sizeof)` case);
          * any other side-effect-free COUNT EXPRESSION (`(n+1)`, `a*b`, ...) -- evaluated once into a hidden
            immutable local `__bcir_extK` at the alloc, immune to any later mutation of its inputs (the
            SNAPSHOT). Re-evaluating it for the snapshot is sound exactly because it is pure. (§5.12)"""
        if p_ct.kind != "pointer" or p_ct.of is None:
            return
        if self._mut_assigned.get(p_name, 0) != 1 or p_name in self._mut_addr:
            return
        count = self._alloc_count_node(init, p_ct.of.size)
        if count is None:
            return
        if isinstance(count, cast.Name) and count.ident in self.env:       # the stable-Name fast path
            rid, ct = self.env[count.ident]
            if (ct.kind == "scalar" and not getattr(ct, "is_float", False)
                    and self._mut_body.get(count.ident, 0) == 0 and count.ident not in self._mut_addr):
                self.ptr_extent[p_rid] = rid
            return                                                         # a Name is captured by name or not at all
        if _is_pure(count):                                                # an EXPRESSION count -> snapshot it
            v = self._rvalue(count)
            vct = self.rtypes.get(v)
            if vct is not None and vct.kind == "scalar" and not getattr(vct, "is_float", False):
                ext = self._storage(vct, f"__bcir_ext{self._ext_ctr}")
                self._ext_ctr += 1
                self._emit("c.copy", Opcode.ADD, (v,), (ext,))
                self.ptr_extent[p_rid] = ext

    def _load_unit(self, lv: "_LV") -> int:
        if lv.bit_width:
            # a bitfield's storage unit is read as an unsigned wide enough to hold it (a `long long` field, or
            # a packed field straddling into bits >= 32, needs 64 bits, not a uint32); only the `unit_bytes()`
            # spanned bytes are read (a PACKED field's unit may not fill the whole declared type / may straddle
            # words), into a zeroed temp so the unread high bytes are 0.
            ub = lv.unit_bytes()
            t = self._temp(scalar("uint32_t" if ub <= 4 else "uint64_t", self.abi), "ld")
            rd = (lv.rid,) if lv.idx is None else (lv.rid, lv.idx)
            mmio = self._mmio(lv.rid)
            return self._emit("c.load", Opcode.LOAD, rd, (t,), imm=(lv.byte_off, ub),
                              domain=Domain.MMIO if mmio else Domain.RAM, bounds=self._access_bounds(lv),
                              lane=Lane.H if mmio else Lane.U, hazard="barriered" if mmio else "unique")
        unit_ct = lv.ct
        t = self._temp(unit_ct, "ld")
        rd = (lv.rid,) if lv.idx is None else (lv.rid, lv.idx)
        mmio = self._mmio(lv.rid)
        # the byte offset rides in imm (not the strict-bounds `offset`), and the access is
        # assumed_safe (the frontend resolved the member/index). MMIO accesses are ordered. A member
        # array `s.arr[i]` carries BOTH (member offset, element size) and an index -- so the emit lands
        # the element at `&s + member_off + i*elem_size`, distinct from a plain `base[idx]` (no imm).
        if lv.idx is not None:
            imm = ((lv.byte_off, max(1, lv.ct.size)) + ((lv.stride,) if lv.stride else ())   # member array: even
                   if lv.member else ())                # off 0; array-of-structs adds the element stride (imm[2])
        else:
            imm = (lv.byte_off,) if lv.byte_off else ()
        return self._emit("c.load", Opcode.LOAD, rd, (t,), imm=imm,
                          domain=Domain.MMIO if mmio else Domain.RAM, bounds=self._access_bounds(lv),
                          lane=Lane.H if mmio else Lane.U,
                          hazard="barriered" if mmio else "unique")

    def _write(self, lv: "_LV", v: int) -> None:
        if lv.bit_width:                                     # read-modify-write the storage unit
            old = self._load_unit(lv)
            t = self._temp(scalar("uint32_t" if lv.unit_bytes() <= 4 else "uint64_t", self.abi), "bf")
            v = self._emit("c.bf.set", Opcode.ADD, (old, v), (t,), imm=(lv.bit_off, lv.bit_width))
        mmio = self._mmio(lv.rid)
        if lv.idx is None:                                    # member/deref: carry (offset, size) -- a packed
            rd, imm = (lv.rid, v), (lv.byte_off, lv.unit_bytes())   # bitfield writes only its spanned bytes back
        elif lv.member:                                       # s.arr[i]: (member offset, element size)
            rd, imm = (lv.rid, lv.idx, v), (lv.byte_off, max(1, lv.ct.size))
        else:                                                 # base[idx]: a typed array store
            rd, imm = (lv.rid, lv.idx, v), ()
        if imm and lv.ct.name in ("_Bool", "bool"):           # a _Bool slot: flag it so the emit normalizes
            imm = imm + (1,)                                  # the stored value (any nonzero -> 1, §6.3.1.2)
        if lv.stride:                                         # array-of-structs element store: land at
            if len(imm) == 2:                                 # &base + off + idx*stride but copy ct.size bytes --
                imm = imm + (0,)                              # keep (off,size,bool_flag,stride): pad the flag slot
            imm = imm + (lv.stride,)                          # (0 == not a _Bool field) so stride is always imm[3]
        self._emit("c.store", Opcode.STORE, rd, (), imm=imm,
                   domain=Domain.MMIO if mmio else Domain.RAM, bounds=self._access_bounds(lv),
                   lane=Lane.H if mmio else Lane.U,
                   hazard="barriered" if mmio else "unique")

    def _compound_literal(self, node: "cast.CompoundLiteral") -> tuple[int, CType]:
        """Materialize a compound literal `(type){init}` as an anonymous local and initialize it, exactly
        like a braced local decl -- a struct/union/array reuses the `_agg_init` zero-baseline + per-member
        store path; a scalar `(int){v}` copies the single value in. Returns (rid, type) so the result acts
        as an lvalue (address-of / member access) and as an rvalue (a by-value struct arg / scalar read)."""
        ct = self._resolve_type(node.type)
        if len(node.type.array or ()) > 1:
            # A MULTI-DIM `(T[a][b]){...}` literal: `_resolve_type` nests array-of-array and keeps `shape=()`,
            # so the storage is under-sized and the `[i][j]` stride wrong (the #500 silent miscompile). Rebuild
            # it the SAME way the regular multi-dim declarator does (lower.py ~1751): a FLAT `array(leaf, total)`
            # carrying the per-dim `shape`, so `_array_row` descends nested braces by row and `_lvalue` flattens
            # `[i][j]` row-major via `shape`. Capped at 3 dims (the shape / twin-adims table holds 3).
            elem = self._resolve_type(replace(node.type, array=()))   # the scalar leaf
            dims = node.type.array
            if len(dims) > 3:
                raise CLowerError("a multi-dimensional array compound literal of more than 3 dims is not supported")
            if dims[0] in (0, None):                      # `(T[][N]){...}` -- infer the OUTER dim from the init
                n, cursor = 0, 0                          # (max top-level index + 1; positional advances, `[i]=` jumps)
                for key, _expr in node.init.entries:
                    if isinstance(key, tuple):            # a nested chain `[i]...` -> its outer array index
                        idx = key[0][1] if key and key[0][0] == "a" else cursor
                    else:
                        idx = key if isinstance(key, int) else cursor
                    cursor = idx + 1
                    n = max(n, cursor)
                dims = (n or 1,) + tuple(dims[1:])        # inferred outer + the fixed inner dims
            total = 1
            for d in dims:
                total *= d
            ct = replace(array(elem, total), shape=tuple(dims))
        if ct.kind == "array" and len(node.type.array or ()) <= 1 and ct.count == 0:
            n, cursor = 0, 0                              # `(T[]){...}` -- infer the length from the init
            for key, _expr in node.init.entries:         # (max index + 1; positional advances, `[i]=` jumps)
                if isinstance(key, tuple):               # a nested chain `[i]...` -> its outer array index
                    idx = key[0][1] if key and key[0][0] == "a" else cursor
                else:
                    idx = key if isinstance(key, int) else cursor
                cursor = idx + 1
                n = max(n, cursor)
            ct = array(ct.of, n or 1)
        self.cl_ctr += 1
        rid = self._storage(ct, f"_cl{self.cl_ctr}")
        if ct.kind in ("struct", "union", "array"):
            self._agg_init(rid, ct, node.init)
        else:                                             # a scalar compound literal: one positional value
            if node.init.entries:
                v = self._rvalue(node.init.entries[0][1])
            else:                                         # `(int){}` (C23 empty init) -> zero
                v = self._temp(scalar("int", self.abi), "clz")
                self._emit("c.const", Opcode.LOAD, (), (v,), imm=(0,))
            self._emit("c.copy", Opcode.ADD, (v,), (rid,))
        return rid, ct

    def _array_row(self, ct: CType) -> "tuple[CType, int]":
        """For a FLAT multi-dim array object `T a[d0][d1]...` (the `array(leaf, total)` + `shape=(d0,d1,..)`
        representation), the *element* of the outermost dim is a ROW sub-array `T row[d1]...`, NOT the scalar
        leaf `ct.of`. Synthesize that row sub-type (carrying `shape[1:]`) and its byte stride (product of the
        inner dims * leaf size) so a nested brace `{...}` descends by row. A 1-D array (or no `shape`) keeps
        the scalar element + leaf-sized stride."""
        leaf = ct.of or scalar("uint32_t")
        if len(ct.shape) <= 1:                                # 1-D: element IS the scalar leaf
            return leaf, leaf.size
        inner = ct.shape[1:]
        stride = leaf.size
        for d in inner:
            stride *= d
        row = replace(array(leaf, stride // leaf.size), shape=tuple(inner))   # `T row[d1*..]`, carrying shape[1:]
        return row, stride

    def _agg_init(self, rid: int, ct: CType, ag: cast.AggInit) -> None:
        """Lower a braced aggregate initializer to a `= {0}` zero baseline (so uninitialized members
        zero-fill, §6.7.10) + a store per initialized member/element, reusing the member/array store
        path (`_write`). Positional entries advance a cursor; `[i]=` / `.field=` designators jump it."""
        self.zero_init.add(rid)
        cursor = 0
        for key, expr in ag.entries:
            # a member/element that is itself an aggregate can take a NESTED brace `{ {e0,e1,..}, n }`
            # (a struct's array member, a nested struct/array): initialize the sub-object in place rather
            # than treating the brace as an rvalue.
            if isinstance(expr, cast.AggInit):
                if isinstance(key, tuple):
                    tlv = self._designate(rid, ct, key)
                    self._init_subagg(rid, tlv.ct, tlv.byte_off, expr)
                elif ct.kind == "array":
                    idx = key if isinstance(key, int) else cursor
                    cursor = idx + 1
                    elem, stride = self._array_row(ct)        # multi-dim: the element is a ROW sub-array, not the leaf
                    self._init_subagg(rid, elem, idx * stride, expr)
                else:
                    if isinstance(key, str):
                        ftype, boff, _bo, _bw = ct.field(key)
                    else:
                        _fn, ftype, boff, _bo, _bw = ct.fields[cursor]
                        cursor += 1
                    self._init_subagg(rid, ftype, boff, expr)
                continue
            v = self._rvalue(expr)
            if isinstance(key, tuple):                        # a nested designator chain (.a.b / .v[i] / .m[i][j])
                self._write(self._designate(rid, ct, key), v)
                continue
            if ct.kind == "array":
                idx = key if isinstance(key, int) else cursor
                cursor = idx + 1
                ti = self._temp(scalar("int", self.abi), "ai")
                self._emit("c.const", Opcode.LOAD, (), (ti,), imm=(idx,))
                lv = _LV("mem", rid, ct.of or scalar("uint32_t"), idx=ti)
            else:                                             # struct / union member
                if isinstance(key, str):
                    ftype, boff, bo, bw = ct.field(key)
                else:
                    _fn, ftype, boff, bo, bw = ct.fields[cursor]
                    cursor += 1
                lv = _LV("mem", rid, ftype, byte_off=boff, bit_off=bo, bit_width=bw, packed=ct.packed)
            self._write(lv, v)

    def _init_subagg(self, rid: int, ct: CType, base_off: int, ag: "cast.AggInit") -> None:
        """Initialize a nested aggregate sub-object (an array member or a nested struct/union) at byte
        offset `base_off` within `rid` from a braced initializer -- one OFFSET-based store per element /
        member at its absolute offset (so the stores compose through any depth of nesting), riding the
        `= {0}` baseline already emitted for the whole object. Positional + `[i]=` / `.field=` keys."""
        cursor = 0
        for key, expr in ag.entries:
            if ct.kind == "array":
                idx = key if isinstance(key, int) else cursor
                cursor = idx + 1
                ftype, stride = self._array_row(ct)           # multi-dim row sub-array (or the scalar leaf, 1-D)
                off, bo, bw = base_off + idx * stride, 0, 0
            else:                                             # struct / union member
                if isinstance(key, str):
                    ftype, mboff, bo, bw = ct.field(key)
                else:
                    _fn, ftype, mboff, bo, bw = ct.fields[cursor]
                    cursor += 1
                off = base_off + mboff
            if isinstance(expr, cast.AggInit):                # deeper nesting
                self._init_subagg(rid, ftype, off, expr)
            else:
                self._write(_LV("mem", rid, ftype, byte_off=off, bit_off=bo, bit_width=bw, packed=ct.packed),
                            self._rvalue(expr))

    def _designate(self, rid: int, ct: CType, steps: tuple) -> "_LV":
        """Resolve a nested designator chain to an lvalue: walk the aggregate type accumulating a byte
        offset -- a `("m", field)` step descends a struct/union member (carrying its bitfield position), an
        `("a", i)` step folds a constant array index into the offset. The leaf type sizes the store."""
        off, cur, bit_off, bit_w, parent_packed = 0, ct, 0, 0, False
        for kind, val in steps:
            if kind == "m":
                parent_packed = cur.packed                    # the struct that DECLARES this (maybe bitfield) member
                ftype, boff, bo, bw = cur.field(val)
                off += boff
                cur, bit_off, bit_w = ftype, bo, bw
            else:                                             # an array element: fold the constant index
                elem = cur.of if cur.of is not None else scalar("uint32_t")
                off += val * elem.size
                cur, bit_off, bit_w = elem, 0, 0
        return _LV("mem", rid, cur, byte_off=off, bit_off=bit_off, bit_width=bit_w, packed=parent_packed)

    def _incdec_value(self, node: cast.IncDec) -> int:
        """`a++` / `++a` / `a--` / `--a` in EXPRESSION position -> a read-modify-write yielding the OLD value
        (postfix) or the NEW value (prefix). The lvalue is resolved ONCE (C evaluates it once, so a
        side-effecting index `arr[f()]++` runs f() once). A pointer steps by element (c.ptradd/sub). A
        NAMED-LOCAL postfix must SNAPSHOT the old value first -- `_read` returns the mutable storage rid, which
        the store would then clobber. A volatile/MMIO target (an extra access) and a non-scalar memory lvalue
        stay a both-rails fallback."""
        lv = self._lvalue(node.operand)
        if lv.bit_width and lv.kind == "mem" and self._mmio(lv.rid):
            raise CLowerError("inc/dec of a volatile/MMIO bitfield is a follow-on")
        if lv.kind != "var" and not (_is_scalar_member_lv(node.operand, lv) and not self._mmio(lv.rid)):
            raise CLowerError("inc/dec of this lvalue form is a follow-on")  # MMIO / non-scalar member
        cur = self._read(lv)
        if not node.prefix and lv.kind == "var":         # snapshot: _read hands back the mutable storage rid,
            old = self._temp(lv.ct, "id_old")            # which the store below would clobber -- a same-type
            self._emit(f"c.cast:{_cast_name(lv.ct)}", Opcode.ADD, (cur,), (old,))  # cast DECLARES the temp
        else:
            old = cur                                     # a memory read is already a fresh declared temp
        one = self._rvalue(cast.IntLit(1))
        if lv.ct.kind == "pointer":                       # a pointer steps by sizeof(*p): c.ptradd MUTATES the
            self._emit("c.ptradd" if node.op == "+" else "c.ptrsub",   # storage IN PLACE (`p += 1`), so only a
                       Opcode.ADD, (lv.rid, one), (lv.rid,))           # named-local pointer reaches here (mem
            return old if not node.prefix else lv.rid     # pointer lvalues fall back); no separate write/re-read
        opcode, suf = _BIN[node.op]
        rt = self._bin_result_type(node.op, cur, one)
        new = self._temp(rt, f"id_{suf}")
        self._emit(f"c.bin.{suf}", opcode, (cur, one), (new,))
        self._write(lv, new)
        if not node.prefix:
            return old                                    # postfix: the pre-step value
        nt = self.rtypes.get(new)                         # prefix: the STORED new value -- re-read if the target
        if lv.bit_width or (nt is not None and lv.ct.size < nt.size):
            return self._read(lv)                         # narrows (a bitfield / sub-int local), else `new`
        return new

    def _assign(self, node: cast.Assign, stmt: bool = False) -> int:
        # An assignment as a VALUE (a sub-expression: `a = b = c`, `if ((x = f()))`, `(p->x = v) + 1`) yields
        # the assigned value. A NAMED LOCAL returns its storage (a later read sees the converted value); a MEMORY
        # lvalue is written then RE-READ, so the value is the stored/converted value (`(char_m = 300)` -> the
        # (char) value), matching Clang -- exactly the named-local semantics. A bitfield or array-of-structs
        # element target as a VALUE stays a follow-on (it would double the bf.get/strided read surface); both
        # rails fall back. As a STATEMENT (`stmt`) the value is unused, so every lvalue form is fine and there is
        # no re-read.
        named_local = isinstance(node.target, cast.Name) and node.target.ident in self.env
        # pointer compound-assign  p += n / p -= n  (and p++/p--, which desugar to `p = p + 1`):
        # a single pointer-arithmetic claim, so the result stays a pointer (the integer binary result
        # would truncate the pointer). The emit renders `p += n;` and lets C scale by the element size.
        if (isinstance(node.target, cast.Name) and node.target.ident in self.env
                and isinstance(node.value, cast.Binary) and node.value.op in ("+", "-")
                and isinstance(node.value.lhs, cast.Name)
                and node.value.lhs.ident == node.target.ident):
            rid, ct = self._lookup(node.target.ident, node.target.pos)
            if ct.kind == "pointer":
                n = self._rvalue(node.value.rhs)
                self._emit("c.ptradd" if node.value.op == "+" else "c.ptrsub", Opcode.ADD, (rid, n), (rid,))
                return rid
        # compound assignment to a MEMORY lvalue (`a[i] OP= e`, `s.m OP= e`, `*p OP= e`): the parser desugars
        # `lhs OP= e` to `lhs = lhs OP e` REUSING the same target node, so resolve the lvalue ONCE (C
        # evaluates `lhs` once) -- one index/address, read + stored through it -- instead of recomputing it
        # for the read and again for the store (which both diverges from the twin and would double-run a
        # side-effecting index). The node-IDENTITY test distinguishes this from an explicit `a[i]=a[i]*e`.
        if (isinstance(node.value, cast.Binary) and node.value.lhs is node.target
                and not isinstance(node.target, cast.Name)):
            lv = self._lvalue(node.target)
            if not stmt and not (_is_scalar_member_lv(node.target, lv) and not self._mmio(lv.rid)):
                raise CLowerError("this lvalue form's compound assignment as a value is a follow-on")
            cur = self._read(lv)
            b = self._rvalue(node.value.rhs)
            opcode, suf = _BIN[node.value.op]
            rt = self._bin_result_type(node.value.op, cur, b)
            t = self._temp(rt, f"b_{suf}")
            res = self._emit(f"c.bin.{suf}", opcode, (cur, b), (t,))
            self._write(lv, res)
            # the value of a compound assignment is the STORED (narrowed) value, not the raw binop result --
            # when the target is NARROWER than the promoted result the store truncates, so re-read it (like
            # the plain `=` path); `res` would be the un-narrowed sum (a both-rails miscompile). A BITFIELD
            # narrows to its BIT width (its ct.size is the full underlying type, so the size test misses it) --
            # re-read it too. A full-width non-bitfield target needs no re-read (res == the stored value).
            narrows = lv.bit_width or lv.ct.size < rt.size
            return res if (stmt or not narrows) else self._read(lv)
        v = self._rvalue(node.value)
        if named_local:
            rid, _ct = self._lookup(node.target.ident, node.target.pos)           # copy into the mutable storage
            self._emit("c.copy", Opcode.ADD, (v,), (rid,))
            self._bind_extent(rid, _ct, node.target.ident, node.value)            # §5.12: `p = malloc(N*…)` -> N
            return rid
        lv = self._lvalue(node.target)
        if not stmt and not (_is_scalar_member_lv(node.target, lv) and not self._mmio(lv.rid)):
            raise CLowerError("this lvalue form's assignment as a value is a follow-on")  # MMIO re-read unsound
        self._write(lv, v)
        return v if stmt else self._read(lv)                   # value context: the stored/converted value (re-read)

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
            fptr, fpct = self.env[node.callee]              # the funcptr CType carries its return type in `.of`
            t = self._temp(self._call_result_ct(fpct.of), f"icall_{node.callee}")   # a signed return reads back signed
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
        if node.callee in ("atomic_compare_exchange_strong", "atomic_compare_exchange_weak"):
            # C11 bool atomic_compare_exchange_strong/weak(A *obj, C *expected, C desired): if *obj ==
            # *expected set *obj = desired (true), else load *obj into *expected (false). `expected` is a
            # POINTER -- the headline use of address-of (`&exp`), which is why this waited on `&`. Emitted
            # verbatim; the atomic semantics are delegated to the resident backend / Clang.
            op = ("c.c11atom.cas_strong" if node.callee.endswith("strong") else "c.c11atom.cas_weak")
            ptr, exp = actuals[0], (actuals[1] if len(actuals) > 1 else actuals[0])
            des = actuals[2] if len(actuals) > 2 else exp
            t = self._temp(scalar("_Bool"), "c11cas")                 # the comparison result is _Bool
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
        if node.callee in ("va_start", "va_end", "va_copy"):   # opaque void variadic builtins (verbatim emit)
            self._emit(f"c.call.vabuiltin:{node.callee}", Opcode.GEM_DISPATCH, actuals, ())
            return _VOID_RID                           # not added to self.calls: opaque to the R18 call graph
        libm = _libm_type(node.callee)
        if libm is not None:                           # a <math.h> call -> a typed external library edge
            t = self._temp(libm, f"libm_{node.callee}")    # not added to self.calls: opaque to the call
            return self._emit(f"c.call.libm:{node.callee}", Opcode.GEM_DISPATCH, actuals, (t,))  # graph
        if node.callee in _EXTERN_VARIADIC and node.callee not in self.func_rets:
            t = self._temp(scalar("int", self.abi), f"ext_{node.callee}")   # a printf/scanf-family external
            return self._emit(f"c.call.extern:{node.callee}", Opcode.GEM_DISPATCH, actuals, (t,))  # variadic
        if node.callee in _STDLIB_ALLOC and node.callee not in self.func_rets:
            t = self._temp(pointer(scalar("void")), f"mem_{node.callee}")   # malloc/calloc/realloc -> void *
            # a R21 lifetime ALLOC event on the result: it (re-)validates the allocation, so a pointer assigned
            # from it is live again after an earlier free (`p=malloc; free(p); p=malloc; p[i]` is legal). Pairs
            # with the `free` event below; digest-excluded + vacuous unless the smart-lowering verifier runs R21.
            return self._emit(f"c.call.libm:{node.callee}", Opcode.GEM_DISPATCH, actuals, (t,),  # verbatim, opaque
                              lifetime=Lifetime("alloc"))
        if node.callee == "free" and node.callee not in self.func_rets:
            # void external (verbatim, opaque) + a R21 lifetime FREE event: the freed pointer (the actual it
            # reads) dies after this claim, so a later dereference of it is a use-after-free (§5.12). Vacuous
            # unless the smart-lowering verifier runs R21 -- the frontend pass/fail is unchanged.
            self._emit("c.call.libm.void:free", Opcode.GEM_DISPATCH, actuals, (),
                       lifetime=Lifetime("free"))
            return _VOID_RID
        bt = _builtin_type(node.callee, self.abi)
        if bt is not None:                             # a GCC/Clang integer builtin -> verbatim, typed, opaque
            t = self._temp(bt, "blt")                  # the op carries the suffix (the `__builtin_` is re-added
            return self._emit(f"c.call.builtin:{node.callee[len('__builtin_'):]}",  # in emit; the op buffer is small)
                              Opcode.GEM_DISPATCH, actuals, (t,))
        self.calls.append((node.callee, actuals))      # a defined-in-unit callee: a real R18 edge
        ret_ct = self.func_rets.get(node.callee)
        if ret_ct is not None and ret_ct.name == "void":   # a void callee -> a bare call statement, no
            self._emit(f"c.call.void:{node.callee}", Opcode.GEM_DISPATCH, actuals, ())   # result temp
            return _VOID_RID                           # the value of a void call is never read
        # type the result temp by the callee's return type: a float return propagates (so downstream
        # arithmetic stays float), a wide (8-byte) integer return keeps its width, and a signed `int` return
        # keeps its SIGN -- else a downstream `>>` / comparison on the call result would go unsigned (a
        # logical shift / unsigned compare). A struct/union RETURN keeps the aggregate CType (the result is a
        # by-value struct temp -- `struct P t = mk(x);` -- so it can be copied into a local, passed by value, or
        # member-accessed; typing it uint32 emitted invalid C `uint32_t t = mk(x)`).
        t = self._temp(self._call_result_ct(ret_ct), f"call_{node.callee}")
        return self._emit(f"c.call:{node.callee}", Opcode.GEM_DISPATCH, actuals, (t,))

    def _call_result_ct(self, ret_ct):
        """The result-temp CType for a call, by the callee's return type -- shared by direct, indirect
        (funcptr param), and member (funcptr field / dispatch) calls so all four type identically. A struct
        return keeps the aggregate; a float or wide (>4-byte) integer keeps its CType; a SIGNED sub-int return
        promotes to `int` (so a downstream `>>` / compare / `(long)` widen sign-extends); else uint32. A
        funcptr whose return type wasn't captured (`ret_ct is None`) stays uint32 -- today's behaviour."""
        if ret_ct is not None and ret_ct.is_aggregate:
            return ret_ct
        if ret_ct is not None and ret_ct.is_bitint:       # a C23 `_BitInt(N)` return keeps its exact width
            return ret_ct                                 # (it does not promote, and same-type arithmetic on
                                                          # the result must stay `_BitInt(N)`, not canonicalize)
        if ret_ct is not None and (ret_ct.is_float or (ret_ct.is_integer and ret_ct.size > 4)):
            return ret_ct
        if ret_ct is not None and ret_ct.is_integer and ret_ct.signed and ret_ct.size <= 4:
            return scalar("int", self.abi)
        return scalar("uint32_t")

    # --- statements ---
    def _block(self, stmts) -> list:
        block: list = []
        self.block_stack.append(block)
        saved_env = dict(self.env)                            # a `{ ... }` is a scope: locals declared
        for s in stmts:                                       # inside it do not leak to the enclosing
            self._stmt(s)                                     # block (so a shadow does not clobber an
        self.env = saved_env                                 # outer same-named var read after the block)
        self.block_stack.pop()
        return block

    def _stmt(self, st):
        if isinstance(st, cast.Seq):                          # several declarators of one declaration
            for s in st.stmts:
                self._stmt(s)
            return None
        if isinstance(st, cast.Block):                        # a bare `{ ... }` -> its scoped nodes,
            self.block_stack[-1].extend(self._block(st.stmts))  # appended inline (no if(1) wrapper)
            return None
        if isinstance(st, cast.Decl):
            if st.type.vla is not None:
                # a 1-D stack VLA `T a[n]`: evaluate the runtime size ONCE (C semantics), snapshot it into an
                # immutable hidden extent, declare the array IN-BODY (`T a[__ext];`) and mask `a[i]` against the
                # snapshot. A VLA cannot be static or initialized in C -> those route to fallback.
                if st.static_storage or st.init is not None:
                    raise CLowerError("a VLA cannot have static storage or an initializer")
                elem = self._resolve_type(replace(st.type, vla=None, array=()))
                if elem.size <= 0 or elem.kind == "array":
                    raise CLowerError("VLA of an incomplete / void / array element is not supported")
                nval = self._rvalue(st.type.vla)              # the size expression, evaluated exactly once
                nct = self.rtypes.get(nval)
                if nct is None or nct.kind != "scalar" or nct.is_float:
                    raise CLowerError("a VLA size must be an integer expression")
                ext = self._storage(nct, f"__bcir_ext{self._ext_ctr}")     # the snapshot: immutable extent
                self._ext_ctr += 1
                self._emit("c.copy", Opcode.ADD, (nval,), (ext,))
                ct = array(elem, 0)                           # count 0 -> a runtime extent; bounds via ptr_extent
                rid = self._vla_storage(ct, st.name, ext)     # declared in-body; masked `a[i]` vs the snapshot
                self.env[st.name] = (rid, ct)
                self.ptr_extent[rid] = ext
                return None
            if st.type.vla_dims:
                # a MULTI-dim stack VLA `T a[d0][d1]...`: snapshot each dim ONCE (canonical order: dims 0..k-1,
                # then the product), declare the array IN-BODY as a FLAT `T a[__ext_total];` (same memory layout
                # as `T a[d0][d1]`), and mask the row-major Horner index `i*d1 + j` against the total extent.
                # The Horner multiplier at step d is dim d's snapshot rid (a runtime value, not a c.const).
                if st.static_storage or st.init is not None:
                    raise CLowerError("a VLA cannot have static storage or an initializer")
                elem = self._resolve_type(replace(st.type, vla=None, vla_dims=(), array=()))
                if elem.size <= 0 or elem.kind == "array":
                    raise CLowerError("a multi-dimensional VLA of an incomplete / array element is not supported")
                u32 = scalar("uint32_t", self.abi)
                dim_exts = []                                 # 1. snapshot each dim into a named __bcir_extK
                for d in st.type.vla_dims:
                    ext = self._storage(u32, f"__bcir_ext{self._ext_ctr}")
                    self._ext_ctr += 1
                    if isinstance(d, int):                    # a literal dim -> a const temp, copied into the ext
                        kt = self._temp(u32, "kd")
                        self._emit("c.const", Opcode.LOAD, (), (kt,), imm=(d,))
                        self._emit("c.copy", Opcode.ADD, (kt,), (ext,))
                    else:                                     # a runtime dim -> evaluated once, copied in
                        self._emit("c.copy", Opcode.ADD, (self._rvalue(d),), (ext,))
                    dim_exts.append(ext)
                total = dim_exts[0]                           # 2. total extent = product of all dim snapshots
                for d in range(1, len(dim_exts)):
                    prod = self._temp(u32, "b_mul")
                    self._emit("c.bin.mul", Opcode.MUL, (total, dim_exts[d]), (prod,))
                    total = prod
                ext_total = self._storage(u32, f"__bcir_ext{self._ext_ctr}")
                self._ext_ctr += 1
                self._emit("c.copy", Opcode.ADD, (total,), (ext_total,))   # stable name for the decl + the mask
                ct = array(elem, 0)                           # 3. a FLAT runtime-extent array
                rid = self._vla_storage(ct, st.name, ext_total)
                self.env[st.name] = (rid, ct)
                self.ptr_extent[rid] = ext_total
                self.vla_strides[rid] = tuple(dim_exts)       # the per-dim Horner multipliers
                return None
            if len(st.type.array) > 1:
                # a multi-dimensional local array `T m[A][B]`: a flat resource of A*B elements carrying
                # the per-dim shape, so `m[i][j]` flattens row-major to `m[i*B + j]` (the existing Index
                # Horner) and the emit declares `m[A*B]` (same memory layout as `m[A][B]`). Capped at 3
                # dims (the shape / twin-adims table holds 3); deeper defers to the LLVM fallback.
                dims = st.type.array
                if len(dims) > 3:
                    raise CLowerError(
                        f"multi-dimensional local array '{st.name}' of more than 3 dims is not yet supported")
                elem = self._resolve_type(replace(st.type, array=()))
                total = 1
                for d in dims:
                    total *= d
                ct = replace(array(elem, total), shape=tuple(dims))
            elif (len(st.type.array) == 1 and st.type.array[0] in (0, None)
                  and isinstance(st.init, cast.AggInit)):
                # an INFERRED-size local array `T a[] = {...}` (scalar OR struct element): `_resolve_type`
                # would size it `array(elem, 0)` -> `a[0]`, so the per-element init stores write past the
                # storage (UB, a #500-class silent miscompile). Infer the outer count from the initializer
                # the SAME way the compound-literal path does (max index + 1; positional advances a cursor,
                # `[i]=` designators jump it), then build the correctly-sized `array(elem, n)`.
                elem = self._resolve_type(replace(st.type, array=()))
                n, cursor = 0, 0
                for key, _expr in st.init.entries:
                    if isinstance(key, tuple):                # a nested chain `[i]...` -> its outer array index
                        idx = key[0][1] if key and key[0][0] == "a" else cursor
                    else:
                        idx = key if isinstance(key, int) else cursor
                    cursor = idx + 1
                    n = max(n, cursor)
                ct = array(elem, n or 1)
            else:
                ct = self._resolve_type(st.type)
            if st.static_storage:                             # static storage: init once, in the decl
                rid = self._static_storage(ct, st.name, _fold_const(st.init))
                self.env[st.name] = (rid, ct)
                return None
            rid = self._storage(ct, st.name)                  # a mutable named local
            self.env[st.name] = (rid, ct)
            if isinstance(st.init, cast.AggInit):             # struct/union/array `= { ... }`
                self._agg_init(rid, ct, st.init)
            elif st.init is not None:
                self._emit("c.copy", Opcode.ADD, (self._rvalue(st.init),), (rid,))
                self._bind_extent(rid, ct, st.name, st.init)  # §5.12: `T *p = malloc(N*sizeof(T))` -> extent N
        elif isinstance(st, cast.ExprStmt):
            if isinstance(st.expr, cast.Assign):              # a statement assignment: any lvalue form is fine
                self._assign(st.expr, stmt=True)              # (the value is unused -- no named-local restriction)
            else:
                self._rvalue(st.expr)
        elif isinstance(st, cast.Return):
            rid = None if st.value is None else self._rvalue(st.value)
            if rid == _VOID_RID:                              # `return void_call();` -> the call stmt is
                rid = None                                    # already emitted; a void function `return;`s
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
        elif isinstance(st, cast.Switch):
            disc = self._rvalue(st.disc)                      # the discriminant, lowered once
            block: list = []
            self.block_stack.append(block)
            for item in st.body:
                if isinstance(item, cast.Case):
                    block.append(CaseLabel(item.value))
                elif isinstance(item, cast.Default):
                    block.append(DefaultLabel())
                else:
                    self._stmt(item)                          # body stmts (break preserved as BreakNode)
            self.block_stack.pop()
            self.block_stack[-1].append(SwitchNode(disc, block))
        elif isinstance(st, cast.For):
            saved_env = dict(self.env)                         # the for-init + body declarations are
            if st.init is not None:                            # loop-scoped: a `for(unsigned i = ...)`
                self._stmt(st.init)                            # must not leak `i` past the loop (where it
            cond_block2: list = []                             # would shadow a same-named param / outer)
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
            self.env = saved_env                               # pop the loop scope (restore outer bindings)
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
        elif isinstance(st, cast.ComputedGoto):           # `goto *p;` -- the target is lowered to a void*
            self.block_stack[-1].append(ComputedGotoNode(self._rvalue(st.target)))
        elif isinstance(st, cast.Label):
            self.block_stack[-1].append(LabelNode(st.name))
        else:
            raise CLowerError(f"statement {type(st).__name__} is beyond the L1–L6 subset")
        return None

    def lower(self) -> LoweredFunc:
        self.last_return = None
        _scan_mutations(self.func.body, self._mut_assigned, self._mut_body, self._mut_addr)  # §5.12 stability pre-pass
        self.env.update(self.genv)                            # file-scope globals are in scope
        for rid, ct in self.genv.values():
            self.rtypes.setdefault(rid, ct)
        for p in self.func.params:                            # params -> input resources (shadowing)
            if p.type.vla is not None:
                # a VLA parameter `T a[n]`: like any array param it decays to a pointer, BUT we recover the
                # runtime extent `n` (a prior in-scope integer parameter) and bind it via ptr_extent so the
                # param's `a[i]` promotes to masked -- a[BCIR_CHK(rid, i, n, "f:a")] -- exactly like a
                # malloc'd pointer with a recovered count. (Resolve the element with vla/array stripped:
                # _resolve_type raises on a VLA TypeRef otherwise -- params are processed in source order, so
                # `n` is already in env and a later param is not.)
                elem = self._resolve_type(replace(p.type, vla=None, array=()))
                if elem.size <= 0 or elem.kind == "array":
                    raise CLowerError("a VLA parameter of an incomplete / array element is not supported")
                ct = replace(pointer(elem), shape=(0,))
                rid = self._new_rid()
                self._resource(rid, ct, p.name)
                self.env[p.name] = (rid, ct)
                self.params.append((p.name, rid, ct))
                n = p.type.vla                                # bind ONLY for a STABLE prior integer param `n`
                if isinstance(n, cast.Name) and n.ident in self.env:   # (unmutated, not address-taken --
                    n_rid, n_ct = self.env[n.ident]                    #  the _bind_extent stable-Name contract,
                    if (n_ct.kind == "scalar" and not n_ct.is_float    #  so re-emitting `n` by name is sound)
                            and self._mut_body.get(n.ident, 0) == 0 and n.ident not in self._mut_addr):
                        self.ptr_extent[rid] = n_rid
                continue
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
        gnames.update(self.str_globals)                       # string globals render as inline literals
        gnames.update(self.func_globals)                      # function-as-value globals render as the bare name
        m = Module(name=self.func.name)
        for rid in sorted(self.resources):
            m.add_resource(self.resources[rid])
        m.add_phase(Phase(phase_id=0, deps=(), claims=list(claims)))
        ret_ct = self._resolve_type(self.func.ret)
        return LoweredFunc(name=self.func.name, module=m, ret_type=ret_ct, params=self.params,
                           return_rid=self.last_return, claims=list(claims),
                           resources=dict(self.resources), rid_types=dict(self.rtypes),
                           calls=list(self.calls), region=None, body=body, locals=list(self.locals),
                           vla_locals=list(self.vla_locals),
                           statics=list(self.statics), globals_used=gnames,
                           zero_init_locals=set(self.zero_init), variadic=self.func.variadic,
                           reproducible=getattr(self.func, "reproducible", False),
                           ptr_extent=dict(self.ptr_extent))


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
        elif isinstance(node, SwitchNode):
            flush_run()
            parts.append(_block_region(node.body, functions, calls_iter))   # the clause bodies, in order
        elif isinstance(node, (ReturnNode, BreakNode, ContinueNode, GotoNode, ComputedGotoNode, LabelNode,
                               CaseLabel, DefaultLabel)):
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


def lower_unit(unit: cast.Unit, abi=None) -> LoweredUnit:
    """Lower a whole translation unit. Aggregates are laid out first (Clang-compatible for the target
    data model `abi`, defaulting to the host), then each function to its claim-graph Module +
    compose.Function."""
    abi = abi or HOST
    aggregates: dict[str, CType] = {}
    for tag, agg in unit.aggregates.items():
        b = AggregateBuilder(agg.kind, tag, packed=agg.packed, force_align=agg.align)
        for tref, mname, width, malign in agg.members:
            mt = _resolve_member_type(tref, aggregates, abi)
            if mt.is_bitint and width:                    # a `_BitInt(N)` BITFIELD `_BitInt(N) m : W` is now
                # first-class: it packs into the `_BitInt(N)` STORAGE UNIT (its `mt.size` = 1/2/4/8 bytes,
                # the Clang slot) with the same LSB-first packing as a standard-int bitfield of that storage
                # size -- VERIFIED byte-identical to Clang (size/align/bit-layout) for 2<=N<=64, 1<=W<=N.
                # The field value is typed `_BitInt(N)` (its W bits), and the emit prints `_BitInt(N) m : W;`.
                if not (1 <= width <= mt.bit_width):       # W>N is invalid C (Clang errors); W<1 not a named
                    raise CLowerError(                     # field -> conservatively route to fallback.
                        "a `_BitInt` bitfield width outside 1..N is not supported")
            # a PLAIN `_BitInt(N)` member is also first-class: its `bitint` CType carries the Clang storage
            # width (1/2/4/8 bytes) and alignment, so AggregateBuilder lays it out exactly like the
            # equivalent power-of-two int (sizeof/offsetof match Clang); the member load/store uses that
            # storage width typed `_BitInt(N)`, and the emit prints the faithful spelling.
            b.members.append((mname, mt, width, malign))
        aggregates[tag] = b.build()

    genv: dict[str, tuple] = {}                            # file-scope globals: name -> (rid, CType)
    gres: dict[int, Resource] = {}
    for gi, g in enumerate(unit.globals):
        ct = _resolve_member_type(g.type, aggregates, abi)
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
    strctr = [0]                                          # unit-wide string-literal counter (unique rids)
    # pre-scan every function's return type (forward references resolve too), so a call can be typed
    # by its callee: a void call emits a bare statement, a wide/float return keeps its real type.
    func_rets = {fn.name: _resolve_member_type(fn.ret, aggregates, abi) for fn in unit.funcs}
    for idx, fn in enumerate(unit.funcs):
        lf = _FuncLowerer(fn, aggregates, base_rid=100 + idx * 1000, cid=cid,
                          genv=genv, gres=gres, strctr=strctr, func_rets=func_rets, abi=abi).lower()
        functions[fn.name] = lf
        resources.update(lf.resources)
    for lf in functions.values():                          # regions need every callee's param rids
        lf.region = _region_for(lf, functions)
        compose_functions[lf.name] = compose.Function(lf.name, lf.region)
    entry = unit.funcs[-1].name if unit.funcs else ""
    return LoweredUnit(functions=functions, entry=entry, aggregates=aggregates,
                       compose_functions=compose_functions, resources=resources)


def _resolve_member_type(tref: cast.TypeRef, aggregates: dict, abi=None) -> CType:
    abi = abi or HOST
    if tref.funcptr:                                       # a function-pointer member (dispatch table)
        ret = _resolve_member_type(tref.func_ret, aggregates, abi)
        params = tuple(_resolve_member_type(p, aggregates, abi) for p in tref.func_params)
        return funcptr(tref.base, ret, params, abi)
    if tref.aggregate:
        base = aggregates[tref.base]
    elif tref.bit_width:                                  # C23 `_BitInt(N)` (e.g. a function return type)
        base = bitint(tref.bit_width, signed="unsigned" not in tref.base)
    else:
        base = scalar(tref.base, abi)
    t = base
    for _ in range(tref.ptr):
        t = pointer(t, abi)
    for dim in reversed(tref.array):
        t = array(t, dim)
    return t
