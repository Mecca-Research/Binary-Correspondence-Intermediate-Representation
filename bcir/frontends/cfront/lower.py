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

import copy
import dataclasses
from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from ...kbcir import compose
from ...model import Claim, Domain, Lane, Lifetime, Module, Opcode, Phase, Resource, StrideClass
from . import cast
from .abi import HOST
from .clex import split_lit_prefix, str_elem_size, str_units
from .ctype_model import (
    AggregateBuilder,
    BitIntMix,
    CType,
    array,
    bitint,
    funcptr,
    is_scalar_name,
    narrow_bitfield,
    pointer,
    promote_int,
    qualified,
    scalar,
    unqualified,
    usual_arith_int,
    valist,
    with_atomic,
    incomplete_aggregate,
    with_volatile,
)

# C operator -> (cost-class Opcode, op-suffix the emitter maps back to a C operator).
_BIN = {
    "+": (Opcode.ADD, "add"),
    "-": (Opcode.SUB, "sub"),
    "*": (Opcode.MUL, "mul"),
    "/": (Opcode.MUL, "div"),
    "%": (Opcode.MUL, "mod"),
    "&": (Opcode.ADD, "and"),
    "|": (Opcode.ADD, "or"),
    "^": (Opcode.ADD, "xor"),
    "<<": (Opcode.ADD, "shl"),
    ">>": (Opcode.ADD, "shr"),
    "==": (Opcode.SUB, "eq"),
    "!=": (Opcode.SUB, "ne"),
    "<": (Opcode.SUB, "lt"),
    ">": (Opcode.SUB, "gt"),
    "<=": (Opcode.SUB, "le"),
    ">=": (Opcode.SUB, "ge"),
    "&&": (Opcode.ADD, "land"),
    "||": (Opcode.ADD, "lor"),
}
_UN = {"-": (Opcode.SUB, "neg"), "~": (Opcode.ADD, "bnot"), "!": (Opcode.SUB, "lnot")}

# A cast's target type, named by width so both rails emit the same (uintN_t) spelling. The C value
# model tracks everything as a 32-bit unit, so a cast renders by size (a pointer cast keeps its `*`).
_CAST_W = {1: "uint8_t", 2: "uint16_t", 4: "uint32_t", 8: "uint64_t"}
_CAST_W_SIGNED = {1: "int8_t", 2: "int16_t", 4: "int32_t", 8: "int64_t"}


def _strip_quotes(spelling: str) -> str:
    """The inner content of a string-literal source spelling (`"=r"` -> `=r`, `"memory"` -> `memory`).
    Used for inline-asm (ASM1) constraints/clobbers, which are simple unescaped quoted strings; the
    surrounding double quotes are dropped (a non-quoted input is returned unchanged, defensively)."""
    if len(spelling) >= 2 and spelling[0] == '"' and spelling[-1] == '"':
        return spelling[1:-1]
    return spelling


def _pointer_spelling(ct: CType) -> str:
    """A pointer cast target, spelled faithfully: the innermost pointee's own type -- its sign, a
    plain `char`, `_Bool`, a float, a struct/union tag or `void` -- behind its `volatile`, then one
    `*` per level. The cast then yields a real `T *` of exactly the target type (a width-named
    pointee dropped the qualifier and the sign, and gave a struct pointee an integer name). The twin
    spells it byte-identically (`bcir_cfront.c`, `cast_name`); the structural digest keeps it."""
    depth, b = 0, ct
    while b is not None and b.kind == "pointer":
        depth, b = depth + 1, b.of
    if b is None or b.name == "void":
        base = "void"
    elif b.is_aggregate:
        base = f"{b.kind} {b.name}"
    elif b.is_float or b.is_bitint:
        base = b.name
    elif b.name in ("_Bool", "bool"):
        base = "_Bool"
    elif b.name == "char":
        base = "char"
    elif b.signed:
        base = _CAST_W_SIGNED.get(b.size, "int32_t")
    else:
        base = _CAST_W.get(b.size, "uint32_t")
    return ("volatile " if b is not None and b.volatile else "") + base + " " + "*" * depth


def _cast_name(ct: CType) -> str:
    if ct.kind == "pointer":
        return _pointer_spelling(ct)
    if ct.is_bitint:
        return ct.name  # `_BitInt(N)` / `unsigned _BitInt(N)` -- faithful, exact width
    if ct.is_float:
        return ct.name  # float / double / long double
    if ct.name in ("_Bool", "bool"):
        return "_Bool"  # a bool cast normalizes any nonzero to 1 on the
        # FULL value -- (uint8_t) would truncate first
        # (so (_Bool)256 must be 1, not 0; (_Bool)0.5 is 1)
    return _CAST_W.get(ct.size, "uint32_t")


_FLOAT_RANK = {"float": 0, "double": 1, "long double": 2}
_RANK_FLOAT = {0: "float", 1: "double", 2: "long double"}


def _elem_float_name(t) -> str:
    """The element float spelling of a (possibly complex) float type: `double _Complex` -> `double`,
    a plain `double` -> `double`. Used to promote complex arithmetic by element rank."""
    return t.name[: -len(" _Complex")] if t.is_complex else t.name


def _float_lit_type(spelling: str, abi=None) -> CType:
    """The type of a floating literal from its suffix: `f`/`F` -> float, `l`/`L` -> long double,
    else double (the C default) -- laid out by the target's ABI, whose `long double` is 16 bytes on
    x86-64 Linux, 12 on i386 and 8 on Windows x64 (the twin reads the same ABI table)."""
    s = spelling.strip()
    if s and s[-1] in "fF":
        return scalar("float", abi)
    if s and s[-1] in "lL":
        return scalar("long double", abi)
    return scalar("double", abi)


# <math.h> functions whose result is a floating type fixed by the name suffix (base -> double,
# +f -> float, +l -> long double). Most are real-valued; a few take a non-floating argument that is
# simply passed through -- ldexp/scalbn (an int exponent), scalbln (a long), nan (a tag string). They
# lower to an opaque external library edge: the emit calls the real libm function (the harness links
# -lm) so the IEEE-754 result is delegated to the backend, and the call graph sees no callee to
# resolve (R18 treats it like an indirect call).
_LIBM = frozenset(
    {
        "acos",
        "asin",
        "atan",
        "atan2",
        "cos",
        "sin",
        "tan",
        "acosh",
        "asinh",
        "atanh",
        "cosh",
        "sinh",
        "tanh",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log10",
        "log1p",
        "log2",
        "logb",
        "cbrt",
        "fabs",
        "hypot",
        "pow",
        "sqrt",
        "ceil",
        "floor",
        "round",
        "trunc",
        "nearbyint",
        "rint",
        "erf",
        "erfc",
        "lgamma",
        "tgamma",
        "copysign",
        "fdim",
        "fmax",
        "fmin",
        "fmod",
        "remainder",
        "fma",
        "nextafter",
        "ldexp",
        "scalbn",
        "scalbln",
        "nan",
        "frexp",
        "modf",
        "remquo",  # a pointer out-param (rides c.addrof); double result
    }
)
# <math.h> functions with a fixed *integer* result (the f/l suffix types only the argument): ilogb
# returns int; lround/lrint return long and llround/llrint return long long. The emitted result temp
# is declared at the rid type's true width, so the 8-byte returns are not truncated to the 4-byte
# value model. (frexp/modf/remquo, which take a pointer out-param, ride the address-of-argument path.)
_LIBM_INT = {
    "ilogb": "int",
    "lround": "long",
    "lrint": "long",
    "llround": "long long",
    "llrint": "long long",
}

# <complex.h> functions, lowered like a libm call (opaque external edge, emitted verbatim against
# <complex.h>). The result type differs by function: creal/cimag/cabs/carg return the *real* element
# float; conj/cproj AND the C99 complex transcendentals return the *complex* type. The f/l suffix picks
# float / long double (base -> double).
# <complex.h> imaginary unit: `I` is the macro `_Complex_I`, a `const float _Complex` of value i (0+1i).
# (Clang doesn't implement `_Imaginary`, so `_Imaginary_I` is intentionally not recognized.) Lowered to a
# `c.cconst` that emits the token VERBATIM -- the re-emitted twin compiles against <complex.h> just like the
# original, so it resolves to the same imaginary unit. Honored only when NOT shadowed by a declared name.
_IMAG_UNIT = frozenset({"I", "_Complex_I"})

_CPLX_REAL = frozenset({"creal", "cimag", "cabs", "carg"})  # -> real element float
_CPLX_CPLX = frozenset(
    {  # -> the complex type (same width)
        "conj",
        "cproj",  #   algebraic
        "cexp",
        "clog",
        "csqrt",
        "cpow",  #   exp / log / sqrt / power
        "csin",
        "ccos",
        "ctan",
        "casin",
        "cacos",
        "catan",  #   circular + inverse
        "csinh",
        "ccosh",
        "ctanh",
        "casinh",
        "cacosh",
        "catanh",
    }
)  #   hyperbolic + inverse


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
_EXTERN_VARIADIC = frozenset(
    {
        "snprintf",
        "vsnprintf",
        "sprintf",
        "vsprintf",
        "printf",
        "fprintf",
        "vprintf",
        "vfprintf",
        "sscanf",
        "vsscanf",
        "scanf",
        "fscanf",
        "dprintf",
    }
)

# <stdlib.h> memory management -- external libc edges (emitted VERBATIM, opaque to R18, NOT bcir_-renamed),
# the seam the naked-pointer safety track (§5.12) hangs lifetime annotations on. The allocators return a
# `void *` (assignable to any object pointer); `free` returns void.
_STDLIB_ALLOC = frozenset({"malloc", "calloc", "realloc", "aligned_alloc"})

# <string.h> memory routines -- external libc edges like the allocators (verbatim, opaque to R18, NOT
# bcir_-renamed), each returning its destination, a `void *`. The emit spells every plain memory access as a
# `memcpy`, so a re-parsed emit calls it (CF-RTWIDE); the escape analysis already reads a library routine as
# copying between its operands.
_STRING_MEM = frozenset({"memcpy", "memmove", "memset"})

# GCC/Clang integer builtins -- emitted verbatim (no bcir_ twin, opaque to R18) with a fixed result type.
_BUILTIN_INT = frozenset(
    {  # the bit-count family + abs -> int
        "__builtin_popcount",
        "__builtin_popcountl",
        "__builtin_popcountll",
        "__builtin_clz",
        "__builtin_clzl",
        "__builtin_clzll",
        "__builtin_ctz",
        "__builtin_ctzl",
        "__builtin_ctzll",
        "__builtin_ffs",
        "__builtin_ffsl",
        "__builtin_ffsll",
        "__builtin_parity",
        "__builtin_parityl",
        "__builtin_parityll",
        "__builtin_clrsb",
        "__builtin_clrsbl",
        "__builtin_clrsbll",
        "__builtin_abs",
    }
)
_BUILTIN_OTHER = {
    "__builtin_labs": "long",
    "__builtin_llabs": "long long",
    "__builtin_bswap16": "uint16_t",
    "__builtin_bswap32": "uint32_t",
    "__builtin_bswap64": "uint64_t",
}

# ASM2 -- port-mapped I/O intrinsics. Each is a CALL expression (no new syntax), recognized at lowering
# like `_LIBM`, and lowered to a typed, isolated, BARRIERED I/O-port edge (`c.portio.in.<w>:` /
# `c.portio.out.<w>:`). Keyed by name: (direction, width-bytes, result-CType-name | None for the void
# writes). `b`/`w`/`l` are the x86 byte (8b) / word (16b) / long (32b) widths. The READS take one `port`
# arg and RETURN the read value (u8/u16/u32); the WRITES take `(value, port)` in the Linux <asm/io.h>
# convention (value FIRST, port SECOND) and are void. The `port` operand is conceptually a u16 I/O-port
# address. Port I/O exists ONLY on x86 (the `in`/`out` instructions); the per-ISA realization is emitted in
# emit.py keyed off the active `--target` (a non-x86 target raises an honest CLowerError -> LLVM fallback).
_PORTIO_IN = {"inb": (1, "uint8_t"), "inw": (2, "uint16_t"), "inl": (4, "uint32_t")}
_PORTIO_OUT = {"outb": 1, "outw": 2, "outl": 4}


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
    cx = _complex_libm_type(name)  # <complex.h> creal/cimag/conj/... (typed first)
    if cx is not None:
        return cx
    base = name[:-1] if name[-1:] in "fl" else name
    if name in _LIBM_INT:
        return scalar(_LIBM_INT[name])
    if base in _LIBM_INT:  # a suffixed variant (ilogbf/ilogbl) -> same int
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
        if not inq:  # between pieces: only `"` opens one
            inq = ch == '"'
            i += 1
            continue
        if ch == '"':  # closing quote of this piece
            inq = False
            i += 1
            continue
        if ch == "\\" and i + 1 < len(spelling):
            c = spelling[i + 1]
            if c == "x":  # \xHH.. -> consume all hex digits
                i += 2
                while i < len(spelling) and spelling[i] in "0123456789abcdefABCDEF":
                    i += 1
            elif c in "01234567":  # \NNN  -> up to three octal digits
                i += 1
                k = 0
                while k < 3 and i < len(spelling) and spelling[i] in "01234567":
                    i, k = i + 1, k + 1
            else:  # \n, \t, \\, \", \0, ...
                i += 2
        else:
            i += 1
        n += 1
    return n


def _const_spelling(v: int) -> str:
    """A folded integer constant as a file-scope initializer's rendering spells it: its value exactly, in a
    type that holds it -- plain decimal (`int`, `long` or `long long`, whichever holds it first), `Nu` past
    `long long`, and the one negative with no positive counterpart as an expression. The declaration
    converts the value to the global's type, as C converts the source's initializer."""
    if v > (1 << 63) - 1:
        return f"{v}u"
    return "(-9223372036854775807 - 1)" if v == -(1 << 63) else str(v)


def _file_scope_rendering(
    ginit: "_FuncLowerer", g, genv: dict, strings: frozenset
) -> "tuple | None":
    """What the linkable emit renders for the global `g` as its initializer: `()` for none, else a 1-tuple of the
    initializer re-spelled as the source spells it -- its braces and designators kept, which C walks the same way
    in the emit (CF-GBRACE) -- each entry rendered: an integer constant expression folded as a static's entry is,
    in C's own types (`_const_value`, `_const_spelling`; CF-INTCONST), a floating constant by its spelling, a
    string literal that initializes a character array (`strings`, the ids the shape walk recorded) by its own, the
    address of a global declared before it by its name. None when an entry is none of these: the linkable emit
    then refuses the global by name."""
    if g.init is None:
        return ()
    with ginit._scratch():
        try:
            return (_spell_init(ginit, g.init, genv, strings),)
        except CLowerError:
            return None


def _spell_init(ginit: "_FuncLowerer", node, genv: dict, strings: frozenset) -> str:
    """One initializer, or one entry of a brace list, as `_file_scope_rendering` renders it."""
    if isinstance(node, cast.AggInit):
        parts = []
        for key, expr in node.entries:
            if isinstance(key, tuple):  # a designator chain `.a[2].b =`
                pre = "".join(f".{v}" if kind == "m" else f"[{v}]" for kind, v in key) + " = "
            elif isinstance(key, str):
                pre = f".{key} = "
            else:
                pre = "" if key is None else f"[{key}] = "
            parts.append(pre + _spell_init(ginit, expr, genv, strings))
        return "{" + ", ".join(parts) + "}"
    if isinstance(node, cast.FloatLit):
        return node.value  # the source spelling, suffix included
    if (
        isinstance(node, cast.Unary)
        and node.op in ("-", "+")
        and isinstance(node.operand, cast.FloatLit)
    ):
        return ("-" if node.op == "-" else "") + node.operand.value
    if isinstance(node, cast.StringLit):
        if id(node) not in strings:  # a pointer's string: not rendered in this slice
            raise CLowerError(_NOT_CONSTANT)
        return node.value  # spelling incl. quotes (a character array's initializer)
    if (
        isinstance(node, cast.Unary)
        and node.op == "&"
        and isinstance(node.operand, cast.Name)
        and node.operand.ident in genv
    ):
        # Part VII A4: an ADDRESS CONSTANT (&x of a file-scope object declared earlier in the unit) -- the
        # platform linker resolves the relocation; the rendering is just the name. Forward references stay
        # refused (genv accumulates in declaration order -- conservative, recorded).
        return f"&{node.operand.ident}"
    return _const_spelling(ginit._const_value(node).v)  # folded as a static's entry is


def _is_pure(node) -> bool:
    """True if evaluating `node` has NO side effect -- safe to re-evaluate for the §5.12 extent snapshot:
    an integer literal, a name read, `sizeof`/`_Alignof` (unevaluated), or an arithmetic unary / binary /
    cast over pure operands. A call, assignment, `++`/`--`, deref/address-of, or anything else is NOT pure
    (conservative: an unrecognized node is impure, so it stays unmanaged rather than double-run)."""
    if isinstance(node, (cast.IntLit, cast.Name, cast.SizeOf, cast.AlignOf)):
        return True
    if isinstance(node, cast.Unary):
        return node.op in ("-", "~", "+") and _is_pure(node.operand)  # arithmetic only (NOT * / &)
    if isinstance(node, cast.Binary):
        return (
            node.op in {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"}
            and _is_pure(node.lhs)
            and _is_pure(node.rhs)
        )
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
            body[node.target.ident] = (
                body.get(node.target.ident, 0) + 1
            )  # an ordinary (non-decl-init) write
    elif isinstance(node, cast.Decl):
        if node.init is not None:
            assigned[node.name] = (
                assigned.get(node.name, 0) + 1
            )  # a decl-init: counted, but NOT body
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

    kind: str  # "var" | "mem"
    rid: int  # variable rid OR base rid
    ct: CType  # variable type OR accessed element/field type
    idx: int | None = None  # index rid for base[idx]
    byte_off: int = 0  # member byte offset (in imm, not the strict-bounds field)
    bit_off: int = 0  # bitfield bit offset within its storage unit
    bit_width: int = 0  # bitfield width (0 == plain access)
    member: bool = False  # a member-array element `s.arr[i]` (carries offset+size even
    # at offset 0, distinct from a plain `base[idx]`)
    stride: int = 0  # array-of-structs element stride (`s.arr[i].field`): the access
    # lands at &base + byte_off + idx*stride but copies ct.size bytes
    # (stride sizeof(element) != the field copy size). 0 == stride is ct.size.
    packed: bool = False  # a bitfield in a PACKED struct: its storage unit spans only
    # ceil((bit_off+bit_width)/8) bytes, possibly straddling words

    def unit_bytes(self) -> int:
        """The bitfield storage-unit byte span: a PACKED field, and one whose type aligns below its size
        (`narrow_bitfield`: i386's 8-byte integers), covers only the bytes it straddles; otherwise the
        declared type width (an MMIO register must be accessed at its natural width, so a naturally
        aligned unit is unchanged)."""
        if self.bit_width and (self.packed or narrow_bitfield(self.ct)):
            return (self.bit_off + self.bit_width + 7) // 8
        return max(1, self.ct.size)


_INIT_MAX_FRAMES = 64  # the subobjects one brace list may nest into (the twin's IW_MAXFRAMES)

# A struct or union object takes only a value of its own type: its initializer, when not a brace list, is
# one expression of it (C11 6.7.9p13), and so is what `=` assigns it (6.5.16.1p1). CF-STRUCTINIT: the rails
# had lowered `struct s x = "a";`, `= 5`, `= <another struct>` and `x = 5;` to a copy the emit spells
# `x = 5;`, which does not compile. The twin refuses with the same words (`bcir_cfront.c`, `agg_value_ok`).
_STRUCT_INITIALIZED = "a struct or union is initialized by a brace list or a value of its own type"
_STRUCT_ASSIGNED = "a struct or union is assigned a value of its own type"
# The arms of `?:` that are pointers to functions -- designators or function-pointer objects -- point to one function
# type (C11 6.5.15p3); CF-FNSEL. The twin's `select_temp` refuses with the same words.
_FN_SELECTED = "the arms of `?:` point to functions of different types"
# A struct or union is no operand of an operator that takes a scalar -- arithmetic, bitwise, shift, relational,
# equality, logical, a compound assignment, an increment (C11 6.5.2.4p1, 6.5.3.1p1, 6.5.3.3p1, 6.5.5-6.5.14,
# 6.5.16.2p1-2) -- and converts to no scalar type: an initializer, an assignment, a `return`, an argument, a cast
# (6.5.16.1p1, 6.5.4p2). CF-STRUCTARITH: both rails had lowered `a += 5`, `a++` and `uint32_t k = a;` to an emit
# Clang rejects. The twin refuses with the same words (`bcir_cfront.c`, `agg_operand`, `scalar_value_ok`).
_STRUCT_OPERAND = (
    "a struct or union is an operand of an arithmetic, bitwise, logical or comparison operator"
)
_STRUCT_CONVERTED = "a struct or union is converted to a scalar type"


@dataclass
class _InitWalk:
    """What one full initializer's brace lists share (CF-BRACE): the object, the bit ranges stored so
    far, the member each initialized union took, and how many top-level elements were reached."""

    rid: int
    top_array: bool  # the object is an array: its own list stores whole elements indexed
    stores: list = field(default_factory=list)  # [(lo_bit, hi_bit)] of every store, in order
    unions: dict = field(default_factory=dict)  # (byte_off, union tag) -> the member it took
    top_n: int = 0  # the top-level elements reached (an inferred `T a[]` is sized by it)
    const: bool = False  # a static's initializer: its entries fold, its stores make its image
    image: list = field(default_factory=list)  # [(lo_bit, width, value)] of its stores, in order
    shape: bool = False  # a file-scope initializer, walked for its shape only: an entry is
    #   neither lowered nor folded, and a store records its range (CF-GBRACE)
    strings: set = field(default_factory=set)  # ... and the id of each string literal it met that
    #   initializes a character array, which the linkable emit spells as it is


@dataclass
class _IFrame:
    """One aggregate a brace list walks: the list's own object, or a sub-aggregate it entered by brace
    elision or a designator. `idx` is the next subobject; `count` None is an inferred array's."""

    ct: CType
    off: int
    idx: int = 0
    count: int | None = 0
    flat: bool = False  # the declared array or a row of it: its elements are the array's own
    top: bool = False  # the declared object's own list's object (its reach sizes an inferred array)


# --- a static's constant image (CF-STATICTAB) ---------------------------------------------------------
# A static's initializer runs once, before the program does -- never as stores at each call. So it is
# folded, not lowered into the body: the initializer walk runs in constant mode, each entry lowered as any
# expression is and the claims it made evaluated as C evaluates an integer constant expression -- each
# operation in its own type -- then discarded; each scalar the walk initializes is recorded in the static's
# image, which is rendered as its declaration's initializer. The twin's `kfold` / `init_image` / `kimage`.

# the one reason both rails give for an entry the fold cannot evaluate: a variable, a load, a call, an
# address, a floating value -- or arithmetic C leaves undefined (an overflow, a shift past the width, a
# division by zero), which is no constant either
_NOT_CONSTANT = "a static initializer is not an integer constant expression"
# a union given an anonymous member other than its first: C names that member only through its own
# leaves, which the rendered brace list does not spell
_ANON_UNION = "a static union initialized through an anonymous member other than its first"


class _KVal(NamedTuple):
    """A folded constant: its value and its C type -- an integer (`kind` "i") of `bits` bits, signed or
    not; a `_Bool` ("b"); or a null pointer ("p")."""

    v: int
    bits: int = 32
    signed: bool = True
    kind: str = "i"


def _kwrap(v: int, bits: int, signed: bool) -> int:
    """`v` reduced to a `bits`-bit integer type: modulo 2^bits -- C's conversion to an unsigned type, and
    the two's complement every target gives a signed one."""
    v &= (1 << bits) - 1
    return v - (1 << bits) if signed and v >> (bits - 1) else v


def _kfits(v: int, bits: int, signed: bool) -> bool:
    """`v` is a value of the `bits`-bit type: a signed result outside it overflowed, which is undefined."""
    return -(1 << (bits - 1)) <= v < (1 << (bits - 1)) if signed else 0 <= v < (1 << bits)


def _kpromote(a: _KVal) -> _KVal:
    """An operand's integer promotion (C11 6.3.1.1p2): a `_Bool` or a type narrower than `int` is an
    `int`. A pointer is no integer constant."""
    if a.kind == "p":
        raise CLowerError(_NOT_CONSTANT)
    return _KVal(a.v) if a.kind == "b" or a.bits < 32 else a


def _kcommon(a: _KVal, b: _KVal) -> "tuple[int, bool]":
    """The usual arithmetic conversions (6.3.1.8) of two promoted operands: the wider type; of a signed
    and an unsigned type, the unsigned one unless the signed one is wider."""
    if a.signed == b.signed:
        return max(a.bits, b.bits), a.signed
    u, s = (b, a) if a.signed else (a, b)
    return (u.bits, False) if u.bits >= s.bits else (s.bits, True)


_KARITH = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "and": lambda x, y: x & y,
    "or": lambda x, y: x | y,
    "xor": lambda x, y: x ^ y,
}
_KCMP = {
    "eq": lambda x, y: x == y,
    "ne": lambda x, y: x != y,
    "lt": lambda x, y: x < y,
    "gt": lambda x, y: x > y,
    "le": lambda x, y: x <= y,
    "ge": lambda x, y: x >= y,
}


def _kbin(suf: str, a: _KVal, b: _KVal) -> _KVal:
    """`a OP b` as C computes it: a comparison or a logical operator gives an `int`, a shift the promoted
    left operand's type, the rest the operands' common type -- an unsigned one wrapping, a signed one
    that overflows undefined."""
    a, b = _kpromote(a), _kpromote(b)
    if suf in ("land", "lor"):
        x, y = a.v != 0, b.v != 0
        return _KVal(int(x and y) if suf == "land" else int(x or y))
    if suf in ("shl", "shr"):
        if not 0 <= b.v < a.bits:  # a count outside the promoted width: undefined
            raise CLowerError(_NOT_CONSTANT)
        if suf == "shr":  # a negative signed value shifts arithmetically, as on every target
            return a._replace(v=a.v >> b.v)
        r = a.v << b.v
        if a.signed and (a.v < 0 or not _kfits(r, a.bits, True)):
            raise CLowerError(_NOT_CONSTANT)
        return a._replace(v=_kwrap(r, a.bits, a.signed))
    bits, signed = _kcommon(a, b)
    x, y = _kwrap(a.v, bits, signed), _kwrap(b.v, bits, signed)
    if suf in _KCMP:
        return _KVal(int(_KCMP[suf](x, y)))
    if suf in ("div", "mod"):
        if y == 0:
            raise CLowerError(_NOT_CONSTANT)
        q = abs(x) // abs(y) * (-1 if (x < 0) != (y < 0) else 1)  # C truncates toward zero
        if signed and not _kfits(q, bits, True):  # INT_MIN / -1 (and its remainder)
            raise CLowerError(_NOT_CONSTANT)
        r = q if suf == "div" else x - q * y
    elif suf in _KARITH:
        r = _KARITH[suf](x, y)
    else:
        raise CLowerError(_NOT_CONSTANT)
    if signed and not _kfits(r, bits, True):
        raise CLowerError(_NOT_CONSTANT)
    return _KVal(_kwrap(r, bits, signed), bits, signed)


def _kun(suf: str, a: _KVal) -> _KVal:
    """`OP a`: `!` gives an `int`; `-` and `~` the promoted operand's type (`-INT_MIN` overflows)."""
    if suf == "lnot":
        return _KVal(int(a.v == 0))
    a = _kpromote(a)
    if suf not in ("neg", "bnot"):
        raise CLowerError(_NOT_CONSTANT)
    r = -a.v if suf == "neg" else ~a.v
    if a.signed and not _kfits(r, a.bits, True):
        raise CLowerError(_NOT_CONSTANT)
    return a._replace(v=_kwrap(r, a.bits, a.signed))


def _ksel(c: _KVal, a: _KVal, b: _KVal) -> _KVal:
    """`c ? a : b` over arithmetic arms: the chosen arm, in the arms' common type."""
    a, b = _kpromote(a), _kpromote(b)
    bits, signed = _kcommon(a, b)
    return _KVal(_kwrap((a if c.v else b).v, bits, signed), bits, signed)


def _ktype(ct: "CType | None") -> _KVal:
    """The type a constant claim's temp declares -- a literal's, `sizeof`'s `size_t`, a cast's target --
    as a zero of it. A floating or `_BitInt` one is no integer constant expression here."""
    if ct is not None and ct.kind in ("pointer", "funcptr"):
        return _KVal(0, 0, False, "p")
    if ct is None or ct.kind != "scalar" or not ct.is_integer or ct.is_bitint:
        raise CLowerError(_NOT_CONSTANT)
    if ct.name in ("_Bool", "bool"):
        return _KVal(0, 1, False, "b")
    return _KVal(0, ct.size * 8, ct.signed)


def _kconvert(a: _KVal, t: _KVal) -> _KVal:
    """`a` converted to the type `t` (6.3.1.2-3): an integer wraps to its width, a `_Bool` is whether `a`
    is nonzero, and a pointer takes only the null pointer constant."""
    if t.kind == "b":
        return t._replace(v=int(a.v != 0))
    if t.kind == "p":
        if a.v:
            raise CLowerError(_NOT_CONSTANT)
        return t
    return t._replace(v=_kwrap(a.v, t.bits, t.signed))


def _kleaf(a: _KVal, ct: CType, bw: int) -> int:
    """The value the static's scalar subobject `ct` (a bit-field of `bw` bits) holds from the constant
    `a`: C's conversion as if by assignment. A pointer holds only the null pointer; a floating one holds
    the integer itself, which the rendered initializer converts exactly as the source's does."""
    if ct.kind in ("pointer", "funcptr"):
        return _kconvert(a, _KVal(0, 0, False, "p")).v
    if ct.kind != "scalar" or ct.name == "void":
        raise CLowerError(_NOT_CONSTANT)
    if ct.is_float:
        return a.v
    if ct.name in ("_Bool", "bool"):
        return int(a.v != 0)
    return _kwrap(a.v, bw or (ct.bit_width if ct.is_bitint else ct.size * 8), ct.signed)


def _kspell(v: int) -> str:
    """A static's scalar as its rendered initializer spells it: `Nu` -- exact in whatever type it
    initializes -- `-N`, and the one negative with no positive counterpart as an expression."""
    if v >= 0:
        return f"{v}u"
    return "(-9223372036854775807 - 1)" if v == -(1 << 63) else str(v)


# --- an integer constant expression the parser folds (CF-ENUMFOLD) ------------------------------------------------
# An enumerator's value, a case label, an array dimension and a designator are needed where they are parsed, so the
# parser folds them on the target's data model, each operation by the predicate a static's initializer folds with
# (`_kbin`, `_kun`, `_ksel`, `_kconvert`) -- over the expression's nodes, as `_kfold` goes over its claims. The
# twin's `ce_fold`.

# the one reason both rails give for one that is no integer constant expression (C11 6.6p6: an object, a call,
# `sizeof`, the comma, a floating constant, a pointer) or whose arithmetic C leaves undefined (a division by zero, a
# signed overflow, a shift past the width or of a negative value: 6.5p5, 6.5.7p4, 6.6p4), which is no constant either
ICE_NOT = "not an integer constant expression"
# ... an enumerator whose value no int holds, stated or counted on from INT_MAX (6.7.2.2p2): C23 gives one a wider
# type, which neither rail models -- an enumeration constant is an int on both
ENUM_NOT_INT = "an enumerator value not representable as int"
# ... and an array dimension that folds negative (6.7.6.2p1) or past INT_MAX, the twin holding a dimension in an int
DIM_RANGE = "an array dimension outside 0..INT_MAX"


def fold_constant(node, abi, live: bool = True) -> _KVal:
    """`node`, an integer constant expression, folded in C's types on the target `abi`: an integer constant in its C11
    6.4.4.1 type (the parser types it for the target), a character or enumeration constant an `int` (the parser
    substitutes an enumerator's value), `-`, `+`, `~`, `!`, every binary operator but the comma, `?:` and a cast to an
    integer type. An operand C does not evaluate (`live` False: the right one of `0 && e`, the arm `?:` does not take)
    is typed but never refused for its arithmetic -- the predicate runs on a zero and a one of its operands' types --
    as C evaluates it (6.5.13p4, 6.5.15p4). Raises `CLowerError(ICE_NOT)` for anything else."""
    try:
        return _kfold_node(node, abi, live)
    except (CLowerError, KeyError):  # a predicate's refusal, or an unknown type-name
        raise CLowerError(ICE_NOT) from None


def _kfold_node(node, abi, live: bool) -> _KVal:
    """One node of `fold_constant`'s expression."""
    if isinstance(node, cast.IntLit):
        ct = scalar(node.ctype if is_scalar_name(node.ctype) else "int", abi)
        return _kconvert(_KVal(node.value), _ktype(ct))
    if isinstance(node, cast.Unary) and node.op in ("-", "~", "!", "+"):
        a = _kfold_node(node.operand, abi, live)
        if node.op == "+":
            return _kpromote(a)
        r = _kun(_UN[node.op][1], a if live else a._replace(v=0))
        return r if live else r._replace(v=0)
    if isinstance(node, cast.Binary) and node.op in _BIN:
        a = _kfold_node(node.lhs, abi, live)
        if node.op in ("&&", "||"):  # the right one evaluates only when the left does not decide
            b = _kfold_node(node.rhs, abi, live and (a.v != 0) == (node.op == "&&"))
        else:
            b = _kfold_node(node.rhs, abi, live)
        if live:
            return _kbin(_BIN[node.op][1], a, b)
        return _kbin(_BIN[node.op][1], a._replace(v=0), b._replace(v=1))._replace(v=0)
    if isinstance(node, cast.Ternary):
        c = _kfold_node(node.cond, abi, live)
        a = _kfold_node(node.then, abi, live and c.v != 0)
        b = _kfold_node(node.els, abi, live and c.v == 0)
        return _ksel(c, a, b)
    if isinstance(node, cast.Cast):
        t = _ktype(_resolve_member_type(node.type, {}, abi))
        if t.kind == "p":  # it converts only to an integer type (6.6p6)
            raise CLowerError(ICE_NOT)
        return _kconvert(_kfold_node(node.operand, abi, live), t)
    raise CLowerError(ICE_NOT)


def _arith_class(ct: CType) -> int:
    """An arithmetic type's class for C's assignment conversion: 0 integer, 1 real floating, 2 complex
    (a complex type is floating too). A store between classes converts the value; within one it is a
    width or sign change the emit makes (`emit._store_conv`)."""
    return 2 if ct.is_complex else (1 if ct.is_float else 0)


def _access_order(mmio: bool, atomic: bool) -> dict:
    """The lane and hazard of a memory access (CF-ATOMIC): an access to an `_Atomic` object is an atomic
    operation -- lane A and the `atomic` hazard, the contract the C11 generics' claims carry (R5) -- a
    device access is ordered (lane H, `barriered`), and a device access to an `_Atomic` object is both
    (lane H, `atomic`, as R3's pass leaves an atomic claim it moves into the device domain). The twin's
    `mark_access` decides the same."""
    return {
        "lane": Lane.H if mmio else (Lane.A if atomic else Lane.U),
        "hazard": "atomic" if atomic else ("barriered" if mmio else "unique"),
    }


# The opcode of an atomic read-modify-write by its operator: the three the opcode set names, else the
# compare-and-swap every other one is (C11 6.5.16.2p3: a loop of them). The twin's `rmw_opcode`.
_RMW_OPCODE = {
    "add": Opcode.ATOMIC_ADD,
    "sub": Opcode.ATOMIC_SUB,
    "xor": Opcode.ATOMIC_XOR,
    "preinc": Opcode.ATOMIC_ADD,
    "postinc": Opcode.ATOMIC_ADD,
    "predec": Opcode.ATOMIC_SUB,
    "postdec": Opcode.ATOMIC_SUB,
}


def _object_write(ct: CType) -> dict:
    """The claim fields of a write to a named object of type `ct`: a volatile scalar's write is a
    volatile access (a device-domain claim, lane H, `barriered`); anything else is an ordinary copy
    (`_order_device_claims` still orders one that touches a device region, a volatile struct's whole
    copy included)."""
    if ct.volatile and ct.kind == "scalar":
        return {"domain": Domain.MMIO, "lane": Lane.H, "hazard": "barriered", "volatile": True}
    return {}


def _order_device_claims(claims, resources) -> None:
    """R3 over a finished function, one pass (the C twin runs the same, `bcir_cfront.c`'s
    `order_device_claims`): a claim that touches a device region is a device-domain claim and ordered
    (lane H, `barriered` unless it is already `atomic`) -- the copy that binds a pointer to volatile
    storage, the load of a pointer member that points at some, pointer arithmetic or a call over one.
    It is a volatile ACCESS only if it reads or writes volatile storage itself: `Claim.volatile` is set
    where the lvalue is resolved and is left alone here."""
    for c in claims:
        if c.domain == Domain.MMIO:
            continue
        if any(
            (r := resources.get(rid)) is not None and r.domain == Domain.MMIO
            for rid in (*c.rd, *c.wr)
        ):
            c.domain, c.lane = Domain.MMIO, Lane.H
            if c.hazard == "unique":
                c.hazard = "barriered"


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
    cond: int  # rid of the condition value
    then: list  # block (list of Claim | IfNode | WhileNode)
    els: list


@dataclass
class WhileNode:
    cond_block: list  # claims that recompute the condition each iteration
    cond: int  # rid of the condition value
    body: list
    bound: int = 1024  # static iteration upper bound (for the cost model)
    test_at_end: bool = False  # do/while: run body first, then test the condition
    step: list = field(
        default_factory=list
    )  # for-loop step: runs after body, at the continue point
    loop_id: int = 0  # unique id for the emitter's `__cont_<id>` label


@dataclass
class ReturnNode:
    rid: int | None  # the returned value's rid (None == `return;`)


@dataclass
class BreakNode:
    pass  # `break;` -- exit the nearest enclosing loop (emit-only)


@dataclass
class ContinueNode:
    pass  # `continue;` -- jump to the loop's continue point (emit-only)


@dataclass
class GotoNode:
    label: str  # `goto label;` -- an unconditional jump (emit-only)


@dataclass
class LabelNode:
    name: str  # `name:` -- a jump target (emit-only)


@dataclass
class ComputedGotoNode:
    target: int  # `goto *<target>;` -- an indirect jump to a label address (GNU)


@dataclass
class AsmInfo:
    """The verbatim, ISA-neutral payload of one inline-asm claim (ASM1) -- everything the emitter needs to
    reconstruct the exact GNU `__asm__` statement, since the claim's `imm` is int-only. Keyed by claim id in
    `LoweredFunc.asm_meta`. The template + constraints + clobbers are SOURCE spellings (re-emitted unchanged
    -> ISA-neutral pass-through); the operand RIDs live on the claim's `rd`/`wr` (so the alias/effect/verify
    machinery sees the real read/write set). The outputs + inputs are carried as separate parallel lists so
    the emitter splits them back into the two `:` sections; `out_rids` / `in_rids` are the per-operand RIDs in
    source order so the constraints pair up 1:1 with their refs. `is_basic` records the BASIC source form (no
    colon sections) so the emit re-renders the basic 1-colon form ONLY there -- an all-empty EXTENDED asm
    re-emits with its (empty) `:` sections, so a non-volatile `asm("x" : :)` round-trips as non-volatile."""

    template: str
    out_names: tuple  # per-output (symbolic_name | None)
    out_constraints: tuple  # per-output constraint source spelling ("=r", "+r", …)
    out_rids: tuple  # per-output destination RID
    in_names: tuple  # per-input  (symbolic_name | None)
    in_constraints: tuple  # per-input  constraint source spelling ("r", "m", …)
    in_rids: tuple  # per-input  value RID
    clobbers: tuple  # clobber source spellings ("\"memory\"", "\"cc\"", …)
    is_volatile: bool
    is_basic: bool  # the BASIC source form (no colon sections) -- gates the basic-form emit


@dataclass
class SwitchNode:
    disc: int  # the discriminant rid (lowered once)
    body: list  # flat: CaseLabel | DefaultLabel | Claim | control nodes


@dataclass
class CaseLabel:
    value: int  # `case <value>:` -- a folded integer constant


@dataclass
class DefaultLabel:
    pass  # `default:`


def _flatten_block(block: list) -> list:
    """All Claim objects in a body tree, in order (the flat single-phase view verify/plan use)."""
    out: list = []
    for node in block:
        if isinstance(node, IfNode):
            out += _flatten_block(node.then) + _flatten_block(node.els)
        elif isinstance(node, WhileNode):
            out += (
                _flatten_block(node.cond_block)
                + _flatten_block(node.body)
                + _flatten_block(node.step)
            )
        elif isinstance(node, SwitchNode):
            out += _flatten_block(node.body)
        elif isinstance(node, (Claim,)):
            out.append(node)
    return out


# --- the operands C may leave unevaluated (CF-TERNARY) ---
# C evaluates exactly one arm of `c ? a : b` (C11 6.5.15p4) and the right operand of `&&` / `||` only when
# the left one does not decide the result (6.5.13p4, 6.5.14p4). An operand may still be computed eagerly --
# as a `c.select` or a `c.bin.land` / `c.bin.lor` over values already in hand -- when computing it can
# neither trap nor change state: every claim it made is one of these ops, is no volatile access (a device
# register's included: the lowering marks it volatile as it makes it) and writes no declared variable. An
# atomic access is a load, a store or a read-modify-write, none of them on the list. Any other operand is
# lowered as a branch that evaluates it only when C does. The rails decide on the claims each made, which
# parity already holds equal, so they cannot classify an operand differently; the twin's `operand_pure` is
# this predicate.
_PURE_OPS = frozenset(
    {"c.const", "c.copy", "c.select", "c.addrof", "c.ptradd", "c.ptrsub", "c.sizeof.vla"}
)
_PURE_OP_PREFIXES = ("c.fconst:", "c.cconst:", "c.cast:", "c.labeladdr:")
_PURE_BIN = frozenset(
    "add sub mul and or xor shl shr lt gt le ge eq ne land lor".split()
)  # not `div` / `mod`: a zero divisor traps
_PURE_UN = frozenset({"neg", "bnot", "lnot", "creal", "cimag"})


def _operand_pure(block: list, declared: set) -> bool:
    """Whether the operand `block` lowered to may be computed although C would not evaluate it."""
    for n in block:
        if not isinstance(n, Claim):  # an operand it already branches on
            return False
        if n.volatile:  # a volatile access: reading one is a side effect
            return False
        if any(w in declared for w in n.wr):  # an assignment, an increment
            return False
        op = n.op
        if op in _PURE_OPS or op.startswith(_PURE_OP_PREFIXES):
            continue
        if op.startswith("c.bin.") and op[len("c.bin.") :] in _PURE_BIN:
            continue
        if op.startswith("c.un.") and op[len("c.un.") :] in _PURE_UN:
            continue
        return False
    return True


def callee_signature(fct) -> str:
    """The declared indirect-callee signature "ret(param, ...)" (§5.14 Phase 2) rendered from a
    funcptr CType -- the type record the R18 well-formedness check and the effect/commutation
    analysis consume. "" when the type is unknown (the claim stays a fully opaque edge)."""
    if fct is None or getattr(fct, "kind", "") != "funcptr":
        return ""

    def nm(ct):
        return "void" if ct is None else (ct.name or ct.kind)

    return f"{nm(fct.of)}({', '.join(nm(p) for p in fct.params)})"


def _returns_void(fct) -> bool:
    """A function-pointer type whose function returns `void` -- a call through it has no value, so it
    writes no result temp, as a direct call to a void function does (CF-VOIDCB; the twin's
    `bcir_ctype.fp_ret_void`). A return type that was not captured (`of` None) is not void."""
    return (
        fct is not None
        and fct.kind == "funcptr"
        and fct.of is not None
        and fct.of.kind == "scalar"
        and fct.of.name == "void"
    )


@dataclass
class LoweredFunc:
    """One C function lowered: its claim-graph `Module`, the value/SSA bookkeeping the emitter needs,
    and the call sites (for the call graph / R18)."""

    name: str
    module: Module
    ret_type: CType
    params: list  # (name, rid, CType) in C order
    return_rid: int | None
    claims: list  # all claims, flat (== the single phase) for verify/plan
    resources: dict  # rid -> Resource
    rid_types: dict = field(default_factory=dict)  # rid -> CType (for faithful C emission)
    calls: list = field(default_factory=list)  # (callee, (actual_rids...))
    region: object = None  # compose.Region
    body: list = field(default_factory=list)  # the structured body tree (for emission)
    locals: list = field(default_factory=list)  # (rid, name, CType) mutable named locals
    vla_locals: list = field(
        default_factory=list
    )  # (rid, name, CType) 1-D stack VLAs -- declared IN-BODY
    statics: list = field(default_factory=list)  # (rid, name, CType, init) static-storage locals:
    #   init the rendered initializer of its constant image, None when it is zero (CF-STATICTAB)
    thread_statics: frozenset = frozenset()  # the statics of thread storage duration, each
    #   thread's own object, declared `static _Thread_local` (CF-TLS)
    globals_used: dict = field(default_factory=dict)  # rid -> name (file-scope globals referenced)
    zero_init_locals: set = field(default_factory=set)  # aggregate-local rids declared `= {}`
    tu_protos: dict = field(
        default_factory=dict
    )  # cross-TU callee -> (ret CType, (param CType, ...), (param points to const, ...))
    #   -- prototyped, not defined here; the emit declares them
    variadic: bool = False  # a trailing `...` after the named params (variadic function)
    reproducible: bool = False  # a C23 `[[reproducible]]`/`[[unsequenced]]` hint is on the
    #   definition -- a fusion-legality signal (the hint is value-
    #   neutral, so the emit drops it). Default False == every
    #   un-annotated function, undisturbed.
    static_fn: bool = False  # source `static` on the definition -- the linkable emit keeps
    #   the rendered definition `static` (internal linkage honored)
    ptr_extent: dict = field(
        default_factory=dict
    )  # §5.12: pointer rid -> recovered extent (count) variable rid
    asm_meta: dict = field(
        default_factory=dict
    )  # ASM1: claim id -> AsmInfo (the verbatim inline-asm payload
    #   the emitter reconstructs the `__asm__(...)` statement from)
    target: str = HOST.name  # ASM2: the target ABI short name the unit lays out for --
    #   threaded to the emit so the per-ISA port-I/O `in`/`out`
    #   realization keys off `--target` (x86 emits; non-x86 is
    #   an honest unsupported diagnostic -> LLVM fallback)
    target_explicit: bool = False  # ASM3: True iff the caller chose `--target` EXPLICITLY.
    #   The native per-ISA memory-fence asm (mfence / dmb ish /
    #   fence rw,rw) is emitted only then; an unspecified
    #   (host-default) target emits the PORTABLE
    #   __atomic_thread_fence, so the default emit compiles on
    #   ANY host (a cross-arch native compile never sees foreign
    #   asm). Port-I/O has no portable form, so it is unaffected.


@dataclass
class LoweredUnit:
    functions: dict  # name -> LoweredFunc
    entry: str
    aggregates: dict  # tag -> CType
    compose_functions: dict  # name -> compose.Function
    resources: dict  # rid -> Resource (whole unit)
    globals_decl: tuple = ()  # (name, CType, rendered init | None, extern, static) -- what the
    #   LINKABLE emit renders: the initializer as a 1-tuple of its
    #   re-spelled text, () for none (`_file_scope_rendering`);
    #   extern -> a declaration; static -> kept file-local.
    #   None == not renderable in this slice.
    thread_globals: frozenset = frozenset()  # the globals of thread storage duration: the
    #   linkable emit declares them `_Thread_local` (CF-TLS)
    init_refs: frozenset = frozenset()  # every identifier a file-scope initializer names: a
    #   function among them has its address taken (an ops table
    #   `struct ops t = { handler };`) -- the escape analysis
    #   then assumes callers it cannot see
    anon_spelling: dict = field(default_factory=dict)  # how C names each anonymous aggregate (the
    #   parse's `Unit.anon_spelling`), which the emit's `respell_anon` spells it by


# the rid `_call` returns for a void callee -- never read as a value (a void call is a statement); the
# Return lowering maps it back to a bare `return;`.
_VOID_RID = -1

# ASM2 -- the RESERVED rid of the I/O-port "address space" resource (`__ioport`), the single MMIO-domain
# resource ALL port accesses share so they alias each other (ordered) but never normal memory (isolation
# invariant 3). It must be DISJOINT from every count-indexed band -- the function-local temps (`base_rid +
# k`, base = 100 + idx*1000), the shared-global band (`900000 + gi`), and the string/func-value band
# (`970000 + idx`) -- ALL of which grow upward without a fixed ceiling. So the sentinel is parked FAR above
# any reachable band (0x40000000 == 1,073,741,824; a unit would need ~10^9 globals/literals to approach it),
# and -- belt + suspenders -- the unbounded allocators raise an honest CLowerError if they ever reach it
# (`_check_band_rid`), and the global/string merge in `lower()` never overwrites the reserved MMIO resource. The
# reservation is thus PROVABLE (a guarded disjoint constant), not a numeric-gap guess. A positive constant
# (not a negative sentinel) keeps `sorted(self.resources)` / the provenance-digest resource ordering / every
# rid-as-dict-key path on the same non-negative footing every other rid already uses.
_IO_PORT_RID = 0x4000_0000


def _check_band_rid(rid: int) -> int:
    """Honest deterministic guard for the count-indexed rid bands: a translation unit so large that its
    global / string-literal allocator reaches the RESERVED I/O-port rid (`_IO_PORT_RID`) is rejected with a
    clear diagnostic instead of silently colliding (and then having the global/string resource overwrite the
    `__ioport` MMIO resource -- the isolation hole this guards). Unreachable in practice (~10^9 globals or
    literals), but it makes the reservation PROVABLE rather than a numeric-gap assumption. Returns `rid`."""
    if rid == _IO_PORT_RID:
        raise CLowerError(
            "translation unit exhausted the rid band: reached the reserved I/O-port rid -- too many "
            "globals / string literals in one unit"
        )
    return rid


class _FuncLowerer:
    def __init__(
        self,
        func: cast.Func,
        aggregates: dict,
        base_rid: int,
        cid: list,
        genv: dict | None = None,
        gres: dict | None = None,
        strctr: list | None = None,
        func_rets: dict | None = None,
        abi=None,
        protos: dict | None = None,
        func_params: dict | None = None,
        proto_consts: dict | None = None,
        func_variadic: frozenset = frozenset(),
        lowered: dict | None = None,
    ):
        self.func = func
        self.abi = abi or HOST  # the target data model for laying out user types (long/ptr)
        self.func_rets = (
            func_rets or {}
        )  # callee name -> return CType (for void / wide / float returns)
        self.protos = protos or {}  # PROTOTYPED cross-TU callee -> (ret CType, (param CType, ...))
        # same-unit callee -> its parameters' types, None where the pre-scan cannot type one
        # (`_param_types`): what a null pointer constant argument becomes (CF-NULLARG)
        self.func_params = func_params or {}
        # the same-unit functions declared with a trailing `...`: a designator of one has no function type a
        # declarator here can spell (`_fn_type`, CF-FNSEL)
        self.func_variadic = func_variadic
        # the unit's functions lowered before this one: name -> LoweredFunc (`_fn_type`)
        self.lowered = lowered if lowered is not None else {}
        # the function pointers read from a member or an element, which the twin loads as integers: `_fn_type`
        # gives them no function type
        self.fn_loaded: set[int] = set()
        self.tu_used: dict = {}  # the tu callees THIS function actually calls (for the emit decl)
        # a prototyped callee -> whether each parameter points to `const`, which its emitted `extern`
        # declaration keeps (a `const T *` parameter is another type than a `T *` one)
        self.proto_consts = proto_consts or {}
        self.aggregates = aggregates
        self.rid = base_rid
        self.cid = cid  # shared mutable [next_claim_id]
        self.genv = genv or {}  # file-scope globals: name -> (rid, CType)
        self.gres = gres or {}  # global rid -> Resource
        self.strctr = (
            strctr if strctr is not None else [0]
        )  # shared string-literal counter (unique rids)
        self.str_globals: dict[int, str] = {}  # string-literal global rid -> the source spelling
        self.str_pool: dict[str, int] = {}  # spelling -> rid (dedup identical literals)
        self.func_globals: dict[
            int, str
        ] = {}  # function-as-value rid -> the function name (emit verbatim)
        self.func_pool: dict[str, int] = {}  # function name -> rid (dedup)
        self.env: dict[str, tuple[int, CType]] = {}  # name -> (storage rid, type)
        self.resources: dict[int, Resource] = {}
        self.rtypes: dict[int, CType] = {}  # rid -> CType (for emission)
        self.zero_consts: set[int] = set()  # `c.const 0` temps -- a null pointer constant where a
        #   pointer takes one (`_null_pointer`, CF-NULLPTR)
        self.params: list = []
        self.locals: list = []  # (rid, name, CType) mutable named locals
        self.vla_locals: list = []  # (rid, name, CType) 1-D stack VLAs -- declared IN-BODY
        #   (`T a[__ext];`) at the source decl, NOT up front
        self.vla_strides: dict = {}  # multi-dim VLA array rid -> per-dim snapshot rids (the
        #   runtime Horner multipliers: m[i][j] -> i*dim1 + j)
        self.zero_init: set = set()  # aggregate-local rids that emit the `= {}` zero baseline
        self.statics: list = []  # (rid, name, CType, init) static-storage locals
        self.thread_statics: set = set()  # the statics of thread storage duration (CF-TLS)
        self.calls: list = []
        self.block_stack: list = [[]]  # claims/control nodes append to the top block
        self.loop_ctr = 0  # unique loop ids (the `continue` label numbering)
        self.cl_ctr = 0  # unique compound-literal ids (anonymous `_cl<N>` locals)
        # §5.12 recoverable extents: a pointer local bound to malloc(N*sizeof(T)) / calloc(N, sizeof(T)) carries
        # a RECOVERED element-count -> its `p[i]` accesses promote to `masked` (runtime-bounds-checked against N).
        self.ptr_extent: dict[
            int, int
        ] = {}  # pointer rid -> the count VARIABLE's rid (re-emitted by name)
        self.ptr_extent_kind: dict[
            int, str
        ] = {}  # pointer rid -> extent-provenance kind (§5.14 Phase 2:
        #   "recovered_count" stable-Name | "snapshot_extent")
        self._mut_assigned: dict[
            str, int
        ] = {}  # name -> TOTAL assignments incl. decl-init (pointer gate)
        self._mut_body: dict[str, int] = {}  # name -> NON-decl-init assignments only (count gate)
        self._mut_addr: set[str] = set()  # names whose address is taken anywhere (`&x`)
        self._ext_ctr = 0  # unique hidden extent-snapshot locals (`__bcir_extK`)
        self.asm_meta: dict[
            int, AsmInfo
        ] = {}  # ASM1: claim id -> AsmInfo (verbatim inline-asm payload)

    def _next_loop_id(self) -> int:
        self.loop_ctr += 1
        return self.loop_ctr

    def _complete(self, ct: "CType | None") -> "CType | None":
        """An incomplete struct or union -- named where it was not laid out yet: the pointee of a member
        pointer to the struct being defined, or to a later one -- completed by its tag, now that every
        aggregate is laid out (CF-SELFREF). One never defined stays incomplete."""
        if ct is not None and ct.incomplete and ct.name in self.aggregates:
            full = self.aggregates[ct.name]
            return with_volatile(full) if ct.volatile else full
        return ct

    def _complete_ptr(self, ct: CType) -> CType:
        """A member's type with the incomplete struct it points at (through pointers and arrays of them)
        completed, so `a.next->v` and `a.next + 1` see the pointee's layout."""
        if ct.kind in ("pointer", "array") and ct.of is not None:
            of = (
                self._complete_ptr(ct.of)
                if ct.of.kind in ("pointer", "array")
                else self._complete(ct.of)
            )
            if of is not ct.of:
                return replace(ct, of=of)
        return ct

    def _field(self, agg: "CType | None", name: str) -> tuple:
        """Member `name` of the struct or union `agg` -- (type, byte offset, bit offset, bit width) -- its
        type's incomplete pointee completed. A member of a non-aggregate, of a struct never defined, or
        one the struct lacks is a lowering error, never a bare AttributeError or KeyError."""
        agg = self._complete(agg)
        if agg is None or not agg.is_aggregate:
            raise CLowerError(f"a member access `{name}` into a non-aggregate")
        if agg.incomplete:
            raise CLowerError(f"a member access into the incomplete struct or union {agg.name!r}")
        try:
            ftype, off, bo, bw = agg.field(name)
        except KeyError:
            raise CLowerError(f"no member named {name!r} in {agg.kind} {agg.name!r}") from None
        return self._complete_ptr(ftype), off, bo, bw

    # --- resource allocation ---
    def _new_rid(self) -> int:
        self.rid += 1
        return self.rid

    def _resolve_type(self, tref: cast.TypeRef) -> CType:
        if tref.vla is not None:  # `T a[n]` (runtime size): native lowering is a
            raise CLowerError(  # follow-on (the emitter's up-front local declaration
                "variable-length array native lowering is a follow-on"
            )  # can't size it) -- route to fallback
        if tref.funcptr:  # a function-pointer alias (HAL dispatch)
            ret = self._resolve_type(tref.func_ret)
            params = tuple(self._resolve_type(p) for p in tref.func_params)
            return funcptr(tref.base, ret, params, self.abi)
        if tref.typeof_var:  # `typeof(var)` -> the in-scope variable's type
            if tref.typeof_var not in self.env:
                raise CLowerError(f"typeof of unknown variable {tref.typeof_var!r}")
            base = self.env[tref.typeof_var][1]
        elif tref.typeof_expr is not None:  # `typeof(expr)` -> the operand's static type
            base = self._type_of(tref.typeof_expr)
        elif tref.aggregate:  # a pointer may name a struct never defined (an opaque `struct fwd *`)
            base = _aggregate(self.aggregates, tref.base, tref.aggregate, pointee=tref.ptr > 0)
        elif tref.base == "va_list":  # the <stdarg.h> variadic cursor (opaque)
            base = valist(self.abi)
        elif tref.bit_width:  # C23 `_BitInt(N)`: an exact-width, non-promoting int
            base = bitint(tref.bit_width, signed="unsigned" not in tref.base, abi=self.abi)
        elif is_scalar_name(tref.base):
            base = scalar(tref.base, self.abi)
        else:
            raise CLowerError(f"unknown type {tref.base!r}")
        if "volatile" in tref.quals:  # volatile pointee/object -> MMIO region
            base = with_volatile(base)
        if "_Atomic" in tref.quals:  # _Atomic-qualified object (C11/C23), laid out by the ABI
            base = _atomic_type(base, self.abi)
        t = base
        for _ in range(tref.ptr):
            t = pointer(t, self.abi)
        for dim in reversed(tref.array):
            t = array(t, dim)
        return t

    def _resource(self, rid: int, ct: CType, name: str) -> None:
        mmio = ct.touches_mmio  # a volatile object / pointer-to-volatile
        if ct.kind == "array":
            shape, elem = (ct.count or 1,), ct.of.size
        elif ct.kind == "pointer":
            shape, elem = (1 << 16,), ct.of.size if ct.of else 1  # symbolic pointee extent
        elif ct.is_aggregate:
            shape, elem = (1,), max(1, ct.size)
        else:
            shape, elem = (1,), max(1, ct.size)
        self.resources[rid] = Resource(
            rid=rid,
            domain=Domain.MMIO if mmio else Domain.RAM,
            elem_bytes=elem,
            shape=shape,
            access="volatile" if mmio else "rw",
            name=name,
        )
        self.rtypes[rid] = ct

    def _emit(
        self,
        op: str,
        opcode: Opcode,
        rd: tuple,
        wr: tuple,
        *,
        imm: tuple = (),
        lane=Lane.U,
        stride=StrideClass.SCALAR,
        count: int = 1,
        domain=Domain.RAM,
        bounds: str = "strict",
        hazard: str = "unique",
        lifetime=None,
        volatile: bool = False,
        bounds_provenance: str = "",
        callee_sig: str = "",
    ) -> int:
        self.cid[0] += 1
        self.block_stack[-1].append(
            Claim(
                id=self.cid[0],
                opcode=opcode,
                lane=lane,
                stride_class=stride,
                count=count,
                rd=tuple(rd),
                wr=tuple(wr),
                imm=tuple(imm),
                domain=domain,
                op=op,
                bounds=bounds,
                hazard=hazard,
                lifetime=lifetime,
                volatile=volatile,
                bounds_provenance=bounds_provenance,
                callee_sig=callee_sig,
            )
        )
        if op == "c.const" and tuple(imm) == (0,):
            self.zero_consts.add(wr[0])
        return wr[0] if wr else -1

    def _null_pointer(self, v: int, ct: CType) -> int:
        """The value `v` taken by an object of type `ct`: a `c.const 0` taken by a pointer is a null pointer
        constant (C11 6.3.2.3p3), so its temp is typed as that pointer -- the emit declares `T *t = 0u;`,
        where an `int` temp made `p = t;` an integer VARIABLE assigned to a pointer, which Clang and GCC
        reject. The claim graph is unchanged: only the temp's C type moves (CF-NULLPTR)."""
        if v in self.zero_consts and ct.kind in ("pointer", "funcptr"):
            self.rtypes[v] = unqualified(ct)
        return v

    def _null_pointer_args(self, actuals: tuple, params: tuple) -> None:
        """The actuals of a call to a callee whose parameter types are known -- a same-unit definition,
        before or after the caller, or a prototype: an argument converts to its parameter's type as if by
        assignment (C11 6.5.2.2p7), so a `c.const 0` passed to a pointer parameter is a null pointer of that
        parameter's type (`_null_pointer`). A variadic callee's extra actuals have no parameter and keep
        their type. Only the temps' C types move (CF-NULLARG; the twin's `null_pointer_args`)."""
        for v, ct in zip(actuals, params):
            if ct is not None:
                self._null_pointer(v, ct)
                self._scalar_value(ct, v)  # a struct for a scalar parameter (CF-STRUCTARITH)

    def _storage(self, ct: CType, name: str) -> int:
        """A mutable named local — assignments write it, reads read it (so control-flow merges and
        loop accumulators work). The emitter declares it and refers to it by name."""
        rid = self._temp(ct, name)
        self.locals.append((rid, name, ct))
        return rid

    def _declared_rids(self) -> set:
        """The variables the source declares -- parameters, locals, statics, globals -- as opposed to the
        temporaries the lowering makes (`_operand_pure`: writing one is a side effect)."""
        rids = {rid for _n, rid, _ct in self.params}
        rids.update(rid for rid, *_ in (*self.locals, *self.vla_locals, *self.statics))
        rids.update(rid for rid, _ct in self.genv.values())
        return rids

    def _lower_apart(self, expr) -> tuple:
        """An operand C may leave unevaluated, lowered into a block of its own -- (the block, its value) --
        so the caller can compute it eagerly (splice the block in place) or under a branch (CF-TERNARY)."""
        block: list = []
        self.block_stack.append(block)
        try:
            v = self._rvalue(expr)
        finally:
            self.block_stack.pop()
        return block, v

    def _branch_value(self, cond: int, ct: CType, then: list, els: list) -> tuple:
        """The branch an operand C may leave unevaluated lowers to: one named local of type `ct` that each
        arm assigns last -- `if (cond) { then; sel = ...; } else { els; sel = ...; }` -- exactly the source
        form both rails already lower and emit. Returns (the local, the IfNode) for the caller to finish
        each arm and place the node (`_assign_in`)."""
        sel = self._storage(ct, "sel")
        return sel, IfNode(cond, then, els)

    def _assign_in(self, block: list, v: int, sel: int) -> None:
        """`sel = v;` at the end of `block` (one arm of `_branch_value`)."""
        self.block_stack.append(block)
        try:
            self._emit("c.copy", Opcode.ADD, (v,), (sel,))
        finally:
            self.block_stack.pop()

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

    def _static_storage(self, ct: CType, name: str, init: "str | None") -> int:
        """A `static` local: persistent storage with a once-only constant initializer baked into the
        declaration (so the init is NOT a per-call assignment) -- `init` its rendered image, None for
        zero (`_static_init`). Reads/writes hit it like any local."""
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

    @staticmethod
    def _aos_rooted(node) -> bool:
        """`a[i].m...k`: a `.` member chain whose innermost base is a subscript (no `->` hop on the way)."""
        while isinstance(node, cast.Member) and not node.arrow:
            if isinstance(node.base, cast.Index):
                return True
            node = node.base
        return False

    def _aos_member(self, node) -> "tuple | None":
        """A member of an array-of-structs element -- `a[i].f`, and a nested struct/union member at any depth
        `a[i].m.k` (CF-NESTMEM) -- as (the element's lvalue, the element struct, the member's type, its byte
        offset in the element, bit offset, bit width, the aggregate it is a member of). The chain's `.` hops
        only add offsets (and a volatile aggregate's qualifier): the access keeps the element's runtime index
        and strides by the element. None when the chain reaches no subscripted struct element (a `->` hop
        goes through a pointer, which the element's own storage does not hold). The twin's `sdef_elem_field`
        and `aos_member_array` descend the same chain."""
        hops, n = [], node
        while isinstance(n, cast.Member) and not isinstance(n.base, cast.Index):
            if n.arrow:
                return None
            hops.append(n.field)
            n = n.base
        if not isinstance(n, cast.Member):
            return None
        hops.append(n.field)
        el = self._lvalue(n.base)
        if el.kind != "mem" or el.idx is None or el.ct.kind not in ("struct", "union"):
            return None
        elem = self._complete(el.ct)
        agg, ftype, off, fbo, fbw = elem, elem, 0, 0, 0
        for name in reversed(hops):  # the element's struct, then each nested struct/union reached
            agg = self._complete(ftype)
            ftype, foff, fbo, fbw = self._field(agg, name)
            ftype = qualified(ftype, agg.volatile)  # a member of a volatile aggregate
            off += foff
        return el, elem, ftype, off, fbo, fbw, agg

    # --- lvalue resolution ---
    def _lvalue(self, node) -> "_LV":
        if isinstance(node, cast.Name):
            rid, ct = self._lookup(node.ident, node.pos)
            return _LV("var", rid, ct)
        if isinstance(
            node, cast.CompoundLiteral
        ):  # `&(int){v}`, `(struct P){...}.field` -- an lvalue
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
            aos_idx = aos_k = (
                None  # `a[i].m[j]`: the element's index and its stride in `m`'s elements
            )
            whole = False  # a multi-dimensional global indexed like a member array at offset 0
            if isinstance(n, cast.Member) and self._aos_rooted(n):
                # `a[i].m[j]` -- a member ARRAY of an array-of-structs element (CF-SMALL), at any depth of
                # nested struct/union members (`a[i].s.m[j]`, CF-NESTMEM): the element's index, scaled to the
                # member's element size, joins the member's own flattened index, and the access is the member
                # element at `&a + offsetof(s.m) + (i*K + lin)*es`, K = the element struct's size in elements
                # (a struct whose size is no multiple of the element size is refused)
                aos = self._aos_member(n)
                if aos is None or aos[0].member:
                    raise CLowerError(
                        "a member array of a subscripted element that is not a plain array-of-structs element"
                    )
                el, agg, ftype, moff = aos[:4]
                dims, t = [], ftype
                while t.kind == "array":
                    dims.append(t.count)
                    t = t.of if t.of is not None else scalar("uint32_t")
                if not dims or t.kind != "scalar" or len(dims) > 3:
                    raise CLowerError(
                        f"a subscript of the array-of-structs member '{n.field}' is not supported"
                    )
                if len(idx_nodes) != len(dims):
                    raise CLowerError(
                        f"partial indexing of a struct member array ('{n.field}') is not yet supported"
                    )
                if agg.size % t.size:
                    raise CLowerError(
                        "an array-of-structs member array whose element does not divide the struct"
                    )
                base_rid, base_ct, byte_off = el.rid, ftype, el.byte_off + moff
                mem_shape, mem_elem, aos_idx, aos_k = tuple(dims), t, el.idx, agg.size // t.size
            elif isinstance(n, cast.Member):
                # `s.arr[i]` / `s.m[i][j]` -- indexing a struct *member* array. The base is the enclosing
                # struct; the member's byte offset rides in the access imm beside the element size, and
                # the row-major flattened index is scaled by the element, so the element lands at
                # `&s + member_off + lin*elem_size` -- never the enclosing struct's offset 0.
                base_rid, struct_ct, base_off = self._addr(n.base)
                agg = self._complete(struct_ct.of if n.arrow else struct_ct)
                ftype, byte_off, _bo, _bw = self._field(agg, n.field)
                ftype = qualified(ftype, agg.volatile)  # a member of a volatile aggregate
                byte_off += (
                    0 if n.arrow else base_off
                )  # ride the enclosing offset (a non-first member array)
                if ftype.kind == "pointer":
                    # a *pointer* member subscripted (`s->p[i]` == `*(s->p + i)`): load the full pointer
                    # field, then index the loaded pointer like a plain `base[idx]` (no member offset).
                    # (Pointer-value slice 2b -- the load is an 8-byte pointer, not a truncating uint32.)
                    base_rid = self._read(_LV("mem", base_rid, ftype, byte_off=byte_off))
                    base_ct, byte_off, ptr_member = ftype, 0, True
                elif ftype.kind == "array":
                    dims, t = [], ftype  # descend the (nested) member array:
                    while t.kind == "array":  # per-dim sizes + the ultimate scalar element
                        dims.append(t.count)
                        t = t.of if t.of is not None else scalar("uint32_t")
                    if len(dims) > 3:
                        # the dim table (oracle shape / twin field.adims) holds at most 3 dims; defer deeper.
                        raise CLowerError(
                            f"indexing a >3-dimensional struct member array ('{n.field}') is not yet supported"
                        )
                    if len(idx_nodes) != len(dims):
                        # partial indexing of a member array (a row pointer `s.m[i]`): defer to fallback.
                        raise CLowerError(
                            f"partial indexing of a struct member array ('{n.field}') is not yet supported"
                        )
                    base_ct, mem_shape, mem_elem = ftype, tuple(dims), t
                else:
                    raise CLowerError(
                        f"indexing a non-array struct member ('{n.field}[...]') is not yet supported"
                    )
            else:
                base_rid, base_ct, byte_off = self._addr(
                    n
                )  # a non-Member base: its accumulated offset
                dims, t = [], base_ct
                while t.kind == "array" and not t.shape:
                    dims.append(t.count)
                    t = t.of if t.of is not None else scalar("uint32_t")
                if len(dims) > 1 and len(dims) == len(idx_nodes) and t.kind == "scalar":
                    # a MULTI-dimensional global `T g[A][B]` (declared nested, by its own source) indexed in
                    # full: row-major like a member array of the whole object at offset 0, so the access is
                    # a byte offset its declaration's shape does not matter to (CF-SMALL)
                    if len(dims) > 3:
                        raise CLowerError(
                            "indexing an array of more than 3 dimensions is not yet supported"
                        )
                    mem_shape, mem_elem, whole = tuple(dims), t, True
            idx_rids = [self._rvalue(ix) for ix in idx_nodes]
            member = (isinstance(n, cast.Member) and not ptr_member) or whole
            while True:
                shape = mem_shape if mem_shape is not None else base_ct.shape
                # a multi-dim VLA -> runtime dim multipliers
                vla_str = self.vla_strides.get(base_rid)
                # the subscripts this base takes: one per dimension of a declared multi-dimensional
                # array, a multi-dim VLA's too (Horner-flattened below), else one (a member array
                # took all of them above) -- the twin's `subscript_dims`
                if mem_shape is not None:
                    take = len(idx_rids)
                elif vla_str is not None:
                    take = len(vla_str)
                else:
                    take = max(1, len(shape))
                here, rest = idx_rids[:take], idx_rids[take:]
                if len(here) < take:
                    # fewer subscripts than the array has dimensions: a ROW, whose value would decay to a
                    # row pointer `T (*)[N]` the value model has no spelling for -- indexing its flat
                    # storage instead read an element (CF-DECAY). The twin's `index_chain` refuses alike.
                    raise CLowerError(
                        "partial indexing of a multi-dimensional array (a row used as a value) is not yet supported"
                    )
                lin = here[0]
                for d in range(1, len(here)):
                    if vla_str is not None:  # dim d's snapshot rid (NO c.const -- the runtime
                        k = vla_str[d]  # extent), so `m[i][j]` -> i*dim1 + j masks the total
                    else:
                        dim = shape[d] if d < len(shape) else 1
                        k = self._temp(scalar("uint32_t"), f"k{dim}")
                        self._emit("c.const", Opcode.LOAD, (), (k,), imm=(dim,))
                    m1 = self._temp(scalar("uint32_t"), "b_mul")
                    self._emit("c.bin.mul", Opcode.MUL, (lin, k), (m1,))
                    a1 = self._temp(scalar("uint32_t"), "b_add")
                    self._emit("c.bin.add", Opcode.ADD, (m1, here[d]), (a1,))
                    lin = a1
                if aos_idx is not None:  # `a[i].m[j]`: fold in the element's index
                    k = self._temp(scalar("uint32_t"), f"k{aos_k}")
                    self._emit("c.const", Opcode.LOAD, (), (k,), imm=(aos_k,))
                    m1 = self._temp(scalar("uint32_t"), "b_mul")
                    self._emit("c.bin.mul", Opcode.MUL, (aos_idx, k), (m1,))
                    a1 = self._temp(scalar("uint32_t"), "b_add")
                    self._emit("c.bin.add", Opcode.ADD, (m1, lin), (a1,))
                    lin, aos_idx = a1, None
                elem = (
                    mem_elem
                    if mem_elem is not None
                    else (base_ct.of if base_ct.of else scalar("uint32_t"))
                )
                lv = _LV("mem", base_rid, elem, idx=lin, byte_off=byte_off, member=member)
                if not rest:
                    return lv
                # more subscripts than the base has dimensions: the element is a pointer -- `q[j][i]`
                # on `T *q[N]`, `pp[j][i]` on `T **pp` -- so it is loaded, and the rest index what it
                # holds (the twin's `index_chain`). Flattening them into the base read `q[j + i]`.
                if elem.kind != "pointer":
                    raise CLowerError("a subscript of an element that is not a pointer")
                base_rid, base_ct, idx_rids = self._read(lv), elem, rest
                byte_off, mem_shape, mem_elem, member = 0, None, None, False
        if isinstance(node, cast.Member):
            # `arr[i].field` / `arr[i].m.k` -- a member of an ARRAY-OF-STRUCTS element (nested members at any
            # depth, CF-NESTMEM): keep the element's runtime index, stride by the element size, and land at the
            # member's flattened offset in the element
            aos = self._aos_member(node)
            if aos is not None:
                el, elem, ftype, foff, fbo, fbw, agg = aos
                if fbw or ftype.kind != "scalar":  # a bitfield / struct / array / pointer element
                    raise CLowerError(  # field is a follow-on -- both rails fall back
                        f"array-of-structs non-scalar element field ('.{node.field}') is not yet supported"
                    )
                return _LV(
                    "mem",
                    el.rid,
                    ftype,
                    idx=el.idx,
                    byte_off=el.byte_off + foff,
                    bit_off=fbo,
                    bit_width=fbw,
                    member=True,
                    stride=elem.size,
                    packed=agg.packed,
                )
            base_rid, base_ct, base_off = self._addr(node.base)
            agg = self._complete(base_ct.of if node.arrow else base_ct)
            ftype, byte_off, bit_off, bit_w = self._field(agg, node.field)
            ftype = qualified(ftype, agg.volatile)  # a member of a volatile aggregate
            off = (
                byte_off if node.arrow else base_off + byte_off
            )  # `->` resets to *base; `.` accumulates
            return _LV(
                "mem",
                base_rid,
                ftype,
                byte_off=off,
                bit_off=bit_off,
                bit_width=bit_w,
                packed=agg.packed,
            )
        if isinstance(node, cast.Unary) and node.op == "*":
            operand = node.operand
            if isinstance(operand, cast.Binary) and operand.op == "+":  # *(p + i) == p[i]
                return self._lvalue(cast.Index(operand.lhs, operand.rhs))
            # `*q[j]`: the element is a pointer, loaded, and dereferenced (the twin's general deref of
            # a pointer rvalue); `_addr` knows no subscripted base
            if isinstance(operand, cast.Index):
                el = self._lvalue(operand)
                if el.ct.kind != "pointer":
                    raise CLowerError("dereference of an element that is not a pointer")
                return _LV("mem", self._read(el), el.ct.of or scalar("uint32_t"), byte_off=0)
            # `*(volatile uint32_t *)ADDR`: the cast yields a real `T *` (the twin's general deref of
            # a pointer rvalue) -- unless it is a volatile access at a constant byte offset of a device
            # region, the member access the emit spelled so
            if isinstance(operand, cast.Cast):
                lv = self._byte_offset_access(operand)
                if lv is not None:
                    return lv
                rid = self._rvalue(operand)
                ct = self.rtypes.get(rid)
                if ct is not None and ct.kind == "pointer":
                    return _LV("mem", rid, ct.of or scalar("uint32_t"), byte_off=0)
                raise CLowerError("dereference of a non-pointer cast")
            # any other pointer VALUE -- `*&a`, `*(c ? &a : &b)`, `*p++`, `*(p - 1)` -- is dereferenced at
            # offset 0, as the cast above is and the twin's general deref of a pointer rvalue is (CF-SPLIT2:
            # `_addr` knows only the bases below, so these were refused while the twin lowered them)
            if not isinstance(
                operand,
                (cast.Name, cast.CompoundLiteral, cast.StringLit, cast.Member, cast.CallExpr),
            ) and not (isinstance(operand, cast.Unary) and operand.op == "*"):
                rid = self._rvalue(operand)
                ct = self.rtypes.get(rid)
                if ct is not None and ct.kind == "pointer":
                    return _LV("mem", rid, ct.of or scalar("uint32_t"), byte_off=0)
                raise CLowerError("dereference of a value that is not a pointer")
            base_rid, base_ct, base_off = self._addr(operand)
            return _LV("mem", base_rid, base_ct.of or scalar("uint32_t"), byte_off=base_off)
        raise CLowerError(f"not an lvalue: {type(node).__name__}")

    def _byte_offset_access(self, node: cast.Cast) -> "_LV | None":
        """`*(volatile T *)((char *)p + K)`: one volatile access of `T` at byte offset `K` from the device
        region `p` -- the claim the member access `p->m` lowers to, not a pointer computation and an access
        at offset 0 (CF-RTVOL). It is how the emit spells a volatile member, dereference or bitfield-unit
        access (`*(volatile T *)((const volatile char *)p + off)`, `emit.py`), so emitted C re-lowers each
        such access to the claim it was emitted from -- and ordinary driver code spells a register access
        by byte offset so, which the twin folds alike (`bcir_cfront.c`, `byte_off_access`). Only where that
        is exact, else None (the cast lowers as a pointer value). Types are read resolved, as both rails
        resolve them: the cast's is a pointer to a volatile, non-`_Atomic` integer or floating `T` (its
        width and signedness), the byte pointer's a pointer to plain `char`, whatever its qualifiers; `K` is
        an integer literal (an enumerator or a character constant is one) no wider than an `int`; and `p` is
        a declared pointer or array, or `&s` of a struct or union -- the two bases `_base_ptr` spells --
        whose region holds volatile storage, so the access stays a device access every refusal still sees
        (an inc/dec, a re-read as a value), as it was through the cast pointer."""
        add = node.operand
        if (
            not isinstance(add, cast.Binary)
            or add.op != "+"
            or not isinstance(add.lhs, cast.Cast)
            or not isinstance(add.rhs, cast.IntLit)
            or not 0 <= add.rhs.value <= 0x7FFFFFFF
        ):
            return None
        p = add.lhs.operand
        addr = isinstance(p, cast.Unary) and p.op == "&"  # `&s`: the struct's own storage
        name = p.operand if addr else p
        if not isinstance(name, cast.Name) or name.ident not in self.env:
            return None
        rid, pct = self.env[name.ident]
        if not (pct.is_aggregate if addr else pct.kind in ("pointer", "array")):
            return None
        if not self._mmio(rid):
            return None
        try:
            to, tb = self._resolve_type(node.type), self._resolve_type(add.lhs.type)
        except CLowerError:
            return None
        el = to.of if to.kind == "pointer" else None
        if el is None or not el.volatile or el.atomic or not (el.is_integer or el.is_float):
            return None
        by = tb.of if tb.kind == "pointer" else None
        if by is None or by.kind != "scalar" or by.name != "char" or by.atomic:
            return None
        return _LV("mem", rid, el, byte_off=add.rhs.value)

    def _string_ptr(self, spelling: str) -> int:
        """A string literal -> an anonymous read-only `char[]` global (NUL-terminated); the value is a
        pointer to it (decay). The emitter renders references to it as the inline literal, which is
        Clang-equivalent, so no synthesized global declaration is needed. A wide/UTF prefix
        (`L`/`u`/`U`/`u8`) sets the element width (`wchar_t`/`char16_t`/`char32_t`/`char`)."""
        existing = self.str_pool.get(spelling)  # dedup: identical literals share a global
        if existing is not None:
            return existing
        prefix, _ = split_lit_prefix(spelling)
        elem = str_elem_size(prefix, self.abi)  # 1 (char/u8) / 2 (u) / 4 (U) / wchar_t (L)
        nunits = _str_bytes(spelling) + 1  # the decoded code units + the NUL
        idx = self.strctr[0]
        self.strctr[0] += 1
        rid = _check_band_rid(970000 + idx)  # guard the reserved I/O-port rid (belt + suspenders)
        self.gres[rid] = Resource(
            rid=rid,
            domain=Domain.RAM,
            elem_bytes=elem,
            shape=(nunits,),
            access="ro",
            data_gen=1,
            name=f"__str{idx}",
        )
        self.rtypes[rid] = array(scalar("char" if elem == 1 else f"uint{elem * 8}_t"), nunits)
        self.str_globals[rid] = spelling  # rid -> the literal spelling (for emit)
        self.str_pool[spelling] = rid
        return rid

    def _require_declared(self, name: str, what: str, pos: int | None = None) -> None:
        """A function of the unit -- defined or prototyped -- named where no declaration of it precedes
        the name is refused: C99 dropped the implicit declaration (C11 6.5.1p2), and a call lowered as if
        prototyped typed its result by a declaration the call cannot see. The function's own body sees it.
        `what` names the use: a call, or a designator or `sizeof` operand (the twin's `undecl` check)."""
        declared = self.func.declared
        if declared is not None and name not in declared:
            raise CLowerError(f"{what} {name!r}", pos=pos)

    def _func_ptr_value(self, name: str) -> int:
        """A defined function used as a VALUE (function-to-pointer decay, `o->fn = g`): an anonymous funcptr
        'global' whose value is the function's address. The emitter renders the rid as the bare function name
        (C decays it to a pointer), so -- like a string literal -- no claim is emitted (parity-critical: the
        bare name must cost 0 claims on both rails)."""
        self._require_declared(name, "use of undeclared identifier")
        existing = self.func_pool.get(name)
        if existing is not None:
            return existing
        idx = self.strctr[0]  # share the literal-global rid space (unique rids)
        self.strctr[0] += 1
        rid = _check_band_rid(970000 + idx)  # guard the reserved I/O-port rid (belt + suspenders)
        self.gres[rid] = Resource(
            rid=rid,
            domain=Domain.RAM,
            elem_bytes=self.abi.pointer_size,
            shape=(1,),
            access="ro",
            data_gen=1,
            name=name,
        )
        self.rtypes[rid] = funcptr(name, self.func_rets[name], (), self.abi)
        self.func_globals[rid] = name  # rid -> the function name (rendered verbatim by emit)
        self.func_pool[name] = rid
        return rid

    def _fn_type(self, v: int) -> "CType | None":
        """The function-pointer type of the value `v` when its function type -- return and parameter types -- is
        known here: a designator (a function the unit defines, named as a value, C11 6.3.2.1p4) has its
        definition's; a function-pointer object -- a local, a parameter, a global, a select of them, a null
        pointer typed as one -- its declaration's. None otherwise: a pointer read from a member or an element
        (`fn_loaded`), a designator of a variadic function (a declarator here spells no `...`) or of one whose
        parameters only its own scope types while it has not lowered (CF-FNSEL; the twin's `res_sig`)."""
        ct = self.rtypes.get(v)
        if ct is None or ct.kind != "funcptr" or v in self.fn_loaded:
            return None
        name = self.func_globals.get(v)
        if name is None:
            return ct
        params = self.func_params.get(name)
        if params is not None and None in params and name in self.lowered:
            # a `typeof` or `va_list` parameter the pre-scan leaves untyped: the definition's own, once it has
            # lowered, as the twin types a designator of a function it has parsed
            params = tuple(p[2] for p in self.lowered[name].params)
        if params is None or None in params or name in self.func_variadic:
            return None
        return funcptr(name, self.func_rets[name], params, self.abi)

    def _fn_key(self, fct: CType) -> tuple:
        """A function type's identity: its return and its parameters as `_Generic` tells types apart,
        qualifiers aside (the twin's `sig_same`)."""
        ret = self._type_key(fct.of) if fct.of is not None else ("void",)
        return ret, tuple(self._type_key(p) for p in fct.params)

    def _addr(self, node):
        """The (rid, type, byte_offset) of an aggregate/pointer base used by member/index access. The
        offset ACCUMULATES through nested value-struct/union members (`t.q.a` -- q's byte offset must ride
        into a's access); `cfront_nestmember` only ever nested through a *first* member, so the drop went
        unseen. A `->` resets to the pointee (the access goes through the pointer base), and a
        pointer-valued field used as a base is loaded (offset reset to 0)."""
        if isinstance(node, cast.Name):
            rid, ct = self._lookup(node.ident, node.pos)
            return rid, ct, 0
        if isinstance(node, cast.CompoundLiteral):  # `(struct P){...}.field` / indexing the literal
            rid, ct = self._compound_literal(node)
            return rid, ct, 0
        if isinstance(node, cast.StringLit):  # "abc"[i] -> index the anonymous global
            rid = self._string_ptr(node.value)
            return rid, self.rtypes[rid], 0
        if isinstance(node, cast.Member):
            base_rid, base_ct, base_off = self._addr(node.base)
            agg = self._complete(base_ct.of if node.arrow else base_ct)
            ftype, byte_off, bit_off, bit_w = self._field(agg, node.field)
            ftype = qualified(ftype, agg.volatile)  # a member of a volatile aggregate
            off = (
                byte_off if node.arrow else base_off + byte_off
            )  # `->` resets to *base; `.` accumulates
            if ftype.kind == "pointer":  # a pointer-valued field used as a *base*
                # (`*(s->p)`, `s->t->a`): load the full pointer (at its field offset) and use the loaded
                # pointer as the new base -- the same move as `*q` below. Returning (struct_rid, T*) would
                # instead deref the struct's own address as if it held the pointee. (Pointer-value 2b.)
                lv = _LV(
                    "mem",
                    base_rid,
                    ftype,
                    byte_off=off,
                    bit_off=bit_off,
                    bit_width=bit_w,
                    packed=agg.packed,
                )
                return self._read(lv), ftype, 0
            return base_rid, ftype, off
        if (
            isinstance(node, cast.Unary) and node.op == "*"
        ):  # `*q` as a base (`**pp`): load the pointer it
            rid = self._read(self._lvalue(node))  # holds, and use that loaded pointer as the base
            return rid, self.rtypes.get(rid, pointer(scalar("uint32_t"))), 0
        if isinstance(node, cast.CallExpr):  # `mk(x).field` -- a struct-returning call's result is
            rid = self._call(node)  # a by-value struct temp; address it for member access
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
        self._scalar_operands(ta, tb)  # `a + 1`, `a += 5`, `a++` of a struct (CF-STRUCTARITH)
        if op in ("+", "-") and (a in self.vla_strides or b in self.vla_strides):
            # a multi-dimensional VLA decays to a row pointer (its rows are runtime extents)
            raise CLowerError("arithmetic on a multi-dimensional array is not yet supported")
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
        del a, b  # the constant case uses ta/tb's real literal type
        ba = ta is not None and ta.is_bitint
        if op in ("<<", ">>"):  # shift: the (non-promoting) `_BitInt` left operand;
            if ba:  # the right operand may be any integer (it is not
                return ta  # part of the result type)
            raise CLowerError(
                "a `_BitInt` shift count without a `_BitInt` left operand is not supported"
            )
        # Apply the C23 6.3.1.8 model to the operands' REAL types, keeping ONLY the cases whose result is a
        # `_BitInt(N)`. A bare integer constant carries its true literal type (`int`/`long`/...), so the
        # rank comparison matches Clang exactly (`bi64 + 5` -> `_BitInt(64)`, `bi8 + 5` -> fallback).
        try:
            r = usual_arith_int(ta, tb, self.abi)
        except BitIntMix as e:
            raise CLowerError(str(e)) from e
        if not r.is_bitint:  # defensive: the subset must yield a `_BitInt`
            raise CLowerError("`_BitInt` arithmetic whose C23 result is a standard integer type")
        return r

    def _bin_result_type_ct(self, op: str, ta: CType | None, tb: CType | None) -> CType:
        """The result type of a binary op from its operand *types*. `+ - * /` over a float operand yield
        the *wider* float (float usual arithmetic conversions); a relational/logical op is `int`; a shift
        takes the promoted *left* operand; everything else (`+ - * / %`, bitwise) takes the integer usual
        arithmetic conversions over both operands. Shared by `_rvalue` (emitting) and `_type_of` (typeof)."""
        self._scalar_operands(ta, tb)
        if op in ("+", "-", "*", "/"):
            floats = [t for t in (ta, tb) if t is not None and t.is_float]
            if floats:
                # complex usual arithmetic conversions: if ANY operand is complex the result is complex with
                # the wider ELEMENT float (`float _Complex + double` -> `double _Complex`); otherwise the
                # wider real float. Comparing element ranks makes this order-independent.
                if any(t.is_complex for t in floats):
                    er = max(_FLOAT_RANK.get(_elem_float_name(t), 1) for t in floats)
                    return scalar(f"{_RANK_FLOAT[er]} _Complex", self.abi)
                return scalar(max(floats, key=lambda t: _FLOAT_RANK.get(t.name, 1)).name, self.abi)
        if op in ("+", "-"):  # pointer arithmetic: p + i / i + p / p - i ->
            # an array operand is its first element's address
            ta, tb = self._arith_decay(ta), self._arith_decay(tb)
            pa = (
                ta if (ta is not None and ta.kind == "pointer") else None
            )  # a pointer carrying the pointee.
            pb = (
                tb if (tb is not None and tb.kind == "pointer") else None
            )  # (p - q, both pointers, is a
            if (pa is None) != (pb is None):  # ptrdiff and stays integer below.)
                return pa or pb
            if pa is not None and pb is not None and op == "-":
                # the difference of two pointers is a `ptrdiff_t` (C11 6.5.6p9): signed and pointer-wide
                # on every target, as `intptr_t` is, which is how the type model spells it (CF-PTRDIFF)
                return scalar("intptr_t", self.abi)
        if op in ("<", ">", "<=", ">=", "==", "!=", "&&", "||"):
            return scalar("int", self.abi)  # a relational / logical result is int
        ia = ta if (ta is not None and ta.is_integer) else scalar("int", self.abi)
        if op in ("<<", ">>"):
            return promote_int(ia, self.abi)  # the shift result carries the promoted LHS type
        ib = tb if (tb is not None and tb.is_integer) else scalar("int", self.abi)
        # `_BitInt(N)` does not canonicalize: a same-type pair stays itself; any other mix raises BitIntMix,
        # which is an unsupported form here (the rid-aware `_bin_result_type` already handled the legal
        # `_BitInt op constant` case) -> route the whole unit to fallback rather than mis-type the result.
        try:
            return usual_arith_int(ia, ib, self.abi)
        except BitIntMix as e:
            raise CLowerError(str(e)) from e

    def _arith_decay(self, t: CType | None) -> CType | None:
        """An array operand of `+` / `-` / `?:` converts to a pointer to its first element (C11 6.3.2.1p3):
        typing the result as the integer usual arithmetic conversions give a non-integer operand declared
        `int32_t t = la + 1` -- uncompilable C (CF-DECAY). A multi-dimensional array would decay to a row
        pointer the value model has no spelling for, and the emit declares it flat, so C would step by one
        element where the source steps by a row: refused, as the twin refuses it."""
        if t is not None and t.kind == "pointer" and len(t.shape) > 1:
            # a decayed `T m[][N]` / `T (*m)[N]` parameter: a row pointer, declared flat
            raise CLowerError("arithmetic on a multi-dimensional array is not yet supported")
        if t is None or t.kind != "array":
            return t
        if (t.of is not None and t.of.kind == "array") or len(t.shape) > 1:
            raise CLowerError("arithmetic on a multi-dimensional array is not yet supported")
        # the element's qualifiers ride on the pointee (`t.of`), as a pointer to volatile storage's do
        return pointer(t.of, self.abi) if t.of is not None else t

    def _type_of(self, node) -> CType:
        """The static C type of an expression, computed WITHOUT evaluating or emitting it -- the operand
        of `typeof` is unevaluated (like `sizeof`). Mirrors the result types `_rvalue` assigns its temps,
        so `typeof(expr)` resolves to exactly the type Clang gives the expression. Scoped to the forms
        whose type both rails infer identically (the twin reads it off a speculatively-lowered value);
        calls / address-of / ternary are a deferred follow-on (raised here rather than silently mistyped)."""
        if isinstance(node, cast.IntLit):
            return self._lit_type(node)
        if isinstance(node, cast.FloatLit):
            return _float_lit_type(node.value, self.abi)
        if isinstance(node, (cast.SizeOf, cast.AlignOf)):  # a `size_t` (C11 6.5.3.4p5)
            return scalar("size_t", self.abi)
        if isinstance(node, cast.Name):
            return self._lookup(node.ident, node.pos)[1]
        if isinstance(node, cast.StringLit):  # a string literal has type `char[N]` (not char*)
            prefix, _ = split_lit_prefix(node.value)
            elem = str_elem_size(prefix, self.abi)
            n = _str_bytes(node.value) + 1  # decoded code units + the NUL
            return array(scalar("char" if elem == 1 else f"uint{elem * 8}_t"), n)
        if isinstance(node, cast.Binary):
            return self._bin_result_type_ct(
                node.op, self._type_of(node.lhs), self._type_of(node.rhs)
            )
        if isinstance(node, cast.Unary):
            if node.op == "*":  # deref -> the pointee / element type
                t = self._type_of(node.operand)
                return t.of if t.of is not None else scalar("uint32_t")
            if node.op == "!":  # logical not -> int (0/1)
                return scalar("int", self.abi)
            t = self._type_of(node.operand)  # `-` / `~`: the integer-promoted operand
            if t.is_float:  # (a float stays float -- floats don't promote)
                return t
            if t.is_integer:
                return promote_int(t, self.abi)
            return scalar("uint32_t")
        if isinstance(node, (cast.Cast, cast.CompoundLiteral)):
            return self._resolve_type(node.type)
        if isinstance(node, cast.Member):  # `s.f` / `p->f` -> the field's type
            base_t = self._type_of(node.base)
            return self._field(base_t.of if node.arrow else base_t, node.field)[0]
        if isinstance(node, cast.Index):  # `a[i]` -> the element (pointee) type
            base_t = self._type_of(node.base)
            return base_t.of if base_t.of is not None else scalar("uint32_t")
        if isinstance(node, cast.Generic):  # _Generic -> the selected association's type
            return self._type_of(self._generic_select(node))
        if isinstance(node, cast.StmtExpr):  # `({ ...; e; })` -> the type of the last expr
            if node.stmts and isinstance(node.stmts[-1], cast.ExprStmt):
                return self._type_of(node.stmts[-1].expr)
            return scalar("void")
        raise CLowerError(
            f"typeof of this expression form ({type(node).__name__}) is not yet supported"
        )

    def _sizeof_elem(self, t: CType) -> CType:
        """One subscript or `*` of `t` as `sizeof` sees it (CF-SIZEOF): an array's element -- a whole row
        of a multi-dimensional one, flattened (`shape`) or nested alike -- or a pointer's pointee, which is
        a row again for a decayed `T m[][N]` / `T (*m)[N]` parameter (its `shape`)."""
        if t.kind == "array":
            return self._array_row(t)[0]
        if t.kind == "pointer" and t.of is not None:
            if len(t.shape) > 1:  # the pointee of a decayed multi-dim array parameter is a row
                return self._array_row(replace(array(t.of, 1), shape=t.shape))[0]
            return t.of
        raise CLowerError("sizeof of a subscript of a type that is neither an array nor a pointer")

    def _sizeof_decay(self, t: CType) -> CType:
        """The array-to-pointer conversion every operand but `sizeof`'s, `&`'s and a member access's
        undergoes (C11 6.3.2.1p3), for an operator whose result `_sizeof_type` types from its operands."""
        return pointer(t.of, self.abi) if t.kind == "array" else t

    def _sizeof_member(self, node) -> tuple[CType, int]:
        """A member access as `sizeof` sees it: the member's declared type and its bit-field width (0 for
        an ordinary member). `.` names a member of its operand, `->` of what its operand points at."""
        base_t = self._sizeof_type(node.base)
        agg = self._complete(self._sizeof_elem(base_t) if node.arrow else base_t)
        if not agg.is_aggregate:
            raise CLowerError(f"sizeof of a member of a {agg.kind}")
        ft, _off, _bit, width = self._field(agg, node.field)
        return ft, width

    def _sizeof_object(self, node) -> CType:
        """The declared type of an lvalue operand -- a bit-field's own declared type included, as an
        assignment, an increment and a comma operator yield it (Clang: `sizeof(b.f = 1)` of a `uint8_t f : 2`
        is 1)."""
        if isinstance(node, cast.Member):
            return self._sizeof_member(node)[0]
        return self._sizeof_type(node)

    def _sizeof_operand(self, node) -> CType:
        """The type an operator's operand converts to (C11 6.3.2.1p3, 6.3.1.1p2): an array decays to a
        pointer, and a bit-field takes the type Clang gives its value -- `int` when narrower than `int`, at
        `int`'s width `int` or `unsigned int` by its signedness, and wider its declared type, to which no
        promotion applies. The operator then applies its own conversions (`_bin_result_type_ct`)."""
        if isinstance(node, cast.Member):
            ft, width = self._sizeof_member(node)
            if not width:
                return self._sizeof_decay(ft)
            int_bits = scalar("int", self.abi).size * 8
            if width < int_bits:
                return scalar("int", self.abi)
            if width == int_bits:
                return scalar("int" if ft.signed else "unsigned int", self.abi)
            return unqualified(ft)
        return self._sizeof_decay(self._sizeof_type(node))

    def _sizeof_type(self, node) -> CType:
        """The type `sizeof` measures (CF-SIZEOF): the operand's own type, with neither the lvalue
        conversion nor the array-to-pointer conversion applied (C11 6.5.3.4p2, 6.3.2.1p2-3). An array
        stays its array, a row of a multi-dimensional array is a row, and a member array is its member. An
        operator's result is typed from its own operands once THEY convert (`_sizeof_operand`), as the
        lowering types it; `&`, a call, a conditional, an assignment, an increment, a comma and a nested
        `sizeof` / `_Alignof` are typed here. A bit-field operand is a constraint violation (C11 6.5.3.4p1)
        and a form this cannot type is refused: neither is folded to a guess."""
        if isinstance(node, cast.Name):
            if node.ident not in self.env:
                # not an object: a function designator is a constraint violation (C11 6.5.3.4p1), and the
                # constants `_rvalue` resolves after the environment keep the types it gives them
                if node.ident in self.func_rets or node.ident in self.protos:
                    raise CLowerError(f"sizeof of the function {node.ident!r}", pos=node.pos)
                if node.ident in _IMAG_UNIT:
                    return scalar("float _Complex", self.abi)
                if node.ident in self._MEMORDER:
                    return scalar("int", self.abi)
            rid, ct = self._lookup(node.ident, node.pos)
            if rid in self.vla_strides:
                # a multi-dimensional VLA: every row of it is a runtime extent, so a subscript of it has
                # no constant size (its unknown dimensions make it incomplete here)
                return replace(ct, shape=(0,) * len(self.vla_strides[rid]))
            return ct
        if isinstance(node, cast.Index):
            return self._sizeof_elem(self._sizeof_type(node.base))
        if isinstance(node, cast.Member):
            ft, width = self._sizeof_member(node)
            if width:
                raise CLowerError("sizeof of a bit-field")
            return ft
        if isinstance(node, cast.Unary):
            if node.op == "*":
                return self._sizeof_elem(self._sizeof_type(node.operand))
            if node.op == "&":
                o = node.operand
                if (
                    isinstance(o, cast.Name)
                    and o.ident not in self.env
                    and (o.ident in self.func_rets or o.ident in self.protos)
                ):  # `&f`: a pointer to the function, as wide as any pointer here
                    self._require_declared(o.ident, "use of undeclared identifier", o.pos)
                    return funcptr(o.ident, self.func_rets.get(o.ident), (), self.abi)
                return pointer(self._sizeof_type(o), self.abi)
            if node.op == "!":
                return scalar("int", self.abi)
            t = self._sizeof_operand(node.operand)
            if t.is_float:  # `-` / `~` / `+`: the promoted operand (a float does not promote)
                return t
            if t.is_integer:
                return promote_int(t, self.abi)
            raise CLowerError(f"sizeof of `{node.op}` applied to a {t.kind}")
        if isinstance(node, cast.Binary):
            if node.op == ",":
                # the comma operator yields its right operand's value, unpromoted (C11 6.5.17p2)
                return self._sizeof_decay(unqualified(self._sizeof_object(node.rhs)))
            lhs = self._sizeof_operand(node.lhs)
            return self._bin_result_type_ct(node.op, lhs, self._sizeof_operand(node.rhs))
        if isinstance(node, cast.Ternary):
            a, b = self._sizeof_operand(node.then), self._sizeof_operand(node.els)
            if a.kind == "pointer" or b.kind == "pointer":
                return a if a.kind == "pointer" else b
            if a.is_aggregate:
                return a
            # the usual arithmetic conversions (C11 6.5.15p5)
            return self._bin_result_type_ct("+", a, b)
        if isinstance(node, cast.Assign):
            # an assignment has its left operand's type (C11 6.5.16p3)
            return unqualified(self._sizeof_object(node.target))
        if isinstance(node, cast.IncDec):
            return unqualified(self._sizeof_object(node.operand))
        if isinstance(node, cast.CallExpr):
            if node.callee not in self.env and (
                node.callee in self.func_rets or node.callee in self.protos
            ):
                self._require_declared(node.callee, "call to undeclared function")
            ret = self.func_rets.get(node.callee)
            if ret is None and node.callee in self.env:  # a call through a function-pointer object
                fpt = self.env[node.callee][1]
                ret = fpt.of if fpt.kind == "funcptr" else None
            elif ret is None and node.callee in self.protos:
                ret = self.protos[node.callee][0]
            if ret is None:
                raise CLowerError(
                    f"sizeof of a call to {node.callee!r}: its return type is unknown"
                )
            return ret
        if isinstance(node, (cast.SizeOf, cast.AlignOf)):
            return scalar("size_t", self.abi)
        if isinstance(node, cast.Generic):
            return self._sizeof_type(self._generic_select(node))
        return self._type_of(node)

    def _type_key(self, t: CType):
        """A canonical, comparable identity for a type used by `_Generic` matching (C11 §6.5.1.1, after
        lvalue/array decay; qualifiers do not affect the key). int / int32_t collapse (same width + sign);
        plain `char` stays distinct from signed/unsigned char; floats key on width; aggregates on tag."""
        if t.kind == "array":
            t = (
                pointer(t.of, self.abi)
                if t.of is not None
                else pointer(scalar("uint32_t"), self.abi)
            )
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
        if t.name == "char":  # plain char -- a distinct type from signed/unsigned char
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

    def _lit_type(self, node: "cast.IntLit") -> CType:
        """An integer constant's type (§6.4.4.1) at this target. The parser picks the candidate for the LP64
        model, whose `long` holds 64 bits; where it holds 32 (LLP64, ILP32) a value past it is a `long long`
        -- unsigned for an unsigned candidate -- the next type C's list gives. The twin's `lit_int_type`."""
        ct = scalar(node.ctype, self.abi) if is_scalar_name(node.ctype) else scalar("int", self.abi)
        if ct.name in ("long", "unsigned long") and not _kfits(node.value, ct.size * 8, ct.signed):
            return scalar("long long" if ct.signed else "unsigned long long", self.abi)
        return ct

    def _rvalue(self, node) -> int:
        if isinstance(node, cast.IntLit):
            ct = self._lit_type(node)
            t = self._temp(ct, f"k{node.value}")
            r = self._emit("c.const", Opcode.LOAD, (), (t,), imm=(node.value,))
            return r
        if isinstance(node, cast.FloatLit):  # a float constant -> a typed c.fconst
            # f/F -> float, l/L -> long double, else double -- by the target's ABI
            ct = _float_lit_type(node.value, self.abi)
            t = self._temp(ct, "fk")
            return self._emit(f"c.fconst:{node.value}", Opcode.LOAD, (), (t,))
        if isinstance(node, cast.Name):
            if node.ident not in self.env:
                if node.ident in self.func_rets:
                    return self._func_ptr_value(
                        node.ident
                    )  # a function NAME as a value -> its funcptr
                if node.ident in _IMAG_UNIT:  # <complex.h> imaginary unit (unless shadowed)
                    t = self._temp(scalar("float _Complex"), "imag_unit")
                    return self._emit(f"c.cconst:{node.ident}", Opcode.LOAD, (), (t,))
                if node.ident in self._MEMORDER:  # SEG6.1: a `memory_order_*` / `__ATOMIC_*`
                    val = self._MEMORDER[node.ident]  # constant (unless shadowed by a real decl) ->
                    t = self._temp(
                        scalar("int", self.abi), f"k{val}"
                    )  # an int const, like an IntLit
                    return self._emit("c.const", Opcode.LOAD, (), (t,), imm=(val,))
            rid, ct = self._lookup(node.ident, node.pos)
            # a read of a volatile object is a device access, its value an ordinary one (an array
            # decays; a struct copies whole)
            volatile_read = ct.volatile and ct.kind == "scalar"
            if volatile_read:
                t = self._temp(unqualified(ct), "vld")
                return self._emit(
                    "c.copy",
                    Opcode.ADD,
                    (rid,),
                    (t,),
                    domain=Domain.MMIO,
                    lane=Lane.H,
                    hazard="barriered",
                    volatile=True,
                )
            return rid
        if isinstance(node, cast.StringLit):  # a string value -> the global pointer
            return self._string_ptr(node.value)
        if isinstance(node, cast.Binary) and node.op == ",":
            self._rvalue(node.lhs)  # the comma operator: evaluate the left operand for
            return self._rvalue(node.rhs)  # its side effects, discard it, yield the right value
        if isinstance(node, cast.IncDec):
            return self._incdec_value(node)
        if isinstance(node, cast.LabelAddr):  # `&&L` -- a label's address as a `void *` value (GNU)
            t = self._temp(pointer(scalar("void")), f"labeladdr_{node.label}")
            return self._emit(f"c.labeladdr:{node.label}", Opcode.LOAD, (), (t,))
        if isinstance(node, cast.Binary) and node.op in ("&&", "||"):
            # The right operand is evaluated only when the left does not decide the result (C11 6.5.13p4,
            # 6.5.14p4). One that can neither trap nor change state is computed eagerly (`c.bin.land` /
            # `c.bin.lor` over both values); any other lowers as a branch (CF-TERNARY: `p && *p` read
            # through NULL, `n && m / n` divided by zero). The arm that evaluates it computes the same op --
            # there the left operand is known, so it is the right one's truth -- and the arm the left
            # operand decides stores its constant.
            a = self._rvalue(node.lhs)
            blk_b, b = self._lower_apart(node.rhs)
            opcode, suf = _BIN[node.op]
            rt = self._bin_result_type(node.op, a, b)
            if _operand_pure(blk_b, self._declared_rids()):
                self.block_stack[-1].extend(blk_b)
                t = self._temp(rt, f"b_{suf}")
                return self._emit(f"c.bin.{suf}", opcode, (a, b), (t,))
            decided: list = []
            if node.op == "&&":
                sel, node_if = self._branch_value(a, rt, blk_b, decided)
            else:
                sel, node_if = self._branch_value(a, rt, decided, blk_b)
            k = self._temp(rt, "c")
            self.block_stack.append(decided)
            try:
                self._emit("c.const", Opcode.LOAD, (), (k,), imm=(0 if node.op == "&&" else 1,))
            finally:
                self.block_stack.pop()
            self._assign_in(decided, k, sel)
            t = self._temp(rt, f"b_{suf}")
            self.block_stack.append(blk_b)
            try:
                self._emit(f"c.bin.{suf}", opcode, (a, b), (t,))
            finally:
                self.block_stack.pop()
            self._assign_in(blk_b, t, sel)
            self.block_stack[-1].append(node_if)
            return sel
        if isinstance(node, cast.Binary):
            a, b = self._rvalue(node.lhs), self._rvalue(node.rhs)
            if node.op in ("==", "!="):
                # a null pointer constant compared with a pointer converts to that pointer (C11 6.5.9p5): its
                # temp is the pointer, where an `int` temp compared with it was a constraint violation (6.5.9p2)
                # Clang and GCC only warn about. Only the temp's C type moves (CF-NULLCALL).
                ta, tb = self.rtypes.get(a), self.rtypes.get(b)
                if ta is not None:
                    self._null_pointer(b, ta)
                if tb is not None:
                    self._null_pointer(a, tb)
            opcode, suf = _BIN[node.op]
            # float arithmetic propagates the (wider) float type; comparisons/bitwise stay int. The
            # actual IEEE-754 math is delegated to the emitted C / resident backend (never computed here).
            rt = self._bin_result_type(node.op, a, b)
            t = self._temp(rt, f"b_{suf}")
            return self._emit(f"c.bin.{suf}", opcode, (a, b), (t,))
        if isinstance(node, cast.Unary):
            if node.op in ("__real__", "__imag__"):  # GNU complex part -> the real element float
                v = self._rvalue(node.operand)
                vt = self.rtypes.get(v)
                if vt is not None and vt.is_complex:
                    rt = scalar(
                        {4: "float", 8: "double", 16: "long double"}.get(vt.size // 2, "double")
                    )
                elif vt is not None and vt.is_float:  # __real__ of a real float is the value itself
                    rt = vt
                else:
                    rt = scalar("double")
                suf = "creal" if node.op == "__real__" else "cimag"
                t = self._temp(rt, f"u_{suf}")  # emitted `__real__ x` -- not integer-computed
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
                if (
                    isinstance(operand, cast.Unary) and operand.op == "*"
                ):  # &*p == p (the pointer itself;
                    return self._rvalue(operand.operand)  # &*(p+i) == p+i) -- 0 claims, both rails
                lv = self._lvalue(operand)
                if lv.bit_width:  # &bitfield is illegal in C (no addressable unit)
                    raise CLowerError("cannot take the address of a bit-field")
                if lv.kind == "mem" and lv.idx is None and lv.ct.kind == "array":
                    raise CLowerError(  # &s.arr -- the result is a pointer-TO-ARRAY
                        "address-of an array member is not yet supported"
                    )  # (needs (*)[] declarators; follow-on)
                t = self._temp(pointer(lv.ct), "addr")
                if lv.kind == "var":  # &name / &(compound literal) -> the object's address
                    return self._emit("c.addrof", Opcode.ADD, (lv.rid,), (t,))
                if lv.idx is not None:  # &arr[i] / &s.arr[i] / &arr[i].field
                    stride = lv.stride or (
                        lv.ct.size or 4
                    )  # array-of-structs strides by the STRUCT, else the elem
                    return self._emit(
                        "c.addrof", Opcode.ADD, (lv.rid, lv.idx), (t,), imm=(lv.byte_off, stride)
                    )
                return self._emit(
                    "c.addrof", Opcode.ADD, (lv.rid,), (t,), imm=(lv.byte_off,)
                )  # &base.member
            v = self._rvalue(node.operand)
            # `-a`, `~a`, `!a` of a struct (CF-STRUCTARITH)
            self._scalar_operands(self.rtypes.get(v))
            opcode, suf = _UN[node.op]
            if node.op == "!":  # logical not -> int (0/1)
                rt = scalar("int", self.abi)
            else:  # `-` / `~`: the promoted operand type, so
                vt = self.rtypes.get(v)  # negating a `long` stays 64-bit, and `-x` on
                if vt is not None and vt.is_float:  # a float stays float (floats don't promote --
                    rt = vt  # was forced to uint32, truncating -2.5 -> 4.29e9)
                elif vt is not None and vt.is_integer:  # integer promotion: a sub-int operand (e.g.
                    rt = promote_int(
                        vt, self.abi
                    )  # `unsigned char`) becomes signed int -> ~c is -1
                else:
                    rt = scalar("uint32_t")
            t = self._temp(rt, f"u_{suf}")
            return self._emit(f"c.un.{suf}", opcode, (v,), (t,))
        if isinstance(node, cast.Cast):
            v = self._rvalue(node.operand)
            ct = self._resolve_type(node.type)
            if ct.kind == "scalar" and ct.name == "void":  # `(void)e`: e for its effects, no value
                return _VOID_RID  # (C11 6.3.2.2) -- a cast of it to uint32 did not compile
            self._scalar_value(ct, v)  # `(uint32_t)a` of a struct (C11 6.5.4p2, CF-STRUCTARITH)
            return self._cast_value(v, ct)
        if isinstance(node, (cast.Index, cast.Member)):
            lv = self._lvalue(node)
            if lv.ct.kind == "array" and not lv.bit_width:
                return self._array_value(node, lv)
            v = self._read(lv)
            # a function pointer read from a member or an element: `?:` gives it no function type
            if lv.ct.kind == "funcptr":
                self.fn_loaded.add(v)
            return v
        if isinstance(
            node, cast.CompoundLiteral
        ):  # `(struct P){a,b}` by value / `(int){v}` as a value
            return self._compound_literal(node)[0]
        if isinstance(node, cast.VaArg):  # va_arg(ap, T) -> the next variadic argument (type T)
            ap = self._rvalue(node.ap)  # the va_list cursor (by name)
            vt = self._resolve_type(node.type)  # the result has type T (drives the temp + emit)
            t = self._temp(vt, "vaarg")
            return self._emit("c.call.vaarg", Opcode.GEM_DISPATCH, (ap,), (t,))
        if isinstance(node, cast.Generic):  # _Generic(ctrl, T: e, ..., default: e) -- select on
            return self._rvalue(
                self._generic_select(node)
            )  # ctrl's static type; only the chosen e is lowered
        if isinstance(node, cast.StmtExpr):  # `({ s1; ...; e; })` -> lower the prefix stmts inline
            if not node.stmts:  # (its own scope), then yield the last expr's value
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
                if isinstance(last.expr, cast.Member):
                    val = self._read(self._lvalue(last.expr), declared_bitfield_type=True)
                else:
                    val = self._rvalue(last.expr)
            else:  # a non-expression last stmt -> void (rarely used)
                self._stmt(last)
                val = _VOID_RID
            self.env = saved_env
            return val
        if isinstance(node, cast.Assign):
            return self._assign(node)
        if isinstance(node, cast.SizeOf):
            # sizeof folds to a compile-time constant -- the operand is NOT evaluated. Its value is the
            # size of the operand's own type (`_sizeof_type`: no array decay), and its type is `size_t`
            # (C11 6.5.3.4p5), the ABI's pointer-sized unsigned integer, so `sizeof x * n` and `sizeof a -
            # sizeof b` compute in that width (CF-SIZEOF).
            if isinstance(node.expr, cast.Name) and node.expr.ident in self.env:
                rid, ct = self._lookup(node.expr.ident, node.expr.pos)
                if ct.kind == "array" and ct.count == 0 and rid in self.ptr_extent:
                    # `sizeof a` of a 1-D stack VLA is a RUNTIME value: the snapshot extent × sizeof(element).
                    # (Opcode.ADD: a cost hint only — the emit string carries the multiply, like c.copy/c.vladecl.)
                    ext = self.ptr_extent[rid]
                    t = self._temp(scalar("size_t", self.abi), "szof")
                    return self._emit("c.sizeof.vla", Opcode.ADD, (ext,), (t,), imm=(ct.of.size,))
            ct = (
                self._resolve_type(node.type)
                if node.type is not None
                else self._sizeof_type(node.expr)
            )
            if ct.size <= 0 or (ct.kind == "array" and ct.count == 0):
                # an unsized array, `void`, a call to a void function (C11 6.5.3.4p1), or a row of a
                # multi-dimensional VLA, whose size is a runtime value
                raise CLowerError("sizeof of an incomplete type")
            if ct.size > (1 << (8 * self.abi.pointer_size - 1)) - 1:
                # no object of it fits the target's address space (Clang: "array is too large")
                raise CLowerError("sizeof of a type too large for the target")
            t = self._temp(scalar("size_t", self.abi), "szof")
            return self._emit("c.const", Opcode.LOAD, (), (t,), imm=(ct.size,))
        if isinstance(node, cast.AlignOf):
            # _Alignof folds to the target type's alignment (operand never evaluated, like sizeof); its
            # type is `size_t` too (C11 6.5.3.4p5).
            t = self._temp(scalar("size_t", self.abi), "alof")
            return self._emit(
                "c.const", Opcode.LOAD, (), (t,), imm=(self._resolve_type(node.type).align,)
            )
        if isinstance(node, cast.Ternary):
            # C evaluates the condition, then exactly one arm (C11 6.5.15p4). Arms that can neither trap nor
            # change state are computed eagerly and chosen by a select -- the emitter renders the real C
            # `(cond ? a : b)`; any other pair lowers as a branch that evaluates only the arm C evaluates
            # (CF-TERNARY: `d ? n / d : 0` divided by zero and `p ? *p : s` read through NULL eagerly).
            c = self._rvalue(node.cond)
            blk_a, a = self._lower_apart(node.then)
            blk_b, b = self._lower_apart(node.els)
            if a == _VOID_RID or b == _VOID_RID:
                # a void conditional (C11 6.5.15p3: both arms void -- `c ? f() : g();`, an `assert`'s
                # `c ? (void)0 : fail()`) runs the arm C evaluates for its effects alone: no local, no value
                if a != b:
                    raise CLowerError("one arm of `?:` is void and the other is not")
                declared = self._declared_rids()
                if _operand_pure(blk_a, declared) and _operand_pure(blk_b, declared):
                    self.block_stack[-1].extend(blk_a)
                    self.block_stack[-1].extend(blk_b)
                else:
                    self.block_stack[-1].append(IfNode(c, blk_a, blk_b))
                return _VOID_RID
            # a select over ARITHMETIC arms carries their common type (usual arithmetic conversions), NOT a
            # blanket uint32_t -- otherwise a signed arm loses its sign (a downstream `>>`/compare on the
            # select goes unsigned) and a FLOAT arm is truncated to int (`(c?x:y)` of doubles becomes an
            # int, dropping the value / mis-converting a nan). (pointer arms keep the 4-byte unit.)
            ta, tb = self.rtypes.get(a), self.rtypes.get(b)
            if a in self.vla_strides or b in self.vla_strides:
                raise CLowerError("a multi-dimensional array operand of `?:` is not yet supported")
            pa, pb = self._arith_decay(ta), self._arith_decay(tb)
            if (
                ta is not None
                and tb is not None
                and (ta.is_integer or ta.is_float)
                and (tb.is_integer or tb.is_float)
            ):
                rt = self._bin_result_type_ct("+", ta, tb)
            elif pa is not None and pa.kind == "pointer":
                rt = pa  # a pointer (or a decayed array) arm: the select is that pointer (CF-DECAY)
            elif pb is not None and pb.kind == "pointer":
                rt = pb
            elif (ta is not None and ta.is_aggregate) or (tb is not None and tb.is_aggregate):
                # a struct or union arm: the select is that aggregate, copied whole (CF-STRUCTVAL) -- a
                # uint32 select of it did not compile. Its arms are of one struct or union type (C11
                # 6.5.15p3); any other pairing is refused, as the twin refuses it (`p_cond`)
                if not (
                    ta is not None and tb is not None and (ta.kind, ta.name) == (tb.kind, tb.name)
                ):
                    raise CLowerError(
                        "the arms of `?:` are a struct or union and a value of another type"
                    )
                rt = unqualified(ta)
            else:
                # a function designator decays to a pointer to its function (C11 6.3.2.1p4): an arm that is one,
                # or a function-pointer object, makes the select that function pointer (6.5.15p6) -- a uint32
                # select of it did not compile -- and its arms point to one function type (6.5.15p3) (CF-FNSEL)
                fa, fb = self._fn_type(a), self._fn_type(b)
                if fa is not None and fb is not None and self._fn_key(fa) != self._fn_key(fb):
                    raise CLowerError(_FN_SELECTED)
                rt = fa if fa is not None else fb if fb is not None else scalar("uint32_t")
            declared = self._declared_rids()
            if _operand_pure(blk_a, declared) and _operand_pure(blk_b, declared):
                self.block_stack[-1].extend(blk_a)
                self.block_stack[-1].extend(blk_b)
                t = self._temp(rt, "sel")
                # a null pointer constant arm is the other arm's pointer (6.5.15p6), as in a branch below
                self._null_pointer(a, rt)
                self._null_pointer(b, rt)
                return self._emit("c.select", Opcode.ADD, (c, a, b), (t,))
            sel, node_if = self._branch_value(c, rt, blk_a, blk_b)
            self._assign_in(blk_a, self._null_pointer(a, rt), sel)
            self._assign_in(blk_b, self._null_pointer(b, rt), sel)
            self.block_stack[-1].append(node_if)
            return sel
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
        base_rid, base_ct, _base_off = self._addr(m.base)  # a dispatch base is a pointer (offset 0)
        actuals = tuple(self._rvalue(a) for a in node.args)
        agg = self._complete(
            base_ct.of if m.arrow else base_ct
        )  # the struct with the funcptr field
        try:  # the member's funcptr CType -> its return type
            fct = self._field(agg, m.field)[0] if agg is not None else None
        except CLowerError:
            fct = None
        # a null pointer argument is its parameter's pointer, as in a direct call (CF-NULLCALL)
        if fct is not None and fct.kind == "funcptr":
            self._null_pointer_args(actuals, fct.params)
        if _returns_void(fct):  # a void member function: a claim with no result, the void value
            self._emit(
                f"c.call.imember:{m.field}",
                Opcode.GEM_DISPATCH,
                (base_rid, *actuals),
                (),
                imm=(1 if m.arrow else 0,),
                callee_sig=callee_signature(fct),
            )
            return _VOID_RID
        ret_ct = fct.of if (fct is not None and fct.kind == "funcptr") else None
        t = self._temp(
            self._call_result_ct(ret_ct), f"icall_{m.field}"
        )  # a signed member return reads back signed
        return self._emit(
            f"c.call.imember:{m.field}",
            Opcode.GEM_DISPATCH,
            (base_rid, *actuals),
            (t,),
            imm=(1 if m.arrow else 0,),
            callee_sig=callee_signature(fct),
        )

    # --- memory read/write, with bitfield (mask/shift) + MMIO (ordered) handling ---
    def _read(self, lv: "_LV", *, declared_bitfield_type: bool = False) -> int:
        if lv.kind == "var":
            return lv.rid
        unit = self._load_unit(lv)
        if lv.bit_width:  # bitfield extract: (unit >> off) & mask
            signed = (
                lv.ct.is_integer and lv.ct.signed
            )  # a signed bitfield read sign-extends from bit w-1
            # integer promotion (6.3.1.1): a bitfield narrower than int promotes to int (int holds all its
            # values), so an UNSIGNED sub-int bitfield reads as a SIGNED int -- `bf < x` is a signed compare,
            # not an unsigned one. A full-width (== 32) unsigned bitfield stays unsigned; a WIDE bitfield
            # (> 32 bits, in a long-long unit) keeps its declared 64-bit type -- int/unsigned can't hold it.
            # Clang's GNU statement-expression rule is the deliberate exception:
            # a bare terminal bitfield retains its declared type after ceasing to be
            # a bitfield expression. Thus `-({ s.u; })` is unsigned, unlike `-(s.u)`.
            rt = (
                lv.ct
                if declared_bitfield_type or lv.bit_width > 32
                else scalar("int" if (signed or lv.bit_width < 32) else "uint32_t", self.abi)
            )
            t = self._temp(rt, "bf")
            return self._emit(
                "c.bf.get", Opcode.ADD, (unit,), (t,), imm=(lv.bit_off, lv.bit_width, int(signed))
            )
        return unit

    def _array_value(self, node, lv: "_LV") -> int:
        """An array member used as a value (`s.a`, `p->a`, `g.a`): it decays to a pointer to its first
        element (C11 6.3.2.1p3) -- its ADDRESS, `&s.a[0]`, never a load of that element, which is what the
        member read had emitted (an uncompilable `return s.a;`, and a silent element value under a cast to an
        integer). The one claim `&s.m` emits (`c.addrof` of the base at the member's offset), typed `T *`.
        A row -- of a multi-dimensional array, or an array element that is itself an array -- would decay to
        a `T (*)[N]`, which the value model has no spelling for: refused, as the twin refuses it (CF-DECAY)."""
        elem = lv.ct.of if lv.ct.of is not None else scalar("uint32_t")
        if not isinstance(node, cast.Member) or lv.idx is not None:
            raise CLowerError(
                "an array element that is itself an array, used as a value, is not yet supported"
            )
        if elem.kind == "array" or len(lv.ct.shape) > 1:
            raise CLowerError(
                "a multi-dimensional member array used as a value is not yet supported"
            )
        t = self._temp(pointer(elem, self.abi), "decay")
        return self._emit("c.addrof", Opcode.ADD, (lv.rid,), (t,), imm=(lv.byte_off,))

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
        if (
            lv.kind == "mem"
            and lv.idx is not None
            and not lv.member
            and not self._mmio(lv.rid)
            and lv.rid not in self.str_globals
        ):
            rt = self.rtypes.get(lv.rid)
            if rt is not None and rt.kind == "array" and rt.count:
                return "masked"  # a known-extent local/static array -> runtime-bounds-checked
            if lv.rid in self.ptr_extent:
                return "masked"  # a malloc/calloc pointer with a recovered element count
        return "assumed_safe"

    def _bounds_provenance(self, lv: "_LV") -> str:
        """WHY the access's bounds contract is what it is -- the §5.14 Phase 2 extent-provenance
        signal, carried first-class on the claim so R7 (and the debugger/ML layer) can see checked
        vs trusted WITH the reason. Mirrors `_access_bounds` case-for-case: the masked promotions
        report their extent source; the trusted cases report why no extent was recoverable."""
        if self._mmio(lv.rid):
            return "mmio_register"
        if lv.rid in self.str_globals:
            return "string_literal"
        if lv.kind == "mem" and lv.idx is not None and not lv.member:
            rt = self.rtypes.get(lv.rid)
            if rt is not None and rt.kind == "array" and rt.count:
                return "declared_extent"
            if lv.rid in self.ptr_extent:
                return self.ptr_extent_kind.get(lv.rid, "recovered_count")
            return "unknown_extent"
        return ""  # a scalar / member / non-indexed access: no extent story

    def _alloc_count_node(self, call, pointee_size: int):
        """The element-COUNT AST node of a recoverable `malloc(N*sizeof(T))` / `malloc(sizeof(T)*N)` /
        `calloc(N, sizeof(T))` / `malloc(N)` (1-byte pointee) -- N, where the size operand is `sizeof(T)`
        (T the pointee) or its byte literal. Else None. Recognition only; the binding decides how to capture
        N (a stable Name re-emitted by name, or an expression SNAPSHOTTED). (§5.12)"""
        if not isinstance(call, cast.CallExpr) or pointee_size <= 0:
            return None

        def size_bytes(e):  # a per-element size operand -> its byte width
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
            if pointee_size == 1:  # a byte buffer: malloc(N) bytes == N elements
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
        if isinstance(count, cast.Name) and count.ident in self.env:  # the stable-Name fast path
            rid, ct = self.env[count.ident]
            if (
                ct.kind == "scalar"
                and not getattr(ct, "is_float", False)
                and self._mut_body.get(count.ident, 0) == 0
                and count.ident not in self._mut_addr
            ):
                self.ptr_extent[p_rid] = rid
                self.ptr_extent_kind[p_rid] = "recovered_count"
            return  # a Name is captured by name or not at all
        if _is_pure(count):  # an EXPRESSION count -> snapshot it
            v = self._rvalue(count)
            vct = self.rtypes.get(v)
            if vct is not None and vct.kind == "scalar" and not getattr(vct, "is_float", False):
                ext = self._storage(vct, f"__bcir_ext{self._ext_ctr}")
                self._ext_ctr += 1
                self._emit("c.copy", Opcode.ADD, (v,), (ext,))
                self.ptr_extent[p_rid] = ext
                self.ptr_extent_kind[p_rid] = "snapshot_extent"

    def _load_unit(self, lv: "_LV") -> int:
        if lv.bit_width:
            # a bitfield's storage unit is read as an unsigned wide enough to hold it (a `long long` field, or
            # a packed field straddling into bits >= 32, needs 64 bits, not a uint32); only the `unit_bytes()`
            # spanned bytes are read (a PACKED field's unit may not fill the whole declared type / may straddle
            # words), into a zeroed temp so the unread high bytes are 0.
            ub = lv.unit_bytes()
            t = self._temp(scalar("uint32_t" if ub <= 4 else "uint64_t", self.abi), "ld")
            rd = (lv.rid,) if lv.idx is None else (lv.rid, lv.idx)
            vol = lv.ct.volatile
            mmio = vol or self._mmio(lv.rid)
            return self._emit(
                "c.load",
                Opcode.LOAD,
                rd,
                (t,),
                imm=(lv.byte_off, ub),
                domain=Domain.MMIO if mmio else Domain.RAM,
                bounds=self._access_bounds(lv),
                bounds_provenance=self._bounds_provenance(lv),
                lane=Lane.H if mmio else Lane.U,
                hazard="barriered" if mmio else "unique",
                volatile=vol,
            )
        unit_ct = unqualified(lv.ct)  # the VALUE read from a volatile lvalue is an ordinary value
        t = self._temp(unit_ct, "ld")
        rd = (lv.rid,) if lv.idx is None else (lv.rid, lv.idx)
        # a volatile ACCESS is one whose lvalue is volatile (the pointee of a pointer to volatile, a
        # volatile member or a member of a volatile aggregate, an element of a volatile array); an access
        # through a device region whose own lvalue is not volatile (a plain member beside a volatile one,
        # the pointer element of a `volatile T **`) is a device-domain claim, not a volatile access
        vol = lv.ct.volatile
        mmio = vol or self._mmio(lv.rid)
        # the byte offset rides in imm (not the strict-bounds `offset`), and the access is
        # assumed_safe (the frontend resolved the member/index). MMIO accesses are ordered, and a read
        # of an `_Atomic` object is an atomic load (the emit performs it through an `_Atomic` lvalue,
        # never a byte copy). A member
        # array `s.arr[i]` carries BOTH (member offset, element size) and an index -- so the emit lands
        # the element at `&s + member_off + i*elem_size`, distinct from a plain `base[idx]` (no imm).
        if lv.idx is not None:
            imm = (
                (lv.byte_off, max(1, lv.ct.size))
                + ((lv.stride,) if lv.stride else ())  # member array: even
                if lv.member
                else ()
            )  # off 0; array-of-structs adds the element stride (imm[2])
        else:
            imm = (lv.byte_off,) if lv.byte_off else ()
        return self._emit(
            "c.load",
            Opcode.LOAD,
            rd,
            (t,),
            imm=imm,
            domain=Domain.MMIO if mmio else Domain.RAM,
            bounds=self._access_bounds(lv),
            bounds_provenance=self._bounds_provenance(lv),
            volatile=vol,
            **_access_order(mmio, lv.ct.atomic),
        )

    def _cast_value(self, v: int, ct: CType) -> int:
        """`(ct)v`: one `c.cast` claim into a temp of the target type -- the explicit cast, and C's
        assignment conversion at a store (`_store_conversion`). The twin's `emit_cast`."""
        # a float cast target types the temp float (so downstream arithmetic is float, and the emit
        # declares it float); an integer cast to a uint32 temp reproduces integer-promotion (a
        # narrowing cast masks/zero-extends back), so either way the result matches Clang.
        # a pointer cast yields a pointer of the target type (its pointee, and whether that pointee
        # is volatile, ride on the temp): a uint32 temp truncated the address
        typed = ct.is_integer or ct.is_float or ct.kind == "pointer"
        t = self._temp(ct if typed else scalar("uint32_t"), "cast")
        # A float -> signed-integer conversion needs a SIGNED cast operator: the canonical unsigned
        # name (uint32_t / uint8_t) makes it float -> unsigned, which is UB for a negative value and
        # diverges by target (x86 wraps, aarch64 saturates to 0) -- and even on x86 a sub-int signed
        # target loses the sign. Emit the signed fixed-width operator so `(int)(-5.0f)` is -5.
        vt = self.rtypes.get(v)
        cname = (
            _CAST_W_SIGNED.get(ct.size, "int32_t")
            if (vt is not None and vt.is_float and ct.is_integer and ct.signed and not ct.is_bitint)
            else _cast_name(ct)
        )  # a `_BitInt` target keeps its exact spelling (no width-named fallback)
        return self._emit(f"c.cast:{cname}", Opcode.ADD, (v,), (t,))

    def _store_conversion(self, ct: CType, v: int) -> int:
        """C's assignment conversion (C23 6.5.17.2) at a store the emit spells as a byte copy -- a
        member, a member-array element, a field of an array of structs, a bitfield, a store through a
        pointer: the value converts to the slot's declared type `ct`. The emit picks the stored bytes'
        type from the VALUE, so a value of another arithmetic class -- an integer into a float slot, a
        float into an integer one, a real into a complex one -- converts first, through the `c.cast` an
        explicit `(T)v` lowers to (the twin's `store_conv`), and the conversion is in the claim graph. A
        width or sign change within a class is the emit's (`_store_conv`); a `_Bool` slot normalizes by
        its flag; a pointer, aggregate or function-pointer slot takes no arithmetic conversion."""
        vt = self.rtypes.get(v)
        if ct.kind != "scalar" or ct.name in ("_Bool", "bool") or vt is None or vt.kind != "scalar":
            return v
        if _arith_class(vt) == _arith_class(ct):
            return v
        return self._cast_value(v, unqualified(ct))  # the value is never volatile (6.3.2.1p2)

    def _struct_value(self, ct: CType, v: int, why: str) -> None:
        """Refuse `v` for an object of the struct or union type `ct` unless it is a value of that type --
        the same kind and tag, qualifiers aside (C11 6.7.9p13, 6.5.16.1p1). `why` is the refusal: an
        initializer's or an assignment's (CF-STRUCTINIT). An object of any other type takes no struct
        (`_scalar_value`)."""
        if not ct.is_aggregate:
            self._scalar_value(ct, v)
            return
        vt = self.rtypes.get(v)
        if vt is None or (vt.kind, vt.name) != (ct.kind, ct.name):
            raise CLowerError(why)

    def _scalar_value(self, ct: CType, v: int) -> None:
        """Refuse a struct or union value `v` for an object of the type `ct`, a scalar or a pointer (C11
        6.5.16.1p1, 6.5.4p2): an initializer, an assignment, a `return`, an argument, a cast (CF-STRUCTARITH;
        the twin's `scalar_value_ok`)."""
        vt = self.rtypes.get(v)
        if not ct.is_aggregate and vt is not None and vt.is_aggregate:
            raise CLowerError(_STRUCT_CONVERTED)

    def _scalar_operands(self, *types) -> None:
        """Refuse a struct or union operand of an operator that takes a scalar (CF-STRUCTARITH; the twin's
        `agg_operand`)."""
        if any(t is not None and t.is_aggregate for t in types):
            raise CLowerError(_STRUCT_OPERAND)

    def _write(self, lv: "_LV", v: int) -> int:
        """Store `v` through `lv`; returns the value stored -- after C's conversion, before a
        bitfield's insertion into its unit."""
        # a store the emit spells as a byte copy takes C's conversion first; a typed `base[idx] = v`
        # converts in the emitted C itself -- where a null pointer constant takes the element's type
        if lv.idx is None or lv.member or lv.stride:
            v = self._store_conversion(lv.ct, v)
        else:
            v = self._null_pointer(v, lv.ct)
        stored = v
        if lv.bit_width:  # read-modify-write the storage unit
            old = self._load_unit(lv)
            t = self._temp(
                scalar("uint32_t" if lv.unit_bytes() <= 4 else "uint64_t", self.abi), "bf"
            )
            v = self._emit("c.bf.set", Opcode.ADD, (old, v), (t,), imm=(lv.bit_off, lv.bit_width))
        vol = lv.ct.volatile
        mmio = vol or self._mmio(lv.rid)
        if lv.idx is None:  # member/deref: carry (offset, size) -- a packed
            rd, imm = (
                (lv.rid, v),
                (lv.byte_off, lv.unit_bytes()),
            )  # bitfield writes only its spanned bytes back
        elif lv.member:  # s.arr[i]: (member offset, element size)
            rd, imm = (lv.rid, lv.idx, v), (lv.byte_off, max(1, lv.ct.size))
        else:  # base[idx]: a typed array store
            rd, imm = (lv.rid, lv.idx, v), ()
        if imm and lv.ct.name in ("_Bool", "bool"):  # a _Bool slot: flag it so the emit normalizes
            imm = imm + (1,)  # the stored value (any nonzero -> 1, §6.3.1.2)
        if lv.stride:  # array-of-structs element store: land at
            if len(imm) == 2:  # &base + off + idx*stride but copy ct.size bytes --
                imm = imm + (0,)  # keep (off,size,bool_flag,stride): pad the flag slot
            imm = imm + (lv.stride,)  # (0 == not a _Bool field) so stride is always imm[3]
        self._emit(
            "c.store",
            Opcode.STORE,
            rd,
            (),
            imm=imm,
            domain=Domain.MMIO if mmio else Domain.RAM,
            bounds=self._access_bounds(lv),
            bounds_provenance=self._bounds_provenance(lv),
            volatile=vol,
            **_access_order(mmio, lv.ct.atomic),  # a store to an `_Atomic` object: an atomic store
        )
        return stored

    def _atomic_rmw(self, lv: "_LV", kind: str, v: int | None = None) -> int:
        """One atomic read-modify-write of the `_Atomic` object `lv` designates (CF-ATOMIC): a compound
        assignment `E op= v` (C11 6.5.16.2p3) or an increment or decrement (6.5.2.4p2), whose load
        and store must not come apart -- lowered as a load, an operation and a store, a concurrent
        write between them was lost. `kind` is the operator's suffix (`add`, `shl`, ...) or
        `preinc`/`predec`/`postinc`/`postdec`. One `c.c11atom.rmw:<kind>` claim, addressed as the
        store addresses the object (its reads `(base[, idx][, v])`, its imm the store's), a named
        object as a base at offset 0; its value is the new value (the old for a postfix operator), of
        the object's unqualified type. The operand keeps its own type, so the emitted `E op= v`
        converts as C does (`*p *= 0.5` on an `_Atomic int` multiplies in double). The twin's
        `emit_rmw`."""
        rt = unqualified(lv.ct)
        t = self._temp(rt, f"rmw_{kind}")
        if lv.kind == "var":  # a named object: a memory base at offset 0
            rd, imm = (lv.rid,), (0, max(1, lv.ct.size))
        elif lv.idx is None:  # a member / a dereference
            rd, imm = (lv.rid,), (lv.byte_off, lv.unit_bytes())
        elif lv.member:  # s.arr[i], aos[i].f: (member offset, element size), the store's layout
            rd, imm = (lv.rid, lv.idx), (lv.byte_off, max(1, lv.ct.size))
            if lv.stride:
                imm = imm + (0, lv.stride)
        else:  # base[idx]: a typed element (its bounds guard kept)
            rd, imm = (lv.rid, lv.idx), ()
        if v is not None:
            rd = rd + (v,)
        mmio = lv.ct.volatile or (lv.kind == "mem" and self._mmio(lv.rid))
        return self._emit(
            f"c.c11atom.rmw:{kind}",
            _RMW_OPCODE.get(kind, Opcode.CMPXCHG),
            rd,
            (t,),
            imm=imm,
            domain=Domain.MMIO if mmio else Domain.RAM,
            bounds=self._access_bounds(lv),
            bounds_provenance=self._bounds_provenance(lv),
            volatile=lv.ct.volatile,
            **_access_order(mmio, True),
        )

    def _compound_literal(self, node: "cast.CompoundLiteral") -> tuple[int, CType]:
        """Materialize a compound literal `(type){init}` as an anonymous local and initialize it, exactly
        like a braced local decl -- a struct/union/array reuses the `_agg_init` zero-baseline + per-member
        store path; a scalar `(int){v}` copies the single value in. Returns (rid, type) so the result acts
        as an lvalue (address-of / member access) and as an rvalue (a by-value struct arg / scalar read)."""
        dims = node.type.array or ()
        inferred = False  # `(T[]){...}` / `(T[][B]){...}`: the initializer sizes it
        if len(dims) > 1 or (len(dims) == 1 and dims[0] in (0, None)):
            ct, inferred = self._inferred_array(node.type)
        else:
            ct = self._resolve_type(node.type)
        self.cl_ctr += 1
        rid = self._storage(ct, f"_cl{self.cl_ctr}")
        if ct.kind in ("struct", "union", "array"):
            n = self._agg_init(rid, ct, node.init, inferred=inferred)
            if inferred:
                ct = self._sized_array(ct, n)
                self._resize_local(rid, ct)
        else:  # a scalar compound literal `(int){v}`: its one value (`{}` is zero)
            expr = self._braced_scalar(node.init)
            v = self._rvalue(expr) if expr is not None else self._zero_int("clz")
            self._emit("c.copy", Opcode.ADD, (v,), (rid,))
        return rid, ct

    def _array_row(self, ct: CType) -> "tuple[CType, int]":
        """For a FLAT multi-dim array object `T a[d0][d1]...` (the `array(leaf, total)` + `shape=(d0,d1,..)`
        representation), the *element* of the outermost dim is a ROW sub-array `T row[d1]...`, NOT the scalar
        leaf `ct.of`. Synthesize that row sub-type (carrying `shape[1:]`) and its byte stride (product of the
        inner dims * leaf size) so a nested brace `{...}` descends by row. A 1-D array (or no `shape`) keeps
        the scalar element + leaf-sized stride."""
        leaf = ct.of or scalar("uint32_t")
        if len(ct.shape) <= 1:  # 1-D: element IS the scalar leaf
            return leaf, leaf.size
        inner = ct.shape[1:]
        stride = leaf.size
        for d in inner:
            stride *= d
        row = replace(
            array(leaf, stride // leaf.size), shape=tuple(inner)
        )  # `T row[d1*..]`, carrying shape[1:]
        return row, stride

    # --- C11 6.7.9 initialization: the current-object walk (CF-BRACE) -----------------------------
    #
    # A braced initializer lowers to the object's zero baseline, emitted `= {}` (every subobject no initializer
    # names is zero, §6.7.10), plus one store per initialized scalar, in list order. The walk is C's own:
    # each brace list has a current object whose subobjects -- array elements, struct members in order,
    # a union's first member -- the positional entries fill in turn, descending into a sub-aggregate
    # whose initializer has no brace of its own (brace elision, 6.7.9p20: it takes only as many entries
    # as it has scalars), and a designator re-points the walk, the entries after it continuing from the
    # subobject after the designated one (p17). The twin walks the same way (`init_list`).

    def _agg_init(self, rid: int, ct: CType, ag: cast.AggInit, inferred: bool = False) -> int:
        """Lower the braced initializer `ag` of the local object `rid` of type `ct` (a declared local or
        a compound literal); returns the number of top-level elements it reached, which sizes an
        inferred `T a[] = {...}` (`inferred`: the array's count is the walk's to find)."""
        self.zero_init.add(rid)
        w = _InitWalk(rid=rid, top_array=ct.kind == "array")
        self._init_list(w, ct, 0, ag, top=True, flat=ct.kind == "array", inferred=inferred)
        return w.top_n

    def _init_count(self, ct: CType) -> int:
        """The number of subobjects of aggregate `ct` an initializer walks: an array's elements (a flat
        multi-dimensional array's rows), a struct or union's fields."""
        if ct.kind == "array":
            return ct.shape[0] if len(ct.shape) > 1 else ct.count
        return len(ct.fields)

    def _init_unit(self, fr: "_IFrame") -> tuple:
        """The subobject at the frame's position -- (type, byte offset, bit offset, bit width, the
        declaring aggregate's `packed`) -- and the position after it. An anonymous struct or union
        member is one subobject over its promoted leaves (`CType.anon`); a union holds one."""
        ct, i = fr.ct, fr.idx
        if ct.kind == "array":
            sub, stride = self._array_row(ct)
            return (sub, fr.off + i * stride, 0, 0, False), i + 1
        for first, n, act, aoff in ct.anon:
            if first == i:
                return (act, fr.off + aoff, 0, 0, False), (
                    len(ct.fields) if ct.kind == "union" else first + n
                )
        _fn, fty, fbo, fbit, fbw = ct.fields[i]
        return (fty, fr.off + fbo, fbit, fbw, ct.packed), (
            len(ct.fields) if ct.kind == "union" else i + 1
        )

    def _init_take(self, w: "_InitWalk", fr: "_IFrame") -> tuple:
        """Consume the subobject at the frame's position. A union takes one member per object: an
        initializer that would give an initialized union another member is refused, since its other
        bytes would have to be re-zeroed (the plain stores cannot express that)."""
        unit, nxt = self._init_unit(fr)
        if fr.ct.kind == "union" and w.unions.setdefault((fr.off, fr.ct.name), fr.idx) != fr.idx:
            raise CLowerError("an initializer overrides a prior initialization of a subobject")
        fr.idx = nxt
        if fr.top:
            w.top_n = max(w.top_n, nxt)
        return unit

    def _init_frame(self, parent: "_IFrame", unit: tuple) -> "_IFrame":
        """The frame that walks the sub-aggregate `unit` of `parent` (brace elision or a designator)."""
        return _IFrame(
            unit[0],
            unit[1],
            count=self._init_count(unit[0]),
            flat=parent.flat and unit[0].kind == "array",
        )

    def _init_push(self, frames: list, fr: "_IFrame", unit: tuple) -> "_IFrame":
        """Enter the sub-aggregate `unit` of `fr` (brace elision or a designator step): one more frame
        of the list's walk, bounded as the twin bounds it -- a deeper nesting is refused, never cut."""
        if len(frames) >= _INIT_MAX_FRAMES:
            raise CLowerError(f"an initializer nested deeper than {_INIT_MAX_FRAMES} subobjects")
        frames.append(self._init_frame(fr, unit))
        return frames[-1]

    def _init_next(self, frames: list) -> "_IFrame":
        """The frame whose next subobject the next positional entry fills: the filled sub-aggregates
        entered by elision are left for their parent; past the list's own object is an excess
        initializer, a constraint violation (C11 6.7.9p2) that is refused, never dropped."""
        while True:
            fr = frames[-1]
            if fr.count is None or fr.idx < fr.count:
                return fr
            if len(frames) == 1:
                raise CLowerError("excess elements in an initializer")
            frames.pop()

    def _init_designate(self, w: "_InitWalk", frames: list, key) -> "_IFrame":
        """Point the walk at the subobject a designator names (6.7.9p17-18): back to the list's own
        object, then each step selects a member or an element, descending into it before the next
        step. A member of an anonymous struct or union is reached through that member, so the entries
        after it continue inside it, as C continues them."""
        if isinstance(key, tuple):
            steps = key
        else:
            steps = (("a", key),) if isinstance(key, int) else (("m", key),)
        del frames[1:]
        for k, (kind, val) in enumerate(steps):
            fr = frames[-1]
            if kind == "m":
                if fr.ct.kind not in ("struct", "union"):
                    raise CLowerError("a member designator into a non-aggregate")
                j = next((n for n, f in enumerate(fr.ct.fields) if f[0] == val), None)
                if j is None:
                    raise CLowerError(f"no member named {val!r} to designate")
                while True:
                    grp = next((g for g in fr.ct.anon if g[0] <= j < g[0] + g[1]), None)
                    if grp is None:
                        fr.idx = j
                        break
                    fr.idx = grp[0]  # through the anonymous member that holds it
                    fr, j = self._init_push(frames, fr, self._init_take(w, fr)), j - grp[0]
            else:
                if fr.ct.kind != "array":
                    raise CLowerError("an array designator into a non-array")
                if val < 0 or val > 0x7FFFFFFF or (fr.count is not None and val >= fr.count):
                    raise CLowerError("an array designator outside the array")  # (an int index)
                fr.idx = val
            if k < len(steps) - 1:  # the next step names a subobject of this one
                unit = self._init_take(w, fr)
                if unit[0].kind not in ("struct", "union", "array"):
                    raise CLowerError("a designator into a scalar")
                self._init_push(frames, fr, unit)
        return frames[-1]

    def _init_overrides(self, w: "_InitWalk", lo: int, hi: int) -> None:
        """A subobject initialized as a whole by a brace list or a string literal is zero wherever its
        initializer is silent -- which the baseline gives only while nothing has stored into it yet. An
        earlier store in the range would survive, so that override is refused."""
        if any(a < hi and lo < b for a, b in w.stores):
            raise CLowerError("an initializer overrides a prior initialization of a subobject")

    def _init_store(self, w: "_InitWalk", unit: tuple, v, indexed: bool) -> None:
        """Store `v` into the scalar (or whole struct/union) subobject `unit`: a typed `base[i]` store
        at its flat index when it is a whole element of the declared array its own list initializes
        (`indexed`), else a store at its byte offset (a bit-field's through its storage unit). In a
        static's image (`w.const`) `v` is a folded constant, recorded as the value the subobject holds."""
        ct, off, bo, bw, packed = unit
        lo = off * 8 + bo
        w.stores.append((lo, lo + (bw or ct.size * 8)))
        if w.shape:  # a file-scope initializer's shape: the range, and no value
            return
        if w.const:
            w.image.append((lo, bw or ct.size * 8, _kleaf(v, ct, bw)))
            return
        self._scalar_value(ct, v)  # a scalar subobject takes no struct (CF-STRUCTARITH)
        if indexed:
            ti = self._temp(scalar("int", self.abi), "ai")
            self._emit("c.const", Opcode.LOAD, (), (ti,), imm=(off // ct.size,))
            self._write(_LV("mem", w.rid, ct, idx=ti), v)
            return
        self._write(_LV("mem", w.rid, ct, byte_off=off, bit_off=bo, bit_width=bw, packed=packed), v)

    def _init_char_array(self, ct: CType, lit: cast.StringLit) -> bool:
        """`ct` is an array the string literal `lit` initializes (6.7.9p14-15): one-dimensional, of an
        integer element as wide as the literal's code unit -- a character type, so neither `_Bool` nor
        a `_BitInt(N)` (the twin's `init_char_array`)."""
        if ct.kind != "array" or len(ct.shape) > 1 or ct.of is None:
            return False
        el = ct.of
        if el.name == "_Bool" or el.is_bitint:
            return False
        prefix, _ = split_lit_prefix(lit.value)
        return el.is_integer and el.size == str_elem_size(prefix, self.abi)

    def _init_string(
        self, w: "_InitWalk", unit: tuple, lit: cast.StringLit, indexed: bool, count: int | None
    ) -> None:
        """Initialize the character array `unit` (of `count` elements; None: the declared array its
        literal sizes) from a string literal: one store per code unit, the terminating NUL and the rest
        of the array left to the zero baseline. A literal longer than the array is refused -- it fits
        with its NUL dropped only when exactly as long (p14)."""
        ct, off = unit[0], unit[1]
        try:
            _prefix, units = str_units(lit.value, lambda p: str_elem_size(p, self.abi))
        except ValueError as e:
            raise CLowerError(str(e)) from None
        if count is None:
            count = len(units) + 1
            w.top_n = max(w.top_n, count)
        if len(units) > count:
            raise CLowerError("an initializer-string for a character array is too long")
        w.strings.add(id(lit))
        es = ct.of.size
        self._init_overrides(w, off * 8, (off + es * count) * 8)
        for i, u in enumerate(units):
            if w.const:  # a static's image: the code unit, an `int` constant
                cv = _KVal(u)
            else:
                cv = self._temp(scalar("int", self.abi), "sc")
                self._emit("c.const", Opcode.LOAD, (), (cv,), imm=(u,))
            self._init_store(w, (ct.of, off + i * es, 0, 0, False), cv, indexed)

    def _init_list(
        self,
        w: "_InitWalk",
        ct: CType,
        off: int,
        ag: cast.AggInit,
        *,
        top: bool,
        flat: bool,
        inferred: bool = False,
    ) -> None:
        """One brace-enclosed list, whose current object is the `ct` at byte offset `off`. `top`: the
        declared object's own list (its whole elements store indexed); `flat`: `ct` is the declared
        array or a row of it."""
        own = _IFrame(ct, off, count=None if inferred else self._init_count(ct), flat=flat, top=top)
        ents = ag.entries
        if (  # `{"abc"}` for a character array: the literal initializes the whole array
            len(ents) == 1
            and ents[0][0] is None
            and isinstance(ents[0][1], cast.StringLit)
            and self._init_char_array(ct, ents[0][1])
        ):
            self._init_string(w, (ct, off, 0, 0, False), ents[0][1], top and flat, own.count)
            return
        frames = [own]
        for key, expr in ents:
            indexed_ok = top and w.top_array and (key is None or isinstance(key, int))
            fr = self._init_next(frames) if key is None else self._init_designate(w, frames, key)
            if isinstance(expr, cast.AggInit):  # a nested list initializes the subobject as a whole
                self._init_sublist(w, fr, self._init_take(w, fr), expr)
                continue
            if isinstance(expr, cast.StringLit):  # the character array it fills, found by elision
                while True:
                    sub = self._init_unit(fr)[0][0]
                    if self._init_char_array(sub, expr):
                        unit = self._init_take(w, fr)
                        self._init_string(w, unit, expr, indexed_ok and fr.flat, sub.count)
                        break
                    if sub.kind not in ("struct", "union", "array"):
                        self._init_store(
                            w,
                            self._init_take(w, fr),
                            self._init_value(w, expr),
                            indexed_ok and fr.flat,
                        )
                        break
                    self._init_push(frames, fr, self._init_take(w, fr))
                    fr = self._init_next(frames)
                continue
            v = self._init_value(w, expr)  # evaluated once, before it is placed
            vt = None if w.const else self.rtypes.get(v)  # a folded constant is never a struct
            while True:
                sub = self._init_unit(fr)[0][0]
                if sub.kind in ("struct", "union") and (
                    vt is not None and vt.kind == sub.kind and vt.name == sub.name
                ):  # a struct/union value initializes the subobject whole (6.7.9p13)
                    self._init_store(w, self._init_take(w, fr), v, indexed_ok and fr.flat)
                    break
                if sub.kind not in ("struct", "union", "array"):
                    self._init_store(w, self._init_take(w, fr), v, indexed_ok and fr.flat)
                    break
                self._init_push(frames, fr, self._init_take(w, fr))  # brace elision
                fr = self._init_next(frames)

    def _zero_int(self, name: str) -> int:
        """A constant zero `int` value (the value of an empty initializer `{}` for a scalar)."""
        v = self._temp(scalar("int", self.abi), name)
        self._emit("c.const", Opcode.LOAD, (), (v,), imm=(0,))
        return v

    def _braced_scalar(self, ag: cast.AggInit):
        """The one expression of a braced scalar initializer `{ e }` (C11 6.7.9p11); `{}` has none."""
        if not ag.entries:
            return None
        (key, expr), *rest = ag.entries
        if rest or key is not None or isinstance(expr, cast.AggInit):
            raise CLowerError("a braced scalar initializer holds one expression")
        return expr

    def _resize_local(self, rid: int, ct: CType) -> None:
        """Give the local `rid` the type its initializer sized (an inferred `T a[] = {...}`): its
        resource, its declaration and its name's binding."""
        name = self.resources[rid].name
        self._resource(rid, ct, name)
        self.locals = [(r, n, ct if r == rid else c) for r, n, c in self.locals]
        if self.env.get(name, (None,))[0] == rid:
            self.env[name] = (rid, ct)

    def _inferred_array(self, tref: cast.TypeRef, flat: bool = True) -> "tuple[CType, bool]":
        """A local array's type before its initializer is walked: `T a[]` / `T m[][B]..` is provisional
        -- one element or row, which the walk then counts (`_sized_array`)."""
        dims = tref.array
        if len(dims) > 3:
            raise CLowerError(
                "a multi-dimensional local array of more than 3 dims is not yet supported"
            )
        elem = self._resolve_type(replace(tref, array=()))
        inferred = dims[0] in (0, None)
        dims = ((1,) if inferred else (dims[0],)) + tuple(dims[1:])
        total = 1
        for d in dims:
            total *= d
        if len(dims) == 1:
            return array(elem, total), inferred
        # a MULTI-dim array: a flat resource carrying its per-dim shape, so `m[i][j]` flattens row-major
        # and the emit declares `m[A*B]` (the memory layout of `m[A][B]`)
        return replace(array(elem, total), shape=dims), inferred

    def _sized_array(self, ct: CType, n: int) -> CType:
        """The inferred array `ct` sized to the `n` top-level elements (rows) its initializer reached."""
        n = max(n, 1)
        if len(ct.shape) > 1:
            inner = ct.count // ct.shape[0]
            return replace(array(ct.of, n * inner), shape=(n,) + ct.shape[1:])
        return array(ct.of, n)

    def _init_sublist(self, w: "_InitWalk", fr: "_IFrame", unit: tuple, ag: cast.AggInit) -> None:
        """A nested brace list for the subobject `unit`: it initializes the whole subobject, so no
        earlier store may lie in it. A scalar takes a braced single expression (6.7.9p11)."""
        ct, off, bo, bw, _packed = unit
        lo = off * 8 + bo
        self._init_overrides(w, lo, lo + (bw or ct.size * 8))
        if ct.kind not in ("struct", "union", "array"):
            expr = self._braced_scalar(ag)
            if expr is not None:  # `{}` is zero, which the baseline already holds
                self._init_store(w, unit, self._init_value(w, expr), False)
            return
        self._init_list(w, ct, off, ag, top=False, flat=fr.flat and ct.kind == "array")

    def _init_value(self, w: "_InitWalk", expr):
        """An initializer entry's value, evaluated once, before it is placed: lowered -- or, in a static's
        image (`w.const`), folded (`_const_value`); a file-scope initializer's shape (`w.shape`) takes none."""
        if w.shape:
            return None
        return self._const_value(expr) if w.const else self._rvalue(expr)

    # --- a static's constant image (CF-STATICTAB) ---

    def _static_init(self, ct: CType, init, inferred: bool) -> "tuple[CType, str | None]":
        """A static's initializer: the initializer walk in constant mode -- each entry folded, each store
        recorded in the static's image, no claim left behind (C initializes a static once, before the
        program runs, never at a call) -- and the image rendered as the declaration's initializer. Returns
        the static's type (an inferred `[]` sized by the walk) and its rendered initializer, None when the
        image is zero. A scalar takes one value (`{e}` braced, `{}` zero); an aggregate or an array takes a
        brace list, a character array a string literal. The twin's `init_image`."""
        if init is None:
            return ct, None
        if isinstance(init, cast.StringLit) and ct.kind == "array":
            if not self._init_char_array(ct, init):  # only a character array takes one
                raise CLowerError("an array is initialized by a brace list or a string literal")
            init = cast.AggInit(entries=((None, init),))  # `char s[] = "ab"` is `{"ab"}`
        if ct.kind == "array" and not isinstance(init, cast.AggInit):
            raise CLowerError("an array is initialized by a brace list or a string literal")
        w = _InitWalk(rid=-1, top_array=ct.kind == "array", const=True)
        with self._scratch():  # the entries' claims: folded, then discarded
            unit = (ct, 0, 0, 0, False)
            if isinstance(init, cast.AggInit) and ct.kind in ("struct", "union", "array"):
                self._init_list(
                    w, ct, 0, init, top=True, flat=ct.kind == "array", inferred=inferred
                )
            elif isinstance(init, cast.AggInit):
                self._init_sublist(w, _IFrame(ct, 0), unit, init)
            else:  # an expression: a scalar's value (a struct's is never a constant)
                self._init_store(w, unit, self._init_value(w, init), False)
        if inferred:
            ct = self._sized_array(ct, w.top_n)
        return ct, self._render_image(ct, w)

    def _file_scope_shape(self, ct: CType, init) -> "tuple[CType, frozenset]":
        """A file-scope initializer walked as C walks the current object (CF-GBRACE), for its shape only: an
        entry is neither lowered nor folded -- the source defines the global, and the emit names it -- but the
        walk's constraints hold as for a local or a static (C11 6.7.9p2, p14, p17-19: an excess entry, an
        override, a string too long, a designator outside its object are refused), and an unsized array takes
        the extent the walk reaches: `struct pt g[] = {1u, 2u, 3u, 4u}` is two elements, not four. Returns the
        type, sized, and the ids of the string literals that initialize a character array in it. A scalar's or a
        struct's expression has nothing to walk. The twin's `global_init_shape`."""
        dims, el = 0, ct
        while el.kind == "array":
            dims, el = dims + 1, el.of
        if isinstance(init, cast.StringLit) and ct.kind == "array":
            if not self._init_char_array(ct, init):  # only a character array takes one
                raise CLowerError("an array is initialized by a brace list or a string literal")
            init = cast.AggInit(entries=((None, init),))  # `char s[] = "ab"` is `{"ab"}`
        if not isinstance(init, cast.AggInit):
            if ct.kind == "array":
                raise CLowerError("an array is initialized by a brace list or a string literal")
            return ct, frozenset()
        if dims > 3:  # the twin's walk holds three dimensions, as a local's
            raise CLowerError(
                "an initialized file-scope array of more than 3 dimensions is not supported"
            )
        inferred = ct.kind == "array" and ct.count == 0
        w = _InitWalk(rid=-1, top_array=ct.kind == "array", const=True, shape=True)
        with self._scratch():
            if ct.kind in ("struct", "union", "array"):
                self._init_list(
                    w, ct, 0, init, top=True, flat=ct.kind == "array", inferred=inferred
                )
            else:  # a braced scalar `T g = {e};` (6.7.9p11)
                self._init_sublist(w, _IFrame(ct, 0), (ct, 0, 0, 0, False), init)
        return (array(ct.of, max(w.top_n, 1)) if inferred else ct), frozenset(w.strings)

    @contextmanager
    def _scratch(self):
        """Lower only to fold: the claims go to a scratch block, and the lowerer is wound back after
        (`_snapshot`) -- no claim, temp, literal or claim id is left behind. A static's initializer and a
        file-scope initializer are lowered in one."""
        snap = self._snapshot()
        self.block_stack.append([])
        try:
            yield
        finally:
            self._restore(snap)

    def _snapshot(self) -> dict:
        """The lowerer's whole state, each container copied: a static's initializer lowers its entries only
        to fold them, and `_restore` then leaves no trace of them -- no claim, temp, literal or claim id."""
        return {
            k: copy.copy(v) if isinstance(v, (list, dict, set)) else v
            for k, v in vars(self).items()
        }

    def _restore(self, snap: dict) -> None:
        """Wind the lowerer back to `snap`, each container in place -- so the unit's shared ones (the claim
        ids, the literal pool) keep their identity."""
        for k in [k for k in vars(self) if k not in snap]:
            delattr(self, k)
        for k, v in snap.items():
            cur = getattr(self, k)
            if isinstance(v, list) and isinstance(cur, list):
                cur[:] = v
            elif isinstance(v, (dict, set)) and isinstance(cur, type(v)):
                cur.clear()
                cur.update(v)
            else:
                setattr(self, k, v)

    def _const_value(self, expr) -> _KVal:
        """One entry of a static's initializer, as C evaluates an integer constant expression: lowered as
        any expression is, into the evaluation's scratch block, and the claims it made folded in order -- a
        constant (a literal, a `sizeof`, an enumerator) in its declared type, arithmetic, a cast, a select,
        each over values the entry itself produced. Any other claim, or a value produced elsewhere (a
        variable, a parameter, a string, a function), is not an integer constant expression. The twin's
        `kfold`."""
        block = self.block_stack[-1]
        start = len(block)
        v = self._rvalue(expr)
        vals: dict = {}
        for c in block[start:]:
            if not isinstance(c, Claim) or len(c.wr) != 1 or any(r not in vals for r in c.rd):
                raise CLowerError(_NOT_CONSTANT)
            vals[c.wr[0]] = self._kfold(c, [vals[r] for r in c.rd])
        if v not in vals:
            raise CLowerError(_NOT_CONSTANT)
        return vals[v]

    def _kfold(self, c: Claim, ops: list) -> _KVal:
        """One claim of a static initializer's entry, folded (see `_const_value`)."""
        op = c.op
        if op == "c.const" and not ops and c.imm:
            return _kconvert(_KVal(int(c.imm[0])), _ktype(self.rtypes.get(c.wr[0])))
        if op.startswith("c.cast:") and len(ops) == 1:
            return _kconvert(ops[0], _ktype(self.rtypes.get(c.wr[0])))
        if op.startswith("c.bin.") and len(ops) == 2:
            return _kbin(op[len("c.bin.") :], *ops)
        if op.startswith("c.un.") and len(ops) == 1:
            return _kun(op[len("c.un.") :], ops[0])
        if op == "c.select" and len(ops) == 3:
            return _ksel(*ops)
        raise CLowerError(_NOT_CONSTANT)

    def _render_image(self, ct: CType, w: "_InitWalk") -> "str | None":
        """A static's image as its declaration's initializer -- the brace list both rails render: each
        scalar the walk stored holds its last value, and a zero one is the static's own zero. An array
        lists the elements that hold a value, designating one after a gap (`[5] = v`); a struct its members
        in order, up to the last that holds one; a union the member the walk gave it (`.m = v` for any but
        the first); a multi-dimensional static its elements flat, as it is declared. None when the whole
        image is zero (the declaration's own `{0}` / `0u`). The twin's `kimage`."""
        img: dict = {}
        for lo, width, v in w.image:
            img[(lo, width)] = v
        los = sorted(lo for (lo, _width), v in img.items() if v)
        if not los:
            return None
        if ct.kind == "array" and len(ct.shape) > 1:
            ct = array(ct.of, ct.count)
        return self._render_unit((ct, 0, 0, 0, False), img, los, w.unions, 0)

    def _render_unit(self, unit: tuple, img: dict, los: list, unions: dict, depth: int) -> str:
        """The rendered initializer of one subobject of a static's image (see `_render_image`); `los` the
        sorted bit offsets of the scalars that hold a nonzero value."""
        ct, off, bo, bw, _packed = unit
        lo = off * 8 + bo
        if ct.kind not in ("struct", "union", "array"):
            return _kspell(img.get((lo, bw or ct.size * 8), 0))
        hi = lo + ct.size * 8
        i = bisect_left(los, lo)
        if i == len(los) or los[i] >= hi:
            return "{0}"
        if depth >= _INIT_MAX_FRAMES:
            raise CLowerError(f"an initializer nested deeper than {_INIT_MAX_FRAMES} subobjects")
        fr = _IFrame(ct, off, count=self._init_count(ct))
        parts: list = []
        if ct.kind == "array":
            stride = self._array_row(ct)[1] * 8
            prev = -1
            while i < len(los) and los[i] < hi:
                fr.idx = (los[i] - lo) // stride
                sub = self._init_unit(fr)[0]
                pre = "" if fr.idx == prev + 1 else f"[{fr.idx}] = "
                parts.append(pre + self._render_unit(sub, img, los, unions, depth + 1))
                prev = fr.idx
                i = bisect_left(los, lo + (fr.idx + 1) * stride)
        elif ct.kind == "union":
            k = unions.get((off, ct.name), 0)
            if k and any(g[0] == k for g in ct.anon):
                raise CLowerError(_ANON_UNION)
            fr.idx = k
            sub = self._init_unit(fr)[0]
            pre = f".{ct.fields[k][0]} = " if k else ""
            parts.append(pre + self._render_unit(sub, img, los, unions, depth + 1))
        else:  # every member in order, up to the last that holds a value
            keep = 0
            while fr.idx < fr.count:
                sub, fr.idx = self._init_unit(fr)
                parts.append(self._render_unit(sub, img, los, unions, depth + 1))
                if parts[-1] not in ("0u", "{0}"):
                    keep = len(parts)
            del parts[keep:]
        return "{" + ", ".join(parts) + "}"

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
        if lv.kind != "var" and not (
            _is_scalar_member_lv(node.operand, lv) and not self._mmio(lv.rid)
        ):
            raise CLowerError(
                "inc/dec of this lvalue form is a follow-on"
            )  # MMIO / non-scalar member
        if lv.ct.atomic:  # an `_Atomic` object: one read-modify-write, no snapshot or re-read
            kind = ("pre" if node.prefix else "post") + ("inc" if node.op == "+" else "dec")
            return self._atomic_rmw(lv, kind)
        cur = self._read(lv)
        if (
            not node.prefix and lv.kind == "var"
        ):  # snapshot: _read hands back the mutable storage rid,
            old = self._temp(lv.ct, "id_old")  # which the store below would clobber -- a same-type
            self._emit(
                f"c.cast:{_cast_name(lv.ct)}", Opcode.ADD, (cur,), (old,)
            )  # cast DECLARES the temp
        else:
            old = cur  # a memory read is already a fresh declared temp
        one = self._rvalue(cast.IntLit(1))
        if lv.ct.kind == "pointer":  # a pointer steps by sizeof(*p): c.ptradd MUTATES the
            self._emit(
                "c.ptradd"
                if node.op == "+"
                else "c.ptrsub",  # storage IN PLACE (`p += 1`), so only a
                Opcode.ADD,
                (lv.rid, one),
                (lv.rid,),
            )  # named-local pointer reaches here (mem
            return (
                old if not node.prefix else lv.rid
            )  # pointer lvalues fall back); no separate write/re-read
        opcode, suf = _BIN[node.op]
        rt = self._bin_result_type(node.op, cur, one)
        new = self._temp(rt, f"id_{suf}")
        self._emit(f"c.bin.{suf}", opcode, (cur, one), (new,))
        self._write(lv, new)
        if not node.prefix:
            return old  # postfix: the pre-step value
        nt = self.rtypes.get(new)  # prefix: the STORED new value -- re-read if the target
        if lv.bit_width or (nt is not None and lv.ct.size < nt.size):
            return self._read(lv)  # narrows (a bitfield / sub-int local), else `new`
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
        # Only the parser's desugaring qualifies: it reuses the target node as the left operand. A
        # written-out `p = p + n` is an ordinary assignment of a pointer sum -- `c.bin.add` then a copy,
        # as `q = p + n` and `p = n + p` lower on both rails (the twin never folded it; CF-SPLIT2).
        if (
            isinstance(node.target, cast.Name)
            and node.target.ident in self.env
            and isinstance(node.value, cast.Binary)
            and node.value.op in ("+", "-")
            and node.value.lhs is node.target
        ):
            rid, ct = self._lookup(node.target.ident, node.target.pos)
            if ct.kind == "pointer":
                n = self._rvalue(node.value.rhs)
                self._emit(
                    "c.ptradd" if node.value.op == "+" else "c.ptrsub", Opcode.ADD, (rid, n), (rid,)
                )
                return rid
        # a compound assignment to a NAMED `_Atomic` object (`g += v`, the node-identity desugaring below):
        # one atomic read-modify-write of it, never a read, an operation and a copy back
        if (
            named_local
            and isinstance(node.value, cast.Binary)
            and node.value.lhs is node.target
            and self._lookup(node.target.ident, node.target.pos)[1].atomic
        ):
            lv = self._lvalue(node.target)
            return self._atomic_rmw(lv, _BIN[node.value.op][1], self._rvalue(node.value.rhs))
        # compound assignment to a MEMORY lvalue (`a[i] OP= e`, `s.m OP= e`, `*p OP= e`): the parser desugars
        # `lhs OP= e` to `lhs = lhs OP e` REUSING the same target node, so resolve the lvalue ONCE (C
        # evaluates `lhs` once) -- one index/address, read + stored through it -- instead of recomputing it
        # for the read and again for the store (which both diverges from the twin and would double-run a
        # side-effecting index). The node-IDENTITY test distinguishes this from an explicit `a[i]=a[i]*e`.
        if (
            isinstance(node.value, cast.Binary)
            and node.value.lhs is node.target
            and not isinstance(node.target, cast.Name)
        ):
            lv = self._lvalue(node.target)
            if not stmt and not (_is_scalar_member_lv(node.target, lv) and not self._mmio(lv.rid)):
                raise CLowerError(
                    "this lvalue form's compound assignment as a value is a follow-on"
                )
            if lv.ct.atomic:  # an `_Atomic` object: one read-modify-write, its value the new one
                return self._atomic_rmw(lv, _BIN[node.value.op][1], self._rvalue(node.value.rhs))
            cur = self._read(lv)
            b = self._rvalue(node.value.rhs)
            opcode, suf = _BIN[node.value.op]
            rt = self._bin_result_type(node.value.op, cur, b)
            t = self._temp(rt, f"b_{suf}")
            res = self._emit(f"c.bin.{suf}", opcode, (cur, b), (t,))
            # the value as stored: converted to the target's type when its class differs (`s.i += 0.5f`)
            res = self._write(lv, res)
            # the value of a compound assignment is the STORED (narrowed) value, not the raw binop result --
            # when the target is NARROWER than the promoted result the store truncates, so re-read it (like
            # the plain `=` path); `res` would be the un-narrowed sum (a both-rails miscompile). A BITFIELD
            # narrows to its BIT width (its ct.size is the full underlying type, so the size test misses it) --
            # re-read it too. A full-width non-bitfield target needs no re-read (res == the stored value).
            narrows = lv.bit_width or lv.ct.size < self.rtypes.get(res, rt).size
            return res if (stmt or not narrows) else self._read(lv)
        v = self._rvalue(node.value)
        if named_local:
            rid, _ct = self._lookup(
                node.target.ident, node.target.pos
            )  # copy into the mutable storage
            v = self._null_pointer(v, _ct)
            # a plain `=` assigns a struct only its own type (not `x OP= e`'s desugaring)
            if not (isinstance(node.value, cast.Binary) and node.value.lhs is node.target):
                self._struct_value(_ct, v, _STRUCT_ASSIGNED)
            self._emit("c.copy", Opcode.ADD, (v,), (rid,), **_object_write(_ct))
            self._bind_extent(
                rid, _ct, node.target.ident, node.value
            )  # §5.12: `p = malloc(N*…)` -> N
            if _ct.atomic and not stmt:  # the value stored, converted: a re-read by name would be
                return self._cast_value(v, unqualified(_ct))  # a second atomic access
            return rid
        lv = self._lvalue(node.target)
        if not stmt and not (_is_scalar_member_lv(node.target, lv) and not self._mmio(lv.rid)):
            raise CLowerError(
                "this lvalue form's assignment as a value is a follow-on"
            )  # MMIO re-read unsound
        # a struct member, element or `*p` is assigned only its own type
        self._struct_value(lv.ct, v, _STRUCT_ASSIGNED)
        stored = self._write(lv, v)
        if lv.ct.atomic and not stmt:  # the value stored, converted to the object's type: a re-read
            return self._cast_value(stored, unqualified(lv.ct))  # would be a second atomic access
        return v if stmt else self._read(lv)  # value context: the stored/converted value (re-read)

    # --- inline assembly (ASM1): an ISA-neutral TRUSTED OPAQUE EFFECT EDGE, modeled on c.call.libm.void: ---
    def _asm_output_rid(self, name_or_none, constraint: str, lvalue) -> int:
        """The destination RID for one extended-asm OUTPUT operand `"=c"(lvalue)` / `"+c"(lvalue)`. ASM1
        supports a scalar NAMED LOCAL lvalue (the dominant portable form: `asm("" : "=r"(y) : "r"(x))`), whose
        rid IS its mutable storage -- the asm writes through it, so a downstream read of the variable sees the
        new value, exactly like an assignment. A non-named-local / non-scalar / bitfield / MMIO output lvalue
        is a follow-on (deferred), reported as an honest diagnostic that routes to the LLVM fallback."""
        if isinstance(lvalue, cast.Name) and lvalue.ident in self.env:
            rid, ct = self._lookup(lvalue.ident, lvalue.pos)
            if ct.kind != "scalar":
                raise CLowerError(
                    f"inline-asm output to a non-scalar lvalue ('{lvalue.ident}') is a follow-on"
                )
            return rid
        raise CLowerError(
            "inline-asm output operand must be a scalar local variable in this subset (a member / array / "
            "deref / bitfield / MMIO output lvalue is a follow-on)"
        )

    def _asm_stmt(self, st: cast.AsmStmt) -> None:
        """Lower a GNU inline-asm statement to a TRUSTED OPAQUE EFFECT EDGE -- a `c.asm:` (or
        `c.asm.volatile:`) claim, the sibling of `c.call.libm.void:`. The assembly template + per-operand
        constraints + clobbers are stashed VERBATIM in an `AsmInfo` (keyed by claim id) so the emit
        reconstructs the exact `__asm__` statement (ISA-neutral pass-through); the operand RIDs ride the
        claim's read/write set so the alias/effect/verify machinery sees the real footprint.

        Semantics:
          * an INPUT operand's value rid is a READ; a `"+"` (read-write) OUTPUT lvalue is BOTH read + written;
            a `"="` (write-only) OUTPUT lvalue is a WRITE.
          * a `volatile` asm (and a basic asm, which is implicitly volatile) is a SIDE-EFFECTING edge: it must
            NOT be dead-code-eliminated even with unused outputs. The cfront emit walks `lf.body` (every claim,
            in source order) -- it never reorders or drops a claim -- so the asm edge is conservatively never
            moved/eliminated. A `"memory"` clobber / `volatile` asm additionally carries the `barriered` hazard
            (the existing MMIO ordering-barrier marker) so any consumer of the hazard contract treats it as an
            ordering fence, never to be reordered or fused across.
          * `asm goto` (a label list) is rejected with an honest diagnostic -- its control flow is deferred."""
        if st.is_goto or st.goto_labels:
            raise CLowerError(
                "`asm goto` (label operands) is not yet supported (deferred to a later slice)"
            )
        in_rids: list = []
        in_names: list = []
        in_constraints: list = []
        for nm, constraint, expr in st.inputs:  # inputs first: their value rids are pure reads
            in_rids.append(self._rvalue(expr))
            in_names.append(nm)
            in_constraints.append(_strip_quotes(constraint))
        out_rids: list = []
        out_names: list = []
        out_constraints: list = []
        rw_reads: list = []  # `"+"` outputs are also reads
        for nm, constraint, lvalue in st.outputs:
            dest = self._asm_output_rid(nm, constraint, lvalue)
            out_rids.append(dest)
            out_names.append(nm)
            out_constraints.append(_strip_quotes(constraint))
            if "+" in constraint:  # read-write output: the lvalue is read too
                rw_reads.append(dest)
        # a `"memory"` clobber OR a volatile asm is an ordering barrier; mark it with the existing
        # `barriered` hazard so it is treated as a fence (never reordered/fused across) -- else `unique`.
        clob_spellings = tuple(st.clobbers)
        is_barrier = st.is_volatile or any(_strip_quotes(c) == "memory" for c in clob_spellings)
        hazard = "barriered" if is_barrier else "unique"
        op = "c.asm.volatile:" if st.is_volatile else "c.asm:"
        rd = tuple(in_rids) + tuple(rw_reads)  # inputs + read-write output lvalues (reads)
        wr = tuple(out_rids)  # write-only + read-write output lvalues (writes)
        self._emit(op, Opcode.GEM_DISPATCH, rd, wr, hazard=hazard)
        claim_id = self.cid[0]  # the id `_emit` just assigned (== the claim's .id)
        self.asm_meta[claim_id] = AsmInfo(
            template=st.template,
            out_names=tuple(out_names),
            out_constraints=tuple(out_constraints),
            out_rids=tuple(out_rids),
            in_names=tuple(in_names),
            in_constraints=tuple(in_constraints),
            in_rids=tuple(in_rids),
            clobbers=tuple(_strip_quotes(c) for c in clob_spellings),
            is_volatile=st.is_volatile,
            is_basic=st.is_basic,
        )

    # GCC/Clang atomic + fence builtins -> the BCIR ATOMIC_*/BARRIER opcodes (§5.8).
    _ATOMIC = {
        "__atomic_fetch_add": ("c.atomic.add", Opcode.ATOMIC_ADD),
        "__atomic_fetch_sub": ("c.atomic.sub", Opcode.ATOMIC_SUB),
        "__atomic_fetch_xor": ("c.atomic.xor", Opcode.ATOMIC_XOR),
    }
    # ASM3 -- memory-fence (hardware barrier) intrinsics -> the BARRIER opcode, keyed by KIND. The op string
    # carries the kind so emit.py realizes the per-ISA instruction behind `--target` (full -> mfence/dmb ish/
    # fence rw,rw; acquire -> lfence/dmb ishld/fence r,rw; release -> sfence/dmb ishst/fence rw,w). The full
    # fence keeps the BACKWARD-COMPATIBLE op string `c.fence` (so the existing __atomic_thread_fence /
    # __sync_synchronize claims -- and the C-twin parity corpus -- are unchanged, no digest/parity churn); only
    # the lighter kinds get the new op strings. `__sync_synchronize` is an unconditional SEQ_CST full fence.
    # SEG6.1 -- the order-taking `__atomic_thread_fence(order)` (GCC/Clang) and `atomic_thread_fence(order)`
    # (C11 `<stdatomic.h>`) forms NOW parse their `memory_order` ARGUMENT and route to the kind it implies
    # (acquire/release fold to the lighter op strings; seq_cst/acq_rel/relaxed and any non-constant order fold
    # conservatively to the full `c.fence` -- a stronger fence never under-synchronizes). The bare `_FENCE`
    # entry below is the FALLBACK (no-arg) op for these names: a degenerate `atomic_thread_fence()` with no
    # argument keeps the full fence. The x86-conventional `_mm_{m,l,s}fence` intrinsic NAMES encode the kind
    # directly (they take no order argument), so their kind is read off the name.
    _FENCE = {
        "__atomic_thread_fence": "c.fence",
        "__sync_synchronize": "c.fence",
        "atomic_thread_fence": "c.fence",  # C11 <stdatomic.h> -- order-parameterized
        "_mm_mfence": "c.fence",  # x86 mfence -- full (load+store) fence
        "_mm_lfence": "c.fence.acquire",  # x86 lfence -- load (acquire) fence
        "_mm_sfence": "c.fence.release",
    }  # x86 sfence -- store (release) fence
    # The order-taking fence forms (SEG6.1): these parse `node.args[0]` as a `memory_order` and route to the
    # kind it implies. The no-arg forms (`__sync_synchronize`, `_mm_*`) are NOT in this set and keep `_FENCE`.
    _FENCE_ORDERED = frozenset({"__atomic_thread_fence", "atomic_thread_fence"})
    # The C11 `memory_order_*` constants AND the GCC/Clang `__ATOMIC_*` macro spellings (they share the same
    # integer values). A `memory_order` arg spelled as one of these named constants folds to its value.
    _MEMORDER = {
        "memory_order_relaxed": 0,
        "memory_order_consume": 1,
        "memory_order_acquire": 2,
        "memory_order_release": 3,
        "memory_order_acq_rel": 4,
        "memory_order_seq_cst": 5,
        "__ATOMIC_RELAXED": 0,
        "__ATOMIC_CONSUME": 1,
        "__ATOMIC_ACQUIRE": 2,
        "__ATOMIC_RELEASE": 3,
        "__ATOMIC_ACQ_REL": 4,
        "__ATOMIC_SEQ_CST": 5,
    }
    # order value -> fence-kind op string. acquire(2)/consume(1) -> a load (acquire) fence; release(3) -> a
    # store (release) fence; seq_cst(5)/acq_rel(4)/relaxed(0) -> the full fence (a sound over-approximation --
    # a stronger fence never under-synchronizes, so acq_rel and relaxed both conservatively fold to full).
    _ORDER_KIND = {
        0: "c.fence",
        1: "c.fence.acquire",
        2: "c.fence.acquire",
        3: "c.fence.release",
        4: "c.fence",
        5: "c.fence",
    }

    def _fence_order_kind(self, arg_node) -> str:
        """Resolve a `memory_order` ARGUMENT AST node to a fence-kind op string (SEG6.1). An integer literal
        uses its value; a named `memory_order_*` / `__ATOMIC_*` constant uses its mapped value; anything else
        (a runtime variable, an unrecognized name, a non-constant expression) folds conservatively to the
        FULL fence (`c.fence`) -- sound (a stronger fence never under-synchronizes) and never crashes."""
        if isinstance(arg_node, cast.IntLit):
            order = arg_node.value
        elif (
            isinstance(arg_node, cast.Name)
            and arg_node.ident in self._MEMORDER
            and arg_node.ident not in self.env
            and arg_node.ident not in self.func_rets
        ):  # the constant ONLY when NOT shadowed by a real
            order = self._MEMORDER[arg_node.ident]  #   decl OR a same-named function -- EXACTLY
            #   _rvalue's precedence (an env local, then a func name -> a funcptr value, only THEN the _MEMORDER
            #   constant), so the kind rail never disagrees with the value rail on a shadowed name.
        else:
            order = 5  # non-constant / shadowed / unknown -> seq_cst (full)
        return self._ORDER_KIND.get(order, "c.fence")  # an out-of-range int folds to full too

    # Compare-and-swap -> the CMPXCHG opcode: a 3-read claim (ptr, expected, desired). The `val`
    # form returns the pre-swap value, the `bool` form returns whether the swap happened.
    _CMPXCHG = {
        "__sync_val_compare_and_swap": "c.cmpxchg.val",
        "__sync_bool_compare_and_swap": "c.cmpxchg.bool",
    }
    # C11 <stdatomic.h> generics on _Atomic objects -> the same ATOMIC opcodes, but emitted as the C11
    # functions (which accept an _Atomic* -- the __atomic_* builtins do not).
    _C11_RMW = {
        "atomic_fetch_add": ("c.c11atom.fetch_add", Opcode.ATOMIC_ADD),
        "atomic_fetch_sub": ("c.c11atom.fetch_sub", Opcode.ATOMIC_SUB),
        "atomic_fetch_xor": ("c.c11atom.fetch_xor", Opcode.ATOMIC_XOR),
        "atomic_exchange": ("c.c11atom.exchange", Opcode.ATOMIC_ADD),
    }  # swap: set + return old

    def _atomic_value_type(self, ptr: int) -> CType:
        """The type of the value an atomic builtin reads through `ptr` -- `atomic_load`, a fetch-op,
        `atomic_exchange`, a value compare-and-swap: the pointee's, unqualified (C11 7.17.7, `C
        atomic_load(volatile A *)` returns the non-atomic type of `*obj`). A uint32 temp truncated a
        64-bit counter and converted an `_Atomic float` to an integer (CF-ATOMIC). A pointer to a pointer,
        an aggregate or `void`, or a value the lowering cannot type, keeps the uint32 it had. The twin's
        `atomic_value_temp`."""
        pt = self.rtypes.get(ptr)
        el = pt.of if pt is not None and pt.kind in ("pointer", "array") else None
        if el is None or el.kind != "scalar" or el.name == "void":
            return scalar("uint32_t")
        return unqualified(el)

    def _call(self, node: cast.CallExpr) -> int:
        actuals = tuple(self._rvalue(a) for a in node.args)
        # Indirect call through a function-pointer local/param (HAL dispatch): the target is dynamic,
        # so there is no named callee -- it lowers to a `c.call.indirect` claim (reads: the pointer
        # value then the actuals) and is *not* added to the call graph, leaving R18 to treat it as an
        # opaque external edge (no recursion / callee-resolution constraint can apply).
        if node.callee in self.env and self.env[node.callee][1].kind == "funcptr":
            fptr, fpct = self.env[node.callee]  # the funcptr CType carries its return type in `.of`
            # an argument converts to its parameter's type (C11 6.5.2.2p7), through a pointer as in a direct
            # call: a null pointer constant is the parameter's pointer (CF-NULLCALL; CF-NULLARG's direct calls)
            self._null_pointer_args(actuals, fpct.params)
            if _returns_void(fpct):
                # a pointer to a void function: the call has no value -- no result temp (whose emit
                # `uint32_t t = fp();` did not compile), and `c ? cb() : (void)0` / `return cb();` take
                # the void value as a direct void call's (CF-VOIDCB)
                self._emit(
                    "c.call.indirect",
                    Opcode.GEM_DISPATCH,
                    (fptr, *actuals),
                    (),
                    callee_sig=callee_signature(fpct),
                )
                return _VOID_RID
            t = self._temp(
                self._call_result_ct(fpct.of), f"icall_{node.callee}"
            )  # a signed return reads back signed
            return self._emit(
                "c.call.indirect",
                Opcode.GEM_DISPATCH,
                (fptr, *actuals),
                (t,),
                callee_sig=callee_signature(fpct),
            )
        # Atomics run on the A lane. A scalar atomic counter is a single-location RMW (not on
        # the decoupled GGG/scatter tail), so it stays SCALAR-shaped -- the lane law (R6) admits
        # lane A for SCALAR, and the atomic/barriered hazard discharges R5.
        if node.callee in self._FENCE and node.callee not in self.func_rets:
            # SEG6.1: the order-taking forms (`__atomic_thread_fence`/`atomic_thread_fence`) route by their
            # `memory_order` ARGUMENT (read off the ORIGINAL AST node -- the arg was already evaluated into
            # `actuals` above, emitting the same const claim the C twin emits, so parity is preserved); the
            # no-arg forms (`__sync_synchronize`, `_mm_{m,l,s}fence`) encode their KIND in the name.
            if node.callee in self._FENCE_ORDERED and node.args:
                op = self._fence_order_kind(node.args[0])
            else:
                op = self._FENCE[
                    node.callee
                ]  # c.fence | c.fence.acquire | c.fence.release (the KIND)
            t = self._temp(scalar("uint32_t"), "fence")
            self._emit(op, Opcode.BARRIER, (), (), lane=Lane.A, hazard="barriered")
            return t
        if node.callee in self._ATOMIC:
            op, oc = self._ATOMIC[node.callee]
            ptr = actuals[0]
            val = actuals[1] if len(actuals) > 1 else actuals[0]
            t = self._temp(self._atomic_value_type(ptr), "atom")
            dom = self.resources[ptr].domain if ptr in self.resources else Domain.RAM
            return self._emit(op, oc, (ptr, val), (t,), lane=Lane.A, domain=dom, hazard="atomic")
        if node.callee in self._CMPXCHG:
            op = self._CMPXCHG[node.callee]
            ptr = actuals[0]
            exp = actuals[1] if len(actuals) > 1 else ptr
            des = actuals[2] if len(actuals) > 2 else exp
            vt = self._atomic_value_type(ptr) if op == "c.cmpxchg.val" else scalar("uint32_t")
            t = self._temp(vt, "cas")  # the value form returns the old `*p`; the bool form 0/1
            dom = self.resources[ptr].domain if ptr in self.resources else Domain.RAM
            return self._emit(
                op, Opcode.CMPXCHG, (ptr, exp, des), (t,), lane=Lane.A, domain=dom, hazard="atomic"
            )
        if node.callee in ("atomic_compare_exchange_strong", "atomic_compare_exchange_weak"):
            # C11 bool atomic_compare_exchange_strong/weak(A *obj, C *expected, C desired): if *obj ==
            # *expected set *obj = desired (true), else load *obj into *expected (false). `expected` is a
            # POINTER -- the headline use of address-of (`&exp`), which is why this waited on `&`. Emitted
            # verbatim; the atomic semantics are delegated to the resident backend / Clang.
            op = "c.c11atom.cas_strong" if node.callee.endswith("strong") else "c.c11atom.cas_weak"
            ptr, exp = actuals[0], (actuals[1] if len(actuals) > 1 else actuals[0])
            des = actuals[2] if len(actuals) > 2 else exp
            t = self._temp(scalar("_Bool"), "c11cas")  # the comparison result is _Bool
            dom = self.resources[ptr].domain if ptr in self.resources else Domain.RAM
            return self._emit(
                op, Opcode.CMPXCHG, (ptr, exp, des), (t,), lane=Lane.A, domain=dom, hazard="atomic"
            )
        if node.callee in self._C11_RMW:  # atomic_fetch_add/sub/xor(p, v) -> old value
            op, oc = self._C11_RMW[node.callee]
            ptr = actuals[0]
            val = actuals[1] if len(actuals) > 1 else actuals[0]
            t = self._temp(self._atomic_value_type(ptr), "c11")
            dom = self.resources[ptr].domain if ptr in self.resources else Domain.RAM
            return self._emit(op, oc, (ptr, val), (t,), lane=Lane.A, domain=dom, hazard="atomic")
        if node.callee == "atomic_load":  # atomic_load(p) -> *p (ordered)
            ptr = actuals[0]
            t = self._temp(self._atomic_value_type(ptr), "c11ld")
            dom = self.resources[ptr].domain if ptr in self.resources else Domain.RAM
            return self._emit(
                "c.c11atom.load",
                Opcode.LOAD,
                (ptr,),
                (t,),
                lane=Lane.A,
                domain=dom,
                hazard="atomic",
            )
        if node.callee == "atomic_store":  # atomic_store(p, v) (ordered, no value)
            ptr = actuals[0]
            val = actuals[1] if len(actuals) > 1 else actuals[0]
            dom = self.resources[ptr].domain if ptr in self.resources else Domain.RAM
            self._emit(
                "c.c11atom.store",
                Opcode.STORE,
                (ptr, val),
                (),
                lane=Lane.A,
                domain=dom,
                hazard="atomic",
            )
            return ptr
        if node.callee in (
            "va_start",
            "va_end",
            "va_copy",
        ):  # opaque void variadic builtins (verbatim emit)
            self._emit(f"c.call.vabuiltin:{node.callee}", Opcode.GEM_DISPATCH, actuals, ())
            return _VOID_RID  # not added to self.calls: opaque to the R18 call graph
        if (
            node.callee in _PORTIO_IN or node.callee in _PORTIO_OUT
        ) and node.callee not in self.func_rets:
            return self._portio(node, actuals)  # ASM2: a typed, isolated, barriered I/O-port edge
        libm = _libm_type(node.callee)
        if libm is not None:  # a <math.h> call -> a typed external library edge
            t = self._temp(
                libm, f"libm_{node.callee}"
            )  # not added to self.calls: opaque to the call
            return self._emit(
                f"c.call.libm:{node.callee}", Opcode.GEM_DISPATCH, actuals, (t,)
            )  # graph
        if node.callee in _EXTERN_VARIADIC and node.callee not in self.func_rets:
            t = self._temp(
                scalar("int", self.abi), f"ext_{node.callee}"
            )  # a printf/scanf-family external
            return self._emit(
                f"c.call.extern:{node.callee}", Opcode.GEM_DISPATCH, actuals, (t,)
            )  # variadic
        if node.callee in _STDLIB_ALLOC and node.callee not in self.func_rets:
            # `realloc(0, n)`: a null `void *` (CF-NULLCALL)
            if node.callee == "realloc" and actuals:
                self._null_pointer(actuals[0], pointer(scalar("void"), self.abi))
            t = self._temp(
                pointer(scalar("void")), f"mem_{node.callee}"
            )  # malloc/calloc/realloc -> void *
            # a R21 lifetime ALLOC event on the result: it (re-)validates the allocation, so a pointer assigned
            # from it is live again after an earlier free (`p=malloc; free(p); p=malloc; p[i]` is legal). Pairs
            # with the `free` event below; digest-excluded + vacuous unless the smart-lowering verifier runs R21.
            return self._emit(
                f"c.call.libm:{node.callee}",
                Opcode.GEM_DISPATCH,
                actuals,
                (t,),  # verbatim, opaque
                lifetime=Lifetime("alloc"),
            )
        if node.callee in _STRING_MEM and node.callee not in self.func_rets:
            # a <string.h> routine: its result is its destination, a `void *`
            t = self._temp(pointer(scalar("void")), f"mem_{node.callee}")
            return self._emit(f"c.call.libm:{node.callee}", Opcode.GEM_DISPATCH, actuals, (t,))
        if node.callee == "free" and node.callee not in self.func_rets:
            # void external (verbatim, opaque) + a R21 lifetime FREE event: the freed pointer (the actual it
            # reads) dies after this claim, so a later dereference of it is a use-after-free (§5.12). Vacuous
            # unless the smart-lowering verifier runs R21 -- the frontend pass/fail is unchanged.
            if actuals:  # `free(0)`: a null `void *` (CF-NULLCALL)
                self._null_pointer(actuals[0], pointer(scalar("void"), self.abi))
            self._emit(
                "c.call.libm.void:free", Opcode.GEM_DISPATCH, actuals, (), lifetime=Lifetime("free")
            )
            return _VOID_RID
        bt = _builtin_type(node.callee, self.abi)
        if bt is not None:  # a GCC/Clang integer builtin -> verbatim, typed, opaque
            t = self._temp(bt, "blt")  # the op carries the suffix (the `__builtin_` is re-added
            return self._emit(
                f"c.call.builtin:{node.callee[len('__builtin_') :]}",  # in emit; the op buffer is small)
                Opcode.GEM_DISPATCH,
                actuals,
                (t,),
            )
        if node.callee in self.func_rets or node.callee in self.protos:
            self._require_declared(node.callee, "call to undeclared function")
        if node.callee in self.protos and node.callee not in self.func_rets:
            # Phase 3 LINKING: a PROTOTYPED cross-TU callee -- a TYPED external edge the host LINKER
            # resolves from a sibling object. Like a libm edge it is opaque to the in-unit R18 call
            # graph (not appended to self.calls) and emits VERBATIM (no bcir_ rename -- external
            # linkage); unlike libm it derives NO -l flag (linkflags: a sibling TU, not a library).
            # The emit declares the recorded signature so the emitted TU compiles standalone.
            ret_ct, param_cts = self.protos[node.callee]
            self._null_pointer_args(actuals, param_cts)
            consts = self.proto_consts.get(node.callee, (False,) * len(param_cts))
            self.tu_used[node.callee] = (ret_ct, param_cts, consts)
            if ret_ct.name == "void":
                self._emit(f"c.call.tu:{node.callee}", Opcode.GEM_DISPATCH, actuals, ())
                return _VOID_RID
            t = self._temp(self._call_result_ct(ret_ct), f"tu_{node.callee}")
            return self._emit(f"c.call.tu:{node.callee}", Opcode.GEM_DISPATCH, actuals, (t,))
        self.calls.append((node.callee, actuals))  # a defined-in-unit callee: a real R18 edge
        self._null_pointer_args(actuals, self.func_params.get(node.callee, ()))
        ret_ct = self.func_rets.get(node.callee)
        if (
            ret_ct is not None and ret_ct.name == "void"
        ):  # a void callee -> a bare call statement, no
            self._emit(
                f"c.call.void:{node.callee}", Opcode.GEM_DISPATCH, actuals, ()
            )  # result temp
            return _VOID_RID  # the value of a void call is never read
        # type the result temp by the callee's return type: a float return propagates (so downstream
        # arithmetic stays float), a wide (8-byte) integer return keeps its width, and a signed `int` return
        # keeps its SIGN -- else a downstream `>>` / comparison on the call result would go unsigned (a
        # logical shift / unsigned compare). A struct/union RETURN keeps the aggregate CType (the result is a
        # by-value struct temp -- `struct P t = mk(x);` -- so it can be copied into a local, passed by value, or
        # member-accessed; typing it uint32 emitted invalid C `uint32_t t = mk(x)`).
        t = self._temp(self._call_result_ct(ret_ct), f"call_{node.callee}")
        return self._emit(f"c.call:{node.callee}", Opcode.GEM_DISPATCH, actuals, (t,))

    # the I/O-port "address space" resource: a single MMIO-domain resource ALL port accesses share, so two
    # port ops alias each other (ordered against one another, never reordered/fused) but a port access never
    # aliases a normal-memory RID. Its rid is the RESERVED `_IO_PORT_RID` sentinel -- provably disjoint from
    # every count-indexed band (the allocators are guarded against ever reaching it, and the global/string
    # merge never overwrites this MMIO resource). One per function, lazily created -- the isolation seam.
    def _io_port_space(self) -> int:
        rid = getattr(self, "_io_port_rid", None)
        if rid is None:
            rid = _IO_PORT_RID  # reserved sentinel: never produced by base+count
            self.resources[rid] = Resource(
                rid=rid,
                domain=Domain.MMIO,
                elem_bytes=1,
                shape=(1 << 16,),  # the 64K I/O port address space
                access="volatile",
                name="__ioport",
            )
            self._io_port_rid = rid
        return rid

    def _portio(self, node: cast.CallExpr, actuals: tuple) -> int:
        """ASM2 -- lower a port-mapped I/O intrinsic CALL to a typed, isolated, BARRIERED I/O-port edge.

        The edge is `c.portio.in.<b|w|l>:` (a READ -> the value) or `c.portio.out.<b|w|l>:` (a void WRITE),
        carrying the access WIDTH + DIRECTION in the op suffix and the I/O-port-space rid on the read/write
        set so the alias/effect machinery sees a real I/O footprint. Three properties the design pins:

          * ISOLATED -- the port access reads/writes the dedicated `__ioport` MMIO resource (the I/O address
            space), so it can never alias a normal-memory RID; `domain=Domain.MMIO` carries it.
          * BARRIERED -- `hazard='barriered'` (the existing MMIO ordering marker, reused): port I/O is volatile
            + ordered, so two port ops must never be reordered / fused / eliminated. The cfront emit walks the
            body in source order and never drops a claim, so the edge also survives even when its result is
            unused (a side-effecting probe).
          * OFF THE LEGALITY VALUE-PATH -- a trusted effect (Opcode.GEM_DISPATCH), no Diagnostic verdict, like
            the `c.asm`/`c.call.libm` edges.

        The per-ISA `in`/`out` instruction is NOT chosen here (ISA-neutral IR): emit.py realizes it keyed off
        the function's target (x86 -> the real instruction as `__asm__ __volatile__`; a non-x86 target raises
        the honest unsupported diagnostic). Arity + operand width are validated here (an honest CLowerError)."""
        io = self._io_port_space()
        if node.callee in _PORTIO_IN:  # inb/inw/inl(port) -> u8/u16/u32
            width, rty = _PORTIO_IN[node.callee]
            if len(actuals) != 1:
                raise CLowerError(
                    f"{node.callee}(port) takes exactly 1 argument (the u16 I/O port), got {len(actuals)}"
                )
            port = actuals[0]
            self._check_port_operand(node.callee, port)
            suf = {1: "b", 2: "w", 4: "l"}[width]
            t = self._temp(scalar(rty, self.abi), f"portio_{node.callee}")
            # rd = (port value, the I/O-port space); wr = (the result temp). The io rid on BOTH sides makes a
            # read ALSO an effect on the port space, so two reads do not commute (ordered, never fused).
            return self._emit(
                f"c.portio.in.{suf}:{node.callee}",
                Opcode.GEM_DISPATCH,
                (port, io),
                (t, io),
                imm=(width,),
                domain=Domain.MMIO,
                lane=Lane.H,
                hazard="barriered",
            )
        # outb/outw/outl(value, port) -- the Linux <asm/io.h> convention: VALUE first, PORT second (void).
        width = _PORTIO_OUT[node.callee]
        if len(actuals) != 2:
            raise CLowerError(
                f"{node.callee}(value, port) takes exactly 2 arguments (value, then the u16 I/O port), "
                f"got {len(actuals)} -- note the Linux out*(value, port) order"
            )
        value, port = actuals[0], actuals[1]
        self._check_port_operand(node.callee, port)
        self._check_value_operand(node.callee, value, width)
        suf = {1: "b", 2: "w", 4: "l"}[width]
        # rd = (value, port, the I/O-port space); wr = (the port space). The void write produces no value
        # temp; its write of `io` makes the edge a real effect that orders against every other port op.
        self._emit(
            f"c.portio.out.{suf}:{node.callee}",
            Opcode.GEM_DISPATCH,
            (value, port, io),
            (io,),
            imm=(width,),
            domain=Domain.MMIO,
            lane=Lane.H,
            hazard="barriered",
        )
        return _VOID_RID  # a void write: its value is never read

    def _check_port_operand(self, callee: str, port: int) -> None:
        """The `port` operand is the I/O port address -- conceptually a u16. Accept any integer operand
        (the value model is integer; a wider expression is masked to u16 by the `"Nd"`/`%w` constraint the
        x86 emit uses), but reject a non-integer (float / pointer / aggregate) port as an honest diagnostic."""
        pct = self.rtypes.get(port)
        if pct is not None and not pct.is_integer and pct.kind != "scalar":
            raise CLowerError(f"{callee}: the I/O port must be an integer (u16), not a {pct.kind}")
        if pct is not None and pct.is_float:
            raise CLowerError(
                f"{callee}: the I/O port must be an integer (u16), not a floating value"
            )

    def _check_value_operand(self, callee: str, value: int, width: int) -> None:
        """The `value` written by out{b,w,l} is a u8/u16/u32 matching the suffix width. Accept any integer
        (the x86 `"a"` accumulator constraint takes the low byte/word/dword); reject a non-integer value."""
        vct = self.rtypes.get(value)
        if vct is not None and vct.is_float:
            raise CLowerError(
                f"{callee}: the written value must be an integer (u{width * 8}), not a float"
            )
        if vct is not None and not vct.is_integer and vct.kind != "scalar":
            raise CLowerError(
                f"{callee}: the written value must be an integer (u{width * 8}), not a {vct.kind}"
            )

    def _call_result_ct(self, ret_ct):
        """The result-temp CType for a call, by the callee's return type -- shared by direct, indirect
        (funcptr param), and member (funcptr field / dispatch) calls so all four type identically. A struct
        return keeps the aggregate; a float or wide (>4-byte) integer keeps its CType; a SIGNED sub-int return
        promotes to `int` (so a downstream `>>` / compare / `(long)` widen sign-extends); else uint32. A
        funcptr whose return type wasn't captured (`ret_ct is None`) stays uint32 -- today's behaviour."""
        if ret_ct is not None and ret_ct.is_aggregate:
            return ret_ct
        # a pointer return stays a pointer -- `f()->v`, `*f()`, `T *p = f()` (a 4-byte unit truncated it)
        if ret_ct is not None and ret_ct.kind == "pointer":
            return self._complete_ptr(ret_ct)
        if (
            ret_ct is not None and ret_ct.is_bitint
        ):  # a C23 `_BitInt(N)` return keeps its exact width
            return ret_ct  # (it does not promote, and same-type arithmetic on
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
        saved_env = dict(self.env)  # a `{ ... }` is a scope: locals declared
        for s in stmts:  # inside it do not leak to the enclosing
            self._stmt(s)  # block (so a shadow does not clobber an
        self.env = saved_env  # outer same-named var read after the block)
        self.block_stack.pop()
        return block

    def _stmt(self, st):
        if isinstance(st, cast.Seq):  # several declarators of one declaration
            for s in st.stmts:
                self._stmt(s)
            return None
        if isinstance(st, cast.Block):  # a bare `{ ... }` -> its scoped nodes,
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
                    raise CLowerError(
                        "VLA of an incomplete / void / array element is not supported"
                    )
                nval = self._rvalue(st.type.vla)  # the size expression, evaluated exactly once
                nct = self.rtypes.get(nval)
                if nct is None or nct.kind != "scalar" or nct.is_float:
                    raise CLowerError("a VLA size must be an integer expression")
                ext = self._storage(
                    nct, f"__bcir_ext{self._ext_ctr}"
                )  # the snapshot: immutable extent
                self._ext_ctr += 1
                self._emit("c.copy", Opcode.ADD, (nval,), (ext,))
                ct = array(elem, 0)  # count 0 -> a runtime extent; bounds via ptr_extent
                rid = self._vla_storage(
                    ct, st.name, ext
                )  # declared in-body; masked `a[i]` vs the snapshot
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
                    raise CLowerError(
                        "a multi-dimensional VLA of an incomplete / array element is not supported"
                    )
                u32 = scalar("uint32_t", self.abi)
                dim_exts = []  # 1. snapshot each dim into a named __bcir_extK
                for d in st.type.vla_dims:
                    ext = self._storage(u32, f"__bcir_ext{self._ext_ctr}")
                    self._ext_ctr += 1
                    if isinstance(d, int):  # a literal dim -> a const temp, copied into the ext
                        kt = self._temp(u32, "kd")
                        self._emit("c.const", Opcode.LOAD, (), (kt,), imm=(d,))
                        self._emit("c.copy", Opcode.ADD, (kt,), (ext,))
                    else:  # a runtime dim -> evaluated once, copied in
                        self._emit("c.copy", Opcode.ADD, (self._rvalue(d),), (ext,))
                    dim_exts.append(ext)
                total = dim_exts[0]  # 2. total extent = product of all dim snapshots
                for d in range(1, len(dim_exts)):
                    prod = self._temp(u32, "b_mul")
                    self._emit("c.bin.mul", Opcode.MUL, (total, dim_exts[d]), (prod,))
                    total = prod
                ext_total = self._storage(u32, f"__bcir_ext{self._ext_ctr}")
                self._ext_ctr += 1
                self._emit(
                    "c.copy", Opcode.ADD, (total,), (ext_total,)
                )  # stable name for the decl + the mask
                ct = array(elem, 0)  # 3. a FLAT runtime-extent array
                rid = self._vla_storage(ct, st.name, ext_total)
                self.env[st.name] = (rid, ct)
                self.ptr_extent[rid] = ext_total
                self.vla_strides[rid] = tuple(dim_exts)  # the per-dim Horner multipliers
                return None
            inferred = False  # `T a[] = {...}` / `T m[][B] = {...}`: the initializer sizes it
            if len(st.type.array) == 1 and st.type.array[0] in (0, None) and st.init is None:
                # `T a[];` has no size and nothing to count (6.7.9p22) -- it was a zero-length array
                raise CLowerError("an array of unknown size needs an initializer")
            if len(st.type.array) > 1 or (
                len(st.type.array) == 1
                and st.type.array[0] in (0, None)
                and isinstance(st.init, (cast.AggInit, cast.StringLit))
            ):
                ct, inferred = self._inferred_array(st.type)
                if inferred and st.init is None:
                    raise CLowerError("an array of unknown size needs an initializer")
            else:
                ct = self._resolve_type(st.type)
            if st.static_storage:  # static storage: its constant image, in the decl (CF-STATICTAB)
                rid = self._static_storage(ct, st.name, None)
                if st.thread_storage:  # each thread's own object (CF-TLS)
                    self.thread_statics.add(rid)
                self.env[st.name] = (rid, ct)  # in scope in its own initializer, as in C
                ct, init = self._static_init(ct, st.init, inferred)
                self._resource(rid, ct, st.name)  # an inferred `[]` takes the walk's count
                self.statics[-1] = (rid, st.name, ct, init)
                self.env[st.name] = (rid, ct)
                return None
            rid = self._storage(ct, st.name)  # a mutable named local
            self.env[st.name] = (rid, ct)
            init = st.init
            if isinstance(init, cast.StringLit) and ct.kind == "array":
                if not self._init_char_array(ct, init):  # only a character array takes one
                    raise CLowerError("an array is initialized by a brace list or a string literal")
                init = cast.AggInit(entries=((None, init),))  # `char s[] = "ab"` is `{"ab"}`
            if isinstance(init, cast.AggInit):
                if ct.kind in ("struct", "union", "array"):  # `= { ... }`
                    n = self._agg_init(rid, ct, init, inferred=inferred)
                    if inferred:
                        self._resize_local(rid, self._sized_array(ct, n))
                else:  # a braced scalar `T x = {e}` is `T x = e`, and `{}` is zero (6.7.9p11)
                    expr = self._braced_scalar(init)
                    v = self._rvalue(expr) if expr is not None else self._zero_int("zi")
                    v = self._null_pointer(v, ct)
                    self._scalar_value(ct, v)  # `T x = {a}` of a struct (CF-STRUCTARITH)
                    self._emit("c.copy", Opcode.ADD, (v,), (rid,), **_object_write(ct))
            elif init is not None:
                if ct.kind == "array" and inferred and len(st.type.array) > 1:
                    # the parser models `T (*p)[N]` as `T p[][N]` (a parameter's row pointer); as a
                    # local it has no storage spelling here (and `T p[][N] = e` is not C)
                    raise CLowerError("a local pointer to an array is not supported")
                if ct.kind == "array":
                    raise CLowerError("an array is initialized by a brace list or a string literal")
                # a struct takes a brace list (above) or a value of its own type
                v = self._null_pointer(self._rvalue(init), ct)
                self._struct_value(ct, v, _STRUCT_INITIALIZED)
                self._emit("c.copy", Opcode.ADD, (v,), (rid,), **_object_write(ct))
                self._bind_extent(
                    rid, ct, st.name, init
                )  # §5.12: `T *p = malloc(N*sizeof(T))` -> extent N
        elif isinstance(st, cast.ExprStmt):
            if isinstance(st.expr, cast.Assign):  # a statement assignment: any lvalue form is fine
                self._assign(
                    st.expr, stmt=True
                )  # (the value is unused -- no named-local restriction)
            else:
                self._rvalue(st.expr)
        elif isinstance(st, cast.Return):
            rid = None if st.value is None else self._rvalue(st.value)
            if rid == _VOID_RID:  # `return void_call();` -> the call stmt is
                rid = None  # already emitted; a void function `return;`s
            elif rid is not None:  # `return 0;` from a function returning a pointer
                rid = self._null_pointer(rid, self._resolve_type(self.func.ret))
                # `return a;` of a struct from a function returning a scalar (CF-STRUCTARITH)
                self._scalar_value(self._resolve_type(self.func.ret), rid)
            self.block_stack[-1].append(ReturnNode(rid))
            if rid is not None:
                self.last_return = rid
        elif isinstance(st, cast.If):
            cond = self._rvalue(st.cond)  # condition claims -> current block
            node = IfNode(cond, self._block(st.then), self._block(st.els))
            self.block_stack[-1].append(node)
        elif isinstance(st, cast.While):
            cond_block: list = []
            self.block_stack.append(cond_block)
            cond = self._rvalue(st.cond)
            self.block_stack.pop()
            self.block_stack[-1].append(
                WhileNode(cond_block, cond, self._block(st.body), loop_id=self._next_loop_id())
            )
        elif isinstance(st, cast.Switch):
            disc = self._rvalue(st.disc)  # the discriminant, lowered once
            block: list = []
            self.block_stack.append(block)
            for item in st.body:
                if isinstance(item, cast.Case):
                    block.append(CaseLabel(item.value))
                elif isinstance(item, cast.Default):
                    block.append(DefaultLabel())
                else:
                    self._stmt(item)  # body stmts (break preserved as BreakNode)
            self.block_stack.pop()
            self.block_stack[-1].append(SwitchNode(disc, block))
        elif isinstance(st, cast.For):
            saved_env = dict(self.env)  # the for-init + body declarations are
            if st.init is not None:  # loop-scoped: a `for(unsigned i = ...)`
                self._stmt(st.init)  # must not leak `i` past the loop (where it
            cond_block2: list = []  # would shadow a same-named param / outer)
            self.block_stack.append(cond_block2)
            cond = self._rvalue(st.cond)
            self.block_stack.pop()
            body = self._block(st.body)
            step: list = []
            if st.step is not None:  # the step runs after body (the continue point)
                self.block_stack.append(step)
                self._stmt(st.step)
                self.block_stack.pop()
            self.block_stack[-1].append(
                WhileNode(cond_block2, cond, body, step=step, loop_id=self._next_loop_id())
            )
            self.env = saved_env  # pop the loop scope (restore outer bindings)
        elif isinstance(st, cast.DoWhile):  # body runs, then the cond is tested
            body = self._block(st.body)
            cond_block3: list = []
            self.block_stack.append(cond_block3)
            cond = self._rvalue(st.cond)
            self.block_stack.pop()
            self.block_stack[-1].append(
                WhileNode(cond_block3, cond, body, test_at_end=True, loop_id=self._next_loop_id())
            )
        elif isinstance(st, cast.Break):
            self.block_stack[-1].append(BreakNode())
        elif isinstance(st, cast.Continue):
            self.block_stack[-1].append(ContinueNode())
        elif isinstance(st, cast.Goto):
            self.block_stack[-1].append(GotoNode(st.label))
        elif isinstance(st, cast.ComputedGoto):  # `goto *p;` -- the target is lowered to a void*
            self.block_stack[-1].append(ComputedGotoNode(self._rvalue(st.target)))
        elif isinstance(st, cast.Label):
            self.block_stack[-1].append(LabelNode(st.name))
        elif isinstance(st, cast.AsmStmt):
            self._asm_stmt(st)
        else:
            raise CLowerError(f"statement {type(st).__name__} is beyond the L1–L6 subset")
        return None

    def lower(self) -> LoweredFunc:
        self.last_return = None
        _scan_mutations(
            self.func.body, self._mut_assigned, self._mut_body, self._mut_addr
        )  # §5.12 stability pre-pass
        self.env.update(self.genv)  # file-scope globals are in scope
        for rid, ct in self.genv.values():
            self.rtypes.setdefault(rid, ct)
        for p in self.func.params:  # params -> input resources (shadowing)
            if p.type.vla is not None:
                # a VLA parameter `T a[n]`: like any array param it decays to a pointer, BUT we recover the
                # runtime extent `n` (a prior in-scope integer parameter) and bind it via ptr_extent so the
                # param's `a[i]` promotes to masked -- a[BCIR_CHK(rid, i, n, "f:a")] -- exactly like a
                # malloc'd pointer with a recovered count. (Resolve the element with vla/array stripped:
                # _resolve_type raises on a VLA TypeRef otherwise -- params are processed in source order, so
                # `n` is already in env and a later param is not.)
                elem = self._resolve_type(replace(p.type, vla=None, array=()))
                if elem.size <= 0 or elem.kind == "array":
                    raise CLowerError(
                        "a VLA parameter of an incomplete / array element is not supported"
                    )
                ct = replace(pointer(elem, self.abi), shape=(0,))
                rid = self._new_rid()
                self._resource(rid, ct, p.name)
                self.env[p.name] = (rid, ct)
                self.params.append((p.name, rid, ct))
                n = p.type.vla  # bind ONLY for a STABLE prior integer param `n`
                if (
                    isinstance(n, cast.Name) and n.ident in self.env
                ):  # (unmutated, not address-taken --
                    n_rid, n_ct = self.env[n.ident]  #  the _bind_extent stable-Name contract,
                    if (
                        n_ct.kind == "scalar"
                        and not n_ct.is_float  #  so re-emitting `n` by name is sound)
                        and self._mut_body.get(n.ident, 0) == 0
                        and n.ident not in self._mut_addr
                    ):
                        self.ptr_extent[rid] = n_rid
                continue
            ct = self._resolve_type(p.type)
            if ct.kind == "array":  # an array param decays to a flat
                dims, elem = [], ct  # element pointer + a recorded shape
                while elem.kind == "array":
                    dims.append(elem.count)
                    elem = elem.of
                ct = replace(pointer(elem, self.abi), shape=tuple(dims))
            rid = self._new_rid()
            self._resource(rid, ct, p.name)
            self.env[p.name] = (rid, ct)
            self.params.append((p.name, rid, ct))
        for st in self.func.body:
            self._stmt(st)
        body = self.block_stack[0]
        claims = _flatten_block(body)
        touched = {r for c in claims for r in (tuple(c.rd) + tuple(c.wr))}
        if self.last_return is not None:
            touched.add(self.last_return)  # a BARE `return g;` reaches a global
            # with ZERO claims -- it must still be
            # pulled in and NAMED (else it renders
            # as the raw rid temp)
        for rid in touched & set(self.gres):  # pull in referenced global resources
            if rid == _IO_PORT_RID:
                # belt + suspenders: NEVER let a (guarded-impossible) global/string-band collision overwrite
                # the reserved `__ioport` MMIO resource with a RAM resource -- that would make a port access
                # alias normal memory (the isolation hole). The allocator guard already forbids producing
                # this rid, so this branch is unreachable; it pins the invariant locally regardless.
                continue
            self.resources[rid] = self.gres[rid]
        _order_device_claims(claims, self.resources)
        gnames = {rid: nm for nm, (rid, _ct) in self.genv.items() if rid in touched}
        gnames.update(self.str_globals)  # string globals render as inline literals
        gnames.update(self.func_globals)  # function-as-value globals render as the bare name
        m = Module(name=self.func.name)
        for rid in sorted(self.resources):
            m.add_resource(self.resources[rid])
        m.add_phase(Phase(phase_id=0, deps=(), claims=list(claims)))
        ret_ct = self._resolve_type(self.func.ret)
        return LoweredFunc(
            name=self.func.name,
            module=m,
            ret_type=ret_ct,
            params=self.params,
            return_rid=self.last_return,
            claims=list(claims),
            resources=dict(self.resources),
            rid_types=dict(self.rtypes),
            calls=list(self.calls),
            region=None,
            body=body,
            locals=list(self.locals),
            vla_locals=list(self.vla_locals),
            statics=list(self.statics),
            thread_statics=frozenset(self.thread_statics),
            globals_used=gnames,
            zero_init_locals=set(self.zero_init),
            variadic=self.func.variadic,
            tu_protos=dict(self.tu_used),
            reproducible=getattr(self.func, "reproducible", False),
            static_fn=getattr(self.func, "static_fn", False),
            ptr_extent=dict(self.ptr_extent),
            asm_meta=dict(self.asm_meta),
            target=self.abi.name,
        )


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
            parts.append(
                compose.Cond(
                    "c",
                    _block_region(node.then, functions, calls_iter),
                    _block_region(node.els, functions, calls_iter),
                )
            )
        elif isinstance(node, WhileNode):
            flush_run()
            _block_region(
                node.cond_block, functions, calls_iter
            )  # cond claims (cost folded in body)
            parts.append(_block_region(node.body + node.step, functions, calls_iter))
        elif isinstance(node, SwitchNode):
            flush_run()
            parts.append(
                _block_region(node.body, functions, calls_iter)
            )  # the clause bodies, in order
        elif isinstance(
            node,
            (
                ReturnNode,
                BreakNode,
                ContinueNode,
                GotoNode,
                ComputedGotoNode,
                LabelNode,
                CaseLabel,
                DefaultLabel,
            ),
        ):
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
            if mt.atomic and width:  # GCC and Clang reject it (C leaves it implementation-defined)
                raise CLowerError("a bit-field of `_Atomic` type is not supported")
            if mt.is_bitint and width:  # a `_BitInt(N)` BITFIELD `_BitInt(N) m : W` is now
                # first-class: it packs into the `_BitInt(N)` STORAGE UNIT (its `mt.size` = 1/2/4/8 bytes,
                # the Clang slot) with the same LSB-first packing as a standard-int bitfield of that storage
                # size -- VERIFIED byte-identical to Clang (size/align/bit-layout) for 2<=N<=64, 1<=W<=N.
                # The field value is typed `_BitInt(N)` (its W bits), and the emit prints `_BitInt(N) m : W;`.
                if not (
                    1 <= width <= mt.bit_width
                ):  # W>N is invalid C (Clang errors); W<1 not a named
                    raise CLowerError(  # field -> conservatively route to fallback.
                        "a `_BitInt` bitfield width outside 1..N is not supported"
                    )
            # a PLAIN `_BitInt(N)` member is also first-class: its `bitint` CType carries the Clang storage
            # width (1/2/4/8 bytes) and alignment, so AggregateBuilder lays it out exactly like the
            # equivalent power-of-two int (sizeof/offsetof match Clang); the member load/store uses that
            # storage width typed `_BitInt(N)`, and the emit prints the faithful spelling.
            b.members.append((mname, mt, width, malign))
        aggregates[tag] = b.build()

    # pre-scan every function's return type (forward references resolve too), so a call can be typed
    # by its callee: a void call emits a bare statement, a wide/float return keeps its real type.
    func_rets = {fn.name: _resolve_member_type(fn.ret, aggregates, abi) for fn in unit.funcs}
    # ... and their parameter types, so a null pointer constant a call passes to a pointer parameter is
    # a null pointer of that type, whichever of the caller and the callee is defined first (CF-NULLARG)
    func_params = {fn.name: _param_types(fn.params, aggregates, abi) for fn in unit.funcs}
    func_variadic = frozenset(fn.name for fn in unit.funcs if fn.variadic)
    # PROTOTYPED cross-TU callees (Phase 3 linking): a prototype whose definition is in this unit is
    # just a forward declaration (the definition wins); the rest resolve at LINK time. A parameter is
    # read as the definition binds it -- an array parameter is a pointer -- so the emit's `extern`
    # declaration spells `T *` for `T a[]` (the element type alone conflicted with the prototype).
    protos = {
        name: (
            _resolve_member_type(ret, aggregates, abi),
            tuple(_param_type(p, aggregates, abi) for p in params),
        )
        for name, (ret, params) in unit.protos.items()
        if name not in func_rets
    }
    # ... and whether each of a prototype's parameters points to `const` -- the base type's qualifier of a
    # pointer or of an array, which decays to one -- for the emit's `extern` declaration of the callee
    proto_consts = {
        name: tuple(
            "const" in p.quals and bool(p.ptr or p.array or p.vla is not None) for p in params
        )
        for name, (_ret, params) in unit.protos.items()
        if name not in func_rets
    }
    genv: dict[str, tuple] = {}  # file-scope globals: name -> (rid, CType)
    gres: dict[int, Resource] = {}
    gdecls: list = []  # the linkable emit's global surface
    # The file-scope initializers' lowerer: a function with no body, in whose scope a global's initializer
    # is lowered only to be folded -- by the fold a static local's initializer takes (`_const_value`), each
    # operation in C's own types -- and never kept. The globals declared so far are in its scope, a global
    # in its own initializer as in C (`sizeof g` folds; a read of one is no constant).
    ginit = _FuncLowerer(
        cast.Func(ret=cast.TypeRef(base="void"), name="", params=(), body=()),
        aggregates,
        base_rid=0,
        cid=[0],
        strctr=[0],
        func_rets=func_rets,
        abi=abi,
        protos=protos,
        func_params=func_params,
    )
    for gi, g in enumerate(unit.globals):
        # the declared type: an initialized scalar or struct keeps it -- it is one object, not a table of
        # its initializers (CF-GINIT) -- and an unsized array takes the extent its initializer's walk reaches
        # (CF-GBRACE): a brace list's, with elision and designators, or a character array's string literal
        ct = _resolve_member_type(g.type, aggregates, abi)
        rid = _check_band_rid(900000 + gi)  # guard the reserved I/O-port rid (belt + suspenders)
        ginit.env[g.name] = (rid, ct)  # in scope in its own initializer, as in C
        ginit.rtypes[rid] = ct
        strings = frozenset()
        if g.init is not None:
            ct, strings = ginit._file_scope_shape(ct, g.init)
        gres[rid] = Resource(
            rid=rid,
            # a volatile global, or a pointer to volatile storage, is a device region: the base of
            # every access through it (R3 holds the claims touching it to the MMIO domain)
            domain=Domain.MMIO if ct.touches_mmio else Domain.RAM,
            elem_bytes=(ct.of.size if ct.of else ct.size),
            # an array's elements; any other object is one element of its own type -- an initialized
            # scalar or struct is not a table of its initializers (CF-GINIT)
            shape=((ct.count or 1) if ct.kind == "array" else 1,),
            access="ro",
            data_gen=1,
            name=g.name,
        )
        genv[g.name] = (rid, ct)
        ginit.env[g.name] = (rid, ct)
        ginit.rtypes[rid] = ct
        gdecls.append(
            (
                g.name,
                ct,
                _file_scope_rendering(ginit, g, genv, strings),
                getattr(g, "extern_decl", False),
                getattr(g, "static_storage", False),
            )
        )

    functions: dict[str, LoweredFunc] = {}
    compose_functions: dict[str, compose.Function] = {}
    resources: dict[int, Resource] = dict(gres)
    cid = [1000]
    strctr = [0]  # unit-wide string-literal counter (unique rids)
    for idx, fn in enumerate(unit.funcs):
        lf = _FuncLowerer(
            fn,
            aggregates,
            base_rid=100 + idx * 1000,
            cid=cid,
            genv=genv,
            gres=gres,
            strctr=strctr,
            func_rets=func_rets,
            abi=abi,
            protos=protos,
            func_params=func_params,
            proto_consts=proto_consts,
            func_variadic=func_variadic,
            lowered=functions,
        ).lower()
        functions[fn.name] = lf
        resources.update(lf.resources)
    for lf in functions.values():  # regions need every callee's param rids
        lf.region = _region_for(lf, functions)
        compose_functions[lf.name] = compose.Function(lf.name, lf.region)
    entry = unit.funcs[-1].name if unit.funcs else ""
    return LoweredUnit(
        functions=functions,
        entry=entry,
        aggregates=aggregates,
        compose_functions=compose_functions,
        resources=resources,
        globals_decl=tuple(gdecls),
        thread_globals=frozenset(g.name for g in unit.globals if g.thread_storage),
        init_refs=frozenset(n for g in unit.globals for n in _names_in(g.init)),
        anon_spelling=dict(unit.anon_spelling),
    )


def _names_in(node) -> list:
    """Every identifier (`cast.Name`) inside an initializer expression tree."""
    out: list = []
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, cast.Name):
            out.append(n.ident)
        elif isinstance(n, (tuple, list)):
            stack.extend(n)
        elif dataclasses.is_dataclass(n) and not isinstance(n, type):
            stack.extend(getattr(n, f.name) for f in dataclasses.fields(n))
    return out


def _atomic_type(base: CType, abi) -> CType:
    """`base` `_Atomic`-qualified: the target ABI's atomic layout (`ctype_model.with_atomic`), the one
    answer for a local's, a parameter's, a member's and a global's type. An `_Atomic` struct or union is
    refused, as on the twin: its promoted layout would reach every declaration, copy and extent, and each
    access would have to be one atomic operation on the whole object."""
    if base.is_aggregate:
        raise CLowerError("an `_Atomic` struct or union is not supported")
    return with_atomic(base, abi=abi)


def _aggregate(aggregates: dict, tag: str, kind: str = "", pointee: bool = False) -> CType:
    """The laid-out struct or union `tag`. One the unit has not laid out at this point -- the struct
    being defined (`struct node *next`), one defined later, or one never defined (an opaque `struct fwd
    *`) -- is, as a pointer's pointee (`pointee`), its incomplete type: a pointer to it is complete
    (C11 6.2.5p22) and the lowering completes it where its members are used. Any other use needs its
    layout, and is refused as a lowering error the pipeline can route, never a bare KeyError."""
    if tag in aggregates:
        return aggregates[tag]
    if pointee and kind in ("struct", "union"):
        return incomplete_aggregate(kind, tag)
    raise CLowerError(f"the incomplete struct or union {tag!r} has no layout here")


def _resolve_member_type(tref: cast.TypeRef, aggregates: dict, abi=None) -> CType:
    abi = abi or HOST
    if tref.funcptr:  # a function-pointer member (dispatch table)
        ret = _resolve_member_type(tref.func_ret, aggregates, abi)
        params = tuple(_resolve_member_type(p, aggregates, abi) for p in tref.func_params)
        return funcptr(tref.base, ret, params, abi)
    if tref.aggregate:  # a pointer member may name a struct not laid out yet (or ever)
        base = _aggregate(aggregates, tref.base, tref.aggregate, pointee=tref.ptr > 0)
    elif tref.bit_width:  # C23 `_BitInt(N)` (e.g. a function return type)
        base = bitint(tref.bit_width, signed="unsigned" not in tref.base, abi=abi)
    else:
        base = scalar(tref.base, abi)
    if "volatile" in tref.quals:  # a volatile member / global, or a pointer to one: device storage
        base = with_volatile(base)
    if "_Atomic" in tref.quals:  # an _Atomic member / global: the ABI's atomic layout, as a local's
        base = _atomic_type(base, abi)
    t = base
    for _ in range(tref.ptr):
        t = pointer(t, abi)
    for dim in reversed(tref.array):
        t = array(t, dim)
    return t


def _param_type(tref: cast.TypeRef, aggregates: dict, abi=None) -> CType:
    """A parameter's type as its function binds it (`_FuncLowerer.lower`): an array -- a VLA `T a[n]`
    too -- decays to a pointer to its element, keeping the dimensions a subscript needs; any other type
    is its own. A prototype's parameters and a definition's, read ahead of the function's lowering."""
    abi = abi or HOST
    if tref.vla is not None:
        elem = _resolve_member_type(replace(tref, vla=None, array=()), aggregates, abi)
        return replace(pointer(elem, abi), shape=(0,))
    ct = _resolve_member_type(tref, aggregates, abi)
    if ct.kind == "array":
        dims, elem = [], ct
        while elem.kind == "array":
            dims.append(elem.count)
            elem = elem.of
        ct = replace(pointer(elem, abi), shape=tuple(dims))
    return ct


def _param_types(params: tuple, aggregates: dict, abi=None) -> tuple:
    """A definition's parameter types (`_param_type`), for its callers: a null pointer constant a call
    passes is typed by them (CF-NULLARG). None for a parameter only the function's own scope types
    (`typeof`) or that does not resolve here -- `_FuncLowerer.lower` decides it when the function lowers,
    and a None types nothing."""
    out = []
    for p in params:
        if p.type.typeof_var or p.type.typeof_expr is not None:
            out.append(None)
            continue
        try:
            out.append(_param_type(p.type, aggregates, abi))
        except (CLowerError, KeyError):  # an unknown or incomplete type, `va_list`
            out.append(None)
    return tuple(out)
