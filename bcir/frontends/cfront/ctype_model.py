"""The C type model + layout (sizeof / alignof / field offsets) for the C-frontend subset.

Just enough of C23's type system for the L1–L4 ladder: fixed-width integers (`<stdint.h>` +
the core ints), `_Bool`/`char`, `void`, pointers, arrays, and `struct`/`union` aggregates. Layout
follows the usual C rule (each member aligned to its own alignment; aggregate size rounded up to its
alignment) so `offsetof`/`sizeof` match what Clang computes — which the behaviour-equivalence check
relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .clex import int_literal_parts

# Scalar integer/base types -> (size in bytes, signed). C23 fixed-width names + the core set.
_SCALAR = {
    "void": (0, False),
    "_Bool": (1, False),
    "bool": (1, False),
    "char": (1, True),
    "signed char": (1, True),
    "unsigned char": (1, False),
    "short": (2, True),
    "unsigned short": (2, False),
    "int": (4, True),
    "unsigned int": (4, False),
    "unsigned": (4, False),
    "long": (8, True),
    "unsigned long": (8, False),
    "long long": (8, True),
    "unsigned long long": (8, False),
    "int8_t": (1, True),
    "uint8_t": (1, False),
    "int16_t": (2, True),
    "uint16_t": (2, False),
    "int32_t": (4, True),
    "uint32_t": (4, False),
    "int64_t": (8, True),
    "uint64_t": (8, False),
    "size_t": (8, False),
    "wchar_t": (4, True),  # the target's `wchar_t` integer (`scalar` resolves it per ABI)
    "intptr_t": (8, True),
    "uintptr_t": (8, False),
}
# Floating types -> size in bytes (Linux/Clang ABI: long double is 80-bit, 16-byte-aligned).
_FLOAT = {"float": 4, "double": 8, "long double": 16}
# Complex types (C99 _Complex) -> size in bytes (a pair of the element float; element-aligned, so the
# alignment is size/2). Modeled as a float *scalar* carrying the `is_complex` flag: it rides every
# existing float code path (binop result typing, load/store-as-itself, no truncation) and the emitter
# prints the `<elem> _Complex` spelling + native operators, so Clang lowers `*`/`/` (__mul/__div) the
# same way in the original and the re-emitted bcir_*.
_COMPLEX = {"float _Complex": 8, "double _Complex": 16, "long double _Complex": 32}
PTR_SIZE = 8


@dataclass(frozen=True)
class CType:
    """A resolved C type. ``kind`` in {scalar, pointer, array, struct, union}."""

    kind: str
    name: str = ""  # scalar/aggregate name
    size: int = 0
    align: int = 0
    signed: bool = False
    of: "CType | None" = None  # element type (pointer/array), or return type (funcptr)
    count: int = 0  # array length
    fields: tuple = ()  # ((name, CType, byte_off, bit_off, bit_width), ...)
    volatile: bool = False  # a volatile-qualified type -> an MMIO resource
    atomic: bool = False  # an _Atomic-qualified type (C11/C23 atomics)
    packed: bool = False  # an __attribute__((packed)) struct/union (no padding; a bitfield
    #   packs bit-by-bit and its access unit spans only the bytes it covers)
    params: tuple = ()  # parameter CTypes (funcptr only) — for faithful emit
    variadic: bool = False  # funcptr only: the function takes a trailing `...` (CF-FPRET)
    fquals: tuple = ()  # funcptr only: the qualifiers its return and parameters keep below their top level --
    #   (return levels, (parameter levels, ...)), each level a tuple of `const` / `restrict` from what it points
    #   to out (`lower._qual_sig`); () when none is qualified. Part of the function type's identity, and
    #   spelled wherever the type is (CF-QUALS)
    shape: tuple = ()  # array dims of a decayed multi-dim array param (m[i][j])
    bit_width: int = 0  # a C23 `_BitInt(N)` type's EXACT width N (0 == a normal type; >0 ==
    #   `_BitInt(N)`). A distinct integer type that does NOT promote and does
    #   not canonicalize to a power-of-two width: `name` carries the verbatim
    #   spelling (`_BitInt(12)` / `unsigned _BitInt(12)`) so the emit prints it
    #   faithfully -- Clang then applies the N-bit semantics in both rails.
    natural: tuple = ()  # an `_Atomic` type's own (size, align) before the ABI's atomic promotion
    #   (`with_atomic`), which `unqualified` restores: the value read from an
    #   `_Atomic float _Complex` is a `float _Complex`, aligned to 4, not 8
    incomplete: bool = (
        False  # a struct/union named before (or without) its definition -- the pointee
    )
    #   of `struct node *next` inside `struct node`, of `struct fwd *` -- which a pointer to it
    #   does not need laid out (C11 6.2.5p22); the lowering completes it by tag where its members
    #   are used (`_FuncLowerer._complete`)
    anon: tuple = ()  # a struct/union's ANONYMOUS members, in declaration order: (first, n, CType,
    #   byte_off) -- `fields[first:first+n]` are the member's promoted leaves. An initializer
    #   takes the member as ONE subobject of its own type (C11 6.7.2.1p13), so an anonymous
    #   union takes one positional value, as in C, where its flattened leaves would take several

    @property
    def is_bitint(self) -> bool:
        return self.kind == "scalar" and self.bit_width > 0

    @property
    def is_integer(self) -> bool:
        return (
            self.kind == "scalar"
            and self.name != "void"
            and self.name not in _FLOAT
            and self.name not in _COMPLEX
        )

    @property
    def is_float(self) -> bool:  # complex rides the float paths (so is_float == True)
        return self.kind == "scalar" and (self.name in _FLOAT or self.name in _COMPLEX)

    @property
    def is_complex(self) -> bool:
        return self.kind == "scalar" and self.name in _COMPLEX

    @property
    def is_aggregate(self) -> bool:
        return self.kind in ("struct", "union")

    @property
    def volatile_storage(self) -> bool:
        """The object itself holds volatile storage: it is volatile, an array of such, or an
        aggregate with such a member at any depth. A pointer member holds an address -- what it
        points at is not the aggregate's storage."""
        if self.volatile:
            return True
        if self.kind == "array":
            return bool(self.of and self.of.volatile_storage)
        if self.is_aggregate:
            return any(f[1].volatile_storage for f in self.fields)
        return False

    @property
    def touches_mmio(self) -> bool:
        """A resource of this type is an MMIO region: it holds volatile storage, or it points (through
        any number of pointers) at volatile storage. The twin decides the same (`bcir_cfront.c`,
        `ty_mmio`)."""
        if self.volatile_storage:
            return True
        return self.kind == "pointer" and bool(self.of and self.of.touches_mmio)

    def field(self, name: str):
        """Return (CType, byte_offset, bit_offset, bit_width) for a member; bit_width 0 == plain."""
        for entry in self.fields:
            if entry[0] == name:
                return entry[1], entry[2], entry[3], entry[4]
        raise KeyError(name)


def incomplete_aggregate(kind: str, tag: str) -> CType:
    """The incomplete struct or union `tag` -- known by name, not (yet) laid out -- as a pointer's pointee."""
    return CType(kind, name=tag, size=0, align=1, incomplete=True)


def with_volatile(ct: CType, vol: bool = True) -> CType:
    from dataclasses import replace

    return replace(ct, volatile=vol) if vol else ct


def unqualified(ct: CType) -> CType:
    """The type of the VALUE an lvalue of type `ct` yields: lvalue conversion drops the qualifiers and
    the atomicity (C23 6.3.2.1p2), so a value read from a volatile register is an ordinary value, and one
    read from an `_Atomic` object has the non-atomic type -- its layout too, which the ABI's atomic
    promotion widened (`with_atomic`). The twin keeps an `_Atomic` type's own layout and promotes only
    where it lays out storage, so clearing the flag is its whole answer (`bcir_cfront.c`, `store_conv`)."""
    from dataclasses import replace

    if ct.atomic:
        size, align = ct.natural or (ct.size, ct.align)
        return replace(ct, volatile=False, atomic=False, size=size, align=align, natural=())
    return replace(ct, volatile=False) if ct.volatile else ct


def qualified(ct: CType, vol: bool) -> CType:
    """`ct` as a member of a volatile aggregate sees it: volatile itself, and an array member's
    elements volatile too (C23 6.5.2.3p3, the member designator takes the aggregate's qualifiers)."""
    from dataclasses import replace

    if not vol:
        return ct
    if ct.kind == "array" and ct.of is not None:
        return replace(ct, of=qualified(ct.of, True))
    return ct if ct.volatile else replace(ct, volatile=True)


def with_atomic(ct: CType, at: bool = True, abi=None) -> CType:
    """`ct` as `_Atomic` qualifies it, laid out by the target ABI's atomic promotion: an `_Atomic`
    type no wider than `abi.atomic_promote_size` (Clang's `MaxAtomicPromoteWidth`: 16 bytes on the
    64-bit targets, 8 on i386) rounds its size up to a power of two and aligns to that size, so an
    `_Atomic float _Complex` aligns to 8, an `_Atomic double _Complex` to 16 on a 64-bit target, and a
    3-byte `_Atomic struct` occupies 4. A wider one keeps its own layout. The twin asks the same rule
    (`bcir_cfront.c` `atomic_layout`)."""
    from dataclasses import replace

    if not at:
        return ct
    width = abi.atomic_promote_size if abi is not None else 16  # no abi: the host LP64 model
    size, align = ct.size, ct.align
    if ct.kind != "array" and 0 < size <= width:
        size = 1 << (size - 1).bit_length()
        align = size
    natural = ct.natural or (ct.size, ct.align)  # `_Atomic` twice: the first's own layout
    return replace(ct, atomic=True, size=size, align=align, natural=natural)


def scalar_align(size: int, abi=None) -> int:
    """The ABI alignment of a scalar (or a complex type's element) of `size` bytes: its size, except an
    8-byte one, which takes the ABI's `eight_byte_align` (4 on i386, where `double`, `long long` and
    `int64_t` align to 4 while their size stays 8). The twin asks the same rule (`scalar_align`)."""
    if size == 8 and abi is not None:
        return abi.eight_byte_align
    return max(1, size)


def scalar(name: str, abi=None) -> CType:
    """A scalar `CType`. With no `abi`, sizes are the host LP64 model (unchanged). With a `TargetABI`,
    the size-varying types follow the selected data model: `long` and the pointer-tracking integers
    take the ABI's widths, `long double` takes the ABI's size *and* alignment (which can differ --
    12-byte/4-aligned on ILP32), and an 8-byte scalar the ABI's `eight_byte_align`."""
    if name in _FLOAT:
        if name == "long double" and abi is not None:
            return CType(
                "scalar",
                name=name,
                size=abi.long_double_size,
                align=abi.long_double_align,
                signed=True,
            )
        size = _FLOAT[name]
        return CType("scalar", name=name, size=size, align=scalar_align(size, abi), signed=True)
    # a _Complex pair: aligned as its element (a `double _Complex` as a double)
    if name in _COMPLEX:
        if name == "long double _Complex" and abi is not None:
            return CType(
                "scalar",
                name=name,
                size=abi.long_double_size * 2,
                align=abi.long_double_align,
                signed=True,
            )
        size = _COMPLEX[name]
        return CType(
            "scalar", name=name, size=size, align=scalar_align(size // 2, abi), signed=True
        )
    if name == "wchar_t":  # a typedef of the target's integer type (`abi.wchar_type`), so it
        name = (
            abi.wchar_type if abi is not None else "int"
        )  # lowers, emits and converts as that type
    if name not in _SCALAR:
        raise KeyError(f"unknown scalar type {name!r}")
    size, signed = _SCALAR[name]
    if abi is not None:
        size = abi.scalar_size(name, size)
    return CType("scalar", name=name, size=size, align=scalar_align(size, abi), signed=signed)


def _bitint_storage(n: int) -> int:
    """The storage-unit byte width of a `_BitInt(N)`: the smallest standard integer width (1/2/4/8 bytes)
    that holds N bits. (Clang lays a `_BitInt(N)`, 2<=N<=64, in 1/2/4/8 bytes -- e.g. `_BitInt(12)` is
    2-byte/2-aligned.) The EXACT width is tracked separately in `bit_width`; this only sizes the slot so
    a same-width store/load round-trips and `sizeof`/layout match Clang. (N>64 is out of the supported
    subset -- see cparse; it never reaches here.)"""
    bytes_ = (n + 7) // 8
    unit = 1
    while unit < bytes_:
        unit *= 2
    return unit


def bitint(n: int, signed: bool, abi=None) -> CType:
    """A C23 `_BitInt(N)` scalar CType. It carries the EXACT width N in `bit_width` and the verbatim
    spelling in `name` (`_BitInt(N)` / `unsigned _BitInt(N)`), so every emit site that prints the type
    spelling reproduces it faithfully -- and the value model keeps it OUT of the power-of-two integer
    canonicalization (`promote_int`/`usual_arith_int` short-circuit on `bit_width`), since `_BitInt(N)`
    does not undergo integer promotion. `size` is the storage slot (1/2/4/8 bytes) so a same-type
    store/load and `sizeof` match Clang, and it aligns as a scalar of that size (an 8-byte slot to 4 on
    i386, like `long long`); `signed` drives the spelling + any same-type signed arithmetic."""
    sz = _bitint_storage(n)
    spelling = f"_BitInt({n})" if signed else f"unsigned _BitInt({n})"
    return CType(
        "scalar", name=spelling, size=sz, align=scalar_align(sz, abi), signed=signed, bit_width=n
    )


def is_scalar_name(name: str) -> bool:
    return name in _SCALAR or name in _FLOAT or name in _COMPLEX


# --- integer promotions + usual arithmetic conversions (C23 §6.3.1.1 / §6.3.1.8) -----------------
# The value model needs only (width, signedness): the observable result of integer arithmetic is
# fixed by those two, so types that share a width (long / long long; int / int32_t) collapse onto one
# canonical fixed-width type. The actual computation is delegated to the emitted C / resident backend.
_INT_CANON = {
    (1, True): "int8_t",
    (1, False): "uint8_t",
    (2, True): "int16_t",
    (2, False): "uint16_t",
    (4, True): "int32_t",
    (4, False): "uint32_t",
    (8, True): "int64_t",
    (8, False): "uint64_t",
}


def int_type(size: int, signed: bool, abi=None) -> CType:
    """The canonical fixed-width integer CType for a (size, signedness) pair."""
    return scalar(_INT_CANON.get((size, bool(signed)), "int32_t"), abi)


class BitIntMix(Exception):
    """A `_BitInt(N)` operand combination whose C23 result type is OUTSIDE the modeled first-class subset:
    a `_BitInt` mixed with a standard integer (or a different `_BitInt`) whose usual-arithmetic-conversion
    result is a STANDARD integer type, not a `_BitInt`. The first-class subset (see `bitint_arith_result`)
    only carries results that are themselves a `_BitInt(N)` -- where the bit-precise operand wins the C23
    rank, so the result spelling is one the emit already reproduces faithfully + Clang-verified. Where the
    result would be a standard type, the long/long-long width-collapse in the value model loses the exact
    spelling, so the lowering catches this and routes to fallback (a `CLowerError`) rather than emit a
    result type that could diverge from Clang's `_Generic` view."""


def bitint_arith_result(a: CType, b: CType) -> CType | None:
    """The C23 6.3.1.8 usual-arithmetic-conversions common type for two integer operands when AT LEAST ONE
    is a `_BitInt(N)`, RESTRICTED to the first-class subset: returns the result CType iff it is itself a
    `_BitInt(N)` (the bit-precise operand wins the C23 rank); returns None iff one or both operands are
    NOT a `_BitInt` AND the standard sub-int operand would need integer promotion before the comparison
    (caller already promoted standard operands); raises `BitIntMix` iff the modeled result is a STANDARD
    integer type (out of the conservative subset). Operands here are assumed already integer-promoted, so
    a standard operand is `int`/`unsigned`/`long`/.../`long long` (rank fixed by width + standard sub-rank).

    The C23 rank (6.2.5 + 6.3.1.1, as VERIFIED against Clang 18): a `_BitInt(N)` has rank GREATER than any
    standard/extended integer of LESS width, LESS than any standard integer of GREATER width, and for the
    SAME width the standard integer has the greater rank. So the result is a `_BitInt(N)` exactly when the
    bit-precise operand's width strictly exceeds the other operand's width (a tie goes to the standard, or
    -- two `_BitInt`s -- to the wider, equal width combining signedness). When a `_BitInt` wins, the result
    is that `_BitInt(N)` with ITS OWN signedness (a strictly-wider type represents the narrower one, so the
    sign rules never flip the winner)."""
    if not (a.is_bitint or b.is_bitint):
        return None
    # a `_BitInt` mixed with a FLOAT converts to the float -> the result is a FLOATING type, NOT a `_BitInt`
    # (and the integer rank rules below do not apply). Route to fallback (out of the first-class subset).
    if a.is_float or b.is_float:
        raise BitIntMix("`_BitInt` mixed with a floating type (result is not a `_BitInt`)")
    if a.is_bitint and b.is_bitint:
        if a.bit_width > b.bit_width:
            return a
        if b.bit_width > a.bit_width:
            return b
        # equal width: combine signedness (unsigned iff either unsigned); a same-type pair stays itself.
        # The unsigned result keeps the operands' layout (their slot's ABI alignment).
        if a.signed and b.signed:
            return a
        from dataclasses import replace

        return replace(bitint(a.bit_width, signed=False), align=a.align)
    bi = a if a.is_bitint else b
    std = b if a.is_bitint else a
    # `std` is already integer-promoted (>= int), so its width is its rank-width; std_width 4/8 bytes here.
    if bi.bit_width > std.size * 8:
        return bi  # the `_BitInt` strictly wider -> it wins, own sign
    raise BitIntMix("`_BitInt` arithmetic whose C23 result is a standard integer type")


def promote_int(t: CType, abi=None) -> CType:
    """Integer promotion (§6.3.1.1): a type of rank lower than `int` promotes to `int` (which holds
    every value of any sub-int type), so char/short/_Bool/bitfield operands become signed int. A C23
    `_BitInt(N)` does NOT promote (§6.3.1.1p2 excludes it) -- it stays `_BitInt(N)`, so its spelling and
    exact width survive a unary `-`/`~` and a shift's left operand."""
    if t.is_bitint:
        return t
    if not t.is_integer:
        return t
    return int_type(4, True, abi) if t.size < 4 else t


def usual_arith_int(a: CType, b: CType, abi=None) -> CType:
    """Usual arithmetic conversions (§6.3.1.8) for two integer operands -> their common type. After
    promoting both, the wider width wins (carrying the wider operand's signedness, since a strictly
    wider signed type represents every value of the narrower one); on equal width the result is
    unsigned iff either operand is unsigned.

    A C23 `_BitInt(N)` does NOT promote, so the conversions follow the bit-precise rank rules (6.2.5 +
    6.3.1.8): the first-class subset carries the result iff it is itself a `_BitInt(N)` (the bit-precise
    operand wins the C23 rank). Same-type `_BitInt(N)` op `_BitInt(N)` stays `_BitInt(N)`; a wider `_BitInt`
    mixed with a narrower standard int (or a narrower `_BitInt`) yields the wider `_BitInt`. A mix whose
    C23 result would be a STANDARD integer type raises `BitIntMix` (out of the conservative subset) -- the
    lowering catches it (handling the `_BitInt` op integer-CONSTANT case there, where the literal context
    is known) and otherwise routes to fallback rather than emit a width-collapsed (mis-typed) result."""
    if a.is_bitint or b.is_bitint:
        # promote the standard operand (a `_BitInt` does not promote) before the rank comparison, so a
        # `char`/`short` operand is compared at its post-promotion `int` width (its real rank-width).
        r = bitint_arith_result(promote_int(a, abi), promote_int(b, abi))
        if r is not None:
            return r
        raise BitIntMix("unsupported `_BitInt` operand combination")
    pa, pb = promote_int(a, abi), promote_int(b, abi)
    if pa.size != pb.size:
        wider = pa if pa.size > pb.size else pb
        return int_type(wider.size, wider.signed, abi)
    return int_type(pa.size, pa.signed and pb.signed, abi)


def int_literal_type(text: str, long_size: int = 8) -> str:
    """The type of an integer constant (§6.4.4.1): from its `u`/`l`/`ll` suffix and magnitude, the
    first type in the suffix-permitted candidate list that can hold the value. Decimal literals only
    pick an unsigned type when `u`-suffixed; hex/octal literals may at any rank. Returns a canonical
    scalar name (`int` / `unsigned int` / `long` / ... ). `long_size`: the target's `long`, in bytes --
    where it is 4 (LLP64, ILP32) `0xFFFFFFFFL` is an `unsigned long`, as the twin's `lit_int_type` has it,
    and a value past it the next type the list gives (CF-ENUMFOLD)."""
    val, decimal, suf = int_literal_parts(text)  # the value, as the lexer checked it
    suf = suf.lower()
    u, lrank = ("u" in suf), suf.count("l")  # lrank: 0 none / 1 long / 2 long long
    INT, UINT = ("int", 4, True), ("unsigned int", 4, False)
    LONG, ULONG = ("long", long_size, True), ("unsigned long", long_size, False)
    LL, ULL = ("long long", 8, True), ("unsigned long long", 8, False)
    if u:
        cands = {0: [UINT, ULONG, ULL], 1: [ULONG, ULL], 2: [ULL]}[lrank]
    elif decimal:
        cands = {0: [INT, LONG, LL], 1: [LONG, LL], 2: [LL]}[lrank]
    else:  # hex/octal unsuffixed: unsigned allowed
        cands = {0: [INT, UINT, LONG, ULONG, LL, ULL], 1: [LONG, ULONG, LL, ULL], 2: [LL, ULL]}[
            lrank
        ]
    for name, size, signed in cands:
        if val <= (1 << (size * 8 - (1 if signed else 0))) - 1:
            return name
    return cands[-1][0]


def pointer(of: CType, abi=None) -> CType:
    size = abi.pointer_size if abi is not None else PTR_SIZE
    return CType("pointer", name="ptr", size=size, align=size, signed=False, of=of)


def valist(abi=None) -> CType:
    """The `va_list` type (<stdarg.h> variadic cursor). A distinct opaque kind -- NOT a scalar, so it is
    neither integer nor float (no arithmetic conversions apply); the emitter renders it `va_list`."""
    size = abi.pointer_size if abi is not None else PTR_SIZE
    return CType("valist", name="va_list", size=size, align=size)


def funcptr(
    name: str, ret: CType, params: tuple = (), abi=None, variadic: bool = False, fquals: tuple = ()
) -> CType:
    """A function-pointer type — pointer-sized (per the target ABI), carrying its return + parameter
    types so the emitter can reconstruct a call (``name`` is the typedef spelling, used verbatim),
    whether the function is variadic (CF-FPRET), and the qualifiers of its return and parameters
    (CF-QUALS)."""
    size = abi.pointer_size if abi is not None else PTR_SIZE
    return CType(
        "funcptr",
        name=name,
        size=size,
        align=size,
        signed=False,
        of=ret,
        params=tuple(params),
        variadic=variadic,
        fquals=fquals,
    )


def array(of: CType, count: int) -> CType:
    return CType("array", name="array", size=of.size * count, align=of.align, of=of, count=count)


def bitfield_start(dbits: int, width: int, mtype: CType) -> int:
    """Where Clang (the Itanium layout) starts a non-packed bitfield of `width` bits and type `mtype` at
    the bit cursor `dbits`: at the cursor, unless the field would overflow a storage unit of its type's
    SIZE that begins at an ALIGNMENT boundary -- then at the next alignment boundary. When the alignment
    equals the size (every type but i386's 8-byte integers) this is the storage-unit rule: a field never
    crosses a unit boundary. On i386 a `long long x : 40` after 40 bits stays at bit 40 (40 % 32 + 40
    fits in 64), where x86-64 bumps it to 64. The twin asks the same rule (`bitfield_start`)."""
    field_align, unit_bits = mtype.align * 8, mtype.size * 8
    if (dbits % field_align) + width > unit_bits:
        dbits += field_align - (dbits % field_align)
    return dbits


def narrow_bitfield(ct: CType) -> bool:
    """A bitfield of type `ct` is accessed over only the bytes it spans, not a storage unit of its
    type's size: its type aligns below its size (i386's 8-byte integers), so a unit starting at the
    field could run past the struct's end. The layout records such a field at its first byte
    (`AggregateBuilder`), as it does a packed one."""
    return ct.align < ct.size


@dataclass
class AggregateBuilder:
    """Compute a struct/union layout the way Clang does (natural alignment). Bitfields pack LSB-first
    into storage units of their declared type (the little-endian Clang rule the equivalence check
    relies on). ``packed`` drops inter-member + tail padding (member alignment forced to 1);
    ``force_align`` raises the aggregate alignment (`aligned(N)`/`alignas`)."""

    kind: str
    name: str
    members: list = field(
        default_factory=list
    )  # (name, CType, bit_width, req_align)  bit_width 0 == plain;
    packed: bool = False  # req_align 0 == natural (else over-aligns the member)
    force_align: int = 0

    def build(self) -> CType:
        # A bit cursor (`dbits` = data size in bits) drives layout so a bitfield can pack at the
        # current position the way Clang/Itanium do: a bitfield following a sub-word member (e.g.
        # `short m0; unsigned m1:1;`) lands in the SAME storage unit as long as it does not cross an
        # `alignof(T)` boundary -- it is NOT bumped to a fresh, type-aligned unit. Packed structs keep
        # the older byte-granular unit model (no packed-bitfield fixtures exercise the difference).
        dbits = 0
        align = 1
        laid: list = []
        anon: list = []  # the anonymous members' leaf groups (`CType.anon`)
        bf_unit_off = None  # byte offset of the active bitfield storage unit (packed path)
        bf_bits = 0  # bits already used in it
        bf_unit_size = 0

        def malign(mtype: CType, req: int) -> int:
            # `_Alignas(N)`/`aligned(N)` over-aligns a member (and survives `packed`); otherwise the
            # member takes its natural alignment, which `packed` drops to 1.
            return max(req, 1 if self.packed else mtype.align)

        for mname, mtype, width, req in self.members:
            ma = malign(mtype, req)
            if (
                mname == "" and mtype.kind == "scalar"
            ):  # an UNNAMED (`int :3`) or ZERO-WIDTH (`int :0`)
                if self.kind == "struct":  # bitfield: positions the cursor but is NOT a
                    field_align = mtype.align * 8  # field, nor raises the struct's alignment.
                    if width == 0:  # zero-width -> the next boundary of its type's ALIGNMENT
                        if dbits % field_align:  # (packed or not, as Clang does)
                            dbits += field_align - (dbits % field_align)
                    elif self.packed:
                        dbits += width
                    else:
                        dbits = bitfield_start(dbits, width, mtype) + width
                continue
            align = max(align, ma)
            if mname == "":  # an ANONYMOUS struct/union member: it occupies
                if self.kind == "union":  # space as a unit, but its leaf fields PROMOTE
                    off = 0  # into this aggregate's namespace at shifted
                else:  # offsets (so `p->x` resolves directly).
                    a8 = ma * 8
                    if dbits % a8:
                        dbits += a8 - (dbits % a8)
                    off = dbits // 8
                    dbits += mtype.size * 8
                if mtype.fields:  # one initializable subobject over its promoted leaves
                    anon.append((len(laid), len(mtype.fields), mtype, off))
                for fn, fty, fbo, fbit, fbw in mtype.fields:
                    laid.append((fn, fty, off + fbo, fbit, fbw))
                continue
            if width and self.kind == "struct":
                if self.packed:  # packed: pack bit-by-bit, NO unit reservation
                    laid.append(
                        (mname, mtype, dbits // 8, dbits % 8, width)
                    )  # field at the running bit cursor
                    dbits += width  # its access unit spans only the bytes it covers
                else:  # natural: pack at the bit cursor unless it would overflow its unit
                    dbits = bitfield_start(dbits, width, mtype)
                    if mtype.align == mtype.size:  # the size-aligned storage unit holding it
                        unit_off = (dbits // (mtype.size * 8)) * mtype.size
                        laid.append((mname, mtype, unit_off, dbits - unit_off * 8, width))
                    else:  # an under-aligned type (i386 `long long`): its unit can run past the
                        # struct's end (`struct { char c; long long x : 8; }` is 4 bytes), so the
                        # field is recorded at its first byte and accessed over the bytes it spans
                        # (`narrow_bitfield`), as a packed one is
                        laid.append((mname, mtype, dbits // 8, dbits % 8, width))
                    dbits += width
            elif self.kind == "union":
                laid.append((mname, mtype, 0, 0, width))
            else:
                bf_unit_off, bf_bits, bf_unit_size = None, 0, 0  # a plain member flushes the unit
                a8 = ma * 8
                if dbits % a8:
                    dbits += a8 - (dbits % a8)
                laid.append((mname, mtype, dbits // 8, 0, width))
                dbits += mtype.size * 8
        align = max(align, self.force_align)
        size = (
            max((m[1].size for m in self.members), default=0)
            if self.kind == "union"
            else (dbits + 7) // 8
        )
        if align and size % align:
            size += align - (size % align)
        return CType(
            self.kind,
            name=self.name,
            size=size,
            align=max(1, align),
            fields=tuple(laid),
            packed=self.packed,
            anon=tuple(anon),
        )
