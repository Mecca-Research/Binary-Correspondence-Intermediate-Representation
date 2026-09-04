"""The C type model + layout (sizeof / alignof / field offsets) for the C-frontend subset.

Just enough of C23's type system for the L1–L4 ladder: fixed-width integers (`<stdint.h>` +
the core ints), `_Bool`/`char`, `void`, pointers, arrays, and `struct`/`union` aggregates. Layout
follows the usual C rule (each member aligned to its own alignment; aggregate size rounded up to its
alignment) so `offsetof`/`sizeof` match what Clang computes — which the behaviour-equivalence check
relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    shape: tuple = ()  # array dims of a decayed multi-dim array param (m[i][j])
    bit_width: int = 0  # a C23 `_BitInt(N)` type's EXACT width N (0 == a normal type; >0 ==
    #   `_BitInt(N)`). A distinct integer type that does NOT promote and does
    #   not canonicalize to a power-of-two width: `name` carries the verbatim
    #   spelling (`_BitInt(12)` / `unsigned _BitInt(12)`) so the emit prints it
    #   faithfully -- Clang then applies the N-bit semantics in both rails.

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
    def touches_mmio(self) -> bool:
        """A volatile object — or a pointer/array to one — accesses an MMIO region."""
        return self.volatile or bool(self.of and self.of.touches_mmio)

    def field(self, name: str):
        """Return (CType, byte_offset, bit_offset, bit_width) for a member; bit_width 0 == plain."""
        for entry in self.fields:
            if entry[0] == name:
                return entry[1], entry[2], entry[3], entry[4]
        raise KeyError(name)


def with_volatile(ct: CType, vol: bool = True) -> CType:
    from dataclasses import replace

    return replace(ct, volatile=vol) if vol else ct


def with_atomic(ct: CType, at: bool = True) -> CType:
    from dataclasses import replace

    return replace(ct, atomic=at) if at else ct


def scalar(name: str, abi=None) -> CType:
    """A scalar `CType`. With no `abi`, sizes are the host LP64 model (unchanged). With a `TargetABI`,
    the size-varying types follow the selected data model: `long` and the pointer-tracking integers
    take the ABI's widths, and `long double` takes the ABI's size *and* alignment (which can differ --
    12-byte/4-aligned on ILP32)."""
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
        return CType("scalar", name=name, size=size, align=max(1, size), signed=True)
    if name in _COMPLEX:  # a _Complex pair: element-aligned (align == size/2)
        if name == "long double _Complex" and abi is not None:
            return CType(
                "scalar",
                name=name,
                size=abi.long_double_size * 2,
                align=abi.long_double_align,
                signed=True,
            )
        size = _COMPLEX[name]
        return CType("scalar", name=name, size=size, align=max(1, size // 2), signed=True)
    if name not in _SCALAR:
        raise KeyError(f"unknown scalar type {name!r}")
    size, signed = _SCALAR[name]
    if abi is not None:
        size = abi.scalar_size(name, size)
    return CType("scalar", name=name, size=size, align=max(1, size), signed=signed)


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


def bitint(n: int, signed: bool) -> CType:
    """A C23 `_BitInt(N)` scalar CType. It carries the EXACT width N in `bit_width` and the verbatim
    spelling in `name` (`_BitInt(N)` / `unsigned _BitInt(N)`), so every emit site that prints the type
    spelling reproduces it faithfully -- and the value model keeps it OUT of the power-of-two integer
    canonicalization (`promote_int`/`usual_arith_int` short-circuit on `bit_width`), since `_BitInt(N)`
    does not undergo integer promotion. `size` is the storage slot (1/2/4/8 bytes) so a same-type
    store/load and `sizeof` match Clang; `signed` drives the spelling + any same-type signed arithmetic."""
    sz = _bitint_storage(n)
    spelling = f"_BitInt({n})" if signed else f"unsigned _BitInt({n})"
    return CType("scalar", name=spelling, size=sz, align=max(1, sz), signed=signed, bit_width=n)


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
        return a if (a.signed and b.signed) else bitint(a.bit_width, signed=False)
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


def int_literal_type(text: str) -> str:
    """The type of an integer constant (§6.4.4.1): from its `u`/`l`/`ll` suffix and magnitude, the
    first type in the suffix-permitted candidate list that can hold the value. Decimal literals only
    pick an unsigned type when `u`-suffixed; hex/octal literals may at any rank. Returns a canonical
    scalar name (`int` / `unsigned int` / `long` / ... )."""
    s = text.replace("'", "")  # strip C23 digit separators
    i = len(s)
    while i > 0 and s[i - 1] in "uUlL":
        i -= 1
    body, suf = s[:i], s[i:].lower()
    u, lrank = ("u" in suf), suf.count("l")  # lrank: 0 none / 1 long / 2 long long
    if body[:2] in ("0x", "0X"):
        val, decimal = int(body, 16), False
    elif body[:2] in ("0b", "0B"):
        val, decimal = int(body, 2), False
    elif len(body) > 1 and body[0] == "0":
        val, decimal = int(body, 8), False
    else:
        val, decimal = int(body or "0", 10), True
    INT, UINT = ("int", 4, True), ("unsigned int", 4, False)
    LONG, ULONG = ("long", 8, True), ("unsigned long", 8, False)
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


def funcptr(name: str, ret: CType, params: tuple = (), abi=None) -> CType:
    """A function-pointer type — pointer-sized (per the target ABI), carrying its return + parameter
    types so the emitter can reconstruct a call (``name`` is the typedef spelling, used verbatim)."""
    size = abi.pointer_size if abi is not None else PTR_SIZE
    return CType(
        "funcptr", name=name, size=size, align=size, signed=False, of=ret, params=tuple(params)
    )


def array(of: CType, count: int) -> CType:
    return CType("array", name="array", size=of.size * count, align=of.align, of=of, count=count)


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
                    unit_bits = mtype.size * 8  # field and does NOT raise the struct's alignment.
                    if width == 0:  # zero-width -> bump to the next unit boundary
                        if dbits % unit_bits:
                            dbits += unit_bits - (dbits % unit_bits)
                    elif self.packed:
                        dbits += width
                    else:
                        if (dbits % unit_bits) + width > unit_bits:
                            dbits += unit_bits - (dbits % unit_bits)
                        dbits += width
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
                for fn, fty, fbo, fbit, fbw in mtype.fields:
                    laid.append((fn, fty, off + fbo, fbit, fbw))
                continue
            if width and self.kind == "struct":
                unit_bits = mtype.size * 8
                if self.packed:  # packed: pack bit-by-bit, NO unit reservation
                    laid.append(
                        (mname, mtype, dbits // 8, dbits % 8, width)
                    )  # field at the running bit cursor
                    dbits += width  # its access unit spans only the bytes it covers
                else:  # natural: pack at the bit cursor
                    if (
                        dbits % unit_bits
                    ) + width > unit_bits:  # would cross a storage-unit boundary
                        dbits += unit_bits - (dbits % unit_bits)  # -> bump to the next one
                    unit_off = (dbits // unit_bits) * mtype.size
                    laid.append((mname, mtype, unit_off, dbits - unit_off * 8, width))
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
        )
