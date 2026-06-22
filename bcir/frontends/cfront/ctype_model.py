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
    "_Bool": (1, False), "bool": (1, False),
    "char": (1, True), "signed char": (1, True), "unsigned char": (1, False),
    "short": (2, True), "unsigned short": (2, False),
    "int": (4, True), "unsigned int": (4, False), "unsigned": (4, False),
    "long": (8, True), "unsigned long": (8, False),
    "long long": (8, True), "unsigned long long": (8, False),
    "int8_t": (1, True), "uint8_t": (1, False),
    "int16_t": (2, True), "uint16_t": (2, False),
    "int32_t": (4, True), "uint32_t": (4, False),
    "int64_t": (8, True), "uint64_t": (8, False),
    "size_t": (8, False), "intptr_t": (8, True), "uintptr_t": (8, False),
}
# Floating types -> size in bytes (Linux/Clang ABI: long double is 80-bit, 16-byte-aligned).
_FLOAT = {"float": 4, "double": 8, "long double": 16}
PTR_SIZE = 8


@dataclass(frozen=True)
class CType:
    """A resolved C type. ``kind`` in {scalar, pointer, array, struct, union}."""
    kind: str
    name: str = ""                       # scalar/aggregate name
    size: int = 0
    align: int = 0
    signed: bool = False
    of: "CType | None" = None            # element type (pointer/array), or return type (funcptr)
    count: int = 0                       # array length
    fields: tuple = ()                   # ((name, CType, byte_off, bit_off, bit_width), ...)
    volatile: bool = False               # a volatile-qualified type -> an MMIO resource
    atomic: bool = False                 # an _Atomic-qualified type (C11/C23 atomics)
    params: tuple = ()                   # parameter CTypes (funcptr only) — for faithful emit
    shape: tuple = ()                    # array dims of a decayed multi-dim array param (m[i][j])

    @property
    def is_integer(self) -> bool:
        return self.kind == "scalar" and self.name != "void" and self.name not in _FLOAT

    @property
    def is_float(self) -> bool:
        return self.kind == "scalar" and self.name in _FLOAT

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
            return CType("scalar", name=name, size=abi.long_double_size,
                         align=abi.long_double_align, signed=True)
        size = _FLOAT[name]
        return CType("scalar", name=name, size=size, align=max(1, size), signed=True)
    if name not in _SCALAR:
        raise KeyError(f"unknown scalar type {name!r}")
    size, signed = _SCALAR[name]
    if abi is not None:
        size = abi.scalar_size(name, size)
    return CType("scalar", name=name, size=size, align=max(1, size), signed=signed)


def is_scalar_name(name: str) -> bool:
    return name in _SCALAR or name in _FLOAT


# --- integer promotions + usual arithmetic conversions (C23 §6.3.1.1 / §6.3.1.8) -----------------
# The value model needs only (width, signedness): the observable result of integer arithmetic is
# fixed by those two, so types that share a width (long / long long; int / int32_t) collapse onto one
# canonical fixed-width type. The actual computation is delegated to the emitted C / resident backend.
_INT_CANON = {(1, True): "int8_t", (1, False): "uint8_t", (2, True): "int16_t", (2, False): "uint16_t",
              (4, True): "int32_t", (4, False): "uint32_t", (8, True): "int64_t", (8, False): "uint64_t"}


def int_type(size: int, signed: bool, abi=None) -> CType:
    """The canonical fixed-width integer CType for a (size, signedness) pair."""
    return scalar(_INT_CANON.get((size, bool(signed)), "int32_t"), abi)


def promote_int(t: CType, abi=None) -> CType:
    """Integer promotion (§6.3.1.1): a type of rank lower than `int` promotes to `int` (which holds
    every value of any sub-int type), so char/short/_Bool/bitfield operands become signed int."""
    if not t.is_integer:
        return t
    return int_type(4, True, abi) if t.size < 4 else t


def usual_arith_int(a: CType, b: CType, abi=None) -> CType:
    """Usual arithmetic conversions (§6.3.1.8) for two integer operands -> their common type. After
    promoting both, the wider width wins (carrying the wider operand's signedness, since a strictly
    wider signed type represents every value of the narrower one); on equal width the result is
    unsigned iff either operand is unsigned."""
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
    s = text.replace("'", "")                                  # strip C23 digit separators
    i = len(s)
    while i > 0 and s[i - 1] in "uUlL":
        i -= 1
    body, suf = s[:i], s[i:].lower()
    u, lrank = ("u" in suf), suf.count("l")                    # lrank: 0 none / 1 long / 2 long long
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
    else:                                                      # hex/octal unsuffixed: unsigned allowed
        cands = {0: [INT, UINT, LONG, ULONG, LL, ULL], 1: [LONG, ULONG, LL, ULL], 2: [LL, ULL]}[lrank]
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
    return CType("funcptr", name=name, size=size, align=size, signed=False,
                 of=ret, params=tuple(params))


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
    members: list = field(default_factory=list)   # (name, CType, bit_width)  (bit_width 0 == plain)
    packed: bool = False
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
        bf_unit_off = None        # byte offset of the active bitfield storage unit (packed path)
        bf_bits = 0               # bits already used in it
        bf_unit_size = 0

        def malign(mtype: CType) -> int:
            return 1 if self.packed else mtype.align

        for mname, mtype, width in self.members:
            align = max(align, malign(mtype))
            if width and self.kind == "struct":
                unit_bits = mtype.size * 8
                if self.packed:                                   # packed: byte-granular sz-byte units
                    if bf_unit_off is None or bf_unit_size != mtype.size or bf_bits + width > unit_bits:
                        bf_unit_off, bf_bits, bf_unit_size = dbits // 8, 0, mtype.size
                        dbits += mtype.size * 8
                    laid.append((mname, mtype, bf_unit_off, bf_bits, width))
                    bf_bits += width
                else:                                             # natural: pack at the bit cursor
                    if (dbits % unit_bits) + width > unit_bits:   # would cross a storage-unit boundary
                        dbits += unit_bits - (dbits % unit_bits)  # -> bump to the next one
                    unit_off = (dbits // unit_bits) * mtype.size
                    laid.append((mname, mtype, unit_off, dbits - unit_off * 8, width))
                    dbits += width
            elif self.kind == "union":
                laid.append((mname, mtype, 0, 0, width))
            else:
                bf_unit_off, bf_bits, bf_unit_size = None, 0, 0     # a plain member flushes the unit
                a8 = malign(mtype) * 8
                if dbits % a8:
                    dbits += a8 - (dbits % a8)
                laid.append((mname, mtype, dbits // 8, 0, width))
                dbits += mtype.size * 8
        align = max(align, self.force_align)
        size = (max((m[1].size for m in self.members), default=0) if self.kind == "union"
                else (dbits + 7) // 8)
        if align and size % align:
            size += align - (size % align)
        return CType(self.kind, name=self.name, size=size, align=max(1, align), fields=tuple(laid))
