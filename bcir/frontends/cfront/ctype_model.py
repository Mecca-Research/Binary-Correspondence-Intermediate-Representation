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
        return self.kind == "scalar" and self.name != "void"

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


def scalar(name: str) -> CType:
    if name not in _SCALAR:
        raise KeyError(f"unknown scalar type {name!r}")
    size, signed = _SCALAR[name]
    return CType("scalar", name=name, size=size, align=max(1, size), signed=signed)


def is_scalar_name(name: str) -> bool:
    return name in _SCALAR


def pointer(of: CType) -> CType:
    return CType("pointer", name="ptr", size=PTR_SIZE, align=PTR_SIZE, signed=False, of=of)


def funcptr(name: str, ret: CType, params: tuple = ()) -> CType:
    """A function-pointer type — pointer-sized, carrying its return + parameter types so the emitter
    can reconstruct a call (``name`` is the typedef spelling, used verbatim in faithful emission)."""
    return CType("funcptr", name=name, size=PTR_SIZE, align=PTR_SIZE, signed=False,
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
        offset = 0
        align = 1
        laid: list = []
        bf_unit_off = None        # byte offset of the active bitfield storage unit
        bf_bits = 0               # bits already used in it
        bf_unit_size = 0

        def malign(mtype: CType) -> int:
            return 1 if self.packed else mtype.align

        for mname, mtype, width in self.members:
            align = max(align, malign(mtype))
            if width and self.kind == "struct":
                unit_bits = mtype.size * 8
                if bf_unit_off is None or bf_unit_size != mtype.size or bf_bits + width > unit_bits:
                    if offset % malign(mtype):
                        offset += malign(mtype) - (offset % malign(mtype))
                    bf_unit_off, bf_bits, bf_unit_size = offset, 0, mtype.size
                    offset += mtype.size
                laid.append((mname, mtype, bf_unit_off, bf_bits, width))
                bf_bits += width
            elif self.kind == "union":
                laid.append((mname, mtype, 0, 0, width))
            else:
                bf_unit_off, bf_bits, bf_unit_size = None, 0, 0     # a plain member flushes the unit
                if offset % malign(mtype):
                    offset += malign(mtype) - (offset % malign(mtype))
                laid.append((mname, mtype, offset, 0, width))
                offset += mtype.size
        align = max(align, self.force_align)
        size = (max((m[1].size for m in self.members), default=0) if self.kind == "union"
                else offset)
        if align and size % align:
            size += align - (size % align)
        return CType(self.kind, name=self.name, size=size, align=max(1, align), fields=tuple(laid))
