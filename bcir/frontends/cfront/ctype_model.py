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
    of: "CType | None" = None            # element type (pointer/array)
    count: int = 0                       # array length
    fields: tuple = ()                   # ((name, CType, offset), ...) for struct/union

    @property
    def is_integer(self) -> bool:
        return self.kind == "scalar" and self.name != "void"

    @property
    def is_aggregate(self) -> bool:
        return self.kind in ("struct", "union")

    def field(self, name: str):
        for fname, ftype, off in self.fields:
            if fname == name:
                return ftype, off
        raise KeyError(name)


def scalar(name: str) -> CType:
    if name not in _SCALAR:
        raise KeyError(f"unknown scalar type {name!r}")
    size, signed = _SCALAR[name]
    return CType("scalar", name=name, size=size, align=max(1, size), signed=signed)


def is_scalar_name(name: str) -> bool:
    return name in _SCALAR


def pointer(of: CType) -> CType:
    return CType("pointer", name="ptr", size=PTR_SIZE, align=PTR_SIZE, signed=False, of=of)


def array(of: CType, count: int) -> CType:
    return CType("array", name="array", size=of.size * count, align=of.align, of=of, count=count)


@dataclass
class AggregateBuilder:
    """Compute a struct/union layout the way Clang does (natural alignment, no packing)."""
    kind: str
    name: str
    members: list = field(default_factory=list)   # (name, CType)

    def build(self) -> CType:
        offset = 0
        align = 1
        laid: list = []
        for mname, mtype in self.members:
            align = max(align, mtype.align)
            if self.kind == "union":
                laid.append((mname, mtype, 0))
            else:
                if offset % mtype.align:
                    offset += mtype.align - (offset % mtype.align)   # pad to member alignment
                laid.append((mname, mtype, offset))
                offset += mtype.size
        size = (max((m[1].size for m in self.members), default=0) if self.kind == "union"
                else offset)
        if align and size % align:
            size += align - (size % align)                            # round up to aggregate align
        return CType(self.kind, name=self.name, size=size, align=max(1, align), fields=tuple(laid))
