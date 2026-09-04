"""BCIR opcode set (small but real, LangRef Sec. 1)."""

from __future__ import annotations

from enum import IntEnum


class Opcode(IntEnum):
    NOP = 0
    LOAD = 1
    STORE = 2
    ADD = 3
    SUB = 4
    MUL = 5
    ATOMIC_ADD = 6
    ATOMIC_SUB = 7
    ATOMIC_XOR = 8
    CMPXCHG = 9
    BARRIER = 10
    PHASE_ENTER = 11
    PHASE_LEAVE = 12
    GGG_LOAD = 13  # gather
    GGG_STORE = 14  # scatter
    T_MACC = 15  # tile matmul-accumulate
    GEM_DISPATCH = 16
    PROV_NOTE = 17  # provenance note (H lane)


#: The opcodes whose semantics are an ATOMIC read-modify-write. Defined next to the
#: enum rather than in one consumer because both the verifier (R5, R9) and candidate
#: generation have to agree on the set: they disagreed, and the verifier certified a
#: plan that realized an atomic as a 16-wide vector op.
ATOMIC_OPCODES = frozenset(
    {Opcode.ATOMIC_ADD, Opcode.ATOMIC_SUB, Opcode.ATOMIC_XOR, Opcode.CMPXCHG}
)
