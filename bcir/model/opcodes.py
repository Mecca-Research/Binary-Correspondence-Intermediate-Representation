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
    GGG_LOAD = 13   # gather
    GGG_STORE = 14  # scatter
    T_MACC = 15     # tile matmul-accumulate
    GEM_DISPATCH = 16
    PROV_NOTE = 17  # provenance note (H lane)
