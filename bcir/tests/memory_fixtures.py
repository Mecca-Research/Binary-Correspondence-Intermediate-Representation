"""Shared fixtures for the G5 gates (S1-D): the seven-resource layout corpus of the
2026-08-12 report's section 6.4, its worst-case witness, and the module builder that runs
a fixture through the real static memory planner at 64-byte alignment.

Used by `test_static_memory.py` and by the GEM+ baseline harness
(`tools/perf/gemplus_baseline.py::measure_memory`), so the tests and the graded rows
measure ONE definition of each gate. Not a test module (run_all collects `test_*.py`).

The report's own corpus is not in the tree; this one reproduces its shape -- 500
deterministic seven-resource fixtures with unit sizes and phase lifetimes -- and its RED
numbers are close to the report's (first-fit suboptimal on 40.4% against 38.6%, worst ratio
1.6x). `WORST_FIXTURE` reproduces the report's worst case exactly: first-fit 21 units against
a proved 13-unit optimum, 1,344 against 832 bytes at 64-byte alignment through the real
planner.
"""

from __future__ import annotations

import random

from bcir.kbcir.static_memory import LayoutItem, exact_layout, first_fit_layout
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

#: The corpus parameters (frozen): 7 resources, sizes 1..8 units, 8 phases, spans 0..4.
CORPUS_SIZE = 500
RESOURCES = 7
SIZE_MAX = 8
HORIZON = 8
SPAN_MAX = 4
#: One size unit is one 64-byte line through the real planner (the report's alignment).
UNIT_BYTES = 64

#: The report's worst case, reproduced: (rid, size units, first phase, last phase).
WORST_FIXTURE = (
    (1, 2, 0, 2),
    (2, 1, 0, 4),
    (3, 5, 3, 7),
    (4, 2, 1, 4),
    (5, 3, 1, 3),
    (6, 5, 4, 6),
    (7, 3, 0, 2),
)


def fixture(seed: int) -> tuple[tuple[int, int, int, int], ...]:
    """The corpus fixture for `seed`: (rid, size units, first phase, last phase) x 7."""
    rng = random.Random(seed)
    rows = []
    for rid in range(1, RESOURCES + 1):
        size = rng.randint(1, SIZE_MAX)
        lo = rng.randint(0, HORIZON - 1)
        hi = min(HORIZON - 1, lo + rng.randint(0, SPAN_MAX))
        rows.append((rid, size, lo, hi))
    return tuple(rows)


def corpus():
    for seed in range(CORPUS_SIZE):
        yield seed, fixture(seed)


def items_of(rows, unit: int = 1, alignment: int = 1) -> list[LayoutItem]:
    """The layout items of a fixture: sizes in `unit`s, phase liveness as half-open ticks."""
    return [LayoutItem(rid, size * unit, alignment, lo, hi + 1) for rid, size, lo, hi in rows]


def unit_layouts(rows, budget: int = 200_000):
    """(first-fit extent, exact result) of a fixture in size units."""
    items = items_of(rows)
    return first_fit_layout(items).extent, exact_layout(items, budget)


def module_of(rows, name: str = "layout") -> tuple[Module, dict[int, str]]:
    """A module whose phase liveness is the fixture's: each resource is `size` lines of 64
    bytes at 64-byte alignment, touched by one claim in its first phase and one in its last,
    across a chain of HORIZON phases; every resource is bound to the one RAM bank."""
    module = Module(name)
    phases = [Phase(index, () if index == 0 else (index - 1,), []) for index in range(HORIZON)]
    claim_id = 1
    for rid, size, lo, hi in rows:
        module.add_resource(
            Resource(rid, Domain.RAM, elem_bytes=UNIT_BYTES, shape=(size,), align=UNIT_BYTES)
        )
        for phase_id in sorted({lo, hi}):
            phases[phase_id].claims.append(
                Claim(
                    claim_id,
                    Opcode.ADD,
                    Lane.U,
                    StrideClass.UNIT,
                    count=1,
                    rd=(rid,),
                    op="tensor.touch",
                    domain=Domain.RAM,
                )
            )
            claim_id += 1
    for phase in phases:
        module.add_phase(phase)
    return module, {rid: "ram" for rid, _size, _lo, _hi in rows}
