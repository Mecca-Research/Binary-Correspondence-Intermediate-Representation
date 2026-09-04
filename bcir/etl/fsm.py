"""A finite-state transducer (the nervous reflex of GEM-E).

A token-driven transducer: states (some accepting/error), transitions keyed by
`(state, symbol)` with optional capture actions. `Transducer.run` consumes a
symbol stream and reports acceptance, the final state, captures, and the trace.
A `"*"` transition symbol is a catch-all/else edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass(frozen=True)
class State:
    name: str
    accepting: bool = False
    error: bool = False


@dataclass(frozen=True)
class Transition:
    src: str
    dst: str
    on: str  # symbol/token kind to match ("*" = catch-all)
    action: Optional[str] = None  # capture action label
    guard: Optional[str] = None  # documented guard name (advisory in the oracle)


@dataclass
class TransduceResult:
    accepted: bool
    final_state: str
    captures: list[tuple[str, str]] = field(default_factory=list)
    trace: list[tuple[str, str, str]] = field(default_factory=list)


class Transducer:
    def __init__(self, states: Iterable[State], transitions: Iterable[Transition], start: str):
        self.states = {s.name: s for s in states}
        self.start = start
        self._tx: dict[tuple[str, str], Transition] = {}
        for t in transitions:
            self._tx[(t.src, t.on)] = t
        if start not in self.states:
            raise ValueError(f"unknown start state {start!r}")

    def run(self, symbols: Iterable[str]) -> TransduceResult:
        cur = self.start
        res = TransduceResult(False, cur)
        for sym in symbols:
            t = self._tx.get((cur, sym)) or self._tx.get((cur, "*"))
            if t is None or self.states[cur].error:
                res.final_state = cur
                res.accepted = False
                return res
            if t.action:
                res.captures.append((t.action, sym))
            res.trace.append((cur, sym, t.dst))
            cur = t.dst
        res.final_state = cur
        res.accepted = self.states[cur].accepting and not self.states[cur].error
        return res
