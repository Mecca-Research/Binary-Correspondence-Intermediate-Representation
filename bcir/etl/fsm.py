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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"fsm state: name must be a non-empty string (got {self.name!r})")
        if self.accepting and self.error:
            raise ValueError(f"fsm state {self.name!r}: a state is accepting or an error, not both")


@dataclass(frozen=True)
class Transition:
    src: str
    dst: str
    on: str  # symbol/token kind to match ("*" = catch-all)
    action: Optional[str] = None  # capture action label
    guard: Optional[str] = None  # documented guard name (advisory in the oracle)

    def __post_init__(self) -> None:
        for name in ("src", "dst", "on"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"fsm transition: {name} must be a non-empty string (got {value!r})"
                )


@dataclass
class TransduceResult:
    accepted: bool
    final_state: str
    captures: list[tuple[str, str]] = field(default_factory=list)
    trace: list[tuple[str, str, str]] = field(default_factory=list)


class Transducer:
    """Validated at construction (S0-6): every state name is unique, every transition names
    two declared states, no two transitions share a `(state, symbol)` key (the old table
    silently kept the LAST one), and the start state exists. The MLIR twin (`bcir.fsm.*`)
    applies the same rules in its op verifiers."""

    def __init__(self, states: Iterable[State], transitions: Iterable[Transition], start: str):
        self.states: dict[str, State] = {}
        for s in states:
            if s.name in self.states:
                raise ValueError(f"duplicate fsm state {s.name!r}")
            self.states[s.name] = s
        self.start = start
        self._tx: dict[tuple[str, str], Transition] = {}
        for t in transitions:
            for end in (t.src, t.dst):
                if end not in self.states:
                    raise ValueError(
                        f"fsm transition {t.src!r} -> {t.dst!r} names unknown state {end!r}"
                    )
            if (t.src, t.on) in self._tx:
                raise ValueError(
                    f"fsm transition on {t.on!r} from state {t.src!r} is declared twice"
                )
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
