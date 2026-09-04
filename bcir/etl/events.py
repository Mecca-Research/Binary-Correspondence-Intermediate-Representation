"""Event streams and event kinds (BCIR event law)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EventStream:
    """An input or runtime event stream descriptor."""

    name: str
    kind: str = "text"  # text | binary | telemetry | packet | driver | token
    encoding: str = "utf8"  # utf8 | bytes | le | be | records | custom
    element_bits: int = 8
    max_window: int = 4096


@dataclass(frozen=True)
class EventKind:
    """A named event class with a payload shape."""

    name: str
    payload_shape: str = ""


@dataclass(frozen=True)
class Event:
    """An emitted event: a kind, a payload, and the input span it came from."""

    kind: str
    payload: tuple = ()
    span: tuple[int, int] = (0, 0)
