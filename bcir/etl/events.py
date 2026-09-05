"""Event streams and event kinds (BCIR event law).

The descriptors validate at construction (S0-6): a stream of an unknown kind or encoding, a
zero element width or window, an unnamed event kind, an inverted span -- refused where it is
written. The MLIR twins (`bcir.event.stream` / `bcir.event.kind`) apply the same rules in
their op verifiers; the structural corpus holds the two rails together.
"""

from __future__ import annotations

from dataclasses import dataclass

STREAM_KINDS = ("text", "binary", "telemetry", "packet", "driver", "token")
STREAM_ENCODINGS = ("utf8", "bytes", "le", "be", "records", "custom")


@dataclass(frozen=True)
class EventStream:
    """An input or runtime event stream descriptor."""

    name: str
    kind: str = "text"  # text | binary | telemetry | packet | driver | token
    encoding: str = "utf8"  # utf8 | bytes | le | be | records | custom
    element_bits: int = 8
    max_window: int = 4096

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"event stream: name must be a non-empty string (got {self.name!r})")
        if self.kind not in STREAM_KINDS:
            raise ValueError(
                f"stream {self.name!r}: kind must be one of {'|'.join(STREAM_KINDS)} "
                f"(got {self.kind!r})"
            )
        if self.encoding not in STREAM_ENCODINGS:
            raise ValueError(
                f"stream {self.name!r}: encoding must be one of {'|'.join(STREAM_ENCODINGS)} "
                f"(got {self.encoding!r})"
            )
        for name in ("element_bits", "max_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"stream {self.name!r}: {name} must be positive (got {value!r})")


@dataclass(frozen=True)
class EventKind:
    """A named event class with a payload shape."""

    name: str
    payload_shape: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"event kind: name must be a non-empty string (got {self.name!r})")


@dataclass(frozen=True)
class Event:
    """An emitted event: a kind, a payload, and the input span it came from."""

    kind: str
    payload: tuple = ()
    span: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError(f"event: kind must be a non-empty string (got {self.kind!r})")
        lo, hi = self.span
        if lo < 0 or hi < lo:
            raise ValueError(
                f"event {self.kind!r}: span must satisfy 0 <= begin <= end (got {self.span})"
            )
