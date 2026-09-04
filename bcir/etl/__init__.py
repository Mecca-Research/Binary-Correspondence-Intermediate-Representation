"""BCIR Event Transduction Layer (GEM-E / M5).

The next organ after compute: a universal ingestion engine that turns structured
text, binary formats, driver packets, and telemetry into BCIR claims/events via
the same correspondence machinery. The slogans:

    parser generator = grammar -> event machine
    GEM-E            = event machine -> streams -> target execution

This package is runnable today: `parse.parse_rop` lexes+parses a mini ROP-claim
grammar into a BCIR `Module` (which `kbcir.optimize` then plans), `binary.decode`
decodes a packed record (e.g. an NVMe SQE header) into fields, and
`fsm.Transducer` runs a finite-state transducer over a token stream. The MLIR law
mirrors these as `bcir.event.* / bcir.fsm.* / bcir.parse.* / bcir.binary.*`.
"""

from .binary import BinaryField, BinaryFormat, BinaryRecord, decode
from .events import Event, EventKind, EventStream
from .fsm import State, TransduceResult, Transducer, Transition
from .parse import Grammar, Lexer, Token, TokenRule, parse_rop

__all__ = [
    "BinaryField",
    "BinaryFormat",
    "BinaryRecord",
    "decode",
    "Event",
    "EventKind",
    "EventStream",
    "State",
    "TransduceResult",
    "Transducer",
    "Transition",
    "Grammar",
    "Lexer",
    "Token",
    "TokenRule",
    "parse_rop",
]
