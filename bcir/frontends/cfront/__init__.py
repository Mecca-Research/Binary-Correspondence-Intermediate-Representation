"""The C frontend (Phase C.1 / C.1-MVP) — a small, Clang-compatible C subset → the BCIR claim graph.

This is the *input seam* the roadmap calls the keystone: it ingests driver/kernel-relevant C (the
ladder stages L1–L4 here: fixed-width integer expressions → structs/unions → pointers/arrays →
functions + call graph → R18) and lowers it to the **same** claim graph (`Resource`/`Claim`/`Phase`)
the oracle already reasons over — so R1–R18 + the K_BCIR cost model apply unchanged (the dual-rail
invariant).

The public entry is :func:`compile_unit`, which runs the full six-artifact pipeline for one C
translation unit: the parsed C, the lowered claim graph, the K_BCIR plan, the emitted C output, the
R1–R18 verifier checkpoint, the Clang behaviour-equivalence verdict, and the `bcir-explain` text.
"""

from __future__ import annotations

from .cparse import CParseError, parse_unit
from .diagnostics import (
    DiagnosticReport,
    SourceDiagnostic,
    Span,
    diagnostic_to_dict,
    line_col,
    render,
)
from .lower import lower_unit
from .pipeline import CompileResult, compile_unit, compile_with_fallback, diagnose, emit_selfcheck

__all__ = [
    "CParseError",
    "CompileResult",
    "DiagnosticReport",
    "SourceDiagnostic",
    "Span",
    "compile_unit",
    "compile_with_fallback",
    "diagnose",
    "diagnostic_to_dict",
    "emit_selfcheck",
    "line_col",
    "lower_unit",
    "parse_unit",
    "render",
]
