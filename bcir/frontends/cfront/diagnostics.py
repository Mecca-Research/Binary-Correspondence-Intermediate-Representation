"""Clang-grade source diagnostics for the C frontend (Phase 4) — the source-location model + the
renderer that anchors a caret under the offending span. This is the foundation the rest of the
"Clang-grade diagnostics" segment builds on: include/macro-expansion-stack notes, fix-it hints,
parser error recovery (many diagnostics per run), and machine-readable (JSON) output.

A diagnostic carries half-open `[start, end)` byte offsets into the *preprocessed* translation-unit
text (the lexer/parser already track each token's `pos`). The renderer resolves an offset to a 1-based
line/column and prints the Clang layout — a `file:line:col: severity: message` banner, the source
line, and a `^~~~` underline — so the output is drop-in familiar and downstream-parseable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_SEVERITIES = ("error", "warning", "note")


@dataclass(frozen=True)
class Span:
    """A half-open `[start, end)` range of byte offsets into the translation-unit source. A zero-width
    span (`start == end`, e.g. an insertion point) still anchors a single caret."""
    start: int
    end: int

    @staticmethod
    def at(pos: int, length: int = 1) -> "Span":
        return Span(pos, pos + max(0, length))


@dataclass(frozen=True)
class FixIt:
    """A suggested edit: replace the source over `span` with `replacement` (an insertion has a
    zero-width span; a deletion has an empty replacement)."""
    span: Span
    replacement: str


@dataclass
class Note:
    """A secondary message attached to a diagnostic — an include-stack frame ("in file included
    from ..."), a macro-expansion site, or a "previous definition was here" back-reference."""
    message: str
    span: Span | None = None


@dataclass
class SourceDiagnostic:
    """One rendered-ready diagnostic. `phase` records which front-end stage raised it (lex /
    preprocess / parse / lower) for machine-readable provenance; `notes`/`fixits` are the secondary
    spans and suggested edits."""
    severity: str
    message: str
    span: Span | None = None
    notes: list = field(default_factory=list)         # list[Note]
    fixits: list = field(default_factory=list)         # list[FixIt]
    phase: str = ""

    def __post_init__(self):
        if self.severity not in _SEVERITIES:
            raise ValueError(f"unknown diagnostic severity {self.severity!r}")


@dataclass
class DiagnosticReport:
    """A run's diagnostics bundled with the exact text their spans index into (the *preprocessed*
    translation unit, or the original source if preprocessing itself failed) plus the file name, so
    the report renders itself without the caller having to thread the right source back in."""
    diagnostics: list                                 # list[SourceDiagnostic]
    source: str
    filename: str = "<source>"

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    def render(self) -> str:
        return "\n".join(render(d, self.source, self.filename) for d in self.diagnostics)


def line_col(source: str, offset: int) -> tuple[int, int]:
    """The 1-based `(line, column)` of a byte offset, clamped into `[0, len(source)]`. Column counts
    bytes from the line start (a tab is one column here; the caret line reproduces leading tabs so it
    still aligns visually under any tab width)."""
    offset = max(0, min(offset, len(source)))
    line = source.count("\n", 0, offset) + 1
    col = offset - (source.rfind("\n", 0, offset) + 1) + 1
    return line, col


def _line_bounds(source: str, offset: int) -> tuple[int, int]:
    """The `[start, end)` offsets of the line containing `offset` (end excludes the newline)."""
    offset = max(0, min(offset, len(source)))
    start = source.rfind("\n", 0, offset) + 1
    end = source.find("\n", offset)
    return start, (len(source) if end == -1 else end)


def _caret_prefix(line_text: str, upto: int) -> str:
    """The blank run that positions a caret under column `upto` (0-based within the line), reproducing
    leading tabs as tabs so the alignment holds under any tab width."""
    return "".join(ch if ch == "\t" else " " for ch in line_text[:upto])


def _snippet(source: str, span: Span) -> list[str]:
    """The source line for `span` plus a caret/underline line. The underline spans the columns the
    diagnostic covers *within that first line* (a `^` then `~`s), with at least one caret."""
    lstart, lend = _line_bounds(source, span.start)
    line_text = source[lstart:lend]
    col = span.start - lstart                                   # 0-based column of the caret
    width = max(1, min(span.end, lend) - span.start)            # clamp the underline to the line
    underline = _caret_prefix(line_text, col) + "^" + "~" * (width - 1)
    return [line_text, underline]


def render(diag: SourceDiagnostic, source: str, filename: str = "<source>") -> str:
    """Render `diag` in the Clang text layout. A diagnostic with no span prints a file-level banner
    (no caret); notes and fix-its render under it."""
    out: list[str] = []

    def banner(severity: str, message: str, span: Span | None) -> None:
        if span is None:
            out.append(f"{filename}: {severity}: {message}")
        else:
            ln, col = line_col(source, span.start)
            out.append(f"{filename}:{ln}:{col}: {severity}: {message}")
            out.extend(_snippet(source, span))            # not `out +=`: that would rebind in this closure

    banner(diag.severity, diag.message, diag.span)
    for fx in diag.fixits:                                      # a one-line suggested edit under the caret
        verb = "remove" if fx.replacement == "" else ("insert" if fx.span.start == fx.span.end
                                                       else "replace with")
        out.append(f"fix-it: {verb} {fx.replacement!r}".rstrip())
    for note in diag.notes:
        banner("note", note.message, note.span)
    return "\n".join(out)
