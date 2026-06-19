"""Phase 4 — Clang-grade diagnostics, segment 1a: the source-location model + the caret renderer,
and the `diagnose()` front-end entry that turns a lex/parse error into a located diagnostic instead
of an uncaught exception."""
from __future__ import annotations

from bcir.frontends.cfront import diagnose
from bcir.frontends.cfront.cparse import CParseError
from bcir.frontends.cfront.diagnostics import (
    FixIt,
    Note,
    SourceDiagnostic,
    Span,
    line_col,
    render,
)


def test_line_col_resolves_1based_with_clamping():
    src = "ab\ncde\nf"
    assert line_col(src, 0) == (1, 1)
    assert line_col(src, 2) == (1, 3)          # the newline column on line 1
    assert line_col(src, 3) == (2, 1)          # first column of line 2
    assert line_col(src, 5) == (2, 3)
    assert line_col(src, 7) == (3, 1)
    assert line_col(src, 9999) == (3, 2)       # past EOF clamps to the last position
    assert line_col(src, -5) == (1, 1)         # negative clamps to the start


def test_render_points_a_caret_under_the_span():
    src = "int x = 0;\nreturn x +;\n"
    bad = src.index("+")                        # the offending column
    ln, col = line_col(src, bad)
    diag = SourceDiagnostic("error", "expected expression", span=Span.at(bad), phase="parse")
    out = render(diag, src, "t.c").splitlines()
    assert (ln, col) == (2, 10)                                # line 2, column 10 (1-based)
    assert out[0] == "t.c:2:10: error: expected expression"
    assert out[1] == "return x +;"                             # the source line, verbatim
    assert out[2] == " " * (col - 1) + "^"                     # caret under the '+'
    assert out[2].index("^") == bad - src.index("return")     # caret aligns with the '+' column


def test_render_underlines_a_multichar_span_and_keeps_tabs():
    src = "\tfoobar = 1;\n"                     # a leading tab: the caret line must reproduce it
    span = Span(src.index("foobar"), src.index("foobar") + len("foobar"))
    out = render(SourceDiagnostic("warning", "unused", span=span), src, "t.c").splitlines()
    assert out[0].startswith("t.c:1:2: warning: unused")
    assert out[2] == "\t^~~~~~"                 # tab preserved, then ^ + 5 tildes (6-wide underline)


def test_render_file_level_banner_without_span():
    out = render(SourceDiagnostic("error", "no such file"), "irrelevant", "t.c")
    assert out == "t.c: error: no such file"   # no caret line when there is no span


def test_render_includes_notes_and_fixits():
    src = "int x\n"
    diag = SourceDiagnostic(
        "error", "expected ';'", span=Span.at(src.index("x") + 1),
        fixits=[FixIt(Span.at(src.index("x") + 1, 0), ";")],
        notes=[Note("declaration started here", Span.at(0))])
    out = render(diag, src, "t.c")
    assert "fix-it: insert ';'" in out
    assert "t.c:1:1: note: declaration started here" in out


def test_invalid_severity_rejected():
    try:
        SourceDiagnostic("fatal", "x")         # only error/warning/note are valid severities
        raise AssertionError("expected ValueError for an unknown severity")
    except ValueError:
        pass


def test_diagnose_parse_error_is_located_not_raised():
    rep = diagnose("int f(void){ return 1 }\n", filename="t.c")
    assert not rep.ok and len(rep.diagnostics) == 1
    d = rep.diagnostics[0]
    assert d.severity == "error" and d.phase == "parse" and d.span is not None
    ln, col = line_col(rep.source, d.span.start)
    assert ln == 1 and rep.source[d.span.start] == "}"        # caret on the unexpected '}'
    assert "t.c:1:" in rep.render() and "^" in rep.render()


def test_lexer_error_carries_a_source_offset():
    from bcir.frontends.cfront.clex import CLexError, tokenize
    src = 'char *s = "unterminated'                            # no closing quote -> a lex error
    try:
        tokenize(src)
        raise AssertionError("expected CLexError for an unterminated string")
    except CLexError as e:
        assert e.pos == src.index('"')                        # the offset of the opening quote


def test_diagnose_undeclared_identifier_is_a_clean_diagnostic():
    # an undeclared name used to crash the front end with a bare KeyError; now it is a located
    # lowering diagnostic that points a caret at the identifier (segment 1c: AST source spans).
    rep = diagnose("int f(void){ return zzz + 1; }\n", filename="t.c")
    d = rep.diagnostics[0]
    assert not rep.ok and d.phase == "lower" and d.span is not None
    assert rep.source[d.span.start:d.span.start + 3] == "zzz"      # caret on the offending name
    assert "use of undeclared identifier 'zzz'" in rep.render()


def test_diagnose_clean_unit_reports_nothing():
    rep = diagnose("int f(int x){ return x + 1; }\n")
    assert rep.ok and rep.diagnostics == []


# --- segment 1b: parser error recovery (several diagnostics per run) ------------------------------

def test_recovery_reports_multiple_statement_errors_in_one_body():
    # two malformed statements in the same function: panic-mode recovery resynchronizes on `;` and
    # reports both, each located, rather than stopping at the first.
    src = "int f(void){\n    int a = ;\n    int b = 1;\n    return b +;\n}\n"
    rep = diagnose(src, filename="t.c")
    assert len(rep.diagnostics) == 2 and all(d.phase == "parse" for d in rep.diagnostics)
    lines = [line_col(rep.source, d.span.start)[0] for d in rep.diagnostics]
    assert lines == sorted(lines)                              # reported in source order
    rendered = rep.render()
    assert rendered.count("error:") == 2 and rendered.count("^") == 2


def test_recovery_resynchronizes_and_still_parses_the_good_declaration():
    from bcir.frontends.cfront.cparse import parse_with_recovery
    # a broken global, then a well-formed function: recovery skips the bad decl and parses `good`.
    unit, diags = parse_with_recovery("int 3bad;\nint good(void){ return 0; }\n")
    assert len(diags) == 1 and diags[0].phase == "parse"
    assert any(fn.name == "good" for fn in unit.funcs)         # the later declaration survived


def test_recovery_reports_several_top_level_errors():
    rep = diagnose("int 1x;\nint 2y;\nint ok(void){ return 0; }\n", filename="t.c")
    assert len(rep.diagnostics) == 2                           # both bad globals, the good fn is silent


def test_compile_unit_still_raises_on_a_parse_error():
    # the compile path keeps its one-shot contract (it needs a well-formed AST); only diagnose recovers.
    from bcir.frontends.cfront import compile_unit
    try:
        compile_unit("int f(void){ return 1 }\n", check_clang=False)
        raise AssertionError("compile_unit should raise CParseError on malformed input")
    except CParseError:
        pass


def test_single_error_still_reported_after_recovery_added():
    rep = diagnose("int f(void){ return 1 }\n", filename="t.c")
    assert len(rep.diagnostics) == 1 and rep.diagnostics[0].phase == "parse"


# --- segment 1c: AST source spans -> semantic-error carets ----------------------------------------

def test_name_node_carries_its_source_offset():
    from bcir.frontends.cfront.cparse import parse_unit
    src = "int f(int a){ return a; }\n"
    unit = parse_unit(src)
    ret = unit.funcs[0].body[0]                                   # the `return a;` statement
    assert ret.value.ident == "a" and ret.value.pos == src.rindex("a")   # the `a` in `return a`


def test_semantic_caret_points_at_the_assignment_target():
    rep = diagnose("int f(void){\n    nope = 5;\n    return 0;\n}\n", filename="t.c")
    d = rep.diagnostics[0]
    ln, col = line_col(rep.source, d.span.start)
    assert d.phase == "lower" and (ln, col) == (2, 1)             # caret under `nope` on line 2
    assert "use of undeclared identifier 'nope'" in rep.render() and "^" in rep.render()
