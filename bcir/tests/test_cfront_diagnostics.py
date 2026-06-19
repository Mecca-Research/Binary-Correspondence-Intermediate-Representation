"""Phase 4 — Clang-grade diagnostics. The source-location engine + caret renderer (1a), parser error
recovery so one run reports several diagnostics (1b), AST source spans for semantic-error carets (1c),
parser fix-its for missing punctuation (1d), machine-readable JSON output (1e), and include/expansion
-stack notes via the preprocessor line map (1c-ii)."""
from __future__ import annotations

from bcir.frontends.cfront import diagnose
from bcir.frontends.cfront.cparse import CParseError
from bcir.frontends.cfront.diagnostics import (
    FixIt,
    Note,
    SourceDiagnostic,
    Span,
    diagnostic_to_dict,
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


# --- segment 1e: machine-readable (JSON) output ---------------------------------------------------

def test_json_output_carries_location_and_phase():
    import json
    rep = diagnose("int f(void){ return 1 }\n", filename="t.c")
    obj = json.loads(rep.to_json())                              # a valid JSON array
    assert isinstance(obj, list) and len(obj) == 1
    d = obj[0]
    assert d["severity"] == "error" and d["phase"] == "parse" and d["file"] == "t.c"
    assert d["line"] == 1 and d["column"] >= 1
    assert d["range"]["end"] > d["range"]["start"]               # a non-empty byte range


def test_json_semantic_range_covers_the_identifier():
    import json
    rep = diagnose("int f(void){ return missing; }\n", filename="t.c")
    d = json.loads(rep.to_json())[0]
    assert d["phase"] == "lower"
    assert rep.source[d["range"]["start"]:d["range"]["end"]] == "m"   # span starts at the identifier


def test_json_serializes_fixits_and_notes():
    src = "int x\n"
    diag = SourceDiagnostic(
        "error", "expected ';'", span=Span.at(len("int x")),
        fixits=[FixIt(Span.at(len("int x"), 0), ";")],
        notes=[Note("declaration started here", Span.at(0))])
    d = diagnostic_to_dict(diag, src, "t.c")
    assert d["fixits"] == [{"replacement": ";", "range": {"start": 5, "end": 5}}]
    assert d["notes"][0]["message"] == "declaration started here" and d["notes"][0]["line"] == 1


def test_json_clean_unit_is_empty_array():
    import json
    assert json.loads(diagnose("int f(int x){ return x; }\n").to_json()) == []


# --- segment 1d: parser fix-its -------------------------------------------------------------------

def test_missing_semicolon_fixit_repairs_the_source():
    src = "int f(void){ return 1 }\n"
    rep = diagnose(src, filename="t.c")
    fx = rep.diagnostics[0].fixits[0]
    assert fx.replacement == ";" and fx.span.start == fx.span.end     # a zero-width insertion
    assert rep.source[fx.span.start - 1] == "1"                       # inserted right after `return 1`
    fixed = rep.source[:fx.span.start] + fx.replacement + rep.source[fx.span.end:]
    assert diagnose(fixed).ok                                         # applying it yields a clean parse


def test_fixit_renders_and_serializes():
    import json
    rep = diagnose("int f(void){ return 1 }\n", filename="t.c")
    assert "fix-it: insert ';'" in rep.render()
    obj = json.loads(rep.to_json())[0]
    assert obj["fixits"] == [{"replacement": ";", "range": {"start": 20, "end": 20}}]


def test_no_fixit_for_a_non_punctuation_error():
    # a semantic (lower) error is not a missing-punctuation case -> no suggested edit.
    rep = diagnose("int f(void){ return zzz; }\n", filename="t.c")
    assert rep.diagnostics[0].fixits == []


# --- segment 1c-ii: include/expansion-stack notes (preprocessor line map) --------------------------

def test_error_in_an_included_header_shows_the_include_frame():
    hdr = "int g(void){\n    return undefined_var;\n}\n"
    main = '#include "header.h"\nint mainf(void){ return g(); }\n'
    rep = diagnose(main, includes={"header.h": hdr}, filename="main.c")
    out = rep.render()
    assert "In file included from main.c:1:" in out         # the include frame, Clang-style
    assert "header.h:2:8: error: use of undeclared identifier 'undefined_var'" in out   # origin loc


def test_include_stack_in_json():
    import json
    hdr = "int g(void){ return undefined_var; }\n"
    main = '#include "header.h"\nint mainf(void){ return g(); }\n'
    d = json.loads(diagnose(main, includes={"header.h": hdr}, filename="main.c").to_json())[0]
    assert d["file"] == "header.h" and d["line"] == 1
    assert d["includedFrom"] == [{"file": "main.c", "line": 1}]


def test_line_map_corrects_the_line_after_an_inlined_include():
    # the header inlines 1 line, but the top-level error on main.c line 2 still reports line 2
    # (not the shifted preprocessed line) thanks to the line map.
    main = '#include "h.h"\nint mainf(void){ return nope; }\n'
    rep = diagnose(main, includes={"h.h": "int g(void){ return 0; }\n"}, filename="main.c")
    ln, col = line_col(rep.source, rep.diagnostics[0].span.start)   # preprocessed line
    out = rep.render()
    assert "main.c:2:" in out and "In file included from" not in out   # top-level: no frame, true line


def test_nested_includes_stack_two_frames():
    files = {"a.h": '#include "b.h"\n', "b.h": "int g(void){ return undef; }\n"}
    main = '#include "a.h"\nint m(void){ return 0; }\n'
    out = diagnose(main, includes=files, filename="main.c").render()
    assert "In file included from main.c:1:" in out
    assert "In file included from a.h:1:" in out
    assert "b.h:1:" in out and "undef" in out


def test_no_include_unit_has_no_frame_or_includedfrom():
    import json
    rep = diagnose("int f(void){ return zzz; }\n", filename="t.c")
    assert "In file included from" not in rep.render()
    assert "includedFrom" not in json.loads(rep.to_json())[0]
