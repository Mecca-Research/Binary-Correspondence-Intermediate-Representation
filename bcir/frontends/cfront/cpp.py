"""A C preprocessor for the frontend (ladder stage L7): object-, function- and variadic-like `#define`
macros (with `#` stringize, `##` paste, `__VA_ARGS__` and C23 `__VA_OPT__`), `#undef`, conditional
compilation (`#if`/`#ifdef`/`#ifndef`/
`#elif`/`#elifdef`/`#elifndef`/`#else`/`#endif`) with a constant-expression evaluator (`defined`,
`__has_include`, `__has_embed`, `__has_attribute`, `__has_builtin`, `__has_c_attribute`), the
predefined macros `__FILE__`/`__LINE__`/`__DATE__`/`__TIME__`
(and the static `__STDC__`/`__STDC_VERSION__`/`__STDC_HOSTED__`), the `#line` directive, the
`_Pragma` operator, `#include` of project headers, and C23 `#embed`.

It runs *before* the lexer/parser, producing fully-expanded source text (the lexer still skips any
residual `#`-line, but after this pass there are none). Translation phase 3 happens here too: line
continuations are spliced and comments are replaced by a single space *before* directives are
scanned, so a comment in a macro body or on a directive line is gone and a `#define` that only
appears inside a comment never takes effect (a block comment keeps its newlines so `__LINE__` holds).
`#include`/`#embed` resolve against an
in-memory file map (the real driver path mounts a header search path); the resulting text is what
both the parser and the Clang behaviour-equivalence harness consume, so the comparison validates the
lowering of the *preprocessed* program.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..._artifact_json import read_bounded_text
from .abi import HOST, TargetABI
from .clex import (
    IDENT_MAX,
    NONASCII,
    CLexError,
    char_constant_units,
    int_literal_parts,
    parse_int_literal,
)

_PREDEFINED = {"__STDC__": "1", "__STDC_VERSION__": "202311L", "__STDC_HOSTED__": "1"}
# dynamic predefined macros: expanded per source position, not stored as static bodies.
_DYNAMIC = ("__FILE__", "__LINE__")
# feature-test operators usable in #if (and reported as `defined`); evaluated in _eval.
_HAS_OPS = ("__has_include", "__has_embed", "__has_attribute", "__has_builtin", "__has_c_attribute")
_SUPPORTED_ATTRS = frozenset({"packed", "aligned"})  # attributes the L8 ABI honours (GCC __x__ ok)
#: The feature-test operators whose parenthesized operand the twin's `#if` walk reads as tokens.
_HAS_PAREN = ("__has_attribute", "__has_builtin", "__has_c_attribute")
#: The reason both rails refuse a `defined` with no name, or with `(` and a name but no `)`, for (CF-PPLIMITS).
_BAD_DEFINED = "malformed defined operator"
#: The white space a directive line is read across, as the twin reads it (`bcir_cpp.c`): a space or a tab before its
#: `#`, between the `#` and its name and after the name, the name a run of ASCII identifier characters. `str.strip`
#: had read every Unicode space as white space -- `\u00a0#define K 3u` defined `K` here and was refused by the twin --
#: and the name had ended only at a space, so `#define\tK 3u` was an unknown directive here (CF-PPLIMITS).
_DIRECTIVE_SPACE = " \t"
_DIRECTIVE_NAME = re.compile(r"[A-Za-z0-9_]*", re.ASCII)
#: The ASCII white space a directive's operand is stripped of at its ends: never a character past ASCII, which stays
#: in the operand for the reader that refuses it (NONASCII).
_ASCII_SPACE = " \t\r\f\v"
# preprocessing tokens: identifier, number, string, char, or punctuation (multi-char first).
_PUNCT = [
    "<<=",
    ">>=",
    "...",
    "->",
    "++",
    "--",
    "<<",
    ">>",
    "<=",
    ">=",
    "==",
    "!=",
    "&&",
    "||",
    "##",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "#",
    "+",
    "-",
    "*",
    "/",
    "%",
    "&",
    "|",
    "^",
    "~",
    "!",
    "<",
    ">",
    "=",
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    ";",
    ",",
    ".",
    ":",
    "?",
]
#: ASCII (CF-PPLIMITS): `\w` and `\d` had taken every Unicode letter and digit, so `café` was one name here and `caf`
#: and two bytes on the twin; and a character no alternative matched was dropped -- `x + caf€` lowered as `x + caf` --
#: where the twin hands each such byte on as a token of its own.
_TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"'  # string
    r"|'(?:\\.|[^'\\])*'"  # char
    r"|(?:u8|[LuU])'(?:\\.|[^'\\])*'"  # a prefixed char: one token (C11 6.4.4.4), its `L` no macro (CF-PPARITH)
    r"|\.?\d(?:[eEpP][-+]|[\w.'])*"  # pp-number (C23 ' seps; ints, hex/bin, floats w/ exp+suffix)
    r"|[A-Za-z_]\w*"  # identifier
    rf"|{'|'.join(re.escape(p) for p in _PUNCT)}"  # the punctuators, longest first
    r"|\\"  # a stray backslash is its own token (a path in a stringize arg `#x` -> `"C:\\tmp"`)
    r"|\S",  # any other character, one token: the lexer refuses it (`@`, a NUL, one past ASCII)
    re.ASCII,
)


class CPPError(Exception):
    """A preprocessing error. `pos` is a source byte offset when known (else a file-level banner)."""

    def __init__(self, message: str, pos: int | None = None):
        super().__init__(message)
        self.pos = pos


@dataclass
class Macro:
    name: str
    body: list  # replacement tokens
    params: list | None = None  # None == object-like; [] == nullary function-like
    variadic: bool = False


def _tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(s)


def _is_id(t: str) -> bool:
    return bool(t) and (t[0].isascii() and t[0].isalpha() or t[0] == "_")


def _digit(c: str) -> bool:
    """An ASCII decimal digit -- `str.isdigit` takes `٣` and `²` too (CF-PPLIMITS)."""
    return "0" <= c <= "9" and len(c) == 1


#: The one reason each rail gives for a macro name, and for a macro parameter, longer than
#: `IDENT_MAX` (CF-LIMITS): C11 5.2.4.1's 63 significant initial characters, the bound both lexers
#: hold an identifier to (CF-BUF). A macro name never reaches a lexer -- the preprocessor replaces
#: it -- so it is bounded where a directive reads it. The twin's preprocessor (`bcir_cpp.c`) holds a
#: name in 64 bytes and gives these reasons.
MACRO_NAME_TOO_LONG = "macro name is too long"
MACRO_PARAM_TOO_LONG = "macro parameter is too long"
_NEEDS_NAME = "conditional directive requires an identifier"
#: A name as the tokenizer reads one (`_TOKEN_RE`), so a directive reads the name its text spells: ASCII, the
#: one the twin's `macro_name` reads.
_NAME_RE = re.compile(r"[ \t]*([A-Za-z_]\w*)", re.ASCII)


def _macro_name(text: str, why: str) -> tuple[str, int]:
    """The macro name a directive reads at the start of `text` -- the one `#define` or `-D` defines,
    `#undef` removes, `#ifdef`/`#ifndef`/`#elifdef`/`#elifndef`/`defined` tests -- and the index just
    past it: the identifier there (`_NAME_RE`). A CPPError with `why`, the directive's own reason
    (the twin's), when none starts there, with MACRO_NAME_TOO_LONG when it is longer than IDENT_MAX, and
    with NONASCII when a non-ASCII character runs on from it -- `#ifdef café` beside `#define caf 5` had
    tested `café` here and `caf` on the twin (CF-PPLIMITS)."""
    m = _NAME_RE.match(text)
    if m is None:
        raise CPPError(why)
    if len(m.group(1)) > IDENT_MAX:
        raise CPPError(MACRO_NAME_TOO_LONG)
    if not text[m.end() : m.end() + 1].isascii():
        raise CPPError(NONASCII)
    return m.group(1), m.end()


#: The most parameters a function-like macro takes (the twin's `Macro.params`), and a parameter's name.
_MAX_PARAMS = 16
_PARAM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _params(rest: str, i: int) -> tuple[list[str], int]:
    """The parameter list of a function-like `#define` from just past its `(` at `rest[i]`, and the index just past
    its `)`: names separated by commas, with spaces and tabs around them, a last `...` (`__VA_ARGS__`). Read by the
    twin's grammar, for its reasons (CF-PPLIMITS) -- the oracle had split the text at commas and taken whatever lay
    between them as a parameter, so `F(a b)`, `F(1)`, `F(a, a)`, `F(a` and seventeen parameters defined a macro here
    that the twin refused, and `F(café)` one named `café`."""
    params: list[str] = []

    def skip(k: int) -> int:
        while rest[k : k + 1] in (" ", "\t"):
            k += 1
        return k

    while True:
        i = skip(i)
        if rest[i : i + 1] == ")":
            return params, i + 1
        if params:
            if rest[i : i + 1] != ",":
                raise CPPError("invalid macro parameter list")
            i = skip(i + 1)
            if i >= len(rest) or rest[i] == ")":
                raise CPPError("invalid macro parameter list")
        if len(params) >= _MAX_PARAMS:
            raise CPPError("too many macro parameters")
        if rest.startswith("...", i):
            params.append("__VA_ARGS__")
            i = skip(i + 3)
            if rest[i : i + 1] != ")":
                raise CPPError("variadic macro parameter must be last")
            return params, i + 1
        if not rest[i : i + 1].isascii():
            raise CPPError(NONASCII)
        m = _PARAM_RE.match(rest, i)
        if m is None:
            raise CPPError("invalid macro parameter")
        if len(m.group()) > IDENT_MAX:
            raise CPPError(MACRO_PARAM_TOO_LONG)
        if m.group() in params:
            raise CPPError("duplicate macro parameter")
        params.append(m.group())
        i = m.end()
        if not rest[i : i + 1].isascii():
            raise CPPError(NONASCII)


def _bound_names(toks: list[str]) -> list[str]:
    """`toks`, the tokens of an evaluated `#if`/`#elif`, refused when a name it looks up -- a macro
    name, though one too long to have been defined -- is longer than IDENT_MAX (the twin reads them
    before and after expansion)."""
    for t in toks:
        m = _NAME_RE.match(t)
        if m is not None and len(m.group(1)) > IDENT_MAX:
            raise CPPError(MACRO_NAME_TOO_LONG)
    return toks


class Preprocessor:
    def __init__(
        self,
        includes: dict | None = None,
        embeds: dict | None = None,
        search_paths: list | None = None,
        defines: dict | None = None,
        abi: TargetABI | None = None,
    ):
        self.includes = includes or {}  # name -> header text (the in-memory mount)
        self.abi = abi or HOST  # the target whose character types a `#if` reads (CF-PPARITH)
        self.embeds = embeds or {}  # name -> bytes
        self.search_paths = list(search_paths or [])  # -I dirs (+ the source dir): on-disk headers
        self._disk_cache: dict[str, str | None] = {}  # resolved header name -> text (or None)
        self.macros: dict[str, Macro] = {n: Macro(n, _tokens(v)) for n, v in _PREDEFINED.items()}
        date, clock = _translation_datetime()  # __DATE__/__TIME__: object macros (string bodies)
        self.macros["__DATE__"] = Macro("__DATE__", _tokens(f'"{date}"'))
        self.macros["__TIME__"] = Macro("__TIME__", _tokens(f'"{clock}"'))
        for n, v in (defines or {}).items():  # -D name[=value]  (value "" -> defined as 1)
            _macro_name(n, "macro name must be an identifier")  # read as `#define` reads it
            self.macros[n] = Macro(n, _tokens(str(v) if v != "" else "1"))
        self._depth = 0
        self._cur_file = "<source>"  # __FILE__: the file currently being processed
        self._cur_line = 0  # __LINE__: its presumed line number
        self._presumed = 0  # presumed line of the *next* line (#line sets it)
        self._incstack: list = []  # active #include sites: (including_file, line), outer-first
        self.linemap: list = []  # per output line: (file, line, include-stack snapshot)
        self.dep_paths: list = []  # on-disk headers resolved by _resolve, in first-hit order
        #   (the -M/-MM dependency-output source, Phase 3)

    # --- public ---
    def process(self, text: str, name: str = "<source>") -> str:
        # a NUL: the twin's drivers refuse it (`READ-ERR`), for a C string ends there (CF-PPLIMITS)
        if "\0" in text:
            raise CPPError("the source holds a NUL byte")
        out: list[str] = []
        self._incstack, self.linemap = [], []
        self._run(self._logical_lines(text), out, name)
        return "\n".join(out) + "\n"

    def _out(self, out: list, text: str) -> None:
        """Append a finished output line, recording its provenance (origin file + presumed line + the
        active #include stack) in `linemap` so a diagnostic offset maps back to where it really is."""
        out.append(text)
        self.linemap.append((self._cur_file, self._cur_line, tuple(self._incstack)))

    # --- header resolution: the in-memory mount first, then the on-disk search path ---
    def _resolve(self, target: str) -> str | None:
        """The header text for `target`: the mounted map, else searched on disk (-I dirs + source
        dir), else None. Quoted and angle includes share the search path here (a driver MVP)."""
        if target in self.includes:
            return self.includes[target]
        if target in self._disk_cache:
            return self._disk_cache[target]
        import os  # noqa: PLC0415

        text = None
        for d in self.search_paths:
            p = os.path.join(d, target)
            if os.path.isfile(p):
                try:
                    text = read_bounded_text(p, "C header", max_bytes=16 << 20)
                except ValueError as exc:
                    raise CPPError(str(exc)) from exc
                self.dep_paths.append(p)  # first on-disk hit -> a build dependency (-M/-MM)
                break
        self._disk_cache[target] = text
        return text

    # --- line handling ---
    @staticmethod
    def _logical_lines(text: str) -> list[str]:
        # phase 2 then phase 3: splice backslash-newlines, then replace every comment with a space
        # (before directives are scanned, so a `#define` *inside* a comment is never executed and a
        # comment in a macro body / on a directive line is gone). Runs for the source and headers.
        return _strip_comments(text.replace("\\\n", "")).splitlines()

    def _run(self, lines: list[str], out: list[str], name: str) -> None:
        # conditional stack: each entry is [currently_active, any_branch_taken, parent_active]
        cond: list[list[bool]] = []

        def active() -> bool:
            return all(c[0] for c in cond)

        # __FILE__/__LINE__ track the current file + presumed line; the line starts at 1 and counts
        # up, but `#line` can reset both, and a nested #include saves/restores them (each file numbers
        # from 1 in its own name).
        saved = (self._cur_file, self._cur_line, self._presumed)
        self._cur_file, self._presumed = name, 1
        try:
            for raw in lines:
                self._cur_line = self._presumed
                self._presumed += 1
                line = raw.lstrip(_DIRECTIVE_SPACE)
                if line.startswith("#"):
                    self._directive(line[1:].lstrip(_DIRECTIVE_SPACE), cond, out, name, active)
                    continue
                if active():
                    self._out(out, self._expand_text(raw))
        finally:
            self._cur_file, self._cur_line, self._presumed = saved
        if cond:
            raise CPPError(f"unterminated #if in {name}")

    def _directive(self, d: str, cond, out, name, active) -> None:
        op = _DIRECTIVE_NAME.match(d).group()
        rest = d[len(op) :].strip(_ASCII_SPACE)
        parent = all(c[0] for c in cond)
        # a name running on past ASCII -- `#café`, `#\u00a0define` -- in a group read: the twin's reason
        if not d[len(op) : len(op) + 1].isascii() and active():
            raise CPPError(NONASCII)

        if op in ("ifdef", "ifndef", "if", "elifdef", "elifndef", "elif", "else", "endif"):
            self._conditional(op, rest, cond, parent)
            return
        if not active():
            return  # skip directives in an inactive branch
        if op == "define":
            self._define(rest)
        elif op == "undef":
            self.macros.pop(_macro_name(rest, "#undef requires an identifier")[0], None)
        elif op == "include":
            self._include(rest, out, name)
        elif op == "embed":
            self._out(out, self._embed(rest))
        elif op == "line":
            self._line(rest)
        elif op in ("error",):
            raise CPPError(f"#error {self._expand_text(rest)} (in {name})")
        elif op in ("warning", "pragma", ""):
            pass  # accepted, no effect on lowering
        else:
            raise CPPError(f"unknown directive #{op} in {name}")

    def _conditional(self, op, rest, cond, parent) -> None:
        # A directive in a skipped group is processed only through its name (C11 6.10.1p6), as is
        # an `#elif...` after a group was taken: its operand is never read, so no macro name or
        # expression in it can refuse the unit.
        if op in ("ifdef", "ifndef", "if"):
            if op == "ifdef":
                taken = parent and self._defined(_macro_name(rest, _NEEDS_NAME)[0])
            elif op == "ifndef":
                taken = parent and not self._defined(_macro_name(rest, _NEEDS_NAME)[0])
            else:
                taken = parent and self._eval(rest) != 0
            cond.append([parent and taken, taken, parent])
        elif op == "endif":
            if not cond:
                raise CPPError("#endif without #if")
            cond.pop()
        else:  # elif / elifdef / elifndef / else
            if not cond:
                raise CPPError(f"#{op} without #if")
            top = cond[-1]
            par = top[2]
            if op == "else":
                take = not top[1]
            elif op == "elifdef":
                take = (not top[1]) and par and self._defined(_macro_name(rest, _NEEDS_NAME)[0])
            elif op == "elifndef":
                take = (not top[1]) and par and not self._defined(_macro_name(rest, _NEEDS_NAME)[0])
            else:  # elif
                take = (not top[1]) and (par and self._eval(rest) != 0)
            top[0] = par and take
            top[1] = top[1] or take

    # --- #define ---
    def _define(self, rest: str) -> None:
        nm, nameend = _macro_name(rest, "macro name must be an identifier")
        if nameend < len(rest) and rest[nameend] == "(":  # function-like
            params, i = _params(rest, nameend + 1)
            variadic = bool(params) and params[-1] == "__VA_ARGS__"
            self.macros[nm] = Macro(nm, _tokens(rest[i:].strip(_ASCII_SPACE)), params, variadic)
        else:  # object-like
            self.macros[nm] = Macro(nm, _tokens(rest[nameend:].strip(_ASCII_SPACE)))

    # --- #line ---
    def _line(self, rest: str) -> None:
        """`#line digits ["file"]` — the presumed line number of the *following* line becomes
        `digits` (decimal), and __FILE__ becomes `"file"` if given. Operands are macro-expanded
        first (C23 6.10.5). Malformed directives are ignored (a strict compiler would diagnose)."""
        toks = self._expand(_tokens(rest), set())
        if not toks:
            return
        m = re.match(r"[0-9]+", toks[0])  # a decimal digit sequence
        if not m:
            return
        self._presumed = int(m.group())
        for t in toks[1:]:  # an optional new file name
            if t[:1] == '"':
                self._cur_file = _unescape_str(t)
                break

    # --- #include / #embed ---
    def _include(self, rest: str, out, name) -> None:
        rest_x = self._expand_text(rest).strip(_ASCII_SPACE)
        system = rest_x.startswith("<")
        target = self._header_name(rest)
        text = self._resolve(target)
        if text is None:
            if system:
                return  # an unmapped <system> header: the frontend
                #                                          models the standard types intrinsically.
            raise CPPError(
                f"#include {target!r} not found (in {name}); searched the mount + "
                f"{len(self.search_paths)} -I path(s)"
            )
        # a NUL in a header, in the twin's words (`bcir_cpp.c`); the tokenizer had dropped it (CF-PPLIMITS)
        if "\0" in text:
            raise CPPError(f"#include {target} contains NUL")
        if self._depth >= 64:
            raise CPPError("#include nesting too deep")
        self._depth += 1
        self._incstack.append((name, self._cur_line))  # the #include site, for the diagnostic frame
        try:
            self._run(self._logical_lines(text), out, target)
        finally:
            self._incstack.pop()
            self._depth -= 1

    def _embed(self, rest: str) -> str:
        target = self._header_name(rest.split()[0] if rest else rest)
        data = self.embeds.get(target)
        if data is None:
            raise CPPError(f"#embed {target!r} not found")
        return ", ".join(str(b) for b in data)

    def _header_name(self, rest: str) -> str:
        rest = self._expand_text(rest).strip(_ASCII_SPACE)
        if rest.startswith(("<", '"')):
            return rest[1:].split(">" if rest[0] == "<" else '"', 1)[0]
        return rest

    # --- macro expansion ---
    def _defined(self, name: str) -> bool:
        """Whether `name` is a defined macro for `#ifdef`/`defined()` — the macro table plus the
        dynamic predefined macros (`__FILE__`/`__LINE__`) and the `__has_*` feature-test operators,
        none of which are stored in the table."""
        return name in self.macros or name in _DYNAMIC or name in _HAS_OPS

    def _dynamic_value(self, t: str) -> str:
        """The expansion of a dynamic predefined macro at the current source position."""
        if t == "__LINE__":
            return str(self._cur_line)
        esc = self._cur_file.replace("\\", "\\\\").replace('"', '\\"')  # __FILE__: a string literal
        return f'"{esc}"'

    def _expand_text(self, text: str) -> str:
        return _join(self._expand(_tokens(text), set()))

    def _expand(self, toks: list[str], hide: set) -> list[str]:
        out: list[str] = []
        i = 0
        while i < len(toks):
            t = toks[i]
            if _is_id(t) and t in self.macros and t not in hide:
                mac = self.macros[t]
                if mac.params is None:  # object-like
                    # C 6.10.4.3: `##` pastes ANY two adjacent replacement-list tokens, not only ones
                    # adjacent to a parameter -- so an OBJECT-macro body `a##c` / `1##2` pastes too. Run
                    # the body through the SHARED substitute/paste path (with no args: `#`/`##` then see
                    # only literal tokens) so object + function macros use ONE paste engine, then rescan.
                    out += self._expand(self._substitute(mac, []), hide | {t})
                    i += 1
                    continue
                # function-like: needs a '(' next
                j = i + 1
                if j < len(toks) and toks[j] == "(":
                    args, j = self._collect_args(toks, j)
                    out += self._expand(self._substitute(mac, args), hide | {t})
                    i = j
                    continue
            elif _is_id(t) and t in _DYNAMIC:  # __FILE__ / __LINE__ (a #define wins)
                out.append(self._dynamic_value(t))
                i += 1
                continue
            elif t == "_Pragma" and i + 1 < len(toks) and toks[i + 1] == "(":
                _, i = self._collect_args(toks, i + 1)  # _Pragma("..."): a lowering no-op
                continue  # (like #pragma) — consume, emit nothing
            out.append(t)
            i += 1
        return out

    @staticmethod
    def _collect_args(toks, j):
        depth, args, cur = 0, [], []
        assert toks[j] == "("
        j += 1
        while j < len(toks):
            t = toks[j]
            if t == "(":
                depth += 1
                cur.append(t)
            elif t == ")":
                if depth == 0:
                    args.append(cur)
                    return args, j + 1
                depth -= 1
                cur.append(t)
            elif t == "," and depth == 0:
                args.append(cur)
                cur = []
            else:
                cur.append(t)
            j += 1
        raise CPPError("unterminated macro argument list")

    def _substitute(self, mac: Macro, args: list) -> list:
        params = mac.params or []
        # C11/C23 6.10.4.1: a parameter has TWO replacement forms. When it is an operand of `#` or `##`
        # the RAW (unexpanded) argument tokens are used; everywhere else the argument is COMPLETELY
        # macro-expanded *first* (argument prescan) and that expansion is substituted. We build both maps:
        # `amap` (raw) feeds `#`/`##`; `eamap` (prescanned) feeds the ordinary substitution -- this is what
        # makes the classic two-level `XSTR(__LINE__)` -> the line NUMBER, and an argument that is itself a
        # macro call (`INC(VAL)` with `VAL == INC(5)`) expand all the way down.
        amap: dict[str, list] = {}
        eamap: dict[str, list] = {}
        for k, p in enumerate(params):
            if p == "__VA_ARGS__":
                rest = args[k:] if k < len(args) else []
                flat: list = []
                for n, a in enumerate(rest):
                    if n:
                        flat.append(",")
                    flat += a
                amap[p] = flat
            else:
                amap[p] = args[k] if k < len(args) else []
            eamap[p] = self._expand(
                list(amap[p]), set()
            )  # prescan: full expansion in a fresh context
        return self._subst_tokens(list(mac.body), amap, eamap)

    def _subst_tokens(self, body: list, amap: dict, eamap: dict | None = None) -> list:
        # `amap` = raw arguments (for `#`/`##`); `eamap` = prescanned arguments (for plain substitution).
        # `eamap is None` -> a nested `__VA_OPT__` body re-enters with the same pair (passed through below).
        if eamap is None:
            eamap = amap
        out: list = []
        i = 0
        while i < len(body):
            t = body[i]
            if t == "__VA_OPT__" and i + 1 < len(body) and body[i + 1] == "(":  # C23 __VA_OPT__
                content, i = _balanced(body, i + 1)
                if amap.get("__VA_ARGS__"):  # __VA_ARGS__ non-empty -> the content
                    out += self._subst_tokens(content, amap, eamap)
                continue
            if t == "#" and i + 1 < len(body) and body[i + 1] in amap:  # stringize (RAW arg)
                out.append(_stringize(amap[body[i + 1]]))
                i += 2
                continue
            if t == "##" and i + 1 < len(body):  # paste (RAW args; placemarkers)
                right = amap.get(body[i + 1], [body[i + 1]])
                left = out.pop() if out else ""
                # An empty operand acts as a placemarker (C 6.10.4.3): `a ## b` with an empty side yields
                # just the non-empty side and NO literal `##`. Gluing only happens when both inner tokens
                # exist; otherwise the surviving side passes through untouched. (Pasting against an empty
                # `out` -- the left arg expanded to nothing -- still elides the `##` instead of emitting it.)
                if left == "":
                    out += right  # empty left -> result is the right operand
                elif right:
                    out.append(left + right[0])  # glue the inner tokens, keep the right's tail
                    out += right[1:]
                else:
                    out.append(left)  # empty right -> result is the left operand
                i += 2
                continue
            out += eamap.get(t, [t])  # prescanned arg or literal
            i += 1
        return out

    # --- constant-expression evaluation (#if / #elif) ---
    def _defined_walk(self, expr: str) -> str:
        """`expr` with each `defined X` and `defined(X)` replaced by 1 or 0, read token by token as the twin's
        `eval_if` reads it: a name longer than IDENT_MAX refused where it stands, a non-ASCII character outside a
        literal refused (NONASCII), and `defined`'s operand read as `#ifdef` reads one (`_macro_name`) -- no name,
        or `(` and a name with no `)`, refused as `malformed defined operator`. Two regular expressions had matched
        only a well-formed `defined` and left a malformed one standing, to be refused as a malformed expression
        (CF-PPLIMITS). A `__has_*` operator's parenthesized operand is its own (the passes below read it); the twin
        skips it as this does."""
        out, last, pos = [], 0, 0
        while (m := _TOKEN_RE.search(expr, pos)) is not None:
            t, pos = m.group(), m.end()
            if _is_id(t) and len(t) > IDENT_MAX:
                raise CPPError(MACRO_NAME_TOO_LONG)
            if not t[0].isascii():
                raise CPPError(NONASCII)
            if t == "defined":
                rest = expr[pos:]
                j = len(rest) - len(rest.lstrip(" \t"))
                if rest[j : j + 1] == "(":
                    name, k = _macro_name(rest[j + 1 :], _BAD_DEFINED)
                    close = _TOKEN_RE.search(rest, j + 1 + k)
                    if close is None or close.group() != ")":
                        raise CPPError(_BAD_DEFINED)
                    end = close.end()
                else:
                    name, end = _macro_name(rest, _BAD_DEFINED)
                out += [expr[last : m.start()], "1" if self._defined(name) else "0"]
                pos = last = m.end() + end
            elif t == "__has_include":
                # to the first `(` after it and the `)` that closes it, as the twin
                o = expr.find("(", pos)
                if o >= 0:
                    depth, e = 1, o + 1
                    while e < len(expr) and depth:
                        depth += {"(": 1, ")": -1}.get(expr[e], 0)
                        e += 1
                    pos = e
            elif t in _HAS_PAREN:  # a `(` next, then the tokens to the `)` that closes it
                o = _TOKEN_RE.search(expr, pos)
                if o is not None and o.group() == "(":
                    depth, pos = 1, o.end()
                    while depth and (a := _TOKEN_RE.search(expr, pos)) is not None:
                        pos = a.end()
                        depth += {"(": 1, ")": -1}.get(a.group(), 0)
        return "".join(out) + expr[last:]

    def _eval(self, expr: str) -> int:
        # handle defined / the __has_* operators BEFORE macro expansion, then expand. A `defined`
        # operand and every name the expression looks up, before and after expansion, are macro
        # names: each is bounded (CF-LIMITS).
        expr = self._defined_walk(expr)
        expr = re.sub(
            r"\b__has_include\s*\(([^)]*)\)",
            lambda m: "1" if self._resolve(self._header_name(m.group(1))) is not None else "0",
            expr,
        )
        expr = re.sub(
            r"\b__has_embed\s*\(([^)]*)\)",
            lambda m: "1" if self._header_name(m.group(1)) in self.embeds else "0",
            expr,
        )
        # feature-test macros for supported language features: only the L8 ABI attributes are
        # honoured today (no compiler builtins, no C23 [[...]] attributes), reported conservatively.
        expr = re.sub(
            r"\b__has_attribute\s*\(\s*(\w+)\s*\)",
            lambda m: "1" if m.group(1).strip("_") in _SUPPORTED_ATTRS else "0",
            expr,
            flags=re.ASCII,
        )
        expr = re.sub(r"\b__has_builtin\s*\([^)]*\)", "0", expr)
        expr = re.sub(r"\b__has_c_attribute\s*\([^)]*\)", "0", expr)
        toks = self._expand(_bound_names(_tokens(expr)), set())
        return _ConstEval(toks, self.abi).parse()


def _translation_datetime() -> tuple[str, str]:
    """The `__DATE__` ("Mmm dd yyyy", space-padded day) and `__TIME__` ("hh:mm:ss") strings. Frozen
    from `SOURCE_DATE_EPOCH` (the reproducible-builds convention, interpreted as UTC) when it is a
    plain integer, else the current UTC time. The C twin shares the exact convention, so the
    dual-rail output is byte-identical whenever the epoch is pinned."""
    import os, time  # noqa: PLC0415,E401

    epoch = os.environ.get("SOURCE_DATE_EPOCH", "")
    tm = time.gmtime(int(epoch)) if epoch.isascii() and epoch.isdigit() else time.gmtime()
    mon = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")[
        tm.tm_mon - 1
    ]
    return (
        f"{mon} {tm.tm_mday:2d} {tm.tm_year}",
        f"{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}",
    )


def _balanced(toks: list, k: int) -> tuple[list, int]:
    """The tokens strictly inside the parenthesis at ``toks[k] == '('`` and the index just past the
    matching ``')'`` (used for ``__VA_OPT__(...)``)."""
    j, depth, inner = k + 1, 1, []
    while j < len(toks):
        t = toks[j]
        if t == "(":
            depth += 1
        elif t == ")":
            depth -= 1
            if depth == 0:
                return inner, j + 1
        inner.append(t)
        j += 1
    return inner, j


_DIGITS = frozenset("0123456789")
_IDSTART = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_IDCHARS = _IDSTART | _DIGITS


def _strip_comments(text: str) -> str:
    """Translation phase 3: replace every `//` and `/* */` comment with a single space, leaving
    string/char literals untouched. A block comment keeps the newlines it spanned, so `__LINE__`
    and the per-line directive scan stay aligned with the source. An identifier and a preprocessing number (6.4.2,
    6.4.8) are copied whole, so a C23 digit separator (`1'000`, `0xca'fe`) -- a `'` inside a pp-number -- is not
    mistaken for a char-literal quote, and the quote after an identifier is one: a `'` flanked by hex digits had been
    taken for a separator in `u8'a'` (and `case'a'`), the constant's closing quote then opening a literal that ran
    past the next comment, which survived as tokens (CF-CONSTEXPR2; the twin's `strip_comments`)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in _IDCHARS or (c == "." and i + 1 < n and text[i + 1] in _DIGITS):
            j, num = i + 1, c not in _IDSTART  # a pp-number begins with a digit or `.digit`
            while j < n:
                ch = text[j]
                if num and ch in "eEpP" and j + 1 < n and text[j + 1] in "+-":
                    j += 2  # an exponent's sign
                elif num and ch == "'" and j + 1 < n and text[j + 1] in _IDCHARS:
                    j += 2  # a digit separator
                elif ch in _IDCHARS or (num and ch == "."):
                    j += 1
                else:
                    break
            out.append(text[i:j])
            i = j
            continue
        if c in "\"'":
            out.append(c)  # a string/char literal: copy verbatim
            i += 1
            while i < n:
                ch = text[i]
                if ch == "\\" and i + 1 < n:  # an escape: copy the pair (incl. \" \')
                    out.append(ch)
                    out.append(text[i + 1])
                    i += 2
                    continue
                out.append(ch)
                i += 1
                if ch == c:  # the closing quote
                    break
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":  # line comment -> one space
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            out.append(" ")
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":  # block comment -> space + kept \n's
            i += 2
            nl = 0
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                if text[i] == "\n":
                    nl += 1
                i += 1
            i += 2  # consume the closing */
            out.append(" " + "\n" * nl)
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _unescape_str(tok: str) -> str:
    """The bytes of a `"..."` string-literal token (drop the quotes, resolve `\\` escapes)."""
    body = tok[1:-1] if len(tok) >= 2 and tok[-1] == '"' else tok[1:]
    out, i = [], 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            out.append(body[i + 1])
            i += 2
        else:
            out.append(body[i])
            i += 1
    return "".join(out)


# Two characters that, side by side, may begin a punctuator longer than the token they end (C 6.4.6):
# the first two of a multi-character one (the ellipsis's `..` is its own rule, in `_pastes`). Whether it
# does is maximal munch's to say (6.4p4): `+` then `++`, written with no space, lexes as `++` `+`, so
# `a + ++g` spelled `a+++g` is `(a++) + g`; `--` then `>` lexes as `--` `>` again. The twin keeps the
# same set (`bcir_cpp.c`, `pastes`).
_PASTE_PAIRS = frozenset(
    ("<<", "<=", ">>", ">=", "->", "++", "--", "==", "!=", "&&", "||", "##",
     "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=")
)  # fmt: skip
_PUNCT_RE = re.compile("|".join(re.escape(p) for p in _PUNCT))  # longest first: maximal munch


def _is_ppnum(t: str) -> bool:
    """Whether the preprocessing token `t` is a pp-number (C 6.4.8): a digit, or `.` and a digit, first."""
    return _digit(t[:1]) or (t[:1] == "." and _digit(t[1:2]))


def _pastes(prev: str, t: str) -> bool:
    """Whether the token `t`, written right after the token `prev`, would lex as part of another token:
    two words run together, a comment opens (`y / *p` spelled `y/*p`, C 6.4.9), a punctuator runs on
    into a longer one, two `.` start an ellipsis (`..` is no punctuator, so maximal munch over two
    tokens cannot see it), a pp-number runs on through a `.` or an exponent's sign, or a `.` begins a
    number. A space between them keeps them two (the twin's `pastes`, byte for byte)."""
    a, b = prev[-1], t[0]
    if (_is_id(a) or _digit(a)) and (_is_id(b) or _digit(b)):
        return True
    if (a == "/" and b in "*/") or (a == "." and b == "."):
        return True
    if a + b in _PASTE_PAIRS:
        m = _PUNCT_RE.match(prev + t[:2])
        if m is not None and len(m.group()) > len(prev):
            return True
    if _is_ppnum(prev) and (b == "." or (a in "eEpP" and b in "+-")):
        return True
    if b == "'" and a in "LuU8":  # a word that may be an encoding prefix (CF-PPARITH)
        return True
    return a == "." and _digit(b)


def _join(toks: list[str]) -> str:
    """The tokens as one line of text, with a space only between two that would otherwise lex as
    another (`_pastes`), so the text re-lexes to exactly these tokens."""
    out, prev = "", ""
    for t in toks:
        if not t:
            continue
        if prev and _pastes(prev, t):
            out += " "
        out += t
        prev = t
    return out


def _stringize(toks: list[str]) -> str:
    """The `#`-operator spelling of an argument (C 6.10.4.2): the argument's preprocessing tokens joined
    with single spaces, leading/trailing whitespace removed, wrapped in `"..."`. A `\\` is inserted before
    each `"` and `\\` *that is part of a character-constant or string-literal token* -- and ONLY there: a
    bare backslash floating in the token stream (`S(a\\b)` -> `"a\\b"`, like clang/gcc) is NOT doubled, but
    one inside a literal is (`S("q")` -> `"\\"q\\""`, `S("c:\\t")` -> `"\\"c:\\\\t\\""`). So the escape is
    applied per-token to literals, not blanket to the joined text."""
    esc = []
    for t in toks:
        if (t[:1] in ('"', "'") or _CHAR_TOKEN.match(t)) and len(
            t
        ) >= 2:  # a literal -> escape \ and "
            esc.append(t.replace("\\", "\\\\").replace('"', '\\"'))
        else:  # any other token, incl. a bare `\`, verbatim
            esc.append(t)
    return '"' + _join(esc) + '"'


#: The reasons both rails refuse a `#if`/`#elif` expression for (CF-PPARITH; the twin's `bcir_cpp.c` gives the same
#: words). One the grammar does not admit -- nothing, a token no operand or operator is (a string, a floating
#: constant, `=`), two operands with no operator between them, an unclosed `(`, a `?` with no `:` -- is malformed,
#: wherever it stands. The rest are C's undefined behaviour or constraint violations in an evaluated operand (6.6p3,
#: 6.6p4, 6.5.5p5, 6.5.7p3-4): a signed result past `intmax_t`, a division by zero, a shift by a negative count or
#: by 64 or more, a negative value shifted left, a comma operator. An operand C leaves unevaluated -- the right of
#: `&&` after 0 and of `||` after a nonzero, the arm of `?:` not taken -- refuses for none of them.
PP_MALFORMED = "malformed #if expression"
PP_OVERFLOW = "integer overflow in #if expression"
PP_SHIFT = "invalid shift in #if expression"
PP_DIVZERO = "division by zero in #if expression"
PP_COMMA = "comma operator in #if expression"
#: The bounds the twin holds a `#if` to, held here too: 512 tokens after expansion, each at most 63 characters, and
#: at most 63 nested parentheses and conditional operators (C11 5.2.4.1's minimum).
PP_TOO_MANY = "too many tokens in #if expression"
PP_SPELLING_LONG = "preprocessor token too long"
PP_DEEP = "#if expression nested too deeply"
_PP_MAX_TOKENS, _PP_MAX_TOKEN, _PP_MAX_DEPTH = 512, 63, 63

_M64 = (1 << 64) - 1
_IMAX = (1 << 63) - 1
_CHAR_TOKEN = re.compile(r"(?:u8|[LuU])?'")


@dataclass(frozen=True)
class _PPValue:
    """A `#if` operand (C11 6.10.1p4): its 64 bits, and whether it is a `uintmax_t` (else an `intmax_t`)."""

    bits: int
    unsigned: bool

    def signed(self) -> int:
        return self.bits - (1 << 64) if self.bits >> 63 else self.bits

    def value(self) -> int:
        return self.bits if self.unsigned else self.signed()


_ZERO, _ONE = _PPValue(0, False), _PPValue(1, False)


class _ConstEval:
    """The `#if`/`#elif` constant expression (C11 6.10.1), evaluated as C says: every operand an `intmax_t` or a
    `uintmax_t` -- an integer constant unsigned where its suffix has `u` or its value is past `INTMAX_MAX`, a
    character constant by the target's character types, `true` 1 and any other identifier 0 -- and each operator by
    the usual arithmetic conversions (an unsigned operand makes both unsigned; a shift takes its left operand's
    type; `!`, a comparison, `&&` and `||` give an `int`). Python's unbounded `int` had read `-1 > 0u` as false and
    `0xFFFFFFFFFFFFFFFF == -1` had raised a bare ValueError; a character constant had been 0, `?:` taken whole and an
    unclosed `(` or a trailing token ignored (CF-PPARITH; the twin's `ce_*`)."""

    _BINARY = {
        "||": 1, "&&": 2, "|": 3, "^": 4, "&": 5, "==": 6, "!=": 6, "<": 7, ">": 7, "<=": 7, ">=": 7,
        "<<": 8, ">>": 8, "+": 9, "-": 9, "*": 10, "/": 10, "%": 10,
    }  # fmt: skip

    def __init__(self, toks: list[str], abi: TargetABI):
        for k, t in enumerate(
            toks
        ):  # each token in turn, as the twin reads them: its name, its length, its count
            _bound_names([t])
            if len(t) > _PP_MAX_TOKEN:
                raise CPPError(PP_SPELLING_LONG)
            if k >= _PP_MAX_TOKENS:
                raise CPPError(PP_TOO_MANY)
        self.t, self.i, self.abi = toks, 0, abi

    def parse(self) -> int:
        v = self._comma(True, 0)
        if self.i != len(self.t):
            raise CPPError(PP_MALFORMED)
        return int(v.bits != 0)

    def _peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def _expect(self, tok: str) -> None:
        if self._peek() != tok:
            raise CPPError(PP_MALFORMED)
        self.i += 1

    def _comma(self, live: bool, depth: int) -> _PPValue:
        v = self._cond(live, depth)
        while self._peek() == ",":
            if live:
                raise CPPError(PP_COMMA)
            self.i += 1
            v = self._cond(live, depth)
        return v

    def _cond(self, live: bool, depth: int) -> _PPValue:
        c = self._binary(1, live, depth)
        if self._peek() != "?":
            return c
        if depth >= _PP_MAX_DEPTH:
            raise CPPError(PP_DEEP)
        self.i += 1
        a = self._comma(live and c.bits != 0, depth + 1)
        self._expect(":")
        b = self._cond(live and c.bits == 0, depth + 1)
        return _PPValue((a if c.bits else b).bits, a.unsigned or b.unsigned)

    def _binary(self, minp: int, live: bool, depth: int) -> _PPValue:
        lhs = self._unary(live, depth)
        while True:
            op = self._peek()
            p = self._BINARY.get(op)
            if p is None or p < minp:
                return lhs
            self.i += 1
            rlive = live and not (op == "&&" and not lhs.bits) and not (op == "||" and lhs.bits)
            lhs = _apply(op, lhs, self._binary(p + 1, rlive, depth), live)

    def _unary(self, live: bool, depth: int) -> _PPValue:
        ops = []
        while self._peek() in ("+", "-", "~", "!"):
            ops.append(self.t[self.i])
            self.i += 1
        v = self._primary(live, depth)
        for op in reversed(ops):
            if op == "!":
                v = _PPValue(int(v.bits == 0), False)
            elif op == "~":
                v = _PPValue(~v.bits & _M64, v.unsigned)
            elif op == "-":
                if not v.unsigned and v.bits == 1 << 63 and live:
                    raise CPPError(PP_OVERFLOW)
                v = _PPValue(-v.bits & _M64, v.unsigned)
        return v

    def _primary(self, live: bool, depth: int) -> _PPValue:
        t = self._peek()
        if t is None:
            raise CPPError(PP_MALFORMED)
        self.i += 1
        if t == "(":
            if depth >= _PP_MAX_DEPTH:
                raise CPPError(PP_DEEP)
            v = self._comma(live, depth + 1)
            self._expect(")")
            return v
        if _digit(t[:1]):
            return _int_lit(t)
        if _CHAR_TOKEN.match(t):
            return _char_lit(t, self.abi)
        if _NAME_RE.fullmatch(t):
            return (
                _ONE if t == "true" else _ZERO
            )  # C23 6.10.1p11: `true` is 1, any other identifier 0
        raise CPPError(PP_MALFORMED)


def _int_lit(t: str) -> _PPValue:
    """An integer constant of a `#if`, read as the lexer reads one (`clex.parse_int_literal`): its digits checked
    against its base and its suffix checked, a malformed one (`08`, `1lL`) or one no type holds refused as the lexer
    refuses it, in an operand C leaves unevaluated too, for it is no constant at all (CF-SUFFIX). It is a
    `uintmax_t` where its suffix has `u` or its value is past `INTMAX_MAX` (6.10.1p4; a hexadecimal, octal or binary
    one: 6.4.4.1p5), else an `intmax_t` (CF-PPARITH)."""
    try:
        value = parse_int_literal(t)
        _value, _decimal, suffix = int_literal_parts(t)
    except CLexError as e:
        raise CPPError(str(e)) from None
    return _PPValue(value, "u" in suffix.lower() or value > _IMAX)


def _char_lit(t: str, abi) -> _PPValue:
    """A character constant of a `#if` (6.10.1p4), read as Clang and GCC read one: a plain one by the target's plain
    `char` -- its byte sign-extended, or zero-extended and the operand a `uintmax_t` where `char` is unsigned (so
    `'a' - 98 < 0` is false on AArch64); a multi-character one packed big-endian into an `int`'s 32 bits and
    extended likewise -- a `u8` one an `unsigned char`, a `u` and a `U` one `char16_t` and `char32_t`, unsigned, an
    `L` one the target's `wchar_t`. Refused for `clex.CHAR_UNSUPPORTED` in any operand (CF-PPARITH)."""
    bits = 8 * abi.wchar_size
    try:
        prefix, units = char_constant_units(t, bits)
    except CLexError as e:
        raise CPPError(str(e)) from None
    if prefix == "":
        bits, signed, v = 8, abi.char_signed, units[0]
        if len(units) > 1:
            bits, v = 32, 0
            for u in units:
                v = ((v << 8) | u) & 0xFFFFFFFF
    elif prefix == "L":
        signed, v = abi.wchar_signed, units[0]
    else:
        bits, signed, v = {"u8": 8, "u": 16, "U": 32}[prefix], False, units[0]
    if signed and v >> (bits - 1):
        v -= 1 << bits
    return _PPValue(v & _M64, not signed)


def _apply(op: str, a: _PPValue, b: _PPValue, live: bool) -> _PPValue:
    """`a op b` in C's `#if` arithmetic; `live` where C evaluates it, so its undefined behaviour refuses the unit
    (an unevaluated one's value is never read, but its type still decides a `?:`'s)."""
    if op in ("||", "&&"):
        return _PPValue(
            int(bool(a.bits) or bool(b.bits)) if op == "||" else int(bool(a.bits) and bool(b.bits)),
            False,
        )
    if op in ("<<", ">>"):
        n = b.value()
        if not 0 <= n < 64:
            if live:
                raise CPPError(PP_SHIFT)
            return _PPValue(0, a.unsigned)
        if op == ">>":
            return _PPValue((a.bits if a.unsigned else a.signed()) >> n & _M64, a.unsigned)
        if not a.unsigned and live and (a.signed() < 0 or a.signed() << n > _IMAX):
            raise CPPError(PP_OVERFLOW)
        return _PPValue(a.bits << n & _M64, a.unsigned)
    u = a.unsigned or b.unsigned
    x, y = (a.bits, b.bits) if u else (a.signed(), b.signed())
    if op in ("==", "!=", "<", ">", "<=", ">="):
        return _PPValue(
            int(
                {"==": x == y, "!=": x != y, "<": x < y, ">": x > y, "<=": x <= y, ">=": x >= y}[op]
            ),
            False,
        )
    if op in ("&", "|", "^"):
        return _PPValue({"&": x & y, "|": x | y, "^": x ^ y}[op] & _M64, u)
    if op in ("/", "%"):
        if y == 0:
            if live:
                raise CPPError(PP_DIVZERO)
            return _PPValue(0, u)
        q = abs(x) // abs(y) * (1 if (x < 0) == (y < 0) else -1)  # C truncates toward zero
        if (
            not u and q > _IMAX and live
        ):  # INTMAX_MIN / -1, whose remainder is undefined too (6.5.5p6)
            raise CPPError(PP_OVERFLOW)
        r = q if op == "/" else x - q * y
    else:
        r = {"+": x + y, "-": x - y, "*": x * y}[op]
    if not u and not -(1 << 63) <= r <= _IMAX and live:
        raise CPPError(PP_OVERFLOW)
    return _PPValue(r & _M64, u)


def preprocess(
    text: str,
    *,
    includes: dict | None = None,
    embeds: dict | None = None,
    search_paths: list | None = None,
    defines: dict | None = None,
    name: str = "<source>",
    return_map: bool = False,
    abi: TargetABI | None = None,
):
    """Preprocess `text` to the flat translation-unit string. With `return_map=True`, also return the
    per-output-line provenance map (file, line, #include stack) so diagnostics resolve to their
    origin file even across inlined includes. `abi` is the target whose character types a `#if` reads (the host's,
    x86-64 Linux, when None; CF-PPARITH)."""
    p = Preprocessor(includes, embeds, search_paths, defines, abi)
    out = p.process(text, name)
    return (out, p.linemap) if return_map else out
