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

_PREDEFINED = {"__STDC__": "1", "__STDC_VERSION__": "202311L", "__STDC_HOSTED__": "1"}
# dynamic predefined macros: expanded per source position, not stored as static bodies.
_DYNAMIC = ("__FILE__", "__LINE__")
# feature-test operators usable in #if (and reported as `defined`); evaluated in _eval.
_HAS_OPS = ("__has_include", "__has_embed", "__has_attribute", "__has_builtin", "__has_c_attribute")
_SUPPORTED_ATTRS = frozenset({"packed", "aligned"})    # attributes the L8 ABI honours (GCC __x__ ok)
# preprocessing tokens: identifier, number, string, char, or punctuation (multi-char first).
_PUNCT = ["<<=", ">>=", "...", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
          "##", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
          "#", "+", "-", "*", "/", "%", "&", "|", "^", "~", "!", "<", ">", "=",
          "(", ")", "{", "}", "[", "]", ";", ",", ".", ":", "?"]
_TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"'                          # string
    r"|'(?:\\.|[^'\\])*'"                         # char
    r"|\.?\d(?:[eEpP][-+]|[\w.'])*"               # pp-number (C23 ' seps; ints, hex/bin, floats w/ exp+suffix)
    r"|[A-Za-z_]\w*"                             # identifier
    r"|" + "|".join(re.escape(p) for p in _PUNCT) +
    r"|\\")                                       # a stray backslash is its own token (e.g. a path in a
    #                                               stringize arg `#x` -> `"C:\\tmp"`); it would otherwise be
    #                                               dropped, corrupting the `#`-spelling.


class CPPError(Exception):
    """A preprocessing error. `pos` is a source byte offset when known (else a file-level banner)."""
    def __init__(self, message: str, pos: int | None = None):
        super().__init__(message)
        self.pos = pos


@dataclass
class Macro:
    name: str
    body: list                    # replacement tokens
    params: list | None = None    # None == object-like; [] == nullary function-like
    variadic: bool = False


def _tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(s)


def _is_id(t: str) -> bool:
    return bool(t) and (t[0].isalpha() or t[0] == "_")


class Preprocessor:
    def __init__(self, includes: dict | None = None, embeds: dict | None = None,
                 search_paths: list | None = None, defines: dict | None = None):
        self.includes = includes or {}            # name -> header text (the in-memory mount)
        self.embeds = embeds or {}                # name -> bytes
        self.search_paths = list(search_paths or [])   # -I dirs (+ the source dir): on-disk headers
        self._disk_cache: dict[str, str | None] = {}   # resolved header name -> text (or None)
        self.macros: dict[str, Macro] = {
            n: Macro(n, _tokens(v)) for n, v in _PREDEFINED.items()}
        date, clock = _translation_datetime()    # __DATE__/__TIME__: object macros (string bodies)
        self.macros["__DATE__"] = Macro("__DATE__", _tokens(f'"{date}"'))
        self.macros["__TIME__"] = Macro("__TIME__", _tokens(f'"{clock}"'))
        for n, v in (defines or {}).items():      # -D name[=value]  (value "" -> defined as 1)
            self.macros[n] = Macro(n, _tokens(str(v) if v != "" else "1"))
        self._depth = 0
        self._cur_file = "<source>"               # __FILE__: the file currently being processed
        self._cur_line = 0                         # __LINE__: its presumed line number
        self._presumed = 0                         # presumed line of the *next* line (#line sets it)
        self._incstack: list = []                 # active #include sites: (including_file, line), outer-first
        self.linemap: list = []                   # per output line: (file, line, include-stack snapshot)

    # --- public ---
    def process(self, text: str, name: str = "<source>") -> str:
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
                with open(p, encoding="utf-8") as f:
                    text = f.read()
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
                line = raw.strip()
                if line.startswith("#"):
                    self._directive(line[1:].strip(), cond, out, name, active)
                    continue
                if active():
                    self._out(out, self._expand_text(raw))
        finally:
            self._cur_file, self._cur_line, self._presumed = saved
        if cond:
            raise CPPError(f"unterminated #if in {name}")

    def _directive(self, d: str, cond, out, name, active) -> None:
        op, _, rest = d.partition(" ")
        rest = rest.strip()
        parent = all(c[0] for c in cond)

        if op in ("ifdef", "ifndef", "if", "elifdef", "elifndef", "elif", "else", "endif"):
            self._conditional(op, rest, cond, parent)
            return
        if not active():
            return                                       # skip directives in an inactive branch
        if op == "define":
            self._define(rest)
        elif op == "undef":
            self.macros.pop(rest.split()[0], None) if rest else None
        elif op == "include":
            self._include(rest, out, name)
        elif op == "embed":
            self._out(out, self._embed(rest))
        elif op == "line":
            self._line(rest)
        elif op in ("error",):
            raise CPPError(f"#error {self._expand_text(rest)} (in {name})")
        elif op in ("warning", "pragma", ""):
            pass                                         # accepted, no effect on lowering
        else:
            raise CPPError(f"unknown directive #{op} in {name}")

    def _conditional(self, op, rest, cond, parent) -> None:
        if op in ("ifdef", "ifndef", "if"):
            if op == "ifdef":
                taken = self._defined(rest.split()[0])
            elif op == "ifndef":
                taken = not self._defined(rest.split()[0])
            else:
                taken = self._eval(rest) != 0
            cond.append([parent and taken, taken, parent])
        elif op == "endif":
            if not cond:
                raise CPPError("#endif without #if")
            cond.pop()
        else:                                            # elif / elifdef / elifndef / else
            if not cond:
                raise CPPError(f"#{op} without #if")
            top = cond[-1]
            par = top[2]
            if op == "else":
                take = not top[1]
            elif op == "elifdef":
                take = (not top[1]) and self._defined(rest.split()[0])
            elif op == "elifndef":
                take = (not top[1]) and not self._defined(rest.split()[0])
            else:                                        # elif
                take = (not top[1]) and (par and self._eval(rest) != 0)
            top[0] = par and take
            top[1] = top[1] or take

    # --- #define ---
    def _define(self, rest: str) -> None:
        m = re.match(r"(\w+)", rest)
        if not m:
            return
        nameend = m.end()
        nm = m.group(1)
        if nameend < len(rest) and rest[nameend] == "(":      # function-like
            depth, i = 0, nameend
            while i < len(rest):
                if rest[i] == "(":
                    depth += 1
                elif rest[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            params_src = rest[nameend + 1:i]
            body = rest[i + 1:].strip()
            params = [p.strip() for p in params_src.split(",") if p.strip()]
            variadic = bool(params) and params[-1] == "..."
            if variadic:
                params[-1] = "__VA_ARGS__"
            self.macros[nm] = Macro(nm, _tokens(body), params, variadic)
        else:                                                 # object-like
            self.macros[nm] = Macro(nm, _tokens(rest[nameend:].strip()))

    # --- #line ---
    def _line(self, rest: str) -> None:
        """`#line digits ["file"]` — the presumed line number of the *following* line becomes
        `digits` (decimal), and __FILE__ becomes `"file"` if given. Operands are macro-expanded
        first (C23 6.10.5). Malformed directives are ignored (a strict compiler would diagnose)."""
        toks = self._expand(_tokens(rest), set())
        if not toks:
            return
        m = re.match(r"\d+", toks[0])                         # a decimal digit sequence
        if not m:
            return
        self._presumed = int(m.group())
        for t in toks[1:]:                                    # an optional new file name
            if t[:1] == '"':
                self._cur_file = _unescape_str(t)
                break

    # --- #include / #embed ---
    def _include(self, rest: str, out, name) -> None:
        rest_x = self._expand_text(rest).strip()
        system = rest_x.startswith("<")
        target = self._header_name(rest)
        text = self._resolve(target)
        if text is None:
            if system:
                return                                   # an unmapped <system> header: the frontend
                #                                          models the standard types intrinsically.
            raise CPPError(f"#include {target!r} not found (in {name}); searched the mount + "
                           f"{len(self.search_paths)} -I path(s)")
        if self._depth > 64:
            raise CPPError("#include nesting too deep")
        self._depth += 1
        self._incstack.append((name, self._cur_line))    # the #include site, for the diagnostic frame
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
        rest = self._expand_text(rest).strip()
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
        esc = self._cur_file.replace("\\", "\\\\").replace('"', '\\"')   # __FILE__: a string literal
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
                if mac.params is None:                        # object-like
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
            elif _is_id(t) and t in _DYNAMIC:                 # __FILE__ / __LINE__ (a #define wins)
                out.append(self._dynamic_value(t))
                i += 1
                continue
            elif t == "_Pragma" and i + 1 < len(toks) and toks[i + 1] == "(":
                _, i = self._collect_args(toks, i + 1)        # _Pragma("..."): a lowering no-op
                continue                                      # (like #pragma) — consume, emit nothing
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
            eamap[p] = self._expand(list(amap[p]), set())     # prescan: full expansion in a fresh context
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
            if t == "__VA_OPT__" and i + 1 < len(body) and body[i + 1] == "(":   # C23 __VA_OPT__
                content, i = _balanced(body, i + 1)
                if amap.get("__VA_ARGS__"):                # __VA_ARGS__ non-empty -> the content
                    out += self._subst_tokens(content, amap, eamap)
                continue
            if t == "#" and i + 1 < len(body) and body[i + 1] in amap:     # stringize (RAW arg)
                out.append(_stringize(amap[body[i + 1]]))
                i += 2
                continue
            if t == "##" and i + 1 < len(body):                            # paste (RAW args; placemarkers)
                right = amap.get(body[i + 1], [body[i + 1]])
                left = out.pop() if out else ""
                # An empty operand acts as a placemarker (C 6.10.4.3): `a ## b` with an empty side yields
                # just the non-empty side and NO literal `##`. Gluing only happens when both inner tokens
                # exist; otherwise the surviving side passes through untouched. (Pasting against an empty
                # `out` -- the left arg expanded to nothing -- still elides the `##` instead of emitting it.)
                if left == "":
                    out += right                            # empty left -> result is the right operand
                elif right:
                    out.append(left + right[0])             # glue the inner tokens, keep the right's tail
                    out += right[1:]
                else:
                    out.append(left)                        # empty right -> result is the left operand
                i += 2
                continue
            out += eamap.get(t, [t])                                       # prescanned arg or literal
            i += 1
        return out

    # --- constant-expression evaluation (#if / #elif) ---
    def _eval(self, expr: str) -> int:
        # handle defined / the __has_* operators BEFORE macro expansion, then expand.
        expr = re.sub(r"\bdefined\s*\(\s*(\w+)\s*\)",
                      lambda m: "1" if self._defined(m.group(1)) else "0", expr)
        expr = re.sub(r"\bdefined\s+(\w+)",
                      lambda m: "1" if self._defined(m.group(1)) else "0", expr)
        expr = re.sub(r"\b__has_include\s*\(([^)]*)\)",
                      lambda m: "1" if self._resolve(self._header_name(m.group(1))) is not None
                      else "0", expr)
        expr = re.sub(r"\b__has_embed\s*\(([^)]*)\)",
                      lambda m: "1" if self._header_name(m.group(1)) in self.embeds else "0", expr)
        # feature-test macros for supported language features: only the L8 ABI attributes are
        # honoured today (no compiler builtins, no C23 [[...]] attributes), reported conservatively.
        expr = re.sub(r"\b__has_attribute\s*\(\s*(\w+)\s*\)",
                      lambda m: "1" if m.group(1).strip("_") in _SUPPORTED_ATTRS else "0", expr)
        expr = re.sub(r"\b__has_builtin\s*\([^)]*\)", "0", expr)
        expr = re.sub(r"\b__has_c_attribute\s*\([^)]*\)", "0", expr)
        toks = self._expand(_tokens(expr), set())
        return _ConstEval(toks).parse()


def _translation_datetime() -> tuple[str, str]:
    """The `__DATE__` ("Mmm dd yyyy", space-padded day) and `__TIME__` ("hh:mm:ss") strings. Frozen
    from `SOURCE_DATE_EPOCH` (the reproducible-builds convention, interpreted as UTC) when it is a
    plain integer, else the current UTC time. The C twin shares the exact convention, so the
    dual-rail output is byte-identical whenever the epoch is pinned."""
    import os, time  # noqa: PLC0415,E401
    epoch = os.environ.get("SOURCE_DATE_EPOCH", "")
    tm = time.gmtime(int(epoch)) if epoch.isdigit() else time.gmtime()
    mon = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")[tm.tm_mon - 1]
    return (f"{mon} {tm.tm_mday:2d} {tm.tm_year}",
            f"{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}")


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


_HEXD = frozenset("0123456789abcdefABCDEF")


def _strip_comments(text: str) -> str:
    """Translation phase 3: replace every `//` and `/* */` comment with a single space, leaving
    string/char literals untouched. A block comment keeps the newlines it spanned, so `__LINE__`
    and the per-line directive scan stay aligned with the source. A C23 digit separator (`1'000`,
    `0xca'fe`) is not mistaken for a char-literal quote — a `'` flanked by hex digits is data."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or (c == "'" and not (i and text[i - 1] in _HEXD
                                          and i + 1 < n and text[i + 1] in _HEXD)):
            out.append(c)                                      # a string/char literal: copy verbatim
            i += 1
            while i < n:
                ch = text[i]
                if ch == "\\" and i + 1 < n:                   # an escape: copy the pair (incl. \" \')
                    out.append(ch)
                    out.append(text[i + 1])
                    i += 2
                    continue
                out.append(ch)
                i += 1
                if ch == c:                                    # the closing quote
                    break
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":      # line comment -> one space
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            out.append(" ")
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":      # block comment -> space + kept \n's
            i += 2
            nl = 0
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                if text[i] == "\n":
                    nl += 1
                i += 1
            i += 2                                             # consume the closing */
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


def _join(toks: list[str]) -> str:
    out = ""
    for t in toks:
        if out and (_is_id(out[-1]) or out[-1].isdigit()) and (_is_id(t) or t[:1].isdigit()):
            out += " "
        out += t
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
        if t[:1] in ('"', "'") and len(t) >= 2:                # a string/char literal -> escape \ and "
            esc.append(t.replace("\\", "\\\\").replace('"', '\\"'))
        else:                                                  # any other token, incl. a bare `\`, verbatim
            esc.append(t)
    return '"' + _join(esc) + '"'


class _ConstEval:
    """A small integer constant-expression evaluator for #if (undefined identifiers -> 0)."""
    _LEVELS = [("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="),
               ("<", ">", "<=", ">="), ("<<", ">>"), ("+", "-"), ("*", "/", "%")]

    def __init__(self, toks: list[str]):
        self.t = toks
        self.i = 0

    def parse(self) -> int:
        v = self._ternary()
        return int(v)

    def _peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def _ternary(self) -> int:
        c = self._binary(0)
        if self._peek() == "?":
            self.i += 1
            a = self._ternary()
            if self._peek() == ":":
                self.i += 1
            b = self._ternary()
            return a if c else b
        return c

    def _binary(self, lvl: int) -> int:
        if lvl >= len(self._LEVELS):
            return self._unary()
        v = self._binary(lvl + 1)
        while self._peek() in self._LEVELS[lvl]:
            op = self.t[self.i]
            self.i += 1
            r = self._binary(lvl + 1)
            v = _apply(op, v, r)
        return v

    def _unary(self) -> int:
        t = self._peek()
        if t in ("!", "-", "+", "~"):
            self.i += 1
            v = self._unary()
            return {"!": int(not v), "-": -v, "+": +v, "~": ~v}[t]
        return self._primary()

    def _primary(self) -> int:
        t = self._peek()
        if t == "(":
            self.i += 1
            v = self._ternary()
            if self._peek() == ")":
                self.i += 1
            return v
        self.i += 1
        if t is None:
            return 0
        if t[:1].isdigit():
            return _int_lit(t)
        return 0                                          # undefined identifier -> 0


def _int_lit(t: str) -> int:
    t = t.replace("'", "")
    while t and t[-1] in "uUlL":
        t = t[:-1]
    if t[:2] in ("0x", "0X"):
        return int(t, 16)
    if t[:2] in ("0b", "0B"):
        return int(t[2:], 2)
    if len(t) > 1 and t[0] == "0" and t.isdigit():
        return int(t, 8)
    return int(t or "0", 10)


def _apply(op: str, a: int, b: int) -> int:
    if op == "/":
        return int(a / b) if b else 0
    if op == "%":
        return a % b if b else 0
    return {
        "||": int(bool(a) or bool(b)), "&&": int(bool(a) and bool(b)), "|": a | b, "^": a ^ b,
        "&": a & b, "==": int(a == b), "!=": int(a != b), "<": int(a < b), ">": int(a > b),
        "<=": int(a <= b), ">=": int(a >= b), "<<": a << b, ">>": a >> b, "+": a + b, "-": a - b,
        "*": a * b,
    }[op]


def preprocess(text: str, *, includes: dict | None = None, embeds: dict | None = None,
               search_paths: list | None = None, defines: dict | None = None,
               name: str = "<source>", return_map: bool = False):
    """Preprocess `text` to the flat translation-unit string. With `return_map=True`, also return the
    per-output-line provenance map (file, line, #include stack) so diagnostics resolve to their
    origin file even across inlined includes."""
    p = Preprocessor(includes, embeds, search_paths, defines)
    out = p.process(text, name)
    return (out, p.linemap) if return_map else out
