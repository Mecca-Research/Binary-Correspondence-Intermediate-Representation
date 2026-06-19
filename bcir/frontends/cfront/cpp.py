"""A C preprocessor for the frontend (ladder stage L7): object- and function-like `#define` macros
(with `#` stringize and `##` paste), `#undef`, conditional compilation (`#if`/`#ifdef`/`#ifndef`/
`#elif`/`#elifdef`/`#elifndef`/`#else`/`#endif`) with a constant-expression evaluator (`defined`,
`__has_include`, `__has_embed`), the predefined macros `__FILE__`/`__LINE__`/`__DATE__`/`__TIME__`
(and the static `__STDC__`/`__STDC_VERSION__`/`__STDC_HOSTED__`), the `#line` directive, the
`_Pragma` operator, `#include` of project headers, and C23 `#embed`.

It runs *before* the lexer/parser, producing fully-expanded source text (the lexer still skips any
residual `#`-line, but after this pass there are none). `#include`/`#embed` resolve against an
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
# preprocessing tokens: identifier, number, string, char, or punctuation (multi-char first).
_PUNCT = ["<<=", ">>=", "...", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
          "##", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
          "#", "+", "-", "*", "/", "%", "&", "|", "^", "~", "!", "<", ">", "=",
          "(", ")", "{", "}", "[", "]", ";", ",", ".", ":", "?"]
_TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"'                          # string
    r"|'(?:\\.|[^'\\])*'"                         # char
    r"|(?:0[xXbB][0-9a-fA-F']+|\d[\d.']*)[uUlL]*"  # number (C23 ' separators, with int suffix)
    r"|[A-Za-z_]\w*"                             # identifier
    r"|" + "|".join(re.escape(p) for p in _PUNCT))


class CPPError(Exception):
    pass


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

    # --- public ---
    def process(self, text: str, name: str = "<source>") -> str:
        out: list[str] = []
        self._run(self._logical_lines(text), out, name)
        return "\n".join(out) + "\n"

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
        return text.replace("\\\n", "").splitlines()

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
                    out.append(self._expand_text(raw))
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
            out.append(self._embed(rest))
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
        self._run(self._logical_lines(text), out, target)
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
        dynamic predefined macros (`__FILE__`/`__LINE__`), which are not stored in the table."""
        return name in self.macros or name in _DYNAMIC

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
                    out += self._expand(list(mac.body), hide | {t})
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
        amap: dict[str, list] = {}
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
        out: list = []
        body = list(mac.body)
        i = 0
        while i < len(body):
            t = body[i]
            if t == "#" and i + 1 < len(body) and body[i + 1] in amap:     # stringize
                out.append('"' + _join(amap[body[i + 1]]).replace('"', '\\"') + '"')
                i += 2
                continue
            if t == "##" and out and i + 1 < len(body):                    # paste
                right = amap.get(body[i + 1], [body[i + 1]])
                left = out.pop()
                glued = left + (_join(right[:1]) if right else "")
                out.append(glued)
                out += right[1:]
                i += 2
                continue
            out += amap.get(t, [t])                                        # arg or literal
            i += 1
        return out

    # --- constant-expression evaluation (#if / #elif) ---
    def _eval(self, expr: str) -> int:
        # handle defined / __has_include / __has_embed BEFORE macro expansion, then expand.
        expr = re.sub(r"\bdefined\s*\(\s*(\w+)\s*\)",
                      lambda m: "1" if self._defined(m.group(1)) else "0", expr)
        expr = re.sub(r"\bdefined\s+(\w+)",
                      lambda m: "1" if self._defined(m.group(1)) else "0", expr)
        expr = re.sub(r"\b__has_include\s*\(([^)]*)\)",
                      lambda m: "1" if self._resolve(self._header_name(m.group(1))) is not None
                      else "0", expr)
        expr = re.sub(r"\b__has_embed\s*\(([^)]*)\)",
                      lambda m: "1" if self._header_name(m.group(1)) in self.embeds else "0", expr)
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
               name: str = "<source>") -> str:
    return Preprocessor(includes, embeds, search_paths, defines).process(text, name)
