"""Recursive-descent parser for the C-frontend subset (L1–L4) → the `cast` AST.

Grammar (the slice drivers/kernels need): a translation unit is a sequence of struct/union
declarations and function definitions; statements are declarations, assignments, returns, `if`/
`while` (parsed for grammar stability; lowered at L6), and expression statements; expressions use
standard C precedence with `[]` / `.` / `->` / call postfixes.
"""
from __future__ import annotations

from . import cast
from .clex import KEYWORDS, Tok, parse_char_literal, parse_int_literal, tokenize
from .ctype_model import int_literal_type
from .ctype_model import is_scalar_name
from .diagnostics import FixIt, SourceDiagnostic, Span


class CParseError(Exception):
    """A parse error. `pos` is the source byte offset of the offending token (for the caret); `fixit`
    is an optional suggested edit (e.g. inserting a missing `;`)."""
    def __init__(self, message: str, pos: int | None = None, fixit: "FixIt | None" = None):
        super().__init__(message)
        self.pos = pos
        self.fixit = fixit


# punctuation whose absence is a high-confidence fix-it: suggest inserting it after the prior token.
_FIXABLE_PUNCT = frozenset({";", ")", "}", "]"})

# Recursion-depth cap for the recursive-descent grammar. A pathologically deeply-nested input (e.g.
# 100000 nested `(`, `{{{...}}}`, or `int a[((((...))))]`) would otherwise drive the mutually-recursive
# descent (_comma/_assign/_ternary/_binary/_unary/_postfix/_primary->`(`->_comma; _block<->_stmt;
# _init_value; _type_spec; _const_eval) past Python's native recursion limit and raise an UNCAUGHT
# RecursionError -- which is NOT in pipeline._FALLBACK_PHASE, so compile_with_fallback would CRASH instead
# of routing to fallback. _descend() bumps a depth counter at every recursive-cycle entry and raises a
# clean CParseError ("nesting too deep") on overflow; CParseError IS a fallback phase, so deep input
# routes to LLVM (needs_fallback=True) exactly as the C twin returns a clean rc-1 PARSE-ERR -- the two
# rails agree on the over-deep boundary. The cap is set well below the per-level frame budget that trips
# Python's default 1000-frame limit (~17 frames/level -> RecursionError near depth 58), so this guard
# fires FIRST with a clean error rather than a crash, yet far above any real program's nesting.
_MAX_DEPTH = 50


# type-start keywords (a statement beginning with one of these is a declaration).
_TYPE_KW = frozenset({"void", "_Bool", "bool", "char", "short", "int", "long", "unsigned",
                      "signed", "float", "double", "_Complex", "complex", "const", "volatile",
                      "static", "extern", "inline", "_Thread_local", "thread_local", "struct", "union",
                      "_BitInt"})   # C23 bit-precise integer `_BitInt(N)` (a type-start keyword)
# `typeof` / `typeof_unqual` (C23) / `__typeof__` (GNU): a type-specifier whose type is the operand's
# -- supported for a type-name operand `typeof(int*)` and a bare in-scope variable `typeof(x)`.
_TYPEOF_KW = frozenset({"typeof", "typeof_unqual", "__typeof__"})
# binary operators by ascending precedence groups (C order).
_PRECEDENCE = [
    ("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="), ("<", ">", "<=", ">="),
    ("<<", ">>"), ("+", "-"), ("*", "/", "%"),
]
_COMPOUND = {"+=": "+", "-=": "-", "*=": "*", "/=": "/", "%=": "%", "&=": "&", "|=": "|",
             "^=": "^", "<<=": "<<", ">>=": ">>"}


class _Descend:
    """The context manager returned by _Parser._descend(): decrements the parser's recursion-depth
    counter on exit (the increment + overflow check happen in _descend before this is constructed)."""
    __slots__ = ("_p",)

    def __init__(self, p: "_Parser"):
        self._p = p

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._p._depth -= 1
        return False


class _Parser:
    def __init__(self, toks: list[Tok], tags: set):
        self.t = toks
        self.i = 0
        self.tags = tags                      # known struct/union tags (for type detection)
        self.typedefs: dict[str, cast.TypeRef] = {}   # typedef name -> the aliased type
        self.enums: dict[str, int] = {}               # enumerator name -> its integer value
        self.recover = False                  # panic-mode recovery: collect diagnostics, don't raise
        self.diags: list = []                 # list[SourceDiagnostic] accumulated when recover is on
        self.unit: cast.Unit | None = None    # the unit being built (so a nested inline aggregate registers)
        self._anon_ctr = 0                    # synthesizes unique tags for tagless inline aggregates
        self._depth = 0                       # recursive-descent nesting depth (see _MAX_DEPTH / _descend)

    def _descend(self) -> "_Descend":
        """Enter one recursive-grammar level; raise CParseError on overflow (caught as a fallback phase so
        deep input routes to LLVM, never an uncaught RecursionError). Used as `with self._descend():` at
        each recursive-cycle entry point so a pathological deeply-nested input fails cleanly + crash-free,
        matching the C twin's depth guard (both rails route over-deep input to fallback / a clean error)."""
        self._depth += 1
        if self._depth > _MAX_DEPTH:
            self._depth -= 1   # unwind the counter on the failing entry so recovery mode stays consistent
            raise CParseError("nesting too deep", pos=self.peek().pos)
        return _Descend(self)

    # --- error recovery (panic mode): on a parse error, record a diagnostic and skip to the next
    # synchronization boundary, so one run reports several independent errors instead of just the
    # first. Each helper consumes at least one token off a non-boundary error, so progress (and
    # therefore termination) is guaranteed.
    def _record(self, e: "CParseError") -> None:
        pos = getattr(e, "pos", None)
        span = Span.at(pos) if pos is not None else None
        fixits = [e.fixit] if getattr(e, "fixit", None) is not None else []
        self.diags.append(SourceDiagnostic("error", str(e), span=span, fixits=fixits, phase="parse"))

    def _sync_toplevel(self) -> None:
        """Skip to the next top-level boundary: past a depth-0 `;` (a global/forward decl) or the
        `}` that closes the body we were parsing, so `parse_unit` can resume at the next declaration."""
        depth = 0
        while not self.at("EOF"):
            t = self.nxt()
            if t.kind == "PUNCT" and t.text == "{":
                depth += 1
            elif t.kind == "PUNCT" and t.text == "}":
                depth -= 1
                if depth <= 0:
                    return
            elif t.kind == "PUNCT" and t.text == ";" and depth == 0:
                return

    def _sync_stmt(self) -> bool:
        """Skip to the next statement boundary inside a block: past a depth-0 `;`, returning True to
        resume the block. Returns False at the block's own closing `}` (left for `_block` to consume)
        or at EOF, so the block loop stops instead of spinning."""
        depth = 0
        while not self.at("EOF"):
            if depth == 0 and self.at("PUNCT", "}"):
                return False
            t = self.nxt()
            if t.kind == "PUNCT" and t.text == "{":
                depth += 1
            elif t.kind == "PUNCT" and t.text == "}":
                depth -= 1
            elif t.kind == "PUNCT" and t.text == ";" and depth == 0:
                return True
        return False

    # --- token helpers ---
    def peek(self, k: int = 0) -> Tok:
        return self.t[min(self.i + k, len(self.t) - 1)]

    def nxt(self) -> Tok:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def at(self, kind: str, text: str | None = None) -> bool:
        tk = self.peek()
        return tk.kind == kind and (text is None or tk.text == text)

    def _prev_end(self) -> int:
        """The source offset just past the last consumed token -- where a missing `;`/`)` belongs."""
        if self.i == 0:
            return self.peek().pos
        prev = self.t[self.i - 1]
        return prev.pos + len(prev.text)

    def eat(self, kind: str, text: str | None = None) -> Tok:
        if not self.at(kind, text):
            tk = self.peek()
            fix = None
            if kind == "PUNCT" and text in _FIXABLE_PUNCT:    # suggest inserting the missing punctuation
                ins = self._prev_end()                        # right after the prior token (Clang-style)
                fix = FixIt(Span(ins, ins), text)
            raise CParseError(f"expected {text or kind!r}, got {tk.kind} {tk.text!r}",
                              pos=tk.pos, fixit=fix)
        return self.nxt()

    # --- unit ---
    def _attributes(self) -> dict:
        """Consume `__attribute__((packed))` / `__attribute__((aligned(N)))` / `alignas(N)` runs, plus a
        C23 `[[ ... ]]` attribute run. The only C23 attributes acted on are the value-neutral hints
        `[[unsequenced]]` / `[[reproducible]]` (recorded in the dict); any other C23 attribute (incl.
        namespaced `gnu::packed` / argument forms) is scanned over and dropped."""
        attrs: dict = {}
        while True:
            if self.at("PUNCT", "[") and self.peek(1).kind == "PUNCT" and self.peek(1).text == "[":
                self.nxt()                                # consume the opening `[[`
                self.nxt()
                while not (self.at("PUNCT", "]") and self.peek(1).kind == "PUNCT"   # scan to the closing `]]`
                           and self.peek(1).text == "]"):
                    if self.at("EOF"):
                        raise CParseError("unterminated [[...]] attribute", pos=self.peek().pos)
                    if self.at("IDENT", "unsequenced") or self.at("IDENT", "reproducible"):
                        attrs["reproducible"] = True      # both hints fold to one fusion-legality flag
                    self.nxt()                            # robust to args/namespaces: skip every other token
                self.nxt()                                # eat the closing `]]`
                self.nxt()
            elif self.at("IDENT", "__attribute__"):
                self.nxt()
                self.eat("PUNCT", "(")
                self.eat("PUNCT", "(")
                while not self.at("PUNCT", ")"):
                    a = self.eat("IDENT").text
                    if a in ("packed", "__packed__"):
                        attrs["packed"] = True
                    elif a in ("aligned", "__aligned__"):
                        self.eat("PUNCT", "(")
                        attrs["aligned"] = parse_int_literal(self.eat("INT").text)
                        self.eat("PUNCT", ")")
                    if self.at("PUNCT", ","):
                        self.nxt()
                self.eat("PUNCT", ")")
                self.eat("PUNCT", ")")
            elif self.at("IDENT", "alignas") or self.at("IDENT", "_Alignas"):
                self.nxt()
                self.eat("PUNCT", "(")
                attrs["aligned"] = parse_int_literal(self.eat("INT").text)
                self.eat("PUNCT", ")")
            else:
                return attrs

    def parse_unit(self) -> cast.Unit:
        unit = cast.Unit()
        self.unit = unit
        while not self.at("EOF"):
            if not self.recover:
                self._toplevel_item(unit)
                continue
            try:                                               # panic-mode recovery: keep going past
                self._toplevel_item(unit)                      # a bad declaration to report the next
            except CParseError as e:
                self._record(e)
                self._sync_toplevel()
        return unit

    def _toplevel_item(self, unit: cast.Unit) -> None:
        """Parse one top-level item (typedef / enum / aggregate definition / function / global)."""
        # A C23 `[[unsequenced]]`/`[[reproducible]]` (or any leading attribute) may precede a function /
        # global. `_attributes()` returns {} when the next token is not an attribute, so this is a no-op
        # before a typedef / enum / struct / plain type -- NON-DISTURBANCE for every existing item.
        lead = self._attributes()
        if self.at("IDENT", "typedef"):                        # a type alias (resolved at parse time)
            self._typedef(unit)
            return
        if self.at("IDENT", "enum"):
            save = self.i
            self.nxt()
            tag = self.eat("IDENT").text if self.at("IDENT") else ""
            if self.at("PUNCT", "{"):                          # an enum definition: register the values
                self._enum_body(tag)
                self.eat("PUNCT", ";")
                return
            self.i = save                                      # `enum tag` used as a type
        if self.at("IDENT", "struct") or self.at("IDENT", "union"):
            save = self.i
            kind = self.nxt().text
            attrs = self._attributes()
            tag = self.eat("IDENT").text if self.at("IDENT") else ""
            if self.at("PUNCT", "{"):                          # an aggregate definition
                agg = self._aggregate_body(kind, tag, attrs)
                unit.aggregates[agg.tag] = agg
                self.tags.add(agg.tag)
                self.eat("PUNCT", ";")
                return
            self.i = save                                      # a struct *type* (func ret / global)
        tref = self._type_spec()
        tref, name = self._declarator(tref)
        if self.at("PUNCT", "("):                              # a function definition
            unit.funcs.append(self._func_body(tref, name, reproducible=bool(lead.get("reproducible"))))
        else:                                                  # a file-scope global variable
            unit.globals.append(self._global(tref, name))

    def _typedef(self, unit: cast.Unit) -> None:
        """`typedef <type> <name>;` -- register `name` -> the aliased type (resolved at parse time,
        so the lowered claim graph is identical to spelling the underlying type out). Handles scalar/
        pointer/qualified aliases, `typedef struct/union [tag] {...} Name;`, and `typedef enum {...}
        Name;`."""
        self.eat("IDENT", "typedef")
        if self.at("IDENT", "enum"):
            self.nxt()
            tag = self.eat("IDENT").text if self.at("IDENT") and not self.at("PUNCT", "{") else ""
            if self.at("PUNCT", "{"):
                self._enum_body(tag)
            base = cast.TypeRef(base="int")                    # an enum is an int-sized scalar
        elif self.at("IDENT", "struct") or self.at("IDENT", "union"):
            kind = self.nxt().text
            attrs = self._attributes()
            tag = self.eat("IDENT").text if self.at("IDENT") else ""
            if self.at("PUNCT", "{"):
                agg = self._aggregate_body(kind, tag, attrs)
                unit.aggregates[agg.tag] = agg
                self.tags.add(agg.tag)
                tag = agg.tag
            base = cast.TypeRef(base=tag, aggregate=kind)
        else:
            base = self._type_spec()
        if self.at("PUNCT", "(") and self.peek(1).kind == "OP" and self.peek(1).text == "*":
            tref, name = self._funcptr_declarator(base)    # typedef RET (*NAME)(PARAMS);
        else:
            tref, name = self._declarator(base)
        self.typedefs[name] = tref
        self.eat("PUNCT", ";")

    def _funcptr_declarator(self, ret: cast.TypeRef):
        """`( * NAME ) ( param-type-list )` — a function-pointer declarator. Returns a funcptr TypeRef
        (carrying the return + parameter types for faithful emit) and the declared NAME. The name is
        also stashed in ``base`` so a later use of the alias renders verbatim."""
        self.eat("PUNCT", "(")
        self.eat("OP", "*")
        name = self.eat("IDENT").text
        self.eat("PUNCT", ")")
        self.eat("PUNCT", "(")
        params: list[cast.TypeRef] = []
        if self.at("IDENT", "void") and self.peek(1).text == ")":
            self.nxt()
        elif not self.at("PUNCT", ")"):
            while True:
                pt = self._type_spec()
                while self.at("OP", "*"):                  # pointer parameter
                    pt = cast.TypeRef(base=pt.base, ptr=pt.ptr + 1, aggregate=pt.aggregate,
                                      quals=pt.quals)
                    self.nxt()
                if self.at("IDENT"):                       # an optional parameter name (ignored)
                    self.nxt()
                params.append(pt)
                if self.at("PUNCT", ","):
                    self.nxt()
                    continue
                break
        self.eat("PUNCT", ")")
        return (cast.TypeRef(base=name, funcptr=True, func_ret=ret, func_params=tuple(params)), name)

    def _is_funcptr_declarator(self) -> bool:
        """True if the cursor is at `( * NAME ) (` — a function-pointer declarator (`int (*g)(int)`),
        as opposed to the row-pointer `( * NAME ) [` form that `_declarator` handles."""
        return (self.at("PUNCT", "(") and self.peek(1).kind == "OP" and self.peek(1).text == "*"
                and self.peek(2).kind == "IDENT"
                and self.peek(3).kind == "PUNCT" and self.peek(3).text == ")"
                and self.peek(4).kind == "PUNCT" and self.peek(4).text == "(")

    def _declarator_or_funcptr(self, base: cast.TypeRef):
        """A declarator (param or local position) that may be an inline function-pointer
        `RET (*NAME)(PARAMS)` — for which there is no typedef name, so the full signature is captured."""
        if self._is_funcptr_declarator():
            return self._funcptr_declarator(base)
        return self._declarator(base)

    def _enum_body(self, tag: str) -> None:
        """Parse `{ A, B = expr, C }` -- assign each enumerator its C value (prev+1, or the given
        constant) and register it so a later use resolves to that integer literal."""
        self.eat("PUNCT", "{")
        value = 0
        while not self.at("PUNCT", "}"):
            name = self.eat("IDENT").text
            if self.at("OP", "="):
                self.nxt()
                value = self._const_eval(self._binary(0))
            self.enums[name] = value
            value += 1
            if self.at("PUNCT", ","):
                self.nxt()
        self.eat("PUNCT", "}")

    def _const_eval(self, node) -> int:
        """Fold a constant enum initializer (integer literals, prior enumerators, basic arithmetic)."""
        if isinstance(node, cast.IntLit):
            return node.value
        if isinstance(node, cast.Name):
            if node.ident in self.enums:
                return self.enums[node.ident]
            raise CParseError(f"non-constant enum initializer {node.ident!r}")
        if isinstance(node, cast.Unary):
            v = self._const_eval(node.operand)
            return {"-": -v, "~": ~v, "!": int(not v)}.get(node.op, v)
        if isinstance(node, cast.Binary):
            a, b = self._const_eval(node.lhs), self._const_eval(node.rhs)
            ops = {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b else 0,
                   "%": a % b if b else 0, "&": a & b, "|": a | b, "^": a ^ b,
                   "<<": a << b, ">>": a >> b}
            return ops.get(node.op, 0)
        raise CParseError("unsupported constant enum initializer")

    def _global(self, tref: cast.TypeRef, name: str) -> cast.Global:
        init: tuple = ()
        if self.at("OP", "="):
            self.nxt()
            if self.at("PUNCT", "{"):                          # an initializer list (positional + [i]= designated)
                self.nxt()
                slots: dict[int, object] = {}                  # array index -> element expr
                cursor = 0
                while not self.at("PUNCT", "}"):
                    if self.at("PUNCT", "["):                  # an array designator: [const-index] = expr
                        self.nxt()
                        cursor = self._const_eval(self._expr())   # folds int literals + enumerators
                        self.eat("PUNCT", "]")
                        self.eat("OP", "=")
                    slots[cursor] = self._expr()
                    cursor += 1
                    if self.at("PUNCT", ","):
                        self.nxt()
                self.eat("PUNCT", "}")
                n = (max(slots) + 1) if slots else 0           # the highest index reached sizes a `T[]`
                init = tuple(slots.get(i, cast.IntLit(0)) for i in range(n))   # gaps zero-fill (§6.7.10)
            else:
                init = (self._expr(),)
        self.eat("PUNCT", ";")
        return cast.Global(type=tref, name=name, init=init)

    def _aggregate_body(self, kind: str, tag: str, attrs: dict) -> cast.Aggregate:
        self.eat("PUNCT", "{")
        members = []
        while not self.at("PUNCT", "}"):
            matt = self._attributes()                     # member-leading `_Alignas(N)`/`alignas(N)`/
            malign = matt.get("aligned", 0)               # `__attribute__((aligned(N)))` -- over-aligns this member
            inline_agg = False                            # `struct {...}` / `union {...}` defined inline as a member
            if self.at("IDENT", "struct") or self.at("IDENT", "union"):
                save = self.i
                ikind = self.nxt().text
                iattrs = self._attributes()
                itag = self.eat("IDENT").text if (self.at("IDENT") and not self.at("PUNCT", "{")) else ""
                if self.at("PUNCT", "{"):                 # an INLINE aggregate definition (anonymous or tagged)
                    if not itag:
                        itag = f"$anon{self._anon_ctr}"
                        self._anon_ctr += 1
                    iagg = self._aggregate_body(ikind, itag, {**iattrs, **matt})
                    self.unit.aggregates[iagg.tag] = iagg
                    self.tags.add(iagg.tag)
                    base = cast.TypeRef(base=iagg.tag, aggregate=ikind)
                    inline_agg = True
                else:
                    self.i = save                         # `struct Tag member;` -- not an inline definition
            if inline_agg and self.at("PUNCT", ";"):      # ANONYMOUS member: an inline aggregate, no declarator --
                self.nxt()                                # its leaves promote into THIS aggregate (name "" marks it)
                members.append((base, "", 0, malign))
                continue
            if not inline_agg:
                base = self._type_spec()
            while True:                                   # one or more declarators off one specifier:
                if self.at("PUNCT", ":"):                 # an UNNAMED/zero-width bitfield `type : width;` (no
                    self.nxt()                            # declarator): it positions the layout cursor but is not
                    w = parse_int_literal(self.eat("INT").text)   # accessible -- name "" + a scalar base marks it
                    members.append((base, "", w, malign))
                    if self.at("PUNCT", ","):
                        self.nxt()
                        continue
                    break
                if self.at("PUNCT", "(") and self.peek(1).kind == "OP" and self.peek(1).text == "*":
                    tref, name = self._funcptr_declarator(base)   # `RET (*name)(params)` -- a funcptr member
                else:                                             # (8-byte; set from a funcptr value, called
                    tref, name = self._declarator(base)   #   `unsigned x, y, z;` / `unsigned a:3, b:5;`  `o->fn(a)`)
                width = 0
                if self.at("PUNCT", ":"):                 # bitfield:  type name : width;
                    self.nxt()
                    width = parse_int_literal(self.eat("INT").text)
                members.append((tref, name, width, malign))   # `malign` applies to every declarator here
                if self.at("PUNCT", ","):                 # another member off the same specifier
                    self.nxt()
                    continue
                break
            self.eat("PUNCT", ";")
        self.eat("PUNCT", "}")
        trailing = self._attributes()                     # `} __attribute__((packed))` (caller eats `;`)
        attrs = {**attrs, **trailing}
        return cast.Aggregate(kind=kind, tag=tag, members=tuple(members),
                              packed=bool(attrs.get("packed")), align=attrs.get("aligned", 0))

    # --- types ---
    def _type_spec(self) -> cast.TypeRef:
        # depth guard: _type_spec recurses via `typeof(type-name)` / `_Atomic(type-name)` -> _type_spec.
        with self._descend():
            return self._type_spec_inner()

    def _type_spec_inner(self) -> cast.TypeRef:
        quals: list[str] = []
        words: list[str] = []
        aggregate = ""
        base = ""
        bit_width = 0                                     # a C23 `_BitInt(N)` width (set when a `_BitInt` is seen)
        saw_bitint = False                                # `_BitInt` seen (separate from bit_width, since N==0 is
        td: cast.TypeRef | None = None                    #   falsy -- the validation below rejects it as out-of-range)
        while self.at("IDENT"):
            w = self.peek().text
            if w == "_BitInt":                            # C23 `_BitInt ( N )` -- a bit-precise integer type
                self.nxt()
                self.eat("PUNCT", "(")
                if not self.at("INT"):
                    raise CParseError("expected the width N in `_BitInt(N)`", pos=self.peek().pos)
                bit_width = parse_int_literal(self.eat("INT").text)
                self.eat("PUNCT", ")")
                if saw_bitint:                            # a second `_BitInt` in one specifier run -> fallback
                    raise CParseError("duplicate `_BitInt` type specifier")
                saw_bitint = True
                words.append("_BitInt")
                continue
            if w == "_Atomic" and self.peek(1).text == "(":   # `_Atomic ( type-name )` -- atomic type specifier
                self.nxt()
                self.eat("PUNCT", "(")
                inner = self._type_spec()
                ip = 0
                while self.at("OP", "*"):
                    ip += 1
                    self.nxt()
                self.eat("PUNCT", ")")
                return cast.TypeRef(base=inner.base, ptr=ip, array=inner.array, aggregate=inner.aggregate,
                                    quals=tuple(quals) + ("_Atomic",) + inner.quals)
            if w in ("const", "volatile", "_Atomic"):
                quals.append(w)
                self.nxt()
            elif w in ("static", "extern", "_Thread_local", "thread_local", "inline"):
                self.nxt()                                # storage/inline ignored
            elif w in ("struct", "union"):
                aggregate = w
                self.nxt()
                base = self.eat("IDENT").text             # the tag
                break
            elif w == "enum":                             # `enum [tag] [{...}]` -> an int scalar
                self.nxt()
                if self.at("IDENT") and not self.at("PUNCT", "{"):
                    self.nxt()                            # the tag (ignored; enum is int-sized)
                if self.at("PUNCT", "{"):
                    self._enum_body("")
                base = "int"
                break
            elif w in _TYPEOF_KW and not words and not aggregate:   # typeof(type-name) / typeof(variable)
                self.nxt()
                self.eat("PUNCT", "(")
                if self._is_decl_start():                  # typeof ( type-name ), incl. `typeof(int*)`
                    inner = self._type_spec()
                    ip = 0
                    while self.at("OP", "*"):
                        ip += 1
                        self.nxt()
                    self.eat("PUNCT", ")")
                    return cast.TypeRef(base=inner.base, ptr=ip, array=inner.array,
                                        aggregate=inner.aggregate, quals=tuple(quals) + inner.quals)
                expr = self._expr()                        # typeof ( expression ) -- a general operand
                self.eat("PUNCT", ")")
                if isinstance(expr, cast.Name):            # a bare in-scope variable keeps the fast path
                    return cast.TypeRef(base="", typeof_var=expr.ident, quals=tuple(quals))
                return cast.TypeRef(base="", typeof_expr=expr, quals=tuple(quals))
            elif w in ("va_list", "__builtin_va_list") and not words and not aggregate:   # variadic cursor type
                self.nxt()
                base = "va_list"
                break
            elif not words and w in self.typedefs:        # a typedef name -> expand the alias
                td = self.typedefs[w]
                self.nxt()
                break
            elif w in _TYPE_KW or is_scalar_name(w):
                words.append(w)
                self.nxt()
            else:
                break
        if td is not None:                                # merge the alias with any leading quals
            if td.funcptr:                                # a function-pointer alias carries its own shape
                return td
            return cast.TypeRef(base=td.base, ptr=td.ptr, array=td.array,
                                aggregate=td.aggregate, quals=tuple(quals) + td.quals)
        if saw_bitint:                                    # C23 `_BitInt(N)`: the spelling carries N + signedness;
            return self._bitint_typeref(words, bit_width, quals)   # lowering builds the `bitint` CType from it
        if not aggregate and not base:                    # `enum [tag]` already set base="int"; only
            base = self._canon_scalar(words)              # canonicalize a scalar keyword run otherwise
        return cast.TypeRef(base=base, aggregate=aggregate, quals=tuple(quals))

    @staticmethod
    def _bitint_typeref(words: list[str], bit_width: int, quals) -> cast.TypeRef:
        """Build the TypeRef for a `_BitInt(N)` keyword run. The supported subset is a single `_BitInt`
        with at most one of `signed`/`unsigned` (and cv-quals, already split off): `_BitInt(N)` /
        `signed _BitInt(N)` are signed, `unsigned _BitInt(N)` unsigned. Any other word in the run (a base
        int keyword, a second `_BitInt`, a width N<2 or N>64) is rejected -- a CParseError routes the whole
        unit to fallback, the conservative boundary. The `base` is the verbatim emit spelling."""
        extra = [x for x in words if x not in ("_BitInt", "signed", "unsigned")]
        if extra or words.count("_BitInt") != 1 or ("signed" in words and "unsigned" in words):
            raise CParseError(f"unsupported `_BitInt` type specifier {' '.join(words)!r}")
        if not (2 <= bit_width <= 64):                    # the faithful-emit subset (N<2 / N>64 -> fallback)
            raise CParseError(f"`_BitInt({bit_width})` is outside the supported width range 2..64")
        signed = "unsigned" not in words
        spelling = f"_BitInt({bit_width})" if signed else f"unsigned _BitInt({bit_width})"
        return cast.TypeRef(base=spelling, quals=tuple(quals), bit_width=bit_width)

    @staticmethod
    def _canon_scalar(words: list[str]) -> str:
        if not words:
            raise CParseError("expected a type")
        # `signed char` is a DISTINCT type from plain `char`: it is signed on every target, whereas
        # plain `char`'s signedness is implementation-defined (signed on x86, unsigned on ARM). Keep it
        # so the emit spells it faithfully. For every OTHER integer, `signed` is the default -- drop it.
        if set(words) == {"signed", "char"}:
            return "signed char"
        # C99 `_Complex` (the <complex.h> spelling is `complex`): `<float> _Complex` in either keyword
        # order; a bare `_Complex` with no float keyword means `double _Complex`.
        if "_Complex" in words or "complex" in words:
            rest = [w for w in words if w not in ("_Complex", "complex", "signed")]
            flt = " ".join(rest) if rest else "double"
            if flt not in ("float", "double", "long double"):
                raise CParseError(f"unsupported _Complex element type {flt!r}")
            return f"{flt} _Complex"
        words = [w for w in words if w != "signed"]
        if not words:
            return "int"                                  # a bare `signed` == int
        joined = " ".join(words)
        # canonicalize the legal multi-word combos; otherwise it's a single fixed-width name.
        table = {"unsigned": "unsigned int", "unsigned int": "unsigned int",
                 "long long": "long long", "unsigned long": "unsigned long",
                 "unsigned long long": "unsigned long long", "unsigned char": "unsigned char",
                 "unsigned short": "unsigned short"}
        return table.get(joined, words[-1] if len(words) == 1 else joined)

    def _declarator(self, base: cast.TypeRef):
        """Parse `*` pointer prefixes, the name, and `[N]` array suffixes onto `base`."""
        ptr = 0
        while self.at("OP", "*"):
            ptr += 1
            self.nxt()
            while (self.at("IDENT", "const") or self.at("IDENT", "volatile")
                   or self.at("IDENT", "restrict") or self.at("IDENT", "__restrict")
                   or self.at("IDENT", "__restrict__")):     # `restrict` is an aliasing hint -- consumed
                self.nxt()                                # pointer qualifier (ignored for layout)
        # pointer-to-array declarator `(*name)[N]...` -- a "row pointer" (what `T m[][N]` decays to);
        # modeled as the equivalent multi-dim array param (outer dim unspecified) so `m[i][j]` flattens
        # row-major exactly as for `T m[A][N]`. The remaining vendor-header declarator form.
        if ptr == 0 and self.at("PUNCT", "("):
            save = self.i
            self.nxt()                                    # (
            inner = 0
            while self.at("OP", "*"):
                inner += 1
                self.nxt()
                while (self.at("IDENT", "const") or self.at("IDENT", "volatile")
                       or self.at("IDENT", "restrict") or self.at("IDENT", "__restrict")
                       or self.at("IDENT", "__restrict__")):
                    self.nxt()
            if (inner == 1 and self.at("IDENT")
                    and self.peek(1).kind == "PUNCT" and self.peek(1).text == ")"):
                nm = self.nxt().text                      # the name
                self.nxt()                                # )
                if self.at("PUNCT", "["):
                    dims = []
                    while self.at("PUNCT", "["):
                        self.nxt()
                        dims.append(0 if self.at("PUNCT", "]")
                                    else parse_int_literal(self.eat("INT").text))
                        self.eat("PUNCT", "]")
                    return cast.TypeRef(base=base.base, ptr=0, array=tuple([0] + dims),
                                        aggregate=base.aggregate, quals=base.quals,
                                        typeof_var=base.typeof_var, typeof_expr=base.typeof_expr,
                                        bit_width=base.bit_width), nm
            self.i = save                                 # not `(*name)[..]` -> a normal declarator
        name = self.eat("IDENT").text
        dims = []
        vla = None
        vla_dims: tuple = ()
        raw: list = []                                    # per-dim: an int (literal) or an expression (runtime)
        while self.at("PUNCT", "["):
            self.nxt()
            if self.at("PUNCT", "]"):
                raw.append(0)                             # an incomplete `[]` dimension
            else:
                e = self._assign()                        # the dim expression
                raw.append(e.value if isinstance(e, cast.IntLit) else e)   # a single int literal -> a static dim
            self.eat("PUNCT", "]")
        if any(not isinstance(d, int) for d in raw):      # at least one RUNTIME dim -> a VLA
            if len(raw) > 3:
                raise CParseError("a variable-length array of more than 3 dimensions is not supported")
            if len(raw) == 1:
                vla = raw[0]                              # a 1-D VLA -- the existing representation (unchanged path)
                dims = [0]
            else:
                vla_dims = tuple(raw)                     # a MULTI-dim VLA `T a[m][n]` -- per-dim (literal | expr)
                dims = [0] * len(raw)
        else:
            dims = raw                                    # all-literal -> a static (possibly multi-dim) array
        if base.funcptr and ptr == 0 and not dims:        # `binop_fn fn` — keep the funcptr shape
            return base, name
        return cast.TypeRef(base=base.base, ptr=ptr + base.ptr,         # base.ptr != 0 only for typeof(T*)
                            array=tuple(base.array) + tuple(dims), vla=vla, vla_dims=vla_dims,
                            aggregate=base.aggregate, quals=base.quals,
                            typeof_var=base.typeof_var, typeof_expr=base.typeof_expr,
                            bit_width=base.bit_width), name

    # --- functions ---
    def _func_body(self, ret: cast.TypeRef, name: str, reproducible: bool = False) -> cast.Func:
        self.eat("PUNCT", "(")
        params = []
        variadic = False
        if not (self.at("PUNCT", ")") or self.at("IDENT", "void") and self.peek(1).text == ")"):
            while True:
                if self.at("PUNCT", "..."):            # a trailing `...` -- the function is variadic
                    self.nxt()
                    variadic = True
                    break
                ptype = self._type_spec()
                ptype, pname = self._declarator_or_funcptr(ptype)
                params.append(cast.Param(ptype, pname))
                if self.at("PUNCT", ","):
                    self.nxt()
                    continue
                break
        elif self.at("IDENT", "void"):
            self.nxt()
        self.eat("PUNCT", ")")
        body = self._block()
        return cast.Func(ret=ret, name=name, params=tuple(params), body=body, variadic=variadic,
                         reproducible=reproducible)

    # --- statements ---
    def _block(self) -> tuple:
        with self._descend():   # depth guard: _block<->_stmt nesting (`{{{...}}}`) cycle
            self.eat("PUNCT", "{")
            stmts = []
            while not self.at("PUNCT", "}"):
                if not self.recover:
                    stmts.append(self._stmt())
                    continue
                try:                                              # statement-level recovery: a bad
                    stmts.append(self._stmt())                    # statement doesn't abandon the block
                except CParseError as e:
                    self._record(e)
                    if not self._sync_stmt():                     # hit the block's `}` / EOF -> stop
                        break
            self.eat("PUNCT", "}")
            return tuple(stmts)

    def _is_decl_start(self) -> bool:
        if not self.at("IDENT"):
            return False
        w = self.peek().text
        return (w in _TYPE_KW or w in _TYPEOF_KW or w == "enum" or is_scalar_name(w)
                or w in ("va_list", "__builtin_va_list", "_Atomic")
                or w in self.tags or w in self.typedefs)

    def _stmt(self):
        if self.at("PUNCT", ";"):                             # empty statement -> a no-op (e.g. the
            self.nxt()                                        # body of `for(...);` / `while(...);`, `if(c);`)
            return cast.Seq(())
        if self.at("PUNCT", "{"):
            return cast.Block(self._block())                  # bare `{ ... }` -> an inline-lowered scope
        if self.at("IDENT", "return"):
            self.nxt()
            val = None if self.at("PUNCT", ";") else self._expr()
            self.eat("PUNCT", ";")
            return cast.Return(val)
        if self.at("IDENT", "if"):
            return self._if()
        if self.at("IDENT", "while"):
            self.nxt()
            self.eat("PUNCT", "(")
            cond = self._expr()
            self.eat("PUNCT", ")")
            return cast.While(cond, self._block() if self.at("PUNCT", "{") else (self._stmt(),))
        if self.at("IDENT", "for"):
            return self._for()
        if self.at("IDENT", "switch"):
            return self._switch()
        if self.at("IDENT", "do"):
            self.nxt()
            body = self._block() if self.at("PUNCT", "{") else (self._stmt(),)
            self.eat("IDENT", "while")
            self.eat("PUNCT", "(")
            cond = self._expr()
            self.eat("PUNCT", ")")
            self.eat("PUNCT", ";")
            return cast.DoWhile(cond, body)
        if self.at("IDENT", "break"):
            self.nxt()
            self.eat("PUNCT", ";")
            return cast.Break()
        if self.at("IDENT", "continue"):
            self.nxt()
            self.eat("PUNCT", ";")
            return cast.Continue()
        if self.at("IDENT", "goto"):
            self.nxt()
            if self.at("OP", "*"):                            # `goto *expr;` -- a computed (indirect) goto (GNU)
                self.nxt()
                target = self._expr()
                self.eat("PUNCT", ";")
                return cast.ComputedGoto(target)
            label = self.eat("IDENT").text
            self.eat("PUNCT", ";")
            return cast.Goto(label)
        if self.at("IDENT") and self.peek(1).kind == "PUNCT" and self.peek(1).text == ":":
            name = self.nxt().text                            # `label:` — the labeled stmt follows
            self.eat("PUNCT", ":")
            return cast.Label(name)
        if self.at("IDENT", "static") or self._is_decl_start():
            return self._decl_stmt()
        expr = self._incdec() or self._expr()             # i++ / ++i / i-- / --i, else an expression
        self.eat("PUNCT", ";")
        return cast.ExprStmt(expr)

    def _for(self) -> cast.For:
        """`for (init ; cond ; step) body` — desugars onto the while machinery in lowering:
        `init; while(cond){ body; step }` (no `break`/`continue` yet, so this is exact)."""
        self.eat("IDENT", "for")
        self.eat("PUNCT", "(")
        if self.at("PUNCT", ";"):                  # empty init
            init = None
            self.nxt()
        elif self._is_decl_start():
            init = self._decl_stmt()               # a declaration (consumes its `;`)
        else:
            init = cast.ExprStmt(self._incdec() or self._expr())
            self.eat("PUNCT", ";")
        cond = cast.IntLit(1) if self.at("PUNCT", ";") else self._expr()
        self.eat("PUNCT", ";")
        step = None if self.at("PUNCT", ")") else self._for_step()
        self.eat("PUNCT", ")")
        body = self._block() if self.at("PUNCT", "{") else (self._stmt(),)
        return cast.For(init, cond, step, body)

    def _for_step(self):
        """The for-loop step: one or more comma-separated simple expressions (`i++, j--`), each an
        inc/dec or an assignment, run in order at the end of every iteration (the comma operator in
        its dominant position). A single element returns a bare ExprStmt; several wrap in a Seq."""
        parts = [cast.ExprStmt(self._incdec() or self._assign())]
        while self.at("PUNCT", ","):
            self.nxt()
            parts.append(cast.ExprStmt(self._incdec() or self._assign()))
        return parts[0] if len(parts) == 1 else cast.Seq(tuple(parts))

    def _incdec(self):
        """`i++` / `++i` / `i--` / `--i` with the value discarded (statement / for-clause) -> the
        assignment `i = i ± 1`. Returns the Assign, or None (consuming nothing) if not an inc/dec."""
        if self.at("OP", "++") or self.at("OP", "--"):
            op = self.nxt().text[0]
            tk = self.eat("IDENT")
            nm = cast.Name(tk.text, pos=tk.pos)
            return cast.Assign(nm, cast.Binary(op, nm, cast.IntLit(1)))
        if self.at("IDENT") and self.peek(1).kind == "OP" and self.peek(1).text in ("++", "--"):
            tk = self.nxt()
            op = self.nxt().text[0]
            nm = cast.Name(tk.text, pos=tk.pos)
            return cast.Assign(nm, cast.Binary(op, nm, cast.IntLit(1)))
        return None

    def _switch(self):
        """`switch (disc) { case C: ...; break; default: ...; }` -> a real C `switch`: the
        discriminant + a flat body sequence (`Case` / `Default` labels interleaved with statements,
        `break;` preserved as a `Break`). Cross-clause fallthrough is modeled exactly (the body runs
        from the matched label until a break), and the case labels are folded constant expressions."""
        self.eat("IDENT", "switch")
        self.eat("PUNCT", "(")
        disc = self._expr()
        self.eat("PUNCT", ")")
        self.eat("PUNCT", "{")
        body: list = []
        while not self.at("PUNCT", "}"):
            if self.at("IDENT", "case"):
                self.nxt()
                body.append(cast.Case(self._const_eval(self._expr())))   # case <const>:
                self.eat("PUNCT", ":")
            elif self.at("IDENT", "default"):
                self.nxt()
                self.eat("PUNCT", ":")
                body.append(cast.Default())
            else:
                body.append(self._stmt())                   # a statement (incl. `break;` -> cast.Break)
        self.eat("PUNCT", "}")
        return cast.Switch(disc, tuple(body))

    def _if(self) -> cast.If:
        self.eat("IDENT", "if")
        self.eat("PUNCT", "(")
        cond = self._expr()
        self.eat("PUNCT", ")")
        then = self._block() if self.at("PUNCT", "{") else (self._stmt(),)
        els: tuple = ()
        if self.at("IDENT", "else"):
            self.nxt()
            els = self._block() if self.at("PUNCT", "{") else (self._stmt(),)
        return cast.If(cond, then, els)

    def _decl_stmt(self):
        """A local declaration, possibly with several comma-separated declarators sharing one
        type-specifier: `T a = x, b, c = z;` == `T a = x; T b; T c = z;`. Each declarator re-derives
        its own pointer/array shape from the base (so `int *p, q;` types p pointer, q int)."""
        is_static = False
        if self.at("IDENT", "static"):                # storage class (otherwise eaten by _type_spec)
            is_static = True
            self.nxt()
        base = self._type_spec()
        decls = []
        while True:
            tref, name = self._declarator_or_funcptr(base)
            init = None
            if self.at("OP", "="):
                self.nxt()
                init = self._init_value()
            decls.append(cast.Decl(tref, name, init, static_storage=is_static))
            if self.at("PUNCT", ","):                 # another declarator off the same specifier
                self.nxt()
                continue
            break
        self.eat("PUNCT", ";")
        return decls[0] if len(decls) == 1 else cast.Seq(tuple(decls))

    def _init_value(self):
        """A declarator initializer: a scalar expression, or a braced aggregate initializer (positional
        + `[i]=` / `.field=` designators) for a struct/union/array local."""
        if not self.at("PUNCT", "{"):
            return self._expr()
        with self._descend():   # depth guard: nested braced initializers (`{{{...}}}`) self-recurse here
            return self._init_value_braced()

    def _init_value_braced(self):
        self.nxt()
        entries = []
        while not self.at("PUNCT", "}"):
            key = None
            if self.at("PUNCT", "[") or self.at("PUNCT", "."):   # a designator (possibly a nested chain)
                steps = []                                       # .a.b / .v[i] / [i][j] -> a step list
                while self.at("PUNCT", "[") or self.at("PUNCT", "."):
                    if self.at("PUNCT", "["):
                        self.nxt()
                        steps.append(("a", self._const_eval(self._expr())))
                        self.eat("PUNCT", "]")
                    else:
                        self.nxt()
                        steps.append(("m", self.eat("IDENT").text))
                self.eat("OP", "=")
                # one designator keeps the scalar key (int index / str field); a chain is a tuple of steps
                key = steps[0][1] if len(steps) == 1 else tuple(steps)
            # the value may itself be a braced initializer (a nested aggregate: a struct's array member
            # `{ {e0,e1,..}, n }`, or a nested struct/array) -- recurse through `_init_value`.
            entries.append((key, self._init_value()))
            if self.at("PUNCT", ","):
                self.nxt()
        self.eat("PUNCT", "}")
        return cast.AggInit(entries=tuple(entries))

    # --- expressions (precedence climbing) ---
    def _expr(self):
        return self._assign()

    def _comma(self):
        """A FULL expression with the comma operator (lowest precedence): `a, b` evaluates `a` for its side
        effects, discards it, and yields `b`. Only valid where C allows a full `expression` -- used for the
        PRIMARY parenthesized `( ... )`. Call ARGUMENTS and initializer ELEMENTS keep `_assign` (there the
        comma is a separator, not the operator), so `f(a, b)` / `{a, b}` are unaffected."""
        with self._descend():   # depth guard: _comma is the parenthesized-expression re-entry (`(...)`)
            e = self._assign()
            while self.at("PUNCT", ","):
                self.nxt()
                e = cast.Binary(",", e, self._assign())
            return e

    def _assign(self):
        lhs = self._ternary()
        if self.at("OP", "="):
            self.nxt()
            return cast.Assign(lhs, self._assign())
        if self.peek().kind == "OP" and self.peek().text in _COMPOUND:
            op = _COMPOUND[self.nxt().text]
            return cast.Assign(lhs, cast.Binary(op, lhs, self._assign()))
        return lhs

    def _ternary(self):
        cond = self._binary(0)
        if self.at("PUNCT", "?"):                  # cond ? then : els  (right-associative)
            self.nxt()
            then = self._assign()                  # the middle is a full expression
            self.eat("PUNCT", ":")
            els = self._assign()                   # the else nests another conditional
            return cast.Ternary(cond, then, els)
        return cond

    def _binary(self, level: int):
        if level >= len(_PRECEDENCE):
            return self._unary()
        lhs = self._binary(level + 1)
        while self.peek().kind == "OP" and self.peek().text in _PRECEDENCE[level]:
            op = self.nxt().text
            rhs = self._binary(level + 1)
            lhs = cast.Binary(op, lhs, rhs)
        return lhs

    def _unary(self):
        # depth guard: _unary is a recursive-cycle entry (unary `+`/`-`/`*`/`&`/`++` chains, and
        # _unary->_postfix->_primary->`(`->_comma re-entry). Bump/check once per level, then delegate.
        with self._descend():
            return self._unary_inner()

    def _unary_inner(self):
        if self.at("OP", "+"):                             # unary plus is a no-op
            self.nxt()
            return self._unary()
        if self.at("IDENT", "__real__") or self.at("IDENT", "__imag__"):   # GNU complex part extraction
            op = self.nxt().text
            return cast.Unary(op, self._unary())
        if self.at("OP", "&&"):                            # `&&label` -- a label's address as a value (GNU)
            self.nxt()
            return cast.LabelAddr(self.eat("IDENT").text)
        if self.peek().kind == "OP" and self.peek().text in ("++", "--"):  # PREFIX ++a / --a -> yields the NEW value
            op = self.nxt().text[0]
            return cast.IncDec(op, self._unary(), prefix=True)
        if self.peek().kind == "OP" and self.peek().text in ("-", "~", "!", "*", "&"):
            op = self.nxt().text
            return cast.Unary(op, self._unary())
        if self._is_cast():                            # (type)operand — a cast binds at the unary level
            self.eat("PUNCT", "(")
            tref = self._type_spec()
            ptr = 0
            while self.at("OP", "*"):                   # `(uint32_t *)p` — a pointer cast
                ptr += 1
                self.nxt()
            dims = []
            while self.at("PUNCT", "["):                # `(int[N]){...}` / `(int[]){...}` — an array type-name
                self.nxt()
                dims.append(0 if self.at("PUNCT", "]") else parse_int_literal(self.eat("INT").text))
                self.eat("PUNCT", "]")
            self.eat("PUNCT", ")")
            tref = cast.TypeRef(base=tref.base, ptr=ptr, array=tuple(dims),
                                aggregate=tref.aggregate, quals=tref.quals, bit_width=tref.bit_width)
            if self.at("PUNCT", "{"):                   # `(type){ init }` — a C99 compound literal, not a cast
                # supported in rvalue position (`f((struct P){...})`, `x = (struct P){...}`), under `&`
                # (`&(int){v}`), and now with direct postfix on the literal (`(struct P){...}.field`,
                # including nested `.a.b` -- the literal is an lvalue, so it reads like any struct base).
                return self._postfix_tail(cast.CompoundLiteral(tref, self._init_value()))
            return cast.Cast(tref, self._unary())
        return self._postfix()

    def _is_cast(self) -> bool:
        """At `(`, decide whether it opens a cast `(type-name)` rather than a parenthesized expr."""
        if not self.at("PUNCT", "("):
            return False
        nxt = self.peek(1)
        if nxt.kind != "IDENT":
            return False
        w = nxt.text
        return (w in _TYPE_KW or w in _TYPEOF_KW or w in ("struct", "union", "enum", "const", "volatile")
                or is_scalar_name(w) or w in self.typedefs)

    def _postfix(self):
        return self._postfix_tail(self._primary())

    def _postfix_tail(self, node):
        """Apply the postfix operators (`[i]`, `.f`, `->f`, `(args)`) to an already-parsed base — shared
        by `_postfix` (after a primary) and `_unary` (a compound literal, so `(struct P){...}.f` works)."""
        while True:
            if self.at("PUNCT", "["):
                self.nxt()
                idx = self._expr()
                self.eat("PUNCT", "]")
                node = cast.Index(node, idx)
            elif self.at("PUNCT", "."):
                self.nxt()
                node = cast.Member(node, self.eat("IDENT").text, arrow=False)
            elif self.at("OP", "->"):
                self.nxt()
                node = cast.Member(node, self.eat("IDENT").text, arrow=True)
            elif (self.at("PUNCT", "(") and isinstance(node, cast.Name)
                  and node.ident == "va_arg"):                # va_arg(ap, T) -- 2nd operand is a type-name
                self.nxt()
                ap = self._expr()
                self.eat("PUNCT", ",")
                tref = self._type_spec()
                ptr = 0
                while self.at("OP", "*"):
                    ptr += 1
                    self.nxt()
                tref = cast.TypeRef(base=tref.base, ptr=ptr, aggregate=tref.aggregate, quals=tref.quals)
                self.eat("PUNCT", ")")
                node = cast.VaArg(ap, tref)
            elif self.at("PUNCT", "(") and isinstance(node, (cast.Name, cast.Member)):
                self.nxt()
                args = []
                if not self.at("PUNCT", ")"):
                    while True:
                        args.append(self._expr())
                        if self.at("PUNCT", ","):
                            self.nxt()
                            continue
                        break
                self.eat("PUNCT", ")")
                node = (cast.CallExpr(node.ident, tuple(args)) if isinstance(node, cast.Name)
                        else cast.CallMember(node, tuple(args)))   # o->fnptr(args): dispatch table
            elif self.peek().kind == "OP" and self.peek().text in ("++", "--"):  # POSTFIX a++ / a-- -> OLD value
                op = self.nxt().text[0]
                node = cast.IncDec(op, node, prefix=False)
            else:
                break
        return node

    def _sizeof(self):
        """`sizeof ( type-name )` or `sizeof unary-expr` -> a SizeOf node folded to a constant in
        lowering (the type model -- incl. struct/union layout -- is only known there)."""
        self.eat("IDENT", "sizeof")
        if self.at("PUNCT", "("):
            save = self.i
            self.nxt()
            if self._is_decl_start():              # sizeof ( type-name )
                tref = self._type_spec()
                ptr = 0
                while self.at("OP", "*"):          # `sizeof(uint32_t *)` etc.
                    ptr += 1
                    self.nxt()
                self.eat("PUNCT", ")")
                return cast.SizeOf(type=cast.TypeRef(base=tref.base, ptr=ptr,
                                                     aggregate=tref.aggregate, quals=tref.quals))
            self.i = save                          # not a type -> `sizeof ( expr )`
        return cast.SizeOf(expr=self._unary())     # sizeof expr / sizeof (expr)

    def _alignof(self):
        """`_Alignof ( type-name )` / `alignof(...)` -> a constant: the type's alignment (folded in
        lowering from the shared layout model; unlike sizeof, only the type-name form is valid C)."""
        self.nxt()                                 # _Alignof / alignof
        self.eat("PUNCT", "(")
        tref = self._type_spec()
        ptr = 0
        while self.at("OP", "*"):                   # `_Alignof(uint32_t *)`
            ptr += 1
            self.nxt()
        self.eat("PUNCT", ")")
        return cast.AlignOf(cast.TypeRef(base=tref.base, ptr=ptr, aggregate=tref.aggregate,
                                         quals=tref.quals))

    def _generic(self):
        """`_Generic ( assignment-expr , (type-name : assignment-expr | default : assignment-expr)+ )`
        (C11 §6.5.1.1) -- a type-name label / `default` per association; lowering selects on the
        controlling expression's static type and evaluates only the chosen association's expression."""
        self.nxt()                                 # _Generic
        self.eat("PUNCT", "(")
        controlling = self._expr()                 # the controlling expression (unevaluated)
        self.eat("PUNCT", ",")
        assocs = []
        while True:
            if self.at("IDENT", "default"):
                self.nxt()
                tref = None
            else:
                tref = self._type_spec()
                ptr = 0
                while self.at("OP", "*"):           # a pointer type-name label, e.g. `int *`
                    ptr += 1
                    self.nxt()
                tref = cast.TypeRef(base=tref.base, ptr=ptr, aggregate=tref.aggregate, quals=tref.quals)
            self.eat("PUNCT", ":")
            assocs.append((tref, self._expr()))
            if self.at("PUNCT", ","):
                self.nxt()
                continue
            break
        self.eat("PUNCT", ")")
        return cast.Generic(controlling, tuple(assocs))

    def _primary(self):
        if self.at("INT"):
            text = self.nxt().text                            # type from the suffix + magnitude (§6.4.4.1)
            return cast.IntLit(parse_int_literal(text), int_literal_type(text))
        if self.at("CHAR"):                                   # a character constant -> its int value
            return cast.IntLit(parse_char_literal(self.nxt().text))
        if self.at("FLOAT"):                                  # a floating-point literal (1.5 / 3.14f)
            return cast.FloatLit(self.nxt().text)
        if self.at("STRING"):
            text = self.nxt().text
            while self.at("STRING"):                          # adjacent literals concatenate (phase 6),
                text += " " + self.nxt().text                 # kept adjacent so escapes don't merge
            return cast.StringLit(text)
        if self.at("IDENT"):
            w = self.peek().text
            if w in self.enums:                           # an enumerator -> its integer literal
                self.nxt()
                return cast.IntLit(self.enums[w])
            if w == "sizeof":
                return self._sizeof()
            if w in ("_Alignof", "alignof"):
                return self._alignof()
            if w == "_Generic":
                return self._generic()
            if w in KEYWORDS and w != "sizeof":
                raise CParseError(f"unexpected keyword {w!r} in expression", pos=self.peek().pos)
            tk = self.nxt()
            return cast.Name(tk.text, pos=tk.pos)
        if self.at("PUNCT", "("):
            if self.peek(1).text == "{":                      # `({ ... })` -- a GCC statement expression
                self.nxt()
                stmts = self._block()
                self.eat("PUNCT", ")")
                return cast.StmtExpr(stmts)
            self.nxt()
            e = self._comma()                                 # a full expression: the comma OPERATOR is allowed here
            self.eat("PUNCT", ")")
            return e
        tk = self.peek()
        raise CParseError(f"unexpected {tk.kind} {tk.text!r}", pos=tk.pos)


def parse_unit(src: str) -> cast.Unit:
    """Parse one C translation unit (the L1–L4 subset) into the `cast` AST, raising on the first
    error (the compile path needs a well-formed AST)."""
    toks = tokenize(src)
    return _Parser(toks, set()).parse_unit()


def parse_with_recovery(src: str) -> tuple[cast.Unit, list]:
    """Parse with panic-mode recovery, returning the (partial) AST and *every* parse diagnostic the
    run found -- the diagnostics entry uses this so one invocation reports several errors. A CLexError
    still propagates (lexer recovery is a separate concern)."""
    p = _Parser(tokenize(src), set())
    p.recover = True
    unit = p.parse_unit()
    return unit, p.diags
