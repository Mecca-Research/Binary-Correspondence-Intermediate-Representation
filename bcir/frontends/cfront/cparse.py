"""Recursive-descent parser for the C-frontend subset (L1–L4) → the `cast` AST.

Grammar (the slice drivers/kernels need): a translation unit is a sequence of struct/union
declarations and function definitions; statements are declarations, assignments, returns, `if`/
`while` (parsed for grammar stability; lowered at L6), and expression statements; expressions use
standard C precedence with `[]` / `.` / `->` / call postfixes.
"""
from __future__ import annotations

from . import cast
from .clex import KEYWORDS, Tok, parse_int_literal, tokenize
from .ctype_model import is_scalar_name


class CParseError(Exception):
    pass


# type-start keywords (a statement beginning with one of these is a declaration).
_TYPE_KW = frozenset({"void", "_Bool", "bool", "char", "short", "int", "long", "unsigned",
                      "signed", "const", "volatile", "static", "inline", "struct", "union"})
# binary operators by ascending precedence groups (C order).
_PRECEDENCE = [
    ("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="), ("<", ">", "<=", ">="),
    ("<<", ">>"), ("+", "-"), ("*", "/", "%"),
]
_COMPOUND = {"+=": "+", "-=": "-", "*=": "*", "/=": "/", "%=": "%", "&=": "&", "|=": "|",
             "^=": "^", "<<=": "<<", ">>=": ">>"}


class _Parser:
    def __init__(self, toks: list[Tok], tags: set):
        self.t = toks
        self.i = 0
        self.tags = tags                      # known struct/union tags (for type detection)
        self.typedefs: dict[str, cast.TypeRef] = {}   # typedef name -> the aliased type
        self.enums: dict[str, int] = {}               # enumerator name -> its integer value

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

    def eat(self, kind: str, text: str | None = None) -> Tok:
        if not self.at(kind, text):
            tk = self.peek()
            raise CParseError(f"expected {text or kind!r}, got {tk.kind} {tk.text!r} at {tk.pos}")
        return self.nxt()

    # --- unit ---
    def _attributes(self) -> dict:
        """Consume `__attribute__((packed))` / `__attribute__((aligned(N)))` / `alignas(N)` runs."""
        attrs: dict = {}
        while True:
            if self.at("IDENT", "__attribute__"):
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
        while not self.at("EOF"):
            if self.at("IDENT", "typedef"):                    # a type alias (resolved at parse time)
                self._typedef(unit)
                continue
            if self.at("IDENT", "enum"):
                save = self.i
                self.nxt()
                tag = self.eat("IDENT").text if self.at("IDENT") else ""
                if self.at("PUNCT", "{"):                      # an enum definition: register the values
                    self._enum_body(tag)
                    self.eat("PUNCT", ";")
                    continue
                self.i = save                                  # `enum tag` used as a type
            if self.at("IDENT", "struct") or self.at("IDENT", "union"):
                save = self.i
                kind = self.nxt().text
                attrs = self._attributes()
                tag = self.eat("IDENT").text if self.at("IDENT") else ""
                if self.at("PUNCT", "{"):                      # an aggregate definition
                    agg = self._aggregate_body(kind, tag, attrs)
                    unit.aggregates[agg.tag] = agg
                    self.tags.add(agg.tag)
                    self.eat("PUNCT", ";")
                    continue
                self.i = save                                  # a struct *type* (func ret / global)
            tref = self._type_spec()
            tref, name = self._declarator(tref)
            if self.at("PUNCT", "("):                          # a function definition
                unit.funcs.append(self._func_body(tref, name))
            else:                                              # a file-scope global variable
                unit.globals.append(self._global(tref, name))
        return unit

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
            if self.at("PUNCT", "{"):                          # an initializer list { e0, e1, ... }
                self.nxt()
                elems = []
                while not self.at("PUNCT", "}"):
                    elems.append(self._expr())
                    if self.at("PUNCT", ","):
                        self.nxt()
                self.eat("PUNCT", "}")
                init = tuple(elems)
            else:
                init = (self._expr(),)
        self.eat("PUNCT", ";")
        return cast.Global(type=tref, name=name, init=init)

    def _aggregate_body(self, kind: str, tag: str, attrs: dict) -> cast.Aggregate:
        self.eat("PUNCT", "{")
        members = []
        while not self.at("PUNCT", "}"):
            self._attributes()                            # member-leading alignas/attrs (consumed)
            tref = self._type_spec()
            tref, name = self._declarator(tref)
            width = 0
            if self.at("PUNCT", ":"):                     # bitfield:  type name : width;
                self.nxt()
                width = parse_int_literal(self.eat("INT").text)
            members.append((tref, name, width))
            self.eat("PUNCT", ";")
        self.eat("PUNCT", "}")
        trailing = self._attributes()                     # `} __attribute__((packed))` (caller eats `;`)
        attrs = {**attrs, **trailing}
        return cast.Aggregate(kind=kind, tag=tag, members=tuple(members),
                              packed=bool(attrs.get("packed")), align=attrs.get("aligned", 0))

    # --- types ---
    def _type_spec(self) -> cast.TypeRef:
        quals: list[str] = []
        words: list[str] = []
        aggregate = ""
        base = ""
        td: cast.TypeRef | None = None
        while self.at("IDENT"):
            w = self.peek().text
            if w in ("const", "volatile"):
                quals.append(w)
                self.nxt()
            elif w in ("static", "inline", "signed"):
                self.nxt()                                # storage/inline ignored; 'signed' implied
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
        if not aggregate:
            base = self._canon_scalar(words)
        return cast.TypeRef(base=base, aggregate=aggregate, quals=tuple(quals))

    @staticmethod
    def _canon_scalar(words: list[str]) -> str:
        if not words:
            raise CParseError("expected a type")
        joined = " ".join(words)
        # canonicalize the legal multi-word combos; otherwise it's a single fixed-width name.
        table = {"unsigned": "unsigned int", "unsigned int": "unsigned int",
                 "long long": "long long", "unsigned long": "unsigned long",
                 "unsigned long long": "unsigned long long", "unsigned char": "unsigned char",
                 "unsigned short": "unsigned short", "signed char": "char"}
        return table.get(joined, words[-1] if len(words) == 1 else joined)

    def _declarator(self, base: cast.TypeRef):
        """Parse `*` pointer prefixes, the name, and `[N]` array suffixes onto `base`."""
        ptr = 0
        while self.at("OP", "*"):
            ptr += 1
            self.nxt()
            while self.at("IDENT", "const") or self.at("IDENT", "volatile"):
                self.nxt()                                # pointer qualifier (ignored for layout)
        name = self.eat("IDENT").text
        dims = []
        while self.at("PUNCT", "["):
            self.nxt()
            dims.append(0 if self.at("PUNCT", "]") else parse_int_literal(self.eat("INT").text))
            self.eat("PUNCT", "]")
        if base.funcptr and ptr == 0 and not dims:        # `binop_fn fn` — keep the funcptr shape
            return base, name
        return cast.TypeRef(base=base.base, ptr=ptr, array=tuple(dims),
                            aggregate=base.aggregate, quals=base.quals), name

    # --- functions ---
    def _func_body(self, ret: cast.TypeRef, name: str) -> cast.Func:
        self.eat("PUNCT", "(")
        params = []
        if not (self.at("PUNCT", ")") or self.at("IDENT", "void") and self.peek(1).text == ")"):
            while True:
                ptype = self._type_spec()
                ptype, pname = self._declarator(ptype)
                params.append(cast.Param(ptype, pname))
                if self.at("PUNCT", ","):
                    self.nxt()
                    continue
                break
        elif self.at("IDENT", "void"):
            self.nxt()
        self.eat("PUNCT", ")")
        body = self._block()
        return cast.Func(ret=ret, name=name, params=tuple(params), body=body)

    # --- statements ---
    def _block(self) -> tuple:
        self.eat("PUNCT", "{")
        stmts = []
        while not self.at("PUNCT", "}"):
            stmts.append(self._stmt())
        self.eat("PUNCT", "}")
        return tuple(stmts)

    def _is_decl_start(self) -> bool:
        if not self.at("IDENT"):
            return False
        w = self.peek().text
        return (w in _TYPE_KW or w == "enum" or is_scalar_name(w)
                or w in self.tags or w in self.typedefs)

    def _stmt(self):
        if self.at("PUNCT", "{"):
            return cast.If(cast.IntLit(1), self._block())     # bare block -> always-true If (rare)
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
        if self._is_decl_start():
            return self._decl_stmt()
        expr = self._expr()
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
            init = cast.ExprStmt(self._expr())
            self.eat("PUNCT", ";")
        cond = cast.IntLit(1) if self.at("PUNCT", ";") else self._expr()
        self.eat("PUNCT", ";")
        step = None if self.at("PUNCT", ")") else cast.ExprStmt(self._expr())
        self.eat("PUNCT", ")")
        body = self._block() if self.at("PUNCT", "{") else (self._stmt(),)
        return cast.For(init, cond, step, body)

    def _switch(self):
        """`switch (disc) { case C: ...; break; case A: case B: ...; break; default: ...; }` ->
        a nested if/else-if chain. A clause's labels OR together (`disc==A || disc==B`); a top-level
        `break;` terminates the clause; `default` is the final `else`. Cross-clause fallthrough (a
        non-empty case without a break) is not modeled -- each clause is independent (the
        Clang-equivalence gate catches any divergence). `disc` is re-evaluated per label (cheap +
        idempotent for the variable/field discriminants drivers use)."""
        self.eat("IDENT", "switch")
        self.eat("PUNCT", "(")
        disc = self._expr()
        self.eat("PUNCT", ")")
        self.eat("PUNCT", "{")
        clauses: list = []                                  # (labels, stmts, is_default)
        labels, stmts, have, is_def = [], [], False, False
        while not self.at("PUNCT", "}"):
            if self.at("IDENT", "case") or self.at("IDENT", "default"):
                if have:                                    # a label after statements -> a new clause
                    clauses.append((labels, stmts, is_def))
                    labels, stmts, have, is_def = [], [], False, False
                if self.at("IDENT", "case"):
                    self.nxt()
                    labels.append(self._expr())
                    self.eat("PUNCT", ":")
                else:
                    self.nxt()
                    self.eat("PUNCT", ":")
                    is_def = True
            elif self.at("IDENT", "break"):                 # the switch terminator (dropped)
                self.nxt()
                self.eat("PUNCT", ";")
                clauses.append((labels, stmts, is_def))
                labels, stmts, have, is_def = [], [], False, False
            else:
                stmts.append(self._stmt())
                have = True
        if labels or stmts or is_def:
            clauses.append((labels, stmts, is_def))
        self.eat("PUNCT", "}")
        default_stmts: tuple = ()
        chained = []                                        # (cond_expr, stmts)
        for labs, sts, isd in clauses:
            if isd:
                default_stmts = tuple(sts)
            else:
                cond = None
                for v in labs:
                    cmp = cast.Binary("==", disc, v)
                    cond = cmp if cond is None else cast.Binary("||", cond, cmp)
                if cond is not None:
                    chained.append((cond, tuple(sts)))
        els: tuple = default_stmts
        for cond, sts in reversed(chained):
            els = (cast.If(cond, sts, els),)
        return els[0] if els else cast.If(cast.IntLit(0), (), ())

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

    def _decl_stmt(self) -> cast.Decl:
        tref = self._type_spec()
        tref, name = self._declarator(tref)
        init = None
        if self.at("OP", "="):
            self.nxt()
            init = self._expr()
        self.eat("PUNCT", ";")
        return cast.Decl(tref, name, init)

    # --- expressions (precedence climbing) ---
    def _expr(self):
        return self._assign()

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
            self.eat("PUNCT", ")")
            return cast.Cast(cast.TypeRef(base=tref.base, ptr=ptr, aggregate=tref.aggregate,
                                          quals=tref.quals), self._unary())
        return self._postfix()

    def _is_cast(self) -> bool:
        """At `(`, decide whether it opens a cast `(type-name)` rather than a parenthesized expr."""
        if not self.at("PUNCT", "("):
            return False
        nxt = self.peek(1)
        if nxt.kind != "IDENT":
            return False
        w = nxt.text
        return (w in _TYPE_KW or w in ("struct", "union", "enum", "const", "volatile")
                or is_scalar_name(w) or w in self.typedefs)

    def _postfix(self):
        node = self._primary()
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

    def _primary(self):
        if self.at("INT"):
            return cast.IntLit(parse_int_literal(self.nxt().text))
        if self.at("IDENT"):
            w = self.peek().text
            if w in self.enums:                           # an enumerator -> its integer literal
                self.nxt()
                return cast.IntLit(self.enums[w])
            if w == "sizeof":
                return self._sizeof()
            if w in ("_Alignof", "alignof"):
                return self._alignof()
            if w in KEYWORDS and w != "sizeof":
                raise CParseError(f"unexpected keyword {w!r} in expression at {self.peek().pos}")
            return cast.Name(self.nxt().text)
        if self.at("PUNCT", "("):
            self.nxt()
            e = self._expr()
            self.eat("PUNCT", ")")
            return e
        tk = self.peek()
        raise CParseError(f"unexpected {tk.kind} {tk.text!r} at {tk.pos}")


def parse_unit(src: str) -> cast.Unit:
    """Parse one C translation unit (the L1–L4 subset) into the `cast` AST."""
    toks = tokenize(src)
    return _Parser(toks, set()).parse_unit()
