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
    def parse_unit(self) -> cast.Unit:
        unit = cast.Unit()
        while not self.at("EOF"):
            if self.at("IDENT", "struct") or self.at("IDENT", "union"):
                # Could be an aggregate *definition* or a function returning a struct. Look ahead:
                # `struct TAG {` is a definition; `struct TAG name(` is a function.
                if self.peek(2).kind == "PUNCT" and self.peek(2).text == "{":
                    agg = self._aggregate()
                    unit.aggregates[agg.tag] = agg
                    self.tags.add(agg.tag)
                    continue
            unit.funcs.append(self._func())
        return unit

    def _aggregate(self) -> cast.Aggregate:
        kind = self.eat("IDENT").text                     # struct | union
        tag = self.eat("IDENT").text
        self.eat("PUNCT", "{")
        members = []
        while not self.at("PUNCT", "}"):
            tref = self._type_spec()
            tref, name = self._declarator(tref)
            members.append((tref, name))
            self.eat("PUNCT", ";")
        self.eat("PUNCT", "}")
        self.eat("PUNCT", ";")
        return cast.Aggregate(kind=kind, tag=tag, members=tuple(members))

    # --- types ---
    def _type_spec(self) -> cast.TypeRef:
        quals: list[str] = []
        words: list[str] = []
        aggregate = ""
        base = ""
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
            elif w in _TYPE_KW or is_scalar_name(w):
                words.append(w)
                self.nxt()
            else:
                break
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
            dims.append(parse_int_literal(self.eat("INT").text))
            self.eat("PUNCT", "]")
        return cast.TypeRef(base=base.base, ptr=ptr, array=tuple(dims),
                            aggregate=base.aggregate, quals=base.quals), name

    # --- functions ---
    def _func(self) -> cast.Func:
        ret = self._type_spec()
        ret, name = self._declarator(ret)
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
        return w in _TYPE_KW or is_scalar_name(w) or w in self.tags

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
        if self._is_decl_start():
            return self._decl_stmt()
        expr = self._expr()
        self.eat("PUNCT", ";")
        return cast.ExprStmt(expr)

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
        lhs = self._binary(0)
        if self.at("OP", "="):
            self.nxt()
            return cast.Assign(lhs, self._assign())
        if self.peek().kind == "OP" and self.peek().text in _COMPOUND:
            op = _COMPOUND[self.nxt().text]
            return cast.Assign(lhs, cast.Binary(op, lhs, self._assign()))
        return lhs

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
        return self._postfix()

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
            elif self.at("PUNCT", "(") and isinstance(node, cast.Name):
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
                node = cast.CallExpr(node.ident, tuple(args))
            else:
                break
        return node

    def _primary(self):
        if self.at("INT"):
            return cast.IntLit(parse_int_literal(self.nxt().text))
        if self.at("IDENT"):
            w = self.peek().text
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
