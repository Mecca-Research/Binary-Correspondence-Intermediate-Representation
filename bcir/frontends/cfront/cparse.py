"""Recursive-descent parser for the C-frontend subset (L1–L4) → the `cast` AST.

Grammar (the slice drivers/kernels need): a translation unit is a sequence of struct/union
declarations and function definitions; statements are declarations, assignments, returns, `if`/
`while` (parsed for grammar stability; lowered at L6), and expression statements; expressions use
standard C precedence with `[]` / `.` / `->` / call postfixes.
"""

from __future__ import annotations

import dataclasses

from . import cast
from .abi import HOST
from .clex import KEYWORDS, Tok, char_constant, parse_int_literal, tokenize
from .ctype_model import int_literal_type
from .ctype_model import is_scalar_name
from .diagnostics import FixIt, SourceDiagnostic, Span
from .lower import (
    BF_WIDTH,
    DIM_RANGE,
    ENUM_INCOMPLETE,
    ENUM_NOT_INT,
    FN_RET_FP,
    GENERIC_QUALIFIED,
    QUAL_DEEP,
    QUAL_LEVELS,
    TYPE_NAME_NAMED,
    CAST_ARRAY,
    ATOMIC_LITERAL,
    VOLATILE_PTR,
    CLowerError,
    fold_constant,
    layout_aggregates,
)


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
def ast_walk(node):
    """Yield `node` and every cast-node reachable from it (dataclass fields, tuples/lists) --
    a generic expression-tree walk for structural guards (e.g. no-call-in-global-init)."""
    import dataclasses  # noqa: PLC0415

    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, (tuple, list)):
            stack.extend(cur)
        elif dataclasses.is_dataclass(cur) and not isinstance(cur, type):
            yield cur
            stack.extend(getattr(cur, f.name) for f in dataclasses.fields(cur))


_TYPE_KW = frozenset(
    {
        "void",
        "_Bool",
        "bool",
        "char",
        "short",
        "int",
        "long",
        "unsigned",
        "signed",
        "float",
        "double",
        "_Complex",
        "complex",
        "const",
        "volatile",
        "static",
        "extern",
        "inline",
        "_Thread_local",
        "thread_local",
        "struct",
        "union",
        "_BitInt",
    }
)  # C23 bit-precise integer `_BitInt(N)` (a type-start keyword)
# the qualifiers a `*` may carry (C11 6.7.6.1p1), by the name a function type spells each with
_PTR_QUAL = {
    "const": "const",
    "volatile": "volatile",
    "restrict": "restrict",
    "__restrict": "restrict",
    "__restrict__": "restrict",
}


def _qual_levels(levels: tuple, pos: int | None = None) -> tuple:
    """`levels`, the qualifiers of each `*` from the base out, refused when one past the eighth is qualified
    (`QUAL_DEEP`, as the twin's per-level masks refuse it, CF-QUALS)."""
    if any(levels[QUAL_LEVELS:]):
        raise CParseError(QUAL_DEEP, pos=pos)
    return levels


def _join_levels(inner: cast.TypeRef, n: int, outer: tuple) -> tuple:
    """The qualifiers of each `*` of `inner`'s pointer levels, then of `n` more around them (`outer`, one tuple
    per `*`): the `ptr_quals` of a declarator over a type that is a pointer already -- `typedef char *str_t;
    str_t *const *p` is `char **const *` (CF-QUALS). () when none is qualified."""
    levels = tuple(inner.ptr_quals or ((),) * inner.ptr) + tuple(outer or ((),) * n)
    return _qual_levels(levels) if any(levels) else ()


# the qualifiers and storage classes a declaration may spell after its type specifier (C11 6.7p1)
_TRAILING_SPEC = frozenset(
    {"const", "volatile", "static", "extern", "inline", "_Thread_local", "thread_local"}
)
# ... and the ones a file-scope declaration of objects may spell before a struct or union it defines
_OBJECT_SPEC = frozenset({"const", "volatile", "static", "extern", "_Thread_local", "thread_local"})
# thread storage duration (C11 6.2.4p4): the C11 keyword and C23's
_THREAD_SPEC = frozenset({"_Thread_local", "thread_local"})
# `typeof` / `typeof_unqual` (C23) / `__typeof__` (GNU): a type-specifier whose type is the operand's
# -- supported for a type-name operand `typeof(int*)` and a bare in-scope variable `typeof(x)`.
_TYPEOF_KW = frozenset({"typeof", "typeof_unqual", "__typeof__"})
# binary operators by ascending precedence groups (C order).
_PRECEDENCE = [
    ("||",),
    ("&&",),
    ("|",),
    ("^",),
    ("&",),
    ("==", "!="),
    ("<", ">", "<=", ">="),
    ("<<", ">>"),
    ("+", "-"),
    ("*", "/", "%"),
]
_COMPOUND = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "/=": "/",
    "%=": "%",
    "&=": "&",
    "|=": "|",
    "^=": "^",
    "<<=": "<<",
    ">>=": ">>",
}


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
    def __init__(self, toks: list[Tok], tags: set, abi=None):
        self.t = toks
        self.i = 0
        self.tags = tags  # known struct/union tags (for type detection)
        # the target's data model: an integer constant's type and the integer constant expressions folded here (an
        # enumerator, a case label, an array dimension, a designator) take its `long` and pointer widths, as the
        # twin's single pass does (CF-ENUMFOLD)
        self.abi = abi or HOST
        self.typedefs: dict[str, cast.TypeRef] = {}  # typedef name -> the aliased type
        self.enums: dict[str, int] = {}  # enumerator name -> its integer value
        # enum tag -> the integer type its enumeration is compatible with (`_enum_body`; CF-ENUMOBJ)
        self.enum_tags: dict[str, str] = {}
        # the names the function being parsed declares, one map per block scope, innermost last (its parameters
        # first): an object's name to None, an enumerator's to its value. The innermost declaration of a name is
        # the one it denotes (C11 6.2.1p4): an object hides an enumerator of an enclosing scope and a block's own
        # enumerator an outer object or enumerator, to the end of its block (`_enumerator`; CF-ENUMSCOPE, a
        # block's enumerators CF-CONSTEXPR2)
        self.scopes: list[dict] = []
        self.recover = False  # panic-mode recovery: collect diagnostics, don't raise
        self.diags: list = []  # list[SourceDiagnostic] accumulated when recover is on
        self.unit: cast.Unit | None = (
            None  # the unit being built (so a nested inline aggregate registers)
        )
        self._anon_ctr = 0  # synthesizes unique tags for tagless inline aggregates
        self._layout_memo = None  # (the aggregate definitions laid out, their layouts): `_layouts`
        self._depth = 0  # recursive-descent nesting depth (see _MAX_DEPTH / _descend)
        self._declared: set = set()  # the file-scope functions declared so far (`Func.declared`)
        # how C names each anonymous aggregate (`$anonN`, `Unit.anon_spelling`): the file-scope typedef that
        # names it (`typedef struct {...} P;`), else the named member it is the type of -- the aggregate that
        # holds the member, the member's name, and its declarator's `*`s and array dimensions
        self._anon_typedef: dict = {}
        self._anon_member: dict = {}
        self._anon_ptrdef: dict = {}  # ... else a typedef of a pointer to it: `typedef struct {...} *PP;`

    def _descend(self) -> "_Descend":
        """Enter one recursive-grammar level; raise CParseError on overflow (caught as a fallback phase so
        deep input routes to LLVM, never an uncaught RecursionError). Used as `with self._descend():` at
        each recursive-cycle entry point so a pathological deeply-nested input fails cleanly + crash-free,
        matching the C twin's depth guard (both rails route over-deep input to fallback / a clean error)."""
        self._depth += 1
        if self._depth > _MAX_DEPTH:
            self._depth -= (
                1  # unwind the counter on the failing entry so recovery mode stays consistent
            )
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
        self.diags.append(
            SourceDiagnostic("error", str(e), span=span, fixits=fixits, phase="parse")
        )

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
            if (
                kind == "PUNCT" and text in _FIXABLE_PUNCT
            ):  # suggest inserting the missing punctuation
                ins = self._prev_end()  # right after the prior token (Clang-style)
                fix = FixIt(Span(ins, ins), text)
            raise CParseError(
                f"expected {text or kind!r}, got {tk.kind} {tk.text!r}", pos=tk.pos, fixit=fix
            )
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
                self.nxt()  # consume the opening `[[`
                self.nxt()
                while not (
                    self.at("PUNCT", "]")
                    and self.peek(1).kind == "PUNCT"  # scan to the closing `]]`
                    and self.peek(1).text == "]"
                ):
                    if self.at("EOF"):
                        raise CParseError("unterminated [[...]] attribute", pos=self.peek().pos)
                    if self.at("IDENT", "unsequenced") or self.at("IDENT", "reproducible"):
                        attrs["reproducible"] = True  # both hints fold to one fusion-legality flag
                    self.nxt()  # robust to args/namespaces: skip every other token
                self.nxt()  # eat the closing `]]`
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
                        attrs["aligned"] = self._const_int()
                        self.eat("PUNCT", ")")
                    if self.at("PUNCT", ","):
                        self.nxt()
                self.eat("PUNCT", ")")
                self.eat("PUNCT", ")")
            elif self.at("IDENT", "alignas") or self.at("IDENT", "_Alignas"):
                self.nxt()
                self.eat("PUNCT", "(")
                attrs["aligned"] = self._const_int()
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
            try:  # panic-mode recovery: keep going past
                self._toplevel_item(unit)  # a bad declaration to report the next
            except CParseError as e:
                self._record(e)
                self._sync_toplevel()
        unit.anon_spelling = {
            tag: sp for tag in unit.aggregates if (sp := self._anon_spell(tag, unit)) is not None
        }
        unit.typedef_names = frozenset(self.typedefs)
        return unit

    def _anon_spell(self, tag: str, unit: cast.Unit) -> str | None:
        """How C names the anonymous aggregate `tag`, or None when it cannot: the name a file-scope typedef
        gives it (`typedef struct {...} P;` -> `P`; `typedef struct {...} *PP;` -> `__typeof__(*(PP)0)`, the
        pointee of a null pointer of that type), or the type of the named member it is -- `struct {...}
        m[N];` in `P` -> `__typeof__(((P *)0)->m[0])`, a `*` before the member for each pointer level --
        through the spelling of the aggregate that holds the member, up to one a typedef names or that has a
        tag. The emit spells an anonymous aggregate by its synthesized tag, which no compiler knows (CF-ANON);
        the twin's `anon_spelling` walks the same chain."""
        if not tag.startswith("$anon"):
            return None
        chain, top = [], tag
        while (
            top.startswith("$anon")
            and top not in self._anon_typedef
            and top not in self._anon_ptrdef
        ):
            if top not in self._anon_member:
                return None
            chain.append(top)
            top = self._anon_member[top][0]
        if not chain and tag not in self._anon_typedef and tag not in self._anon_ptrdef:
            return None
        if top in self._anon_typedef:
            spelling = self._anon_typedef[top]
        elif top in self._anon_ptrdef:  # only a pointer typedef names it: the pointee of a null one
            pname, stars = self._anon_ptrdef[top]
            spelling = f"__typeof__({'*' * stars}({pname})0)"
        else:
            spelling = f"{unit.aggregates[top].kind} {top}"
        for x in reversed(chain):  # the innermost aggregate's member first
            _parent, member, stars, dims = self._anon_member[x]
            spelling = f"__typeof__({'*' * stars}(({spelling} *)0)->{member}{'[0]' * dims})"
        return spelling

    def _spelled(self, start: int, end: int) -> str:
        """The tokens `start` to `end` as C text, one space apart -- what they spell, their own spellings (a
        literal's prefix and quotes, an attribute) kept: the linkable emit's copy of a type definition."""
        return " ".join(t.text for t in self.t[start:end])

    def _toplevel_item(self, unit: cast.Unit) -> None:
        """Parse one top-level item (typedef / enum / aggregate definition / function / global)."""
        # A C23 `[[unsequenced]]`/`[[reproducible]]` (or any leading attribute) may precede a function /
        # global. `_attributes()` returns {} when the next token is not an attribute, so this is a no-op
        # before a typedef / enum / struct / plain type -- NON-DISTURBANCE for every existing item.
        lead = self._attributes()
        self.storage = set()  # per-item storage-class record (see _type_spec)
        if self.at("IDENT", "typedef"):  # a type alias (resolved at parse time)
            start = self.i
            self._typedef(unit)
            unit.type_defs.append(self._spelled(start, self.i))
            return
        if self.at("IDENT", "enum"):
            save = self.i
            self.nxt()
            tag = self.eat("IDENT").text if self.at("IDENT") else ""
            if self.at("PUNCT", "{"):  # an enum definition: register the values
                self._enum_body(tag)
                self.eat("PUNCT", ";")
                unit.type_defs.append(self._spelled(save, self.i))
                return
            self.i = save  # `enum tag` used as a type
        if self._aggregate_definition(unit):  # `[static] struct t { ... } [a, b];`
            return
        if self.at("IDENT", "struct") or self.at("IDENT", "union"):
            save = self.i
            self.nxt()
            self._attributes()
            tag = self.eat("IDENT").text if self.at("IDENT") else ""
            # `struct tag;` -- a forward declaration: the tag names an incomplete type until
            # (unless) it is defined (CF-SELFREF)
            if tag and self.at("PUNCT", ";"):
                self.nxt()
                unit.type_defs.append(self._spelled(save, self.i))
                return
            self.i = save  # a struct *type* (func ret / global)
        base = self._type_spec()
        if self._fn_returning_fp_at():  # `T (*f(P))(Q)` (CF-FPRET)
            raise CParseError(FN_RET_FP, pos=self.peek().pos)
        # a global may be a function pointer, an array of them or a pointer to one (CF-FPTAB)
        tref, name = self._declarator_or_funcptr(base)
        if self.at("PUNCT", "("):  # a function definition (or a prototype)
            fn = self._func_body(
                tref,
                name,
                reproducible=bool(lead.get("reproducible")),
                static="static" in self.storage,
            )
            if fn is not None:  # None == a prototype (recorded in protos)
                unit.funcs.append(fn)
        else:  # file-scope global variables: one per declarator off the specifier (C11 6.7p1)
            self._globals(unit, base, tref, name)

    def _globals(self, unit: cast.Unit, base: cast.TypeRef, tref: cast.TypeRef, name: str) -> None:
        """The declarators of one file-scope declaration, `T a[3], *p, b = 5;`, each a global of its own
        type -- its `*`s and dimensions its own, the specifier's storage class every one's -- then its `;`.
        The first declarator (`tref`, `name`) is already parsed. The twin's `p_global`."""
        extern, static = "extern" in self.storage, "static" in self.storage
        while True:
            unit.globals.append(self._global(tref, name, extern=extern, static=static))
            if not self.at("PUNCT", ","):
                break
            self.nxt()
            tref, name = self._declarator_or_funcptr(base)
        self.eat("PUNCT", ";")

    def _aggregate_definition(self, unit: cast.Unit) -> bool:
        """A struct or union defined at file scope -- `struct t { ... };`, and one whose declaration goes on to
        declare objects of it, `static struct t { ... } a, b[2];` (C11 6.7.2.1): its storage classes and
        qualifiers, then the definition, then the declarators, globals of `struct t`. An untagged one declaring
        objects is refused: the emit names a struct by its tag. False, the cursor unmoved, when the item is not
        a definition. The twin's `try_top_decl`."""
        save = self.i
        storage, quals = set(), []
        while self.peek().kind == "IDENT" and self.peek().text in _OBJECT_SPEC:
            word = self.nxt().text
            (quals.append if word in ("const", "volatile") else storage.add)(word)
        if not (self.at("IDENT", "struct") or self.at("IDENT", "union")):
            self.i = save
            return False
        head = self.i
        kind = self.nxt().text
        attrs = self._attributes()
        tag = self.eat("IDENT").text if self.at("IDENT") else ""
        if not self.at("PUNCT", "{"):
            self.i = save
            return False
        agg = self._aggregate_body(kind, tag, attrs)
        unit.aggregates[agg.tag] = agg
        self.tags.add(agg.tag)
        if tag:  # the definition alone -- its storage class and declarators are the globals' (CF-LINKEMIT)
            unit.type_defs.append(self._spelled(head, self.i) + " ;")
        if self.at("PUNCT", ";"):
            self.nxt()
            return True
        if not tag:
            raise CParseError(
                "an object of an untagged struct or union at file scope is not supported",
                pos=self.peek().pos,
            )
        self.storage = storage
        base = cast.TypeRef(base=agg.tag, aggregate=kind, quals=tuple(quals))
        tref, name = self._declarator(base)
        self._globals(unit, base, tref, name)
        return True

    def _typedef(self, unit: cast.Unit) -> None:
        """`typedef <type> <name>;` -- register `name` -> the aliased type (resolved at parse time,
        so the lowered claim graph is identical to spelling the underlying type out). Handles scalar/
        pointer/qualified aliases, `typedef struct/union [tag] {...} Name;`, and `typedef enum {...}
        Name;`."""
        self.eat("IDENT", "typedef")
        if self.at("IDENT", "enum"):
            base = cast.TypeRef(base=self._enum_spec())  # its compatible integer type (CF-ENUMOBJ)
        elif self.at("IDENT", "struct") or self.at("IDENT", "union"):
            kind = self.nxt().text
            attrs = self._attributes()
            tag = self.eat("IDENT").text if self.at("IDENT") else ""
            if self.at("PUNCT", "{"):
                if not tag:
                    # anonymous: a synthesized tag, unique as an inline member's is -- two of them had
                    # shared the empty tag, and the second one's layout was lost
                    tag = f"$anon{self._anon_ctr}"
                    self._anon_ctr += 1
                agg = self._aggregate_body(kind, tag, attrs)
                unit.aggregates[agg.tag] = agg
                self.tags.add(agg.tag)
                tag = agg.tag
            base = cast.TypeRef(base=tag, aggregate=kind)
        else:
            base = self._type_spec()
        # `typedef T *(*NAME)(PARAMS);`: a pointer return (CF-FPRET)
        base = self._fp_return_stars(base)
        if self.at("PUNCT", "(") and self.peek(1).kind == "OP" and self.peek(1).text == "*":
            tref, name = self._funcptr_declarator(base)  # typedef RET (*NAME)(PARAMS);
        else:
            tref, name = self._declarator(base)
            if (
                tref.aggregate
                and tref.base.startswith("$anon")
                and not tref.ptr
                and not tref.array
                and tref.base not in self._anon_member
            ):  # `typedef struct {...} P;`: C names the aggregate `P`
                self._anon_typedef.setdefault(tref.base, name)
            elif tref.aggregate and tref.base.startswith("$anon") and tref.ptr and not tref.array:
                # `typedef struct {...} *PP;`: C names the aggregate through the pointer
                self._anon_ptrdef.setdefault(tref.base, (name, tref.ptr))
        self.typedefs[name] = tref
        self.eat("PUNCT", ";")

    def _funcptr_declarator(self, ret: cast.TypeRef, abstract: bool = False):
        """`( * NAME ) ( param-type-list )` — a function-pointer declarator. Returns a funcptr TypeRef
        (carrying the return + parameter types for faithful emit) and the declared NAME. The name is
        also stashed in ``base`` so a later use of the alias renders verbatim. `abstract`: a parameter's,
        `( * ) ( ... )`, whose name may be left out."""
        self.eat("PUNCT", "(")
        self.eat("OP", "*")
        # the function pointer's own qualifiers (`(*const fp)`), then the `*`s of a pointer to one -- `(**pp)(...)`
        # (CF-FPTAB) -- each with its own (CF-QUALS)
        own = self._star_quals()
        ptr, pq = self._stars()
        name = self.eat("IDENT").text if not abstract or self.at("IDENT") else ""
        dims = []  # `(*t[2])(...)`: an array of function pointers, each dimension a constant
        while self.at("PUNCT", "["):
            self.nxt()
            dim = 0 if self.at("PUNCT", "]") else self._dim(self._assign())
            if not isinstance(dim, int):
                raise CParseError(
                    "a variable-length array of function pointers is not supported",
                    pos=self.peek().pos,
                )
            if len(dims) == 3:  # at most three, as for any array here (the twin's `fp_inline_decl`)
                raise CParseError(
                    "an array of function pointers of more than 3 dimensions is not supported",
                    pos=self.peek().pos,
                )
            dims.append(dim)
            self.eat("PUNCT", "]")
        self.eat("PUNCT", ")")
        self.eat("PUNCT", "(")
        params: list[cast.TypeRef] = []
        variadic = False
        if self.at("IDENT", "void") and self.peek(1).text == ")":
            self.nxt()
        elif not self.at("PUNCT", ")"):
            while True:
                if params and self.at("PUNCT", "..."):  # `(T, ...)`: a variadic function (CF-FPRET)
                    self.nxt()
                    variadic = True
                    break
                pt = self._type_spec()
                n, levels = self._stars()  # a pointer parameter, each `*` with its qualifiers
                if n:
                    pt = cast.TypeRef(
                        base=pt.base,
                        ptr=pt.ptr + n,
                        aggregate=pt.aggregate,
                        quals=pt.quals,
                        ptr_quals=_join_levels(pt, n, levels),
                    )
                if self.at("IDENT"):  # an optional parameter name (ignored)
                    self.nxt()
                params.append(pt)
                if self.at("PUNCT", ","):
                    self.nxt()
                    continue
                break
        self.eat("PUNCT", ")")
        fp = cast.TypeRef(
            base=name, funcptr=True, func_ret=ret, func_params=tuple(params), func_variadic=variadic
        )
        # a function pointer's `quals` are its own, the pointee of a `*` around it (CF-QUALS)
        return dataclasses.replace(fp, ptr=ptr, array=tuple(dims), quals=own, ptr_quals=pq), name

    def _is_funcptr_declarator(self, abstract: bool = False, at: int = 0) -> bool:
        """True if the cursor is at `( * NAME ) (` — a function-pointer declarator (`int (*g)(int)`),
        as opposed to the row-pointer `( * NAME ) [` form that `_declarator` handles -- or at a pointer to
        one, `( * * NAME ) (`, or an array of them, `( * NAME [ N ] ) (` (CF-FPTAB). `abstract`: a
        parameter's, which may leave the name out -- `( * ) (`. `at`: tokens past the cursor."""
        if not (
            self.peek(at).kind == "PUNCT"
            and self.peek(at).text == "("
            and self.peek(at + 1).kind == "OP"
            and self.peek(at + 1).text == "*"
        ):
            return False
        k = self._skip_stars(at + 1)
        if self.peek(k).kind == "IDENT":
            k += 1
        elif not abstract:
            return False
        while self.peek(k).kind == "PUNCT" and self.peek(k).text == "[":  # a dimension's tokens
            depth = 0
            while True:
                tk = self.peek(k)
                if tk.kind == "EOF":
                    return False
                if tk.kind == "PUNCT" and tk.text == "[":
                    depth += 1
                elif tk.kind == "PUNCT" and tk.text == "]":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            k += 1
        return (
            self.peek(k).kind == "PUNCT"
            and self.peek(k).text == ")"
            and self.peek(k + 1).kind == "PUNCT"
            and self.peek(k + 1).text == "("
        )

    def _declarator_or_funcptr(self, base: cast.TypeRef, abstract: bool = False):
        """A declarator (param or local position) that may be an inline function-pointer
        `RET (*NAME)(PARAMS)` — for which there is no typedef name, so the full signature is captured.
        `abstract`: a parameter's, whose name may be left out (`uint32_t *`, `uint32_t (*)(uint32_t)`);
        the name is then ""."""
        base = self._fp_return_stars(base, abstract)
        if self._is_funcptr_declarator(abstract):
            return self._funcptr_declarator(base, abstract)
        return self._declarator(base, abstract)

    def _fn_returning_fp_at(self) -> bool:
        """Whether the cursor is at `( * NAME (` -- the declarator of a function that returns a function pointer,
        `T (*f(P))(Q)`, which neither rail parses (`FN_RET_FP`)."""
        if not (self.at("PUNCT", "(") and self.peek(1).kind == "OP" and self.peek(1).text == "*"):
            return False
        k = self._skip_stars(1)
        return (
            self.peek(k).kind == "IDENT"
            and self.peek(k + 1).kind == "PUNCT"
            and self.peek(k + 1).text == "("
        )

    def _fp_return_stars(self, base: cast.TypeRef, abstract: bool = False) -> cast.TypeRef:
        """The `*`s between a specifier and a function-pointer declarator, `T *(*pf)(T *)`, belong to the
        function's return type -- a pointer (CF-FPRET): taken into `base`, the cursor left on the
        declarator's `(`. Anything else is left for the declarator, `base` unchanged."""
        k = self._skip_stars(0)
        if not k or not self._is_funcptr_declarator(abstract, at=k):
            return base
        n, levels = self._stars()  # each `*` of the return with its qualifiers (CF-QUALS)
        return dataclasses.replace(base, ptr=base.ptr + n, ptr_quals=_join_levels(base, n, levels))

    def _enum_spec(self) -> str:
        """`enum [tag] [{...}]` at the parser's position: the integer type the enumeration is compatible with, as
        `_enum_body` decides it -- a tag without a body names the one its definition gave; one no definition has
        given is refused (`ENUM_INCOMPLETE`), never read as an `int`."""
        tk = self.eat("IDENT", "enum")
        tag = self.eat("IDENT").text if self.at("IDENT") and not self.at("PUNCT", "{") else ""
        if self.at("PUNCT", "{"):
            return self._enum_body(tag)
        if tag not in self.enum_tags:
            raise CParseError(ENUM_INCOMPLETE, pos=tk.pos)
        return self.enum_tags[tag]

    def _enum_body(self, tag: str) -> str:
        """Parse `{ A, B = expr, C }` -- assign each enumerator its C value (prev+1, or the given
        constant, folded in C's types: `_const_eval`) and register it so a later use resolves to that
        integer literal. An enumeration constant is an `int` (C11 6.4.4.3): a value no int holds, given or
        counted on from INT_MAX, is refused (`ENUM_NOT_INT`, 6.7.2.2p2), never cut to one. Returns the integer
        type the enumerated type is compatible with (6.7.2.2p4) -- `unsigned int` where no enumerator is negative
        on a target that makes it so (`TargetABI.enum_unsigned`: GCC and Clang on System V), else `int` -- and
        records it for the tag (CF-ENUMOBJ: both rails had typed every enum object `int`)."""
        self.eat("PUNCT", "{")
        value = 0
        negative = False
        while not self.at("PUNCT", "}"):
            tk = self.eat("IDENT")
            if self.at("OP", "="):
                self.nxt()
                value = self._const_eval(self._ternary())  # the full conditional-expression
                # level (C 6.7.2.2: an enum value
                # is a constant-expression)
            if not -(1 << 31) <= value < 1 << 31:
                raise CParseError(ENUM_NOT_INT, pos=tk.pos)
            # in scope just after its own enumerator (6.2.1p7): a block's to the end of the block (CF-CONSTEXPR2)
            (self.scopes[-1] if self.scopes else self.enums)[tk.text] = value
            negative = negative or value < 0
            value += 1
            if self.at("PUNCT", ","):
                self.nxt()
        self.eat("PUNCT", "}")
        compatible = "unsigned int" if self.abi.enum_unsigned and not negative else "int"
        if tag:
            self.enum_tags[tag] = compatible
        return compatible

    def _dim(self, e):
        """An array declarator's dimension: its value when it is an integer constant expression -- the array is
        then no variable-length one, whatever the expression's form (`N`, `2 + 1`, `N * 2`; C11 6.7.6.2p4) --
        else the expression, a runtime dimension. A constant outside 0..INT_MAX is refused (`DIM_RANGE`), as the
        twin's `ce_dim` refuses one (CF-ENUMFOLD)."""
        try:
            v = fold_constant(e, self.abi, layouts=self._layouts).v
        except CLowerError:
            return e
        return self._dim_in_range(v)

    def _dim_in_range(self, v: int) -> int:
        """A constant dimension `v`, refused outside 0..INT_MAX (`DIM_RANGE`) -- the one range check of `_dim` and
        `_const_dim`, as the twin's `ce_dim` is."""
        if not 0 <= v < 1 << 31:
            raise CParseError(DIM_RANGE, pos=self.peek().pos)
        return v

    def _const_eval(self, node) -> int:
        """Fold an integer constant expression -- an enumerator's value, a case label, a designator, a
        type-name's dimension -- in C's own types on the target, as C evaluates one (`lower.fold_constant`,
        the predicate a static's initializer folds with; CF-ENUMFOLD): an operand C does not evaluate (`0 &&
        e`, the arm `?:` does not take) is not folded. One that is no integer constant expression, or whose
        arithmetic C leaves undefined (a division by zero, a signed overflow), is refused for that one reason
        (`ICE_NOT`). A `sizeof` or `_Alignof` of a type-name reads the unit's aggregates laid out so far
        (`_layouts`; CF-CONSTEXPR2)."""
        try:
            return fold_constant(node, self.abi, layouts=self._layouts).v
        except CLowerError as e:
            raise CParseError(str(e), pos=self.peek().pos) from None

    def _layouts(self) -> dict:
        """The unit's aggregates defined so far, laid out on the target (`lower.layout_aggregates`), for a `sizeof` or
        `_Alignof` an integer constant expression folds (CF-CONSTEXPR2): laid out again only when a definition was
        added or completed since. A definition the layout refuses is refused here for that reason."""
        defs = self.unit.aggregates if self.unit is not None else {}
        seen = tuple(defs.values())
        if (
            self._layout_memo is None
            or len(self._layout_memo[0]) != len(seen)
            or any(a is not b for a, b in zip(self._layout_memo[0], seen))
        ):
            try:
                self._layout_memo = (seen, layout_aggregates(defs, self.abi))
            except CLowerError as e:
                raise CParseError(str(e), pos=self.peek().pos) from None
        return self._layout_memo[1]

    def _const_int(self) -> int:
        """The integer constant expression at the cursor -- a conditional-expression -- folded (`_const_eval`), where
        C takes one and both rails had read a literal only: a bit-field's width, `_BitInt(N)`'s width, `aligned(N)`
        and `alignas(N)` (CF-CONSTEXPR2; the twin's `ce_fold`). `enum { W = 3 }; ... uint32_t a : W;` had been refused
        here and laid out with a width of 0 by the twin -- a member as wide as its type."""
        return self._const_eval(self._ternary())

    def _const_dim(self) -> int:
        """... and a dimension where C takes only a constant one -- a compound literal's `(T[N]){...}` (6.5.2.5p1)
        and a row pointer's `(*p)[N]` -- refused outside 0..INT_MAX (`DIM_RANGE`), as `_dim` refuses one (the twin's
        `ce_dim`)."""
        return self._dim_in_range(self._const_int())

    def _bit_width(self, named: bool) -> int:
        """A bit-field's width, the cursor past its `:` (`_const_int`): a negative one, or a named one of zero, is
        refused (`BF_WIDTH`, C11 6.7.2.1p4) -- one wider than its type when the aggregate is laid out."""
        w = self._const_int()
        if w < 0 or (named and w == 0):
            raise CParseError(BF_WIDTH, pos=self.peek().pos)
        return w

    def _global(
        self, tref: cast.TypeRef, name: str, extern: bool = False, static: bool = False
    ) -> cast.Global:
        init = None
        if self.at("OP", "="):
            self.nxt()
            # the initializer a local takes (CF-GBRACE): an expression, or a brace list whose entries are
            # expressions or lists, positional or designated (`[i] =`, `.m =`, a chain) -- walked in lowering
            # as C walks the current object; a flat list of expressions had refused a nested one
            init = self._init_value()
            # C 6.7.10: a file-scope initializer must be a CONSTANT expression -- a CALL in it
            # is invalid C (gcc/clang: "initializer element is not constant"). Reject it here
            # rather than mislower: before prototypes landed this was caught incidentally (the
            # `T k(void);` line was a parse error); now the prototype parses, so guard the
            # construct itself. Routes to fallback under --fallback like any rejected form.
            for node in ast_walk(init):
                if isinstance(node, (cast.CallExpr, cast.CallMember, cast.CallPtr)):
                    raise CParseError(
                        f"file-scope initializer of {name!r} calls a "
                        f"function (not a constant expression)",
                        pos=self.peek().pos,
                    )
        return cast.Global(
            type=tref,
            name=name,
            init=init,
            extern_decl=extern,
            static_storage=static,
            thread_storage=bool(self.storage & _THREAD_SPEC),
        )

    def _aggregate_body(self, kind: str, tag: str, attrs: dict) -> cast.Aggregate:
        self.eat("PUNCT", "{")
        members = []
        while not self.at("PUNCT", "}"):
            matt = self._attributes()  # member-leading `_Alignas(N)`/`alignas(N)`/
            malign = matt.get(
                "aligned", 0
            )  # `__attribute__((aligned(N)))` -- over-aligns this member
            inline_agg = False  # `struct {...}` / `union {...}` defined inline as a member
            if self.at("IDENT", "struct") or self.at("IDENT", "union"):
                save = self.i
                ikind = self.nxt().text
                iattrs = self._attributes()
                itag = (
                    self.eat("IDENT").text
                    if (self.at("IDENT") and not self.at("PUNCT", "{"))
                    else ""
                )
                if self.at("PUNCT", "{"):  # an INLINE aggregate definition (anonymous or tagged)
                    if not itag:
                        itag = f"$anon{self._anon_ctr}"
                        self._anon_ctr += 1
                    iagg = self._aggregate_body(ikind, itag, {**iattrs, **matt})
                    self.unit.aggregates[iagg.tag] = iagg
                    self.tags.add(iagg.tag)
                    base = cast.TypeRef(base=iagg.tag, aggregate=ikind)
                    inline_agg = True
                else:
                    self.i = save  # `struct Tag member;` -- not an inline definition
            if inline_agg and self.at(
                "PUNCT", ";"
            ):  # ANONYMOUS member: an inline aggregate, no declarator --
                self.nxt()  # its leaves promote into THIS aggregate (name "" marks it)
                members.append((base, "", 0, malign))
                continue
            if not inline_agg:
                base = self._type_spec()
            while True:  # one or more declarators off one specifier:
                if self.at("PUNCT", ":"):  # an UNNAMED/zero-width bitfield `type : width;` (no
                    self.nxt()  # declarator): it positions the layout cursor but is not
                    w = self._bit_width(False)  # accessible -- name "" + a scalar base marks it
                    members.append((base, "", w, malign))
                    if self.at("PUNCT", ","):
                        self.nxt()
                        continue
                    break
                fbase = self._fp_return_stars(base)  # `T *(*name)(params)` (CF-FPRET)
                if self.at("PUNCT", "(") and self.peek(1).kind == "OP" and self.peek(1).text == "*":
                    tref, name = self._funcptr_declarator(
                        fbase
                    )  # `RET (*name)(params)` -- a funcptr member
                else:  # (8-byte; set from a funcptr value, called
                    tref, name = self._declarator(
                        base
                    )  #   `unsigned x, y, z;` / `unsigned a:3, b:5;`  `o->fn(a)`)
                    if inline_agg and base.base.startswith("$anon"):
                        # `struct {...} m;`: the anonymous type is m's, its first declarator's
                        member = (tag, name, tref.ptr, len(tref.array))
                        self._anon_member.setdefault(base.base, member)
                width = 0
                if self.at("PUNCT", ":"):  # bitfield:  type name : width;
                    self.nxt()
                    width = self._bit_width(True)
                members.append(
                    (tref, name, width, malign)
                )  # `malign` applies to every declarator here
                if self.at("PUNCT", ","):  # another member off the same specifier
                    self.nxt()
                    continue
                break
            self.eat("PUNCT", ";")
        self.eat("PUNCT", "}")
        trailing = self._attributes()  # `} __attribute__((packed))` (caller eats `;`)
        attrs = {**attrs, **trailing}
        return cast.Aggregate(
            kind=kind,
            tag=tag,
            members=tuple(members),
            packed=bool(attrs.get("packed")),
            align=attrs.get("aligned", 0),
        )

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
        bit_width = 0  # a C23 `_BitInt(N)` width (set when a `_BitInt` is seen)
        saw_bitint = False  # `_BitInt` seen (separate from bit_width, since N==0 is
        td: cast.TypeRef | None = (
            None  #   falsy -- the validation below rejects it as out-of-range)
        )
        while self.at("IDENT"):
            w = self.peek().text
            if w == "_BitInt":  # C23 `_BitInt ( N )` -- a bit-precise integer type
                self.nxt()
                self.eat("PUNCT", "(")
                bit_width = self._const_int()
                self.eat("PUNCT", ")")
                if saw_bitint:  # a second `_BitInt` in one specifier run -> fallback
                    raise CParseError("duplicate `_BitInt` type specifier")
                saw_bitint = True
                words.append("_BitInt")
                continue
            if (
                w == "_Atomic" and self.peek(1).text == "("
            ):  # `_Atomic ( type-name )` -- atomic type specifier
                self.nxt()
                self.eat("PUNCT", "(")
                inner = self._type_spec()
                ip, ipq = self._stars()
                self.eat("PUNCT", ")")
                return cast.TypeRef(
                    base=inner.base,
                    ptr=ip,
                    array=inner.array,
                    aggregate=inner.aggregate,
                    quals=tuple(quals) + ("_Atomic",) + inner.quals,
                    ptr_quals=ipq,
                )
            if w in ("const", "volatile", "_Atomic"):
                quals.append(w)
                self.nxt()
            elif w in ("static", "extern", "_Thread_local", "thread_local", "inline"):
                self.storage.add(w)  # recorded (linkable emit reads `extern` and
                self.nxt()  # `static`); otherwise storage/inline ignored
            elif w in ("struct", "union"):
                aggregate = w
                self.nxt()
                base = self.eat("IDENT").text  # the tag
                break
            elif w == "enum":  # `enum [tag] [{...}]` -> its compatible integer type (CF-ENUMOBJ)
                base = self._enum_spec()
                break
            elif (
                w in _TYPEOF_KW and not words and not aggregate
            ):  # typeof(type-name) / typeof(variable)
                self.nxt()
                self.eat("PUNCT", "(")
                if self._is_decl_start():  # typeof ( type-name ), incl. `typeof(int*)`
                    inner = self._type_spec()
                    ip, ipq = self._stars()
                    self.eat("PUNCT", ")")
                    return cast.TypeRef(
                        base=inner.base,
                        ptr=ip,
                        array=inner.array,
                        aggregate=inner.aggregate,
                        quals=tuple(quals) + inner.quals,
                        ptr_quals=ipq,
                    )
                expr = self._expr()  # typeof ( expression ) -- a general operand
                self.eat("PUNCT", ")")
                if isinstance(expr, cast.Name):  # a bare in-scope variable keeps the fast path
                    return cast.TypeRef(base="", typeof_var=expr.ident, quals=tuple(quals))
                return cast.TypeRef(base="", typeof_expr=expr, quals=tuple(quals))
            elif (
                w in ("va_list", "__builtin_va_list") and not words and not aggregate
            ):  # variadic cursor type
                self.nxt()
                base = "va_list"
                break
            elif not words and self._typedef_visible(w):  # a typedef name -> expand the alias
                td = self.typedefs[w]
                self.nxt()
                break
            elif w in _TYPE_KW or is_scalar_name(w):
                if w not in _TYPE_KW and words:
                    # `typedef unsigned long size_t;`: a <stdint.h> or <stddef.h> name is a typedef name, no keyword,
                    # so after a type it is the declarator, as any name is -- a unit may declare it itself (CF-RTFP;
                    # the twin's `p_type`), and its typedef then names the type, as the twin's does
                    break
                words.append(w)
                self.nxt()
                if w not in _TYPE_KW:  # ... and alone it is the whole type
                    break
            else:
                break
        # declaration specifiers come in any order (C11 6.7p1): a qualifier or storage class after a
        # struct/union tag or a typedef name (`struct s static x`, `uint32_t_alias volatile y`) is the
        # same specifier it is before one -- the scalar keyword run above already took those it met
        while self.at("IDENT") and self.peek().text in _TRAILING_SPEC:
            w = self.nxt().text
            if w in ("const", "volatile"):
                quals.append(w)
            else:
                self.storage.add(w)
        if td is not None:  # merge the alias with any leading quals
            # a qualifier of a pointer typedef qualifies the pointer, not what it points to (C11 6.7.8p3: a
            # typedef is the type it names): `const str_t p` of `typedef char *str_t` is `char *const p`, and a
            # `volatile` one is a volatile pointer (CF-QUALS)
            if "volatile" in quals and (td.ptr or td.funcptr):
                raise CParseError(VOLATILE_PTR, pos=self.peek().pos)
            if td.funcptr:  # a function-pointer alias carries its own shape
                if "const" in quals and not td.ptr and not td.array:
                    return dataclasses.replace(td, quals=tuple(sorted({"const", *td.quals})))
                if "const" in quals:  # `const opp_t pp` of `typedef op_t *opp_t`: its outer `*`
                    levels = list(td.ptr_quals or ((),) * td.ptr)
                    if levels:
                        levels[-1] = tuple(sorted({"const", *levels[-1]}))
                        return dataclasses.replace(td, ptr_quals=_qual_levels(tuple(levels)))
                return td
            levels = td.ptr_quals
            if td.ptr and "const" in quals:
                outer = list(td.ptr_quals or ((),) * td.ptr)
                outer[-1] = tuple(sorted({"const", *outer[-1]}))
                levels, quals = _qual_levels(tuple(outer)), [q for q in quals if q != "const"]
            return cast.TypeRef(
                base=td.base,
                ptr=td.ptr,
                array=td.array,
                aggregate=td.aggregate,
                quals=tuple(quals) + td.quals,
                ptr_quals=levels,
            )
        if saw_bitint:  # C23 `_BitInt(N)`: the spelling carries N + signedness;
            return self._bitint_typeref(
                words, bit_width, quals
            )  # lowering builds the `bitint` CType from it
        if not aggregate and not base:  # `enum [tag]` already set base="int"; only
            base = self._canon_scalar(words)  # canonicalize a scalar keyword run otherwise
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
        if not (2 <= bit_width <= 64):  # the faithful-emit subset (N<2 / N>64 -> fallback)
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
            return "int"  # a bare `signed` == int
        joined = " ".join(words)
        # canonicalize the legal multi-word combos; otherwise it's a single fixed-width name.
        table = {
            "unsigned": "unsigned int",
            "unsigned int": "unsigned int",
            "long long": "long long",
            "unsigned long": "unsigned long",
            "unsigned long long": "unsigned long long",
            "unsigned char": "unsigned char",
            "unsigned short": "unsigned short",
        }
        return table.get(joined, words[-1] if len(words) == 1 else joined)

    def _star_quals(self) -> tuple:
        """The qualifiers after a `*` (C11 6.7.6.1p1), consumed: `const` and `restrict`, which a function type
        spells and compares (CF-QUALS). A `volatile` one makes the pointer object volatile, every access of which
        C performs as written (6.7.3p7) and no lowering here does -- refused, as the twin refuses it."""
        quals = set()
        while self.at("IDENT") and self.peek().text in _PTR_QUAL:
            q = _PTR_QUAL[self.peek().text]
            if q == "volatile":
                raise CParseError(VOLATILE_PTR, pos=self.peek().pos)
            quals.add(q)
            self.nxt()
        return tuple(sorted(quals))

    def _stars(self) -> tuple:
        """The `*`s at the cursor and each one's qualifiers, consumed: their count and their `ptr_quals` (one
        tuple per `*`, from the base out; () when none is qualified)."""
        levels, pos = [], self.peek().pos
        while self.at("OP", "*"):
            self.nxt()
            levels.append(self._star_quals())
        return len(levels), (_qual_levels(tuple(levels), pos) if any(levels) else ())

    def _skip_stars(self, k: int) -> int:
        """Past the `*`s at `k` tokens from the cursor and their qualifiers (a lookahead; nothing consumed)."""
        while self.peek(k).kind == "OP" and self.peek(k).text == "*":
            k += 1
            while self.peek(k).kind == "IDENT" and self.peek(k).text in _PTR_QUAL:
                k += 1
        return k

    def _declarator(self, base: cast.TypeRef, abstract: bool = False):
        """Parse `*` pointer prefixes, the name, and `[N]` array suffixes onto `base`. `abstract`: a
        parameter's, whose name may be left out (the name is then ""). Each `*` keeps its qualifiers
        (`ptr_quals`, CF-QUALS); `restrict` is an aliasing hint no layout reads."""
        ptr, pq = self._stars()
        # pointer-to-array declarator `(*name)[N]...` -- a "row pointer" (what `T m[][N]` decays to);
        # modeled as the equivalent multi-dim array param (outer dim unspecified) so `m[i][j]` flattens
        # row-major exactly as for `T m[A][N]`. The remaining vendor-header declarator form.
        if ptr == 0 and self.at("PUNCT", "("):
            save = self.i
            self.nxt()  # (
            # the row pointer's own qualifiers: a top-level `const` or `restrict` of the parameter
            inner, _quals = self._stars()
            if (
                inner == 1
                and self.at("IDENT")
                and self.peek(1).kind == "PUNCT"
                and self.peek(1).text == ")"
            ):
                nm = self.nxt().text  # the name
                self.nxt()  # )
                if self.at("PUNCT", "["):
                    dims = []
                    while self.at("PUNCT", "["):
                        self.nxt()
                        dims.append(0 if self.at("PUNCT", "]") else self._const_dim())
                        self.eat("PUNCT", "]")
                    return cast.TypeRef(
                        base=base.base,
                        ptr=0,
                        array=tuple([0] + dims),
                        aggregate=base.aggregate,
                        quals=base.quals,
                        typeof_var=base.typeof_var,
                        typeof_expr=base.typeof_expr,
                        bit_width=base.bit_width,
                    ), nm
            self.i = save  # not `(*name)[..]` -> a normal declarator
        name = self.eat("IDENT").text if not abstract or self.at("IDENT") else ""
        dims = []
        vla = None
        vla_dims: tuple = ()
        raw: list = []  # per-dim: an int (literal) or an expression (runtime)
        while self.at("PUNCT", "["):
            self.nxt()
            if self.at("PUNCT", "]"):
                raw.append(0)  # an incomplete `[]` dimension
            else:
                raw.append(self._dim(self._assign()))  # a constant: a static dim, else a runtime
            self.eat("PUNCT", "]")
        if any(not isinstance(d, int) for d in raw):  # at least one RUNTIME dim -> a VLA
            if len(raw) > 3:
                raise CParseError(
                    "a variable-length array of more than 3 dimensions is not supported"
                )
            if len(raw) == 1:
                vla = raw[0]  # a 1-D VLA -- the existing representation (unchanged path)
                dims = [0]
            else:
                vla_dims = tuple(raw)  # a MULTI-dim VLA `T a[m][n]` -- per-dim (literal | expr)
                dims = [0] * len(raw)
        else:
            dims = raw  # all-literal -> a static (possibly multi-dim) array
        if (
            base.funcptr
        ):  # `binop_fn fn` -- the funcptr shape, and the pointers to it and the arrays of it
            if vla is not None or vla_dims:  # (`op_t *p`, `op_t ops[3]`) keep it too (CF-FPTAB)
                raise CParseError(
                    "a variable-length array of function pointers is not supported",
                    pos=self.peek().pos,
                )
            if ptr == 0 and not dims:
                return base, name
            shaped = dataclasses.replace(
                base,
                ptr=base.ptr + ptr,
                array=tuple(dims) + tuple(base.array),
                ptr_quals=_join_levels(base, ptr, pq),
            )
            return shaped, name
        if ptr and base.array:  # `row_t *p` of `typedef T row_t[N]`: a pointer to an array has no
            raise CParseError(  # TypeRef spelling (it would read as an array of pointers)
                "a pointer to a typedef'd array is not supported", pos=self.peek().pos
            )
        return cast.TypeRef(
            base=base.base,
            ptr=ptr + base.ptr,  # base.ptr != 0 only for typeof(T*)
            # the declarator's own dims are the outer ones: `row_t rw[2]` is two `row_t`s
            array=tuple(dims) + tuple(base.array),
            vla=vla,
            vla_dims=vla_dims,
            aggregate=base.aggregate,
            quals=base.quals,
            typeof_var=base.typeof_var,
            typeof_expr=base.typeof_expr,
            bit_width=base.bit_width,
            ptr_quals=_join_levels(base, ptr, pq),
        ), name

    # --- functions ---
    def _func_body(
        self, ret: cast.TypeRef, name: str, reproducible: bool = False, static: bool = False
    ) -> cast.Func:
        self.eat("PUNCT", "(")
        params = []
        variadic = False
        # the list's own scope: a parameter hides a typedef of its name from the end of its declarator, so no later
        # parameter's type is read through it -- `f(uint32_t T, T x)` (C11 6.2.1p4, p7; the twin binds each as read)
        listed: dict = {}
        self.scopes.append(listed)
        try:
            if not (self.at("PUNCT", ")") or self.at("IDENT", "void") and self.peek(1).text == ")"):
                while True:
                    if self.at("PUNCT", "..."):  # a trailing `...` -- the function is variadic
                        self.nxt()
                        variadic = True
                        break
                    ptype = self._type_spec()
                    # a prototype may leave a parameter unnamed (`T g(uint32_t *, uint32_t);`); a definition
                    # names every one it binds, which the check below the list asks
                    ptype, pname = self._declarator_or_funcptr(ptype, abstract=True)
                    params.append(cast.Param(ptype, pname))
                    if pname:
                        listed[pname] = None
                    if self.at("PUNCT", ","):
                        self.nxt()
                        continue
                    break
            elif self.at("IDENT", "void"):
                self.nxt()
        finally:
            self.scopes.pop()
        self.eat("PUNCT", ")")
        if self.at("PUNCT", ";"):  # a PROTOTYPE (`T name(params);`): record the
            self.nxt()  # signature -- a cross-TU callee (Phase 3 linking)
            self.unit.protos[name] = (
                ret,
                tuple(p.type for p in params),
            )  # or an in-unit forward decl
            # `T name(P, ...);`: its `extern` declaration keeps the `...` (CF-EXTDESIG)
            if variadic:
                self.unit.variadic_protos.add(name)
            self._declared.add(name)
            return None
        if any(not p.name for p in params):
            raise CParseError(
                f"a parameter of the definition of {name!r} has no name", pos=self.peek().pos
            )
        # the function is in scope from its declarator on, so its own body may name it (C11 6.2.1p7)
        self._declared.add(name)
        declared = frozenset(self._declared)
        # the parameters' scope encloses the body's block
        self.scopes.append(dict.fromkeys(p.name for p in params))
        try:
            body = self._block()
        finally:
            self.scopes.pop()
        return cast.Func(
            ret=ret,
            name=name,
            params=tuple(params),
            body=body,
            variadic=variadic,
            reproducible=reproducible,
            static_fn=static,
            declared=declared,
        )

    # --- statements ---
    def _block(self) -> tuple:
        with self._descend():  # depth guard: _block<->_stmt nesting (`{{{...}}}`) cycle
            self.eat("PUNCT", "{")
            stmts = []
            self.scopes.append({})  # a block is a scope: the names it declares end with it,
            tags = dict(self.enum_tags)  # its enumeration tags too (CF-CONSTEXPR2)
            try:
                while not self.at("PUNCT", "}"):
                    if not self.recover:
                        stmts.append(self._stmt())
                        continue
                    try:  # statement-level recovery: a bad
                        stmts.append(self._stmt())  # statement doesn't abandon the block
                    except CParseError as e:
                        self._record(e)
                        if not self._sync_stmt():  # hit the block's `}` / EOF -> stop
                            break
            finally:
                self.scopes.pop()
                self.enum_tags = tags
            self.eat("PUNCT", "}")
            return tuple(stmts)

    def _enumerator(self, w: str):
        """The value of the enumerator `w` names where it is read, or None: the innermost declaration of `w` is
        the one it denotes (C11 6.2.1p4) -- a name the function declares in a block scope enclosing this point (a
        local, a parameter, a loop's own declaration) hides an enumerator of an enclosing scope, which is then no
        constant (CF-ENUMSCOPE), and an enumerator a block declares hides an outer object or enumerator to the end
        of the block (CF-CONSTEXPR2; the twin's `visible_enum`). Reading the enumerator there had folded it in
        place of the object."""
        for scope in reversed(self.scopes):
            if w in scope:
                return scope[w]
        return self.enums.get(w)

    def _typedef_visible(self, w: str) -> bool:
        """Whether `w` names a typedef where it is read: a name the function declares in a block scope enclosing this
        point -- a local, a parameter, a loop's own declaration, an enumerator a block declares -- hides a file-scope
        typedef of its name from the end of its declarator to the end of its block (C11 6.2.1p4, p7), so it starts
        no declaration there, no cast and no `sizeof` type-name (CF-TYPEDEFSCOPE; the twin's `visible_typedef`).
        Reading the typedef there had refused `uint32_t T = s; T = T * 3u;`, read `T * x;` as a declaration and
        `(T) - s` as a cast, and taken `sizeof(T)` for the type's size: the last two silently."""
        return w in self.typedefs and not any(w in scope for scope in self.scopes)

    def _is_decl_start(self) -> bool:
        """A declaration starts with a type: a keyword, a scalar or typedef name. A struct tag alone is
        not one -- tags have their own name space (C11 6.2.3), so a local `s` beside a `struct s` is an
        expression statement, never a declaration."""
        if not self.at("IDENT"):
            return False
        w = self.peek().text
        return (
            w in _TYPE_KW
            or w in _TYPEOF_KW
            or w == "enum"
            or is_scalar_name(w)
            or w in ("va_list", "__builtin_va_list", "_Atomic")
            or self._typedef_visible(w)
        )

    def _stmt(self):
        if self.at("PUNCT", ";"):  # empty statement -> a no-op (e.g. the
            self.nxt()  # body of `for(...);` / `while(...);`, `if(c);`)
            return cast.Seq(())
        if self.at("PUNCT", "{"):
            return cast.Block(self._block())  # bare `{ ... }` -> an inline-lowered scope
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
            if self.at("OP", "*"):  # `goto *expr;` -- a computed (indirect) goto (GNU)
                self.nxt()
                target = self._expr()
                self.eat("PUNCT", ";")
                return cast.ComputedGoto(target)
            label = self.eat("IDENT").text
            self.eat("PUNCT", ";")
            return cast.Goto(label)
        if self._at_asm_stmt():
            return self._asm()  # GNU inline assembly (ASM1 trusted opaque edge)
        if self.at("IDENT") and self.peek(1).kind == "PUNCT" and self.peek(1).text == ":":
            name = self.nxt().text  # `label:` — the labeled stmt follows
            self.eat("PUNCT", ":")
            return cast.Label(name)
        if self.at("IDENT", "static") or self._is_decl_start():
            return self._decl_stmt()
        expr = self._incdec() or self._expr()  # i++ / ++i / i-- / --i, else an expression
        self.eat("PUNCT", ";")
        return cast.ExprStmt(expr)

    def _for(self) -> cast.For:
        """`for (init ; cond ; step) body` — desugars onto the while machinery in lowering:
        `init; while(cond){ body; step }` (no `break`/`continue` yet, so this is exact)."""
        self.eat("IDENT", "for")
        self.eat("PUNCT", "(")
        # the loop's own declaration is in scope to the end of its body (6.8.5p5)
        self.scopes.append({})
        try:
            if self.at("PUNCT", ";"):  # empty init
                init = None
                self.nxt()
            elif self._is_decl_start():
                init = self._decl_stmt()  # a declaration (consumes its `;`)
            else:
                init = cast.ExprStmt(self._incdec() or self._expr())
                self.eat("PUNCT", ";")
            cond = cast.IntLit(1) if self.at("PUNCT", ";") else self._expr()
            self.eat("PUNCT", ";")
            step = None if self.at("PUNCT", ")") else self._for_step()
            self.eat("PUNCT", ")")
            body = self._block() if self.at("PUNCT", "{") else (self._stmt(),)
        finally:
            self.scopes.pop()
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
        assignment `i = i ± 1`. Returns the Assign, or None (consuming nothing) if not an inc/dec -- or if the
        operand is more than a name (`++h.next->v`, `++p[i]`, `++*p`), which the expression grammar parses as
        an increment of that lvalue (CF-SPLIT2: taking `++h` here refused the rest of the statement)."""
        if (self.at("OP", "++") or self.at("OP", "--")) and not (
            self.peek(1).kind == "IDENT"
            and not (
                (self.peek(2).kind == "PUNCT" and self.peek(2).text in (".", "[", "("))
                or (self.peek(2).kind == "OP" and self.peek(2).text in ("->", "++", "--"))
            )
        ):
            return None
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

    def _at_asm_stmt(self) -> bool:
        """Whether the cursor begins a GNU inline-asm STATEMENT. `asm` is NOT a keyword (it stays a usable
        ISO-C identifier under -std=c11), so it is recognized CONTEXTUALLY: the `asm`/`__asm__` token must be
        FOLLOWED BY `(` (basic/extended asm) or a `volatile`/`__volatile__`/`goto`/`inline` qualifier
        (extended asm). Otherwise `asm` is an ordinary identifier (`int asm = 3; asm = 4; return asm + 1;`)
        and this falls through to the normal declaration/expression-statement parsing. (`__asm__` IS a reserved
        keyword, but is recognized the same way so a stray `__asm__;` still degrades gracefully.)"""
        if not (self.at("IDENT", "asm") or self.at("IDENT", "__asm__")):
            return False
        nxt = self.peek(1)
        return (nxt.kind == "PUNCT" and nxt.text == "(") or (
            nxt.kind == "IDENT" and nxt.text in ("volatile", "__volatile__", "goto", "inline")
        )

    def _asm_template(self) -> str:
        """The asm TEMPLATE string: a string literal with adjacent-literal concatenation (`"a\\n" "b"`),
        returned as the SOURCE spelling (quotes intact) so it re-emits verbatim. Concatenation is legitimate
        for the template (unlike a constraint/clobber, which is exactly one token -- see _asm_single_string)."""
        if not self.at("STRING"):
            tk = self.peek()
            raise CParseError(
                f"expected a string literal in asm, got {tk.kind} {tk.text!r}", pos=tk.pos
            )
        text = self.nxt().text
        while self.at("STRING"):  # adjacent literals concatenate (kept adjacent,
            text += " " + self.nxt().text  # like the expression-level string concatenation)
        return text

    def _asm_single_string(self, what: str) -> str:
        """Exactly ONE string-literal token (a constraint or a clobber) -- NO adjacent-literal concatenation,
        so a missing comma (`"cc" "memory"`) is a clean CParseError, not a silently-merged operand. Returned
        as the SOURCE spelling (quotes intact)."""
        if not self.at("STRING"):
            tk = self.peek()
            raise CParseError(
                f"expected a {what} string literal in asm, got {tk.kind} {tk.text!r}", pos=tk.pos
            )
        return self.nxt().text

    def _asm_operand(self):
        """One extended-asm operand: `[ [ symbolic-name ] ] "constraint" ( expr )`. The optional bracketed
        symbolic name (`[name]`) precedes the constraint (exactly one string token); the operand is a
        parenthesized expression (reused for both outputs and inputs -- the output-lvalue restriction is
        enforced in lowering with an honest diagnostic). Returns (symbolic_name | None, constraint_str, expr)."""
        name = None
        if self.at("PUNCT", "["):  # an optional `[symbolic-name]`
            self.nxt()
            name = self.eat("IDENT").text
            self.eat("PUNCT", "]")
        constraint = self._asm_single_string("constraint")
        self.eat("PUNCT", "(")
        expr = self._expr()  # reuse the expression/lvalue parser
        self.eat("PUNCT", ")")
        return (name, constraint, expr)

    def _asm_operands(self) -> tuple:
        """A (possibly empty) comma-separated operand list, terminated by `:` or `)`. An empty section
        (`: :`) is valid."""
        if self.at("PUNCT", ":") or self.at("PUNCT", ")"):
            return ()
        ops = [self._asm_operand()]
        while self.at("PUNCT", ","):
            self.nxt()
            ops.append(self._asm_operand())
        return tuple(ops)

    def _asm_clobbers(self) -> tuple:
        """A (possibly empty) comma-separated clobber list of string literals (`"memory"`, `"cc"`, …), each
        exactly ONE token (a missing comma is a clean error), terminated by `:` or `)`."""
        if self.at("PUNCT", ":") or self.at("PUNCT", ")"):
            return ()
        clob = [self._asm_single_string("clobber")]
        while self.at("PUNCT", ","):
            self.nxt()
            clob.append(self._asm_single_string("clobber"))
        return tuple(clob)

    def _asm_labels(self) -> tuple:
        """The `asm goto` label list — a comma-separated list of label identifiers, terminated by `)`."""
        if self.at("PUNCT", ")"):
            return ()
        labels = [self.eat("IDENT").text]
        while self.at("PUNCT", ","):
            self.nxt()
            labels.append(self.eat("IDENT").text)
        return tuple(labels)

    def _asm(self) -> cast.AsmStmt:
        """GNU inline assembly as a statement (ASM1 — a trusted opaque effect edge):
          basic:    `asm ( "template" ) ;`
          extended: `asm [volatile|__volatile__] ( "template" [: outputs [: inputs [: clobbers [: labels]]]] ) ;`
        Both `asm` and `__asm__` introduce it. The TEMPLATE + constraints + clobbers are kept as source
        spellings so they re-emit verbatim (ISA-neutral pass-through); the operands reuse the normal
        expression/lvalue parsers. A malformed asm raises a clean CParseError rather than crashing."""
        self.nxt()  # the `asm` / `__asm__` keyword
        is_volatile = False
        is_goto = False
        while (
            self.at("IDENT", "volatile")
            or self.at("IDENT", "__volatile__")
            or self.at("IDENT", "inline")
            or self.at("IDENT", "goto")
        ):
            qual = self.nxt().text  # asm-qualifiers: volatile / inline / goto (any order)
            if qual in ("volatile", "__volatile__"):
                is_volatile = True
            elif qual == "goto":
                is_goto = True  # `asm goto` -- a label list (parsed; rejected below)
            # `inline` is accepted + ignored (a size hint, value-neutral)
        self.eat("PUNCT", "(")
        template = self._asm_template()
        outputs: tuple = ()
        inputs: tuple = ()
        clobbers: tuple = ()
        labels: tuple = ()
        is_basic = not self.at("PUNCT", ":")  # BASIC == no colon sections at all (so the emit can
        #   re-render the basic form ONLY for a true basic asm,
        #   not an all-empty EXTENDED asm -- else a non-volatile
        #   `asm("x" : :)` would round-trip to volatile/barriered)
        if self.at("PUNCT", ":"):  # extended form: at least the outputs section
            self.nxt()
            outputs = self._asm_operands()
            if self.at("PUNCT", ":"):
                self.nxt()
                inputs = self._asm_operands()
                if self.at("PUNCT", ":"):
                    self.nxt()
                    clobbers = self._asm_clobbers()
                    if is_goto and self.at(
                        "PUNCT", ":"
                    ):  # the 5th section: ONLY `asm goto` has labels -- a
                        self.nxt()  #   non-goto trailing `: labels` falls through to the
                        labels = (
                            self._asm_labels()
                        )  #   `)` and raises a clean syntax error (not a misleading
                        #   "asm goto" downstream message)
        else:
            is_volatile = True  # a BASIC asm is implicitly volatile (§6.47.1)
        self.eat("PUNCT", ")")
        self.eat("PUNCT", ";")
        return cast.AsmStmt(
            template=template,
            outputs=outputs,
            inputs=inputs,
            clobbers=clobbers,
            is_volatile=is_volatile,
            goto_labels=labels,
            is_goto=is_goto,
            is_basic=is_basic,
        )

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
                body.append(cast.Case(self._const_eval(self._expr())))  # case <const>:
                self.eat("PUNCT", ":")
            elif self.at("IDENT", "default"):
                self.nxt()
                self.eat("PUNCT", ":")
                body.append(cast.Default())
            else:
                body.append(self._stmt())  # a statement (incl. `break;` -> cast.Break)
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
        self.storage = set()  # this declaration's own storage classes, wherever they are spelled
        start = self.i
        base = self._type_spec()
        # `enum [tag] { ... };`: a declaration of the enumeration's constants (and tag) alone, in scope to the end of
        # the block (C11 6.7p2; CF-CONSTEXPR2)
        spec = {t.text for t in self.t[start : self.i]}
        if self.at("PUNCT", ";") and "enum" in spec and not spec & {"struct", "union"}:
            self.nxt()
            return cast.Seq(())
        # a block-scope `extern` names an object defined elsewhere, never a new local: binding it as
        # one read an uninitialized object (CF-STORAGE)
        if "extern" in self.storage:
            raise CParseError(
                "a block-scope extern declaration is not supported", pos=self.peek().pos
            )
        # `static` in any position: `volatile static T n` and `T static n` are static (6.7p1)
        is_static = "static" in self.storage
        # thread storage is kept (CF-TLS): each thread has its own object. At block scope it needs `static` too
        # (C11 6.7.1p3; a block-scope `extern` is refused above)
        is_thread = bool(self.storage & _THREAD_SPEC)
        if is_thread and not is_static:
            raise CParseError(
                "a block-scope `_Thread_local` object that is not `static`", pos=self.peek().pos
            )
        decls = []
        while True:
            tref, name = self._declarator_or_funcptr(base)
            # in scope from the end of its declarator, its own initializer included (6.2.1p7)
            if self.scopes:
                self.scopes[-1][name] = None
            init = None
            if self.at("OP", "="):
                self.nxt()
                init = self._init_value()
            decls.append(
                cast.Decl(tref, name, init, static_storage=is_static, thread_storage=is_thread)
            )
            if self.at("PUNCT", ","):  # another declarator off the same specifier
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
        with (
            self._descend()
        ):  # depth guard: nested braced initializers (`{{{...}}}`) self-recurse here
            return self._init_value_braced()

    def _init_value_braced(self):
        self.nxt()
        entries = []
        while not self.at("PUNCT", "}"):
            key = None
            if self.at("PUNCT", "[") or self.at(
                "PUNCT", "."
            ):  # a designator (possibly a nested chain)
                steps = []  # .a.b / .v[i] / [i][j] -> a step list
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
        with (
            self._descend()
        ):  # depth guard: _comma is the parenthesized-expression re-entry (`(...)`)
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
        if self.at("PUNCT", "?"):  # cond ? then : els  (right-associative)
            self.nxt()
            then = self._assign()  # the middle is a full expression
            self.eat("PUNCT", ":")
            els = self._assign()  # the else nests another conditional
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
        if self.at("IDENT", "__real__") or self.at(
            "IDENT", "__imag__"
        ):  # GNU complex part extraction
            op = self.nxt().text
            return cast.Unary(op, self._unary())
        if self.at("OP", "&&"):  # `&&label` -- a label's address as a value (GNU)
            self.nxt()
            return cast.LabelAddr(self.eat("IDENT").text)
        if self.peek().kind == "OP" and self.peek().text in (
            "++",
            "--",
        ):  # PREFIX ++a / --a -> yields the NEW value
            op = self.nxt().text[0]
            return cast.IncDec(op, self._unary(), prefix=True)
        # `+` included: it promotes its operand (C11 6.5.3.3p2), which the parser had dropped -- so `sizeof(+c)`
        # was `sizeof(c)` and `_Generic(+c, ...)` chose by `c`'s type, on both rails (CF-UNARY)
        if self.peek().kind == "OP" and self.peek().text in ("+", "-", "~", "!", "*", "&"):
            op = self.nxt().text
            return cast.Unary(op, self._unary())
        if self._is_cast():  # (type)operand — a cast binds at the unary level
            self.eat("PUNCT", "(")
            tref = self._fp_type_name(self._type_spec())
            ptr, pq = (
                self._stars()
            )  # `(uint32_t *)p` — a pointer cast; `(char *const *)p` (CF-QUALS)
            dims = []
            while self.at("PUNCT", "["):  # `(int[N]){...}` / `(int[]){...}` — an array type-name
                self.nxt()
                dims.append(0 if self.at("PUNCT", "]") else self._const_dim())
                self.eat("PUNCT", "]")
            self.eat("PUNCT", ")")
            if ptr and tref.array:  # a pointer to a typedef'd array: no TypeRef spelling
                raise CParseError(
                    "a pointer to a typedef'd array is not supported", pos=self.peek().pos
                )
            # the type-name keeps a typedef's pointer and array shape (`(ip)v` of `typedef T *ip`, a
            # `(row_t){...}` literal of `typedef T row_t[N]`), which rebuilding it from its base dropped -- and a
            # function pointer's return and parameters (`(op_t)f`, CF-RTFP), which rebuilding it dropped too
            tref = dataclasses.replace(
                tref,
                ptr=tref.ptr + ptr,
                array=tuple(dims) + tuple(tref.array),
                ptr_quals=_join_levels(tref, ptr, pq),
            )
            literal = self.at("PUNCT", "{")
            # a cast names no array type (C11 6.5.4p2), and no rail models an `_Atomic` object a compound literal
            # would make (CF-RTFP)
            if tref.array and not literal:
                raise CParseError(CAST_ARRAY, pos=self.peek().pos)
            if literal and "_Atomic" in tref.quals and not tref.ptr:
                raise CParseError(ATOMIC_LITERAL, pos=self.peek().pos)
            if literal:  # `(type){ init }` — a C99 compound literal, not a cast
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
        return (
            w in _TYPE_KW
            or w in _TYPEOF_KW
            or w
            in (
                "struct",
                "union",
                "enum",
                "const",
                "volatile",
                "_Atomic",
            )  # `(_Atomic T *)p` (CF-RTFP)
            or is_scalar_name(w)
            or self._typedef_visible(w)
        )

    def _postfix(self):
        return self._postfix_tail(self._primary())

    def _postfix_tail(self, node):
        """Apply the postfix operators (`[i]`, `.f`, `->f`, `(args)`) to an already-parsed base — shared
        by `_postfix` (after a primary) and `_unary` (a compound literal, so `(struct P){...}.f` works).
        A parenthesized dereference folds into the chain (CF-PAREN): `(*X)[i]` is `X[0][i]` (C11 6.5.2.1p2)
        and `(*X).f` is `X->f` (6.5.2.3p4) -- the row a row pointer addresses is indexed where it lies (not
        loaded as a scalar), a struct member through the pointer (not a load of the whole struct) -- the
        spelling the twin rewrites the tokens to before it parses."""
        while True:
            if self.at("PUNCT", "["):
                self.nxt()
                idx = self._expr()
                self.eat("PUNCT", "]")
                if isinstance(node, cast.Unary) and node.op == "*":
                    node = cast.Index(node.operand, cast.IntLit(0))
                node = cast.Index(node, idx)
            elif self.at("PUNCT", "."):
                self.nxt()
                field = self.eat("IDENT").text
                if isinstance(node, cast.Unary) and node.op == "*":
                    node = cast.Member(node.operand, field, arrow=True)
                else:
                    node = cast.Member(node, field, arrow=False)
            elif self.at("OP", "->"):
                self.nxt()
                node = cast.Member(node, self.eat("IDENT").text, arrow=True)
            elif (
                self.at("PUNCT", "(") and isinstance(node, cast.Name) and node.ident == "va_arg"
            ):  # va_arg(ap, T) -- 2nd operand is a type-name
                self.nxt()
                ap = self._expr()
                self.eat("PUNCT", ",")
                tref = self._type_spec()
                ptr, pq = self._stars()
                tref = cast.TypeRef(
                    base=tref.base,
                    ptr=ptr,
                    aggregate=tref.aggregate,
                    quals=tref.quals,
                    ptr_quals=pq,
                )
                self.eat("PUNCT", ")")
                node = cast.VaArg(ap, tref)
            elif self.at("PUNCT", "("):
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
                if isinstance(node, cast.Name):
                    node = cast.CallExpr(node.ident, tuple(args))
                elif isinstance(node, cast.Member):  # o->fnptr(args): dispatch table
                    node = cast.CallMember(node, tuple(args))
                else:  # `(*fp)(x)`, `ops[i](x)`: a call through the expression's value (CF-FPTAB)
                    node = cast.CallPtr(node, tuple(args))
            elif self.peek().kind == "OP" and self.peek().text in (
                "++",
                "--",
            ):  # POSTFIX a++ / a-- -> OLD value
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
            if self._is_decl_start() and not self._literal_follows():  # sizeof ( type-name )
                # -- not `sizeof (T[N]){...}`, a compound literal's unary-expression form (CF-RTFP)
                tref = self._abstract_type_name()
                self.eat("PUNCT", ")")
                return cast.SizeOf(type=tref)
            self.i = save  # not a type -> `sizeof ( expr )`
        return cast.SizeOf(expr=self._unary())  # sizeof expr / sizeof (expr)

    def _literal_follows(self) -> bool:
        """Whether the parenthesized type name starting at the cursor (past its `(`) is a compound literal's: a `{`
        follows its `)` (CF-RTFP; the twin's `literal_close`). Nothing is consumed."""
        k, depth = 0, 0
        while self.peek(k).kind != "EOF":
            t = self.peek(k)
            if t.kind == "PUNCT" and t.text in ("(", "["):
                depth += 1
            elif t.kind == "PUNCT" and t.text in (")", "]"):
                if not depth:
                    break
                depth -= 1
            k += 1
        nxt = self.peek(k + 1)
        return self.peek(k).text == ")" and nxt.kind == "PUNCT" and nxt.text == "{"

    def _fp_type_name(self, tref: cast.TypeRef) -> cast.TypeRef:
        """The rest of a type name whose specifier is `tref`, where it is a function pointer -- `(R (*)(P))f`,
        `(R (**)(P))p`, `(T *(*)(P))f`, `sizeof(R (*)(P))`: its abstract declarator, read as a parameter's is
        (CF-RTFP; the twin's `p_cast_type`) -- else `tref` as it was. A type name declares no identifier (C11
        6.7.7p1)."""
        if not self._is_funcptr_declarator(True, at=self._skip_stars(0)):
            return tref
        tref, name = self._declarator_or_funcptr(tref, abstract=True)
        if name:
            raise CParseError(TYPE_NAME_NAMED, pos=self.peek().pos)
        return tref

    def _abstract_type_name(self) -> cast.TypeRef:
        """A type-name as `sizeof ( ... )` and `_Alignof ( ... )` take it (C11 6.7.7): the specifier, then
        any `*`s, then any array dimensions `[N]` (integer constant expressions). The specifier's own
        TypeRef is kept whole -- a typedef's array dimensions, a `_BitInt(N)` width, a `typeof` operand --
        where rebuilding it from its base and qualifiers alone had dropped them (`sizeof(row_t)` of
        `typedef uint16_t row_t[6]` was 2, not 12). A pointer to an array type (`T (*)[N]`, or `*` after
        an array typedef) has no TypeRef spelling and is refused (CF-SIZEOF)."""
        tref = self._fp_type_name(self._type_spec())  # `sizeof(uint32_t (*)(uint32_t))` (CF-RTFP)
        ptr, pq = self._stars()  # `sizeof(uint32_t *)` etc.
        dims = []
        while self.at("PUNCT", "["):  # `sizeof(uint32_t[10])`: an array of what precedes
            self.nxt()
            dims.append(self._const_eval(self._assign()))
            self.eat("PUNCT", "]")
        if ptr and tref.array:
            raise CParseError(
                "a pointer to an array type-name is not supported", pos=self.peek().pos
            )
        return dataclasses.replace(
            tref,
            ptr=tref.ptr + ptr,
            array=tuple(dims) + tuple(tref.array),
            ptr_quals=_join_levels(tref, ptr, pq),
        )

    def _alignof(self):
        """`_Alignof ( type-name )` / `alignof(...)` -> a constant: the type's alignment (folded in
        lowering from the shared layout model; unlike sizeof, only the type-name form is valid C)."""
        self.nxt()  # _Alignof / alignof
        self.eat("PUNCT", "(")
        tref = self._abstract_type_name()
        self.eat("PUNCT", ")")
        return cast.AlignOf(tref)

    def _generic(self):
        """`_Generic ( assignment-expr , (type-name : assignment-expr | default : assignment-expr)+ )`
        (C11 §6.5.1.1) -- a type-name label / `default` per association; lowering selects on the
        controlling expression's static type and evaluates only the chosen association's expression."""
        self.nxt()  # _Generic
        self.eat("PUNCT", "(")
        controlling = self._expr()  # the controlling expression (unevaluated)
        self.eat("PUNCT", ",")
        assocs = []
        while True:
            if self.at("IDENT", "default"):
                self.nxt()
                tref = None
            else:
                at = self.peek().pos
                tref = self._type_spec()
                ptr, pq = self._stars()  # a pointer type-name label, e.g. `int *`
                # a qualified one -- `const char *`, `char *const *`, `const int` -- neither rail tells from
                # the unqualified one, the controlling expression's type carrying no qualifier (CF-QUALS)
                if tref.quals or pq or tref.ptr_quals:
                    raise CParseError(GENERIC_QUALIFIED, pos=at)
                tref = cast.TypeRef(
                    base=tref.base, ptr=ptr, aggregate=tref.aggregate, quals=tref.quals
                )
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
            tk = self.nxt()  # type from the suffix + magnitude (§6.4.4.1), with the target's `long`
            return cast.IntLit(
                parse_int_literal(tk.text, tk.pos), int_literal_type(tk.text, self.abi.long_size)
            )
        if self.at("CHAR"):  # a character constant: its value, typed by its prefix (CF-CONSTEXPR2)
            return cast.IntLit(*char_constant(self.nxt().text, self.abi))
        if self.at("FLOAT"):  # a floating-point literal (1.5 / 3.14f)
            return cast.FloatLit(self.nxt().text)
        if self.at("STRING"):
            text = self.nxt().text
            while self.at("STRING"):  # adjacent literals concatenate (phase 6),
                text += " " + self.nxt().text  # kept adjacent so escapes don't merge
            return cast.StringLit(text)
        if self.at("IDENT"):
            w = self.peek().text
            k = self._enumerator(w)
            if k is not None:  # an enumerator in scope -> its integer literal
                self.nxt()
                return cast.IntLit(k)
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
            if self.peek(1).text == "{":  # `({ ... })` -- a GCC statement expression
                self.nxt()
                stmts = self._block()
                self.eat("PUNCT", ")")
                return cast.StmtExpr(stmts)
            self.nxt()
            e = self._comma()  # a full expression: the comma OPERATOR is allowed here
            self.eat("PUNCT", ")")
            return e
        tk = self.peek()
        raise CParseError(f"unexpected {tk.kind} {tk.text!r}", pos=tk.pos)


def parse_unit(src: str, abi=None) -> cast.Unit:
    """Parse one C translation unit (the L1–L4 subset) into the `cast` AST, raising on the first
    error (the compile path needs a well-formed AST). `abi`: the target the unit is lowered for (the
    host's by default) -- its integer constants and the constant expressions folded at parse take its
    data model."""
    toks = tokenize(src)
    return _Parser(toks, set(), abi).parse_unit()


def parse_with_recovery(src: str, abi=None) -> tuple[cast.Unit, list]:
    """Parse with panic-mode recovery, returning the (partial) AST and *every* parse diagnostic the
    run found -- the diagnostics entry uses this so one invocation reports several errors. A CLexError
    still propagates (lexer recovery is a separate concern)."""
    p = _Parser(tokenize(src), set(), abi)
    p.recover = True
    unit = p.parse_unit()
    return unit, p.diags
