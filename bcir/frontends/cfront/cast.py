"""C AST nodes for the frontend subset (L1–L4). Small, frozen dataclasses — a faithful but minimal
abstract syntax for fixed-width integer expressions, struct/union types, pointers/arrays, and
function definitions with calls. Control flow (L6) is parsed into `If`/`While` nodes already so the
grammar is stable, but the L1–L4 lowering only needs declarations, assignments, returns, and calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- type references (resolved against the type table during lowering) ---
@dataclass(frozen=True)
class TypeRef:
    base: str  # scalar name or struct/union tag (a funcptr alias: its spelling)
    ptr: int = 0  # pointer depth
    array: tuple = ()  # array dimensions (outer-first)
    aggregate: str = ""  # "struct" | "union" | "" (scalar)
    quals: tuple = ()  # ("const",) / ("volatile",) — drives MMIO lowering (L5)
    funcptr: bool = False  # a function-pointer alias (RET (*name)(PARAMS)) — base is its name
    func_ret: object = None  # the return TypeRef (funcptr only)
    func_params: tuple = ()  # the parameter TypeRefs (funcptr only) — for faithful emit
    typeof_var: str = ""  # `typeof(var)` — resolved in lowering to the in-scope variable's type
    typeof_expr: object = (
        None  # `typeof(expr)` — the unevaluated operand; lowering infers its static type
    )
    vla: object = None  # a 1-D variable-length-array dim `T a[n]` — the runtime size EXPRESSION
    #   (None == a normal/constant array); lowering snapshots it for masked bounds
    vla_dims: tuple = ()  # a MULTI-dim VLA `T a[m][n]` — per-dim (int literal | size expr), at least
    #   one runtime; lowering flattens to a runtime m*n extent + Horner strides
    bit_width: int = 0  # a C23 `_BitInt(N)` type's exact width N (0 == not a `_BitInt`); `base` is
    #   the verbatim spelling (`_BitInt(N)` / `unsigned _BitInt(N)`) lowering builds
    #   the `bitint` CType from. Carried on the TypeRef so the no-promotion +
    #   faithful-emit width survives parse -> lower.


# --- expressions ---
@dataclass(frozen=True)
class IntLit:
    value: int
    ctype: str = "int"  # the constant's C type name (§6.4.4.1): int / unsigned int / long / ...


@dataclass(frozen=True)
class AggInit:
    """A braced aggregate initializer `{ ... }` for a local struct/union/array. Each entry is
    (key, expr): key is None (positional), an int (array designator `[i]=`), or a str (member
    designator `.field=`). Uninitialized members zero-fill (§6.7.10)."""

    entries: tuple = ()


@dataclass(frozen=True)
class Switch:
    """A real C `switch`: the discriminant + a flat body sequence (Case | Default | statement), so
    case labels and cross-clause fallthrough are preserved exactly (not desugared to if/else)."""

    disc: object
    body: tuple = ()


@dataclass(frozen=True)
class Case:
    value: int  # the case label -- a folded integer constant expression (§6.4.4 / enum)


@dataclass(frozen=True)
class Default:
    pass


@dataclass(frozen=True)
class FloatLit:
    value: str  # the literal's source spelling, incl. any suffix (1.5 / 1e10 / 3.14f)


@dataclass(frozen=True)
class StringLit:
    value: str  # the literal's source spelling, including the surrounding quotes


@dataclass(frozen=True)
class Name:
    ident: str
    pos: int | None = None  # source byte offset of the identifier (for semantic-error carets)


@dataclass(frozen=True)
class Unary:
    op: str  # - ~ ! * & (deref / address-of)
    operand: object


@dataclass(frozen=True)
class IncDec:
    op: str  # '+' (++) or '-' (--) -- the direction
    operand: object  # the lvalue being stepped
    prefix: bool  # True: ++a/--a (yield the NEW value); False: a++/a-- (yield the OLD value)


@dataclass(frozen=True)
class Cast:
    type: TypeRef  # the target type — `(type)operand`
    operand: object


@dataclass(frozen=True)
class CompoundLiteral:
    """A C99 compound literal `( type-name ) { initializer-list }` — an anonymous object of `type`
    with automatic storage, materialized as a nameless local and yielded as an lvalue (so `&(int){x}`,
    `f((struct P){a,b})`, and `(struct P){...}.field` all work). `init` is the braced AggInit."""

    type: TypeRef
    init: AggInit


@dataclass(frozen=True)
class VaArg:
    """`va_arg(ap, T)` — pull the next variadic argument of type T from the cursor `ap`. The 2nd
    operand is a type-name (parsed specially, not as an expression); the result has type T."""

    ap: object
    type: TypeRef


@dataclass(frozen=True)
class Generic:
    """`_Generic(ctrl, T1: e1, ..., default: eN)` (C11 §6.5.1.1) — a compile-time selection on the static
    type of the (UNEVALUATED) controlling expression. `assocs` is a tuple of (TypeRef | None, expr); a
    None type-name is the `default`. Exactly the selected association's expression is evaluated."""

    controlling: object
    assocs: tuple


@dataclass(frozen=True)
class SizeOf:
    type: object = None  # a TypeRef for `sizeof(type)`, else None
    expr: object = None  # the operand for `sizeof expr` (its static type's size)


@dataclass(frozen=True)
class AlignOf:
    type: TypeRef  # `_Alignof(type-name)` — folds to the type's alignment


@dataclass(frozen=True)
class Binary:
    op: str  # + - * / % & | ^ << >> == != < > <= >= && ||
    lhs: object
    rhs: object


@dataclass(frozen=True)
class Ternary:
    cond: object  # `cond ? then : els` (conditional expression)
    then: object
    els: object


@dataclass(frozen=True)
class Assign:
    target: object  # an lvalue expr
    value: object


@dataclass(frozen=True)
class Index:
    base: object
    index: object


@dataclass(frozen=True)
class Member:
    base: object
    field: str
    arrow: bool  # True for `->`, False for `.`


@dataclass(frozen=True)
class CallExpr:
    callee: str
    args: tuple


@dataclass(frozen=True)
class CallMember:
    callee: object  # a Member node (`o->fn` / `o.fn`) — an indirect call via a
    args: tuple  # function-pointer struct member (HAL dispatch table)


# --- statements ---
@dataclass(frozen=True)
class Decl:
    type: TypeRef
    name: str
    init: object = None
    static_storage: bool = False  # `static T name = init;` — static storage duration (persists)


@dataclass(frozen=True)
class Return:
    value: object = None


@dataclass(frozen=True)
class ExprStmt:
    expr: object


@dataclass(frozen=True)
class Seq:
    stmts: tuple = ()  # a flat sequence of statements lowered in order -- the comma-
    # separated declarators of one declaration (`T a = x, b = y;`)


@dataclass(frozen=True)
class Block:
    stmts: tuple = ()  # a bare `{ ... }` compound statement: a new scope, lowered inline
    # (no `if(1)` wrapper), so its locals do not leak to the enclosing block


@dataclass(frozen=True)
class StmtExpr:
    """A GCC statement expression `({ s1; ...; expr; })` -- a compound statement, in its own scope, whose
    value is the last statement's (an expression statement). The prefix statements lower inline before the
    use site; `stmts` is the block's statement tuple (the last is the value-yielding ExprStmt)."""

    stmts: tuple = ()


@dataclass(frozen=True)
class If:
    cond: object
    then: tuple
    els: tuple = ()


@dataclass(frozen=True)
class While:
    cond: object
    body: tuple = ()


@dataclass(frozen=True)
class For:
    init: object  # a Decl / ExprStmt, or None
    cond: object  # the loop test (IntLit(1) if omitted)
    step: object  # an ExprStmt run at the end of each iteration, or None
    body: tuple = ()


@dataclass(frozen=True)
class DoWhile:
    cond: object  # `do body while (cond);` — body runs, then cond is tested
    body: tuple = ()


@dataclass(frozen=True)
class Break:
    pass  # `break;` — exit the nearest enclosing loop


@dataclass(frozen=True)
class Continue:
    pass  # `continue;` — jump to the nearest enclosing loop's next iteration


@dataclass(frozen=True)
class Goto:
    label: str  # `goto label;` — an unconditional jump (driver error-cleanup paths)


@dataclass(frozen=True)
class Label:
    name: str  # `name:` — a jump target at function-body scope


@dataclass(frozen=True)
class LabelAddr:
    label: str  # `&&L` — the address of a label as a value (GNU), a `void *`


@dataclass(frozen=True)
class ComputedGoto:
    target: object  # `goto *expr;` — an indirect jump to a taken label address (GNU)


@dataclass(frozen=True)
class AsmStmt:
    """GNU inline assembly as an ISA-neutral trusted opaque effect edge (ASM1). The assembly TEMPLATE is
    opaque/trusted and re-emitted VERBATIM; BCIR owns only the *calling side* — the operand binding, the
    constraints, the clobber declaration, and the ordering semantics. NOT interpreted, computes no verified
    value (off the legality value-path), exactly like the `c.call.libm.void:` external-effect family.
      * `template`     — the assembly string's source spelling (the surrounding quotes intact; adjacent
                         string-literal concatenation is joined here, like other string operands).
      * `outputs`      — list of (symbolic_name | None, constraint_str, lvalue_expr). A `"="` output is
                         write-only; a `"+"` output is read+write (its lvalue is also read).
      * `inputs`       — list of (symbolic_name | None, constraint_str, value_expr).
      * `clobbers`     — list of clobber-string source spellings (`"memory"`, `"cc"`, `"rax"`, …).
      * `is_volatile`  — True for `asm volatile`/`asm __volatile__`, AND implicitly for a *basic* asm (no
                         operand sections): a volatile/basic asm is a side-effecting edge that must NOT be
                         dead-code-eliminated, and a `"memory"` clobber / volatile asm is an ordering barrier.
      * `goto_labels`  — the `asm goto` label list (parsed for grammar completeness; rejected with an honest
                         diagnostic in lowering — `asm goto` control flow is deferred)."""

    template: str
    outputs: tuple = ()
    inputs: tuple = ()
    clobbers: tuple = ()
    is_volatile: bool = False
    goto_labels: tuple = ()
    is_goto: bool = False  # the `asm goto` qualifier was present (parsed for grammar completeness,
    #   rejected in lowering even if the label list is grammatically empty)
    is_basic: bool = (
        False  # the BASIC form `asm("template")` -- NO colon operand sections at all (so
    )
    #   the emit re-renders the basic form only here, never for an all-empty
    #   EXTENDED asm `asm("x" : :)`, which must round-trip as non-volatile)


# --- top level ---
@dataclass(frozen=True)
class Param:
    type: TypeRef
    name: str


@dataclass(frozen=True)
class Func:
    ret: TypeRef
    name: str
    params: tuple
    body: tuple
    variadic: bool = False  # a trailing `...` after the named params (variadic function)
    reproducible: bool = (
        False  # a C23 `[[reproducible]]`/`[[unsequenced]]` hint attribute is present
    )
    #   on the definition. The hint is value-neutral (no effect on computed
    #   results), so the emit DROPS it; recorded here only as a fusion-
    #   legality signal (default False == every existing function, untouched).
    static_fn: bool = False  # source `static` on the definition (internal linkage). The default
    #   emit is static regardless; the LINKABLE emit keeps `static` only
    #   when this is set (source-static honoring).


@dataclass(frozen=True)
class Aggregate:
    kind: str  # struct | union
    tag: str
    members: tuple  # (TypeRef, name, bit_width, member_align)  member_align 0 == natural
    packed: bool = False  # __attribute__((packed)) — no inter-member padding
    align: int = 0  # __attribute__((aligned(N))) / alignas(N) — forced aggregate alignment


@dataclass(frozen=True)
class Global:
    """A top-level (file-scope) variable, e.g. a `const uint8_t table[] = { ... }` (the seam C23
    `#embed` initializers land in once the preprocessor has expanded them to a byte list)."""

    type: TypeRef
    name: str
    init: tuple = ()  # initializer element expressions (for an array/scalar)
    extern_decl: bool = False  # `extern T g;` -- a DECLARATION of another TU's definition
    #   (the linkable emit prints `extern ...;`, never a definition)
    static_storage: bool = False  # source `static` (internal linkage) -- the linkable emit keeps
    #   the definition file-local instead of exporting it


@dataclass
class Unit:
    aggregates: dict = field(default_factory=dict)  # tag -> Aggregate
    funcs: list = field(default_factory=list)  # Func, in order
    globals: list = field(default_factory=list)  # Global, in order
    protos: dict = field(default_factory=dict)  # name -> (ret TypeRef, (param TypeRef, ...)) --
    #   file-scope function PROTOTYPES (Phase 3 linking:
    #   a cross-TU callee, or an in-unit forward decl)
