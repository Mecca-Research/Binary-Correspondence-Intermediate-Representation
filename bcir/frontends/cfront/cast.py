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
    base: str                       # scalar name or struct/union tag (a funcptr alias: its spelling)
    ptr: int = 0                    # pointer depth
    array: tuple = ()               # array dimensions (outer-first)
    aggregate: str = ""             # "struct" | "union" | "" (scalar)
    quals: tuple = ()               # ("const",) / ("volatile",) — drives MMIO lowering (L5)
    funcptr: bool = False           # a function-pointer alias (RET (*name)(PARAMS)) — base is its name
    func_ret: object = None         # the return TypeRef (funcptr only)
    func_params: tuple = ()         # the parameter TypeRefs (funcptr only) — for faithful emit


# --- expressions ---
@dataclass(frozen=True)
class IntLit:
    value: int


@dataclass(frozen=True)
class Name:
    ident: str


@dataclass(frozen=True)
class Unary:
    op: str                         # - ~ ! * & (deref / address-of)
    operand: object


@dataclass(frozen=True)
class Cast:
    type: TypeRef                   # the target type — `(type)operand`
    operand: object


@dataclass(frozen=True)
class SizeOf:
    type: object = None             # a TypeRef for `sizeof(type)`, else None
    expr: object = None             # the operand for `sizeof expr` (its static type's size)


@dataclass(frozen=True)
class Binary:
    op: str                         # + - * / % & | ^ << >> == != < > <= >= && ||
    lhs: object
    rhs: object


@dataclass(frozen=True)
class Ternary:
    cond: object                    # `cond ? then : els` (conditional expression)
    then: object
    els: object


@dataclass(frozen=True)
class Assign:
    target: object                  # an lvalue expr
    value: object


@dataclass(frozen=True)
class Index:
    base: object
    index: object


@dataclass(frozen=True)
class Member:
    base: object
    field: str
    arrow: bool                     # True for `->`, False for `.`


@dataclass(frozen=True)
class CallExpr:
    callee: str
    args: tuple


# --- statements ---
@dataclass(frozen=True)
class Decl:
    type: TypeRef
    name: str
    init: object = None


@dataclass(frozen=True)
class Return:
    value: object = None


@dataclass(frozen=True)
class ExprStmt:
    expr: object


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
    init: object                    # a Decl / ExprStmt, or None
    cond: object                    # the loop test (IntLit(1) if omitted)
    step: object                    # an ExprStmt run at the end of each iteration, or None
    body: tuple = ()


@dataclass(frozen=True)
class DoWhile:
    cond: object                    # `do body while (cond);` — body runs, then cond is tested
    body: tuple = ()


@dataclass(frozen=True)
class Break:
    pass                            # `break;` — exit the nearest enclosing loop


@dataclass(frozen=True)
class Continue:
    pass                            # `continue;` — jump to the nearest enclosing loop's next iteration


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


@dataclass(frozen=True)
class Aggregate:
    kind: str                       # struct | union
    tag: str
    members: tuple                  # (TypeRef, name, bit_width)
    packed: bool = False            # __attribute__((packed)) — no inter-member padding
    align: int = 0                  # __attribute__((aligned(N))) / alignas(N) — forced alignment


@dataclass(frozen=True)
class Global:
    """A top-level (file-scope) variable, e.g. a `const uint8_t table[] = { ... }` (the seam C23
    `#embed` initializers land in once the preprocessor has expanded them to a byte list)."""
    type: TypeRef
    name: str
    init: tuple = ()                # initializer element expressions (for an array/scalar)


@dataclass
class Unit:
    aggregates: dict = field(default_factory=dict)   # tag -> Aggregate
    funcs: list = field(default_factory=list)        # Func, in order
    globals: list = field(default_factory=list)      # Global, in order
