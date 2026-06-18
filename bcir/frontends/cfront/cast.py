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
    base: str                       # scalar name or struct/union tag
    ptr: int = 0                    # pointer depth
    array: tuple = ()               # array dimensions (outer-first)
    aggregate: str = ""             # "struct" | "union" | "" (scalar)
    quals: tuple = ()               # ("const",) / ("volatile",) — drives MMIO lowering (L5)


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
class Binary:
    op: str                         # + - * / % & | ^ << >> == != < > <= >= && ||
    lhs: object
    rhs: object


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
    members: tuple                  # (TypeRef, name)


@dataclass
class Unit:
    aggregates: dict = field(default_factory=dict)   # tag -> Aggregate
    funcs: list = field(default_factory=list)        # Func, in order
