"""Generic registerize -> stackify lowering: the shared foundation for the
stack-machine bytecode targets (WASM, JVM bytecode, .NET CIL).

BCIR is register/registry-first; WASM/JVM/CIL are stack machines. The clean way to
target all three is to linearize a register-form expression DAG into a single
**stack-op sequence once** (`stackify`), then attach thin per-format encoders. The
BCIR claim body's compute graph (loads + `bcir.compute` binops + store) maps onto
the `Expr` tree here; `stackify` emits postfix order, and `to_wasm/to_jvm/to_cil`
render the same sequence into each target's mnemonics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

# --- a minimal register-form expression IR (what a BCIR compute fragment becomes) ---


@dataclass(frozen=True)
class Const:
    value: float


@dataclass(frozen=True)
class Local:
    index: int
    name: str = ""


@dataclass(frozen=True)
class BinOp:
    op: str  # add | sub | mul | div
    lhs: "Expr"
    rhs: "Expr"


Expr = Union[Const, Local, BinOp]

_BINOPS = {"add", "sub", "mul", "div"}


@dataclass(frozen=True)
class StackOp:
    kind: str  # const | local_get | local_set | add | sub | mul | div
    arg: object = None


def stackify(expr: Expr) -> list[StackOp]:
    """Linearize an expression tree into a postfix stack-op sequence.

    This is the registerize->stackify core: operands are emitted before the
    operator, so the result is exactly the operand-stack order every stack VM
    (WASM/JVM/CIL) expects. (Shared subexpressions become locals upstream.)
    """
    if isinstance(expr, Const):
        return [StackOp("const", expr.value)]
    if isinstance(expr, Local):
        return [StackOp("local_get", expr.index)]
    if isinstance(expr, BinOp):
        if expr.op not in _BINOPS:
            raise ValueError(f"unknown binop {expr.op!r}")
        return stackify(expr.lhs) + stackify(expr.rhs) + [StackOp(expr.op)]
    raise TypeError(f"not an Expr: {expr!r}")


def assign(local_index: int, expr: Expr) -> list[StackOp]:
    """Stack ops for `local[i] = expr` (evaluate, then store)."""
    return stackify(expr) + [StackOp("local_set", local_index)]


# --- thin per-target encoders (one stack sequence -> three bytecodes) ---
#
# HONESTY (H5): these are MNEMONIC-TEXT emitters -- they render the stack sequence into each target's
# assembly mnemonics, NOT into assembled bytes. Execution-validation lives in tests/test_stackify_exec.py:
# the JVM path is assembled to a real `.class` and RUN on the in-container `java` (a small class-file writer
# for this bounded float subset); all three encoders are SEMANTICALLY differentiated across a fuzz corpus by
# interpreting the emitted mnemonics under each VM's stack semantics. CIL stays a documented skip (no
# ilasm/mono/dotnet in CI); the execution-validated WASM path is the resident clang->wasm-ld->node pipeline in
# lower/wasm.py (run by tests/test_wasm.py), distinct from this `to_wasm` text encoder.

_WASM = {"add": "f32.add", "sub": "f32.sub", "mul": "f32.mul", "div": "f32.div"}
_JVM = {"add": "fadd", "sub": "fsub", "mul": "fmul", "div": "fdiv"}
_CIL = {"add": "add", "sub": "sub", "mul": "mul", "div": "div"}


def to_wasm(ops: list[StackOp]) -> list[str]:
    out = []
    for o in ops:
        if o.kind == "const":
            out.append(f"f32.const {o.arg}")
        elif o.kind == "local_get":
            out.append(f"local.get {o.arg}")
        elif o.kind == "local_set":
            out.append(f"local.set {o.arg}")
        else:
            out.append(_WASM[o.kind])
    return out


def to_jvm(ops: list[StackOp]) -> list[str]:
    out = []
    for o in ops:
        if o.kind == "const":
            out.append(f"ldc {o.arg}f")
        elif o.kind == "local_get":
            out.append(f"fload {o.arg}")
        elif o.kind == "local_set":
            out.append(f"fstore {o.arg}")
        else:
            out.append(_JVM[o.kind])
    return out


def to_cil(ops: list[StackOp]) -> list[str]:
    out = []
    for o in ops:
        if o.kind == "const":
            out.append(f"ldc.r4 {o.arg}")
        elif o.kind == "local_get":
            out.append(f"ldloc.{o.arg}")
        elif o.kind == "local_set":
            out.append(f"stloc.{o.arg}")
        else:
            out.append(_CIL[o.kind])
    return out
