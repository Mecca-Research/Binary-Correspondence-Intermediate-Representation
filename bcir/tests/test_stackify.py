"""Generic stackify (Phase 7): one stack sequence -> WASM / JVM / CIL."""

from bcir.lower.stackify import (
    BinOp,
    Const,
    Local,
    StackOp,
    assign,
    stackify,
    to_cil,
    to_jvm,
    to_wasm,
)


def _expr():
    # (A + B) * D, with A=local0, B=local1, D=local2
    return BinOp("mul", BinOp("add", Local(0, "A"), Local(1, "B")), Local(2, "D"))


def test_stackify_is_postfix():
    ops = stackify(_expr())
    assert [(o.kind, o.arg) for o in ops] == [
        ("local_get", 0), ("local_get", 1), ("add", None),
        ("local_get", 2), ("mul", None),
    ]


def test_one_sequence_three_targets():
    ops = stackify(_expr())
    assert to_wasm(ops) == ["local.get 0", "local.get 1", "f32.add", "local.get 2", "f32.mul"]
    assert to_jvm(ops) == ["fload 0", "fload 1", "fadd", "fload 2", "fmul"]
    assert to_cil(ops) == ["ldloc.0", "ldloc.1", "add", "ldloc.2", "mul"]


def test_const_and_assign():
    ops = assign(3, BinOp("add", Local(0), Const(1.5)))
    assert ops[-1] == StackOp("local_set", 3)
    assert to_wasm(ops) == ["local.get 0", "f32.const 1.5", "f32.add", "local.set 3"]
