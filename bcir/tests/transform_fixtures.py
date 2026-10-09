"""G31 fixtures, shared by `test_transform_export` and the GEM+ harness so both grade the same
way: K_BCIR's own matmul plans over a spread of shapes, and their exports' fidelity under
stock mlir-opt."""

from __future__ import annotations

from bcir.kbcir.matmul import TilePlan, plan_matmul
from bcir.lower.transform import faithful

#: Shapes the planner tiles: square, skinny, ragged (dimensions no tile divides), small.
SHAPES = (
    (64, 64, 64),
    (48, 96, 32),
    (100, 60, 20),
    (128, 32, 64),
    (37, 23, 11),
    (96, 96, 8),
    (16, 128, 48),
    (72, 40, 56),
)


def planned():
    """(M, N, K, plan) for the planner's own plan of every shape, and two more per shape with
    the other loop orders the planner costs and tiles that leave a remainder."""
    out = []
    for M, N, K in SHAPES:
        out.append((M, N, K, plan_matmul(M, N, K)))
        for order in ("ikj", "jik"):
            out.append(
                (
                    M,
                    N,
                    K,
                    TilePlan(max(1, M // 3), max(1, N // 2), max(1, K // 4), order, 0, 0, 0, True),
                )
            )
    return out


def mismatches() -> int | None:
    """Plans whose export MLIR does not apply as planned; None without mlir-opt."""
    bad = 0
    for M, N, K, plan in planned():
        verdict, _why = faithful(M, N, K, plan)
        if verdict.startswith("skip:"):
            return None
        bad += verdict != "match"
    return bad


def measure() -> dict[str, float]:
    """The G31 harness row, where mlir-opt is."""
    bad = mismatches()
    return {} if bad is None else {"transform.export.mismatch": float(bad)}
