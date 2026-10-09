"""G31: a K_BCIR tiling plan exported as an MLIR Transform script, applied by stock mlir-opt.

The parent had no Transform-dialect code (the audit's item 14), so no stock MLIR pipeline
could apply a decision K_BCIR made. These witnesses hold the export to MLIR's own output: the
`scf.for` nest `mlir-opt --transform-interpreter` produces from the script must visit exactly
the plan's tile origins, in the plan's order -- on every loop order (the 3-cycles, where an
interchange and its inverse differ, included), on full-dimension and ragged tiles, and on the
planner's own plans. A skip without mlir-opt, never a pass.
"""

from __future__ import annotations

from bcir.kbcir.matmul import TilePlan
from bcir.lower import transform
from bcir.lower.transform import (
    apply,
    export_matmul,
    interchange,
    loop_nest,
    nest_origins,
)
from bcir.tests.transform_fixtures import mismatches, planned


def _reference(M, N, K, plan):
    steps = {"i": plan.tile_m, "j": plan.tile_n, "k": plan.tile_k}
    limits = {"i": M, "j": N, "k": K}
    axes = plan.loop_order
    out = []
    for a in range(0, limits[axes[0]], steps[axes[0]]):
        for b in range(0, limits[axes[1]], steps[axes[1]]):
            for c in range(0, limits[axes[2]], steps[axes[2]]):
                at = dict(zip(axes, (a, b, c)))
                out.append((at["i"], at["j"], at["k"]))
    return out


def test_the_script_is_the_plan_in_text():
    plan = TilePlan(32, 16, 8, "kij", 0, 0, 0, True)
    text = export_matmul(64, 48, 40, plan)
    assert "tile_sizes [32, 16, 8] interchange = [2, 0, 1]" in text
    assert "linalg.matmul ins(%a, %b : memref<64x40xf32>, memref<40x48xf32>)" in text
    assert interchange("ijk") == [0, 1, 2] and interchange("jki") == [1, 2, 0]
    for bad in ("ij", "iik", "abc"):
        try:
            interchange(bad)
        except ValueError:
            continue
        raise AssertionError(f"the loop order {bad!r} was exported")


def test_mlir_applies_every_loop_order_as_planned():
    """All six loop orders over a shape no tile divides: the nest MLIR builds visits the
    plan's origins in the plan's order. The inverse interchange -- the reading a careless
    export would make -- is caught on the two 3-cycles."""
    if apply(export_matmul(8, 8, 8, TilePlan(4, 4, 4, "ijk", 0, 0, 0, True)))[0].startswith(
        "skip:"
    ):
        return
    caught = 0
    original = transform.interchange

    def inverse(order):  # the inverse permutation: loop t's dimension read the other way
        forward = original(order)
        return sorted(range(3), key=lambda d: forward[d])

    for order in ("ijk", "ikj", "jik", "jki", "kij", "kji"):
        plan = TilePlan(32, 16, 8, order, 0, 0, 0, True)
        verdict, out = apply(export_matmul(70, 50, 30, plan))
        assert verdict == "ok", out
        assert list(nest_origins(loop_nest(out), order)) == _reference(70, 50, 30, plan), order
        try:
            transform.interchange = inverse
            verdict, out = apply(export_matmul(70, 50, 30, plan))
            got = list(nest_origins(loop_nest(out), order))
            caught += got != _reference(70, 50, 30, plan)
        finally:
            transform.interchange = original
    assert caught == 2  # exactly the 3-cycles kij and jki, whose inverses differ


def test_every_planned_export_is_applied_as_planned():
    """The planner's own plans over eight shapes, and two more loop orders with remainders per
    shape: MLIR applies every export exactly as planned. Listing the tile sizes in the loop
    order instead of linalg's (i, j, k) is caught."""
    bad = mismatches()
    if bad is None:
        return
    assert bad == 0 and len(planned()) == 24
    original = transform.matmul_transform

    def loop_ordered(plan):
        sizes = {"i": plan.tile_m, "j": plan.tile_n, "k": plan.tile_k}
        text = original(plan)
        wrong = ", ".join(str(sizes[a]) for a in plan.loop_order)
        return text.replace(f"[{plan.tile_m}, {plan.tile_n}, {plan.tile_k}]", f"[{wrong}]")

    try:
        transform.matmul_transform = loop_ordered
        assert mismatches() > 0
    finally:
        transform.matmul_transform = original
