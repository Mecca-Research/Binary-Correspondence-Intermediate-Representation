"""The MLIR Transform export: a K_BCIR tiling plan as a Transform-dialect script (G31).

The 2026-10-06 audit (item 14) found no MLIR Transform-dialect code: BCIR's plans reached
MLIR only as BCIR's own ops, so a stock MLIR pipeline could not apply a decision K_BCIR made.
This module exports one such decision -- the matmul's tiling plan (`kbcir.matmul.TilePlan`:
tile extents and the outer loop order) -- as the script stock `mlir-opt` applies with
`--transform-interpreter`:

    transform.named_sequence @__transform_main(%root) {
      %mm = transform.structured.match ops{["linalg.matmul"]} in %root
      transform.structured.tile_using_for %mm tile_sizes [tm, tn, tk] interchange = [...]
    }

beside a `linalg.matmul` payload of the planned shape. `linalg.matmul` iterates (i, j, k) over
(M, N, K), so the plan's loop order is the interchange: "jik" is [1, 0, 2].

**Held to the plan, not to itself.** `apply` runs stock `mlir-opt`; `loop_nest` reads the
`scf.for` nest MLIR produced -- bounds and steps resolved through the `arith.constant`s
that define them, outermost first -- and `nest_origins` enumerates the tile origins that nest
visits. The export is faithful when those are exactly `kbcir.matmul.tile_origins` for the
plan, in order: what MLIR does with the script is what K_BCIR planned, judged from MLIR's
output alone. A host without `mlir-opt` is a skip, never a pass.

Not built: vectorization, packing and the other Transform operations; plans other than the
matmul's tiling; reading a Transform script back into a plan.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

#: linalg.matmul's iteration dimensions, in its own order.
_DIMS = "ijk"


def interchange(loop_order: str) -> list[int]:
    """The `interchange` of a loop order: the position of each outer loop's dimension in
    linalg.matmul's (i, j, k)."""
    if sorted(loop_order) != sorted(_DIMS):
        raise ValueError(f"a loop order is a permutation of {_DIMS!r}, got {loop_order!r}")
    return [_DIMS.index(axis) for axis in loop_order]


def matmul_payload(M: int, N: int, K: int, elem: str = "f32") -> str:
    """A `linalg.matmul` of the planned shape over memrefs: C[M x N] += A[M x K] B[K x N]."""
    a, b, c = f"memref<{M}x{K}x{elem}>", f"memref<{K}x{N}x{elem}>", f"memref<{M}x{N}x{elem}>"
    return (
        f"func.func @bcir_matmul(%a: {a}, %b: {b}, %c: {c}) {{\n"
        f"  linalg.matmul ins(%a, %b : {a}, {b}) outs(%c : {c})\n"
        "  return\n}\n"
    )


def matmul_transform(plan) -> str:
    """The Transform script that applies `plan` (a `kbcir.matmul.TilePlan`)."""
    order = ", ".join(str(d) for d in interchange(plan.loop_order))
    any_op = "!transform.any_op"
    return (
        "module attributes {transform.with_named_sequence} {\n"
        f"  transform.named_sequence @__transform_main(%root: {any_op} "
        "{transform.readonly}) {\n"
        f'    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %root : '
        f"({any_op}) -> {any_op}\n"
        f"    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %mm tile_sizes "
        f"[{plan.tile_m}, {plan.tile_n}, {plan.tile_k}] interchange = [{order}] : "
        f"({any_op}) -> ({any_op}, {any_op}, {any_op}, {any_op})\n"
        "    transform.yield\n"
        "  }\n"
        "}\n"
    )


def export_matmul(M: int, N: int, K: int, plan, elem: str = "f32") -> str:
    """The payload and its script, one file stock `mlir-opt --transform-interpreter` reads."""
    return matmul_payload(M, N, K, elem) + matmul_transform(plan)


def apply(text: str) -> tuple[str, str]:
    """Run stock `mlir-opt --transform-interpreter` over an export. Returns (verdict, output):
    `ok`, `failed`, or `skip:<reason>` without a coherent mlir-opt -- a skip, never a pass."""
    from ..toolchain import resolve_llvm_tools

    tools = resolve_llvm_tools("mlir-opt", pipeline="MLIR Transform export")
    if not tools.ok:
        return f"skip:{tools.message}", ""
    with tempfile.TemporaryDirectory(prefix="bcir-transform-") as work:
        path = os.path.join(work, "export.mlir")
        with open(path, "w", newline="\n") as f:
            f.write(text)
        run = subprocess.run(
            [tools.paths["mlir-opt"], "--transform-interpreter", path],
            capture_output=True,
            text=True,
        )
    if run.returncode != 0:
        return "failed", run.stdout + run.stderr
    return "ok", run.stdout


_CONST = re.compile(r"(%[\w.]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*index")
_FOR = re.compile(r"scf\.for\s+%[\w.]+\s*=\s*(%[\w.]+)\s+to\s+(%[\w.]+)\s+step\s+(%[\w.]+)")


def loop_nest(output: str) -> list[tuple[int, int, int]]:
    """The `scf.for` loops of the payload function as (lower, upper, step), outermost first,
    each bound read through the constant that defines it. A loop bound that is not such a
    constant is refused (ValueError): the export's loops are over the static shape."""
    payload = output.split("module attributes", 1)[0]
    consts = {name: int(value) for name, value in _CONST.findall(payload)}
    nest = []
    for lo, hi, step in _FOR.findall(payload):
        try:
            nest.append((consts[lo], consts[hi], consts[step]))
        except KeyError as exc:
            raise ValueError(f"a loop bound {exc} is not an index constant") from None
    return nest


def nest_origins(nest, loop_order: str):
    """The (i, j, k) tile origins the nest visits, in its own order -- the nest's loops read
    as the plan's axes, outermost first."""
    if len(nest) != len(loop_order):
        raise ValueError(f"a {len(loop_order)}-deep tiling produced {len(nest)} loops")
    ranges = [range(lo, hi, step) for lo, hi, step in nest]
    for a in ranges[0]:
        for b in ranges[1]:
            for c in ranges[2]:
                at = dict(zip(loop_order, (a, b, c)))
                yield at["i"], at["j"], at["k"]


def faithful(M: int, N: int, K: int, plan) -> tuple[str, str]:
    """Whether MLIR, applying the export, visits exactly the plan's tile origins in order:
    (`match`, ''), (`MISMATCH`, why), or the skip/failure verdict of `apply`."""
    from ..kbcir.matmul import tile_origins

    verdict, out = apply(export_matmul(M, N, K, plan))
    if verdict != "ok":
        return verdict, out
    try:
        got = list(nest_origins(loop_nest(out), plan.loop_order))
    except ValueError as exc:
        return "MISMATCH", str(exc)
    want = list(tile_origins(M, N, K, plan))
    if got != want:
        return "MISMATCH", f"MLIR visits {got[:6]}..., the plan {want[:6]}..."
    return "match", ""


__all__ = [
    "apply",
    "export_matmul",
    "faithful",
    "interchange",
    "loop_nest",
    "matmul_payload",
    "matmul_transform",
    "nest_origins",
]
