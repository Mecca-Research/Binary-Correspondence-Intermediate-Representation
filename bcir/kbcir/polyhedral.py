"""Polyhedral depth two: a 2-D loop nest's exact dependences and what they make legal (G30).

The 2026-10-06 audit (item 14) found the polyhedral model one-dimensional: the affine region
reads 1-D maps, and a tile claim -- a 2-D nest -- was opaque, so nothing could say whether its
loops may be interchanged, tiled, parallelized or vectorized. A claim that declares its nest
(`model.graph.LoopNest`: extents (R, C), one affine map per operand) is now a `nest` region
whose local model is the nest's exact dependence structure.

**The semantics.** Iterations (i, j) run in lexicographic order; each reads every operand it
reads before it writes any. Two accesses to one address are a dependence when at least one
writes: flow (a write, then a read), anti (a read, then a write) or output (two writes), from
the earlier iteration P to the later Q, with the distance vector Q - P -- lexicographically
positive by construction. (Accesses of one iteration are ordered by that iteration's own
body, not by the loops, and carry no distance.)

**The analysis** (`nest_model`) solves, for every source instance and every target map, the
target's access equation `t0 + tr * i' + tc * j' = a` over the box exactly -- O(R * C * R),
never the O((R * C)^2) enumeration of every instance pair the reference uses -- and returns
the set of distance vectors with their kinds. From that set:

* **interchange** (j outer) is legal iff every distance stays lexicographically positive
  read as (dj, di) -- i.e. iff no distance has dj < 0;
* **rectangular tiling** is legal iff the nest is fully permutable, every distance
  non-negative in both components -- at depth two the same condition, stated separately
  because deeper nests part them;
* the **outer loop is parallel** iff no dependence is carried by it (di > 0), the **inner
  loop** iff none is carried by it (di == 0, dj > 0) -- the latter is what lets a vector
  width run across j.

A map that leaves its resource, a malformed nest, or one whose maps do not match the claim's
operands is refused (`NestError`), never analysed. Not built: depth three and beyond, skewing
and wavefronts, fusion across claims, and parametric (symbolic) extents.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Above this many iterations the analysis refuses (its cost is R * C * R per map pair).
NEST_LIMIT = 4096


class NestError(ValueError):
    """A loop nest the depth-two model cannot analyse: malformed, mismatched or out of bounds."""


@dataclass(frozen=True)
class NestDependence:
    rid: int
    kind: str  # "flow" | "anti" | "output"
    distance: tuple[int, int]


@dataclass(frozen=True)
class NestModel:
    """A nest's dependences (sorted, distinct) and the legality they imply."""

    extents: tuple[int, int]
    dependences: tuple[NestDependence, ...]
    interchange_legal: bool
    tiling_legal: bool
    outer_parallel: bool
    inner_parallel: bool

    @property
    def distances(self) -> frozenset:
        return frozenset(d.distance for d in self.dependences)


def _int(value, what: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NestError(f"{what} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise NestError(f"{what} must be at least {minimum}, got {value}")
    return value


def nest_accesses(claim, module) -> list[tuple[int, str, int, int, int]]:
    """The claim's checked accesses: (rid, mode, offset, row stride, col stride), reads first."""
    nest = claim.nest
    if nest is None:
        raise NestError(f"claim {claim.id} declares no loop nest")
    if not isinstance(nest.extents, tuple) or len(nest.extents) != 2:
        raise NestError(f"claim {claim.id}'s nest extents are not (rows, cols)")
    rows, cols = (_int(e, "a nest extent", 1) for e in nest.extents)
    if rows * cols > NEST_LIMIT:
        raise NestError(f"claim {claim.id}'s nest has {rows * cols} iterations, over {NEST_LIMIT}")
    operands = [(rid, "read") for rid in claim.rd] + [(rid, "write") for rid in claim.wr]
    if len(nest.maps) != len(operands):
        raise NestError(
            f"claim {claim.id}'s nest gives {len(nest.maps)} maps for {len(operands)} operands"
        )
    out = []
    for (rid, mode), entry in zip(operands, nest.maps):
        if len(entry) != 4:
            raise NestError("a nest map is (rid, offset, row stride, col stride)")
        mrid, offset, rs, cs = (_int(v, "a nest map field") for v in entry)
        if mrid != rid:
            raise NestError(f"claim {claim.id}'s nest maps resource {mrid} where it names {rid}")
        resource = module.resources.get(rid)
        if resource is None:
            raise NestError(f"resource {rid} is not in the module")
        low = offset + min(0, rs * (rows - 1)) + min(0, cs * (cols - 1))
        high = offset + max(0, rs * (rows - 1)) + max(0, cs * (cols - 1))
        if low < 0 or high >= resource.count:
            raise NestError(
                f"claim {claim.id}'s map of resource {rid} reaches [{low}, {high}], outside "
                f"[0, {resource.count - 1}]"
            )
        out.append((rid, mode, offset, rs, cs))
    return out


def _kind(first: str, second: str) -> str | None:
    if first == "write" and second == "read":
        return "flow"
    if first == "read" and second == "write":
        return "anti"
    if first == "write" and second == "write":
        return "output"
    return None


def nest_model(claim, module) -> NestModel:
    """The nest's exact dependences and legality (module docstring)."""
    accesses = nest_accesses(claim, module)
    rows, cols = claim.nest.extents
    found = set()
    for srid, smode, s0, sr, sc in accesses:
        for trid, tmode, t0, tr, tc in accesses:
            if srid != trid or (smode == "read" and tmode == "read"):
                continue
            for i in range(rows):
                for j in range(cols):
                    rest = s0 + sr * i + sc * j - t0
                    for i2 in range(rows):
                        r = rest - tr * i2
                        if tc == 0:
                            js = range(cols) if r == 0 else ()
                        elif r % tc == 0 and 0 <= r // tc < cols:
                            js = (r // tc,)
                        else:
                            js = ()
                        for j2 in js:
                            if (i2, j2) > (i, j):  # the target instance runs later
                                kind = _kind(smode, tmode)
                                if kind is not None:
                                    found.add(NestDependence(srid, kind, (i2 - i, j2 - j)))
    deps = tuple(sorted(found, key=lambda d: (d.distance, d.rid, d.kind)))
    distances = [d.distance for d in deps]
    interchange = all(dj > 0 or (dj == 0 and di > 0) for di, dj in distances)
    tiling = all(di >= 0 and dj >= 0 for di, dj in distances)
    outer = not any(di > 0 for di, _dj in distances)
    inner = not any(di == 0 and dj > 0 for di, dj in distances)
    return NestModel((rows, cols), deps, interchange, tiling, outer, inner)


__all__ = ["NEST_LIMIT", "NestDependence", "NestError", "NestModel", "nest_accesses", "nest_model"]
