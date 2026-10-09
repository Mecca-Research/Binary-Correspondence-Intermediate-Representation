"""G30 fixtures, shared by `test_polyhedral` and the GEM+ harness so both grade the same way:
generated depth-two nests, a reference that enumerates every pair of iteration instances and
judges legality by simulating each candidate execution order, and the grading itself."""

from __future__ import annotations

import random

from bcir.kbcir.polyhedral import NestError, nest_model
from bcir.model import Claim, Lane, LoopNest, Module, Opcode, Phase, Resource, StrideClass

NESTS = 160
#: The tilings the reference simulates: square, oblong both ways, and the two degenerate
#: tilings -- one row of tiles is the original order, one column of tiles is the interchange.
_TILES = ((2, 2), (2, 3), (3, 2), (None, 1), (1, None))


def nest_module(rows: int, cols: int, reads, writes, size: int = 256) -> Module:
    """A one-claim module whose claim is a nest: `reads` / `writes` are (rid, offset, row
    stride, col stride) maps."""
    m = Module(name="nest")
    for rid in sorted({r for r, *_ in (*reads, *writes)}):
        m.add_resource(Resource(rid=rid, shape=(size,)))
    claim = Claim(
        id=1,
        opcode=Opcode.ADD,
        lane=Lane.T,
        stride_class=StrideClass.TILE,
        count=rows * cols,
        rd=tuple(r for r, *_ in reads),
        wr=tuple(r for r, *_ in writes),
        op="nest",
        nest=LoopNest((rows, cols), tuple(reads) + tuple(writes)),
    )
    m.add_phase(Phase(phase_id=0, claims=[claim]))
    return m


def _map(r: random.Random, rid: int, rows: int, cols: int):
    rs = r.choice((0, 1, -1, 2, cols, -cols, cols + 1))
    cs = r.choice((0, 1, -1, 2, rows))
    low = min(0, rs * (rows - 1)) + min(0, cs * (cols - 1))
    return (rid, -low + r.randint(0, 6), rs, cs)


def random_nest(seed: int) -> Module:
    """Up to 6 x 6 iterations, one write map and one or two read maps over one or two
    resources, strides that alias (zero, negative, row-sized) as often as not."""
    r = random.Random(seed)
    rows, cols = r.randint(1, 6), r.randint(1, 6)
    rids = (10,) if r.random() < 0.6 else (10, 11)
    reads = [_map(r, r.choice(rids), rows, cols) for _ in range(r.randint(1, 2))]
    writes = [_map(r, rids[0], rows, cols)]
    return nest_module(rows, cols, reads, writes)


def reference(module: Module):
    """(dependences as {(rid, kind, distance)}, interchange legal, tiling legal, outer
    parallel, inner parallel) by enumerating every pair of instances and simulating orders."""
    claim = module.phases[0].claims[0]
    rows, cols = claim.nest.extents
    modes = ["read"] * len(claim.rd) + ["write"] * len(claim.wr)
    rids = list(claim.rd) + list(claim.wr)
    touches = []  # (instance, rid, mode, address)
    for i in range(rows):
        for j in range(cols):
            for (mrid, off, rs, cs), mode, rid in zip(claim.nest.maps, modes, rids):
                touches.append(((i, j), rid, mode, off + rs * i + cs * j))
    pairs = set()
    deps = set()
    for p, prid, pmode, paddr in touches:
        for q, qrid, qmode, qaddr in touches:
            if p < q and prid == qrid and paddr == qaddr and "write" in (pmode, qmode):
                kind = {"write": {"read": "flow", "write": "output"}, "read": {"write": "anti"}}[
                    pmode
                ][qmode]
                deps.add((prid, kind, (q[0] - p[0], q[1] - p[1])))
                pairs.add((p, q))

    def legal(key) -> bool:
        return all(key(p) < key(q) for p, q in pairs)

    interchange = legal(lambda x: (x[1], x[0]))
    tiling = all(
        legal(lambda x, th=th or rows, tw=tw or cols: (x[0] // th, x[1] // tw, x[0], x[1]))
        for th, tw in _TILES
    )
    outer = not any(p[0] != q[0] for p, q in pairs)
    inner = not any(p[0] == q[0] and p[1] != q[1] for p, q in pairs)
    return deps, interchange, tiling, outer, inner


def graded(module: Module):
    """(dependences agree, every legality flag agrees) of the model against the reference."""
    claim = module.phases[0].claims[0]
    model = nest_model(claim, module)
    deps, interchange, tiling, outer, inner = reference(module)
    ours = {(d.rid, d.kind, d.distance) for d in model.dependences}
    flags = (
        model.interchange_legal,
        model.tiling_legal,
        model.outer_parallel,
        model.inner_parallel,
    )
    return ours == deps, flags == (interchange, tiling, outer, inner)


def measure(count: int = NESTS) -> dict[str, float]:
    """The G30 rows: generated nests whose dependences, or whose legality, the model misjudges
    against the reference."""
    dep_bad = legal_bad = 0
    for seed in range(count):
        try:
            deps_ok, legal_ok = graded(random_nest(seed))
        except NestError:
            deps_ok = legal_ok = False
        dep_bad += not deps_ok
        legal_bad += not legal_ok
    return {
        "nest.dependence.disagreements": float(dep_bad),
        "nest.legality.disagreements": float(legal_bad),
    }
