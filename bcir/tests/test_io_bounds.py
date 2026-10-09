"""G25: the red-blue pebble I/O bound and the hierarchical roofline -- in the stack and held sound.

The parent had no data-movement bound at all (the audit's item 15): every member of the stack
bounded time. These witnesses hold the bound to an exact optimum it shares no code with
(`optimal_io`, dynamic programming over every eviction policy), show the soundness check fires
on three ways of getting it wrong, and pin hand-derived instances -- including the one that
separates taking the largest excess from summing them.
"""

from __future__ import annotations

import random
import sys
import types
from fractions import Fraction

from bcir.kbcir import io_bounds
from bcir.kbcir.io_bounds import IoError, optimal_io, pebble_bound, roofline_bound


def generated(count: int = 300):
    """Seeded small instances: up to seven resources, eight steps, a capacity at least the
    largest working set, and a random live-out set."""
    out = []
    for seed in range(count):
        r = random.Random(seed)
        nres = r.randint(2, 7)
        sizes = {k: r.randint(1, 9) for k in range(nres)}
        steps = []
        for _ in range(r.randint(1, 8)):
            picked = r.sample(range(nres), r.randint(1, min(3, nres)))
            reads = tuple(x for x in picked if r.random() < 0.7)
            writes = tuple(x for x in picked if x not in reads or r.random() < 0.3)
            steps.append((reads, writes))
        live = tuple(x for x in range(nres) if r.random() < 0.4)
        need = max(sum(sizes[x] for x in set(a) | set(b)) for a, b in steps)
        out.append((steps, sizes, r.randint(need, need + 12), live))
    return out


def unsound(module, instances) -> int:
    return sum(module.pebble_bound(*inst).total > module.optimal_io(*inst) for inst in instances)


def test_the_bound_never_exceeds_the_exact_optimum_and_the_check_fires():
    """Over 300 generated instances the bound is at most the optimum Q (and equal to it on most);
    three wrong bounds are each caught: counting a live-out value as a reload, counting every
    read as compulsory, and -- on the hand-derived sweep below -- summing the excesses."""
    instances = generated()
    assert unsound(io_bounds, instances) == 0
    tight = sum(pebble_bound(*i).total == optimal_io(*i) for i in instances)
    assert tight >= 240, tight
    source = open(io_bounds.__file__, encoding="utf-8").read()
    for name, old, new, cases in (
        (
            "live_out_reload",
            "if read_later[i + 1].get(r, False))",
            "if (read_later[i + 1].get(r, False) or r in live_out))",
            instances,
        ),
        (
            "every_read",
            "if r not in touched and r not in written:",
            "if True:",
            instances,
        ),
        (
            "summed",
            "        if live - capacity > excess:\n            excess, at = live - capacity, i",
            "        if live - capacity > 0:\n            excess, at = excess + live - capacity, i",
            [SWEEP],
        ),
    ):
        assert source.count(old) == 1, name
        mutant = types.ModuleType(f"bcir.kbcir._io_{name}")
        mutant.__package__ = "bcir.kbcir"
        sys.modules[mutant.__name__] = mutant
        try:
            exec(compile(source.replace(old, new), mutant.__name__, "exec"), mutant.__dict__)
            assert unsound(mutant, cases) > 0, f"the soundness check cannot see {name}"
        finally:
            del sys.modules[mutant.__name__]


#: Four inputs read in a cyclic sweep twice, room for two: Belady keeps the soonest-used, so
#: the second pass reloads b and c -- Q = 4 compulsory loads + 2 = 6, which the bound states
#: exactly (largest excess 2, after the fourth read). Summing the excesses would claim more.
SWEEP = (
    [((r,), ()) for r in "abcd"] + [((r,), ()) for r in "abcd"],
    {r: 1 for r in "abcd"},
    2,
    (),
)


def test_hand_derived_instances():
    assert optimal_io(*SWEEP) == 6
    bound = pebble_bound(*SWEEP)
    assert (bound.compulsory_loads, bound.compulsory_stores, bound.capacity_excess) == (4, 0, 2)
    assert bound.total == 6 and bound.at_step == 3
    # a producer and a consumer: the value is born in fast memory and never moves; only the
    # live-out result is stored
    pipe = ([((), ("t",)), (("t",), ("y",))], {"t": 8, "y": 8}, 16, ("y",))
    assert pebble_bound(*pipe).total == optimal_io(*pipe) == 8
    # the same, with room for one: t must leave fast memory before y is written -- impossible,
    # since the consumer needs both resident at once: refused, never priced
    try:
        pebble_bound(pipe[0], pipe[1], 8, pipe[3])
    except IoError as exc:
        assert "cannot run" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a step over the capacity was priced")


def test_the_roofline_is_the_largest_of_compute_and_every_boundarys_traffic():
    """Two levels, fastest first: room for two at level 0, two more at level 1 (four in all).
    The sweep's traffic across the boundary below level 0 is the pebble bound at capacity 2
    (6 bytes), below level 1 at capacity 4 (the 4 compulsory loads)."""
    steps, sizes, _cap, live = SWEEP
    bound, terms = roofline_bound(steps, sizes, work=10, peak=5, levels=[(2, 3), (2, 1)])
    assert dict(terms) == {"compute": Fraction(2), "level0": Fraction(2), "level1": Fraction(4)}
    assert bound == 4
    for capacity in range(2, 10):  # more room never raises the traffic
        assert (
            pebble_bound(steps, sizes, capacity + 1).total
            <= pebble_bound(steps, sizes, capacity).total
        )


def test_malformed_inputs_are_refused():
    for bad in (
        lambda: pebble_bound([((1,), ())], {1: 0}, 4),
        lambda: pebble_bound([((1,), ())], {1: 2}, -1),
        lambda: pebble_bound([((2,), ())], {1: 2}, 4),
        lambda: pebble_bound([((1,), ())], {1: 2}, 4, live_out=(9,)),
        lambda: optimal_io([((k,), ()) for k in range(11)], {k: 1 for k in range(11)}, 4),
    ):
        try:
            bad()
        except IoError:
            continue
        raise AssertionError("a malformed instance was priced")
