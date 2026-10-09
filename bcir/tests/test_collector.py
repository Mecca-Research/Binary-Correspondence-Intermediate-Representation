"""OR-GC: the cyclic collector is out of the oracle's pure hot paths, and they leave it no work.

The 2026-10-06 audit (section 5.1) measured CPython's cyclic collector at up to 68% of a hot
path's time at 32,768 claims, inside calls that leave no cyclic garbage: every collection walked
the live heap and freed nothing. The entry points below now pause it (`bcir._collector.paused`),
and their value types carry no attribute dict (`slots=True`, section 5.2).

`test_no_collection_runs_inside_a_hot_path` was RED on the parent (`70e839d`): at scale 2 the
planner alone ran 9 collections per call. The garbage witness holds on both trees -- it is the
precondition that makes the pause free, and the reason a collection inside these paths could
never have freed anything.
"""

from __future__ import annotations

import dataclasses
import gc
import pickle

from bcir import _collector
from bcir.abi import AbiError
from bcir.abi.execution_plan_abi import decode_plan, encode_plan
from bcir.abi.streampack_abi import decode, encode
from bcir.gem import schedule, streampack
from bcir.gem.delta_chain import DeltaChain
from bcir.gem.execution_plan import plan_from_realization
from bcir.gem.overlap import optimize_scheduled
from bcir.kbcir import realize
from bcir.kbcir.cost import CostVector
from bcir.kbcir.realize import Candidate
from bcir.kbcir.weights import ENERGY, PERF
from bcir.model import Lane
from bcir.performance_audit import kbcir_streampack_fixture
from bcir.tests.sweep_fixtures import general_fixture
from bcir import verify as verifier


def _paths(scale: int = 2) -> dict:
    """Each paused entry point, as a call over the K_BCIR -> StreamPack fixture at `scale`."""
    module, target, theta = kbcir_streampack_fixture(scale)
    result = realize.optimize(module, target, theta)
    pack = streampack.hydrate_pipelined(module, result, plan="tmsao", depth=2)
    wire, durations = encode(pack), schedule.durations_from(result)
    blob = encode_plan(plan_from_realization(module, result, target, "eft"))
    read = decode_plan(blob)
    general = general_fixture(4 * scale, 8)
    return {
        "optimize": lambda: realize.optimize(module, target, theta),
        "hydrate": lambda: streampack.hydrate(module, result, "plan0"),
        "hydrate_pipelined": lambda: streampack.hydrate_pipelined(
            module, result, plan="tmsao", depth=2
        ),
        "decode": lambda: decode(wire),
        "verify_plan": lambda: verifier.verify_plan(
            module, result, target, theta=theta, policy=PERF
        ),
        "verify_pack": lambda: verifier.verify_pack(module, pack),
        "schedule_eft": lambda: schedule.schedule_eft(module, durations, target),
        "schedule_plan": lambda: schedule.schedule_plan(module, result, target, "eft"),
        "execute_tokens": lambda: schedule.execute_tokens(module, durations, target),
        "plan_from_realization": lambda: plan_from_realization(module, result, target, "eft"),
        "decode_plan": lambda: decode_plan(blob),
        "verify_execution_plan": lambda: verifier.verify_execution_plan(
            module, read, target=target
        ),
        "delta_chain.apply": _delta_apply(),
        "optimize_scheduled": lambda: optimize_scheduled(general, target, theta, ENERGY),
    }


_PAUSED = (
    realize.optimize,
    streampack.hydrate,
    streampack.hydrate_pipelined,
    decode,
    verifier.verify_plan,
    verifier.verify_pack,
    verifier.verify_execution_plan,
    schedule.schedule_eft,
    schedule.schedule_plan,
    schedule.execute_tokens,
    plan_from_realization,
    decode_plan,
    optimize_scheduled,
    DeltaChain.apply,
    DeltaChain.build.__func__,
)


def test_each_hot_path_leaves_no_cyclic_garbage():
    """The witness the pause rests on: a call made with the collector off leaves nothing for
    a collection to find, so pausing it inside the call changes no memory and no result."""
    paths = _paths()
    enabled = gc.isenabled()
    try:
        for name, call in paths.items():
            gc.collect()
            gc.disable()
            call()
            assert gc.collect() == 0, name
            gc.enable()
    finally:
        (gc.enable if enabled else gc.disable)()


def test_no_collection_runs_inside_a_hot_path():
    """The collector is paused for the length of each call: no collection starts inside one,
    however much it allocates (the parent ran 9 inside the planner at this scale)."""
    paths = _paths()
    started: list[str] = []

    def count(phase, _info):
        if phase == "start":
            started.append(phase)

    gc.callbacks.append(count)
    try:
        for name, call in paths.items():
            gc.collect()
            started.clear()
            call()
            assert started == [], (name, len(started))
    finally:
        gc.callbacks.remove(count)


def test_every_hot_entry_point_wears_the_pause():
    """The paused set is the audit's list -- planner, hydrate, decode, plan verification, the
    schedulers -- plus the plan's own path; a refactor that drops the decorator from one is
    caught here before the collection rows notice."""
    for fn in _PAUSED:
        assert getattr(fn, "__wrapped__", None) is not None, fn.__qualname__
        assert fn.__code__ is _collector.paused(len).__code__, fn.__qualname__


def test_the_pause_restores_the_callers_collector_on_every_way_out():
    enabled = gc.isenabled()
    try:
        gc.enable()
        decode(encode(streampack.hydrate(*_tiny())))
        assert gc.isenabled(), "a return left the collector paused"
        try:
            decode(b"\x00" * 8)
        except AbiError:
            pass
        assert gc.isenabled(), "an exception left the collector paused"
        gc.disable()
        decode(encode(streampack.hydrate(*_tiny())))
        assert not gc.isenabled(), "the pause turned on a collector its caller had turned off"
    finally:
        (gc.enable if enabled else gc.disable)()


def test_a_nested_entry_leaves_the_pause_to_the_outermost():
    """hydrate places its schedule and the plan verifier re-derives one: the inner entry point
    must not resume the collector while the outer one is still running."""
    seen: list[bool] = []

    @_collector.paused
    def inner():
        return None

    @_collector.paused
    def outer():
        inner()
        seen.append(gc.isenabled())

    enabled = gc.isenabled()
    try:
        gc.enable()
        outer()
        assert seen == [False] and gc.isenabled()
    finally:
        (gc.enable if enabled else gc.disable)()


def test_the_hot_value_types_carry_no_attribute_dict():
    """Section 5.2: `CostVector` and `Candidate` are slotted, and still hash, compare, copy,
    replace and pickle as the frozen dataclasses they were."""
    cost = CostVector.of(compute=3)
    cand = Candidate(lane=Lane.U, width=8, name="vec8", base=cost, reads=(1,), writes=(2,))
    for value in (cost, cand):
        assert not hasattr(value, "__dict__"), type(value).__name__
        assert pickle.loads(pickle.dumps(value)) == value
    assert dataclasses.replace(cand, width=16).width == 16
    assert hash(cand) == hash(dataclasses.replace(cand))
    try:
        cand.width = 4  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:  # pragma: no cover - a regression
        raise AssertionError("a slotted Candidate accepted an assignment")


def _delta_apply():
    """A call that applies the next declared delta to a chain built outside it (G18's path):
    one claim's realization replaced per call, as the planner's audit fixture declares it."""
    from bcir.kbcir.delta import Delta
    from bcir.tests.delta_fixtures import PLAN, audit_delta
    from bcir.tests.planner_fixtures import audit_fixture

    module, h, theta = audit_fixture(2)
    chain = DeltaChain.build(module, h, theta, PERF, PLAN, 2)
    step = iter(range(1 << 30))

    def apply():
        claims, resources = audit_delta(chain.module, next(step))
        return chain.apply(Delta(claims, resources))

    return apply


def _tiny():
    from bcir.examples import matmul_tiled
    from bcir.kbcir.cost import TargetProfile, Theta

    module = matmul_tiled(n=16, tile=8)
    return module, realize.optimize(module, TargetProfile.x86_avx2(), Theta.mem_bound())
