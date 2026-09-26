"""G8 (S5-C): data movement as a first-class transformation (`bcir/kbcir/movement.py`).

Every module here is written into the test (`movement_fixtures`), so the suite runs from the
installed package. Each test drives a failure the parent shipped or a law the transform could
break: the example programs that read two memory tiers with no move (D-R2), a placement that
prices its transfers after the fact, a planner answer that is not the optimum it claims, a plan
whose edges lie about what moves, and one variant per movement law -- each refused by its own
law. The v3 wire laws are driven on the Python codec here and on the C twin in
`test_execution_plan.py`.
"""

from __future__ import annotations

from dataclasses import replace

from bcir.abi.execution_plan_abi import decode_plan, encode_plan, plan_version
from bcir.abi.streampack_abi import AbiError
from bcir.gem.execution_plan import plan_from_realization
from bcir.kbcir.device_manifest import check_bank_moves
from bcir.kbcir.movement import (
    MovementSpec,
    execution_order,
    execution_plan_of,
    plan_movement,
    price,
    races,
    replay_safe,
    site_choices,
    verify_movement,
    versions_of,
)
from bcir.model import Claim, Domain, Module, Opcode, Phase, Resource
from bcir.tests import movement_fixtures as mf
from bcir.tests import plan_fixtures as pf
from bcir.verify import verify_all, verify_execution_plan


def _plans():
    return mf.planned()


def test_the_example_programs_read_two_tiers_until_the_planner_moves_their_operands():
    """D-R2: a claim whose operands span two memory tiers must be an explicit move. The corpus
    carried nine (tiled_matmul 1, matmul_tiled 8); the planner's transform carries none."""
    before, after = mf.implicit_cross_tier()
    assert (before, after) == (9, 0)


def test_every_fixture_is_a_legal_program_before_it_moves():
    """A fixture the R-laws refuse would hide a transform defect behind its own."""
    from bcir.kbcir.realize import optimize

    h, theta = mf.target_and_theta()
    for name, module, _spec in mf.fixtures():
        assert verify_all(module, optimize(module, h, theta)) == [], name


def test_every_plan_the_planner_prices_satisfies_every_law():
    h, _theta = mf.target_and_theta()
    for name, (module, spec, mp) in _plans().items():
        for label, pm in [("best", mp.best), *mp.baselines.items()]:
            if pm is None:
                continue
            ep = execution_plan_of(pm, h)
            back = decode_plan(encode_plan(ep))
            assert back == ep, (name, label)
            m2 = pm.transform.module
            assert verify_movement(module, spec, m2, back) == [], (name, label)
            assert (
                verify_execution_plan(m2, back, target=h, result=pm.result, movement=(module, spec))
                == []
            ), (name, label)
            assert verify_all(m2, pm.result) == [], (name, label)
            assert check_bank_moves(m2) == [], (name, label)
            assert races(m2) == [], (name, label)
            assert ep.makespan == pm.makespan, (name, label)


def test_the_exact_planner_answers_the_optimum_of_the_joint_objective():
    """Every candidate the space holds, priced independently of the planner
    (`movement_fixtures.joint_optimum`), is no better than the answer, on every fixture."""
    for name, (module, spec, mp) in _plans().items():
        assert mp.method == "exact" and mp.optimum and mp.tmsao == "TMSAO-1", name
        best = mf.joint_optimum(module, spec)
        assert best is not None and best.key == mp.best.key, name


def test_movement_and_compute_are_chosen_jointly_not_priced_after_placement():
    """The disconnected step (compute-only sites, transfers priced afterwards) and the pre-G8
    greedy rule both pay makespan the joint planner does not; nothing beats the planner."""
    plans = _plans()
    worse = {"sequential": 0, "greedy": 0, "home": 0}
    for name, (_m, _s, mp) in plans.items():
        for label, pm in mp.baselines.items():
            if pm is None:
                continue
            assert pm.key >= mp.best.key, (name, label)
            worse[label] += pm.makespan > mp.best.makespan
    assert worse["sequential"] >= 4 and worse["greedy"] >= 3 and worse["home"] >= 4, worse
    assert mf.measure("greedy")["movement.excess"] > 0
    assert mf.measure()["movement.excess"] == 0


def test_a_module_that_needs_no_movement_is_planned_byte_for_byte_as_before():
    h, _theta = mf.target_and_theta()
    module, _spec, mp = _plans()["amortize"]
    best = mp.best
    assert not best.transform.moves and best.transform.module == module
    ep = execution_plan_of(best, h)
    plain = plan_from_realization(module, best.result, h, "eft", static_plan=best.static)
    assert encode_plan(ep) == encode_plan(plain) and plan_version(ep) < 3
    assert verify_execution_plan(module, ep, target=h) == []


def test_each_movement_law_refuses_its_own_violation():
    variants = mf.law_variants()
    laws = {law for _n, law, *_rest in variants}
    assert laws == {f"MV{i}" for i in range(1, 12)}, sorted(laws)
    for name, law, src, spec, mod, plan in variants:
        got = {d.law for d in verify_movement(src, spec, mod, plan)}
        assert law in got, (name, law, got)


def test_a_plan_that_moves_data_is_refused_without_its_source_and_spec():
    h, _theta = mf.target_and_theta()
    module, _spec, mp = _plans()["writeback"]
    ep = execution_plan_of(mp.best, h)
    got = {d.law for d in verify_execution_plan(mp.best.transform.module, ep, target=h)}
    assert got == {"MV11"}, got


def test_immutable_weights_are_dropped_and_reloaded_only_when_the_bank_is_full():
    """Semantic Swap: four weights cycle through an HBM that holds three. The planner evicts
    (Belady) and reloads; every reload is forced, and the same plan in a roomy HBM is thrash."""
    module, spec, mp = _plans()["stream"]
    best = mp.best
    assert best.transform.evictions and len(best.transform.moves) > 4
    reloads = [m for m in best.transform.moves if m.version == 0]
    assert len({m.rid for m in reloads}) < len(reloads)  # some weight was loaded twice
    h, _theta = mf.target_and_theta()
    ep = execution_plan_of(best, h)
    assert verify_movement(module, spec, best.transform.module, ep) == []
    roomy = replace(spec, hardware=mf.hardware(hbm=1 << 30))
    got = verify_movement(
        module, roomy, best.transform.module, replace(ep, spec_hash=roomy.digest())
    )
    assert {d.law for d in got} == {"MV9"} and any("thrash" in d.message for d in got)


def test_a_recomputable_temporary_is_rematerialized_from_a_replay_certified_producer():
    module, spec, mp = _plans()["remat"]
    best = mp.best
    remats = [m for m in best.transform.moves if m.kind == "rematerialized"]
    assert len(remats) == 1 and remats[0].producer == 10 and remats[0].cert
    assert mp.baselines["sequential"].makespan > best.makespan
    # the producer's certificate names what it read at which version
    versions = versions_of(module)
    assert versions.inputs[10] == ((1, 0),) and versions.writes[(10, 2)] == 1
    # a fenced producer, an in-place update, a device access are not replayable
    c = next(c for _p, c in execution_order(module) if c.id == 10)
    assert replay_safe(module, c) == ""
    assert replay_safe(module, replace(c, hazard="barriered"))
    assert replay_safe(module, replace(c, rd=(1, 2)))
    assert replay_safe(module, replace(c, volatile=True))


def test_an_approximate_transfer_needs_every_reader_to_tolerate_it():
    module, spec, mp = _plans()["compressed"]
    best = mp.best
    moved = best.transform.moves
    assert [m.kind for m in moved] == ["compressed"] and moved[0].bits == 8 and moved[0].cert
    assert mp.baselines["greedy"].makespan > best.makespan
    # a reader that declares no tolerance takes the exact transfer instead
    h, theta = mf.target_and_theta()
    strict = replace(
        module,
        phases=[
            replace(ph, claims=[replace(c, tolerance_ulp=0) for c in ph.claims])
            for ph in module.phases
        ],
    )
    exact = plan_movement(strict, spec, h, theta).best
    assert [m.kind for m in exact.transform.moves] == ["direct"]


def test_mutable_state_is_written_back_home_with_the_version_it_ends_at():
    module, spec, mp = _plans()["writeback"]
    best = mp.best
    wb = [m for m in best.transform.moves if m.coherence == "writeback"]
    final = versions_of(module).final[1]
    assert len(wb) == 1 and wb[0].version == final == 2 and wb[0].dst_copy == 1
    assert wb[0].dst_bank == "ram"


def test_a_route_with_no_direct_link_stages_through_the_host():
    module, _spec, mp = _plans()["staged"]
    hops = mp.best.transform.moves
    assert [(m.src_bank, m.dst_bank, m.kind) for m in hops] == [
        ("ssd", "ram", "staged"),
        ("ram", "hbm", "staged"),
    ]
    assert all(m.route == ("ssd", "ram", "hbm") for m in hops)


def test_a_device_register_never_moves_and_its_claim_stays_pinned():
    module, spec, mp = _plans()["device"]
    choices = site_choices(module, spec)
    assert choices[10] == ("ram",)
    m2 = mp.best.transform.module
    dev = next(c for ph in m2.phases for c in ph.claims if c.id == 10)
    assert dev.wr == (1,) and dev.domain == Domain.MMIO
    assert all(m.rid != 1 for m in mp.best.transform.moves)


def test_a_version_produced_in_the_same_phase_cannot_cross_banks():
    """A move sits in its own phase, so a claim cannot read, in another bank, what a claim of its
    own phase produced: that choice is infeasible, never silently stale."""
    h, theta = mf.target_and_theta()
    m = Module(name="samephase")
    m.add_resource(Resource(1, Domain.RAM, 4, (1024,), name="a"))
    m.add_resource(Resource(2, Domain.RAM, 4, (1024,), name="b"))
    m.add_resource(Resource(3, Domain.RAM, 4, (1024,), name="c"))
    m.add_phase(
        Phase(
            0,
            (),
            [
                Claim(id=1, opcode=Opcode.ADD, count=1024, rd=(1,), wr=(2,), op="vector.copy"),
                Claim(id=2, opcode=Opcode.ADD, count=1024, rd=(2,), wr=(3,), op="vector.copy"),
            ],
        )
    )
    spec = MovementSpec(mf.hardware())
    assert price(m, spec, {1: "ram", 2: "hbm"}, h, theta) is None
    assert price(m, spec, {1: "hbm", 2: "hbm"}, h, theta) is not None


def test_the_greedy_search_makes_no_optimality_claim_and_stays_legal():
    h, theta = mf.target_and_theta()
    module, spec, _mp = _plans()["matmul_tiled"]
    mp = plan_movement(module, spec, h, theta, method="greedy", budget=40)
    assert mp.method == "greedy" and not mp.optimum and mp.tmsao == "TMSAO-4"
    ep = execution_plan_of(mp.best, h)
    assert verify_movement(module, spec, mp.best.transform.module, ep) == []
    try:
        plan_movement(module, spec, h, theta, method="exact", budget=4)
        raise AssertionError("an exact search over a space past its budget was accepted")
    except ValueError as exc:
        assert "budget" in str(exc)


def test_the_spec_refuses_what_it_cannot_mean():
    hw = mf.hardware()
    for bad in (
        dict(classes=((1, "volatile"),)),
        dict(codecs=((1, 0),)),
        dict(codecs=((1, 32),)),
        dict(homes=((1, "tape"),)),
        dict(sites=((1, ()),)),
        dict(deadline=-1),
        dict(max_hops=0),
    ):
        try:
            MovementSpec(hw, **bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    a, b = MovementSpec(hw), MovementSpec(hw, classes=((1, "immutable"),))
    assert a.digest() != b.digest() and a.digest() == MovementSpec(hw).digest()


def test_the_lifetime_cover_law_reads_phase_positions_not_ids():
    """A module whose phase ids are not their topological positions (a movement transform's
    inbound phases; here, phases 7 then 3) is judged by where its phases run: the parent compared
    positions with ids and refused this correct plan; a lifetime that truly misses a phase is
    still refused."""
    from bcir.kbcir.realize import optimize
    from bcir.kbcir.static_memory import ResourceBankBinding, plan_static_memory

    h, theta = mf.target_and_theta()
    m = Module(name="relabelled")
    m.add_resource(Resource(1, Domain.RAM, 4, (1024,), name="a"))
    m.add_resource(Resource(2, Domain.RAM, 4, (1024,), name="b"))
    m.add_phase(
        Phase(
            7, (), [Claim(id=1, opcode=Opcode.ADD, count=1024, rd=(1,), wr=(2,), op="vector.copy")]
        )
    )
    m.add_phase(
        Phase(
            3,
            (7,),
            [Claim(id=2, opcode=Opcode.ADD, count=1024, rd=(2,), wr=(1,), op="vector.copy")],
        )
    )
    static = plan_static_memory(
        m, (ResourceBankBinding(1, "ram"), ResourceBankBinding(2, "ram")), mf.hardware()
    )
    result = optimize(m, h, theta)
    ep = plan_from_realization(m, result, h, "eft", static_plan=static)
    assert verify_execution_plan(m, ep, target=h) == []
    short = replace(
        ep, lifetimes=[replace(ep.lifetimes[0], last_phase=0, last_tick=1), *ep.lifetimes[1:]]
    )
    assert any("does not cover" in d.message for d in verify_execution_plan(m, short, target=h))


# --- the v3 wire, on the Python codec ------------------------------------------------------


def test_the_v3_wire_round_trips_and_its_laws_refuse_every_malformed_edge():
    plan = pf.v3_plan()
    blob = encode_plan(plan)
    assert plan_version(plan) == 3 and decode_plan(blob) == plan and pf.raw_encode(plan) == blob
    variants = pf.v3_variants()
    assert len(variants) >= 30
    for name, bad, broken in variants:
        # the decoder refuses the bytes (the C twin refuses the same bytes:
        # test_execution_plan.py) ...
        assert pf.wire_refuses(bad), name
        if broken is not None:
            # ... and the encoder refuses the plan before it is published
            try:
                encode_plan(broken)
                raise AssertionError(f"{name} was published")
            except AbiError:
                pass
    # a newer version than the reader's is refused
    raw = bytearray(blob)
    assert raw[4] == 3
    raw[4] = 4
    assert pf.wire_refuses(pf.reseal(bytes(raw)))
