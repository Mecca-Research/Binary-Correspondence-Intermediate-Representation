"""PX-3b-2: measured prices inside K_BCIR's own candidates.

A target may carry a frozen, generation-tagged table of measured alternatives
(`cost.MeasuredPrices`, `TargetProfile.measured`): a claim whose (op, count) it names is offered,
after each of its own realizations, that realization run each named alternative -- the same lane,
width and operands, the name `<name>@<alternative>`, the base after the intra-phase discount with
its compute and memory axes scaled by the entry's Q8 (`realize._offer_rows`). The planner prices
them like any candidate; R9 re-derives them from the same offer; the target's content address
folds the table; the planner's records carry it (BKPI and BKPR version 1) and the law rail's
capability does (`measured_*`). No table is today's offer, plan, hash, record and text exactly.

The witnesses hold, without a compiler: the table's one spelling and its refusals; the offer's
measured rows, their order, names and prices, `only` naming them, control ops never measured;
non-disturbance; the plan taking a cheaper measured realization and never pricing above the
plan without the table; R9 accepting the planner's own measured plan only under its table and
refusing a forged alternative or price; the hash; the records' version 1, their one spelling and
the encoder leaving out what no claim reads; the emitted capability. The native planner and
hydrate are held to all of it byte for byte by `planner.measured.parity` (test_planner_compact).
"""

from dataclasses import replace

from bcir.abi import planner_abi as pa
from bcir.kbcir import realize
from bcir.kbcir.cost import (
    MeasuredEntry,
    MeasuredPrices,
    TargetProfile,
    Theta,
    measured_alternative_key,
    spelled_as_opcode,
)
from bcir.kbcir.provenance import hash_target
from bcir.model import Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.tests import planner_fixtures as pf
from bcir.verify import verify_plan


def _expect_refused(label, fn, reason, refusal=ValueError):
    try:
        fn()
    except refusal as exc:
        assert reason in str(exc), (label, str(exc))
        return
    raise AssertionError(f"{label} admitted")


def _module():
    """Two phases: a unit-stride add, a tile multiply-accumulate, a barrier and an op-less claim."""
    m = Module(name="measured")
    for rid in (1, 2, 3):
        m.add_resource(Resource(rid=rid, shape=(64,)))
    m.add_phase(Phase(0, (), [
        pf._claim(1, Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=64, rd=(1,),
                  wr=(2,), op="k.add"),
        pf._claim(2, Opcode.T_MACC, lane=Lane.T, stride_class=StrideClass.TILE, count=4096,
                  rd=(1, 2), wr=(3,), op="k.mm"),
    ]))  # fmt: skip
    m.add_phase(Phase(1, (0,), [
        pf._claim(3, Opcode.BARRIER, lane=Lane.H, stride_class=StrideClass.SCALAR, count=1,
                  rd=(), wr=(), op="k.check"),
        pf._claim(4, Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=64, rd=(3,),
                  wr=(2,), op=""),
    ]))  # fmt: skip
    return m


def _table(*entries):
    rows = sorted((MeasuredEntry(*e) for e in entries),
                  key=lambda e: (e.op, e.count, measured_alternative_key(e.alternative)))  # fmt: skip
    return MeasuredPrices(1, tuple(rows))


def test_a_measured_table_has_one_spelling_and_refuses_the_rest():
    """A table is a positive generation and at least one entry, each a non-empty op of at most
    65,535 UTF-8 bytes never spelled as an opcode, a count of 1 .. 2**32 - 1, an alternative
    `i<bit>` (1, 2, 4, 8) or `t<k>` (1 .. 32767, ASCII, no leading zero) and a price of 1 .. 255,
    strictly ascending by (op, count, alternative), at most seven sharing an op and count -- every
    other spelling refused by its own rule; `lookup` returns a key's entries in order and
    `most_per_key` the most one key holds."""
    t = _table(("k.mm", 4096, "t2", 200), ("k.mm", 4096, "i1", 220), ("k.mm", 4096, "t10", 230),
               ("k.add", 64, "t1", 128))  # fmt: skip
    assert [e.alternative for e in t.lookup("k.mm", 4096)] == ["i1", "t2", "t10"]
    assert t.lookup("k.mm", 64) == () and t.lookup("nope", 4096) == ()
    assert t.most_per_key == 3
    assert [measured_alternative_key(a) for a in ("i1", "i8", "t1", "t32767")] == [
        (0, 1), (0, 8), (1, 1), (1, 32767)]  # fmt: skip
    entry = MeasuredEntry("k.mm", 4096, "i1", 200)
    for label, args, reason in (
        ("generation zero", (0, (entry,)), "generation is a positive"),
        ("generation a bool", (True, (entry,)), "generation is a positive"),
        ("no entries", (1, ()), "at least one entry"),
        ("not an entry", (1, (("k.mm", 4096, "i1", 200),)), "holds MeasuredEntry values"),
        ("an empty op", (1, (replace(entry, op=""),)), "non-empty string"),
        ("an op too long", (1, (replace(entry, op="x" * 65536),)), "non-empty string"),
        ("an opcode's spelling", (1, (replace(entry, op="T_MACC"),)), "spelled as an opcode"),
        ("an opcode's shape", (1, (replace(entry, op="GEMM2"),)), "spelled as an opcode"),
        ("a count of zero", (1, (replace(entry, count=0),)), "count is 1 .. 2**32 - 1"),
        ("a count past u32", (1, (replace(entry, count=1 << 32),)), "count is 1 .. 2**32 - 1"),
        ("two bits", (1, (replace(entry, alternative="i3"),)), "names no instance"),
        ("no workers", (1, (replace(entry, alternative="t0"),)), "is i<bit> or t<workers>"),
        ("a leading zero", (1, (replace(entry, alternative="t01"),)), "is i<bit> or t<workers>"),
        ("non-ASCII digits", (1, (replace(entry, alternative="t١"),)), "i<bit> or t<work"),
        ("too many workers", (1, (replace(entry, alternative="t32768"),)), "more than 32767"),
        ("another letter", (1, (replace(entry, alternative="w1"),)), "is i<bit> or t<workers>"),
        ("a price of zero", (1, (replace(entry, q8=0),)), "price is 1 .. 255"),
        ("the default's price", (1, (replace(entry, q8=256),)), "price is 1 .. 255"),
        ("out of order", (1, (entry, replace(entry, alternative="t1", q8=100),
                              replace(entry, alternative="i2"))), "strictly ascending"),
        ("twice", (1, (entry, entry)), "strictly ascending"),
        ("eight on one key", (1, tuple(replace(entry, alternative=f"t{k}") for k in range(1, 9))),
         "more than 7 measured entries"),
    ):  # fmt: skip
        _expect_refused(label, lambda a=args: MeasuredPrices(*a), reason)
    seven = tuple(replace(entry, alternative=f"t{k}") for k in range(1, 8))
    assert MeasuredPrices(1, seven).most_per_key == 7  # the bound itself is a table
    assert all(spelled_as_opcode(o.name) for o in Opcode)  # the law rail's op-less spelling
    for op in ("k.mm", "gemm", "Attn", "_X", "A-B", "A.B", "\u00c4", "A\u00c4", "qkv.fwd"):
        assert not spelled_as_opcode(op), op
        MeasuredPrices(1, (replace(entry, op=op),))  # admitted
    _expect_refused("a profile's table not a table",
                    lambda: replace(TargetProfile.x86_avx512(), measured={"k.mm": 1}),
                    "MeasuredPrices table or None")  # fmt: skip


def test_the_offer_appends_each_realization_run_each_alternative():
    """A claim the table names is offered its own realizations, then each of them run each entry
    in the order (realization, then entry): the realization's lane, width and operands, the name
    `<name>@<alternative>`, the base with compute and memory scaled by q8/256 (floored) and every
    other axis kept; `only` names a measured realization as any; a barrier and an op-less claim
    are never measured; without a table, or an entry, the offer is the claim's own exactly."""
    m = _module()
    h0 = TargetProfile.x86_avx512()
    table = _table(("k.add", 64, "i1", 200), ("k.add", 64, "t2", 100), ("k.mm", 4096, "i4", 128),
                   ("k.check", 1, "t1", 50))  # fmt: skip
    h1 = h0.with_measured(table)
    plain, measured = realize.fused_offer(m, h0), realize.fused_offer(m, h1)
    for j, claim in enumerate(plain.claims):
        own = plain.rows[j]
        rows = measured.rows[j]
        entries = table.lookup(claim.op, claim.count)
        if claim.opcode in realize.UNMEASURED_OPCODES or not entries:
            assert rows == own, claim.id  # the barrier ("k.check" is named) and the op-less add
            continue
        assert rows[: len(own)] == own
        want = [(lane, w, f"{name}@{e.alternative}",
                 ((b[0] * e.q8) >> 8, (b[1] * e.q8) >> 8) + b[2:], rw)
                for lane, w, name, b, rw in own for e in entries]  # fmt: skip
        assert rows[len(own) :] == want, claim.id
        for row in rows[len(own) :]:  # strictly cheaper on compute and memory where nonzero
            base = next(b for _, w, n, b, _ in own if w == row[1] and row[2].startswith(n + "@"))
            assert all(row[3][a] < base[a] for a in (0, 1) if base[a]), row
    assert {r[2] for r in measured.rows[0]} >= {"vec16@i1", "vec16@t2", "scalar@t2"}
    # R9's single-candidate offer names a measured realization as it names any
    only = {2: {(Lane.T, 16, "tile@i4")}}
    picked = realize.fused_offer(m, h1, only=only)
    assert [r[2] for r in picked.rows[1]] == ["tile@i4"]
    assert realize.fused_offer(m, h1, only={2: {(Lane.T, 16, "tile@i8")}}).rows[1] == []
    # a table naming nothing a claim has: the offer as without one
    other = h0.with_measured(_table(("k.mm", 4097, "i1", 10)))
    assert realize.fused_offer(m, other).rows == plain.rows


def test_the_plan_takes_a_cheaper_measured_realization_and_r9_holds_it():
    """The planner takes a measured realization where it is cheaper, never prices above the
    plan without the table, holds every row of a claim priced at the table's most entries a key,
    and R9 accepts its own plan under the table while refusing it without the table, and refusing
    a forged alternative or a forged price under it."""
    m = _module()
    h0 = TargetProfile.x86_avx512()
    h1 = h0.with_measured(_table(("k.mm", 4096, "i4", 128), ("k.add", 64, "t2", 64)))
    theta = Theta.cool()
    plain, plan = realize.optimize(m, h0, theta), realize.optimize(m, h1, theta)
    names = {s.claim_id: s.candidate.name for s in plan.steps}
    assert names[2] == "tile@i4" and names[1].endswith("@t2"), names
    assert plan.score < plain.score
    # every row held: a module whose one claim is priced at the table's most entries a key leaves
    # the planner no slack -- its slots hold each realization and each of it run each entry
    solo = Module(name="solo")
    for rid in (1, 2):
        solo.add_resource(Resource(rid=rid, shape=(64,)))
    solo.add_phase(Phase(0, (), [pf._claim(1, Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
                                           count=64, rd=(1,), wr=(2,), op="k.add")]))  # fmt: skip
    priced = h0.with_measured(_table(("k.add", 64, "i1", 200), ("k.add", 64, "t2", 64)))
    rows = realize.fused_offer(solo, priced).rows[0]
    assert len(rows) == 3 * len(realize.fused_offer(solo, h0).rows[0])
    assert [s.candidate.name for s in realize.optimize(solo, priced, theta).steps] == ["vec16@t2"]
    assert not [d for d in verify_plan(m, plan, h1, theta=theta) if d.law in ("R8", "R9")]
    refused = sorted(d.message for d in verify_plan(m, plan, h0, theta=theta) if d.law == "R9")
    assert len(refused) == 2, refused  # both measured steps, each named, nothing else
    assert "'vec16@t2'" in refused[0] and "'tile@i4'" in refused[1], refused
    for label, edit in (
        ("a forged alternative", lambda c: replace(c, name="tile@i8")),
        ("a forged price", lambda c: replace(c, base=type(c.base)(
            (c.base.v[0] + 1,) + c.base.v[1:]))),
    ):  # fmt: skip
        steps = [replace(s, candidate=edit(s.candidate)) if s.claim_id == 2 else s
                 for s in plan.steps]  # fmt: skip
        forged = replace(plan, steps=steps)
        diags = [d for d in verify_plan(m, forged, h1, theta=theta) if d.law == "R9"]
        assert diags, label


def test_no_table_is_every_plan_record_and_hash_as_before():
    """Non-disturbance: a target without a table plans, encodes, hashes and emits as before --
    every fixed-corpus module's version-zero BKPI and BKPR records are the encoder's without the
    field read at all, its hash the profile's, and a table that names no claim of a module leaves
    its BKPI record version zero and its plan the plain one."""
    h0 = TargetProfile.x86_avx512()
    assert h0.measured is None and h0 == replace(h0, measured=None)
    for case in pf.corpus_cases()[::211]:
        data = pa.encode_input(case.module, case.h, case.theta, case.policy)
        assert data[4] == pa.VERSION and not pa.decode_input(data).measured
        status, plan = pf.realization_bytes(
            realize.optimize(case.module, case.h, case.theta, case.policy))  # fmt: skip
        assert status != "BCIR_OK" or plan[4] == pa.VERSION
    m = _module()
    unread = h0.with_measured(_table(("k.mm", 1, "i1", 1), ("k.check", 1, "t1", 1)))
    assert pa.encode_input(m, unread, Theta.cool()) == pa.encode_input(m, h0, Theta.cool())
    assert realize.optimize(m, unread, Theta.cool()).score == realize.optimize(
        m, h0, Theta.cool()).score  # fmt: skip


def test_the_hash_folds_every_field_of_the_table():
    """The target's content address (R13) is the profile's exactly without a table, and moves with
    the table's generation and with every entry's op, count, alternative and price."""
    h = TargetProfile.x86_avx512()
    base = MeasuredEntry("k.mm", 4096, "i4", 128)
    seen = {hash_target(h), hash_target(h.with_measured(MeasuredPrices(1, (base,))))}
    for edited in (MeasuredPrices(2, (base,)), MeasuredPrices(1, (replace(base, op="k.mn"),)),
                   MeasuredPrices(1, (replace(base, count=4097),)),
                   MeasuredPrices(1, (replace(base, alternative="i8"),)),
                   MeasuredPrices(1, (replace(base, q8=129),))):  # fmt: skip
        seen.add(hash_target(h.with_measured(edited)))
    assert len(seen) == 7


def test_the_records_carry_the_table_at_version_one():
    """BKPI is version 1 exactly when an entry some claim reads is encoded -- `n_measured` at @40,
    the entries after the op bytes as (op index, count, code, q8), the unread ones (another count,
    an unmeasured opcode) left out -- and BKPR exactly when a step runs an alternative, its code
    in the step's reserved u16; both decode to what was encoded, one spelling, and the codes are
    an instance's bit or 0x8000 | k."""
    m = _module()
    table = _table(("k.mm", 4096, "i4", 128), ("k.add", 64, "t2", 64), ("k.add", 64, "t32767", 90),
                   ("k.mm", 7, "i1", 1), ("k.check", 1, "t1", 50))  # fmt: skip
    h = TargetProfile.x86_avx512().with_measured(table)
    data = pa.encode_input(m, h, Theta.cool())
    v = pa.decode_input(data)
    pa.check_input(v)
    assert data[4] == pa.VERSION_MEASURED and pa._pack_input(v) == data
    ops = {name.decode(): i for i, name in enumerate(v.ops)}
    assert v.measured == ((ops["k.add"], 64, 0x8002, 64), (ops["k.add"], 64, 0xFFFF, 90),
                          (ops["k.mm"], 4096, 4, 128))  # fmt: skip
    assert [pa.measured_code(a) for a in ("i1", "i2", "i4", "i8", "t1", "t32767")] == [
        1, 2, 4, 8, 0x8001, 0xFFFF]  # fmt: skip
    assert [pa.measured_alternative(c) for c in (1, 8, 0x8001, 0xFFFF)] == [
        "i1", "i8", "t1", "t32767"]  # fmt: skip
    for code in (0, 3, 16, 0x8000):
        _expect_refused(f"code {code:#x}", lambda c=code: pa.measured_alternative(c), "names no",
                        pa.PlannerAbiError)  # fmt: skip
    plan = realize.optimize(m, h, Theta.cool())
    record = pa.encode_realization(plan)
    real = pa.decode_realization(record)
    assert record[4] == pa.VERSION_MEASURED
    by_claim = {s.claim_id: s.alternative for s in real.steps}
    assert by_claim[2] == 4 and by_claim[1] in (0x8002, 0xFFFF) and by_claim[4] == 0


def test_the_law_rail_capability_carries_the_table():
    """`to_mlir` writes the table on `bcir.target.capability` -- its generation, ops,
    alternatives and (count, q8) values in the table's order -- and nothing at all without one,
    so a module emitted without a table is the text it always was."""
    from bcir.lower.mlir import to_mlir

    m = _module()
    h0 = TargetProfile.x86_avx512()
    table = _table(("k.mm", 4096, "i4", 128), ("k.add", 64, "t2", 64))
    plain = to_mlir(m, h0, Theta.cool(), run_pipeline=False)
    text = to_mlir(m, h0.with_measured(table), Theta.cool(), run_pipeline=False)
    assert "measured_" not in plain
    want = ('measured_gen = 1 : i64, measured_ops = ["k.add", "k.mm"], '
            'measured_alternatives = ["t2", "i4"], measured_values = array<i64: 64, 64, 4096, 128>')  # fmt: skip
    assert want in text, text
    cap = next(line for line in text.splitlines() if "bcir.target.capability" in line)
    assert cap.replace(", " + want, "") in plain.splitlines()  # all the capability gained
    # a measured realization's path symbol is not a bare identifier: it is written quoted, the
    # spelling bcir-opt prints it in, and the plan selects it by that spelling
    assert 'bcir.kbcir.path @"path_2_tile@i4" {' in text and "@path_2_tile@" not in text
    assert 'selected = @"path_2_tile@i4"' in text
