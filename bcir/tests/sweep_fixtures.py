"""Fixtures of the re-selection sweep (G2 / S2-A), shared by the tests and the harness.

`sweep_fixture` is the report's section 6.3 shape: one phase of `count` conflicting unit-
stride claims on one resource -- a serial hazard chain, no step-shortening alternative
anywhere, so the sweep's cost there is its fixed overhead over the serial pass.

`general_fixture` is the general case the roadmap names for G2: many step-shortening
trials. Pairs of claims share a read operand under the ENERGY policy: a small claim
(64 elements) whose scalar realization is cheaper for its own step, followed by a large
one (4,096 elements) realized vec8. The serial optimum picks vec8 for the small claim
because the successor's locality discount outweighs its own extra cost, so the small
claim's scalar alternative shortens its own step and the sweep must place it -- one
trial per pair -- while lengthening the successor (the discount is lost), so no trial is
adopted: the plan is the serial optimum, and every trial is a full re-placement without
delta pricing.

`adoption_fixture` is the smallest module on which a trial IS adopted: a small claim
whose vector realization discounts its large, independent successor, and a long claim
that depends on the small one -- the serial optimum takes the discount, but the small
claim sits on the critical chain, and its scalar realization shortens that chain by
more than the large claim grows on its own stream.

`random_module` generates hazard-bearing multi-phase modules (unit, strided, random and
cacheline claims, occasional fences) for the differential tests.
"""

from __future__ import annotations

import random

from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass


def sweep_fixture(count: int) -> Module:
    module = Module(name=f"sweep{count}")
    module.add_resource(Resource(rid=1, shape=(64,)))
    module.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(
                    id=index + 1,
                    opcode=Opcode.ADD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=64,
                    rd=(1,),
                    wr=(1,),
                    op="vector.add",
                )
                for index in range(count)
            ],
        )
    )
    return module


def _pair(module: Module, cid: int, rid: int) -> tuple[Claim, Claim]:
    a, b, c, d, e = rid, rid + 1, rid + 2, rid + 3, rid + 4
    for r, n in ((a, 4096), (b, 64), (c, 64), (d, 4096), (e, 4096)):
        module.add_resource(Resource(rid=r, shape=(n,)))
    small = Claim(
        id=cid,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=64,
        rd=(a, b),
        wr=(c,),
        op="vector.add",
    )
    large = Claim(
        id=cid + 1,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=4096,
        rd=(a, d),
        wr=(e,),
        op="vector.add",
    )
    return small, large


def general_fixture(phases: int, pairs: int) -> Module:
    """`phases` phases of `pairs` (small, large) pairs on disjoint resources: 2 x phases x
    pairs claims, one step-shortening trial per pair under ENERGY."""
    module = Module(name=f"general{phases}x{pairs}")
    rid = cid = 1
    for phase_id in range(phases):
        claims: list[Claim] = []
        for _ in range(pairs):
            claims.extend(_pair(module, cid, rid))
            cid += 2
            rid += 5
        module.add_phase(Phase(phase_id=phase_id, claims=claims))
    return module


def adoption_fixture(dependent: int = 5000) -> Module:
    """One pair plus a claim that depends on the small one (it reads what the small claim
    writes) and is long enough to be the critical path: the serial optimum's vec8 small
    claim (the discount on the large one) sits on that chain, the scalar one shortens it
    by more than the large claim grows on its own stream."""
    module = Module(name="adoption")
    small, large = _pair(module, 1, 1)
    for r in (10, 11):
        module.add_resource(Resource(rid=r, shape=(dependent,)))
    chain = Claim(
        id=3,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=dependent,
        rd=(small.wr[0], 10),
        wr=(11,),
        op="vector.add",
    )
    module.add_phase(Phase(phase_id=0, claims=[small, large, chain]))
    return module


def random_module(seed: int) -> Module:
    rng = random.Random(seed)
    module = Module(name=f"rand{seed}")
    rids = list(range(1, rng.randint(3, 12)))
    for r in rids:
        module.add_resource(Resource(rid=r, shape=(rng.choice([64, 256, 1024, 4096]),)))
    cid = 1
    for phase_id in range(rng.randint(1, 5)):
        claims = []
        for _ in range(rng.randint(1, 12)):
            sc = rng.choice(
                [
                    StrideClass.UNIT,
                    StrideClass.UNIT,
                    StrideClass.STRIDED,
                    StrideClass.RANDOM,
                    StrideClass.CACHELINE,
                ]
            )
            lane = Lane.GGG if sc == StrideClass.RANDOM else Lane.U
            rd = tuple(rng.sample(rids, rng.randint(1, min(3, len(rids)))))
            wr = (rng.choice(rids),)
            contract = {"hazard": "barriered"} if rng.random() < 0.1 else {}
            claims.append(
                Claim(
                    id=cid,
                    opcode=rng.choice([Opcode.ADD, Opcode.MUL]),
                    lane=lane,
                    stride_class=sc,
                    count=rng.choice([64, 1024, 4096]),
                    rd=rd,
                    wr=wr,
                    op="vector.add",
                    **contract,
                )
            )
            cid += 1
        module.add_phase(Phase(phase_id=phase_id, claims=claims))
    return module


__all__ = ["adoption_fixture", "general_fixture", "random_module", "sweep_fixture"]
