"""The G18 fixtures and grader (S4-B): the K_BCIR -> StreamPack chain advanced by declared deltas,
held to the chain run from scratch on the module each delta declares.

One function, `measure`, grades the exact rows over the declared corpora. The tests, the harness
(`tools/perf/gemplus_baseline.py --group delta`) and `tools/perf/check_delta.py` all call it:

    planner.delta.parity      (case, step) pairs where the chain's plan (`kbcir.delta`, advanced by
                              `gem.delta_chain.DeltaChain`) differs from `optimize` of the declared
                              module -- the steps, their costs, the score, the BKPR bytes or refusal
                              -- or the chain's module differs from the declared one
    pack.delta.identity       (case, step) pairs where the delta StreamPack (`gem.delta_pack`)
                              differs from `encode(hydrate_pipelined(...))` of the declared module
                              and its plan: the pack, its bytes, or the refusal (the same error with
                              the same message)
    verify.delta.identity     (case, step) pairs where the incremental verdict (`verify.delta`)
                              differs from the full verdict -- over the chain's own links, and over
                              a second rail that hands the verifier each step's plan and pack with
                              one field forged (`FORGERIES`)
    delta.malformed.accepted  (malformed input, rail) pairs not refused with `DeltaError` before
                              anything moved: every delta the chain does not admit on
                              `apply_delta`, `IncrementalPlan.apply` and `DeltaChain.apply`, a module
                              that declares a claim id twice on every `build`, and a module changed
                              outside a delta; a refusal after which the next honest delta does not
                              reproduce the chain from scratch counts too

A case is a module under one (target, Theta, policy, pipelining depth): its build is one step and
each of its `ROUNDS` deltas another. The rounds are drawn per case from the edit families, in
rotation so that every family meets every kind of module:

  * `CLAIM_EDITS` -- one replacement moving one field of a claim, every field the chain reads
    (a law violation among the values of each: an undeclared operand, an unknown contract
    mnemonic, an illegal lane, an isolated domain, a mask the event law refuses);
  * `CONE_EDITS` -- replacements aimed at the dependency cones the incremental offer re-derives:
    a read shared with the neighbour column or dropped, a write feeding a later reader or
    cleared, a claim made a value-numbered copy of an earlier one or broken out of one, the
    fence raised or lowered, two adjacent claims, the first and the last claim together;
  * `RESOURCE_EDITS` -- one registry replacement moving one field of a resource (tier, access,
    shape, alignment, the generation vector);
  * a random mixture of the above;
  * `UNEMITTABLE` -- a value the StreamPack wire cannot carry (a u64 count, offset or stride, a
    u32 RID, a u16 string length, a u32 generation) -- and in the next round its repair, which
    the chain must re-emit in full;
  * `EVENT_EDITS`, over a module with event phases -- every arming claim disarmed (EV2), the mask
    window opened or the handler writing what the consumer holds (EV3), the consumer made a
    Lane.A atomic (EV3's other lawful shape); a random mixture over any other module.

The corpora: the fixed corpus (`examples.PROGRAMS`, the audit's matmul fixture, the planner's
`coverage_modules`, `delta_modules` -- one module per construct a delta moves -- and the U4
event-phase fixture), each under `SCOPES` rotating scopes and the planner's `custom_scopes` on
every third, and a seeded sample of `differential.gen_module`.

Neither the corpus nor its reference needs the G18 mechanisms: `declared` is this module's own
application of a delta and `reference_chain` the chain from scratch, both spelled with entry points
the parent tree has. On the parent every mechanism is absent, so every comparison it owns fails
(RED, docs/security/laws.md L1, L25).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

ROWS = (
    "planner.delta.parity",
    "pack.delta.identity",
    "verify.delta.identity",
    "delta.malformed.accepted",
)

PLAN = "plan0"  # the StreamPack's source plan name, as `DeltaChain` defaults it
ROUNDS = 7  # deltas per case, after its build
SCOPES = 3  # rotating scopes per fixed module
GENERATED = 96  # modules drawn from differential.gen_module
GENERATED_SEED = 18
CASE_SEED = 1818
UNDECLARED = 9_999  # a RID no corpus module declares

CLAIM_EDITS = (
    "count",
    "lane",
    "stride",
    "rd",
    "wr",
    "op",
    "mask",
    "hazard",
    "opcode",
    "offset",
    "stride_k",
    "imm",
    "verify",
    "bounds",
    "domain",
    "cost_class",
    "callee",
    "volatile",
    "primary",
    "dynamic",
    "numeric",
    "timing",
    "lifetime",
    "provenance",
    "same",
    "self",
)
CONE_EDITS = (
    "rd.neighbour",
    "rd.clear",
    "wr.reader",
    "wr.clear",
    "sig.copy",
    "sig.break",
    "fence.raise",
    "fence.lower",
    "adjacent",
    "ends",
    "empty",
)
RESOURCE_EDITS = (
    "domain",
    "access",
    "shape",
    "align",
    "gen",
    "elem_bytes",
    "layout",
    "priority",
    "name",
    "same",
)
UNEMITTABLE = ("count", "offset", "stride_k", "rid", "wr.rid", "op", "gen")
EVENT_EDITS = ("disarm", "unmask.window", "atomic.consumer", "handler.write")
PLAN_FORGERIES = (
    "plan.name",
    "plan.cost",
    "plan.phase",
    "plan.width",
    "plan.lane",
    "plan.score",
    "plan.claim",
    "plan.drop",
    "plan.stale",
)
PACK_FORGERIES = (
    "pack.read",
    "pack.prefetch",
    "pack.target",
    "pack.claim",
    "pack.note",
    "pack.generation",
    "pack.drop",
    "pack.stale",
)
FORGERIES = ("none",) + PLAN_FORGERIES + PACK_FORGERIES


@dataclass
class Case:
    label: str
    module: object
    h: object
    theta: object
    policy: object
    depth: int
    index: int  # the case's place in the corpus: it picks the rotating families and the seed


# --- the reference: a delta applied, and the chain from scratch --------------------------------


def declared(module, claims=(), resources=()):
    """The module a delta declares, built here without the G18 mechanisms: a new `Module` in
    which each claim whose id a replacement names, and each resource whose RID one names, is that
    replacement; every other claim, phase and resource is shared, and `module` is unchanged."""
    by_id = {c.id: c for c in claims}
    phases = []
    for ph in module.phases:
        if any(c.id in by_id for c in ph.claims):
            ph = replace(ph, claims=[by_id.get(c.id, c) for c in ph.claims])
        phases.append(ph)
    registry = dict(module.resources)
    for r in resources:
        registry[r.rid] = r
    return replace(module, resources=registry, phases=phases)


@dataclass
class Reference:
    """The chain from scratch: the plan, the pack, its bytes or the wire's refusal, and the
    verdict (None when the wire refused: the chain stops at the bytes)."""

    result: object
    pack: object
    data: bytes | None
    refusal: Exception | None
    module_laws: list
    diagnostics: list | None


def reference_chain(module, h, theta, policy, depth: int, plan: str = PLAN) -> Reference:
    """plan -> pack -> bytes -> verdict, from scratch (`gem.delta_chain.full_chain`, spelled
    with the entry points the parent tree has)."""
    from bcir.abi.streampack_abi import AbiError, encode
    from bcir.gem.streampack import hydrate_pipelined
    from bcir.kbcir.realize import optimize
    from bcir.verify import verify, verify_pack, verify_plan

    result = optimize(module, h, theta, policy)
    pack = hydrate_pipelined(module, result, plan, depth)
    module_laws = verify(module)
    try:
        data = encode(pack)
    except AbiError as exc:
        return Reference(result, pack, None, exc, module_laws, None)
    diags = (
        module_laws
        + verify_plan(module, result, h, theta=theta, policy=policy)
        + verify_pack(module, pack)
    )
    return Reference(result, pack, data, None, module_laws, diags)


def full_verdict(module_laws, module, result, pack, h, theta, policy) -> list:
    """The chain's verdict over a (plan, pack) pair -- the forgery rail's reference."""
    from bcir.verify import verify_pack, verify_plan

    return (
        module_laws
        + verify_plan(module, result, h, theta=theta, policy=policy)
        + verify_pack(module, pack)
    )


def _steps(result) -> list:
    return [(s.claim_id, s.phase_id, s.candidate, s.cost) for s in result.steps]


def _refusal(exc) -> tuple:
    return (type(exc).__name__, str(exc))


# --- the corpus -------------------------------------------------------------------------------


def _claim(cid, opcode=None, **kw):
    from bcir.model import Claim, Opcode

    return Claim(id=cid, opcode=Opcode.ADD if opcode is None else opcode, **kw)


def _module(name, resources, phases):
    from bcir.model import Module

    m = Module(name=name)
    for r in resources:
        m.add_resource(r)
    for ph in phases:
        m.add_phase(ph)
    return m


def delta_modules() -> list[tuple[str, object]]:
    """One module per construct a delta moves that the rest of the corpus holds only by
    accident: a value-numbered identity with copies before and after a rewrite of its read, a
    deforested producer, a fence, a pipelined chain of phases with a transition that prefetches
    one operand and one that prefetches none, a generation vector at nonzero generations, an
    operand in every memory tier across phases, a phase of one claim, and a registry with no
    claims at all."""
    from bcir.model import Lane, Opcode, Phase, Resource, StrideClass
    from bcir.model.lanes import Domain

    def res(*rids, shape=(256,), **kw):
        return [Resource(rid=r, shape=shape, **kw) for r in rids]

    out: list[tuple[str, object]] = []
    out.append(
        (
            "delta.identity",
            _module(
                "delta.identity",
                res(1, 2, 3, 4, 5, 6),
                [
                    Phase(
                        0,
                        (),
                        [
                            _claim(1, rd=(1, 2), wr=(3,), op="vector.add", count=256),
                            _claim(2, rd=(1, 2), wr=(4,), op="vector.add", count=256),
                            _claim(3, Opcode.MUL, rd=(3,), wr=(5,), op="vector.mul", count=256),
                            _claim(4, rd=(5, 4), wr=(1,), op="vector.add", count=256),
                            _claim(5, rd=(1, 2), wr=(6,), op="vector.add", count=256),
                            _claim(6, rd=(1, 2), wr=(4,), op="vector.add", count=256),
                        ],
                    )
                ],
            ),
        )
    )
    out.append(
        (
            "delta.fence",
            _module(
                "delta.fence",
                res(1, 2, 3, 4, 5, shape=(64,)),
                [
                    Phase(
                        0,
                        (),
                        [
                            _claim(
                                11, rd=(1,), wr=(2,), hazard="barriered", op="stage.a", count=64
                            ),
                            _claim(12, rd=(2,), wr=(3,), op="stage.b", count=64),
                            _claim(13, rd=(3,), wr=(4,), op="stage.c", count=64),
                            _claim(14, rd=(4, 2), wr=(5,), op="stage.d", count=64),
                        ],
                    )
                ],
            ),
        )
    )
    out.append(
        (
            "delta.pipeline",
            _module(
                "delta.pipeline",
                res(1, 2, 3, 4, 5, 6, 7, 8, shape=(128,)),
                [
                    Phase(
                        0,
                        (),
                        [
                            _claim(21, rd=(1,), wr=(2,), op="p0.a", count=128),
                            _claim(22, rd=(1, 3), wr=(4,), op="p0.b", count=128),
                        ],
                    ),
                    Phase(
                        1,
                        (0,),
                        [
                            _claim(23, rd=(2,), wr=(5,), op="p1.a", count=128),
                            _claim(24, rd=(4, 2), wr=(6,), op="p1.b", count=128),
                        ],
                    ),
                    Phase(2, (1,), [_claim(25, rd=(6,), wr=(7,), op="p2.a", count=128)]),
                    Phase(
                        3,
                        (2,),
                        [
                            _claim(
                                26,
                                Opcode.STORE,
                                stride_class=StrideClass.SCALAR,
                                wr=(8,),
                                op="p3.flush",
                                primary_rid=8,
                            )
                        ],
                    ),
                ],
            ),
        )
    )
    out.append(
        (
            "delta.generations",
            _module(
                "delta.generations",
                [
                    Resource(rid=1, shape=(64,), map_gen=3, data_gen=5),
                    Resource(rid=2, shape=(64,), map_gen=7, data_gen=1),
                    Resource(rid=3, shape=(64,), map_gen=0, data_gen=9),
                ],
                [
                    Phase(
                        0,
                        (),
                        [
                            _claim(31, rd=(1,), wr=(2,), op="g.a", count=64),
                            _claim(32, rd=(2, 1), wr=(3,), op="g.b", count=64),
                        ],
                    )
                ],
            ),
        )
    )
    tiers = [Domain.RAM, Domain.HBM, Domain.CXL, Domain.VRAM, Domain.NVM]
    out.append(
        (
            "delta.tiers",
            _module(
                "delta.tiers",
                [Resource(rid=40 + i, domain=d, shape=(512,)) for i, d in enumerate(tiers)]
                + [Resource(rid=49, domain=Domain.MMIO, shape=(1,))],
                [
                    Phase(
                        0,
                        (),
                        [
                            _claim(41, rd=(40, 41), wr=(42,), op="t.a", count=512),
                            _claim(42, rd=(41,), wr=(43,), op="t.b", count=512),
                        ],
                    ),
                    Phase(
                        1,
                        (0,),
                        [
                            _claim(43, rd=(42, 43), wr=(44,), op="t.c", count=512),
                            _claim(
                                44,
                                Opcode.LOAD,
                                stride_class=StrideClass.SCALAR,
                                rd=(49,),
                                op="t.status",
                                domain=Domain.MMIO,
                                hazard="barriered",
                                volatile=True,
                            ),
                            _claim(
                                45,
                                Opcode.GGG_LOAD,
                                lane=Lane.GGG,
                                stride_class=StrideClass.RANDOM,
                                rd=(40,),
                                wr=(41,),
                                op="t.scatter",
                                count=512,
                            ),
                        ],
                    ),
                ],
            ),
        )
    )
    out.append(
        (
            "delta.single",
            _module(
                "delta.single",
                res(1, 2, shape=(8,)),
                [Phase(0, (), [_claim(51, rd=(1,), wr=(2,), op="one", count=8)])],
            ),
        )
    )
    out.append(("delta.no-claims", _module("delta.no-claims", res(1, 2, shape=(8,)), [])))
    return out


def event_module(masked: bool = True):
    """The U4 UART receive path (driver roadmap A1/B1): a handler phase triggered by `uart_rx`
    filling a ring, and a program-side consumer draining it -- inside a mask window when
    `masked`, bare (an EV3 violation) otherwise. The verdict re-derives the event laws over the
    whole flow, so a delta to it is re-verified from scratch (`VerifyState.rebuilds`)."""
    from bcir.kbcir.events import mask_claim, unmask_claim
    from bcir.model import Lane, Opcode, Phase, Resource, StrideClass
    from bcir.model.lanes import Domain

    ier, iir, rbr, ring, head, tail = 1, 2, 3, 4, 5, 6
    consume = _claim(
        3,
        Opcode.LOAD,
        lane=Lane.U,
        stride_class=StrideClass.SCALAR,
        rd=(ring, head),
        wr=(tail,),
        op="uart.consume",
    )
    body = (
        [mask_claim("uart_rx", ier, cid=2), consume, unmask_claim("uart_rx", ier, cid=4)]
        if masked
        else [consume]
    )
    handler = [
        _claim(
            100,
            Opcode.LOAD,
            stride_class=StrideClass.SCALAR,
            rd=(iir,),
            op="uart.iir_dispatch",
            domain=Domain.MMIO,
            hazard="barriered",
            volatile=True,
        ),
        _claim(
            101,
            Opcode.LOAD,
            stride_class=StrideClass.SCALAR,
            rd=(rbr,),
            wr=(ring, head),
            op="uart.rx_fill",
            domain=Domain.MMIO,
            hazard="barriered",
            volatile=True,
        ),
    ]
    return _module(
        "uart16550_rx_u4" if masked else "uart16550_rx_u4.bare",
        [
            Resource(rid=ier, domain=Domain.MMIO, shape=(1,), name="IER"),
            Resource(rid=iir, domain=Domain.MMIO, shape=(1,), name="IIR"),
            Resource(rid=rbr, domain=Domain.MMIO, shape=(1,), name="RBR"),
            Resource(rid=ring, shape=(16,), name="RING"),
            Resource(rid=head, shape=(1,), name="HEAD"),
            Resource(rid=tail, shape=(1,), name="TAIL"),
        ],
        [
            Phase(0, (), [unmask_claim("uart_rx", ier, cid=1)]),
            Phase(1, (0,), body),
            Phase(10, (), handler, event="uart_rx"),
        ],
    )


def fixed_modules() -> list[tuple[str, object]]:
    from bcir.examples import PROGRAMS, matmul_tiled
    from bcir.tests import planner_fixtures as pf

    fixed = [(name, build()) for name, build in sorted(PROGRAMS.items())]
    fixed.append(("audit.kbcir-streampack.1", matmul_tiled(n=32, tile=8)))
    # 512 claims: a pack of two record chunks (`gem.delta_pack._CHUNK`)
    fixed.append(("audit.kbcir-streampack.2", matmul_tiled(n=64, tile=8)))
    fixed += pf.coverage_modules()
    fixed += delta_modules()
    fixed += [("event.u4", event_module()), ("event.u4.bare", event_module(masked=False))]
    return fixed


def _unique_ids(module) -> bool:
    ids = [c.id for ph in module.phases for c in ph.claims]
    return len(ids) == len(set(ids))


def corpus_cases() -> list[Case]:
    """Every fixed module under `SCOPES` rotating scopes (every target, Theta, policy and depth
    meets every module shape across the corpus) and every third under the planner's custom
    scopes, and the generated sample under one rotating scope each. A module that declares a
    claim id twice is outside what a delta can address; it is the malformed corpus's."""
    from bcir.kbcir.differential import THETAS, gen_module
    from bcir.kbcir.weights import POLICIES
    from bcir.tests import planner_fixtures as pf

    tg = sorted(pf.targets().items())
    thetas = sorted(THETAS.items())
    policies = sorted(POLICIES.items())
    depths = (2, 1, 3)
    cases: list[Case] = []

    def scope(k):
        tn, h = tg[k % len(tg)]
        thn, th = thetas[(k // len(tg)) % len(thetas)]
        pn, pol = policies[(k // 2) % len(policies)]
        depth = depths[k % len(depths)]
        return f"{tn}/{thn}/{pn}/d{depth}", h, th, pol, depth

    fixed = [(name, m) for name, m in fixed_modules() if _unique_ids(m)]
    k = 0
    for i, (name, m) in enumerate(fixed):
        for _ in range(SCOPES):
            label, h, th, pol, depth = scope(k)
            cases.append(Case(f"{name}/{label}", m, h, th, pol, depth, len(cases)))
            k += 1
        if i % 3 == 0:
            for sname, h, th, pol in pf.custom_scopes():
                cases.append(Case(f"{name}/{sname}", m, h, th, pol, 2, len(cases)))
    rng = random.Random(GENERATED_SEED)
    for g in range(GENERATED):
        m = gen_module(rng)
        label, h, th, pol, depth = scope(k)
        cases.append(Case(f"gen{g}/{label}", m, h, th, pol, depth, len(cases)))
        k += 1
    return cases


# --- the edits --------------------------------------------------------------------------------


def _flat(module) -> list:
    return [c for ph in module.phases for c in ph.claims]


def claim_edit(rng, claim, rids, family):
    """One replacement of `claim` moving the field `family` names."""
    from bcir.model import Lane, Opcode, StrideClass
    from bcir.model.graph import Lifetime, Timing
    from bcir.model.lanes import Domain

    if family == "count":
        return replace(claim, count=rng.choice([0, 1, 7, 64, 4096, 10**6]))
    if family == "lane":
        return replace(claim, lane=rng.choice(list(Lane)))
    if family == "stride":
        return replace(claim, stride_class=rng.choice(list(StrideClass)))
    if family == "rd":
        k = rng.randint(0, 3)
        return replace(claim, rd=tuple(rng.choice(rids + [UNDECLARED]) for _ in range(k)))
    if family == "wr":
        k = rng.randint(0, 2)
        return replace(claim, wr=tuple(rng.choice(rids + [UNDECLARED + 1]) for _ in range(k)))
    if family == "op":
        return replace(claim, op=rng.choice(["a", "b", "reduce.sum", "reduce.gather", ""]))
    if family == "mask":
        return replace(
            claim, op=rng.choice(["irq.mask:uart_rx", "irq.unmask:uart_rx", "irq.mask:"])
        )
    if family == "hazard":
        return replace(claim, hazard=rng.choice(["unique", "atomic", "barriered", "bogus"]))
    if family == "opcode":
        return replace(claim, opcode=rng.choice(list(Opcode)))
    if family == "offset":
        return replace(claim, offset=rng.choice([0, 3, 100, 1 << 20]))
    if family == "stride_k":
        return replace(claim, stride_k=rng.choice([0, 1, 4, 64]))
    if family == "imm":
        return replace(claim, imm=rng.choice([(), (1,), (3, -2)]))
    if family == "verify":
        return replace(claim, verify=rng.choice(["none", "bounds", "exact", "hash", "nope"]))
    if family == "bounds":
        return replace(claim, bounds=rng.choice(["strict", "masked", "assumed_safe", "wat"]))
    if family == "domain":
        return replace(claim, domain=rng.choice(list(Domain)))
    if family == "cost_class":
        return replace(claim, cost_class=rng.choice(["latency", "bandwidth", "compute", "x"]))
    if family == "callee":
        return replace(claim, callee_sig=rng.choice(["", "int(int)", "broken"]))
    if family == "volatile":
        return replace(claim, volatile=not claim.volatile)
    if family == "primary":
        return replace(claim, primary_rid=rng.choice(rids + [None, UNDECLARED]))
    if family == "dynamic":
        return replace(claim, dynamic=not claim.dynamic)
    if family == "numeric":
        return replace(
            claim,
            precision=rng.choice(["", "compensated"]),
            tolerance_ulp=rng.choice([0, 4]),
            quantized_bits=rng.choice([0, 4]),
        )
    if family == "timing":
        return replace(
            claim,
            timing=rng.choice([None, Timing(clock_domain="clk1", latency_cycles=3)]),
        )
    if family == "lifetime":
        return replace(
            claim,
            lifetime=rng.choice([None, Lifetime("alloc", 0), Lifetime("free", 1)]),
        )
    if family == "provenance":
        return replace(claim, bounds_provenance=rng.choice(["", "declared_extent", "odd"]))
    if family == "same":
        return replace(claim)  # an equal replacement: a new object, nothing moved
    if family == "self":
        return claim  # the object itself: replaced by what it already is
    raise ValueError(family)


def cone_edit(rng, module, family) -> tuple[tuple, tuple]:
    """Replacements aimed at a dependency cone the incremental offer and DP re-derive."""
    flat = _flat(module)
    n = len(flat)
    if family == "empty" or not n:
        return (), ()
    p = rng.randrange(n)
    c = flat[p]
    prev = flat[p - 1] if p else None
    later = flat[p + 1 :]
    if family == "rd.neighbour":
        return (replace(c, rd=tuple(prev.rd) if prev is not None else ()),), ()
    if family == "rd.clear":
        return (replace(c, rd=()),), ()
    if family == "wr.reader":
        readers = [x for x in later if x.rd]
        return (replace(c, wr=tuple(rng.choice(readers).rd) if readers else ()),), ()
    if family == "wr.clear":
        return (replace(c, wr=()),), ()
    if family == "sig.copy":
        earlier = flat[:p]
        if not earlier:
            return (replace(c),), ()
        src = rng.choice(earlier)
        return (
            replace(
                c,
                op=src.op,
                opcode=src.opcode,
                lane=src.lane,
                stride_class=src.stride_class,
                count=src.count,
                stride_k=src.stride_k,
                offset=src.offset,
                domain=src.domain,
                dynamic=src.dynamic,
                rd=tuple(src.rd),
            ),
        ), ()
    if family == "sig.break":
        return (replace(c, op=f"{c.op}.broken", count=c.count + 1),), ()
    if family == "fence.raise":
        return (replace(c, hazard="barriered"),), ()
    if family == "fence.lower":
        return (replace(c, hazard="unique", volatile=False),), ()
    if family == "adjacent":
        if p + 1 >= n:
            return (replace(c, count=c.count + 1),), ()
        d = flat[p + 1]
        return (replace(c, rd=tuple(d.rd)), replace(d, count=d.count + 3)), ()
    if family == "ends":
        if n == 1:
            return (replace(c, count=c.count + 2),), ()
        a, b = flat[0], flat[-1]
        return (replace(a, count=a.count + 1), replace(b, rd=tuple(a.rd))), ()
    raise ValueError(family)


def resource_edit(rng, resource, family):
    """One replacement of `resource` moving the field `family` names."""
    from bcir.model.lanes import Domain

    if family == "domain":
        return replace(resource, domain=rng.choice(list(Domain)))
    if family == "access":
        return replace(resource, access="ham" if resource.access == "flat" else "flat")
    if family == "shape":
        return replace(resource, shape=rng.choice([(8,), (0,), (4096,), (), (2, 2), (1 << 40,)]))
    if family == "align":
        return replace(resource, align=rng.choice([64, 3, 0, 4096]))
    if family == "gen":
        return replace(
            resource,
            map_gen=resource.map_gen + rng.randint(0, 2),
            data_gen=resource.data_gen + rng.randint(1, 3),
        )
    if family == "elem_bytes":
        return replace(resource, elem_bytes=rng.choice([0, 1, 2, 8]))
    if family == "layout":
        return replace(resource, layout=rng.choice(["soa", "aos"]))
    if family == "priority":
        return replace(resource, priority=rng.choice([0, 3]))
    if family == "name":
        return replace(resource, name=rng.choice(["", "renamed"]))
    if family == "same":
        return replace(resource)
    raise ValueError(family)


def unemittable_edit(rng, module, family) -> tuple[tuple, tuple]:
    """A replacement carrying a value the StreamPack wire cannot (the chain must refuse the pack
    as `encode` does): a u64 count, offset or stride, a u32 read or write RID, a u16 string
    length, a u32 generation. Empty when the module has nothing it applies to."""
    flat = _flat(module)
    if family == "gen":
        if not module.resources:
            return (), ()
        r = module.resources[rng.choice(sorted(module.resources))]
        return (), (replace(r, map_gen=1 << 32),)
    if not flat:
        return (), ()
    c = flat[rng.randrange(len(flat))]
    if family == "count":
        return (replace(c, count=1 << 64),), ()
    if family == "offset":
        return (replace(c, offset=-1),), ()
    if family == "stride_k":
        return (replace(c, stride_k=1 << 64),), ()
    if family == "rid":
        return (replace(c, rd=(1 << 32,)),), ()
    if family == "wr.rid":
        return (replace(c, wr=(1 << 33,)),), ()
    if family == "op":
        return (replace(c, op="x" * 70_000),), ()
    raise ValueError(family)


def round_edit(rng, module, case: Case, r: int, pending) -> tuple[tuple, tuple, str, tuple]:
    """The replacements of round `r` of `case` over `module`: (claims, resources, family, the
    replacements carrying a value the wire refuses -- empty but in round 4)."""
    flat = _flat(module)
    rids = sorted(module.resources)
    k = case.index + r
    if r == 5 and pending is not None:  # the repair of round 4's unemittable value
        claims, resources = pending
        return claims, resources, "repair", ((), ())
    if r == 0:
        family = CLAIM_EDITS[k % len(CLAIM_EDITS)]
        if not flat:
            return (), (), "empty", ((), ())
        c = flat[rng.randrange(len(flat))]
        return (claim_edit(rng, c, rids, family),), (), f"claim.{family}", ((), ())
    if r == 1:
        family = CONE_EDITS[k % len(CONE_EDITS)]
        claims, resources = cone_edit(rng, module, family)
        return claims, resources, f"cone.{family}" if claims or resources else "empty", ((), ())
    if r == 2:
        family = RESOURCE_EDITS[k % len(RESOURCE_EDITS)]
        if not rids:
            return (), (), "empty", ((), ())
        res = resource_edit(rng, module.resources[rng.choice(rids)], family)
        return (), (res,), f"resource.{family}", ((), ())
    if r == 4:
        family = UNEMITTABLE[k % len(UNEMITTABLE)]
        claims, resources = unemittable_edit(rng, module, family)
        if not (claims or resources):
            return (), (), "empty", ((), ())
        refused = (claims, resources)
        # an emittable edit beside it, which must survive the refusal: the repair restores only
        # the value the wire refused, so the next pack re-emits this edit or is stale
        bad = {c.id for c in claims}
        others = [c for c in flat if c.id not in bad]
        if others:
            c = others[rng.randrange(len(others))]
            claims += (replace(c, count=c.count + 5),)
        return claims, resources, f"unemittable.{family}", refused
    if r == 6 and _events(module):
        family = EVENT_EDITS[case.index % len(EVENT_EDITS)]
        claims, resources = event_edit(module, family)
        return claims, resources, f"event.{family}" if claims or resources else "empty", ((), ())
    # a mixture: one to three claims and, one time in three, a resource
    claims = []
    if flat:
        for c in rng.sample(flat, min(rng.choice([1, 2, 3]), len(flat))):
            claims.append(claim_edit(rng, c, rids, rng.choice(CLAIM_EDITS)))
    resources = []
    if rids and rng.random() < 1 / 3:
        rid = rng.choice(rids)
        resources.append(resource_edit(rng, module.resources[rid], rng.choice(RESOURCE_EDITS)))
    return tuple(claims), tuple(resources), "mixture", ((), ())


def event_edit(module, family) -> tuple[tuple, tuple]:
    """Replacements aimed at the event laws over the U4 fixture's claims (named by their ops)."""
    from bcir.model import Lane

    flat = _flat(module)
    if family == "disarm":
        return tuple(replace(c, op="uart.idle") for c in flat if c.op.startswith("irq.unmask:")), ()
    if family == "unmask.window":
        return tuple(replace(c, op="uart.idle") for c in flat if c.op.startswith("irq.mask:")), ()
    if family == "atomic.consumer":
        return tuple(
            replace(c, lane=Lane.A, hazard="atomic") for c in flat if c.op == "uart.consume"
        ), ()
    if family == "handler.write":
        fill = [c for c in flat if c.op == "uart.rx_fill"]
        consume = [c for c in flat if c.op == "uart.consume"]
        if fill and consume:
            return (replace(fill[0], wr=tuple(fill[0].wr) + tuple(consume[0].wr)),), ()
        return (), ()
    raise ValueError(family)


def _events(module) -> bool:
    return any(getattr(ph, "event", "") for ph in module.phases)


def _repair(module, claims, resources, copies: bool) -> tuple[tuple, tuple]:
    """The replacements that restore what `claims` and `resources` replaced in `module`: the very
    objects, or equal copies of them when `copies`."""
    by_id = {c.id: c for c in _flat(module)}
    old_claims = tuple(by_id[c.id] for c in claims)
    old_resources = tuple(module.resources[r.rid] for r in resources)
    if copies:
        return tuple(replace(c) for c in old_claims), tuple(replace(r) for r in old_resources)
    return old_claims, old_resources


def forgery(case: Case, r: int) -> tuple[str, str]:
    """The (plan, pack) forgeries of round `r` of `case`, on rotations of coprime length so every
    pair meets across the corpus: the plan of the step before over the cone edits (a planner that
    did not move meets an offer that did), the pack of the step before over the resource edits (an
    emitter that did not move meets a generation that did)."""
    plans, packs = ("none",) + PLAN_FORGERIES, ("none",) + PACK_FORGERIES
    plan = "plan.stale" if r == 1 else plans[(case.index + r) % len(plans)]
    pack = "pack.stale" if r == 2 else packs[(case.index + r) % len(packs)]
    return plan, pack


def forge(kind: str, result, pack, previous):
    """(result, pack) with one field forged: a step or a segment replaced, one of them dropped,
    the score moved, a trace note or a generation record changed, the plan or the pack of the
    step before handed in again (`previous`: a planner or an emitter that did not move, so its
    objects are the ones the verdict already holds) -- or the honest pair."""
    from bcir.model import Lane

    if kind == "plan.stale":
        return previous[0], pack
    if kind == "pack.stale":
        return result, previous[1]
    steps, segments = list(result.steps), list(pack.segments)
    if kind.startswith("plan.") and steps:
        i = len(steps) // 2
        st = steps[i]
        cand = st.candidate
        if kind == "plan.name":
            steps[i] = replace(st, candidate=replace(cand, name="forged"))
        elif kind == "plan.cost":
            steps[i] = replace(st, cost=st.cost + 1)
        elif kind == "plan.phase":
            steps[i] = replace(st, phase_id=st.phase_id + 1_000_003)
        elif kind == "plan.width":  # wide <-> narrow: the next step's fused edge moves too
            width = 1 if cand.width > 1 else 4
            steps[i] = replace(st, candidate=replace(cand, width=width))
        elif kind == "plan.lane":
            lane = Lane.A if cand.lane is not Lane.A else Lane.U
            steps[i] = replace(st, candidate=replace(cand, lane=lane))
        elif kind == "plan.score":
            return replace(result, score=result.score + 1), pack
        elif kind == "plan.claim":
            steps[i] = replace(st, claim_id=st.claim_id + 10_000_000)
        elif kind == "plan.drop":
            del steps[i]
        return replace(result, steps=steps), pack
    if kind.startswith("pack.") and segments:
        i = len(segments) // 2
        seg = segments[i]
        if kind == "pack.read":
            segments[i] = replace(seg, reads=tuple(seg.reads) + (UNDECLARED + 2,))
        elif kind == "pack.prefetch":
            segments[i] = replace(seg, prefetch="forged")
        elif kind == "pack.target":  # a prefetch retargeted, the segment naming it untouched
            prefetches = list(pack.prefetches)
            k = next((k for k, pf in enumerate(prefetches) if pf.name == seg.prefetch), None)
            if k is None:
                return result, pack
            prefetches[k] = replace(prefetches[k], targets=(UNDECLARED + 3,))
            return result, replace(pack, prefetches=prefetches)
        elif kind == "pack.claim":
            segments[i] = replace(seg, claim_id=seg.claim_id + 10_000_000)
        elif kind == "pack.note":
            notes = list(pack.trace_notes)
            notes[i] = replace(notes[i], claim_id=notes[i].claim_id + 10_000_000)
            return result, replace(pack, trace_notes=notes)
        elif kind == "pack.generation":
            gens = list(pack.generations)
            if not gens:
                return result, pack
            gens[0] = replace(gens[0], data_gen=gens[0].data_gen + 1)
            return result, replace(pack, generations=gens)
        elif kind == "pack.drop":
            del segments[i]
        return result, replace(pack, segments=segments)
    return result, pack


# --- the grader -------------------------------------------------------------------------------


def _built(build, rows, out: dict[str, float]) -> list:
    """A corpus, or none with every row it feeds failed: a corpus that cannot be built is a
    finding in a named row, never a traceback (L1)."""
    try:
        return build()
    except Exception:  # noqa: BLE001 -- an oracle that cannot build its corpus decided nothing
        for row in rows:
            out[row] += 1
        return []


def _note(seen: dict | None, key: str, value=1) -> None:
    if seen is not None:
        if isinstance(value, set):
            seen.setdefault(key, set()).update(value)
        else:
            seen[key] = seen.get(key, 0) + value


def _grade_step(chain, delta, want: Reference, new, out: dict, seen):
    """One chain step against the chain from scratch (each row at most once). Returns the link,
    or None when the chain emitted none."""
    from bcir.abi.streampack_abi import AbiError
    from bcir.tests import planner_fixtures as pf

    try:
        link = chain.apply(delta)
        refusal = None
    except AbiError as exc:
        link, refusal = None, exc
    except Exception:  # noqa: BLE001 -- a chain that raises fails every comparison it owns
        for row in ROWS[:3]:
            out[row] += 1
        return None
    try:
        result = chain.result
        plan_ok = (
            chain.module == new
            and result == want.result
            and _steps(result) == _steps(want.result)
            and pf.realization_bytes(result) == pf.realization_bytes(want.result)
        )
    except Exception:  # noqa: BLE001
        plan_ok = False
    out["planner.delta.parity"] += not plan_ok
    if want.refusal is not None or refusal is not None:
        _note(seen, "refused")
        same = (
            want.refusal is not None
            and refusal is not None
            and _refusal(want.refusal) == _refusal(refusal)
        )
        out["pack.delta.identity"] += not same
        out["verify.delta.identity"] += not same  # no verdict on either side, or a mismatch
        return link
    out["pack.delta.identity"] += not (link.pack == want.pack and link.data == want.data)
    out["verify.delta.identity"] += link.diagnostics != want.diagnostics
    if seen is not None:
        seen.setdefault("laws", set()).update(d.law for d in want.diagnostics)
    return link


def _verdict(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 -- a verdict that raised is compared as one
        return ("RAISE",) + _refusal(exc)


def grade_case(case: Case, out: dict[str, float], seen: dict | None = None) -> None:
    """Build the chain for `case`, advance it through the case's rounds, and grade every step
    against the chain from scratch; hold the verifier's state over the forged rail beside it."""
    rng = random.Random(CASE_SEED + case.index)
    args = (case.h, case.theta, case.policy)
    steps = 1 + ROUNDS
    try:
        want = reference_chain(case.module, *args, case.depth)
    except Exception:  # noqa: BLE001 -- the reference itself raised: a corpus defect (L1)
        for row in ROWS[:3]:
            out[row] += steps
        return
    try:
        from bcir.abi.streampack_abi import AbiError
        from bcir.gem.delta_chain import DeltaChain
        from bcir.kbcir.delta import Delta
        from bcir.verify.delta import VerifyState
    except Exception:  # noqa: BLE001 -- the mechanisms are absent: every comparison fails
        for row in ROWS[:3]:
            out[row] += steps
        out["verify.delta.identity"] += steps  # the forged rail
        return

    # -- the build -----------------------------------------------------------------------------
    try:
        chain = DeltaChain.build(case.module, *args, PLAN, case.depth)
        link, refusal = chain.link, None
    except AbiError as exc:
        chain, link, refusal = None, None, exc
    except Exception:  # noqa: BLE001
        for row in ROWS[:3]:
            out[row] += steps
        out["verify.delta.identity"] += steps
        return
    if want.refusal is not None or refusal is not None:
        same = (
            want.refusal is not None
            and refusal is not None
            and _refusal(want.refusal) == _refusal(refusal)
        )
        for row in ROWS[:3]:
            out[row] += not same
        _note(seen, "unbuilt")
        chain = None  # nothing to advance: the case's rounds run the forged rail alone
    else:
        out["planner.delta.parity"] += not (
            link.result == want.result and _steps(link.result) == _steps(want.result)
        )
        out["pack.delta.identity"] += not (link.pack == want.pack and link.data == want.data)
        out["verify.delta.identity"] += link.diagnostics != want.diagnostics
    # The forged rail starts from the chain's own objects and is handed the chain's links, so
    # its state advances by identity over what a delta really changed, plus the forgery.
    honest = (link.result, link.pack) if chain is not None else (want.result, want.pack)
    try:
        vs = VerifyState.build(case.module, *honest, *args)
        forged_ok = vs.diagnostics == full_verdict(want.module_laws, case.module, *honest, *args)
    except Exception:  # noqa: BLE001
        vs, forged_ok = None, False
    out["verify.delta.identity"] += not forged_ok

    # -- the rounds ----------------------------------------------------------------------------
    module = case.module
    pending = None
    for r in range(ROUNDS):
        claims, resources, family, refused = round_edit(rng, module, case, r, pending)
        pending = _repair(module, *refused, case.index % 2 == 1) if r == 4 else None
        _note(seen, "families", {family})
        new = declared(module, claims, resources)
        try:
            want = reference_chain(new, *args, case.depth)
        except Exception:  # noqa: BLE001
            for row in ROWS[:3]:
                out[row] += 1
            out["verify.delta.identity"] += 1
            module = new
            continue
        previous = honest
        if chain is not None:
            link = _grade_step(chain, Delta(claims, resources), want, new, out, seen)
            honest = (chain.result, want.pack if link is None else link.pack)
        else:
            honest = (want.result, want.pack)
        # the forged rail: the verifier's state handed a plan and pack it did not produce
        plan_kind, pack_kind = forgery(case, r)
        _note(seen, "forgeries", {plan_kind, pack_kind})
        fres, _ = forge(plan_kind, *honest, previous)
        _, fpack = forge(pack_kind, *honest, previous)
        if vs is None:
            out["verify.delta.identity"] += 1
        else:
            got = _verdict(vs.apply, new, fres, fpack)
            ref = _verdict(full_verdict, want.module_laws, new, fres, fpack, *args)
            out["verify.delta.identity"] += got != ref
            if seen is not None and isinstance(ref, list):
                seen.setdefault("laws", set()).update(d.law for d in ref)
        module = new
    _note(seen, "graded")
    if chain is not None:
        _note(seen, "chain.rebuilds", chain.rebuilds)
        if not _events(case.module):
            _note(seen, "chain.rebuilds.without-events", chain.rebuilds)
    if vs is not None:
        _note(seen, "forged.rebuilds", vs.rebuilds)


# --- the malformed corpus ---------------------------------------------------------------------


def malformed_base():
    """A small legal module the malformed deltas address: two phases, four claims, three
    resources."""
    from bcir.model import Phase, Resource

    return _module(
        "delta.malformed",
        [Resource(rid=r, shape=(64,)) for r in (1, 2, 3)],
        [
            Phase(
                0,
                (),
                [
                    _claim(1, rd=(1,), wr=(2,), op="m.a", count=64),
                    _claim(2, rd=(2,), wr=(3,), op="m.b", count=64),
                ],
            ),
            Phase(
                1,
                (0,),
                [
                    _claim(3, rd=(3,), wr=(1,), op="m.c", count=64),
                    _claim(4, rd=(1, 3), wr=(2,), op="m.d", count=64),
                ],
            ),
        ],
    )


def malformed_deltas(Delta) -> list[tuple[str, object]]:
    """(label, delta) pairs no rail may admit: each must be refused with `DeltaError` before
    anything moved."""
    m = malformed_base()
    c = _flat(m)
    r = [m.resources[rid] for rid in sorted(m.resources)]
    return [
        ("a tuple of replacements, not a Delta", (tuple(c[:1]), ())),
        ("None, not a Delta", None),
        ("a Resource as a claim replacement", Delta((r[0],), ())),
        ("an int as a claim replacement", Delta((7,), ())),
        ("a Claim as a resource replacement", Delta((), (c[0],))),
        ("a claim id the module does not declare", Delta((replace(c[0], id=99_991),), ())),
        ("a RID the module does not declare", Delta((), (replace(r[0], rid=99_991),))),
        ("one claim replaced twice", Delta((c[0], replace(c[0], count=3)), ())),
        ("one claim replaced twice by one object", Delta((c[1], c[1]), ())),
        ("one resource replaced twice", Delta((), (r[0], replace(r[0], align=128)))),
        ("the claims a list", Delta([replace(c[0], count=5)], ())),
        ("the resources None", Delta((), None)),
        ("the claims a generator", Delta((x for x in (replace(c[0], count=6),)), ())),
        (
            "an admissible replacement beside an undeclared id",
            Delta((replace(c[0], count=7), replace(c[1], id=99_992)), ()),
        ),
        (
            "an admissible replacement beside an undeclared RID",
            Delta((replace(c[2], count=8),), (replace(r[1], rid=99_993),)),
        ),
    ]


_DUPLICATE_RAILS = 4  # IncrementalPlan.build, DeltaChain.build, VerifyState.build, apply_delta
_TOUCHED_RAILS = 2  # DeltaChain.apply, IncrementalPlan.apply


@dataclass(frozen=True)
class _StandIn:
    """The shape of a `Delta`, to count the malformed corpus where the mechanism is absent."""

    claims: object = ()
    resources: object = ()


def duplicate_module():
    """A module declaring claim id 2 twice: a delta cannot address it."""
    m = malformed_base()
    ph = m.phases[1]
    m.phases[1] = replace(ph, claims=[ph.claims[0], replace(ph.claims[1], id=2)])
    m.touch()
    return m


def _honest(module):
    """The follow-up delta a refusal must not have disturbed: every claim's count moved."""
    return tuple(replace(c, count=c.count + 1) for c in _flat(module)), ()


def grade_malformed(out: dict[str, float], seen: dict | None = None) -> None:
    """delta.malformed.accepted: one per (malformed input, rail) not refused with DeltaError
    before anything moved, or refused and then unable to reproduce the next honest delta."""
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.kbcir.weights import PERF

    row = "delta.malformed.accepted"
    h, theta = TargetProfile.x86_avx2(), Theta.cool()
    try:
        from bcir.gem.delta_chain import DeltaChain
        from bcir.kbcir.delta import Delta, DeltaError, IncrementalPlan, apply_delta
        from bcir.verify.delta import VerifyState
    except Exception:  # noqa: BLE001 -- the mechanisms are absent: nothing is refused
        out[row] += 3 * len(malformed_deltas(_StandIn)) + _DUPLICATE_RAILS + _TOUCHED_RAILS
        return

    def refused(fn, *args) -> bool:
        try:
            fn(*args)
        except DeltaError:
            return True
        except Exception:  # noqa: BLE001 -- refused, but not as the contract declares (L1)
            return False
        return False

    base = malformed_base()
    items = _built(lambda: malformed_deltas(Delta), [row], out)
    for _label, delta in items:
        _note(seen, "malformed")
        # apply_delta: refused, the module untouched
        m = malformed_base()
        revision = m.revision
        ok = refused(apply_delta, m, delta) and m == base and m.revision == revision
        out[row] += not ok
        # IncrementalPlan.apply: refused, then the next honest delta is `optimize` of its module
        try:
            m = malformed_base()
            plan = IncrementalPlan.build(m, h, theta, PERF)
            ok = refused(plan.apply, delta) and plan.module is m
            claims, resources = _honest(m)
            got = plan.apply(Delta(claims, resources))
            want = reference_chain(declared(m, claims, resources), h, theta, PERF, 2).result
            ok = ok and got == want and _steps(got) == _steps(want)
        except Exception:  # noqa: BLE001
            ok = False
        out[row] += not ok
        # DeltaChain.apply: refused, then the next honest delta is the chain from scratch
        try:
            m = malformed_base()
            chain = DeltaChain.build(m, h, theta, PERF, PLAN, 2)
            before = chain.link
            ok = refused(chain.apply, delta) and chain.link is before
            claims, resources = _honest(m)
            link = chain.apply(Delta(claims, resources))
            want = reference_chain(declared(m, claims, resources), h, theta, PERF, 2)
            ok = (
                ok
                and link.result == want.result
                and link.data == want.data
                and link.diagnostics == want.diagnostics
            )
        except Exception:  # noqa: BLE001
            ok = False
        out[row] += not ok

    # a module that declares a claim id twice: every build refuses, and apply_delta refuses a
    # delta addressing the id
    dup = _built(lambda: [duplicate_module()], [row], out)
    for m in dup:
        _note(seen, "malformed", _DUPLICATE_RAILS)
        out[row] += not refused(IncrementalPlan.build, m, h, theta, PERF)
        out[row] += not refused(DeltaChain.build, m, h, theta, PERF, PLAN, 2)
        # hydration refuses the module itself, so the verdict's state is handed the module's
        # plan and a pack of the module it was made from: it must refuse the module first
        try:
            from bcir.kbcir.realize import optimize

            stand_in = reference_chain(malformed_base(), h, theta, PERF, 2).pack
            result = optimize(m, h, theta, PERF)
        except Exception:  # noqa: BLE001
            out[row] += 1
        else:
            out[row] += not refused(VerifyState.build, m, result, stand_in, h, theta, PERF)
        # even a delta naming only a claim whose id is unique: the module is refused whole
        out[row] += not refused(apply_delta, m, Delta((_flat(m)[0],), ()))

    # a module changed outside a delta (a declared mutation, S1-B): the chain and the plan refuse
    # to advance from a module they no longer describe
    _note(seen, "malformed", _TOUCHED_RAILS)
    try:
        m = malformed_base()
        chain = DeltaChain.build(m, h, theta, PERF, PLAN, 2)
        plan = IncrementalPlan.build(m, h, theta, PERF)
        m.touch()
        delta = Delta(*_honest(m))
        out[row] += not (refused(chain.apply, delta) and chain.module is m)
        out[row] += not (refused(plan.apply, delta) and plan.module is m)
    except Exception:  # noqa: BLE001
        out[row] += _TOUCHED_RAILS


def measure(stride: int = 1, seen: dict | None = None) -> dict[str, float]:
    """The G18 rows (module docstring) over every `stride`-th case of the corpus. `seen`, when
    given, collects what the corpus exercised: the edit families, the forgeries, the laws the
    full verdicts raised, the wire refusals, the verdicts re-derived from scratch."""
    out = {row: 0.0 for row in ROWS}
    cases = _built(corpus_cases, ROWS[:3], out)
    for case in cases[::stride]:
        grade_case(case, out, seen)
    grade_malformed(out, seen)
    return out


# --- the harness rows beyond the gate ----------------------------------------------------------

RATIO_SCALE = 4  # the audit's K_BCIR->StreamPack fixture at scale 4: 4,096 claims
CALLS_SCALE = 8  # ... at scale 8: 32,768 claims


def audit_delta(module, k: int) -> tuple[tuple, tuple]:
    """The k-th one-claim delta of the audit fixture: one claim's count moved (the reads, the
    writes and the legality kept, so the cone is the claim's column and the DP's)."""
    flat = _flat(module)
    c = flat[(len(flat) // 2 + k * 997) % len(flat)]
    return (replace(c, count=c.count + 1 + k),), ()


def delta_ratio(scale: int = RATIO_SCALE, rounds: int = 9) -> float:
    """Median time of one `DeltaChain.apply` over the median time of the chain from scratch on
    the module the delta declares, the audit fixture at `scale`, in one process."""
    import gc
    import statistics
    import time

    from bcir.gem.delta_chain import DeltaChain
    from bcir.kbcir.delta import Delta
    from bcir.kbcir.weights import PERF
    from bcir.tests.planner_fixtures import audit_fixture

    module, h, theta = audit_fixture(scale)
    chain = DeltaChain.build(module, h, theta, PERF, PLAN, 2)
    full, step = [], []
    for k in range(rounds):
        claims, resources = audit_delta(chain.module, k)
        new = declared(chain.module, claims, resources)
        gc.collect()
        t0 = time.perf_counter()
        reference_chain(new, h, theta, PERF, 2)
        full.append(time.perf_counter() - t0)
        gc.collect()
        t0 = time.perf_counter()
        chain.apply(Delta(claims, resources))
        step.append(time.perf_counter() - t0)
    return statistics.median(step) / statistics.median(full)


def _profiled_calls(fn, *args) -> int:
    """cProfile's total for one call of `fn`, with the collector run first and paused during it:
    a finalizer the collector happens to run inside the window is a call the code did not make."""
    import cProfile
    import gc
    import pstats

    gc.collect()
    gc.disable()
    profile = cProfile.Profile()
    try:
        profile.enable()
        fn(*args)
        profile.disable()
    finally:
        gc.enable()
    return pstats.Stats(profile).total_calls


def delta_calls(scale: int = CALLS_SCALE) -> tuple[int, int]:
    """(calls of one `DeltaChain.apply`, calls of the chain from scratch) for the one-claim delta
    of the audit fixture at `scale` (cProfile's total, builtins included). Deterministic for one
    interpreter and different between CPython versions, so a gate compares the two in one
    process."""
    from bcir.gem.delta_chain import DeltaChain
    from bcir.kbcir.delta import Delta
    from bcir.kbcir.weights import PERF
    from bcir.tests.planner_fixtures import audit_fixture

    module, h, theta = audit_fixture(scale)
    chain = DeltaChain.build(module, h, theta, PERF, PLAN, 2)
    chain.apply(Delta(*audit_delta(chain.module, 1)))  # warm imports and caches outside the count
    claims, resources = audit_delta(chain.module, 0)
    new = declared(chain.module, claims, resources)
    step = _profiled_calls(chain.apply, Delta(claims, resources))
    return step, _profiled_calls(reference_chain, new, h, theta, PERF, 2)


def full_calls(scale: int = CALLS_SCALE) -> int:
    """Calls of the chain from scratch on the audit fixture's one-claim delta at `scale` -- what a
    delta costs without G18 (the RED of `kbcir-streampack.delta.calls`)."""
    from bcir.kbcir.weights import PERF
    from bcir.tests.planner_fixtures import audit_fixture

    module, h, theta = audit_fixture(scale)
    reference_chain(module, h, theta, PERF, 2)  # warm
    claims, resources = audit_delta(module, 0)
    return _profiled_calls(reference_chain, declared(module, claims, resources), h, theta, PERF, 2)
