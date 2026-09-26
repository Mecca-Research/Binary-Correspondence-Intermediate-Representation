"""G8 (staged plan S5-C) fixtures and rows: the movement corpus, the law variants and `measure`.

One module the tests, the harness group (`tools/perf/gemplus_baseline.py --group movement`) and
the gate (`tools/perf/check_movement.py`) share, so the three read the same numbers.

The corpus is small by design -- every fixture's candidate space fits the exact planner's budget,
so every planner row is the optimum of the joint objective (TMSAO-1) and every baseline's excess
over it is exact. Each fixture exercises one thing the transform must get right:

  tiled_matmul, matmul_tiled  the example programs whose claims read RAM and HBM at once (the
                              D-R2 violations the corpus carried): moving the A/B tiles is optimal
  reduce6                     six reductions of one RAM array in one phase: the pre-G8 rules send
                              all six to HBM for its compute and pay seven transfers for it;
                              staying home is the optimum
  amortize                    moving pays for nobody: staying home is optimal (the identity)
  pingpong                    the sequential rule moves a large array to compute a small one
  stream                      four immutable weights cycled through an HBM that holds three:
                              forced evictions, each reload forced
  remat                       a recomputable temporary replayed in HBM beats moving it
  compressed                  a tolerant consumer takes an 8-bit transfer
  staged                      SSD has no link to HBM: the route stages through RAM
  writeback                   mutable state updated in HBM is written back home
  device                      an MMIO claim is pinned and its register never moves

Rows (`measure`), each `exact`:

  movement.implicit_cross_tier  claims of the example programs that read or write two memory
                                tiers without an explicit move (D-R2), after the planner
  movement.excess               summed makespan of the chosen plans over the exact optimum
                                (`joint_optimum`: every candidate priced, never the planner's own
                                answer)
  movement.laws.accepted        law variants (one per law, each violating exactly one) accepted
  movement.laws.misattributed   variants refused, but not by the law they break
  movement.identity.drift       fixtures that need no movement whose plan is not byte-identical
                                to the plan of the module itself
  movement.roundtrip.mismatch   planned plans whose v3 bytes do not decode to themselves
  movement.plans.unlawful       fixtures whose chosen plan any law refuses -- the movement laws
                                with their source and spec, R1-R25 over the transformed module,
                                or an access that races
  movement.asn1.accepted        (malformed plan, transfer syntax) pairs a BCIR-ExecutionPlan
                                decoder (DER, OER, JER) accepts
  movement.parity.mismatch      planned plans the C twin does not read, bind to their pack and
                                registry and round-trip as the Python rail does, and malformed
                                moves and bindings (`plan_fixtures.v3_variants`) not refused by
                                both rails -- needs a C compiler and the checkout's runtime/c
                                (CC_ROWS): NOT-MEASURED without them, never zero by default
"""

from __future__ import annotations

from dataclasses import replace

from ..frontends.models.assessment import BankEnvelope, HardwareEnvelope, LinkEnvelope
from ..model import Claim, Domain, Module, Opcode, Phase, Resource

#: The fixtures whose joint optimum moves nothing: their plan is the plan of the module itself
#: (v1, byte for byte); every other fixture's plan moves data (v3).
IDENTITY = frozenset({"reduce6", "amortize"})

#: The rows `measure` returns, in order.
ROWS = (
    "movement.implicit_cross_tier",
    "movement.excess",
    "movement.laws.accepted",
    "movement.laws.misattributed",
    "movement.identity.drift",
    "movement.roundtrip.mismatch",
    "movement.plans.unlawful",
    "movement.asn1.accepted",
    "movement.parity.mismatch",
)
#: The rows that need a C compiler (the plan harness): left out of `measure` without one.
CC_ROWS = ("movement.parity.mismatch",)


def hardware(*, hbm: int = 1 << 30, ssd: bool = False, mmio: bool = False) -> HardwareEnvelope:
    """A host (RAM) and an accelerator (HBM) joined both ways, optionally an SSD that reaches
    only RAM and a device register file."""
    banks = [
        BankEnvelope("ram", "RAM", "host", 1 << 30, 1 << 30, 20_000_000_000, 20_000_000_000),
        BankEnvelope("hbm", "HBM", "accel", hbm, hbm, 80_000_000_000, 80_000_000_000),
    ]
    links = [
        LinkEnvelope("ram", "hbm", 16_000_000_000, 1000),
        LinkEnvelope("hbm", "ram", 16_000_000_000, 1000),
    ]
    if ssd:
        banks.append(
            BankEnvelope("ssd", "NVM", "storage", 1 << 30, 1 << 30, 2_000_000_000, 2_000_000_000)
        )
        links += [
            LinkEnvelope("ssd", "ram", 2_000_000_000, 5000),
            LinkEnvelope("ram", "ssd", 2_000_000_000, 5000),
        ]
    if mmio:
        banks.append(
            BankEnvelope("regs", "MMIO", "device", 4096, 4096, 1_000_000_000, 1_000_000_000)
        )
    return HardwareEnvelope("g8-fixture", "x86_64", tuple(banks), tuple(links), ())


def _claim(cid, opcode, count, rd, wr, op, **kw) -> Claim:
    return Claim(id=cid, opcode=opcode, count=count, rd=rd, wr=wr, op=op, **kw)


def _reduce6(k: int = 6, n: int = 1 << 16) -> Module:
    m = Module(name="reduce6")
    m.add_resource(Resource(1, Domain.RAM, 4, (n,), name="X"))
    for i in range(k):
        m.add_resource(Resource(10 + i, Domain.RAM, 4, (16,), name=f"y{i}"))
    m.add_phase(
        Phase(
            0, (), [_claim(100 + i, Opcode.ADD, n, (1,), (10 + i,), "reduce.sum") for i in range(k)]
        )
    )
    return m


def _amortize(k: int = 4, n: int = 1 << 16) -> Module:
    m = Module(name="amortize")
    m.add_resource(Resource(1, Domain.RAM, 4, (n,), name="X"))
    for i in range(k):
        m.add_resource(Resource(10 + i, Domain.RAM, 4, (n,), name=f"y{i}"))
    m.add_phase(
        Phase(
            0,
            (),
            [_claim(100 + i, Opcode.ADD, n, (1,), (10 + i,), "vector.scale") for i in range(k)],
        )
    )
    return m


def _pingpong(big: int = 1 << 16, small: int = 64) -> Module:
    m = Module(name="pingpong")
    m.add_resource(Resource(1, Domain.RAM, 4, (big,), name="A"))
    m.add_resource(Resource(2, Domain.RAM, 4, (small,), name="t"))
    m.add_resource(Resource(3, Domain.HBM, 4, (small,), name="W"))
    m.add_resource(Resource(4, Domain.HBM, 4, (small,), name="out"))
    m.add_phase(Phase(0, (), [_claim(100, Opcode.ADD, big, (1,), (2,), "reduce.sum")]))
    m.add_phase(Phase(1, (0,), [_claim(101, Opcode.MUL, small, (3, 2), (4,), "vector.mul")]))
    return m


def _stream(n: int = 4, size: int = 1 << 14) -> Module:
    m = Module(name="stream")
    for i in range(n):
        m.add_resource(Resource(1 + i, Domain.RAM, 4, (size,), name=f"W{i}"))
    m.add_resource(Resource(50, Domain.HBM, 4, (size,), name="acc"))
    for k, i in enumerate((0, 1, 2, 3, 0, 1, 2, 3)):
        m.add_phase(
            Phase(
                k,
                (k - 1,) if k else (),
                [_claim(100 + k, Opcode.ADD, size, (1 + i, 50), (50,), "vector.add")],
            )
        )
    return m


def _remat(n: int = 1 << 15) -> Module:
    m = Module(name="remat")
    m.add_resource(Resource(1, Domain.HBM, 4, (n,), name="x"))
    m.add_resource(Resource(2, Domain.RAM, 4, (n,), name="t"))
    m.add_resource(Resource(3, Domain.HBM, 4, (n,), name="y"))
    m.add_resource(Resource(4, Domain.RAM, 4, (16,), name="s"))
    m.add_phase(Phase(0, (), [_claim(10, Opcode.MUL, n, (1,), (2,), "vector.square")]))
    m.add_phase(Phase(1, (0,), [_claim(11, Opcode.ADD, n, (2,), (4,), "reduce.sum")]))
    m.add_phase(Phase(2, (1,), [_claim(12, Opcode.ADD, n, (2, 1), (3,), "vector.add")]))
    return m


def _compressed(n: int = 1 << 15) -> Module:
    m = Module(name="compressed")
    m.add_resource(Resource(1, Domain.RAM, 4, (n,), name="W"))
    m.add_resource(Resource(2, Domain.HBM, 4, (n,), name="y"))
    m.add_phase(
        Phase(0, (), [_claim(10, Opcode.MUL, n, (1,), (2,), "vector.scale", tolerance_ulp=8)])
    )
    return m


def _staged(n: int = 4096) -> Module:
    m = Module(name="staged")
    m.add_resource(Resource(1, Domain.NVM, 4, (n,), name="blob"))
    m.add_resource(Resource(2, Domain.HBM, 4, (n,), name="y"))
    m.add_phase(
        Phase(0, (), [_claim(10, Opcode.ADD, n, (1,), (2,), "vector.copy", domain=Domain.HBM)])
    )
    return m


def _writeback(n: int = 4096) -> Module:
    m = Module(name="writeback")
    m.add_resource(Resource(1, Domain.RAM, 4, (n,), name="state"))
    m.add_resource(Resource(2, Domain.HBM, 4, (n,), name="g"))
    m.add_phase(Phase(0, (), [_claim(10, Opcode.ADD, n, (1, 2), (1,), "vector.add")]))
    m.add_phase(Phase(1, (0,), [_claim(11, Opcode.ADD, n, (1, 2), (1,), "vector.add")]))
    return m


def _device(n: int = 4096) -> Module:
    m = Module(name="device")
    m.add_resource(Resource(1, Domain.MMIO, 4, (1,), name="ctrl"))
    m.add_resource(Resource(2, Domain.RAM, 4, (1,), name="cmd"))
    m.add_resource(Resource(3, Domain.RAM, 4, (n,), name="buf"))
    m.add_resource(Resource(4, Domain.HBM, 4, (n,), name="out"))
    m.add_phase(
        Phase(
            0,
            (),
            [
                _claim(
                    10,
                    Opcode.STORE,
                    1,
                    (2,),
                    (1,),
                    "mmio.write",
                    domain=Domain.MMIO,
                    volatile=True,
                    hazard="barriered",
                )
            ],
        )
    )
    m.add_phase(Phase(1, (0,), [_claim(11, Opcode.ADD, n, (3,), (4,), "vector.copy")]))
    return m


def fixtures():
    """`[(name, module, spec)]`: the corpus, in the order the rows sum it."""
    from ..examples import PROGRAMS
    from ..kbcir.movement import MovementSpec

    hw = hardware()
    weights = tuple((1 + i, "immutable") for i in range(4))
    return [
        ("tiled_matmul", PROGRAMS["tiled_matmul"](), MovementSpec(hw)),
        ("matmul_tiled", PROGRAMS["matmul_tiled"](), MovementSpec(hw)),
        ("reduce6", _reduce6(), MovementSpec(hw, classes=((1, "immutable"),))),
        ("amortize", _amortize(), MovementSpec(hw, classes=((1, "immutable"),))),
        ("pingpong", _pingpong(), MovementSpec(hw, classes=((1, "immutable"), (3, "immutable")))),
        (
            "stream",
            _stream(),
            MovementSpec(
                hardware(hbm=3 * (1 << 16) + 4096),
                classes=weights,
                sites=tuple((100 + k, ("hbm",)) for k in range(8)),
            ),
        ),
        (
            "remat",
            _remat(),
            MovementSpec(
                hw,
                classes=((1, "immutable"), (2, "recomputable")),
                remat=(2,),
                sites=((10, ("ram",)), (11, ("ram",)), (12, ("hbm",))),
            ),
        ),
        (
            "compressed",
            _compressed(),
            MovementSpec(
                hw, classes=((1, "immutable"),), codecs=((1, 8),), sites=((10, ("hbm",)),)
            ),
        ),
        (
            "staged",
            _staged(),
            MovementSpec(hardware(ssd=True), classes=((1, "immutable"),), sites=((10, ("hbm",)),)),
        ),
        ("writeback", _writeback(), MovementSpec(hw, classes=((2, "immutable"),))),
        ("device", _device(), MovementSpec(hardware(mmio=True), classes=((3, "immutable"),))),
    ]


def target_and_theta():
    from ..kbcir.cost import TargetProfile, Theta

    return TargetProfile.x86_avx512(), Theta()


def joint_optimum(module: Module, spec):
    """The optimum of the joint objective, found WITHOUT the planner: every site choice, every
    subset of the remats and every subset of the codecs the spec declares, each priced by
    `movement.price` (the objective itself) and the least key kept. `movement.excess` and the
    tests measure the planner against this, never against its own answer."""
    import itertools

    from ..kbcir.movement import price, site_choices

    h, theta = target_and_theta()
    choices = site_choices(module, spec)
    ids = sorted(choices)

    def subsets(items):
        items = sorted(items)
        return [
            set(combo) for k in range(len(items) + 1) for combo in itertools.combinations(items, k)
        ]

    remats = subsets(
        r for r in spec.remat if r in module.resources and spec.class_of(r) == "recomputable"
    )
    codecs = subsets(r for r, _bits in spec.codecs if r in module.resources)
    best = None
    for combo in itertools.product(*(choices[c] for c in ids)):
        for rm in remats:
            for cp in codecs:
                p = price(module, spec, dict(zip(ids, combo)), h, theta, remat=rm, compress=cp)
                if p is not None and (best is None or p.key < best.key):
                    best = p
    return best


def planned(name: str | None = None):
    """`{name: (module, spec, MovementPlan)}` over the corpus (or one fixture)."""
    from ..kbcir.movement import plan_movement

    h, theta = target_and_theta()
    out = {}
    for fname, module, spec in fixtures():
        if name is None or fname == name:
            out[fname] = (module, spec, plan_movement(module, spec, h, theta))
    return out


# --- the law variants -----------------------------------------------------------------------


class VariantUnavailable(LookupError):
    """The planned corpus lacks the edge a law variant breaks: the planner or the transform no
    longer produces what the variants are minted from (their rows are then left unmeasured)."""


def _edge_index(plan, kind=None, coherence=None) -> int:
    for i, mv in enumerate(plan.moves):
        if (kind is None or mv.kind == kind) and (coherence is None or mv.coherence == coherence):
            return i
    raise VariantUnavailable(f"no {kind or 'edge'} with coherence {coherence or 'any'} to break")


def _swap_claim(module: Module, cid: int, new: Claim) -> Module:
    phases = [
        replace(ph, claims=[new if c.id == cid else c for c in ph.claims]) for ph in module.phases
    ]
    return replace(module, phases=phases)


def _drop_claim(module: Module, cid: int) -> Module:
    phases = [replace(ph, claims=[c for c in ph.claims if c.id != cid]) for ph in module.phases]
    return replace(module, phases=phases)


def _claim_of(module: Module, cid: int) -> Claim:
    return next(c for ph in module.phases for c in ph.claims if c.id == cid)


def law_variants():
    """`[(name, law, source, spec, module, plan)]`: each a legal (M', plan) of the corpus with ONE
    thing broken, and the movement law that must refuse it. A variant that changes the source
    or the spec re-binds the plan's hashes, so only the law it names is violated by design."""
    from ..kbcir.movement import (
        MovementSpec,
        accuracy_certificate,
        execution_plan_of,
        source_digest,
    )

    h, _theta = target_and_theta()
    base = planned()
    out = []

    def mint(name):
        module, spec, mp = base[name]
        return module, spec, mp.best.transform.module, execution_plan_of(mp.best, h)

    # writeback: a fetch into HBM and the writeback home
    src, spec, mod, plan = mint("writeback")
    wb = _edge_index(plan, coherence="writeback")
    fetch = _edge_index(plan, coherence="none")
    out.append(
        ("binding.source", "MV11", src, spec, mod, replace(plan, source_hash=plan.source_hash ^ 1))
    )
    out.append(
        ("binding.spec", "MV11", src, spec, mod, replace(plan, spec_hash=plan.spec_hash ^ 1))
    )

    def edge(i, **kw):
        moves = list(plan.moves)
        moves[i] = replace(moves[i], **kw)
        return replace(plan, moves=moves)

    out.append(
        ("edge.version", "MV11", src, spec, mod, edge(fetch, version=plan.moves[fetch].version + 1))
    )
    out.append(
        (
            "edge.window",
            "MV11",
            src,
            spec,
            mod,
            edge(fetch, flags=plan.moves[fetch].flags & ~2, before_claim=0),
        )
    )
    out.append(("edge.route", "MV11", src, spec, mod, edge(fetch, route="hbm>ram")))
    out.append(
        (
            "edge.generation",
            "MV11",
            src,
            spec,
            mod,
            edge(fetch, map_gen=plan.moves[fetch].map_gen + 1),
        )
    )
    out.append(
        (
            "edge.missing",
            "MV11",
            src,
            spec,
            mod,
            replace(plan, moves=[m for i, m in enumerate(plan.moves) if i != fetch]),
        )
    )
    wb_claim = plan.moves[wb].claim
    out.append(
        (
            "writeback.dropped",
            "MV5",
            src,
            spec,
            _drop_claim(mod, wb_claim),
            replace(plan, moves=[m for i, m in enumerate(plan.moves) if i != wb]),
        )
    )
    # a race: the phase that reads the fetched copy no longer waits for the phase that fetches it
    phase_of = {c.id: ph.phase_id for ph in mod.phases for c in ph.claims}
    fetch_phase = phase_of[plan.moves[fetch].claim]
    reader_phase = phase_of[plan.moves[fetch].before_claim]
    raced = replace(
        mod,
        phases=[
            replace(ph, deps=tuple(d for d in ph.deps if d != fetch_phase))
            if ph.phase_id == reader_phase
            else ph
            for ph in mod.phases
        ],
    )
    out.append(("race", "MV4", src, spec, raced, plan))
    first = _claim_of(mod, 10)
    out.append(
        (
            "claim.changed",
            "MV2",
            src,
            spec,
            _swap_claim(mod, 10, replace(first, op="vector.sub")),
            plan,
        )
    )
    mv = _claim_of(mod, plan.moves[fetch].claim)
    out.append(
        (
            "move.spelling",
            "MV3",
            src,
            spec,
            _swap_claim(mod, mv.id, replace(mv, op=mv.op.replace("far", "near"))),
            plan,
        )
    )
    copy_rid = mv.wr[0]
    bent = dict(mod.resources)
    bent[copy_rid] = replace(bent[copy_rid], shape=(bent[copy_rid].count // 2,))
    out.append(("copy.shape", "MV1", src, spec, replace(mod, resources=bent), plan))
    tight = MovementSpec(
        hardware(hbm=4096), classes=spec.classes, codecs=spec.codecs, sites=spec.sites
    )
    out.append(("capacity", "MV9", src, tight, mod, replace(plan, spec_hash=tight.digest())))
    late = replace(spec, deadline=max(1, plan.makespan - 1))
    out.append(("deadline", "MV10", src, late, mod, replace(plan, spec_hash=late.digest())))
    frozen = replace(spec, classes=((1, "immutable"), (2, "immutable")))
    out.append(
        ("class.immutable", "MV6", src, frozen, mod, replace(plan, spec_hash=frozen.digest()))
    )

    # stream: forced evictions become unforced in a roomier HBM
    src, spec, mod, plan = mint("stream")
    roomy = replace(spec, hardware=hardware(hbm=1 << 30))
    out.append(("thrash", "MV9", src, roomy, mod, replace(plan, spec_hash=roomy.digest())))

    # remat: the certificate and the producer
    src, spec, mod, plan = mint("remat")
    rm = _edge_index(plan, kind="rematerialized")

    def redge(i, **kw):
        moves = list(plan.moves)
        moves[i] = replace(moves[i], **kw)
        return replace(plan, moves=moves)

    out.append(("remat.cert", "MV7", src, spec, mod, redge(rm, cert=plan.moves[rm].cert ^ 1)))
    unsafe = [
        replace(ph, claims=[replace(c, hazard="barriered") if c.id == 10 else c for c in ph.claims])
        for ph in src.phases
    ]
    usrc = replace(src, phases=unsafe)
    umod = _swap_claim(mod, 10, replace(_claim_of(mod, 10), hazard="barriered"))
    remat_id = plan.moves[rm].claim
    umod = _swap_claim(umod, remat_id, replace(_claim_of(umod, remat_id), hazard="barriered"))
    out.append(
        ("remat.unsafe", "MV7", usrc, spec, umod, replace(plan, source_hash=source_digest(usrc)))
    )

    # compressed: the certificate, the codec, the consumer's tolerance
    src, spec, mod, plan = mint("compressed")
    cp = _edge_index(plan, kind="compressed")

    def cedge(i, **kw):
        moves = list(plan.moves)
        moves[i] = replace(moves[i], **kw)
        return replace(plan, moves=moves)

    out.append(("compressed.cert", "MV8", src, spec, mod, cedge(cp, cert=plan.moves[cp].cert ^ 1)))
    wide = replace(spec, codecs=((1, 16),))
    out.append(("compressed.codec", "MV8", src, wide, mod, replace(plan, spec_hash=wide.digest())))
    strict = [
        replace(ph, claims=[replace(c, tolerance_ulp=0) if c.id == 10 else c for c in ph.claims])
        for ph in src.phases
    ]
    ssrc = replace(src, phases=strict)
    smod = _swap_claim(mod, 10, replace(_claim_of(mod, 10), tolerance_ulp=0))
    # the certificate re-derives for the intolerant reader, so only the tolerance law is broken
    edge = plan.moves[cp]
    recert = accuracy_certificate(ssrc, edge.rid, edge.bits, [_claim_of(ssrc, 10)])
    out.append(
        (
            "compressed.tolerance",
            "MV8",
            ssrc,
            spec,
            smod,
            replace(cedge(cp, cert=recert), source_hash=source_digest(ssrc)),
        )
    )

    # matmul_tiled: an original claim reads a copy in another bank (implicit cross-tier)
    src, spec, mod, plan = mint("matmul_tiled")
    c = _claim_of(mod, 4000)
    out.append(
        (
            "claim.two_banks",
            "MV2",
            src,
            spec,
            _swap_claim(mod, 4000, replace(c, rd=(1000000,) + c.rd[1:])),
            plan,
        )
    )
    out.extend(_malformed_variants(mint))
    return out


def _malformed_variants(mint):
    """Transforms and plans a planner never mints -- an undeclared operand, an edge naming no
    source resource, a copy or an edge in an isolated domain, a lifetime in an undeclared bank
    or the wrong domain, a spec with no home for a resource, a remat of a producer that reads a
    device register: each answered by its law, never by a traceback (every exit is a verdict)."""
    from ..kbcir.movement import MovementSpec, source_digest

    out = []
    src, spec, mod, plan = mint("writeback")
    fetch = _edge_index(plan, coherence="none")
    moves = list(plan.moves)
    moves[fetch] = replace(moves[fetch], rid=999)
    out.append(("edge.unknown", "MV11", src, spec, mod, replace(plan, moves=moves)))
    mv = _claim_of(mod, plan.moves[fetch].claim)
    out.append(
        (
            "claim.undeclared",
            "MV1",
            src,
            spec,
            _swap_claim(mod, mv.id, replace(mv, rd=(123456,))),
            plan,
        )
    )
    isolated = dict(mod.resources)
    isolated[mv.wr[0]] = replace(isolated[mv.wr[0]], domain=Domain.MMIO)
    out.append(("copy.isolated", "MV1", src, spec, replace(mod, resources=isolated), plan))
    hw = spec.hardware
    banks = tuple(b for b in hw.banks if b.domain != "HBM")
    kept = {b.name for b in banks}
    links = tuple(ln for ln in hw.links if ln.source in kept and ln.destination in kept)
    homeless = replace(spec, hardware=replace(hw, banks=banks, links=links))
    out.append(
        ("spec.homeless", "MV1", src, homeless, mod, replace(plan, spec_hash=homeless.digest()))
    )

    # device: the register's lifetime and an edge that moves it
    src, spec, mod, plan = mint("device")
    reg = next(i for i, lt in enumerate(plan.lifetimes) if lt.rid == 1)
    for name, bank in (("lifetime.undeclared", "nowhere"), ("lifetime.domain", "ram")):
        lifetimes = list(plan.lifetimes)
        lifetimes[reg] = replace(lifetimes[reg], bank=bank)
        out.append((name, "MV1", src, spec, mod, replace(plan, lifetimes=lifetimes)))
    moves = list(plan.moves)
    moves[0] = replace(moves[0], rid=1)
    out.append(("edge.isolated", "MV3", src, spec, mod, replace(plan, moves=moves)))

    # remat: the replayed producer (and its clone, and the original) also read a register the
    # plan does not place -- the replay is not certified
    src, spec, mod, plan = mint("remat")
    rm = _edge_index(plan, kind="rematerialized")
    producer, clone = plan.moves[rm].producer, plan.moves[rm].claim
    reg = Resource(777, Domain.MMIO, 4, (1,), name="reg")

    def reading(module, *cids):
        for cid in cids:
            c = _claim_of(module, cid)
            module = _swap_claim(module, cid, replace(c, rd=(*c.rd, 777)))
        return replace(module, resources={**module.resources, 777: reg})

    dsrc = reading(src, producer)
    out.append(
        (
            "remat.device_input",
            "MV7",
            dsrc,
            spec,
            reading(mod, producer, clone),
            replace(plan, source_hash=source_digest(dsrc)),
        )
    )
    return out


# --- the rows -------------------------------------------------------------------------------


def implicit_cross_tier() -> tuple[int, int]:
    """(D-R2 claims of the example programs before, after the planner)."""
    from ..examples import PROGRAMS
    from ..kbcir.device_manifest import check_bank_moves
    from ..kbcir.movement import MovementSpec, plan_movement

    h, theta = target_and_theta()
    spec = MovementSpec(hardware())
    before = after = 0
    for name in sorted(PROGRAMS):
        module = PROGRAMS[name]()
        before += len(check_bank_moves(module))
        mp = plan_movement(module, spec, h, theta)
        after += len(check_bank_moves(mp.best.transform.module))
    return before, after


def _unlawful(module, spec, pm, h) -> bool:
    """Whether any law refuses the plan of `pm`: the movement laws with their source and spec
    (inside `verify_execution_plan`), R1-R25 over the transformed module, or a racing access."""
    from ..kbcir.movement import execution_plan_of, races
    from ..verify import verify_all, verify_execution_plan

    m2 = pm.transform.module
    plan = execution_plan_of(pm, h)
    return bool(
        verify_execution_plan(m2, plan, target=h, result=pm.result, movement=(module, spec))
        or verify_all(m2, pm.result)
        or races(m2)
    )


def parity_mismatch(exe: str, tmp: str) -> int:
    """`movement.parity.mismatch` with the plan harness `exe` (plan_fixtures.build_harness)."""
    from ..abi import encode as encode_pack
    from ..abi.execution_plan_abi import encode_plan
    from ..gem.streampack import generation_vector, hydrate
    from ..kbcir.movement import execution_plan_of
    from ..verify import verify_execution_plan
    from .plan_fixtures import c_refuses, parse_c_dump, run_harness, v3_variants, wire_refuses

    h, _theta = target_and_theta()
    mismatch = 0
    for _name, (module, spec, mp) in planned().items():
        plan = execution_plan_of(mp.best, h)
        blob = encode_plan(plan)
        m2 = mp.best.transform.module
        pack = hydrate(m2, mp.best.result, "plan0")
        code, out = run_harness(
            exe, tmp, blob, dump=True, pack_bytes=encode_pack(pack), live=generation_vector(m2)
        )
        c_ok = (
            code == 0
            and "\nOK\n" in out + "\n"
            and "pack=BCIR_OK" in out
            and "vector=BCIR_OK" in out
            and encode_plan(parse_c_dump(out)) == blob
        )
        py_ok = not verify_execution_plan(m2, plan, pack=pack, movement=(module, spec))
        mismatch += not (c_ok and py_ok)
    for _name, blob, _plan in v3_variants():
        mismatch += not (wire_refuses(blob) and c_refuses(exe, tmp, blob))
    return mismatch


def measure(baseline: str | None = None, cc: bool = True) -> dict[str, float]:
    """The rows (module docstring). `baseline` reads `movement.excess` off one of the planner's
    baselines instead (`greedy`: the pre-G8 transfer-aware rule -- the RED value). With `cc`, the
    parity row is measured when a C compiler builds the plan harness, and left out otherwise
    (never zero by default)."""
    import os
    import tempfile

    from ..abi.execution_plan_abi import decode_plan, encode_plan
    from ..gem.execution_plan import plan_from_realization
    from ..kbcir.movement import execution_plan_of, verify_movement
    from .plan_fixtures import C_DIR, asn1_accepted, build_harness

    h, _theta = target_and_theta()
    rows: dict[str, float] = {}
    _before, after = implicit_cross_tier()
    rows["movement.implicit_cross_tier"] = float(after)
    excess = drift = roundtrip = unlawful = 0
    for _name, (module, spec, mp) in planned().items():
        best = mp.best
        optimum = joint_optimum(module, spec)
        chosen = best if baseline is None else mp.baselines[baseline]
        excess += (chosen.makespan - optimum.makespan) if chosen is not None else 0
        ep = execution_plan_of(best, h)
        if decode_plan(encode_plan(ep)) != ep:
            roundtrip += 1
        if not best.transform.moves and best.transform.module == module:
            plain = plan_from_realization(module, best.result, h, "eft", static_plan=best.static)
            if encode_plan(plain) != encode_plan(ep):
                drift += 1
        unlawful += _unlawful(module, spec, best, h)
    rows["movement.excess"] = float(excess)
    try:
        variants = law_variants()
    except VariantUnavailable:
        variants = None  # the planned corpus lost an edge: the law rows go unmeasured (a finding)
    if variants is not None:
        accepted = misattributed = 0
        for _name, law, src, spec, mod, plan in variants:
            laws = {d.law for d in verify_movement(src, spec, mod, plan)}
            if not laws:
                accepted += 1
            elif law not in laws:
                misattributed += 1
        rows["movement.laws.accepted"] = float(accepted)
        rows["movement.laws.misattributed"] = float(misattributed)
    rows["movement.identity.drift"] = float(drift)
    rows["movement.roundtrip.mismatch"] = float(roundtrip)
    rows["movement.plans.unlawful"] = float(unlawful)
    rows["movement.asn1.accepted"] = float(asn1_accepted()[0])
    if cc and os.path.isdir(C_DIR):  # the checkout's runtime/c: an installed package has none
        with tempfile.TemporaryDirectory() as tmp:
            # None only without a compiler (an absence, reported as NOT-MEASURED); a twin that
            # does not build raises -- a finding, never a skip (L2)
            exe = build_harness(tmp)
            if exe is not None:
                rows["movement.parity.mismatch"] = float(parity_mismatch(exe, tmp))
    return rows
