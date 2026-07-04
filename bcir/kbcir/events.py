"""A1 + B1 (driver roadmap Part VII): EVENT PHASES -- first-class asynchronous entry.

The gap this closes: BCIR's phase DAG stated PROGRAM order only, so an interrupt
handler had no honest representation -- it is a phase whose trigger is an EVENT SOURCE,
not another phase. `Phase.event` (model rail) names that source; this module carries
the three laws that make it safe (caller-passed messages, the D-R pattern -- [] == ok):

  * **EV1 (asynchronous entry)**: an event phase declares NO phase deps. Its ordering
    against the program flow comes from hazards + masking, never from program order --
    deps on an event phase are the "interrupts are subroutines" lie.
  * **EV2 (explicit arming)**: every event source named by an event phase must be armed
    by an explicit `irq.unmask:<src>` claim in the program flow. Enablement is a CLAIM
    over the controller resource (IER/IMR/...), riding the ordinary hazard discipline
    -- never an implicit side effect. (`irq.mask:*`/`irq.unmask:*` claims write exactly
    ONE controller resource and read none -- the well-formedness sub-law.)
  * **EV3 (the B1 interrupt-context ordering seam)**: any resource WRITTEN by an event
    phase may be touched by program claims only (a) inside a masked window for that
    source (`irq.mask:<src>` ... `irq.unmask:<src>`), or (b) as a `Lane.A` atomic
    claim. §5.14's volatile/atomic laws police each single access; THIS law orders the
    two contexts against each other -- the classic lost-update / torn-read seam between
    a handler and the flow it interrupts.

First fixture: the 16550/16750 RX interrupt path (UART blueprint U4) -- the handler
reads IIR (dispatch) + RBR and fills a ring buffer; the consumer drains it under mask.

Recorded first-slice bounds (not hidden): EV3 walks program phases in module LIST
order -- construction order IS program order in every existing builder; a
path-sensitive DAG walk is deferred until a fixture needs branching program flow.
Handler-vs-handler sharing (two sources, one resource) is likewise deferred -- the
16550 has one line. Non-disturbance: `Phase.event` defaults to "", making every check
here VACUOUS over the entire existing corpus (measured in the tests). Cost-side
module: imports no verifier (two-truth); the tests verify."""

from __future__ import annotations

from ..model import Claim, Domain, Lane, Module, Opcode, StrideClass

_MASK, _UNMASK = "irq.mask:", "irq.unmask:"


def mask_claim(source: str, ctrl_rid: int, cid: int) -> Claim:
    """An explicit `irq.mask:<source>` claim: one atomic-hazard volatile write to the
    controller resource. Disabling delivery is a program ACTION with a place in the
    hazard graph, never a side effect."""
    return _ctl(_MASK + source, ctrl_rid, cid)


def unmask_claim(source: str, ctrl_rid: int, cid: int) -> Claim:
    """The arming twin: `irq.unmask:<source>` (EV2's required claim)."""
    return _ctl(_UNMASK + source, ctrl_rid, cid)


def _ctl(op: str, ctrl_rid: int, cid: int) -> Claim:
    return Claim(id=cid, opcode=Opcode.STORE, lane=Lane.U, stride_class=StrideClass.SCALAR,
                 count=1, wr=(ctrl_rid,), op=op, domain=Domain.MMIO, hazard="atomic",
                 volatile=True)


def _source_of(op: str) -> str:
    return op.split(":", 1)[1]


def check_event_phases(module: Module) -> list:
    """EV1-EV3 (caller-passed messages, [] == lawful). Vacuous when no phase carries an
    event source -- the entire pre-A1 corpus."""
    msgs: list = []
    event_phases = [p for p in module.phases if getattr(p, "event", "")]
    program = [p for p in module.phases if not getattr(p, "event", "")]
    # mask/unmask well-formedness holds wherever the claims appear (even eventless
    # modules may mint them ahead of their handler).
    for p in module.phases:
        for c in p.claims:
            if c.op.startswith((_MASK, _UNMASK)) and (len(c.wr) != 1 or c.rd):
                msgs.append(f"EV: claim {c.id} ({c.op}) must write exactly ONE "
                            f"controller resource and read none; got rd={tuple(c.rd)} "
                            f"wr={tuple(c.wr)}")
    if not event_phases:
        return msgs
    # EV1: asynchronous entry has no program-order predecessor.
    for p in event_phases:
        if p.deps:
            msgs.append(f"EV1: event phase {p.phase_id} (source {p.event!r}) declares "
                        f"phase deps {tuple(p.deps)} -- asynchronous entry has no "
                        "program-order predecessor (hazards + masking order it)")
    # EV2: every named source is explicitly armed in the program flow.
    armed = {_source_of(c.op) for p in program for c in p.claims
             if c.op.startswith(_UNMASK)}
    for s in sorted({p.event for p in event_phases} - armed):
        msgs.append(f"EV2: event source {s!r} is never armed -- enablement must be an "
                    f"explicit irq.unmask:{s} claim in the program flow, never implicit")
    # EV3 (B1): the handler-shared resources are touched only under mask or atomically.
    handler_writes: dict = {}
    for p in event_phases:
        for c in p.claims:
            for rid in c.wr:
                handler_writes.setdefault(rid, set()).add(p.event)
    masked: set = set()
    for p in program:                # list order IS program order (first slice, recorded)
        for c in p.claims:
            if c.op.startswith(_MASK):
                masked.add(_source_of(c.op))
                continue
            if c.op.startswith(_UNMASK):
                masked.discard(_source_of(c.op))
                continue
            if c.lane == Lane.A:     # single-claim atomicity is the other legal shape
                continue
            for rid in c.io_rids():
                for src in sorted(handler_writes.get(rid, frozenset()) - masked):
                    msgs.append(f"EV3: claim {c.id} ({c.op or c.opcode.name}) touches "
                                f"resource {rid}, which the {src!r} handler writes, "
                                "outside a masked window -- mask the source around it "
                                "or make the touch a Lane.A atomic (the interrupted "
                                "flow must order against the handler)")
    return msgs
