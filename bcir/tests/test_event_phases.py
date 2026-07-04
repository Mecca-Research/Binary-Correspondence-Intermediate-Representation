"""A1 + B1 (`kbcir/events.py`): event phases -- first-class asynchronous entry.

The U4 fixture (16550/16750 RX interrupt path): a handler phase triggered by
`uart_rx` reads IIR (dispatch) + RBR and fills a ring buffer; the program-side
consumer drains it inside an explicit mask window. EV1 refuses deps on an event phase
(asynchronous entry has no program-order predecessor), EV2 refuses an unarmed source
(enablement is an explicit irq.unmask claim, never implicit), EV3 -- the B1
interrupt-context ordering seam -- refuses program touches of handler-written
resources outside a masked window unless the touch is a Lane.A atomic. The good
fixture is R-law clean, prices, and hydrates like any program; and every check is
VACUOUS over the existing corpus (non-disturbance, measured)."""

from bcir.kbcir.events import check_event_phases, mask_claim, unmask_claim
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

_IER, _IIR, _RBR, _RING, _HEAD, _TAIL = 1, 2, 3, 4, 5, 6


def _uart_rx_module(*, armed: bool = True, masked_consumer: bool = True,
                    atomic_consumer: bool = False, deps_on_event: bool = False) -> Module:
    """UART blueprint U4 as a claim graph. Program flow: init (arm the line), then the
    consumer draining the ring. Event phase (source `uart_rx`): IIR dispatch + RBR ->
    ring fill. The knobs build the negative arms."""
    m = Module(name="uart16550_rx_u4")
    for r in (Resource(rid=_IER, domain=Domain.MMIO, shape=(1,), name="IER"),
              Resource(rid=_IIR, domain=Domain.MMIO, shape=(1,), name="IIR"),
              Resource(rid=_RBR, domain=Domain.MMIO, shape=(1,), name="RBR"),
              Resource(rid=_RING, domain=Domain.RAM, shape=(16,), name="RING"),
              Resource(rid=_HEAD, domain=Domain.RAM, shape=(1,), name="HEAD"),
              Resource(rid=_TAIL, domain=Domain.RAM, shape=(1,), name="TAIL")):
        m.add_resource(r)
    init: list = [unmask_claim("uart_rx", _IER, cid=1)] if armed else []
    m.add_phase(Phase(phase_id=0, deps=(), claims=init))
    lane = Lane.A if atomic_consumer else Lane.U
    consume = Claim(id=3, opcode=Opcode.LOAD, lane=lane, stride_class=StrideClass.SCALAR,
                    count=1, rd=(_RING, _HEAD), wr=(_TAIL,), op="uart.consume",
                    domain=Domain.RAM,
                    hazard="atomic" if atomic_consumer else "unique")
    body = ([mask_claim("uart_rx", _IER, cid=2), consume,
             unmask_claim("uart_rx", _IER, cid=4)]
            if masked_consumer else [consume])
    m.add_phase(Phase(phase_id=1, deps=(0,), claims=body))
    dispatch = Claim(id=100, opcode=Opcode.LOAD, lane=Lane.U,
                     stride_class=StrideClass.SCALAR, count=1, rd=(_IIR,), wr=(),
                     op="uart.iir_dispatch", domain=Domain.MMIO, hazard="barriered",
                     volatile=True)
    fill = Claim(id=101, opcode=Opcode.LOAD, lane=Lane.U, stride_class=StrideClass.SCALAR,
                 count=1, rd=(_RBR,), wr=(_RING, _HEAD), op="uart.rx_fill",
                 domain=Domain.MMIO, hazard="barriered", volatile=True)
    m.add_phase(Phase(phase_id=10, deps=(0,) if deps_on_event else (),
                      claims=[dispatch, fill], event="uart_rx"))
    return m


def test_the_u4_fixture_is_lawful_prices_and_hydrates():
    """The good handler+consumer module: EV-clean, R-law clean, and an ordinary citizen
    of the cost rail (prices, hydrates to an R10/R11-clean pack) -- an event phase is a
    phase, not a new universe."""
    from bcir.gem.streampack import hydrate
    from bcir.kbcir import TARGETS
    from bcir.kbcir.cost import Theta
    from bcir.kbcir.realize import optimize
    from bcir.verify import verify, verify_pack
    m = _uart_rx_module()
    assert check_event_phases(m) == [], check_event_phases(m)
    assert verify(m) == [], verify(m)
    result = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    pack = hydrate(m, result, plan=m.name)
    assert verify_pack(m, pack) == []
    assert m.phases[2].event == "uart_rx" and m.phases[2].deps == ()


def test_ev1_refuses_program_deps_on_asynchronous_entry():
    msgs = check_event_phases(_uart_rx_module(deps_on_event=True))
    assert any("EV1" in s and "asynchronous entry" in s for s in msgs), msgs


def test_ev2_refuses_an_unarmed_source():
    """No irq.unmask anywhere in the program flow (the consumer goes atomic so no mask
    window arms it as a side effect): the source is never enabled -- refused."""
    m = _uart_rx_module(armed=False, masked_consumer=False, atomic_consumer=True)
    msgs = check_event_phases(m)
    assert msgs and all("EV2" in s for s in msgs), msgs
    assert "never armed" in msgs[0] and "uart_rx" in msgs[0]


def test_ev3_orders_the_interrupted_flow_against_the_handler():
    """The B1 seam three ways: an unmasked non-atomic consumer touching the
    handler-written RING + HEAD is refused (one message per touched resource); the
    masked window is clean; the Lane.A atomic touch is the other legal shape."""
    bare = check_event_phases(_uart_rx_module(masked_consumer=False))
    assert len([s for s in bare if "EV3" in s]) == 2, bare       # RING and HEAD
    assert all("masked window" in s for s in bare)
    assert check_event_phases(_uart_rx_module()) == []           # masked: clean
    atomic = _uart_rx_module(masked_consumer=False, atomic_consumer=True)
    assert check_event_phases(atomic) == []                      # atomic: clean


def test_mask_claims_are_well_formed_or_refused():
    """The controller claims write exactly ONE resource and read none -- a two-target
    'mask' is malformed wherever it appears (even in an eventless module)."""
    m = Module(name="ctl_shape")
    m.add_resource(Resource(rid=1, domain=Domain.MMIO, shape=(1,), name="IER"))
    m.add_resource(Resource(rid=2, domain=Domain.MMIO, shape=(1,), name="IMR"))
    bad = mask_claim("x", 1, cid=1)
    bad.wr = (1, 2)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[bad]))
    msgs = check_event_phases(m)
    assert msgs and "exactly ONE" in msgs[0], msgs


def test_the_laws_are_vacuous_over_the_existing_corpus():
    """Non-disturbance, measured: every existing planned module (train step, streamed
    step, decode session, batched sessions) and the real GPIO driver fixture carry no
    event phase and pass untouched."""
    import os
    from bcir.frontends.cfront import compile_unit
    from bcir.frontends.models.decode import DecoderSpec
    from bcir.frontends.models.paged_kv import batched_sessions_module
    from bcir.frontends.models.serve import decode_session_module
    from bcir.kbcir.train_graph import (TrainStepSpec, train_step_module,
                                        train_stream_module)
    from bcir.tests.test_c_executor import _RUNTIME_C
    spec = DecoderSpec(vocab_size=8, d_model=8, n_heads=2, n_layers=1, d_ff=4)
    for mod in (train_step_module(TrainStepSpec()),
                train_stream_module(TrainStepSpec(), 2),
                decode_session_module(spec, 2, 2),
                batched_sessions_module(spec, ((2, 2), (3, 1)))):
        assert check_event_phases(mod) == [], mod.name
    src = open(os.path.join(_RUNTIME_C, "cfront_driver_gpio.c"), encoding="utf-8").read()
    r = compile_unit(src, check_clang=False, search_paths=[_RUNTIME_C])
    for fn in r.lowered.functions.values():
        assert check_event_phases(fn.module) == [], fn.name


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
