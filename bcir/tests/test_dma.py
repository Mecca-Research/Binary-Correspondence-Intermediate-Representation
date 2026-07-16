"""A2 (`kbcir/dma.py`): DMA programming from StridedViews -- D-R4's other half.

The stride vector IS the scatter-gather program: a fully-contiguous view pair compiles
to ONE descriptor, a gappy outer stride to one descriptor per row, a non-unit innermost
stride degrades honestly to per-element scatter. Refusals are D-R4 live (inadmissible
view, shape/width mismatch, same-bank overlap). Pricing is D-R3 (distance-priced bytes
+ per-descriptor setup, so fragmentation genuinely costs). And the transfer module
composes the whole law stack -- explicit mem.move (D-R2), event-phase completion (A1),
masked-window reaping (B1/EV3) -- into one R-law-clean, priced, hydratable program."""

from bcir.kbcir.device_manifest import StridedView, check_bank_moves
from bcir.kbcir.dma import Descriptor, dma_cost, dma_descriptors, dma_transfer_module
from bcir.kbcir.events import check_event_phases
from bcir.tests.test_device_manifest import _MAN


def _view(bank: str, off: int = 0, shape=(16, 32), strides=(32, 1)) -> StridedView:
    return StridedView(bank=bank, offset_bytes=off, shape=shape, strides=strides)


def test_contiguity_is_read_straight_off_the_stride_vector():
    """One descriptor for a dense pair; one per row when the source rows are gapped;
    per-element scatter when the innermost stride is non-unit -- and byte totals are
    conserved in every shape."""
    dense = dma_descriptors(_view("hbm0"), _view("hbm1"), _MAN)
    assert dense == [Descriptor(src_off=0, dst_off=0, nbytes=16 * 32 * 4)]
    gappy = dma_descriptors(_view("hbm0", strides=(64, 1)), _view("hbm1"), _MAN)
    assert len(gappy) == 16 and all(d.nbytes == 32 * 4 for d in gappy)
    assert [d.src_off for d in gappy] == [256 * i for i in range(16)]   # src walks gapped
    assert [d.dst_off for d in gappy] == [128 * i for i in range(16)]   # dst stays dense
    scatter = dma_descriptors(_view("hbm0", shape=(16, 16), strides=(64, 2)),
                              _view("hbm1", shape=(16, 16), strides=(16, 1)), _MAN)
    assert len(scatter) == 256 and all(d.nbytes == 4 for d in scatter)
    assert [d.src_off for d in scatter[:17]] == [8 * i for i in range(16)] + [256]
    assert [d.src_off for d in scatter[-16:]] == [15 * 256 + 8 * i for i in range(16)]
    assert len({d.src_off for d in scatter}) == 256
    for descs in (dense, gappy, scatter):
        assert sum(d.nbytes for d in descs) in (2048, 1024)       # 16x32x4 or 16x16x4


def test_refusals_are_d_r4_live():
    """The compiler refuses BEFORE any descriptor exists: a fragmenting 15-wide view
    (the native-tile law), a shape mismatch, an element-width mismatch, and a same-bank
    overlap (in-place DMA is the corruption shape)."""
    cases = (
        (_view("hbm0", shape=(15, 15), strides=(15, 1)), _view("hbm1", shape=(15, 15),
         strides=(15, 1)), "native tile 16"),
        (_view("hbm0"), _view("hbm1", shape=(16, 16), strides=(16, 1)), "shape mismatch"),
        (_view("hbm0"), StridedView(bank="hbm1", offset_bytes=0, shape=(16, 32),
         strides=(32, 1), elem_bytes=8), "width mismatch"),
        (_view("hbm0"), _view("hbm0", off=256), "overlap"),
    )
    for src, dst, needle in cases:
        try:
            dma_descriptors(src, dst, _MAN)
            raise AssertionError(f"expected the {needle!r} refusal")
        except ValueError as e:
            assert needle in str(e), (needle, str(e))
    apart = dma_descriptors(_view("hbm0"), _view("hbm0", off=16 * 32 * 4), _MAN)
    assert len(apart) == 1                                        # disjoint same-bank: legal


def test_pricing_is_d_r3_with_a_fragmentation_premium():
    """Same bytes: the far pair (sram0->hbm1) costs more than the near pair
    (hbm0->hbm1); the gapped source costs more than the dense one (descriptor setup is
    real work); staying put prices only the setup."""
    dense = dma_descriptors(_view("hbm0"), _view("hbm1"), _MAN)
    near = dma_cost(_MAN, _view("hbm0"), _view("hbm1"), dense)
    far_v = _view("sram0")
    far = dma_cost(_MAN, far_v, _view("hbm1"), dma_descriptors(far_v, _view("hbm1"), _MAN))
    assert 0 < near < far
    gap_v = _view("hbm0", strides=(64, 1))
    gappy = dma_cost(_MAN, gap_v, _view("hbm1"), dma_descriptors(gap_v, _view("hbm1"), _MAN))
    assert gappy > near                                           # fragmentation costs


def test_invalid_descriptor_programs_cannot_underprice_or_copy_backwards():
    """Externally supplied descriptor lists are a trust boundary, not an opportunity
    to mint negative transfer/setup costs or zero-length commands."""
    src, dst = _view("hbm0"), _view("hbm1")
    for descs, setup in (([Descriptor(-1, 0, 4)], 16),
                         ([Descriptor(0, 0, 0)], 16),
                         ([Descriptor(0, 0, 4)], -1),
                         ([Descriptor(0, 0, 4)], 16),
                         ([Descriptor(True, 0, 4)], 16)):
        try:
            dma_cost(_MAN, src, dst, descs, desc_setup=setup)
            raise AssertionError("expected invalid descriptor/cost refusal")
        except ValueError:
            pass
    gapped = _view("hbm0", strides=(64, 1))
    collapsed = [Descriptor(0, 0, 16 * 32 * 4)]
    try:
        dma_cost(_MAN, gapped, dst, collapsed)
        raise AssertionError("a forged coalesced descriptor must not erase setup cost")
    except ValueError:
        pass


def test_the_transfer_module_composes_the_whole_law_stack():
    """D-R2 (the explicit mem.move rides the completion phase), A1/EV (completion is an
    event phase, armed by an explicit unmask), B1/EV3 (the reap claim reads the done
    flag inside a masked window), R-laws + bank moves clean, and the module prices and
    hydrates like any program."""
    from bcir.gem.streampack import hydrate
    from bcir.kbcir import TARGETS
    from bcir.kbcir.cost import Theta
    from bcir.kbcir.realize import optimize
    from bcir.verify import verify, verify_pack
    m = dma_transfer_module(_MAN, _view("sram0"), _view("hbm0"), engine="dma0")
    assert verify(m) == [], verify(m)
    assert check_event_phases(m) == [], check_event_phases(m)
    assert check_bank_moves(m) == [], check_bank_moves(m)
    ev = [p for p in m.phases if p.event]
    assert len(ev) == 1 and ev[0].event == "dma.done:dma0" and ev[0].deps == ()
    assert any(c.op.startswith("mem.move.far:sram0->hbm0") for c in ev[0].claims)
    pack = hydrate(m, optimize(m, TARGETS["x86_avx512"], Theta.cool()), plan=m.name)
    assert verify_pack(m, pack) == []
    # the unmasked-reap arm: strip the mask claims -> EV3 refuses (B1, measured).
    bare = dma_transfer_module(_MAN, _view("sram0"), _view("hbm0"))
    reap_phase = bare.phases[3]
    reap_phase.claims = [c for c in reap_phase.claims if not c.op.startswith("irq.")]
    msgs = check_event_phases(bare)
    assert any("EV3" in s and "masked window" in s for s in msgs), msgs


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
