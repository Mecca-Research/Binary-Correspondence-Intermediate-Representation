"""Rung 7 opener (`frontends/models/paged_kv.py`): paged KV + continuous batching.

PagedKV pages are registry Resources allocated as wave-11 StridedViews against a
DeviceManifest bank -- D-R4 LIVE at the serving layer (a fragmenting page size refuses
at construction, a ghost bank vetoes, an over-budget page table hits the bank capacity
wall); the `gem.kv_cache` capacity law runs live (`pos N exceeds capacity C`, the MLIR
verifier's words at runtime); the ids stay `decode_with_kv_cache`'s BIT-FOR-BIT because
the numerics ride the proven KVCache (paging is a registry story). Page data_gens ARE
R11's currency: a StreamPack hydrated over the pages goes stale the moment another
token lands. And `batched_sessions_module` + `BatchCertificate` measure-then-pin the
rung-7 thesis: continuous batching is wave scheduling over the merged claim graph."""

from bcir.frontends.models.decode import DecoderSpec, decode_with_kv_cache
from bcir.frontends.models.paged_kv import (BatchCertificate, PagedKV,
                                            batched_sessions_module, certify_batch,
                                            generate_paged, paged_session_module)
from bcir.kbcir import TARGETS
from bcir.kbcir.cost import Theta
from bcir.kbcir.device_manifest import check_strided_view
from bcir.tests.test_device_manifest import _MAN
from bcir.tests.test_model_decode import _toy_weights

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()
SPEC = DecoderSpec(vocab_size=64, d_model=8, n_heads=2, n_layers=2, d_ff=16)
# SPEC's page row: 2 layers x kv_dim 8 x (K+V) = 32 f32 elements -- 16-tile admissible.


def test_paged_ids_are_bit_for_bit_and_pages_account_every_write():
    """The paged path IS the cached path: same ids for a 2-page session; every position
    lands in its page (page 0 and page 1 each take exactly page_size writes); the views
    are admissible, disjoint, and one per page; eos stops with the serve.py law (the
    eos id is emitted, never fed back -- the last page's generation shows it)."""
    w = _toy_weights(SPEC)
    ids, kv = generate_paged([3, 9, 1], SPEC, w, 29, man=_MAN, bank="hbm0", page_size=16)
    assert ids == decode_with_kv_cache([3, 9, 1], SPEC, w, max_new=29)
    assert kv.pos == 32 == kv.capacity and kv.n_pages == 2
    assert [p.data_gen for p in kv.pages] == [16, 16]             # every write accounted
    assert [p.rid for p in kv.pages] == [7000, 7001]              # the page rid band
    assert all(check_strided_view(v, _MAN) == [] for v in kv.views)
    assert kv.views[0].offset_bytes == 0 and kv.views[1].offset_bytes == 16 * 32 * 4
    stopped, kv2 = generate_paged([3, 9, 1], SPEC, w, 29, man=_MAN, bank="hbm0",
                                  page_size=16, eos_id=ids[0])
    assert stopped == [ids[0]]
    assert kv2.pos == 3 and kv2.pages[0].data_gen == 3            # eos never fed back


def test_the_capacity_law_is_live_in_the_mlir_verifiers_words():
    """`gem.kv_cache`'s paging law at runtime: filling to capacity is legal; one more
    row raises with the verifier's own message shape (pos N exceeds capacity C / the
    paging lie). generate_paged sizes capacity = prompt + max_new, so it can never
    trip its own law (the budget is exact by construction)."""
    w = _toy_weights(SPEC)
    kv = PagedKV(SPEC, 16, 16, _MAN, "hbm0")
    row = w.embedding.row(1)
    for _ in range(16):
        kv.step_row(row, SPEC, w)
    assert kv.pos == 16
    try:
        kv.step_row(row, SPEC, w)
        raise AssertionError("expected the capacity refusal")
    except ValueError as e:
        assert "pos 17 exceeds capacity 16" in str(e) and "paging lie" in str(e)
    assert kv.pos == 16                                           # the refusal wrote nothing


def test_page_allocation_is_d_r4_live():
    """Construction IS allocation, and D-R4 refuses BEFORE any token flows: a 15-row
    page against the 16-native bank is the fragmentation law; a ghost bank is a veto;
    a page table larger than the bank hits the capacity wall (sram0 is 64 KiB; 33
    16-row pages of SPEC's 2 KiB want more)."""
    for cap, size, bank, needle in ((30, 15, "hbm0", "native tile 16"),
                                    (16, 16, "hbm9", "no bank"),
                                    (16 * 33, 16, "sram0", "capacity")):
        try:
            PagedKV(SPEC, cap, size, _MAN, bank)
            raise AssertionError(f"expected the {needle!r} refusal")
        except (ValueError, KeyError) as e:
            assert needle in str(e), (needle, str(e))


def test_page_generations_are_r11s_currency():
    """The registry-first paging pin: hydrate a StreamPack over a module that carries
    the live page resources -- clean; land ONE more token (a page data_gen bumps);
    the rebuilt module now flags the old pack R11-STALE (rehydrate: replan). Paging
    needs no new law: generation validity already speaks KV."""
    from bcir.gem.streampack import hydrate
    from bcir.kbcir.realize import optimize
    from bcir.verify import verify, verify_pack
    w = _toy_weights(SPEC)
    kv = PagedKV(SPEC, 32, 16, _MAN, "hbm0")
    row = w.embedding.row(1)
    for _ in range(5):
        kv.step_row(row, SPEC, w)                                 # mid-page-0: gens [5, 0]
    assert [p.data_gen for p in kv.pages] == [5, 0]
    m1 = paged_session_module(SPEC, 3, 2, kv)
    assert verify(m1) == [], verify(m1)                           # pages disturb no R-law
    pack = hydrate(m1, optimize(m1, AVX, COOL), plan=m1.name)
    assert verify_pack(m1, pack) == []
    kv.step_row(row, SPEC, w)                                     # one more token lands
    m2 = paged_session_module(SPEC, 3, 2, kv)
    stale = verify_pack(m2, pack)
    assert any(d.law == "R11" and "stale" in d.message and "data_gen" in d.message
               for d in stale), stale


def test_continuous_batching_is_wave_scheduling():
    """The rung-7 thesis, measured then pinned: three sessions merged into one module
    (disjoint rid bands, shared read-only WTS) verify law-clean, and the token DAG
    overlaps them with ZERO new machinery -- the pipelined makespan strictly beats the
    phase-barriered one (the overlap win IS continuous batching), while a single
    session has nothing to overlap (win == 0)."""
    from bcir.verify import verify
    sessions = ((3, 4), (2, 5), (4, 3))
    m = batched_sessions_module(SPEC, sessions)
    assert verify(m) == [], verify(m)
    rids = sorted(m.resources)
    assert rids[0] == 1 and 2 in rids and len(rids) == 1 + 3 * len(sessions)
    cert, sched = certify_batch(SPEC, sessions, AVX, COOL)
    assert isinstance(cert, BatchCertificate) and cert.admitted
    assert cert.pipelined < cert.barriered <= cert.serial         # batching genuinely wins
    assert cert.overlap_win > 0
    solo, _ = certify_batch(SPEC, ((3, 4),), AVX, COOL)
    assert solo.admitted and solo.overlap_win == 0                # nothing to overlap
    assert sched.makespan == cert.pipelined


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
