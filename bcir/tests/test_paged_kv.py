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
                                            generate_paged, paged_session_module,
                                            schedule_eviction)
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


def test_page_claim_wiring_makes_page_hazards_visible():
    """A3: the wired session module threads page RIDs through the claims -- the
    prefill writes page 0 (the 3-token prompt fits in it), the first decode step reads
    only page 0, the last one reads both and writes page 1. R-law clean, hydrates."""
    from bcir.gem.streampack import hydrate
    from bcir.kbcir.realize import optimize
    from bcir.verify import verify, verify_pack
    w = _toy_weights(SPEC)
    _, kv = generate_paged([3, 9, 1], SPEC, w, 29, man=_MAN, bank="hbm0", page_size=16)
    m = paged_session_module(SPEC, 3, 29, kv)
    assert verify(m) == [], verify(m)
    claims = {c.op: c for p in m.phases for c in p.claims if not p.event}
    decodes = [c for p in m.phases for c in p.claims if c.op == "gem.decode_step"]
    assert 7000 in claims["gem.prefill"].wr and 7001 not in claims["gem.prefill"].wr
    assert 7000 in decodes[0].rd and 7001 not in decodes[0].rd    # pos 3: page 0 only
    assert decodes[0].wr[-1] == 7000
    assert {7000, 7001} <= set(decodes[-1].rd)                    # pos 31: both pages
    assert decodes[-1].wr[-1] == 7001
    pack = hydrate(m, optimize(m, AVX, COOL), plan=m.name)
    assert verify_pack(m, pack) == []


def test_eviction_is_a_registry_act_and_a_scheduled_claim():
    """A3: evicting a live session refuses (full-context attention still reads every
    page -- the numerics lie); a completed session evicts (map_gen bumps, the view
    frees, writing through it afterwards refuses, double-evict refuses); the hydrated
    pack goes R11-stale as 'rehydrate: repack'; and `schedule_eviction` appends the
    same act as a lawful kv.evict claim."""
    from bcir.gem.streampack import hydrate
    from bcir.kbcir.realize import optimize
    from bcir.verify import verify, verify_pack
    w = _toy_weights(SPEC)
    live = PagedKV(SPEC, 32, 16, _MAN, "hbm0")
    row = w.embedding.row(1)
    live.step_row(row, SPEC, w)
    try:
        live.evict(0)
        raise AssertionError("expected the live-session refusal")
    except ValueError as e:
        assert "numerics lie" in str(e)
    _, kv = generate_paged([3, 9, 1], SPEC, w, 29, man=_MAN, bank="hbm0", page_size=16)
    m1 = paged_session_module(SPEC, 3, 29, kv)
    pack = hydrate(m1, optimize(m1, AVX, COOL), plan=m1.name)
    assert verify_pack(m1, pack) == []
    view = kv.evict(0)
    assert view is not None and kv.views[0] is None
    assert kv.pages[0].map_gen == 1
    stale = verify_pack(paged_session_module(SPEC, 3, 29, kv), pack)
    assert any(d.law == "R11" and "map_gen" in d.message and "repack" in d.message
               for d in stale), stale
    for act, needle in ((lambda: kv.evict(0), "already evicted"),
                        (lambda: kv.step_row(row, SPEC, w), "exceeds capacity")):
        try:
            act()
            raise AssertionError(f"expected the {needle!r} refusal")
        except ValueError as e:
            assert needle in str(e), (needle, str(e))
    m2 = schedule_eviction(paged_session_module(SPEC, 3, 29, kv), kv, 1)
    assert verify(m2) == [], verify(m2)
    tail = m2.phases[-1]
    assert tail.claims[0].op == "kv.evict:1" and tail.claims[0].rd == (7001,)
    assert tail.deps == (m2.phases[-2].phase_id,)                 # scheduled AFTER the tail


def test_admission_is_appending_phases():
    """A3: a session admitted to a LIVE 2-session batch produces the IDENTICAL claim
    graph to the 3-session batch built upfront (hash_module equality -- the strongest
    pin available), so the wave-scheduling certificate is unchanged by WHEN a session
    arrived; the admitted module still certifies pipelined < barriered."""
    from bcir.frontends.models.paged_kv import admit_session
    from bcir.kbcir.provenance import hash_module
    from bcir.verify import verify
    grown = batched_sessions_module(SPEC, ((3, 4), (2, 5)))
    admit_session(grown, SPEC, 4, 3)
    upfront = batched_sessions_module(SPEC, ((3, 4), (2, 5), (4, 3)))
    assert hash_module(grown) == hash_module(upfront)
    assert verify(grown) == [], verify(grown)
    cert, _ = certify_batch(SPEC, (), AVX, COOL, module=grown)
    assert cert.admitted and cert.pipelined < cert.barriered
    up_cert, _ = certify_batch(SPEC, ((3, 4), (2, 5), (4, 3)), AVX, COOL)
    assert (cert.serial, cert.barriered, cert.pipelined) == (
        up_cert.serial, up_cert.barriered, up_cert.pipelined)


def test_paged_boundaries_refuse_before_cache_or_graph_mutation():
    """A mismatched row/spec, out-of-range page, under-capacity graph, and colliding
    eviction ID all fail while the live cache/module remain unchanged."""
    import dataclasses
    from bcir.frontends.models.paged_kv import admit_session
    from bcir.model import Domain, Resource
    w = _toy_weights(SPEC)
    kv = PagedKV(SPEC, 16, 16, _MAN, "hbm0")
    before = (kv.pos, tuple(kv.pages), tuple(kv.views))
    wrong = dataclasses.replace(SPEC, rms_norm_eps=1e-5)
    huge = list(w.embedding.row(1))
    huge[0] = 10 ** 10000
    for action, needle in (
            (lambda: kv.step_row(w.embedding.row(1), wrong, w), "does not match"),
            (lambda: kv.step_row(huge, SPEC, w), "finite numbers"),
            (lambda: kv.page_of(-1), "position"),
            (lambda: paged_session_module(SPEC, 16, 1, kv), "capacity")):
        try:
            action()
            raise AssertionError(f"expected {needle!r} refusal")
        except ValueError as exc:
            assert needle in str(exc), (needle, str(exc))
    assert (kv.pos, tuple(kv.pages), tuple(kv.views)) == before

    module = paged_session_module(SPEC, 3, 2, kv)
    phase_count = len(module.phases)
    try:
        schedule_eviction(module, kv, 0, cid=1)
        raise AssertionError("expected colliding eviction claim refusal")
    except ValueError as exc:
        assert "already exists" in str(exc)
    assert len(module.phases) == phase_count

    batch = batched_sessions_module(SPEC, ((3, 2),))
    batch.add_resource(Resource(rid=99, domain=Domain.RAM, shape=(1,), name="injected"))
    snapshot = (batch.name, dict(batch.resources), tuple(batch.phases))
    try:
        admit_session(batch, SPEC, 2, 1)
        raise AssertionError("expected malformed live-batch refusal")
    except ValueError:
        pass
    assert (batch.name, batch.resources, tuple(batch.phases)) == snapshot

    batch = batched_sessions_module(SPEC, ((3, 2),))
    batch.phases[1].claims[0].op = "forged.decode"
    snapshot = (batch.name, dict(batch.resources), tuple(batch.phases))
    try:
        admit_session(batch, SPEC, 2, 1)
        raise AssertionError("expected forged live-batch graph refusal")
    except ValueError as exc:
        assert "phase/claim graph" in str(exc)
    assert (batch.name, batch.resources, tuple(batch.phases)) == snapshot


def test_batch_claim_bands_and_inventories_are_bounded():
    """A session longer than the former 1000-ID band cannot collide with its neighbor;
    a graph beyond the explicit reference limit is rejected before construction."""
    module = batched_sessions_module(SPEC, ((1, 901), (1, 901)))
    claim_ids = [claim.id for phase in module.phases for claim in phase.claims]
    assert len(claim_ids) == len(set(claim_ids))
    try:
        batched_sessions_module(SPEC, ((1, 1 << 16),))
        raise AssertionError("expected oversized batch refusal")
    except ValueError as exc:
        assert "claims" in str(exc)
    try:
        PagedKV(SPEC, (1 << 16) + 1, 1, _MAN, "hbm0")
        raise AssertionError("expected oversized page table refusal")
    except ValueError as exc:
        assert "pages" in str(exc)


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
