"""The DEVICE MANIFEST + the driver-seam hardening laws (`kbcir/device_manifest.py`).

D-R1 static schema (digestable, tamper-refusing, veto-not-steer probes), D-R2 bank
typing (memory-tier crossings need an explicit `mem.move.*`; MMIO exempt -- measured
against the real GPIO driver fixture, whose MMIO/RAM load-store mixing is by design),
D-R3 distance-aware moves (the Q8 matrix prices near < far), D-R4 strided views only
(full stride vector required; native-tile fragmentation refused at plan time). The
corpus-vacuousness pin: every existing planned module (train step, streamed step,
decode session) passes `check_bank_moves` untouched -- non-disturbance measured."""

import os
import tempfile

from bcir.kbcir.device_manifest import (
    DeviceManifest,
    MemoryBank,
    StridedView,
    check_bank_moves,
    check_device_manifest,
    check_strided_view,
    load_device_manifest,
    move_claim,
    move_cost,
    probe_agree,
    save_device_manifest,
)
from bcir.model import Domain, Module, Phase, Resource

_MAN = DeviceManifest(
    device="toy_npu",
    banks=(
        MemoryBank("sram0", Domain.RAM, 1 << 16, native_tile=16, align=64),
        MemoryBank("hbm0", Domain.HBM, 1 << 24, native_tile=16, align=64),
        MemoryBank("hbm1", Domain.HBM, 1 << 24, native_tile=16, align=64),
    ),
    distance=((0, 512, 640), (512, 0, 256), (640, 256, 0)),
    target="x86_avx512",
    cal_gen=3,
)


def test_the_manifest_is_checked_digested_and_tamper_refused():
    """Well-formedness catches every lie class; the digest is deterministic; the
    envelope refuses wrong-kind / newer-schema / wrong-target / STALE / tampered."""
    import dataclasses
    import json

    assert check_device_manifest(_MAN) == []
    assert _MAN.digest == dataclasses.replace(_MAN).digest  # deterministic identity
    bad_dist = dataclasses.replace(_MAN, distance=((0, 512, 640), (512, 0, 256), (999, 256, 0)))
    assert any("asymmetric" in m for m in check_device_manifest(bad_dist))
    bad_cap = dataclasses.replace(_MAN, banks=_MAN.banks[:2] + (MemoryBank("hbm1", Domain.HBM, 0),))
    assert any("capacity" in m for m in check_device_manifest(bad_cap))
    dup = dataclasses.replace(_MAN, banks=_MAN.banks[:2] + (_MAN.banks[1],))
    assert any("duplicate" in m for m in check_device_manifest(dup))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dev.json")
        save_device_manifest(path, _MAN)
        back = load_device_manifest(path, expect_target="x86_avx512", expect_cal_gen=3)
        assert back == _MAN and back.digest == _MAN.digest
        for mutate, needle in (
            (lambda d: d.update(kind="nope"), "not a"),
            (lambda d: d.update(schema=99), "newer"),
            (lambda d: d.update(target="other"), "retrain"),
            (lambda d: d.update(cal_gen=9), "STALE"),
            (lambda d: d["banks"][0].__setitem__(2, 123), "tampered"),
        ):
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            mutate(doc)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            try:
                load_device_manifest(path, expect_target="x86_avx512", expect_cal_gen=3)
                raise AssertionError(f"expected a refusal containing {needle!r}")
            except ValueError as e:
                assert needle in str(e), (needle, str(e))
            save_device_manifest(path, _MAN)  # restore for the next case


def test_distance_aware_moves_price_near_below_far():
    """D-R3: the same bytes cost less between HBM peers (near) than SRAM->far-HBM; the
    minted op-codes carry the near/far distinction and land in the DESTINATION domain."""
    near = move_cost(_MAN, "hbm0", "hbm1", 4096)
    far = move_cost(_MAN, "sram0", "hbm1", 4096)
    assert 0 < near < far
    assert move_cost(_MAN, "hbm0", "hbm0", 4096) == 0  # staying put is free
    c_near = move_claim(_MAN, "hbm0", "hbm1", 10, 11, 4096, cid=1)
    c_far = move_claim(_MAN, "sram0", "hbm1", 12, 11, 4096, cid=2)
    assert c_near.op.startswith("mem.move.near:")
    assert c_far.op.startswith("mem.move.far:")
    assert c_near.domain == Domain.HBM == c_far.domain  # bytes land at the dst


def _two_bank_module(with_move: bool, malformed: bool = False) -> Module:
    from bcir.model import Claim, Lane, Opcode, StrideClass

    m = Module(name="two_bank")
    m.add_resource(Resource(rid=1, domain=Domain.RAM, shape=(64,), name="host"))
    m.add_resource(Resource(rid=2, domain=Domain.HBM, shape=(64,), name="dev"))
    m.add_resource(Resource(rid=3, domain=Domain.HBM, shape=(64,), name="dev2"))
    if with_move:
        c = move_claim(_MAN, "sram0", "hbm0", 1, 2, 64, cid=1)
        if malformed:  # two sources: not a move
            c = Claim(
                id=1,
                opcode=Opcode.ADD,
                lane=Lane.U,
                stride_class=StrideClass.UNIT,
                count=64,
                rd=(1, 3),
                wr=(2,),
                op="mem.move.far:sram0->hbm0",
                domain=Domain.HBM,
                bounds="assumed_safe",
            )
    else:  # an IMPLICIT crossing
        c = Claim(
            id=1,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=64,
            rd=(1,),
            wr=(2,),
            op="gem.matmul",
            domain=Domain.HBM,
            bounds="assumed_safe",
        )
    m.add_phase(Phase(phase_id=0, deps=(), claims=[c]))
    return m


def test_bank_typing_requires_explicit_moves_and_is_corpus_vacuous():
    """D-R2 three ways: an implicit RAM->HBM claim is flagged; the explicit mem.move is
    clean (and R-law clean); a two-source 'move' is malformed. And the law is VACUOUS
    over every existing planned module + the real MMIO driver fixture (measured
    non-disturbance: MMIO/RAM mixing in plain load/store is the driver norm)."""
    from bcir.verify import verify

    bad = _two_bank_module(with_move=False)
    msgs = check_bank_moves(bad)
    assert msgs and "explicit" in msgs[0] and "gem.matmul" in msgs[0]
    good = _two_bank_module(with_move=True)
    assert check_bank_moves(good) == []
    assert verify(good) == [], verify(good)  # the cast is R-law clean
    assert any("exactly one" in m for m in check_bank_moves(_two_bank_module(True, malformed=True)))
    # corpus vacuousness: the planned modules of D1/rung-6, untouched.
    from bcir.frontends.models.decode import DecoderSpec
    from bcir.frontends.models.serve import decode_session_module
    from bcir.kbcir.train_graph import TrainStepSpec, train_step_module, train_stream_module

    for mod in (
        train_step_module(TrainStepSpec()),
        train_stream_module(TrainStepSpec(), 2),
        decode_session_module(
            DecoderSpec(vocab_size=8, d_model=8, n_heads=2, n_layers=1, d_ff=4), 2, 2
        ),
    ):
        assert check_bank_moves(mod) == [], mod.name
    # ... and the REAL driver fixture: MMIO/RAM mixing stays legal (the exemption).
    from bcir.frontends.cfront import compile_unit
    from bcir.tests.test_c_executor import _RUNTIME_C

    src = open(os.path.join(_RUNTIME_C, "cfront_driver_gpio.c"), encoding="utf-8").read()
    r = compile_unit(src, check_clang=False, search_paths=[_RUNTIME_C])
    for fn in r.lowered.functions.values():
        assert check_bank_moves(fn.module) == [], fn.name


def test_strided_views_are_the_only_allocation_currency():
    """D-R4: a missing stride vector cannot allocate; a 15x15 view against the 16-native
    bank refuses (the fragmentation law); capacity and alignment are enforced; the
    admissible tiled view passes."""
    ok = StridedView(bank="hbm0", offset_bytes=0, shape=(4, 32, 32), strides=(1024, 32, 1))
    assert check_strided_view(ok, _MAN) == []
    ragged = StridedView(bank="hbm0", offset_bytes=0, shape=(4, 32, 32), strides=(1024, 32))
    assert any("FULL stride vector" in m for m in check_strided_view(ragged, _MAN))
    frag = StridedView(bank="hbm0", offset_bytes=0, shape=(15, 15), strides=(15, 1))
    msgs = check_strided_view(frag, _MAN)
    assert any("native tile 16" in m for m in msgs), msgs
    huge = StridedView(bank="sram0", offset_bytes=0, shape=(16, 4096), strides=(4096, 1))
    assert any("capacity" in m for m in check_strided_view(huge, _MAN))
    skew = StridedView(bank="hbm0", offset_bytes=7, shape=(16, 16), strides=(16, 1))
    assert any("alignment" in m for m in check_strided_view(skew, _MAN))
    for width in (0, -4, True):
        bad_width = StridedView(
            bank="hbm0", offset_bytes=0, shape=(16, 16), strides=(16, 1), elem_bytes=width
        )
        assert any("elem_bytes" in m for m in check_strided_view(bad_width, _MAN))


def test_loaded_manifest_must_be_semantically_valid_even_with_a_matching_digest():
    """A SHA over attacker-controlled JSON proves identity, not admissibility. Loading
    always reruns the bank/matrix validator before the manifest reaches planning."""
    import dataclasses
    import json

    malformed = dataclasses.replace(
        _MAN, banks=(dataclasses.replace(_MAN.banks[0], align=3),) + _MAN.banks[1:]
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "self-consistent-but-invalid.json")
        save_device_manifest(path, _MAN)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        doc["banks"][0][4] = 3
        doc["digest"] = malformed.digest
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        try:
            load_device_manifest(path)
            raise AssertionError("expected semantic manifest refusal")
        except ValueError as exc:
            assert "power of two" in str(exc)


def test_manifest_loader_rejects_missing_digest_duplicates_and_type_coercion():
    import json

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "device.json")
        save_device_manifest(path, _MAN)
        with open(path, encoding="utf-8") as f:
            base = json.load(f)
        for mutate in (
            lambda doc: doc.pop("digest"),
            lambda doc: doc.update(cal_gen="3"),
            lambda doc: doc["banks"][0].__setitem__(2, True),
        ):
            doc = json.loads(json.dumps(base))
            mutate(doc)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            try:
                load_device_manifest(path)
                raise AssertionError("ambiguous/coerced manifest must be rejected")
            except ValueError:
                pass
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"kind":"bcir.device_manifest","kind":"other"}')
        try:
            load_device_manifest(path)
            raise AssertionError("duplicate JSON keys must be rejected")
        except ValueError as exc:
            assert "duplicate" in str(exc)

    import dataclasses

    malformed = dataclasses.replace(
        _MAN, banks=(dataclasses.replace(_MAN.banks[0], capacity_bytes="large"),) + _MAN.banks[1:]
    )
    assert any("capacity" in error for error in check_device_manifest(malformed))
    bad_offset = StridedView("hbm0", "zero", (16, 16), (16, 1))
    assert check_strided_view(bad_offset, _MAN)


def test_probes_may_veto_but_never_steer():
    """D-R1: an agreeing probe is silent; a capacity/tile disagreement REFUSES by
    message (the runtime never adapts); an unknown bank is a veto, not an adoption."""
    assert probe_agree(_MAN, {"hbm0": {"capacity_bytes": 1 << 24, "native_tile": 16}}) == []
    lies = probe_agree(_MAN, {"hbm0": {"capacity_bytes": 123}})
    assert lies and "REFUSE" in lies[0] and "do not adapt" in lies[0]
    ghost = probe_agree(_MAN, {"hbm9": {"capacity_bytes": 1}})
    assert ghost and "veto, never steer" in ghost[0]


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
