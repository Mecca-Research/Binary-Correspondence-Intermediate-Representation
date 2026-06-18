"""Fuzzing the trust boundaries (seeded by gen_module): valid inputs round-trip,
malformed inputs are rejected gracefully (no unexpected-exception crash). The
generator-seeded counterpart to the parity campaign."""

from bcir.kbcir.fuzz import (
    _fuzz_frontends,
    _fuzz_json,
    _fuzz_mlir,
    _fuzz_streampack,
    run_fuzz,
)
import os
import random


def test_fuzz_campaign_is_clean():
    # every trust boundary: valid round-trips + graceful malformed rejection. A fast smoke
    # check by default; CI runs the deep `fuzz -n 4000` (x2) + the libFuzzer/ASan C harness.
    findings = run_fuzz(n=1500 if os.environ.get("BCIR_THOROUGH") else 400, seed=20240601)
    assert findings == [], findings[:5]


def test_fuzz_is_deterministic():
    assert run_fuzz(n=120, seed=9) == run_fuzz(n=120, seed=9)


def test_each_boundary_independently_clean():
    rng = random.Random(3)
    for fn in (_fuzz_streampack, _fuzz_json, _fuzz_frontends, _fuzz_mlir):
        findings: list = []
        for _ in range(300):
            fn(rng, findings)
        assert findings == [], (fn.__name__, findings[:3])


def test_streampack_roundtrip_holds_on_valid_encodings():
    from bcir.abi import streampack_abi as abi
    from bcir.gem import hydrate
    from bcir.kbcir import TARGETS, optimize
    from bcir.kbcir.cost import Theta
    from bcir.kbcir.differential import gen_module

    rng = random.Random(1)
    for _ in range(50):
        m = gen_module(rng)
        pack = hydrate(m, optimize(m, TARGETS["x86_avx512"], Theta.cool()))
        back = abi.decode(abi.encode(pack))
        assert back.segments == pack.segments        # full per-field round-trip, not just ids
