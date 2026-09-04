"""The provenance-digest C twin (runtime/c/bcir_provenance.{h,c}): the FNV-1a manifest chain
byte-identical to bcir/kbcir/provenance.py, so a driver verifies a recorded ProvenanceManifest
with NO Python on the attestation path (R13 off-host).

The differential compiles a tiny C harness against the twin and compares: the raw item chain
(strings + signed integers incl. INT64_MIN), a real manifest digest (a planned vector_add's
component hashes + normalized artifacts, recomputed by the twin from the same inputs), and the
verify entry point (accepts the recorded digest, rejects a flipped artifact generation).
Toolchain-gated: a clean documented skip without a C compiler."""

import os
import shutil
import subprocess
import tempfile

from bcir.examples import vector_add
from bcir.kbcir import TARGETS
from bcir.kbcir.cost import Theta
from bcir.kbcir.provenance import _fnv, build_manifest

_RUNTIME_C = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runtime", "c"
)


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


_HARNESS = r"""
#include <stdio.h>
#include <inttypes.h>
#include "bcir_provenance.h"

int main(void) {
    /* (1) the raw item chain: mixed strings + signed integers, INT64_MIN included */
    uint64_t h = bcir_prov_begin();
    h = bcir_prov_item_str(h, "cal_fp");
    h = bcir_prov_item_i64(h, 12345);
    h = bcir_prov_item_i64(h, -7);
    h = bcir_prov_item_i64(h, 0);
    h = bcir_prov_item_i64(h, INT64_MIN);
    printf("chain %" PRIu64 "\n", bcir_prov_end(h));

    /* (2) the manifest digest from the component hashes + normalized artifacts */
    bcir_prov_artifact arts[2] = { {"cal_gen", ART0}, {"gate", ART1} };
    uint64_t d = bcir_prov_digest(MM, MT, MTH, MP, arts, 2);
    printf("digest %" PRIu64 "\n", d);
    printf("verify_ok %d\n", bcir_prov_verify(RECORDED, MM, MT, MTH, MP, arts, 2));
    arts[0].value += 1;   /* a swapped decision-rule generation must change the digest */
    printf("verify_tampered %d\n", bcir_prov_verify(RECORDED, MM, MT, MTH, MP, arts, 2));
    printf("null_item %d\n", bcir_prov_item_str(h, NULL) == 0);
    printf("null_array %d\n", bcir_prov_digest(MM, MT, MTH, MP, NULL, 1) == 0);
    arts[0].name = NULL;
    printf("verify_null %d\n", bcir_prov_verify(0, MM, MT, MTH, MP, arts, 2));
    return 0;
}
"""


def test_prov_twin_matches_the_oracle_byte_for_byte():
    cc = _cc()
    if cc is None:
        return  # documented toolchain-gated skip

    m = build_manifest(
        vector_add(), TARGETS["x86_avx512"], Theta.cool(), artifacts=(("cal_gen", 3), ("gate", 71))
    )
    chain = _fnv("cal_fp", 12345, -7, 0, -(2**63))

    src = (
        _HARNESS.replace("MM", str(m.m_module))
        .replace("MTH", str(m.m_theta))
        .replace("MT", str(m.m_target))
        .replace("MP", str(m.m_policy))
        .replace("RECORDED", f"{m.digest}ULL")
        .replace("ART0", str(dict(m.artifacts)["cal_gen"]))
        .replace("ART1", str(dict(m.artifacts)["gate"]))
    )
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "prov_main.c")
        with open(c, "w", encoding="utf-8") as f:
            f.write(src)
        exe = os.path.join(d, "prov")
        r = subprocess.run(
            [
                cc,
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-I",
                _RUNTIME_C,
                c,
                os.path.join(_RUNTIME_C, "bcir_provenance.c"),
                "-o",
                exe,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, r.stderr
        out = subprocess.run([exe], capture_output=True, text=True, timeout=60).stdout.split()
    got = dict(zip(out[0::2], out[1::2]))
    assert int(got["chain"]) == chain, (got["chain"], chain)  # the raw FNV chain
    assert int(got["digest"]) == m.digest, (got["digest"], m.digest)  # the manifest digest
    assert got["verify_ok"] == "1"  # attestation accepts
    assert got["verify_tampered"] == "0"  # a rule swap is caught
    assert got["null_item"] == "1" and got["null_array"] == "1"  # invalid input is safe
    assert got["verify_null"] == "0"  # even recorded zero cannot pass


def test_prov_twin_artifact_order_is_the_normalized_manifest_order():
    # the oracle records artifacts sorted (name, value); the twin chains them in that order --
    # the same inputs UNSORTED would produce a different digest, so the contract is explicit.
    m = build_manifest(
        vector_add(), TARGETS["x86_avx512"], Theta.cool(), artifacts=(("zeta", 1), ("alpha", 2))
    )
    assert m.artifacts == (("alpha", 2), ("zeta", 1))


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
