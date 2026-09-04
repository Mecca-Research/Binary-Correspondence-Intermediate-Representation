"""Frozen Q8 memory-tier table baked into the C runtime via C23 `#embed`.

Roadmap §5 #2: `#embed` for frozen Q8 tables in the C runtime. The source of truth is
the oracle (`MemoryHierarchy.default()`); `bcir.abi.q8_tables` freezes it to a binary
blob + a generated header that `#embed`s the blob when the toolchain supports it and
falls back to the identical bytes otherwise. These tests guard the committed artifact:
the drift gate (committed == fresh emission), the value faithfulness, and that the
header builds + self-checks under C23 and C11.
"""

import os
import shutil
import struct
import subprocess
import tempfile

from bcir.abi import q8_tables
from bcir.kbcir.cost import MemTier, MemoryHierarchy

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME_C = os.path.join(_ROOT, "runtime", "c")


def test_blob_matches_the_oracle_table():
    blob = q8_tables.q8_tiers_blob()
    assert len(blob) == len(MemTier) * 4
    h = MemoryHierarchy.default()
    for i, t in enumerate(MemTier):
        bw, lat = struct.unpack_from("<HH", blob, i * 4)
        tier = h.by_name(t.name)
        assert (bw, lat) == (tier.bw_factor, tier.lat_factor), t.name


def test_committed_files_match_a_fresh_emission():
    """Drift gate (CI also enforces it in check_runtime.sh): the committed blob +
    header are a deterministic function of the oracle table."""
    with open(os.path.join(_RUNTIME_C, "q8_tiers.bin"), "rb") as f:
        assert f.read() == q8_tables.q8_tiers_blob(), "q8_tiers.bin is stale (--emit)"
    with open(os.path.join(_RUNTIME_C, "bcir_q8_tables.h"), encoding="utf-8") as f:
        assert f.read() == q8_tables.q8_tables_header(), "bcir_q8_tables.h is stale (--emit)"


def test_header_uses_embed_guarded_with_a_fallback():
    h = q8_tables.q8_tables_header()
    assert '#embed "q8_tiers.bin"' in h  # the C23 directive
    assert "defined(__has_embed)" in h  # feature-probed
    assert "#  if __has_embed(" in h  # ... in a NESTED #if (pre-#embed safe)
    assert "BCIR_Q8_EMBED 0" in h  # the fallback path
    assert "0x" in h  # the fallback array bytes
    assert "_Static_assert(sizeof(bcir_q8_tiers_blob)" in h


def test_header_builds_and_self_checks_under_c11_and_c23():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return  # skip cleanly without a C compiler
    src = os.path.join(_RUNTIME_C, "test_q8_tables.c")
    with tempfile.TemporaryDirectory() as d:
        for std in ("c11", "c23"):
            exe = os.path.join(d, f"q8_{std}")
            build = subprocess.run(
                [cc, f"-std={std}", "-Wall", "-Wextra", "-I", _RUNTIME_C, src, "-o", exe],
                capture_output=True,
                text=True,
            )
            assert build.returncode == 0, f"{std} build: {build.stderr}"
            run = subprocess.run([exe], capture_output=True, text=True)
            assert run.returncode == 0 and run.stdout.startswith("OK q8"), run.stdout + run.stderr
            # Older Clang/GCC take the byte-identical fallback; newer Clang exposes
            # __has_embed even in C11 extension mode. Pin the reported capability value,
            # not a stale compiler-version assumption.
            assert "embed=0" in run.stdout or "embed=1" in run.stdout
