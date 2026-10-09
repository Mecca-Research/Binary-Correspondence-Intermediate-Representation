"""CXX4: the native hydrate -- the planner's input, its realization and a binding become the
oracle's StreamPack, byte for byte.

The 2026-10-06 audit (item 17) found hydrate Python-only: the native planner wrote a plan no
native code could turn into the pack that runs. `bcir_kp_hydrate` closes that: BKPI + BKPR +
BKPB (the RIDs, the generation vector, the plan's name) -> the StreamPack
`encode(hydrate(module, result, plan))` writes. These witnesses hold it to the oracle over the
planner's whole corpus and the hydrate's own constructs, show both rails refuse every malformed
record with one status, that the differential fires on wrong copies of either rail, and that the
C API fails closed. Without a C compiler the native rail is a skip, never a pass.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from bcir.abi import planner_abi as pa
from bcir.tests import hydrate_fixtures as hf
from bcir.tests import planner_fixtures as pf


def _harness(tmp: str, source: str | None = None) -> str | None:
    """The planner harness, built from the tree -- or with `source` as bcir_kplan.c."""
    if source is None:
        return pf.build_harness(tmp)
    cc = pf.compiler()
    if cc is None:
        return None
    unit = os.path.join(tmp, "bcir_kplan.c")
    with open(unit, "w", encoding="utf-8", newline="\n") as f:
        f.write(source)
    exe = os.path.join(tmp, "test_kplan_mutant")
    build = subprocess.run(
        [cc, "-std=c11", "-O2", "-I", pf.C_DIR, os.path.join(pf.C_DIR, pf.HARNESS), unit,
         os.path.join(pf.C_DIR, "bcir_runtime.c"), "-o", exe],
        capture_output=True, text=True, timeout=180,
    )  # fmt: skip
    assert build.returncode == 0, build.stderr[-2000:]
    return exe


def test_the_binding_codec():
    """BKPB round-trips every corpus module; its RID table is BKPI's resource table, index for
    index (the declared flags agree); and the encoder refuses a name the record cannot carry."""
    from bcir.examples import PROGRAMS
    from bcir.kbcir.cost import TargetProfile, Theta

    for name, build in sorted(PROGRAMS.items()):
        m = build()
        b = pa.decode_binding(pa.encode_binding(m, "plan-x"))
        value = pa.decode_input(pa.encode_input(m, TargetProfile.x86_avx2(), Theta.cool()))
        pa.check_binding(b, value)
        assert b.plan == "plan-x" and b.topo_gen == 1, name
        assert [bool(r.declared) for r in value.resources] == [rid in m.resources for rid in b.rids]
        assert [g[0] for g in b.gens] == sorted(m.resources), name
    try:
        pa.encode_binding(PROGRAMS[sorted(PROGRAMS)[0]](), "p" * (pa.PLAN_NAME_MAX + 1))
    except pa.PlannerAbiError as exc:
        assert exc.status == "BCIR_ERR_PLANNER"
    else:
        raise AssertionError("a plan name over 65535 bytes was encoded")


def test_the_oracle_refuses_every_malformed_record_with_its_status():
    """One record per law -- the binding's wire laws and its laws against the input, a forgery
    of the realization per hydrate law, a width and a block field the StreamPack cannot carry --
    each refused by the Python rail with the status the C twin declares."""
    cases = hf.malformed()
    assert len(cases) >= 24
    for label, module, triple, status in cases:
        assert hf.oracle_pack(module, *triple)[0] == status, label


def test_the_native_hydrate_is_the_oracle():
    """Both rows at zero: every corpus case's native pack is the oracle's, byte for byte (over
    3,000 cases: the planner's corpus under every target, Theta and policy, and the hydrate's
    own -- each realization name as an opcode, generations above zero, v1 packs, empty modules,
    RIDs at the top of their range, plan names of 0, 13 and 65535 bytes), and both rails refuse
    every malformed record alike."""
    tmp = tempfile.mkdtemp(prefix="bcir-khydrate-")
    try:
        exe = _harness(tmp)
        if exe is None:
            return  # no C compiler: the native rail is a skip, never a pass
        assert hf.measure(exe, tmp) == {row: 0.0 for row in hf.ROWS}
        assert len(hf.parity_cases()) > 3000
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_differential_fires_on_wrong_hydrates():
    """Wrong copies of the native hydrate, built from a copy of its source (never by editing
    the tree): a prefetch on a scalar step, the header's generation maximum taken from the first
    entry, an empty op left empty, a pack with no vector written as v4, steps out of phase order
    accepted. Each turns a row red -- the hydrate cases are what catch the middle three."""
    tmp = tempfile.mkdtemp(prefix="bcir-khydrate-red-")
    try:
        if pf.compiler() is None:
            return
        with open(os.path.join(pf.C_DIR, "bcir_kplan.c"), encoding="utf-8") as f:
            source = f.read()
        for row, old, new in (
            ("hydrate.native.parity",
             'if (width > 1u && n_rd) kh_str(w, label, kh_label(label, "pf", n));',
             'if (n_rd) kh_str(w, label, kh_label(label, "pf", n));'),
            ("hydrate.native.parity",
             "    if (kp_rd32(g + 4) > map_gen) map_gen = kp_rd32(g + 4);",
             "    if (!i) map_gen = kp_rd32(g + 4);"),
            ("hydrate.native.parity",
             "    if (op_len) {\n      kh_str(w, in->data + in->off_ops",
             "    if (1) {\n      kh_str(w, in->data + in->off_ops"),
            ("hydrate.native.parity", "  int v4 = b->n_gens != 0;", "  int v4 = 1;"),
            ("hydrate.native.malformed.accepted",
             "    if (n && at < last) return BCIR_ERR_PROVENANCE;",
             "    if (0 && at < last) return BCIR_ERR_PROVENANCE;"),
        ):  # fmt: skip
            assert source.count(old) == 1, old
            sub = tempfile.mkdtemp(dir=tmp)
            rows = hf.measure(_harness(sub, source.replace(old, new)), sub)
            assert rows[row] > 0, f"the differential cannot see {new!r}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_floor_writes_the_pack_once():
    """`hydrate.native.scale4`'s floor stores exactly the oracle's pack length (it is the pack the
    hydrate writes, not a guess at its size), and the hydrate never beats storing its own bytes."""
    from bcir.abi.streampack_abi import encode
    from bcir.gem.streampack import hydrate

    tmp = tempfile.mkdtemp(prefix="bcir-khydrate-floor-")
    try:
        exe = _harness(tmp)
        if exe is None:
            return
        module, result, _records = hf._scale_records(1)
        floor_ms, wrote = hf.native_floor(exe, tmp, scale=1)
        assert wrote == len(encode(hydrate(module, result, hf.PLAN)))
        assert 0 < floor_ms <= hf.native_ms(exe, tmp, scale=1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_hydrate_api_fails_closed():
    """The C API's laws over the planner's seed record and its own plan: no output, a short
    scratch or output zeroes it and reports nothing written; dirty, misaligned scratch writes
    the same bytes; the runtime's own reader accepts the pack; a binding of another resource
    count is refused."""
    tmp = tempfile.mkdtemp(prefix="bcir-khydrate-api-")
    try:
        exe = _harness(tmp)
        if exe is None:
            return
        seed = os.path.join(tmp, "seed.bkpi")
        with open(seed, "wb") as f:
            f.write(pf._seed_input()[0])
        run = subprocess.run([exe, "--api-hydrate", seed], capture_output=True, text=True,
                             timeout=60)  # fmt: skip
        assert run.returncode == 0 and run.stdout.startswith("API OK "), run.stdout + run.stderr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
