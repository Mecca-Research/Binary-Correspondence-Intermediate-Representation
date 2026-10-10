"""The C StreamPack executor (runtime/c/bcir_exec.c): Python<->C parity + robustness.

Roadmap (BCIR_MASTER_ROADMAP.md §5.2 / §6): port the deterministic executor
(`bcir/gem/execute.py`) to C so the StreamPack is a no-Python hot artifact a driver runs
end to end. These tests pin the C executor's **dispatch order + per-phase telemetry**
against the oracle `gem.execute` over a Python-encoded pack (the executor's parity gate,
the analog of test_c_runtime for the decoder), and exercise the bounds/`NOSPACE` paths.
"""

import dataclasses
import os
import random
import shutil
import subprocess
import tempfile

from bcir.abi import encode
from bcir.abi.streampack_abi import decode
from bcir.examples import matmul_tiled, multi_histogram, scan, vector_add
from bcir.frontends.models.decode import DecoderSpec
from bcir.frontends.models.decoder_program import decoder_program
from bcir.gem import execute, hydrate
from bcir.gem.streampack import (
    Block,
    LaneSegment,
    Prefetch,
    StreamPack,
    TraceNote,
    hydrate_pipelined,
)
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME_C = os.path.join(_ROOT, "runtime", "c")
AVX = TargetProfile.x86_avx512()
COOL = Theta.cool()


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


#: The compiler these tests need, as a capability `tools/testing/check_tests.py --require` reads.
_CC = _cc()


def _out_of_order_module():
    """One phase whose claims are *declared* 30, 10, 20 -- the executor must dispatch
    them ascending by id (10, 20, 30), matching gem.execute's intra-phase id sort."""
    m = Module(name="ooo")
    for rid in range(1, 10):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(1024,)))
    claims = [
        Claim(
            id=i,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1024,
            rd=(1, 2),
            wr=(3,),
            op="vector.add",
            domain=Domain.RAM,
        )
        for i in (30, 10, 20)
    ]
    m.add_phase(Phase(phase_id=0, deps=(), claims=claims))
    return m


def _build_harness(tmp):
    cc = _cc()
    if cc is None:
        return None
    exe = os.path.join(tmp, "test_exec")
    r = subprocess.run(
        [
            cc,
            "-std=c23",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            _RUNTIME_C,
            os.path.join(_RUNTIME_C, "bcir_exec.c"),
            os.path.join(_RUNTIME_C, "bcir_runtime.c"),
            os.path.join(_RUNTIME_C, "test_exec.c"),
            "-o",
            exe,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return exe


def _run(exe, tmp, pack: bytes):
    path = os.path.join(tmp, "pack.bin")
    with open(path, "wb") as f:
        f.write(pack)
    r = subprocess.run([exe, path], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    order, phases = [], []
    for line in r.stdout.strip().splitlines():
        if line.startswith("order:"):
            order = [int(x) for x in line.split()[1:]]
        elif line.startswith("phase "):
            _, pid, sched, ex = line.split()
            phases.append((int(pid), int(sched), int(ex)))
    return order, phases


def test_c_executor_builds_freestanding():
    cc = _cc()
    if cc is None:
        return
    for std in ("c11", "c23"):
        r = subprocess.run(
            [
                cc,
                "-ffreestanding",
                "-nostdlib",
                f"-std={std}",
                "-Wall",
                "-Wextra",
                "-I",
                _RUNTIME_C,
                "-c",
                os.path.join(_RUNTIME_C, "bcir_exec.c"),
                "-o",
                os.devnull,
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"{std}: {r.stderr}"


def test_dispatch_order_and_telemetry_match_gem_execute():
    """The headline parity: over a real pack, the C executor's dispatch order + per-phase
    (scheduled, executed) + phase order equal bcir.gem.execute on the same module."""
    if _cc() is None:
        return
    mods = {
        "vector_add": vector_add(),
        "multi_histogram": multi_histogram(),
        "matmul_tiled": matmul_tiled(),
        "scan": scan(),
        "out_of_order": _out_of_order_module(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        assert exe is not None
        for name, m in mods.items():
            pack = encode(hydrate(m, optimize(m, AVX, COOL)))
            c_order, c_phases = _run(exe, tmp, pack)
            er = execute(m)
            assert c_order == er.order, f"{name}: order {c_order} != {er.order}"
            assert [p[0] for p in c_phases] == er.phase_order, f"{name}: phase order"
            py_phases = [(ps.phase_id, ps.scheduled, ps.executed) for ps in er.phases]
            assert c_phases == py_phases, f"{name}: telemetry {c_phases} != {py_phases}"


def test_intra_phase_sort_is_by_claim_id():
    """The declared order is 30,10,20; the executor must dispatch 10,20,30 (id order)."""
    if _cc() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        pack = encode(hydrate(_out_of_order_module(), optimize(_out_of_order_module(), AVX, COOL)))
        order, phases = _run(exe, tmp, pack)
        assert order == [10, 20, 30]
        assert phases == [(0, 3, 3)]


def test_malformed_pack_is_rejected_not_crashed():
    """A truncated / garbage pack returns a status (the harness prints ERR), never an
    out-of-bounds read (ASan/UBSan-fuzzed separately in tools/c/fuzz_streampack.sh)."""
    if _cc() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        good = encode(hydrate(vector_add(), optimize(vector_add(), AVX, COOL)))
        for bad in (b"", b"BSPK", good[:40], good[:-1], bytes(len(good))):
            path = os.path.join(tmp, "bad.bin")
            with open(path, "wb") as f:
                f.write(bad)
            r = subprocess.run([exe, path], capture_output=True, text=True)
            # Either a clean ERR <status> line or (for the rare valid prefix) a clean run;
            # the point is no crash / nonzero-from-signal.
            assert r.returncode in (0, 1), (bad[:8], r.returncode, r.stdout, r.stderr)


# --- QUAL-LIN: the R10 walk and the dispatch, linear on the packs hydrate writes -------------
#
# A decoder program is thousands of claims, and the walk that refuses a semantically corrupt
# pack (bcir_sp_verify_semantic) re-scanned the pack for every segment: 764 ms of a 765 ms run
# at 2900 claims. Its lookups now start where the previous one matched and its uniqueness checks
# rescan only for an id or a name out of order -- exact on every pack, linear on hydrate's. The
# first test holds the verdicts and the dispatch order to the pack laws over packs in every
# other order; the second holds the cost to linear.

_BCIR_ERR_PROVENANCE = 9  # runtime/c/bcir_runtime.h


def _run_any(exe, tmp, pack: bytes):
    """("ok", order, phases) or ("err", status) -- the harness's verdict on any pack."""
    path = os.path.join(tmp, "pack.bin")
    with open(path, "wb") as f:
        f.write(pack)
    r = subprocess.run([exe, path], capture_output=True, text=True)
    if r.returncode == 1 and r.stdout.startswith("ERR "):
        return ("err", int(r.stdout.split()[1]))
    assert r.returncode == 0, r.stdout + r.stderr
    order, phases = [], []
    for line in r.stdout.strip().splitlines():
        if line.startswith("order:"):
            order = [int(x) for x in line.split()[1:]]
        elif line.startswith("phase "):
            _, pid, sched, ex = line.split()
            phases.append((int(pid), int(sched), int(ex)))
    return ("ok", order, phases)


def _pack_laws_hold(pack: StreamPack) -> bool:
    """The pack-level R10 laws bcir_sp_verify_semantic holds a pack to, restated plainly:
    claim ids, trace ids and prefetch names each unique; every segment's claim traced; a
    declared prefetch exists and feeds at least one of the segment's reads."""
    seg_ids = [s.claim_id for s in pack.segments]
    trace_ids = [t.claim_id for t in pack.trace_notes]
    names = [pf.name for pf in pack.prefetches]
    if any(len(set(xs)) != len(xs) for xs in (seg_ids, trace_ids, names)):
        return False
    targets = {pf.name: set(pf.targets) for pf in pack.prefetches}
    for s in pack.segments:
        if s.claim_id not in set(trace_ids):
            return False
        if s.prefetch:
            if s.prefetch not in targets:
                return False
            if s.reads and not targets[s.prefetch] & set(s.reads):
                return False
    return True


def _gem_order(pack: StreamPack):
    """GEM's order over the pack: phases by first appearance, claims ascending within one."""
    rank: dict[int, int] = {}
    for s in pack.segments:
        rank.setdefault(s.phase_id, len(rank))
    order = sorted(pack.segments, key=lambda s: (rank[s.phase_id], s.claim_id))
    counts = {pid: 0 for pid in rank}
    for s in pack.segments:
        counts[s.phase_id] += 1
    return [s.claim_id for s in order], [(pid, n, n) for pid, n in counts.items()]


def _disorder(pack: StreamPack, rng: random.Random, kind: str) -> StreamPack:
    """`pack` with one disorder: a reordering the laws allow, or a violation they refuse."""
    p = dataclasses.replace(
        pack,
        segments=list(pack.segments),
        prefetches=list(pack.prefetches),
        blocks=list(pack.blocks),
        trace_notes=list(pack.trace_notes),
    )
    segs, pfs, notes = p.segments, p.prefetches, p.trace_notes

    def seg(i, **kw):
        segs[i] = dataclasses.replace(segs[i], **kw)

    def rename(new_name):
        renamed = {pf.name: new_name(i, pf.name) for i, pf in enumerate(pfs)}
        pfs[:] = [dataclasses.replace(pf, name=renamed[pf.name]) for pf in pfs]
        for i, s in enumerate(segs):
            if s.prefetch:
                seg(i, prefetch=renamed.get(s.prefetch, s.prefetch))

    with_pf = [i for i, s in enumerate(segs) if s.prefetch]
    if kind == "shuffle_segments":
        rng.shuffle(segs)
    elif kind == "reverse_segments":
        segs.reverse()
    elif kind == "shuffle_trace":
        rng.shuffle(notes)
    elif kind == "rotate_trace" and notes:
        k = rng.randrange(len(notes))
        notes[:] = notes[k:] + notes[:k]
    elif kind == "shuffle_prefetches":
        rng.shuffle(pfs)
    elif kind == "names_descending":  # the shortlex order backwards: every name rescans
        rename(lambda i, _: "z" * (len(pfs) - i))
    elif kind == "names_one_length":  # one length, octets descending, multi-octet UTF-8
        rename(lambda i, _: "\u00e9" + chr(0x4E00 + len(pfs) - i))
    elif kind == "claims_descending":
        ids = {s.claim_id: (1 << 64) - 1 - 7 * i for i, s in enumerate(segs)}
        for i, s in enumerate(segs):
            seg(i, claim_id=ids[s.claim_id])
        notes[:] = [dataclasses.replace(t, claim_id=ids.get(t.claim_id, t.claim_id)) for t in notes]
    elif kind == "relabel_phases":
        old = sorted({s.phase_id for s in segs})
        new = old[:]
        rng.shuffle(new)
        for i, s in enumerate(segs):
            seg(i, phase_id=dict(zip(old, new))[s.phase_id])
    elif kind == "reverse_within_phase":
        out, run = [], []
        for s in segs:
            if run and s.phase_id != run[-1].phase_id:
                out, run = out + run[::-1], []
            run.append(s)
        segs[:] = out + run[::-1]
    elif kind == "split_phase_run" and len(segs) > 2:
        segs.insert(rng.randrange(len(segs)), segs.pop(rng.randrange(len(segs))))
    elif kind == "repeated_segment" and segs:  # a copy right after: ties the greatest id
        k = rng.randrange(len(segs))
        segs.insert(k + 1, segs[k])
    elif kind == "repeated_trace" and notes:
        k = rng.randrange(len(notes))
        notes.insert(k + 1, notes[k])
    elif kind == "repeated_prefetch" and pfs:
        k = rng.randrange(len(pfs))
        pfs.insert(k + 1, pfs[k])
    elif kind == "duplicate_claim" and len(segs) > 1:
        i, j = rng.sample(range(len(segs)), 2)
        seg(j, claim_id=segs[i].claim_id)
    elif kind == "duplicate_trace" and len(notes) > 1:
        i, j = rng.sample(range(len(notes)), 2)
        notes[j] = dataclasses.replace(notes[j], claim_id=notes[i].claim_id)
    elif kind == "duplicate_name" and len(pfs) > 1:
        i, j = rng.sample(range(len(pfs)), 2)
        pfs[j] = dataclasses.replace(pfs[j], name=pfs[i].name)
    elif kind == "dropped_trace" and notes:
        notes.pop(rng.randrange(len(notes)))
    elif kind == "redirected_prefetch" and with_pf and pfs:
        seg(rng.choice(with_pf), prefetch=rng.choice(pfs).name)
    elif kind == "unresolved_prefetch" and with_pf:
        seg(rng.choice(with_pf), prefetch="ghost")
    elif kind == "emptied_targets" and pfs:
        i = rng.randrange(len(pfs))
        pfs[i] = dataclasses.replace(pfs[i], targets=())
    return p


_DISORDERS = (
    "shuffle_segments", "reverse_segments", "shuffle_trace", "rotate_trace",
    "shuffle_prefetches", "names_descending", "names_one_length", "claims_descending",
    "relabel_phases", "reverse_within_phase", "split_phase_run", "repeated_segment",
    "repeated_trace", "repeated_prefetch", "duplicate_claim", "duplicate_trace",
    "duplicate_name", "dropped_trace", "redirected_prefetch", "unresolved_prefetch",
    "emptied_targets",
)  # fmt: skip


def test_r10_and_the_dispatch_hold_to_the_pack_laws_in_every_order():
    """Hydrated packs, each disordered every way above, three disorders deep: the C walk
    accepts exactly the packs the restated laws accept (and refuses the rest as R10
    provenance), and runs each accepted one in GEM order with GEM's telemetry. The packs are
    the plain and the pipelined (double-buffer prefetch) hydrations of five programs -- a
    decoder program's 65 claims among them -- and a 24-claim pack of hydrate's shape, so every
    lookup is met in hydrate's order -- the linear paths -- and out of it -- the rescans."""
    if _cc() is None:
        return
    spec = DecoderSpec(
        vocab_size=64, d_model=32, n_heads=2, n_layers=1, d_ff=64, n_kv_heads=1,
        activation="silu_gate",
    )  # fmt: skip
    bases = [decode(_hydrated_shape(24))]
    for m in (
        vector_add(),
        multi_histogram(),
        matmul_tiled(),
        scan(),
        decoder_program(spec, 2, 3).module,
    ):
        r = optimize(m, AVX, COOL)
        bases += [hydrate(m, r), hydrate_pipelined(m, r, depth=2)]
    rng = random.Random(0x5EED_10)
    accepted = refused = 0
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        for base in bases:
            cases = [base] + [_disorder(base, rng, kind) for kind in _DISORDERS]
            for _ in range(6):
                p = base
                for kind in rng.sample(_DISORDERS, 3):
                    p = _disorder(p, rng, kind)
                cases.append(p)
            for pack in cases:
                got = _run_any(exe, tmp, encode(pack))
                if _pack_laws_hold(pack):
                    order, phases = _gem_order(pack)
                    assert got == ("ok", order, phases), (got, order, phases)
                    accepted += 1
                else:
                    assert got == ("err", _BCIR_ERR_PROVENANCE), got
                    refused += 1
    # Both verdicts were exercised, many times over (the corpus is not vacuous either way).
    assert accepted >= 100 and refused >= 80, (accepted, refused)


def _hydrated_shape(n: int) -> bytes:
    """n claims laid out as hydrate_pipelined lays them: ascending claim ids two to a phase,
    trace notes and one prefetch per claim in claim order, a double-buffer prefetch per phase
    edge after them. Built directly: hydrating a program of 64k claims is slower than C."""
    p = StreamPack(source_plan="plan0", topo_gen=1, map_gen=1, data_gen=1, pipeline_depth=2)
    for i in range(n):
        reads = (1 + i % 7, 100 + i)
        p.segments.append(
            LaneSegment(
                name=f"seg{i}",
                claim_id=i,
                phase_id=i // 2,
                lane=Lane.U,
                width=4,
                opcode="f32.add",
                reads=reads,
                writes=(1_000_000 + i,),
                prefetch=f"pf{i}",
                dispatch="core",
                channel="host",
            )
        )
        p.prefetches.append(Prefetch(f"pf{i}", 4, reads, buffers=2))
        p.blocks.append(Block(base=i, count=4, strides=(0,)))
        p.trace_notes.append(TraceNote(claim_id=i))
    for ph in range(n // 2 - 1):
        p.prefetches.append(Prefetch(f"dbpf{ph}_{ph + 1}", 1, (102 + 2 * ph,), buffers=2))
    return encode(p)


def _trace_heavy(n: int) -> bytes:
    """One segment and n trace notes in ascending claim order: a pack whose cost is its trace
    stream, so a rescan of that stream -- the lightest quadratic, an 8-octet compare a pair --
    is not hidden under the segments' cost."""
    p = StreamPack(source_plan="plan0", topo_gen=1, map_gen=1, data_gen=1)
    p.segments.append(
        LaneSegment(
            name="seg0", claim_id=0, phase_id=0, lane=Lane.U, width=4, opcode="f32.add",
            reads=(1,), writes=(2,), prefetch=None, dispatch="core", channel="host",
        )
    )  # fmt: skip
    p.trace_notes.extend(TraceNote(claim_id=i) for i in range(n))
    return encode(p)


def test_the_r10_walk_and_the_dispatch_cost_linear_in_the_claims():
    """The complexity witness: the fastest of several runs of bcir_sp_execute (the R10 walk
    included) over one pack shape at two sizes. Linear work grows with the size ratio (16x
    and 32x here: measured 17x, about 40x and 32x); a walk that rescans the pack per segment
    grows with its square (measured 267x for the first pair before QUAL-LIN). The first pair
    catches a lost fast path within seconds even when it is quadratic; the second, larger,
    catches a lighter one -- a search that forgot where the last one matched -- which only
    shows at scale; the trace-heavy pair isolates the lightest, a uniqueness check that
    rescans the trace ids. Each bound is a wide band, several times the linear ratio, so a
    shared runner's noise cannot reach it; neither can a quadratic walk get under it."""
    if _cc() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)

        def fastest(shape, n: int, reps: int) -> float:
            path = os.path.join(tmp, f"shape{n}.bin")
            with open(path, "wb") as f:
                f.write(shape(n))
            r = subprocess.run([exe, "--time", path, str(reps)], capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            fields = r.stdout.split()
            assert fields[0] == "segments", r.stdout
            return max(float(fields[3]), 1.0)

        pairs = (
            (_hydrated_shape, 500, 8_000, 64.0),
            (_hydrated_shape, 2_000, 64_000, 256.0),
            (_trace_heavy, 2_000, 64_000, 128.0),
        )
        for shape, small, large, bound in pairs:
            ratio = fastest(shape, large, 3) / fastest(shape, small, 5)
            assert ratio < bound, (
                f"{shape.__name__}: {small}->{large} cost {ratio:.0f}x (bound {bound:.0f}x)"
            )
