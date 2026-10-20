"""PX-4 (PLACE): the native step's transient values placed by their lifetimes.

PX-2b ran the step from a StreamPack and its ExecutionPlanV1 with every value placed where the C step
keeps it -- the C plan's fixed spans, 1.326 times the bench step's concurrent-live bound
(`train.memory.transient`). `native_pack.place_by_lifetime` places every activation and scratch value
by its lifetime instead, in one bank, with GEM+'s static memory planner over the pack's own values; a
trainer lends exactly the placement's extent (`NativeDecoder(transient=...)`), and
`bcir_decoder_run_program` holds every operand inside the arenas lent. The witness holds every placed
run of the trace corpus to the C step's bits (`train.plan.place.mismatch` 0), every placement to the
plan's alias law -- read again here, pair by pair -- and to G26's bound, recomputed here too: the
bench shapes' placements equal to it (a proved optimum), every corpus placement at or above it (its
gap read where the exact search stops at its budget), and a placed trainer to the arenas it lends:
the fixed step refused, a program one byte past them refused with nothing written. The two fuzz
harnesses' lending, the C twin of `fuzz_lent`, is held to the same seeds' verdicts by a replay
build of each harness.
"""

import copy
import ctypes as ct
import dataclasses
import subprocess
import sys
import tempfile
from pathlib import Path

from bcir.abi.streampack_abi import AbiError
from bcir.frontends.models.decode import DecoderSpec
from bcir.hosted.models import native as nat
from bcir.hosted.models.native_launch import RECORD_WORDS
from bcir.hosted.models.native_pack import (
    PLACE_ALIGNMENT,
    Placement,
    pack_artifacts,
    place_by_lifetime,
    placed_words,
    training_pack,
)
from bcir.hosted.models.native_program import TRANSIENT, training_step_program
from bcir.tests import native_decoder_fixtures as nf

#: The host's C compiler, or None: the witnesses' one capability (a fault sweep `--require`s it,
#: so a host without one is never read as a sweep the witnesses passed).
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()

#: Packs the placement is read on without a compiler: the bench step under each contract family,
#: several steps of several micro-batches, a recompute set, and the panels' bench under bf16.
_SHAPES = (
    (nf.BENCH, dict(contract=0)),
    (nf.BENCH, dict(contract=0, steps=2, accumulation=2)),
    (nf.BENCH, dict(contract=0, recompute=3)),
    (nf.BENCH, dict(contract=1 | 2 | 8)),
    (nf.BENCH, dict(contract=16, workspace_bytes=1 << 20, panels_need=1 << 20)),
    (
        nf.PANELS_BENCH,
        dict(contract=16 | 64, workspace_bytes=1 << 21, panels_need=1 << 21, recompute=1),
    ),
    ((DecoderSpec(64, 32, 4, 1, 64, activation="silu_gate", n_kv_heads=2), 1, 4), dict(contract=0)),
)


def _bound(rows, alignment: int) -> int:
    """G26's alignment-aware concurrent-live bound over `rows` (placements), written out again:
    at each phase, the live values' sizes rounded up to `alignment`, less the largest rounding
    among them (the topmost value needs only its own size)."""
    best = 0
    for t in range(min(r.first for r in rows), max(r.last for r in rows) + 1):
        live = [r for r in rows if r.first <= t <= r.last]
        if live:
            padded = [-(-r.size // alignment) * alignment for r in live]
            slack = max(p - r.size for p, r in zip(padded, live))
            best = max(best, sum(padded) - slack)
    return best


def _lawful(tp, placed) -> None:
    """`placed` is a placement of `tp`'s values by their lifetimes (module docstring), every claim
    of it checked here rather than taken from the planner."""
    before = {p.rid: p for p in tp.placements}
    after = {p.rid: p for p in placed.placements}
    assert set(after) == set(before) and len(placed.placements) == len(tp.placements)
    moved = [p for p in placed.placements if before[p.rid].bank in TRANSIENT]
    assert moved, "a pack with no transient value"
    for p in placed.placements:
        old = before[p.rid]
        if old.bank in TRANSIENT:  # placed: in the one bank, aligned, the same value and lifetime
            assert p.bank == "activations" and p.offset % PLACE_ALIGNMENT == 0, p
            assert (p.size, p.first, p.last) == (old.size, old.first, old.last), (p, old)
        else:  # everything else where it was
            assert p == old, (p, old)
    # the plan's alias law, pair by pair: values live at a common phase hold disjoint bytes
    for i, p in enumerate(moved):
        for q in moved[i + 1 :]:
            if p.first <= q.last and q.first <= p.last:
                assert p.offset + p.size <= q.offset or q.offset + q.size <= p.offset, (p, q)
    assert placed.extent == max(p.offset + p.size for p in moved)
    # the bound recomputed, and the placement at it: a proved optimum (TMSAO-1)
    assert placed.bound == _bound(moved, PLACE_ALIGNMENT), placed
    assert placed.extent == placed.bound and placed.optimal, placed
    assert placed.floats * 4 >= placed.extent > (placed.floats - 1) * 4
    # the parts: a value never lives across the phase where the next part begins
    starts = sorted({p.first for p in moved})
    cuts = [t for t in starts[1:] if not any(p.first < t <= p.last for p in moved)]
    assert placed.parts == len(cuts) + 1, (placed.parts, cuts)


def test_the_placement_is_lawful_and_at_its_bound():
    """For packs of every contract family, several steps of several micro-batches, a recompute set
    and the bf16 panels: every activation and scratch value placed in one bank, aligned, its size and
    lifetime kept, every other value where it was; no two values live at one phase on one byte
    (the alias law read pair by pair); the extent at G26's bound written out again here -- a
    proved optimum -- and below the C plan's spans; the plan verifies under the plan's laws; and a
    placement that moved one value onto another live one is refused by them."""
    from bcir.kbcir.cost import TargetProfile
    from bcir.verify import verify_execution_plan

    for (spec, batch, time_), decisions in _SHAPES:
        tp = training_pack(spec, batch, time_, **decisions)
        placed = place_by_lifetime(tp)
        _lawful(tp, placed)
        # the words the lowering must write: the program's at the C step's places, and placed
        # only every transient operand's space and offset moved, to its value's place
        assert placed_words(tp, tp.placements) == tp.words, decisions
        moved_words = placed_words(tp, placed.placements)
        changed = {
            i % RECORD_WORDS for i, (a, b) in enumerate(zip(moved_words, tp.words)) if a != b
        }
        assert changed and changed <= {11 + k for k in range(30) if k % 3 < 2}, sorted(changed)
        held = training_step_program(spec, batch, time_, decisions.get("recompute", 0))
        assert placed.extent < held.transient_bytes(), (decisions, placed.extent)
        _, _, pack, plan = pack_artifacts(tp, placements=placed.placements)
        target = TargetProfile.x86_avx512()
        assert verify_execution_plan(tp.module, plan, target=target, pack=pack) == [], decisions
        # a placement the planner did not make, one value moved onto another live one: the
        # plan's codec refuses to write it, and the plan's laws refuse it as an object
        moved = sorted((p for p in placed.placements if p.bank == "activations"),
                       key=lambda p: (p.first, p.offset))  # fmt: skip
        a, b = next((p, q) for i, p in enumerate(moved) for q in moved[i + 1 :]
                    if p.first <= q.last and q.first <= p.last)  # fmt: skip
        bad = [dataclasses.replace(p, offset=a.offset) if p.rid == b.rid else p
               for p in placed.placements]  # fmt: skip
        try:
            pack_artifacts(tp, placements=bad)
        except AbiError as e:
            assert "alias" in str(e), (decisions, e)
        else:
            raise AssertionError(f"a placement onto a live value written ({decisions})")
        aliased = copy.deepcopy(plan)
        aliased.lifetimes = [dataclasses.replace(x, offset=a.offset) if x.rid == b.rid else x
                             for x in aliased.lifetimes]  # fmt: skip
        assert verify_execution_plan(tp.module, aliased, target=target, pack=pack), decisions


def test_the_parts_are_laid_out_apart():
    """Values no phase holds together lay out apart and share addresses: four parts of a stated
    set -- the largest the second -- each laid out from offset 0, the extent the largest part's, the
    parts counted, the bound the largest part's, and the values every other bank holds left where
    they were; a value written at the phase its predecessor is last read shares the part (both are
    live there), and values of sizes off the alignment still start aligned."""
    from types import SimpleNamespace

    rows = (
        Placement(1, "weights", 0, 4096, 0, 9),  # not transient: kept
        Placement(10, "activations", 0, 640, 0, 2),  # part 1: 640 + 128 live together
        Placement(11, "scratch", 640, 128, 1, 2),
        Placement(12, "activations", 2048, 1024, 3, 5),  # part 2: 1024 + 512 + 256 at phase 4
        Placement(13, "scratch", 0, 512, 4, 5),
        Placement(14, "activations", 4096, 256, 4, 4),
        Placement(15, "scratch", 64, 192, 6, 9),  # part 3: one value
        # part 4: a claim reading 16 writes 17, both live at phase 11; sizes off the alignment
        Placement(16, "activations", 0, 300, 10, 11),
        Placement(17, "scratch", 0, 200, 11, 12),
    )
    placed = place_by_lifetime(SimpleNamespace(placements=rows))
    at = {p.rid: p for p in placed.placements}
    assert at[1] == rows[0]
    assert placed.parts == 4 and placed.optimal
    assert placed.extent == placed.bound == 1024 + 512 + 256
    assert {at[r].offset for r in (10, 12, 15)} == {0}  # each part from the bottom
    assert all(at[r].bank == "activations" for r in (10, 11, 12, 13, 14, 15, 16, 17))
    assert all(at[r].offset % PLACE_ALIGNMENT == 0 for r in (10, 11, 12, 13, 14, 15, 16, 17))
    for a, b in ((10, 11), (12, 13), (12, 14), (13, 14), (16, 17)):  # live together: apart
        lo, hi = sorted((at[a], at[b]), key=lambda p: p.offset)
        assert lo.offset + lo.size <= hi.offset, (lo, hi)


def test_the_bench_row_reads_the_placed_arena():
    """`train.memory.transient` reads the bench step's pack with its transient values placed by
    lifetime: 754,176 bytes over a 754,176-byte bound -- 1.0, a proved optimum -- where the C plan
    reserves 999,424 over the step program's 753,664; the 512 bytes between the two bounds are the
    loss rows' terms, which the pack holds as a value of their own and the step program inside
    the scratch."""
    ratio, extent, bound, optimal = nf.placed_transient()
    assert (ratio, extent, bound, optimal) == (1.0, 754_176, 754_176, True)
    c_ratio, held, step_bound, _ = nf.transient_ratio()
    assert (held, step_bound) == (999_424, 753_664) and c_ratio > 1.3
    spec, batch, time_ = nf.BENCH
    tp = training_pack(spec, batch, time_, contract=0)
    rows = {p.rid: p for p in tp.placements}
    terms = [r for r in tp.module.resources.values() if r.elem_bytes == 8]  # the loss's terms (f64)
    assert len(terms) == 1 and rows[terms[0].rid].size == bound - step_bound == 8 * batch * time_


def test_every_placed_run_trains_the_c_steps_bits():
    """Every pack run of the trace corpus (the program runs' configurations, `pack_runs`), its
    transient values placed by lifetime and run on a trainer lending exactly the placement's arena,
    lowers, runs, and trains the C step's state and events on the same case bit for bit, no
    placement below its bound (`train.plan.place.mismatch` 0); every run's layout is read, its gap
    to the bound 0 wherever the search proves the layout optimal at the bound."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            return  # no C compiler, or no C units (an installed package): a documented skip
        bad, runs, notes, gaps = nf.place_mismatch(path)
        assert bad == 0, notes[:3]
        assert runs == nf.program_run_count(), (runs, nf.program_run_count())
        assert len(gaps) == runs and all(gap >= 0 for gap, _ in gaps), gaps


def _state(trainer) -> tuple:
    """The state a refusal must leave: its digest, and the transient arenas' bytes."""
    lent = trainer.transient or (trainer.activation_count, trainer.state.plan.scratch)
    return (nf.state_digest(trainer),
            ct.string_at(trainer.buffers["activations"], 4 * lent[0]),
            ct.string_at(trainer.buffers["scratch"], 4 * lent[1]))  # fmt: skip


def test_a_placed_trainer_runs_only_what_it_holds():
    """A trainer lending the placement's arena runs the placed program and nothing past it: the
    fixed step, whose spans are the plan's, is refused (BCIR_DT_CAPACITY) with nothing written, and
    so is its forward on an arena one float short of the plan's; the placed program on an arena one
    float short, or with one value moved past the arena's end, is refused whole (BCIR_DT_INVALID)
    with nothing written; the placed program on a trainer lending
    more than it places runs the same bits; and the binding takes only (activation floats, scratch
    floats) pairs, counting what it lends."""
    case = nf.trace_corpus()[0]
    tokens, targets, optimizer = nf._case_inputs(case)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            return
        step = nf.trace_case(path, case, trace=False)
        full, decisions = nf._case_trainer(path, case)
        full.close()
        tp = training_pack(case.spec, case.batch, case.time, **decisions)
        placed = place_by_lifetime(tp)
        _lawful(tp, placed)
        pack, plan, _, _ = pack_artifacts(tp, placements=placed.placements)

        def run(lent, the_plan=plan, fixed=False):
            trainer, _ = nf._case_trainer(path, case, transient=lent)
            try:
                before = _state(trainer)
                try:
                    if fixed:
                        trainer.run(tokens, targets, steps=case.steps, optimizer=optimizer,
                                    gradient_accumulation=case.accumulation)  # fmt: skip
                    else:
                        words = trainer.lower_pack(pack, the_plan)
                        events = trainer.run_program(words, tokens, targets, steps=case.steps,
                                                     optimizer=optimizer,
                                                     gradient_accumulation=case.accumulation)  # fmt: skip
                        return 0, nf.state_digest(trainer), events, trainer.memory_bytes
                except nat.NativeRunStopped as stopped:
                    assert stopped.completed == 0, stopped
                    return stopped.status, before == _state(trainer), (), trainer.memory_bytes
            finally:
                trainer.close()

        status, state, events, lent_bytes = run((placed.floats, 0))
        assert status == 0 and (state, events) == (step.state, step.events)
        status, state, events, more_bytes = run((placed.floats + 1000, 7))
        assert status == 0 and (state, events) == (step.state, step.events)
        assert more_bytes == lent_bytes + 4 * 1007
        # the fixed step on the placed arena: refused (BCIR_DT_CAPACITY), nothing written
        status, untouched, events, _ = run((placed.floats, 0), fixed=True)
        assert (status, untouched, events) == (2, True, ()), status
        # the fixed step's own entries keep the plan's spans: a forward on an activation arena one
        # float short of the C plan's is refused (BCIR_DT_CAPACITY), nothing written
        probe, _ = nf._case_trainer(path, case)
        acts, scratch = probe.activation_count, probe.state.plan.scratch
        probe.close()
        trainer, _ = nf._case_trainer(path, case, transient=(acts - 1, scratch))
        try:
            before = _state(trainer)
            try:
                trainer.forward(tokens[: case.batch * case.time])
            except ValueError as refused:
                assert "capacity" in str(refused), refused
            else:
                raise AssertionError("a forward on an arena short of the plan's spans was admitted")
            assert _state(trainer) == before
        finally:
            trainer.close()
        # one float short: the program's last bytes lie past the arena
        status, untouched, events, _ = run((placed.floats - 1, 0))
        assert (status, untouched, events) == (1, True, ()), status
        # a lent length past the address space: refused (BCIR_DT_CAPACITY), nothing written
        trainer, _ = nf._case_trainer(path, case, transient=(placed.floats, 0))
        try:
            before = _state(trainer)
            trainer.state.activation_capacity = 2**63
            words = trainer.lower_pack(pack, plan)
            try:
                trainer.run_program(words, tokens, targets, steps=case.steps, optimizer=optimizer,
                                    gradient_accumulation=case.accumulation)  # fmt: skip
            except nat.NativeRunStopped as stopped:
                assert (stopped.status, stopped.completed) == (2, 0), stopped
            else:
                raise AssertionError("a lent length past the address space was admitted")
            trainer.state.activation_capacity = placed.floats
            assert _state(trainer) == before
        finally:
            trainer.close()
        # a value moved past the arena's end: refused whole
        top = max((p for p in placed.placements if p.bank == "activations"),
                  key=lambda p: p.offset + p.size)  # fmt: skip
        past = [dataclasses.replace(p, offset=placed.extent) if p.rid == top.rid else p
                for p in placed.placements]  # fmt: skip
        _, beyond, _, _ = pack_artifacts(tp, placements=past)
        status, untouched, events, _ = run((placed.floats, 0), the_plan=beyond)
        assert (status, untouched, events) == (1, True, ()), status
        bad_args = [dict(transient=bad) for bad in ((1,), [1, 2], (1, -1), (1.0, 2), (True, 0),
                                                     (1, 2**62))]  # fmt: skip
        bad_args.append(dict(transient=(1, 2), recompute="auto"))  # a placement is one program's
        for bad in bad_args:
            try:
                nat.NativeDecoder(path, case.spec, **bad)
            except ValueError as refused:
                assert "transient" in str(refused), refused
            else:
                raise AssertionError(f"{bad!r} admitted")


def test_the_arguments_lie_outside_the_lent_arenas():
    """A program run holds its arguments apart from the arenas as the trainer lends them, not as
    the placement fills them: tokens lent inside the activation arena past the placement's end --
    bytes the trainer lends, no value is placed in, and a program may write -- are refused
    (BCIR_DT_INVALID) with nothing written, and the same tokens in a buffer of their own run the
    case's bits."""
    case = nf.trace_corpus()[0]
    tokens, targets, optimizer = nf._case_inputs(case)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            return
        step = nf.trace_case(path, case, trace=False)
        full, decisions = nf._case_trainer(path, case)
        full.close()
        tp = training_pack(case.spec, case.batch, case.time, **decisions)
        placed = place_by_lifetime(tp)
        pack, plan, _, _ = pack_artifacts(tp, placements=placed.placements)
        n = len(tokens)
        for inside in (True, False):
            # the arena lent a token row past the placement's end: bytes no value is placed in
            trainer, _ = nf._case_trainer(path, case, transient=(placed.floats + n, 0))
            try:
                words = trainer.lower_pack(pack, plan)
                arena = trainer.buffers["activations"]
                held = ((ct.c_uint32 * n).from_buffer(arena, 4 * placed.floats) if inside
                        else (ct.c_uint32 * n)())  # fmt: skip
                held[:] = tokens
                aims = (ct.c_uint32 * n)(*targets)
                fn = trainer.lib.bcir_decoder_run_program
                fn.argtypes = [ct.POINTER(nat._State), ct.POINTER(nat._Optimizer),
                               ct.POINTER(ct.c_uint64), ct.c_size_t, nat._UP, nat._UP, ct.c_size_t,
                               ct.POINTER(ct.c_double), ct.c_size_t, ct.c_size_t,
                               ct.POINTER(nat._Event), ct.c_size_t, ct.POINTER(ct.c_size_t)]  # fmt: skip
                fn.restype = ct.c_int
                view = (ct.c_uint64 * len(words)).from_buffer(words)
                opt = nat._Optimizer(**dataclasses.asdict(optimizer))
                events, done = (nat._Event * case.steps)(), ct.c_size_t(77)
                before = _state(trainer)
                rc = fn(ct.byref(trainer.state), ct.byref(opt), view, len(words) // RECORD_WORDS,
                        held, aims, n, None, case.steps, case.accumulation, events, case.steps,
                        ct.byref(done))  # fmt: skip
                if inside:
                    assert (rc, done.value) == (1, 77), (rc, done.value)
                    assert _state(trainer) == before
                else:
                    assert (rc, done.value) == (0, case.steps), (rc, done.value)
                    assert nf.state_digest(trainer) == step.state
            finally:
                trainer.close()


def test_the_place_comparison_sees_each_divergence():
    """`place_problems` -- what `train.plan.place.mismatch` counts and the corpus witness asserts
    empty -- names each way a placed run can differ from what it must be, one problem each (a
    refused lowering, other words than the program's at the placed places, a placement below its
    bound, a stopped run, another state, other events), and none in a run that matches, nor in one
    whose layout lies above its bound or is not proved: a gap, not a divergence. The witness can
    see what it exists to count, and counts nothing else."""
    spec, batch, time_ = nf.BENCH
    tp = training_pack(spec, batch, time_, contract=0)
    placed = place_by_lifetime(tp)
    words = placed_words(tp, placed.placements)
    step = nf.TraceRun([], [], 0, "state", ({"step": 1},))
    ok = nf.PackRun(tp, words, "state", ({"step": 1},), 0, 0, placed)
    assert nf.place_problems(step, ok) == []
    for run, expected in (
        (ok._replace(words=(), status=-1, why=105), "refused"),
        (ok._replace(words=tp.words), "words"),
        (ok._replace(placed=dataclasses.replace(placed, extent=placed.bound - 64)), "below"),
        (ok._replace(status=3), "status 3"),
        (ok._replace(state="other"), "state"),
        (ok._replace(events=({"step": 2},)), "events"),
    ):
        found = nf.place_problems(step, run)
        assert len(found) == 1 and expected in found[0], (run.status, found)
    for gap in (dataclasses.replace(placed, extent=placed.bound + 64),
                dataclasses.replace(placed, optimal=False)):  # fmt: skip
        assert nf.place_problems(step, ok._replace(placed=gap)) == []


#: A driver in libFuzzer's place: each file named on the command line through the harness once,
#: and the statuses its replay build records (BCIR_FUZZ_REPLAY) printed one line a file.
_REPLAY_DRIVER = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
extern int bcir_fuzz_replay[2];
int main(int argc, char **argv) {
  int i;
  for (i = 1; i < argc; i++) {
    FILE *f = fopen(argv[i], "rb");
    unsigned char *buf;
    long n;
    if (!f || fseek(f, 0, SEEK_END) != 0 || (n = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0)
      return 2;
    buf = (unsigned char *)malloc(n ? (size_t)n : 1u);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) return 2;
    fclose(f);
    LLVMFuzzerTestOneInput(buf, (size_t)n);
    printf("%d %d\n", bcir_fuzz_replay[0], bcir_fuzz_replay[1]);
    free(buf);
  }
  return 0;
}
"""


def _replay(directory: Path, harness: str, seeds) -> list[tuple[int, int]] | None:
    """Every seed through `harness` (a runtime/manifest.json fuzzer) built from its manifest closure
    with the replay driver in libFuzzer's place: each seed's two recorded statuses, in order. None
    without a compiler, or outside a source checkout (the wheel ships no C units)."""
    from bcir.tests.gate_links import manifest_links
    from bcir.toolchain import build_units, host_link_args

    found = manifest_links("fuzzers", harness)
    if found is None or _CC is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory / "driver.c"
    driver.write_text(_REPLAY_DRIVER, encoding="utf-8", newline="\n")
    exe = directory / ("replay.exe" if sys.platform == "win32" else "replay")
    cmd = [_CC, "-std=c11", "-O1", "-DBCIR_FUZZ_REPLAY", "-I", str(nf._C), str(driver),
           *(str(nf._C / unit) for unit in sorted(found[1])), "-o", str(exe)]  # fmt: skip
    if sys.platform != "win32":
        cmd.insert(1, "-pthread")
    build_units(cmd + host_link_args(["-lm"]), f"{harness} replay build", timeout=600)
    files = []
    for i, (_, data, _) in enumerate(seeds):
        files.append(directory / f"seed{i:03d}.bin")
        files[-1].write_bytes(data)
    out = subprocess.run([str(exe), *map(str, files)], capture_output=True, text=True,
                         timeout=900)  # fmt: skip
    assert out.returncode == 0, (harness, out.returncode, out.stderr[-2000:])
    rows = [tuple(int(x) for x in line.split()) for line in out.stdout.splitlines()]
    assert len(rows) == len(seeds) and all(len(row) == 2 for row in rows), out.stdout[-2000:]
    return rows


def test_the_fuzz_harnesses_hold_their_seeds_to_their_verdicts():
    """The two decoder fuzz harnesses, built with a replay driver in libFuzzer's place
    (BCIR_FUZZ_REPLAY: each input's statuses), hold every seed tools/c/fuzz_streampack.sh seeds them
    with to the verdict its Python twin holds it to. The runner's (runtime/c/fuzz_decoder_train.c):
    every planner program admitted and run -- the placed ones (PX-4) on the arenas they reach,
    inside the room the harness allocates for a placement -- every near miss (the kept layers'
    program on the recompute set's arena among them: a harness lending a recompute run the whole
    room admits it) and the placed program a float short refused. The lowering's (fuzz_decoder_pack.c): every planner pack lowered and
    run, the placed pack a float short refused at the run, every near miss refused at the
    lowering. A harness whose lending drifted from its twin's (`fuzz_lent`) would start its
    campaign outside the run (L2): only a replay of the C sees it."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        seeds = nf.fuzz_seeds()
        rows = _replay(Path(tmp) / "train", "fuzz_decoder_train", seeds)
        if rows is None:
            return  # no C compiler, or no C units (an installed package): a documented skip
        assert any(data[0] & nf.FUZZ_PLACED for _, data, _ in seeds)
        wrong = [(name, row) for (name, _, admitted), row in zip(seeds, rows)
                 if row != ((0 if admitted else 1), -1)]  # fmt: skip
        assert not wrong, wrong
        packs = nf.fuzz_pack_seeds()
        rows = _replay(Path(tmp) / "pack", "fuzz_decoder_pack", packs)
        assert rows is not None and any(data[0] & nf.FUZZ_PLACED for _, data, _ in packs)
        wrong = []
        for (name, data, ok), (lowered, ran) in zip(packs, rows):
            short = data[0] & nf.FUZZ_PLACED and data[0] & nf.FUZZ_SHORT
            if (not ok and (lowered not in (1, 2) or ran != -1)) or (
                ok and (lowered, ran) != (0, 1 if short else 0)
            ):
                wrong.append((name, lowered, ran))
        assert not wrong, wrong
