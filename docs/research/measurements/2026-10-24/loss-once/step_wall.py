"""The GEO step's wall time against its jobs' clocks in the same run: the native trainer built with the job clock
(BCIR_DT_TIME_PROBE), its default step program at the GEO dimensions, STEPS steps a call; per round the call's wall
time (perf_counter_ns around run_program) and the sum of every job's nanoseconds. usage: step_wall.py REPO T ROUNDS
STEPS [THREADS]"""

import statistics
import sys
import tempfile
import time
from array import array
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from bcir.frontends.models.decode import DecoderSpec  # noqa: E402
from bcir.hosted.models import native_launch as nl  # noqa: E402
from bcir.hosted.models.native import NativeDecoder  # noqa: E402
from bcir.hosted.models.native_choose import timed_run  # noqa: E402

T, ROUNDS, STEPS = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
THREADS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
spec = DecoderSpec(
    vocab_size=260,
    d_model=128,
    n_heads=4,
    n_layers=3,
    d_ff=256,
    activation="silu_gate",
    n_kv_heads=4,
    tied_embeddings=True,
    rms_norm_eps=1e-5,
)
with tempfile.TemporaryDirectory() as tmp:
    lib = NativeDecoder.build(Path(tmp) / "lib", probe="time")
    tr = NativeDecoder(
        lib,
        spec,
        batch_size=4,
        context_length=T,
        threads=THREADS,
        contract="kblock+exp+reduce",
        max_memory_bytes=4 << 30,
    )
    decisions = tr.program_decisions(steps=STEPS)
    program = nl.step_program(spec, 4, T, **decisions)
    words = array("Q", nl.program_words(program))
    starts, at = [], 0
    for launch in program:
        starts.append(at)
        at += len(launch.records)
    n = 4 * T
    tokens = [(7 * i + 3) % spec.vocab_size for i in range(n * STEPS)]
    targets = [(5 * i + 1) % spec.vocab_size for i in range(n * STEPS)]
    timed_run(tr, words, tokens, targets, steps=STEPS)
    walls, jobs = [], []
    for _ in range(ROUNDS):
        t0 = time.perf_counter_ns()
        _, ns = timed_run(tr, words, tokens, targets, steps=STEPS)
        walls.append((time.perf_counter_ns() - t0) / STEPS)
        jobs.append(sum(ns[s] for s in starts) / STEPS)
    tr.close()
w, j = statistics.median(walls), statistics.median(jobs)
gaps = sorted(a - b for a, b in zip(walls, jobs, strict=True))
print(
    f"T{T} threads {THREADS} steps/call {STEPS}: wall {w / 1e6:.3f} ms a step, jobs {j / 1e6:.3f} ms, "
    f"outside the jobs {statistics.median(gaps) / 1e6:.3f} ms ({statistics.median(gaps) / w:.1%}); "
    f"launches {len(program)}"
)
