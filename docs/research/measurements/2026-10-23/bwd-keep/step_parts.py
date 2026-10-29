"""The GEO step a part at a time: the native trainer built with the job clock (BCIR_DT_TIME_PROBE), its default
step program at the GEO dimensions (vocab 260, width 128, 4 heads, 3 layers, FF 256, tied, rms_eps 1e-5, batch 4),
one untimed run then ROUNDS timed runs of one step; each job's nanoseconds summed by its claim name
(native_pack.op_name). usage: step_parts.py T CONTRACT ROUNDS OUT.json [THREADS]"""

import json
import statistics
import sys
import tempfile
from array import array
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]  # the repository
sys.path.insert(0, str(ROOT))
from bcir.frontends.models.decode import DecoderSpec  # noqa: E402
from bcir.hosted.models import native_launch as nl  # noqa: E402
from bcir.hosted.models.native import NativeDecoder  # noqa: E402
from bcir.hosted.models.native_choose import timed_run  # noqa: E402
from bcir.hosted.models.native_pack import op_name  # noqa: E402

T, CONTRACT, ROUNDS, OUT = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
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
        contract=CONTRACT,
        max_memory_bytes=4 << 30,
    )
    decisions = tr.program_decisions(steps=1)
    program = nl.step_program(spec, 4, T, **decisions)
    words = array("Q", nl.program_words(program))
    starts, names, at = [], [], 0
    for launch in program:
        starts.append(at)
        names.append(op_name(launch.records[0]))
        at += len(launch.records)
    n = 4 * T
    tokens = [(7 * i + 3) % spec.vocab_size for i in range(n)]
    targets = [(5 * i + 1) % spec.vocab_size for i in range(n)]
    timed_run(tr, words, tokens, targets, steps=1)
    per = defaultdict(list)
    totals = []
    for _ in range(ROUNDS):
        _, ns = timed_run(tr, words, tokens, targets, steps=1)
        acc = defaultdict(int)
        for s, name in zip(starts, names):
            acc[name] += ns[s]
        for name in set(names):
            per[name].append(acc[name])
        totals.append(sum(ns[s] for s in starts))
    tr.close()
rows = sorted(
    ((statistics.median(v) / 1e6, k, names.count(k)) for k, v in per.items()), reverse=True
)
tot = statistics.median(totals) / 1e6
doc = {
    "T": T,
    "contract": CONTRACT,
    "threads": THREADS,
    "rounds": ROUNDS,
    "jobs_total_ms_median": tot,
    "parts": [{"claim": k, "jobs": c, "ms_median": ms, "share": ms / tot} for ms, k, c in rows],
    "raw_ns": {k: v for k, v in per.items()},
    "raw_total_ns": totals,
}
OUT.write_text(json.dumps(doc, indent=1) + "\n")
print(f"T{T} {CONTRACT} {THREADS} threads: jobs {tot:.1f} ms a step (median of {ROUNDS})")
for ms, k, c in rows:
    print(f"  {ms:8.2f} ms {100 * ms / tot:5.1f}%  x{c:<3d} {k}")
