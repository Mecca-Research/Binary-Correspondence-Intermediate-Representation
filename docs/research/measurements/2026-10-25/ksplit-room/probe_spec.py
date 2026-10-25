"""The GEO step's pool jobs a worker at a time: the native trainer built from WORK (a checkout whose
runtime/c holds probe_jobs.py's copies) with the job clock, its default step program at the GEO
dimensions (vocab 260, width 128, 4 heads, 3 layers, FF 256, tied, batch 4), one untimed step then
ROUNDS logged steps; each pool job named by its launch's claim (native_pack.op_name). Per claim:
jobs a step, wall us a job, the workers' entry delay, busy share and tail, each worker's busy
time and finish, and the claim's own job clock (the trainer's, BCIR_DT_TIME_PROBE).
usage: probe_spec.py WORK T CONTRACT ROUNDS OUT.json THREADS SPEC (vocab,d,heads,kv,layers,ff,batch)"""

import ctypes as ct
import json
import statistics
import sys
import tempfile
from array import array
from collections import defaultdict
from pathlib import Path

WORK = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(WORK))
from bcir.frontends.models.decode import DecoderSpec  # noqa: E402
from bcir.hosted.models import native_launch as nl  # noqa: E402
from bcir.hosted.models.native import NativeDecoder  # noqa: E402
from bcir.hosted.models.native_choose import timed_run  # noqa: E402
from bcir.hosted.models.native_pack import op_name  # noqa: E402

T, CONTRACT, ROUNDS, OUT = int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
THREADS = int(sys.argv[6])
SPLIT = False
V, D, H, KV, L, FF, B = (int(x) for x in sys.argv[7].split(","))
assert NativeDecoder.__module__ and str(WORK) in sys.modules["bcir"].__file__, "bcir is not WORK's"
spec = DecoderSpec(
    vocab_size=V,
    d_model=D,
    n_heads=H,
    n_layers=L,
    d_ff=FF,
    activation="silu_gate",
    n_kv_heads=KV,
    tied_embeddings=True,
    rms_norm_eps=1e-5,
)
CAP = 8192 * 19
with tempfile.TemporaryDirectory() as tmp:
    lib = NativeDecoder.build(Path(tmp) / "lib", probe="time")
    tr = NativeDecoder(
        lib,
        spec,
        batch_size=B,
        context_length=T,
        threads=THREADS,
        contract=CONTRACT,
        max_memory_bytes=4 << 30,
    )
    read = tr.lib.bcir_probe_jobs
    read.argtypes, read.restype = [ct.POINTER(ct.c_ulonglong), ct.c_size_t], ct.c_size_t
    rate = tr.lib.bcir_probe_tsc_per_us
    rate.argtypes, rate.restype = [], ct.c_double
    tsc_us = rate()
    decisions = tr.program_decisions(steps=1)
    program = nl.step_program(spec, B, T, **decisions)
    words = array("Q", nl.program_words(program))
    name_at, at = {}, 0
    for launch in program:
        name_at[at] = op_name(launch.records[0])
        at += len(launch.records)
    n = B * T
    tokens = [(7 * i + 3) % spec.vocab_size for i in range(n)]
    targets = [(5 * i + 1) % spec.vocab_size for i in range(n)]
    buf = (ct.c_ulonglong * CAP)()
    timed_run(tr, words, tokens, targets, steps=1)
    read(buf, CAP)
    per = defaultdict(lambda: defaultdict(list))
    clock_ns = defaultdict(list)
    for _ in range(ROUNDS):
        _, ns = timed_run(tr, words, tokens, targets, steps=1)
        jobs = read(buf, CAP)
        acc = defaultdict(int)
        for s, name in name_at.items():
            acc[name] += ns[s]
        for name, v in acc.items():
            clock_ns[name].append(v)
        for j in range(jobs):
            o = buf[19 * j : 19 * j + 19]
            name = name_at.get(o[0], f"record {o[0]}") + (f" [{o[1]} tasks]" if SPLIT else "")
            wall = o[2]
            enter, busy, leave, ntask = o[3:7], o[7:11], o[11:15], o[15:19]
            ran = [q for q in range(THREADS) if ntask[q]]
            p = per[name]
            p["wall_us"].append(wall / tsc_us)
            p["tasks"].append(o[1])
            p["busy_share"].append(sum(busy[:THREADS]) / (THREADS * wall) if wall else 0.0)
            p["worker_enter_us"].append(max(enter[1:THREADS]) / tsc_us)
            p["first_leave_us"].append(min(leave[q] for q in range(THREADS)) / tsc_us)
            p["last_leave_us"].append(max(leave[q] for q in range(THREADS)) / tsc_us)
            p["idle_workers"].append(THREADS - len(ran))
            for q in range(THREADS):
                p[f"leave_w{q}"].append(leave[q] / tsc_us)
                p[f"busy_w{q}"].append(busy[q] / tsc_us)
                p[f"ntask_w{q}"].append(ntask[q])
    tr.close()
rows = []
for name, p in per.items():
    k = len(p["wall_us"]) / ROUNDS
    row = {"claim": name, "jobs_a_step": k, "tsc_per_us": tsc_us}
    for key in (
        "wall_us",
        "busy_share",
        "worker_enter_us",
        "first_leave_us",
        "last_leave_us",
        "tasks",
    ):
        row[key + "_median"] = statistics.median(p[key])
    row["wall_us_a_step"] = k * statistics.mean(p["wall_us"])
    row["idle_workers_mean"] = statistics.mean(p["idle_workers"])
    row["per_worker"] = [
        {
            "leave_us_mean": statistics.mean(p[f"leave_w{q}"]),
            "busy_us_mean": statistics.mean(p[f"busy_w{q}"]),
            "tasks_mean": statistics.mean(p[f"ntask_w{q}"]),
        }
        for q in range(THREADS)
    ]
    base = name.split(" [")[0]  # the claim's job clock, whatever its jobs' task counts
    row["clock_ms_a_step_median"] = (
        statistics.median(clock_ns[base]) / 1e6 if base in clock_ns else None
    )
    rows.append(row)
rows.sort(key=lambda r: -r["wall_us_a_step"])
OUT.write_text(
    json.dumps(
        {"T": T, "contract": CONTRACT, "threads": THREADS, "rounds": ROUNDS, "rows": rows}, indent=1
    )
    + "\n"
)
tot = sum(r["wall_us_a_step"] for r in rows)
print(
    f"T{T} {CONTRACT} {THREADS} threads: pool jobs {tot / 1e3:.2f} ms a step (TSC {tsc_us:.0f}/us)"
)
for r in rows:
    print(
        f"  {r['wall_us_a_step'] / 1e3:6.2f} ms x{r['jobs_a_step']:<4.0f} {r['claim']:28s} {r['wall_us_median']:7.1f} us a job, "
        f"busy {r['busy_share_median']:.0%}, workers in by {r['worker_enter_us_median']:5.1f} us, "
        f"leave {r['first_leave_us_median']:6.1f}-{r['last_leave_us_median']:6.1f}, tasks {r['tasks_median']:.0f}"
    )
