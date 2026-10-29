"""The GEO step's pool jobs in order, one step: each job's launch claim, tasks, wall us and the workers'
entry delay. usage: probe_order.py WORK T CONTRACT [THREADS]"""

import ctypes as ct
import sys
import tempfile
from array import array
from pathlib import Path

WORK = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(WORK))
from bcir.frontends.models.decode import DecoderSpec  # noqa: E402
from bcir.hosted.models import native_launch as nl  # noqa: E402
from bcir.hosted.models.native import NativeDecoder  # noqa: E402
from bcir.hosted.models.native_choose import timed_run  # noqa: E402
from bcir.hosted.models.native_pack import op_name  # noqa: E402

T, CONTRACT = int(sys.argv[2]), sys.argv[3]
THREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
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
    read = tr.lib.bcir_probe_jobs
    read.argtypes, read.restype = [ct.POINTER(ct.c_ulonglong), ct.c_size_t], ct.c_size_t
    rate = tr.lib.bcir_probe_tsc_per_us
    rate.argtypes, rate.restype = [], ct.c_double
    tsc = rate()
    program = nl.step_program(spec, 4, T, **tr.program_decisions(steps=1))
    words = array("Q", nl.program_words(program))
    name_at, at = {}, 0
    for launch in program:
        name_at[at] = op_name(launch.records[0])
        at += len(launch.records)
    n = 4 * T
    tokens = [(7 * i + 3) % 260 for i in range(n)]
    targets = [(5 * i + 1) % 260 for i in range(n)]
    buf = (ct.c_ulonglong * (8192 * 19))()
    for _ in range(3):
        timed_run(tr, words, tokens, targets, steps=1)
        jobs = read(buf, 8192 * 19)
    tr.close()
for j in range(jobs):
    o = buf[19 * j : 19 * j + 19]
    enter = max(o[4 : 3 + THREADS]) / tsc
    print(
        f"{j:3d} {name_at.get(o[0], o[0])!s:28s} tasks {o[1]:4d} wall {o[2] / tsc:8.1f} us  workers in by {enter:6.1f} us"
    )
