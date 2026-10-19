"""Screens (2026-10-19): two trainer libraries' training steps alternated in one process -- the
9.84M decoder at 8 x 128 tokens, one step each in turn, the order reversed every round, so the arms
run tens of milliseconds apart and the host's drift reaches them alike -- their losses held equal
every round (both arms compute one state). Prints one JSON line: each arm's median step, the
round-by-round ratios' median (A over B: above 1 means B is faster), a 95% interval (percentile
bootstrap over rounds, a fixed seed) and the rounds B was faster in. Wall time: indicative, never a
gate.

    python3 pair.py REPO SRC_A SRC_B THREADS ROUNDS CONTRACT

SRC_A and SRC_B are runtime/c directories (this tree's, and one with a screen's patch applied).
"""

import json
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

repo, srcs = Path(sys.argv[1]), (Path(sys.argv[2]), Path(sys.argv[3]))
threads, rounds, contract = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
sys.path.insert(0, str(repo))
from bcir.frontends.models.decode import DecoderSpec  # noqa: E402
from bcir.hosted.models.native import NativeDecoder  # noqa: E402


def build(out, src):
    """The trainer library from SRC's units, the shared-build flags of the tests' build_library."""
    out.mkdir(parents=True, exist_ok=True)
    units = []
    for name in ("bcir_tensor.c", "bcir_decoder_train.c"):
        (out / name).write_text((src / name).read_text())
        units.append(str(out / name))
    lib = out / "libtrain.so"
    cmd = ["gcc", "-std=c11", "-O3", "-ffp-contract=off", "-DBCIR_DT_BUILD_SHARED", "-shared", "-fPIC",
           "-pthread", "-I", str(src), *units, "-o", str(lib), "-lm"]  # fmt: skip
    subprocess.run(cmd, check=True)
    return lib


tmp = Path(tempfile.mkdtemp())
libs = [build(tmp / f"arm{i}", src) for i, src in enumerate(srcs)]
spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4, tied_embeddings=True)
B, T = 8, 128
toks = [(i * 37 + 11) % 260 for i in range(B * T)]
tgts = [(i * 29 + 5) % 260 for i in range(B * T)]
nds = [NativeDecoder(lib, spec, batch_size=B, context_length=T, threads=threads, contract=contract,
                     max_memory_bytes=1 << 31) for lib in libs]  # fmt: skip
for nd in nds:
    nd.run(toks, tgts, steps=1)  # two warm-up steps
    nd.run(toks, tgts, steps=1)
times = [[], []]
for r in range(rounds):
    losses = [None, None]
    for a in (0, 1) if r % 2 == 0 else (1, 0):
        t0 = time.perf_counter()
        losses[a] = nds[a].run(toks, tgts, steps=1)[0]["loss"]
        times[a].append(time.perf_counter() - t0)
    assert losses[0] == losses[1], (r, losses)
ratios = [x / y for x, y in zip(times[0], times[1])]
rng = random.Random(20261019)
meds = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
print(json.dumps({"threads": threads, "rounds": rounds, "contract": contract,
                  "median_ms": [round(1000 * statistics.median(t), 2) for t in times],
                  "a_over_b": round(statistics.median(ratios), 4),
                  "ci95": [round(meds[50], 4), round(meds[1950], 4)],
                  "b_faster_rounds": sum(q > 1 for q in ratios)}))  # fmt: skip
for nd in nds:
    nd.close()
