"""Two trainer libraries from patches.py's copies, alternated step by step in one process (the order
swapped every round), the losses asserted equal every round, under bf16+exp+reduce+attn at the
9.84M decoder's shapes. Prints one JSON object: each arm's median step, the paired ratio's median
(the first arm's step over the second's) with a 95% percentile-bootstrap interval, the rounds the
second arm was faster in, and -- where both libraries carry the probe -- each arm's median pack and
compute phase walls and its pack tasks' time by role and operand.

    python3 pair.py REPO VARIANTS_DIR THREADS ROUNDS ARM_A ARM_B"""

import ctypes as ct
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

repo, root, threads, rounds = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
)
arms = sys.argv[5:7]
sys.path.insert(0, str(repo))
from bcir.frontends.models.decode import DecoderSpec  # noqa: E402
from bcir.hosted.models.native import NativeDecoder  # noqa: E402

NAMES = ("fwd A", "fwd B (weights)", "dx A", "dx B (weights)", "dW A", "dW B (activations)")


def build(d):
    lib = d / "libtrain.so"
    cmd = ["gcc", "-std=c11", "-O3", "-ffp-contract=off", "-fno-math-errno", "-DBCIR_DT_BUILD_SHARED",
           "-shared", "-fPIC", "-pthread", "-I", str(repo / "runtime" / "c"), str(d / "bcir_tensor.c"),
           str(d / "bcir_decoder_train.c"), "-o", str(lib), "-lm"]  # fmt: skip
    subprocess.run(cmd, check=True)
    return lib


libs = {a: build(root / a) for a in arms}
spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4, tied_embeddings=True)
B, T = 8, 128
toks = [(i * 37 + 11) % 260 for i in range(B * T)]
tgts = [(i * 29 + 5) % 260 for i in range(B * T)]
nds = {a: NativeDecoder(libs[a], spec, batch_size=B, context_length=T, threads=threads,
                        contract="bf16+exp+reduce+attn", max_memory_bytes=1 << 31) for a in arms}  # fmt: skip
probes = {a: ct.CDLL(str(libs[a].resolve())) for a in arms}
probed = all(hasattr(p, "bcir_tensor_pack_probe") for p in probes.values())
buf = (ct.c_ulonglong * 9)()
for a in arms:
    for _ in range(2):
        nds[a].run(toks, tgts, steps=1)
    if probed:
        probes[a].bcir_tensor_pack_probe(buf)
times = {a: [] for a in arms}
packs = {a: [] for a in arms}
for r in range(rounds):
    order = arms if r % 2 == 0 else arms[::-1]
    losses = {}
    for a in order:
        t0 = time.perf_counter()
        losses[a] = nds[a].run(toks, tgts, steps=1)[0]["loss"]
        times[a].append(time.perf_counter() - t0)
        if probed:
            probes[a].bcir_tensor_pack_probe(buf)
            packs[a].append(list(buf))
    assert len(set(losses.values())) == 1, (r, losses)
ratios = [x / y for x, y in zip(times[arms[0]], times[arms[1]])]
rng = random.Random(20261018)
boots = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
out = {"threads": threads, "rounds": rounds}
for a in arms:
    out[a] = {"step_ms": round(statistics.median(times[a]) * 1e3, 2)}
    if probed:
        med = lambda i, a=a: statistics.median(v[i] for v in packs[a]) / 1e6  # noqa: E731
        out[a].update({"pack_wall_ms": round(med(6), 3), "compute_wall_ms": round(med(7), 3),
                       "pack_task_ms": {n: round(med(i), 3) for i, n in enumerate(NAMES)}})  # fmt: skip
out[f"{arms[0]}/{arms[1]}"] = round(statistics.median(ratios), 4)
out["ci95"] = [round(boots[50], 4), round(boots[1949], 4)]
out[f"{arms[1]}_faster_rounds"] = sum(x > 1 for x in ratios)
print(json.dumps(out, indent=1))
