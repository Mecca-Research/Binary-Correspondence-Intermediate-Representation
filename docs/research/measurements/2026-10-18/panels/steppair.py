"""BF16-KERN / BF16-PANELS-1 (2026-10-18): the 9.84M decoder's training step under bf16+exp+reduce+attn at THREADS
threads, three libraries in one process, one step each in turn -- the order rotated every round -- so
the arms run tens of milliseconds apart and the host's drift reaches them alike:

  base    d5bc457's runtime/c (BASE, an extracted tree)
  kern    this tree built with BCIR_DT_NO_PANELS: the new AMX kernel, the packs as before
  panels  this tree: the kernel and the SwiGLU block's panels

The three compute one state (every change is bit-identical), so every round's three losses must be
equal; the script asserts it. Prints one JSON line: each arm's median step, and per pair of arms the
round-by-round ratios' median, a 95% interval (percentile bootstrap over rounds, fixed seed) and how
many rounds the second arm was faster in.

    python3 steppair.py REPO BASE_RUNTIME_C THREADS ROUNDS
"""

import json
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

repo, base_c = Path(sys.argv[1]), Path(sys.argv[2])
threads, rounds = int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, str(repo))
from bcir.frontends.models.decode import DecoderSpec  # noqa: E402
from bcir.hosted.models.native import NativeDecoder  # noqa: E402

C = repo / "runtime" / "c"


def build(out, src, define=None):
    """The trainer library from SRC's units, the shared-build flags of the tests' build_library."""
    out.mkdir(parents=True, exist_ok=True)
    units = []
    for name in ("bcir_tensor.c", "bcir_decoder_train.c"):
        text = (src / name).read_text()
        if define and name == "bcir_decoder_train.c":
            text = f"#define {define} 1\n" + text
        (out / name).write_text(text)
        units.append(str(out / name))
    lib = out / "libtrain.so"
    cmd = ["gcc", "-std=c11", "-O3", "-ffp-contract=off", "-DBCIR_DT_BUILD_SHARED", "-shared", "-fPIC",
           "-pthread", "-I", str(src), *units, "-o", str(lib), "-lm"]  # fmt: skip
    subprocess.run(cmd, check=True)
    return lib


tmp = Path(tempfile.mkdtemp())
libs = {
    "base": build(tmp / "base", base_c),
    "kern": build(tmp / "kern", C, "BCIR_DT_NO_PANELS"),
    "panels": build(tmp / "panels", C),
}
spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4, tied_embeddings=True)
B, T = 8, 128
toks = [(i * 37 + 11) % 260 for i in range(B * T)]
tgts = [(i * 29 + 5) % 260 for i in range(B * T)]
arms = tuple(libs)
nds = {a: NativeDecoder(libs[a], spec, batch_size=B, context_length=T, threads=threads,
                        contract="bf16+exp+reduce+attn", max_memory_bytes=1 << 31) for a in arms}  # fmt: skip
for nd in nds.values():
    nd.run(toks, tgts, steps=1)  # two warm-up steps
    nd.run(toks, tgts, steps=1)
times = {a: [] for a in arms}
for r in range(rounds):
    order = arms[r % 3 :] + arms[: r % 3]
    losses = {}
    for a in order:
        t0 = time.perf_counter()
        losses[a] = nds[a].run(toks, tgts, steps=1)[0]["loss"]
        times[a].append(time.perf_counter() - t0)
    assert len(set(losses.values())) == 1, (r, losses)


def interval(ratios, draws=2000):
    rng = random.Random(20261018)
    meds = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(draws))
    return round(meds[int(0.025 * draws)], 4), round(meds[int(0.975 * draws)], 4)


doc = {"threads": threads, "rounds": rounds, "contract": "bf16+exp+reduce+attn",
       "median_ms": {a: round(1000 * statistics.median(times[a]), 2) for a in arms}, "pairs": {}}  # fmt: skip
for a, b in (("base", "kern"), ("kern", "panels"), ("base", "panels")):
    ratios = [x / y for x, y in zip(times[a], times[b])]
    doc["pairs"][f"{a}/{b}"] = {"median_ratio": round(statistics.median(ratios), 4),
                                "interval_95": interval(ratios), "faster_rounds": sum(q > 1 for q in ratios)}  # fmt: skip
print(json.dumps(doc))
for nd in nds.values():
    nd.close()
