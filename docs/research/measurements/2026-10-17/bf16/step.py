"""NDT-BF16: the 9.84M decoder's training step at THREADS threads under kblock+exp+reduce and under
bf16+exp+reduce, the contracts alternated round by round (their order reversed every other round),
every step timed alone; README.json.

    python3 step.py REPO THREADS ROUNDS STEPS

Both trainers start from the same seeded weights and take the same batch every step, so each line's
loss is its contract's own after the same steps (the two differ: bf16 is another arithmetic).
"""

import statistics
import sys
import tempfile
import time

sys.path.insert(0, sys.argv[1])
from bcir.frontends.models.decode import DecoderSpec  # noqa: E402
from bcir.hosted.models.native import NativeDecoder  # noqa: E402

threads, rounds, steps = (int(v) for v in sys.argv[2:5])
spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4, tied_embeddings=True)
lib = NativeDecoder.build(tempfile.mkdtemp())
B, T = 8, 128
toks = [(i * 37 + 11) % 260 for i in range(B * T)]
tgts = [(i * 29 + 5) % 260 for i in range(B * T)]
contracts = ("kblock+exp+reduce", "bf16+exp+reduce")
nds = {
    c: NativeDecoder(
        lib,
        spec,
        batch_size=B,
        context_length=T,
        threads=threads,
        contract=c,
        max_memory_bytes=1 << 31,
    )
    for c in contracts
}
times = {c: [] for c in contracts}
loss = {}
for c, nd in nds.items():  # two warm-up steps each
    nd.run(toks, tgts, steps=1)
    nd.run(toks, tgts, steps=1)
for r in range(rounds):
    for c in contracts if r % 2 == 0 else contracts[::-1]:
        for _ in range(steps):
            t0 = time.perf_counter()
            loss[c] = nds[c].run(toks, tgts, steps=1)[0]["loss"]
            times[c].append(time.perf_counter() - t0)
for c in contracts:
    ms = 1000 * statistics.median(times[c])
    print(
        f"{c:20s} median {ms:7.1f} ms  min {1000 * min(times[c]):7.1f}  ({len(times[c])} steps)"
        f"  {B * T / (ms / 1000):7.0f} tokens/s  last loss {loss[c]:.4f}"
        f"  workspace {nds[c].workspace_bytes} bytes"
    )
print(
    f"ratio {statistics.median(times[contracts[0]]) / statistics.median(times[contracts[1]]):.3f}"
)
for nd in nds.values():
    nd.close()
