"""Rung 5, second half (§7.4): the LLM decode stage kernels as C twins
(`runtime/c/bcir_decode.c`), differential-gated against the rung-3 oracle references --
the prototype-then-port discipline applied to the decoder the same way bcir_train.c
applied it to training. Same accumulation order + a shared libm => the rails agree to
<=1e-12 (the gate absorbs cross-libm CI); the embedding twin REFUSES an out-of-range id
exactly where the oracle raises. Toolchain-gated (returns early without a C compiler)."""

import os
import random
import subprocess
import tempfile

from bcir.kbcir.transformer_grads import rmsnorm_reference, rope_reference
from bcir.kbcir.unsupervised import EmbeddingTable, embedding_lookup
from bcir.tests.test_c_executor import _RUNTIME_C, _cc


def _build(tmp):
    exe = os.path.join(tmp, "test_decode")
    r = subprocess.run([_cc(), "-std=c11", "-O2", "-Wall", "-Wextra", "-I", _RUNTIME_C,
                        os.path.join(_RUNTIME_C, "test_decode.c"),
                        os.path.join(_RUNTIME_C, "bcir_decode.c"), "-o", exe, "-lm"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe

def _run(exe, feed):
    r = subprocess.run([exe], input=feed, capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stderr)
    return [float(t) for t in r.stdout.split()]


def test_decode_kernels_match_the_oracle_references():
    if _cc() is None:
        return
    rng = random.Random(0xDEC0)
    rows, dim = 5, 8
    x = [rng.uniform(-2.0, 2.0) for _ in range(rows * dim)]
    gamma = [rng.uniform(0.5, 1.5) for _ in range(dim)]
    vocab, n = 12, 6
    table = [rng.uniform(-1.0, 1.0) for _ in range(vocab * dim)]
    ids = [rng.randrange(vocab) for _ in range(n)]
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        cases = [
            (f"rmsnorm {rows} {dim}\n" + " ".join(repr(v) for v in x) + "\n"
             + " ".join(repr(v) for v in gamma) + "\n",
             rmsnorm_reference(x, rows, dim, gamma)),
            (f"rope {rows} {dim} 10000.0 3\n" + " ".join(repr(v) for v in x) + "\n",
             rope_reference(x, rows, dim, 10000.0, pos_offset=3)),
            (f"embedding {vocab} {dim} {n}\n" + " ".join(repr(v) for v in table) + "\n"
             + " ".join(str(i) for i in ids) + "\n",
             embedding_lookup(EmbeddingTable(table=tuple(table), n_vocab=vocab, dim=dim),
                              ids, dim)),
        ]
        for feed, want in cases:
            got = _run(exe, feed)
            assert len(got) == len(want)
            assert all(abs(a - b) <= 1e-12 for a, b in zip(got, want)), feed.split()[0]
        # the embedding twin REFUSES an out-of-range id (exit 3), exactly where the oracle raises.
        bad = (f"embedding {vocab} {dim} 1\n" + " ".join(repr(v) for v in table) + f"\n{vocab}\n")
        r = subprocess.run([exe], input=bad, capture_output=True, text=True)
        assert r.returncode == 3, r.returncode


if __name__ == "__main__":
    import sys

    try:
        test_decode_kernels_match_the_oracle_references()
        print("PASS test_decode_kernels_match_the_oracle_references")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {e!r}")
        sys.exit(1)
