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
from bcir.toolchain import host_link_args


def _build(tmp):
    exe = os.path.join(tmp, "test_decode")
    r = subprocess.run(
        host_link_args(
            [
                _cc(),
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-I",
                _RUNTIME_C,
                os.path.join(_RUNTIME_C, "test_decode.c"),
                os.path.join(_RUNTIME_C, "bcir_decode.c"),
                "-o",
                exe,
                "-lm",
            ]
        ),
        capture_output=True,
        text=True,
    )
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
            (
                f"rmsnorm {rows} {dim}\n"
                + " ".join(repr(v) for v in x)
                + "\n"
                + " ".join(repr(v) for v in gamma)
                + "\n",
                rmsnorm_reference(x, rows, dim, gamma),
            ),
            (
                f"rope {rows} {dim} 10000.0 3\n" + " ".join(repr(v) for v in x) + "\n",
                rope_reference(x, rows, dim, 10000.0, pos_offset=3),
            ),
            (
                f"embedding {vocab} {dim} {n}\n"
                + " ".join(repr(v) for v in table)
                + "\n"
                + " ".join(str(i) for i in ids)
                + "\n",
                embedding_lookup(
                    EmbeddingTable(table=tuple(table), n_vocab=vocab, dim=dim), ids, dim
                ),
            ),
        ]
        for feed, want in cases:
            got = _run(exe, feed)
            assert len(got) == len(want)
            assert all(abs(a - b) <= 1e-12 for a, b in zip(got, want)), feed.split()[0]
        # the embedding twin REFUSES an out-of-range id (exit 3), exactly where the oracle raises.
        bad = f"embedding {vocab} {dim} 1\n" + " ".join(repr(v) for v in table) + f"\n{vocab}\n"
        r = subprocess.run([exe], input=bad, capture_output=True, text=True)
        assert r.returncode == 3, r.returncode


def test_gqa_kernel_matches_the_oracle_and_the_cached_row_is_bitwise():
    """Rung 5's GQA + KV-cache twins: (1) bcir_gqa_attention differential vs
    gqa_attention_reference (GQA and the n_kv_heads == n_heads MHA regression case);
    (2) the ragged head-group refusal (exit 3) exactly where the oracle raises; (3) the
    KV-cached row twin emits the full recompute's LAST row BITWISE (%.17g-identical --
    one shared code path in the kernel, the naive-vs-cached invariant at kernel level)."""
    if _cc() is None:
        return
    from bcir.frontends.models.decode import gqa_attention_reference

    rng = random.Random(0x69A2)
    seq, nh, nkv, dk = 4, 4, 2, 2
    q = [rng.uniform(-1.0, 1.0) for _ in range(seq * nh * dk)]
    k = [rng.uniform(-1.0, 1.0) for _ in range(seq * nkv * dk)]
    v = [rng.uniform(-1.0, 1.0) for _ in range(seq * nkv * dk)]
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        feed = (
            f"gqa {seq} {nh} {nkv} {dk}\n"
            + " ".join(repr(x) for x in q)
            + "\n"
            + " ".join(repr(x) for x in k)
            + "\n"
            + " ".join(repr(x) for x in v)
            + "\n"
        )
        got = _run(exe, feed)
        want = gqa_attention_reference(q, k, v, seq, nh, nkv, dk)
        assert len(got) == len(want) == seq * nh * dk
        assert all(abs(a - b) <= 1e-12 for a, b in zip(got, want))
        # the MHA regression case rides the same kernel (n_kv_heads == n_heads).
        kf = [rng.uniform(-1.0, 1.0) for _ in range(seq * nh * dk)]
        vf = [rng.uniform(-1.0, 1.0) for _ in range(seq * nh * dk)]
        feed = (
            f"gqa {seq} {nh} {nh} {dk}\n"
            + " ".join(repr(x) for x in q)
            + "\n"
            + " ".join(repr(x) for x in kf)
            + "\n"
            + " ".join(repr(x) for x in vf)
            + "\n"
        )
        got = _run(exe, feed)
        want = gqa_attention_reference(q, kf, vf, seq, nh, nh, dk)
        assert all(abs(a - b) <= 1e-12 for a, b in zip(got, want))
        # ragged head groups refuse (exit 3), exactly where the oracle raises.
        k3 = [rng.uniform(-1.0, 1.0) for _ in range(seq * 3 * dk)]
        bad = (
            f"gqa {seq} {nh} 3 {dk}\n"
            + " ".join(repr(x) for x in q)
            + "\n"
            + " ".join(repr(x) for x in k3)
            + "\n"
            + " ".join(repr(x) for x in k3)
            + "\n"
        )
        r = subprocess.run([exe], input=bad, capture_output=True, text=True)
        assert r.returncode == 3, r.returncode
        # A zero-length cached row formerly dereferenced scores[0]; reject it before arithmetic.
        r = subprocess.run([exe], input="gqa_row 0 1 1 1\n0\n", capture_output=True, text=True)
        assert r.returncode == 3, (r.returncode, r.stderr)
        # the KV-cached row is the full recompute's last row, BITWISE (%.17g string equality).
        full = subprocess.run(
            [exe],
            input=(
                f"gqa {seq} {nh} {nkv} {dk}\n"
                + " ".join(repr(x) for x in q)
                + "\n"
                + " ".join(repr(x) for x in k)
                + "\n"
                + " ".join(repr(x) for x in v)
                + "\n"
            ),
            capture_output=True,
            text=True,
        )
        q_last = q[(seq - 1) * nh * dk :]
        rowr = subprocess.run(
            [exe],
            input=(
                f"gqa_row {seq} {nh} {nkv} {dk}\n"
                + " ".join(repr(x) for x in q_last)
                + "\n"
                + " ".join(repr(x) for x in k)
                + "\n"
                + " ".join(repr(x) for x in v)
                + "\n"
            ),
            capture_output=True,
            text=True,
        )
        assert full.returncode == 0 and rowr.returncode == 0
        assert full.stdout.split()[-nh * dk :] == rowr.stdout.split()  # bitwise, not epsilon


def test_decode_kernels_preflight_indices_and_pointer_arithmetic():
    if _cc() is None:
        return
    source = r"""
#include <limits.h>
#include "bcir_decode.h"
int main(void){
  double table[8]={0},out[4]={17,17,17,17},one=1;int ids[2]={1,9};
  if(bcir_embedding(table,4,2,ids,2,out)!=-1)return 1;
  for(int i=0;i<4;i++)if(out[i]!=17)return 2;
  bcir_rmsnorm(&one,INT_MAX,INT_MAX,&one,1e-6,out);if(out[0]!=17)return 3;
  if(bcir_embedding(&one,INT_MAX,INT_MAX,ids,1,out)!=-1)return 4;
  if(bcir_gqa_attention(&one,&one,&one,INT_MAX,INT_MAX,1,INT_MAX,&one,out)!=-1)return 5;
  return 0;
}
"""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "decode_preflight.c")
        with open(src, "w", encoding="utf-8") as f:
            f.write(source)
        exe = os.path.join(tmp, "decode_preflight")
        build = subprocess.run(
            host_link_args(
                [
                    _cc(),
                    "-std=c11",
                    "-O1",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    _RUNTIME_C,
                    os.path.join(_RUNTIME_C, "bcir_decode.c"),
                    src,
                    "-o",
                    exe,
                    "-lm",
                ]
            ),
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        assert run.returncode == 0, (run.stdout, run.stderr)


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
