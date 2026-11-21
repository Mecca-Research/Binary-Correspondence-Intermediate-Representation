"""The native data loader: token shards and the batches over them, the oracle and its C twin.

The C subset audit's native data loader. `bcir/frontends/models/token_shards.py` is the oracle (the BTKS shard,
its windows, the RNG's per-epoch shuffle, index-addressed batches) and `runtime/c/bcir_tokload.c` its freestanding
twin, driven here through the hosted `bcir-tokload`, which maps shards from files. Held:

- **The oracle's laws.** A shard round-trips; every refusal has its code; an epoch visits every window once; a
  batch is a function of its index alone; neighbouring windows share one token; an unshuffled loader walks the
  windows in shard order.
- **Byte for byte.** `bcir-tokload pack` writes the oracle's shard octets, at both widths.
- **Every verdict as the oracle's.** `info` over shards and their corruptions -- each field, each refusal --
  answers as `open_shard` does; `batch` over generated corpora (short shards, every context and size to small
  bounds, shuffled and not, batches straddling epochs and near the last sample) prints the oracle's rows, and its
  refusals (two vocabularies, no window, a context or size of 0, a batch past 2^64 samples) the oracle's codes.
- **The harness's contracts:** `test_tokload` (every refusal leaves the caller's shard as it was; an exact resume).
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import tempfile

from bcir.frontends.models import token_shards as ts
from bcir.toolchain import host_c_compiler

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CC = host_c_compiler()  # None in the quick tier, which hides the host's compilers
_U64 = (1 << 64) - 1
# the tool and the library units it links: tools/bcir-tokload's closure in runtime/manifest.json
_TOOL = ("bcir_tokload_tool.c", "bcir_tokload.c", "bcir_digest_hw.c", "bcir_rng.c", "bcir_sha256.c")
_HARNESS = ("test_tokload.c", "bcir_tokload.c", "bcir_digest_hw.c", "bcir_rng.c", "bcir_sha256.c")


def _refused(call) -> int:
    try:
        call()
    except ts.ShardError as exc:
        return exc.code
    raise AssertionError("taken")


def test_the_oracle_holds_its_laws():
    data = ts.encode([3, 1, 4, 1, 5, 9, 2, 6], 10, 2)
    shard = ts.open_shard(data)
    assert shard.tokens == (3, 1, 4, 1, 5, 9, 2, 6) and shard.vocab == 10 and shard.width == 2
    assert shard.digest == hashlib.sha256(data[64:]).digest() and len(data) == 64 + 16
    assert _refused(lambda: ts.encode([10], 10)) == ts.ERR_ARG
    assert _refused(lambda: ts.encode([1], 70_000, 2)) == ts.ERR_ARG
    assert _refused(lambda: ts.open_shard(data[:-1])) == ts.ERR_FORMAT
    assert _refused(lambda: ts.open_shard(b"BTKX" + data[4:])) == ts.ERR_FORMAT
    bad = bytearray(data)
    bad[40] ^= 1
    assert _refused(lambda: ts.open_shard(bytes(bad))) == ts.ERR_DIGEST
    # a token past the vocabulary is refused before the digest is asked
    bad = bytearray(data)
    bad[64] = 0xFF
    assert _refused(lambda: ts.open_shard(bytes(bad))) == ts.ERR_TOKEN
    r = random.Random(5)
    shards = [
        ts.open_shard(ts.encode([r.randrange(50) for _ in range(n)], 50)) for n in (40, 3, 77)
    ]
    loader = ts.Loader(shards, 4, (9, 9))
    # 40 tokens hold 9 windows of 5, not 10: neighbours share one token, so the last needs a 41st
    assert loader.windows == 9 + 0 + 19
    seen = []
    for b in range(loader.windows):
        inputs, targets = loader.batch(b, 1)
        assert inputs[0][1:] == targets[0][:-1]
        seen.append(tuple(inputs[0]))
    plain = ts.Loader(shards, 4, (9, 9), shuffle=False)
    walk = [tuple(plain.batch(b, 1)[0][0]) for b in range(plain.windows)]
    assert sorted(seen) == sorted(walk) and seen != walk  # one epoch: every window once, shuffled
    assert walk[0] == shards[0].tokens[:4] and walk[9] == shards[2].tokens[:4]
    assert ts.Loader(shards, 4, (9, 9)).batch(17, 3) == loader.batch(
        17, 3
    )  # a pure function of the index
    # each epoch its own order: epoch 1 is not epoch 0 again
    assert loader.batch(0, loader.windows) != loader.batch(1, loader.windows)
    assert _refused(lambda: ts.Loader([shards[1]], 4, (0, 0))) == ts.ERR_EMPTY
    other = ts.open_shard(ts.encode([1, 2, 3, 4, 5, 6], 51))
    assert _refused(lambda: ts.Loader([shards[0], other], 2, (0, 0))) == ts.ERR_VOCAB
    assert _refused(lambda: loader.batch(_U64 // 3, 3)) == ts.ERR_ARG
    assert loader.batch((_U64 - 2) // 3, 3)


def _build(tmp: str, name: str, sources) -> str | None:
    from bcir.tests.run_all import _is_source_checkout

    if _CC is None or not _is_source_checkout():
        return None
    exe = os.path.join(tmp, name + (".exe" if os.name == "nt" else ""))
    argv = [_CC, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", _C]
    argv += [os.path.join(_C, s) for s in sources] + ["-o", exe]
    try:
        built = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssertionError(f"{name} did not build: {exc}") from exc
    assert built.returncode == 0, built.stderr[-3000:]
    return exe


def _run(argv) -> tuple[int, str]:
    try:
        ran = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssertionError(f"bcir-tokload did not run: {exc}") from exc
    assert ran.returncode in (0, 1), ran.stdout[-2000:] + ran.stderr[-2000:]
    return ran.returncode, ran.stdout


def _info_want(data: bytes) -> str:
    try:
        s = ts.open_shard(data)
    except ts.ShardError as exc:
        return f"ERR {exc.code}"
    return f"ok {s.vocab} {s.width} {len(s.tokens)} {s.digest.hex()}"


def _corruptions(data: bytes, r: random.Random):
    """The shard and one corruption of each field, and some anywhere."""
    out = [data, data[:-1], data[:63], data + b"\0"]
    for at in (0, 4, 6, 8, 12, 16, 23, 27, 40):
        for flip in (1, 0x80):
            bad = bytearray(data)
            bad[at] ^= flip
            out.append(bytes(bad))
    for _ in range(12):
        bad = bytearray(data)
        bad[r.randrange(len(bad))] ^= 1 << r.randrange(8)
        out.append(bytes(bad))
    return out


def check_tool(exe: str) -> dict:
    """bcir-tokload == the oracle: pack's octets, info's verdicts, batch's rows and refusals; what ran. The one
    comparison the test and the gate's `tokload` section run."""
    r = random.Random(23)
    seen = {"packed": 0, "info": 0, "refused": 0, "batches": 0, "loader_refusals": 0}
    with tempfile.TemporaryDirectory() as tmp:
        shards = []
        for k, (n, vocab, width) in enumerate(
            [
                (0, 5, 4),
                (1, 5, 4),
                (37, 9, 2),
                (64, 65536, 2),
                (300, 1 << 20, 4),
                (5, 9, 4),
                (120, 9, 2),
            ]
        ):
            tokens = [r.randrange(vocab) for _ in range(n)]
            ids = os.path.join(tmp, f"ids{k}")
            with open(ids, "w", encoding="ascii") as f:
                f.write(" ".join(map(str, tokens)) + ("\n" if k % 2 else ""))
            path = os.path.join(tmp, f"s{k}.btks")
            rc, out = _run([exe, "pack", str(vocab), str(width), ids, path])
            want = ts.encode(tokens, vocab, width)
            with open(path, "rb") as f:
                assert f.read() == want, (k, n, vocab, width)
            assert rc == 0 and out == f"ok {n} {want[32:64].hex()}\n", out
            seen["packed"] += 1
            shards.append((path, want))
        # info over every shard and its corruptions
        paths, wants = [], []
        for k, (_path, data) in enumerate(shards[1:4]):
            for j, blob in enumerate(_corruptions(data, r)):
                p = os.path.join(tmp, f"c{k}_{j}")
                with open(p, "wb") as f:
                    f.write(blob)
                paths.append(p)
                wants.append(_info_want(blob))
        _rc, out = _run([exe, "info", *paths])
        assert out.splitlines() == wants
        seen["info"] = len(wants)
        seen["refused"] = sum(w.startswith("ERR") for w in wants)
        # batches over the vocabulary-9 shards (5 and 120 tokens, and 37 at width 2)
        group = [shards[2], shards[5], shards[6]]
        opened = [ts.open_shard(d) for _p, d in group]
        for _ in range(60):
            context, size, shuffle = r.randrange(1, 10), r.randrange(1, 8), r.randrange(2)
            key = (r.getrandbits(32), r.getrandbits(32))
            pick = r.sample(range(3), r.randrange(1, 4))
            loader_shards = [opened[i] for i in pick]
            try:
                loader = ts.Loader(loader_shards, context, key, bool(shuffle))
            except ts.ShardError as exc:
                want_lines = [f"ERR {exc.code}"]
                first, count = 0, 1
                seen["loader_refusals"] += 1
            else:
                count = r.randrange(1, 5)
                first = r.choice((0, r.randrange(50), loader.windows // size, _U64 // size - count))
                want_lines = []
                for b in range(first, first + count):
                    inputs, targets = loader.batch(b, size)
                    rows = ";".join(",".join(map(str, row)) for row in inputs)
                    trows = ";".join(",".join(map(str, row)) for row in targets)
                    want_lines.append(f"b {b} {rows} | {trows}")
                seen["batches"] += count
            argv = [exe, "batch", str(key[0]), str(key[1]), str(context), str(size), str(shuffle)]
            argv += [str(first), str(count)] + [group[i][0] for i in pick]
            _rc, out = _run(argv)
            assert out.splitlines() == want_lines, (argv[2:], out[:300], want_lines[:1])
        # the loader's refusals: two vocabularies, a context or size of 0, a batch past the last sample
        mixed = [shards[2][0], shards[4][0]]
        for argv, code in (
            ([exe, "batch", "0", "0", "4", "2", "1", "0", "1", *mixed], ts.ERR_VOCAB),
            ([exe, "batch", "0", "0", "0", "2", "1", "0", "1", shards[6][0]], ts.ERR_ARG),
            ([exe, "batch", "0", "0", "4", "0", "1", "0", "1", shards[6][0]], ts.ERR_ARG),
            (
                [exe, "batch", "0", "0", "4", "3", "1", str(_U64 // 3), "1", shards[6][0]],
                ts.ERR_ARG,
            ),
            ([exe, "batch", "0", "0", "4", "3", "0", "0", "1", shards[0][0]], ts.ERR_EMPTY),
        ):
            rc, out = _run(argv)
            assert rc == 1 and out == f"ERR {code}\n", (argv[2:], out)
            seen["loader_refusals"] += 1
        # a token exactly at the vocabulary: the shard re-headed one short, its digest still whole
        tight = bytearray(ts.encode([0, 8, 3], 9))
        tight[8] = 8
        p = os.path.join(tmp, "tight")
        with open(p, "wb") as f:
            f.write(bytes(tight))
        assert _info_want(bytes(tight)) == f"ERR {ts.ERR_TOKEN}"
        assert _run([exe, "info", p]) == (1, f"ERR {ts.ERR_TOKEN}\n")
        # a shard refused at open is the batch's verdict too
        bad = bytearray(shards[6][1])
        bad[40] ^= 1
        p = os.path.join(tmp, "bad")
        with open(p, "wb") as f:
            f.write(bytes(bad))
        assert _run([exe, "batch", "0", "0", "4", "3", "1", "0", "1", p]) == (
            1,
            f"ERR {ts.ERR_DIGEST}\n",
        )
    return seen


def test_the_c_tool_answers_as_the_oracle_does():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp, "bcir-tokload", _TOOL)
        if exe is None:
            return
        seen = check_tool(exe)
        # the corpus is not vacuous: shards packed, refusals of every kind reached, batches compared
        assert seen["packed"] == 7 and seen["refused"] > 40 and seen["batches"] > 80, seen
        assert seen["loader_refusals"] >= 5, seen


def test_the_c_harness_holds_its_contracts():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp, "test_tokload", _HARNESS)
        if exe is None:
            return
        ran = subprocess.run([exe], capture_output=True, text=True, timeout=300)
        assert ran.returncode == 0 and ran.stdout.strip() == "test_tokload: OK", (
            ran.stdout + ran.stderr
        )
