"""The counter-based RNG: Philox4x32-10 and its streams, the oracle and its freestanding C twin.

The C subset audit's first native addition. `bcir/frontends/models/rng.py` is the oracle and
`runtime/c/bcir_rng.c` its twin. Both are held to the generator's published known-answer vectors (Random123), and
to each other over generated requests: blocks at the counter space's corners, fills from every lane offset, bounded
draws across the rejection band, permutations, and the two float conversions, compared exactly.
"""

from __future__ import annotations

import os
import random
import subprocess
import tempfile

from bcir.frontends.models import rng
from bcir.toolchain import host_c_compiler

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CC = host_c_compiler()  # None in the quick tier, which hides the host's compilers
_U32 = 0xFFFFFFFF
_U64 = (1 << 64) - 1


def test_the_oracle_gives_the_published_known_answers():
    for counter, key, output in rng.KNOWN_ANSWERS:
        assert rng.philox4x32_10(counter, key) == output
    # one word of the counter or key changed changes every output word: a bijection keyed as published
    base = rng.philox4x32_10((1, 2, 3, 4), (5, 6))
    for k in range(4):
        moved = list((1, 2, 3, 4))
        moved[k] ^= 1
        assert all(a != b for a, b in zip(rng.philox4x32_10(tuple(moved), (5, 6)), base))


def test_the_oracle_refuses_what_is_not_a_word():
    for bad in ((-1, 0), (0, 1 << 32), (0,), (True, 0)):
        try:
            rng.u32(bad, 0, 0)
        except ValueError:
            continue
        raise AssertionError(f"key {bad!r} taken")
    for args in ((0, -1), (-1, 0), (1 << 64, 0), (0, 1 << 64)):
        try:
            rng.u32((0, 0), *args)
        except ValueError:
            continue
        raise AssertionError(f"stream, index {args!r} taken")
    # a request that draws nothing is refused alike: the check comes before the first draw
    for call in (
        lambda: rng.fill_u32((0, 0), 1 << 64, 0, 0),
        lambda: rng.fill_u32((0, 1 << 32), 0, 0, 0),
        lambda: rng.below((0, 0), -1, 0, 0),
        lambda: rng.below((-1, 0), 0, 0, 0),
        lambda: rng.below((0, 0), 0, 1 << 64, 0),
        lambda: rng.permutation((0, 0), 1 << 64, 0, 1),
        lambda: rng.permutation((True, 0), 0, 0, 0),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("a request drawing nothing was taken unchecked")


def _requests(seed: int = 7):
    """(request line, the oracle's answer) pairs."""
    r = random.Random(seed)
    out = []
    corners = (0, 1, _U32 - 1, _U32)
    for counter, key, output in rng.KNOWN_ANSWERS:
        out.append(
            (
                f"philox {' '.join(map(str, counter))} {' '.join(map(str, key))}",
                " ".join(f"{w:08x}" for w in output),
            )
        )
    for _ in range(300):
        counter = tuple(
            r.choice(corners) if r.random() < 0.3 else r.getrandbits(32) for _ in range(4)
        )
        key = (r.getrandbits(32), r.getrandbits(32))
        words = rng.philox4x32_10(counter, key)
        out.append(
            (
                f"philox {' '.join(map(str, counter))} {' '.join(map(str, key))}",
                " ".join(f"{w:08x}" for w in words),
            )
        )
    for _ in range(120):
        key = (r.getrandbits(32), r.getrandbits(32))
        stream = r.choice((0, 1, _U64, r.getrandbits(64), r.getrandbits(32) << 32))
        first = r.choice((0, r.randrange(4), r.getrandbits(40), _U64 - r.randrange(8)))
        n = r.randrange(0, 40)
        n = min(n, _U64 + 1 - first)
        words = rng.fill_u32(key, stream, first, n)
        out.append(
            (f"fill {key[0]} {key[1]} {stream} {first} {n}", " ".join(f"{w:08x}" for w in words))
        )
    for _ in range(400):
        key = (r.getrandbits(32), r.getrandbits(32))
        stream, index = r.getrandbits(64), r.getrandbits(48)
        bound = r.choice(
            (0, 1, 2, 3, 7, 1000, 1 << 30, 1 << 31, (1 << 31) + 1, _U32, _U32 - 1)
            + (r.getrandbits(32), 3 << 30)
        )
        value, nxt = rng.below(key, stream, index, bound)
        out.append((f"below {key[0]} {key[1]} {stream} {index} {bound}", f"{value} {nxt}"))
    for _ in range(80):
        key = (r.getrandbits(32), r.getrandbits(32))
        stream, index, n = (
            r.getrandbits(64),
            r.getrandbits(40),
            r.choice((0, 1, 2, 5, r.randrange(400))),
        )
        perm, nxt = rng.permutation(key, stream, index, n)
        out.append(
            (
                f"perm {key[0]} {key[1]} {stream} {index} {n}",
                f"{','.join(map(str, perm)) or '-'} {nxt}",
            )
        )
    for bits in (0, 1, 255, 256, _U32, _U32 - 255, *(r.getrandbits(32) for _ in range(200))):
        out.append((f"f32 {bits}", rng.unit_f32_bits(bits)))
    for _ in range(200):
        hi, lo = (
            r.choice((0, _U32, r.getrandbits(32))),
            r.choice((0, 2047, 2048, _U32, r.getrandbits(32))),
        )
        out.append((f"f64 {hi} {lo}", rng.unit_f64(hi, lo)))
    return out


def _build(tmp: str) -> str | None:
    from bcir.tests.run_all import _is_source_checkout

    if _CC is None or not _is_source_checkout():
        return None
    exe = os.path.join(tmp, "test_rng" + (".exe" if os.name == "nt" else ""))
    argv = [_CC, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", _C]
    argv += [os.path.join(_C, s) for s in ("test_rng.c", "bcir_rng.c")] + ["-o", exe]
    try:
        built = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssertionError(f"test_rng did not build: {exc}") from exc
    assert built.returncode == 0, built.stderr[-3000:]
    return exe


def test_the_c_twin_holds_its_contracts_and_the_known_answers():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        ran = subprocess.run([exe], capture_output=True, text=True, timeout=300)
        assert ran.returncode == 0 and ran.stdout.strip() == "test_rng: OK", ran.stdout + ran.stderr


def check_driver(exe: str) -> int:
    """`test_rng FILE` == the oracle over every generated request; the number of requests. The one comparison the
    test and the gate's `rng` section run."""
    requests = _requests()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "requests")
        with open(path, "w", encoding="ascii") as f:
            f.write("".join(line + "\n" for line, _want in requests))
        try:
            ran = subprocess.run([exe, path], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AssertionError(f"test_rng did not run: {exc}") from exc
    assert ran.returncode == 0, ran.stdout[-2000:] + ran.stderr[-2000:]
    got = ran.stdout.splitlines()
    assert len(got) == len(requests), (len(got), len(requests))
    for (line, want), have in zip(requests, got):
        # C prints %a; Python's own hex differs in its spelling only
        if isinstance(want, float):
            assert float.fromhex(have) == want, (line, have, want)
        else:
            assert have == want, (line, have, want)
    # the corpus reached the rejection loop: a bound whose band rejected a draw
    assert any(
        line.startswith("below") and int(want.split()[1]) > int(line.split()[4]) + 1
        for line, want in requests
        if isinstance(want, str)
    )
    return len(requests)


def test_the_c_twin_answers_every_request_as_the_oracle_does():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        assert check_driver(exe) > 1000
