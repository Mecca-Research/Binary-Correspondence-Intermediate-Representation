"""The hardware digests: SHA-256 and CRC-32 on the CPU's own instructions, beside the portable code.

The C subset audit's third addition. `runtime/c/bcir_digest_hw.c` computes the SHA-256 of `bcir_sha256.c` and the
CRC-32 of `bcir_crc32.h` on SHA-NI or the ARMv8 SHA-2 instructions, and by carry-less-multiply folding or the ARMv8
CRC32 instructions; the portable code stays the reference. Held here:

- **The fold's constants are their definition.** x^544, x^480, x^160 and x^96 modulo the CRC's polynomial,
  bit-reflected, derived here and read out of the source.
- **Each path answers as `hashlib` and `zlib` do,** forced one at a time -- every path the probe reports, the
  probe's whole mask, and 0 (the portable code) -- over every length to 300 bytes and longer messages, from running
  CRC states; and the forced path is the one that did the work (a witness build counts each kernel's runs).
- **The probe says what the OS says.** On Linux, its mask is the one `/proc/cpuinfo`'s flags give.
- **The harness's own contracts:** the standards' vectors on every path, each path against the portable code from
  every alignment and every split of an update.

`hashlib` and `zlib` are differential oracles, never a source. A path this host's CPU lacks is not run, and the
test says which ran; CI's x86-64 and AArch64 runners run the rest natively.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import subprocess
import tempfile
import zlib

from bcir.toolchain import host_c_compiler

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CC = host_c_compiler()  # None in the quick tier, which hides the host's compilers
_SHANI, _SHA_ARM, _CLMUL, _CRC_ARM = 1, 2, 4, 8
_P = (1 << 32) | 0x04C11DB7  # the CRC-32 polynomial, x^32 + x^26 + ... + 1


def _xmod(e: int) -> int:
    r = 1
    for _ in range(e):
        r <<= 1
        if r >> 32:
            r ^= _P
    return r


def fold_constant(e: int) -> int:
    """The fold constant for x^e: x^e mod P with x^d at bit 32 - d (reflected, and one place up for the product of
    two reflected operands)."""
    r = _xmod(e)
    return sum(1 << (32 - d) for d in range(32) if r >> d & 1)


def test_the_fold_constants_are_their_definition():
    with open(os.path.join(_C, "bcir_digest_hw.c"), encoding="utf-8") as f:
        src = f.read()
    k512 = re.search(r"k512 = _mm_set_epi64x\((0x[0-9a-f]+)LL, (0x[0-9a-f]+)LL\)", src)
    k128 = re.search(r"k128 = _mm_set_epi64x\((0x[0-9a-f]+)LL, (0x[0-9a-f]+)LL\)", src)
    assert k512 and k128, "the fold's constants moved"
    # _mm_set_epi64x(high, low): the low half multiplies a register's earlier eight bytes
    assert (int(k512[2], 16), int(k512[1], 16)) == (fold_constant(544), fold_constant(480))
    assert (int(k128[2], 16), int(k128[1], 16)) == (fold_constant(160), fold_constant(96))
    # the published pair (Gopal et al., and every zlib descendant that folds): the derivation is the one they use
    assert (fold_constant(544), fold_constant(480)) == (0x154442BD4, 0x1C6E41596)


def _build(tmp: str, witness: bool) -> str | None:
    from bcir.tests.run_all import _is_source_checkout

    if _CC is None or not _is_source_checkout():
        return None
    exe = os.path.join(
        tmp, "test_digest_hw" + ("_w" if witness else "") + (".exe" if os.name == "nt" else "")
    )
    argv = [_CC, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", _C]
    argv += ["-DBCIR_DIGEST_HW_WITNESS"] if witness else []
    argv += [os.path.join(_C, s) for s in ("test_digest_hw.c", "bcir_digest_hw.c", "bcir_sha256.c")]
    argv += ["-o", exe]
    try:
        built = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssertionError(f"test_digest_hw did not build: {exc}") from exc
    assert built.returncode == 0, built.stderr[-3000:]
    return exe


def _drive(exe: str, lines: list[str]) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "requests")
        with open(path, "w", encoding="ascii") as f:
            f.write("".join(line + "\n" for line in lines))
        try:
            ran = subprocess.run([exe, path], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AssertionError(f"test_digest_hw did not run: {exc}") from exc
    assert ran.returncode == 0, ran.stdout[-2000:] + ran.stderr[-2000:]
    out = ran.stdout.splitlines()
    assert len(out) == len(lines), (len(out), len(lines))
    return out


def _messages(seed: int = 11) -> list[bytes]:
    r = random.Random(seed)
    out = [bytes(r.getrandbits(8) for _ in range(n)) for n in range(301)]
    out += [
        bytes(r.getrandbits(8) for _ in range(n))
        for n in (511, 512, 1000, 4095, 4096, 4097, 20_000)
    ]
    return out


def check_paths(exe: str) -> dict:
    """Every mask worth forcing against hashlib and zlib; {mask: requests} of what ran. The one comparison the test
    and the gate's `digest_hw` section run."""
    probe, compiled = (int(v) for v in _drive(exe, ["probe", "compiled"]))
    assert probe & ~compiled == 0, (probe, compiled)
    masks = {0, probe} | {bit for bit in (1, 2, 4, 8) if probe & bit}
    # a bit the build does not carry is ignored, never run: on x86-64, the AArch64 paths' bits
    masks = sorted(masks | ({~compiled & 15} if ~compiled & 15 else set()))
    r = random.Random(3)
    ran = {}
    for mask in masks:
        lines, want = [], []
        for msg in _messages():
            lines.append(f"sha256 {mask} {msg.hex()}")
            want.append(hashlib.sha256(msg).hexdigest())
            state = r.getrandbits(32)
            lines.append(f"crc32 {mask} {state} {msg.hex()}")
            want.append(f"{zlib.crc32(msg, state ^ 0xFFFFFFFF) ^ 0xFFFFFFFF:08x}")
        lines.append("witness")
        got = _drive(exe, lines)
        for line, w, g in zip(lines, want, got):
            assert g == w, (line[:60], w, g)
        witness = got[-1]
        if witness != "off":
            counts = [int(c) for c in witness.split()]
            for k, bit in enumerate((1, 2, 4, 8)):
                # a forced path did the work; a path not asked for never ran
                assert (counts[k] > 0) == bool(mask & compiled & bit), (mask, counts)
        ran[mask] = len(lines) - 1
    return {"probe": probe, "compiled": compiled, "ran": ran}


def test_each_path_answers_as_hashlib_and_zlib_do():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp, witness=True)
        if exe is None:
            return
        seen = check_paths(exe)
        assert 0 in seen["ran"] and all(n > 600 for n in seen["ran"].values()), seen


def _os_mask() -> int | None:
    """The paths /proc/cpuinfo's flags allow, or None off Linux x86-64 and AArch64."""
    import platform

    machine = platform.machine().lower()
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    flags = set()
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() in ("flags", "Features"):
            flags |= set(value.split())
    if machine in ("x86_64", "amd64"):
        mask = _CLMUL if "pclmulqdq" in flags else 0
        return mask | (_SHANI if {"sha_ni", "ssse3", "sse4_1"} <= flags else 0)
    if machine in ("aarch64", "arm64"):
        return (_SHA_ARM if "sha2" in flags else 0) | (_CRC_ARM if "crc32" in flags else 0)
    return None


def test_the_probe_says_what_the_os_says():
    want = _os_mask()
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp, witness=False)
        if exe is None or want is None:
            return
        assert int(_drive(exe, ["probe"])[0]) == want


def test_the_harness_holds_its_contracts():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp, witness=False)
        if exe is None:
            return
        ran = subprocess.run([exe], capture_output=True, text=True, timeout=600)
        lines = ran.stdout.strip().splitlines()
        assert ran.returncode == 0 and lines[-1] == "test_digest_hw: OK", ran.stdout + ran.stderr
        assert lines[0].startswith("paths:"), lines
