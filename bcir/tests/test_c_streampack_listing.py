"""The C StreamPack listing and dump (runtime/c/bcir_sp_listing.c, PORT-1): the oracle's octets.

`bcir-pack dis|hexdump` in C prints what `bcir.abi.streampack_tool` prints, octet for octet, over a
corpus of every example program in all four wire versions and of hand-built packs whose strings,
claims and records reach every branch of the format (quotes, escapes, non-ASCII, fences, untraced
segments, a claim traced twice, records longer than the C formatter's output chunk). Over resealed
mutations of those packs the two rails refuse the same packs -- the C walk applies the Python
decoder's laws, R10 excluded -- and a refused pack prints nothing on either.

`check_tool` is the one comparison: this module runs it against a binary it builds from the
manifest closure (held there by `assert_fixture_links`), and tools/c/sections/sp_listing.sh runs it
against the binary BCIR Make and CMake build.
"""

from __future__ import annotations

import os
import random
import struct
import subprocess
import sys
import tempfile
import zlib

from bcir.abi import AbiError, decode, encode
from bcir.abi.streampack_tool import format_hexdump, format_listing
from bcir.examples import PROGRAMS
from bcir.gem import hydrate
from bcir.gem.streampack import (
    Block,
    Generation,
    LaneSegment,
    Prefetch,
    StreamPack,
    TraceNote,
    hydrate_pipelined,
)
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.model import Lane

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME_C = os.path.join(_ROOT, "runtime", "c")
# The tool's own main and the units it links (the manifest's closure of tools/bcir-pack, minus the
# libraries' units it does not reach).
_SOURCES = ("bcir_pack_cli.c", "bcir_sp_listing.c", "bcir_runtime.c")
_WIDTHS = (1, 3, 16, 64)
_U64 = (1 << 64) - 1

# Strings that reach every branch of the quoting: both quote choices, the escaped quote, the
# backslash, the named and numbered controls, DEL, and each non-ASCII escape width -- U+0378 is
# unassigned and U+1FAE8 assigned only since Unicode 15, the two repr would spell by interpreter.
_STRINGS = (
    "plan0",
    "it's",
    'say "hi"',
    "both ' and \"",
    "a\\b\tc\nd\re",
    "\x00\x01\x1f\x7f",
    "\xe9Ā\U0001fae8͸",
    "",
)


def _hydrated() -> list[tuple[str, bytes]]:
    packs: list[tuple[str, bytes]] = []
    theta = Theta.cool()
    for target_name, target in (
        ("avx512", TargetProfile.x86_avx512()),
        ("neon", TargetProfile.arm64_neon()),
    ):
        for name in sorted(PROGRAMS):
            module = PROGRAMS[name]()
            result = optimize(module, target, theta)
            v4 = hydrate(module, result)
            packs.append((f"{name}-{target_name}-v4", encode(v4)))
            if target_name == "avx512":
                v1 = hydrate(module, result)
                v1.generations = []
                packs.append((f"{name}-v1", encode(v1)))
                v2 = hydrate_pipelined(module, result, depth=2)
                packs.append((f"{name}-v4-pipelined", encode(v2)))
                v2.generations = []
                packs.append((f"{name}-v2", encode(v2)))
    return packs


def _crafted() -> list[tuple[str, bytes]]:
    """Packs the hydrator never writes: the format's every branch, R10 laws broken on purpose."""
    packs: list[tuple[str, bytes]] = [("empty-v1", encode(StreamPack(source_plan="")))]
    for i, text in enumerate(_STRINGS):
        pack = StreamPack(source_plan=text, topo_gen=_U64 >> 32, map_gen=7, data_gen=3)
        pack.segments.append(
            LaneSegment(
                text or "unnamed",
                _U64,
                0xFFFFFFFF,
                Lane.H,
                1 << 31,
                text + ".op",
                (0, 0xFFFFFFFF),
                (),
                "pf" + text,
                fence_before=(text, "f'", 'g"'),
                fence_after=() if i % 2 else (text,),
                dispatch="pim" if i % 2 else "core",
                channel="hbm_pim" if i % 3 else text,
            )
        )
        # untraced (claim 5), and a claim traced twice: the listing shows the last note
        pack.segments.append(LaneSegment("s1", 5, 1, Lane.GGG, 2, "gather", (1,), (2,)))
        pack.segments.append(LaneSegment("s2", 9, 2, Lane.UX, 64, "ld", (), (3,), "nowhere"))
        pack.trace_notes += [TraceNote(_U64, 1, 2), TraceNote(9, 3, 4), TraceNote(9, _U64, 6)]
        pack.prefetches.append(
            Prefetch("pf" + text, 0xFFFFFFFF, (0, 1, 0xFFFFFFFF), text, text, buffers=2 - i % 2)
        )
        pack.blocks += [Block(_U64, 0, ()), Block(0, _U64, tuple(range(9)) + (_U64,))]
        pack.generations += [Generation(0, 7, 0), Generation(9, 1, 3), Generation(0xFFFFFFFF)]
        packs.append((f"strings-{i}-v4", encode(pack)))
        pack.generations = []
        packs.append((f"strings-{i}-v3", encode(pack)))
    long = StreamPack(source_plan="p" * 4000)
    long.segments.append(LaneSegment("n" * 3000, 1, 0, Lane.U, 4, "o" * 2500, (1,) * 700, ()))
    long.trace_notes.append(TraceNote(1, 0xDEADBEEF, 0xFEEDFACE))
    packs.append(("long-records", encode(long)))
    return packs


def corpus() -> list[tuple[str, bytes]]:
    """Every pack the parity is held over, each with its name."""
    return _hydrated() + _crafted()


def _reseal(body: bytes) -> bytes:
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def mutants(blob: bytes, rng: random.Random, count: int) -> list[bytes]:
    """Packs one change away from `blob`: most resealed (their CRC recomputed, so the change
    reaches the record laws), some raw, some truncated, one with a trailing octet."""
    body = blob[:-4]
    out = [blob[: rng.randrange(len(blob))], _reseal(body + b"\x00")]
    for _ in range(count):
        b = bytearray(body)
        for _ in range(rng.choice((1, 1, 2, 4))):
            at = rng.randrange(len(b))
            b[at] = rng.choice((b[at] ^ (1 << rng.randrange(8)), 0, 0xFF, rng.randrange(256)))
        out.append(_reseal(bytes(b)) if rng.random() < 0.9 else bytes(b) + blob[-4:])
    return out


def _oracle_accepts(blob: bytes) -> bool:
    try:
        decode(blob)
    except AbiError:
        return False
    return True


def check_tool(tool: str, work: str, *, mutations: int = 6, seed: int = 20261110) -> list[str]:
    """Hold the C `bcir-pack` at `tool` to the oracle; return the failures (empty when it agrees).

    Every corpus pack: `dis` and `hexdump` at each width are the oracle's octets. Every mutant: the
    two rails accept the same packs, an accepted one lists the same octets, and a refused one exits
    2 with nothing on standard output and `bcir-pack: ` on standard error."""
    failures: list[str] = []
    path = os.path.join(work, "pack.bspk")

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([tool, *args], capture_output=True, timeout=60)

    rng = random.Random(seed)
    for name, blob in corpus():
        with open(path, "wb") as f:
            f.write(blob)
        got = run("dis", path)
        if got.returncode != 0 or got.stdout != format_listing(blob).encode("utf-8"):
            failures.append(f"{name}: dis differs (exit {got.returncode}) {got.stderr[:200]!r}")
        for width in _WIDTHS:
            got = run("hexdump", path, "--width", str(width))
            want = format_hexdump(blob, width=width).encode("utf-8")
            if got.returncode != 0 or got.stdout != want:
                failures.append(f"{name}: hexdump --width {width} differs (exit {got.returncode})")
        for j, mutant in enumerate(mutants(blob, rng, mutations)):
            with open(path, "wb") as f:
                f.write(mutant)
            got = run("dis", path)
            if _oracle_accepts(mutant):
                if got.returncode != 0 or got.stdout != format_listing(mutant).encode("utf-8"):
                    failures.append(f"{name} mutant {j}: the oracle lists it, the C tool does not")
            elif got.returncode != 2 or got.stdout or not got.stderr.startswith(b"bcir-pack: "):
                failures.append(
                    f"{name} mutant {j}: the oracle refuses it, the C tool exits "
                    f"{got.returncode} with {len(got.stdout)} octets of output"
                )
    return failures


def _build(tmp: str) -> str | None:
    from bcir.toolchain import host_c_compiler

    cc = host_c_compiler()
    if cc is None:
        return None
    exe = os.path.join(tmp, "bcir-pack.exe" if os.name == "nt" else "bcir-pack")
    sources = [os.path.join(_RUNTIME_C, s) for s in _SOURCES]
    last = None
    for std in ("-std=c23", "-std=c2x"):  # GCC 13 knows C23 only as c2x
        last = subprocess.run(
            [cc, std, "-O2", "-Wall", "-Wextra", "-I", _RUNTIME_C, *sources, "-o", exe],
            capture_output=True,
            text=True,
        )
        if last.returncode == 0:
            return exe
    raise AssertionError(f"bcir-pack did not build: {last.stderr if last else ''}")


def test_the_fixture_builds_what_the_gate_links():
    from bcir.tests.gate_links import assert_fixture_links

    assert_fixture_links("tools", "bcir-pack", _SOURCES)


def test_the_corpus_reaches_every_wire_version_and_every_record_kind():
    blobs = [blob for _, blob in corpus()]
    assert {blob[4] for blob in blobs} == {1, 2, 3, 4}
    packs = [decode(blob) for blob in blobs]
    assert any(p.prefetches for p in packs) and any(p.blocks for p in packs)
    assert any(s.fence_before for p in packs for s in p.segments)
    traced = [{t.claim_id for t in p.trace_notes} for p in packs]
    assert any(s.claim_id not in t for p, t in zip(packs, traced) for s in p.segments), "untraced"
    assert any(len(p.trace_notes) > len(t) for p, t in zip(packs, traced)), "a claim traced twice"
    assert any(len(blob) > 8192 for blob in blobs), "records longer than the C output chunk"


def test_the_c_listing_and_dump_are_the_oracles_octets():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return  # no C compiler on this tier (quick hides it); the c-runtime tier and the gate run it
        mutations = 24 if os.environ.get("BCIR_THOROUGH") else 6
        failures = check_tool(exe, tmp, mutations=mutations)
        assert not failures, "\n".join(failures[:20])


def test_the_two_command_lines_write_the_same_octets_whatever_the_console():
    """An ASCII console: the Python tool still writes UTF-8 octets with "\\n" line ends, and they are
    the C tool's."""
    pack = StreamPack(source_plan="\U0001fae8")
    pack.segments.append(LaneSegment("ség", 1, 0, Lane.U, 4, "Ā", (), ()))
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        path = os.path.join(tmp, "pack.bspk")
        with open(path, "wb") as f:
            f.write(encode(pack))
        env = dict(os.environ, PYTHONIOENCODING="ascii", PYTHONPATH=_ROOT)
        oracle = subprocess.run(
            [sys.executable, "-m", "bcir.abi.streampack_tool", "dis", path],
            capture_output=True,
            env=env,
            timeout=60,
        )
        native = subprocess.run([exe, "dis", path], capture_output=True, timeout=60)
        assert oracle.returncode == 0, oracle.stderr
        assert native.returncode == 0, native.stderr
        assert oracle.stdout == native.stdout == format_listing(encode(pack)).encode("utf-8")


def test_the_c_command_line_refuses_as_the_python_one_does():
    good = encode(StreamPack())
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        path = os.path.join(tmp, "pack.bspk")
        with open(path, "wb") as f:
            f.write(good)
        missing = os.path.join(tmp, "nowhere.bspk")
        cases = (
            (["--max-bytes", "67", "dis", path], b"at least 68"),
            ([f"--max-bytes={len(good) - 1}", "dis", path], b"exceeds --max-bytes"),
            (["dis", missing], b"cannot open"),
            (["hexdump", path, "--width", "0"], b"width"),
            (["hexdump", path, "--width=65"], b"width"),
            (["hexdump", path, "--width", "+8"], b"--width takes a number"),
            (["dis", path, "--width", "8"], b"unknown option"),
            (["list", path], b"dis or hexdump"),
            (["dis"], b"FILE"),
            ([], b"command"),
        )
        for args, reason in cases:
            got = subprocess.run([exe, *args], capture_output=True, timeout=60)
            assert got.returncode == 2 and not got.stdout, (args, got)
            assert got.stderr.splitlines()[-1].startswith(b"bcir-pack: "), (args, got.stderr)
            assert reason in got.stderr, (args, got.stderr)
        ok = subprocess.run(
            [exe, f"--max-bytes={len(good)}", "hexdump", "--width", "8", path],
            capture_output=True,
            timeout=60,
        )
        assert ok.returncode == 0 and ok.stdout == format_hexdump(good, width=8).encode()
