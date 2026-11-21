"""N12's run-time half in C: strided-view admission and the DMA descriptor ring, at parity with the oracle.

`runtime/c/bcir_dma.c` is the C twin of `bcir.kbcir.device_manifest.check_strided_view` (D-R4) and
`bcir.kbcir.dma.dma_descriptors` (A2), what a driver needs at run time; the planning half (pricing, the transfer
module, its events) stays the oracle's. Held here, over generated banks and view pairs -- every rank to 3, dense,
gapped and scattering strides, aligned and not, inside and past the bank, on and off the native tile, one bank and
two, overlapping and not, shapes and widths unequal:

- **Admission** is the oracle's message for message: each refusal it writes is one bit of `bcir_dma_admit`'s.
- **The ring** is `dma_descriptors`' descriptor for descriptor, and each refusal its refusal.
- **The contracts** the oracle cannot state run in `test_dma` itself: a ring past its capacity, arithmetic past 64
  bits, every NULL.
"""

from __future__ import annotations

import os
import random
import subprocess
import tempfile

from bcir.kbcir.device_manifest import DeviceManifest, MemoryBank, StridedView, check_strided_view
from bcir.kbcir.dma import dma_descriptors
from bcir.model import Domain
from bcir.toolchain import host_c_compiler

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CC = host_c_compiler()  # None in the quick tier, which hides the host's compilers

# bcir_dma.h's refusal bits and status codes
BANK, EMPTY, RANK, EXTENT, ELEM, ALIGN, CAPACITY, TILE_OUTER, TILE_INNER = (
    1 << k for k in range(9)
)
ERR_SRC, ERR_DST, ERR_SHAPE, ERR_WIDTH, ERR_OVERLAP = 2, 3, 4, 5, 6


def refusal_bits(view: StridedView, man: DeviceManifest) -> int:
    """The oracle's admission messages as bcir_dma_admit's bits: one bit per message."""
    msgs = check_strided_view(view, man)
    bits = 0
    for kind, bit in (
        ("no bank", BANK),
        ("empty shape", EMPTY),
        ("dimensions but", RANK),
        ("extents and strides must be >= 1", EXTENT),
        ("elem_bytes must be", ELEM),
        ("violates the bank alignment", ALIGN),
        ("exceeds the bank capacity", CAPACITY),
    ):
        hits = [m for m in msgs if kind in m]
        assert len(hits) <= 1, msgs
        bits |= bit if hits else 0
    tiles = [m for m in msgs if "native tile" in m]
    if tiles:
        tile = man.bank(view.bank).native_tile
        off = [d % tile != 0 for d in view.shape[-2:]]
        assert len(tiles) == sum(off), (msgs, view)
        # shape[-2:] is (outer, inner) for two or more dimensions, (inner,) for one
        if off[-1]:
            bits |= TILE_INNER
        if len(off) == 2 and off[0]:
            bits |= TILE_OUTER
    assert bits or not msgs, msgs
    return bits


def _text(view: StridedView, names) -> str:
    bank = names.index(view.bank) if view.bank in names else len(names)
    shape, strides = list(view.shape), list(view.strides)
    return " ".join(
        map(
            str,
            [bank, view.offset_bytes, view.elem_bytes, len(shape), *shape, len(strides), *strides],
        )
    )


def _random_view(r: random.Random, man: DeviceManifest, names) -> StridedView:
    bank = r.choice(names) if r.random() > 0.04 else "nowhere"
    rank = r.choice((1, 2, 2, 3)) if r.random() > 0.03 else 0
    shape = [r.choice((1, 2, 3, 4, 8, 12, 16)) if r.random() > 0.03 else 0 for _ in range(rank)]
    style = r.randrange(4)
    elems = []  # a dense row-major stride vector, then perturbed
    acc = 1
    for d in reversed(shape):
        elems.append(acc)
        acc *= max(d, 1)
    strides = list(reversed(elems))
    if style == 1 and rank:  # gapped outer rows
        strides = [s * r.choice((2, 3)) for s in strides[:-1]] + strides[-1:]
    elif style == 2 and rank:  # a non-unit innermost stride: scatter
        strides = [s * 2 for s in strides]
    elif style == 3 and rank:
        strides = [r.randrange(1, 40) for _ in range(rank)]
    if r.random() < 0.03 and rank:
        strides = strides[:-1]
    if r.random() < 0.03 and strides:
        strides[r.randrange(len(strides))] = 0
    elem = r.choice((1, 2, 4, 8)) if r.random() > 0.03 else 0
    align = man.bank(bank).align if bank != "nowhere" else 1
    offset = r.randrange(0, 2048 // align + 1) * align
    if r.random() < 0.1:
        offset += r.randrange(1, 8)
    return StridedView(bank, offset, tuple(shape), tuple(strides), elem)


def _manifest(r: random.Random) -> DeviceManifest:
    banks = tuple(
        MemoryBank(
            f"b{k}",
            Domain.RAM,
            r.choice((1 << 12, 1 << 14, 1 << 20)),
            native_tile=r.choice((1, 1, 2, 4, 8)),
            align=r.choice((1, 4, 64)),
        )
        for k in range(3)
    )
    distance = ((0, 256, 256), (256, 0, 256), (256, 256, 0))
    return DeviceManifest(device="parity", banks=banks, distance=distance)


def _expected(src: StridedView, dst: StridedView, man: DeviceManifest) -> str:
    try:
        descs = dma_descriptors(src, dst, man)
    except ValueError as exc:
        text = str(exc)
        if text.startswith("dma src view refused"):
            return f"ERR {ERR_SRC} {refusal_bits(src, man)}"
        if text.startswith("dma dst view refused"):
            return f"ERR {ERR_DST} {refusal_bits(dst, man)}"
        for prefix, code in (
            ("dma shape mismatch", ERR_SHAPE),
            ("dma element width mismatch", ERR_WIDTH),
            ("dma overlap", ERR_OVERLAP),
        ):
            if text.startswith(prefix):
                return f"ERR {code} 0"
        raise
    return f"ok {len(descs)}" + "".join(f" {d.src_off}:{d.dst_off}:{d.nbytes}" for d in descs)


def requests(seed: int = 41, manifests: int = 25, pairs: int = 40):
    """(request lines, the oracle's answers) over generated manifests and view pairs."""
    r = random.Random(seed)
    lines, want = [], []
    for _ in range(manifests):
        man = _manifest(r)
        names = [b.name for b in man.banks]
        lines.append("reset")
        lines += [f"bank {b.capacity_bytes} {b.native_tile} {b.align}" for b in man.banks]
        # each bank's edge: a dense view ending at its last byte, and the same view one alignment on, past it
        for b in man.banks:
            eb = 4
            for off in (0, b.align):
                edge = StridedView(b.name, off, (b.capacity_bytes // eb,), (1,), eb)
                lines.append("admit " + _text(edge, names))
                want.append(str(refusal_bits(edge, man)))
        for _ in range(pairs):
            src = _random_view(r, man, names)
            if r.random() < 0.6:  # the same shape and width, often on one bank
                dst = _random_view(r, man, names)
                dst = StridedView(
                    dst.bank, dst.offset_bytes, src.shape, dst.strides, src.elem_bytes
                )
                if len(dst.strides) != len(dst.shape):
                    dst = StridedView(
                        dst.bank, dst.offset_bytes, src.shape, src.strides, src.elem_bytes
                    )
                if r.random() < 0.3:
                    dst = StridedView(
                        src.bank, dst.offset_bytes, dst.shape, dst.strides, dst.elem_bytes
                    )
                if r.random() < 0.08:  # the shape kept, the width not
                    width = 8 if dst.elem_bytes != 8 else 2
                    dst = StridedView(dst.bank, dst.offset_bytes, dst.shape, dst.strides, width)
            else:
                dst = _random_view(r, man, names)
            for view in (src, dst):
                lines.append("admit " + _text(view, names))
                want.append(str(refusal_bits(view, man)) if view.bank in names else str(BANK))
            lines.append(f"dma {_text(src, names)} {_text(dst, names)}")
            want.append(_expected(src, dst, man))
    return lines, want


def check_driver(exe: str) -> dict:
    """`test_dma FILE` == the oracle over every request; what the corpus reached. The one comparison the test and
    the gate's `dma` section run."""
    lines, want = requests()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "requests")
        with open(path, "w", encoding="ascii") as f:
            f.write("".join(line + "\n" for line in lines))
        try:
            ran = subprocess.run([exe, path], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AssertionError(f"test_dma did not run: {exc}") from exc
    assert ran.returncode == 0, ran.stdout[-2000:] + ran.stderr[-2000:]
    got = ran.stdout.splitlines()
    asked = [line for line in lines if line.startswith(("admit", "dma"))]
    assert len(got) == len(want) == len(asked), (len(got), len(want))
    for line, w, g in zip(asked, want, got):
        assert g == w, (line, w[:200], g[:200])
    seen = {"admitted": 0, "refused": 0, "rings": 0, "descriptors": 0}
    for line, w in zip(asked, want):
        if line.startswith("admit"):
            seen["admitted" if w == "0" else "refused"] += 1
        elif w.startswith("ok"):
            seen["rings"] += 1
            seen["descriptors"] += int(w.split()[1])
        else:
            code = int(w.split()[1])
            seen[f"err{code}"] = seen.get(f"err{code}", 0) + 1
    return seen


def _build(tmp: str) -> str | None:
    from bcir.tests.run_all import _is_source_checkout

    if _CC is None or not _is_source_checkout():
        return None
    exe = os.path.join(tmp, "test_dma" + (".exe" if os.name == "nt" else ""))
    argv = [_CC, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", _C]
    argv += [os.path.join(_C, s) for s in ("test_dma.c", "bcir_dma.c")] + ["-o", exe]
    try:
        built = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssertionError(f"test_dma did not build: {exc}") from exc
    assert built.returncode == 0, built.stderr[-3000:]
    return exe


def test_the_c_twin_admits_and_programs_as_the_oracle_does():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        seen = check_driver(exe)
        # not vacuous: views admitted and refused, rings built, and every pair refusal reached
        assert seen["admitted"] > 300 and seen["refused"] > 200 and seen["rings"] > 100, seen
        assert all(
            seen.get(f"err{c}", 0) > 3
            for c in (ERR_SRC, ERR_DST, ERR_SHAPE, ERR_WIDTH, ERR_OVERLAP)
        ), seen


def test_the_c_twin_holds_its_contracts():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        ran = subprocess.run([exe], capture_output=True, text=True, timeout=300)
        assert ran.returncode == 0 and ran.stdout.strip() == "test_dma: OK", ran.stdout + ran.stderr
