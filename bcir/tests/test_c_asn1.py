"""Python ↔ C parity for the X.690 decoder.

`runtime/c/bcir_asn1.c` is the freestanding twin of `bcir/asn1/`. Both parse the same
untrusted octets, so a divergence is not a style difference — it is one rail accepting
an encoding the other rejects, which at a trust boundary means the C driver and the
Python oracle disagree about what a peer sent.

The gate compares three things per input: the node tree in document order (class, tag
number, form, content length), the BER verdict, and the DER verdict. Inputs are real
StreamPack projections, the X.690 worked examples, and a seeded adversarial campaign.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from bcir.asn1 import Asn1Error
from bcir.asn1.der import is_der
from bcir.asn1.streampack import encode_pack
from bcir.asn1.tlv import decode_one
from bcir.examples import PROGRAMS
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME_C = os.path.join(_ROOT, "runtime", "c")

#: X.690's own worked examples plus the malformed shapes a decoder must refuse.
_FIXTURES: tuple[bytes, ...] = (
    bytes.fromhex("300a1605536d697468 0101ff".replace(" ", "")),   # §8.9 SEQUENCE
    bytes.fromhex("1a054a6f6e6573"),                               # §8.23.5 primitive
    bytes.fromhex("3a0904034a6f6e04026573"),                       # §8.23.5 definite
    bytes.fromhex("3a8004034a6f6e040265730000"),                   # §8.23.5 indefinite
    bytes.fromhex("0307040a3b5f291cd0"),                           # §8.6.4.2 primitive
    bytes.fromhex("23800303000a3b0305045f291cd00000"),             # §8.6.4.2 constructed
    bytes.fromhex("06032a0304"),                                   # OBJECT IDENTIFIER
    bytes.fromhex("0603883703"),                                   # §8.19 {2 999 3}
    bytes.fromhex("010101"),                                       # §11.1 TRUE != 0xFF
    bytes.fromhex("1a81054a6f6e6573"),                             # §10.1 non-minimal
    bytes.fromhex("3106020102020101"),                             # §11.6 misordered
    bytes.fromhex("0500"),                                         # NULL
    b"", b"\x30", b"\x30\x05", b"\x30\x80", b"\x00\x00", b"\x1f",
    b"\x02\xff", b"\x05\x80", b"\x02\x01", b"\x30\x84\xff\xff\xff\xff",
    b"\x9f\x80\x01\x00",                                           # §8.1.2.4.2 c)
)


def _compiler() -> str | None:
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _build(tmp: str) -> str | None:
    cc = _compiler()
    if cc is None:
        return None
    exe = os.path.join(tmp, "test_asn1")
    build = subprocess.run(
        [cc, "-std=c23", "-O2", "-Wall", "-Wextra", "-Werror", "-I", _RUNTIME_C,
         os.path.join(_RUNTIME_C, "bcir_asn1.c"),
         os.path.join(_RUNTIME_C, "test_asn1.c"), "-o", exe],
        capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    return exe


def _c_view(exe: str, tmp: str, raw: bytes) -> tuple[list[tuple], str, str]:
    path = os.path.join(tmp, "input.der")
    with open(path, "wb") as handle:
        handle.write(raw)
    result = subprocess.run([exe, path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    nodes: list[tuple] = []
    validate = der = "?"
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts[0] == "validate":
            validate = parts[1]
        elif parts[0] == "der":
            der = parts[1]
        elif parts[0] == "error":
            nodes = []
        else:
            depth, cls, number, constructed, indefinite, content_len, _total = parts
            nodes.append((int(depth), int(cls), int(number), int(constructed),
                          int(indefinite), int(content_len)))
    return nodes, validate, der


def _py_view(raw: bytes) -> tuple[list[tuple], str, str]:
    try:
        tlv = decode_one(raw)
    except Asn1Error:
        return [], "error", "error"
    nodes: list[tuple] = []

    def walk(node, depth: int) -> None:
        content_len = (len(node.content) if not node.constructed
                       else sum(_encoded_len(c) for c in node.children))
        nodes.append((depth, int(node.tag.cls), node.tag.number,
                      int(node.tag.constructed), int(node.indefinite), content_len))
        for child in node.children:
            walk(child, depth + 1)

    walk(tlv, 0)
    return nodes, "ok", "ok" if is_der(tlv) else "not-der"


def _encoded_len(node) -> int:
    from bcir.asn1.length import encode_length
    from bcir.asn1.tags import encode_tag

    body = (len(node.content) if not node.constructed
            else sum(_encoded_len(c) for c in node.children))
    if node.indefinite:
        return len(encode_tag(node.tag)) + 1 + body + 2
    return len(encode_tag(node.tag)) + len(encode_length(body)) + body


def _compare(exe: str, tmp: str, raw: bytes, label: str) -> None:
    c_nodes, c_validate, c_der = _c_view(exe, tmp, raw)
    py_nodes, py_validate, py_der = _py_view(raw)
    assert (c_validate == "ok") == (py_validate == "ok"), (
        f"{label}: BER verdict differs (c={c_validate} python={py_validate}) "
        f"for {raw.hex()}")
    if py_validate != "ok":
        return                                   # both rejected: nothing else to compare
    assert c_nodes == py_nodes, (
        f"{label}: node tree differs for {raw.hex()}\n  c     ={c_nodes}\n"
        f"  python={py_nodes}")
    assert (c_der == "ok") == (py_der == "ok"), (
        f"{label}: DER verdict differs (c={c_der} python={py_der}) for {raw.hex()}")


def test_c_decoder_builds_freestanding():
    """No libc, no allocation: the driver rail must link it into a freestanding image."""
    cc = _compiler()
    if cc is None:
        return
    for std in ("c11", "c2x" if "gcc" in cc and "clang" not in cc else "c23"):
        result = subprocess.run(
            [cc, "-ffreestanding", "-nostdlib", f"-std={std}", "-Wall", "-Wextra",
             "-Werror", "-I", _RUNTIME_C, "-c",
             os.path.join(_RUNTIME_C, "bcir_asn1.c"), "-o", os.devnull],
            capture_output=True, text=True)
        assert result.returncode == 0, f"{std}: {result.stderr}"


def test_rails_agree_on_the_x690_worked_examples_and_malformed_input():
    if _compiler() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        assert exe is not None
        for i, raw in enumerate(_FIXTURES):
            _compare(exe, tmp, raw, f"fixture[{i}]")


def test_rails_agree_on_every_streampack_projection():
    if _compiler() is None:
        return
    h, theta = TargetProfile.x86_avx512(), Theta.cool()
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        assert exe is not None
        for name, build in sorted(PROGRAMS.items()):
            module = build()
            der = encode_pack(hydrate(module, optimize(module, h, theta)))
            _compare(exe, tmp, der, name)


def test_rails_agree_over_a_seeded_mutation_campaign():
    """Byte mutations of a real artifact: the two rails must accept/reject in lockstep."""
    import random

    if _compiler() is None:
        return
    module = PROGRAMS["vector_add"]()
    base = encode_pack(hydrate(module, optimize(
        module, TargetProfile.x86_avx512(), Theta.cool())))
    rng = random.Random(690)
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        assert exe is not None
        for i in range(200):
            raw = bytearray(base)
            for _ in range(rng.randint(1, 3)):
                op = rng.random()
                if op < 0.6 and raw:
                    raw[rng.randrange(len(raw))] = rng.getrandbits(8)
                elif op < 0.8 and len(raw) > 4:
                    del raw[rng.randrange(len(raw)):]
                else:
                    raw.extend(rng.getrandbits(8) for _ in range(rng.randint(1, 4)))
            _compare(exe, tmp, bytes(raw), f"mutant[{i}]")


def test_c_rejects_the_nesting_bomb_the_python_rail_rejects():
    """Both rails bound nesting; neither may recurse on attacker-chosen depth."""
    if _compiler() is None:
        return
    bomb = b"\x30\x80" * 128 + b"\x00\x00" * 128
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        assert exe is not None
        _, c_validate, _ = _c_view(exe, tmp, bomb)
        assert c_validate != "ok", "the C rail accepted a 128-deep nesting bomb"
        try:
            decode_one(bomb)
            raise AssertionError("the Python rail accepted a 128-deep nesting bomb")
        except Asn1Error:
            pass
