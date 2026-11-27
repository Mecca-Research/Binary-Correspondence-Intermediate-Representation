"""The C subset's ratchet (CSUB-1): `tools/c/csub_ratchet.py` holds `bcir-cc`'s verdict on every production unit to
`runtime/c/csub_ratchet.json`.

The record must name every unit the inventory (`runtime/c/MEMORY_CLASSIFICATION.txt`) lists and a rung for each
blocked one; the gate must refuse each way the compiler and the record can disagree -- a shrink, an unrecorded
promotion, a changed first blocker, an unrecorded or a stale unit, a compile that gives no verdict -- and call a
malformed record INVALID. Each refusal is driven here with a stub compiler whose verdicts the test chooses, so the
rules are witnessed failing (L2, L11) without building anything; where a C compiler is on PATH the gate also runs
over the real `bcir-cc`, as the `csub_ratchet` section does."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import stat
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOOL = os.path.join(_ROOT, "tools", "c", "csub_ratchet.py")


def _is_source_checkout() -> bool:
    return os.path.isfile(_TOOL) and os.path.isfile(
        os.path.join(_ROOT, "runtime", "c", "csub_ratchet.json")
    )


def _ratchet():
    spec = importlib.util.spec_from_file_location("csub_ratchet", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: The units the C subset audit says CSUB-1 brings under the verified compiler (its ladder's first row), and the one
#: whose own change lets the gate admit it (`bcir_per.c`, R18).
_CSUB1_PROMISE = (
    "bcir_asn1.c",
    "bcir_binrec.c",
    "bcir_encode.c",
    "bcir_runtime.c",
    "bcir_exec.c",
    "bcir_control_plane.c",
    "bcir_ai_kernels.c",
    "bcir_per.c",
)


def test_the_record_names_every_production_unit_and_a_rung_for_each_blocker():
    """Every unit the inventory lists is recorded, and nothing else; each blocked one names its first refusal, the
    construct and a rung of the ladder; and the units CSUB-1 promises are recorded clean."""
    if not _is_source_checkout():
        return
    r = _ratchet()
    inventory, record = r.read_inventory(), r.read_record()
    assert set(record) == set(inventory), set(record) ^ set(inventory)
    clean = {u for u, e in record.items() if e["verdict"] == "clean"}
    assert all(u in clean for u in _CSUB1_PROMISE), sorted(set(_CSUB1_PROMISE) - clean)
    for unit, entry in record.items():
        if entry["verdict"] == "blocked":
            assert (
                entry["rung"] in r.RUNGS and entry["construct"].strip() and entry["blocker"].strip()
            ), unit


def _stub_compiler(d: str, verdicts: dict) -> str:
    """An executable standing in for `bcir-cc`: for each unit named in `verdicts`, the exit status and the line it
    prints (`crash` dies by a signal, `silent` exits 1 printing nothing); any other unit compiles clean. Its logic is a
    Python file; the executable is a shebang script on POSIX and a `.cmd` file on Windows, which runs no shebang."""
    table = os.path.join(d, "verdicts.json")
    with open(table, "w", encoding="utf-8") as fh:
        json.dump(verdicts, fh)
    logic = os.path.join(d, "stub_cc.py")
    with open(logic, "w", encoding="utf-8") as fh:
        fh.write(
            "import json, os, signal, sys\n"
            f"v = json.load(open({table!r})).get(os.path.basename(sys.argv[-1]))\n"
            "if v is None: sys.exit(0)\n"
            "if v == 'crash': os.kill(os.getpid(), signal.SIGSEGV)\n"
            "if v == 'silent': sys.exit(1)\n"
            "sys.stderr.write(sys.argv[-1] + ': ' + v + '\\n'); sys.exit(1)\n"
        )
    if os.name == "nt":
        stub = os.path.join(d, "bcir-cc.cmd")
        with open(stub, "w", encoding="utf-8") as fh:
            fh.write(f'@"{sys.executable}" "{logic}" %*\n')
        return stub
    stub = os.path.join(d, "bcir-cc")
    with open(stub, "w", encoding="utf-8") as fh:
        fh.write(f'#!/bin/sh\nexec {_sh_quote(sys.executable)} {_sh_quote(logic)} "$@"\n')
    os.chmod(stub, os.stat(stub).st_mode | stat.S_IXUSR)
    return stub


def _sh_quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def _world(d: str, units: dict, record_units: dict) -> tuple:
    """A runtime directory holding `units` (unit -> class), its inventory, and a record of `record_units`."""
    root = os.path.join(d, "c")
    os.makedirs(root)
    for unit in units:
        open(os.path.join(root, unit), "w").close()
    inventory = os.path.join(root, "MEMORY_CLASSIFICATION.txt")
    with open(inventory, "w", encoding="utf-8") as fh:
        fh.write("# inventory\n" + "".join(f"{cls} {u}\n" for u, cls in units.items()))
    record = os.path.join(root, "csub_ratchet.json")
    with open(record, "w", encoding="utf-8") as fh:
        json.dump({"schema": "bcir.csub-ratchet/1", "units": record_units}, fh)
    return root, inventory, record


_BLOCKED = {
    "verdict": "blocked",
    "blocker": "parse error: unknown struct",
    "construct": "a table",
    "rung": "CSUB-2",
}


def test_the_ratchet_refuses_every_way_the_compiler_and_the_record_disagree():
    """The gate passes where the compiler gives what the record says, and fails, naming the unit and the rule, on each
    disagreement: a clean unit refused (a shrink), a blocked one compiling clean (an unrecorded promotion), a blocked
    one refused for another reason, a unit the inventory lists and the record lacks, one the record lists and the
    inventory lacks, and a compile that crashes or refuses without a reason."""
    r = _ratchet()
    units = {"a.c": "freestanding_core", "b.c": "hosted_tool"}
    agree = {"a.c": {"verdict": "clean"}, "b.c": dict(_BLOCKED)}
    cases = [
        ({"b.c": "parse error: unknown struct"}, agree, None),
        (
            {"a.c": "parse error: (", "b.c": "parse error: unknown struct"},
            agree,
            "a.c: recorded clean, now refused",
        ),
        ({}, agree, "b.c: now compiles R1-R18 clean -- promote it"),
        ({"b.c": "parse error: unknown type"}, agree, "b.c: the first refusal changed"),
        ({"b.c": "parse error: unknown struct"}, {"b.c": dict(_BLOCKED)}, "a.c: not recorded"),
        (
            {"b.c": "parse error: unknown struct"},
            {**agree, "gone.c": {"verdict": "clean"}},
            "gone.c: recorded, but the inventory lists no such production unit",
        ),
        (
            {"a.c": "crash", "b.c": "parse error: unknown struct"},
            agree,
            "a.c: bcir-cc gave no verdict",
        ),
        (
            {"a.c": "silent", "b.c": "parse error: unknown struct"},
            agree,
            "a.c: bcir-cc gave no verdict",
        ),
    ]
    for verdicts, record_units, want in cases:
        with tempfile.TemporaryDirectory() as d:
            root, inventory, record = _world(d, units, record_units)
            stub = _stub_compiler(d, verdicts)
            got = r.check(
                r.read_inventory(inventory),
                r.read_record(record),
                {u: r.compile_unit(stub, u, root) for u in sorted(units)},
            )
            rc = r.main(
                ["--cc", stub, "--inventory", inventory, "--record", record, "--root", root]
            )
        if want is None:
            assert got == [] and rc == 0, got
        else:
            assert rc == 1 and len(got) == 1 and got[0].startswith(want), (want, got)


def test_a_malformed_record_or_inventory_is_invalid():
    """A record or an inventory the gate cannot read as one is INVALID (exit 2), never a pass: a repeated key, an
    unknown verdict or rung, a blocked unit with no construct, a clean one carrying more than its verdict, another
    schema, a unit listed twice, and a compiler that is not an executable."""
    r = _ratchet()
    units = {"a.c": "freestanding_core"}
    bad_records = (
        '{"schema": "bcir.csub-ratchet/1", "units": {"a.c": {"verdict": "clean"}, "a.c": {"verdict": "clean"}}}',
        '{"schema": "bcir.csub-ratchet/1", "units": {"a.c": {"verdict": "maybe"}}}',
        json.dumps(
            {"schema": "bcir.csub-ratchet/1", "units": {"a.c": {**_BLOCKED, "rung": "CSUB-9"}}}
        ),
        json.dumps(
            {"schema": "bcir.csub-ratchet/1", "units": {"a.c": {**_BLOCKED, "construct": " "}}}
        ),
        json.dumps(
            {
                "schema": "bcir.csub-ratchet/1",
                "units": {"a.c": {"verdict": "clean", "rung": "CSUB-2"}},
            }
        ),
        json.dumps({"schema": "bcir.csub-ratchet/2", "units": {"a.c": {"verdict": "clean"}}}),
        "not json",
    )
    for text in bad_records:
        with tempfile.TemporaryDirectory() as d:
            root, inventory, record = _world(d, units, {})
            with open(record, "w", encoding="utf-8") as fh:
                fh.write(text)
            stub = _stub_compiler(d, {})
            assert (
                r.main(["--cc", stub, "--inventory", inventory, "--record", record, "--root", root])
                == 2
            ), text
    with tempfile.TemporaryDirectory() as d:
        root, inventory, record = _world(d, units, {"a.c": {"verdict": "clean"}})
        stub = _stub_compiler(d, {})
        with open(inventory, "a", encoding="utf-8") as fh:
            fh.write("hosted_tool a.c\n")
        assert (
            r.main(["--cc", stub, "--inventory", inventory, "--record", record, "--root", root])
            == 2
        )
        assert r.main(["--cc", os.path.join(d, "none"), "--record", record, "--root", root]) == 2


def test_the_ratchet_holds_on_the_built_compiler():
    """Where a C compiler is on PATH: `bcir-cc`, built from this tree, gives each production unit exactly the
    verdict the record names -- the `csub_ratchet` section's check, in the test suite too."""
    if not _is_source_checkout() or shutil.which("cc") is None and shutil.which("clang") is None:
        return
    from bcir.tests.test_c_cfront import _CC, _build_bcir_cc, _session_build_dir

    if not _CC:
        return
    exe = _build_bcir_cc(_session_build_dir())
    assert _ratchet().main(["--cc", exe]) == 0


# CSUB-3c: the runtime's one copy of its fixed-width integer loads and stores. Fifteen units had each defined their own
# (`rd16`, `rd32`, `rd64`, `wr16`, `wr32`, `wr64`, `ld32`, `st32`, `read_u32`, `rd32be`, `load64_be`): one mechanism,
# fifteen copies (L14). `runtime/c/bcir_bytes.h` is the one copy.
_BYTES_H = os.path.join(_ROOT, "runtime", "c", "bcir_bytes.h")
#: A load or a store of a fixed-width integer through a byte pointer, by its name's shape -- every spelling the copies
#: took, and `get64`, `put32` and `le32` beside them. The scan's declared scope: a static function of that name and of
#: that first parameter; a helper of another name is the review's to catch, not this scan's.
_BYTE_HELPER = re.compile(
    r"^static\s+(?:inline\s+)?(?:uint(?:16|32|64)_t|void)\s+"
    r"((?:rd|wr|ld|st|load|store|read|write|get|put|le|be)_?u?(?:16|32|64)(?:_?(?:be|le))?)"
    r"\s*\(\s*(?:const\s+)?(?:uint8_t|unsigned\s+char)\s*\*",
    re.M,
)
#: The header's functions: little- and big-endian loads of every width, little-endian stores of every width, and the
#: big-endian 32-bit store the digests write
_BYTES_FUNCS = (
    "bcir_ld16le",
    "bcir_ld32le",
    "bcir_ld64le",
    "bcir_st16le",
    "bcir_st32le",
    "bcir_st64le",
    "bcir_ld16be",
    "bcir_ld32be",
    "bcir_ld64be",
    "bcir_st32be",
)


def _runtime_sources() -> list:
    """Every C unit and header of the runtime but its test drivers and fuzzers, which keep their own encoders: an
    independent spelling of the bytes is what they check the runtime's against."""
    c = os.path.join(_ROOT, "runtime", "c")
    out = []
    for root, _dirs, files in os.walk(c):
        for name in sorted(files):
            if name.endswith((".c", ".h")) and not name.startswith(("test_", "fuzz_")):
                out.append(os.path.join(root, name))
    return out


def test_the_runtime_has_one_copy_of_its_byte_order_helpers():
    """CSUB-3c. No unit or header of the runtime but `bcir_bytes.h` defines a load or a store of a fixed-width integer
    through a byte pointer, and the header defines each of its functions once. The scan is witnessed on every spelling
    the fifteen copies took -- `rd16` of a `const unsigned char *` and of a `BCIR_RESTRICT` one among them -- so it is
    not vacuous, and the units that had the copies include the header."""
    if not os.path.isfile(_BYTES_H):
        return
    for spelling in (
        "static uint16_t rd16(const uint8_t *p) {",
        "static uint16_t rd16(const unsigned char *p) {",
        "static uint16_t rd16(const uint8_t *BCIR_RESTRICT p) {",
        "static uint32_t rd16(const uint8_t *p) { return (uint32_t)p[0]; }",
        "static void wr64(uint8_t *p, uint64_t v) {",
        "static uint64_t load64_be(const uint8_t *p) {",
        "static uint32_t read_u32(const uint8_t *p) {",
        "static uint32_t ld32(const uint8_t *p) {",
        "static void st32(uint8_t *p, uint32_t v) {",
        "static uint32_t rd32be(const uint8_t *p) {",
        "static inline uint64_t get64(const uint8_t *p) {",
    ):
        assert _BYTE_HELPER.search(spelling), spelling
    sources = _runtime_sources()
    found = [
        (os.path.relpath(p, _ROOT), m.group(1))
        for p in sources
        if p != _BYTES_H
        for m in _BYTE_HELPER.finditer(open(p, encoding="utf-8").read())
    ]
    assert not found, found
    header = open(_BYTES_H, encoding="utf-8").read()
    for fn in _BYTES_FUNCS:
        assert len(re.findall(r"^static inline \w+ " + fn + r"\(", header, re.M)) == 1, fn
    users = {
        os.path.basename(p)
        for p in sources
        if '#include "bcir_bytes.h"' in open(p, encoding="utf-8").read()
    }
    assert users >= {
        "bcir_artifact_bundle.c", "bcir_control_plane.c", "bcir_ed25519.c", "bcir_handoff.c", "bcir_jer.c",
        "bcir_plan_sign.c", "bcir_q8_model.c", "bcir_ring.c", "bcir_runtime.c", "bcir_sha256.c",
        "bcir_shard_manifest.c", "bcir_sp_listing.c", "bcir_telemetry_envelope.c", "bcir_telemetry_frame.c",
        "bcir_tokload.c",
    }, users  # fmt: skip


def test_the_byte_order_helpers_are_int_from_bytes_and_lower_alike_on_both_rails():
    """CSUB-3c. Each load of `bcir_bytes.h`, built by the host's C compiler, reads what Python's `int.from_bytes` reads
    of the same octets, and each store writes what `int.to_bytes` writes, on every width and both byte orders: 0, all
    ones, a lone top bit, ascending octets, and seeded ones. And the header is in the verified subset: a unit of every
    function lowers alike on both C front-end rails, R18 clean, each rail's emit run as the host compiler's build of
    it (`_equiv`)."""
    if not os.path.isfile(_BYTES_H):
        return
    from bcir.tests.test_c_cfront import (
        _CC,
        _build_frontend,
        _c_run,
        _equiv,
        _oracle,
        _session_build_dir,
    )

    if not _CC:
        return
    rng = random.Random(0x5EED)
    cases = [bytes(8), b"\xff" * 8, b"\x80" + bytes(7), bytes(7) + b"\x80", bytes(range(1, 9))]
    cases += [bytes(rng.randrange(256) for _ in range(8)) for _ in range(59)]
    rows = "".join("{" + ",".join(str(b) for b in c) + "},\n" for c in cases)
    src = (
        '#include <stdio.h>\n#include "bcir_bytes.h"\n'
        f"static const uint8_t in[{len(cases)}][8] = {{\n{rows}}};\n"
        'static void hex(const uint8_t *p, int n) { for (int i = 0; i < n; i++) printf("%02x", p[i]); printf(" "); }\n'
        "int main(void) {\n"
        f"  for (int k = 0; k < {len(cases)}; k++) {{\n"
        "    const uint8_t *p = in[k]; uint8_t o[8] = {0};\n"
        "    uint64_t v = bcir_ld64le(p);\n"
        '    printf("%u %lu %llu %u %lu %llu ", (unsigned)bcir_ld16le(p), (unsigned long)bcir_ld32le(p),\n'
        "           (unsigned long long)v, (unsigned)bcir_ld16be(p), (unsigned long)bcir_ld32be(p),\n"
        "           (unsigned long long)bcir_ld64be(p));\n"
        "    bcir_st16le(o, (uint16_t)v); hex(o, 2); bcir_st32le(o, (uint32_t)v); hex(o, 4);\n"
        '    bcir_st64le(o, v); hex(o, 8); bcir_st32be(o, (uint32_t)v); hex(o, 4); printf("\\n");\n'
        "  }\n  return 0;\n}\n"
    )
    with tempfile.TemporaryDirectory() as d:
        c, e = os.path.join(d, "bytes.c"), os.path.join(d, "bytes")
        with open(c, "w", encoding="utf-8") as fh:
            fh.write(src)
        b = subprocess.run(
            [
                _CC,
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                os.path.dirname(_BYTES_H),
                c,
                "-o",
                e,
            ],
            capture_output=True,
            text=True,
        )
        assert b.returncode == 0, b.stderr[-2000:]
        got = subprocess.run([e], capture_output=True, text=True).stdout.splitlines()
    want = []
    for c8 in cases:
        v = int.from_bytes(c8, "little")
        want.append(
            " ".join(
                str(x)
                for x in (
                    int.from_bytes(c8[:2], "little"),
                    int.from_bytes(c8[:4], "little"),
                    v,
                    int.from_bytes(c8[:2], "big"),
                    int.from_bytes(c8[:4], "big"),
                    int.from_bytes(c8, "big"),
                )
            )
            + " "
            + (v & 0xFFFF).to_bytes(2, "little").hex()
            + " "
            + (v & 0xFFFFFFFF).to_bytes(4, "little").hex()
            + " "
            + v.to_bytes(8, "little").hex()
            + " "
            + (v & 0xFFFFFFFF).to_bytes(4, "big").hex()
            + " "
        )
    assert got == want, next(
        (k, got[k], want[k]) for k in range(len(want)) if k >= len(got) or got[k] != want[k]
    )
    # ... and the header is in the verified subset, on both rails
    header = open(_BYTES_H, encoding="utf-8").read()
    unit = header + (
        "uint64_t f(uint32_t k) {\n  uint8_t b[8], o[8];\n"
        "  for (uint32_t i = 0; i < 8u; i++) b[i] = (uint8_t)(k * (2u * i + 1u) + i * 37u);\n"
        "  uint64_t acc = bcir_ld16le(b) ^ bcir_ld32le(b + 1) ^ bcir_ld64le(b) ^ bcir_ld16be(b + 3)"
        " ^ bcir_ld32be(b + 2) ^ bcir_ld64be(b);\n"
        "  bcir_st16le(o, (uint16_t)acc); bcir_st32le(o + 2, (uint32_t)(acc >> 7)); acc ^= bcir_ld64le(o);\n"
        "  bcir_st64le(o, acc * 3u); bcir_st32be(o + 4, k); return acc + bcir_ld64le(o);\n}\n"
    )
    summary, r, entry = _oracle(unit)
    assert " ok=1 " in summary, summary
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bytes_unit.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(unit)
        twin_summary, twin_emit = _c_run(exe, path)
    assert twin_summary == summary, (twin_summary, summary)
    for rail, emit in (("oracle", "\n".join(r.emitted.values())), ("twin", twin_emit)):
        assert _equiv(unit, emit, entry) == "MATCH", rail
