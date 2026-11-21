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
import shutil
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
