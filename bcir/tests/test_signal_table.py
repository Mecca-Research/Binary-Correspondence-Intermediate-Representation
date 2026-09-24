"""The version-zero signal definition table (GEM+ roadmap G15, staged plan S3-B).

A telemetry envelope names its signal by a u32 ID, and a resident driver cannot import the Python
registry that assigns them. So the taxonomy must exist as bytes both rails agree on: one 64-byte
row per built-in definition, generated into runtime/c/bcir_signal_table.h (never hand-edited), an
ID-range policy that keeps BCIR's, vendors' and devices' IDs apart, and the unknown-required-signal
law the intake applies.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile

from bcir.kbcir.cost import DIMS
from bcir.signal_registry import MetricDefinition, MetricKind, Temporality, Unit, default_registry
from bcir.signal_table import (
    C_HEADER,
    COST_DIM_NONE,
    NAME_WIDTH,
    ROW_SIZE,
    SignalTableError,
    admit_signal,
    builtin_definitions,
    decode_row,
    emit_c_header,
    encode_row,
    encode_table,
    main,
    signal_range,
    table_ids,
)
from bcir.tests import ring_fixtures as rf

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_the_table_is_every_registry_id_once_in_order():
    defs = builtin_definitions()
    ids = [d.signal_id for d in defs]
    assert ids == sorted(set(ids)) == list(range(1, 16))
    # every ID the default registry serves is in the table, with the same definition
    served = {p.definition.signal_id: p.definition for p in default_registry().providers()}
    for signal_id, definition in served.items():
        assert definition in defs, signal_id
    assert table_ids() == frozenset(ids)


def test_a_row_is_sixty_four_bytes_and_decodes_to_its_definition():
    table = encode_table()
    assert len(table) == ROW_SIZE * len(builtin_definitions())
    for index, definition in enumerate(builtin_definitions()):
        row = table[index * ROW_SIZE : (index + 1) * ROW_SIZE]
        assert row == encode_row(definition)
        fields = decode_row(row)
        assert fields[0] == definition.signal_id and fields[-1] == definition.name
        cost = COST_DIM_NONE if definition.cost_dim is None else DIMS.index(definition.cost_dim)
        assert fields[7] == cost and fields[8] == definition.min_interval_ns


def test_the_id_ranges_partition_the_u32_space():
    assert signal_range(0) == "unassigned"
    assert signal_range(1) == signal_range(0xFFFF) == "bcir"
    assert signal_range(0x10000) == signal_range(0x7FFFFFFF) == "vendor"
    assert signal_range(0x80000000) == signal_range(0xFFFFFFFE) == "device"
    assert signal_range(0xFFFFFFFF) == "reserved"
    for bad in (-1, 1 << 32, "1", True):
        try:
            signal_range(bad)
        except SignalTableError:
            continue
        raise AssertionError(bad)


def test_a_row_refuses_what_it_cannot_carry():
    good = builtin_definitions()[0]
    for label, bad in {
        "vendor-range id": MetricDefinition("x.y", Unit.PERCENT, None, signal_id=0x10000),
        "unassigned id": MetricDefinition("x.y", Unit.PERCENT, None, signal_id=0),
        "long name": MetricDefinition("n" * NAME_WIDTH, Unit.PERCENT, None, signal_id=99),
        "non-ascii name": MetricDefinition("thermal.é", Unit.PERCENT, None, signal_id=99),
        "invalid definition": MetricDefinition(
            "c", Unit.COUNT, None, signal_id=99, metric_kind=MetricKind.COUNTER
        ),
    }.items():
        try:
            encode_row(bad)
        except SignalTableError:
            continue
        raise AssertionError(label)
    try:
        encode_table([good, good])
    except SignalTableError:
        pass
    else:
        raise AssertionError("duplicate rows")
    counter = MetricDefinition(
        "c.x",
        Unit.COUNT,
        None,
        signal_id=99,
        metric_kind=MetricKind.COUNTER,
        temporality=Temporality.DELTA,
        monotonic=True,
    )
    assert decode_row(encode_row(counter))[4] == 1  # the monotonic flag


def test_the_unknown_required_signal_law():
    known = table_ids()
    assert admit_signal(1, required=True, known=known) == "known"
    assert admit_signal(1, required=False, known=known) == "known"
    assert admit_signal(0x10001, required=True, known=known) == "refused"
    assert admit_signal(0x10001, required=False, known=known) == "skipped"
    assert admit_signal(16, required=True) == "refused"  # a BCIR-range ID the table lacks


def test_the_generated_c_table_is_the_generator_s_output():
    path = os.path.join(_ROOT, C_HEADER)
    if not os.path.isfile(path):
        from bcir.tests.run_all import _is_source_checkout

        assert not _is_source_checkout(), f"{C_HEADER} is missing from the checkout"
        return
    assert open(path, encoding="utf-8", newline="").read() == emit_c_header()
    code, said = _cli(["--check"])
    assert code == 0 and said.startswith("PASS"), said


def _cli(argv) -> tuple[int, str]:
    """Run the generator's CLI with both streams captured (a PASS goes to stdout, a FAIL to
    stderr), so the witness asserts what it said instead of printing it into the suite's log."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = main(argv)
    return code, out.getvalue()


def test_the_check_refuses_a_drifted_copy():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "table.h")
        assert _cli(["--emit", "--path", path])[0] == 0
        code, said = _cli(["--check", "--path", path])
        assert code == 0 and said.startswith("PASS") and "matches the generator" in said
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("/* hand edit */\n")
        code, said = _cli(["--check", "--path", path])
        assert code == 1 and said.startswith("FAIL") and "differs from the generator" in said
        code, said = _cli(["--check", "--path", os.path.join(tmp, "missing.h")])
        assert code == 1 and said.startswith("FAIL") and "missing.h" in said


def test_the_c_rows_are_the_python_rows():
    with tempfile.TemporaryDirectory() as tmp:
        exe = rf.build_harness(tmp)
        if exe is None:
            return
        assert rf.c_signals(exe) == [encode_row(d).hex() for d in builtin_definitions()]
