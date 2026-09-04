"""Deterministic guards for the bounded TMSAO performance-audit rail."""

from __future__ import annotations

import json
import os
import tempfile

from bcir.performance_audit import run_tmsao_audit, write_tmsao_report


def test_audit_covers_each_execution_and_ml_group_with_valid_measurements():
    report = run_tmsao_audit(scale=1, repeats=1)
    groups = {sample.group for sample in report.samples}
    assert groups == {
        "graph",
        "gem",
        "memory",
        "telemetry",
        "quantization",
        "ml-unsupervised",
        "ml-linear-algebra",
        "ml-sequence",
        "ml-training",
        "ml-hardware-policy",
    }
    assert report.timing_is_gate is False
    assert len(report.definition_sha256) == len(report.correctness_sha256) == 64
    for sample in report.samples:
        assert sample.items > 0 and sample.repeats == 1
        assert 0 < sample.min_ns <= sample.median_ns <= sample.max_ns
        assert len(sample.result_sha256) == 64 and sample.metrics


def test_audit_correctness_digest_is_repeatable_while_timings_are_informational():
    first = run_tmsao_audit(scale=1, repeats=1)
    second = run_tmsao_audit(scale=1, repeats=1)
    assert first.definition_sha256 == second.definition_sha256
    assert first.correctness_sha256 == second.correctness_sha256
    assert [(row.group, row.name, row.result_sha256) for row in first.samples] == [
        (row.group, row.name, row.result_sha256) for row in second.samples
    ]


def test_audit_rejects_unbounded_or_coerced_controls():
    for scale, repeats in ((0, 1), (5, 1), (True, 1), (1, 0), (1, 10), (1, 1.5)):
        try:
            run_tmsao_audit(scale=scale, repeats=repeats)
            raise AssertionError("unbounded/fractional audit controls must fail")
        except ValueError:
            pass


def test_report_write_is_atomic_timestamp_free_and_schema_complete():
    report = run_tmsao_audit(scale=1, repeats=1)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "report.json")
        write_tmsao_report(path, report)
        encoded = open(path, "rb").read()
        document = json.loads(encoded)
        assert document["schema"] == "bcir.tmsao_audit.v1"
        assert set(document) == {
            "schema",
            "timing_is_gate",
            "host",
            "scale",
            "repeats",
            "definition_sha256",
            "correctness_sha256",
            "samples",
        }
        assert "timestamp" not in encoded.decode("utf-8").lower()
        previous = encoded
        real_replace = os.replace

        def fail_replace(_source, _target):
            raise OSError("injected publish failure")

        os.replace = fail_replace
        try:
            try:
                write_tmsao_report(path, report)
                raise AssertionError("injected atomic publish failure must propagate")
            except OSError:
                pass
        finally:
            os.replace = real_replace
        assert open(path, "rb").read() == previous
        assert os.listdir(directory) == ["report.json"]


def test_audit_commands_say_what_they_are_not():
    """`bcir-performance-audit` is the command; `bcir-tmsao-audit` stays as an alias that prints
    that it is a precursor and not a TMSAO certificate (every result is TMSAO-4)."""
    import contextlib
    import io
    import os
    import tempfile

    from bcir.performance_audit import main, main_compat

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.json")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            assert main(["--repeats", "1", "--output", out]) == 0
        assert "not a certificate" in stdout.getvalue()
        assert stderr.getvalue() == ""
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            assert main_compat(["--repeats", "1", "--output", out]) == 0
        assert "not implemented" in stderr.getvalue()
        assert "bcir-performance-audit" in stderr.getvalue()
        assert os.path.exists(out)
