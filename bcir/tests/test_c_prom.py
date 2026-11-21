"""N13 in C: the telemetry export's Prometheus text exposition, octet for octet the oracle's.

`runtime/c/bcir_prom.c` is the C twin of `bcir.telemetry_export.to_prometheus` for integer-valued snapshots: the
body a native telemetry or ledger service returns to `GET /metrics`. Held here, over generated snapshots -- names
with dots, digits first, non-ASCII and astral code points; descriptions with newlines and Python's whitespace at
both ends; label values with quotes, backslashes and newlines; counters and gauges; signals that read and signals
that do not, with a known definition and without; values across int64, negative and zero -- with and without the
sensitivity ranking and the integrity witness (frame counters zero and not):

- **The exposition** is `to_prometheus`'s text, octet for octet.
- **The contracts** the oracle cannot state run in `test_prom` itself: a short output, names out of order or not
  UTF-8, a reading without its definition, every NULL, a known snapshot's text, counters past 2**63.
"""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
from types import SimpleNamespace

from bcir.kbcir.cost import DIMS
from bcir.signal_registry import (
    MetricDefinition,
    MetricKind,
    Provenance,
    Reading,
    Temporality,
    Unit,
    default_registry,
)
from bcir.telemetry import TelemetryIntegrity
from bcir.telemetry_export import to_prometheus
from bcir.toolchain import host_c_compiler

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CC = host_c_compiler()  # None in the quick tier, which hides the host's compilers

_KNOWN = {p.definition.name: p.definition for p in default_registry().providers()}
_FRAME_FIELDS = (
    "frames_accepted",
    "frames_rejected",
    "frames_missing",
    "frames_reordered",
    "frames_duplicated",
)
_WORDS = ("thermal", "die", "x", "9lives", "a.b.c", "é", "日本", "\U0001f600", "q:r", "-dash", "_u")
_TEXTS = ("", "plain", "  padded\t", "line\none\n", '"quoted" \\back', "　wide ", "\x85nel\xa0")


def _name(r: random.Random) -> str:
    return ".".join(r.choice(_WORDS) for _ in range(r.randrange(1, 4)))


def _definition(r: random.Random, name: str) -> MetricDefinition:
    counter = r.random() < 0.4
    return MetricDefinition(
        name=name if r.random() < 0.9 else _name(r),
        unit=r.choice(sorted(Unit.ALL)),
        cost_dim=r.choice(list(DIMS) + [None]),
        provenance=r.choice(sorted(Provenance.ALL)),
        description=r.choice(_TEXTS),
        metric_kind=MetricKind.COUNTER if counter else MetricKind.GAUGE,
        temporality=Temporality.CUMULATIVE if counter else Temporality.UNSPECIFIED,
        monotonic=counter,
    )


def _value(r: random.Random) -> int:
    return r.choice((0, 1, -1, 42, -(2**63), 2**63 - 1, r.randrange(-(10**12), 10**12)))


def _hex(text: str) -> str:
    return text.encode().hex() if text else "-"


def _snapshot(r: random.Random) -> dict:
    snap: dict = {}
    for _ in range(r.randrange(0, 8)):
        roll = r.random()
        if roll < 0.15:
            snap[r.choice(sorted(_KNOWN))] = None  # absent, its definition known
        elif roll < 0.25:
            snap["zz." + _name(r)] = None  # absent, no definition
        else:
            name = _name(r)
            snap[name] = Reading(_definition(r, name), _value(r), r.choice(sorted(Provenance.ALL)))
    return {k: snap[k] for k in sorted(snap)}


def _line(snap: dict, channel: str, sens, witness) -> str:
    parts = [_hex(channel), str(len(snap))]
    for name, reading in snap.items():
        if reading is not None:
            d = reading.definition
            prov = reading.provenance
            fields = (1, 1, int(d.metric_kind == MetricKind.COUNTER), d.name, d.description)
        else:
            d = _KNOWN.get(name)
            prov = d.provenance if d is not None else ""
            fields = (0, int(d is not None), 0, d.name if d else "", d.description if d else "")
        cost = (d.cost_dim or "") if d is not None else ""
        unit = d.unit if d is not None else ""
        parts += [
            _hex(name),
            str(fields[0]),
            str(fields[1]),
            str(fields[2]),
            _hex(fields[3]),
            _hex(fields[4]),
        ]
        parts += [
            _hex(cost),
            _hex(prov),
            _hex(unit),
            str(reading.value if reading is not None else 0),
        ]
    parts.append(str(len(sens or ())))
    for row in sens or ():
        parts += [_hex(row.signal), str(row.abs_delta)]
    if witness is None:
        parts.append("-")
    else:
        parts += [
            str(getattr(witness, f))
            for f in (
                "accepted",
                "rejected",
                "dropped",
                "frames_accepted",
                "frames_rejected",
                "frames_missing",
                "frames_reordered",
                "frames_duplicated",
            )
        ]
    return "prom " + " ".join(parts)


def requests(seed: int = 53, count: int = 600):
    """(request lines, the oracle's texts as hex) over generated snapshots."""
    r = random.Random(seed)
    lines, want = [], []
    for _ in range(count):
        snap = _snapshot(r)
        channel = r.choice(("host", 'gpu"0', "a\\b", "line\nbreak", "é"))
        sens = None
        if r.random() < 0.4:
            sens = [
                SimpleNamespace(signal=_name(r), abs_delta=r.randrange(0, 10**9))
                for _ in range(r.randrange(1, 4))
            ]
        witness = None
        if r.random() < 0.5:
            frames = {k: 0 for k in _FRAME_FIELDS}
            roll = r.random()
            if roll < 0.3:  # one frame counter alone: each is frame evidence by itself
                frames[r.choice(_FRAME_FIELDS)] = r.randrange(1, 4)
            elif roll < 0.6:
                frames = {k: r.randrange(0, 3) for k in _FRAME_FIELDS}
            witness = TelemetryIntegrity(
                accepted=r.choice((0, 3, 10**6)),
                rejected=r.randrange(0, 5),
                dropped=r.randrange(0, 5),
                **frames,
            )
        lines.append(_line(snap, channel, sens, witness))
        text = to_prometheus(snap, channel=channel, sensitivity=sens, witness=witness)
        want.append("ok " + text.encode().hex())
    return lines, want


def check_driver(exe: str) -> dict:
    """`test_prom FILE` == the oracle over every snapshot; what the corpus reached. The one comparison the test and
    the gate's `prom` section run."""
    lines, want = requests()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "requests")
        with open(path, "w", encoding="ascii") as f:
            f.write("".join(line + "\n" for line in lines))
        try:
            ran = subprocess.run([exe, path], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AssertionError(f"test_prom did not run: {exc}") from exc
    assert ran.returncode == 0, ran.stdout[-2000:] + ran.stderr[-2000:]
    got = ran.stdout.splitlines()
    assert len(got) == len(want) == len(lines), (len(got), len(want), len(lines))
    for line, w, g in zip(lines, want, got):
        if g != w:
            wt = bytes.fromhex(w[3:]).decode() if w.startswith("ok ") else w
            gt = bytes.fromhex(g[3:]).decode() if g.startswith("ok ") else g
            raise AssertionError((line[:300], wt[:2000], gt[:2000]))
    seen = {
        "snapshots": len(want),
        "series": 0,
        "absent": 0,
        "counters": 0,
        "witnesses": 0,
        "frames": 0,
        "octets": 0,
    }
    for w in want:
        text = bytes.fromhex(w[3:]).decode()
        seen["octets"] += len(text)
        seen["series"] += text.count("\nbcir_signal_up{")
        seen["absent"] += sum(
            1
            for row in text.splitlines()
            if row.startswith("bcir_signal_up{") and row.endswith(" 0")
        )
        seen["counters"] += sum(
            1 for row in text.splitlines() if row.startswith("# TYPE ") and row.endswith(" counter")
        )
        seen["witnesses"] += "bcir_telemetry_blind" in text
        seen["frames"] += "bcir_telemetry_frame_monotonic" in text
    return seen


def _build(tmp: str) -> str | None:
    from bcir.tests.run_all import _is_source_checkout

    if _CC is None or not _is_source_checkout():
        return None
    exe = os.path.join(tmp, "test_prom" + (".exe" if os.name == "nt" else ""))
    argv = [_CC, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", _C]
    argv += [os.path.join(_C, s) for s in ("test_prom.c", "bcir_prom.c")] + ["-o", exe]
    try:
        built = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssertionError(f"test_prom did not build: {exc}") from exc
    assert built.returncode == 0, built.stderr[-3000:]
    return exe


def test_the_c_twin_exposes_as_the_oracle_does():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        seen = check_driver(exe)
        # not vacuous: series that read and that did not, counters, the witness with and without frames
        assert seen["series"] > 1000 and seen["absent"] > 200 and seen["counters"] > 300, seen
        assert seen["witnesses"] > 100 and 50 < seen["frames"] < seen["witnesses"], seen


def test_the_c_twin_holds_its_contracts():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        ran = subprocess.run([exe], capture_output=True, text=True, timeout=300)
        assert ran.returncode == 0 and ran.stdout.strip() == "test_prom: OK", (
            ran.stdout + ran.stderr
        )
