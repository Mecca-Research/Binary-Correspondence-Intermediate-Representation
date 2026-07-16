"""Non-flaky performance budgets + trend tracking for the measured-evidence rail.

The benchmark rail (`bcir.bench`) is environment-dependent: the same kernel times
differently on a loaded shared CI runner than on a quiet bare-metal box, and a hard
speedup threshold checked everywhere would flake (the #257 lesson — "honest where
realized, never faked"). This module makes the *gate* honest:

  * **Always strict — correctness + measurement validity.** Both realizations must be
    self-checked-correct (`ok`), the measurement must be valid (positive ns/call, a
    finite measurable ratio), and — when asked — the cost model must have selected the
    intended realization width. A regression here is a real bug and fails everywhere.
  * **Perf floors — bare-metal only.** A minimum speedup or a maximum ns/call is a
    *budget*, enforced only when `BCIR_BAREMETAL=1` (a controlled box). Off bare-metal
    the floor is *waived* (recorded, not asserted) so shared CI never flakes on it.
  * **Trend tracking.** Every evaluation can append a provenance-tagged line to a JSONL
    trend log (host, machine, timestamp, ns/call, speedup), so perf is watched as a
    moving series rather than a brittle per-run pass/fail.

Pure-Python and dependency-free; it operates on any `bcir.bench.Comparison`-shaped
object (duck-typed), so the budget logic is testable without the toolchain.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field

from ._artifact_json import strict_json_loads

_MAX_TREND_BYTES = 64 << 20
_MAX_TREND_LINE_BYTES = 1 << 20
_MAX_TREND_RECORDS = 1_000_000

def is_baremetal() -> bool:
    """True on a controlled box where perf floors are meaningful (set BCIR_BAREMETAL=1)."""
    return os.environ.get("BCIR_BAREMETAL", "").strip() not in ("", "0", "false", "no")


@dataclass(frozen=True)
class Budget:
    """A named benchmark budget. Correctness + measurement validity are *always* required;
    `min_speedup_milli` / `max_ns_per_call` / `match_band` are bare-metal-only (1000 == parity).

    For a *structural win* (gather avoidance, reduction blocking) use `min_speedup_milli` — a floor
    well below the realized speedup. For a *dense match* (BCIR should neither beat nor lose to Clang
    on a memory-bound kernel — the cost model correctly declines to "optimize") use `match_band`,
    a [lo, hi] parity window: it catches both a regression that makes BCIR much slower AND a
    suspicious "win" that usually means a broken/elided kernel. `reference_milli` is the documented
    Clang-comparison number (e.g. 6000 == the 6.0x gather result); it is *tracked in the trend log,
    never asserted* — the gate is deliberately not strict on the exact speedup in CI."""
    name: str
    min_speedup_milli: int | None = None      # e.g. 1500 == require >=1.5x (bare-metal only)
    max_ns_per_call: int | None = None        # absolute latency ceiling (bare-metal only)
    match_band: tuple | None = None           # (lo, hi) milli parity window (bare-metal only)
    expect_width: int | None = None           # the cost model must select this lane width
    reference_milli: int | None = None        # documented speedup (trend context; never asserted)


@dataclass(frozen=True)
class BudgetResult:
    name: str
    ok: bool
    baremetal: bool
    speedup_milli: int
    ns_per_call: int
    reasons: tuple[str, ...] = ()             # why it failed (empty when ok)
    enforced: tuple[str, ...] = ()            # the strict checks that actually ran
    waived: tuple[str, ...] = ()              # perf floors skipped because not bare-metal
    reference_milli: int | None = None        # documented Clang-comparison number (trend context)

    def __bool__(self) -> bool:
        return self.ok


def evaluate(comp, budget: Budget, *, baremetal: bool | None = None) -> BudgetResult:
    """Evaluate a Comparison-shaped object against `budget`.

    Strict (everywhere): both `.ok`, positive ns/call, a finite speedup, and the expected
    selected width. Lenient (bare-metal only): the speedup / latency floors.
    """
    bare = is_baremetal() if baremetal is None else baremetal
    reasons: list[str] = []
    enforced: list[str] = []
    waived: list[str] = []

    bcir, base = comp.bcir, comp.baseline
    speedup = int(comp.speedup_milli)

    # --- always strict: correctness ---
    enforced.append("correctness")
    if not bcir.ok:
        reasons.append(f"bcir realization incorrect: {bcir.detail or 'self-check failed'}")
    if not base.ok:
        reasons.append(f"baseline realization incorrect: {base.detail or 'self-check failed'}")

    # --- always strict: measurement validity ---
    enforced.append("measurement-validity")
    if bcir.ns_per_call <= 0 or base.ns_per_call <= 0:
        reasons.append(f"invalid measurement: non-positive ns/call "
                       f"(bcir={bcir.ns_per_call}, baseline={base.ns_per_call})")
    if speedup <= 0:
        reasons.append("speedup not finite/measurable (got 0)")

    if budget.expect_width is not None:
        enforced.append(f"selected-width=={budget.expect_width}")
        if bcir.width != budget.expect_width:
            reasons.append(f"cost model selected width {bcir.width}, expected "
                           f"{budget.expect_width}")

    # --- perf floors: bare-metal only ---
    if budget.min_speedup_milli is not None:
        if bare:
            enforced.append(f"min_speedup_milli>={budget.min_speedup_milli}")
            if speedup < budget.min_speedup_milli:
                reasons.append(f"speedup {speedup} below floor {budget.min_speedup_milli} "
                               f"(bare-metal)")
        else:
            waived.append(f"min_speedup_milli>={budget.min_speedup_milli}")
    if budget.max_ns_per_call is not None:
        if bare:
            enforced.append(f"max_ns_per_call<={budget.max_ns_per_call}")
            if bcir.ns_per_call > budget.max_ns_per_call:
                reasons.append(f"ns/call {bcir.ns_per_call} over ceiling "
                               f"{budget.max_ns_per_call} (bare-metal)")
        else:
            waived.append(f"max_ns_per_call<={budget.max_ns_per_call}")
    if budget.match_band is not None:
        lo, hi = budget.match_band
        if bare:
            enforced.append(f"match_band[{lo},{hi}]")
            if not (lo <= speedup <= hi):
                reasons.append(f"speedup {speedup} outside parity band [{lo},{hi}] (bare-metal) — a "
                               f"dense kernel should match Clang, not diverge")
        else:
            waived.append(f"match_band[{lo},{hi}]")

    return BudgetResult(name=budget.name, ok=not reasons, baremetal=bare, speedup_milli=speedup,
                        ns_per_call=bcir.ns_per_call, reasons=tuple(reasons),
                        enforced=tuple(enforced), waived=tuple(waived),
                        reference_milli=budget.reference_milli)


# --- trend tracking --------------------------------------------------------------

def _provenance() -> dict:
    return {"ts": int(time.time()), "host": platform.node(), "machine": platform.machine(),
            "baremetal": is_baremetal()}


def record_trend(result: BudgetResult, path: str) -> dict:
    """Append a provenance-tagged JSON line for `result` to the trend log at `path`."""
    row = {**_provenance(), "name": result.name, "ok": result.ok,
           "speedup_milli": result.speedup_milli, "ns_per_call": result.ns_per_call,
           "reference_milli": result.reference_milli}
    encoded = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
    try:
        current_size = os.path.getsize(path)
    except FileNotFoundError:
        current_size = 0
    if current_size + len(encoded) > _MAX_TREND_BYTES:
        raise ValueError(
            f"trend log would exceed {_MAX_TREND_BYTES} bytes; rotate {path!r}")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "ab") as f:
        f.write(encoded)
    return row


def read_trend(path: str) -> list[dict]:
    try:
        with open(path, "rb") as f:
            rows = []
            consumed = 0
            for line_number, raw in enumerate(f, 1):
                consumed += len(raw)
                if consumed > _MAX_TREND_BYTES:
                    raise ValueError(f"trend log exceeds {_MAX_TREND_BYTES} bytes")
                if len(raw) > _MAX_TREND_LINE_BYTES:
                    raise ValueError(f"trend log line {line_number} exceeds the line limit")
                if not raw.strip():
                    continue
                if len(rows) >= _MAX_TREND_RECORDS:
                    raise ValueError("trend log exceeds the record limit")
                try:
                    text = raw.decode("utf-8")
                except UnicodeError as exc:
                    raise ValueError(f"trend log line {line_number} is not UTF-8") from exc
                row = strict_json_loads(text, f"trend log line {line_number}",
                                        max_bytes=_MAX_TREND_LINE_BYTES)
                fields = {"ts", "host", "machine", "baremetal", "name", "ok",
                          "speedup_milli", "ns_per_call", "reference_milli"}
                if not isinstance(row, dict) or set(row) != fields:
                    raise ValueError(f"trend log line {line_number} has an invalid schema")
                for key in ("host", "machine", "name"):
                    value = row[key]
                    if (not isinstance(value, str) or (key == "name" and not value) or
                            len(value) > 4096 or
                            any(ord(ch) < 0x20 for ch in value)):
                        raise ValueError(
                            f"trend log line {line_number} {key} is not a bounded string")
                for key in ("baremetal", "ok"):
                    if not isinstance(row[key], bool):
                        raise ValueError(f"trend log line {line_number} {key} is not boolean")
                timestamp = row["ts"]
                if (isinstance(timestamp, bool) or not isinstance(timestamp, int) or
                        not 0 <= timestamp <= (1 << 63) - 1):
                    raise ValueError(
                        f"trend log line {line_number} ts is not a non-negative i63")
                for key in ("speedup_milli", "ns_per_call"):
                    value = row[key]
                    if (isinstance(value, bool) or not isinstance(value, int) or
                            not -(1 << 63) <= value <= (1 << 63) - 1):
                        raise ValueError(
                            f"trend log line {line_number} {key} is not a signed i64")
                reference = row["reference_milli"]
                if (reference is not None and
                        (isinstance(reference, bool) or not isinstance(reference, int) or
                         not 0 <= reference <= (1 << 63) - 1)):
                    raise ValueError(
                        f"trend log line {line_number} reference_milli is invalid")
                rows.append(row)
            return rows
    except OSError:
        return []


def trend_summary(path: str, name: str) -> dict:
    """Simple moving-series stats for one budget — count, last, min/median speedup, and (if the
    budget carries one) the documented reference so drift against it can be watched over time."""
    rows = [r for r in read_trend(path)
            if r.get("name") == name and isinstance(r.get("speedup_milli"), int)]
    if not rows:
        return {"name": name, "count": 0}
    series = [r["speedup_milli"] for r in rows]
    ordered = sorted(series)
    out = {"name": name, "count": len(series), "last": series[-1],
           "min": ordered[0], "median": ordered[len(ordered) // 2], "max": ordered[-1]}
    refs = [r["reference_milli"] for r in rows if isinstance(r.get("reference_milli"), int)]
    if refs:
        out["reference"] = refs[-1]
        out["last_vs_reference_pct"] = round(100 * series[-1] / refs[-1]) if refs[-1] else None
    return out
