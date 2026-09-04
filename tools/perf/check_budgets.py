#!/usr/bin/env python3
"""The non-flaky benchmark gate. Times a few canonical BCIR realizations and checks each
against a `bcir.perf_budget.Budget`.

Honest by construction:
  * STRICT everywhere — both realizations self-check-correct and the measurement is valid
    (positive ns/call, a finite speedup). A regression here is a real bug; the gate fails.
  * Perf floors (min speedup) are BARE-METAL ONLY (`BCIR_BAREMETAL=1`); off a controlled box
    they are recorded, not asserted, so shared CI never flakes on timing. The floors are set
    well below the realized win (the #257-tuned robust floors) and only on programs whose win
    is cost-model-structural (gather avoidance, reduction blocking), never on the memory-bound
    elementwise path (which is legitimately measured-neutral vs the scalar baseline).
  * Degrades cleanly: with no C compiler present it skips (exit 0) — run it under the
    `silicon-degrade` or `thorough` test tier, or on a host with clang/cc/gcc.

    python tools/perf/check_budgets.py                  # correctness + validity (+ trend)
    BCIR_BAREMETAL=1 python tools/perf/check_budgets.py # also enforce the perf floors
    python tools/perf/check_budgets.py --trend build/perf/trend.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from bcir.perf_budget import Budget, evaluate, is_baremetal, record_trend, trend_summary


class _Norm:
    """Adapt a specialized comparison (.direct/.gather, .blocked/.gather, ...) to the uniform
    .bcir/.baseline/.speedup_milli shape `evaluate` expects (fast realization vs the one it beats)."""

    def __init__(self, fast, slow, speedup_milli: int):
        self.bcir, self.baseline, self.speedup_milli = fast, slow, int(speedup_milli)


def _cases():
    """(Budget, thunk -> _Norm). The five named Clang-comparison budgets from the comparison doc.

    `reference_milli` records the *documented* result (tracked in the trend log, never asserted —
    CI is deliberately not strict on the exact speedup). Dense kernels carry a `match_band` (BCIR
    should match Clang, not diverge); structural wins carry a conservative `min_speedup_milli` floor
    set well below the realized number and robust cross-arch. All floors/bands are bare-metal-only.
    Imported late so the lowering path only loads when timing."""
    from bcir.bench import compare, compare_gather, compare_reduce, compare_strided

    def dense_stream():  # doc: dense streaming 0.98x (match)
        c = compare("vector_add", opt="-O2", n=1 << 20, reps=30)
        return _Norm(c.bcir, c.baseline, c.speedup_milli)

    def dense_l1():  # doc: dense L1-resident 1.00x (match)
        c = compare("vector_add", opt="-O2", n=1 << 12, reps=200)  # small n stays L1-resident
        return _Norm(c.bcir, c.baseline, c.speedup_milli)

    def gather():  # doc: gather avoidance 6.0x
        c = compare_gather("vector_add", opt="-O2", n=1 << 18, reps=30, shuffle=True)
        return _Norm(c.direct, c.gather, c.speedup_milli)

    def reduce():  # doc: reduction order 14.1x
        c = compare_reduce("gather_reduce", opt="-O2", n=1 << 18, reps=20)
        return _Norm(c.blocked, c.gather, c.speedup_milli)

    def strided():  # doc: strided access 1.33x
        c = compare_strided("saxpy_strided", opt="-O2", n=1 << 20, reps=20)
        return _Norm(c.blocked, c.gather, c.speedup_milli)  # direct strided vs the gather form

    return [
        # dense: BCIR must MATCH Clang (the cost model correctly declines to "optimize") — a wide
        # parity band catches a real slowdown OR a suspicious win (usually an elided kernel).
        (Budget("dense streaming", match_band=(700, 1300), reference_milli=980), dense_stream),
        (Budget("dense L1", match_band=(700, 1300), reference_milli=1000), dense_l1),
        # structural wins: a robust cross-arch floor far below the documented number.
        (Budget("gather avoidance", min_speedup_milli=1500, reference_milli=6000), gather),
        (Budget("reduction order", min_speedup_milli=2000, reference_milli=14100), reduce),
        (Budget("strided access", min_speedup_milli=1100, reference_milli=1330), strided),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--trend",
        default=os.path.join(ROOT, "build", "perf", "trend.jsonl"),
        help="append a provenance-tagged measurement per budget to this JSONL log",
    )
    args = ap.parse_args()

    from bcir.bench import bench_available

    if not bench_available():
        sys.stderr.write(
            "[check_budgets] no C compiler (clang/cc/gcc) — skipping the measured "
            "gate. Run under the silicon-degrade/thorough tier or on a build host.\n"
        )
        return 0

    bare = is_baremetal()
    sys.stderr.write(
        f"[check_budgets] baremetal={bare} "
        f"({'enforcing' if bare else 'waiving'} perf floors; correctness + "
        f"measurement validity are strict either way)\n"
    )
    failed = 0
    for budget, thunk in _cases():
        res = evaluate(thunk(), budget, baremetal=bare)
        record_trend(res, args.trend)
        mark = "ok " if res.ok else "FAIL"
        ref = f" (doc {res.reference_milli / 1000:.2f}x)" if res.reference_milli else ""
        sys.stderr.write(
            f"  [{mark}] {res.name:<18} {res.speedup_milli / 1000:.2f}x{ref} "
            f"ns/call={res.ns_per_call}"
            f"{' waived=' + str(list(res.waived)) if res.waived else ''}\n"
        )
        for why in res.reasons:
            sys.stderr.write(f"        - {why}\n")
        failed += not res.ok

    sys.stderr.write(
        f"[check_budgets] continuous trend log: {args.trend} "
        f"(gather samples: {trend_summary(args.trend, 'gather avoidance').get('count', 0)})\n"
    )
    if failed:
        sys.stderr.write(
            f"[check_budgets] {failed} budget(s) failed a STRICT (correctness / "
            f"measurement-validity) check.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
