"""LM-PANELS (2026-10-18): BCIR with BF16-KERN and BF16-PANELS-1 against PyTorch's CPU mixed precision.

Reads one directory of train records (train_comparison.py's train stage, copied as <arm>-<seed>.json
for the arms bcir-panels, torch-compiled-bf16 and bcir-base; run.sh) and states, with the analysis
fixed before the runs:

  speed    per seed and pair of arms the ratio of the two median warm steps (the second arm's
           over the first's, so above 1 means the first arm is faster, and it is the ratio of
           the tokens a second); their geometric mean and its 95% Student-t interval over the
           seeds' log ratios; how many seeds the first arm won. The pairs, the claim first:
             bcir-panels / torch-compiled-bf16   BCIR with this commit against PyTorch's bf16
             bcir-panels / bcir-base             what BF16-KERN and BF16-PANELS-1 buy
             bcir-base / torch-compiled-bf16     LM-ATTN's comparison, again in this session
  quality  per seed the same pairs' final validation and test cross-entropy differences (the
           first arm's minus the second's); their mean and its 95% t interval; the margin fixed
           beforehand: the test difference's interval inside +-0.01 nats, for every pair
  memory   peak RSS (VmHWM of the train process) and the persistent, transient and workspace
           bytes each record carries
  states   whether bcir-panels and bcir-base end in one state seed for seed (the commit keeps
           every state), and with --previous (LM-ATTN's summary.json) whether both end in
           LM-ATTN's bcir-bf16-attn states

  python3 paired_panels.py DIR [--previous SUMMARY] [--json OUT]
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# two-sided 95% Student-t critical values by degrees of freedom
T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
}
MARGIN = 0.01  # nats: a pair's test cross-entropy difference must lie inside +-MARGIN
ARMS = ("bcir-panels", "torch-compiled-bf16", "bcir-base")
PAIRS = (
    ("bcir-panels", "torch-compiled-bf16"),
    ("bcir-panels", "bcir-base"),
    ("bcir-base", "torch-compiled-bf16"),
)


def interval(values):
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return mean, None, None
    half = T95[n - 1] * statistics.stdev(values) / math.sqrt(n)
    return mean, mean - half, mean + half


def _load(directory: Path, arm: str, seed: int):
    path = directory / f"{arm}-{seed}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    assert record["seed"] == seed, path
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir", type=Path)
    p.add_argument("--previous", type=Path, help="LM-ATTN's summary.json")
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    seeds = sorted(int(f.stem.split("-")[-1]) for f in a.dir.glob("bcir-panels-*.json"))
    records = {s: {arm: _load(a.dir, arm, s) for arm in ARMS} for s in seeds}
    seeds = [s for s in seeds if all(records[s][arm] for arm in ARMS)]
    if not seeds:
        sys.exit("no seed has every arm")
    steps = {records[s][arm]["steps"] for s in seeds for arm in ARMS}
    assert len(steps) == 1, steps
    out = {"seeds": seeds, "arms": {}, "pairs": [], "margin_nats": MARGIN}
    print("| seed | " + " | ".join(f"{arm} ms (tokens/s)" for arm in ARMS) + " |")
    print("|---" * (len(ARMS) + 1) + "|")
    for s in seeds:
        r = records[s]
        cells = [
            f"{r[arm]['median_step_ms']:.1f} ({r[arm]['tokens_per_second']:,.0f})" for arm in ARMS
        ]
        print(f"| {s} | " + " | ".join(cells) + " |")
    for arm in ARMS:
        rs = [records[s][arm] for s in seeds]
        out["arms"][arm] = {
            "median_step_ms": [r["median_step_ms"] for r in rs],
            "tokens_per_second": [r["tokens_per_second"] for r in rs],
            "validation": [r["final"]["validation"]["cross_entropy"] for r in rs],
            "test": [r["final"]["test"]["cross_entropy"] for r in rs],
            "peak_rss_mb": [r["rss_mb"]["peak"] for r in rs],
            "memory": [r.get("memory") for r in rs],
            "state_sha256": [r["state_sha256"] for r in rs],
            "compile_and_first_steps_seconds": [r.get("compile_and_first_steps_seconds") for r in rs],
            "commit": sorted({r["environment"]["commit"] for r in rs}),
        }  # fmt: skip
        mean_ce = statistics.fmean(out["arms"][arm]["test"])
        print(
            f"{arm}: test CE {', '.join(f'{v:.4f}' for v in out['arms'][arm]['test'])} "
            f"(mean {mean_ce:.4f}); peak RSS "
            f"{min(out['arms'][arm]['peak_rss_mb']):.0f}-{max(out['arms'][arm]['peak_rss_mb']):.0f} MB"
        )
    for first, second in PAIRS:
        ratios = [
            records[s][second]["median_step_ms"] / records[s][first]["median_step_ms"]
            for s in seeds
        ]
        m, lo, hi = interval([math.log(v) for v in ratios])
        dval = [records[s][first]["final"]["validation"]["cross_entropy"]
                - records[s][second]["final"]["validation"]["cross_entropy"] for s in seeds]  # fmt: skip
        dtest = [records[s][first]["final"]["test"]["cross_entropy"]
                 - records[s][second]["final"]["test"]["cross_entropy"] for s in seeds]  # fmt: skip
        vm, vlo, vhi = interval(dval)
        tm, tlo, thi = interval(dtest)
        row = {
            "first": first, "second": second, "ratios": ratios,
            "geometric_mean_ratio": math.exp(m),
            "ci95": None if lo is None else [math.exp(lo), math.exp(hi)],
            "first_faster_seeds": sum(v > 1 for v in ratios),
            "validation_difference": {"mean": vm, "ci95": None if vlo is None else [vlo, vhi]},
            "test_difference": {"mean": tm, "ci95": None if tlo is None else [tlo, thi]},
            "within_margin": tlo is not None and -MARGIN < tlo and thi < MARGIN,
        }  # fmt: skip
        out["pairs"].append(row)
        ci = row["ci95"] and [round(v, 3) for v in row["ci95"]]
        tci = row["test_difference"]["ci95"] and [
            round(v, 5) for v in row["test_difference"]["ci95"]
        ]
        verdict = " inside" if row["within_margin"] else " NOT inside"
        print(
            f"{first} / {second}: ratio {row['geometric_mean_ratio']:.3f}, 95% CI {ci}, first faster "
            f"on {row['first_faster_seeds']} of {len(seeds)}; test CE difference {tm:+.5f}, 95% CI "
            f"{tci}{verdict} the +-{MARGIN} margin; validation {vm:+.5f}"
        )
    same = {str(s): records[s]["bcir-panels"]["state_sha256"] == records[s]["bcir-base"]["state_sha256"]
            for s in seeds}  # fmt: skip
    print(
        f"states: bcir-panels ends in bcir-base's state on {sum(same.values())} of {len(same)} seeds"
    )
    out["states_panels_as_base"] = same
    if a.previous:
        before = json.loads(a.previous.read_text())
        old = dict(
            zip(
                before["seeds"],
                before["arms"]["bcir-bf16-attn"]["state_sha256"],
                strict=True,
            )
        )
        prev = {}
        for arm in ("bcir-panels", "bcir-base"):
            prev[arm] = {
                str(s): records[s][arm]["state_sha256"] == old.get(s) for s in seeds if s in old
            }
            print(
                f"states: {arm} ends in LM-ATTN's bcir-bf16-attn state on "
                f"{sum(prev[arm].values())} of {len(prev[arm])} seeds"
            )
        out["states_as_lm_attn"] = prev
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
