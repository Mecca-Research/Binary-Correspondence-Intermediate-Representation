"""LM-PAIRED (2026-10-17): BCIR against the fastest matched PyTorch, seed by seed.

Reads one directory of train records (train_comparison.py's train stage, copied as
bcir-<seed>.json and torch-compiled-<seed>.json) and states, with the analysis fixed before
the runs:

  speed    per seed the ratio of the two median warm steps (torch / BCIR, so above 1 means
           BCIR is faster, and it is the ratio of the tokens a second); their geometric mean
           and its 95% Student-t interval over the seeds' log ratios; how many seeds BCIR won
  quality  per seed BCIR's final validation and test cross-entropy minus PyTorch's; their
           mean and its 95% t interval; the margin fixed beforehand: the interval of the test
           difference inside +-0.01 nats
  memory   peak RSS (VmHWM of the train process), and the persistent and transient bytes
           each record carries (BCIR: its planned arenas; PyTorch: its parameters, gradients
           and moments after training, no transient counter)

Added after the runs, as a check and labelled so: where a seed also has torch-eager-<seed>.json,
the same ratio against the faster of PyTorch's two arms on that seed.

  python3 paired.py DIR [--json OUT]
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# two-sided 95% Student-t critical values by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
MARGIN = 0.01  # nats: the test cross-entropy difference's interval must lie inside +-MARGIN


def interval(values):
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return mean, None, None
    half = T95[n - 1] * statistics.stdev(values) / math.sqrt(n)
    return mean, mean - half, mean + half


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    seeds = sorted(int(f.stem.split("-")[-1]) for f in a.dir.glob("bcir-*.json"))
    rows = []
    for seed in seeds:
        b = json.loads((a.dir / f"bcir-{seed}.json").read_text())
        t = json.loads((a.dir / f"torch-compiled-{seed}.json").read_text())
        assert b["seed"] == t["seed"] == seed and b["steps"] == t["steps"], seed
        eager = a.dir / f"torch-eager-{seed}.json"
        e = json.loads(eager.read_text()) if eager.exists() else None
        if e is not None:
            assert e["seed"] == seed and e["steps"] == b["steps"], seed
        fastest = min(t["median_step_ms"], e["median_step_ms"]) if e else t["median_step_ms"]
        rows.append({
            "seed": seed,
            "bcir_ms": b["median_step_ms"], "torch_ms": t["median_step_ms"],
            "bcir_tokens_per_second": b["tokens_per_second"],
            "torch_tokens_per_second": t["tokens_per_second"],
            "ratio": t["median_step_ms"] / b["median_step_ms"],
            "eager_ms": e["median_step_ms"] if e else None,
            "eager_test": e["final"]["test"]["cross_entropy"] if e else None,
            "fastest_ratio": fastest / b["median_step_ms"],
            "validation": [b["final"]["validation"]["cross_entropy"],
                           t["final"]["validation"]["cross_entropy"]],
            "test": [b["final"]["test"]["cross_entropy"], t["final"]["test"]["cross_entropy"]],
            "peak_rss_mb": [b["rss_mb"]["peak"], t["rss_mb"]["peak"]],
            "memory": [b.get("memory"), t.get("memory")],
            "torch_compile_s": t["compile_and_first_steps_seconds"],
            "commit": b["environment"]["commit"],
        })  # fmt: skip
    if not rows:
        sys.exit("no pairs")
    logs = [math.log(r["ratio"]) for r in rows]
    m, lo, hi = interval(logs)
    fm, flo, fhi = interval([math.log(r["fastest_ratio"]) for r in rows])
    dval = [r["validation"][0] - r["validation"][1] for r in rows]
    dtest = [r["test"][0] - r["test"][1] for r in rows]
    vm, vlo, vhi = interval(dval)
    tm, tlo, thi = interval(dtest)
    out = {
        "pairs": rows,
        "speed": {"geometric_mean_ratio": math.exp(m),
                  "ci95": None if lo is None else [math.exp(lo), math.exp(hi)],
                  "bcir_faster_seeds": sum(r["ratio"] > 1 for r in rows), "seeds": len(rows)},
        "check_fastest_arm": {"eager_seeds": [r["seed"] for r in rows if r["eager_ms"] is not None],
                              "geometric_mean_ratio": math.exp(fm),
                              "ci95": None if flo is None else [math.exp(flo), math.exp(fhi)]},
        "quality": {"validation_minus_torch": {"mean": vm, "ci95": None if vlo is None else [vlo, vhi]},
                    "test_minus_torch": {"mean": tm, "ci95": None if tlo is None else [tlo, thi]},
                    "margin_nats": MARGIN,
                    "within_margin": tlo is not None and -MARGIN < tlo and thi < MARGIN},
    }  # fmt: skip
    print(
        "| seed | BCIR ms | PyTorch ms | tokens/s BCIR / PyTorch | ratio | val CE B / T | "
        "test CE B / T | peak RSS MB B / T |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['seed']} | {r['bcir_ms']:.1f} | {r['torch_ms']:.1f} | "
            f"{r['bcir_tokens_per_second']:,.0f} / {r['torch_tokens_per_second']:,.0f} | "
            f"{r['ratio']:.3f} | {r['validation'][0]:.4f} / {r['validation'][1]:.4f} | "
            f"{r['test'][0]:.4f} / {r['test'][1]:.4f} | "
            f"{r['peak_rss_mb'][0]:.0f} / {r['peak_rss_mb'][1]:.0f} |"
        )
    s = out["speed"]
    print(
        f"speed: geometric mean ratio {s['geometric_mean_ratio']:.3f}, 95% CI "
        f"{s['ci95'] and [round(v, 3) for v in s['ci95']]}, BCIR faster on "
        f"{s['bcir_faster_seeds']} of {s['seeds']} seeds"
    )
    c = out["check_fastest_arm"]
    if c["eager_seeds"]:
        print(
            f"check (after the runs): against the faster PyTorch arm where eager also ran "
            f"({c['eager_seeds']}): geometric mean ratio {c['geometric_mean_ratio']:.3f}, 95% CI "
            f"{c['ci95'] and [round(v, 3) for v in c['ci95']]}"
        )
        for r in rows:
            if r["eager_ms"] is not None:
                print(f"  seed {r['seed']}: eager {r['eager_ms']:.1f} ms against compiled "
                      f"{r['torch_ms']:.1f} (test CE {r['eager_test']:.4f})")  # fmt: skip
    q = out["quality"]
    print(
        f"quality: test CE BCIR - PyTorch mean {tm:+.5f}, 95% CI "
        f"{q['test_minus_torch']['ci95'] and [round(v, 5) for v in q['test_minus_torch']['ci95']]}"
        f" (margin +-{MARGIN}: {'inside' if q['within_margin'] else 'NOT inside'}); "
        f"validation mean {vm:+.5f}"
    )
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
