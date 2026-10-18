"""BF16-KERN + BF16-PANELS-1 (2026-10-18): the in-place A/B's summary. Three trees ran the step at four
threads, statement by statement (docs/research/measurements/2026-10-13/scan/attribution.py), their order
rotated round by round: base (d5bc457's runtime/c), kern (the base with this commit's AMX kernel,
kern.patch) and panels (this commit). Per arm, the medians over the rounds of the step's median wall
time, the timed total and the GEMM + SwiGLU block (gemm.forward, gemm.backward, silu,
swiglu.backward); per pair of arms, the round-by-round ratios' median and how many rounds the second
arm was faster in. Writes JSON to stdout.

    python3 ab_summary.py AB_DIR ROUNDS
"""

import json
import statistics
import sys
from pathlib import Path

ARMS = ("base", "kern", "panels")
BLOCK = ("gemm.forward", "gemm.backward", "silu", "swiglu.backward")
PAIRS = (("base", "kern"), ("kern", "panels"), ("base", "panels"))


def readings(record):
    rows = record["rows_ms"]
    out = {"step_ms": record["step_wall_ms_median"], "timed_ms": record["timed_ms"]}
    out.update({name: rows.get(name, 0.0) for name in BLOCK})
    out["block_ms"] = sum(out[name] for name in BLOCK)
    return out


def main():
    root, rounds = Path(sys.argv[1]), int(sys.argv[2])
    runs = {arm: [] for arm in ARMS}
    for r in range(1, rounds + 1):
        for arm in ARMS:
            record = json.loads((root / f"attribution-t4-{arm}-{r}.json").read_text())
            if record["contract"] != "bf16+exp+reduce+attn" or record["threads"] != 4:
                raise SystemExit(f"{arm} round {r}: not the A/B's run")
            runs[arm].append(readings(record))
    keys = list(runs["base"][0])
    doc = {
        "rounds": rounds,
        "contract": "bf16+exp+reduce+attn",
        "threads": 4,
        "medians_ms": {
            arm: {key: round(statistics.median(x[key] for x in runs[arm]), 3) for key in keys}
            for arm in ARMS
        },
        "pairs": {},
    }
    for a, b in PAIRS:
        pair = {}
        for key in ("step_ms", "timed_ms", "block_ms", "gemm.forward", "gemm.backward"):
            ratios = [x[key] / y[key] for x, y in zip(runs[a], runs[b])]
            pair[key] = {
                "median_ratio": round(statistics.median(ratios), 4),
                "faster_rounds": sum(q > 1.0 for q in ratios),
            }
        doc["pairs"][f"{a}/{b}"] = pair
    json.dump(doc, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
