"""Q8-LANES' decode A/B: the standalone C decoder (bcir-llama: runtime/c/bcir_llama_cli.c and the units it links)
built from the tree before Q8-LANES (the first arm, BEFORE) and from the tree after it (the other arms: its widest
instance, and capped at AVX2 and at the portable loops), and BEFORE again (the A/A control); each process a greedy
generation from MODEL. A round runs every arm in rotated order, each arm twice -- 8 and 200 new tokens from one
5-token prompt -- its per-token time (t200 - t8) / 192 and its request (t200: the process's wall time, the model's
load included); each arm's ratio over BEFORE's in the same round, the geometric mean and its 95% interval (Student t
to 24 rounds, then 1.96 + 2.4/(n-1)), after a warm round. Every run's generated tokens and final logits are
BEFORE's bytes at its length (the run exits otherwise).
usage: decode_ab.py ROUNDS MODEL OUT.json NAME=BINARY...  (the first NAME is BEFORE)"""

import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

T95 = [0, 0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
       2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069]  # fmt: skip
PROMPT = "1,100,200,300,400"


def interval(ratios):
    logs = [math.log(r) for r in ratios]
    n, m = len(logs), statistics.fmean(logs)
    h = (T95[n] if n <= 24 else 1.96 + 2.4 / (n - 1)) * statistics.stdev(logs) / math.sqrt(n)
    return [round(math.exp(m), 4), round(math.exp(m - h), 4), round(math.exp(m + h), 4)]


def run(binary, model, new, logits):
    t0 = time.perf_counter()
    done = subprocess.run(
        [binary, "--model", model, "--prompt-ids", PROMPT, "--max-new", str(new), "--logits-out", logits],
        check=True, capture_output=True,
    )  # fmt: skip
    seconds = time.perf_counter() - t0
    return seconds, hashlib.sha256(Path(logits).read_bytes() + done.stdout).hexdigest()


def main():
    rounds, model, out = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    arms = dict(arg.split("=", 1) for arg in sys.argv[4:])
    before = next(iter(arms))
    arms["A/A"] = arms[before]
    names = list(arms)
    per = {a: [] for a in names}
    req = {a: [] for a in names}
    digest = {}
    with tempfile.TemporaryDirectory() as tmp:
        logits = str(Path(tmp) / "logits.bin")
        for r in range(-1, rounds):
            for i in range(len(names)):
                arm = names[(i + max(r, 0)) % len(names)]
                t8, d8 = run(arms[arm], model, 8, logits)
                t200, d200 = run(arms[arm], model, 200, logits)
                for new, d in ((8, d8), (200, d200)):
                    if digest.setdefault(new, d) != d:
                        sys.exit(
                            f"{arm}'s tokens or logits at {new} new tokens differ from {before}'s"
                        )
                if r >= 0:
                    per[arm].append((t200 - t8) / 192 * 1e3)
                    req[arm].append(t200 * 1e3)
    rows = {}
    for arm in names:
        rows[arm] = {
            "per_token_ms_median": round(statistics.median(per[arm]), 4),
            "request_200_ms_median": round(statistics.median(req[arm]), 3),
        }
        if arm != before:
            rows[arm]["per_token_ratio"] = interval([a / b for a, b in zip(per[arm], per[before])])
            rows[arm]["request_ratio"] = interval([a / b for a, b in zip(req[arm], req[before])])
    record = {
        "rounds": rounds,
        "prompt": [int(t) for t in PROMPT.split(",")],
        "new_tokens": [8, 200],
        "tokens_and_logits_sha256": {str(k): v for k, v in digest.items()},
        "every_run_the_same_bytes": True,
        "arms": rows,
        "raw_per_token_ms": {a: [round(v, 4) for v in per[a]] for a in names},
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(rows, indent=1))


main()
