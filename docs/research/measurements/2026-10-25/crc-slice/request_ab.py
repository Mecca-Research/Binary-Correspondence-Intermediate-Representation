"""CRC-SLICE's request A/B: the standalone C decoder (bcir-llama) built from the tree before CRC-SLICE (BEFORE, the
first arm) and from the tree after it (AFTER), and BEFORE again (the A/A control); each process one greedy request
from MODEL -- the model's load, a 5-token prompt and NEW new tokens -- timed whole. A round runs every arm in rotated
order at every length; each arm's ratio over BEFORE's in the same round, the geometric mean and its 95% interval
(Student t to 24 rounds, then 1.96 + 2.4/(n-1)), after a warm round. Every run's tokens and final logits are
BEFORE's bytes at its length (the run exits otherwise).
usage: request_ab.py ROUNDS MODEL OUT.json NEW,NEW,... NAME=BINARY...  (the first NAME is BEFORE)"""

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
    lengths = [int(n) for n in sys.argv[4].split(",")]
    arms = dict(arg.split("=", 1) for arg in sys.argv[5:])
    before = next(iter(arms))
    arms["A/A"] = arms[before]
    names = list(arms)
    ms = {(a, n): [] for a in names for n in lengths}
    digest = {}
    with tempfile.TemporaryDirectory() as tmp:
        logits = str(Path(tmp) / "logits.bin")
        for r in range(-1, rounds):
            for i in range(len(names)):
                arm = names[(i + max(r, 0)) % len(names)]
                for n in lengths:
                    seconds, d = run(arms[arm], model, n, logits)
                    if digest.setdefault(n, d) != d:
                        sys.exit(
                            f"{arm}'s tokens or logits at {n} new tokens differ from {before}'s"
                        )
                    if r >= 0:
                        ms[(arm, n)].append(seconds * 1e3)
    rows = {}
    for n in lengths:
        row = {"before_ms_median": round(statistics.median(ms[(before, n)]), 3)}
        for arm in names[1:]:
            row[arm] = {
                "ms_median": round(statistics.median(ms[(arm, n)]), 3),
                "ratio": interval([a / b for a, b in zip(ms[(arm, n)], ms[(before, n)])]),
            }
        rows[str(n)] = row
    record = {
        "rounds": rounds,
        "prompt": [int(t) for t in PROMPT.split(",")],
        "new_tokens": lengths,
        "tokens_and_logits_sha256": {str(k): v for k, v in digest.items()},
        "every_run_the_same_bytes": True,
        "requests": rows,
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(rows, indent=1))


main()
