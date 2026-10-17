"""NDT-PACK's A/B over the step's GEMM shapes alone (2026-10-17): each arm's tensor unit built
into ../../2026-10-16/beta/gemm_entries.c (the step's twelve shapes through
bcir_tensor_mm_kblock_pool at the betas the trainer passes), the builds run in turn with the
order rotated each round; per shape the median of the rounds' medians, the call-weighted step
total, every C digest against the first arm's, and in how many rounds each arm's total was
below the first's.

  python3 gemm_ab.py ROUNDS REPS THREADS NAME=DIR[:DEFINES] ...
"""

import json
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1].parent / "2026-10-16" / "beta" / "gemm_entries.c"


def main(argv):
    rounds, reps, threads = int(argv[1]), argv[2], argv[3]
    out = Path(tempfile.mkdtemp(prefix="gemm-ab-"))
    arms = []
    for spec in argv[4:]:
        name, rest = spec.split("=", 1)
        d, _, defs = rest.partition(":")
        exe = out / f"entries-{name}"
        cmd = ["gcc", "-O3", "-ffp-contract=off", "-fno-math-errno", "-pthread", *defs.split(),
               "-I", d, str(SRC), f"{d}/bcir_tensor.c", "-o", str(exe), "-lm"]  # fmt: skip
        subprocess.run([c for c in cmd if c], check=True)
        arms.append((name, exe))
    ms = defaultdict(lambda: defaultdict(list))
    dig, calls = defaultdict(dict), {}
    for r in range(rounds):
        for name, exe in arms[r % len(arms) :] + arms[: r % len(arms)]:
            p = subprocess.run(
                [str(exe), reps, threads], check=True, capture_output=True, text=True
            )
            for line in p.stdout.splitlines():
                rec = json.loads(line)
                key = (rec["ta"], rec["tb"], rec["m"], rec["n"], rec["k"])
                ms[name][key].append(rec["median_ms"])
                dig[name][key] = rec["c_fnv1a"]
                calls[key] = rec["calls"]
    names = [n for n, _ in arms]
    print("%-22s" % "shape (ta tb m n k)" + "".join("%12s" % n for n in names))
    tot = {n: 0.0 for n in names}
    for key in calls:
        row = "%-22s" % " ".join(map(str, key))
        for n in names:
            m = statistics.median(ms[n][key])
            tot[n] += m * calls[key]
            row += "%12.3f" % m
        print(row)
    print("%-22s" % "step total, ms" + "".join("%12.2f" % tot[n] for n in names))
    base = names[0]
    for n in names[1:]:
        same = all(dig[n][k] == dig[base][k] for k in calls)
        wins = sum(
            1 for r in range(rounds)
            if sum(ms[n][k][r] * calls[k] for k in calls) < sum(ms[base][k][r] * calls[k] for k in calls)
        )  # fmt: skip
        print(f"{n}: digests {'IDENTICAL' if same else 'DIFFERENT'} to {base}; "
              f"step total lower in {wins} of {rounds} rounds")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
