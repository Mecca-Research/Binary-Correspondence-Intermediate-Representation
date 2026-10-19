"""BF16-ATTN (2026-10-19): the bf16 GEMM's library-against-library ratio over several processes.

One gemm process reads its ratio through one placement of the two libraries and their buffers, and
the A/A controls (controls-gemm.txt) show that placement alone moving a ratio by up to about 3%, its
sign changing from process to process. This runs PROCS gemm processes, the two libraries' slots
swapped every other process, reads each process's median ratio as BASE over THIS (above 1: THIS is
faster), and prints one JSON line: the per-process ratios, their mean and a 95% Student-t interval
over the processes. Run with BASE and a copy of BASE as THIS, it is the A/A the pair is read
against. Wall time: indicative, never a gate.

    python3 procs.py GEMM BASE.so THIS.so THREADS ROUNDS PROCS
"""

import json
import statistics
import subprocess
import sys

# two-sided 95% Student-t quantiles by degrees of freedom (1-30); 1.96 beyond
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042]  # fmt: skip

gemm, base, this = sys.argv[1], sys.argv[2], sys.argv[3]
threads, rounds, procs = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
if procs < 2:
    raise SystemExit("PROCS >= 2")
ratios, same = [], True
for i in range(procs):
    swapped = i % 2 == 1
    first, second = (this, base) if swapped else (base, this)
    done = subprocess.run([gemm, first, second, str(threads), str(rounds)], capture_output=True,
                          text=True, check=True)  # fmt: skip
    line = json.loads(done.stdout)
    same = same and line["same_bits"]
    r = line["base_over_this"]["median"]  # the first slot's time over the second's
    ratios.append(round(1.0 / r if swapped else r, 4))
mean = statistics.fmean(ratios)
quantile = T975[procs - 2] if procs - 1 <= len(T975) else 1.96  # procs - 1 degrees of freedom
half = quantile * statistics.stdev(ratios) / procs**0.5
print(json.dumps({"threads": threads, "rounds": rounds, "processes": procs, "same_bits": same,
                  "base_over_this_per_process": ratios, "mean": round(mean, 4),
                  "ci95": [round(mean - half, 4), round(mean + half, 4)]}))  # fmt: skip
