"""FACTS-RUN (2026-10-18): one native run call of K steps against K calls of one step, with the
commit's facts and without them.

Four arms, rotated round by round in one process on the 9.84M comparison decoder (B=8, T=128) at the
given threads and contract, each arm K steps a round on its own trainer from the same seeded
weights and the same token stream:

  new-run    this tree's trainer, one run call of K steps: the steps after the first trust the
             commit (FACTS-RUN) and skip the opening weight scan
  new-calls  this tree's trainer, K run calls of one step: every step scans
  base-run   the trainer before FACTS-RUN (BASE sources), one run call of K steps: every step scans
  base-calls the same, K calls of one step

so base-run / new-run is FACTS-RUN's own share (the same calls, the scans removed), new-calls /
new-run the whole difference a K-step call makes, and base-calls / base-run the per-call cost alone
(the Python binding's call and its token conversion). Each ratio's geometric mean over the rounds
and its 95% Student-t interval; the four trainers' states compared at the end (each pair of arms on
one library ends in one state: the same steps, called differently).

  python3 calls.py BASE_SOURCES OUT.json [--k 10] [--rounds 16] [--threads 4]
      [--contract bf16+exp+reduce+attn] [--cc gcc]
"""

import argparse
import ctypes
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
T95 = {9: 2.262, 11: 2.201, 15: 2.131, 19: 2.093, 23: 2.069, 31: 2.042, 47: 2.012, 63: 1.998}
FLAGS = ("-O3", "-ffp-contract=off", "-fno-math-errno")


def build(src: Path, out: Path, cc: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    target = out / "bcir_decoder_train.so"
    subprocess.run([cc, "-std=c11", *FLAGS, "-DBCIR_DT_BUILD_SHARED", "-shared", "-fPIC", "-I",
                    str(src), str(src / "bcir_tensor.c"), str(src / "bcir_decoder_train.c"), "-o",
                    str(target), "-lm", "-pthread"], check=True, timeout=600)  # fmt: skip
    return target


def interval(ratios):
    logs = [math.log(v) for v in ratios]
    m = statistics.fmean(logs)
    half = T95.get(len(logs) - 1, 2.0) * statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(m), [math.exp(m - half), math.exp(m + half)]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--rounds", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--contract", default="bf16+exp+reduce+attn")
    p.add_argument("--cc", default="gcc")
    a = p.parse_args()
    from bcir.frontends.models.decode import DecoderSpec
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder

    work = a.out.parent / (a.out.stem + "-build")
    libs = {"new": build(REPO / "runtime" / "c", work / "new", a.cc),
            "base": build(a.base, work / "base", a.cc)}  # fmt: skip
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4,
                       tied_embeddings=True)  # fmt: skip
    batch, context, k = 8, 128, a.k
    n = batch * context
    tokens = [(7 * i + 3) % 260 for i in range(n * k)]
    targets = [(7 * i + 4) % 260 for i in range(n * k)]
    opt = NativeAdamW(lr=1e-4)
    arms = ("new-run", "new-calls", "base-run", "base-calls")
    trainers = {arm: NativeDecoder(libs[arm.split("-")[0]], spec, batch_size=batch,
                                   context_length=context, threads=a.threads,
                                   contract=a.contract) for arm in arms}  # fmt: skip

    def once(arm):
        nd = trainers[arm]
        t0 = time.perf_counter()
        if arm.endswith("-run"):
            nd.run(tokens, targets, steps=k, optimizer=opt)
        else:
            for s in range(k):
                nd.run(tokens[s * n : (s + 1) * n], targets[s * n : (s + 1) * n], steps=1,
                       optimizer=opt)  # fmt: skip
        return 1000 * (time.perf_counter() - t0) / k

    for arm in arms:  # warm each trainer: its pages, its pool, the tile unit's admission
        once(arm)
    ms = {arm: [] for arm in arms}
    for r in range(a.rounds):
        for j in range(len(arms)):
            arm = arms[(r + j) % len(arms)]
            ms[arm].append(once(arm))
    pairs = {"base-run / new-run (FACTS-RUN's share)": ("base-run", "new-run"),
             "new-calls / new-run (a K-step call, all of it)": ("new-calls", "new-run"),
             "base-calls / base-run (the calls alone)": ("base-calls", "base-run")}  # fmt: skip
    out = {"cpu": next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                       if line.startswith("model name")),
           "vcpus": len([line for line in open("/proc/cpuinfo") if line.startswith("processor")]),
           "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
           "k": k, "rounds": a.rounds, "threads": a.threads, "contract": a.contract,
           "compiler": subprocess.run([a.cc, "--version"], capture_output=True,
                                      text=True).stdout.splitlines()[0],
           "ms_per_step": {arm: [round(v, 3) for v in ms[arm]] for arm in arms},
           "median_ms_per_step": {arm: statistics.median(ms[arm]) for arm in arms},
           "ratios": {}}  # fmt: skip
    for name, (slow, fast) in pairs.items():
        mean, ci = interval([x / y for x, y in zip(ms[slow], ms[fast])])
        out["ratios"][name] = {"geometric_mean": mean, "ci95": ci,
                               "rounds_faster": sum(x > y for x, y in zip(ms[slow], ms[fast]))}  # fmt: skip
        print(f"{name}: x{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    digest = {}
    for arm in arms:
        nd = trainers[arm]
        h = hashlib.sha256()
        for buf in ("weights", "moment1", "moment2"):
            h.update(ctypes.string_at(nd.buffers[buf], nd.state.plan.parameters * 4))
        digest[arm] = h.hexdigest()[:16]
        nd.close()
    out["state_sha256_16"] = digest
    out["same_state_per_library"] = {
        "new": digest["new-run"] == digest["new-calls"],
        "base": digest["base-run"] == digest["base-calls"],
        "new_as_base": digest["new-run"] == digest["base-run"],
    }
    print(json.dumps(out["median_ms_per_step"]), json.dumps(out["same_state_per_library"]))
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
