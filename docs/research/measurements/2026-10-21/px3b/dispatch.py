"""PX-3b (2026-10-21): the kernel instance and the pool's thread split as choices of the step program
-- what the measurement finds, what the chosen dispatch does to the step's time, and that every
dispatch trains the same bits.

On the 9.84M comparison decoder (B=8, T=128) at the given threads and contract, for calls of K
steps:

  measure  `native_choose.measure` on a trainer of a library built with the job clock
           (`NativeDecoder.build(probe="time")`), over every dispatch kind
           (`native_choose.DISPATCH_CHOICE_KINDS`): each alternative this host admits -- every
           instance a GEMM's or an attention's family admits but its widest (the default), every
           worker count below the pool's threads -- the program with every job of the kind
           dispatched so, ROUNDS rotated rounds on the one trainer, each kind's jobs read in place
           by their dispatch labels; the table (each alternative's time over the default's, Q8,
           and its 95% interval, the shape naming the host's instance masks) and the raw readings
  choose   `native_choose.choose_dispatch` on that table: for each kind, the alternative resolved
           faster (its whole interval below 256) with the lowest mean; every other kind the default

then trainers of a library without the clock (the library a caller runs), from one seed, take
turns round by round, each running a K-step program stated once before the rounds:

  default   the default dispatch (the program before PX-3b, word for word)
  default2  the same on a second trainer: the A/A control
  chosen    the chosen dispatch

in AB_ROUNDS rotated rounds; then, in SLOW_ROUNDS rotated rounds of their own beside a trainer of
their own running the default (default.slow: a program on the portable instance runs for seconds
where the default runs for a tenth of one, and its effect needs few rounds to resolve):

  worst     each kind at the alternative the measurement prices slowest, where one is slower than
            the default (absent when none is)
  portable  every GEMM and attention job on the portable instance: the price of the vector units
            the default takes (when the host's families admit more than it)

Each ratio's geometric mean over its rounds and its 95% Student-t interval, beside the job-time
ratio the measurement's own readings predict for the chosen, the worst and the portable dispatch.
Last, every trainer's state: one state, bit for bit, among the trainers of each rotation (the two
ran different numbers of steps; every dispatch is the same bits by construction).

  python3 dispatch.py OUT.json [--k 4] [--rounds 24] [--ab-rounds 32] [--slow-rounds 6]
      [--threads 4] [--contract bf16+exp+reduce+attnbf16] [--cc gcc] [--smoke]
"""

import argparse
import ctypes as ct
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from array import array
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
# the Student-t 0.975 quantile; a df between rows takes the row below (the wider interval)
T95 = {3: 3.182, 4: 2.776, 5: 2.571, 7: 2.365, 9: 2.262, 11: 2.201, 15: 2.131, 19: 2.093,
       23: 2.069, 31: 2.040, 47: 2.012, 63: 1.998}  # fmt: skip


def interval(ratios):
    if len(ratios) < 4:
        raise SystemExit("an interval needs four rounds or more")
    logs = [math.log(v) for v in ratios]
    m = statistics.fmean(logs)
    t = T95[max(df for df in T95 if df <= len(logs) - 1)]
    half = t * statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(m), [math.exp(m - half), math.exp(m + half)]


def digest(nd) -> str:
    h = hashlib.sha256()
    for buf in ("weights", "moment1", "moment2"):
        h.update(ct.string_at(nd.buffers[buf], nd.state.plan.parameters * 4))
    h.update(repr((nd.state.step, nd.state.beta1_power, nd.state.beta2_power)).encode())
    return h.hexdigest()[:16]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--ab-rounds", type=int, default=32)
    p.add_argument("--slow-rounds", type=int, default=6)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--contract", default="bf16+exp+reduce+attnbf16")
    p.add_argument("--cc", default="gcc")
    p.add_argument(
        "--smoke", action="store_true", help="a two-layer toy decoder: the driver's own check"
    )
    a = p.parse_args()
    from bcir.frontends.models.decode import DecoderSpec
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder
    from bcir.hosted.models.native_choose import (
        DISPATCH_CHOICE_KINDS,
        choose_dispatch,
        dispatch_kind,
        measure,
        trainer_shape,
    )
    from bcir.hosted.models.native_launch import (
        INSTANCE_KINDS,
        RECORD_WORDS,
        dispatch_alternatives,
        program_words,
        step_program,
    )

    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4,
                       tied_embeddings=True)  # fmt: skip
    batch, context, k = 8, 128, a.k
    if a.smoke:
        spec = DecoderSpec(260, 64, 2, 2, 128, activation="silu_gate", n_kv_heads=1,
                           tied_embeddings=True)  # fmt: skip
        batch, context = 2, 32
    common = dict(batch_size=batch, context_length=context, threads=a.threads,
                  contract=a.contract)  # fmt: skip

    # the measurement, on a clocked library
    timed = NativeDecoder.build(a.out.parent / (a.out.stem + "-time"), cc=a.cc, probe="time")
    probe = NativeDecoder(timed, spec, **common)
    instances = probe.instance_families()
    raw: list = []
    t0 = time.perf_counter()
    table = measure(probe, steps=k, rounds=a.rounds, raw=raw, kinds=DISPATCH_CHOICE_KINDS)
    measure_s = time.perf_counter() - t0
    shape = trainer_shape(probe, steps=k, instances=True)
    alternatives = dispatch_alternatives(**probe.program_decisions(steps=k))
    probe.close()
    chosen = choose_dispatch(table, spec, shape)
    # the dispatch the measurement prices slowest: each kind at its slowest alternative, where that
    # is slower than the default (q8 above 256)
    slowest: dict = {}
    for e in table.entries:
        if e.q8 > 256 and (e.kind not in slowest or e.q8 > slowest[e.kind].q8):
            slowest[e.kind] = e
    worst = {dispatch_kind(kind): e.alternative for kind, e in slowest.items()}
    portable = {kind: "i1" for kind in sorted(INSTANCE_KINDS) if "i1" in alternatives[kind]}

    def predicted(dispatch):
        """The step's job time a dispatch should take over the default's, from the measurement's
        own readings: each dispatched kind's job time against the default's at that kind, summed."""
        ratios = []
        for r in raw:
            total = r["total_ns"]["default"]
            delta = sum(r["site_ns"][f"dispatch:{kind}={alt}"] - r["site_ns"][f"default@dispatch:{kind}"]
                        for kind, alt in dispatch.items())  # fmt: skip
            ratios.append((total + delta) / total)
        return statistics.fmean(ratios)

    print(f"measured in {measure_s:.1f} s:", json.dumps(
        [[e.kind, e.alternative, e.q8, e.low_q8, e.high_q8] for e in table.entries]), flush=True)  # fmt: skip
    print("chosen:", json.dumps(chosen) if chosen else "none", flush=True)
    print("slowest:", json.dumps(worst), flush=True)

    # the intervention, on the library a caller runs
    lib = NativeDecoder.build(a.out.parent / (a.out.stem + "-build"), cc=a.cc)
    n = batch * context
    tokens = [(7 * i + 3) % 260 for i in range(n * k)]
    targets = [(7 * i + 4) % 260 for i in range(n * k)]
    opt = NativeAdamW(lr=1e-4)
    d1 = NativeDecoder(lib, spec, **common)
    decisions = d1.program_decisions(steps=k)

    def program(dispatch):
        return step_program(spec, batch, context, **decisions, dispatch=dispatch)

    arm_dispatch = {"default": None, "default2": None, "chosen": chosen}
    if worst or portable:
        arm_dispatch["default.slow"] = None
    if worst:
        arm_dispatch["worst"] = worst
    if portable:
        arm_dispatch["portable"] = portable
    arms = {arm: (d1 if arm == "default" else NativeDecoder(lib, spec, **common),
                  array("Q", program_words(program(dispatch))))
            for arm, dispatch in arm_dispatch.items()}  # fmt: skip
    assert all(nd.program_decisions(steps=k) == decisions for nd, _ in arms.values())

    def once(arm):
        nd, words = arms[arm]
        t0 = time.perf_counter()
        nd.run_program(words, tokens, targets, steps=k, optimizer=opt)
        return 1000 * (time.perf_counter() - t0) / k

    for arm in arms:  # warm each trainer: the pages, the pool, the tile unit's admission
        once(arm)
    ms: dict = {arm: [] for arm in arms}

    def rotate(names, rounds):
        for r in range(rounds):
            for j in range(len(names)):
                arm = names[(r + j) % len(names)]
                ms[arm].append(once(arm))

    fast_arms = ("default", "default2", "chosen")
    slow_arms = tuple(arm for arm in arms if arm not in fast_arms)  # default.slow first
    rotate(fast_arms, a.ab_rounds)
    print("fast arms done", flush=True)
    if slow_arms:
        rotate(slow_arms, a.slow_rounds)
    names = tuple(arms)
    pairs = {"default2 / default (A/A: the default dispatch against itself)": ("default2", "default"),
             "chosen / default (the chosen dispatch against the default)": ("chosen", "default")}  # fmt: skip
    if "worst" in arms:
        pairs["worst / default (the dispatch priced slowest against the default)"] = (
            "worst", "default.slow")  # fmt: skip
    if "portable" in arms:
        pairs["portable / default (every GEMM and attention on the portable instance)"] = (
            "portable", "default.slow")  # fmt: skip
    out = {"cpu": next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                       if line.startswith("model name")),
           "vcpus": len([line for line in open("/proc/cpuinfo") if line.startswith("processor")]),
           "flags": sorted({f for line in open("/proc/cpuinfo") if line.startswith("flags")
                            for f in line.split(":", 1)[1].split()
                            if f in ("avx2", "fma", "avx512f", "avx512bw", "avx512_bf16",
                                     "amx_tile", "amx_bf16")}),
           "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
           "worktree_diff_sha256_16": hashlib.sha256(subprocess.run(
               ["git", "-C", str(REPO), "diff", "HEAD", "--", "runtime/c", "bcir"],
               capture_output=True).stdout).hexdigest()[:16],
           # the code measured, by content: the git tree ids of the two directories at HEAD (with
           # an empty diff above, any commit whose trees are these ran the same code)
           "trees": {path: subprocess.run(["git", "-C", str(REPO), "rev-parse", f"HEAD:{path}"],
                                          capture_output=True, text=True).stdout.strip()
                     for path in ("bcir", "runtime/c")},
           "k": k, "rounds": a.rounds, "ab_rounds": a.ab_rounds,
           "slow_rounds": a.slow_rounds if slow_arms else 0, "threads": a.threads,
           "contract": a.contract, "instances": instances,
           "compiler": subprocess.run([a.cc, "--version"], capture_output=True,
                                      text=True).stdout.splitlines()[0],
           "alternatives": {kind: list(alts) for kind, alts in alternatives.items() if alts},
           "table": json.loads(table.to_json()),
           "measure_s": round(measure_s, 1),
           "raw": raw,
           "chosen": chosen,
           "slowest": worst,
           "portable": portable,
           "predicted_job_ratio": {"chosen": predicted(chosen), "worst": predicted(worst),
                                   "portable": predicted(portable)},
           "program_records": {arm: len(w) // RECORD_WORDS for arm, (_, w) in arms.items()},
           "ms_per_step": {arm: [round(v, 3) for v in ms[arm]] for arm in names},
           "median_ms_per_step": {arm: statistics.median(ms[arm]) for arm in names},
           "ratios": {}}  # fmt: skip
    for name, (slow, fast) in pairs.items():
        mean, ci = interval([x / y for x, y in zip(ms[slow], ms[fast])])
        out["ratios"][name] = {"geometric_mean": mean, "ci95": ci,
                               "rounds_slower": sum(x > y for x, y in zip(ms[slow], ms[fast]))}  # fmt: skip
        print(f"{name}: x{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]", flush=True)
    states = {arm: digest(nd) for arm, (nd, _) in arms.items()}
    for nd, _ in arms.values():
        nd.close()
    out["state_sha256_16"] = states
    # one state among the trainers of each rotation: the same steps, any dispatch
    out["one_state"] = all(len({states[arm] for arm in group}) == 1
                           for group in (fast_arms, slow_arms) if group)  # fmt: skip
    print(json.dumps(out["median_ms_per_step"]), "one state:", out["one_state"])
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
