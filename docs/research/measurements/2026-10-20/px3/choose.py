"""PX-3 (2026-10-20): the step program's per-site choices taken from measured candidates -- what the
measurement finds, what the chosen program does to the step's time, and that it trains the same
bits.

On the 9.84M comparison decoder (B=8, T=128) at the given threads and contract, for calls of K
steps:

  measure  `native_choose.measure` on a trainer of a library built with the job clock
           (`NativeDecoder.build(probe="time")`): every kind of choice site, each alternative that
           states a program of its own, ROUNDS rotated rounds on the one trainer, each kind's sites
           read in place (the job clock's nanoseconds of the launches the kind's scope labels); the
           table -- each alternative's time over the default's, Q8, and its 95% interval -- and the
           raw readings, round by round
  choose   `native_choose.choose` on that table: for each kind, the alternative resolved faster
           (its whole interval below 256) with the lowest mean; every other site the default

then three trainers of a library without the clock (the library a caller runs), from one seed, take
turns round by round, each running a K-step program stated once before the rounds:

  default   the planner's default program (`default_sites`: the C step's own, word for word)
  default2  the same on a second trainer: the A/A control, the band a program reads against itself
  chosen    the chosen program
  worst     each kind at the alternative the measurement prices slowest, where one is slower than
            the default (absent when none is)

so default2 / default is the noise, chosen / default what the choice does to the whole step and
worst / default what the measurement's slowest choice does (the site readings see a kind's own
launches; the step sees every effect of the choice). Each ratio's geometric mean over the rounds and
its 95% Student-t interval, beside the job-time ratio the measurement's own readings predict for
the chosen and the worst program (each switched kind's site time against the default's at that
kind, summed). Last, every trainer's state: one state, bit for bit (the same calls, programs of the
same step).

  python3 choose.py OUT.json [--k 4] [--rounds 24] [--ab-rounds 32] [--threads 4]
      [--contract bf16+exp+reduce+attnbf16] [--cc gcc]
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
        choose,
        kind_of,
        measure,
        trainer_shape,
    )
    from bcir.hosted.models.native_launch import (
        RECORD_WORDS,
        choice_sites,
        default_sites,
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
    raw: list = []
    t0 = time.perf_counter()
    table = measure(probe, steps=k, rounds=a.rounds, raw=raw)
    measure_s = time.perf_counter() - t0
    shape = trainer_shape(probe, steps=k)
    probe.close()
    chosen = choose(table, spec, shape)
    defaults = default_sites(spec)
    changed = {site: alt for site, alt in chosen.items() if alt != defaults[site]}
    # the program the measurement prices slowest: each kind at its slowest alternative, where that is
    # slower than the default (q8 above 256)
    slowest: dict = {}
    for e in table.entries:
        if e.q8 > 256 and (e.kind not in slowest or e.q8 > slowest[e.kind].q8):
            slowest[e.kind] = e
    worst = {site: slowest[kind_of(site)].alternative if kind_of(site) in slowest else alt
             for site, alt in defaults.items()}  # fmt: skip

    def predicted(sites):
        """The step's job time a program of `sites` should take over the default's, from the
        measurement's own readings: each switched kind's site time against the default's at that
        kind, summed (a first-order sum: the panels' scope overlaps the two VJPs it changes)."""
        kinds = {kind_of(site): alt for site, alt in sites.items() if alt != defaults[site]}
        ratios = []
        for r in raw:
            total = r["total_ns"]["default"]
            delta = sum(r["site_ns"][f"{kind}={alt}"] - r["site_ns"][f"default@{kind}"]
                        for kind, alt in kinds.items())  # fmt: skip
            ratios.append((total + delta) / total)
        return statistics.fmean(ratios)

    print(f"measured in {measure_s:.1f} s:", json.dumps(
        [[e.kind, e.alternative, e.q8, e.low_q8, e.high_q8] for e in table.entries]))  # fmt: skip
    print("chosen (sites off their default):", json.dumps(changed) if changed else "none")
    print("slowest kinds:", json.dumps({kind: e.alternative for kind, e in slowest.items()}))

    # the intervention, on the library a caller runs
    lib = NativeDecoder.build(a.out.parent / (a.out.stem + "-build"), cc=a.cc)
    n = batch * context
    tokens = [(7 * i + 3) % 260 for i in range(n * k)]
    targets = [(7 * i + 4) % 260 for i in range(n * k)]
    opt = NativeAdamW(lr=1e-4)
    d1 = NativeDecoder(lib, spec, **common)
    decisions = d1.program_decisions(steps=k)

    def program(sites):
        return step_program(spec, batch, context, **decisions, sites=sites)

    def words(sites):
        return array("Q", program_words(program(sites)))

    arm_sites = {"default": None, "default2": None, "chosen": chosen}
    if worst != defaults:
        arm_sites["worst"] = worst
    arms = {arm: (d1 if arm == "default" else NativeDecoder(lib, spec, **common), words(sites))
            for arm, sites in arm_sites.items()}  # fmt: skip
    assert all(nd.program_decisions(steps=k) == decisions for nd, _ in arms.values())

    def once(arm):
        nd, words = arms[arm]
        t0 = time.perf_counter()
        nd.run_program(words, tokens, targets, steps=k, optimizer=opt)
        return 1000 * (time.perf_counter() - t0) / k

    for arm in arms:  # warm each trainer: the pages, the pool, the tile unit's admission
        once(arm)
    names = tuple(arms)
    ms = {arm: [] for arm in names}
    for r in range(a.ab_rounds):
        for j in range(len(names)):
            arm = names[(r + j) % len(names)]
            ms[arm].append(once(arm))
    pairs = {"default2 / default (A/A: the default program against itself)": ("default2", "default"),
             "chosen / default (the chosen program against the default)": ("chosen", "default")}  # fmt: skip
    if "worst" in arms:
        pairs["worst / default (the program priced slowest against the default)"] = (
            "worst",
            "default",
        )
    kinds = sorted({kind_of(site) for site in choice_sites(spec)})
    out = {"cpu": next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                       if line.startswith("model name")),
           "vcpus": len([line for line in open("/proc/cpuinfo") if line.startswith("processor")]),
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
           "k": k, "rounds": a.rounds, "ab_rounds": a.ab_rounds, "threads": a.threads,
           "contract": a.contract,
           "compiler": subprocess.run([a.cc, "--version"], capture_output=True,
                                      text=True).stdout.splitlines()[0],
           "sites": {kind: sum(kind_of(s) == kind for s in choice_sites(spec)) for kind in kinds},
           "defaults": {kind: next(defaults[s] for s in defaults if kind_of(s) == kind)
                        for kind in kinds},
           "table": json.loads(table.to_json()),
           "measure_s": round(measure_s, 1),
           "raw": raw,
           "chosen": changed,
           "slowest": {kind: e.alternative for kind, e in slowest.items()},
           "predicted_job_ratio": {"chosen": predicted(chosen), "worst": predicted(worst)},
           "program_records": {arm: len(w) // RECORD_WORDS for arm, (_, w) in arms.items()},
           # the jobs each arm's program issues: a choice regroups the same records into jobs
           "program_launches": {arm: len(program(sites)) for arm, sites in arm_sites.items()},
           "ms_per_step": {arm: [round(v, 3) for v in ms[arm]] for arm in names},
           "median_ms_per_step": {arm: statistics.median(ms[arm]) for arm in names},
           "ratios": {}}  # fmt: skip
    for name, (slow, fast) in pairs.items():
        mean, ci = interval([x / y for x, y in zip(ms[slow], ms[fast])])
        out["ratios"][name] = {"geometric_mean": mean, "ci95": ci,
                               "rounds_slower": sum(x > y for x, y in zip(ms[slow], ms[fast]))}  # fmt: skip
        print(f"{name}: x{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    states = {arm: digest(nd) for arm, (nd, _) in arms.items()}
    for nd, _ in arms.values():
        nd.close()
    out["state_sha256_16"] = states
    out["one_state"] = len(set(states.values())) == 1
    print(json.dumps(out["median_ms_per_step"]), "one state:", out["one_state"])
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
