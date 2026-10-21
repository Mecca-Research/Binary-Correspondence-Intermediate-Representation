"""PX-4 (2026-10-20): the native step's transient values placed by their lifetimes -- what the
placement saves, what it costs to make, and what a run on the placed arena costs against a run on
the C step's spans.

On the 9.84M comparison decoder (B=8, T=128) at the given threads and contract, for a call of K
steps:

  bytes    the transient arenas the C plan reserves (the activation arena under the trainer's
           recompute set and the scratch), the placed arena (`native_pack.place_by_lifetime`: every
           activation and scratch value of the K-step pack laid out by its lifetime in one bank),
           and G26's concurrent-live bound over the values it places -- and the same three at the
           decoder's other recompute sets and contracts (no run: the model's figures alone)
  place    `place_by_lifetime` on the K-step pack: PLACES calls, the median (wall, indicative)

then three trainers of this tree's pack library, from one seed, take turns round by round, each
running the K-step pack's program lowered once before the rounds (a caller holding its artifacts'
program):

  c        the pack placed at the C step's own spans, on a trainer lending the C plan's arenas
  c2       the same on a second such trainer: the A/A control, the band a path reads against itself
  placed   the pack placed by lifetime, on a trainer lending exactly the placed arena

so c2 / c is the noise and placed / c what the placement does to the step's time (its working set
shrinks; the kernels and their arithmetic are the same). Each ratio's geometric mean over the rounds
and its 95% Student-t interval. Last, every trainer's state: one state, bit for bit (the same calls,
three placements of the same program).

  python3 place.py OUT.json [--k 10] [--rounds 24] [--threads 4]
      [--contract bf16+exp+reduce+attnbf16] [--cc gcc] [--places 20]
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
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--contract", default="bf16+exp+reduce+attnbf16")
    p.add_argument("--cc", default="gcc")
    p.add_argument("--places", type=int, default=20)
    a = p.parse_args()
    from bcir.frontends.models.decode import DecoderSpec
    from bcir.hosted.models import native as nat
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder
    from bcir.hosted.models.native_launch import SPACES
    from bcir.hosted.models.native_pack import (
        pack_artifacts,
        place_by_lifetime,
        placed_words,
        training_pack,
    )

    lib = NativeDecoder.build(a.out.parent / (a.out.stem + "-build"), cc=a.cc, pack=True)
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4,
                       tied_embeddings=True)  # fmt: skip
    batch, context, k = 8, 128, a.k
    n = batch * context
    tokens = [(7 * i + 3) % 260 for i in range(n * k)]
    targets = [(7 * i + 4) % 260 for i in range(n * k)]
    opt = NativeAdamW(lr=1e-4)
    common = dict(batch_size=batch, context_length=context, threads=a.threads,
                  contract=a.contract)  # fmt: skip
    c1 = NativeDecoder(lib, spec, **common)

    def decisions(nd, contract=None, recompute=None):
        return dict(
            contract=nat._CONTRACTS[contract or a.contract],
            recompute=nd.recompute if recompute is None else recompute,
            workspace_bytes=nd.workspace_bytes, panels_need=nd.plan_memory.workspace_panels,
            alignment={SPACES[name]: ct.addressof(buf) % 64 for name, buf in nd.buffers.items()},
        )  # fmt: skip

    tp = training_pack(spec, batch, context, steps=k, accumulation=1, **decisions(c1))
    t_place = []
    for _ in range(a.places):
        t0 = time.perf_counter()
        placed = place_by_lifetime(tp)
        t_place.append(time.perf_counter() - t0)
    c_bytes = 4 * (c1.activation_count + c1.state.plan.scratch)
    assert placed.extent == placed.bound and placed.optimal, (placed.extent, placed.bound)
    c2 = NativeDecoder(lib, spec, **common)
    pl = NativeDecoder(lib, spec, transient=(placed.floats, 0), **common)
    pack_c, plan_c, _, _ = pack_artifacts(tp)
    pack_p, plan_p, _, _ = pack_artifacts(tp, placements=placed.placements)
    words_c = c1.lower_pack(pack_c, plan_c)
    words_p = pl.lower_pack(pack_p, plan_p)
    assert tuple(words_c) == tp.words and tuple(words_p) == placed_words(tp, placed.placements)

    # the model's figures at the decoder's other recompute sets and contracts (one step a call)
    table = []
    for contract in ("v1", a.contract):
        for recompute in (0, 0b111, 0b111111):
            probe = NativeDecoder(lib, spec, batch_size=batch, context_length=context,
                                  contract=contract, recompute=recompute)  # fmt: skip
            one = training_pack(spec, batch, context, **decisions(probe, contract, recompute))
            got = place_by_lifetime(one)
            table.append({"contract": contract, "recompute": recompute,
                          "c_bytes": 4 * (probe.activation_count + probe.state.plan.scratch),
                          "placed_bytes": got.extent, "bound_bytes": got.bound,
                          "optimal": got.optimal, "values": len(got.placements)})  # fmt: skip
            probe.close()
            print(json.dumps(table[-1]))

    arms = {"c": (c1, words_c), "c2": (c2, words_c), "placed": (pl, words_p)}

    def once(arm):
        nd, words = arms[arm]
        t0 = time.perf_counter()
        nd.run_program(words, tokens, targets, steps=k, optimizer=opt)
        return 1000 * (time.perf_counter() - t0) / k

    for arm in arms:  # warm each trainer: the pages, the pool, the tile unit's admission
        once(arm)
    names = tuple(arms)
    ms = {arm: [] for arm in names}
    for r in range(a.rounds):
        for j in range(len(names)):
            arm = names[(r + j) % len(names)]
            ms[arm].append(once(arm))
    pairs = {"c2 / c (A/A: the C spans against themselves)": ("c2", "c"),
             "placed / c (the placed arena against the C spans)": ("placed", "c")}  # fmt: skip
    out = {"cpu": next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                       if line.startswith("model name")),
           "vcpus": len([line for line in open("/proc/cpuinfo") if line.startswith("processor")]),
           "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
           "worktree_diff_sha256_16": hashlib.sha256(subprocess.run(
               ["git", "-C", str(REPO), "diff", "HEAD", "--", "runtime/c", "bcir"],
               capture_output=True).stdout).hexdigest()[:16],
           "k": k, "rounds": a.rounds, "threads": a.threads, "contract": a.contract,
           "compiler": subprocess.run([a.cc, "--version"], capture_output=True,
                                      text=True).stdout.splitlines()[0],
           "pack": {"claims": tp.claims, "launches": len(tp.program), "values": len(tp.placements),
                    "transient_values": sum(q.bank in ("activations", "scratch")
                                            for q in tp.placements),
                    "parts": placed.parts},
           "bytes": {"c_plan": c_bytes, "placed": placed.extent, "bound": placed.bound,
                     "placed_over_c": placed.extent / c_bytes, "optimal": placed.optimal,
                     "trainer_memory": {"c": c1.memory_bytes, "placed": pl.memory_bytes}},
           "place_ms": {"median": 1000 * statistics.median(t_place), "min": 1000 * min(t_place),
                        "calls": a.places},
           "table": table,
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
    print(json.dumps(out["bytes"]), f"place {out['place_ms']['median']:.1f} ms",
          json.dumps(out["median_ms_per_step"]), "one state:", out["one_state"])  # fmt: skip
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
