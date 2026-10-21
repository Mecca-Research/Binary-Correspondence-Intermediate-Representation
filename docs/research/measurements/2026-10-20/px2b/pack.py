"""PX-2b (2026-10-20): the native step run from a StreamPack and its ExecutionPlanV1 -- what the
artifacts cost to make, what the C lowering costs, and what a run from them costs besides the run.

On the 9.84M comparison decoder (B=8, T=128) at the given threads and contract, for a call of K
steps:

  build    the program `native_launch.step_program` states for the trainer's own decisions, as a
           module (`native_pack.program_pack`), planned by K_BCIR (`optimize`), hydrated by GEM into
           a StreamPack (`hydrate`), its ExecutionPlanV1 minted from the same realization
           (`plan_from_realization`, the lifetimes the C step's own places), both encoded -- each
           stage timed once (a caller makes its artifacts once; the times are wall, indicative)
  lower    `bcir_decoder_lower_pack` on the two artifacts, the sizes first: CHECKS calls, the
           median; and `bcir_ep_check_pack` alone (the plan's wire laws, its alias law among them,
           and the pack's binding), the same count -- the part of the lowering that is quadratic in
           the lifetimes

then three call paths rotated round by round on ONE trainer of this tree's pack library, each K
steps a round from one token stream -- every path trains the same bits (train.plan.pack.mismatch),
so they take turns on one state and no ratio compares two placements of it:

  program  NativeDecoder.run_program lent the array('Q') the lowering wrote (lowered once, before
           the rounds: a caller holding its artifacts' program)
  program2 the same call again: the A/A control, the band one path reads against itself
  pack     NativeDecoder.run_pack: the lowering inside every call, then the same run

so program2 / program is the noise and pack / program what lowering in every call costs. Each
ratio's geometric mean over the rounds and its 95% Student-t interval. Last, a second trainer that
ran every one of the same calls through `run` (the fixed sequence): its state against the rotated
one's (one state: the same steps, three paths and the step's own).

  python3 pack.py OUT.json [--k 10] [--rounds 24] [--threads 4]
      [--contract bf16+exp+reduce+attnbf16] [--cc gcc] [--checks 50]
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
T95 = {9: 2.262, 11: 2.201, 15: 2.131, 19: 2.093, 23: 2.069, 31: 2.042, 47: 2.012, 63: 1.998}


def interval(ratios):
    logs = [math.log(v) for v in ratios]
    m = statistics.fmean(logs)
    half = T95.get(len(logs) - 1, 2.0) * statistics.stdev(logs) / math.sqrt(len(logs))
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
    p.add_argument("--checks", type=int, default=50)
    a = p.parse_args()
    from bcir.abi.execution_plan_abi import encode_plan
    from bcir.abi.streampack_abi import encode
    from bcir.frontends.models.decode import DecoderSpec
    from bcir.gem.execution_plan import Lifetime, plan_from_realization
    from bcir.gem.streampack import hydrate
    from bcir.hosted.models import native as nat
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder
    from bcir.hosted.models.native_launch import SPACES, program_words, step_program
    from bcir.hosted.models.native_pack import program_pack
    from bcir.kbcir import optimize
    from bcir.kbcir.cost import TargetProfile, Theta

    lib = NativeDecoder.build(a.out.parent / (a.out.stem + "-build"), cc=a.cc, pack=True)
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4,
                       tied_embeddings=True)  # fmt: skip
    batch, context, k = 8, 128, a.k
    n = batch * context
    tokens = [(7 * i + 3) % 260 for i in range(n * k)]
    targets = [(7 * i + 4) % 260 for i in range(n * k)]
    opt = NativeAdamW(lr=1e-4)
    nd, ref = (NativeDecoder(lib, spec, batch_size=batch, context_length=context, threads=a.threads,
                             contract=a.contract) for _ in range(2))  # fmt: skip

    # the artifacts, stage by stage (native_pack.pack_artifacts' own stages)
    stages = {}
    t0 = time.perf_counter()
    program = step_program(
        spec, batch, context, contract=nat._CONTRACTS[a.contract], recompute=nd.recompute,
        workspace_bytes=nd.workspace_bytes, panels_need=nd.plan_memory.workspace_panels,
        alignment={SPACES[name]: ct.addressof(buf) % 64 for name, buf in nd.buffers.items()},
        steps=k, accumulation=1)  # fmt: skip
    stages["step_program"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    tp = program_pack(spec, batch, context, program, recompute=nd.recompute)
    stages["program_pack (the module)"] = time.perf_counter() - t0
    target = TargetProfile.x86_avx512()
    t0 = time.perf_counter()
    result = optimize(tp.module, target, Theta.cool())
    stages["optimize (K_BCIR)"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pack_obj = hydrate(tp.module, result)
    stages["hydrate (GEM)"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    plan_obj = plan_from_realization(tp.module, result, target)
    plan_obj.lifetimes = [Lifetime(q.rid, q.bank, q.offset, q.size, 4, q.first, q.last)
                          for q in sorted(tp.placements, key=lambda q: q.rid)]  # fmt: skip
    stages["plan_from_realization"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pack, plan = encode(pack_obj), encode_plan(plan_obj)
    stages["encode (both)"] = time.perf_counter() - t0
    words = nd.lower_pack(pack, plan)
    assert list(words) == list(program_words(list(program))) == list(tp.words), "not the program"

    # the lowering alone, and the plan's own check inside it
    t_lower = []
    for _ in range(a.checks):
        t0 = time.perf_counter()
        nd.lower_pack(pack, plan)
        t_lower.append(time.perf_counter() - t0)
    check = nd.lib.bcir_ep_check_pack
    check.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_char_p, ct.c_size_t]
    check.restype = ct.c_int
    t_check = []
    for _ in range(a.checks):
        t0 = time.perf_counter()
        rc = check(plan, len(plan), pack, len(pack))
        t_check.append(time.perf_counter() - t0)
        assert rc == 0, rc

    arms = ("program", "program2", "pack")
    calls = 0

    def once(arm):
        nonlocal calls
        t0 = time.perf_counter()
        if arm == "pack":
            nd.run_pack(pack, plan, tokens, targets, steps=k, optimizer=opt)
        else:
            nd.run_program(words, tokens, targets, steps=k, optimizer=opt)
        calls += 1
        return 1000 * (time.perf_counter() - t0) / k

    for arm in arms:  # warm each path: the pages, the pool, the tile unit's admission
        once(arm)
    ms = {arm: [] for arm in arms}
    for r in range(a.rounds):
        for j in range(len(arms)):
            arm = arms[(r + j) % len(arms)]
            ms[arm].append(once(arm))
    pairs = {"program2 / program (A/A: one path against itself)": ("program2", "program"),
             "pack / program (the lowering inside every call)": ("pack", "program")}  # fmt: skip
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
           "artifacts": {"claims": tp.claims, "launches": len(tp.program),
                         "values": len(tp.placements), "pack_bytes": len(pack),
                         "plan_bytes": len(plan), "program_records": len(words) // 48,
                         "program_bytes": 8 * len(words)},
           "build_s": {name: round(v, 4) for name, v in stages.items()},
           "build_s_total": round(sum(stages.values()), 4),
           "lower_ms": {"median": 1000 * statistics.median(t_lower), "min": 1000 * min(t_lower),
                        "calls": a.checks},
           "plan_check_ms": {"median": 1000 * statistics.median(t_check),
                             "min": 1000 * min(t_check), "calls": a.checks},
           "ms_per_step": {arm: [round(v, 3) for v in ms[arm]] for arm in arms},
           "median_ms_per_step": {arm: statistics.median(ms[arm]) for arm in arms},
           "ratios": {}}  # fmt: skip
    for name, (slow, fast) in pairs.items():
        mean, ci = interval([x / y for x, y in zip(ms[slow], ms[fast])])
        out["ratios"][name] = {"geometric_mean": mean, "ci95": ci,
                               "rounds_slower": sum(x > y for x, y in zip(ms[slow], ms[fast]))}  # fmt: skip
        print(f"{name}: x{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    for _ in range(calls):  # the reference: every call the rotated trainer made, through run
        ref.run(tokens, targets, steps=k, optimizer=opt)
    states = {"rotated": digest(nd), "run alone": digest(ref)}
    nd.close()
    ref.close()
    out["calls"] = calls
    out["state_sha256_16"] = states
    out["one_state"] = states["rotated"] == states["run alone"]
    print(json.dumps(out["artifacts"]), json.dumps(out["build_s"]),
          f"lower {out['lower_ms']['median']:.2f} ms (plan check {out['plan_check_ms']['median']:.2f} ms)",
          json.dumps(out["median_ms_per_step"]), "one state:", out["one_state"])  # fmt: skip
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
