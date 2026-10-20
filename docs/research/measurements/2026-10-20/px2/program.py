"""PX-2a (2026-10-20): the native step run from its launch program against the step's own sequence.

Four call paths on ONE trainer of this tree's library, rotated round by round in one process on
the 9.84M comparison decoder (B=8, T=128) at the given threads and contract, each path K steps a
round from the same token stream -- every path trains the same bits (train.plan.program.mismatch),
so they take turns on one state and no comparison is between two placements of it:

  run      NativeDecoder.run: bcir_decoder_run_accum, the step's fixed sequence
  run2     the same call again: the A/A control, the band one path reads against itself
  program  NativeDecoder.run_program lent an array('Q') of the program native_launch.step_program
           states for the trainer's own decisions (made once, before the rounds: a caller holding
           its plan) -- bcir_decoder_run_program's check of the whole program, then every record
  binding  the same words as a list: the binding's checks and conversion inside every call

so run2 / run is the noise, program / run the runner's cost (its check and its dispatch) and
binding / program what a list costs besides. Each ratio's geometric mean over the rounds and its
95% Student-t interval. Then: the binding's work on the list alone (its type check and conversion,
the best of 20); the check alone, called directly -- the program run as one step more than it
states, refused at the check's last rule, every record walked and nothing run -- over CHECKS calls,
the state held unchanged by them; and a second trainer that ran every one of the same calls through
run, its state against the rotated one's (one state: the same steps, three paths).

  python3 program.py OUT.json [--k 10] [--rounds 24] [--threads 4]
      [--contract bf16+exp+reduce+attnbf16] [--cc gcc] [--checks 200]
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
from dataclasses import asdict
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
    p.add_argument("--checks", type=int, default=200)
    a = p.parse_args()
    from bcir.frontends.models.decode import DecoderSpec
    from bcir.hosted.models import native as nat
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder
    from bcir.hosted.models.native_launch import SPACES, program_words, step_program

    lib = build(REPO / "runtime" / "c", a.out.parent / (a.out.stem + "-build"), a.cc)
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4,
                       tied_embeddings=True)  # fmt: skip
    batch, context, k = 8, 128, a.k
    n = batch * context
    tokens = [(7 * i + 3) % 260 for i in range(n * (k + 1))]
    targets = [(7 * i + 4) % 260 for i in range(n * (k + 1))]
    opt = NativeAdamW(lr=1e-4)
    arms = ("run", "run2", "program", "binding")
    nd, ref = (NativeDecoder(lib, spec, batch_size=batch, context_length=context, threads=a.threads,
                             contract=a.contract) for _ in range(2))  # fmt: skip

    def words_for(nd, steps):
        """The program for `nd`'s own decisions, as `native_decoder_fixtures` states them."""
        return program_words(step_program(
            spec, batch, context, contract=nat._CONTRACTS[a.contract], recompute=nd.recompute,
            workspace_bytes=nd.workspace_bytes, panels_need=nd.plan_memory.workspace_panels,
            alignment={SPACES[name]: ct.addressof(buf) % 64 for name, buf in nd.buffers.items()},
            steps=steps, accumulation=1))  # fmt: skip

    words = words_for(nd, k)
    code = array("Q", words)
    records = len(code) // 48
    xs, ys = tokens[: n * k], targets[: n * k]
    calls = 0

    def once(arm):
        nonlocal calls
        t0 = time.perf_counter()
        if arm in ("run", "run2"):
            nd.run(xs, ys, steps=k, optimizer=opt)
        else:
            nd.run_program(code if arm == "program" else words, xs, ys, steps=k, optimizer=opt)
        calls += 1
        return 1000 * (time.perf_counter() - t0) / k

    for arm in arms:  # warm each path: the pages, the pool, the tile unit's admission
        once(arm)
    ms = {arm: [] for arm in arms}
    for r in range(a.rounds):
        for j in range(len(arms)):
            arm = arms[(r + j) % len(arms)]
            ms[arm].append(once(arm))
    best = 1e9
    for _ in range(20):  # the binding's work on a list: its type check and its conversion
        t0 = time.perf_counter()
        if not set(map(type, words)) - {int}:
            array("Q", words)
        best = min(best, time.perf_counter() - t0)
    # the check alone, called directly: refused at its last rule (K steps for a call of K + 1)
    fn = nd.lib.bcir_decoder_run_program
    fn.argtypes = [ct.POINTER(nat._State), ct.POINTER(nat._Optimizer), ct.POINTER(ct.c_uint64),
                   nat._SZ, nat._UP, nat._UP, nat._SZ, ct.POINTER(ct.c_double), nat._SZ, nat._SZ,
                   ct.POINTER(nat._Event), nat._SZ, ct.POINTER(nat._SZ)]  # fmt: skip
    fn.restype = ct.c_int
    view = (ct.c_uint64 * len(code)).from_buffer(code)
    x1, y1 = nd._tokens(tokens, steps=k + 1), nd._tokens(targets, steps=k + 1)
    o, events, done = nat._Optimizer(**asdict(opt)), (nat._Event * (k + 1))(), nat._SZ()
    before = digest(nd)
    t0 = time.perf_counter()
    for _ in range(a.checks):
        rc = fn(ct.byref(nd.state), ct.byref(o), view, records, x1, y1, len(x1), None, k + 1, 1,
                events, k + 1, ct.byref(done))  # fmt: skip
        assert rc == 1, rc
    check_us = 1e6 * (time.perf_counter() - t0) / a.checks
    assert digest(nd) == before, "a refused program wrote the state"
    pairs = {"run2 / run (A/A: one path against itself)": ("run2", "run"),
             "program / run (the runner: its check and dispatch)": ("program", "run"),
             "binding / program (a list's checks and conversion)": ("binding", "program"),
             "binding / run (all of it, from a list)": ("binding", "run")}  # fmt: skip
    out = {"cpu": next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                       if line.startswith("model name")),
           "vcpus": len([line for line in open("/proc/cpuinfo") if line.startswith("processor")]),
           "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
           "runtime_c_diff_sha256_16": hashlib.sha256(subprocess.run(
               ["git", "-C", str(REPO), "diff", "HEAD", "--", "runtime/c"],
               capture_output=True).stdout).hexdigest()[:16],
           "k": k, "rounds": a.rounds, "threads": a.threads, "contract": a.contract,
           "compiler": subprocess.run([a.cc, "--version"], capture_output=True,
                                      text=True).stdout.splitlines()[0],
           "program": {"records": records, "records_per_step": records / k,
                       "bytes": 8 * len(code)},
           "check_us": round(check_us, 2), "check_us_per_step": round(check_us / k, 3),
           "list_conversion_ms": round(1000 * best, 3),
           "ms_per_step": {arm: [round(v, 3) for v in ms[arm]] for arm in arms},
           "median_ms_per_step": {arm: statistics.median(ms[arm]) for arm in arms},
           "ratios": {}}  # fmt: skip
    for name, (slow, fast) in pairs.items():
        mean, ci = interval([x / y for x, y in zip(ms[slow], ms[fast])])
        out["ratios"][name] = {"geometric_mean": mean, "ci95": ci,
                               "rounds_slower": sum(x > y for x, y in zip(ms[slow], ms[fast]))}  # fmt: skip
        print(f"{name}: x{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    for _ in range(calls):  # the reference: every call the rotated trainer made, through run
        ref.run(xs, ys, steps=k, optimizer=opt)
    states = {"rotated": digest(nd), "run alone": digest(ref)}
    nd.close()
    ref.close()
    out["calls"] = calls
    out["state_sha256_16"] = states
    out["one_state"] = states["rotated"] == states["run alone"]
    print(
        json.dumps(out["median_ms_per_step"]),
        f"check {check_us:.1f} us, list {1000 * best:.2f} ms",
        json.dumps(out["program"]),
        "one state:",
        out["one_state"],
    )
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
