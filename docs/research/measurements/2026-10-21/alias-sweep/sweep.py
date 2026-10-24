"""ALIAS-SWEEP (2026-10-21): what the plan's alias law costs judged pairwise and by the sweep, on
both rails, against the steps a training pack carries -- the 9.84M comparison decoder (B=8,
T=128), packs of K steps, each the median of REPS calls:

  ep_verify            the plan's laws, the alias law pairwise (`bcir_ep_verify`: no scratch)
  ep_verify_lent       the same laws, the alias law swept over lent scratch (`bcir_ep_verify_lent`,
                       exactly `bcir_ep_sweep_bytes` of it)
  walk                 the lowering's one walk of the plan's and the pack's laws
                       (`bcir_ep_check_pack_walk`, no callbacks) without scratch: the walk the
                       lowering made before ALIAS-SWEEP
  walk_lent            the same walk with exactly `bcir_ep_sweep_bytes` of scratch: the walk now
  lower_pack           the whole lowering (`NativeDecoder.lower_pack`), its walk lent
  lower_pack_before    with --before CHECKOUT: the same lowering in that checkout (the tree before
                       ALIAS-SWEEP), timed by its own PX-2b driver (measurements/2026-10-20/px2b)
  py_pairwise          the Python codec's alias law as it was (each lifetime against every earlier
                       one), over the plan's lifetimes
  py_sweep             `execution_plan_abi.first_alias` over the same lifetimes

every verdict checked first (the plan admitted, the same on both rails and both ways), with the
pack's claims, the plan's lifetimes and their bytes. Then the same two Python rows over GEM's
reuse shape at 2,000 to 32,000 lifetimes (two spans of one bank, each value live two ticks).

  python3 sweep.py OUT.json [--steps 1,2,5,10,20] [--reps 15] [--contract kblock+exp+reduce+attn]
      [--cc gcc] [--before CHECKOUT]
"""

import argparse
import ctypes as ct
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))


def pairwise_first_alias(rows) -> int:
    """The alias law as both rails judged it before ALIAS-SWEEP."""
    from bcir.abi.execution_plan_abi import lifetimes_alias

    for i, a in enumerate(rows):
        for b in rows[:i]:
            if lifetimes_alias(b, a):
                return i
    return len(rows)


def reused(n: int) -> list:
    from bcir.gem.execution_plan import Lifetime

    return [Lifetime(rid=i + 1, bank="ram", offset=64 * (i % 2), size_bytes=64, alignment=64,
                     first_phase=i, last_phase=i + 1, first_tick=i, last_tick=i + 2)
            for i in range(n)]  # fmt: skip


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--steps", default="1,2,5,10,20")
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--contract", default="kblock+exp+reduce+attn")
    p.add_argument("--cc", default="gcc")
    p.add_argument("--before", type=Path, help="a checkout of the tree before ALIAS-SWEEP")
    a = p.parse_args()
    before = {}
    if a.before is not None:  # its own driver, its own interpreter run: its code, not this tree's
        out_before = a.out.parent / (a.out.stem + "-before.json")
        subprocess.run([sys.executable, str(a.before / "docs/research/measurements/2026-10-20/px2b/lower.py"),
                        str(out_before), "--steps", a.steps, "--reps", str(a.reps),
                        "--contract", a.contract, "--cc", a.cc], check=True)  # fmt: skip
        doc = json.loads(out_before.read_text())
        before = {"commit": doc["commit"], "diff": doc["worktree_diff_sha256_16"],
                  "lower_pack": {r["steps"]: r["ms"]["lower_pack"] for r in doc["rows"]}}  # fmt: skip
    from bcir.abi.execution_plan_abi import decode_plan, first_alias
    from bcir.frontends.models.decode import DecoderSpec
    from bcir.hosted.models import native as nat
    from bcir.hosted.models.native import NativeDecoder
    from bcir.hosted.models.native_launch import SPACES, step_program
    from bcir.hosted.models.native_pack import pack_artifacts, program_pack

    lib = NativeDecoder.build(a.out.parent / (a.out.stem + "-build"), cc=a.cc, pack=True)
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4,
                       tied_embeddings=True)  # fmt: skip
    nd = NativeDecoder(lib, spec, batch_size=8, context_length=128, threads=1, contract=a.contract)
    L = nd.lib
    L.bcir_ep_verify.argtypes = [ct.c_char_p, ct.c_size_t]
    L.bcir_ep_verify_lent.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
    L.bcir_ep_sweep_bytes.argtypes = [ct.c_uint32]
    L.bcir_ep_sweep_bytes.restype = ct.c_size_t
    # no callbacks (NULL): the walk's own cost, as the lowering's C callbacks add little to it
    L.bcir_ep_check_pack_walk.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_char_p, ct.c_size_t,
                                          ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                          ct.c_size_t]  # fmt: skip

    def median_ms(fn, *args, reps=None) -> float:
        times = []
        for _ in range(reps or a.reps):
            t0 = time.perf_counter()
            fn(*args)
            times.append(time.perf_counter() - t0)
        return 1000 * statistics.median(times)

    rows = []
    for k in (int(x) for x in a.steps.split(",")):
        program = step_program(
            spec, 8, 128, contract=nat._CONTRACTS[a.contract], recompute=nd.recompute,
            workspace_bytes=nd.workspace_bytes, panels_need=nd.plan_memory.workspace_panels,
            alignment={SPACES[n]: ct.addressof(b) % 64 for n, b in nd.buffers.items()},
            steps=k, accumulation=1)  # fmt: skip
        tp = program_pack(spec, 8, 128, program, recompute=nd.recompute)
        pack, plan, _, _ = pack_artifacts(tp)
        lifetimes = list(decode_plan(plan).lifetimes)
        need = L.bcir_ep_sweep_bytes(len(lifetimes))
        scratch = (ct.c_uint8 * need)()
        walk = L.bcir_ep_check_pack_walk
        verdicts = (L.bcir_ep_verify(plan, len(plan)), L.bcir_ep_verify_lent(plan, len(plan), scratch, need),
                    walk(plan, len(plan), pack, len(pack), None, None, None, None, 0),
                    walk(plan, len(plan), pack, len(pack), None, None, None, scratch, need))  # fmt: skip
        assert verdicts == (0, 0, 0, 0), verdicts
        assert first_alias(lifetimes) == pairwise_first_alias(lifetimes) == len(lifetimes)
        row = {"steps": k, "claims": tp.claims, "lifetimes": len(lifetimes),
               "plan_bytes": len(plan), "pack_bytes": len(pack), "scratch_bytes": need,
               "ms": {"ep_verify": median_ms(L.bcir_ep_verify, plan, len(plan)),
                      "ep_verify_lent": median_ms(L.bcir_ep_verify_lent, plan, len(plan), scratch,
                                                  need),
                      "walk": median_ms(walk, plan, len(plan), pack, len(pack), None, None,
                                        None, None, 0),
                      "walk_lent": median_ms(walk, plan, len(plan), pack, len(pack), None, None,
                                             None, scratch, need),
                      "lower_pack": median_ms(nd.lower_pack, pack, plan),
                      "py_pairwise": median_ms(pairwise_first_alias, lifetimes, reps=3),
                      "py_sweep": median_ms(first_alias, lifetimes)}}  # fmt: skip
        if k in before.get("lower_pack", {}):
            row["ms"]["lower_pack_before"] = before["lower_pack"][k]
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("steps", "claims", "lifetimes")}),
              json.dumps({key: round(v, 3) for key, v in row["ms"].items()}), flush=True)  # fmt: skip
    nd.close()
    shape = []
    for n in (2_000, 8_000, 32_000):
        rows_n = reused(n)
        assert first_alias(rows_n) == n
        ms = {"py_sweep": median_ms(first_alias, rows_n, reps=5),
              "py_pairwise": median_ms(pairwise_first_alias, rows_n, reps=1) if n <= 8_000
              else None}  # fmt: skip
        shape.append({"lifetimes": n, "ms": ms})
        print(json.dumps(shape[-1]), flush=True)
    out = {"cpu": next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                       if line.startswith("model name")),
           "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
           "worktree_diff_sha256_16": hashlib.sha256(subprocess.run(
               ["git", "-C", str(REPO), "diff", "HEAD", "--", "runtime/c", "bcir"],
               capture_output=True).stdout).hexdigest()[:16],
           "trees": {path: subprocess.run(["git", "-C", str(REPO), "rev-parse", f"HEAD:{path}"],
                                          capture_output=True, text=True).stdout.strip()
                     for path in ("bcir", "runtime/c")},
           "contract": a.contract, "reps": a.reps, "python": sys.version.split()[0],
           "compiler": subprocess.run([a.cc, "--version"], capture_output=True,
                                      text=True).stdout.splitlines()[0],
           "before": {k: v for k, v in before.items() if k != "lower_pack"},
           "rows": rows, "reuse_shape": shape}  # fmt: skip
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
