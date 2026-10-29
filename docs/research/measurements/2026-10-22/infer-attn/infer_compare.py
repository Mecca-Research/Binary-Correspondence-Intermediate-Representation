"""INFER-ATTN on the GEO study's attention model: one forward of 4 x T tokens on the level-2 build
before INFER-ATTN (B-BEFORE), after it (B-AFTER, twice: an A/A control), the bundle's Torch
equations (P-EAGER) and torch.compile (P-IND), rounds in rotated order; the two builds' logits held
equal byte for byte first. usage: python infer_compare.py BEFORE_WORK AFTER_WORK OUT.json [--rounds 12]
"""

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "0")
p = argparse.ArgumentParser()
p.add_argument("before", type=Path)
p.add_argument("after", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--rounds", type=int, default=12)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--times", default="1024,2048")
a = p.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402


def load(work: Path, name: str):
    """The work copy's experiment module under its own name: its ROOT, so its build/, is its own."""
    sys.path.insert(0, str(work.resolve()))
    spec = importlib.util.spec_from_file_location(name, work / "experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


before, after = load(a.before, "exp_before"), load(a.after, "exp_after")
sys.path.insert(0, str(a.after.resolve()))
import oracle  # noqa: E402  (the bundle's Torch equations: the same file in both copies)

torch.set_num_threads(a.threads)


def interval(ratios):
    logs = [math.log(r) for r in ratios]
    m = statistics.fmean(logs)
    t = {7: 2.365, 11: 2.201, 15: 2.131, 23: 2.069}.get(len(logs) - 1, 2.0)
    h = t * statistics.stdev(logs) / math.sqrt(len(logs))
    return {
        "geometric_mean": math.exp(m),
        "ci95": [math.exp(m - h), math.exp(m + h)],
        "rounds": len(logs),
    }


def case(t):
    r = replace(before.Recipe(kind="transformer", ff=256, rope=True), context=t, batch=4)
    rng = np.random.default_rng(20262100 + t)  # geo_compare.py's inference draws
    x = rng.integers(0, r.vocab, (r.batch, r.context), dtype=np.uint32)
    tx = torch.from_numpy(x.astype(np.int64))
    flat = before.initial(r, 881)
    out = {"model": "attention", "batch": 4, "time": t}
    with (
        before.Native(r, seed=881, threads=a.threads, library_name="attention") as nb,
        after.Native(r, seed=881, threads=a.threads, library_name="attention") as na,
        after.Native(r, seed=881, threads=a.threads, library_name="attention") as na2,
    ):
        eager = oracle.torch_model_next(r, flat)
        comp = torch.compile(eager)
        lb, la = nb.forward(x).copy(), na.forward(x).copy()
        out["logits_equal_bytes"] = bool(lb.tobytes() == la.tobytes())
        assert out["logits_equal_bytes"], "INFER-ATTN changed the logits"
        with torch.no_grad():
            ref = eager(tx).numpy()
            t0 = time.perf_counter()
            comp(tx)
            out["compile_first_call_s"] = time.perf_counter() - t0
        out["logits_max_abs_delta_vs_eager"] = float(np.max(np.abs(la - ref)))

        def timed(f):
            def run():
                s = time.perf_counter()
                f()
                return time.perf_counter() - s

            return run

        def tinfer(m):
            def run():
                with torch.no_grad():
                    m(tx)

            return run

        arms = {"B-BEFORE": timed(lambda: nb.forward(x)), "B-AFTER": timed(lambda: na.forward(x)),
                "B-AFTER-AA": timed(lambda: na2.forward(x)), "P-EAGER": timed(tinfer(eager)),
                "P-IND": timed(tinfer(comp))}  # fmt: skip
        for f in arms.values():
            f()
        ms = {k: [] for k in arms}
        names = list(arms)
        for q in range(a.rounds):
            for j in range(len(names)):
                k = names[(q + j) % len(names)]
                ms[k].append(1000 * arms[k]())
    out["ms_per_forward"] = {k: statistics.median(v) for k, v in ms.items()}
    out["after_over"] = {k: interval([x / y for x, y in zip(ms["B-AFTER"], ms[k])])
                         for k in ("B-BEFORE", "B-AFTER-AA", "P-EAGER", "P-IND")}  # fmt: skip
    out["raw_ms"] = {k: [round(v, 4) for v in vs] for k, vs in ms.items()}
    print(json.dumps({k: out[k] for k in ("time", "ms_per_forward")}), flush=True)
    return out


cpu = next(
    (ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")),
    "?",
)
doc = {"host": {"cpu": cpu, "threads": a.threads, "python": platform.python_version(),
                "torch": str(torch.__version__), "numpy": np.__version__},
       "rounds": a.rounds, "cases": [case(int(t)) for t in a.times.split(",")]}  # fmt: skip
a.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
print("wrote", a.out)
