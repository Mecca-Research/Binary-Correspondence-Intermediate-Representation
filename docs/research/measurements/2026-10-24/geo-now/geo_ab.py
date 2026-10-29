"""GEO-NOW: the PR818 study's three models on main's tensor unit (level 2, BEFORE) and on this branch's (AFTER: the same
overlays, the same headers, build_amx.py's builds), training and inference, both builds in one process.

B-BEFORE, B-AFTER and B-AFTER-AA (an A/A control, its own instance): the bundle's own `Native` host over each work
copy's build, the study's binary32 contract (14), four pool threads. Training: one observation is `steps` updates
(geo_compare.py's train_case draws), each arm restored to the same initial state outside the clock, the two builds'
trained states (weights, both moments, the step) held equal by digest first. Inference: one forward of 4 x T
tokens (geo_compare.py's infer_case draws), the two builds' logits held equal byte for byte first. Rounds rotate the
arms' order. usage: python geo_ab.py BEFORE_WORK AFTER_WORK OUT.json [--rounds 16] [--steps 3]
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
p = argparse.ArgumentParser()
p.add_argument("before", type=Path)
p.add_argument("after", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--rounds", type=int, default=16)
p.add_argument("--steps", type=int, default=3)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--models", default="global,query_full,attention")
p.add_argument("--train-t", default="128,256,512")
p.add_argument("--infer-t", default="1024,2048")
a = p.parse_args()

import numpy as np  # noqa: E402


def load(work: Path, name: str):
    """The work copy's experiment module under its own name: its ROOT, so its build/, is its own."""
    sys.path.insert(0, str(work.resolve()))
    spec = importlib.util.spec_from_file_location(name, work / "experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


before, after = load(a.before, "exp_before"), load(a.after, "exp_after")
LIB = {"global": "guard_compact", "query_full": "query_guard_compact", "attention": "attention"}
T95 = {7: 2.365, 11: 2.201, 15: 2.131, 23: 2.069}


def recipe(model):  # geo_compare.py's
    if model == "attention":
        return before.Recipe(kind="transformer", ff=256, rope=True)
    return before.Recipe(
        kind="global", ff=257, query_aware=model.startswith("query"), precision_guard=True
    )


def interval(ratios):
    logs = [math.log(r) for r in ratios]
    m = statistics.fmean(logs)
    t = T95.get(len(logs) - 1, 1.96 + 2.4 / (len(logs) - 1))
    h = t * statistics.stdev(logs) / math.sqrt(len(logs))
    return {
        "geometric_mean": math.exp(m),
        "ci95": [math.exp(m - h), math.exp(m + h)],
        "rounds": len(logs),
    }


def rotate(arms, per):
    for f in arms.values():
        f()  # warm
    ms = {k: [] for k in arms}
    names = list(arms)
    for q in range(a.rounds):
        for j in range(len(names)):
            k = names[(q + j) % len(names)]
            ms[k].append(1000 * arms[k]() / per)
    return ms


def summary(out, ms, unit):
    out[f"ms_per_{unit}"] = {k: statistics.median(v) for k, v in ms.items()}
    out["after_over"] = {k: interval([x / y for x, y in zip(ms["B-AFTER"], ms[k])])
                         for k in ("B-BEFORE", "B-AFTER-AA")}  # fmt: skip
    out["raw_ms"] = {k: [round(v, 4) for v in vs] for k, vs in ms.items()}
    o = out["after_over"]
    print(json.dumps({"model": out["model"], "mode": out["mode"], "time": out["time"],
                      "ms": {k: round(v, 2) for k, v in out[f"ms_per_{unit}"].items()},
                      "after_over": {k: [round(v["geometric_mean"], 3), *(round(x, 3) for x in v["ci95"])]
                                     for k, v in o.items()}}), flush=True)  # fmt: skip
    return out


def train_case(model, t):
    r = replace(recipe(model), context=t, batch=4)
    rng = np.random.default_rng(81820261018 + t + r.ff)
    xs = rng.integers(0, r.vocab, (a.steps, r.batch, r.context), dtype=np.uint32)
    ys = np.roll(xs, -1, axis=2).copy()
    rates = np.full(a.steps, 1e-4, np.float64)
    out = {"model": model, "mode": "train", "batch": 4, "time": t, "steps_per_observation": a.steps}
    lib = LIB[model]
    with (
        before.Native(r, seed=881, threads=a.threads, library_name=lib) as nb,
        after.Native(r, seed=881, threads=a.threads, library_name=lib) as na,
        after.Native(r, seed=881, threads=a.threads, library_name=lib) as na2,
    ):
        inits = {"B-BEFORE": nb.snapshot(), "B-AFTER": na.snapshot(), "B-AFTER-AA": na2.snapshot()}
        nb.train(xs, ys, rates)
        na.train(xs, ys, rates)
        out["trained_state_digest"] = {"before": nb.digest(), "after": na.digest()}
        assert out["trained_state_digest"]["before"] == out["trained_state_digest"]["after"], (
            "the bits moved"
        )

        def bcir(n, key):
            def run():
                n.restore(inits[key])
                s, _ = n.train(xs, ys, rates)
                return s

            return run

        ms = rotate({"B-BEFORE": bcir(nb, "B-BEFORE"), "B-AFTER": bcir(na, "B-AFTER"),
                     "B-AFTER-AA": bcir(na2, "B-AFTER-AA")}, a.steps)  # fmt: skip
    return summary(out, ms, "update")


def infer_case(model, t):
    r = replace(recipe(model), context=t, batch=4)
    rng = np.random.default_rng(20262100 + t)
    x = rng.integers(0, r.vocab, (r.batch, r.context), dtype=np.uint32)
    out = {"model": model, "mode": "infer", "batch": 4, "time": t}
    lib = LIB[model]
    with (
        before.Native(r, seed=881, threads=a.threads, library_name=lib) as nb,
        after.Native(r, seed=881, threads=a.threads, library_name=lib) as na,
        after.Native(r, seed=881, threads=a.threads, library_name=lib) as na2,
    ):
        lb, la = nb.forward(x).copy(), na.forward(x).copy()
        out["logits_equal_bytes"] = bool(lb.tobytes() == la.tobytes())
        assert out["logits_equal_bytes"], "the logits moved"

        def timed(f):
            def run():
                s = time.perf_counter()
                f()
                return time.perf_counter() - s

            return run

        ms = rotate({"B-BEFORE": timed(lambda: nb.forward(x)), "B-AFTER": timed(lambda: na.forward(x)),
                     "B-AFTER-AA": timed(lambda: na2.forward(x))}, 1)  # fmt: skip
    return summary(out, ms, "forward")


cpu = next(
    (ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")),
    "?",
)
models = a.models.split(",")
doc = {"what": __doc__, "host": {"cpu": cpu, "threads": a.threads, "python": platform.python_version(),
                                 "numpy": np.__version__},
       "rounds": a.rounds, "train": [], "infer": []}  # fmt: skip
for m in models:
    for t in (int(x) for x in a.train_t.split(",")):
        doc["train"].append(train_case(m, t))
        a.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
for m in models:
    for t in (int(x) for x in a.infer_t.split(",")):
        doc["infer"].append(infer_case(m, t))
        a.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
print("wrote", a.out)
