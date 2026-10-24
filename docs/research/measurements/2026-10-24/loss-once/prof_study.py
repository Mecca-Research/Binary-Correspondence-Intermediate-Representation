"""Runs one of the PR818 study's models from a work copy for a profile: COUNT observations of training (3 updates
each, geo_ab.py's draws) or of inference (one forward of 4 x T tokens). usage: prof_study.py WORK MODEL MODE T COUNT
[THREADS]"""

import importlib.util
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
work, model, mode, t, count = (
    Path(sys.argv[1]),
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
    int(sys.argv[5]),
)
threads = int(sys.argv[6]) if len(sys.argv) > 6 else 4
import numpy as np  # noqa: E402

sys.path.insert(0, str(work.resolve()))
spec = importlib.util.spec_from_file_location("exp_work", work / "experiment.py")
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
LIB = {"global": "guard_compact", "query_full": "query_guard_compact", "attention": "attention"}
if model == "attention":
    r = exp.Recipe(kind="transformer", ff=256, rope=True)
else:
    r = exp.Recipe(
        kind="global", ff=257, query_aware=model.startswith("query"), precision_guard=True
    )
r = replace(r, context=t, batch=4)
with exp.Native(r, seed=881, threads=threads, library_name=LIB[model]) as n:
    if mode == "train":
        rng = np.random.default_rng(81820261018 + t + r.ff)
        xs = rng.integers(0, r.vocab, (3, r.batch, r.context), dtype=np.uint32)
        ys = np.roll(xs, -1, axis=2).copy()
        rates = np.full(3, 1e-4, np.float64)
        init = n.snapshot()
        n.train(xs, ys, rates)
        s = time.perf_counter()
        for _ in range(count):
            n.restore(init)
            n.train(xs, ys, rates)
        el = time.perf_counter() - s
        print(f"{model} train T{t}: {1000 * el / (3 * count):.2f} ms an update (restore included)")
    else:
        rng = np.random.default_rng(20262100 + t)
        x = rng.integers(0, r.vocab, (r.batch, r.context), dtype=np.uint32)
        n.forward(x)
        s = time.perf_counter()
        for _ in range(count):
            n.forward(x)
        print(f"{model} infer T{t}: {1000 * (time.perf_counter() - s) / count:.2f} ms a forward")
