"""BWD-WIDTH on the GEO study's attention model: its training update (forward, backward, clipping,
AdamW: geo_compare.py's train_case) of 4 x T tokens on the level-3 build before the change
(B-BEFORE), after it (B-AFTER, twice: an A/A control), the bundle's Torch equations (P-EAGER) and
torch.compile (P-IND), rounds in rotated order; each arm restored to the same initial state before
each observation, outside the clock; the two builds' trained states (weights, both moments, the step)
held equal by digest first. usage: python train_compare.py BEFORE_WORK AFTER_WORK OUT.json
[--rounds 12] [--steps 3] [--threads 4] [--times 512,1024,2048]
"""

import argparse
import copy
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
p.add_argument("--steps", type=int, default=3)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--times", default="512,1024,2048")
a = p.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


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


class TorchArm:  # geo_compare.py's
    def __init__(self, r, flat, compiled):
        self.model = oracle.torch_model_next(r, flat)
        self.params = list(self.model.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                                     fused=compiled or None)  # fmt: skip
        self.fn = torch.compile(self.model) if compiled else self.model
        self.v = r.vocab

    def update(self, x, y):
        logits = self.fn(x)
        loss = F.cross_entropy(logits.reshape(-1, self.v), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return float(loss)

    def snapshot(self):
        return [q.detach().clone() for q in self.params], copy.deepcopy(self.opt.state_dict())

    def restore(self, snap):
        with torch.no_grad():
            for q, q0 in zip(self.params, snap[0]):
                q.copy_(q0)
        self.opt.load_state_dict(snap[1])


def case(t):
    r = replace(before.Recipe(kind="transformer", ff=256, rope=True), context=t, batch=4)
    rng = np.random.default_rng(81820261018 + t + r.ff)  # geo_compare.py's training draws
    xs = rng.integers(0, r.vocab, (a.steps, r.batch, r.context), dtype=np.uint32)
    ys = np.roll(xs, -1, axis=2).copy()
    rates = np.full(a.steps, 1e-4, np.float64)
    flat = before.initial(r, 881)
    tx, ty = torch.from_numpy(xs.astype(np.int64)), torch.from_numpy(ys.astype(np.int64))
    out = {"model": "attention", "batch": 4, "time": t, "steps_per_observation": a.steps}
    with (
        before.Native(r, seed=881, threads=a.threads, library_name="attention") as nb,
        after.Native(r, seed=881, threads=a.threads, library_name="attention") as na,
        after.Native(r, seed=881, threads=a.threads, library_name="attention") as na2,
    ):
        inits = {"B-BEFORE": nb.snapshot(), "B-AFTER": na.snapshot(), "B-AFTER-AA": na2.snapshot()}
        nb.train(xs, ys, rates)
        na.train(xs, ys, rates)
        out["trained_state_digest"] = {"before": nb.digest(), "after": na.digest()}
        assert out["trained_state_digest"]["before"] == out["trained_state_digest"]["after"], (
            "the bits moved"
        )
        eager, comp = TorchArm(r, flat, False), TorchArm(r, flat, True)
        t0 = time.perf_counter()
        comp.update(tx[0], ty[0])
        out["compile_first_update_s"] = time.perf_counter() - t0
        eager.update(tx[0], ty[0])
        snaps = {"P-EAGER": eager.snapshot(), "P-IND": comp.snapshot()}

        def bcir(n, key):
            def run():
                n.restore(inits[key])
                s, _ = n.train(xs, ys, rates)
                return s

            return run

        def torch_run(arm, key):
            def run():
                arm.restore(snaps[key])
                start = time.perf_counter()
                for i in range(a.steps):
                    arm.update(tx[i], ty[i])
                return time.perf_counter() - start

            return run

        arms = {"B-BEFORE": bcir(nb, "B-BEFORE"), "B-AFTER": bcir(na, "B-AFTER"),
                "B-AFTER-AA": bcir(na2, "B-AFTER-AA"), "P-EAGER": torch_run(eager, "P-EAGER"),
                "P-IND": torch_run(comp, "P-IND")}  # fmt: skip
        for f in arms.values():
            f()  # warm
        ms = {k: [] for k in arms}
        names = list(arms)
        for q in range(a.rounds):
            for j in range(len(names)):
                k = names[(q + j) % len(names)]
                ms[k].append(1000 * arms[k]() / a.steps)
    out["ms_per_update"] = {k: statistics.median(v) for k, v in ms.items()}
    out["after_over"] = {k: interval([x / y for x, y in zip(ms["B-AFTER"], ms[k])])
                         for k in ("B-BEFORE", "B-AFTER-AA", "P-EAGER", "P-IND")}  # fmt: skip
    out["raw_ms"] = {k: [round(v, 4) for v in vs] for k, vs in ms.items()}
    print(json.dumps({k: out[k] for k in ("time", "ms_per_update")}), flush=True)
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
