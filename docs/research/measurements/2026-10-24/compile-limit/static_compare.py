"""COMPILE-LIMIT on the GEO study's attention model at level 3 (a build of this branch's tensor unit): its training
update (forward, backward, clipping, AdamW: geo_compare.py's train_case) or its inference forward, 4 x T tokens,
against PyTorch with torch.compile at each case's own fixed shapes.

B (twice: an A/A control): the level-3 build in WORK. P-EAGER: the bundle's Torch equations
(oracle.torch_model_next). P-IND: torch.compile (Inductor, default mode; fused AdamW) from a fresh Dynamo state
(torch._dynamo.reset() at every case) at the case's shapes (dynamic=False). P-IND-DYN: the same module compiled
with the tokens' time dimension dynamic (torch._dynamo.maybe_mark_dynamic(ids, 1)) -- what train_compare.py and
infer_compare.py compiled for every case after their first: one process, one forward code object, so Dynamo's
automatic dynamic shapes marked L['ids'] dim 1 dynamic at the first recompile. Every arm its own module (its own
class), so each compiled arm is its own cache entry; a recompile past Dynamo's limit is a hard failure, not an
eager fallback; each compile's shape symbols (create_symbol) and graphs are recorded and checked: P-IND none and at
least one graph, P-IND-DYN L['ids'] dim 1. Rounds rotate the arms' order; each training arm is restored to the same
initial state before each observation, outside the clock.
usage: python static_compare.py WORK OUT.json [--mode train|infer] [--rounds 24] [--steps 3] [--threads 4]
       [--times 512,1024,2048]
"""

import argparse
import copy
import importlib.util
import json
import logging
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
p.add_argument("work", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--mode", choices=("train", "infer"), default="train")
p.add_argument("--rounds", type=int, default=24)
p.add_argument("--steps", type=int, default=3)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--times", default="512,1024,2048")
a = p.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402


def load(work: Path, name: str):
    """The work copy's experiment module under its own name: its ROOT, so its build/, is its own."""
    sys.path.insert(0, str(work.resolve()))
    spec = importlib.util.spec_from_file_location(name, work / "experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


ex = load(a.work, "exp_work")
sys.path.insert(0, str(a.work.resolve()))
import oracle  # noqa: E402  (the bundle's Torch equations)

torch.set_num_threads(a.threads)
torch._dynamo.config.fail_on_recompile_limit_hit = (
    True  # an eager fallback is a failure, never a timing
)


class Symbols(logging.Handler):
    """Dynamo's create_symbol events: a static compile makes none; a dynamic one names its dimension."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.events = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("create_symbol"):
            self.events.append(message.split(" [")[0])


SYMBOLS = Symbols()
_shapes_log = logging.getLogger("torch.fx.experimental.symbolic_shapes")
_shapes_log.setLevel(logging.INFO)
_shapes_log.addHandler(SYMBOLS)


def compiling(fn):
    """Run fn (a compiled arm's first call); return its seconds and the compile's witness."""
    n0, g0 = len(SYMBOLS.events), counters["stats"]["unique_graphs"]
    t0 = time.perf_counter()
    value = fn()
    seconds = time.perf_counter() - t0
    witness = {"symbols": SYMBOLS.events[n0:], "graphs": counters["stats"]["unique_graphs"] - g0}
    return value, seconds, witness


def check(witness, dynamic):
    assert witness["graphs"] >= 1, ("no graph compiled", witness)
    if dynamic:
        assert any("L['ids'].size()[1]" in s for s in witness["symbols"]), (
            "not dynamic in time",
            witness,
        )
    else:
        assert not witness["symbols"], ("a static compile made symbols", witness)


def interval(ratios):
    logs = [math.log(r) for r in ratios]
    m = statistics.fmean(logs)
    t = {7: 2.365, 11: 2.201, 15: 2.131, 23: 2.069, 31: 2.040}.get(len(logs) - 1, 2.0)
    h = t * statistics.stdev(logs) / math.sqrt(len(logs))
    return {
        "geometric_mean": math.exp(m),
        "ci95": [math.exp(m - h), math.exp(m + h)],
        "rounds": len(logs),
    }


def compiled(model, dynamic):
    return torch.compile(model, dynamic=None) if dynamic else torch.compile(model, dynamic=False)


def marked(t, dynamic):
    t = t.clone()
    if dynamic:
        torch._dynamo.maybe_mark_dynamic(t, 1)
    return t


class TorchArm:  # geo_compare.py's, compiled at fixed shapes (or with the time dimension dynamic)
    def __init__(self, r, flat, compile_mode):
        self.model = oracle.torch_model_next(r, flat)
        self.params = list(self.model.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                                     fused=compile_mode is not None or None)  # fmt: skip
        self.fn = (
            self.model if compile_mode is None else compiled(self.model, compile_mode == "dynamic")
        )
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


def train_case(t):
    torch._dynamo.reset()
    counters.clear()
    r = replace(ex.Recipe(kind="transformer", ff=256, rope=True), context=t, batch=4)
    rng = np.random.default_rng(81820261018 + t + r.ff)  # geo_compare.py's training draws
    xs = rng.integers(0, r.vocab, (a.steps, r.batch, r.context), dtype=np.uint32)
    ys = np.roll(xs, -1, axis=2).copy()
    rates = np.full(a.steps, 1e-4, np.float64)
    flat = ex.initial(r, 881)
    tx, ty = torch.from_numpy(xs.astype(np.int64)), torch.from_numpy(ys.astype(np.int64))
    out = {
        "model": "attention",
        "mode": "train",
        "batch": 4,
        "time": t,
        "steps_per_observation": a.steps,
    }
    with (
        ex.Native(r, seed=881, threads=a.threads, library_name="attention") as b1,
        ex.Native(r, seed=881, threads=a.threads, library_name="attention") as b2,
    ):
        inits = {"B": b1.snapshot(), "B-AA": b2.snapshot()}
        first = {"B": b1.backward(xs[0], ys[0])}
        b1.restore(inits["B"])
        torches = {"P-EAGER": TorchArm(r, flat, None), "P-IND": TorchArm(r, flat, "static"),
                   "P-IND-DYN": TorchArm(r, flat, "dynamic")}  # fmt: skip
        inputs = {
            k: ([marked(tx[i], k == "P-IND-DYN") for i in range(a.steps)], ty) for k in torches
        }
        out["first_update_s"], out["compile_witness"] = {}, {}
        for k, arm in torches.items():
            xi, yi = inputs[k]
            first[k], out["first_update_s"][k], witness = compiling(
                lambda: arm.update(xi[0], yi[0])  # noqa: B023 (called here)
            )
            if k != "P-EAGER":
                check(witness, k == "P-IND-DYN")
                out["compile_witness"][k] = witness
        snaps = {k: arm.snapshot() for k, arm in torches.items()}
        for k, arm in torches.items():
            arm.restore(snaps[k])

        def bcir(n, key):
            def run():
                n.restore(inits[key])
                s, _ = n.train(xs, ys, rates)
                return s

            return run

        def torch_run(k):
            arm, (xi, yi) = torches[k], inputs[k]

            def run():
                arm.restore(snaps[k])
                start = time.perf_counter()
                for i in range(a.steps):
                    arm.update(xi[i], yi[i])
                return time.perf_counter() - start

            return run

        arms = {"B": bcir(b1, "B"), "B-AA": bcir(b2, "B-AA"), **{k: torch_run(k) for k in torches}}
        ms = rotate(arms, a.steps)
        out["deterministic_b"] = b1.digest() == b2.digest()
    out["first_update_loss"] = first
    out["dynamo"] = {
        k: dict(v) for k, v in counters.items() if k in ("frames", "stats", "graph_break")
    }
    out["ms_per_update"] = {k: statistics.median(v) for k, v in ms.items()}
    out["b_over"] = {k: interval([x / y for x, y in zip(ms["B"], ms[k])]) for k in ms if k != "B"}
    out["ind_dyn_over_ind"] = interval([x / y for x, y in zip(ms["P-IND-DYN"], ms["P-IND"])])
    out["raw_ms"] = {k: [round(v, 4) for v in vs] for k, vs in ms.items()}
    return out


def infer_case(t):
    torch._dynamo.reset()
    counters.clear()
    r = replace(ex.Recipe(kind="transformer", ff=256, rope=True), context=t, batch=4)
    rng = np.random.default_rng(20262100 + t)  # geo_compare.py's inference draws
    x = rng.integers(0, r.vocab, (r.batch, r.context), dtype=np.uint32)
    tx = torch.from_numpy(x.astype(np.int64))
    flat = ex.initial(r, 881)
    out = {"model": "attention", "mode": "infer", "batch": 4, "time": t}
    with (
        ex.Native(r, seed=881, threads=a.threads, library_name="attention") as b1,
        ex.Native(r, seed=881, threads=a.threads, library_name="attention") as b2,
    ):
        models = {"P-EAGER": oracle.torch_model_next(r, flat), "P-IND": oracle.torch_model_next(r, flat),
                  "P-IND-DYN": oracle.torch_model_next(r, flat)}  # fmt: skip
        fns = {"P-EAGER": models["P-EAGER"], "P-IND": compiled(models["P-IND"], False),
               "P-IND-DYN": compiled(models["P-IND-DYN"], True)}  # fmt: skip
        inputs = {k: marked(tx, k == "P-IND-DYN") for k in fns}
        logits = b1.forward(x).copy()
        out["first_call_s"], out["compile_witness"], delta = {}, {}, {}
        for k, fn in fns.items():
            with torch.no_grad():
                ref, out["first_call_s"][k], witness = compiling(lambda: fn(inputs[k]).numpy())  # noqa: B023 (called here)
            delta[k] = float(np.max(np.abs(logits - ref)))
            if k != "P-EAGER":
                check(witness, k == "P-IND-DYN")
                out["compile_witness"][k] = witness
        out["logits_max_abs_delta_vs_b"] = delta

        def timed(f):
            def run():
                s = time.perf_counter()
                f()
                return time.perf_counter() - s

            return run

        def tinfer(k):
            def run():
                with torch.no_grad():
                    fns[k](inputs[k])

            return run

        arms = {"B": timed(lambda: b1.forward(x)), "B-AA": timed(lambda: b2.forward(x)),
                **{k: timed(tinfer(k)) for k in fns}}  # fmt: skip
        ms = rotate(arms, 1)
    out["dynamo"] = {
        k: dict(v) for k, v in counters.items() if k in ("frames", "stats", "graph_break")
    }
    out["ms_per_forward"] = {k: statistics.median(v) for k, v in ms.items()}
    out["b_over"] = {k: interval([x / y for x, y in zip(ms["B"], ms[k])]) for k in ms if k != "B"}
    out["ind_dyn_over_ind"] = interval([x / y for x, y in zip(ms["P-IND-DYN"], ms["P-IND"])])
    out["raw_ms"] = {k: [round(v, 4) for v in vs] for k, vs in ms.items()}
    return out


cpu = next(
    (ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")),
    "?",
)
doc = {"what": __doc__, "host": {"cpu": cpu, "threads": a.threads, "python": platform.python_version(),
                                 "torch": str(torch.__version__), "numpy": np.__version__},
       "mode": a.mode, "rounds": a.rounds, "cases": []}  # fmt: skip
for t in (int(x) for x in a.times.split(",")):
    c = (train_case if a.mode == "train" else infer_case)(t)
    doc["cases"].append(c)
    m = c.get("ms_per_update") or c.get("ms_per_forward")
    print(json.dumps({"mode": a.mode, "time": t, "ms": {k: round(v, 2) for k, v in m.items()},
                      "b_over": {k: round(v["geometric_mean"], 3) for k, v in c["b_over"].items()},
                      "dyn_over_ind": round(c["ind_dyn_over_ind"]["geometric_mean"], 3),
                      "witness": c["compile_witness"]}), flush=True)  # fmt: skip
    a.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
print("wrote", a.out)
