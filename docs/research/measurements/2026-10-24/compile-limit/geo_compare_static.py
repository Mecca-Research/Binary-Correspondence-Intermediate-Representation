"""GEO-2 / OT-05, OT-10: the PR818 study's models on BCIR's native C rail against PyTorch.

B-DIRECT: Codex's research overlays ported to current main (GEO-2b level 2: main's tensor unit and
training rail, ABI 5), driven by the bundle's own `Native` host (experiment.py), the compact cached
arena (Codex's paired-timing layout), four pool threads.
P-EAGER: the bundle's independent Torch equations (oracle.torch_model_next, Codex's reference for
every validation of the head: FP64 exactly where the native numerical contract keeps FP64 -- the
prefix, the plane norms, the precision guard, the query contraction), the same initial weights,
torch.optim.AdamW (decoupled decay 0.1, betas 0.9/0.95, eps 1e-8) after clip_grad_norm_(1.0), four
intra-op threads.
P-IND: the same module through torch.compile (Inductor, default mode), fused AdamW.

Training: one observation is `steps` updates on fixed synthetic dense targets (next token), each
arm restored to the same initial state before it (outside the clock); rounds rotate the arms'
order; B-DIRECT runs twice (an A/A control, its own instance), so every ratio is read against this
host's own noise. Inference: one forward of B x T tokens, no gradient. Compile cost (the first
compiled call) is recorded apart. The first update's loss on each route is recorded (a sanity
check of matched work: the routes are not bit-identical -- PyTorch's kernels keep their own
summation orders).

COMPILE-LIMIT (2026-10-24): geo_compare.py as GEO-2 ran it but for P-IND's compile, which now starts every case
from a fresh Dynamo state (torch._dynamo.reset()) at the case's own fixed shapes (dynamic=False). GEO-2's one
process compiled one forward code object (oracle.torch_model_next's) for all fifteen cases: Dynamo's automatic
dynamic shapes compiled every case after the first with dynamic dimensions, and past its recompile limit (8) the
ninth case on (attention at T512, and every inference case) ran P-IND eagerly. Here a recompile past the limit is
a hard failure, and each compile's shape symbols (create_symbol) and graphs are recorded and checked: none, and at
least one graph.

usage: python geo_compare_static.py WORK_DIR OUT.json [--rounds 16] [--steps 3] [--quick]
"""

import argparse
import copy
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

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # park torch's OpenMP team between its calls
os.environ.setdefault("GOMP_SPINCOUNT", "0")

p = argparse.ArgumentParser()
p.add_argument("work", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--rounds", type=int, default=16)
p.add_argument("--steps", type=int, default=3)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--quick", action="store_true")
p.add_argument("--models", default="global,query_full,attention")
p.add_argument("--train-t", default="128,256,512")
p.add_argument("--infer-t", default="1024,2048")
a = p.parse_args()
sys.path.insert(0, str(a.work.resolve()))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import experiment as ex  # noqa: E402  (the bundle's host: Native, Recipe, layout, initial)
import oracle  # noqa: E402  (the bundle's independent Torch equations)
from torch._dynamo.utils import counters  # noqa: E402

torch.set_num_threads(a.threads)
torch._dynamo.config.fail_on_recompile_limit_hit = (
    True  # an eager fallback is a failure, never a timing
)


class Symbols(logging.Handler):
    """Dynamo's create_symbol events: a static compile makes none."""

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
    """Run fn (P-IND's first call); return its seconds and the compile's witness, checked static."""
    n0, g0 = len(SYMBOLS.events), counters["stats"]["unique_graphs"]
    t0 = time.perf_counter()
    fn()
    seconds = time.perf_counter() - t0
    witness = {"symbols": SYMBOLS.events[n0:], "graphs": counters["stats"]["unique_graphs"] - g0}
    assert witness["graphs"] >= 1 and not witness["symbols"], (
        "P-IND not compiled at fixed shapes",
        witness,
    )
    return seconds, witness


def fresh():
    torch._dynamo.reset()  # a fresh Dynamo state: this case's own compile, at its own fixed shapes
    counters.clear()


def dynamo():
    return {k: dict(v) for k, v in counters.items() if k in ("frames", "stats", "graph_break")}


LIB = {
    "global": "guard_compact",
    "query_full": "query_guard_compact",
    "query_matched": "query_guard_compact",
    "attention": "attention",
}


def recipe(model):  # study.recipe, without importing the data pipeline
    if model == "attention":
        return ex.Recipe(kind="transformer", ff=256, rope=True)
    return ex.Recipe(
        kind="global",
        ff=250 if model == "query_matched" else 257,
        query_aware=model.startswith("query"),
        precision_guard=True,
    )


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


class TorchArm:
    def __init__(self, r, flat, compiled):
        self.model = oracle.torch_model_next(r, flat)
        self.params = list(self.model.parameters())
        self.opt = torch.optim.AdamW(
            self.params,
            lr=1e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            fused=compiled or None,
        )
        self.fn = torch.compile(self.model, dynamic=False) if compiled else self.model
        self.compiled = compiled
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

    def infer(self, x):
        with torch.no_grad():
            return self.fn(x)


def train_case(model, t):
    fresh()
    r = replace(recipe(model), context=t, batch=4)
    rng = np.random.default_rng(81820261018 + t + r.ff)
    xs = rng.integers(0, r.vocab, (a.steps, r.batch, r.context), dtype=np.uint32)
    ys = np.roll(xs, -1, axis=2).copy()
    rates = np.full(a.steps, 1e-4, np.float64)
    flat = ex.initial(r, 881)
    tx = torch.from_numpy(xs.astype(np.int64))
    ty = torch.from_numpy(ys.astype(np.int64))
    out = {"model": model, "batch": 4, "time": t, "parameters": sum(n for *_, n in ex.layout(r))}
    with (
        ex.Native(r, seed=881, threads=a.threads, library_name=LIB[model]) as b1,
        ex.Native(r, seed=881, threads=a.threads, library_name=LIB[model]) as b2,
    ):
        init1, init2 = b1.snapshot(), b2.snapshot()
        loss0 = b1.backward(xs[0], ys[0])  # the first batch's loss on the native rail
        b1.restore(init1)
        eager, comp = TorchArm(r, flat, False), TorchArm(r, flat, True)
        out["first_loss"] = {
            "B-DIRECT": loss0,
            "P-EAGER": float(
                F.cross_entropy(eager.model(tx[0]).reshape(-1, r.vocab), ty[0].reshape(-1))
            ),
        }
        # compile: forward, backward, fused optimizer
        out["compile_first_update_s"], out["compile_witness"] = compiling(
            lambda: comp.update(tx[0], ty[0])
        )
        eager.update(tx[0], ty[0])
        snaps = {"P-EAGER": eager.snapshot(), "P-IND": comp.snapshot()}

        def bcir(n, init):
            def run():
                n.restore(init)
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

        arms = {
            "B-DIRECT": bcir(b1, init1),
            "B-DIRECT-AA": bcir(b2, init2),
            "P-EAGER": torch_run(eager, "P-EAGER"),
            "P-IND": torch_run(comp, "P-IND"),
        }
        for f in arms.values():
            f()  # warm
        ms = {k: [] for k in arms}
        names = list(arms)
        for q in range(a.rounds):
            for j in range(len(names)):
                k = names[(q + j) % len(names)]
                ms[k].append(1000 * arms[k]() / a.steps)
    out["dynamo"] = dynamo()
    out["ms_per_update"] = {k: statistics.median(v) for k, v in ms.items()}
    out["bcir_over"] = {
        k: interval([b / o for b, o in zip(ms["B-DIRECT"], ms[k])])
        for k in ("B-DIRECT-AA", "P-EAGER", "P-IND")
    }
    out["raw_ms"] = {k: [round(x, 4) for x in v] for k, v in ms.items()}
    print(json.dumps({k: out[k] for k in ("model", "time", "ms_per_update")}), flush=True)
    return out


def infer_case(model, t):
    fresh()
    r = replace(recipe(model), context=t, batch=4)
    rng = np.random.default_rng(20262100 + t)
    x = rng.integers(0, r.vocab, (r.batch, r.context), dtype=np.uint32)
    tx = torch.from_numpy(x.astype(np.int64))
    flat = ex.initial(r, 881)
    out = {"model": model, "batch": 4, "time": t}
    with (
        ex.Native(r, seed=881, threads=a.threads, library_name=LIB[model]) as b1,
        ex.Native(r, seed=881, threads=a.threads, library_name=LIB[model]) as b2,
    ):
        eager, comp = TorchArm(r, flat, False), TorchArm(r, flat, True)
        logits = b1.forward(x).copy()
        ref = eager.infer(tx).numpy()
        out["logits_max_abs_delta_vs_eager"] = float(np.max(np.abs(logits - ref)))
        out["bcir_arena_bytes"] = int(4 * (b1.plan.activations + b1.plan.scratch))
        out["compile_first_call_s"], out["compile_witness"] = compiling(lambda: comp.infer(tx))

        def timed(f):
            def run():
                s = time.perf_counter()
                f()
                return time.perf_counter() - s

            return run

        arms = {
            "B-DIRECT": timed(lambda: b1.forward(x)),
            "B-DIRECT-AA": timed(lambda: b2.forward(x)),
            "P-EAGER": timed(lambda: eager.infer(tx)),
            "P-IND": timed(lambda: comp.infer(tx)),
        }
        for f in arms.values():
            f()
        ms = {k: [] for k in arms}
        names = list(arms)
        for q in range(a.rounds):
            for j in range(len(names)):
                k = names[(q + j) % len(names)]
                ms[k].append(1000 * arms[k]())
    out["dynamo"] = dynamo()
    out["ms_per_forward"] = {k: statistics.median(v) for k, v in ms.items()}
    out["bcir_over"] = {
        k: interval([b / o for b, o in zip(ms["B-DIRECT"], ms[k])])
        for k in ("B-DIRECT-AA", "P-EAGER", "P-IND")
    }
    out["raw_ms"] = {k: [round(x, 4) for x in v] for k, v in ms.items()}
    print(json.dumps({k: out[k] for k in ("model", "time", "ms_per_forward")}), flush=True)
    return out


models = a.models.split(",")
train_t = [int(x) for x in a.train_t.split(",")] if not a.quick else [64]
infer_t = [int(x) for x in a.infer_t.split(",")] if not a.quick else [256]
cpu = next(
    (ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")),
    "?",
)
doc = {
    "host": {
        "cpu": cpu,
        "threads": a.threads,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
    },
    "work": str(a.work),
    "rounds": a.rounds,
    "steps_per_observation": a.steps,
    "train": [train_case(m, t) for m in models for t in train_t],
    "infer": [infer_case(m, t) for m in models for t in infer_t],
}
a.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
print("wrote", a.out)
