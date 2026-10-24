"""GEO-AMX against PyTorch: the PR818 study's models trained on bfloat16 on this AMX host, BCIR and PyTorch.

B-FP32: the study's binary32 contract (14) on the native C rail (build_amx.py's builds: GEO-2b's level-2 overlays
with this branch's tensor unit), the bundle's own `Native` host, four pool threads.
B-BF16: the study's bf16 arm (contract 26: the GEMMs on bfloat16 operands, through the AMX instance here).
B-BF16-ATTN: contract 90 (26 + the bf16 attention, BF16-ATTN), the attention model only.
P-EAGER-BF16: the bundle's independent Torch equations (oracle.torch_model_next) under torch.autocast to bfloat16
around the forward and the loss -- weights, gradients, moments and AdamW in FP32 (decoupled decay 0.1, betas
0.9/0.95, eps 1e-8, after clip_grad_norm_(1.0)) -- four intra-op threads; PyTorch's CPU mixed precision, whose
bf16 matmuls run oneDNN's AMX kernels on this host.
P-IND-BF16: the same module through torch.compile (Inductor, default mode) under the same autocast, fused AdamW.
P-IND: torch.compile in FP32 (GEO-2's P-IND), for reference.
One observation is `steps` updates on fixed synthetic dense targets, each arm restored to the same initial state
outside the clock; rounds rotate the arms' order; the first update (a compiled arm's compile: forward, backward)
is recorded apart, with the first update's loss on each route (a check of matched work: the routes are not
bit-identical). Every case compiles from a fresh Dynamo state (torch._dynamo.reset()) at its own fixed shapes
(dynamic=False): one process and one forward code object for every case would otherwise reach Dynamo's recompile
limit (8) and run the later cases' compiled arms eagerly, and compile them past the first case with the time
dimension dynamic. A recompile past the limit is a hard failure; each compile's shape symbols (create_symbol) and
graphs are recorded and checked: none, and at least one graph.
usage: python geo_bf16_compare.py WORK OUT.json [--rounds 16] [--steps 3] [--cases attention:256,...]
"""

import argparse
import copy
import json
import logging
import math
import os
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
p.add_argument("--rounds", type=int, default=16)
p.add_argument("--steps", type=int, default=3)
p.add_argument("--threads", type=int, default=4)
p.add_argument(
    "--cases",
    default="attention:256,attention:512,attention:1024,global:256,global:512,query_full:256",
)
a = p.parse_args()
sys.path.insert(0, str(a.work.resolve()))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402
import experiment as ex  # noqa: E402
import oracle  # noqa: E402

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
    """Run fn (an arm's first update); return its value, its seconds and the compile's witness."""
    n0, g0 = len(SYMBOLS.events), counters["stats"]["unique_graphs"]
    t0 = time.perf_counter()
    value = fn()
    seconds = time.perf_counter() - t0
    return (
        value,
        seconds,
        {"symbols": SYMBOLS.events[n0:], "graphs": counters["stats"]["unique_graphs"] - g0},
    )


LIB = {"global": "guard_compact", "query_full": "query_guard_compact", "attention": "attention"}
T95 = {7: 2.365, 11: 2.201, 15: 2.131, 23: 2.069}


def recipe(model):
    if model == "attention":
        return ex.Recipe(kind="transformer", ff=256, rope=True)
    return ex.Recipe(
        kind="global", ff=257, query_aware=model.startswith("query"), precision_guard=True
    )


def interval(ratios):
    logs = [math.log(x) for x in ratios]
    n = len(logs)
    t = T95.get(n - 1, 1.96 + 2.4 / (n - 1))
    m, h = statistics.fmean(logs), t * statistics.stdev(logs) / math.sqrt(n)
    return {"geometric_mean": math.exp(m), "ci95": [math.exp(m - h), math.exp(m + h)], "rounds": n}


class TorchArm:
    def __init__(self, r, flat, compiled, bf16):
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
        self.bf16, self.v = bf16, r.vocab

    def loss(self, x, y):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.bf16):
            logits = self.fn(x)
            return F.cross_entropy(logits.reshape(-1, self.v).float(), y.reshape(-1))

    def update(self, x, y):
        loss = self.loss(x, y)
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


def case(model, t):
    torch._dynamo.reset()  # a fresh Dynamo state: this case's own compiles, at its own fixed shapes
    counters.clear()
    r = replace(recipe(model), context=t, batch=4)
    rng = np.random.default_rng(81820261018 + t + r.ff)
    xs = rng.integers(0, r.vocab, (a.steps, r.batch, r.context), dtype=np.uint32)
    ys = np.roll(xs, -1, axis=2).copy()
    rates = np.full(a.steps, 1e-4, np.float64)
    flat = ex.initial(r, 881)
    tx, ty = torch.from_numpy(xs.astype(np.int64)), torch.from_numpy(ys.astype(np.int64))
    out = {"model": model, "batch": 4, "time": t, "parameters": sum(n for *_, n in ex.layout(r))}
    contracts = {"B-FP32": 14, "B-BF16": 26}
    if model == "attention":
        contracts["B-BF16-ATTN"] = 90
    natives = {
        k: ex.Native(replace(r, contract=c), seed=881, threads=a.threads, library_name=LIB[model])
        for k, c in contracts.items()
    }
    try:
        inits = {k: n.snapshot() for k, n in natives.items()}
        first = {}
        for k, n in natives.items():
            first[k] = n.backward(xs[0], ys[0])
            n.restore(inits[k])
        torches = {
            "P-EAGER-BF16": TorchArm(r, flat, False, True),
            "P-IND-BF16": TorchArm(r, flat, True, True),
            "P-IND": TorchArm(r, flat, True, False),
        }
        first_s, witness = {}, {}
        for (
            k,
            arm,
        ) in torches.items():  # the first update: a compiled arm's compile, forward and backward
            first[k], first_s[k], w = compiling(lambda: arm.update(tx[0], ty[0]))  # noqa: B023 (called here)
            if k != "P-EAGER-BF16":
                assert w["graphs"] >= 1 and not w["symbols"], (k, "not compiled at fixed shapes", w)
                witness[k] = w
        snaps = {k: arm.snapshot() for k, arm in torches.items()}

        def bcir(k):
            def run():
                natives[k].restore(inits[k])
                s, _ = natives[k].train(xs, ys, rates)
                return s

            return run

        def torch_run(k):
            def run():
                torches[k].restore(snaps[k])
                start = time.perf_counter()
                for i in range(a.steps):
                    torches[k].update(tx[i], ty[i])
                return time.perf_counter() - start

            return run

        arms = {**{k: bcir(k) for k in natives}, **{k: torch_run(k) for k in torches}}
        for f in arms.values():
            f()  # warm
        ms = {k: [] for k in arms}
        names = list(arms)
        for q in range(a.rounds):
            for j in range(len(names)):
                k = names[(q + j) % len(names)]
                ms[k].append(1000 * arms[k]() / a.steps)
    finally:
        for n in natives.values():
            n.close()
    best = "B-BF16-ATTN" if "B-BF16-ATTN" in ms else "B-BF16"
    out.update(
        {
            "contracts": contracts,
            "first_update_loss": first,
            "first_update_s": first_s,
            "compile_witness": witness,
            "dynamo": {
                k: dict(v) for k, v in counters.items() if k in ("frames", "stats", "graph_break")
            },
            "ms_per_update": {k: statistics.median(v) for k, v in ms.items()},
            "b_bf16_over": {
                k: interval([b / o for b, o in zip(ms["B-BF16"], ms[k])])
                for k in ms
                if k != "B-BF16"
            },
            "best_bcir": best,
            "best_bcir_over": {
                k: interval([b / o for b, o in zip(ms[best], ms[k])]) for k in ms if k != best
            },
            "raw_ms": {k: [round(x, 4) for x in v] for k, v in ms.items()},
        }
    )
    print(
        json.dumps(
            {
                "case": f"{model}/B4T{t}",
                "ms": {k: round(v, 2) for k, v in out["ms_per_update"].items()},
                "best_bcir_over": {
                    k: round(v["geometric_mean"], 3) for k, v in out["best_bcir_over"].items()
                },
            }
        ),
        flush=True,
    )
    return out


cases = []
for spec in a.cases.split(","):
    model, t = spec.split(":")
    cases.append(case(model, int(t)))
    a.out.write_text(
        json.dumps({"what": __doc__, "torch": torch.__version__, "cases": cases}, indent=1) + "\n"
    )
