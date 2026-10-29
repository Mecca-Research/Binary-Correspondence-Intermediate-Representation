"""COMPILE-LIMIT's witness: the flaw and the fix, on the bundle's attention model (oracle.torch_model_next) under
torch.no_grad, in one process the way geo_compare.py, train_compare.py and infer_compare.py ran their cases.

1. automatic dynamic shapes: the forward compiled at T128, then a second module (a new class: one forward code
   object for both) at T256 -- the second compile creates a symbol for L['ids'] dim 1, the tokens' time.
2. the recompile limit: more modules at more lengths, each a recompile of the one code object; past Dynamo's limit
   (8) a call compiles nothing and runs eagerly, warning once -- and the code object stays eager. From a fresh state
   with fail_on_recompile_limit_hit set, the same ninth call raises.
3. the fix: torch._dynamo.reset() before a case and dynamic=False -- its compile creates no symbol.
Each step records the call's shape symbols (create_symbol), the graphs it compiled and Dynamo's frame counts.
usage: python witness.py WORK OUT.json
"""

import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402
import experiment as ex  # noqa: E402
import oracle  # noqa: E402

torch.set_num_threads(4)


class Symbols(logging.Handler):
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


def call(t, dynamic=None):
    r = replace(ex.Recipe(kind="transformer", ff=256, rope=True), context=t, batch=4)
    model = oracle.torch_model_next(r, ex.initial(r, 881))
    fn = torch.compile(model, dynamic=dynamic)
    x = torch.from_numpy(np.random.default_rng(t).integers(0, r.vocab, (4, t)).astype(np.int64))
    n0, g0 = len(SYMBOLS.events), counters["stats"]["unique_graphs"]
    step = {"time": t, "dynamic_arg": dynamic}
    try:
        with torch.no_grad():
            fn(x)
        step["raised"] = None
    except Exception as e:  # the hard failure under fail_on_recompile_limit_hit
        step["raised"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    step["symbols"] = SYMBOLS.events[n0:]
    step["graphs"] = counters["stats"]["unique_graphs"] - g0
    step["frames"] = dict(counters["frames"])
    return step


out = {"what": __doc__, "torch": torch.__version__, "recompile_limit": torch._dynamo.config.recompile_limit,
       "automatic_dynamic_shapes": torch._dynamo.config.automatic_dynamic_shapes}  # fmt: skip
torch._dynamo.reset()
counters.clear()
out["one_process"] = [call(128 + 16 * i) for i in range(10)]
torch._dynamo.reset()
counters.clear()
torch._dynamo.config.fail_on_recompile_limit_hit = True
hard = [call(128 + 16 * i) for i in range(9)]
torch._dynamo.config.fail_on_recompile_limit_hit = False
out["with_fail_on_recompile_limit_hit"] = hard
out["past_the_limit_with_fail_on_recompile_limit_hit"] = hard[-1]
out["fixed"] = []
for t in (128, 256):
    torch._dynamo.reset()
    counters.clear()
    out["fixed"].append(call(t, dynamic=False))
steps = out["one_process"]
assert steps[0]["graphs"] == 1 and not steps[0]["symbols"], "the first compile is static"
assert any("L['ids'].size()[1]" in s for s in steps[1]["symbols"]), (
    "the second compile is dynamic in time"
)
eager = [s["time"] for s in steps if s["graphs"] == 0 and s["raised"] is None]
assert eager, "the limit was not reached"
out["compiled_cases"] = [s["time"] for s in steps if s["graphs"] >= 1]
out["eager_cases"] = eager
assert all(s["raised"] is None for s in hard[:8]), "a compile under the limit raised"
assert out["past_the_limit_with_fail_on_recompile_limit_hit"]["raised"], (
    "the hard failure did not fire"
)
assert all(s["graphs"] == 1 and not s["symbols"] for s in out["fixed"]), "the fix is not static"
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
summary = {"compiled": out["compiled_cases"], "eager": eager, "second_symbols": steps[1]["symbols"],
           "raised": out["past_the_limit_with_fail_on_recompile_limit_hit"]["raised"],
           "fixed_symbols": [s["symbols"] for s in out["fixed"]]}  # fmt: skip
print(json.dumps(summary))
