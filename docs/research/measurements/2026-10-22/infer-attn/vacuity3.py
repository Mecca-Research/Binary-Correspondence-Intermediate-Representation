"""Level 3's vacuity: after one forward from a zeroed arena, the floats of the activation arena
each build wrote -- the stored build every layer's t x t probabilities, the scratch build only its
slots -- with the logits equal. usage: python vacuity3.py BEFORE_WORK AFTER_WORK T OUT.json"""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

before_w, after_w, t, out = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    int(sys.argv[3]),
    Path(sys.argv[4]),
)


def load(work, name):
    sys.path.insert(0, str(work.resolve()))
    spec = importlib.util.spec_from_file_location(name, work / "experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


doc = {"time": t}
logits = {}
for side, work in (("before", before_w), ("after", after_w)):
    ex = load(work, f"exp_{side}")
    r = replace(ex.Recipe(kind="transformer", ff=256, rope=True), context=t, batch=4)
    x = np.random.default_rng(7 + t).integers(0, r.vocab, (r.batch, r.context), dtype=np.uint32)
    with ex.Native(r, seed=881, threads=4, library_name="attention") as n:
        logits[side] = n.forward(x).copy()
        act = n.buffers["activations"]
        doc[side] = {
            "activation_floats": int(act.size),
            "nonzero_floats": int(np.count_nonzero(act)),
        }
doc["probabilities_floats_per_layer"] = 4 * 4 * t * t
doc["logits_equal_bytes"] = bool(logits["before"].tobytes() == logits["after"].tobytes())
doc["after_writes_fewer"] = doc["after"]["nonzero_floats"] < doc["before"]["nonzero_floats"]
assert doc["logits_equal_bytes"] and doc["after_writes_fewer"], doc
out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
print(json.dumps(doc))
