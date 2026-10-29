"""INFER-ATTN on Codex's trained attention models: the forward's logits from one build of the GEO
work copy, for a checkpoint's weights at a context, written to OUT.npy (a second process runs the
other build; the caller compares the bytes). usage: python logits.py WORK CKPT.npz T OUT.npy"""

import json
import sys
from dataclasses import replace
from pathlib import Path

work, ckpt, t, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
sys.path.insert(0, str(work.resolve()))
import numpy as np  # noqa: E402
import experiment as ex  # noqa: E402

z = np.load(ckpt)
recipe = replace(ex.Recipe(**json.loads(str(z["recipe"]))), context=t, batch=4)
rng = np.random.default_rng(20261022 + t)
x = rng.integers(0, recipe.vocab, (recipe.batch, recipe.context), dtype=np.uint32)
with ex.Native(
    recipe, seed=int(z["seed"]), threads=4, weights=z["weights"], library_name="attention"
) as n:
    logits = n.forward(x).copy()
np.save(out, logits)
print(out, logits.shape, float(np.abs(logits).max()))
