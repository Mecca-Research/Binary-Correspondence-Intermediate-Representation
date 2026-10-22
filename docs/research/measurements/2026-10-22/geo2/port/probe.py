"""GEO-2 probe: a few updates of every head under both contracts in one work copy, through the
study's own Runner/BudgetPolicy, printed as JSON (state digest after each update, losses, norms,
the policy's choices) -- for comparing builds bit for bit before a full run.
usage: python probe.py WORK_DIR OUT.json
"""

import json, sys
from pathlib import Path

work = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(work))
import study  # noqa: E402  (the bundle's driver; it imports policy, tasks, experiment)
from dataclasses import replace  # noqa: E402

data, _ = study.bytes_data()
out = {}
for contract in (14, 26):
    for task, model in (
        ("kv_curriculum", "global"),
        ("kv_curriculum", "query_full"),
        ("ordered", "query_matched"),
        ("kv_curriculum", "attention"),
        ("lm", "hybrid"),
        ("lm", "query_full"),
    ):
        r = replace(study.recipe(model), contract=contract)
        rs = study.records(task, 20261018, data)[:6]
        pol = study.BudgetPolicy(
            r,
            study.family(model),
            study.shape_set(task),
            32 * 2**20,
            4,
            micro_mode="all" if model == "attention" else "full",
        )
        rates = study.rates(len(study.records(task, 20261018, data)))
        rows = []
        with study.Runner(pol, seed=20261018) as n:
            for row, lr in zip(rs, rates):
                x, y = study.trim(row)
                e = n.update(x, y, float(lr))
                rows.append(
                    {
                        "loss": e["loss"],
                        "grad_norm": e["grad_norm"],
                        "digest": n.digest(),
                        "variant": e["variant"],
                        "microbatch": e["microbatch"],
                    }
                )
        out[f"{task}-{model}-c{contract}"] = {
            "updates": rows,
            "choices": [
                {
                    k: c[k]
                    for k in (
                        "variant",
                        "microbatch",
                        "accumulation",
                        "context",
                        "activations",
                        "scratch",
                        "packing_bytes",
                        "resident_bound_bytes",
                    )
                }
                for c in pol.choices.values()
            ],
        }
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print("probe", work.name, len(out), "cases")
