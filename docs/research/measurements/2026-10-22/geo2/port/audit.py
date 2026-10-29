"""GEO-2 audit: a rerun of the PR818 study held to Codex's frozen records, field by field.

usage: python3 -I audit.py THEIRS_DIR MINE_DIR OUT.json [--label NAME]

THEIRS_DIR is the bundle as uploaded (docs/research/wedge-pr818 extracted), MINE_DIR a work copy
whose results/ and bf16/results/ a rerun wrote. For every one of their training records the rerun's
must carry the same parameters, recipe, steps, input digest, positions, supervised tokens, selected
step, selected validation CE, selected and final state digests, validation history and per-update
(step, loss, gradient norm) -- exact equality, no tolerance -- and the same budget-policy choices
(variant, micro-batch, accumulation, context, plan sizes, packing bytes, resident bound). For every
evaluation record: the same rows (context, pairs, metrics, input digest, predictions, labels) and
counterfactuals. Wall times, source-derived IDs (policy_id, candidate_id, source_sha256) and
checkpoint container hashes are excluded, as Codex's own audit excludes them.
"""

import json, os, sys, glob

TRAIN_KEYS = (
    "parameters",
    "recipe",
    "steps",
    "input_sha256",
    "physical_positions",
    "supervised_tokens",
    "selected_step",
    "selected_validation_ce",
    "state_sha256",
    "final_training_state_sha256",
    "history",
)
#: the policy's decisions (what ran: the plan, its micro-batch and accumulation) -- held exactly
CHOICE_KEYS = (
    "variant",
    "effective_batch",
    "microbatch",
    "accumulation",
    "context",
    "parameters",
    "activations",
    "scratch",
    "threads",
    "staging_tokens",
    "budget_bytes",
)
#: the policy's byte accounting -- reported as deltas: a port's state record or workspace rule may
#: move them without moving a decision (ABI 5's recompute word, WS-PLAN's exact need)
BYTE_KEYS = (
    "packing_bytes",
    "packing_offset_floats",
    "workspace_floats",
    "arena_bytes",
    "fixed_bytes",
    "resident_bound_bytes",
)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def choices(policy):
    return [{k: c.get(k) for k in CHOICE_KEYS} for c in policy["choices"]]


def byte_deltas(a, b):
    out = {}
    for ca, cb in zip(a["choices"], b["choices"]):
        for k in BYTE_KEYS:
            if ca.get(k) != cb.get(k):
                out.setdefault(k, set()).add(cb.get(k) - ca.get(k))
    if a.get("resident_bound_bytes") != b.get("resident_bound_bytes"):
        out.setdefault("policy_resident_bound_bytes", set()).add(
            b["resident_bound_bytes"] - a["resident_bound_bytes"]
        )
    return {k: sorted(v) for k, v in out.items()}


def train_row(a, b):
    checks = {k: a.get(k) == b.get(k) for k in TRAIN_KEYS}
    ea, eb = a["events"], b["events"]
    checks["events_step_loss_grad_norm"] = len(ea) == len(eb) and all(
        x[k] == y[k] for x, y in zip(ea, eb) for k in ("step", "loss", "grad_norm")
    )
    checks["policy_decisions"] = choices(a["policy"]) == choices(b["policy"])
    first = next(
        (
            i
            for i, (x, y) in enumerate(zip(ea, eb))
            if any(x[k] != y[k] for k in ("step", "loss", "grad_norm"))
        ),
        None,
    )
    return checks, first


def eval_row(a, b):
    out = {
        "rows_equal": a["rows"] == b["rows"],
        "paired_equal": a["paired"] == b["paired"],
        "rows": len(b["rows"]),
        "paired": len(b["paired"]),
        "state_sha256_equal": a["state_sha256"] == b["state_sha256"],
        "policy_decisions": choices(a["evaluation_policy"]) == choices(b["evaluation_policy"]),
        "policy_byte_deltas": byte_deltas(a["evaluation_policy"], b["evaluation_policy"]),
    }
    if not out["rows_equal"]:
        diff = []
        for ra, rb in zip(a["rows"], b["rows"]):
            if ra != rb:
                diff.append(
                    {
                        "context": ra.get("context"),
                        "pairs": ra.get("pairs"),
                        "test_theirs": ra["test"],
                        "test_mine": rb["test"],
                        "predictions_differ": sum(
                            x != y for x, y in zip(ra["predictions"], rb["predictions"])
                        ),
                    }
                )
        out["row_differences"] = diff[:8]
    return out


def audit(theirs, mine, sub):
    rows, missing = [], []
    for p in sorted(glob.glob(os.path.join(theirs, sub, "train-*.json"))):
        name = os.path.basename(p)
        q = os.path.join(mine, sub, name)
        if not os.path.exists(q):
            missing.append(name)
            continue
        a, b = load(p), load(q)
        checks, first = train_row(a, b)
        row = {
            "case": name[len("train-") : -len(".json")],
            "checks": checks,
            "all_equal": all(checks.values()),
            "first_differing_update": first,
            "policy_byte_deltas": byte_deltas(a["policy"], b["policy"]),
        }
        ep, eq = p.replace("train-", "eval-"), q.replace("train-", "eval-")
        if os.path.exists(ep) and os.path.exists(eq):
            row["evaluation"] = eval_row(load(ep), load(eq))
        elif os.path.exists(ep):
            row["evaluation"] = "not run"
        rows.append(row)
    return rows, missing


def main():
    theirs, mine, out = sys.argv[1], sys.argv[2], sys.argv[3]
    label = (
        sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--label" else os.path.basename(mine)
    )
    report = {"label": label, "theirs": theirs, "mine": mine, "parts": {}}
    for sub in ("results", "bf16/results"):
        rows, missing = audit(theirs, mine, sub)
        evals = [r["evaluation"] for r in rows if isinstance(r.get("evaluation"), dict)]
        report["parts"][sub] = {
            "training_runs_compared": len(rows),
            "missing": missing,
            "training_runs_all_equal": sum(r["all_equal"] for r in rows),
            "updates_compared": sum(
                len(load(os.path.join(mine, sub, f"train-{r['case']}.json"))["events"])
                for r in rows
            ),
            "evaluations_compared": len(evals),
            "evaluations_all_equal": sum(
                e["rows_equal"]
                and e["paired_equal"]
                and e["state_sha256_equal"]
                and e["policy_decisions"]
                for e in evals
            ),
            "policy_byte_deltas": sorted(
                {f"{k}:{d}" for r in rows for k, ds in r["policy_byte_deltas"].items() for d in ds}
            ),
            "evaluation_rows": sum(e["rows"] for e in evals),
            "counterfactual_groups": sum(e["paired"] for e in evals),
            "rows": rows,
        }
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")
    for sub, part in report["parts"].items():
        print(
            f"{label} {sub}: training {part['training_runs_all_equal']}/{part['training_runs_compared']} "
            f"equal over {part['updates_compared']} updates (missing {len(part['missing'])}); "
            f"evaluations {part['evaluations_all_equal']}/{part['evaluations_compared']} equal "
            f"({part['evaluation_rows']} rows, {part['counterfactual_groups']} counterfactual groups); "
            f"policy byte deltas {part['policy_byte_deltas'] or 'none'}"
        )
        for r in part["rows"]:
            if not r["all_equal"]:
                bad = [k for k, v in r["checks"].items() if not v]
                print(
                    "   DIFF", r["case"], bad, "first differing update", r["first_differing_update"]
                )


if __name__ == "__main__":
    main()
