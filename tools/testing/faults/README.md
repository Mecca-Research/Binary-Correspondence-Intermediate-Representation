# Fault tables — the standing evidence that the gates can fail

A gate that has only ever been seen to pass is a gate nobody has seen work. Each
table here is a set of defects, each paired with the check that must notice it,
and each one was injected and observed to fire before the fix it guards was
trusted (`docs/security/laws.md` L2).

They are data rather than scripts because the mechanics of injecting, running,
restoring and *proving the injection landed* belong in one place — L14, and the
reason is in L25: four separately-written sweeps produced one wrong verdict
between them, because a same-length edit can leave bytecode CPython still
believes.

```bash
# one table, all of its faults
python3 tools/testing/red_sweep.py --faults tools/testing/faults/training-langref.json

# just the faults about one check
python3 tools/testing/red_sweep.py --faults tools/testing/faults/training-database.json --only S18

# and a machine-readable record of what fired
python3 tools/testing/red_sweep.py --faults ... --json-out build/red/database.json
```

| Table | Gate | Faults |
|---|---|---|
| `training-langref.json` | `training/tools/verify_langref.py` | the document drifting from the code, and the code from the document |
| `training-database.json` | `training/tools/verify_database.py` | the table, the grammar, the planner, publication, and the thresholds |
| `training-embeddings.json` | `training/tools/verify_embeddings.py` | the stored derived column, and each of its fallbacks |

## What runs in CI, and what does not

A full sweep runs its gate once per fault plus a control, so the database table
costs sixteen gate runs — minutes each. That does not belong on every push.

What *is* cheap is keeping the tables honest, and that runs in the quick tier:
`bcir/tests/test_red_sweep.py::test_every_committed_table_loads_and_anchors_exactly_once`
re-reads every fault's anchor and fails if it no longer matches exactly one site.
That is the drift this evidence is actually exposed to — the code moves, the
anchor rots, and a sweep nobody reran reports `ANCHOR x0` as a fault that went
uncaught. The same file drives the harness into each of its refusals, so the
thing measuring the gates is itself measured.

Run the full sweep when you change a gate, when you change the code a fault
anchors into, and before claiming in a pull request that a check can fail.

## Adding a fault

```json
{
  "label": "what the defect is, in the reader's words",
  "expects": "the prefix of the check that must fire",
  "path": "training/tools/plan.py",
  "old": "exact text, occurring exactly once",
  "new": "the defective text"
}
```

`expects` is matched as a prefix of the name the gate prints, so `"S18"` catches
`S18: 2 recorded plan decision(s) moved`. A fault whose gate goes red for a
*different* reason is reported `WRONG CHECK`, not as a catch — going red is not
the same as being noticed by the check you were testing.
