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
costs one run more than it has faults — minutes each. That does not belong on
every push.

What *is* cheap is keeping the tables honest, and that runs in the quick tier:
`bcir/tests/test_red_sweep.py::test_every_committed_table_loads_and_anchors_exactly_once`
re-reads every fault's anchor and fails if it no longer matches exactly one site.
That is the drift this evidence is actually exposed to — the code moves, the
anchor rots, and a sweep nobody reran reports `ANCHOR x0` as a fault that went
uncaught. The same file drives the harness into each of its refusals, so the
thing measuring the gates is itself measured.

Run the full sweep when you change a gate, when you change the code a fault
anchors into, and before claiming in a pull request that a check can fail.

## A fault names the gate that owns the law, not the gate you were looking at

The first sweep of the full database table returned one `NOT CAUGHT`, and the
gate was right. The fault removed `newline="\n"` from a writer in
`build_chunks.py`, and it expected `verify_database.py`'s `line endings` check to
notice. That check inspects the *built bytes*, and on Linux text mode writes LF
either way — so the defect it was pointed at is invisible on this host by
construction, and the check's own docstring says so: the rule about **source**
belongs to `bcir/tests/test_line_endings.py`, which reads the tree statically and
therefore answers the same on every host.

The fault table was wrong, not the gate. Two things came out of it, and both are
in the tables now:

- the source-side fault is proved against the rail that owns it —
  `bcir/tests/test_line_endings.py` fires
  `these writes let the host choose the line ending … training/tools/build_chunks.py:373`;
- the database table keeps a line-ending fault that its own gate *can* catch
  anywhere, by writing `newline="\r\n"` explicitly: the built corpus then
  carries carriage returns on every host, which is the law `line endings` exists
  to hold.

So when a fault comes back `NOT CAUGHT`, ask first whether the check you named is
the one that owns the law — and whether the defect can even be *observed* on the
host running the sweep (`docs/security/laws.md` L12). A fault that only
manifests on one platform belongs with a host-independent witness, or it is
evidence about that platform and nothing else.

## An anchor that still matches is not a fault that still bites

The second full sweep returned a `NOT CAUGHT` of a different kind. The fault

    old: if not prefix.endswith("/"):
             return None
    new: if False:
             return None

was written to reintroduce the `^=` defect of PR #784 by removing the
trailing-separator guard. It did, once. Then the line *below* the guard changed
from `prefix.rstrip("/")` to `prefix[:-1]` — a fix for a different defect — and
the injection stopped expressing anything: without the guard, `training/data`
becomes `training/dat`, which names no counted directory, so the shortcut
declines and the walk gives the *right* answer. The fault had decayed into a
no-op while its anchor still matched exactly one site.

That is the gap between the two things that watch this evidence. The cheap
anchor test in the quick tier asks *does this fault still apply cleanly*, and the
answer was yes. Only the sweep asks *does it still break what it names*, and only
the sweep could have noticed. So:

- **re-run the full sweep whenever you change the code a fault anchors into**,
  not only when you change a gate — the anchor test passing is not the same
  evidence;
- **a `NOT CAUGHT` is three questions, not one**: is the named check the one that
  owns the law (above), can the defect be observed on this host (above), and does
  the injection still *produce* the defect it is named for. The third is the one
  that a fix in neighbouring code silently breaks.

The entry now replaces the whole shortcut rather than half of it, and
reintroducing it reports what it always should have:

    S4: 'training/data' means 'starts with' and resolved 2 row(s) where 4 path(s)
      start with it -- missing ['training/database.md', 'training/datastore.md']

## When N copies become one predicate, N faults become one

The embedding set records four digests, and each reader used to check its own --
or not. Three did not, which is why they became one predicate,
`embed_chunks.verified_bytes`. The fault that used to test the row-squares
reader's *private* digest check went with them: retargeted at the call into the
shared predicate, it stopped expressing the law and started expressing a crash.

    the derived column is read around the digest predicate   (none)   WRONG CHECK
        expected 'row_squares'; the gate failed but 'row_squares' did not fire

Bypassing `self._verified(spec, "row_squares")` removes the digest check *and*
the `OSError` handling behind it, so the gate's own missing-column probe --
which points the manifest at `squares.u32.absent` on purpose -- died with a
`FileNotFoundError` before any check could report. Going red is not being
noticed by the check you named, and the harness says so rather than counting it.

The entry is gone rather than repaired. The law it tested is carried by
`a declared artifact is read on the manifest's word again`, whose one injection
fires `index`, `quantized` *and* `row_squares` -- which is what a shared
predicate is for. Keeping a second, weaker copy beside it would be the mirror
list this tree refuses everywhere else (L14, L15).

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
