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
| `ring.json` | `tools/c/check_ring.py` (the G15 rows, both rails; needs a C compiler) | the live ring's seqlock, loss count, backpressure, epoch and takeover laws and its geometry laws on both rails; the envelope's wire laws, continuity, the stale-generation and unknown-required-signal laws; the generated signal table's bytes; the control ring as a transport |
| `handoff.json` | `tools/c/check_handoff.py` (the G16 rows on the oracle, the C twin and the C++ seam; needs a C and a C++ compiler) | the pack table's admission, its registry binding across a restarted plane, its epoch and pin laws and its state digest; the C++ owner's move; dispatch in place; the per-step freeze's header and claim laws; the shard manifest's wire laws, the partition, the sub-pack and frame spellings and the one reassembly predicate; and, through the Stage 3 exit flow, the other boundaries one generation crosses -- the plane's compare-and-swap, the intake's stale law, the telemetry ring's loss count against the intake's gaps |
| `delta.json` | `tools/perf/check_delta.py` (the G18 rows: the chain advanced by declared deltas against the chain from scratch; pure Python) | the incremental plan's cutoff, lazy shift, fused-edge dirtying, backtrack and kept steps; the incremental offer's reader, identity, tier and fence cones and its copy rule; the delta's admission (shape, duplicates, revision, addressing, the one id predicate); the delta pack's records, transitions, generation records, chunks, stale state and header maxima; the incremental verdict's resource, pair, offer, cost, prefetch, event, generation, total, descent and registry units |
| `alias.json` | `tools/perf/check_alias.py --require-llvm` (the G9 rows: the declared alias facts on every emitter of the elementwise kernel; needs a coherent clang/llvm-link/opt) | the one derivation's partition, hazard, element-size, declaration and qualifier rules and its fences; the LLVM kernel's `noalias`, alias scopes, domain, TBAA root and type, volatility and exit fence (four judged by LLVM itself); the ABI header's plan, the Q-fixed kernel's qualifier, the gather form's fence and the specialist's facts; the harness binding and the node harness's operation; R12's noalias, volatility, own-scope, TBAA, fence and restrict findings and its metadata reading |
| `encode.json` | `tools/perf/check_encode.py` (SP-ENC: the compiled StreamPack encoder against the encoder before it, and its call floor) | the plain layouts' exact-type gates (a bool claim id, an `__index__` target, a bool stride, a missing fence array), a swapped v3 segment layout and a dropped v2 buffer count; the encode contract's fast paths (width, lane, buffers) and the generation maxima; the field path's u16 string length, u32 element, u8 lane and trace-hash checks and its empty-array shortcut; its single walk of an array that is not a tuple or a list (u32, u64, strings) and its refusal of a generator at `len()`; the version test's dispatch; the saving itself (each plain path never taken) |
| `escape.json` | `tools/perf/check_escape.py --require-cc` (the G10 rows: escape verdicts, indirect-call narrowing and the effect footprint behind `commute`, judged by generated units, a dynamic witness and the twin's reports; needs a C compiler) | the oracle's store-as-write, static naming, heap object, cast, ops-table, device, declared-kind, lent, escape-root and two-target rules; the lowering's file-scope initializer names; the twin's twins of these, its device predicate's base half, its declared-pointer kind (read and set), its array and initializer marks, and its driver reporting on nothing |
| `planner.json` | `tools/c/check_planner.py` (the G17 rows on the compact planner, the pre-G17 reference and the native twin; needs a C compiler) | the compact planner's tie-break, fusion, thermal coupling, discounts, fence, value numbering, CSE exclusions and sink; R9's lane identity, phase binding, total diagnostics and base comparison; the native planner's tie-break, coupling, discounts, exclusions, fence, phase order, weights and 128-bit carry; both codecs' laws |

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

## ...and when one predicate becomes three, one fault becomes three

The converse happened in the same PR, and it is the same rule read the other way.
`generations._sync_tree` was a single helper that walked the staged tree and
fsynced everything in it, so a single fault -- delete the call -- removed every
durability claim at once and the `durability` check fired.

Then the helper had to go: reopening each staged file `"r+b"` to sync it is what
Windows needs and what fails on a read-only source chunk, so writing and syncing
now happen through one descriptor, in `_copy_durably` for the chunk copies and
`_write_durably` for the manifest, leaving `_sync_directories` only the entries
that name them. Three mechanisms, three separate assertions in the gate --
chunk syncs, a manifest sync, a directory sync.

One fault on any one of them would then have gone red and *looked* like evidence
for all three, while the other two were untested: the sweep would report the law
proved, and two thirds of it would never have been injected. So the entry became
three, one per mechanism, each replacing its code with the exact defect it
replaced (`shutil.copy2`, `write_text`, and no directory sync at all). A witness
must hit the law it exists to test (L11), and after a refactor the question is
not "does the old fault still apply" but "how many laws are there now".

## A public spelling nobody measures, and a corpus that cannot be built

The first sweep of `handoff.json` caught 29 of 31, and both misses were the
gate's, not the table's.

    Python frame: the blocks are dropped                        (none)   NOT CAUGHT
    Python sub-pack: a shard carries no generation vector ...   (none)   WRONG CHECK

The first was injected into `frame_of`, the public spelling of a shard set's
frame, and nothing noticed, because the oracle spelled the frame three times:
`frame_of`, and again inline in `split` and in `reassemble`. The grader compared
the C twin only against `split`, so the public function could drift from the
split, and from `bcir_shm_frame`, and no row would move. Now one private
`_frame` is the only spelling, and the grader compares the C harness's frame
and shards through `frame_of` and `sub_pack`, the twins of the C calls the
harness prints. Every public entry point is on a measured path (L14).

The second went red with no row named. The shards had lost their vector, so the
manifest encoder refused its own output (the self-check did its job), and it
did so while `measure()` was still *building* the hostile corpus it grades
against, outside every fail-closed wrapper. The grader died with a traceback, a
lost finding (L1). The corpora are the oracle's own splits and freezes, so a
defect in the oracle can first surface as a corpus that cannot be built. Now
every corpus is built through one guard that fails each row the corpus feeds,
and the entry fires `handoff.reentry.divergent` as it was written to.

## A witness that cannot see its law

The first sweep of `planner.json` caught 26 of 29. In all three misses the code was right, and
the witness could not see the law it was named for (L11).

    Python offer: a write does not bump its operand's version   (none)   NOT CAUGHT
    R9: the chosen realization's base cost is not compared      (none)   NOT CAUGHT
    Python decoder: an undeclared resource may carry a domain   (none)   NOT CAUGHT

- **The version counter.** The corpus rewrote an operand once. A first write yields version 1
  whether the counter increments or not, so only a second rewrite, with a duplicate reading after
  it, tells a counter from a flag. `coverage.cse` now has one.
- **R9's base comparison.** Every forgery was graded with the scope. There, R9 re-derives each
  step's cost from its base, so a forged base is refused by the cost law whether or not the offer
  compares it. `verify_plan(module, plan, h)` without Theta is a supported call, and there the
  offer is the only guard. `planner.r9.misjudged` now grades each forgery both ways.
- **The undeclared resource.** The variant broke a resource that was also a primary. A later law
  refused the record with the same status, and a status-only comparison cannot tell two laws
  apart. The seed now has a resource that only this law reads.

The fourth lesson came before the sweep. `tools/c/check_runtime.sh` injects a planner whose
128-bit addition drops its carry, and the gate passed it. The fixture written for the 128-bit
path, `wide_path_case`, has losing paths heavier than 2⁶⁴, but their high words come from the
multiply, which carries correctly, so no decision depended on the addition. `carry_case` does:
its one edge crosses 2⁶⁴ only by adding terms that each fit a u64. A fixture that exercises
large numbers is not a fixture that exercises the carry.

A witness has to be the only thing standing between the defect and a pass. When a law shares a
status with a later one, the variant must stop at it; when a law is shadowed by a stricter call
shape, the gate must also grade the call shape where it stands alone.

## A pair the witness cannot tell apart, and a row that compares nothing

The first sweep of `escape.json` caught 18 of 19:

    Footprint: a static local is private to its activation   effects.parity.mismatch   WRONG CHECK
        expected 'effects.commute.unsound'; the gate failed but 'effects.commute.unsound' did not fire

The footprints were wrong, and the parity row saw the two rails disagree. The witness row did not,
and it is the one judge that shares no code with the analysis. No two generated functions touched
one static, so every pair's two orders agreed whatever the footprints said. Each generated unit now
ends with `s_a` and `s_b`, which differ only through the static counter of a helper both call. Their
two orders diverge, and a footprint that drops the static calls them commuting.

The same slice found the parity rows blind to a twin that reports on nothing. They skipped any unit
the twin refused, so a driver whose every report failed read 0 on both rows. A refusal now counts,
except the one pinned preprocessor limit, and the table holds a fault that makes every report fail
(L2). A third addition narrows the open-world sites to no function. That is a claim no sound
analysis can make, and it fires `icall.unknown` under its proved floor. The last sweep caught all 26.

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
