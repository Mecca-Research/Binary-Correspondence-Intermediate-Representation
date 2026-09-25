# The delta chain — K_BCIR → StreamPack advanced by declared deltas (GEM+ G18)

A plan that is re-derived incrementally is only honest if its laws are re-checked incrementally,
and if the incremental verdict is proved equal to the full one (GEM+ roadmap G18, staged plan
S4-B). This is the reference for the mechanism that does both. The executable oracle is:

| Module | Holds |
|---|---|
| [`bcir/kbcir/delta.py`](../../bcir/kbcir/delta.py) | the declared `Delta`, `apply_delta` (the reference application), `IncrementalOffer`, `IncrementalPlan` |
| [`bcir/gem/delta_pack.py`](../../bcir/gem/delta_pack.py) | `PackState`: the StreamPack kept as records and their encodings |
| [`bcir/verify/delta.py`](../../bcir/verify/delta.py) | `VerifyState`: the chain's verdict kept per unit |
| [`bcir/gem/delta_chain.py`](../../bcir/gem/delta_chain.py) | `DeltaChain` (the three states, advanced together) and `full_chain` (the chain from scratch) |

The chain is plan → pack → bytes → verdict:

```
result = realize.optimize(module, h, theta, policy)
pack   = hydrate_pipelined(module, result, plan, depth)
data   = streampack_abi.encode(pack)
diags  = verify(module) + verify_plan(module, result, h, theta=theta, policy=policy)
                        + verify_pack(module, pack)
```

`full_chain` runs it from scratch. `DeltaChain.apply(delta)` advances the three states and
returns a `Link` equal to `full_chain` of the module the delta declares: the same steps, costs
and score; the same pack object and the same bytes, or the same refusal with the same message;
the same diagnostics in the same order. Only the work differs.

## The declared delta (v0)

A `Delta` carries replacements and nothing else: `claims`, a tuple of `Claim`s each addressed by
its own id, and `resources`, a tuple of `Resource`s each addressed by its own RID. A replacement
cannot move a claim to another id, phase or position, so a delta keeps the module's shape.

Every rail refuses what v0 does not admit with `DeltaError`, before anything moves:

- a delta that is not a `Delta`;
- replacements that are not tuples;
- a replacement of the wrong type;
- two replacements for one id or RID;
- an id or RID the module does not declare;
- a module that declares a claim id twice. A delta names claims by id, so the whole module is
  refused (`_unique_where`, the one spelling of that law, on every rail);
- a module changed outside a delta. The S1-B rule is that a declared mutation moves
  `Module.revision`; a state built on the old revision refuses to advance.

What v0 does not express is a new plan (`DeltaChain.build`): adding or removing a claim or a
resource, moving a claim, editing a phase, or changing the target, Theta or policy.

`apply_delta(module, delta)` is the reference application, O(module). It builds a new `Module`
that shares every unchanged phase, claim and resource with the old one, and leaves the old one as
it was. The test fixtures also carry an application of their own that does not import the
mechanism (`bcir/tests/delta_fixtures.py::declared`), so the grader runs on a parent tree that has
no delta at all.

## The plan — `IncrementalOffer` and `IncrementalPlan`

**The offer** is `realize.fused_offer`, kept with the position indexes its discounts are read
from. Per phase block it keeps:

- the positions that write each operand, with multiplicity (a write bumps a version);
- the positions that write it under a barrier;
- the positions that read it;
- the positions holding each value-numbered identity.

Per resource it keeps the positions whose tier, or whose CSE eligibility, reads it.

A delta re-derives, in increasing position order:

- the claims it replaced;
- the later readers in the phase of an operand an edited claim writes differently, or under a
  different fence;
- the later holders of an identity that moved;
- the claims reading a resource whose domain or access changed. Those are the only resource
  fields the offer reads (`_resource_factors`, `cse_eligible`).

The discount is looked up in one rule table, `realize._DISCOUNT[duplicate][consumes][fenced]`.
`fused_offer`'s sequential walk gathers the three facts its own way and reads the same table.
The rows come from the one enumeration, `realize._offer_rows`.

**The plan** keeps, per column (a planning position, which is also the plan's step index), the
least weight into each realization and the predecessor it came from. A delta re-relaxes from the
first column whose rows or fused edge changed, through the one relaxation both planners run
(`realize._relax_column`). It stops at the first column that leaves the next column's input as it
was: the same cheapest narrow slot, the same cheapest wide slot, and the same gap between their
weights. Every later column is then the old one shifted by a constant. The shift is applied
lazily, through a Fenwick tree of column shifts, and the pass jumps to the next dirty column, if
any.

The path is walked back from the sink through the recomputed columns until it rejoins the old
path. The old path's steps are kept as they were, and only the changed steps are new objects.
`changed` names those step columns and `edited` names the columns whose claim was replaced.

Measured on a 512-claim single-phase module: a count edit re-relaxes exactly one column, and a
read edit two (its own, and the fused edge into the next).

## The pack — `PackState`

`PackState` keeps the hydrated pack as records, each with its own encoding, joined in chunks of
256 records. A delta re-emits only:

- the records of the changed steps, through `streampack.step_records`, the function `hydrate`
  builds them with. A record equal to the old one is kept as it was;
- the double-buffer prefetch of each transition into a phase whose reads an edit changed, through
  `streampack.double_buffer`, which `hydrate_pipelined` also uses;
- the generation records of the replaced resources, and the header maxima over them.

The re-emitted records are checked with the encoder's own contract functions, in the encoder's
order: header, segments, prefetches, generation vector. Every kept record passed those checks when
it was first emitted, so the first refusal is the one `encode` would raise. The records are then
written with the encoder's own writers. The header is re-derived for the new section counts,
counted from the sections themselves rather than from a counter kept beside them. Only the chunks
holding a re-emitted record are re-joined, and one CRC-32 covers the whole.

A pack the wire refuses raises what `encode` raises. The plan has moved by then, so the state is
behind its plan, and the next delta re-emits the pack in full.

The pack's version follows from facts a delta keeps: the pipelining depth, and whether the
registry declares a resource. A delta over today's hydrator therefore cannot move it. The state
still keeps the version's facts as counts, and re-emits in full a delta that would move them.

## The verdict — `VerifyState`

The three verifiers assemble their verdict from units, and the refactor that exposed them keeps
the full verifiers byte-identical. The units are:

- `_claim_laws`, per claim;
- `_pair_law`, per same-phase decoupled-tail pair;
- `_step_laws` and `_cost_law`, per plan step;
- `_segment_laws`, per segment;
- `_prefetch_law`, per prefetch;
- a few module-, plan- and pack-wide laws.

`VerifyState` keeps every unit's verdict. It re-derives only the units whose inputs moved, and
re-assembles the verdict in the verifiers' own order:

| Unit | Reads | Re-derived for |
|---|---|---|
| a claim's laws | the claim, the registry entries it names | every claim object that changed, every claim naming a resource that changed |
| a decoupled-tail pair | its two claims | the pairs that touch a claim that changed |
| a step's laws | the step, its claim, the declaring phase, the claim's offer rows | every step object that changed, every step whose claim changed or whose claim's rows moved |
| a step's cost | the step and the step before it | each changed step and the next known step |
| a segment's laws | the segment, the trace notes' claim ids, the prefetch targets by name, the registry | every changed segment, every segment whose claim's trace status or whose prefetch moved |
| the wide laws (registry, coverage, score, phase order, duplicates, header, generation vector) | what each names | when what it reads moved; the multisets are kept as counts |

**What changed is found by object identity** between the old and the new module, plan and pack
(`map(is_not, ...)` at C speed). The verdict never asks the planner or the emitter what they
changed. Its offer is its own `IncrementalOffer`, advanced from the module alone, and the plan
under verification never supplies the offer it is judged against. A forged or mis-reported change
is therefore re-verified like any other (L9). A plan or pack handed in unchanged after its module
moved (a planner or emitter that did not move) is re-derived through the claims and offers that
did move.

The verdict is re-derived from scratch, and `rebuilds` counts it, for anything outside the
delta's shape:

- other phases;
- a claim id at another position;
- other RIDs;
- a plan whose steps change count or claim ids;
- a pack whose segments or trace notes change count;
- a module with event phases whose claims changed. EV3 walks the whole program flow.

Over the corpus, the chain's own verdict is rebuilt only for modules with event phases.

A claim or resource mutated in place is not a change an identity diff can see. That is the S1-B
boundary: a mutation that follows the S1-B rule is refused (the revision moved). One that does not
is found where S1-B finds it, by the content identity (`provenance.module_identity` against
`digest_of`).

## The gates

`bcir/tests/delta_fixtures.py::measure` grades four exact rows. The tests
(`bcir/tests/test_delta_chain.py`), the harness (`tools/perf/gemplus_baseline.py --group delta`)
and the fault-table gate (`tools/perf/check_delta.py`) all share it.

| Row | Counts | RED (parent) | GREEN |
|---|---|---|---|
| `planner.delta.parity` | (case, step) pairs where the chain's plan differs from `optimize` of the declared module: steps, costs, score, BKPR bytes or refusal, or the module | 2,064 | 0 |
| `pack.delta.identity` | (case, step) pairs where the pack, its bytes, or the refusal and its message differ from the hydrated pack's | 2,064 | 0 |
| `verify.delta.identity` | (case, step, rail) triples where the incremental verdict differs from the full one, over the chain's own links and over a forged plan and pack each step | 4,128 | 0 |
| `delta.malformed.accepted` | (malformed input, rail) pairs not refused with `DeltaError` before anything moved | 51 | 0 |

**The corpus** is 258 cases of 8 steps each: a build and 7 deltas. The cases are:

- every fixed module (`examples.PROGRAMS`, the audit fixture at 64 and at 512 claims — two
  record chunks — the planner's `coverage_modules`, one module per construct a delta moves, and
  the U4 event-phase fixture, masked and bare), each under three rotating scopes (target, Theta,
  policy, pipelining depth 1–3);
- the planner's custom scopes on every third fixed module;
- 96 generated modules.

**The rounds.** Each round draws from edit families in rotation:

- every claim field;
- the dependency cones the offer re-derives;
- every resource field;
- a random mixture;
- a value the wire cannot carry (u64, u32, u16 length), beside an emittable edit that must survive
  the refusal, followed by the repair of that value alone;
- the event laws: disarm (EV2), open the mask window (EV3), the atomic consumer.

**The forged rail.** Each step, a second `VerifyState` is handed the chain's own plan and pack,
with a plan field and a pack field forged. The two forgery rotations have coprime lengths:
the planner's stale plan meets the cone edits, and the emitter's stale pack meets the resource
edits.

The tests assert what the corpus reaches:

- every declared family and forgery;
- every law the verdict can raise under a replacement: R1–R11, R18, EV, EV2 and EV3. R1.1 and
  EV1 are out of reach, because a delta cannot give two claims one id or move a phase's
  dependencies;
- a wire refusal in almost every case.

`tools/testing/faults/delta.json` holds 31 injected defects, each caught by its own row: the
cutoff, the lazy shift, the fused-edge dirtying, the backtrack, the offer's cones, admission, the
pack's record, transition, generation, chunk, stale and maxima logic, and the verdict's resource,
pair, offer, cost, prefetch, event, generation, total, descent and registry units. The first sweep
missed four, and each miss was a finding:

- a prefetch counter no reachable value could observe, now removed rather than kept as untestable
  state;
- a forgery pair the rotations could never schedule (coprime rotations now);
- two unit paths reachable only through a stale plan, now forged every cone round.

## Measured

At scale 8 (32,768 claims, CPython 3.11), a one-claim delta of the audit fixture costs **491
calls**, against **10,855,666** for the chain from scratch on the parent. At scale 4 it is also
491 calls: constant in the module in calls. The in-process factor against this tree's own chain
from scratch is 10,142,998 / 491, about 20,650×. The timed ratio at scale 4 is ~0.0085–0.0097, a
one-claim delta about 105–118× faster than the chain from scratch.

Since SP-ENC compiled the StreamPack encoder's record layouts (2026-09-25), the records a delta
re-emits go through the same compiled records: a one-claim delta costs **472 calls** at scale 8,
and the chain from scratch it is measured against **3,588,368** (the encoder was 74% of it).

What remains linear is C-level list copying (the new plan's and pack's lists), so wall time is
not constant in the module; the calls are.

Measured with the plan unchanged. The corpus parity above is `optimize`'s plan byte for byte, so
the delta plan carries exactly the certificate class the full plan does. That class is TMSAO-4
until G4's bounds cover this row.

## Not claimed

- A native twin of the delta chain.
- Incremental verification on the MLIR rail.
- Deltas that insert, remove or move claims or resources, edit phases, or change the scope.
- Incremental event laws: a module with event phases is re-verified from scratch.
- Detection of an in-place mutation that bypasses S1-B.
- Sublinear wall time: the per-delta list copies are O(n) at C speed.
