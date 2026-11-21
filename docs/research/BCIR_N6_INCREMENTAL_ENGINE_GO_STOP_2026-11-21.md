# N6, a native incremental engine: the GO/STOP (PORT-7)

**Status:** decision record, 2026-11-21. **Decision: STOP**, with the conditions that reopen it (§4).
**Owner of the question:** the native port audit
([`BCIR_NATIVE_PORT_AUDIT_2026-11-09.md`](BCIR_NATIVE_PORT_AUDIT_2026-11-09.md), N6 and PORT-7).
**Evidence:** [`measurements/2026-11-21/n6-incremental/`](measurements/2026-11-21/n6-incremental/README.json),
wall class, indicative.

## 1. The question

N6 is the incremental engine: `kbcir.delta` (the declared delta, the incremental offer, the incremental
planner), `gem.delta_chain` and `gem.delta_pack` (the plan, pack and bytes advanced by a delta) and
`verify.delta` (the verdict advanced by one), 1,797 lines. The audit placed it in the C++ library (CXX4)
and said why: "a native engine is for re-planning cost at scale". `docs/kernel/BCIR_DELTA_CHAIN.md` lists a
native twin of the delta chain as not claimed, and `PARITY.md` declares the chain oracle-only, because its
outputs are the full chain's and each of those has a native twin already (`bcir_kplan.c`'s planner and
hydrator, `bcir_verify.c`).

So the question is not whether the incremental engine can be written natively, but what it would buy over
what is native now: the C planner's whole plan of the module a delta declares.

## 2. The evidence

One-claim deltas of the planner audit fixture (the delta-ratio row's own deltas), per scale:

| Claims | Oracle, from scratch | Oracle, incremental | C planner, whole plan |
|---|---|---|---|
| 512 | 6.45 ms | 0.18 ms | 0.16 ms |
| 4,096 | 34.9 ms | 0.31 ms | 1.40 ms |
| 13,824 | 126 ms | 0.94 ms | 4.32 ms |
| 32,768 | 312 ms | 2.12 ms | 10.5 ms |

- **The C planner already re-plans at C speed.** About 0.32 µs a claim, linear: 10.5 ms for 32,768 claims,
  thirty times faster than the oracle from scratch.
- **Hosted re-planning already has an incremental path.** The oracle's `IncrementalPlan` applies a
  one-claim delta in 0.18 to 2.1 ms, below the C planner's whole plan from 4,096 claims on. What remains
  linear in it is C-level list copying (`BCIR_DELTA_CHAIN.md`), not interpretation: 472 calls a delta at
  scale 8.
- **No consumer re-plans natively from declared deltas.** The deltas v0 admits are claim and resource
  replacements under a fixed (h, Θ, policy); a change of Θ is a new plan. The on-target re-planning the
  roadmaps name -- N14's budgeted selection under a live Θ -- re-plans on a change of Θ, which v0 does not
  carry as a delta, and N14 had not landed when this was decided. It has since landed in C; see §4's third
  condition.
- **The native engine would be a fourth spelling.** The planner exists in the oracle, in C
  (`bcir_kplan.c`) and in MLIR (`-bcir-plan`, the min-plus path `optimize` runs). AGENTS.md's port rule
  forbids a fourth copy of an algorithm with a C twin and an MLIR pass; a native incremental planner would
  have to be the C planner's own DP keeping its columns, not a C++ re-implementation as CXX4 first sketched.

## 3. The decision

**STOP.** A native incremental engine would save at most the C planner's whole plan per delta -- 10.5 ms
at 32,768 claims -- to a consumer that does not exist, and would add a second derivation of a DP that
already has three. The hosted rail keeps the oracle's incremental chain, held byte for byte to the full
chain (`planner.delta.parity`, `pack.delta.identity`, `verify.delta.identity`); the target keeps the C
planner's whole plan.

## 4. What reopens it (GO when any holds)

1. **A native consumer re-plans from declared deltas** under a budget the C whole plan misses at its module
   size: the AI system's planning service (AIS) or a target loop that swaps claims, with the budget written
   down beside the module size it holds for.
2. **Modules grow past the C whole plan's budget**: at about 0.32 µs a claim, a 1 ms re-planning budget
   holds to some 3,000 claims and a 10 ms one to some 31,000; a module past the stated budget's size with
   deltas to apply reopens it.
3. **Deltas carry Θ.** N14's budgeted DP landed in C on 2026-11-21 (`bcir_kp_plan_budgeted`, beside the
   oracle's `optimize_constrained` and the law rail's `-bcir-rcsp`). The caps are the planner's argument, so
   the planner records carry no budget. When a v1 delta carries a change of Θ, on-target re-planning becomes
   the delta chain's own consumer.

## 5. What GO will contain

- The C planner's DP keeping every column (the cheapest narrow and wide slots and their predecessors) in
  caller-owned memory, and a delta entry that relaxes from the first dirty column until a column leaves the
  state the next one reads as it was, as `IncrementalPlan` does -- in `bcir_kplan.c`, not a new library, so
  the DP keeps one C spelling.
- Its parity gate: every delta of `delta_fixtures`' corpus planned incrementally equals `bcir_kp_plan` of
  the module it declares, byte for byte (BKPR), and the oracle's `IncrementalPlan`; and a work counter
  showing the columns relaxed are the delta's cone.
- The pack and verdict halves (`delta_pack`, `verify.delta`) after it, each held to its whole twin
  (`bcir_kp_hydrate`, `bcir_verify.c`) the same way.
