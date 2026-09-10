# Index: Exercises by skill

Use this index when you want to *practise* something rather than read about it.
Rows are phrased as the skill being exercised, so you can find the exercise from
what you want to get better at. Where a worked solution exists it is linked
beside the prompt; read the prompt first.

## Writing IR by hand

| Skill | Exercise |
|---|---|
| Emitting arithmetic on integer values | [`exercises/001-add.prompt.md`](../exercises/001-add.prompt.md) |
| Joining values that arrive from different branches | [`exercises/002-if-else-phi.prompt.md`](../exercises/002-if-else-phi.prompt.md) |
| Writing a counted loop in SSA form | [`exercises/003-loop-counter.prompt.md`](../exercises/003-loop-counter.prompt.md) |
| Reading and writing module-level state | [`exercises/004-global-load-store.prompt.md`](../exercises/004-global-load-store.prompt.md) |
| Computing the address of a field inside an aggregate | [`exercises/005-struct-gep.prompt.md`](../exercises/005-struct-gep.prompt.md) |
| Reducing a vector to a scalar with an intrinsic | [`exercises/007-vector-reduction.prompt.md`](../exercises/007-vector-reduction.prompt.md) |
| Building a lock-free update out of compare-and-exchange | [`exercises/008-cmpxchg-loop.prompt.md`](../exercises/008-cmpxchg-loop.prompt.md) |
| Lowering graph attributes into typed IR | [`exercises/010-vertex-edge-attribute-lowering.prompt.md`](../exercises/010-vertex-edge-attribute-lowering.prompt.md) |

## Repairing broken IR

| Skill | Exercise |
|---|---|
| Fixing a block argument that names the wrong predecessor | [`exercises/016-fix-phi-predecessor.prompt.md`](../exercises/016-fix-phi-predecessor.prompt.md) |
| Resolving two definitions of the same symbol | [`exercises/017-fix-duplicate-symbol.prompt.md`](../exercises/017-fix-duplicate-symbol.prompt.md) |
| Strengthening an ordering that fails to protect its invariant | [`exercises/019-fix-atomic-ordering.prompt.md`](../exercises/019-fix-atomic-ordering.prompt.md) |

## Reviewing someone else's work

| Skill | Exercise | Worked solution |
|---|---|---|
| Judging whether declared attributes match actual behaviour | [`exercises/025-attribute-contract-review.prompt.md`](../exercises/025-attribute-contract-review.prompt.md) | [`exercises/025-attribute-contract-review.solution.md`](../exercises/025-attribute-contract-review.solution.md) |
| Deciding when relaxed floating-point maths becomes unsafe | [`exercises/027-fast-math-risk-review.prompt.md`](../exercises/027-fast-math-risk-review.prompt.md) | [`exercises/027-fast-math-risk-review.solution.md`](../exercises/027-fast-math-risk-review.solution.md) |
| Reviewing a lowering for pattern fidelity | [`exercises/015-mlir-to-llvm-lowering-review.prompt.md`](../exercises/015-mlir-to-llvm-lowering-review.prompt.md) | |
| Reading evidence off a compiled binary | [`exercises/041-interpret-static-binary-evidence.prompt.md`](../exercises/041-interpret-static-binary-evidence.prompt.md) | [`exercises/041-interpret-static-binary-evidence.solution.md`](../exercises/041-interpret-static-binary-evidence.solution.md) |
| Checking whether a claim's provenance survives scrutiny | [`exercises/042-review-evidence-provenance.prompt.md`](../exercises/042-review-evidence-provenance.prompt.md) | [`exercises/042-review-evidence-provenance.solution.md`](../exercises/042-review-evidence-provenance.solution.md) |

## Working across the MLIR boundary

| Skill | Exercise | Worked solution |
|---|---|---|
| Deciding which dialect an operation belongs to | [`exercises/032-identify-mlir-dialect-boundaries.prompt.md`](../exercises/032-identify-mlir-dialect-boundaries.prompt.md) | [`exercises/032-identify-mlir-dialect-boundaries.solution.md`](../exercises/032-identify-mlir-dialect-boundaries.solution.md) |
| Converting a graph operation down to the LLVM dialect | | [`exercises/033-lower-mlir-graph-op-to-llvm-dialect.solution.md`](../exercises/033-lower-mlir-graph-op-to-llvm-dialect.solution.md) |
| Checking a type conversion for silent information loss | [`exercises/034-review-mlir-to-llvm-type-conversion.prompt.md`](../exercises/034-review-mlir-to-llvm-type-conversion.prompt.md) | [`exercises/034-review-mlir-to-llvm-type-conversion.solution.md`](../exercises/034-review-mlir-to-llvm-type-conversion.solution.md) |

## Diagnosing backend and JIT failures

| Skill | Worked solution |
|---|---|
| Tracing a symbol that fails to resolve at link time | [`exercises/035-diagnose-missing-symbol-relocation.solution.md`](../exercises/035-diagnose-missing-symbol-relocation.solution.md) |
| Finding which JIT layer dropped a definition | [`exercises/036-identify-orc-layer-failure.solution.md`](../exercises/036-identify-orc-layer-failure.solution.md) |
| Following a target description through to an emitted instruction | [`exercises/037-tablegen-to-mcinst-review.solution.md`](../exercises/037-tablegen-to-mcinst-review.solution.md) |
| Enforcing an invariant from inside a custom pass | [`exercises/038-custom-pass-bcir-invariants.solution.md`](../exercises/038-custom-pass-bcir-invariants.solution.md) |

## Agent task templates

Reusable prompts for driving an agent through a task rather than solving it yourself.

| Task | Template |
|---|---|
| Adding metadata without invalidating the verifier | [`exercises/templates/add-metadata-preserve-verifier-validity.prompt.md`](../exercises/templates/add-metadata-preserve-verifier-validity.prompt.md) |
| Diagnosing mapping drift introduced by the optimizer | [`exercises/templates/diagnose-optimizer-bcir-mapping-drift.prompt.md`](../exercises/templates/diagnose-optimizer-bcir-mapping-drift.prompt.md) |
| Lowering a graph fragment onto a fixed register mapping | [`exercises/templates/lower-bcir-graph-fragment-1to1-registers.prompt.md`](../exercises/templates/lower-bcir-graph-fragment-1to1-registers.prompt.md) |
