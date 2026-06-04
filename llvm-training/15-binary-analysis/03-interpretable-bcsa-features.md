# Interpretable BCSA Feature Triage

Dense neural embeddings are useful for difficult binary-code similarity analysis
(BCSA), but they should not be the first or only tool. Many triage decisions can
be made with cheap, explainable features extracted from LLVM IR, machine CFGs, or
recovered binary structure.

## Why keep a lightweight layer

Interpretable features help an agent:

- reject obviously unrelated functions before expensive embedding inference;
- explain why two functions were grouped or separated;
- detect compiler-pipeline effects such as inlining, outlining, or block layout;
- identify cases that need dynamic evidence rather than more static tokens;
- reserve dense models for obfuscated, stripped, or high-risk functions.

## Static feature families

| Feature | Extraction idea | Why it helps |
| --- | --- | --- |
| Basic block count | Count labels/terminators in IR or recovered CFG. | Fast size/shape proxy. |
| Edge count and cyclomatic complexity | `M = E - N + 2P` for CFG components. | Captures control complexity. |
| Opcode histogram | Count normalized instruction opcodes. | Robust to local renaming. |
| Callsite summary | Count direct calls, indirect calls, intrinsic calls, and external helpers. | Separates wrappers, kernels, and runtime-heavy code. |
| Memory-access summary | Count loads/stores/atomics/GEPs and address-space use. | Highlights data-movement kernels. |
| Constant and type sketch | Bucket integer widths, vector widths, struct/array use, constants. | Captures ABI and algorithm clues. |
| Local structural hashes | Hash small neighborhoods such as block opcode sequences. | Provides explainable anchors for matching. |
| Metadata/profile signals | Branch weights, loop metadata, debug locations, TBAA. | Reveals frontend and profile context. |

See [`examples/bcsa-feature-sample.csv`](examples/bcsa-feature-sample.csv) for a
small schema that can be populated by a script or agent pass.

## Triage policy

1. **Normalize identifiers**: ignore SSA names and local labels unless they carry
   semantic meaning through debug info or symbols.
2. **Compare cheap features first**: size, CFG shape, opcode histograms, and call
   summaries.
3. **Escalate on ambiguity**: use graph matching, symbolic summaries, dense
   embeddings, or dynamic traces only when cheap features cannot separate the
   candidates.
4. **Flag compiler-pipeline distortion**: large call-count drops, block-count
   jumps, or profile-weight changes may mean inlining, LTO, or BOLT changed the
   binary shape.
5. **Keep explanations**: every triage result should include the top features
   that drove the decision.

## Example scoring sketch

```text
score =
  0.30 * cfg_shape_similarity +
  0.25 * opcode_histogram_similarity +
  0.20 * callsite_summary_similarity +
  0.15 * memory_access_similarity +
  0.10 * local_hash_overlap
```

This is not a universal model. It is a reviewable baseline. If a dense model
contradicts the interpretable score, inspect whether obfuscation, inlining,
vectorization, or post-link layout explains the difference.

## BCIR agent checklist

- Extract basic block, edge, call, memory, and opcode summaries before launching a
  heavyweight embedding workflow.
- Use dynamic traces for functions where security or profile behavior matters.
- Do not collapse all evidence into one opaque vector; keep raw feature columns
  and a short rationale.
- Treat tiny wrappers, intrinsic shims, and runtime ABI thunks as special cases:
  their structural features may be intentionally small but semantically important.
