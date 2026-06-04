# Binary Analysis Beyond Static LLVM IR

Static LLVM IR is necessary but not sufficient for binary-code similarity,
security review, or performance triage. A final executable reflects profile data,
link-time decisions, object layout, code generation, dynamic inputs, and physical
CPU behavior. This section teaches an agent to reconcile the clean IR graph with
runtime evidence.

Read these chapters after the backend/JIT path:

1. [`01-microarchitecture-side-channels.md`](01-microarchitecture-side-channels.md)
   — cache, branch-prediction, and timing side channels that can invalidate a
   static "looks safe" conclusion.
2. [`02-dynamic-traces-and-counters.md`](02-dynamic-traces-and-counters.md) — how
   to pair IR/control-flow facts with sampled traces and hardware performance
   counters.
3. [`03-interpretable-bcsa-features.md`](03-interpretable-bcsa-features.md) —
   lightweight static features and triage heuristics for BCSA before expensive
   dense embeddings.

The example data files are deliberately tiny. They define schemas and review
patterns rather than claiming hardware-portable benchmark numbers.
