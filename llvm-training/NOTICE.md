# Attribution

## Source material

This repository's chapter prose draws on the **LLVM IR Quick Reference**
by **Ayman Alheraki** (https://simplifycpp.org). Text is paraphrased and
restructured for agent consumption — examples and pitfall sections are
original to this repo, drawn from the LLVM Language Reference Manual and
from real bugs in the sibling BCIR project.

## Grammar

`10-grammar/llvm-ir.tm` is a full local snapshot of the Textmapper grammar from
`llir/grammar` commit `5a3820b516f7903e27ad16ebe4add1ec634f1c05`, consumed by
the `github.com/llir/ll` parser generator project. Upstream `llir/grammar` is
offered under 0BSD and Unlicense terms; it is included here so agents can answer
syntax-shape questions without leaving the repository. LLVM LangRef and the
target `llvm-as` remain the final authorities for current-version semantics.

## Corpus-specific original material

The BCIR mapping guide, repair exercise prompts, MLIR bridge summaries,
backend/JIT diagnostic notes, binary-analysis evidence schemas, and generated
example-governance rules are original documentation for this repository unless a
file states otherwise. They are intended as context-pack material for agents and
reviewers, not as canonical replacements for LLVM, MLIR, ORC, BOLT, or platform
vendor documentation.

## LLVM upstream

The canonical truth for LLVM IR syntax and semantics is the LLVM
Language Reference Manual: https://llvm.org/docs/LangRef.html

When this repo and the LangRef disagree, the LangRef wins.

## License

This repository inherits Apache-2.0 from the parent BCIR project.
