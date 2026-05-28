# Attribution

## Source material

This repository's chapter prose draws on the **LLVM IR Quick Reference**
by **Ayman Alheraki** (https://simplifycpp.org). Text is paraphrased and
restructured for agent consumption — examples and pitfall sections are
original to this repo, drawn from the LLVM Language Reference Manual and
from real bugs in the sibling BCIR project.

## Grammar

`10-grammar/llvm-ir.tm` is a Textmapper grammar covering LLVM IR
(~LLVM 7+, with updates through opaque pointers and modern attributes).
The grammar declares its own package as `github.com/llir/ll`. Used here
as a reference; consult upstream for the canonical version.

## LLVM upstream

The canonical truth for LLVM IR syntax and semantics is the LLVM
Language Reference Manual: https://llvm.org/docs/LangRef.html

When this repo and the LangRef disagree, the LangRef wins.

## License

This repository inherits Apache-2.0 from the parent BCIR project.
