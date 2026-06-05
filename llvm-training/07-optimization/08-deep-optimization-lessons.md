# Deep Optimization Lessons for BCIR

This lesson connects LLVM's deeper optimizer machinery to BCIR lowering work.
The goal is not to memorize every pass, but to know when a pass can invalidate a
BCIR assumption such as one source node mapping to one IR region, a diagnostic
attachment staying near the operation it explains, or a loop retaining the shape
expected by the backend.

Official references:

- [Using the New Pass Manager](https://llvm.org/docs/NewPassManager.html)
- [Writing an LLVM New PM Pass](https://llvm.org/docs/WritingAnLLVMNewPMPass.html)
- [MemorySSA](https://llvm.org/docs/MemorySSA.html)
- [LLVM's Analysis and Transform Passes](https://llvm.org/docs/Passes.html)

## PassBuilder and New Pass Manager plugin syntax

The new pass manager builds pipelines from text with `PassBuilder`. For normal
experiments, use `opt -passes=...`:

```bash
opt -S -passes='verify,mem2reg,instcombine,sccp,simplifycfg' input.ll -o out.ll
opt -S -passes='function(mem2reg,instcombine),module(function(sccp))' input.ll -o out.ll
opt -S -passes='default<O2>' input.ll -o o2.ll
```

For an out-of-tree pass plugin, the common `opt` shape is:

```bash
opt -load-pass-plugin=./libBCIROptPasses.so \
  -passes='bcir-preserve-map,verify' \
  input.ll -S -o checked.ll
```

A plugin exposes a `llvmGetPassPluginInfo()` entry point and registers pipeline
parsing callbacks through `PassBuilder`. In review, verify four things:

1. The textual pass name is stable and documented.
2. The pass kind matches the IR unit it rewrites: module, CGSCC, function, or
   loop.
3. The pass reports preserved analyses accurately; stale `MemorySSA`, dominator,
   or loop analyses are worse than recomputing them.
4. Any BCIR verifier pass runs both before and after transforms that may split,
   merge, clone, or erase mapped IR.

### BCIR risk checklist

- **Losing 1:1 mapping:** a pipeline can split one BCIR operation across several
  blocks or fold several operations into one SSA value. Record whether metadata
  is a provenance set, not a single required source node, before enabling it.
- **Erasing diagnostic metadata:** transforms may drop unknown attachments when
  replacing an instruction. If the metadata drives diagnostics, add a verifier or
  preservation test instead of assuming it survives.
- **Changing loop/header shape before lowering:** do not run loop canonicalizers
  before a BCIR lowering phase that requires the original header/latch layout.
- **Speculating poison:** plugin passes that hoist or fold operations must obey
  LLVM poison rules and insert `freeze` where required by the transform.

## MemorySSA and memory-dependence reasoning

MemorySSA summarizes memory operations as a def-use graph: stores, calls, and
other memory-writing operations are memory definitions; loads are memory uses;
control-flow joins can create memory PHI nodes. It helps a pass answer questions
like: "what memory write can this load observe?" without scanning arbitrary
instruction ranges.

Try the alias-shape example:

```bash
opt -passes='print<memoryssa>' examples/memoryssa-alias-shape.ll -disable-output
opt -S -passes=gvn examples/memoryssa-alias-shape.ll -o -
```

In `memoryssa-alias-shape.ll`, a store through `%maybe_alias` forces conservative
reasoning for a later load from `%slot`, while a store to `%neighbor` has a more
specific shape. The important lesson for BCIR is that pointer shape controls how
much motion or elimination is legal. If BCIR lowers two graph slots to opaque
`ptr` values without usable provenance, the optimizer must assume more aliases
and will keep more memory operations. If BCIR adds inaccurate `noalias`,
`readonly`, TBAA, or access-group metadata, the optimizer may remove or reorder
operations that were distinct BCIR events.

MemorySSA is not a substitute for alias analysis. It organizes memory accesses;
alias analysis still decides whether two locations may overlap. When reviewing a
memory transform, ask:

- Which MemoryDef can the MemoryUse legally read from?
- Did a call, volatile access, atomic access, or unknown pointer escape create a
  barrier?
- Did the transform update or invalidate MemorySSA after changing memory IR?
- Does the before/after IR still preserve BCIR's diagnostic/provenance metadata?

## SCCP and constant propagation hazards

Sparse conditional constant propagation (`sccp`) combines value propagation with
reachability. It can prove a value constant, delete now-unreachable edges, and
expose follow-up simplifications.

Try:

```bash
opt -S -passes='sccp,simplifycfg' examples/sccp-before.ll -o -
diff -u examples/sccp-before.ll examples/sccp-after.ll
```

SCCP is powerful because it reasons over SSA values without touching unrelated
instructions. That also makes hazards easy to miss:

- If a BCIR diagnostic branch becomes constant, the branch and the metadata on
  the dead side can disappear.
- If one BCIR node's value folds into another node's expression, a strict 1:1
  node-to-instruction mapping is gone.
- If a transform treats `undef` or poison like an ordinary known value, it can
  speculate undefined behavior. LLVM permits many folds around poison, but a
  pass that creates a new control dependency from a poison value may need
  `freeze` to make the value stable.
- Constants derived from attributes or metadata are only valid if those
  contracts are true for every BCIR lowering path.

The paired SCCP examples intentionally keep `!bcir.diag` attachments near the
branch and call sites so a reviewer can see which diagnostics would be erased by
constant reachability.

## LoopRotate and canonical loop shape

`loop-rotate` rewrites many top-tested loops into a rotated loop where the body
is entered only after a preheader check and the loop condition is evaluated on
the latch. This is a canonical shape for later loop transforms, but it changes
which block looks like "the header" to a source-level lowering pass.

Try:

```bash
opt -S -passes=loop-rotate examples/loop-rotate-bcir-before.ll -o -
diff -u examples/loop-rotate-bcir-before.ll examples/loop-rotate-bcir-after.ll
```

BCIR-specific cautions:

- If a lowering phase expects a header block to carry graph-loop metadata, run it
  before rotation or teach it to recognize rotated headers, preheaders, latches,
  and LCSSA exits.
- If diagnostics point at "loop entry" or "loop body" instructions, rotation may
  clone the entry test and move the repeated test to the latch.
- If one BCIR loop maps to one contiguous region, rotation adds a preheader path
  and an exit critical edge that may need the same provenance set.
- If the loop condition computes with `nsw`, `nuw`, `exact`, or poison-capable
  values, do not speculate it earlier unless the transform's legality proof
  handles poison or freezes the value.

For BCIR training data, keep both pre-canonical and post-canonical fixtures.
The before file documents what the lowering emitted; the after file documents
what optimizer consumers are likely to see.

## Review recipe for optimizer changes

1. Run `opt -passes=verify` before the experiment.
2. Run exactly one transform first, using `-S`, and diff the output.
3. Re-run `verify` and any BCIR mapping verifier.
4. Check whether metadata moved, disappeared, or became many-to-one.
5. Only then place the pass in a broader `default<O*>` or plugin pipeline.

## See also

- [`01-pass-model.md`](01-pass-model.md)
- [`02-common-analysis-passes.md`](02-common-analysis-passes.md)
- [`03-common-transform-passes.md`](03-common-transform-passes.md)
- [`examples/memoryssa-alias-shape.ll`](examples/memoryssa-alias-shape.ll)
- [`examples/sccp-before.ll`](examples/sccp-before.ll)
- [`examples/loop-rotate-bcir-before.ll`](examples/loop-rotate-bcir-before.ll)
