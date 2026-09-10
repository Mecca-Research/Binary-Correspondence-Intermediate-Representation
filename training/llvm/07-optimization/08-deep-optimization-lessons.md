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

## Putting advanced passes into a BCIR pipeline

Treat advanced LLVM passes as named pipeline components with explicit BCIR
inputs and outputs, not as a single opaque `-O2` blob. A useful teaching pipeline
is:

```bash
opt -S -passes='sccp,loop-rotate,loop-vectorize,verify' \
  examples/bcir-sccp-freeze-after.ll -o -
```

For a custom memory pass, spell the analysis requirement at the point where the
function pipeline consumes it:

```bash
opt -S -passes='require<memoryssa>,function(bcir-memory-audit,verify)' \
  examples/bcir-memoryssa-pipeline.ll -o checked.ll
```

`require<memoryssa>` makes the module pipeline materialize the MemorySSA analysis
for functions before the nested function pipeline. It does not make later
transforms preserve MemorySSA automatically: a pass that changes memory accesses,
CFG, dominance, or loop structure must either update the analysis through the
right updater API or report that it invalidated the analysis so the new pass
manager recomputes it.

### Inputs and boundaries for each component

| Component | BCIR pipeline role | Review focus |
|---|---|---|
| MemorySSA | Memory-dependence structure for custom passes that need def/use reasoning over loads, stores, calls, and memory PHIs. | MemorySSA organizes memory accesses, but alias analysis still decides whether two locations overlap. Unknown calls, pointer escapes, volatile/atomic operations, address-space rules, and imprecise TBAA or alias scopes can force conservative answers; inaccurate `noalias` metadata can make a transform unsound. |
| SCCP | Value/reachability simplifier that can delete dead BCIR diagnostic paths before heavier loop work. | SCCP must not turn poison or `undef` into a stable branch/control decision. If lowering creates a poison-capable condition, insert `freeze` before the condition becomes a branch or a value that SCCP may use for reachability. Compare [`examples/bcir-sccp-freeze-before.ll`](examples/bcir-sccp-freeze-before.ll) with [`examples/bcir-sccp-freeze-after.ll`](examples/bcir-sccp-freeze-after.ll). |
| LoopRotate | Canonicalization step after BCIR loop lowering and before vectorization. | Run it after lowering has emitted all source-shape diagnostics, then before `loop-vectorize` so vectorization sees a canonical preheader/header/latch/exiting-block shape. Keep the lowering boundary explicit because rotation may clone tests and move metadata-bearing instructions. |
| Profile metadata and PGO | Pipeline input that biases inlining, layout, unrolling, and vectorization cost decisions. | `!prof` branch weights and instrumentation/sample profiles are evidence, not proof of legality. Preserve or deliberately drop profile metadata when cloning or deleting BCIR regions, and avoid stale profile counts after major CFG edits. |
| MLGO | Optimization-policy input, such as inlining or register-allocation advice, selected by a model. | MLGO is not a verifier. It can choose among legal optimization actions, but BCIR verifiers still have to check mapping metadata, lowering boundaries, and semantic invariants. |
| Masked vector lowering | Lowering bridge from predicated BCIR lanes to `llvm.masked.*` intrinsics, selects, VP intrinsics, or target masks. | Masks are legality devices only when inactive lanes truly do not access memory. Ordinary vector loads execute all lanes; masked loads/stores or gathers/scatters are needed when inactive lanes might be out of bounds or semantically disabled. |
| Interleaved access vectorization | Profitability-dependent widening for AoS/SoA patterns after BCIR has made stride groups visible. | The legality proof depends on aliasing, alignment, and dependence facts; the profitability decision depends on shuffle/interleave costs. Keep legality separate from profitability in remarks and code review. |
| RISC-V scalable vectors | Target-specific lowering for `<vscale x N x T>` and RVV predication. | Do not assume a fixed lane count, fixed tail length, or cheap gather/scatter. BCIR vector metadata should describe element semantics and masks, not a hard-coded number of physical lanes. See [`../09-vectorization/examples/bcir-interleaved-riscv-sketch.ll`](../09-vectorization/examples/bcir-interleaved-riscv-sketch.ll). |
| AVX/AVX-512 masks | Target-specific fixed-vector lowering that may use k-register masks, masked moves, blends, or scalar fallbacks. | AVX2 often lowers masks through blends and full-width memory operations; AVX-512 can suppress inactive memory lanes for masked loads/stores, but mask materialization, all-ones masks, fault-suppression details, and downclock/cost effects are target-subtarget questions. See [`../09-vectorization/examples/bcir-avx512-mask-sketch.ll`](../09-vectorization/examples/bcir-avx512-mask-sketch.ll). |

A BCIR-oriented order is therefore: finish semantic lowering, preserve truthful
alias/profile/mask metadata, run local value cleanup such as SCCP only after
poison-sensitive values are frozen, canonicalize loops with `loop-rotate`, then
let vectorization and target lowering make legal/profitable choices.

### Advanced-pass review checklist

- Did the pipeline preserve BCIR metadata that is still semantically required,
  or deliberately convert it into a provenance set when instructions were
  cloned, merged, or erased?
- Did every pass that edited memory, dominance, CFG, or loops update analyses or
  invalidate stale analyses instead of relying on old MemorySSA, alias, dominator,
  or loop information?
- Did the transform avoid speculating poison and insert `freeze` before using a
  poison-capable value for branch reachability, masking, or profile-guided
  control decisions?
- Did the pipeline keep lowering boundaries explicit, especially the boundary
  between BCIR semantic lowering, loop canonicalization, vectorization, and
  target-specific mask/scalable-vector lowering?
- Did review notes separate legality from profitability: alias/dependence/mask
  correctness first, cost model, PGO, MLGO, and target heuristics second?


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
- [`examples/bcir-memoryssa-pipeline.ll`](examples/bcir-memoryssa-pipeline.ll)
- [`examples/sccp-before.ll`](examples/sccp-before.ll)
- [`examples/bcir-sccp-freeze-before.ll`](examples/bcir-sccp-freeze-before.ll)
- [`examples/bcir-sccp-freeze-after.ll`](examples/bcir-sccp-freeze-after.ll)
- [`examples/loop-rotate-bcir-before.ll`](examples/loop-rotate-bcir-before.ll)
- [`../09-vectorization/07-masked-and-interleaved-access.md`](../09-vectorization/07-masked-and-interleaved-access.md)
- [`../17-new-pass-manager/05-mlgo-and-profile-guided-pipelines.md`](../17-new-pass-manager/05-mlgo-and-profile-guided-pipelines.md)
