# PassBuilder and textual pipelines

`PassBuilder` is the front door to the new pass manager. It knows how to
register built-in analyses, construct default optimization pipelines, parse
textual pipeline names, and call plugin callbacks.

## The four analysis managers

Modern LLVM keeps analyses in managers that match the IR nesting level:

| Manager | IR unit | Common examples |
| --- | --- | --- |
| `ModuleAnalysisManager` (`MAM`) | whole module | target library info, profile summary, call graph summaries |
| `CGSCCAnalysisManager` (`CGAM`) | call-graph SCC | inliner-related SCC information |
| `FunctionAnalysisManager` (`FAM`) | function | dominator tree, loop info, scalar evolution, MemorySSA |
| `LoopAnalysisManager` (`LAM`) | loop | loop-local facts used by loop passes |

Registration normally follows this shape:

```cpp
LoopAnalysisManager LAM;
FunctionAnalysisManager FAM;
CGSCCAnalysisManager CGAM;
ModuleAnalysisManager MAM;

PassBuilder PB;
PB.registerModuleAnalyses(MAM);
PB.registerCGSCCAnalyses(CGAM);
PB.registerFunctionAnalyses(FAM);
PB.registerLoopAnalyses(LAM);
PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
```

`crossRegisterProxies` is not ceremonial. It connects the managers so a module
or CGSCC transform can invalidate function and loop analyses nested under the IR
it changed, and so adaptors can retrieve inner-manager results safely.

## Textual `opt -passes=...` syntax

The modern command-line spelling is `-passes=`. It composes pass names and
adaptors into a single textual pipeline:

```bash
opt -S -passes='verify,sccp,loop-rotate,verify' input.ll -o output.ll
```

Useful patterns:

- `verify` runs the IR verifier at the module level.
- `function(instcombine,simplifycfg)` adapts function passes into a module
  pipeline.
- `function(require<domtree>)` materializes a function analysis before later
  passes or debug printing.
- `default<O2>` asks `PassBuilder` for the default optimization pipeline at a
  named level.
- `print<...>` and `require<...>` are analysis utility pass forms where the
  analysis supports textual access.

A GAADMSF-oriented experiment can start with:

```bash
opt -S \
  -passes='verify,function(require<domtree>),sccp,loop-rotate,verify' \
  llvm-training/17-new-pass-manager/examples/gaadmsf-pipeline-before.ll \
  -o /tmp/gaadmsf-pipeline-after.ll
```

The command is intentionally conservative: it verifies before and after the
pipeline and uses modern pass names rather than legacy `-sccp -loop-rotate`
flags.

## BCIR pipeline placement rules

BCIR lowering adds contracts that LLVM's generic verifier cannot see:

- Run a BCIR invariant verifier before and after any pass that may delete,
  merge, clone, sink, hoist, vectorize, outline, or otherwise destructively
  rewrite BCIR-correspondent operations.
- Preserve 1:1 register correspondence until a named lowering stage consumes the
  mapping and emits replacement metadata or diagnostics.
- Invalidate BCIR analyses when metadata attachments, memory effects, call-site
  attributes, loop structure, or CFG shape changes.
- Keep hardware-specific GAADMSF transforms behind explicit attributes,
  metadata, or hardware profile checks.

## Pitfall: legacy pass-manager assumptions

Do not mix old mental models into a new-pass-manager pipeline. A legacy pass that
expects `getAnalysisUsage`, a legacy `FunctionPass`, or a command using
`opt -mem2reg -sccp` is not the same integration surface as a modern pass that
implements `run(IRUnit &, AnalysisManager &) -> PreservedAnalyses` and is named
through `PassBuilder` parsing callbacks.
