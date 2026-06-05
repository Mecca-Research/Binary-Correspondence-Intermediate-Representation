# Custom passes, analyses, and preservation

Modern passes are small value types. A transform pass exposes a `run` method for
its IR unit and returns `PreservedAnalyses`; an analysis pass exposes a stable
`Key` and returns a result object.

## Transform-pass shape

A function transform pass usually looks like this:

```cpp
struct BcirNormalizePass : llvm::PassInfoMixin<BcirNormalizePass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
    auto &DT = FAM.getResult<llvm::DominatorTreeAnalysis>(F);
    (void)DT;

    bool Changed = normalizeBcirOperations(F);
    if (!Changed)
      return llvm::PreservedAnalyses::all();

    llvm::PreservedAnalyses PA;
    PA.preserve<llvm::DominatorTreeAnalysis>(); // only if the edit truly keeps it valid
    return PA;
  }
};
```

The return value is a correctness statement. If a pass changes the CFG, it
probably did not preserve dominator tree, loop info, post-dominators, or scalar
evolution. If a pass changes memory effects or alias metadata, it probably did
not preserve alias analyses or MemorySSA. If a pass only adds a diagnostic remark
without changing semantics, it may preserve most analyses, but a BCIR-specific
metadata analysis may still need invalidation.

## Analysis-pass shape

A custom analysis has an analysis object, a result object, and an invalidation
policy:

```cpp
struct BcirRegisterMapAnalysis : llvm::AnalysisInfoMixin<BcirRegisterMapAnalysis> {
  struct Result {
    llvm::DenseMap<const llvm::Value *, unsigned> SourceRegister;

    bool invalidate(llvm::Function &, const llvm::PreservedAnalyses &PA,
                    llvm::FunctionAnalysisManager::Invalidator &) {
      auto PAC = PA.getChecker<BcirRegisterMapAnalysis>();
      return !(PAC.preserved() || PAC.preservedSet<llvm::AllAnalysesOn<llvm::Function>>());
    }
  };

  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
    return buildRegisterMap(F);
  }

  static llvm::AnalysisKey Key;
};
```

Use a precise invalidation policy when the analysis depends on metadata rather
than only CFG. For BCIR, the register map should be invalidated when:

- `!bcir.reg`, `!bcir.map`, `!bcir.diag`, HAM, or GAADMSF metadata changes;
- a value with a mapped register is deleted, replaced, cloned, vectorized, or
  sunk into a different control-flow region;
- memory-effect metadata or attributes change in a way that changes the BCIR
  operation boundary; or
- CFG shape changes alter the dominance or execution conditions of mapped
  operations.

## `PreservedAnalyses` discipline

Choose the most conservative correct return value:

- `PreservedAnalyses::all()` only when the pass did not mutate any IR or
  analysis-visible side table.
- `PreservedAnalyses::none()` when the pass changed CFG, memory effects, calls,
  or BCIR metadata and you have not audited each analysis.
- A custom `PA.preserve<AnalysisT>()` list only after documenting why each
  preserved analysis remains valid.

Overclaiming preservation is worse than doing extra recomputation: it can make a
later pass consume stale dominator trees, stale loop information, stale
MemorySSA, or stale BCIR register maps.

## BCIR invariant verifier pass

A BCIR verifier can be implemented as a transform-like checking pass that does
not mutate IR:

```cpp
struct BcirInvariantVerifierPass : llvm::PassInfoMixin<BcirInvariantVerifierPass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
    const auto &Map = FAM.getResult<BcirRegisterMapAnalysis>(F);
    verifyOneToOneRegisterMapping(F, Map);
    verifyDiagnosticMetadata(F);
    return llvm::PreservedAnalyses::all();
  }
};
```

Place it before and after destructive transforms. The generic LLVM verifier
checks IR legality; the BCIR verifier checks lowering contracts that LLVM does
not know.
