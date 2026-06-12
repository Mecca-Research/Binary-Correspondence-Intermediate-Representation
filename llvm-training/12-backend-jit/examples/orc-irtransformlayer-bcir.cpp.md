# BCIR `IRTransformLayer` sketch

This is an ownership and sequencing sketch, not a version-pinned standalone
program. Adapt callback signatures and pipeline construction to the LLVM release
in use.

```cpp
using namespace llvm;
using namespace llvm::orc;

struct BCIRTransformPolicy {
  std::string PipelineText;       // e.g. "function(instcombine,simplifycfg)"
  std::string PipelineVersion;
  std::string TargetDescriptorID;
  uint64_t ProfileGeneration;
};

Expected<ThreadSafeModule>
optimizeBCIRModule(ThreadSafeModule TSM,
                   const MaterializationResponsibility &MR,
                   BCIRTransformPolicy Policy) {
  Error TransformError = Error::success();

  TSM.withModuleDo([&](Module &M) {
    if (M.getDataLayout().isDefault()) {
      TransformError = make_error<StringError>(
          "BCIR module has no executor data layout", inconvertibleErrorCode());
      return;
    }

    PassBuilder PB;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Production code registers BCIR analysis/pass callbacks here, then parses
    // Policy.PipelineText or builds a fixed, versioned ModulePassManager.
    ModulePassManager MPM;
    MPM.addPass(VerifierPass());
    MPM.addPass(buildBCIRPipeline(PB, Policy.PipelineText)); // application API
    MPM.addPass(VerifierPass());

    stampDeploymentMetadata(M, Policy.TargetDescriptorID,
                             Policy.PipelineVersion,
                             Policy.ProfileGeneration);       // application API
    MPM.run(M, MAM);
  });

  if (TransformError)
    return std::move(TransformError);
  return std::move(TSM);
}

// CompileLayer ultimately owns target compilation. TransformLayer only mutates
// ThreadSafeModule before forwarding it.
auto Transform = std::make_unique<IRTransformLayer>(
    ES, CompileLayer,
    [Policy](ThreadSafeModule TSM,
             const MaterializationResponsibility &MR) mutable {
      return optimizeBCIRModule(std::move(TSM), MR, Policy);
    });
```

Review points:

- The module triple/data layout must already describe the executor target.
- Profile data is an immutable generation snapshot, not a live counter view.
- BCIR-specific pass registration must occur before parsing textual pipelines.
- Failure must propagate through `Expected`, preventing object emission.
- The materialization responsibility is useful for diagnostics and requested
  symbol context; it does not replace semantic module verification.
