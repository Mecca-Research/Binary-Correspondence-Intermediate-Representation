# BCIR pass plugin skeleton

This file is a documentation-only C++ sketch. It focuses on the modern pass
manager integration points rather than on build-system details.

```cpp
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

struct BcirRegisterMapAnalysis : AnalysisInfoMixin<BcirRegisterMapAnalysis> {
  struct Result {
    bool HasOneToOneMapping = true;

    bool invalidate(Function &, const PreservedAnalyses &PA,
                    FunctionAnalysisManager::Invalidator &) {
      auto PAC = PA.getChecker<BcirRegisterMapAnalysis>();
      return !(PAC.preserved() ||
               PAC.preservedSet<AllAnalysesOn<Function>>());
    }
  };

  Result run(Function &F, FunctionAnalysisManager &) {
    Result R;
    // Walk F, read !bcir.reg / !bcir.map metadata, and populate the map.
    // Set R.HasOneToOneMapping = false if two live IR values claim the same
    // source register before a lowering stage has consumed correspondence.
    return R;
  }

  static AnalysisKey Key;
};

AnalysisKey BcirRegisterMapAnalysis::Key;

struct BcirInvariantVerifierPass : PassInfoMixin<BcirInvariantVerifierPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    const auto &Map = FAM.getResult<BcirRegisterMapAnalysis>(F);
    if (!Map.HasOneToOneMapping)
      report_fatal_error("BCIR register correspondence invariant failed");

    // Also verify diagnostic metadata, HAM gates, and stage attributes here.
    return PreservedAnalyses::all();
  }
};

struct BcirHamLoweringPass : PassInfoMixin<BcirHamLoweringPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    if (!F.hasFnAttribute("bcir.ham"))
      return PreservedAnalyses::all();

    bool Changed = false;
    // Rewrite only operations marked with the expected HAM metadata.
    // If metadata, memory effects, or CFG changes, do not over-preserve.

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "BcirPassPlugin", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerAnalysisRegistrationCallback(
                [](FunctionAnalysisManager &FAM) {
                  FAM.registerPass([] { return BcirRegisterMapAnalysis(); });
                });

            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "bcir-verify") {
                    FPM.addPass(BcirInvariantVerifierPass());
                    return true;
                  }
                  if (Name == "bcir-ham-lower") {
                    FPM.addPass(BcirHamLoweringPass());
                    return true;
                  }
                  return false;
                });

            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                  MPM.addPass(createModuleToFunctionPassAdaptor(
                      BcirInvariantVerifierPass()));
                });
          }};
}
```

Example modern command once the plugin is compiled:

```bash
opt -load-pass-plugin=./libBcirPassPlugin.so -S \
  -passes='verify,function(bcir-verify,bcir-ham-lower,bcir-verify),verify' \
  input.ll -o output.ll
```
