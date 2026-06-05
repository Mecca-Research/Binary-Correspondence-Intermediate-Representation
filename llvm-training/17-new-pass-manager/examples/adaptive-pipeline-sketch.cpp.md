# Adaptive pipeline sketch

This documentation-only sketch shows how a driver can choose a modern pipeline
from explicit BCIR and GAADMSF evidence. The exact APIs vary by embedding tool;
the important point is that policy decisions are evidence-based and the emitted
pipeline still uses modern pass-manager names.

```cpp
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

struct HardwareProfile {
  bool SupportsGAADMSF = false;
  bool SupportsHAMPrefetch = false;
};

static bool hasModuleFlag(const Module &M, StringRef Name) {
  return M.getModuleFlag(Name) != nullptr;
}

static bool hasAnyHamFunction(const Module &M) {
  for (const Function &F : M)
    if (F.hasFnAttribute("bcir.ham"))
      return true;
  return false;
}

static std::string buildBcirPipeline(const Module &M,
                                     const HardwareProfile &Profile) {
  SmallVector<StringRef, 16> Passes;
  Passes.push_back("verify");
  Passes.push_back("function(bcir-verify,require<domtree>)");
  Passes.push_back("sccp");
  Passes.push_back("function(bcir-verify)");
  Passes.push_back("loop-rotate");
  Passes.push_back("function(bcir-verify)");

  const bool HasGaadmsfIR = hasModuleFlag(M, "bcir.gaadmsf") ||
                            M.getNamedMetadata("gaadmsf.profile");
  if (HasGaadmsfIR && Profile.SupportsGAADMSF &&
      Profile.SupportsHAMPrefetch && hasAnyHamFunction(M))
    Passes.push_back("function(gaadmsf-ham-prefetch,bcir-verify)");

  Passes.push_back("verify");

  std::string Text;
  raw_string_ostream OS(Text);
  for (size_t I = 0; I < Passes.size(); ++I) {
    if (I)
      OS << ',';
    OS << Passes[I];
  }
  return OS.str();
}

static void runPipeline(Module &M, ModuleAnalysisManager &MAM,
                        const HardwareProfile &Profile) {
  PassBuilder PB;
  ModulePassManager MPM;
  ExitOnError Exit;

  std::string Pipeline = buildBcirPipeline(M, Profile);
  Exit(PB.parsePassPipeline(MPM, Pipeline));
  MPM.run(M, MAM);
}
```

Review questions:

- What happens if `bcir.gaadmsf` is absent?
- Which analyses become stale if `gaadmsf-ham-prefetch` adds memory operations?
- Does every destructive section have a nearby `bcir-verify` fencepost?
