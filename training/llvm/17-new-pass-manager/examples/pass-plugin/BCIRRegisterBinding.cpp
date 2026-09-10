//===- BCIRRegisterBinding.cpp - out-of-tree New PM plugin ----------------===//
//
// A complete, buildable LLVM pass plugin for the training corpus. It exists to
// show the four things a real out-of-tree pass needs and that a code sketch
// cannot demonstrate:
//
//   1. a custom *analysis* with an AnalysisKey, a Result type, and a correct
//      invalidate() hook;
//   2. a *module* transform that reaches function analyses through the proxy;
//   3. a *function* transform that returns an honest PreservedAnalyses;
//   4. registration through llvmGetPassPluginInfo(), including pipeline
//      parsing, a parameterized pass name, an analysis-registration callback,
//      and an extension point.
//
// The domain is BCIR's 1:1 register-correspondence contract: an instruction
// may carry `!bcir.reg !{!"rN"}` naming the source register it realizes, and
// within one function no two instructions may claim the same register. That
// contract is exactly the kind of invariant an optimizer breaks silently --
// duplicating an instruction duplicates its metadata -- so it is worth a pass
// that can be inserted before and after a destructive stage.
//
// Build and run:  see README.md in this directory, and the chapter
// training/llvm/17-new-pass-manager/06-building-an-out-of-tree-pass.md
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

/// The metadata kind that carries the BCIR source-register name.
constexpr const char *BindingMDKind = "bcir.reg";

/// Opt-in: also run the checker at the start of the default pipeline. This is
/// off by default so that plugging the library in does not change the output of
/// an unrelated `-O2` run.
cl::opt<bool> VerifyAtPipelineStart(
    "bcir-verify-at-pipeline-start",
    cl::desc("Run bcir-verify-bindings at the PipelineStart extension point"),
    cl::init(false), cl::Hidden);

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

/// One BCIR source register bound to one IR value.
struct Binding {
  std::string Register;
  const Instruction *Value;
};

class BCIRBindingAnalysis;

/// Result of BCIRBindingAnalysis: every binding in the function, plus the
/// bindings that collide with an earlier one.
struct BCIRBindingInfo {
  std::vector<Binding> Bindings;
  std::vector<std::pair<std::string, const Instruction *>> Collisions;

  /// A Result may keep itself alive across a transform that did not disturb it.
  /// This is the canonical form: survive only if the pass explicitly preserved
  /// this analysis, or preserved every function analysis.
  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &Inv);
};

class BCIRBindingAnalysis : public AnalysisInfoMixin<BCIRBindingAnalysis> {
  friend AnalysisInfoMixin<BCIRBindingAnalysis>;
  static AnalysisKey Key;

public:
  using Result = BCIRBindingInfo;

  Result run(Function &F, FunctionAnalysisManager &);

  /// Analyses are named the same way passes are, and for the same reason:
  /// `-debug-pass-manager` output is unreadable when every entry is a mangled
  /// C++ type from an anonymous namespace.
  static StringRef name() { return "bcir-binding-analysis"; }
};

AnalysisKey BCIRBindingAnalysis::Key;

bool BCIRBindingInfo::invalidate(Function &, const PreservedAnalyses &PA,
                                 FunctionAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<BCIRBindingAnalysis>();
  return !PAC.preserved() && !PAC.preservedSet<AllAnalysesOn<Function>>();
}

/// Read the register name out of `!bcir.reg !{!"rN"}`, or return an empty
/// StringRef when the instruction carries no well-formed binding.
StringRef bindingOf(const Instruction &I) {
  const MDNode *MD = I.getMetadata(BindingMDKind);
  if (!MD || MD->getNumOperands() == 0)
    return StringRef();
  const auto *Name = dyn_cast_or_null<MDString>(MD->getOperand(0).get());
  if (!Name)
    return StringRef();
  return Name->getString();
}

BCIRBindingInfo BCIRBindingAnalysis::run(Function &F, FunctionAnalysisManager &) {
  BCIRBindingInfo Info;
  StringMap<const Instruction *> Seen;

  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {
      StringRef Reg = bindingOf(I);
      if (Reg.empty())
        continue;

      Info.Bindings.push_back({Reg.str(), &I});
      if (Seen.count(Reg))
        Info.Collisions.emplace_back(Reg.str(), &I);
      else
        Seen[Reg] = &I;
    }
  }

  return Info;
}

//===----------------------------------------------------------------------===//
// Module transform: the checker
//===----------------------------------------------------------------------===//

class BCIRVerifyBindingsPass : public PassInfoMixin<BCIRVerifyBindingsPass> {
  raw_ostream &OS;
  bool Strict;

public:
  BCIRVerifyBindingsPass(raw_ostream &OS, bool Strict) : OS(OS), Strict(Strict) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

  /// The checker must run even on functions marked `optnone`: a legality
  /// verdict is not an optimization.
  static bool isRequired() { return true; }

  /// Without this, PassInfoMixin derives the name from the C++ type and both
  /// `-debug-pass-manager` and `--print-pipeline-passes` show
  /// `{anonymous}::BCIRVerifyBindingsPass` -- a spelling that cannot be fed
  /// back into `-passes=`. Name every pass after the name it registers.
  static StringRef name() { return "bcir-verify-bindings"; }
};

PreservedAnalyses BCIRVerifyBindingsPass::run(Module &M, ModuleAnalysisManager &MAM) {
  // Reaching a function analysis from a module pass goes through the proxy.
  // This is what crossRegisterProxies() wires up in a hand-built pipeline.
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  unsigned Bound = 0;
  unsigned Violations = 0;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    const BCIRBindingInfo &Info = FAM.getResult<BCIRBindingAnalysis>(F);
    Bound += static_cast<unsigned>(Info.Bindings.size());

    for (const auto &Collision : Info.Collisions) {
      ++Violations;
      OS << "bcir-verify-bindings: violation in function '" << F.getName()
         << "': register '" << Collision.first
         << "' is bound to more than one value\n";
      OS << "bcir-verify-bindings:   second binding:";
      Collision.second->print(OS);
      OS << "\n";
    }
  }

  OS << "bcir-verify-bindings: " << Bound << " binding(s), " << Violations
     << " violation(s)\n";

  if (Violations != 0 && Strict)
    report_fatal_error(Twine("bcir-verify-bindings: 1:1 register correspondence "
                             "violated (") +
                           Twine(Violations) + Twine(" violation(s))"),
                       /*gen_crash_diag=*/false);

  // A checker observes; it never changes IR.
  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// Function transform: consume the contract
//===----------------------------------------------------------------------===//

/// Removes every `!bcir.reg` binding. This models the lowering stage that
/// *consumes* the 1:1 contract: after it, correspondence is no longer claimed,
/// so no later pass can be blamed for losing it.
class BCIRStripBindingsPass : public PassInfoMixin<BCIRStripBindingsPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &);

  static StringRef name() { return "bcir-strip-bindings"; }
};

PreservedAnalyses BCIRStripBindingsPass::run(Function &F, FunctionAnalysisManager &) {
  bool Changed = false;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (I.getMetadata(BindingMDKind)) {
        I.setMetadata(BindingMDKind, nullptr);
        Changed = true;
      }
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  // `bcir.reg` is a custom kind that no in-tree analysis reads, so everything
  // else really is still valid. Abandon exactly the analysis this pass
  // invalidated. If the metadata being dropped were `!range`, `!alias.scope`,
  // or `!prof`, "all" would be a stale-analysis bug and the correct answer
  // would be a much smaller preserved set.
  PreservedAnalyses PA = PreservedAnalyses::all();
  PA.abandon<BCIRBindingAnalysis>();
  return PA;
}

//===----------------------------------------------------------------------===//
// Printer: the `print<...>` convention
//===----------------------------------------------------------------------===//

class BCIRBindingPrinterPass : public PassInfoMixin<BCIRBindingPrinterPass> {
  raw_ostream &OS;

public:
  explicit BCIRBindingPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    const BCIRBindingInfo &Info = FAM.getResult<BCIRBindingAnalysis>(F);
    OS << "bcir-bindings: function '" << F.getName() << "': "
       << Info.Bindings.size() << " binding(s), " << Info.Collisions.size()
       << " collision(s)\n";
    for (const Binding &B : Info.Bindings)
      OS << "bcir-bindings:   " << B.Register << "\n";
    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }

  static StringRef name() { return "print<bcir-bindings>"; }
};

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerCallbacks(PassBuilder &PB) {
  // Without this the pipeline cannot ask for BCIRBindingAnalysis at all: an
  // out-of-tree analysis has to be added to the manager that will be used.
  PB.registerAnalysisRegistrationCallback([](FunctionAnalysisManager &FAM) {
    FAM.registerPass([] { return BCIRBindingAnalysis(); });
  });

  // Module-level names, including one parameterized spelling.
  PB.registerPipelineParsingCallback(
      [](StringRef Name, ModulePassManager &MPM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "bcir-verify-bindings") {
          MPM.addPass(BCIRVerifyBindingsPass(errs(), /*Strict=*/false));
          return true;
        }
        if (Name == "bcir-verify-bindings<strict>") {
          MPM.addPass(BCIRVerifyBindingsPass(errs(), /*Strict=*/true));
          return true;
        }
        return false;
      });

  // Function-level names.
  PB.registerPipelineParsingCallback(
      [](StringRef Name, FunctionPassManager &FPM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "bcir-strip-bindings") {
          FPM.addPass(BCIRStripBindingsPass());
          return true;
        }
        if (Name == "print<bcir-bindings>") {
          FPM.addPass(BCIRBindingPrinterPass(errs()));
          return true;
        }
        return false;
      });

  // An extension point: inject the checker into a *default* pipeline without
  // naming it in -passes. Opt-in, so an unrelated -O2 run is unchanged.
  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel) {
        if (VerifyAtPipelineStart)
          MPM.addPass(BCIRVerifyBindingsPass(errs(), /*Strict=*/false));
      });
}

} // end anonymous namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "BCIRRegisterBinding", LLVM_VERSION_STRING,
          registerCallbacks};
}
