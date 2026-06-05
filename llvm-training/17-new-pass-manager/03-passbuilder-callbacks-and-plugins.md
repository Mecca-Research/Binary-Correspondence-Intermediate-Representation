# PassBuilder callbacks and plugins

Out-of-tree modern passes are commonly delivered as pass plugins. A plugin gives
LLVM a `PassPluginLibraryInfo` record from `llvmGetPassPluginInfo`; the record
contains an API version, plugin name, plugin version, and a function commonly
described as `registerPassBuilderCallbacks`: it receives a `PassBuilder &` and
registers parsing and extension-point callbacks.

## Minimal registration flow

```cpp
extern "C" LLVM_ATTRIBUTE_WEAK llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "BcirPasses", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(...);
            PB.registerOptimizerLastEPCallback(...);
          }};
}
```

When `opt` sees `-load-pass-plugin=/path/libBcirPasses.so`, it loads the plugin,
calls `llvmGetPassPluginInfo`, and lets the callback attach new pipeline names
or extension-point behavior.

## Pipeline parsing callbacks

Parsing callbacks map textual names to pass objects. A function-pass parser can
recognize a custom name and append the pass to a `FunctionPassManager`:

```cpp
PB.registerPipelineParsingCallback(
  [](llvm::StringRef Name, llvm::FunctionPassManager &FPM,
     llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
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
```

After registration, a user can write:

```bash
opt -load-pass-plugin=./libBcirPasses.so \
  -passes='verify,function(bcir-verify,bcir-ham-lower,bcir-verify),verify' \
  input.ll -S -o output.ll
```

Use names that are specific enough to avoid collisions. `lower` or `verify` is
likely to collide with another plugin or an upstream pass; `bcir-verify` and
`bcir-ham-lower` communicate ownership.

## Extension-point callbacks

Extension-point callbacks let a plugin inject passes into a default pipeline
without forcing every user to spell the full sequence. Common choices include:

- early pipeline starts for cheap validators or metadata canonicalization;
- scalar optimizer late points for post-canonicalization BCIR checks;
- optimizer-last points for final invariant verification; and
- vectorizer boundaries when BCIR register correspondence must be consumed or
  guarded before vectorization changes value shape.

Example policy:

```cpp
PB.registerOptimizerLastEPCallback(
  [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel Level) {
    (void)Level;
    MPM.addPass(llvm::createModuleToFunctionPassAdaptor(BcirInvariantVerifierPass()));
  });
```

Extension points should not smuggle in hardware-specific behavior without an IR
contract. A GAADMSF transform inserted at an extension point must still inspect
attributes, metadata, or an explicit hardware profile before changing IR.

## Plugin pitfalls

- **Pipeline-name collisions:** choose namespaced names such as `bcir-verify` or
  `gaadmsf-ham-prefetch`, not generic names such as `verify2` or `lower`.
- **Legacy assumptions:** modern callbacks append modern pass objects; they do
  not provide `getAnalysisUsage` or legacy pass IDs.
- **Stale analyses:** a plugin pass must return accurate `PreservedAnalyses` even
  when loaded through a textual pipeline.
- **Metadata loss:** upstream canonicalization may delete or merge instructions.
  If BCIR diagnostic metadata matters, validate it after extension-point
  insertion just as you would after an explicit custom pass.
