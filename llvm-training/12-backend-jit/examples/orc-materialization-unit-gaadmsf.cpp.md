# Custom GAADMSF `MaterializationUnit` sketch

This pseudocode advertises a GAADMSF kernel family and defers graph lowering and
target selection until ORC requests materialization. Exact ORC APIs vary by LLVM
release.

```cpp
using namespace llvm;
using namespace llvm::orc;

class GAADMSFMaterializationUnit final : public MaterializationUnit {
public:
  GAADMSFMaterializationUnit(SymbolFlagsMap Symbols,
                             std::shared_ptr<const GraphRecipe> Recipe,
                             DeploymentService &Deploy)
      : MaterializationUnit(Interface(std::move(Symbols), nullptr)),
        Recipe(std::move(Recipe)), Deploy(Deploy) {}

  StringRef getName() const override { return "GAADMSF graph family"; }

private:
  void materialize(std::unique_ptr<MaterializationResponsibility> R) override {
    // Keep R alive across asynchronous work. Recipe is immutable and shared so
    // it cannot disappear while compilation is queued.
    Deploy.enqueue(
        *Recipe, R->getRequestedSymbols(),
        [R = std::move(R)](Expected<ThreadSafeModule> TSM) mutable {
          if (!TSM) {
            R->failMaterialization();
            consumeError(TSM.takeError()); // production code reports context
            return;
          }

          // add() must associate the module with R's target JITDylib/resource
          // ownership and eventually resolve every advertised definition.
          if (Error Err = addThroughBCIRTransformAndCompileLayers(
                  *R, std::move(*TSM))) { // application adapter
            R->failMaterialization();
            consumeError(std::move(Err));
            return;
          }
        });
  }

  void discard(const JITDylib &JD, const SymbolStringPtr &Name) override {
    // An overriding definition won. Cancel per-symbol work if it has not started,
    // and ensure materialize() no longer promises Name. Real implementations
    // need synchronization with Deploy.enqueue and any delegated responsibility.
    Deploy.discard(Recipe->graphID(), *Name);
  }

  std::shared_ptr<const GraphRecipe> Recipe;
  DeploymentService &Deploy;
};
```

A production implementation should additionally:

1. build its symbol interface from normalized graph exports;
2. delegate independent kernels rather than making one giant responsibility;
3. preserve requested-vs-unrequested symbol semantics;
4. associate generated artifacts with a per-generation resource tracker;
5. report errors through the `ExecutionSession`; and
6. guarantee that every advertised symbol is emitted, delegated, discarded, or
   failed exactly once.
