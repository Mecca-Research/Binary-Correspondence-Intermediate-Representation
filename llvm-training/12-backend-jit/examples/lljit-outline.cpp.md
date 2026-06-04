# Minimal `LLJIT` outline

This is intentionally an outline, not a standalone project. ORC APIs evolve and
real code needs error handling, target initialization, IR construction/parsing,
and build-system integration. Use the official Kaleidoscope ORC tutorial for a
complete walkthrough.

```cpp
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

llvm::Expected<int> runWithLLJIT() {
  auto JITOrErr = llvm::orc::LLJITBuilder().create();
  if (!JITOrErr)
    return JITOrErr.takeError();

  auto JIT = std::move(*JITOrErr);
  auto Ctx = std::make_unique<llvm::LLVMContext>();
  auto M = std::make_unique<llvm::Module>("demo", *Ctx);
  M->setDataLayout(JIT->getDataLayout());

  // Build or parse IR into M here. For example, define:
  //   extern "C" int entry();

  llvm::orc::ThreadSafeModule TSM(std::move(M), std::move(Ctx));
  if (auto Err = JIT->addIRModule(std::move(TSM)))
    return std::move(Err);

  auto Sym = JIT->lookup("entry");
  if (!Sym)
    return Sym.takeError();

  using EntryFn = int (*)();
  auto *Entry = Sym->getAddress().toPtr<EntryFn>();
  return Entry();
}
```

Checklist when turning this outline into real code:

- initialize the native target if your embedding setup requires it;
- set the module data layout from `LLJIT`;
- keep the function pointer type ABI-compatible with the IR definition;
- decide which `JITDylib` owns the definitions;
- use resource trackers if you need to remove JITed code later.
