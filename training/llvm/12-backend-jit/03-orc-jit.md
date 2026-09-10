# ORC JIT and LLJIT

ORC is LLVM's modern JIT infrastructure. It provides composable layers for
compiling LLVM IR, linking generated objects, managing symbols, supporting lazy
compilation, and coordinating code/data ownership across a JIT session.

Official references:

- [ORC Design and Implementation](https://llvm.org/docs/ORCv2.html)
- [Building a JIT: Starting out with KaleidoscopeJIT](https://llvm.org/docs/tutorial/BuildingAJIT1.html)
- [Kaleidoscope tutorial table of contents](https://llvm.org/docs/tutorial/)

This chapter gives the mental model. For a full tutorial project, follow the
Kaleidoscope ORC JIT tutorial rather than embedding a large buildable example
here.

For lower-level ORC internals, including `ExecutionSession`, `JITDylib`,
compile/object/transform layers, materialization responsibility, symbol
interning, and the JITLink relocation boundary, continue with
[`05-orc-layers.md`](05-orc-layers.md).

## What ORC provides

ORC is a framework for runtime compilation and linking. It can:

- accept LLVM IR modules and compile them for the host or another configured
  target;
- link generated object code into the current process or a separate executor;
- manage symbol definitions and lookups across JIT dynamic libraries;
- support lazy and on-demand compilation;
- track resources so code and data can be removed together;
- compose layers for compile, object-link, transform, and materialization steps.

Older LLVM examples may use MCJIT. New code should generally start with ORC,
especially the high-level `LLJIT` wrapper, unless maintaining legacy code.

## `LLJIT` high-level architecture

`LLJIT` is the convenient entry point for common in-process JIT use cases. It
wraps several lower-level ORC concepts:

| Concept | Role |
|---|---|
| `ExecutionSession` | Owns the symbol string pool, error reporting, and session-wide JIT state |
| `JITDylib` | A symbol namespace roughly analogous to a dynamic library |
| compile layer | Compiles LLVM IR to object code |
| object/link layer | Links object code and makes it executable |
| data layout and target machine setup | Ensures modules match the JIT target expectations |
| resource trackers | Group allocations and materializations so they can be removed together |

The usual high-level flow is:

```text
create LLJIT
  ↓
create or parse LLVM Module in a ThreadSafeContext
  ↓
wrap it in ThreadSafeModule
  ↓
add it to the JIT
  ↓
look up a symbol
  ↓
cast the address to a function pointer and call it
```

## Adding modules

`LLJIT::addIRModule` takes a `ThreadSafeModule`. The `ThreadSafeModule` pairs a
`std::unique_ptr<Module>` with a `ThreadSafeContext`, making ownership and
concurrency expectations explicit.

Minimal outline:

```cpp
auto Ctx = std::make_unique<llvm::LLVMContext>();
auto M = std::make_unique<llvm::Module>("demo", *Ctx);
M->setDataLayout(JIT->getDataLayout());

// Fill M with IR here.

llvm::orc::ThreadSafeModule TSM(std::move(M), std::move(Ctx));
if (auto Err = JIT->addIRModule(std::move(TSM)))
  return Err;
```

Set the module data layout from the JIT before adding IR. A data-layout mismatch
can produce incorrect code or hard-to-diagnose target lowering failures.

## Symbol lookup

After adding a module, use `lookup` to find a compiled symbol:

```cpp
auto Sym = JIT->lookup("entry");
if (!Sym)
  return Sym.takeError();
```

Symbol names are subject to target mangling conventions. `LLJIT` provides helper
APIs and a data-layout-aware mangle-and-intern path for lower-level use. In small
examples, exported C-style function names are the easiest symbols to find.

## Getting function pointers

A lookup result contains an executor address. For an in-process JIT, you can turn
that address into a callable function pointer with the expected signature:

```cpp
using EntryFn = int (*)();
auto *Entry = Sym->getAddress().toPtr<EntryFn>();
int Result = Entry();
```

The cast is your responsibility. If the function type does not match the IR
function's actual ABI, the C++ call has undefined behavior.

## Resource and session ownership basics

ORC makes ownership explicit because JIT code has runtime lifetime concerns:

- `ExecutionSession` coordinates session-wide state and must outlive materialized
  code managed by the JIT.
- `JITDylib` owns symbol definitions for a namespace.
- `ThreadSafeContext` owns the LLVM context for one or more modules.
- `ThreadSafeModule` transfers a module into the JIT.
- `ResourceTracker` can group definitions so they can be removed as a unit.

For simple tools, owning a single `std::unique_ptr<LLJIT>` until process exit is
often enough. For long-lived servers, REPLs, plugin systems, or hot-reloaders,
plan resource trackers and symbol namespaces up front.

## Link to Kaleidoscope instead of embedding a project

The official Kaleidoscope ORC tutorial walks through a real JIT incrementally:

- [Building a JIT: Starting out with KaleidoscopeJIT](https://llvm.org/docs/tutorial/BuildingAJIT1.html)
- [Building a JIT: Adding Optimizations](https://llvm.org/docs/tutorial/BuildingAJIT2.html)
- [Building a JIT: Lazy Compilation](https://llvm.org/docs/tutorial/BuildingAJIT3.html)

Use those chapters for full source and build context. This training repo keeps
only a compact outline in [`examples/lljit-outline.cpp.md`](examples/lljit-outline.cpp.md).

## Common mistakes

- Adding a module with a data layout that does not match the JIT.
- Looking up the unmangled name when the target expects a mangled symbol.
- Casting the address to the wrong C++ function pointer type.
- Destroying contexts, sessions, or resource trackers earlier than the code that
  depends on them.
- Copying an MCJIT-era tutorial into ORC code without adapting ownership and
  symbol-resolution concepts.
