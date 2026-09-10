# The Driver and the Frontend Pipeline

## TL;DR

`clang` is a **driver**. It does not compile anything. It works out what jobs
are needed, builds an argument list for each, and runs them. The thing that
actually compiles is `clang -cc1`, a different program with a different — and
much larger, much less stable — flag surface.

```console
$ clang -ccc-print-phases -c control-flow-lowering.c
         +- 0: input, "control-flow-lowering.c", c
      +- 1: preprocessor, {0}, cpp-output
   +- 2: compiler, {1}, ir
+- 3: backend, {2}, assembler
4: assembler, {3}, object
```

Only phase 2 is "the frontend" in the sense this chapter uses the word. Phase 3
is LLVM's target backend, running inside the same process.

## Driver versus `-cc1`

| | Driver (`clang`) | Frontend (`clang -cc1`) |
| --- | --- | --- |
| Stability | user-facing, GCC-compatible, stable | internal, changes freely between releases |
| Job | pick jobs, find the toolchain, assemble arguments | preprocess, parse, analyse, emit IR, run codegen |
| Flags | `-O2`, `-g`, `-std=c17`, `-target ...` | `-triple`, `-emit-obj`, `-mrelocation-model`, ... |
| Reaching it | — | `clang -Xclang <flag>` or `clang -cc1 ...` directly |

`-###` prints the exact `-cc1` command the driver would run, without running
it. This is the single most useful debugging tool in the frontend:

```console
$ clang -### -c control-flow-lowering.c
 "/usr/lib/llvm-18/bin/clang" "-cc1" "-triple" "x86_64-pc-linux-gnu"
   "-emit-obj" "-mrelocation-model" "pic" ... "-fmath-errno" "-target-cpu" "x86-64"
   ... "-o" "control-flow-lowering.o" "-x" "c" "control-flow-lowering.c"
```

Every "why is my flag being ignored?" question ends here. Common answers it
gives immediately:

- the flag was consumed by the driver and translated into something else;
- the flag was passed through unchanged and the frontend ignored it;
- the driver never saw it because it landed after `--` or inside a response file;
- a later flag overrode it (the last one wins for most flags).

`-Xclang` passes one argument straight through to `-cc1`. That is how the
diagnostic flags in this chapter are reached:

```bash
clang -Xclang -ast-dump -fsyntax-only file.c
clang++ -Xclang -fdump-vtable-layouts -c file.cpp -o /dev/null
clang -Xclang -disable-llvm-passes -S -emit-llvm file.c -o -   # raw frontend IR
```

`-Xclang -disable-llvm-passes` is worth memorising: it emits the IR the
frontend produced *before* any LLVM pass ran, including the always-run cleanup
passes. It is how you tell "the frontend emitted this" apart from "an early
pass produced this" — a distinction that decides which project a bug belongs to.

## Inside phase 2

```
tokens ──▶ Parser ──▶ AST ──▶ Sema ──▶ (checked AST) ──▶ CodeGen ──▶ LLVM IR
   ▲                            │
Preprocessor              diagnostics
```

- **Lexer/Preprocessor** produces tokens. Macros, includes, and conditional
  compilation are finished here; nothing downstream can see them. `-E` stops
  after this phase, and `-E -dM` lists the macros that survived.
- **Parser** builds AST nodes as it goes, calling into Sema at each construct.
  Clang does not build a raw parse tree and then analyse it; parsing and
  semantic analysis are interleaved.
- **Sema** resolves names, performs overload resolution, applies implicit
  conversions, instantiates templates, checks types, and emits diagnostics. It
  *adds nodes* to the AST — this is why `-ast-dump` shows constructs the source
  does not contain.
- **CodeGen** (`clang/lib/CodeGen`) walks the checked AST and emits IR. It owns
  ABI lowering, aggregate layout, vtable emission, and destructor placement.
  `CodeGenFunction` handles one function body; `CodeGenModule` owns
  module-level state.

The important structural fact: **CodeGen never re-decides anything Sema
decided.** By the time IR is emitted, the overload is chosen, the conversion
sequence is fixed, and the template is instantiated. If the wrong function is
being called, the AST already says so.

## Where each flag actually acts

| Flag | Acts in |
| --- | --- |
| `-D`, `-I`, `-include` | preprocessor |
| `-std=`, `-f[no-]exceptions`, `-fno-rtti` | parser/Sema (and Sema's output shape) |
| `-target`, `-march=`, `-mcpu=` | driver → `-triple`/`-target-cpu`; changes ABI *and* backend |
| `-O0`..`-O3`, `-flto` | CodeGen (which pass pipeline to build) and the backend |
| `-g` | CodeGen (debug-info emission) |
| `-fsanitize=` | CodeGen (instrumentation) plus a runtime link job in the driver |
| `-mllvm <x>` | passed to LLVM's own `cl::opt` parsing, not to Clang |

`-mllvm` and `-Xclang` are frequently confused. `-Xclang` reaches the Clang
frontend; `-mllvm` reaches LLVM's command-line options — the same registry a
pass plugin registers into (see
[`../17-new-pass-manager/06-building-an-out-of-tree-pass.md`](../17-new-pass-manager/06-building-an-out-of-tree-pass.md)).

## Useful inspection commands

```bash
clang -ccc-print-phases -c file.c        # what jobs, in what order
clang -###                 -c file.c     # the exact -cc1 command
clang -E -dM               file.c        # surviving macro definitions
clang -Xclang -ast-dump -fsyntax-only file.c
clang -S -emit-llvm        file.c -o -   # IR after the frontend's own pipeline
clang -Xclang -disable-llvm-passes -S -emit-llvm file.c -o -   # raw frontend IR
clang -S                   file.c -o -   # target assembly
clang -Rpass=inline -O2 -c file.c        # optimization remarks
clang -fsave-optimization-record -O2 -c file.c   # machine-readable remarks (YAML)
```

## Pitfalls

- **Reading `-cc1` flags as stable API.** They change between releases without
  deprecation. Script the driver, not the frontend.
- **Assuming `-S -emit-llvm` is "what the frontend emitted".** At `-O0` Clang
  still runs a small always-on pipeline; at `-O2` it runs the full one. Use
  `-Xclang -disable-llvm-passes` when the distinction matters.
- **Blaming the optimizer for an ABI decision.** If the IR *signature* is wrong,
  the optimizer never had a chance; look at
  [`05-abi-and-target-lowering.md`](05-abi-and-target-lowering.md).
- **Forgetting that `-target` changes semantics.** It changes type sizes,
  alignments, argument classification, and default `char` signedness — not just
  which backend runs.
- **Comparing IR across optimization levels as if only speed changed.** `-O0`
  IR is deliberately naive (every local is an `alloca`); the differences are
  not defects.
- **Passing a `-cc1`-only flag without `-Xclang`.** The driver rejects it, or
  worse, treats it as an input filename.

## BCIR notes

- BCIR treats a resident toolchain as a *realization*, not a source of truth,
  so this pipeline is the thing BCIR plans **above**. Knowing exactly which
  stage owns a decision is what keeps a BCIR-level claim from silently
  duplicating (or contradicting) a frontend-level one.
- The `-###` habit transfers directly to the BCIR C-front rail: its diagnostics
  are Clang-style precisely so that a fallback-to-LLVM signal can be compared
  against the driver's own answer for the same input. See
  [`../../docs/languages/CFRONT_GUIDE.md`](../../docs/languages/CFRONT_GUIDE.md).

## See also

- [`02-ast-and-sema.md`](02-ast-and-sema.md) — the next stage in detail
- [`../07-optimization/05-debugging-passes.md`](../07-optimization/05-debugging-passes.md) — once the IR exists
- [`../12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) — phase 3
