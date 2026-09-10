# BCIR register-binding pass plugin (buildable)

A complete out-of-tree LLVM pass plugin. Unlike the `.cpp.md` sketches in this
chapter's `examples/` directory, this one **compiles, loads, and runs**, and a
gate script asserts its behaviour.

Chapter: [`../../06-building-an-out-of-tree-pass.md`](../../06-building-an-out-of-tree-pass.md)

## Files

| File | Kind |
| --- | --- |
| `BCIRRegisterBinding.cpp` | plugin source: one analysis, two transforms, one printer, and the registration entry point |
| `CMakeLists.txt` | standalone out-of-tree CMake project |
| `binding-ok.ll` | known-good IR; the checker reports zero violations |
| `binding-collision.ll` | known-good IR that the *checker* rejects: two values claim one register |
| `unroll-breaks-binding.ll` | known-good IR that a stock LLVM pass breaks |

All three `.ll` files are valid LLVM IR and are part of the known-good example
manifest — they assemble with `llvm-as` and pass `opt -passes=verify`. That is
the point: the invariant they violate is a BCIR-level contract, not an LLVM
one, so LLVM's own verifier has nothing to say about it.

## What the plugin registers

| Name | IR unit | Effect |
| --- | --- | --- |
| `bcir-verify-bindings` | module | Reports every source register bound to more than one value; exit code unchanged |
| `bcir-verify-bindings<strict>` | module | Same report, nonzero exit on any violation |
| `bcir-strip-bindings` | function | Drops every `!bcir.reg`; models the stage that consumes the contract |
| `print<bcir-bindings>` | function | Prints the analysis result |
| `bcir-binding-analysis` | function | The analysis itself (requested by the passes above) |

Flag: `-bcir-verify-at-pipeline-start` also runs the checker at the
`PipelineStart` extension point. It is off by default so that loading the
library does not change an unrelated `-O2` run.

## Build

Requires LLVM **development** files (`llvm-dev` or an installed LLVM tree),
CMake, and a C++17 compiler. The plugin must be built against the same LLVM
major version as the `opt` that will load it.

```bash
cmake -S training/llvm/17-new-pass-manager/examples/pass-plugin \
      -B build/bcir-pass-plugin \
      -DLLVM_DIR="$(llvm-config --cmakedir)"
cmake --build build/bcir-pass-plugin -j2
```

## Run

```bash
LIB=build/bcir-pass-plugin/libBCIRRegisterBinding.so
D=training/llvm/17-new-pass-manager/examples/pass-plugin

opt -load-pass-plugin=$LIB -passes='bcir-verify-bindings' -disable-output $D/binding-ok.ll
# bcir-verify-bindings: 4 binding(s), 0 violation(s)

opt -load-pass-plugin=$LIB -passes='bcir-verify-bindings' -disable-output $D/binding-collision.ll
# bcir-verify-bindings: violation in function 'kernel': register 'r2' is bound to more than one value
# bcir-verify-bindings: 4 binding(s), 1 violation(s)

opt -load-pass-plugin=$LIB \
    -passes='function(loop(loop-unroll-full)),bcir-verify-bindings' \
    -disable-output $D/unroll-breaks-binding.ll
# bcir-verify-bindings: 10 binding(s), 7 violation(s)
```

## Gate

```bash
./training/llvm/tools/build-pass-plugin.sh
```

Builds the plugin and runs nine checks, including the negative cases (the
checker fires on a violating module; `<strict>` exits nonzero). It **skips
cleanly**, printing the reason and exiting 0, when the LLVM headers are missing
or when `opt` and `llvm-config` report different LLVM major versions. A skip is
always labelled as a skip, never as a pass.

Override the build directory with `PASS_PLUGIN_BUILD_DIR`, and the parallelism
with `PASS_PLUGIN_JOBS` (default 2).

## Observed evidence

Built and run against **LLVM 18.1.3** on x86-64 Linux: 9/9 checks pass, and the
gate was confirmed to fail when the collision is removed from
`binding-collision.ll`. Counts in `unroll-breaks-binding.ll` depend on the LLVM
version's unrolling decision, so the gate asserts the stable shape — zero
violations before unrolling, more than zero after — rather than pinning `7`.
