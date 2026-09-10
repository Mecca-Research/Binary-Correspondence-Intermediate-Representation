# Building an Out-of-Tree Pass: Source, Build, Run, Test

Chapters [`02`](02-custom-passes-and-analyses.md) and
[`03`](03-passbuilder-callbacks-and-plugins.md) describe the shapes. This
chapter closes the loop with a **complete plugin that is checked in, built, and
exercised by a gate**: [`examples/pass-plugin/`](examples/pass-plugin).

Everything below was observed on **LLVM 18.1.3**. Counts and diagnostic
spellings shift between releases; the structure does not.

## What the plugin does, and why that problem

BCIR's 1:1 register-correspondence contract says: an instruction may carry
`!bcir.reg !{!"rN"}` naming the source register it realizes, and within one
function no two instructions may claim the same register.

LLVM has no opinion about that invariant. `opt -passes=verify` accepts a module
that violates it, because it is not an LLVM law. And the stock pipeline breaks
it routinely — any pass that duplicates an instruction duplicates its metadata:

```console
$ opt -load-pass-plugin=$LIB -passes='bcir-verify-bindings' -disable-output \
    llvm-training/17-new-pass-manager/examples/pass-plugin/unroll-breaks-binding.ll
bcir-verify-bindings: 3 binding(s), 0 violation(s)

$ opt -load-pass-plugin=$LIB \
    -passes='function(loop(loop-unroll-full)),bcir-verify-bindings' -disable-output \
    llvm-training/17-new-pass-manager/examples/pass-plugin/unroll-breaks-binding.ll
bcir-verify-bindings: 10 binding(s), 7 violation(s)
```

That is the case for owning a pass: an invariant nobody else enforces, broken by
a transform nobody would call buggy, detected by nothing until you write the
detector.

## The four pieces

### 1. An analysis with a key, a result, and an `invalidate` hook

```cpp
class BCIRBindingAnalysis : public AnalysisInfoMixin<BCIRBindingAnalysis> {
  friend AnalysisInfoMixin<BCIRBindingAnalysis>;
  static AnalysisKey Key;                     // the identity; one per analysis
public:
  using Result = BCIRBindingInfo;
  Result run(Function &F, FunctionAnalysisManager &);
  static StringRef name() { return "bcir-binding-analysis"; }
};

AnalysisKey BCIRBindingAnalysis::Key;         // exactly one definition
```

The `Result` decides its own survival:

```cpp
bool BCIRBindingInfo::invalidate(Function &, const PreservedAnalyses &PA,
                                 FunctionAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<BCIRBindingAnalysis>();
  return !PAC.preserved() && !PAC.preservedSet<AllAnalysesOn<Function>>();
}
```

Read it as: *stay alive only if a pass explicitly preserved me, or preserved
every function analysis.* Omitting `invalidate` gets you the conservative
default, which is safe. Writing a **wrong** one — returning `false`
unconditionally, say — is a stale-analysis bug that will be blamed on the
transform, not on this function.

### 2. A module pass that reaches function analyses through the proxy

```cpp
FunctionAnalysisManager &FAM =
    MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
...
const BCIRBindingInfo &Info = FAM.getResult<BCIRBindingAnalysis>(F);
```

The proxy is what `crossRegisterProxies` sets up in a hand-built pipeline, and
it is the only supported way for a module pass to ask a function question. A
module pass that caches raw `FunctionAnalysisManager` state across IR changes is
building its own stale-analysis bug.

### 3. A function transform with an honest `PreservedAnalyses`

```cpp
PreservedAnalyses BCIRStripBindingsPass::run(Function &F, FunctionAnalysisManager &) {
  bool Changed = /* drop every !bcir.reg */;
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA = PreservedAnalyses::all();
  PA.abandon<BCIRBindingAnalysis>();
  return PA;
}
```

`all()` minus one analysis is correct **here** and only here, because
`bcir.reg` is a custom kind no in-tree analysis reads. The same shape would be a
miscompile if the metadata being dropped were `!range`, `!alias.scope`,
`!noalias`, or `!prof` — those feed analyses whose results would then be served
stale. The rule:

| What the pass changed | Correct answer |
| --- | --- |
| Nothing | `PreservedAnalyses::all()` |
| Only custom metadata nothing else reads | `all()`, then `abandon<YourAnalysis>()` |
| Instructions, not the CFG | `PreservedAnalyses(); PA.preserveSet<CFGAnalyses>()` |
| The CFG | `PreservedAnalyses::none()` |
| Not sure | `PreservedAnalyses::none()` — slow and correct |

Returning too much is a correctness bug. Returning too little costs time.
Those are not symmetric, and the asymmetry is the whole decision rule.

### 4. Registration

```cpp
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "BCIRRegisterBinding", LLVM_VERSION_STRING,
          registerCallbacks};
}
```

`registerCallbacks` installs four different things, and each one answers a
different question:

| Callback | Answers |
| --- | --- |
| `registerAnalysisRegistrationCallback` | "how does the pipeline get my analysis at all?" |
| `registerPipelineParsingCallback` (module) | "what does `-passes=bcir-verify-bindings` mean?" |
| `registerPipelineParsingCallback` (function) | same, inside `function(...)` |
| `registerPipelineStartEPCallback` | "how do I run without being named in `-passes`?" |

Two details that are easy to get wrong:

- **Parsing callbacks are per IR unit.** A module-level callback never sees a
  name written inside `function(...)`, and vice versa. Register the name in the
  manager that will actually hold the pass.
- **An extension point that always fires changes unrelated builds.** The plugin
  puts its `PipelineStartEP` behind a `cl::opt` that defaults to off, so loading
  the library into an ordinary `-O2` run adds nothing:

```cpp
cl::opt<bool> VerifyAtPipelineStart("bcir-verify-at-pipeline-start", ...,
                                    cl::init(false), cl::Hidden);
```

### Name every pass after the name it registers

`PassInfoMixin` derives a pass's display name from its C++ type. Without an
override you get this:

```console
$ opt -load-pass-plugin=$LIB -passes='bcir-verify-bindings,function(bcir-strip-bindings)' \
    --print-pipeline-passes -disable-output ...
Could not parse dumped pass pipeline: unknown pass name '{anonymous}::BCIRVerifyBindingsPass'
{anonymous}::BCIRVerifyBindingsPass,function({anonymous}::BCIRStripBindingsPass),verify
```

The dumped pipeline cannot be fed back into `-passes=`, which defeats the point
of dumping it. One line per class fixes it:

```cpp
static StringRef name() { return "bcir-verify-bindings"; }
```

```console
bcir-verify-bindings,function(bcir-strip-bindings),verify
```

Do the same on the analysis class, or every invalidation line in
`-debug-pass-manager` output stays a mangled type name.

## Build integration

The plugin is a **standalone CMake project**, deliberately not wired into the
training corpus's own `CMakeLists.txt`: building it needs LLVM development
headers, and the core corpus gates only assume `llvm-as` and `opt`.

```cmake
find_package(LLVM REQUIRED CONFIG)

add_library(BCIRRegisterBinding MODULE BCIRRegisterBinding.cpp)
set_target_properties(BCIRRegisterBinding PROPERTIES CXX_STANDARD 17 ...)
target_include_directories(BCIRRegisterBinding SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(BCIRRegisterBinding PRIVATE ${LLVM_DEFINITIONS})

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(BCIRRegisterBinding PRIVATE -fno-rtti)
endif()

if(APPLE)
  target_link_options(BCIRRegisterBinding PRIVATE -Wl,-undefined,dynamic_lookup)
endif()
```

Four facts behind those lines:

- **`MODULE`, not `SHARED`.** The plugin is `dlopen`ed by `opt`, never linked
  against.
- **It links against no LLVM libraries.** Symbols resolve from the loading
  process. Linking `LLVMCore` into a plugin gives you two copies of LLVM's
  global state, and the failure looks like anything but that.
- **RTTI must match.** LLVM is normally built `-fno-rtti`. A plugin compiled
  with RTTI against an `-fno-rtti` LLVM links and loads and then misbehaves
  where you least expect it. `LLVM_ENABLE_RTTI` comes from `LLVMConfig.cmake`,
  so the build asks rather than guesses.
- **macOS needs `-undefined dynamic_lookup`.** ELF platforms tolerate
  undefined symbols in a shared module; Mach-O does not, unless told.

```bash
cmake -S llvm-training/17-new-pass-manager/examples/pass-plugin \
      -B build/bcir-pass-plugin \
      -DLLVM_DIR="$(llvm-config --cmakedir)"
cmake --build build/bcir-pass-plugin -j2
```

## Running it

```bash
LIB=build/bcir-pass-plugin/libBCIRRegisterBinding.so

# check a module
opt -load-pass-plugin=$LIB -passes='bcir-verify-bindings' -disable-output in.ll

# fail the build on a violation
opt -load-pass-plugin=$LIB -passes='bcir-verify-bindings<strict>' -disable-output in.ll

# print the analysis result
opt -load-pass-plugin=$LIB -passes='function(print<bcir-bindings>)' -disable-output in.ll

# bracket a destructive stage
opt -load-pass-plugin=$LIB \
    -passes='bcir-verify-bindings,function(loop(loop-unroll-full)),bcir-verify-bindings' \
    -disable-output in.ll
```

Observed on the checked-in fixtures:

```console
$ opt ... -passes='bcir-verify-bindings' ... binding-ok.ll
bcir-verify-bindings: 4 binding(s), 0 violation(s)

$ opt ... -passes='bcir-verify-bindings' ... binding-collision.ll
bcir-verify-bindings: violation in function 'kernel': register 'r2' is bound to more than one value
bcir-verify-bindings:   second binding:  %r2.cold = mul i32 %r1, 3, !bcir.reg !1
bcir-verify-bindings: 4 binding(s), 1 violation(s)
```

Report-only mode exits `0`; `<strict>` exits nonzero via `report_fatal_error`.
That split matters: a checker whose verdict cannot change an exit status is a
log line, not a gate.

## Watching invalidation actually happen

`-debug-pass-manager` is the tool that answers "did my analysis get recomputed,
and why":

```console
$ opt -load-pass-plugin=$LIB \
    -passes='function(print<bcir-bindings>,bcir-strip-bindings,print<bcir-bindings>)' \
    -debug-pass-manager -disable-output binding-ok.ll
Running pass: print<bcir-bindings> on kernel (4 instructions)
Running analysis: bcir-binding-analysis on kernel
Running pass: bcir-strip-bindings on kernel (4 instructions)
Invalidating analysis: bcir-binding-analysis on kernel
Running pass: print<bcir-bindings> on kernel (4 instructions)
Running analysis: bcir-binding-analysis on kernel
```

Computed, invalidated by the transform that changed its input, recomputed. If
the second `Running analysis` line were missing, the second printer would be
reading a result describing IR that no longer exists — and it would print
plausible numbers while doing it.

| Flag | Question it answers |
| --- | --- |
| `-debug-pass-manager` | what ran, in what order, what was invalidated |
| `--print-pipeline-passes` | what pipeline did my flags actually build |
| `-print-after-all` / `-print-before=<pass>` | what did the IR look like around a pass |
| `-opt-bisect-limit=N` | which pass number first introduces the bad behaviour |
| `-passes='...,verify,...'` | which pass first produces invalid IR |

`-opt-bisect-limit` plus a checker pass is the fastest route from "the output is
wrong" to "pass N did it": bisect on the limit, then re-run with the checker
bracketing that pass.

## Testing a pass

Three layers, in increasing cost:

1. **A gate script.** [`tools/build-pass-plugin.sh`](../tools/build-pass-plugin.sh)
   builds the plugin and asserts nine behaviours, including the negative one —
   that the checker fires on a violating module and that `<strict>` exits
   nonzero. It skips cleanly (exit 0, with the reason printed) when LLVM
   development headers are missing or when `opt` and `llvm-config` disagree on
   the LLVM major version, because a plugin can only be loaded by an `opt` from
   its own major.

2. **`lit` + `FileCheck`,** the in-tree convention:

   ```llvm
   ; RUN: opt -load-pass-plugin=%bcirplugin -passes='bcir-verify-bindings' \
   ; RUN:     -disable-output %s 2>&1 | FileCheck %s
   ; CHECK: 4 binding(s), 1 violation(s)
   ```

   Use `CHECK-NOT` for the property that must *not* appear, and `--check-prefix`
   to run the same file under several pipelines.

3. **Unit tests** against the analysis result, when the interesting logic is in
   the `Result` rather than in the IR walk.

Whichever layer you use, write the failing case first. A test that only asserts
the clean module passes will keep passing against a pass that does nothing at
all — which is exactly the state a refactor leaves it in.

## Version and portability boundary

- A plugin is loadable only by tools from the **same LLVM major version**. The
  ABI is not stable across majors, and `LLVM_PLUGIN_API_VERSION` is checked at
  load time.
- `-load-pass-plugin` is the new-PM flag. `-load` is the legacy-PM flag, and the
  legacy pass manager is gone for the optimization pipeline.
- Static registration into a build of `opt` is an alternative to `dlopen`; the
  registration code is identical, only the entry point differs.
- On Windows, plugins need explicit symbol export and a matching runtime; treat
  a Windows plugin build as its own port, not as a rebuild.

## Pitfalls

- **Two definitions (or zero) of `AnalysisKey Key`.** One out-of-line definition
  per analysis; a missing one is a link error, a duplicated one is worse.
- **Registering the pass name with the wrong manager.** A module name written
  inside `function(...)` simply is not found, and the error says "unknown pass",
  which sends people looking at the wrong thing.
- **Forgetting `registerAnalysisRegistrationCallback`.** `getResult<YourAnalysis>`
  then aborts at run time, not at pipeline-parse time.
- **Linking LLVM libraries into the plugin.** Duplicate global state; symptoms
  nowhere near the cause.
- **RTTI/exception mismatch with the host LLVM.** Ask `LLVMConfig.cmake`.
- **An always-on extension point.** Turns your plugin into a global side effect
  of loading a library.
- **Missing `name()`.** Breaks `--print-pipeline-passes` round-tripping and
  makes `-debug-pass-manager` output unreadable.
- **`PreservedAnalyses::all()` after mutating IR.** The most expensive mistake
  in this chapter, and the quietest.

## BCIR notes

- A BCIR pipeline should bracket every destructive stage with the checker, not
  run it once at the end: the report tells you *that* correspondence broke, and
  only the bracket tells you *which pass* broke it.
- `<strict>` belongs in CI; report-only belongs in developer runs. A gate that
  cannot fail the build is documentation.
- The checker is a **legality** verdict, so it must not consult a measured or
  learned input. That is the two-truth separation the repository enforces
  elsewhere: learned data may rank and calibrate, and never decides a verdict.
  See [`04-adaptive-bcir-pipelines.md`](04-adaptive-bcir-pipelines.md) for the
  policy side of the same pipeline.
- When a lowering stage legitimately consumes the 1:1 contract, run
  `bcir-strip-bindings` there and record the replacement mapping. Leaving stale
  bindings behind means every later pass is measured against a contract that no
  longer applies.

## See also

- [`examples/pass-plugin/README.md`](examples/pass-plugin/README.md) — build and run commands for the checked-in plugin
- [`02-custom-passes-and-analyses.md`](02-custom-passes-and-analyses.md) — pass and analysis shapes
- [`03-passbuilder-callbacks-and-plugins.md`](03-passbuilder-callbacks-and-plugins.md) — the callback catalog
- [`../07-optimization/05-debugging-passes.md`](../07-optimization/05-debugging-passes.md) — bisecting a bad pipeline
- [`../exercises/038-custom-pass-bcir-invariants.solution.md`](../exercises/038-custom-pass-bcir-invariants.solution.md) — the review-level exercise this chapter makes executable
