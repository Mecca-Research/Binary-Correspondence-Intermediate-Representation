# PGO, LTO, ThinLTO, and BOLT

Modern production binaries are often shaped by profile-guided optimization (PGO),
link-time optimization (LTO/ThinLTO), and post-link layout tools such as BOLT.
An agent that only reads pre-optimization IR can miss why the final binary has a
different call graph, block order, branch layout, and performance profile.

## Mental model

```text
source / frontend IR
  -> instrumentation or sampled profile collection
  -> PGO-weighted IR optimization
  -> LTO or ThinLTO cross-module import/inlining
  -> code generation and linking
  -> post-link profile collection
  -> BOLT function/basic-block layout and binary rewriting
```

Each phase can change the evidence available to binary analysis:

- PGO changes branch weights, hot/cold splitting, inlining decisions, and loop
  decisions.
- Full LTO exposes whole-program definitions, enabling aggressive cross-module
  inlining and dead-stripping.
- ThinLTO imports summaries selectively, trading full visibility for scalable
  distributed builds.
- BOLT works after linking, using binary-level profiles to reorder functions and
  basic blocks and improve I-cache and branch locality.

## PGO in LLVM IR

PGO often appears as metadata attached to branches, switches, indirect-call
promotion sites, or function entry counts:

```llvm
br i1 %cond, label %hot, label %cold, !prof !0

!0 = !{!"branch_weights", i32 10000, i32 3}
```

The numbers are relative weights, not a semantic guarantee. They tell the
optimizer which path was hot in the training workload. A stale or unrepresentative
profile can make the final binary worse for a different workload.

## Why PGO + LTO can be non-monotonic

Adding more optimization is not always additive. Cross-module inlining can expose
scalar optimizations and remove call overhead, but it can also:

- grow hot functions until instruction-cache locality degrades;
- duplicate code across call sites;
- perturb block layout and branch prediction;
- specialize for a profile that does not match deployment inputs;
- hide function boundaries that a BCSA pipeline expected to match.

Treat "PGO + LTO" as a configuration to measure, not as a theorem that dominates
PGO-only or non-LTO builds on every microarchitecture.

## ThinLTO review points

ThinLTO leaves useful artifacts for an agent to reason about:

| Artifact | What to inspect |
| --- | --- |
| Module summaries | Which functions were importable, hot, or externally visible. |
| Import lists | Which cross-module callees became visible to local optimization. |
| Inline remarks | Whether a hot callsite was inlined or rejected. |
| Cache keys | Whether an incremental build reused stale summaries or objects. |
| Final symbol table | Which functions survived internalization and dead stripping. |

When reverse-engineering an optimized binary, expect function boundaries and
callsite counts to differ from the original translation units.

## BOLT and post-link layout

BOLT consumes a linked binary plus profile data. It can reorder functions, split
hot and cold basic blocks, align hot loops, and rewrite branches. These changes
happen after LLVM IR is gone, so IR-only analysis cannot see the final layout.
For a concrete fixture and inspection flow, see
[`07-bolt-layout-walkthrough.md`](07-bolt-layout-walkthrough.md) and
[`examples/bolt-layout-demo.c`](examples/bolt-layout-demo.c).

Questions to ask:

- Was the binary built with relocations or metadata needed for post-link
  rewriting?
- Which profile drove BOLT: instrumentation, sampling, or stale data?
- Did hot/cold splitting move rare error blocks away from the main trace?
- Did function reordering change addresses used by a similarity or patching
  pipeline?

## Agent workflow

1. Record the full build matrix: baseline, PGO-only, LTO-only, PGO+LTO,
   ThinLTO, and BOLT variants where available.
2. Preserve optimization remarks, profile summaries, and final symbol maps.
3. Compare IR-level branch weights with binary-level block execution counts.
4. Compare code size and I-cache counters across configurations before declaring
   one pipeline better.
5. For BCSA, label feature vectors with the optimization pipeline so models do
   not confuse compiler configuration with semantic difference.

## CI optimization smoke matrix

The repository CI has a dedicated `llvm-training-optimization-smoke` job that
keeps this chapter's optimization guidance executable without turning every pull
request into a full performance lab. It uses the small `bcir-tools` executable as
the smoke target and caches compiler outputs with `ccache` before CMake
configuration.

| Matrix entry | Scope | Training/rebuild behavior |
| --- | --- | --- |
| `thinlto` | Configures Clang with `-flto=thin`, links with LLD, builds only `bcir-tools`, and runs `bcir-tools --runtime-diag`. | Verifies that the project still accepts a ThinLTO path and that a tiny optimized executable starts successfully. |
| `pgo` | Builds an instrumented `bcir-tools`, runs `bcir-tools --runtime-diag` as the training command, merges `*.profraw` with `llvm-profdata`, rebuilds with `-fprofile-use`, and runs the rebuilt executable. | Exercises the minimum PGO lifecycle: generate profile data, merge it, consume it, and keep the smoke workload intentionally small. |

The main `build-and-test` job also enables CMake compiler launchers when
`ccache` is available. The optimization matrix is intentionally separate from the
full test suite so PGO and ThinLTO regressions are visible while CI latency stays
bounded.

## Commands to recognize

```bash
# Instrumented PGO sketch.
clang -O2 -fprofile-generate=/tmp/prof app.c -o /tmp/app-instr
LLVM_PROFILE_FILE=/tmp/prof/%p.profraw /tmp/app-instr
llvm-profdata merge /tmp/prof/*.profraw -o /tmp/app.profdata
clang -O2 -fprofile-use=/tmp/app.profdata app.c -o /tmp/app-pgo

# ThinLTO sketch.
clang -O2 -flto=thin -fprofile-use=/tmp/app.profdata app.c -o /tmp/app-thinlto-pgo

# BOLT sketch; exact flags depend on platform and collected profile format.
llvm-bolt /tmp/app-thinlto-pgo -o /tmp/app-bolt -data=/tmp/app.fdata
```

The commands are intentionally schematic. Use the project build system's real
compiler, linker, target CPU, and profile-collection mode when producing data for
review.
