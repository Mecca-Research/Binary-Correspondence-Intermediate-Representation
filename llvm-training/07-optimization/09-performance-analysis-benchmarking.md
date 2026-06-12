# Performance Analysis and Benchmarking

## Key takeaways

- Verify correctness before measuring speed: invalid IR, wrong results, or lost
  BCIR invariants invalidate every timing and counter collected afterward.
- Compare one controlled baseline with one controlled candidate, preserving the
  exact IR, pipeline, target, profile, model, workload, and environment for both.
- Keep IR-level evidence (instruction counts, CFG shape, remarks, metadata) apart
  from machine-code evidence (code size, layout, instructions, cycles, misses).
- Measure compile time and runtime separately; for JITs, split cold compilation,
  linking/materialization, warm-up, and hot invocation into distinct telemetry.
- PGO, MLGO, and BOLT are driven by profile or policy inputs. Report their input
  quality and provenance rather than treating their output as deterministic.
- For BCIR workloads, speed is not sufficient: record result correctness,
  event/claim throughput, mapping and diagnostic retention, hydration/execution
  phases, and telemetry-schema identity.

Performance analysis is an evidence pipeline, not a single stopwatch reading.
This lesson uses
[`examples/bcir-benchmark-kernel.ll`](examples/bcir-benchmark-kernel.ll) and the
companion
[`examples/bcir-benchmark-pipeline-notes.md`](examples/bcir-benchmark-pipeline-notes.md)
as a small, target-portable starting point. The fixture has no fixed data layout
or target triple, so IR experiments are portable; object and runtime experiments
must add and record one explicit target.

## What question is the experiment answering?

Choose the metric before choosing the command:

| Question | Primary evidence | Important controls |
| --- | --- | --- |
| Did a pass change the intended IR? | IR diff, verifier, remarks, metadata audit | Same input bytes and LLVM build |
| Did the pipeline become more expensive? | per-pass timing, wall time, peak RSS | Warm filesystem caches or deliberately cold runs, repeated processes |
| Did target code improve? | object size, disassembly, block layout, static scheduling analysis | Same triple, CPU, features, ABI, linker, relocation/code model |
| Did execution improve? | latency/throughput distributions and hardware counters | Same binary, workload, affinity, frequency and thermal state |
| Did profile guidance help? | profile coverage/representativeness, remarks, runtime counters | Same profile snapshot and matching build identifiers |
| Did BCIR behavior improve? | BCIR phase and workload telemetry plus correctness | Same workload/schema and preserved provenance |

A smaller IR file is not necessarily faster. A larger binary may be faster due to
inlining or alignment. Fewer retired instructions may lose to worse cache
behavior. State the hypothesis and use multiple evidence layers to explain the
result.

## Measuring pass-pipeline effects

Start with exact textual pipelines rather than vague labels such as "optimized":

```bash
input=llvm-training/07-optimization/examples/bcir-benchmark-kernel.ll

opt -passes=verify -disable-output "$input"
opt -S -passes='default<O1>' "$input" -o /tmp/bcir-baseline.ll
opt -S -passes='default<O2>' "$input" -o /tmp/bcir-candidate.ll
diff -u /tmp/bcir-baseline.ll /tmp/bcir-candidate.ll || true
```

For a focused experiment, vary only the pass under study:

```bash
opt -S -passes='verify,loop-simplify,loop-rotate,verify' \
  "$input" -o /tmp/bcir-loop-rotate.ll
```

Record both the pipeline spelling and `opt --version`. Default pipelines and
individual pass behavior can change across LLVM releases. Use
[`../17-new-pass-manager/`](../17-new-pass-manager/) for PassBuilder syntax,
analysis invalidation, plugins, and adaptive pipeline policy.

### Compile-time measurement

`-time-passes` reports time attributed by LLVM's pass instrumentation:

```bash
opt -S -passes='default<O2>' -time-passes \
  "$input" -o /tmp/bcir-o2.ll 2>/tmp/bcir-o2.time.txt
```

Availability and output details vary by LLVM build and release. If the installed
build offers another supported timing flow, such as time-trace output, use it
consistently for both variants and document the tool version. Pair pass timing
with repeated whole-process wall-time and peak-memory measurements: pass timing
does not include every startup, parsing, serialization, allocator, or operating
system cost.

### Optimization remarks

Remarks explain why profitable transformations happened or did not happen:

```bash
opt -S -passes='default<O2>' \
  -pass-remarks='.*' \
  -pass-remarks-missed='.*' \
  -pass-remarks-analysis='.*' \
  -pass-remarks-output=/tmp/bcir-o2.remarks.yaml \
  "$input" -o /tmp/bcir-o2-with-remarks.ll
```

Filters and serialization support are toolchain-dependent. Save the complete
record before narrowing it to a pass such as `loop-vectorize` or `inline`.
Remarks are decisions and explanations, not runtime proof; correlate them with
IR diffs, target code, and counters.

## Avoiding benchmark contamination

Control or report factors that can overwhelm the pipeline effect:

- pin the toolchain revision and command lines;
- use identical input bytes, workload sizes, random seeds, and result checks;
- separate deliberately cold-cache experiments from warmed steady-state runs;
- control CPU affinity, frequency scaling, turbo policy, thermal throttling,
  background load, NUMA placement, and allocator behavior where relevant;
- randomize or alternate baseline/candidate run order to reduce temporal bias;
- use enough repetitions to report a distribution, variance, and confidence,
  not merely the minimum or best run;
- avoid compiler output, logging, tracing, sanitizers, assertions, and profilers
  in only one variant;
- distinguish process startup and input generation from the measured kernel;
- keep target triple, CPU features, linker, libraries, environment variables,
  and data layout identical for machine-code comparisons.

A useful harness computes a checksum or compares complete output outside the
timed region. Fast wrong code is a failed experiment.

## IR-level metrics versus machine-code metrics

### IR-level evidence

IR evidence includes:

- instruction, basic-block, call-edge, and loop counts;
- scalar versus vector operations and vectorization width;
- inlining, unrolling, rotation, and CFG simplification visible in the IR;
- alias assumptions, branch weights, loop metadata, debug locations, and
  `!bcir.*` provenance or diagnostic attachments;
- analysis output and optimization remarks.

These metrics help explain optimizer behavior and remain useful before selecting
a target. They do **not** account for instruction selection, register allocation,
spills, scheduling, encoding size, linker relaxation, or final layout.

### Machine-code and runtime evidence

Once a target is fixed, lower both variants with identical settings:

```bash
llc -filetype=obj -mtriple=x86_64-unknown-linux-gnu -mcpu=<pinned-cpu> \
  /tmp/bcir-candidate.ll -o /tmp/bcir-candidate.o
llvm-size /tmp/bcir-candidate.o
llvm-objdump -d --no-show-raw-insn /tmp/bcir-candidate.o
```

Then measure code size, disassembly shape, spills, branches, cycles, retired
instructions, cache/TLB misses, branch misses, and runtime latency or throughput.
Counter names and event semantics are CPU-specific. Normalize derived ratios
carefully and retain raw counts, multiplexing/scaling information, and run time.
The lessons under [`../15-binary-analysis/`](../15-binary-analysis/) connect
static binary evidence with traces and hardware counters.

Do not lower and compare target-portable IR until the target choices are explicit.
Two modules with different data layouts can assign different sizes, alignments,
pointer arithmetic, ABI rules, and legal transformations even when their source
looks similar.

## Compile-time versus runtime tradeoffs

A more aggressive pipeline may improve a hot service kernel while making an
interactive JIT unusably slow. Measure at least:

- optimizer wall time and per-pass time;
- compiler peak memory;
- object/link time and cache hit behavior;
- cold-start or first-request latency;
- steady-state latency, tail latency, and throughput;
- code size and instruction-cache pressure; and
- break-even executions: extra compile cost divided by runtime saved per call.

Choose the objective for the deployment. AOT batch workloads, build farms,
short-lived command-line tools, and continuously running JIT services have
fundamentally different acceptable tradeoffs.

## PGO profile quality

PGO results are only as useful as the profile:

- **Representativeness:** does training cover production inputs, phases, errors,
  and rare but expensive paths?
- **Coverage:** which functions, blocks, edges, and value sites are absent or
  under-sampled?
- **Freshness:** does the profile match the source, IR, binary, build ID, target,
  and current workload mix?
- **Collection bias:** did instrumentation overhead or sampling frequency alter
  behavior or miss short-lived code?
- **Aggregation:** were incompatible hosts, tenants, phases, or schemas merged?
- **Stability:** do repeated profiles lead to similar hotness and decisions?

Pin and hash the profile input, retain merge commands, inspect profile-use
warnings and remarks, and run an unprofiled control. Do not call profile-guided
output deterministic unless the profile, merge order/controls, toolchain, target,
and build inputs are fixed. See
[`06-pgo-lto-bolt.md`](06-pgo-lto-bolt.md) for instrumentation, sample PGO, LTO,
and the existing BOLT/PGO command references.

## MLGO policy input

MLGO models influence profitability policy; they do not establish legality.
Record the model artifact or release, feature schema, advisor/configuration,
toolchain revision, target, and any fallback behavior. Compare against the same
pipeline without the learned advisor, and inspect decision remarks when the
integration exposes them.

A model trained on different code-size budgets, targets, or workload classes can
make internally valid but locally poor choices. BCIR verifier gates, metadata
contracts, and GAADMSF/HAM legality remain mandatory regardless of model score.
See
[`../17-new-pass-manager/05-mlgo-and-profile-guided-pipelines.md`](../17-new-pass-manager/05-mlgo-and-profile-guided-pipelines.md).

## BOLT and post-link analysis

BOLT operates after linking, where function order, basic-block placement, branch
reach, padding, and hot/cold splitting are concrete. Preserve the pre-BOLT
binary, post-BOLT binary, build IDs/symbols, relocation requirements, profile,
BOLT command, logs, and both binaries' counter results.

```bash
# Schematic only: profile collection and supported flags depend on the platform.
llvm-bolt /tmp/app -o /tmp/app.bolt -data=/tmp/app.fdata
llvm-size /tmp/app /tmp/app.bolt
llvm-objdump -d /tmp/app.bolt > /tmp/app.bolt.disasm.txt
```

An IR diff cannot reveal a BOLT-only layout change. Use
[`06-pgo-lto-bolt.md`](06-pgo-lto-bolt.md),
[`07-bolt-layout-walkthrough.md`](07-bolt-layout-walkthrough.md), and
[`../15-binary-analysis/`](../15-binary-analysis/) to relate profile provenance,
layout, symbols, disassembly, traces, and counters.

## JIT runtime telemetry

Do not report a JIT's first call as though it were hot kernel runtime. Instrument
separate timestamps and counters for:

1. IR admission and verification;
2. optimization;
3. code generation;
4. object linking and materialization;
5. symbol lookup and lazy-compilation trigger cost;
6. warm-up invocations;
7. warmed kernel invocations; and
8. reoptimization, replacement, and retired-code cleanup.

Key the record by module/kernel identity, generation, exact pipeline, target,
profile/model snapshot, code size, and telemetry schema. For remote ORC
execution, distinguish host compiler time, transport time, executor materialization,
and executor runtime. Snapshot adaptive-policy telemetry before compiling so a
single build does not read a moving input. See
[`../12-backend-jit/07-advanced-orc-runtime-integration.md`](../12-backend-jit/07-advanced-orc-runtime-integration.md)
for lazy compilation, hot re-JIT, resource tracking, and BCIR kernel lifecycle.

## BCIR workload-specific metrics

BCIR experiments should augment generic compiler metrics with workload evidence:

- verified result, checksum, or conformance-oracle agreement;
- kernels, events, claims, records, or bytes processed per second;
- latency by hydration, optimization, lowering, execution, M5 transduction, and
  telemetry-feedback phase where those phases participate in the workload;
- GEM hydration/execution counts and cache behavior;
- ROP/MAP front-end volume and failure/diagnostic rates;
- JIT/AOT lowering counts, code-cache occupancy, re-JIT count, and break-even;
- preservation or intentional consumption of BCIR register mappings;
- diagnostic/provenance attachment retention after cloning, folding, outlining,
  and lowering;
- HAM/GAADMSF policy decisions and the hardware/profile evidence that gated them;
- data-DNA telemetry lag, snapshot age, dropped events, and schema version.

Metric names must match the actual harness. If a phase is absent, mark it not
applicable rather than fabricating a zero. The cross-topic
[`../indexes/bcir-patterns.md`](../indexes/bcir-patterns.md) index points to the
relevant mapping, runtime, optimizer, and binary-analysis lessons.

## Recommended measurement workflow

1. **Verify IR.** Run `opt -passes=verify`; also run applicable BCIR invariant
   checks and a semantic result oracle.
2. **Run the baseline pipeline.** Save exact input/output IR, tool version,
   pipeline, timing, remarks, metadata counts, and profile/model identity.
3. **Run the candidate pipeline.** Change only the intended variable and capture
   the same artifacts.
4. **Diff IR.** Inspect CFG, loops, calls, memory operations, poison-sensitive
   rewrites, and `!bcir.*`, `!prof`, alias, loop, and debug metadata.
5. **Inspect optimization remarks.** Explain performed, missed, and analyzed
   transformations; do not treat remarks as performance measurements.
6. **Lower to object if target-portable.** First pin the triple, CPU, features,
   ABI/data layout, relocation/code model, linker, and libraries; preserve object
   size and disassembly evidence.
7. **Collect runtime counters.** Validate results, isolate timed regions, separate
   JIT phases, repeat runs, and collect distributions plus raw hardware counts.
8. **Compare telemetry.** Join compile-time, IR, machine-code, runtime, profile,
   and BCIR-specific records by stable experiment and artifact identifiers.

The companion
[`examples/bcir-benchmark-pipeline-notes.md`](examples/bcir-benchmark-pipeline-notes.md)
provides a fill-in command skeleton and results table.

## Pitfalls

- **Benchmarking unverifiable IR:** optimizer behavior after malformed IR is not
  evidence. Verify before and after relevant transforms.
- **Comparing different data layouts:** pointer size, alignment, ABI, and legal
  lowering may differ; reject the comparison or make target differences the
  explicit independent variable.
- **Measuring cold JIT compile time as hot kernel runtime:** report verification,
  optimization, codegen, materialization, warm-up, and invocation separately.
- **Treating profile-guided output as deterministic without profile controls:**
  pin profile bytes, merge procedure, model/policy input, toolchain, and target.
- **Ignoring metadata loss:** a speedup that loses required BCIR diagnostics,
  provenance, register correspondence, alias facts, branch weights, loop hints,
  or debug locations may be invalid or impossible to diagnose.
- **Using IR size as a runtime verdict:** instruction selection and layout can
  reverse the apparent result.
- **Changing multiple variables:** a new pipeline, target CPU, profile, linker,
  and workload in one comparison cannot isolate a cause.
- **Reporting only the best run:** show the distribution, discarded-run policy,
  and environmental controls.

## Related reading

- [`06-pgo-lto-bolt.md`](06-pgo-lto-bolt.md) — PGO, LTO/ThinLTO, and BOLT.
- [`../12-backend-jit/07-advanced-orc-runtime-integration.md`](../12-backend-jit/07-advanced-orc-runtime-integration.md) — ORC runtime telemetry and hot re-JIT.
- [`../15-binary-analysis/`](../15-binary-analysis/) — static and dynamic binary evidence.
- [`../17-new-pass-manager/`](../17-new-pass-manager/) — pipeline construction, plugins, MLGO, and profile policy.
- [`../indexes/bcir-patterns.md`](../indexes/bcir-patterns.md) — BCIR concepts across the corpus.
