# BCIR benchmark pipeline experiment notes

Use this worksheet with [`bcir-benchmark-kernel.ll`](bcir-benchmark-kernel.ll).
It records *how* to compare pipelines; it does not contain portable performance
claims. Replace tool paths, target flags, and counter names with values from one
pinned LLVM toolchain and one controlled benchmark host.

## Experiment identity

| Field | Record before running |
| --- | --- |
| LLVM tools | `opt --version`, `llc --version`, compiler build/revision |
| Host and target | CPU model, OS/kernel, target triple, `-mcpu`/`-mattr` |
| Inputs | Fixture hash, element count, value distribution, random seed |
| Pipelines | Exact baseline and candidate `-passes=` strings |
| Profiles/models | PGO profile hash and provenance; MLGO model/config hash |
| Runtime controls | Affinity, frequency policy, warm-up count, repetitions |
| BCIR schema | Workload, provenance, diagnostics, and telemetry versions |

## Reproducible command skeleton

```bash
input=llvm-training/07-optimization/examples/bcir-benchmark-kernel.ll
baseline=/tmp/bcir-baseline.ll
candidate=/tmp/bcir-candidate.ll

# Legality first.
opt -passes=verify -disable-output "$input"

# Preserve textual outputs and collect separate pass timing reports.
opt -S -passes='default<O1>' -time-passes "$input" -o "$baseline" \
  2>/tmp/bcir-baseline.time.txt
opt -S -passes='default<O2>' -time-passes "$input" -o "$candidate" \
  2>/tmp/bcir-candidate.time.txt

# Compare structure and retained BCIR metadata.
diff -u "$baseline" "$candidate" || true
rg '!bcir\\.|!prof|!llvm.loop' "$baseline" "$candidate"

# Capture optimization decisions. Pass-name filters are optional and
# toolchain-dependent; start broad, then narrow a noisy report.
opt -S -passes='default<O2>' \
  -pass-remarks='.*' -pass-remarks-missed='.*' \
  -pass-remarks-analysis='.*' \
  -pass-remarks-output=/tmp/bcir-candidate.remarks.yaml \
  "$input" -o /tmp/bcir-candidate-with-remarks.ll

# Only after choosing and recording a target. Do not compare this object with an
# object produced for a different triple, CPU, feature set, relocation model, or
# code model.
llc -filetype=obj -mtriple=x86_64-unknown-linux-gnu -mcpu=<pinned-cpu> \
  "$candidate" -o /tmp/bcir-candidate.o
llvm-size /tmp/bcir-candidate.o
llvm-objdump -d --no-show-raw-insn /tmp/bcir-candidate.o \
  > /tmp/bcir-candidate.disasm.txt
```

If a particular `opt` build does not expose `-time-passes`, use that release's
supported pass-timing interface (for example a debug/timing-enabled build or
`-time-trace` where available), and record the exact command. Never silently
mix timing mechanisms between baseline and candidate.

## Results table

| Metric | Baseline | Candidate | Delta | Scope and units |
| --- | ---: | ---: | ---: | --- |
| `opt` wall time | | | | compile-time, repeated process runs |
| Peak compiler RSS | | | | compile-time |
| LLVM instruction count | | | | IR-level, same pipeline endpoint |
| Basic blocks / loop shape | | | | IR-level |
| BCIR metadata retained | | | | attachment counts plus semantic audit |
| `.text` size | | | | object/binary-level |
| Retired instructions | | | | runtime counter |
| Cycles | | | | runtime counter |
| Branch misses | | | | runtime counter |
| L1/LLC misses | | | | runtime counter |
| Kernel latency | | | | median plus distribution |
| JIT materialization time | | | | cold compile/link only |
| Hot invocation time | | | | warmed kernel only |
| BCIR result/checksum | | | | correctness oracle |
| BCIR events/claims per second | | | | workload throughput |
| BCIR diagnostic/provenance loss | | | | count and affected IDs |

## Interpretation checklist

- Reject the run if verification, result checks, or BCIR invariant checks fail.
- Attribute compile-time changes to pass timing only after controlling process
  startup, caches, and identical input bytes.
- Treat IR counts as explanations, not as substitutes for target code and
  runtime measurements.
- Keep cold JIT compilation/materialization separate from warmed invocation.
- Confirm that PGO profiles and MLGO policy inputs match the measured workload.
- Investigate missing `!bcir.*`, `!prof`, loop, alias, or debug metadata before
  accepting an apparent speedup.
- Report distributions and run-to-run variance, not only the best observation.
