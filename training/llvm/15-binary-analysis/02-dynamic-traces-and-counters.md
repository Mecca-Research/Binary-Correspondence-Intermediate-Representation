# Dynamic Traces and Hardware Counters

Dynamic evidence gives an agent a way to test whether the final executable
behaves like the static IR suggested. The goal is not to replace IR analysis; it
is to attach runtime observations to the same functions, blocks, branches, and
memory-access families.

## Minimal data model

A useful record links one binary run back to static structure:

| Field | Meaning |
| --- | --- |
| `binary_id` | Hash or build identifier for the executable under test. |
| `function` | Symbol, debug name, or recovered function ID. |
| `input_class` | Public label for the run, such as `all_zero_secret` or `random_secret`. |
| `basic_blocks` | Ordered or sampled block IDs observed during execution. |
| `cycles` | Retired-cycle or wall-clock proxy, collected consistently. |
| `instructions` | Retired instruction count. |
| `branches`, `branch_misses` | Branch predictor behavior. |
| `cache_misses`, `l1d_misses`, `llc_misses` | Memory hierarchy signals. |
| `notes` | Tool, CPU, pinning, warmup, and noise controls. |

See [`examples/dynamic-trace-sample.csv`](examples/dynamic-trace-sample.csv) and
[`examples/perf-counter-sample.csv`](examples/perf-counter-sample.csv) for tiny
schemas.

## Pairing traces with IR

1. Preserve stable function names or emit a map from IR functions to object
   symbols.
2. Keep block labels when possible; otherwise emit debug locations or remarks that
   help map machine blocks back to IR regions.
3. Store build configuration next to traces: target triple, CPU, optimization
   level, LTO mode, PGO profile, and post-link tools.
4. Compare runs by **input class pairs**, not by isolated measurements.
5. Keep raw data; summarize separately so reviewers can revisit noise controls.

## Counter interpretation cautions

- Counters are CPU-specific. Do not compare raw event names across vendors or
  microarchitectures without a mapping layer.
- Sampling can miss rare paths. Use deterministic tracing or instrumentation for
  small security-critical kernels when possible.
- Branch misses can reveal secret-dependent control flow, but a constant branch
  trace does not prove constant-time memory behavior.
- Cache misses can reveal address-dependent leaks, but prefetchers and co-tenancy
  can add noise.
- PGO and BOLT may intentionally change layout; always record which binary was
  measured.

## Example review pattern

| Static IR finding | Dynamic follow-up |
| --- | --- |
| Secret-derived `br` | Check path traces and branch misses across secret classes. |
| Secret-derived GEP | Check cache/TLB events and load-latency distributions. |
| Hot loop expected after PGO | Confirm block execution counts and branch weights align with the profile. |
| LTO added inlining | Compare I-cache and instruction counts against non-LTO builds. |

## Agent checklist

Before giving a security or performance verdict, require:

- the exact binary build ID;
- target CPU and feature set;
- optimization pipeline including PGO/LTO/BOLT;
- trace/counter schema version;
- paired input classes;
- at least one explanation tying dynamic differences back to IR, machine blocks,
  or object layout.
