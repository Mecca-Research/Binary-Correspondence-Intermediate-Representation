# Reproducible LTO and BOLT experiment matrix

This lab turns the introductory PGO/LTO/BOLT discussion into a small,
repeatable artifact experiment. It does **not** claim that one configuration is
faster. Runtime benchmarking, sampling design, confidence intervals, machine
isolation, and workload representativeness remain separate research tasks.

## Experiment contract

The checked-in manifest is
[`experiments/matrix.json`](experiments/matrix.json). Its dimensions are:

- LTO mode: no LTO, ThinLTO, or FullLTO;
- optimization level: `O0` or `O2`;
- target triple: `host` (resolved and recorded as `clang -dumpmachine`);
- optional BOLT rewrite: false for the core matrix and true for the explicit
  `full-O2-host-bolt` leg.

The two tiny translation units intentionally expose cross-module decisions:
[`main.c`](experiments/main.c) calls functions in
[`worker.c`](experiments/worker.c), which also contains a local helper. The
runner asks LLD to retain LTO temporary bitcode. For ThinLTO and FullLTO cases it
converts import/internalize/optimized snapshots to textual LLVM IR and records
the `define`/`declare` lines in `lto-ir-summary.txt`.

## Run the matrix

```bash
llvm-training/tools/run-lto-matrix.sh \
  --output /tmp/lto-matrix \
  --require-supported
```

Useful filters are `--lto none|thin|full`, `--optimization O0|O2`, and
`--target host|TRIPLE`. Set `LLVM_SUFFIX=-20` (or another installed suffix) to
select an exact LLVM family. Without that override the runner searches
version-suffixed tool families from newest to oldest before trying unsuffixed
names.

The core runner explicitly discovers `clang`, `ld.lld`, `llvm-nm`,
`llvm-objdump`, `llvm-readobj`, and `llvm-size`. Their LLVM major versions must
match. A missing tool, version mismatch, unsupported target, or rejected LTO
mode is reported as `unsupported`; it is never converted into a passing result.
Use `--require-supported` when unsupported core configurations should fail the
calling job.

## Deterministic observations

Each successful case retains:

- the exact compile/link command and tool versions in the top-level report;
- the executable and its byte size plus per-section sizes;
- normalized defined symbols and a `symbol-presence.json` map for the tracked
  fixture functions;
- LTO resolution data and import/internalization IR summaries when emitted by
  LLD;
- section headers and relocation records/counts;
- normalized disassembly and its SHA-256 digest.

These are observations, not universal golden values. Compare cases produced by
one compatible toolchain invocation. Object sizes, section names, and hashes can
legitimately change when LLVM, the linker, target, C library, or fixture changes.
A useful review is:

```bash
jq '.results[] | {id, status, measurements}' /tmp/lto-matrix/report.json
diff -u /tmp/lto-matrix/cases/none-O2-host/symbols.txt \
        /tmp/lto-matrix/cases/full-O2-host/symbols.txt
diff -u /tmp/lto-matrix/cases/thin-O2-host/lto-ir-summary.txt \
        /tmp/lto-matrix/cases/full-O2-host/lto-ir-summary.txt
```

## JSON report

`report.json` is the evaluation-runner interface. It contains:

- `schema_version` and `kind`;
- generation time, manifest path, and exact tool paths/versions;
- aggregate counts for `passed`, `skipped`, `unsupported`, and `failed`;
- one result per selected manifest configuration with its dimensions, status,
  reason, artifact paths, byte size, relocation count, symbol count, and
  normalized disassembly hash.

`skipped` means an optional leg was not requested. `unsupported` means it was
requested or selected but the compatible environment was unavailable. `failed`
means a supported command or fixture check failed.

## Optional BOLT leg

BOLT is deliberately not required by the core CI matrix. To request it, provide
a profile that `llvm-bolt` can consume:

```bash
llvm-training/tools/run-lto-matrix.sh \
  --lto full --optimization O2 --include-bolt \
  --bolt-profile path/to/profile.fdata \
  --output /tmp/lto-bolt
```

The standalone equivalent is:

```bash
llvm-training/tools/run-bolt-experiment.sh \
  --profile path/to/profile.fdata \
  --output /tmp/bolt-experiment
```

A real rewrite directory retains the baseline binary, copied profile input,
exact rewrite command, rewritten binary and summary, logs, and tool versions.
If BOLT, a matching tool family, or the explicit profile is absent, the runner
retains whatever baseline evidence it could produce and writes an
`unsupported` JSON report. It does not fabricate a profile or silently pass.
`smoke-bolt.sh` exercises this CI-safe baseline/skip path.

## Interpretation boundaries

This matrix can answer questions such as “did FullLTO internalize the public
worker functions?”, “which tracked symbols remain?”, or “did the section and
relocation summaries change?”. It cannot establish a speedup. A performance
claim requires a stable machine, representative profile collection, repeated
measurements, randomized or interleaved trials, noise controls, and statistical
analysis; those studies remain out of scope for this training corpus.
