# Reproducible Evidence Pipelines

Binary-analysis tables are useful only when a reviewer can tell **where each row
came from**, **which claims can be reproduced**, and **which observations depend
on the machine that collected them**. This lesson turns tiny source fixtures into
checked static evidence without pretending that shared CI runners are stable
benchmark machines.

## The evidence contract

Chapter 15 uses [`evidence-manifest.json`](evidence-manifest.json) as the index of
checked-in CSV evidence. Every CSV below this chapter has exactly one entry and a
provenance classification:

| Classification | Meaning | CI policy |
|---|---|---|
| `deterministic` | Normalized static facts from a declared source, target, build, and collector | Regenerate and fail on a diff |
| `host-sensitive` | Timing or hardware-counter observations affected by CPU, kernel, load, permissions, or sampling | Keep optional; never use as a portable golden value |
| `schematic` | Hand-authored rows that teach a schema or reasoning pattern | Validate the schema, but do not present the values as measured evidence |

A manifest entry for generated evidence records the source fixture, build
command, target triple, optimization flags, expected artifact type, collection
command, checked-in CSV, and companion provenance file. The provenance file also
records the Clang and Python versions used for the last regeneration.

Schematic rows remain teaching examples unless a provenance manifest explicitly
connects them to a reproducible collection pipeline. A plausible-looking number
is not evidence by itself.

## Why the fixtures use assembly

The fixtures in [`fixtures/`](fixtures) are tiny x86-64 assembly source programs.
That choice keeps the lesson focused on evidence collection rather than on
compiler optimization drift: a C compiler is free to select different, equally
valid instructions across releases. Clang's integrated assembler builds a fixed
`x86_64-unknown-linux-gnu` ELF relocatable object, and the dependency-free Python
collector reads the ELF section and symbol tables directly.

The normalized CSVs cover:

- defined function symbol names;
- instruction-class counts rather than presentation-sensitive disassembly text;
- basic-block counts derived from function entries and conditional-branch
  leaders;
- direct call-edge summaries;
- `.text`, `.data`, and `.bss` section type/flag/size summaries.

The collector deliberately supports only the small instruction vocabulary used
by these fixtures. An unexpected opcode fails closed instead of silently being
misclassified. Object files are temporary build products; only normalized CSV
and provenance JSON are checked in.

## Regenerate and detect drift

From the repository root, regenerate the deterministic evidence:

```sh
python3 llvm-training/tools/generate-binary-analysis-fixtures.py \
  --write --require-tools
```

Check that a clean regeneration has no semantic diff:

```sh
python3 llvm-training/tools/generate-binary-analysis-fixtures.py \
  --check --require-tools
```

`--check` rebuilds objects in a temporary directory and compares normalized CSV
bytes. It also compares stable provenance fields, while treating recorded tool
version strings as informational metadata. This distinction prevents a harmless
patch-level tool update from changing the evidence claim, but still leaves an
audit trail of the environment used for the last explicit regeneration.

Without `--require-tools`, missing Clang produces a clear reduced-coverage skip.
CI installs Clang and uses `--require-tools`, so the deterministic rail cannot
silently disappear there. Use `--fixture ID` to regenerate or check one manifest
fixture.

Validate manifest coverage and provenance separately:

```sh
python3 llvm-training/tools/verify-binary-analysis-evidence.py
```

This verifier fails when a Chapter 15 CSV lacks a manifest entry, a classification
is missing, a deterministic source/provenance file disappears, generated rows do
not match their fixture/classification, or the generated fixture set stops
covering one of the required static evidence families.

## Host-sensitive measurements are optional evidence

Wall-clock time, cycles, instructions retired, branch misses, cache misses, and
similar performance-monitoring counters are affected by factors including:

- CPU model, microcode, frequency scaling, and simultaneous multithreading;
- kernel and `perf_event_paranoid` policy;
- scheduler placement, interrupts, thermal state, and neighboring CI jobs;
- input data, warm-up policy, sampling period, multiplexing, and repetition
  count.

A local experiment may collect those values, but it should put them in a
`host-sensitive` manifest entry and record the host, kernel, CPU, command, input,
warm-up, and repetition policy. Shared CI should not fail because a timing or
counter value moved. At most, CI may check that an optional collector executes
when the necessary tool and permissions are present.

## Reading similarity without overclaiming equivalence

Static evidence is a triage signal, not a semantic proof. Two functions can have
the same instruction-class vector and basic-block count while differing in:

- constants, operand order, signedness, overflow behavior, or memory addresses;
- branch predicates and call targets;
- ABI, relocation, exception, or concurrency behavior;
- data-dependent timing and side-channel leakage;
- behavior outside the observed input or trace set.

Conversely, semantically equivalent functions can have different static features
because of inlining, register allocation, instruction selection, vectorization,
or control-flow normalization. A sound report therefore says that features are
**consistent with**, **support**, or **weaken** a correspondence hypothesis. It
does not turn feature similarity into semantic equivalence.

## Review checklist

Before accepting a binary-analysis claim, ask:

1. Is the row generated, host-sensitive, or schematic?
2. Does the manifest identify the source, target, flags, artifact, and collector?
3. Can deterministic evidence be regenerated with `--check`?
4. Are toolchain and target metadata recorded?
5. Are host-sensitive values optional and accompanied by a measurement protocol?
6. Does the conclusion distinguish feature similarity from semantic equivalence?
7. What additional proof, tests, traces, or manual review would resolve the
   remaining uncertainty?
