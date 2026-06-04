# BOLT layout walkthrough

This walkthrough turns the schematic BOLT command from
[`06-pgo-lto-bolt.md`](06-pgo-lto-bolt.md) into a small, inspectable artifact.
The goal is not to prove a universal speedup; it is to teach agents how to read
baseline binary layout, collect a post-link profile, and compare the BOLT-rebuilt
layout against the original executable.

## Fixture

Use [`examples/bolt-layout-demo.c`](examples/bolt-layout-demo.c). The source is
intentionally tiny and names the functions and blocks to look for during layout
inspection:

- `main` drives a biased loop.
- `hot_path` is called for the common case.
- `cold_path` is called only for the rare case.
- `checksum` is shared by both paths.
- `hot_path` has an uncommon adjustment block guarded by `(x & 127) == 0`.

## Build a baseline binary

BOLT normally needs relocation information so it can rewrite the linked binary.
Keep frame pointers to make sampled profiles easier to symbolize:

```bash
clang -O2 -fno-omit-frame-pointer -Wl,--emit-relocs \
  llvm-training/07-optimization/examples/bolt-layout-demo.c \
  -o /tmp/bolt-layout-demo
```

Confirm that the expected functions survived optimization and record the initial
symbol order:

```bash
llvm-objdump -t /tmp/bolt-layout-demo | \
  sed -n '/ main$/p;/ hot_path$/p;/ cold_path$/p;/ checksum$/p'
```

Then capture a readable baseline disassembly. The exact addresses are target- and
linker-dependent, but the function order and fall-through/call edges are the
facts to compare:

```bash
llvm-objdump -d --symbolize-operands /tmp/bolt-layout-demo | \
  sed -n '/<main>:/,/<.*>:/p;/<hot_path>:/,/<.*>:/p;/<cold_path>:/,/<.*>:/p'
```

Inspection prompts:

- Are `hot_path` and `cold_path` near each other even though one is much hotter?
- Does `hot_path` contain a rare conditional edge that could be moved away from
  the common trace?
- Do call targets and symbol addresses match the static source-level intuition?

## Collect a BOLT profile

On Linux hosts with `perf`, collect sampled branch data, then convert it to BOLT
`.fdata` format:

```bash
perf record -e cycles:u -j any,u -o /tmp/bolt-layout-demo.perf.data \
  /tmp/bolt-layout-demo
perf2bolt -p /tmp/bolt-layout-demo.perf.data \
  -o /tmp/bolt-layout-demo.fdata \
  /tmp/bolt-layout-demo
```

If `perf` is unavailable, use a profile collected on a comparable machine. Do not
silently reuse stale data: record the workload, CPU, compiler, linker, and exact
binary hash next to the `.fdata` file.

## Rewrite and inspect the BOLT layout

Run BOLT with explicit layout-oriented flags so the intent is visible in logs:

```bash
llvm-bolt /tmp/bolt-layout-demo \
  -o /tmp/bolt-layout-demo.bolt \
  -data=/tmp/bolt-layout-demo.fdata \
  -reorder-functions=hfsort \
  -reorder-blocks=ext-tsp \
  -split-functions \
  -print-finalized
```

Now compare symbol and block layout:

```bash
llvm-objdump -t /tmp/bolt-layout-demo.bolt | \
  sed -n '/ main$/p;/ hot_path$/p;/ cold_path$/p;/ checksum$/p'

llvm-objdump -d --symbolize-operands /tmp/bolt-layout-demo.bolt | \
  sed -n '/<main>:/,/<.*>:/p;/<hot_path>:/,/<.*>:/p;/<cold_path>:/,/<.*>:/p'
```

Expected observations for the demo workload:

- `hot_path` should dominate the profile, while `cold_path` should have much
  lower execution count.
- Function order may change so hot functions are clustered for locality.
- Rare blocks from `hot_path` may be split or placed after the hot trace.
- Addresses are not semantic identities after post-link rewriting; use symbols,
  relocation-aware disassembly, and profile counts when comparing binaries.

## Guarded smoke check

[`../tools/smoke-bolt.sh`](../tools/smoke-bolt.sh) is intentionally conservative.
It exits successfully with a clear skip message when `llvm-bolt` is not installed,
which allows CI environments without BOLT packages to keep validating the rest of
the training corpus. When BOLT is present, the script builds this fixture, records
baseline layout text, and reports whether the host also has `perf2bolt` for a full
profile-driven run.
