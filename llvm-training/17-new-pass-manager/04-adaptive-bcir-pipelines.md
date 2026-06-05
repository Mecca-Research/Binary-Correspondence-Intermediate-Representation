# Adaptive BCIR pipelines

An adaptive pipeline selects passes from explicit module evidence rather than
from hope. For BCIR and GAADMSF, the evidence can include function attributes,
module flags, named metadata, target features, profile summaries, and an
external hardware profile supplied by the driver.

## Policy inputs

Good adaptive inputs are auditable:

- a function attribute such as `"bcir.stage"="register-bound"`;
- module metadata such as `!bcir.module` or `!gaadmsf.profile`;
- operation metadata such as `!bcir.reg`, `!bcir.diag`, `!ham`, or
  `!gaadmsf.region`;
- target features and data layout; and
- profile information loaded through the normal PGO/MLGO path.

Avoid implicit inputs such as “this pipeline name contains gaadmsf”. A pipeline
name can select a policy, but the transform must still verify the IR contract it
is about to consume.

## Destructive-transform fenceposts

For BCIR, surround destructive regions with verifier passes:

```text
bcir-verify,
canonicalize-bcir-metadata,
sccp,
bcir-verify,
loop-rotate,
bcir-verify,
gaadmsf-ham-prefetch,
bcir-verify
```

A transform is destructive if it may change value identity, register mapping,
metadata attachments, memory effects, call boundaries, or CFG shape. SCCP can
replace instructions with constants; loop rotation changes CFG shape; custom HAM
prefetch lowering can add memory-like operations and consume metadata. Each one
needs either exact preservation reasoning or invalidation.

## Register correspondence

A BCIR register-binding stage often wants a 1:1 mapping between source registers
and LLVM SSA values or memory locations. Preserve that mapping until a lowering
stage explicitly consumes it. If a pass combines two mapped values, splits one
mapped value into several lanes, or replaces a mapped operation with a runtime
call, it must record the new correspondence or deliberately end the mapping
contract.

## GAADMSF gating

GAADMSF-specific transforms should refuse to run unless the required evidence is
present. Reasonable gates include:

- a module flag naming the GAADMSF ABI or hardware profile;
- a function attribute that says HAM lowering is allowed;
- metadata marking the exact operations or regions to transform; or
- a driver-provided hardware profile validated by the plugin.

A pass that cannot find its gate should return `PreservedAnalyses::all()` after
emitting a diagnostic, or fail the pipeline if absence of the gate means the IR
is malformed for that compilation mode.

## Command walkthrough

Start with a generic modern pipeline:

```bash
opt -S \
  -passes='verify,function(require<domtree>),sccp,loop-rotate,verify' \
  llvm-training/17-new-pass-manager/examples/gaadmsf-pipeline-before.ll \
  -o /tmp/gaadmsf-pipeline-after.ll
```

Then add plugin-specific guards once `bcir-verify` and GAADMSF passes are
registered:

```bash
opt -load-pass-plugin=./libBcirPasses.so -S \
  -passes='verify,function(bcir-verify,require<domtree>),sccp,function(bcir-verify),loop-rotate,function(bcir-verify,gaadmsf-ham-prefetch,bcir-verify),verify' \
  llvm-training/17-new-pass-manager/examples/gaadmsf-pipeline-before.ll \
  -o /tmp/gaadmsf-pipeline-after.ll
```

The second command illustrates placement, not a checked-in binary dependency.
The plugin provides the BCIR names; LLVM provides `verify`, `require<domtree>`,
`sccp`, and `loop-rotate`.
