# Solution 042: Review an evidence provenance claim

The conclusion overclaims twice. Matching arithmetic and basic-block counts is
static feature similarity, not semantic equivalence: the binaries may use
different constants, predicates, operands, memory locations, call targets, or
undefined-behavior assumptions. Establishing equivalence needs stronger
control/data-flow comparison plus tests, symbolic reasoning, or another semantic
oracle appropriate to the code.

A single wall-clock result on a shared CI host is host-sensitive. Scheduler load,
CPU model and frequency, thermal state, input, warm-up, and neighboring jobs can
change it. It must not become a deterministic golden value.

For static evidence, the manifest should record source fixture, build command,
target triple, optimization flags, expected artifact type, collection command,
classification, checked-in CSV, and provenance path. Provenance should include
toolchain versions and target metadata. An optional timing study should also
record CPU, kernel, affinity, input, warm-up, repetitions, summary statistic, and
variance or confidence interval.

A defensible conclusion is: “For the declared target and collector, the binaries
share two coarse static features. In this explicitly host-sensitive experiment,
A had a lower observed median time. These observations support further
investigation but prove neither semantic equivalence nor universal speedup.”
