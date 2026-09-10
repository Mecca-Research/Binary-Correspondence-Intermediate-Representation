# Dialect as Data

## Key takeaways

- ODS is **C++ generated at build time**: a tool can validate your dialect only if
  it was compiled with it. IRDL is **a file read at run time**: a tool that has
  never heard of your dialect can validate it from the file alone.
- BCIR ships that file. Stock `mlir-opt` — not `bcir-opt`, not anything built
  from this repository — parses and verifies BCIR IR given only
  `--irdl-file mlir/irdl/bcir.irdl.mlir`.
- That is a checkable property, not a slogan: the same input without the
  projection is rejected as an unregistered dialect.
- The projection is a **subset**, and the repository says which subset in a
  manifest that names every unprojected operation and why. A projection that
  claimed completeness would be the more comfortable lie.

## What ODS cannot do

An ODS dialect becomes C++ through `mlir-tblgen`, and that C++ is linked into a
binary. Everything downstream follows from that one fact: validating your IR
requires *your* tool. A reviewer, a fuzzer, a third-party pass pipeline, or a
verifier written by someone who has never seen your project cannot check your IR
without first building your project.

IRDL moves the dialect definition from generated code into **data** — operations,
operands, results, attributes and constraints, written in MLIR itself. The
definition becomes an artifact you can hand to someone.

## The demonstration

```console
$ mlir-opt --irdl-file=mlir/irdl/bcir.irdl.mlir mlir/test/irdl/00_smoke_generic.mlir
module {
  "bcir.module"() ({
  }) {cost_model = "k_bcir", execution_model = "gem", sym_name = "smoke", …} : () -> ()
  …
}
```

and the control, the same file with the projection withheld:

```console
$ mlir-opt mlir/test/irdl/00_smoke_generic.mlir
error: operation being parsed with an unregistered dialect. If this is intended,
please use -allow-unregistered-dialect with the MLIR tool used
```

The first command is stock MLIR. The dialect it accepted came from a file, and the
file is the whole of what it needed. `tools/irdl/check_corpus.sh` runs that
round-trip over every corpus file and checks the `// CHECK:` patterns in each,
which is the gate this chapter's claim rests on.

## The projection, counted

<!-- generated: irdl-projection -->
| | count |
| --- | ---: |
| operations the ODS dialect defines | 133 |
| operations the IRDL projection declares | 95 |
| projected under the same name | 10 |
| projected under a renamed spelling (IRDL admits no dots) | 85 |
| declared unprojected, each with a stated reason | 38 |
| generic-syntax corpus files the projection is validated against | 13 |
<!-- /generated -->

Three things in that table are worth reading slowly.

**Not every operation is projected.** The projection covers the structural subset —
registry, claim, phase, plan, GEM stream — and the driver-subset operations
(values, registers, ports, descriptors, entry/trampoline) are outside it. Every
one of those is listed in `mlir/irdl/MANIFEST.json` with the reason and the
condition for projecting it: *together with an IRDL corpus fixture that exercises
it*. An unprojected operation is a declared gap, not an omission.

**Some names are spelled differently.** IRDL admits only `[a-z0-9_]` in an
operation name, so a dotted ODS name arrives with underscores —
`gem.stream_pack` becomes `gem_stream_pack`. The renaming is a rule
(`dots-to-underscores`), applied uniformly, and `tools/irdl/check_inventory.py`
reconciles the two spellings so a renamed operation cannot pass as a missing one.

**The reconciliation needs no toolchain.** The inventory gate is pure text over
three sources — the ODS dialect, the projection, the manifest — so it runs on a
host with no MLIR at all. That is why the table above is identical everywhere,
and why the round-trip is checked separately, where a toolchain exists.

## What this makes checkable that ODS does not

- **A third party can validate BCIR IR.** Anyone with stock MLIR and the `.irdl`
  file can check that a module is well-formed BCIR, with no build of this
  repository. For an IR whose entire argument is that legality is decided before
  cost, being externally checkable is not a convenience.
- **The dialect's shape becomes diffable data.** A change to an operation's
  operands shows up as a change to a file that a reviewer reads, rather than as a
  change to generated C++ nobody diffs.
- **The gap between definition and projection is measurable.** Two spellings of
  the same dialect exist, so the interesting question — do they still agree? — can
  be asked mechanically, and is, on every run.

## Pitfalls checklist

- Do not read the projection as the dialect. It is the structural subset; the ODS
  files are the definition and the manifest names the difference.
- Do not compare operation names across the two rails without applying the
  renaming rule; a dotted name and its underscored twin are the same operation.
- Do not assume a green inventory means the corpus round-trips: that check is
  text-only by design, and the round-trip needs `mlir-opt`.
- Do not add an ODS operation and expect the projection to follow. Either project
  it *with a corpus fixture that exercises it*, or declare it in the manifest with
  a reason — the inventory gate accepts nothing else.

## Checks

[`../tools/verify-bcir-approach.py`](../tools/verify-bcir-approach.py) regenerates
the table above from the three sources, and — where `mlir-opt` is installed — runs
`tools/irdl/check_corpus.sh` and requires it to report a round-trip, so the
chapter's central claim is exercised rather than asserted. Where no `mlir-opt`
exists the round-trip is named as skipped; the MLIR rail job owns that toolchain.
