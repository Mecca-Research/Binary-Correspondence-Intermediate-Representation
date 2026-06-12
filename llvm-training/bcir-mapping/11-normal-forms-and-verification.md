# Normal Forms and Verification

## Key takeaways

- BCIR normal form is a **stage contract**, not merely LLVM IR that passes the
  generic verifier: source register identity, claim identity, byte layout,
  address spaces, runtime boundaries, and diagnostic provenance remain explicit.
- A transform may consume an invariant only at a named stage, and it must either
  preserve the invariant or reify the same information as values, metadata, or a
  recognized intrinsic/runtime call.
- Run a BCIR verifier before and after transforms that can change value identity,
  memory addressing, call boundaries, metadata, poison behavior, or diagnostics.
- The LLVM verifier catches structural IR errors; a BCIR verifier catches legal
  LLVM IR that has drifted away from the BCIR mapping contract.
- MLIR conversion failures and LLVM-side mapping-drift diagnostics should share
  stable claim IDs and source locations so one lowering can be followed end to end.

## What “normal form” means

A BCIR normal form is the canonical representation expected at a particular
lowering boundary. It is deliberately stricter than “valid LLVM IR.” Two LLVM
modules can compute the same result and still differ in whether an agent or pass
can prove which source register, claim, graph operation, address-space resource,
or runtime obligation each instruction represents.

Normal form therefore has a version and a stage. A module or function should
identify the stage it claims to satisfy, for example with a function attribute,
a module flag, or metadata such as `!bcir.stage`. A verifier interprets the
invariants below relative to that stage and rejects silent transitions.

## Normal-form invariants

### 1. 1:1 register correspondence

While the register-bound contract is active, each BCIR source register has one
unambiguous LLVM representative: an SSA value, stack/resource slot, table index,
or ABI handle. Combining two mapped registers, splitting one into lanes, or
replacing it with a call is legal only when the pass records the replacement map
or explicitly advances to a stage where 1:1 correspondence has been consumed.
Physical machine-register allocation is not part of this invariant.

### 2. Stable claim IDs

A claim ID is immutable across rewrites. Cloning may add a distinct instance or
clone ID, but it must not renumber the originating claim. If execution needs the
ID, carry it as an operand, field, or runtime argument; metadata alone is not
semantic storage. Diagnostics and replacement-map entries should use the same ID.

### 3. Explicit byte strides

Mixed-stride addressing expresses row, column, plane, and record strides in
bytes. Prefer integer byte-offset arithmetic followed by `getelementptr i8`.
A typed GEP may be used only when its source element type and scale exactly model
the declared byte layout; the scale must never be inferred from an obsolete
pointee type.

### 4. Explicit runtime-wrapper boundaries

Operations owned by the runtime remain calls to recognized wrappers or
intrinsics until the stage authorized to inline or lower them. The call exposes
ownership, effects, ABI arguments, failure behavior, and claim identity. Replacing
it with an ordinary load/store is drift unless the lowering proves equivalence
and records that the boundary was consumed.

### 5. Metadata preservation

Claim, register, graph, rule, source-location, and diagnostic attachments move to
the replacement instruction most closely representing the source operation.
When a rewrite has several replacements, use a documented primary operation plus
replacement-map metadata or a side table. Metadata that affects execution must
instead be reified as an operand, field, intrinsic, or runtime call.

### 6. No hidden typed-pointer assumptions

Pointers are opaque. A pass must derive element size and layout from explicit
schema, data layout, operation attributes, or descriptor values—not from an
assumed pointee type. Every GEP states its source element type, and BCIR byte
layout should normally be visible as byte arithmetic.

### 7. Safe `poison` / `undef` / `freeze` handling

A transform must not introduce a path where poison reaches a branch condition,
address, memory side effect, runtime argument, or other immediate-UB boundary.
Do not replace a defined BCIR value with `undef`. Use `freeze` at a justified
boundary when the contract permits an arbitrary but stable value; otherwise
preserve or reconstruct definedness. The verifier should report the source claim
and the first unsafe use, not merely the final instruction.

### 8. No implicit address-space collapse

A resource pointer retains its address space through GEPs, loads, stores, calls,
and descriptor materialization. `addrspacecast` is allowed only when target and
ABI rules define it and the consuming stage is prepared for the result. Casting
every pointer to address space zero for convenience is invalid drift.

### 9. Diagnostics survive until the consuming stage

A diagnostic obligation remains attached or otherwise reified until the pass
that reports, serializes, or deliberately consumes it. Canonicalization may move
the obligation, but may not erase the last claim/source/rule correlation needed
to explain a failure. Consumption should be explicit in the pass contract and,
when useful, recorded in stage metadata.

## Invariant tables

### Must preserve

| Invariant | Required evidence while active | Typical verifier question |
| --- | --- | --- |
| 1:1 register correspondence | Unique register ID on an SSA value, slot, index, or handle | Does every live source register have exactly one current representative? |
| Stable claim IDs | Same claim value in operands/records and claim metadata | Did any rewrite renumber, omit, or ambiguously merge claims? |
| Explicit byte strides | Byte-valued stride operands/constants and byte-offset arithmetic | Can the byte address be reconstructed without a pointee-type guess? |
| Runtime-wrapper boundary | Recognized call/intrinsic with complete ABI operands | Was a runtime-owned effect bypassed or partially inlined? |
| Metadata/provenance | Claim, source, graph/rule, and diagnostic correlation on replacements | Can a failure still be attributed to the originating operation? |
| Address space | Original address space or an authorized, documented cast | Did a transform silently collapse a nonzero address space? |
| Definedness | Defined operands or justified `freeze` before sensitive uses | Can poison/undef reach control flow, an address, or a side effect? |

### May lower away after this stage

| Representation | Earliest stage that may consume it | Required result of consumption |
| --- | --- | --- |
| Source register labels | After register-correspondence consumers and replacement-map emission | Stable value/slot mapping or an explicit “mapping consumed” stage transition |
| Planning-only HAM hints | After the HAM/prefetch selection pass | Selected intrinsic/runtime action, or a recorded no-op decision |
| Graph shape annotations | After descriptor, affine, or loop structure is materialized | Runtime descriptor fields, loop bounds, or replacement provenance |
| Runtime wrapper | ABI/runtime lowering or authorized inlining stage | Equivalent explicit effects plus retained claim and diagnostic identity |
| Diagnostic attachment | Diagnostic reporting/serialization stage | Emitted diagnostic, side-table record, or explicit consumed marker |
| Normal-form stage marker | On entry to the next named normal form | Replacement marker naming the new form/version |

“May lower away” never means “may disappear whenever convenient.” The named
consumer must validate its input and establish the next contract.

### Must be reified as metadata or intrinsic call

| BCIR fact | Acceptable reification | Not sufficient |
| --- | --- | --- |
| Diagnostic/source provenance | `!bcir.diag`, debug location, or side table keyed by claim ID | A comment or transient MLIR-only attribute after conversion |
| Register replacement map | `!bcir.reg`/named metadata or a verifier-visible mapping record | Similar SSA names such as `%r7.new` |
| Advisory HAM intent | Metadata, `llvm.prefetch`, or a recognized target/runtime intrinsic | An undocumented load ordering coincidence |
| Runtime-owned operation | Recognized wrapper or intrinsic call with explicit operands | An unmarked load/store sequence |
| Clone/split lineage | Origin claim plus clone/part metadata on each replacement | Reusing a new claim ID without a link to the origin |
| Consumed diagnostic obligation | Stage/result metadata or a reporting intrinsic/call | Dropping the last attachment silently |

Semantic facts needed at runtime must be ordinary operands or memory as well as,
optionally, metadata. This table describes verifier-visible representation, not a
license to encode executable behavior only in metadata.

### Invalid drift patterns

| Drift pattern | Why invalid | Expected diagnostic focus |
| --- | --- | --- |
| Two mapped registers folded into one value with no replacement map | Breaks 1:1 correspondence | Both register IDs and the combining instruction |
| Claim ID changed during cloning or canonicalization | Breaks cross-stage identity | Old ID, new ID, and pass/stage |
| `getelementptr i32` used to hide a declared 6-byte or dynamic byte stride | Reintroduces typed-pointer/layout assumptions | Declared stride and computed scale |
| Runtime wrapper replaced by raw memory operations before ABI lowering | Hides ownership/effects and diagnostics | Wrapper name, claim, and replacement operations |
| Last `!bcir.diag` attachment dropped | Makes later failures unattributable | Claim/source and the rewrite that lost provenance |
| Poison-producing value reaches a branch, address, or runtime call | Can create immediate UB or unstable behavior | Def-use path and suggested `freeze`/repair point |
| Nonzero resource pointer cast to generic address space without authorization | Loses target memory-domain semantics | Source/destination spaces and ABI rule |
| MLIR conversion erases an op after emitting only a generic error | Loses operation/claim correlation | MLIR location, operation name, claim ID, and legality failure |

## Examples

- [`examples/normal-form-valid.ll`](examples/normal-form-valid.ll) is ordinary,
  verifier-clean opaque-pointer LLVM IR. It keeps a 1:1 `r7` representative,
  claim `42017`, explicit 64-byte/4-byte strides, address space 1, a justified
  `freeze`, a runtime wrapper, and diagnostic metadata.
- [`examples/normal-form-drift.invalid.ll.txt`](examples/normal-form-drift.invalid.ll.txt)
  is intentionally **semantic-only invalid**: LLVM accepts it, but a BCIR
  verifier should reject register coalescing, address-space collapse, hidden GEP
  scaling, and wrapper bypass.
- [`examples/normal-form-metadata-loss.invalid.ll.txt`](examples/normal-form-metadata-loss.invalid.ll.txt)
  is also semantic-only invalid: the executable call remains valid LLVM IR, but
  the last mapping and diagnostic evidence has disappeared.

The `.invalid.ll.txt` suffix keeps semantic-negative fixtures out of known-good
LLVM sweeps. Their sentinel comment tells `verify-invalid-fixtures.sh` that LLVM
acceptance is expected and that the lesson concerns the stronger BCIR contract.

## Verifier design

### What a BCIR verifier pass checks

A practical `bcir-verify` pass should:

1. read the declared normal-form stage/version and reject unknown combinations;
2. build maps from register IDs and claim IDs to current LLVM representatives;
3. check uniqueness, completeness, and replacement lineage at function and
   module boundaries;
4. inspect GEP/index arithmetic against explicit byte-stride and data-layout facts;
5. validate recognized runtime wrappers/intrinsics, signatures, address spaces,
   effect summaries, and required claim operands;
6. check that required metadata moved to replacements and that no diagnostic
   obligation vanished before its consumer;
7. trace risky poison/undef producers to sensitive uses and validate justified
   `freeze` boundaries; and
8. emit a deterministic summary keyed by function, claim, register, rule, and
   source location.

The pass should distinguish an **IR error** (malformed contract), a **mapping
warning** (advisory evidence lost but semantics retained), and a **stage error**
(a pass ran before its prerequisites or after its contract expired). Strict
conformance pipelines should promote mapping warnings to errors.

### Placement in New PM pipelines

Run the generic LLVM `verify` first when malformed IR could make custom analysis
unsafe. Run `bcir-verify`:

- immediately after LLVM IR import or MLIR translation establishes a BCIR normal form;
- before a pass that consumes register identity, metadata, wrappers, strides,
  address spaces, or diagnostic obligations;
- after every transform that can alter those properties; and
- once more before code generation or handoff to a non-BCIR pipeline.

A module/function pipeline can use fenceposts such as:

```text
verify,
function(bcir-verify,canonicalize-bcir-metadata,bcir-verify),
sccp,
function(bcir-verify),
loop-rotate,
function(bcir-verify,gaadmsf-ham-prefetch,bcir-verify),
verify
```

The custom pass must use the correct New PM granularity. A function verifier can
check local def-use and attachments; a module verifier is needed for named
metadata, claim uniqueness across functions, wrapper declarations, and side
tables. See [Adaptive BCIR pipelines](../17-new-pass-manager/04-adaptive-bcir-pipelines.md).

### Reporting mapping drift

A drift diagnostic should be actionable and stable. Include:

- severity and invariant name;
- normal-form stage/version and the pass boundary being checked;
- function/block/instruction or module entity;
- stable claim, register, graph, and rule IDs when available;
- source location and MLIR operation location when preserved;
- the expected representation, observed representation, and last known valid
  representative; and
- a repair hint, such as attaching replacement metadata, retaining an address
  space, inserting a justified `freeze`, or advancing the stage explicitly.

Prefer one primary diagnostic at the earliest provable drift point, with notes
for downstream symptoms. Sort aggregate diagnostics by stable IDs and IR order
so tests and automated agents receive deterministic output.

### Interaction with MLIR conversion diagnostics

MLIR conversion should reject an illegal BCIR operation before erasing it and
report the operation name, location, claim ID, and failed legality/materialization
rule. When conversion succeeds, transfer those identifiers to LLVM dialect
attributes, explicit operands, or side tables that survive translation to LLVM
IR. The first LLVM-side `bcir-verify` then validates the translation boundary.

Do not report the same root cause as unrelated MLIR and LLVM failures. Preserve a
stable diagnostic/correlation ID: the LLVM verifier can cite the earlier MLIR
conversion record as a note, while still reporting any newly introduced drift.
See [BCIR dialect to LLVM dialect](../18-mlir-lowering-to-llvm/04-bcir-dialect-to-llvm.md)
and [Metadata and Diagnostics](10-metadata-and-diagnostics.md).

## Verification commands

From the repository root:

```bash
llvm-as llvm-training/bcir-mapping/examples/normal-form-valid.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/normal-form-valid.ll -o /dev/null
./llvm-training/tools/verify-invalid-fixtures.sh
./llvm-training/tools/verify-bcir-mapping.sh
```

The current fixtures permit deterministic generic LLVM and semantic-negative
checks, so no fixture-specific logic is required in `verify-bcir-mapping.sh`.
A future executable `bcir-verify` should add explicit expected-diagnostic checks
for the two semantic-negative fixtures rather than guessing from comments.

## Adversarial normal-form tests

The [adversarial exercise track](../exercises/adversarial/) exercises the gap
between generic LLVM validity and BCIR normal-form validity. Its classified
fixtures are suitable for pre/post-pass correspondence checks, metadata side-table
checks, poison-sensitive control-flow review, and reproducible lowering fuzzing.
