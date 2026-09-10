# 06 — Transform dialect for BCIR lowering

The Transform dialect is a **scriptable control layer** over another piece of
MLIR. That other piece is the **payload IR**: the BCIR graph, functions, affine
loops, vector operations, and eventually LLVM-dialect operations that will
become executable code. The Transform dialect program is **transform IR**: a
compile-time strategy that finds payload objects and tells MLIR which rewrites,
passes, or conversions to apply to them.

This separation is useful for BCIR because lowering policy can change without
making policy part of graph semantics. A transform sequence can select one
regular graph region for vectorization while leaving another region scalar, or
can test a new affine-to-vector schedule without adding scheduling fields to
`bcir.vertex`.

Transform IR is not a replacement for BCIR lowering patterns, conversion
targets, or passes. It orchestrates those implementations. It also lowers away
completely and must not be required by the generated program at runtime.

## Payload IR, transform IR, and handles

A transform SSA value is normally a **handle**, not a payload SSA value. For
example, a handle typed as `!transform.op<"bcir.vertex">` may refer to one,
many, or no `bcir.vertex` operations in the payload. Applying a transform to
that handle usually applies it to every associated payload operation.

```mlir
transform.named_sequence @__transform_main(
    %root: !transform.any_op {transform.readonly}) {
  %vertices = transform.select "bcir.vertex" in %root
    : (!transform.any_op) -> !transform.op<"bcir.vertex">
  transform.print %vertices {name = "matched BCIR vertices"}
  transform.yield
}
```

Keep these levels distinct:

- `%root` and `%vertices` are values in **transform IR**.
- The operations associated with `%vertices` are in **payload IR**.
- A transform handle can denote a set; do not assume it denotes exactly one
  operation unless a matcher or transform establishes that condition.
- A transform that erases or replaces payload operations may consume or
  invalidate handles to those operations and their nested objects. Re-match or
  use a replacement handle returned by the transform rather than using a
  dangling handle.

The entry-point convention commonly uses a named sequence called
`@__transform_main`. A Transform dialect interpreter associates its root block
argument with the selected payload root, then executes the nested transform
operations in order.

## Matching BCIR graph operations

Operation-name matching is the simplest way to select BCIR graph objects:

```mlir
%vertices = transform.select "bcir.vertex" in %root
  : (!transform.any_op) -> !transform.op<"bcir.vertex">
%edges = transform.select "bcir.edge" in %root
  : (!transform.any_op) -> !transform.op<"bcir.edge">
```

Name matching is deliberately syntactic. It is appropriate when the next action
applies to every operation with that name. A production BCIR transform
extension should use a stronger matcher when it needs properties such as:

- vertices in a particular graph space;
- edges with statically known fanout or stride;
- operations carrying a claim ID;
- regions whose memory effects permit vectorization;
- HAM hints that have not already been materialized.

A typed handle rejects payload operations with the wrong operation name at
transform execution time, but it does not prove BCIR semantic preconditions.
Those checks belong in a BCIR matcher/action, verifier, rewrite pattern, or
conversion pass, with a diagnostic that names the failing payload operation.

## Sequencing canonicalization

Canonicalization should expose a stable graph shape before a script relies on
that shape. For BCIR, canonicalization can fold aliases, normalize equivalent
edge spellings, remove empty graph wrappers, and make statically known
attributes uniform. A sequence can request registered canonicalization
patterns directly:

```mlir
transform.apply_patterns to %root {
  transform.apply_patterns.canonicalization
} : !transform.any_op
```

The canonicalizer only runs patterns registered by loaded dialects. Therefore,
BCIR-specific normalization must exist as BCIR canonicalization patterns or as a
BCIR transform/pass invoked by the sequence. Do not write a script that assumes
`bcir.edge` has a canonical `weight`, `stride`, or `layout` attribute unless the
preceding step guarantees it.

Match after any canonicalization that may replace the operations you intend to
handle. This both avoids stale handles and prevents the script from branching on
non-canonical payload shapes.

## BCIR-specific walkthrough

A practical BCIR lowering sequence has the following control flow. The complete
strategy sketch is in
[`examples/bcir-transform-sequence.mlir`](examples/bcir-transform-sequence.mlir).

### 1. Canonicalize the payload root

Run generic and BCIR canonicalization before matching. This establishes the
input contract for later actions and avoids selecting aliases that will soon be
erased.

### 2. Match `bcir.vertex` and `bcir.edge`

Create typed operation handles for the graph objects. Empty handles may be valid
for optional graph fragments; a required-match action should diagnose an empty
match rather than silently doing no work.

### 3. Normalize edge attributes

Normalize edge fields such as `kind`, `stride`, `weight`, and claim provenance
before affine or vector planning. The examples invoke an illustrative registered pass named
`bcir-normalize-edge-attributes`. A production implementation could instead
provide a typed `transform.bcir.normalize_edge_attributes` operation through the
Transform dialect extension mechanism. Either form is BCIR-owned, not an
upstream graph transformation.

Normalization must preserve dynamic information. For example, do not replace a
dynamic edge weight with a planning attribute; materialize the dynamic value in
payload IR and only canonicalize its representation.

### 4. Pre-lock register bindings

Resolve logical register/resource requests while BCIR identity and diagnostics
are still available. A pre-lock action should either:

- attach a validated logical binding that a later BCIR-to-LLVM conversion can
  materialize as a resource-table lookup or ABI operand; or
- emit a diagnostic identifying the vertex, requested bank, claim, and reason
  the binding cannot be honored.

A transform script must not pretend that LLVM's register allocator will infer a
required BCIR pre-lock from an arbitrary attribute.

### 5. Attach HAM hints

Attach HAM policy to the payload operations that will survive long enough to
produce a prefetch, runtime scheduling call, or metadata anchor. HAM hints are
optimization guidance, but dropping one should be an explicit policy decision.
Required synchronization or data movement must never be encoded as a hint.

### 6. Lower through affine/vector staging

Invoke the BCIR lowering implementation that turns regular graph traversal into
`affine`, `scf`, `arith`, `memref`, and `vector` operations. The transform script
chooses and orders this action; the lowering patterns still own type conversion,
replacement values, legality, and metadata transfer.

The vector-focused example
[`examples/bcir-transform-vectorize-then-lower.mlir`](examples/bcir-transform-vectorize-then-lower.mlir)
shows a sequence that selects vectorizable vertices, requests a width-four BCIR
vectorization action, and then drives the generic lowering boundary.

### 7. Apply conversion to LLVM dialect

Once BCIR facts have been materialized and graph operations are gone, lower the
staging dialects in dependency order. A representative pass sequence is:

```text
lower-affine
  -> convert-vector-to-scf (when required by the chosen vector lowering)
  -> convert-scf-to-cf
  -> convert-vector-to-llvm
  -> finalize-memref-to-llvm
  -> convert-arith-to-llvm
  -> convert-func-to-llvm
  -> reconcile-unrealized-casts
```

Exact pass names and ordering are toolchain- and pipeline-dependent. Transform
IR may invoke registered passes with `transform.apply_registered_pass`, or a
project may use `transform.apply_conversion_patterns` plus its LLVM conversion
extensions. In either case, the final conversion must use a `ConversionTarget`
and `TypeConverter` that make all remaining non-LLVM operations and types legal
or reject the payload with actionable diagnostics.

Do not treat the presence of an `llvm.*` operation as proof that conversion is
complete. Verify that no illegal BCIR, affine, vector, memref, arith, func, or
unrealized-cast operations remain for the selected boundary, then translate and
verify the resulting LLVM IR.

## Driving affine and vector lowering deliberately

Transform scripts are valuable when only some graph fragments should take the
structured path. A script can:

1. match vertices whose shape is regular enough for affine lowering;
2. annotate or pass parameters such as tile size and vector width;
3. invoke the BCIR graph-to-affine/vector implementation;
4. re-match the produced affine loops or vector operations;
5. run cleanup patterns between structural transforms;
6. invoke conversion to LLVM dialect only after the selected schedule is
   complete.

The script should carry **policy**, while payload attributes carry facts that
must survive and lowering implementations carry semantics. A vector width of
four may be transform policy. A claim ID, required register binding, or dynamic
edge stride is payload meaning and must not exist only in the script.

## Failure modes and diagnostics

Transform application has success, recoverable (silenceable) failure, and
irrecoverable failure behavior. Treat that distinction as part of the lowering
contract:

| Failure | Good BCIR behavior |
| --- | --- |
| No optional vectorizable vertex matched | Return an empty handle or a recoverable failure and use a documented scalar fallback. |
| A required `bcir.vertex` is absent | Emit a failure naming the expected graph scope and stop the required sequence. |
| Edge attributes cannot be normalized | Diagnose the edge location, conflicting attributes, and expected canonical form before mutating the payload. |
| A required register pre-lock is impossible | Emit an irrecoverable error with vertex/claim identity and requested resource class. |
| HAM hint cannot be materialized on the target | Follow the declared hint policy: diagnose and drop an optional hint, or fail if the operation was incorrectly marked as semantic. |
| Dialect conversion leaves an illegal operation | Report the remaining operation and missing conversion rather than marking it legal to silence the failure. |
| A consumed handle is reused | Treat the script as invalid; enable transform handle-use checking and re-match or use returned handles. |

Diagnostics are compiler output, not disposable debug logging. Preserve payload
locations on replacement operations, propagate the message emitted by failed
transforms, and make CI fail when a required top-level transform sequence fails.
For development, `transform.print` can expose associated payload objects, but it
does not replace structured failure messages.

## Pitfalls

### Transform scripts become stale when dialect operation names change

A string match for `"bcir.vertex"` will stop selecting payload operations if the
operation is renamed. Keep transform scripts in the same compatibility review as
ODS/TableGen changes, test required matches, and prefer typed BCIR transform
extensions when a semantic concept spans multiple operation spellings.

### Confusing transform IR with payload IR

A handle is not a vertex value, and a transform parameter is not automatically a
runtime constant. Never wire transform SSA values into program computation or
store required runtime meaning only in the transform module.

### Dropping diagnostics from failed transformations

Do not silence every recoverable failure merely to continue a pipeline. Silence
only an anticipated miss with a defined alternative, such as falling back from
vector to scalar lowering. Propagate unexpected matcher, rewrite, pre-lock, and
conversion failures with payload locations.

### Running transform scripts before required canonicalization

A matcher written for canonical edges may find nothing—or find the wrong
operations—if aliases and legacy attribute forms still exist. Canonicalize first,
then match, and re-match after any transformation that replaces handled payload
operations.

### Assuming handles stay valid after mutation

Transforms that replace or erase payload operations can invalidate handles to
those operations and nested objects. Do not retain `%vertices` across a graph
lowering that destroys `bcir.vertex`; use the new handle returned by the action
or select the produced affine/vector operations again.

### Hiding semantics in orchestration

Register requirements, claims, diagnostics, and dynamic graph facts must be
represented in payload IR or explicit lowering configuration. A script may
choose when to lower them, but removing the script must not change the meaning
of the payload program.

## Running and checking a sequence

A downstream BCIR tool should register the Transform dialect, the required
extensions, BCIR operations, and every pass named by the script. A typical
workflow is conceptually:

```sh
bcir-opt payload.mlir \
  --transform-interpreter='transform-file-name=strategy.mlir' \
  --verify-each \
  -o lowered.mlir
```

Interpreter options vary across MLIR versions and downstream drivers, so pin the
command used by the project rather than copying it blindly. Also run transform
handle-use checks when available, verify the final MLIR, translate LLVM dialect
to LLVM IR, and run the LLVM verifier.

The example files are strategy sketches because this training repository does
not register the BCIR dialect or the illustrative BCIR passes named by
`transform.apply_registered_pass`. They are still useful for reviewing the
boundary between upstream Transform operations and BCIR-owned actions.

## Review checklist

- Is transform IR clearly separate from payload IR?
- Does every handle have the intended cardinality and operation type?
- Does canonicalization run before matchers that depend on canonical form?
- Are edge normalization, register pre-locking, and HAM attachment implemented
  by verifiable BCIR actions rather than comments or transform-only state?
- Are handles re-matched after payload-replacing transformations?
- Does vector/affine policy remain separate from semantic BCIR facts?
- Are optional misses distinguished from required failures?
- Are diagnostics propagated with payload locations and claim/graph identity?
- Does the final conversion reject all operations illegal at the LLVM boundary?
- Can the resulting LLVM dialect translate to verifier-clean LLVM IR?

## Related material

- [`../14-mlir-bridge/04-bcir-as-custom-dialect.md`](../14-mlir-bridge/04-bcir-as-custom-dialect.md)
  introduces `bcir.vertex`, edges, HAM hints, and register binding as custom
  dialect concepts.
- [`04-bcir-dialect-to-llvm.md`](04-bcir-dialect-to-llvm.md) defines the semantic
  choices that the transform sequence orchestrates.
- [`05-affine-vector-llvm-lowering-pipeline.md`](05-affine-vector-llvm-lowering-pipeline.md)
  explains the affine/vector/LLVM staging order.
- [`../bcir-mapping/01-vertex-edge-attribute.md`](../bcir-mapping/01-vertex-edge-attribute.md)
  maps vertex, edge, and attribute facts to LLVM-level representations.
- [`../indexes/bcir-patterns.md`](../indexes/bcir-patterns.md) indexes related
  BCIR lowering patterns, hazards, examples, and exercises.
