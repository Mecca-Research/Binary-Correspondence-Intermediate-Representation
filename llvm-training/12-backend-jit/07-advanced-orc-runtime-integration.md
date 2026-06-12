# Advanced ORC Runtime Integration for BCIR Kernels

## Key takeaways

- Put BCIR-aware LLVM optimization in an `IRTransformLayer`, before target
  compilation, and keep semantic BCIR verification separate from ORC linking.
- Treat lazy, speculative, and hot re-JIT compilation as explicit runtime state
  transitions. Stable entry points, generation checks, and resource trackers are
  required before old code can be retired safely.
- Use `JITDylib`s as policy and lifetime boundaries: separate runtime ABI
  symbols, stable kernel entry points, generated implementations, and optional
  target/device adapters.
- A custom `MaterializationUnit` can defer target choice and artifact production,
  but it must resolve, emit, delegate, or fail every symbol in its interface.
- JITLink owns object graph allocation, fixups, and finalization; it does not
  prove that an object preserves BCIR graph semantics, ABI metadata, or numerical
  behavior.
- Remote and heterogeneous deployment must derive triple, data layout, pointer
  width, calling convention, and memory model from the executor target rather
  than from the compiler host.

This lesson extends [`05-orc-layers.md`](05-orc-layers.md). Read it with the
[New Pass Manager chapter](../17-new-pass-manager/), the
[MLIR-to-LLVM lowering chapter](../18-mlir-lowering-to-llvm/), the
[BCIR runtime-boundary lesson](../bcir-mapping/09-runtime-call-boundaries.md),
and the [binary-analysis chapter](../15-binary-analysis/). The code fragments are
architecture sketches: ORC APIs evolve, and production code must be adapted to
its exact LLVM release and executor protocol.

## The advanced layer stack

A runtime-integrated BCIR JIT usually needs more policy than a single `LLJIT`
instance exposes:

```text
BCIR graph + deployment policy + profile snapshot
  -> MLIR dialect selection and conversion to LLVM dialect / LLVM IR
  -> IRTransformLayer (BCIR-aware New PM pipeline)
  -> custom/lazy/speculative compile policy
  -> IRCompileLayer or target-specific artifact producer
  -> ObjectTransformLayer (optional inspection/instrumentation)
  -> ObjectLinkingLayer / JITLink
  -> local or remote executor memory
  -> stable entry stub -> current kernel implementation
  -> telemetry -> reoptimization/replacement decision
```

The layers should preserve evidence needed by the next boundary. For example,
the MLIR lowering records the selected target and ABI; the IR transform records
its pipeline/version; object linking records allocations and symbol addresses;
and the runtime records which implementation generation served each call.

## `IRTransformLayer`: BCIR-aware optimization at materialization time

`IRTransformLayer` sits above an IR layer and receives a `ThreadSafeModule` plus
a `MaterializationResponsibility`. It is a natural place to:

- verify the already-lowered LLVM IR;
- run a BCIR-aware New PM pipeline assembled with `PassBuilder`;
- choose a baseline or optimized pipeline from immutable profile input;
- rewrite BCIR runtime boundaries or instrumentation hooks;
- attach an optimization generation and deployment-policy identifier; and
- reject modules whose triple, data layout, ABI version, or required runtime
  symbols do not match the selected executor.

The transform should be deterministic for a given module, target descriptor,
pipeline version, and profile snapshot. Do not read mutable counters throughout
a pass run: snapshot telemetry first, then compile against that snapshot. The
[`orc-irtransformlayer-bcir.cpp.md`](examples/orc-irtransformlayer-bcir.cpp.md)
sketch shows the ownership flow.

A transform layer is not an MLIR conversion layer. Complete BCIR/MLIR dialect
conversion and legality checks before handing LLVM IR to ORC; see
[`../18-mlir-lowering-to-llvm/`](../18-mlir-lowering-to-llvm/). The transform may
still preserve and consume LLVM metadata produced by that lowering.

## Custom compile layers

Use a custom compile layer when compilation is more than "LLVM IR to one native
object." A BCIR deployment layer may:

1. inspect a target descriptor and kernel capability set;
2. compile native CPU LLVM IR to an object;
3. emit Wasm for a sandbox executor;
4. lower a kernel to a RISC-V object for a remote board;
5. call an FPGA toolchain and return a host-side launch shim plus a bitstream
   handle; or
6. load a content-addressed artifact from a validated cache.

Keep the ORC contract precise. If the lower layer consumes object files, the
custom layer must eventually provide a compatible object. If a target does not
produce a host-linkable object, expose a small object-level adapter whose symbols
implement the BCIR runtime ABI and whose body dispatches to the device artifact.
Do not pretend that a bitstream or Wasm module is a native object merely to fit a
layer interface.

Cache keys should include at least the normalized BCIR graph identity, lowering
pipeline version, LLVM revision/toolchain identity, target triple, data layout,
CPU/features, ABI version, and optimization/profile generation. A cache hit is
usable only after its manifest is checked against the current runtime contract.

## Lazy compilation

Lazy compilation delays work until symbol lookup or first call. ORC can model
lookup-triggered laziness with deferred materialization and call-through
laziness with stable stubs/redirectable symbols. For BCIR kernels:

- define the public kernel name as a stable entry point;
- defer the implementation symbols as a materialization unit;
- compile a baseline implementation on first demand;
- publish the implementation only after all required symbols and runtime ABI
  dependencies are ready; and
- retain the graph/lowering recipe long enough to rematerialize or reoptimize.

Lazy compilation improves startup when many graph kernels are never invoked, but
moves failures into the request path. Preflight dialect legality, target
capability, runtime ABI, and symbol manifests before accepting the graph so the
first call does not discover an avoidable semantic error.

## Speculative compilation

Speculative compilation predicts future demand and starts materialization before
lookup blocks. Useful signals include graph adjacency, queue depth, observed call
sequences, and binary-analysis/profile evidence from
[`../15-binary-analysis/`](../15-binary-analysis/).

Speculation must be cancellable and lower priority than demand compilation. Give
each speculative candidate its own resource tracker and generation token. If the
policy changes before publication, remove or abandon the candidate without
modifying the stable entry point. Never let two candidates race to publish merely
because both compiled successfully; a serialized commit step must choose the
winning generation.

## Hot function re-JIT and symbol replacement

A hot kernel can be recompiled with a profile-guided or target-specialized New PM
pipeline. The safe pattern is indirection:

```text
public bcir.kernel.42
  -> stable stub / redirectable symbol
  -> bcir.impl.42.g7
```

Compile `g8` under a new resource tracker, validate it, atomically redirect the
stable entry to `g8`, wait until the runtime's quiescence rule says no thread can
enter or return through `g7`, and only then remove `g7`'s tracker. Never redefine
a symbol in place while another thread may be executing its old allocation.

Symbol replacement also covers policy changes unrelated to heat: a corrected
implementation, a different target, a newly available accelerator, or a fallback
after device failure. Record the graph ID, implementation generation, target,
pipeline, and telemetry schema together so callers and observations do not refer
to a stale implementation.

## Custom `MaterializationUnit`

A `MaterializationUnit` advertises a symbol interface before producing the
symbols. Its `materialize` method receives a `MaterializationResponsibility` and
must eventually:

- emit the promised symbols through a lower layer;
- delegate a subset to another materialization unit;
- replace definitions using an ORC-supported redirection/reexport mechanism; or
- report failure and fail the responsibility.

Its `discard` hook must handle definitions overridden before materialization.
Custom units are useful for graph families whose target, specialization, or
artifact is selected only when requested. They are also easy to get wrong:
advertising symbols that are never resolved leaves lookups suspended, and
capturing graph state with a shorter lifetime than the unit causes use-after-free
bugs.

The
[`orc-materialization-unit-gaadmsf.cpp.md`](examples/orc-materialization-unit-gaadmsf.cpp.md)
sketch models a GAADMSF graph fragment as a deferred group of kernel and metadata
symbols. Keep one logical materialization responsibility per atomic publication
unit; delegate independent kernels when they may compile separately.

## Resource tracking and retirement

A `ResourceTracker` groups ORC resources so they can be removed together. Create
a tracker for each unloadable kernel generation or deployment artifact, then add
IR, objects, graph allocations, and side metadata through that tracker where the
API permits.

Removal is a lifetime operation, not just a symbol-table edit. Before calling
`remove()`:

1. prevent new calls from selecting the generation;
2. redirect the stable entry or mark it unavailable;
3. wait for active calls, callbacks, and asynchronous materialization to drain;
4. detach telemetry references that contain raw executor addresses;
5. remove ORC resources; and
6. release associated runtime/device artifacts.

A default tracker is convenient for process-lifetime runtime symbols. It is a
poor choice for replaceable kernels because it couples unrelated allocations and
makes selective retirement difficult.

## `JITDylib` layout for BCIR kernels

One defensible layout is:

| `JITDylib` | Contents | Lifetime and policy |
| --- | --- | --- |
| `bcir.runtime` | Stable runtime ABI, telemetry hooks, allocator/device adapters | Process or executor lifetime; allowlisted exports |
| `bcir.entry` | Stable public kernel stubs or reexports | Graph/session lifetime; names do not change across re-JIT |
| `bcir.impl.gN` | Generated implementation symbols for generation `N` | Per-generation resource tracker; normally hidden from clients |
| `bcir.support.<target>` | Target-specific helper objects and launch shims | Target/session lifetime |
| `bcir.speculative` | Unpublished candidate implementations | Short-lived, isolated, removable |

Give generated implementations a lookup path to the runtime/support dylibs, but
avoid making runtime lookup search every generated implementation. Narrow search
orders reduce accidental interposition. Export only the stable entry symbols to
clients; implementation names are diagnostic identities, not ABI.

For multi-tenant graphs, use separate entry/implementation dylibs or stricter
symbol prefixes when tenants must not resolve one another's kernels. The
[`dynamic-kernel-deployment-sketch.md`](examples/dynamic-kernel-deployment-sketch.md)
example follows this layout.

## JITLink integration

`ObjectLinkingLayer` uses JITLink to turn object files or link graphs into
executor allocations. Integration points can inspect or transform a `LinkGraph`
to:

- validate object format, architecture, sections, and external symbol policy;
- register graph passes before allocation/fixup/finalization;
- collect section ranges, symbol addresses, and unwind/debug registrations;
- apply target-specific edges and relocation handling; and
- associate finalized allocations with an ORC resource key for removal.

Keep JITLink diagnostics correlated with graph ID, kernel generation, object
cache key, target triple, and materialization responsibility. A successful link
only establishes that the graph's symbols/edges were allocated and fixed up. It
does **not** establish MLIR conversion legality, BCIR verifier success, runtime
ABI compatibility, bounds safety, or numerical equivalence.

## Remote execution

ORC separates compiler and executor through `ExecutorProcessControl` and related
remote-execution protocols. In a remote configuration, symbol addresses are
executor addresses. They are not pointers that the host compiler process may
dereference or cast to host function pointers.

A remote deployment should negotiate an explicit target descriptor containing:

- target triple and object format;
- LLVM data layout and pointer widths by address space;
- CPU/features or Wasm/RISC-V extension set;
- calling convention, endianness, and stack/alignment constraints;
- runtime ABI and telemetry protocol versions;
- supported memory allocation/protection operations; and
- supported artifact kinds and device capabilities.

Compilation, linking, invocation, telemetry transfer, and resource removal are
separate protocol operations. Treat disconnects and partial failures as normal:
attach idempotency keys to deployment, preserve a state machine, and do not mark
a generation active until the executor acknowledges finalization and entry-point
publication.

## Heterogeneous deployment

The same BCIR graph can have several implementation forms, but they do not share
a single implicit ABI:

| Target | Typical artifact | Runtime boundary |
| --- | --- | --- |
| Native CPU | ELF/Mach-O/COFF object linked by JITLink | Direct stable entry or local runtime call |
| Wasm | Wasm module/component | Typed imports/exports, linear-memory offsets, sandbox handle |
| FPGA | Bitstream plus host launch shim | Queue/buffer/device handle API; asynchronous completion |
| RISC-V | Native object for local or remote JITLink | Executor-target ABI and negotiated ISA extensions |

Target selection belongs before code generation and may begin in MLIR dialect
selection: affine/vector lowering for CPUs, sandbox-compatible lowering for
Wasm, accelerator extraction for FPGA, or RISC-V vector/scalar lowering. Preserve
a common logical kernel signature in a versioned BCIR runtime descriptor, then
adapt it explicitly to each target ABI. The
[`remote-jitlink-heterogeneous-sketch.md`](examples/remote-jitlink-heterogeneous-sketch.md)
example shows the control plane.

## BCIR kernel lifecycle

1. **A BCIR graph arrives.** Parse and semantically validate it, assign a stable
   graph/kernel identity, and snapshot deployment requirements. Do not accept a
   graph merely because a later object linker could resolve its symbols.
2. **MLIR lowering selects a dialect pipeline.** Choose legal conversions and a
   target path using the techniques in
   [`../18-mlir-lowering-to-llvm/`](../18-mlir-lowering-to-llvm/). Record the
   chosen target descriptor and runtime ABI.
3. **LLVM IR is optimized with a BCIR-aware New PM pipeline.** An
   `IRTransformLayer` runs a baseline, profile-guided, or target-specialized
   pipeline assembled as described in
   [`../17-new-pass-manager/`](../17-new-pass-manager/). Verify before and after
   the transform and preserve required BCIR metadata/runtime boundaries.
4. **ORC materializes symbols.** A standard or custom materialization unit
   produces the selected artifact, compile/object layers process it, and JITLink
   or a target adapter publishes an implementation behind a stable entry symbol.
5. **The runtime records telemetry.** Calls, latency, counters, failures, target,
   graph ID, implementation generation, and pipeline version are recorded without
   embedding stale raw addresses as durable identities. Runtime calls follow
   [`../bcir-mapping/09-runtime-call-boundaries.md`](../bcir-mapping/09-runtime-call-boundaries.md).
6. **Hot kernels are reoptimized or replaced.** The policy snapshots telemetry,
   compiles a new generation, validates it, atomically redirects the stable entry,
   waits for quiescence, and retires the old generation with its resource tracker.

## Pitfalls and review checks

### Symbol lifetime errors

A looked-up address remains usable only while its defining resources and any
entry stub remain alive. Do not cache implementation addresses beyond their
resource tracker. Cache the stable entry identity or a versioned executor handle.

### Re-JIT races

Compilation completion order is not publication order. Serialize generation
commit, compare the candidate's expected predecessor generation, and use a
runtime quiescence protocol before freeing replaced code.

### Stale runtime metadata

Telemetry keyed only by symbol name can mix generations. Include graph ID,
generation, target, ABI, and pipeline version; invalidate address-bearing side
tables during resource removal.

### Host/target data layout mismatch

Never copy the host module layout into a remote module by convenience. Construct
and validate the module against the executor target descriptor before optimization
or object emission.

### Assuming native pointer size for remote targets

An executor address, a Wasm linear-memory offset, an FPGA buffer handle, and a
host pointer are different types even if all fit in `uint64_t`. Serialize them
with explicit width/address-space/type tags and perform target-side operations
through the executor protocol.

### Treating JITLink success as semantic BCIR validation

JITLink checks link-graph mechanics, not BCIR meaning. Require independent BCIR
validation, MLIR conversion legality, LLVM verification, ABI checks, and—where
needed—differential or conformance tests before publication.

## Example sketches

- [`examples/orc-irtransformlayer-bcir.cpp.md`](examples/orc-irtransformlayer-bcir.cpp.md)
  — materialization-time New PM optimization.
- [`examples/orc-materialization-unit-gaadmsf.cpp.md`](examples/orc-materialization-unit-gaadmsf.cpp.md)
  — deferred graph-family materialization.
- [`examples/dynamic-kernel-deployment-sketch.md`](examples/dynamic-kernel-deployment-sketch.md)
  — lazy/speculative deployment, generations, and retirement.
- [`examples/remote-jitlink-heterogeneous-sketch.md`](examples/remote-jitlink-heterogeneous-sketch.md)
  — executor negotiation and native/Wasm/FPGA/RISC-V dispatch.
