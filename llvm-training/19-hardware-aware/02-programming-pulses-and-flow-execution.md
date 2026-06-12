# Programming Pulses and Flow Execution

## Programming pulse

A GAADMSF programming pulse configures a hardware-facing execution context: for
example a graph window, lane mode, pulse width, calibration epoch, or memory
policy. Model fields that affect execution as explicit operands. A pulse token or
context handle should flow to later operations when ordering and identity matter.

[`examples/gaadmsf-programming-pulse.ll`](examples/gaadmsf-programming-pulse.ll)
shows a custom intrinsic-shaped pulse with constant mode fields and an explicit
returned context ID. The metadata records advisory affinity and memory policy;
the required pulse parameters are not hidden in metadata.

A production lowering has three reasonable implementations:

1. select a target pseudo that later encodes one or more device instructions;
2. rewrite to a runtime call that programs the device or queue;
3. reject the target with a diagnostic if neither path exists.

Do not scalarize a pulse into unrelated stores unless the target ABI explicitly
defines those stores and their ordering.

## Flow execution

Flow execution consumes a prepared context plus graph/data addresses, executes a
bounded unit of work, and returns status or progress. The wrapper in
[`examples/dragon-egg-flow-execution.ll`](examples/dragon-egg-flow-execution.ll)
keeps context, buffers, element count, and flags explicit.

Treat these aspects independently:

- **Value semantics:** addresses, counts, status, and context IDs are ordinary IR
  values.
- **Ordering:** use the intrinsic/runtime memory-effects contract and explicit
  synchronization; a metadata attachment is not a fence.
- **Policy:** affinity, preferred queue, or cache residency may be metadata.
- **Target realization:** instruction selection, pseudo expansion, or a runtime
  symbol owns the actual launch sequence.

## Pulse/flow sequencing pattern

A lowering should preserve the conceptual chain:

```text
calibration state -> programming pulse -> context/token -> flow execution -> status
```

An optimizer may only reorder or remove nodes according to their declared side
effects. If pulse and flow communicate through inaccessible device state, the
intrinsic definitions or runtime calls must expose enough effects to prevent
unsafe motion.

## Failure modes

- Storing a required pulse mode only in metadata makes execution depend on an
  annotation that legal transforms may drop.
- Marking a state-changing pulse `readnone`/`memory(none)` can make it dead-code
  eliminable even though hardware state changes.
- Replacing a target-specific flow with a generic call without preserving its
  synchronization and buffer effects changes semantics.
- Assuming the `%context` SSA name denotes a dedicated hardware register
  confuses IR values with post-selection allocation.
