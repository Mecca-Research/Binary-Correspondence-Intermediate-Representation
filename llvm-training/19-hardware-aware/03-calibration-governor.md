# Calibration Governor State

A calibration governor selects or refines hardware parameters such as pulse
width, voltage/current band, timing window, lane grouping, or thermal budget.
Separate **required state** from **advisory policy**.

## Explicit required state

If a flow must use calibration epoch 17 and pulse width 32 to be correct, pass
those values as operands, load them from an explicit state object, or return them
from a governor runtime call. Version the state layout and define concurrency:
who publishes an epoch, who reads it, and what synchronization makes the update
visible?

A useful state record might contain:

- epoch/generation;
- selected calibration profile;
- bounded pulse parameters;
- measured confidence or error;
- thermal/power budget snapshot;
- fallback mode and status.

Do not infer required state from a metadata node after optimization.

## Advisory governor metadata

[`examples/calibration-governor-metadata.ll`](examples/calibration-governor-metadata.ll)
attaches a preferred governor profile, confidence, and fallback policy to an
explicit runtime call. The call operands still carry the required epoch and
bounds. A pass may use the metadata to specialize code or emit diagnostics; it
must remain correct if the metadata disappears.

Define metadata as a small schema:

- a stable tag/version;
- named fields rather than unexplained positional integers;
- units for numeric values;
- fallback behavior;
- provenance or confidence when policy is measurement-derived.

## Update boundaries

Calibration updates can race with flow execution. Choose one explicit model:

- snapshot state before the pulse and use that epoch through the flow;
- serialize updates and launches in the runtime;
- use atomic publication plus a retry when the epoch changes;
- make calibration immutable for the compiled object/JIT specialization.

LLVM metadata does not establish atomicity, happens-before edges, or device
visibility. Use atomics, fences, calls, or target intrinsics with accurate
side-effect definitions.

## Review questions

1. Which governor fields affect correctness and therefore must be operands/state?
2. Which fields are merely optimization preferences?
3. Can a pass clone, merge, or delete the annotated instruction safely?
4. What happens on unsupported hardware or low confidence?
5. Which layer validates bounds and units?
