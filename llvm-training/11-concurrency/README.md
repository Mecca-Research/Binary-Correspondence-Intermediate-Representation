# Concurrency: Atomics, Volatile, and Memory Models

## Key takeaways

- Atomic orderings are part of an operation's synchronization contract; weaker or invalid orderings can be verifier or correctness bugs.
- `volatile` preserves observable memory accesses but does not create inter-thread synchronization.
- `cmpxchg`, `atomicrmw`, `fence`, atomic `load`, and atomic `store` each have distinct ordering constraints.
- Map C++/Rust memory models explicitly rather than inferring semantics from instruction names alone.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Atomic ordering lattice and valid use cases | [`01-atomic-orderings.md`](01-atomic-orderings.md) |
| Atomic load/store, atomicrmw, cmpxchg, and fence syntax | [`02-atomic-instructions.md`](02-atomic-instructions.md) |
| Why volatile is not atomic synchronization | [`03-volatile-vs-atomic.md`](03-volatile-vs-atomic.md) |
| Mapping source-language memory models into LLVM IR | [`04-memory-model-mapping.md`](04-memory-model-mapping.md) |

## Examples

Open the `examples/` directory in this chapter for standalone artifacts and small fixtures that accompany the lessons. Files ending in `.ll` are intended to assemble unless the lesson or filename says they are intentionally invalid.
