# Volatile vs atomic

## TL;DR

`volatile` and `atomic` answer different questions:

| Modifier | Controls | Does not provide |
|---|---|---|
| `volatile` | Whether a memory access remains observable and is not removed, merged, or freely reordered by the optimizer | Inter-thread synchronization, happens-before edges, or data-race safety |
| `atomic` | Atomicity and memory-ordering constraints between threads | Device/MMIO access semantics by itself |

They are orthogonal. An access can be plain, volatile, atomic, or both
volatile and atomic.

Official references:

- [LLVM Atomic Instructions and Concurrency Guide](https://llvm.org/docs/Atomics.html)
- [LangRef: volatile memory accesses](https://llvm.org/docs/LangRef.html#volatile-memory-accesses)
- [LangRef: Atomic Memory Ordering Constraints](https://llvm.org/docs/LangRef.html#atomic-memory-ordering-constraints)

## Plain access

```llvm
%v = load i32, ptr %p, align 4
store i32 %v, ptr %p, align 4
```

Plain accesses are best for ordinary local memory or memory whose
thread-safety is provided elsewhere, such as behind a mutex.

## Volatile access

```llvm
@STATUS = external addrspace(1) global i32

%status = load volatile i32, ptr addrspace(1) @STATUS, align 4
store volatile i32 1, ptr addrspace(1) @STATUS, align 4
```

Use volatile for memory-mapped I/O, signal-like externally observable
accesses, or ABI cases where the access itself must happen as written.
Do **not** use volatile as a substitute for a lock or an atomic flag.

## Atomic access

```llvm
store i32 123, ptr @payload, align 4
store atomic i8 1, ptr @ready release, align 1

%r = load atomic i8, ptr @ready acquire, align 1
```

Atomic accesses are the tool for inter-thread atomicity and ordering.
The ordering keyword (`monotonic`, `acquire`, `release`, `acq_rel`, or
`seq_cst`) is what communicates the synchronization contract.

## Volatile and atomic together

```llvm
%old = atomicrmw volatile add ptr @device_counter, i32 1 seq_cst, align 4
```

This says both things: the access is volatile, and the operation is an
atomic read-modify-write with sequentially consistent ordering. Only use
both when you truly need both observable-access behavior and atomic
ordering.

## Common pitfalls

- **Using `volatile` for thread safety.** A volatile load/store is not an
  acquire/release synchronization operation.
- **Forgetting that atomics still need alignment.** Atomic examples
  should include explicit `align N` so the IR states the memory contract.
- **Assuming `atomic` means `seq_cst`.** LLVM requires an ordering; choose
  the one that matches the pattern.
- **Using volatile to prevent all optimization.** Volatile constrains the
  marked access, not arbitrary surrounding computation.
- **Ignoring the source language model.** LLVM IR should preserve the
  frontend language's concurrency rules; do not invent stronger or weaker
  semantics accidentally.

## Rule of thumb

- Hardware register or externally observed access? Start with `volatile`.
- Shared data between threads? Start with `atomic` or a lock.
- Both hardware-observable and shared? Consider volatile atomic, but make
  the requirement explicit in comments or frontend lowering notes.

## See also

- [`01-atomic-orderings.md`](01-atomic-orderings.md)
- [`02-atomic-instructions.md`](02-atomic-instructions.md)
- [`../04-memory/02-load-store.md`](../04-memory/02-load-store.md)
