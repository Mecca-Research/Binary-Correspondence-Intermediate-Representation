# Atomic orderings

## TL;DR

LLVM IR has both ordinary memory operations and atomic memory operations:

```llvm
; Not atomic: no cross-thread atomicity or synchronization contract.
%plain = load i32, ptr %p, align 4
store i32 7, ptr %p, align 4

; Atomic: an ordering keyword is part of the instruction.
%a = load atomic i32, ptr %p acquire, align 4
store atomic i32 7, ptr %p release, align 4
```

Use the weakest ordering that states the property you actually need.
`seq_cst` is a safe starting point while learning, but production IR
should avoid making every atomic sequentially consistent unless the
program really requires one global order.

Official references:

- [LLVM Atomic Instructions and Concurrency Guide](https://llvm.org/docs/Atomics.html)
- [LangRef: Atomic Memory Ordering Constraints](https://llvm.org/docs/LangRef.html#atomic-memory-ordering-constraints)
- [LangRef: Atomic instructions](https://llvm.org/docs/LangRef.html#memory-access-and-addressing-operations)

## Ordering ladder

| Ordering | Where it appears | What it is for |
|---|---|---|
| Not atomic | Plain `load`/`store` | Single-threaded memory, stack slots, or data protected by some other synchronization. No atomicity or inter-thread ordering is attached to the access. |
| `unordered` | Atomic load/store only | Atomic access without synchronization. Useful for languages that need loads/stores not to tear but do not want ordering edges. |
| `monotonic` | Atomic load/store/RMW/CAS | Coherent atomic access to one location, but no ordering of other memory. Similar to C/C++ relaxed atomics. |
| `acquire` | Loads, successful `cmpxchg`, RMW, fences | After an acquire observes a matching release, later operations in this thread cannot be moved before it for synchronization purposes. |
| `release` | Stores, successful `cmpxchg`, RMW, fences | Earlier operations in this thread are published before the release operation. |
| `acq_rel` | RMW, successful `cmpxchg`, fences | Both acquire and release. Not valid for a plain load or plain store. |
| `seq_cst` | Atomic operations and fences | Acquire/release-style constraints plus participation in one sequentially consistent order. Strongest and often most expensive. |

The exact rules are in LangRef; this file is a practical map for
reading and writing examples.

## Not atomic

Plain memory operations have no ordering keyword:

```llvm
%old = load i32, ptr @plain_count, align 4
%new = add i32 %old, 1
store i32 %new, ptr @plain_count, align 4
```

This is **not** a thread-safe increment. If multiple threads can touch
`@plain_count` without synchronization, use an atomic operation or add a
higher-level synchronization mechanism such as a lock.

## `unordered`

```llvm
%v = load atomic i32, ptr %p unordered, align 4
store atomic i32 %v, ptr %p unordered, align 4
```

`unordered` is weaker than `monotonic`. It prevents tearing for the
atomic object, but it is not a synchronization primitive. It is rarely
what hand-written IR wants unless a frontend is modeling a language with
specific unordered atomic requirements.

## `monotonic`

```llvm
%old = atomicrmw add ptr @counter, i64 1 monotonic, align 8
```

Use `monotonic` for counters, statistics, unique IDs, and other patterns
where each operation must be atomic but no other data is being published
or consumed through the operation.

## Acquire and release

A common flag pattern is:

```llvm
; Producer: write payload, then publish readiness.
store i32 123, ptr @payload, align 4
store atomic i8 1, ptr @ready release, align 1

; Consumer: acquire the flag before reading payload.
%r = load atomic i8, ptr @ready acquire, align 1
%ready = icmp ne i8 %r, 0
br i1 %ready, label %read_payload, label %not_ready
```

If the consumer's acquire load observes the producer's release store,
the consumer can then read the payload with the synchronization intended
by the source program.

## Acquire-release

`acq_rel` belongs on read-modify-write operations or fences:

```llvm
%old = atomicrmw xchg ptr @state, i32 1 acq_rel, align 4
```

It is useful when an operation both consumes state written by another
thread and publishes state for another thread.

## Sequentially consistent

```llvm
store atomic i32 1, ptr @flag seq_cst, align 4
%v = load atomic i32, ptr @flag seq_cst, align 4
```

`seq_cst` is easiest to reason about because all sequentially consistent
operations participate in a single order. The pitfall is cost and lost
optimization freedom. Prefer `monotonic`, `acquire`, `release`, or
`acq_rel` when they express the actual contract.

## Choosing orderings for common patterns

| Pattern | Typical ordering | Why |
|---|---|---|
| Statistics counter | `atomicrmw add ... monotonic` | Atomic increment is needed; no payload is published. |
| Reference count increment | Often `monotonic` | Increment usually only needs atomicity. Destruction paths often need stronger ordering. |
| Publish payload then set ready flag | `store` payload normally, `store atomic ... release` flag | Release prevents payload writes from being reordered after publication. |
| Wait for ready flag then read payload | `load atomic ... acquire` flag | Acquire pairs with release before reading payload. |
| Lock acquire with CAS | `cmpxchg ... acquire monotonic` or stronger success ordering | Successful CAS acquires protected data; failure usually only retries. |
| Lock release | `store atomic ... release` | Publishes writes made while holding the lock. |
| Compare-exchange update loop | Success: `monotonic`, `acquire`, `release`, or `acq_rel` depending on what is updated; failure: usually `monotonic` or `acquire` | Failure ordering applies to the load part when the comparison fails. |

## Pitfalls

- **Using `volatile` for thread safety.** `volatile` affects observable
  access behavior; it does not create acquire/release synchronization.
- **Invalid `cmpxchg` failure ordering.** Failure ordering cannot be
  `release` or `acq_rel`, and it cannot be stronger than the success
  ordering.
- **Missing alignment on atomics.** Put an explicit `align N` on atomic
  examples. Alignment is part of the contract you give LLVM.
- **Overusing `seq_cst`.** It is correct for many simple examples, but
  it can pessimize code and hide the real synchronization pattern.

## See also

- [`02-atomic-instructions.md`](02-atomic-instructions.md) — instruction syntax
- [`03-volatile-vs-atomic.md`](03-volatile-vs-atomic.md) — why volatile is not synchronization
- [`examples/atomic-counter.ll`](examples/atomic-counter.ll)
- [`examples/cmpxchg-loop.ll`](examples/cmpxchg-loop.ll)
- [`examples/fence.ll`](examples/fence.ll)
