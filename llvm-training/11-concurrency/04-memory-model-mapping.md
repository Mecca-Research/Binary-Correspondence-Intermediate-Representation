# Mapping LLVM Atomics to C++ and Rust Memory Models

LLVM IR orderings are close to C++ and Rust atomics, but frontends still need to
map source operations deliberately. LLVM orderings describe constraints on IR
memory operations; C++ and Rust also define source-language data-race rules,
lifetimes, provenance, and library APIs.

## Ordering map

| Source ordering | LLVM ordering | Notes |
| --- | --- | --- |
| C++ `memory_order_relaxed`, Rust `Ordering::Relaxed` | `monotonic` | Atomic and participates in modification order, but adds no synchronization edges. |
| C++ `memory_order_acquire`, Rust `Ordering::Acquire` | `acquire` | Valid for loads, successful `cmpxchg`, and fences. Prevents later memory operations from moving before the acquire. |
| C++ `memory_order_release`, Rust `Ordering::Release` | `release` | Valid for stores, successful RMWs, and fences. Prevents earlier memory operations from moving after the release. |
| C++ `memory_order_acq_rel`, Rust `Ordering::AcqRel` | `acq_rel` | For read-modify-write operations and fences; combines acquire and release. |
| C++ `memory_order_seq_cst`, Rust `Ordering::SeqCst` | `seq_cst` | Strongest ordering; participates in a global sequentially consistent order. |
| Plain non-atomic access | ordinary `load`/`store` | Source language must prove no data race or use atomics. |
| Volatile access | `volatile load`/`volatile store` | Preserves observable access behavior; not a synchronization primitive. |

LLVM also has `unordered`, which is mainly for languages or runtimes that need
atomicity without C++-style synchronization semantics. C++ and Rust frontends
normally map relaxed atomics to `monotonic`, not `unordered`.

## `cmpxchg` success and failure orderings

LLVM `cmpxchg` carries two orderings:

```llvm
%pair = cmpxchg ptr %addr, i32 %expected, i32 %desired acq_rel acquire
```

The first ordering applies when the exchange succeeds. The second applies when
it fails and only performs the load. Because a failed compare-exchange does not
store, its failure ordering cannot be `release` or `acq_rel`. Typical mappings:

| Source operation | LLVM success | LLVM failure |
| --- | --- | --- |
| C++/Rust success `AcqRel`, failure `Acquire` | `acq_rel` | `acquire` |
| success `Release`, failure `Relaxed` | `release` | `monotonic` |
| success `SeqCst`, failure `SeqCst` | `seq_cst` | `seq_cst` |

## Read-modify-write operations

`atomicrmw` operations are both a load and a store. Use:

- `monotonic` for relaxed counters;
- `release` when publishing data through the write side only;
- `acquire` when consuming data through the read side only;
- `acq_rel` when both sides matter;
- `seq_cst` when the source requested sequential consistency.

## Fences

A source fence maps to LLVM `fence` with the same conceptual ordering:

```llvm
fence release
fence acquire
fence seq_cst
```

A release fence plus a later relaxed store is not the same IR shape as a release
store. Preserve the source language's specified pattern unless the frontend or
optimizer has a proof that the transformation is legal.

## C++ frontend notes

- Non-atomic C++ data races are undefined behavior. Lower source atomics to LLVM
  atomics instead of trying to recover with `volatile`.
- `memory_order_consume` is usually treated as acquire by production compilers;
  do not invent a weaker LLVM mapping in a BCIR frontend unless you own the full
  dependency model.
- Keep object size and alignment legal for the target's atomic lowering; illegal
  widths may become libcalls or loops.

## Rust frontend notes

- Rust's `Ordering` variants map naturally to the C++ rows above.
- Rust also has aliasing and mutability rules outside atomics; LLVM atomics do
  not by themselves encode borrow-checker facts.
- `UnsafeCell` is a source-level permission to mutate through shared references;
  the actual synchronization still comes from atomics, locks, or other runtime
  protocols.

## BCIR lowering checklist

1. Choose the LLVM ordering from the source operation, not from the target CPU.
2. Use atomic instructions for synchronization and `volatile` only for volatile
   observability.
3. For `cmpxchg`, validate the failure ordering separately.
4. Keep sync scopes explicit if a frontend uses a narrower synchronization domain
   than system-wide memory.
5. Add comments or metadata only after the IR operation itself has the correct
   ordering; comments cannot repair a relaxed operation.
