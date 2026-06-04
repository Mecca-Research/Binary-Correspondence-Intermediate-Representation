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

## Concrete source-to-IR ordering families

The snippets below are intentionally small and show the operation shape a
frontend is trying to preserve. Real C++ and Rust frontends also attach target
data layout, ABI attributes, debug info, and sometimes stronger alignment facts.

### Relaxed / LLVM `monotonic`

Use relaxed atomics for counters or flags where atomicity matters but no
synchronizes-with edge is required.

```cpp
// C++
std::atomic<int> counter;
int old = counter.fetch_add(1, std::memory_order_relaxed);
```

```rust
// Rust
use std::sync::atomic::{AtomicI32, Ordering};
let old = counter.fetch_add(1, Ordering::Relaxed);
```

```llvm
%old = atomicrmw add ptr %counter, i32 1 monotonic, align 4
```

A relaxed load/store pair maps the same way:

```llvm
%v = load atomic i32, ptr %flag monotonic, align 4
store atomic i32 %v, ptr %other monotonic, align 4
```

### Acquire / LLVM `acquire`

Acquire is normally used by a load that observes a release-published value and
then safely reads data initialized before that release.

```cpp
// C++
if (ready.load(std::memory_order_acquire)) {
  use(payload);
}
```

```rust
// Rust
if ready.load(Ordering::Acquire) {
    use_payload(payload);
}
```

```llvm
%ready = load atomic i1, ptr %ready_ptr acquire, align 1
br i1 %ready, label %use_payload, label %exit
```

The acquire constrains memory operations after the load; it is not valid on an
ordinary atomic store because a store has no read side to acquire from.

### Release / LLVM `release`

Release is normally used by a store that publishes prior writes to another
thread that later performs an acquire load.

```cpp
// C++
payload = 42;
ready.store(true, std::memory_order_release);
```

```rust
// Rust
*payload = 42;
ready.store(true, Ordering::Release);
```

```llvm
store i32 42, ptr %payload, align 4
store atomic i1 true, ptr %ready_ptr release, align 1
```

The ordinary payload store is sequenced before the release store in the source
program. The release ordering prevents that payload store from being moved after
the publication store in a way that would break the source memory model.

### Acquire-release / LLVM `acq_rel`

Read-modify-write operations have both a read side and a write side, so they can
carry acquire-release semantics.

```cpp
// C++
int old = state.fetch_or(READY_BIT, std::memory_order_acq_rel);
```

```rust
// Rust
let old = state.fetch_or(READY_BIT, Ordering::AcqRel);
```

```llvm
%old = atomicrmw or ptr %state, i32 %ready_bit acq_rel, align 4
```

For compare-exchange, model success and failure separately:

```cpp
// C++
head.compare_exchange_weak(expected, desired,
                           std::memory_order_acq_rel,
                           std::memory_order_acquire);
```

```rust
// Rust
let _ = head.compare_exchange_weak(expected, desired,
                                   Ordering::AcqRel,
                                   Ordering::Acquire);
```

```llvm
%pair = cmpxchg weak ptr %head, i64 %expected, i64 %desired acq_rel acquire, align 8
%loaded = extractvalue { i64, i1 } %pair, 0
%success = extractvalue { i64, i1 } %pair, 1
```

Failure only reads the old value, so `acquire` is legal for failure while
`release` and `acq_rel` are not.

### Sequentially consistent / LLVM `seq_cst`

Sequential consistency is the strongest portable source ordering and maps to
LLVM `seq_cst` on the corresponding atomic operation.

```cpp
// C++
x.store(1, std::memory_order_seq_cst);
int seen = y.load(std::memory_order_seq_cst);
```

```rust
// Rust
x.store(1, Ordering::SeqCst);
let seen = y.load(Ordering::SeqCst);
```

```llvm
store atomic i32 1, ptr %x seq_cst, align 4
%seen = load atomic i32, ptr %y seq_cst, align 4
```

Use `seq_cst` because the source asked for the global sequentially consistent
order. Do not silently weaken it to acquire/release in a frontend.

### Fences

Fences are standalone synchronization operations. They are not loads or stores,
so keep the source fence shape explicit in IR.

```cpp
// C++
std::atomic_thread_fence(std::memory_order_release);
flag.store(1, std::memory_order_relaxed);
```

```rust
// Rust
std::sync::atomic::fence(Ordering::Release);
flag.store(1, Ordering::Relaxed);
```

```llvm
fence release
store atomic i32 1, ptr %flag monotonic, align 4
```

An acquire fence after a relaxed load is another common pattern:

```llvm
%flag = load atomic i32, ptr %flag_ptr monotonic, align 4
%is_set = icmp ne i32 %flag, 0
br i1 %is_set, label %acquire_path, label %exit

acquire_path:
  fence acquire
  br label %use_payload
```

### Plain and volatile operations are different families

Plain source operations lower to ordinary IR memory operations only when the
frontend has already proved the source program is data-race-free.

```cpp
// C++ non-atomic object, protected by a mutex or confined to one thread.
plain = plain + 1;
```

```rust
// Rust ordinary access through unique or otherwise synchronized access.
*plain += 1;
```

```llvm
%old = load i32, ptr %plain, align 4
%new = add i32 %old, 1
store i32 %new, ptr %plain, align 4
```

Volatile preserves observable accesses, such as memory-mapped I/O polling, but
it does not create an inter-thread synchronization edge.

```cpp
// C++ device-style volatile access, not a thread synchronization primitive.
volatile int *mmio = get_mmio();
int status = *mmio;
```

```rust
// Rust equivalent generally uses core::ptr::read_volatile.
let status = unsafe { core::ptr::read_volatile(mmio) };
```

```llvm
%status = load volatile i32, ptr %mmio, align 4
```
