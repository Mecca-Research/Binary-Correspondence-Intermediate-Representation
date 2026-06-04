# Pitfall 09 — Atomic Ordering Mismatch

## BCIR instance

| Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|
| `runtime/llvm/bcir_ops.ll`; `runtime/llvm/bcir_claim_verify.ll` | `f78ba96`; `08a0011` | `opt -passes=verify <bcir-atomics>.ll -o /dev/null` | Use legal `cmpxchg` failure orderings and preserve acquire/release semantics instead of defaulting to `monotonic`. | [`11-concurrency/01-atomic-orderings.md`](../11-concurrency/01-atomic-orderings.md); [`11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md); [`11-concurrency/03-volatile-vs-atomic.md`](../11-concurrency/03-volatile-vs-atomic.md) |

## The error

For an invalid compare-exchange failure ordering:

```text
atomic load cannot use Release ordering
```

Or:

```text
cmpxchg failure ordering may not be stronger than success ordering
```

For valid-but-wrong synchronization, the symptom is usually a race or flaky
behavior rather than a verifier error:

```text
consumer observes ready == 1 but reads stale payload
```

## Minimal reproducer

Invalid IR:

```llvm
define i32 @bad_cmpxchg(ptr %p, i32 %old, i32 %new) {
entry:
  %pair = cmpxchg ptr %p, i32 %old, i32 %new monotonic release, align 4 ; ❌
  %loaded = extractvalue { i32, i1 } %pair, 0
  ret i32 %loaded
}
```

The failure ordering describes the load that happens when the comparison fails,
so it cannot be `release` or `acq_rel`.

Valid IR with a broken publish/consume protocol:

```llvm
@payload = global i32 0, align 4
@ready = global i8 0, align 1

define void @publish() {
  store i32 42, ptr @payload, align 4
  store atomic i8 1, ptr @ready monotonic, align 1 ; ❌ no release
  ret void
}

define i32 @consume() {
  %r = load atomic i8, ptr @ready monotonic, align 1 ; ❌ no acquire
  %ok = icmp eq i8 %r, 1
  br i1 %ok, label %read, label %empty
read:
  %v = load i32, ptr @payload, align 4
  ret i32 %v
empty:
  ret i32 0
}
```

## Why it happens

Atomic ordering is not decorative syntax. It states which ordering guarantees a
thread can rely on. `monotonic` gives atomicity for one object but does not
publish surrounding writes. `release` publishes preceding writes. `acquire`
consumes a matching release before reading dependent data. `seq_cst` adds a
global sequentially consistent order.

`cmpxchg` has two orderings because success is a read-modify-write, while failure
is only a load. LLVM therefore restricts failure orderings.

## Fix pattern

Use an acquire/release pair for publication:

```llvm
define void @publish() {
  store i32 42, ptr @payload, align 4
  store atomic i8 1, ptr @ready release, align 1
  ret void
}

define i32 @consume() {
  %r = load atomic i8, ptr @ready acquire, align 1
  ; after seeing 1 from the release store, reading @payload is synchronized
  %v = load i32, ptr @payload, align 4
  ret i32 %v
}
```

For `cmpxchg`, choose a failure ordering that is no stronger than the success
ordering and is never `release` or `acq_rel`:

```llvm
%pair = cmpxchg ptr %p, i32 %old, i32 %new acq_rel monotonic, align 4
```

## BCIR-relevant note

When BCIR models binary atomics, carry the source instruction's memory-ordering
semantics explicitly. Do not lower every atomic to `monotonic` just because the
operation is indivisible, and do not upgrade every operation to `seq_cst` unless
the binary or source model really requires it. Both choices can hide bugs in
recovered concurrency behavior.

## See also

- [`../11-concurrency/01-atomic-orderings.md`](../11-concurrency/01-atomic-orderings.md) — choosing orderings
- [`../11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md) — `load atomic`, `store atomic`, `atomicrmw`, and `cmpxchg`
- [`../11-concurrency/03-volatile-vs-atomic.md`](../11-concurrency/03-volatile-vs-atomic.md) — volatile is not synchronization
