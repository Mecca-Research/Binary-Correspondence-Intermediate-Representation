# Pitfall 10 — `volatile` Is Not Atomic

## BCIR instance

| Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|
| `runtime/llvm/bcir_claim_verify.ll`; `runtime/llvm/bcir_ops.ll` | Unknown | `opt -passes=verify <bcir-volatile-or-atomic>.ll -o /dev/null` (semantic review still required) | Use `volatile` only for observable accesses/MMIO and model inter-thread synchronization with atomic operations and orderings. | [`11-concurrency/03-volatile-vs-atomic.md`](../11-concurrency/03-volatile-vs-atomic.md); [`11-concurrency/01-atomic-orderings.md`](../11-concurrency/01-atomic-orderings.md); [`04-memory/02-load-store.md`](../04-memory/02-load-store.md) |

## The symptom

```text
works at -O0, flakes or races under optimization or on another CPU
```

Thread sanitizers and reviewers may describe the same issue as:

```text
data race on volatile flag
```

The LLVM verifier accepts volatile loads and stores because they are valid IR.
They are just not a synchronization primitive.

## Minimal reproducer

```llvm
@payload = global i32 0, align 4
@ready = global i8 0, align 1

define void @writer() {
  store i32 42, ptr @payload, align 4
  store volatile i8 1, ptr @ready, align 1 ; ❌ observable, not release
  ret void
}

define i32 @reader() {
  %r = load volatile i8, ptr @ready, align 1 ; ❌ observable, not acquire
  %ok = icmp eq i8 %r, 1
  br i1 %ok, label %read, label %empty
read:
  %v = load i32, ptr @payload, align 4
  ret i32 %v
empty:
  ret i32 0
}
```

The volatile accesses keep the flag load/store visible, but they do not create a
happens-before edge for `@payload`.

## Why it happens

`volatile` constrains the optimizer's treatment of a specific memory access. It
is appropriate for memory-mapped I/O, externally observed accesses, and ABI cases
where the access itself must happen. It does not promise atomicity, does not
select an atomic ordering, and does not make neighboring non-volatile accesses
safe between threads.

`atomic` is the IR feature that states inter-thread atomicity and ordering.

## Fix pattern

Use atomics for thread communication:

```llvm
define void @writer() {
  store i32 42, ptr @payload, align 4
  store atomic i8 1, ptr @ready release, align 1
  ret void
}

define i32 @reader() {
  %r = load atomic i8, ptr @ready acquire, align 1
  %ok = icmp eq i8 %r, 1
  br i1 %ok, label %read, label %empty
read:
  %v = load i32, ptr @payload, align 4
  ret i32 %v
empty:
  ret i32 0
}
```

Use `volatile atomic` only when both requirements are true: the access must be
externally observable as written and it must participate in atomic ordering.

## BCIR-relevant note

Binary lifting often sees instructions that look like ordinary loads/stores to
special addresses. If the address is MMIO, `volatile` may be appropriate. If the
instruction implements a lock, atomic flag, or inter-thread protocol, represent
the atomic operation and ordering. Do not conflate hardware-observable I/O with
thread synchronization.

## See also

- [`../11-concurrency/03-volatile-vs-atomic.md`](../11-concurrency/03-volatile-vs-atomic.md) — the main volatile-vs-atomic chapter
- [`../11-concurrency/01-atomic-orderings.md`](../11-concurrency/01-atomic-orderings.md) — acquire/release and `seq_cst`
- [`../04-memory/02-load-store.md`](../04-memory/02-load-store.md) — ordinary load/store syntax
