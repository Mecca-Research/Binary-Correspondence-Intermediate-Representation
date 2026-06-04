# Atomic instruction syntax

## TL;DR

```llvm
%v = load atomic i32, ptr %p acquire, align 4
store atomic i32 %v, ptr %p release, align 4
%pair = cmpxchg ptr %p, i32 %expected, i32 %desired acq_rel monotonic, align 4
%old = atomicrmw add ptr %p, i32 1 monotonic, align 4
fence acquire
```

Atomic instructions combine an operation with a memory ordering. Some
forms also accept `volatile` and `syncscope("...")`; keep those separate
from the ordering itself.

Official references:

- [LLVM Atomic Instructions and Concurrency Guide](https://llvm.org/docs/Atomics.html)
- [LangRef: `load` instruction](https://llvm.org/docs/LangRef.html#load-instruction)
- [LangRef: `store` instruction](https://llvm.org/docs/LangRef.html#store-instruction)
- [LangRef: `cmpxchg` instruction](https://llvm.org/docs/LangRef.html#cmpxchg-instruction)
- [LangRef: `atomicrmw` instruction](https://llvm.org/docs/LangRef.html#atomicrmw-instruction)
- [LangRef: `fence` instruction](https://llvm.org/docs/LangRef.html#fence-instruction)

## `load atomic`

```llvm
%v = load atomic i64, ptr @value monotonic, align 8
%ready = load atomic i8, ptr @ready acquire, align 1
%sc = load atomic i32, ptr @flag seq_cst, align 4
```

Valid load orderings are `unordered`, `monotonic`, `acquire`, and
`seq_cst`. A load cannot be `release` or `acq_rel` because it does not
publish prior writes.

## `store atomic`

```llvm
store atomic i64 %v, ptr @value monotonic, align 8
store atomic i8 1, ptr @ready release, align 1
store atomic i32 1, ptr @flag seq_cst, align 4
```

Valid store orderings are `unordered`, `monotonic`, `release`, and
`seq_cst`. A store cannot be `acquire` or `acq_rel` because it does not
read a value to acquire from.

## `atomicrmw`

`atomicrmw` atomically reads a memory location, computes a new value,
stores it, and returns the old value.

```llvm
%old = atomicrmw add ptr @counter, i64 1 monotonic, align 8
%prev = atomicrmw xchg ptr @state, i32 1 acq_rel, align 4
%bits = atomicrmw or ptr @flags, i32 8 release, align 4
```

Common operations include `xchg`, `add`, `sub`, `and`, `nand`, `or`,
`xor`, `max`, `min`, `umax`, `umin`, `fadd`, `fsub`, `fmax`, and `fmin`.
Use `monotonic` for simple counters and stronger orderings when the RMW
also publishes or consumes other memory.

## `cmpxchg`

`cmpxchg` compares the current value with an expected value. If they
match, it writes the desired value. It returns a two-field aggregate:
`{ <ty>, i1 }`, where field 0 is the loaded old value and field 1 says
whether the exchange succeeded.

```llvm
%pair = cmpxchg ptr @slot, i32 %expected, i32 %desired acq_rel monotonic, align 4
%old = extractvalue { i32, i1 } %pair, 0
%ok = extractvalue { i32, i1 } %pair, 1
```

The first ordering is the success ordering. The second is the failure
ordering, which applies to the load performed when the comparison fails.
Failure ordering cannot be `release` or `acq_rel`, and it cannot be
stronger than the success ordering.

A retry loop updates the expected value from the loaded old value:

```llvm
entry:
  br label %loop

loop:
  %expected = phi i32 [ %initial, %entry ], [ %old, %again ]
  %desired = add i32 %expected, 1
  %pair = cmpxchg ptr @counter, i32 %expected, i32 %desired monotonic monotonic, align 4
  %old = extractvalue { i32, i1 } %pair, 0
  %ok = extractvalue { i32, i1 } %pair, 1
  br i1 %ok, label %done, label %again

again:
  br label %loop

done:
  ret i32 %desired
```

## `fence`

A fence applies ordering without naming a memory address:

```llvm
fence release
store atomic i8 1, ptr @ready monotonic, align 1

%r = load atomic i8, ptr @ready monotonic, align 1
fence acquire
```

Prefer acquire/release operations on the actual synchronization object
when possible. Use fences when you specifically need to separate the
ordering edge from the load or store instruction.

## Optional `syncscope`

```llvm
%v = load atomic i32, ptr %p syncscope("singlethread") monotonic, align 4
fence syncscope("singlethread") seq_cst
```

A synchronization scope restricts where the ordering applies. If you do
not specify one, LLVM uses the default system scope.

## See also

- [`01-atomic-orderings.md`](01-atomic-orderings.md)
- [`03-volatile-vs-atomic.md`](03-volatile-vs-atomic.md)
- [`../reference/instruction-quickref.md`](../reference/instruction-quickref.md)
