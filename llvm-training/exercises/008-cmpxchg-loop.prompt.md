# Exercise 008: `cmpxchg` increment loop

Write a standalone LLVM IR module that atomically increments an `i32` in memory
using a compare-exchange loop:

```llvm
define i32 @atomic_increment(ptr %addr)
```

The function should return the new value after a successful increment.

## Required LLVM constructs

- An initial atomic load.
- A loop with a `phi` for the expected value.
- `add` to compute the desired value.
- `cmpxchg` with success ordering `acq_rel` and failure ordering `monotonic`.
- `extractvalue` to read both the observed value and success flag.
- A conditional branch that retries on failure.

## Expected observation

The module assembles successfully. The loop retries if another thread changes
`*addr` between the load and `cmpxchg`.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/008-cmpxchg-loop.solution.ll
```
