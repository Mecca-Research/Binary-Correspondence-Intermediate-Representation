# SLP Vectorizer

The SLP Vectorizer combines independent scalar operations inside a basic block or
short straight-line region. Unlike the Loop Vectorizer, it does not need a loop;
it searches for isomorphic scalar instruction trees that can become one vector
instruction tree.

## Best input shape

SLP likes repeated scalar lanes such as:

```llvm
%a0 = load i32, ptr %p0
%a1 = load i32, ptr %p1
%s0 = add i32 %a0, %b0
%s1 = add i32 %a1, %b1
```

If the operations have matching opcodes, compatible types, and profitable data
movement, SLP can pack them into vector loads/inserts, vector arithmetic, and
extracts or vector stores.

## Typical command loop

```bash
opt -S -passes='slp-vectorizer' llvm-training/09-vectorization/examples/slp-scalars.ll -o /tmp/slp.ll
opt -passes=verify /tmp/slp.ll -o /dev/null
```

Compare [`examples/slp-scalars-before.ll`](examples/slp-scalars-before.ll) with
[`examples/slp-scalars-after-slp.ll`](examples/slp-scalars-after-slp.ll).

## What to look for in IR

Successful SLP output often has:

- `insertelement` chains that pack scalar values;
- vector `add`, `mul`, `icmp`, or other lane-wise operations;
- `extractelement` if the surrounding ABI still needs scalar results;
- vector stores if the packed result is written contiguously;
- `shufflevector` when lane order has to be rearranged.

## Common misses

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Similar operations stay scalar | Types, flags, or opcodes do not match | Canonicalize with `instcombine` first. |
| Too many inserts/extracts | Packing cost exceeds vector compute benefit | Keep data contiguous or leave scalar. |
| Loads cannot be packed | Pointers are not consecutive or aliasing is unclear | Emit clearer GEPs and alignment facts. |
| One lane has a call | No vectorizable equivalent for that lane | Use an intrinsic or split the sequence. |

## BCIR lowering advice

When lowering batches of independent BCIR lane work, keep lane operations in a
regular order and avoid hiding lane identity behind opaque helper calls. SLP can
recover straight-line SIMD only when the scalar tree remains visible in IR.
