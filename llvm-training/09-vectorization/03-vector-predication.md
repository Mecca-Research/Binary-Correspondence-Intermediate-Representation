# Vector Predication and Masked Execution

Vector predication is the family of techniques that lets vector code handle
conditional lanes, tails, and scalable vector lengths. It matters when a loop is
almost vectorizable but needs per-lane masks or target-specific predicated
instructions.

## Core ideas

| Idea | Meaning |
| --- | --- |
| Mask | A vector of `i1` values saying which lanes are active. |
| Predicated operation | Operation executes or commits only for active lanes. |
| Tail handling | Processing leftover elements when the trip count is not a multiple of vector width. |
| Scalable vector | `<vscale x N x T>`, where hardware chooses the runtime multiple. |

LLVM can represent predication through masked intrinsics, vector selects, target
extension intrinsics, or target-specific lowering after vectorization.

## Recognizable IR shapes

```llvm
%mask = icmp ult <4 x i32> %idxs, %limit
%old = load <4 x i32>, ptr %fallback
%new = add <4 x i32> %a, %b
%merged = select <4 x i1> %mask, <4 x i32> %new, <4 x i32> %old
```

For memory, LLVM also has masked load/store and gather/scatter intrinsic
families. Use target-independent operations where possible; target-specific
intrinsics belong behind a portability boundary.

## When to care

- Trip counts are dynamic and scalar remainder loops dominate runtime.
- The target has native predication, such as SVE-style scalable vectors.
- Branches are lane-local and can be represented as masks instead of CFG splits.
- BCIR lanes have enable bits or validity masks that naturally map to vector
  predicates.

## Pitfalls

- A mask is not a bounds proof for inactive lanes unless the operation is truly
  masked; an ordinary vector load may still touch every lane's address.
- `select` prevents a value from being chosen, but both operands must already be
  safe to compute.
- Scalable vectors are not arrays with a compile-time element count. Avoid code
  that assumes `vscale` is known during IR construction.

## BCIR lowering advice

If BCIR already has lane masks, preserve them as explicit `i1` vectors or mask
values long enough for vector passes and target lowering to see them. Do not
scalarize masks into unrelated branches unless later passes need scalar CFG.
