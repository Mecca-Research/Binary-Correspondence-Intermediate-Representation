# Poison, `undef`, and `freeze`

LLVM IR has values that are verifier-valid but not ordinary runtime values.
Understanding them is mandatory before adding arithmetic flags, speculative
control flow, vector transforms, or ABI attributes. The verifier checks shape;
it does not prove that every dynamic execution avoids undefined behavior.

## Three different ideas

| Concept | Short meaning | Safe mental model |
| --- | --- | --- |
| `undef` | A value whose use may pick any bit pattern of the type. Different uses may pick different bit patterns. | "The producer did not define these bits." Do not assume two uses are equal. |
| Poison value | A deferred undefined-behavior marker produced by operations whose semantic promises were violated. | "This computation is already invalid, but UB may be triggered only when the poison reaches certain uses." |
| `freeze` | Instruction that turns `undef` or poison into one fixed, arbitrary concrete value for that dynamic execution. | "Choose once, then reuse that chosen value normally." |

These concepts are intentionally separate. `undef` is not a random number source,
poison is not a trap instruction, and `freeze` is not a sanitizer check. They are
IR semantics that allow optimizers to reason about dead bits, invalid source
program behavior, and speculative rewrites.

## `undef`

`undef` can stand for any value of its type at each use. This means the following
pattern is unsafe as an equality proof:

```llvm
%a = add i32 undef, 1
%b = add i32 undef, 1
%same = icmp eq i32 %a, %b ; may be true or false
```

Even if both operands came from the same textual `undef`, each use may choose a
different concrete value. Common sources of undef-like reasoning include padding
bits, uninitialized allocas after promotion, unused lanes, and values introduced
by transforms that prove the exact bits are irrelevant.

Use `undef` only when *every* possible value of the type is semantically valid
for the remaining computation. If a later branch, memory address, shift amount,
or ABI value needs a stable value, use a defined value or insert `freeze` at the
point where an arbitrary but fixed choice is acceptable.

## Poison values

Poison is produced when an instruction-level promise is violated. Examples:

```llvm
%sum = add nsw i32 %x, 1     ; poison if signed overflow occurs
%q = udiv exact i32 %n, 4    ; poison if %n is not exactly divisible by 4
%p = getelementptr inbounds i32, ptr %base, i64 %i ; poison if inbounds is false
```

The instruction still has a type and the module may assemble. The danger is
semantic: a poison result contaminates most dependent operations and can make the
whole execution undefined once it reaches a poison-consuming use.

## How poison propagates

### Arithmetic and casts

Most ordinary arithmetic, bitwise, cast, and select-like data operations produce
poison if an operand is poison. Flags such as `nsw`, `nuw`, `exact`, `inbounds`,
and floating-point fast-math flags add more ways to create poison because they
are promises about all dynamic executions.

```llvm
%a = add nsw i32 %x, 1 ; may become poison on signed overflow
%b = xor i32 %a, 42    ; poison if %a is poison
%c = trunc i32 %b to i8 ; poison if %b is poison
```

A `select` only propagates poison from the chosen arm, plus poison in the
condition. This makes it useful for avoiding unused poisoned values, but the
condition itself must be non-poison.

### Comparisons

Integer and floating-point comparisons produce poison if a compared operand is
poison. The comparison result is often an `i1`, which is dangerous because it is
frequently fed directly to a branch or select:

```llvm
%sum = add nsw i32 %x, 1
%take = icmp sgt i32 %sum, 0 ; poison if %sum is poison
br i1 %take, label %yes, label %no ; UB if %take is poison
```

Floating-point comparisons also interact with fast-math flags. For example,
marking an `fcmp` or its producers with `nnan` promises that NaNs do not need to
be handled according to ordinary IEEE expectations. See
[`06-fast-math-flags.md`](06-fast-math-flags.md) before using those flags.

### Branches and other immediate UB uses

Some uses cannot accept poison or undef without triggering undefined behavior.
The most common are:

- a `br i1` condition;
- a `switch` condition;
- an `indirectbr` destination;
- a `ret` value or call argument that has `noundef` in the ABI contract;
- memory operations whose pointer, alignment, or dereferenceability promises are
  violated;
- instructions with semantic side conditions such as division by zero, invalid
  shift amounts, or invalid vector indices.

A branch on a poison `i1` is not "either branch may run"; it is undefined
behavior for that dynamic execution. Optimizers may therefore assume it never
happens.

### Vector lanes

Poison is lane-sensitive for vector values. One lane can be poison while other
lanes remain defined:

```llvm
%r = add nsw <4 x i32> %v, <i32 1, i32 1, i32 1, i32 1>
%lane2 = extractelement <4 x i32> %r, i32 2
```

If only lane 2 overflows, extracting or using lane 2 observes poison; using lane
0 may still be defined. This distinction matters for vector predication,
masked operations, SLP vectorization, and BCIR lane-validity lowering. Do not
collapse "one bad lane" into "the whole vector is usable" when a later reduction,
comparison, or branch can read the bad lane.

## `freeze`

`freeze` converts a poison or `undef` operand into a single arbitrary concrete
value of the same type. If the operand was already a concrete value, the result
is that value. If the operand was poison or `undef`, LLVM chooses one value for
that execution and all uses of the `freeze` result see the same value.

```llvm
%x = freeze i32 undef
%a = add i32 %x, %x ; both operands use the same chosen value
```

`freeze` is useful when a transform needs to speculate a value into a context
that cannot accept poison, especially a branch condition. It does not prove the
original program was safe, and it does not preserve every source-language notion
of uninitialized behavior. It simply prevents poison/undef from escaping past the
freeze point.

## BCIR safe speculation with `freeze`

BCIR lowering often starts with conservative, explicit dataflow and then lets
LLVM clean up or if-convert the result. Insert `freeze` at the point where the
lowering intentionally changes from "this value may be poison/undef if a BCIR
precondition failed" to "the following speculative IR needs one stable bit
pattern." The freeze should be close to the speculation boundary so it documents
which use is being protected.

Freeze before these BCIR speculation patterns when the value or mask is derived
from poison-capable arithmetic, recovered binary state, optional lanes, or
partially initialized fields, and when choosing an arbitrary stable value is
permitted by the BCIR operation semantics:

- **Speculative branches and switches.** If a branch condition is computed from
  `nsw`/`nuw` arithmetic, `exact` divisions, `inbounds` pointer tests,
  out-of-range-prone conversions, or other promises that may produce poison,
  freeze the `i1` condition before `br` or `switch` consumes it. Freezing the
  condition is usually narrower than freezing all source operands.
- **Speculative `select` formation.** If lowering or SimplifyCFG-style cleanup
  turns BCIR control flow into a `select`, ensure the select condition is
  non-poison. Also freeze a chosen value before the select only if that value may
  later cross a poison-consuming boundary such as a `noundef` call/return,
  address calculation, or control-flow decision.
- **Vector masks and predicated lanes.** Lane masks used by vector `select`,
  masked memory intrinsics, gathers/scatters, reductions, or scalarized exits
  must not contain poison lanes. Freeze the `<N x i1>` mask, or freeze the scalar
  predicate used to build it, before it controls lane activity. Continue to track
  lane validity separately; `freeze` stabilizes a mask but does not prove inactive
  addresses or values are semantically valid.
- **Metadata-preserving transformations.** Moving a debug-, profile-, or
  provenance-tagged instruction above a guard, merging tagged branches, or
  preserving metadata while if-converting a branch can make a previously guarded
  value speculative. Preserve the metadata on surviving instructions, but insert
  `freeze` before the new control/data speculation point if the tagged value is
  poison-capable.

Do **not** use `freeze` as a blanket repair for missing BCIR proofs. Prefer plain
arithmetic without poison-generating flags, avoid `inbounds` on recovered or
speculative pointers, and keep runtime validity checks when BCIR semantics require
an error, trap, or unchanged value instead of an arbitrary choice. A good review
question is: "Would every stable value of this type be acceptable here if the
original BCIR proof failed?" If the answer is no, add a guard or remove the
poison-producing promise instead of freezing.

See [`examples/bcir-freeze-safe-speculation.ll`](examples/bcir-freeze-safe-speculation.ll)
for a before/after module that freezes scalar branch conditions, vector masks,
and metadata-preserving if-conversion inputs. The repair exercise
[`../exercises/026-poison-freeze-repair.prompt.md`](../exercises/026-poison-freeze-repair.prompt.md)
uses the same branch-safety rule, and the debug metadata exercise
[`../exercises/023-debug-metadata-preservation.prompt.md`](../exercises/023-debug-metadata-preservation.prompt.md)
is a good place to review metadata preservation separately from poison safety.

## Why `noundef` matters at ABI boundaries

`noundef` says the value crossing the boundary is neither `undef` nor poison. It
can be attached to parameters, return values, and call-site operands. At an ABI
boundary, this is more than a local optimization hint:

```llvm
declare noundef i32 @callee(i32 noundef)
```

The declaration promises that callers pass a fully defined `i32` and that the
callee returns a fully defined `i32`. If a caller passes `undef` or poison, the
call has undefined behavior even though the module may be verifier-valid.

For BCIR lowering, use `noundef` when the source/runtime ABI truly requires a
fully initialized, non-poison value. Do not add it merely because the type looks
scalar. Padding, partially initialized aggregates, optional lanes, and recovered
binary values may not be provably defined.

## Verifier-valid but semantically unsafe patterns

The verifier accepts many patterns because it cannot decide all dynamic facts:

| Pattern | Why it is unsafe |
| --- | --- |
| `add nsw` on values that can signed-overflow | Overflow creates poison; later compare/branch/return may be UB. |
| `getelementptr inbounds` for recovered or speculative pointers | If the pointer leaves the original allocated object, the GEP result is poison. |
| Branching on a comparison fed by poison-capable arithmetic | The branch condition can be poison, causing UB. |
| Passing `undef` to a `noundef` parameter | ABI contract violation at the call. |
| Returning a poison-capable value from a `noundef` function | ABI contract violation at return. |
| Treating two uses of `undef` as the same value | Each use can choose independently. |
| Reducing all vector lanes after only some lanes are valid | A poison inactive lane can contaminate the reduction result. |
| Freezing too late | UB may already have occurred if poison reached a branch, store address, or `noundef` boundary before the freeze. |

## BCIR checklist

1. Emit plain arithmetic unless BCIR has proved the stronger `nsw`, `nuw`, or
   `exact` side condition for all dynamic inputs.
2. Track lane validity separately from vector type width; do not reduce or branch
   on lanes that may be poison.
3. Add `freeze` before speculative control-flow use only when choosing an
   arbitrary concrete value is an acceptable semantic weakening.
4. Add `noundef` at runtime and C ABI boundaries only when the lowering has a
   proof that all bits are initialized and no poison can reach the boundary.
5. Prefer explicit validity checks over relying on verifier success. The verifier
   proves syntactic well-formedness, not semantic safety.

## Standalone examples

See [`examples/poison-undef-freeze.ll`](examples/poison-undef-freeze.ll) for a
small module that demonstrates stable `freeze` use, poison-capable arithmetic,
branch conditions, vector-lane propagation, and `noundef` ABI calls. The example
is verifier-valid, but several functions are intentionally labeled as unsafe in
comments to show patterns that optimizers may treat as undefined for some
inputs.
