# Fast-Math Flags

Fast-math flags (FMFs) are per-instruction promises on floating-point operations.
They let LLVM ignore parts of strict IEEE floating-point behavior, which can
unlock reassociation, vectorization, fused operations, reciprocal transforms, and
library-call substitutions. They are also easy to misuse: an incorrect fast-math
flag can make valid inputs produce poison or let optimizers remove behavior that
BCIR needed to preserve.

## Flag summary

| Flag | Promise / permission | Common consequence |
| --- | --- | --- |
| `nnan` | No NaN operands or results need to be considered. | Optimizers may assume comparisons involving NaN cases are irrelevant; a NaN where prohibited can become poison. |
| `ninf` | No positive or negative infinity operands or results need to be considered. | Infinite overflow/inputs can be ignored or treated as impossible. |
| `nsz` | The sign of zero is insignificant. | `+0.0` and `-0.0` may be interchanged; transforms may drop negative-zero distinctions. |
| `arcp` | Reciprocal approximations are allowed. | Division may be rewritten as multiply by reciprocal, often with different rounding. |
| `contract` | Floating multiply and add may be fused where legal. | `fmul` + `fadd` may become FMA with one rounding instead of two. |
| `afn` | Approximate math functions are allowed. | Calls/intrinsics such as exp, log, sin, or sqrt-like operations may use approximations when otherwise legal. |
| `reassoc` | Reassociation is allowed. | `(a + b) + c` may become `a + (b + c)` or a tree reduction. |
| `fast` | Shorthand for the full aggressive set of fast-math permissions. | Enables broad non-strict transforms; use only when all strict FP details are irrelevant. |

The exact set of instructions that accept FMFs changes by LLVM version and
operation class, but the safe rule is stable: do not attach a flag unless the
source semantics prove that flag for every dynamic execution of that instruction.

## Syntax examples

```llvm
%sum = fadd nnan ninf float %a, %b
%prod = fmul contract float %x, %y
%wide = fadd reassoc nnan ninf nsz <4 x float> %v0, %v1
%all = fadd fast double %p, %q
```

Flags are attached to the instruction result. A flag on one `fadd` does not
automatically bless every producer or consumer. If a transform clones,
reassociates, or combines operations, the optimizer must preserve only flags that
remain valid on the new operations.

## Floating-point comparisons

Fast-math flags interact with comparisons in two ways:

1. The compared values may have been produced under flags such as `nnan`, `ninf`,
   or `nsz`.
2. Some LLVM versions and instruction forms allow FMFs directly on `fcmp`.

`nnan` is especially important. Ordinary ordered/unordered comparisons differ in
how they treat NaN. If BCIR emits `nnan`, it is saying NaN cases do not need to
be preserved. Optimizers can fold or simplify comparisons in ways that would be
wrong for a source that distinguishes quiet NaNs, signaling NaNs, ordered
predicates, unordered predicates, or exception behavior.

`nsz` affects equality-like reasoning around zero. Code that observes the sign
bit of zero, preserves exact IEEE min/max behavior, serializes FP bits, or maps
FP values back into binary-analysis features should not use `nsz` unless that
loss of distinction is explicitly allowed.

## Reductions

Floating-point addition and multiplication are not associative under strict IEEE
rounding. `reassoc` tells LLVM that regrouping is allowed:

```llvm
%s01 = fadd reassoc float %a, %b
%s23 = fadd reassoc float %c, %d
%sum = fadd reassoc float %s01, %s23
```

With `reassoc`, a scalar loop reduction can be transformed into a different tree,
a vector reduction, or a partially unrolled reduction. This may change low bits,
exception timing, NaN propagation order, signed-zero results, and overflow to
infinity unless the other flags rule those cases out.

If a BCIR reduction represents an exact recovered binary behavior, avoid
`reassoc` and `fast`. If it represents a numerical kernel whose language or user
contract opts into relaxed math, attach only the flags that the contract allows.

## Vectorization and reassociation consequences

FMFs are a major enabler for vectorization:

- `reassoc` lets the vectorizer change the reduction tree and combine partial
  sums in a different order.
- `nnan` and `ninf` remove special-case lanes that would otherwise constrain
  transformations.
- `nsz` lets lane operations ignore whether a transformed zero is `+0.0` or
  `-0.0`.
- `contract` allows vector FMA generation when target lowering and surrounding
  options permit it.
- `arcp` may replace vector division with reciprocal estimates and refinement
  sequences.

These are performance tools, not metadata decorations. In vector IR, one lane
with NaN, infinity, or signed-zero behavior that the original program observes
can make an FMF unsound for the whole vector instruction. For masked or
predicated BCIR lowering, attach flags only to operations whose active lanes all
satisfy the promised facts.

## When BCIR should avoid emitting FMFs

Avoid fast-math flags when any of the following are true:

- The input is recovered from a binary and BCIR is trying to preserve exact
  machine behavior rather than source-language relaxed semantics.
- NaN payloads, quiet/signaling NaN behavior, infinities, exception timing,
  rounding, or signed zero may be observable.
- The result is compared, hashed, serialized, stored into a trace, or used as a
  key where bitwise differences matter.
- A reduction must match the original order for reproducibility.
- The operation feeds a control-flow decision where NaN or signed-zero cases are
  meaningful.
- Lane validity is partial or reconstructed, and inactive lanes may contain
  arbitrary or poison values.
- The only justification is "the optimizer might generate faster code." That is
  not a semantic proof.

## Safer BCIR policy

1. Default to strict floating-point operations with no FMFs.
2. Record relaxed-math intent in BCIR first, including which exceptional cases
   are excluded.
3. Lower only proven flags to LLVM IR; do not collapse all relaxed modes to
   `fast`.
4. Prefer narrow flags (`contract` alone, or `nnan ninf` alone) over `fast` when
   the source contract is narrow.
5. Keep comparisons and reductions under strict semantics unless the source
   explicitly permits changed NaN, signed-zero, rounding, and order behavior.

## Standalone examples

See [`examples/fast-math-flags.ll`](examples/fast-math-flags.ll) for strict,
partially relaxed, and fully `fast` floating-point functions. The examples are
small enough to inspect after optimization with commands such as:

```bash
opt -S -passes=instcombine,reassociate,loop-vectorize examples/fast-math-flags.ll
```
