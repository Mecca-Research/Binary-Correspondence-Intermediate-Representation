# Masked and Interleaved Access Patterns

Simple vectorization examples often use one load, one arithmetic operation, and
one store per iteration. Real loops are noisier: they may store only on selected
lanes, load from strided structures, or require a target-specific cost decision
before vector IR is profitable. This lesson uses two advanced-but-common shapes:

- **Masked load/store**: a vector operation is guarded by a `<N x i1>` predicate
  so inactive lanes do not read or write memory.
- **Interleaved access**: a loop reads or writes fields in an array-of-structs
  layout, then LLVM deinterleaves or interleaves lanes with vector shuffles.

Both patterns are still controlled by the same two questions as the simpler
lessons: is vectorization **legal**, and is it **profitable** for the selected
target?

## Files in this lesson

| File | Role |
|---|---|
| [`examples/masked-load-store-before.ll`](examples/masked-load-store-before.ll) | Scalar conditional copy with `noalias` pointers, aligned memory operations, and loop metadata requesting width 4. |
| [`examples/masked-load-store-after-vectorize.ll`](examples/masked-load-store-after-vectorize.ll) | Cleaned-up vector-body snapshot that uses a vector compare and `@llvm.masked.store`. |
| [`examples/interleaved-access-before.ll`](examples/interleaved-access-before.ll) | Scalar AoS-to-SoA deinterleave loop: input stride 2, output stride 1. |
| [`examples/interleaved-access-after-vectorize.ll`](examples/interleaved-access-after-vectorize.ll) | Cleaned-up vector-body snapshot with a wide load and two `shufflevector` deinterleaves. |
| [`examples/bcir-interleaved-riscv-sketch.ll`](examples/bcir-interleaved-riscv-sketch.ll) | Target-specific scalable-vector sketch for BCIR interleaved lowering boundaries on RISC-V/RVV. |
| [`examples/bcir-avx512-mask-sketch.ll`](examples/bcir-avx512-mask-sketch.ll) | Target-specific AVX-512 masked-load/store sketch for BCIR lane masks. |

The `*-after-vectorize.ll` files are teaching snapshots. They are verifier-valid
IR fixtures, but they intentionally omit version-specific runtime checks,
epilogue loops, and exact naming that `opt` may print on a particular LLVM
release.

## Commands and remarks

Run the Loop Vectorizer and ask for passed, analysis, and missed remarks:

```sh
opt -S -passes=loop-vectorize \
  -pass-remarks=loop-vectorize \
  -pass-remarks-analysis=loop-vectorize \
  -pass-remarks-missed=loop-vectorize \
  llvm-training/09-vectorization/examples/masked-load-store-before.ll -o -

opt -S -passes=loop-vectorize \
  -pass-remarks=loop-vectorize \
  -pass-remarks-analysis=loop-vectorize \
  -pass-remarks-missed=loop-vectorize \
  llvm-training/09-vectorization/examples/interleaved-access-before.ll -o -
```

Force a visible width while experimenting:

```sh
opt -S -passes=loop-vectorize \
  -force-vector-width=4 \
  -force-vector-interleave=1 \
  -pass-remarks=loop-vectorize \
  -pass-remarks-analysis=loop-vectorize \
  -pass-remarks-missed=loop-vectorize \
  llvm-training/09-vectorization/examples/masked-load-store-before.ll -o -

opt -S -passes=loop-vectorize \
  -force-vector-width=4 \
  -force-vector-interleave=1 \
  -pass-remarks=loop-vectorize \
  -pass-remarks-analysis=loop-vectorize \
  -pass-remarks-missed=loop-vectorize \
  llvm-training/09-vectorization/examples/interleaved-access-before.ll -o -
```

Then compare with the cleaned-up snapshots:

```sh
cat llvm-training/09-vectorization/examples/masked-load-store-after-vectorize.ll
cat llvm-training/09-vectorization/examples/interleaved-access-after-vectorize.ll
```

Use forced width as a microscope, not as a performance recommendation. The
natural vector width and interleave count depend on the selected triple,
subtarget features, data layout, and LLVM version.

## Masked conditional stores

The scalar input conditionally stores only values above a threshold:

```llvm
%value = load i32, ptr %src.p, align 16
%keep = icmp sgt i32 %value, %threshold
br i1 %keep, label %store, label %latch
```

A vectorized form computes the condition for all lanes and passes the resulting
`<4 x i1>` mask to a predicated memory operation:

```llvm
%values = load <4 x i32>, ptr %src.p, align 16
%mask = icmp sgt <4 x i32> %values, %threshold.splat
call void @llvm.masked.store.v4i32.p0(<4 x i32> %values, ptr %dst.p, i32 16, <4 x i1> %mask)
```

Important details:

- The ordinary vector load is legal here because every lane reads from `%src`.
  Only the store is conditional.
- A different loop shape, such as `if (i < limit) load src[i]`, may need a
  masked load or scalarization because inactive lanes must not read memory.
- Predication usually helps when the target has efficient masked operations or
  when scalarizing the condition would be worse than keeping a vector body.

## Interleaved AoS-to-SoA accesses

The scalar input reads adjacent pairs from one buffer and writes the fields to
two separate output arrays:

```llvm
%base.index = shl nuw nsw i64 %i, 1
%x = load i32, ptr %x.p, align 8
%y = load i32, ptr %y.p, align 4
store i32 %x, ptr %xs.p, align 4
store i32 %y, ptr %ys.p, align 4
```

A vectorized teaching shape loads four pairs at once and separates lanes:

```llvm
%wide = load <8 x i32>, ptr %pairs.p, align 8
%xs.vec = shufflevector <8 x i32> %wide, <8 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
%ys.vec = shufflevector <8 x i32> %wide, <8 x i32> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
```

On some targets, LLVM may lower this pattern to dedicated interleaved-load or
shuffle instructions. On others, the shuffle cost can outweigh the benefit of a
wide load, so the Loop Vectorizer may leave the loop scalar unless you force an
experiment.

## Legality and profitability signals

When a vectorization remark is surprising, check these signals first:

| Signal | Why it matters in these examples |
|---|---|
| Alignment | The fixtures use explicit `align` operands. Higher, truthful alignment can reduce load/store cost; overstated alignment is undefined behavior if callers violate it. |
| Aliasing | `noalias` on `%src`, `%dst`, `%pairs`, `%xs`, and `%ys` tells LLVM that vector stores cannot clobber later vector loads through another pointer. Without this, LLVM may need runtime checks or may reject vectorization. |
| Stride | The masked copy uses unit-stride `%src[i]` and `%dst[i]`. The deinterleave loop has stride-2 input fields but unit-stride outputs, so LLVM must recognize the interleaved group and model shuffles. |
| Predication | Conditional memory effects require a lane mask. The mask is a legality device because inactive lanes must not perform stores, and it is a profitability question because masked operations can be expensive. |
| Target cost model | The same legal loop may be vectorized, scalarized, or left alone depending on vector width, masked-memory support, shuffle costs, and interleave count for the selected target. |

## BCIR pipeline placement

Teach masked and interleaved access as a late BCIR pipeline component: semantic lowering first, poison-sensitive SCCP repair with `freeze` where needed, `loop-rotate` after BCIR loop lowering, then vectorization and target-specific mask/scalable-vector lowering. The optimization chapter shows the complete review checklist and commands such as `opt -passes='sccp,loop-rotate,loop-vectorize,verify'` in [`../07-optimization/08-deep-optimization-lessons.md#putting-advanced-passes-into-a-bcir-pipeline`](../07-optimization/08-deep-optimization-lessons.md#putting-advanced-passes-into-a-bcir-pipeline). The New Pass Manager chapters explain why custom passes that rewrite memory or loops must preserve or invalidate analyses explicitly; start with [`../17-new-pass-manager/01-passbuilder-and-pipelines.md`](../17-new-pass-manager/01-passbuilder-and-pipelines.md).

### Target caveats

- **RISC-V/RVV scalable vectors:** `<vscale x N x T>` values do not have a compile-time lane count. The sketch in [`examples/bcir-interleaved-riscv-sketch.ll`](examples/bcir-interleaved-riscv-sketch.ll) keeps BCIR lane vectors and masks explicit so target lowering can choose RVV instructions without hard-coding physical lanes.
- **AVX/AVX-512 masks:** AVX-512 masked memory operations can suppress inactive lanes, but AVX/AVX2-style lowering may use blends or scalar fallback. The sketch in [`examples/bcir-avx512-mask-sketch.ll`](examples/bcir-avx512-mask-sketch.ll) is a boundary example, not a profitability guarantee.

## Fixture policy

The main before/after `.ll` files in this lesson are intentionally included in the normal
standalone LLVM IR manifest. Target-specific `bcir-*-sketch.ll` files are documented in the global examples README as target-specific teaching sketches. They should assemble and pass `opt -passes=verify`
like other non-`invalid` examples. If an example is later changed into a
version-specific or intentionally broken diagnostic fixture, rename it according
to the invalid-example convention instead of leaving it as a normal `.ll` file.

## See also

- [`03-vector-predication.md`](03-vector-predication.md) — masks and tail handling.
- [`04-vectorization-legality.md`](04-vectorization-legality.md) — legality blockers and runtime checks.
- [`05-example-walkthroughs.md`](05-example-walkthroughs.md) — command patterns and remark flags.
- [`06-recognizing-vector-ir.md`](06-recognizing-vector-ir.md) — IR clues such as vector types, `shufflevector`, and reductions.
- [`../07-optimization/08-deep-optimization-lessons.md#putting-advanced-passes-into-a-bcir-pipeline`](../07-optimization/08-deep-optimization-lessons.md#putting-advanced-passes-into-a-bcir-pipeline) — BCIR advanced-pass pipeline placement.
- [`../17-new-pass-manager/05-mlgo-and-profile-guided-pipelines.md`](../17-new-pass-manager/05-mlgo-and-profile-guided-pipelines.md) — profile and MLGO policy inputs.
