# 02 — `TypeConverter` and materialization

A `TypeConverter` owns type decisions for conversion patterns. It should be the
single source of truth for how BCIR custom types become LLVM-compatible types.

## Type conversion responsibilities

For a BCIR lowering, a type converter commonly maps:

| Source type | Target type shape |
| --- | --- |
| `!bcir.vertex<kind>` | `i64` vertex ID, pointer to vertex record, or LLVM struct descriptor |
| `!bcir.edge<src,dst>` | edge index, pointer to edge record, or `{i64, i64, ...}` struct |
| `!bcir.register<bank>` | pointer/table entry or ABI integer handle |
| `!bcir.ham_hint` | immediate policy integer plus metadata or intrinsic wrapper operands |
| `!bcir.claim` | metadata attachment, side-table index, or explicit runtime argument |

Do not duplicate these mappings inside each pattern. If two patterns lower the
same type differently, the module will eventually fail translation or silently
lose ABI consistency.

## Source materialization

Source materialization creates a value of the original source type from converted
values when an operation that has not yet been converted still expects the source
type. It is a temporary bridge used during partial conversion.

BCIR use case: after a register handle has been lowered to an `i64` table index,
a still-legal diagnostic operation might expect `!bcir.register`. Source
materialization can wrap the index in a temporary cast-like op until the
diagnostic op lowers.

## Target materialization

Target materialization creates a converted target-typed value for a consumer that
has already been lowered. It is often the bridge from a source-typed producer to
an LLVM-typed consumer.

BCIR use case: a `bcir.vertex` result may be materialized as an `i64` vertex ID
for an LLVM-dialect call while graph traversal patterns are still being ported.

## Argument materialization

Argument materialization handles block and function arguments. This is critical
for function signatures, region arguments, loop-carried values, and affine/scf
staging boundaries.

BCIR examples:

- Function argument `!bcir.graph` becomes a pointer to a graph descriptor.
- Loop-carried `!bcir.vertex` becomes an index or pointer.
- A block argument carrying claim state becomes an `i64` claim ID plus attached
  metadata on branch-like replacements.

## Attribute preservation during type conversion

Types and attributes often share meaning. If `!bcir.vertex<kind = "load">` lowers
to `i64`, the `kind` fact must move somewhere:

- an LLVM metadata node;
- a field in a descriptor struct;
- a side table indexed by vertex ID;
- a debug or diagnostic attachment;
- or an explicit statement that it was planning-only and intentionally removed.

The pitfall is not just dropping attributes; it is dropping them without deciding
whether they were semantic, diagnostic, or optimization-only.

## A worked pair: a type LLVM lacks, and operations it only has as intrinsics

The sketches above are BCIR's own types. Upstream MLIR has the same problem in two
different shapes, and [`examples/complex-and-math-to-llvm.mlir`](examples/complex-and-math-to-llvm.mlir)
runs both down to real LLVM IR so the difference is visible.

**A type the target does not have.** `complex<f32>` is a first-class MLIR type. LLVM IR has
no complex type at all, so the converter has to choose a representation:

```mlir
func.func @cmul(%a: complex<f32>, %b: complex<f32>) -> complex<f32> {
  %r = complex.mul %a, %b : complex<f32>
  return %r : complex<f32>
}
```
```llvm
define { float, float } @cmul({ float, float } %0, { float, float } %1)
```

An anonymous two-element struct, real part first. That choice is the entire content of the
type conversion, and everything downstream follows from it — including the calling
convention, since `{ float, float }` is what the ABI now sees.

The operation lowering follows the representation:

```llvm
  %3 = extractvalue { float, float } %0, 0     ; a.re
  %4 = extractvalue { float, float } %0, 1     ; a.im
  %5 = extractvalue { float, float } %1, 0     ; b.re
  %6 = extractvalue { float, float } %1, 1     ; b.im
  %7 = fmul float %5, %3
  %8 = fmul float %6, %4
  %9 = fmul float %4, %5
  %10 = fmul float %3, %6
  %11 = fsub float %7, %8                      ; re = a.re*b.re - a.im*b.im
  %12 = fadd float %9, %10                     ; im = a.im*b.re + a.re*b.im
  %13 = insertvalue { float, float } poison, float %11, 0
  %14 = insertvalue { float, float } %13, float %12, 1
```

Four multiplies, one subtract, one add: the schoolbook complex product, written out. One
operation in MLIR became eleven in LLVM IR, and no intrinsic was involved — there was
nothing to call, so the converter had to *expand*.

**Notice where the result is built from.** The aggregate starts at
`insertvalue { float, float } poison`, not at zero and not at `undef`. That is the
idiomatic way to construct an aggregate whose fields are all about to be written: starting
from poison says "every field here is meaningless until I fill it", and both fields are
filled immediately. [`../13-advanced-ir/05-poison-undef-freeze.md`](../13-advanced-ir/05-poison-undef-freeze.md)
treats poison as a hazard, which it is — but this is the other face of it, poison used
correctly as a base value.

**Operations the target has, under another name.** `math.sqrt` and `math.exp` need no
representation decision at all, because LLVM IR already has them:

```mlir
  %s = math.sqrt %x : f32
  %e = math.exp %s : f32
```

```llvm
  %2 = call float @llvm.sqrt.f32(float %0)
  %3 = call float @llvm.exp.f32(float %2)
```

A rename, effectively — `math.<op>` to `llvm.<op>.<type>`. Nothing expands, nothing is
chosen, and the `math` dialect exists mainly so that code above the LLVM dialect can spell
these without depending on it.

**The contrast is the lesson.** Two dialects, one conversion pipeline, two completely
different amounts of work: a missing *type* forces a representation choice and an
open-coded expansion, while a missing *spelling* for an operation the target already has is
a table lookup. When you write a `TypeConverter`, the first question is which of those two
situations you are in.

## Minimal converter sketch

```c++
TypeConverter converter;
converter.addConversion([](Type type) { return type; });
converter.addConversion([&](bcir::VertexType type) -> Type {
  return IntegerType::get(type.getContext(), 64);
});
converter.addConversion([&](bcir::RegisterType type) -> Type {
  return LLVM::LLVMPointerType::get(type.getContext());
});
```

Real code should also add materializations and signature conversion helpers so
function and region boundaries do not become ad hoc special cases.
