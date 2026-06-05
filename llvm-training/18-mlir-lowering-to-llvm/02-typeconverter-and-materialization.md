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
