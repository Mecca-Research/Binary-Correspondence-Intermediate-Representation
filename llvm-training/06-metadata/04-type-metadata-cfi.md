# Type Metadata and CFI-Style Checks

## TL;DR

`!type` metadata lets an IR producer attach type identifiers to global objects
or functions. LLVM's type-test machinery can then ask whether a runtime pointer
belongs to the set of addresses associated with one identifier. That is the IR
shape behind several Control Flow Integrity (CFI) and whole-program
devirtualization patterns: tag valid targets, test an indirect target or vtable
address point, and branch to a trap before making the indirect call.

Official references:

- [LLVM Type Metadata](https://llvm.org/docs/TypeMetadata.html)
- [LLVM LangRef — Metadata](https://llvm.org/docs/LangRef.html#metadata)
- [Clang Control Flow Integrity](https://clang.llvm.org/docs/ControlFlowIntegrity.html)

See the standalone example:
[`examples/type-metadata-cfi.ll`](examples/type-metadata-cfi.ll).

## What `!type` attaches to

Unlike `!dbg`, `!prof`, or `!llvm.loop`, `!type` is commonly a **global object
metadata attachment**. It is placed on a global variable or function definition,
not on the indirect-call instruction itself:

```llvm
@vtable.Widget = internal constant { ptr } { ptr @widget_do }, !type !0
!0 = !{i64 0, !"typeid.Widget"}
```

The metadata tuple has two operands:

| Operand | Meaning |
|---|---|
| byte offset | Offset from the start of the global to the address being classified. Vtable examples often classify an ABI-specific address point. Small teaching examples may use offset `0`. |
| type identifier | Metadata object naming the set. A metadata string such as `!"typeid.Widget"` is common in hand-written examples. |

Multiple `!type` attachments can appear on the same global when one address is
valid for more than one type identifier, or when different offsets in a larger
vtable object are valid for different base-subobject views.

## Type identifiers as pointer sets

A useful way to read `!type` is: "the address `@global + offset` is a member of
set `typeid`". LLVM can collect all members for a type identifier during LTO or
whole-program analysis, then lower membership tests into compact bitsets, range
checks, jump tables, or constants where possible.

For a C++-like vtable pattern, the frontend does not merely say "this object has
class `Widget`". Instead it tags the vtable address point that a virtual pointer
may hold. A call site can load the object's vtable pointer and test that pointer
against the expected type identifier before loading a function pointer from the
vtable.

This relationship is why type metadata is useful for both:

- **CFI hardening** — reject unexpected vtable or function-pointer targets before
  an indirect control transfer.
- **Whole-program devirtualization** — restrict the possible target set for a
  virtual call and turn it into a direct call, guarded call, or smaller dispatch
  where legal.

## `llvm.type.test`

`llvm.type.test` asks whether a pointer is associated with a type identifier:

```llvm
declare i1 @llvm.type.test(ptr, metadata) nounwind readnone

%ok = call i1 @llvm.type.test(ptr %vptr, metadata !"typeid.Widget")
br i1 %ok, label %checked, label %trap
```

In source-like pseudocode, the pattern is:

```text
vptr = obj->vptr
if (!type_test(vptr, "Widget")) trap()
fn = vptr[0]
return fn(obj)
```

The intrinsic is not the final machine-code check. It is a high-level IR marker
that later whole-program or LTO lowering can replace with an efficient target
membership test. If the relevant global definitions are unavailable, LLVM may be
unable to lower the test as precisely as it could in a closed-world build.

## `llvm.type.checked.load`

`llvm.type.checked.load` combines a type membership test with a load from a
vtable-like address. It is useful when LLVM must correlate a checked virtual-call
site with the vtable slot it loads from, especially for virtual-call visibility
and whole-program devirtualization flows.

Conceptually, it represents:

```text
ok = type_test(vtable_address_point, typeid)
fn = load(vtable_address_point + slot_offset)
```

and returns both the loaded pointer and the validity bit. Use it when you need
that combined checked-load contract; use an explicit `llvm.type.test` plus a
normal `load` when teaching, debugging, or modeling a simpler CFI guard.

## Small vtable-like example

[`examples/type-metadata-cfi.ll`](examples/type-metadata-cfi.ll) defines a tiny
object with a first-field vtable pointer:

```llvm
%Object = type { ptr, i32 }
@vtable.Widget = internal constant { ptr } { ptr @widget_do }, !type !0
!0 = !{i64 0, !"typeid.Widget"}
```

`@dispatch_widget` loads the vtable pointer, checks that it is a member of
`!"typeid.Widget"`, and only then loads the slot and performs the indirect call.
A second untagged vtable is present to show what should fail the check if it is
installed in an object.

## IR-level hardening and binary analysis

Type metadata is an IR-level promise that a later pipeline can turn into binary
hardening. When reviewing the final executable, connect this chapter to
[`../15-binary-analysis/`](../15-binary-analysis/):

- Static IR review should confirm that each indirect call of interest is guarded
  by a type test or checked load before the call edge.
- Codegen/LTO review should confirm that the intrinsic marker did not survive as
  an unresolved call in the final binary and that the lowered check dominates the
  indirect branch or call.
- Dynamic traces and counters can show whether the trap path is cold and whether
  indirect-branch behavior changes after CFI lowering.
- Binary-code similarity features should account for hardening blocks, jump
  tables, and trap edges; these can make two semantically similar functions look
  structurally different.

## Pitfalls

- **Testing the wrong address.** The pointer passed to `llvm.type.test` must
  match the address point encoded by the `!type` offset, not merely any pointer
  into the same global.
- **Inconsistent identifiers.** A spelling mismatch such as `!"Widget"` versus
  `!"typeid.Widget"` creates different sets.
- **Assuming open-world precision.** Precise lowering depends on the compiler
  seeing the relevant tagged globals, usually through LTO or whole-program
  visibility.
- **Using metadata as the only semantic check.** The trap branch and control-flow
  shape enforce the check. Metadata only classifies valid targets for lowering
  and analysis.
- **Forgetting optimizer interaction.** Devirtualization, global layout, and CFI
  lowering can rewrite the obvious IR shape. Inspect optimized IR and post-codegen
  evidence when auditing hardening.

## See also

- [`examples/type-metadata-cfi.ll`](examples/type-metadata-cfi.ll)
- [`01-metadata-basics.md`](01-metadata-basics.md) — attachment syntax and metadata tuples
- [`03-profile-and-optimization-metadata.md`](03-profile-and-optimization-metadata.md) — other optimizer-facing metadata
- [`../15-binary-analysis/README.md`](../15-binary-analysis/README.md) — post-codegen evidence schemas
- [`../15-binary-analysis/02-dynamic-traces-and-counters.md`](../15-binary-analysis/02-dynamic-traces-and-counters.md) — trace/counter evidence
