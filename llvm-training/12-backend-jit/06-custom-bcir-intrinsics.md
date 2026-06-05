# Custom BCIR Intrinsics in Backends and JITs

Custom BCIR intrinsics are useful when an operation must remain visible as a
single hardware-shaped node until instruction selection. Use them sparingly: a
plain runtime call is easier to link, mock, and JIT unless the backend needs
register-bank constraints, immediate operands, target legalization, or a
pseudo-instruction before normal call lowering.

## When to add a custom intrinsic

Prefer a custom intrinsic when the operation has at least one of these needs:

- **Selection-visible semantics:** instruction selection must recognize a GEM,
  gather, stream, or queue operation rather than rediscovering it from scalar
  arithmetic.
- **Register-oriented operands:** operands are tile fragments, vector registers,
  accumulator registers, masks, or register-bank-specific values.
- **Immediate control fields:** mode, tile shape, memory policy, or hazard-domain
  fields should be validated as literal immediates, not arbitrary runtime values.
- **Target scheduling hooks:** the operation maps to a target pseudo-instruction
  with latency, resource, or hazard data that the machine scheduler should see.
- **JIT policy split:** ORC can select a BCIR-aware backend when available or
  rewrite the intrinsic-shaped call to a runtime ABI fallback before compiling
  for a generic target.

Prefer a runtime call when the operation is an opaque library action, an
accelerator queue submission, or a policy decision outside the compiler backend.

## Declaration shape

A BCIR custom intrinsic declaration should be treated like a contract between
IR generation, backend TableGen, and the JIT layer:

```llvm
; Backend-owned path. The final i32 is an immediate tile/memory mode.
declare <4 x float> @llvm.bcir.gem.mixed.stride.v4f32(
  <4 x float>, <4 x float>, <4 x float>, i32 immarg)

; Portable fallback path with the same register-oriented payload shape.
declare <4 x float> @bcir.runtime.gem.v4f32(
  <4 x float>, <4 x float>, <4 x float>, i32)
```

Keep the intrinsic name, overload suffix, return type, argument types, and
`immarg` positions synchronized with the backend's intrinsic definition. Keep the
runtime fallback close enough that a JIT transform can rewrite the call without
repacking every operand.

## Mixed-stride GEM lowering flow

A hardware-aware mixed-stride GEM lowering normally has two phases:

1. **Layout phase in LLVM IR:** compute A, B, and C addresses with explicit byte
   offsets derived from row, column, reduction, and operand-specific stride
   fields. Load each tile fragment into a vector SSA value.
2. **Backend/JIT phase:** either lower a custom intrinsic call to a target
   pseudo-instruction or rewrite it to a runtime call before code generation.

The key is to keep layout and register shape separate. Mixed-stride address math
belongs in ordinary IR where optimizers can simplify it. Register-oriented tile
semantics belong at the boundary where the backend or JIT can preserve them.

## Register-oriented operand layout

For a GEM-like tile operation, make register grouping explicit before the custom
hook:

- `%a_vec` represents the A fragment in the register order expected by the
  backend selector.
- `%b_vec` represents the B fragment in the corresponding panel/column order.
- `%acc_vec` represents the accumulator fragment.
- The intrinsic returns `%result_vec` in the same register-oriented layout so the
  store or following tile operation does not need to reconstruct the grouping.

This layout avoids hiding hardware intent in memory-only descriptors. If a target
uses scalable vectors, matrix registers, or multiple physical registers per tile,
model that in the backend lowering while keeping the IR-level contract stable.

## Hierarchical memory hint metadata

Attach BCIR metadata to the address calculations, loads/stores, and custom hook
when memory policy matters:

```llvm
%a_addr = getelementptr i8, ptr %a_base, i64 %a_offset, !bcir.memory !0
%tile = call <4 x float> @llvm.bcir.gem.mixed.stride.v4f32(
  <4 x float> %a_vec, <4 x float> %b_vec, <4 x float> %acc_vec, i32 1), !bcir.memory !3

!0 = !{!"bcir.memory", !"level:l1", !"role:gem.a", !4, !5}
!3 = !{!"bcir.memory", !"operation:gem.mixed_stride", !"tile:m4n4k4", !0}
!4 = !{!"reuse", !"scope:k-tile", i32 3}
!5 = !{!"stride", !"row-major-a", !"row_stride:param", !"k_stride:param"}
```

Metadata should be advisory and ignorable. It can guide prefetch insertion,
cache-level policy, memory-space selection, object-layer specialization, or
backend diagnostics, but it must not be required for verifier correctness.

## JIT handling

An ORC-based JIT has three common options:

1. **Compile with a BCIR-aware target:** leave the intrinsic call intact and let
   instruction selection lower it.
2. **Rewrite to a runtime ABI:** run an IR transform before materialization that
   replaces `@llvm.bcir.*` calls with `@bcir.runtime.*` calls and registers the
   runtime symbols in the JITDylib.
3. **Reject with diagnostics:** if no target lowering or runtime fallback exists,
   fail before object emission and report the intrinsic name, tile mode, and
   metadata policy that could not be honored.

Do not wait until the object linker to discover an unresolved backend-only hook.
Resolve the policy before `IRCompileLayer` hands the module to target codegen.

## Example

- [`examples/custom-bcir-intrinsic-jit.ll`](examples/custom-bcir-intrinsic-jit.ll)
  shows the custom intrinsic declaration, runtime fallback declaration,
  register-oriented vector operands, and hierarchical memory hint metadata.
- [`../bcir-mapping/examples/hardware-aware-gem-lowering.ll`](../bcir-mapping/examples/hardware-aware-gem-lowering.ll)
  shows the mixed-stride address computation that feeds the backend hook.

## Verifier and codegen risks

- Unknown custom intrinsic names require backend support before production
  codegen. Keep fallback rewriting available for generic targets.
- `immarg` operands must be literal constants at the call site.
- Metadata can be dropped by transforms unless passes explicitly preserve or
  rebuild it.
- Register-oriented vectors are a contract; changing lane order without updating
  backend lowering creates silent numerical errors.
- JIT symbol resolution must register the fallback runtime name if rewriting is
  enabled.
