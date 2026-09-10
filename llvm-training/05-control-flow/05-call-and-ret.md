# Calls and Returns (`call`, `ret`, and the tail-call markers)

## TL;DR

```llvm
%sum = call i32 @add(i32 %a, i32 %b)      ; direct call, result is an SSA value
       call void @side_effect()            ; void call produces no value
%r   = call i32 %fnptr(i32 %a)             ; indirect call through a ptr value
       ret i32 %sum                        ; terminator; type must match the signature
```

A `call` is an ordinary instruction, not a terminator: it does not end its basic
block, and control always returns to the next instruction. Two other
instructions *do* call and *do* terminate — `invoke` (exceptions) and `callbr`
(`asm goto`). `ret` is the only normal way out of a function.

## Syntax

```
<result> = [tail | musttail | notail] call [cconv] [ret-attrs] <ty> [<addrspace>]
           <fnptrval>(<args>) [fn-attrs] [operand bundles]
ret <ty> <value>
ret void
```

- `<ty>` is the **return type**, or the full function type `<ret> (<params>)`
  when the callee is variadic or the signature must be spelled explicitly.
- Under opaque pointers the callee is just a `ptr`. The call site — not the
  pointer — carries the type. This is the single biggest difference from
  pre-LLVM-15 IR: there is no longer a bitcast to a "correct" function-pointer
  type to get wrong.
- A `ret` must be the last instruction in its block, and its type must match the
  function's declared return type exactly. Returning `void` uses `ret void`
  with no operand.

### Spelling a variadic call

Variadic call sites must write the whole function type so the parser knows where
the fixed parameters stop:

```llvm
declare i32 @printf(ptr, ...)

; the type "i32 (ptr, ...)" is required here, not just "i32"
%n = call i32 (ptr, ...) @printf(ptr @.fmt, i32 %value)
```

Omitting the signature on a variadic call is one of the most common
hand-written-IR parse errors.

## Calling conventions

The convention is part of the call contract and must agree between the
declaration and every call site.

| Spelling | Use |
| --- | --- |
| (omitted) / `ccc` | C calling convention; the default |
| `fastcc` | Internal calls the optimizer may re-shape; not ABI-stable |
| `coldcc` | Rarely-taken paths; caller-save biased |
| `tailcc` | Required for guaranteed tail calls in some frontends |
| `cc <n>` | Numbered target-specific conventions (GHC, SPIR kernels, ...) |

A mismatch between the declared convention and the call site is not a verifier
error in every configuration, but it *is* an ABI bug: the arguments end up in
the wrong registers at run time. Treat "the declaration and the call site were
edited separately" as a review finding, not a style nit.

## Tail-call markers

| Marker | Meaning | Backend obligation |
| --- | --- | --- |
| none | ordinary call | none |
| `tail` | the callee does not access the caller's stack frame or its `alloca`s | may become a tail call; may not |
| `musttail` | this call **must** be emitted as a tail call | hard requirement; fails codegen if impossible |
| `notail` | never turn this into a tail call | suppresses the optimization |

`musttail` is a contract with real preconditions, all of which the verifier
enforces:

- the call must be immediately followed by `ret` (returning the call's result,
  or `ret void` when the callee returns `void`);
- the calling conventions must match;
- the parameter and return types must match, modulo the return type when the
  caller returns `void`;
- neither side may use `byval`/`inalloca`/`sret` in incompatible ways, and the
  caller must not be variadic in ways the target cannot forward.

```llvm
define i32 @trampoline(i32 %n) {
entry:
  %r = musttail call i32 @worker(i32 %n)
  ret i32 %r
}
```

`tail` is weaker but has a *semantic* obligation that is easy to violate:
marking a call `tail` promises the callee never reads memory derived from the
caller's stack frame. Passing a pointer to a local `alloca` into a `tail` call
is undefined behavior, and it is the classic way a hand-written or
machine-generated tail marker turns into a use-after-free after inlining.

## Parameter and return attributes that change the ABI

These are not decoration. Each one changes how the value is passed, and each one
must be present on *both* the declaration and the call site.

| Attribute | Effect |
| --- | --- |
| `sret(<ty>)` | The pointer argument receives the return value; the function returns `void` |
| `byval(<ty>)` | The callee gets a private copy of the pointee; the caller allocates it |
| `byref(<ty>)` | The callee gets a read-only alias, no implicit copy |
| `zeroext` / `signext` | The narrow integer is extended by the caller to the target's register width |
| `noalias` | This pointer argument does not alias anything else the call can reach |
| `nonnull`, `dereferenceable(N)`, `align N` | Facts the optimizer may exploit; lying is UB |
| `noundef` | The argument is never `undef`/`poison` |
| `inreg`, `nest`, `inalloca(<ty>)` | Target- or trampoline-specific placement |

`zeroext`/`signext` are the ones that silently miscompile when they drift:
dropping `signext` from an `i8` parameter on a target whose ABI requires it
means the callee reads whatever garbage was in the upper bits.

```llvm
declare void @fill(ptr sret({ i64, i64 }) %out, i32 signext %seed)

define void @use() {
entry:
  %slot = alloca { i64, i64 }, align 8
  call void @fill(ptr sret({ i64, i64 }) %slot, i32 signext 7)
  ret void
}
```

## Returning more than one value

LLVM functions return exactly one value. Multiple results are expressed as a
literal struct return, built with `insertvalue` and consumed with
`extractvalue`:

```llvm
define { i32, i1 } @divmod_checked(i32 %a, i32 %b) {
entry:
  %ok  = icmp ne i32 %b, 0
  %q   = sdiv i32 %a, %b
  %p0  = insertvalue { i32, i1 } poison, i32 %q, 0
  %p1  = insertvalue { i32, i1 } %p0, i1 %ok, 1
  ret { i32, i1 } %p1
}
```

The alternative is an `sret` out-parameter. The struct return keeps the values
in SSA form, which the optimizer handles far better; `sret` is what you use when
a C ABI already fixed the shape.

## Function attributes that make calls optimizable

Attributes on the *callee* decide what the optimizer may do at every call site.

| Attribute | Promise |
| --- | --- |
| `nounwind` | The call never unwinds; no `invoke` needed |
| `willreturn` | The call always returns to the caller (no infinite loop, no `exit`) |
| `memory(none)` (LLVM 16+; `readnone` earlier) | No memory is read or written |
| `memory(read)` (`readonly` earlier) | Memory is read, never written |
| `memory(argmem: readwrite)` | Only memory reachable from the pointer arguments |
| `norecurse`, `nofree`, `speculatable` | Additional narrow promises |

A call to a `memory(none) willreturn nounwind` function with an unused result is
deleted outright. That is the mechanism, and it is also the hazard: an attribute
that is *aspirationally* true is a miscompile waiting for the first pass that
believes it. See [`../13-advanced-ir/04-attributes.md`](../13-advanced-ir/04-attributes.md).

## Calls that terminate their block

| Instruction | Why it terminates | Read |
| --- | --- | --- |
| `invoke` | It has a normal edge and an unwind edge | [`../16-exception-handling/02-itanium-landingpad.md`](../16-exception-handling/02-itanium-landingpad.md) |
| `callbr` | Inline assembly can jump to one of several labels | [`../01-syntax/04-inline-asm.md`](../01-syntax/04-inline-asm.md) |

When a callee may unwind and the caller has cleanups, converting a `call` into
an `invoke` is a control-flow change: every affected block gains a predecessor,
so every downstream `phi` must be updated in the same edit.

## Operand bundles ride on the call site

Deoptimization state, GC state, and funclet membership travel as operand bundles
attached to the call, not as ordinary arguments:

```llvm
%v = call i32 @f(i32 %x) [ "deopt"(i32 %frame_id) ]
```

Bundles constrain what the optimizer may assume about a call — an unknown bundle
is treated conservatively. Dropping one during a transform is a semantic loss
that no type check catches; see
[`../13-advanced-ir/07-operand-bundles.md`](../13-advanced-ir/07-operand-bundles.md).

## Checked example

[`examples/call-and-ret.ll`](examples/call-and-ret.ll) is a standalone module
covering direct calls, indirect calls through a `ptr`, a variadic call site, an
`sret`/`byval` boundary, a struct return, and a `musttail` chain.

```bash
llvm-as llvm-training/05-control-flow/examples/call-and-ret.ll -o /dev/null
opt -passes=verify -S llvm-training/05-control-flow/examples/call-and-ret.ll -o /dev/null
```

## Pitfalls

- **Variadic call without the full type.** `call i32 @printf(ptr @.fmt)` is a
  parse error; write `call i32 (ptr, ...) @printf(...)`.
- **Argument attributes only on the declaration.** `byval`, `sret`, `zeroext`,
  and `signext` must appear at the call site too. Under opaque pointers the
  parser can no longer infer the pointee type for `byval`/`sret`, so the type
  argument is mandatory.
- **`musttail` not followed by `ret`.** The verifier rejects any instruction
  between the `musttail` call and the return.
- **`tail` on a call that receives an `alloca`.** Legal to write, undefined to
  run. This is the marker to audit first when a tail-called function reads
  garbage.
- **`ret` type drift.** Changing a function's return type without updating every
  `ret` in it is a verifier error; changing it without updating every *caller*
  is not, and produces the worse failure.
- **Assuming a `call` ends a block.** Code after a `call` is reachable. Code
  after an `invoke` or `callbr` in the same block is a parse error.
- **Dropping the calling convention when cloning a call.** Cloned call sites
  that lose `fastcc` still verify and still pass arguments in the wrong places.

## BCIR notes

- Runtime boundary calls are the seam where BCIR's memory classes meet: the
  freestanding rail may not hand a caller-stack pointer across a boundary that
  a driver adapter will outlive. See
  [`../bcir-mapping/05-runtime-abi.md`](../bcir-mapping/05-runtime-abi.md) and
  [`../../docs/languages/C_MEMORY_DISCIPLINE.md`](../../docs/languages/C_MEMORY_DISCIPLINE.md).
- A lowered BCIR claim that becomes a runtime wrapper call must keep the
  wrapper's attribute set and calling convention identical on both rails; a
  one-rail attribute edit makes the oracle and the twin disagree about a
  contract neither one verifies at run time.
- `memory(...)` and `noalias` on BCIR runtime calls are *alias facts handed to
  LLVM*. Emit them only from a real analysis result; a blanket `noalias` is the
  exact defect GEM+ slice G9 removed.

## See also

- [`../13-advanced-ir/04-attributes.md`](../13-advanced-ir/04-attributes.md) — the full attribute catalog
- [`../13-advanced-ir/07-operand-bundles.md`](../13-advanced-ir/07-operand-bundles.md) — bundle semantics
- [`../16-exception-handling/README.md`](../16-exception-handling/README.md) — `invoke` and unwinding
- [`06-comparisons-and-select.md`](06-comparisons-and-select.md) — producing the `i1` a branch consumes
- [`../reference/instruction-quickref.md`](../reference/instruction-quickref.md) — one-line syntax reference
