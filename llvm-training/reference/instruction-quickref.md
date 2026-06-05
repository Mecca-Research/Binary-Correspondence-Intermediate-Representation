# Instruction Quick Reference

Compact table of every LLVM IR instruction, grouped by category. For
syntax details, see [`../01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md) and the
relevant chapter file.

## Terminators (must be last in basic block)

Every basic block ends in exactly one terminator. EH terminators connect normal
control-flow blocks with exception pads; `callbr` is the unusual terminator used
for calls (most often inline asm) that may branch to labels. See
[`../01-syntax/04-inline-asm.md`](../01-syntax/04-inline-asm.md) for inline asm
constraints and asm-goto label operands.

| Op | Syntax | Successors / result | Notes |
|---|---|---|---|
| `ret` | `ret void` / `ret <ty> <val>` | Function exit | Return from the current function. The value type must match the function return type. |
| `br` | `br label %T` | One successor | Unconditional branch; see [`../05-control-flow/01-unconditional-br.md`](../05-control-flow/01-unconditional-br.md). |
| `br` | `br i1 %c, label %T, label %F` | Two successors | Conditional branch on an `i1`; see [`../05-control-flow/02-conditional-br.md`](../05-control-flow/02-conditional-br.md). |
| `switch` | `switch <ty> %v, label %D [ <ty> <c>, label %L ... ]` | Default plus case successors | Multi-way branch with constant case values; see [`../05-control-flow/03-switch.md`](../05-control-flow/03-switch.md). |
| `indirectbr` | `indirectbr ptr %addr, [ label %A, label %B ]` | Any listed destination | Runtime target must be a valid `blockaddress`; see [`../05-control-flow/04-indirectbr.md`](../05-control-flow/04-indirectbr.md). |
| `invoke` | `%r = invoke <ty> @f(...) [ "deopt"(...) ] to label %N unwind label %U` | Normal and unwind successors; result available on normal edge | Call that transfers exceptions to an EH pad. Operand bundles, when present, appear before `to`; see [`../16-exception-handling/02-itanium-landingpad.md`](../16-exception-handling/02-itanium-landingpad.md) and [`../13-advanced-ir/07-operand-bundles.md`](../13-advanced-ir/07-operand-bundles.md). |
| `callbr` | `%r = callbr <ty> asm "...", "...,!i"(...) to label %N [ label %L ... ]` | Normal successor plus indirect labels; optional result | Call with label destinations, primarily for inline assembly `asm goto`-style control flow. `!i` label constraints consume labels in the bracket list; see [`../01-syntax/04-inline-asm.md`](../01-syntax/04-inline-asm.md). |
| `resume` | `resume <ty> %v` | Function unwind exit | Continue propagation of an exception value produced by `landingpad`; see [`../16-exception-handling/04-cleanups-and-resume.md`](../16-exception-handling/04-cleanups-and-resume.md). |
| `catchswitch` | `%cs = catchswitch within <parent> [ label %H, ... ] unwind label %U` / `unwind to caller` | Handler successors plus unwind edge | WinEH terminator and pad. It must be the only non-`phi` instruction in its block and yields a `token` for `catchpad`; see [`../16-exception-handling/03-wineh-funclets.md`](../16-exception-handling/03-wineh-funclets.md). |
| `catchret` | `catchret from <token> %cp to label %T` | One normal successor | Exits a `catchpad` funclet; calls inside the funclet usually carry a `"funclet"` operand bundle. |
| `cleanupret` | `cleanupret from <token> %cp unwind label %U` / `unwind to caller` | Optional unwind successor | Exits a `cleanuppad` funclet and continues WinEH unwinding. |
| `unreachable` | `unreachable` | No successors | Asserts the block cannot be reached; executing it is undefined behavior. |

EH pads and unwind edges are covered in more detail in
[`../16-exception-handling/README.md`](../16-exception-handling/README.md) and
[`../13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md).
When editing CFGs, keep `phi` predecessor lists consistent with terminator
successors; see [`../08-pitfalls/02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md).

## Arithmetic (integer)

| Op | Notes |
|---|---|
| `add`, `sub`, `mul` | Optional flags: `nsw`, `nuw` |
| `sdiv`, `udiv` | Optional flag: `exact` |
| `srem`, `urem` | Signed/unsigned remainder |

```llvm
%x = add nsw i32 %a, %b      ; no signed wrap
%y = mul nuw i64 %u, %v      ; no unsigned wrap
%z = sdiv exact i32 %p, 4    ; exact divide (no remainder)
```

## Arithmetic (floating-point)

| Op | Notes |
|---|---|
| `fadd`, `fsub`, `fmul`, `fdiv` | Standard |
| `frem` | Remainder |
| `fneg` | Unary negation |

Fast-math flags (combinable, in any order):

| Flag | Meaning |
|---|---|
| `nnan` | Assume no NaN |
| `ninf` | Assume no Inf |
| `nsz` | No signed zero |
| `arcp` | Allow reciprocal substitution |
| `contract` | Allow FMA contraction |
| `afn` | Approximate functions OK |
| `reassoc` | Allow reassociation |
| `fast` | All of the above |

```llvm
%y = fadd fast float %x, 1.0
%z = fdiv ninf nsz double %a, %b
```

For strict exception/rounding behavior, use the constrained FP intrinsics in
[`intrinsics.md#constrained-floating-point-intrinsics`](intrinsics.md#constrained-floating-point-intrinsics).

## Bitwise

| Op | Notes |
|---|---|
| `and`, `or`, `xor` | Logical |
| `shl` | Logical/arithmetic shift left; flags `nsw`, `nuw` |
| `lshr` | Logical shift right; flag `exact` |
| `ashr` | Arithmetic shift right; flag `exact` |

## Vector instructions

LLVM IR vectors may be fixed-width (`<4 x i32>`) or scalable (`<vscale x 4 x
i32>`). The three vector-specific instructions manipulate lanes; arithmetic,
bitwise, comparison, conversion, `select`, `load`, and `store` also operate on
vector types. See [`../09-vectorization/README.md`](../09-vectorization/README.md)
for Loop Vectorizer and SLP examples.

| Op | Syntax | Notes |
|---|---|---|
| `extractelement` | `%e = extractelement <N x T> %v, <idx-ty> %idx` | Reads one lane. `%idx` is an integer; out-of-range dynamic indices produce poison. |
| `insertelement` | `%r = insertelement <N x T> %v, T %e, <idx-ty> %idx` | Returns a new vector with one lane replaced; the original vector is unchanged. |
| `shufflevector` | `%r = shufflevector <N x T> %a, <M x T> %b, <K x i32> <mask>` | Builds a vector by selecting lanes from two input vectors. Mask elements may be poison/undef-like don't-care lanes. |

Common vectorization patterns:

| Pattern | Typical IR shape | Notes / links |
|---|---|---|
| Widened arithmetic | `load <VF x T>`, vector `add`/`fadd`, `store <VF x T>` | Loop vectorizer output for unit-stride loops; see [`../09-vectorization/examples/sum-loop-after-loop-vectorize.ll`](../09-vectorization/examples/sum-loop-after-loop-vectorize.ll). |
| Scalar pack / SLP | Several scalar ops become `<N x T>` ops plus `insertelement`/`extractelement` | Common in straight-line code; see [`../09-vectorization/examples/slp-scalars-after-slp.ll`](../09-vectorization/examples/slp-scalars-after-slp.ll). |
| Lane permutation | `shufflevector` splats, blends, concatenates, reverses, or interleaves lanes | Optimizers recognize many shuffle masks and map them to target permutes. |
| Reduction | Vector loop accumulates lanes, then calls `llvm.vector.reduce.*` | Intrinsics are summarized in [`intrinsics.md#vector-reduction-intrinsics`](intrinsics.md#vector-reduction-intrinsics). |
| Masked or predicated memory | Target/legalization may form masked load/store/gather/scatter intrinsics | Often appears when vectorizing conditionals or non-unit-stride memory. |

Vectorization is often blocked by missing aliasing, alignment, or dependency
facts; see [`../08-pitfalls/12-vectorization-blocked-by-aliasing.md`](../08-pitfalls/12-vectorization-blocked-by-aliasing.md).

## Aggregate instructions

Aggregates (`struct` and `array` values) are SSA values just like scalars, but
only `extractvalue` and `insertvalue` index into them directly. Do not confuse
these with `getelementptr`, which computes an address into memory.

| Op | Syntax | Notes |
|---|---|---|
| `extractvalue` | `%field = extractvalue { i32, float } %a, 1` | Indices are unsigned integer constants baked into the instruction. Reads from an aggregate SSA value. |
| `extractvalue` | `%nested = extractvalue { [4 x i32], i8 } %a, 0, 2` | Multiple indices walk nested arrays/structs. |
| `insertvalue` | `%b = insertvalue { i32, float } %a, float %v, 1` | Returns a new aggregate value with one field replaced. |
| `insertvalue` | `%c = insertvalue { [4 x i32], i8 } %a, i32 %v, 0, 2` | Nested update; original aggregate remains unchanged. |

Aggregate operations frequently appear with intrinsics that return structs, such
as `llvm.*.with.overflow.*`; see [`intrinsics.md#overflow-checked-arithmetic`](intrinsics.md#overflow-checked-arithmetic).
If a frontend's aggregate layout model drifts from the IR type, see
[`../08-pitfalls/05-type-schema-drift.md`](../08-pitfalls/05-type-schema-drift.md).

## Memory

| Op | Syntax (sketch) | Notes |
|---|---|---|
| `alloca` | `alloca <ty> [, <n> elements] [, align N]` | Stack alloc |
| `load` | `load <ty>, ptr %p [, align N]` | Read |
| `store` | `store <ty> %v, ptr %p [, align N]` | Write |
| `fence` | `fence [syncscope("...")] <ordering>` | Memory fence; see [`../11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md) |
| `cmpxchg` | `cmpxchg ptr %p, <ty> %cmp, <ty> %new <succ-ord> <fail-ord> [, align N]` | Compare-and-swap; returns `{ <ty>, i1 }`; see [`../11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md) |
| `atomicrmw` | `atomicrmw <op> ptr %p, <ty> %v <ord> [, align N]` | Atomic RMW; returns old value; see [`../11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md) |
| `getelementptr` | `getelementptr [inbounds] <ty>, ptr %p, <ty> %idx, ...` | Address compute; address spaces must match pointer types |

Memory intrinsics such as `llvm.memcpy`, `llvm.memmove`, `llvm.memset`,
`llvm.lifetime.*`, and `llvm.prefetch` are summarized in
[`intrinsics.md#memory-intrinsics`](intrinsics.md#memory-intrinsics).
For address-space mistakes, see
[`../08-pitfalls/11-address-space-confusion.md`](../08-pitfalls/11-address-space-confusion.md).

### Atomic instruction forms and ordering constraints

Atomic IR uses the same memory instructions plus ordering keywords. Use
[`../11-concurrency/01-atomic-orderings.md`](../11-concurrency/01-atomic-orderings.md)
for the meaning of each ordering, and
[`../11-concurrency/03-volatile-vs-atomic.md`](../11-concurrency/03-volatile-vs-atomic.md)
for the distinction between visibility and synchronization.

| Form | Syntax | Valid orderings | Notes |
|---|---|---|---|
| Atomic load | `%v = load atomic <ty>, ptr %p <ord>, align N` | `unordered`, `monotonic`, `acquire`, `seq_cst` | `release` and `acq_rel` are invalid because a load cannot release prior writes. Alignment is required for atomic loads. |
| Atomic store | `store atomic <ty> %v, ptr %p <ord>, align N` | `unordered`, `monotonic`, `release`, `seq_cst` | `acquire` and `acq_rel` are invalid because a store cannot acquire later reads. Alignment is required for atomic stores. |
| Atomic RMW | `%old = atomicrmw <op> ptr %p, <ty> %v <ord>, align N` | `monotonic`, `acquire`, `release`, `acq_rel`, `seq_cst` | Read-modify-write; returns the old value. |
| Compare exchange | `%pair = cmpxchg ptr %p, <ty> %cmp, <ty> %new <succ> <fail>, align N` | Success: `monotonic`, `acquire`, `release`, `acq_rel`, `seq_cst`; failure: `monotonic`, `acquire`, `seq_cst` and not stronger than success | Returns `{ old_value, success_i1 }`. `weak` may fail spuriously; `volatile` may be combined only for volatile semantics. |
| Fence | `fence [syncscope("...")] <ord>` | `acquire`, `release`, `acq_rel`, `seq_cst` | Orders other memory operations rather than accessing memory itself. |

`atomicrmw` operations: `xchg`, `add`, `sub`, `and`, `nand`, `or`,
`xor`, `max`, `min`, `umax`, `umin`, `fadd`, `fsub`, `fmax`, `fmin`.
Some floating-point forms depend on target support.

Atomic orderings (weakest → strongest): not atomic, `unordered`,
`monotonic`, `acquire`/`release`, `acq_rel`, `seq_cst`. Common pitfalls:
[`../08-pitfalls/09-atomic-ordering-mismatch.md`](../08-pitfalls/09-atomic-ordering-mismatch.md)
and [`../08-pitfalls/10-volatile-is-not-atomic.md`](../08-pitfalls/10-volatile-is-not-atomic.md).

## Comparison

| Op | Syntax | Result |
|---|---|---|
| `icmp` | `icmp <pred> <ty> %a, %b` | `i1` or vector of `i1` for vector operands |
| `fcmp` | `fcmp <pred> <ty> %a, %b` | `i1` or vector of `i1` for vector operands |

Integer predicates: `eq`, `ne`, `slt`, `sle`, `sgt`, `sge`, `ult`,
`ule`, `ugt`, `uge`.

FP predicates: `oeq`, `one`, `olt`, `ole`, `ogt`, `oge`, `ueq`,
`une`, `ult`, `ule`, `ugt`, `uge`, `ord`, `uno`, `true`, `false`.

(`o` = ordered: no NaN. `u` = unordered: NaN-permissive.)

## Conversion instructions

Conversions are explicit. Most work element-wise on vectors when source and
target have the same lane count. Pointer/address-space conversions are target
sensitive; see [`../08-pitfalls/11-address-space-confusion.md`](../08-pitfalls/11-address-space-confusion.md).

| Op | Source → Target | Syntax sketch | Notes |
|---|---|---|---|
| `trunc` | wider int → narrower int | `trunc i64 %x to i32` | Drops high bits. |
| `zext` | narrower int → wider int | `zext i8 %x to i32` | Zero-extend. |
| `sext` | narrower int → wider int | `sext i8 %x to i32` | Sign-extend. |
| `fptrunc` | wider float → narrower float | `fptrunc double %x to float` | May round; use constrained FP intrinsics if rounding/exception behavior must be explicit. |
| `fpext` | narrower float → wider float | `fpext float %x to double` | Extends FP precision. |
| `fptoui` | float → unsigned int | `fptoui float %x to i32` | Out-of-range or NaN input is poison unless using a saturating intrinsic. |
| `fptosi` | float → signed int | `fptosi double %x to i64` | Out-of-range or NaN input is poison unless using a saturating intrinsic. |
| `uitofp` | unsigned int → float | `uitofp i32 %x to double` | May round if target FP type cannot represent all values. |
| `sitofp` | signed int → float | `sitofp i32 %x to float` | May round. |
| `ptrtoint` | pointer → integer | `ptrtoint ptr %p to i64` | Exposes an implementation-defined pointer representation; integer width should match target assumptions. |
| `inttoptr` | integer → pointer | `inttoptr i64 %x to ptr` | Produces a pointer value from bits; provenance/target semantics matter. |
| `bitcast` | same-size non-aggregate values or pointer casts within same address space | `bitcast <4 x i32> %v to <2 x i64>` | Reinterprets bits; cannot change address spaces. |
| `addrspacecast` | pointer in one address space → pointer in another | `addrspacecast ptr addrspace(1) %p to ptr` | Only valid when the target defines the conversion. |

## Other and special constructs

| Op | Syntax | Notes |
|---|---|---|
| `phi` | `phi <ty> [ %v1, %P1 ], [ %v2, %P2 ], ...` | SSA merge at a block start; incoming blocks must match predecessors. See [`../08-pitfalls/02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md). |
| `select` | `select i1 %c, <ty> %t, <ty> %f` | Ternary expression. With vector conditions, selects lane-wise. |
| `freeze` | `%x = freeze <ty> %v` | Converts `undef`/poison into one arbitrary but fixed value for this execution, preventing later UB from propagating through uses. Useful before control-flow decisions derived from possibly poison values. |
| `call` | `%r = call <ty> @f(<args>) [ "deopt"(...) ]` / `%r = call <ty> asm "...", "..."(...)` | Function, intrinsic, or inline-asm call. Normal calls do not have unwind successors; use `invoke` if unwinding is represented. Operand bundles are call-site semantic payloads and must be preserved when rewriting calls; see [`../01-syntax/04-inline-asm.md`](../01-syntax/04-inline-asm.md), [`../13-advanced-ir/07-operand-bundles.md`](../13-advanced-ir/07-operand-bundles.md), and [`../16-exception-handling/03-wineh-funclets.md#operand-bundle-interaction-funclet`](../16-exception-handling/03-wineh-funclets.md#operand-bundle-interaction-funclet). |
| `va_arg` | `%v = va_arg ptr %ap, <ty>` | Variadic argument fetch. ABI details are target-shaped; preserve `llvm.va_start`/`llvm.va_end`/`va_arg` sequences unless intentionally modeling the selected target ABI. See [`../12-backend-jit/01-codegen-pipeline.md#varargs-and-abi-variance`](../12-backend-jit/01-codegen-pipeline.md#varargs-and-abi-variance). |
| `landingpad` | `%lp = landingpad <result-ty> cleanup catch <ty> <val> filter <array-ty> <val>` | Itanium-style pad. It must be the first non-`phi` instruction in an `invoke` unwind destination, and there is at most one per landing pad block. Clauses describe catches/filters/cleanup; see [`../16-exception-handling/02-itanium-landingpad.md`](../16-exception-handling/02-itanium-landingpad.md). |
| `catchpad` | `%cp = catchpad within <token> %cs [<args>]` | Begins a catch funclet and returns a `token`; normally the first non-`phi` instruction in a handler block targeted by `catchswitch`. |
| `cleanuppad` | `%cp = cleanuppad within <parent> [<args>]` | Begins a cleanup funclet and returns a `token`; exited by `cleanupret`; see [`../16-exception-handling/04-cleanups-and-resume.md`](../16-exception-handling/04-cleanups-and-resume.md). |

For token-typed EH pads, funclets, and special types, see
[`../16-exception-handling/README.md`](../16-exception-handling/README.md) and
[`../13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md). Operand bundles on `call` and
`invoke`, including `"funclet"`, are covered in
[`../13-advanced-ir/07-operand-bundles.md`](../13-advanced-ir/07-operand-bundles.md).

## Constant expressions

A subset of operations can appear as **constant expressions** in
operand position (e.g., as global initializers). The form is the
operation parenthesized:

```llvm
@p   = global ptr getelementptr inbounds (i32, ptr @arr, i32 5)
@neg = global i32 sub (i32 0, i32 42)
@i   = global i64 ptrtoint (ptr @foo to i64)
@bc  = global ptr bitcast (ptr @foo to ptr)
```

Operations available as constant expressions: `add`, `sub`, `mul`,
`shl`, `lshr`, `ashr`, `and`, `or`, `xor`, all conversions
(`trunc`, `zext`, ..., `bitcast`, `addrspacecast`, `ptrtoint`,
`inttoptr`), `getelementptr`, `icmp`, `fcmp`, `select`,
`extractelement`, `insertelement`, `shufflevector`, `extractvalue`,
`insertvalue`, `blockaddress`.

**Constant expressions require all operands to be constants**
(literals, globals, or other constant expressions). They cannot
take SSA values. See [`../08-pitfalls/01-nested-instruction-expressions.md`](../08-pitfalls/01-nested-instruction-expressions.md).

## See also

- [`../01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md) — general shape
- [`../01-syntax/04-inline-asm.md`](../01-syntax/04-inline-asm.md) — inline asm expressions and `callbr` labels
- [`../09-vectorization/README.md`](../09-vectorization/README.md) — vectorization overview and examples
- [`../11-concurrency/`](../11-concurrency/) — atomic orderings, atomic instructions, and volatile-vs-atomic
- [`../13-advanced-ir/`](../13-advanced-ir/) — intrinsics, target intrinsics, special types, and tokens
- [`../08-pitfalls/README.md`](../08-pitfalls/README.md) — common mistakes
- [`../10-grammar/llvm-ir.tm`](../10-grammar/llvm-ir.tm) — exact syntax productions
- LLVM LangRef: https://llvm.org/docs/LangRef.html#instruction-reference
