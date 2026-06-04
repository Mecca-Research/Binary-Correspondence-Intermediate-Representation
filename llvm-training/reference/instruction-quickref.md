# Instruction Quick Reference

Compact table of every LLVM IR instruction, grouped by category. For
syntax details, see [`../01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md) and the
relevant chapter file.

## Terminators (must be last in basic block)

| Op | Syntax | Notes |
|---|---|---|
| `ret` | `ret void` / `ret <ty> <val>` | Return from function |
| `br` | `br label %T` | Unconditional |
| `br` | `br i1 %c, label %T, label %F` | Conditional |
| `switch` | `switch <ty> %v, label %D [ <ty> <c>, label %L ... ]` | Multi-way |
| `indirectbr` | `indirectbr ptr %addr, [ label %A, label %B ]` | Runtime target |
| `invoke` | `invoke <ty> @f(...) to label %N unwind label %U` | Call w/ EH |
| `callbr` | `callbr <ty> @f(...) to label %N [ label %L ... ]` | Call w/ jumps |
| `resume` | `resume <ty> %v` | Resume unwind |
| `catchswitch` | `catchswitch within X [ label %H ] unwind to caller` | EH |
| `catchret` | `catchret from %cp to label %T` | EH |
| `cleanupret` | `cleanupret from %cp unwind label %U` | EH |
| `unreachable` | `unreachable` | UB if reached |

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

## Bitwise

| Op | Notes |
|---|---|
| `and`, `or`, `xor` | Logical |
| `shl` | Logical/arithmetic shift left; flags `nsw`, `nuw` |
| `lshr` | Logical shift right; flag `exact` |
| `ashr` | Arithmetic shift right; flag `exact` |

## Vector

| Op | Syntax | Notes |
|---|---|---|
| `extractelement` | `extractelement <ty> %v, <ty> %idx` | Single lane out |
| `insertelement` | `insertelement <ty> %v, <ty> %e, <ty> %idx` | Lane replace |
| `shufflevector` | `shufflevector <ty> %a, <ty> %b, <ty> %mask` | Per-lane select |

## Aggregate

| Op | Syntax | Notes |
|---|---|---|
| `extractvalue` | `extractvalue <ty> %a, <idx>, [<idx> ...]` | Indices are constants |
| `insertvalue` | `insertvalue <ty> %a, <ty> %v, <idx>, [...]` | Indices are constants |

## Memory

| Op | Syntax (sketch) | Notes |
|---|---|---|
| `alloca` | `alloca <ty> [, <n> elements] [, align N]` | Stack alloc |
| `load` | `load <ty>, ptr %p [, align N]` | Read |
| `load atomic` | `load atomic <ty>, ptr %p <ord> [, align N]` | Atomic read |
| `store` | `store <ty> %v, ptr %p [, align N]` | Write |
| `store atomic` | `store atomic <ty> %v, ptr %p <ord> [, align N]` | Atomic write |
| `fence` | `fence [syncscope("...")] <ordering>` | Memory fence; see [`../11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md) |
| `cmpxchg` | `cmpxchg ptr %p, <ty> %cmp, <ty> %new <succ-ord> <fail-ord> [, align N]` | Compare-and-swap; returns `{ <ty>, i1 }`; see [`../11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md) |
| `atomicrmw` | `atomicrmw <op> ptr %p, <ty> %v <ord> [, align N]` | Atomic RMW; returns old value; see [`../11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md) |
| `getelementptr` | `getelementptr [inbounds] <ty>, ptr %p, <ty> %idx, ...` | Address compute |

`atomicrmw` operations: `xchg`, `add`, `sub`, `and`, `nand`, `or`,
`xor`, `max`, `min`, `umax`, `umin`, `fadd`, `fsub`, `fmax`, `fmin`.

Atomic orderings (weakest → strongest): not atomic, `unordered`,
`monotonic`, `acquire`, `release`, `acq_rel`, `seq_cst`. For choosing
orderings and avoiding volatile/atomic confusion, see
[`../11-concurrency/01-atomic-orderings.md`](../11-concurrency/01-atomic-orderings.md)
and [`../11-concurrency/03-volatile-vs-atomic.md`](../11-concurrency/03-volatile-vs-atomic.md).

## Comparison

| Op | Syntax | Result |
|---|---|---|
| `icmp` | `icmp <pred> <ty> %a, %b` | `i1` |
| `fcmp` | `fcmp <pred> <ty> %a, %b` | `i1` |

Integer predicates: `eq`, `ne`, `slt`, `sle`, `sgt`, `sge`, `ult`,
`ule`, `ugt`, `uge`.

FP predicates: `oeq`, `one`, `olt`, `ole`, `ogt`, `oge`, `ueq`,
`une`, `ult`, `ule`, `ugt`, `uge`, `ord`, `uno`, `true`, `false`.

(`o` = ordered: no NaN. `u` = unordered: NaN-permissive.)

## Conversion

| Op | Source → Target | Notes |
|---|---|---|
| `trunc` | wider int → narrower int | Drop high bits |
| `zext` | narrower int → wider int | Zero-extend |
| `sext` | narrower int → wider int | Sign-extend |
| `fptrunc` | wider float → narrower float | |
| `fpext` | narrower float → wider float | |
| `fptoui` | float → unsigned int | |
| `fptosi` | float → signed int | |
| `uitofp` | unsigned int → float | |
| `sitofp` | signed int → float | |
| `ptrtoint` | ptr → int | |
| `inttoptr` | int → ptr | |
| `bitcast` | T1 → T2, same width | Reinterpret bits |
| `addrspacecast` | `ptr addrspace(A)` → `ptr addrspace(B)` | |

## Other

| Op | Syntax | Notes |
|---|---|---|
| `phi` | `phi <ty> [ %v1, %P1 ], [ %v2, %P2 ], ...` | SSA merge |
| `select` | `select i1 %c, <ty> %t, <ty> %f` | Ternary expression |
| `freeze` | `freeze <ty> %v` | Convert poison/undef to arbitrary fixed value |
| `call` | `call <ty> @f(<args>)` | Function call |
| `va_arg` | `va_arg <ty>* %ap, <ty>` | Variadic arg fetch |
| `landingpad` | `landingpad <ty> [cleanup] [<clause>...]` | Exception landing |
| `catchpad` | `catchpad within %cs [<args>]` | EH (Windows / C++) |
| `cleanuppad` | `cleanuppad within X [<args>]` | EH cleanup |

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
- `08-pitfalls/` — common mistakes
- `10-grammar/llvm-ir.tm` — exact syntax productions
