# Instruction Format and Operand Types

## TL;DR

Most LLVM IR instructions take the shape:

```
%result = <op> <type>, <operand>, <operand>, ...
```

Result-producing instructions assign to an SSA name on the left.
Effect-only instructions (`store`, `br`, `ret`, `fence`) have no LHS.
Each operand has a type, which is usually written *once* at the start
of the instruction and then implied for subsequent operands of the
same type — but read the grammar (`10-grammar/llvm-ir.tm`) for the
exact rules per instruction.

## General shape

```
[<result> '='] <opcode> [<flags>] <type> <op1> [, <op2> [, <op3> ...]] [, <metadata>]
```

### Examples by category

```llvm
; Arithmetic — produces a value
%sum = add i32 %a, %b
%prod = mul nsw i32 %a, %b               ; with flag (no signed wrap)
%diff = fsub fast float %x, %y           ; with fast-math flag

; Memory
%v = load i32, ptr %p, align 4           ; load
store i32 %v, ptr %p, align 4            ; store (no result)

; Control flow
br label %next                            ; unconditional, no result
br i1 %cond, label %t, label %f           ; conditional, no result
ret i32 %v                                ; return, no result

; Call
%r = call i32 @some_fn(i32 %a, ptr %p)

; Comparison
%cmp = icmp slt i32 %a, %b                ; produces i1
%fcmp = fcmp oeq float %x, %y             ; produces i1

; Cast
%w = zext i32 %v to i64

; Aggregate
%y = extractvalue { i32, i32 } %pair, 0
%z = insertvalue { i32, i32 } %pair, i32 7, 1
```

## Operands

Operands fall into a small set of categories:

| Category | Example | Notes |
|---|---|---|
| Constant (integer) | `42`, `-1`, `0xff` | Type carried from the instruction |
| Constant (float) | `3.14`, `0x4000000000000000` | Hex form for bit-exact floats |
| Constant (null) | `null` | Pointer null |
| Constant (undef) | `undef` | Any-value-allowed |
| Constant (poison) | `poison` | UB if observed (preferred over undef) |
| Constant (aggregate) | `{ i32 1, i32 2 }`, `[2 x i32] [i32 1, i32 2]` | Inline |
| Constant (string) | `c"hi\00"` | Char array literal |
| Local SSA value | `%result`, `%42` | Defined elsewhere in same function |
| Global | `@foo` | Module-scope |
| Label | `label %target` | Used by branches |
| Metadata | `!42`, `!{...}` | Tagged with `!dbg` etc. |

## Where types appear

LLVM IR is **strongly typed**. Every value has a type; the compiler
will not "infer" one. Conventions for where types go:

- **Once per instruction in most cases.** For `add i32 %a, %b`, the
  `i32` covers both operands and the result.
- **Once per operand when operands have different types.** For `bitcast
  i32 %v to i64`, the source and target types are both written.
- **Composite operands declare their full type** inline. For
  `extractvalue { i32, float } %x, 0`, the aggregate type and the
  index are explicit.
- **Constants that are inline carry the type of their context.**
  `add i32 5, 10` — the `5` and `10` are i32 because the instruction
  says `i32`.

When in doubt, write the type. Verbose IR assembles; under-typed IR
may not parse.

## Result naming

```llvm
%foo = add i32 %a, %b      ; explicit name
%42  = add i32 %a, %b      ; numeric name (auto-assigned by the assembler if you skip)
```

Numeric names must be sequential within a function (`%0`, `%1`, `%2`,
...). If you mix explicit names and numeric, the numeric counter
starts at the first unused integer. Mostly you'll only see numeric
names in machine-generated IR.

## Instruction families

Grouped by what they do (links into the quick reference):

### Arithmetic (integer)

`add`, `sub`, `mul`, `udiv`, `sdiv`, `urem`, `srem`
- Flags: `nsw` (no signed wrap), `nuw` (no unsigned wrap), `exact` (for
  divides where remainder is zero).

### Arithmetic (floating-point)

`fadd`, `fsub`, `fmul`, `fdiv`, `frem`, `fneg`
- Flags (any subset, in any order): `nnan`, `ninf`, `nsz`, `arcp`,
  `contract`, `afn`, `reassoc`, `fast` (= all of the above).

### Bitwise

`and`, `or`, `xor`, `shl`, `lshr`, `ashr`

### Memory

`alloca`, `load`, `store`, `getelementptr`, `cmpxchg`, `atomicrmw`,
`fence`

### Comparison

`icmp <pred>` — integer comparison, returns `i1`
`fcmp <pred>` — floating-point comparison, returns `i1`

Integer predicates: `eq`, `ne`, `slt`, `sle`, `sgt`, `sge`,
`ult`, `ule`, `ugt`, `uge`.
Float predicates: `oeq`, `one`, `olt`, `ole`, `ogt`, `oge`, `ueq`,
`une`, `ult`, `ule`, `ugt`, `uge`, `ord`, `uno`, `true`, `false`.

(`o` = ordered: neither operand is NaN. `u` = unordered: NaN-permissive.)

### Conversion

`trunc`, `zext`, `sext`, `fptrunc`, `fpext`, `fptoui`, `fptosi`,
`uitofp`, `sitofp`, `bitcast`, `addrspacecast`, `inttoptr`,
`ptrtoint`

### Vector

`extractelement`, `insertelement`, `shufflevector`

### Aggregate

`extractvalue`, `insertvalue`

### Control flow (terminators)

`ret`, `br`, `switch`, `indirectbr`, `invoke`, `callbr`, `resume`,
`catchswitch`, `catchret`, `cleanupret`, `unreachable`

### Other

`call`, `phi`, `select`, `freeze`, `va_arg`, `landingpad`, `catchpad`,
`cleanuppad`

## Instruction-level metadata

Attach `!metadata` after the operands:

```llvm
%v = load i32, ptr %p, align 4, !tbaa !0, !alias.scope !1
```

Where `!0` and `!1` reference metadata definitions elsewhere in the
module. Common attachments:

- `!dbg !N` — debug location
- `!tbaa !N` — type-based alias analysis
- `!alias.scope`, `!noalias` — noalias scoping
- `!nontemporal`, `!nonnull`, `!range`, `!align` — value/access hints
- `!llvm.loop` — attached to loop latch terminators

See `01-syntax/03-comments-metadata.md`.

## Pitfalls

- **Trying to nest instruction-producing forms.** This is not legal:
  ```llvm
  %x = or i1 (xor i1 %y, true), %z   ; WRONG
  ```
  Each instruction must produce a named result; expressions don't
  nest. Split:
  ```llvm
  %not_y = xor i1 %y, true
  %x     = or i1 %not_y, %z          ; correct
  ```
  See `08-pitfalls/01-nested-instruction-expressions.md`.

  *(Constant expressions can be parenthesized in operand position;
  SSA value instructions cannot. The look-alike confuses many
  generators.)*

- **Forgetting type on a constant.** `add 1, 2` won't parse. You need
  `add i32 1, 2`.

- **Mismatched operand types.** `add i32 %a, i64 %b` is rejected. Cast
  one first.

- **Using a name before it's defined** (within a single basic block).
  Order matters — instructions execute top-to-bottom.

- **Forgetting `nsw`/`nuw` semantics.** `add nsw` produces *poison* if
  the addition overflows in signed interpretation; subsequent uses of
  that poison value yield UB. Only attach when the property genuinely
  holds.

## See also

- `00-foundations/02-ssa.md` — result names are SSA
- `03-comments-metadata.md` — `;` and `!N` syntax
- `reference/instruction-quickref.md` — full table
- `08-pitfalls/01-nested-instruction-expressions.md` — the most common
  generator mistake
- `10-grammar/llvm-ir.tm` — exact grammar productions
