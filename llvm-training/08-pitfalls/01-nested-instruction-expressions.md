# Pitfall 01 — Nested Instruction-as-Expression

## BCIR instance

| Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|
| `runtime/llvm/bcir_claim_verify.ll` | `1f62e86` | `llvm-as runtime/llvm/bcir_claim_verify.ll -o /dev/null` | Split nested boolean expressions into named SSA temporaries. | [`01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md); [`03-constants/04-global-vs-local.md`](../03-constants/04-global-vs-local.md); [`10-grammar/llvm-ir.tm`](../10-grammar/llvm-ir.tm) |

## The error

```
llvm-as: claim_verify.ll:71:33: error: expected value token
  %ok = and i1 %atomic_lane_ok, (xor i1 %is_a_lane_non_atomic, true)
                                ^
```

Or for the `or i1 (or i1 ...)` flavor:

```
llvm-as: claim_verify.ll:103:35: error: expected value token
  %needs_bounds = or i1 %is_load, (or i1 %is_store, %is_atomic)
                                  ^
```

## What's happening

LLVM IR's syntax has **constant expressions** and **value
instructions**. They look superficially similar, but they're
**not interchangeable**:

```llvm
; Constant expression — OK at module scope where a constant is required
@p = global ptr getelementptr (i32, ptr @arr, i32 5)

; Instruction — produces a named SSA value
%v = add i32 %a, %b
```

You cannot use an **instruction** as a sub-expression of another
instruction. Each instruction must produce exactly one named SSA
value, and operands must already be SSA values or constants — not
nested instructions.

This is wrong:

```llvm
%ok = and i1 %x, (xor i1 %y, true)        ; ❌
%r  = or  i1 %a, (or i1 %b, %c)           ; ❌
```

This is right:

```llvm
%not_y = xor i1 %y, true
%ok    = and i1 %x, %not_y                ; ✓

%inner = or i1 %b, %c
%r     = or i1 %a, %inner                 ; ✓
```

## Why generators trip on this

Many IR generators emit IR by string concatenation:

```python
def emit_or(a, b):
    return f"or i1 {a}, {b}"

# Generator inadvertently passes an instruction string in operand position:
expr = emit_or(emit_xor(x, "true"), y)
# → "or i1 xor i1 %x, true, %y"   ← wrong shape
```

The fix is to emit each instruction *and bind it to a fresh SSA
name*, then reference the name. Stop nesting strings.

## The real BCIR instance

`runtime/llvm/bcir_claim_verify.ll` had:

```llvm
%ok = and i1 %atomic_lane_ok, (xor i1 %is_a_lane_non_atomic, true)
```

And:

```llvm
%load_ok = or i1 (xor i1 %is_load, true), %rd_ok
```

These were rejected by `llvm-as`, breaking every `validate_*.sh`
script downstream. Fixed in commit `1f62e86` ("Fix LLVM IR
validation blockers and schema drift") by splitting each into an
explicit SSA temp:

```llvm
%not_a_lane_non_atomic = xor i1 %is_a_lane_non_atomic, true
%ok                    = and i1 %atomic_lane_ok, %not_a_lane_non_atomic

%not_load = xor i1 %is_load, true
%load_ok  = or i1 %not_load, %rd_ok
```

## Exception: parenthesized constant expressions

You **can** use a parenthesized constant expression in operand
position, because constant expressions are evaluated at compile time
and produce a constant (not a named SSA value):

```llvm
@gep_init = global ptr getelementptr (i32, ptr @arr, i32 5)
@neg      = global i32 sub (i32 0, i32 42)
```

This is what looks superficially similar and trips people up. The
test is: **does the parenthesized form involve only constants and
globals?** If yes, it's a constant expression. If it involves any
`%name` SSA value, it has to be a separate instruction.

## See also

- [`../01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md) — instruction shape
- [`../03-constants/04-global-vs-local.md`](../03-constants/04-global-vs-local.md) — constant expressions
- `10-grammar/llvm-ir.tm` — grammar productions for `ConstantExpr` vs
  `ValueInstruction`
