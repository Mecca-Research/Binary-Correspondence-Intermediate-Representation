# Inline Assembler Expressions

## TL;DR

- LLVM IR inline asm is a **callee value** used by `call`, `invoke`, or
  `callbr`, not a normal instruction named `asm`.
- The core shape is:

  ```llvm
  %r = call i32 asm "movl $1, $0", "=r,r"(i32 %x)
  ```

  The first string is the target assembly template. The second string is the
  comma-separated constraint list that tells LLVM how operands are allocated.
- Constraint order matters: **outputs first, inputs second, clobbers last**.
- Inline asm is target-specific. Syntax that assembles and verifies can still
  fail to lower through `llc` on a different target, with a different backend,
  or with a default triple that does not understand the asm text or constraints.
- Use `callbr` for asm-goto-style inline asm that can branch to labels.

## Inline asm expression syntax

Inline asm appears where a function pointer/callee normally appears:

```llvm
%out = call i32 asm "bswap $0", "=r,0"(i32 %in)
```

Read that as:

| Part | Meaning |
|---|---|
| `call i32` | The call returns `i32`. Use `call void` for no result. |
| `asm` | The callee is an inline assembler expression. |
| `sideeffect` | Optional marker saying the asm has effects not fully modeled by operands. |
| `alignstack` | Optional marker asking the backend to preserve conservative stack alignment. |
| `inteldialect` | Optional target-specific x86 printing/parsing dialect marker. |
| `"bswap $0"` | Assembly template string. `$0`, `$1`, ... refer to constraint operands. |
| `"=r,0"` | Constraint string. Each comma-separated item describes one output/input/clobber. |
| `(i32 %in)` | Ordinary call arguments consumed by input constraints. |

Keyword order is conventional and parser-sensitive: when several are present,
write `sideeffect` before `alignstack`, and dialect markers last.

The assembly template is passed to the target backend. `$N` expands to the
register, memory operand, or label chosen for the `N`th constraint item.
`${N:modifier}` asks the target printer to use a target-specific operand
modifier, and `$$` emits a literal dollar sign.

Inline asm with a visible SSA result must use output constraints. Multiple
outputs are returned as an aggregate value, commonly a struct, which you can
split with `extractvalue`.

## Constraint strings at a high level

A constraint list is a comma-separated string. The positions in that string are
numbered left-to-right and match template placeholders `$0`, `$1`, ... . The
items must be grouped as outputs, then inputs, then clobbers.

### Register and memory operands: `r` and `m`

- `r` asks LLVM to allocate a target-appropriate register class. The exact
  register class is target-specific.
- `{rax}`, `{eax}`, `{x0}`, and similar forms name explicit physical registers
  when the target supports them.
- `m` asks for a memory operand/addressing mode. Some frontends and targets use
  more precise memory forms such as `*m` for an indirect memory operand.
- Immediate constraints also exist, but their letters and legal ranges are
  target-specific.

```llvm
; x86-64-flavored example: output in a register, input in a register.
%y = call i32 asm "bswap $0", "=r,0"(i32 %x)

; x86-64-flavored example: input is a memory operand pointing at an i32.
%v = call i32 asm "movl $1, $0", "=r,*m"(ptr elementtype(i32) %p)
```

### Output constraints: `=`

An output starts with `=`:

```llvm
%out = call i32 asm "bswap $0", "=r,0"(i32 %in)
```

`=r` means the asm writes a register and LLVM returns that register's final
value as the call result. Outputs do **not** consume call arguments unless they
are special indirect outputs; the following `0` input consumes `%in` and ties it
to the output register.

### Early-clobber outputs: `=&r`

Normally, LLVM may reuse the same physical register for an output and an input
when it proves that is legal for the constraint set. If the asm writes an output
before all inputs have been read, add `&` to the output:

```llvm
%sum = call i32 asm "addl $2, $0", "=&r,0,r,~{cc}"(i32 %x, i32 %y)
```

`=&r` means the output register is clobbered early. LLVM should not assign that
same register to unrelated inputs. The `0` input is a tied operand, described
next, and is allowed to share the output by construction.

### Tied operands: `0`, `1`, ...

An input constraint can be a decimal number instead of a letter. That ties the
input to a previous register output constraint:

```llvm
%sum = call i32 asm "addl $2, $0", "=&r,0,r,~{cc}"(i32 %x, i32 %y)
```

Here:

- constraint `0` is output `=&r`, printed as `$0`;
- constraint `1` is input `0`, tied to output constraint 0 and consuming `%x`;
- constraint `2` is input `r`, consuming `%y` and printed as `$2`;
- `addl $2, $0` adds the second input into the tied input/output register.

Ties are the IR way to model read-modify-write register operands. They should
only tie to register outputs, and only one input may tie to a given output.

### Clobbers: `~`

A clobber starts with `~`. It tells LLVM that the asm overwrites a resource but
does not produce an SSA value or consume a call argument.

Common forms include:

| Clobber | Meaning |
|---|---|
| `~{cc}` / `~{flags}` | Condition-code or flags register, target-dependent spelling. |
| `~{eax}` / `~{rax}` | A named physical register on targets that define it. |
| `~{memory}` | The asm may read/write arbitrary memory not otherwise modeled. |

Use `sideeffect` when the asm has externally visible behavior beyond its SSA
outputs, such as device I/O, fences, or a deliberately empty compiler barrier:

```llvm
call void asm sideeffect "", "~{memory}"()
```

Do not list the same named register as both an output and a clobber.

## `callbr` and asm-goto-style labels

`callbr` is a terminator. It behaves like a call with an explicit fallthrough
successor plus a list of indirect label destinations:

```llvm
callbr void asm sideeffect "testl $0, $0; jne ${1:l}", "r,!i,~{cc}"(i32 %x)
        to label %fallthrough [label %taken]
```

Important pieces:

- The `to label %fallthrough` edge is where execution continues if the asm does
  not branch away.
- The bracketed labels are possible indirect destinations. They are part of the
  CFG, so `phi` nodes in those blocks must include the `callbr` block as a
  predecessor.
- A label constraint uses the `!` prefix, usually `!i`. Label constraints do not
  consume call arguments; they consume labels from the bracketed indirect
  destination list.
- The number of label constraints must match the number of indirect labels.
- A `callbr` result, if any, is available in the fallthrough block and the
  indirect destination blocks.

`callbr` is most commonly produced from C/C++ `asm goto`. It should be treated
as a control-flow terminator during CFG rewrites, not as an ordinary non-terminator
`call`.

## Portability warning

Inline asm is intentionally a backend escape hatch. The IR syntax is shared, but
these pieces are target-specific:

- assembly mnemonics and operand order;
- register names and register classes;
- memory and immediate constraint letters;
- template operand modifiers such as `${1:l}`;
- whether a backend supports lowering a particular constraint combination.

For training examples, prefer `llvm-as` and `opt -passes=verify` as syntax and
well-formedness checks. Do not assume a standalone inline-asm example belongs in
a portable `llc` smoke allowlist unless it has been tested on the intended
target triple.

## Example file

See [`examples/inline-asm.ll`](examples/inline-asm.ll) for a compact module with
register outputs, memory inputs, tied operands, clobbers, a compiler barrier,
and a `callbr` asm-goto sketch.

## See also

- [`02-instruction-format.md`](02-instruction-format.md) — where `call` and
  terminators fit in instruction syntax
- [`../reference/instruction-quickref.md`](../reference/instruction-quickref.md) —
  quick lookup for `call` and `callbr`
- LLVM LangRef: <https://llvm.org/docs/LangRef.html#inline-assembler-expressions>
- LLVM LangRef: <https://llvm.org/docs/LangRef.html#callbr-instruction>
