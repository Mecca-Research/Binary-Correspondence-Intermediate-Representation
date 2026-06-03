# Curriculum — Reading Order

Three suggested paths depending on how much time you (or your agent
context) have.

## Path 1: 30-minute fast pass (just enough to not be dangerous)

Read these in order. Skip examples; just the prose.

1. [`00-foundations/01-what-is-llvm-ir.md`](00-foundations/01-what-is-llvm-ir.md) — what is IR, what isn't it
2. [`00-foundations/02-ssa.md`](00-foundations/02-ssa.md) — SSA + phi nodes (the single most
   important concept)
3. [`01-syntax/01-modules-functions-blocks.md`](01-syntax/01-modules-functions-blocks.md) — the hierarchy
4. [`01-syntax/02-instruction-format.md`](01-syntax/02-instruction-format.md) — `%result = op type, operands`
5. [`08-pitfalls/README.md`](08-pitfalls/README.md) — the index of what breaks

You now know enough to read existing IR. You can't write it safely yet.

Practice next: [`exercises/001-add.prompt.md`](exercises/001-add.prompt.md) and
[`exercises/002-if-else-phi.prompt.md`](exercises/002-if-else-phi.prompt.md).

## Path 2: 2-hour working knowledge

After Path 1, add:

6. [`02-types/01-primitive-types.md`](02-types/01-primitive-types.md)
7. [`02-types/02-composite-types.md`](02-types/02-composite-types.md)
8. [`02-types/03-opaque-and-pointer-types.md`](02-types/03-opaque-and-pointer-types.md)
9. [`04-memory/01-alloca.md`](04-memory/01-alloca.md)
10. [`04-memory/02-load-store.md`](04-memory/02-load-store.md)
11. [`05-control-flow/01-unconditional-br.md`](05-control-flow/01-unconditional-br.md)
12. [`05-control-flow/02-conditional-br.md`](05-control-flow/02-conditional-br.md)
13. All six files in `08-pitfalls/` — each is ≤ 5 minutes

Now you can read and write straightforward IR. Verifier failures should
make sense.

Practice next: [`exercises/003-loop-counter.prompt.md`](exercises/003-loop-counter.prompt.md),
[`exercises/004-global-load-store.prompt.md`](exercises/004-global-load-store.prompt.md), and
[`exercises/005-struct-gep.prompt.md`](exercises/005-struct-gep.prompt.md).

## Path 3: Deep dive (one sitting; pick up the rest as needed)

Read everything in numerical order:

```
00-foundations/   →  01-syntax/   →  02-types/  →  03-constants/
                                                    ↓
05-control-flow/  ←  04-memory/  ←─────────────────┘
        ↓
08-pitfalls/      →  10-grammar/  →  reference/
```

Cross-references inside each chapter (`See also:`) let you jump
forward when curiosity strikes; come back via the index.

Practice next: complete all exercises in [`exercises/README.md`](exercises/README.md)
and compare against the standalone `.ll` solutions.

## Chapter dependency graph

Mostly linear. The real dependencies:

```
foundations ────────┐
        ↓           ↓
     syntax ───→ types ───→ constants
        ↓           ↓
     memory ←──────┘
        ↓
   control-flow
        ↓
   pitfalls (read alongside everything above)
        ↓
   grammar (open as reference)
```

## What's intentionally NOT here yet

If your task touches these, you'll need external references:

- **Optimization passes** — `opt` flags, pass pipeline design
- **MLIR** — the dialect framework above LLVM IR
- **Backend / codegen** — `llc`, target lowering, register allocation
- **Debug info (DWARF)** — beyond `!dbg !N` attachment syntax
- **JIT (`lli`, ORC, MCJIT)**
- **C/C++ frontend internals** — Clang, AST, lowering rules
- **TableGen** — used to define targets and instruction sets

These are roadmap items; PRs welcome.

## Self-test prompts

After each path, the agent should be able to answer (without grepping
LLVM source):

**After Path 1**
- What does SSA stand for and why does it require phi nodes?
- What's the difference between `@foo` and `%foo`?
- Why must a basic block end with a terminator?

**After Path 2**
- What's the type of the pointer returned by `alloca i32`?
- Why is `add i32 (load ...), 1` invalid as a single expression?
- When does a `br i1` need two labels, and what's the type of the
  condition?

**After Path 3**
- Why does the grammar treat `Linkage` and `ExternLinkage` as separate
  productions?
- What's the difference between `dso_local` and `dso_preemptable`?
- What's the layout convention for `%bcir.claim`-style aggregate types,
  and what breaks when consumers disagree on the field count?
  (See [`08-pitfalls/05-type-schema-drift.md`](08-pitfalls/05-type-schema-drift.md).)
