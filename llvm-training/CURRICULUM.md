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
7. [`02-types/02-composite-types.md`](02-types/02-composite-types.md) — include the GEP basics in
   `Accessing struct fields` / aggregate access examples
8. [`02-types/03-opaque-and-pointer-types.md`](02-types/03-opaque-and-pointer-types.md)
9. [`04-memory/01-alloca.md`](04-memory/01-alloca.md)
10. [`04-memory/02-load-store.md`](04-memory/02-load-store.md) — typed memory operations,
    especially explicit access types with opaque pointers
11. [`05-control-flow/01-unconditional-br.md`](05-control-flow/01-unconditional-br.md)
12. [`05-control-flow/02-conditional-br.md`](05-control-flow/02-conditional-br.md)
13. [`06-metadata/01-metadata-basics.md`](06-metadata/01-metadata-basics.md) — metadata syntax and common attachments
14. [`06-metadata/02-debug-info.md`](06-metadata/02-debug-info.md) — source locations and debug-info nodes
15. [`06-metadata/03-profile-and-optimization-metadata.md`](06-metadata/03-profile-and-optimization-metadata.md) — branch weights and loop hints
16. [`reference/instruction-quickref.md`](reference/instruction-quickref.md) — read the sections for
    terminators, comparison, memory, conversion, and other/call instructions
17. All six files in `08-pitfalls/` — each is ≤ 5 minutes

Now you can read and write straightforward IR. Verifier failures should
make sense.

Practice next: [`exercises/003-loop-counter.prompt.md`](exercises/003-loop-counter.prompt.md),
[`exercises/004-global-load-store.prompt.md`](exercises/004-global-load-store.prompt.md), and
[`exercises/005-struct-gep.prompt.md`](exercises/005-struct-gep.prompt.md).


## After basics: concurrency path

After Path 2, add this chapter when reading or writing shared-memory IR:

1. [`11-concurrency/01-atomic-orderings.md`](11-concurrency/01-atomic-orderings.md) — not atomic vs `unordered`, `monotonic`, acquire/release, `acq_rel`, and `seq_cst`
2. [`11-concurrency/02-atomic-instructions.md`](11-concurrency/02-atomic-instructions.md) — `load atomic`, `store atomic`, `cmpxchg`, `atomicrmw`, and `fence` syntax
3. [`11-concurrency/03-volatile-vs-atomic.md`](11-concurrency/03-volatile-vs-atomic.md) — why volatile access behavior and atomic synchronization are orthogonal

Practice next: inspect and assemble the examples in
[`11-concurrency/examples/atomic-counter.ll`](11-concurrency/examples/atomic-counter.ll),
[`11-concurrency/examples/cmpxchg-loop.ll`](11-concurrency/examples/cmpxchg-loop.ll), and
[`11-concurrency/examples/fence.ll`](11-concurrency/examples/fence.ll).


## After basics: optimization path

After Path 2, add this chapter when you need to inspect or explain optimizer
behavior with `opt`:

1. [`07-optimization/01-pass-model.md`](07-optimization/01-pass-model.md) — analysis vs transform vs utility passes, new pass manager syntax, and common pitfalls
2. [`07-optimization/02-common-analysis-passes.md`](07-optimization/02-common-analysis-passes.md) — alias analysis, CFG printing/viewing, loop analysis, and scalar evolution
3. [`07-optimization/03-common-transform-passes.md`](07-optimization/03-common-transform-passes.md) — `mem2reg`, `instcombine`, `simplifycfg`, `adce`, `gvn`, and `loop-unroll`
4. [`07-optimization/04-optimization-levels.md`](07-optimization/04-optimization-levels.md) — conceptual map for `-O0`, `-O1`, `-O2`, `-O3`, `-Os`, and `-Oz`

Practice next: run the commands embedded in
[`07-optimization/examples/mem2reg-before.ll`](07-optimization/examples/mem2reg-before.ll),
[`07-optimization/examples/dead-code-before.ll`](07-optimization/examples/dead-code-before.ll), and
[`07-optimization/examples/loop-before.ll`](07-optimization/examples/loop-before.ll).

## Path 3: Deep dive (one sitting; pick up the rest as needed)

Read everything in numerical order:

```
00-foundations/   →  01-syntax/   →  02-types/  →  03-constants/
                                                    ↓
05-control-flow/  ←  04-memory/  ←─────────────────┘
        ↓
06-metadata/
        ↓
09-vectorization/
        ↓
08-pitfalls/      →  10-grammar/  →  11-concurrency/  →  reference/
```

Cross-references inside each chapter (`See also:`) let you jump
forward when curiosity strikes; come back via the index.

Practice next: complete all exercises in [`exercises/README.md`](exercises/README.md)
and compare against the standalone `.ll` solutions.

## Path 4: Post-optimization vectorization path

After Path 2, or after reading the optimization metadata chapter, add this
focused path when you need to understand LLVM's vectorized IR and diagnostics:

1. Re-read [`06-metadata/03-profile-and-optimization-metadata.md`](06-metadata/03-profile-and-optimization-metadata.md)
   for loop transformation hints and the limits of metadata.
2. Read [`09-vectorization/README.md`](09-vectorization/README.md) for the
   difference between the Loop Vectorizer and SLP Vectorizer.
3. Run the commands in [`09-vectorization/examples/sum-loop.c`](09-vectorization/examples/sum-loop.c)
   to compare successful and missed loop-vectorization remarks.
4. Run `opt -S -passes=loop-vectorize` on
   [`09-vectorization/examples/sum-loop.ll`](09-vectorization/examples/sum-loop.ll)
   and inspect vector loop structure, vector loads/stores, and reductions.
5. Run `opt -S -passes=slp-vectorizer` on
   [`09-vectorization/examples/slp-scalars.ll`](09-vectorization/examples/slp-scalars.ll)
   and inspect vector packing, `shufflevector`, and straight-line vector IR.
6. Repeat with `-force-vector-width` and `-force-vector-interleave` to separate
   legality questions from profitability choices.

This path is intentionally about reading and experimenting with transformed IR,
not about writing a custom LLVM pass pipeline.

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
   metadata
        ↓
   pitfalls (read alongside everything above)
        ↓
   grammar (open as reference)
        ↓
   concurrency (when shared memory appears)
```

## What's intentionally NOT here yet

If your task touches these, you'll need external references:

- **Custom optimization pass design** — pass-manager internals beyond the introductory `opt` vectorization commands
- **MLIR** — the dialect framework above LLVM IR
- **Backend / codegen** — `llc`, target lowering, register allocation
- **JIT (`lli`, ORC, MCJIT)**
- **C/C++ frontend internals** — Clang, AST, lowering rules
- **TableGen** — used to define targets and instruction sets
- **Calls / returns / comparisons** — a small dedicated chapter may be worth adding if
  this training set keeps expanding beyond the quick reference

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
- How do you follow an instruction `!dbg` attachment back to a source
  file, line, and column?
- Why does the grammar treat `Linkage` and `ExternLinkage` as separate
  productions?
- How do `opt -passes=mem2reg`, `opt -passes=instcombine`, and
  `opt -passes='default<O2>'` differ in scope and intent?
- Why might `-O3` be a bad default for a size-sensitive workload?
- What's the difference between `dso_local` and `dso_preemptable`?
- What's the layout convention for `%bcir.claim`-style aggregate types,
  and what breaks when consumers disagree on the field count?
  (See [`08-pitfalls/05-type-schema-drift.md`](08-pitfalls/05-type-schema-drift.md).)

**After Path 4**
- When should you expect the Loop Vectorizer rather than the SLP Vectorizer to act?
- What source or IR facts help LLVM prove a loop has predictable memory access and no unsafe dependencies?
- Which commands show successful vs missed loop-vectorization remarks?
- What IR clues suggest vectorization occurred (`<N x T>`, vector loads/stores, `shufflevector`, reductions)?
