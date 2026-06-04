# Exercises

This directory contains beginner LLVM IR writing exercises. Follow the
conventions in [`../EXAMPLES.md`](../EXAMPLES.md). Each exercise should include:

- a prompt describing the task;
- the expected command to check the learner's answer or the checked-in solution;
- the expected observation, such as successful assembly or a specific diagnostic;
- an optional standalone `*.solution.ll` file containing one expected answer.

Solutions must assemble with LLVM >= 15, where opaque pointers are the default.
Use `ptr` for pointer-typed values instead of typed pointers such as `i32*`.

If your LLVM tools are installed with a version suffix, replace `llvm-as` in the
commands with the matching binary, for example `llvm-as-15` or `llvm-as-18`.

## Exercise list

1. [`001-add.prompt.md`](001-add.prompt.md) — write `@add(i32, i32)`.
2. [`002-if-else-phi.prompt.md`](002-if-else-phi.prompt.md) — write an if/else with `phi`.
3. [`003-loop-counter.prompt.md`](003-loop-counter.prompt.md) — write a loop counter.
4. [`004-global-load-store.prompt.md`](004-global-load-store.prompt.md) — load and store a global.
5. [`005-struct-gep.prompt.md`](005-struct-gep.prompt.md) — index a struct field with `getelementptr`.

## Verification

Run a solution through the assembler and discard the bitcode output:

```sh
llvm-as -disable-output llvm-training/exercises/001-add.solution.ll
```

Each prompt gives the exact command for its matching solution file.
