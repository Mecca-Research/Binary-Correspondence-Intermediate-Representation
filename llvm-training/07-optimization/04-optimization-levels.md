# Optimization Levels and Predefined Pipelines

LLVM exposes predefined optimization pipelines so users do not have to assemble
large pass sequences by hand. In `opt` with the new pass manager, the most
teachable spelling is `default<LEVEL>`.

Official references:

- [LLVM's Analysis and Transform Passes](https://llvm.org/docs/Passes.html)
- [Using the New Pass Manager](https://llvm.org/docs/NewPassManager.html)
- [`opt` command guide](https://llvm.org/docs/CommandGuide/opt.html)

## New pass manager pipeline examples

```bash
opt -S -passes='default<O1>' input.ll -o input.O1.ll
opt -S -passes='default<O2>' input.ll -o input.O2.ll
opt -S -passes='default<O3>' input.ll -o input.O3.ll
opt -S -passes='default<Os>' input.ll -o input.Os.ll
opt -S -passes='default<Oz>' input.ll -o input.Oz.ll
```

Quote the pipeline because `<` and `>` can be interpreted by your shell.

For a standalone pass:

```bash
opt -passes=verify input.ll -disable-output
opt -S -passes=mem2reg input.ll -o promoted.ll
opt -S -passes=instcombine input.ll -o combined.ll
```

## Conceptual map

| Level | Conceptual intent | Typical tradeoff |
|---|---|---|
| `-O0` | Preserve source shape and compile quickly | Minimal optimization; Clang may emit `optnone` attributes that block many `opt` transforms |
| `-O1` | Enable cheap, broadly useful cleanups | Better IR/code with modest compile-time cost |
| `-O2` | Balanced default optimization pipeline | Strong general-purpose speedups without the most aggressive size/compile-time costs |
| `-O3` | More aggressive performance pipeline | May increase code size and compile time; not always faster at runtime |
| `-Os` | Optimize for speed while reducing code-size growth | Avoids some size-expanding transforms |
| `-Oz` | Optimize more strongly for minimum size | May sacrifice speed to shrink code further |

These levels are not single passes. They expand to many analyses, transforms,
and utilities in an order chosen by LLVM's pipeline builder.

## Inspecting what changed

Use textual IR output and compare:

```bash
opt -S -passes='default<O2>' examples/dead-code-before.ll -o /tmp/dead-code-o2.ll
diff -u examples/dead-code-before.ll /tmp/dead-code-o2.ll
```

You can compare optimization levels too:

```bash
opt -S -passes='default<O2>' examples/loop-before.ll -o /tmp/loop-o2.ll
opt -S -passes='default<O3>' examples/loop-before.ll -o /tmp/loop-o3.ll
diff -u /tmp/loop-o2.ll /tmp/loop-o3.ll
```

## `-O0` and `optnone`

When IR comes from Clang at `-O0`, functions may carry attributes such as
`optnone` and `noinline`. Many optimization passes intentionally skip
`optnone` functions. For pass experiments, generate IR with a low-but-optimizable
frontend mode when possible, or remove attributes only if you understand the
semantic/debugging consequences.

## Why hand-written pipelines still matter

Predefined pipelines are best for production-like optimization. Hand-written
pipelines are best for learning and debugging:

```bash
opt -S -passes='mem2reg,instcombine,simplifycfg,adce,gvn' input.ll -o out.ll
```

Use them to isolate one idea at a time, then return to predefined pipelines for
real performance comparisons.

## Pitfalls recap

- Do not rely on accidental pass ordering from a previous LLVM release.
- Do not assume `-O3` beats `-O2`, `-Os`, or `-Oz` on your workload.
- Do not paste legacy pass syntax into a new-pass-manager command and expect it
  to work.
- Always inspect changed IR with `opt -S` before drawing conclusions.

## See also

- [`01-pass-model.md`](01-pass-model.md)
- [`03-common-transform-passes.md`](03-common-transform-passes.md)
- [`examples/dead-code-before.ll`](examples/dead-code-before.ll)
- [`examples/loop-before.ll`](examples/loop-before.ll)
