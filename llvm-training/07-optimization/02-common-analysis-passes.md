# Common Analysis and Utility Passes

Analysis passes answer questions about IR. They usually do not rewrite the IR;
transform passes consume their answers to make safer or more profitable changes.
Utility passes such as verifiers and printers make the pass pipeline observable.

Official references:

- [LLVM's Analysis and Transform Passes](https://llvm.org/docs/Passes.html)
- [Using the New Pass Manager](https://llvm.org/docs/NewPassManager.html)
- [`opt` command guide](https://llvm.org/docs/CommandGuide/opt.html)

## Verifier: the first utility pass to know

```bash
opt -passes=verify input.ll -disable-output
```

`verify` checks IR invariants: type consistency, terminators, PHI predecessor
rules, dominance requirements, metadata shape, and more. Run it before blaming
an optimizer for malformed input.

## Alias analysis

**Alias analysis** asks whether two memory locations may refer to the same
storage. Optimizations such as load/store elimination, GVN, LICM, and many loop
transforms become more powerful when aliasing is constrained.

Conceptual answers include:

| Answer | Meaning |
|---|---|
| No alias | The locations do not overlap |
| May alias | The analysis cannot prove they are separate |
| Partial alias | The locations overlap but neither fully contains the other |
| Must alias | The locations are the same storage |

Useful habits:

- Preserve accurate pointer provenance when generating IR.
- Add metadata or attributes only when they are true.
- Remember that opaque pointers still require typed `load`, `store`, and GEP
  source element types.

## CFG viewing and printing

The **control-flow graph** (CFG) is the graph of basic blocks and branch edges.
CFG utilities are usually printing/viewing utilities rather than transforms.
Depending on LLVM version and build options, common workflows include printing a
CFG view or emitting Graphviz `.dot` files.

Examples to try with your installed `opt --help` output:

```bash
opt -passes='print<cfg>' input.ll -disable-output
opt -passes=dot-cfg input.ll -disable-output
```

If a printer name is unavailable, check `opt --print-passes` and `opt --help`;
printer pass names have changed across LLVM releases.

## Loop analysis

Loop analysis identifies natural loops, headers, latches, exits, and nesting.
Loop transforms use this information to decide whether an operation can be
hoisted, unswitched, unrolled, vectorized, or deleted.

Example exploration commands:

```bash
opt -passes='print<loops>' examples/loop-before.ll -disable-output
opt -S -passes='loop-unroll' examples/loop-before.ll -o /tmp/loop-unroll.ll
```

Loop-related output is easiest to read when the input has simple structured
control flow and a clear induction variable.

## Scalar evolution

**Scalar evolution** (often shortened to SCEV) models how scalar expressions
change across loop iterations. It is especially useful for induction variables,
trip-count reasoning, bounds checks, and loop profitability decisions.

Example exploration command:

```bash
opt -passes='print<scalar-evolution>' examples/loop-before.ll -disable-output
```

If your LLVM uses a different printer spelling, check:

```bash
opt --print-passes | rg 'scalar|scev|loop'
```

## Analysis invalidation

Transform passes can invalidate analysis results. For example, deleting a block
changes CFG analyses, and rewriting memory operations can change alias-analysis
queries. The new pass manager tracks preserved analyses so later passes know
which cached answers are still valid.

## See also

- [`01-pass-model.md`](01-pass-model.md)
- [`03-common-transform-passes.md`](03-common-transform-passes.md)
- [`examples/loop-before.ll`](examples/loop-before.ll)
