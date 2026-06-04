# LLVM Training Tools

These scripts keep the checked LLVM training examples and smoke fixtures aligned
between local development and CI.

| Script | Purpose | Required tools |
|---|---|---|
| `verify-examples.sh` | Discovers known-good standalone `*/examples/*.ll` files, assembles each one with `llvm-as`, and runs `opt -passes=verify`. | `llvm-as`, `opt` |
| `verify-manifest.sh` | Compares discovered standalone `*/examples/*.ll` files with the human-readable table in `llvm-training/examples/README.md`. | `bash`, `python3` |
| `smoke-llc.sh` | Lowers a curated portable subset through `llc`. | `llc` |
| `smoke-lli.sh` | Runs curated examples that have safe executable entry points. | `lli` |

## Manifest contract

`llvm-training/examples/README.md` is the human-readable manifest for
standalone LLVM IR examples. Add a row there whenever a new checked
`*/examples/*.ll` file is added, including final LLVM IR snapshots produced by
MLIR walkthroughs such as `14-mlir-bridge/examples/bcir-final.ll`.

Run these from the repository root after adding or removing examples:

```bash
./llvm-training/tools/verify-manifest.sh
./llvm-training/tools/verify-examples.sh
```

The manifest checker has no LLVM dependency and is wired into CI plus the CMake
target below:

```bash
cmake --build build --target llvm-training-verify-manifest
```

The example verifier intentionally skips `.ll.txt` files and filenames
containing `invalid`; use those names only for expected-failure fixtures.
