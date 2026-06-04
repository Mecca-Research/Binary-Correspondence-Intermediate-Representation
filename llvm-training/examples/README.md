# Standalone LLVM IR example manifest

Every standalone LLVM IR example in this training pack lives under a chapter
`examples/` directory and must assemble with a modern `llvm-as` (LLVM >= 15,
opaque pointers). Embedded fenced `llvm` snippets in chapter prose are not part
of this manifest unless they are moved into one of these files.

Run the full manifest with:

```bash
find llvm-training -path '*/examples/*.ll' -print0 | sort -z | while IFS= read -r -d '' f; do
  llvm-as "$f" -o /dev/null || exit 1
done
```

| File | Expected command |
|---|---|
| `llvm-training/00-foundations/examples/simple-add.ll` | `llvm-as llvm-training/00-foundations/examples/simple-add.ll -o /dev/null` |
| `llvm-training/00-foundations/examples/ssa-phi.ll` | `llvm-as llvm-training/00-foundations/examples/ssa-phi.ll -o /dev/null` |
| `llvm-training/01-syntax/examples/module-anatomy.ll` | `llvm-as llvm-training/01-syntax/examples/module-anatomy.ll -o /dev/null` |
| `llvm-training/02-types/examples/types-cookbook.ll` | `llvm-as llvm-training/02-types/examples/types-cookbook.ll -o /dev/null` |
| `llvm-training/06-metadata/examples/debug-location.ll` | `llvm-as llvm-training/06-metadata/examples/debug-location.ll -o /dev/null` |
| `llvm-training/06-metadata/examples/loop-metadata.ll` | `llvm-as llvm-training/06-metadata/examples/loop-metadata.ll -o /dev/null` |
| `llvm-training/11-concurrency/examples/atomic-counter.ll` | `llvm-as llvm-training/11-concurrency/examples/atomic-counter.ll -o /dev/null` |
| `llvm-training/11-concurrency/examples/cmpxchg-loop.ll` | `llvm-as llvm-training/11-concurrency/examples/cmpxchg-loop.ll -o /dev/null` |
| `llvm-training/11-concurrency/examples/fence.ll` | `llvm-as llvm-training/11-concurrency/examples/fence.ll -o /dev/null` |
