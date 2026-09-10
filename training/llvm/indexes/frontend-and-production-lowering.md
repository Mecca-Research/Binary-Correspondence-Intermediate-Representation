# Index: Frontends, dialect lowering, and production-shaped code

Use this index when the question is how source becomes IR, how a dialect becomes
LLVM, or what the difference is between a sketch that explains an idea and code
shaped to survive review.

| Concept | Where it is covered |
|---|---|
| How C constructs become LLVM IR | [`20-clang-frontend/06-bcir-cfront-correspondence.md`](../20-clang-frontend/06-bcir-cfront-correspondence.md) |
| Comparing BCIR's own C front against Clang | [`20-clang-frontend/06-bcir-cfront-correspondence.md`](../20-clang-frontend/06-bcir-cfront-correspondence.md) |
| Writing a conversion pass fit for production | [`18-mlir-lowering-to-llvm/09-production-conversion-pass.md`](../18-mlir-lowering-to-llvm/09-production-conversion-pass.md) |
| The skeleton of a dialect-to-dialect conversion | [`18-mlir-lowering-to-llvm/examples/bcir-conversion-pass-skeleton.cpp.md`](../18-mlir-lowering-to-llvm/examples/bcir-conversion-pass-skeleton.cpp.md) |
| A dialect small enough to read and complete enough to build | [`18-mlir-lowering-to-llvm/examples/production-dialect/README.md`](../18-mlir-lowering-to-llvm/examples/production-dialect/README.md) |
| Building a pass outside the LLVM source tree | [`17-new-pass-manager/examples/pass-plugin/README.md`](../17-new-pass-manager/examples/pass-plugin/README.md) |
| A pass that changes its own pipeline as it runs | [`17-new-pass-manager/examples/adaptive-pipeline-sketch.cpp.md`](../17-new-pass-manager/examples/adaptive-pipeline-sketch.cpp.md) |
| Mapping graph vertices, edges, and attributes into IR | [`bcir-mapping/01-vertex-edge-attribute.md`](../bcir-mapping/01-vertex-edge-attribute.md) |
| A one-page reminder of BCIR lowering shapes | [`quickref/bcir-lowering.md`](../quickref/bcir-lowering.md) |
| A one-page reminder of metadata syntax | [`quickref/metadata.md`](../quickref/metadata.md) |
| A one-page reminder of opaque pointer rules | [`quickref/opaque-pointers.md`](../quickref/opaque-pointers.md) |
| A one-page reminder of vectorization pragmas and hints | [`quickref/vectorization.md`](../quickref/vectorization.md) |
