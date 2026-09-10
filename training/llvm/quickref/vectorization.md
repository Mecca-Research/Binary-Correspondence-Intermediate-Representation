# Quickref: Vectorization

## First question: loop or SLP?

- **Loop vectorizer** widens work across loop iterations and needs loop legality, trip-count, alias, and side-effect facts.
- **SLP vectorizer** packs isomorphic scalar operations inside a basic block and needs compatible types, operation order, and low shuffle cost.

## Legality checklist

- Memory operations can be reordered or widened safely: alias facts, `noalias`, TBAA, alignment, and dependence distance are credible.
- Loop control flow is simple enough, or predication/masking can represent the conditional work.
- Calls are vectorizable, side-effect-free, or mapped to vector intrinsics/libraries.
- Reductions and inductions are recognizable after canonicalization.
- Metadata hints match reality; do not force vectorization around unresolved dependencies.

## Diagnostics workflow

1. Compile or run `opt` with optimization remarks for missed/vectorized loops.
2. Inspect before/after IR for widened types, vector loads/stores, masks, interleaving, gathers/scatters, and reduction code.
3. If vectorization is blocked, check aliasing, non-canonical loops, unknown calls, volatile/atomic operations, and pass order.
4. Compare against the chapter examples before adding metadata or rewriting control flow.

## Common IR shapes

| Shape | Meaning |
| --- | --- |
| `<N x T>` | Fixed-width vector value. |
| `<vscale x N x T>` | Scalable vector value. |
| `llvm.masked.load/store` | Predicated memory operation. |
| `llvm.vector.reduce.*` | Reduction across lanes. |
| Interleaved access groups | Widened strided memory access, often with shuffles. |

## Deep links

- [`../09-vectorization/README.md`](../09-vectorization/README.md)
- [`../09-vectorization/01-loop-vectorizer.md`](../09-vectorization/01-loop-vectorizer.md)
- [`../09-vectorization/02-slp-vectorizer.md`](../09-vectorization/02-slp-vectorizer.md)
- [`../09-vectorization/04-vectorization-legality.md`](../09-vectorization/04-vectorization-legality.md)
- [`../08-pitfalls/12-vectorization-blocked-by-aliasing.md`](../08-pitfalls/12-vectorization-blocked-by-aliasing.md)
