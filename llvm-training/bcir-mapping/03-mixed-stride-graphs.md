# Mixed-Stride Graphs

Mixed-stride graph lowering covers layouts where vertex or edge attributes are
not a simple contiguous `i + 1` walk. Examples include row/column graph tiles,
structure-of-arrays plus array-of-structures hybrids, gathered edge lists, and
claim streams with compact stride codes plus full immediates.

## BCIR-level meaning

- A stride describes how to move from one logical graph element to the next in a
  memory representation.
- Mixed stride combines row stride, element stride, byte bias, gather indexes,
  lane grouping, or schedule batch ranges.
- BCIR claim control bits can carry a compact stride code; an immediate can carry
  the full stride encoding when the compact code is insufficient.
- Graph schedules may group claims by phase, lane, opcode, type, or memory
  locality to make irregular layouts executable in predictable batches.

## Likely LLVM IR representation

- Compute byte offsets explicitly with `mul` and `add` before a byte-wise GEP:
  `getelementptr i8, ptr %base, i64 %offset`.
- Use separate arrays for gathered indexes when stride cannot be expressed as a
  simple affine formula.
- Prefer `i64` for byte offsets and table counts; cast smaller IDs once at the
  boundary.
- Represent schedule groups as arrays of batch/range structs and loop over them
  with canonical PHIs.
- See [`examples/mixed-stride.ll`](examples/mixed-stride.ll) for row stride,
  element stride, and bias combined into one load address.

## Relevant runtime ABI structs/functions

- [`@bcir.claim.stride_code`](../../runtime/llvm/bcir_claim_accessors.ll),
  [`@bcir.claim.opstride`](../../runtime/llvm/bcir_claim_accessors.ll), and
  [`@bcir.claim.imm`](../../runtime/llvm/bcir_claim_accessors.ll) expose compact
  and full stride information.
- [`%bcir.batch`](../../runtime/llvm/bcir_schedule_schema.ll),
  [`%bcir.phase.range`](../../runtime/llvm/bcir_schedule_schema.ll),
  [`%bcir.prefetch.profile`](../../runtime/llvm/bcir_schedule_schema.ll), and
  [`%bcir.stream.pack`](../../runtime/llvm/bcir_schedule_schema.ll) group mixed
  layouts into executable streams.
- [`@bcir.op.ggg.load.v8i32.ref`](../../runtime/llvm/bcir_ops.ll) and
  [`@bcir.op.ggg.store.v8i32.ref`](../../runtime/llvm/bcir_ops.ll) show explicit
  gathered index lowering.
- [`@bcir.classify.memory_lane`](../../runtime/llvm/bcir_lane_classifier.ll)
  uses lane and stride fields to classify memory behavior.
- Existing examples: [`runtime/llvm/bcir_examples_phase3.ll`](../../runtime/llvm/bcir_examples_phase3.ll),
  [`runtime/llvm/bcir_stream_pack.ll`](../../runtime/llvm/bcir_stream_pack.ll),
  and [`llvm-training/exercises/013-mixed-stride-indexing.prompt.md`](../exercises/013-mixed-stride-indexing.prompt.md).

## Verifier risks

- Do not put `mul`, `add`, or `load` inline inside another instruction operand;
  create SSA values in sequence.
- Loop PHIs for mixed-stride walks must name the preheader and backedge blocks
  exactly.
- Vector gather/scatter shuffles and masks must use vector constants with the
  exact vector length and element type.
- If stride decoding calls an intrinsic with `immarg`, pass a literal immediate,
  not a runtime-decoded value.

## Optimization risks

- LLVM may not prove that two irregular strides do not alias, blocking
  vectorization and unrolling.
- Incorrect `inbounds` on byte GEPs is dangerous when the graph format admits
  sentinel or biased addresses.
- Integer overflow in offset math can become poison with `nsw`/`nuw`; add those
  flags only when BCIR bounds prove them.
- Prefetching the wrong mixed-stride address is usually valid IR but can hurt
  performance or touch unsuitable memory domains.

## Pitfall links

- [`01-nested-instruction-expressions.md`](../08-pitfalls/01-nested-instruction-expressions.md)
- [`02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md)
- [`06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md)
- [`11-address-space-confusion.md`](../08-pitfalls/11-address-space-confusion.md)
- [`12-vectorization-blocked-by-aliasing.md`](../08-pitfalls/12-vectorization-blocked-by-aliasing.md)
- [`13-pass-pipeline-ordering-surprise.md`](../08-pitfalls/13-pass-pipeline-ordering-surprise.md)
