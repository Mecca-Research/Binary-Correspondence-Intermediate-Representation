# Mixed-Stride Graphs

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


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
- See [`examples/hardware-aware-gem-lowering.ll`](examples/hardware-aware-gem-lowering.ll)
  for a mixed-stride GEM tile that keeps A, B, and accumulator fragments in
  vector SSA values before a backend hook consumes them.

## Hardware-aware GEM tile lowering

A hardware-aware GEM operation is still a mixed-stride graph problem before it
is a target instruction problem. Lower the graph layout first, then preserve the
shape that the backend cares about:

1. Decode each logical operand address independently. For a tile multiply, A may
   use `(row * a_row_stride) + (k * a_k_stride)`, B may use
   `(k * b_k_stride) + (col * b_col_stride)`, and C may use
   `(row * c_row_stride) + (col * c_col_stride)`.
2. Convert byte offsets to byte-wise GEPs. Avoid `inbounds` unless BCIR layout
   proofs exclude biased or sentinel addresses.
3. Load register-oriented fragments into vector SSA values such as
   `<4 x float> %a_vec`, `<4 x float> %b_vec`, and `<4 x float> %acc_vec`. This
   makes operand grouping visible to instruction selection without requiring the
   IR optimizer to infer a matrix tile from scalar loads.
4. Attach hierarchical memory hint metadata to the address-producing
   instructions, loads/stores, and optional backend hook. Keep the metadata
   advisory: it should guide scheduling, prefetch, memory-space choice, or JIT
   policy, not change language-level correctness.

Example: [`examples/hardware-aware-gem-lowering.ll`](examples/hardware-aware-gem-lowering.ll)
shows all four steps plus a custom intrinsic-shaped backend hook.

## Relevant runtime ABI structs/functions

- `@bcir.claim.stride_code`,
  `@bcir.claim.opstride`, and
  `@bcir.claim.imm` expose compact
  and full stride information.
- `%bcir.batch`,
  `%bcir.phase.range`,
  `%bcir.prefetch.profile`, and
  `%bcir.stream.pack` group mixed
  layouts into executable streams.
- `@bcir.op.ggg.load.v8i32.ref` and
  `@bcir.op.ggg.store.v8i32.ref` show explicit
  gathered index lowering.
- `@bcir.classify.memory_lane`
  uses lane and stride fields to classify memory behavior.
- Existing examples: `runtime/llvm/bcir_examples_phase3.ll`,
  `runtime/llvm/bcir_stream_pack.ll`,
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

## Hardware-aware continuation

For GAA-aware register-allocation advice and hierarchical memory hints layered on correct mixed-stride addressing, continue with [`../19-hardware-aware/06-register-allocation-and-memory-hints.md`](../19-hardware-aware/06-register-allocation-and-memory-hints.md).
