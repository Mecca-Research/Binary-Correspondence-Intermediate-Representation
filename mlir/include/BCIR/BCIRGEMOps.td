//===- BCIRGEMOps.td - GEM StreamPack ops (BCIR-4) ----------*- tablegen -*-===//
#ifndef BCIR_GEM_OPS
#define BCIR_GEM_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRTypes.td"
include "BCIR/BCIRAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_GEMStreamPackOp : BCIR_Op<"gem.stream_pack", [Symbol, SymbolTable, NoTerminator]> {
  let summary = "Hydrated GEM StreamPack (hot executable artifact)";
  // `pipeline_depth` (StreamPack ABI v2, append-only): phases in flight; 2 =
  // double-buffered software pipelining (mirrors gem.streampack.hydrate_pipelined).
  let arguments = (ins
    SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$source_plan,
    I64Attr:$topo_gen, I64Attr:$map_gen, I64Attr:$data_gen,
    DefaultValuedAttr<I32Attr, "1">:$pipeline_depth);
  // MLIR 22 forbids a Symbol op from producing an SSA result; the StreamPack is
  // addressed by its symbol -- the old !bcir.stream handle was vestigial.
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_GEMPrefetchOp : BCIR_Op<"gem.prefetch", [Symbol]> {
  let summary = "GEM prefetch profile (operand data, not descriptors)";
  // `buffers` (ABI v2, append-only): 2 = a double-buffer contract feeding the
  // next pipelined phase while the current one computes.
  let arguments = (ins SymbolNameAttr:$sym_name, I32Attr:$distance,
                       FlatSymbolRefArrayAttr:$targets, StrAttr:$hint, StrAttr:$pattern,
                       DefaultValuedAttr<I32Attr, "1">:$buffers);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_GEMScheduleOp : BCIR_Op<"gem.schedule", [Symbol]> {
  let summary = "Duration-aware schedule certificate (EFT waves / token DAG)";
  // mode: "waves" (unit-time CT2), "eft" (HEFT-lite: LPT + earliest finish +
  // locality + the bandwidth knee), "tokens" (!bcir.token DAG, pipelined
  // phases). Mirrors bcir/gem/schedule.py; consistency is verified under R9.
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$plan,
                       StrAttr:$mode, I64Attr:$makespan,
                       DefaultValuedAttr<I32Attr, "1">:$knee,
                       DefaultValuedAttr<I32Attr, "1">:$pipeline_depth);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_GEMBlockOp : BCIR_Op<"gem.block", [Symbol]> {
  let summary = "GEM block descriptor";
  let arguments = (ins SymbolNameAttr:$sym_name, I64Attr:$base, I64Attr:$count,
                       I64Attr:$strideA, I64Attr:$strideB, I64Attr:$strideD);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_GEMMatmulOp : BCIR_Op<"gem.matmul", [Symbol]> {
  let summary = "Planned GEM matmul: the K_BCIR-chosen tiling/loop-order realization (B1)";
  // The law-rail record of the B1 plan (bcir/kbcir/matmul.py::plan_matmul). A matmul is a grid of dot
  // products; its realization -- tile extents + loop order -- is a FREE choice the dual-semiring cost
  // search picks (min,+ over the path cost; max,+ for the roofline bottleneck), and every realization
  // reproduces the same result. The op carries the shape (m/n/k), the chosen tiling (tile_m/n/k +
  // loop_order), the analytic compute/memory roofline terms, the bottleneck = max(compute, memory), and
  // the quantized lane width (0 = dense). `quant_bits > 0` ties to the A1 Q8 bridge (the R17 accuracy
  // bound rides the accuracy cost axis; mirrors CostVector via matmul.cost_vector). hasVerifier checks
  // well-formedness: positive dims, each tile in [1, dim], a known loop order, and bottleneck == the max.
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$m, I64Attr:$n, I64Attr:$k,
    I64Attr:$tile_m, I64Attr:$tile_n, I64Attr:$tile_k,
    StrAttr:$loop_order,
    I64Attr:$compute_cost, I64Attr:$mem_cost, I64Attr:$bottleneck,
    DefaultValuedAttr<I32Attr, "0">:$quant_bits);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def BCIR_GEMActivationOp : BCIR_Op<"gem.activation", [Symbol]> {
  let summary = "Planned GEM activation: the K_BCIR-chosen lane-width realization of an elementwise tensor op (G1)";
  // The law-rail record of the G1 plan (bcir/kbcir/activation.py::plan_activation). An activation is the
  // elementwise tensor-op family relu/sigmoid/tanh/gelu/softmax: its RESULT is fixed by a single reference,
  // while its REALIZATION -- the lane width of the elementwise loop -- is a FREE choice the cost search
  // picks (max,+ over the roofline bottleneck; min,+ to select the width), exactly as gem.matmul picks a
  // tiling. The op carries the activation kind, the row-major tensor shape, the element dtype ("f32"/"i32"),
  // the softmax last-axis length (0 for the elementwise four), the K_BCIR-chosen lane `width`, the analytic
  // compute/memory roofline terms, the bottleneck = max(compute, memory), and the quantized lane width
  // (quant_bits > 0 ties to the A1 Q8 bridge: the R17 accuracy bound rides the accuracy cost axis; mirrors
  // activation.cost_vector). `acc_bound` records that R17 quantization bound (0 = dense).
  //
  // THE QUARANTINE SPLIT (the load-bearing G1 design constraint, mirrored from activation.check_activation):
  // relu is integer/Q-fixed CLEAN -- max(0,x), 0 ULP, no transcendental, so it lowers on the deterministic
  // rail with NO accuracy contract and MAY be i32 OR f32. The transcendentals (sigmoid/tanh/gelu/softmax)
  // route a libm call (expf/tanhf) through the trusted c.call.libm: FFI edge (the B5 wrap), so they need a
  // FLOAT result -- never i32. hasVerifier enforces the op-level shape/dtype laws check_activation enforces
  // (shape-preserving via spec shape == count, a known dtype, the softmax axis, the quarantine dtype rule,
  // and bottleneck == max). NO new globally-numbered R-law (gem.matmul added none; this is the inline,
  // op-level shape/dtype check the oracle rail runs).
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$kind,
    DenseI64ArrayAttr:$shape,
    StrAttr:$dtype,
    DefaultValuedAttr<I64Attr, "0">:$axis_len,
    DefaultValuedAttr<I64Attr, "1">:$width,
    I64Attr:$compute_cost, I64Attr:$mem_cost, I64Attr:$bottleneck,
    DefaultValuedAttr<I32Attr, "0">:$quant_bits,
    DefaultValuedAttr<I64Attr, "0">:$acc_bound);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def BCIR_GEMConvOp : BCIR_Op<"gem.conv", [Symbol]> {
  let summary = "Planned GEM convolution: the K_BCIR-chosen direct|im2col realization of a 2-D single-group conv (G7)";
  // The law-rail record of the G7 plan (bcir/kbcir/conv.py::plan_conv). A 2-D single-group convolution is
  // a STRUCTURED MATMUL: the im2col (patch-gather) reshaping turns the direct conv into the gemm
  // C[out_pix x C_out] = patches[out_pix x (C_in*Kh*Kw)] @ W[(C_in*Kh*Kw) x C_out], so its REALIZATION --
  // the lowering STRATEGY (direct vs im2col+gemm) and the tile/loop order of that gemm -- is a FREE choice
  // the dual-semiring cost search picks (max,+ over the roofline bottleneck; min,+ to select the strategy),
  // exactly as gem.matmul picks a tiling. Crucially the conv is PRICED THROUGH THE EXISTING matmul roofline
  // (a conv IS a structured matmul; NO bespoke roofline term) -- the im2col gemm's (compute, mem) is exactly
  // matmul.cost_of at the chosen tile; the direct strategy is the UNTILED gemm (whole M,N,K).
  //
  // The op carries the conv shape metadata (in_c/in_h/in_w, out_c, kh/kw, stride, pad), the DERIVED output
  // dims (out_h/out_w) + the equivalent im2col gemm dims (gemm_m = out_h*out_w, gemm_n = out_c,
  // gemm_k = in_c*kh*kw), the K_BCIR-chosen realization (strategy + the inner gemm tile tile_m/n/k +
  // loop_order), the analytic compute/memory roofline terms (priced via the matmul roofline), the
  // bottleneck = max(compute, memory), and the quantized lane width (quant_bits > 0 ties to the A1 Q8
  // bridge: the R17 accuracy bound rides the accuracy cost axis; mirrors conv.cost_vector). `acc_bound`
  // records that R17 quantization bound (0 = dense).
  //
  // hasVerifier enforces the op-level shape/dtype/realization laws check_conv enforces (mirrors
  // bcir/kbcir/conv.py::check_conv, the inline op-level check gem.matmul/gem.activation run; NO new
  // globally-numbered R-law): positive extents, the kernel fits the padded input, a known + preserved
  // dtype, the DERIVED out_h/out_w + gemm dims are internally consistent, the strategy is direct|im2col,
  // each inner tile in [1, gemm dim] with a known loop order, bottleneck == max(compute, mem), and the R17
  // acc_bound is 0 dense / 1 ULP quantized.
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    // the conv shape metadata (NCHW-flat input, C_out*C_in*Kh*Kw weights).
    I64Attr:$in_c, I64Attr:$in_h, I64Attr:$in_w,
    I64Attr:$out_c, I64Attr:$kh, I64Attr:$kw,
    DefaultValuedAttr<I64Attr, "1">:$stride,
    DefaultValuedAttr<I64Attr, "0">:$pad,
    StrAttr:$dtype,
    // the DERIVED output dims + the equivalent im2col gemm dims (carried for the verifier to cross-check).
    I64Attr:$out_h, I64Attr:$out_w,
    I64Attr:$gemm_m, I64Attr:$gemm_n, I64Attr:$gemm_k,
    // the K_BCIR-chosen realization: the strategy + the inner im2col gemm's tile/loop plan.
    StrAttr:$strategy,
    I64Attr:$tile_m, I64Attr:$tile_n, I64Attr:$tile_k,
    StrAttr:$loop_order,
    // the roofline cost (priced through the matmul roofline) + the R17 accuracy axis.
    I64Attr:$compute_cost, I64Attr:$mem_cost, I64Attr:$bottleneck,
    DefaultValuedAttr<I32Attr, "0">:$quant_bits,
    DefaultValuedAttr<I64Attr, "0">:$acc_bound);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def BCIR_GEMAttentionOp : BCIR_Op<"gem.attention", [Symbol]> {
  let summary = "Planned GEM attention: the K_BCIR-chosen two-matmul realization of single-head scaled-dot-product attention (G7)";
  // The law-rail record of the G7 attention plan (bcir/kbcir/attention.py::plan_attention). Single-head
  // scaled-dot-product attention softmax(Q@K^T/sqrt(d_k))@V is NOT a new primitive: it DECOMPOSES into the
  // ops BCIR already has -- two gem.matmuls with a gem.activation softmax between them:
  //     S  = Q @ K^T              (a gem.matmul: seq_len x d_k  times  d_k x seq_len -> seq x seq)
  //     S' = S / sqrt(d_k)        (an exact elementwise scale, 0 ULP, on the deterministic rail)
  //     A  = softmax(S', axis=seq)(the EXISTING gem.activation softmax -- reused, NOT reinvented)
  //     out= A @ V                (a second gem.matmul: seq x seq  times  seq x d_k -> seq x d_k)
  // So its REALIZATION -- the tile/loop plan of each underlying matmul -- is a FREE choice the dual-semiring
  // cost search picks (min,+ over the path; max,+ for the roofline bottleneck), exactly as gem.matmul. The
  // attention is PRICED THROUGH THE EXISTING matmul roofline for BOTH matmuls (NO bespoke attention term):
  // its (compute, mem) is the SUM of the two priced matmul costs (the O(seq^2) scale/softmax passes are
  // dominated by, and folded into, the Q@K^T traffic).
  //
  // The op carries the attention shape (seq_len, d_k), the DERIVED two gemm dims (scores Q@K^T = seq,seq,d_k;
  // context A@V = seq,d_k,seq -- carried for the verifier to cross-check), the K_BCIR-chosen tile/loop plan
  // of each gemm, the PER-GEMM priced roofline terms (scores_compute/mem, context_compute/mem), and the
  // SUMMED roofline (compute_cost = scores_compute + context_compute, mem_cost = scores_mem + context_mem,
  // bottleneck = max(compute_cost, mem_cost)), plus the quantized lane width (quant_bits > 0 ties to the A1
  // Q8 bridge: the R17 accuracy bound rides the accuracy axis; mirrors attention.cost_vector). `acc_bound`
  // records that R17 quantization bound (0 = dense).
  //
  // THE QUARANTINE BOUNDARY (the load-bearing G7 constraint, mirrored from attention.py): the softmax's exp
  // is a TRANSCENDENTAL that rides the trusted c.call.libm: edge (expf), fully OFF the deterministic legality
  // rail -- the same float story as gem.activation softmax. The two matmuls and the scale are exact. So the
  // ONLY certified error of a QUANTIZED attention is the Q8 bridge step (R17, 1 ULP), the matmul arithmetic
  // being exact and the libm kernel trusted; a dense attention carries acc_bound 0. The softmax forces an
  // f32 result (the libm edge returns float), never i32.
  //
  // hasVerifier enforces the op-level shape/dtype/realization laws check_attention enforces (mirrors
  // bcir/kbcir/attention.py::check_attention + the cost invariants -- the inline op-level check gem.matmul/
  // gem.activation/gem.conv run; NO new globally-numbered R-law): positive seq_len/d_k, a known + preserved
  // dtype that is f32 (the softmax quarantine rule), the DERIVED two gemm dims internally consistent, each
  // gemm tile in [1, gemm dim] with a known loop order, the per-gemm bottleneck == max-per-gemm, the SUMMED
  // compute/mem == the two per-gemm costs, bottleneck == max(compute, mem), and the R17 acc_bound.
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    // the attention shape (single head: d_model == d_k); Q/K/V/out are each seq_len x d_k.
    I64Attr:$seq_len, I64Attr:$d_k,
    StrAttr:$dtype,
    // the DERIVED two gemm dims (carried for the verifier to cross-check the decomposition):
    //   scores Q@K^T : (seq_len, seq_len, d_k);  context A@V : (seq_len, d_k, seq_len).
    I64Attr:$scores_m, I64Attr:$scores_n, I64Attr:$scores_k,
    I64Attr:$context_m, I64Attr:$context_n, I64Attr:$context_k,
    // the K_BCIR-chosen tile/loop plan of each underlying matmul.
    I64Attr:$scores_tile_m, I64Attr:$scores_tile_n, I64Attr:$scores_tile_k,
    StrAttr:$scores_loop_order,
    I64Attr:$context_tile_m, I64Attr:$context_tile_n, I64Attr:$context_tile_k,
    StrAttr:$context_loop_order,
    // the PER-GEMM priced roofline terms (each == matmul.cost_of at its chosen tile).
    I64Attr:$scores_compute, I64Attr:$scores_mem,
    I64Attr:$context_compute, I64Attr:$context_mem,
    // the SUMMED roofline (compute = sum of the two computes, mem = sum of the two mems) + the R17 axis.
    I64Attr:$compute_cost, I64Attr:$mem_cost, I64Attr:$bottleneck,
    DefaultValuedAttr<I32Attr, "0">:$quant_bits,
    DefaultValuedAttr<I64Attr, "0">:$acc_bound);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def BCIR_GEMFusedMatmulActivationOp : BCIR_Op<"gem.fused_matmul_activation", [Symbol]> {
  let summary = "Fused GEM matmul+activation epilogue: the activation applied INLINE per matmul output element (G2)";
  // The law-rail record of an ADOPTED matmul -> activation epilogue fusion (bcir/kbcir/fusion.py::
  // optimize_fused + FusionCertificate). A gem.matmul whose output feeds a SOLE-consumer gem.activation
  // is fused into ONE pass: the activation is applied inline as each matmul output element is produced, so
  // the intermediate C-buffer never materializes (deforestation). This op is what -bcir-fuse-matmul-
  // activation rewrites the adjacent pair into when fusion is the PRICED CHOICE (the fused score, with the
  // intermediate deforested, strictly beats the unfused score, with a phase barrier materializing it).
  //
  // It carries BOTH halves of the fused plan: the matmul tile (m/n/k + tile_m/n/k + loop_order, copied
  // verbatim off the gem.matmul) AND the activation epilogue (kind + dtype + the quarantine fields). It
  // also pins the deforestation-priced fusion DECISION as proof-carrying evidence (mirrors FusionCertificate):
  // unfused_score (the intermediate materialized at a barrier), fused_score (deforested), and gain =
  // unfused - fused (> 0 for an adopted fusion).
  //
  // THE QUARANTINE SPLIT carries THROUGH the fused epilogue, exactly as the standalone gem.activation:
  // relu is EXACT (inline max(0,x), 0 ULP, no libm, may be i32 or f32); the transcendentals (sigmoid/tanh/
  // gelu) route a libm call (expf/tanhf) through the trusted c.call.libm: edge and need f32. softmax is a
  // last-axis ROW REDUCTION -- it needs the whole row before any output element is final, so it can NOT be
  // an inline epilogue: hasVerifier REJECTS a fused softmax (the documented non-fusible kind), mirroring
  // fusion.is_fusible_activation. NO new globally-numbered R-law (G2 added none; this is the op-level
  // fusion-legality check the oracle rail runs).
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    // the matmul half (the B1 tile plan, copied verbatim off the producer gem.matmul).
    I64Attr:$m, I64Attr:$n, I64Attr:$k,
    I64Attr:$tile_m, I64Attr:$tile_n, I64Attr:$tile_k,
    StrAttr:$loop_order,
    // the activation epilogue half (the G1 quarantine fields).
    StrAttr:$kind, StrAttr:$dtype,
    // the deforestation-priced fusion DECISION (the FusionCertificate evidence).
    I64Attr:$unfused_score, I64Attr:$fused_score, I64Attr:$gain);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def BCIR_GEMMatmulBufferOp : BCIR_Op<"gem.matmul_buffer", []> {
  let summary = "Buffer-form GEM matmul: real SSA memref operands C += A*B with a tiling plan (toward executable code)";
  // The next step past gem.matmul (an attribute-only plan record): a NORMAL op carrying
  // three real rank-2 f32 memref operands -- A (MxK), B (KxN), C (MxN) accumulated IN PLACE --
  // plus the K_BCIR tile plan (tile_m/n/k + loop_order). M/N/K are DERIVED from the operands'
  // static shapes (not duplicated as attributes). -bcir-lower-gem-matmul-buffer lowers this to a
  // concrete TILED scf.for loop nest (memref.load/store + arith.mulf/addf) computing C += A*B.
  // hasVerifier checks: A/B/C are rank-2 f32 memrefs with static shapes; A.dim1==B.dim0 (K),
  // A.dim0==C.dim0 (M), B.dim1==C.dim1 (N); each tile extent in [1, its dim]; a known loop order.
  let arguments = (ins
    AnyMemRef:$A, AnyMemRef:$B, AnyMemRef:$C,
    I64Attr:$tile_m, I64Attr:$tile_n, I64Attr:$tile_k,
    StrAttr:$loop_order);
  let assemblyFormat = [{
    `tile` `(` $tile_m `,` $tile_n `,` $tile_k `)` `order` $loop_order attr-dict
    $A `,` $B `,` $C `:` type($A) `,` type($B) `,` type($C)
  }];
  let hasVerifier = 1;
}

def BCIR_GEMLaneSegmentOp : BCIR_Op<"gem.lane_segment", [Symbol]> {
  let summary = "Lane-specialized GEM executable segment";
  let arguments = (ins
    SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$claim, FlatSymbolRefAttr:$phase,
    BCIR_LaneAttr:$lane, DefaultValuedAttr<I32Attr, "1">:$stride_k, I32Attr:$width,
    StrAttr:$opcode, FlatSymbolRefArrayAttr:$reads, FlatSymbolRefArrayAttr:$writes,
    OptionalAttr<FlatSymbolRefAttr>:$prefetch,
    FlatSymbolRefArrayAttr:$fence_before, FlatSymbolRefArrayAttr:$fence_after,
    // CT2: thread->cache affinity domain (-1 = unpinned) and cache-unroll factor.
    DefaultValuedAttr<I32Attr, "-1">:$affinity,
    DefaultValuedAttr<I32Attr, "1">:$unroll,
    // Duration-aware scheduling: the K_BCIR step cost as execution duration.
    DefaultValuedAttr<I64Attr, "0">:$duration,
    // CIM/PIM dispatch target (append-only; mirrors gem.streampack.LaneSegment.dispatch).
    // "core" = execute on the compute core; "pim" = dispatch to the memory controller
    // (processing-in-memory). R14 (lower-to-llvm) legalizes: "pim" only on a reduction.
    DefaultValuedAttr<StrAttr, "\"core\"">:$dispatch,
    // Phase-aware DVFS clock (Q8; 256 = nominal x1.0; mirrors gem.dvfs). R15
    // (lower-to-llvm): must be a legal step in [64, 512] (0.25x..2x), and a "pim"
    // (memory-bound) segment must NOT overclock (clock_q8 <= 256).
    DefaultValuedAttr<I32Attr, "256">:$clock_q8);
  let assemblyFormat = "$sym_name attr-dict";
  // Parse-time well-formedness: the SIMD lane width is a positive power of two, and the
  // stride class multiplier is positive.
  let hasVerifier = 1;
}

#endif // BCIR_GEM_OPS
