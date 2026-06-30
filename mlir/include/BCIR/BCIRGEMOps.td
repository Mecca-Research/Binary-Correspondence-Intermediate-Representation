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

def BCIR_GEMContentionOp : BCIR_Op<"gem.contention", [Symbol]> {
  let summary = "Cache-line/bank-conflict CONTENTION cost signal: the frozen analytic predictor for one access shape (G4)";
  // The law-rail record + producer of the G4 cache-line-utilization / memory-bank-conflict cost SIGNAL
  // (port of bcir/kbcir/cache_predict.py::CachePredictor.contention_q8). An access touches `count` elements
  // of `elem_bytes` each at element `stride` (1 == unit/contiguous; large == gather-like). From that shape +
  // the FROZEN cache/bank geometry (`cacheline` bytes per line; `banks` == TargetProfile.mem_channels) the
  // analytic predictor computes a Q8 CONTENTION signal -- the cache-line-utilization WASTE plus the
  // bank-conflict serialization degree -- and the integer CONTENTION cost (the per-element Q8 scaled by
  // `count`). -bcir-cache-contention RECOMPUTES that signal from the shape + the frozen params and annotates
  // the SAME line_waste_q8 / bank_conflict_q8 / contention_q8 / contention_cost on the op (the cost annotation).
  //
  // THE TWO-TRUTH / QUARANTINE CONSTRAINT (the non-negotiable G4 discipline, mirrored from cache_predict.py):
  // a predicted cache/bank cost is a GRADED signal. It rides the CONTENTION cost axis ONLY -- it may INFORM
  // which realization is cheapest (rank plans) but it MUST NEVER change a legality verdict and MUST NEVER gate
  // legality. The R1-R16 verifier reads only R8/R9 (lane legality + 12-d cost completeness/non-negativity) and
  // NEVER reads CONTENTION, so a conflict-prone plan and a conflict-free plan are EXACTLY as legal; this op's
  // signal only moves the CONTENTION dimension. It carries NOTHING on the legality path. `gen` is the R13-style
  // provenance generation (0 = the seeded analytic constants).
  //
  // hasVerifier checks the op is internally consistent with the frozen analytic model (the same recompute the
  // pass cross-checks; NO new globally-numbered R-law): positive shape (stride/elem_bytes/count >= 1, cacheline/
  // banks >= 1), line_waste_q8 == (min(stride, cacheline/elem_bytes) - 1) * 256, bank_conflict_q8 ==
  // (gcd(stride, banks) - 1) * 256, contention_q8 == waste + conflict (unit weights), and contention_cost ==
  // (contention_q8 * count) >> 8. A unit-stride sweep is the clean NO-OP (waste 0, conflict 0, cost 0).
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    // the access SHAPE the predictor prices (read off a claim's declared geometry).
    I64Attr:$stride, I64Attr:$elem_bytes, I64Attr:$count,
    // the FROZEN cache/bank geometry (cacheline bytes; banks == mem_channels).
    I64Attr:$cacheline, I64Attr:$banks,
    // the recomputed CONTENTION signal: the Q8 terms + the integer cost (on the CONTENTION axis ONLY).
    I64Attr:$line_waste_q8, I64Attr:$bank_conflict_q8, I64Attr:$contention_q8, I64Attr:$contention_cost,
    DefaultValuedAttr<I32Attr, "0">:$gen);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def BCIR_GEMLayoutPivotOp : BCIR_Op<"gem.layout_pivot", [Symbol]> {
  let summary = "SoA<->AoS layout-pivot pricing: the cost-model-priced layout choice for a multi-field resource (G3)";
  // The law-rail record of the G3 SoA<->AoS layout-pivot PRICING DECISION (port of bcir/kbcir/layout.py::
  // plan_layout + LayoutCertificate). A multi-field (F-field) resource can live as SoA (one contiguous array
  // per field) or AoS (records interleaved). A single-field SWEEP is UNIT under SoA / STRIDED(stride=F) under
  // AoS; a whole-record access is the exact mirror -- so the two access shapes have MIRRORED costs and which
  // layout wins is PRICED, not asserted. -bcir-layout-pivot recomputes the SoA-vs-AoS workload cost through
  // the SAME stride-penalty memory terms realize.candidates_for prices every claim by (a UNIT access pays
  // base_overhead*1 + vectorizes; a STRIDED access pays base_overhead*min(F, cacheline/elem) and is pinned
  // scalar), picks the MIN-cost layout (ties keep the declared default SoA), and annotates the chosen layout +
  // the SoA/AoS cost + the gain (a LayoutCertificate-equivalent).
  //
  // THE HARD INVARIANT is the oracle's (layout changes ADDRESSES, never VALUES -- proved bit-exact by
  // lower.layout_kernel); the law records only the PRICED choice. Like G4, this is INFORMS-ONLY: the priced
  // layout/cost informs the plan but the R1-R16 verifier never reads it as a legality verdict (it is the
  // memory-cost lever, off the legality path -- the same two-truth quarantine).
  //
  // The op carries the resource id, the record width F, the per-shape access workload reduced to two summed
  // counts (`single_field_records` = records swept single-field, `whole_record_records` = records accessed
  // whole-record) over the frozen memory geometry (cacheline/elem_bytes/base_overhead/streams/vector_width --
  // the pinned cost constants), the priced soa_cost/aos_cost, the chosen layout, and the gain. hasVerifier
  // recomputes the two costs through the closed-form stride-penalty memory terms and checks the op is
  // internally consistent (the chosen layout is the min-cost one, ties keep SoA, gain >= 0, F <= 1 is a
  // clean no-op). NO new globally-numbered R-law (the op-level pricing check gem.conv/attention run inline).
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$rid, I64Attr:$fields,
    // the access workload reduced to the two mirrored shapes (records swept under each).
    I64Attr:$single_field_records, I64Attr:$whole_record_records,
    // the FROZEN memory geometry the stride-penalty terms price over (the pinned cost constants).
    // gather_penalty is the random-gather access overhead -- a STRIDED claim's cheapest candidate is the
    // min of its `strided` (penalty min(F, cacheline/elem)) and `gather` (penalty gather_penalty) realizations,
    // so a target whose gather is cheaper than a large stride (e.g. PTX gp=16) prices the gather instead.
    I64Attr:$cacheline, I64Attr:$elem_bytes, I64Attr:$base_overhead, I64Attr:$streams, I64Attr:$vector_width,
    I64Attr:$gather_penalty,
    // the priced SoA-vs-AoS workload cost + the chosen layout + the modeled win.
    I64Attr:$soa_cost, I64Attr:$aos_cost,
    StrAttr:$layout, I64Attr:$gain);
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

def BCIR_GEMAutodiffOp : BCIR_Op<"gem.autodiff", [Symbol]> {
  let summary = "Serialized closed-set forward autodiff DAG (the B3 content-addressed differentiable graph as a law object)";
  // The law-rail record of the B3 reverse-mode-AD oracle (bcir/kbcir/autodiff.py). Slice 5 PROVED that B3's
  // differentiable expression DAG is a CLOSED SET, content-addressed (hash-consed) DAG over a FIXED vocabulary
  // {const,var,neg,add,sub,mul,div,dot,select} (autodiff.CLOSED_SET / _ARITY / differentiable_ops): differentiating
  // any expression in that set produces a gradient DAG that STAYS in the same set, no foreign op kind ever appears.
  // That closure is exactly what lets the law rail carry the forward autodiff DAG as a FIRST-CLASS op with a
  // fixed-vocabulary structural verifier. This op SERIALIZES one such forward Tape (the hash-consed Node list) so it
  // round-trips through bcir-opt and is checked against the closed vocabulary + per-op arity + DAG index bounds.
  //
  // THE ENCODING (parallel per-node arrays, mirroring how gem.activation carries DenseI64ArrayAttr:$shape; the node
  // count is opcodes.size()). A Tape Node (nid, op, args, const) is serialized positionally -- node i is described by
  // the i-th entry of each array, and operands reference STRICTLY EARLIER nodes (the Tape interns operands before
  // their users, so the array is already topological/acyclic):
  //   * n_inputs   -- the number of `var` leaves (function inputs); a var's slot must be < n_inputs.
  //   * opcodes    -- per node, the op kind as a fixed int code (the closed vocabulary):
  //                     0=const 1=var 2=neg 3=add 4=sub 5=mul 6=div 7=dot 8=select  (mirrors autodiff._ARITY order).
  //   * arities    -- per node, its operand count. Fixed for every op (neg=1; add/sub/mul/div/dot... see _ARITY) EXCEPT
  //                   `dot`, which is VARIADIC (2k operands: k u-ids then k v-ids), so the arity is carried per node.
  //   * arg_base   -- per node, the offset into the flat `args` array where this node's operands begin.
  //   * args       -- the FLATTENED operand node-indices: node i's operands are args[arg_base[i] : arg_base[i]+arities[i]].
  //   * consts     -- per node a payload (integer, matching the dialect's all-integer attr idiom): for a `const` node its
  //                   integer/Q-fixed value; for a `var` node its input SLOT in [0, n_inputs); a sentinel -1 for every
  //                   non-leaf op (which carries no payload). (B3's oracle keeps float const values; the law serializes
  //                   the STRUCTURE -- op/operands/leaf-slot -- and carries the const as an integer payload. See the
  //                   caveat: a float const value is out of scope for this structural law object.)
  //   * output     -- the result node index (the function's value), in [0, node_count).
  //
  // This is a LAW OBJECT: the DAG is fully serialized in attrs (like a constant/declarative op), so it carries NO SSA
  // operands -- exactly like the sibling attribute-carrying gem plan ops (gem.matmul/activation/conv). No lowering and
  // no cost pass yet (Phase 2a is the op + its closed-set verifier + round-trip; the lowering/cost pass is the next
  // slice). hasVerifier enforces the CLOSED-SET LAW (the slice's core value), NOT a new globally-numbered R-law -- it is
  // the structural well-formedness check the sibling gem ops run inline: (1) every opcode is in the closed vocabulary;
  // (2) each node's arity matches _ARITY for its op (the variadic `dot` carries an even 2k arity); (3) every operand
  // index references a STRICTLY EARLIER node (< the node's own position -> acyclic + topological), a var's slot is in
  // [0, n_inputs), and `output` is a valid node index; (4) per-node consistency (a const/var leaf carries a payload, a
  // non-leaf carries the -1 sentinel; the parallel arrays are equal length and `args` has exactly sum(arities) entries).
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$n_inputs,
    DenseI64ArrayAttr:$opcodes,
    DenseI64ArrayAttr:$arities,
    DenseI64ArrayAttr:$arg_base,
    DenseI64ArrayAttr:$args,
    DenseI64ArrayAttr:$consts,
    I64Attr:$output);
  let assemblyFormat = "$sym_name attr-dict";
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
