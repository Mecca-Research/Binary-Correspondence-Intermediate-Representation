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
