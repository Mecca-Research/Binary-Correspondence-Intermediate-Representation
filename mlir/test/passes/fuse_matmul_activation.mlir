// RUN: bcir-opt -bcir-fuse-matmul-activation %s | FileCheck %s
//
// The -bcir-fuse-matmul-activation fusion (the G2 port of bcir/kbcir/fusion.py::optimize_fused):
// a gem.matmul whose output feeds a SOLE-CONSUMER gem.activation is fused into a single inline
// epilogue -- the activation is applied as each matmul output element is produced, so the
// intermediate C-buffer never round-trips through memory (deforestation). On the law rail the
// producer->consumer epilogue is ADJACENCY: a gem.matmul whose IMMEDIATELY following op is a
// gem.activation is the pair; an op reading the un-activated intermediate would break the
// adjacency (the sole-consumer legality). Both ops are ERASED into one gem.fused_matmul_activation.
//
// THE PRICED CHOICE (mirrors optimize_fused exactly): fusion is adopted iff the DEFORESTATION-priced
// fused score strictly beats the unfused score. We reuse the EXISTING discount (realize._DEFOREST_
// FACTOR == x0.75 memory, 192/256), NOT a bespoke one:
//   unfused_score = matmul.bottleneck + activation.bottleneck            (two passes; barrier materializes)
//   fused_score   = matmul.bottleneck + max(act.compute, act.mem*192/256) (one inline pass; deforested)
//   ADOPT iff fused_score < unfused_score (a strict win)
//
// THE QUARANTINE SPLIT carries through: relu is exact (i32 or f32, no libm); the transcendentals
// (sigmoid/tanh/gelu) need f32 (the libm edge returns float). softmax is SCOPED OUT (a row reduction
// is not an inline epilogue) -- see fuse_matmul_activation_neg.mlir for the verifier rejection.

bcir.module @epilogues {
  // (1) matmul -> relu (the EXACT epilogue). matmul bottleneck 200; relu mem 32 -> deforested
  // 32*192/256 = 24, fused act bottleneck = max(compute 2, 24) = 24. unfused = 200+32 = 232,
  // fused = 200+24 = 224, gain = 8 > 0 -> ADOPTED. relu may be f32 (or i32, the clean rail).
  bcir.gem.matmul @mm0 { m = 4 : i64, n = 4 : i64, k = 4 : i64,
                         tile_m = 2 : i64, tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk",
                         compute_cost = 100 : i64, mem_cost = 200 : i64, bottleneck = 200 : i64,
                         quant_bits = 0 : i32 }
  bcir.gem.activation @relu0 { kind = "relu", shape = array<i64: 16>, dtype = "f32",
                              axis_len = 0 : i64, width = 16 : i64,
                              compute_cost = 2 : i64, mem_cost = 32 : i64, bottleneck = 32 : i64,
                              quant_bits = 0 : i32, acc_bound = 0 : i64 }

  // (2) matmul -> gelu (a TRANSCENDENTAL epilogue, the tanhf libm edge, needs f32). Same matmul
  // roofline; gelu compute 24, mem 32 -> deforested 24, fused act bottleneck = max(24, 24) = 24.
  // unfused = 232, fused = 224, gain = 8 > 0 -> ADOPTED. The quarantine (f32) carries through.
  bcir.gem.matmul @mm1 { m = 4 : i64, n = 4 : i64, k = 4 : i64,
                         tile_m = 2 : i64, tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk",
                         compute_cost = 100 : i64, mem_cost = 200 : i64, bottleneck = 200 : i64,
                         quant_bits = 0 : i32 }
  bcir.gem.activation @gelu0 { kind = "gelu", shape = array<i64: 16>, dtype = "f32",
                              axis_len = 0 : i64, width = 16 : i64,
                              compute_cost = 24 : i64, mem_cost = 32 : i64, bottleneck = 32 : i64,
                              quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// Both adopted pairs are FUSED: the matmul + the activation are gone, one fused op in their place.
// CHECK-NOT: bcir.gem.matmul
// CHECK-NOT: bcir.gem.activation

// (1) relu epilogue: the matmul tile half copied verbatim (m/n/k 4, tiles 2, loop_order ijk), the
// activation half (kind relu, dtype f32), and the FusionCertificate evidence (unfused 232, fused 224,
// gain 8). The fused op is named <matmul>_<activation>_fused.
// CHECK: bcir.gem.fused_matmul_activation @mm0_relu0_fused {dtype = "f32", fused_score = 224 : i64, gain = 8 : i64, k = 4 : i64, kind = "relu", loop_order = "ijk", m = 4 : i64, n = 4 : i64, tile_k = 2 : i64, tile_m = 2 : i64, tile_n = 2 : i64, unfused_score = 232 : i64}

// (2) gelu epilogue: kind gelu, dtype f32 (the quarantine carries through), same deforestation win.
// CHECK: bcir.gem.fused_matmul_activation @mm1_gelu0_fused {dtype = "f32", fused_score = 224 : i64, gain = 8 : i64, k = 4 : i64, kind = "gelu", loop_order = "ijk", m = 4 : i64, n = 4 : i64, tile_k = 2 : i64, tile_m = 2 : i64, tile_n = 2 : i64, unfused_score = 232 : i64}

// -----------------------------------------------------------------------------------------------
// Clean NO-OPs (fusion is the optimizer's priced choice, not a forced rewrite): a non-sole-consumer
// pair (a third op between producer and consumer breaks the adjacency) and a softmax epilogue (a row
// reduction, scoped out) are LEFT UNTOUCHED -- both ops survive, no fused op emitted.

bcir.module @noops {
  // non-sole-consumer: a gem.block sits between the matmul and the relu (it would read the
  // un-activated intermediate), so the relu is NOT the immediate consumer -> NO fusion.
  bcir.gem.matmul @nm0 { m = 4 : i64, n = 4 : i64, k = 4 : i64,
                         tile_m = 2 : i64, tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk",
                         compute_cost = 100 : i64, mem_cost = 200 : i64, bottleneck = 200 : i64,
                         quant_bits = 0 : i32 }
  bcir.gem.block @reader { base = 0 : i64, count = 4 : i64, strideA = 1 : i64, strideB = 0 : i64, strideD = 1 : i64 }
  bcir.gem.activation @nrelu { kind = "relu", shape = array<i64: 16>, dtype = "f32",
                              axis_len = 0 : i64, width = 16 : i64,
                              compute_cost = 2 : i64, mem_cost = 32 : i64, bottleneck = 32 : i64,
                              quant_bits = 0 : i32, acc_bound = 0 : i64 }

  // softmax epilogue: a last-axis row reduction, scoped out of inline fusion -> NO fusion.
  bcir.gem.matmul @sm_mm { m = 4 : i64, n = 4 : i64, k = 4 : i64,
                          tile_m = 2 : i64, tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk",
                          compute_cost = 100 : i64, mem_cost = 200 : i64, bottleneck = 200 : i64,
                          quant_bits = 0 : i32 }
  bcir.gem.activation @sm0 { kind = "softmax", shape = array<i64: 4, 4>, dtype = "f32",
                            axis_len = 4 : i64, width = 4 : i64,
                            compute_cost = 8 : i64, mem_cost = 24 : i64, bottleneck = 24 : i64,
                            quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// The no-op module fuses NOTHING: both matmuls + both activations survive, no fused op emitted.
// CHECK-LABEL: bcir.module @noops
// CHECK: bcir.gem.matmul @nm0
// CHECK: bcir.gem.block @reader
// CHECK: bcir.gem.activation @nrelu
// CHECK: bcir.gem.matmul @sm_mm
// CHECK: bcir.gem.activation @sm0
// CHECK-NOT: bcir.gem.fused_matmul_activation
