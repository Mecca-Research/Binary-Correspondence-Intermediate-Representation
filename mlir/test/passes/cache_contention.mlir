// RUN: bcir-opt -bcir-cache-contention %s | FileCheck %s
//
// The -bcir-cache-contention cost producer: a gem.contention carrier op (the access SHAPE
// stride/elem_bytes/count + the FROZEN cache/bank geometry) has its G4 cache-line/bank-conflict
// CONTENTION signal RECOMPUTED (port of bcir/kbcir/cache_predict.py::CachePredictor.contention_q8)
// and annotated as kbcir.* attrs. Unlike the lower-gem-* passes this is a COST PRODUCER, not a
// lowering -- the op is NOT erased.
//
// This is the DUAL-RAIL PARITY gate (the law reproduces the oracle's cache_predict.py signal):
//   line_waste_q8    = (min(stride, cacheline/elem_bytes) - 1) * 256   (0 == perfect utilization)
//   bank_conflict_q8 = (gcd(stride, banks) - 1)            * 256        (0 == reaches all banks)
//   contention_q8    = waste + conflict                                 (unit Q8 weights)
//   contention_cost  = (contention_q8 * count) >> 8                     (per-element Q8 * count)
// Re-asserted host-agnostically in bcir/tests/test_cache_contention_law_parity.py (the frozen
// predictor is pinned: cacheline 64, banks 4).
//
// THE TWO-TRUTH / QUARANTINE: the signal is INFORMS-ONLY -- it rides the CONTENTION cost axis and
// NEVER gates legality (the R1-R16 verifier reads only R8/R9, never CONTENTION). The pass annotates
// kbcir.informs_only = true + kbcir.contention_axis = "CONTENTION" to record that.

bcir.module @contentions {
  // (1) a UNIT sweep (stride 1): full cache-line utilization, all banks reached -> the clean NO-OP.
  // epl = 64/4 = 16, reach = min(1, 16) = 1 -> waste (1-1)*256 = 0; gcd(1, 4) = 1 -> conflict (1-1)*256 = 0.
  // contention_q8 = 0, contention_cost = 0. conflict_free + noop.
  bcir.gem.contention @unit { stride = 1 : i64, elem_bytes = 4 : i64, count = 64 : i64,
                              cacheline = 64 : i64, banks = 4 : i64,
                              line_waste_q8 = 0 : i64, bank_conflict_q8 = 0 : i64,
                              contention_q8 = 0 : i64, contention_cost = 0 : i64 }

  // (2) a bank-colliding stride (stride 4 == #banks): gcd(4, 4) = 4 -> conflict (4-1)*256 = 768; reach =
  // min(4, 16) = 4 -> waste (4-1)*256 = 768. contention_q8 = 1536, cost = (1536*64)>>8 = 384. NOT conflict-free.
  bcir.gem.contention @bank { stride = 4 : i64, elem_bytes = 4 : i64, count = 64 : i64,
                              cacheline = 64 : i64, banks = 4 : i64,
                              line_waste_q8 = 768 : i64, bank_conflict_q8 = 768 : i64,
                              contention_q8 = 1536 : i64, contention_cost = 384 : i64 }

  // (3) a strided gather (stride 16 == cacheline/elem -> a fresh line per element): reach = min(16, 16) = 16
  // -> waste (16-1)*256 = 3840 (max); gcd(16, 4) = 4 -> conflict 768. contention_q8 = 4608, cost =
  // (4608*64)>>8 = 1152. The worst-case utilization a gather pays.
  bcir.gem.contention @gather { stride = 16 : i64, elem_bytes = 4 : i64, count = 64 : i64,
                                cacheline = 64 : i64, banks = 4 : i64,
                                line_waste_q8 = 3840 : i64, bank_conflict_q8 = 768 : i64,
                                contention_q8 = 4608 : i64, contention_cost = 1152 : i64 }
}

// The carrier ops are NOT erased (a cost producer, not a lowering).
// CHECK: bcir.gem.contention @unit
// (1) the unit sweep: the clean no-op -- 0 waste, 0 conflict, 0 cost, conflict_free + noop.
// CHECK-SAME: kbcir.bank_conflict_q8 = 0 : i64
// CHECK-SAME: kbcir.conflict_free = true
// CHECK-SAME: kbcir.contention_axis = "CONTENTION"
// CHECK-SAME: kbcir.contention_cost = 0 : i64
// CHECK-SAME: kbcir.contention_q8 = 0 : i64
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.line_waste_q8 = 0 : i64
// CHECK-SAME: kbcir.noop = true

// (2) the bank-colliding stride: waste 768, conflict 768, contention 1536, cost 384, NOT conflict-free.
// CHECK: bcir.gem.contention @bank
// CHECK-SAME: kbcir.bank_conflict_q8 = 768 : i64
// CHECK-SAME: kbcir.conflict_free = false
// CHECK-SAME: kbcir.contention_cost = 384 : i64
// CHECK-SAME: kbcir.contention_q8 = 1536 : i64
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.line_waste_q8 = 768 : i64
// CHECK-SAME: kbcir.noop = false

// (3) the strided gather: max waste 3840, conflict 768, contention 4608, cost 1152.
// CHECK: bcir.gem.contention @gather
// CHECK-SAME: kbcir.bank_conflict_q8 = 768 : i64
// CHECK-SAME: kbcir.contention_cost = 1152 : i64
// CHECK-SAME: kbcir.contention_q8 = 4608 : i64
// CHECK-SAME: kbcir.line_waste_q8 = 3840 : i64
// CHECK-SAME: kbcir.noop = false
