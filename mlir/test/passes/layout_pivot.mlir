// RUN: bcir-opt -bcir-layout-pivot %s | FileCheck %s
//
// The -bcir-layout-pivot cost producer: a gem.layout_pivot carrier op (a multi-field resource's
// access workload reduced to its two mirrored shapes + the frozen memory geometry) has its G3
// SoA<->AoS PRICING DECISION RECOMPUTED (port of bcir/kbcir/layout.py::plan_layout) and annotated
// as kbcir.* attrs (a LayoutCertificate-equivalent). Unlike the lower-gem-* passes this is a COST
// PRODUCER, not a lowering -- the op is NOT erased.
//
// This is the DUAL-RAIL PARITY gate (the law reproduces the oracle's layout.py choice). Each shape
// is priced THROUGH THE SAME stride-penalty memory terms realize.candidates_for prices every claim by:
//   UNIT (contiguous) min-mem = n*streams + ceil(n/vector_width)*streams*base_overhead*1
//   STRIDED (step F)  min-mem = n*streams + min(n*streams*base_overhead*min(F,cacheline/elem),
//                                                n*streams*base_overhead*gather_penalty)
// A single-field sweep is UNIT under SoA / STRIDED under AoS; a whole-record access is the mirror.
// The MIN-cost layout wins; AoS only on a STRICT win (ties keep the declared default SoA); gain =
// soa_cost - chosen_cost. Re-asserted host-agnostically in bcir/tests/test_layout_pivot_law_parity.py
// (the oracle target pinned to x86-avx512: cacheline 64, elem 4, base_overhead 4, vector_width 16,
// gather_penalty 32, streams 3).
//
// THE HARD INVARIANT (layout changes ADDRESSES, never VALUES) is the oracle's (lower.layout_kernel
// proves SoA result == AoS result, bit-exact); the law records only the PRICED choice. INFORMS-ONLY:
// the priced layout/cost informs the plan's memory cost but NEVER gates legality (kbcir.informs_only).

bcir.module @pivots {
  // (1) a single-field-dominated workload (64 single-field sweeps, 8 whole-record) over F=4: SoA prices
  // 648 <= AoS 3300 -> keep SoA, gain 0 (the clean no-op).
  bcir.gem.layout_pivot @sf_dom { rid = 1 : i64, fields = 4 : i64,
                                  single_field_records = 64 : i64, whole_record_records = 8 : i64,
                                  cacheline = 64 : i64, elem_bytes = 4 : i64, base_overhead = 4 : i64,
                                  streams = 3 : i64, vector_width = 16 : i64, gather_penalty = 32 : i64,
                                  soa_cost = 648 : i64, aos_cost = 3300 : i64, layout = "soa", gain = 0 : i64 }

  // (2) a whole-record-dominated workload (8 single-field, 64 whole-record) over F=4: AoS prices 648 <
  // SoA 3300 -> PIVOT to AoS with gain 3300 - 648 = 2652 (a strict win).
  bcir.gem.layout_pivot @wr_dom { rid = 2 : i64, fields = 4 : i64,
                                  single_field_records = 8 : i64, whole_record_records = 64 : i64,
                                  cacheline = 64 : i64, elem_bytes = 4 : i64, base_overhead = 4 : i64,
                                  streams = 3 : i64, vector_width = 16 : i64, gather_penalty = 32 : i64,
                                  soa_cost = 3300 : i64, aos_cost = 648 : i64, layout = "aos", gain = 2652 : i64 }

  // (3) a one-field resource (F=1): no SoA/AoS distinction -> both costs equal (240), keep SoA, gain 0.
  bcir.gem.layout_pivot @one_field { rid = 3 : i64, fields = 1 : i64,
                                     single_field_records = 64 : i64, whole_record_records = 0 : i64,
                                     cacheline = 64 : i64, elem_bytes = 4 : i64, base_overhead = 4 : i64,
                                     streams = 3 : i64, vector_width = 16 : i64, gather_penalty = 32 : i64,
                                     soa_cost = 240 : i64, aos_cost = 240 : i64, layout = "soa", gain = 0 : i64 }

  // (4) a balanced workload (32 single-field, 32 whole-record) over F=4: SoA == AoS (1752) -> the TIE keeps
  // the declared default SoA, gain 0.
  bcir.gem.layout_pivot @balanced { rid = 4 : i64, fields = 4 : i64,
                                    single_field_records = 32 : i64, whole_record_records = 32 : i64,
                                    cacheline = 64 : i64, elem_bytes = 4 : i64, base_overhead = 4 : i64,
                                    streams = 3 : i64, vector_width = 16 : i64, gather_penalty = 32 : i64,
                                    soa_cost = 1752 : i64, aos_cost = 1752 : i64, layout = "soa", gain = 0 : i64 }
}

// The carrier ops are NOT erased (a cost producer, not a lowering).
// (1) single-field-dominated: keep SoA, gain 0, noop.
// CHECK: bcir.gem.layout_pivot @sf_dom
// CHECK-SAME: kbcir.aos_cost = 3300 : i64
// CHECK-SAME: kbcir.chosen_cost = 648 : i64
// CHECK-SAME: kbcir.declared_layout = "soa"
// CHECK-SAME: kbcir.gain = 0 : i64
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.layout = "soa"
// CHECK-SAME: kbcir.noop = true
// CHECK-SAME: kbcir.soa_cost = 648 : i64

// (2) whole-record-dominated: PIVOT to AoS, chosen 648, gain 2652, not a no-op.
// CHECK: bcir.gem.layout_pivot @wr_dom
// CHECK-SAME: kbcir.aos_cost = 648 : i64
// CHECK-SAME: kbcir.chosen_cost = 648 : i64
// CHECK-SAME: kbcir.gain = 2652 : i64
// CHECK-SAME: kbcir.layout = "aos"
// CHECK-SAME: kbcir.noop = false
// CHECK-SAME: kbcir.soa_cost = 3300 : i64

// (3) one-field: equal costs, keep SoA, no-op.
// CHECK: bcir.gem.layout_pivot @one_field
// CHECK-SAME: kbcir.aos_cost = 240 : i64
// CHECK-SAME: kbcir.gain = 0 : i64
// CHECK-SAME: kbcir.layout = "soa"
// CHECK-SAME: kbcir.noop = true
// CHECK-SAME: kbcir.soa_cost = 240 : i64

// (4) balanced tie: SoA == AoS, the tie keeps SoA, no-op.
// CHECK: bcir.gem.layout_pivot @balanced
// CHECK-SAME: kbcir.aos_cost = 1752 : i64
// CHECK-SAME: kbcir.gain = 0 : i64
// CHECK-SAME: kbcir.layout = "soa"
// CHECK-SAME: kbcir.noop = true
// CHECK-SAME: kbcir.soa_cost = 1752 : i64
