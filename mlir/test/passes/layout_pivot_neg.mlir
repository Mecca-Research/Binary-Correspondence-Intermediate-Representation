// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// The gem.layout_pivot op verifier (hasVerifier; mirrors bcir/kbcir/layout.py::plan_layout + the
// LayoutCertificate cost invariants -- the op-level pricing well-formedness check gem.conv/attention
// validate inline, NOT a new globally-numbered R-law). Each section carries a single violation. The
// priced choice is INFORMS-ONLY (it informs the plan's memory cost but never gates legality; the hard
// "layout changes addresses, never values" invariant is the oracle's) -- these checks only enforce
// that the declared cost/layout/gain ARE the cost model's priced decision, not a legality verdict.

// the frozen geometry must be well-formed (each >= 1).
bcir.module @bad_geometry {
  // expected-error @+1 {{layout_pivot: frozen geometry cacheline/elem_bytes/base_overhead/streams/vector_width/gather_penalty must be >= 1}}
  bcir.gem.layout_pivot @bad { rid = 1 : i64, fields = 4 : i64,
                               single_field_records = 8 : i64, whole_record_records = 64 : i64,
                               cacheline = 64 : i64, elem_bytes = 0 : i64, base_overhead = 4 : i64,
                               streams = 3 : i64, vector_width = 16 : i64, gather_penalty = 32 : i64,
                               soa_cost = 3300 : i64, aos_cost = 648 : i64, layout = "aos", gain = 2652 : i64 }
}

// -----

// the priced SoA cost must equal the recomputed stride-penalty memory cost (3300 for this workload).
bcir.module @bad_soa_cost {
  // expected-error @+1 {{layout_pivot: soa_cost must equal the priced SoA workload memory cost = 3300 (got 999)}}
  bcir.gem.layout_pivot @bad { rid = 2 : i64, fields = 4 : i64,
                               single_field_records = 8 : i64, whole_record_records = 64 : i64,
                               cacheline = 64 : i64, elem_bytes = 4 : i64, base_overhead = 4 : i64,
                               streams = 3 : i64, vector_width = 16 : i64, gather_penalty = 32 : i64,
                               soa_cost = 999 : i64, aos_cost = 648 : i64, layout = "aos", gain = 2652 : i64 }
}

// -----

// the priced AoS cost must equal the recomputed stride-penalty memory cost (648 for this workload).
bcir.module @bad_aos_cost {
  // expected-error @+1 {{layout_pivot: aos_cost must equal the priced AoS workload memory cost = 648 (got 1)}}
  bcir.gem.layout_pivot @bad { rid = 3 : i64, fields = 4 : i64,
                               single_field_records = 8 : i64, whole_record_records = 64 : i64,
                               cacheline = 64 : i64, elem_bytes = 4 : i64, base_overhead = 4 : i64,
                               streams = 3 : i64, vector_width = 16 : i64, gather_penalty = 32 : i64,
                               soa_cost = 3300 : i64, aos_cost = 1 : i64, layout = "aos", gain = 3299 : i64 }
}

// -----

// the layout must be the MIN-cost choice: AoS prices strictly cheaper here, so declaring "soa" is rejected.
bcir.module @bad_layout_choice {
  // expected-error @+1 {{layout_pivot: layout must be the min-cost choice 'aos' (ties keep the default soa; soa_cost 3300 vs aos_cost 648), got 'soa'}}
  bcir.gem.layout_pivot @bad { rid = 4 : i64, fields = 4 : i64,
                               single_field_records = 8 : i64, whole_record_records = 64 : i64,
                               cacheline = 64 : i64, elem_bytes = 4 : i64, base_overhead = 4 : i64,
                               streams = 3 : i64, vector_width = 16 : i64, gather_penalty = 32 : i64,
                               soa_cost = 3300 : i64, aos_cost = 648 : i64, layout = "soa", gain = 0 : i64 }
}

// -----

// the gain must equal soa_cost - chosen_cost (3300 - 648 = 2652 for this AoS pivot).
bcir.module @bad_gain {
  // expected-error @+1 {{layout_pivot: gain must equal soa_cost - chosen_cost = 2652 (got 0)}}
  bcir.gem.layout_pivot @bad { rid = 5 : i64, fields = 4 : i64,
                               single_field_records = 8 : i64, whole_record_records = 64 : i64,
                               cacheline = 64 : i64, elem_bytes = 4 : i64, base_overhead = 4 : i64,
                               streams = 3 : i64, vector_width = 16 : i64, gather_penalty = 32 : i64,
                               soa_cost = 3300 : i64, aos_cost = 648 : i64, layout = "aos", gain = 0 : i64 }
}
