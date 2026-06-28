// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// The gem.contention op verifier (hasVerifier; mirrors bcir/kbcir/cache_predict.py::CachePredictor --
// the op-level analytic-model well-formedness check gem.conv/attention validate inline, NOT a new
// globally-numbered R-law). Each section carries a single violation. The signal is INFORMS-ONLY (it
// rides the CONTENTION cost axis and never gates legality), so these checks only enforce that the
// declared Q8 terms ARE the frozen analytic model's terms -- consistency, not a legality verdict.

// positive access shape: stride/elem_bytes/count must be >= 1.
bcir.module @bad_shape {
  // expected-error @+1 {{contention: access shape stride/elem_bytes/count must be >= 1 (got stride 0, elem_bytes 4, count 64)}}
  bcir.gem.contention @bad { stride = 0 : i64, elem_bytes = 4 : i64, count = 64 : i64,
                             cacheline = 64 : i64, banks = 4 : i64,
                             line_waste_q8 = 0 : i64, bank_conflict_q8 = 0 : i64,
                             contention_q8 = 0 : i64, contention_cost = 0 : i64 }
}

// -----

// the cache-line-utilization WASTE: line_waste_q8 must equal (min(stride, cacheline/elem_bytes)-1)*256.
// stride 16, epl = 64/4 = 16 -> reach 16 -> waste 3840; declaring 999 is rejected.
bcir.module @bad_waste {
  // expected-error @+1 {{contention: line_waste_q8 must equal (min(stride, cacheline/elem_bytes) - 1)*256 = 3840 (got 999)}}
  bcir.gem.contention @bad { stride = 16 : i64, elem_bytes = 4 : i64, count = 64 : i64,
                             cacheline = 64 : i64, banks = 4 : i64,
                             line_waste_q8 = 999 : i64, bank_conflict_q8 = 768 : i64,
                             contention_q8 = 1767 : i64, contention_cost = 441 : i64 }
}

// -----

// the bank-conflict degree: bank_conflict_q8 must equal (gcd(stride, banks)-1)*256.
// stride 4, banks 4 -> gcd 4 -> conflict 768; declaring 0 is rejected.
bcir.module @bad_conflict {
  // expected-error @+1 {{contention: bank_conflict_q8 must equal (gcd(stride, banks) - 1)*256 = 768 (got 0)}}
  bcir.gem.contention @bad { stride = 4 : i64, elem_bytes = 4 : i64, count = 64 : i64,
                             cacheline = 64 : i64, banks = 4 : i64,
                             line_waste_q8 = 768 : i64, bank_conflict_q8 = 0 : i64,
                             contention_q8 = 768 : i64, contention_cost = 192 : i64 }
}

// -----

// the combined Q8 signal: contention_q8 must equal line_waste_q8 + bank_conflict_q8.
bcir.module @bad_sum {
  // expected-error @+1 {{contention: contention_q8 must equal line_waste_q8 + bank_conflict_q8 = 4608 (got 9999)}}
  bcir.gem.contention @bad { stride = 16 : i64, elem_bytes = 4 : i64, count = 64 : i64,
                             cacheline = 64 : i64, banks = 4 : i64,
                             line_waste_q8 = 3840 : i64, bank_conflict_q8 = 768 : i64,
                             contention_q8 = 9999 : i64, contention_cost = 1152 : i64 }
}

// -----

// the integer CONTENTION cost: contention_cost must equal (contention_q8 * count) >> 8.
bcir.module @bad_cost {
  // expected-error @+1 {{contention: contention_cost must equal (contention_q8 * count) >> 8 = 1152 (got 7)}}
  bcir.gem.contention @bad { stride = 16 : i64, elem_bytes = 4 : i64, count = 64 : i64,
                             cacheline = 64 : i64, banks = 4 : i64,
                             line_waste_q8 = 3840 : i64, bank_conflict_q8 = 768 : i64,
                             contention_q8 = 4608 : i64, contention_cost = 7 : i64 }
}
