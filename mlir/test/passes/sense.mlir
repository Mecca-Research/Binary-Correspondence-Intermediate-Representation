// RUN: bcir-opt -bcir-sense %s | FileCheck %s
//
// The regret-driven telemetry-sensing gate (kbcir/sensing.py RegretSensor.sense). Per
// segment, cv_milli = 1000*stdev/mean over the trace.data_dna cycles (population variance,
// floor-isqrt). segA's two samples (100, 300) -> mean 200, stdev 100, cv 500 >= the 200
// threshold with n >= 2 -> "high" (uncertain, worth instrumenting). segB (200, 205) ->
// cv 9 < threshold but n >= 2 -> "low". segC has a single sample (n < 2) -> "off" (not yet
// enough to judge). Deterministic, ranked by (-cv_milli, segment); matches the oracle.

bcir.module @sense {
  bcir.trace.data_dna @r1 { segment = @segA, cycles = 100 : i64, bytes = 0 : i64, misses = 0 : i32, thermal = 0 : i32, voltage = 0 : i32, utilization = 0 : i32 }
  bcir.trace.data_dna @r2 { segment = @segA, cycles = 300 : i64, bytes = 0 : i64, misses = 0 : i32, thermal = 0 : i32, voltage = 0 : i32, utilization = 0 : i32 }
  bcir.trace.data_dna @r3 { segment = @segB, cycles = 200 : i64, bytes = 0 : i64, misses = 0 : i32, thermal = 0 : i32, voltage = 0 : i32, utilization = 0 : i32 }
  bcir.trace.data_dna @r4 { segment = @segB, cycles = 205 : i64, bytes = 0 : i64, misses = 0 : i32, thermal = 0 : i32, voltage = 0 : i32, utilization = 0 : i32 }
  bcir.trace.data_dna @r5 { segment = @segC, cycles = 500 : i64, bytes = 0 : i64, misses = 0 : i32, thermal = 0 : i32, voltage = 0 : i32, utilization = 0 : i32 }
  // Squaring either sample exceeds i64. The sensor must retain n=2 and take
  // the conservative max-uncertainty path instead of wrapping or classifying
  // the segment "off" as if no samples had arrived.
  bcir.trace.data_dna @r6 { segment = @segOverflow, cycles = 9223372036854775807 : i64, bytes = 0 : i64, misses = 0 : i32, thermal = 0 : i32, voltage = 0 : i32, utilization = 0 : i32 }
  bcir.trace.data_dna @r7 { segment = @segOverflow, cycles = 9223372036854775807 : i64, bytes = 0 : i64, misses = 0 : i32, thermal = 0 : i32, voltage = 0 : i32, utilization = 0 : i32 }
}

// CHECK-DAG: bcir.trace.data_dna @r1 {{.*}}kbcir.sense_cv = 500{{.*}}kbcir.sense_resolution = "high"
// CHECK-DAG: bcir.trace.data_dna @r3 {{.*}}kbcir.sense_cv = 9{{.*}}kbcir.sense_resolution = "low"
// CHECK-DAG: bcir.trace.data_dna @r5 {{.*}}kbcir.sense_cv = 0{{.*}}kbcir.sense_resolution = "off"
// CHECK-DAG: bcir.trace.data_dna @r6 {{.*}}kbcir.sense_cv = 9223372036854775807{{.*}}kbcir.sense_resolution = "high"
