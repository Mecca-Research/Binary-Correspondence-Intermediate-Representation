// RUN: bcir-opt -bcir-schedule-eft -bcir-async -bcir-overlap %s | FileCheck %s
//
// G1 / S1-A: ONE canonical schedule artifact (BCIRSchedule.h, the twin of
// gem.schedule.schedule_plan), read by -bcir-schedule-eft, -bcir-async and -bcir-overlap alike,
// with the hazard DAG built over every claim of the phase BEFORE the tail split. The module is
// emitted by the oracle (bcir.lower.mlir.to_mlir over the x86_avx512 profile; the numbers below
// are the oracle's, pinned by bcir/tests/test_schedule.py::test_mlir_schedule_hazards_are_in_sync):
//
//   * @c1 (vector.add, 5248) writes @r11; @c2 is a GGG gather (396288) that READS @r11 -- a RAW
//     across the streams. The parent build placed the tail chain at the phase start, so the
//     gather started at 0, before its producer had finished (assessment row 6, the negative
//     witness); it now starts at 5248 on the tail stream (-1) and awaits [1] under tokens.
//   * @c3 is `barriered` and shares no resource with anything: an ordering fence. Nothing
//     overlaps it -- it waits for @c1 and @c2 (awaits [1, 2]) and @c4, data-independent of
//     everything, waits for it (awaits [3]). The parent build overlapped both.
//   * -bcir-overlap prices exactly this placement: makespan 412032 == serial (gain 0). The
//     parent's wave pricer overlapped the tail with its producer and priced 396288 (gain 15744)
//     -- a schedule the executor never ran.

bcir.module @hazards {
  bcir.target.capability @cpu { target_name = "x86-64-avx512", triple = "x86_64-avx512", isa_features = ["avx2", "avx512f", "fma"], lane_widths = array<i64: 1, 8, 16>, warp = 0 : i32, scalable = false, cacheline = 64 : i32, gather_penalty = 32 : i32, affinity_domains = 8 : i32, mem_channels = 4 : i32, thermal_density = 64 : i32, power_density = 64 : i32, mem_unit = 1 : i32, base_overhead = 4 : i32, per_op_heat = 1 : i32, elem_bytes = 4 : i32, cal_gen = 0 : i64, mem_tier_names = ["L1", "L2", "L3", "DRAM", "HBM", "CXL", "SSD"], mem_tier_values = array<i64: 4, 16, 16, 0, 12, 32, 48, 0, 40, 96, 96, 0, 200, 256, 256, 0, 160, 64, 192, 0, 350, 384, 512, 0, 5000, 1024, 4096, 0> }
  bcir.registry @RES {
    bcir.resource @r10 { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, access = #bcir.access<flat> }
    bcir.resource @r11 { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, access = #bcir.access<flat> }
    bcir.resource @r12 { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, access = #bcir.access<flat> }
    bcir.resource @r13 { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, access = #bcir.access<flat> }
    bcir.resource @r14 { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, access = #bcir.access<flat> }
    bcir.resource @r15 { rid = 15 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, access = #bcir.access<flat> }
    bcir.resource @r16 { rid = 16 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, access = #bcir.access<flat> }
  }
  bcir.kbcir.theta @theta { thermal = 0 : i32, power = 0 : i32, mem_pressure = 0 : i32, contention = 0 : i32, noise = 0 : i32, wear = 0 : i32, utilization = 0 : i32, voltage = 0 : i32 }
  bcir.kbcir.policy @perf { mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>, base_weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1> }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes { claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@r10], writes = [@r11], count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<atomic>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32 } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes { claim_id = 2 : i32, phase = @p0, op = "histogram.gather", reads = [@r11], writes = [@r12], count = 1024 : i64, lane = #bcir.lane<ggg>, stride_class = #bcir.stride_class<random>, stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<atomic>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 13 : i32 } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c3 attributes { claim_id = 3 : i32, phase = @p0, op = "vector.add", reads = [@r13], writes = [@r14], count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<barriered>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32 } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c4 attributes { claim_id = 4 : i32, phase = @p0, op = "vector.add", reads = [@r15], writes = [@r16], count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32 } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.kbcir.plan @plan0 {
    bcir.kbcir.path @path_1_scalar { claim = @c1, realization = "u.scalar", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 1024, memory = 10240, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> }
    bcir.kbcir.path @path_1_vec8 { claim = @c1, realization = "u.vec8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 128, memory = 3072, fabric = 0, sync = 0, compile = 0, thermal = 640, power = 640, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> }
    bcir.kbcir.path @path_1_vec16 { claim = @c1, realization = "u.vec16", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 64, memory = 2560, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> }
    %sel_1 = bcir.kbcir.select @c1 from [@path_1_scalar, @path_1_vec8, @path_1_vec16] { policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>, selected = @path_1_vec16, score = 5248 : i64 } : !bcir.path
    bcir.kbcir.path @path_2_gather { claim = @c2, realization = "ggg.gather", lane = #bcir.lane<ggg>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 0, memory = 198144, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> }
    %sel_2 = bcir.kbcir.select @c2 from [@path_2_gather] { policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>, selected = @path_2_gather, score = 396288 : i64 } : !bcir.path
    bcir.kbcir.path @path_3_scalar { claim = @c3, realization = "u.scalar", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 1024, memory = 10240, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> }
    bcir.kbcir.path @path_3_vec8 { claim = @c3, realization = "u.vec8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 128, memory = 3072, fabric = 0, sync = 0, compile = 0, thermal = 640, power = 640, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> }
    bcir.kbcir.path @path_3_vec16 { claim = @c3, realization = "u.vec16", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 64, memory = 2560, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> }
    %sel_3 = bcir.kbcir.select @c3 from [@path_3_scalar, @path_3_vec8, @path_3_vec16] { policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>, selected = @path_3_vec16, score = 5248 : i64 } : !bcir.path
    bcir.kbcir.path @path_4_scalar { claim = @c4, realization = "u.scalar", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 1024, memory = 10240, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> }
    bcir.kbcir.path @path_4_vec8 { claim = @c4, realization = "u.vec8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 128, memory = 3072, fabric = 0, sync = 0, compile = 0, thermal = 640, power = 640, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> }
    bcir.kbcir.path @path_4_vec16 { claim = @c4, realization = "u.vec16", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 64, memory = 2560, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> }
    %sel_4 = bcir.kbcir.select @c4 from [@path_4_scalar, @path_4_vec8, @path_4_vec16] { policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>, selected = @path_4_vec16, score = 5248 : i64 } : !bcir.path
  }
  bcir.gem.lane_segment @seg_1 { claim = @c1, phase = @p0, lane = #bcir.lane<u>, stride_k = 1 : i32, width = 16 : i32, opcode = "vector.add", reads = [@r10], writes = [@r11], fence_before = [], fence_after = [] }
  bcir.gem.lane_segment @seg_2 { claim = @c2, phase = @p0, lane = #bcir.lane<ggg>, stride_k = 1 : i32, width = 1 : i32, opcode = "histogram.gather", reads = [@r11], writes = [@r12], fence_before = [], fence_after = [] }
  bcir.gem.lane_segment @seg_3 { claim = @c3, phase = @p0, lane = #bcir.lane<u>, stride_k = 1 : i32, width = 16 : i32, opcode = "vector.add", reads = [@r13], writes = [@r14], fence_before = [], fence_after = [] }
  bcir.gem.lane_segment @seg_4 { claim = @c4, phase = @p0, lane = #bcir.lane<u>, stride_k = 1 : i32, width = 16 : i32, opcode = "vector.add", reads = [@r15], writes = [@r16], fence_before = [], fence_after = [] }
}

// CHECK-LABEL: bcir.module @hazards
// CHECK-SAME: kbcir.async_makespan = 412032 : i64
// CHECK-SAME: kbcir.overlap_gain = 0 : i64
// CHECK-SAME: kbcir.overlap_makespan = 412032 : i64
// CHECK-SAME: kbcir.overlap_serial = 412032 : i64
// CHECK-SAME: kbcir.sched_knee = 4 : i64
// CHECK-SAME: kbcir.sched_makespan = 412032 : i64
// c1: the wave producer, first on domain 0.
// CHECK: bcir.claim @c1
// CHECK-SAME: kbcir.async_awaits = []
// CHECK-SAME: kbcir.async_start = 0 : i64
// CHECK-SAME: kbcir.sched_domain = 0 : i64
// CHECK-SAME: kbcir.sched_finish = 5248 : i64
// CHECK-SAME: kbcir.sched_start = 0 : i64
// c2: the tail consumer WAITS for its producer (start == c1's finish) on the tail stream.
// CHECK: bcir.claim @c2
// CHECK-SAME: kbcir.async_awaits = [1]
// CHECK-SAME: kbcir.async_domain = -1 : i64
// CHECK-SAME: kbcir.async_start = 5248 : i64
// CHECK-SAME: kbcir.sched_domain = -1 : i64
// CHECK-SAME: kbcir.sched_finish = 401536 : i64
// CHECK-SAME: kbcir.sched_start = 5248 : i64
// c3: the fence waits for everything before it ...
// CHECK: bcir.claim @c3
// CHECK-SAME: kbcir.async_awaits = [1, 2]
// CHECK-SAME: kbcir.async_start = 401536 : i64
// CHECK-SAME: kbcir.sched_finish = 406784 : i64
// CHECK-SAME: kbcir.sched_start = 401536 : i64
// c4: ... and everything after it waits for the fence, data-independent or not.
// CHECK: bcir.claim @c4
// CHECK-SAME: kbcir.async_awaits = [3]
// CHECK-SAME: kbcir.async_start = 406784 : i64
// CHECK-SAME: kbcir.sched_finish = 412032 : i64
// CHECK-SAME: kbcir.sched_start = 406784 : i64
