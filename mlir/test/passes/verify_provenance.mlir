// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// R13 provenance digest recompute. When a manifest carries the four component hashes (the
// FNV content hashes of module / target / theta / policy, provenance.hash_*), -bcir-verify
// recomputes the digest from first principles (provenance._digest = _fnv(m_module, m_target,
// m_theta, m_policy, artifacts)) and rejects a tampered one -- the law no longer trusts the
// declared `digest`. The constants are a real vector_add manifest's hashes (build_manifest
// on x86_avx512 / cool), so this is a genuine oracle<->law cross-check of the FNV chain.

// A correct manifest (no artifacts): the digest IS the FNV-1a chain of the four component
// hashes -- recomputed and accepted (no diagnostic).
bcir.module @r13_digest_ok {
  bcir.kbcir.provenance_manifest @man_ok {
    digest = 9201837206445197944 : i64, score = 7808 : i64, n_artifacts = 0 : i64,
    reproduced = true,
    m_module = 7127522701151166272 : i64, m_target = 5864064355688965777 : i64,
    m_theta = 1870846051561339781 : i64, m_policy = 4048695575545564183 : i64
  }
}

// -----

// A correct manifest WITH artifacts: the normalized (name, generation) pairs fold in after
// the components, in sorted order -- recomputed and accepted.
bcir.module @r13_digest_ok_artifacts {
  bcir.kbcir.provenance_manifest @man_arts {
    digest = 3780911091132933688 : i64, score = 7808 : i64, n_artifacts = 2 : i64,
    reproduced = true,
    m_module = 7127522701151166272 : i64, m_target = 5864064355688965777 : i64,
    m_theta = 1870846051561339781 : i64, m_policy = 4048695575545564183 : i64,
    artifact_names = ["cal_gen", "map_gen"], artifact_gens = array<i64: 4, 2>
  }
}

// -----

// A tampered manifest: the declared digest does not match the FNV recompute of its own
// component hashes -- rejected (the digest is no longer taken on trust).
bcir.module @r13_digest_tampered {
  // expected-error @+1 {{R13: provenance digest 123456789 does not match the digest recomputed from its component hashes 9201837206445197944}}
  bcir.kbcir.provenance_manifest @man_bad {
    digest = 123456789 : i64, score = 7808 : i64, n_artifacts = 0 : i64,
    reproduced = true,
    m_module = 7127522701151166272 : i64, m_target = 5864064355688965777 : i64,
    m_theta = 1870846051561339781 : i64, m_policy = 4048695575545564183 : i64
  }
}

// -----

// R13 m_theta cross-check against the IR (ok): the cool components + a cool kbcir.theta op
// (all eight pressures 0). hash_theta recomputed from the op equals the declared m_theta
// -- accepted (and the digest still recomputes).
bcir.module @r13_theta_ok {
  bcir.kbcir.theta @theta { thermal = 0 : i32 }
  bcir.kbcir.provenance_manifest @man_theta {
    digest = 9201837206445197944 : i64, score = 7808 : i64, n_artifacts = 0 : i64,
    reproduced = true,
    m_module = 7127522701151166272 : i64, m_target = 5864064355688965777 : i64,
    m_theta = 1870846051561339781 : i64, m_policy = 4048695575545564183 : i64
  }
}

// -----

// R13 m_theta cross-check against the IR (mismatch): the manifest records the cool m_theta
// but the IR's kbcir.theta op is hot (thermal 80) -- the manifest is attached to a different
// runtime state than the one in the IR, and is rejected.
bcir.module @r13_theta_mismatch {
  bcir.kbcir.theta @theta { thermal = 80 : i32 }
  // expected-error @+1 {{R13: manifest m_theta 1870846051561339781 does not match the value recomputed from the IR 7137236898939919207}}
  bcir.kbcir.provenance_manifest @man_theta_bad {
    digest = 9201837206445197944 : i64, score = 7808 : i64, n_artifacts = 0 : i64,
    reproduced = true,
    m_module = 7127522701151166272 : i64, m_target = 5864064355688965777 : i64,
    m_theta = 1870846051561339781 : i64, m_policy = 4048695575545564183 : i64
  }
}

// -----

// The FULL component cross-check: the actual vector_add module (the real goal graph G + the
// x86_avx512 capability + the cool theta + the latency policy's unfolded base weights). Every
// component hash the manifest declares -- m_module / m_target / m_theta / m_policy -- recomputes
// from this IR byte-identically to provenance.hash_* and agrees. No diagnostic.
bcir.module @vec_add {
  bcir.target.capability @cpu {
    target_name = "x86-64-avx512", triple = "x86_64-avx512", isa_features = ["avx512f"],
    lane_widths = array<i64: 1, 8, 16>, warp = 0 : i32, scalable = false, cacheline = 64 : i32,
    gather_penalty = 32 : i32, affinity_domains = 8 : i32, mem_channels = 4 : i32,
    mem_unit = 1 : i32, base_overhead = 4 : i32, thermal_density = 64 : i32,
    power_density = 64 : i32, per_op_heat = 1 : i32, elem_bytes = 4 : i32, cal_gen = 0 : i64
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>,
    base_weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.kbcir.theta @theta { thermal = 0 : i32 }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c attributes {
    claim_id = 1000 : i32, phase = @p0, op = "vector.add", opcode = 3 : i32,
    reads = [@A, @B], writes = [@C], count = 1024 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32, domain = #bcir.domain<ram>,
    hazard = #bcir.hazard<unique>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.kbcir.provenance_manifest @man_full {
    digest = 9201837206445197944 : i64, score = 7808 : i64, n_artifacts = 0 : i64,
    reproduced = true,
    m_module = 7127522701151166272 : i64, m_target = 5864064355688965777 : i64,
    m_theta = 1870846051561339781 : i64, m_policy = 4048695575545564183 : i64
  }
}

// -----

// Tamper the goal graph: the claim's opcode is 4 (SUB), not the real 3 (ADD). The recomputed
// hash_module no longer matches the manifest's m_module -- rejected.
bcir.module @vec_add_bad_opcode {
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c attributes {
    claim_id = 1000 : i32, phase = @p0, op = "vector.add", opcode = 4 : i32,
    reads = [@A, @B], writes = [@C], count = 1024 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32, domain = #bcir.domain<ram>,
    hazard = #bcir.hazard<unique>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  // expected-error @+1 {{R13: manifest m_module 7127522701151166272 does not match the value recomputed from the IR}}
  bcir.kbcir.provenance_manifest @man_bad_mod {
    digest = 9201837206445197944 : i64, score = 7808 : i64, n_artifacts = 0 : i64,
    reproduced = true,
    m_module = 7127522701151166272 : i64, m_target = 5864064355688965777 : i64,
    m_theta = 1870846051561339781 : i64, m_policy = 4048695575545564183 : i64
  }
}

// -----

// Tamper the target: gather_penalty is 16, not the real 32. The recomputed hash_target no
// longer matches the manifest's m_target -- rejected.
bcir.module @cap_bad_target {
  bcir.target.capability @cpu {
    target_name = "x86-64-avx512", triple = "x86_64-avx512", isa_features = ["avx512f"],
    lane_widths = array<i64: 1, 8, 16>, warp = 0 : i32, scalable = false, cacheline = 64 : i32,
    gather_penalty = 16 : i32, affinity_domains = 8 : i32, mem_channels = 4 : i32,
    mem_unit = 1 : i32, base_overhead = 4 : i32, thermal_density = 64 : i32,
    power_density = 64 : i32, per_op_heat = 1 : i32, elem_bytes = 4 : i32, cal_gen = 0 : i64
  }
  // expected-error @+1 {{R13: manifest m_target 5864064355688965777 does not match the value recomputed from the IR}}
  bcir.kbcir.provenance_manifest @man_bad_tgt {
    digest = 9201837206445197944 : i64, score = 7808 : i64, n_artifacts = 0 : i64,
    reproduced = true,
    m_module = 7127522701151166272 : i64, m_target = 5864064355688965777 : i64,
    m_theta = 1870846051561339781 : i64, m_policy = 4048695575545564183 : i64
  }
}
