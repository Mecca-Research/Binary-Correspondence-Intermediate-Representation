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
