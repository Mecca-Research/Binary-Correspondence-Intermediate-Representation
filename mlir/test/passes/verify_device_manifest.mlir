// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// Phase D driver-seam hardening: the DEVICE MANIFEST as the static hardware schema
// (D-R1, the kbcir/device_manifest.py twin) and the NATIVE-TILE fragmentation law
// (D-R4 on the law rail): a gem.matmul adjacent to a manifest must submit tiles the
// hardware can schedule. Vacuous when no adjacent pair is present.

// (1 -- NEGATIVE, op law) an ASYMMETRIC distance matrix: an interconnect is a metric;
// an asymmetric entry is a typo, not a topology.
bcir.module @asymmetric_interconnect {
  // expected-error @+1 {{distance[0][1] 512 != distance[1][0] 999 (asymmetric)}}
  bcir.device_manifest @dev { device = "toy_npu", banks = "sram0,hbm0",
    domains = "RAM,HBM", capacities = array<i64: 65536, 16777216>,
    native_tiles = array<i64: 16, 16>, distance = array<i64: 0, 512, 999, 0> }
}

// -----

// (2 -- NEGATIVE, op law) a free far hop: every off-diagonal distance must cost
// at least one Q8 unit (a zero-cost crossing would make every plan degenerate).
bcir.module @free_far_hop {
  // expected-error @+1 {{distance[0][1] must be >= 1 Q8, got 0}}
  bcir.device_manifest @dev2 { device = "toy_npu", banks = "sram0,hbm0",
    domains = "RAM,HBM", capacities = array<i64: 65536, 16777216>,
    native_tiles = array<i64: 16, 16>, distance = array<i64: 0, 0, 0, 0> }
}

// -----

// (3 -- NEGATIVE, R22 seam) the fragmentation law: a 15-wide tile against a 16-native
// device cannot be scheduled without runtime fragmentation -- refused at compile time.
bcir.module @fragmenting_tile {
  bcir.device_manifest @dev3 { device = "toy_npu", banks = "sram0,hbm0",
    domains = "RAM,HBM", capacities = array<i64: 65536, 16777216>,
    native_tiles = array<i64: 16, 16>, distance = array<i64: 0, 512, 512, 0> }
  // expected-error @+1 {{R22: gem.matmul mm submits tile_m = 15 against the adjacent bcir.device_manifest @dev3 whose native tile is 16 (runtime fragmentation is a compile-time refusal)}}
  bcir.gem.matmul @mm { m = 480 : i64, n = 480 : i64, k = 480 : i64,
                        tile_m = 15 : i64, tile_n = 16 : i64, tile_k = 16 : i64,
                        loop_order = "ijk", compute_cost = 100 : i64,
                        mem_cost = 200 : i64, bottleneck = 200 : i64,
                        quant_bits = 0 : i32 }
}

// -----

// (4 -- POSITIVE) the legal pair: native-tile-aligned tiles (32 = 2x16) schedule
// cleanly; an untiled (native 1) scalar bank constrains nothing.
bcir.module @legal_manifest_pair {
  bcir.device_manifest @dev4 { device = "toy_npu", banks = "sram0,hbm0,hbm1",
    domains = "RAM,HBM,HBM", capacities = array<i64: 65536, 16777216, 16777216>,
    native_tiles = array<i64: 16, 16, 16>,
    distance = array<i64: 0, 512, 640, 512, 0, 256, 640, 256, 0> }
  bcir.gem.matmul @mm2 { m = 64 : i64, n = 64 : i64, k = 64 : i64,
                         tile_m = 32 : i64, tile_n = 16 : i64, tile_k = 16 : i64,
                         loop_order = "ijk", compute_cost = 100 : i64,
                         mem_cost = 200 : i64, bottleneck = 200 : i64,
                         quant_bits = 0 : i32 }
}
