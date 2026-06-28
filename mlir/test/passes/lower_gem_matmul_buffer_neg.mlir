// RUN: bcir-opt -split-input-file -verify-diagnostics %s
//
// Parse-time op-verifier negative cases for bcir.gem.matmul_buffer (hasVerifier=1).
// Each module is malformed; -verify-diagnostics asserts the op emits the pinned error.

// Mismatched contraction dim K: A is 4x4 (K=4) but B is 8x4 (K=8).
func.func @mm_bad_k(%A: memref<4x4xf32>, %B: memref<8x4xf32>, %C: memref<4x4xf32>) {
  // expected-error @+1 {{contraction dim mismatch}}
  bcir.gem.matmul_buffer tile(2, 2, 2) order "ijk"
    %A, %B, %C : memref<4x4xf32>, memref<8x4xf32>, memref<4x4xf32>
  return
}

// -----

// Mismatched M: A.dim0 = 4 but C.dim0 = 8.
func.func @mm_bad_m(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<8x4xf32>) {
  // expected-error @+1 {{M mismatch}}
  bcir.gem.matmul_buffer tile(2, 2, 2) order "ijk"
    %A, %B, %C : memref<4x4xf32>, memref<4x4xf32>, memref<8x4xf32>
  return
}

// -----

// Dynamic shape is rejected (static shapes are required to derive M/N/K).
func.func @mm_dynamic(%A: memref<?x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  // expected-error @+1 {{must have a static shape}}
  bcir.gem.matmul_buffer tile(2, 2, 2) order "ijk"
    %A, %B, %C : memref<?x4xf32>, memref<4x4xf32>, memref<4x4xf32>
  return
}

// -----

// Tile extent out of range: tile_m = 9 > M = 4.
func.func @mm_bad_tile(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  // expected-error @+1 {{each tile extent must be in [1, dim]}}
  bcir.gem.matmul_buffer tile(9, 2, 2) order "ijk"
    %A, %B, %C : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
  return
}

// -----

// Unknown loop order.
func.func @mm_bad_order(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  // expected-error @+1 {{loop_order must be one of ijk|ikj|jik}}
  bcir.gem.matmul_buffer tile(2, 2, 2) order "kji"
    %A, %B, %C : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
  return
}

// -----

// Wrong element type: i32 instead of f32.
func.func @mm_bad_elt(%A: memref<4x4xi32>, %B: memref<4x4xi32>, %C: memref<4x4xi32>) {
  // expected-error @+1 {{element type must be f32}}
  bcir.gem.matmul_buffer tile(2, 2, 2) order "ijk"
    %A, %B, %C : memref<4x4xi32>, memref<4x4xi32>, memref<4x4xi32>
  return
}
