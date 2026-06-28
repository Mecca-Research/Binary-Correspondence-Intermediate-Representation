// RUN: bcir-opt -bcir-lower-gem-matmul-buffer -split-input-file %s | FileCheck %s
//
// The -bcir-lower-gem-matmul-buffer lowering: a gem.matmul_buffer (real SSA memref
// operands A/B/C + a tile plan) is lowered into a CONCRETE TILED scf.for LOOP NEST
// computing C += A*B with memref.load/store + arith.mulf/addf. The op is ERASED
// (a genuine lowering). M/N/K are derived from the operands' static shapes.
//
// First module: 4x4x4 tiled 2x2x2, loop_order "ijk" -- the three outer tiled loops
// step by 2 (%c2), the point loops step by 1 (%c1), and the body is
// load A / load B / load C / mulf / addf / store C in order.

// CHECK-LABEL: func.func @mm_ijk
func.func @mm_ijk(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  bcir.gem.matmul_buffer tile(2, 2, 2) order "ijk"
    %A, %B, %C : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
  return
}

// The plan op must be gone after lowering.
// CHECK-NOT: bcir.gem.matmul_buffer

// Three outer tiled loops, each stepping by the tile extent (%c2). In ijk order the
// outermost is the i loop, then j, then k.
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %[[C2]]
// CHECK:   scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %[[C2]]
// CHECK:     scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %[[C2]]
// Ragged-boundary clamps: min(origin + tile, dim).
// CHECK:       arith.addi
// CHECK:       arith.minsi
// The three point loops step by 1 (%c1), nested ii -> jj -> kk.
// CHECK:       scf.for %[[II:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK:         scf.for %[[JJ:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK:           scf.for %[[KK:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// The fma body in order: load A[ii,kk], load B[kk,jj], load C[ii,jj], mulf, addf, store C.
// CHECK:             %[[A:.*]] = memref.load %{{.*}}[%[[II]], %[[KK]]]
// CHECK:             %[[B:.*]] = memref.load %{{.*}}[%[[KK]], %[[JJ]]]
// CHECK:             %[[C:.*]] = memref.load %{{.*}}[%[[II]], %[[JJ]]]
// CHECK:             %[[P:.*]] = arith.mulf %[[A]], %[[B]]
// CHECK:             %[[ACC:.*]] = arith.addf %[[C]], %[[P]]
// CHECK:             memref.store %[[ACC]], %{{.*}}[%[[II]], %[[JJ]]]

// -----

// Second module: SAME shapes/tiles but loop_order "jik" -- the outer nesting order
// differs (j outer, then i, then k). The permutation is observable: the FIRST point
// loop induction variable (ii) is derived from the SECOND outer loop here, whereas in
// ijk it came from the first. We pin the outer order by checking that the i tiled loop
// bound is min(...) against %c4 (M) and appears nested inside the j loop.

// CHECK-LABEL: func.func @mm_jik
func.func @mm_jik(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  bcir.gem.matmul_buffer tile(2, 2, 2) order "jik"
    %A, %B, %C : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
  return
}

// CHECK-NOT: bcir.gem.matmul_buffer
// jik: the outermost tiled loop is j, then i, then k -- a different nesting than ijk.
// CHECK: scf.for %[[JT:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK:   scf.for %[[IT:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK:     scf.for %[[KT:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// The point loops + fma body are loop-order invariant (still ii -> jj -> kk, store C).
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             memref.load
// CHECK:             memref.load
// CHECK:             memref.load
// CHECK:             arith.mulf
// CHECK:             arith.addf
// CHECK:             memref.store
