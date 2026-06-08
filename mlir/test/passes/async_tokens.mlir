// RUN: bcir-opt %s | FileCheck %s
//
// Phase 8: !bcir.token async dependency model. fork launches a claim and yields a
// completion token; await joins a set of tokens.

// CHECK-LABEL: func.func @async_demo
func.func @async_demo() {
  // CHECK: %[[T0:.*]] = bcir.async.fork @c0 : !bcir.token
  %t0 = bcir.async.fork @c0 : !bcir.token
  // CHECK: %[[T1:.*]] = bcir.async.fork @c1 : !bcir.token
  %t1 = bcir.async.fork @c1 : !bcir.token
  // CHECK: bcir.async.await %[[T0]], %[[T1]] : !bcir.token, !bcir.token
  bcir.async.await %t0, %t1 : !bcir.token, !bcir.token
  return
}
