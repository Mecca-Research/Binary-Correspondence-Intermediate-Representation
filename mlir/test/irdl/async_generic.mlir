// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// Phase 8 async tokens in generic syntax, round-tripped through the IRDL
// projection on stock mlir-opt: fork yields a !bcir.token, await joins them.

%t0 = "bcir.async_fork"() {claim = @c0} : () -> !bcir.token
%t1 = "bcir.async_fork"() {claim = @c1} : () -> !bcir.token
"bcir.async_await"(%t0, %t1) : (!bcir.token, !bcir.token) -> ()

// CHECK: "bcir.async_fork"
// CHECK: !bcir.token
// CHECK: "bcir.async_await"
