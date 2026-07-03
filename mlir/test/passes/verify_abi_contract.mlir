// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// §5.14 Phase 2 (the last area): the CALL-ABI CONTRACT law -- the R12 lowering-contract
// discipline extended to the call ABI. A bcir.abi_contract must name a target from the
// normative data-model matrix and declare that target's pointer/long sizes truthfully.
// Mirrors the oracle's abi.verify_abi_contract; vacuous when no contract op is present.

// (1 -- NEGATIVE) a contract lying about the LLP64 data model: x86_64-windows has long_size 4.
bcir.module @r12_llp64_lie {
  // expected-error @+1 {{R12: abi_contract f declares pointer_size=8/long_size=8 but target 'x86_64-windows' has pointer_size=8/long_size=4}}
  bcir.abi_contract @f { target = "x86_64-windows", pointer_size = 8 : i32, long_size = 8 : i32,
                         param_sizes = array<i64: 8, 4>, param_aligns = array<i64: 8, 4>,
                         ret_size = 4 : i64, ret_align = 4 : i64 }
}

// -----

// (2 -- NEGATIVE) an unknown target name: the matrix is normative.
bcir.module @r12_unknown_target {
  // expected-error @+1 {{R12: abi_contract g names an unknown target 'sparc64-solaris'}}
  bcir.abi_contract @g { target = "sparc64-solaris", pointer_size = 8 : i32, long_size = 8 : i32,
                         param_sizes = array<i64: 4>, param_aligns = array<i64: 4> }
}

// -----

// (3 -- POSITIVE) the truthful record the oracle emits for an ILP32 unit.
bcir.module @r12_legal_contract {
  bcir.abi_contract @h { target = "i386-linux", pointer_size = 4 : i32, long_size = 4 : i32,
                         param_sizes = array<i64: 4, 4>, param_aligns = array<i64: 4, 4>,
                         ret_size = 4 : i64, ret_align = 4 : i64 }
}
