//===- bcir.irdl.mlir - pure-IRDL BCIR projection -------------------------===//
//
// The portability rail: a structural BCIR dialect definition shipped as IR data
// and loaded by stock `mlir-opt` -- no BCIR-authored C++ compiled first. This is
// the closest MLIR-native version of the WASM-like property (artifacts are pure
// data; the standard engine loads them).
//
//   mlir-opt --irdl-file=mlir/irdl/bcir.irdl.mlir <program-in-bcir-generic.mlir>
//
// Scope: structural only. It constrains op/type/attr *shape* (and region
// presence), not deep semantics. Phase-DAG cycle checking, resource-domain
// resolution, K_BCIR min-plus scoring, and generation freshness remain in the
// ODS + bcir-opt rail and in the bcir/ oracle.
//
// Rules followed (per the IRDL docs): no `irdl.c_pred` (would require compiled
// C++ and block runtime registration); generic attributes via `irdl.any` where
// EnumAttr sugar is too heavy; structural constraints only.
//
// NOT validated on this host (no mlir-opt). Validate with
// tools/irdl/run_stock_mlir_opt.sh once an MLIR toolchain is present; the exact
// `--irdl-file`/load flag is discovered by tools/irdl/probe_irdl_flags.sh.
//
//===----------------------------------------------------------------------===//

irdl.dialect @bcir {
  // ---- Types ----
  irdl.type @resource
  irdl.type @path
  irdl.type @stream

  // ---- BCIR-0..2: core containers + claims ----
  irdl.operation @module {
    %body = irdl.region
    irdl.regions(body: %body)
  }

  irdl.operation @registry {
    %body = irdl.region
    irdl.regions(body: %body)
  }

  irdl.operation @resource {
    %res = irdl.base @bcir::@resource
    irdl.results(handle: %res)
  }

  irdl.operation @phase
  irdl.operation @claim {
    %body = irdl.region
    irdl.regions(body: %body)
  }

  irdl.operation @index_range {
    %idx = irdl.is index
    irdl.results(iv: %idx)
  }

  irdl.operation @load {
    %idx = irdl.is index
    %value = irdl.any
    irdl.operands(index: %idx)
    irdl.results(value: %value)
  }

  irdl.operation @store {
    %idx = irdl.is index
    %value = irdl.any
    irdl.operands(value: %value, index: %idx)
  }

  irdl.operation @compute {
    %value = irdl.any
    irdl.operands(args: variadic %value)
    irdl.results(result: %value)
  }

  irdl.operation @barrier

  // ---- CT1: target-open container + memory hierarchy ----
  irdl.operation @"target.capability"
  irdl.operation @"mem.tier"
  irdl.operation @"mem.ham"
  irdl.operation @"mem.cxl_swap"

  // ---- BCIR-3: K_BCIR planning ----
  irdl.operation @"kbcir.policy"
  irdl.operation @"kbcir.plan" {
    %body = irdl.region
    irdl.regions(body: %body)
  }
  irdl.operation @"kbcir.path" {
    %p = irdl.base @bcir::@path
    irdl.results(path: %p)
  }
  irdl.operation @"kbcir.select" {
    %p = irdl.base @bcir::@path
    irdl.results(result: %p)
  }

  // ---- BCIR-4: GEM StreamPack ----
  irdl.operation @"gem.stream_pack" {
    %s = irdl.base @bcir::@stream
    %body = irdl.region
    irdl.results(stream: %s)
    irdl.regions(body: %body)
  }
  irdl.operation @"gem.prefetch"
  irdl.operation @"gem.block"
  irdl.operation @"gem.lane_segment"
  irdl.operation @"trace.note"
}
