//===- bcir_orchestrator.cpp - the C<->C++ hand-off seam (scaffold) -------===//
//
// Implementation of the seam declared in bcir_orchestrator.hpp. See that header
// and docs/languages/CPP_HANDOFF_BOUNDARY.md for the contract. The single-node backend is
// REAL (it re-enters the existing freestanding C kernels); the dynamic-graph and
// distributed backends are STUBS that document a real implementation but throw.
//===----------------------------------------------------------------------===//
#include "bcir_orchestrator.hpp"

namespace bcir {

// ---------------------------------------------------------------------------
// admit(): delegate legality to the C/IR rail's own verifier. The C++ side does
// NOT re-derive an R-law verdict (the two-truth quarantine forbids it) -- it asks
// the authority (bcir_sp_verify_semantic) and carries the verdict through.
// ---------------------------------------------------------------------------
bcir_status Orchestrator::admit(const std::uint8_t* data, std::size_t len,
                                std::uint32_t expected_map_gen,
                                std::uint32_t expected_data_gen) const {
  if (data == nullptr) return BCIR_ERR_TRUNCATED;
  return bcir_sp_verify_semantic(data, len, expected_map_gen, expected_data_gen);
}

// ---------------------------------------------------------------------------
// The re-entry: walk the artifact's segments through the EXISTING C decoder
// (bcir_sp_for_each_segment in bcir_runtime.c) and collect claim_ids in dispatch
// order. This is the single, shared "run it through the C kernels" primitive that
// the single-node backend uses -- it is exactly the direct C/IR path, so its
// output is identical to calling the C kernels directly (the round-trip identity).
// ---------------------------------------------------------------------------
namespace {

// The per-segment callback handed to the C decoder. ctx is the order vector.
extern "C" int collect_claim(const bcir_segment_view* seg, void* ctx) {
  auto* order = static_cast<std::vector<std::uint64_t>*>(ctx);
  order->push_back(seg->claim_id);
  return 0;  // 0 == keep walking
}

}  // namespace

// --- SingleNodeOrchestrator (REAL) -----------------------------------------

std::vector<Shard> SingleNodeOrchestrator::shard(const std::uint8_t* data,
                                                 std::size_t len) const {
  // Single node: one shard covering the whole pack. We read n_segments from the
  // validated header so the shard range is real, not assumed.
  bcir_streampack_header hdr{};
  Shard s;
  s.index = 0;
  s.node = 0;
  s.seg_begin = 0;
  s.seg_end = (bcir_sp_validate(data, len, &hdr) == BCIR_OK) ? hdr.n_segments : 0;
  return {s};
}

DispatchResult SingleNodeOrchestrator::dispatch(const std::uint8_t* data,
                                                std::size_t len,
                                                unsigned max_retries) const {
  DispatchResult r;
  r.nodes_used = 1;
  r.shards = 1;

  // The single-node placement (one shard on node 0). We compute it so the contract
  // surface is exercised even though the whole pack runs as one unit.
  const std::vector<Shard> shards = shard(data, len);
  r.shards = shards.size();

  // Re-enter the existing C kernels. Idempotence: the artifact bytes are immutable
  // and the C walk is deterministic, so a retry recomputes the SAME order -- we may
  // safely re-attempt on a (transient) failure without accumulating state. The
  // single-node decoder has no transient faults, so this loop converges immediately;
  // it demonstrates the retry contract the distributed backend would actually use.
  bcir_status st = BCIR_OK;
  for (unsigned attempt = 0; attempt <= max_retries; ++attempt) {
    r.claim_order.clear();
    st = bcir_sp_for_each_segment(data, len, &collect_claim, &r.claim_order);
    if (st == BCIR_OK) break;  // success: stop retrying
    // A real distributed retry would re-place the shard on a healthy node here.
  }
  r.status = st;
  if (st != BCIR_OK) r.claim_order.clear();  // a failed run yields no order
  return r;
}

// --- DynamicGraphOrchestrator (STUB) ---------------------------------------

std::vector<Shard> DynamicGraphOrchestrator::shard(const std::uint8_t* data,
                                                   std::size_t len) const {
  // A real dynamic-graph backend re-freezes a fresh claim-graph each step and hands
  // the resulting StreamPack down. With no graph-builder built, sharding is the
  // single-node placement of whatever frozen artifact it was given.
  bcir_streampack_header hdr{};
  Shard s;
  s.seg_end = (bcir_sp_validate(data, len, &hdr) == BCIR_OK) ? hdr.n_segments : 0;
  return {s};
}

DispatchResult DynamicGraphOrchestrator::dispatch(const std::uint8_t* /*data*/,
                                                  std::size_t /*len*/,
                                                  unsigned /*max_retries*/) const {
  // NOT BUILT. A real implementation owns a mutable C++ graph-builder ABOVE the
  // rail (RL node spawning / mixed-length token graphs), materializes a fresh
  // claim-graph each step, FREEZES it to a StreamPack via the C/IR rail, then hands
  // that immutable artifact through the single-node backend. We fail loudly rather
  // than pretend, so the boundary stays honest.
  throw HandoffError(
      "DynamicGraphOrchestrator is a documented STUB: runtime graph topology lives "
      "in C++ above the rail and freezes to a StreamPack; that builder is not built "
      "(see docs/languages/CPP_HANDOFF_BOUNDARY.md). Use SingleNodeOrchestrator on a frozen "
      "artifact.");
}

// --- DistributedOrchestrator (STUB; real sharding, stubbed dispatch) -------

std::vector<Shard> DistributedOrchestrator::shard(const std::uint8_t* data,
                                                  std::size_t len) const {
  // The sharding LOGIC is real and testable with no cluster: partition the segment
  // stream into `world_size` contiguous, balanced ranges -- exactly the placement a
  // real MPI/NCCL run would hand to each rank. Only the cross-node DISPATCH needs
  // hardware; the partition does not.
  bcir_streampack_header hdr{};
  std::uint32_t n = (bcir_sp_validate(data, len, &hdr) == BCIR_OK) ? hdr.n_segments : 0;
  std::size_t world = world_size_ == 0 ? 1 : world_size_;
  std::vector<Shard> shards;
  std::uint32_t per = (world > 0) ? static_cast<std::uint32_t>((n + world - 1) / world) : n;
  std::uint32_t begin = 0;
  for (std::size_t node = 0; node < world && begin < n; ++node) {
    Shard s;
    s.index = node;
    s.node = node;
    s.seg_begin = begin;
    s.seg_end = (begin + per < n) ? (begin + per) : n;
    shards.push_back(s);
    begin = s.seg_end;
  }
  if (shards.empty()) {  // an empty/zero-segment pack still yields one (empty) shard
    Shard s; s.seg_begin = 0; s.seg_end = 0; shards.push_back(s);
  }
  return shards;
}

DispatchResult DistributedOrchestrator::dispatch(const std::uint8_t* /*data*/,
                                                 std::size_t /*len*/,
                                                 unsigned /*max_retries*/) const {
  // NOT BUILT. A real implementation places each shard on a node (MPI rank / NCCL
  // communicator), dispatches it into that node's local SingleNodeOrchestrator (the
  // re-entry), reduces the per-shard results, and retries/replicates on a node
  // failure. That requires a REAL MPI/NCCL dependency + multi-node hardware we
  // deliberately do not add (it would be untested debt). We fail loudly.
  throw HandoffError(
      "DistributedOrchestrator is a documented STUB: cross-node dispatch needs "
      "MPI/NCCL + a multi-node cluster, deliberately not added (see "
      "docs/languages/CPP_HANDOFF_BOUNDARY.md). shard() is real; per-node re-entry runs the "
      "existing SingleNodeOrchestrator on each rank.");
}

// --- factory ---------------------------------------------------------------

std::unique_ptr<Orchestrator> make_orchestrator(Backend b, std::size_t world_size) {
  switch (b) {
    case Backend::SingleNode:
      return std::make_unique<SingleNodeOrchestrator>();
    case Backend::DynamicGraph:
      return std::make_unique<DynamicGraphOrchestrator>();
    case Backend::Distributed:
      return std::make_unique<DistributedOrchestrator>(world_size);
  }
  return std::make_unique<SingleNodeOrchestrator>();
}

}  // namespace bcir
