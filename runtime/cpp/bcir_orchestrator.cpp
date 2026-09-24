//===- bcir_orchestrator.cpp - the C<->C++ hand-off seam -------------------===//
//
// Implementation of the seam declared in bcir_orchestrator.hpp; the contract is
// docs/languages/CPP_HANDOFF_BOUNDARY.md. Admission and dispatch go through the pack table
// (bcir_handoff.hpp), so the lifetime and generation laws are the C twin's, not re-derived here.
//===----------------------------------------------------------------------===//
#include "bcir_orchestrator.hpp"

namespace bcir {

namespace {

// The per-segment callback handed to the C decoder. ctx is the order vector.
extern "C" int collect_claim(const bcir_segment_view *seg, void *ctx) {
  static_cast<std::vector<std::uint64_t> *>(ctx)->push_back(seg->claim_id);
  return 0; // 0 == keep walking
}

// The segment count of a live, well-formed artifact (0 for a dead view or a malformed pack: the
// placement of nothing).
std::uint32_t segments_of(const PackView &pack) {
  Borrow b = pack.borrow();
  bcir_streampack_header hdr{};
  if (!b.ok() || bcir_sp_validate(b.data(), b.size(), &hdr) != BCIR_OK)
    return 0;
  return hdr.n_segments;
}

// The single-node re-entry: dispatch the admitted artifact through the table (which checks the
// view, the admission and the plane) into the existing C walk.
DispatchResult run_single(const PackView &pack, bcir_ctl_state &plane, unsigned max_retries) {
  DispatchResult r;
  r.nodes_used = 1;
  r.shards = 1;
  for (unsigned attempt = 0; attempt <= max_retries; ++attempt) {
    r.claim_order.clear();
    r.handoff = dispatch_view(pack, plane, &collect_claim, &r.claim_order);
    // A refusal is a verdict about immutable bytes and a plane state: re-dispatching the same
    // artifact cannot change it, so only a walk failure (none on a single node) would be retried.
    if (r.handoff.ok() || r.handoff.refusal() != BCIR_HO_REFUSAL_WALK)
      break;
  }
  r.status = r.handoff.status();
  if (!r.handoff.ok())
    r.claim_order.clear(); // a refused run yields no order
  return r;
}

} // namespace

HandoffResult Orchestrator::admit(const PackView &pack, bcir_ctl_state &plane) const {
  return admit_view(pack, plane);
}

// --- SingleNodeOrchestrator (REAL) -----------------------------------------

std::vector<Shard> SingleNodeOrchestrator::shard(const PackView &pack) const {
  Shard s;
  s.seg_end = segments_of(pack);
  return {s};
}

DispatchResult SingleNodeOrchestrator::dispatch(const PackView &pack, bcir_ctl_state &plane,
                                                unsigned max_retries) const {
  return run_single(pack, plane, max_retries);
}

// --- DynamicGraphOrchestrator (REAL) ---------------------------------------

std::vector<Shard> DynamicGraphOrchestrator::shard(const PackView &pack) const {
  Shard s;
  s.seg_end = segments_of(pack);
  return {s};
}

DispatchResult DynamicGraphOrchestrator::dispatch(const PackView &pack, bcir_ctl_state &plane,
                                                  unsigned max_retries) const {
  return run_single(pack, plane, max_retries); // a frozen artifact is a frozen artifact
}

StepResult DynamicGraphOrchestrator::step(const GraphBuilder &graph, PackArena &arena,
                                          const std::vector<bcir_generation_view> &registry,
                                          std::uint32_t topo_gen, bcir_ctl_state &plane) const {
  StepResult r;
  PackOwner owner;
  r.freeze = graph.freeze(arena, registry, topo_gen, owner);
  if (!r.freeze.ok())
    return r;
  r.admission = admit(owner.view(), plane);
  if (r.admission.ok())
    r.dispatch = run_single(owner.view(), plane, 0);
  r.release = owner.release(); // the step's artifact dies with the step
  return r;
}

// --- DistributedOrchestrator (REAL partition and shards; STUB dispatch) -----

std::vector<Shard> DistributedOrchestrator::shard(const PackView &pack) const {
  const std::uint32_t n = segments_of(pack);
  const std::uint32_t world =
      world_size_ == 0
          ? 1u
          : (world_size_ > 0xFFFFFFFFu ? 0xFFFFFFFFu : static_cast<std::uint32_t>(world_size_));
  std::vector<std::uint32_t> begins(BCIR_SHM_SHARDS_MAX), ends(BCIR_SHM_SHARDS_MAX);
  std::uint32_t count = 0;
  std::vector<Shard> shards;
  if (bcir_shm_partition(n, world, begins.data(), ends.data(), BCIR_SHM_SHARDS_MAX, &count) !=
      BCIR_OK)
    return shards; // past the manifest's shard bound: no placement
  for (std::uint32_t i = 0; i < count; ++i) {
    Shard s;
    s.index = i;
    s.node = i;
    s.seg_begin = begins[i];
    s.seg_end = ends[i];
    shards.push_back(s);
  }
  return shards;
}

ShardSet DistributedOrchestrator::cut(const PackView &pack, PackArena &arena) const {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
  for (const Shard &s : shard(pack))
    ranges.emplace_back(s.seg_begin, s.seg_end);
  if (ranges.empty()) {
    ShardSet out;
    out.result = HandoffResult::refused(BCIR_HO_REFUSAL_MALFORMED, BCIR_ERR_SHARD);
    return out;
  }
  return cut_shards(pack, arena, ranges);
}

DispatchResult DistributedOrchestrator::dispatch(const PackView & /*pack*/,
                                                 bcir_ctl_state & /*plane*/,
                                                 unsigned /*max_retries*/) const {
  // NOT BUILT. A real implementation ships the manifest and each shard to its rank (cut()),
  // where the rank's SingleNodeOrchestrator admits and runs it, and reduces the per-shard
  // results, retrying/replicating on a node failure. That needs a REAL MPI/NCCL dependency and
  // multi-node hardware we deliberately do not add (untested debt). We fail loudly.
  throw HandoffError(
      "DistributedOrchestrator dispatch is a documented STUB: cross-node transport needs "
      "MPI/NCCL + a multi-node cluster, deliberately not added (see "
      "docs/languages/CPP_HANDOFF_BOUNDARY.md). shard() and cut() are real; each rank runs "
      "its shard through SingleNodeOrchestrator.");
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

} // namespace bcir
