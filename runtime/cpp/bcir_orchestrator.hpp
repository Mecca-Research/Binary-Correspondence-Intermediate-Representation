//===- bcir_orchestrator.hpp - the C<->C++ hand-off seam -------------------===//
//
// The DESIGNED boundary between BCIR's deterministic single-node C/IR rail and the high-level
// C++ ABOVE it (dynamic graph topology + distributed multi-node orchestration). The contract is
// specified in docs/languages/CPP_HANDOFF_BOUNDARY.md; this header is the compilable seam.
//
// THE SEAM ARTIFACT is the frozen StreamPack (runtime/c/bcir_streampack.h, the "WASM analog"):
// the C/IR rail PRODUCES it, the C++ Orchestrator CONSUMES it, decides placement/topology, and
// dispatches shards back into the existing single-node C kernels (re-entry). The C++ side may
// schedule / shard / retry / replicate, but it may NEVER mutate a frozen artifact's bytes or
// semantics and it may NEVER become an R-law verdict -- the two-truth quarantine extends across
// the boundary (see the doc).
//
// THE HAND-OFF (G16, S3-C). The artifact crosses as a BORROWED VIEW WITH AN EXPLICIT LIFETIME
// (bcir_handoff.hpp over the C pack table): it is written once into an arena slot and read in
// place by every consumer; a view that outlives its owner is refused, never read. ADMISSION is
// the LIVE control plane's (G14) predicate against the installed registry -- no caller-supplied
// generation numbers -- and DISPATCH runs only what was admitted at the resident generation of
// that registry, as a phase of the plane. admit() is not virtual: no backend can override
// legality.
//
// HONEST DEPTH: SingleNodeOrchestrator is REAL. DynamicGraphOrchestrator is REAL: step() freezes
// a fresh graph through the C/IR rail (bcir_plan_func + bcir_hydrate_generations) into a slot,
// admits it against the live plane and runs it through the single-node path. The
// DistributedOrchestrator's partition and its manifest-of-shards are REAL (each shard a pack a
// node admits and runs by itself; the shards reassemble to the whole by digest); only its
// cross-node DISPATCH is a stub -- it needs MPI/NCCL and a cluster we deliberately do not add --
// and it fails loudly.
//
// Plain C++17, standalone. It links against the freestanding C runtime (bcir_runtime.c and the
// hand-off units); it is NOT part of the MLIR/LLVM cmake.
//===----------------------------------------------------------------------===//
#ifndef BCIR_ORCHESTRATOR_HPP
#define BCIR_ORCHESTRATOR_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bcir_handoff.hpp"

extern "C" {
#include "bcir_runtime.h"
}

namespace bcir {

// --- the data contract crossing the seam -----------------------------------
//
// A Shard is the unit the C++ side may place on a node: a segment range [seg_begin, seg_end) of
// the whole pack and a placement decision. The distributed backend cuts each range into its own
// runnable pack (cut_shards); a shard NEVER mutates the whole's bytes.
struct Shard {
  std::size_t index = 0; // shard ordinal (0 for the single-node whole-pack)
  std::size_t node = 0;  // the node the C++ orchestrator placed it on
  std::uint32_t seg_begin = 0;
  std::uint32_t seg_end = 0;
};

// The result of a dispatch: the per-segment dispatch sequence the C kernels produced (claim_id in
// dispatch order), the hand-off outcome that decided whether it ran, and a status. Equality of
// claim_order with the direct C walk is the proof the seam round-trips losslessly.
struct DispatchResult {
  bcir_status status = BCIR_OK;           // BCIR_OK, or the refusal's status (carried up)
  HandoffResult handoff;                  // the table/plane decision, verbatim
  std::vector<std::uint64_t> claim_order; // claim_ids in C-kernel dispatch order
  std::size_t nodes_used = 0;
  std::size_t shards = 0;

  bool ok() const { return status == BCIR_OK; }
};

// A hand-off error that is NOT a deterministic R-law verdict: an OPERATIONAL fault the C++ side
// may handle (retry/replicate/abort) -- an unbuilt cross-node transport, an exhausted retry. It
// never reaches back and rewrites an artifact's legality. Verdicts arrive as a bcir_status.
class HandoffError : public std::runtime_error {
public:
  explicit HandoffError(const std::string &what) : std::runtime_error(what) {}
};

// --- the Orchestrator interface (abstract base) ----------------------------
class Orchestrator {
public:
  virtual ~Orchestrator() = default;

  virtual std::string name() const = 0;

  // (a) ADMIT -- NOT virtual: the live plane's one predicate (bcir_ctl_admit_pack through the
  // pack table) against the installed registry. The C++ side carries the verdict; it does not
  // derive one, and it passes no generation numbers of its own.
  HandoffResult admit(const PackView &pack, bcir_ctl_state &plane) const;

  // (b) SHARD: the placement decision over the artifact's segment stream (read-only).
  virtual std::vector<Shard> shard(const PackView &pack) const = 0;

  // (c) DISPATCH + RE-ENTER: run an ADMITTED artifact through the existing single-node C kernels
  // as a phase of the plane. Refusals (a dead view; no admission at the resident generation; a
  // draining plane) are verdicts in the result, not exceptions, and are never retried -- the
  // bytes are immutable, so only a transient fault could change the outcome.
  virtual DispatchResult dispatch(const PackView &pack, bcir_ctl_state &plane,
                                  unsigned max_retries = 0) const = 0;
};

// --- (REAL) the single-node reference implementation -----------------------
class SingleNodeOrchestrator : public Orchestrator {
public:
  std::string name() const override { return "single-node-reference"; }
  std::vector<Shard> shard(const PackView &pack) const override;
  DispatchResult dispatch(const PackView &pack, bcir_ctl_state &plane,
                          unsigned max_retries = 0) const override;
};

// --- (REAL) the dynamic-graph backend ----------------------------------------
//
// The dynamic part lives entirely in C++ ABOVE the rail: a GraphBuilder is the step's mutable
// claim graph; step() FREEZES it through the C/IR rail into an arena slot bound to the live
// registry, admits it against the plane and hands that immutable artifact to the single-node
// path. What crosses down is always a frozen artifact.
struct StepResult {
  HandoffResult freeze;
  HandoffResult admission;
  DispatchResult dispatch;
  HandoffResult release;

  bool ok() const { return freeze.ok() && admission.ok() && dispatch.ok() && release.ok(); }
};

class DynamicGraphOrchestrator : public Orchestrator {
public:
  std::string name() const override { return "dynamic-graph"; }
  std::vector<Shard> shard(const PackView &pack) const override;
  DispatchResult dispatch(const PackView &pack, bcir_ctl_state &plane,
                          unsigned max_retries = 0) const override;
  // One step: freeze -> admit -> dispatch -> release. A step that fails at any stage leaves no
  // slot behind; the stage that refused carries its status.
  StepResult step(const GraphBuilder &graph, PackArena &arena,
                  const std::vector<bcir_generation_view> &registry, std::uint32_t topo_gen,
                  bcir_ctl_state &plane) const;
};

// --- (STUB dispatch; REAL partition and shards) the distributed backend ------
//
// shard() partitions the segment stream into `world_size` contiguous ranges (bcir_shm_partition:
// the one partition both rails use), and cut() turns them into a manifest-of-shards -- each
// shard a pack the rank's own SingleNodeOrchestrator admits and runs, the set reassembling to
// the whole by digest. Cross-node DISPATCH needs MPI/NCCL and a cluster: a stub that throws.
class DistributedOrchestrator : public Orchestrator {
public:
  explicit DistributedOrchestrator(std::size_t world_size = 1) : world_size_(world_size) {}
  std::string name() const override { return "distributed-MPI/NCCL-STUB"; }
  std::vector<Shard> shard(const PackView &pack) const override;
  ShardSet cut(const PackView &pack, PackArena &arena) const;
  DispatchResult dispatch(const PackView &pack, bcir_ctl_state &plane,
                          unsigned max_retries = 0) const override;

private:
  std::size_t world_size_;
};

enum class Backend { SingleNode, DynamicGraph, Distributed };

std::unique_ptr<Orchestrator> make_orchestrator(Backend b, std::size_t world_size = 1);

} // namespace bcir

#endif // BCIR_ORCHESTRATOR_HPP
