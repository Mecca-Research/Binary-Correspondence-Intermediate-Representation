//===- bcir_orchestrator.hpp - the C<->C++ hand-off seam (scaffold) -------===//
//
// The DESIGNED boundary between BCIR's deterministic single-node C/IR rail and
// the high-level C++ ABOVE it (dynamic graph topology + distributed multi-node
// orchestration). The contract is specified in docs/languages/CPP_HANDOFF_BOUNDARY.md; this
// header is the compilable seam.
//
// THE SEAM ARTIFACT is the frozen StreamPack (runtime/c/bcir_streampack.h, the
// "WASM analog"): the C/IR rail PRODUCES it, the C++ Orchestrator CONSUMES it,
// decides placement/topology, and dispatches shards back into the existing
// single-node C kernels (re-entry). The C++ side may schedule / shard / retry /
// replicate, but it may NEVER mutate a frozen artifact's bytes or semantics and
// it may NEVER become an R-law verdict -- the two-truth quarantine extends across
// the boundary (see the doc).
//
// HONEST DEPTH: SingleNodeOrchestrator is REAL -- it consumes the artifact, runs
// it through the existing freestanding C decoder (bcir_runtime.c), and proves the
// seam round-trips (the smoke test asserts C++-seam result == direct-C result).
// DynamicGraphOrchestrator and DistributedOrchestrator are STUBS behind the same
// interface: they document what a real implementation would do but are not built
// (a real one needs multi-node hardware + an MPI/NCCL dependency we deliberately
// do not add). They are clearly marked and fail loudly if dispatched.
//
// Plain C++17, standalone. It links ONLY against the existing freestanding C
// runtime (runtime/c/bcir_runtime.c). It is NOT part of the MLIR/LLVM cmake.
//===----------------------------------------------------------------------===//
#ifndef BCIR_ORCHESTRATOR_HPP
#define BCIR_ORCHESTRATOR_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// The frozen StreamPack ABI + the freestanding single-node C decoder/kernels.
// This is the ONLY thing the C++ seam links against -- the C/IR rail stays the
// authority; C++ is a consumer of its frozen artifact.
extern "C" {
#include "bcir_runtime.h"
}

namespace bcir {

// --- the data contract crossing the seam -----------------------------------
//
// A Shard is the unit the C++ side may place on a node. In this single-node
// scaffold there is exactly one shard (the whole pack); the distributed backend
// would partition the segment stream into many shards across nodes. A shard NEVER
// owns or copies the bytes mutably -- it is a (read-only) view + a placement
// decision. The artifact bytes are immutable across the boundary.
struct Shard {
  std::size_t index = 0;      // shard ordinal (0 for the single-node whole-pack)
  std::size_t node = 0;       // the node the C++ orchestrator placed it on
  // [begin,end) segment range this shard owns (whole pack: [0, n_segments)).
  std::uint32_t seg_begin = 0;
  std::uint32_t seg_end = 0;
};

// The result of running the seam: the per-segment dispatch sequence the C kernels
// produced (claim_id in dispatch order), plus a status. This is the OBSERVABLE the
// round-trip smoke test compares against the direct C path -- if the C++ seam ran
// the artifact through anything OTHER than the existing C kernels, the sequences
// would diverge. Equality is the proof the seam round-trips losslessly.
struct DispatchResult {
  bcir_status status = BCIR_OK;             // the C-rail status (BCIR_OK on success)
  std::vector<std::uint64_t> claim_order;   // claim_ids in C-kernel dispatch order
  std::size_t nodes_used = 0;               // how many nodes the orchestrator used
  std::size_t shards = 0;                   // how many shards it dispatched

  bool ok() const { return status == BCIR_OK; }
};

// A hand-off error that is NOT a deterministic R-law verdict. Per the two-truth
// quarantine extension: an orchestration failure (a node died, a retry was
// exhausted, a backend is unbuilt) is an OPERATIONAL fault the C++ side may handle
// (retry/replicate/abort) -- it never reaches back and rewrites the artifact's
// legality. C/IR R-law verdicts arrive as a bcir_status inside DispatchResult, not
// as a C++ exception.
class HandoffError : public std::runtime_error {
 public:
  explicit HandoffError(const std::string& what) : std::runtime_error(what) {}
};

// --- the Orchestrator interface (abstract base) ----------------------------
//
// The single seam contract. A backend (a) CONSUMES the serialized hand-off
// artifact, (b) exposes the shard/dispatch/retry contract, and (c) RE-ENTERS the
// existing single-node C kernels per shard. The artifact is passed as an immutable
// byte view -- the C++ side gets read access, never write access.
class Orchestrator {
 public:
  virtual ~Orchestrator() = default;

  // Human-readable backend identity (for logs / the audit trail).
  virtual std::string name() const = 0;

  // (a) CONSUME: validate the artifact is a well-formed, semantically-clean
  // StreamPack BEFORE any placement. This delegates to the C/IR rail's own
  // verifier (bcir_sp_verify_semantic) -- the C++ side does NOT re-derive
  // legality; it asks the authority. Returns the C-rail status (BCIR_OK == admit).
  // Generation gates default to "skip" ((uint32_t)-1): a real deployment passes
  // the live registry generation so a STALE pack is rejected at the boundary.
  virtual bcir_status admit(const std::uint8_t* data, std::size_t len,
                            std::uint32_t expected_map_gen = 0xFFFFFFFFu,
                            std::uint32_t expected_data_gen = 0xFFFFFFFFu) const;

  // (b) SHARD: decide the placement/topology -- how to split the artifact across
  // nodes. The single-node reference returns one shard (the whole pack); the
  // distributed backend would return many. The artifact is read-only input.
  virtual std::vector<Shard> shard(const std::uint8_t* data,
                                   std::size_t len) const = 0;

  // (c) DISPATCH + RE-ENTER: run the artifact (or each shard) through the existing
  // single-node C kernels and collect the observable dispatch result. With
  // max_retries > 0 a backend MAY re-dispatch a failed shard (idempotent: the
  // artifact bytes are immutable, so a retry recomputes the same deterministic
  // result -- it never accumulates state). The reference is single-attempt.
  virtual DispatchResult dispatch(const std::uint8_t* data, std::size_t len,
                                  unsigned max_retries = 0) const = 0;
};

// --- (REAL) the single-node reference implementation -----------------------
//
// Proves the seam round-trips. It consumes the artifact, places it as one shard on
// node 0, and re-enters the EXISTING freestanding C decoder/kernels
// (bcir_sp_for_each_segment in bcir_runtime.c) to produce the dispatch order. This
// is exactly the direct C/IR path with the Orchestrator interface wrapped around
// it -- so its result is identical to calling the C kernels directly (the round-
// trip identity the smoke test asserts). No dynamic topology, no networking.
class SingleNodeOrchestrator : public Orchestrator {
 public:
  std::string name() const override { return "single-node-reference"; }
  std::vector<Shard> shard(const std::uint8_t* data, std::size_t len) const override;
  DispatchResult dispatch(const std::uint8_t* data, std::size_t len,
                          unsigned max_retries = 0) const override;
};

// --- (STUB) dynamic-graph topology backend ---------------------------------
//
// NOT BUILT. Documents the seam for work that creates graph nodes/edges at RUNTIME
// -- RL agents spawning network nodes, transformers allocating mixed-length token
// graphs on the fly. A real implementation would: own a mutable graph-builder in
// C++ (OO + RAII + the STL the flat C registry deliberately lacks); materialize a
// *fresh* claim-graph each step; FREEZE it to a StreamPack via the C/IR rail; then
// hand THAT immutable artifact back through SingleNodeOrchestrator. The dynamic
// part lives entirely in C++ ABOVE the rail; what crosses down is always a frozen
// artifact. This stub holds the seam open without pulling that machinery in.
class DynamicGraphOrchestrator : public Orchestrator {
 public:
  std::string name() const override { return "dynamic-graph-STUB"; }
  std::vector<Shard> shard(const std::uint8_t* data, std::size_t len) const override;
  // Throws HandoffError: a real dynamic-graph backend is not built (it needs the
  // runtime graph-builder above). Marked so dead code can never silently "work".
  DispatchResult dispatch(const std::uint8_t* data, std::size_t len,
                          unsigned max_retries = 0) const override;
};

// --- (STUB) distributed multi-node backend ---------------------------------
//
// NOT BUILT. Documents the seam for MPI/NCCL multi-node orchestration: networking,
// failure recovery, job queuing, thread pooling. A real implementation would:
// partition the segment stream into shards, place each on a node (MPI rank / NCCL
// communicator), dispatch each shard into THAT node's local SingleNodeOrchestrator
// (the re-entry), and reduce the results -- with retry/replication on a node
// failure. Crucially it would add a REAL MPI/NCCL dependency + multi-node hardware,
// which we deliberately do NOT add (untested debt). This stub records exactly that
// design and fails loudly if dispatched, so the boundary is explicit and honest.
class DistributedOrchestrator : public Orchestrator {
 public:
  explicit DistributedOrchestrator(std::size_t world_size = 1)
      : world_size_(world_size) {}
  std::string name() const override { return "distributed-MPI/NCCL-STUB"; }
  // The sharding LOGIC is real + testable (it partitions the segment stream into
  // world_size contiguous ranges) -- it is the only piece that needs no cluster.
  std::vector<Shard> shard(const std::uint8_t* data, std::size_t len) const override;
  // Throws HandoffError: real cross-node dispatch needs MPI/NCCL + a cluster.
  DispatchResult dispatch(const std::uint8_t* data, std::size_t len,
                          unsigned max_retries = 0) const override;

 private:
  std::size_t world_size_;
};

// --- a factory for the seam (the placement decision point) -----------------
//
// In a real deployment the C++ layer inspects the workload (static known-shape +
// single-node -> the C/IR rail; dynamic topology -> dynamic backend; too big for
// one node -> distributed) and picks a backend. The single-node reference is the
// only one that is built; the others are returned so the seam is exercised but
// fail loudly on dispatch.
enum class Backend { SingleNode, DynamicGraph, Distributed };

std::unique_ptr<Orchestrator> make_orchestrator(Backend b,
                                                std::size_t world_size = 1);

}  // namespace bcir

#endif  // BCIR_ORCHESTRATOR_HPP
