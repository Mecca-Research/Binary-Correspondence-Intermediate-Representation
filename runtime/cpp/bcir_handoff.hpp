//===- bcir_handoff.hpp - the data-plane hand-off: RAII over the C pack table ---===//
//
// The C++ half of the G16 seam (GEM+ roadmap G16, staged plan S3-C). The freestanding C pack
// table (runtime/c/bcir_handoff.h) decides every operation; these types only give its handles a
// C++ lifetime, so the rules the table enforces at run time are also the shape of the code:
//
//   PackArena    owns the table's slots and arena (shared_ptr: views hold it weakly, so a view
//                that outlives its arena is refused, never dereferenced). One mutex serializes
//                every table operation, so an arena may be shared across threads.
//   Reservation  a producer's slot: written ONCE through data(), then commit() -> a PackOwner or
//                abort(); destroyed uncommitted, it aborts -- nothing half-built is published.
//   PackOwner    move-only owner of one frozen artifact; release() (or the destructor) ends the
//                incarnation: every view of it is refused from then on (BCIR_ERR_LIFETIME).
//   PackView     copyable and non-owning; it may outlive its owner, and then every borrow is
//                refused -- it never reads freed or reused bytes.
//   Borrow       move-only RAII pin: data()/size() stay valid for its lifetime even if the owner
//                releases meanwhile (the slot retires and its last borrow frees it); returned
//                exactly once, by give_back() or the destructor.
//   GraphBuilder the dynamic-graph builder: a step's mutable claim graph above the rail, frozen
//                THROUGH the rail (bcir_plan_func + bcir_hydrate_generations) straight into a
//                reserved slot -- the only write the artifact's bytes ever get.
//
// Legality stays the rails' verdict (the two-truth quarantine): admission is the live control
// plane's predicate (bcir_ctl_admit_pack via bcir_ho_admit), and every refusal is a bcir_status
// carried up in a HandoffResult, never a C++ exception.
//===----------------------------------------------------------------------===//
#ifndef BCIR_HANDOFF_HPP
#define BCIR_HANDOFF_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include "bcir_control_plane.h"
#include "bcir_handoff.h"
#include "bcir_hydrate.h"
#include "bcir_plan.h"
#include "bcir_shard_manifest.h"
}

namespace bcir {

// One table operation's outcome, verbatim from the C twin.
struct HandoffResult {
  bcir_ho_outcome outcome{};

  bool ok() const { return outcome.verdict == BCIR_HO_APPLIED; }
  bcir_status status() const { return outcome.status; }
  bcir_ho_refusal refusal() const { return static_cast<bcir_ho_refusal>(outcome.refusal); }

  static HandoffResult refused(bcir_ho_refusal refusal, bcir_status status,
                               std::uint32_t generation = 0);
};

class PackArena;
class PackOwner;

// A borrow in progress (RAII pin).
class Borrow {
public:
  Borrow() = default;
  Borrow(Borrow &&other) noexcept;
  Borrow &operator=(Borrow &&other) noexcept;
  Borrow(const Borrow &) = delete;
  Borrow &operator=(const Borrow &) = delete;
  ~Borrow();

  const HandoffResult &result() const { return result_; }
  bool ok() const { return result_.ok() && arena_ != nullptr; }
  const std::uint8_t *data() const { return ok() ? view_.data : nullptr; }
  std::size_t size() const { return ok() ? view_.len : 0u; }
  // Return the pin now (the destructor does it otherwise). A second call is refused
  // (BCIR_ERR_LIFETIME) without touching the table: this pin is already returned.
  HandoffResult give_back();

private:
  friend class PackView;
  std::shared_ptr<PackArena> arena_; // the bytes live in the arena: held while pinned
  bcir_ho_view view_{};
  HandoffResult result_{};
};

// A non-owning view of one incarnation.
class PackView {
public:
  PackView() = default;
  // Pin the artifact: refused (BCIR_ERR_LIFETIME) when its owner released it, its slot was
  // reused, or its arena is gone.
  Borrow borrow() const;
  bcir_ho_handle handle() const { return handle_; }
  std::shared_ptr<PackArena> arena() const { return arena_.lock(); }

private:
  friend class PackOwner;
  friend class Reservation;
  std::weak_ptr<PackArena> arena_;
  bcir_ho_handle handle_{};
};

// The move-only owner of one frozen artifact.
class PackOwner {
public:
  PackOwner() = default;
  PackOwner(PackOwner &&other) noexcept;
  PackOwner &operator=(PackOwner &&other) noexcept;
  PackOwner(const PackOwner &) = delete;
  PackOwner &operator=(const PackOwner &) = delete;
  ~PackOwner();

  bool valid() const { return handle_.epoch != 0u; }
  PackView view() const;
  bcir_ho_handle handle() const { return handle_; }
  // End the incarnation; a second call (or one on an empty owner) is refused.
  HandoffResult release();

private:
  friend class Reservation;
  std::weak_ptr<PackArena> arena_;
  bcir_ho_handle handle_{};
};

// A producer's reserved slot.
class Reservation {
public:
  Reservation() = default;
  Reservation(Reservation &&other) noexcept;
  Reservation &operator=(Reservation &&other) noexcept;
  Reservation(const Reservation &) = delete;
  Reservation &operator=(const Reservation &) = delete;
  ~Reservation(); // an uncommitted reservation aborts

  bool live() const { return handle_.epoch != 0u; }
  bcir_ho_handle handle() const { return handle_; }
  // The write region (nullptr, 0 once committed or aborted).
  std::uint8_t *data();
  std::size_t capacity();
  // Write `bytes` at `offset` (refused: LIFETIME once spent, CAPACITY past the region).
  HandoffResult write(std::size_t offset, const std::uint8_t *bytes, std::size_t n);
  // Freeze the first `length` bytes; the reservation is spent either way it succeeds.
  HandoffResult commit(std::size_t length, PackOwner &owner);
  HandoffResult abort();

private:
  friend class PackArena;
  std::weak_ptr<PackArena> arena_;
  bcir_ho_handle handle_{};
};

class PackArena : public std::enable_shared_from_this<PackArena> {
public:
  // `slots` slots of `slot_capacity` bytes. Throws std::invalid_argument on a shape the C table
  // refuses (it is a construction error, not a verdict).
  static std::shared_ptr<PackArena> create(std::uint32_t slots, std::size_t slot_capacity);
  PackArena(const PackArena &) = delete;
  PackArena &operator=(const PackArena &) = delete;

  // Reserve the lowest free slot (refused: FULL).
  HandoffResult reserve(Reservation &out);

  // Admission against the LIVE plane and dispatch as a plane phase (bcir_ho_admit /
  // bcir_ho_dispatch). The plane is the caller's: it serializes the plane across arenas.
  HandoffResult admit(const PackView &pack, bcir_ctl_state &plane);
  HandoffResult dispatch(const PackView &pack, bcir_ctl_state &plane, bcir_seg_fn fn, void *ctx);

  std::uint32_t slots() const { return table_.n_slots; }
  std::size_t slot_capacity() const { return table_.capacity; }
  std::array<std::uint8_t, 32> state_digest() const;

private:
  friend class Borrow;
  friend class PackView;
  friend class PackOwner;
  friend class Reservation;
  PackArena() = default;

  mutable std::mutex mu_;
  std::vector<bcir_ho_slot> slots_;
  std::vector<std::uint8_t> bytes_;
  bcir_ho_table table_{};
};

// Admission / dispatch of a view whose arena may be gone: LIFETIME, never a dangling access.
HandoffResult admit_view(const PackView &pack, bcir_ctl_state &plane);
HandoffResult dispatch_view(const PackView &pack, bcir_ctl_state &plane, bcir_seg_fn fn, void *ctx);

// --- the dynamic-graph builder -----------------------------------------------------------

// One step's mutable claim graph. Claim ids are assigned ascending in creation order (the
// canonical order bcir_hydrate_generations requires); clear() starts the next step's graph.
class GraphBuilder {
public:
  struct ClaimSpec {
    bcir_opcode opcode = BCIR_OP_ADD;
    std::uint8_t lane = 0;
    bcir_domain domain = BCIR_DOM_RAM;
    std::uint32_t count = 1;
    std::vector<std::uint32_t> reads;
    std::vector<std::uint32_t> writes;
    std::string label = "c.add";
  };

  std::uint32_t add(const ClaimSpec &spec);
  // Append a claim with an explicit id (the builder then continues after it) -- a harness
  // replaying another rail's graph uses this; ids must still ascend at freeze.
  void add_with_id(std::uint32_t id, const ClaimSpec &spec);
  void clear();
  std::size_t size() const { return claims_.size(); }

  // FREEZE through the C/IR rail: reserve a slot, plan (bcir_plan_func) and hydrate
  // (bcir_hydrate_generations) straight into it, commit. Transactional: any failure aborts the
  // slot -- nothing published, nothing leaked -- and carries the rail's status.
  HandoffResult freeze(PackArena &arena, const std::vector<bcir_generation_view> &gens,
                       std::uint32_t topo_gen, PackOwner &owner) const;

private:
  std::vector<bcir_claim> claims_;
  std::uint32_t next_id_ = 1;
};

// --- the manifest-of-shards -------------------------------------------------------------

// A whole pack as a frame and its shards, each frozen into an arena slot, and the manifest that
// binds them by digest (bcir_shard_manifest.h).
struct ShardSet {
  HandoffResult result;
  std::vector<std::uint8_t> manifest;
  PackOwner frame;
  std::vector<PackOwner> shards;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
};

// Cut `whole` over `ranges` (each output written once, into its own slot of `arena`).
ShardSet cut_shards(const PackView &whole, PackArena &arena,
                    const std::vector<std::pair<std::uint32_t, std::uint32_t>> &ranges);

// Reassemble a manifest's whole from views of its frame and shards (any order: they are found by
// digest) into a fresh slot of `arena`; refused with the manifest's status or BCIR_ERR_SHARD.
HandoffResult reassemble_shards(const std::vector<std::uint8_t> &manifest,
                                const std::vector<PackView> &blobs, PackArena &arena,
                                PackOwner &whole);

} // namespace bcir

#endif // BCIR_HANDOFF_HPP
