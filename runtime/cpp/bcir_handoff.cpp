//===- bcir_handoff.cpp - the data-plane hand-off: RAII over the C pack table ---===//
//
// See bcir_handoff.hpp. Every decision is the C twin's (bcir_handoff.c); this file only binds
// handles to C++ lifetimes and takes the arena's mutex around each table operation. A lock is
// never held while another handle's destructor could run (a destructor takes the same lock), so
// every assignment that may destroy a live handle happens before the lock is taken.
//===----------------------------------------------------------------------===//
#include "bcir_handoff.hpp"

#include <cstring>
#include <stdexcept>

namespace bcir {

namespace {

bcir_ho_outcome made(std::uint8_t verdict, std::uint8_t refusal, bcir_status status,
                     std::uint32_t generation) {
  bcir_ho_outcome o{};
  o.verdict = verdict;
  o.refusal = refusal;
  o.generation = generation;
  o.status = status;
  return o;
}

HandoffResult lifetime() {
  return HandoffResult::refused(BCIR_HO_REFUSAL_LIFETIME, BCIR_ERR_LIFETIME);
}

// The arena a view names, if it is still alive and is the arena it claims.
std::shared_ptr<PackArena> arena_of(const std::weak_ptr<PackArena> &weak) {
  return weak.lock();
}

} // namespace

HandoffResult HandoffResult::refused(bcir_ho_refusal refusal, bcir_status status,
                                     std::uint32_t generation) {
  HandoffResult r;
  r.outcome = made(BCIR_HO_REFUSED, static_cast<std::uint8_t>(refusal), status, generation);
  return r;
}

// --- Borrow ----------------------------------------------------------------------------------

Borrow::Borrow(Borrow &&other) noexcept
    : arena_(std::move(other.arena_)), view_(other.view_), result_(other.result_) {
  other.arena_.reset();
  other.view_ = bcir_ho_view{};
  other.result_ = lifetime();
}

Borrow &Borrow::operator=(Borrow &&other) noexcept {
  if (this != &other) {
    give_back();
    arena_ = std::move(other.arena_);
    view_ = other.view_;
    result_ = other.result_;
    other.arena_.reset();
    other.view_ = bcir_ho_view{};
    other.result_ = lifetime();
  }
  return *this;
}

Borrow::~Borrow() {
  give_back();
}

HandoffResult Borrow::give_back() {
  if (!arena_)
    return lifetime(); // already returned (or never pinned): the table is untouched
  HandoffResult r;
  {
    std::lock_guard<std::mutex> guard(arena_->mu_);
    r.outcome = bcir_ho_give_back(&arena_->table_, &view_);
  }
  arena_.reset();
  view_ = bcir_ho_view{};
  return r;
}

// --- PackView --------------------------------------------------------------------------------

Borrow PackView::borrow() const {
  Borrow b;
  std::shared_ptr<PackArena> arena = arena_of(arena_);
  if (!arena) {
    b.result_ = lifetime(); // the arena is gone: refused, never dereferenced
    return b;
  }
  {
    std::lock_guard<std::mutex> guard(arena->mu_);
    b.result_.outcome = bcir_ho_borrow(&arena->table_, handle_, &b.view_);
  }
  if (b.result_.ok())
    b.arena_ = std::move(arena);
  return b;
}

// --- PackOwner -------------------------------------------------------------------------------

PackOwner::PackOwner(PackOwner &&other) noexcept
    : arena_(std::move(other.arena_)), handle_(other.handle_) {
  other.arena_.reset();
  other.handle_ = bcir_ho_handle{};
}

PackOwner &PackOwner::operator=(PackOwner &&other) noexcept {
  if (this != &other) {
    release();
    arena_ = std::move(other.arena_);
    handle_ = other.handle_;
    other.arena_.reset();
    other.handle_ = bcir_ho_handle{};
  }
  return *this;
}

PackOwner::~PackOwner() {
  release();
}

PackView PackOwner::view() const {
  PackView v;
  v.arena_ = arena_;
  v.handle_ = handle_;
  return v;
}

HandoffResult PackOwner::release() {
  if (!valid())
    return lifetime();
  std::shared_ptr<PackArena> arena = arena_of(arena_);
  bcir_ho_handle h = handle_;
  arena_.reset();
  handle_ = bcir_ho_handle{};
  if (!arena)
    return lifetime();
  HandoffResult r;
  std::lock_guard<std::mutex> guard(arena->mu_);
  r.outcome = bcir_ho_release(&arena->table_, h);
  return r;
}

// --- Reservation -----------------------------------------------------------------------------

Reservation::Reservation(Reservation &&other) noexcept
    : arena_(std::move(other.arena_)), handle_(other.handle_) {
  other.arena_.reset();
  other.handle_ = bcir_ho_handle{};
}

Reservation &Reservation::operator=(Reservation &&other) noexcept {
  if (this != &other) {
    abort();
    arena_ = std::move(other.arena_);
    handle_ = other.handle_;
    other.arena_.reset();
    other.handle_ = bcir_ho_handle{};
  }
  return *this;
}

Reservation::~Reservation() {
  abort();
}

std::uint8_t *Reservation::data() {
  std::shared_ptr<PackArena> arena = arena_of(arena_);
  if (!live() || !arena)
    return nullptr;
  std::uint8_t *p = nullptr;
  std::size_t cap = 0;
  std::lock_guard<std::mutex> guard(arena->mu_);
  return bcir_ho_reserved(&arena->table_, handle_, &p, &cap) == BCIR_OK ? p : nullptr;
}

std::size_t Reservation::capacity() {
  std::shared_ptr<PackArena> arena = arena_of(arena_);
  if (!live() || !arena)
    return 0u;
  std::uint8_t *p = nullptr;
  std::size_t cap = 0;
  std::lock_guard<std::mutex> guard(arena->mu_);
  return bcir_ho_reserved(&arena->table_, handle_, &p, &cap) == BCIR_OK ? cap : 0u;
}

HandoffResult Reservation::write(std::size_t offset, const std::uint8_t *bytes, std::size_t n) {
  std::shared_ptr<PackArena> arena = arena_of(arena_);
  if (!arena)
    return lifetime();
  std::uint8_t *p = nullptr;
  std::size_t cap = 0;
  std::lock_guard<std::mutex> guard(arena->mu_);
  if (bcir_ho_reserved(&arena->table_, handle_, &p, &cap) != BCIR_OK)
    return lifetime();
  if (offset > cap || n > cap - offset)
    return HandoffResult::refused(BCIR_HO_REFUSAL_CAPACITY, BCIR_ERR_NOSPACE);
  if (n)
    std::memcpy(p + offset, bytes, n);
  HandoffResult r;
  r.outcome = made(BCIR_HO_APPLIED, BCIR_HO_REFUSAL_NONE, BCIR_OK, 0u);
  r.outcome.handle = handle_;
  return r;
}

HandoffResult Reservation::commit(std::size_t length, PackOwner &owner) {
  owner = PackOwner(); // release a held owner before the lock is taken
  std::shared_ptr<PackArena> arena = arena_of(arena_);
  if (!arena) {
    handle_ = bcir_ho_handle{};
    return lifetime();
  }
  HandoffResult r;
  {
    std::lock_guard<std::mutex> guard(arena->mu_);
    r.outcome = bcir_ho_commit(&arena->table_, handle_, length);
  }
  if (r.ok()) {
    owner.arena_ = arena_;
    owner.handle_ = handle_;
  }
  if (r.ok() || r.refusal() == BCIR_HO_REFUSAL_LIFETIME) { // spent; a CAPACITY refusal is not
    arena_.reset();
    handle_ = bcir_ho_handle{};
  }
  return r;
}

HandoffResult Reservation::abort() {
  if (!live())
    return lifetime();
  std::shared_ptr<PackArena> arena = arena_of(arena_);
  bcir_ho_handle h = handle_;
  arena_.reset();
  handle_ = bcir_ho_handle{};
  if (!arena)
    return lifetime();
  HandoffResult r;
  std::lock_guard<std::mutex> guard(arena->mu_);
  r.outcome = bcir_ho_abort(&arena->table_, h);
  return r;
}

// --- PackArena -------------------------------------------------------------------------------

std::shared_ptr<PackArena> PackArena::create(std::uint32_t slots, std::size_t slot_capacity) {
  if (slots == 0u || slot_capacity == 0u || slot_capacity > SIZE_MAX / slots)
    throw std::invalid_argument("PackArena: a shape the pack table refuses");
  std::shared_ptr<PackArena> arena(new PackArena());
  arena->slots_.resize(slots);
  arena->bytes_.resize(static_cast<std::size_t>(slots) * slot_capacity);
  if (bcir_ho_init(&arena->table_, arena->slots_.data(), slots, arena->bytes_.data(),
                   arena->bytes_.size(), slot_capacity) != BCIR_OK)
    throw std::invalid_argument("PackArena: a shape the pack table refuses");
  return arena;
}

HandoffResult PackArena::reserve(Reservation &out) {
  out = Reservation(); // abort a live reservation before the lock is taken
  HandoffResult r;
  {
    std::lock_guard<std::mutex> guard(mu_);
    r.outcome = bcir_ho_reserve(&table_);
  }
  if (r.ok()) {
    out.arena_ = weak_from_this();
    out.handle_ = r.outcome.handle;
  }
  return r;
}

HandoffResult PackArena::admit(const PackView &pack, bcir_ctl_state &plane) {
  if (pack.arena().get() != this)
    return lifetime(); // another arena's handle names another table
  HandoffResult r;
  std::lock_guard<std::mutex> guard(mu_);
  r.outcome = bcir_ho_admit(&table_, pack.handle(), &plane);
  return r;
}

HandoffResult PackArena::dispatch(const PackView &pack, bcir_ctl_state &plane, bcir_seg_fn fn,
                                  void *ctx) {
  if (pack.arena().get() != this)
    return lifetime();
  HandoffResult r;
  std::lock_guard<std::mutex> guard(mu_); // version zero: a dispatch holds the arena for its walk
  r.outcome = bcir_ho_dispatch(&table_, pack.handle(), &plane, fn, ctx);
  return r;
}

std::array<std::uint8_t, 32> PackArena::state_digest() const {
  std::array<std::uint8_t, 32> out{};
  std::lock_guard<std::mutex> guard(mu_);
  bcir_ho_state_digest(&table_, out.data());
  return out;
}

HandoffResult admit_view(const PackView &pack, bcir_ctl_state &plane) {
  std::shared_ptr<PackArena> arena = pack.arena();
  return arena ? arena->admit(pack, plane) : lifetime();
}

HandoffResult dispatch_view(const PackView &pack, bcir_ctl_state &plane, bcir_seg_fn fn,
                            void *ctx) {
  std::shared_ptr<PackArena> arena = pack.arena();
  return arena ? arena->dispatch(pack, plane, fn, ctx) : lifetime();
}

// --- GraphBuilder ----------------------------------------------------------------------------

namespace {

bcir_claim claim_of(std::uint32_t id, const GraphBuilder::ClaimSpec &spec) {
  bcir_claim c{};
  c.id = id;
  c.opcode = spec.opcode;
  c.lane = spec.lane;
  c.domain = spec.domain;
  c.count = spec.count;
  c.n_rd = static_cast<std::uint8_t>(spec.reads.size() > 255u ? 255u : spec.reads.size());
  for (std::size_t i = 0; i < spec.reads.size() && i < BCIR_CLAIM_MAX_RD; ++i)
    c.rd[i] = spec.reads[i];
  c.n_wr = static_cast<std::uint8_t>(spec.writes.size() > 255u ? 255u : spec.writes.size());
  for (std::size_t i = 0; i < spec.writes.size() && i < BCIR_CLAIM_MAX_WR; ++i)
    c.wr[i] = spec.writes[i];
  // A label of 32 bytes or more keeps no terminator, so the rail's label law refuses it.
  std::size_t n = spec.label.size() < sizeof c.op ? spec.label.size() : sizeof c.op;
  std::memcpy(c.op, spec.label.data(), n);
  if (n < sizeof c.op)
    c.op[n] = '\0';
  return c;
}

} // namespace

std::uint32_t GraphBuilder::add(const ClaimSpec &spec) {
  std::uint32_t id = next_id_++;
  claims_.push_back(claim_of(id, spec));
  return id;
}

void GraphBuilder::add_with_id(std::uint32_t id, const ClaimSpec &spec) {
  claims_.push_back(claim_of(id, spec));
  next_id_ = id + 1u;
}

void GraphBuilder::clear() {
  claims_.clear();
  next_id_ = 1;
}

HandoffResult GraphBuilder::freeze(PackArena &arena, const std::vector<bcir_generation_view> &gens,
                                   std::uint32_t topo_gen, PackOwner &owner) const {
  owner = PackOwner();
  Reservation slot;
  HandoffResult r = arena.reserve(slot);
  if (!r.ok())
    return r;
  bcir_func f{};
  f.claims =
      const_cast<bcir_claim *>(claims_.data()); // the rail reads it through a const bcir_func*
  f.n_claims = claims_.size();
  f.cap_claims = claims_.size();
  std::vector<bcir_plan_step> steps(claims_.size());
  bcir_plan plan{};
  std::size_t len = 0;
  bcir_status st = bcir_plan_func(&f, steps.data(), steps.size(), &plan);
  if (st == BCIR_OK)
    st = bcir_hydrate_generations(&f, &plan, topo_gen, gens.empty() ? nullptr : gens.data(),
                                  gens.size(), slot.data(), slot.capacity(), &len);
  if (st != BCIR_OK) {
    slot.abort(); // transactional: nothing published, nothing leaked
    return HandoffResult::refused(
        st == BCIR_ERR_NOSPACE ? BCIR_HO_REFUSAL_CAPACITY : BCIR_HO_REFUSAL_MALFORMED, st);
  }
  return slot.commit(len, owner);
}

// --- the manifest-of-shards ------------------------------------------------------------------

namespace {

// Emit one output of `whole` into a fresh slot: `emit(out, cap, &len)`.
template <typename Emit>
HandoffResult emit_into(PackArena &arena, PackOwner &owner, Emit emit) {
  Reservation slot;
  HandoffResult r = arena.reserve(slot);
  if (!r.ok())
    return r;
  std::size_t len = 0;
  bcir_status st = emit(slot.data(), slot.capacity(), &len);
  if (st != BCIR_OK) {
    slot.abort();
    return HandoffResult::refused(
        st == BCIR_ERR_NOSPACE ? BCIR_HO_REFUSAL_CAPACITY : BCIR_HO_REFUSAL_MALFORMED, st);
  }
  return slot.commit(len, owner);
}

struct Store {
  std::vector<const std::uint8_t *> data;
  std::vector<std::size_t> len;
  std::vector<std::array<std::uint8_t, 32>> digest;
};

extern "C" int store_fetch(const std::uint8_t digest[32], const std::uint8_t **data,
                           std::size_t *len, void *ctx) {
  const Store *s = static_cast<const Store *>(ctx);
  for (std::size_t i = 0; i < s->data.size(); ++i)
    if (std::memcmp(s->digest[i].data(), digest, 32) == 0) {
      *data = s->data[i];
      *len = s->len[i];
      return 0;
    }
  return 1;
}

} // namespace

ShardSet cut_shards(const PackView &whole, PackArena &arena,
                    const std::vector<std::pair<std::uint32_t, std::uint32_t>> &ranges) {
  ShardSet out;
  out.ranges = ranges;
  Borrow src = whole.borrow();
  if (!src.ok()) {
    out.result = src.result();
    return out;
  }
  const std::uint8_t *w = src.data();
  std::size_t wlen = src.size();
  out.result = emit_into(arena, out.frame, [&](std::uint8_t *o, std::size_t cap, std::size_t *n) {
    return bcir_shm_frame(w, wlen, o, cap, n);
  });
  for (std::size_t i = 0; i < ranges.size() && out.result.ok(); ++i) {
    out.shards.emplace_back();
    const std::uint32_t b = ranges[i].first, e = ranges[i].second;
    out.result =
        emit_into(arena, out.shards.back(), [&](std::uint8_t *o, std::size_t cap, std::size_t *n) {
          return bcir_shm_sub_pack(w, wlen, b, e, o, cap, n);
        });
  }
  if (out.result.ok()) {
    std::vector<Borrow> pins;
    std::vector<const std::uint8_t *> ptrs;
    std::vector<std::size_t> lens;
    std::vector<std::uint32_t> begins, ends;
    Borrow frame = out.frame.view().borrow();
    for (const PackOwner &s : out.shards) {
      pins.push_back(s.view().borrow());
      ptrs.push_back(pins.back().data());
      lens.push_back(pins.back().size());
    }
    for (const auto &r : ranges) {
      begins.push_back(r.first);
      ends.push_back(r.second);
    }
    std::size_t mlen = 0;
    const auto n = static_cast<std::uint32_t>(ranges.size());
    bcir_status st = bcir_shm_encode(w, wlen, frame.data(), frame.size(), begins.data(),
                                     ends.data(), ptrs.data(), lens.data(), n, nullptr, 0, &mlen);
    if (st == BCIR_OK) {
      out.manifest.resize(mlen);
      st = bcir_shm_encode(w, wlen, frame.data(), frame.size(), begins.data(), ends.data(),
                           ptrs.data(), lens.data(), n, out.manifest.data(), mlen, &mlen);
    }
    if (st != BCIR_OK) {
      out.manifest.clear();
      out.result = HandoffResult::refused(BCIR_HO_REFUSAL_MALFORMED, st);
    }
  }
  if (!out.result.ok()) { // nothing half-cut survives
    out.frame = PackOwner();
    out.shards.clear();
  }
  return out;
}

HandoffResult reassemble_shards(const std::vector<std::uint8_t> &manifest,
                                const std::vector<PackView> &blobs, PackArena &arena,
                                PackOwner &whole) {
  whole = PackOwner();
  std::vector<Borrow> pins;
  Store store;
  for (const PackView &v : blobs) {
    pins.push_back(v.borrow());
    if (!pins.back().ok())
      return pins.back().result(); // a dead view: refused as such
    store.data.push_back(pins.back().data());
    store.len.push_back(pins.back().size());
    store.digest.emplace_back();
    bcir_sha256_digest(pins.back().data(), pins.back().size(), store.digest.back().data());
  }
  return emit_into(arena, whole, [&](std::uint8_t *o, std::size_t cap, std::size_t *n) {
    return bcir_shm_reassemble(manifest.data(), manifest.size(), store_fetch, &store, o, cap, n);
  });
}

} // namespace bcir
