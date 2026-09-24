//===- test_handoff.cpp - the G16 C++ seam harness ---------------------------===//
//
// Drives the C++ hand-off seam (bcir_handoff.hpp, bcir_orchestrator.hpp) so the G16 rows run on
// the C++ rail; graded by bcir/tests/handoff_fixtures.py against the Python oracle and the C twin.
//
//   --script <file>  the C harness's scenario format (runtime/c/test_handoff.c), replayed through
//       the RAII types: one trace line per operation, identical to the other rails'. Operations
//       with no C++ spelling (a forged handle, a copied borrow, a poked counter) are refused as an
//       input error -- the grader never sends them (their laws are C and Python witnesses).
//   --shards <file>  per case (plane records, a whole pack, a world size): admit and dispatch the
//       whole; cut it (DistributedOrchestrator::cut); admit and dispatch every shard by itself;
//       reassemble. "OK <manifest hex> <frame hex> <shard hex>... | w=<whole claims> r=<ranks'
//       claims> a=<reassembled sha256> x=<copies>" or "ERR <stage> <status>".
//   --lifetime       the C++-only lifetime witnesses: "<name> ok|FAIL" per witness.
//
// A register that is overwritten keeps its old owner/borrow alive in a graveyard until the
// scenario ends: RAII must not release what the C rail's harness only forgets, or the table
// traces would differ for a reason no rail decided. The exit status is 0 when the input was
// well-formed; 2 on a usage or input error.
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "bcir_orchestrator.hpp"
#include "test_stage3.h"

namespace {

using namespace bcir;

const char *status_name(bcir_status s) {
  switch (s) {
  case BCIR_OK:
    return "BCIR_OK";
  case BCIR_ERR_TRUNCATED:
    return "BCIR_ERR_TRUNCATED";
  case BCIR_ERR_MAGIC:
    return "BCIR_ERR_MAGIC";
  case BCIR_ERR_VERSION:
    return "BCIR_ERR_VERSION";
  case BCIR_ERR_CRC:
    return "BCIR_ERR_CRC";
  case BCIR_ERR_NOSPACE:
    return "BCIR_ERR_NOSPACE";
  case BCIR_ERR_LANE:
    return "BCIR_ERR_LANE";
  case BCIR_ERR_WIDTH:
    return "BCIR_ERR_WIDTH";
  case BCIR_ERR_DISPATCH:
    return "BCIR_ERR_DISPATCH";
  case BCIR_ERR_PROVENANCE:
    return "BCIR_ERR_PROVENANCE";
  case BCIR_ERR_STALE:
    return "BCIR_ERR_STALE";
  case BCIR_ERR_OVERFLOW:
    return "BCIR_ERR_OVERFLOW";
  case BCIR_ERR_TRAILING:
    return "BCIR_ERR_TRAILING";
  case BCIR_ERR_RESERVED:
    return "BCIR_ERR_RESERVED";
  case BCIR_ERR_UTF8:
    return "BCIR_ERR_UTF8";
  case BCIR_ERR_GENERATION:
    return "BCIR_ERR_GENERATION";
  case BCIR_ERR_PLAN:
    return "BCIR_ERR_PLAN";
  case BCIR_ERR_CONTROL:
    return "BCIR_ERR_CONTROL";
  case BCIR_ERR_MAC:
    return "BCIR_ERR_MAC";
  case BCIR_ERR_TELEMETRY:
    return "BCIR_ERR_TELEMETRY";
  case BCIR_ERR_RING:
    return "BCIR_ERR_RING";
  case BCIR_ERR_FULL:
    return "BCIR_ERR_FULL";
  case BCIR_ERR_BUSY:
    return "BCIR_ERR_BUSY";
  case BCIR_ERR_LIFETIME:
    return "BCIR_ERR_LIFETIME";
  case BCIR_ERR_SHARD:
    return "BCIR_ERR_SHARD";
  }
  return "BCIR_ERR_UNKNOWN";
}

// --- input -----------------------------------------------------------------------------------

struct Reader {
  const std::vector<std::uint8_t> &d;
  std::size_t pos = 0;
  bool err = false;

  const std::uint8_t *take(std::size_t n) {
    if (err || n > d.size() - pos) {
      err = true;
      return nullptr;
    }
    const std::uint8_t *p = d.data() + pos;
    pos += n;
    return p;
  }
  std::uint8_t u8() {
    const std::uint8_t *p = take(1);
    return p ? p[0] : 0u;
  }
  std::uint32_t u32() {
    const std::uint8_t *p = take(4);
    return p ? static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
                   (static_cast<std::uint32_t>(p[2]) << 16) |
                   (static_cast<std::uint32_t>(p[3]) << 24)
             : 0u;
  }
  std::uint64_t u64() {
    std::uint64_t lo = u32();
    return lo | (static_cast<std::uint64_t>(u32()) << 32);
  }
  std::vector<std::uint8_t> blob() {
    std::uint32_t n = u32();
    const std::uint8_t *p = take(n);
    return p ? std::vector<std::uint8_t>(p, p + n) : std::vector<std::uint8_t>();
  }
};

bool read_file(const char *path, std::vector<std::uint8_t> &out) {
  std::FILE *f = std::fopen(path, "rb");
  if (!f)
    return false;
  int c;
  while ((c = std::fgetc(f)) != EOF)
    out.push_back(static_cast<std::uint8_t>(c));
  std::fclose(f);
  return true;
}

void hex(const std::uint8_t *p, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    std::printf("%02x", p[i]);
}

// The claims of a dispatch, digested (u64le ids), and every segment view held to the slot it came
// from: a byte outside it is a copy the seam made where a borrow sufficed.
struct Walk {
  bcir_sha256 h;
  const std::uint8_t *lo = nullptr, *hi = nullptr;
  std::size_t outside = 0;
};

bool inside(const Walk *w, const void *p, std::size_t n) {
  const auto *b = static_cast<const std::uint8_t *>(p);
  if (n == 0 || !w->lo)
    return true;
  return b >= w->lo && b <= w->hi && n <= static_cast<std::size_t>(w->hi - b);
}

extern "C" int walk_claims(const bcir_segment_view *seg, void *ctx) {
  auto *w = static_cast<Walk *>(ctx);
  std::uint8_t b[8];
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<std::uint8_t>(seg->claim_id >> (8 * i));
  bcir_sha256_update(&w->h, b, sizeof b);
  if (!inside(w, seg->name, seg->name_len) || !inside(w, seg->opcode, seg->opcode_len) ||
      !inside(w, seg->reads, static_cast<std::size_t>(seg->n_reads) * 4u) ||
      !inside(w, seg->writes, static_cast<std::size_t>(seg->n_writes) * 4u))
    w->outside++;
  return 0;
}

// Dispatch a view with its slot as the walk's bounds (the pin taken to read the bounds is
// returned before the dispatch, so the table is untouched by it). `w.h` is the caller's running
// digest of claim ids: one dispatch, or every rank's in order.
HandoffResult dispatch_checked(const PackView &v, bcir_ctl_state &plane, Walk &w) {
  w.lo = nullptr;
  w.hi = nullptr;
  {
    Borrow b = v.borrow();
    if (b.ok()) {
      w.lo = b.data();
      w.hi = b.data() + b.size();
    }
  }
  return dispatch_view(v, plane, &walk_claims, &w);
}

// A step's claim graph in the harness encoding (see runtime/c/test_handoff.c read_graph).
bool read_graph(Reader &r, GraphBuilder &g, std::vector<bcir_generation_view> &gens,
                std::uint32_t &topo) {
  topo = r.u32();
  std::uint32_t n_gens = r.u32();
  if (r.err || n_gens > 1000000u)
    return false;
  for (std::uint32_t i = 0; i < n_gens; ++i) {
    bcir_generation_view v{};
    v.rid = r.u32();
    v.map_gen = r.u32();
    v.data_gen = r.u32();
    gens.push_back(v);
  }
  std::uint32_t n = r.u32();
  if (r.err || n > 1000000u)
    return false;
  for (std::uint32_t i = 0; i < n && !r.err; ++i) {
    GraphBuilder::ClaimSpec s;
    std::uint32_t id = r.u32();
    s.opcode = static_cast<bcir_opcode>(r.u8());
    s.lane = r.u8();
    s.domain = static_cast<bcir_domain>(r.u8());
    s.count = r.u32();
    std::uint8_t n_rd = r.u8();
    for (std::uint8_t k = 0; k < n_rd; ++k)
      s.reads.push_back(r.u32());
    std::uint8_t n_wr = r.u8();
    for (std::uint8_t k = 0; k < n_wr; ++k)
      s.writes.push_back(r.u32());
    std::uint8_t ll = r.u8();
    const std::uint8_t *label = r.take(ll);
    if (!label)
      return false;
    s.label.assign(reinterpret_cast<const char *>(label), ll);
    g.add_with_id(id, s);
  }
  return !r.err;
}

// --- --script ------------------------------------------------------------------------------

enum {
  OP_SUBMIT = 1,
  OP_ENTER = 2,
  OP_LEAVE = 3,
  OP_ADVANCE = 4,
  OP_RESERVE = 5,
  OP_WRITE = 6,
  OP_COMMIT = 7,
  OP_ABORT = 8,
  OP_BORROW = 9,
  OP_GIVE_BACK = 10,
  OP_RELEASE = 11,
  OP_ADMIT = 12,
  OP_DISPATCH = 13,
  OP_ADMIT_MANIFEST = 14,
  OP_FREEZE = 15,
  OP_RESTART = 20
};
constexpr std::size_t kRegs = 16;

HandoffResult plane_step(bcir_ctl_outcome c) {
  HandoffResult r;
  r.outcome.verdict = c.verdict == BCIR_CTL_REFUSED ? BCIR_HO_REFUSED : BCIR_HO_APPLIED;
  r.outcome.generation = c.generation;
  r.outcome.status = c.status;
  return r;
}

void trace(std::uint32_t scn, std::uint32_t step, const HandoffResult &r, int op,
           const std::uint8_t *claims, const PackArena &arena, const bcir_ctl_state &plane) {
  const bcir_ho_outcome &o = r.outcome;
  bool hide = (op == OP_ADMIT || op == OP_DISPATCH) &&
              (o.refusal == BCIR_HO_REFUSAL_MALFORMED || o.refusal == BCIR_HO_REFUSAL_WALK);
  std::array<std::uint8_t, 32> td = arena.state_digest();
  std::uint8_t pd[32];
  bcir_ctl_state_digest(&plane, pd);
  std::printf("%u %u %u %u %s ", scn, step, o.verdict, o.refusal,
              hide ? "-" : status_name(o.status));
  if (o.handle.epoch)
    std::printf("h=%u:%u ", o.handle.index, o.handle.epoch);
  else
    std::printf("h=- ");
  std::printf("g=%u c=", o.generation);
  if (claims)
    hex(claims, 16);
  else
    std::printf("-");
  std::printf(" t=");
  hex(td.data(), 32);
  std::printf(" p=");
  hex(pd, 32);
  std::printf("\n");
}

int run_script(const char *path) {
  std::size_t copies = 0; // segment views read outside the dispatched slot, over every dispatch
  std::vector<std::uint8_t> data;
  if (!read_file(path, data))
    return 2;
  Reader r{data};
  const std::uint8_t *magic = r.take(4);
  if (!magic || std::memcmp(magic, "BHOS", 4) != 0)
    return 2;
  std::uint32_t n_scn = r.u32();
  for (std::uint32_t scn = 0; scn < n_scn && !r.err; ++scn) {
    std::uint32_t n_slots = r.u32(), capacity = r.u32();
    std::vector<std::uint8_t> key = r.blob();
    std::uint8_t scope = r.u8();
    std::uint64_t subject = r.u64();
    std::uint32_t n_steps = r.u32();
    if (r.err || n_slots == 0u || n_slots > 4096u || capacity == 0u || capacity > (1u << 24))
      return 2;
    std::shared_ptr<PackArena> arena = PackArena::create(n_slots, capacity);
    bcir_ctl_state plane;
    if (bcir_ctl_init(&plane, key.data(), key.size(), scope, subject) != BCIR_OK)
      return 2;
    std::array<Reservation, kRegs> res;
    std::array<PackOwner, kRegs> owners;
    std::array<PackView, kRegs> views;
    std::array<Borrow, kRegs> borrows;
    std::vector<PackOwner> dead_owners; // forgotten, never released, until the scenario ends
    std::vector<Reservation> dead_res;
    std::vector<Borrow> dead_borrows;
    for (std::uint32_t step = 0; step < n_steps && !r.err; ++step) {
      std::uint8_t op = r.u8(), reg = r.u8(), vreg = r.u8();
      std::vector<std::uint8_t> d = r.blob();
      if (r.err || reg >= kRegs || vreg >= kRegs)
        return 2;
      HandoffResult out;
      std::uint8_t claims[32];
      const std::uint8_t *claims_p = nullptr;
      switch (op) {
      case OP_SUBMIT:
        out = plane_step(bcir_ctl_submit(&plane, d.data(), d.size()));
        break;
      case OP_ENTER:
        out = plane_step(bcir_ctl_enter(&plane));
        break;
      case OP_LEAVE:
        out = plane_step(bcir_ctl_leave(&plane));
        break;
      case OP_ADVANCE:
        out = plane_step(bcir_ctl_advance(&plane));
        break;
      case OP_RESTART: // the plane restarts: generation 0, no registry, the same key
        if (bcir_ctl_init(&plane, key.data(), key.size(), scope, subject) != BCIR_OK)
          return 2;
        out.outcome.verdict = BCIR_HO_APPLIED;
        out.outcome.generation = plane.generation;
        out.outcome.status = BCIR_OK;
        break;
      case OP_RESERVE: {
        Reservation fresh;
        out = arena->reserve(fresh);
        if (out.ok()) {
          dead_res.push_back(std::move(res[reg]));
          res[reg] = std::move(fresh);
          dead_owners.push_back(std::move(owners[reg]));
          views[reg] = PackView();
        }
        break;
      }
      case OP_WRITE: {
        if (d.size() < 4)
          return 2;
        std::uint32_t offset =
            static_cast<std::uint32_t>(d[0]) | (static_cast<std::uint32_t>(d[1]) << 8) |
            (static_cast<std::uint32_t>(d[2]) << 16) | (static_cast<std::uint32_t>(d[3]) << 24);
        out = res[reg].write(offset, d.data() + 4, d.size() - 4);
        break;
      }
      case OP_COMMIT: {
        Reader w{d};
        std::uint64_t length = w.u64();
        if (w.err)
          return 2;
        PackOwner fresh;
        out = res[reg].commit(static_cast<std::size_t>(length), fresh);
        if (out.ok()) {
          dead_owners.push_back(std::move(owners[reg]));
          owners[reg] = std::move(fresh);
          views[reg] = owners[reg].view();
        }
        break;
      }
      case OP_ABORT:
        out = res[reg].abort();
        break;
      case OP_BORROW: {
        Borrow fresh = views[reg].borrow();
        out = fresh.result();
        dead_borrows.push_back(std::move(borrows[vreg]));
        borrows[vreg] = std::move(fresh);
        break;
      }
      case OP_GIVE_BACK:
        out = borrows[vreg].give_back();
        break;
      case OP_RELEASE:
        out = owners[reg].release();
        break;
      case OP_ADMIT:
        out = admit_view(views[reg], plane);
        break;
      case OP_DISPATCH: {
        Walk w;
        bcir_sha256_init(&w.h);
        out = dispatch_checked(views[reg], plane, w);
        copies += w.outside;
        bcir_sha256_final(&w.h, claims);
        if (out.ok())
          claims_p = claims;
        break;
      }
      case OP_ADMIT_MANIFEST:
        out.outcome = bcir_ho_admit_manifest(&plane, d.data(), d.size());
        break;
      case OP_FREEZE: {
        Reader g{d};
        GraphBuilder builder;
        std::vector<bcir_generation_view> gens;
        std::uint32_t topo = 0;
        if (!read_graph(g, builder, gens, topo))
          return 2;
        PackOwner fresh;
        out = builder.freeze(*arena, gens, topo, fresh);
        if (out.ok()) {
          dead_owners.push_back(std::move(owners[reg]));
          owners[reg] = std::move(fresh);
          views[reg] = owners[reg].view();
        }
        break;
      }
      default:
        std::fprintf(stderr, "op %u has no C++ spelling\n", op);
        return 2;
      }
      trace(scn, step, out, op, claims_p, *arena, plane);
    }
    // the scenario is over: the graveyard and the registers release after the last line
  }
  if (!r.err)
    std::printf("COPIES %zu\n", copies);
  return r.err ? 2 : 0;
}

// --- --shards --------------------------------------------------------------------------------

const std::uint8_t kRoot[32] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a,
                                0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25,
                                0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f};

// per case: u32 n_records, records (framed), whole (framed), u32 world
int run_shards(const char *path) {
  std::vector<std::uint8_t> data;
  if (!read_file(path, data))
    return 2;
  Reader r{data};
  std::uint32_t n = r.u32();
  for (std::uint32_t i = 0; i < n && !r.err; ++i) {
    std::uint32_t n_rec = r.u32();
    std::vector<std::vector<std::uint8_t>> recs;
    for (std::uint32_t k = 0; k < n_rec && !r.err; ++k)
      recs.push_back(r.blob());
    std::vector<std::uint8_t> whole = r.blob();
    std::uint32_t world = r.u32();
    if (r.err)
      return 2;
    bcir_ctl_state plane;
    if (bcir_ctl_init(&plane, kRoot, sizeof kRoot, 0, 7) != BCIR_OK)
      return 2;
    for (auto &rec : recs)
      (void)bcir_ctl_submit(&plane, rec.data(), rec.size());
    std::shared_ptr<PackArena> arena = PackArena::create(world + 8u, whole.size() + 4096u);
    Reservation slot;
    PackOwner owner;
    if (!arena->reserve(slot).ok() || !slot.write(0, whole.data(), whole.size()).ok() ||
        !slot.commit(whole.size(), owner).ok()) {
      std::printf("ERR store %s\n", status_name(BCIR_ERR_NOSPACE));
      continue;
    }
    HandoffResult adm = admit_view(owner.view(), plane);
    Walk ww;
    bcir_sha256_init(&ww.h);
    HandoffResult disp = adm.ok() ? dispatch_checked(owner.view(), plane, ww) : adm;
    std::uint8_t wclaims[32];
    bcir_sha256_final(&ww.h, wclaims);
    std::size_t outside = ww.outside;
    DistributedOrchestrator dist(world);
    ShardSet cut = dist.cut(owner.view(), *arena);
    if (!cut.result.ok()) {
      std::printf("ERR cut %s\n", status_name(cut.result.status()));
      continue;
    }
    Walk ranks; // every rank's claims in rank order, one running digest
    bcir_sha256_init(&ranks.h);
    bool ranks_ok = true;
    for (const PackOwner &s : cut.shards) {
      HandoffResult a = admit_view(s.view(), plane);
      HandoffResult dd = a.ok() ? dispatch_checked(s.view(), plane, ranks) : a;
      if (!dd.ok())
        ranks_ok = false;
    }
    outside += ranks.outside;
    std::uint8_t rclaims[32];
    bcir_sha256_final(&ranks.h, rclaims);
    std::vector<PackView> blobs;
    for (const PackOwner &s : cut.shards)
      blobs.push_back(s.view());
    blobs.push_back(cut.frame.view());
    PackOwner back;
    HandoffResult re = reassemble_shards(cut.manifest, blobs, *arena, back);
    std::uint8_t asha[32] = {0};
    if (re.ok()) {
      Borrow b = back.view().borrow();
      bcir_sha256_digest(b.data(), b.size(), asha);
    }
    std::printf("OK ");
    hex(cut.manifest.data(), cut.manifest.size());
    {
      Borrow f = cut.frame.view().borrow();
      std::printf(" ");
      hex(f.data(), f.size());
    }
    for (const PackOwner &s : cut.shards) {
      Borrow b = s.view().borrow();
      std::printf(" ");
      hex(b.data(), b.size());
    }
    std::printf(" | admit=%s dispatch=%s ranks=%s w=", status_name(adm.status()),
                status_name(disp.status()), ranks_ok ? "ok" : "refused");
    hex(wclaims, 16);
    std::printf(" r=");
    hex(rclaims, 16);
    std::printf(" a=");
    if (re.ok())
      hex(asha, 32);
    else
      std::printf("%s", status_name(re.status()));
    std::printf(" x=%zu\n", outside);
  }
  return r.err ? 2 : 0;
}

// --- --lifetime ------------------------------------------------------------------------------

int witnesses = 0, failures = 0;
void witness(const char *name, bool ok) {
  witnesses++;
  if (!ok)
    failures++;
  std::printf("%s %s\n", name, ok ? "ok" : "FAIL");
}

PackOwner freeze_bytes(PackArena &arena, const std::vector<std::uint8_t> &bytes) {
  Reservation r;
  PackOwner o;
  if (arena.reserve(r).ok() && r.write(0, bytes.data(), bytes.size()).ok())
    (void)r.commit(bytes.size(), o);
  return o;
}

int run_lifetime() {
  const std::vector<std::uint8_t> one(64, 0x11), two(64, 0x22);
  bcir_ctl_state plane;
  (void)bcir_ctl_init(&plane, kRoot, sizeof kRoot, 0, 7);
  {
    auto arena = PackArena::create(2, 128);
    PackOwner a = freeze_bytes(*arena, one);
    PackView v = a.view();
    (void)a.release();
    Borrow b = v.borrow();
    witness("view-after-release", !b.ok() && b.result().status() == BCIR_ERR_LIFETIME);
  }
  {
    auto arena = PackArena::create(2, 128);
    PackView v;
    {
      PackOwner a = freeze_bytes(*arena, one);
      v = a.view();
    } // the owner leaves scope
    witness("view-after-owner-scope", v.borrow().result().status() == BCIR_ERR_LIFETIME);
  }
  {
    PackView v;
    PackOwner survivor;
    {
      auto arena = PackArena::create(2, 128);
      survivor = freeze_bytes(*arena, one);
      v = survivor.view();
    } // the arena goes while its owner still holds a live handle
    Borrow b = v.borrow();
    witness("view-after-arena",
            !b.ok() && b.result().status() == BCIR_ERR_LIFETIME &&
                admit_view(v, plane).status() == BCIR_ERR_LIFETIME &&
                dispatch_view(v, plane, nullptr, nullptr).status() == BCIR_ERR_LIFETIME &&
                survivor.release().status() == BCIR_ERR_LIFETIME);
  }
  {
    auto arena = PackArena::create(1, 128);
    PackOwner a = freeze_bytes(*arena, one);
    PackView v = a.view();
    (void)a.release();
    PackOwner c = freeze_bytes(*arena, two); // the same slot, the next incarnation
    Borrow b = v.borrow();
    Borrow bc = c.view().borrow();
    witness("reused-slot-old-view", !b.ok() && b.result().status() == BCIR_ERR_LIFETIME &&
                                        bc.ok() && c.handle().index == v.handle().index &&
                                        std::memcmp(bc.data(), two.data(), two.size()) == 0);
  }
  {
    auto arena = PackArena::create(1, 128);
    PackOwner a = freeze_bytes(*arena, one);
    Borrow b = a.view().borrow();
    (void)a.release(); // released while pinned: retired, bytes kept for the borrow
    Reservation r;
    bool full = arena->reserve(r).refusal() == BCIR_HO_REFUSAL_FULL;
    bool kept = b.ok() && std::memcmp(b.data(), one.data(), one.size()) == 0;
    (void)b.give_back();
    bool freed = arena->reserve(r).ok();
    witness("borrow-outlives-release", full && kept && freed);
  }
  {
    auto arena = PackArena::create(1, 128);
    {
      Reservation r;
      (void)arena->reserve(r);
      (void)r.write(0, one.data(), one.size());
    } // destroyed uncommitted: aborted, nothing published
    PackOwner a = freeze_bytes(*arena, two);
    witness("reservation-aborts", a.valid() && a.handle().epoch == 2u);
  }
  {
    auto arena = PackArena::create(2, 128);
    PackOwner a = freeze_bytes(*arena, one);
    PackOwner moved = std::move(a);
    witness("moved-owner", a.release().status() == BCIR_ERR_LIFETIME && moved.view().borrow().ok());
  }
  {
    auto arena = PackArena::create(2, 128);
    PackOwner a = freeze_bytes(*arena, one);
    Borrow b = a.view().borrow();
    Borrow c = a.view().borrow();
    bool first = b.give_back().ok();
    bool second = b.give_back().status() == BCIR_ERR_LIFETIME;
    bool other = c.ok() && std::memcmp(c.data(), one.data(), one.size()) == 0;
    witness("double-give-back", first && second && other);
  }
  {
    auto arena_a = PackArena::create(1, 128), arena_b = PackArena::create(1, 128);
    PackOwner a = freeze_bytes(*arena_a, one);
    witness("foreign-arena",
            arena_b->admit(a.view(), plane).status() == BCIR_ERR_LIFETIME &&
                arena_b->dispatch(a.view(), plane, nullptr, nullptr).status() == BCIR_ERR_LIFETIME);
  }
  {
    auto arena = PackArena::create(1, 128);
    Reservation r;
    (void)arena->reserve(r);
    std::uint8_t *region = r.data();
    (void)r.write(0, one.data(), one.size());
    PackOwner a;
    (void)r.commit(one.size(), a);
    Borrow b = a.view().borrow();
    witness("bytes-moved-once", b.ok() && b.data() == region); // the consumer reads the slot
  }
  std::printf("%s %d\n", failures ? "FAILED" : "OK", witnesses);
  return 0;
}

// --- --bench ---------------------------------------------------------------------------------

extern "C" int bench_collect(const bcir_segment_view *seg, void *ctx) {
  static_cast<std::vector<std::uint64_t> *>(ctx)->push_back(seg->claim_id);
  return 0;
}

// The seam's dispatch of an admitted pack against the direct C walk of the same bytes, the same
// callback collecting claim ids: median over rounds of the per-dispatch time ratio. The pack and
// the plane records come from the grader (the pack's own registry installed).
int run_bench(const char *pack_path, const char *plane_path) {
  std::vector<std::uint8_t> pack, plane_file;
  if (!read_file(pack_path, pack) || !read_file(plane_path, plane_file))
    return 2;
  bcir_ctl_state plane;
  if (bcir_ctl_init(&plane, kRoot, sizeof kRoot, 0, 7) != BCIR_OK)
    return 2;
  Reader r{plane_file};
  std::uint32_t n_rec = r.u32();
  for (std::uint32_t i = 0; i < n_rec && !r.err; ++i) {
    std::vector<std::uint8_t> rec = r.blob();
    (void)bcir_ctl_submit(&plane, rec.data(), rec.size());
  }
  auto arena = PackArena::create(2, pack.size() + 64);
  Reservation slot;
  PackOwner owner;
  if (!arena->reserve(slot).ok() || !slot.write(0, pack.data(), pack.size()).ok() ||
      !slot.commit(pack.size(), owner).ok() || !admit_view(owner.view(), plane).ok())
    return 2;
  SingleNodeOrchestrator single;
  std::vector<std::uint64_t> order;
  order.reserve(4096);
  const int iters = 2000, rounds = 7;
  std::vector<double> ratios;
  for (int round = 0; round < rounds; ++round) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      order.clear();
      if (bcir_sp_for_each_segment(pack.data(), pack.size(), &bench_collect, &order) != BCIR_OK)
        return 2;
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      if (!single.dispatch(owner.view(), plane).ok())
        return 2;
    auto t2 = std::chrono::steady_clock::now();
    double direct = std::chrono::duration<double>(t1 - t0).count();
    double seam = std::chrono::duration<double>(t2 - t1).count();
    ratios.push_back(seam / direct);
  }
  std::sort(ratios.begin(), ratios.end());
  std::printf("BENCH ratio=%.4f bytes=%zu\n", ratios[ratios.size() / 2], pack.size());
  return 0;
}

// --- --stage3: the Stage 3 exit flow, the data boundary through the seam ------------------
// test_stage3.h drives the same flow as the C harness; here the pack is written through a
// Reservation, owned by a PackOwner, and admitted and dispatched through its PackView.

struct Stage3Seam {
  std::shared_ptr<PackArena> arena = PackArena::create(4, 4096);
  std::array<PackOwner, 2> owners;
};

bcir_ho_outcome s3cpp_store(void *ctx, int which, const std::uint8_t *data, std::size_t len) {
  auto *s = static_cast<Stage3Seam *>(ctx);
  Reservation r;
  HandoffResult out = s->arena->reserve(r);
  if (!out.ok())
    return out.outcome;
  out = r.write(0, data, len);
  if (!out.ok())
    return out.outcome;
  PackOwner fresh;
  out = r.commit(len, fresh);
  if (out.ok())
    s->owners[static_cast<std::size_t>(which)] = std::move(fresh);
  return out.outcome;
}

bcir_ho_outcome s3cpp_admit(void *ctx, int which, bcir_ctl_state *plane) {
  auto *s = static_cast<Stage3Seam *>(ctx);
  return admit_view(s->owners[static_cast<std::size_t>(which)].view(), *plane).outcome;
}

bcir_ho_outcome s3cpp_dispatch(void *ctx, int which, bcir_ctl_state *plane, bcir_seg_fn fn,
                               void *fn_ctx) {
  auto *s = static_cast<Stage3Seam *>(ctx);
  return dispatch_view(s->owners[static_cast<std::size_t>(which)].view(), *plane, fn, fn_ctx)
      .outcome;
}

bcir_ho_outcome s3cpp_release(void *ctx, int which) {
  auto *s = static_cast<Stage3Seam *>(ctx);
  return s->owners[static_cast<std::size_t>(which)].release().outcome;
}

void s3cpp_digest(void *ctx, std::uint8_t out[32]) {
  std::array<std::uint8_t, 32> d = static_cast<Stage3Seam *>(ctx)->arena->state_digest();
  std::memcpy(out, d.data(), d.size());
}

int run_stage3(const char *path) {
  std::vector<std::uint8_t> data;
  if (!read_file(path, data))
    return 2;
  Stage3Seam seam;
  s3_ops ops;
  ops.ctx = &seam;
  ops.store = s3cpp_store;
  ops.admit = s3cpp_admit;
  ops.dispatch = s3cpp_dispatch;
  ops.release = s3cpp_release;
  ops.digest = s3cpp_digest;
  ops.status = status_name;
  return s3_run(data.data(), data.size(), &ops);
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "--lifetime") == 0)
    return run_lifetime();
  if (argc != 3 && !(argc == 4 && std::strcmp(argv[1], "--bench") == 0)) {
    std::fprintf(stderr,
                 "usage: test_handoff_cpp --script|--shards|--stage3 <file> | --bench <pack> "
                 "<plane> | --lifetime\n");
    return 2;
  }
  if (std::strcmp(argv[1], "--script") == 0)
    return run_script(argv[2]);
  if (std::strcmp(argv[1], "--shards") == 0)
    return run_shards(argv[2]);
  if (std::strcmp(argv[1], "--stage3") == 0)
    return run_stage3(argv[2]);
  if (argc == 4 && std::strcmp(argv[1], "--bench") == 0)
    return run_bench(argv[2], argv[3]);
  std::fprintf(stderr, "unknown mode %s\n", argv[1]);
  return 2;
}
