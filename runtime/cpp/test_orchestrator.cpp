//===- test_orchestrator.cpp - round-trip smoke test for the hand-off seam ==//
//
// Proves the C<->C++ hand-off seam round-trips: a StreamPack (the serialized hand-off artifact,
// produced by the existing C/IR path) is written ONCE into an arena slot, admitted against the
// LIVE control plane, and dispatched through the single-node Orchestrator; the result is asserted
// EQUAL to the DIRECT C/IR decode of the same bytes. Equality is the proof the seam is lossless.
//
// It also exercises the rest of the contract surface (G16):
//   - admission is the plane's verdict: the admitted pack runs; after a registry switch the
//     same pack is refused at dispatch (stale) and at re-admission;
//   - a view that outlives its owner is refused (BCIR_ERR_LIFETIME), never read;
//   - the distributed backend's partition and manifest-of-shards are real: each shard admits and
//     runs by itself, the ranks' orders concatenate to the whole's, and the shards reassemble
//     to the whole pack's bytes; its cross-node dispatch is a stub that fails loudly;
//   - the dynamic-graph backend freezes a fresh graph through the C/IR rail per step.
//
// Usage:  test_orchestrator <pack.bin> <plane.bin>
//         test_orchestrator --reject <pack.bin> <plane.bin>
//   plane.bin: u32le-framed ControlRecordV1 records under the fixture root key -- a lease grant,
//   the generation switch installing the pack's registry, and a second switch to another one
//   (bcir/tests/handoff_fixtures.py::seam_artifacts mints both files for the gate and the tests).
// Prints "OK <n> segments" and exits 0 on success; prints "FAIL: ..." + exits 1. --reject admits
// and dispatches a (corrupted) pack against the live plane: "REJECTED" and exit 0 when admission
// refuses it as malformed -- the plane's verdict, carried -- and nothing runs; "ADMITTED" and
// exit 1 otherwise.
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bcir_orchestrator.hpp"

namespace {

const std::uint8_t kRoot[32] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a,
                                0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25,
                                0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f};

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

std::vector<std::vector<std::uint8_t>> frames(const std::vector<std::uint8_t> &b) {
  std::vector<std::vector<std::uint8_t>> out;
  std::size_t pos = 0;
  while (pos + 4 <= b.size()) {
    std::uint32_t n = static_cast<std::uint32_t>(b[pos]) |
                      (static_cast<std::uint32_t>(b[pos + 1]) << 8) |
                      (static_cast<std::uint32_t>(b[pos + 2]) << 16) |
                      (static_cast<std::uint32_t>(b[pos + 3]) << 24);
    pos += 4;
    if (n > b.size() - pos)
      break;
    out.emplace_back(b.begin() + static_cast<long>(pos), b.begin() + static_cast<long>(pos + n));
    pos += n;
  }
  return out;
}

extern "C" int direct_collect(const bcir_segment_view *seg, void *ctx) {
  static_cast<std::vector<std::uint64_t> *>(ctx)->push_back(seg->claim_id);
  return 0;
}

extern "C" int gen_collect(const bcir_generation_view *g, void *ctx) {
  static_cast<std::vector<bcir_generation_view> *>(ctx)->push_back(*g);
  return 0;
}

int fail(const std::string &msg) {
  std::printf("FAIL: %s\n", msg.c_str());
  return 1;
}

std::string st(bcir_status s) {
  return std::to_string(static_cast<int>(s));
}

// --reject: a corrupted artifact is refused at admission and never dispatched.
int run_reject(const char *pack_path, const char *plane_path) {
  std::vector<std::uint8_t> bad, plane_file;
  if (!read_file(pack_path, bad) || !read_file(plane_path, plane_file))
    return 2;
  std::vector<std::vector<std::uint8_t>> records = frames(plane_file);
  if (records.size() < 2)
    return 2;
  bcir_ctl_state plane;
  if (bcir_ctl_init(&plane, kRoot, sizeof kRoot, 0, 7) != BCIR_OK)
    return 2;
  for (std::size_t i = 0; i < 2; ++i)
    (void)bcir_ctl_submit(&plane, records[i].data(), records[i].size());
  std::shared_ptr<bcir::PackArena> arena = bcir::PackArena::create(2, bad.size() + 64);
  bcir::Reservation r;
  bcir::PackOwner o;
  if (!arena->reserve(r).ok() || !r.write(0, bad.data(), bad.size()).ok() ||
      !r.commit(bad.size(), o).ok())
    return 2;
  bcir::SingleNodeOrchestrator s;
  bcir::HandoffResult a = s.admit(o.view(), plane);
  bcir::DispatchResult d = s.dispatch(o.view(), plane);
  bool refused = !a.ok() && a.refusal() == BCIR_HO_REFUSAL_MALFORMED && !d.ok();
  std::printf(refused ? "REJECTED\n" : "ADMITTED\n");
  return refused ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 4 && std::strcmp(argv[1], "--reject") == 0)
    return run_reject(argv[2], argv[3]);
  if (argc < 3)
    return fail("usage: test_orchestrator <pack.bin> <plane.bin> | --reject <pack.bin> <plane.bin>");
  std::vector<std::uint8_t> bytes, plane_file;
  if (!read_file(argv[1], bytes))
    return fail(std::string("cannot read ") + argv[1]);
  if (!read_file(argv[2], plane_file))
    return fail(std::string("cannot read ") + argv[2]);
  std::vector<std::vector<std::uint8_t>> records = frames(plane_file);
  if (records.size() != 3)
    return fail("plane.bin must hold a grant and two generation switches");

  using namespace bcir;

  // The live plane: a lease, then the registry the pack was hydrated under.
  bcir_ctl_state plane;
  if (bcir_ctl_init(&plane, kRoot, sizeof kRoot, 0, 7) != BCIR_OK)
    return fail("plane init");
  for (int i = 0; i < 2; ++i)
    if (bcir_ctl_submit(&plane, records[static_cast<std::size_t>(i)].data(),
                        records[static_cast<std::size_t>(i)].size())
            .verdict != BCIR_CTL_APPLIED)
      return fail("the plane refused its bootstrap record " + std::to_string(i));

  // The artifact, written once into an arena slot.
  std::shared_ptr<PackArena> arena = PackArena::create(64, bytes.size() * 2 + 4096);
  Reservation slot;
  PackOwner owner;
  if (!arena->reserve(slot).ok() || !slot.write(0, bytes.data(), bytes.size()).ok() ||
      !slot.commit(bytes.size(), owner).ok())
    return fail("could not freeze the artifact into the arena");
  PackView view = owner.view();

  // (1) admit(): the plane's verdict, carried -- not derived, no generation numbers passed.
  SingleNodeOrchestrator single;
  HandoffResult adm = single.admit(view, plane);
  if (!adm.ok())
    return fail("the current pack was not admitted (status=" + st(adm.status()) + ")");

  // (2) THE ROUND TRIP: seam dispatch == the direct C walk of the same bytes.
  DispatchResult seam = single.dispatch(view, plane);
  if (!seam.ok())
    return fail("seam dispatch failed (status=" + st(seam.status) + ")");
  std::vector<std::uint64_t> direct;
  if (bcir_sp_for_each_segment(bytes.data(), bytes.size(), &direct_collect, &direct) != BCIR_OK)
    return fail("direct C decode failed");
  if (seam.claim_order != direct)
    return fail("ROUND-TRIP MISMATCH: seam order != direct C order");
  if (seam.nodes_used != 1 || seam.shards != 1)
    return fail("single-node used unexpected nodes/shards");

  // (3) the shard contract: one shard covering the whole pack.
  std::vector<Shard> sh = single.shard(view);
  if (sh.size() != 1 || sh[0].seg_begin != 0 || sh[0].seg_end != direct.size())
    return fail("single-node sharding contract violated");

  // (4) the distributed backend: a real partition and a real manifest-of-shards.
  DistributedOrchestrator dist(3);
  std::vector<Shard> parts = dist.shard(view);
  for (std::size_t i = 1; i < parts.size(); ++i)
    if (parts[i].seg_begin != parts[i - 1].seg_end)
      return fail("partition is not contiguous");
  if (parts.empty() || parts.front().seg_begin != 0 || parts.back().seg_end != direct.size())
    return fail("partition does not cover the segment stream");
  ShardSet cut = dist.cut(view, *arena);
  if (!cut.result.ok())
    return fail("cut failed (status=" + st(cut.result.status()) + ")");
  std::vector<std::uint64_t> ranks;
  for (const PackOwner &s : cut.shards) {
    if (!single.admit(s.view(), plane).ok())
      return fail("a shard was not admitted by itself");
    DispatchResult r = single.dispatch(s.view(), plane);
    if (!r.ok())
      return fail("a shard did not run by itself");
    ranks.insert(ranks.end(), r.claim_order.begin(), r.claim_order.end());
  }
  if (ranks != direct)
    return fail("the ranks' orders do not concatenate to the whole's");
  std::vector<PackView> blobs;
  for (const PackOwner &s : cut.shards)
    blobs.push_back(s.view());
  blobs.push_back(cut.frame.view());
  PackOwner whole;
  HandoffResult re = reassemble_shards(cut.manifest, blobs, *arena, whole);
  Borrow wb = whole.view().borrow();
  if (!re.ok() || !wb.ok() || wb.size() != bytes.size() ||
      std::memcmp(wb.data(), bytes.data(), bytes.size()) != 0)
    return fail("the shards did not reassemble to the whole pack's bytes");
  wb.give_back();

  // (5) the dynamic-graph backend: a fresh graph frozen through the rail under the live registry.
  std::vector<bcir_generation_view> registry;
  if (bcir_sp_for_each_generation(bytes.data(), bytes.size(), &gen_collect, &registry) != BCIR_OK ||
      registry.empty())
    return fail("the pack carries no generation vector");
  DynamicGraphOrchestrator dyn;
  GraphBuilder g;
  GraphBuilder::ClaimSpec load;
  load.opcode = BCIR_OP_LOAD;
  load.reads = {registry.front().rid};
  load.label = "c.load";
  g.add(load);
  GraphBuilder::ClaimSpec add;
  add.reads = {registry.front().rid};
  add.writes = {registry.back().rid};
  g.add(add);
  StepResult step = dyn.step(g, *arena, registry, plane.reg_topo_gen, plane);
  if (!step.ok() || step.dispatch.claim_order != std::vector<std::uint64_t>{1, 2})
    return fail("a dynamic-graph step did not freeze, admit and run (freeze=" +
                st(step.freeze.status()) + " admit=" + st(step.admission.status()) + ")");

  // (6) the stubs fail loudly.
  bool dist_threw = false;
  try {
    dist.dispatch(view, plane);
  } catch (const HandoffError &) {
    dist_threw = true;
  }
  if (!dist_threw)
    return fail("distributed stub did not fail loudly");

  // (7) idempotence: re-dispatching with retries yields the same order.
  DispatchResult retried = single.dispatch(view, plane, /*max_retries=*/3);
  if (retried.claim_order != seam.claim_order)
    return fail("retry changed the result");

  // (8) generation gating: the registry moves on -- the admitted pack is now stale at dispatch,
  // and re-admission refuses it by the plane's predicate.
  if (bcir_ctl_submit(&plane, records[2].data(), records[2].size()).verdict != BCIR_CTL_APPLIED)
    return fail("the plane refused the second switch");
  DispatchResult after = single.dispatch(view, plane);
  if (after.ok() || after.handoff.refusal() != BCIR_HO_REFUSAL_STALE)
    return fail("a pack admitted before the switch was dispatched after it");
  HandoffResult readmit = single.admit(view, plane);
  if (readmit.ok() || readmit.status() != BCIR_ERR_STALE)
    return fail("a stale pack was re-admitted");

  // (9) lifetime: the owner lets go; the view outlives it and is refused, never read.
  if (!owner.release().ok())
    return fail("release");
  Borrow dead = view.borrow();
  if (dead.ok() || dead.result().status() != BCIR_ERR_LIFETIME)
    return fail("a view that outlived its owner was not refused");
  DispatchResult late = single.dispatch(view, plane);
  if (late.ok() || late.status != BCIR_ERR_LIFETIME)
    return fail("a dead view was dispatched");

  std::printf("OK %zu segments\n", seam.claim_order.size());
  return 0;
}
