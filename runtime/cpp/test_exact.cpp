//===- test_exact.cpp - the exact scheduler's contracts, and its text driver ---===//
//
// PORT-5 (bcir_exact.h). Two modes:
//
//   test_exact          the façade's contracts: a phase the search closes, the budget's stops and pending phases,
//                       every refusal (INVALID, CYCLIC, CAPACITY) with nothing written past it, determinism, and the
//                       allocation discipline -- every allocation of a run failed in turn gives NOMEM with nothing
//                       left live, and a run that is never refused frees all it took. Prints `test_exact: OK`.
//   test_exact FILE     the problems in FILE, each `budget N` and its phases, scheduled one after another and printed
//                       line for line as bcir/tests/test_cpp_exact.py prints the oracle's `exact_schedule`:
//
//     in:   `budget N`, then per phase `phase ID DOMAINS KNEE HEURISTIC`, a `claim ID DURATION PREDS STREAMS` line
//           per claim (PREDS and STREAMS comma-separated, `-` for none) and a `hslot CLAIM STREAM START FINISH` line
//           per slot of the heuristic's placement
//     out:  per phase `phase ID heuristic H incumbent U lower L <optimal|budget> expansions E bounds A,B,C,D slots
//           C:S:ST:F,...`, then `total heuristic H incumbent U lower L <optimal|budget> expansions E`; a refusal is
//           `ERR <status>`. A malformed file prints `TOOL <reason>` and exits 2.
//===----------------------------------------------------------------------===//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "bcir_exact.h"

namespace {

int failures = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
  } while (0)

// --- a phase built from vectors -----------------------------------------------------------------------------------

struct ClaimIn { uint32_t id; int64_t dur; std::vector<uint32_t> preds, streams; };
struct PhaseIn {
  uint32_t id = 0, domains = 1, knee = 1;
  int64_t heuristic = 0;
  std::vector<ClaimIn> claims;
  std::vector<bcir_exact_slot> hslots;
};

struct Run {
  std::vector<std::vector<bcir_exact_claim>> claims;
  std::vector<bcir_exact_phase> ins;
  std::vector<std::vector<bcir_exact_slot>> slots;
  std::vector<bcir_exact_phase_result> res;
  bcir_exact_result total{};
  bcir_exact_status status = BCIR_EXACT_OK;
};

Run run(const std::vector<PhaseIn> &phases, uint64_t budget, const bcir_host_allocator *allocator = nullptr,
        size_t short_by = 0) {
  Run r;
  r.claims.resize(phases.size()); r.ins.resize(phases.size()); r.slots.resize(phases.size()); r.res.resize(phases.size());
  for (size_t k = 0; k < phases.size(); k++) {
    for (const ClaimIn &c : phases[k].claims)
      r.claims[k].push_back(bcir_exact_claim{c.id, c.dur, c.preds.data(), c.preds.size(), c.streams.data(), c.streams.size()});
    r.ins[k] = bcir_exact_phase{phases[k].id, r.claims[k].data(), r.claims[k].size(), phases[k].domains,
                                phases[k].knee, phases[k].heuristic, phases[k].hslots.data(), phases[k].hslots.size()};
    size_t cap = r.claims[k].size() >= short_by ? r.claims[k].size() - short_by : 0;
    r.slots[k].assign(cap + 1, bcir_exact_slot{0xFFFFFFFFu, 99, -7, -7});   // one sentinel past the capacity
    r.res[k] = bcir_exact_phase_result{};
    r.res[k].slots = r.slots[k].data(); r.res[k].cap_slots = cap;
  }
  r.status = bcir_exact_schedule(r.ins.data(), r.ins.size(), budget, allocator, r.res.data(), &r.total);
  return r;
}

// The placement a result names must be a schedule of its phase whose span is the incumbent -- checked here by code
// that shares nothing with the solver's own check of the heuristic it is given.
bool is_schedule(const PhaseIn &p, const bcir_exact_phase_result &r) {
  if (r.n_slots != p.claims.size()) return false;
  int64_t span = 0;
  for (size_t a = 0; a < r.n_slots; a++) {
    const bcir_exact_slot &s = r.slots[a];
    const ClaimIn *c = nullptr;
    for (const ClaimIn &x : p.claims) if (x.id == s.claim) c = &x;
    if (!c || s.finish - s.start != c->dur || s.start < 0) return false;
    uint32_t stream = s.stream < 0 ? p.domains : static_cast<uint32_t>(s.stream);
    bool eligible = false;
    for (uint32_t e : c->streams) eligible = eligible || e == stream;
    if (!eligible) return false;
    for (size_t b = 0; b < r.n_slots; b++) {
      const bcir_exact_slot &o = r.slots[b];
      if (b != a && o.claim == s.claim) return false;
      for (uint32_t q : c->preds) if (o.claim == q && o.finish > s.start) return false;
      if (b != a && o.stream == s.stream && s.finish > s.start && o.finish > o.start && s.start < o.finish &&
          o.start < s.finish)
        return false;
    }
    if (s.finish > span) span = s.finish;
  }
  return span == r.incumbent;
}

// --- a counting allocator that refuses attempt `fail_at` (0: none) ---------------------------------------------------

struct Counting { size_t attempts = 0, fail_at = 0, live = 0; };
void *count_alloc(void *ctx, size_t n) {
  Counting *c = static_cast<Counting *>(ctx);
  if (++c->attempts == c->fail_at) return nullptr;
  void *p = std::malloc(n ? n : 1);
  if (p) c->live++;
  return p;
}
void *count_realloc(void *ctx, void *p, size_t n) {
  Counting *c = static_cast<Counting *>(ctx);
  if (++c->attempts == c->fail_at) return nullptr;
  if (!p) { void *q = std::malloc(n ? n : 1); if (q) c->live++; return q; }
  return std::realloc(p, n ? n : 1);
}
void count_free(void *ctx, void *p) {
  Counting *c = static_cast<Counting *>(ctx);
  if (p) { c->live--; std::free(p); }
}
bcir_host_allocator counting(Counting *c) { return bcir_host_allocator{c, count_alloc, count_realloc, count_free}; }

// --- the phases the contracts use --------------------------------------------------------------------------------

// Two independent claims of 3 placed one after the other on stream 0: the heuristic 6, the optimum 3 (both waves).
PhaseIn two_serial() {
  PhaseIn p; p.id = 1; p.domains = 2; p.knee = 2; p.heuristic = 6;
  p.claims = {{1, 3, {}, {0, 1}}, {2, 3, {}, {0, 1}}};
  p.hslots = {{1, 0, 0, 3}, {2, 0, 3, 6}};
  return p;
}

// A diamond of six claims over two waves and the tail, its heuristic the serial order on stream 0: the search has
// work to do, so a small budget stops it.
PhaseIn diamond(uint32_t id) {
  PhaseIn p; p.id = id; p.domains = 2; p.knee = 1;
  p.claims = {{10, 4, {}, {0, 1}}, {11, 2, {10}, {0, 1}}, {12, 5, {10}, {0, 1}}, {13, 1, {11, 12}, {2}},
              {14, 3, {}, {0}}, {15, 2, {14}, {0, 1}}};
  int64_t t = 0;
  for (const ClaimIn &c : p.claims) {
    bool tail = c.streams.size() == 1 && c.streams[0] == p.domains;
    p.hslots.push_back(bcir_exact_slot{c.id, tail ? -1 : 0, t, t + c.dur});
    t += c.dur;
  }
  p.heuristic = t;
  return p;
}

void test_a_phase_the_search_closes() {
  Run r = run({two_serial()}, 1000);
  CHECK(r.status == BCIR_EXACT_OK);
  const bcir_exact_phase_result &p = r.res[0];
  CHECK(p.phase_id == 1 && p.heuristic == 6 && p.incumbent == 3 && p.lower_bound == 3 && p.optimal);
  CHECK(p.bounds[0] == 3 && p.bounds[1] == 3 && p.bounds[2] == 0 && p.bounds[3] == 0);
  CHECK(is_schedule(two_serial(), p));
  CHECK(r.total.heuristic == 6 && r.total.incumbent == 3 && r.total.lower_bound == 3 && r.total.optimal);
  CHECK(r.total.budget == 1000 && r.total.expansions == p.expansions && p.expansions > 0);
  CHECK(r.slots[0][p.cap_slots].claim == 0xFFFFFFFFu);   // nothing written past the capacity
}

void test_the_budget_stops_a_phase_and_leaves_the_rest_pending() {
  Run full = run({diamond(1), diamond(2)}, 1000000);
  CHECK(full.status == BCIR_EXACT_OK && full.res[0].optimal && full.res[1].optimal && full.total.optimal);
  CHECK(is_schedule(diamond(1), full.res[0]) && is_schedule(diamond(2), full.res[1]));
  CHECK(full.res[0].incumbent < full.res[0].heuristic);
  for (uint64_t budget : {0ull, 1ull, 3ull}) {
    Run r = run({diamond(1), diamond(2)}, budget);
    CHECK(r.status == BCIR_EXACT_OK);
    CHECK(r.res[0].expansions == budget && !r.res[0].optimal && !r.total.optimal);
    CHECK(r.res[0].lower_bound <= r.res[0].incumbent && r.res[0].incumbent <= r.res[0].heuristic);
    CHECK(is_schedule(diamond(1), r.res[0]) && is_schedule(diamond(2), r.res[1]));
    // pending: the heuristic stands, the root stack its bound, nothing spent
    const bcir_exact_phase_result &q = r.res[1];
    int64_t root = q.bounds[0];
    for (int64_t b : q.bounds) root = b > root ? b : root;
    CHECK(q.expansions == 0 && q.incumbent == q.heuristic && q.lower_bound == root && !q.optimal);
    CHECK(r.total.expansions == budget && r.total.lower_bound <= full.total.incumbent);
  }
  // a phase that closes hands what is left of the budget to the next
  Run first = run({diamond(1)}, 1000000);
  Run both = run({diamond(1), diamond(2)}, first.res[0].expansions + 1);
  CHECK(both.res[0].optimal && both.res[1].expansions == 1);
}

void test_an_empty_module_and_an_empty_phase() {
  Run none = run({}, 5);
  CHECK(none.status == BCIR_EXACT_OK && none.total.optimal && none.total.incumbent == 0 && none.total.expansions == 0);
  PhaseIn empty; empty.id = 9;
  Run r = run({empty}, 5);
  CHECK(r.status == BCIR_EXACT_OK && r.res[0].optimal && r.res[0].incumbent == 0 && r.res[0].n_slots == 0);
}

void test_every_refusal() {
  bcir_exact_result total{};
  CHECK(bcir_exact_schedule(nullptr, 1, 1, nullptr, nullptr, &total) == BCIR_EXACT_ERR_INVALID);
  PhaseIn ok = two_serial();
  {
    Run r = run({ok}, 1);
    CHECK(bcir_exact_schedule(r.ins.data(), 1, 1, nullptr, r.res.data(), nullptr) == BCIR_EXACT_ERR_INVALID);
    CHECK(bcir_exact_schedule(r.ins.data(), 1, 1, nullptr, nullptr, &total) == BCIR_EXACT_ERR_INVALID);
    bcir_host_allocator incomplete{nullptr, count_alloc, nullptr, nullptr};
    CHECK(bcir_exact_schedule(r.ins.data(), 1, 1, &incomplete, r.res.data(), &total) == BCIR_EXACT_ERR_INVALID);
  }
  struct Case { const char *what; PhaseIn p; bcir_exact_status want; };
  std::vector<Case> cases;
  auto with = [&](const char *what, bcir_exact_status want, auto edit) { PhaseIn p = two_serial(); edit(p); cases.push_back({what, p, want}); };
  with("a negative duration", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.claims[0].dur = -1; });
  with("a repeated claim id", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.claims[1].id = 1; });
  with("a predecessor outside the phase", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.claims[1].preds = {7}; });
  with("a claim its own predecessor", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.claims[1].preds = {2}; });
  with("a stream past the tail", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.claims[0].streams = {3}; });
  with("a claim with no stream", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.claims[0].streams = {}; });
  with("a claim the tail and a wave could both take", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.claims[1].streams = {0, 2}; });
  with("work that leaves 64 bits", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) {
    p.claims[0].dur = INT64_MAX / 2; p.claims[1].dur = INT64_MAX / 2; });
  // two claims whose durations sum past 64 bits, placed as a schedule (one on a wave, one on the tail): refused for
  // the sum alone
  with("a phase's work summing past 64 bits", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) {
    const int64_t d = INT64_MAX / 2 + 1;
    p.claims = {{1, d, {}, {0}}, {2, d, {}, {2}}};
    p.hslots = {{1, 0, 0, d}, {2, -1, 0, d}}; p.heuristic = d; });
  with("a heuristic slot missing", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.hslots.pop_back(); });
  with("a heuristic slot on a claim twice", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.hslots[1].claim = 1; });
  with("a heuristic slot for no claim", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.hslots[1].claim = 5; });
  with("a heuristic slot on a stream not eligible", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.hslots[1].stream = -1; });
  with("a heuristic slot on no stream", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.hslots[1].stream = 2; });
  with("a heuristic slot of the wrong length", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.hslots[1].finish = 7; p.heuristic = 7; });
  with("a heuristic slot before time 0", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.hslots[0] = {1, 1, -1, 2}; });
  with("two heuristic slots at once on one stream", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.hslots[1] = {2, 0, 2, 5}; p.heuristic = 5; });
  with("a heuristic span that is not the placement's", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) { p.heuristic = 5; });
  with("a heuristic slot before its predecessor ends", BCIR_EXACT_ERR_INVALID, [](PhaseIn &p) {
    p.claims[1].preds = {1}; p.hslots[1] = {2, 1, 2, 5}; p.heuristic = 5; });
  with("a dependency cycle", BCIR_EXACT_ERR_CYCLIC, [](PhaseIn &p) { p.claims[0].preds = {2}; p.claims[1].preds = {1}; });
  for (const Case &c : cases) {
    Run r = run({c.p}, 100);
    if (r.status != c.want) { std::fprintf(stderr, "%s: status %d, not %d\n", c.what, r.status, c.want); failures++; }
  }
  // zero-length slots occupy nothing: two on one stream at the same instant are a schedule
  PhaseIn zeros; zeros.id = 3; zeros.domains = 1; zeros.knee = 1;
  zeros.claims = {{1, 0, {}, {0}}, {2, 0, {}, {0}}, {3, 4, {}, {0}}};
  zeros.hslots = {{1, 0, 2, 2}, {2, 0, 2, 2}, {3, 0, 0, 4}}; zeros.heuristic = 4;
  CHECK(run({zeros}, 100).status == BCIR_EXACT_OK);
  // a result buffer smaller than its phase
  Run small = run({two_serial()}, 100, nullptr, 1);
  CHECK(small.status == BCIR_EXACT_ERR_CAPACITY);
  CHECK(small.slots[0][0].claim == 0xFFFFFFFFu && small.slots[0][1].claim == 0xFFFFFFFFu);
}

void test_the_search_is_deterministic() {
  for (uint64_t budget : {2ull, 40ull, 1000000ull}) {
    Run a = run({diamond(1), two_serial(), diamond(4)}, budget), b = run({diamond(1), two_serial(), diamond(4)}, budget);
    CHECK(a.status == BCIR_EXACT_OK && b.status == BCIR_EXACT_OK);
    for (size_t k = 0; k < a.res.size(); k++) {
      CHECK(a.res[k].incumbent == b.res[k].incumbent && a.res[k].expansions == b.res[k].expansions);
      CHECK(a.res[k].n_slots == b.res[k].n_slots &&
            (a.res[k].n_slots == 0 || std::memcmp(a.res[k].slots, b.res[k].slots, a.res[k].n_slots * sizeof(bcir_exact_slot)) == 0));
    }
  }
}

void test_every_allocation_failed_in_turn() {
  std::vector<PhaseIn> module = {diamond(1), two_serial(), diamond(4)};
  Counting probe;
  bcir_host_allocator a = counting(&probe);
  Run whole = run(module, 1000000, &a);
  CHECK(whole.status == BCIR_EXACT_OK && probe.attempts > 0 && probe.live == 0);
  Run plain = run(module, 1000000);
  for (size_t k = 0; k < module.size(); k++) CHECK(whole.res[k].incumbent == plain.res[k].incumbent);
  size_t refused = 0;
  for (size_t k = 1; k <= probe.attempts; k++) {
    Counting c; c.fail_at = k;
    bcir_host_allocator fa = counting(&c);
    Run r = run(module, 1000000, &fa);
    if (r.status != BCIR_EXACT_ERR_NOMEM || c.live != 0) {
      std::fprintf(stderr, "allocation %zu of %zu failed: status %d, %zu live\n", k, probe.attempts, r.status, c.live);
      failures++;
    }
    refused++;
  }
  CHECK(refused == probe.attempts);
}

// --- the text driver -------------------------------------------------------------------------------------------------

bool list_of(const std::string &s, std::vector<uint32_t> &out) {
  if (s == "-") return true;
  std::stringstream ss(s); std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos || item.size() > 9) return false;
    out.push_back(static_cast<uint32_t>(std::stoul(item)));
  }
  return true;
}

void print(const std::vector<PhaseIn> &phases, uint64_t budget) {
  Run r = run(phases, budget);
  if (r.status != BCIR_EXACT_OK) { std::printf("ERR %d\n", static_cast<int>(r.status)); return; }
  for (const bcir_exact_phase_result &p : r.res) {
    std::printf("phase %u heuristic %lld incumbent %lld lower %lld %s expansions %llu bounds %lld,%lld,%lld,%lld slots ",
                p.phase_id, static_cast<long long>(p.heuristic), static_cast<long long>(p.incumbent),
                static_cast<long long>(p.lower_bound), p.optimal ? "optimal" : "budget",
                static_cast<unsigned long long>(p.expansions), static_cast<long long>(p.bounds[0]),
                static_cast<long long>(p.bounds[1]), static_cast<long long>(p.bounds[2]),
                static_cast<long long>(p.bounds[3]));
    for (size_t k = 0; k < p.n_slots; k++)
      std::printf("%s%u:%d:%lld:%lld", k ? "," : "", p.slots[k].claim, p.slots[k].stream,
                  static_cast<long long>(p.slots[k].start), static_cast<long long>(p.slots[k].finish));
    std::printf("\n");
  }
  std::printf("total heuristic %lld incumbent %lld lower %lld %s expansions %llu\n",
              static_cast<long long>(r.total.heuristic), static_cast<long long>(r.total.incumbent),
              static_cast<long long>(r.total.lower_bound), r.total.optimal ? "optimal" : "budget",
              static_cast<unsigned long long>(r.total.expansions));
}

int tool_fail(const char *why) { std::printf("TOOL %s\n", why); return 2; }

int drive(const char *path) {
  std::ifstream in(path);
  if (!in) return tool_fail("cannot read the file");
  bool open = false; uint64_t budget = 0; std::vector<PhaseIn> phases; std::string line;
  try {
    while (std::getline(in, line)) {
      std::istringstream ls(line); std::string kw;
      if (!(ls >> kw)) continue;
      if (kw == "budget") {
        if (open) print(phases, budget);
        phases.clear(); open = true; ls >> budget;
      } else if (kw == "phase" && open) {
        PhaseIn p; ls >> p.id >> p.domains >> p.knee >> p.heuristic; phases.push_back(p);
      } else if (kw == "claim" && !phases.empty()) {
        ClaimIn c{}; std::string preds, streams; ls >> c.id >> c.dur >> preds >> streams;
        if (!list_of(preds, c.preds) || !list_of(streams, c.streams)) return tool_fail("malformed list");
        phases.back().claims.push_back(c);
      } else if (kw == "hslot" && !phases.empty()) {
        bcir_exact_slot s{}; ls >> s.claim >> s.stream >> s.start >> s.finish; phases.back().hslots.push_back(s);
      } else {
        return tool_fail("malformed line");
      }
      if (ls.fail()) return tool_fail("malformed line");
      std::string rest;
      if (ls >> rest) return tool_fail("trailing text");
    }
  } catch (...) {
    return tool_fail("malformed number");
  }
  if (open) print(phases, budget);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 2) return drive(argv[1]);
  if (argc != 1) { std::fprintf(stderr, "usage: test_exact [FILE]\n"); return 2; }
  test_a_phase_the_search_closes();
  test_the_budget_stops_a_phase_and_leaves_the_rest_pending();
  test_an_empty_module_and_an_empty_phase();
  test_every_refusal();
  test_the_search_is_deterministic();
  test_every_allocation_failed_in_turn();
  if (failures) { std::printf("test_exact: %d FAILED\n", failures); return 1; }
  std::printf("test_exact: OK\n");
  return 0;
}
