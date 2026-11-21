//===- bcir_exact.cpp - the bounded exact scheduler (the C++ twin of bcir/gem/exact.py) -===//
//
// PORT-5 (bcir_exact.h). The search is the oracle's `_PhaseSearch.solve` step for step -- the same root bounds, the
// same node bound, the same symmetry rules, the same child order and the same stack discipline -- so a run spends the
// oracle's expansions and stops where it stops. Failures are values (`bcir::expected`), every container draws from
// one phase's arena over the caller's allocator, and the C entry point catches every exception the standard library
// may raise (an allocation refused) and reports it as a status.
//===----------------------------------------------------------------------===//
#include "bcir_exact.h"
#include "bcir_solver.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <new>
#include <queue>
#include <utility>
#include <vector>

namespace {

using bcir::expected;
using bcir::unexpected;
using bcir::solver::host_resource;
using bcir::solver::phase_arena;
using bcir::solver::pvec;


int64_t ceil_div(int64_t a, int64_t b) { return a >= 0 ? (a + b - 1) / b : -((-a) / b); }

struct Node {
  pvec<int64_t> free;      // per stream: wave 0..domains-1, then the tail
  pvec<int64_t> finish;    // per claim index: its finish, or -1 unplaced
  size_t placed;
  int64_t work_left, tail_left;
  pvec<int32_t> indegree;  // per claim index
  pvec<bcir_exact_slot> slots;
  explicit Node(std::pmr::memory_resource *r) : free(r), finish(r), placed(0), work_left(0), tail_left(0),
                                                indegree(r), slots(r) {}
};

struct Phase {
  // the oracle's `_PhaseSearch` fields, by claim index
  std::pmr::memory_resource *r;
  size_t n;
  uint32_t domains, knee, tail;
  pvec<uint32_t> ids;
  pvec<int64_t> dur;
  pvec<pvec<size_t>> preds, succs, eligible;
  pvec<size_t> order;
  pvec<char> on_tail;
  pvec<int64_t> tail_path;
  pvec<std::ptrdiff_t> twin_before;   // -1 none
  explicit Phase(std::pmr::memory_resource *res)
      : r(res), n(0), domains(0), knee(0), tail(0), ids(res), dur(res), preds(res), succs(res), eligible(res),
        order(res), on_tail(res), tail_path(res), twin_before(res) {}
};

size_t claim_index(const Phase &p, uint32_t id) {
  for (size_t k = 0; k < p.ids.size(); k++) if (p.ids[k] == id) return k;
  return p.n;
}

// The placement the caller hands in as the heuristic must be a schedule of the phase -- every claim once, on a
// stream it may run on (-1 the tail), for its duration, after its predecessors, no two claims of nonzero length on
// one stream at once -- whose span is the heuristic: the search starts from it as its incumbent and returns it when
// nothing beats it, so a placement that is not a schedule would come back certified.
bool valid_placement(const Phase &p, const bcir_exact_slot *slots, size_t n, int64_t span) {
  if (n != p.n || span < 0) return false;
  pvec<std::ptrdiff_t> at(p.n, -1, p.r);
  int64_t last = 0;
  for (size_t k = 0; k < n; k++) {
    const bcir_exact_slot &s = slots[k];
    size_t i = claim_index(p, s.claim);
    if (i == p.n || at[i] >= 0) return false;
    at[i] = static_cast<std::ptrdiff_t>(k);
    if (s.stream < -1 || (s.stream >= 0 && static_cast<uint32_t>(s.stream) >= p.domains)) return false;
    size_t stream = s.stream == -1 ? p.tail : static_cast<size_t>(s.stream);
    if (std::find(p.eligible[i].begin(), p.eligible[i].end(), stream) == p.eligible[i].end()) return false;
    if (s.start < 0 || s.finish < s.start || s.finish - s.start != p.dur[i]) return false;
    last = std::max(last, s.finish);
  }
  if (last != span) return false;
  for (size_t i = 0; i < p.n; i++)
    for (size_t q : p.preds[i])
      if (slots[at[q]].finish > slots[at[i]].start) return false;
  pvec<size_t> by(p.r);
  for (size_t k = 0; k < n; k++) if (slots[k].finish > slots[k].start) by.push_back(k);
  std::sort(by.begin(), by.end(), [&](size_t a, size_t b) {
    return slots[a].stream != slots[b].stream ? slots[a].stream < slots[b].stream : slots[a].start < slots[b].start; });
  for (size_t k = 1; k < by.size(); k++)
    if (slots[by[k]].stream == slots[by[k - 1]].stream && slots[by[k - 1]].finish > slots[by[k]].start) return false;
  return true;
}

expected<bool, bcir_exact_status> build(Phase &p, const bcir_exact_phase &in) {
  p.n = in.n_claims; p.domains = in.domains; p.knee = in.knee; p.tail = in.domains;
  if (in.n_claims && !in.claims) return unexpected(BCIR_EXACT_ERR_INVALID);
  int64_t total = 0;
  for (size_t i = 0; i < p.n; i++) {
    const bcir_exact_claim &c = in.claims[i];
    if (c.duration < 0 || (c.n_preds && !c.preds) || (c.n_streams && !c.streams) || c.n_streams == 0)
      return unexpected(BCIR_EXACT_ERR_INVALID);
    for (size_t j = 0; j < i; j++) if (p.ids[j] == c.id) return unexpected(BCIR_EXACT_ERR_INVALID);
    if (c.duration > INT64_MAX - total) return unexpected(BCIR_EXACT_ERR_INVALID);
    total += c.duration;
    p.ids.push_back(c.id); p.dur.push_back(c.duration);
  }
  // Every time the search forms is at most the phase's total work (a claim starts at 0 or at another's finish), and
  // the widest sum it forms is the wave's free times plus the work left: (domains + 1) totals. Refused where that
  // would leave 64 bits, so no arithmetic below can overflow.
  if (total > INT64_MAX / (static_cast<int64_t>(p.domains) + 2)) return unexpected(BCIR_EXACT_ERR_INVALID);
  auto index_of = [&](uint32_t id) { return claim_index(p, id); };
  p.preds.resize(p.n, pvec<size_t>(p.r)); p.succs.resize(p.n, pvec<size_t>(p.r)); p.eligible.resize(p.n, pvec<size_t>(p.r));
  for (size_t i = 0; i < p.n; i++) {
    const bcir_exact_claim &c = in.claims[i];
    for (size_t k = 0; k < c.n_preds; k++) {
      size_t j = index_of(c.preds[k]);
      if (j == p.n || j == i) return unexpected(BCIR_EXACT_ERR_INVALID);
      p.preds[i].push_back(j); p.succs[j].push_back(i);
    }
    bool tail = false, wave = false;
    for (size_t k = 0; k < c.n_streams; k++) {
      if (c.streams[k] > p.tail) return unexpected(BCIR_EXACT_ERR_INVALID);
      (c.streams[k] == p.tail ? tail : wave) = true;
      p.eligible[i].push_back(c.streams[k]);
    }
    // the tail alone or wave streams alone, as the oracle's dispatch gives them: the work bound charges every claim
    // that may run on a wave to the waves, which a claim the tail could also take would make no bound
    if (tail && wave) return unexpected(BCIR_EXACT_ERR_INVALID);
  }
  // a topological order: Kahn's, the smallest claim id first (the oracle's heap)
  pvec<int32_t> indeg(p.r);
  for (size_t i = 0; i < p.n; i++) indeg.push_back(static_cast<int32_t>(p.preds[i].size()));
  auto later = [&](size_t a, size_t b) { return p.ids[a] > p.ids[b]; };
  std::priority_queue<size_t, pvec<size_t>, decltype(later)> ready(later, pvec<size_t>(p.r));
  for (size_t i = 0; i < p.n; i++) if (indeg[i] == 0) ready.push(i);
  while (!ready.empty()) {
    size_t c = ready.top(); ready.pop(); p.order.push_back(c);
    for (size_t s : p.succs[c]) if (--indeg[s] == 0) ready.push(s);
  }
  if (p.order.size() != p.n) return unexpected(BCIR_EXACT_ERR_CYCLIC);
  for (size_t i = 0; i < p.n; i++) p.on_tail.push_back(p.eligible[i].size() == 1 && p.eligible[i][0] == p.tail);
  p.tail_path.assign(p.n, 0);
  for (size_t k = p.n; k-- > 0;) {
    size_t c = p.order[k]; int64_t best = 0;
    for (size_t s : p.succs[c]) best = std::max(best, p.tail_path[s]);
    p.tail_path[c] = p.dur[c] + best;
  }
  // twins: the same duration, predecessors (in order), successors (as a set) and eligibility; the most recent
  // earlier one is placed first
  auto sorted_succ_ids = [&](size_t i) {
    pvec<uint32_t> v(p.r); for (size_t s : p.succs[i]) v.push_back(p.ids[s]); std::sort(v.begin(), v.end()); return v; };
  for (size_t i = 0; i < p.n; i++) {
    std::ptrdiff_t twin = -1;
    pvec<uint32_t> si = sorted_succ_ids(i);
    for (size_t j = i; j-- > 0;) {
      if (p.dur[j] != p.dur[i] || p.preds[j] != p.preds[i] || p.eligible[j] != p.eligible[i]) continue;
      if (sorted_succ_ids(j) != si) continue;
      twin = static_cast<std::ptrdiff_t>(j); break; }
    p.twin_before.push_back(twin);
  }
  return true;
}

// `root_bounds`: critical-path, work/capacity, bandwidth/knee, tail-serial.
void root_bounds(const Phase &p, int64_t out[BCIR_EXACT_BOUNDS]) {
  pvec<int64_t> head(p.n, 0, p.r);
  int64_t critical = 0;
  for (size_t c : p.order) {
    int64_t best = 0;
    for (size_t q : p.preds[c]) best = std::max(best, head[q]);
    head[c] = best + p.dur[c]; critical = std::max(critical, head[c]);
  }
  int64_t wave = 0, band = 0, tailw = 0; bool any_band = false;
  for (size_t i = 0; i < p.n; i++) {
    if (p.on_tail[i]) { tailw += p.dur[i]; continue; }
    wave += p.dur[i];
    bool is_band = p.knee < p.domains && p.eligible[i].size() == p.knee;
    for (size_t k = 0; is_band && k < p.knee; k++) if (p.eligible[i][k] != k) is_band = false;
    if (is_band) { band += p.dur[i]; any_band = true; }
  }
  out[0] = critical;
  out[1] = ceil_div(wave, std::max<int64_t>(1, p.domains));
  out[2] = any_band ? ceil_div(band, std::max<int64_t>(1, p.knee)) : 0;
  out[3] = tailw;
}

struct PhaseOut { int64_t upper, lower; bool optimal; uint64_t expansions; pvec<bcir_exact_slot> slots; };

expected<bool, bcir_exact_status> solve(const Phase &p, int64_t heuristic, const pvec<bcir_exact_slot> &hslots,
                                        uint64_t budget, const int64_t bounds[BCIR_EXACT_BOUNDS], PhaseOut &out) {
  int64_t root = *std::max_element(bounds, bounds + BCIR_EXACT_BOUNDS);
  int64_t upper = heuristic;
  out.slots.assign(hslots.begin(), hslots.end());
  if (upper <= root || p.n == 0) { out.upper = out.lower = upper; out.optimal = true; out.expansions = 0; return true; }
  const size_t streams = p.domains + 1u;
  pvec<std::pair<int64_t, Node>> stack(p.r);
  {
    Node root_node(p.r);
    root_node.free.assign(streams, 0); root_node.finish.assign(p.n, -1);
    for (size_t i = 0; i < p.n; i++) { if (p.on_tail[i]) root_node.tail_left += p.dur[i]; else root_node.work_left += p.dur[i]; }
    for (size_t i = 0; i < p.n; i++) root_node.indegree.push_back(static_cast<int32_t>(p.preds[i].size()));
    stack.emplace_back(root, std::move(root_node));
  }
  uint64_t expansions = 0;
  struct Child { int64_t finish, bound; Node node; };
  pvec<Child> children(p.r);
  pvec<int64_t> seen_free(p.r);
  while (!stack.empty()) {
    if (stack.back().first >= upper) { stack.pop_back(); continue; }
    if (expansions >= budget) break;   // the frontier stays: the oracle's resumable stop
    Node node = std::move(stack.back().second); stack.pop_back();
    expansions++;
    if (node.placed == p.n) {
      int64_t span = *std::max_element(node.free.begin(), node.free.end());
      if (span < upper) { upper = span; out.slots.assign(node.slots.begin(), node.slots.end()); }
      continue;
    }
    children.clear();
    for (size_t c = 0; c < p.n; c++) {
      if (node.indegree[c] != 0 || node.finish[c] >= 0) continue;
      std::ptrdiff_t twin = p.twin_before[c];
      if (twin >= 0 && node.finish[static_cast<size_t>(twin)] < 0) continue;   // its lower-id twin goes first
      int64_t release = 0;
      for (size_t q : p.preds[c]) release = std::max(release, node.finish[q]);
      seen_free.clear();
      for (size_t s : p.eligible[c]) {
        bool empty_by_release = node.free[s] <= release;
        if (empty_by_release && std::find(seen_free.begin(), seen_free.end(), release) != seen_free.end()) continue;
        if (empty_by_release) seen_free.push_back(release);
        int64_t start = std::max(node.free[s], release), finish = start + p.dur[c];
        Node child(p.r);
        child.free = node.free; child.free[s] = finish;
        child.finish = node.finish; child.finish[c] = finish;
        child.indegree = node.indegree;
        for (size_t succ : p.succs[c]) child.indegree[succ]--;
        bool tail = s == p.tail;
        child.work_left = node.work_left - (tail ? 0 : p.dur[c]);
        child.tail_left = node.tail_left - (tail ? p.dur[c] : 0);
        child.placed = node.placed + 1;
        // the node bound: the stack at this node
        int64_t frontier_t = *std::max_element(child.free.begin(), child.free.end());
        int64_t wave_free = 0;
        for (size_t k = 0; k < p.domains; k++) wave_free += child.free[k];
        int64_t cap = ceil_div(wave_free + child.work_left, std::max<int64_t>(1, p.domains));
        int64_t tail_bound = child.free[p.tail] + child.tail_left;
        int64_t path = frontier_t;
        for (size_t o = 0; o < p.n; o++) {
          if (child.finish[o] >= 0) continue;
          int64_t est = 0;
          for (size_t q : p.preds[o]) if (child.finish[q] >= 0 && child.finish[q] > est) est = child.finish[q];
          path = std::max(path, est + p.tail_path[o]);
        }
        int64_t bound = std::max({frontier_t, cap, tail_bound, path, root});
        if (bound >= upper) continue;
        child.slots = node.slots;
        child.slots.push_back(bcir_exact_slot{p.ids[c], tail ? -1 : static_cast<int32_t>(s), start, finish});
        children.push_back(Child{finish, bound, std::move(child)});
      }
    }
    // earliest finish first: the oracle sorts (finish, bound) descending, stably, and pushes in that order
    std::stable_sort(children.begin(), children.end(), [](const Child &a, const Child &b) {
      return a.finish != b.finish ? a.finish > b.finish : a.bound > b.bound; });
    for (Child &ch : children) stack.emplace_back(ch.bound, std::move(ch.node));
  }
  bool live = false; int64_t least = 0;
  for (auto &e : stack) if (e.first < upper) { if (!live || e.first < least) least = e.first; live = true; }
  out.upper = upper; out.expansions = expansions;
  if (!live) { out.lower = upper; out.optimal = true; return true; }
  out.lower = std::max(root, std::min(upper, least)); out.optimal = false;
  return true;
}

bcir_exact_status schedule(const bcir_exact_phase *phases, size_t n_phases, uint64_t budget,
                           const bcir_host_allocator *allocator, bcir_exact_phase_result *results,
                           bcir_exact_result *total) {
  host_resource host(allocator);
  bool stopped = false;   // an earlier phase stopped on the budget: the rest are pending
  uint64_t left = budget, spent = 0;
  bcir_exact_result t{0, 0, 0, 1, 0, budget};
  for (size_t ph = 0; ph < n_phases; ph++) {
    phase_arena pool(&host);   // one arena per phase, released at its end
    const bcir_exact_phase &in = phases[ph];
    bcir_exact_phase_result &res = results[ph];
    if ((in.n_heuristic_slots && !in.heuristic_slots) || (res.cap_slots && !res.slots)) return BCIR_EXACT_ERR_INVALID;
    if (res.cap_slots < in.n_claims || in.n_heuristic_slots != in.n_claims) return in.n_heuristic_slots != in.n_claims
                                                                                  ? BCIR_EXACT_ERR_INVALID : BCIR_EXACT_ERR_CAPACITY;
    Phase p(&pool);
    auto built = build(p, in);
    if (!built) return built.error();
    if (!valid_placement(p, in.heuristic_slots, in.n_heuristic_slots, in.heuristic)) return BCIR_EXACT_ERR_INVALID;
    if (in.heuristic > INT64_MAX - t.heuristic) return BCIR_EXACT_ERR_INVALID;   // the module's totals in 64 bits
    pvec<bcir_exact_slot> hslots(in.heuristic_slots, in.heuristic_slots + in.n_heuristic_slots, &pool);
    int64_t bounds[BCIR_EXACT_BOUNDS];
    root_bounds(p, bounds);
    PhaseOut out{0, 0, false, 0, pvec<bcir_exact_slot>(&pool)};
    if (stopped) {   // pending: the heuristic stands, the root stack its bound
      int64_t root = *std::max_element(bounds, bounds + BCIR_EXACT_BOUNDS);
      out.upper = in.heuristic; out.lower = std::min(in.heuristic, root); out.optimal = out.upper == out.lower;
      out.expansions = 0; out.slots.assign(hslots.begin(), hslots.end());
    } else {
      auto solved = solve(p, in.heuristic, hslots, left, bounds, out);
      if (!solved) return solved.error();
      left -= out.expansions;
    }
    if (out.lower > out.upper || out.upper > in.heuristic || out.slots.size() > res.cap_slots) return BCIR_EXACT_ERR_INTERNAL;
    res.phase_id = in.phase_id; res.heuristic = in.heuristic; res.incumbent = out.upper; res.lower_bound = out.lower;
    res.optimal = out.optimal; res.expansions = out.expansions;
    std::copy(bounds, bounds + BCIR_EXACT_BOUNDS, res.bounds);
    std::copy(out.slots.begin(), out.slots.end(), res.slots); res.n_slots = out.slots.size();
    t.heuristic += in.heuristic; t.incumbent += out.upper; t.lower_bound += out.lower;
    spent += out.expansions;
    if (!out.optimal) { stopped = true; t.optimal = 0; }
  }
  t.expansions = spent;
  *total = t;
  return BCIR_EXACT_OK;
}

}  // namespace

extern "C" bcir_exact_status bcir_exact_schedule(const bcir_exact_phase *phases, size_t n_phases, uint64_t budget,
                                                 const bcir_host_allocator *allocator, bcir_exact_phase_result *results,
                                                 bcir_exact_result *total) {
  if ((n_phases && (!phases || !results)) || !total) return BCIR_EXACT_ERR_INVALID;
  if (allocator && !bcir_host_allocator_valid(allocator)) return BCIR_EXACT_ERR_INVALID;
  try {
    return schedule(phases, n_phases, budget, allocator, results, total);
  } catch (const std::bad_alloc &) {
    return BCIR_EXACT_ERR_NOMEM;
  } catch (...) {   // nothing else is thrown by the code above; a defect is reported, never propagated
    return BCIR_EXACT_ERR_INTERNAL;
  }
}
