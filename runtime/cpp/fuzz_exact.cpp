//===- fuzz_exact.cpp - the exact scheduler's façade, under libFuzzer or replayed ---===//
//
// PORT-5 (bcir_exact.h). The input's octets decode to a module of up to three phases, each up to eight claims over
// up to three wave streams and the tail, with predecessors, eligible streams, a heuristic placement, a budget and,
// on request, an allocation the allocator refuses. Most inputs are well formed by construction (predecessors point
// back, the heuristic is a serial schedule); a flag octet adds what the façade must refuse -- a back edge, a stream
// set left empty, a repeated id, work past 64 bits, a heuristic slot moved. What must hold, whatever the octets:
//
//   - the status is one of the façade's, and nothing the allocator gave is still live when the call returns;
//   - a well-formed module with no refused allocation is scheduled; CYCLIC only where a back edge was added,
//     INVALID only where the module was made malformed;
//   - NOMEM only where an allocation was refused;
//   - every phase scheduled satisfies L <= U <= H, optimal exactly when L == U, every root bound <= L, and its
//     placement is a schedule of the phase spanning U -- checked here, sharing nothing with the solver;
//   - the totals are the phases' sums, the expansions within the budget, and a second run gives the same octets.
//
// Built with -DBCIR_LIBFUZZER under -fsanitize=fuzzer (tools/cpp/check_solver.sh) it is a libFuzzer target; built
// without, `fuzz_exact` replays the files it is given, or with none a fixed stream of generated inputs, and prints
// what it saw (the CMake and BCIR Make builds, and the gate's sanitized replay).
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bcir_exact.h"

namespace {

[[noreturn]] void violated(const char *what) {
  std::fprintf(stderr, "fuzz_exact: invariant violated: %s\n", what);
  std::abort();
}
#define HOLD(cond) \
  do {             \
    if (!(cond)) violated(#cond); \
  } while (0)

struct Cursor {
  const uint8_t *p; size_t n;
  uint8_t next() { if (!n) return 0; n--; return *p++; }
};

struct Counting { size_t attempts = 0, fail_at = 0, live = 0; };
void *count_alloc(void *ctx, size_t n) {
  Counting *c = static_cast<Counting *>(ctx);
  if (++c->attempts == c->fail_at) return nullptr;
  void *q = std::malloc(n ? n : 1);
  if (q) c->live++;
  return q;
}
void *count_realloc(void *ctx, void *q, size_t n) {
  Counting *c = static_cast<Counting *>(ctx);
  if (++c->attempts == c->fail_at) return nullptr;
  if (!q) { void *r = std::malloc(n ? n : 1); if (r) c->live++; return r; }
  return std::realloc(q, n ? n : 1);
}
void count_free(void *ctx, void *q) {
  Counting *c = static_cast<Counting *>(ctx);
  if (q) { c->live--; std::free(q); }
}

struct PhaseIn {
  uint32_t id = 0, domains = 1, knee = 0;
  int64_t heuristic = 0;
  std::vector<uint32_t> ids;
  std::vector<int64_t> durs;
  std::vector<std::vector<uint32_t>> preds, streams;
  std::vector<bcir_exact_slot> hslots;
};

struct Module {
  std::vector<PhaseIn> phases;
  uint64_t budget = 0;
  size_t fail_at = 0;
  bool malformed = false, back_edge = false;
};

constexpr uint64_t kBudgets[] = {0, 1, 2, 3, 5, 8, 17, 50, 200, 1000, 100000};

Module decode(const uint8_t *data, size_t size) {
  Cursor in{data, size};
  Module m;
  uint8_t flags = in.next();
  size_t n_phases = flags & 3u;
  bool corrupt = (flags & 4u) != 0;
  m.budget = kBudgets[in.next() % (sizeof kBudgets / sizeof kBudgets[0])];
  if (flags & 8u) m.fail_at = 1u + in.next() % 48u;   // most runs make a few dozen allocations
  for (size_t ph = 0; ph < n_phases; ph++) {
    PhaseIn p;
    p.id = static_cast<uint32_t>(ph + 1);
    uint8_t shape = in.next();
    p.domains = 1u + shape % 3u;
    p.knee = (shape / 3u) % (p.domains + 1u);
    size_t n = in.next() % 9u;
    for (size_t k = 0; k < n; k++) {
      uint8_t idb = in.next(), durb = in.next(), predb = in.next(), streamb = in.next();
      uint32_t id = static_cast<uint32_t>(10 + 2 * k + (idb & 1u));
      if (corrupt && idb == 0xFF && k) { id = p.ids.back(); m.malformed = true; }   // a repeated id
      int64_t dur = durb % 16u;
      if (corrupt && durb == 0xFE) { dur = INT64_MAX / 3; m.malformed = true; }      // work past 64 bits
      std::vector<uint32_t> preds;
      for (size_t j = 0; j < k && j < 7; j++) if (predb & (1u << j)) preds.push_back(p.ids[j]);
      // the tail alone or wave streams alone, as the oracle's dispatch gives them; both, or none, is malformed
      std::vector<uint32_t> streams;
      for (uint32_t s = 0; s < p.domains; s++) if (streamb & (1u << s)) streams.push_back(s);
      bool tail = (streamb & (1u << p.domains)) != 0;
      if (corrupt && (streamb & 0x40u) && tail && !streams.empty()) {
        streams.push_back(p.domains); m.malformed = true;                              // the tail and a wave
      } else if (streams.empty() && corrupt && (streamb & 0x80u)) {
        m.malformed = true;                                                            // no stream at all
      } else if (streams.empty() || (tail && (streamb & 0x80u))) {
        streams.assign(1, p.domains);
      }
      p.ids.push_back(id); p.durs.push_back(dur); p.preds.push_back(preds); p.streams.push_back(streams);
    }
    if (corrupt && n >= 2 && (shape & 0x80u)) {   // a back edge: the first claim waits for the last -- a cycle where
      p.preds[0].push_back(p.ids[n - 1]);           // the last waits on the first, else a heuristic (the index order)
      m.back_edge = m.malformed = true;             // that is not a schedule
    }
    // the heuristic: every claim in index order, one after another, each on its first stream
    int64_t t = 0;
    for (size_t k = 0; k < n; k++) {
      if (p.durs[k] > INT64_MAX / 4 - t) break;
      uint32_t s = p.streams[k].empty() ? 0 : p.streams[k][0];
      p.hslots.push_back(bcir_exact_slot{p.ids[k], s == p.domains ? -1 : static_cast<int32_t>(s), t, t + p.durs[k]});
      t += p.durs[k];
    }
    p.heuristic = t;
    if (p.hslots.size() != n) m.malformed = true;
    if (corrupt && !p.hslots.empty() && (shape & 0x40u)) {   // one heuristic slot moved: earlier, off its stream, or longer
      bcir_exact_slot &s = p.hslots[in.next() % p.hslots.size()];
      uint8_t how = in.next() % 3u;
      if (how == 0) { s.start -= 1; s.finish -= 1; }
      else if (how == 1) s.stream = static_cast<int32_t>(p.domains);
      else s.finish += 1;
      m.malformed = true;   // perhaps still a schedule (a zero-length slot moved); the call decides
    }
    m.phases.push_back(p);
  }
  return m;
}

struct Outcome {
  bcir_exact_status status;
  std::vector<bcir_exact_phase_result> res;
  std::vector<std::vector<bcir_exact_slot>> slots;
  bcir_exact_result total;
};

Outcome schedule(const Module &m, const bcir_host_allocator *allocator) {
  std::vector<std::vector<bcir_exact_claim>> claims(m.phases.size());
  std::vector<bcir_exact_phase> ins(m.phases.size());
  Outcome o{BCIR_EXACT_OK, std::vector<bcir_exact_phase_result>(m.phases.size()),
            std::vector<std::vector<bcir_exact_slot>>(m.phases.size()), bcir_exact_result{}};
  for (size_t k = 0; k < m.phases.size(); k++) {
    const PhaseIn &p = m.phases[k];
    for (size_t i = 0; i < p.ids.size(); i++)
      claims[k].push_back(bcir_exact_claim{p.ids[i], p.durs[i], p.preds[i].data(), p.preds[i].size(),
                                           p.streams[i].data(), p.streams[i].size()});
    ins[k] = bcir_exact_phase{p.id, claims[k].data(), claims[k].size(), p.domains, p.knee, p.heuristic,
                              p.hslots.data(), p.hslots.size()};
    o.slots[k].assign(p.ids.size(), bcir_exact_slot{});
    o.res[k].slots = o.slots[k].data(); o.res[k].cap_slots = o.slots[k].size();
  }
  o.status = bcir_exact_schedule(ins.data(), ins.size(), m.budget, allocator, o.res.data(), &o.total);
  return o;
}

void check_phase(const PhaseIn &p, const bcir_exact_phase_result &r) {
  HOLD(r.phase_id == p.id && r.heuristic == p.heuristic);
  HOLD(r.lower_bound <= r.incumbent && r.incumbent <= r.heuristic);
  HOLD((r.optimal != 0) == (r.lower_bound == r.incumbent));
  for (int64_t b : r.bounds) HOLD(b >= 0 && b <= r.lower_bound);
  HOLD(r.n_slots == p.ids.size());
  int64_t span = 0;
  for (size_t a = 0; a < r.n_slots; a++) {
    const bcir_exact_slot &s = r.slots[a];
    size_t i = 0;
    while (i < p.ids.size() && p.ids[i] != s.claim) i++;
    HOLD(i < p.ids.size());
    HOLD(s.start >= 0 && s.finish - s.start == p.durs[i]);
    uint32_t stream = s.stream < 0 ? p.domains : static_cast<uint32_t>(s.stream);
    bool eligible = false;
    for (uint32_t e : p.streams[i]) eligible = eligible || e == stream;
    HOLD(eligible);
    for (size_t b = 0; b < r.n_slots; b++) {
      const bcir_exact_slot &o = r.slots[b];
      HOLD(b == a || o.claim != s.claim);
      for (uint32_t q : p.preds[i]) HOLD(o.claim != q || o.finish <= s.start);
      HOLD(b == a || o.stream != s.stream || s.finish == s.start || o.finish == o.start || s.start >= o.finish ||
           o.start >= s.finish);
    }
    if (s.finish > span) span = s.finish;
  }
  HOLD(span == r.incumbent);
}

struct Tally { size_t inputs = 0, scheduled = 0, searched = 0, closed = 0, stopped = 0, refused = 0, nomem = 0; };
Tally tally;

int one_input(const uint8_t *data, size_t size) {
  Module m = decode(data, size);
  Counting count; count.fail_at = m.fail_at;
  bcir_host_allocator a{&count, count_alloc, count_realloc, count_free};
  Outcome o = schedule(m, &a);
  tally.inputs++;
  HOLD(count.live == 0);
  HOLD(o.status >= BCIR_EXACT_OK && o.status <= BCIR_EXACT_ERR_INTERNAL);
  HOLD(o.status != BCIR_EXACT_ERR_INTERNAL && o.status != BCIR_EXACT_ERR_CAPACITY);
  if (o.status == BCIR_EXACT_ERR_NOMEM) { HOLD(m.fail_at != 0); tally.nomem++; return 0; }
  if (o.status == BCIR_EXACT_ERR_CYCLIC) { HOLD(m.back_edge); tally.refused++; return 0; }
  if (o.status == BCIR_EXACT_ERR_INVALID) { HOLD(m.malformed); tally.refused++; return 0; }
  HOLD(o.status == BCIR_EXACT_OK);   // a back edge or a moved slot may leave a schedule (zero lengths): scheduled
  int64_t h = 0, u = 0, l = 0;
  uint64_t spent = 0;
  bool all = true;
  for (size_t k = 0; k < m.phases.size(); k++) {
    check_phase(m.phases[k], o.res[k]);
    h += o.res[k].heuristic; u += o.res[k].incumbent; l += o.res[k].lower_bound;
    spent += o.res[k].expansions; all = all && o.res[k].optimal;
    tally.searched += o.res[k].expansions > 0;
    tally.closed += o.res[k].optimal != 0;
    tally.stopped += o.res[k].optimal == 0;
  }
  HOLD(o.total.heuristic == h && o.total.incumbent == u && o.total.lower_bound == l);
  HOLD(o.total.expansions == spent && spent <= m.budget && (o.total.optimal != 0) == all);
  tally.scheduled++;
  Outcome again = schedule(m, nullptr);   // the default allocator, and the same answer
  HOLD(again.status == BCIR_EXACT_OK);
  for (size_t k = 0; k < m.phases.size(); k++) {
    HOLD(again.res[k].incumbent == o.res[k].incumbent && again.res[k].lower_bound == o.res[k].lower_bound);
    HOLD(again.res[k].expansions == o.res[k].expansions && again.res[k].n_slots == o.res[k].n_slots);
    HOLD(o.res[k].n_slots == 0 ||
         std::memcmp(again.res[k].slots, o.res[k].slots, o.res[k].n_slots * sizeof(bcir_exact_slot)) == 0);
  }
  return 0;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) { return one_input(data, size); }

#ifndef BCIR_LIBFUZZER
int main(int argc, char **argv) {
  if (argc > 1) {
    for (int k = 1; k < argc; k++) {
      std::FILE *f = std::fopen(argv[k], "rb");
      if (!f) { std::fprintf(stderr, "fuzz_exact: cannot read %s\n", argv[k]); return 2; }
      std::vector<uint8_t> bytes;
      int c;
      while ((c = std::fgetc(f)) != EOF) bytes.push_back(static_cast<uint8_t>(c));
      std::fclose(f);
      one_input(bytes.data(), bytes.size());
    }
  } else {
    uint64_t state = 0x9E3779B97F4A7C15ull;   // a fixed stream: the same inputs on every host
    std::vector<uint8_t> bytes;
    for (size_t k = 0; k < 6000; k++) {
      bytes.clear();
      size_t len = 0;
      for (size_t j = 0; j < 128; j++) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        if (j == 0) len = 2 + state % 120;
        else if (j <= len) bytes.push_back(static_cast<uint8_t>(state >> 24));
      }
      one_input(bytes.data(), bytes.size());
    }
  }
  std::printf("fuzz_exact: %zu inputs, %zu scheduled (%zu phases searched, %zu closed, %zu stopped on the budget), "
              "%zu refused, %zu refused an allocation\n", tally.inputs, tally.scheduled, tally.searched, tally.closed,
              tally.stopped, tally.refused, tally.nomem);
  // a replay that exercised nothing proves nothing
  if (argc == 1 && (!tally.searched || !tally.closed || !tally.stopped || !tally.refused || !tally.nomem)) {
    std::printf("fuzz_exact: the generated stream left an outcome unexercised\n");
    return 1;
  }
  return 0;
}
#endif
