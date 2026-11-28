//===- fuzz_csp.cpp - the CSP solver's façade, under libFuzzer or replayed -------===//
//
// PORT-6 (bcir_csp.h). The input's octets decode to a small model -- up to five variables over domains within
// [-4, 7], up to four constraints of the three kinds, an objective, a budget, perhaps an incumbent and, on request,
// an allocation the allocator refuses -- small enough that this file enumerates every assignment itself. What must
// hold, whatever the octets:
//
//   - the status is one of the façade's, NOMEM only where an allocation was refused, and nothing the allocator gave
//     is still live when the call returns;
//   - an incumbent is refused exactly when it is not a solution, by this file's own reading of the constraints;
//   - an answer's values are a solution, its bound at most its objective, `optimal` the brute-force optimum (and
//     its bound that optimum), `infeasible` only where brute force finds no solution, `budget` an objective no
//     better than the optimum over a bound no worse;
//   - a second run gives the same answer.
//
// Built with -DBCIR_LIBFUZZER under -fsanitize=fuzzer (tools/cpp/check_solver.sh) it is a libFuzzer target; built
// without, `fuzz_csp` replays the files it is given, or with none a fixed stream of generated inputs.
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bcir_csp.h"

namespace {

[[noreturn]] void violated(const char *what) {
  std::fprintf(stderr, "fuzz_csp: invariant violated: %s\n", what);
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

struct Con {
  int kind;
  std::vector<bcir_csp_term> terms; int relation; int64_t rhs;
  std::vector<bcir_csp_task> tasks; int64_t capacity;
  std::vector<bcir_csp_rect> rects;
};
struct Problem {
  std::vector<bcir_csp_var> vars;
  std::vector<Con> cons;
  std::vector<bcir_csp_term> objective; int64_t offset = 0;
  uint64_t budget = 0; size_t fail_at = 0;
  bool seeded = false; std::vector<int64_t> incumbent;
};

constexpr uint64_t kBudgets[] = {0, 1, 2, 3, 5, 9, 17, 40, 200, 5000, 200000};

Problem decode(const uint8_t *data, size_t size) {
  Cursor in{data, size};
  Problem p;
  uint8_t flags = in.next();
  size_t n = 1 + flags % 5u;
  p.budget = kBudgets[in.next() % (sizeof kBudgets / sizeof kBudgets[0])];
  if (flags & 0x20u) p.fail_at = 1u + in.next() % 40u;
  for (size_t v = 0; v < n; v++) {
    int64_t lo = static_cast<int64_t>(in.next() % 8u) - 4, width = in.next() % 5u;
    p.vars.push_back(bcir_csp_var{lo, std::min<int64_t>(lo + width, 7), static_cast<int64_t>(in.next() % 3u)});
  }
  size_t n_cons = (flags >> 3) % 5u;
  for (size_t k = 0; k < n_cons; k++) {
    Con c{};
    uint8_t kind = in.next();
    c.kind = kind % 3u;
    if (c.kind == BCIR_CSP_LINEAR) {
      size_t terms = 1 + in.next() % 3u;
      for (size_t t = 0; t < terms; t++)
        c.terms.push_back(bcir_csp_term{static_cast<int64_t>(in.next() % 7u) - 3, static_cast<uint32_t>(in.next() % n)});
      c.relation = in.next() % 3u;
      c.rhs = static_cast<int64_t>(in.next() % 17u) - 8;
    } else if (c.kind == BCIR_CSP_CUMULATIVE) {
      size_t tasks = 1 + in.next() % 3u;
      for (size_t t = 0; t < tasks; t++)
        c.tasks.push_back(bcir_csp_task{static_cast<uint32_t>(in.next() % n), static_cast<int64_t>(in.next() % 4u)});
      c.capacity = 1 + in.next() % 2u;
    } else {
      size_t rects = 1 + in.next() % 3u;
      for (size_t t = 0; t < rects; t++) {
        uint8_t g = in.next();
        bcir_csp_rect r{static_cast<uint32_t>(in.next() % n), static_cast<int64_t>(in.next() % 3u),
                        static_cast<uint32_t>(in.next() % n), static_cast<int64_t>(in.next() % 4u),
                        static_cast<uint32_t>(in.next() % n), 1 + static_cast<int64_t>(in.next() % 3u),
                        static_cast<int64_t>(t)};
        if ((g & 3u) == 0 && t) {   // a part of the previous rectangle's object: its group, its y and height
          r.group = c.rects.back().group; r.y = c.rects.back().y; r.h = c.rects.back().h;
        }
        c.rects.push_back(r);
      }
    }
    p.cons.push_back(c);
  }
  size_t terms = in.next() % 4u;
  for (size_t t = 0; t < terms; t++)
    p.objective.push_back(bcir_csp_term{static_cast<int64_t>(in.next() % 9u) - 4, static_cast<uint32_t>(in.next() % n)});
  p.offset = static_cast<int64_t>(in.next() % 11u) - 5;
  if (flags & 0x80u) {
    p.seeded = true;
    for (size_t v = 0; v < n; v++) p.incumbent.push_back(static_cast<int64_t>(in.next() % 12u) - 4);
  }
  return p;
}

// --- the constraints' definitions, this file's own reading --------------------------------------------------------

bool solution(const Problem &p, const int64_t *x) {
  for (size_t v = 0; v < p.vars.size(); v++)
    if (x[v] < p.vars[v].lo || x[v] > p.vars[v].hi) return false;
  for (const Con &c : p.cons) {
    if (c.kind == BCIR_CSP_LINEAR) {
      int64_t total = 0;
      for (const bcir_csp_term &t : c.terms) total += t.coef * x[t.var];
      if (c.relation == BCIR_CSP_LE ? total > c.rhs : c.relation == BCIR_CSP_GE ? total < c.rhs : total != c.rhs)
        return false;
    } else if (c.kind == BCIR_CSP_CUMULATIVE) {
      for (const bcir_csp_task &a : c.tasks) {
        if (a.duration <= 0) continue;
        int64_t t = x[a.start], running = 0;
        for (const bcir_csp_task &b : c.tasks)
          if (b.duration > 0 && x[b.start] <= t && t < x[b.start] + b.duration) running++;
        if (running > c.capacity) return false;
      }
    } else {
      for (size_t i = 0; i < c.rects.size(); i++)
        for (size_t j = i + 1; j < c.rects.size(); j++) {
          const bcir_csp_rect &a = c.rects[i], &b = c.rects[j];
          if (a.group == b.group) continue;
          int64_t ax0 = x[a.x0] + a.a, ax1 = x[a.x1] + a.b, bx0 = x[b.x0] + b.a, bx1 = x[b.x1] + b.b;
          bool time = ax0 < ax1 && bx0 < bx1 && ax0 < bx1 && bx0 < ax1;
          bool space = x[a.y] < x[b.y] + b.h && x[b.y] < x[a.y] + a.h;
          if (time && space) return false;
        }
    }
  }
  return true;
}

int64_t objective(const Problem &p, const int64_t *x) {
  int64_t total = p.offset;
  for (const bcir_csp_term &t : p.objective) total += t.coef * x[t.var];
  return total;
}

// The optimum by enumeration: false when there is no solution.
bool brute(const Problem &p, int64_t &best) {
  std::vector<int64_t> x;
  for (const bcir_csp_var &v : p.vars) x.push_back(v.lo);
  bool any = false;
  for (;;) {
    if (solution(p, x.data())) {
      int64_t o = objective(p, x.data());
      if (!any || o < best) best = o;
      any = true;
    }
    size_t k = 0;
    while (k < x.size() && x[k] == p.vars[k].hi) { x[k] = p.vars[k].lo; k++; }
    if (k == x.size()) return any;
    x[k]++;
  }
}

struct Answer { bcir_csp_status status; bcir_csp_result r; std::vector<int64_t> values; };

Answer run(const Problem &p, const bcir_host_allocator *a) {
  std::vector<bcir_csp_constraint> cons;
  for (const Con &c : p.cons)
    cons.push_back(bcir_csp_constraint{c.kind, c.terms.data(), c.terms.size(), c.relation, c.rhs, c.tasks.data(),
                                       c.tasks.size(), c.capacity, c.rects.data(), c.rects.size()});
  bcir_csp_model m{p.vars.data(), p.vars.size(), cons.data(), cons.size(), p.objective.data(), p.objective.size(),
                   p.offset};
  Answer ans{BCIR_CSP_OK, bcir_csp_result{}, std::vector<int64_t>(p.vars.size(), 0)};
  ans.r.values = ans.values.data(); ans.r.cap_values = ans.values.size();
  ans.status = bcir_csp_solve(&m, p.budget, p.seeded ? p.incumbent.data() : nullptr, a, &ans.r);
  return ans;
}

struct Tally { size_t inputs = 0, optimal = 0, infeasible = 0, budget = 0, refused = 0, nomem = 0; };
Tally tally;

int one_input(const uint8_t *data, size_t size) {
  Problem p = decode(data, size);
  Counting count; count.fail_at = p.fail_at;
  bcir_host_allocator a{&count, count_alloc, count_realloc, count_free};
  Answer o = run(p, &a);
  tally.inputs++;
  HOLD(count.live == 0);
  HOLD(o.status >= BCIR_CSP_OK && o.status <= BCIR_CSP_ERR_INTERNAL);
  HOLD(o.status != BCIR_CSP_ERR_INTERNAL && o.status != BCIR_CSP_ERR_CAPACITY && o.status != BCIR_CSP_ERR_RANGE);
  HOLD(o.status != BCIR_CSP_ERR_INVALID);   // the decoder builds well-formed models only
  if (o.status == BCIR_CSP_ERR_NOMEM) { HOLD(p.fail_at != 0); tally.nomem++; return 0; }
  bool seed_ok = !p.seeded || solution(p, p.incumbent.data());
  if (o.status == BCIR_CSP_ERR_INCUMBENT) { HOLD(!seed_ok); tally.refused++; return 0; }
  HOLD(o.status == BCIR_CSP_OK && seed_ok);
  int64_t best = 0;
  bool feasible = brute(p, best);
  if (o.r.has_values) {
    HOLD(solution(p, o.values.data()) && objective(p, o.values.data()) == o.r.objective);
    HOLD(feasible && o.r.objective >= best);
  }
  if (o.r.has_lower) HOLD(!o.r.has_values || o.r.lower_bound <= o.r.objective);
  switch (o.r.stop) {
  case BCIR_CSP_OPTIMAL:
    HOLD(o.r.has_values && o.r.objective == best && o.r.has_lower && o.r.lower_bound == best);
    tally.optimal++;
    break;
  case BCIR_CSP_INFEASIBLE:
    HOLD(!feasible && !o.r.has_values && !p.seeded);
    tally.infeasible++;
    break;
  case BCIR_CSP_BUDGET:
    HOLD(o.r.nodes == p.budget && o.r.has_lower && (!feasible || o.r.lower_bound <= best));
    tally.budget++;
    break;
  default:
    violated("an unknown stop");
  }
  if (o.r.stop != BCIR_CSP_BUDGET) HOLD(o.r.nodes <= p.budget);
  Answer again = run(p, nullptr);   // the default allocator, and the same answer
  HOLD(again.status == BCIR_CSP_OK && again.r.stop == o.r.stop && again.r.nodes == o.r.nodes);
  HOLD(again.r.has_values == o.r.has_values && (!o.r.has_values || again.values == o.values));
  HOLD(again.r.has_lower == o.r.has_lower && (!o.r.has_lower || again.r.lower_bound == o.r.lower_bound));
  return 0;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) { return one_input(data, size); }

#ifndef BCIR_LIBFUZZER
int main(int argc, char **argv) {
  if (argc > 1) {
    for (int k = 1; k < argc; k++) {
      std::FILE *f = std::fopen(argv[k], "rb");
      if (!f) { std::fprintf(stderr, "fuzz_csp: cannot read %s\n", argv[k]); return 2; }
      std::vector<uint8_t> bytes;
      int c;
      while ((c = std::fgetc(f)) != EOF) bytes.push_back(static_cast<uint8_t>(c));
      std::fclose(f);
      one_input(bytes.data(), bytes.size());
    }
  } else {
    uint64_t state = 0xD1B54A32D192ED03ull;   // a fixed stream: the same inputs on every host
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
  std::printf("fuzz_csp: %zu inputs, %zu optimal, %zu infeasible, %zu stopped on the budget, %zu incumbents refused, "
              "%zu refused an allocation\n", tally.inputs, tally.optimal, tally.infeasible, tally.budget, tally.refused,
              tally.nomem);
  if (argc == 1 && (!tally.optimal || !tally.infeasible || !tally.budget || !tally.refused || !tally.nomem)) {
    std::printf("fuzz_csp: the generated stream left an outcome unexercised\n");
    return 1;
  }
  return 0;
}
#endif
