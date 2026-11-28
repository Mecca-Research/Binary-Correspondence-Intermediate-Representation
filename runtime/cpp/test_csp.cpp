//===- test_csp.cpp - the CSP solver's contracts, and its text driver -----------===//
//
// PORT-6 (bcir_csp.h). Two modes:
//
//   test_csp          the façade's contracts: a model solved to its optimum, infeasibility, the budget's stop and
//                     its bound, an incumbent seeding the search and one refused, every refusal (INVALID, RANGE,
//                     CAPACITY), term merging, determinism, and every allocation of a solve refused in turn (NOMEM,
//                     nothing left live). Prints `test_csp: OK`.
//   test_csp FILE     the problems in FILE solved one after another, each printed as bcir/tests/test_cpp_csp.py prints
//                     the oracle's `solve`:
//
//     in:   `problem`, then `var LO HI PRIORITY` per variable, `linear OP RHS TERMS` (OP le, ge or eq; TERMS
//           `coef:var,...` or `-`), `cumulative CAPACITY TASKS` (`start:duration,...` or `-`), `rects RECTS`
//           (`x0:a:x1:b:y:h:group;...` or `-`), `objective OFFSET TERMS`, optionally `incumbent v,v,...`, and
//           `budget N` to solve it
//     out:  `stop <optimal|infeasible|budget> values <v,v,...|-> objective <n|-> lower <n|-> nodes N`, or
//           `ERR <status>`. A malformed file prints `TOOL <reason>` and exits 2.
//===----------------------------------------------------------------------===//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "bcir_csp.h"

namespace {

int failures = 0;
#define CHECK(cond)                                                                                \
  do {                                                                                             \
    if (!(cond)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
  } while (0)

struct Con {
  int kind = BCIR_CSP_LINEAR;
  std::vector<bcir_csp_term> terms; int relation = BCIR_CSP_LE; int64_t rhs = 0;
  std::vector<bcir_csp_task> tasks; int64_t capacity = 1;
  std::vector<bcir_csp_rect> rects;
};
struct Problem {
  std::vector<bcir_csp_var> vars;
  std::vector<Con> cons;
  std::vector<bcir_csp_term> objective; int64_t offset = 0;
  bool has_incumbent = false; std::vector<int64_t> incumbent;
};

struct Answer { bcir_csp_status status; bcir_csp_result r; std::vector<int64_t> values; };

Answer solve(const Problem &p, uint64_t budget, const bcir_host_allocator *a = nullptr, size_t short_by = 0) {
  std::vector<bcir_csp_constraint> cons;
  for (const Con &c : p.cons)
    cons.push_back(bcir_csp_constraint{c.kind, c.terms.data(), c.terms.size(), c.relation, c.rhs, c.tasks.data(),
                                       c.tasks.size(), c.capacity, c.rects.data(), c.rects.size()});
  bcir_csp_model m{p.vars.data(), p.vars.size(), cons.data(), cons.size(), p.objective.data(), p.objective.size(),
                   p.offset};
  Answer ans{BCIR_CSP_OK, bcir_csp_result{}, std::vector<int64_t>(p.vars.size() + 1, -77)};
  ans.r.values = ans.values.data();
  ans.r.cap_values = p.vars.size() >= short_by ? p.vars.size() - short_by : 0;
  ans.status = bcir_csp_solve(&m, budget, p.has_incumbent ? p.incumbent.data() : nullptr, a, &ans.r);
  return ans;
}

// --- a counting allocator that refuses attempt `fail_at` (0: none) ---------------------------------------------------

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

// --- the models the contracts use ----------------------------------------------------------------------------------

// minimize 3x - 2y + 1 subject to x + y >= 4, x - y <= 1, over x in [0, 5], y in [0, 3]: x = 1, y = 3, objective -2.
Problem small() {
  Problem p;
  p.vars = {{0, 5, 0}, {0, 3, 0}};
  Con a; a.terms = {{1, 0}, {1, 1}}; a.relation = BCIR_CSP_GE; a.rhs = 4;
  Con b; b.terms = {{1, 0}, {-1, 1}}; b.relation = BCIR_CSP_LE; b.rhs = 1;
  p.cons = {a, b};
  p.objective = {{3, 0}, {-2, 1}}; p.offset = 1;
  return p;
}

// Three tasks of 2 on one machine, with three rectangles over the same starts: a schedule and a layout at once.
Problem scheduling() {
  Problem p;
  p.vars = {{0, 6, 0}, {0, 6, 0}, {0, 6, 0}, {0, 4, 1}, {0, 4, 1}, {0, 4, 1}, {0, 8, 2}};
  Con c; c.kind = BCIR_CSP_CUMULATIVE; c.tasks = {{0, 2}, {1, 2}, {2, 2}}; c.capacity = 1;
  Con r; r.kind = BCIR_CSP_NO_OVERLAP_2D;
  r.rects = {{0, 0, 0, 2, 3, 2, 0}, {1, 0, 1, 2, 4, 3, 1}, {2, 0, 2, 2, 5, 1, 2}};
  Con mk; mk.terms = {{1, 6}, {-1, 0}}; mk.relation = BCIR_CSP_GE; mk.rhs = 2;
  Con mk1; mk1.terms = {{1, 6}, {-1, 1}}; mk1.relation = BCIR_CSP_GE; mk1.rhs = 2;
  Con mk2; mk2.terms = {{1, 6}, {-1, 2}}; mk2.relation = BCIR_CSP_GE; mk2.rhs = 2;
  p.cons = {c, r, mk, mk1, mk2};
  p.objective = {{1, 6}};
  return p;
}

void test_a_model_solved_to_its_optimum() {
  Answer a = solve(small(), 100000);
  CHECK(a.status == BCIR_CSP_OK && a.r.stop == BCIR_CSP_OPTIMAL && a.r.has_values && a.r.has_lower);
  CHECK(a.r.objective == -2 && a.r.lower_bound == -2 && a.values[0] == 1 && a.values[1] == 3 && a.r.nodes > 0);
  CHECK(a.values[2] == -77);   // nothing written past the model
  Answer s = solve(scheduling(), 1000000);
  CHECK(s.status == BCIR_CSP_OK && s.r.stop == BCIR_CSP_OPTIMAL && s.r.objective == 6);
}

void test_infeasibility_and_the_budget() {
  Problem p = small();
  Con c; c.terms = {{1, 0}, {1, 1}}; c.relation = BCIR_CSP_LE; c.rhs = 3;   // with x + y >= 4: none
  p.cons.push_back(c);
  Answer a = solve(p, 100000);
  CHECK(a.status == BCIR_CSP_OK && a.r.stop == BCIR_CSP_INFEASIBLE && !a.r.has_values && !a.r.has_lower);
  Answer z = solve(scheduling(), 0);   // no node expanded: the root's bound, no assignment
  CHECK(z.status == BCIR_CSP_OK && z.r.stop == BCIR_CSP_BUDGET && z.r.nodes == 0 && !z.r.has_values && z.r.has_lower);
  CHECK(z.r.lower_bound == 0);
  Answer b = solve(scheduling(), 3);
  Answer full = solve(scheduling(), 1000000);
  CHECK(b.r.stop == BCIR_CSP_BUDGET && b.r.nodes == 3 && b.r.lower_bound <= full.r.objective);
}

void test_an_incumbent_seeds_the_search_and_a_wrong_one_is_refused() {
  Problem p = small();
  p.has_incumbent = true; p.incumbent = {2, 3};   // a solution, objective 1
  Answer a = solve(p, 100000);
  CHECK(a.status == BCIR_CSP_OK && a.r.stop == BCIR_CSP_OPTIMAL && a.r.objective == -2);
  Answer z = solve(p, 0);   // no node: the incumbent stands, the bound below it
  CHECK(z.status == BCIR_CSP_OK && z.r.stop == BCIR_CSP_BUDGET && z.r.has_values && z.r.objective == 1);
  CHECK(z.values[0] == 2 && z.values[1] == 3 && z.r.lower_bound <= 1);
  p.incumbent = {0, 0};   // x + y >= 4 broken
  CHECK(solve(p, 100).status == BCIR_CSP_ERR_INCUMBENT);
  p.incumbent = {9, 3};   // outside x's domain
  CHECK(solve(p, 100).status == BCIR_CSP_ERR_INCUMBENT);
}

void test_terms_are_merged_as_the_model_merges_them() {
  Problem p = small(), q = small();
  p.vars.push_back({0, 3, 0});   // z, in no row once merged
  q.vars.push_back({0, 3, 0});
  q.cons[0].terms = {{1, 1}, {2, 0}, {-1, 0}, {4, 2}, {-4, 2}};   // y + 2x - x + 4z - 4z: x + y, z's term dropped
  q.objective = {{-2, 1}, {1, 0}, {2, 0}, {5, 1}, {-5, 1}, {6, 2}, {-6, 2}};
  Answer a = solve(p, 100000), b = solve(q, 100000);
  CHECK(b.status == BCIR_CSP_OK && a.r.objective == b.r.objective && a.r.nodes == b.r.nodes);
  CHECK(a.values[0] == b.values[0] && a.values[1] == b.values[1] && a.values[2] == b.values[2]);
}

void test_every_refusal() {
  bcir_csp_result r{};
  CHECK(bcir_csp_solve(nullptr, 1, nullptr, nullptr, &r) == BCIR_CSP_ERR_INVALID);
  struct Case { const char *what; Problem p; bcir_csp_status want; };
  std::vector<Case> cases;
  auto with = [&](const char *what, bcir_csp_status want, auto edit) { Problem p = small(); edit(p); cases.push_back({what, p, want}); };
  with("an empty domain", BCIR_CSP_ERR_INVALID, [](Problem &p) { p.vars[0] = {3, 2, 0}; });
  with("a term of no variable", BCIR_CSP_ERR_INVALID, [](Problem &p) { p.cons[0].terms[0].var = 7; });
  with("an unknown relation", BCIR_CSP_ERR_INVALID, [](Problem &p) { p.cons[0].relation = 3; });
  with("an unknown kind", BCIR_CSP_ERR_INVALID, [](Problem &p) { p.cons[0].kind = 9; });
  with("a negative duration", BCIR_CSP_ERR_INVALID, [](Problem &p) {
    Con c; c.kind = BCIR_CSP_CUMULATIVE; c.tasks = {{0, -1}}; c.capacity = 1; p.cons.push_back(c); });
  with("a capacity of 0", BCIR_CSP_ERR_INVALID, [](Problem &p) {
    Con c; c.kind = BCIR_CSP_CUMULATIVE; c.tasks = {{0, 1}}; c.capacity = 0; p.cons.push_back(c); });
  with("a task of no variable", BCIR_CSP_ERR_INVALID, [](Problem &p) {
    Con c; c.kind = BCIR_CSP_CUMULATIVE; c.tasks = {{5, 1}}; c.capacity = 1; p.cons.push_back(c); });
  with("a height of 0", BCIR_CSP_ERR_INVALID, [](Problem &p) {
    Con c; c.kind = BCIR_CSP_NO_OVERLAP_2D; c.rects = {{0, 0, 0, 1, 1, 0, 0}}; p.cons.push_back(c); });
  with("a group of two shapes", BCIR_CSP_ERR_INVALID, [](Problem &p) {
    Con c; c.kind = BCIR_CSP_NO_OVERLAP_2D; c.rects = {{0, 0, 0, 1, 1, 1, 4}, {0, 1, 0, 2, 1, 2, 4}}; p.cons.push_back(c); });
  with("a bound past 2^60", BCIR_CSP_ERR_RANGE, [](Problem &p) {   // a variable in no row: refused for its bound alone
    p.vars.push_back({0, (int64_t{1} << 60) + 1, 0}); });
  with("a row past 2^61", BCIR_CSP_ERR_RANGE, [](Problem &p) { p.cons[0].terms[0].coef = int64_t{1} << 59; });
  with("an objective past 2^61", BCIR_CSP_ERR_RANGE, [](Problem &p) { p.offset = (int64_t{1} << 61) + 1; });
  with("coefficients that sum past 64 bits", BCIR_CSP_ERR_RANGE, [](Problem &p) {
    p.cons[0].terms = {{INT64_MAX, 0}, {INT64_MAX, 0}}; });
  // INT64_MIN as a bound, a coefficient and an offset: past every bound, never negated (it had been undefined, and a
  // fixed variable at it had been solved)
  with("a bound of INT64_MIN", BCIR_CSP_ERR_RANGE, [](Problem &p) { p.vars.push_back({INT64_MIN, INT64_MIN, 0}); });
  with("a coefficient of INT64_MIN", BCIR_CSP_ERR_RANGE, [](Problem &p) { p.cons[0].terms[0].coef = INT64_MIN; });
  with("an offset of INT64_MIN", BCIR_CSP_ERR_RANGE, [](Problem &p) { p.offset = INT64_MIN; });
  with("a rectangle offset of INT64_MIN", BCIR_CSP_ERR_RANGE, [](Problem &p) {
    Con c; c.kind = BCIR_CSP_NO_OVERLAP_2D; c.rects = {{0, INT64_MIN, 0, 1, 1, 1, 0}}; p.cons.push_back(c); });
  with("a duration past 2^60", BCIR_CSP_ERR_RANGE, [](Problem &p) {
    Con c; c.kind = BCIR_CSP_CUMULATIVE; c.tasks = {{0, (int64_t{1} << 60) + 1}}; p.cons.push_back(c); });
  for (const Case &c : cases) {
    Answer a = solve(c.p, 100);
    if (a.status != c.want) { std::fprintf(stderr, "%s: status %d, not %d\n", c.what, a.status, c.want); failures++; }
  }
  Answer small_buffer = solve(small(), 100, nullptr, 1);
  CHECK(small_buffer.status == BCIR_CSP_ERR_CAPACITY && small_buffer.values[0] == -77);
  bcir_host_allocator incomplete{nullptr, count_alloc, nullptr, nullptr};
  CHECK(solve(small(), 100, &incomplete).status == BCIR_CSP_ERR_INVALID);
}

void test_the_search_is_deterministic() {
  for (uint64_t budget : {2ull, 40ull, 1000000ull}) {
    Answer a = solve(scheduling(), budget), b = solve(scheduling(), budget);
    CHECK(a.status == b.status && a.r.stop == b.r.stop && a.r.nodes == b.r.nodes && a.r.objective == b.r.objective);
    CHECK(a.r.lower_bound == b.r.lower_bound && a.values == b.values);
  }
}

void test_every_allocation_failed_in_turn() {
  Counting probe;
  bcir_host_allocator a{&probe, count_alloc, count_realloc, count_free};
  Answer whole = solve(scheduling(), 1000000, &a);
  CHECK(whole.status == BCIR_CSP_OK && probe.attempts > 0 && probe.live == 0);
  for (size_t k = 1; k <= probe.attempts; k++) {
    Counting c; c.fail_at = k;
    bcir_host_allocator fa{&c, count_alloc, count_realloc, count_free};
    Answer r = solve(scheduling(), 1000000, &fa);
    if (r.status != BCIR_CSP_ERR_NOMEM || c.live != 0) {
      std::fprintf(stderr, "allocation %zu of %zu refused: status %d, %zu live\n", k, probe.attempts, r.status, c.live);
      failures++;
    }
  }
}

// --- the text driver -------------------------------------------------------------------------------------------------

bool split(const std::string &s, char sep, std::vector<std::string> &out) {
  out.clear();
  if (s == "-") return true;
  std::stringstream ss(s); std::string item;
  while (std::getline(ss, item, sep)) { if (item.empty()) return false; out.push_back(item); }
  return true;
}
bool number(const std::string &s, int64_t &v) {
  if (s.empty() || s.size() > 20 || s.find_first_not_of("-0123456789") != std::string::npos) return false;
  v = std::stoll(s);
  return true;
}
bool terms_of(const std::string &s, std::vector<bcir_csp_term> &out) {
  std::vector<std::string> items, parts;
  if (!split(s, ',', items)) return false;
  for (const std::string &it : items) {
    int64_t c = 0, v = 0;
    if (!split(it, ':', parts) || parts.size() != 2 || !number(parts[0], c) || !number(parts[1], v) || v < 0) return false;
    out.push_back(bcir_csp_term{c, static_cast<uint32_t>(v)});
  }
  return true;
}

void print(const Problem &p, uint64_t budget) {
  Answer a = solve(p, budget);
  if (a.status != BCIR_CSP_OK) { std::printf("ERR %d\n", static_cast<int>(a.status)); return; }
  static const char *const stops[] = {"optimal", "infeasible", "budget"};
  std::printf("stop %s values ", stops[a.r.stop]);
  if (!a.r.has_values) std::printf("-");
  for (size_t k = 0; a.r.has_values && k < p.vars.size(); k++)
    std::printf("%s%lld", k ? "," : "", static_cast<long long>(a.values[k]));
  if (a.r.has_values) std::printf(" objective %lld", static_cast<long long>(a.r.objective));
  else std::printf(" objective -");
  if (a.r.has_lower) std::printf(" lower %lld", static_cast<long long>(a.r.lower_bound));
  else std::printf(" lower -");
  std::printf(" nodes %llu\n", static_cast<unsigned long long>(a.r.nodes));
}

int tool_fail(const char *why) { std::printf("TOOL %s\n", why); return 2; }

int drive(const char *path) {
  std::ifstream in(path);
  if (!in) return tool_fail("cannot read the file");
  Problem p; std::string line;
  try {
    while (std::getline(in, line)) {
      std::istringstream ls(line); std::string kw;
      if (!(ls >> kw)) continue;
      std::string a, b, c;
      if (kw == "problem") { p = Problem{}; continue; }
      if (kw == "var") {
        bcir_csp_var v{}; ls >> v.lo >> v.hi >> v.priority; p.vars.push_back(v);
      } else if (kw == "linear") {
        Con con; ls >> a >> con.rhs >> b;
        con.relation = a == "le" ? BCIR_CSP_LE : a == "ge" ? BCIR_CSP_GE : a == "eq" ? BCIR_CSP_EQ : -1;
        if (con.relation < 0 || !terms_of(b, con.terms)) return tool_fail("malformed linear");
        p.cons.push_back(con);
      } else if (kw == "cumulative") {
        Con con; con.kind = BCIR_CSP_CUMULATIVE; ls >> con.capacity >> a;
        std::vector<std::string> items, parts;
        if (!split(a, ',', items)) return tool_fail("malformed tasks");
        for (const std::string &it : items) {
          int64_t s = 0, d = 0;
          if (!split(it, ':', parts) || parts.size() != 2 || !number(parts[0], s) || !number(parts[1], d) || s < 0)
            return tool_fail("malformed task");
          con.tasks.push_back(bcir_csp_task{static_cast<uint32_t>(s), d});
        }
        p.cons.push_back(con);
      } else if (kw == "rects") {
        Con con; con.kind = BCIR_CSP_NO_OVERLAP_2D; ls >> a;
        std::vector<std::string> items, parts;
        if (!split(a, ';', items)) return tool_fail("malformed rects");
        for (const std::string &it : items) {
          int64_t f[7];
          if (!split(it, ':', parts) || parts.size() != 7) return tool_fail("malformed rect");
          for (int k = 0; k < 7; k++) if (!number(parts[k], f[k])) return tool_fail("malformed rect");
          if (f[0] < 0 || f[2] < 0 || f[4] < 0) return tool_fail("malformed rect");
          con.rects.push_back(bcir_csp_rect{static_cast<uint32_t>(f[0]), f[1], static_cast<uint32_t>(f[2]), f[3],
                                            static_cast<uint32_t>(f[4]), f[5], f[6]});
        }
        p.cons.push_back(con);
      } else if (kw == "objective") {
        ls >> p.offset >> a;
        p.objective.clear();
        if (!terms_of(a, p.objective)) return tool_fail("malformed objective");
      } else if (kw == "incumbent") {
        ls >> a;
        std::vector<std::string> items;
        if (!split(a, ',', items)) return tool_fail("malformed incumbent");
        p.has_incumbent = true; p.incumbent.clear();
        for (const std::string &it : items) { int64_t v = 0; if (!number(it, v)) return tool_fail("malformed incumbent"); p.incumbent.push_back(v); }
        if (p.incumbent.size() != p.vars.size()) return tool_fail("an incumbent of another length");
      } else if (kw == "budget") {
        uint64_t budget = 0; ls >> budget;
        if (ls.fail()) return tool_fail("malformed budget");
        print(p, budget);
        continue;
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
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 2) return drive(argv[1]);
  if (argc != 1) { std::fprintf(stderr, "usage: test_csp [FILE]\n"); return 2; }
  test_a_model_solved_to_its_optimum();
  test_infeasibility_and_the_budget();
  test_an_incumbent_seeds_the_search_and_a_wrong_one_is_refused();
  test_terms_are_merged_as_the_model_merges_them();
  test_every_refusal();
  test_the_search_is_deterministic();
  test_every_allocation_failed_in_turn();
  if (failures) { std::printf("test_csp: %d FAILED\n", failures); return 1; }
  std::printf("test_csp: OK\n");
  return 0;
}
