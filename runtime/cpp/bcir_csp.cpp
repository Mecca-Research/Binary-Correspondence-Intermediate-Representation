//===- bcir_csp.cpp - the exact finite-domain solver (the C++ twin of bcir/kbcir/csp.py) ---===//
//
// PORT-6 (bcir_csp.h). `solve` step for step: the same propagators in the same order to the same fixpoint, the same
// cut, the same branching rule and stack discipline, so a run expands the oracle's nodes and returns its answer. The
// one change of representation is the cumulative's timetable, kept as sorted intervals where the oracle keeps a set
// of instants: the narrowing is the same, its cost no longer the domain's width. Failures are values, containers
// draw from one arena over the caller's allocator (bcir_solver.hpp), and the C entry point catches what the standard
// library may throw.
//===----------------------------------------------------------------------===//
#include "bcir_csp.h"
#include "bcir_solver.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

namespace {

using bcir::expected;
using bcir::unexpected;
using bcir::solver::host_resource;
using bcir::solver::phase_arena;
using bcir::solver::pvec;

constexpr int64_t kValue = int64_t{1} << 60;  // a bound, an offset, a duration, a height
constexpr int64_t kSum = int64_t{1} << 61;    // a row's coefficient-weighted bound sum, the heights of a constraint

int64_t floor_div(int64_t a, int64_t b) {
  int64_t q = a / b;
  if (a % b != 0 && ((a < 0) != (b < 0))) q--;
  return q;
}
int64_t ceil_div(int64_t a, int64_t b) { return -floor_div(-a, b); }  // the oracle's -((-a) // b)
// |v|, saturating: |INT64_MIN| is past every bound it is compared with (kValue, kSum) and every product checked, as the
// oracle's abs() puts it -- negating it in int64_t had been undefined, and let the value through
int64_t mag(int64_t v) { return v >= 0 ? v : v == INT64_MIN ? INT64_MAX : -v; }

struct Term { int64_t c; size_t v; };
struct Task { size_t s; int64_t d; };
struct Rect { size_t x0; int64_t a; size_t x1; int64_t b; size_t y; int64_t h; int64_t g; };
// One constraint: its kind and its range [begin, end) of the flat term, task or rectangle arrays.
struct Con { int kind; size_t begin, end; int op; int64_t rhs, capacity; };

struct Model {
  size_t n = 0;
  pvec<int64_t> lo, hi, prio;
  pvec<Term> terms;   // every linear row's terms, then the objective's
  pvec<Task> tasks;
  pvec<Rect> rects;
  pvec<Con> cons;
  size_t obj_begin = 0, obj_end = 0;
  int64_t offset = 0;
  explicit Model(std::pmr::memory_resource *r) : lo(r), hi(r), prio(r), terms(r), tasks(r), rects(r), cons(r) {}
};

// `Model._terms`: one term per variable, the coefficients summed, zeros dropped, in variable order.
expected<bool, bcir_csp_status> merge_terms(Model &m, const bcir_csp_term *in, size_t n, std::pmr::memory_resource *r) {
  if (n && !in) return unexpected(BCIR_CSP_ERR_INVALID);
  pvec<Term> raw(r);
  for (size_t k = 0; k < n; k++) {
    if (in[k].var >= m.n) return unexpected(BCIR_CSP_ERR_INVALID);
    raw.push_back(Term{in[k].coef, in[k].var});
  }
  std::stable_sort(raw.begin(), raw.end(), [](const Term &a, const Term &b) { return a.v < b.v; });
  for (size_t k = 0; k < raw.size();) {
    size_t v = raw[k].v;
    int64_t c = 0;
    for (; k < raw.size() && raw[k].v == v; k++)
      if (__builtin_add_overflow(c, raw[k].c, &c)) return unexpected(BCIR_CSP_ERR_RANGE);
    if (c) m.terms.push_back(Term{c, v});
  }
  return true;
}

// Whether sum(|c| * max(|lo|, |hi|)) + |extra| over [begin, end) stays within kSum.
bool row_in_range(const Model &m, size_t begin, size_t end, int64_t extra) {
  int64_t total = mag(extra), w = 0;
  if (total > kSum) return false;
  for (size_t k = begin; k < end; k++) {
    int64_t bound = std::max(mag(m.lo[m.terms[k].v]), mag(m.hi[m.terms[k].v]));
    if (__builtin_mul_overflow(mag(m.terms[k].c), bound, &w) || __builtin_add_overflow(total, w, &total) || total > kSum)
      return false;
  }
  return true;
}

expected<bool, bcir_csp_status> build(Model &m, const bcir_csp_model &in, std::pmr::memory_resource *r) {
  if ((in.n_vars && !in.vars) || (in.n_constraints && !in.constraints)) return unexpected(BCIR_CSP_ERR_INVALID);
  m.n = in.n_vars;
  for (size_t v = 0; v < m.n; v++) {
    const bcir_csp_var &x = in.vars[v];
    if (x.lo > x.hi) return unexpected(BCIR_CSP_ERR_INVALID);
    if (mag(x.lo) > kValue || mag(x.hi) > kValue) return unexpected(BCIR_CSP_ERR_RANGE);
    m.lo.push_back(x.lo); m.hi.push_back(x.hi); m.prio.push_back(x.priority);
  }
  for (size_t k = 0; k < in.n_constraints; k++) {
    const bcir_csp_constraint &c = in.constraints[k];
    Con con{c.kind, 0, 0, c.relation, c.rhs, c.capacity};
    if (c.kind == BCIR_CSP_LINEAR) {
      if (c.relation < BCIR_CSP_LE || c.relation > BCIR_CSP_EQ) return unexpected(BCIR_CSP_ERR_INVALID);
      con.begin = m.terms.size();
      auto merged = merge_terms(m, c.terms, c.n_terms, r);
      if (!merged) return unexpected(merged.error());
      con.end = m.terms.size();
      if (!row_in_range(m, con.begin, con.end, c.rhs)) return unexpected(BCIR_CSP_ERR_RANGE);
    } else if (c.kind == BCIR_CSP_CUMULATIVE) {
      if ((c.n_tasks && !c.tasks) || c.capacity < 1) return unexpected(BCIR_CSP_ERR_INVALID);
      con.begin = m.tasks.size();
      for (size_t t = 0; t < c.n_tasks; t++) {
        if (c.tasks[t].start >= m.n || c.tasks[t].duration < 0) return unexpected(BCIR_CSP_ERR_INVALID);
        if (c.tasks[t].duration > kValue) return unexpected(BCIR_CSP_ERR_RANGE);
        m.tasks.push_back(Task{c.tasks[t].start, c.tasks[t].duration});
      }
      con.end = m.tasks.size();
    } else if (c.kind == BCIR_CSP_NO_OVERLAP_2D) {
      if (c.n_rects && !c.rects) return unexpected(BCIR_CSP_ERR_INVALID);
      con.begin = m.rects.size();
      int64_t heights = 0;
      for (size_t t = 0; t < c.n_rects; t++) {
        const bcir_csp_rect &x = c.rects[t];
        if (x.x0 >= m.n || x.x1 >= m.n || x.y >= m.n || x.h < 1) return unexpected(BCIR_CSP_ERR_INVALID);
        if (mag(x.a) > kValue || mag(x.b) > kValue || x.h > kValue) return unexpected(BCIR_CSP_ERR_RANGE);
        heights += x.h;
        if (heights > kSum) return unexpected(BCIR_CSP_ERR_RANGE);
        for (size_t u = con.begin; u < m.rects.size(); u++)   // one group, one object: one y, one height
          if (m.rects[u].g == x.group && (m.rects[u].y != x.y || m.rects[u].h != x.h))
            return unexpected(BCIR_CSP_ERR_INVALID);
        m.rects.push_back(Rect{x.x0, x.a, x.x1, x.b, x.y, x.h, x.group});
      }
      con.end = m.rects.size();
    } else {
      return unexpected(BCIR_CSP_ERR_INVALID);
    }
    m.cons.push_back(con);
  }
  m.obj_begin = m.terms.size();
  auto merged = merge_terms(m, in.objective, in.n_objective, r);
  if (!merged) return unexpected(merged.error());
  m.obj_end = m.terms.size();
  m.offset = in.offset;
  if (!row_in_range(m, m.obj_begin, m.obj_end, m.offset)) return unexpected(BCIR_CSP_ERR_RANGE);
  return true;
}

int64_t row_value(const Model &m, size_t begin, size_t end, const int64_t *x) {
  int64_t total = 0;
  for (size_t k = begin; k < end; k++) total += m.terms[k].c * x[m.terms[k].v];
  return total;
}

// `satisfied`: the constraints' definitions alone, sharing no code with the propagators.
bool satisfied(const Model &m, const int64_t *x) {
  for (size_t v = 0; v < m.n; v++)
    if (x[v] < m.lo[v] || x[v] > m.hi[v]) return false;
  for (const Con &c : m.cons) {
    if (c.kind == BCIR_CSP_LINEAR) {
      int64_t total = row_value(m, c.begin, c.end, x);
      if ((c.op == BCIR_CSP_LE && total > c.rhs) || (c.op == BCIR_CSP_GE && total < c.rhs) ||
          (c.op == BCIR_CSP_EQ && total != c.rhs))
        return false;
    } else if (c.kind == BCIR_CSP_CUMULATIVE) {
      for (size_t i = c.begin; i < c.end; i++) {
        if (m.tasks[i].d <= 0) continue;
        int64_t t = x[m.tasks[i].s], running = 0;
        for (size_t j = c.begin; j < c.end; j++) {
          if (m.tasks[j].d <= 0) continue;
          int64_t a = x[m.tasks[j].s], b = a + m.tasks[j].d;
          if (a <= t && t < b) running++;
        }
        if (running > c.capacity) return false;
      }
    } else {
      for (size_t i = c.begin; i < c.end; i++)
        for (size_t j = i + 1; j < c.end; j++) {
          const Rect &p = m.rects[i], &q = m.rects[j];
          if (p.g == q.g) continue;
          int64_t p0 = x[p.x0] + p.a, p1 = x[p.x1] + p.b, q0 = x[p.y], q1 = x[p.y] + p.h;
          int64_t r0 = x[q.x0] + q.a, r1 = x[q.x1] + q.b, s0 = x[q.y], s1 = x[q.y] + q.h;
          if (p0 < p1 && r0 < r1 && p0 < r1 && r0 < p1 && q0 < s1 && s0 < q1) return false;
        }
    }
  }
  return true;
}

// --- propagation: each returns -1 when the node is infeasible, else whether a bound moved --------------------------

int linear_le(const Model &m, size_t begin, size_t end, int sign, int64_t rhs, pvec<int64_t> &lo, pvec<int64_t> &hi,
              pvec<int64_t> &mins) {
  mins.clear();
  int64_t total = 0;
  for (size_t k = begin; k < end; k++) {
    int64_t c = sign * m.terms[k].c;
    size_t v = m.terms[k].v;
    mins.push_back(c > 0 ? c * lo[v] : c * hi[v]);
    total += mins.back();
  }
  if (total > rhs) return -1;
  int moved = 0;
  for (size_t k = begin; k < end; k++) {
    int64_t c = sign * m.terms[k].c;
    size_t v = m.terms[k].v;
    int64_t slack = rhs - (total - mins[k - begin]);
    if (c > 0) {
      int64_t bound = floor_div(slack, c);
      if (bound < hi[v]) { hi[v] = bound; moved = 1; }
    } else {
      int64_t bound = ceil_div(slack, c);
      if (bound > lo[v]) { lo[v] = bound; moved = 1; }
    }
    if (lo[v] > hi[v]) return -1;
  }
  return moved;
}

int linear(const Model &m, size_t begin, size_t end, int op, int64_t rhs, pvec<int64_t> &lo, pvec<int64_t> &hi,
           pvec<int64_t> &mins) {
  int moved = 0;
  if (op == BCIR_CSP_LE || op == BCIR_CSP_EQ) {
    int r = linear_le(m, begin, end, 1, rhs, lo, hi, mins);
    if (r < 0) return -1;
    moved |= r;
  }
  if (op == BCIR_CSP_GE || op == BCIR_CSP_EQ) {
    int r = linear_le(m, begin, end, -1, -rhs, lo, hi, mins);
    if (r < 0) return -1;
    moved |= r;
  }
  return moved;
}

struct Part { size_t owner; int64_t a, b; };
struct Span { int64_t a, b; };

struct Scratch {
  pvec<int64_t> mins;
  pvec<Part> parts;
  pvec<std::pair<int64_t, int>> events;
  pvec<Span> full;
  struct Live { int64_t g; size_t y; int64_t h; };
  struct Certain { int64_t s, e; size_t y; int64_t h, g; };
  pvec<Certain> certain;
  pvec<Live> live;
  explicit Scratch(std::pmr::memory_resource *r)
      : mins(r), parts(r), events(r), full(r), certain(r), live(r) {}
};

// The instants the compulsory parts other than `skip`'s fill to `capacity`, as sorted disjoint spans (the oracle's
// `full`, a set of instants); with `skip` past every owner, whether any instant holds more than `capacity`.
bool sweep(Scratch &w, size_t skip, int64_t capacity, bool overload_only) {
  w.events.clear(); w.full.clear();
  for (const Part &p : w.parts)
    if (p.owner != skip) { w.events.emplace_back(p.a, 1); w.events.emplace_back(p.b, -1); }
  std::sort(w.events.begin(), w.events.end());
  int64_t count = 0;
  for (size_t k = 0; k < w.events.size();) {
    int64_t x = w.events[k].first;
    for (; k < w.events.size() && w.events[k].first == x; k++) count += w.events[k].second;
    if (overload_only) {
      if (count > capacity) return true;
      continue;
    }
    if (count >= capacity && k < w.events.size()) {
      int64_t next = w.events[k].first;
      if (!w.full.empty() && w.full.back().b == x) w.full.back().b = next;
      else w.full.push_back(Span{x, next});
    }
  }
  return false;
}

// `_cumulative`: the compulsory-part timetable. A task's [latest start, earliest end) is occupied in every solution,
// and no start may cover an instant the other tasks' compulsory parts already fill.
int cumulative(const Model &m, const Con &c, pvec<int64_t> &lo, pvec<int64_t> &hi, Scratch &w) {
  w.parts.clear();
  for (size_t i = c.begin; i < c.end; i++) {
    const Task &t = m.tasks[i];
    if (t.d > 0 && hi[t.s] < lo[t.s] + t.d) w.parts.push_back(Part{i, hi[t.s], lo[t.s] + t.d});
  }
  if (sweep(w, SIZE_MAX, c.capacity, true)) return -1;
  int moved = 0;
  for (size_t i = c.begin; i < c.end; i++) {
    const Task &task = m.tasks[i];
    if (task.d == 0) continue;
    sweep(w, i, c.capacity, false);
    if (w.full.empty()) continue;
    const int64_t d = task.d;
    const size_t s = task.s;
    // The earliest start >= lo whose [t, t + d) holds no full instant, and below, the latest <= hi: the oracle's walk
    // from full instant to full instant, by whole spans. A start in [t, f.b) of a span f meeting [t, t + d) covers an
    // instant of f -- its own, or f.a -- so no start before f.b is one, and the scan resumes there; going down, none
    // after f.a - d. Each pass passes a span, where stepping by d had made the cost the span's length over d.
    int64_t t = lo[s];
    while (t <= hi[s]) {   // the latest full span meeting [t, t + d), and past it
      bool hit = false;
      int64_t past = 0;
      for (const Span &f : w.full)
        if (f.a < t + d && f.b > t) { past = f.b; hit = true; }
      if (!hit) break;
      t = past;
    }
    if (t > hi[s]) return -1;
    if (t > lo[s]) { lo[s] = t; moved = 1; }
    t = hi[s];
    while (t >= lo[s]) {   // the earliest full span meeting [t, t + d), and a start that ends before it
      bool hit = false;
      int64_t first = 0;
      for (const Span &f : w.full)
        if (f.a < t + d && f.b > t) { first = f.a; hit = true; break; }
      if (!hit) break;
      t = first - d;
    }
    if (t < lo[s]) return -1;
    if (t < hi[s]) { hi[s] = t; moved = 1; }
  }
  return moved;
}

bool before(size_t v_end, int64_t end_off, size_t v_start, int64_t start_off, const pvec<int64_t> &lo,
            const pvec<int64_t> &hi) {
  return lo[v_end] + end_off <= hi[v_start] + start_off;
}

int enforce_before(size_t v_end, int64_t end_off, size_t v_start, int64_t start_off, pvec<int64_t> &lo,
                   pvec<int64_t> &hi) {
  int moved = 0;
  int64_t cap = hi[v_start] + start_off - end_off;
  if (cap < hi[v_end]) { hi[v_end] = cap; moved = 1; }
  int64_t floor = lo[v_end] + end_off - start_off;
  if (floor > lo[v_start]) { lo[v_start] = floor; moved = 1; }
  return moved;
}

// `_no_overlap_2d`: the energy of the objects certainly live at one instant, then the disjunction on each pair.
int no_overlap(const Model &m, const Con &c, pvec<int64_t> &lo, pvec<int64_t> &hi, Scratch &w) {
  w.certain.clear();
  for (size_t i = c.begin; i < c.end; i++) {
    const Rect &r = m.rects[i];
    int64_t start = hi[r.x0] + r.a, end = lo[r.x1] + r.b;
    if (start < end) w.certain.push_back(Scratch::Certain{start, end, r.y, r.h, r.g});
  }
  for (const Scratch::Certain &at : w.certain) {
    int64_t t = at.s;
    w.live.clear();
    for (const Scratch::Certain &o : w.certain)
      if (o.s <= t && t < o.e) w.live.push_back(Scratch::Live{o.g, o.y, o.h});
    std::stable_sort(w.live.begin(), w.live.end(), [](const Scratch::Live &a, const Scratch::Live &b) { return a.g < b.g; });
    size_t groups = 0;
    int64_t low = 0, high = 0, sum = 0;
    for (size_t k = 0; k < w.live.size(); k++) {
      if (k + 1 < w.live.size() && w.live[k + 1].g == w.live[k].g) continue;   // one entry per group: its last
      const Scratch::Live &l = w.live[k];
      low = groups ? std::min(low, lo[l.y]) : lo[l.y];
      high = groups ? std::max(high, hi[l.y] + l.h) : hi[l.y] + l.h;
      sum += l.h;
      groups++;
    }
    if (groups > 1 && sum > high - low) return -1;
  }
  int moved = 0;
  for (size_t i = c.begin; i < c.end; i++)
    for (size_t j = i + 1; j < c.end; j++) {
      const Rect &A = m.rects[i], &B = m.rects[j];
      if (A.g == B.g) continue;
      bool x_overlap = hi[A.x0] + A.a < lo[B.x1] + B.b && hi[B.x0] + B.a < lo[A.x1] + A.b &&
                       hi[A.x0] + A.a < lo[A.x1] + A.b && hi[B.x0] + B.a < lo[B.x1] + B.b;
      if (x_overlap) {   // then the addresses must be disjoint: a below b, or b below a
        bool below = lo[A.y] + A.h <= hi[B.y];
        bool above = lo[B.y] + B.h <= hi[A.y];
        if (!below && !above) return -1;
        if (below && !above) moved |= enforce_before(A.y, A.h, B.y, 0, lo, hi);
        else if (above && !below) moved |= enforce_before(B.y, B.h, A.y, 0, lo, hi);
      }
      bool y_overlap = hi[A.y] < lo[B.y] + B.h && hi[B.y] < lo[A.y] + A.h;
      if (y_overlap) {   // then the lifetimes must be disjoint (or one of them empty)
        bool a_first = before(A.x1, A.b, B.x0, B.a, lo, hi);
        bool b_first = before(B.x1, B.b, A.x0, A.a, lo, hi);
        bool a_empty = lo[A.x1] + A.b <= hi[A.x0] + A.a;
        bool b_empty = lo[B.x1] + B.b <= hi[B.x0] + B.a;
        if (!(a_first || b_first || a_empty || b_empty)) return -1;
        if (a_first && !(b_first || a_empty || b_empty)) moved |= enforce_before(A.x1, A.b, B.x0, B.a, lo, hi);
        else if (b_first && !(a_first || a_empty || b_empty)) moved |= enforce_before(B.x1, B.b, A.x0, A.a, lo, hi);
      }
      for (size_t v : {A.x0, A.x1, A.y, B.x0, B.x1, B.y})
        if (lo[v] > hi[v]) return -1;
    }
  return moved;
}

// `_propagate`: every constraint, then the objective's cut, to a fixpoint; false when the node is infeasible.
bool propagate(const Model &m, pvec<int64_t> &lo, pvec<int64_t> &hi, bool cut, int64_t cut_rhs, Scratch &w) {
  for (;;) {
    int moved = 0;
    for (const Con &c : m.cons) {
      int r = c.kind == BCIR_CSP_LINEAR       ? linear(m, c.begin, c.end, c.op, c.rhs, lo, hi, w.mins)
              : c.kind == BCIR_CSP_CUMULATIVE ? cumulative(m, c, lo, hi, w)
                                              : no_overlap(m, c, lo, hi, w);
      if (r < 0) return false;
      moved |= r;
    }
    if (cut) {
      int r = linear(m, m.obj_begin, m.obj_end, BCIR_CSP_LE, cut_rhs, lo, hi, w.mins);
      if (r < 0) return false;
      moved |= r;
    }
    if (!moved) return true;
  }
}

int64_t bound_of(const Model &m, const pvec<int64_t> &lo, const pvec<int64_t> &hi) {
  int64_t total = m.offset;
  for (size_t k = m.obj_begin; k < m.obj_end; k++) {
    const Term &t = m.terms[k];
    total += t.c > 0 ? t.c * lo[t.v] : t.c * hi[t.v];
  }
  return total;
}

struct Node {
  pvec<int64_t> lo, hi;
  explicit Node(std::pmr::memory_resource *r) : lo(r), hi(r) {}
};

bcir_csp_status solve(const bcir_csp_model &in, uint64_t budget, const int64_t *incumbent,
                      const bcir_host_allocator *allocator, bcir_csp_result &out) {
  host_resource host(allocator);
  phase_arena arena(&host);   // the solve's one release point
  Model m(&arena);
  auto built = build(m, in, &arena);
  if (!built) return built.error();
  if (out.cap_values < m.n || (m.n && !out.values)) return BCIR_CSP_ERR_CAPACITY;
  bool have = false;
  int64_t best_obj = 0;   // offset + the objective's terms at the best values
  pvec<int64_t> best(&arena);
  if (incumbent) {
    if (!satisfied(m, incumbent)) return BCIR_CSP_ERR_INCUMBENT;
    best.assign(incumbent, incumbent + m.n);
    best_obj = m.offset + row_value(m, m.obj_begin, m.obj_end, incumbent);
    have = true;
  }
  Scratch w(&arena);
  pvec<Node> stack(&arena);
  {
    Node root(&arena);
    root.lo.assign(m.lo.begin(), m.lo.end());
    root.hi.assign(m.hi.begin(), m.hi.end());
    stack.push_back(std::move(root));
  }
  uint64_t nodes = 0;
  while (!stack.empty()) {
    if (nodes >= budget) {
      int64_t floor = bound_of(m, stack[0].lo, stack[0].hi);
      for (const Node &n : stack) floor = std::min(floor, bound_of(m, n.lo, n.hi));
      if (have) floor = std::min(floor, best_obj);
      out.stop = BCIR_CSP_BUDGET; out.has_lower = 1; out.lower_bound = floor; out.nodes = nodes;
      out.has_values = have; out.objective = have ? best_obj : 0;
      if (have) std::copy(best.begin(), best.end(), out.values);
      return BCIR_CSP_OK;
    }
    Node node = std::move(stack.back());
    stack.pop_back();
    nodes++;
    if (!propagate(m, node.lo, node.hi, have, have ? best_obj - 1 - m.offset : 0, w)) continue;
    size_t pick = m.n;
    for (size_t v = 0; v < m.n; v++) {
      if (node.lo[v] >= node.hi[v]) continue;
      if (pick == m.n || m.prio[v] < m.prio[pick] ||
          (m.prio[v] == m.prio[pick] && node.hi[v] - node.lo[v] < node.hi[pick] - node.lo[pick]))
        pick = v;
    }
    if (pick == m.n) {   // fully fixed: a solution only if the definitions agree
      if (!satisfied(m, node.lo.data())) return BCIR_CSP_ERR_INTERNAL;
      int64_t obj = m.offset + row_value(m, m.obj_begin, m.obj_end, node.lo.data());
      if (!have || obj < best_obj) { best.assign(node.lo.begin(), node.lo.end()); best_obj = obj; have = true; }
      continue;
    }
    int64_t mid = floor_div(node.lo[pick] + node.hi[pick], 2);
    Node upper(&arena);
    upper.lo.assign(node.lo.begin(), node.lo.end());
    upper.hi.assign(node.hi.begin(), node.hi.end());
    upper.lo[pick] = mid + 1;
    node.hi[pick] = mid;
    stack.push_back(std::move(upper));
    stack.push_back(std::move(node));
  }
  out.nodes = nodes;
  if (!have) {
    out.stop = BCIR_CSP_INFEASIBLE; out.has_values = 0; out.has_lower = 0; out.objective = 0; out.lower_bound = 0;
    return BCIR_CSP_OK;
  }
  out.stop = BCIR_CSP_OPTIMAL; out.has_values = 1; out.objective = best_obj; out.has_lower = 1;
  out.lower_bound = best_obj;
  std::copy(best.begin(), best.end(), out.values);
  return BCIR_CSP_OK;
}

}  // namespace

extern "C" bcir_csp_status bcir_csp_solve(const bcir_csp_model *model, uint64_t budget, const int64_t *incumbent,
                                         const bcir_host_allocator *allocator, bcir_csp_result *result) {
  if (!model || !result) return BCIR_CSP_ERR_INVALID;
  if (allocator && !bcir_host_allocator_valid(allocator)) return BCIR_CSP_ERR_INVALID;
  try {
    return solve(*model, budget, incumbent, allocator, *result);
  } catch (const std::bad_alloc &) {
    return BCIR_CSP_ERR_NOMEM;
  } catch (...) {   // nothing else is thrown by the code above; a defect is reported, never propagated
    return BCIR_CSP_ERR_INTERNAL;
  }
}
