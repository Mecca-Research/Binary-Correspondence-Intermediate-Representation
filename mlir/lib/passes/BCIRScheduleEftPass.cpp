//===- BCIRScheduleEftPass.cpp - duration-aware EFT + pipelined token scheduling -*- C++ -*-===//
//
// Two duration-aware schedulers sharing the same earliest-finish-time placement (the law-rail
// port of gem.schedule), built on per-claim durations from the K_BCIR plan:
//
//   -bcir-schedule-eft : gem.schedule.schedule_eft -- phase-barriered HEFT-lite waves. Per phase
//                        (topo order): LPT priority, EFT domain placement, locality tie-breaks,
//                        the bandwidth-knee clamp; the GGG/random tail runs decoupled; phases
//                        compose serially. Annotates kbcir.sched_domain/start/finish + makespan/knee.
//
//   -bcir-async        : gem.async_tokens.async_plan + schedule.execute_tokens -- the !bcir.token
//                        fork/await DAG drives a SINGLE cross-phase dispatch (no phase barriers),
//                        so an independent claim of a later phase overlaps an earlier one --
//                        software pipelining falls out of the dependency structure. Annotates
//                        kbcir.async_awaits (the awaited claim ids) + async_domain/start/finish +
//                        async_makespan. The phase-barriered schedule is its degenerate case.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRCostModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <string>

using namespace mlir;

namespace bcir {
namespace {

static constexpr int kTailStream = -1; // the decoupled GGG/random tail's own stream

static bool isSparse(ClaimOp c) {
  return c.getLane() == Lane::GGG || c.getStrideClass() == StrideClass::Random;
}

// gem.concurrency._conflict: a RAW/WAR/WAW between two claims' read/write symbol sets.
static bool conflict(ArrayRef<StringRef> ar, ArrayRef<StringRef> aw, ArrayRef<StringRef> br,
                     ArrayRef<StringRef> bw) {
  auto hits = [](ArrayRef<StringRef> x, ArrayRef<StringRef> y) {
    for (StringRef a : x)
      for (StringRef b : y)
        if (a == b)
          return true;
    return false;
  };
  return hits(aw, br) || hits(aw, bw) || hits(bw, ar);
}

struct Info {
  ClaimOp claim;
  int32_t phase;
  int64_t id;
  int64_t dur;
  bool bandwidth; // cost_class == bandwidth (the default) -> knee-clamped
  SmallVector<StringRef> reads, writes;
  llvm::DenseSet<StringRef> rids; // _rids = set(rd) | set(wr) -- deduped, for locality
};

// Per-claim durations from the K_BCIR plan: the chosen coupled edge cost (durations_from).
static SmallVector<Info> buildInfos(const std::vector<cm::Column> &cols, ArrayRef<int> chosen,
                                    int64_t theta, ArrayRef<int64_t> w) {
  SmallVector<Info> infos;
  for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
    cm::Cost e = cols[i].cands[chosen[i]].cost;
    cm::Factor f = (i > 0)
        ? cm::contextFactor(theta, cols[i - 1].reads, cols[i - 1].cands[chosen[i - 1]].width,
                            cols[i].reads, cols[i].cands[chosen[i]].width)
        : cm::contextFactor(theta, {}, 0, cols[i].reads, cols[i].cands[chosen[i]].width);
    cm::applyFactor(e, f);
    Info in;
    in.claim = cols[i].claim;
    in.phase = cols[i].phase;
    in.id = static_cast<int64_t>(in.claim.getClaimId());
    in.dur = std::max<int64_t>(1, cm::scalarize(e, w));
    in.bandwidth = !in.claim.getCostClass() ||
                   *in.claim.getCostClass() == CostClass::Bandwidth;
    in.reads.assign(cols[i].reads.begin(), cols[i].reads.end());
    in.writes.assign(cols[i].writes.begin(), cols[i].writes.end());
    for (StringRef r : in.reads)
      in.rids.insert(r);
    for (StringRef r : in.writes)
      in.rids.insert(r);
    infos.push_back(std::move(in));
  }
  return infos;
}

// gem.schedule._dispatch: event-driven LPT list scheduling onto the affinity domains, placing
// each claim at the earliest finish time (locality + knee), annotating kbcir.<prefix>_*.
static void eftDispatch(ArrayRef<Info *> claims,
                        const llvm::DenseMap<int64_t, SmallVector<int64_t>> &preds, int64_t t0,
                        int64_t domains, int64_t knee, SmallVector<int64_t> &domainFree,
                        SmallVector<llvm::DenseSet<StringRef>> &resident,
                        llvm::DenseMap<int64_t, int64_t> &finishOf, Builder &b,
                        StringRef prefix) {
  std::string p = prefix.str();
  std::string an = "kbcir." + p + "_domain";
  std::string sn = "kbcir." + p + "_start";
  std::string fn = "kbcir." + p + "_finish";
  SmallVector<Info *> pending(claims.begin(), claims.end());
  while (!pending.empty()) {
    // Ready = every predecessor has finished. Pick the longest duration first (ties by id).
    Info *pick = nullptr;
    unsigned pickIdx = 0;
    for (unsigned k = 0; k < pending.size(); ++k) {
      Info *c = pending[k];
      bool ready = true;
      for (int64_t p : preds.lookup(c->id))
        if (!finishOf.count(p)) {
          ready = false;
          break;
        }
      if (!ready)
        continue;
      if (!pick || c->dur > pick->dur || (c->dur == pick->dur && c->id < pick->id)) {
        pick = c;
        pickIdx = k;
      }
    }
    if (!pick)
      break; // a cycle (shouldn't happen on a legal DAG)
    pending.erase(pending.begin() + pickIdx);

    int64_t readyT = t0;
    for (int64_t p : preds.lookup(pick->id))
      readyT = std::max(readyT, finishOf.lookup(p));
    int64_t width = pick->bandwidth ? knee : domains;

    int bestD = 0;
    int64_t bestFinish = std::numeric_limits<int64_t>::max(), bestScore = -1;
    for (int64_t d = 0; d < width; ++d) {
      int64_t start = std::max(domainFree[d], readyT);
      int64_t finish = start + pick->dur;
      int64_t score = 0;
      for (StringRef r : pick->rids)
        if (resident[d].count(r))
          ++score;
      // key = (finish, -score, d) ascending.
      if (finish < bestFinish || (finish == bestFinish && score > bestScore)) {
        bestFinish = finish;
        bestScore = score;
        bestD = static_cast<int>(d);
      }
    }
    int64_t start = std::max(domainFree[bestD], readyT);
    int64_t finish = start + pick->dur;
    domainFree[bestD] = finish;
    for (StringRef r : pick->rids)
      resident[bestD].insert(r);
    finishOf[pick->id] = finish;
    pick->claim->setAttr(an, b.getI64IntegerAttr(bestD));
    pick->claim->setAttr(sn, b.getI64IntegerAttr(start));
    pick->claim->setAttr(fn, b.getI64IntegerAttr(finish));
  }
}

// gem.concurrency._topo_phase_ids: phases in dependency order (deps first).
static SmallVector<int32_t> topoPhases(Operation *root) {
  SmallVector<PhaseOp> phases;
  llvm::DenseMap<int32_t, PhaseOp> byId;
  llvm::DenseMap<StringRef, int32_t> idOf;
  root->walk([&](PhaseOp p) {
    phases.push_back(p);
    byId[p.getId()] = p;
    idOf[p.getSymName()] = p.getId();
  });
  SmallVector<int32_t> order;
  llvm::DenseMap<int32_t, int> color;
  std::function<void(int32_t)> visit = [&](int32_t pid) {
    color[pid] = 1;
    auto it = byId.find(pid);
    if (it != byId.end())
      for (Attribute a : it->second.getDeps())
        if (auto ref = dyn_cast<FlatSymbolRefAttr>(a)) {
          auto d = idOf.find(ref.getValue());
          if (d != idOf.end() && color.lookup(d->second) == 0)
            visit(d->second);
        }
    order.push_back(pid);
  };
  for (PhaseOp p : phases)
    if (color.lookup(p.getId()) == 0)
      visit(p.getId());
  return order;
}

// Shared front end: plan the module and build the per-claim Info list. Returns false if the
// module is not plannable (no capability / weights / claims).
static bool planInfos(Operation *root, SmallVector<Info> &infos, int64_t &domains,
                      int64_t &knee, int64_t &theta) {
  auto capOp = cm::firstCapability(root);
  if (!capOp)
    return false;
  cm::Cap h = cm::readCap(capOp);
  ArrayRef<int64_t> w = cm::firstWeights(root);
  if (w.empty())
    return false;
  auto resByName = cm::resourcesByName(root);
  std::vector<cm::Column> cols = cm::fusedColumns(root, h, resByName);
  if (cols.empty())
    return false;
  theta = cm::firstThetaThermal(root);
  int64_t total = 0;
  SmallVector<int> chosen = cm::planChosen(cols, w, theta, total);
  if (chosen.empty())
    return false;
  domains = std::max<int64_t>(1, capOp.getAffinityDomains());
  knee = std::max<int64_t>(1, std::min<int64_t>(domains, capOp.getMemChannels()));
  infos = buildInfos(cols, chosen, theta, w);
  return true;
}

struct ScheduleEftPass : public PassWrapper<ScheduleEftPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleEftPass)
  StringRef getArgument() const final { return "bcir-schedule-eft"; }
  StringRef getDescription() const final {
    return "Duration-aware EFT wave scheduling (gem.schedule.schedule_eft): LPT priority + "
           "earliest-finish placement + locality + the bandwidth-knee clamp; annotates "
           "kbcir.sched_domain / sched_start / sched_finish / sched_makespan / sched_knee.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b);
    });
  }

  void runOnModule(Operation *root, Builder &b) {
    SmallVector<Info> infos;
    int64_t domains = 1, knee = 1, theta = 0;
    if (!planInfos(root, infos, domains, knee, theta))
      return;

    SmallVector<llvm::DenseSet<StringRef>> resident(domains); // locality memory, persists
    int64_t t0 = 0, makespan = 0;
    for (int32_t pid : topoPhases(root)) {
      SmallVector<Info *> phaseClaims;
      for (Info &in : infos)
        if (in.phase == pid)
          phaseClaims.push_back(&in);
      std::sort(phaseClaims.begin(), phaseClaims.end(),
                [](Info *a, Info *z) { return a->id < z->id; });
      SmallVector<Info *> main, tail;
      for (Info *in : phaseClaims)
        (isSparse(in->claim) ? tail : main).push_back(in);

      // Intra-phase hazard DAG (lower claim id is the producer of a conflicting pair).
      llvm::DenseMap<int64_t, SmallVector<int64_t>> preds;
      for (unsigned i = 0; i < main.size(); ++i)
        for (unsigned j = 0; j < i; ++j)
          if (conflict(main[j]->reads, main[j]->writes, main[i]->reads, main[i]->writes))
            preds[main[i]->id].push_back(main[j]->id);

      SmallVector<int64_t> domainFree(domains, t0);
      llvm::DenseMap<int64_t, int64_t> finishOf;
      eftDispatch(main, preds, t0, domains, knee, domainFree, resident, finishOf, b, "sched");

      int64_t tt = t0;
      for (Info *in : tail) { // the decoupled tail: a serial chain on its own stream
        in->claim->setAttr("kbcir.sched_domain", b.getI64IntegerAttr(kTailStream));
        in->claim->setAttr("kbcir.sched_start", b.getI64IntegerAttr(tt));
        in->claim->setAttr("kbcir.sched_finish", b.getI64IntegerAttr(tt + in->dur));
        tt += in->dur;
      }
      int64_t next = std::max(t0, tt);
      for (auto &kv : finishOf)
        next = std::max(next, kv.second);
      t0 = next;
      makespan = next;
    }
    root->setAttr("kbcir.sched_makespan", b.getI64IntegerAttr(makespan));
    root->setAttr("kbcir.sched_knee", b.getI64IntegerAttr(knee));
  }
};

struct AsyncPass : public PassWrapper<AsyncPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AsyncPass)
  StringRef getArgument() const final { return "bcir-async"; }
  StringRef getDescription() const final {
    return "Async token plan + pipelined schedule (gem.async_tokens.async_plan + "
           "schedule.execute_tokens): the fork/await DAG drives a single cross-phase EFT "
           "dispatch; annotates kbcir.async_awaits / async_domain / async_start / async_finish "
           "/ async_makespan (later-phase claims overlap earlier ones -- software pipelining).";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b);
    });
  }

  void runOnModule(Operation *root, Builder &b) {
    SmallVector<Info> infos;
    int64_t domains = 1, knee = 1, theta = 0;
    if (!planInfos(root, infos, domains, knee, theta))
      return;

    // The fork order: claims in topo-phase order, then by id (gem.async_tokens.async_plan).
    SmallVector<Info *> flat;
    for (int32_t pid : topoPhases(root)) {
      SmallVector<Info *> ph;
      for (Info &in : infos)
        if (in.phase == pid)
          ph.push_back(&in);
      std::sort(ph.begin(), ph.end(), [](Info *a, Info *z) { return a->id < z->id; });
      flat.append(ph.begin(), ph.end());
    }

    // await DAG: each claim awaits the EARLIER (in fork order) claims it conflicts with --
    // the full cross-phase RAW/WAR/WAW dependency (no phase barriers).
    llvm::DenseMap<int64_t, SmallVector<int64_t>> awaits;
    for (unsigned i = 0; i < flat.size(); ++i) {
      SmallVector<int64_t> &aw = awaits[flat[i]->id];
      for (unsigned j = 0; j < i; ++j)
        if (conflict(flat[j]->reads, flat[j]->writes, flat[i]->reads, flat[i]->writes))
          aw.push_back(flat[j]->id);
      flat[i]->claim->setAttr("kbcir.async_awaits", b.getI64ArrayAttr(aw));
    }

    // A single dispatch over every claim, preds = the awaits -> independent later-phase claims
    // start at 0 and overlap earlier phases (schedule.execute_tokens).
    SmallVector<int64_t> domainFree(domains, 0);
    SmallVector<llvm::DenseSet<StringRef>> resident(domains);
    llvm::DenseMap<int64_t, int64_t> finishOf;
    eftDispatch(flat, awaits, 0, domains, knee, domainFree, resident, finishOf, b, "async");
    int64_t makespan = 0;
    for (auto &kv : finishOf)
      makespan = std::max(makespan, kv.second);
    root->setAttr("kbcir.async_makespan", b.getI64IntegerAttr(makespan));
  }
};

} // namespace

std::unique_ptr<Pass> createScheduleEftPass() { return std::make_unique<ScheduleEftPass>(); }
std::unique_ptr<Pass> createAsyncPass() { return std::make_unique<AsyncPass>(); }

} // namespace bcir
