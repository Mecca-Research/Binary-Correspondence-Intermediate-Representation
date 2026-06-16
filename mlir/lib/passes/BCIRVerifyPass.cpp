//===- BCIRVerifyPass.cpp - the -bcir-verify semantic laws R1-R17 -*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRPassSupport.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <array>
#include <functional>
#include <optional>

using namespace mlir;

namespace bcir {
namespace {
struct VerifyPass : public PassWrapper<VerifyPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyPass)

  StringRef getArgument() const final { return "bcir-verify"; }
  StringRef getDescription() const final {
    return "Verify the BCIR semantic laws R1-R17: registry, domain, phase DAG, "
           "hazard, lane, bounds, cost, plan, provenance, generation, lowering, "
           "policy provenance, CIM/PIM dispatch, DVFS clock, allocator placement, "
           "accuracy contract.";
  }

  void runOnOperation() override {
    Operation *root = getOperation();
    bool ok = true;

    // R1: registry uniqueness -- every RID unique.
    llvm::DenseMap<uint32_t, ResourceOp> rids;
    llvm::DenseMap<StringRef, ResourceOp> resourceByName;
    root->walk([&](ResourceOp r) {
      resourceByName[r.getSymName()] = r;
      uint32_t rid = r.getRid();
      if (rids.count(rid)) {
        r.emitError("R1: duplicate RID ") << rid;
        ok = false;
      } else {
        rids[rid] = r;
      }
    });

    // R2: registry resolution -- claim reads/writes resolve to declared resources.
    auto checkRefs = [&](ClaimOp c, ArrayAttr refs, StringRef which) {
      for (Attribute a : refs) {
        auto ref = dyn_cast<FlatSymbolRefAttr>(a);
        if (ref && !resourceByName.count(ref.getValue())) {
          c.emitError("R2: claim ")
              << c.getSymName() << " " << which << " undeclared resource @"
              << ref.getValue();
          ok = false;
        }
      }
    };
    root->walk([&](ClaimOp c) {
      checkRefs(c, c.getReads(), "reads");
      checkRefs(c, c.getWrites(), "writes");
    });

    // R3: domain legality -- claim domain contracts correspond to registry
    // placement; MMIO writes need an ordered hazard; HAM is illegal on MMIO.
    root->walk([&](ResourceOp r) {
      if (r.getAccess() && *r.getAccess() == Access::HAM &&
          r.getDomainKind() == Domain::MMIO) {
        r.emitError("R3: resource ")
            << r.getSymName() << " HAM access is illegal in the MMIO domain";
        ok = false;
      }
    });
    root->walk([&](ClaimOp c) {
      bool anyResolved = false, domainBacked = false;
      auto touch = [&](ArrayAttr refs) {
        for (Attribute a : refs) {
          auto ref = dyn_cast<FlatSymbolRefAttr>(a);
          if (!ref)
            continue;
          auto it = resourceByName.find(ref.getValue());
          if (it == resourceByName.end())
            continue;
          anyResolved = true;
          if (it->second.getDomainKind() == c.getDomain())
            domainBacked = true;
        }
      };
      touch(c.getReads());
      touch(c.getWrites());
      if (anyResolved && !domainBacked) {
        c.emitError("R3: claim ")
            << c.getSymName()
            << " declares a domain not backed by any touched resource";
        ok = false;
      }
      for (Attribute a : c.getWrites()) {
        auto ref = dyn_cast<FlatSymbolRefAttr>(a);
        if (!ref)
          continue;
        auto it = resourceByName.find(ref.getValue());
        if (it != resourceByName.end() &&
            it->second.getDomainKind() == Domain::MMIO &&
            !hazardOrdered(c.getHazard())) {
          c.emitError("R3: claim ")
              << c.getSymName() << " MMIO write to @" << ref.getValue()
              << " requires an atomic/barriered hazard";
          ok = false;
        }
      }
    });
    root->walk([&](LoadOp l) {
      auto it = resourceByName.find(l.getSrc());
      if (it != resourceByName.end() &&
          it->second.getDomainKind() != l.getDomain()) {
        l.emitError(
            "R3: load domain contract does not match the resource domain of @")
            << l.getSrc();
        ok = false;
      }
    });
    root->walk([&](StoreOp s) {
      auto it = resourceByName.find(s.getDst());
      if (it != resourceByName.end() &&
          it->second.getDomainKind() != s.getDomain()) {
        s.emitError(
            "R3: store domain contract does not match the resource domain of @")
            << s.getDst();
        ok = false;
      }
    });

    // R4: phase DAG legality -- dependencies form an acyclic graph.
    llvm::DenseMap<StringRef, SmallVector<StringRef, 2>> deps;
    llvm::DenseMap<StringRef, PhaseOp> phaseOps;
    SmallVector<StringRef> phases;
    root->walk([&](PhaseOp p) {
      phases.push_back(p.getSymName());
      phaseOps[p.getSymName()] = p;
      for (Attribute a : p.getDeps())
        if (auto ref = dyn_cast<FlatSymbolRefAttr>(a))
          deps[p.getSymName()].push_back(ref.getValue());
    });
    // 0 = unvisited, 1 = on stack, 2 = done.
    llvm::DenseMap<StringRef, int> color;
    std::function<bool(StringRef)> hasCycle = [&](StringRef n) -> bool {
      color[n] = 1;
      for (StringRef d : deps.lookup(n)) {
        int c = color.lookup(d);
        if (c == 1)
          return true;
        if (c == 0 && hasCycle(d))
          return true;
      }
      color[n] = 2;
      return false;
    };
    for (StringRef p : phases) {
      if (color.lookup(p) == 0 && hasCycle(p)) {
        phaseOps[p].emitError("R4: phase dependency cycle through @") << p;
        ok = false;
        break;
      }
    }

    // R5: hazard legality -- atomic semantics carry an ordered hazard contract,
    // and same-phase conflicts through the decoupled GGG tail are ordered.
    llvm::DenseMap<StringRef, SmallVector<ClaimOp, 4>> claimsByPhase;
    llvm::DenseMap<StringRef, ClaimOp> claimByName;
    root->walk([&](ClaimOp c) {
      claimByName[c.getSymName()] = c;
      claimsByPhase[c.getPhase()].push_back(c);
      StringRef op = c.getOp();
      bool atomicSemantics = c.getLane() == Lane::A ||
                             op.starts_with("atomic") || op.contains("cmpxchg");
      if (atomicSemantics && !hazardOrdered(c.getHazard())) {
        c.emitError("R5: claim ")
            << c.getSymName()
            << " atomic semantics require an atomic/barriered hazard";
        ok = false;
      }
    });
    for (auto &entry : claimsByPhase) {
      auto &group = entry.second;
      for (size_t i = 0; i < group.size(); ++i) {
        for (size_t j = i + 1; j < group.size(); ++j) {
          ClaimOp a = group[i], b = group[j];
          if (!(claimSparse(a) || claimSparse(b)) || !claimsConflict(a, b))
            continue;
          for (ClaimOp c : {a, b}) {
            if (!hazardOrdered(c.getHazard())) {
              c.emitError("R5: claim ")
                  << c.getSymName()
                  << " conflicts across the decoupled GGG tail in phase @"
                  << entry.first << " without an atomic/barriered hazard";
              ok = false;
            }
          }
        }
      }
    }

    // R6: lane legality -- lane matches the declared access pattern.
    root->walk([&](ClaimOp c) {
      if (!laneLegalForStride(c.getLane(), c.getStrideClass())) {
        c.emitError("R6: claim ")
            << c.getSymName() << " lane illegal for its stride class";
        ok = false;
      }
    });

    // R7: bounds legality -- strict bounds discharge statically for affine
    // patterns; data-dependent patterns need a runtime verify contract.
    auto resourceCount = [](ResourceOp r) -> int64_t {
      ArrayRef<int64_t> shape = r.getShape();
      if (shape.empty())
        return 0; // unknown extent: not statically checkable
      int64_t n = 1;
      for (int64_t d : shape)
        n *= d;
      return n;
    };
    root->walk([&](ClaimOp c) {
      if (c.getBounds() != Bounds::Strict)
        return;
      StrideClass sc = c.getStrideClass();
      bool dataDependent =
          sc == StrideClass::Cacheline || sc == StrideClass::Random;
      if (dataDependent) {
        if (c.getVerify() == Verify::None) {
          c.emitError("R7: claim ")
              << c.getSymName()
              << " data-dependent access with strict bounds requires a "
                 "runtime verify contract";
          ok = false;
        }
        return;
      }
      // Affine: the stride applies to the streamed read source; writes land
      // unit-stride (a conservative under-approximation, never a false
      // positive). A reduction (op "reduce.*") accumulates count reads into a
      // single location, so its write extent is one element, not count (mirrors
      // bcir/verify R7).
      int64_t count = static_cast<int64_t>(c.getCount());
      int64_t offset = static_cast<int64_t>(c.getOffset());
      int64_t k = std::max<int64_t>(1, c.getStrideK());
      int64_t readExtent = count > 0 ? offset + (count - 1) * k + 1 : 0;
      bool isReduction = c.getOp().starts_with("reduce.");
      int64_t writeExtent = offset + (isReduction ? 1 : count);
      auto checkExtent = [&](ArrayAttr refs, int64_t extent, StringRef kind) {
        for (Attribute a : refs) {
          auto ref = dyn_cast<FlatSymbolRefAttr>(a);
          if (!ref)
            continue;
          auto it = resourceByName.find(ref.getValue());
          if (it == resourceByName.end())
            continue;
          int64_t n = resourceCount(it->second);
          if (n > 0 && extent > n) {
            c.emitError("R7: claim ")
                << c.getSymName() << " " << kind << " of @" << ref.getValue()
                << " overruns the resource (extent " << extent << " > " << n
                << ")";
            ok = false;
          }
        }
      };
      checkExtent(c.getReads(), readExtent, "read");
      checkExtent(c.getWrites(), writeExtent, "write");
    });

    // R8/R9: K_BCIR plan laws -- every candidate carries a declared cost path
    // (R8); the selection is drawn from its candidate set and realizes the
    // claim it plans (R9).
    llvm::DenseMap<StringRef, KBCIRPathOp> pathByName;
    llvm::DenseMap<StringRef, KBCIRPlanOp> planByName;
    llvm::DenseMap<StringRef, KBCIRBudgetOp> budgetByName;
    root->walk([&](KBCIRPathOp p) { pathByName[p.getSymName()] = p; });
    root->walk([&](KBCIRPlanOp p) { planByName[p.getSymName()] = p; });
    static const llvm::DenseSet<StringRef> kCostDims = {
        "compute", "memory", "fabric", "sync", "compile", "thermal",
        "power", "reliability", "security", "accuracy", "contention",
        "verification"};
    root->walk([&](KBCIRBudgetOp b) {
      budgetByName[b.getSymName()] = b;
      if (b.getDims().size() != static_cast<size_t>(b.getCaps().size())) {
        b.emitError("R8: budget ")
            << b.getSymName() << " dims/caps arity mismatch";
        ok = false;
      }
      for (Attribute a : b.getDims()) {
        auto name = dyn_cast<StringAttr>(a);
        if (!name || !kCostDims.count(name.getValue())) {
          b.emitError("R8: budget ")
              << b.getSymName() << " names an unknown cost dimension";
          ok = false;
        }
      }
    });
    root->walk([&](KBCIRSelectOp s) {
      if (s.getFrom().empty()) {
        s.emitError("R8: empty candidate set for claim @") << s.getClaim();
        ok = false;
      }
      bool selectedInFrom = false;
      for (Attribute a : s.getFrom()) {
        auto ref = dyn_cast<FlatSymbolRefAttr>(a);
        if (!ref)
          continue;
        if (ref.getValue() == s.getSelected())
          selectedInFrom = true;
        if (!pathByName.count(ref.getValue())) {
          s.emitError("R8: candidate path @")
              << ref.getValue() << " does not resolve to a declared cost path";
          ok = false;
        }
      }
      if (!selectedInFrom) {
        s.emitError("R9: selected path @")
            << s.getSelected() << " is not among the candidate set";
        ok = false;
      }
      if (auto p = pathByName.lookup(s.getSelected())) {
        if (p.getClaim() != s.getClaim()) {
          s.emitError("R9: selected path @")
              << s.getSelected() << " realizes claim @" << p.getClaim()
              << ", not @" << s.getClaim();
          ok = false;
        }
      }
      if (static_cast<int64_t>(s.getScore()) < 0) {
        s.emitError("R9: negative plan score");
        ok = false;
      }
      // Constrained rail (RCSP): a selection under a budget must pick a path
      // whose cost satisfies every cap -- R(pi) <= B (LangRef Sec. 2).
      if (auto bref = s.getBudgetAttr()) {
        auto b = budgetByName.lookup(bref.getValue());
        if (!b) {
          s.emitError("R8: budget @")
              << bref.getValue() << " does not resolve";
          ok = false;
        } else if (auto p = pathByName.lookup(s.getSelected())) {
          ArrayAttr dims = b.getDims();
          ArrayRef<int64_t> caps = b.getCaps();
          for (size_t i = 0; i < dims.size() && i < caps.size(); ++i) {
            auto name = dyn_cast<StringAttr>(dims[i]);
            if (!name)
              continue;
            auto v = costDim(p.getCost(), name.getValue());
            if (v && *v > caps[i]) {
              s.emitError("R9: selected path @")
                  << s.getSelected() << " violates budget @" << bref.getValue()
                  << " (" << name.getValue() << " " << *v << " > " << caps[i]
                  << ")";
              ok = false;
            }
          }
        }
      }
    });

    // R9: the (max,+) scheduled price is consistent with its serial bound --
    // makespan + overlap_gain == serial -- and references a declared plan.
    root->walk([&](KBCIRScheduledPriceOp sp) {
      if (!planByName.count(sp.getPlan())) {
        sp.emitError("R9: scheduled price references unknown plan @")
            << sp.getPlan();
        ok = false;
      }
      int64_t makespan = static_cast<int64_t>(sp.getMakespan());
      int64_t serial = static_cast<int64_t>(sp.getSerial());
      int64_t gain = static_cast<int64_t>(sp.getOverlapGain());
      if (makespan < 0 || makespan + gain != serial) {
        sp.emitError("R9: inconsistent scheduled price (makespan ")
            << makespan << " + overlap_gain " << gain << " != serial " << serial
            << ")";
        ok = false;
      }
    });

    // R9: the finite-temperature soft select is consistent with its tropical
    // twin -- the Gibbs free energy never exceeds the hard minimum (softmin <=
    // min), and at temperature 0 it recovers it exactly.
    root->walk([&](KBCIRSoftSelectOp ss) {
      int64_t T = static_cast<int64_t>(ss.getTemperatureMilli());
      int64_t F = static_cast<int64_t>(ss.getFreeEnergy());
      int64_t score = static_cast<int64_t>(ss.getScore());
      if (T < 0) {
        ss.emitError("R9: soft_select temperature must be >= 0");
        ok = false;
      }
      if (F > score) {
        ss.emitError("R9: soft_select free_energy ")
            << F << " exceeds the hard minimum " << score;
        ok = false;
      }
      if (T == 0 && F != score) {
        ss.emitError("R9: at temperature 0 soft_select free_energy ")
            << F << " must equal the tropical score " << score;
        ok = false;
      }
    });

    // R9: the equality-saturation building-blocks engine never raises the
    // selected cost -- a rewrite is legal only if optimized_cost <= original_cost
    // (the bound on the "abundance of unliked pairs"; LangRef Sec. 11).
    root->walk([&](EGraphExtractOp eg) {
      int64_t orig = static_cast<int64_t>(eg.getOriginalCost());
      int64_t opt = static_cast<int64_t>(eg.getOptimizedCost());
      if (orig < 0 || opt < 0 || static_cast<int64_t>(eg.getIterations()) < 0) {
        eg.emitError("R9: egraph.extract costs/iterations out of range");
        ok = false;
      }
      if (opt > orig) {
        eg.emitError("R9: egraph rewrite raised the cost (optimized ")
            << opt << " > original " << orig << ")";
        ok = false;
      }
    });

    // R8: an L1 calibration table is well-formed -- the streaming baseline is
    // the Q8 unit by definition, ratios floor at it, the table is
    // generation-tagged, and its target resolves.
    llvm::DenseMap<StringRef, TargetCapabilityOp> capabilityByName;
    root->walk(
        [&](TargetCapabilityOp t) { capabilityByName[t.getSymName()] = t; });
    root->walk([&](KBCIRCalibrationOp cal) {
      if (!capabilityByName.count(cal.getTarget())) {
        cal.emitError("R8: calibration target @")
            << cal.getTarget() << " does not resolve";
        ok = false;
      }
      if (static_cast<int64_t>(cal.getStreamQ8()) != 256) {
        cal.emitError(
            "R8: calibration stream_q8 must be 256 (the Q8 baseline)");
        ok = false;
      }
      if (static_cast<int64_t>(cal.getCalGen()) < 1 ||
          static_cast<int64_t>(cal.getStridedQ8()) < 256 ||
          static_cast<int64_t>(cal.getRandomQ8()) < 256 ||
          static_cast<int64_t>(cal.getComputeQ8()) < 256) {
        cal.emitError("R8: calibration cal_gen/ratios out of range");
        ok = false;
      }
      // Conformal guarantee (Bayesian table): a claimed coverage must be a real
      // probability and the certified +/- delta non-negative. coverage 0 = a
      // plain point table (no guarantee), which is always well-formed.
      int64_t cov = static_cast<int64_t>(cal.getCoverageMilli());
      if (cov != 0 && !(cov > 0 && cov < 1000)) {
        cal.emitError("R8: conformal coverage ")
            << cov << "/1000 out of range (0,1000)";
        ok = false;
      }
      if (static_cast<int64_t>(cal.getRandomDeltaQ8()) < 0) {
        cal.emitError("R8: conformal delta must be >= 0");
        ok = false;
      }
    });

    // R9: L2 learning placement -- portfolios carry parallel generation/
    // certification state over declared policies; a replay certificate admits
    // only on zero regressions over at least one episode.
    llvm::DenseMap<StringRef, KBCIRPolicyOp> policyByName;
    root->walk([&](KBCIRPolicyOp p) { policyByName[p.getSymName()] = p; });
    root->walk([&](KBCIRPortfolioOp pf) {
      size_t n = pf.getPolicies().size();
      if (n != static_cast<size_t>(pf.getGens().size()) ||
          n != static_cast<size_t>(pf.getCertified().size())) {
        pf.emitError("R9: portfolio policies/gens/certified arity mismatch");
        ok = false;
      }
      for (Attribute a : pf.getPolicies()) {
        auto ref = dyn_cast<FlatSymbolRefAttr>(a);
        if (ref && !policyByName.count(ref.getValue())) {
          pf.emitError("R9: portfolio references unknown policy @")
              << ref.getValue();
          ok = false;
        }
      }
      for (int64_t g : pf.getGens())
        if (g < 1) {
          pf.emitError("R9: portfolio generation tags must be >= 1");
          ok = false;
          break;
        }
      for (int64_t c : pf.getCertified())
        if (c != 0 && c != 1) {
          pf.emitError("R9: portfolio certification flags must be 0 or 1");
          ok = false;
          break;
        }
    });
    root->walk([&](KBCIRReplayCertificateOp rc) {
      if (!policyByName.count(rc.getCandidate()) ||
          !policyByName.count(rc.getIncumbent())) {
        rc.emitError("R9: replay certificate references an unknown policy");
        ok = false;
      }
      int64_t episodes = static_cast<int64_t>(rc.getEpisodes());
      int64_t regressions = static_cast<int64_t>(rc.getRegressions());
      if (episodes < 1 || regressions < 0) {
        rc.emitError("R9: replay certificate episodes/regressions out of range");
        ok = false;
      }
      if (rc.getAdmitted() && regressions != 0) {
        rc.emitError("R9: admitted replay certificate carries ")
            << regressions << " regression(s)";
        ok = false;
      }
    });

    // R13: policy provenance -- every decision rule in force is generation-
    // tagged and witnessed. A promoted portfolio entry (gen > 1) requires an
    // admitting replay certificate for that policy; a calibrated capability
    // (cal_gen >= 1) requires its matching frozen-table certificate; a regret
    // ledger's books must balance. Rule swaps are never silent.
    llvm::DenseSet<StringRef> admittedCandidates;
    root->walk([&](KBCIRReplayCertificateOp rc) {
      if (rc.getAdmitted())
        admittedCandidates.insert(rc.getCandidate());
    });
    root->walk([&](KBCIRPortfolioOp pf) {
      ArrayAttr policies = pf.getPolicies();
      ArrayRef<int64_t> gens = pf.getGens();
      for (size_t i = 0; i < policies.size() && i < gens.size(); ++i) {
        auto ref = dyn_cast<FlatSymbolRefAttr>(policies[i]);
        if (ref && gens[i] > 1 && !admittedCandidates.count(ref.getValue())) {
          pf.emitError("R13: promoted policy @")
              << ref.getValue() << " (gen " << gens[i]
              << ") has no admitting replay certificate";
          ok = false;
        }
      }
    });
    llvm::DenseMap<StringRef, KBCIRCalibrationOp> calibrationByTarget;
    root->walk([&](KBCIRCalibrationOp cal) {
      calibrationByTarget[cal.getTarget()] = cal;
    });
    root->walk([&](TargetCapabilityOp t) {
      int64_t gen = static_cast<int64_t>(t.getCalGen());
      if (gen < 1)
        return; // seeded constants: nothing to certify
      auto cal = calibrationByTarget.lookup(t.getSymName());
      if (!cal || static_cast<int64_t>(cal.getCalGen()) != gen) {
        t.emitError("R13: capability @")
            << t.getSymName() << " claims cal_gen " << gen
            << " without a matching calibration certificate";
        ok = false;
      }
    });
    root->walk([&](KBCIRRegretLedgerOp rl) {
      if (!policyByName.count(rl.getRule())) {
        rl.emitError("R13: regret ledger references unknown rule @")
            << rl.getRule();
        ok = false;
      }
      int64_t episodes = static_cast<int64_t>(rl.getEpisodes());
      int64_t total = static_cast<int64_t>(rl.getTotalRegret());
      int64_t worst = static_cast<int64_t>(rl.getWorstRegret());
      if (episodes < 0 || worst < 0 || total < worst ||
          static_cast<int64_t>(rl.getGen()) < 1) {
        rl.emitError("R13: regret ledger books do not balance");
        ok = false;
      }
      // The MDL / Bayesian-evidence retune trigger: a "retune" verdict must be
      // backed by description-length evidence (data_fit > complexity); a "keep"
      // must not sit on regret that has already outgrown the complexity cost.
      int64_t fit = static_cast<int64_t>(rl.getDataFitMilli());
      int64_t cx = static_cast<int64_t>(rl.getComplexityMilli());
      StringRef verdict = rl.getVerdict();
      if (verdict != "keep" && verdict != "retune") {
        rl.emitError("R13: regret ledger verdict must be keep or retune");
        ok = false;
      } else if ((verdict == "retune") != (fit > cx)) {
        rl.emitError("R13: regret verdict '")
            << verdict << "' inconsistent with the MDL evidence (data_fit "
            << fit << " vs complexity " << cx << ")";
        ok = false;
      }
    });

    // R13: learned-gate provenance -- a deployed MoE gate (kbcir.moegate) routes
    // among certified portfolio experts and may deploy only behind an admitting
    // replay certificate (zero regressions vs the incumbent classify router). The
    // network proposes a route; the verifier disposes.
    llvm::DenseSet<StringRef> portfolioNames;
    root->walk([&](KBCIRPortfolioOp pf) { portfolioNames.insert(pf.getSymName()); });
    root->walk([&](KBCIRMoEGateOp gate) {
      if (!portfolioNames.count(gate.getPortfolio())) {
        gate.emitError("R13: MoE gate routes to unknown portfolio @")
            << gate.getPortfolio();
        ok = false;
      }
      if (static_cast<int64_t>(gate.getNumExperts()) < 1 ||
          static_cast<int64_t>(gate.getHidden()) < 1 ||
          static_cast<int64_t>(gate.getGateGen()) < 1 ||
          static_cast<int64_t>(gate.getEpisodes()) < 1 ||
          static_cast<int64_t>(gate.getRegressions()) < 0) {
        gate.emitError("R13: MoE gate dimensions/books out of range");
        ok = false;
      }
      int64_t regressions = static_cast<int64_t>(gate.getRegressions());
      if (gate.getAdmitted() && regressions != 0) {
        gate.emitError("R13: admitted MoE gate carries ")
            << regressions << " regression(s)";
        ok = false;
      }
      if (!gate.getAdmitted()) {
        gate.emitError("R13: deployed MoE gate did not pass its replay gate");
        ok = false;
      }
    });

    // R13: search-accelerator provenance -- a learned candidate ordering
    // (kbcir.accel) speeds the exact search but must reproduce the exact optimum.
    // A deployed accelerator carries an equivalence certificate with zero
    // mismatches; ordering changes work, never the result.
    root->walk([&](KBCIRSearchAccelOp acc) {
      int64_t checked = static_cast<int64_t>(acc.getChecked());
      int64_t mism = static_cast<int64_t>(acc.getMismatches());
      if (checked < 1 || mism < 0) {
        acc.emitError("R13: search accelerator checked/mismatches out of range");
        ok = false;
      }
      if (acc.getAdmitted() && mism != 0) {
        acc.emitError("R13: admitted search accelerator changed the optimum (")
            << mism << " mismatch(es))";
        ok = false;
      }
      if (!acc.getAdmitted()) {
        acc.emitError("R13: deployed search accelerator is not certified exact");
        ok = false;
      }
    });

    // R13: amortization provenance (the L1 cost throttle) -- a deployed learned
    // component must belong at its tier: L0 carries no inference, and elsewhere
    // it pays for itself (gain >= inference_cost) within the tier's budget.
    root->walk([&](KBCIRAmortizationOp am) {
      int64_t cost = static_cast<int64_t>(am.getInferenceCost());
      int64_t gain = static_cast<int64_t>(am.getGain());
      int64_t budget = static_cast<int64_t>(am.getBudget());
      if (am.getTier() == "L0" && cost != 0) {
        am.emitError("R13: component ")
            << am.getComponent() << " at L0 runs learned inference (cost " << cost
            << "); the hot path carries decisions, not models";
        ok = false;
      } else if (cost < 0 || gain < cost || cost > budget) {
        am.emitError("R13: component ")
            << am.getComponent() << " fails amortization (gain " << gain
            << " vs cost " << cost << ", budget " << budget << ")";
        ok = false;
      }
    });

    // R13: provenance-manifest reproducibility -- a deployed plan's manifest (the
    // commit hash of its inputs + in-force decision-rule generations) must have
    // reproduced its recorded score/shape on replay. Manifest equality => the
    // identical plan; a plan that cannot be reproduced from its provenance is not
    // a closed branch.
    root->walk([&](KBCIRProvenanceManifestOp pm) {
      if (static_cast<int64_t>(pm.getScore()) < 0 ||
          static_cast<int64_t>(pm.getNArtifacts()) < 0) {
        pm.emitError("R13: manifest score/n_artifacts out of range");
        ok = false;
      }
      if (!pm.getReproduced()) {
        pm.emitError("R13: deployed plan manifest did not reproduce on replay");
        ok = false;
      }
    });

    // R9: a duration-aware schedule certificate is well-formed -- known mode,
    // non-negative makespan, knee/pipeline >= 1, and a declared plan.
    root->walk([&](GEMScheduleOp sc) {
      StringRef mode = sc.getMode();
      if (mode != "waves" && mode != "eft" && mode != "tokens") {
        sc.emitError("R9: unknown schedule mode '") << mode << "'";
        ok = false;
      }
      if (!planByName.count(sc.getPlan())) {
        sc.emitError("R9: schedule references unknown plan @") << sc.getPlan();
        ok = false;
      }
      if (static_cast<int64_t>(sc.getMakespan()) < 0 || sc.getKnee() < 1 ||
          sc.getPipelineDepth() < 1) {
        sc.emitError("R9: schedule makespan/knee/pipeline_depth out of range");
        ok = false;
      }
    });

    // R10: stream provenance -- segments map back to live claims/phases/
    // prefetches/resources; packs hydrate from a declared plan -- and the v2
    // pipeline/double-buffer contracts are well-formed.
    llvm::DenseMap<StringRef, GEMPrefetchOp> prefetchByName;
    root->walk([&](GEMPrefetchOp p) {
      prefetchByName[p.getSymName()] = p;
      if (p.getBuffers() != 1 && p.getBuffers() != 2) {
        p.emitError("R10: prefetch ")
            << p.getSymName() << " invalid buffer count (1 or 2)";
        ok = false;
      }
    });
    root->walk([&](GEMLaneSegmentOp seg) {
      if (!claimByName.count(seg.getClaim())) {
        seg.emitError("R10: segment ")
            << seg.getSymName() << " references unknown claim @"
            << seg.getClaim();
        ok = false;
      }
      if (!phaseOps.count(seg.getPhase())) {
        seg.emitError("R10: segment ")
            << seg.getSymName() << " references unknown phase @"
            << seg.getPhase();
        ok = false;
      }
      if (auto pf = seg.getPrefetchAttr()) {
        if (!prefetchByName.count(pf.getValue())) {
          seg.emitError("R10: segment ")
              << seg.getSymName() << " references undeclared prefetch @"
              << pf.getValue();
          ok = false;
        }
      }
      for (ArrayAttr refs : {seg.getReads(), seg.getWrites()}) {
        for (Attribute a : refs) {
          auto ref = dyn_cast<FlatSymbolRefAttr>(a);
          if (ref && !resourceByName.count(ref.getValue())) {
            seg.emitError("R10: segment ")
                << seg.getSymName() << " references undeclared resource @"
                << ref.getValue();
            ok = false;
          }
        }
      }
      // R14 (CIM/PIM dispatch legality): a segment dispatched to processing-in-memory
      // must be a reduction -- PIM does element-local reduce work, not general SIMD.
      // First-class here in -bcir-verify (dual-rail with verify.verify_cim and the
      // -bcir-lower-to-llvm checkpoint); mirrors gem.cim (only reduce.* is offloaded).
      if (seg.getDispatch() == "pim" && !seg.getOpcode().starts_with("reduce.")) {
        seg.emitError("R14: pim dispatch illegal for non-reduction op '")
            << seg.getOpcode() << "' (claim @" << seg.getClaim() << ")";
        ok = false;
      }
      // R15 (DVFS clock legality): clock_q8 must be a legal step in [64, 512]
      // (0.25x..2x), and a pim (memory-bound) segment must not overclock (clock_q8
      // <= 256) -- more core frequency does not help bandwidth-bound work. Dual-rail
      // with verify.verify_dvfs; mirrors gem.dvfs.
      int64_t clk = seg.getClockQ8();
      if (clk < 64 || clk > 512) {
        seg.emitError("R15: clock_q8 ") << clk << " out of legal range [64, 512]";
        ok = false;
      } else if (seg.getDispatch() == "pim" && clk > 256) {
        seg.emitError("R15: pim (memory-bound) segment must not overclock (clock_q8 ")
            << clk << " > 256)";
        ok = false;
      }
    });

    // R16 (allocator placement legality): an on-chip placement must fit -- a resource
    // placed in L1 must be <= 64 KiB, in L2 <= 4 MiB (static caps; size =
    // product(shape) * 4 B); L3/DRAM/HBM have no cap. First-class here in -bcir-verify
    // (dual-rail with verify.verify_allocator); mirrors kbcir.allocator's capacity gate.
    root->walk([&](ResourceOp r) {
      auto pl = r.getPlacement();
      if (!pl)
        return;
      ArrayRef<int64_t> shape = r.getShape();
      if (shape.empty())
        return; // dynamic extent: not statically checkable
      int64_t bytes = 4;
      for (int64_t d : shape)
        bytes *= (d > 0 ? d : 1);
      int64_t cap = (*pl == MemTier::L1) ? (64 * 1024)
                  : (*pl == MemTier::L2) ? (4 * 1024 * 1024) : 0;
      if (cap && bytes > cap) {
        r.emitError("R16: placement ") << stringifyMemTier(*pl) << " does not fit @"
            << r.getSymName() << " (" << bytes << " B > " << cap << " B)";
        ok = false;
      }
    });

    // R17 (accuracy-contract legality): a claim that declares an accuracy tolerance
    // (precision contract with tol > 0) must realize within it -- its static worst-case
    // Q8-ULP error bound must not exceed the tolerance. A reduce.* over `count` terms
    // drifts up to `count` ULP with the naive accumulator but only 1 ULP when exact
    // (compensated, the residual-carry MAC); any other op truncates at most 1 ULP. So a
    // tight tolerance on a long reduction is the law that FORCES the compensated
    // realization. First-class here in -bcir-verify (dual-rail with verify.verify_accuracy).
    root->walk([&](ClaimOp c) {
      auto prec = c.getPrecision();
      if (!prec)
        return;
      int64_t tol = prec.getToleranceQ16();
      if (tol <= 0)
        return; // no declared tolerance: unconstrained
      int64_t count = std::max<int64_t>(1, static_cast<int64_t>(c.getCount()));
      int64_t bound = c.getOp().starts_with("reduce.") ? (prec.getExact() ? 1 : count) : 1;
      if (bound > tol) {
        c.emitError("R17: accuracy bound ") << bound << " ULP exceeds tolerance " << tol
            << " ULP @" << c.getSymName()
            << " (a compensated reduction would bound it at 1)";
        ok = false;
      }
    });

    // R11: generation validity -- pack tags match the live registry maxima; a
    // mismatch is a stale pack that must rehydrate (patch/repack/replan).
    uint64_t regMapGen = 0, regDataGen = 0;
    bool anyResources = !resourceByName.empty();
    for (auto &it : resourceByName) {
      regMapGen = std::max(regMapGen, it.second.getMapGen());
      regDataGen = std::max(regDataGen, it.second.getDataGen());
    }
    root->walk([&](GEMStreamPackOp sp) {
      if (!planByName.count(sp.getSourcePlan())) {
        sp.emitError("R10: stream pack source plan @")
            << sp.getSourcePlan() << " does not resolve";
        ok = false;
      }
      if (sp.getPipelineDepth() < 1) {
        sp.emitError("R10: invalid pipeline_depth (must be >= 1)");
        ok = false;
      }
      if (sp.getTopoGen() < 1) {
        sp.emitError("R11: invalid topo_gen (must be >= 1)");
        ok = false;
      }
      if (!anyResources)
        return;
      if (sp.getMapGen() != regMapGen) {
        sp.emitError("R11: stale StreamPack: map_gen ")
            << sp.getMapGen() << " != registry " << regMapGen
            << " (rehydrate: repack)";
        ok = false;
      }
      if (sp.getDataGen() != regDataGen) {
        sp.emitError("R11: stale StreamPack: data_gen ")
            << sp.getDataGen() << " != registry " << regDataGen
            << " (rehydrate: replan)";
        ok = false;
      }
    });

    // R12: lowering legality -- a lowering contract preserves the BCIR
    // semantic (lane is the contract's own field; bounds/hazard/precision must
    // be named) or carries an explicit discharge attribute.
    root->walk([&](TargetLowerContractOp lc) {
      StringRef p = lc.getPreserves();
      bool preserved = p.contains("bounds") && p.contains("hazard") &&
                       p.contains("precision");
      if (!preserved && !lc->hasAttr("discharge")) {
        lc.emitError("R12: lowering contract ")
            << lc.getSymName()
            << " must preserve bounds/hazard/precision or carry an explicit "
               "discharge";
        ok = false;
      }
    });

    if (!ok)
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createVerifyPass() {
  return std::make_unique<VerifyPass>();
}

}  // namespace bcir
