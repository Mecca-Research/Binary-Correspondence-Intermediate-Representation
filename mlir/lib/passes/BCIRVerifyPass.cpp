//===- BCIRVerifyPass.cpp - the -bcir-verify semantic laws R1-R24 -*- C++ -*-===//
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
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>

using namespace mlir;

namespace bcir {
namespace {

// FNV-1a over a canonical item sequence, byte-identical to provenance._fnv: each item's
// UTF-8 string is folded in byte by byte, then a 0xFF field separator, and the result is
// masked into signed-i63 range. The uint64 multiply wraps mod 2^64 exactly as the Python
// `& _MASK`. Used to recompute the R13 provenance digest (provenance._digest).
static constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
static constexpr uint64_t kFnvPrime = 1099511628211ULL;

static uint64_t fnvItem(uint64_t h, llvm::StringRef s) {
  for (unsigned char c : s)
    h = (h ^ static_cast<uint64_t>(c)) * kFnvPrime;
  h = (h ^ 0xFFULL) * kFnvPrime; // field separator
  return h;
}
static int64_t fnvMask(uint64_t h) {
  return static_cast<int64_t>(h & 0x7FFFFFFFFFFFFFFFULL);
}
static uint64_t fnvInt(uint64_t h, int64_t v) { return fnvItem(h, std::to_string(v)); }

// provenance.hash_theta recomputed from a kbcir.theta op: FNV-1a over the eight 0..100
// pressures (str(int) each, byte-identical to the oracle).
static int64_t hashThetaFromIR(KBCIRThetaOp t) {
  uint64_t h = kFnvOffset;
  for (int64_t v : {static_cast<int64_t>(t.getThermal()), static_cast<int64_t>(t.getPower()),
                    static_cast<int64_t>(t.getMemPressure()),
                    static_cast<int64_t>(t.getContention()),
                    static_cast<int64_t>(t.getNoise()), static_cast<int64_t>(t.getWear()),
                    static_cast<int64_t>(t.getUtilization()),
                    static_cast<int64_t>(t.getVoltage())})
    h = fnvInt(h, v);
  return fnvMask(h);
}

// provenance.hash_policy: _fnv(policy.name, tuple(policy.base)). The name is the mode
// spelling (Policy.name == the mode, e.g. "latency"); base is the UNFOLDED base_weights.
static int64_t hashPolicyFromIR(KBCIRPolicyOp pol) {
  uint64_t h = kFnvOffset;
  h = fnvItem(h, stringifyPolicyMode(pol.getMode()));
  if (auto base = pol.getBaseWeights())
    for (int64_t w : *base)
      h = fnvInt(h, w);
  return fnvMask(h);
}

// provenance.hash_target: _fnv over the TargetProfile fields, flattened (sorted lane widths
// expand to scalars; scalable is a Python bool -> "True"/"False").
static int64_t hashTargetFromIR(TargetCapabilityOp cap) {
  uint64_t h = kFnvOffset;
  h = fnvItem(h, cap.getTargetName());
  h = fnvItem(h, cap.getTriple());
  h = fnvInt(h, cap.getCacheline());
  h = fnvInt(h, cap.getElemBytes());
  SmallVector<int64_t> widths(cap.getLaneWidths().begin(), cap.getLaneWidths().end());
  std::sort(widths.begin(), widths.end());
  for (int64_t w : widths)
    h = fnvInt(h, w);
  h = fnvInt(h, cap.getWarp());
  h = fnvItem(h, cap.getScalable() ? StringRef("True") : StringRef("False"));
  h = fnvInt(h, cap.getGatherPenalty());
  h = fnvInt(h, cap.getMemUnit());
  h = fnvInt(h, cap.getBaseOverhead());
  h = fnvInt(h, cap.getThermalDensity());
  h = fnvInt(h, cap.getPowerDensity());
  h = fnvInt(h, cap.getPerOpHeat());
  h = fnvInt(h, cap.getAffinityDomains());
  h = fnvInt(h, cap.getMemChannels());
  h = fnvInt(h, cap.getCalGen());
  return fnvMask(h);
}

// provenance.hash_module recomputed from the bcir.module IR: the goal-graph name/cacheline/
// align, then each resource (sorted by rid) and each phase's claims (sorted by id), every
// field flattened to a scalar exactly as _flatten yields it. Reads/writes are the resolved
// RIDs (the oracle hashes c.rd / c.wr, the integer RIDs).
static int64_t hashModuleFromIR(Operation *modOp) {
  uint64_t h = kFnvOffset;
  StringRef name;
  if (auto a = modOp->getAttrOfType<StringAttr>("sym_name"))
    name = a.getValue();
  auto attrI = [&](StringRef k, int64_t def) -> int64_t {
    if (auto a = modOp->getAttrOfType<IntegerAttr>(k))
      return a.getInt();
    return def;
  };
  h = fnvItem(h, name);
  h = fnvInt(h, attrI("cacheline", 64));
  h = fnvInt(h, attrI("align", 64));

  SmallVector<ResourceOp> resources;
  llvm::DenseMap<StringRef, int64_t> ridOf;
  modOp->walk([&](ResourceOp r) {
    resources.push_back(r);
    ridOf[r.getSymName()] = static_cast<int64_t>(r.getRid());
  });
  std::sort(resources.begin(), resources.end(),
            [](ResourceOp a, ResourceOp b) { return a.getRid() < b.getRid(); });
  for (ResourceOp r : resources) {
    h = fnvInt(h, static_cast<int64_t>(r.getRid()));
    h = fnvInt(h, static_cast<int64_t>(static_cast<int>(r.getDomainKind())));
    for (int64_t d : r.getShape())
      h = fnvInt(h, d);
    h = fnvItem(h, stringifyLayout(r.getLayout()));
    h = fnvInt(h, static_cast<int64_t>(r.getAlign()));
    h = fnvItem(h, r.getAccess() ? stringifyAccess(*r.getAccess()) : StringRef("flat"));
    h = fnvInt(h, static_cast<int64_t>(r.getPriority()));
    h = fnvInt(h, static_cast<int64_t>(r.getMapGen()));
    h = fnvInt(h, static_cast<int64_t>(r.getDataGen()));
  }

  SmallVector<PhaseOp> phases;
  llvm::DenseMap<StringRef, int64_t> phaseIdOf;
  modOp->walk([&](PhaseOp p) {
    phases.push_back(p);
    phaseIdOf[p.getSymName()] = static_cast<int64_t>(p.getId());
  });
  for (PhaseOp ph : phases) {
    h = fnvInt(h, static_cast<int64_t>(ph.getId()));
    SmallVector<int64_t> deps;
    for (Attribute a : ph.getDeps())
      if (auto ref = dyn_cast<FlatSymbolRefAttr>(a))
        deps.push_back(phaseIdOf.lookup(ref.getValue()));
    std::sort(deps.begin(), deps.end());
    for (int64_t d : deps)
      h = fnvInt(h, d);
    SmallVector<ClaimOp> claims;
    modOp->walk([&](ClaimOp c) {
      if (c.getPhase() == ph.getSymName())
        claims.push_back(c);
    });
    std::sort(claims.begin(), claims.end(),
              [](ClaimOp a, ClaimOp b) { return a.getClaimId() < b.getClaimId(); });
    for (ClaimOp c : claims) {
      h = fnvInt(h, static_cast<int64_t>(c.getClaimId()));
      h = fnvInt(h, static_cast<int64_t>(c.getOpcode()));
      h = fnvInt(h, static_cast<int64_t>(static_cast<int>(c.getLane())));
      h = fnvInt(h, static_cast<int64_t>(static_cast<int>(c.getStrideClass())));
      h = fnvInt(h, static_cast<int64_t>(c.getCount()));
      h = fnvInt(h, static_cast<int64_t>(c.getStrideK()));
      for (Attribute a : c.getReads())
        if (auto ref = dyn_cast<FlatSymbolRefAttr>(a))
          h = fnvInt(h, ridOf.lookup(ref.getValue()));
      for (Attribute a : c.getWrites())
        if (auto ref = dyn_cast<FlatSymbolRefAttr>(a))
          h = fnvInt(h, ridOf.lookup(ref.getValue()));
      h = fnvItem(h, stringifyHazardMode(c.getHazard()));
      h = fnvInt(h, static_cast<int64_t>(static_cast<int>(c.getDomain())));
      h = fnvItem(h, stringifyVerify(c.getVerify()));
      h = fnvItem(h, stringifyBounds(c.getBounds()));
      h = fnvItem(h, c.getOp());
      h = fnvInt(h, static_cast<int64_t>(c.getOffset()));
      h = fnvItem(h, c.getCostClass() ? stringifyCostClass(*c.getCostClass())
                                      : StringRef("bandwidth"));
    }
  }
  return fnvMask(h);
}

struct VerifyPass : public PassWrapper<VerifyPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyPass)

  StringRef getArgument() const final { return "bcir-verify"; }
  StringRef getDescription() const final {
    return "Verify the BCIR semantic laws R1-R24: registry, domain, phase DAG, "
           "hazard, lane, bounds, cost, plan, provenance, generation, lowering, "
           "policy provenance, CIM/PIM dispatch, DVFS clock, allocator placement, "
           "accuracy contract, compositional call graph, synchronous timing, "
           "clock-domain crossing, pointer lifetime, gem shape/dtype seams.";
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
    // A device-ISOLATED domain: an MMIO register or NVM cell is a distinct address
    // space the host cannot transparently substitute for RAM/HBM/VRAM/CXL (which
    // are mutually addressable host memory a compute claim may legitimately stage
    // tiles across -- e.g. a matmul reading RAM tiles, accumulating in HBM). A claim
    // that declares a host domain but TOUCHES an MMIO/NVM resource (or vice versa) is
    // the data-redirection / MMIO-as-RAM gap: an isolated resource silently treated
    // as the wrong address space. Such a touch must MATCH the claim's declared domain.
    auto isIsolatedDomain = [](Domain d) {
      return d == Domain::MMIO || d == Domain::NVM;
    };
    root->walk([&](ClaimOp c) {
      bool anyResolved = false, domainBacked = false;
      // Tighten the FIRST-MATCH weakness: the old check set domainBacked on the FIRST
      // same-domain resource and accepted the claim even when OTHER touched resources
      // were cross-domain. We keep that "at least one backs it" check (host domains may
      // legitimately mix) BUT additionally require every touched ISOLATED-domain resource
      // to match the claim's declared domain -- so a claim cannot reach a device register
      // (MMIO) or NVM cell while declaring itself RAM/HBM, the redirection gap.
      auto touch = [&](ArrayAttr refs, StringRef which) {
        for (Attribute a : refs) {
          auto ref = dyn_cast<FlatSymbolRefAttr>(a);
          if (!ref)
            continue;
          auto it = resourceByName.find(ref.getValue());
          if (it == resourceByName.end())
            continue;
          anyResolved = true;
          Domain rd = it->second.getDomainKind();
          if (rd == c.getDomain())
            domainBacked = true;
          else if (isIsolatedDomain(rd) || isIsolatedDomain(c.getDomain())) {
            // Name whichever SIDE is the device-isolated one (the resource domain, the claim
            // domain, or both) so the diagnostic is never factually wrong -- a RAM resource
            // touched by an MMIO claim must not be called "device-isolated ram".
            const char *which_isolated =
                (isIsolatedDomain(rd) && isIsolatedDomain(c.getDomain())) ? "both domains are device-isolated and differ"
                : isIsolatedDomain(rd) ? "the resource is in a device-isolated domain"
                                       : "the claim declares a device-isolated domain";
            c.emitError("R3: claim ")
                << c.getSymName() << " " << which << " @" << ref.getValue()
                << " (domain " << stringifyDomain(rd) << ") does not match the claim domain "
                << stringifyDomain(c.getDomain()) << " -- " << which_isolated
                << ", so an isolated resource may not be reached as another address space";
            ok = false;
          }
        }
      };
      touch(c.getReads(), "reads");
      touch(c.getWrites(), "writes");
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
      // §5.14 Phase 2 (indirect-call effect): a dispatch claim's DECLARED callee signature,
      // when carried, must be well-formed "ret(params)" -- a malformed record would poison the
      // R18/commutation consumers silently. Vacuous when absent (the opaque-edge default).
      if (auto sig = c.getCalleeSig()) {
        if (!sig->contains("(")) {
          c.emitError("R18: claim ")
              << c.getSymName() << " malformed indirect-callee signature '" << *sig
              << "' (expected 'ret(params)')";
          ok = false;
        }
      }
      // §5.14 Phase 2: a VOLATILE access (MMIO) must carry an ordered hazard -- volatility is
      // an ordering/legality signal, never cosmetic. Mirrors verify.verify() R5 on the oracle;
      // vacuous unless a claim opts into `is_volatile` (non-disturbance).
      if (c.getIsVolatile() && !hazardOrdered(c.getHazard())) {
        c.emitError("R5: claim ")
            << c.getSymName()
            << " volatile access requires an atomic/barriered hazard";
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

    // R3/EV (driver roadmap Part VII A1+B1): EVENT PHASES -- first-class asynchronous
    // entry. EV1: an event phase (non-empty `event` source) declares no phase deps --
    // its trigger is the event, never program order. EV2: every named source is ARMED
    // by an explicit irq.unmask:<src> claim in a program phase (enablement is a claim
    // over the controller resource, never implicit). EV3 (the interrupt-context
    // ordering seam): a resource WRITTEN by an event phase may be touched by program
    // claims only inside a masked window (irq.mask:<src> .. irq.unmask:<src>; program
    // phases walked in id order, claims in textual order -- the same first-slice bound
    // as the oracle) or by a Lane::A atomic claim. Mirrors kbcir/events.py
    // check_event_phases; vacuous when no phase carries an event source.
    {
      constexpr StringLiteral kMask("irq.mask:"), kUnmask("irq.unmask:");
      SmallVector<PhaseOp> eventPhases, programPhases;
      for (StringRef name : phases) {
        PhaseOp p = phaseOps[name];
        if (!p.getEvent().empty())
          eventPhases.push_back(p);
        else
          programPhases.push_back(p);
      }
      if (!eventPhases.empty()) {
        for (PhaseOp p : eventPhases) {
          if (!p.getDeps().empty()) {
            p.emitError("R3: EV1: event phase @")
                << p.getSymName() << " (source '" << p.getEvent()
                << "') declares phase deps -- asynchronous entry has no program-order "
                   "predecessor (hazards + masking order it)";
            ok = false;
          }
        }
        llvm::sort(programPhases,
                   [](PhaseOp a, PhaseOp b) { return a.getId() < b.getId(); });
        std::set<std::string> armed;
        for (PhaseOp p : programPhases)
          for (ClaimOp c : claimsByPhase.lookup(p.getSymName()))
            if (c.getOp().starts_with(kUnmask))
              armed.insert(c.getOp().drop_front(kUnmask.size()).str());
        std::set<std::string> seenSources;
        for (PhaseOp p : eventPhases) {
          std::string src = p.getEvent().str();
          if (seenSources.insert(src).second && !armed.count(src)) {
            p.emitError("R3: EV2: event source '")
                << src << "' is never armed -- enablement must be an explicit "
                << kUnmask << src << " claim in the program flow, never implicit";
            ok = false;
          }
        }
        std::map<std::string, std::set<std::string>> handlerWrites;
        for (PhaseOp p : eventPhases)
          for (ClaimOp c : claimsByPhase.lookup(p.getSymName()))
            for (Attribute a : c.getWrites())
              if (auto ref = dyn_cast<FlatSymbolRefAttr>(a))
                handlerWrites[ref.getValue().str()].insert(p.getEvent().str());
        std::set<std::string> masked;
        for (PhaseOp p : programPhases) {
          for (ClaimOp c : claimsByPhase.lookup(p.getSymName())) {
            StringRef opName = c.getOp();
            if (opName.starts_with(kMask)) {
              masked.insert(opName.drop_front(kMask.size()).str());
              continue;
            }
            if (opName.starts_with(kUnmask)) {
              masked.erase(opName.drop_front(kUnmask.size()).str());
              continue;
            }
            if (c.getLane() == Lane::A) // single-claim atomicity: the other legal shape
              continue;
            auto touch = [&](ArrayAttr refs) {
              for (Attribute a : refs) {
                auto ref = dyn_cast<FlatSymbolRefAttr>(a);
                if (!ref)
                  continue;
                auto it = handlerWrites.find(ref.getValue().str());
                if (it == handlerWrites.end())
                  continue;
                for (const std::string &src : it->second) {
                  if (!masked.count(src)) {
                    c.emitError("R3: EV3: claim ")
                        << c.getSymName() << " touches @" << ref.getValue()
                        << ", which the '" << src
                        << "' handler writes, outside a masked window -- mask the "
                           "source around it or make the touch a Lane.A atomic (the "
                           "interrupted flow must order against the handler)";
                    ok = false;
                  }
                }
              }
            };
            touch(c.getReads());
            touch(c.getWrites());
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
    auto resourceCount = [&](ResourceOp r) -> std::optional<int64_t> {
      ArrayRef<int64_t> shape = r.getShape();
      if (shape.empty())
        return 0; // unknown extent: not statically checkable
      int64_t n = 1;
      for (int64_t d : shape) {
        if (!checkedMulNonnegative(n, d, n)) {
          r.emitError("R7: resource shape element count exceeds signed 64-bit range");
          ok = false;
          return std::nullopt;
        }
      }
      return n;
    };
    root->walk([&](ClaimOp c) {
      // §5.12 item 4, dual-railed (previously oracle-only) + the §5.14 Phase 2 extent-provenance
      // signal: a masked (runtime-bounds-checked) access must DECLARE the bounds verify contract
      // -- a promotion the backend would emit without a guard is a silent loss of the check. The
      // provenance (WHY the access is checked) is surfaced in the diagnostic when carried.
      if (c.getBounds() == Bounds::Masked && c.getVerify() != Verify::Bounds) {
        auto err = c.emitError("R7: claim ");
        err << c.getSymName()
            << " masked (runtime-bounds-checked) access must carry a 'bounds' verify contract";
        if (auto bp = c.getBoundsProvenance())
          err << " (extent provenance: " << *bp << ")";
        ok = false;
      }
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
      int64_t readExtent = 0;
      int64_t writeExtent = 0;
      bool isReduction = c.getOp().starts_with("reduce.");
      int64_t lastOffset;
      if (offset < 0 || count < 0 ||
          (count > 0 &&
           (!checkedMulNonnegative(count - 1, k, lastOffset) ||
            !checkedAddNonnegative(offset, lastOffset, readExtent) ||
            !checkedAddNonnegative(readExtent, 1, readExtent))) ||
          !checkedAddNonnegative(offset, isReduction ? 1 : count,
                                 writeExtent)) {
        c.emitError("R7: affine access extent exceeds signed 64-bit range");
        ok = false;
        return;
      }
      auto checkExtent = [&](ArrayAttr refs, int64_t extent, StringRef kind) {
        for (Attribute a : refs) {
          auto ref = dyn_cast<FlatSymbolRefAttr>(a);
          if (!ref)
            continue;
          auto it = resourceByName.find(ref.getValue());
          if (it == resourceByName.end())
            continue;
          std::optional<int64_t> n = resourceCount(it->second);
          if (n && *n > 0 && extent > *n) {
            c.emitError("R7: claim ")
                << c.getSymName() << " " << kind << " of @" << ref.getValue()
                << " overruns the resource (extent " << extent << " > " << *n
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
    root->walk([&](KBCIRPathOp p) {
      pathByName[p.getSymName()] = p;
      int64_t values[12];
      costToArray(p.getCost(), values);
      for (int64_t value : values) {
        if (value < 0) {
          p.emitError("R8: path cost dimensions must be non-negative");
          ok = false;
          break;
        }
      }
    });
    root->walk([&](KBCIRPlanOp p) { planByName[p.getSymName()] = p; });
    root->walk([&](KBCIRPolicyOp p) {
      auto validateWeights = [&](ArrayRef<int64_t> weights, StringRef which) {
        if (weights.size() != 12 ||
            std::any_of(weights.begin(), weights.end(),
                        [](int64_t value) { return value < 0; })) {
          p.emitError("R8: ") << which
              << " must contain exactly 12 non-negative weights";
          ok = false;
        }
      };
      validateWeights(p.getWeights(), "policy weights");
      if (auto base = p.getBaseWeights())
        validateWeights(*base, "policy base_weights");
    });
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
      for (int64_t cap : b.getCaps()) {
        if (cap < 0) {
          b.emitError("R8: budget caps must be non-negative");
          ok = false;
          break;
        }
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
      int64_t reconstructed;
      if (makespan < 0 || gain < 0 ||
          !checkedAddNonnegative(makespan, gain, reconstructed) ||
          reconstructed != serial) {
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

    // R13: memory-module admissibility -- a frozen a = Lim(Res(U)) is memory only when it
    // is a *saturated* fixpoint AND generation-tagged (gen >= 1); a budget cutoff or an
    // untagged artifact is not admissible. Mirrors bcir/kbcir/memory.py MemoryModule.admissible.
    root->walk([&](KBCIRMemoryModuleOp mm) {
      if (!mm.getSaturated() || mm.getGeneration() < 1) {
        mm.emitError("R13: memory module ")
            << mm.getSymName() << " is not admissible (saturated="
            << (mm.getSaturated() ? "true" : "false") << ", generation="
            << mm.getGeneration() << "); a = Lim(Res(U)) must be a saturated, "
            << "generation-tagged fixpoint";
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
      // R13 digest recompute: when the manifest carries the four component hashes
      // (the FNV content hashes of module/target/theta/policy), recompute the digest
      // from first principles (provenance._digest = _fnv(m_module, m_target, m_theta,
      // m_policy, artifacts)) and reject a tampered one -- the law no longer trusts the
      // declared `digest`. The artifacts (sorted (name, generation) pairs) fold in after
      // the components. Absent components (all zero) => back-compat range/reproduced only.
      int64_t mm = static_cast<int64_t>(pm.getMModule());
      int64_t mt = static_cast<int64_t>(pm.getMTarget());
      int64_t mth = static_cast<int64_t>(pm.getMTheta());
      int64_t mp = static_cast<int64_t>(pm.getMPolicy());
      if (mm || mt || mth || mp) {
        uint64_t h = kFnvOffset;
        for (int64_t comp : {mm, mt, mth, mp})
          h = fnvItem(h, std::to_string(comp));
        ArrayAttr names = pm.getArtifactNamesAttr();
        if (auto gens = pm.getArtifactGens()) {
          for (size_t i = 0; i < gens->size(); ++i) {
            if (names && i < names.size())
              if (auto s = dyn_cast<StringAttr>(names[i]))
                h = fnvItem(h, s.getValue());
            h = fnvItem(h, std::to_string((*gens)[i]));
          }
        }
        int64_t recomputed = static_cast<int64_t>(h & 0x7FFFFFFFFFFFFFFFULL);
        if (recomputed != static_cast<int64_t>(pm.getDigest())) {
          pm.emitError("R13: provenance digest ")
              << static_cast<int64_t>(pm.getDigest())
              << " does not match the digest recomputed from its component hashes "
              << recomputed << " (tampered or stale manifest)";
          ok = false;
        }
      }
      // R13 component cross-check against the IR: each component hash the manifest declares
      // (m_theta / m_policy / m_target / m_module) is recomputed from the actual IR of the
      // manifest's enclosing bcir.module -- byte-identical to provenance.hash_* -- and must
      // agree. So a manifest cannot be re-pointed at a different runtime state, policy,
      // target, or goal graph than the one it is attached to. Each check fires only when the
      // component is declared (non-zero) AND the IR carries the thing to hash (so a manifest
      // recorded alone -- digest only -- is unaffected; back-compatible).
      Operation *modOp = pm->getParentOp();
      while (modOp && modOp->getName().getStringRef() != "bcir.module")
        modOp = modOp->getParentOp();
      if (!modOp)
        return;
      KBCIRThetaOp theta;
      TargetCapabilityOp cap;
      KBCIRPolicyOp pol;
      bool hasClaim = false;
      modOp->walk([&](Operation *op) {
        if (auto t = dyn_cast<KBCIRThetaOp>(op)) {
          if (!theta)
            theta = t;
        } else if (auto cp = dyn_cast<TargetCapabilityOp>(op)) {
          if (!cap)
            cap = cp;
        } else if (auto p = dyn_cast<KBCIRPolicyOp>(op)) {
          if (!pol && p.getBaseWeights())
            pol = p;
        } else if (isa<ClaimOp>(op)) {
          hasClaim = true;
        }
      });
      auto crossCheck = [&](StringRef field, int64_t declared, int64_t recomputed) {
        if (declared && recomputed != declared) {
          pm.emitError("R13: manifest ")
              << field << " " << declared
              << " does not match the value recomputed from the IR " << recomputed
              << " (manifest attached to different inputs)";
          ok = false;
        }
      };
      if (mth && theta)
        crossCheck("m_theta", mth, hashThetaFromIR(theta));
      if (mp && pol)
        crossCheck("m_policy", mp, hashPolicyFromIR(pol));
      if (mt && cap)
        crossCheck("m_target", mt, hashTargetFromIR(cap));
      if (mm && hasClaim)
        crossCheck("m_module", mm, hashModuleFromIR(modOp));
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

    // R10 (cross-segment alias isolation): GEMLaneSegmentOp::verify is per-op and cannot
    // see siblings, so a write-write / write-read alias BETWEEN two concurrently-runnable
    // segments slips past parse-time verification. The verify pass DOES see all segments,
    // so do the cross-segment check here. Two segments in the SAME PHASE on DIFFERENT LANES
    // that alias on a write -- one writes a resource the other writes (WAW) or reads (RAW/
    // WAR) -- run concurrently (distinct lanes have no implicit wave serialization between
    // them, mirroring the R5 same-phase reasoning), so an unsynchronized write alias is a
    // data race / isolation violation. It is legal ONLY if an explicit fence on the aliased
    // resource orders the two: the resource appears in one segment's fence_after or the
    // other's fence_before. SAME-LANE aliases (the legitimate accumulator chain -- a tiled
    // matmul's C += A*B over K-tiles writes the same accumulator from successive same-lane
    // segments) ARE wave-serialized and not flagged, so the pretty corpus stays clean.
    llvm::DenseMap<StringRef, SmallVector<GEMLaneSegmentOp, 8>> segsByPhase;
    root->walk([&](GEMLaneSegmentOp seg) {
      segsByPhase[seg.getPhase()].push_back(seg);
    });
    for (auto &entry : segsByPhase) {
      auto &group = entry.second;
      for (size_t i = 0; i < group.size(); ++i) {
        for (size_t j = i + 1; j < group.size(); ++j) {
          GEMLaneSegmentOp a = group[i], b = group[j];
          if (a.getLane() == b.getLane())
            continue; // same lane: wave-serialized, the accumulator chain is legal
          auto aw = symbolSet(a.getWrites()), ar = symbolSet(a.getReads());
          auto bw = symbolSet(b.getWrites()), br = symbolSet(b.getReads());
          // the aliased resources: WAW + WAR + RAW between the two segments.
          llvm::DenseSet<StringRef> aliased;
          for (StringRef w : aw)
            if (bw.count(w) || br.count(w))
              aliased.insert(w);
          for (StringRef w : bw)
            if (ar.count(w))
              aliased.insert(w);
          if (aliased.empty())
            continue;
          // A fence ORDERS the alias only as a release/acquire PAIR: one segment must PUBLISH the
          // resource with `fence_after` (a release after its access) AND the other must ACQUIRE it
          // with `fence_before` (gate its access behind that publish). A LONE fence on one side does
          // not order the two -- a `fence_before` on the writer gates the writer behind something
          // else, not the reader behind the writer (pooling all four lists wrongly accepts that). So
          // require the full pair in EITHER direction: (a.after & b.before) OR (b.after & a.before).
          auto aAfter = symbolSet(a.getFenceAfter()), aBefore = symbolSet(a.getFenceBefore());
          auto bAfter = symbolSet(b.getFenceAfter()), bBefore = symbolSet(b.getFenceBefore());
          for (StringRef r : aliased) {
            bool aPubBAcq = aAfter.count(r) && bBefore.count(r);
            bool bPubAAcq = bAfter.count(r) && aBefore.count(r);
            if (aPubBAcq || bPubAAcq)
              continue; // a release/acquire fence pair orders the cross-lane access to @r
            a.emitError("R10: segment ")
                << a.getSymName() << " (lane " << laneSpelling(a.getLane())
                << ") and segment @" << b.getSymName() << " (lane "
                << laneSpelling(b.getLane()) << ") have a cross-lane write alias on @"
                << r << " in phase @" << entry.first
                << " without an ordering fence (a fence_after on one naming @" << r
                << " paired with a fence_before on the other) -- an isolation violation";
            ok = false;
          }
        }
      }
    }

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
      for (int64_t d : shape) {
        if (!checkedMulNonnegative(bytes, d > 0 ? d : 1, bytes)) {
          r.emitError("R16: resource byte extent exceeds signed 64-bit range");
          ok = false;
          return;
        }
      }
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
      auto prec = c.getPrecision();   // std::optional<PrecisionAttr>
      if (!prec)
        return;
      int64_t tol = prec->getToleranceQ16();
      if (tol <= 0)
        return; // no declared tolerance: unconstrained
      int64_t count = std::max<int64_t>(1, static_cast<int64_t>(c.getCount()));
      int64_t bound = c.getOp().starts_with("reduce.") ? (prec->getExact() ? 1 : count) : 1;
      if (bound > tol) {
        c.emitError("R17: accuracy bound ") << bound << " ULP exceeds tolerance " << tol
            << " ULP @" << c.getSymName()
            << " (a compensated reduction would bound it at 1)";
        ok = false;
      }
    });

    // R18 (compositional call-graph integrity): every kbcir.call resolves to a kbcir.func,
    // and the call graph is acyclic -- no recursion. The oracle's compose.plan_composite
    // raises on an undefined callee (KeyError) and on a recursive call (RecursionError, for
    // bounded compile time); this is that law on the law rail for the kbcir.func/call/cond
    // op family.
    {
      llvm::DenseMap<StringRef, Operation *> funcByName;
      root->walk([&](KBCIRFuncOp f) { funcByName[f.getSymName()] = f.getOperation(); });
      // call-graph edges: a func -> the callees of the kbcir.call ops in its body (attributed
      // to the nearest enclosing kbcir.func, so calls inside a kbcir.cond branch still count).
      llvm::DenseMap<StringRef, SmallVector<StringRef>> edges;
      root->walk([&](KBCIRCallOp c) {
        StringRef callee = c.getCallee();
        if (!funcByName.count(callee)) {
          c.emitError("R18: call to undefined function @") << callee;
          ok = false;
        }
        Operation *p = c->getParentOp();
        while (p && p->getName().getStringRef() != "bcir.kbcir.func")
          p = p->getParentOp();
        if (p)
          edges[cast<KBCIRFuncOp>(p).getSymName()].push_back(callee);
      });
      // DFS over the func graph (white 0 / gray 1 / black 2): a back edge to a gray node is
      // a recursive cycle. Roots in sorted name order for a deterministic verdict.
      SmallVector<StringRef> names;
      for (auto &kv : funcByName)
        names.push_back(kv.first);
      std::sort(names.begin(), names.end());
      llvm::DenseMap<StringRef, int> color;
      std::function<bool(StringRef)> visit = [&](StringRef n) -> bool {
        color[n] = 1;
        for (StringRef m : edges.lookup(n)) {
          if (!funcByName.count(m))
            continue; // unresolved callee already reported above
          int cm = color.lookup(m);
          if (cm == 1) {
            funcByName[m]->emitError("R18: recursive call cycle through @") << m;
            ok = false;
            return true;
          }
          if (cm == 0 && visit(m))
            return true;
        }
        color[n] = 2;
        return false;
      };
      for (StringRef n : names)
        if (color.lookup(n) == 0)
          visit(n);
    }

    // R19 (synchronous-timing legality) + R20 (clock-domain-crossing) over the OPTIONAL
    // claim.timing metadata -- the RTL/synchronous-timing track (§5.11) -- and R21
    // (pointer-lifetime: use-after-free / double-free) over the OPTIONAL claim.lifetime (§5.12).
    // A claim with no timing/lifetime is unconstrained, so the whole scalar/C subset verifies
    // identically (the non-disturbance invariant, exactly like R14-R17). Dual-rail with the
    // oracle's verify.verify_timing / verify.verify_lifetime. Claims are walked in claim_id order
    // so the writer->clock-domain map (R20) and the freed set (R21) accumulate deterministically.
    {
      SmallVector<ClaimOp> ordered;
      root->walk([&](ClaimOp c) { ordered.push_back(c); });
      std::sort(ordered.begin(), ordered.end(),
                [](ClaimOp a, ClaimOp b) { return a.getClaimId() < b.getClaimId(); });
      llvm::DenseMap<StringRef, std::string> writerDomain; // resource -> clock_domain of last writer
      llvm::DenseSet<StringRef> freed;                     // resources freed and not re-allocated
      for (ClaimOp c : ordered) {
        // --- R19 / R20 (timing) ---
        auto tm = c.getTiming();   // std::optional<TimingAttr>
        if (tm) {
          StringRef sync = tm->getSyncType();
          if (!(sync.empty() || sync == "synchronous" || sync == "asynchronous" ||
                sync == "mixed")) {
            c.emitError("R19: claim ")
                << c.getSymName() << " unknown sync_type '" << sync << "'";
            ok = false;
          }
          if (tm->getLatencyCycles() < 0) {
            c.emitError("R19: claim ") << c.getSymName() << " negative latency_cycles";
            ok = false;
          }
          if (tm->getSetupHoldMargin() < 0) {
            c.emitError("R19: claim ") << c.getSymName() << " negative setup_hold_margin";
            ok = false;
          }
          if (tm->getClockFrequencyMhz() < 0) {
            c.emitError("R19: claim ") << c.getSymName() << " negative clock_frequency_mhz";
            ok = false;
          }
          if (sync == "synchronous" && tm->getClockFrequencyMhz() <= 0) {
            c.emitError("R19: claim ")
                << c.getSymName() << " a synchronous claim needs a positive clock_frequency_mhz";
            ok = false;
          }
          if (tm->getLatencyCycles() > 0 &&
              tm->getSetupHoldMargin() > tm->getLatencyCycles()) {
            c.emitError("R19: claim ")
                << c.getSymName() << " setup_hold_margin " << tm->getSetupHoldMargin()
                << " exceeds the stage latency_cycles " << tm->getLatencyCycles();
            ok = false;
          }
          // R20: a RAW dependency that crosses clock domains must be synchronized -- the consumer
          // declares sync_type='mixed' OR a barriered hazard (the synchronizer / handshake).
          bool synchronized =
              (sync == "mixed") || (c.getHazard() == HazardMode::Barriered);
          StringRef dom = tm->getClockDomain();
          if (!dom.empty() && !synchronized) {
            for (Attribute a : c.getReads()) {
              auto ref = dyn_cast<FlatSymbolRefAttr>(a);
              if (!ref)
                continue;
              auto it = writerDomain.find(ref.getValue());
              if (it != writerDomain.end() && !it->second.empty() && it->second != dom) {
                c.emitError("R20: claim ")
                    << c.getSymName() << " reads @" << ref.getValue() << " from clock domain '"
                    << it->second << "' into '" << dom
                    << "' without a synchronizer (declare sync_type='mixed' or a barriered "
                       "hazard) -- an unguarded clock-domain crossing";
                ok = false;
              }
            }
          }
        }
        // every claim's writes set the clock domain of the resource they produce (dom = "" when
        // the writer carries no timing) -- so a later cross-domain read is detectable.
        std::string wdom = tm ? tm->getClockDomain().str() : std::string();
        for (Attribute a : c.getWrites())
          if (auto ref = dyn_cast<FlatSymbolRefAttr>(a))
            writerDomain[ref.getValue()] = wdom;

        // --- R21 (lifetime) ---
        auto lt = c.getLifetime();   // std::optional<LifetimeAttr>
        StringRef event = lt ? lt->getEvent() : StringRef("use");
        if (lt && event != "use" && event != "alloc" && event != "free") {
          c.emitError("R21: claim ")
              << c.getSymName() << " unknown lifetime event '" << event << "'";
          ok = false;
          event = "use";
        }
        for (Attribute a : c.getReads()) { // a READ of a freed resource is the dangling deref
          auto ref = dyn_cast<FlatSymbolRefAttr>(a);
          if (ref && freed.contains(ref.getValue())) {
            c.emitError("R21: claim ")
                << c.getSymName() << (event == "free" ? " double-free of @" : " use-after-free of @")
                << ref.getValue() << " (freed and not re-allocated)";
            ok = false;
          }
        }
        if (event == "free") // the read resources die after this claim
          for (Attribute a : c.getReads())
            if (auto ref = dyn_cast<FlatSymbolRefAttr>(a))
              freed.insert(ref.getValue());
        for (Attribute a : c.getWrites()) // a WRITE (reassignment / alloc) re-validates
          if (auto ref = dyn_cast<FlatSymbolRefAttr>(a))
            freed.erase(ref.getValue());
      }
    }

    // R12 (call-ABI contract, §5.14 Phase 2 -- the last area): a bcir.abi_contract must name a
    // target from the NORMATIVE data-model matrix and declare that target's pointer/long sizes
    // truthfully -- the same lowering-contract discipline R12 applies to ISA/packet lowering,
    // extended to the call ABI. Mirrors the oracle's verify_abi_contract; vacuous when no
    // contract op is present (the whole existing corpus).
    {
      struct DM { const char *name; uint32_t ptr; uint32_t lng; };
      static const DM kMatrix[] = {{"x86_64-linux", 8, 8},   {"aarch64-linux", 8, 8},
                                   {"riscv64-linux", 8, 8},  {"x86_64-windows", 8, 4},
                                   {"i386-linux", 4, 4}};
      root->walk([&](AbiContractOp a) {
        const DM *hit = nullptr;
        for (const DM &d : kMatrix)
          if (a.getTarget() == d.name)
            hit = &d;
        if (!hit) {
          a.emitError("R12: abi_contract ")
              << a.getSymName() << " names an unknown target '" << a.getTarget()
              << "' (the normative matrix: x86_64-linux, aarch64-linux, riscv64-linux, "
                 "x86_64-windows, i386-linux)";
          ok = false;
          return;
        }
        if (a.getPointerSize() != hit->ptr || a.getLongSize() != hit->lng) {
          a.emitError("R12: abi_contract ")
              << a.getSymName() << " declares pointer_size=" << a.getPointerSize()
              << "/long_size=" << a.getLongSize() << " but target '" << a.getTarget()
              << "' has pointer_size=" << hit->ptr << "/long_size=" << hit->lng;
          ok = false;
        }
      });
    }

    // R22/R23: shape-consistency + dtype-compatibility over the gem.* tensor ops (D2 -- the
    // ML/AI-roadmap promotion, the same six-artifact pattern as R19-R21). The op-level verifiers
    // (BCIRDialect.cpp) reject a malformed SINGLE op at parse time; the LAW checks the
    // producer->consumer SEAM no op verifier can see -- the adjacency contract
    // -bcir-fuse-matmul-activation fuses on (R22: the activation epilogue consumes the matmul's
    // full m*n product) and the dtype handover from a conv/attention producer to its activation
    // epilogue (R23). Vacuous for IR with no adjacent gem pair (the non-disturbance invariant).
    // Oracle twins: verify.verify_shape (R22, the gem count seam) + verify.verify_ml_spec
    // (R22/R23, the E3-E6 spec rules).
    root->walk([&](GEMActivationOp act) {
      Operation *prev = act->getPrevNode();
      if (!prev)
        return;
      int64_t extent = 1;
      for (int64_t d : act.getShape()) {
        if (!checkedMulNonnegative(extent, d, extent)) {
          act.emitError("R22: activation shape element count exceeds signed 64-bit range");
          ok = false;
          return;
        }
      }
      if (auto mm = dyn_cast<GEMMatmulOp>(prev)) {
        int64_t matmulExtent;
        if (!checkedMulNonnegative(mm.getM(), mm.getN(), matmulExtent)) {
          mm.emitError("R22: matmul output element count exceeds signed 64-bit range");
          ok = false;
          return;
        }
        if (extent != matmulExtent) {
          act.emitError("R22: gem.activation ")
              << act.getSymName() << " consumes the adjacent gem.matmul @" << mm.getSymName()
              << " but declares a shape extent of " << extent
              << " elements != the matmul's m*n = " << matmulExtent;
          ok = false;
        }
      } else if (auto cv = dyn_cast<GEMConvOp>(prev)) {
        if (cv.getDtype() != act.getDtype()) {
          act.emitError("R23: gem.activation ")
              << act.getSymName() << " consumes the adjacent gem.conv @" << cv.getSymName()
              << " but declares dtype '" << act.getDtype() << "' != the producer's '"
              << cv.getDtype() << "'";
          ok = false;
        }
      } else if (auto at = dyn_cast<GEMAttentionOp>(prev)) {
        if (at.getDtype() != act.getDtype()) {
          act.emitError("R23: gem.activation ")
              << act.getSymName() << " consumes the adjacent gem.attention @" << at.getSymName()
              << " but declares dtype '" << act.getDtype() << "' != the producer's '"
              << at.getDtype() << "'";
          ok = false;
        }
      }
    });

    // R22/R23 over the rung-5 LLM decode chain (open-weight ladder §7.4) -- the SAME adjacency
    // discipline extended to the decoder's stages. embedding -> rmsnorm: the normalizer's rows*dim
    // extent must equal the gather's n_ids*dim (R22) and the dtype must hand over (R23);
    // rope -> attention: the attention's d_k must equal the rope's rotated dim (R22 -- RoPE is
    // applied per head, so a d_k mismatch means the rotation straddled head boundaries) and the
    // dtype must hand over (R23). Vacuous for IR with no adjacent pair (non-disturbance).
    // Oracle twin: frontends/models/decode.py (decoder_layer_reference composes exactly this chain).
    root->walk([&](GEMRMSNormOp rn) {
      Operation *prev = rn->getPrevNode();
      if (!prev)
        return;
      if (auto emb = dyn_cast<GEMEmbeddingOp>(prev)) {
        int64_t normExtent, embeddingExtent;
        if (!checkedMulNonnegative(rn.getRows(), rn.getDim(), normExtent) ||
            !checkedMulNonnegative(emb.getNIds(), emb.getDim(),
                                   embeddingExtent)) {
          rn.emitError("R22: decoder seam extent exceeds signed 64-bit range");
          ok = false;
          return;
        }
        if (normExtent != embeddingExtent) {
          rn.emitError("R22: gem.rmsnorm ")
              << rn.getSymName() << " consumes the adjacent gem.embedding @" << emb.getSymName()
              << " but declares an extent of " << normExtent
              << " elements != the gather's n_ids*dim = " << embeddingExtent;
          ok = false;
        }
        if (emb.getDtype() != rn.getDtype()) {
          rn.emitError("R23: gem.rmsnorm ")
              << rn.getSymName() << " consumes the adjacent gem.embedding @" << emb.getSymName()
              << " but declares dtype '" << rn.getDtype() << "' != the producer's '"
              << emb.getDtype() << "'";
          ok = false;
        }
      }
    });
    root->walk([&](GEMAttentionOp at2) {
      Operation *prev = at2->getPrevNode();
      if (!prev)
        return;
      if (auto rp = dyn_cast<GEMRopeOp>(prev)) {
        if (static_cast<int64_t>(at2.getDK()) != static_cast<int64_t>(rp.getDim())) {
          at2.emitError("R22: gem.attention ")
              << at2.getSymName() << " consumes the adjacent gem.rope @" << rp.getSymName()
              << " but declares d_k = " << at2.getDK() << " != the rotated dim = " << rp.getDim();
          ok = false;
        }
        if (rp.getDtype() != at2.getDtype()) {
          at2.emitError("R23: gem.attention ")
              << at2.getSymName() << " consumes the adjacent gem.rope @" << rp.getSymName()
              << " but declares dtype '" << at2.getDtype() << "' != the producer's '"
              << rp.getDtype() << "'";
          ok = false;
        }
      }
    });

    // R22/R23 over the GQA/KV-cache pair (rung 5's remaining ops): rope -> gqa_attention rides
    // the same d_k handover as plain attention; kv_cache -> gqa_attention must agree on the
    // SHARED head geometry (n_kv_heads and d_k -- a cache sized for different heads is exactly
    // the paged-serving lie the seam exists to catch) and the dtype must hand over (R23).
    // Vacuous with no adjacent pair. Oracle twin: decode.py (KVCache feeds the head loop).
    root->walk([&](GEMGqaAttentionOp gq) {
      Operation *prev = gq->getPrevNode();
      if (!prev)
        return;
      if (auto rp = dyn_cast<GEMRopeOp>(prev)) {
        if (static_cast<int64_t>(gq.getDK()) != static_cast<int64_t>(rp.getDim())) {
          gq.emitError("R22: gem.gqa_attention ")
              << gq.getSymName() << " consumes the adjacent gem.rope @" << rp.getSymName()
              << " but declares d_k = " << gq.getDK() << " != the rotated dim = " << rp.getDim();
          ok = false;
        }
        if (rp.getDtype() != gq.getDtype()) {
          gq.emitError("R23: gem.gqa_attention ")
              << gq.getSymName() << " consumes the adjacent gem.rope @" << rp.getSymName()
              << " but declares dtype '" << gq.getDtype() << "' != the producer's '"
              << rp.getDtype() << "'";
          ok = false;
        }
      }
      if (auto kc = dyn_cast<GEMKvCacheOp>(prev)) {
        if (static_cast<int64_t>(gq.getNKvHeads()) != static_cast<int64_t>(kc.getNKvHeads())) {
          gq.emitError("R22: gem.gqa_attention ")
              << gq.getSymName() << " reads the adjacent gem.kv_cache @" << kc.getSymName()
              << " but declares n_kv_heads = " << gq.getNKvHeads()
              << " != the cache's " << kc.getNKvHeads();
          ok = false;
        }
        if (static_cast<int64_t>(gq.getDK()) != static_cast<int64_t>(kc.getDK())) {
          gq.emitError("R22: gem.gqa_attention ")
              << gq.getSymName() << " reads the adjacent gem.kv_cache @" << kc.getSymName()
              << " but declares d_k = " << gq.getDK() << " != the cache's " << kc.getDK();
          ok = false;
        }
        if (kc.getDtype() != gq.getDtype()) {
          gq.emitError("R23: gem.gqa_attention ")
              << gq.getSymName() << " reads the adjacent gem.kv_cache @" << kc.getSymName()
              << " but declares dtype '" << gq.getDtype() << "' != the cache's '"
              << kc.getDtype() << "'";
          ok = false;
        }
      }
    });

    // R22 over the DEVICE-MANIFEST seam (Phase D hardening, D-R4's tiling law on the law
    // rail): a gem.matmul ADJACENT to a bcir.device_manifest submits tiles the hardware can
    // actually schedule -- every tile extent must be a MULTIPLE of the device's native tile
    // (the max over banks: a tile the widest bank cannot slice fragments somewhere). A 15x15
    // tile against a 16-native device is runtime fragmentation, refused at compile time.
    // Vacuous with no adjacent pair (non-disturbance). Oracle twin: check_strided_view.
    root->walk([&](GEMMatmulOp mm) {
      Operation *prev = mm->getPrevNode();
      if (!prev)
        return;
      if (auto dev = dyn_cast<DeviceManifestOp>(prev)) {
        int64_t native = 1;
        for (int64_t t : dev.getNativeTiles())
          native = std::max(native, t);
        if (native <= 1)
          return;
        const int64_t tiles[3] = {static_cast<int64_t>(mm.getTileM()),
                                  static_cast<int64_t>(mm.getTileN()),
                                  static_cast<int64_t>(mm.getTileK())};
        const char *names[3] = {"tile_m", "tile_n", "tile_k"};
        for (int i = 0; i < 3; ++i)
          if (tiles[i] % native != 0) {
            mm.emitError("R22: gem.matmul ")
                << mm.getSymName() << " submits " << names[i] << " = " << tiles[i]
                << " against the adjacent bcir.device_manifest @" << dev.getSymName()
                << " whose native tile is " << native
                << " (runtime fragmentation is a compile-time refusal)";
            ok = false;
          }
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
      // MOPC objective-support refinement (mapping.py::dropped): under the identity
      // dim-map, every nonzero source objective dimension must remain nonzero in the
      // target (f(Supp(J)) subseteq Supp(J')) unless this contract discharges it. A
      // lowering may rescale/sharpen/fuse a cost dimension but not silently drop one.
      if (auto srcOpt = lc.getSourceSupport()) {
        ArrayRef<int64_t> tgt = lc.getTargetSupport().value_or(ArrayRef<int64_t>{});
        ArrayRef<int64_t> dis = lc.getDischarges().value_or(ArrayRef<int64_t>{});
        auto has = [](ArrayRef<int64_t> s, int64_t d) {
          return std::find(s.begin(), s.end(), d) != s.end();
        };
        for (int64_t d : *srcOpt)
          if (!has(dis, d) && !has(tgt, d)) {
            lc.emitError("R12: lowering contract ")
                << lc.getSymName() << " drops objective dimension " << d
                << " (in Supp(J) but not Supp(J'), with no discharge)";
            ok = false;
          }
      }
    });

    // ------------------------------------------------------------------------
    // R24: ASN.1 / X.690 encoding-rule legality.
    //
    // The law-rail half of the ASN.1 interop profile (LangRef §17,
    // docs/BCIR_ASN1_X690_ABI.md). Oracle twin: bcir/asn1/schema.py + der.py.
    //
    // These are the schema faults that are decidable STATICALLY -- without any value
    // ever being encoded. That is the whole reason they belong here rather than in the
    // oracle: a SET whose components share a tag is undecodable no matter what values
    // it later carries, and X.680 says so about the type, not about an encoding of it.
    //
    // Vacuous for IR with no bcir.asn1.* operation (the non-disturbance invariant that
    // R14-R23 also hold to).
    root->walk([&](Asn1ModuleOp m) {
      // X.690 9.1 makes the indefinite length form MANDATORY for constructed CER
      // encodings, so a CER artifact cannot be byte-stable; BER leaves the spelling to
      // the sender. Neither can carry a digest, and BCIR digests what it emits.
      if (m.getRules() != Asn1Rules::Der) {
        m.emitError("R24: ASN.1 module ")
            << m.getSymName() << " declares encoding rules "
            << stringifyAsn1Rules(m.getRules())
            << "; BCIR emits DER only (X.690 clause 10 + 11)";
        ok = false;
      }
      // X.690 8.19.4 NOTE: only three values are allocated from the root node, and
      // under arcs 0 and 1 the second component is 0..39 (that is what makes the
      // X*40 + Y packing invertible).
      ArrayRef<int64_t> oid = m.getOid();
      if (oid.size() < 2) {
        m.emitError("R24: ASN.1 module ")
            << m.getSymName() << " object identifier needs at least two components";
        ok = false;
      } else {
        if (oid[0] < 0 || oid[0] > 2) {
          m.emitError("R24: ASN.1 module ")
              << m.getSymName() << " object identifier root arc " << oid[0]
              << " is not 0, 1 or 2 (X.690 8.19.4)";
          ok = false;
        } else if (oid[0] < 2 && (oid[1] < 0 || oid[1] >= 40)) {
          m.emitError("R24: ASN.1 module ")
              << m.getSymName() << " object identifier second arc " << oid[1]
              << " must be 0..39 under root arc " << oid[0] << " (X.690 8.19.4)";
          ok = false;
        }
        for (int64_t arc : oid)
          if (arc < 0) {
            m.emitError("R24: ASN.1 module ")
                << m.getSymName() << " object identifier arc " << arc
                << " is negative";
            ok = false;
            break;
          }
      }
    });

    root->walk([&](Asn1TypeOp t) {
      StringRef kind = t.getKind();
      bool isPrimitive = kind == "primitive";
      bool isOf = kind == "sequence_of" || kind == "set_of";
      bool isStructured = kind == "sequence" || kind == "set" || kind == "choice";
      if (!isPrimitive && !isOf && !isStructured) {
        t.emitError("R24: ASN.1 type ")
            << t.getSymName() << " has unknown kind '" << kind
            << "' (primitive | sequence | sequence_of | set | set_of | choice)";
        ok = false;
        return;
      }
      // A universal tag number is meaningful for exactly one kind, both ways round:
      // a primitive without one is unencodable, and a constructor with one is naming
      // a tag it does not own.
      if (isPrimitive && !t.getUniversal()) {
        t.emitError("R24: ASN.1 type ")
            << t.getSymName() << " is primitive but names no universal tag number";
        ok = false;
      }
      if (!isPrimitive && t.getUniversal()) {
        t.emitError("R24: ASN.1 type ")
            << t.getSymName() << " has kind '" << kind
            << "' but names a universal tag number";
        ok = false;
      }
      if (t.getUniversal()) {
        int64_t u = *t.getUniversal();
        // X.680 Table 1: 0 is the encoding rules' own (end-of-contents), 15 is
        // reserved for future editions, and 37+ for addenda. A conforming sender
        // never emits one, so accepting one would silently admit garbage.
        if (u < 1 || u == 15 || u > 36) {
          t.emitError("R24: ASN.1 type ")
              << t.getSymName() << " names reserved universal tag number " << u
              << " (X.680 Table 1: 0, 15 and 37+ are reserved)";
          ok = false;
        }
      }
      if (isOf && !t.getElement()) {
        t.emitError("R24: ASN.1 type ")
            << t.getSymName() << " has kind '" << kind << "' but names no element type";
        ok = false;
      }
      if (!isOf && t.getElement()) {
        t.emitError("R24: ASN.1 type ")
            << t.getSymName() << " has kind '" << kind
            << "' but names an element type";
        ok = false;
      }

      // Component-level laws, and the distinct-tag rule that is the reason this law
      // exists at all.
      llvm::DenseMap<int64_t, StringRef> byTag;
      bool sawUntagged = false;
      t.walk([&](Asn1ComponentOp c) {
        // X.680 25.5: OPTIONAL and DEFAULT are alternatives -- a DEFAULT already makes
        // the component omissible, so carrying both is a contradiction, not a
        // reinforcement.
        if (c.getOptional() && c.getHasDefault()) {
          c.emitError("R24: ASN.1 component ")
              << c.getName() << " is both OPTIONAL and DEFAULT (X.680 25.5)";
          ok = false;
        }
        // X.690 11.5 requires a DER encoder to omit a component equal to its DEFAULT.
        // It cannot do that without the value, so declaring DEFAULT without one makes
        // the type unencodable under DER.
        if (c.getHasDefault() && !c.getDefaultValue()) {
          c.emitError("R24: ASN.1 component ")
              << c.getName()
              << " declares DEFAULT but carries no value; X.690 11.5 requires the "
                 "encoder to compare against it";
          ok = false;
        }
        if (!c.getHasDefault() && c.getDefaultValue()) {
          c.emitError("R24: ASN.1 component ")
              << c.getName() << " carries a DEFAULT value but is not DEFAULT";
          ok = false;
        }
        if (c.getTagging() && !c.getTag()) {
          c.emitError("R24: ASN.1 component ")
              << c.getName() << " states a tagging mode but carries no tag";
          ok = false;
        }
        if (c.getTag()) {
          int64_t tag = *c.getTag();
          if (tag < 0) {
            c.emitError("R24: ASN.1 component ")
                << c.getName() << " has negative tag number " << tag;
            ok = false;
          }
          // X.680 24.4 (SEQUENCE) / 25.3 (SET) / 29.3 (CHOICE): the tags of the
          // components shall be distinct. Without that a decoder cannot tell which
          // component it is looking at -- the type is undecodable as written, for
          // every value it could ever hold.
          auto [it, inserted] = byTag.try_emplace(tag, c.getName());
          if (!inserted) {
            c.emitError("R24: ASN.1 type ")
                << t.getSymName() << " components " << it->second << " and "
                << c.getName() << " share tag [" << tag
                << "] (X.680 24.4/25.3/29.3: component tags shall be distinct)";
            ok = false;
          }
        } else {
          sawUntagged = true;
        }
      });
      // A SET is order-free on the wire (X.690 8.11.2), so its components are told
      // apart by tag alone; an untagged one among tagged siblings is ambiguous in a
      // way the SEQUENCE case (which has position to fall back on) is not.
      if (kind == "set" && sawUntagged && !byTag.empty()) {
        t.emitError("R24: ASN.1 SET ")
            << t.getSymName()
            << " mixes tagged and untagged components; a set is order-free on the "
               "wire (X.690 8.11.2) so every component needs a distinct tag";
        ok = false;
      }
    });

    root->walk([&](Asn1EncodeOp e) {
      if (e.getRules() != Asn1Rules::Der) {
        e.emitError("R24: ASN.1 encode ")
            << e.getSymName() << " declares encoding rules "
            << stringifyAsn1Rules(e.getRules())
            << "; BCIR emits DER only (X.690 clause 10 + 11)";
        ok = false;
      }
    });

    root->walk([&](Asn1DecodeOp d) {
      // Decoding is the permissive half by design -- `ber` here is correct, not a
      // defect. Only the contradiction is a fault.
      if (d.getStrictDer() && d.getRules() == Asn1Rules::Ber) {
        d.emitError("R24: ASN.1 decode ")
            << d.getSymName()
            << " is marked strict_der but declares it accepts BER";
        ok = false;
      }
    });

    root->walk([&](Asn1ProjectionOp p) {
      // A projection that replaced a frozen format would invalidate every digest and
      // provenance manifest taken over the native octets, so the IR refuses to
      // express one at all.
      if (!p.getAdditive()) {
        p.emitError("R24: ASN.1 projection ")
            << p.getSymName() << " of " << p.getNative()
            << " is not marked additive; an ASN.1 projection is a SECOND transfer "
               "syntax, never a replacement for a frozen wire format";
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
