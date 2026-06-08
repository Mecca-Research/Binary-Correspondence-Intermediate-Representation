//===- BCIRAttrs.td - BCIR enums, cost vector, contracts ------*- tablegen -*-===//
//
// Enums are dialect EnumAttrs so they print/parse as `#bcir.<mnemonic><value>`.
// Integer values MUST match bcir/model/lanes.py + bcir/kbcir/cost.py (docs/PARITY.md).
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_ATTRS
#define BCIR_ATTRS

include "BCIR/BCIRDialect.td"
include "mlir/IR/EnumAttr.td"

class BCIR_Enum<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::bcir";
  let genSpecializedAttr = 0;
}

class BCIR_EnumAttr<EnumAttrInfo info, string mnemonic>
    : EnumAttr<BCIR_Dialect, info, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

// --- Lane ---
def BCIR_Lane : BCIR_Enum<"Lane", "BCIR execution-geometry lane", [
  I32EnumAttrCase<"U", 0, "u">, I32EnumAttrCase<"UX", 1, "ux">, I32EnumAttrCase<"T", 2, "t">,
  I32EnumAttrCase<"GGG", 3, "ggg">, I32EnumAttrCase<"A", 4, "a">, I32EnumAttrCase<"H", 5, "h">]>;
def BCIR_LaneAttr : BCIR_EnumAttr<BCIR_Lane, "lane">;

// --- StrideClass ---
def BCIR_StrideClass : BCIR_Enum<"StrideClass", "BCIR access-pattern shape", [
  I32EnumAttrCase<"Scalar", 0, "scalar">, I32EnumAttrCase<"Unit", 1, "unit">,
  I32EnumAttrCase<"Strided", 2, "strided">, I32EnumAttrCase<"Cacheline", 3, "cacheline">,
  I32EnumAttrCase<"Tile", 4, "tile">, I32EnumAttrCase<"Random", 5, "random">]>;
def BCIR_StrideClassAttr : BCIR_EnumAttr<BCIR_StrideClass, "stride_class">;

// --- Domain ---
def BCIR_Domain : BCIR_Enum<"Domain", "BCIR memory domain", [
  I32EnumAttrCase<"RAM", 0, "ram">, I32EnumAttrCase<"VRAM", 1, "vram">, I32EnumAttrCase<"NVM", 2, "nvm">,
  I32EnumAttrCase<"MMIO", 3, "mmio">, I32EnumAttrCase<"CXL", 4, "cxl">, I32EnumAttrCase<"HBM", 5, "hbm">]>;
def BCIR_DomainAttr : BCIR_EnumAttr<BCIR_Domain, "domain">;

// --- Hazard / verify / bounds ---
def BCIR_HazardMode : BCIR_Enum<"HazardMode", "BCIR hazard mode", [
  I32EnumAttrCase<"Unique", 0, "unique">, I32EnumAttrCase<"Atomic", 1, "atomic">,
  I32EnumAttrCase<"Barriered", 2, "barriered">]>;
def BCIR_HazardModeAttr : BCIR_EnumAttr<BCIR_HazardMode, "hazard">;

def BCIR_Verify : BCIR_Enum<"Verify", "BCIR verification contract", [
  I32EnumAttrCase<"None", 0, "none">, I32EnumAttrCase<"Bounds", 1, "bounds">,
  I32EnumAttrCase<"Exact", 2, "exact">, I32EnumAttrCase<"Hash", 3, "hash">]>;
def BCIR_VerifyAttr : BCIR_EnumAttr<BCIR_Verify, "verify">;

def BCIR_Bounds : BCIR_Enum<"Bounds", "BCIR bounds mode (law R7)", [
  I32EnumAttrCase<"Strict", 0, "strict">, I32EnumAttrCase<"Masked", 1, "masked">,
  I32EnumAttrCase<"AssumedSafe", 2, "assumed_safe">]>;
def BCIR_BoundsAttr : BCIR_EnumAttr<BCIR_Bounds, "bounds">;

// --- Layout ---
def BCIR_Layout : BCIR_Enum<"Layout", "BCIR data layout contract", [
  I32EnumAttrCase<"SoA", 0, "soa">, I32EnumAttrCase<"AoS", 1, "aos">,
  I32EnumAttrCase<"AoSoA", 2, "aosoa">, I32EnumAttrCase<"Blocked", 3, "blocked">]>;
def BCIR_LayoutAttr : BCIR_EnumAttr<BCIR_Layout, "layout">;

// --- K_BCIR policy / semiring / cost class ---
def BCIR_PolicyMode : BCIR_Enum<"PolicyMode", "K_BCIR scalarization policy", [
  I32EnumAttrCase<"Latency", 0, "latency">, I32EnumAttrCase<"Throughput", 1, "throughput">,
  I32EnumAttrCase<"Energy", 2, "energy">, I32EnumAttrCase<"Safe", 3, "safe">,
  I32EnumAttrCase<"Research", 4, "research">]>;
def BCIR_PolicyModeAttr : BCIR_EnumAttr<BCIR_PolicyMode, "policy_mode">;

def BCIR_Semiring : BCIR_Enum<"Semiring", "Path-composition semiring", [
  I32EnumAttrCase<"MinPlus", 0, "min_plus">, I32EnumAttrCase<"MaxPlus", 1, "max_plus">]>;
def BCIR_SemiringAttr : BCIR_EnumAttr<BCIR_Semiring, "semiring">;

def BCIR_CostClass : BCIR_Enum<"CostClass", "Dominant cost axis hint", [
  I32EnumAttrCase<"Latency", 0, "latency">, I32EnumAttrCase<"Bandwidth", 1, "bandwidth">,
  I32EnumAttrCase<"Compute", 2, "compute">]>;
def BCIR_CostClassAttr : BCIR_EnumAttr<BCIR_CostClass, "cost_class">;

// --- Memory hierarchy (CT1: HBM / HAM / CXL) ---
def BCIR_MemTier : BCIR_Enum<"MemTier", "Memory hierarchy tier", [
  I32EnumAttrCase<"L1", 0, "l1">, I32EnumAttrCase<"L2", 1, "l2">, I32EnumAttrCase<"L3", 2, "l3">,
  I32EnumAttrCase<"DRAM", 3, "dram">, I32EnumAttrCase<"HBM", 4, "hbm">,
  I32EnumAttrCase<"CXL", 5, "cxl">, I32EnumAttrCase<"SSD", 6, "ssd">]>;
def BCIR_MemTierAttr : BCIR_EnumAttr<BCIR_MemTier, "mem_tier">;

// --- Addressing model: flat O(1) vs Hierarchical Access Memory O(log n) ---
def BCIR_Access : BCIR_Enum<"Access", "Resource addressing model", [
  I32EnumAttrCase<"Flat", 0, "flat">, I32EnumAttrCase<"HAM", 1, "ham">]>;
def BCIR_AccessAttr : BCIR_EnumAttr<BCIR_Access, "access">;

// --- Memory ordering (Phase 8): the BCIR -> LLVM atomic-ordering memory model ---
// Values/mnemonics mirror LLVM's AtomicOrdering and bcir/lower/memory_model.py.
def BCIR_MemOrdering : BCIR_Enum<"MemOrdering", "Memory ordering (maps to LLVM AtomicOrdering)", [
  I32EnumAttrCase<"Unordered", 0, "unordered">, I32EnumAttrCase<"Monotonic", 1, "monotonic">,
  I32EnumAttrCase<"Acquire", 2, "acquire">, I32EnumAttrCase<"Release", 3, "release">,
  I32EnumAttrCase<"AcqRel", 4, "acq_rel">, I32EnumAttrCase<"SeqCst", 5, "seq_cst">]>;
def BCIR_MemOrderingAttr : BCIR_EnumAttr<BCIR_MemOrdering, "mem_ordering">;

// --- Precision contract ---
def BCIR_PrecisionAttr : BCIR_Attr<"Precision", "precision"> {
  let summary = "Numerical precision and tolerance contract";
  let parameters = (ins "::mlir::Type":$elem, "bool":$exact, "int64_t":$toleranceQ16);
  let assemblyFormat = "`<` $elem `,` `exact` `=` $exact `,` `tol` `=` $toleranceQ16 `>`";
}

// --- 12-d K_BCIR cost vector (Q16 fixed point; LangRef decimals are normalized sugar) ---
def BCIR_CostVectorAttr : BCIR_Attr<"CostVector", "costvec"> {
  let summary = "12-dimensional K_BCIR cost vector, Q16 fixed point";
  let parameters = (ins
    "int64_t":$compute, "int64_t":$memory, "int64_t":$fabric, "int64_t":$sync,
    "int64_t":$compile, "int64_t":$thermal, "int64_t":$power, "int64_t":$reliability,
    "int64_t":$security, "int64_t":$accuracy, "int64_t":$contention, "int64_t":$verification);
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // BCIR_ATTRS
