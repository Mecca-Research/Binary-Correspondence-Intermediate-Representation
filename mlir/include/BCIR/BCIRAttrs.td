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

// --- Port-I/O direction (ASM2, SEG8.2): x86 port-mapped I/O edge direction ---
// `in` is a READ (the `in{b,w,l}` instruction: port -> accumulator, one result); `out` is a void
// WRITE (the `out{b,w,l}` instruction: accumulator -> port, no result). The law-rail twin of the
// cfront `c.portio.in.*` / `c.portio.out.*` claim suffix (bcir/frontends/cfront/lower.py::_portio).
def BCIR_PortDir : BCIR_Enum<"PortDir", "x86 port-mapped I/O direction (ASM2)", [
  I32EnumAttrCase<"In", 0, "in">, I32EnumAttrCase<"Out", 1, "out">]>;
def BCIR_PortDirAttr : BCIR_EnumAttr<BCIR_PortDir, "port_dir">;

// --- x86 control register (boot/CPU-state asm edge, D1.3) ---
// The named x86-64 control registers; the enum value is the register NUMBER (CR2/CR3/CR4 are 2/3/4;
// CR8 is the long-mode task-priority register). The lowering renders the keyword `crN` into the
// `mov %crN, $0` / `mov $0, %crN` inline-asm template.
def BCIR_CtrlReg : BCIR_Enum<"CtrlReg", "x86-64 control register", [
  I32EnumAttrCase<"CR0", 0, "cr0">, I32EnumAttrCase<"CR2", 2, "cr2">,
  I32EnumAttrCase<"CR3", 3, "cr3">, I32EnumAttrCase<"CR4", 4, "cr4">,
  I32EnumAttrCase<"CR8", 8, "cr8">]>;
def BCIR_CtrlRegAttr : BCIR_EnumAttr<BCIR_CtrlReg, "ctrl_reg">;

// --- x86 descriptor table (boot/interrupt asm edge) ---
// The two architectural descriptor-table registers accepted by lgdt/lidt. The task register is
// deliberately a separate typed op because ltr consumes a segment selector, not a descriptor pointer.
def BCIR_DescriptorTable : BCIR_Enum<"DescriptorTable", "x86 descriptor-table register", [
  I32EnumAttrCase<"GDT", 0, "gdt">, I32EnumAttrCase<"IDT", 1, "idt">]>;
def BCIR_DescriptorTableAttr : BCIR_EnumAttr<BCIR_DescriptorTable, "descriptor_table">;

// --- Precision contract ---
def BCIR_PrecisionAttr : BCIR_Attr<"Precision", "precision"> {
  let summary = "Numerical precision and tolerance contract";
  let parameters = (ins "::mlir::Type":$elem, "bool":$exact, "int64_t":$toleranceQ16);
  let assemblyFormat = "`<` $elem `,` `exact` `=` $exact `,` `tol` `=` $toleranceQ16 `>`";
}

// --- Timing contract (R19/R20, §5.11 RTL/synchronous-timing) ---
// OPTIONAL on a claim; mirrors bcir/model/graph.py::Timing field-for-field. A claim with no
// timing carries none and the timing laws are vacuous over it (the non-disturbance invariant,
// exactly like R14-R17 are vacuous for the scalar subset). Architecture-agnostic: the channel/
// lowering layer interprets the fields per target (CPU microarchitectural cost vs FPGA/ASIC RT).
def BCIR_TimingAttr : BCIR_Attr<"Timing", "timing"> {
  let summary = "RTL/synchronous-timing metadata (laws R19/R20)";
  let parameters = (ins
    StringRefParameter<"clock net / frequency domain ('' = implicit default)">:$clock_domain,
    "int64_t":$latency_cycles,
    "int64_t":$min_throughput_q16,
    "bool":$critical_path,
    StringRefParameter<"power / voltage domain ('' = implicit default)">:$power_domain,
    StringRefParameter<"'' | synchronous | asynchronous | mixed">:$sync_type,
    "int64_t":$clock_frequency_mhz,
    "int64_t":$setup_hold_margin);
  let assemblyFormat = "`<` struct(params) `>`";
}

// --- Lifetime annotation (R21, §5.12 naked-pointer safety) ---
// OPTIONAL on a claim; mirrors bcir/model/graph.py::Lifetime. A claim may ALLOCATE the resources
// it writes (event = "alloc") or FREE the resources it reads (event = "free", the free(p) shape);
// any other claim is an implicit "use". `epoch` distinguishes a re-allocation. Vacuous when absent.
def BCIR_LifetimeAttr : BCIR_Attr<"Lifetime", "lifetime"> {
  let summary = "Pointer-lifetime annotation (law R21: use-after-free / double-free)";
  let parameters = (ins
    StringRefParameter<"lifetime event: use | alloc | free">:$event,
    "int64_t":$epoch);
  let assemblyFormat = "`<` struct(params) `>`";
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

// --- ASN.1 / X.690 (Rec. ITU-T X.690 02/2021 | ISO/IEC 8825-1:2021) ---
//
// Integer values are NORMATIVE and MUST match bcir/asn1/ exactly (docs/PARITY.md):
// BCIR_Asn1Class mirrors `tags.TagClass` and the encoding of X.690 Table 1 (the two
// high bits of an identifier octet), so a class attribute and a wire octet agree by
// construction rather than by a lookup table.
def BCIR_Asn1Class : BCIR_Enum<"Asn1Class", "ASN.1 tag class (X.690 Table 1)", [
  I32EnumAttrCase<"Universal", 0, "universal">,
  I32EnumAttrCase<"Application", 1, "application">,
  I32EnumAttrCase<"Context", 2, "context">,
  I32EnumAttrCase<"Private", 3, "private">]>;
def BCIR_Asn1ClassAttr : BCIR_EnumAttr<BCIR_Asn1Class, "asn1_class">;

// The transfer syntax an operation emits or accepts.
//
// ONE ENUM, NOT A (FAMILY, PROFILE) PAIR OF ATTRIBUTES. A transfer syntax is what a
// peer either speaks or does not; "family" and "profile" are a decomposition of that
// one identity, useful for stating laws, not two independently settable facts. Storing
// both would create a pair that can disagree -- an op declaring family X.691 with an
// X.697 profile is expressible and meaningless -- and R24 would then be policing a
// contradiction the IR should not have been able to write down. The decomposition is
// DERIVED instead (`asn1FamilyOf`, `isCanonicalAsn1Rules`, `asn1RulesAreAligned` in
// BCIRDialect.cpp), so there is exactly one source of truth and the predicates the laws
// are written over cannot drift from it.
//
// ADDITIVE. `ber`/`cer`/`der` keep values 0/1/2, so every existing artifact, bytecode
// file and .mlir fixture still parses and still means what it meant. The new cases are
// appended.
//
// WHAT EACH ONE IS FOR. BCIR EMITS only a canonical transfer syntax -- one where the
// octets are a function of the abstract value -- because it digests and replays what it
// emits, and a sender's-option encoding lets a peer pick the digest by picking a
// spelling. Everything else is namable so a peer's syntax can be DESCRIBED and DECODED;
// R24 forbids emitting it. That generalizes the old "der only" rule without weakening
// it: `der` is simply the X.690 member of the canonical set.
//
// `bcir_canonical_jer` is deliberately NOT called "CJER". Rec. ITU-T X.697 (02/2021)
// 42.2 registers exactly ONE object identifier for JER and defines no canonical variant
// at all, so this names BCIR's own profile and has no registered OID -- unlike
// CANONICAL-PER and COER, which do. The oracle's `selection.py` records the same
// asymmetry, and test_asn1_law_parity pins the two lists against each other.
def BCIR_Asn1Rules : BCIR_Enum<"Asn1Rules", "ASN.1 transfer syntax", [
  // X.690 (02/2021) | ISO/IEC 8825-1
  I32EnumAttrCase<"Ber", 0, "ber">,
  I32EnumAttrCase<"Cer", 1, "cer">,
  I32EnumAttrCase<"Der", 2, "der">,
  // X.691 (02/2021) | ISO/IEC 8825-2. Alignment is part of the syntax's identity, not a
  // modifier on it: an aligned and an unaligned encoder produce different octets for the
  // same value, so a peer must be told which one, and folding it into the enum makes
  // "aligned" underivable-but-unset impossible to express.
  I32EnumAttrCase<"BasicPerAligned", 3, "basic_per_aligned">,
  I32EnumAttrCase<"BasicPerUnaligned", 4, "basic_per_unaligned">,
  I32EnumAttrCase<"CanonicalPerAligned", 5, "canonical_per_aligned">,
  I32EnumAttrCase<"CanonicalPerUnaligned", 6, "canonical_per_unaligned">,
  // X.696 (02/2021) | ISO/IEC 8825-7
  I32EnumAttrCase<"Oer", 7, "oer">,
  I32EnumAttrCase<"Coer", 8, "coer">,
  // X.693 (02/2021) | ISO/IEC 8825-4
  I32EnumAttrCase<"Xer", 9, "xer">,
  I32EnumAttrCase<"Cxer", 10, "cxer">,
  // X.697 (02/2021) | ISO/IEC 8825-8
  I32EnumAttrCase<"Jer", 11, "jer">,
  I32EnumAttrCase<"BcirCanonicalJer", 12, "bcir_canonical_jer">]>;
def BCIR_Asn1RulesAttr : BCIR_EnumAttr<BCIR_Asn1Rules, "asn1_rules">;

// X.690 8.14.3 (explicit: the tag wraps a complete base encoding, always constructed)
// versus 8.14.4 (implicit: the tag REPLACES the base tag and the form is inherited).
def BCIR_Asn1Tagging : BCIR_Enum<"Asn1Tagging", "ASN.1 tagging mode (X.690 8.14)", [
  I32EnumAttrCase<"Implicit", 0, "implicit">,
  I32EnumAttrCase<"Explicit", 1, "explicit">]>;
def BCIR_Asn1TaggingAttr : BCIR_EnumAttr<BCIR_Asn1Tagging, "asn1_tagging">;

// One ASN.1 tag: the class, the number, and whether THIS ENCODING is constructed
// (identifier octet bit 6, X.690 8.1.2.5). `constructed` describes the encoding, not
// the tag's identity, which is why the DER set ordering of X.690 10.3 sorts on
// (class, number) alone.
def BCIR_Asn1TagAttr : BCIR_Attr<"Asn1Tag", "asn1_tag"> {
  let summary = "ASN.1 tag: class, number, and encoding form (X.690 8.1.2)";
  let parameters = (ins
    "::bcir::Asn1Class":$tag_class,
    "int64_t":$number,
    "bool":$constructed);
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // BCIR_ATTRS
