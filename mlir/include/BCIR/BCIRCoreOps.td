//===- BCIRCoreOps.td - core BCIR ops (BCIR-0/1/2) ----------*- tablegen -*-===//
//
// Modules own registries; registries own resources; claims declare contracts;
// phases own ordering. Mirrors bcir/model + bcir/verify. A resource is a pure
// symbol (addressed by @name; MLIR 22 forbids a Symbol op from having an SSA
// result, so the old !bcir.resource handle is gone). The claim body holds the
// realization region directly (no bcir.region wrapper); module/registry default
// string attrs are properly quoted.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_CORE_OPS
#define BCIR_CORE_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRTypes.td"
include "BCIR/BCIRAttrs.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Contract bundle carried by memory/compute claims (LangRef Sec. 5).
defvar BCIR_ClaimAttrs = (ins
  BCIR_LaneAttr:$lane,
  BCIR_StrideClassAttr:$stride_class,
  DefaultValuedAttr<I32Attr, "1">:$stride_k,
  BCIR_DomainAttr:$domain,
  BCIR_HazardModeAttr:$hazard,
  BCIR_VerifyAttr:$verify,
  BCIR_BoundsAttr:$bounds,
  OptionalAttr<BCIR_PrecisionAttr>:$precision,
  OptionalAttr<BCIR_CostClassAttr>:$cost_class,
  // The model.Opcode integer (model/opcodes.py); carried so -bcir-verify can recompute the
  // provenance hash_module (R13) from the IR. Defaulted (NOP) so existing IR parses.
  DefaultValuedAttr<I32Attr, "0">:$opcode,
  // Dynamic shape (compose): `count` is a STATIC UPPER BOUND, so the plan priced at it is a
  // worst-case latency bound that holds for any actual size <= the bound (compose.dynamic /
  // plan_holds_for). Defaulted false (an exact static count) so existing IR parses.
  DefaultValuedAttr<BoolAttr, "false">:$dynamic,
  // OPTIONAL timing (§5.11, laws R19/R20) + lifetime (§5.12, law R21) metadata, mirroring
  // model.graph Timing/Lifetime. Absent for the entire scalar/C subset, so R19/R20/R21 are
  // vacuous over it (the non-disturbance invariant -- existing IR parses and verifies unchanged).
  OptionalAttr<BCIR_TimingAttr>:$timing,
  OptionalAttr<BCIR_LifetimeAttr>:$lifetime);

def BCIR_ModuleOp : BCIR_Op<"module",
    [IsolatedFromAbove, NoTerminator, SymbolTable, Symbol]> {
  let summary = "A BCIR module: registries -> claims -> phases -> plan -> stream";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DefaultValuedAttr<StrAttr, "\"0.1\"">:$version,
    DefaultValuedAttr<StrAttr, "\"registry-first\"">:$target_model,
    DefaultValuedAttr<StrAttr, "\"gem\"">:$execution_model,
    DefaultValuedAttr<StrAttr, "\"k_bcir\"">:$cost_model,
    // The goal-graph cacheline/align (model.Module); carried for the R13 hash_module
    // recompute. Defaulted to the model defaults so existing IR is unchanged.
    DefaultValuedAttr<I32Attr, "64">:$cacheline,
    DefaultValuedAttr<I32Attr, "64">:$align);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_RegistryOp : BCIR_Op<"registry", [NoTerminator, Symbol, SymbolTable]> {
  let summary = "A registry namespace of resources";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_ResourceOp : BCIR_Op<"resource", [Symbol]> {
  let summary = "Declare a registry-governed resource (RID-addressed, not a pointer)";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I32Attr:$rid,
    BCIR_DomainAttr:$domain_kind,
    DenseI64ArrayAttr:$shape,
    BCIR_LayoutAttr:$layout,
    DefaultValuedAttr<I32Attr, "64">:$align,
    OptionalAttr<BCIR_AccessAttr>:$access,
    DefaultValuedAttr<I32Attr, "0">:$priority,
    DefaultValuedAttr<I64Attr, "0">:$map_gen,
    DefaultValuedAttr<I64Attr, "0">:$data_gen,
    // Allocator placement tier (append-only; mirrors kbcir.allocator.Placement).
    // R16 (lower-to-llvm): an on-chip placement must fit -- a resource placed in L1
    // must be <= 64 KiB, in L2 <= 4 MiB (static caps; size = product(shape) * 4 B).
    OptionalAttr<BCIR_MemTierAttr>:$placement);
  // MLIR 22's Symbol verifier forbids a symbol op from producing an SSA result; a
  // resource is addressed by its registry symbol (@A) -- the old !bcir.resource handle
  // was vestigial (never consumed), so this op is now a pure symbol.
  let assemblyFormat = "$sym_name attr-dict";
  // Parse-time structural well-formedness (align is a positive power of two; the shape
  // extents are positive). Cheaper + earlier than -bcir-verify (which carries the
  // cross-op R-laws); runs on every parse/builder, not only when the pass is invoked.
  let hasVerifier = 1;
}

def BCIR_PhaseOp : BCIR_Op<"phase", [Symbol]> {
  let summary = "A scheduling phase with explicit dependency edges (LangRef Sec. 6)";
  let arguments = (ins SymbolNameAttr:$sym_name, I32Attr:$id, FlatSymbolRefArrayAttr:$deps);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_ClaimOp : BCIR_Op<"claim", [Symbol, NoTerminator]> {
  let summary = "A declarative transition: op + resources + contract + phase";
  let arguments = !con(
    (ins SymbolNameAttr:$sym_name,
         I32Attr:$claim_id,
         FlatSymbolRefAttr:$phase,
         StrAttr:$op,
         FlatSymbolRefArrayAttr:$reads,
         FlatSymbolRefArrayAttr:$writes,
         DefaultValuedAttr<I64Attr, "0">:$offset,
         DefaultValuedAttr<I64Attr, "1">:$count),
    BCIR_ClaimAttrs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  // Parse-time well-formedness: the iteration extent is non-negative and the stride
  // multiplier is positive.
  let hasVerifier = 1;
}

def BCIR_IndexRangeOp : BCIR_Op<"index_range", [Pure]> {
  let summary = "An affine iteration domain [lo, hi) step s";
  let arguments = (ins I64Attr:$lo, I64Attr:$hi, DefaultValuedAttr<I64Attr, "1">:$step);
  let results = (outs Index:$iv);
  let assemblyFormat = "$lo `to` $hi `step` $step attr-dict";
}

def BCIR_LoadOp : BCIR_Op<"load"> {
  let summary = "Load through (RID, index) with a lane/bounds contract";
  let arguments = !con((ins FlatSymbolRefAttr:$src, Index:$index), BCIR_ClaimAttrs);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$src `[` $index `]` attr-dict `:` type($value)";
}

def BCIR_StoreOp : BCIR_Op<"store"> {
  let summary = "Store through (RID, index) with a lane/bounds contract";
  let arguments = !con((ins AnyType:$value, FlatSymbolRefAttr:$dst, Index:$index), BCIR_ClaimAttrs);
  let assemblyFormat = "$value `,` $dst `[` $index `]` attr-dict `:` type($value)";
}

def BCIR_ComputeOp : BCIR_Op<"compute", [Pure]> {
  let summary = "Pure elementwise computation (kind = fadd/fsub/fmul/...)";
  let arguments = (ins StrAttr:$kind, Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$kind `(` $operands `)` attr-dict `:` functional-type($operands, $result)";
}

def BCIR_BarrierOp : BCIR_Op<"barrier"> {
  let summary = "Memory or execution fence at a given scope (H lane)";
  // `ordering` (Phase 8 memory model) maps to the LLVM fence ordering; default seq_cst.
  let arguments = (ins I32Attr:$scope, OptionalAttr<BCIR_MemOrderingAttr>:$ordering);
  let assemblyFormat = "$scope attr-dict";
}

def BCIR_AsmOp : BCIR_Op<"asm"> {
  let summary = "ISA-neutral verbatim inline assembly (ASM1, an opaque trusted effect edge)";
  // The law-rail twin of the cfront `c.asm:` / `c.asm.volatile:` claim
  // (bcir/frontends/cfront/lower.py::_asm_stmt + AsmInfo). GNU extended inline asm is a
  // TRUSTED OPAQUE EFFECT EDGE: the assembly template, the per-operand constraints, and the
  // clobbers are carried VERBATIM (ISA-neutral pass-through), the operand SSA values ride the
  // op's variadic operands, and the lowering reconstructs the exact `llvm.inline_asm` (which is
  // LLVM's `call asm` / GCC `__asm__`). The closest sibling op is `bcir.barrier`: a `"memory"`
  // clobber / volatile form is an ordering fence, never reordered/fused across.
  //
  // Fields (mirroring AsmInfo):
  //   * `asm_template`    -- the asm template string, carried VERBATIM (e.g. "mfence", "movl $1, $0").
  //                          (Named `asm_template`, not `template`, since `template` is a C++ keyword the
  //                          ODS-generated accessors would collide with.)
  //   * `is_volatile`     -- a `__asm__ __volatile__` (a side-effecting edge that must not be DCE'd).
  //                          The lowering ALWAYS marks `has_side_effects` (asm is conservatively
  //                          side-effecting); `is_volatile` records the source spelling for round-trip.
  //   * `out_constraints` -- one per result, each already an output constraint (e.g. "=r", "+r").
  //   * `in_constraints`  -- one per input operand (e.g. "r", "i", "Nd").
  //   * `clobbers`        -- the clobber list, BARE (e.g. "memory", "cc", "rax"); the lowering
  //                          renders each as a `~{...}` constraint ENTRY (LLVM has no separate
  //                          clobber field).
  //
  // Operands: `$args` are the outputs-then-inputs in the cfront SSA order (so the first
  // `out_constraints.size()` operands are the output lvalue SSA values, the rest the inputs).
  // Results: `$results` are one per output constraint (the values the asm writes back).
  //
  // The verifier (`hasVerifier`) is the op-level structural well-formedness check (NOT a new
  // globally-numbered R-law): `args.size() == out_constraints.size() + in_constraints.size()`,
  // `results.size() == out_constraints.size()`, and every `out_constraints` entry must itself be an
  // output constraint -- begin with LLVM's write-only `=` or read-write `+`. The last check keeps a
  // stray input spelling (e.g. a bare `"r"` as an output) from passing the count laws and lowering to
  // a result-returning `llvm.inline_asm` whose constraint string has no `=` output -- structurally
  // invalid asm the LLVM IR translator rejects far from the source.
  //
  // LOWERING SCOPE (the first slice, `-convert-bcir-to-llvm`): 0 or 1 result, write-only ("=")
  // outputs. A 0-output asm lowers to a void `llvm.inline_asm`; a 1-output asm lowers to one
  // returning that type. The LLVM `call asm` operand list is the INPUT operands ONLY -- a
  // write-only "=" output is the RESULT, never an asm-call argument -- so the lowering passes
  // `args[out_constraints.size():]` (the inputs) as operands. A multi-output asm (LLVM returns a
  // struct needing extractvalue unpacking) is a follow-on (SEG8.x); the lowering rejects
  // `results.size() > 1` with a clear diagnostic rather than shipping a wrong struct-unpack.
  // (Read-write "+" outputs, which tie an input via a matching constraint, are the same follow-on;
  // the op itself + verifier already model any output arity -- only the LLVM lowering is scoped.)
  let arguments = (ins
    StrAttr:$asm_template,
    UnitAttr:$is_volatile,
    StrArrayAttr:$out_constraints,
    StrArrayAttr:$in_constraints,
    StrArrayAttr:$clobbers,
    Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    (`volatile` $is_volatile^)? $asm_template
    `outs` $out_constraints `ins` $in_constraints `clobbers` $clobbers
    ( `(` $args^ `)` )? attr-dict (`:` functional-type($args, $results)^)?
  }];
  let hasVerifier = 1;
}

def BCIR_PortioOp : BCIR_Op<"portio"> {
  let summary = "x86 port-mapped I/O edge (ASM2, an opaque side-effecting trusted edge)";
  // The law-rail twin of the cfront `c.portio.in.{b,w,l}:` / `c.portio.out.{b,w,l}:` claim
  // (bcir/frontends/cfront/lower.py::_portio + emit.py::_portio_stmt). x86 port-mapped I/O (the
  // `in`/`out` instructions) is a TRUSTED OPAQUE EFFECT EDGE: it is volatile + ordered (barriered),
  // off the legality value-path, and reuses the `llvm.inline_asm` lowering established by `bcir.asm`
  // (SEG8.1). It is x86-ONLY -- port I/O is inherently x86; ARM/RISC-V have no port I/O (only MMIO),
  // matching the cfront claim, which raises an honest diagnostic on a non-x86 target.
  //
  // Fields:
  //   * `direction` -- `in` (a READ: the port is the input, the read value is the result) or `out`
  //                    (a void WRITE: value-then-port are the inputs, no result). Mirrors the cfront
  //                    `c.portio.in.*` / `c.portio.out.*` claim suffix.
  //   * `width`     -- the access width in BITS, one of {8, 16, 32} (verifier-checked). Maps to the
  //                    GCC operand-size modifier b=8 / w=16 / l=32 (the cfront `{b,w,l}` suffix).
  //
  // Operands / results, keyed off `direction` (matching the cfront rd/wr ordering):
  //   * in:  ONE operand `$port`; ONE result (the read value).
  //   * out: TWO operands `$value, $port` (value first, the Linux `out(value, port)` order); ZERO
  //          results.
  //
  // The verifier (`hasVerifier`) is the op-level structural well-formedness check (NOT a new globally-
  // numbered R-law): `width in {8,16,32}`; for `in`, exactly one operand (the port) and one result;
  // for `out`, exactly two operands (value, port) and no result. (When present, the in-result /
  // out-value integer width must match the op width i8/i16/i32 and the port must be an integer.)
  //
  // LOWERING (`-convert-bcir-to-llvm`): selects the x86 template from (direction, width) -- BYTE-
  // IDENTICAL to cfront's `_PORTIO_IN_ASM` / `_PORTIO_OUT_ASM` -- and emits an `llvm.inline_asm` via
  // the SAME generic attribute-list builder as `AsmOpLowering`. in: constraint "=a,Nd", returns the
  // i{width}; out: constraint "a,Nd", no result. `has_side_effects` is ALWAYS true (port I/O is
  // volatile, never DCE'd/reordered -- like the cfront `__volatile__`).
  let arguments = (ins
    BCIR_PortDirAttr:$direction,
    I32Attr:$width,
    Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $direction `width` $width ( ` ` `(` $args^ `)` )? attr-dict
    (`:` functional-type($args, $results)^)?
  }];
  let hasVerifier = 1;
}

#endif // BCIR_CORE_OPS
