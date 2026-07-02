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
  OptionalAttr<BCIR_LifetimeAttr>:$lifetime,
  // §5.14 Phase 2: the VOLATILE qualifier on the contract-carrying claim rail (bcir.claim /
  // bcir.load / bcir.store) -- distinct from bcir.volatile_load/store, which are the LOWERED
  // integer-address MMIO accessors. This bit rides the RID/contract layer so the ORDERING LAW
  // sees volatility structurally: R5 requires an ordered hazard on a volatile claim, and the
  // optimizer fences it exactly like `barriered` (-bcir-bundle never reorders/bundles across
  // it). Mirrors model.graph.Claim.volatile (DIGEST-EXCLUDED from R13, like timing/lifetime);
  // defaulted false so all existing IR parses and hashes unchanged (non-disturbance).
  DefaultValuedAttr<BoolAttr, "false">:$is_volatile);

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
  //   * `asm_template`    -- the asm template string in the user's (cfront/C source) GCC operand syntax
  //                          (e.g. "mfence", "movl %1, %0"). The LOWERING TRANSLATES the GCC operand
  //                          syntax to LLVM-IR `$`-syntax before storing it on `llvm.inline_asm`
  //                          (`%%`->`%`, `%<letter><N>`->`${N:<letter>}`, `%<N>`->`$N`, a literal
  //                          `$`->`$$`; exotic `%[name]`/`%P`/`%=` are REJECTED at lowering), since a
  //                          verbatim GCC `%w1` fails `llc: invalid register name`. (Named `asm_template`,
  //                          not `template`, since `template` is a C++ keyword the ODS-generated
  //                          accessors would collide with.)
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
  // the op itself + verifier already model any output arity -- only the LLVM lowering is scoped. The
  // lowering REJECTS a "+" out-constraint explicitly: a "+" output is read-write, not a "=" result, so
  // it lowers to `llvm.inline_asm ... "+r,r" %x : (i32) -> i32` which opt/llc reject -- "inline asm
  // without outputs must return void".)
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
  // LOWERING (`-convert-bcir-to-llvm`): selects the x86 template from (direction, width) in LLVM-IR
  // operand syntax (VERIFIED to assemble via `llc` to `inb %dx,%al` etc.) -- in: `inb ${1:w}, ${0:b}`
  // / `inw ${1:w}, ${0:w}` / `inl ${1:w}, ${0:k}`; out: `outb ${0:b}, ${1:w}` / `outw ${0:w}, ${1:w}`
  // / `outl ${0:k}, ${1:w}` (accumulator `${0:b/w/k}` = al/ax/eax, port `${1:w}` = the 16-bit dx) --
  // and emits an `llvm.inline_asm` via the SAME generic attribute-list builder as `AsmOpLowering`. in:
  // constraint "={ax},N{dx}", returns the i{width}; out: constraint "{ax},N{dx}", no result. The
  // FULLY-QUALIFIED register names are what LLVM's x86 backend needs: cfront's GCC `%w1`/`=a,Nd`
  // spellings are correct on the C->clang path but `llc` rejects them on this MLIR->LLVM path
  // ("invalid register name" / "couldn't allocate output register"); clang's frontend does the
  // `%b0`->`${0:b}`, `=a`->`={ax}`, `Nd`->`N{dx}` translation, which the frontend-less MLIR rail emits
  // directly. `has_side_effects` is ALWAYS true (port I/O is volatile, never DCE'd/reordered -- like
  // the cfront `__volatile__`).
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

def BCIR_VolatileLoadOp : BCIR_Op<"volatile_load"> {
  let summary = "Ordered volatile load from a device register (first-class MMIO read, H lane)";
  // The law-rail twin of the cfront MMIO read (a `volatile`-qualified register access:
  // bcir/frontends/cfront/lower.py marks the resource `domain=Domain.MMIO, access="volatile"`,
  // and emit.py renders the ordered volatile load `*(volatile T *)(ptr + off)`). Until now the MLIR
  // rail carried MMIO only as the `Domain.MMIO` enum value on a claim/resource, with device access
  // riding the barrier/hazard machinery; this op makes the MMIO ACCESSOR first-class, as it already is
  // in cfront emit. It is the LOWERED accessor -- the post-bounds-check device read -- so it takes the
  // resolved integer register ADDRESS, mirroring the cfront emit layer (`*(volatile T*)(intaddr)`); the
  // RID/extent/bounds contract (R3) lives at the `bcir.claim`/`bcir.load` layer, exactly as in cfront
  // (the claim carries the isolation law; the emit is a raw ordered volatile access).
  //
  //   * `$addr`  -- the device-register address (a signless integer, e.g. an i64 physical address).
  //   * `$value` -- the loaded value (the register's natural width: i8/i16/i32/i64/...).
  //
  // LOWERING (`-convert-bcir-to-llvm`): `llvm.inttoptr $addr` -> `llvm.load volatile` of the (type-
  // converted) result type. `volatile` is ALWAYS set (a device read must never be elided, duplicated,
  // or reordered with other volatile accesses -- the MMIO guarantee, matching the cfront `volatile`).
  //
  // DCE/CSE SAFETY: this op carries NO `Pure`/`NoMemoryEffect` trait by DESIGN -- a device read is
  // side-effecting (re-reading a status register yields fresh values), so MLIR's conservative
  // unknown-effect default keeps it from being DCE'd even with an unused result and from being CSE-merged
  // with a second identical read. DO NOT add `Pure` or `MemoryEffects<[]>` (it would silently make device
  // reads removable/mergeable). The same holds for every asm-edge op (asm/portio/creg).
  //
  // The verifier (`hasVerifier`) constrains the operands the LOWERING relies on: `$value` is a SCALAR
  // hardware-register type (an integer or float -- a vector/index/struct device register is rejected, as
  // the op doc promises a scalar natural width), and `$addr` is at least pointer-width (>= 32 bits) so a
  // too-narrow address is not silently zero-extended into the device pointer.
  let arguments = (ins AnySignlessInteger:$addr);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$addr attr-dict `:` type($addr) `->` type($value)";
  let hasVerifier = 1;
}

def BCIR_VolatileStoreOp : BCIR_Op<"volatile_store"> {
  let summary = "Ordered volatile store to a device register (first-class MMIO write, H lane)";
  // The law-rail twin of the cfront MMIO write (emit.py renders `*(volatile T *)(ptr + off) = value`).
  // The void counterpart of `bcir.volatile_load`: the LOWERED MMIO write accessor, taking the value and
  // the resolved integer register address (the RID/bounds contract is the `bcir.claim` layer, as in
  // cfront). LOWERING: `llvm.inttoptr $addr` -> `llvm.store volatile $value`. `volatile` is ALWAYS set
  // (a device write must never be elided/duplicated/reordered with other volatile accesses).
  //
  //   * `$value` -- the value to write (the register's natural width).
  //   * `$addr`  -- the device-register address (a signless integer).
  //
  // Same DCE/CSE-safety + verifier discipline as `bcir.volatile_load`: no `Pure` trait (a device write is
  // side-effecting), `$value` is a scalar hardware-register type, `$addr` is >= pointer-width.
  let arguments = (ins AnyType:$value, AnySignlessInteger:$addr);
  let assemblyFormat = "$value `,` $addr attr-dict `:` type($value) `,` type($addr)";
  let hasVerifier = 1;
}

def BCIR_AtomicRMWOp : BCIR_Op<"atomic_rmw"> {
  let summary = "First-class atomic read-modify-write (A lane, §5.14 Phase 2)";
  // The law-rail twin of the cfront atomic RMW claims -- `c.atomic.{add,sub,xor}` (the GCC
  // `__atomic_fetch_*` family) and `c.c11atom.{fetch_add,fetch_sub,fetch_xor,exchange}` (C11
  // `<stdatomic.h>`) -- which until now reached MLIR only as opcode-named `bcir.claim` STRING
  // dispatch. This op lifts the RMW to a first-class structure carrying the existing
  // `#bcir.mem_ordering` attr, so the ordering law sees the atomicity structurally (the §5.14
  // Phase 2 contract). Like `bcir.volatile_load`, it is the LOWERED accessor: it takes the
  // resolved integer ADDRESS; the RID/extent/bounds contract stays at the `bcir.claim` layer.
  //
  //   * `$kind`     -- the RMW operation, one of {add, sub, xor, exchange} (verifier-checked;
  //                    a string like ComputeOp's `$kind`, matching the cfront claim suffixes --
  //                    no model-integer enum exists for RMW kinds, so a string avoids inventing
  //                    an unparitied integer encoding).
  //   * `$addr`     -- the resolved object address (signless integer, >= 32 bits).
  //   * `$value`    -- the operand (an INTEGER: the cfront atomic families are integer-typed).
  //   * `$ordering` -- OPTIONAL #bcir.mem_ordering; ABSENT means seq_cst (the C11/`__sync`
  //                    default and the `bcir.barrier` precedent). unordered/monotonic clamp to
  //                    monotonic at the LLVM lowering (an RMW must be at least monotonic).
  //   * `$result`   -- the OLD value (fetch semantics), same type as `$value`.
  //
  // LOWERING (`-convert-bcir-to-llvm`): `llvm.inttoptr $addr` -> `llvm.atomicrmw <kind>` with
  // the mapped ordering. DCE/CSE SAFETY: NO `Pure` trait by design (an atomic RMW is a
  // side-effecting synchronization access -- the same discipline as the volatile/asm-edge ops).
  let arguments = (ins StrAttr:$kind, AnySignlessInteger:$addr, AnyType:$value,
                       OptionalAttr<BCIR_MemOrderingAttr>:$ordering);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $kind $value `,` $addr attr-dict `:` type($value) `,` type($addr) `->` type($result)
  }];
  let hasVerifier = 1;
}

def BCIR_AtomicCASOp : BCIR_Op<"atomic_cas"> {
  let summary = "First-class atomic compare-and-swap (A lane, §5.14 Phase 2)";
  // The law-rail twin of the cfront CAS claims -- `c.cmpxchg.{val,bool}` (`__sync_*_compare_and_swap`)
  // and `c.c11atom.cas_{strong,weak}` (C11 `atomic_compare_exchange_*`) -- lifted from opcode-named
  // string dispatch to a first-class op carrying `#bcir.mem_ordering`. Same LOWERED-accessor stance
  // as `bcir.atomic_rmw` (resolved integer address; the RID contract stays on `bcir.claim`).
  //
  //   * `$addr`     -- the resolved object address (signless integer, >= 32 bits).
  //   * `$expected` / `$desired` -- the comparand and the replacement (INTEGERS, same type).
  //   * `$ordering` -- OPTIONAL success ordering (absent = seq_cst); the failure ordering is
  //                    DERIVED per the LLVM strongest-failure rule (release -> monotonic,
  //                    acq_rel -> acquire, else the success ordering) -- never stronger than
  //                    success, never a release on failure.
  //   * `$weak`     -- a `cas_weak` (spurious failure allowed; the cfront `_weak` suffix).
  //   * `$result`   -- the OLD value (the `c.cmpxchg.val` form; a boolean success form is a
  //                    follow-on comparison in the consumer, exactly as cfront lowers it).
  //
  // LOWERING: `llvm.inttoptr $addr` -> `llvm.cmpxchg` (+ weak flag) -> `llvm.extractvalue [0]`.
  // NO `Pure` trait (side-effecting synchronization access).
  let arguments = (ins AnySignlessInteger:$addr, AnyType:$expected, AnyType:$desired,
                       OptionalAttr<BCIR_MemOrderingAttr>:$ordering, UnitAttr:$weak);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    (`weak` $weak^)? $expected `,` $desired `,` $addr attr-dict
    `:` type($expected) `,` type($desired) `,` type($addr) `->` type($result)
  }];
  let hasVerifier = 1;
}

def BCIR_CRegReadOp : BCIR_Op<"creg_read"> {
  let summary = "Read an x86-64 control register (boot/CPU-state asm edge, D1.3)";
  // A control-register read (CR0/CR2/CR3/CR4/CR8) is a TRUSTED OPAQUE EFFECT EDGE -- there is no C
  // value-semantics for it, so it is the irreducible-assembly tier (BCIR_DRIVER_KERNEL_ROADMAP.md Part
  // II Tier 1), like `bcir.asm`/`bcir.portio`. Unlike those two it has NO cfront dual-rail twin (the
  // cfront rail expresses CR access only as a raw `c.asm` blob), so the template is authored DIRECTLY
  // in LLVM-IR inline-asm syntax (`$0` operand, single-`%` register) -- empirically the exact form
  // clang emits for `__asm__("mov %%cr3, %0" : "=r"(v))`. x86-only.
  //
  //   * `reg`    -- which control register (a `#bcir.ctrl_reg` enum: cr0/cr2/cr3/cr4/cr8).
  //   * result   -- the 64-bit register value (control registers are 64-bit in long mode).
  //
  // LOWERING (`-convert-bcir-to-llvm`): `llvm.inline_asm has_side_effects "mov %<reg>, $0", "=r"`,
  // returning i64. `has_side_effects` is ALWAYS set (a control-register read is not a pure value --
  // CR2 holds the faulting address, CR3 the live page-table base; it must never be reordered/elided).
  let arguments = (ins BCIR_CtrlRegAttr:$reg);
  let results = (outs I64:$value);
  let assemblyFormat = "$reg attr-dict `->` type($value)";
}

def BCIR_CRegWriteOp : BCIR_Op<"creg_write"> {
  let summary = "Write an x86-64 control register (boot/CPU-state asm edge, D1.3)";
  // The void counterpart of `bcir.creg_read`. A control-register WRITE has system-wide effects (writing
  // CR3 reloads the page tables + flushes the TLB; CR0/CR4 toggle paging/protection/feature bits), so it
  // is conservatively side-effecting AND carries a `~{memory}` clobber. x86-only.
  //
  //   * `value`  -- the 64-bit value to load into the register.
  //   * `reg`    -- which control register (a `#bcir.ctrl_reg` enum).
  //
  // LOWERING: `llvm.inline_asm has_side_effects "mov $0, %<reg>", "r,~{memory}"`, no result.
  let arguments = (ins I64:$value, BCIR_CtrlRegAttr:$reg);
  let assemblyFormat = "$reg `,` $value attr-dict `:` type($value)";
}

def BCIR_MsrReadOp : BCIR_Op<"msr_read"> {
  let summary = "Read an x86-64 model-specific register (boot/CPU-state asm edge, D1.4)";
  // A model-specific-register read (`rdmsr`) is a TRUSTED OPAQUE EFFECT EDGE in the same Tier-1 family
  // as `bcir.creg_read` (BCIR_DRIVER_KERNEL_ROADMAP.md Part II Tier 1): no C value-semantics, no cfront
  // dual-rail twin (the cfront rail expresses MSR access only as a raw `c.asm` blob), so the template is
  // authored DIRECTLY in LLVM-IR inline-asm syntax. Unlike `creg_read` the register is NOT an enum but a
  // RUNTIME index (`rdmsr` reads the MSR numbered by ECX), so the op takes a 32-bit index operand. x86-only.
  //
  //   * `$index` -- the MSR number (the 32-bit value `rdmsr` consumes in ECX; an `#bcir`-built i32).
  //   * result   -- the 64-bit MSR value (`rdmsr` returns it split EDX:EAX = high:low).
  //
  // LOWERING (`-convert-bcir-to-llvm`): the index is placed in ECX and `rdmsr` returns the two 32-bit
  // halves via a multi-output inline-asm (`llvm.inline_asm has_side_effects "rdmsr", "={ax},={dx},{cx},
  // ~{memory}"` -> `!llvm.struct<(i32, i32)>`); the lowering then REASSEMBLES the i64 (`(zext hi) << 32 |
  // zext lo`), so the op surface is a clean 64-bit value (matching the kernel `native_read_msr` idiom).
  // `has_side_effects` is ALWAYS set and the `~{memory}` clobber is intentional even for a READ: an MSR
  // is live CPU state (IA32_APIC_BASE, IA32_EFER, the TSC-deadline MSR, ...) and a read is an ORDERING
  // POINT, exactly as for `creg_read` -- without the clobber LLVM could reorder an unrelated load across it.
  //
  // DCE/CSE SAFETY: carries NO `Pure`/`NoMemoryEffect` trait by DESIGN (an MSR read is side-effecting --
  // re-reading yields fresh CPU state and must not be merged/elided), matching every asm-edge op.
  let arguments = (ins I32:$index);
  let results = (outs I64:$value);
  let assemblyFormat = "$index attr-dict `:` type($index) `->` type($value)";
}

def BCIR_MsrWriteOp : BCIR_Op<"msr_write"> {
  let summary = "Write an x86-64 model-specific register (boot/CPU-state asm edge, D1.4)";
  // The void counterpart of `bcir.msr_read`. A model-specific-register write (`wrmsr`) has system-wide
  // effects (writing IA32_EFER toggles long mode, IA32_APIC_BASE relocates the local APIC, the
  // SYSENTER/STAR MSRs install syscall entry points), so it is conservatively side-effecting AND carries
  // a `~{memory}` clobber. x86-only.
  //
  //   * `$index` -- the MSR number (placed in ECX).
  //   * `$value` -- the 64-bit value to write (`wrmsr` consumes it split EDX:EAX = high:low).
  //
  // LOWERING: the i64 value is SPLIT into its low/high 32-bit halves (`trunc value` / `trunc (value >>
  // 32)`), the index goes to ECX, and `llvm.inline_asm has_side_effects "wrmsr", "{cx},{ax},{dx},
  // ~{memory}"` consumes (index, low, high); no result. `has_side_effects` always.
  let arguments = (ins I32:$index, I64:$value);
  let assemblyFormat = "$index `,` $value attr-dict `:` type($index) `,` type($value)";
}

#endif // BCIR_CORE_OPS
