# N8, a native kernel lowering in MLIR: the GO/STOP (PORT-10)

**Status:** decision record, 2026-11-21. **Decision: STOP**, with the conditions that reopen it (§4).
**Owner of the question:** the native port audit
([`BCIR_NATIVE_PORT_AUDIT_2026-11-09.md`](BCIR_NATIVE_PORT_AUDIT_2026-11-09.md)), whose §5 puts PORT-10
"behind a GO/STOP written after PORT-5". PORT-5 landed the hosted C++ library's foundation on 2026-11-20;
this is that record.

## 1. The question

N8 is the oracle's LLVM lowering -- `lower.llvm`, `lower.alias_facts`, `lower.poison`, with `verify.alias` and
`verify.poison`, 1,495 lines -- realized natively, as a BCIR-to-LLVM-dialect lowering in the MLIR rail. The
audit's row: the law rail "emits no LLVM kernel"; `-convert-bcir-to-llvm` covers only the machine edge (inline
assembly, MMIO, atomics, interrupt frames); the JIT microkernel's small runtime codegen slices (driver roadmap
§7.5) would need the native path. GO means building that pass now. STOP means the Python emitter stays the one
LLVM emitter of the kernel subset, and the question is reopened when a condition below holds.

## 2. The evidence

- **What the lowering is today.** `PARITY.md`'s row is a *partial* AOT/JIT subset: exactly one 2-read/1-write
  add, sub or mul claim, with the runtime-`n` tail contract, the declared alias facts (S5-A) and the proved
  poison flags; everything else is refused (`bcir.target.lower_contract`), not lowered. A native port of that
  subset is a port of a contract one claim wide.
- **There is already a second path to LLVM.** The C backend (`lower.c_kernel`, compiled by clang) reaches LLVM
  independently of the textual emitter, and `alias.backends.disagree` holds the two to one alias-fact set,
  judged by LLVM's own alias analysis. The differential the audit's rule 4 asks of a native port exists for this
  subset without one: an MLIR emitter would be a third realization of the alias facts, and the parity gate would
  have to grow to three.
- **No consumer needs it yet.** The JIT microkernel is driver-program row 13, after the driver rows that supply
  its evidence (§7.5: "true runtime code generation is limited to small verifiable slices with an explicit
  code-integrity and rollback policy"). No slice in the master roadmap's current order instantiates code at run
  time. The AOT path (the emitter, then clang) and the JIT path (the emitter, then `lli`) both work, under the
  CI's LLVM 22 and 23 cells.
- **What PORT-5 taught.** The hosted C++ library's conventions hold (a C ABI, one arena per solve, failures as
  values, a parity corpus, a façade fuzzer, the allocation campaign), and a standard library component can fail
  the campaign (libstdc++'s pool resource). An MLIR pass is a different rail with its own conventions -- the law
  rail must not link the solver library (audit §4.2), and its passes are held by lit and the ODS corpus, not by
  `check_solver.sh` -- so PORT-5 lowers no part of N8's cost.
- **What GO would cost.** A BCIR-to-LLVM-dialect conversion for the claim subset, the alias-scope and TBAA
  metadata the textual emitter writes, the poison flags with their proof obligations, the tail contract, and a
  three-way parity harness (textual emitter, C backend, MLIR) over the lowering corpus under LLVM 22 and 23. That
  is a pass family on the scale of the GEM lowering passes, landed for a contract that refuses all but one claim
  shape.

## 3. The decision

**STOP.** A native lowering would duplicate a contract that is one claim wide, add a third emitter to an
alias-fact parity that already has two independent paths, and serve no consumer the current order schedules. The
open question in this area is not where the lowering runs but how wide its contract is -- arbitrary-graph
lowering, which `PARITY.md` declares out of scope -- and widening it is a design slice for the master roadmap,
not a port. The Python emitters (`lower.llvm`, `c_kernel`, `wasm`, `jvm_class`) stay the generators and oracles,
as the audit's row already says.

## 4. What reopens it (GO when any holds)

1. **A runtime consumer is scheduled**: a driver or kernel slice that instantiates code on a target at run time
   (driver roadmap §7.5's small verifiable slices, row 13) reaches the master roadmap's order. The slice's
   code-integrity and rollback policy is the GO's precondition, written first.
2. **The lowering contract widens**: a slice admits more than the straight-line claim (`lower_contract` grows a
   rung). Then the native and textual emitters would both have to carry the wider contract, and the cheaper
   moment to add the native one is before the textual one grows further.
3. **The Python emitter is scheduled for retirement** ([the TMSAO proposal](BCIR_TMSAO_ASN1_JSON_DRIVER_PROPOSAL.md)'s CXX7, "remove only Python duplicates
   with stable native coverage"): the native lowering must exist, at parity, before the Python one may go.

## 5. What GO will contain

When a condition in §4 holds, PORT-10 is: the conversion pass in the MLIR rail (`mlir/lib/passes`), its lit
fixtures beside the existing `convert_to_llvm.mlir`, the alias facts and poison flags as LLVM-dialect metadata,
and the three-way differential -- textual emitter, C backend, MLIR pass -- over the lowering corpus, with
`alias.backends.disagree` extended to three, under the CI's LLVM 22 and 23 cells. Every injected fault in the
pass is swept RED, as the C and C++ twins' are.
