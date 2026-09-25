# Declared alias facts — the elementwise kernel's aliasing, element type, volatility and ordering, carried to LLVM (GEM+ G9)

An alias analysis *infers* what may alias. BCIR does not have to: a claim **declares** the
resources it reads and writes (RIDs), its hazard contract and its volatility, and every resource
declares its element size. The lowering's job is to carry those facts to LLVM exactly — never
fewer than declared (LLVM then knows less than BCIR does) and never more (LLVM then reorders or
elides on a fact the declaration contradicts, which is undefined behaviour, not a missed
optimization). This is the reference for how the elementwise kernel `C = A op B` does it (GEM+
roadmap G9, staged plan S5-A). The executable oracle is:

| Module | Holds |
|---|---|
| [`bcir/lower/alias_facts.py`](../../bcir/lower/alias_facts.py) | `kernel_facts` — the one derivation; `hazard_refusal` — the subset's ordering boundary; `bound_harness` — the self-checks' RID binding |
| [`bcir/lower/llvm.py`](../../bcir/lower/llvm.py) | the LLVM kernel (`emit_kernel_ll`, which the AOT, JIT and WASM paths share) and its harness |
| [`bcir/lower/c_kernel.py`](../../bcir/lower/c_kernel.py), [`specialist.py`](../../bcir/lower/specialist.py) | the C kernel, the gather form, the Q-fixed kernel, the ABI header, the hot-shape specialist, and their self-checks |
| [`bcir/lower/wasm.py`](../../bcir/lower/wasm.py) | the node harness's binding (`harness_binding`) |
| [`bcir/verify/alias.py`](../../bcir/verify/alias.py) | R12's reader of the facts a kernel's text carries, on both backends |
| [`bcir/tests/alias_fixtures.py`](../../bcir/tests/alias_fixtures.py), [`tools/perf/check_alias.py`](../../tools/perf/check_alias.py) | the corpus, the grader and the gate |

## The facts

`kernel_facts(module, claim, elem)` reads the declaration once. Every emitter of the elementwise
claim and R12 on both backends read the result — it replaced four predicates that disagreed
(docs/security/laws.md L14).

**The RID partition.** The operand positions are (A, B, C) = (`rd[0]`, `rd[1]`, `wr[0]`). A
position is **exclusive** when no other position names its resource. Only an exclusive pointer
carries LLVM `noalias` or C `restrict`. Each distinct resource is one alias scope, in one domain
per kernel; an access names its own resource's scope in `!alias.scope` and every other resource's
in `!noalias`. An in-place claim (`rd=(1, 2), wr=(1,)`):

```llvm
define void @bcir_kernel(ptr %A, ptr noalias %B, ptr %C, i64 %n) {
  ...
  %va = load <8 x float>, ptr %pa, align 4, !alias.scope !3, !noalias !4, !tbaa !8
  %vb = load <8 x float>, ptr %pb, align 4, !alias.scope !4, !noalias !3, !tbaa !8
  store <8 x float> %vc, ptr %pc, align 4, !alias.scope !3, !noalias !4, !tbaa !8
  ...
!0 = distinct !{!0, !"bcir.bcir_kernel"}               ; the kernel's alias domain
!1 = distinct !{!1, !0, !"bcir.bcir_kernel.rid1"}      ; resource 1: A and C
!2 = distinct !{!2, !0, !"bcir.bcir_kernel.rid2"}      ; resource 2: B
!3 = !{!1}
!4 = !{!2}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!"omnipotent char", !5, i64 0}
!7 = !{!"float", !6, i64 0}
!8 = !{!7, !7, i64 0}
```

The partition lives in the scopes, not in TBAA, on purpose. Scopes are per call site: an inliner
clones a function's scopes for every call it inlines, so they relate the accesses of one
execution and assert nothing about another. A TBAA type is a property of *memory*, never cloned.
One type per resource would be false the moment a caller passed one buffer as resource 1 to one
call and as resource 2 to the next.

**The element type.** It is the lowering contract's (`elem`: `f32` or `i32`), and every operand
resource must declare its size. A 4-byte kernel over a resource of 8-byte elements addresses the
wrong bytes; a resource the module does not declare has no element type at all. Both are refused,
never emitted. Every access carries the TBAA tag clang gives the same C type (`float`, `int` —
`int32_t` is `int` on every target the kernel builds for). The tag is sound because the kernel is
called through a C ABI with exactly those pointer types (the harness's prototype, the C backend's
signature, the ABI header), so it asserts nothing C's effective-type rule does not already
require of the caller. It also composes: under LTO the kernel and its C caller share one type
system, instead of two roots LLVM cannot compare.

**Volatility.** Every access of a volatile claim is volatile, on every emitter, and no access of
any other claim. A vector kernel's volatile access is `W` elements wide: the planner's selected
width is the realization, and volatility is carried on each access it performs.

**The hazard.** `unique` needs no ordering. `barriered` is a full barrier: `fence seq_cst`
(LLVM) or `atomic_thread_fence(memory_order_seq_cst)` (C) before the kernel's first access and
after its last. An `atomic` hazard needs atomic element operations, which the subset does not
generate, and an unknown hazard names no ordering. Both are refused by `find_elementwise` — the
gate every lowering entry point passes — and by `kernel_facts`, which shares the predicate to stay
total. Before S5-A every emitter lowered them to plain loads and stores, and every ordered kernel
then failed its own R12. A volatile claim, which R5 requires to be ordered, could never produce a
kernel the law accepts.

## Every emitter

| Emitter | `noalias` / `restrict` | alias scopes | TBAA | `volatile` | fences | refuses |
|---|---|---|---|---|---|---|
| LLVM kernel (`emit_kernel_ll`) | exclusive pointers | every access | every access | every access | entry, every exit | hazard, element size, undeclared |
| C kernel (`emit_kernel_c`) | exclusive pointers | — (clang derives them from `restrict` under inlining) | clang's own | pointees | first and last statement | hazard, element size, undeclared |
| gather form (`emit_gather_kernel_c`) | exclusive pointers (`idx` stays `restrict`) | — | clang's own | pointees | first and last statement | hazard, element size, undeclared |
| hot-shape specialist (`synthesize`) | exclusive pointers | — | clang's own | pointees | first and last statement | hazard, element size, undeclared |
| ABI header (`emit_header_c`, given the plan) | exclusive pointers | — | — | pointees (the definition's own type) | — (a prototype) | hazard, element size, undeclared |
| Q-fixed kernel (`emit_qfixed_kernel_c`) | exclusive pointers | — | clang's own | pointees | first and last statement | hazard (its lanes are its own representation, so it reads no declared element) |

Without the plan the header can only state the disjoint contract, which was the one it
published for every claim. The library facade (`bcir.api.build_artifact`) passes the plan.

## R12

`verify_lowering` and `verify_c_lowering` read the facts out of the kernel's text
(`bcir/verify/alias.py`) and hold each to `kernel_facts`. The findings name what went wrong:

- a **false fact** — `noalias` / `restrict` on a pointer whose resource another operand names;
  an access naming its own resource's scope in `!noalias`; a plain access in a volatile claim;
  a TBAA type the claim does not declare;
- a **dropped fact** — a missing `noalias` on an exclusive pointer, a missing scope or tag, a
  `!noalias` omitting a resource, a fence the hazard needs;
- an **undeclared one** — `volatile` or a fence in a claim that declares neither.

The reader is total: dangling or self-referential metadata, a truncated kernel, text with no
function — each is a finding, never a traceback (L1). It reads the whole language it is handed,
not a subset of it (L4):

- every `load` and `store`, in any spelling LLVM's parser accepts: with or without an alignment
  (the parser fills it in), atomic or not. An access through a pointer no operand explains is a
  finding;
- a fence's sync scope. Only the system scope — no `syncscope`, or `syncscope("")`, its
  spelled-out name — is the barrier a hazard contract declares. A
  `syncscope("singlethread")` fence orders a thread against its own signal handlers and nothing
  else;
- in C, the body as well as the declarations. The parameter qualifiers are the facts of every
  access only while every access goes through them. So an operand is used only by subscript, no
  address is taken, and a barriered kernel has no `return` between its fences. A cast or an
  address sheds `volatile` (or `const`), and a `return` skips the exit fence.

## The self-checks

Every harness binds one buffer per declared resource, so an in-place claim runs in place. The
harnesses are the LLVM AOT/JIT harness, the C and Q-fixed self-checks, and the WASM node harness
(`harness_binding`). Each buffer is initialized, snapshotted and checked after the call. The
written one must hold `A op B` below `n` and its snapshot from `n` on, so an unmasked tail fails.
Every other buffer must hold its snapshot everywhere, so a write through a read pointer fails.
Before S5-A each harness passed three private buffers whatever the RIDs said, so no kernel's
alias facts were ever executed against the aliasing they describe. The node harness also computed
`A + B` whatever the claim's operation, so a correct SUB or MUL kernel failed it.

## The rows

`bcir/tests/alias_fixtures.py::measure` grades them over 504 lawful kernels (7 RID partitions ×
3 operations × 2 element types × 4 widths × 3 contracts) and 112 modules the subset must refuse.
The partitions are all five of three positions plus two relabellings, so no check can key on RID
values or their order. The grader derives the declared facts itself, parses the emitted ones
itself and reads the harness bindings from the harness text, all through entry points the parent
tree has. RED is the parent (ad4ebff0), GREEN this tree:

| Row | RED | GREEN |
|---|---:|---:|
| `alias.noalias.mismatch` | 648 | **0** |
| `alias.scope.mismatch` | 2,646 | **0** |
| `alias.tbaa.mismatch` | 2,646 | **0** |
| `alias.volatile.mismatch` | 1,533 | **0** |
| `alias.fence.mismatch` | 742 | **0** |
| `alias.refusal.accepted` | 602 | **0** |
| `alias.differential.silent` | 1,072 | **0** |
| `alias.harness.unaliased` | 75 | **0** |
| `alias.r12.rejected` | 336 | **0** |
| `alias.r12.forgery.accepted` | 1,084 | **0** |
| `alias.llvm.false_noalias` (LLVM's default alias analysis) | 0 | **0** |
| `alias.llvm.scope_facts.missing` (the scoped-noalias analysis alone) | 300 | **0** |
| `alias.backends.disagree` (clang's IR for the C kernel against the LLVM kernel) | 132 | **0** |
| `alias.llvm.caller_memory` (the kernel inlined into a C caller: the caller's own data around it) | 56 | **0** |
| `alias.exec.failed` (LLVM AOT/JIT, C, Q-fixed, WASM under node) | 9 | **0** |

The last five need a coherent clang, llvm-link and opt, and are NOT-MEASURED without them.
`alias.llvm.false_noalias` was already zero: it is the guard the landed half of G9 established.

## What LLVM gains, measured

Two measurements, and they point different ways, which is the honest result.

**The kernel alone: nothing changes, and that is the bound.** With three pointers, at most one
resource is shared, so `noalias` on the exclusive pointers already carried the whole partition.
LLVM's scoped-noalias analysis now proves the same pairs from the scopes alone
(`alias.llvm.scope_facts.missing` 300 → 0), but it proves nothing new. A single-type kernel gives
TBAA nothing to separate. Compiled at `-O2` by clang 18 and 23, for `x86-64-v3` and baseline
`x86-64`, the machine code of every unique kernel is byte-identical to the parent's: 112 of 112
(7 partitions × 2 operations × 2 widths × 2 LLVM majors × 2 x86 levels). This row is at its
bound, not missed.

**The kernel inlined into its C caller: the facts pay.** A caller that keeps its own `long` in
memory around the call —

```c
void caller(const float *A, const float *B, float *C, long n, long *count) {
  *count = 7;
  k(A, B, C, n);        /* inlined */
  *count += n;
}
```

— linked with the kernel and optimized (`llvm-link`, `opt -O2`): on the parent, LLVM keeps the
store of 7 and reloads `*count` after the kernel, because the kernel's float stores may alias
it. `noalias` parameters do not help. They relate only the accesses inside the call, and the
caller's are outside it. On this tree the kernel's accesses carry clang's `float` / `int` tags,
the caller's carry clang's `long` tag in the same type system, and LLVM proves them disjoint:
the store is dead and the value after the kernel is known (`%r = add nsw i64 %n, 7`). That is
one store and one load fewer per call site, for every partition, both widths and both element
types, on LLVM 18 and 23 alike (`alias.llvm.caller_memory` 56 → 0 over 84 callers). The same row
checks the other direction: a barriered kernel must keep both, because its fences order the
caller's memory too. A fault that drops both fences is caught there, by LLVM's optimizer rather
than by a text check.

So TBAA under clang's root is not decoration. It is what lets the kernel compose with the C code
around it, and on a separate root that composition would be lost. The scopes will pay the same
way inside a kernel once one carries more than one shared class, which G8's fused movement
kernels will: `noalias` cannot express `{A, B} ⟂ {C, D}`, and scopes can.

## Not claimed

- **Kernels beyond the subset.** The facts are derived for the single-claim elementwise kernel.
  A multi-claim or fused kernel (G8's movement) will need the same derivation per operand. That
  is where the scopes and TBAA carry information `noalias` cannot.
- **Atomic element operations.** They are refused, not lowered.
- **The Q-fixed kernel's element size.** Its lanes are `lane_bits` wide whatever the resources
  declare. That is a pre-existing representation question this slice records but does not
  settle.
- **A law-rail twin.** The MLIR rail emits no LLVM kernel: `-bcir-lower-to-llvm` checks the
  segment contract, and the law's lowering record is `bcir.target.lower_contract`. The C backend
  is the second path to LLVM, and `alias.backends.disagree` holds the two to one fact set.
