# BCIR C Frontend — User Guide (`bcir-cfront`)

`bcir-cfront` is the C-frontend driver: it compiles a useful subset of C23 through the BCIR claim
graph, verifies the result against the R1–R18 laws, emits behaviour-equivalent C, and reports
Clang-style diagnostics. What it cannot yet compile, it cleanly hands off (the LLVM-backend fallback
contract) — so it behaves like a *verified-subset* compiler, not a research toy.

This guide is the practical quickstart + capability/limits reference. For the numbers (test counts,
coverage) see [`STATUS.md`](STATUS.md); for where it sits in the project, the
[master roadmap](BCIR_MASTER_ROADMAP.md); for the Clang comparison, [`CLANG_COMPARISON.md`](CLANG_COMPARISON.md).

## Quickstart

```sh
# compile a file (verified C + R1–R18 status to stdout)
python -m bcir.frontends.cfront hello.c

# syntax/semantic check only — Clang-style diagnostics, no output
python -m bcir.frontends.cfront -fsyntax-only hello.c

# machine-readable diagnostics (for editors / CI)
python -m bcir.frontends.cfront --emit-json hello.c

# lay the types out for another target ABI
python -m bcir.frontends.cfront --target x86_64-windows hello.c

# graceful degradation: report a fallback-to-LLVM signal instead of erroring out
python -m bcir.frontends.cfront --fallback hello.c
```

The driver lives in [`bcir/frontends/cfront/__main__.py`](../bcir/frontends/cfront/__main__.py); the
library entry points are `compile_unit`, `diagnose`, and `compile_with_fallback`.

## Command-line options

| Option | Meaning |
| --- | --- |
| `-I <dir>` | add a `#include` search-path directory (repeatable; the source's own dir is always searched) |
| `-D name[=val]` | predefine an object macro (`val` defaults to 1) |
| `-U name` | undefine a predefined / `-D` macro |
| `-std=<std>` | language standard: `c23`/`c2x` (default), `c17`, `c11` |
| `-E` | preprocess only — print the expanded translation unit |
| `--target <abi>` | the target data model the unit is laid out for (default `x86_64-linux`) |
| `-fsyntax-only` | parse + check only; print diagnostics, emit no compiled output |
| `--emit-json` | print diagnostics as a machine-readable JSON array |
| `--fallback` | report a fallback-to-LLVM signal (exit 2) for unsupported constructs instead of erroring |
| `--r21 <policy>` | how a detected use-after-free / double-free (R21, §5.12) gates the compile: `advisory` (default; surfaced, never gates), `fallback` (route the unit to LLVM, exit 2), or `reject` (a hard verify error, exit 1) |
| `-o <file>` | write output to `<file>` instead of stdout |
| `--explain` | also print the per-function plan/explain record |
| `--selfcheck` | print the generated dual-rail self-check harness |

Exit codes: `0` clean, `1` a diagnostic (error), `2` a usage error or a fallback-to-LLVM signal.

## Diagnostics

Errors are reported in the Clang layout — a `file:line:col: severity: message` banner, the source
line, and a `^~~~` caret — with parser **error recovery** (one run reports several errors),
**fix-it** hints for missing punctuation, **`In file included from …`** frames for errors in headers,
and a `--emit-json` machine-readable form. The engine is
[`bcir/frontends/cfront/diagnostics.py`](../bcir/frontends/cfront/diagnostics.py).

```
hello.c:2:10: error: use of undeclared identifier 'undeclared'
    return undeclared + 1;
           ^
```

## The target ABI matrix

`--target` selects the data model the frontend lays types out for
([`abi.py`](../bcir/frontends/cfront/abi.py)). The named targets:

| Target | Data model | `long` | pointer | `long double` |
| --- | --- | --- | --- | --- |
| `x86_64-linux`, `aarch64-linux`, `riscv64-linux` | LP64 | 8 | 8 | 16 |
| `x86_64-windows` | LLP64 | 4 | 8 | 8 |
| `i386-linux` | ILP32 | 4 | 4 | 12 |

So `struct { long a; char b; }` is 16 bytes on LP64 and 8 bytes on LLP64 / ILP32. The host target's
output is validated against the host C compiler (behaviour-equivalence); a cross-target layout is not
byte-compatible with the host compiler, so its equivalence check is reported as `skip:cross-target`
(the layout is conformance-checked instead). Float math and calling conventions are delegated to the
backend.

## The LLVM-backend fallback contract

BCIR compiles the supported, fully-verified subset; for anything outside it, `--fallback` (library:
`compile_with_fallback`) returns a result whose `needs_fallback` is set and whose `fallback` names the
rejecting stage + reason — the signal for a driver to route the unit to the LLVM backend rather than
fail. Without `--fallback`, an unsupported construct is a normal diagnostic.

```
fb.c: fallback to LLVM backend: lower: static initializer is not a constant expression
```

## Pointer-bounds policy (§5.12)

What the compiler does with every array / pointer indexing, normatively:

- **Recoverable extent → checked.** A local or static array with a known declared shape, and a
  `malloc`/`calloc` pointer whose element count is recoverable (a stable count variable, or a
  side-effect-free count expression snapshotted once into a hidden immutable local at the
  allocation), promote from `assumed_safe` to **`masked`**: the emit carries
  `a[BCIR_CHK(rid, i, N, "<func>:<array>")]`. In-bounds the guard is transparent
  (behaviour-identical to the raw access); out-of-bounds it quarantines — the weak default handler
  records the provenance (site, index, extent) into a ring readable via `bcir_quarantine_report`
  and aborts fail-fast, while a strong override (the debugger / ML-layer seam) may record and
  recover through the decide-audit ring.
- **Not recoverable → `assumed_safe` (trusted).** A pointer parameter without a provable bound, a
  struct-member array (today), and an MMIO register access stay unguarded — zero overhead, trusted
  to land in their allocation.
- **Never fabricated.** BCIR does not invent a bound it cannot recover — there is no silent
  "proof"; an unprovable case stays trusted or routes to the LLVM backend under `--fallback`.

Both rails promote identically (the R13 digest includes the bounds decision, so a one-rail split is
a hard test failure), and `bcir-cc --emit-c` output containing a masked access is self-contained:
it pulls in `bcir_quarantine.h` and compiles + links against `runtime/c/bcir_quarantine.c`.

## Pointer-lifetime policy (R21, §5.12)

The frontend stamps lifetime events on `malloc`/`calloc` (alloc) and `free` (free), so the R21 law
catches a use-after-free / double-free a C program would otherwise leave as UB. By default this is
**advisory** — surfaced (e.g. via `--emit-claimgraph` / the oracle's `lifetime_diagnostics`) but never
gating the compile, so the supported corpus and the UB-free fuzzer are undisturbed. The `--r21` policy
promotes it to a verdict:

- `--r21=advisory` (default) — detect + surface only.
- `--r21=fallback` — a detected UAF/double-free routes the unit to the LLVM backend (exit 2), like the
  `--fallback` contract.
- `--r21=reject` — a detected UAF/double-free is a hard verify error (exit 1).

The Python oracle (`bcir-cfront`) and the C twin (`bcir-cc`) draw the **same exit code** for the same
input under the same policy; the cross-rail exit-code parity is gated in `tools/c/check_runtime.sh`.

```
uaf.c: lifetime error: R21 f: use-after-free of RID 102 (freed and not re-allocated)
```

## Inline assembly (ASM1)

GNU inline assembly — `asm` / `__asm__`, basic and extended — is modeled as an **ISA-neutral trusted
opaque effect edge**, exactly like the `c.call.libm.void:` external-effect family. It is **not
interpreted**: the assembly *template* is opaque and trusted, and BCIR owns only the **calling side** —
the operand binding, the constraints, the clobber declaration, and the ordering semantics.

```c
asm("nop");                                    // basic — implicitly volatile
asm volatile("" ::: "memory");                 // a compiler ordering barrier
asm("" : "=r"(out) : "r"(in));                 // extended: outputs : inputs : clobbers
asm("" : [o]"=r"(out) : [i]"r"(in) : "cc");    // symbolic names + a clobber
```

- **Verbatim, ISA-neutral pass-through.** The template + per-operand constraints + clobber list are
  re-emitted unchanged as a GNU statement, in the reserved `__asm__` / `__volatile__` spellings so the
  output is valid even under `-std=c11 -pedantic`. The same asm therefore compiles on whatever target the
  C compiler targets — no per-ISA logic in this slice.
- **BCIR owns the calling side.** Each lowers to a `c.asm:` (or `c.asm.volatile:`) claim whose **read**
  operands are the input values (plus any `"+"` read-write output lvalue read) and whose **written**
  operands are the output lvalues — so the alias/effect/verify machinery sees the real footprint. Output
  operands must be scalar local variables in this slice (a member / array / deref / bitfield / MMIO output
  lvalue is a follow-on).
- **Off the legality value-path.** The asm edge computes no verified value and emits no R-law verdict — it
  is a trusted opaque effect, like a `c.call.libm` edge.
- **Side-effect + barrier ordering.** A `volatile` asm (and a *basic* asm, which is implicitly volatile) is
  a side-effecting edge that is **never dead-code-eliminated**, even with unused outputs: the emit walks the
  claim graph in source order and never reorders or drops a claim. A `"memory"`-clobber or `volatile` asm
  additionally carries the `barriered` hazard — an **ordering fence** that is never reordered or fused
  across.
- **Deferred.** `asm goto` (label operands) is parsed for grammar completeness but rejected with an honest
  diagnostic. Per-ISA *semantic* modeling (port-I/O intrinsics, hardware barriers) is **ASM2 / ASM3**, not
  this slice — here raw inline asm is a trusted, re-emitted-verbatim edge.

## Port-mapped I/O (ASM2)

Unlike raw inline asm, port I/O has *known* semantics, so BCIR models the six intrinsics as a **typed
port-access trusted edge** — not an opaque template. Each is an ordinary **CALL expression** (no new
syntax), recognized at lowering like the `<math.h>` family:

```c
unsigned v = inb(0x60);            // read  u8  from an I/O port  -> inw -> u16, inl -> u32
outb(value, 0x60);                 // write u8  to an I/O port    -> outw, outl (u16/u32)
```

- **Six intrinsics.** Reads return the value — `inb`→`u8`, `inw`→`u16`, `inl`→`u32`; writes are `void` —
  `outb`/`outw`/`outl`. The `port` is conceptually a `u16` I/O-port address.
- **The Linux `out*(value, port)` convention.** The written **value is the first argument** and the **port
  is the second** (matching `<asm/io.h>`). Some headers use the reverse order — the convention is pinned
  here and tested, so a reversed call would fail rather than silently miscompile.
- **Typed, isolated, barriered edge.** Each lowers to a `c.portio.in.{b,w,l}:` / `c.portio.out.{b,w,l}:`
  claim carrying the access **width + direction** in the op suffix (the IR records "a width-1 port read",
  not an opaque blob). The access is **isolated** under the I/O address space — it reads/writes a dedicated
  `__ioport` resource in the **MMIO domain**, so a port access can never alias a normal-memory RID — and is
  **`barriered`** (volatile + ordered): two port ops share that resource, so they never reorder, fuse, or
  eliminate. It is **off the legality value-path** (a trusted effect, no R-law verdict), exactly like the
  inline-asm and `c.call.libm` edges.
- **Per-`--target` emit (ISA-neutral IR, per-ISA realization).** Port I/O exists **only on x86** (the
  `in`/`out` instructions). For an **x86 target** (`x86_64-linux` / `i386-linux` / `x86_64-windows`) the
  edge emits the real instruction as a GNU `__asm__ __volatile__`, reusing the ASM1 trusted-edge + barrier
  machinery and the standard `<asm/io.h>` operand constraints (`"=a"`/`"a"` accumulator, `"Nd"`
  immediate-or-`dx` port):

  ```c
  __asm__ __volatile__ ("inb %w1, %b0" : "=a" (v) : "Nd" (port));        // inb (inw %w0, inl %k0)
  __asm__ __volatile__ ("outb %b0, %w1" :  : "a" (value), "Nd" (port));  // outb (outw %w0, outl %k0)
  ```

  For a **non-x86 target** (`aarch64-linux` / `riscv64-linux`) port I/O is genuinely **unsupported** —
  these ISAs have no port I/O, only MMIO — so the emit raises an honest `CLowerError` (*"port-mapped I/O
  (inb) requires an x86 target; aarch64-linux has no port I/O — use MMIO"*) that routes the unit to the
  LLVM fallback (the established honest-depth pattern).
- **Privileged-execution honest boundary (assemble-only).** Executing `in`/`out` from userspace **traps**
  — it needs `iopl`/`ioperm` + ring-0. So the emitted asm is verified by **assembling** it (`gcc -c` /
  `clang -c`, `-std=c11 -pedantic`) — proving it is valid x86 the toolchain accepts — and is **never linked
  or run**. That is the honest seam: the emitted instruction is real and assembles; execution is privileged
  and gated, like the SYCL device path.
- **Deferred.** String/block I/O (`insb`/`outsb` …), the paused `*_p` variants (`inb_p`/`outb_p`), and any
  non-integer port/value are out of this slice (a non-integer port or value is an honest diagnostic).

> **Python-rail-only (C-twin gap).** Inline asm (ASM1) and port-mapped I/O (ASM2) are a **Python-frontend-only**
> feature today: the C twin (`runtime/c/bcir_cfront.c`) does **not** parse inline assembly or port I/O at all.
> So H1's C-twin sanitizer / fuzz sweep — the malformed-input robustness coverage the rest of the C subset gets
> — does **not** reach `_asm_stmt` / `_portio`. That robustness is instead covered on the Python rail by the
> dedicated red-team `bcir/tests/test_cfront_asm_portio_redteam.py`, which feeds malformed / adversarial asm +
> portio snippets through `compile_unit` / `diagnose` and asserts each one either lowers cleanly or raises a
> clean cfront diagnostic (`CParseError` / `CLexError` / `CPPError` / `CLowerError`), never an uncaught internal
> Python exception. (The cfront fuzz corruptor `cfuzz.py` likewise has no asm/portio vocabulary — extending it
> is a follow-up; the red-team is the primary robustness gate for these paths.)

## Hardware barriers (ASM3)

The memory-fence intrinsic was already a recognized `barriered` claim; ASM3 deepens it the same way ASM2
deepened raw asm into typed port I/O — **typed fence kinds** plus **real per-ISA assembly emit behind
`--target`**. Each fence is an ordinary **CALL expression** (no new syntax), recognized at lowering like the
atomic / port-I/O families:

```c
__sync_synchronize();              // full (seq_cst) fence  -> c.fence
atomic_thread_fence(5);            // C11 <stdatomic.h>, full fence -> c.fence
_mm_mfence();                      // x86-conventional full  fence -> c.fence
_mm_lfence();                      //                   load (acquire) fence -> c.fence.acquire
_mm_sfence();                      //                   store (release) fence -> c.fence.release
```

- **Recognized intrinsics + kinds.** `__sync_synchronize`, the GCC/Clang `__atomic_thread_fence`, and the
  C11 `<stdatomic.h>` `atomic_thread_fence` (newly recognized) are **full (seq_cst)** fences; the
  x86-conventional `_mm_mfence` is also full, `_mm_lfence` is the **load (acquire)** fence, and `_mm_sfence`
  is the **store (release)** fence. The kind is read off the intrinsic **name** — no `memory_order` argument
  is parsed (those constants are not part of this subset).
- **Backward-compatible op strings.** The **full** fence keeps the existing op string **`c.fence`** — so the
  existing `__atomic_thread_fence` / `__sync_synchronize` claims, and the Python↔C dual-rail parity digest,
  are **unchanged** (no digest/parity churn). The two lighter kinds get the new op strings **`c.fence.acquire`**
  and **`c.fence.release`**. The edge stays `Opcode.BARRIER`, `lane A`, **`barriered`** (never reordered /
  fused across), and off the legality value-path (a trusted effect, no R-law verdict) — exactly as before.
- **Per-`--target` emit (ISA-neutral IR, per-ISA realization).** The bare portable
  `__atomic_thread_fence(__ATOMIC_SEQ_CST);` is replaced by the real hardware-barrier instruction behind a
  GNU `__asm__ __volatile__ (… ::: "memory")`, keyed off `--target`. The `"memory"` clobber is the
  **required compiler-barrier half** of the fence:

  | kind | x86 | aarch64 | riscv64 |
  |------|-----|---------|---------|
  | full (`c.fence`)            | `mfence` | `dmb ish`   | `fence rw,rw` |
  | acquire (`c.fence.acquire`) | `lfence` | `dmb ishld` | `fence r,rw`  |
  | release (`c.fence.release`) | `sfence` | `dmb ishst` | `fence rw,w`  |

  ```c
  __asm__ __volatile__ ("mfence" ::: "memory");      // x86 full fence (lfence / sfence for acquire / release)
  __asm__ __volatile__ ("dmb ish" ::: "memory");     // aarch64 full fence (dmb ishld / ishst)
  __asm__ __volatile__ ("fence rw,rw" ::: "memory"); // riscv64 full fence (fence r,rw / rw,w)
  ```

  **Unlike port I/O, every ISA has a fence** — so a target *outside* the three families is **not** an
  unsupported diagnostic; it keeps the portable `__atomic_thread_fence(__ATOMIC_SEQ_CST);` as an honest
  default. All five shipping ABIs are covered by the three families, so the default is a safety net only.
- **Per-ISA assemble (host-arch-gated, carried-forward lesson).** Barriers are per-ISA and cannot be
  cross-assembled (the aarch64 CI runner has no x86 sysroot, and vice-versa). So the gate assembles each
  fence for the **host's own native arch** (`gcc -c` / `clang -c`, assemble-only) and asserts the emit
  **text** for non-native targets without assembling — real assembled coverage on every CI lane (x86 lanes
  assemble `mfence`/`lfence`/`sfence`; the aarch64 lane assembles `dmb ish`/`ishld`/`ishst`).
- **Cross-claim ordering enforcement (ASM3b) — done.** ASM3 is a frontend emit/recognition slice (typed
  fence kinds + native emit); **ASM3b** makes `barriered` *forbid* the optimizer from reordering or fusing
  **other** claims across the edge. A `barriered`-hazard claim is now a **first-class ordering edge**:
  - **No reorder across it.** `bundle._conflict` treats a barriered claim as conflicting with *every* other
    claim, so `find_bundles` / `_legal_reorder` never bundle a barriered claim and never move any claim past
    one (a hard reorder fence — independent of any data hazard).
  - **No fusion across it.** `realize.fused_candidates` **skips** the ×0.75 memory deforestation discount
    when the consumer is `barriered` **or** a shared operand was produced by a `barriered` producer — the
    fence forces the intermediate to materialize, so the producer→consumer round-trip is not elided. The
    MLIR cost model (`BCIRCostModel.h::fusedColumns`) mirrors this byte-for-byte for **R13 parity** (the
    FileCheck twin is `mlir/test/passes/cost_model_barrier.mlir`).
  - **Scope: all `barriered` claims** — memory fences (`c.fence*`), MMIO loads/stores (`Domain.MMIO`),
    port-I/O (`c.portio.*`), and volatile/`"memory"`-clobber inline asm — so real MMIO/port-I/O/asm ordering
    is enforced, not just the fence intrinsic.
  - **A structural property, not a verdict R-law.** `verify.verify_barrier_ordering(module, plan)` verifies
    a realized plan never schedules a claim across a barrier — checked **out of** the frontend verdict
    (`CompileResult.is_clean`), exactly like the R21 lifetime advisory; barriers stay off the legality
    value-path. It is a **safe no-op** on any module with no barriered claim (neither guard fires).

## What's supported

- Fixed-width and core integer types, `_Bool`/`char`, `void`, `float`/`double`/`long double`, pointers,
  arrays, `struct`/`union` (Clang-compatible layout, per target), `enum`, `typedef`.
- Integer + IEEE-754 floating arithmetic and comparisons, casts and the usual arithmetic conversions,
  `sizeof`/`_Alignof`, bitfields, `<math.h>` library calls.
- Functions, the call graph (R18: callee resolution, no recursion), inter-procedural summary reuse,
  function pointers — as a `typedef`'d parameter, a `struct` member (HAL dispatch table), **and as a
  local variable** (`RET (*f)(PARAMS) = fn;`, reassignable, called indirectly, return-type-signed).
- **Array compound literals — the full surface:** 1-D scalar (indexed `(T[]){...}[i]`, sized + zero-fill
  `(T[N]){...}`, signed-element), **multi-dimensional scalar** `(T[A][B]){...}[i][j]` (incl. an inferred
  outer dim `(T[][N]){...}` and a designated outer `{[1]=..,[0]=..}`), **1-D aggregate-element**
  `(struct P[]){...}[i].field`, and **multi-dimensional aggregate-element** `(struct P[A][B]){...}[i][j].field`.
- Local array declarations with initializers, including nested-brace multi-dim (`T a[A][B]={{..},{..}}`),
  inferred-size (`T a[]={..}`), and array-of-structs (`struct P a[N]={{..},{..}}`).
- **Computed goto** — the GNU label-as-value `&&L` (a `void *`) and the indirect `goto *p`.
- String/character literals (with prefixes), `static` locals, file-scope globals, `volatile` (MMIO).
- The preprocessor: `#include`/`#embed`, conditionals, object/function-like + variadic macros, the
  predefined macros, `#line`, `_Pragma`, and the `__has_*` feature-test operators.

## Known limits

These are reported as diagnostics, or — with `--fallback` — as a fallback-to-LLVM signal:

- Non-constant `static`/global initializers; constructs beyond the L1–L6 statement subset.
- 64-bit-integer **results** of a few `<math.h>` functions and pointer out-params are supported, but a
  general 64-bit *value* model and Windows/ILP32 *code generation* (vs. layout) are not.
- The i386 in-struct `double`-alignment quirk is not modelled (the `long`/pointer/`long double` data
  axes are).
- `_Decimal32`/`_Decimal64`/`_Decimal128` are **blocked, not unsupported in principle**: Clang 18
  cannot compile `_Decimal`, so the form is un-validatable under the Clang-equivalence methodology and
  is gated out until a `_Decimal`-capable reference compiler is available.
- An `Index`-base array-member access (`x[i].v[j]`, a member array indexed off an indexed base) stays a
  general limitation; the array-compound-literal `[i][j].field` form is fully supported.
- Cross-target builds are layout-only here; running them needs a cross toolchain.

When in doubt, run `-fsyntax-only` (or `--fallback`) — the frontend names exactly what it can't do.
