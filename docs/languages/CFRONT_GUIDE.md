# BCIR C Frontend — User Guide (`bcir-cfront`)

`bcir-cfront` is the C-frontend driver: it compiles a useful subset of C23 through the BCIR claim
graph, verifies the result against the R1–R18 laws, emits behaviour-equivalent C, and reports
Clang-style diagnostics. What it cannot yet compile, it cleanly hands off (the LLVM-backend fallback
contract) — so it behaves like a *verified-subset* compiler, not a research toy.

This guide is the practical quickstart + capability/limits reference. For the numbers (test counts,
coverage) see [`STATUS.md`](../STATUS.md); for where it sits in the project, the
[master roadmap](../BCIR_MASTER_ROADMAP.md); for the Clang comparison, [`CLANG_COMPARISON.md`](../research/CLANG_COMPARISON.md).

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

The driver lives in [`bcir/frontends/cfront/__main__.py`](../../bcir/frontends/cfront/__main__.py); the
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
| `--r21 <policy>` | how a detected use-after-free / double-free (R21, LangRef §10) gates the compile: `advisory` (default; surfaced, never gates), `fallback` (route the unit to LLVM, exit 2), or `reject` (a hard verify error, exit 1) |
| `-o <file>` | write output to `<file>` instead of stdout |
| `--explain` | also print the per-function plan/explain record |
| `--selfcheck` | print the generated dual-rail self-check harness |

Exit codes: `0` clean, `1` a diagnostic (error), `2` a usage error or a fallback-to-LLVM signal.

## Diagnostics

Errors are reported in the Clang layout — a `file:line:col: severity: message` banner, the source
line, and a `^~~~` caret — with parser **error recovery** (one run reports several errors),
**fix-it** hints for missing punctuation, **`In file included from …`** frames for errors in headers,
and a `--emit-json` machine-readable form. The engine is
[`bcir/frontends/cfront/diagnostics.py`](../../bcir/frontends/cfront/diagnostics.py).

```
hello.c:2:10: error: use of undeclared identifier 'undeclared'
    return undeclared + 1;
           ^
```

## The target ABI matrix

`--target` selects the data model the frontend lays types out for
([`abi.py`](../../bcir/frontends/cfront/abi.py)). The named targets:

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
fb.c: fallback to LLVM backend: lower: a static initializer is not an integer constant expression
```

## Pointer-bounds policy (LangRef §4)

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

## Pointer-lifetime policy (R21, LangRef §10)

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
  diagnostic. Kernel interrupt entry does not weaken this rule: the MLIR rail uses the dedicated,
  typed `bcir.interrupt_trampoline` module-assembly op for the normal x86 entry/`iretq` shim, while
  the verified C rail contains only the handler body. Per-ISA *semantic* modeling (port-I/O
  intrinsics, hardware barriers) is **ASM2 / ASM3**, not this slice — here raw inline asm is a
  trusted, re-emitted-verbatim edge.

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
  arrays, `struct`/`union` (Clang-compatible layout, per target), `enum`, `typedef`. An object of an enumerated type
  has the integer type its enumeration is compatible with, as Clang gives it on the target: `unsigned int` when no
  enumerator is negative on the System V targets, `int` otherwise and on the MSVC target; the enumeration constants
  are `int` (C11 6.7.2.2p4, 6.4.4.3).
- Integer constants in every base (`0x`, `0b`, a leading `0` octal, decimal) and with every suffix C spells -- a
  `u` before or after an `l`, `L`, `ll` or `LL` -- each its exact value in its C11 6.4.4.1 type on the target
  (`0xFFFFFFFFFFFFFFFFu` is 2^64 - 1, an `unsigned long` where `long` is 64 bits and an `unsigned long long` where
  it is 32; `017` is 15, in a `#if` too). Any other run of `u`s and `l`s (`1lL`, `1uu`, `1lul`) is refused in
  Clang's words, in code and in a `#if` (`invalid suffix 'lL' on integer constant`).
- Integer constant expressions -- an enumerator's value, a case label, an array dimension, a designator --
  folded where they are parsed, in C's own types on the target, by the predicate a static's initializer folds
  with: the integer promotions and the usual arithmetic conversions (`~0u > 5` is 1, `-1 < 0u` is 0, `-1L < 1u`
  is 1 where `long` is 64 bits and 0 where it is 32), `/` and `%` truncating toward zero (`-7 / 2` is -3, `-7 % 2`
  is -1), shifts, a cast to an integer type (`(uint8_t)300` is 44), `?:` in its arms' common type, and an
  operand C does not evaluate left unevaluated (`0 && 1 / 0` is 0). A dimension that is an integer constant
  expression (`N * 2`, `2 + 1`) makes a fixed array (C11 6.7.6.2p4) for a local, a member, a typedef, a
  parameter and a global. Both emits spell a negative constant signed (`-3`, not `-3u` or its 64-bit two's
  complement) and a case label past `LLONG_MAX` with `u`.
- Integer + IEEE-754 floating arithmetic and comparisons, casts and the usual arithmetic conversions,
  `sizeof`/`_Alignof`, bitfields, `<math.h>` library calls.
- Functions, the call graph (R18: callee resolution, no recursion), inter-procedural summary reuse,
  function pointers — as a `typedef`'d parameter, a `struct` member (HAL dispatch table), **and as a
  local variable** (`RET (*f)(PARAMS) = fn;`, reassignable, called indirectly, return-type-signed).
  A call through a pointer to a `void` function has no value, as a direct void call has none
  (`cb();`, `c ? cb() : (void)0`, `return cb();` in a void function). `c ? f : g` whose arms are
  function designators or function-pointer objects is a pointer to their one function type; arms that
  point to functions of different types are refused. The constant `0` compared with a pointer by `==`
  or `!=`, passed through a function pointer to a pointer parameter, or given to `free` and as
  `realloc`'s pointer is a null pointer of that type.
- Tables of function pointers, typedef'd or spelled inline (`uint32_t (*t[N])(uint32_t)`, up to three
  dimensions), local or file-scope, struct members, pointers to them (`op_t *p`, `uint32_t (**p)(uint32_t)`) and
  parameters of them; their elements read through `t[i]`, `*(t + i)` and `*p` as the function pointers they are, and
  a member read as one. A call takes any expression whose value is a function pointer: `t[i](x)`, `(*t[i])(x)`,
  `p->fn[i](x)`, `(c ? f : g)(x)`, `_Generic(...)(x)`, and `(*fp)(x)`, `(**fp)(x)` or `(*f)(x)`, whose `*` names the
  function again (C11 6.5.3.2p4). Each operator takes only the operand C allows -- a call a function pointer, `*` and
  `[]` a pointer to an object or an array, `.` a struct or union, `->` a pointer to one -- and anything else is refused
  for one reason on both rails (`called object is not a function or function pointer`, `dereference of a
  non-pointer`, `subscripted value is not an array or a pointer to an object`, ...), as is storing to or stepping a
  function (`a function designator is not an lvalue`) and `sizeof` of one.
- The unary operators as C types them, under `sizeof`, `typeof` and `_Generic` too: `+a` is `a` promoted (`sizeof(+c)`
  of a `char` is 4), `-z` and `~z` of a complex are complex, `__real__` and `__imag__` of a complex its element type
  and of a real operand its own, a bit-field operand the type of its value (`int` when narrower), `&x` a pointer to
  `x`'s type. Each takes only the operand C gives it -- `+` and `-` an arithmetic one, `~` an integer one or a complex,
  `!` a scalar -- and anything else is refused (`invalid argument type to unary expression`; a struct or union for
  the reason every operator gives). `if`, `while`, `for`, `do` and `?:` take a scalar, `switch` an integer
  (`statement requires expression of integer type`); `++` and `--` of a struct are refused as any operator's operand;
  `++(x)` is `++x`, and an array compared with 0 is the pointer it decays to. A void expression is used only where C
  discards its value -- an expression statement, `(void)e`, an operand of `,`, an arm of `?:`, a `_Generic`
  association, a statement expression's last statement, `return f();` in a void function -- and its value used
  anywhere else is refused (`the value of a void expression is used`).
- A function returning a function pointer, declared with a typedef for its return type (`op_t pick(uint32_t s);`):
  its call is that function pointer -- held, compared, selected, returned, or called at once (`pick(s)(x)`,
  `(*pick(s))(x)`) -- whether the function is defined, declared by a prototype, or reached through a pointer to it.
  A call through a function pointer returns what the function's type says: a pointer (`T *(*pf)(T *)`, so
  `*h(&v)` and `h(s)->v`), a struct whose member is read (`m(s).a`, `(*m)(s).b`, `o.mk(s).a`), or a function pointer.
  A pointer to a variadic function (`uint32_t (*g)(uint32_t, ...)`, a typedef, a member, a table of them) is declared
  and called, and a variadic function named as an arm of `?:` is a pointer to it.
- A function the unit only prototypes -- another unit defines it -- is a value as a defined one is: passed, held,
  selected, stored in a member or a table, compared, its address taken. `&f` of any function is the same pointer
  as `f`, `*&f` and `&*f` name it again, and `(&f)(x)` is the call `f(x)`. Each emit declares every function it
  names `extern` as its prototype does, a variadic one with its `...`. A function is no object: `f = g`, `f++`,
  `++f` and `f += 1` are refused (`a function designator is not an lvalue`), as `*f = v` is.
- Qualifiers below a type's top level -- what a pointer points to and each `*` under the outermost (`const char *const
  *v`, `char *restrict *`), a `const` pointer typedef (`const str_t *`), and the qualifiers of a function pointer's
  own parameters and return -- are kept: every `extern` declaration and function-pointer type the emit spells is the
  prototype's, and two function types that differ only in a qualifier are two types (`?:` of them is refused). The
  emit spells its own objects without qualifiers, so where C converts none -- `char **` passed to `const char *const
  *`, a `const T *` result -- a call casts.
- Casts to a function-pointer type -- spelled inline (`(uint32_t (*)(uint32_t))f`), through a typedef (`(op_t)f`),
  or a pointer to one (`(uint32_t (**)(uint32_t))p`) -- and `0` cast to any pointer type, the null pointer of that
  type. `_Atomic` starts a type name, and a pointer to an `_Atomic` member at its byte offset,
  `*(_Atomic T *)((char *)p + K)`, reaches the member as one atomic operation, a compound assignment and a step
  included. With the function-pointer store through a pointer to its own type, `*(R (**)(P))((char *)p + K) = f;`
  (a generic `*(void (**)(void))((char *)p + K) = (void (*)(void))f;` reads too), these are the forms each emit
  writes for a member it reaches at its byte offset, so the emit of a unit with a function-pointer or `_Atomic`
  member reads back.
- A typedef of a table of function pointers or of a pointer to one (`typedef uint32_t (*tab_t[2])(uint32_t);`, `pp_t`
  of `uint32_t (**)(uint32_t)`) as a local, a global, a parameter or a member; a compound literal of function
  pointers, called through (`(op_t[2]){f, g}[i](x)`); `__typeof__` of an element or a member that is a function
  pointer (`__typeof__(t[0])`, `__typeof__(o.fn)`); a braced function-pointer initializer (`op_t g = {f};`, `= {}`,
  C11 6.7.9p11). `( E ) = v;` and `( E ) OP= v;` of an lvalue `E`: the parentheses change nothing (6.5.1p5).
- A unit that declares a `<stdint.h>` or `<stddef.h>` name itself (`typedef unsigned long size_t;`), as freestanding
  code does.
- **Array compound literals — the full surface:** 1-D scalar (indexed `(T[]){...}[i]`, sized + zero-fill
  `(T[N]){...}`, signed-element), **multi-dimensional scalar** `(T[A][B]){...}[i][j]` (incl. an inferred
  outer dim `(T[][N]){...}` and a designated outer `{[1]=..,[0]=..}`), **1-D aggregate-element**
  `(struct P[]){...}[i].field`, and **multi-dimensional aggregate-element** `(struct P[A][B]){...}[i][j].field`.
  `sizeof` of a sized one is the array's size (C11 6.5.2.5p4), parenthesized or not (`sizeof (T[3]){...}`). A
  compound literal of a pointer or function-pointer type (`(uint32_t *){&g}`, `&(uint32_t *){&g}`, `(op_t){f}(x)`)
  is an object of its type, and `0` or `{}` in one is a null pointer.
- Local array declarations with initializers, including nested-brace multi-dim (`T a[A][B]={{..},{..}}`),
  inferred-size (`T a[]={..}`), and array-of-structs (`struct P a[N]={{..},{..}}`).
- **Computed goto** — the GNU label-as-value `&&L` (a `void *`) and the indirect `goto *p`.
- The conditional operators as C evaluates them: `?:` evaluates one arm, and `&&`/`||` their right operand
  only when the left one does not decide. An operand that can trap or change state (a division, a
  dereference, a call, a volatile read, an assignment) lowers as a branch, a pure one as a select; a
  conditional whose arms are void (`c ? f() : (void)0`, an `assert`) runs its arm for its effects.
- Members of array elements in every access form (read, store, compound assignment, increment, `&`):
  `a[i].m[j]`, `a[i].m.k`, `a[i].m.arr[j]`, on a local, global or pointer base.
- An object reached through a pointer in every access form (read, store, compound assignment, increment and
  decrement as a statement or a value, an assignment used as a value, `&`): a member through a pointer the lvalue
  loads (`h.next->v`, `n->next->v`, `s->p[i]`), an element of a pointer (`p[i]`), a dereference of any pointer value
  (`*p`, `*(p + i)`, `*&a`, `*(c ? &a : &b)`, `*p++`) and the first element of a member array (`*q->a`). A call
  through a parenthesized callee, `(fp)(x)` or `(o.fn)(x)`, is the call without the parentheses.
- The libc memory routines as external edges, opaque to R18 and linked with no flag: `<stdlib.h>`'s
  `malloc`/`calloc`/`realloc`/`aligned_alloc`/`free` and `<string.h>`'s `memcpy`/`memmove`/`memset`, each string
  routine returning its destination. A unit that defines one of these names -- before its call or after it --
  calls its own function.
- Functions declared by a prototype and defined later, or only prototyped (another unit defines them): a
  prototype may leave its parameters unnamed (`uint32_t g(uint32_t *, uint32_t);`). Each emit declares the unit's
  functions a function calls ahead of it, and a prototyped callee's `extern` declaration keeps a pointer
  parameter's `const` and spells a function-pointer parameter as C does.
- Structs and unions declared without a tag and named by a typedef (`typedef struct { ... } P;`, or only through a
  pointer, `typedef struct { ... } *PP;`), with nested anonymous members: the emit names each as C does -- by the
  typedef's name, `__typeof__(*(PP)0)`, or `__typeof__` of the member whose type it is.
- String/character literals (with prefixes), `static` locals, file-scope globals, `volatile` (MMIO). A string
  literal's element has its prefix's type (`char`, `char16_t`, `char32_t`, the target's `wchar_t`), read by `"ab"[i]`
  and `*("ab" + i)` alike; the pieces of one literal take the one prefix they carry (`"a" L"b"` is a wide literal),
  and pieces of two encodings (`u"a" U"b"`) are refused, as Clang refuses them. A plain `char` element, and
  `(char)v`, are `char` in both emits -- signed or not as the target's `char` is.
- File-scope declarations of several objects (`uint32_t a[3], b[2], *p;`, `static struct t { ... } x, y;`),
  character tables sized by their string literals (`char name[] = "bcir";`, `char name[] = ("bcir");` too), and a
  multi-dimensional global passed to a row-pointer parameter (`T (*p)[N]`, `T m[][N]`). A member of an element of a
  2-D or 3-D file-scope array of structs or unions (`gm[i][j].x`: read, stored, stepped, copied, its address
  taken) and an element of a 2-D table of pointers (`*gp[i][j]`, `gp[i][j][k]`). `*(p + i - j)` is the element
  `p + i - j` points at. A `const` global at any level (`const uint32_t k[3]`, `const char *const names[2]`), which
  both emits name through an lvalue of its unqualified type -- the emit's own objects carry no qualifiers.
- The linkable emit (`--linkable`) of the Python reference: the unit as one standalone translation unit -- its
  struct, union, enum and typedef definitions as the source spells them, in its order; every function declared
  before the globals; each global with its qualifiers and its initializer, a pointer's string literal, `&g[k]`, an
  array and a function among them as the address constants they are; and the headers its own text names (its
  copies' `<string.h>`, `<stdarg.h>`, `<stdatomic.h>`, `<complex.h>`). A global whose type is an untagged
  aggregate no typedef names is refused by name.
- File-scope initializers as a local's: nested braces, brace elision and designators for arrays of structs, rows
  and character tables, an unsized global sized by what its initializer reaches (`struct pt g[] = {1u, 2u, 3u,
  4u};` is two elements).
- `_Thread_local` globals and `static _Thread_local` locals: each thread has its own object, and both emits
  keep the storage class.
- The preprocessor: `#include`/`#embed`, conditionals, object/function-like + variadic macros, the
  predefined macros, `#line`, `_Pragma`, and the `__has_*` feature-test operators.

## Known limits

These are reported as diagnostics, or — with `--fallback` — as a fallback-to-LLVM signal:

- Non-constant `static`/global initializers; constructs beyond the L1–L6 statement subset.
- A function called, named as a value or used in a `sizeof` operand before any declaration of it -- C99 dropped
  the implicit declaration (C11 6.5.1p2): `call to undeclared function 'g'`, `use of undeclared identifier 'g'`.
  Declare it first with a prototype. A definition that leaves a parameter unnamed is refused as well.
- An integer constant no type in its list can hold: one past `unsigned long long`, and a decimal constant
  without `u` past `long long`, whose type C leaves to the implementation (GCC gives it `__int128`, Clang
  `unsigned long long`). Both rails refuse it (`an integer constant too large for every type its base and suffix
  allow`); write `9223372036854775808u`, or `INT64_MIN` as `-9223372036854775807 - 1`.
- An integer constant expression C requires a diagnostic for, or that is none. An enumerator no `int` holds,
  stated or counted on from `INT_MAX` (C11 6.7.2.2p2; C23 gives one a wider type, which neither rail models):
  `an enumerator value not representable as int`. A division or a remainder by zero, a signed overflow
  (`INT_MAX + 1`, `INT_MIN / -1`), a shift by the width or more, by a negative count, of a negative value or
  past its signed type (`-1 << 1` and `1 << 31`, which Clang folds without a word), and `sizeof`, `_Alignof`, the
  comma operator, a floating constant (`(int)1.5`), an object or a pointer in one: `not an integer constant
  expression`. A constant array dimension outside 0..`INT_MAX`: `an array dimension outside 0..INT_MAX`. Both
  rails refuse each for that one reason. A compound literal's dimension `(T[N]){...}` and a row pointer's
  `(*p)[N]` take an integer literal only, and an `enum` defined at block scope is refused.
- A file-scope initializer C refuses (an excess entry, a string too long for its array, a designator outside its
  object) or that overrides a subobject a brace list or a string initialized, and an initialized file-scope array
  of more than three dimensions. A block-scope `_Thread_local` object that is not `static` (C11 6.7.1p3).
- An identifier (a function, parameter, local, global, struct or union tag, member, typedef, enum constant or
  label) or a floating constant longer than 63 characters — C11 5.2.4.1's significant initial characters of an
  internal identifier. Both rails refuse it where it is lexed (`an identifier longer than 63 characters is not
  supported`, `a floating constant longer than 63 characters is not supported`): the C twin's claim graph holds
  63, and would otherwise have to cut the rest. The emitted C itself has no size limit. A macro name never
  reaches a lexer, so both preprocessors bound it where a directive reads it: a name `#define`, `-D` or `#undef`
  names, one `#ifdef`, `#ifndef`, `#elifdef`, `#elifndef` or `defined` tests, or one an evaluated `#if` or `#elif`
  looks up, past 63 characters is refused (`macro name is too long`), and so is a macro parameter (`macro
  parameter is too long`). A directive in a skipped group, or an `#elif` after a group was taken, is read only
  through its name (C11 6.10.1p6).
- An increment, or an assignment used as a value, of a device object -- a `volatile` object, or any member of a
  struct that holds volatile storage reached through a pointer: its value would be a second device access. Both
  rails refuse it; the statement forms (`dev->ctrl = v;`, `dev->ctrl |= m;`) lower.
- A `static` table of function pointers in a block (`static op_t t[2] = {f, g};`): refused on both rails as no integer
  constant expression; a file-scope table holds the same designators. Arithmetic on a function pointer (`fp + 1`,
  `fp++`), which C does not define, lowers today and is a recorded follow-up; `i[p]` (the pointer as the index) is
  refused -- write `p[i]`.
- A function declared to return a function pointer without a typedef (`uint32_t (*pick(uint32_t s))(uint32_t)`):
  refused on both rails (`a function returning a function pointer is not supported without a typedef`); declare
  its return type with a typedef. A function pointer given a function of another type (`op_t g = va;`), which C
  forbids, lowers today and is a recorded follow-up.
- A function declared inside a block (`uint32_t f(uint32_t s) { uint32_t g(uint32_t); ... }`): refused on both
  rails; declare it at file scope.
- A pointer object that is itself `volatile` (`T *volatile p`, `volatile str_t p` of a pointer typedef), whose every
  access C performs as written: refused on both rails (`a volatile-qualified pointer is not supported`); a pointer
  to volatile storage (`volatile T *p`) lowers. A `_Generic` association of a qualified type, which no controlling
  expression's type has: refused (`a \`_Generic\` association of a qualified type is not supported`). A qualifier on a
  `*` past the eighth from the base: refused (`a qualified pointer nested more than 8 deep is not supported`).
- A member access straight through an element that is a pointer (`arr[i]->m`, `pp[i]->m`, `o.p[i]->m`): refused on
  both rails (`unsupported base expression Index`); read the element first (`T *e = arr[i]; e->m`).
- A cast to an array type, which C forbids (6.5.4p2) -- `(uint32_t[2])s`, `(tab_t)f` of a table typedef: refused on
  both rails (`a cast to an array type`). A type name that names an identifier (`(uint32_t (*p)(uint32_t))f`,
  6.7.7p1): refused (`a type name names an identifier`). A compound literal of an `_Atomic` type, an `_Atomic`
  object (`&(_Atomic uint32_t){x}`): refused (`a compound literal of \`_Atomic\` type is not supported`). A cast to an
  `_Atomic` type (`(_Atomic uint32_t)x`, a typedef of one, under `sizeof` or `typeof` too): refused (`a cast to an
  \`_Atomic\` type is not supported`) -- C17 6.5.4p5 gives it the unqualified type, as GCC does, but Clang types it
  `_Atomic` and rejects it as an operand; a cast to a pointer to an `_Atomic` object lowers. A braced
  scalar or function-pointer initializer of more than one expression, or nested braces: refused (`a braced scalar
  initializer holds one expression`).
- `sizeof` of a compound literal its initializer sizes (`sizeof((uint32_t[]){1u, 2u})`, which C sizes as two
  elements): refused on both rails (`sizeof of an incomplete type`) -- a recorded follow-up; give the literal its
  dimension. A call through a function-pointer member of an indexed struct element (`a[i].fn(s)`): refused on both
  rails; take the element first (`struct ops *e = &a[i]; e->fn(s)`). Two splits recorded for follow-up: reading
  that member as a value (`op_t g = a[i].fn;`), and `__typeof__` of a call, of `?:` or of a function designator,
  are refused by the Python reference as not yet supported and lowered by the C twin.
- An `enum` tag used before its enumerator list, or with none -- an object, a cast, `sizeof`, a typedef, a parameter,
  and a pointer to the incomplete type too, which C allows -- is refused on both rails (`an enumerated type with no
  definition`); define the enumeration first.
- `#if` evaluates in each rail's own integers, not in C's `intmax_t` and `uintmax_t`: an unsigned operand does not
  make a comparison unsigned (`#if -1 > 0u` takes the `#else` branch), a character constant is 0 there, and the
  Python reference refuses `#if 0xFFFFFFFFFFFFFFFF == -1` with an internal error -- a recorded follow-up.
- The linkable emit's definitions drop a parameter's qualifiers below its top level (`uint32_t f(const uint32_t
  *p)` is defined taking `uint32_t *`), so a function pointer of the source's type takes such a function only
  through a cast; and the C twin's `--linkable` emits the unit's functions alone. Both are recorded follow-ups.
- A subscript of a pointer member of an element of an array of structs (`gt[i].name[1]`), a pointer to an array of
  structs (`struct pt (*q)[3] = gm;`), a member through the address of an element (`(&gm[0][1])->y`), a bit-field
  of a 2-D array of structs and an element of a string literal plus an offset (`("ab" + 1)[i]`): refused on both
  rails; read the element, or the member, into a local first.
- 64-bit-integer **results** of a few `<math.h>` functions and pointer out-params are supported, but a
  general 64-bit *value* model and Windows/ILP32 *code generation* (vs. layout) are not.
- `_Decimal32`/`_Decimal64`/`_Decimal128` are **blocked, not unsupported in principle**: Clang 18
  cannot compile `_Decimal`, so the form is un-validatable under the Clang-equivalence methodology and
  is gated out until a `_Decimal`-capable reference compiler is available.
- Cross-target builds are layout-only here; running them needs a cross toolchain.

When in doubt, run `-fsyntax-only` (or `--fallback`) — the frontend names exactly what it can't do.
