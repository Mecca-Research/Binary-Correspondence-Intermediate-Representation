# The BCIR C subset against the runtime's own C, and the subsystems still to build (2026-11-10)

> Dated, non-normative audit. It answers two questions the owner asked:
> 1. The verified C subset that `bcir-cfront` and `bcir-cc` compile was built for drivers and
>    kernels. What are the next logical steps to expand it for database, data-serialization and
>    AI workloads?
> 2. The native C trainer was a major step. Which other subsystems does the build need now, and
>    which later?
>
> It follows the language placement settled in [`BCIR_NATIVE_PORT_AUDIT_2026-11-09.md`](BCIR_NATIVE_PORT_AUDIT_2026-11-09.md):
> Python and C for what runs on a target, C++ for the hosted toolchain. The subset is the C half
> of that.

## 0. The answers

**1. The verified subset parses 6 of the 33 runtime library units today, and three small
conformance fixes would bring that to 15.**
- **What blocks most units is ordinary C the subset does not yet read.** A macro call spread
  over two lines, a file-scope `static_assert`, and the standard macros `NULL`, `SIZE_MAX`,
  `UINT32_MAX` and `UINT64_C` all fail. These are not driver-only restrictions.
- **The fixes reach the serialization core.** With them, the ASN.1 core, PER, the binary
  records, the StreamPack codec and runtime, the executor, the ring, the control plane and the
  AI kernels' portable path also parse.
- **Ten units compile and pass R1–R18 end to end.** Most of the rest fail on table idioms that
  serialization code uses everywhere: static string tables, untagged structs, pointers to
  arrays and array parameters in function-pointer types.
- **The AI kernels sit beyond the subset**: threads, SIMD intrinsics and bf16 storage. They
  should stay outside it, compiled by clang and GCC and held by parity, as today.

The next steps are a six-rung ladder, CSUB-1 to CSUB-6 (§4), ordered by how much runtime each
rung brings under the verified compiler. One unit should change instead of the subset:
`bcir_per.c` recurses one level, against R18 and the repository's own rule that decoders keep an
explicit stack.

**2. Three subsystems are missing now, and five more come later** (§5). The ones needed now are
for the AI and database work already queued:
- **a deterministic counter-based random-number generator.** The C rail has none, so native
  weight initialization, sampling, dropout and data shuffling all depend on Python;
- **a native data loader** of token shards, batches and shuffling. The trainer takes token-id
  arrays from its caller;
- **hardware-accelerated digests**: SHA-256 and the CRC, with the portable code kept as the
  reference. Every content address, generation and signed plan pays for them.

The native tokenizers are already PORT-2.

## 1. The subset, as it stands

`bcir-cfront` (Python) and `bcir-cc` (its C twin, `bcir_cfront.c`) compile a subset of C23:
- through the BCIR claim graph;
- verified against R1–R18, with R21 advisory;
- emitting behaviour-equivalent C;
- handing anything outside the subset to LLVM under the fallback contract
  ([`../languages/CFRONT_GUIDE.md`](../languages/CFRONT_GUIDE.md)).

It reads fixed-width integers, floating types, structs, unions, enums, typedefs, `_BitInt`,
`_Atomic` types, `_Generic`, `typeof`, `#embed` and flexible array members. It also reads the
allocator family and `<string.h>`'s `memcpy`/`memmove`/`memset`, with pointer-bounds and
pointer-lifetime policies. By design (R18) it refuses recursion.

The runtime's own C, roughly 55,000 library lines, is compiled by clang and GCC. Nothing gates
it under the verified compiler, so how much of it the subset reads had never been measured.

## 2. Method

Three passes over the 33 library units of `runtime/c` (tests, fuzzers, C-front fixtures and the
UART simulator excluded):
1. **Syntax today:** `bcir-cfront -fsyntax-only` on the unchanged sources.
2. **Syntax past the shallow gaps:** on scratch copies, the multi-line static assertions were
   removed and a prelude defined the missing standard macros. This simulates CSUB-1 without
   building it.
3. **Full compile:** `bcir-cfront --fallback` on every unit that parsed, reading its per-function
   law report.

`bcir-cc` (the C twin) was run on the same small probes and agreed: the multi-line call, the
file-scope assertion and the missing macros are refused on both rails. Each "blocker" below is
the first one a unit reports; a unit can hold more behind it.

## 3. Results

| Units | Syntax today | Past CSUB-1 (simulated) | Full compile and R1–R18 |
|---|---|---|---|
| `bcir_crc32.h`, `bcir_sha256.c`, `bcir_oer.c`, `bcir_per_plan.c`, `bcir_xer.c`, `uart_driver.c` | pass | pass | **pass** |
| `bcir_asn1.c`, `bcir_encode.c`, `bcir_exec.c`, `bcir_control_plane.c` | fail (macros, multi-line call) | pass | **pass** |
| `bcir_binrec.c`, `bcir_runtime.c`, `bcir_ai_kernels.c` | fail | pass | fail: `INT64_C`, `UINT64_C`, `UINT32_C` read as calls to undefined functions (CSUB-1) |
| `bcir_ring.c` | fail | pass | fail: `atomic_load_explicit`, the `<stdatomic.h>` call API (CSUB-4) |
| `bcir_per.c` | fail | pass | fail: R18, `bcir_per_constrained` calls itself (fix the unit) |
| `bcir_ed25519.c`, `bcir_q4_kernel.c`, `bcir_decode.c`, `bcir_train.c` | fail | fail: `UINT64_C`, `NAN`, `PTRDIFF_MAX`, `INT_MAX` | — (CSUB-1) |
| `bcir_kplan.c`, `bcir_telemetry_frame.c` | fail | fail: other multi-line macro calls | — (CSUB-1) |
| `bcir_jer.c`, `bcir_artifact_bundle.c` | fail | fail: a block-scope `static const char *const names[] = {...}` table | — (CSUB-2) |
| `bcir_emit.c` | fail | fail: an untagged `static const struct { ... } t[]` | — (CSUB-2) |
| `bcir_verify.c`, `bcir_hydrate.c` | fail | fail: the member `char (*calls)[BCIR_CIR_NAME]` in `bcir_cir.h` | — (CSUB-2) |
| `bcir_handoff.c` | fail | fail: an array parameter in a function-pointer typedef (`bcir_shard_manifest.h`) | — (CSUB-2) |
| `bcir_provenance.c` | fail | fail: a string member in a static struct table | — (CSUB-2) |
| `bcir_tensor.c` | fail | fail: `_Alignas(64)` on a local; then threads and SIMD | — (CSUB-2, then outside) |
| `bcir_decoder_train.c` | fail | fail: `FLT_MAX`; then threads, SIMD and bf16 | — (CSUB-1, then outside) |
| `bcir_quarantine.c`, `bcir_llama.c`, `bcir_q8_model.c` | fail | fail: `<stdio.h>`'s `FILE` | — (CSUB-5, hosted) |

**Totals:** 6 of 33 parse today, 15 of 33 past the simulated CSUB-1, and 10 of 33 compile with
R1–R18 clean.

One more finding came out of reading the units. **Nine units each define their own
little-endian `rd16`/`rd32`/`rd64`, and four to six define `wr16`/`wr32`/`wr64`.** That is the
L14 shape, a repeated mechanism, and CSUB-3 retires it.

### 3.1 The native ports, measured (2026-11-10)

The six units PORT-1 and PORT-2a added were measured the same way, after they landed. CSUB-1 was
simulated as above, and `max_align_t` was supplied too.

| Unit | Syntax today | First blocker past CSUB-1 (simulated) |
|---|---|---|
| `bcir_sp_listing.c` (freestanding) | fail: the multi-line static assertions of `bcir_streampack.h` | a block-scope `static const char *const` table (CSUB-2) |
| `bcir_ab_listing.c` (freestanding) | fail: the same header | an untagged `static const struct { ... } t[]` (CSUB-2) |
| `bcir_bundle_cli.c` (hosted) | fail: the same header | an untagged struct table of `offsetof` values (CSUB-2) |
| `bcir_pack_cli.c`, `bcir_tokenize_cli.c` (hosted) | fail | `<stdio.h>`'s `FILE` (CSUB-5) |
| `bcir_bpe.c` (hosted) | fail: `max_align_t`, `NULL` and `offsetof` in `bcir_host_alloc.h` | an aggregate member of an array element, `a[i].m.f` or `a[i].m = v` (in no rung yet) |

None of the six parses today; none would compile with CSUB-1 alone. Three findings follow:
- **`bcir_host_alloc.h` blocks every hosted unit.** It uses `max_align_t` and `offsetof`, and
  every hosted unit includes it. CSUB-1 lists `offsetof` but not `max_align_t`, so CSUB-5 would
  not open any hosted unit without it. CSUB-1 now names both.
- **The record table is not in the ladder.** The front end refuses a member of a struct-typed
  member of an array element (`"array-of-structs non-scalar element field ... is not yet
  supported"`), whether read (`t[i].token.n`) or copied (`t[i].token = x`). This is the shape
  of every open-addressed table of records, and `bcir_bpe.c`'s vocab, merge and id tables are
  written in it. CSUB-2 now names it.
- **The subset fell further behind while the ports landed.** PORT-1 added two freestanding units
  outside it, and `bcir_bpe.c` carries the tenth copy of the little-endian reads. A ratchet that
  covers only existing units would not have caught either. The gate below therefore also holds
  new units.

## 4. The expansion ladder

Each rung lands on both rails at once, the Python front end and the C twin. That means parity
over the existing corpus, a Clang-equivalence witness per construct, a refusal for each case C
itself forbids, and RED before GREEN.

| Rung | What the subset gains | What it brings under the verified compiler |
|---|---|---|
| **CSUB-1** Conformance (**landed 2026-11-11**: `bcir-cc` compiles 21 of the runtime's 58 production units with R1–R18 clean, where it compiled 4; the ratchet below holds all 58. Both rails' preprocessors are now one engine, Prosser's algorithm with the argument prescan, one rescan and `#`'s spacing, which also closed four twin miscompiles §4.1 names) | Function-like macro calls whose arguments span lines (C11 6.10.3p10). `static_assert` and `_Static_assert` at file scope. Complete freestanding headers: `<stdint.h>`'s limits and its `INTn_C`/`UINTn_C` macros; `<stddef.h>`'s `NULL`, `offsetof`, `max_align_t` and `PTRDIFF_MAX` (`bcir_host_alloc.h` needs `max_align_t` and `offsetof`, §3.1); `<stdbool.h>`, `<limits.h>` and `<float.h>`; `<math.h>`'s `NAN` and `INFINITY`. | The serialization core: `bcir_asn1.c`, `bcir_binrec.c`, `bcir_encode.c`, `bcir_runtime.c`, `bcir_exec.c`, `bcir_control_plane.c` and `bcir_ai_kernels.c`'s portable path. It is also every program in these domains that includes a standard header. |
| **CSUB-2** Table idioms | Static tables whose initializers are address constants (C11 6.6p9), such as string tables. Untagged struct types in a declaration. Struct-typed members of array elements, read and copied (`t[i].m.f`, `t[i].m = v`): the record table (§3.1). String members in static struct tables. Pointer-to-array types (`T (*p)[N]`) as objects and members. Array parameters in function-pointer types. `_Alignas` on any object. Three more the ratchet found first in units the measurement had not reached: a member read through a local's address, `(&v)->m`, as a macro writes it (`bcir_kplan.c`); a row of a multi-dimensional array used as a value, `f(t[1])` of `fe t[4][10]` (`bcir_ed25519.c`); and the address-of, increment and assignment of an lvalue reached through a loaded pointer (`bcir_spm.c`, `bcir_decoder_gem.c`, both quarantine units). | `bcir_jer.c`, `bcir_emit.c`, `bcir_artifact_bundle.c`, `bcir_provenance.c`, `bcir_verify.c`, `bcir_hydrate.c` and `bcir_handoff.c`: the table-driven dispatch every codec and catalog reader is written in. |
| **CSUB-3** C23 bits and checked arithmetic | `<stdbit.h>` (endian queries, bit width, leading and trailing zeros, population count) and `<stdckdint.h>` (`ckd_add`, `ckd_sub`, `ckd_mul`), each with its exact C23 semantics. The master roadmap lists both as near-term gaps (§4.1). One shared `bcir_bytes.h` built on them replaces the nine copies of the endian helpers, byte-identical. | Varints, bitmaps, length fields and every bounds check of the L3 kind ("bounds live where the resource commits") written as checked arithmetic rather than hand-rolled comparisons. |
| **CSUB-4** The atomics API | `<stdatomic.h>`'s explicit-order calls: `atomic_load_explicit`, `atomic_store_explicit`, `atomic_fetch_add_explicit` and `atomic_compare_exchange_*_explicit`, with `memory_order`. They map onto the memory model the oracle already states (`lower.memory_model`) and the law rail's atomic orderings. | `bcir_ring.c` and the lock-free single-producer, single-consumer structures the AI system's services run on (`BCIR_AI_SYSTEM_ROADMAP.md` §5.2). |
| **CSUB-5** A hosted profile | A declared hosted class, not part of the freestanding subset: `<stdio.h>` streams and file I/O under the injected-allocator memory class (`../languages/C_MEMORY_DISCIPLINE.md`). Memory mapping goes through a declared host interface. | The loaders `bcir_q8_model.c` and `bcir_llama.c`, and the quarantine's recovery |
| **CSUB-6** AI numerics, at the source level | A bf16 storage type and `_Float16` with declared conversion semantics. `#pragma STDC FP_CONTRACT OFF`, so a kernel's no-contraction contract (M17) is stated in its source as well as its flags. C23 `unreachable()` and an assume form feeding verifier facts (the master roadmap's "verifier-fed assume"). | The scalar halves of the AI kernels. Threads, vector types and ISA intrinsics stay outside the subset under the fallback contract. The tensor pool and the AVX-512/AMX kernels are measured, parity-held runtime services, not verified-subset code. |

**Two things should not change:**
- **R18 stays.** `bcir_per.c` should drop its one-level recursion: the length of an
  indefinite-length constrained integer is itself constrained to at most 8, so a helper for the
  short case removes the self-call. The subset should not admit recursion. (Landed with CSUB-1:
  `read_length_octets`, held by a malformed-length witness on both rails.)
- **The fallback contract stays the boundary for vector code.** Admitting intrinsics would make
  the verifier speak for code it cannot model.

**A ratchet is the gate.** CSUB-1 adds a gate that runs `bcir-cc` over the runtime's
freestanding units and holds the set that compiles with R1–R18 clean. (Landed as
`tools/c/csub_ratchet.py`, the `csub_ratchet` section, over every production unit
`runtime/c/MEMORY_CLASSIFICATION.txt` lists, not only the freestanding ones: its record,
`runtime/c/csub_ratchet.json`, names each blocked unit's first refusal, construct and rung.) Each later rung grows the
set, and any shrink fails the gate. A unit added after the gate lands joins the set, or is listed
with its first blocker and the rung that lifts it, so the distance cannot grow unrecorded (§3.1). The interface phase gains a matching check: `cgen`'s
generated codecs (IF-4 to IF-6) compile under the verified subset, so a generated codec is held
to R1–R18 as well as to octet parity.

### 4.1 What CSUB-1 found (2026-11-11)

Bringing the units under the compiler drove the preprocessor harder than its corpus had. The twin's
engine rewrote a line until it stopped changing, with no hide sets and no argument prescan. It
miscompiled four forms that C defines:
- `f(1)(2)` of `#define f(x) (x) + f` was `(1) + (2) + f`, for `(1) + f(2)`;
- `f(f)(y)` of `#define f(x) x` was `y`, for `f(y)`;
- `f LP 1)` of `#define LP (` invoked `f`, a token C had already passed;
- a two-level `XSTR(__LINE__)` spelled `"__LINE__"`, and `#define H # x` spelled `""`.

It refused `#define EPERM EPERM`, as `<errno.h>` writes it, and any line that expanded past 8 KiB.
The twin now runs the oracle's engine, ported step for step. The suite's three pinned twin
divergences agree with Clang now, and the pins are gone.

Five more defects were on both rails or on the oracle:
- **Stringizing.** `#` spelled an argument as a line is written, so `S(a + b)` was `"a+b"` on
  both rails. Each token now carries the white space before it (C11 6.10.3.2p2).
- **A number before a character constant.** `1.5 'c'` was written `1.5'c'`, which C23's digit
  separator reads as one pp-number.
- **`#if` spacing.** The twin's `#if` read `defined(A) defined(B)` as `11`.
- **Deep arguments.** The oracle crashed with a `RecursionError` on arguments nested some 400
  deep.
- **Integer libm results.** The twin read a negative `ilogb` or `lround` result as unsigned.

Both rails now hold one bound for an argument's prescan (63 deep) and one work bound for an
expansion, counted at the same events. A generated differential holds them to one answer: the same
preprocessed text, or the same refusal. Where Clang and GCC agree, the oracle also gives their
tokens.

Left open. The rails split on the first two, but only by one refusing what the other takes; none
is a miscompile:
- the twin holds 1 024 macros, and the oracle any number;
- the twin refuses a computed `#include MACRO`, which the oracle expands;
- `sizeof` of a block-scope object in an integer constant expression is refused on both rails;
- both PER rails accept an indefinite-length integer spelled in more octets than its value needs,
  a second spelling of one value (`bcir_per.c` and `bcir/asn1/per.py` alike).

## 5. Subsystems

**Built.** Every native subsystem the build has today:

| Domain | Units |
|---|---|
| The planner, its artifacts and execution | the native planner and hydrate (`bcir_kplan.c`), the executor, the control plane, the live ring, the data-plane hand-off |
| Codecs | the codec family: X.690, PER, OER, XER, JER and the plan emitter |
| Provenance and signing | the provenance digest, plan signing (Ed25519, SHA-256, HMAC) |
| Telemetry | the telemetry frame and envelope |
| AI | the tensor library and its thread pool, the Q8 and Q4 kernels, BCIRQ8 inference (`bcir_llama.c`), the native decoder trainer |
| Compiler and build | the C front end, BCIR Make |

**Missing now** (the work already queued needs them):

| Subsystem | Why now | Shape |
|---|---|---|
| **A deterministic counter-based RNG** (a Philox-style generator: key plus counter, no hidden state) | The C rail has none; the only generator is in a test. Native weight initialization, temperature and top-k sampling, dropout and shuffling all go through Python. Counter-based generation makes any element reproducible at any thread count, the property the trainer already holds for its arithmetic. | Freestanding C. Held to a Python twin over published known-answer vectors. |
| **A native data loader** | The trainer's API takes token-id arrays (`bcir_decoder_forward`). Tokenizing, batching and shuffling happen in Python. | Memory-mapped token shards, generation-addressed. A deterministic shuffle on the RNG above. Batches written straight into the trainer's arena. Tokenizing at ingest uses PORT-2's tokenizers. |
| **Hardware digests** | SHA-256 and CRC-32 sit under every content address, generation, StreamPack and signed plan. CRC-SLICE took the portable half of the CRC work: slicing-by-8 brought the CRC from 7–11 times zlib's time to about twice ([`measurements/2026-10-25/crc-slice/`](measurements/2026-10-25/crc-slice/README.json)). The hardware half is open. | SHA-NI and the ARMv8 SHA-2 instructions; carry-less-multiply CRC folding. Each beside the portable code, which stays the reference, and selected by a probe the parity gate covers. |

**Next**, with the AI system's slices and the interface phase:
- the store role's C reader for catalog generations: columns, postings, the sorted index and
  the locator over mapped generations (AIS-2);
- batched masked top-k (AL2);
- the fused text pipeline (AL3);
- equi-joins (AL1);
- generated codecs from `cgen` (IF-4 to IF-6);
- the crypto provider façade (IF-19).

**Later**, each gated on its consumer:
- native serving: paged KV, batched decode and sampling on the RNG above;
- compression for generations and checkpoints, wrapped from an established library rather than
  written;
- the network stack over virtio-net (IF-24, IF-25, driver row 3);
- GPU and NPU execution through vendor runtimes (driver row 12);
- distributed collectives in the C++ seam.

## 6. Order

| Item | When |
|---|---|
| CSUB-1 and the ratchet gate | Landed 2026-11-11, between interface-phase slices, with PORT-2 |
| The `bcir_per.c` recursion fix | Landed with CSUB-1 |
| CSUB-2 and CSUB-3 | With the interface phase's stage A, which is when generated codecs start passing through the verified compiler |
| CSUB-4 | Before the AI system's AIS-6, which runs services on SPSC rings |
| CSUB-5, CSUB-6 | When a hosted loader or a kernel's scalar half is wanted under the verifier |
| The RNG, the data loader, the hardware digests | Alongside PORT-2; the RNG first, because the loader's shuffle uses it |
