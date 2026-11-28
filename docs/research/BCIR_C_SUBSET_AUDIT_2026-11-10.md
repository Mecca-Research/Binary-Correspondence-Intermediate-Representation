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
  should stay outside it, compiled by clang and GCC and held by parity, as today. (Their scalar
  halves are inside it now: `bcir_tensor.c` and `bcir_decoder_train.c` compile clean, CSUB-6a,
  §4.8.)

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
| **CSUB-2** Table idioms (**landed 2026-11-27**: CSUB-2a 2026-11-14; CSUB-2b, CSUB-2c's first two slices, CSUB-2d's two and CSUB-2e's first 2026-11-15, CSUB-2c's third and CSUB-2e's second 2026-11-16, CSUB-2e's third to fifth 2026-11-20, the close-out 2026-11-27, §4.2; 32 of the 58 units compile clean by it, and no unit's first refusal is the rung's) | Static tables whose initializers are address constants (C11 6.6p9), such as string tables. Untagged struct types in a declaration. Struct-typed members of array elements, read and copied (`t[i].m.f`, `t[i].m = v`): the record table (§3.1). String members in static struct tables. Pointer-to-array types (`T (*p)[N]`) as objects and members. Array parameters in function-pointer types. `_Alignas` on any object. Four more the ratchet found first in units the measurement had not reached: a member read through a local's address, `(&v)->m`, as a macro writes it (`bcir_kplan.c`); a row of a multi-dimensional array used as a value, `f(t[1])` of `fe t[4][10]` (`bcir_ed25519.c`); the address-of, increment and assignment of an lvalue reached through a loaded pointer (`bcir_spm.c`, `bcir_emit.c`, both quarantine units); and a struct returned by a call through a prototype (`bcir_handoff.c`). | `bcir_jer.c`, `bcir_emit.c`, `bcir_artifact_bundle.c`, `bcir_provenance.c`, `bcir_verify.c`, `bcir_hydrate.c`, `bcir_handoff.c` and `bcir_ed25519.c`: the table-driven dispatch every codec and catalog reader is written in, and the field arithmetic of the signature code. |
| **CSUB-3** C23 bits and checked arithmetic (**landed 2026-11-27**: `<stdckdint.h>`, `<stdbit.h>` and the one `bcir_bytes.h`, §4.4) | `<stdbit.h>` (endian queries, bit width, leading and trailing zeros, population count) and `<stdckdint.h>` (`ckd_add`, `ckd_sub`, `ckd_mul`), each with its exact C23 semantics. The master roadmap lists both as near-term gaps (§4.1). One shared `bcir_bytes.h` built on them replaces the nine copies of the endian helpers, byte-identical. | Varints, bitmaps, length fields and every bounds check of the L3 kind ("bounds live where the resource commits") written as checked arithmetic rather than hand-rolled comparisons. |
| **CSUB-4** The atomics API (**landed 2026-11-27**, §4.6) | `<stdatomic.h>`'s explicit-order calls: `atomic_load_explicit`, `atomic_store_explicit`, `atomic_fetch_add_explicit` and `atomic_compare_exchange_*_explicit`, with `memory_order`. They map onto the memory model the oracle already states (`lower.memory_model`) and the law rail's atomic orderings. | `bcir_ring.c` and the lock-free single-producer, single-consumer structures the AI system's services run on (`BCIR_AI_SYSTEM_ROADMAP.md` §5.2). |
| **CSUB-5** A hosted profile (**four slices landed**: the `<string.h>` reads 2026-11-20, the hosted headers 2026-11-27, the tools' next forms and `atomic_flag` 2026-11-28, §4.3; 61 of the 68 units compile clean) | A declared hosted class, not part of the freestanding subset: `<stdio.h>` streams and file I/O under the injected-allocator memory class (`../languages/C_MEMORY_DISCIPLINE.md`). Memory mapping goes through a declared host interface. The `<string.h>` reads -- `memcmp`, `strcmp`, `strncmp`, `memchr`, `strchr`, `strrchr`, `strstr` and `strlen`, freestanding since C23 -- landed first: `bcir_decoder_gem.c`, `bcir_decoder_pack.c`, `bcir_verify.c`, `bcir_spm.c`, `bcir_bpe.c` and `bcir_diag.c` stopped at one. | The loaders `bcir_q8_model.c` and `bcir_llama.c`, and the quarantine's recovery |
| **CSUB-6** AI numerics, at the source level | A bf16 storage type and `_Float16` with declared conversion semantics. `#pragma STDC FP_CONTRACT OFF`, so a kernel's no-contraction contract (M17) is stated in its source as well as its flags. C23 `unreachable()` and an assume form feeding verifier facts (the master roadmap's "verifier-fed assume"). A call of more than 16 arguments: the claim graph's bound on a claim's reads (`BCIR_CALL_MAX_ARGS`, CF-CALLARGS), which an attention kernel's workspace call passes with 18 (`bcir_decoder_train.c`). (Landed as CSUB-6a, §4.8: 24 arguments on both rails.) | The scalar halves of the AI kernels. Threads, vector types and ISA intrinsics stay outside the subset under the fallback contract. The tensor pool and the AVX-512/AMX kernels are measured, parity-held runtime services, not verified-subset code. |

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
- the twin holds 1 024 macros, and the oracle any number (closed 2026-11-20 by CF-MACROCAP: one
  bound of 65 534 on both rails);
- the twin refuses a computed `#include MACRO`, which the oracle expands (closed 2026-11-20 by
  CF-INCLUDEX: both rails expand it);
- `sizeof` of a block-scope object in an integer constant expression is refused on both rails;
- both PER rails accept an indefinite-length integer spelled in more octets than its value needs,
  a second spelling of one value (`bcir_per.c` and `bcir/asn1/per.py` alike) (closed 2026-11-20 by
  PER-MIN: both rails refuse it, and five more longer spellings in clause 11, under both variants).

### 4.2 CSUB-2, slice by slice

The rung lands in slices, each on both rails, with its fixture, its refusals and its RED sweep. The ratchet's record
moves with each: a unit's first refusal is the next construct the rung must take.

**CSUB-2a (2026-11-14): two declarations.**
- **An array parameter of a function-pointer type** is a pointer to its element (C11 6.7.6.3p7), as a definition's
  is. It covers typedefs, members and inline declarators, sized or not. `bcir_shard_manifest.h`'s
  `const uint8_t digest[32]` had been a parse error. An array of arrays and a variable-length one are refused.
- **`_Alignas(N)` on an object.** It covers locals, statics and file-scope objects, and each emit spells it where
  the object is declared. Both rails refuse alike what C forbids or the emit cannot spell:
  - an alignment that is no power of two, is past 4096, or is weaker than the type;
  - the type-name form;
  - one on a typedef, a parameter, a function, a type name, a VLA, or a declaration that defines the struct it
    aligns.

  The oracle had read a file-scope one and dropped it.

`bcir_shard_manifest.c` now compiles clean. `bcir_handoff.c` and `bcir_tensor.c` stop at the next constructs of
this rung: an element's address through a loaded pointer (`&t->slots[i]`), and a row of a two-dimensional array
used as a value.

**CSUB-2b, first half (2026-11-15): string tables.**
- **A string literal in a static's image.** A string literal is the address of its array, an address constant (C11
  6.6p9). It is now an entry of a static's image:
  - a block-scope `static const char *const names[]`;
  - a pointer member of a static table's records, by braces, brace elision or designators;
  - a scalar static.
- **One spelling.** Both rails render each entry by one canonical spelling: printable ASCII as itself, every other
  code unit and `"`, `\` and `?` an octal escape. So the image, and the digest over it, agree whatever escapes and
  pieces the source used, and the emit never spells a trigraph, which `-std=c11` reads.
- **Where it is taken.** Only where C converts the address without a diagnostic: into a pointer to plain `char`,
  from a literal of no prefix, and only the literal itself. An entry that merely evaluates to one, `("a", "b")`,
  is no constant. Every other slot and prefix is refused alike, for one reason.

`bcir_artifact_bundle.c`, `bcir_jer.c` and `bcir_sp_listing.c` now compile clean. `bcir_decoder_pack.c` stops at
an element's address through a loaded pointer (`&c->claims[v->claim_id]`).

**CSUB-2b, second half (2026-11-15): untagged struct tables.**
- **An untagged struct or union a block declaration defines**, `static const struct { ... } t[] = {...}`, as the
  runtime writes its small tables. It may be static or automatic, a union, hold an untagged member, take several
  declarators, or be a `for` declaration's.
- **How the emit names it.** Nothing else can name its type (C11 6.7.2.3p6), so each emit opens the function with a
  typedef of it, its members as the source spells them, under `__bcir_anonN`, a name reserved to the
  implementation. Every object, temp and copy of the type is spelled by that name.
- **What is refused.** A tag defined in a block is refused alike, since its scope would end with the block and the
  unit has one tag table. So is a struct or union defined anywhere else a type is named: a cast, `sizeof`, a
  parameter, a compound literal, `typeof`, `_Atomic(...)`, a member's qualified type.

`bcir_ab_listing.c` and `bcir_emit.c` now stop at the record table: a pointer member read from an array element,
`order[i].name` (CSUB-2c).

**CSUB-2c, first slice (2026-11-15): pointer members of array elements.**
- **A pointer member of an array element is a value of its pointer type**: `order[i].name`, `d->notes[i].message`,
  and `set->slots[i].name.s` at a nested member's depth. Each rail loads it as an element's member is loaded,
  strided by the element. It is then compared, passed, subscripted, dereferenced and followed with `->`, stored and
  stored through, as a plain struct's pointer member is. Both rails had refused it as a non-scalar element field.
- **Stepping, and addresses.** An increment through one (`t[i].next->v++`) is lowered. The member's own address is
  a `T **`, and the address of an object it reaches a `T *`. Stepping the member itself (`t[i].p++`) is refused
  alike, as a plain struct's pointer member's step is.
- **What stays refused.** A function-pointer, bit-field or struct member of an element. What C forbids after an
  element's member is refused for C's reason on both rails: a subscript or `->` of a member that is no pointer, a
  `.` of one that is, `->` through a pointer to no struct.
- **What the witnesses found.** Five twin defects, three of them invalid C, which the digest cannot see because it
  carries no declared type:
  - `&s->name` of a `const char *` member was typed `int8_t **`, and `&s->pp` of a `T **` member `T **`;
  - a function-pointer member of an element was loaded as an integer, then assigned to the function pointer, and
    its address was typed an integer's;
  - `&s.arr[i].in` of a struct member was taken, which the oracle refuses;
  - `->` through a pointer to no struct was refused for the twin's own reason, not C's.

  Each lowered form's emits are now also compiled strictly beside their source. The oracle, for its part, had
  refused a subscript of a member array element's scalar member, `s.a[i].x[0]`, with its own limit's text before
  C's reason.

`bcir_ab_listing.c` and `bcir_provenance.c` now compile clean: 27 of the 58 units. `bcir_bpe.c`, `bcir_diag.c` and
`bcir_emit.c` stop at an element's address through a loaded pointer (`&set->slots[j]`, `&d->fixits[i]`), a CSUB-2e
form. `bcir_decoder_train.c` stops at a call of 18 arguments, past the claim graph's 16 reads, which CSUB-6's row
now names.

**CSUB-2e, first slice (2026-11-15): struct elements through a loaded pointer.** Taken before CSUB-2c's struct
members, because it was the first refusal of five units and the struct members of none.
- **An element of a pointer to structs reached through a loaded pointer is an lvalue**: `&set->slots[j]`,
  `&d->fixits[i]`, `&c->claims[v->claim_id]`, `&t->slots[i]`. Its address is a pointer to the struct. A member of
  it can be stepped, assigned as a value, compound-assigned (an `_Atomic` one as one atomic step) or addressed. A
  store or step reaches through a pointer member it holds (`s->rows[j].buf[1] = v`). The oracle had lowered each.
  The twin's lvalue walk through a loaded pointer stopped at a struct element. Its store through one asked for an
  `=` after a pointer member, a gap CSUB-2c had opened.
- **`->` of a struct element.** `a[i]->m`, where `a[i]` is the struct itself, is refused on both rails for C's
  reason (6.5.2.3p2). Both had read it as `a[i].m` and emitted C no compiler takes, in every position: a read, a
  store, a step, an address, a member array's element.
- **What stays refused**, for one reason on both rails: a subscript of the element, a bit-field member's address,
  the struct element stepped or assigned as a value.

The count stays 27 of 58, but seven units move on:
- `bcir_bpe.c` and `bcir_diag.c` reach the record table's struct members: `set->slots[j].name = name` and
  `d->notes[i].span` passed by value.
- `bcir_emit.c` and `bcir_spm.c` reach two later CSUB-2e forms: `*low = *high = &absent`, and a pointer member's
  assignment used as a value, `!(t->pieces = ...)`.
- `bcir_handoff.c` reaches a struct returned by a call through a prototype, `bcir_ctl_admit_pack(...)`. The CSUB-2
  row now names it.
- The hosted tools `bcir_decoder_gem.c` and `bcir_decoder_pack.c` reach `strlen`, a hosted `<string.h>` call R18
  does not know. CSUB-5's row now names it.

**CSUB-2c, second slice (2026-11-15): struct members of array elements.**
- **A struct or union member of an array element is the aggregate value** on both rails: read, stored, copied
  element to element, passed and returned by value, and addressed (a pointer to the struct). This covers a direct
  array, a member array and a loaded pointer. Examples: `set->slots[j].name = name` (`bcir_bpe.c`) and
  `d->notes[i].span` passed by value (`bcir_diag.c`). Both rails had refused it as a non-scalar element field.
- **What stays refused.** An aggregate member assigned as a value, `x = (t[i].sp = v)`, is refused for the oracle's
  reasons on every path. The twin had lowered it on a direct array. The aggregate as an operand (`t[i].sp++`,
  `t[i].sp == u`) is refused for C's reason. A bit-field, function-pointer or array member of an element stays
  refused, as before.

The count stays 27 of 58. `bcir_bpe.c` moves on to a struct member's address through a loaded pointer,
`&s->t->arena`, a CSUB-2e form. The hosted `bcir_diag.c` moves on to `strlen` (CSUB-5).

**CSUB-2d, first slice (2026-11-15): pointer-to-array members.**
- **A member `T (*m)[N]` is a pointer to a row of N scalars** on both rails. `bcir_cir.h` declares one,
  `char (*calls)[BCIR_CIR_NAME]`, and it was the first refusal of the five units that include it. The member is laid
  out as a pointer and typed as a `T m[][N]` parameter is, so `p->m[k][j]` is element `k*N + j` of the flat pointer
  it is loaded as. The element is read, stored, stepped, compound-assigned and addressed, and `sizeof *p->m` is the
  row. The pointer is tested for null, copied, taken from a two-dimensional array and passed to a row-pointer
  parameter, and `typeof(p->m)` declares one. This covers a struct value, an element of a member array, a loaded
  pointer and a union. Both rails had refused the declaration.
- **A defect on both rails at the base.** `*rows` of a row-pointer parameter and `*m` of a two-dimensional array,
  used as a value, are the row they designate, but both rails read its first element: `char *r = *rows` became an
  assignment of a `char` to a `char *`, which Clang 23 refuses and older compilers only warn of. Both now refuse
  each as a row used as a value; `(*m)[i]` stays the element it is.
- **What stays refused, for one reason each:**
  - a row used as a value (`p->m[k]`, `*p->m`);
  - arithmetic on the pointer, and a step of it;
  - a row of no scalar (a struct, a pointer, a function pointer, a typedef'd array), of a volatile, `_Atomic` or
    `_BitInt` scalar, of two dimensions, of no size, or longer than 65535;
  - one at file scope or in a typedef, which the oracle had read as an array of rows.

The count is now **29 of 58**: `bcir_hydrate.c` and `bcir_plan.c` compile clean. `bcir_verify.c` and
`bcir_channel.c` stop at a row used as a value, `f->calls[k]` and `seen_keys[i]`. `bcir_cc.c` stops at `FILE`
(CSUB-5).

**A struct returned through a prototype (2026-11-15).**
- **A struct or union a call returns through a prototype** -- a callee another unit defines, as `bcir_handoff.c` calls
  `c = bcir_ctl_admit_pack(...)` -- is the aggregate value on the twin, as a defined callee's is. It is initialized
  from, assigned, read a member of in place (`mk(x).b`), passed and returned on by value, stored, and chosen by `?:`.
  The oracle had lowered it and the twin had refused it, the one form of `bcir_handoff.c` the twin refused.
- **A gap at the base.** The twin had read no `->` or `[]` after a pointer a prototyped callee returns (`mkp(x)->a`,
  `arr()[2]`), which the oracle lowered. A call's postfix now reads a prototype's return as a definition's.
- **What stays as it was.** A prototype whose return type is qualified at its top level, `const uint32_t g(void);`,
  is declared without the qualifier in both emits, which Clang refuses beside the source. No runtime unit declares
  one.

The count is now **30 of 58**: `bcir_handoff.c` compiles R1–R18 clean.

**CSUB-2d, second slice: a row used as a value (2026-11-15).**
- **A row of scalars used as a value** is the pointer to its first element on both rails, `&m[k][0]`, lowered as that
  form lowers: `f->calls[k]` (`bcir_verify.c`), `seen_keys[i]` (`bcir_channel.c`), `fsub(a, p[1], p[0])` of a
  `gf p[4]` parameter (`bcir_ed25519.c`) and `ys[v]` (`bcir_tensor.c`), the first refusal of all four. It is passed,
  assigned, initialized from, chosen by `?:`, stepped, compared, subtracted, sized and dereferenced (`*m[k]`,
  `*(m[k] + i)`, stored through), of a block static, a global, a VLA, a `T m[][N]` or `gf p[4]` parameter, a member
  array (of a value, through a loaded pointer, of an element of an array of structs, at three dimensions) and the row a
  pointer-to-array member points at.
- **Refused alike, for one reason each:** a row's address, `&m[k]`, a pointer to an array (`the address of a row of a
  multi-dimensional array is not supported`, the twin's words, which both rails now give); `typeof` of a row, the array
  it is; a row of rows, of structs or of pointers; and the row `*` designates (`*m`, `*rows`, `*p->m`), as before.
- **Defects found at the base and closed:**
  - The twin read `&g[k]` of a two-dimensional global and `&s->m[k]` of a member array as the address of element `k`
    of row 0, `&g[0][k]`, a miscompile, and the oracle stepped `&g[k] + 1` by one element. Both refuse the row's
    address now.
  - The twin read a member array's row used as a value, `f(s->m[k])` or `uint16_t *r = s.m[k]`, as its first
    element: an integer passed or assigned where a pointer is wanted, an emit no C compiler takes. The oracle had
    refused it. Both lower it as the row's pointer now.
  - The oracle typed `typeof(m[k])` as its element, so `sizeof` read 2 where C reads 6.
  - The twin typed the address of a volatile array's element, a volatile member and a volatile member array's element
    (`&a[i]`, `&s->x`, `&s->q[i]`, `&s->a[i].v`) a `T *`, which loses the volatile access a store through it makes.
    Each is a `volatile T *` now, as `&x` was.
- **Found and left, tracked:** `typeof` of an array object (`__typeof__(a) b` of `uint16_t a[3]`) declares one element
  of it on the twin and the array on the oracle, so `sizeof b` differs. No runtime unit uses `typeof`. (Closed
  2026-11-20 by CF-TYPEOFARR: both rails declare the array, and `DEVELOPMENT_HISTORY.md` records the six other
  `typeof` defects that slice found.)

The count is now **31 of 58**: `bcir_ed25519.c` compiles R1–R18 clean. `bcir_channel.c` and `bcir_verify.c` move on
to an array member of an element used as a value (`channels[best].name`, `f->qcasts[k].type`), CSUB-2c's next form;
`bcir_tensor.c` to a subscript of a pointer element of a member array through a loaded pointer (`j->pa[k][i] = v`).

**CSUB-2c, third slice: an array member of an element used as a value (2026-11-16).**
- **An array member of scalars of an array element used as a value** is the pointer to its first element on both
  rails, `&a[i].m[0]`, lowered as that form lowers: `strcmp(channels[i].name, channels[best].name)` (`bcir_channel.c`)
  and `fixed_string(f->qcasts[k].type, sizeof f->qcasts[k].type)` (`bcir_verify.c`), the first refusal of both, which
  each rail had refused as `array-of-structs non-scalar element field`. It is passed, assigned, chosen by `?:`,
  compared, subtracted, stepped, dereferenced (`*a[i].m`) and sized, of a block static's element, of a pointer
  parameter's (`p[k].name`), through a loaded pointer member (`f->qcasts[k].type`) and a `.` hop on (`g[1].m.arr`).
  - The oracle names the form in `_elem_member_array` (a `.` chain from a subscript whose base is a name or a pointer
    member) and takes it in `_scalar_row`; the value, `*` and `typeof` sites read it as they read a row.
  - The twin takes the member array itself, `a[i].m`, in `aos_member_array_x` with `pad` 2, at the read, the
    loaded-pointer and the assignment-target sites.
- **Refused alike, for one reason each, as before:** its address, `&a[i].m`, a pointer to an array; `typeof` of it
  (until CF-TYPEOFARR, 2026-11-20: the member's array on both rails since); a
  member array of structs, of pointers or of two dimensions used as a value; and an array member of an element of a
  member array, `h->arr[k].m` and `f->q2[k].t`.

The count stays **31 of 58**. `bcir_channel.c` and `bcir_verify.c` move on to `memcmp` and `memchr`, `<string.h>` calls
R18 does not know yet (CSUB-5).

**CSUB-2e, second slice: volatile objects stepped, and reached through a pointer sum (2026-11-16).**
- **A step of a volatile scalar is one read and one write on both rails**, to one digest. Both quarantine units stop
  first at `bcir_quarantine.h`'s `return (*c)++;` of a `volatile unsigned long *`. The step may go through a pointer
  (`*c`, `c[k]`, `*(c + k)`, `**pp`), or be of a volatile array's element, of a volatile member or a plain member of
  a device struct, of a volatile struct's member, member array element or array element, or through a loaded pointer
  into a device struct (`h.d->v++`, `t[i].p->r++`). A named object steps as a statement or a value.
- **Refused alike, for one reason each:**
  - a prefix step that narrows (`++*c` of a `volatile uint8_t *`), whose value would re-read the object, a second
    access;
  - a bitfield of a device region;
  - an assignment or a compound assignment through a pointer to volatile storage used as a value.
- **Defects found at the base and closed:**
  - **A miscompile on the twin.** The twin typed a pointer sum of a pointer to volatile storage a `T *`
    (`*(c + k - 1u)`, `*(k + c)`, `volatile uint32_t *q = c + k`). Every read and store through it lost its volatile
    access, behind a warning only. It is a `volatile T *` now, as `&a[i]` is.
  - **On the oracle, a named volatile object's postfix value read it twice**, once for its snapshot and once for the
    step. It reads it once now.
  - **The twin's `gv++;` claimed neither of its device accesses**: it read the object inside the add and left the
    store unmarked.
  - **The twin loaded a pointer member of a volatile struct's element into an integer temp**, an emit no compiler
    takes (`p[k].q[1]` of a `volatile struct s *`).
  - The oracle had refused every step of a device object but a named one's. The twin had lowered the members and
    refused the rest. The members included two forms the oracle refused: a device struct's bitfield and a narrowing
    prefix step.

The count stays **31 of 58**. Both quarantine units move on to `<stdio.h>`'s `FILE` (CSUB-5). Measured again on
2026-11-16:
- Of the forms below, the oracle alone lowers `&s->t->arena` and a member array's element through a loaded pointer
  (`p->q->arr[1]++`, of a device struct too).
- Both rails refuse the rest.

**CSUB-2e, third slice: a member through an object's own address (2026-11-20).**
- **`(&v)->m` is `v.m` on both rails** (C11 6.5.2.3p4, 6.5.3.2p3), and `(&*p)->m` is `p->m`. That covers a local, an
  element, a global's element, a member, a device member and macros (`bcir_kplan.c`'s `MS_OP(&v, i)`), at any depth
  of parentheses. The oracle folds the parse (`_name_rooted`), and the twin's CF-PAREN pass folds the tokens
  (`unparen_body`). Both had refused it.
- **On the twin, a parenthesized operand of `&` or `*`** -- `&(x)`, `&((x).m[1])`, `*(p)` -- is the operand, as the
  oracle's parse never kept the parentheses. The twin had refused `&(x)` (`unsupported address-of`) and read
  `(*(p)).n` as a struct converted to a scalar. An inner run of `(` now closes with its own closers, so `(&(x))->n`
  and `(++(y))` lower, where the twin had refused the whole run.
- `(&*&*p)->m` and `(&**q)->m` stay refused on both rails, in different words.

The count is **32 of 58**: `bcir_kplan.c` compiles clean.

**CSUB-2e, fourth slice: a pointer assigned as a value, and a for-step as a statement (2026-11-20).**
- **A pointer assigned as a value is stored, then reloaded, as a scalar one is**, on both rails:
  - a member, of a value and through a pointer (`!(t->pieces = (piece_t *)...)`, `bcir_spm.c`);
  - an element of an array of pointers, of function pointers or through a `T **` (`r = (v[1] = x)`), a null pointer
    stored as the element's type;
  - a dereferenced `T **`, of a scalar or a struct pointee, in a chain (`*low = *high = &absent`, `bcir_emit.c`);
  - a function-pointer member, which the oracle alone had refused.
- **What stays the oracle's follow-on is refused for its reason on both rails**: a pointer's compound assignment as a
  value (a member, an element, `*pp`), and an assignment into device storage as a value, whose reload would be a
  second access. The twin had refused the first in its own words (`)`). It lowered the second with a volatile
  reload, an access the program does not make: `(t->n = 1)` through a `volatile struct t *`, of a volatile struct
  local, or of a struct holding volatile storage.
- **CF-STEP: a for-step is a statement on the twin**, as the oracle's `_for_step` lowers it. Probing the refusals
  above found the twin reading a step through the expression grammar: of 21 steps, 12 disagreed with the oracle.
  - A stored member or element was reloaded.
  - A pointer member's `+=` was left unread. `for (t->p = buf; t->p != buf + 4; t->p += 1)` emitted a loop that
    never advanced.
  - A step read in part was lowered with the rest dropped: `(x) += 1u`, and `x++, s.p -= 0, s.p++`.
  - A volatile `*a += 1u` was refused, as a value is.

  A step piece that stores to a member, an element or through a pointer is now read by the statement grammar, with
  the piece's end as the statement's `;`. A step not read whole is refused. CF-PAREN's pass takes a step piece as a
  statement start, and so an unbraced statement after a condition, `else`, `do` or a for header: `(x) += 1u` there
  is `x += 1u`, as it is among a block's statements.
- A store through a call's result, `h()->n = 1u`, is a recorded split: the oracle lowers it, and the twin refuses it
  in every position.

The count stays 32 of 58, but two units move on:
- `bcir_spm.c` reaches `memcmp`, a hosted `<string.h>` call R18 does not know (CSUB-5).
- `bcir_emit.c` reaches R18's refusal of recursion. The emitter descends the schema tree recursively
  (`der_write_full` and `der_write_content`, `per_node`, `jer_node`). R18 stays (§4), so no rung lifts it: the unit
  needs an explicit stack, as `bcir_per.c` did. The ratchet records it `outside`.

**CSUB-2e, fifth slice: an object through a loaded pointer (2026-11-20).**
- **The address of a member of the struct a loaded pointer reaches** is taken on the twin as the oracle takes it, as
  `&s->m` of a named pointer is. That covers a struct member (`&s->t->arena`, `bcir_bpe.c`), a scalar or nested
  member (`&s->t->arena.used`), a pointer member, a member array's element of one or two dimensions, and a volatile
  member, a pointer to volatile. The twin's address of a named pointer's member is now one function
  (`addr_of_member`), and the walk through a loaded pointer ends in it. A pointer element's address, `&s->pp[j]`, is
  a `T **`.
- **A member array's element through a loaded pointer is an lvalue**: `s->t->arr[i]` stored, compound-assigned,
  assigned as a value and stepped, in a for-step too. An `_Atomic` one is one atomic step. The twin had refused each
  but the read.
- An assignment as a value into device storage through a loaded pointer is refused in the oracle's words. The twin
  had said `)`.

The count stays 32 of 58. `bcir_bpe.c` moves on to `memcmp` (CSUB-5).

**CSUB-2, the close-out (2026-11-27).**
- **The pointer an element of a member array of pointers holds is subscripted** on both rails: `j->pa[k][i] = v` of
  `uint16_t *pa[2]` (`bcir_tensor.c`) is `v[k][i]` of a local `T *v[N]`, one subscript past the array's dimensions
  indexing the loaded element. It is read, stored, compound-assigned, stepped, assigned as a value and addressed, of a
  two-dimensional member array, through a loaded pointer, of a union, of a struct, volatile or `_Atomic` pointee.
  Both rails had refused it. The twin reads it in one function, `load_member_ptr`, which every path through a
  member's pointer now takes.
- **A `T **` member's or element's pointer is subscripted in turn** on the twin, as the oracle subscripts it
  (`s.p[0][1] = x`, `s.pa[k][i][j]`; `ptr_elem_load`). Two twin defects at the base: a `T **` member's subscript
  chain was refused after one step, where the oracle lowered it, and a loaded `T **` member was typed a `T *` in the
  emit, which no compiler took.
- **`(*q)->m` through a dereferenced `T **`** (A6) is the twin's as it is the oracle's: read, stored,
  compound-assigned, stepped, assigned as a value and addressed, `(*q)->a[i]` and `(*q)->p[i]` too
  (`paren_deref_base`). The twin had refused every one.
- **A prototype whose return is qualified at its top level**, `const uint32_t g(void);`, keeps the qualifier in the
  twin's declaration of it, which Clang compares with the source's and had refused as conflicting.
- **What stays refused, in one set of words on both rails:** a member through a pointer element (`s.pp[i]->m`,
  as `pp[i]->m` is, CF-IDXARROW); `(&**q)->m`, an address no fold reduces to its object (the twin had said `;`); a
  subscript of a scalar element (`s.p[1][0]`, read or stored, where the twin had said `;` and `=`); and a pointer to
  an array as a block object, `T (*row)[N]`, initialized or not (the twin had said `expected declarator name`, the
  oracle `an array of unknown size needs an initializer` without an initializer). A member or a parameter is one
  (CSUB-2d); a block object needs a storage spelling neither emit has. An array-of-structs element's member array of
  pointers subscripted, `js[i].pa[k][j]`, stays refused in each rail's words, as the rung's other members of such
  an element are.

The count stays **46 of 68**: `bcir_tensor.c`, the one unit whose first refusal was CSUB-2's, moves on to a call of
18 arguments (CSUB-6). Every first refusal the rung names in §3 and §3.1 is now a later rung's.

### 4.3 CSUB-5, slice by slice

**CSUB-5a: the `<string.h>` reads (2026-11-20).**
- **`memcmp`, `strcmp`, `strncmp`, `memchr`, `strchr`, `strrchr`, `strstr` and `strlen` are libc edges on both
  rails**, as `memcpy`, `memmove` and `memset` are: called verbatim, opaque to R18, never renamed, each typed by its
  declared result -- a comparison's `int`, `memchr`'s `void *`, a string search's `char *`, `strlen`'s `size_t`. C23
  made each freestanding (4p7), so they sit below the hosted class this rung declares.
- **What changed on each rail.** Both had read each as a call of an undefined function. The oracle typed its result
  an `int32_t`, and R18 refused the unit. One set now names every `<string.h>` routine either rail calls (the
  oracle's `_STRING_CALLS`, the twin's `string_read` and `str_calls`): the linkable artifact includes `<string.h>`
  for each, never `<math.h>`, and the link-flag map knows each as libc.
- A unit that defines one of these names calls its own function, as it does `memcpy`. `sizeof` of a call to any
  `<string.h>` routine, `memcpy`'s too, stays refused on both rails, in different words: the `sizeof` walk knows no
  library call's type.

The count is **38 of 58**: `bcir_spm.c`, `bcir_bpe.c`, `bcir_verify.c`, `bcir_diag.c`, `bcir_decoder_gem.c` and
`bcir_decoder_pack.c` compile clean. `bcir_channel.c` moves on to R18's refusal of its recursive JSON skip
(`jskip_depth` and `jskip_object`, `jskip_array`), which no rung lifts: the unit needs an explicit stack, as
`bcir_per.c` did and as `bcir_emit.c` does. The ratchet records both `outside`.

**CSUB-5b: the hosted headers (2026-11-27).**
- **Both rails read `<stdio.h>`, `<stdlib.h>`, `<errno.h>`, `<time.h>` and `<inttypes.h>` as the target's C library
  declares what the runtime's tools call of it.** `FILE` is glibc's `struct _IO_FILE` on Linux and the UCRT's
  `struct _iobuf` on Windows, and `errno` each library's own location (`__errno_location`, `_errno`). `PRIu64` is
  `"lu"` where `long` is 64 bits and `"llu"` where it is not. POSIX's `fileno`, `fdopen`, `mkstemp` and
  `clock_gettime` are declared on Linux alone.
- **No new lowering.** Each call is a typed external edge the host linker resolves, a prototype as Phase 3 linking
  already takes, and each emitted `extern` declaration agrees with the library's own, which the artifact is built
  beside. A `FILE` is the library's: only a pointer to one, so a `FILE` object and `sizeof(FILE)` are refused (Clang
  takes them against glibc, whose `FILE` is complete).
- **Two defects at the base, found by the units the headers let through:**
  - The twin numbered an anonymous struct by every aggregate before it, the oracle by the anonymous ones. A cast to
    a pointer to one names it in its op, which the digest holds, so `bcir_cc.c` lowered to two digests. The twin now
    counts the anonymous slots below it, which a speculative parse's rollback truncates.
  - The twin declared another unit's or the library's function by its emit's spelling, `int64_t strtoll(...)`. On
    LP64 that is `long`, where glibc returns `long long`, a conflicting declaration beside the library's header. On
    Windows `int32_t strtol(...)` is `int`, where the UCRT returns `long`. The twin's type now records an integer's
    `long`s as the source spells them, and its library declarations spell them so.
- **Witnesses**: the five headers are one text on both rails for the five targets. Every function they declare,
  called where no run reaches, lowers alike, and each rail's emit compiles beside glibc's headers as C11 strictly.
  Six units -- a `tmpfile` round trip, a number parsed with `errno` and an overflow's `ERANGE`, a sort and a search
  through the unit's comparator, the clocks, the library's streams and conversions, and the anonymous struct after
  tagged ones -- lower alike and run as the host build against glibc.

The count is **55 of 68**. `bcir_cc.c`, `bcir_llama_cli.c`, `bcir_pack_cli.c`, `bcir_qualify_cli.c`,
`bcir_tokenize_cli.c`, `bcir_tokload_tool.c`, `bcir_asn1_bench.c` and `bcir_ai_microbench.c` compile clean, each to
one digest on both rails.

**CSUB-5c: what the tools stopped at next (2026-11-28).**
- **A `for` init's comma expression**, `for (r = 0, w = n; ...)`: its pieces in order, read as the step's are. On the
  twin one helper reads both clauses' lists (`p_simple_list`), the whole list or a refusal.
- **A cast to a pointer type as a base**, `((char *)b)[len] = 0`, `((struct P *)v)->a`: its value is the base, as a
  call's is, in a read, a store, a compound store, a step, an assignment's value and an address. On the twin it is
  A6's parenthesized base, now measured (`paren_base_len`), so `(*q)->m` and the cast share every path. A cast to no
  pointer is no base on either rail, in the oracle's words. `*((T *)b + i)` is the pointer value the sum is,
  dereferenced, as CF-DEREFSUM's `*(p + i + j)` is.
- **`<string.h>`'s writes**, `strcpy`, `strncpy`, `strcat`, `strncat` and `strtok`: typed libc edges beside the
  reads, each a `char *`.
- **`<ctype.h>`**: its fourteen functions are libc edges too, recognized by name, with the artifact including the
  library's `<ctype.h>`. They are not declared by a header the rails supply, because a library may define each as a
  function-like macro as well (C11 7.1.4p1). glibc does, and an emitted `extern int isspace(int);` beside its header
  failed to compile: the declarations witness found it.
- **POSIX's `<unistd.h>`** on Linux: `close`, `fsync`, `read`, `write`, `unlink`, `getpid`, `fork` and `_exit`, its
  `ssize_t` glibc's `long` where pointers are 64 bits and its `int` where they are 32. On Windows, whose UCRT has
  none, it is a header the rails do not supply, read as nothing as any other is.
- **One twin defect at the base.** A call of the function being defined, whose result is a struct, was typed an
  unsigned int, and refused as the struct's initializer before R18 saw the cycle the oracle reports. The twin now types
  a function's own call by its return, as the oracle's `func_rets` does (C11 6.2.1p7).
- **Witnesses**: 5 units lower alike, R18 clean, and run as Clang's build with each rail's emit compiled as C11
  strictly. 2 forms are refused alike, and the self-call reaches R18 on both rails. `<unistd.h>` is one text on both
  rails for the five targets, and every function the slice declares or recognizes, called where no run reaches, lowers
  alike with each rail's emit compiled beside glibc's headers.

The count is **59 of 68**. `bcir_q8_model.c`, `bcir_etl_tool.c`, `bcir_microbench.c` and `bcir_bundle_cli.c` compile
clean, each to one digest on both rails.

**CSUB-5d: `<stdatomic.h>`'s flag (2026-11-28).**
- **`atomic_flag`** (C11 7.17.8) is a structure a compiler's own header defines: Clang's is tagged, GCC's is not.
  Both rails read it as the built-in untagged `struct { _Bool __bcir_flag; }`, one byte as both compilers' is. The
  emit names it by its typedef, as CF-ANON names any untagged structure, so an emitted `atomic_flag *` is the
  compiler's own type.
- **Its operations**, both forms, are CSUB-4's generics, with the orders as the claim's immediates and emitted by
  their own names. A test-and-set reads the flag and returns whether it was set. A clear stores to it, and takes no
  `acquire` or `acq_rel` order (7.17.8.2p2).
- **Witnesses**: the header is one text on both rails for the five targets. A static flag, an element of an array of
  them, a structure's member and a local, set and cleared through both forms, lower alike and run as Clang's build
  against Clang's own `<stdatomic.h>`, compiled as C11 strictly. Six forms are refused alike, in the oracle's words.
- **GNU's `weak`, which the ratchet had recorded past the flag, is the source's choice.** `bcir_quarantine.c` puts it
  under `#if defined(__GNUC__) || defined(__clang__)`, and neither rail is either compiler, so neither defines
  them. Compiled by the verified compiler, the default handler is the strong definition the source asks a non-GNU
  compiler for. An override is then linked instead of that unit, not beside it: two strong definitions do not link,
  as with any compiler that is not GCC or Clang.

The count is **61 of 68**: the quarantine's two units compile clean, each to one digest on both rails. What the rest
stop at, as the ratchet records it:
- `<sys/stat.h>`'s `struct stat` and `<dirent.h>` (`bcir_make.c`): POSIX's host interface past the C library, whose
  layouts differ by target. Past them is a member through a pointer element, `bf->targets[i]->name`, which both rails
  refuse by design today (CF-IDXARROW). Before them was a twin defect, now fixed (CSUB-5e, below);
- a multi-dimensional member array used as a value (`bcir_cpp.c`), and past it the `#if` evaluator's recursive
  descent and the recursion per `#include`, which R18 refuses by design: the unit needs explicit stacks, as
  `bcir_per.c` did.

**CSUB-5e, first slice: a pointer element of a `T **` member, stored (2026-11-28).** `bcir_make.c` grows its string
vectors by `s->v[s->n++] = x`. The twin read every subscript of a `T **` member as the start of a walk through the
element it selects (`s->v[i][j] = c`) and refused the `=`, where the oracle lowered it. The element is now stored
whenever it is the store's own target, and a null pointer constant becomes its pointer type, as a named pointer's store
converts it. `test_a_pointer_element_of_a_member_is_stored_alike_on_both_rails` runs both forms beside a walk through
such an element, each rail's emit running as Clang's build and compiled as C11 strictly. `bcir_make.c` now stops at
`struct stat` itself. Laying out POSIX's `struct stat` and `struct dirent` for each target, with a witness per target,
is the slice's remainder.

**CSUB-5f: `<sys/stat.h>` and `<dirent.h>` (2026-11-28).**
- **What the rails supply.** Both headers are one text on both rails for each target:
  - `struct stat` and `struct dirent`, each laid out as glibc lays it out;
  - `S_IFMT`, `S_IFDIR`, `S_IFREG` and `S_IFLNK`, and the `S_ISDIR`, `S_ISREG` and `S_ISLNK` classifiers;
  - `stat`, `lstat` and `fstat`;
  - an opaque `DIR`, with `opendir`, `readdir` and `closedir`.

  `time_t` and `struct timespec` are one guarded definition that `<time.h>` and `<sys/stat.h>` share. On Windows
  neither header is supplied: the UCRT has no `<dirent.h>`, and its `<sys/stat.h>` is another interface (`_stat`, no
  `lstat`). Each call is a typed external edge, and the artifact carries the structures' definitions as the rails
  laid them out.
- **The layouts.** glibc has two:
  - its x86 one, a single header on x86-64 and i386, where `__x86_64__` chooses the members' widths;
  - its generic Linux one, on aarch64 and riscv64.

  Both rails choose the layout from facts they already share: a 4-byte pointer means i386, an x87 `long double`
  x86-64, and a quad one the generic layout.
- **Witnesses.** The emit reads a member at its offset, so each layout must be the C library's exactly, and
  `test_the_posix_file_headers_lay_out_their_structures_as_the_c_library_on_every_target` holds it three ways:
  - On every target, Clang's layout of the rails' text is asserted in a unit both rails compile for it: each
    structure's size and alignment, and each member's offset.
  - On a glibc host, Clang's view of the rails' text matches the library's own headers for the host's target, member
    sizes and the mode classifiers included. On an x86-64 host it matches for i386 too, through the same biarch x86
    headers. An empty `gnu/stubs-32.h` stands in for the 32-bit library's list of unimplemented calls, which lays
    nothing out.
  - riscv64's text is aarch64's, so the aarch64 runner's witness covers riscv64 as well.

  A tree walk written as `bcir_make.c` writes one, over a directory the test lays out, lowers alike on both rails,
  R18 clean, and runs as the host compiler's build. Its emit reads a file's size and an entry's name. The artifact of
  each call compiles alone, with implicit declarations refused.

`bcir_make.c` now stops at `bf->targets[i]->name`: a member through a pointer element (CF-IDXARROW), which both rails
refuse alike, as `unsupported base expression Index`. The count stays **64 of 68**.

### 4.4 CSUB-3, slice by slice

**CSUB-3a: `<stdckdint.h>` (2026-11-20).**
- **Both rails read `<stdckdint.h>` as Clang and GCC write it**: `ckd_add(R, A, B)` is the overflow builtin
  `__builtin_add_overflow((A), (B), (R))`, and `ckd_sub` and `ckd_mul` alike, beside `__STDC_VERSION_STDCKDINT_H__`.
  The standard-header test holds each rail's expansion of the three to the other's and to Clang's on every target.
- **The builtins lower as the integer builtins do**: verbatim, opaque to R18, each a `bool`, true when the
  mathematical result does not fit `*R`, which takes it wrapped. The escape analysis reads each, a library routine,
  as writing what `R` points at. That covers signed, unsigned, narrow and wide operands and results, mixed types, a
  local, a member (through a pointer too) and a global, and a condition and a loop. Neither rail had the header or
  the builtins.
- **What stays refused**, alike and as Clang refuses it: a `bool` operand or result, a floating operand, a pointer
  operand, a result that is no pointer to an integer, a call of any other arity. A result through a pointer to
  `_Atomic` storage is refused, as Clang refuses it. One through a pointer to volatile storage is refused too, though
  Clang takes it: it is a device access, a follow-on. C23's further constraints on `ckd_*` -- no plain `char`, no
  enumeration, no bit-precise integer -- are not the builtins' constraints, and neither rail adds them.

The count stays 38 of 58: no unit's first refusal was a checked operation.

**CSUB-3b: `<stdbit.h>` (2026-11-27).**
- **Both rails read `<stdbit.h>` as one text for the target.** It holds:
  - `__STDC_VERSION_STDBIT_H__` and the three endian macros, `__STDC_ENDIAN_NATIVE__` the target's byte order;
  - for each of the five standard unsigned types, the fourteen typed functions of 7.18.3-7.18.16. Each is a
    `static inline` function in the subset's own C over the integer builtins both rails lower verbatim: a narrow
    type's count taken in the builtin's width and narrowed, a zero argument's count the width, and `bit_ceil` of a
    value whose ceiling the type cannot hold 0, which C23 leaves to the implementation;
  - each type-generic form a `_Generic` of the typed functions.

  The oracle's text is `stdheaders._stdbit`, the twin's `h_stdbit`. Neither rail had the header.
- **The functions compute what the clause says**: each of the seventy, built for the host, equals a Python model of
  7.18 over every single bit, every run of low ones, their complements and the half points of its type. Each rail's
  emit of a unit of them runs as the host compiler's build of the source against glibc's own `<stdbit.h>`.
- **What stays refused**, alike: a type-generic form of a signed, `bool`, floating or pointer argument, which 7.18.1p3
  leaves out. It names no `_Generic` association, refused now in one set of words on both rails. glibc's
  type-generic forms check no argument's type, so Clang takes each against them.
- **On the way, CF-ARGC.** A call of a function with a prototype now passes as many arguments as it has parameters,
  and at least as many where it ends in `...`. Both rails had lowered `g()` of a `g(unsigned x)` to an emit no
  compiler builds, through every call form; each count Clang refuses is refused alike, in its words.

The count stays 46 of 68: no unit's first refusal was a bit utility.

**CSUB-3c: one `bcir_bytes.h` (2026-11-27).**
- **The runtime has one copy of its fixed-width loads and stores.** Fifteen units had each defined their own:
  `rd16`, `rd32`, `rd64`, `wr16`, `wr32`, `wr64`, `ld32`, `st32`, `read_u32`, `rd32be` and `load64_be`, of a
  `const uint8_t *`, a `const unsigned char *` and a `BCIR_RESTRICT` one, one returning a `uint32_t` for 16 bits.
  §3 counted nine and four to six; reading every unit found fifteen. `runtime/c/bcir_bytes.h` is the one copy:
  little- and big-endian loads of every width, little-endian stores of every width, and the big-endian 32-bit store
  SHA-256's digest writes, each a `static inline` function of `<stdint.h>` alone. The fifteen units include it, and
  SHA-256's open-coded digest store is the header's store.
- **The bytes stay the bytes.** Each load equals Python's `int.from_bytes` of the same octets, and each store writes
  `int.to_bytes`'s, on every width and both byte orders. The runtime's own byte-identity gates (StreamPack, BCAB,
  BTLM, the telemetry frame and envelope, the control plane, the ring, the shard manifest, plan signing, the Q8 model,
  token shards) hold every format's bytes as before. The machine code is not the same: GCC inlines a `static inline`
  helper where it had kept a plain `static` one out of line.
- **In the verified subset**: a unit of every function lowers alike on both C front-end rails, R18 clean, and runs as
  the host compiler's build. Each of the fifteen units has the ratchet verdict it had.
- **One copy stays one**: a scan refuses any unit or header of the runtime, but the header, that defines a static load
  or store of a fixed-width integer through a byte pointer, by its name's shape. Its scope is declared in the test:
  the test drivers and fuzzers keep their own encoders, an independent spelling of the bytes. The scan is witnessed on
  every spelling the fifteen copies took.

CSUB-3 is complete. The checked-arithmetic rewrites the ladder's "what it unlocks" column names -- varints, bitmaps,
length fields and the L3 bounds checks as `ckd_*` -- are not done here: they belong to the units that are rewritten.

### 4.5 Attributes and a statement's timing (PORT-3, 2026-11-20)

The native port audit's PORT-3 puts the timing laws R19 and R20 on the C verifier. They read a claim's optional
timing, which no C front end could give a claim: neither rail took an attribute before a statement at all.

- **A C23 attribute run leads a statement and a block declaration on both rails** (6.8p1, 6.7p1), and a member's
  on the twin as on the oracle. A standard attribute is read over and dropped, as at file scope, whatever its
  arguments.
- **`[[bcir::timing(field = value, ...)]]` gives a statement's claims their timing.** It is also spelled `__bcir__`
  and `__timing__`, and `__has_c_attribute(bcir::timing)` is 1 on both preprocessors, so a unit Clang builds too can
  guard it (Clang answers 0 and drops the attribute):
  - The fields are the model's eight, each named at most once: `clock_domain`, `power_domain` and `sync_type` each
    one string literal of at most 63 letters, digits, `_`, `.` and `-`; `latency_cycles`, `min_throughput_q16`,
    `clock_frequency_mhz` and `setup_hold_margin` each an integer constant expression of 64 bits, as the law rail's
    `#bcir.timing` holds one; `critical_path` 0 or 1.
  - Every claim the statement lowers carries the timing: an expression, a block, each control statement with the
    claims of its condition, body and step, a `return`, a call.
  - The cross-rail canon holds each claim's timing (`|t:...`), so a unit one rail timed differently is a digest
    apart; a unit without one keeps its canon byte for byte.
- **R19 and R20 are in both verdicts.** The oracle's pipeline runs `verify_timing` per function and the twin's
  `bcir_verify_unit` refuses at the first diagnostic, in the oracle's words. `bcir_verify_timing` lists every one,
  which the cross-rail test holds to the oracle's list, claim and RID numbers apart. The twin's control-flow markers
  take no part, as the oracle's claim graph has none. `bcir-cc` refuses a unit that breaks either law, naming it.
- **What stays refused**, alike and in one set of words: a timing before a declaration, a function, a member, a
  label, a `case`, a `default` or a null statement (none would time a claim); a second one on a statement, or one
  inside a statement another annotates; an unknown `bcir` attribute; a malformed field list, an unknown or repeated
  field, a name outside the alphabet or past 63 characters, a count past 64 bits, a flag other than 0 and 1.
- **CF-VENDORATTR, a defect at the base and closed.** Both rails read an attribute run over and dropped what it held.
  The oracle did so at file scope and before a member, so `[[gnu::aligned(8)]]` and `[[gnu::packed]]` laid a struct
  out at offsets Clang's are not: `sizeof` 8 where Clang's is 16 and 5. A vendor's attribute is refused alike now,
  wherever a run is read, unless it is one of the GNU or Clang hints both rails know (`VENDOR_HINTS`: `const`,
  `pure`, `hot`, `noinline`, `nonnull` and the like -- a promise about a function or an object, or a placement hint),
  which is dropped as a standard one is. No production unit writes a vendor's; they use `[[nodiscard]]`.

**Boundaries, the oracle's too.** R20 follows the claim graph's resources, function by function, as the oracle
defines it: a crossing through a global between two functions is no crossing, and a store through a pointer writes
no resource, so a device register written in one domain and read in another is not one R20 sees. Each rail still
refuses some attribute forms in its own words, as before: one before a for-loop's first clause, one that ends a
block, one before a block `typedef`, one as a statement expression's last, and one after a `struct` keyword or
body, which the oracle reads and the twin refuses.

The count stays 38 of 58: no unit's first refusal was an attribute. PORT-4 then added two units: `bcir_etl.c`, the
event transducer, compiles clean (39 of 60), and `bcir_etl_tool.c`, a hosted CLI, waits on CSUB-5's `<stdio.h>`.

### 4.6 CSUB-4, the atomics API (2026-11-27)

- **Both rails read `<stdatomic.h>`'s macros**: `ATOMIC_*_LOCK_FREE` of each standard type, 2 (always lock-free) on
  every target of the matrix, as Clang's `__CLANG_ATOMIC_*_LOCK_FREE` read there; `ATOMIC_FLAG_INIT`; and
  `kill_dependency`. The standard-header test holds each to Clang's on every target. C23 removed `ATOMIC_VAR_INIT`,
  and Clang 18 has no `ATOMIC_CHAR8_T_LOCK_FREE`, so neither is defined. Both rails had read the header as nothing,
  and `bcir_ring.c`'s `_Static_assert(ATOMIC_LLONG_LOCK_FREE == 2, ...)` was no constant.
- **Each generic's explicit-order form lowers as its implicit form's claim**, with its memory orders as the claim's
  immediates, which the digest holds. The forms are `atomic_load_explicit`, `atomic_store_explicit`,
  `atomic_exchange_explicit`, the three fetch-ops' and both compare-exchanges'. Each is emitted as the explicit call,
  each order spelled by name. An order is a constant read as the fence's is: a name, a parenthesized or `+` name,
  `__ATOMIC_*`, an integer, an enumeration constant. A variable order, a local named as one and an expression of one
  are refused as a follow-on.
- **An order the operation does not take is refused alike** (C11 7.17.7): a load's release or acq_rel, a store's
  consume, acquire or acq_rel, a compare-exchange's failure release or acq_rel, a value past seq_cst. Clang only
  warns of those (`-Watomic-memory-ordering`); their behaviour is undefined.
- **A generic of any other number of operands is refused in CF-ARGC's words**, explicit or implicit. The oracle had
  raised an `IndexError` at `atomic_load()`, and the twin lowered a claim of no object.
- **On the way, the oracle's pointer casts.** A cast to a pointer to a 64-bit integer the source names `unsigned long
  long` was emitted `(uint64_t *)`, beside a temp declared as the source names it. On an LP64 target those are
  different types, which GCC 14 refuses. A data-pointer cast is now spelled as its temp is declared.

`bcir_ring.c` compiles with R1-R18 clean, to one digest on both rails: the count is 47 of 68. The claim graph's LLVM
lowering still orders an atomic claim by its hazard (`lower.memory_model`); reading the claim's own orders there is a
follow-on.

### 4.7 Outside the ladder: the units that recurse (O1, 2026-11-28)

R18 refuses recursion by design (§4: "The subset should not admit recursion"), so a unit that recurses is outside
every rung until its walk takes an explicit stack, as `bcir_per.c`'s length reader did with CSUB-1.
- **O1: `bcir_channel.c`.** Its reader skipped a value it does not use -- a key it does not know -- by a recursive
  descent over objects and arrays, 64 containers deep at most. The skip is now one loop over an explicit stack of the
  containers open around the value being read: it reads a value, closes each container the value ends, then reads the
  next member. Its grammar and its bound are the recursive walk's:
  - **Differentially** against the recursive version at the parent commit, over 20,030 documents: random nestings with
    malformed numbers, literals, strings, keys and trailing commas, and 60 to 69 nested arrays, objects and empty
    containers across the bound. The verdicts were identical, 8,411 accepted and 11,619 refused.
  - **Standing:** `test_the_channel_readers_json_skip_is_its_grammar_at_every_depth` holds the C reader to a
    reference walk, written from the grammar, over a seeded corpus of 3,030 documents, each verdict drawn more than
    500 times.

  The unit then compiles with R1-R18 clean, to one digest on both rails: the count is **62 of 68**.
- **Still outside**, each for the same reason, each a larger walk:
  - `bcir_emit.c`, the ASN.1 emitter's descent over the schema tree (`der_write_full` and `der_write_content`,
    `per_node`, `jer_node`);
  - `bcir_cpp.c`, the `#if` evaluator's recursive descent (`ce_comma` down to `ce_prim`) and the recursion per
    `#include`, past a multi-dimensional member array used as a value;
  - `bcir_cfront.c`, the verified compiler's own rail: past the front end's input bound, it is a recursive-descent
    parser throughout.
- **Their size, measured (2026-11-28)** by the recursive call cycles of each unit's call graph, the strongly connected
  components R18 refuses:

  | Unit | Cycles | Lines in them | The cycles |
  |---|---|---|---|
  | `bcir_emit.c` | 11 | 1,182 of 2,371 | DER's measure and write passes, BER's, JER's, OER's and PER's node walks (`per_node` with its payload and length), the plan reader and the node skip, and three smaller walks of the schema tree |
  | `bcir_cpp.c` | 3 | 419 of 2,374 | the `#if` evaluator (`ce_comma` down to `ce_prim`, 49 lines), the macro expander's prescan and rescan (`ppx_*`, 225), and `cpp_process` once per `#include` (145) |
  | `bcir_cfront.c` | 4 | 12,002 of 13,938 | one cycle of 142 functions, 11,921 lines: the parser itself; and three small ones |

  `bcir_cpp.c` is the nearest of the three. Its first refusal is a front-end form (CSUB-2's multi-dimensional member
  array used as a value), and its three walks are each bounded already: an `#if` expression by its nesting bound
  (`#if expression nested too deeply`), the expander by the nesting and budget bounds CSUB-1e shares across both rails,
  and `#include` at 64 levels. Each walk moves onto
  an explicit stack as O1's did, held differentially against the recursive version. `bcir_emit.c` is that work eleven
  times over, one encoder a slice. `bcir_cfront.c` stays outside: its recursion is the parser.

### 4.8 CSUB-6, slice by slice

**CSUB-6a: a call of 24 arguments, and a call through a member of a member (2026-11-28).**
- **The bound.** A call carries 24 arguments on both rails, and a twenty-fifth is refused alike (`BCIR_CALL_MAX_ARGS`,
  `lower.MAX_CALL_ARGS`, CF-CALLARGS). It was 16, and the attention kernels' workspace calls pass 18. A claim's reads
  are one more, and every rail that mirrors that capacity holds 25: the freeze's `CLAIM_MAX_RD` and its fixtures, and
  the law registry's L11 entry.
- **The defect past it was the oracle's.** A call through a function pointer that is a member of a struct-valued
  member, `s->provider.silu(ctx, n, x, y)`, has no dispatch base of its own. Its address is the outer base plus the
  inner member's offset. The oracle fused it into one `c.call.imember:silu` claim over the outer base and emitted
  `s.silu(...)`, which compiles nowhere. The twin read it correctly: it loads the function pointer at its offset and
  calls through the value, a `c.call.indirect`. The oracle now reads it the same way (`_call_member` hands the call
  to `_call_ptr`). The dispatch forms keep their fused claim on both rails: `o->fn(x)`, `o.fn(x)`, `(*p).fn(x)` and
  `q->pp->fn(x)`.
- **Witnesses.**
  - `test_a_function_pointer_member_of_a_struct_member_is_called_through_its_value_on_both_rails` covers eight units:
    the member at its offset or at 0, through `(*q).`, two levels down, void, and the fused dispatch form beside them.
    Each lowers alike, runs as Clang's build on both rails, and compiles as C11 strictly.
  - The same test lowers both runtime units, each to one digest on both rails.
  - `test_a_call_carries_twenty_four_arguments_and_a_twenty_fifth_is_refused_on_both_rails` holds the bound.

The count is **64 of 68**. `bcir_tensor.c` and `bcir_decoder_train.c` compile with R1-R18 clean, each to one digest
on both rails. Under the verified compiler each is its serial scalar build:
- the vector kernels sit behind `__GNUC__` or `__clang__`, and the pool's threads behind `__unix__`, `__APPLE__` or
  `_WIN32`. Neither rail defines any of them, so the pool reports no rail (`bcir_tensor_pool_rails` returns 0);
- the threaded and vector builds stay what §0 says: compiled by Clang and GCC, and held by parity.

The rest of the rung is unchanged: bf16 and `_Float16` with declared conversions, `#pragma STDC FP_CONTRACT OFF`, and
`unreachable()` with an assume form that feeds verifier facts.

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
| **A deterministic counter-based RNG** (a Philox-style generator: key plus counter, no hidden state; **landed 2026-11-21**: Philox4x32-10 in `runtime/c/bcir_rng.c` with streams, floats, bounded draws and permutations, held to Random123's known answers and to `bcir/frontends/models/rng.py`) | The C rail has none; the only generator is in a test. Native weight initialization, temperature and top-k sampling, dropout and shuffling all go through Python. Counter-based generation makes any element reproducible at any thread count, the property the trainer already holds for its arithmetic. | Freestanding C. Held to a Python twin over published known-answer vectors. |
| **A native data loader** (**landed 2026-11-21**: `runtime/c/bcir_tokload.c` reads BTKS token shards in place, each checked whole and addressed by its SHA-256, and fills the trainer's `tokens` and `targets` with index-addressed batches shuffled per epoch on the RNG; `bcir-tokload` maps shards from files; the oracle is `bcir/frontends/models/token_shards.py`. Tokenizing at ingest stays PORT-2's tools) | The trainer's API takes token-id arrays (`bcir_decoder_forward`). Tokenizing, batching and shuffling happen in Python. | Memory-mapped token shards, generation-addressed. A deterministic shuffle on the RNG above. Batches written straight into the trainer's arena. Tokenizing at ingest uses PORT-2's tokenizers. |
| **Hardware digests** (**landed 2026-11-21**: `runtime/c/bcir_digest_hw.c`, SHA-NI and the ARMv8 SHA-2 instructions, carry-less-multiply folding and the ARMv8 CRC32 instructions, selected by `bcir_digest_probe` and held to the portable code, `hashlib` and `zlib`; the existing call sites keep the portable code) | SHA-256 and CRC-32 sit under every content address, generation, StreamPack and signed plan. CRC-SLICE took the portable half of the CRC work: slicing-by-8 brought the CRC from 7–11 times zlib's time to about twice ([`measurements/2026-10-25/crc-slice/`](measurements/2026-10-25/crc-slice/README.json)). The hardware half is open. | SHA-NI and the ARMv8 SHA-2 instructions; carry-less-multiply CRC folding. Each beside the portable code, which stays the reference, and selected by a probe the parity gate covers. |

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
