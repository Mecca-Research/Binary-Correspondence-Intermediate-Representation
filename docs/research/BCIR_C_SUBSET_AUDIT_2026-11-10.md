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
| **CSUB-2** Table idioms (**in progress**: CSUB-2a landed 2026-11-14; CSUB-2b, CSUB-2c, CSUB-2d's two slices and CSUB-2e's first 2026-11-15, §4.2; 31 of the 58 units compile clean) | Static tables whose initializers are address constants (C11 6.6p9), such as string tables. Untagged struct types in a declaration. Struct-typed members of array elements, read and copied (`t[i].m.f`, `t[i].m = v`): the record table (§3.1). String members in static struct tables. Pointer-to-array types (`T (*p)[N]`) as objects and members. Array parameters in function-pointer types. `_Alignas` on any object. Four more the ratchet found first in units the measurement had not reached: a member read through a local's address, `(&v)->m`, as a macro writes it (`bcir_kplan.c`); a row of a multi-dimensional array used as a value, `f(t[1])` of `fe t[4][10]` (`bcir_ed25519.c`); the address-of, increment and assignment of an lvalue reached through a loaded pointer (`bcir_spm.c`, `bcir_emit.c`, both quarantine units); and a struct returned by a call through a prototype (`bcir_handoff.c`). | `bcir_jer.c`, `bcir_emit.c`, `bcir_artifact_bundle.c`, `bcir_provenance.c`, `bcir_verify.c`, `bcir_hydrate.c`, `bcir_handoff.c` and `bcir_ed25519.c`: the table-driven dispatch every codec and catalog reader is written in, and the field arithmetic of the signature code. |
| **CSUB-3** C23 bits and checked arithmetic | `<stdbit.h>` (endian queries, bit width, leading and trailing zeros, population count) and `<stdckdint.h>` (`ckd_add`, `ckd_sub`, `ckd_mul`), each with its exact C23 semantics. The master roadmap lists both as near-term gaps (§4.1). One shared `bcir_bytes.h` built on them replaces the nine copies of the endian helpers, byte-identical. | Varints, bitmaps, length fields and every bounds check of the L3 kind ("bounds live where the resource commits") written as checked arithmetic rather than hand-rolled comparisons. |
| **CSUB-4** The atomics API | `<stdatomic.h>`'s explicit-order calls: `atomic_load_explicit`, `atomic_store_explicit`, `atomic_fetch_add_explicit` and `atomic_compare_exchange_*_explicit`, with `memory_order`. They map onto the memory model the oracle already states (`lower.memory_model`) and the law rail's atomic orderings. | `bcir_ring.c` and the lock-free single-producer, single-consumer structures the AI system's services run on (`BCIR_AI_SYSTEM_ROADMAP.md` §5.2). |
| **CSUB-5** A hosted profile | A declared hosted class, not part of the freestanding subset: `<stdio.h>` streams and file I/O under the injected-allocator memory class (`../languages/C_MEMORY_DISCIPLINE.md`). Memory mapping goes through a declared host interface. The hosted `<string.h>` calls R18 does not know yet, such as `strlen` (`bcir_decoder_gem.c`, `bcir_decoder_pack.c`). | The loaders `bcir_q8_model.c` and `bcir_llama.c`, and the quarantine's recovery |
| **CSUB-6** AI numerics, at the source level | A bf16 storage type and `_Float16` with declared conversion semantics. `#pragma STDC FP_CONTRACT OFF`, so a kernel's no-contraction contract (M17) is stated in its source as well as its flags. C23 `unreachable()` and an assume form feeding verifier facts (the master roadmap's "verifier-fed assume"). A call of more than 16 arguments: the claim graph's bound on a claim's reads (`BCIR_CALL_MAX_ARGS`, CF-CALLARGS), which an attention kernel's workspace call passes with 18 (`bcir_decoder_train.c`). | The scalar halves of the AI kernels. Threads, vector types and ISA intrinsics stay outside the subset under the fallback contract. The tensor pool and the AVX-512/AMX kernels are measured, parity-held runtime services, not verified-subset code. |

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
  of it on the twin and the array on the oracle, so `sizeof b` differs. No runtime unit uses `typeof`.

The count is now **31 of 58**: `bcir_ed25519.c` compiles R1–R18 clean. `bcir_channel.c` and `bcir_verify.c` move on
to an array member of an element used as a value (`channels[best].name`, `f->qcasts[k].type`), CSUB-2c's next form;
`bcir_tensor.c` to a subscript of a pointer element of a member array through a loaded pointer (`j->pa[k][i] = v`).

**Next:** an array member of an element used as a value, the first refusal of `bcir_channel.c` and `bcir_verify.c`.
Then a pointer to an array as a block object, and the remaining lvalue forms (CSUB-2e). Six of the last stand, all on
plain structs as on elements, and for each the oracle lowers and the twin refuses:
- `->` through a dereferenced `T **`, `(*q)->v`;
- `(*c)++` used as a value (the quarantine units);
- an assignment through a dereferenced pointer used as a value (`*low = *high = &absent`);
- a pointer member's assignment used as a value;
- through a loaded pointer: a volatile object's address, a pointer or struct member's address (`&p->q->name`,
  `&s->t->arena`), a member array's element (`&p->q->arr[1]`, `p->q->arr[1]++`), and a pointer element's address
  (`&p->pp[j]`).

Every first refusal the rung names in §3 and §3.1 is one of these.

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
