# BCIR ASN.1 JER compilation roadmap

> **Status:** canonical execution roadmap for schema-bound JSON compilation, based on
> repository state through [PR #670](https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/pull/670).
> It defines future gates; it does not claim that the C, MLIR, SIMD, driver, or kernel
> stages described below are implemented.
>
> **Boundary:** JER is a build-, control-, configuration-, and load-plane format. A
> privileged or latency-sensitive execution path consumes verified claims, StreamPack,
> BCAB, or a native object—not JSON text.

## 1. Decision

BCIR will treat ASN.1 JER as a **schema-bound JSON source and interchange rail** that can
be compiled into existing execution machinery. The schema removes runtime type guessing
and permits generated field dispatch, direct typed sinks, and bounded storage. It does
not turn JSON into a binary format or remove the need to parse and validate its lexical
form.

The target pipeline is:

```text
ASN.1 schema + bounded JER input
  -> scalar structural and UTF-8 validation
  -> optional hosted SIMD structural index
  -> generated schema-specialized event parser
  -> typed value or transactional claim builder
  -> R24-compatible ASN.1 legality
  -> GEM/K_BCIR planning
  -> StreamPack
  -> BCAB/native backend artifact
```

The text is consumed once. Passes enrich immutable typed IR and side tables rather than
rewriting JSON with register assignments or optimization annotations. Content addresses
bind the input, schema, instruction set, compiler version, target profile, and resulting
artifact.

This program does **not**:

- put JSON parsing in interrupt, DMA, submission, or other freestanding hot paths;
- claim fixed byte offsets into variable-length JSON strings;
- infer memory safety from a schema without verifying lowering, lifetimes, effects,
  concurrency, MMIO ordering, and generated code;
- change device-defined register offsets, widths, endianness, or barrier requirements;
- equate CRC or SHA-256 integrity with artifact authenticity;
- promise secure live driver replacement from schema compatibility alone; or
- revive user-defined ECN encodings without a measured workload that the fixed candidate
  set cannot express. *(This condition has since been met and the half is built: the
  workload is a scaled-length frame header, and all five candidates are executed against it
  rather than argued about. See §2's ECN row and the build-out roadmap's section G.)*

## 2. Source-backed baseline through PR #670

“Landed” below means implementation plus deterministic tests are present. Static
repository inventories remain generated in [`STATUS.md`](STATUS.md).

| Surface | State | Evidence and exact boundary |
|---|---|---|
| X.697 value encoding | **Landed on the Python oracle** | [`jer.py`](../bcir/asn1/jer.py) implements clauses 20–41 over the supported ASN.1 type model |
| JER encoding instructions | **Landed on the Python oracle** | `ARRAY`, `BASE64`, `NAME`, `OBJECT`, `TEXT`, and `UNWRAPPED`, including clause 13 precedence, are gated in [`test_asn1_jer.py`](../bcir/tests/test_asn1_jer.py) |
| JER input rejection | **Landed on the Python oracle (J1)** | [`jer_bounded.py`](../bcir/asn1/jer_bounded.py) enforces every §4.3 limit in a single octet pass *before* any value graph exists, then validates UTF-8, then the schema; gated at each boundary by [`test_asn1_jer_bounded.py`](../bcir/tests/test_asn1_jer_bounded.py). `json.loads` still builds the value graph after the bounding pass approves the input — removing that materialization is J3 streaming work, not an input-rejection gap |
| Canonical-byte validation | **Landed on the Python oracle (J1)** | §3.2's re-encode-and-compare oracle: a canonical decode reads the input with BASIC (§6.3), then refuses any octet the canonical encoder would not have produced, reporting the first differing offset |
| Framed input and diagnostics | **Landed on the Python oracle (J1)** | §3.3's version/sequence/generation/length/CRC-32 frame, verified before any payload is returned; §4.2's stable error code, byte offset and required capacity as a structured `JerDiagnostic` |
| `bcir-asn1c` JER modes | **Landed** | `--jer`, `--basic`, `--framed` and `--transcode TYPE`; decoding always runs the bounded oracle |
| Deterministic JER emission | **Landed as a BCIR profile** | `JerRules.CANONICAL` fixes BCIR choices; X.697 defines no standard canonical JER variant, so this profile has no standards OID |
| X.680/X.681/X.682/X.683 schema rail | **Landed within the documented subset** | Front-end, information objects, constraints, open-type table resolution, and parameterization feed the shared schema model |
| Other encoding candidates | **Landed on their stated rails** | DER/BER, canonical PER aligned/unaligned, COER/OER, and CXER/XER are recorded in the ASN.1 build-out roadmap; their C coverage differs by format |
| ECN | **Parts 1, 2 and 3 landed** | [`ecn.py`](../bcir/asn1/ecn.py) models classes, objects, object sets, EDM/ELM, and built-in BER/PER sets. [`ecn_user.py`](../bcir/asn1/ecn_user.py) adds the user-defined half: bit-level encoding spaces, justification, `#PAD`, object-chosen transmission order, `INT-TO-INT`/`INT-TO-BITS` transforms with inverses, and `#OUTER`. The §6 gate's reopening condition is met **and executed** — a scaled-length frame header that none of DER, canonical PER aligned/unaligned, COER or CJER reproduces, with canonical PER landing on the same octet count and different octets, so the gap is expressiveness and not size. A test fails if any candidate ever matches. **Citations are checked against Rec. ITU-T X.692 (02/2021)**: the pass confirmed the clause-level attributions (19 mapping values, 20–23 defined syntax, 24 `#TRANSFORM`, 25 `#OUTER`) and found two semantic divergences, both corrected — §24.3.5 allows an object *precisely one* arithmetic operation with composition via an `ORDERED` list (§22.4.1.1), where the first version fused offset and scale into one; and Table 6 makes reversibility per-operation, with `modulo:n` **Never reversible** yet legal, where the first version refused every non-invertible value. §24.3.7's `divide:n` truncates toward zero, which Python's `//` does not. **The surface syntax is now parsed.** [`ecn_syntax.py`](../bcir/asn1/ecn_syntax.py) reads clause 20's defined syntax — the bracket-optional keyword grammar clause 23's `WITH SYNTAX` statements spell out — from an `ENCODING-DEFINITIONS` module, and [`BCIR-FrameHeader.ecn`](../bcir/asn1/BCIR-FrameHeader.ecn) is the gate's own workload written in it, producing `AA 00` byte-for-byte against the Python-assembled objects. Reading the `WITH SYNTAX` text found three more divergences: §22.2/§22.8's property *groups* were booleans (an `align_before` flag is `ALIGNED TO NEXT octet PADDING zero` with the unit, padding and pattern all frozen); §21.8.1's `Justification` is `CHOICE {left INTEGER(0..MAX), right INTEGER(0..MAX)}` and the **offset** was dropped, so a field two bits in from the top of its space was unreachable; and §22.10.1.1 gives a concatenation object only `{textual, tag, random}`, so the free transmission-order tuple is really the §16.5 `ConcatenationStructure`'s textual order (§22.10.3.1), which the module now states. Every unbuilt property group — `REPLACE`, `START-POINTER`, `IF`/`IF-ALL`, `DETERMINED BY`, `USING`, `UNUSED BITS`, `EXHIBITS HANDLE`, `BIT-REVERSAL` — is recognized and refused with its clause rather than skipped. **The plan-v6 question is answered, and answered no**: an ECN encoding is not a sixth column in [`encode_plan`](../bcir/asn1/encode_plan.py), because that plan describes an ASN.1 *type* and the frame header's wire order and its `reserved` bits are properties of an encoding structure — `EncodeNode` has a slot for neither, and adding them would make a node's meaning depend on which candidate read it. It is a third compilation with its own version counter and its own digest, which is the thing an ECN specification lacked. **The remaining property groups are built**: §21.3/§22.3/§22.8's determinants over one back-patching mechanism (§22.8.3.7's NOTE describes the suspension exactly), §21.11's `IF`/`IF-ALL` range conditions with §21.12's `Comparison` — which is what makes an ECN integer encoding schema-directed, since §23.6.3.1 selects on the *bounds of the type* rather than on any value — §22.12's four reversals, and §22.1's replacement semantics with §22.1.3.6's head-end insertions hoisted as a block in component order. A seventh divergence turned up and it is in the text: §21.14.6 describes `ReversalSpecification` "in the order of enumerations listed above" and gives a different order than §21.14.1, where §22.12.3.2 and the names both agree with §21.14.1 — recorded on both rails rather than silently resolved. **ECN is now on the MLIR law rail as R25**: `bcir.ecn.module`/`.class`/`.structure`/`.field`/`.object`/`.condition` with eleven statically decidable rules, one negative fixture each plus two positive modules, and a parity gate that reads the ODS directly. **Both of the refusals this row used to end on are now built**: §22.1's replacement *notation* reads from module text (§22.1.2.2/§22.1.2.4's X.683 parameterization applied to ECN arrived with `ecn_param.py`), and §21.3.6's `container` determination is implemented with its §21.5.6/§21.7.8 siblings — an element determined that way must be the last encoding in its container, which is checked rather than trusted, for both presence states. `_UNSUPPORTED_KEYWORDS` now holds no unbuilt group at all, and R25 covers Annex C parameterization via `bcir.ecn.parameterized` |
| JER bounded reader in C (J3) | **Landed on the C rail** | [`bcir_jer.c`](../runtime/c/bcir_jer.c) runs the same three stages in the same order — §4.3's limits in one octet pass, §7.6.2's encoding, then the grammar — with **no allocation, no recursion and no floating point**: the container stack and decode scratch are the caller's, and a number event hands back the raw token rather than a parsed double. Canonical-byte validation (§3.2) and schema legality stay on the Python rail and are deliberately *not* reimplemented, because a second definition of canonicality is free to drift from the encoder that is the actual definition. Building it found three defects in J1's rail; see §7.2 |
| JER schema plan (J2) | **Landed on the Python oracle** | [`jer_plan.py`](../bcir/asn1/jer_plan.py) compiles a root type into the §5.1 descriptor — identity and source hash, family/profile/instruction hash, sorted member dispatch, required/default/extension metadata, recursion bounds and static capacity. **Static capacity is `None` almost everywhere**, and that is a property of JER rather than of the compiler: §7.2.2 l) and h) hide integer and string constraints from a JER encoder, so only BOOLEAN, NULL, ENUMERATED and a single-size BIT STRING (§7.2.1 a) are derivable. J3's C interface must therefore take its capacity from the caller |
| Encoding selection harness | **Landed as measurement evidence** | [`selection.py`](../bcir/asn1/selection.py) measures exact wire size and Python-oracle timing for a fixed candidate set; it is not a native K_BCIR table or selection certificate |
| Existing bounded C JSON precedent | **Shape-specific only** | [`bcir_channel.c`](../runtime/c/bcir_channel.c) bounds bytes and depth and decodes `channel.json`; it is not a JER engine or generated ASN.1 parser |
| MLIR family/profile and R24 (J4 part 1) | **Landed on the law rail** | `BCIR_Asn1Rules` names every transfer syntax the repository speaks; family, canonicality and PER alignment are **derived** from it rather than stored beside it, so no two attributes can disagree about one syntax. R24's emission law generalizes from "emits DER" to "emits a syntax whose octets are a function of the abstract value", which closed two holes the X.690-shaped test had left open. `bcir.asn1.transcode` and `strict_canonical` are added. See §5.3 for why this is one enum and not the (family, profile) pair originally planned |
| JER-to-claims lowering (J4 part 3) | **Landed for the manifest surface** | [`manifest.py`](../bcir/asn1/manifest.py) gives `channel.json`, `DeviceManifest` and the §6.2 selection envelope ASN.1 schemas, and §5.4's two sinks commute over all nine built-in channels. **Not** claimed: GEM graphs and BCAB have no direct JER builder, and the value graph still exists behind the walk because `json.loads` builds it — removing that materialization is J6 streaming work |
| Native calibration and certificates (J6) | **Landed for the encoding-selection surface** | [`certified.py`](../bcir/asn1/certified.py) gives distribution-free intervals, frozen generation-tagged tables and §6.2's certificate; [`native_bench.py`](../bcir/asn1/native_bench.py) produces a genuinely `measured` table from a native C harness and refuses every candidate the C rail does not implement; `select_budgeted` adds RCSP — a timing minimized across stages under a total octet budget, with the union-bound coverage decay reported rather than hidden. **Now claimed**: the native encode column measures nine of ten candidates (`run_native_encode_bench`), including all four PER rows once E2's bit-oriented writer landed — the only absences are the ones a *law* keeps out. Target calibration is gated by [`calibration.py`](../bcir/asn1/calibration.py): a record carries the host's own steal and throttle counters, the CPUs its rounds ran on, and a digest of the fixed corpus, and `refusals()` decides whether it may be frozen into a table — so `measured_table(target=...)` no longer takes the target on trust. **One target is admitted**: a Samsung S24+ / Snapdragon 8 Gen 3, pinned to cpu 7 under Termux, 41 rounds x 64 iterations across nine candidates on both axes ([`asn1_calibration.json`](../docs/measurements/asn1_calibration.json)). Taking it found a measurement defect worth more than the numbers: **every distinct value in the record is a multiple of 52.083 ns**, the 19.2 MHz ARM architectural timer period, so forty-one identical samples reflect the clock rather than certainty. A degenerate `[104, 104]` would have made two candidates one tick apart "significantly different" under `Interval.overlaps`; `table()` now widens every interval to the quantum it estimates from the samples, which leaves a genuine four-tick gap disjoint and makes a one-tick pair correctly indistinguishable. **Two targets are admitted**, and they are the same phone on different core clusters — a Cortex-X4 and a Cortex-A520, which is the point rather than a duplicate. The A520 is **4.0–5.1× slower** on every candidate, so averaging them would describe no core that exists, which is what the multi-CPU refusal prevents. The candidate *ranking* survives the core change (BER < DER < JER on both) but the *spacing* does not — 4.04× on BER against 5.08× on JER — so one table cannot serve both cores, which is what per-target calibration is for. A second measurement effect showed up too: on the X4, BER and DER decode become indistinguishable after clock widening, while on the slower A520 they separate cleanly (182 against 221 ns). The slow core resolves a difference the fast core cannot, because the same work spans more timer ticks. A `#targetabi` gate now compiles the freestanding core for **aarch64-linux-android**, armv7a, i686, riscv64 and wasm32, and refuses any source that hand-declares a libc function — the two things that were checkable after a bench shipped broken under Termux, because the existing cross-compile gates target `aarch64-linux-gnu`, the right architecture and the wrong libc. The **hosted** tools are still unchecked against bionic: there is no sysroot here and the network policy denies the NDK, so that half closes only by running on a device. **Target hardware counters are built but never exercised**, and the reason is access rather than code: Android answers `perf_event_open` with `Permission denied` (`perf_event_paranoid=3`), and this container exposes **no `cpu` event source at all**, so the syscall returns `ENOENT` even though it runs as root with `CAP_PERFMON` and `perf_event_paranoid=2`. Privilege is not the constraint and cannot be made into one — see [`BCIR_TARGET_ACCESS.md`](BCIR_TARGET_ACCESS.md), which states the capability floor for every remaining phase and the bare-metal targets that would clear it |
| Driver/kernel use | **Missing** | No resident driver, Linux module, native-kernel parser, or physical-device comparison uses JER |

The current PersonnelRecord experiment proves a useful but narrow point. Exact encoded
sizes span 84 bytes for canonical unaligned PER through 385 bytes for BCIR-canonical JER,
and a bandwidth objective selects unaligned PER without user-defined ECN. Decode timings
are Python implementation measurements: `json.loads` is native code while COER decode is
currently Python, so those timings cannot establish a target-independent ordering.

**That caveat is no longer a prediction — it is measured, and the ordering is inverted.**
The native harness ([`bcir_asn1_bench.c`](../runtime/c/bcir_asn1_bench.c),
[`native_bench.py`](../bcir/asn1/native_bench.py)) times two C decoders against each other
on the same abstract value:

| Rail | DER | BCIR-canonical JER | Verdict |
|---|---|---|---|
| Python oracle | ~30.8 µs | ~13.7 µs | JER **2.2× faster** |
| Native C | ~51 ns | ~209 ns | DER **4.1× faster** |

Same values, same encodings, opposite answer. The oracle was ranking `json.loads` against a
Python DER decoder and reporting the result as a property of the *encodings*. This is
exactly what J6's refusal to decide a timing objective from an oracle table exists to
prevent, and it is now pinned by a test rather than argued from first principles.

## 3. Standards and terminology

### 3.1 JER remains JSON text

[ITU-T X.697](https://www.itu.int/rec/T-REC-X.697-202102-I) specifies JSON Encoding
Rules for ASN.1. [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259.html) defines JSON
as a text-based format with strings, numbers, objects, arrays, literals, escapes, and
UTF-8 interoperability requirements. Consequently, every conforming fast path still
has to:

- locate structural tokens while respecting quoted strings and escapes;
- validate UTF-8 and escaped Unicode, including surrogate pairing policy;
- reject malformed numbers and enforce ASN.1 range and precision constraints;
- detect duplicate, unknown, missing, and renamed members under the active schema and
  JER instructions; and
- prove complete input consumption and enforce explicit resource limits.

Schema specialization eliminates general-purpose type discovery and can avoid a generic
DOM. It does not make member locations compile-time constants: whitespace, escaping,
number length, member order under BASIC JER, and variable-size values remain data
dependent.

### 3.2 Canonical profiles

X.697 registers JER but no canonical JER transfer syntax. BCIR therefore names its
private deterministic profile **BCIR canonical JER profile v1** and identifies it with a
BCIR profile identifier plus a content hash, never an invented standards OID.

The profile must define byte-level output and validation, including member ordering,
whitespace, escape spelling, number spelling, default omission, SET OF ordering, and
instruction effects. A canonical decoder must reject non-canonical bytes, not merely
decode them to the same abstract value. Re-encoding and byte comparison is the initial
oracle; a later streaming implementation may enforce the same rules directly.

**Implementation note (J1).** A canonical decode reads the input with the BASIC profile
first — §6.3 makes BASIC the decoder that "shall support all JER encoding alternatives" — and
only then compares octets against a re-encode. Splitting it that way leaves exactly one
mechanism responsible for canonicality. Asking the schema layer to *also* judge it, as an
earlier draft did for member order, creates a second definition that can drift from the
encoder and produces two different diagnostics for one property; the octet comparison cannot
drift, because the canonical encoder is the definition.

[RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html) is a comparison corpus, not an
automatic replacement for the BCIR profile. JCS number and string rules must not silently
truncate ASN.1 integer or real semantics.

### 3.3 Streaming

Internal BCIR streams use an explicit version, length, integrity field, sequence, and
generation around each complete document. No claim or artifact becomes visible before
the complete frame passes lexical, schema, semantic, and integrity checks.

[RFC 7464](https://www.rfc-editor.org/rfc/rfc7464.html) JSON Text Sequences may be
supported for external logs and tooling. It is not an internal transaction or integrity
protocol, and a truncated record never commits partial state.

## 4. Trust and ownership contract

### 4.1 Parser classes

| Rail | Allocation and dependency policy | Intended use |
|---|---|---|
| Python oracle | Dependency-free package; bounded before parse | Specification, differential oracle, schema compilation, diagnostics |
| Scalar C core | Caller-owned input, output, and scratch; no hidden heap; fixed-width counters | Portable correctness rail, hosted tools, and freestanding-compatible generated parsers |
| Hosted C++ SIMD adapter | Optional C++17 implementation behind the same C ABI and trace; runtime feature detection with scalar fallback | Accelerated structural and UTF-8 scanning on measured hosts |
| Driver/kernel adapters | No JSON in execution queues; offsets and generation-tagged handles across boundaries | Load verified binary artifacts and expose conventional UAPI behavior |

The scalar rail is authoritative for native parser correctness. SIMD is an optimization
candidate, not a separate semantic implementation. External
[simdjson research](https://arxiv.org/abs/1902.08318) and
[Mison](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/mison-vldb17.pdf)
provide benchmark baselines for structural indexing and schema/query-aware parsing; no
third-party parser source is vendored into the freestanding core.

### 4.2 Failure atomicity

A decode/lower operation is a transaction:

1. Validate the frame and declared limits.
2. Scan and validate complete JSON structure and UTF-8.
3. Apply the compiled schema and JER instructions.
4. Build into caller scratch or a rollback-capable event sink.
5. Run ASN.1 semantic checks and BCIR legality.
6. Publish the typed result or claims in one commit.

On failure, the destination, active generation, and prior artifact remain unchanged.
Diagnostics report a stable error code, byte offset, schema path, and required capacity.
Destroy/reset functions are idempotent, and a retry cannot observe stale pointers or a
partially advanced sink.

### 4.3 Required limits

Every public decode operation carries explicit maxima for:

- input and framed-document bytes;
- nesting depth and total nodes;
- object members and array elements;
- decoded string and raw number-token bytes;
- integer digits and exponent magnitude;
- schema recursion and open-type resolution steps;
- scratch, output, claims, resources, and diagnostics; and
- total work, so an input cannot hide quadratic duplicate/member lookup.

Checked addition, multiplication, alignment, and offset arithmetic precede every access.
Limits are part of the compiled plan and may be tightened by a caller, never silently
expanded.

## 5. Compiled schema and lowering contract

### 5.1 Immutable schema plan

The X.680 front end will compile each supported root type into a deterministic descriptor
containing:

- schema/module/type identities and source SHA-256;
- JER family, profile, instruction set, and instruction hash;
- member-name dispatch tables, required/default/extension metadata, and duplicate policy;
- primitive conversion, ASN.1 constraints, open-type selectors, and recursion bounds;
- exact scratch/output upper bounds where statically derivable;
- deterministic schema-plan version and compiler identity; and
- an optional direct-builder contract for a named BCIR consumer.

Repeated compilation is byte-identical. Unknown required descriptor features fail
closed. Descriptors are data; they contain no process pointers or executable callbacks
when serialized.

### 5.2 Native interface target

The future native API is a bounded operation over a constant schema plan:

```c
int bcir_jer_decode(
    const bcir_jer_schema_plan *plan,
    const uint8_t *input,
    size_t input_size,
    const bcir_jer_limits *limits,
    void *scratch,
    size_t scratch_size,
    const bcir_jer_sink *sink,
    bcir_jer_diagnostic *diagnostic);
```

The exact structs are frozen only with the C implementation. Their contract is already
fixed: borrowed input/plan, caller-owned scratch and diagnostic, transactional sink,
explicit capacity, no hidden allocation, no partial output, and deterministic errors.
Generated fixed-schema wrappers may remove generic dispatch but must produce the same
event trace and diagnostics as the table-driven scalar implementation.

### 5.3 MLIR and R24

**Landed, and not in the shape this section originally proposed.** The plan here was a
*second* attribute, `#bcir.asn1_encoding<family = …, profile = …>`, sitting beside the
legacy `#bcir.asn1_rules<ber|cer|der>`, with R24 checking "family/profile consistency".
Building it made the objection plain: **consistency between two attributes is a check that
only exists because the IR was allowed to write down the contradiction.** A pair
`family = jer, profile = canonical_per_aligned` is expressible and meaningless, and an
operation carrying *both* a legacy rules attribute and an encoding profile has two answers
to one question. That is the same drift the byte-comparison oracle in §3.2 exists to avoid,
one level up.

What landed instead: `BCIR_Asn1Rules` is extended **in place and additively** to name every
transfer syntax the repository speaks, and family, profile, canonicality and PER alignment
are **derived** from it (`asn1FamilyOf`, `isCanonicalAsn1Rules`, `asn1RulesAreAligned` in
`mlir/lib/BCIRDialect.cpp`). One stored fact, one source of truth, and the decomposition
the laws are written over cannot disagree with it. Each switch is exhaustive with no
`default:`, so adding a syntax without classifying it is a `-Werror=switch` build failure
rather than a silent misclassification — which for canonicality would mean R24 quietly
permitting a sender's-option encoding to be emitted and digested.

```text
#bcir.asn1_rules<der>                       // X.690,  canonical
#bcir.asn1_rules<canonical_per_unaligned>   // X.691,  canonical
#bcir.asn1_rules<coer>                      // X.696,  canonical
#bcir.asn1_rules<cxer>                      // X.693,  canonical
#bcir.asn1_rules<bcir_canonical_jer>        // X.697,  canonical, no registered OID
#bcir.asn1_rules<jer>                       // X.697,  decode target only
```

`ber`/`cer`/`der` keep integer values 0/1/2, so every artifact, bytecode file and fixture
written before the other families existed still parses and still means what it meant — the
source compatibility this section asked for, obtained without a second attribute.

**R24 keeps its number and its laws generalize rather than multiply.** "BCIR emits DER
only" becomes **"BCIR emits only a transfer syntax whose octets are a function of the
abstract value"** — the property a digest actually rests on, of which DER is merely the
X.690 member. That closed two holes the X.690-shaped law had left open:

- `strict_der` with `cer` passed verification, because the test was `rules == ber`. X.690
  §9.1 makes the indefinite length form mandatory for constructed CER encodings, so a CER
  artifact is exactly as un-byte-stable as a BER one.
- `strict_der` on a non-X.690 decode passed, though "strict DER" is a category error about
  a PER or JER decoder rather than a stricter setting. `strict_canonical` is the
  family-neutral spelling, and R24 now directs callers to it.

`bcir.asn1.transcode` is added as the law-rail form of `bcir-asn1c --transcode` and of what
`selection.py` measures: one schema, one abstract value, two transfer syntaxes. Its target
is emitted, so it falls under the same canonicality law; `preserve_value` additionally
requires a **canonical source**, because a syntax admitting several encodings of one value
gives the sender a choice a replay cannot reproduce.

JER *instruction* legality and descriptor identity remain future work — they need the
descriptor on the law rail, which is J5's `#5.1` promotion, not this one.

### 5.4 Direct lowering

The first integration target is `channel.json`, followed by `DeviceManifest` and the
encoding-selection envelope. Each receives an ASN.1 schema, a compiled JER descriptor,
and two sinks:

1. a typed-value sink for diagnostics and round trips; and
2. a direct claim builder that emits the same resources, claims, dependencies, and
   metadata as the existing programmatic constructor.

Direct lowering must commute:

```text
JER -> typed value -> claims
  == JER -> direct claim builder
```

Both paths then pass normal law verification. Telemetry JSON/JSONL remains evidence or
export data and cannot steer legality or mutate an in-flight plan.

**Landed in J4 part 3** ([`manifest.py`](../bcir/asn1/manifest.py)), over all nine built-in
`HardwareChannel`s. The design decision that makes the law testable rather than tautological:
**both sinks consume ONE event walk.** Two independent readers would be two parsers, free to
disagree about what the document says, and "the paths commute" would then be testing parser
agreement rather than builder agreement. With one walk feeding both, a difference in the
result is a difference in the *builders*, which is what §5.4 is about. That the direct path
really is direct is checked structurally rather than asserted: a test watches
`ChannelManifest.__init__` and fails if the direct builder constructs one.

The walk visits members in **schema order**, not document order, which is what lets a
generated fixed-schema consumer exist at all (§5.2). A sink keyed on arrival order would be
right for canonical input and wrong for a BASIC document that wrote its members differently.

Building it produced one finding, and it is a general hazard rather than a one-off. **A
DEFAULT in an ASN.1 schema is not free**: X.690 §11.5, via X.697 §21.2, makes the canonical
encoder *omit* a component equal to its default, while `channel_plugin`'s validator requires
every declared key to be present. `MemoryTier.capacity` defaults to 0 in the dataclass and is
read with `.get("capacity", 0)`, so `DEFAULT 0` looked faithful — but `profile_to_schema`
always writes the key, so the canonical JER for a zero-capacity tier was a document the
repository's own loader refused. The schema was the wrong rail and was corrected; loosening a
third-party-input validator to match a schema would have been fixing the wrong side. A test
now forces every zero-valued member and re-checks the loader accepts the canonical form.

Two limits worth stating. A schema gives well-formedness, never correctness: `lane_widths`
must start at 1 and be strictly increasing, which is a rule about a cost model that X.680
constraints cannot express, so those checks stay in `channel_plugin`. And the direct builder
reuses `schema_to_profile` deliberately — a second construction of `TargetProfile` would be a
second definition of the K_BCIR cost model, free to drift from the one the optimizer reads.

## 6. K_BCIR selection and artifacts

### 6.1 Cost model

The optimizer keeps the existing twelve axes:

`compute`, `memory`, `fabric`, `sync`, `compile`, `thermal`, `power`,
`reliability`, `security`, `accuracy`, `contention`, and `verification`.

Wire bytes price memory/fabric; parser cycles and instructions price compute; branch and
cache effects price compute/memory/contention; schema compilation prices compile; bounds,
canonical checks, and certificate work price verification/security. No thirteenth
“cache-line” axis is introduced.

Raw counters remain graded evidence. They enter a frozen, generation-tagged target table
only after provenance, variance, environment, and replay checks. They cannot waive ASN.1
or BCIR legality.

### 6.2 Selection certificate

A promoted encoding decision records:

- abstract schema/type and value/content identity;
- candidate family/profile/OID where standardized;
- exact encoded size and representability verdict;
- target profile, hardware manifest, compiler, and calibration generation;
- benchmark protocol, samples, prediction/confidence intervals, and counter provenance;
- hard limits and the twelve-axis cost vector for every admitted candidate;
- Pareto set, RCSP budget, selected candidate, and deterministic tie break; and
- law, round-trip, canonicality, and native-artifact hashes.

The current Python harness is retained as an oracle experiment. Production selection
reads the frozen target table and refuses an unmeasured required target instead of
substituting Python timings.

**Landed in J6** ([`certified.py`](../bcir/asn1/certified.py)). Three things are worth
recording about how, because each was a choice with an alternative that looks reasonable and
is not:

*Timings are carried as **intervals**, never scalars.* A median comparison always produces a
winner — including when the difference is scheduler noise — and produces a *different* winner
on the next host. Two candidates whose intervals overlap are reported as **indistinguishable**,
and the decision falls to exact encoded size, which has no distribution. That is two-truth
applied to the tie-break: the noisy measurement says "I cannot separate these", and the exact
one decides. The certificate records which candidates could not be separated, because "we
chose A over B" and "A and B were the same and A sorted first" are different decisions.

*The intervals are distribution-free*, from order statistics rather than a normal
approximation. Timing distributions are heavy-tailed and asymmetric, so a normal CI would
understate the spread exactly where scheduler noise lives. Coverage is computed in exact
integer arithmetic and reported in parts per million — a coverage figure that varied with the
host's rounding would not be one, and it is written into a certificate.

*The refusal fires where a timing is **consulted**, not at the door.* A wire-size objective
reads no interval, so it is decidable from exact arithmetic on any table, and refusing it
would teach callers to pass `allow_oracle_table=True` by reflex — carrying that habit into
the timing decisions where the guard is load-bearing. A guard that fires when it is not
needed is a guard people learn to disable.

**The native microbench protocol landed** with J6's follow-on
([`bcir_asn1_bench.c`](../runtime/c/bcir_asn1_bench.c) +
[`native_bench.py`](../bcir/asn1/native_bench.py)), so a `measured` table can now be
produced and a timing objective can actually be decided. `build_table` still defaults to
`provenance="oracle"`, because it runs under a Python codec; `measured_table` is the
function that earns the other label.

Its protocol is where the numbers become comparable: one corpus with identical octets every
round, warmup discarded, **interleaved round-robin** so a drifting CPU biases no single
candidate, per-round medians so clock granularity cannot quantize a short decode to zero,
and every round emitted so the interval is derived by the reader rather than baked in.

**The refusal moves rather than disappears, and this is the design.** The harness measures
what the C rail natively implements and refuses the rest — so the measured table is smaller
than the candidate list, and `select_certified` then refuses any objective needing a missing
row. Two absences, for different reasons that are recorded separately because they call for
different decisions:

- **PER cannot have one.** X.691 §7.2 makes a PER encoding non-self-delimiting — without
  the type, the octets cannot be walked — so there is no schema-free structural pass to
  time and no comparable native number will ever exist. `bcir_per.c` implements the reading
  *primitives*; timing those against a whole-document scan would compare unlike work and
  call the difference an encoding cost.
- **OER cannot either**, and this entry was **wrong when first written**. It read "no C OER
  decoder exists yet", which called a law an ordinary gap. X.696 §6.2 states the same rule
  as X.691 §7.2: *"without knowledge of the type of the value encoded, it is not possible to
  determine the structure of the encoding"*. [`bcir_oer.c`](../runtime/c/bcir_oer.c) now
  decodes OER natively — and writing it is exactly what exposed the mislabel, because the
  decoder is schema-**directed** while every row in this table is a schema-free structural
  scan. The decoder exists *and* the row is still absent, for a stated reason.

**RCSP integration landed** as
[`select_budgeted`](../bcir/asn1/certified.py): minimize a timing across a chain of stages
subject to a total octet budget, exactly, by dynamic programming over `(stage, octets spent)`.
It is a real constrained problem rather than two independent selections — the optimum is not
monotone in either axis, so taking each stage's local best can miss the only feasible plan.
The part worth recording is what summing intervals does to them: each is a statement holding
with some probability, and the statement about the *sum* holds only when all of them do, so
the union bound gives `1 − n(1 − c)`. **A chain of twenty 95% intervals certifies nothing.**
The plan still returns the optimum and reports `certified=False`, because withholding the
answer and overstating it are both worse than saying which one you have.

**The native encode column: not the decode table's mirror, and the reason is a law.** The
obvious expectation is that the write side is easier — you are handed the value, not the
octets, so X.691 §7.2 and X.696 §6.2 do not apply. Checking it against the oracle's own
encoders says otherwise, and the partition is *recorded and derived-checked* in
[`ENCODE_OPS`](../bcir/asn1/native_bench.py) rather than asserted:

| | schema-free decode (a structural scan) | schema-free encode |
|---|---|---|
| BER, DER | yes | **yes** — a TLV tree carries its own tags and lengths |
| JER, JER-BCIR-CANONICAL | yes | **no** — X.697 §22.2 puts member *identifiers* in the document, and an identifier exists only in the type |
| PER (4 rows), OER (2 rows) | no, by §7.2 / §6.2 | no — the type fixes field widths, presence bits and the preamble |

So the two absences do not overlap: **JER is measurable one way and not the other**, and PER
and OER — permanently absent from the decode table — are perfectly encodable *given a plan*.
A schema-free encode harness is cheap to build and would yield a two-row table with JER
missing, which reads as an unfinished implementation rather than as the law it is. A
schema-**directed** encode harness instead covers *every* candidate, including the two the
decode table can never hold.

**E1 landed — the write-side plan and its reference emitters.** The harness splits the way
J2 and J3 split, for the same reason: [`encode_plan.py`](../bcir/asn1/encode_plan.py)
compiles a *write* plan (J2's is compiled for reading — it carries `json_kinds` and dispatch
tables, and no ASN.1 tags, because a JER document has none), and
[`emit.py`](../bcir/asn1/emit.py) drives DER, BER, JER and CANONICAL-OER from it. Every
emitter reads **one format-neutral value stream**, which is what makes the costs comparable
at all: hand DER its own octets and JER a Python object and the harness measures the
adapters. All four are byte-identical to the oracle over a 26-case corpus.

*Recorded from building it, and it is an argument for having a neutral stream at all:* the
oracle's three encoders **disagree about how a Python value spells ASN.1 NULL**. `codec`
wants its `NULL` sentinel and refuses `None`; `encode_jer` wants `None` and refuses `NULL`;
`encode_oer` takes either. There is no single value you can hand all three. The ambiguity is
in the value mapping rather than in any encoding, and it stays invisible until something
drives every encoder from one input — which is exactly what a matched comparison must do. It
is pinned by a test rather than papered over, so unifying the spelling stays a deliberate act.

**E2 landed — the C twin, and with it the first native encode timings.**
[`bcir_emit.{h,c}`](../runtime/c/bcir_emit.c) parses the serialized descriptor with a fixed
stack and no allocation, and emits DER, BER, JER and CANONICAL-OER from the same neutral
value stream. It is byte-identical to E1's reference across 152 cases × 4 candidates, at
`-O0` and `-O3` alike.

**The encode column reaches a row the decode column never can.** CANONICAL-OER is measured
here; X.696 §6.2 bars it from the decode table permanently. `CostRow` needs both axes, so
OER still has no two-axis row — `run_native_encode_bench` returns the number it *does* have
and `measured_table` declines to invent the half it does not, which is the same discipline
§6.2 applies one level up.

**PER joined the encode column on the Python rail** once the plan carried what X.691 reads
(§6.2.1–§6.2.3). It is **four** rows rather than one: the ALIGNED/UNALIGNED split is a real
cost trade — ALIGNED pads so multi-octet fields start on octet boundaries, UNALIGNED never
pads — and the CANONICAL/BASIC split decides §19.5's DEFAULT rule. `_emit_per` is
byte-identical to `encode_per` across all four over 268 cases including every constrained
one. The §11 field arithmetic is *imported* from the oracle rather than retyped: those
functions read no schema at all, and a second copy of §11 here would let the parity test
compare two implementations of the same misunderstanding. What E1 supplies independently is
the schema-directed half — bounds, extension markers, enumerations and alternative order out
of a **descriptor** — which is exactly what the C twin must then reproduce.

Building it found two things. The first is a **fifth value-mapping disagreement**: `codec`,
`encode_jer` and `encode_oer` all take a CHOICE value as an `(alternative, value)` pair, and
`encode_per` takes a single-entry mapping and refuses the pair. Same family as the NULL
disagreement §6.2 already records, and invisible for the same reason — only driving every
encoder from one input exposes it. The second is that **§19.2's presence bitmap has OER's
problem**: it precedes all the components while the neutral stream interleaves a presence
octet with each value, so collecting the bits first desynchronizes the reader. It takes
OER's answer too — the bitmap's width comes from the plan and no value can change it, so the
slots are reserved and each bit is patched as its flag is read.

**A cost difference the standard predicts, now measured.** §10.1 forbids DER the indefinite
length form, so a DER encoder must know each constructed length before writing its header —
two passes, or a shift. §8.1.3.6 lets BER leave the length open and close with an EOC, so it
needs one pass and no scratch. On this rail BER encodes at **0.65×** DER's median and
CANONICAL-OER at **0.44×**; the gap is a property of the encodings rather than of the
implementation, which is what makes it worth recording at all.

**And a cost the standard predicts in the other direction.** With PER's four rows measured
natively, the ordering inverts against wire size: PER produces the *smallest* documents and
costs the *most* to write — roughly **2.3× DER** on this host, with the four variants within
3% of each other. Part of that is X.691 itself, whose unit of composition is a bit-field
rather than an octet (§10.5), so an encoder shifts where the octet-aligned rules copy.

**Part of it is this implementation, and saying which is which matters.** `bcir_emit`'s PER
writer emits one bit at a time. A word-at-a-time writer would close some of the gap and none
of §10.5. So the honest claim is the *ordering* — the compact encoding is the expensive one
to produce — and not the multiple, which is why the multiple is written here beside its host
rather than asserted in a test. §8's rule again: shared CI gates validity and trend evidence,
not timing thresholds.

*Three defects found by building the twin, none of them reachable from E1:*

1. **A silent 64-bit truncation.** The first `put_int_decimal` accumulated into a `uint64_t`,
   so `2**64 + 7` emitted as `7` — well-formed JER, wrong value, no error anywhere. Python's
   arbitrary-precision integers meant the reference could never have shown it. The
   conversion is now long division on the magnitude octets, and a width past the buffer is a
   refusal rather than a wrap.
2. **An exponential re-walk**, found by the fuzzer. OER's preamble precedes components that
   the stream interleaves with values, and the first version walked each SEQUENCE's members
   twice to collect the presence bits — so every nesting level re-walked its whole subtree.
   The preamble's *size* comes from the plan and no value can change it, so the space is now
   reserved and the bits patched in during one pass.
3. **An unbounded element count**, also from the fuzzer. A `SEQUENCE OF` whose element
   consumes *no* stream octets — a NULL — turns four attacker-chosen bytes into four billion
   iterations that produce output and read nothing. The bound now comes from the plan, which
   is trusted, rather than from the stream, which is not.

The second and third were reachable only by fuzzing the **descriptor** alongside the value,
which is why `fuzz_emit.c` does, as `fuzz_oer.c` does.

#### 6.2.1 Plan version 3 — a constraint is not a comment

Scoping the PER column turned up two defects in the code above, and both were the same
failure: **the corpus never asked.**

Plan version 2 recorded kinds, tags, member names and optionality, and deliberately dropped
**subtype constraints**. For DER, BER and JER that is correct — X.690 encodes the same
octets whether or not a constraint exists, and X.697 §7.2.2 l)/h) hide integer and string
constraints from JER outright. It is *wrong* for OER: X.696 §10.3 gives a constrained
INTEGER a fixed-width form with no length determinant, so `INTEGER (0..255)` holding 42 is
`2A` where the unconstrained type is `01 2A`. The emitter had been writing the unconstrained
spelling for every type since it landed — a well-formed document of a different value.

The second was found while writing the test for the first: **ENUMERATED shared a branch with
INTEGER**, and X.696 §11 is not §10. Every enumerated this emitter ever encoded was wrong,
constrained or not — `5` came out as `01 05` where the standard says `05`.

Neither needed a subtle input. `_CORPUS` had no constrained type and no ENUMERATED, and a
construct absent from the corpus is untested however many tests run over it. Both corpora —
the Python differential's and `check_runtime.sh`'s — now carry both.

Version 3 records what each rule *reads* off a constraint rather than the constraint itself,
because §5.1 makes a descriptor data and a `Constraint` is an object graph with `permits()`
on it. It carries **four** bound pairs, not two, and that is the part worth stating: X.696
§8.2.2 g) makes an extensible constraint invisible to OER, while X.691 §13.1 emits one bit
and then encodes against the extension *root*. Those are different facts about the same
constraint and they genuinely differ — intersect an extensible `(0..255, ...)` with a plain
`(0..1000)` and OER reads `0..1000` while PER's root reads `0..255` — so deriving one from
the other would be a guess.

The extension-root bounds and the permitted alphabet are PER's alone and nothing emits them
yet. A field nothing reads is a field nothing checks, so the C driver reads the parsed table
back and the differential compares it against the compiler that wrote it.

#### 6.2.2 Plan version 4 — and a third bug of the same family

Writing the test for the ENUMERATED defect turned up a **third**, in the same place and for
the same reason. X.697 §22.2 spells an enumerated value as *"the identifier of the chosen
enumeration item"*, and §22.1 gives it **no numeric spelling at all**. X.690 §8.4 encodes the
number. The two rules disagree by design, and the plan-driven JER emitter shared a branch
with INTEGER — so it wrote `4` where the standard requires `"red"`, a document no JER decoder
can map back to an enumeration item.

This one could not be fixed in the emitter. The identifier is **not derivable** from the
number, so a plan that dropped the enumeration made the correct output unreachable however
careful the emitter was. Version 4 records the enumeration as `(identifier, number)` pairs,
and a bare ENUMERATED — which X.690 and X.696 encode happily from the number alone — is now
**refused at compile time**, because one plan drives four emitters and two of them need what
a number cannot supply.

Version 4 also records the **extension marker** on a SEQUENCE, a CHOICE and an ENUMERATED.
X.691 §19.1, §23.5 and §14.3 each emit a leading bit for it, so two schemas differing only
there encode differently under PER and identically under everything else.

The C reader's four caller-owned tables now arrive as one `bcir_emit_tables` struct rather
than as pointer/capacity pairs. The parameter list grew twice while the format did, and a
positional argument inserted mid-list is precisely the mis-assignment the version check
exists to catch.

**What still stands between this plan and a PER emitter is one thing: extension additions.**
`_compile_members` refuses a component marked `extension`, because X.691 §19.7 splits the
root from the additions and X.690 does not — one plan cannot describe both until the emitter
that needs the split exists. Everything else X.691 reads is now in the descriptor.

#### 6.2.3 Plan version 5 — the DEFAULT a sender must not send

A **fourth** defect, and the only one that hit every emitter at once. X.690 §11.5 forbids
DER an encoding for a component whose value equals its default; X.696 §31.9 and X.697's
CJER say the same; X.691 §19.5 says it for CANONICAL-PER. Versions 1–4 emitted it:

```
SEQUENCE { a INTEGER, b BOOLEAN DEFAULT FALSE }, value { a 1, b FALSE }
  plan-driven DER : 30 06 02 01 01 01 01 00     <- what shipped
  oracle DER      : 30 03 02 01 01               <- what X.690 11.5 requires
```

The corpus supplied `{"a": 1, "b": TRUE}` against `DEFAULT FALSE` — a default component
that *differed* — and never one that matched.

**This one could not go in the value stream**, which is what makes it interesting. The
stream is format-neutral by design, and the rule is not: plain BER keeps the freedom, since
X.690 clause 11 is titled *"Restrictions on BER employed by both CER and DER"*. Deciding
presence once, neutrally, would have taken that freedom away silently and the BER row of the
cost table would have stopped measuring BER.

So version 5 records the DEFAULT **in neutral-stream octets** and each candidate applies its
own law. The comparison is a memcmp against a constant the plan carries, which is sound
because the stream is a *canonical* form of a value — minimal two's complement, UTF-8,
components in plan order — so byte identity is value identity. The freestanding twin
therefore needs no value model of its own to answer a question three encoding rules ask it.

Deciding to omit means the component's octets must still be **consumed**: the stream
describes it, and leaving it unread would leave a suffix `emit` correctly refuses. That is
what `_skip_node` is for, and its C counterpart carries a budget of `default_len + 1` — a
value that has already outrun the default cannot match it, and the early exit also bounds a
`SEQUENCE OF` whose elements consume no stream octets, the same unbounded-count shape the
fuzzer found once already.

#### 6.2.4 What four bugs in one place are actually evidence of

All four were found by *adding a construct to the corpus*, not by reading the code. Each had
survived every parity test, every fuzz run and every `-O0 == -O3` differential, because the
corpus contained no constrained type, no ENUMERATED, and no DEFAULT component whose value
equalled its default — and **a construct absent from the corpus is untested however many
tests run over it.** The fourth is the sharpest case: the corpus *did* carry a DEFAULT
component. It carried the one value for which the rule does not fire.

The dual-rail differential made it worse rather than better in one specific way: the C twin's
expectations come from E1's Python emitter, while E1's parity with the oracle is asserted
separately over `_CORPUS`. A construct present in one corpus and absent from the other lets
**both rails agree on a wrong answer**, which is exactly what happened to the JER enumerated.
`test_the_twin_agrees_with_the_oracle_and_not_merely_with_the_python_rail` now takes the
expectation from the oracle for every case the constrained corpus adds, which closes that
loop rather than widening the corpus and hoping.

### 6.3 StreamPack and BCAB

JER is never a replacement for native StreamPack or BCAB. A JER projection, if a
consumer requires one, is additive and must reconstruct byte-identical native artifacts.
It allocates no new BCAB payload-kind semantics merely to carry source text.

BCAB CRCs and SHA-256 fields detect corruption and bind content; they are not signatures.
Execution authorization remains future trusted-loader work involving signatures,
certificate/revocation policy, relocation validation, W^X, platform loader policy, and
errata admission.

## 7. Delivery phases

| Phase | Deliverable | Exit gate |
|---|---|---|
| **J0 — truth and boundaries** | This roadmap plus reconciled ASN.1/current-state/driver documentation | Docs governance and full repository CI green; no implementation claim added |
| **J1 — bounded Python oracle** · **DELIVERED** | Pre-parse limits, exact canonical-byte validation, schema/path diagnostics, framed input, and `bcir-asn1c` JER encode/decode/transcode modes | Met: limits are asserted at each N/N+1 boundary, every encoder's option that decodes to the right value is refused by octet comparison, every truncated prefix of a framed document is refused, and a failed decode leaves a retry succeeding unchanged |
| **J2 — schema-plan compiler** · **DELIVERED** | Deterministic descriptor, bound derivation, instruction compilation, version/hash contract, and first `channel.json` schema | Met: [`jer_plan.py`](../bcir/asn1/jer_plan.py) regenerates byte-identically, refuses a bare ENUMERATED, an open type, a duplicate JSON member name and an undiscriminable UNWRAPPED choice at compile time, and its plan-driven trace equals the direct one |
| **J3 — scalar C twin** | **Landed.** [`bcir_jer.{h,c}`](../runtime/c/bcir_jer.c): allocation-free bounded scanner, whole-document UTF-8 check, ECMA-404 parser driving a caller's event sink, §4.2 diagnostics, §3.3 unframing, and the twelfth fuzz target | Python/C error-class, byte-offset, required-capacity and event-trace parity in [`test_c_jer.py`](../bcir/tests/test_c_jer.py); `-O0 == -O3` over 667 cases in `check_runtime.sh` `#jer`; freestanding `-Werror` under C11 and C23; ASan/UBSan fuzz green |
| **J4 — law and execution lowering** | **Landed.** Part 1 the transfer-syntax rail and generalized R24 (§5.3); part 2 the commuting projection [`dialect.py`](../bcir/asn1/dialect.py) and StreamPack over JER; part 3 the [`manifest.py`](../bcir/asn1/manifest.py) schemas for `channel.json`, `DeviceManifest` and the §6.2 selection envelope, with §5.4's two sinks | **§7.1's two laws hold** over all 26 law fixtures; **§5.4's commutation holds** over all nine built-in channels — `JER -> typed value -> claims` equals `JER -> direct builder`, both fed by one event walk; native StreamPack octets survive the JER round trip (§6.3) |
| **J5 — hosted SIMD rail** | **Landed, gate MET for UTF-8 validation** ([`bcir_jer_simd.cpp`](../runtime/cpp/bcir_jer_simd.cpp)): a C++17 UTF-8 accept-scanner behind the scalar C ABI, with SSE2/AVX2/NEON tiers, runtime detection and scalar fallback | **Corpus and trace: met, on two materially different aarch64 implementations.** Every tier returns an identical status *and* byte offset to `bcir_jer_validate_utf8` over 489 documents — including multi-byte sequences straddling every offset in a 32-octet block and invalid sequences at every offset — on x86-64, on CI's Ampere/Cobalt **server** ARM runners, and under Termux on a Snapdragon 8 Gen 3 **phone** (`tiers 3 neon 1,0,0,1`). `#jersimd` also cross-compiles the tier this host cannot run and disassembles it, so a tier that silently degraded to scalar fails where nothing executes it — see §7.3.3. **No unsupported-CPU fault: met.** A tier the CPU does not advertise, or this build did not compile, degrades to scalar rather than faulting or refusing. **Advantage on at least two hosts: MET.** [`jer_simd_hosts.json`](../docs/measurements/jer_simd_hosts.json) holds one record per measured machine and [`simd_hosts.py`](../bcir/asn1/simd_hosts.py) decides admissibility against §8 — dedicated tenancy *corroborated by the host's own steal and throttling counters*, a vector tier, enough rounds for an order-statistic interval, and every round on one CPU. **Two hosts on two architectures are admitted**: Samsung S24+ / Snapdragon 8 Gen 3 on NEON at **10.4×** (`[886, 937]` ns against `[9739, 9739]`), and the Claude Code cloud container on x86-64 AVX2 at **23.2×** (`[552, 554]` against `[12807, 12818]`). A test fails if the store and this table ever disagree. §7.3 has the numbers and the mis-diagnosis that delayed them. Covers **UTF-8 validation only**; the structural index is a separate build and §7.4 says why it is not the same shape. It is **now landed and vectorized** ([`bcir_jer_index.cpp`](../runtime/cpp/bcir_jer_index.cpp)): `bcir_jer_scan_cursor` exports the dispatch's state so the index rebuilds *only the dispatch* and reuses the token scanners verbatim, with an SSE2/AVX2/NEON whitespace pass behind the UTF-8 rail's own tier detection. Proven identical to `bcir_jer_scan` in status, offset, `needed` and node count at **every compiled tier**, across fifteen work ceilings per document — including ceilings that fail inside a bulk whitespace charge. Measured (informally, one host) at **1.04–1.09× on dense and pretty-printed input, 2.3× on wide indent and ~31× on whitespace-heavy input** — no document shape is a loss, after two corrections §7.4.2 records: the per-octet charge moved into the header so both rails inline one definition of it, and the tier branch hoisted from per-run to per-document. A whole-program LTO build is what proved the original diagnosis wrong. `bcir_jer_scan` still stays the default, because parity-plus on one host is not §8 admission. See §7.4.1–§7.4.2 |
| **J6 — certified K_BCIR choice** | **Landed on the Python oracle** ([`certified.py`](../bcir/asn1/certified.py)): distribution-free prediction intervals from order statistics, a frozen generation-tagged cost table with declared provenance, §6.2's certificate, and a production select that **refuses** an oracle table for any timing objective. [`native_bench.py`](../bcir/asn1/native_bench.py) now produces a genuinely `measured` table from a native C harness and refuses every candidate the C rail does not implement, and `select_budgeted` adds RCSP with the union-bound coverage decay reported rather than hidden — this row previously said both remained open, which §2's inventory row had already contradicted. **The native encode column is built** — `run_native_encode_bench` measures 9 of the 10 candidates natively (every one E2 can emit; only BASIC-OER is skipped, and by the stated law that it is not a distinct encode cost from CANONICAL-OER), and `measured_table` carries a real encode interval rather than a copy of the decode figure. This row said otherwise until the closing audit; it was written before `bcir_emit` landed. Still open: **target hardware counters** (no PMU on either available host) and the two-axis consequence §6.2 records — `CostRow` needs an encode AND a decode interval, and X.691 §7.2 / X.696 §6.2 deny PER and OER a schema-free decode permanently, so those rows have an encode number and no two-axis row | Exact sizes decide wire-size objectives with no timing consulted; repeatability is a refusal rather than an average; legality-first and canonical-or-excluded precede every comparison; deterministic selection on two tables, each certificate bound to the table digest it read — [`test_asn1_certified.py`](../bcir/tests/test_asn1_certified.py) |
| **J7 — driver experiment** | Userspace/simulator driver specification ingest, generated views, and sequential BCIR-Linux module comparison. **Blocked on access, not design**: D0/D1 need kernel headers and module loading, D2/D3 need a real device or VFIO with an IOMMU, and neither available host has any of them — see [`BCIR_TARGET_ACCESS.md`](BCIR_TARGET_ACCESS.md) | D0–D3 driver gates, signed modules, direct/Linux trace parity, teardown/restart tests, and controlled performance evidence |

User-defined ECN classes were closed after the J0 sign-off: the built-in sets and ordinary
BCIR lowering contracts are sufficient for the measured fixed-candidate result, and
reopening ECN required a written workload, a missing-expressiveness proof, and approval
separate from the JER implementation.

**All three are now on the record and the half is built** ([`ecn_user.py`](../bcir/asn1/ecn_user.py)).
The workload is a fixed-layout frame header whose length field is scaled in 4-octet units,
whose flag is active low, which carries two reserved bits, and which transmits the length
before the version. The proof is executed rather than written: all five fixed candidates run
against the same abstract value, and none reproduces the octets. Canonical PER is the case
worth reading — it lands on the *same octet count* and different octets, so the gap is
expressiveness and not compactness. The structural reason is that DER, PER, OER and JER all
encode the abstract value, while the header needs the transmitted value to be a declared
function of it; that function is `#TRANSFORM`, and no constraint tightening or
canonical-variant choice in the fixed set produces one.

The closure this replaces was correct on the evidence it had. What changed is the evidence,
not the standard — and the test that carries it fails if a fixed candidate ever matches,
so the paragraph cannot outlive the fact.

### 7.1 J4's bidirectionality extension

J4 was originally scoped as a **one-way** lowering: JER text in, claims and StreamPack
bytes out. [`BCIR_JSON_PROGRAM_REPRESENTATION.md`](BCIR_JSON_PROGRAM_REPRESENTATION.md)
examines whether JER can also carry *programs* — BCIR's own IR — and concludes that this
is not a new representation but a **third serialization of the `bcir.*` dialect BCIR
already has**, alongside MLIR textual assembly and MLIR bytecode. That conclusion adds one
requirement to J4, recorded here so it is designed for rather than retrofitted:

> The projection between the `bcir.*` dialect and its JER form must **commute in both
> directions**. `MLIR -> JER -> MLIR` must be the identity on the dialect, and
> `JER -> MLIR -> JER` must be byte-identical under the canonical profile.

**Both hold as of J4 part 2** ([`dialect.py`](../bcir/asn1/dialect.py)), over all 26
modules in the law fixture corpus — including the *negative* ones, because a projection is
not a filter: a module R24 rejects must round-trip as faithfully as one it accepts, or a
document that lost an attribute in transit could come back legal.

The asymmetry between the two laws is deliberate and is the part most easily got wrong.
Canonical JER defines exactly one octet string per abstract value, so a byte claim is
meaningful there. **MLIR textual assembly defines no such thing** — `mlir-opt` may reprint
attributes in another order and remain correct — so a byte claim about MLIR text would be a
statement about a formatter, failing for reasons unrelated to the projection. What must
survive that direction is the dialect itself, which is what "identity on the dialect" says.

Building it produced one finding in a neighbouring module. The BCIR-StreamPack schema's
`Lane` and `Dispatch` carried **no enumeration**, though this file's own ASN.1 comment block
declared both. DER and OER encode an enumeration's *value* (X.690 §8.4, X.696 §11), so the
omission was invisible for as long as those were the only projections; X.691 §14 needs the
*index* and X.697 §22.2 needs the *identifier*, neither derivable from the number. The
module was therefore DER/OER-only by accident rather than by design. Adding the
enumerations is additive — DER octets are unchanged, and `test_asn1_streampack.py` pins
that — and it is what makes `encode_pack_jer` emit `"lane":"t"` rather than a number.

Nothing else in J0–J7 changes. J3 is unaffected and more strongly motivated: if JER can
carry programs, the bounded C parser sits on a materially more important path. Phase H
gains a consumer rather than a new law. The program-representation work has its own phase
ladder (P0–P6) in that note; P1 depends on J2, P2 depends on J4, and no P phase is
scheduled ahead of the JER phase it depends on.

### 7.2 What building the C twin found in the Python rail

Three defects, all in `jer_bounded.py`, all fixed in the same change as J3. They are recorded
because the pattern generalizes: a second implementation is worth more as a *question asked
of the first* than as a performance artifact.

1. **An unpaired surrogate escaped the §4.2 contract entirely.** `json.loads` accepts
   `"\ud800"` and returns a `str` holding a lone surrogate — a value with no UTF-8 encoding,
   which is exactly why `jer.py`'s *encoder* already refused to emit one. The decoder had no
   matching refusal, so the rail could decode a value it could never re-encode. Under the
   canonical profile that was worse than a wrong value: `_canonical` re-encoded, the encoder
   raised its §7.6.2 error, and that error left `decode_bounded` **unstructured** — no
   `JerDiagnostic`, no stable code, no byte offset, contradicting §4.2 for an input an
   attacker chooses freely. Both rails now refuse it in the octet pass, as `NOT_UTF8` rather
   than `MALFORMED`: the JSON is well formed and it is the *encoding* that has no answer.
2. **The bounding pass admitted a number grammar ECMA-404 does not have.** `01` passed
   `scan` and was refused downstream by `json.loads`, arriving as a `SCHEMA` diagnostic for
   what is a lexical fault. The accept/reject decision was already right; the diagnostic was
   not, and a reader that admits a leading zero has a grammar the encoder does not.
3. **Two work-accounting loops charged different offsets** for the same exhausted budget —
   `pos` in the main loop, `pos + 1` inside `_scan_number`. `work` is the one limit whose
   diagnostic offset is not otherwise derivable, so the C twin would have had to reproduce
   the discrepancy to stay in parity. Made uniform instead.

A fourth item is a deliberate divergence rather than a defect: `bcir_jer_parse` reports
`TRAILING_INPUT` where the Python rail reports `SCHEMA`, because on that rail the trailing
octets are `json.loads`'s complaint rather than the bounding pass's. The C diagnostic is the
better one; the acceptance decision is identical, which is what the parity test compares.

### 7.3 J5's measurement, and the clause it does not close

| document (≈30 KB) | scalar | AVX2 | speedup |
|---|---|---|---|
| all ASCII | 28470 ns | 1029 ns | **27.7×** |
| one accent near the front | 28438 ns | 1138 ns | **25.0×** |
| an accent in every node | 36614 ns | 15056 ns | 2.43× |
| CJK in every node | 75353 ns | 27498 ns | 2.74× |
| emoji in every node | 19012 ns | 12990 ns | 1.46× |

**No second UTF-8 implementation, and that was a design decision rather than a shortcut.**
The obvious way to accelerate multi-byte text is to vectorize UTF-8 validation itself. That
would be a second definition of *valid UTF-8* — the risk §8's table names — and a bug in the
fast one produces a **wrong accept**, the silent failure. §4.1 already settles it: *"the
scalar rail is authoritative for native parser correctness. SIMD is an optimization
candidate, not a separate semantic implementation."*

It is also unnecessary. **An ASCII octet can never be a continuation octet**, because
continuations are `80`–`BF`. So the next ASCII octet is always a sequence boundary, and no
legal multi-byte sequence can span it — which means `[first non-ASCII, next ASCII)` can be
handed to the scalar rail *in isolation* and yields exactly the answer validating it in
context would, including for a truncated sequence, which is invalid either way. The vector
passes therefore answer only **where the runs are**, never what is valid, and the two runs
alternate.

*The first version got this wrong in an instructive way.* It handed everything from the
first non-ASCII octet to the **end** of the document to scalar, so a single `café` near the
front cost a 29 KB document its entire acceleration: **1.00×**. Alternating restored it to
25×. The regression is pinned by a test asserting the early-accent ratio against the
all-ASCII ratio, so a return to a single hand-off fails rather than merely slows down.

**The two-host clause is MET.** Two dedicated hosts on two architectures, each with a vector
interval strictly below its scalar interval:

| ≈29 KB all-ASCII claim graph | scalar | vector | speedup |
|---|---|---|---|
| Samsung S24+, Snapdragon 8 Gen 3, **NEON**, pinned to cpu7 | `[9739, 9739]` ns | `[886, 937]` ns | **10.4×** |
| Claude Code cloud container, x86-64, **AVX2**, pinned to cpu1 | `[12807, 12818]` ns | `[552, 554]` ns | **23.2×** |

`python3 -m bcir.asn1.simd_hosts` renders the verdict from
[`jer_simd_hosts.json`](measurements/jer_simd_hosts.json), and a test fails if this prose and
that store ever disagree — which is how the paragraph came to say *met*: the evidence landed
and the test refused to let the sentence stay.

The aarch64 CI lane is a second *machine*, and the NEON path is proven **correct** there —
`tiers 3 neon` resolves and all 489 documents agree with the scalar rail on status and
offset. It is not a second *measured host*: §8 refuses timing thresholds on shared runners
("shared CI gates validity and trend evidence, not noisy timing thresholds"), so a
controlled rig is what closes this clause.

#### 7.3.0 The aarch64 measurement, and what it cost to be admissible

| ≈29 KB all-ASCII claim graph | scalar | vector | speedup |
|---|---|---|---|
| Samsung S24+, Snapdragon 8 Gen 3, **NEON**, pinned to cpu7 | 9739 ns | 937 ns | **10.4×** |
| x86-64, **AVX2** (the table above) | 28470 ns | 1029 ns | 27.7× |

`python3 -m bcir.asn1.simd_hosts` **admits** the S24+ record: dedicated, tier `neon`, 41
rounds per candidate, every round on cpu7, and a vector interval `[886, 937]` strictly below
a scalar interval `[9739, 9739]`. That is **one of the two hosts** the clause needs.

**NEON's 10.4× against AVX2's 27.7× is the expected shape, not a disappointment.** NEON is 16
octets wide and AVX2 is 32, and the ASCII-run kernel's whole job is to answer *"are these all
ASCII"* per block — so halving the block roughly halves the run each vector step skips. The
Cortex-X4's *scalar* rail is also nearly 3× faster than the x86 figure (9739 ns against
28470 ns), which shrinks the headroom a vector pass can win back. Both ratios are what a
reader should expect on that silicon, which is why every figure now carries its host.

**The phone produced a cleaner measurement than the container's first attempt did**, which
is worth recording because it inverts the intuition that a handset is the noisy environment.
Pinned to the big core, on mains, 37 of 41 scalar rounds landed on *exactly* 9739 ns.

**And the reason the container's first run looked worse was diagnosed wrongly at the time.**
It came back bimodal at roughly 500 and 2500 ns, and that was attributed to contention from
other tenants — the machine being a cloud container. It was not. That run was simply
**unpinned**: the same core-migration failure §7.3.2 builds a guard for on the phone, hiding
in plain sight on x86. Pinned to cpu1 the same container yields a scalar interval of
`[12807, 12818]` ns — an eleven-nanosecond spread across 41 rounds.

The mislabelling had a second cause worth naming: `dedicated` was a **declaration nobody
checked**, so a host was classified by *what kind of machine it is* rather than by what it
was doing. A record now carries the hypervisor steal ticks and cgroup throttling accumulated
**during the measured rounds**, and either being nonzero refuses a `dedicated` claim. This
container reports zero of both, no CPU quota (`nr_periods 0`) and load 0.01 — so the claim is
now backed by the machine's own accounting instead of by an assumption about its hosting.
The check can only ever catch a *false* declaration: a host that does not report the counters
is not penalised for staying silent.

**NEON is now also proven correct on a phone-class core**, which CI could not have shown.
GitHub's ARM runners are Ampere/Cobalt *server* parts; a Snapdragon 8 Gen 3 is a
Cortex-X4/A720/A520 cluster with different pipeline widths and a different memory system.
Running the gate under Termux on a Samsung S24+ reports `tiers 3 neon 1,0,0,1` — NEON
compiled and resolved, no x86 tier present — and the same **489 documents × 5 tiers agree
with the scalar rail on status and offset**, with 40 distinct rejection offsets exercised.
That is the corpus-and-trace clause met on a second, materially different aarch64
implementation rather than on a second instance of the same one.

#### 7.3.1 The clause is now decidable, and the store is the authority

The rule used to live only in this prose, which meant "is it met yet" was a question you
answered by reading. It is now a function of evidence:
[`docs/measurements/jer_simd_hosts.json`](measurements/jer_simd_hosts.json) holds one record
per machine somebody ran [`tools/silicon/measure_jer_simd.sh`](../tools/silicon/measure_jer_simd.sh)
on, and [`simd_hosts.py`](../bcir/asn1/simd_hosts.py) decides admissibility against §8's
rules — `python3 -m bcir.asn1.simd_hosts` prints the verdict. A record is admissible only
when it declares a **dedicated** machine, resolves to a **vector** tier, carries enough
rounds for an order-statistic interval, and ran every round on **one CPU**. Two admissible
hosts on two **architectures**, each with a vector interval strictly below its scalar
interval, close the clause; nothing else does.

Two architectures rather than two machines is the load-bearing reading. A vector rail can be
fast on the ISA it was written for and a wash on another, and NEON is the path no x86 host
exercises at all — so two x86 boxes agreeing is the same evidence twice.

**The advantage is disjoint intervals, never a ratio.** `certified`'s distribution-free
order statistics already exist for this: a median speedup with no spread beside it is a
number that happens to be true of one run, and a normal-theory interval would assume a
distribution timing data does not have.

#### 7.3.2 What a phone changes, and what it does not

The first *dedicated* aarch64 machine within reach is a handset rather than a Raspberry Pi,
and the runbook is written for one. A Snapdragon 8 Gen 3 is big.LITTLE — one Cortex-X4, four
A720s, three A520s — and **the difference between the largest and smallest core exceeds the
advantage being measured**. Three consequences:

1. **Which CPU each round ran on is now recorded.** `test_jer_simd.cpp`'s bench reports
   `sched_getcpu()` per round, and a record whose rounds span more than one CPU is refused:
   that is two machines averaged, not one machine measured. `--pin N` asks for a core and
   the record says whether the kernel agreed.
2. **A migrating run refuses itself even without that check.** Migration makes the samples
   bimodal, the interval widens to span both modes, and the overlap test then reports no
   advantage. The protocol degrades to *unproven* rather than to *wrong* — asserted by a
   test, because a safety property nobody checks is a hope.
3. **Thermal throttling needs no new machinery.** It is monotone drift, and the runbook
   interleaves scalar and vector rounds so a downward frequency ramp is spread evenly across
   both — the same argument `bcir_asn1_bench.c` already makes for its own round-robin.

#### 7.3.3 Every host compiles every tier

The differential can only exercise the tiers the running CPU *has*, so an x86 developer
editing the NEON path got no feedback until CI — and an aarch64 one got none on SSE2/AVX2.
That is how a tier rots.

Clang carries every backend, so `#jersimd` now **cross-compiles the other architecture's
tier** on whichever host it runs. Compile-only needs no sysroot: the file includes clang's
own `stdint`/`arm_neon` headers and nothing from libc.

**"It compiled" is deliberately not the check.** With `BCIR_SIMD_ARM` never defined the file
still compiles — to scalar, silently — which is exactly the failure worth catching and
precisely the one no run on an x86 host could show. So the object is disassembled and the
tier's own instructions must appear: `umaxv`/`uminv` over a `.16b` register for NEON,
`pmovmskb`/`pcmpgt` for x86. Deliberately breaking the ARM guard makes the gate fail while
every other check on that host still passes, which is what says the check has teeth.

The NEON kernel it proves present is two vector instructions, and that is the design rather
than a shortfall:

```
ldr   q0, [x10, x11]      load 16 octets
umaxv b0, v0.16b          horizontal max across all 16
tbz   w13, #0x7, ...      bit 7 clear => all 16 are ASCII, continue the run
ldrsb w12, [x10, x11]     otherwise hand off to the scalar rail
```

*"Vector says WHERE, scalar says WHAT"* needs exactly a horizontal reduction and a bit
test.

Correctness runs first and a failure aborts before anything is timed: a number from a build
that disagrees with the scalar rail is not a weak measurement, it is a measurement of the
wrong answer.

A test compares this section against the store and fails if the store ever closes the clause
while the prose still says otherwise — so the evidence leads and the paragraph follows.

### 7.4 The structural index is a different problem from the UTF-8 rail

§1's pipeline lists *"optional hosted SIMD structural index"* next to the UTF-8 scanner, and
they look like the same kind of work. They are not, and the difference is worth stating
before someone builds the wrong thing.

**`bcir_jer_validate_utf8` has no cost budget.** Skipping an ASCII run is semantically free:
the function's answer is a property of the octets alone, so a vector pass that proves a run
irrelevant can skip it and change nothing.

**`bcir_jer_scan` charges one work unit per octet**, against §4.3's `work` ceiling, and
`BCIR_JER_WORK_EXCEEDED` carries *the exact octet at which the budget ran out*. A ceiling of
100 rejects a 410-octet document at octet 100, needing 101. So the scan's cost is not
incidental — it is **observable output**, and §4.3 designed it that way so a sender "cannot
buy unbounded work with few bytes".

Two consequences for any structural index:

1. **A vector pass may not simply skip — but it may charge in bulk, exactly.** Skipping a
   run without charging for it accepts documents the scalar rail rejects. Charging in bulk,
   however, is not an approximation: the main loop charges **exactly one unit per octet, at
   that octet's own position**, so a run of `n` octets starting with `w` units already spent
   against a ceiling of `L` fails — when it fails — at octet `L - w` reporting `needs L + 1`,
   in closed form. Verified over every ceiling from 1 to 59 against `jer_bounded`: zero
   mispredictions.

   *This corrects the first version of this section*, which claimed a crossing run had to be
   re-walked per octet to report the right offset. It does not: uniform positional charging
   makes the failure point arithmetic. The budget therefore constrains the design without
   capping the speed-up, which is a materially different conclusion.
2. **It cannot be a drop-in accelerator.** The UTF-8 rail works because the C ABI exposes a
   whole-document function the C++ adapter can wrap. `bcir_jer_scan` carries its state
   internally, so accelerating it means either putting SIMD inside the freestanding core —
   the wrong direction across the layering §4.1 sets up — or a **stage-2 parser that walks
   the index**, which is a second scanner and needs §8's full mitigation: same C ABI, same
   corpus, scalar fallback, and differential fuzzing against `bcir_jer_scan`'s events,
   diagnostics, offsets *and* work accounting.

   *This overstated the choice.* Both options assume the state `bcir_jer_scan` carries is
   indivisible, so that reaching it means either breaching §4.1's layering or duplicating
   the whole scanner. It is not: the state splits at a seam, and §7.4.1 records where.

### 7.4.1 The seam: a second dispatch loop, not a second scanner

`bcir_jer_scan`'s loop is a **dispatch** — skip whitespace, recognise a structural octet, or
hand off to a token scanner — wrapped around token scanners that hold the semantics: §4.3's
`string_bytes` and `number_digits` limits, escape validity, the exponent ceiling. Only the
dispatch is vectorizable. Nothing about finding the next non-whitespace octet requires
knowing what a valid `\u` escape is.

So `bcir_jer.h` exports the dispatch's state as `bcir_jer_scan_cursor` — the limits, the work
spent, the diagnostic sink — with four entry points onto the existing scanners
(`bcir_jer_scan_spend`, `..._string_token`, `..._number_token`, `..._literal_token`). They are
thin forwards to statics that already existed; no semantics moved, and the freestanding core
stays freestanding under C11 and C23.

`runtime/cpp/bcir_jer_index.cpp` then rebuilds **only the dispatch**. That is what makes it a
second dispatch loop rather than the second scanner consequence 2 predicted, and it changes
what §8's table has to cover: the differential still needs the same C ABI, the same corpus, a
scalar fallback and fuzzing against status, offset, `needed` and node count — but the surface
under test is one loop, not a parser. §4.1's "no second semantics rail" survives the
optimization instead of being spent on it.

**The seam was proven scalar first, deliberately.** A differential that only begins to exist
alongside the optimization cannot tell you which of the two broke it. So the rebuilt loop was
shown to reproduce `bcir_jer_scan`'s status, offset, `needed` and node count over the corpus —
at fifteen work ceilings per document, including ceilings that fail *inside* a bulk whitespace
charge, the one place the two rails compute the answer differently rather than identically —
while there was still only one variable. A structural test reads the source and refuses it if
it names a §4.3 limit or a UTF-8 boundary of its own.

**The vector pass then landed under that harness.** `whitespace_run` is the one function the
tier dispatch replaces, and consequence 1's closed form is what licenses it: skipping ahead a
block at a time cannot lose the budget's failure position, because that position is arithmetic
rather than a consequence of having walked there. SSE2 and NEON settle sixteen octets per
block, AVX2 thirty-two, each returning the last block boundary still entirely whitespace; one
scalar line then makes the bound exact, so a tier can only ever be conservative, never wrong
about where a run ends.

Three things keep this from being the second semantics rail:

- **One whitespace set, named once.** The scalar predicate and every vector width are written
  against the same four constants, and a test fails if the octets are ever spelled out twice.
  A vector pass that also matched FORM FEED would accept documents the scalar rail refuses —
  and only for runs long enough to reach a wide block.
- **One CPU detection.** Tier resolution is the UTF-8 rail's `bcir_jer_simd_tier_available`,
  not a second probe, so J5's "no unsupported-CPU fault" clause holds on both rails or
  neither. A test fails if the index calls `__builtin_cpu_supports` or `getauxval` itself.
- **Every compiled tier is swept**, not just the widest the CPU advertises — a tier that
  degraded to scalar would otherwise pass by never running — and `#jerindex` cross-compiles
  the tier this host *cannot* run and disassembles it, so a tier that lowered to scalar fails
  where nothing executes it.

`#jerindex` also rebuilds the freestanding core at C11 and C23 on every run. Exporting a seam
is where a hosted dependency most easily leaks into a core that must not have one, and no test
that *runs* the index would notice, because the hosted build has libc.

**A short-run guard, and why it is exact.** A vector helper cannot advance unless its first
whole block is entirely whitespace, so a run shorter than the narrowest block can never reach
it. The loop therefore walks one block scalar *first* and goes wide only if still in
whitespace: a document of short runs then pays exactly what the scalar tier pays, because the
walk is the work that tier would do anyway, while a long run pays at most sixteen iterations
once. Octets below the block are settled by the same comparison either way, so no tier's
answer changes.

#### 7.4.2 What the measurement says, and the diagnosis it overturned

Informal timing, x86-64 AVX2, pinned, medians of 31 rounds × 200 iterations, three repeats.
**Not** an admitted §8 record — one host, and §8 admits a speed-up only with non-overlapping
intervals on two. Absolute times on this container move by 2× between sessions, so only
*within-run* ratios are meaningful and only those are quoted.

| Document | `bcir_jer_scan` → index, scalar tier | scalar tier → vector tier | net |
|---|---|---|---|
| Minified, 6.7 KB, no whitespace | 0.97–1.00× | 1.09–1.11× | **1.06–1.09×** |
| Pretty-printed, 8-space indent, 14 KB | 0.99–1.03× | 1.04–1.05× | **1.04–1.08×** |
| Wide 64-space indent, 16 KB | 1.40–1.49× | 1.57–1.60× | **2.20–2.34×** |
| 20 KB leading whitespace | ~1.26× | 24.4–24.7× | **30.8–31.1×** |

No document shape is a loss. Getting there took two corrections, and both are worth keeping
because in each case the obvious diagnosis was wrong.

**The first version of this section blamed the token-scanner calls.** The index was 0.51–0.55×
on pretty-printed input, and the natural explanation was that `bcir_jer_scan` inlines its
helpers while the index reaches them across a module boundary. A whole-program **LTO build
settles it**: LTO inlines everything and moves that case *not at all*. The diagnosis was wrong,
and would have justified a large per-implementation build refactor that bought nothing.

Two things were actually costing:

1. **The per-octet charge, which really was a cross-module call.** §4.3's budget arithmetic is
   four lines, and the scalar rail gets it inlined for free because it lives in the same
   translation unit. Moving the definition into `bcir_jer.h` as `bcir_jer_scan_charge` — so
   `bcir_jer.c`'s own `spend` *is* that function, rather than a second copy of it — took
   minified from 0.69× to 1.10×. A duplicated copy would have been just as fast and would
   have been the defect this seam exists to prevent.
2. **The per-run tier `switch`.** Choosing a vector width once per whitespace *run* costs more
   than the bulk charge saves when runs are one and eight octets long, which is what ordinary
   indentation is. The loop is now instantiated once per tier and the branch is taken once per
   *document*; that alone moved pretty-printed from 0.83× to 1.04×.

**The guard was corrected the same way.** Probing the octet at width − 1 to skip a pointless
vector call is exact and did fix the original regression — but it is a test the scalar tier
does not pay, and it left the vector tier at 0.79–0.83× of the index's *own* scalar tier: an
accelerator still losing to the thing it accelerates, by less. Walking one block scalar before
going wide has no such asymmetry, because the walk is work the scalar tier does anyway. It
costs the 64-space case some headroom — 3.1× down to 2.3×, since a quarter of each run is now
walked before the vector starts — and that is the right trade: being at or above parity on
every shape matters more for a rail that might be selected by default than a larger win on an
indent width few documents use.

**`bcir_jer_scan` nonetheless remains the default.** Parity-plus on one host is not §8
admission, and nothing in the pipeline switched to the index. What changed is that the index
is no longer *disqualified* — it is now a candidate that a two-host measurement could admit,
where before it was a rail that lost on the documents people actually have.

## 8. Validation and performance method

### 8.1 Correctness corpus

The differential corpus covers every supported ASN.1 kind and JER instruction, plus:

- every byte split and truncated prefix of valid framed documents;
- invalid UTF-8, byte-order marks, control characters, bad and lone surrogates;
- malformed escapes, escaped member aliases, duplicate keys, unknown fields, and
  BASIC-versus-canonical member order;
- integer boundaries, excessive digits/exponents, negative zero policy, and real
  special-value encodings;
- depth/node/member/array/string/scratch/output limit boundaries;
- open-type selector misses, extension additions, renamed fields, defaults, ARRAY null
  elision, and UNWRAPPED ambiguity;
- late semantic failure after an otherwise valid prefix; and
- repeated cleanup/retry with the prior output and active generation unchanged.

Fuzzers are bounded in pull-request CI and extended in scheduled cloud jobs. Every defect
receives a deterministic regression.

### 8.2 Benchmark matrix

Measure parsing separately from parsing plus schema conversion and full lowering:

| Candidate | Purpose |
|---|---|
| Current Python JER | semantic/reference baseline |
| Generated scalar C | portable native correctness and allocation baseline |
| Optional C++ SIMD | acceleration candidate |
| External simdjson | non-normative hosted comparison |
| COER and canonical PER | binary transfer alternatives |
| Native StreamPack | execution-path floor after compilation |

Fixtures include small control messages, wide objects, nested manifests, long strings,
numeric-heavy arrays, mixed valid/invalid traffic, and representative claim batches.
Report bytes/s, documents/s, cycles/byte, instructions, branches/misses, allocations,
peak scratch/output, cold/hot latency, tails, and variance with intervals.

No absolute “machine-speed JSON” or “thousands of instructions per millisecond” claim
ships without reproducible evidence. SIMD is admitted only when it beats scalar C with
non-overlapping intervals on a declared target and remains byte/error/trace equivalent.
Bare-metal performance floors stay in the hardware runbook; shared CI gates validity
and trend evidence, not noisy timing thresholds.

## 9. Driver and kernel boundary

An ASN.1 schema may describe register fields, packets, manifests, configuration, and
telemetry. It does not authorize accesses or change the hardware layout. Device register
offsets, bit widths, endianness, reserved values, access classes, barriers, and errata
remain authoritative device facts. K_BCIR may optimize access order, staging,
coalescing, prefetch, and explicitly permitted packet alternatives only after law and
simulator checks.

The first driver experiment runs through userspace/direct RuntimeChannel and a simulator.
Linux testing then uses disposable BCIR-Linux/cloud runners. Drivers are compared on
separate virtual devices or through sequential binding; Linux does not bind competing
drivers to one physical device simultaneously. The local workstation is not a kernel
module test rig.

Dynamic replacement additionally requires:

- authenticated artifacts and module-signing policy;
- UAPI/feature compatibility, not Linux internal-ABI promises;
- quiescence and consistency across tasks and callbacks;
- state-transfer and cancellation semantics;
- generation-tagged handles, stale-event refusal, and peer/device-death behavior;
- rollback and idempotent teardown; and
- isolation and W^X loader policy.

These requirements follow the Linux
[driver-binding](https://docs.kernel.org/driver-api/driver-model/binding.html),
[module-signing](https://docs.kernel.org/admin-guide/module-signing.html),
[livepatch consistency](https://docs.kernel.org/livepatch/livepatch.html), and
[unstable internal driver-interface](https://docs.kernel.org/process/stable-api-nonsense.html)
contracts. They remain in the driver/kernel roadmap until J1–J6 and the D0–D3 driver
gates are proven.

## 10. Risk register

| Risk | Control / stop condition |
|---|---|
| “Schema-driven” becomes “parser-free” marketing | Keep lexical/UTF-8/bounds work explicit and benchmark the complete path |
| Canonical profile accepts alternate bytes | Byte-level canonical corpus and re-encode equality before optimized direct checks |
| Generated parser and table parser diverge | Require identical event, diagnostic, and final-offset traces |
| SIMD creates a second semantics rail | Same C ABI/corpus, scalar fallback, differential fuzzing, and runtime feature checks |
| Direct sinks expose partial state | Scratch/rollback transaction with one publish point after full validation |
| Schema constraints are mistaken for memory safety | Preserve BCIR laws, ownership rules, checked arithmetic, effect verification, and sanitizers |
| JSON enters the driver data plane | Architecture gate rejects JER in interrupt/DMA/submission paths |
| Measurement becomes legality | Two-truth quarantine and a certificate that records legality independently of costs |
| ECN scope returns without evidence | Reopen only with an approved workload and missing-expressiveness proof. **Both are on the record**; the proof is a test that runs all five fixed candidates and fails if any ever reproduces the target octets, so the justification cannot silently rot |
| Integrity is described as authenticity | Keep BCAB CRC/SHA and future signature/loader policy separate |
| Physical register layout is “optimized” illegally | Device schema fixes MMIO facts; optimize only access schedules and declared packet variants |
| JER-as-program-representation forks a second IR | J4's §7.1 gate: the projection must reproduce the existing `bcir.*` dialect exactly, in both directions |
| Linux module experiments endanger a workstation | Disposable CI/BCIR-Linux environments, bounded tests, signed artifacts, and explicit hardware gates |

## 11. References

- ITU-T X.697 (02/2021), *ASN.1 encoding rules: Specification of JavaScript
  Object Notation Encoding Rules (JER)*:
  <https://www.itu.int/rec/T-REC-X.697-202102-I>
- RFC 8259, *The JavaScript Object Notation (JSON) Data Interchange Format*:
  <https://www.rfc-editor.org/rfc/rfc8259.html>
- RFC 8785, *JSON Canonicalization Scheme (JCS)*:
  <https://www.rfc-editor.org/rfc/rfc8785.html>
- RFC 7464, *JavaScript Object Notation (JSON) Text Sequences*:
  <https://www.rfc-editor.org/rfc/rfc7464.html>
- Langdale and Lemire, *Parsing Gigabytes of JSON per Second*:
  <https://arxiv.org/abs/1902.08318>
- [`BCIR_JSON_PROGRAM_REPRESENTATION.md`](BCIR_JSON_PROGRAM_REPRESENTATION.md) — whether
  schema-bound JSON can carry BCIR programs, and the P0–P6 ladder that would follow. §7.1
  above records the one change it makes to this roadmap.
- Li et al., *Mison: A Fast JSON Parser for Data Analytics*:
  <https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/mison-vldb17.pdf>
- Linux kernel documentation: driver binding, module signing, livepatch consistency,
  and the intentionally unstable internal driver interface, linked in §9.
