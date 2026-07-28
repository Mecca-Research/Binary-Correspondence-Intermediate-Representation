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
  set cannot express.

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
| ECN | **Part 1 landed** | [`ecn.py`](../bcir/asn1/ecn.py) models classes, objects, object sets, EDM/ELM, and built-in BER/PER sets; it does not parse or lower user-defined encoding classes |
| JER bounded reader in C (J3) | **Landed on the C rail** | [`bcir_jer.c`](../runtime/c/bcir_jer.c) runs the same three stages in the same order — §4.3's limits in one octet pass, §7.6.2's encoding, then the grammar — with **no allocation, no recursion and no floating point**: the container stack and decode scratch are the caller's, and a number event hands back the raw token rather than a parsed double. Canonical-byte validation (§3.2) and schema legality stay on the Python rail and are deliberately *not* reimplemented, because a second definition of canonicality is free to drift from the encoder that is the actual definition. Building it found three defects in J1's rail; see §7.2 |
| JER schema plan (J2) | **Landed on the Python oracle** | [`jer_plan.py`](../bcir/asn1/jer_plan.py) compiles a root type into the §5.1 descriptor — identity and source hash, family/profile/instruction hash, sorted member dispatch, required/default/extension metadata, recursion bounds and static capacity. **Static capacity is `None` almost everywhere**, and that is a property of JER rather than of the compiler: §7.2.2 l) and h) hide integer and string constraints from a JER encoder, so only BOOLEAN, NULL, ENUMERATED and a single-size BIT STRING (§7.2.1 a) are derivable. J3's C interface must therefore take its capacity from the caller |
| Encoding selection harness | **Landed as measurement evidence** | [`selection.py`](../bcir/asn1/selection.py) measures exact wire size and Python-oracle timing for a fixed candidate set; it is not a native K_BCIR table or selection certificate |
| Existing bounded C JSON precedent | **Shape-specific only** | [`bcir_channel.c`](../runtime/c/bcir_channel.c) bounds bytes and depth and decodes `channel.json`; it is not a JER engine or generated ASN.1 parser |
| MLIR family/profile and R24 (J4 part 1) | **Landed on the law rail** | `BCIR_Asn1Rules` names every transfer syntax the repository speaks; family, canonicality and PER alignment are **derived** from it rather than stored beside it, so no two attributes can disagree about one syntax. R24's emission law generalizes from "emits DER" to "emits a syntax whose octets are a function of the abstract value", which closed two holes the X.690-shaped test had left open. `bcir.asn1.transcode` and `strict_canonical` are added. See §5.3 for why this is one enum and not the (family, profile) pair originally planned |
| JER-to-claims lowering (J4 part 3) | **Landed for the manifest surface** | [`manifest.py`](../bcir/asn1/manifest.py) gives `channel.json`, `DeviceManifest` and the §6.2 selection envelope ASN.1 schemas, and §5.4's two sinks commute over all nine built-in channels. **Not** claimed: GEM graphs and BCAB have no direct JER builder, and the value graph still exists behind the walk because `json.loads` builds it — removing that materialization is J6 streaming work |
| Native calibration and certificates | **Missing** | Python timing cannot stand in for target counters, frozen cost tables, confidence intervals, or an RCSP/K_BCIR verdict |
| Driver/kernel use | **Missing** | No resident driver, Linux module, native-kernel parser, or physical-device comparison uses JER |

The current PersonnelRecord experiment proves a useful but narrow point. Exact encoded
sizes span 84 bytes for canonical unaligned PER through 385 bytes for BCIR-canonical JER,
and a bandwidth objective selects unaligned PER without user-defined ECN. Decode timings
are Python implementation measurements: `json.loads` is native code while COER decode is
currently Python, so those timings cannot establish a target-independent ordering.

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

Still open in this phase: the **native microbench protocol** that would produce a genuinely
`measured` table, and RCSP integration. `build_table` therefore defaults to
`provenance="oracle"`, so producing a `measured` table is a deliberate argument rather than
something that happens by omission.

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
| **J5 — hosted SIMD rail** | Optional C++17 structural/UTF-8 scanner behind the C ABI with scalar fallback | Same accepted/rejected corpus and trace; statistically significant measured advantage on at least two hosts; no unsupported-CPU fault |
| **J6 — certified K_BCIR choice** | **Landed on the Python oracle** ([`certified.py`](../bcir/asn1/certified.py)): distribution-free prediction intervals from order statistics, a frozen generation-tagged cost table with declared provenance, §6.2's certificate, and a production select that **refuses** an oracle table for any timing objective. The native microbench protocol and RCSP integration remain open | Exact sizes decide wire-size objectives with no timing consulted; repeatability is a refusal rather than an average; legality-first and canonical-or-excluded precede every comparison; deterministic selection on two tables, each certificate bound to the table digest it read — [`test_asn1_certified.py`](../bcir/tests/test_asn1_certified.py) |
| **J7 — driver experiment** | Userspace/simulator driver specification ingest, generated views, and sequential BCIR-Linux module comparison | D0–D3 driver gates, signed modules, direct/Linux trace parity, teardown/restart tests, and controlled performance evidence |

User-defined ECN classes are closed after the J0 sign-off. The built-in sets and ordinary
BCIR lowering contracts are sufficient for the measured fixed-candidate result. Reopening
ECN requires a written workload, a missing expressiveness proof, and approval separate
from the JER implementation.

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
| ECN scope returns without evidence | Reopen only with an approved workload and missing-expressiveness proof |
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
