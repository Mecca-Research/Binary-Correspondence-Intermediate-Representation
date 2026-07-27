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
| JER input rejection | **Partial** | Duplicate keys and non-JSON `NaN`/`Infinity` tokens are refused, but decode currently converts the complete input to text and materializes a Python object graph through `json.loads` |
| Deterministic JER emission | **Landed as a BCIR profile** | `JerRules.CANONICAL` fixes BCIR choices; X.697 defines no standard canonical JER variant, so this profile has no standards OID |
| X.680/X.681/X.682/X.683 schema rail | **Landed within the documented subset** | Front-end, information objects, constraints, open-type table resolution, and parameterization feed the shared schema model |
| Other encoding candidates | **Landed on their stated rails** | DER/BER, canonical PER aligned/unaligned, COER/OER, and CXER/XER are recorded in the ASN.1 build-out roadmap; their C coverage differs by format |
| ECN | **Part 1 landed** | [`ecn.py`](../bcir/asn1/ecn.py) models classes, objects, object sets, EDM/ELM, and built-in BER/PER sets; it does not parse or lower user-defined encoding classes |
| Encoding selection harness | **Landed as measurement evidence** | [`selection.py`](../bcir/asn1/selection.py) measures exact wire size and Python-oracle timing for a fixed candidate set; it is not a native K_BCIR table or selection certificate |
| Existing bounded C JSON precedent | **Shape-specific only** | [`bcir_channel.c`](../runtime/c/bcir_channel.c) bounds bytes and depth and decodes `channel.json`; it is not a JER engine or generated ASN.1 parser |
| C or C++ JER twin | **Missing** | No `bcir_jer` runtime component, schema descriptor, differential rail, or JER fuzz target exists |
| MLIR JER family/profile | **Missing** | `BCIR_Asn1Rules` names only BER/CER/DER and current `bcir.asn1.*` operations and R24 checks are X.690-oriented |
| JER-to-claims lowering | **Missing** | No direct builder produces BCIR resources, claims, GEM graphs, StreamPack, or BCAB from JER |
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

The existing `#bcir.asn1_rules<ber|cer|der>` attribute remains the X.690 compatibility
surface. It is not extended by assigning JER an arbitrary enum value.

An additive encoding-family/profile attribute will identify at least:

```text
#bcir.asn1_encoding<family = jer, profile = basic>
#bcir.asn1_encoding<family = jer, profile = bcir_canonical_v1>
```

Legacy X.690 operations remain source-compatible. New or generalized ASN.1 encode/decode
operations accept exactly one legacy rules attribute or one encoding profile. R24 keeps
its number and expands from X.690-only emission checks to family/profile consistency,
JER instruction legality, schema constraints, descriptor identity, and additive
projection rules. Python/MLIR negative-fixture parity is required before promotion.

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
| **J1 — bounded Python oracle** | Pre-parse limits, exact canonical-byte validation, schema/path diagnostics, framed input, and `bcir-asn1c` JER encode/decode/transcode modes | Existing X.697 corpus plus hostile size/depth/Unicode/number/duplicate/truncation cases; no mutation on failure |
| **J2 — schema-plan compiler** | Deterministic descriptor, bound derivation, instruction compilation, version/hash contract, and first `channel.json` schema | Byte-identical descriptor regeneration; unsupported schema/instruction refusal; Python table-driven/direct traces agree |
| **J3 — scalar C twin** | Allocation-free bounded scanner/parser, generated wrappers, event sink, diagnostics, and fuzz target | Python/C value, trace, error-class, and final-offset parity at `-O0`/`-O3`; strict warnings, sanitizers, allocator-independence, and bounded fuzz green |
| **J4 — law and execution lowering** | Additive MLIR family/profile representation, expanded R24, typed/direct claim builders, StreamPack lowering, and `DeviceManifest`/selection schemas | Positive/negative Python↔MLIR parity; direct/typed claim graphs and StreamPack bytes agree |
| **J5 — hosted SIMD rail** | Optional C++17 structural/UTF-8 scanner behind the C ABI with scalar fallback | Same accepted/rejected corpus and trace; statistically significant measured advantage on at least two hosts; no unsupported-CPU fault |
| **J6 — certified K_BCIR choice** | Native microbench protocol, frozen target tables, prediction intervals, RCSP integration, and selection certificate | Exact candidate sizes, controlled counters, repeatability, legality-first refusal, and deterministic selection on at least two targets |
| **J7 — driver experiment** | Userspace/simulator driver specification ingest, generated views, and sequential BCIR-Linux module comparison | D0–D3 driver gates, signed modules, direct/Linux trace parity, teardown/restart tests, and controlled performance evidence |

User-defined ECN classes are closed after the J0 sign-off. The built-in sets and ordinary
BCIR lowering contracts are sufficient for the measured fixed-candidate result. Reopening
ECN requires a written workload, a missing expressiveness proof, and approval separate
from the JER implementation.

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
- Li et al., *Mison: A Fast JSON Parser for Data Analytics*:
  <https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/mison-vldb17.pdf>
- Linux kernel documentation: driver binding, module signing, livepatch consistency,
  and the intentionally unstable internal driver interface, linked in §9.
