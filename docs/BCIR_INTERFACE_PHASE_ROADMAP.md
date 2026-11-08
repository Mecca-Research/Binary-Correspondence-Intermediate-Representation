# BCIR interface phase: the ASN.1 compiler backend, peer notations and formats, cryptographic standards and networking

> **Status: design (2026-11-08); no IF slice has landed.** By the owner's direction this phase
> runs **first**, before the foundation phase ([`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md)
> §3.1) and before the driver order resumes. The database, Unicode and laboratory work went ahead
> of the compiler backend; this phase finishes the backend first.
>
> It sequences and sizes work that is partly owned elsewhere:
> - ASN.1 itself: [`BCIR_ASN1_BUILDOUT_ROADMAP.md`](BCIR_ASN1_BUILDOUT_ROADMAP.md), measured
>   against other compilers in [`BCIR_ASN1_COMPILER_COMPARISON.md`](BCIR_ASN1_COMPILER_COMPARISON.md).
> - JER: [`BCIR_ASN1_JSON_ROADMAP.md`](BCIR_ASN1_JSON_ROADMAP.md).
> - Peer notations, foreign formats and cryptographic syntax: the UDSI milestones of
>   [`BCIR_UNIVERSAL_INTERFACE_ROADMAP.md`](BCIR_UNIVERSAL_INTERFACE_ROADMAP.md).
> - Network devices: [`kernel/BCIR_DRIVER_KERNEL_ROADMAP.md`](kernel/BCIR_DRIVER_KERNEL_ROADMAP.md).
>
> This document owns what none of them owns: the phase's order and exit, the qualification
> needed before claiming that ASN.1 beats other encodings, the investigation of a minimal
> compiled JSON format, the cryptographic standards beyond syntax (PKIX, CMS, PKCS, S/MIME,
> TLS) and the networking protocols. It changes no law, no frozen ABI and no landed slice.

## 1. The answer in brief

**What is strong.** Every ASN.1 encoding-rule family is built, with its canonical variant out and
its permissive variant in:
- BER/DER;
- aligned and unaligned PER;
- OER and canonical OER;
- XER and CXER;
- JER, with a scalar C twin and a SIMD structural index.

X.681–X.683 are built, and so is X.692 ECN in all three parts, which none of asn1c, asn1scc and
FFASN1 has. Laws R24/R25 put the schema and ECN on the law rail, and K_BCIR selects among
encodings under a certificate.

**What is missing, against the list the owner named:**

| Gap | State today | Where it goes |
|---|---|---|
| **The compiler backend.** | `bcir.asn1.cgen` compiles a module to straight-line C, but it is study-scale: no plan, StreamPack path or R-law consumes it. It emits only three rules (canonical OER, unaligned PER and the ITS study's binary rule). Of the primitive types it translates only INTEGER, ENUMERATED, BOOLEAN, NULL and OCTET STRING, alongside SEQUENCE, SEQUENCE OF and CHOICE. It has no OBJECT IDENTIFIER or BIT STRING, which every certificate and CMS message uses. At generation time it refuses `DEFAULT`, extension additions, `SET` and `SET OF`, every character string type and unaligned-PER fragmentation. It has no DER/BER or aligned-PER backend. | IF-4 to IF-6 |
| **The front end.** | Three X.680 constructs are refused: `COMPONENTS OF`, selection types, and `WITH SUCCESSORS`/`WITH DESCENDANTS`. The real-world grammar corpus is two PKIX modules, against asn1c's 165 test grammars plus eleven protocol modules and asn1scc's 803 grammars. None of the comparison's seven ranked imports has landed: the corpus, the refusals, random values, a regression corpus, a bit-layout report, a CLI and a checksum extension. | IF-1 to IF-3, IF-7 |
| **CSN.1 and ABNF.** | Planned as UDSI-6; not built. | IF-12, IF-13 |
| **SDXF, XDR, XML, Protocol Buffers and peers.** | Planned as UDSI-2, -3 and -9; none built. XER covers XML only as an ASN.1 encoding; there is no general XML 1.0 reader or writer. | IF-11, IF-14 to IF-18 |
| **The "smaller than binary" result.** | The ITS study ([`research/BCIR_ITS_ASN1_BINARY_STUDY.md`](research/BCIR_ITS_ASN1_BINARY_STUDY.md)) found canonical OER on a tuned schema smaller than the binary rule on every envelope, and faster at 0.55–0.74 of its time. The comparison is against BCIR's own reconstruction of that binary rule, built by the same generator, on one envelope family. The October assessment ([`research/BCIR_GEMPLUS_ASN1_ML_PERFORMANCE_2026-10-10.md`](research/BCIR_GEMPLUS_ASN1_ML_PERFORMANCE_2026-10-10.md) §2) concluded that the result does not establish that ASN.1 generally beats a good custom codec. | §4; IF-8, IF-9 |
| **A minimal compiled JSON format.** | Open question. | §5; IF-10 |
| **Cryptographic standards.** | SHA-256, HMAC-SHA256 (RFC 4231 vectors), Ed25519 (RFC 8032, for signed plans) and CRC32 exist. There is no AEAD, key exchange, ECDSA, RSA or post-quantum primitive. CMS, PKCS, S/MIME and TLS have no support, and X.509 has no path validation. | IF-19 to IF-23 |
| **Networking.** | No socket, packet, DNS, HTTP or TLS code exists. virtio-net is in the driver roadmap's evidence sequence (§5.2, row 4) but absent from the build order (§5.6). | IF-24, IF-25; virtio-net in driver row 3 |

**The phase's exit is twenty-five slices (§3), in six stages:**
- A: the ASN.1 compiler, front to back;
- B: the qualification that decides what may be claimed;
- C: the JSON investigation;
- D: notations and formats;
- E: cryptographic standards;
- F: networking over hosted transports.

Native networking (virtio-net and a native stack) needs DMA and the virtqueue proof, so it joins
the driver order rather than this phase (§6).

## 2. Why this order, and how it meets Unicode

**No IF slice needs the database or the laboratory.** Some need Unicode, and Unicode has not
started (UC-1 is still open). Those slices go **ASCII-first, with declared refusals**, so this
phase does not wait on the foundation phase:

| Slice | What it needs | Before the Unicode slice lands | After |
|---|---|---|---|
| IF-20 PKIX names | RFC 5280 §7.1 compares internationalized names after RFC 4518 string preparation: NFKC, case folding and prohibited characters | Non-ASCII directory strings compare by exact octets only. Name constraints that need preparation are refused, by name. | UC-4's normalization and case folding |
| IF-20, IF-24, IF-25 domain names | IDNA2008 (RFC 5891) and UTS #46 | A-labels (`xn--…`) pass as ASCII; U-labels are refused | UC-6's UTS #46 and Punycode |
| IF-17 XML names | XML 1.0 (Fifth Edition) §2.3 defines `NameStartChar` and `NameChar` by explicit code-point ranges | Needs no Unicode database: the ranges are the specification's own | — |
| IF-12, IF-22 headers | RFC 6532 UTF-8 headers in MIME | UTF-8 validation by the one UTF-8 rail; no normalization | — |

One consequence for the Unicode roadmap: UC-6's UTS #46 and Punycode algorithm is needed by
IR-side protocol code. It therefore lives IR-side beside UC-4's UAX #15 implementation, for the
same reason UC-4's does. Only its catalog columns stay corpus-side. The Unicode roadmap's UC-6
carries this note.

## 3. The slices

Each is one PR, with the slice id in the title and RED before GREEN. Each new check gets a fault
entry, and each PR body records the exact gate output. Where a row names a UDSI milestone, that
roadmap's design rules and SI tests apply unchanged.

### Stage A — the ASN.1 compiler, front to back

| Slice | What | Gate |
|---|---|---|
| **IF-1** | **The real-world grammar corpus.** Public protocol modules become front-end conformance input:<br>• the IETF's 2002/2008-syntax PKIX and CMS modules (RFC 5911, RFC 5912);<br>• the PKCS modules;<br>• LDAP;<br>• SNMP;<br>• the 3GPP control-plane modules (RRC, NGAP, S1AP);<br>• IEEE 1609.2 and ETSI ITS;<br>• asn1c's and asn1scc's test grammars.<br>Each source is reviewed for its license first. A module whose redistribution is unclear is fetched at test time by pinned URL and SHA-256 instead of vendored, the way UC-1 fetches the UCD. | Every module either compiles to a stable type-inventory digest or is refused naming the construct. The refusal census is recorded and becomes IF-2's list. *RED:* an unpinned or digest-mismatched module is refused. |
| **IF-2** | **Close the front-end refusals.** `COMPONENTS OF`, selection types and `WITH SUCCESSORS`/`WITH DESCENDANTS`, plus every construct IF-1's census finds in more than one module. Each lands on the oracle, the schema IR (R24) and the parity gates together. | Every IF-1 module that failed only on these now compiles. Each construct's expansion equals the hand-expanded module. *RED:* inline `COMPONENTS OF` without its extension-marker rules. |
| **IF-3** | **Schema-driven random values.** `random_value(type, seed)` produces only values the schema admits, covering every constructed and constrained type. It feeds an encode → decode → compare differential across the oracle, the C twins and cgen, and seeds the fuzz harnesses. | Every generated value is admitted by the constraint checker; every rule round-trips it on every rail. *RED:* a generator that ignores a size constraint is caught by the checker. |
| **IF-4** | **The backend's constructs.** cgen gains:<br>• `DEFAULT`;<br>• extension additions, with the canonical rules for each encoding;<br>• `SET`;<br>• the restricted character string types (UTF8String validated by the one UTF-8 rail);<br>• `OBJECT IDENTIFIER` and `RELATIVE-OID`, `BIT STRING`, `SET OF`, `REAL` and the time types (`UTCTime`, `GeneralizedTime` and X.680's `TIME` family where IF-1's modules use them);<br>• open types resolved through table constraints;<br>• unaligned-PER fragmentation. | Octet parity with the oracle over IF-3's values for every construct, and every canonicality refusal the oracle makes; the ASan/UBSan differential and a libFuzzer campaign per new construct. *RED:* a decoder that accepts a non-minimal length. |
| **IF-5** | **The backend's rules.** DER out and BER in; aligned PER (the ITS study's next step: octet-aligned key and signature runs). The October assessment's decode work also lands here:<br>• grouped fixed-width integer decode;<br>• decode into a typed sink;<br>• bounds checks hoisted where schema-derived bounds prove every access safe. | Parity as IF-4. The hoisted checks are refused wherever the proof is missing; truncation, padding and range refusals are unchanged. |
| **IF-6** | **The backend in the artifact path.** A generated codec becomes a candidate K_BCIR's encoding selection (phase H) can choose. Its source and object digests are bound into the selection certificate. The plan interpreters (`bcir_oer.c`, `bcir_per_plan.c`, `bcir_emit.c`) stay the reference it is differentialed against. | For every corpus module and rule, generated and interpreted codecs agree byte for byte. A certificate naming a generated codec replays. |
| **IF-7** | **Tooling:**<br>• `tools/asn1` with `transcode`, `validate` and `layout`;<br>• the bit-layout report, emitted as part of the selection certificate (asn1scc's interface control document in intent);<br>• the security-regression corpus `runtime/c/security-regression/<case>/`, back-filled with the four defects the earlier sweep fixed;<br>• ACN's checksum construct (`post-encoding-function` with `save-position`) as a BCIR ECN extension in its own namespace, outside the X.692 conformance claim and refused unless enabled. | The CLI transcodes every built rule pair over the corpus. Each regression case fails on its pre-fix code. The ECN checker refuses the extension by default. |

### Stage B — qualification: what ASN.1 may claim

| Slice | What | Gate |
|---|---|---|
| **IF-8** | **The comparator harness** (§4). It builds independent implementations from pinned revisions in a CI job: asn1c and asn1scc, Protocol Buffers, FlatBuffers, Cap'n Proto, a CBOR library, msgpack-c, XDR (libtirpc), and simdjson and yyjson for JSON. It measures them against BCIR on the same abstract values. Comparators are built and run beside BCIR, never linked into it. | Every comparator's output decodes to the same abstract value BCIR's does, or the row is refused. An A/A control runs beside every ratio. |
| **IF-9** | **The workloads and the claim** (§4.2). Five workloads are measured as whole processes on x86-64 and aarch64 runners. The results table records a verdict per (workload, comparator, metric, host class). | A claim appears in prose only where the claim rule of §4.3 holds. Everything else is recorded as measured, with its verdict. |

### Stage C — the JSON investigation

| Slice | What | Gate |
|---|---|---|
| **IF-10** | **A minimal compiled JSON format: the investigation** (§5). The four hypotheses are measured against CBOR, MessagePack, Protocol Buffers and JSON text (simdjson) on three public corpora of different shape: a social-media API dump, a ticketing catalog and a GeoJSON file of mostly numbers (`twitter.json`, `citm_catalog.json` and `canada.json` from the JSON parser benchmark sets, license reviewed first). The result is a written GO or STOP. | GO only under §5's rule. STOP records the numbers and why, and changes nothing. |

### Stage D — notations and formats

| Slice | What | UDSI | Gate |
|---|---|---|---|
| **IF-11** | The contract registry and the native BCIR JSON contract/value frontend, with its ASN.1/JER bridge: the base every provider maps through. | UDSI-0, UDSI-1 | SI-01/02/04/05: a contract loaded without the ASN.1 frontend, then encoded by two providers |
| **IF-12** | **ABNF** (RFC 5234 with RFC 7405) compiled to bounded incremental parsers, with source positions and exact octet consumption. Semantic binding to typed fields is separate. First grammars: URI (RFC 3986), HTTP/1.1 messages (RFC 9112), Internet Message Format (RFC 5322) and MIME (RFC 2045). | UDSI-6 | SI-08/11: independent accept/reject agreement; split and truncated input; incomplete distinguished from invalid; never silently reread as ordered PEG choice |
| **IF-13** | **CSN.1** (3GPP TS 24.007 Annex B), a separate bit-layout frontend. First messages: GPRS RLC/MAC (3GPP TS 44.060). | UDSI-6 | SI-11: bit-exact fixtures derived from the specifications. Equivalence to ECN is claimed only where proved; CSN.1 is never labeled an ASN.1 encoding. |
| **IF-14** | **XDR** (RFC 4506), a bounded control-message profile. | UDSI-2 | Interoperability with libtirpc both ways; golden bytes; refusal of every unsupported type |
| **IF-15** | **CBOR** (RFC 8949) with **CDDL** (RFC 8610). | UDSI-3 | Tag, map-key, duplicate and numeric policies declared; RFC 8949 Appendix A's examples; a deterministic profile where one is selected |
| **IF-16** | **Protocol Buffers**: proto3 and a declared editions subset, encode and decode. | UDSI-3 | The official conformance test suite passes over the declared subset; presence, unknown-field, enum and `oneof` corpus (SI-03) |
| **IF-17** | **XML 1.0** (Fifth Edition) with Namespaces in XML 1.0, as a general bounded reader and writer beside XER. External entities are refused, as is any entity expansion beyond the five predefined entities and character references. XSD import through X.694 is a separate, later decision. | — | The W3C XML Conformance Test Suite over the declared subset; billion-laughs and XXE inputs refused; byte and depth bounds |
| **IF-18** | **SDXF** (RFC 3072) and **MessagePack**, both small, pulled forward from UDSI-9 at the owner's request. | UDSI-9 | RFC 3072's examples; MessagePack's specification examples; extension-type registry per profile |

Deferred out of the exit, each still a UDSI row: FlatBuffers and the Arrow C Data bridge
(UDSI-5), EXI, Avro, Thrift, Parquet and OGDL Binary (UDSI-8/9), and X.694.

### Stage E — cryptographic standards

| Slice | What | Gate |
|---|---|---|
| **IF-19** | **The provider contract and the primitives.** An algorithm and key policy, with the provider's identity recorded in every certificate that uses it. Two providers:<br>• OpenSSL as the hosted reference provider;<br>• a formally verified C provider (HACL* or libcrux, both Apache-2.0) for freestanding and production use.<br>Primitives: SHA-2 and SHA-3; HMAC and HKDF (RFC 5869); AES-GCM and ChaCha20-Poly1305 (RFC 8439); X25519 (RFC 7748); ECDSA P-256/P-384 and RSA-PSS/PKCS #1 v1.5 (FIPS 186-5, RFC 8017); ML-KEM and ML-DSA (FIPS 203, FIPS 204). BCIR's own SHA-256, HMAC-SHA256 and Ed25519 stay, held to both providers. No new in-tree primitive unless no verified provider covers the target. | Wycheproof and NIST CAVP vectors on every provider. The in-tree units equal both providers on those vectors. A constant-time check (a dudect-style statistical test, plus Valgrind secret-taint runs) on every in-tree secret-dependent path. *RED:* a provider that accepts a Wycheproof invalid case. |
| **IF-20** | **X.509 and PKIX** (RFC 5280): certificates, CRLs and the standard extensions, parsed by the compiled backend; path validation per §6, names ASCII-first (§2). | x509-limbo and NIST PKITS over the declared subset; every expected rejection rejected; DER strictness (a BER-only spelling is refused in a certificate) |
| **IF-21** | **CMS** (RFC 5652) SignedData, verify and sign, with the signed-attributes DER rule; then EnvelopedData and AuthEnvelopedData. **PKCS #8 / RFC 5958** key packages, **PKCS #1** encodings and **PKCS #10** (RFC 2986) requests. PKCS #12 stays workload-gated. This is UDSI-7. | SI-09: provider-backed positive and negative fixtures. Content, attribute and parameter changes are detected; wrong keys and unsupported algorithms refused. Syntax-only results and verified results are reported separately. |
| **IF-22** | **S/MIME 4.0** (RFC 8551) over MIME (IF-12's grammars) and CMS. | RFC 9216's example keys and certificates and RFC 4134's example messages; signed and enveloped round trips against OpenSSL's `cms` command |
| **IF-23** | **TLS 1.3** (RFC 8446) as a sans-I/O engine: record layer, handshake codecs, key schedule, state machine. It is research-grade until an external review. Until then the production path is the hosted provider's TLS. | RFC 8448's example handshakes reproduced byte for byte; interoperability with OpenSSL's `s_server` and `s_client` over loopback; malformed-record and downgrade refusals |

### Stage F — networking over hosted transports

| Slice | What | Gate |
|---|---|---|
| **IF-24** | **Packet and name codecs**, sans-I/O, as ECN modules: Ethernet, IPv4 (RFC 791), IPv6 (RFC 8200), UDP (RFC 768), TCP (RFC 9293) and ICMP headers, with the Internet checksum (RFC 1071) through IF-7's checksum extension; DNS messages (RFC 1035) with name compression, bounded. Names are ASCII-first (§2). | Bit-exact against the RFCs' layouts and independent captures; checksum vectors; compression loops and over-long names refused |
| **IF-25** | **Hosted transports**: a POSIX-socket adapter for UDP and TCP, and io_uring for batched submission (the IPC program's rung-I3 comparator). HTTP/1.1 (RFC 9110, RFC 9112) client and server on IF-12's grammar, with TLS through IF-19's provider and IF-23's engine. | Loopback interoperability with curl and OpenSSL; split reads and writes, timeouts and peer resets; every bound enforced at entry |

## 4. Qualification: what it takes to claim ASN.1 is faster or smaller

### 4.1 Where the evidence stands

The ITS result is real and reproducible: smaller in every profile, 0.55–0.74 of the binary
rule's time under three compilers, no heap. Its scope is narrow in three ways:
- **The control is BCIR's own.** It is a reconstruction of the 2015 binary rule, emitted by the
  same generator, not an independently built and deployed codec.
- **The workload is one envelope family** with a schema tuned for it.
- **It times the codec, not a process.** No signature check, transport or consumer conversion
  is included, and the October assessment found that a different workload (fixed-size token
  arrays) reaches a different ranking.

By the GEM+ measurement discipline these are `ratio` rows on one host class. They support "on
this workload, against this control". They do not support a general superiority claim.

### 4.2 The five workloads (IF-9)

| Workload | BCIR side | Comparators | What decides it |
|---|---|---|---|
| **General serialization**: four public datasets of a published Rust serialization benchmark (log, mesh, save data, game state), reviewed for license before use, their schemas written in ASN.1 | the IF-5 generated codecs, best rule per dataset | Protocol Buffers, FlatBuffers, Cap'n Proto, CBOR, MessagePack | size; encode; decode plus full consumption; selected-field access measured as a separate task |
| **Certificates**: a public certificate corpus parsed and path-validated | IF-20 on the generated DER backend | OpenSSL's X.509 parser and verifier | certificates per second, parse and validate, with identical verdicts |
| **Telecom control plane**: 3GPP RRC and NGAP messages, from IF-3's generator and from licensable public captures | IF-5's aligned and unaligned PER | asn1c's generated decoders | decode plus dispatch to a typed handler |
| **ITS secured messages, with signatures**: the study's envelopes plus ECDSA P-256 verification | canonical OER from IF-5 plus IF-19's provider | the study's binary control and an independent 1609.2 encoder where one is available | messages per second, verified; crypto is expected to dominate, and the claim must say so |
| **JSON documents**: public JSON corpora | schema-bound JER (J-series) and IF-10's candidates | simdjson, yyjson | bytes in to typed values out |

### 4.3 The claim rule

A sentence of the form "BCIR's ASN.1 is faster or smaller than X" may appear outside a
measurement record only when all four of these hold:
1. **Same work.** The comparator is independent and pinned, and both sides do the same work:
   the same abstract values, validation and consumer.
2. **Resolved.** The paired ratio's interval excludes 1 on x86-64 and on aarch64, with the A/A
   control inside its band.
3. **Whole process.** The workload is timed as a whole process (§4.2's last column), not only as
   a codec call.
4. **Scoped.** The sentence names the workload, the comparator and the metric.

"Superior" in general is claimed only if one encoding wins three of the five workloads on size
and on time, and the losses are stated beside the wins.

Expected losses, written down in advance so they are not explained away afterwards:
- FlatBuffers and Cap'n Proto win selected-field access, because they do not decode.
- Unaligned PER loses time to any octet-aligned rule wherever 32-octet keys sit at bit offsets
  (ITS study §4.3).
- On the ITS workload, signature verification dominates.

## 5. The minimal compiled JSON format (IF-10)

**The question.** The ITS result came from moving the schema's knowledge out of the bytes. A JSON
document carries that knowledge in every message: keys, quotes, decimal digits and delimiters.
Can JSON data with a schema be compiled to an encoding that is minimal, fast and still has a
canonical identity? And is that encoding anything other than ASN.1 with an inferred schema?

There is no ITU-T Recommendation that maps JSON Schema into ASN.1 the way X.694 maps W3C XML
Schema. X.697 goes the other way, from ASN.1 to JSON.

| Hypothesis | Construction | Comparators | GO if |
|---|---|---|---|
| **H1. Schemaless** | A JSON value type (`CHOICE` of null, boolean, number, string, array, object) under canonical OER and unaligned PER. | CBOR, MessagePack | Expected **STOP**: without a schema there is little to remove. It is measured so the expectation is a number. |
| **H2. Schema-bound** | JSON Schema, JSON Type Definition or a BCIR JSON contract mapped to ASN.1 with the tightest constraints the schema states, then encoded by IF-5's codecs. | Protocol Buffers; JSON text + zstd; CBOR | It is smaller than every comparator on two of three public corpora, and decode plus consumption is no slower than Protocol Buffers on either ISA. |
| **H3. Inferred constraints** | Ranges, sizes and enumerations inferred from a corpus. Every inferred constraint goes behind an **extension marker**, so a value outside it still encodes: X.691's extensibility is the sound fallback. The inferer only proposes; the checker decides. | H2's result | It improves on H2 by a stated margin with zero lossy cases over held-out data. |
| **H4. Canonical identity** | The canonical binary form's digest as the document's content address. | JCS (RFC 8785) | Two documents with equal JSON values get one digest, and unequal values never share one, over a generated corpus with an adversarial key-order and number-spelling set. |

A GO yields a **named BCIR profile**: versioned, with no standards OID, and its own identity
beside JER rather than a replacement for it. Its first users are IF-11's contract registry and
AIS's control records (`BCIR_AI_SYSTEM_ROADMAP.md` §5.2). A STOP records the numbers in the
measurement tree and closes the question until a workload reopens it.

## 6. Native networking, in the driver order

Native networking needs a network device driver, DMA leases and a stack that owns its buffers.
None of that belongs to this phase:
- **virtio-net** joins the driver roadmap's §5.6 row 3, after virtio-blk, on the same virtqueue
  proof. Its first gate is QEMU packet loopback with backpressure and multi-queue.
- **A native user-space network stack** follows over virtio-net with the DMA ownership engine's
  leases. Its codecs are IF-24's, and its socket-free API is IF-25's sans-I/O engines.
- **QUIC and HTTP/2 and /3** wait for a workload.

IF-24 and IF-25 make every protocol above the device run and interoperate on hosted transports
first. That is the platform-neutral rule (master roadmap invariant 5): the direct contract first,
then adapters around it.

## 7. Order, relative to everything else

1. **Now: this phase**, IF-1 to IF-25 in order. Stages A and B come first because they decide
   what the rest measures against. IF-19 and IF-20 can move ahead of stages C and D once IF-6
   lands, if certificates are wanted sooner: path validation needs the generated DER backend and
   a provider to verify signatures.
2. **Then the foundation phase** (master roadmap §3.1), starting at UC-1. UC-4 and UC-6 lift
   §2's ASCII-first refusals when they land.
3. **Then the driver order** (driver roadmap §5.6), alternating one slice at a time with the
   deferred work:
   - the deferred UDSI rows;
   - PKCS #12;
   - QUIC and HTTP/2 and /3;
   - UC-9 to UC-11 and SLM-8 to SLM-13;
   - the AI system's AIS-1 to AIS-9.

   virtio-net and the native stack come with driver row 3 and DMA.

## 8. What this roadmap does not claim

- **No superiority is claimed today.** §4.3 says when one may be.
- **No in-tree cryptographic implementation is called secure** without an external review. The
  verified and reference providers carry the production path.
- **No format or protocol is "supported" beyond its declared subset** and its conformance suite.
- **No corpus, test suite or comparator enters the tree without its license reviewed.** Where
  redistribution is unclear, the test fetches by pinned digest. Nothing from material under
  confidentiality terms enters the tree, a commit or a PR.
