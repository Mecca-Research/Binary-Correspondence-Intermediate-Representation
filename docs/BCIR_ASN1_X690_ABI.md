# BCIR ↔ ASN.1 / X.690 binary format compatibility (normative)

BCIR speaks **ASN.1** as an interoperability rail. This document is the normative
contract: which encoding rules BCIR emits and accepts, how the BCIR ABI projects into
an ASN.1 module, and which laws hold across the boundary.

Standards implemented:

- **Rec. ITU-T X.690 (02/2021) | ISO/IEC 8825-1:2021** — BER, CER and DER.
  Erratum 1 (09/2021) redraws Figure 4 and changes no normative rule.
- **Rec. ITU-T X.680 (02/2021) | ISO/IEC 8824-1:2021** — the tag assignments,
  type notation, and the SET component ordering DER inherits (§8.6).

Rails: `bcir/asn1/` is the executable reference; `runtime/c/bcir_asn1.{h,c}` is the
freestanding C twin. Both must agree, and `bcir/tests/test_c_asn1.py` gates it.

What is built here is X.690 only. The rest of the suite — PER, OER, JER, ECN, the
X.680 front-end, constraints and information objects — is scoped in
[`BCIR_ASN1_BUILDOUT_ROADMAP.md`](BCIR_ASN1_BUILDOUT_ROADMAP.md).

## 1. The stance: DER out, BER in

**BCIR emits DER and only DER. BCIR accepts the full BER surface on input.**

This is not a preference. BCIR digests, replays, and byte-compares its artifacts — a
StreamPack has a provenance digest, a plan is replayed from its manifest, and the
parity gates compare octets. BER's sender's options make a value's encoding
non-unique: the same abstract value can be spelled with an indefinite length, a
constructed string, a non-minimal length, or `TRUE` as any non-zero octet. An
artifact whose octets a *peer* gets to choose cannot carry a meaningful digest.

DER (X.690 clause 10 + clause 11) removes every one of those choices, so the
projection of a given StreamPack is a single, reproducible byte string.

**CER is deliberately not implemented.** §9.1 makes the indefinite length form
*mandatory* for constructed encodings, which is irreconcilable with a frozen,
digested artifact. A CER encoding is accepted on input like any other BER encoding;
it is never emitted.

Accepting BER on input is the interoperability half: a peer built against a BER
toolkit remains able to talk to BCIR. `bcir.asn1.reencode_as_der` is the conversion
point — it takes a peer's BER and returns the canonical octets, which is where a
foreign artifact becomes something BCIR can store and digest.

| Direction | Rules | Entry point |
|---|---|---|
| emit | DER only | `encode_der`, `Module.encode`, `streampack.encode_pack` |
| accept, trusted store | DER only (`Strictness.DER`) | `decode_der` |
| accept, foreign peer | full BER (`Strictness.BER`) | `decode_value` |
| normalize | BER → DER | `reencode_as_der` |

## 2. Coverage

The whole of X.690 clause 8 is implemented: BOOLEAN, INTEGER, ENUMERATED, REAL
(binary bases 2/8/16, ISO 6093 decimal NR1/NR2/NR3, and the four SpecialRealValues),
BIT STRING, OCTET STRING, NULL, OBJECT IDENTIFIER, RELATIVE-OID, OID-IRI,
RELATIVE-OID-IRI, SEQUENCE, SEQUENCE OF, SET, SET OF, CHOICE, prefixed/tagged types
(§8.14 implicit and explicit), open types, the restricted character strings, the
unrestricted character string, the useful types, and the TIME family
(TIME/DATE/TIME-OF-DAY/DATE-TIME/DURATION).

Clause 10 and clause 11 are enforced. The two rules that need a *type definition* —
§8.9.3 OPTIONAL/DEFAULT presence and §11.5 DEFAULT-value omission — live in the
schema layer (`bcir/asn1/schema.py`), because a schema-free walk cannot see a
component's DEFAULT. Everything else is checked structurally by `der_violations`,
on both rails.

Out of scope, and not claimed: X.681 information object classes, X.682 constraints,
X.683 parameterization, X.691 PER, X.692 ECN, X.693 XER. A type the schema layer
cannot express is a type the projection does not use.

## 3. The BCIR-StreamPack module

The native StreamPack wire format ([`BCIR_STREAMPACK_ABI.md`](kernel/BCIR_STREAMPACK_ABI.md))
is **frozen and unchanged**. The ASN.1 module is an *additional transfer syntax for
the same abstract value*, not a replacement.

The projection is of the abstract StreamPack, not of its octets. Wrapping the native
bytes in an OCTET STRING would have been trivial and useless — a peer could not read
a field without implementing BCIR's format. Every field is named, so an ASN.1 peer
can consume a BCIR plan with nothing but the module below.

The text below is not a transcription: it is
[`bcir/asn1/BCIR-StreamPack.asn1`](../bcir/asn1/BCIR-StreamPack.asn1) verbatim, the
file the X.680 front-end compiles (a test asserts the two are identical). The
compiled model produces byte-identical DER to the hand-built one in
`bcir/asn1/streampack.py` for every corpus program, so this module IS the schema
rather than a description of one.

```asn1
BCIR-StreamPack { iso(1) identified-organization(3) dod(6) internet(1)
                  private(4) enterprise(1) 62596 1 }
DEFINITIONS IMPLICIT TAGS ::= BEGIN

  StreamPack ::= SEQUENCE {
      version        [0] INTEGER DEFAULT 1,
      sourcePlan     [1] UTF8String,
      topoGen        [2] INTEGER DEFAULT 1,
      mapGen         [3] INTEGER DEFAULT 0,
      dataGen        [4] INTEGER DEFAULT 0,
      pipelineDepth  [5] INTEGER DEFAULT 1,
      segments       [6] SEQUENCE OF LaneSegment,
      prefetches     [7] SEQUENCE OF Prefetch  DEFAULT {},
      blocks         [8] SEQUENCE OF Block     DEFAULT {},
      traceNotes     [9] SEQUENCE OF TraceNote DEFAULT {} }

  LaneSegment ::= SEQUENCE {
      name           [0] UTF8String,
      claimId        [1] INTEGER,
      phaseId        [2] INTEGER,
      lane           [3] Lane,
      width          [4] INTEGER,
      opcode         [5] UTF8String,
      reads          [6] SEQUENCE OF INTEGER,
      writes         [7] SEQUENCE OF INTEGER,
      prefetch       [8] UTF8String OPTIONAL,
      fenceBefore    [9] SEQUENCE OF UTF8String DEFAULT {},
      fenceAfter    [10] SEQUENCE OF UTF8String DEFAULT {},
      dispatch      [11] Dispatch  DEFAULT core,
      channel       [12] UTF8String DEFAULT "host" }

  Lane     ::= ENUMERATED { u(0), ux(1), t(2), ggg(3), a(4), h(5) }
  Dispatch ::= ENUMERATED { core(0), pim(1) }

  Prefetch ::= SEQUENCE {
      name           [0] UTF8String,
      distance       [1] INTEGER,
      targets        [2] SEQUENCE OF INTEGER,
      hint           [3] UTF8String DEFAULT "T0",
      pattern        [4] UTF8String DEFAULT "linear",
      buffers        [5] INTEGER    DEFAULT 1 }

  Block ::= SEQUENCE {
      base           [0] INTEGER,
      count          [1] INTEGER,
      strides        [2] SEQUENCE OF INTEGER DEFAULT { 1 } }

  TraceNote ::= SEQUENCE {
      claimId        [0] INTEGER,
      srcHash        [1] INTEGER DEFAULT 0,
      traceHash      [2] INTEGER DEFAULT 0 }

END
```

Design notes worth stating because they are the non-obvious choices:

- **Enumerations are ENUMERATED, not INTEGER.** `lane` and `dispatch` are closed sets
  in BCIR; X.680 §20 says exactly that, and it lets a peer's schema reject an unknown
  lane rather than silently accept `250`.
- **DEFAULTs mirror the native format's implicit ones.** The native encoder emits v1
  bytes when no v2/v3 feature is used; the same values are DEFAULTs here, so §11.5
  omits them. This is why the DER projection is *smaller* than the native encoding on
  the corpus — the native format writes every field unconditionally.
- **The module OID is in private-enterprise space.** It needs no registration and can
  never collide with an allocation made by a registration authority.
- **`stride_k` is not projected.** The native segment record reserves it as a constant
  zero and carries stride per-claim, so projecting it would add a field that never
  varies.
- **The projection version is independent of the native StreamPack version.** The
  module can gain a field without the native format moving, and vice versa.

## 4. The laws

Gated in `bcir/tests/test_asn1_streampack.py` and `bcir/tests/test_c_asn1.py`:

| Law | Statement |
|---|---|
| **A1 faithful** | `decode_pack(encode_pack(p)) == p` for every corpus program |
| **A2 canonical** | `encode_pack(p)` is DER, and `encode_pack(decode_pack(encode_pack(p))) == encode_pack(p)` |
| **A3 additive** | `abi.encode(decode_pack(encode_pack(abi.decode(b)))) == b` — native octets survive the ASN.1 round trip |
| **A4 normalizing** | `reencode_as_der` is idempotent, and its output satisfies A2 |
| **A5 dual-rail** | the C and Python decoders agree on the node tree, the BER verdict, and the DER verdict for every input |

A3 is the compatibility claim: a pack can leave BCIR as DER, be handled by an
ASN.1-speaking peer, and come back as the *same native octets*, so StreamPack digests
and provenance manifests stay valid across the boundary.

## 5. The law rail (R24)

The ASN.1 schema is also **IR**: `bcir.asn1.module` / `type` / `component` / `encode` /
`decode` / `projection` in `mlir/include/BCIR/BCIRAsn1Ops.td`, verified by law **R24**
in `-bcir-verify`.

The split between the two rails is deliberate and is about *when* a fault is
detectable. The oracle rejects a bad schema when a value is encoded through it; R24
rejects it when the **type is written**, before any value exists. Those are different
sets of faults, and the ones R24 owns are the ones that are properties of the type:

| Rule | Clause | Why it is static |
|---|---|---|
| module/encode rules must be `der` | X.690 10 + 11 | BER and CER leave the octets to the sender; BCIR digests what it emits |
| module OID well-formed | X.690 8.19.4 | root arc 0–2, second arc 0–39 under arcs 0 and 1 |
| universal tag assigned | X.680 Table 1 | 0, 15 and 37+ are reserved; a conforming sender never emits one |
| primitive ⇔ universal tag | — | a primitive without one is unencodable; a constructor with one names a tag it does not own |
| `sequence_of`/`set_of` ⇔ element | — | an "of" type with no element names nothing |
| **component tags distinct** | X.680 24.4/25.3/29.3 | a type with duplicate tags is undecodable **for every value it could ever hold** |
| OPTIONAL xor DEFAULT | X.680 25.5 | a DEFAULT already makes the component omissible |
| DEFAULT carries its value | X.690 11.5 | the encoder must compare against it to omit it |
| SET tags all present | X.690 8.11.2 | a set is order-free on the wire, so tag is the only discriminator |
| `strict_der` ⇏ accepts BER | — | a direct contradiction |
| projection is `additive` | — | a replacement would invalidate every digest over the native octets |

Vacuous for IR with no `bcir.asn1.*` operation — the non-disturbance invariant R14–R23
also hold to. Negative fixtures: `mlir/test/passes/verify_asn1.mlir` (1 positive, 13
negatives, one per diagnostic). Enum values, the module OID, and the
diagnostic-to-fixture pairing are pinned across the rails by
`bcir/tests/test_asn1_law_parity.py`.

## 6. Trust boundary

Every octet handed to a decoder is untrusted. The contract is **total**: for any
input, every entry point either returns a value or reports a fault — it never reads
outside the buffer and never recurses on attacker-chosen structure.

Decoder policy beyond X.690, applied identically on both rails:

- **Nesting is bounded** (64 levels by default). X.690 sets no limit; an unbounded
  decoder is a stack-exhaustion surface. The C rail uses an explicit stack and never
  recurses.
- **A declared length is bounded before it is believed** (8 length octets). A hostile
  encoding can claim a 2⁶⁴-octet body in eight bytes.
- **A tag number is bounded** to 32 bits.
- **A constructed encoding's declared length is authoritative over its children.** A
  child that overruns its parent is truncated, not accepted against the outer buffer.
- **Iteration-end and a malformed end-of-contents are distinct statuses.** Sharing one
  lets a stray `00 00` read as "no more children" and silently truncate a value.

The last two are parser-differential defences: a decoder that stops early where
another keeps reading is how content gets smuggled past one implementation.
`runtime/c/fuzz_asn1.c` fuzzes the C rail under ASan/UBSan as part of
`tools/c/fuzz_streampack.sh`.

## 7. Transfer syntax identity

X.690 §12 assigns object identifiers to the encoding rules themselves:

| Rules | OID | OID-IRI |
|---|---|---|
| BER | `{joint-iso-itu-t asn1(1) basic-encoding(1)}` = 2.1.1 | `/ASN.1/Basic-Encoding` |
| DER | `{joint-iso-itu-t asn1(1) ber-derived(2) distinguished-encoding(1)}` = 2.1.2.1 | `/ASN.1/BER-Derived/Distinguished-Encoding` |

Exported as `bcir.asn1.BER_OID` / `DER_OID` and their `_IRI` forms, so a protocol that
must name its transfer syntax can do so without hard-coding the arcs.

## 8. Validation

```bash
python -m bcir.tests.run_all --tier quick          # includes the four ASN.1 modules
FUZZ_RUNS=500000 bash tools/c/fuzz_streampack.sh   # includes the X.690 harness
bash tools/wsl/check_passes.sh                     # includes the R24 fixture
```

Evidence recorded at the time of writing: X.690's own worked examples reproduce
byte-identically (§8.9 SEQUENCE, §8.14 tagged types, §8.19.4 `{2 999 3}`, §8.20.5
`{8571 3 2}`, §8.23.5 "Jones" in all three sender's-option forms, §8.6.4.2 BIT STRING
primitive and constructed); the DER projection round-trips all 12 corpus programs; and
the Python↔C differential agreed on 12 000 mutants across 3 seeds.
