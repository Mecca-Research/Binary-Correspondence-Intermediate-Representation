# BCIR ASN.1 build-out roadmap

Where the ASN.1 rail goes after X.690. This is the portfolio document for the ASN.1
program: dependency order, promotion gates, stop conditions, and the one idea that
makes the whole thing more than a second codec.

Companion documents: [`BCIR_ASN1_X690_ABI.md`](BCIR_ASN1_X690_ABI.md) is the normative
contract for what is already built; [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md)
owns cross-program ordering. [`BCIR_ASN1_JSON_ROADMAP.md`](BCIR_ASN1_JSON_ROADMAP.md)
owns the post-X.697 plan for compiling schema-bound JER into verified BCIR artifacts.

## 0. The thesis: encoding rules are a realization choice

BCIR already selects among legal realizations of an abstract value under a
12-dimensional cost vector and live pressure Θ. That is what `K_BCIR` *is*.

An abstract ASN.1 value has legal realizations too — BER, DER, PER-ALIGNED,
PER-UNALIGNED, OER, JER, XER — differing by orders of magnitude in size, encode cost,
and decode cost, with exactly the same abstract semantics. The ASN.1 world calls the
choice "which encoding rules"; BCIR would call it *a candidate set*.

**The correspondence is structural, not a metaphor:**

| ASN.1 (X.692 ECN) | BCIR |
|---|---|
| encoding class (`#INT`, `#SEQUENCE`) | claim opcode / lane |
| encoding object | candidate realization |
| encoding object set applied to a type | a plan π |
| PER-visible constraint | the geometry that prices a candidate |
| `#TRANSFORM` / `#OUTER` | lowering contract |

No ASN.1 toolchain in existence *selects* encoding rules by cost — the rules are a
protocol design decision, fixed at specification time. BCIR has the optimizer, the cost
algebra, the budgets, and the two-truth quarantine already built. **Cost-governed
encoding selection is the thing BCIR can do here that nothing else can**, and every
phase below is either a prerequisite for it or a consequence of it.

That is also the honest answer to "how far does this go". The end state is not "BCIR
supports the ASN.1 suite". It is: *an abstract value, a target, and a budget go in; a
provably legal wire format comes out, with a certificate saying why.*

## 1. Standards inventory (verified 2026-07-26)

The suite is at its **2021 edition** across the board; no 2024/2025 revision exists.
X.690 carries Erratum 1 (09/2021), which redraws Figure 4 and changes no normative
rule.

| Rec. | ISO/IEC | Title | BCIR status |
|---|---|---|---|
| X.680 | 8824-1:2021 | Basic notation | **supported subset built** — parser/printer/lowering consume the module and type surface used by the current rails; unsupported notation fails closed |
| X.681 | 8824-2:2021 | Information object specification | **built** — classes, WITH SYNTAX, objects, object sets, associated tables (cl. 13). The one exclusion is X.683 parameterization applied to *objects*; §10.1 explains why that is narrower than it reads, now that Annex C's parameterization is built for ECN objects and the front-end has parameterized object/object-set assignments |
| X.682 | 8824-3:2021 | Constraint specification | **built** — table + component relation (cl. 10), user-defined (cl. 9), contents (cl. 11) |
| X.683 | 8824-4:2021 | Parameterization | **built** — parameterized type/object/object-set assignments and references; cross-module tag-default nuance (§9.8) excluded |
| X.690 | 8825-1:2021 | BER / CER / DER | **built** (DER out, BER in; CER by design excluded) |
| X.691 | 8825-2:2021 | PER | **built** (CANONICAL-PER out, BASIC-PER in; both variants; validated against Annex A.1–A.4) |
| X.692 | 8825-3:2021 | ECN | **parts 1, 2 and 3 built** — class/object/object-set model (cl. 9-18), EDM/ELM, the seven built-in BER/PER object sets; and [`ecn_user.py`](../bcir/asn1/ecn_user.py) for the user-defined half (cl. 19-25): bit-level encoding spaces, justification, `#PAD`, stated transmission order, `INT-TO-INT`/`INT-TO-BITS` `#TRANSFORM`s and `#OUTER`. The §6 gate's reopening condition is **met and executed** — see section G. Part 3 adds [`ecn_syntax.py`](../bcir/asn1/ecn_syntax.py): clause 20's defined syntax read from an `ENCODING-DEFINITIONS` module, with [`BCIR-FrameHeader.ecn`](../bcir/asn1/BCIR-FrameHeader.ecn) reproducing the gate's octets from text, and a canonical serialization so an ECN specification can finally be hashed. §21.3/§22.3/§22.8's determinants, §21.11's range conditions, §22.12's bit reversal and §22.1's replacement semantics are all built, and ECN is on the law rail as **R25** (`bcir.ecn.*`, statically decidable X.692 rules citing 43 distinct subclauses, 27 of them fixture-pinned). Clause 24's nineteen transforms, §22.7's repetition, the string/null/tag categories, and the constructor categories (§23.1 alternatives, §23.11 optionality, §22.9 identification handles, §22.5/§22.6 determination, §22.10 concatenation order) are all built. §22.11's contained types and §21.3.6/§21.5.6/§21.7.8's `container` determination are built too, clause 19's six value mappings are in [`ecn_mapping.py`](../bcir/asn1/ecn_mapping.py), and clause 12's encoding link module with clause 13's application-point algorithm is in [`ecn_link.py`](../bcir/asn1/ecn_link.py) — which retires the `AUXILIARY` and `BOUNDS` stated deviations by deriving both from the link rather than declaring them. Annex C's parameterization — X.683 as ECN rewrites it, `{<`/`>}` delimiters and all — is in [`ecn_param.py`](../bcir/asn1/ecn_param.py), together with §22.1.2's rules on the definitions a `REPLACE` names and §17.5.17's breadth-first `ComponentIdList` scan; C.2's three parameterized assignments now parse from module text and reach the digest. §17.5.1's `EncodeStructure` — the `ENCODE STRUCTURE { <field> <object>, ... } WITH <set>` object body that names an encoding per component — is in [`ecn_encode.py`](../bcir/asn1/ecn_encode.py) and readable from module text. §16.5's `OPTIONAL-ENCODING` marker with its `#OPTIONAL` objects, and §16.3's `AlternativesStructure` with its `#ALTERNATIVES` objects, are both readable. §22.1's `REPLACE` defined syntax reads from module text too, so a replacement is a specification rather than a Python assembly, and `_UNSUPPORTED_KEYWORDS` is down to one row. **A module now declares more than one encoding structure** — `EcnModule.structures` holds a `Structure` each — which made §22.1.2.7's `INSERT AT HEAD` readable from text, and §16.2.1's nested structures parse and reach the digest under their path, at `SYNTAX_VERSION` 8. Still refused, each for a stated reason: §22.11's contained-type *notation* (its semantics are built), §22.1.1.7 e)'s second replacement group, §16.4's `RepetitionStructure`, §21.7.6/§21.7.7's per-element continuation flag, R25's law-rail coverage of Annex C parameterization (the ECN dialect has `ecn.module`, `.class`, `.structure`, `.field`, `.object` and `.condition`, and no operation carrying a dummy parameter), and §16.5.6's per-level *application* of a nested structure — its notation is read, and an object applied to a structure that nests one is refused rather than given the parent's encoding |
| X.693 | 8825-4:2021 | XER | **built** — BASIC-XER + CXER (CXER out, both in; validated against Annex A.3/A.4); EXTENDED-XER by design excluded |
| X.694 | 8825-5:2021 | Mapping W3C XML Schema into ASN.1 | out of scope (see §7) |
| X.695 | 8825-6 | Registration of PER encoding instructions | follows X.691 |
| X.696 | 8825-7:2021 | OER | **built on both rails** (COER out, BASIC-OER in; validated against Annex A). [`bcir_oer.c`](../runtime/c/bcir_oer.c) is the C twin of the decoding half — schema-directed, because §6.2 leaves no choice — with dual-rail parity, a `#oer` gate at `-O0 == -O3`, and the twelfth fuzz target |
| X.697 | 8825-8:2021 | JER | **built** — clauses 20-41, the §7.2 constraint visibility rules, and the encoding instructions of cl. 14-19 (ARRAY, BASE64, NAME, OBJECT, TEXT, UNWRAPPED) with clause 13 precedence; X.697 registers NO canonical variant (§42.2), so the canonical profile is BCIR's own and carries no OID; the cl. 10-11 assignment syntax (X.680 surface) excluded. Bounded J1 oracle in `jer_bounded.py`: §4.3 limits, canonical-byte validation, framing and structured diagnostics — see [the JER compilation roadmap](BCIR_ASN1_JSON_ROADMAP.md) |

Sizes, as a rough effort signal (converted spec text, lines): X.681 1 125 · X.682 345 ·
X.683 567 · X.691 2 733 · X.692 **7 599** · X.693 2 562. ECN is the largest document in
the suite by a factor of three, and §0 is why it is nevertheless the destination.

## 2. What is built (baseline through PR #670)

- The whole of X.690 clause 8 on the oracle rail (`bcir/asn1/`, ~2 160 lines), clauses
  10 + 11 as a checker and a BER→DER rewrite.
- A freestanding, allocation-free, non-recursive C twin (`runtime/c/bcir_asn1.{h,c}`),
  fuzzed under ASan/UBSan, dual-rail differentialed over 12 000 mutants.
- The `BCIR-StreamPack` module and its **additive** DER projection, with the A1–A4 laws.
- X.691 PER on the oracle rail (`bcir/asn1/per.py`), ALIGNED and UNALIGNED, validated
  byte-identically against all four of the standard's own Annex A vectors in both
  variants, plus a freestanding C twin of clause 11 (`runtime/c/bcir_per.{h,c}`)
  differentialed against it and fuzzed under ASan/UBSan.
- The `BCIR-ArtifactBundle` module and additive DER/COER/CANONICAL-PER projection. It preserves the
  complete abstract BCAB directory and payloads while native offsets, padding, CRCs, and
  digests are recomputed; native→DER/COER→native is byte-identical. This is the second
  artifact family to prove that the encoding-rule rail is reusable rather than
  StreamPack-specific.
- X.681 information objects and X.682 constraints: object sets carry their **associated
  table** (X.681 cl. 13), and a component relation constraint RESOLVES an open type from a
  governing sibling at decode time — so `AttributeTypeAndValue`'s `value` decodes as the
  type its `type` OID names instead of opaque octets. Table constraints are deliberately
  invisible to the encoders (X.691 §10.3.4/§10.3.5), which is pinned by a law.
- The schema in the IR (`bcir.asn1.*`) and verifier law **R24**.

Two constraints inherited from that baseline shape everything below:

1. **DER out, BER in** — BCIR digests what it emits, so an encoding whose octets a peer
   may choose cannot be emitted. Every new encoding rule must name its canonical
   variant or be decode-only.
2. **Additive** — the native StreamPack and BCAB formats are frozen. A new encoding
   rule adds a transfer syntax; it never replaces one.

## 3. Dependency order

```
                    ┌─────────────────────────────────┐
                    │  A. X.680 front-end (parser)    │  ← unblocks everything
                    └───────────────┬─────────────────┘
                                    │
          ┌─────────────────────────┼──────────────────────────┐
          ▼                         ▼                          ▼
  ┌───────────────┐        ┌────────────────┐         ┌────────────────┐
  │ B. X.682      │        │ D. X.696 OER   │         │ E. X.697 JER   │
  │    constraints│        │   (octet-align)│         │   (JSON)       │
  └───────┬───────┘        └────────────────┘         └────────────────┘
          │
          ▼
  ┌───────────────┐        ┌────────────────┐
  │ C. X.691 PER  │        │ F. X.681/683   │
  │  ALIGNED +    │        │  info objects  │
  │  UNALIGNED    │        │  + parameters  │
  └───────┬───────┘        └───────┬────────┘
          └────────────┬───────────┘
                       ▼
              ┌──────────────────┐
              │ G. X.692 ECN     │  the metaprogramming layer
              └────────┬─────────┘
                       ▼
              ┌──────────────────────────────┐
              │ H. K_BCIR encoding selection │  the thesis
              └──────────────────────────────┘
```

Two hard dependencies, both from the standards rather than from BCIR:

- **PER cannot be built before constraints.** X.691 §10.3 defines *PER-visible
  constraints* — PER's entire size advantage comes from encoding a value in the
  fewest bits its constraint permits. Without X.682, PER degenerates to
  worst-case-width encoding and is pointless.
- **ECN cannot be built before there is more than one encoding to control.** ECN
  composes encoding objects, and its built-in object sets are "all the variants of BER
  and PER" (X.692 §9.5.3). With only DER in hand there is nothing to compose.

## 4. The phases

### A. X.680 front-end — the ASN.1 compiler front-end · **BUILT**

> **Status: delivered, and its stop condition since cleared.** `bcir/frontends/asn1/`
> (lexer · parser · printer · lowering), the `bcir-asn1c` CLI, and
> `bcir/tests/test_asn1_frontend.py`. `BCIR-StreamPack` is *parsed from its own text* and
> produces byte-identical DER for all 12 corpus programs. Both third-party gates now pass:
> `AuthorityKeyIdentifier` (37/37 real extensions) **and** `SubjectPublicKeyInfo`
> (**152/152** real certificates), the latter unblocked by phase F's open type.

Parse real ASN.1 module text into the existing `schema.py` type model. Today the model
is hand-built in Python; a peer's `.asn1` module cannot be consumed at all.

Deliverables: lexer + parser for the X.680 notation subset the schema layer can already
express, `bcir-asn1c` CLI, module/import resolution, and a round-trip law (parse →
model → print → parse yields an identical model). The natural home is `bcir/frontends/`
alongside the ROP/MAP/C front-ends, which already establishes the grammar → claims
pattern.

Gate: the module in `BCIR_ASN1_X690_ABI.md` §3, currently hand-built, is instead
*parsed* from its own ASN.1 text and produces byte-identical DER for every corpus
program. Third-party validation: parse the X.509 `AuthorityKeyIdentifier` and
`SubjectPublicKeyInfo` modules and match a known-good DER certificate field-for-field.

Stop condition: if the parsed subset cannot express X.509 without X.681 information
objects, stop and take phase F first rather than inventing a dialect.

### B. X.682 constraints · **BUILT**

X.680 clauses 49–51 (subtype constraints) landed first, because OER and PER choose an
encoding *from* a constraint. X.682's own three forms are now built on top:

* **Table and component relation constraints** (cl. 10) — see phase F above; this is the
  half that resolves open types.
* **User-defined constraints** (cl. 9) — `CONSTRAINED BY {...}` is recorded and never
  consulted by an encoder. §9 NOTE 1 calls it "a special form of ASN.1 comment" and X.691
  §10.3.3 makes it not PER-visible, so recording without acting is the whole contract.
* **Contents constraints** (cl. 11) — `CONTAINING Type [ENCODED BY oid]`. Unlike a value-set
  constraint this one says what the contents octets *are*, so it is modelled rather than
  discarded, and it resolves the same way a table-constrained open type does. It used to be
  refused outright, which was the honest answer while unimplemented. §11.3's restriction to
  OCTET STRING and BIT STRING is enforced.

The governing law across all three is negative: **none of them may move a bit**. X.691
§10.3.3–§10.3.6 make user-defined, table, component relation and table-dependent constraints
PER-invisible, and X.696 agrees, so a table-constrained value field's column lands on
`Primitive.table_values` and never on `Primitive.constraint` — the latter is what OER and PER
read to size a field. `test_a_table_constraint_never_moves_a_bit_of_any_encoding` pins it.

### C. X.691 PER (ALIGNED + UNALIGNED) · **BUILT**

`bcir/asn1/per.py`: clause 11's whole-number and length machinery (constrained,
semi-constrained, unconstrained, normally-small; the one- and two-octet length forms; the
§11.9.3.8 16K fragmentation), and clauses 12–14, 17, 19–24 and 30 for the types the BCIR
modules use. CANONICAL-PER out, BASIC-PER in, in both variants.

Extensibility is complete in both of its forms — the component-list marker (§19.1/§19.7–9,
including extension addition **groups** and the §19.9 open-type wrapper that lets an older
reader skip an addition it does not know) and the *constraint* marker
(§13.1/§17.3/§20.4/§30.4, where one bit says which side of the extension root a value fell
on and the root's own bounds supply the width).

**Validated byte-identically against all four Annex A records in both variants** — A.1
(unconstrained, 94/84 octets), A.2 (subtype constraints, 74/61), A.3 (extension markers,
83/65) and A.4 (extension addition groups, 8/8). That corpus paid for itself three times:
it found that serial constraint application was dropping the inner permitted alphabet, that
X.680 §50.11 *erases* a parent's extension marker under serial application, and that a
version bracket inside a CHOICE is presentational only (§23.8 NOTE).

The dual-rail C twin is `runtime/c/bcir_per.{h,c}` — freestanding, allocation-free,
non-recursive, `-Werror` clean, and required to give identical answers at `-O0` and `-O3`.
`bcir/tests/test_c_per.py` pushes one campaign through both rails and compares the decoded
value *and* the final bit position; `runtime/c/fuzz_per.c` is the tenth libFuzzer target on
the PR path. BCAB gains `native_to_per`/`per_to_native` under the same byte-identity law as
DER and OER, and the size law holds: 512 octets against DER's 551.

**Not built:** clauses 15, 16, 28 and 31 (REAL, BIT STRING, EMBEDDED PDV, unrestricted
character strings), none of which appears in a BCIR module. X.695 (registration of PER
encoding instructions) remains the small appendix it always was.

#### Original plan

The first *compact* encoding, and the first real test of the additive posture: a
StreamPack projected to UNALIGNED PER should be materially smaller than its DER form,
which makes it the right wire format for a bandwidth-capped channel — precisely the
condition BCIR's Θ already models.

Canonical variant: X.691 defines CANONICAL-PER, which is what BCIR emits (the DER-out
discipline generalized). ALIGNED and UNALIGNED are both emitted, because the choice
between them is a genuine cost trade (alignment padding vs. bit-shifting work) and is
therefore a *candidate*, not a configuration.

Gate: the A1–A4 laws restated for PER, plus a size law — `len(per(p)) <= len(der(p))`
for every corpus program — and a dual-rail C twin with the same differential
discipline. X.695 (registration of PER encoding instructions) follows as a small
appendix once PER lands.

### D. X.696 OER + DER→native fast path · **BUILT**

> **Status: delivered, both halves.** OER is `bcir/asn1/oer.py` (BASIC-OER +
> CANONICAL-OER, clauses 8–32), validated **byte-for-byte against the standard's own
> Annex A worked example** (95 octets). The DER→native fast path is
> `runtime/c/bcir_asn1_streampack.{h,c}`, wired into `check_runtime.sh` and fuzzed as the
> eighth trust boundary. Measured on the corpus: OER is **76.4 %** of DER and **41.6 %**
> of the native format.


Octet Encoding Rules: octet-aligned, no bit-shifting, and designed for fast
encode/decode rather than minimum size. Whether it has the lowest decode cost on a
specific target is a measurement question; the Python oracle does not establish that
ordering.

This phase pairs naturally with the **DER→native fast path**: a C-side decoder that
reconstructs a StreamPack directly from octets without the Python oracle. OER is the
format that path should prefer, and building both together avoids designing the C
reconstruction twice.

No dependency on constraints (OER's canonical variant, COER, is well-defined without
them), so this can run in parallel with B/C.

**How the OER half was validated.** X.696 Annex A carries a complete worked example — a
personnel record, 95 octets, with a per-field commentary. That fixture is transcribed into
`bcir/tests/test_asn1_oer.py` and the encoder reproduces it byte for byte. This matters
more than a round-trip test: a round trip passes just as happily when the encoder and
decoder share the same wrong assumption about the length determinant, the SEQUENCE
preamble bitmap or the SEQUENCE OF quantity field, which is exactly the failure mode an
implementation written from recollection produces. Annex A also happens to exercise the
three rules most easily got wrong — SET components in canonical tag order (§18.2, and the
record's `name` is `[APPLICATION 1]`, sorting ahead of `title`'s `[0]`), the presence
bitmap (§16.2.3), and the quantity field, which is a length determinant followed by the
count rather than a bare count (§17.2).

**The constraint dependency, stated.** Clauses 10, 13, 14 and 27 pick between a
fixed-width and a length-prefixed form from the type's *effective constraint* (§8.2.7,
§8.2.8). BCIR has no constraint model yet (phase B), so every integer takes §10.4 e) and
every string the length-prefixed form. That is not a shortcut — it is what X.696
specifies for an unconstrained type — and phase B adds the narrower forms without
changing any octets already emitted for one.

**What the fast path delivers on its own.** A driver that receives a DER projection can
now reconstruct the native artifact in freestanding C, with no Python in the path. The
law is byte identity, not equivalence:

    bcir_asn1_to_streampack(encode_pack(P)) == encode(P)

which forces the C rail to re-derive the three things the projection does not carry —
the native StreamPack version (v1/v2/v3 is a function of content; the module's own
`version` field is the *projection* version and is deliberately independent), the
reserved `stride_k` the projection omits by design, and the CRC. Proven byte-identical
on all 12 corpus programs and on all three native versions.

### E. X.697 JER · **BUILT, ORACLE AND NATIVE**

`bcir/asn1/jer.py` implements X.697 clauses 20–41 and the ARRAY, BASE64, NAME,
OBJECT, TEXT, and UNWRAPPED instructions with clause 13 precedence. The conformance
corpus covers deterministic emission and malformed-input rejection on the Python
oracle.

Interaction with the existing rails worth noting: BCIR already emits JSON in several
places (telemetry export, performance reports, model plans). JER can give selected
surfaces an ASN.1 schema and deterministic **textual** projection rather than an ad-hoc
shape. JER remains UTF-8 JSON and still requires lexical, UTF-8, escaping, bounds, and
numeric validation.

X.697 defines no canonical variant. BCIR's deterministic emit mode is the private,
versioned **BCIR canonical JER profile**, carries no standards OID, and must not be
described as CJER.

This paragraph used to end by listing six things as open — the C scalar parser, generated
schema descriptors, the MLIR family/profile representation, direct claim lowering, the hosted
SIMD scanner, and native K_BCIR measurements. **All six have since landed** as J2, J3, J4
parts 1 and 3, J5 and J6 respectively; see the phase table in
[`BCIR_ASN1_JSON_ROADMAP.md`](BCIR_ASN1_JSON_ROADMAP.md), which is where their exit gates are
recorded. J7, the driver experiment, is the only phase still open, and it is blocked on
hardware access rather than on design (§5.1).

### F. X.681 information objects + X.683 parameterization · **BUILT**

**Built.** Classes with WITH SYNTAX; objects in both DefaultSyntax (`{&f v, ...}`) and
DefinedSyntax (`{"A" 1 INTEGER}`, matched against the class's syntax list including its
literal words); object sets with union, reference splicing and the §12.3 extension marker;
and the **associated table** of clause 13 — rows are objects, columns are class fields, and
a type field's column is a column of *types*.

That last point is what the whole chapter is for. X.682 §10.19/§10.20 select a row from the
values of sibling components, and a selected row names the type an open type's octets
actually are. `bcir/tests/test_asn1_objects.py` carries X.682 clause 10's own worked example
(ERROR-CLASS / ErrorSet / ErrorReturn) plus the X.509-shaped `AttributeTypeAndValue`.

Resolution is an **enrichment, not a replacement**: the octets stay exactly as they arrived
and the decoded value appears alongside them under `<name>.resolved`. An unmatched row
produces no key at all rather than a guess — X.681 §12.9 explicitly permits a peer to use an
object outside an extensible set, so an unresolvable open type is ordinary traffic.

**X.683 parameterization is built.** A parameterized assignment keeps its body
UNRESOLVED — §9.7 makes instantiation a substitution of actuals for dummy references, so
lowering the body eagerly would have to invent a type for each dummy and any type it
invented would be wrong for some instantiation. A reference substitutes structurally over
the AST and lowers the result, memoised on the ACTUALS rather than the name so two
instantiations of one template stay independent.

That is what makes X.681/682 fire on real modules. RFC 5280 writes
`AttributeTypeAndValue {ATTRIBUTE:Supported}` with its table constraints naming the DUMMY
set, so instantiation has to rewrite `{Supported}` to the actual before the associated table
can be built. Before this the machinery was correct and simply never triggered.

**Not built.** §9.8's NOTE — the actual parameter's TAGGING ENVIRONMENT applies, not the
dummy's, which differs only when the actual crosses a module boundary with a different tag
default. This front-end lowers one module at a time, so the two coincide. Also
`ObjectFromObject` (§15) and the self-referential link fields of §13.2 b), whose column set
is deliberately infinite.

### G. X.692 ECN · **BUILT-IN MODEL LANDED; USER-DEFINED HALF REOPENED ON EVIDENCE**

The Encoding Control Notation: a language for **defining encodings**, in which encoding
classes are declared, encoding objects realize them, and encoding object sets are
applied to ASN.1 types to determine their wire form.

ECN is where the §0 correspondence stops being an analogy and becomes an
implementation. An ECN encoding object set *is* a plan; applying one to a type *is*
realization. BCIR's contribution is that ECN never specified how to *choose* among
legal object sets — it is a notation, not an optimizer — and BCIR has the optimizer.

The Python oracle contains the ECN class/object/object-set model, EDM/ELM, the built-in
BER and PER object sets, and the one-object-per-class checks. It now also contains the
user-defined half: bit-level encoding spaces with stated widths, justification within a
space, `#PAD` fields that carry no abstract value, transmission order chosen by the object
rather than the type, `INT-TO-INT` and `INT-TO-BITS` transforms with an inverse each, and
`#OUTER`. It now also **parses the ECN surface syntax**: [`ecn_syntax.py`](../bcir/asn1/ecn_syntax.py)
reads clause 20's defined syntax — the bracket-optional keyword grammar that clause 23's
`WITH SYNTAX` statements spell out — from an `ENCODING-DEFINITIONS` module, resolving clause
11's class assignments and §16.5's `ConcatenationStructure`, and
[`BCIR-FrameHeader.ecn`](../bcir/asn1/BCIR-FrameHeader.ecn) is this section's own workload
written in that notation. It produces `aa00` byte-for-byte against the Python-assembled
objects, which makes it a second opinion rather than a second spelling. No ECN representation
has been added to MLIR.

**Reading the `WITH SYNTAX` text corrected three more things**, all in the same direction as
the earlier citation pass — properties that had been collapsed into flags:

- §22.2 and §22.8 are property *groups*. An `align_before: bool` is `ALIGNED TO NEXT octet
  PADDING zero` with the unit, the padding and the pattern all frozen, where §22.2.1.1 gives
  all three; legacy layouts align to nibbles and to 16-bit words and pad with ones.
- §21.8.1's `Justification` is `CHOICE {left INTEGER(0..MAX), right INTEGER(0..MAX)}`, and the
  **offset** had been dropped. §22.8.3.3/§22.8.3.4 split the padding as `b-n`/`n`, so a field
  sitting two bits in from the top of its space was simply unreachable before.
- §22.10.1.1 gives a concatenation object only `{textual, tag, random}` — there is no property
  naming a component order. §22.10.3.1 reads `textual` from "the ASN.1 type specification **or
  the ECN structure definition**", so the free order tuple is really the §16.5 structure's, and
  the module now states that source instead of implying it.

Every property group the repository has not built — `REPLACE`, `START-POINTER`, `IF`/`IF-ALL`
conditions, `DETERMINED BY`, `USING`, `UNUSED BITS`, `EXHIBITS HANDLE`, `BIT-REVERSAL` — is
recognized by the parser and refused with the clause that defines it. Skipping an unimplemented
keyword would emit octets the handed specification does not describe, which is precisely the
defect class the triple-rail design exists to catch.

**The remaining property groups are built, and ECN is now on the law rail.** Six clauses
that the first surface pass refused by name are implemented:

- **§21.3 determinant-based encoding spaces**, **§22.3 start pointers** and **§22.8 `UNUSED
  BITS` with a field reference** all run off one mechanism, because they are one problem.
  §22.8.3.7's NOTE states it: "the encoding of the `USING` reference ... appears earlier in
  the encoding than the encoding of this field, and an encoder will need to **suspend** the
  encoding of that field until the value to be determined". So `BitWriter` reserves an
  auxiliary field's bits where it sits and patches them when the determinant is known — one
  pass, no re-derivation of encoder's-option choices a second pass would have to reproduce.
  An auxiliary field nobody sets is a refusal, not zeros.
- **§21.11 `IF`/`IF-ALL` range conditions** with §21.12's `Comparison`. This is the clause
  that makes an ECN integer encoding *schema-directed*: §23.6.3.1 selects "the first
  `#CONDITIONAL-INT` encoding object whose conditions are satisfied", and §21.11.3 tests the
  **bounds of the type** rather than any value — so one object set encodes `INTEGER (0..255)`
  in eight bits and `INTEGER (0..65535)` in sixteen with no value involved in the choice.
- **§22.12 bit reversal**, all four `ReversalSpecification` values, over the encoding space's
  contents including value padding and excluding pre-alignment (§22.12.1.4's NOTE 2), with
  `#OUTER`'s different subject: §22.12.3.1 divides "the entire encoding (after any `PADDING`
  has been applied)".
- **§22.1 `REPLACE`** semantics: all five actions' restrictions, instantiation of a
  single-class-parameter structure around the replaced field (§22.1.3.1, §22.1.3.5), and
  §22.1.3.6's head-end insertions hoisted to the front **as a block, in component order** —
  not interleaved with the components they belong to, which is the reading that first
  suggests itself and is what makes them useful as location determinants.

**A seventh divergence, and it is in the text rather than in the code.** §21.14.1 lists
`ReversalSpecification` as `{no-reversal, reverse-bits-in-units, reverse-half-units,
reverse-bits-in-half-units}`. §21.14.6 then describes four actions "in the order of
enumerations listed above" and gives a *different* order. §22.12.3.2 agrees with §21.14.1 and
with what the names say, so §21.14.6's listing is the outlier — recorded on both rails rather
than silently resolved, because the other reading produces well-formed octets of the wrong
shape and a parity test is the only thing that would catch it.

**ECN is on the law rail as R25.** `mlir/include/BCIR/BCIREcnOps.td` gives `bcir.ecn.module`,
`.class`, `.structure`, `.field`, `.object` and `.condition`; `BCIRVerifyPass.cpp` enforces
statically decidable X.692 rules over them, citing **43 distinct subclauses** as of the ECN
build-out's close — 27 of those checks are pinned to the fixture that trips them by
[`test_asn1_ecn_law_parity.py`](../bcir/tests/test_asn1_ecn_law_parity.py), because a law with
no witness is a claim rather than a check. (This sentence said "eleven" from the slice that
introduced R25 until the closing audit; the count below in the slice-C paragraph is *history*
and is left alone deliberately.) R25 exists for a sharper version of R24's
reason: an encoding definition module is written once and applied to *many* types, so a fault
that only fires on the right value can sit in one indefinitely. The bit-level values stay in
the oracle — "this pattern is 0101" is not a proposition that can be false — and what the IR
holds is every property whose *combination* with another can be. `verify_ecn.mlir` carries one
negative fixture per rule plus two positive modules, and `test_asn1_ecn_law_parity.py` reads
the ODS directly and pins each rule to the fixture that trips it.

**Clause 24 is complete: all nineteen transforms.** The first pass built two — `int-to-int`
and a fixed-width `int-to-bits`. The rest are now here, and reading them turned up structure
worth stating. They fall into three groups, not one: eleven *value* transforms, three
*composite constructors* (§24.14–§24.16) that turn a string into §24.2.1's transform composite,
and three *collapsers* (§24.17–§24.19) that put one back. Every value transform states the
composite rule in the same words — §24.4.4 is representative — so it is implemented once in
`Transform.apply` and each transform writes only the scalar case; eleven hand-written loops
would be eleven chances to differ in a way no test distinguishes.

Three clause readings are worth recording because the obvious implementation gets each wrong:

- **§24.7.13 pads in front of the sign.** It pre-fixes the pad to §24.7.9's representation,
  which already carries the `-`. So `-7` in a four-character field with zero padding is `00-7`,
  not the `-007` every printf produces. Nothing in clause 24 says the pad goes after the sign,
  and saying so would have taken a sentence.
- **§24.8.17 fills with the sign bit, not with zero.** A two's-complement encoding widened to
  a fixed size "shall have bits prefixed **equal in value to the original leading bit**";
  zero-extending would change the sign.
- **§24.10.10.2 and §24.11.6.2 check different things.** `char-to-bits` requires distinct
  *characters*; `bits-to-char` requires distinct characters **and** distinct bitstrings. The
  asymmetry is deliberate: two characters sharing a bitstring is lossy but well defined, while
  two identical source bitstrings make the transform not a function.

The module layering moved to make room. Clause 21's property types are now
[`ecn_props.py`](../bcir/asn1/ecn_props.py) and clause 24 is
[`ecn_transform.py`](../bcir/asn1/ecn_transform.py), with `ecn_user` re-exporting both — a
file-layout decision, not a change to the public surface.

**Repetition, and the categories that turn out to need it.** The plan for this slice was
"clause 23's remaining bit-field categories", and reading them changed it. §23.2's `#BITS`,
§23.9's `#OCTETS` and §23.4's `#CHARS` have **no `ENCODING-SPACE` group at all** — their
`WITH SYNTAX` gives pre-alignment, a start pointer, `VALUE-REVERSAL`, `TRANSFORMS` and
`REPETITION-ENCODING(S)`, so a string's size comes from §22.7's repetition space and not from a
stated width. §21.7.3 says the same thing from the other side: `RepetitionSpaceDetermination`
"**replaces** use of an encoding property of type `EncodingSpaceDetermination` in the encoding
of repetitions". Sibling, not subtype. So the three string categories could not be built before
repetition was, and they arrive together with §21.13, §22.7, §23.13 and §23.14.

**§21.13 is §21.11's sibling and their NOTEs disagree, deliberately.** §21.11.4's says "For any
given set of bounds, exactly one predicate will be satisfied"; §21.13.4's says "Only the
`fixed-size` case overlaps with other predicates". A fixed size *is* an upper bound with a
lower bound, so `SIZE(4)` genuinely satisfies two shapes. Carrying the integer sibling's
exhaustiveness across would pick the wrong encoding for every fixed-size string — which is the
common case, not an edge. §21.13.4 a) also turns on the lower bound being **zero** where
§21.11.4 a) turned on one *existing*, because an X.680 size always has one.

Five of §21.7's eight repetition-space determinations are built: a count field
(`field-to-be-set` / `field-to-be-used`), a terminator (`pattern` — this is the NUL-terminated
string, and it is why the group carries a `Pattern` at all), §21.7.10's identification handle,
and §21.7.11's fixed count. The other three are refused by name: `flag-to-be-set` and
`flag-to-be-used` put a continuation flag **inside the repeated element** (§21.7.6/§21.7.7),
which needs the element's own structure; `container` needs containment.

§23.8's `#NUL` and §23.15's `#TAG` needed nothing new and are built. `#NUL` is the one category
where `VALUE-PADDING` *is* the value encoding, since X.680's NULL carries no information;
`#TAG` is §20.2's composition — "preceded by one or more instances of a class in the tag
category" — which is exactly how BER's identifier octet relates to its contents.

**The constructor categories, and the one mechanism four clauses were waiting on.** §22.9's
identification handle is what ECN offers *instead of* a discriminant field, and four separate
clauses depend on it: §21.5.7 for optionality, §21.6.6 for alternatives, §21.7.10 for
repetition end, §22.10.2.1 for a randomly ordered concatenation. Every one of those was
previously refused with the words "§22.9's identification handles are not built". Building the
handle turned all four on at once — which is why §23.1's `#ALTERNATIVES`, §23.11's `#OPTIONAL`,
§22.5, §22.6, §22.10 and §21.16 arrive together rather than one clause at a time.

A handle is not a field. §22.9.1.4's three parts are a name, "the bit positions that form the
handle", and "the possible bit patterns ... occurring in the encodings produced by this
encoding object" — so it is a *declaration about bits that are there anyway*, which is how
BER's tag and IPv4's version nibble actually discriminate. Three readings shaped the
implementation:

- **§22.9.1.5 puts position zero after pre-alignment**, and after any §22.12 bit reversal. So
  the window is into the encoding *space*, not into the padding that precedes it, and the
  §22.9.3.1 check has to run against the written buffer rather than against the value.
- **§22.9.1.6 makes the positions a set**, "not necessarily contiguous, and not necessarily in
  ascending order in the ECN specification", ordered "from the zero position ... upwards". A
  handle can be bits 0, 3 and 7 written in any order.
- **§21.16's six alternatives all reduce to integer ranges.** The question the four consuming
  clauses ask is disjointness, and that is not answerable by enumerating 2^n patterns — so
  `HandleValueSet` normalizes to inclusive ranges, and `tag:any` (§21.16.5) is *refused* until
  a tag number resolves it rather than defaulting to a set that matches nothing.

Two determination enums arrived with them, and they diverge for the reason §21.3 and §21.4 do.
§21.5.1 lists five values; §21.6.1 lists three. A CHOICE always encodes exactly one
alternative, so neither `container` nor `pointer` has anything to say about which one it was.
§22.6.1.1's ordering property has the same shape of difference — `{textual, tag}` where
§22.10.1.1's has `{textual, tag, random}`.

**Two stale refusals fell out.** §22.1.1.7 c) and d)'s `REPLACE OPTIONALS` and `REPLACE
NON-OPTIONALS` were refused with "the optionality category is not built on this rail"; it is,
so they now select. §22.1.3.4 then supplied the rule the obvious implementation gets wrong: a
replaced optional component is replaced "with a **non-optional** instantiation", and the actual
parameter de-references to the component "**except for any class in the optionality
category**". `REPLACE OPTIONALS` removes optionality rather than wrapping it.

R25 grew to twenty-two rules for the same slice: §22.9.1.6, §22.9.1.9, §22.9.2.1 and §22.9.2.3
over handles, the eight parallel restrictions of §22.5.2 and §22.6.2, §22.5.2.4's start-pointer
requirement, §22.6.1.1's two-valued ordering and §22.10.2.1's handle prerequisite. §22.9.2.1 and
§22.9.2.3 are the two that could only ever live here: both relate one `EXHIBITS HANDLE` clause
to every other in the module, so no object-local check reaches them.

**Containment is two relationships, and reading them as one is the mistake.** §22.11's
`CONTENTS-ENCODING` is *"my contents are another type"* — an `OCTET STRING (CONTAINING Inner)`
whose contents are encoded by a **different object set**. §21.3.6's, §21.5.6's and §21.7.8's
`container` determination is *"my end is what bounds a component"*. Both are built, and they
needed different machinery.

§22.11.2 is a **five-row table, and X.692 contradicts itself about the last row** —
`CONTENTS-ENCODING` set, an `ENCODED BY` contents constraint present, `OVERRIDE` left FALSE.
§22.11.2.2's closing sentence sends that case to "the combined encoding set applied to the
**containing type**"; §13.2.10.6 a) sends the same case to the `ENCODED BY`, saying an object
that "specifies that it should not override an `ENCODED BY`" leaves it that "the `ENCODED BY`
specification **shall be used**". §13.2.10.6 a) is taken as correct on three counts: §22.11.1.3
makes the group's purpose deciding "whether an … `ENCODED BY` … shall be **overridden**", and
declining to override should leave it standing rather than discard it; §22.11.2.1 gives the
parallel unset case to the `ENCODED BY` too; and §13.2.10.6 is the application-point algorithm,
the operative procedure. Recorded rather than silently resolved, in the same style as
§21.14.6's ordering — the first implementation followed §22.11.2.2 and was corrected when
clause 13 was read.

§22.11.1.4's combination is §9.23.2's, which §13.2.3 b) states again: a **left-biased** merge
where `COMPLETED BY` fills gaps and never overrides, which is what makes
`COMPLETED BY PER-BASIC-UNALIGNED` safe under a handful of specialized objects.

The `container` determination's encoder actions are nil — §22.7.3.6 says "there is no further
encoder action" outright — and what each clause *does* give the encoder is a rule to **check**:
the element must be the last encoding placed in the container. §21.3.6's NOTE says why that is
worth checking rather than trusting, and the symptom otherwise is a decoder reading one field's
bits as another's.

**The notation has no `OUTER` keyword**, which is the reading that shaped the model. §21.3.6
calls the second form "a specification that the end of the PDU determines the end of the
encoding space (using `OUTER`)", but §22.4.1.2's syntax is `USING &encoding-space-reference` in
every case and §22.4.1.6 calls that reference one "to an auxiliary field or to a field carrying
abstract values, **or to a container**". So the PDU is simply the outermost container and
clause 25's `#OUTER` names it — a reserved reference, and the grammar needed nothing new.

One implementation decision is worth recording because the first attempt got it backwards. A
contained type gets its own **reference scope** and not its own bit buffer: §9.24.2 moves the
application point into it, so its `REFERENCE`s resolve among its own fields, but its bits go
straight into the containing encoding — because "the last encoding placed in the container" is
a question about offsets in one stream, and isolating the buffer made that rule unanswerable.

**Clause 19's six value mappings are built, and they answer a different question from
everything else in the ECN half.** Clauses 21–25 say how a value becomes bits; §19.1.1 says
*which* value the fields of one encoding structure hand to the fields of another. That is what
lets an ASN.1 `INTEGER` be encoded as a concatenation of fields, or a four-way CHOICE as a
compact integer — neither of which is a transform on bits. §19.1.7's production lists **six**,
not the five its own table of contents suggests: `MappingIntToBits` is §19.7.

Three sentences in the clause are traps, and [`ecn_mapping.py`](../bcir/asn1/ecn_mapping.py)
exists largely for them:

- **§19.5.5 orders `TRUE` before `FALSE`.** Every programming language orders booleans the
  other way — Python's `sorted([True, False])` is `[False, True]` — so an ordering derived from
  the host's comparison is exactly backwards, and backwards *silently*: both directions produce
  well-formed encodings that differ only in what the bits mean. This is the most dangerous
  sentence in clause 19 and `BooleanOrdering` states it once, in longhand, so no later
  "simplification" into `sorted(...)` can pass review.
- **§19.4.6 requires reversibility where the value path does not.** Table 6 lets `modulo:n` be
  legal and lossy when a transform is *encoding* a value; a mapping a decoder cannot undo loses
  the value instead. The same asymmetry §22.3.2.3 and §22.8.2.4 impose on determinants.
- **§19.5.11 and §19.5.12 are not symmetric, and neither is an error.** A destination ordering
  shorter than the source means some abstract values cannot be encoded; a longer one means some
  encodings will never be generated. Both are conforming, so both are *reported* rather than
  refused — only asking to map a value past the end fails.

§19.3's `FIELDS` mapping is the `AUXILIARY` deviation seen from the clause's own side: §19.3.1
gives the target "fields corresponding to the components of the type, **but also … added fields
for determinants**". Naming those fields rather than inferring them is what keeps a typo in a
field name from silently becoming a determinant.

**The encoding link module is clause 12, and it retires two stated deviations.** Clause 12's
own NOTE separates it from the EDM — "There are two top-level productions in ECN, the
`ELMDefinition` specified in this clause and the `EDMDefinition` specified in clause 14" — and
this roadmap cited clause 14 for it until the text was read. §12.1.9 gives the ELM one job:
"the sole function of an ELM is to apply encodings."

[`ecn_link.py`](../bcir/asn1/ecn_link.py) builds §12.2.1's `ENCODE <class>,+ WITH <primary>
[COMPLETED BY <secondary>]`, §13.2.3's combined set, §13.2.10's application-point resolution,
and the *link* itself. That link is what the two deviations were standing in for:

- **`AUXILIARY`** was a keyword X.692 does not have, because §22.1.2.6's classification comes
  from the ELM: a structure field with no ASN.1 component behind it is auxiliary. §19.3.1 says
  the same from clause 19's side — a structure "has fields corresponding to the components of
  the type, **but also has added fields for determinants**". `LinkedStructure.auxiliary_fields`
  now *computes* that set from the two field lists.
- **`BOUNDS`** was the type's bounds written on the encoding object. §21.11.3 tests "the bounds
  on the integer values associated with an encoding class", and §23.7.2.6's NOTE insists the
  condition is tested "on the bounds of the original value". `LinkedStructure.bounds_for` reads
  them off the ASN.1 component's constraint, and an unconstrained INTEGER answers
  `unbounded-or-no-lower-bound` — which is an answer, not a gap.

Both keywords stay accepted, because a specification written against this rail may have no
ASN.1 type in hand. They are fallbacks now rather than the source of truth, which is the
difference between a deviation and a convenience.

§13.2.10's algorithm is three sentences and worth having exactly: apply an object of the same
class if the combined set has one (§13.2.10.1); otherwise de-reference the class and recurse
(§13.2.10.1 a), §13.2.10.7); otherwise "the ECN specification is in error" (§13.2.10.8). The
de-referencing is what makes clause 11's `#Version ::= #INT` do any work — one object written
for `#INT` covers every class assigned from it.

**Annex C is a rewrite of X.683, not a reference to it** — and the rewrite is two characters
wide on each side. [`ecn_param.py`](../bcir/asn1/ecn_param.py) is that model: C.1's
`ParameterList ::= "{<" Parameter "," + ">}"` against X.683 §8.3's `"{" ... "}"`, the five
things a dummy may stand for and the governor each one requires, C.4's ten actual-parameter
alternatives against its eight correspondence rules, and §22.1.2's rules about the definitions
a `REPLACE` names. An implementation that reuses the X.683 parser it already has accepts
`#Length-prefixed{#D}` and refuses `#Length-prefixed{<#D>}` — the one spelling ECN admits —
while citing X.683 correctly throughout.

Three readings in there are worth carrying:

- **C.1's NOTE forbids `DummyGovernor`s in ECN.** X.683 lets one dummy govern another; the
  identical text is an error here, so the check belongs to the *list* rather than to the
  parameter — a governor is only a dummy governor relative to its siblings.
- **C.3 gives `{<>}` a meaning opposite to `{< ... >}`'s.** `ParameterizedReference ::=
  Reference | Reference "{<" ">}"` makes an empty actual list a legal way to *name* a
  definition, while C.1's `"," +` makes an empty parameter list not a `ParameterList` at all.
  The two productions share their delimiters and disagree about zero.
- **§17.5.17's scan is breadth-first.** A `ComponentIdList`'s first identifier is resolved "by
  the first match in a scan (in textual order) of the outer-level identifiers, then by a scan
  of the second level identifiers, and so on". The obvious recursive walk is depth-first, and
  the two disagree exactly when an inner name shadows an outer one — at which point both still
  name a real field, so nothing fails and the encoding simply points elsewhere.

**Two citations in this repository were each other's,** and reading §16.2.12 is what caught it:
`AlternativesStructure` is **§16.3**, `RepetitionStructure` is §16.4, and
`ConcatenationStructure` — the shape this rail's parser models — is **§16.5**. The optional
component marker is §16.5.1's own `ConcatComponentPresence`, spelled `OPTIONAL-ENCODING`
followed by an `OptionalClass`, not `OPTIONAL`. Both refusal messages and the test that pins
them are corrected; the same class of error as the ELM's "clause 14", found the same way.

**The notation is read now too.** `ecn_syntax.py` lexes `{<` and `>}` as single tokens and
parses C.2's three parameterized assignments, so `#Length-prefixed{<#D>} ::= #CONCATENATION {
length #INT, value #D }` and its `ENCODED BY` partner are module text rather than Python. Three
things fell out of building it:

- **A parameterized class body is two things.** §16.2.12's `EncodingStructureDefn` is a class
  *and* a braced field list, so a body that stopped at the class would leave the braces to be
  misread as the next assignment — which is what the first attempt did.
- **§22.1.2.2 and §22.1.2.4 are checkable at declaration time**, before any `REPLACE` names the
  pair. A module may define a replacement pair and apply it from an ELM this rail never reads,
  and a pair that could never be instantiated is invalid on its own terms either way.
- **Governors belong in the digest.** `render` gives the bare form §22.1.2.2 requires at a
  *use*; a declaration carries governors, and two modules differing only in one describe
  different octets. `SYNTAX_VERSION` moved to **5** for the same reason it moved to 4 for
  `EXHIBITS HANDLE`: a name two specifications share is a name that means two things.

**What is still refused at the surface, and D.3.2.3 names it precisely.** The remaining gap
was recorded as §22.1.2.6's auxiliary-field binding until the annex's worked replacement was
read, and that example shows the real one. D.3.2.3 writes:

```
optional-with-determinant-encoding
{<#Element, #ENCODINGS:Sequence2-combined-encoding-object-set>}
#Optional-with-determinant{<#Element>} ::= {
    ENCODE STRUCTURE {
        determinant determinant-encoding,
        component   USE-SET OPTIONAL-ENCODING if-component-present-encoding{<determinant>} }
    WITH Sequence2-combined-encoding-object-set }
```

Every piece of Annex C this rail now builds is in those five lines — the `{< >}` delimiters, an
`#ENCODINGS` governor, a governor instantiated with the object's own dummy — and **one whole
object-body form it does not**: §17.5.1's `EncodeStructure`. That is how an `ENCODED BY` object
says which object encodes each field of the replacement structure, and without it §22.1.3.5's
"set according to the specification in the replacement structure encoding object" has nothing
to read. §22.1.2.6 *classifies* the auxiliary fields; it never says how they are encoded, which
is why citing it was the wrong answer to "what is missing".

So the order for the remainder was fixed by that example rather than chosen, and the first
step is **built**: [`ecn_encode.py`](../bcir/asn1/ecn_encode.py) is §17.5.1's `EncodeStructure`,
and `ecn_syntax.py` reads it as a `#CONCATENATION` object body.

**Three clauses demand the trailing `WITH <object set>`, for three different reasons**, and an
implementation that checks one accepts specifications the other two forbid — so they are three
checks with three messages, because they are three different repairs:

| clause | why the set is required | the repair |
| --- | --- | --- |
| §17.5.3 | no `STRUCTURED WITH`, so nothing encodes the constructor itself — its NOTE: "a complete encoding has to be produced" | add a `STRUCTURED WITH` |
| §17.5.6 | some `EncodingOrUseSet` is `USE-SET`, which *means* "apply the `CombinedEncodings`" | drop the `USE-SET` |
| §17.5.10 | a component has no `ComponentEncoding`, and the set must "provide a complete encoding of that component" | write the component in |

§17.5.9 and §17.5.11 are **biconditionals**, the same shape §22.1.2.5 uses for `INSERT AT HEAD`:
the optional-component spec is used "if and only if the component is optional", and the
identifier is omitted "if and only if" the governor is a repetition class with no identifier on
its element. Both directions of both are faults.

**What this body form buys, concretely.** §9.5.2 permits at most one encoding object per class
*in the object set*, so the property-group body reaches every field through its class and two
fields of one class necessarily share an encoding. §17.5.10's `ComponentEncoding` names an
object directly — a different route to the same field, not bound by the set — so a module with
two objects for one class is a specification the old body cannot use and this one can. §17.5.13
keeps that honest: a named object "shall be governed by the corresponding encoding class", or an
integer object could encode a boolean field and produce well-formed octets of the wrong shape.

The digest does **not** move for an `EncodeStructure` whose every component says `USE-SET`, and
that is the correct answer rather than a gap: a canonical serialization names what octets a
specification describes, not how it was spelled. It is the exact opposite of the `EXHIBITS
HANDLE` and parameterized-assignment cases, where the spelling changes what a decoder reads and
the hash has to move.

**§16.5's half of the structure notation is built too.** `ConcatComponentPresence ::=
OPTIONAL-ENCODING OptionalClass` (§16.5.1) now parses on a concatenation component, `#OPTIONAL`
joined the built-in classes, and §22.5's `PRESENCE` group is read from an `#OPTIONAL` object
body — so `PRESENCE` left the unimplemented-keyword table.

The pairing is the interesting part, and it has **two owners**. §16.5.4 puts the mechanism on
the object — "the mechanism used to determine whether there is an encoding of the corresponding
`EncodingStructure` is specified by the encoding object which encodes the `OptionalClass`" —
while the component is the structure's. Neither half can do it alone, so `optional_wrapped` is
the one place that knows both, and it takes a field name and a spec rather than living on
either. §16.5.3 supplies the other case: an *unmarked* component "shall appear precisely once
in the encoding", which is why the wrap is safe to attempt on every field and returns the
unmarked ones untouched.

`structure_optional` is a sidecar mapping rather than a third element of `structure`'s tuples,
and §16.5.3 is the reason: the marker's absence is the common case *and* is meaningful, so
widening every field's tuple to carry a mostly-absent fact would touch every reader of
`structure` to express nothing new.

`SYNTAX_VERSION` moved to **6**. A component that may be absent is read differently from one
that is always there, so two modules differing only in the marker describe different octets —
the same argument that moved it to 4 for `EXHIBITS HANDLE` and 5 for Annex C, and the exact
opposite of §17.5's all-`USE-SET` `EncodeStructure`, which is a second spelling of one encoding
and deliberately hashes the same.

**§16.3's `AlternativesStructure` is built too, and it did not need the structure tree.**
§16.2.12 names three `EncodingStructureDefn`s — `AlternativesStructure` (§16.3),
`RepetitionStructure` (§16.4), `ConcatenationStructure` (§16.5) — and the two that are read
**share their body**: §16.3.1's `NamedField ::= identifier EncodingStructure` is what both are
built from. What differs is meaning, not shape. §16.3.2 has the structure identify "the presence
in an encoding of **precisely one** of the `EncodingStructure`s in its `NamedFields`", against
§16.5.2's zero-or-one for each. Same text, opposite semantics, and nothing but the governor's
category tells them apart — so one function reads both and `structure_category` records which.

The tree was the anticipated cost and it turned out to be the wrong thing to buy: nesting is
what needs a tree, and nesting is a separate refusal (§16.2.1) that neither §16.3 nor §16.5
requires. Both are readable flat.

Three consequences worth keeping:

- **The object and the structure must agree on the category.** §16.3.3 and §16.5.6 both make
  their structure "an encoding constructor" the application point proceeds through, so an
  `#ALTERNATIVES` object over a concatenation would encode one field where all of them belong —
  a valid encoding of a *different* type, which is the kind of mistake that produces well-formed
  octets and no complaint.
- **The `OPTIONAL-ENCODING` tail is a concatenation's alone.** §16.5.1 hangs
  `ConcatComponentPresence` off a `ConcatComponent`; §16.3.1's `NamedField` has no such tail,
  and §16.3.2 is the reason rather than an accident of the grammar.
- **§22.6.1.1's `&alternative-ordering` is `ENUMERATED {textual, tag}` — two values where
  §22.10.1.1's concatenation group has three.** `random` would be meaningless: a CHOICE encodes
  exactly one alternative, so there is no order to randomize.

`SYNTAX_VERSION` moved to **7**. `ALTERNATIVE` left the unimplemented-keyword table with
`PRESENCE`; what remains there is `CONTAINED`, plus the groups that *are* built and are refused
when written in a way their clause forbids.

**A module holds more than one encoding structure, at `SYNTAX_VERSION` 8.** The single flat
structure on `EcnModule` became `structures: dict[str, Structure]`, and two refusals fell out of
that one change because both had the same cause.

The cause was a misread. The repository took "one structure per module" off **§13.2**, which
says the *link* walks one application point — nothing in clause 16 caps how many structures a
module may **declare**. The two are different facts, and conflating them made §22.1.2.7's
`INSERT AT HEAD` unreachable from text: that clause needs a second ordinary structure to exist,
so its semantics sat in `ecn_user.HeadEndStructure` with no syntax able to reach them. The
dialect had it right all along — `BCIR_EcnModuleOp`'s `applies_to` is optional precisely because
"an EDM that named its application point would be claiming a binding the notation puts
elsewhere".

- **§22.1.2.7 reads from module text**, with each of its three sentences a check: the structure
  has no dummy parameters (so a parameterized name is refused — the one property separating it
  from the `WITH` structure §22.1.2.2 requires to *be* parameterized), it is not the application
  point itself, and every field is auxiliary with an encoding object under §9.5.2.
- **§16.2.1's nested structures parse**, recursively and by the same function, since the nested
  production *is* the top-level one. A nested definition is named by its path (`Outer.head`) and
  reaches the digest under it, so two structures each nesting a `head` keep their own.

Two things this slice deliberately did **not** do, both stated rather than left to be discovered:

- **Which structure is the application point is a repository convention: the first declared.**
  §12.1's ELM is what binds a type to a structure, and an `ENCODING-DEFINITIONS` module alone
  never says which of its structures is applied. First-declared rather than sole-declared is
  forced by timing — clause 23's objects resolve their structure *while the module is still
  being parsed*, so a rule needing the whole module ("the one nothing else claims") would answer
  differently depending on how far the reader had got. A test pins the convention and says it is
  the assertion to change when an ELM section becomes readable.
- **Reading a nested structure is not encoding one.** §16.5.6's application point "proceeds to
  each of the `EncodingStructure`s" — one object per level — and this rail builds one object from
  one flat field list. Applying that object to a parent whose field is a whole structure would
  give the nested fields the *parent's* encoding: well-formed octets of the wrong shape. So the
  notation is read and the application is refused by name, which keeps the protection the old
  outright refusal provided.

A structure that is neither the application point nor claimed by anything is refused at `END` —
the first moment every claim has been seen, so a `REPLACE` may still be written after the
structure it names.

**Two misdirected diagnostics, both found by reading §22.11 against the parser.** The next
slice was meant to be §22.11's contained-type notation. Checking the keyword first turned up
that the notation had not been reachable at all, and that the plan for it was wrong.

- **The refusal was keyed on a word X.692 does not use.** `_UNSUPPORTED_KEYWORDS` held
  `CONTAINED`; §22.11.1.2 spells the group `CONTENTS-ENCODING`, and §22.11.1.5 makes that
  keyword the thing the specification is "considered set" by. X.692 uses "contained" only in
  prose ("contained type"). So the careful §22.11 citation could only be produced by writing a
  word no ECN module contains, while the real keyword fell through to the next group's error —
  writing `CONTENTS-ENCODING` on a `#PAD` object complained that `ENCODING-SPACE` is mandatory,
  which points at an unrelated clause. A refusal that fires on the wrong token is not a
  refusal, and its own fixture had been pinning the wrong word too.
- **Seven built-in classes were reported as though they did not exist.** `#BITS`, `#CHARS`,
  `#OCTETS`, `#NUL`, `#TAG`, `#REPETITION` and `#CONDITIONAL-REPETITION` each got "is not a
  built-in encoding class and no assignment defines it" — true for a typo, false for these:
  X.692 defines every one and `ecn_user` implements every one. `_UNREADABLE_CLASSES` names them
  with their clause and their spec, so "you misspelled it" and "this parser cannot spell it
  yet" stop being the same message.

**And the priority was wrong.** §22.11's notation was listed as the highest single remaining
item. It is three slices, in a forced order: §22.11.1.2 hangs the group off §23.2's `#BITS` and
§23.9's `#OCTETS`, and those string classes take their size from §22.7's repetition space rather
than from an `ENCODING-SPACE` — so their `WITH SYNTAX` is written in terms of
`REPETITION-ENCODING(S)`, and §23.14's `#CONDITIONAL-REPETITION` has to be readable first. The
order is: `#CONDITIONAL-REPETITION`, then the string classes, then `CONTENTS-ENCODING`.

**Step one is built: §23.14's `#CONDITIONAL-REPETITION` reads from module text**, at
`SYNTAX_VERSION` 9. It is deliberately read by §23.7's helpers, because §23.14.2.1 repeats
§23.7.2.2's three-list rule verbatim and §23.14.2.2 repeats §23.7.2.4's "at most one of `IF`,
`IF-ALL` and `ELSE`". Two things genuinely differ, and both are the point:

- **The predicate vocabulary is §21.13's, not §21.11's.** They are siblings over different
  quantities — a *size* against an integer's *value* — and §21.11.4's NOTE says its five
  partition where §21.13.4's says only `fixed-size` overlaps. One table for both would let a
  specification select on a predicate that cannot hold, so there are two tables and the
  refusal cites the clause whose vocabulary was expected.
- **`REPETITION-SPACE` replaces `ENCODING-SPACE` rather than varying it.** §21.7.3 says so in
  as many words, so an object carrying both is refused; and §23.14.1 gives `REPETITION-SPACE`
  no brackets, which makes it mandatory on the same reading that makes `ENCODING-SPACE`
  mandatory for `#CONDITIONAL-INT`. §23.14.2.5's flat prohibition on `SIZE fixed-to-max` is
  checked where it is written.

**Step two is built: §23.2's `#BITS` and §23.9's `#OCTETS` read from module text**, at
`SYNTAX_VERSION` 10, and an `#OCTETS` field now encodes end to end from an EDM — a count field
written by §22.7's repetition space, then the octets, with the length prefix stated by no
property of the object itself.

**Neither class has an `ENCODING-SPACE` group, and that is the structural fact.** §23.2.1 and
§23.9.1 give pre-alignment, a start pointer, `VALUE-REVERSAL`, `TRANSFORMS`,
`REPETITION-ENCODING(S)`, a handle and `CONTENTS-ENCODING` — so a string's size comes from the
§22.7 repetition space of the `#CONDITIONAL-REPETITION` objects it names, which is why these
could not be read until §23.14 was, and why one function reads both clauses.

`PendingConditionalRepetition.bind` earned its shape here: the *same* `#CONDITIONAL-REPETITION`
object gives a `#BITS` class 1-bit elements and an `#OCTETS` class 8-bit ones, because
§23.2.2.1 b) and §23.9.2.1 b) put the element on the class rather than in the object. Deferring
it was right rather than merely tidy.

**Step three is built: §22.11's `CONTENTS-ENCODING` reads from module text**, at
`SYNTAX_VERSION` 11, and `_UNSUPPORTED_KEYWORDS` now holds **no unbuilt group at all** — every
row left is a group that *is* built and is refused where a clause forbids the way it was
written.

It needed two things that scoping turned up, both larger than "parse a keyword":

- **§18.1's `EncodingObjectSetAssignment`, because the group takes two `#ENCODINGS`
  references** and this parser had no named object sets — only the single implicit set a
  module forms from its objects. Both of §18.1.1's forms read: §18.1.5's braced union (with
  `|` and `UNION` as the same mark, and §18.1.7's distinct-classes rule enforced) and §18.2.1's
  seven built-in names through `ecn.builtin_object_set`. §18.1.8's `COMPLETED BY` sends the
  braced spec through §13.2 as `PrimaryEncodings`, left-biased per §9.23.2 — the completion
  fills gaps and never overrides. §18.1.2 is what tells an object-set assignment from an
  object one: the notation is "governed by the reserved word `#ENCODINGS`".
- **A modelling decision in `ecn_user`, which slices H2a and H2b deliberately avoided.**
  §22.11's group is in §23.2.1's and §23.9.1's `WITH SYNTAX`, so it lands on the `StringSpec`
  those clauses produce — but contained-type semantics lived on `ContainerSpec`, a different
  class with a stated `width` and no repetition. `StringSpec` gains `contents`, `encoded_by`
  and `contained_class`, and `contained_objects` routes through the *same*
  `ContainedType.select`. Two copies of a five-row table where one row already contradicts
  §22.11.2.2 would be two places for that reading to drift. `ContainerSpec` stays what it
  models: the fixed-width container.

**One deviation, stated.** §22.11.1.3 makes the group's purpose deciding "the encoding of a
contained type", and X.692 takes that type from the ASN.1 `CONTAINING` constraint through
clause 12's link. No ELM section is readable here, so the class is written beside the group as
`CONTAINING <class>` — the same shape and the same reason as `#INT`'s `BOUNDS` and `AUXILIARY`.
Omitting it is refused rather than defaulted: a contained type nobody named is a set applied to
nothing.

**§16.4's `RepetitionStructure` is built**, at `SYNTAX_VERSION` 12 — the third and last of
§16.2.12's `EncodingStructureDefn`s, and the one shaped unlike the other two. §16.3 and §16.5
take a *list* of `NamedField`s; §16.4.1 takes exactly one `EncodingStructure` between its
braces, whose identifier is **optional**, followed by an optional `Size`.

That optional identifier had been load-bearing since slice G1 with no structure able to
exercise it. §17.5.11 makes an `EncodeStructure`'s identifier omitted "if and only if the
governing encoding constructor is a class in the repetition category with no identifier on the
repeated element", and `ecn_encode.EncodeStructure.unnamed_element` has checked that
biconditional against a structure no module could write. It can now.

§16.4.2's `Size` is bounds on the **number of repetitions** rather than on a value, which is
why it lands in `SizeBounds`; §16.2.11 constrains it twice and both are refused — "MIN shall
not be used in `Size`" and the number "shall be non-negative when used in `Size`". It reaches
the digest because §23.2.2.3 has §23.14's `#CONDITIONAL-REPETITION` select on exactly those
bounds.

One lexer change came with it: `(` and `)` are punctuation now. §16.2.10's `Bounds` and `Size`
are the only productions that bracket with parentheses, and they arrived with this clause. `..`
is deliberately *not* punctuation — a bare `.` is a token nothing else in ECN wants, so
`0..MAX` arrives as one word and is split where it is read.

**§16.5.6's per-level application is built**, and the nested notation from slice H1 now
encodes. "The application point then proceeds to each of the `EncodingStructure`s in its named
fields" — one object per level.

The blocker was not the descent but the *pairing*. Before nesting there was one structure and
one constructor object, so "the module's structure" and "the structure this object governs"
were the same question and `require_structure` answered both. With a nested field they part
company: a module has an object for the outer class and one for the inner, and an inner object
handed the outer field list would encode the parent's components in the child's position.
`Structure` now records the class governing it, and `structure_for_class` walks from the
application point to find the level an object belongs to — which also makes the two objects
resolvable in either declaration order, since each reads only its own fields.

A first attempt at this shipped `field_spec` alone and was reverted: it removed the refusal
without fixing the pairing, so the inner object's body tried to resolve the outer structure —
including itself — and failed §9.5.2 with "0 objects in this module". The refusal was worth
more than the half-fix.

One guard needed widening with it. `object_set` refused a second constructor object per class
by testing `class_name in applied`, which was right when no field could carry a constructor
class; a nested field legitimately does, so the loop above now adds the very object the second
loop is about to. §9.5.2 forbids a **second** object for a class, not the same one reached two
ways.

**§23.4's `#CHARS` stayed unreadable, and the reason is stated.** It shares the same
`WITH SYNTAX`, but its repeated element is a *character* whose width comes from the ASN.1
type's character set through clause 12's link, where §23.2's bit and §23.9's octet are
intrinsic to the class. It is also not on the path to §22.11: that group appears in §23.2.1's
and §23.9.1's syntax and in no other class's, so `CONTENTS-ENCODING` is now one slice away, on
classes that exist.

**Two tests this slice invalidated, both rewritten rather than deleted.**
`test_a_class_this_rail_cannot_execute_is_named_rather_than_ignored` used `#BITS` as its
example of an unspellable class; it uses `#CHARS` now, which is the one that is still true of,
and the `#BITS` case became a test that a string class refuses an `ENCODING-SPACE` by name. And
the pinned `_UNREADABLE_CLASSES` set shrank again — the third time that assertion has turned a
table edit into a deliberate one.

**Two decisions worth keeping.** The parser yields a `PendingConditionalRepetition`, not a
`ConditionalRepetitionSpec`: §23.14.1's syntax carries no element, because §23.2.2.1 b) has the
*referencing* class supply it ("the bits are then considered as a repetition of bit"). Inventing
a placeholder element would have spent `ConditionalRepetitionSpec.__post_init__`'s refusal to
make the parser tidier, so `bind` is the seam and the refusal survives. And §22.7.1.1's
`&repetition-space-size` has no slot on `ecn_user.RepetitionSpace` — that class models how a
decoder finds the **end** of a repetition — so the pending record holds it and the digest emits
it. A property read and neither kept nor acted on is a property dropped.

**A stale test this slice found.** `test_a_nested_encoding_structure_is_refused_rather_than_flattened`
asserted `"16.2.1" in str(error)`, and the error it actually got was §16.2.**12**'s — a
different rule about a different fault, matching on a substring. It had been passing for the
wrong reason; the replacement asserts the nesting is read, hashed under its path, and refused
only where it would encode.

**One bug this slice introduced and its own tests caught.** §16.5.2's check for the *marker's*
category reused the variable holding the *structure's*, so a single `OPTIONAL-ENCODING` field
turned its concatenation into an "optional" structure and every later object was rejected
against it. Two different facts about two different things, now with two different names, and a
regression test asserts the structure's category survives a marked component.

**§22.1's `REPLACE` defined syntax is readable now, which closes the ECN notation.** The chain
it completes is three clauses long and took two corrections to find: §22.1.3.5 says a
replacement structure's other fields are "set according to the specification in the
**replacement structure encoding object**"; §17.5.1's `EncodeStructure` is that specification;
and §22.1.2.6 classifies which fields those are — "all fields of the replacement structure that
are not part of the encoding class parameter are auxiliary fields".

So the dummy field is found by **computation** — the one whose class is the structure's single
parameter — and every other field is auxiliary. Same shape as
[`ecn_link.auxiliary_fields`](../bcir/asn1/ecn_link.py), which derives §22.1.2.6's other half
from the link rather than from a declaration.

§22.1.2.1 is what makes the syntax easy to read and easy to get wrong: "exactly one of the
permitted syntaxes between `REPLACE` and `WITH` shall be used", a closed set of five, with
§22.1.1.8 making `COMPONENT` "a synonym for `REPLACE ALL COMPONENTS`" rather than a sixth
action. Both failure directions are refused — none of the words, and two of them.
`ecn_param.bare_use` enforces §22.1.2.2 and §22.1.2.4's shared closing sentence, so a structure
written `#Length-prefixed{<#D>}` where it is defined is `#Length-prefixed` here.

**What is left of clause 22.1, and why.** §22.1.2.7's `INSERT AT HEAD` structure "shall not have
dummy parameters", so it is an *ordinary* encoding structure — and this module already declares
its one ordinary structure as the §13.2 application point. That is a limit of this rail rather
than of the clause, and the two facts are worth separating: §13.2 walks one application point,
but nothing in clause 16 says a module declares only one *structure*. The hoisting semantics
(§22.1.3.6) are built and exercised from Python. §22.1.1.7 e)'s
`REPLACE NON-OPTIONALS ... AND OPTIONALS WITH ...` needs a second replacement group beside
`ConcatenationSpec.replacement`, which is a change to the semantics rather than to the notation.

Two of §21.7's eight repetition-space determinations remain: `flag-to-be-set` and
`flag-to-be-used` put a continuation flag **inside the repeated element** (§21.7.6/§21.7.7),
which needs the element's own structure to reserve a field for it.

**The plan-v6 question, answered.** The open question was whether an ECN encoding is a sixth
column in [`encode_plan`](../bcir/asn1/encode_plan.py), carried by a version 6 of that
descriptor. It is not, and the workload is the evidence: `encode_plan` describes an ASN.1
*type*, and its five emitters read the same node and apply their own rule to it. The frame
header's wire order puts `payloadOctets` first where the plan's members are in schema order,
and its `reserved` bits correspond to no ASN.1 component at all. `EncodeNode` has a slot for
neither, because both are properties of an encoding *structure*. Carrying them would make a
node's meaning depend on which candidate read it — the one thing that plan's design rules out.
So an ECN encoding is a **third compilation** of the same schema, with its own version counter
and its own digest, by the same argument `encode_plan` already makes for why a write plan is
not a read plan. That digest is the practical gain: until now an ECN specification could not be
hashed, compared or named, and two of them could only be diffed as Python source.

The §6 reduction gate fired and was signed off: the fixed
DER/PER/OER/BCIR-canonical-JER candidate set already demonstrates cost-governed
selection, so user-defined ECN classes, `#TRANSFORM`, and `#OUTER` lowering were closed
as not active prerequisites. Reopening them required an approved measured workload **and**
a proof that ordinary BCIR lowering contracts cannot express it.

**Both conditions are now met, and the second one is executed rather than asserted.**
The approval is the project owner's and was given. The proof is not something approval can
supply, so it is built as evidence: `legacy_frame_workload()` is a fixed-layout frame header
with a length field scaled in 4-octet units, a flag that is active low, two reserved bits,
and the length transmitted before the version — every part of which real link-layer and
IP-family headers do. `refuted_by()` then runs all five fixed candidates against the same
abstract value and reports what each produces:

| Candidate | Octets | |
|---|---|---|
| user-defined ECN | `aa00` | 2 |
| DER | `30090201050101ff020128` | 11 |
| CANONICAL-PER-ALIGNED | `ba00` | 2 |
| CANONICAL-PER-UNALIGNED | `ba00` | 2 |
| COER | `05ff28` | 3 |
| CJER | `{"version":5,...}` | 46 |

The sharpest line in that table is canonical PER. It produces **exactly the same number of
octets** and still not the same octets, so the gap is not compactness — at equal size the
fixed set cannot reach the layout, because a layout is not a size question. The reason is
structural: DER, PER, OER and JER all encode *the abstract value*, and given 40 they write
40. A field that transmits 40 as `1010` because it is scaled in 4-octet units is asking for
the encoded value to be a declared function of the abstract one, which is what `#TRANSFORM`
is and what no constraint tightening or canonical-variant choice in the fixed set produces.

A test pins this as the live gate: it **fails if any fixed candidate ever produces the target
octets**, because that is the day the expressiveness argument is false and this section is
wrong.

### H. K_BCIR encoding selection · **CERTIFIED RAIL AND BOTH DECODE TABLES BUILT; TARGET CALIBRATION OPEN**

Encoding rules become a candidate dimension in the optimizer. Given an abstract value,
a target profile H, live Θ, and an RCSP budget, `K_BCIR` selects the encoding rules the
same way it selects a lane width today.

The cost vector already has the right axes: `memory` for wire size, `compute` for
encode/decode work, `compile` for schema processing, `verification` for canonical-form
checking. What each encoding rule needs is a calibrated entry in the cost table —
which is exactly what `kbcir/microbench.py` and the frozen-table machinery are for.

Laws this phase must carry:

- **legality first** — an encoding is a candidate only if the abstract value is
  representable in it, which is a verifier question, never a cost question;
- **two-truth** — a measured encode/decode cost is graded truth and must not become a
  legality verdict (the existing quarantine applies unchanged);
- **canonical or excluded** — a rule with no canonical variant may be decoded but never
  selected for emission, since a selected encoding is a digested artifact.

Gate: on a bandwidth-capped Θ the optimizer selects UNALIGNED PER over DER and the
selection is certified; on a decode-latency-capped Θ it selects OER; with no budget it
reproduces today's DER exactly (the degenerate case, pinning that nothing regresses).

> **The decode-latency gate needed a second table, not a faster clock.** As first written,
> that gate could never pass: it asks the optimizer to select OER, and OER can never hold a
> row in the schema-free decode table — X.696 §6.2 denies it a schema-free decode
> *permanently*, and `CostRow` needs both axes. The blocker was a category error in the
> table, not missing hardware.
>
> `directed_decode_table` is the fix: a **second** table whose `decode` interval is a
> schema-**directed** measurement, built on `bcir_oer.c`'s plan-driven decoder and the
> harness's `dircase` arm. CANONICAL-OER now has the two-axis row it could never have had.
> The two tables are never merged — the schema-free one asks *can untrusted octets be walked
> with no type in hand*, the directed one asks *what does decode cost in deployment, where the
> type is always known* — and `EncodingCostTable.decode_kind` is inside the digest, so a
> certificate bound to one cannot be mistaken for a certificate bound to the other.
> `COST_TABLE_VERSION` moved to 2 for exactly that reason.
>
> The label is also **copied onto the certificate**, next to `provenance` and for the same
> reason: a verdict has to say what kind of truth it read without the reader having to fetch
> the table. It is the one field that can make two otherwise identical certificates — same
> schema, value, target, `cal_gen` and `measured` provenance — name different decode-latency
> winners. `select_certified` records it and deliberately does **not** gate on it: unlike an
> oracle provenance, whose numbers are the wrong kind of evidence for every candidate, both
> decode columns are honest measurements, and the two kinds cannot meet inside one table
> because `decode_kind` is a property of the table rather than of a row. The one residual
> case — a decode-latency objective pointed at a table that does not hold the candidate — is
> already the missing-row refusal's job.
>
> **What the gate can and cannot yet show.** The directed table has **one row**, because
> `bcir_oer.c` is the only plan-driven decoder in the C rail. So "selects OER" is true and
> *vacuous*: there is nothing to select against. Closing it properly needs a plan-driven PER
> decoder — `bcir_per.h` has the bit reader (`get_bits`, `constrained`, `length`) and no
> whole-value decode. That is a **gap, not a law**: X.691 §7.2 bars only the schema-*free*
> decode and says nothing against a schema-directed one. **Priority: high** — it is the one
> remaining piece that turns phase H's second gate from answerable into answered, and it needs
> no hardware.
>
> **Measurement status (`bcir/asn1/selection.py`).** Two of the three hold today. The
> no-budget case reproduces DER exactly, and the bandwidth case selects
> CANONICAL-PER-UNALIGNED at 84 octets against DER's 136 — decided by arithmetic, so it
> holds on every host. **The decode-latency case does not select OER on the Python
> oracle**, and the reason is an implementation artifact rather than a property of the
> rules: JER decodes through `json.loads`, which is C, while COER decodes through pure
> Python. This is recorded rather than papered over precisely because the *two-truth* law
> above forbids promoting a measured cost to a verdict — phase H's real selection reads the
> calibrated table in `kbcir/microbench.py`, and a Python-oracle timing is not that table.

## 5. Sequencing recommendation after PR #670

Phases A–F and **all** of G are landed on their documented rails. Phase H has exact
wire-size evidence, a Python measurement harness, a native measured table
([`native_bench.py`](../bcir/asn1/native_bench.py)) and §6.2's certificate
([`certified.py`](../bcir/asn1/certified.py)); what it still lacks is target hardware
counters and a native *encode* column. The sequence below is recorded as written and is
**complete through step 7** — every item landed as J1–J6:

1. harden the JER oracle with pre-parse limits and byte-exact canonical validation;
2. compile schemas/instructions into deterministic descriptors;
3. build the bounded scalar C JER twin and differential/fuzz rail;
4. add an additive MLIR family/profile representation and extend R24;
5. lower selected schemas directly to claims and StreamPack;
6. measure scalar and optional SIMD implementations on controlled targets; and
7. freeze target tables and issue K_BCIR selection certificates.

The complete dependency and driver boundary are in
[`BCIR_ASN1_JSON_ROADMAP.md`](BCIR_ASN1_JSON_ROADMAP.md). X.681/X.683 are no longer a
sequencing fork: both are already built within their documented subset.

## 5.1 Access limits on the remaining phases

Everything left in H and beyond is blocked on **access** rather than on code, and the limits
are now measured rather than assumed. [`BCIR_TARGET_ACCESS.md`](BCIR_TARGET_ACCESS.md) records
them in full; the short version is that neither available host has a hardware PMU, cpufreq
control, CPU isolation, kernel headers, or any device-binding path.

The session container is the misleading one: it runs as root with `cap_sys_module`,
`cap_sys_rawio` and `cap_perfmon`, and still cannot count a cycle or load a module, because
those capabilities are permissions to use hardware surfaces the hypervisor never exposed.
Granting more privilege would change nothing.

What this does **not** block is worth stating as plainly: the Python oracle, the MLIR law
rail, the C twins, the differential and fuzz gates, the cross-ABI sweep, and all remaining
ECN work run here unimpeded. The blocked set is exactly *measurement* and *device* evidence.

## 6. Stop conditions and decision boundaries

- **Phase A's X.509 stop condition — TRIGGERED, and measured.** §4 A said: *"if the
  parsed subset cannot express X.509 without X.681 information objects, stop and take
  phase F first rather than inventing a dialect."* It half-triggers, and the split is
  clean:

  | RFC 5280 type | Result |
  |---|---|
  | `AuthorityKeyIdentifier` (+ `GeneralName`, `Name`, `RDNSequence`, `DirectoryString`) | **Parses, lowers, and re-encodes 37/37 real trust-store extensions byte-for-byte.** |
  | `SubjectPublicKeyInfo` | **Blocked.** `AlgorithmIdentifier.parameters` is `ANY DEFINED BY algorithm` — withdrawn X.680:1988 notation whose modern spelling is an X.681 open type. |

  The blocker is not marginal: **152 of 152** certificates in the host trust store carry
  `parameters`, so the type cannot be usefully expressed without X.681 at all. The
  front-end refuses `ANY` by name rather than inventing an open-type dialect, which is
  what the stop condition asked for.

  **Consequence for sequencing:** phase **F** (X.681) is what unblocks X.509, not phase
  B or C. F is no longer only "nice for real-world modules" — it is the gate on
  consuming the single most widely deployed ASN.1 schema family there is. See §5.

- **ECN reduction (§4 G) — MEASURED AND SIGNED OFF.** The candidate set is
  {DER, CANONICAL-PER-ALIGNED, CANONICAL-PER-UNALIGNED, COER, **BCIR-canonical JER**} —
  note the fifth: §4 G and this bullet previously said "CJER", but X.697 §42.2 registers
  exactly one object identifier and defines **no canonical variant at all**, so the
  candidate is BCIR's own profile and carries no OID. The measurement now exists
  (`bcir/asn1/selection.py`, recorded by `bcir/tests/test_asn1_selection.py`); on the
  X.691 Annex A.1 record the five legal encodings span **84 to 385 octets, a 4.6× spread**,
  and a bandwidth-capped objective selects UNALIGNED PER at **61.8% of DER** with no
  user-defined encoding class involved. On Annex A.2 the same value falls to 61 octets
  because constraints are PER-visible and, per §7.2.2 l), invisible to JER. That is the
  win the gate asks about. ECN's built-in object sets are sufficient for the current
  thesis; user-defined encoding classes remain closed unless a measured workload proves
  that ordinary BCIR lowering contracts are insufficient.
- **X.694 stays out.** Mapping W3C XML Schema into ASN.1 serves XML interop BCIR has no
  stake in. Revisit only if a concrete consumer appears.
- **XER (X.693) is decode-oriented, and is now built on those terms.** BASIC-XER and
  CXER are in (`bcir/asn1/xer.py`, byte-identical to Annex A.4 and matching A.3's stated
  653 octets); **EXTENDED-XER is not**, and is a deliberate exclusion rather than a gap.
  Clauses 10 and 18-39 are a second language — XER type prefixes, an encoding control
  section, XML namespaces and attributes — and none of it changes a BASIC-XER or CXER
  encoding, so it buys interchange BCIR has no consumer for. XML remains a poor fit for a
  digested artifact and XER is still **not** a selection candidate for phase H.
- **No encoding rule ships without a canonical variant** on the emit path. BER taught
  this lesson already; the rule generalizes.
- **No native fast path ships without a C twin and a dual-rail differential.** JER is
  currently an oracle-only surface, not a promoted native path. The scalar C twin and
  its differential are explicit gates in the JER compilation roadmap.

## 7. Risk register

| Risk | Phase | Mitigation |
|---|---|---|
| PER without constraints is worthless | C | hard-ordered behind B; do not start C early |
| Information objects are needed sooner than F | A | phase A carries an explicit stop condition that promotes F |
| ECN's size (7 599 lines) swamps the program | G | the §6 reduction gate is decided by measurement from C–E |
| Encoding selection becomes a policy knob rather than a law | H | legality-first + two-truth are stated as gate laws, not guidance |
| A new encoding rule quietly replaces a frozen format | all | R24's `additive` requirement already refuses to express it |
| Third-party module parsing pulls in unbounded X.680 surface | A | subset is defined by what `schema.py` can express; anything else is a documented reject |

## 8. What this is not

This roadmap records the ASN.1 portfolio through PR #670; many phases below X.690 are
now implemented on explicitly bounded rails. The inventory in §1 and the per-phase
status labels are the source-backed statement of scope, while `docs/STATUS.md` remains
the generated static inventory. A component is promoted only when its implementation,
positive and negative tests, and required parity gates exist.

## 9. Encoding-rule coverage, and the rules deliberately not built

Written at the close of the ECN build-out, in answer to "how much of ASN.1 does BCIR
actually cover, and what is left". Two things make this table readable:

**The in/out asymmetry is a design, not an accident.** Every family accepts the *permissive*
variant and emits the *canonical* one — BER in / DER out, BASIC-PER in / CANONICAL-PER out,
BASIC-XER in / CXER out, BASIC-OER in / COER out, JER in / BCIR-canonical JER out. That is
Postel's rule with the safety inverted: liberal at the trust boundary, strict at the digest,
because §0's end state is *a provably legal wire format with a certificate*, and a certificate
over a non-canonical encoding names an octet string that two conforming encoders could both
produce differently.

**"Coverage" here means clauses honoured, not features demoed.** Every percentage below is
backed by the conformance corpus and the dual-rail C parity gates, and every exclusion is a
sentence in a standard rather than a preference.

### 9.1 Built

| Rule | Standard | Direction | Coverage | What remains, and its priority |
|---|---|---|---|---|
| **BER** | X.690 | in | complete for the supported type surface | — |
| **DER** | X.690 | **out** | complete | — |
| **BASIC-PER** aligned + unaligned | X.691 | in | complete; validated against Annex A.1–A.4 | — |
| **CANONICAL-PER** aligned + unaligned | X.691 | **out** | complete | — |
| **BASIC-OER** | X.696 | in | complete; validated against Annex A | — |
| **CANONICAL-OER** | X.696 | **out** | complete, both rails; `bcir_oer.c` is the schema-directed C twin | — |
| **BASIC-XER** | X.693 | in | complete; Annex A.3/A.4 | — |
| **CXER** | X.693 | **out** | complete | — |
| **JER** | X.697 | in | clauses 20–41, §7.2 constraint visibility, cl. 14–19 encoding instructions with cl. 13 precedence | cl. 10–11 assignment syntax (X.680 surface) — **low**, it is notation not encoding |
| **BCIR canonical JER** | *BCIR-private* | **out** | complete; versioned, carries **no** standards OID | — |
| **ECN** | X.692 | both | cl. 9–25 model, EDM/ELM, the seven built-in object sets, the user-defined half, multi-structure modules with §22.1.2.7's `INSERT AT HEAD`, and clause 20's defined syntax read from module text | three named refusals (§G) — **low to medium**; §22.11's contained-type notation, §18.1's encoding object sets, §16.4's `RepetitionStructure` and §16.5.6's per-level application are all **built**. What remains: §22.1.1.7 e)'s second replacement group, §21.7.6/§21.7.7's continuation flag, and R25's Annex C parameterization coverage |

### 9.2 Excluded by design, with the sentence that excludes them

| Rule | Standard | Recommendation | Justification |
|---|---|---|---|
| **CER** | X.690 | **exclude** | CER and DER are both canonical subsets of BER differing in one axis: CER uses indefinite-length constructed form for large values, DER definite. DER is the one security-critical infrastructure actually uses (X.509, CMS, PKCS). Emitting both canonical forms would give one abstract value two certified spellings with no selection criterion to choose between them — which is the *opposite* of what a digest is for. BER-in already accepts anything a CER encoder produces. |
| **EXTENDED-XER** | X.693 cl. 8+ | **exclude** | E-XER exists to make ASN.1 emit an author-chosen XML vocabulary — attributes, element renaming, list forms, arbitrary namespaces. It is a *presentation* mapping, and its output is XML whose shape is chosen by encoding instructions rather than derived from the type. BCIR selects encodings by cost against a budget; E-XER's degrees of freedom are aesthetic, so there is nothing to optimize and a great deal to verify. BASIC-XER + CXER already cover the machine-readable case. |
| **GSER** | IETF RFC 3641 (+ 3642, 4792) | **exclude** | RFC 3641 states GSER "does not necessarily enable the exact octet encoding of certain string types to be reconstructed" and "MUST NOT be used to re-encode values whose original binary encoding must be recoverable". BCIR's *canonical-or-excluded* law forbids selecting an encoding for emission when the encoding is not canonical; a rule that cannot round-trip its own input cannot be a certified artifact. GSER is a human-readable directory-debugging format and BCIR already has one of those in JER. |
| **RXER** | IETF RFC 4910/4911 | **exclude** | **Experimental** status, published 2007, no deployed base BCIR targets, and its purpose — robustness against unknown XML elements in an XML-enabled directory — is a schema-evolution story that ASN.1 extensibility markers plus X.691 Annex A.3/A.4 already handle on the rails that are built. |
| **BACnet encoding** | ASHRAE 135 | **exclude** (revisit only with a workload) | BACnet's application-layer encoding is ASN.1-*flavoured* — it borrows tag/length/value shape — but it is defined by ASHRAE, not ITU-T, and its tagging is not X.690's. Supporting it is writing a *new* codec against a building-automation standard, not covering more of ASN.1. Reopen only under the same gate ECN's user-defined half had to pass: a written workload, a missing-expressiveness proof, and approval. |
| **CSN.1** | 3GPP (GSM/GPRS) | **exclude** | CSN.1 is a **different abstract notation**, not an encoding rule for ASN.1. It has its own syntax and its own semantics for bit-level layout. The honest comparison is that CSN.1 is a peer of ASN.1, and the ASN.1 answer to CSN.1's problem domain is **ECN** — which is built. A CSN.1 front-end would be a second language, and §8's "what this is not" applies. |
| **"SER"** | *unidentified* | **cannot recommend as stated** | No ASN.1 encoding rule by that abbreviation was found in the ITU-T X.69x series or the IETF RFC corpus. If this means *Signalling*, *Simple*, or *Streaming* Encoding Rules, or a vendor-specific rule, the name needs resolving before a recommendation means anything. Recorded as unresolved rather than guessed at. |

### 9.3 Custom BCIR high-speed coding rules

**Already built, and it is ECN.** This is the one worth stating plainly, because "a custom
high-speed rule" sounds like something still to do and it is not. §G's gate produced a
fixed-layout frame header — length scaled in 4-octet units, active-low flag, reserved bits,
length transmitted before the field it measures — and **none of the five fixed candidates
reproduces those octets**. Canonical PER is the instructive failure: it lands on the *same
octet count* and different octets, so the gap is expressiveness, not compactness.

X.692 is exactly the standardized mechanism for "define your own rule", so a BCIR-private
binary rule would be re-inventing ECN with no standards traceability and no ELM. The
BCIR-specific parts that *are* private — the canonical JER profile, StreamPack framing, the
K_BCIR selection certificate — are private because no standard defines them, and each says so
and carries no OID.

**Recommendation: no new private binary rule.** If a workload needs one, express it as an ECN
encoding-definition module; that is what §22.1's `REPLACE`, clause 24's transforms and
`#OUTER` exist for, and it stays hashable and reviewable as a specification.

## 10. Information objects beyond X.681, and the ASN.1 bindings

### 10.1 What is already built, so the question is narrower than it looks

X.681's classes, `WITH SYNTAX`, objects, object sets and **clause 13 associated tables** are
built, as are X.682's table and component-relation constraints. That combination is the
working half of information objects: a `CLASS` with fields, an object set whose rows are
reachable, and a constraint that ties a component to a row of it. The open-type machinery that
makes `TYPE-IDENTIFIER`-style dispatch work is therefore already load-bearing on the PER and
BER rails, which is why the roadmap has never listed IOC as a separate phase.

**So "generalized IO / IOS" is not a missing feature; it is X.683 parameterization applied to
objects** — the one X.681 exclusion still recorded in the standards table. That exclusion is
now inconsistent with what shipped: Annex C's parameterization is built for **ECN** objects
(`ecn_param.py`), and the ASN.1 side has parameterized type/object/object-set assignments in
the front-end. What is genuinely absent is the *cross-product*: a parameterized object set
whose actual parameter is itself an object set, used as a table constraint.

**Priority: medium.** It is the natural completion of X.681/X.683 and needs no new standard —
but no workload in this repository has asked for it, and the last time a clause was built
without a workload the §6 gate refused it. **Recommendation: build when a workload names it,
and record the gate the way ECN's was recorded.**

### 10.2 SQL, IDL, CORBA, IIOP — one recommendation, three reasons

**Recommendation: exclude all four from BCIR.** They are not ASN.1 coverage, and treating them
as the next step would change what BCIR is.

- **SQL** and ASN.1 meet only through X.681's *associated tables* (cl. 13), which are already
  built. An actual SQL binding means a relational schema mapping, a query surface and a type
  system with three-valued logic — a database project wearing an ASN.1 hat. The `channel.json`
  and `DeviceManifest` schemas already show the shape BCIR wants for structured configuration,
  and they are ASN.1 all the way down.
- **CORBA IDL** has a standardized ASN.1 relationship (ITU-T X.892 / the ASN.1-IDL mapping),
  so this is not an unreasonable question — but IDL is an *interface* description language.
  BCIR describes values and their encodings, not remote operations. The interface layer BCIR
  does have is the lowering contract, and it is deliberately not an ORB.
- **IIOP** is CORBA's *transport* (GIOP over TCP), with its own CDR encoding. Supporting it
  means implementing CDR — a fourteenth encoding rule, from a different family, whose alignment
  rules resemble PER's without being PER's. §9.2's reasoning against BACnet applies verbatim:
  this is writing a new codec against a non-ITU-T standard, not covering more of ASN.1.

The honest summary: **X.681 gives ASN.1 an object model; it does not make ASN.1 a middleware.**
BCIR's differentiator is cost-governed encoding *selection*, and none of these four has a
selection question in it.

### 10.3 TTCN-3 as a testing language: **exclude, and the reason is structural**

TTCN-3 (ETSI ES 201 873) is a real conformance-testing language with a standardized ASN.1
integration — Part 7 maps ASN.1 types and values into TTCN-3, so the question is fair.

**It is still the wrong tool for this repository, because BCIR's testing thesis is different
in kind.** TTCN-3 exists to test an *implementation under test* across an interface: you write
abstract test cases, an adapter binds them to a real system, and a verdict comes back. BCIR
does not have an IUT across an interface — it has **two rails that must agree**, and the way it
establishes correctness is differential:

- the Python oracle and the C twin encode/decode the same value and are compared **octet for
  octet**, not verdict for verdict;
- the MLIR law rail refuses statically what neither rail should ever be asked to do;
- the fuzz targets and the `-O0 == -O3` gates cover the space TTCN-3 test cases would sample.

A TTCN-3 layer would restate in a second language properties the differential rail already
proves more strongly — an octet comparison is stronger evidence than a pass verdict — while
adding a compiler, a runtime and an adapter to maintain. **Where TTCN-3 would genuinely help is
the case BCIR does not have: certifying someone else's ASN.1 implementation against a
specification.** If that ever becomes a goal, this decision should be revisited on that
evidence, and the ETSI Part 7 mapping is where it starts.

**Recommendation: exclude, with the reopening condition stated** — a third-party implementation
BCIR must certify across an interface it does not own.
