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
| X.681 | 8824-2:2021 | Information object specification | **built** — classes, WITH SYNTAX, objects, object sets, associated tables (cl. 13); X.683 parameterized objects excluded |
| X.682 | 8824-3:2021 | Constraint specification | **built** — table + component relation (cl. 10), user-defined (cl. 9), contents (cl. 11) |
| X.683 | 8824-4:2021 | Parameterization | **built** — parameterized type/object/object-set assignments and references; cross-module tag-default nuance (§9.8) excluded |
| X.690 | 8825-1:2021 | BER / CER / DER | **built** (DER out, BER in; CER by design excluded) |
| X.691 | 8825-2:2021 | PER | **built** (CANONICAL-PER out, BASIC-PER in; both variants; validated against Annex A.1–A.4) |
| X.692 | 8825-3:2021 | ECN | **parts 1, 2 and 3 built** — class/object/object-set model (cl. 9-18), EDM/ELM, the seven built-in BER/PER object sets; and [`ecn_user.py`](../bcir/asn1/ecn_user.py) for the user-defined half (cl. 19-25): bit-level encoding spaces, justification, `#PAD`, stated transmission order, `INT-TO-INT`/`INT-TO-BITS` `#TRANSFORM`s and `#OUTER`. The §6 gate's reopening condition is **met and executed** — see section G. Part 3 adds [`ecn_syntax.py`](../bcir/asn1/ecn_syntax.py): clause 20's defined syntax read from an `ENCODING-DEFINITIONS` module, with [`BCIR-FrameHeader.ecn`](../bcir/asn1/BCIR-FrameHeader.ecn) reproducing the gate's octets from text, and a canonical serialization so an ECN specification can finally be hashed. §21.3/§22.3/§22.8's determinants, §21.11's range conditions, §22.12's bit reversal and §22.1's replacement semantics are all built, and ECN is on the law rail as **R25** (`bcir.ecn.*`, twenty-five statically decidable X.692 rules). Clause 24's nineteen transforms, §22.7's repetition, the string/null/tag categories, and the constructor categories (§23.1 alternatives, §23.11 optionality, §22.9 identification handles, §22.5/§22.6 determination, §22.10 concatenation order) are all built. §22.11's contained types and §21.3.6/§21.5.6/§21.7.8's `container` determination are built too, clause 19's six value mappings are in [`ecn_mapping.py`](../bcir/asn1/ecn_mapping.py), and clause 12's encoding link module with clause 13's application-point algorithm is in [`ecn_link.py`](../bcir/asn1/ecn_link.py) — which retires the `AUXILIARY` and `BOUNDS` stated deviations by deriving both from the link rather than declaring them. Annex C's parameterization — X.683 as ECN rewrites it, `{<`/`>}` delimiters and all — is in [`ecn_param.py`](../bcir/asn1/ecn_param.py), together with §22.1.2's rules on the definitions a `REPLACE` names and §17.5.17's breadth-first `ComponentIdList` scan; C.2's three parameterized assignments now parse from module text and reach the digest at `SYNTAX_VERSION` 5. Still refused: §22.1's `REPLACE` defined syntax (§22.1.2.6's auxiliary-field binding is what remains), §16.3/§16.5's constructor *structure* notation, and §21.7.6/§21.7.7's per-element continuation flag |
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

### E. X.697 JER · **PYTHON ORACLE BUILT; NATIVE COMPILATION OPEN**

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
described as CJER. The C scalar parser, generated schema descriptors, MLIR
family/profile representation, direct claim lowering, optional hosted SIMD scanner,
and native K_BCIR measurements remain open in
[`BCIR_ASN1_JSON_ROADMAP.md`](BCIR_ASN1_JSON_ROADMAP.md).

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
eleven statically decidable X.692 rules over them. R25 exists for a sharper version of R24's
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

So the order for the remainder is fixed by that example rather than chosen: §17.5.1's
`EncodeStructure` (with §17.5.3's checkable rule — if `STRUCTURED WITH` is absent then
`CombinedEncodings` "shall be present ... otherwise the ECN specification is in error"), then
§16.3's `AlternativesStructure` and §16.5's `OPTIONAL-ENCODING` marker, which the same example
uses and which need `EcnModule` to hold a structure *tree* rather than the one flat
concatenation it models today, and only then §22.1's `REPLACE` defined syntax on top of both.

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

### H. K_BCIR encoding selection · **MEASUREMENT HARNESS BUILT; CERTIFIED RAIL OPEN**

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

Phases A–F and the reduced, built-in part of G are landed on their documented rails.
Phase H has exact wire-size evidence and a Python measurement harness, but not native
target calibration or a K_BCIR certificate. The next sequence is therefore:

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
