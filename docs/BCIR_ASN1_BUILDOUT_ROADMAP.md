# BCIR ASN.1 build-out roadmap

Where the ASN.1 rail goes after X.690. This is the portfolio document for the ASN.1
program: dependency order, promotion gates, stop conditions, and the one idea that
makes the whole thing more than a second codec.

Companion documents: [`BCIR_ASN1_X690_ABI.md`](BCIR_ASN1_X690_ABI.md) is the normative
contract for what is already built; [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md)
owns cross-program ordering.

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
| X.680 | 8824-1:2021 | Basic notation | **partial** — tag assignments consumed; notation not parsed |
| X.681 | 8824-2:2021 | Information object specification | **partial** — classes, objects, object sets, open types; table constraints recorded not resolved |
| X.682 | 8824-3:2021 | Constraint specification | **partial** — X.680 cl. 49–51 subtype constraints built; X.682's table/user-defined constraints not |
| X.683 | 8824-4:2021 | Parameterization | not started |
| X.690 | 8825-1:2021 | BER / CER / DER | **built** (DER out, BER in; CER by design excluded) |
| X.691 | 8825-2:2021 | PER | not started |
| X.692 | 8825-3:2021 | ECN | not started |
| X.693 | 8825-4:2021 | XER | not started |
| X.694 | 8825-5:2021 | Mapping W3C XML Schema into ASN.1 | out of scope (see §7) |
| X.695 | 8825-6 | Registration of PER encoding instructions | follows X.691 |
| X.696 | 8825-7:2021 | OER | **built** (COER out, BASIC-OER in; validated against Annex A) |
| X.697 | 8825-8:2021 | JER | not started |

Sizes, as a rough effort signal (converted spec text, lines): X.681 1 125 · X.682 345 ·
X.683 567 · X.691 2 733 · X.692 **7 599** · X.693 2 562. ECN is the largest document in
the suite by a factor of three, and §0 is why it is nevertheless the destination.

## 2. What is built (baseline through PR #660 plus the artifact projection)

- The whole of X.690 clause 8 on the oracle rail (`bcir/asn1/`, ~2 160 lines), clauses
  10 + 11 as a checker and a BER→DER rewrite.
- A freestanding, allocation-free, non-recursive C twin (`runtime/c/bcir_asn1.{h,c}`),
  fuzzed under ASan/UBSan, dual-rail differentialed over 12 000 mutants.
- The `BCIR-StreamPack` module and its **additive** DER projection, with the A1–A4 laws.
- The `BCIR-ArtifactBundle` module and additive DER/COER projection. It preserves the
  complete abstract BCAB directory and payloads while native offsets, padding, CRCs, and
  digests are recomputed; native→DER/COER→native is byte-identical. This is the second
  artifact family to prove that the encoding-rule rail is reusable rather than
  StreamPack-specific.
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

### B. X.682 constraints · **BUILT** (the X.680 subtype half) · unblocks PER

> **Status: delivered.** `bcir/asn1/constraints.py` — single values, value ranges
> with open endpoints and MIN/MAX, SIZE, FROM, and the UNION/INTERSECTION/EXCEPT
> composition of clause 49 — plus the *effective* value/size constraint
> (X.696 §8.2.7/§8.2.8), which is what an encoding is chosen from. OER now narrows:
> `INTEGER (0..255)` is one octet where unconstrained INTEGER is two, and a
> fixed-SIZE OCTET STRING or known-multiplier string drops its length determinant
> entirely. R24 gained three diagnostics for an empty value set.
>
> Note the split the roadmap glossed: the *notation* phase B needs is **X.680**
> clauses 49–51, not X.682. X.682 itself is table, component-relation and
> user-defined constraints, which stay with phase F.

Size, value-range, permitted-alphabet, `SIZE`, `FROM`, inner-subtyping, and the
extensibility marker. Table and component-relation constraints (X.682 §10) need X.681
and belong to phase F.

The BCIR-specific payoff arrives immediately and independently of PER: **a constrained
ASN.1 type is a claim geometry.** `INTEGER (0..255)` is an 8-bit lane; `SEQUENCE
(SIZE(1..64)) OF` is a bounded extent. Constraints are exactly the information
`realize.candidates_for` needs to price a decode, so this phase is where the ASN.1 rail
starts feeding the optimizer rather than sitting beside it.

Gate: R24 extends to reject a constraint that is unsatisfiable (empty value set) — a
static fault in the same family as duplicate component tags.

### C. X.691 PER (ALIGNED + UNALIGNED)

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


Octet Encoding Rules: octet-aligned, no bit-shifting, designed for fast encode/decode
rather than minimum size. OER is the encoding rule with the *best decode cost* in the
suite, which makes it the natural default for a driver-side or DMA-fed path.

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

### E. X.697 JER · independent

JSON Encoding Rules. Low implementation cost, disproportionate integration value: it
makes every BCIR artifact readable by tooling that will never speak ASN.1, and X.697
explicitly supports using ASN.1 as a *schema language for JSON*.

Interaction with the existing rails worth noting: BCIR already emits JSON in several
places (telemetry export, performance reports, model plans). JER would give those a
schema with a canonical binary projection, rather than each being an ad-hoc shape.

Canonical variant: CJER, for the digest discipline.

### F. X.681 information objects + X.683 parameterization · **X.681 PARTIAL, the X.509 blocker cleared**

> **Status: the open type is built.** `OpenType` in `bcir/asn1/schema.py` (DER) and
> `bcir/asn1/oer.py` §30 (OER); the front-end parses X.681 §9 class definitions, §11
> objects, §12 object sets, `CLASS.&field` references, and the withdrawn-but-ubiquitous
> `ANY DEFINED BY` spelling. **The X.509 gate passes: 152/152 real certificates.**
>
> What is deliberately *not* built: X.682's table constraints are recorded, not resolved,
> so an open type stays open rather than being narrowed to the type an object set implies.
> That is honest rather than lossy — the octets are carried through untouched, which is
> exactly the open-type contract — and it is what a later phase would add. X.683
> parameterization is still refused by name.

ASN.1's own generics and open-type machinery. `CLASS`, `WITH SYNTAX`, information
object sets, and the `&Type` / `&id` field references that make `ALGORITHM-IDENTIFIER`
and the whole X.509 algorithm-agility pattern expressible.

This is the phase that unlocks *real-world* modules. X.509, LDAP, SNMP, 3GPP and
S1AP/NGAP all use information object classes; a front-end without them can parse toy
modules only. It also completes X.682 (table and component-relation constraints).

Sequencing note: this phase is placed after PER/OER deliberately, because information
objects are only worth the complexity once there is a reason to consume third-party
modules at scale. If phase A's stop condition fires, it moves ahead of B.

### G. X.692 ECN · the metaprogramming layer

The Encoding Control Notation: a language for **defining encodings**, in which encoding
classes are declared, encoding objects realize them, and encoding object sets are
applied to ASN.1 types to determine their wire form.

ECN is where the §0 correspondence stops being an analogy and becomes an
implementation. An ECN encoding object set *is* a plan; applying one to a type *is*
realization. BCIR's contribution is that ECN never specified how to *choose* among
legal object sets — it is a notation, not an optimizer — and BCIR has the optimizer.

Deliverables: the ECN class/object/object-set model in the IR (a natural extension of
`bcir.asn1.*`), the built-in BER and PER object sets, `#TRANSFORM` and `#OUTER` mapped
onto BCIR lowering contracts, and R24 extended with ECN's own well-formedness rules
(one object per class per set, X.692 §9.5.2 — a static law of exactly the R24 kind).

Stop condition, stated up front: ECN is 7 599 lines of specification and is not widely
deployed. If phases C–E show that cost-governed selection over a *fixed* candidate set
(DER/PER/OER/JER) already delivers the win, **ECN's user-defined encodings are not
required for the thesis** and this phase should be cut to the built-in object sets
only. That decision is a gate, not a preference — see §6.

### H. K_BCIR encoding selection · the thesis

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

## 5. Sequencing recommendation

Three phases can run in parallel because they share no dependency: **A** (front-end),
**D** (OER + native fast path), **E** (JER). Of those, **A is the one to start**, since
every later phase consumes parsed modules and A is the only phase that is a hard
prerequisite for the rest.

Suggested order: **A → D → B → C → E → F → G(reduced) → H**, with D and E parallel to
A/B if there is capacity. H is reachable after C+D, before F/G — the thesis does not
need ECN to be demonstrated over a fixed candidate set, which is precisely why §4 G
carries a cut condition.

**Revision after building A.** A is done, and its stop condition fired with a result
that changes what F is for (§6). The order above is still right *if the goal is the
BCIR thesis*: B → C → H is the path to cost-governed encoding selection, and none of it
needs X.681. But if the goal is **consuming real-world schemas**, F moves to the front —
X.509, LDAP, SNMP, and the 3GPP families all use information object classes, and A
cannot express any of them. The two orders are:

| Goal | Order |
|---|---|
| Demonstrate the thesis (cost-governed encoding selection) | **A → D → B → C → H**, F/G after |
| Consume real-world schemas | **A → F → D → B → C → H** |

This is a genuine fork, not a detail — it is worth an explicit decision rather than a
default.

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

- **ECN reduction (§4 G).** If cost-governed selection over the fixed set
  {DER, CANONICAL-PER-ALIGNED, CANONICAL-PER-UNALIGNED, COER, CJER} demonstrates the
  win, build ECN's built-in object sets only and do not implement user-defined encoding
  classes. Record the decision with the measurement that justified it.
- **X.694 stays out.** Mapping W3C XML Schema into ASN.1 serves XML interop BCIR has no
  stake in. Revisit only if a concrete consumer appears.
- **XER (X.693) is decode-oriented.** BASIC-XER and CXER are worth having for
  interchange, but XML is a poor fit for a digested artifact and XER should never be a
  selection candidate. Build it if a peer requires it, not on principle.
- **No encoding rule ships without a canonical variant** on the emit path. BER taught
  this lesson already; the rule generalizes.
- **No phase ships without a C twin and a dual-rail differential.** Phase 1 found three
  parser differentials in the C decoder that way; that is the discipline that found
  them, not a formality.

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

This roadmap does not claim any of the phases below X.690 are started. The inventory in
§1 is the truth; `docs/STATUS.md` remains the generated record of what is executed. A
phase is complete when its laws are gated in CI, not when its code exists.
