# BCIR's ASN.1 toolchain against ffasn1, asn1c and asn1scc

A comparison of BCIR's ASN.1 build with three established compilers, and a ranked list of what
is worth importing from each.

The three are not interchangeable reference points. **asn1c** (Lev Walkin) and **asn1scc**
(European Space Agency) are code generators: ASN.1 in, C/Ada/Scala source out, compiled into
the application. **ffasn1dump** is the free demo binary of Fabrice Bellard's FFASN1 compiler —
the generator itself is not in the distribution, so what can be examined is a transcoding tool
and its documentation. BCIR's ASN.1 is a fourth thing: an interpreting oracle in Python, a set
of hand-written freestanding C twins, and a law rail in MLIR that has to agree with both.

That difference decides which gaps below are defects and which are category differences. A
missing `asn1c -P` in BCIR is not a hole in BCIR's ASN.1; a missing `COMPONENTS OF` is.

## 1. What each one is

| | BCIR | ffasn1dump | asn1c | asn1scc |
|---|---|---|---|---|
| Kind | interpreting oracle + hand-written C twins + MLIR law rail | transcoder (demo of a C generator) | C code generator | C / Ada / Scala code generator |
| Source available | yes | **no** — `ffasn1dump.exe` only | yes | yes |
| Implementation | Python + C + MLIR/C++ | — | C | F# (.NET), StringTemplate backends |
| Primary output | a *decision* (which encoding, with a certificate) plus decoded values | a re-encoded message | `.c`/`.h` per ASN.1 type | `.c`/`.h`, `.adb`/`.ads`, `.scala` per type |
| Origin/user | BCIR's certified-artifact pipeline | Bellard's commercial FFASN1 | telecom/protocol stacks | ESA flight + ground software |

## 2. Transfer syntaxes

| Recommendation | BCIR | ffasn1 | asn1c | asn1scc |
|---|---|---|---|---|
| X.690 BER | decode | decode | decode | encode + decode |
| X.690 DER | encode + decode | encode + decode | encode + decode | — |
| X.690 CER | **excluded by design** | — | decode only | — |
| X.691 UNALIGNED PER | encode + decode, basic + canonical | encode + decode | encode + decode, basic + canonical | encode + decode |
| X.691 ALIGNED PER | encode + decode, basic + canonical | encode + decode | **not supported** | — |
| X.696 OER / COER | encode + decode | encode + decode | encode + decode | — |
| X.693 XER / CXER | encode + decode | encode + decode | encode + decode | encode + decode |
| X.697 JER | encode + decode | encode + decode | — | — |
| RFC 3641 GSER | — | encode + decode | — | — |
| X.692 ECN | **all three parts** | — | — | — (has ACN instead) |
| ACN | — | — | — | encode + decode |

BCIR's CER exclusion is a recorded decision, not an omission: CER and DER are both canonical
subsets of BER, and emitting both would give one abstract value two certified spellings with no
criterion to choose between them. BER-in already accepts anything a CER encoder produces.

BCIR, ffasn1 and asn1c have no BER-specific *encoder* — all three emit DER, which is valid
BER. asn1scc's is a genuine BER encoder: `BerEncodeLengthStart`/`BerEncodeLengthEnd` write
X.690 §8.1.3.6's indefinite-length form, which DER forbids.

**BCIR is the only one of the four with X.692 ECN**, and the only one with aligned PER *and*
JER *and* OER together. asn1c's aligned-PER gap is notable given how much of the telecom
corpus it is used on.

## 3. Schema language coverage

| Feature | BCIR | ffasn1 | asn1c | asn1scc |
|---|---|---|---|---|
| X.680 core, AUTOMATIC TAGS, nested SET OF, REAL, TIME/DURATION, EXTERNAL | yes | yes | yes | partial (embedded subset) |
| X.681 information object classes / object sets | yes | yes | yes | — |
| X.682 table + component-relation constraints | yes | open types only | yes | — |
| X.683 parameterization | yes | yes | yes | — |
| `COMPONENTS OF` | **refused** | yes | yes (`pull_components_of`) | yes |
| Selection type (`field < Type`) | **refused** | yes | yes | — |
| `WITH SUCCESSORS` / `WITH DESCENDANTS` imports | **refused** | yes | yes | — |
| `CONTAINING` honoured on decode | XER + JER | XER + JER + GSER (`-c`) | — | — |
| X.208 `ANY` | — | yes | yes | — |

The three refusals were confirmed by feeding each construct to `compile_module` and reading the
diagnostic, not inferred from the source. `COMPONENTS OF` reports that it "requires
component-list inlining, which this front-end does not implement"; the selection type parses as
far as "expected a type, found 'a'"; `WITH SUCCESSORS` fails at "expected 'FROM', found 'WITH'".

## 4. What BCIR has that none of the other three do

Worth stating before the gap list, because several of the gaps below are only worth closing if
they do not cost these.

* **X.692 ECN, all three parts** — the model and built-in object sets, the user-defined
  encodings, and clause 20's defined syntax, with clause 24's transforms built out to 17
  transform classes. asn1scc has ACN, which is the same *idea* under a different, non-standard
  notation; asn1c and ffasn1 have nothing in this space.
* **Two decode tables that are never merged.** X.691 §7.2 and X.696 §6.2 deny PER and OER a
  schema-free decode permanently. BCIR keeps schema-free and schema-directed measurements in
  separate tables and records which one a number came from (`decode_kind`). The other three do
  not measure at all, so the question does not arise for them — but any tool that added
  measurement without this distinction would let a rule that cannot be walked without a schema
  appear to compete with one that can.
* **Certified, cost-governed encoding selection.** BCIR *chooses* an encoding under a measured
  cost model and emits a certificate. The other three encode what you tell them to.
* **The law rail.** R1–R25 in MLIR, verified independently of the Python oracle and the C
  twins. Three implementations that must agree, where the others have one.
* **Canonical-or-excluded.** A rule with no canonical variant may be decoded but never selected
  for emission, because a selected encoding becomes a digested artifact.

## 5. What the other three do that BCIR does not

Ordered by how much the absence actually costs.

### 5.1 A real-world grammar corpus (asn1c, asn1scc)

BCIR's ASN.1 test corpus is two PKIX modules (`PKIX1Explicit88.asn1`, `PKIX1Implicit88.asn1`)
plus BCIR's own two schemas. asn1c carries **165 compiler test grammars** and eleven real
protocol modules — rfc3280, PKIX1, LTE-RRC, RRC, J2735, TAP3, MEGACO, LDAP3, 1609.2, ULP,
MHEG5. asn1scc's `v4Tests` holds **803 ASN.1 grammars and 439 ACN files**, and the test run
compiles, links, executes and coverage-checks generated code for all of them.

This is the largest single gap, and it is architecture-independent: BCIR's front end can be
pointed at those grammars today. Real protocol modules are where front ends break — deeply
nested information object classes, tag conflicts across imports, constraint expressions nobody
writes by hand. Two PKIX files do not exercise that.

### 5.2 Random value generation (ffasn1, asn1c)

ffasn1's `-I random` takes a seed and produces a valid message of any type in the module.
asn1c ships `asn_random_fill.c`, a per-type random value constructor used for round-trip
testing of generated code.

BCIR has fifteen `fuzz_*.c` harnesses, all of which fuzz the *decoder* over arbitrary octets.
Nothing generates a *valid abstract value* from a schema. That is the missing half: schema-driven
value generation feeds encode → decode → compare across all three rails, which is exactly the
differential test BCIR's triple-rail design exists to support and currently cannot seed
automatically.

### 5.3 A security regression corpus (asn1scc)

`v4Tests/security-regression` holds three directories, one per fixed memory-safety bug: an XER
primitive decoder buffer overflow, an XER `SEQUENCE OF` out-of-bounds write, and an ACN
null-terminated-string stack overflow. Each carries a README naming the issue, the minimal
grammar, the malicious input, and a `reproduce_issue.sh` that regenerates the decoder and
proves it now fails safely.

BCIR's recent sweep fixed four defects of exactly this class in `bcir_per_plan.c` and
`bcir_asn1_bench.c` — two memory-safety, one silent corruption, one false rejection. They are
covered by tests, but as ordinary conformance tests. Nothing in the tree says "this input was
once a memory-safety bug, here is the octet string that triggered it". That framing is what
makes a regression corpus readable to someone auditing the codec three years from now.

### 5.4 Interface Control Document generation (asn1scc)

asn1scc's `StgVarious` backend emits ICDs — human-readable documents laying out, field by
field, the exact bit positions a message occupies under uPER or ACN, in HTML, XML and JSON.
`v4Tests/icd-tests` pins six of them.

BCIR has no equivalent. It knows the layout precisely — the plan format *is* a bit layout — but
nothing renders it for a human. For a project whose central claim is that an encoding choice is
certified and auditable, "here is the bit layout that was certified" is a document the
certificate implies but does not produce.

### 5.5 ACN features X.692 does not have (asn1scc)

ACN is a competing custom-binary-encoding notation, and several of its constructs have no ECN
counterpart. Confirmed absent from BCIR by grep across `bcir/asn1/*.py` and the ECN ODS —
the CRC hits in the tree are all BCIR's own container framing (artifact bundle, streampack,
JER frames), not user-declarable hooks:

* **`present-when` with a boolean expression language.** ACN lets a field's presence depend on
  an arbitrary predicate over other fields: `enm [present-when (int1 <10 and int1%2 == 0) or
  (int1>=10 and int1 <=14)]`, with `and`/`or`/`not`, six comparisons and `+ - * / %`. X.692
  §21.5's five `OptionalityDetermination` values (`field-to-be-set`, `field-to-be-used`,
  `container`, `handle`, `pointer`) let presence depend on a *determinant field*, never on an
  expression. BCIR implements all five faithfully; the expression form is simply not in the
  Recommendation.
* **`post-encoding-function` / `post-decoding-validator`.** A named hook that runs over the
  finished encoding — the standard way to write a CRC or checksum field whose value depends on
  the bytes around it. X.692 has no such concept.
* **`save-position`.** Records a field's bit offset for later reference, which is what a
  post-encoding CRC needs to know where its coverage starts.
* **`size deduced`.** A length field whose value is inferred from what surrounds it rather than
  declared.
* **`mapping-function`.** An application-supplied value mapping, beyond clause 24's transforms.
* **`encoding BCD` and `encoding ASCII`** as first-class integer encodings.

### 5.6 A CLI (all three)

`ffasn1dump module.asn Type in.der -O uper out.uper` transcodes between any two encodings in
one command; `-n` disables constraint checking; `-c` follows `CONTAINING` into embedded
messages. asn1c has `unber`/`enber` for round-tripping BER through an editable text form.
asn1scc runs as a daemon and ships a language server.

BCIR's ASN.1 is reachable only from Python. There is no `tools/asn1` CLI. Every capability for
a transcoder exists — six transfer syntaxes, encode and decode on each — but nothing exposes
them at a shell prompt, which is the difference between a codec someone can try in thirty
seconds and one they have to read the test suite to use.

### 5.7 Code generation (asn1c, asn1scc, FFASN1)

The other three turn a schema into compilable source: C structs mirroring the ASN.1 types, with
per-type encoders, decoders and constraint validators. asn1c does this for C; asn1scc for C,
Ada and Scala; FFASN1 for C.

BCIR's C twins are hand-written and speak BCIR's plan format, not per-schema generated code.
This is the one entry on this list that is a **category difference and should stay one** — see
§7.

### 5.8 Formal verification of generated code (asn1scc)

asn1scc's Scala backend emits code carrying Stainless proofs (`ProofGen.fs`, `ProofAst.fs`),
so the generated codec is machine-checked, not merely tested.

BCIR's law rail is the analogous idea aimed at a different target: R1–R25 constrain the *IR*,
and the three rails cross-check each other. Neither approach subsumes the other. Worth noting
as the one place where another compiler's assurance story reaches somewhere BCIR's does not —
into the generated encoder's own control flow.

## 6. What to import, ranked

**1. asn1c's and asn1scc's grammar corpora, as front-end conformance input.** Highest value per
unit of work and no architectural commitment: vendor a set of real protocol modules and assert
that `compile_module` accepts each and produces a stable type inventory. This will surface the
§3 refusals immediately — `COMPONENTS OF` alone appears throughout LTE-RRC and TAP3 — and
probably several more. Do this before anything else on this list, because it tells you what the
rest of the list should have been.

**2. Close the three front-end refusals.** `COMPONENTS OF`, the selection type, and
`WITH SUCCESSORS`/`WITH DESCENDANTS`. All three are X.680 constructs BCIR claims coverage of,
all three are supported by every other compiler here, and all three are inlining or resolution
work in the front end that touches no rail below it. asn1c's `libasn1fix/asn1fix_constr.c`
(`asn1f_pull_components_of`) is a readable reference for the first.

**3. Schema-driven random value generation.** A `random_value(kind, seed)` in the oracle, in
ffasn1's and asn1c's spirit. Feeds three things at once: differential testing across the Python
oracle, the C twins and the law rail; seed corpora for the fifteen existing fuzz harnesses; and
round-trip coverage on the corpus from item 1. This is the single highest-leverage *new* piece
of machinery, because BCIR's triple-rail design means every generated value is checked three
ways rather than one.

**4. A security regression corpus, asn1scc's shape.** `runtime/c/security-regression/<case>/`
with a README naming the defect, the input that triggered it, and a script that proves the fix
holds. Start by back-filling the four defects the recent sweep found. The cost is low and the
value is that a future reader can tell a conformance test from a memory-safety pin.

**5. A bit-layout report.** Render the plan format as a human-readable field-by-field layout,
asn1scc's ICD in intent though not in format. BCIR should emit it as part of the certificate
rather than as a separate document, which is the natural form for a project where the layout is
already a certified artifact.

**6. A CLI.** `tools/asn1/` with transcode, validate and layout-report subcommands. Small, and
it makes six built transfer syntaxes usable by someone who is not reading the test suite.

**7. ACN's CRC/checksum construct — as an ECN *extension*, explicitly outside the conformance
claim.** `post-encoding-function` + `save-position` together solve a real problem X.692 does not
address: a field whose value covers bytes around it. Real protocols need it. But it must be
modelled as a proprietary encoding object in its own namespace, kept out of the X.692
conformance surface, and rejected by the ECN syntax checker unless explicitly enabled.
Folding it into the ECN rail would make "BCIR implements X.692" untestable, which is a worse
outcome than not having CRC fields. The same reasoning applies to `present-when` expressions:
they are strictly more expressive than §21.5's five determinations, and that is precisely why
they are not §21.5.

## 7. What not to import

**Per-schema code generation.** It is the defining feature of the other three and it would cost
BCIR its defining feature. BCIR's C twins exist to be *the same code* on every schema, so that
a fuzz finding or a sanitizer run says something about the codec rather than about one
generated instance. Generated code multiplies the attack surface by the number of schemas and
gives the law rail nothing fixed to constrain. If BCIR ever wants generated codecs, they should
be a *fourth* rail with its own equivalence obligation against the twins, not a replacement.

**CER.** Already excluded with a recorded reason; asn1c decodes it and encodes nothing, which is
the same posture BCIR takes with BER.

**ACN as a notation.** Adopting ACN's syntax alongside ECN would give one custom-encoding idea
two spellings in a project whose whole discipline is that one abstract value has one canonical
form. Take the individual constructs ACN has and ECN lacks, as extensions, in ECN's notation.

**`-n` / disable constraint checking.** ffasn1 offers it and it is useful for debugging a peer.
In BCIR it would produce a value outside its declared ASN.1 type, at exactly the boundary where
octets become values — the boundary the recent PER strictness work spent five clauses closing.

**GSER.** RFC 3641 is a human-readable directory-oriented syntax with no canonical form and no
security-critical deployment. BCIR already has three readable encodings (XER, CXER, JER) and its
selection rule excludes non-canonical syntaxes from emission anyway.
