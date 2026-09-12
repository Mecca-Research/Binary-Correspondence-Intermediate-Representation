# BCIR Unicode integration roadmap — an independent Unicode 17.0 database

> **Status: planning document, no slice landed.** This roadmap owns the *future* of
> a repository-owned Unicode 17.0 database: what it is built from, what it looks
> like on disk, which algorithms it carries, in what order the scripts arrive, and
> how each slice is gated. When the first slice lands, the normative statements
> here move into `training/UNICODE_LANGREF.md` and this document keeps only the
> order of work, the way [`ROADMAP.md`](ROADMAP.md) and
> [`DATABASE_ROADMAP.md`](DATABASE_ROADMAP.md) relate to
> [`TRAINING_LANGREF.md`](TRAINING_LANGREF.md). Landing notes go to
> `docs/DEVELOPMENT_HISTORY.md`, never back into this file.
>
> The database is **independent of the small language model** described in
> [`SLM_ROADMAP.md`](SLM_ROADMAP.md): nothing here depends on it. The SLM depends on
> this database, and says so in its own dependency table. Authority order is the
> repository's: LangRef → generated evidence and implementation → current-state
> audit → master roadmap → companion roadmaps (this one) → research notes.

## 1. The three questions, answered

The request that opened this work asked three things, and each has an answer that
a reviewer can check rather than an opinion.

**Do we copy the characters out of the official PDF, or use the existing Unicode
backend and copy each character we need?** Neither. The PDF code charts are
*renderings* of the Unicode Character Database, produced from the same
machine-readable files this roadmap ingests; they carry a glyph and a name and
omit almost every property a database needs (general category, combining class,
decomposition, case mappings, script, collation weights, confusability, IDNA
status). Copying from them would be a manual re-derivation of data that already
exists as text, with no way to check the copy against the standard's own
conformance tests. The "existing Unicode backend" is the interpreter's
`unicodedata` module, and on this repository's Python floor it is **Unicode
14.0**, not 17.0 (§2.1) — three major versions and roughly six thousand
characters behind, and a different version on every interpreter the suite runs
under. The source of truth is the **machine-readable UCD 17.0 text files**,
vendored into the repository, pinned by SHA-256, and parsed with their own line
grammars (§3). The PDF is reference material for a human; `unicodedata` is a
*differential oracle* over the part of the repertoire it covers (§3.2), never a
source.

**Do we use `.md`, HTML, or another format for the BCIR Unicode database?** None
of the three as the database. Markdown and HTML are *views*; a database is what a
query engine reads, and this repository already has one: the training catalog
(`training/tools/catalog.py`), with content-addressed sets, postings on indexed
columns, packed numeric columns with sorted indexes and zone maps, exact joint
statistics, a legality-first planner, and a C kernel that ranks Q15 vectors under a
predicate mask. The database is therefore a **generated, content-addressed
artifact** built from the pinned UCD files by a deterministic builder — canonical
JSONL rows through the catalog machinery, plus a canonical DER wire form of each
character record for the one consumer that serializes records (the SLM token
pipeline). §5 gives the format in full, with the reasons. `.md` is used for
exactly one thing: the normative document (`UNICODE_LANGREF.md`) that a gate
reconciles against the tools, the way `verify_langref.py` reconciles
`TRAINING_LANGREF.md` today. HTML is not used at all.

**All languages at once, or Latin first?** Latin first, because the Latin stage
already forces every mechanism the later stages need — the file grammars, the
normalization and collation engines, the security and IDNA tables, the gates —
over a repertoire small enough that a wrong answer is legible. Then the scripts
that add one mechanism each (Greek and Cyrillic: nothing new; Arabic and Hebrew:
bidirectional classes and shaping-relevant joining types; Devanagari: combining
classes and grapheme clusters in earnest; Hangul: algorithmic decomposition),
and finally the CJK Unified Ideographs, which add Unihan, radical-stroke data,
implicit collation weights, and a repertoire of over a hundred thousand
characters. §7 gives the staging with what each stage adds and what gates it.

## 2. What exists today

Everything below was read out of the tree for this document, with the file and
line the claim rests on. Counts are deliberately absent; they belong to generated
output.

### 2.1 The host's Unicode tables are the wrong version, and nothing records it

The repository's Python floor is 3.11 (`pyproject.toml:15`). On this host,
`python3 --version` is 3.11.15 and `unicodedata.unidata_version` is **14.0.0**
(measured while writing this document). Unicode 17.0 was released in September
2025; the interpreter tables lag it by three major versions, and they change with
the interpreter: 3.12 ships 15.0, 3.13 ships 15.1, 3.14 ships 16.0. The suite runs
under 3.11 and 3.12 in CI (`.github/workflows/ci.yml`, the floor job and the
portability matrix), so **today two CI jobs normalize text with two different
Unicode versions and no digest can tell**.

Three call sites depend on those tables:

| Site | What it does | Why it matters |
|---|---|---|
| `bcir/hosted/training/data.py:129,133` | `unicodedata.normalize("NFC", ...)` and `unicodedata.category(ch) == "Cc"` on every corpus document | the prepared corpus's content address (`sha256` of the UTF-8 bytes) depends on which NFC ran |
| `bcir/hosted/training/bpe.py:26` | NFC before byte-fallback BPE; the tokenizer JSON records `"normalization": "NFC+LF"` with no version | `CorpusManifest.tokenizer_sha256` and the BCIRQ8 header's `tokenizer_sha256` vouch for a tokenizer whose normalization is host-defined |
| `training/tools/embed_chunks.py:173` | `unicodedata.normalize("NFKC", ...)` + casefold in `LexicalHashProvider._normalize_text` | the provider declares `deterministic = True` and "bit-identical across runs and hosts" while its NFKC tables are the interpreter's |

The byte-native rail is the one place that already has this right:
`bcir/kbcir/byte_latent.py` (`encode_raw_sequence`) refuses to normalize and
records each scalar's exact UTF-8 span, and the ML roadmap states the rule that
every rail should follow — "no implicit normalization" (§1.7) and "Unicode
normalization is not inferred" (§1.8). This roadmap makes normalization an
explicit, versioned, repository-owned operation everywhere (§8).

### 2.2 There is no Unicode database, and the planned home is empty

No code point, name, script, general-category, decomposition, collation,
confusable, or IDNA table exists anywhere in the tree; no UCD file is vendored; no
Unicode version is pinned. `training/formats/README.md` is the planned subject
whose scope already names "Unicode: encoding forms, normalisation, collation, and
the security consequences of confusables", whose reference list names "The
Unicode Standard and UAX #15, #31, #39", and whose verification plan already says
the two things this roadmap needs said: parser lessons are "checked against the
specification's own conformance suites where one exists", and "Never parse a
format with a host-language parser". It is `PLANNED` with no lessons and no gate,
and it sits at Phase 5 of the subject ladder, after `low-level/`
(`training/ROADMAP.md`, "Phases 2–8"). §6 explains how this roadmap opens the
*data* half of that folder now without pretending the *lesson* half is done.

### 2.3 What can be reused, by name

- **Strict UTF-8 in C, twice.** `bcir_jer_utf8_next` / `bcir_jer_validate_utf8`
  (`runtime/c/bcir_jer.c:109-171`) refuse overlongs, surrogates, values above
  U+10FFFF and the C0/C1/F5..FF lead bytes; `valid_utf8_text`
  (`runtime/c/bcir_artifact_bundle.c:229`) is a second, static validator that
  also rejects NUL. Two predicates for one law is the shape
  `docs/security/laws.md` L14 exists to remove; the first Unicode slice that
  needs C UTF-8 unifies them rather than adding a third.
- **A per-scalar record already exists.** `UnicodeByteReference` in
  `bcir/kbcir/byte_latent.py:135` carries `byte_start`, `byte_end`, `codepoint`,
  `utf8_hex`, refuses surrogates, and self-checks against `chr(codepoint)`. The
  character record of §5 is a superset of it, not a rival.
- **The catalog rail.** `schema.Table`/`schema.Column`, `catalog.build(chunk_dir,
  table=...)`, content-addressed publication behind a `CURRENT` pointer, packed
  `<q` numeric columns with a sorted index and per-part zone maps, exact joint
  statistics over indexed pairs, and the predicate grammar with its declared
  refusals — all in `training/tools/`, all gated by `verify_database.py`, all
  reconciled against `TRAINING_LANGREF.md` by `verify_langref.py`. §5 names the
  three places the rail is currently one-table-shaped and what the second table
  needs from it.
- **Content-addressed generations.** `generations.publish/verify/open_catalog`
  give an immutable, verifiable snapshot with a parent link; one Unicode
  *version* is one generation.
- **DER records.** `bcir/asn1/schema.py` declares a type as a module-level
  `Sequence` of tagged `Component`s under `Module(name, (*BCIR_ARC, arc), {...})`;
  `Module.encode` emits DER and `Module.decode` requires DER by default; the X.680
  source ships as package data and is compiled byte-identical against the
  hand-bound model (`bcir/frontends/asn1/lower.compile_module`, the
  `BCIR-StreamPack.asn1` precedent). Arcs 1 (StreamPack), 2 (ArtifactBundle) and
  33 (Manifests) are in use under `1.3.6.1.4.1.62596`.
- **Provenance pattern for third-party inputs.** `docs/machine-learning/THIRD_PARTY_MODELS.md`
  pins an immutable revision, a SHA-256 and a declared license per source, and
  states that nothing downstream is "redistributed"; the UCD files *are*
  redistributed here (their license permits it), so the pin record carries the
  license text as well (§3.5).

## 3. The source of truth

### 3.1 The machine-readable UCD, pinned

Every file below is fetched from `https://www.unicode.org/Public/` under the
17.0.0 directories, verified against a pinned SHA-256, and committed under
`training/formats/unicode/ucd/17.0.0/`. Sizes are approximate (read off the 16.0
files; 17.0 is slightly larger) and exist only to set the budget in §4 — the
pinned byte counts are what the gate asserts.

| File | Directory | What it carries | Consumed by | Conformance suite | ~Size |
|---|---|---|---|---|---|
| `UnicodeData.txt` | `ucd/` | code point; name; general category; canonical combining class; bidi class; decomposition (with `<tag>`); decimal/digit/numeric; bidi mirrored; Unicode 1 name; simple upper/lower/title case | the character table (§5.2) | — | 2.0 MB |
| `DerivedCoreProperties.txt` | `ucd/` | Alphabetic, Lowercase, Uppercase, Math, ID_Start/ID_Continue, XID_*, Default_Ignorable_Code_Point, Grapheme_Base/Extend, … | boolean property columns; UAX #31 identifiers | — | 1.3 MB |
| `PropList.txt` | `ucd/` | White_Space, Dash, Hyphen, Quotation_Mark, Terminal_Punctuation, Pattern_Syntax, Pattern_White_Space, Ideographic, Radical, Unified_Ideograph, … | the "grammar/syntax controls" and "ASCII punctuation/symbols" facets | — | 0.14 MB |
| `Scripts.txt`, `ScriptExtensions.txt` | `ucd/` | Script and Script_Extensions | `script` column; UTS #39 mixed-script | — | 0.2 MB |
| `Blocks.txt` | `ucd/` | block ranges and names | `block` column; the Latin→Han staging | — | 10 KB |
| `PropertyAliases.txt`, `PropertyValueAliases.txt` | `ucd/` | short/long names of every property and value | the vocabulary the LangRef spells values in | — | 0.2 MB |
| `NameAliases.txt`, `DerivedAge.txt` | `ucd/` | formal name aliases; the version each code point was assigned in | name resolution; the `unicodedata` differential (§3.2) | — | 0.2 MB |
| `CaseFolding.txt`, `SpecialCasing.txt` | `ucd/` | full case folding and context-sensitive case mappings | casefold for the lexical provider; UTS #46 | — | 0.2 MB |
| `CompositionExclusions.txt`, `DerivedNormalizationProps.txt` | `ucd/` | composition exclusions; NFC/NFD/NFKC/NFKD quick-check; Full_Composition_Exclusion; NFKC_Casefold | UAX #15 engine | `NormalizationTest.txt` (2.3 MB) | 0.8 MB |
| `auxiliary/GraphemeBreakProperty.txt`, `WordBreakProperty.txt`, `SentenceBreakProperty.txt`, `emoji/emoji-data.txt` | `ucd/` | UAX #29 break properties; Extended_Pictographic | grapheme/word segmentation (the SLM's boundary-aware patching) | `auxiliary/*BreakTest.txt` | 0.6 MB |
| `BidiBrackets.txt`, `BidiMirroring.txt`, `LineBreak.txt`, `EastAsianWidth.txt` | `ucd/` | bracket pairs, mirroring, UAX #14 and UAX #11 classes | punctuation/bracket facets; later stages | `BidiCharacterTest.txt` (later) | 0.6 MB |
| `allkeys.txt` | `UCA/17.0.0/` | the Default Unicode Collation Element Table (DUCET) | UTS #10 engine | `CollationTest_NON_IGNORABLE_SHORT.txt`, `CollationTest_SHIFTED_SHORT.txt` (~5 MB each) | 2.1 MB |
| `confusables.txt`, `confusablesSummary.txt`, `IdentifierStatus.txt`, `IdentifierType.txt`, `intentional.txt` | `security/17.0.0/` | UTS #39 confusable mappings, identifier status and types, intentional confusables | skeleton engine; identifier restriction levels | `confusablesSummary.txt` is a derived check of `confusables.txt` | 1.0 MB |
| `IdnaMappingTable.txt` | `idna/17.0.0/` | UTS #46 status (valid / ignored / mapped / deviation / disallowed / disallowed_STD3_*) and mappings | IDNA engine | `IdnaTestV2.txt` (0.3 MB) | 0.55 MB |
| `CJKRadicals.txt`, `EquivalentUnifiedIdeograph.txt` | `ucd/` | the 214 KangXi radicals → CJK Radicals Supplement + unified ideograph; radical/stroke equivalents | Han stage | — | 25 KB |
| `Unihan_RadicalStrokeCounts.txt`, `Unihan_DictionaryLikeData.txt`, `Unihan_IRGSources.txt` (unzipped from `Unihan.zip`) | `ucd/` | `kRSUnicode` (radical.residual), `kTotalStrokes`, IRG source references; other fields as later stages need them | Han stage | — | 3 + 5 + 12 MB |

Two files that are *not* in the UCD, stated so nobody looks for them: the
Ideographic Description Sequences and per-character **stroke sequences** that the
SLM text calls "sub-character stroke vects" are community data (CHISE IDS,
Make-Me-a-Hanzi), under copyleft licenses, and are out of this roadmap's scope
until a written GO/STOP decision admits them (§7.4). The UCD gives a radical, a
residual stroke count, and a total stroke count per ideograph, and that is what
the Han stage builds on.

### 3.2 `unicodedata` as a differential oracle, not a source

The interpreter's tables are wrong-versioned but not wrong: on every code point
whose `DerivedAge` is ≤ 14.0 and whose properties did not change between 14.0 and
17.0, our tables must agree with `unicodedata.category`, `.combining`,
`.decomposition`, `.numeric`, `.normalize("NFC"/"NFD"/"NFKC"/"NFKD")`. That
agreement is a gate with two halves: the agreement itself, and an anti-vacuity
floor on the number of code points compared — the interpreter's version is read
from `unicodedata.unidata_version` at run time and the comparison set is derived
from it, so the gate runs honestly on 3.11 (14.0) and on 3.12 (15.0) and compares
more on the newer one. Properties that *did* change are carried in a pinned,
explained exception list; the list being empty on a version where changes are
known is itself a finding. This is the `docs/PARITY.md` posture applied to a
third-party table: a differential, not an assertion.

### 3.3 Fetching is a tool, not a pipe

Fetching runs where there is egress (CI has it; this sandbox does not) and
follows the lesson `tools/ci/add_llvm_apt_repo.sh` was written for: download to
a temporary file, retry with backoff, *inspect what came back* (a UCD file begins
with a recognisable header line and its declared version), verify the SHA-256
against `PINS.json`, and only then move it into place. No pipeline, no
`curl | tee`, no partial file ever lands. `PINS.json` records, per file: URL,
SHA-256, byte count, the Unicode version, and the date of the pin. The repository
gate (`verify_unicode.py`, UC-1) recomputes every digest over the committed bytes
and refuses any drift — the vendored tree is the artifact, and its bytes are the
claim.

### 3.4 What the repository's own gates will say about vendored data

These are constraints read out of the tools, not preferences:

- **The secret scanner opens archives** (`tools/security/scan_secrets.py`:
  `ARCHIVE_SUFFIXES` includes `.zip`, `ARCHIVE_MEMBER_CAP = 1 << 20`) and treats a
  member it cannot inspect as a failing finding (`docs/security/laws.md` L3, and
  the scanner's own docstring). `Unihan.zip` has members well over 1 MiB, so it
  is **never committed as a zip**: the needed `Unihan_*.txt` files are committed
  unzipped, as text the scanner reads.
- **Every tracked text file must be LF** (`bcir/tests/test_line_endings.py`,
  `_TREES` includes `training`), and `.gitattributes` applies `text=auto eol=lf`.
  UCD files are LF already; the pin digest is taken over the bytes *as committed*,
  and the gate compares committed bytes, so a checkout on any host verifies.
  `training/llvm/dataset` is the precedent for a data tree under `training/`
  (`_SKIP_ROOTS` in the line-ending test, `extend-exclude` in `pyproject.toml`
  `[tool.ruff]`); the UCD tree is registered the same way only if a gate needs
  it, and the roadmap prefers that it need nothing.
- **The wheel packages only `bcir*`** (`pyproject.toml` `[tool.setuptools.packages.find]`).
  `training/` never ships; the database is a repository/tooling artifact. The
  moment something under `bcir/` needs a Unicode table at import time (the SLM's
  tokenizer will), that table ships as *generated package data* under the
  consuming package — "a package must contain what it registers" (L21) — and is
  gated from the installed wheel the way `bcir.tests.run_all` already runs from
  one.
- **Size budget.** The Latin stage vendors about 10 MB of text (UnicodeData,
  DerivedCoreProperties, PropList, Scripts, Blocks, aliases, case files,
  normalization props and test, the UCA table and the two SHORT collation tests,
  the security and IDNA files). The Han stage adds about 20 MB of Unihan text
  if only the three files above are taken. Both numbers are gated as pins, not
  quoted as facts, and the roadmap declines the full ~40 MB of Unihan until a
  slice names the field it needs.

### 3.5 License

The UCD, the UCA table and the security and IDNA data are published under the
Unicode License v3 (permissive; requires the notice). The notice is committed
verbatim as `training/formats/unicode/ucd/17.0.0/LICENSE` and the dependency
audit's inventory (`tools/security/audit_dependencies.py`) gains a row for the
vendored data so that a license the repository redistributes is one the audit
lists. No file from a copyleft source enters the tree under this roadmap.

## 4. Constraints, as a checklist

Each row is a rule the tree already enforces and the slice that must satisfy it.

| Rule | Where it is enforced | Consequence for this roadmap |
|---|---|---|
| Never parse a wire format with a host-language parser; enforce the grammar first | `docs/security/laws.md` (the host-parser rule), `training/formats/README.md` verification plan | one line-grammar module per UCD file family (§6.6); `int()`/`split()` only after the grammar admitted the line |
| Attribute or refuse; never subset a grammar | laws L4 | a line the grammar does not admit is a refusal that names the file and line, never a skipped row |
| Every exit is a verdict | L1 | the builder and every engine report PASS / FAIL / UNAVAILABLE; a traceback is a defect |
| Anti-vacuity is a state | L2 | every gate carries a floor on rows examined, tests executed, and code points compared |
| Prove the gate can fail | L2, L11, L25 | every slice lands with fault entries in `tools/testing/faults/training-unicode.json` and a RED sweep in its PR body |
| One predicate per repeated defect | L14 | one UTF-8 validator in C; one normalization implementation consumed by three call sites; one grammar per file family |
| An artifact's bytes do not depend on the host | L24 | every writer pins `newline="\n"`; binaries are little-endian by contract; no timestamps in manifests |
| Counts live in generated output | `TRAINING_LANGREF.md` §0, `training/ROADMAP.md` governance | this document quotes no character counts; the builder prints them, the gate asserts them against the pins |
| Legality before cost | `training/tools/plan.py:16` | a Unicode query is refused by a structural rule or priced, never priced into legality |
| Schema version bumps only when a reader would misread | `TRAINING_LANGREF.md` §4.3 | new identifiers `bcir-training/unicode-…/v1`; additions a reader may ignore do not bump |
| `training/` is never a build dependency of BCIR | `CORPUS_STANDARD.md`, `bcir_native.py:12-16` | engines live in `training/tools/unicode/`; when `bcir/` needs a table it is generated *into* `bcir/` as package data, not imported *from* `training/` |
| Measured, modelled, estimated are three words | `training/ROADMAP.md` governance | this document labels every number; sizes in §3.1 are estimates until pinned |

## 5. The database on disk

### 5.1 The decision, with the reasons a reviewer can check

The Unicode database is a **generated, content-addressed artifact** in two
canonical forms produced by one deterministic builder from the pinned files:

1. **Rows through the catalog.** One canonical-JSON record per assigned code
   point (sorted keys, compact separators, `\n`, no timestamps), written to
   `build/training/unicode/chunks/*.chunks.jsonl` and built by
   `catalog.build(chunk_dir, table=UNICODE_TABLE)` into a catalog set
   (`catalog.json`, `postings.json`, `ids.txt`, `locator.bin`, `numeric.bin`,
   `order.bin`) behind a `CURRENT` pointer, published as a generation
   `g-<digest16>`. This is the *queryable* database: postings on the
   low-cardinality columns, exact joint statistics, packed numeric columns with
   a sorted index and zone maps, the predicate grammar, the planner, and the C
   kernel under a predicate mask — all of it already gated.
2. **Records as DER.** The same records, encoded by a `BCIR-Unicode` ASN.1
   module (a new arc under `1.3.6.1.4.1.62596`, allocated in
   `docs/BCIR_ASN1_X690_ABI.md`'s registry) as a DER `SEQUENCE OF
   UnicodeCharacterRecord` in code-point order, `records.der`, with its SHA-256
   in the generation manifest. This is the *wire* form: the one consumer that
   serializes character records across the `training/` → `bcir/` boundary
   (the SLM token pipeline, [`SLM_ROADMAP.md`](SLM_ROADMAP.md) §3) reads bytes
   with a canonical form that `bcir`'s own DER decoder verifies (laws A1–A5),
   because `bcir/` may not import `training/`.

A gate asserts the two forms agree field for field (DER-decoded record ==
JSON row for every row), so there is one truth with two readers, and the
dual-rail law (L12) has a witness from the first slice.

Why not the alternatives, each with the fact that decides it:

- **Markdown or HTML as the database.** A database is what a query engine
  reads. `search_chunks --where 'script=Latn AND general_category=Lu'
  --count --group-by block` is a question the catalog answers exactly from
  statistics without a scan; a Markdown table answers it by a human reading.
  Markdown code-chart pages can be *generated from* the generation for the
  `formats/` lessons (they flow through `build_chunks` unchanged, kind
  `table`), and that is their place: a view, derived, never edited by hand.
  HTML is a rendering of a rendering.
- **DER alone, with no catalog.** Then every predicate is a scan and every
  count a loop; the rail's exact statistics, sorted index and planner are
  thrown away for a format that has no query surface. DER is the wire form
  because the wire needs canonical bytes; it is not the index.
- **A third-party database engine.** `DATABASE_ROADMAP.md` already decided this
  question for the training rail ("Why this points inward"); a Unicode table
  is the smaller, better-structured case of the same question, and adding an
  engine would add a dependency to a tree whose oracle has none.
- **The host's `unicodedata`.** Unicode 14.0, unversioned, no DUCET, no
  confusables, no IDNA, no Unihan (§2.1).

### 5.2 The row schema — `bcir-training/unicode-character/v1`

One row per assigned code point. Roles follow `TRAINING_LANGREF.md` §3.1 —
the role decides what may be done to a column and is the only thing that
decides it — and the declaration lives in one place
(`training/tools/unicode/schema_unicode.py`, reconciled with
`training/schema/unicode-character-v1.json` by the schema gate, the way
`schema.py` and `chunk-v1.json` are reconciled today).

| Column | Role | Value | Notes |
|---|---|---|---|
| `record_id` | key | `U+XXXX` (4–6 uppercase hex digits) | exact reverse lookup by `Catalog.row_of`; the only identity |
| `source_path` | path | `blocks/<plane>/<block-slug>` | the hierarchy `^=` and the directory postings understand: `^=blocks/0/` is the BMP, `^=blocks/0/cjk-unified-ideographs` one block. Named `source_path` because readers use the module-level `PATH_COLUMN`; UC-3 decides whether that name becomes a table attribute (§5.4) |
| `code_point` | numeric | 0..1114111 | ranges and ORDER BY; equality spelled as a closed interval; the row order |
| `general_category` | indexed | the 30 `gc` short values | |
| `script` | indexed | `Script` short value | ~170 values; joint statistics with `general_category` measured on the Latin slice before `block` joins the indexed set |
| `block` | indexed | block name slug | |
| `bidi_class`, `decomposition_type`, `age`, `east_asian_width`, `line_break` | indexed | UCD short values | low cardinality by role |
| `ccc` | numeric | 0..255 | |
| `numeric_num`, `numeric_den` | numeric | the `Numeric_Value` as a reduced rational, absent for non-numerics | absence is the null the rail already models |
| `uca_primary`, `uca_secondary`, `uca_tertiary` | numeric | the first collation element's weights, or the implicit weight for Han | SLM_ROADMAP §2.6 |
| `uca_variable` | indexed | `0`/`1` | |
| `confusable_class` | numeric | the smallest code point with the same UTS #39 skeleton | |
| `identifier_status` | indexed | `Allowed` / `Restricted` | |
| `idna_status` | indexed | the seven UTS #46 statuses | |
| `alphabetic`, `math`, `pattern_syntax`, `white_space`, `dash`, `quotation_mark`, `terminal_punctuation`, `ideographic`, `unified_ideograph`, `default_ignorable` | indexed | `0`/`1` | the booleans a predicate needs; every other boolean property stays in `properties` below. UC-3 measures the manifest's growth per indexed boolean and may demote some to text |
| `han_radical`, `han_residual_strokes`, `han_total_strokes` | numeric | from `kRSUnicode` / `kTotalStrokes`; absent outside Han | |
| `utf8_length` | numeric | 1..4 | |
| `name`, `name_aliases`, `decomposition`, `case_upper`, `case_lower`, `case_title`, `idna_mapping`, `properties`, `identifier_type` | text | materialized only, never filtered | `decomposition` as `"0041 0301"`; `properties` as the sorted list of every true boolean property; `identifier_type` as the UTS #39 set |

Row identity is `record_id`; row order is `unicode_row_sort_key(record) =
(code_point,)`, imported by the catalog builder and by every embedding-set
writer from one module (the row-order law of `TRAINING_LANGREF.md` §4.4,
applied to a second table). A row with an undeclared field is refused
(`Table.check_row`); a row missing a NOT NULL column is refused (S17).

### 5.3 What the catalog rail needs before it can hold a second table

The maps of the rail found three places where it is one-table-shaped, and
UC-3 removes them — additively, with the chunk table's behaviour byte-identical
before and after (the gate is the existing `verify_database.py` battery plus a
digest comparison of a catalog built before and after the change):

1. `catalog._scan_chunk_file` projects a hardcoded set of column names plus
   `table.numeric`; a new *indexed* column would be posted under the null key.
   The projection becomes `table.projectable` — every declared column.
2. `Table.__init__` requires exactly one `key` and one `path` column, and the
   readers (`rows_absent`, `rows_present`, `count_equals`, `plan.estimate`,
   `_paths_starting_with`) use the module-level `PATH_COLUMN`. Either the
   Unicode table names its hierarchy column `source_path` (the minimal path,
   §5.2), or `PATH_COLUMN` becomes `Catalog.table.path` (the generalization).
   The roadmap recommends the generalization: two tables reading one module
   global is the "second spelling" L14 exists to remove, and `Catalog.load`
   already infers columns from `statistics` and can be handed the table.
3. `Catalog.load` takes no table. It gains `table=` with the chunk table as
   the default, so every existing caller is unchanged.

Ceilings the second table will meet, all declared in the LangRef and measured
by UC-9/UC-10 rather than assumed: `locator.bin`'s `u16` file index (at most
65536 chunk files — a character table needs one file per block, well under),
`MAX_INDEXABLE_ROWS = 2^32−1`, parts of 128 rows listed individually in
`catalog.json` with per-part zone maps (the manifest is read on every load;
its size at the Han stage is a measurement the slice records), and the joint
statistics' full cross product of observed index-key pairs per column pair.

### 5.4 Generations, versions, and the sets bound to them

One Unicode version is one generation. The generation manifest's `extra`
carries `ucd_version`, the digest of `PINS.json`, the admitted block list of
the stage (§7), and `records_sha256`. Embedding sets over the character table
(the stroke and radical spaces, UC-10) are ordinary `bcir-training/embedding-set/v1`
sets, one provider each (`unicode-stroke-v1`, `unicode-radical-v1`, both
`deterministic = True`, both derived from record fields and never from free
text), bound to the generation by ids-in-order and refused otherwise. A set
built over one generation cannot serve another; the check is structural.

### 5.5 Where everything lives

| Path | What | Committed? |
|---|---|---|
| `training/formats/unicode/ucd/17.0.0/*.txt`, `LICENSE`, `PINS.json` | the vendored source of truth | yes — pinned bytes |
| `training/tools/unicode/{fetch,ucd,schema_unicode,build_unicode,normalize,collate,security,idna,segment,lookup,providers_unicode,verify_unicode,verify_unicode_langref}.py` | the rail | yes |
| `training/schema/unicode-character-v1.json` | the JSON schema twin of the declaration | yes |
| `training/UNICODE_LANGREF.md` | the normative document (from UC-1 on) | yes |
| `build/training/unicode/…` | chunks, catalog set, generations, `records.der`, sets | no — generated |
| `bcir/asn1/BCIR-Unicode.asn1`, `bcir/asn1/unicode_record.py` | the DER module and its hand-bound twin (UC-8) | yes |
| `bcir/hosted/training/unicode_tables.py` (generated, Latin-stage sized) and the Latin record fixture | the package-data tables `bcir/` may use without importing `training/` (UC-7) | yes — generated, digest-recorded |
| `tools/testing/faults/training-unicode.json` | the RED-sweep fault table | yes |

## 6. Algorithms — dependency-free, oracle first, gated by the standard's own tests

Every engine below is written once in Python under `training/tools/unicode/`,
against the pinned tables, with the standard's conformance file as its gate. None
imports `unicodedata` except the differential gate of §3.2. A C twin is written
only when a consumer on the C rail exists (`docs/BCIR_MASTER_ROADMAP.md` §8:
"do not reimplement a standard without two concrete consumers or a documented
device-local need"); today the one C-side consumer is UTF-8 validation, which
already exists twice.

### 6.1 UAX #15 — normalization

Canonical decomposition (recursive, with the Hangul syllable algorithm), canonical
ordering (a stable sort of each run of non-starters by combining class),
canonical composition (with the composition exclusions and the Hangul algorithm),
and the compatibility variants. The quick-check properties are used as they are
meant to be — a fast path whose answer is verified against the full algorithm in
the gate, never a substitute for it.

*Gate:* every line of `NormalizationTest.txt` — five columns, all sixteen implied
identities (`NFC(c1) == NFC(c2) == NFC(c3) == c2`, …) — plus the file's declared
invariant that every code point not listed in Part 1 is its own normalization in
all four forms, which is the anti-vacuity half (it walks the whole repertoire).
*RED:* drop the composition-exclusion check; the test's Part 1 fails on the first
excluded composition.

### 6.2 UTS #10 — the Unicode Collation Algorithm over DUCET

Collation elements are read from `allkeys.txt` (contractions by longest match
over the normalized string, expansions as written); the sort key is the
concatenation of level-1 weights, a `0000` separator, level-2 weights, `0000`,
level-3 weights, and — under the *shifted* variable-weighting option — a fourth
level; the two options the conformance tests exercise, non-ignorable and shifted,
are both implemented. Characters with no entry get **implicit weights** computed
from the code point; for CJK Unified Ideographs the standard's formula is

```
AAAA = 0xFB40 + (cp >> 15)          BBBB = (cp & 0x7FFF) | 0x8000
```

with `0xFB80` as the base for the extension blocks and `0xFBC0` for unassigned
code points, yielding the two-element sequence `[AAAA.0020.0002][BBBB.0000.0000]`.
This is the deterministic "base form" the SLM text asks for: two strings with
equal level-1 keys differ only in accents, case, or variable characters, and that
equivalence is computed, not learned.

*Gate:* `CollationTest_NON_IGNORABLE_SHORT.txt` and `CollationTest_SHIFTED_SHORT.txt`
— consecutive lines must compare `≤` under the respective option, every line —
with an anti-vacuity floor on lines compared. *RED:* swap the level-2 and level-3
weights in the key; the shifted test fails on the first accented pair.

### 6.3 UTS #39 — security mechanisms

`skeleton(s) = NFD(map_confusables(NFD(s)))`; two strings are confusable when
their skeletons are equal. Identifier status and type come from
`IdentifierStatus.txt` / `IdentifierType.txt`; mixed-script detection from
`Scripts.txt` and `ScriptExtensions.txt`; the restriction levels (ASCII-only,
single script, highly restrictive, moderately restrictive, minimally restrictive)
are computed as the report defines them. For the SLM this is the canonical-form
map that collapses homoglyph attacks and duplicate token representations onto one
row.

*Gate:* `confusablesSummary.txt` is a derived rendering of `confusables.txt`; the
engine regenerates the summary's equivalence classes from the source table and
they must match — a differential against the standard's own derivation. *RED:*
skip the second NFD in the skeleton; a class whose prototype decomposes splits in
two.

### 6.4 UTS #46 — IDNA compatibility processing, with RFC 3492

Mapping by `IdnaMappingTable.txt` status, NFC, label splitting on U+002E, per-label
validity (NFC, hyphen rules, no leading combining mark, the Bidi rule, ContextJ
and ContextO), Punycode (RFC 3492) decoding of `xn--` labels and encoding for
ToASCII, with `UseSTD3ASCIIRules` and `Transitional_Processing` as explicit flags
of the engine, never defaults chosen silently. Punycode is implemented in the
repository, dependency-free; the transitive `idna 3.19` in the dependency
inventory is a hosted-only differential oracle at most, never an import in
`training/tools/`.

*Gate:* every line of `IdnaTestV2.txt` (source, ToUnicode result and status,
ToAsciiN and status, ToAsciiT and status). *RED:* accept a leading combining mark
in a label; the test's `V6` cases pass when they must fail.

### 6.5 UAX #29 — grapheme and word segmentation

Needed by the SLM's byte-native rail, which today patches on byte indices with no
boundary awareness (`bcir/hosted/training/byte_latent.py`, the maps' finding), and
by the lexical provider's word regex, which is ASCII-only. Grapheme cluster and
word boundaries from the auxiliary property files, with the emoji rules.

*Gate:* `GraphemeBreakTest.txt` and `WordBreakTest.txt`, every line. *RED:* drop
rule GB9 (do not break before extending characters); the first combining-mark
case fails.

### 6.6 Line grammars, one per file family

The UCD files are semicolon-separated fields with `#` comments and `..` ranges;
`allkeys.txt` has its own `[.XXXX.YYYY.ZZZZ]` element grammar; `confusables.txt`
and `IdnaMappingTable.txt` have their own; Unihan is tab-separated
`U+XXXX<TAB>kField<TAB>value`. Each family gets one grammar module that admits a
line or refuses it *by name* (file, line number, the first offending byte) before
any `int(x, 16)` is called — the rule the repository paid for on its text rails
(the JER scanner is the model, `docs/security/laws.md`). A refused line is never
a skipped row: it is a FAIL that names itself.

## 7. Staging — Latin first, Han last

| Stage | Repertoire admitted | What it adds beyond the previous stage | Gate that closes it |
|---|---|---|---|
| **Latin** | Basic Latin, Latin-1 Supplement, Latin Extended-A/B/Additional, Combining Diacritical Marks, General Punctuation, Currency, Letterlike, Number Forms, Arrows, Mathematical Operators, Box Drawing, Geometric Shapes, Misc. Symbols, Dingbats (blocks named from `Blocks.txt`, not spelled here) | the full table schema; all six engines; all conformance suites restricted to the admitted repertoire; ASCII punctuation, syntax controls (`Pattern_Syntax`, `Pattern_White_Space`), bracket pairs, math operators (`Math` property) | `verify_unicode.py` PASSED with every conformance suite's admitted subset passing and the anti-vacuity floors met |
| **Greek, Cyrillic, Armenian, Georgian** | those blocks | nothing mechanical: a repertoire test of the same engines; the first non-Latin confusables (`о`/`o`, `а`/`a`) exercise UTS #39 for real | the same gate, larger admitted subset |
| **Hebrew, Arabic, Syriac, Thaana, NKo** | those blocks | bidi classes and mirroring used in earnest; joining types (`ArabicShaping.txt`, added at this stage) ; the Bidi rule of UTS #46 | `BidiCharacterTest.txt` added; IDNA Bidi cases |
| **Devanagari and the Brahmic scripts, Thai, Lao, Tibetan** | those blocks | combining classes and reordering that the normalization gate could not reach on Latin; grapheme clusters of many code points | the full `NormalizationTest.txt` and `GraphemeBreakTest.txt` |
| **Hangul** | Hangul Jamo, Syllables, Compatibility Jamo | algorithmic decomposition/composition; the collation table's contractions and expansions | full collation tests |
| **CJK Unified Ideographs (Han)** | the URO, Extensions A–J, Compatibility Ideographs, Kangxi and CJK Radicals, IDCs | Unihan (`kRSUnicode`, `kTotalStrokes`, IRG sources), `CJKRadicals.txt`, `EquivalentUnifiedIdeograph.txt`; implicit collation weights; the radical-stroke columns of the table; Extension J, new in 17.0 | the full repertoire under every suite; the radical-stroke columns gated against Unihan; the implicit-weight formula gated against the collation tests' Han lines |
| **The rest of 17.0** | every remaining block, including the four scripts new in 17.0 and the emoji additions | a repertoire test | the whole repertoire admitted; the pinned total from the 17.0 release matches the row count |

Two remarks that keep the staging honest. First, a stage is a *repertoire
filter on one builder*, not a fork: the same tools, tables and gates run at every
stage, and the admitted set is a declared list of blocks that the gate reads back
(a block admitted but absent from the table is a finding; a block present but
not admitted is one too). Second, the "final boss" is not the algorithms — the
implicit-weight formula is four lines — it is the *data*: the Han stage triples
the vendored bytes and multiplies the row count by roughly two orders of
magnitude, which is exactly where the catalog's declared ceilings
(`MAX_INDEXABLE_ROWS`, the u16 file index in `locator.bin`, parts of 128 rows
each listed in `catalog.json`) are met and must be measured rather than assumed
(§5 and UC-9).

### 7.4 What the Han stage cannot get from the UCD

Sub-character *structure* — which components an ideograph is composed of, in
what arrangement, and the stroke *sequence* — is not Unicode data. The
Ideographic Description Characters exist (U+2FF0..U+2FFF) but the sequences that
use them are community-maintained (CHISE IDS, Wikimedia's IDS lists), as are
stroke-order datasets (Make-Me-a-Hanzi), and all of these are under copyleft or
mixed licenses. The SLM text's "sub-character stroke vectors" therefore has two
honest realizations: (a) vectors over what the UCD does give — radical, residual
strokes, total strokes, IRG source, and the equivalent-unified-ideograph relation
— which is enough to make characters sharing a semantic determinative adjacent,
and (b) a later GO/STOP decision on admitting an external IDS dataset, with a
license review by the existing audit and a pinned revision, written in the style
of `docs/BCIR_NATIVE_OBJECT_GATE.md` before any byte of it is read.

## 8. Replacing the host tables, and putting the version into every digest

The three call sites of §2.1 are replaced in one slice (UC-6), by one
implementation: `training/tools/unicode/normalize.py` for the two `training/`
sites, and a *generated* `bcir/hosted/training/unicode_tables.py` (package data,
built from the same generation, digest-recorded) for the two `bcir/hosted` sites —
because `bcir/` may not import `training/`. From that slice on:

- `bcir.byte_bpe.v1` is superseded by `bcir.byte_bpe.v2`, whose `normalization`
  field names the form *and* the Unicode generation digest (`"NFC@17.0.0:<digest>"`);
  `from_json` continues to refuse any other key set, so a v1 tokenizer is read as
  what it is — normalized by an unrecorded table — and never silently promoted.
- `DataPreparationSpec.policy_sha256` (`bcir/hosted/training/data.py:44-46`)
  folds the generation digest into the policy, so a prepared corpus's content
  address changes when its normalization tables do, which is the point.
- The embedding-set manifest (`bcir-training/embedding-set/v1`) gains no field: a
  provider whose normalization tables changed is a new provider *revision*, which
  the manifest already records; `LexicalHashProvider` becomes revision 2 the day
  it reads repository tables, and its "bit-identical across hosts" claim becomes
  true.
- The BCIRQ8 header's `tokenizer_sha256` needs no change: it vouches for the
  tokenizer JSON, which now vouches for the tables.

`unicodedata` remains importable in exactly one place, the differential gate of
§3.2. A `bcir/tests/` witness asserts that no module under `bcir/hosted/training/`
or `training/tools/` imports it anywhere else — the same shape as
`test_hot_cold.py`'s import law.

## 9. What this roadmap refuses to claim

- That the repository's normalization is Unicode 17.0 today. It is the
  interpreter's, unversioned, and the roadmap's first gate says so before any
  slice fixes it.
- That vendored sizes, character totals, or block ranges quoted here are facts.
  They are expectations that the pin gate asserts; the gate's output is the fact.
- That any engine is fast. The engines are oracles, gated for correctness against
  the standard's own tests; a C twin is written when a C consumer exists and is
  gated by parity, and no timing is quoted until it is measured with provenance.
- That stroke sequences or ideographic structure are available from the UCD. They
  are not (§7.4).
- That `training/formats/` is open as a *subject*. Its data half opens under this
  roadmap; its lesson half stays on the subject ladder until the three-condition
  gate of `training/ROADMAP.md` is met for lessons, which this roadmap advances
  (the references are inventoried and pinned by UC-1) but does not complete.

## 10. The build slices

One gateable slice per PR, id in the title, RED before GREEN, a fault entry in
`tools/testing/faults/training-unicode.json` per new check, exact gate output
in the PR body. Every check group named below is registered in
`verify_unicode.py`, listed in the gates section of `UNICODE_LANGREF.md` (the
normative document UC-1 opens, §5.5; its §16 mirrors `TRAINING_LANGREF.md`
§16), and reconciled both ways by `verify_unicode_langref.py` — the
`check_check_groups` discipline `verify_langref.py` already enforces for the
chunk table.

### UC-1 — the pins, the fetch tool, and the source tree

`training/tools/unicode/fetch.py` (temp file, retries, header inspection,
digest before install — never a pipe), `PINS.json`, the `LICENSE`, the Latin
stage's files under `training/formats/unicode/ucd/17.0.0/`, the audit row for
the redistributed license, and `verify_unicode.py` with its first check group.

*Payoff:* the repository owns a Unicode 17.0 source of truth whose bytes are
the claim.
*Gate:* check group `pins` — every vendored file's SHA-256 and byte count equal
`PINS.json`; every pinned file exists; every file's header declares the pinned
version; the tree is LF; the scanner passes over it. *RED:* flip one byte in a
vendored file — `pins` names the file; add a file to the tree without a pin —
`pins` names it as unpinned. *Depends on:* nothing.

### UC-2 — line grammars and the character record

`ucd.py`: one grammar per file family (§6.6), refusing by file and line before
any conversion; the in-memory `CharacterRecord` assembled from `UnicodeData`,
`DerivedCoreProperties`, `PropList`, `Scripts`, `Blocks`, aliases, case files
and `DerivedAge`, with `..` ranges and the `<…, First>`/`<…, Last>`
convention expanded; the Latin repertoire filter as a declared block list.

*Payoff:* every later slice reads records, not files.
*Gate:* check group `grammar` — every admitted line of every file parses;
a corrupted line (a non-hex field, a missing semicolon, a range with `First`
and no `Last`) is refused naming file and line; check group `records` — the
`unicodedata` differential of §3.2 with its anti-vacuity floor; the admitted
block list and the built repertoire agree both ways. *RED:* accept a line
whose code-point field is `0O41` — the grammar witness fires; drop
`First/Last` expansion — the Han-range rows vanish and the block gate fires.
*Depends on:* UC-1.

### UC-3 — the second declared table

The three generalizations of §5.3, `schema_unicode.py` and
`unicode-character-v1.json`, `unicode_row_sort_key`, `build_unicode.py`
writing canonical JSONL and building a catalog set, published as a generation
with `ucd_version` in `extra`.

*Payoff:* `search_chunks --catalog build/training/unicode/catalog --where
'script=Latn AND general_category=Lu' --count --group-by block` answers from
statistics, exactly.
*Gate:* check group `schema` — declaration == JSON schema; an undeclared field
is refused; a NOT NULL column absent is refused; check group `build` — two
builds give one `publication_id` and one generation id; every indexed column's
postings sum to `rows_total`; the chunk table's existing battery
(`verify_database.py`) is green and a chunk catalog built before and after the
generalization has the same publication id. *RED:* build with an indexed
column outside the old hardcoded projection — its postings land under the
null key; revert `Catalog.load(table=)` — the Unicode catalog reads
`source_path` prefixes as file paths. *Depends on:* UC-2.

### UC-4 — UAX #15 normalization

`normalize.py` (§6.1), the quick-check fast path verified against the full
algorithm, the `NormalizationTest.txt` gate, and the `unicodedata`
differential for normalization.

*Gate:* check group `normalize` — every test line, all four forms, all implied
identities; the Part-1 invariant walk over the whole admitted repertoire;
agreement with the host on the ≤ 14.0 overlap with the pinned exception list
non-empty-or-explained. *RED:* drop the composition exclusions. *Depends on:*
UC-2.

### UC-5 — UTS #10 collation

`collate.py` (§6.2): the `allkeys.txt` grammar, collation elements with
contractions and expansions, sort keys under both variable-weighting options,
implicit weights, and the `uca_*` columns written into the table by
`build_unicode.py`.

*Gate:* check group `collate` — both SHORT conformance files, every line;
`ORDER BY uca_primary` agrees with the level-1 order of the full key on every
admitted pair; the implicit-weight formula gated on the conformance files'
Han lines (UC-10 widens the repertoire). *RED:* swap level-2 and level-3
weights; change the Han base to `0xFB41`. *Depends on:* UC-3, UC-4.

### UC-6 — UTS #39 and UTS #46, with Punycode

`security.py` and `idna.py` (§6.3–6.4), their columns, and the RFC 3492 codec.

*Gate:* check group `security` — the summary regenerates from the source
table; every pair in `confusables.txt` shares a `confusable_class`; check group
`idna` — every `IdnaTestV2.txt` line; Punycode round-trips the RFC's sample
strings; a leading combining mark is refused. *RED:* skip the second NFD in
the skeleton; accept a leading combining mark. *Depends on:* UC-3, UC-4.

### UC-7 — replacing the host tables

`bcir/hosted/training/unicode_tables.py` generated from the generation
(Latin-stage sized, digest-recorded) and consumed by `data.py` and `bpe.py`;
`embed_chunks.LexicalHashProvider` at revision 2 reading `normalize.py`;
`bcir.byte_bpe.v2`; the import witness that no module under
`bcir/hosted/training/` or `training/tools/` imports `unicodedata` outside the
differential gate; the generation digest folded into `policy_sha256`.

*Payoff:* the same corpus normalizes to the same bytes on 3.11 and 3.12, and
every digest says which tables did it.
*Gate:* a prepared corpus's content address is identical under two
interpreters with different `unidata_version`; the import witness; a v1
tokenizer JSON still loads as v1 and is reported as unversioned; the
generated tables' digest matches the generation's. *RED:* put `import
unicodedata` back in `data.py` — the witness fires; regenerate the tables from
a different generation — the digest gate fires. *Depends on:* UC-4 (and UC-5
for casefold).

### UC-8 — the DER record, and UAX #29

`bcir/asn1/BCIR-Unicode.asn1` and `unicode_record.py` (the sketch in
SLM_ROADMAP §3.2, made exact), `records.der` in the generation, the A1–A5
gates and R24 static rules; `segment.py` with the grapheme and word break
gates.

*Gate:* check group `records-der` — DER re-encodes byte-identically; BER
decodes; DER-decoded record == JSON row for every row; the X.680 source
compiles byte-identical to the hand-bound module; check group `segment` —
every `GraphemeBreakTest`/`WordBreakTest` line. *RED:* reorder two
components; drop rule GB9. *Depends on:* UC-3 (rows), UC-5, UC-6 (fields).

### UC-9 — the script stages

Greek/Cyrillic/Armenian/Georgian, then the right-to-left scripts (with
`ArabicShaping.txt` and `BidiCharacterTest.txt` added to the pins), then the
Brahmic scripts, Thai, Lao, Tibetan, then Hangul — each a repertoire
extension of the admitted block list with the full conformance suites, and a
recorded measurement of `catalog.json`'s size and the joint-statistics cross
product at each stage.

*Gate:* the same battery over the larger admitted set; the manifest-size
measurement recorded in the slice's PR (class `wall`, indicative). *RED:* the
existing faults, re-run — a fault that stopped biting on the larger repertoire
is the finding (`tools/testing/faults/README.md`). *Depends on:* UC-8.

### UC-10 — Han

Unihan ingest (three files, unzipped, each under the scanner's caps),
`CJKRadicals.txt` and `EquivalentUnifiedIdeograph.txt`, the `han_*` columns,
implicit collation weights over the whole repertoire, the two deterministic
providers (`unicode-stroke-v1` over radical/residual/total strokes and their
one-hot structure; `unicode-radical-v1` over the radical and the equivalent
unified ideograph relation), the sharding of sets above 65536 rows, and the
measurement of every catalog ceiling §5.3 names.

*Payoff:* the "final boss" is data, and the data is in.
*Gate:* the Han lines of both collation conformance files; `han_radical` and
strokes agree with Unihan for every ideograph; every shard ≤ 65536 rows and
bound by ids-in-order; the pinned Extension J block is present; the ceilings
are measured and under. *RED:* offset one shard's base; drop the implicit
weight for the extension blocks. *Depends on:* UC-9.

### UC-11 — the rest of 17.0, the LangRef, and the subject question

Every remaining block (the four scripts new in 17.0, the emoji additions),
the pinned total from the release matching the row count, `UNICODE_LANGREF.md`
promoted to normative with `verify_unicode_langref.py` reconciling roles ×
operators, binary formats, artifact names, schema ids, gate names and check
groups both ways, and a written note to `training/ROADMAP.md` on whether the
`formats/` *lesson* half opens now that its references are inventoried and
pinned.

*Gate:* the whole repertoire under every suite; the LangRef gate green;
`check_check_groups` both ways. *RED:* register a check group the LangRef
does not list. *Depends on:* UC-10.

## 11. CI wiring and the gate battery

| Gate | Job | Flag |
|---|---|---|
| `training/tools/unicode/verify_unicode.py` | `training-llvm` (beside `verify_database.py`) and `host-portability` (both OSes) | none needed — no native kernel until UC-10's sets, then `--require-native` where the job installs a compiler |
| `training/tools/unicode/verify_unicode_langref.py` | `host-portability` (beside `verify_langref.py`) | — |
| the `unicodedata` differential | wherever the Python floor job runs 3.11 and the portability job runs 3.12 — both, so the comparison set differs and the gate proves it reads the version | — |
| `python -m bcir.tests.run_all` witnesses (import law, record-module A1–A5, layout literal witness) | the oracle shards | — |
| RED sweep | local before every PR, `tools/testing/red_sweep.py --faults tools/testing/faults/training-unicode.json`, output in the PR body | — |

Local runs stay bounded: two workers, heavy gates serialized. The Han stage's
build is the one step that may be slow; it is measured, and the measurement is
labeled `wall`.
