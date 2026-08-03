# BCIR JSON program representation — research note and roadmap

> **Status:** research note plus a proposed phase ladder. It defines future gates and
> records prior art and corrections; it claims **no** implementation. Nothing in this
> document is built.
>
> **Relationship to existing roadmaps:** this extends
> [`BCIR_ASN1_JSON_ROADMAP.md`](BCIR_ASN1_JSON_ROADMAP.md), which stops at compiling JER
> *data* into claims. This note asks the next question — whether JER can carry a *program* —
> and answers it with one scope change to J4 and a new dependent track. **J3 is unaffected.**

## 1. The proposal, restated precisely

The proposal is that a schema-bound JSON format can be a program representation: source
code *is* the AST, any front end (text, visual, generative) emits the schema, and the
compiler consumes it directly. Programs may then rewrite their own logic blocks at runtime
and re-enter the optimizer.

Restating it in terms of what this repository already has changes the work considerably,
and for the better:

> **BCIR already has a serialized IR.** The `bcir.*` MLIR dialect plus MLIR bytecode is a
> schema-bound, machine-first program representation with a verifier (R1–R25), an optimizer
> (K_BCIR), and a lowering path to machine code. The proposal is therefore **not** "invent a
> program representation". It is **"add a third serialization of the IR BCIR already has,
> in JER, and prove the projection commutes"** — which is the same shape as the existing
> Python-oracle ↔ MLIR-law ↔ C-twin discipline, and the same shape as R24's additive
> projection requirement.

That reframing is the single most useful result of this note. It converts an open-ended
language-design programme into a bounded projection problem with an existing correctness
law to satisfy.

## 2. Prior art, because none of the core ideas are new — and that is the good news

| Idea in the proposal | Where it already exists | What BCIR would take from it |
|---|---|---|
| Source code *is* the AST | Lisp s-expressions (1958); homoiconicity | The idea works and is 65 years old. The interesting part is never the representation, it is the *reference* and *scope* story below |
| Binary/structured AST as the shipped artifact | WebAssembly, LLVM bitcode, **MLIR bytecode**, Roslyn syntax trees | A serialized IR is normal engineering, not a novelty. MLIR bytecode is the closest analogue and BCIR already emits it |
| Content-addressed code, text as a *view* | [Unison](https://www.unison-lang.org/docs/the-big-idea/) | The decisive discipline: the canonical artifact is authoritative and the human syntax is a **projection**, never the source of truth. This solves the verbosity objection outright |
| JSON-shaped ASTs | ESTree/Babel, JSON-LD | Proof the shape is workable, and a catalogue of exactly where it hurts |
| Typed cross-references in a schema | **ASN.1 X.681 information object classes + X.682 table constraints** | Already built in this repo. See §4.1 — this is the answer to the reference problem, and it is better than inventing `$ref` |
| Program transformation as a cost problem over a graph | Min-plus (tropical) algebra; Dijkstra/Bellman–Ford; Karp's minimum mean cycle | The formal grounding for §5, and it is genuinely sound |

**What is actually novel in the proposal** is the combination: a *standards-bound* schema
(ASN.1, with a canonical byte form and a registry), a *cost-governed* selection stage
(K_BCIR over the twelve-axis vector), and a *verifier* (R1–R25) applied to a program
representation rather than to data. That combination does not exist elsewhere, and it is
worth pursuing. The individual pieces should be borrowed, not reinvented.

## 3. Three claims in the proposal that must be corrected before they reach a design

Recording these because a roadmap that carries an overclaim tends to grow a feature that
defends it.

### 3.1 "Compilers waste immense energy parsing human text variations"

This is the weakest claim and, as stated, it is very likely false for optimizing compilers.
In a modern `-O2` build, lexing and parsing are a small share of wall time; semantic
analysis, optimization and code generation dominate, and in C++ the true cost is
*preprocessing and header re-parsing*, not grammar. Deleting the parser deletes a small
constant.

The real wins of a schema-bound representation are different, larger, and honest:

- **No ambiguous or under-specified grammar corners** — the schema either admits a
  construct or it does not, and `jer_plan.py` already refuses what it cannot represent.
- **Structural invariants are checkable before semantics** — J1's bounded oracle enforces
  depth, node and size limits before a value graph exists.
- **Stable content addressing** — a canonical byte form gives every program fragment an
  identity, which is what makes incremental and distributed builds skip work. *This* is
  where the compile-time win actually lives, and it is a caching win rather than a parsing
  win.
- **One front end per surface, not one parser per language.**

**Gate:** no phase in §7 may claim a compile-time improvement without a measured
before/after on a declared workload, separating parse, analysis, optimization and codegen.
The existing PERFORMANCE_AUDIT discipline applies unchanged.

### 3.2 "Compiles perfectly on the first try"

A schema guarantees **well-formedness**. It does not guarantee well-typedness, semantic
validity, termination, memory safety, or absence of undefined behaviour. A JSON tree that
satisfies an ASN.1 schema can still call a function with the wrong arity if arity is not in
the schema, divide by zero, deadlock, or leak.

The accurate claim is: *a schema-bound representation makes an entire class of errors —
syntax and structural malformation — unrepresentable, and moves the remaining classes to
the verifier where BCIR already checks them (R1–R25).* That is a strong claim. It should be
made instead of the stronger one.

### 3.3 "Programs can safely alter themselves at runtime"

This is the highest-risk item in the proposal and it collides with commitments the JSON
roadmap has already made: §1 excludes JSON from interrupt/DMA/submission paths, §6.3
separates integrity from authenticity, and §9 defers execution authorization to a trusted
loader involving signatures, revocation policy, relocation validation and W^X.

Self-modification that *emits* a new plan is tractable. Self-modification that *executes
new machine code in-process* is a code-signing and W^X problem that no schema solves. See
§6 for the staged model that keeps the useful half and gates the dangerous half.

## 4. The three deficiencies, and the mechanisms that answer them

The proposal identifies these correctly. Each already has a known answer; two of them are
already built in this repository.

### 4.1 The reference problem — solved by X.681/X.682, not by inventing `$ref`

JSON is a tree; programs are graphs. The usual answers are JSON Pointer (RFC 6901),
JSON-LD `@id`, or content addressing. **BCIR should use none of them as its primary
mechanism**, because ASN.1 already has a *typed*, *resolvable*, *standardized* cross-
reference construct, and this repo has implemented it:

- **X.681 information object classes and object sets** (`bcir/asn1/ecn.py`,
  `bcir/asn1/schema.py::ObjectSetTable`) give a table whose rows are objects and whose
  columns are class fields — including **type** fields, so a row can name a *type*.
- **X.682 §10.19/§10.20 table and component-relation constraints** select a row from the
  value of a sibling component. This is a reference with a **resolution law**, not a
  convention.
- The repo already resolves these and records the result *alongside* the octets under
  `<name>.resolved` — an enrichment, never a replacement.

A program graph is then: nodes are objects in an object set; an edge is a table-constrained
open type resolved by a sibling's identifier. Three properties fall out that a `$ref`
scheme does not give:

1. **The reference is typed.** The table says what type the target is, so a dangling or
   mistyped edge is a schema violation rather than a runtime `null`.
2. **Unresolvable is a legitimate state.** X.681 §12.9 permits a peer to use an object
   outside an extensible set, so "I do not know this node" is ordinary traffic rather than
   corruption — exactly what a versioned program format needs.
3. **It already has a C twin path.** The resolution machinery is on the existing rails.

Content addressing (a SHA-256 of the canonical JER of a subtree) is the **secondary**
mechanism, for identity and deduplication across files — Unison's model. The two compose:
the table gives typed intra-program edges, the hash gives global identity.

**Answered in P1.** Cycles are held by a **flat node table with integer index edges** — the
shape LLVM bitcode and MLIR bytecode both use, and the guess above turned out to be the right
one. A mutually recursive pair is two ordinary rows; the JSON's depth is constant however
tangled the graph gets, which matters because §4.3's depth ceiling is 64 and a nested
representation of a thousand-node chain could not be read at all.

**And one half of the mechanism above did not survive contact with X.697.** The proposal that
a node's payload be an OPEN TYPE resolved through the table, with §12.9's extensibility making
an unknown node ordinary traffic, is only half right. X.697 §41 says an open type's encoding
*is* the contained value's encoding and — unlike XER §8.5 — gives **no hexadecimal fallback**,
so `jer.py` refuses an unresolvable open type outright. It is right to: there is no JSON
spelling for "some octets whose type I do not know". An open-type payload would therefore have
made an unknown node **unencodable**, which is the exact opposite of the property P1 needed.

The fix keeps the mechanism and moves the layer. The object set is still the typing authority
— it says what a `kind` must carry, and `resolve` checks both nodes and edges against it — but
the payload travels in ordinary declared components and resolution is an **enrichment** that
*reports*. So an unknown node decodes, re-emits byte-identically, and is reported as
unresolved with §12.9 cited; a dangling edge is likewise reported rather than raised. That is
what "unresolvable is a value, not a fault" has to mean if the document is to survive at all,
and it is the same posture the repository already takes when it records a resolved result
alongside the octets rather than in place of them.

### 4.2 Verbosity — solved by making the canonical form authoritative and the syntax a view

The proposal's own answer is right, and Unison proves it works. The discipline that makes
it work, and that must be written down before anyone builds an editor:

- The **canonical JER is the artifact**. It is what is hashed, signed, stored, diffed by
  the build system, and fed to the compiler.
- Every surface — the visual node editor, the sparse Lisp-like text language — is a
  **lossless projection** of it. Round-tripping surface → canonical → surface must be the
  identity on the canonical side, which is exactly the round-trip law `bcir-asn1c --check`
  already enforces for ASN.1 modules.
- Formatting, naming and layout live in a **side table**, never in the canonical form, so
  two programs that differ only in presentation are byte-identical and hash identically.

The minimal text language should be treated as a *typing shortcut with a printer*, not as a
language: no macro system, no syntax extensions, no semantics of its own. The moment it
acquires semantics the canonical form stops being authoritative.

**Recorded from building it (P5).** Both defects the surface shipped with had one shape: the
reader was *stricter than the canonical form*, so a document that decodes had no spelling.
An empty attribute name — `Attribute.name` is an unconstrained `UTF8String`, so `""` is legal
— was refused as "a colon with no name after it"; and `roots` was treated as a keyword *by
position*, which left a node whose `kind` is literally `roots` printing as text the reader
then rejected. The lesson generalizes past this file: a surface is lossy exactly where it is
stricter than the form it projects, and the convenient rule that makes a parser simpler is
the usual way that strictness gets in. The keyword is now disambiguated by the following
**token** — a node's second token is always its quoted label — rather than by where it sits.

The same reasoning is why an out-of-range edge target is *accepted* by the reader: P1 already
decided a dangling edge is an `EdgeFault`, a value, so refusing it in the surface would make
a decodable document unspellable while looking like a safety check.

### 4.3 Scope and lifetime — BCIR already has these laws; JER must project them

The proposal is right that raw JSON has no notion of ownership or time, and right that
explicit block ownership is needed. But this is not a gap to fill from scratch:

- **R21** is a pointer-lifetime law (use-after-free / double-free) carried on the dialect as
  `#bcir.lifetime`, with `malloc`/`free` events stamped for it.
- **R19/R20** are synchronous-timing and clock-domain-crossing laws carried as
  `#bcir.timing`.
- Effects, ownership discipline and checked arithmetic are already verifier concerns.

So the JER program schema does not *invent* `{"block": "local", "variables": [...]}`; it
**projects the attributes the dialect already carries**, and R24's additive-projection rule
is what makes that projection checkable. The proposal's instinct is correct and the work is
smaller than it looks.

## 5. Control flow as a cost problem — the technically strongest part of the proposal

Mapping conditionals and loops into the tropical (min-plus) semiring is sound, and it has a
precise formulation worth recording so it is implemented as mathematics rather than as
heuristics.

In the min-plus semiring \((\mathbb{R} \cup \{+\infty\}, \min, +)\), with a control-flow
graph whose edges carry the twelve-axis cost projected onto a scalar objective:

- **Straight-line composition** is semiring multiplication: costs add along a path.
- **A conditional** is semiring addition: `min` over the alternative paths. An `if/else`
  is exactly a two-term min, which is why the mapping is natural rather than forced.
- **A loop is the semiring closure** — the Kleene star \(a^* = \min_k a^{\otimes k}\).
  This is where the proposal's "weighted cycles" becomes concrete: over min-plus, the
  closure of a cycle with non-negative weight is the empty path (cost 0), and a cycle with
  *negative* weight diverges to \(-\infty\), which is the algebra telling you the cost model
  is wrong rather than that the loop is free.
- **Optimal unrolling** is then a **minimum mean cycle** problem — Karp's algorithm — not a
  search. The mean cycle weight gives the steady-state cost per iteration, and the unroll
  factor that minimizes total cost given prologue/epilogue overhead follows from it.
- **Shortest path with non-negative costs** is Dijkstra; with the general case it is
  Bellman–Ford, and Bellman–Ford's negative-cycle detection is precisely the divergence
  check above.

Two honest limits to record now:

1. **Min-plus optimizes a path, and a program is a distribution over paths.** Without
   profile weights or branch probabilities, `min` picks the *best case*, which is the wrong
   answer for expected cost. The cost model needs edge probabilities — which makes it a
   Markov chain expectation, not a shortest path — or an explicitly worst-case objective
   (max-plus) for real-time work. **The semiring must be a declared parameter of the
   objective**, not a fixed choice. *In P4 it is:* `Semiring.MIN_PLUS` and
   `Semiring.MAX_PLUS` are both reachable and the caller names which. No expected-cost
   objective is offered, deliberately — expectation needs branch probabilities the cost
   model does not carry, and offering one would be the overclaim this limit warns about.

   A deflating result worth recording from the implementation: for a loop whose body cost is
   unaffected by unrolling, the `iterations × mean` term **cancels**, so the optimum is
   bounded entirely by the prologue/epilogue overhead. A pass reporting a large win from
   unrolling a body it did not change would be reporting an artefact.
2. **This is a legality-preserving optimization only if the graph is legality-checked
   first.** Phase H's laws apply unchanged: legality first, two-truth, canonical or
   excluded.

## 6. Self-modification: the staged model

The useful half of the proposal is achievable; the dangerous half is a loader problem. The
staged model:

```text
running program
  -> emits a candidate subtree as canonical JER            (data, not code)
  -> schema validation + R1-R25 verification               (legality, not cost)
  -> K_BCIR selection over the candidate set               (cost, gated by legality)
  -> compilation to a native artifact                      (offline-equivalent path)
  -> signing + trusted-loader admission                    (authority)
  -> generation-tagged swap, quiescence, rollback          (liveness)
  -> execute
```

Every arrow is an existing or already-planned BCIR stage. The properties that make it
"safe" are **not** properties of JSON:

- **W^X**: the emitting program never writes executable memory. It writes *data*.
- **Verification precedes admission**: a subtree that fails R1–R25 is never compiled.
- **Signing separates integrity from authority** — the JSON roadmap §6.3 already insists on
  this distinction, and self-modification is where conflating them would be fatal.
- **Generation-tagged handles and quiescence** are the same requirements the driver roadmap
  §9 already lists for live replacement; nothing about self-modification makes them
  cheaper.

**The one-line honest summary: a program can safely propose new logic at runtime. It cannot
safely admit it without a trusted loader, and no schema changes that.**

**Built (P6).** [`staged.py`](../bcir/asn1/staged.py) is that loader. Two things are worth
carrying forward from writing it.

The first is that **every separation above wants to be a type**. A discipline that lives in
review comments is one refactor from being gone, so `Proposal` has exactly two fields and no
method that produces an `Artifact`; the loader holds the key and is the only producer. W^X
then stops depending on every future caller choosing correctly.

The second is that **a signature must cover the label, not only the payload**. Signing the
compiled code alone leaves a signed artifact that can be re-tagged with another generation or
another proposal's digest and still verify — it would be attesting to something nobody
admitted. The HMAC here covers generation, proposal digest and code together. It is still
symmetric, and therefore proves possession of the loader's key rather than the identity of a
signer; a deployment where the proposer and the admitting authority are different principals
needs asymmetric signing, and the key cannot live on both sides.

Rollback moves the generation **forward**. Reusing the old number would let two different live
states share a tag, and a tag that does not identify a state cannot decide whether a caller is
holding a stale handle — which is the only thing generation tags are for.

## 7. Phase ladder

Each phase depends on the JER phases in the sibling roadmap and inherits their gates.

| Phase | Deliverable | Exit gate | Depends on |
|---|---|---|---|
| **P0 — this note** | Prior art, corrections, mechanism selection, and a recorded scope boundary | Docs governance green; no implementation claim added | — |
| **P1 — graph representation** | **Landed** ([`graph.py`](../bcir/asn1/graph.py)): a flat node table with **integer index edges** — the LLVM-bitcode / MLIR-bytecode shape — X.681/X.682 typing as an *enrichment*, and cycle-safe content addressing | **Met.** A mutually recursive pair, a self-loop and a 1000-node chain all round-trip byte-identically; every edge is typed through §10.19 row selection and a mistyped one is named; an unknown node kind and a dangling edge are both **values** that still re-emit. See §4.1's correction below | J2 |
| **P2 — IR projection** | **Landed** ([`graph.py`](../bcir/asn1/graph.py)): `dialect_to_graph` / `graph_to_dialect`, projecting the `bcir.asn1.*` dialect into the P1 node table | **Met** over all 26 law fixtures, legal and illegal alike: the dialect survives the graph round trip, the JER is byte-identical, and `MLIR -> graph -> JER -> graph -> MLIR` composes with J4 part 2's text rail. A component now points at its type with a real **edge** rather than a name, which is what makes mutually recursive types representable | J4, P1 |
| **P3 — scope and lifetime projection** | **Landed** ([`program.py`](../bcir/asn1/program.py)): a `Module`'s phases, claims, resources, `Timing` and `Lifetime` projected through the P1 node table, with phase and claim **order carried structurally** in the ordered edge lists | **Met** over all 12 corpus programs, plus fixtures that trip R19, R20 and R21 and one that is *excused* by a barriered hazard — because a projection that made every module more illegal would be just as wrong and easier to ship. Absent timing and all-default timing stay distinguishable, or every untimed claim in the repository would fall under the timing laws | P2 |
| **P4 — cost-graph execution** | **Landed** ([`tropical.py`](../bcir/kbcir/tropical.py)): composition adds, a conditional is `min`, a loop is the Kleene closure, Karp (1978) gives the minimum mean cycle, and `Semiring` is a caller's parameter | **Met.** A negative cycle raises `NegativeCycle` **naming the cycle** rather than returning −∞ or clamping; Karp reproduces hand-derived means (6/2, 3/3, and an exact 4/3 as a `Fraction`); the unroll optimum is derived from the overhead term and checked against hand arithmetic. Legality-first is checked *structurally* — a test asserts the module never references the verifier | P2, phase H |
| **P5 — surface projections** | **Landed** ([`surface.py`](../bcir/asn1/surface.py)): the sparse text language as a lossless view, with `Presentation` — aliases, comments, indentation — returned as a *separate* record the canonical form cannot see | **Met.** Identity on the canonical side over all 12 corpus programs and all 26 dialect modules; three deliberately divergent spellings of one program give byte-identical JER and one content address; both directions are iterative, checked by round-tripping a 1000-node chain under a recursion limit of 80. The visual editor is not built — the text surface is what the gate names first and what the node table can be checked against | P1 |
| **P6 — staged self-modification** | **Landed** ([`staged.py`](../bcir/asn1/staged.py)): §6's pipeline with a `TrustedLoader` holding the only key, and each separation carried by a *type* rather than by convention | **Met.** `Proposal` has two fields — octets and an origin — and no path to an `Artifact` except through the loader. "Verification precedes compilation" is checked with a compilation **counter**, not by reading statement order, over P3's own R20 fixture. "Unsigned refused" is checked with an artifact whose SHA-256 is perfectly correct, because the failure worth catching is a loader accepting integrity as authority. Rollback moves the generation *forward* and re-signs. **Stated limitation:** the signature is HMAC, so it proves possession of the key, not the identity of a signer | P3, P4, trusted loader |

**Registry.** The proposal's "globally unique schema registry, cryptographically signed and
versioned" spans all phases and needs its two halves kept distinct:

- **Identity** is a content address — the SHA-256 of the canonical JER. It is free,
  immutable, and needs no trust root. `jer_plan.py` already produces exactly this for a
  schema descriptor.
- **Authority** is a signature over that address, and it needs a trust root, a revocation
  policy and an expiry story. TUF and Sigstore are the reference designs. **A CRC or a
  bare hash is not a signature** — the JSON roadmap risk register already names conflating
  these as a tracked risk, and a program registry is where it would do the most damage.

## 8. What this changes in the existing roadmaps

- **J3 (scalar C twin) — unchanged, and more strongly motivated.** If JER carries programs,
  the bounded C parser is on a far more important path. Proceed as planned.
- **J4 (law and execution lowering) — scope extended.** J4 currently plans a *one-way*
  JER → claims lowering. P2 needs the projection to be **bidirectional and commuting**
  against the `bcir.*` dialect. That is an addition to J4's exit gate, recorded now so it
  is designed for rather than retrofitted.
- **Phase H (K_BCIR selection) — gains a consumer.** P4 makes control flow a candidate
  dimension alongside encoding rules and lane width. Its laws are unchanged.
- **ECN — unaffected.** Nothing here reopens user-defined ECN classes. If anything, P2
  strengthens the closure: a program representation projected from the dialect is *ordinary
  BCIR lowering*, which is precisely what the reduction gate said was sufficient.

## 9. Risk register

| Risk | Control / stop condition |
|---|---|
| "Source is the AST" is treated as novel and re-invents Lisp or MLIR badly | §2's prior-art table is the required reading before P1; borrow, do not reinvent |
| Parsing-cost claim drives design | §3.1's gate: no compile-time claim without a measured, staged before/after |
| "Compiles first try" is read as a correctness guarantee | §3.2 fixed in this note; verifier laws remain the correctness story |
| A `$ref` scheme is invented alongside X.681/X.682 | P1 exit gate requires every edge to be typed and table-resolved |
| Cycles force a tree schema into knots | P1 answers this before any code; flat node table is the presumed shape |
| Text surface acquires semantics and becomes the real language | P5 gate: canonical is authoritative, surface round-trips to identity, no macros |
| Min-plus optimizes best case and is read as expected cost | §5's limit 1: the semiring is a declared objective parameter, and probabilities are required for expectation |
| Self-modification is described as safe because the schema is strict | §6: W^X, verification before compilation, signing before admission — none supplied by a schema |
| Registry integrity is described as authenticity | §7: content address ≠ signature; separate trust root required |
| The programme grows without a consumer | Each phase names a concrete artifact it must reproduce; P2 must reproduce the existing dialect exactly |

## 10. References

- Unison, content-addressed code and text-as-a-view:
  <https://www.unison-lang.org/docs/the-big-idea/>
- MLIR bytecode format (the closest existing analogue to the proposal's artifact):
  <https://mlir.llvm.org/docs/BytecodeFormat/>
- WebAssembly binary format: <https://webassembly.github.io/spec/core/binary/index.html>
- RFC 6901, JSON Pointer: <https://www.rfc-editor.org/rfc/rfc6901.html>
- Karp, *A characterization of the minimum cycle mean in a digraph* (1978) — the basis for
  §5's unroll derivation.
- Baccelli, Cohen, Olsder and Quadrat, *Synchronization and Linearity* — min-plus algebra
  applied to discrete event systems.
- TUF, *The Update Framework*: <https://theupdateframework.io/> — registry trust root and
  revocation.
- Rec. ITU-T X.681 and X.682 — the information object and table-constraint mechanisms §4.1
  builds the reference system from.
