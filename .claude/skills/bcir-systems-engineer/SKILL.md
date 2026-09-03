---
name: bcir-systems-engineer
description: >-
  Operate as a BCIR systems engineer in the Binary-Correspondence-Intermediate-Representation
  repository. Use this skill for ANY substantive work in this repo — designing, implementing,
  reviewing, testing, or landing a change on any rail (`bcir/` Python oracle, `mlir/` law,
  `runtime/c` / `runtime/cpp`), answering architecture or status questions, planning a slice,
  writing a PR, or running validation. Trigger on mentions of: BCIR, K_BCIR, GEM, GEM+,
  StreamPack, BCAB, cfront/bcir-cc, BCIRQ8, verifier laws / R-laws (R1–R25), parity,
  oracle/law/twin, TMSAO, ASN.1/ECN/DER/BER/PER/OER/XER/JER, telemetry, driver/UART/virtio
  roadmap, HAM, hardware channels, or model/ML substrate work — even when the request
  doesn't say "skill". Load it BEFORE editing code or docs here.
---

# BCIR systems engineer

How to think and work like the engineers who built this repository — distilled from the
full PR record (#2–#749), the commit history, the docs tree, and the repo's own audits
— including the 42-round adversarial campaign of #749, whose 240 graded findings are
distilled into `docs/security/laws.md`. This skill is the *method*; the repo docs are
the *law*. When they conflict, the docs win, in this authority order:

> LangRef → generated/static evidence and implementation → current-state audit →
> master roadmap → companion roadmaps → research notes → development history.

## 1. What you are working on

BCIR is a registry-first, phase-ordered, lane-typed, **cost-governed correspondence IR**:

```
K_BCIR(G | H, Θ) = min_{π ∈ Legal(G,H)} M(π, Θ)   s.t.   R(π, Θ) ⪯ B(H, Θ)
```

Legality (verifier laws **R1–R25**) is decided *before* cost; a 12-axis integer/Q8 cost
vector prices legal candidates; GEM hydrates the selected plan into StreamPack for
execution. BCIR is a planning/verification/artifact/runtime layer **above** resident
LLVM/GCC/vendor toolchains — it deliberately owns no general instruction selector,
register allocator, or linker (see `docs/BCIR_NATIVE_OBJECT_GATE.md`).

| Rail | Role | Boundary to respect |
|---|---|---|
| `bcir/` | executable Python conformance oracle (dependency-free core) | `bcir.hosted.*` is an import-quarantined opt-in extra, never the default path |
| `mlir/` | ODS/TableGen/C++ law rail: `bcir-opt`, R1–R25, optimizer passes, IRDL projection | `bcir-aot` is *partial* preparation; targets LLVM/MLIR **22** — one coherent major |
| `runtime/c/` | production C rail: freestanding StreamPack/codecs, the `bcir_cfront.c` twin, hosted model/compiler tools, RuntimeChannel | three memory classes (freestanding heap-free / hosted allocator-injected / driver handles+offsets) — `docs/languages/C_MEMORY_DISCIPLINE.md` |
| `runtime/cpp/` | narrow orchestration seam above the C ABI | single-node real; distributed/dynamic backends are honest stubs |
| `llvm-training/` | agent training corpus | **not the IR**; neither may depend on the other |

A change earns a rail through **differential parity, not assertion** (`docs/PARITY.md`).

## 2. Session start

1. Invoke the `token-optimization` skill and read `.claude/context/BCIR_DIGEST.md` first —
   it replaces ~500k tokens of exploration. If `build_digest.py --check` says STALE at
   session end and you learned something durable, regenerate it.
2. Counts live **only** in generated `docs/STATUS.md` (`python tools/docs/gen_status.py`).
   Never hard-code a test/op/pass count into prose — hard-coded counts are how the
   580/615/631 drift happened, and a CI gate now rejects it.
3. Route by need using the reading map in `docs/ONBOARDING_DEEP_DIVE.md` §11. The
   short version: laws/BCIRQ8/ASN.1 → `BCIR_LANGREF.md`; implementation truth →
   `REPO_CURRENT_STATE_AUDIT.md`; execution order → `BCIR_MASTER_ROADMAP.md`;
   correspondence → `PARITY.md`; drivers → `kernel/BCIR_DRIVER_KERNEL_ROADMAP.md`;
   ML → `machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md`; performance/optimality →
   `research/BCIR_GEMPLUS_ROADMAP.md` + `PERFORMANCE_AUDIT.md`; how-it-was-built →
   `DEVELOPMENT_HISTORY.md`.

## 3. Non-negotiable invariants

These are load-bearing across every subsystem; a PR that bends one will be reverted.

1. **Legality precedes optimization.** R-laws reject before K_BCIR prices. Learned or
   measured data may rank and calibrate; it never becomes a legality verdict, never
   steers in-flight work, never runs on the execution hot path (two-truth quarantine,
   verified by `bcir/tests/test_hot_cold.py`).
2. **One semantic truth, multiple realizations.** Prototype in the oracle, port to the
   production rail, lock with parity gates (bit-exact scores, byte-identical artifacts,
   structural digests). Never extend the prototype as if it were the product — that rule
   became explicit in PR #266 after it was violated.
3. **Frozen ABIs are frozen.** StreamPack v1, BCAB v1, BTLM v1, the telemetry frame:
   new capability lands *append-only* (v2/v3 records) or as an *additive* projection
   (the ASN.1 modules). A byte change is a doc + both-rail change landed together.
4. **DER out, BER in.** BCIR digests what it emits, so it never emits an encoding whose
   octets a peer may choose. Every new encoding rule names its canonical variant or is
   decode-only.
5. **Non-disturbance for new laws.** New optional semantics ship vacuous-by-default:
   the whole existing corpus must verify byte-identically with the new law wired in
   (this is how R17, R19–R21, R24, R25 landed without perturbing any plan).
6. **Artifacts are immutable within a generation** — content-addressed,
   generation-tagged, stale on any relevant input change; promotion only at quiescent
   generation boundaries with rollback. No on-stack replacement, ever (recorded
   decision, `research/BCIR_ADVANCED_TECHNIQUE_TRIAGE.md` D4).
7. **Unsupported work fails honestly.** Partial AOT, modeled channels, compiler
   fixtures, fallbacks, and hardware-gated code are labeled as such. A clean
   tool/hardware skip is *never* reported as a passing measurement.

## 4. The engineering method (how every arc here was actually built)

- **PR-sized slices inside named ladders.** Declare the ladder in a roadmap doc first
  (L1–L8, RT1–RT7, G1–G8, A/B/M/E/T/SEG/U/D series, ASN.1 phases A–H, JER J1–J6, ECN
  slices A–G3, GEM+ G0–G10), then land one gateable slice per PR with the slice ID in
  the title. If a slice spans rails, write the build reference doc *first*
  (`docs/PER_DECODER_HANDOFF.md` is the model: "a finding written down before the code
  is worth more than a half-finished branch").
- **Verification-first PR bodies.** End every substantive PR body with a Verification
  section quoting *exact* gate commands and outputs (pass counts, sanitizer results,
  skip reasons). "Ran the tests" is not evidence; `3334 passed, 0 failed` is.
- **Generated, adversarial verification over curated pins.** Prefer a generated
  differential (Python↔MLIR plans, C-twin vs oracle vs Clang, fault-injection where
  every law must catch its own injected violation, libFuzzer+ASan/UBSan on every
  trust-boundary decoder) to a hand-picked worked example. Pins anchor; differentials
  prove.
- **Tests must drive the failure, not the success.** A test asserting the honest path
  would have passed against all 15 defects of the 2026-08-12 audit. For every checker,
  write the negative case: inject the violation, assert the specific refusal. Prove
  gates can fail: when you add or speed up a gate, break the thing it guards and watch
  it fire (this caught a `grep -q`/SIGPIPE/pipefail CI gate reporting a *found* pattern
  as a failure).
- **A construct absent from the corpus is untested, however many tests run over it.**
  All four shipped encoder bugs of #687–#689 were found by adding constructs to the
  corpus, not by reading code. Validate against the standard's own worked examples,
  not round trips — "a round trip passes when encoder and decoder share the same
  misreading."
- **Ask whether two inputs differing only in the new thing differ in the digest.**
  This one question caught silent drops nothing else caught (ECN slices). Pin exact
  sets, not membership.
- **Expensive directions go behind written GO/STOP gates** (native-object gate, the
  #592 feasibility verdict *against* a general native backend, the ECN §6 reopening
  condition — which was later met, executed, and documented). Retract false wins;
  label modeled numbers modeled.
- **Reconciliation waves.** When docs fall behind code, land a dedicated docs PR that
  reconciles them; landing notes go in `DEVELOPMENT_HISTORY.md`, *never* back into the
  roadmaps (roadmaps own the future, history owns the past).

## 5. Placing a change

- **Semantic change** → oracle first, then the applicable law/production twin, plus a
  differential regression, in one PR when feasible.
- **Cross-rail hashes/content addresses** (e.g. `hash_module`/`hash_target` vs
  `BCIRVerifyPass.cpp`'s recomputation) must change **on both rails in one commit** —
  a one-rail change makes the rails silently disagree about a content address, which
  is worse than the gap it fixes (this is why the provenance-hash memory-hierarchy gap
  was deliberately left open in the 2026-08-12 audit).
- **New C sources feeding a gate**: update `tools/c/check_runtime.sh`'s gate block AND
  the Python harness source list together (`native_bench._SOURCES`). This exact drift
  broke PR #719 mid-session; `test_the_gate_and_the_harness_link_the_same_sources`
  guards it now — extend that pattern for new gate/harness pairs.
- **New test files** must be registered in `bcir/tests/run_all.py`
  (`test_registry_complete` fails otherwise) — and registration is a claim about the
  **shipped package**, not only the checkout. If a test needs an asset, ask first
  whether the asset should ship (`[tool.setuptools.package-data]`) and whether library
  code reads it through `importlib.resources` rather than a checkout- or
  CWD-relative path; only then consider `_REPO_ONLY_MODULES`, and scope the entry to
  the dependent **test**, never the whole module. Two rounds of #749 were spent on
  exclusions that were hiding a broken wheel and a mis-resolved resource — 98 runnable
  tests between them (L21).
- **Editing large modules**: use exact-match edits, never scripted `str.replace` over a
  large file — anchors that match elsewhere have caused real mis-targets here.
- **Read two rails out of their own sources** rather than mirroring into a third list;
  a mirror list *will* drift.

## 6. Validation and publication

Local hosts are bounded: **at most two workers**, heavy gates serialized, never
concurrent Python/C/model/MLIR gates, never unbounded fuzzing, inference, emulation,
or nested build loops, no local ARM emulation to fake a matrix cell.

```bash
python -m bcir.tests.run_all --tier quick -j 2      # bounded; hides toolchains on purpose
python -m bcir.tests.run_all --tier thorough -j 2   # restores the real host toolset
bash tools/c/check_runtime.sh                       # strict C gates + sanitizers
bash tools/cpp/check_handoff.sh
bash tools/wsl/check_passes.sh                      # ONLY with a coherent LLVM/MLIR 22 toolset
bash tools/irdl/check_corpus.sh
python tools/docs/gen_status.py --check             # + regenerate STATUS.md LAST after test changes
python tools/docs/check_links.py
git diff --check

# the assurance rails (#749) — each one a required CI job
python tools/security/scan_secrets.py
python tools/security/audit_dependencies.py
python tools/security/audit_tool_boundaries.py
python tools/security/run_decoder_campaign.py --mutations 24 --fuzz-runs 200 --fuzz-seconds 8
python tools/security/run_malformed_differential.py
python tools/security/independent_review.py --self-check
```

- Quick tier *intentionally* hides compiler/toolchain capabilities and expects explicit
  skips; don't "fix" those skips. Thorough must use one coherent LLVM major (22). On a
  host with only LLVM 18, the MLIR rail is an **honest documented skip** — CI's
  `mlir-rail-validate` owns it; do not touch ODS/pass sources you cannot build.
- Before commit/PR: read `.github/workflows/ci.yml`, map every affected job/matrix
  cell, run what the host supports, delegate the rest (Windows, native ARM, long fuzz,
  model gates) to Actions. Record exact commands and results in the PR body. Confirm
  tests leave tracked files unchanged.
- After pushing, **wait for the complete Actions run**; a pending or failing required
  check means the PR is not done.
- **Do not stack PRs on kept base branches.** GitHub only auto-retargets a stacked PR
  when its base branch is deleted on merge; this repo twice merged slices into a kept
  stack base instead of `main` (#708/#709 and #713–#715), each needing a recovery PR
  (#710, #716) that landed the identical tree with `merge-base --is-ancestor` proof.
  Target `main` directly, or delete head branches on merge.

## 7. Measurement discipline and certificates

- Classify every metric row: `exact` (deterministic — gates anywhere, 2% band),
  `ratio` (timed ratio — wide 25% band), `wall` (absolute ms — INDICATIVE only, never
  gates). A slice gates on its `exact` rows. The GEM+ baseline harness is frozen:
  `python tools/perf/gemplus_baseline.py --list|--compare`.
- Optimality claims use the TMSAO ladder (`research/BCIR_GEMPLUS_ROADMAP.md` §2):
  TMSAO-1 exact optimum / TMSAO-2 bounded gap / TMSAO-3 best measured / TMSAO-4
  heuristic-no-claim. **Everything BCIR emits today is TMSAO-4** until G4's
  lower-bound stack lands. No optimality claim on a row with no lower bound; no
  sublinear claim on an Ω(n) operation without naming the admitted work that changed.
- Per-slice analysis protocol: on GAIN report magnitude + remaining headroom + what the
  residual gap is made of; on NO-CHANGE find which of mis-assigned / cancelled /
  already-at-the-bound (a proved bound retires the row — that's a success); on
  REGRESSION the slice does not land until explained.
- **No silicon certificate from a virtualized host.** This container has no PMU
  (`ENOENT` on `perf_event_open` regardless of privilege — see
  `docs/BCIR_TARGET_ACCESS.md`); calibration records carry refusal predicates and
  `calibration.py` correctly refuses shared runners for frozen tables. Privilege is
  not capability. Widen timing intervals to the clock quantum (the S24+ 52 ns timer
  lesson); record an unreadable counter as null, never zero.

## 8. Security-engineering lessons (paid for; do not re-buy)

**The registry owns this now.** `docs/security/laws.md` holds **21 gate-authoring laws
(L1–L21)**, each with its witness tests and a C/C++ port note, derived from 240 graded
findings across #749's 42 review rounds. Read it before writing or reviewing a gate;
`bcir-cicd` is the companion skill for the pipeline half. What follows is the part of
that experience that generalizes beyond gates, and the shape of the registry so you can
navigate it:

- The four laws that produced the most findings are the four to design against first:
  **L5** everything committed is scannable data (tree names, symlink targets, archive
  members, every text encoding), **L3** bounds live where the resource commits (a `stat`
  answers about the past; the *read* is the commitment), **L1** every exit is a verdict
  (a traceback is a lost finding *and* a lie about the outcome), **L11** a witness must
  hit the law it exists to test, on every rail.
- **L14 — one predicate per repeated defect — is the campaign's own summary.** The
  dominant defect shape was never a novel bug: it was a mechanism landed on two rails
  out of three. Fixing a repeated defect per-site is how there came to be N of them; a
  shared predicate must also be **total**, because a precondition living outside it is a
  defect held in reserve for its second caller.
- **A budget that measures the wrong resource is not a budget** (L3). Empty concatenated
  compressed members advance a bytes-out cap by zero while each one costs a decompressor;
  a byte-statistics heuristic encodes the script of the text it was tested on.
- **A report is an egress surface** (L7). Findings carry fingerprints, never values; a
  wrapped tool's stdout is a report field; a reviewer *quotes the code it reviews*, so
  its findings are the field guaranteed to carry the secret. Redact by position, and
  with the same predicate that decides what to report.
- **A skip is where a shipping defect hides** (L21). An exclusion converts "the artifact
  is broken" into "this does not run here", and in a green run those read identically.
- **A `--require-X` flag claims the rail RAN**, not that it was discoverable at startup
  (L2); and a witness that asserts a *substring* of what should have been removed passes
  on output that violates the law (L11) — that one was found by CodeQL, not by the loop,
  because a review pair is structurally blind to its own assertions.
- **Two failure classes** still cover most defects here. Class A, *a second spelling*: a
  strict decoder accepting bytes the canonical rules spell exactly once — every
  canonicality fix must assert both halves (bad spelling refused AND the encoder's own
  output still round-trips). Class B, *a vacuous check*: a law returning clean over
  something it never examined (an empty StreamPack satisfied all of R10). When adding
  a checker ask: what input makes every loop in it iterate zero times?
- **Never parse a wire format with a host-language parser.** `float()`/`int()`,
  `str.isdigit()`, and regex `\d` without `re.ASCII` accept a larger language than any
  ITU clause (PEP 515 underscores, `inf`/`NaN`, Unicode digits). Enforce the format's
  own grammar *before* conversion (JER's scanner is the model). Octet rails that
  decode contents as ASCII first are safe; **text rails (XER/JER) have no earlier
  gate** and are where this bites.
- **Five local repairs is how there came to be five defects** — fix a repeated
  validation with one shared predicate (`values.py:is_ascii_digits`), not per-site.
- **`repr` is never a content address.** An `object()` sentinel repr embeds a heap
  address; a dict repr follows insertion order. Digest canonical octets — they are
  canonical by construction.
- Decoders are total on untrusted input: bounded nesting (explicit stack, no
  recursion), bounded declared lengths *checked in a type wide enough on 32-bit
  targets* (the `size_t` length-wrap lesson), parent length authoritative over
  children, distinct statuses for iteration-end vs malformed end-of-contents. Fuzz
  every trust boundary under ASan/UBSan.
- Deliberate unsigned wraps in the C twins are commented and well-defined;
  `-fsanitize=integer` findings there need the recorded suppressions, not "fixes".

## 9. Current state and open edges (2026-09, post-#749)

Landed and gated: R1–R25 dual-rail with negative fixtures; the C-front twin
(driver-subset C23, ~21 fuzzer-found miscompiles turned into gates; `_Decimal*`
blocked); frozen StreamPack v1 + v2/v3; BCAB v1; the ASN.1 portfolio (X.680–X.697:
DER/BER, PER, OER, XER, JER, ECN complete with R24/R25; cost-governed encoding
selection with measured native tables and two admitted calibration targets); BCIRQ8 +
TinyLlama standalone-C parity; the bounded model labs; HAM metadata planning; GEM+
G0 landed and G9 half-landed (alias facts to LLVM — the emitter no longer lies with
blanket `noalias`); and the maintained **assurance rails** (`tools/security/`: secret
scan, dependency audit, tool-boundary audit, decoder campaign, malformed differential,
fail-closed independent review), each a required CI job, with their laws registered in
`docs/security/laws.md`. Two repo-wide facts came out of that arc: the Python floor is
**3.11** with a `python-floor` job that refuses any other interpreter, and the wheel is
a tested artifact — the suite runs from an installed package, so a resource a test reads
must actually ship.

Open, deliberately: resident UART/virtio drivers and UAPI v1 (UART + virtio-blk
evidence must come first); live telemetry transports; GEM+ G1–G8/G10 (one schedule
artifact, delta pricing, digest-once, lower bounds → first TMSAO-2, exact memory,
typed regions, movement, escape analysis); the provenance-hash memory-hierarchy
closure (two-rail); R11 per-resource generation vectors; whole-decoder Q4, GPU
execution, production serving. Check `REPO_CURRENT_STATE_AUDIT.md` before promising
any of these exists.

## 10. Writing it down

- PR titles carry the slice ID; bodies end with Verification (exact outputs).
- Counts → `STATUS.md` (generated, regenerated **last**). Chronology →
  `DEVELOPMENT_HISTORY.md`. Execution order → roadmaps. Honest snapshot →
  `REPO_CURRENT_STATE_AUDIT.md`. Docs carry machine-checked claim markers
  (`test_docs_claims`) — a summary that stops being true fails CI, so update the
  claim's predicate with the claim.
- Say "measured" only for real host/silicon evidence with provenance; "modeled" for
  models; name every skip. The repository's credibility is the product: a certificate
  is only worth what its refusal conditions cost.
