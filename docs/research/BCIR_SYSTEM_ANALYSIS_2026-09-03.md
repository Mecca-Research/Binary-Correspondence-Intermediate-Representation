# BCIR whole-system analysis and next-step recommendations (2026-09-03)

> **Status.** Dated, non-normative research snapshot taken at `main` = `d52558c` (the merge of
> PR #750), package version `0.2.0`. It sits at the *research notes* level of the authority
> order (LangRef → generated/static evidence → current-state audit → master roadmap →
> companion roadmaps → research notes → history). Every count below is a **measurement
> made on 2026-09-03**; the live inventory is generated [`STATUS.md`](../STATUS.md), and
> the honest capability snapshot remains
> [`REPO_CURRENT_STATE_AUDIT.md`](../REPO_CURRENT_STATE_AUDIT.md). This document does not
> change any contract; it records what was read, what was executed, what disagreed with
> what, and what to build next.

**Method.** The whole tree was surveyed (three rails, tools, channels, the training corpus,
CI); every Markdown document under `docs/` was read (54 files, ~23.8k lines), the full PR
record (#2–#750) and the complete git history (1,759 commits) were reviewed, and every
local gate the host can run was executed, including the MLIR rail against freshly
installed LLVM/MLIR 22.1.8 **and** 23.1.0 and the quick tier on Python 3.14.7 and
3.15.0rc2. Section 3 lists the exact commands and outcomes; section 10 records how the
toolchains were obtained on a network-restricted host so the next session can repeat it.

---

## 1. Executive summary

**Verdict.** BCIR is a coherent, unusually well-gated *planning, verification, artifact and
runtime layer* above resident toolchains. Its three rails (Python oracle, MLIR law, C
runtime) genuinely correspond: every gate the host could run is green, including the
thorough Python tier, the strict C rail with sanitizers and fuzzing, the docs-governance
gates, the six assurance rails from PR #749, and the compiled MLIR rail. The engineering
discipline visible in the record (prototype-then-port, differential parity over curated
pins, non-disturbance for new laws, frozen ABIs, generated counts, verification-first PR
bodies, a 21-law gate-authoring registry) is the project's real asset and is what makes
its claims checkable.

**The gap is between substrate and intelligence, and between docs and code.** The
optimizer, ASN.1 portfolio and ML organs are far ahead of the physical substrate they are
meant to steer: there is no resident driver, no live telemetry transport, no measured
silicon certificate, and every optimality claim is still TMSAO-4 (heuristic, no lower
bound). The documentation corpus (larger than the C runtime) has visibly drifted in
places: stale law ranges in ~20 active documents, a 3,190-line snapshot appended to the
normative LangRef by a direct commit, one fully stale hand-off note, and several
self-contradicting roadmap paragraphs. None of this is hidden; the repository's own
audits say the same thing. It is now the cheapest, highest-leverage work available.

**Top recommendations (details in §9).**

1. **Land a documentation-currency PR first** (one day of work): fix the stale `R1–R23`
   ranges, move the LangRef's appended system report to `docs/research/`, retire
   [`PER_DECODER_HANDOFF.md`](../PER_DECODER_HANDOFF.md), reconcile the ASN.1 roadmap's
   phase-G paragraphs and the four different R25 rule counts, and add a *law-range drift*
   check to docs governance so this class of drift cannot recur.
2. **Close the certificate spine** on the optimizer: GEM+ G1 (one schedule artifact) →
   G3 (digest once) → the two-rail provenance-hash closure (B7) → R11 per-resource
   generation vectors → G4 lower bounds, which yields the first TMSAO-2 certificate.
3. **Start the driver chain that unblocks 0.3b**: the version-zero telemetry triple
   (generated signal table, identity-carrying envelope, SPSC ring), then UART U0–U2. The
   0.3b draft is explicitly blocked on "the first direct driver package"; nothing else on
   the roadmap moves that gate.
4. **Decide the LLVM 23 question deliberately.** LLVM/MLIR 23.1.0 is released and CI's own
   comment says the MLIR rail "tracks the LATEST release". See §3.3 for what happened when
   the rail was built against 23 here.
5. **Make the dependency audit's advisory half real** (`pip-audit` or an equivalent in the
   `security-assurance` job); today it is `UNAVAILABLE/SKIPPED` locally and never runs in CI.

---

## 2. What the system is, by the numbers

BCIR realizes `K_BCIR(G | H, Θ) = min_{π ∈ Legal(G,H)} M(π, Θ) subject to R(π, Θ) ⪯ B(H, Θ)`:
verifier laws R1–R25 decide legality first, a 12-axis integer/Q8 cost vector prices legal
candidates under live pressure Θ and RCSP budgets, GEM hydrates the selected plan into a
frozen StreamPack, and resident LLVM/Clang/GCC/vendor toolchains do instruction selection,
register allocation and linking. The normative reference is
[`BCIR_LANGREF.md`](../BCIR_LANGREF.md); the correspondence contract is
[`PARITY.md`](../PARITY.md).

| Tree | Role | Size on 2026-09-03 |
|---|---|---|
| `bcir/` | dependency-free Python conformance oracle + import-quarantined hosted extras | 527 files; ~97k lines of non-test Python; 3,519 `test_*` functions in 254 files (~74k lines) |
| `mlir/` | ODS/TableGen/C++ law rail (`bcir-opt`, R1–R25, 37 passes, IRDL projection) | 186 files; ~15.4k lines of C++/ODS; 133 ODS ops; 117 fixture files with 300 `expected-error` markers |
| `runtime/c/` | freestanding StreamPack/codec/executor twin, the C-front compiler twin, hosted model tools, RuntimeChannel v1 | 303 files (301 sources/headers); ~31.3k lines of C, of which the `cfront_*.c` fixture corpus is 184 files |
| `runtime/cpp/` | narrow orchestration seam above the C ABI | 11 files; ~1.8k lines |
| `tools/` | validation scripts, docs governance, perf harness, security rails, model gates | 62 files; ~17.7k lines |
| `docs/` | LangRef, roadmaps, ABIs, audits, research | 54 Markdown files; ~23.8k lines |
| `llvm-training/` | agent training corpus, explicitly not the IR | 612 files (199 `.ll`); ~37.8k lines |
| `channels/` | pluggable channel descriptors | 5 JSON files |

Two facts about the shape matter for planning. The documentation corpus is now ~75% of the
size of the C runtime it describes, so it is a first-class maintenance surface. And the
test corpus is concentrated: `bcir/tests/test_c_cfront.py` (4,869 lines) and
`bcir/tests/test_security_assurance.py` (4,023 lines) together hold a large share of the
suite, and the largest production modules are the ECN rail (`bcir/asn1/ecn_syntax.py`
3,534 lines, `ecn_user.py` 3,157), the C-front lowering (`bcir/frontends/cfront/lower.py`
2,606; `runtime/c/bcir_cfront.c` 5,702) and the MLIR verifier (`BCIRVerifyPass.cpp`
2,686). Those five files are where review attention pays most.

---

## 3. Evidence: what was executed on this host

Host: 4 vCPU / 15 GiB, Linux 6.18, Python 3.11.15, Ubuntu clang/LLVM 18.1.3, GCC 13.3,
CMake 3.28, no PMU (`perf_event_open` → `ENOENT`), no RAPL, no cpufreq governor. Two
workers throughout; heavy gates serialized, as the repository requires.

### 3.1 Gates as CI runs them (all green)

| Gate | Command | Outcome |
|---|---|---|
| Quick oracle tier | `python -m bcir.tests.run_all --tier quick -j 2` | **3519 passed, 0 failed** (86 s) |
| Thorough oracle tier | `python -m bcir.tests.run_all --tier thorough -j 2` | **3519 passed, 0 failed** (5 m 49 s) |
| C runtime (StreamPack byte identity, ASN.1 twins, cfront sanitizer harness, C↔C++ seam) | `bash tools/c/check_runtime.sh` | `[c-runtime] ok` (3 m 18 s; 358 PASS lines). Clang ASan was absent at first, so the harness fell back to gcc ASan/UBSan + Valgrind and *said so* |
| cfront twin sanitizer sweep (after installing `libclang-rt-18-dev`) | `SANITIZE_SKIP_VALGRIND=1 SANITIZE_ENGINES_PARALLEL=1 bash tools/c/sanitize_cfront.sh` | exit 0 |
| Allocator-failure sweep under ASan/UBSan/LSan | `MEMORY_DISCIPLINE_SANITIZE=1 bash tools/c/check_memory_discipline.sh` | exit 0 |
| Binary trust-boundary fuzz (bounded) | `FUZZ_RUNS=20000 bash tools/c/fuzz_streampack.sh` | exit 0 |
| Docs governance | `gen_status.py --check`, `check_links.py`, `check_retired_paths.py`, `check_claims.py`, `import_graph.py --check` | current / no broken links / no retired paths / 5 claims hold / 80 cold organs, none eager |
| Secret scan | `python tools/security/scan_secrets.py` | PASS: 1781 tracked, 0 findings |
| Dependency audit | `python tools/security/audit_dependencies.py` | PASS: inventory asserted (6 expected, 0 mismatches); **advisory=UNAVAILABLE/SKIPPED** |
| Tool-boundary policy | `python tools/security/audit_tool_boundaries.py` | PASS: 534 files, 0 findings |
| Independent review contract | `python tools/security/independent_review.py --self-check` | PASS, fail-closed |
| Malformed differential | `run_malformed_differential.py` (and `--require-compiled` once `bcir-opt` was built) | PASS: 6 cases, 0 disagreements |
| Decoder campaign | `run_decoder_campaign.py --mutations 24 --fuzz-runs 200 --fuzz-seconds 8 --require-c` | PASS (StreamPack/BCAB/BCIRQ8; C campaign ran once compiler-rt was installed) |
| Generated Python↔MLIR differential | `python -m bcir.kbcir.differential -n 2000 --seed 1` | 2018 checks clean, 0 bugs, 0 coupling gaps; verifier differential 0 misses |
| Trust-boundary fuzz (oracle) | `python -m bcir.kbcir.fuzz -n 1000 --seed 1` | 0 findings |
| TMSAO cross-organ audit | `tools/perf/run_tmsao_audit.py --repeats 2` | 13 cases; correctness digest `649f621b…` |
| Perf budgets | `tools/perf/check_budgets.py` | ok; `baremetal=False` so floors waived by design (gather 4.02×, reduction 11.68×, strided 1.73×, dense 1.00×/0.85× on this virtualized host) |
| Measured-replan runbook | `bash tools/silicon/measure_replan.sh` | `rig-ready: NO` (PMU/RAPL/DVFS unavailable); degraded synthetic verdict, no measured win claimed |
| Worked-example pins | `python -m bcir.run vector_add --target x86_avx512 --theta cool` / `--budget thermal=700 --overlap` | score **7808** (vec16) / **9472** (vec8), as pinned |
| `llvm-training` validators | manifest, exercise manifests, autograder self-tests, dataset export, `verify-examples.sh`, opaque pointers, BCIR mapping | all exit 0 (157 examples, 42 exercises, 42 dataset records) |
| Whitespace | `git diff --check` | clean; tests left tracked files unchanged |

### 3.2 The MLIR rail on a coherent LLVM/MLIR 22 toolset

<!-- MLIR22-RESULTS -->

### 3.3 The same rail against LLVM/MLIR 23.1.0 (the actual latest release)

<!-- MLIR23-RESULTS -->

### 3.4 Newer interpreters and compilers

<!-- LATEST-TOOLCHAIN-RESULTS -->

### 3.5 What could not be produced here

No silicon certificate of any kind: the host has no PMU, RAPL or governor, so
`calibration.py` correctly refuses to freeze cost tables and the replan runbook reports
`rig-ready: NO`. This matches [`BCIR_TARGET_ACCESS.md`](../BCIR_TARGET_ACCESS.md) exactly
and is not a defect. Windows, native ARM and the hosted-model (PyTorch) jobs were left to
GitHub Actions, where the last 30 `main` runs are all green.

---

## 4. Architecture assessment by subsystem

### 4.1 The Python oracle (`bcir/`)
*Real and strong.* The whole correspondence chain runs end to end deterministically
(integer/Q8), with the two-truth quarantine machine-checked (`test_hot_cold.py`,
`import_graph.py`: 80 cold organs, none imported eagerly). The K_BCIR core (min-plus,
RCSP/Pareto, (max,+) overlap, soft-DP, bundle, e-graph, provenance manifests) is mirrored
bit-exactly on the MLIR rail and cross-checked by a *generated* differential rather than
pins. The oracle is also where most of the surface area lives: the ASN.1 portfolio (X.680–
X.697 plus complete X.692 ECN with R24/R25), the C-front compiler, telemetry, the ML
organs and the bounded model labs. The honest boundary is that the oracle is not the
product: `bcir.hosted.*` is quarantined, the LLVM AOT/JIT path accepts exactly one
elementwise claim, and the learned organs (MoE gate, bayescal, regret ledger) are kept off
legality by discipline plus R13, not by type.

### 4.2 The MLIR law rail (`mlir/`)
*Real, dual-rail, and one LLVM major behind its own policy.* `-bcir-verify` implements
R1–R25 with a negative fixture per law; the deterministic optimizer core is C++ and
reproduces the oracle's scores on the widened corpus and the six-target matrix; the x86
asm-edge lowerings are assemble-smoke-gated through `mlir-translate | llc`; IRDL gives a
portability projection on stock `mlir-opt`. `bcir-aot` is *partial* preparation and says
so. The one open rail-level defect is the provenance hash: `hash_target` omits the memory
hierarchy, and closing it needs an ODS attribute plus `hashTargetFromIR` in the same commit
(G0 deliberately routed around it with `ExecutionScopeV1`; see §9).

### 4.3 The C rail (`runtime/c/`, `runtime/cpp/`)
*The production-grade half.* Three enforced memory classes (freestanding heap-free, hosted
allocator-injected with fail-every-allocation tests, driver adapters with handles/offsets),
freestanding twins for StreamPack v1–v3, BCAB v1, BTLM v1, and every ASN.1 transfer syntax
(X.690, PER incl. the plan-driven decoder, OER, XER, bounded JER), each `-Werror` at
C11/C23, `-O0 == -O3`, and fuzzed under ASan/UBSan. The C-front twin is a driver-subset
C23 compiler with ~21 fuzzer-found miscompiles turned into permanent gates; `_Decimal*`
is blocked on a reference compiler. `runtime/cpp/` is honest: single-node dispatch is
real, dynamic/distributed orchestration throws `HandoffError`. The C rail is where BCIR
would ship; it has no resident driver yet.

### 4.4 The ASN.1 portfolio
*Complete on its documented subsets, uneven in its documentation.* The build-out thesis
(encoding rules are a realization choice, so K_BCIR can select a certified legal wire
format under a budget) has been executed end to end, including calibrated native tables
for two admitted ARM targets and a plan-driven PER decoder. The two remaining substantive
gaps are named by the repository itself: no real-protocol grammar corpus (asn1c's 165 /
asn1scc's 803 modules) and three X.680 front-end refusals (`COMPONENTS OF`, selection
types, `WITH SUCCESSORS/DESCENDANTS`), plus no schema-driven random value generator to
seed the triple-rail differential. The four shipped encoder bugs of #687–#689 were found
by adding constructs to the corpus, not by tests, which is the strongest argument for the
corpus work.

### 4.5 The ML substrate and model labs
*Bounded references, honestly labeled.* BCIRQ8 v1 with TinyLlama Python↔C parity, the
hosted train-to-C micro gate, offline staged training, payload-free placement, HAM
metadata planning, hardware-RL over six *simulated* episodes, and three architecture
labs all exist with deterministic gates, and each one refuses live promotion from its own
provenance. What does not exist: whole-decoder Q4, GPU execution, production serving, a
trained 32M model, real policy episodes, physical HAM adapters. WMR-3 (seeded samplers
over a shared Python/C RNG) is the single open slice of the core capstone.

### 4.6 Telemetry, drivers, kernel
*Codecs landed, live plane missing.* Signal registry (IDs 1–8 real, 9–15 honestly
unavailable), BTLM v1 frame codec, derived metrics, Prometheus/OTLP/Redfish-shaped
serialization, RuntimeChannel v1 loopback, EV1–EV3 events, DMA descriptors and the x86
long-mode entry/interrupt edges are code-backed. There is no UART sender, HTTP/OTLP
client, resident driver, UAPI, Linux module or native IPC. The driver roadmap's order is
right and is repeated consistently across five documents: telemetry identity → UART →
Linux adapter → virtio → UAPI v1 → IPC.

### 4.7 CI and tooling
Thirteen jobs in [`ci.yml`](../../.github/workflows/ci.yml) (two oracle shards with the
8,000-module differential and 4,000-iteration fuzz, C runtime + real-model gate, C
analysis with 500k-run libFuzzer, host portability on Ubuntu and Windows, hosted-model
train-to-C on both, the training corpus, native ARM oracle and C runtime, the MLIR rail
on LLVM 22, docs governance, a Python-floor job that refuses any interpreter but 3.11,
and the security-assurance rails) plus a weekly deep C sweep with Valgrind and cppcheck.
Actions are SHA-pinned, the token is read-only, superseded runs are cancelled. This is a
mature pipeline for a project of this age.

---

## 5. History and process review

**Shape of the record.** 1,759 commits on `main` between 2026-04-23 and 2026-09-03, 757 of
them merges of PRs #2–#750; only 16 first-parent commits bypassed a PR, all of them
documentation edits by the maintainer. Head-branch prefixes: `claude/*` 551 PRs, `codex/*`
157, `agent/*` 10, a handful of `security/`, `fix/`, `audit/`, `revert-*`. June 2026 alone
carried 1,216 commits and 551 merged PRs (the C-front arc; 106 commits on 2026-06-19),
July 252/86, August 209/55. The most-churned files are `docs/STATUS.md` (500 touches,
generated), the master roadmap (263), the C-front test file (240), `run_all.py` (198),
`check_runtime.sh` (194) and `bcir_cfront.c` (194): the compiler twin and its gates are
where the effort went.

**What the PR bodies show.** The verification-first convention is real: #744 (GEM+ G0) ends
with exact quick-tier counts, ruff and docs-gate outcomes, and records that its own harness
"first reported a 1.99 → 1.11 GAIN against a slice nobody had written"; #749 records
"3,343 passed, 0 failed", the secret-scan inventory, and that the local C fuzzer was
unavailable on a Clang 14 host rather than claiming it ran. The #749 campaign (42 review
rounds, 240 graded findings, 20 new laws) is the strongest single process artifact in the
repository, and its registry ([`laws.md`](../security/laws.md)) also records its own
tension honestly: the staleness rule fired at round 37 and five more rounds ran anyway.

**Where the process leaked.** The one substantive direct commit, `3decf69` (2026-08-11,
"Update BCIR_LANGREF.md"), appended a 3,190-line "comprehensive system report" to the
normative LangRef, pinned to a HEAD one commit older than itself, with its own §1–§33
numbering colliding with the normative §1–§19. CODEOWNERS routes everything to one
account, so there was no second reader. The mechanism that catches summary drift
(`check_claims.py`) holds only five claims today and could not see this.

**CI health.** 725 workflow runs on `main`; the last 30 are all successful. No open pull
requests, no open issues. The repository is effectively a single-maintainer project
driven by two agent fleets, which makes the written gates (not reviewers) the safety net.

---

## 6. Documentation audit: drift and inconsistencies found

Every item below was verified against the tree on 2026-09-03. Items are ordered by how
much they can mislead a reader or an agent.

| # | Finding | Where | Fix |
|---|---|---|---|
| D1 | A dated system report (3,190 lines, ~70% of the file) is appended to the **normative** LangRef with colliding section numbers, a stale HEAD pin (`997511de`), hard-coded counts, and an explicit "does not prescribe next work" disclaimer | [`BCIR_LANGREF.md`](../BCIR_LANGREF.md) from the second top-level heading onward | Move to `docs/research/` (or delete; this document supersedes its inventory), leave a one-line pointer |
| D2 | Stale or narrowed law ranges (`R1–R23`, `R1–R24`, `R14–R23`, `R1–R21`, `R1–R22`) in active docs while the rail is R1–R25 | README (`verify/ … R1–R23`), `AGENTS.md`, `BCIR_Repo_Structure.md`, `RELEASE_NOTES_0.3b.md`, LangRef §10 and `BCIR_ASN1_X690_ABI.md` §6 (`R14–R23`), LangRef §11 (`R1–R12`), `TELEMETRY_PIPELINE_RESEARCH.md`, `CPP_HANDOFF_BOUNDARY.md` (×3), `BCIR_ML_AI_INTEGRATION_ROADMAP.md` (×6), `ML_LANGUAGE_PLACEMENT_ANALYSIS.md` (×6), `BCIR_PYTHON_NATIVE_BOUNDARY_AUDIT.md` (×2), `BCIR_GAME_OPTIMIZATION_ROADMAP.md` (×3), `BCIR_TRITON_COMPARATIVE_ANALYSIS.md` (×2), `BCIR_NATIVE_BACKEND_FEASIBILITY.md`, the two security docs, the TMSAO report, and the agent skill alias table (`R-laws=R1–R21`). The C-front's *scoped* `R1–R18` statements are intentional and correct | One sweep, then a governance check that fails on `R1–R<N>` where N is not the generated last law unless the line carries a scope marker |
| D3 | [`PER_DECODER_HANDOFF.md`](../PER_DECODER_HANDOFF.md) is entirely stale: it says "No whole-value decode" and "one row", but `runtime/c/bcir_per_plan.{c,h}` exist, the `#per` gate passes, and the build-out roadmap's phase-H blockquote already reports three rows | `docs/PER_DECODER_HANDOFF.md` | Retire it (move its verification-discipline list into the systems-engineer skill, which already carries it) |
| D4 | Phase-G paragraphs contradict each other inside one section ("clause 21.7 is complete" vs "two of §21.7's eight determinations remain"; "second replacement group is built" vs "needs a second replacement group"); R25's rule count is given as 11, 22, 43 and "dozens" across four documents; headers still say "baseline through PR #670" | [`BCIR_ASN1_BUILDOUT_ROADMAP.md`](../BCIR_ASN1_BUILDOUT_ROADMAP.md) §1/§4-G/§9, [`BCIR_ASN1_JSON_ROADMAP.md`](../BCIR_ASN1_JSON_ROADMAP.md) §2, LangRef §10, [`BCIR_ASN1_COMPILER_COMPARISON.md`](../BCIR_ASN1_COMPILER_COMPARISON.md) (17 vs 19 transform classes) | Reconcile against `bcir/asn1/ecn_*.py`; derive the R25 rule count from the pass source the way `gen_status.py` derives the law range |
| D5 | Status block says "It claims no implementation. Nothing in this document is built" while §7 marks P0–P6 landed with gates met | [`BCIR_JSON_PROGRAM_REPRESENTATION.md`](../BCIR_JSON_PROGRAM_REPRESENTATION.md) | Rewrite the status block |
| D6 | The UART blueprint reserves **R24** for a future UART law (R24 is now ASN.1), pins `PATH=/usr/lib/llvm-18/bin` (rail is 22), cites abolished roadmap "Part IV/VI" numbering, defers the IRQ driver because "BCIR has no event-triggered phase model" although EV1–EV3 landed, and pins gates at "~3× headroom" in §0 but "~⅔ of measured" in §6 | [`BCIR_UART_DRIVER_BLUEPRINT.md`](../kernel/BCIR_UART_DRIVER_BLUEPRINT.md) | Fix before U0 starts; the implementer will otherwise inherit the collisions |
| D7 | Neither [`HARDWARE_VALIDATION.md`](../kernel/HARDWARE_VALIDATION.md) nor any kernel/ML document cites [`BCIR_TARGET_ACCESS.md`](../BCIR_TARGET_ACCESS.md), the probe-backed record of what is blocked and why | kernel/, machine-learning/ | Add the cross-reference where "hardware-gated" is claimed |
| D8 | Four unrelated slice-ID schemes share the letter G: GEM+ G0–G10, the game roadmap's G1–G11, the native-object gate's criteria G1–G4, and the historical vision-gap program G1–G8; WMR-5/6/7 duplicate the ML roadmap's §1.6–1.8 | research/, kernel/, machine-learning/ | Prefix the game slices (`GO1…`) and the gate criteria (`NG1…`); pick one home for the labs |
| D9 | `pyproject.toml` declares `requires-python = ">=3.11"` (enforced by CI) but ruff's `target-version = "py310"` | `pyproject.toml` | Set `py311` |
| D10 | Companion docs still carry "current through PR #645/#670/#719" banners and pre-v2 phase numbers ("Phase 8", "Part IX", "wave 15") | `DEVELOPMENT_HISTORY.md` header, `BCIR_STREAMPACK_ABI.md`, AMD roadmap, ML roadmap | Replace with a date and a git SHA, which do not rot the same way |

The security audits' own open items also remain open and are tracked below (§9, spine):
the two-rail provenance-hash closure (B7), R11 tracking only the maximum generation, the
integer-sanitizer class invisible to CI, and two unreproduced findings from
[`BCIR_SECURITY_AUDIT_2026-08-12.md`](BCIR_SECURITY_AUDIT_2026-08-12.md) §5 (CSE merging
effectful claims; a decoupled GGG tail overlapping barriered work) that should be either
reproduced or formally retired.

---

## 7. Code-quality observations

- **No TODO/FIXME/XXX markers in production code.** Unsupported paths raise
  `NotImplementedError` (31 sites in `bcir/`) or return typed refusals; this is consistent
  with the "fail honestly" invariant and makes the boundary greppable.
- **ruff drift.** `ruff 0.15.8` reports 22 findings against the tree; 15 are `B023`
  (closure over a loop variable) and every one was inspected: all are immediately invoked
  inside the same iteration (`autodiff_program.py:133–136`, `schedule_artifact.py:208`,
  `fuzz_cfront.py:685/882`, `test_train_graph.py:324`), so they are false positives, not
  bugs. Six are auto-fixable (`F541`, `UP012`, `UP034`, `E713`). ruff runs only in
  pre-commit, not CI, and its version is unpinned, so the "calibrated green" promise in
  `pyproject.toml` no longer holds on a current ruff. Pin it or run it in CI.
- **Hotspots.** The five largest files (§2) are also among the most-churned; the ECN rail
  in particular (~6.7k lines across two modules with a `SYNTAX_VERSION` counter at 13) has
  no natural seam yet. A split by X.692 part (syntax / user-defined / built-in) would
  match the standard and the documentation.
- **Test registry discipline works.** `run_all.py` registers 314 modules, and
  `_REPO_ONLY_MODULES` is now derived per test rather than per module; the packaged wheel
  is exercised on the floor interpreter. The `test_the_gate_and_the_harness_link_the_same_sources`
  pattern (guarding the check_runtime/native_bench source lists) is worth generalizing to
  every gate/harness pair that lists sources twice.
- **Deliberate unsigned wraps** in the C twins are commented; the three `-fsanitize=integer`
  findings cleared in the second August audit still need their suppressions landed so CI
  can run that sanitizer class at all.

---

## 8. Risks, ranked

1. **Intelligence ahead of substrate.** The learned organs, HAM planner, hardware-RL and
   the ASN.1 selection certificates all wait on measured telemetry from a device BCIR
   does not yet drive. Every optimality claim is TMSAO-4. Mitigation is sequencing, not
   more modeling: telemetry v0 → UART → virtio before any further learned-organ breadth.
2. **Documentation as a second, unverified rail.** With ~24k lines of prose and five
   machine-checked claims, summaries rot faster than they are reread (the repository's
   own `check_claims.py` docstring says exactly this). §6 lists the current debt.
3. **Single-reader governance.** One CODEOWNER, direct commits to the normative document,
   and no open issues means the gates are the reviewers. The #749 campaign shows the
   review loop is "structurally blind to its own assertions"; keep the fail-closed
   independent-review rail and consider branch protection that requires a PR for
   `docs/BCIR_LANGREF.md` and the ABI documents.
4. **Toolchain drift on the MLIR rail.** LLVM 23 is released; the rail is pinned to 22 and
   the Ubuntu apt source for either is unreachable from some hosts (see §10). A stale
   pin is fine as a decision; it is a risk as an accident.
5. **Advisory blind spot.** The dependency audit asserts the inventory (good) but never
   consults an advisory database anywhere, so a vulnerable `setuptools`, `torch`,
   `safetensors` or `numpy` pin would be found only by a human (the setuptools 83 floor was).
6. **Hardware access.** J6/J7, D2/D3, B1 two-target evidence, HMF-D1+, calibration
   freezes and every silicon certificate are blocked on a bare-metal x86-64 and aarch64
   box meeting [`BCIR_TARGET_ACCESS.md`](../BCIR_TARGET_ACCESS.md) §4. This is the one
   risk that cannot be engineered away in this repository.

---

## 9. Recommended next development steps

Ordered by leverage per unit of risk, respecting the repository's own dependency order and
its PR-sized-slice method. "Ready" means no hardware and no other slice is required.

### 9.0 Hygiene wave (one or two PRs, ready)
1. Documentation currency: everything in §6 (D1–D10). Add a `law-range` predicate to
   `tools/docs/` that reads the last law from the same source `gen_status.py` uses and
   fails any active doc whose `R1–R<N>` differs unless the line carries a scope marker
   (`<!-- law-scope: cfront -->`), then register it in the docs-governance job.
2. Pin ruff (or run `ruff check` in CI on the calibrated version) and apply the six
   auto-fixes; set `target-version = "py311"`.
3. Add an advisory scan to `audit_dependencies.py` under the `security-assurance` job,
   keeping the inventory assertion first and treating a missing scanner as FAIL there
   (L1/L2: a skip in a required job is where a shipping defect hides).
4. Refresh the agent digest and skill alias table (`R-laws=R1–R25`).

### 9.1 The certificate spine (ready; the optimizer's correctness debt)
5. **GEM+ G1**: one canonical schedule artifact, gated on `pricing.eft.divergence`
   1.9922 → 1.0 (`tools/perf/gemplus_baseline.py --compare`). Prerequisite for G2, G4, G5.
6. **G3**: digest computed once, with the cache-invalidation half as a Class-B defence.
7. **B7, two-rail**: widen `hash_target` with a `DenseI64ArrayAttr` for memory tiers and
   claim-order fields, plus `hashTargetFromIR`/`hashModuleFromIR`, in **one commit** with
   the differential regression; CI's `mlir-rail-validate` job builds the rail, so the
   host limitation that deferred it in August no longer applies.
8. **R11 per-resource generation vectors**: an append-only StreamPack v4 record, its C
   twin, and the ASN.1 projection, landed together with `BCIR_STREAMPACK_ABI.md`.
9. **G4**: bounded exact solvers plus the ten-bound lower-bound stack, producing the first
   TMSAO-2 certificate and retiring the five bound-less baseline rows.
10. **G9 remainder** (alias scopes, TBAA, `volatile` facts to LLVM) and **G7** (repair the
    native measurement rig so the bare-metal refusal is mechanical).

### 9.2 The driver chain (ready up to the hardware line; unblocks 0.3b)
11. **Telemetry version-zero triple**: generated fixed-width C signal table with ID-range
    policy; source/session/generation/clock-aware driver envelope (a *new* versioned
    frame, not a reinterpretation of BTLM v1); live SPSC ring with head/tail,
    acquire/release publication, per-slot sequence, declared overwrite/backpressure, loss
    counters; restart/stale-generation/wrap/saturation/peer-death tests; Python↔C parity.
12. **UART U0** (registry, `RegMapContract`, generated header), **U1** (five protocol laws
    on both rails with negatives and a vacuousness sweep; assign real law numbers first,
    since the blueprint's reserved R24 is taken), **U2** (polled driver + spec-exact
    `sim16550` + a linked binary built only from emitted TUs). U2 is the first *running*
    BCIR-compiled driver and the evidence 0.3b is waiting for.
13. **D3 Linux-hosted adapter and virtio-console/virtio-blk** follow, but D3 needs kernel
    headers and module loading, which no currently available host provides.

### 9.3 ASN.1 corpus and front end (ready)
14. Vendor asn1c's and asn1scc's grammar corpora as front-end conformance input; then
    close the three X.680 refusals; then a schema-driven `random_value(kind, seed)` to
    seed the triple-rail differential and the fuzz harnesses; then the security-regression
    corpus and the extension-additions gap in `encode_plan` (the last blocker for a
    plan-driven PER emitter).

### 9.4 ML substrate (ready, but explicitly behind the driver chain)
15. **WMR-3** seeded samplers over a replayable shared Python/C RNG; the whole-decoder Q4
    *contract* (wire layout, activation/outlier policy, drift/NLL gates) before any Q4
    kernel; keep every learned organ off the hot path.

### 9.5 Blocked on hardware access (do not simulate)
16. J6 hardware counters and calibration freezes, D2/D3 device binding, B1 two-target
    schedule evidence, HMF-D1–D5, AMD Phase 2+, the native-object gate (G1/G2 unmet),
    and any silicon TMSAO certificate. One x86-64 and one aarch64 bare-metal box would
    unblock all of them at once; nothing else on this list is waiting on them.

### 9.6 The LLVM 23 decision
17. Either bump the `mlir-rail-validate` matrix to `["23"]` (and the skill/digest text)
    once §3.3's residue is fixed, or change the CI comment so the pin is a stated
    decision rather than a claim of tracking the latest release. Running the matrix on
    both majors for one release cycle is cheap and is how #243–#246 moved to 22.

---

## 10. Reproducing the toolchains on a network-restricted host

This session's egress policy allowed `archive.ubuntu.com`, `security.ubuntu.com`, PyPI and
`conda.anaconda.org`, and denied `apt.llvm.org`, `github.com`, `python.org`, Launchpad PPAs
and `api.anaconda.org`. The coherent toolsets were obtained without any of the denied hosts:

- **micromamba** from conda-forge's own package (`micromamba-2.9.0-0.tar.bz2`, found by
  decoding `repodata.json.zst` with the `zstandard` wheel from PyPI).
- **LLVM/MLIR 22.1.8 and 23.1.0**: `micromamba create -p /opt/llvmNN -c conda-forge
  mlir=NN.x llvmdev=NN.x clangdev=NN.x clang=NN.x clangxx=NN.x llvm-tools=NN.x lld=NN.x
  compiler-rt=NN.x` (about 3.4 GiB each). The rail scripts resolve tools from
  `/usr/lib/llvm-*/bin`, so a symlink `/usr/lib/llvm-NN → /opt/llvmNN` plus
  `MLIR_DIR`/`LLVM_DIR`/`MLIR_INCLUDE`/`CMAKE_PREFIX_PATH` makes `build_mlir.sh`,
  `tblgen_check.sh` and `check_corpus.sh` work unchanged. `FileCheck` is not in the
  conda `llvm-tools` package; Ubuntu's `llvm-18-tools` supplies it and the scripts accept
  any major for FileCheck.
- **Python 3.14.7 and 3.15.0rc2**: `micromamba create -p /opt/py314 -c conda-forge
  python=3.14`, and `-c conda-forge/label/python_rc -c conda-forge python=3.15.0rc2`.
- **compiler-rt for the system clang 18** (`libclang-rt-18-dev`) from the Ubuntu archive
  turns the honest "clang ASan skipped" line into a real clang ASan/UBSan/libFuzzer run.
- Ubuntu's noble-updates also carry native `clang-20`/`libmlir-20-dev`/`mlir-20-tools`;
  they were not used because the rail requires one coherent major (22).

---

## Appendix A. Reading map used for this report

Normative and state: [`BCIR_LANGREF.md`](../BCIR_LANGREF.md), [`STATUS.md`](../STATUS.md),
[`REPO_CURRENT_STATE_AUDIT.md`](../REPO_CURRENT_STATE_AUDIT.md),
[`BCIR_MASTER_ROADMAP.md`](../BCIR_MASTER_ROADMAP.md), [`PARITY.md`](../PARITY.md),
[`RELEASE_NOTES_0.3b.md`](../RELEASE_NOTES_0.3b.md),
[`DEVELOPMENT_HISTORY.md`](../DEVELOPMENT_HISTORY.md),
[`VISION_ALIGNMENT_AUDIT.md`](../VISION_ALIGNMENT_AUDIT.md),
[`PERFORMANCE_AUDIT.md`](../PERFORMANCE_AUDIT.md),
[`BCIR_TARGET_ACCESS.md`](../BCIR_TARGET_ACCESS.md),
[`BCIR_NATIVE_OBJECT_GATE.md`](../BCIR_NATIVE_OBJECT_GATE.md).
Companions: the ASN.1 build-out, JSON and compiler-comparison documents; the
driver/kernel, UART, AMD, HAM, BCAB, StreamPack, telemetry, hardware-validation, channel,
signal-registry and SYCL documents under `kernel/`; the C-front, C++ boundary and C memory
documents under `languages/`; the ML roadmap, native-boundary audit, whole-model reference,
language-placement analysis, OpenAI research and third-party-model register under
`machine-learning/`; and every research note, including
[`BCIR_GEMPLUS_ROADMAP.md`](BCIR_GEMPLUS_ROADMAP.md), the TMSAO report and proposal, the
triage, game and Triton studies, the three security audits and the threat model, and
[`laws.md`](../security/laws.md) with its harvest CSV.
