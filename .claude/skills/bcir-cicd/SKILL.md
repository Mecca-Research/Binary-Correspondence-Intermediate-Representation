---
name: bcir-cicd
description: >-
  BCIR-compliant CI/CD protocols and procedures for the
  Binary-Correspondence-Intermediate-Representation repository. Load this skill
  whenever work touches `.github/workflows/`, adds or edits a CI gate or
  checker/scanner/audit tool (secret scan, dependency audit, fuzz/decoder
  campaign, differential, review contract), debugs a red check, responds to a
  review bot (Codex, Copilot, CodeQL), prepares a PR for publication, or designs
  validation for a new rail — even when the request just says "CI is failing",
  "fix the pipeline", "add a check", or "address the review". Pairs with
  bcir-systems-engineer (the general engineering method); this skill owns the
  gate-authoring and pipeline rules specifically.
---

# BCIR CI/CD protocols

How to author gates, run the pipeline, and survive review loops in this
repository without shipping fail-open checks or chasing a bot forever. Distilled
from `.github/workflows/ci.yml`, AGENTS.md/CONTRIBUTING.md, the 2026-08-12
security audits, and the PR #749 assurance-rails arc — 42 adversarial review
rounds, 240 graded findings, a scope rollback and the regressions it
reintroduced, now merged.

## 1. The pipeline as it exists

Three parallel workflows run on every push/PR; the main CI's jobs and what each
*owns* (read `ci.yml` before trusting this list — it is the authority):

| Job | Owns |
|---|---|
| BCIR oracle 1/2 + 2/2 | thorough tier, sharded by discovery-order stride (`BCIR_THOROUGH=1 run_all -j0 --shard N/2`); whole-suite campaigns run once on shard 1 |
| BCIR C runtime / C analysis | StreamPack byte-identity, decoder fuzz, sanitizers, static analysis |
| BCIR MLIR rail (LLVM 22 / LLVM 23) | tblgen, `bcir-opt` build, ODS corpus, pass tests — the only place compiled-law work is verified |
| LLVM training corpus | the separate curriculum's validators |
| Docs governance | STATUS drift, links, retired paths, claim markers |
| Host portability (ubuntu/windows, 3.12) + hosted model gates | cross-host oracle + train-to-C |
| aarch64 oracle + C runtime | native ARM evidence (never emulate locally) |
| Security assurance | secret scan, inventory audit, tool-boundary policy, decoder campaign, malformed differential, review contract |
| CodeQL (3 languages) | GitHub-native static analysis |

Local hosts run the **bounded** equivalents: quick/thorough `-j 2`, heavy gates
serialized, never concurrent Python/C/model/MLIR gates. CI-equivalence before
publishing: map every affected job from `ci.yml`, run what the host supports,
delegate the rest, record exact outputs in the PR body, regenerate `STATUS.md`
**last**, confirm tests leave tracked files unchanged, `git diff --check`.

## 2. Gate-authoring laws

> **`docs/security/laws.md` is the authority.** It holds the 21 registered laws
> (L1–L21), each with its witness tests and a C/C++ port note, derived from all
> 240 findings. The eleven rules below are the operational subset — the ones you
> reach for while writing a gate — and each names its registered law so the two
> cannot drift. When they disagree, the registry wins; when you find something
> the registry lacks, add it there, not here.

Apply them to any new checker, scanner, campaign, or CI step.

1. **Fail closed, everywhere, structurally.** *(L1)* Every path out of a checker is one
   of PASS / FAIL / INVALID-VACUOUS / UNAVAILABLE-SKIPPED — including the error
   paths. A missing tool, timeout, undecodable output, unreadable input, or
   malformed configuration returns the structured failure report; a traceback in
   place of a report is itself a defect (it skips the `--json-out` artifact and
   the exit-code contract). Concretely: wrap process launches for `OSError` and
   `TimeoutExpired`; capture bytes and decode explicitly; catch parse errors of
   your own configuration (`shlex.split` raises too).
2. **UNAVAILABLE needs an owner.** *(L2)* A skip is honest only where the absence is
   expected. In the CI job that *installed the tool for this gate*, absence is a
   failure — give the CLI a `--require-<rail>` flag and pass it from that job.
   Locally, the same absence stays a documented skip. Never let one silent skip
   remove the coverage a job exists to provide (the #749 C-campaign hole).
3. **Anti-vacuity is a state, not a comment.** *(L2)* A gate that can pass without
   examining anything is broken even while green: refuse empty corpora and
   zero/negative iteration counts, require a minimum number of *executed*
   negative cases, treat a rejected canonical seed as a finding, and ask of
   every checker: *what input makes every loop in it iterate zero times?* Then
   feed it that input in a test.
4. **Prove the gate can fail.** *(L2, L11)* For each new gate, inject the fault it guards
   and watch it fire (RED) before landing the fix/gate pair (GREEN). A test
   asserting only the honest path passes against every defect. When
   parallelizing or speeding up CI, re-inject faults to prove the sped-up gates
   still fire — a `grep -q` under `pipefail` once reported a *found* pattern as
   a failure here.
5. **Bounds are enforced where the data enters.** *(L3)* A declared cap that is checked
   after materialization is not a cap. Enumerate incrementally, bound reads
   before believing declared sizes, and treat "cannot inspect" (encrypted,
   over-cap, unknown format) as a failing finding — never as clean.
6. **Suppressions match the value, not the line.** *(L6)* Placeholder/fixture
   allowlists inspect exactly the matched credential or an exact-value list.
   Line-level or substring suppression lets adjacent innocuous text hide a real
   finding.
7. **Only claim formats you can open.** *(L4, L5)* If a classifier routes `.7z` to a tar
   parser, every legitimate `.7z` fails the job. A format with no inspection
   path follows the record-don't-parse binary policy; the archive list contains
   exactly what the inspector can read.
8. **Timeouts scale to the whole campaign** *(L8)* (targets × per-target bound ÷
   workers, plus startup), and expiry is a structured FAIL that prints the
   captured diagnostic tails — the temp dir is gone afterwards, so the report is
   the only evidence.
9. **Heavy work never rides the parallel Python pool.** *(L19)* Unit tests of a gate
   mock the expensive rail (C fuzzer, compiled `bcir-opt`, `pip-audit`,
   gitleaks) so the quick tier stays host-independent and bounded; the real
   invocation lives in the serialized job that owns it. An optional engine the
   host happens to have must not change a unit test's verdict.
10. **Opted-in engines gate.** *(L9)* If a run requests an external engine
    (`--allow-gitleaks`), its non-zero result changes the verdict — an engine
    whose failure is metadata is theater.
11. **Workflow hygiene:** *(L14)* SHA-pin every action, keep tokens read-only (both are
    already test-enforced), cap workers at the job's documented level, harden
    apt with retries/timeouts, and change a gate's source list and its harness
    twin together (the #719 wiring trap — add a test that reads both).

Six more the loop bought after these eleven were written, each the registry's
law stated for gate authors:

12. **One predicate per repeated defect, and make it total.** *(L14)* The
    dominant defect shape across 42 rounds was never a novel bug — it was a
    mechanism landed on two rails out of three. And a shared predicate carrying
    an unstated precondition is a defect held in reserve for its second caller.
13. **A gate must not disagree with itself across two paths.** *(L12)* The
    index and the worktree, POSIX and Windows, staged and checked-out: same
    condition, same answer, or a declared boundary in the tool's own docstring.
14. **What a gate scans is reconciled against what the repo tracks.** *(L15)*
    A tracked file the walk never yielded is a finding; skips are path prefixes
    with named roots, never bare component names.
15. **A report is an egress surface.** *(L7)* Findings carry fingerprints, never
    values. A wrapped tool's stdout is a report field; a reviewer *quotes the
    code it reviews*. Redact by position, using the same predicate that decides
    what to report — a report must not remove less than the scan would report.
16. **The instrument must be unswallowable by its subject.** *(L9)* A watchdog
    the bounded code can catch is not a bound: derive it from `BaseException`,
    or in C run it as a separate process.
17. **A package must contain what it registers, and a skip is where a shipping
    defect hides.** *(L21)* An exclusion converts "the artifact is broken" into
    "this does not run here", and in a green run those read identically. Scope
    an exclusion to the dependent test, never the module, and ask first whether
    the asset should simply ship.

## 3. Heuristic checkers: declare scope or drown

A static detector (secret scan, subprocess-boundary audit) can always be beaten
by one more language feature, and a review bot will find them one per round —
each finding individually valid, the sum an interpreter nobody asked for. PR
#749 lived this: alias tracking → assignment tracking → reassignment
invalidation → per-scope constants → control-flow merge → nested closures →
chained targets… then a rollback.

The stable answer, now written into `audit_tool_boundaries.py`'s docstring:

- **Declare the scope in the tool itself** ("this flags literal `shell=True`
  and string commands; aliases, control flow, `**kwargs`, and nested scopes are
  out of scope — those belong to a dedicated linter, not a BCIR rail").
- **Inside the declared scope, be exact**; outside it, refuse to grow. Answer
  the next soundness finding by pointing at the declaration, not by adding an
  interpreter state.
- **Scope the *tree* honestly too**: a "developer-tool policy" that skips
  tracked developer trees (`.claude/`) audits less than it claims — widen the
  tree, and fix the offending script, before narrowing the claim.
- Contract checks on *structured* input (JSON verdicts, wire formats) are
  different: there, completeness is achievable — reject duplicate keys, require
  nonempty required values, enforce the grammar before conversion (never a bare
  host-language parser on wire text).

## 4. Review-bot protocol (Codex, Copilot, and successors)

1. **Verify every finding against the tree before fixing it** — reproduce or
   refute; nothing is inherited on trust. Classify: valid / already-moot
   (anchored to deleted code) / out-of-declared-scope.
2. **Fix a repeated defect with one shared predicate**, not N local repairs —
   five local repairs is how there came to be five defects.
3. **Reply with the fixing commit SHA and resolve the thread**; an
   out-of-scope finding gets a reply citing the declared scope; a moot one gets
   its disposition. Unresolved threads are work, not noise.
4. **Detect non-convergence**: when each fix draws a new or reshaped finding of
   the same family, stop pushing, declare the scope boundary (§3), and record
   it once.
5. **Grade every finding as you fix it, and watch the grade — not the count.**
   #749 ran to round 42 because the findings stayed real; what told us when to
   stop was a second axis recorded per finding: **NEW-LAW** (originates a
   registry entry), **INSTANCE** (a new entry point to a registered law), or
   **LOCAL** (no transfer value beyond the code touched). Over 240 findings the
   split was 20 / 181 / 39, and the rolling NEW-LAW rate is the loop's vital
   sign: a loop still finding *laws* is discovering, a loop finding only
   *instances* is maintaining one implementation. Declare the stop condition
   **before** you need it — #749 declared "three consecutive zero-NEW-LAW
   rounds" in round 31 precisely so the call could not be made in the moment —
   and record the grades where they survive the loop (`docs/security/laws.md`,
   `docs/security/pr749-harvest.csv`).
6. **Two blind spots the loop cannot see itself.** A review pair never audits
   its own *assertions* — a witness checking a substring of what should have
   been removed passes on violating output, and it took a third reviewer
   (CodeQL) to find it. And a defect in the *shipped artifact* sits outside
   every gate: the two findings that reached furthest in #749 were a wheel
   missing library data and a resource opened through a working-directory path,
   both hiding inside a test-runner exclusion. When a round produces only
   instances, look at the witnesses and the package, not harder at the gates.
7. **Roll back scope with an audit, not a revert.** A scope-stripping commit
   reverts *every* fix layered on the stripped code — including accepted,
   valid, non-scope fixes. Before landing it, walk the accepted findings and
   re-apply each one that is about the *contract* rather than the scope. The
   #749 rollback silently reintroduced six closed holes (tar link targets,
   format misclassification, reviewer timeout/startup, env quoting, gitleaks
   propagation, C-require mode); the bot found them all again within hours.
8. Bots reviewing *this* repo read AGENTS.md — so do you, before arguing with
   one.

## 5. Red-check triage

- **Identify the failing test/step exactly** (job logs, not the summary) and
  reproduce with the job's real command — CI runs the *thorough* tier sharded
  (`BCIR_THOROUGH=1 … --shard N/2`), not quick; a green local quick tier proves
  nothing about a thorough-shard failure.
- **Not this PR's failure** only when: the error names a service the diff
  cannot touch, or the same check is red on the base/other PRs (e.g. the
  org-wide `github-advanced-security` Copilot-licensing failure). Say so once;
  never push changes for it.
- **Timing gates on shared runners flake by construction** — classify metrics
  exact/ratio/wall; gate on `exact` rows; a `ratio` row missing its band by a
  hair on a shared runner earns exactly one confirming re-run, and a second
  failure is real.
- Never skip/disable/quarantine a test for green; never empty-commit or
  close/reopen to kick CI.

## 6. Publication

All required checks green — pending is not complete. PR body ends with a
Verification section quoting exact gate outputs and naming every honest skip
with its reason. One PR per gateable slice; target `main` directly (stacked PRs
on kept bases mis-merge — recovery PRs #710/#716). After merge conflicts or
base-branch recovery, merge the base in (never rebase someone else's branch)
and re-run the affected gates before pushing.
