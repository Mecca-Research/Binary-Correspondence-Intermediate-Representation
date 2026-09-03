# Dependency audit — 2026-09-03

> Dated and non-normative. What the repository depends on, what an advisory database says
> about it on this date, how current each pin is, and what the advisory rail enforces from
> this slice on — so that the next audit is a CI run, not a document. The generated
> [`STATUS.md`](../STATUS.md) owns live counts; the gate-authoring laws cited here (L1–L21)
> are [`laws.md`](laws.md). Companion to the 2026-09-03 whole-system analysis
> ([`BCIR_SYSTEM_ANALYSIS_2026-09-03.md`](../research/BCIR_SYSTEM_ANALYSIS_2026-09-03.md)
> §8 risk 5 and §9.0 item 3), which found the advisory half of the dependency audit
> `UNAVAILABLE/SKIPPED` everywhere.

## 1. Verdict

- **No known vulnerability in the declared install set.** pip-audit 2.10.1, against PyPI's
  advisory database, over the six declared requirements: the resolved run (pip's resolver,
  transitive closure included) audited 40 packages, the floor run (each declaration at its
  lowest admitted version) audited 6; 0 advisories, 0 skips, every declared name covered.
- **The advisory rail had never run.** `tools/security/audit_dependencies.py` handed
  pip-audit `--requirement -`; the engine refuses that argument (`invalid requirements
  input: -`), so the first host with pip-audit on PATH would have failed the audit on its own
  argument — and no CI job installed one, so every run to date reported
  `advisory=UNAVAILABLE/SKIPPED` and passed. That is the L2 shape (green over zero
  executions) in a required job. Fixed in this slice (§4).
- **The one security-motivated floor was invisible to the engine.** pip-audit's resolver
  mode upgrades `pip`, `setuptools` and `wheel` inside its scratch virtualenv before it
  resolves, and pip's install report then omits them: a requirements file holding only
  `setuptools>=83.0.0` (the floor raised for GHSA-h35f-9h28-mq5c) audits **nothing** and
  exits 0. The floor run added in this slice audits that floor by name, and the gate now
  reconciles what the engine audited against what was declared (§4).
- **Currency.** Two of three model-lab pins are behind the latest release (torch 2.13.0 →
  2.14.0, numpy 2.2.6 → 2.4.6); the three floors (setuptools, ruff, pre-commit) are
  satisfied by current releases; the CI-only `build==1.2.2.post1` is four minor releases
  behind. None of these is a security finding today; §6 says which are worth moving and
  why they are not moved here.

## 2. Inventory and currency

The declared inventory is `tools/security/expected_inventory.json`, asserted against
`pyproject.toml` on every CI run (mismatch = FAIL). Resolved and latest versions are as
observed on 2026-09-03 from a Linux x86-64 host through PyPI.

| Surface | Declared | Resolves to today | Latest on PyPI | Note |
|---|---|---|---|---|
| build-system | `setuptools>=83.0.0` | 84.0.0 | 84.0.0 | floor set for GHSA-h35f-9h28-mq5c (also GHSA-5rjg-fvgr-3xxf, GHSA-cx63-2mw6-8hw5); satisfied |
| runtime | *(none)* | — | — | asserted empty: the oracle is dependency-free by design |
| optional `dev` | `ruff>=0.6` | 0.16.6 | 0.16.6 | the calibrated lint config pins nothing; CI runs the resolved version |
| optional `dev` | `pre-commit>=3.5` | 4.6.2 | 4.6.2 | pulls cfgv, identify, nodeenv, pyyaml, virtualenv (distlib, filelock, platformdirs, python-discovery) |
| optional `model-lab` | `torch==2.13.0` | 2.13.0 | 2.14.0 | pulls the CUDA 13 runtime wheels (cuda-toolkit 13.0.3.0, nvidia-cudnn-cu13 9.20.0.48, nvidia-nccl-cu13 2.29.7, nvidia-cublas 13.1.1.3, …), triton 3.7.1, sympy 1.14.0, networkx 3.6.1, fsspec 2026.7.0, jinja2 3.1.6, filelock, typing-extensions 4.16.0 |
| optional `model-lab` | `safetensors[numpy]==0.8.0` | 0.8.0 | 0.8.0 | current |
| optional `model-lab` | `numpy==2.2.6` | 2.2.6 | 2.4.6 | last 2.2.x; torch 2.13.0 declares no numpy pin |

CI-only Python tools, pinned in `.github/workflows/ci.yml` and outside the inventory:

| Tool | Pinned | Latest | Where |
|---|---|---|---|
| `build` | 1.2.2.post1 | 1.6.0 | `python-floor` job (the wheel-builds-on-the-floor claim) |
| `pip-audit` | 2.10.1 | 2.10.1 | `security-assurance` job (this slice) |

Native and workflow dependencies:

- **GitHub Actions** (all SHA-pinned; pinning is test-enforced): `actions/checkout`
  (13 uses, annotated `# v4`), `actions/setup-python` (4 uses, `# v5`), `actions/cache`
  (2), `actions/upload-artifact` (3), `awalsh128/cache-apt-pkgs-action` (8). The SHAs'
  currency against upstream releases was **not verified** from this host (`api.github.com` is
  not reachable through its proxy). The runner answered part of the question on this PR's
  first CI run: `actions/checkout` and `actions/setup-python` at the pinned SHAs target
  Node.js 20, which GitHub deprecated on the runners in 2025-09 and now forces onto Node 24
  with a warning in every job log; both actions have Node-24 releases upstream, so those two
  pins are at least one major behind. Three of the five carry no version comment, so a
  reader cannot tell a pin's age without resolving it.
- **apt** in `security-assurance`: `clang lld llvm libclang-rt-18-dev` — the runner's system
  clang is 18 on Ubuntu 24.04, and a sanitizer runtime must match the compiler that emits
  the instrumentation, so the 18 pin is coherent there and independent of the MLIR rail's
  LLVM 23 (`llvm-toolchain-noble-23`, with 22 in the matrix for one cycle).
- **pre-commit hooks**: `pre-commit/pre-commit-hooks` v5.0.0, `mirrors-clang-format`
  v18.1.8 (formats `runtime/c` and `mlir/`; a style, not a compiler, so no coupling to the
  rail's major).

## 3. The advisory scan — method and result

Engine: pip-audit 2.10.1 (its own dependency closure installed from PyPI by version pin),
vulnerability service `pypi` (the default; the `osv` alternative was unreachable from this
host and is not needed), CPython 3.11, 2026-09-03.

**Resolved run** — the declaration as written, resolved by pip into a scratch virtualenv
(`pip install --dry-run --report`) and audited with `--strict` (a dependency the engine
cannot collect fails the run rather than becoming a skip):

```
pip-audit --requirement <declaration> --format json --strict --desc off --progress-spinner off
40 dependencies audited, 0 vulnerabilities, 0 skipped, exit 0, 11 s
```

torch 2.13.0 · safetensors 0.8.0 · numpy 2.2.6 · cuda-toolkit 13.0.3.0 · nvidia-cudnn-cu13
9.20.0.48 · nvidia-cusparselt-cu13 0.8.1 · nvidia-nccl-cu13 2.29.7 · nvidia-nvshmem-cu13
3.4.5 · triton 3.7.1 · cuda-bindings 13.3.1 · nvidia-cublas 13.1.1.3 · nvidia-cuda-cupti
13.0.85 · nvidia-cuda-nvrtc 13.0.88 · nvidia-cuda-runtime 13.0.96 · nvidia-cufft 12.0.0.61 ·
nvidia-cufile 1.15.1.6 · nvidia-curand 10.4.0.35 · nvidia-cusolver 12.0.4.66 ·
nvidia-cusparse 12.6.3.3 · nvidia-nvjitlink 13.3.33 · nvidia-nvtx 13.0.85 · ruff 0.16.6 ·
pre-commit 4.6.2 · cfgv 3.5.0 · cuda-pathfinder 1.8.1 · fsspec 2026.7.0 · identify 2.6.19 ·
networkx 3.6.1 · nodeenv 1.10.0 · pyyaml 6.0.3 · sympy 1.14.0 · mpmath 1.3.0 ·
typing-extensions 4.16.0 · virtualenv 21.7.8 · distlib 0.4.3 · filelock 3.32.5 ·
platformdirs 4.11.7 · python-discovery 1.6.0 · jinja2 3.1.6 · markupsafe 3.0.3.

`setuptools` is absent from that list by construction (§1, third bullet).

The same rail on the runner (`security-assurance`, ubuntu-latest, CPython 3.12.14, the first CI
run with the engine installed) reached the identical verdict in 12 s:

```
audit_dependencies: PASS asserted=True expected=6 mismatches=0 advisory=PASS audited=46 covered=6/6
```

**Floor run** — every declaration pinned at the lowest version it admits (`>=X` becomes
`==X`; `==X` stays), no resolver, no scratch environment, only the advisory database:

```
pip-audit --requirement <floor pins> --no-deps --disable-pip --format json --strict --desc off --progress-spinner off
setuptools 83.0.0 · ruff 0.6 · pre-commit 3.5 · torch 2.13.0 · safetensors 0.8.0 · numpy 2.2.6
6 dependencies audited, 0 vulnerabilities, exit 0, 1 s
```

A declaration is a promise over a *set* of versions. The resolved run audits the set's
practical maximum (what `pip install` picks today, closure included); the floor run audits
its minimum, the version a stale cache or an old lock still satisfies the declaration with,
and the version most likely to be vulnerable. A floor that fails is fixed by raising the
floor. Together the runs cover every declared name (`covered=6/6`); the closure between
minimum and maximum is not enumerated, which is the honest limit of an unlocked
declaration (§5).

## 4. What the rail enforces from this slice on

`tools/security/audit_dependencies.py`, after the inventory assertion and only after it
(the order is unchanged: an unasserted inventory never reaches the engine):

1. **Ownership (L10).** The `security-assurance` job installs `pip-audit==2.10.1` and runs
   the audit with `--require-advisory`: there, no engine is a FAIL that names what was
   missing. The `python-floor` job keeps running the audit without the flag, asserting the
   inventory and recording the engine's absence as `UNAVAILABLE/SKIPPED` — a skip is
   honest only where absence is expected. `test_ci_owns_the_advisory_rail` pins that
   exactly one job installs and requires the engine.
2. **The engine gets files, not stdin (L2, L7).** Each run's requirement set is written to a
   private temporary directory (0700) that is removed on every exit path, including the
   engine never launching; a declaration can carry a credential in URL userinfo.
3. **Two runs, reconciled (L15).** Resolved and floor runs as in §3; the union of audited
   canonical names is compared with the declared names, and a declared name neither run
   reported is a FAIL that says which. Zero audited over a non-empty declaration is
   `INVALID/VACUOUS`.
4. **The floor grammar is declared, exactly (L4, L18).** `name[extras]==version` and
   `name[extras]>=version`, nothing else; a URL, marker, wildcard, compound or
   arbitrary-equality declaration is refused and reported (`unattributable`), never
   approximated into a pin the declaration did not make.
5. **The engine's report is input (L1, L4).** `--format json` parsed strictly (duplicate
   keys refused, shape checked field by field, depth bombs a verdict); a `skip_reason`
   entry is a finding independently of `--strict`; a finding is recorded as its ID, aliases
   and fix versions, tagged with the run that produced it — never the advisory prose.
6. **Egress redacted (L7).** Every string the report carries from the engine (`stderr_tail`,
   `stdout_tail` on unusable output, `vulnerable`, `skipped`, `unattributable`) passes
   through the same URL-userinfo/query-secret predicate the declaration does.
7. **Bounded (L8).** Both runs go through the shared bounded runner: own session, 300 s
   wall bound, 1 MiB per stream, process-group put-down; timeouts, floods and held pipes
   are the run's fail-closed state.
8. **Configurable as the other rails are (L13).** `PIP_AUDIT` names the engine (a path or a
   command), resolved through the same lookup the default takes; a configured engine that
   does not resolve is reported, never replaced by PATH's.
9. **Mocked in the quick tier (L19).** Every unit witness fakes discovery and the bounded
   runner together (and clears `PIP_AUDIT`); the redaction witness drives a stub engine
   through the full spawn path. A host's real pip-audit never decides a unit verdict.

Report shape (`--json-out`), abbreviated from the live run:

```json
"advisory": {
  "state": "PASS", "engine": "pip-audit", "declared": 6, "audited": 46,
  "covered": ["numpy", "pre-commit", "ruff", "safetensors", "setuptools", "torch"],
  "uncovered": [],
  "runs": {
    "resolved": {"state": "PASS", "audited": 40, "returncode": 0, "stderr_tail": "No known vulnerabilities found\n"},
    "floor":    {"state": "PASS", "audited": 6,  "returncode": 0, "stderr_tail": "…No known vulnerabilities found\n"}
  }
}
```

## 5. What this audit does not cover

- **The closure between floor and resolution.** An unlocked `>=` declaration admits every
  version above its floor; only the two ends are audited. A lockfile would make the set
  finite (§6.4).
- **Other platforms' closures.** The resolved run's transitive set is the resolving
  platform's (the CUDA wheels above are Linux x86-64's); CI resolves on `ubuntu-latest`.
  The floor run is platform-independent. The Windows and ARM jobs install the same declared
  names and are not separately audited.
- **The engine's own supply chain.** `pip-audit` (and `build`) are installed by version pin
  without hashes, in the job that then trusts their output. The same is true of every
  action's transitive tooling. Hash-pinning CI tools is §6.4's second half.
- **Native dependencies.** apt and conda-forge packages (LLVM/MLIR, clang, compiler-rt) are
  not advisory-scanned; the distributions' own security processes are relied on.
- **GitHub Actions currency.** SHAs are pinned and the pinning is test-enforced, but whether
  each SHA is the current release of its action was not checked from this host; the runner's
  Node-20 deprecation warning shows two of the five are behind (§2).
- **Static analysis of the repository's own code** is CodeQL's job (three languages, green on
  every recent run) and out of this audit's scope.

## 6. Recommendations

1. **Move the model-lab pins in their own PR** — torch 2.13.0 → 2.14.0 and numpy 2.2.6 →
   2.4.6 — gated by both hosted train-to-C jobs and the model-lab parity gates. The pins
   are numerics-bearing: a version move is a measured change with a stated tolerance, not a
   hygiene edit, which is why it is not folded into this slice. Nothing forces it today
   (0 advisories); the reason to move is the shrinking distance to the next security fix
   landing only on a newer line.
2. **`build` 1.2.2.post1 → 1.6.0** in the `python-floor` job: pure packaging, low risk, and
   the pin is old enough that its next advisory would arrive without a fix on that line.
3. **Move `actions/checkout` and `actions/setup-python` to their Node-24 releases** (the runner
   already warns on every job), annotate the three unannotated action pins with their release
   tags, and, from a host with GitHub API access, compare all five SHAs with upstream's
   current releases.
4. **Decide on a lockfile** for the extras and the CI tools (`pip-compile --generate-hashes`
   or `uv export`): it would let the resolved run become `--require-hashes --disable-pip`
   (no resolver, no scratch environment, and the audit's only unbounded network dependency
   gone) and would hash-pin `pip-audit` and `build` themselves. Until then the two-run scheme
   above is the honest coverage of an unlocked declaration.
5. **Cadence.** The rail runs on every push and pull request; this note is superseded by the
   first FAIL it produces or by the next dated audit. When a floor fails, raise the floor;
   when a resolved package fails, either the pin moves or the finding is fixed upstream —
   there is no allowlist by design (`--ignore-vuln` is not wired), so a finding cannot be
   made quiet without changing a declaration the inventory then has to follow.
