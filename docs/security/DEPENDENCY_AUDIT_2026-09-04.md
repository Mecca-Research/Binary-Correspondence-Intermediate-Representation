# Dependency audit — 2026-09-04 (full surface)

> Dated and non-normative. The 2026-09-03 audit ([`DEPENDENCY_AUDIT_2026-09-03.md`](DEPENDENCY_AUDIT_2026-09-03.md))
> covered the declared Python inventory with one engine and built the rail that now runs it in CI.
> This audit covers **everything the repository depends on** — the declared inventory again, the
> modules the code actually imports, the CI-only tools, the GitHub Actions and pre-commit pins, the
> Ubuntu packages the workflows install, the native toolchains, and vendored code — with **two
> independent advisory sources** wherever two exist. Every number below was produced by a real run on
> this date; the raw outputs are committed in [`audit-2026-09-04/`](audit-2026-09-04/) and the tooling
> that produced them is [`tools/security/audit/`](../../tools/security/audit/README.md), so the next
> audit is a diff. The generated [`STATUS.md`](../STATUS.md) owns live counts; the gate-authoring laws
> cited (L1–L21) are [`laws.md`](laws.md).

## 1. Verdict

- **No known vulnerability in anything the repository installs from PyPI**, by two sources.
  pip-audit 2.10.1 against PyPI's advisory API audited 46 package-versions through the rail (the
  declaration resolved to 40, plus the 6 declared floors), 31 for the CI-only tools and their
  closure, and 2 more (the kafka-python and lit latest releases): 0 findings, 0 skips. An
  independent, offline evaluation of the same 77 (name, version) pairs against OSV's PyPI export
  (25,276 advisory records, 549 withdrawn ones skipped): 0 affected. 19 of the 77 have advisory
  history, and six of the versions in use are the **first release clear of an advisory** — the
  audit sits exactly at the edge it should (§4.1): `setuptools` 83.0.0, `torch` 2.13.0, `urllib3`
  2.7.0, `jinja2` 3.1.6, `kafka-python` 2.3.2, `mpmath` 1.3.0.
- **One cross-source discrepancy, root-caused.** The first OSV pass flagged `torch` 2.13.0 with nine
  PYSEC records. PyPI's API lists none for 2.13.0 (and eight for 2.8.0). The cause is the feed, not
  torch: those PYSEC records carry `last_affected` values that are not PEP 440 (`2.6.0-NA`,
  `2.6.0-cu124`, `2.5.0-NA`, `2.7.1-NA`), and a version evaluator that falls back to text comparison
  puts `2.13` below `2.6`. The evaluator now reads the dotted numeric prefix and records anything
  else as unevaluable; corrected result 0, in agreement with PyPI (§4.1).
- **Every GitHub Action and pre-commit pin is a genuine release commit**, none has an advisory in
  OSV's GitHub Actions export (55 records over 50 actions; none of ours), and four of the five
  actions are **two to three majors behind** upstream: `actions/checkout` v4.3.1 → v7.0.1,
  `actions/setup-python` v5.6.0 → v7.0.0, `actions/cache` v4.3.0 → v6.1.0, `actions/upload-artifact`
  v4.6.2 → v7.0.1. The runner already warns that the pinned builds of the first two target Node.js
  20 and are being forced onto Node 24. Exact target SHAs are in §5 (F5); this PR moves only the
  same-major patch (`awalsh128/cache-apt-pkgs-action` v1.6.1 → v1.6.3) and annotates every pin with
  the release it is.
- **Ubuntu packages the workflows install: nothing affecting, several open.** At the versions
  Ubuntu 24.04's archive resolves today, every advisory with a fix in noble is fixed at or below the
  installed version for all 12 source packages (LLVM 18 toolchain, python3.12, curl, cmake, ccache,
  valgrind, git, ninja-build, cppcheck, mono, nodejs, llvm-defaults). Advisories with **no fix in
  noble** remain open for `nodejs` 18.19.1 (20: 2 high, 18 medium — Node 18 is upstream end-of-life
  since 2025-04-30), `mono` 6.8 (4), `cppcheck` 2.13 (1), `ninja-build` 1.11.1 (1), `git` 2.43 (1,
  low). These run only on ephemeral CI runners against the repository's own inputs (§4.3).
- **The code imports exactly what it declares, plus one thing it did not.** A static scan of all
  552 Python files finds only `torch` and `safetensors` (both declared in `model-lab`), `lit` (LLVM's
  test runner, provided by the LLVM toolchain, in `llvm-training/tests/lit.cfg.py`), and
  `kafka-python`, imported lazily by `bcir.telemetry.KafkaSink.connect()` and declared nowhere, so no
  advisory scan could ever audit it. It is now the `telemetry-kafka` extra with floor 2.3.2 — the
  first release clear of its four advisories — and the rail audits it (`covered=7/7`).
- **One vendored third-party file, and it is 6 commits behind a dormant upstream.**
  `llvm-training/10-grammar/llvm-ir.tm` is `ll.tm` from `llir/grammar` at commit `5a3820b`
  (2022-08-02, 0BSD/Unlicense; byte-identical apart from its six-line provenance header). Upstream
  made six more commits the same day (LLVM 15 syntax) and none since. No other vendored code exists:
  no foreign copyright or SPDX lines, no `third_party`/`vendor` trees, no CMake `FetchContent` or
  `ExternalProject`; the MLIR rail depends on LLVM/MLIR through `find_package` only.

## 2. Scope and method

| Surface | Source of truth | Engine / check | From this host |
|---|---|---|---|
| Declared Python inventory | `pyproject.toml` ↔ `tools/security/expected_inventory.json` | the rail: `audit_dependencies.py --require-advisory` (pip-audit 2.10.1, PyPI advisory API; resolved + floor runs) | ran |
| CI-only Python tools | `pip install` lines in `.github/workflows/*.yml` | pip-audit, resolved run over `build`, `pip-audit`, `ninja`, `clang-format` | ran |
| Second advisory source | OSV bulk export, `PyPI/all.zip` | `tools/security/audit/osv_pypi.py` (offline evaluation) | ran (the GCS export is reachable; `api.osv.dev` is not) |
| Actual imports | every `.py` under the tree | `tools/security/audit/import_scan.py` | ran |
| GitHub Actions, pre-commit hooks | `uses:` SHAs, `rev:` tags | `tools/security/audit/actions_currency.py` (`git ls-remote --tags` against upstream) + OSV `GitHub Actions/all.zip` | ran (git reaches github.com; the HTTP API does not) |
| Ubuntu packages the workflows install | `packages:` / `apt-get install` lines | `tools/security/audit/osv_ubuntu.py` (OSV `Ubuntu:24.04:LTS/all.zip`, versions from `apt-cache` on an up-to-date 24.04.4 host, `dpkg --compare-versions`) | ran |
| torch's CPU-wheel closure (`download.pytorch.org/whl/cpu`, the hosted jobs' actual install) | `ci.yml` hosted train-to-C jobs | — | **not run**: the index is blocked here (§5 F12) |
| LLVM/MLIR 22 and 23 from apt.llvm.org | `mlir-rail-validate` matrix | — | no advisory database covers apt.llvm.org packages; upstream's release process is relied on |
| Windows and ARM closures | matrix jobs | — | not resolved (platform-specific closures; the declared names and versions are the same) |
| Static analysis of the repository's own code | GitHub CodeQL, default setup | c-cpp, python, actions | green on every recent run (GitHub-managed; not pinned by the repository) |
| Secrets | the whole tracked tree | `tools/security/scan_secrets.py` | ran: PASS, 1783 tracked files, 0 findings |
| conda-forge local rail (`tools/local`) | this host only | — | out of scope (developer convenience, not CI) |

## 3. Inventory

### 3.1 Python — declared, resolved, floors (PyPI)

| Surface | Declared | Resolves to today | Floor audited | Note |
|---|---|---|---|---|
| build-system | `setuptools>=83.0.0` | 84.0.0 | 83.0.0 | 83.0.0 is the first release clear of GHSA-h35f-9h28-mq5c; latest 84.0.0 |
| runtime | *(none)* | — | — | asserted empty: the oracle is dependency-free by design |
| `dev` | `ruff>=0.6` | 0.16.6 | 0.6 | |
| `dev` | `pre-commit>=3.5` | 4.6.2 (+ cfgv, identify, nodeenv, pyyaml, virtualenv → distlib, filelock, platformdirs, python-discovery) | 3.5 | |
| `model-lab` | `torch==2.13.0` | 2.13.0 (+ 18 CUDA-13 runtime wheels, triton 3.7.1, sympy, mpmath, networkx, fsspec, jinja2, markupsafe, filelock, typing-extensions) | 2.13.0 | 2.13.0 is the first release clear of GHSA-rrmf-rvhw-rf47 (CVE-2025-3000); latest 2.14.0 |
| `model-lab` | `safetensors[numpy]==0.8.0` | 0.8.0 | 0.8.0 | current |
| `model-lab` | `numpy==2.2.6` | 2.2.6 | 2.2.6 | latest 2.4.6 |
| `telemetry-kafka` *(new)* | `kafka-python>=2.3.2` | 3.0.11 | 2.3.2 | see F1 |

### 3.2 Python — CI-only tools

| Tool | Pinned | Where | Closure audited |
|---|---|---|---|
| `build` | 1.2.2.post1 → **1.6.0** (this PR; 1.2.2.post1 dates from 2024-10-06) | `python-floor` job | with `pyproject-hooks`, `packaging` |
| `pip-audit` | 2.10.1 | `security-assurance` job | 27 packages (CacheControl, cyclonedx-python-lib, requests, rich, urllib3, …) |
| `ninja` | 1.13.0 | Windows MLIR build (`pip install … \|\| true`) | 1 |
| `clang-format` | 18.1.8 | `mirrors-clang-format` hook (PyPI wheel) | 1 |
| `torch` (CPU wheel) | 2.13.0 from `download.pytorch.org/whl/cpu` | hosted train-to-C jobs | torch itself audited (§3.1); the CPU wheel's closure not resolved here (F12) |
| `safetensors[numpy]`, `numpy` | 0.8.0, 2.2.6 | hosted train-to-C jobs | as §3.1 |

### 3.3 Python — what the code actually imports

552 files scanned; four third-party modules:

| Module | Files | Unguarded / guarded imports | Declared by |
|---|---|---|---|
| `torch` | 17 | 7 / 25 | `model-lab` |
| `safetensors` | 4 | 1 / 3 | `model-lab` |
| `kafka` | 1 (`bcir/telemetry.py`) | 0 / 1 (lazy, in `KafkaSink.connect`) | **nothing** → now `telemetry-kafka` (F1) |
| `lit` | 1 (`llvm-training/tests/lit.cfg.py`) | 1 / 0 | the LLVM toolchain (lit's own config file, read by lit) |

Everything else the tree imports is the standard library or the repository's own packages. The
`import_graph.py --check` quarantine separately proves the hot path imports nothing heavy.

### 3.4 GitHub Actions and pre-commit hooks

| Pin | Uses | Pinned commit is | Latest upstream | Latest in pinned major | Majors behind | Disposition |
|---|---|---|---|---|---|---|
| `actions/checkout@34e11487…` | 13 | v4.3.1 | v7.0.1 (`3d3c42e5aac5ba805825da76410c181273ba90b1`) | v4.4.0 (`11d5960a3267…`) | 3 | annotated; **moved to v7.0.1 in the follow-up PR** (F5) |
| `actions/setup-python@a26af69b…` | 4 | v5.6.0 | v7.0.0 (`5fda3b95a4ea…`) | v5.6.0 | 2 | annotated; **moved to v7.0.0 in the follow-up PR** (F5) |
| `actions/cache@0057852b…` | 2 | v4.3.0 | v6.1.0 (`55cc8345863c…`) | v4.3.0 | 2 | annotated; **moved to v6.1.0 in the follow-up PR** (F5) |
| `actions/upload-artifact@ea165f8d…` | 3 | v4.6.2 | v7.0.1 (`043fb46d1a93…`) | v4.6.2 | 3 | annotated; **moved to v7.0.1 in the follow-up PR** (F5) |
| `awalsh128/cache-apt-pkgs-action@681749ae…` | 8 | v1.6.1 | v1.6.3 (`553a35bb8ebd9fcabcb1c9451aa4c98e1b4ca8a9`) | v1.6.3 | 0 | **moved to v1.6.3 in this PR** (F2) |
| `pre-commit/pre-commit-hooks` rev v5.0.0 | 1 | v5.0.0 | v6.0.0 | v5.0.0 | 1 | recommended (F10) |
| `pre-commit/mirrors-clang-format` rev v18.1.8 | 1 | v18.1.8 | v23.1.0 | v18.1.8 | 5 | recommended with a reformat commit (F9) |

Full SHAs for every target are in [`audit-2026-09-04/actions_currency.json`](audit-2026-09-04/actions_currency.json).
SHA pinning and read-only tokens are test-enforced (`test_workflow_dependencies_are_sha_pinned_and_tokens_are_read_only`).
CodeQL runs through GitHub's default setup, so `github/codeql-action` (two OSV records) is
GitHub-managed and not pinned by the repository.

### 3.5 Runners and the Ubuntu packages the workflows install

Runners: `ubuntu-latest` (24.04), `ubuntu-24.04-arm`, `windows-latest`; Python 3.12.14 from the
setup-python toolcache on the jobs that use it. Package versions below are what an up-to-date
Ubuntu 24.04.4 host resolves from the archive on this date (runner images refresh weekly, so a
runner may lag by days, never lead).

| Binary packages (workflow) | Source package | Version today | Advisories on record | Fixed at or below | Affecting | Open, no fix in noble |
|---|---|---|---|---|---|---|
| `clang lld llvm clang-tools libclang-rt-dev` | llvm-defaults | 1:18.0-59~exp2 | 0 | 0 | 0 | 0 |
| `clang-18 lld-18 llvm-18 libclang-rt-18-dev` | llvm-toolchain-18 | 1:18.1.3-1ubuntu1 | 0 | 0 | 0 | 0 |
| `nodejs` (WASM tests) | nodejs | 18.19.1+dfsg-6ubuntu5 | 34 | 14 | 0 | **20** (high: CVE-2023-44487, CVE-2024-22017; 18 medium) |
| `mono-devel` (CIL tests) | mono | 6.8.0.105+dfsg-3.6ubuntu2 | 4 | 0 | 0 | 4 (3 medium, 1 low) |
| `valgrind` | valgrind | 1:3.22.0-0ubuntu3 | 0 | 0 | 0 | 0 |
| `cppcheck` | cppcheck | 2.13.0-2ubuntu3 | 1 | 0 | 0 | 1 (medium, CVE-2023-39070) |
| `cmake` | cmake | 3.28.3-1build7 | 0 | 0 | 0 | 0 |
| `ninja-build` | ninja-build | 1.11.1-2 | 1 | 0 | 0 | 1 (medium, CVE-2024-36823) |
| `ccache` | ccache | 4.9.1-1 | 0 | 0 | 0 | 0 |
| `python3.12` (system) | python3.12 | 3.12.3-1ubuntu0.16 | 16 | 16 | 0 | 0 |
| `git` (runner) | git | 1:2.43.0-1ubuntu7.3 | 7 | 6 | 0 | 1 (low, CVE-2018-1000021) |
| `curl` (runner) | curl | 8.5.0-2ubuntu10.13 | 13 | 13 | 0 | 0 |

The `mlir-rail-validate` matrix additionally installs `llvm-23-dev`, `libmlir-23-dev`,
`mlir-23-tools` (and the 22 set) from `apt.llvm.org`'s `llvm-toolchain-noble-23`/`-22`; no advisory
database indexes those packages. The `security-assurance` job's `libclang-rt-18-dev` matches the
runner's system clang 18, which is what emits the sanitizer instrumentation.

### 3.6 Vendored code

One file: `llvm-training/10-grammar/llvm-ir.tm`, the Textmapper grammar `ll.tm` from
`llir/grammar` at `5a3820b516f7903e27ad16ebe4add1ec634f1c05` (2022-08-02), under 0BSD/Unlicense,
attributed in `llvm-training/NOTICE.md`. Byte-identical to that commit apart from the six-line
provenance header the repository prepends. Upstream HEAD is `05deced` (also 2022-08-02), six commits
later, all LLVM 15 syntax updates (parameter/function attributes, `AllocKind`, atomic `fmin`/`fmax`,
`DISubprogram.targetFuncName`, sanitizer globals, removed constant expressions); the vendored copy
differs from it by 208 lines. The header says the snapshot was last verified against LLVM 18.1.3 on
2026-06-05; the rail now targets LLVM 23. No other third-party source is vendored anywhere.

## 4. Advisory results

### 4.1 PyPI — two sources

The rail ([`audit-2026-09-04/rail_report.json`](audit-2026-09-04/rail_report.json), and
[`rail_report_after_fixes.json`](audit-2026-09-04/rail_report_after_fixes.json) with the new extra):

```
PIP_AUDIT=<venv>/bin/pip-audit python tools/security/audit_dependencies.py --require-advisory
  before: PASS asserted=True expected=6 mismatches=0 advisory=PASS audited=46 covered=6/6   (16 s)
  after:  PASS asserted=True expected=7 mismatches=0 advisory=PASS audited=48 covered=7/7
  runs.resolved: PASS  runs.floor: PASS  "No known vulnerabilities found" on both
```

CI-only tools and extras ([`pip_audit_ci_tools.json`](audit-2026-09-04/pip_audit_ci_tools.json),
[`pip_audit_extra.json`](audit-2026-09-04/pip_audit_extra.json)): 31 + 2 audited, 0 findings, 0 skips.

OSV offline evaluation ([`osv_pypi.json`](audit-2026-09-04/osv_pypi.json)) over the union — the
resolved and floor sets, the CI-only closure, `setuptools` 83.0.0 and 84.0.0, `pip` 26.2.1, `wheel`
0.48.0, `kafka-python` 2.3.2 and 3.0.11, `lit` 23.1.0 — 77 pairs:

| Package in use | Advisory records | Affected | First release clear of |
|---|---|---|---|
| setuptools 83.0.0 / 84.0.0 | 10 | none | 83.0.0: GHSA-h35f-9h28-mq5c (PYSEC-2026-3447) |
| torch 2.13.0 | 43 | none (see below) | GHSA-rrmf-rvhw-rf47 / CVE-2025-3000 |
| urllib3 2.7.0 | 38 | none | GHSA-mf9v-mfxr-j63j, GHSA-qccp-gfcp-xxvc, PYSEC-2026-141, PYSEC-2026-142 |
| pip 26.2.1 | 26 | none | (26.2.0 fixed the latest) |
| jinja2 3.1.6 | 20 | none | GHSA-cpwx-vrp4-4pq7, PYSEC-2026-1471 |
| numpy 2.2.6 | 16 | none | |
| requests 2.34.2 | 16 | none | |
| pygments 2.21.0 | 10 | none | |
| pyyaml 6.0.3 | 8 | none | |
| certifi 2026.7.22, virtualenv 21.7.8 | 6, 6 | none | |
| filelock 3.32.5, idna 3.19, markdown-it-py 4.2.0, kafka-python (both), wheel 0.48.0 | 4 each | none | kafka-python 2.3.2: GHSA-2jcm-hq8r-84wx, GHSA-m3px-q5gj-j9x7, PYSEC-2026-2190, PYSEC-2026-2191 |
| mpmath 1.3.0, msgpack 1.2.2 | 2, 2 | none | mpmath 1.3.0: GHSA-f865-m6cq-j9vx, PYSEC-2021-427 |
| the other 58 pairs | 0 | none | |

**The torch discrepancy.** The first evaluator flagged torch 2.13.0 with PYSEC-2025-189, -190,
-192, -193, -194, -195, -196, -197 and -210. Each of those records encodes its affected range as
`introduced: 0` with `last_affected` values of `2.6.0-cu124`, `2.6.0-NA`, `2.5.0-NA` or `2.7.1-NA`
— build tags and placeholders appended to the version, which PEP 440 does not admit — and an
evaluator that falls back to text comparison for an unparsable value concludes `"2.13.0" <=
"2.6.0-NA"`. PyPI's advisory API, which pip-audit reads, reports 0 vulnerabilities for torch 2.13.0
and 8 for torch 2.8.0; GHSA-rrmf-rvhw-rf47 (the CVE-2025-3000 record, the same issue as
PYSEC-2025-194) names 2.13.0 as its first fixed release. The evaluator now parses the dotted numeric
prefix of such values, never compares text, and records what it could not evaluate (39 values in the
export, all git refs such as `ciflow/…` in torch's `versions` lists); corrected result: 0. The
lesson is written into `osv_pypi.py`'s docstring, and the PYSEC records are worth an upstream report
(F13).

### 4.2 GitHub Actions

OSV's GitHub Actions export (55 records, 50 actions): none for `actions/checkout`,
`actions/setup-python`, `actions/cache`, `actions/upload-artifact` or
`awalsh128/cache-apt-pkgs-action`. The export does hold records for `actions/download-artifact`,
`actions/runner` (5) and `github/codeql-action` (2), none of which the repository pins. Currency is
§3.4; the pinned SHAs all resolve to release tags (no pin points at a non-release commit).

### 4.3 Ubuntu packages

§3.5 in full: 0 affecting, 27 open across five source packages. Exposure: these packages run on
ephemeral GitHub runners, on inputs the repository generates (its own WASM output under `node`, its
own CIL output under `mono`, its own C under `cppcheck`/`valgrind`), never on untrusted input and
never in a shipped artifact. The open `nodejs` set is the one worth acting on: the 18 line is
upstream end-of-life, so its open count only grows (F6).

### 4.4 Pre-commit hooks, vendored code, code scanning, secrets

Hooks: §3.4 (F9, F10). Vendored grammar: §3.6 (F11). CodeQL: green on c-cpp, python and actions on
every recent run; the `github-advanced-security` Copilot agent fails on every run for a GitHub-side
reason (its configured model is rejected by its backend) and is not part of this audit's evidence.
Secrets: `scan_secrets.py` PASS on this date (1783 tracked files, 1790 text, 1 binary, 0 findings).

## 5. Findings and dispositions

| # | Finding | Severity | Disposition |
|---|---|---|---|
| F1 | `kafka-python` imported (lazily) but declared nowhere: unauditable, uninstallable through the package | medium (coverage) | **Fixed here**: `telemetry-kafka = ["kafka-python>=2.3.2"]`, inventory and the `ImportError` text follow; the rail now covers 7/7 declared names |
| F2 | `awalsh128/cache-apt-pkgs-action` at v1.6.1, two patches behind v1.6.3 | low | **Fixed here**: pinned to `553a35bb8ebd9fcabcb1c9451aa4c98e1b4ca8a9` (v1.6.3), 8 uses; CI is the gate |
| F3 | Three of five action pins carried no version comment; the other two said only `v4`/`v5` | low (readability) | **Fixed here**: every pin annotated with the exact release it is |
| F4 | `build==1.2.2.post1` (2024-10-06) four minor releases behind | low | **Fixed here**: `build==1.6.0` in the `python-floor` job (requires Python ≥ 3.10; the floor is 3.11) |
| F5 | `checkout`, `setup-python`, `cache`, `upload-artifact` two to three majors behind; the first two are Node 20 builds the runner forces onto Node 24 | medium (currency; supply-chain age) | **Landed in the follow-up PR** (four commits, one per action, each reviewed against upstream's history: the only contract changes are the Node 24 runtime and additive inputs with behavior-preserving defaults; checkout v7's fork-PR guard concerns `pull_request_target`/`workflow_run`, which no workflow here uses). Targets as recorded: one action per commit, targets `actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1` (v7.0.1), `actions/setup-python@5fda3b95a4ea…` (v7.0.0), `actions/cache@55cc8345863c…` (v6.1.0), `actions/upload-artifact@043fb46d1a93…` (v7.0.1) — full SHAs in `actions_currency.json`; read each action's release notes for the breaking changes between majors first (cache keys, artifact naming and `upload-artifact` v4→v5+ semantics, checkout's credential handling) and let the full matrix be the proof. `test_workflow_dependencies_are_sha_pinned…` keeps the pins honest |
| F6 | `nodejs` from apt is the 18 line: upstream EOL, 20 open advisories in noble (2 high) | medium (runner hygiene) | **Recommended**: run the WASM tests on a maintained LTS through `actions/setup-node` (22 or 24) or on the runner image's preinstalled Node, and drop `nodejs` from the apt cache list; verify what `node` resolves to on the runner first |
| F7 | `mono` 6.8: 4 open advisories, no fix in noble | low | Accepted: CIL back-end tests only, on the repository's own output; revisit if noble moves mono |
| F8 | `cppcheck`, `ninja-build`, `git`: one open advisory each (medium/medium/low) | low | Accepted: developer tools on ephemeral runners |
| F9 | `mirrors-clang-format` v18.1.8 while the rail's toolchain is LLVM 23 | low (style drift) | **Recommended**: move to v23.1.0 with a single reformat commit of `runtime/c` and `mlir/`, nothing else in that PR |
| F10 | `pre-commit-hooks` v5.0.0 → v6.0.0 available | low | **Recommended** alongside F9 (local hooks; not CI-gated) |
| F11 | Vendored `llvm-ir.tm` is six commits behind a dormant upstream; header verified against LLVM 18 | low (corpus currency) | **Recommended**: refresh to `05deced` in an `llvm-training` slice and re-verify the header claim against LLVM 23's `llvm-as` |
| F12 | The hosted jobs' real torch closure (CPU wheels from `download.pytorch.org`) is not what PyPI resolves, and that index is unreachable here | low (coverage gap) | **Recommended**: after the hosted train-to-C job installs its wheels, run `pip-audit` in environment mode there (the closure exists only where it is installed); ~15 s per job |
| F13 | PYSEC torch records with non-PEP-440 `last_affected` values | info (feed quality) | Recorded in `osv_pypi.py`; worth an issue against `pypa/advisory-database` |
| F14 | Windows and ARM closures not separately audited; LLVM/MLIR apt.llvm.org packages have no advisory index | info | Accepted as declared limits (§2) |

## 6. Changes landed with this audit

- `pyproject.toml`, `tools/security/expected_inventory.json`, `bcir/telemetry.py`: the
  `telemetry-kafka` extra (F1).
- `.github/workflows/ci.yml`, `c-memory-deep-sweep.yml`: every action pin annotated with its release
  (F3); `awalsh128/cache-apt-pkgs-action` → v1.6.3 (F2); `build` → 1.6.0 (F4).
- `tools/security/audit/`: the four report-only tools and their README (§2).
- `docs/security/audit-2026-09-04/`: the raw outputs this document summarises.
- `SECURITY.md` and the 2026-09-03 note cross-link this audit; the agent digest carries the summary.

## 7. Reproduction

```bash
# from the repository root; a scratch venv with pip-audit==2.10.1 provides the engine and `packaging`
python tools/security/audit/import_scan.py --out work/import_scan.json
python tools/security/audit/actions_currency.py --out work/actions_currency.json
PIP_AUDIT=venv/bin/pip-audit python tools/security/audit_dependencies.py --require-advisory --json-out work/rail_report.json
printf 'build==1.6.0\npip-audit==2.10.1\nninja==1.13.0\nclang-format==18.1.8\n' > work/ci_tools.txt
venv/bin/pip-audit -r work/ci_tools.txt --format json --strict --desc off --progress-spinner off > work/pip_audit_ci_tools.json
venv/bin/python tools/security/audit/osv_pypi.py --osv-dir work/osv --report work/pip_audit_ci_tools.json ... --out work/osv_pypi.json
python tools/security/audit/osv_ubuntu.py --osv-dir work/osv --release 24.04 --binary clang --binary nodejs ... --out work/osv_ubuntu.json
```

Each tool prints its table and exits non-zero on a finding or an unreadable source. The rail itself
runs on every CI run of `security-assurance`; this document is superseded by the first FAIL it
produces or by the next dated audit.
