# BCIR security red-team audit — 2026-07-15

> **Scope:** merge of PR #538 (`8d88b530e3922d722fb02e0bcc30c89aadc9a630`)
> through the current audited base, merge of PR #639
> (`6e7cd7d4e32a0623721f530230c4fee10367288a`), plus the fixes described here.
> That interval contains 277 commits and touches 499 files. This is a bounded,
> repository-grounded audit, not a claim that all future defects have been proved absent.
>
> **Companion:** [`BCIR_SECURITY_THREAT_MODEL.md`](BCIR_SECURITY_THREAT_MODEL.md).

## Executive verdict

The audit found reachable memory-safety, integer-overflow, parser-ambiguity,
resource-exhaustion, race, path-replacement, lifecycle, and supply-chain defects. The
branch fixes them and pins each material class with deterministic adversarial tests.

No BCIR-originated local privilege escalation was demonstrated on the audited host.
That conclusion is not based on a pattern scan alone:

- the built repository contains no setuid/setgid file, file capability, privileged BCIR
  service, listener, device node, resident driver, or loaded BCIR kernel module;
- RuntimeChannel is currently a direct in-process vtable, not a user/kernel or
  cross-process transport;
- the C rail and generated programs execute with the invoking user's authority; and
- all confirmed BCIR defects therefore stop at same-process memory corruption,
  integrity loss, denial of service, or a **conditional** escalation if a future
  privileged embedding violates the threat-model rules.

One conditional high-severity local-escalation primitive did exist in a developer
bootstrap script: a privileged invocation could execute a pre-positioned
`/tmp/micromamba`. It was not exploitable in the observed state because that path was
root-owned and the host enables protected temporary-file policies, but the unsafe
fresh-host/multi-user condition was real and has been removed.

## Method and safety boundary

The review combined the following, in this order:

1. Compared every post-#538 change and inventoried executable entry points, wire
   decoders, model loaders, compiler frontends, MLIR passes, hosted C allocation,
   telemetry, plugin registries, generated-program harnesses, CI, and bootstrap tools.
2. Mapped data and ownership from attacker-controlled bytes to allocation, mutation,
   execution, persistence, and cleanup. Pattern searches were used only to find
   candidates; each finding was traced to a live caller and build target.
3. Reviewed size arithmetic, offsets, spans, growth, cleanup, duplicate identities,
   graph cardinality, recursion, generated claim/resource IDs, event publication,
   shared mutable state, close/reset behavior, and path replacement windows.
4. Exercised malformed and adversarial inputs on the Python, C, and MLIR rails and
   added a deterministic regression for each accepted defect class.
5. Checked the actual x86 WSL2 environment for privilege surfaces and the prerequisites
   of Copy-Fail/Dirty-Frag-style kernel attacks. No exploit payload or destructive host
   action was run.
6. Reviewed direct dependencies and CI action provenance. The Python runtime remains
   dependency-free.

The local campaign was deliberately bounded to two workers. It did not run unbounded
fuzzing, QEMU, ARM emulation, kernel exploit proof-of-concepts, or large inference.

## Observed environment and exploitability

| Evidence | Observed state | Security consequence |
|---|---|---|
| Identity | `uid=0(root)` in the WSL development environment | A root shell cannot be used to prove a local escalation. The audit instead looked for a BCIR surface reachable by a less-privileged principal. |
| Kernel | `6.18.33.2-microsoft-standard-WSL2`, x86-64 | Relevant kernel-CVE prerequisites were checked separately below. |
| Temporary-file protections | `fs.protected_symlinks=1`, `protected_hardlinks=1`, `protected_regular=2`, `protected_fifos=1` | Several classic shared-`/tmp` attacks are blocked in the observed state, but code was still fixed so safety does not depend on these sysctls. |
| Privileged artifacts | No setuid/setgid file or file capability in the repository | There is no installed BCIR privilege transition. |
| Resident interfaces | No BCIR listener, device node, loaded module, or `xfrm_iptfs` module | Current inputs do not cross into a privileged BCIR kernel/service context. |
| User namespaces | An unprivileged `unshare -Urn` probe succeeds; `CONFIG_USER_NS=y` | Namespace availability alone is not an exploit. It raises the importance of keeping the host kernel current. |
| Relevant kernel config | `XFRM_USER`, `INET_ESP`, and `CRYPTO_USER_API_AEAD` are modules | Some generic network attack prerequisites exist, but the specific residual `xfrm_iptfs` implementation is absent. |
| Direct RuntimeChannel | In-process hooks with borrowed storage and external serialization contract | A caller bug can corrupt its own process; there is no present kernel/user-copy or cross-tenant IPC boundary. |

## Confirmed findings and fixes

Severity describes impact if the affected boundary consumes hostile input in its
documented deployment. “Current exploitability” records the audited environment rather
than assuming a future privileged service already exists.

| ID | Severity | Finding and reachability | Current exploitability | Resolution and regression evidence |
|---|---|---|---|---|
| SR-01 | High | Hosted C loaders/executors had incomplete offset, multiplication, workspace, pointer-arithmetic, and failure-cleanup checks. A malformed Q8/StreamPack/model or hostile dimensions could reach reads, allocations, or partial mutation. | Reachable in the invoking process; no privilege boundary. Memory corruption/DoS was the realistic impact. | Added checked spans/arithmetic, complete preflight, allocator-aware cleanup, idempotent destroy, bounded source/model requests, and failure-preserving growth across Q8, Llama, decoder kernels, planner/hydrator/executor, diagnostics, training, telemetry frames, binary records, provenance, and C frontend/preprocessor. Allocator-failure, sanitizer, corruption, truncation, and strict-warning tests cover the paths. |
| SR-02 | High | C lifetime verification used a fixed 256-entry freed-resource set, allowing larger graphs to lose use-after-free/double-free history. | Reachable with a large hostile C claim graph; same-process compiler-integrity failure. | Replaced the fixed assumption with checked hosted storage and added a regression with more than 256 freed resources. |
| SR-03 | High | Artifact parsers accepted ambiguous duplicate JSON keys, coercive booleans/numbers, non-finite values, unknown/defaulted fields, overlapping/aliased payloads, holes, reserved bits, invalid UTF-8, and impractical counts. Different readers could authorize different artifacts. | Reachable through manifests, priors, proofs, telemetry, model/tokenizer files, compile databases, StreamPack, and BCIRQ8. Integrity/DoS; no current elevation. | Added bounded strict JSON helpers and exact schemas; reject duplicates, non-finite constants, recursion exhaustion, overlap/hole/suffix aliases, invalid digests, stale generations, reserved fields, malformed protobuf/varints, duplicate token ownership, and CRC-valid semantic corruption. |
| SR-04 | High | MLIR verifier/cost/lowering passes performed derived integer arithmetic and materialization without complete overflow/expansion bounds; recursive compose input could exhaust the stack. | Reachable by hostile MLIR input in `bcir-opt`; compiler crash/DoS or wrong cost, not privilege escalation. | Added checked add/multiply/casts, finite geometry/cost limits, materialization ceilings, and iterative/limited compose handling. `verify_overflow.mlir`, `gem_cost_overflow.mlir`, `lower_expansion_limits.mlir`, and `compose_recursive_neg.mlir` pin the refusals. |
| SR-05 | High | ROP resource resolution lived in process-global mutable state. Concurrent parses could bind one program's claims to another program's resource IDs. MAP/ROP also silently defaulted malformed/duplicate fields and had no practical inventory bounds. | Deterministically reachable in a multithreaded host compiler; cross-request integrity failure in one process. | Made resolution parser-local; added strict EOF, identity, opcode/lane/stride/source, duplicate, cardinality, and numeric checks. A barrier-controlled concurrent regression reproduces the old cross-talk and proves isolation. |
| SR-06 | High | Model serving accepted coercive token/dimension inputs and could mutate KV/cache/log state before all graph, weight, schema, or target validation completed. Session claim-ID bands could overlap; forged batch registries could receive meaningless certificates. | Reachable through local serving/reference APIs; integrity/DoS. No network service is shipped. | Added one strict decode boundary, context/shape limits, finite weight/input checks, DFA ownership snapshots, preflight-before-log/mutation, collision-free claim bands, complete batch graph certification, page/RID bounds, generation checks, and atomic session admission. |
| SR-07 | High | Telemetry publication and registries had data races: ring readers could observe a slot during overwrite, validation witness counts could race downstream publication, broker fanout could interleave, provider name/ID registration was non-atomic, and lazy exporter definitions raced. | Reachable from concurrent in-process publishers/providers; future driver risk would be higher. | Added locks around ring transitions, broker fanout, validating witnesses, signal/channel registration, snapshots, exporter cache, file/log emission, and policy/audit rings in hosted C. Regressions force the old temporal windows with barriers/events. |
| SR-08 | High | Telemetry file sinks followed attacker-controlled symlinks and did not bind subsequent appends to the originally created inode. Durable inputs were unbounded and could carry ambiguous/poisoned records. | Path attack requires write access to the selected output directory; no privileged telemetry daemon exists now. | Added private regular-file open, no-follow where supported, post-open identity checks, exclusive durable-log creation, replacement detection, size/line/record ceilings, strict schemas, and poisoned-ring rejection. A future privileged service must additionally own a non-attacker-writable parent directory. |
| SR-09 | High | The local MLIR setup defaulted to executable `/tmp/micromamba` and trusted an existing file; a privileged fresh-host invocation could run another user's pre-positioned binary. The download followed mutable `latest` without an integrity pin. | **Conditional LPE.** Not exploitable in observed state: the existing path was root-owned and protected. The code condition was nevertheless unsafe. | Default moved to a private user cache; directories must be owned and not group/world writable; symlink/non-regular/unowned binaries are refused; micromamba `2.8.1-0` and per-architecture SHA-256 values are pinned; HTTPS/TLS floor and private staging are enforced. A regression installs a marker payload and proves it is rejected before execution. |
| SR-10 | Medium | Validation scripts used predictable shared `/tmp` diagnostics/artifacts. Concurrent runs could delete or consume each other's files; a hostile local staging path could influence results. | Concurrent same-user corruption was reachable; cross-user exploitation was constrained by current sysctls and ownership. | Every affected WSL/IRDL/MLIR-22 script now uses a private `mktemp -d` root with cleanup. A governance test rejects reintroduction of fixed names. |
| SR-11 | High | The pinned-model downloader reused `target.part`. Concurrent gates could unlink each other's staging file, and a pre-existing link/path created a replacement window. | Reachable concurrent corruption/DoS in the model gate; target hashes prevented silent model substitution. | Each attempt now uses a private mode-0600 `mkstemp`, atomic replace, retry cleanup, size, and SHA verification. Concurrent-download and legacy-staging regressions prove isolation. |
| SR-12 | High | SYCL dispatcher first-use, cache lookup, and `close()` were unsynchronized. One thread could remove/replace a work directory or executable while another compiled or executed it; hostile geometry could create an excessive job. | Reachable through concurrent in-process dispatcher use; no resident GPU driver privilege. | Serialized lifecycle/dispatch, clear executable cache on close, reject use-after-close/stale paths, and bound vector/matmul geometry before compilation. |
| SR-13 | Medium | RuntimeChannel loopback state and borrowed hook/event storage had no explicit concurrency/reentry/lifetime contract; event-loss counters could wrap. | Accidental concurrent callers could corrupt same-process state. No IPC or kernel surface exists. | Documented external serialization, no-reentry, and borrowed-lifetime rules; saturated dropped-event accounting; added reset/overflow regressions. Concrete synchronized IPC remains deferred until driver traces exist. |
| SR-14 | Medium | Test execution defaulted to every CPU and inherited unbounded compiler/generated-program children. This reproduced a practical host resource-exhaustion/hang hazard. | Directly reachable during local validation; availability impact only. | Safe default is at most two workers, all-core use requires explicit opt-in, spawn/fork remains portable, and every test child receives a validated default timeout. Help is side-effect-free. Heavy gates remain serialized. |
| SR-15 | Medium | C and Python channel/plugin readers accepted inconsistent defaults, duplicate registration could replace a live backend, and the C routing name accepted escaped control characters. | Same-process routing or audit-log integrity; installed Python code is not sandboxed. | Exact manifest schema/type/provenance/calibration checks, bounded JSON, duplicate backend refusal, registry locking, C parser depth/size/identity checks, and control-free names. Test fixtures reuse an identical installed fixture without weakening production replacement refusal. |
| SR-16 | Medium | Generated graph/cost APIs accepted negative, non-finite, coercive, degenerate, or attacker-scaled dimensions. Several paths could underprice invalid DMA/memory/matmul work or run effectively unbounded loops. | Reachable local DoS/integrity error. | Added exact integer/finite/range checks, checked descriptor direction/copy spans, matrix/tile/group limits, accumulator bounds, proof/provenance completeness, and preflight-before-execution throughout K_BCIR and model paths. |
| SR-17 | Medium | GitHub workflows inherited repository-default token permissions and official actions used mutable major tags. The Python build accepted vulnerable setuptools releases. | Supply-chain compromise could execute in CI. The PR workflow has no deployment secret or write requirement, but unnecessary token authority increased impact. | Set workflow token to `contents: read`; SHA-pinned every action; governance rejects mutable refs and `pull_request_target`. Build floor is `setuptools>=78.1.1`, excluding [CVE-2024-6345](https://github.com/advisories/GHSA-cx63-2mw6-8hw5) and [CVE-2025-47273](https://github.com/advisories/GHSA-5rjg-fvgr-3xxf). Runtime dependencies remain empty. |

## Copy Fail, Dirty Frag, and local-escalation analogues

The repository was searched and traced for Linux user-copy, socket-buffer fragment,
page-fragment, pipe-buffer, slab-free/reuse, and kernel refcount primitives. BCIR has no
kernel implementation and no calls to `copy_from_user`, `skb_*`, `page_frag_*`,
`pipe_buffer`, `kmalloc`, or `kfree`. Consequently the Copy Fail and Dirty Frag
mechanisms are not reachable in BCIR code today.

The model-side `PagedKV` was reviewed as the closest semantic analogue. It does not
reuse evicted page storage between tenants. Eviction is refused while the session is
live, clears the view, advances the mapping generation, and rejects repeated access or
free. No stale page alias survived the current call graph.

The host-kernel disposition is:

| Advisory | Upstream fixed rail reported by NVD | Audited-host disposition |
|---|---:|---|
| [CVE-2026-31431 (Copy Fail)](https://nvd.nist.gov/vuln/detail/CVE-2026-31431) | 6.18.22 | Host `6.18.33.2` is newer. |
| [CVE-2026-43284 (Dirty Frag)](https://nvd.nist.gov/vuln/detail/CVE-2026-43284) | 6.18.28 | Host is newer. |
| [CVE-2026-43500 (Dirty Frag)](https://nvd.nist.gov/vuln/detail/CVE-2026-43500) | 6.18.29 | Host is newer. |
| [CVE-2026-46300 (Fragnesia)](https://nvd.nist.gov/vuln/detail/CVE-2026-46300) | 6.18.33 | Host vendor build is newer than the listed base fix. |
| [CVE-2026-53363](https://nvd.nist.gov/vuln/detail/CVE-2026-53363) | 6.18.36 | Nominal kernel is older, but the affected `xfrm_iptfs` implementation is neither installed nor loaded, so the vulnerable path is not reachable here. Update the WSL kernel when a patched vendor build is available. |

This closes the named vulnerability patterns for the observed environment; it does not
substitute for vendor kernel patching.

## Investigated and closed as non-exploitable in the current environment

| Candidate | Trace result |
|---|---|
| C frontend compatibility wrappers | Process-static wrappers are explicitly non-thread-safe; reentrant context APIs exist. No current concurrent privileged caller uses the compatibility wrappers. |
| Quarantine policy/site pointers | They are borrowed by contract and require static lifetime. Current generated sites are string literals and all policy call sites use static frozen tables. No dangling current caller was found. |
| Public Python `CHANNELS` mapping | Direct mutation bypasses the locked API, but code imported into the Python process already has arbitrary same-user code execution and can alter any module. This is not a sandbox boundary; future out-of-process plugins must not share this object. |
| Compiler and logits output paths | `-o`/`--logits-out` paths are explicitly selected by the invoking user, like ordinary compiler output. No privileged daemon supplies attacker-controlled paths. A future service must broker output descriptors rather than pass client paths. |
| Quiescent shared-ring parser | Documentation and code identify it as a snapshot reader, not a live acquire/release IPC consumer. The future SPSC driver ring is unimplemented and must not reuse the quiescent contract. |
| External compiler/tool hangs | Production developer APIs can invoke the resident toolchain. The complete test inventory now imposes timeouts; production service wrappers must impose their own deadline/cancellation policy because library calls cannot guess one universal SLA. |

## Dependency and CI disposition

- `bcir` has no runtime Python dependencies.
- The build dependency floor is now `setuptools>=78.1.1`.
- `ruff` and `pre-commit` are optional developer tools; no matching GitHub-reviewed
  advisory was found during this audit.
- All workflow `uses:` references are immutable 40-hex commit SHAs, including official
  actions. The third-party apt-cache action is also pinned, but apt package repositories
  remain an external trust dependency.
- GitHub Dependabot alert access was disabled for this repository during the audit.
  Enabling dependency alerts and automated action-update review is recommended; a
  disabled dashboard must not be mistaken for zero vulnerable dependencies.
- Workflow jobs are validation-only and receive `contents: read`.

## Validation record

The frozen local result is a bounded x86-64 WSL2 campaign with at most two workers and
heavy gates serialized:

| Gate | Result |
|---|---|
| Python quick | **2,132 passed, 0 failed** (`-j 2`) |
| Python thorough | **2,132 passed, 0 failed** (`-j 2`; compiler, LLVM IR/JIT/WASM, and large campaigns enabled) |
| Differential oracle | Two independent 8,000-case campaigns; **8,018/8,018 clean checks per seed**, zero verifier misses |
| Python trust-boundary fuzz | Two independent 4,000-case campaigns; **zero findings** |
| Complete C runtime | Passed strict C11/C23 builds, Python/C parity, malformed-input rejection, allocator faults, C++ seam, and portable SYCL fallback |
| C sanitizers and analysis | All 184 C-front fixtures plus bounded valid/malformed campaigns passed under Clang and GCC ASan/UBSan/LSan; Clang trapping-UB passed; **23/23** production units passed ownership analysis |
| Native C fuzz | Four libFuzzer targets at **500,000 runs each** plus ASan/UBSan mutation smoke; zero crashes |
| Real-model gate | Offline pinned Q8 Python/standalone-C parity passed; generated token `635`, deterministic artifact SHA-256 `bf6b460f78ed1937f55cae0875cfa76e10c874c63bdb47a226a7ace8601223de` |
| LLVM training | Aggregate passed with the coherent LLVM 22 toolset; autograder **700/700** |
| MLIR 22 | Full TableGen, R1-R23/GEM/pass, ODS, bytecode, and 22-only IRDL rail passed |
| Silicon/performance | Non-rig degrade path completed honestly; correctness and measurement-validity budgets passed, hardware floors waived |
| Repository governance | Generated status, links, retired paths, import quarantine, Python compilation, changed-shell syntax, and diff hygiene passed |

Local skips are explicit: Valgrind, cppcheck, and a SYCL device compiler are not installed;
the LLVM-training MLIR grading tier is pinned to LLVM 18 while this local rail is LLVM 22.
The full native MLIR 22 rail above closes local pass/IRDL coverage, while scheduled CI owns
the slower analyzer/Valgrind work. Native Windows and ARM64 remain GitHub-CI-gated, and no
ARM emulation was performed on this x86 workstation.

## Residual risk and required follow-up

1. BCIR is not a sandbox. Python plugins and native toolchains execute with the caller's
   authority. Treat untrusted plugins as separate processes when that feature exists.
2. Do not expose compiler/model/library APIs from a privileged service without request
   quotas, deadlines, private output directories/descriptors, seccomp/job isolation,
   and the capability/generation UAPI in the companion threat model.
3. The future live SPSC ring, driver UAPI, Linux module, DMA/IOMMU path, cancellation,
   hotplug, peer-death, and restart semantics are not implemented. They require a new
   pre-merge threat-model review and direct-vtable/IPC differential tests.
4. Borrowed C policy/hook storage must remain frozen for the documented lifetime. If
   dynamic replacement is needed, introduce owned snapshots or refcount/epoch
   reclamation rather than weakening the contract.
5. Scheduled cloud CI should continue long fuzz, Valgrind, cppcheck, scan-build, and
   multi-architecture runs. This local audit intentionally avoided unbounded work.
6. Repeat a full red team before a resident driver, after any allocator/wire-format
   change, and before each release. Every accepted defect needs a deterministic
   regression; an audit report alone is not a control.
