# Adversarial LLVM IR and BCIR lowering fixtures

These fixtures train reviewers and fuzzers to distinguish **LLVM verifier
validity** from **BCIR lowering correctness**. A fixture is evidence for a
specific risk, not a generally reusable implementation pattern. Read its
classification marker before deciding how to execute it.

## Classification contract

Every fixture starts with exactly one marker:

```llvm
; adversarial-class: assemble-valid-semantically-risky
; adversarial-class: intentionally-invalid
; adversarial-class: target-specific
; adversarial-class: metadata-preservation
```

| Class | Expected treatment |
|---|---|
| `assemble-valid-semantically-risky` | `llvm-as` and `opt -passes=verify` should accept it, but a semantic or BCIR-aware review must report the documented hazard. |
| `intentionally-invalid` | `llvm-as` or the LLVM verifier must reject it. Keep these fixtures as `*.invalid.ll.txt`. |
| `target-specific` | Parse/verify separately and only run backend checks with the documented triple and features. Never put it in portable smoke tests. |
| `metadata-preservation` | Parse/verify it and check that a textual `opt -passes=verify -S` round trip retains the marker named by `adversarial-preserve-metadata`. Transform-specific tests must additionally check the real pass pipeline. |

Run the classifier with:

```sh
llvm-training/tools/verify-adversarial-fixtures.sh
```

Use `LLVM_SUFFIX=-18` (or another installed suffix) when tools are versioned.
The script classifies fixtures; it does **not** treat every adversarial input as
an expected rejection.

## Threat-model categories

| Category | Adversarial question | Typical oracle |
|---|---|---|
| Verifier-valid poison hazards | Can `poison` reach a branch, switch, address, call target, or other immediate-UB use after speculation? | Track poison-producing flags/operations and require `freeze` or a proof before sensitive use. |
| Metadata dropped by transforms | Did instruction replacement erase debug, profile, alias, or BCIR correspondence evidence? | Compare required attachments and named side tables before/after each pass. |
| Address-space confusion | Was a non-generic pointer collapsed into address space zero, or was `addrspacecast` assumed to preserve dereferenceability? | Preserve address spaces in the mapping table and apply target-specific cast rules. |
| Operand bundle loss | Did cloning, outlining, or call rebuilding drop `deopt`, `funclet`, GC, or custom semantic bundles? | Compare bundle tags, order, and operands at every rewritten call/invoke. |
| Stale debug info | Do retained locations or variable intrinsics now describe a removed value, wrong scope, or misleading source operation? | Run debug-info verification where available and review location/value provenance after transforms. |
| BCIR 1:1 mapping drift | Were distinct BCIR registers merged, folded, renamed without replacement evidence, or mapped to one LLVM value? | Diff stable BCIR IDs and their LLVM definitions, metadata, or side-table records. |
| Target-specific intrinsic misuse | Is an intrinsic used with the wrong target, feature set, signature, immediate constraints, or fallback policy? | Gate backend tests by triple/features and consult target intrinsic constraints. |
| ABI attribute mismatch | Do declaration, definition, call site, and function-pointer type disagree on `sret`, `byval`, `inreg`, `signext`, `zeroext`, calling convention, or alignment? | Compare the complete ABI contract at all boundaries, not just LLVM types. |
| `memory(...)` overclaiming | Does a function or call claim fewer effects than its implementation/runtime behavior, enabling invalid motion or DCE? | Infer actual reads/writes/locations and reject attributes stronger than the proven contract. |
| Varargs ABI assumptions | Does lowering assume one platform's promotions, register-save area, aggregate passing, or `va_list` layout? | Test per target ABI and keep varargs marshaling behind target-aware lowering. |

Several categories have focused examples in this directory. Every category is a
required review and fuzz dimension even when a target/version-specific executable
fixture would be less portable than the lesson.

## Focus fixtures

- [`poison-branch-valid-risk.ll`](poison-branch-valid-risk.ll): valid IR whose
  branch condition can become poison.
- [`metadata-drop-risk.ll`](metadata-drop-risk.ll): a metadata-preservation
  seed with both an instruction attachment and a named BCIR side table.
- [`address-space-collapse-risk.ll`](address-space-collapse-risk.ll): a valid
  cast that is not proof that the target permits generic dereference.
- [`operand-bundle-loss-risk.ll`](operand-bundle-loss-risk.ll): a call whose
  `deopt` bundle must survive call reconstruction.
- [`bcir-mapping-drift-risk.ll`](bcir-mapping-drift-risk.ll): two logical BCIR
  registers with equal values that must remain independently reconstructible.
- [`target-specific-intrinsic-misuse-risk.ll`](target-specific-intrinsic-misuse-risk.ll):
  an x86-only intrinsic seed kept out of portable backend smoke tests.
- [`abi-attribute-mismatch.invalid.ll.txt`](abi-attribute-mismatch.invalid.ll.txt):
  an intentionally invalid ABI attribute/type combination.

The numbered legacy fixtures remain useful semantic-only seeds. Their
`*.invalid.ll.txt` spelling predates this classifier; their marker, rather than
the suffix alone, defines the expected result.

## Review workflow

1. Record the fixture class, target triple/features, pass pipeline, LLVM version,
   random seed, and exact command line.
2. Check syntax and generic verifier validity when the class requires it.
3. State the semantic invariant independently of LLVM acceptance.
4. Minimize the reproducer without deleting metadata, operand bundles, target
   facts, BCIR IDs, or ABI attributes needed to reproduce the failure.
5. Compare pre- and post-transform semantic inventories: values, memory effects,
   address spaces, bundles, debug records, and BCIR correspondence.
6. Add a deterministic assertion only for stable output. Prefer semantic checks
   over whole-file snapshots that churn across LLVM releases.

## Fuzzing guidance

- Generate from typed operations and explicit invariants, then mutate one risk
  dimension at a time. Blind text mutation mostly rediscovers parser failures.
- Use two oracles: LLVM parse/verify **and** a BCIR correspondence/semantic
  oracle. `llvm-as` success is necessary for valid seeds, never sufficient for
  lowering correctness.
- Preserve reproducer metadata: seed, generator revision, LLVM version, target
  triple/features, data layout, pipeline, expected class, BCIR IDs, and the
  first failing stage.
- Keep target-specific corpora and backend execution out of portable smoke jobs.
- Test pass prefixes to identify the first transform that loses an invariant.
- When reducing a failure, pin required metadata and operand bundles so the
  reducer cannot “fix” the bug by deleting the evidence.
- For `memory(...)`, ABI attributes, and varargs, fuzz declaration, definition,
  and call-site contracts together; local instruction validity is not enough.

A useful reproducer header is:

```text
seed=<integer> generator=<revision> llvm=<version>
target=<triple> features=<features> datalayout=<layout>
pipeline=<passes> class=<classification> first-failing-stage=<stage>
bcir-ids=<stable IDs> preserved-metadata=<metadata kinds>
```

## Pitfalls to avoid

- **Assuming `llvm-as` success means lowering correctness.** It only establishes
  that the module parses and passes the checks performed during assembly.
- **Deleting metadata because the verifier does not require it.** Debug,
  optimization, and BCIR reconstruction contracts can outlive an instruction.
- **Making target-specific examples part of portable smoke tests.** Parse them in
  a separate class and gate code generation/execution on target support.
- **Fuzzing without preserving reproducer metadata.** A minimized module without
  its seed, target, pipeline, and BCIR IDs is often not actionable.

## Related training

- [Pitfalls](../../08-pitfalls/) for verifier failures and semantic traps.
- [Advanced IR](../../13-advanced-ir/) for poison, attributes, intrinsics, and
  operand bundles.
- [New pass manager](../../17-new-pass-manager/) for pipeline instrumentation,
  pass isolation, and preservation analysis.
- [BCIR normal forms and verification](../../bcir-mapping/11-normal-forms-and-verification.md)
  for the stricter-than-LLVM mapping contract.
- [BCIR patterns index](../../indexes/bcir-patterns.md) for adjacent lowering,
  verification, and exercise material.
