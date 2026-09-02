# The Security Gate Laws

A registry of the durable engineering laws extracted from the PR #749
adversarial hardening campaign (Claude ⇄ Codex, 30+ review rounds, ~100
closed findings). The Python gates that produced these laws are one
implementation; the laws are the contract. When BCIR components migrate to
C++ and C, **the laws port, the code does not** — each entry carries a port
note saying what the law becomes on the native side.

Witnesses name tests in `bcir/tests/test_security_assurance.py` that prove
the law can fire. Every law was learned from at least one live finding;
none is speculative.

## The harvest protocol

Each finding from an adversarial review round is graded on two axes:

* **Severity** (T0–T3): T0 verdict-flipping on realistic input · T1
  crash/resource/evasion on hostile input · T2 contract drift between
  paths or platforms · T3 spec-exotica inside declared subsets.
* **Harvest**: **NEW-LAW** (establishes a law not yet in this registry —
  the registry gains an entry) · **INSTANCE** (a new entry point to a
  registered law — the law gains a witness) · **LOCAL** (a fix with no
  transfer value beyond the code it touches).

The rolling three-round NEW-LAW rate is the loop's vital sign, tracked on
the campaign ledger alongside severity.

## The staleness rule (declared, not discretionary)

The adversarial review loop on a PR is **stale** when **three consecutive
review rounds yield zero NEW-LAW findings**. On staleness: merge the PR,
freeze its gate contracts as the version's baseline, and redeploy the loop
to the next newborn component — the loop's value density is highest against
fresh contracts, and a stale loop optimizes one implementation instead of
BCIR.

## The laws

### L1 — Every exit is a verdict
Every path out of a gate returns one of PASS / FAIL / INVALID-VACUOUS /
UNAVAILABLE-SKIPPED — error paths included. A traceback in place of a
structured report is itself a defect: it skips the JSON artifact and the
exit-code contract. Witnesses: `test_campaign_launch_failure_is_structured`,
`test_compiled_verifier_timeout_is_a_structured_failure`,
`test_q8_read_io_failure_is_not_graceful`,
`test_seed_construction_failure_is_a_structured_campaign_verdict`.
**Port note:** every C gate function returns a status enum on every path;
`abort()`/uncaught exceptions in gate code are defects by definition.

### L2 — A gate must be able to fire
Prove RED before landing GREEN. Refuse zero iteration budgets and empty
corpora; require executed negative cases; ask of every checker "what input
makes every loop iterate zero times?" and feed it that input in a test.
Witnesses: `test_decoder_that_accepts_everything_is_a_finding`,
`test_secret_scan_of_the_current_tree_is_non_vacuous`.
**Port note:** identical in any language; fault injection is part of the
gate's definition of done.

### L3 — Bounds live where the resource commits
A cap checked after materialization is not a cap. Decompressors allocate
their declared dictionaries before emitting a byte (found three separate
ways: xz streams, ZIP LZMA members, `ast.parse`); `stat` follows symlinks
while the read that follows does not care what `stat` said; a declared size
is not a read bound — read one byte past the cap and refuse the remainder.
Witnesses: `test_xz_dictionary_memory_is_bounded`,
`test_zip_symlink_under_lzma_is_uninspectable`,
`test_oversized_python_source_is_a_finding`,
`test_symlinked_pyproject_is_unasserted`,
`test_tar_probe_never_parses_compressed_bytes`,
`test_assignment_matcher_is_linear_time` (a quadratic matcher commits
CPU the same way a decompressor commits memory),
`test_staged_blobs_are_bounded_before_materializing` (a compressed
object commits memory when it is expanded, not when it is listed),
`test_concatenated_xz_streams_are_bounded` (a stream COUNT is a
resource: empty streams advance no output cap at all).
**Port note:** this is the memory-safety law. In Python these failures were
OOMs; in C the same shapes are allocator abuse and heap corruption. Every
`malloc` sized from input data is an L3 site.

### L4 — Attribute or refuse; never subset a grammar
A hand-rolled parser for a subset of a language has an unbounded surface of
valid spellings it misreads, and an adversary needs only one. Anything a
reader cannot fully attribute must fail closed — and the stable resolution
is to delete the subset reader entirely (this PR raised the Python floor to
3.11 and removed its TOML fallback for exactly this reason, after the
spelling family produced more findings than any other component).
Witnesses: `test_scalar_dependency_fields_are_unasserted`,
`test_wrong_shaped_metadata_tables_are_unasserted`,
`test_dependency_groups_fail_closed`.
**Port note:** BCIR wire formats get grammar-complete parsers generated
from the registry, or refusal. No "good enough" readers in C, ever.

### L5 — Everything committed is scannable data
File contents are a fraction of what a repository publishes: tree-entry
names, directory components, symlink target blobs, archive member names
and link targets, and every text encoding (BOM-marked UTF-16/32, BOM-less
UTF-16, escaped spellings, block scalars) all ship in every clone and all
carry secrets. A scanner's coverage claim is over committed *data*, not
over "files".
Witnesses: `test_credential_shaped_filenames_are_findings`,
`test_symlink_target_text_is_scanned`,
`test_archive_member_names_are_scanned_for_secrets`,
`test_bomless_utf16_text_is_scanned`,
`test_yaml_block_scalar_secrets_are_findings`,
`test_json_escaped_credential_keys_are_findings`,
`test_toml_multiline_string_secrets_are_findings`,
`test_single_line_toml_multiline_secrets_are_findings`,
`test_bomless_utf16_with_cjk_preamble_is_scanned`,
`test_utf16_probe_survives_a_split_surrogate_pair`,
`test_escaped_toml_delimiters_do_not_end_the_value`.
**Port note:** format-level knowledge; transfers verbatim to any scanner
in any language.

### L6 — Suppress by value shape, never by position
Placeholder allowlists inspect the matched value (filler runs, placeholder
words, template references, short whitespace-bearing prose) — never the
line, file, or tree the match sits in. Positional suppression lets a real
credential hide beside an innocuous neighbor.
Witnesses: `test_schema_prose_is_not_a_secret`,
`test_unquoted_passphrases_are_findings`.
**Port note:** identical everywhere.

### L7 — A report is an egress surface
The gate's own output can republish what it found: findings carry
fingerprints, never values; a path that matched is redacted in every report
field (findings, metadata lists, console), component by component; output
survives strict encodings. A crash while reporting is a double failure —
the finding is lost and the gate lied about its verdict.
Witnesses: `test_credential_shaped_names_are_redacted_in_every_report_field`,
`test_secret_bearing_directories_are_redacted`,
`test_credential_in_non_utf8_filename_is_redacted`,
`test_boundary_findings_survive_strict_stdout`,
`test_non_utf8_archive_member_names_are_findings_not_crashes`.
**Port note:** harsher in C — every format string and every buffer holding
a matched value is an L7 site.

### L8 — Child processes die as trees; pipes drain under budgets
Every spawned tool runs in its own session/group; timeouts and overflows
kill the whole tree (POSIX session kill, Windows tree terminator); both
pipes drain under per-stream byte budgets; a pipe still open after the
reap is a verdict ("descendants held the pipes"), not a hang; and the
put-down itself is total — it never raises inside the bound it protects.
Witnesses: `test_bounded_runner_expires_and_caps`,
`test_flooding_c_campaign_is_bounded`,
`test_reviewer_put_down_kills_the_tree_on_windows`,
`test_put_down_never_raises_from_the_tree_terminator`,
`test_compiled_verifier_descendants_are_a_structured_failure`.
**Port note:** `posix_spawn` + process groups / Job Objects; the shape is
identical, the primitives change.

### L9 — The instrument must be unswallowable and unfoolable by its subject
A watchdog the bounded code can catch is not a bound (the decode watchdog
derives from `BaseException` because a decoder's `except Exception` wrapped
it into that surface's own graceful rejection). An accounting the subject
can satisfy by accident is not an accounting (a blanket exception tuple
counted unchecked indexing as graceful rejection). An opted-in engine whose
failure is metadata is theater — its verdict gates.
Witnesses: `test_decoder_watchdog_cannot_be_swallowed`,
`test_implementation_errors_are_never_graceful`,
`test_gitleaks_nonzero_fails_the_scan`.
**Port note:** in C the watchdog is a separate process; in-process signals
are swallowable by longjmp-style recovery just as exceptions are.

### L10 — Every surface declares its rejection contract
Each decoder names the exact error type/code with which it rejects
malformed input, and only that counts as rejection; an undeclared surface
has an empty graceful set, so a new decoder must state its contract rather
than inherit one. A skipped rail needs an owner: the job that installed
the tool passes `--require-<rail>`, making absence a failure exactly where
absence is unexpected.
Witnesses: `test_decoder_seed_rejection_is_a_finding`,
`test_implementation_errors_are_never_graceful`,
`test_unseeded_c_fuzzing_is_recorded_as_unavailable`.
**Port note:** the C header's error enum IS the contract; the fuzz harness
whitelists those values and nothing else.

### L11 — A witness must hit the law it exists to test, on every rail
`rejected == true` is not parity: a witness that drifts into syntax rot or
a broader check keeps the differential green while its target law
regresses. Each rejecting witness declares its expected law per rail
(Python law, text reason, compiled diagnostic marker), and diagnostics are
captured head-biased so the marker survives note floods.
Witnesses: `test_python_witness_paired_to_its_intended_law`,
`test_witness_rejected_for_the_wrong_law_is_a_disagreement`,
`test_compiled_diagnostic_marker_survives_long_notes`.
**Port note:** this is BCIR's oracle/law/twin differential method itself;
the pairing discipline applies to every future rail unchanged.

### L12 — Platform divergence is a defect or a declared boundary
The same condition handled differently across platforms (case-literal
globs on Linux vs case-folding hosts, sessionless Windows, SIGALRM) is
either fixed to parity or declared a boundary in the tool's own docstring —
never left implicit. Declared boundaries are honored in review instead of
re-litigated.
Witnesses: `test_uppercase_python_suffix_is_audited`,
`test_reviewer_put_down_kills_the_tree_on_windows`.
**Port note:** substitute endianness, ABI, and libc variance for the same
discipline.

### L13 — A gate honors the configuration surface of the tool it wraps
If the wrapped tool reads `CLANG`, `BCIR_OPT`, or a preset's build
directory, the gate's preflight resolves the same configuration — a
preflight stricter than its tool reports available rails as unavailable
and fails `--require` runs that would have passed.
Witnesses: `test_configured_clang_is_honored`,
`test_debug_preset_build_is_discovered`.
**Port note:** identical everywhere.

### L14 — One predicate per repeated defect
The same defect fixed N times locally is how there come to be N defects;
the drain/put-down/redaction/fingerprint predicates are shared modules
(`tools/security/proc_bounds.py`) precisely because their gaps recurred
per-copy until they were unified. A scope rollback is an audit of every
fix layered on the stripped code, never a bare revert.
Witness: the shared modules and their tests, e.g.
`test_bounded_runner_expires_and_caps`.
**Port note:** identical everywhere.

### L15 — Discovery is reconciled; skips are scoped prefixes
What a gate scans is checked against what the repository tracks: a tracked
file the walk never yielded is a finding, discovery failure inside a
checkout is a FAIL (never a downgrade), and generated-tree skips are path
prefixes with named roots — a skip matched anywhere lets tracked code hide
in a directory that shares a name.
Witnesses: `test_nested_build_directories_are_still_audited`,
`test_tool_boundaries_scan_is_non_vacuous`,
`test_staged_secrets_are_scanned` (the INDEX is part of what the
repository tracks: the next commit records it, not the worktree),
`test_index_flagged_paths_are_staged_scanned` (an entry the VCS was
told to stop comparing is one the gate must compare itself),
`test_staged_python_blobs_are_audited` (every rail reconciles, through
one shared predicate — see L14).
**Port note:** identical everywhere.

### L16 — Never green yourself by editing the neighbor
When a gate false-positives on another component, the defect is in the
gate. Rewording the other tree to placate a rule creates the cross-tree
dependency the repository map forbids — the campaign's single P1 finding
was exactly this, and the resolution was to restore the neighbor verbatim
and narrow the rule where it lived.
Witness: `test_schema_prose_is_not_a_secret`.
**Port note:** a process law; it survives every migration.

### L17 — A name an extractor can turn into a path is a path
Archive member names and link targets are filesystem inputs: normalized
across separator conventions and drive-absolute spellings, checked for
traversal in every representation, read through their exact entry (a
later same-named member must not alias the read), bounded, and fail-closed
when unreadable or encrypted.
Witnesses: `test_zip_symlink_targets_are_checked_for_traversal`,
`test_legacy_v7_tar_members_are_inspected`,
`test_corrupt_zip_symlink_payload_is_unreadable_not_a_crash`.
**Port note:** the C extractor's path validation is a security boundary,
not a convenience; every representation an OS will accept must be checked.

### L18 — A heuristic declares its scope in the tool, exactly
A static detector can always be beaten by one more language feature, and a
reviewer will find them one per round — each finding valid, the sum an
interpreter nobody asked for. Declare the scope in the tool's own
docstring; be exact inside it (`args=`, f-strings, shell helpers were
in-scope completions); refuse to grow outside it, answering soundness
findings by pointing at the declaration. Fifteen alias-tracking fixes were
rolled back under this law.
Witnesses: the `audit_tool_boundaries` module docstring;
`test_fstring_subprocess_commands_are_flagged` where present in the suite.
**Port note:** identical for any static analysis shipped as a gate.

### L19 — Unit tests mock the expensive rail
The quick tier stays host-independent and bounded: unit tests of a gate
fake the C fuzzer, compiled verifier, pip-audit, and gitleaks; an optional
engine the host happens to have must not change a unit verdict; heavy work
runs only in the serialized job that owns it — and the fake must cover the
FULL spawn path, because a test that fakes `which` but not the spawn
really executes on one platform and tracebacks on another.
Witnesses: `test_dependency_inventory_must_be_asserted_before_advisories`,
`test_c_campaign_runs_in_its_own_session`,
`test_configured_clang_is_honored`.
**Port note:** identical; in C the "fake" is a stub binary on PATH.

### L20 — Reserved implementation values are not valid domain values
A container's sentinel keys are part of its contract: BCIR claim IDs -1
and -2 collided with `DenseMap`'s empty and tombstone keys, turning valid
domain input into assertion failures or misfiled entries. Domain
identifiers are validated against the implementation's reserved values at
ingestion.
Witness: the claim-id guard in `mlir/lib/passes/BCIRVerifyPass.cpp`.
**Port note:** a C++-specific discovery of a universal law — every hash
map, every tagged union, every "impossible" enum value is an L20 site.

### L21 — A package must contain what it registers
The shipped artifact is a claim about itself: every test the packaged
runner registers, every fixture a registered test reads, must either be
in the package or be a **declared, reported** absence. This repository
had already learned the rule for data — `pyproject.toml` ships the ASN.1
sources because "an sdist/wheel without them installs a package whose own
tests cannot run" — but not for code: four registered test modules import
the repo-only `tools/` tree, which the wheel does not ship, so discovery
raised `ModuleNotFoundError` and took the whole suite down before a single
test executed. A skip is honest only where absence is expected, so the
runner distinguishes the two environments: in a source checkout the tree
is present and any import error stays fatal; in an installed environment
the module is skipped **by name, on stdout**, and a run that collected
nothing is INVALID/VACUOUS rather than a pass.
Witness: `test_registered_suite_survives_missing_repo_only_trees`.
**Port note:** the same law, harder to see in C: an installed library
whose CTest manifest names build-tree fixtures, a pkg-config file
pointing at headers the install step never copied, a `make check` that
passes only in the source directory. The manifest is the promise; the
install tree is the audit.

## Campaign classification summary

Every review-thread finding from the campaign (195 threads, rounds 1–34)
is graded under the harvest protocol. The full per-finding
index is `docs/security/pr749-harvest.csv`; the campaign ledger tracks the
same data round by round.

| Grade | Findings | Share | Meaning |
|---|---|---|---|
| **NEW-LAW** | 20 | **10.3%** | Originated a registry law (L1–L13, L15–L21; L14 emerged from the repetition itself, not one finding) |
| **INSTANCE** | 136 | **69.7%** | New entry point to a registered law — the law gained a witness |
| **LOCAL** | 39 | **20.0%** | No transfer value beyond the code touched |

Rounds through 31 were graded retroactively; from round 32 every finding
is graded at triage. Rounds 32 and 33 were both zero-NEW-LAW, taking the
staleness counter to 2 of 3 — and round 34 **reset it to 0**, because one
of its four findings (the packaged runner) established L21, the first law
in the registry about the shipped artifact rather than a gate's internals.

That reset deserves its caveat, because the harvest protocol is exactly
gameable here: whether a finding is NEW-LAW depends on whether a registry
entry is written for it. L21 is claimed on three grounds — it names a
surface (distribution) no existing law covers, the repository had already
articulated the same rule independently for data files in
`pyproject.toml`, and it ports to C/C++ without translation. Read the
other way — as an instance of L12's "declare the boundary, never leave it
implicit" — the counter would stand at 3 of 3 and the loop would be
stale. The other three round-34 findings were unambiguous instances, and
the loop's overall trend remains a thinning one.

Where the instances concentrated (finding count per law, origin included):
L5 scannable-data coverage 22 · L3 resource-commit bounds 18 · L1
every-exit-a-verdict 12 · L11 witness/law pairing 12 · L8 process
trees/pipes 10 · L10 rejection contracts 9 · L18 heuristic scope 9 · L15
discovery reconciliation 7 · L17 names-are-paths 7 — the remaining laws
account for the rest. The two heaviest laws are exactly the two that port
hardest into C (allocation bounds and data-coverage claims), which is the
campaign's transfer thesis in one line.

Where the noise concentrated: the deleted Python 3.10 TOML fallback
absorbed 21 findings (11.5% of the entire campaign — the single largest
family, all LOCAL, resolved by raising the floor to 3.11 per L4), and the
rolled-back alias/dataflow tracking in the boundary audit absorbed 13 more
(7.1%, resolved by declaring scope per L18). Those two structural changes
retire **87% of all LOCAL findings** — the loop's zero-yield surface —
which is what steers future review rounds toward NEW-LAW territory.
