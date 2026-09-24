# The Security Gate Laws

A registry of the durable engineering laws extracted from the PR #749
adversarial hardening campaign (Claude ⇄ Codex, 30+ review rounds, ~100
closed findings). The Python gates that produced these laws are one
implementation; the laws are the contract. When BCIR components migrate to
C++ and C, **the laws port, the code does not** — each entry carries a port
note saying what the law becomes on the native side.

Witnesses name the tests that prove the law can fire -- for the campaign's
own laws, in `bcir/tests/test_security_assurance.py`; for a law learned on
another rail, in the gate that owns it. Every law was learned from at least
one live finding; none is speculative.

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
`test_seed_construction_failure_is_a_structured_campaign_verdict`,
`test_secret_scan_discovery_failure_is_a_verdict`,
`test_unreadable_expected_inventory_is_a_failing_report` (a gate's own
reference data is input too),
`test_expected_inventory_requires_its_fields` (an ABSENT field is not
an empty one),
`test_differential_setup_failure_is_a_verdict` (fixtures built by the
oracle under test fail like anything else),
`test_malformed_verifier_diagnostics_are_a_disagreement` (a verifier's
RETURN value is input too, not only what it raises),
`test_staged_archive_spool_failure_is_a_finding` (the temporary file a
gate writes is I/O like any other),
`test_compiled_fixture_io_failure_is_a_verdict` (a guard on BUILDING the
fixture is not a guard on WRITING it),
`test_inventory_depth_bomb_is_a_verdict` (every token valid and the
parser still bottoms out — a size cap is no defence against depth).
Post-campaign instance: `test_unwritable_requirements_are_a_verdict` (the
advisory rail's own temporary directory and requirements file are
resources of the run; a full or read-only TMPDIR is that run's fail-closed
verdict, not a traceback out of the required audit).
Post-campaign instance (the installed-environment audit, 2026-09-04):
`test_installed_mode_enumeration_failure_is_a_verdict` (broken distribution
metadata makes `importlib.metadata` raise; in a required job that is a
structured FAIL with the reason, never a traceback).
Review instance (#761, 2026-09-05):
`test_a_manifest_whose_root_is_not_an_object_is_a_finding_not_a_traceback`
(`[]` and `null` are valid JSON; the ODS→IRDL inventory gate dereferenced
the decoded root and raised `AttributeError` in place of its verdict, with
no `--json-out` report and no exit code).
S3-B instance (2026-09-23): the G15 grading function (`ring_fixtures.measure`)
raised `FileNotFoundError` when the harness it was handed did not exist, and let
any exception but the codec's own escape -- a traceback where nine rows belonged.
Every rail call is now a count (a rail that raises, or prints what cannot be
parsed, fails every fixture it was handed), and `tools/c/check_ring.py` exits 0,
1 or 2 (UNAVAILABLE, never a pass); the C gate refuses to score a mutant it could
not grade as a catch.
S3-C instance (2026-09-24): the G16 fault table's first sweep sent one defect
through the grader and got a traceback instead of a row. The shards had lost
their vector, so the manifest encoder refused its own output. The self-check did
its job, but it fired while `measure()` was still *building* the hostile corpus
it grades against, outside every fail-closed wrapper. The corpora are the
oracle's own splits and freezes, so a defect in the oracle can surface first as
a corpus that cannot be built. Every corpus is now built through one guard that
fails each row the corpus feeds (`handoff_fixtures._built`), and so is the Stage
3 exit flow. The witness patches `split` to raise and asserts named rows, not an
exception.
**Port note:** every C gate function returns a status enum on every path;
`abort()`/uncaught exceptions in gate code are defects by definition.

### L2 — A gate must be able to fire
Prove RED before landing GREEN. Refuse zero iteration budgets and empty
corpora; require executed negative cases; ask of every checker "what input
makes every loop iterate zero times?" and feed it that input in a test.
A `--require-X` flag is the same claim in a smaller frame: it promises
the rail RAN, so a preflight that only proves the rail was *discoverable*
leaves the run green over zero executions.
Witnesses: `test_decoder_that_accepts_everything_is_a_finding`,
`test_secret_scan_of_the_current_tree_is_non_vacuous`,
`test_ci_exercises_the_declared_python_floor` (a support claim no job
exercises is asserted by nothing),
`test_require_compiled_demands_the_rail_actually_ran` (discovery resolved
once per campaign, and every unavailable required witness is a
disagreement).
Post-campaign instances (the advisory rail, 2026-09-03):
`test_advisory_requirements_are_handed_to_the_engine_as_files` (the rail
handed pip-audit `--requirement -`, which the engine refuses; the rail had
never run, so every green it reported was over zero executions — found the
day it first ran live), `test_require_advisory_fails_when_the_engine_is_absent`
(the job that installs the engine requires it; elsewhere the skip is
recorded, never silent), `test_advisory_over_zero_dependencies_is_vacuous`
(an engine that exits 0 over an install set it collected nothing from has
audited nothing).
Installed-environment audit instances: `test_installed_mode_requires_an_engine`
(an environment audit with no engine is nothing, and has no flag to opt out),
`test_installed_mode_refuses_an_environment_missing_what_it_claims` (a job that
claims to audit the model-lab closure but runs in an interpreter without torch
has audited some other environment; the missing expectation is a FAIL that
names it, and the engine still ran over what was there). Live instance: the
gate fired on its first CI run, on both runners, over `setuptools` 78.1.0 that
torch's CPU wheel had pulled from the PyTorch index (2026-09-04 audit F12) —
a gate that can fire, and did, before anything else was proven about it.
S0-A instances (the verifier, 2026-09-04): EV1–EV3 lived in
`kbcir.events.check_event_phases` with their own tests and were wired into no
entry point, so the canonical `verify(module)` could not fire on an unarmed
interrupt source — a law nobody calls holds vacuously
(`test_ev_laws_are_part_of_the_canonical_verifier`). R9's cost re-derivation
was vacuous for every caller that omitted the target, and the performance
audit's own K_BCIR→StreamPack case verified its plan with no scope at all, so
a forged step cost passed 4,096 times out of 4,096
(`test_r9_refuses_a_forged_step_cost_only_the_scope_can_see`; the harness row
`verify.plan.r9.vacuous` freezes that 1.0 and now reads 0.0). The C rail's R9
accepted any `cost` and any `width`
(`test_c_planner_width_contract_and_r9_rederives_costs`).
S0-B instances (the MLIR rail, 2026-09-04): two compiled fixtures
(`verify_timing_lifetime.mlir`, `cost_model_barrier.mlir`) carried RUN lines
and expected-error markers that no runner executed, so their five R19–R21
negatives asserted nothing (`test_mlir_fixture_inventory.py` named them
before the runner did); `bcir-optimize` and `bcir-hydrate` advertised
verifier checkpoints and ran no verifier, so an illegal module planned and
hydrated (`pipeline_checkpoints.mlir` refuses it under both); and R2 held
vacuously across modules -- a root-global registry map let one module's
claim resolve another's resource (`verify_module_scope.mlir`).
S0-C instances (both rails, 2026-09-05): the parent build admitted every defect
shape the structural corpus now carries -- a duplicate phase id, a dependency on
an undeclared phase, an i32 device-register address under a 64-bit target, an
unsorted or arity-mismatched manifest artifact record, a calibration certificate
whose constants the capability never used, an M5 field of width 0 -- and
`-bcir-schedule` put a phase before the phase it depends on; on the oracle a
zero stride folded to 1, a negative count verified clean, and an HBM-only MAP
program was refused. The corpus's own comparison is witnessed:
`test_findings_name_every_kind_of_disagreement` injects each way a rail can
disagree with the corpus and asserts each is a finding.
Review instances (#761/#762, 2026-09-05): the ODS→IRDL inventory gate counted
a `//`-commented `irdl.operation` as projected and stayed PASS over a
projection that defined one operation less
(`test_a_commented_out_declaration_is_not_a_declaration`); the
fixture-execution gate counted a `#`-commented runner line as an execution --
green on the exact scenario it exists to catch
(`test_a_commented_out_reference_is_not_an_execution`); and the corpus's
`also_laws` was an allowlist, so a rail that dropped a required diagnostic
passed the subset check
(`test_a_declared_additional_law_is_required_not_merely_allowed`). A text
gate reads its sources the way their compilers do (`active_text`,
`active_shell_text`), and a documented requirement is asserted, never merely
permitted.
S0-E instance (2026-09-05): the maxima-only R11 was a rule whose loop iterated
zero times over its own defect class -- a resource that moved while another
held the maximum, or one declared after hydration, kept the header tags equal to
the registry maxima, so the parent build ACCEPTED both on all three rails (the
oracle over `vector_add`, the C `bcir_sp_check_generation` over the mutator's
`stale_vector`, `-bcir-verify` over a two-resource registry). StreamPack v4's
per-resource vector makes each a refusal, and the adversarial gate keeps the RED
as a witness: `stale_vector`, `missing_vector_entry` and `undeclared_vector_rid`
must pass the maxima-only API and fail the vector
(`test_c_rejects_stale_generation_vectors_per_resource`,
`test_generation_drift_under_the_maxima_is_R11`, `verify_generation_vector.mlir`).
S0-F instance (2026-09-05): the native rig's strided pass iterated n times over
n/16 elements -- `(k * 16) % n` on a power-of-two n -- and reported a
"cache-defeating" regime whose working set was one sixteenth of the buffer;
nothing counted what the walk touched, so the claim could not fail. The walk is
now a proved full-cycle permutation AND a non-timed census counts the unique
elements per regime into the evidence the table carries
(`test_strided_order_is_a_full_cycle_permutation` pins the parent's n/gcd,
`test_native_rig_reports_census_samples_and_an_attested_tenancy` reads the
count back). A measurement's coverage claim needs a witness like any other gate.
S1-A instances (2026-09-05): the divergence row's fixture was built to exhibit
the defect it tracks, and the slice that fixed it keeps the fixture able to fire
-- `test_the_divergence_fixture_still_reproduces_the_reports_own_number` drives
the retired pricer (`price_waves_legacy`) over the same four claims and requires
the report's 1.9922, while `test_the_divergence_row_reads_one_artifact` requires
exactly 1.0 from the canonical price; a fixture that no longer exhibits the
defect cannot grade the fix. The RED was recorded on both rails before any code
moved: a gather placed at 0 before its producer's 5248 by the oracle's three
schedulers and the law rail's three passes, a fence overlapped by a
data-independent claim, and the CSE copy credit granted to a duplicate over a
different count, offset or immediate and to atomic, barriered and volatile
duplicates (`schedule_hazards.mlir` and `cost_model_cse_neg.mlir` pin the
refusals; `test_schedule.py` / `test_fusion.py` are the oracle twins). The
harness itself refused the first cut of the canonical pricer: a full
re-placement per trial made the re-selection sweep 1,100x the serial pass and
the `ratio` row reported REGRESSION -- a gate that fired on the slice that
introduced the regression, before the slice claimed credit.
S1-B instances (2026-09-12): the digest cache the roadmap warned would rebuild
the Class-B vacuous check if it survived a mutation is held to the negative
first -- `test_an_undeclared_in_place_edit_cannot_pass_a_verifier` edits a claim
field with no declaration and requires the verifier to refuse the cached
identity and report the digest mismatch, `test_cross_module_substitution_is_refused`
presents module A's identity for module B, and the exact harness row
`static_memory.digests.2048` counted three full digests per plan-and-verify chain
on the parent (RED) before reading one. A cache that cannot be shown to be
refused is a cache that has passed nothing.
S1-C instances (2026-09-12): the plan-as-bytes gates are counted failures over
fixed corpora and were measured on the parent first -- 26 corpus plans that
could not round-trip, 27 readers that could not read, 6 stale and 39 malformed
(variant, rail) pairs that nothing could refuse -- so the rows had somewhere
to fall from; every malformed variant is minted through a raw writer that
bypasses the codec's laws (`plan_fixtures.raw_encode`), because an encoder
that refuses to emit a malformed plan cannot also be the witness that the
decoder refuses one.
S1-D instances (2026-09-12): the alias fixture was composed with the token
placement and shown to alias before the planner learned schedule liveness, and
the static-memory verifier is held to forgeries the exact rail invites -- a
forged stop reason, a forged lower bound, an offset moved to an alias-free but
non-solver position, a first-fit plan claiming the exact rail's stop reason --
each refused by recomputation, never by reading the plan's own claim about
itself (`test_the_verifier_recomputes_an_exact_layout_rather_than_trust_it`).
S2-A instance (2026-09-13): the general case was made reachable and measured
RED before it was priced incrementally (`sweep_fixtures.general_fixture`: one
step-shortening trial per pair, every trial a full re-placement), and the delta
search is held to the full re-placement it replaces, kept alive as
`optimize_scheduled(delta=False)`: the same assignment claim by claim, the same
step costs, price and artifact on 1,344 cases, and the placer to `schedule_eft`
on 6,840 random trials and 1,690 adoptions -- a faster sweep that picked a
different plan would not have been made faster; it would have been changed.
S2-B instance (2026-09-13): the exact rail is held to oracles that share no
code with it -- the partition optimum on the section 6.1 corpus
(`exact_fixtures.partition_optimum`, a symmetry-broken search over loads) and
the enumeration of every active schedule of a tiny module
(`active_schedule_optimum`, no bound, no symmetry breaking) -- and a
certificate is refused its optimality rung when the scope is undeclared
(`certify_schedule` without a policy: TMSAO-4 with the reason), so the solver
cannot certify itself and a certificate cannot over-claim.
S2-C instance (2026-09-13): a resumed search is held to the uninterrupted run
it claims to equal -- run(b1) then resume(b2) against run(b1 + b2), plans,
bounds, stop reasons, expansions and state digests alike -- and a ranker is
held to the census it claims to order (`dispatch.ranked` refuses a removed,
added or duplicated member), so neither "anytime" nor "learned" can quietly
change what a certificate ranges over.
S2-D instance (2026-09-13): a region is recognized, then RE-VERIFIED against
the module before it is returned, and the verifier is held to forgeries --
non-consecutive claims, a foreign claim, wrong access maps, a local model on
an opaque region, an empty region, a graph of another module -- each refused;
an objective is admitted to the registry only after `verify_objective` proves
its laws on samples, and a lawless operator (a non-associative combine, an
order `select` does not realize) is refused by name, so "semiring" cannot
become a label that admits arbitrary operators.
S2-E instance (2026-09-13): the measured replay gate replays EVERY logged
episode of a scope or refuses -- a candidate without evidence on one episode
is not judged on the rest -- and a `ReplayCertificate` that names a corpus is
admitting only when it covers the log, so the portfolio refuses a
one-episode certificate over a three-episode corpus
(`replay.subset.admitted` 24 -> 0); the corpus reader re-derives the digest
chain and refuses an altered, removed or reordered entry and a forged head;
the class ladder refuses TMSAO-3 without W and M declared and without a
prediction interval from attested silicon on two targets, naming the tenancy
the samples came from.
S2-D/S2-E follow-up instance (2026-09-13): the law has a DUAL a gate can fail
just as badly -- a gate that fires on the machine instead of on the code. The
four `native.*` rows were graded against the baseline host's band although they
divide one compiled kernel by another, so they measure this host's gather
penalty; on this machine that produced a REGRESSION verdict on about one run in
thirty, which is noise wearing a verdict's clothes. `Metric.host_dependent`
reports them INDICATIVE off the baseline host. Ask of every new row both
questions: what input makes it fire when it should not, and what input makes it
stay green when it should not.
S3-A instance (2026-09-23): the obvious quiescent-switch row -- "a switch
requested mid-phase was applied" -- read 0 on the parent tree, because the
loader and context-shard activation REFUSED a mid-phase switch; a refusal is
not a deferral, so the G14 gate counts deferrals LOST (16 on the parent, 0 at
the bound). The C section of `tools/c/check_runtime.sh` proves it can fire: it
rebuilds the plane with the deferral law removed and requires a row to go red,
and refuses to grade the mutant if the law's line changed and the edit did not
apply. Eight injected C faults were each caught by at least one row, and the
state-digest comparison caught the one no verdict saw (a digest that dropped
the drain flag: 52 divergent traces, every decision still conforming). The C
tests fail closed in a checkout: a missing harness source is a failure, never
a skip.
S3-B instance (2026-09-23): the G15 rows' evidence that they can fire is
committed, not narrated: `tools/testing/faults/ring.json` injects 22 defects,
each one law on one rail -- the seqlock re-check, the lap count, the full ring,
the epoch law, a deposed producer, the takeover repair, the progress
publication, the geometry CRC and the control-slot law; the envelope's CRC and
session law, the gap count, the stale generation and the unknown REQUIRED
signal; the table's bytes; the control transport -- and `red_sweep.py` saw all
22 fire their own row (a green control first; every injection and restore
proved by digest). ThreadSanitizer is held to the same standard: the C gate
makes the relaxed atomic stores plain and requires a race REPORT, not merely a
failure. Its absence has an owner: the x86 C runtime job installs the TSan
runtime and sets `BCIR_REQUIRE_TSAN=1`, so there an unavailable TSan fails,
while a runner without it (the aarch64 job) prints an explicit skip -- all three
branches (available, absent, absent-but-required) driven before landing.
S3-C instance (2026-09-24): `tools/testing/faults/handoff.json` injects 34
defects across the oracle, the C twin and the C++ seam. Each one is a single
law: the admission, the registry binding, the epoch, a moved owner, dispatch in
place, the freeze's header and claim laws, the manifest's wire laws, the
partition, the sub-pack and frame spellings, the reassembly predicate, and both
state digests. Through the Stage 3 exit flow it also covers the *other*
boundaries one generation crosses: the plane's compare-and-swap, the intake's
stale law, and the telemetry ring's loss count. The first sweep caught 29, and
both misses were defects in the gate, not the table. One was a public frame
spelling that no row measured (L14); the other was a corpus-build traceback
(L1). After the fixes, all 34 fire their own row.
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
resource: empty streams advance no output cap at all),
`test_worktree_source_read_is_bounded` (stat answers about the past;
the read is the commitment),
`test_secret_scan_worktree_read_is_bounded` (the same bound, the third
rail to receive it),
`test_expected_inventory_is_bounded_at_ingress` (a gate's own reference
data allocates like any other input),
`test_concatenated_compressed_streams_are_counted` (the same stream-COUNT
bound the xz rail got, on gzip and bzip2 — a budget that measures the
wrong resource is not a budget).
Review instances (#762, 2026-09-05): `-bcir-verify` derived `4*strided_q8`
unchecked -- a ratio above INT64_MAX/4 passes R8's floor, and the signed
overflow was undefined behaviour where the law owes a refusal (a wrap prices
the derivation at 1); the product goes through `checkedMulNonnegative` like
every other product in the pass (`calibration.strided_q8_overflow`, both
rails). The oracle's `BinaryField` admitted an end offset the law rail's
signed 64-bit field arithmetic cannot represent (`m5.field.end_overflow`).
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
An AMBIGUOUS document is the same defect from the other side: `json.loads`
keeps the last value for a repeated key, so a document saying two things
parses as one of them and the contradiction is never attributed.
Witnesses: `test_scalar_dependency_fields_are_unasserted`,
`test_wrong_shaped_metadata_tables_are_unasserted`,
`test_dependency_groups_fail_closed`,
`test_expected_inventory_rejects_duplicate_keys` (the review parser has
refused this since R23; the two now share one predicate, see L14).
Post-campaign instances: `test_unusable_advisory_output_is_a_verdict` (the
engine's JSON report is input to the gate, parsed strictly — a duplicate
key, a missing or mis-typed field, a depth that bottoms out the parser
under the byte cap — and anything else is the run's fail-closed verdict
with the tail retained, never a traceback);
`test_floor_pins_refuse_what_they_cannot_attribute` (the floor grammar is
two shapes, declared in the tool; a URL, marker, wildcard, compound or
arbitrary-equality declaration is refused and reported, never approximated
into a pin the declaration did not make).
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
`test_escaped_toml_delimiters_do_not_end_the_value`,
`test_yaml_escaped_credential_keys_are_findings`,
`test_wrapped_mapping_values_are_findings` (a value may reach its key
across a line break in JSON and YAML alike),
`test_dotted_toml_keys_reach_the_continuation_collectors`,
`test_escaped_keys_reach_the_continuation_collectors` (two closed
defects compose into a third that neither fix covered),
`test_bomless_utf32_text_is_scanned` (three NULs per ASCII character
read as binary to every density heuristic),
`test_quoted_dotted_key_segments_are_matched`,
`test_escaped_quotes_in_quoted_key_segments`,
`test_escaped_quotes_inside_inline_values` (an escaped delimiter ends a
value only if the grammar forgets the escape),
`test_multilingual_bomless_utf32_is_scanned` (NUL density is an
ASCII-shaped assumption; decode validity is not),
`test_yaml_node_properties_precede_the_credential` (an anchor or tag NAMES
the node; the value behind it is the same credential),
`test_yaml_explicit_mapping_keys_are_scanned` (`? key` / `: value` is the
same mapping across two lines),
`test_folded_yaml_quoted_scalars_are_scanned` (a quoted scalar folds; the
key half has no closing quote and the value half has no key).
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
`test_non_utf8_archive_member_names_are_findings_not_crashes`,
`test_dependency_declarations_are_redacted_in_reports` (a PEP 508 direct
reference can CARRY a credential, and no scanner rule reads URL
userinfo), `test_advisory_output_is_redacted` (a wrapped tool's stdout and
error text are report fields too),
`test_url_username_only_credentials_are_redacted` (userinfo is credential
material by POSITION — redacting only the password half left a token used
as the username intact),
`test_reviewer_findings_are_redacted` (a reviewer QUOTES the code it
reviews, so its findings are the field guaranteed to carry the secret;
redacted through the scan's own predicate, so a report cannot remove less
than the scan would report).
Post-campaign instance: `test_advisory_output_is_redacted` now drives a stub
engine that names the requirement on stderr and in its JSON report, as
pip-audit does, and every field of the structured advisory (`stderr_tail`,
`vulnerable`, `skipped`, `stdout_tail`) passes through the one redaction
predicate.
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
`test_verifier_watchdog_cannot_be_swallowed` (the differential's watchdog
derived from `Exception` for eleven rounds after the decoder's did not —
one law, two rails, one spelling, see L14),
`test_implementation_errors_are_never_graceful`,
`test_gitleaks_nonzero_fails_the_scan`.
**Port note:** in C the watchdog is a separate process; in-process signals
are swallowable by longjmp-style recovery just as exceptions are.

### L10 — Every surface declares its rejection contract
Each decoder names the exact error type/code with which it rejects
malformed input, and only that counts as rejection; an undeclared surface
has an empty graceful set, so a new decoder must state its contract rather
than inherit one. S1-C instance (2026-09-12): the ExecutionPlanV1 decoder
declares its contract on both rails before its first reader exists --
`AbiError` on the oracle, and on the C rail the named codes `BCIR_ERR_PLAN`
(a plan law), `BCIR_ERR_LANE` / `BCIR_ERR_WIDTH` (the range gate),
`BCIR_ERR_PROVENANCE` (a duplicated claim, a pack that is not the plan's
lowering), `BCIR_ERR_GENERATION`, `BCIR_ERR_STALE`, `BCIR_ERR_OVERFLOW`,
`BCIR_ERR_TRAILING`, `BCIR_ERR_RESERVED`, `BCIR_ERR_VERSION` -- and the stale
witnesses assert the code (`vector=BCIR_ERR_STALE`, `pack=BCIR_ERR_STALE`),
never merely a nonzero exit. A skipped rail needs an owner: the job that installed
the tool passes `--require-<rail>`, making absence a failure exactly where
absence is unexpected.
Witnesses: `test_decoder_seed_rejection_is_a_finding`,
`test_implementation_errors_are_never_graceful`,
`test_unseeded_c_fuzzing_is_recorded_as_unavailable`.
Post-campaign instance: `test_ci_owns_the_advisory_rail` (exactly one job
installs pip-audit, pinned, and that job passes `--require-advisory`; the
audit asserts the inventory only everywhere else).
Installed-environment audit instance: `test_ci_owns_the_installed_audit` (the
hosted model jobs, whose torch wheels come from an index that is not PyPI,
install the engine pinned into a scratch venv of their own, hand it to the
rail as `PIP_AUDIT`, and pass `--installed` with the three names the audit
must find; `test_ci_owns_the_advisory_rail` now pins that every job installing
the engine is a job that requires it, and only those).
S0-G instance (2026-09-05): the LLVM kernel's self-check harness called the
kernel only at the claim's own count -- a multiple of the vector width by
construction -- so the vector loop that stepped to `n` itself never met the
runtime `n` it was wrong for, and every lowering test was green over a
miscompile. The harness now drives every kernel with `count + 7`, a
sub-width count and zero behind canaries, and the parent's kernel fails it
(`FAIL n=1031: wrote past n at 1031`); R12 holds the contract in the text
(`test_missing_tail_contract_is_R12`, `test_harness_drives_the_tail_contract`).
A witness exercises the input the law is about, not the one that flatters it.
**Port note:** the C header's error enum IS the contract; the fuzz harness
whitelists those values and nothing else.

### L11 — A witness must hit the law it exists to test, on every rail
`rejected == true` is not parity: a witness that drifts into syntax rot or
a broader check keeps the differential green while its target law
regresses. Each rejecting witness declares its expected law per rail
(Python law, text reason, compiled diagnostic marker), and diagnostics are
captured head-biased so the marker survives note floods. The assertion
itself is under the same discipline: a witness that checks for a *substring*
of what should have been removed still passes on output where the law is
violated — it must pin the whole shape.
Witnesses: `test_python_witness_paired_to_its_intended_law`,
`test_witness_rejected_for_the_wrong_law_is_a_disagreement`,
`test_compiled_diagnostic_marker_survives_long_notes`,
`test_dependency_declarations_are_redacted_in_reports` (asserts the
redacted requirement exactly, not that a host substring is present).
S0-C instance (2026-09-05): every case of the structural corpus declares the
law family and the diagnostic each rail must produce; a refusal under another
law or for another reason is a finding (`structural_corpus.findings`), and the
law-rail projection pins one `expected-error` per expected diagnostic, so a
case that trips two laws (the MAP device-register write: R3 and R5) declares
both rather than passing on either.
Review instances (#762, 2026-09-05): three M5 descriptor rules had a twin on
one rail only -- the tokenless grammar and the duplicate record NAME (refused
by the oracle's constructors, admitted by the op verifiers, which see symbols
and not names) and the field end overflow (refused by `BinaryFieldOp`,
admitted by the oracle) -- and none had a corpus case, so the differential
could not see them. Each has both twins and its case now
(`m5.grammar.no_tokens`, `m5.format.duplicate_record_name`,
`m5.field.end_overflow`): a construct absent from the corpus is untested,
however many tests run over it.
**Port note:** this is BCIR's oracle/law/twin differential method itself;
the pairing discipline applies to every future rail unchanged.

### L12 — Platform divergence is a defect or a declared boundary
The same condition handled differently across platforms (case-literal
globs on Linux vs case-folding hosts, sessionless Windows, SIGALRM) is
either fixed to parity or declared a boundary in the tool's own docstring —
never left implicit. Declared boundaries are honored in review instead of
re-litigated.
Witnesses: `test_uppercase_python_suffix_is_audited`,
`test_reviewer_put_down_kills_the_tree_on_windows`,
`test_staged_symlink_inputs_are_refused_not_dereferenced` (the dependency
rail's BOTH audited inputs, two rounds after the scan and boundary rails),
`test_staged_inventory_decodes_strictly` (a lenient decode on one path and
a strict one on its sibling is the gate disagreeing with itself),
`test_staged_symlinks_are_recorded_not_parsed` (the index path and the
worktree path of one gate are two paths, and must agree),
`test_staged_symlinks_are_not_classified_by_suffix` (the same divergence
on the second rail, one round later).
**Port note:** substitute endianness, ABI, and libc variance for the same
discipline.

### L13 — A gate honors the configuration surface of the tool it wraps
If the wrapped tool reads `CLANG`, `BCIR_OPT`, or a preset's build
directory, the gate's preflight resolves the same configuration — a
preflight stricter than its tool reports available rails as unavailable
and fails `--require` runs that would have passed.
Witnesses: `test_configured_clang_is_honored`,
`test_debug_preset_build_is_discovered`,
`test_configured_bcir_opt_command_name_is_resolved` (a configured value
may be a PATH or a command NAME; the wrapper accepts both).
Post-campaign instance: `test_configured_advisory_engine_is_honored`
(`PIP_AUDIT` names the engine as `CLANG` and `BCIR_OPT` name theirs, a path
or a command name through the same resolution the default takes; a
configured engine that does not resolve is reported, never replaced by
PATH's).
**Port note:** identical everywhere.

### L14 — One predicate per repeated defect
The same defect fixed N times locally is how there come to be N defects;
the drain/put-down/redaction/fingerprint predicates are shared modules
(`tools/security/proc_bounds.py`) precisely because their gaps recurred
per-copy until they were unified. A scope rollback is an audit of every
fix layered on the stripped code, never a bare revert.
A shared predicate must also be TOTAL: one carrying an unstated
precondition (`redacted_path` returned `<redacted-path>` for any path with
no matching component, correct only because its single caller tested the
path first) is a defect held in reserve for the second caller.
Witnesses: the shared modules and their tests, e.g.
`test_bounded_runner_expires_and_caps`,
`test_boundary_audit_paths_are_redacted` (the scan rail had redacted
secret-bearing path components for sixteen rounds; the boundary rail
printed them to the CI log until it imported the same predicate).
S0-A instances (2026-09-04): the planner's DAG edge weight and R9's re-derived
step cost are one function (`realize.edge_cost`, wrapped by `step_cost`). The
first scoped R9 re-derived from `candidates_for` while the planner priced from
`fused_candidates`, and rejected 3,840 of the planner's own 4,096 steps until
the two shared the offer as well as the price. On the C rail
`bcir_plan_base_cost` moved into `bcir_plan.h` as a header inline so
`bcir_plan.c` and `bcir_verify.c` compute one base cost without every build
that links the verifier needing a new object.
S0-B instance (2026-09-04): the scope of an MLIR pass is ONE predicate
(`BCIRPassSupport.h` `forEachScope` / `walkScope`), shared by `-bcir-verify`,
`-bcir-select-realization`, `-bcir-rcsp` and the GEM batch/schedule/lower passes -- the
finding was the same root-global walk landed in each of them separately.
S0-C instances (2026-09-05): the isolated-domain rule, the triple -> pointer-width
table and the canonical phase order are each ONE predicate with a declared mirror
(`model.ISOLATED_DOMAINS` / `isIsolatedDomain`; `kbcir.cost.pointer_width` /
`pointerWidthOfTriple`; `model.topological_phase_ids` / `canonicalPhaseOrder`),
and the corpus checks that the mirrors agree -- five phase orders and two
isolated-domain rules were in use before.
S0-D instance (2026-09-05): the two cross-rail content hashes are ONE definition each
with a declared mirror (`provenance.hash_target` / `hashTargetFromIR`, `hash_module` /
`hashModuleFromIR`) and were widened on both rails in one commit -- the memory
hierarchy via two dialect attributes with a pinned default, the declared claim order by
dropping a sort on both sides -- with `test_hash_parity.py` driving the law rail over
emitted modules and their real manifests. A hash widened on one rail is a content
address the rails disagree about, which is worse than the gap it closes.
Review instances (#761/#762, 2026-09-05): S0-A's R9 re-derivation had landed on
the oracle and the C rail and not on the law rail, so `bcir-optimize`'s trailing
checkpoint accepted any `kbcir.plan_width` / `plan_cost` / `plan_score` the plan
pass -- or a forger -- wrote: the dominant defect shape once more, a mechanism
on two rails out of three. `-bcir-verify` re-derives the emitted plan through the
planner's own `cm::` functions (`verify_plan_annotations.mlir`: the emitted plan
accepted, four corruptions refused). The ROP frontend's pre-scan registered every
rid and not its domain, so a forward-declared MMIO register derived RAM -- a
shared predicate with an unstated precondition (declaration before use) that its
second caller violated (`test_rop_forward_declared_resource_keeps_its_declared_domain`).
And the pointer-width mirror is compared entry for entry out of both sources
(`test_the_pointer_width_tables_are_one_table_on_both_rails`), not on the
handful of triples the corpus happens to use: `arm64_32` was 64 on both.
S0-E instance (2026-09-05): R11 per resource is ONE predicate declared once
(`BCIR_STREAMPACK_ABI.md` §v4: vector present -> every entry matches the live
registry and every declared resource has an entry; vector absent -> stale against
any registry that declares resources) and mirrored on all three rails in one PR
(`verify_pack::_verify_generation_vector`, `bcir_sp_check_generation_vector`,
`BCIRVerifyPass.cpp`'s R11 walk over the `generations` triples), with the vector's
well-formedness (ascending RIDs, header maxima) shared by the encoder and the
decoder (`_validate_generation_vector`, `bcir_sp_verify_semantic`, and
`GEMStreamPackOp::verify`) and the ASN.1 projection carrying the same record. The
legacy maxima API was kept, not re-implemented, so the two R11 forms cannot drift.
S0-F instance (2026-09-05): the native rig's tenancy verdict is ONE predicate with a
declared Python twin (`attest()` in `bcir_microbench.c` / `microbench.host_attestation`),
and the test holds the rig to the twin field by field. The first test read fewer
signals than the rig (no DMI, no cgroup markers) and the GitHub aarch64 runner -- a
DMI-attested VM that also exposes a PMU -- called the rig wrong for saying
"virtualized": a mirror that checks a subset is a second rule, and it disagrees on
exactly the host that matters.
S1-A instances (2026-09-05): the scheduling hazard is ONE predicate
(`concurrency.hazard_conflict` / `hazard_predecessors`, mirrored by
`BCIRSchedule.h::hazardConflict` / `hazardPredecessors`) -- the bundle
reorderer's fence-aware `_conflict` now delegates to it, and the two schedulers
and both pricers build their DAGs from it, where before the tail split preceded
a data-only DAG over the main claims and the fence lived in the bundle rail
alone. The placement is ONE artifact (`gem.schedule.schedule_plan` /
`BCIRSchedule.h`): `price_scheduled`, `schedule_eft`, `execute_tokens`,
`optimize_scheduled` and the four law-rail passes read the same placement, so
the objective cannot denote a schedule the executor never runs -- the finding
was the same defect on both rails, a pricer and a scheduler that agreed on
nothing but the serial bound. The CSE identity and eligibility are ONE pair of
predicates with declared mirrors (`realize.cse_identity` / `cse_eligible` and
`cm::cseIdentity` / `cseEligible`), and the categorical exclusion precedes the
identity comparison on both rails; the oracle additionally refuses the fields
the IR does not carry (`imm`, `tolerance_ulp`, `quantized_bits`) rather than
credit a duplicate the law rail could not refuse.
S1-B instance (2026-09-12): the R13 canonical item sequence of a module is ONE
walk (`provenance.canonical_stream`) that `hash_module` chains and a
`ModuleIdentity` is validated against, where before the planner, the verifier and
the client each re-flattened the module to hash it -- the same sequence produced
three times, and the digest recomputed three times because no caller could prove
the module had not moved. One producer of the sequence, one memoized renderer of
its items (`_fnv_items`, exact ints and strs only, so a bool never takes an int's
memo and `hash_target` keeps rendering `scalable` as the law rail does), and one
verifier-side predicate (`digest_of`) that decides whether an identity may be
reused: by content, never by the revision counter that keys the cache.
S1-C instance (2026-09-12): the plan's wire laws are ONE predicate per rail --
`validate_plan`, applied by the encoder before publication and by the decoder
on read, and `ep_walk` in the C twin, the single bounded walk every entry point
(`bcir_ep_verify`, the four record walks, the R11 check, the plan/pack binding)
runs -- so the 39 malformed pairs are refused by the same code on every path
in, and the harness rows and the tests measure one definition of each gate
(`bcir/tests/plan_fixtures.py`) rather than two that could drift.
S1-D instance (2026-09-12): the alias law is ONE predicate in three places that
must agree -- `StaticAllocation.overlaps` / `_has_live_alias` on the planner's
rows, `validate_plan` on the plan's bytes and `ep_lifetime_aliases_before` in the
C twin -- all judged on the same half-open ticks; and the liveness interval is
computed by ONE function (`schedule_intervals`) the planner, the verifier and the
plan-bytes verifier share, so "does the schedule refine the phase order" and
"do the lifetimes cover the schedule" are the same question asked of the same
intervals.
S2-A instance (2026-09-13): the dispatch is ONE loop -- `_PhaseDispatch.run`,
of which `_dispatch` is the one-shot form -- run by `schedule_eft`,
`execute_tokens` and the placer's checkpointed replay alike, so the incremental
price cannot drift from the placement it prices; `EftPlacer.schedule()` is
`schedule_eft`'s artifact by construction and by test on every fixture, trial
and adoption.
S2-B instance (2026-09-13): the exact scheduler's eligibility -- which stream
may run a sparse, a bandwidth or a compute claim -- is `_PhaseDispatch.eligible`,
the artifact's own rule, read rather than restated, so a certificate proves a
bound over exactly the placements the executor could make.
S2-C instance (2026-09-13): the choice of solver is ONE function
(`gem.dispatch.dispatch`, a table over region kind, size, requested class and
budget) that every runner and every certificate go through, so "which rail
ran" is read from the record, never inferred from the result.
S2-D instance (2026-09-13): a region's expansion is ONE thing -- exactly the
claims it was recognized from, in declared order (the identity on the
carrier) -- so a stronger local model can never change what the planner
selects over; and the region floor prices candidates through the planner's own
edge predicate (`realize.edge_cost`'s coupling, `fused_candidates`' costs),
not a second cost model, so a floor and a score cannot disagree about what a
step costs.
S2-E instance (2026-09-13): the plan a measured entry names is ONE digest
(`measured.assignment_digest` over the planner's own steps), computed by the
producer when it measures and by the consumer when it re-derives, so stale
evidence is recognized by the predicate that filed it; the host attestation
is the S0-F rig's `host_attestation`, not a second reading of the same
files; and the workload's class is one function (`Workload.classify`) the
portfolio's table reads rather than a second rule over the same fields.
S2-D/S2-E follow-up instance (2026-09-13): a package name is owned by ONE
module. `kbcir.regions` and `kbcir.compose` both listed `Region`, and because
`_NAME_TO_MOD` is a flattening comprehension the later entry silently won:
`bcir.kbcir.Region` changed meaning with no import error, and the loser stayed
in `__all__`. Reachability tests do not catch this -- the name still resolves,
to the wrong object. The predicate is now one test over all three lazy packages
(`test_perf.test_the_export_table_has_no_duplicate_names`), so the next
collision is a failure rather than a rebinding.
S3-A instance (2026-09-23): staleness is ONE predicate (`gem.control.is_stale`,
mirrored by the C plane): the trusted loader and context-shard activation stopped
spelling `generation <= live` twice and express the witness it names. The wire
laws are one ordered predicate per rail -- the encoder and the decoder share
`validate_control`, and the ASN.1 projection re-runs it on decode instead of
restating the laws -- and the two rails name the same first violation. runtime/c
has one SHA-256 (S3-A0), linked by the BCAB reader and the control plane alike;
the registry digest hashes the generation vector's own wire bytes, so the
StreamPack and plan readers feed one walk; and the closed sets (kinds, scopes,
refusals, capabilities, the record bound) are read out of the C header by the
tests, not mirrored into a third list.
S3-B instance (2026-09-23): telemetry continuity is ONE predicate,
`telemetry.SequenceTracker` (C: `bcir_seq_observe`), which the BTLM stream
decoder and the envelope intake both express -- and the refactor is
behaviour-identical to the decoder's own copy over 20,000 random streams across
the 2**31 boundary. The intake's stale-generation law is G14's `is_stale`, not a
third spelling. The G15 rows are graded by one function behind one entry point,
`tools/c/check_ring.py`, which the C gate and the fault table both call where the
gate had carried a copy of its own. The ring's shared constants are read out of
`bcir_ring.h` by the tests, and the control slot is derived from the record ABI's
bound and held to it on both rails (a test, and a `_Static_assert` where the two
headers meet).
S3-C instance (2026-09-24): the oracle spelled a shard set's frame three times:
`frame_of`, and again inline in `split` and in `reassemble`. The grader compared
the C twin only against `split`, so the public function could drift from both the
split and `bcir_shm_frame` while no row moved (the fault table's `NOT CAUGHT`).
There is now one private `_frame`, and the grader reads the frame and the shards
through `frame_of` and `sub_pack`, the twins of the C calls the harness prints.
The same slice made the registry binding one predicate:
`bcir_ctl_pack_registry_digest` is used by `bcir_ctl_admit_pack`, by the shard
manifest's encoder and by its reassembly, where the manifest had been about to
carry a digest of its own. The Stage 3 exit flow is written once for both native
rails (`runtime/c/test_stage3.h`, the data boundary supplied through a table of
operations), not twice.

The #719 trap recurred in the same slice. `bcir/tests/test_cpp_handoff.py` kept
its own copy of the C++ seam's source list and its own copy of the gate's reject
probe. The focused runs never touched it, and the complete quick tier caught it:
two link failures against the pre-G16 API. The test now builds through
`handoff_fixtures.build_cpp_program`, the one C++ build every harness uses. It
shares one artifact minter with the gate (`seam_artifacts`), and one committed
probe (`test_orchestrator --reject`, which must also admit the clean pack, L2).
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
one shared predicate — see L14),
`test_staged_dependency_metadata_is_audited` (the third rail; the same
defect went unfixed on it for two rounds after the first two closed it),
`test_staged_expected_inventory_is_audited` (a rail reconciles every input
it reads, not just the one the first fix reached).
Post-campaign instances: `test_advisory_skipped_dependency_is_a_finding` (a
dependency the engine could not collect is a `skip_reason` entry in its
report; the gate runs the engine `--strict` and refuses the entry
independently, because a skip inside the audited set is coverage lost, not
a quieter pass); `test_advisory_coverage_is_reconciled_against_the_declaration`
(what the engine audited is checked against what was declared: pip-audit's
resolver run drops its scratch venv's own setuptools from the report, so
the one security-motivated floor in the tree was audited by nothing and
exited 0 — the floor run now covers it by name, and a declared name neither
run reports is a FAIL that says which).
Installed-environment audit instance:
`test_installed_mode_reconciles_coverage_and_findings` (every distribution the
interpreter sees must come back audited; the repository's own distribution is
the one declared exclusion, reported, because an unrelated project may own
that name on PyPI).
S0-B instances (2026-09-04): `test_mlir_fixture_inventory.py` reconciles the
fixture directory against the runner scripts both ways (a fixture nothing
runs, a reference to no fixture), reading both rather than a third list;
`tools/irdl/check_inventory.py` reconciles the ODS dialect, the IRDL
projection and the manifest of declared-unprojected operations both ways,
and refuses an empty inventory as vacuous.
S0-C instance (2026-09-05): the law-rail fixture `structural_corpus.mlir` is
GENERATED from the corpus and `--check` refuses drift in the quick tier, so the
cases `check_passes.sh` executes are the cases the oracle runs -- never a third
list that can fall behind either.
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
S0-F instance (2026-09-05): the native cost rig printed `native microbench
(bare-metal)` as a literal -- a scope it never checked, so it held under WSL
and under this session's hypervisor. The tenancy is now DERIVED in the tool
from the host's own signals (hypervisor flag and nodes, DMI, WSL, container,
the PMU event source) with the closed set `bare-metal` / `virtualized` /
`containerized` / `unproven`, "bare-metal" reserved for no virtualization
signal plus an exposed PMU, and the reader refuses a provenance whose claim
the evidence does not attest
(`test_native_rig_reports_census_samples_and_an_attested_tenancy`,
`test_calibrate_native_refuses_an_unproved_bare_metal_claim`). A label a
tool cannot check is a scope it has not declared.
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
Post-campaign instance: the advisory tests fake `which` and the bounded
runner together (and clear `PIP_AUDIT`), and `test_advisory_output_is_redacted`
runs a stub engine end to end, so a host's real pip-audit never decides a
unit verdict and the spawn path is exercised wherever the stub can execute.
Installed-environment audit instance: the enumeration of the interpreter is a
seam (`_installed_distributions`) the witnesses replace, so the quick tier
never audits the host it runs on; `test_installed_mode_audits_the_interpreter_by_exact_public_pin`
fakes both the seam and the bounded runner.
S0-A instance (2026-09-04): `test_find_bcir_opt_never_returns_stock_mlir_opt`
resolved the finder against the real repository root, so on a host that had
built the MLIR rail in-tree (`build/mlir-build/bcir-opt`) the finder rightly
returned that real binary and the unit verdict flipped; it now searches a
temporary root, and what the host has built no longer decides it.
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
**A skip is where a shipping defect hides.** An exclusion converts "the
wheel is broken" into "this module does not run here", and the two read
identically in a green run. `bcir/kbcir/tables/*.json` is library data —
`tile_prior`, `bayescal` and `microbench` read it to apply a measured
profile — so an installed wheel without it raised `cannot read calibrated
profile` from `close_loop()` itself, for every user, not only from a test.
The registry had absorbed that as one more repo-only module, taking six
runnable tests with it. Read every entry as a question about the PACKAGE
first and the test second.
Witnesses: `test_registered_suite_survives_missing_repo_only_trees`,
`test_packaged_library_data_is_registered_for_shipping` (the exclusion was
masking a defect in the shipped library, not classifying a test),
`test_packaged_asn1_modules_are_read_from_the_package` (the second such
exclusion in two rounds: the resource WAS shipped, and a
working-directory-relative `open` could not find it — checked over the
syntax, so a docstring describing the defect is not the defect),
`test_repo_only_modules_are_classified_before_import` (import failure
catches only the modules that import a missing tree; the ones that
import cleanly and then read a missing asset must be declared),
`test_repo_only_registry_covers_the_compiling_tiers` (a registry
validated at a tier that hides the toolchain is not validated: the
quick tier's `which` gate made 23 C-compiling modules self-skip).
**Port note:** the same law, harder to see in C: an installed library
whose CTest manifest names build-tree fixtures, a pkg-config file
pointing at headers the install step never copied, a `make check` that
passes only in the source directory. The manifest is the promise; the
install tree is the audit.

### L22 — Every rule admits a witness
A law's rows are proved both ways: an illegal case the row refuses AND a
legal case it admits. A row whose admissible set is empty under the other
rules in force is not strictness, it is a defect no green run can show. The
R12 address-width table carried 16-bit contracts for `avr` and `msp430`
while the op-level address floor refused every operand narrower than 32
bits: on those targets an i16 address failed the floor and an i32 address
failed R12, so no MMIO or atomic access was legal and every fixture over
them would have been an illegal case (#762 review, 2026-09-05). This is the
dual of L2: prove the gate can PASS on the input it exists to admit, not
only that it can fire. The structural corpus asks it per law family (a legal
and an illegal case each,
`test_the_corpus_is_non_trivial_and_declares_every_rail`); this law asks it
per row, and the rows below the floor are gone from both tables rather than
kept as a contract nothing can meet.
Witnesses: `test_the_pointer_width_tables_are_one_table_on_both_rails` (no
tabulated width below `ADDRESS_FLOOR_BITS` / `kAddressFloorBits`, on either
rail), the corpus's `addr.i32_under_arm64_32` and `addr.i32_under_riscv32`
(a legal address per tabulated width class),
`verify_plan_annotations.mlir` `@plan_ok` (the plan the planner emits is
accepted before four corruptions are refused).
S3-B instance (2026-09-23): S3-A declared the control record bound, 192 bytes, as
"the slot stride G15's ring may adopt" -- but a ring slot spends 24 bytes on its
header, so a 192-byte slot carries 168: a record of the bound was a legal case no
such ring could admit (every v1 kind fits, which is why nothing failed). The
geometry law now refuses a CONTROL ring below 256 bytes, and both halves have
witnesses: `geometry-control-slot` (a 192-byte control ring refused on both rails)
and `test_a_control_ring_carries_every_legal_control_record` (a record of the
bound delivered). The slice's naming rule has its witness in one program:
`test_the_live_ring_and_the_v1_ring_emitter_coexist_in_one_program` compiles,
links and runs the v1 emitter's output beside the live ring, whose C API had
claimed three of the names that emitter emits.
S3-C instance (2026-09-24): the pack table's dispatch law binds an admission to
(generation, registry digest), and the second half had no witness. In every
one-plane scenario, a registry change also moves the generation, so the
generation check alone refused every stale dispatch, and a table that ignored the
registry passed. A plane that restarts reaches generation 1 again, and only the
binding tells the two incarnations apart. That case now has witnesses in both
directions, on all three rails: `admission/restart-other-registry` (refused) and
`admission/restart-same-registry` (still admitted, so the binding is exactly the
pair, not the plane's identity). A mutant that drops the registry compare fires
`handoff.stale.dispatched` on each rail that runs the table.
**Port note:** identical everywhere; in C the shape is a range check whose
lower bound another check has already raised past its upper bound, or an
`enum` value no `switch` arm admits.

### L23 — Every objective is scored against its trivial solution
A reported optimization number is evidence about the mechanism it was meant
to train only beside the value reachable by ignoring that mechanism
entirely. Ask of every objective "what does this score for a model that
never consulted its input?", compute it, and publish it in the same
artifact: a descent from a large loss to a small one is a descent and
nothing more until the floor is beside it. Distilling this repository's own
lexical provider into a hosted embedding student over its own corpus
reported `0.5135 -> 0.0383`, which reads like learning; the loss reached by
any mutually orthogonal embedding — which encodes nothing about the teacher
at all — was 0.0161 on the same targets, the student's off-diagonal Gram
correlated with the teacher's at 0.146, and its held-out error (0.0465)
never reached the floor. Every number that run reported was consistent with
learning. The objective is `mse_loss(E @ E.T, targets)` over an L2-normalized
`E`, so the diagonal contributes exactly zero for any model and the whole
objective is the off-diagonal fit — a regime where the trivial solution
scores well by construction, and best of all where the teacher is most
nearly orthogonal (2026-09-10 embedding experiment,
`training/LEARNED_EMBEDDING_GATE.md` §2).
A one-sided bound has the same defect from the other end: a discrimination
cap refuses a collapsed embedding set at ~1.0, and the input that scores
BEST on it is orthogonal noise, which cannot rank anything either. So the
floor is not a verdict on its own — the same exercise checks that the
untrained student starts on the far side of it (else the comparison says
nothing) and that the trained student has not collapsed.
The repository had already articulated this law on another rail and not
carried it across: `training/tools/verify_retrieval.py` scores its model
against two null strategies — a seeded random ranker and a query-ignoring
constant ranker — gates a 20x lift over each, and requires a shuffled index
to collapse to the noise floor. Three nulls on the retrieval rail; none on
the training rail that feeds it. That is L14 in its usual shape: a mechanism
landed on one rail out of two.
Witnesses:
`test_relational_reference_loss_is_the_floor_that_ignores_the_teacher`
(bcir/tests/test_hosted_training_pipeline.py: the floor is computable, total,
exactly zero for a teacher with nothing to teach, and is the shared objective
evaluated at the trivial solution — with a probe whose diagonal differs from
the targets', because one that matches cannot see a reduction that skips the
diagonal); the `mse_loss` binding in `tools/models/test_training_pipeline.py`
(the pure-Python objective IS the stage's own, at the identity and away from
it, so a reduction change moves the loss and the floor together or fails);
and `exercise_relational_targets` / `exercise_embedding_distillation` in
`training/tools/verify_ml_components.py` (a teacher whose Gram is the
identity is refused as vacuous; the reported loss must equal the objective
evaluated on the model the stage returned; the untrained student must start
above the floor; the trained student must not have collapsed).
`MIN_LIFT_OVER_RANDOM` / `MIN_LIFT_OVER_CONSTANT` / `MAX_SHUFFLED_FRACTION`
in `verify_retrieval.py` are the law's origin evidence, predating the finding
by a rail. This law came from an experiment rather than an adversarial review
round, so it does not move the staleness counter below.
S3-B instance (2026-09-23): the live ring's throughput row was first published
beside "a minimal unchecked queue measures ~15x -- the floor for any two-thread
handoff here", a floor measured with the threads wherever the scheduler put them.
Pinned to each vCPU pair of the same host within one hour, that floor read
~1.1-6x and the ring ~13-49x (82-313 ns per record, `memcpy` steady at ~6.4 ns):
the comparison had measured placement on both sides, and the "floor" was an
artifact of it. A trivial solution is a floor only under the conditions of the
number it bounds -- same placement, same host state -- and on a host that cannot
hold those fixed (a virtualized 4-vCPU runner) the honest report is the range and
the refusal to attribute, which is what the row, the ABI spec and the roadmap now
say. The claim was retracted before the PR opened.
**Port note:** every cost model, speedup ratio and calibration is an L23
site. A plan's price is evidence only against the unfused serial plan; a
kernel's timing only against the baseline it replaced; a coverage number
only against what an empty corpus would report. BCIR already prices a
`ratio` row and re-derives R9's step cost for exactly this reason.

### L24 — An artifact's bytes do not depend on the host that wrote them
A generated file is evidence only if two hosts generate the same one. Python's
text mode translates `"\n"` to the platform's line ending on write, so
`open(p, "w")` and `Path.write_text` without `newline=` emit LF on Linux and
CRLF on Windows from identical source and identical logic — and everything
*derived* from those bytes moves with them while the contents stay logically
equal. `training/tools/build_chunks.py` wrote the retrieval corpus that way:
on Windows the corpus fingerprint, every per-file `sha256` the catalog
manifest records, every part's content digest, and `locator.bin` — whose byte
offsets shift with the length of every line before them — all differed, so two
hosts disagreed about the content address of an identical corpus and a
generation published on one could never verify on the other (2026-09-11,
`training/DATABASE_ROADMAP.md`, defect 5). Six further generators write
*tracked* files the same way, and CI already byte-gates two of them with
`git diff --exit-code` after a fresh emission — a trip-wire that would have
caught this the first time it ran on a host that chose CRLF, and never did
because it runs on ubuntu.
The defect is invisible to the obvious witness. Every drift gate over those
files compares in *text* mode, where universal newlines translate CRLF back to
`"\n"` on read, so each one stays green while the file on disk differs (L11).
What finally surfaced it was pinning a fingerprint into the repository and
letting the Windows host-portability job read it back — a cross-host
comparison of the artifact itself, which nothing had ever done.
A repository that declares this for its tracked tree has not thereby declared
it for what its tools generate: `.gitattributes` opens `* text=auto eol=lf`
and cannot reach a file written at run time.
Witnesses: `test_every_generator_pins_its_line_ending` (no artifact-producing
module under `bcir/`, `tools/`, `training/` or `.claude/` lets the host
choose), `test_the_matcher_sees_both_spellings_and_neither_false_positive`
(the predicate itself, against writes that must and must not be flagged),
`test_no_tracked_text_file_carries_a_carriage_return` (the index, derived
rather than curated), and `check_line_endings` in `verify_database.py` (the
built corpus on disk).
**Port note:** the C shape is any text-mode `fopen` on Windows, where `"w"`
performs the same translation `"wb"` does not; the general shape is any
serializer whose output depends on the platform rather than on its input —
a locale-dependent number format, a path separator, a hash of a `struct`
with padding, a directory listing in filesystem order.

### L25 — A fault injection must reach the code under test
A RED sweep is the only evidence that a gate can fail, so the sweep's own
integrity is load-bearing: a sweep that cannot distinguish *the check did not
fire* from *the defect never arrived* produces a sentence that reads as evidence
in both directions and is one in neither. The injection is a claim about the
process the gate ran in, and until that is established the run has measured
nothing.

CPython makes this concrete. A cached `.pyc` is revalidated against the source's
`(mtime, size)`, so restoring a fault whose replacement is the **same length** —
`ORDERED_INDEX_SHARE = 0.70` becoming `= 0.55`, a digit for a digit — can leave a
cache the interpreter still believes, and the next subprocess imports bytecode
compiled from the other version of the file. Observed 2026-09-11: a sweep over
`verify_langref.py` reported "a threshold moves in the code and not in the
document — DID NOT FIRE" against a checker that was, in fact, defective in a
different way; the fault had been compiled away and the run proved nothing about
either. The same mechanism runs the other way and is worse: faulted bytecode
surviving a restore makes the *next* fault's red run look like a catch.

Four properties make a sweep's verdict worth reading, and all four are mechanical
rather than remembered:
1. **Bytecode is discarded** before the gate runs and again after each restore.
2. **The injection is confirmed to have landed** — the file is re-read and its
   digest compared — so an anchor that matched nothing is a harness error, not a
   fault that went uncaught. An anchor matching twice is equally refused: two
   defects attributed to one check.
3. **The restore is confirmed byte-exact** before the next fault, because a sweep
   that corrupts the tree it audits is worse than no sweep.
4. **The clean tree is GREEN first.** "It went red with the fault in" means
   nothing if it was already red; the control run is the experiment, not a
   formality, and its absence invalidates every row.
A sweep is also subject to L2 in its own right: an empty fault table, or one whose
entries all fail to anchor, is a failure and not a clean run.
The general shape is the *instrumented-subject* error — L9 says the instrument
must be unswallowable by its subject; this says the instrument must be able to
show that it touched its subject at all. Any harness that mutates a system it
also measures owes the same four proofs.
Witnesses: `tools/testing/red_sweep.py` is the single implementation (L14), and
`bcir/tests/test_red_sweep.py` drives each refusal — an empty table, an anchor
matching zero or two sites, a replacement identical to its anchor, a control run
that is already red, and a restore that does not reproduce the original bytes.
The committed fault tables under `tools/testing/faults/` are the standing
evidence that the gates they name can fail.
S3-B instance (2026-09-23): the RED simulation replaced each G15 entry point with
one that raises, and its first version replaced `encode_envelope` in the module
whose corpus builders also call it -- the "absent codec" fault broke the fixtures
before any rail saw them, and the rows would have counted a harness error as a
rail's failure. The corpora are now built before anything is patched, and the
rows the fixture set did not touch reproduce exactly (29 / 22 / 10 / 104 / 6):
that is the control. The slice's own mutation campaign first ran from a scratch
script with no bytecode discipline; it is committed as
`tools/testing/faults/ring.json` and run by the one harness that proves each
injection landed.
**Port note:** the C/C++ shape is a stale object file or a `ccache` hit after a
same-size source edit, and any build system whose staleness test is coarser than
content — timestamps, sizes, or a hash of the command line rather than of the
translation unit. The language-independent shape is any experiment that reports a
negative result without first showing the treatment was applied.

## Campaign classification summary

Every review-thread finding from the campaign (240 threads, rounds 1–42)
is graded under the harvest protocol. The full per-finding
index is `docs/security/pr749-harvest.csv`; the campaign ledger tracks the
same data round by round.

| Grade | Findings | Share | Meaning |
|---|---|---|---|
| **NEW-LAW** | 20 | **8.3%** | Originated a registry law (L1–L13, L15–L21; L14 emerged from the repetition itself, not one finding) |
| **INSTANCE** | 181 | **75.4%** | New entry point to a registered law — the law gained a witness |
| **LOCAL** | 39 | **16.2%** | No transfer value beyond the code touched |

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

Round 35 produced five findings and **no new law**, so the counter stands
at **1 of 3**. Its sharpest result was an instance of L21 itself: the law
was one round old and its implementation already proved incomplete —
import-time classification caught only the modules that import a missing
tree, not the ones that import cleanly and then read a missing asset. A
newly declared law being tested and found under-implemented is the loop
working exactly as intended, and it is the strongest argument yet that
laws should be landed with their full witness set rather than their first
one.

Round 36 produced five findings and no new law, so the counter stands at
**2 of 3**. Its pattern is worth recording: three of the five were the
index/worktree reconciliation and the ingress cap built in rounds 33–35
applied unevenly — the dependency rail never got the reconciliation, the
worktree read never got the `cap + 1` its staged sibling had, and the
staged path never learned the symlink check its worktree sibling always
had. That is **L14** (one predicate per repeated defect) reasserting
itself: a mechanism landed on two rails out of three is a defect on the
third, and the loop found each one.

Round 37 produced six findings and no new law, so **the staleness rule
fires**: three consecutive zero-NEW-LAW rounds. The pattern that began in
round 36 sharpened rather than broke — every one of the six was an
already-registered law reaching one path and not its sibling: the cap that
landed on the boundary audit but not the secret scan, the inventory shape
check that validated fields without requiring them, escaped keys and
wrapped values each handled but not composed, dotted keys crossed by the
inline rule but not the collectors, the repo-only registry validated at a
tier that hides the toolchain, and the seed guard given to the decoder
campaign but not the differential. The registry has been stable for three
rounds while its implementation catches up — which is precisely the
condition the rule was written to detect.

Round 38 (seven findings, again all instances) was run at the owner's
request after the rule had already fired, and it did not disturb the
reading: a BOM-less UTF-32 probe beside the UTF-16 one, quoted segments
beside bare ones in dotted keys, `staged_mode` on the second rail a round
after the first, an ingress cap on the inventory beside the one on
pyproject, and two more unguarded paths. Its one genuinely new *surface*
was L7's: a PEP 508 direct reference carries a credential in URL
userinfo, which no scanner rule reads, so the dependency audit was
republishing it through `--json-out` — a real leak, and still an instance
of a law the registry already held.

Round 39 (six findings, again all instances) is the fifth consecutive
zero-NEW-LAW round, and its shape is the clearest evidence yet that the
loop has turned inward: **three of the six were follow-ons to round 38's
own fixes**. The leak closed in round 38 had been witnessed by a substring
check — `"example.com" in text` — which any redaction shape that keeps the
host would satisfy, so the witness did not hit its law (**L11**); it is now
structural equality on the parsed field. The redaction itself reached the
declared block but not pip-audit's own `stdout_tail` and `error`, so the
same credential still had an exit (**L7**, the second half of one leak).
And the BOM-less UTF-32 probe added in round 38 classified by NUL density,
which is an ASCII-shaped assumption: a CJK document encoded in UTF-32
carries too few NULs to pass the threshold, so the probe written to stop
treating UTF-32 as binary still treated *multilingual* UTF-32 as binary
(**L5**) — decode validity replaced the density count. The remaining three
are the familiar sibling-path pattern: staged-inventory reconciliation
reaching the expected-inventory blob a round after it reached the declared
metadata (**L15**), and escaped quotes handled in one quoting branch but
not the other, twice (**L5**).

This round also introduced a third reviewer. One of the six came from
CodeQL rather than Codex, and it landed on a *witness* rather than a gate —
the one place a review loop is structurally blind to itself, since a test
that passes is not evidence that it would fail. That is worth naming as
the loop's own limit, not a defect it found.

Round 40 (three findings, all instances) is the sixth consecutive
zero-NEW-LAW round, and it is the first to produce a finding graded
**L14** — the law that until now had no originating finding because it
emerged from repetition rather than from any single defect. The boundary
audit copied credential-shaped display paths verbatim into its findings
and its symlink inventory, and printed them to the CI log; the secret scan
has redacted exactly those components since round 24. Sixteen rounds, one
rail apart, same defect. Wiring the boundary rail to the scan rail's
`redacted_path` then exposed a second-order version of the same law: the
predicate was **partial**, returning `<redacted-path>` for any path with no
matching component, and was correct only because its one caller tested the
path first. A shared predicate carrying an unstated precondition is a
defect held in reserve for its second caller — which is what five
previously-green boundary tests said the moment the second caller arrived.
The predicate is now total.

The other two were the round-38-and-39 pattern continuing. YAML node
properties (`password: !!str "x"`, `password: &dbpass x`) stand between a
key and its scalar without changing the scalar, and every RHS rule stopped
at the property — one fragment now steps over them for the inline,
wrapped and block-scalar rules alike, while an *alias* (`*name`) stays
suppressed because it points at another node rather than holding a value.
And the dependency-URL redaction, closed in round 38 and extended in
round 39, was still matching only `user:secret@`: a token used as the
whole username survived, and a username beside a password was preserved
verbatim. Userinfo is credential material by position, so position is what
it redacts now. That is the same leak reaching its third round — which is
the clearest possible statement that the loop is finishing its own work
rather than opening new ground.

Round 41 (nine findings, all instances) is the seventh consecutive
zero-NEW-LAW round, and the largest since round 38. It is also the round
that most clearly separates *what the loop finds* from *what the fix turns
out to be* — because on its best finding the two were different laws.

Codex reported that the packaged runner excluded the whole
`test_calibrator` module although six of its seven tests need no
repository asset, and proposed narrowing the classification. Running the
module against the staged wheel tree confirmed the count exactly — and
named the asset: `bcir/kbcir/tables/x86_64_reference.json`. That file is
not a test fixture. `tile_prior`, `bayescal` and `microbench` read it to
apply a measured profile, `package-data` shipped the ASN.1 sources beside
it but not the tables, and so **every installed wheel raised `cannot read
calibrated profile` from `close_loop()` itself** — a library defect, for
every user, that had been sitting inside a test-runner exclusion. The fix
is to ship the table (L21), not to narrow the skip; the skip then
dissolves on its own. The corollary is now written into L21: *a skip is
where a shipping defect hides*, because an exclusion turns "the wheel is
broken" into "this module does not run here" and the two read identically
in a green run.

The rest divide into three familiar shapes. **YAML keeps yielding
spellings**: explicit mapping keys (`? key` / `: value`) and folded
double-quoted scalars join round 40's node properties — three rounds, one
law, one format, each a production of the same grammar that the rule set
had not enumerated. That is the strongest argument yet for how the C port
should begin: from the format's productions, not from the accumulated
regexes. **The staged rail is still catching up to the worktree rail**:
neither audited input checked `staged_mode`, so a staged symlink's TARGET
was parsed as metadata, and the staged inventory decoded with `replace`
where every sibling read decodes strictly — a gate that disagrees with
itself across two paths (L12), for the third and fourth time. And **two
budgets measured the wrong resource or nothing at all**: concatenated
gzip/bzip2 members advance the logical output cap by zero, exactly as
concatenated xz streams did before round 34 fixed only xz (L3, L14); and a
`RecursionError` from a 40 KiB depth bomb escaped before any report
existed, where the review parser has caught the same shape since round 30
(L1, L14).

The remaining finding is the campaign's cleanest L9: the differential's
`_VerifyHang` derived from `Exception` while the decoder campaign's
`_DecodeHang` has derived from `BaseException` since round 30 — so a
verifier wrapping its work in `except Exception` caught its own watchdog,
returned an ordinary verdict, and was accepted as having answered in time,
with the one-shot timer already spent. Eleven rounds, one law, two rails,
two spellings.

Round 42 (four findings, all instances) is the eighth consecutive
zero-NEW-LAW round, and it settles what round 41 opened. The corollary
written into L21 last round — *a skip is where a shipping defect hides* —
was tested immediately: Codex reported a second over-broad exclusion, the
`test_asn1_constraints` module skipped whole for one of its 21 tests, and
underneath it was the same class of defect. The ASN.1 module the test
compiles **is** shipped; the test opened it as
`open("bcir/asn1/BCIR-StreamPack.asn1")`, a path relative to the working
directory, which resolves only when a test happens to run from the
repository root and fails inside the very wheel that ships the file.
`ecn_syntax.frame_header_source` has read its module through
`importlib.resources` since it was written, with a docstring explaining
exactly why; the two `.asn1` modules beside it had no such reader. One
predicate now (`bcir.asn1.module_source`), shared by all three.

Two rounds, two exclusions, two different underlying defects — a missing
`package-data` entry and a working-directory-relative read — so the
registry was **re-derived** rather than edited twice: every one of its 64
entries was run against a freshly staged wheel tree, per test rather than
per module. Four more entries were excluding modules that pass entirely in
the wheel. With the two Codex found, five entries were hiding **98
runnable tests**, among them 181 in the security suite this very campaign
depends on. The regeneration note now says to read a candidate entry as a
question about the package first and the test second, and records the trap
that made the first survey worthless: importing `bcir.tests.run_all` from
the checkout binds `bcir` to the repository, so every module under test
comes from there and the survey reports uniformly clean.

The other three are single-line statements of laws the registry already
holds. A reviewer QUOTES the code it reviews, so `security_concerns` and
`summary` are the report fields guaranteed to carry whatever secret it
just found, and this rail copied them into `--json-out` verbatim (L7) —
now redacted through the scan's own predicate, so a report cannot remove
less than the scan would report. `json.loads` keeps the last value for a
repeated key, so an inventory declaring `"runtime": ["hidden-package==1"]`
and later `"runtime": []` audited clean while saying two things (L4) — the
review parser has refused exactly that since round 23. And
`--require-compiled` proved only that `bcir-opt` was *discoverable* at
startup: every per-case call re-resolved it, and only `FAIL` became a
disagreement, so a binary that vanished mid-run left every witness
`UNAVAILABLE/SKIPPED` and the report `PASS` over zero compiled executions
(L2). Discovery is resolved once per campaign now, and an unavailable
required witness is a disagreement.

Where the instances concentrated (finding count per law, origin included):
L5 scannable-data coverage 39 · L3 resource-commit bounds 25 · L1
every-exit-a-verdict 21 · L11 witness/law pairing 13 · L15 discovery
reconciliation 12 · L8 process trees/pipes 10 · L10 rejection contracts 9 ·
L18 heuristic scope 9 · L7 report-as-egress 9 · L2 gate-can-fire 8 · L4
attribute-or-refuse 7 · L17 names-are-paths 7 — the remaining laws
account for the rest. The two heaviest laws are exactly the two that port
hardest into C (allocation bounds and data-coverage claims), which is the
campaign's transfer thesis in one line.

Where the noise concentrated: the deleted Python 3.10 TOML fallback
absorbed 21 findings (8.8% of the entire campaign — the single largest
family, all LOCAL, resolved by raising the floor to 3.11 per L4), and the
rolled-back alias/dataflow tracking in the boundary audit absorbed 13 more
(5.4%, resolved by declaring scope per L18). Those two structural changes
retire **87% of all LOCAL findings** — the loop's zero-yield surface —
which is what steers future review rounds toward NEW-LAW territory.
