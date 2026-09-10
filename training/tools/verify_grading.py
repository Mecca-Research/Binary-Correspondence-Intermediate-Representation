#!/usr/bin/env python3
"""Gate the shared grading and dataset rails (Phase 0.7).

An extraction is easy to declare and hard to keep. `training/tools/grading.py`
is only shared machinery for as long as nothing subject-specific creeps back
into it, and the copies it replaced are only gone for as long as nobody writes a
fourth `find_tool`. Both are checkable, so both are checked here rather than
described in a README:

  1. **The shared rail names no subject.** After docstrings and comments are
     stripped -- prose may explain the boundary, code may not cross it -- no
     module under `training/tools/` may mention a compiler, a dialect, or a file
     extension belonging to one subject.
  2. **A subject that is not LLVM can be graded.** A synthetic subject is built
     in a temporary tree, with its own answer kind and its own validity tool,
     and graded end to end through the real kernel. An abstraction that only
     ever runs against its original caller has not been shown to be one.
  3. **The shared predicates are defined once.** Before this extraction three
     tools carried their own `find_tool` and two more their own
     `model_visible_prompt` and `normalized_text`. The copies agreed; nothing
     made them agree. A second definition anywhere under `training/` fails.
  4. **An untrusted answer cannot escape.** Lexical and symlink escapes are
     rejected, and an attempt tree carrying a build file is refused outright.
  5. **A skip is never a pass.** A missing tool leaves its points unearned,
     marks the report `reduced`, and is excluded from the executed score.
  6. **A bound is a bound.** A tool that runs long is a `timed_out` result with
     a verdict, not an exception; output past the cap is truncated and says so.

    python3 training/tools/verify_grading.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import dataset_export  # noqa: E402
import distillation  # noqa: E402
import grading  # noqa: E402
import safe_process  # noqa: E402
from subject_profile import AnswerKind, SubjectProfile, ToolCheck  # noqa: E402

# The modules that must stay subject-neutral. `verify_grading.py` itself is
# excluded: naming what it forbids is its job.
SHARED_MODULES = (
    "grading.py",
    "dataset_export.py",
    "subject_profile.py",
    "safe_process.py",
    "distillation.py",
    "build_distillation.py",
)

# Tokens that would mean a subject leaked into the shared rail. Matched against
# code with docstrings and comments removed, so the explanation of the boundary
# is allowed to name the thing it is drawing the boundary around.
SUBJECT_TOKENS = (
    r"llvm",
    r"mlir",
    r"clang",
    r"\blli\b",
    r"\bopt\b",
    r"opaque[_ ]pointer",
    r"\.ll\b",
)

# Predicates that existed in several copies before the extraction. Each must now
# have exactly one definition anywhere under `training/`.
SINGLE_DEFINITION = (
    "find_tool",
    "model_visible",
    "normalized_text",
    "run_bounded",
    "grade_entry",
    "confined_answer",
    "validate_attempt_tree",
)
# **Declared scope.** These are predicates that carry policy -- a search order, a
# redaction rule, a confinement decision -- where a second copy is a second
# answer waiting to be given. A one-line wrapper over the standard library is
# not that: `sha256_text` is `hashlib.sha256(text.encode()).hexdigest()` in eight
# corpus builders and cannot drift into disagreement, so it is out of scope
# rather than exempted case by case. The next finding of that shape belongs
# here, in this paragraph, not in a growing allowlist.


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def require(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)


def code_without_prose(source: str) -> str:
    """The module's code with docstrings and comments removed."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body.pop(0)
    return ast.unparse(tree)


def check_no_subject_in_shared_rail(report: Report) -> None:
    scanned = 0
    for name in SHARED_MODULES:
        path = TOOLS_DIR / name
        if not report.require(path.is_file(), f"{name} is missing from the shared rail"):
            continue
        code = code_without_prose(path.read_text(encoding="utf-8"))
        scanned += 1
        for token in SUBJECT_TOKENS:
            match = re.search(token, code, re.IGNORECASE)
            report.require(
                match is None,
                f"{name} names {match.group(0)!r} in code; the shared rail must not "
                "know which subject it is grading -- move it to the subject's profile"
                if match
                else "",
            )
    report.require(scanned == len(SHARED_MODULES), "not every shared module was scanned")
    print(f"[neutral] {scanned} shared module(s) carry no subject identifier in code")


def defined_functions(path: Path) -> set[str]:
    """Every function in the module, methods included.

    Methods count: moving `model_visible_prompt` onto `SubjectProfile` did not
    make a second copy of it acceptable, and a scan that only saw module-level
    definitions would have said the predicate had vanished rather than moved.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def check_single_definition(report: Report) -> None:
    """A predicate with two definitions is a drift that has not happened yet."""
    sources = sorted(CORPUS_ROOT.rglob("*.py"))
    report.require(bool(sources), "no python source found under training/; nothing was scanned")
    where: dict[str, list[str]] = {}
    for path in sources:
        relative = path.relative_to(REPO_ROOT).as_posix()
        if "/autograder/tests/" in relative:
            continue  # a test may define a helper of the same name for itself
        for name in defined_functions(path) & set(SINGLE_DEFINITION):
            where.setdefault(name, []).append(relative)
    for name in SINGLE_DEFINITION:
        sites = where.get(name, [])
        report.require(
            len(sites) <= 1,
            f"{name!r} is defined {len(sites)} times ({', '.join(sites)}); the copies "
            "agree today and nothing makes them agree tomorrow",
        )
    defined = sum(1 for name in SINGLE_DEFINITION if where.get(name))
    report.require(
        defined == len(SINGLE_DEFINITION),
        f"only {defined} of {len(SINGLE_DEFINITION)} shared predicates were found at "
        "all; this check would pass over a rail that had been deleted",
    )
    print(f"[single]  {defined} shared predicate(s), one definition each")


def synthetic_subject(root: Path) -> SubjectProfile:
    """A subject that is not LLVM, built to be graded by the real kernel.

    Its answer is a JSON document, its validity tool is the interpreter running
    this gate, and its normalizer rewrites the answer into canonical form. None
    of that resembles a compiler, which is the point: if the kernel carries a
    hidden assumption about the subject it grades, this fails.
    """
    subject = root / "training" / "synthetic"
    (subject / "autograder").mkdir(parents=True)
    (subject / "exercises").mkdir(parents=True)
    (subject / "exercises" / "001.prompt.md").write_text(
        "# Describe a record\n\n## Expected observation\n\nA JSON object with a name.\n",
        encoding="utf-8",
    )
    (subject / "exercises" / "001.solution.json").write_text(
        '{"name": "widget", "parts": 3}\n', encoding="utf-8"
    )
    registry = {
        "schema_version": grading.SCHEMA_VERSION,
        "exercises": [
            {
                "id": "001",
                "prompt_path": "training/synthetic/exercises/001.prompt.md",
                "reference_solution_path": "training/synthetic/exercises/001.solution.json",
                "answer_kind": "json-doc",
                "required_tools": ["python3"],
                "timeout_seconds": 20,
                "expected_points": 100,
                "scoring_dimensions": [
                    {"id": "artifact", "points": 20},
                    {"id": "validity", "points": 30},
                    {"id": "normalization", "points": 20},
                    {"id": "structure", "points": 20},
                    {"id": "rubric", "points": 10},
                ],
                "structural_assertions": [{"type": "contains", "patterns": ['"name"']}],
                "rubric_checks": [{"mode": "all", "terms": ["widget"]}],
            }
        ],
    }
    (subject / "autograder" / "exercises.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )

    parse = "import json,sys; json.load(open(sys.argv[1]))"
    canonical = "import json,sys; print(json.dumps(json.load(open(sys.argv[1])), sort_keys=True))"
    return SubjectProfile(
        name="synthetic",
        root=subject,
        kinds={
            "json-doc": AnswerKind(
                "json-doc",
                ".json",
                (
                    ToolCheck(
                        id="json.parse",
                        dimension="validity",
                        tool="python3",
                        argv=lambda tool, answer: [tool, "-c", parse, str(answer)],
                    ),
                    ToolCheck(
                        id="json.canonical",
                        dimension="normalization",
                        tool="python3",
                        argv=lambda tool, answer: [tool, "-c", canonical, str(answer)],
                        normalizes=True,
                        label="python3 -c canonical",
                        pass_message="answer rewritten in canonical form",
                    ),
                ),
            )
        },
        dataset_id_prefix="synthetic-exercise-",
    )


def check_synthetic_subject(report: Report) -> None:
    with tempfile.TemporaryDirectory(prefix="bcir-verify-grading-") as name:
        root = Path(name)
        profile = synthetic_subject(root)
        out = root / "report.json"
        code = grading.main(
            profile, "synthetic", ["--self-test", "--format", "json", "--output", str(out)]
        )
        report.require(code == 0, f"the synthetic subject did not self-test clean (exit {code})")
        data = json.loads(out.read_text(encoding="utf-8"))
        result = data["results"][0]
        report.require(
            result["score_percent"] == 100.0,
            f"synthetic subject scored {result['score_percent']}, expected 100",
        )
        report.require(
            result["skipped_checks"] == 0,
            "the synthetic subject skipped a check; its tool is the interpreter and "
            "cannot be absent",
        )
        ids = [check["id"] for check in result["checks"]]
        report.require(
            ids
            == [
                "answer.exists",
                "answer.non_empty",
                "json.parse",
                "json.canonical",
                "structure.01",
                "rubric.01",
            ],
            f"the kernel ran {ids}, not the subject's declared checks in order",
        )
        report.require(
            not result["rubric_coverage_only"],
            "a kind that declares no rubric-only flag was reported as rubric-only",
        )

        # Anti-vacuity: a registry with no exercises makes every loop below it
        # iterate zero times, and the grader would report a clean run over
        # nothing. Refusing it is what stops green from meaning "did not run".
        try:
            grading.validate_registry(
                profile, {"schema_version": grading.SCHEMA_VERSION, "exercises": []}, root
            )
        except ValueError:
            report.require(True, "")
        else:
            report.require(False, "a registry declaring no exercises was accepted")

        # An answer kind the subject never declared must be refused by name,
        # not graded with an empty check list and scored 100%.
        try:
            grading.validate_registry(
                profile,
                {
                    "schema_version": grading.SCHEMA_VERSION,
                    "exercises": [dict(registry_entry(profile), answer_kind="not-a-kind")],
                },
                root,
            )
        except ValueError as exc:
            report.require(
                "unsupported answer kind" in str(exc),
                f"an undeclared answer kind failed for the wrong reason: {exc}",
            )
        else:
            report.require(False, "an answer kind the subject never declared was accepted")

        print(f"[foreign] a non-LLVM subject graded end to end: {len(ids)} checks, 100%")


def registry_entry(profile: SubjectProfile) -> dict:
    path = profile.root / "autograder" / "exercises.json"
    return json.loads(path.read_text(encoding="utf-8"))["exercises"][0]


def check_confinement(report: Report) -> None:
    with tempfile.TemporaryDirectory(prefix="bcir-verify-confine-") as name:
        root = Path(name) / "attempts"
        (root / "001").mkdir(parents=True)
        (root / "001" / "answer.json").write_text("{}", encoding="utf-8")
        outside = Path(name) / "outside.json"
        outside.write_text("{}", encoding="utf-8")

        inside = grading.confined_answer(root, root / "001" / "answer.json")
        report.require(inside.is_file(), "a legitimate answer inside the root was rejected")

        for label, candidate in (
            ("lexical", root / ".." / "outside.json"),
            ("absolute", outside),
        ):
            try:
                grading.confined_answer(root, candidate)
            except ValueError:
                report.require(True, "")
            else:
                report.require(False, f"a {label} escape from the attempt root was accepted")

        link = root / "001" / "link.json"
        try:
            link.symlink_to(outside)
        except OSError:
            print("[skip]    symlinks unavailable on this host; escape-by-symlink unchecked")
        else:
            try:
                grading.confined_answer(root, link)
            except ValueError:
                report.require(True, "")
            else:
                report.require(False, "a symlink out of the attempt root was accepted")

        (root / "Makefile").write_text("all:\n\techo owned\n", encoding="utf-8")
        try:
            grading.validate_attempt_tree(root)
        except ValueError:
            report.require(True, "")
        else:
            report.require(False, "an attempt tree carrying a Makefile was accepted")
        print("[confine] escapes rejected; a build file in the attempt tree is refused")


def check_skip_is_not_a_pass(report: Report) -> None:
    """A tool that is not installed must cost points, not silently earn them."""
    with tempfile.TemporaryDirectory(prefix="bcir-verify-skip-") as name:
        root = Path(name)
        profile = synthetic_subject(root)
        missing = "bcir-tool-that-does-not-exist"
        report.require(
            shutil.which(missing) is None, "the 'missing tool' probe is actually installed"
        )
        kind = profile.kinds["json-doc"]
        absent = ToolCheck(
            id="absent.tool", dimension="validity", tool=missing, argv=lambda t, a: [t, str(a)]
        )
        profile = SubjectProfile(
            name=profile.name,
            root=profile.root,
            kinds={"json-doc": AnswerKind("json-doc", ".json", (absent,) + kind.checks)},
            dataset_id_prefix=profile.dataset_id_prefix,
        )
        registry = json.loads(
            (profile.root / "autograder" / "exercises.json").read_text(encoding="utf-8")
        )
        entry = registry["exercises"][0]
        answer = root / "training/synthetic/exercises/001.solution.json"
        result = grading.grade_entry(profile, entry, answer, {"python3": sys.executable})

        skipped = [c for c in result["checks"] if c["status"] == "skip"]
        # Fail closed. An absent skip is exactly the defect this checks for, so
        # reading skipped[0] to report it would trade a finding for a traceback.
        if not report.require(
            len(skipped) == 1,
            f"expected one skipped check for an uninstallable tool, saw {len(skipped)}; "
            "a missing tool is being graded as though it ran",
        ):
            return
        report.require(
            skipped[0]["points_earned"] == 0.0,
            "a skipped check earned points; a missing tool would then look like a pass",
        )
        report.require(
            skipped[0]["outcome"] == "missing_tool",
            "a skipped check did not report missing_tool, so a wrong answer and an "
            "absent toolchain are indistinguishable",
        )
        report.require(
            result["score_confidence"] == "reduced",
            "a report with a skipped check claimed full confidence",
        )
        report.require(
            result["executed_score_percent"] > result["score_percent"],
            "the executed score did not exclude the skipped check, so a partial "
            "toolchain is reported as a lower score rather than a narrower one",
        )
        print(
            f"[skip]    a missing tool costs {skipped[0]['points_available']:.0f} unearned "
            f"point(s); executed {result['executed_score_percent']}% vs "
            f"{result['score_percent']}% overall"
        )


def check_bounds(report: Report) -> None:
    slow = [sys.executable, "-c", "import time; time.sleep(30)"]
    result = safe_process.run_bounded(slow, timeout=1)
    report.require(result.timed_out, "a command past its timeout did not report timed_out")
    report.require(
        result.returncode == 124, f"a timeout returned {result.returncode}, not the 124 convention"
    )
    report.require("timed out" in result.stderr, "a timeout produced no diagnostic to report")

    flood = [sys.executable, "-c", "print('x' * (1 << 20))"]
    bounded = safe_process.run_bounded(flood, timeout=30, max_output_bytes=4096)
    report.require(bounded.output_truncated, "output past the cap was not flagged as truncated")
    report.require(
        len(bounded.stdout) < 8192,
        f"a bounded capture returned {len(bounded.stdout)} bytes; the cap did not hold",
    )
    try:
        safe_process.run_bounded([], timeout=1)
    except ValueError:
        report.require(True, "")
    else:
        report.require(False, "an empty command array was accepted")
    print("[bounds]  timeout is a verdict; capture is capped; an empty argv is refused")


def check_dataset_kernel(report: Report) -> None:
    """The exporter refuses the shapes that would ship half a dataset."""
    with tempfile.TemporaryDirectory(prefix="bcir-verify-dataset-") as name:
        root = Path(name)
        profile = synthetic_subject(root)
        (profile.root / "dataset").mkdir(parents=True)
        (profile.root / "dataset" / "splits-v1.json").write_text(
            json.dumps({"exercises": []}) + "\n", encoding="utf-8"
        )
        (profile.root / "autograder" / "manifests").mkdir(parents=True)
        try:
            dataset_export.export(
                profile,
                split="all",
                include_solutions=False,
                version_assumptions=lambda manifest, paths: {},
                exporter="verify_grading",
            )
        except ValueError as exc:
            report.require(
                "empty dataset" in str(exc),
                f"an empty manifest directory failed for the wrong reason: {exc}",
            )
        else:
            report.require(False, "an export over zero manifests succeeded")

        text = "see training/x/exercises/001.solution.ll for the answer"
        report.require(
            profile.model_visible(text) == text,
            "a subject with no redaction pattern altered its prompt anyway",
        )
        print("[dataset] an empty manifest set is refused; redaction is the subject's rule")


SYNTHETIC_SOURCES = """
from distillation import RecordSource

SYSTEM_PROMPT = "You are answering about a synthetic subject."


def _two(builder):
    return [
        builder.record(
            task="probe",
            user=f"question {index}",
            assistant=f"answer {index}",
            answer_kind="explanation",
            source_paths=["training/synthetic/source-38.md"],
            gate="training/synthetic/tools/gate.sh",
            claim="the gate pins this answer",
            difficulty=2,
        )
        for index in range(8)
    ]


SOURCES = (RecordSource("probe", _two),)
"""


def write_sources(subject_root: Path, body: str) -> None:
    tools = subject_root / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    (tools / distillation.SOURCES_MODULE).write_text(body, encoding="utf-8")


def check_distillation_contract(report: Report) -> None:
    """A record cannot be built without a gate, and a subject owns its sources."""
    with tempfile.TemporaryDirectory(prefix="bcir-verify-distill-") as name:
        root = Path(name) / "training" / "synthetic"
        root.mkdir(parents=True)

        # A subject that is still a scope statement contributes nothing, and
        # says so by having no sources module rather than by erroring.
        records, counts = distillation.build_subject(root)
        report.require(
            not records and not counts,
            "a subject with no record-source module produced records anyway",
        )

        write_sources(root, SYNTHETIC_SOURCES)
        records, counts = distillation.build_subject(root)
        report.require(len(records) == 8, f"the synthetic subject produced {len(records)} records")
        report.require(counts == {"probe": 8}, f"the task counts are {counts}, expected probe=8")
        if records:
            report.require(
                records[0]["verified_by"]["gate"].endswith("gate.sh"),
                "a record lost the gate that justifies it",
            )
            # The split must be a function of the SOURCE FILE and of nothing
            # else -- that is the Tier-3 leakage rule. Checking that each record
            # re-derives its own split from its own source path is exact, where
            # comparing two records to each other only fails by luck. The probe
            # source deliberately hashes to the rarest bucket so an assignment
            # made per record rather than per file cannot coincide with it.
            mismatched = [
                record["record_id"]
                for record in records
                if record["split"] != distillation.assign_split(record["source_paths"][0])
            ]
            report.require(
                not mismatched,
                f"{len(mismatched)} record(s) carry a split that is not their source "
                "file's; two records from one chapter can then straddle the boundary "
                "and a held-out file is also in training",
            )
            report.require(
                len({record["split"] for record in records}) == 1,
                "records from one source file landed in more than one split",
            )
            report.require(
                records[0]["messages"][0]["content"] == "You are answering about a "
                "synthetic subject.",
                "the system turn did not come from the subject",
            )

        for label, body, expected in (
            (
                "a record with no gate",
                SYNTHETIC_SOURCES.replace('gate="training/synthetic/tools/gate.sh"', 'gate=""'),
                "no gate or no claim",
            ),
            (
                "an empty system prompt",
                SYNTHETIC_SOURCES.replace(
                    'SYSTEM_PROMPT = "You are answering about a synthetic subject."',
                    'SYSTEM_PROMPT = "   "',
                ),
                "empty SYSTEM_PROMPT",
            ),
            (
                "a duplicated task name",
                SYNTHETIC_SOURCES.replace(
                    'SOURCES = (RecordSource("probe", _two),)',
                    'SOURCES = (RecordSource("probe", _two), RecordSource("probe", _two))',
                ),
                "twice",
            ),
            (
                "an empty source tuple",
                SYNTHETIC_SOURCES.replace(
                    'SOURCES = (RecordSource("probe", _two),)', "SOURCES = ()"
                ),
                "empty SOURCES",
            ),
            (
                "a record with no source path",
                SYNTHETIC_SOURCES.replace(
                    'source_paths=["training/synthetic/source-38.md"]', "source_paths=[]"
                ),
                "no source path",
            ),
        ):
            # `str.replace` returns the input unchanged when its needle is
            # absent, so a stale anchor here would quietly turn an injection
            # into a no-op and report the guard as broken. Refuse that outright.
            if not report.require(
                body != SYNTHETIC_SOURCES,
                f"the perturbation for {label!r} changed nothing; its anchor is stale "
                "and this case proves nothing",
            ):
                continue
            write_sources(root, body)
            try:
                distillation.build_subject(root)
            except SystemExit as exc:
                report.require(
                    expected in str(exc),
                    f"{label} was refused for the wrong reason: {exc}",
                )
            except Exception as exc:  # noqa: BLE001 - the point is that it must not happen
                # Refused, but by falling over. A traceback where a verdict
                # belongs loses the finding and misreports the outcome, which is
                # the same fail-open shape these rails exist to refuse.
                report.require(
                    False,
                    f"{label} raised {type(exc).__name__} instead of a structured refusal: {exc}",
                )
            else:
                report.require(False, f"{label} was accepted")
        print("[distill] a subject owns its sources; a record without a gate is refused")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    report = Report()
    check_no_subject_in_shared_rail(report)
    check_single_definition(report)
    check_synthetic_subject(report)
    check_confinement(report)
    check_skip_is_not_a_pass(report)
    check_bounds(report)
    check_dataset_kernel(report)
    check_distillation_contract(report)

    if report.failures:
        print("grading rail gate: FAILED", file=sys.stderr)
        for failure in report.failures[:40]:
            if failure:
                print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"grading rail gate: PASSED ({report.checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
