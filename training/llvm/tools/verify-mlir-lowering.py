#!/usr/bin/env python3
"""Grade registered and conversion-aware MLIR examples from the training registry."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
SHARED_TOOLS = TOOLS_DIR.parent.parent / "tools"
for _path in (str(TOOLS_DIR), str(SHARED_TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from llvm_profile import PROFILE  # noqa: E402

TOOL_BASES = ("mlir-opt", "mlir-translate", "llvm-as", "opt")

# Claims about text a tool produced during *this* run: the pipeline's output and the
# translated LLVM IR. Nothing here can be satisfied without mlir-opt (and, for the last
# two, mlir-translate) having actually run and emitted something.
GENERATED_CLAIM_KEYS = (
    "require_lowered",
    "forbid_lowered",
    "require_llvm_ir",
    "required_runtime_calls",
)
# Claims about files already in the tree. Worth checking, but they hold whether or not a
# single conversion ran, so they cannot stand in for the ones above.
FILE_CLAIM_KEYS = (
    "require_source",
    "required_artifact_metadata",
    "required_artifact_runtime_calls",
)
# The spelling authority for the `checks` object. A key outside this set is an error
# rather than a no-op: `require_lowerd` in a manifest reads exactly like a check that
# runs, and a silently ignored check is the same thing as a missing one with a comment
# claiming otherwise.
CHECK_KEYS = frozenset(GENERATED_CLAIM_KEYS + FILE_CLAIM_KEYS)


def expected_major(registry_major: int) -> int:
    """The toolchain major version to grade against.

    The registry pins a canonical major (``toolchain_major``), but the
    multi-version MLIR CI matrix selects a toolchain through ``LLVM_SUFFIX``
    (e.g. ``-19``). When that suffix names a major, grade against it so each
    matrix entry validates its own LLVM instead of the pinned default; absent a
    suffix (the standalone training rail) fall back to the registry pin.
    """
    match = re.search(r"(\d+)", os.environ.get("LLVM_SUFFIX", ""))
    return int(match.group(1)) if match else registry_major


def run(command: list[str], *, output: Path | None = None) -> tuple[bool, str]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if output is not None and completed.returncode == 0:
        output.write_text(completed.stdout, encoding="utf-8")
    return completed.returncode == 0, completed.stdout


def require_strings(text: str, needles: list[str], label: str, errors: list[str]) -> int:
    """Check each needle, and report how many were checked.

    The count is the point as much as the errors are: a check list that is empty, absent
    or misspelt produces no errors at all, which is indistinguishable from a check list
    that passed. Callers accumulate the returns so the gate can refuse to call itself
    green over zero executed assertions.
    """
    for needle in needles:
        if needle not in text:
            errors.append(f"{label}: required text is missing: {needle!r}")
    return len(needles)


def forbid_strings(text: str, needles: list[str], label: str, errors: list[str]) -> int:
    for needle in needles:
        if needle in text:
            errors.append(f"{label}: illegal/unconverted text remains: {needle!r}")
    return len(needles)


def validate_registry(registry: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    examples = registry.get("examples")
    if registry.get("schema_version") != 1 or not isinstance(examples, list):
        return ["registry must have schema_version 1 and an examples array"]
    seen: set[str] = set()
    # Recursive under examples/, matching verify-manifest.sh's artifact rule: a
    # .mlir file one directory deeper (a self-contained dialect skeleton, say)
    # is still an example and still needs a tier declaration.
    registered_sources = {
        path.relative_to(root).as_posix()
        for path in root.glob("training/llvm/**/examples/**/*.mlir")
    }
    registered_sources.add("training/llvm/autograder/fixtures/mlir/incomplete-bcir-conversion.mlir")
    for index, entry in enumerate(examples):
        label = f"examples[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: entry must be an object")
            continue
        required = {
            "source",
            "required_dialects",
            "required_plugins_tools",
            "pass_pipeline",
            "expected_tier",
            "expected_lowered_artifact",
        }
        missing = required - set(entry)
        if missing:
            errors.append(f"{label}: missing fields: {', '.join(sorted(missing))}")
            continue
        source = entry["source"]
        if not isinstance(source, str) or source in seen:
            errors.append(f"{label}: source must be a unique string")
            continue
        seen.add(source)
        if not (root / source).is_file():
            errors.append(f"{label}: source does not exist: {source}")
        tier = entry["expected_tier"]
        if isinstance(tier, bool) or not isinstance(tier, int) or tier not in range(5):
            errors.append(f"{label}: expected_tier must be an integer from 0 through 4")
        pipeline = entry["pass_pipeline"]
        if tier >= 3 and not isinstance(pipeline, str):
            errors.append(f"{label}: Tier {tier} requires a pass_pipeline")
        if tier < 3 and pipeline is not None:
            errors.append(f"{label}: Tier {tier} must not claim a conversion pipeline")
        if tier == 1 and entry.get("classification") != "unregistered-dialect-sketch":
            errors.append(
                f"{label}: Tier 1 entries must be explicitly classified as unregistered sketches"
            )
        artifact = entry["expected_lowered_artifact"]
        if artifact is not None and (
            not isinstance(artifact, str) or not (root / artifact).is_file()
        ):
            errors.append(f"{label}: expected lowered artifact does not exist: {artifact}")
        tools = entry["required_plugins_tools"]
        if not isinstance(tools, list) or any(tool not in TOOL_BASES for tool in tools):
            errors.append(f"{label}: unsupported required tool/plugin declaration")
        if tier == 4 and not {"mlir-opt", "mlir-translate", "llvm-as", "opt"}.issubset(set(tools)):
            errors.append(f"{label}: Tier 4 requires mlir-opt, mlir-translate, llvm-as, and opt")
        checks = entry.get("checks", {})
        if not isinstance(checks, dict):
            errors.append(f"{label}: checks must be an object")
            continue
        unknown = sorted(set(checks) - CHECK_KEYS)
        if unknown:
            errors.append(f"{label}: unknown check keys (a misspelt check never runs): {unknown}")
        if any(not isinstance(checks.get(key, []), list) for key in CHECK_KEYS):
            errors.append(f"{label}: every check must be a list of strings")
            continue
        # A Tier 3 or 4 entry declares a conversion, so it must claim something about what
        # that conversion produced. Without such a claim the pipeline's exit status is the
        # entire test, and a pass that emitted an empty module passes it -- while the tier
        # census on the final line still reads exactly the same. Tiers 0-2 claim no
        # conversion, so there the tier *is* the claim and an empty `checks` is honest.
        declared = sum(len(checks.get(key, [])) for key in GENERATED_CLAIM_KEYS)
        expected_failure = entry.get("expected_failure") or {}
        declared += len(expected_failure.get("operations", []))
        if tier >= 3 and declared == 0:
            errors.append(
                f"{label}: Tier {tier} declares a conversion but claims nothing about its "
                f"output; add one of {', '.join(GENERATED_CLAIM_KEYS)}"
            )
    missing_sources = registered_sources - seen
    extra_sources = seen - registered_sources
    for source in sorted(missing_sources):
        errors.append(f"unregistered MLIR example: {source}")
    for source in sorted(extra_sources):
        errors.append(f"registry source is outside the governed MLIR example set: {source}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=root / "training/llvm/autograder/mlir-examples.json"
    )
    parser.add_argument(
        "--require-tools",
        action="store_true",
        help="fail instead of reporting reduced coverage when declared tools are absent",
    )
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    errors = validate_registry(registry, root)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    major = expected_major(int(registry["toolchain_major"]))
    tools = {base: PROFILE.find_tool(base, major=major) for base in TOOL_BASES}
    missing = sorted(base for base, path in tools.items() if path is None)
    if missing:
        message = f"reduced MLIR grading coverage: missing {', '.join(missing)}; only Tier 0 file/rubric checks can run"
        print(message, file=sys.stderr)
        return 1 if args.require_tools else 0

    version_errors: list[str] = []
    for base, tool in tools.items():
        ok, version_output = run([tool, "--version"])
        match = re.search(r"version\s+(\d+)", version_output, re.IGNORECASE)
        if not ok or not match or int(match.group(1)) != major:
            version_errors.append(f"{base} is not matching-major version {major}: {tool}")
    if version_errors:
        message = "reduced MLIR grading coverage: " + "; ".join(version_errors)
        print(message, file=sys.stderr)
        return 1 if args.require_tools else 0

    counts = {tier: 0 for tier in range(5)}
    expected_failures = 0
    # Assertions executed against text this run generated. `--require-tools` owns the
    # rail, so it has to prove the rail RAN: finding mlir-opt on PATH and matching its
    # major only establishes that a conversion *could* have been graded.
    generated_assertions = 0
    with tempfile.TemporaryDirectory(prefix="bcir-mlir-grade-") as temp_name:
        temp = Path(temp_name)
        for index, entry in enumerate(registry["examples"]):
            source_rel = entry["source"]
            source = root / source_rel
            tier = entry["expected_tier"]
            counts[tier] += 1
            print(f"[Tier {tier}] {source_rel}")
            source_text = source.read_text(encoding="utf-8")
            checks = entry.get("checks", {})
            require_strings(source_text, checks.get("require_source", []), source_rel, errors)

            artifact_text = ""
            artifact = entry["expected_lowered_artifact"]
            if artifact:
                artifact_path = root / artifact
                artifact_text = artifact_path.read_text(encoding="utf-8")
                require_strings(
                    artifact_text, checks.get("required_artifact_metadata", []), artifact, errors
                )
                require_strings(
                    artifact_text,
                    checks.get("required_artifact_runtime_calls", []),
                    artifact,
                    errors,
                )
                ok, output = run(
                    [tools["llvm-as"], str(artifact_path), "-o", str(temp / f"artifact-{index}.bc")]
                )
                if not ok:
                    errors.append(f"{artifact}: llvm-as failed\n{output}")
                elif not run(
                    [
                        tools["opt"],
                        "-passes=verify",
                        "-disable-output",
                        str(temp / f"artifact-{index}.bc"),
                    ]
                )[0]:
                    errors.append(f"{artifact}: opt -passes=verify failed")

            if tier == 0:
                if not source_text.strip():
                    errors.append(f"{source_rel}: Tier 0 file is empty")
                continue

            allow_unregistered = tier == 1 or "expected_failure" in entry
            parse_command = [tools["mlir-opt"]]
            if allow_unregistered:
                parse_command.append("--allow-unregistered-dialect")
            parse_command += [str(source), "-o", os.devnull]
            ok, output = run(parse_command)
            if not ok:
                errors.append(f"{source_rel}: Tier {tier} parse/verifier failed\n{output}")
                continue
            if tier < 3:
                print("  syntax only; no lowering correctness claimed")
                continue

            lowered = temp / f"lowered-{index}.mlir"
            pipeline_command = [tools["mlir-opt"]]
            if allow_unregistered:
                pipeline_command.append("--allow-unregistered-dialect")
            pipeline_command += [f"--pass-pipeline={entry['pass_pipeline']}", str(source)]
            ok, output = run(pipeline_command, output=lowered)
            if not ok:
                errors.append(f"{source_rel}: declared conversion pipeline failed\n{output}")
                continue
            lowered_text = lowered.read_text(encoding="utf-8")
            before = len(errors)
            generated_assertions += require_strings(
                lowered_text, checks.get("require_lowered", []), source_rel, errors
            )
            generated_assertions += forbid_strings(
                lowered_text, checks.get("forbid_lowered", []), source_rel, errors
            )

            expected_failure = entry.get("expected_failure")
            if expected_failure:
                operations = expected_failure.get("operations", [])
                observed = [operation for operation in operations if operation in lowered_text]
                generated_assertions += len(operations)
                if expected_failure.get("kind") == "illegal-ops-remaining" and observed:
                    del errors[before:]
                    expected_failures += 1
                    print(
                        f"  expected failure observed: illegal operation(s) remain: {', '.join(observed)}"
                    )
                else:
                    errors.append(
                        f"{source_rel}: expected illegal-operation failure was not observed"
                    )
                continue

            if tier == 3:
                continue

            llvm_ir = temp / f"translated-{index}.ll"
            ok, output = run(
                [tools["mlir-translate"], "--mlir-to-llvmir", str(lowered)], output=llvm_ir
            )
            if not ok:
                errors.append(f"{source_rel}: mlir-translate --mlir-to-llvmir failed\n{output}")
                continue
            llvm_text = llvm_ir.read_text(encoding="utf-8")
            generated_assertions += require_strings(
                llvm_text, checks.get("require_llvm_ir", []), source_rel, errors
            )
            generated_assertions += require_strings(
                llvm_text, checks.get("required_runtime_calls", []), source_rel, errors
            )
            bitcode = temp / f"translated-{index}.bc"
            ok, output = run([tools["llvm-as"], str(llvm_ir), "-o", str(bitcode)])
            if not ok:
                errors.append(f"{source_rel}: translated LLVM IR did not assemble\n{output}")
            else:
                ok, output = run([tools["opt"], "-passes=verify", "-disable-output", str(bitcode)])
                if not ok:
                    errors.append(
                        f"{source_rel}: translated LLVM IR failed opt -passes=verify\n{output}"
                    )

    if args.require_tools and generated_assertions == 0:
        errors.append(
            "no assertion ran against generated output: every conversion claim in the "
            "registry is empty, absent or misspelt, so the tier census below would read "
            "the same over a pipeline that emitted nothing"
        )

    if errors:
        print("MLIR tier grading failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(
        "MLIR tier grading passed: " + ", ".join(f"Tier {tier}={counts[tier]}" for tier in range(5))
    )
    print(f"Expected conversion failures demonstrated: {expected_failures}")
    print(f"Assertions executed against generated output: {generated_assertions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
