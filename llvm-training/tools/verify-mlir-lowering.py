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

TOOL_BASES = ("mlir-opt", "mlir-translate", "llvm-as", "opt")


def find_tool(base: str, major: int) -> str | None:
    suffix = os.environ.get("LLVM_SUFFIX", "")
    candidates = ([f"{base}{suffix}"] if suffix else []) + [f"{base}-{major}", base]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def run(command: list[str], *, output: Path | None = None) -> tuple[bool, str]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if output is not None and completed.returncode == 0:
        output.write_text(completed.stdout, encoding="utf-8")
    return completed.returncode == 0, completed.stdout


def require_strings(text: str, needles: list[str], label: str, errors: list[str]) -> None:
    for needle in needles:
        if needle not in text:
            errors.append(f"{label}: required text is missing: {needle!r}")


def forbid_strings(text: str, needles: list[str], label: str, errors: list[str]) -> None:
    for needle in needles:
        if needle in text:
            errors.append(f"{label}: illegal/unconverted text remains: {needle!r}")


def validate_registry(registry: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    examples = registry.get("examples")
    if registry.get("schema_version") != 1 or not isinstance(examples, list):
        return ["registry must have schema_version 1 and an examples array"]
    seen: set[str] = set()
    registered_sources = {
        path.relative_to(root).as_posix()
        for path in root.glob("llvm-training/**/examples/*.mlir")
    }
    registered_sources.add("llvm-training/autograder/fixtures/mlir/incomplete-bcir-conversion.mlir")
    for index, entry in enumerate(examples):
        label = f"examples[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: entry must be an object")
            continue
        required = {"source", "required_dialects", "required_plugins_tools", "pass_pipeline", "expected_tier", "expected_lowered_artifact"}
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
            errors.append(f"{label}: Tier 1 entries must be explicitly classified as unregistered sketches")
        artifact = entry["expected_lowered_artifact"]
        if artifact is not None and (not isinstance(artifact, str) or not (root / artifact).is_file()):
            errors.append(f"{label}: expected lowered artifact does not exist: {artifact}")
        tools = entry["required_plugins_tools"]
        if not isinstance(tools, list) or any(tool not in TOOL_BASES for tool in tools):
            errors.append(f"{label}: unsupported required tool/plugin declaration")
        if tier == 4 and not {"mlir-opt", "mlir-translate", "llvm-as", "opt"}.issubset(set(tools)):
            errors.append(f"{label}: Tier 4 requires mlir-opt, mlir-translate, llvm-as, and opt")
    missing_sources = registered_sources - seen
    extra_sources = seen - registered_sources
    for source in sorted(missing_sources):
        errors.append(f"unregistered MLIR example: {source}")
    for source in sorted(extra_sources):
        errors.append(f"registry source is outside the governed MLIR example set: {source}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=root / "llvm-training/autograder/mlir-examples.json")
    parser.add_argument("--require-tools", action="store_true", help="fail instead of reporting reduced coverage when declared tools are absent")
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    errors = validate_registry(registry, root)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    major = int(registry["toolchain_major"])
    tools = {base: find_tool(base, major) for base in TOOL_BASES}
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
                require_strings(artifact_text, checks.get("required_artifact_metadata", []), artifact, errors)
                require_strings(artifact_text, checks.get("required_artifact_runtime_calls", []), artifact, errors)
                ok, output = run([tools["llvm-as"], str(artifact_path), "-o", str(temp / f"artifact-{index}.bc")])
                if not ok:
                    errors.append(f"{artifact}: llvm-as failed\n{output}")
                elif not run([tools["opt"], "-passes=verify", "-disable-output", str(temp / f"artifact-{index}.bc")])[0]:
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
            require_strings(lowered_text, checks.get("require_lowered", []), source_rel, errors)
            forbid_strings(lowered_text, checks.get("forbid_lowered", []), source_rel, errors)

            expected_failure = entry.get("expected_failure")
            if expected_failure:
                operations = expected_failure.get("operations", [])
                observed = [operation for operation in operations if operation in lowered_text]
                if expected_failure.get("kind") == "illegal-ops-remaining" and observed:
                    del errors[before:]
                    expected_failures += 1
                    print(f"  expected failure observed: illegal operation(s) remain: {', '.join(observed)}")
                else:
                    errors.append(f"{source_rel}: expected illegal-operation failure was not observed")
                continue

            if tier == 3:
                continue

            llvm_ir = temp / f"translated-{index}.ll"
            ok, output = run([tools["mlir-translate"], "--mlir-to-llvmir", str(lowered)], output=llvm_ir)
            if not ok:
                errors.append(f"{source_rel}: mlir-translate --mlir-to-llvmir failed\n{output}")
                continue
            llvm_text = llvm_ir.read_text(encoding="utf-8")
            require_strings(llvm_text, checks.get("require_llvm_ir", []), source_rel, errors)
            require_strings(llvm_text, checks.get("required_runtime_calls", []), source_rel, errors)
            bitcode = temp / f"translated-{index}.bc"
            ok, output = run([tools["llvm-as"], str(llvm_ir), "-o", str(bitcode)])
            if not ok:
                errors.append(f"{source_rel}: translated LLVM IR did not assemble\n{output}")
            else:
                ok, output = run([tools["opt"], "-passes=verify", "-disable-output", str(bitcode)])
                if not ok:
                    errors.append(f"{source_rel}: translated LLVM IR failed opt -passes=verify\n{output}")

    if errors:
        print("MLIR tier grading failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("MLIR tier grading passed: " + ", ".join(f"Tier {tier}={counts[tier]}" for tier in range(5)))
    print(f"Expected conversion failures demonstrated: {expected_failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
