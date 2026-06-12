#!/usr/bin/env python3
"""Deterministic, dependency-free grader for llvm-training exercise artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

from safe_process import ProcessResult, run_bounded

SCHEMA_VERSION = "1.0"
SUPPORTED_KINDS = {"llvm-ir", "mlir", "markdown-review", "pass-output", "diagnostic"}
MAX_ANSWER_BYTES = 1024 * 1024
FORBIDDEN_GENERATED_SUFFIXES = {".sh", ".bash", ".so", ".dylib", ".dll", ".o", ".obj", ".a", ".exe"}
FORBIDDEN_GENERATED_NAMES = {"Makefile", "CMakeLists.txt", "build.ninja", "meson.build"}


def confined_answer(root: Path, candidate: Path) -> Path:
    """Resolve an answer while rejecting lexical and symlink escapes."""
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"answer path escapes configured attempt root: {candidate}") from exc
    return resolved


def validate_attempt_tree(root: Path) -> None:
    """Reject generated executable/build artifacts that the baseline must not consume."""
    for path in root.rglob("*"):
        if path.is_symlink():
            confined_answer(root, path)
        if path.is_file() and (path.name in FORBIDDEN_GENERATED_NAMES or path.suffix.lower() in FORBIDDEN_GENERATED_SUFFIXES):
            raise ValueError(f"unsupported generated executable or build artifact: {path}")


@dataclasses.dataclass
class Check:
    id: str
    dimension: str
    status: str
    points_earned: float
    points_available: float
    message: str
    command: list[str] | None = None
    stdout: str = ""
    stderr: str = ""
    outcome: str = "incorrect_answer"

    def as_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["points_earned"] = round(self.points_earned, 3)
        value["points_available"] = round(self.points_available, 3)
        return value


def find_tool(name: str) -> str | None:
    suffix = os.environ.get("LLVM_SUFFIX", "")
    candidates = ([name + suffix] if suffix else []) + [name]
    for version in range(30, 9, -1):
        candidates.append(f"{name}-{version}")
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def run_command(command: list[str], timeout: float, stdin: str | None = None) -> ProcessResult:
    return run_bounded(command, timeout=timeout, stdin=stdin)


def version_for(tool: str | None) -> dict[str, str] | None:
    if not tool:
        return None
    result = run_command([tool, "--version"], 5)
    text = (result.stdout or result.stderr).splitlines()
    return {"path": str(Path(tool).resolve()), "version": text[0].strip() if text else "version unavailable"}


def allocate_points(entry: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, float]:
    dimensions = {item["id"]: float(item["points"]) for item in entry["scoring_dimensions"]}
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec["dimension"]] = counts.get(spec["dimension"], 0) + 1
    return {dim: total / counts[dim] for dim, total in dimensions.items() if counts.get(dim)}


def assertion_specs(entry: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"id": "answer.exists", "dimension": "artifact", "type": "exists"},
        {"id": "answer.non_empty", "dimension": "artifact", "type": "non-empty"},
    ]
    kind = entry["answer_kind"]
    if kind == "llvm-ir":
        specs.extend([
            {"id": "llvm.assemble", "dimension": "validity", "type": "tool", "tool": "llvm-as"},
            {"id": "llvm.verify", "dimension": "validity", "type": "tool", "tool": "opt"},
            {"id": "llvm.normalize", "dimension": "normalization", "type": "normalize", "tool": "opt"},
        ])
    elif kind == "mlir":
        specs.append({"id": "mlir.parse", "dimension": "validity", "type": "tool", "tool": "mlir-opt"})
    for index, assertion in enumerate(entry.get("structural_assertions", []), 1):
        item = dict(assertion)
        item.setdefault("id", f"structure.{index:02d}")
        item.setdefault("dimension", "structure")
        specs.append(item)
    for index, rubric in enumerate(entry.get("rubric_checks", []), 1):
        item = dict(rubric)
        item.setdefault("id", f"rubric.{index:02d}")
        item.setdefault("dimension", "rubric")
        item["type"] = "rubric"
        specs.append(item)
    for index, vector in enumerate(entry.get("semantic_tests", []), 1):
        specs.append({"id": f"semantic.vector.{index:02d}", "dimension": "semantics", "type": "semantic", "vector": vector})
    return specs


def pattern_count(pattern: str, text: str, flags: int = re.MULTILINE | re.IGNORECASE) -> int:
    return len(re.findall(pattern, text, flags))


def structural_check(spec: dict[str, Any], text: str) -> tuple[bool, str]:
    kind = spec["type"]
    patterns = spec.get("patterns", [spec.get("pattern", "")])
    if kind in {"contains", "function", "declaration", "instruction", "metadata", "attribute"}:
        missing = [pattern for pattern in patterns if not re.search(pattern, text, re.MULTILINE | re.IGNORECASE)]
        return (not missing, "all required patterns found" if not missing else "missing: " + ", ".join(missing))
    if kind == "absent":
        found = [pattern for pattern in patterns if re.search(pattern, text, re.MULTILINE | re.IGNORECASE)]
        return (not found, "prohibited patterns absent" if not found else "prohibited match: " + ", ".join(found))
    if kind == "count":
        count = pattern_count(spec["pattern"], text)
        minimum = int(spec.get("min", 0))
        maximum = int(spec.get("max", 2**31 - 1))
        return (minimum <= count <= maximum, f"found {count}; expected {minimum}..{maximum}")
    raise ValueError(f"unsupported structural assertion type: {kind}")


def rubric_check(spec: dict[str, Any], text: str) -> tuple[bool, str]:
    lower = text.casefold()
    mode = spec.get("mode", "all")
    terms = [str(term).casefold() for term in spec.get("terms", [])]
    if mode == "all":
        missing = [term for term in terms if term not in lower]
        return (not missing, "required rubric terms covered" if not missing else "missing rubric terms: " + ", ".join(missing))
    if mode == "any":
        found = [term for term in terms if term in lower]
        return (bool(found), "covered by: " + ", ".join(found) if found else "none of the accepted rubric terms found")
    if mode == "absent":
        found = [term for term in terms if term in lower]
        return (not found, "prohibited claims absent" if not found else "prohibited claims found: " + ", ".join(found))
    if mode == "min-words":
        words = re.findall(r"\b[\w'`.-]+\b", text)
        minimum = int(spec["minimum"])
        return (len(words) >= minimum, f"word count {len(words)}; minimum {minimum}")
    raise ValueError(f"unsupported rubric mode: {mode}")


def semantic_wrapper(vector: dict[str, Any], index: int) -> str:
    return_type = vector.get("return_type", "i32")
    args = vector.get("args", [])
    typed = ", ".join(f"{item['type']} {item['value']}" for item in args)
    expected = vector["expected"]
    predicate = vector.get("predicate", "eq")
    call_name = f"%actual{index}"
    cmp_name = f"%ok{index}"
    return f"  {call_name} = call {return_type} @{vector['function']}({typed})\n  {cmp_name} = icmp {predicate} {return_type} {call_name}, {expected}\n"


def run_semantic(answer: Path, entry: dict[str, Any], vector: dict[str, Any], index: int, lli: str, timeout: float, temp: Path) -> tuple[bool, str, list[str], str, str]:
    source = answer.read_text(encoding="utf-8")
    if re.search(r"\bdefine\s+i32\s+@main\s*\(", source):
        return False, "submission defines @main; semantic harness requires that name", [], "", ""
    wrapper = semantic_wrapper(vector, index)
    module = source + "\n\ndefine i32 @main() {\nentry:\n" + wrapper + f"  %exit = select i1 %ok{index}, i32 0, i32 1\n  ret i32 %exit\n}}\n"
    harness = temp / f"semantic-{index}.ll"
    harness.write_text(module, encoding="utf-8")
    command = [lli, str(harness)]
    result = run_command(command, timeout)
    return result.returncode == 0, ("test vector matched" if result.returncode == 0 else f"lli exited {result.returncode}"), command, result.stdout, result.stderr


def grade_entry(entry: dict[str, Any], answer: Path, tools: dict[str, str | None]) -> dict[str, Any]:
    specs = assertion_specs(entry)
    point_values = allocate_points(entry, specs)
    checks: list[Check] = []
    timeout = float(entry.get("timeout_seconds", 10))
    exists = answer.is_file()
    if exists and answer.stat().st_size > MAX_ANSWER_BYTES:
        exists = False
    text = answer.read_text(encoding="utf-8", errors="replace") if exists else ""
    non_empty = bool(text.strip())
    normalized = text

    with tempfile.TemporaryDirectory(prefix=f"llvm-training-grade-{entry['id']}-") as temp_name:
        temp = Path(temp_name)
        for spec in specs:
            points = point_values.get(spec["dimension"], 0.0)
            status, message, command, stdout, stderr, outcome = "fail", "", None, "", "", "incorrect_answer"
            typ = spec["type"]
            if typ == "exists":
                status, message = ("pass", "answer file exists") if exists else ("fail", "answer file is missing")
                outcome = "pass" if exists else "invalid_answer"
            elif typ == "non-empty":
                status, message = ("pass", "answer file is non-empty") if non_empty else ("fail", "answer file is empty or missing")
                outcome = "pass" if non_empty else "invalid_answer"
            elif not non_empty:
                status, message, outcome = "skip", "answer is unavailable; prerequisite check did not pass", "invalid_answer"
            elif typ == "tool":
                tool = tools.get(spec["tool"])
                if not tool:
                    status, message, outcome = "skip", f"required tool not found: {spec['tool']}", "missing_tool"
                else:
                    if spec["tool"] == "llvm-as":
                        command = [tool, str(answer), "-o", os.devnull]
                    elif spec["tool"] == "opt":
                        command = [tool, "-passes=verify", str(answer), "-o", os.devnull]
                    else:
                        command = [tool, str(answer), "-o", os.devnull]
                    result = run_command(command, timeout)
                    stdout, stderr = result.stdout, result.stderr
                    status = "pass" if result.returncode == 0 else "fail"
                    outcome = "pass" if status == "pass" else ("timeout" if result.timed_out else "invalid_answer")
                    message = "tool accepted answer" if status == "pass" else ("tool timed out" if result.timed_out else f"tool exited {result.returncode}")
            elif typ == "normalize":
                tool = tools.get("opt")
                if not tool:
                    status, message, outcome = "skip", "opt unavailable; structural checks use original text", "missing_tool"
                else:
                    command = [tool, "-S", str(answer), "-o", "-"]
                    result = run_command(command, timeout)
                    stdout, stderr = result.stdout, result.stderr
                    if result.returncode == 0:
                        normalized = result.stdout
                        status, message, outcome = "pass", "opt -S produced normalized textual IR", "pass"
                    else:
                        status, message, outcome = "fail", ("opt -S timed out" if result.timed_out else f"opt -S exited {result.returncode}"), ("timeout" if result.timed_out else "invalid_answer")
            elif typ == "rubric":
                passed, message = rubric_check(spec, text)
                status = "pass" if passed else "fail"
                outcome = "pass" if passed else "incorrect_answer"
            elif typ == "semantic":
                if not entry.get("safe_deterministic_lli", False):
                    status, message, outcome = "fail", "lli execution is not allowlisted for this exercise", "grader_failure"
                    checks.append(Check(spec["id"], spec["dimension"], status, 0.0, points, message, outcome=outcome))
                    continue
                lli = tools.get("lli")
                if not lli:
                    status, message, outcome = "skip", "lli unavailable", "missing_tool"
                else:
                    passed, message, command, stdout, stderr = run_semantic(answer, entry, spec["vector"], int(spec["id"].rsplit(".", 1)[1]), lli, timeout, temp)
                    status = "pass" if passed else "fail"
                    outcome = "pass" if passed else ("timeout" if "timed out" in stderr else "incorrect_answer")
            else:
                passed, message = structural_check(spec, normalized)
                status = "pass" if passed else "fail"
                outcome = "pass" if passed else "incorrect_answer"
            checks.append(Check(spec["id"], spec["dimension"], status, points if status == "pass" else 0.0, points, message, command, stdout, stderr, outcome))

    earned = sum(item.points_earned for item in checks)
    available = sum(item.points_available for item in checks)
    executed_available = sum(item.points_available for item in checks if item.status != "skip")
    executed_earned = sum(item.points_earned for item in checks if item.status != "skip")
    return {
        "exercise_id": entry["id"],
        "answer": str(answer),
        "answer_kind": entry["answer_kind"],
        "prompt_path": entry["prompt_path"],
        "reference_solution_path": entry["reference_solution_path"],
        "comparison_strategy": entry.get("comparison_strategy"),
        "rubric_coverage_only": entry["answer_kind"] in {"markdown-review", "diagnostic"},
        "checks": [item.as_dict() for item in checks],
        "points_earned": round(earned, 3),
        "points_available": round(available, 3),
        "score_percent": round(100 * earned / available, 2) if available else 0.0,
        "executed_score_percent": round(100 * executed_earned / executed_available, 2) if executed_available else 0.0,
        "skipped_checks": sum(item.status == "skip" for item in checks),
    }


def validate_registry(data: dict[str, Any], root: Path) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported registry schema_version {data.get('schema_version')!r}")
    ids: set[str] = set()
    required = {"id", "prompt_path", "reference_solution_path", "answer_kind", "required_tools", "timeout_seconds", "scoring_dimensions", "structural_assertions"}
    for entry in data.get("exercises", []):
        missing = required - entry.keys()
        if missing:
            raise ValueError(f"exercise {entry.get('id', '?')} lacks fields: {sorted(missing)}")
        if not re.fullmatch(r"\d{3}", entry["id"]) or entry["id"] in ids:
            raise ValueError(f"exercise ID must be unique and three digits: {entry['id']!r}")
        ids.add(entry["id"])
        if entry["answer_kind"] not in SUPPORTED_KINDS:
            raise ValueError(f"unsupported answer kind for {entry['id']}: {entry['answer_kind']}")
        for key in ("prompt_path", "reference_solution_path"):
            if not (root / entry[key]).is_file():
                raise ValueError(f"registered {key} does not exist: {entry[key]}")
        points = sum(float(item["points"]) for item in entry["scoring_dimensions"])
        if abs(points - float(entry.get("expected_points", 100))) > 0.001:
            raise ValueError(f"exercise {entry['id']} scoring dimensions total {points}, expected {entry.get('expected_points', 100)}")


def report_text(report: dict[str, Any]) -> str:
    lines = [f"Autograder schema {report['schema_version']}"]
    for result in report["results"]:
        lines.append(f"\nExercise {result['exercise_id']} ({result['answer_kind']}): {result['points_earned']}/{result['points_available']} ({result['score_percent']:.2f}%)")
        if result["rubric_coverage_only"]:
            lines.append("  NOTE: markdown/diagnostic scoring measures deterministic rubric coverage, not semantic proof.")
        for check in result["checks"]:
            marker = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[check["status"]]
            lines.append(f"  [{marker}] {check['id']}: {check['points_earned']}/{check['points_available']} - {check['message']}")
    aggregate = report["aggregate"]
    lines.append(f"\nAggregate: {aggregate['points_earned']}/{aggregate['points_available']} ({aggregate['score_percent']:.2f}%)")
    if report["toolchain"]:
        lines.append("Toolchain:")
        for name, info in report["toolchain"].items():
            lines.append(f"  {name}: {info['version']} ({info['path']})")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, help="directory containing NNN/answer.<ext> attempt directories")
    parser.add_argument("--exercise", action="append", help="grade only this stable three-digit exercise ID (repeatable)")
    parser.add_argument("--answer", type=Path, help="answer file for a single --exercise")
    parser.add_argument("--attempt-root", type=Path, help="configured root that must contain every untrusted answer")
    parser.add_argument("--registry", type=Path, help="override exercises.json")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="write report to this path instead of stdout")
    parser.add_argument("--self-test", action="store_true", help="grade registered checked-in references and fail unless all executed checks pass")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    training_root = Path(__file__).resolve().parents[1]
    repo_root = training_root.parent
    registry_path = args.registry or training_root / "autograder" / "exercises.json"
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_registry(data, repo_root)
    by_id = {item["id"]: item for item in data["exercises"]}

    selected = args.exercise or list(by_id)
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise SystemExit("unknown exercise ID(s): " + ", ".join(unknown))
    if args.answer and (len(selected) != 1 or not args.exercise):
        raise SystemExit("--answer requires exactly one --exercise")
    if args.self_test and (args.attempts or args.answer):
        raise SystemExit("--self-test cannot be combined with --attempts or --answer")
    if not args.self_test and not args.attempts and not args.answer:
        raise SystemExit("provide --attempts, --exercise with --answer, or --self-test")

    tool_names = sorted({tool for exercise_id in selected for tool in by_id[exercise_id]["required_tools"]} | {"opt"})
    tools = {name: find_tool(name) for name in tool_names}
    results = []
    for exercise_id in selected:
        entry = by_id[exercise_id]
        if args.self_test:
            answer = repo_root / entry["reference_solution_path"]
        elif args.answer:
            attempt_root = args.attempt_root or args.answer.parent
            validate_attempt_tree(attempt_root.resolve(strict=True))
            answer = confined_answer(attempt_root, args.answer)
        else:
            attempt_root = args.attempts.resolve(strict=True)
            validate_attempt_tree(attempt_root)
            extension = {"llvm-ir": ".ll", "mlir": ".mlir", "markdown-review": ".md", "pass-output": ".ll", "diagnostic": ".md"}[entry["answer_kind"]]
            answer = confined_answer(attempt_root, attempt_root / exercise_id / ("answer" + extension))
        results.append(grade_entry(entry, answer, tools))

    earned = sum(item["points_earned"] for item in results)
    available = sum(item["points_available"] for item in results)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "registry": str(registry_path),
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "toolchain": {
            "python": {"path": str(Path(sys.executable).resolve()), "version": sys.version.splitlines()[0]},
            **{name: (version_for(path) or {"path": None, "version": "missing"}) for name, path in tools.items()},
        },
        "results": results,
        "aggregate": {"points_earned": round(earned, 3), "points_available": round(available, 3), "score_percent": round(100 * earned / available, 2) if available else 0.0},
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n" if args.format == "json" else report_text(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)

    if args.self_test:
        failed = [item["exercise_id"] for item in results if item["executed_score_percent"] < float(by_id[item["exercise_id"]].get("reference_expected_percent", 100))]
        if failed:
            print("reference self-test failed: " + ", ".join(failed), file=sys.stderr)
            return 1
        return 0
    return 1 if any(any(check["status"] == "fail" for check in item["checks"]) for item in results) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        message = str(exc)
        invalid_markers = ("answer path", "attempt root", "unsupported generated", "answer file")
        outcome = "invalid_answer" if any(marker in message for marker in invalid_markers) else "grader_failure"
        payload = {"schema_version": SCHEMA_VERSION, "error": {"outcome": outcome, "message": message}}
        if "--format" in sys.argv and "json" in sys.argv:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(f"grader failure: {exc}", file=sys.stderr)
        raise SystemExit(2)
