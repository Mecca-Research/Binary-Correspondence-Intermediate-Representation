#!/usr/bin/env python3
"""Generate a small, deterministic, oracle-checked LLVM exercise variant set."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

GENERATOR_VERSION = "1.0.0"
RECORD_SCHEMA_VERSION = "1.0.0"
DEFAULT_BUDGET = 5
SUPPORTED_PARAMETER_KINDS = (
    "integer_width",
    "constant_values",
    "array_length",
    "struct_field_position",
    "branch_shape",
    "legal_alignment",
    "metadata_payload_values",
    "register_claim_identifiers",
)
TYPED_POINTER_RE = re.compile(
    r"(?:\bi\d+|\bhalf|\bfloat|\bdouble|%[-A-Za-z$._0-9]+|\[[^\]\n]+\]|\{[^}\n]+\})\s*\*"
)
TARGET_ASSUMPTION_RE = re.compile(
    r'^\s*(?:target\s+(?:triple|datalayout)|attributes\s+#\d+\s*=.*\btarget-(?:cpu|features)\b)',
    re.MULTILINE,
)


@dataclasses.dataclass(frozen=True)
class VariantParameters:
    """One validated parameter object shared by prompt, solution, and manifest."""

    family: str
    values: dict[str, Any]
    polarity: str = "positive"
    expected_classification: str = "valid"
    expected_diagnostic: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "values": dict(sorted(self.values.items())),
            "polarity": self.polarity,
            "expected_classification": self.expected_classification,
            "expected_diagnostic": self.expected_diagnostic,
        }


@dataclasses.dataclass(frozen=True)
class Family:
    name: str
    parent_id: str
    parameter_kinds: tuple[str, ...]
    sample: Callable[[random.Random], VariantParameters]
    prompt: Callable[[VariantParameters], str]
    solution: Callable[[VariantParameters], str]
    manifest: Callable[[VariantParameters], dict[str, Any]]
    triviality_reason: Callable[[VariantParameters], str | None]
    parent_semantic_key: str


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seed_for_family(seed: int, family: str, attempt: int) -> int:
    digest = hashlib.sha256(f"{GENERATOR_VERSION}:{seed}:{family}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def semantic_key(parameters: VariantParameters) -> str:
    return sha256_text(stable_json(parameters.as_dict()))


def scalar_sample(rng: random.Random) -> VariantParameters:
    width = rng.choice((8, 16, 32, 64))
    limit = min((1 << (width - 2)) - 1, 1000)
    lhs = rng.randint(-limit, limit)
    rhs = rng.randint(-limit, limit)
    return VariantParameters("scalar-add", {"integer_width": width, "lhs_constant": lhs, "rhs_constant": rhs})


def scalar_prompt(p: VariantParameters) -> str:
    v = p.values
    width, lhs, rhs = v["integer_width"], v["lhs_constant"], v["rhs_constant"]
    return f"""# Generated exercise: width-specific integer addition

Write a standalone opaque-pointer LLVM IR module defining:

```llvm
define i{width} @variant_add(i{width} %x)
```

Return `%x + {lhs} + {rhs}` using two explicit `add i{width}` instructions. Do not
pre-fold the constants and do not add target triples or data layouts.
"""


def scalar_solution(p: VariantParameters) -> str:
    v = p.values
    width, lhs, rhs = v["integer_width"], v["lhs_constant"], v["rhs_constant"]
    return f"""define i{width} @variant_add(i{width} %x) {{
entry:
  %first = add i{width} %x, {lhs}
  %result = add i{width} %first, {rhs}
  ret i{width} %result
}}
"""


def scalar_manifest(p: VariantParameters) -> dict[str, Any]:
    v = p.values
    width, lhs, rhs = v["integer_width"], v["lhs_constant"], v["rhs_constant"]
    expected = lhs + rhs + 3
    return llvm_manifest(
        p,
        structural=[
            assertion("structure.function", "function", rf"define\s+i{width}\s+@variant_add\s*\(\s*i{width}"),
            count_assertion("structure.two-adds", rf"\badd\s+i{width}\b", 2, 2),
        ],
        semantic=[{"function": "variant_add", "return_type": f"i{width}", "args": [{"type": f"i{width}", "value": "3"}], "expected": str(expected)}],
    )


def scalar_trivial(p: VariantParameters) -> str | None:
    v = p.values
    if v["lhs_constant"] == 0 or v["rhs_constant"] == 0:
        return "zero operand removes one required transformation"
    if v["lhs_constant"] + v["rhs_constant"] == 0:
        return "constants cancel and make the task semantically trivial"
    return None


def aggregate_sample(rng: random.Random) -> VariantParameters:
    length = rng.choice((3, 4, 5, 6, 7, 8))
    fields = rng.choice((3, 4, 5))
    position = rng.randrange(fields)
    alignment = rng.choice((1, 2, 4))
    value = rng.randint(11, 200)
    return VariantParameters(
        "aggregate-access",
        {"array_length": length, "struct_field_count": fields, "struct_field_position": position, "legal_alignment": alignment, "stored_value": value},
    )


def aggregate_prompt(p: VariantParameters) -> str:
    v = p.values
    fields = ", ".join("i32" for _ in range(v["struct_field_count"]))
    return f"""# Generated exercise: array-of-struct field access

Define `%Record = type {{ {fields} }}` and a no-argument function
`define i32 @variant_aggregate()` that allocates `[{v['array_length']} x %Record]`,
addresses array element `{v['array_length'] - 1}` and struct field
`{v['struct_field_position']}`, stores then reloads `i32 {v['stored_value']}`, and
returns it. Both memory operations must use legal alignment `{v['legal_alignment']}`.
Use opaque pointers and explicit array and struct GEP indices.
"""


def aggregate_solution(p: VariantParameters) -> str:
    v = p.values
    fields = ", ".join("i32" for _ in range(v["struct_field_count"]))
    return f"""%Record = type {{ {fields} }}

define i32 @variant_aggregate() {{
entry:
  %records = alloca [{v['array_length']} x %Record], align 4
  %field = getelementptr inbounds [{v['array_length']} x %Record], ptr %records, i32 0, i32 {v['array_length'] - 1}, i32 {v['struct_field_position']}
  store i32 {v['stored_value']}, ptr %field, align {v['legal_alignment']}
  %value = load i32, ptr %field, align {v['legal_alignment']}
  ret i32 %value
}}
"""


def aggregate_manifest(p: VariantParameters) -> dict[str, Any]:
    v = p.values
    return llvm_manifest(
        p,
        structural=[
            assertion("structure.array", "contains", rf"\[{v['array_length']}\s+x\s+%Record\]"),
            assertion("structure.gep", "instruction", rf"getelementptr\s+inbounds\s+\[{v['array_length']}\s+x\s+%Record\].*i32\s+{v['struct_field_position']}"),
            assertion("structure.alignment", "contains", rf"(?:load|store)\s+i32.*align\s+{v['legal_alignment']}"),
        ],
        semantic=[{"function": "variant_aggregate", "return_type": "i32", "args": [], "expected": str(v["stored_value"])}],
    )


def aggregate_trivial(p: VariantParameters) -> str | None:
    v = p.values
    if v["array_length"] < 3:
        return "array is too short to exercise indexing"
    if not 0 <= v["struct_field_position"] < v["struct_field_count"]:
        return "struct field position is out of range"
    if v["legal_alignment"] not in (1, 2, 4):
        return "alignment is not legal for the generated i32 access"
    return None


def branch_sample(rng: random.Random) -> VariantParameters:
    shape = rng.choice(("diamond", "inverted-diamond", "early-return"))
    threshold = rng.randint(-9, 9)
    true_value = rng.randint(10, 50)
    false_value = rng.randint(-50, -10)
    return VariantParameters("branch-shape", {"branch_shape": shape, "threshold": threshold, "true_value": true_value, "false_value": false_value})


def branch_prompt(p: VariantParameters) -> str:
    v = p.values
    return f"""# Generated exercise: controlled branch shape

Define `i32 @variant_branch(i32 %x)`. Compare `%x` signed-greater-than
`{v['threshold']}` and return `{v['true_value']}` when true or `{v['false_value']}`
when false. Realize the control flow as a **{v['branch_shape']}**. A diamond must
merge with a `phi`; an inverted diamond reverses branch destinations and the
predicate while preserving behavior; an early-return shape returns directly
from both successors. Do not replace the branch with `select`.
"""


def branch_solution(p: VariantParameters) -> str:
    v = p.values
    t, yes, no, shape = v["threshold"], v["true_value"], v["false_value"], v["branch_shape"]
    if shape == "early-return":
        body = f"""  %condition = icmp sgt i32 %x, {t}
  br i1 %condition, label %yes, label %no

yes:
  ret i32 {yes}

no:
  ret i32 {no}"""
    elif shape == "inverted-diamond":
        body = f"""  %condition = icmp sle i32 %x, {t}
  br i1 %condition, label %no, label %yes

yes:
  br label %merge

no:
  br label %merge

merge:
  %result = phi i32 [ {yes}, %yes ], [ {no}, %no ]
  ret i32 %result"""
    else:
        body = f"""  %condition = icmp sgt i32 %x, {t}
  br i1 %condition, label %yes, label %no

yes:
  br label %merge

no:
  br label %merge

merge:
  %result = phi i32 [ {yes}, %yes ], [ {no}, %no ]
  ret i32 %result"""
    return f"define i32 @variant_branch(i32 %x) {{\nentry:\n{body}\n}}\n"


def branch_manifest(p: VariantParameters) -> dict[str, Any]:
    v = p.values
    shape_assertion = (
        count_assertion("structure.direct-returns", r"\bret\s+i32\b", 2, 2)
        if v["branch_shape"] == "early-return"
        else assertion("structure.phi", "instruction", r"\bphi\s+i32\b")
    )
    return llvm_manifest(
        p,
        structural=[
            assertion("structure.conditional-branch", "instruction", r"\bbr\s+i1\b"),
            shape_assertion,
            absent_assertion("structure.no-select", [r"\bselect\s+i1\b"]),
        ],
        semantic=[
            {"function": "variant_branch", "return_type": "i32", "args": [{"type": "i32", "value": str(v["threshold"] + 1)}], "expected": str(v["true_value"])},
            {"function": "variant_branch", "return_type": "i32", "args": [{"type": "i32", "value": str(v["threshold"])}], "expected": str(v["false_value"])},
        ],
    )


def branch_trivial(p: VariantParameters) -> str | None:
    v = p.values
    if v["true_value"] == v["false_value"]:
        return "both branch outcomes are identical"
    if v["branch_shape"] not in {"diamond", "inverted-diamond", "early-return"}:
        return "unsupported branch shape"
    return None


def metadata_sample(rng: random.Random) -> VariantParameters:
    domain = rng.choice(("HBM", "DDR", "L2", "scratch"))
    lane = rng.randint(1, 31)
    locality = rng.randint(1, 7)
    payload = rng.randint(20, 200)
    return VariantParameters("metadata-payload", {"domain": domain, "lane": lane, "locality": locality, "payload_value": payload})


def metadata_prompt(p: VariantParameters) -> str:
    v = p.values
    return f"""# Generated exercise: deterministic metadata payload

Define `i32 @variant_metadata()` returning `{v['payload_value']}`. Attach
`!bcir.variant !0` to its `ret` instruction. Node `!0` must contain the exact
payload `!{{!\"domain\", !\"{v['domain']}\", !\"lane\", i32 {v['lane']},
!\"locality\", i32 {v['locality']}}}` and must also be reachable from named
metadata `!bcir.variant.catalog`. The metadata is descriptive and must not alter
runtime behavior.
"""


def metadata_solution(p: VariantParameters) -> str:
    v = p.values
    return f"""define i32 @variant_metadata() {{
entry:
  ret i32 {v['payload_value']}, !bcir.variant !0
}}

!bcir.variant.catalog = !{{!0}}
!0 = !{{!"domain", !"{v['domain']}", !"lane", i32 {v['lane']}, !"locality", i32 {v['locality']}}}
"""


def metadata_manifest(p: VariantParameters) -> dict[str, Any]:
    v = p.values
    return llvm_manifest(
        p,
        structural=[
            assertion("structure.attachment", "metadata", r"!bcir\.variant\s+!\d+"),
            assertion("structure.named-metadata", "metadata", r"!bcir\.variant\.catalog\s*="),
            assertion("structure.payload", "metadata", rf'!"{re.escape(v["domain"])}".*i32\s+{v["lane"]}.*i32\s+{v["locality"]}'),
        ],
        semantic=[{"function": "variant_metadata", "return_type": "i32", "args": [], "expected": str(v["payload_value"])}],
    )


def metadata_trivial(p: VariantParameters) -> str | None:
    v = p.values
    if v["lane"] <= 0 or v["locality"] <= 0:
        return "zero metadata payload would erase a required distinction"
    return None


def binding_sample(rng: random.Random) -> VariantParameters:
    claim_id = rng.randint(100, 999)
    register_id = rng.randint(1, 63)
    success_value = rng.randint(2, 100)
    return VariantParameters("register-claim-identifiers", {"claim_id": claim_id, "register_id": register_id, "success_value": success_value})


def binding_prompt(p: VariantParameters) -> str:
    v = p.values
    return f"""# Generated exercise: register/claim identifier matching

Define `i32 @variant_binding(i32 %claim, i32 %register)`. Return
`{v['success_value']}` only when `%claim == {v['claim_id']}` **and** `%register ==
{v['register_id']}`; otherwise return zero. Use two `icmp eq`, one `and i1`, and
one `select`. Attach named metadata recording the exact strings
`claim.{v['claim_id']}` and `register.{v['register_id']}`.
"""


def binding_solution(p: VariantParameters) -> str:
    v = p.values
    return f"""define i32 @variant_binding(i32 %claim, i32 %register) {{
entry:
  %claim_ok = icmp eq i32 %claim, {v['claim_id']}
  %register_ok = icmp eq i32 %register, {v['register_id']}
  %matched = and i1 %claim_ok, %register_ok
  %result = select i1 %matched, i32 {v['success_value']}, i32 0
  ret i32 %result
}}

!bcir.variant.bindings = !{{!0}}
!0 = !{{!"claim.{v['claim_id']}", !"register.{v['register_id']}"}}
"""


def binding_manifest(p: VariantParameters) -> dict[str, Any]:
    v = p.values
    return llvm_manifest(
        p,
        structural=[
            count_assertion("structure.two-comparisons", r"\bicmp\s+eq\s+i32\b", 2, 2),
            assertion("structure.and", "instruction", r"\band\s+i1\b"),
            assertion("structure.select", "instruction", r"\bselect\s+i1\b"),
            assertion("structure.identifiers", "metadata", rf'!"claim\.{v["claim_id"]}".*!"register\.{v["register_id"]}"'),
        ],
        semantic=[
            {"function": "variant_binding", "return_type": "i32", "args": [{"type": "i32", "value": str(v["claim_id"])}, {"type": "i32", "value": str(v["register_id"])}], "expected": str(v["success_value"])},
            {"function": "variant_binding", "return_type": "i32", "args": [{"type": "i32", "value": str(v["claim_id"] + 1)}, {"type": "i32", "value": str(v["register_id"])}], "expected": "0"},
        ],
    )


def binding_trivial(p: VariantParameters) -> str | None:
    v = p.values
    if v["claim_id"] == v["register_id"]:
        return "claim and register identifiers collapse to one value"
    if v["success_value"] == 0:
        return "success and failure outcomes are identical"
    return None


def assertion(identifier: str, kind: str, pattern: str) -> dict[str, Any]:
    return {"id": identifier, "dimension": "structure", "type": kind, "pattern": pattern}


def count_assertion(identifier: str, pattern: str, minimum: int, maximum: int) -> dict[str, Any]:
    return {"id": identifier, "dimension": "structure", "type": "count", "pattern": pattern, "min": minimum, "max": maximum}


def absent_assertion(identifier: str, patterns: list[str]) -> dict[str, Any]:
    return {"id": identifier, "dimension": "structure", "type": "absent", "patterns": patterns}


def llvm_manifest(p: VariantParameters, structural: list[dict[str, Any]], semantic: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "000",
        "prompt_path": "generated/prompt.md",
        "reference_solution_path": "generated/reference.solution.ll",
        "answer_kind": "llvm-ir",
        "required_tools": ["llvm-as", "opt", "lli"],
        "timeout_seconds": 8,
        "expected_points": 100,
        "reference_expected_percent": 100,
        "scoring_dimensions": [
            {"id": "artifact", "points": 10},
            {"id": "validity", "points": 25},
            {"id": "normalization", "points": 5},
            {"id": "structure", "points": 40},
            {"id": "semantics", "points": 20},
        ],
        "structural_assertions": structural + [
            absent_assertion("safety.opaque-pointers", [r"(?:\bi\d+|%[-A-Za-z$._0-9]+|\[[^\]\n]+\]|\{[^}\n]+\})\s*\*"])
        ],
        "comparison_strategy": {"type": "structural-and-execution", "normalize_with": "opt -S", "raw_reference_diff": False},
        "semantic_tests": semantic,
        "variant_parameters": p.as_dict(),
    }


FAMILIES = (
    Family("scalar-add", "001", ("integer_width", "constant_values"), scalar_sample, scalar_prompt, scalar_solution, scalar_manifest, scalar_trivial, "parent:001:i32:add-two-inputs"),
    Family("aggregate-access", "006", ("array_length", "struct_field_position", "legal_alignment"), aggregate_sample, aggregate_prompt, aggregate_solution, aggregate_manifest, aggregate_trivial, "parent:006:entry-array-field-1"),
    Family("branch-shape", "002", ("branch_shape", "constant_values"), branch_sample, branch_prompt, branch_solution, branch_manifest, branch_trivial, "parent:002:max-diamond"),
    Family("metadata-payload", "014", ("metadata_payload_values", "constant_values"), metadata_sample, metadata_prompt, metadata_solution, metadata_manifest, metadata_trivial, "parent:014:ham-default"),
    Family("register-claim-identifiers", "011", ("register_claim_identifiers", "constant_values"), binding_sample, binding_prompt, binding_solution, binding_manifest, binding_trivial, "parent:011:first-read-write"),
)


def load_grader(training_root: Path) -> Any:
    path = training_root / "tools" / "grade-exercises.py"
    spec = importlib.util.spec_from_file_location("llvm_training_grade_exercises", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load grader from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def split_assignments(training_root: Path) -> dict[str, dict[str, str]]:
    data = json.loads((training_root / "dataset" / "splits-v1.json").read_text(encoding="utf-8"))
    result = {}
    for item in data["exercises"]:
        result[item["id"].rsplit("-", 1)[-1]] = item
    return result


def normalize_ir_structure(solution: str, opt: str, work: Path) -> tuple[str, str]:
    source = work / "normalize.ll"
    source.write_text(solution, encoding="utf-8")
    completed = subprocess.run([opt, "-S", str(source), "-o", "-"], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise ValueError("normalization failed: " + completed.stderr.strip())
    lines = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("source_filename") or stripped.startswith("ModuleID"):
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    locals_seen: dict[str, str] = {}
    metadata_seen: dict[str, str] = {}

    def local_repl(match: re.Match[str]) -> str:
        token = match.group(0)
        return locals_seen.setdefault(token, f"%v{len(locals_seen)}")

    def metadata_repl(match: re.Match[str]) -> str:
        token = match.group(0)
        return metadata_seen.setdefault(token, f"!m{len(metadata_seen)}")

    text = re.sub(r"%[-A-Za-z$._0-9]+", local_repl, text)
    text = re.sub(r"!\d+", metadata_repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, sha256_text(text)


def preflight_rejection(parameters: VariantParameters, family: Family, prompt: str, solution: str) -> str | None:
    if parameters.polarity == "negative" and not (parameters.expected_diagnostic or parameters.expected_classification != "valid"):
        return "negative variant lacks a deterministic diagnostic or semantic classification"
    if parameters.polarity not in {"positive", "negative"}:
        return "unknown polarity"
    if TYPED_POINTER_RE.search(solution):
        return "solution violates the opaque-pointer policy"
    if TARGET_ASSUMPTION_RE.search(solution) or re.search(r"\b(?:x86|aarch64|amdgpu|nvvm|wasm)\.", solution, re.IGNORECASE):
        return "solution introduces an unsupported target assumption"
    if "solution" in prompt.casefold() or "reference answer" in prompt.casefold():
        return "prompt leaks solution-oriented wording"
    return family.triviality_reason(parameters)


def artifact_hashes(prompt: str, solution: str, manifest: dict[str, Any], normalized_hash: str) -> dict[str, str]:
    return {
        "prompt_sha256": sha256_text(prompt),
        "reference_solution_sha256": sha256_text(solution),
        "grading_manifest_sha256": sha256_text(stable_json(manifest)),
        "normalized_ir_structure_sha256": normalized_hash,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="fixed integer seed; recorded in every generated record")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help=f"maximum accepted records (default: {DEFAULT_BUDGET})")
    parser.add_argument("--max-attempts-per-family", type=int, default=24)
    parser.add_argument("--output", type=Path, help="write accepted JSON Lines records here (default: stdout)")
    parser.add_argument("--report", type=Path, help="write deterministic acceptance/rejection summary JSON here")
    parser.add_argument("--existing", action="append", type=Path, default=[], help="existing variant JSONL used to seed semantic/text/structure deduplication (repeatable)")
    parser.add_argument("--family", action="append", choices=[item.name for item in FAMILIES], help="limit generation to selected family (repeatable)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")
    if args.budget > DEFAULT_BUDGET:
        raise SystemExit(f"--budget exceeds the reviewed maximum of {DEFAULT_BUDGET}")
    if args.max_attempts_per_family <= 0:
        raise SystemExit("--max-attempts-per-family must be positive")

    training_root = Path(__file__).resolve().parents[1]
    grader = load_grader(training_root)
    split_by_parent = split_assignments(training_root)
    tools = {name: grader.find_tool(name) for name in ("llvm-as", "opt", "lli")}
    missing = [name for name, path in tools.items() if not path]
    if missing:
        raise SystemExit("required variant oracle tool(s) not found: " + ", ".join(missing))

    selected = [item for item in FAMILIES if not args.family or item.name in args.family]
    if not selected:
        raise SystemExit("no variant families selected")

    accepted: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    semantic_keys = {item.parent_semantic_key for item in selected}
    normalized_hashes: set[str] = set()
    text_hashes: set[str] = set()
    existing_records = 0
    for existing_path in args.existing:
        try:
            lines = existing_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise SystemExit(f"cannot read --existing {existing_path}: {error}") from error
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                semantic_keys.add(record["semantic_family"]["equivalence_key"])
                text_hashes.add(record["artifacts"]["reference_solution_sha256"])
                normalized_hashes.add(record["artifacts"]["normalized_ir_structure_sha256"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise SystemExit(f"invalid existing variant record {existing_path}:{line_number}: {error}") from error
            existing_records += 1

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    with tempfile.TemporaryDirectory(prefix="llvm-training-variants-") as temp_name:
        temp = Path(temp_name)
        for family in selected:
            if len(accepted) >= args.budget:
                break
            assignment = split_by_parent.get(family.parent_id)
            if assignment is None:
                raise SystemExit(f"parent exercise {family.parent_id} lacks a split assignment")
            for attempt in range(args.max_attempts_per_family):
                rng = random.Random(seed_for_family(args.seed, family.name, attempt))
                parameters = family.sample(rng)
                prompt = family.prompt(parameters)
                solution = family.solution(parameters)
                manifest = family.manifest(parameters)
                reason = preflight_rejection(parameters, family, prompt, solution)
                if reason:
                    reject(reason)
                    continue
                key = semantic_key(parameters)
                if key in semantic_keys:
                    reject("semantic-equivalence group already exists")
                    continue
                solution_hash = sha256_text(solution)
                if solution_hash in text_hashes:
                    reject("reference solution text hash already exists")
                    continue
                variant_dir = temp / family.name / str(attempt)
                variant_dir.mkdir(parents=True)
                answer = variant_dir / "reference.solution.ll"
                answer.write_text(solution, encoding="utf-8")
                result = grader.grade_entry(manifest, answer, tools)
                failed = [check["id"] for check in result["checks"] if check["status"] != "pass"]
                if failed:
                    reject("grading manifest rejected reference: " + ",".join(failed))
                    continue
                try:
                    normalized_structure, normalized_hash = normalize_ir_structure(solution, tools["opt"], variant_dir)
                except ValueError as error:
                    reject(str(error))
                    continue
                if normalized_hash in normalized_hashes:
                    reject("normalized IR structure already exists")
                    continue

                variant_id = f"llvm-training-exercise-{family.parent_id}-variant-{args.seed}-{len(accepted) + 1:02d}"
                hashes = artifact_hashes(prompt, solution, manifest, normalized_hash)
                record = {
                    "record_schema_version": RECORD_SCHEMA_VERSION,
                    "id": variant_id,
                    "parent_exercise_id": f"llvm-training-exercise-{family.parent_id}",
                    "generator": {"name": "generate-exercise-variants.py", "version": GENERATOR_VERSION, "seed": args.seed},
                    "parameters": parameters.as_dict(),
                    "supported_parameter_kinds": list(family.parameter_kinds),
                    "prompt": prompt,
                    "reference_solution": solution,
                    "grading_manifest": manifest,
                    "oracle_result": {
                        "score_percent": result["score_percent"],
                        "executed_score_percent": result["executed_score_percent"],
                        "checks": [{"id": check["id"], "status": check["status"]} for check in result["checks"]],
                    },
                    "semantic_family": {
                        "name": family.name,
                        "equivalence_key": key,
                        "leakage_group": assignment["leakage_group"],
                        "concept_family": assignment["concept_family"],
                        "split": assignment["split"],
                    },
                    "artifacts": hashes,
                    "normalized_ir_structure": normalized_structure,
                    "review": {"status": "generated-unreviewed", "human_review_required": True, "markdown_solution_generated": False},
                }
                accepted.append(record)
                semantic_keys.add(key)
                normalized_hashes.add(normalized_hash)
                text_hashes.add(solution_hash)
                break

    report = {
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "requested_budget": args.budget,
        "reviewed_budget_policy": DEFAULT_BUDGET,
        "attempted": len(accepted) + sum(rejection_counts.values()),
        "accepted": len(accepted),
        "rejected": sum(rejection_counts.values()),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "existing_records": existing_records,
        "families": [item.name for item in selected],
        "supported_parameter_kinds": list(SUPPORTED_PARAMETER_KINDS),
    }
    output = "".join(stable_json(record) + "\n" for record in accepted)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text, encoding="utf-8")
    else:
        print(report_text, file=sys.stderr, end="")
    if len(accepted) < min(args.budget, len(selected)):
        print("variant generation did not fill the available family budget", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
