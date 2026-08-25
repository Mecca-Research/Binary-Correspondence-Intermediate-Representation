#!/usr/bin/env python3
"""Malformed-input differential that follows existing BCIR law rails.

Required rails: Python ``verify`` and an always-on MLIR text parser.
Compiled ``bcir-opt -bcir-verify`` is used only on official witnesses that
the compiled pass already implements (R1 RID uniqueness, isolated R5).
It is never stock ``mlir-opt``. Absence is UNAVAILABLE/SKIPPED.

This campaign does not add compiled laws. Oracle-only laws such as R1.1
(claim-id uniqueness per Module) stay on the Python rail.
"""
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

ROOT = Path(__file__).resolve().parents[2]


def find_bcir_opt(root: Path) -> str | None:
    env = os.environ.get("BCIR_OPT")
    if env:
        path = Path(env)
        if path.is_file() and path.name != "mlir-opt":
            return str(path)
    which = shutil.which("bcir-opt")
    if which and Path(which).name != "mlir-opt":
        return which
    for candidate in (
        root / "build" / "mlir-build" / "bin" / "bcir-opt",
        root / "build" / "mlir" / "bin" / "bcir-opt",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def parse_mlir_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped or "bcir.module" not in stripped:
        return {"state": "PASS", "rejected": True, "reason": "not-a-bcir-module"}
    depth = 0
    for char in stripped:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return {"state": "PASS", "rejected": True, "reason": "unbalanced"}
    if depth != 0:
        return {"state": "PASS", "rejected": True, "reason": "unbalanced"}
    # IDs are unique PER MODULE (the oracle's seen sets are per Module), so a
    # file holding several independent bcir.module operations must not be
    # rejected for cross-module reuse the conformance oracle accepts.
    module_starts = [m.start() for m in re.finditer(r"bcir\.module\b", stripped)]
    module_starts.append(len(stripped))
    for index in range(len(module_starts) - 1):
        body = stripped[module_starts[index]:module_starts[index + 1]]
        claim_ids = re.findall(r"claim_id\s*=\s*(-?\d+)", body)
        if len(claim_ids) != len(set(claim_ids)):
            return {"state": "PASS", "rejected": True, "reason": "duplicate-claim-id"}
        rids = re.findall(r"\brid\s*=\s*(\d+)", body)
        if len(rids) != len(set(rids)):
            return {"state": "PASS", "rejected": True, "reason": "duplicate-rid"}
        starts = [match.start() for match in re.finditer(r"bcir\.claim\b", body)]
        starts.append(len(body))
        for claim_index in range(len(starts) - 1):
            block = body[starts[claim_index]:starts[claim_index + 1]]
            if re.search(r"lane\s*=\s*#bcir\.lane<a>", block) and re.search(
                r"hazard\s*=\s*#bcir\.hazard<unique>", block
            ):
                return {"state": "PASS", "rejected": True, "reason": "r5-atomic-unique"}
    return {"state": "PASS", "rejected": False, "reason": None}


def _official_r1_mlir() -> str:
    return (
        "bcir.module @r1 {\n"
        "  bcir.registry @RES {\n"
        "    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, "
        "shape = array<i64: 4>, layout = #bcir.layout<soa> }\n"
        "    bcir.resource @B { rid = 10 : i32, domain_kind = #bcir.domain<ram>, "
        "shape = array<i64: 4>, layout = #bcir.layout<soa> }\n"
        "  }\n"
        "}\n"
    )


def _official_r5_mlir() -> str:
    return (
        "bcir.module @r5 {\n"
        "  bcir.registry @RES {\n"
        "    bcir.resource @T { rid = 10 : i32, domain_kind = #bcir.domain<ram>, "
        "shape = array<i64: 64>, layout = #bcir.layout<soa> }\n"
        "  }\n"
        "  bcir.phase @p0 { id = 0 : i32, deps = [] }\n"
        "  bcir.claim @c attributes {\n"
        "    claim_id = 1 : i32, phase = @p0, op = \"atomic.add\", reads = [@T], writes = [@T], "
        "count = 64 : i64, lane = #bcir.lane<a>, stride_class = #bcir.stride_class<random>, "
        "stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, "
        "verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>\n"
        "  } { %i = bcir.index_range 0 to 64 step 1 }\n"
        "}\n"
    )


def _r5_module():
    from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
    module = Module(name="r5")
    module.add_resource(Resource(rid=10, domain=Domain.RAM, shape=(64,)))
    module.add_phase(Phase(phase_id=0, deps=(), claims=[
        Claim(
            id=1, opcode=Opcode.ATOMIC_ADD, lane=Lane.A, stride_class=StrideClass.RANDOM,
            count=64, rd=(10,), wr=(10,), op="atomic.add", domain=Domain.RAM, hazard="unique",
        ),
    ]))
    return module


def _duplicate_claim_module(clean):
    from copy import deepcopy
    module = deepcopy(clean)
    first = module.phases[0].claims[0]
    clone = deepcopy(first)
    clone.id = first.id
    module.phases[0].claims.append(clone)
    return module


def _compiled_mlir(text: str, root: Path) -> dict[str, Any]:
    opt = find_bcir_opt(root)
    if not opt:
        return {"state": "UNAVAILABLE/SKIPPED", "reason": "bcir-opt not found"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "case.mlir"
        path.write_text(text, encoding="utf-8")
        try:
            result = subprocess.run(
                [opt, "-bcir-verify", str(path)],
                capture_output=True, text=True, check=False, timeout=20,
            )
        except subprocess.TimeoutExpired:
            # A hanging compiled verifier is exactly what this campaign must
            # record — a structured FAIL, never an escaping traceback.
            return {
                "state": "FAIL",
                "rejected": False,
                "reason": "compiled verifier timed out after 20s",
            }
        except OSError as exc:
            return {
                "state": "FAIL",
                "rejected": False,
                "reason": f"compiled verifier failed to start: {type(exc).__name__}",
            }
    crash = result.returncode < 0 or result.returncode >= 0xC0000000
    if crash:
        return {
            "state": "FAIL",
            "rejected": False,
            "returncode": result.returncode,
            "reason": f"compiled verifier crashed rc={result.returncode}",
        }
    return {
        "state": "PASS",
        "rejected": result.returncode != 0,
        "returncode": result.returncode,
    }


def run_differential(root: Path) -> dict[str, Any]:
    from bcir.examples import vector_add
    from bcir.kbcir import optimize
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.kbcir.differential import gen_illegal_module
    from bcir.lower.mlir import to_mlir
    from bcir.verify import verify
    import random

    clean = vector_add(16)
    result = optimize(clean, TargetProfile.x86_avx512(), Theta.cool())
    good_mlir = to_mlir(clean, TargetProfile.x86_avx512(), Theta.cool(), result=result)
    duplicated = _duplicate_claim_module(clean)
    r5 = _r5_module()
    illegal, _why = gen_illegal_module(random.Random(0))

    cases = [
        {
            "name": "clean-vector-add",
            "expect_reject": False,
            "python": lambda: verify(clean),
            "mlir_text": good_mlir,
            "compile": True,
        },
        {
            "name": "oracle-r1.1-duplicate-claim",
            "expect_reject": True,
            "python": lambda: verify(duplicated),
            "mlir_text": None,
            "compile": False,
        },
        {
            "name": "compiled-official-r1-duplicate-rid",
            "expect_reject": True,
            "python": None,
            "mlir_text": _official_r1_mlir(),
            "compile": True,
        },
        {
            "name": "paired-official-r5",
            "expect_reject": True,
            "python": lambda: verify(r5),
            "mlir_text": _official_r5_mlir(),
            "compile": True,
        },
        {
            "name": "oracle-illegal-module",
            "expect_reject": True,
            "python": lambda: verify(illegal),
            "mlir_text": None,
            "compile": False,
        },
        {
            "name": "truncated-mlir",
            "expect_reject": True,
            "python": None,
            "mlir_text": good_mlir[: max(40, len(good_mlir) // 3)],
            "compile": False,
        },
    ]

    rows = []
    disagreements: list[str] = []
    malformed_rejected = 0
    for case in cases:
        py_diags = None if case["python"] is None else case["python"]()
        python = (
            {"state": "UNAVAILABLE/SKIPPED", "reason": "no python module"}
            if py_diags is None and case["python"] is None
            else {
                "state": "PASS",
                "rejected": bool(py_diags),
                "laws": [diag.law for diag in (py_diags or ())],
            }
        )
        text = (
            {"state": "UNAVAILABLE/SKIPPED", "reason": "no mlir text"}
            if case["mlir_text"] is None
            else parse_mlir_text(case["mlir_text"])
        )
        compiled = (
            {"state": "UNAVAILABLE/SKIPPED", "reason": "not a compiled-law witness"}
            if not case["compile"] or case["mlir_text"] is None
            else _compiled_mlir(case["mlir_text"], root)
        )
        if compiled["state"] == "FAIL":
            disagreements.append(str(compiled.get("reason") or f"{case['name']}: compiled crash"))
        executed = []
        if python["state"] == "PASS":
            executed.append(("python", python["rejected"]))
        if text["state"] == "PASS":
            executed.append(("mlir_text", text["rejected"]))
        if compiled["state"] == "PASS":
            executed.append(("mlir_compiled", compiled["rejected"]))
        for rail, rejected in executed:
            if rejected != case["expect_reject"]:
                disagreements.append(
                    f"{case['name']}/{rail}: rejected={rejected} expect={case['expect_reject']}"
                )
        if case["expect_reject"] and any(rejected for _, rejected in executed):
            malformed_rejected += 1
        rows.append({
            "name": case["name"],
            "expect_reject": case["expect_reject"],
            "python": python,
            "mlir_text": text,
            "mlir_compiled": compiled,
        })

    report = {
        "state": "FAIL" if disagreements else "PASS",
        "cases": rows,
        "disagreements": disagreements,
        "malformed_rejected": malformed_rejected,
        "required_rails": ["python-verify", "mlir-text"],
        "parity": {
            "R1": "compiled official witness + text",
            "R1.1": "oracle only (compiled law not present; not added by this rail)",
            "R5": "oracle + official compiled witness (RANDOM, legal under R6)",
        },
    }
    if malformed_rejected < 3:
        report["state"] = "INVALID/VACUOUS"
        report["error"] = f"only {malformed_rejected} malformed cases were rejected"
    if disagreements:
        report["state"] = "FAIL"
        report["error"] = "; ".join(disagreements[:6])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_malformed_differential")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if str(args.root) not in sys.path:
        sys.path.insert(0, str(args.root))
    report = run_differential(args.root)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"malformed_differential: {report['state']} cases={len(report['cases'])} "
        f"disagreements={len(report['disagreements'])} "
        f"malformed_rejected={report['malformed_rejected']}"
    )
    if report.get("error"):
        print(report["error"])
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
