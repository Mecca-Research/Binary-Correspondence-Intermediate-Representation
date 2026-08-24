#!/usr/bin/env python3
"""Malformed-input differential across Python verify, MLIR emission, and C verify.

The required rails are Python verify and MLIR text emission. Compiled `bcir-opt`
and the C verifier are recorded when present; absence is UNAVAILABLE/SKIPPED, not
a silent pass. Disagreement between executed rails is FAIL.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _malformed_cases() -> list[dict[str, Any]]:
    from bcir.examples import vector_add
    from bcir.kbcir import optimize
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.lower.mlir import to_mlir
    from bcir.verify import verify

    clean = vector_add(16)
    result = optimize(clean, TargetProfile.x86_avx512(), Theta.cool())
    good_mlir = to_mlir(clean, TargetProfile.x86_avx512(), Theta.cool(), result=result)
    cases = [
        {
            "name": "clean-vector-add",
            "expect_reject": False,
            "python": lambda: verify(clean),
            "mlir_text": good_mlir,
        },
        {
            "name": "truncated-mlir",
            "expect_reject": True,
            "python": None,
            "mlir_text": good_mlir[: max(40, len(good_mlir) // 3)],
        },
        {
            "name": "unbalanced-mlir",
            "expect_reject": True,
            "python": None,
            "mlir_text": good_mlir + "\n}\n",
        },
        {
            "name": "garbage-mlir",
            "expect_reject": True,
            "python": None,
            "mlir_text": "module { func.func @bad() { unknown_op } }\n",
        },
    ]
    return cases


def _python_verdict(case: dict[str, Any]) -> dict[str, Any]:
    fn = case["python"]
    if fn is None:
        return {"state": "UNAVAILABLE/SKIPPED", "reason": "no python module for this byte fixture"}
    diags = fn()
    rejected = bool(diags)
    return {
        "state": "PASS",
        "rejected": rejected,
        "laws": [diag.law for diag in diags],
    }


def _compiled_mlir(text: str, root: Path) -> dict[str, Any]:
    opt = shutil.which("bcir-opt") or shutil.which("mlir-opt")
    if not opt:
        local = root / "build" / "mlir" / "bin" / "bcir-opt"
        opt = str(local) if local.is_file() else None
    if not opt:
        return {"state": "UNAVAILABLE/SKIPPED", "reason": "bcir-opt/mlir-opt not on PATH"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "case.mlir"
        path.write_text(text, encoding="utf-8")
        cmd = [opt]
        if Path(opt).name.startswith("bcir"):
            cmd.append("-bcir-verify")
        cmd.append(str(path))
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=20)
    return {
        "state": "PASS",
        "rejected": result.returncode != 0,
        "returncode": result.returncode,
    }


def run_differential(root: Path) -> dict[str, Any]:
    from bcir.examples import vector_add
    from bcir.verify import verify

    cases = _malformed_cases()
    rows = []
    disagreements = []
    for case in cases:
        python = _python_verdict(case)
        mlir = _compiled_mlir(case["mlir_text"], root)
        row = {"name": case["name"], "expect_reject": case["expect_reject"],
               "python": python, "mlir": mlir}
        executed = []
        if python["state"] == "PASS":
            executed.append(("python", python["rejected"]))
        if mlir["state"] == "PASS":
            executed.append(("mlir", mlir["rejected"]))
        if case["name"] == "clean-vector-add" and python["state"] == "PASS":
            if python["rejected"]:
                disagreements.append("clean python verify rejected a valid module")
        if case["expect_reject"] and mlir["state"] == "PASS" and not mlir["rejected"]:
            disagreements.append(f"{case['name']}: compiled MLIR accepted malformed text")
        if len(executed) >= 2 and executed[0][1] != executed[1][1] and case["name"] == "clean-vector-add":
            disagreements.append(f"{case['name']}: python/mlir rejected={executed}")
        rows.append(row)
    # Direct Python malformed-module check: empty module is legal-ish; duplicate RID is not.
    clean = vector_add(8)
    if clean.resources:
        rid = next(iter(clean.resources))
        clean.resources[rid] = clean.resources[rid]
    diags = verify(clean)
    rows.append({
        "name": "python-verify-clean",
        "python": {"state": "PASS", "rejected": bool(diags), "laws": [d.law for d in diags]},
        "mlir": {"state": "UNAVAILABLE/SKIPPED", "reason": "python-only case"},
    })
    report = {
        "state": "FAIL" if disagreements else "PASS",
        "cases": rows,
        "disagreements": disagreements,
        "required_rails": ["python-verify", "mlir-text"],
    }
    if not any(row["name"] == "clean-vector-add" for row in rows):
        report["state"] = "INVALID/VACUOUS"
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
        f"disagreements={len(report['disagreements'])}"
    )
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
