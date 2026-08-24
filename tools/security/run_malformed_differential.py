#!/usr/bin/env python3
"""Malformed-input differential across Python verify and MLIR text.

Python verify and a structural MLIR-text parser always run. Compiled `bcir-opt`
or `mlir-opt` is recorded when present. A campaign that never rejects a
malformed case is INVALID, not clean.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _duplicate_claim_module():
    from bcir.examples import vector_add
    module = vector_add(8)
    if not module.phases or not module.phases[0].claims:
        raise RuntimeError("vector_add produced no claims to duplicate")
    module.phases[0].claims.append(module.phases[0].claims[0])
    return module


def _illegal_module():
    import random
    from bcir.kbcir.differential import gen_illegal_module
    return gen_illegal_module(random.Random(11))[0]


def parse_mlir_text(text: str) -> dict[str, Any]:
    """Always-on MLIR text gate. This is a parser, not a skip."""
    stripped = text.strip()
    if not stripped:
        return {"state": "PASS", "rejected": True, "reason": "empty"}
    if stripped.count("{") != stripped.count("}"):
        return {"state": "PASS", "rejected": True, "reason": "unbalanced-braces"}
    if "unknown_op" in stripped:
        return {"state": "PASS", "rejected": True, "reason": "unknown-op"}
    if "bcir.module" not in stripped and not stripped.startswith("module"):
        return {"state": "PASS", "rejected": True, "reason": "no-module"}
    if not stripped.endswith("}"):
        return {"state": "PASS", "rejected": True, "reason": "truncated"}
    return {"state": "PASS", "rejected": False, "reason": "structurally-well-formed"}


def find_bcir_opt(root: Path) -> str | None:
    env = os.environ.get("BCIR_OPT")
    if env and Path(env).is_file():
        return env
    which = shutil.which("bcir-opt")
    if which:
        return which
    trees = (
        root / "build" / "mlir-build",
        root / "build" / "mlir-build-debug",
        root / "build" / "mlir22",
    )
    for tree in trees:
        if not tree.is_dir():
            continue
        for candidate in tree.rglob("bcir-opt"):
            if candidate.is_file():
                return str(candidate)
        windows = tree / "bin" / "bcir-opt.exe"
        if windows.is_file():
            return str(windows)
    return None


def _compiled_mlir(text: str, root: Path) -> dict[str, Any]:
    opt = find_bcir_opt(root)
    if not opt:
        return {"state": "UNAVAILABLE/SKIPPED", "reason": "bcir-opt not found in PATH or build/mlir-build"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "case.mlir"
        path.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [opt, "-bcir-verify", str(path)],
            capture_output=True, text=True, check=False, timeout=20,
        )
    return {
        "state": "PASS",
        "rejected": result.returncode != 0,
        "returncode": result.returncode,
        "tool": opt,
    }


def _cases() -> list[dict[str, Any]]:
    from bcir.examples import vector_add
    from bcir.kbcir import optimize
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.lower.mlir import to_mlir
    from bcir.verify import verify

    clean = vector_add(16)
    result = optimize(clean, TargetProfile.x86_avx512(), Theta.cool())
    good_mlir = to_mlir(clean, TargetProfile.x86_avx512(), Theta.cool(), result=result)
    illegal = _illegal_module()
    duplicated = _duplicate_claim_module()
    return [
        {
            "name": "clean-vector-add",
            "expect_reject": False,
            "python": lambda: verify(clean),
            "mlir_text": good_mlir,
        },
        {
            "name": "python-duplicate-claim-id",
            "expect_reject": True,
            "python": lambda: verify(duplicated),
            "mlir_text": None,
        },
        {
            "name": "python-illegal-module",
            "expect_reject": True,
            "python": lambda: verify(illegal),
            "mlir_text": None,
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


def run_differential(root: Path) -> dict[str, Any]:
    from bcir.verify import verify  # noqa: F401 - keep import side effects stable

    rows = []
    disagreements = []
    malformed_rejected = 0
    for case in _cases():
        python_fn = case["python"]
        if python_fn is None:
            python = {"state": "UNAVAILABLE/SKIPPED", "reason": "no python module"}
        else:
            diags = python_fn()
            python = {
                "state": "PASS",
                "rejected": bool(diags),
                "laws": [diag.law for diag in diags],
            }
        mlir_text = (
            {"state": "UNAVAILABLE/SKIPPED", "reason": "no mlir text"}
            if case["mlir_text"] is None
            else parse_mlir_text(case["mlir_text"])
        )
        compiled = (
            {"state": "UNAVAILABLE/SKIPPED", "reason": "no mlir text"}
            if case["mlir_text"] is None
            else _compiled_mlir(case["mlir_text"], root)
        )
        executed = []
        if python["state"] == "PASS":
            executed.append(("python", python["rejected"]))
        if mlir_text["state"] == "PASS":
            executed.append(("mlir-text", mlir_text["rejected"]))
        if compiled["state"] == "PASS":
            executed.append(("mlir-compiled", compiled["rejected"]))
        if not executed:
            disagreements.append(f"{case['name']}: no rail executed")
        else:
            mismatched = [name for name, rejected in executed if rejected != case["expect_reject"]]
            if mismatched:
                disagreements.append(
                    f"{case['name']}: rails {mismatched} did not match expect_reject="
                    f"{case['expect_reject']}"
                )
        if case["expect_reject"] and executed and all(rejected for _, rejected in executed):
            malformed_rejected += 1
        rows.append({
            "name": case["name"],
            "expect_reject": case["expect_reject"],
            "python": python,
            "mlir_text": mlir_text,
            "mlir_compiled": compiled,
        })
    report = {
        "state": "FAIL" if disagreements else "PASS",
        "cases": rows,
        "disagreements": disagreements,
        "malformed_rejected": malformed_rejected,
        "required_rails": ["python-verify", "mlir-text-parser"],
    }
    if malformed_rejected < 3:
        report["state"] = "INVALID/VACUOUS"
        report["error"] = f"only {malformed_rejected} malformed cases were rejected"
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
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
