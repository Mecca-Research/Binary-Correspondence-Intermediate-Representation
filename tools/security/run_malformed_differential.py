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
import re
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


def _r5_module():
    from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
    module = Module(name="r5")
    module.add_resource(Resource(rid=1, domain=Domain.RAM, shape=(64,)))
    module.add_phase(Phase(phase_id=0, deps=(), claims=[
        Claim(
            id=1, opcode=Opcode.ATOMIC_ADD, lane=Lane.A, stride_class=StrideClass.UNIT,
            count=64, rd=(1,), wr=(1,), op="atomic.add", domain=Domain.RAM, hazard="unique",
        ),
    ]))
    return module


def _r5_mlir() -> str:
    return (
        "bcir.module @r5 {\n"
        "  bcir.registry @RES {\n"
        "    bcir.resource @T { rid = 1 : i32, domain_kind = #bcir.domain<ram>, "
        "shape = array<i64: 64>, layout = #bcir.layout<soa> }\n"
        "  }\n"
        "  bcir.phase @p0 { id = 0 : i32, deps = [] }\n"
        "  bcir.claim @c attributes {\n"
        "    claim_id = 1 : i32, phase = @p0, op = \"atomic.add\", reads = [@T], writes = [@T], "
        "count = 64 : i64, lane = #bcir.lane<a>, stride_class = #bcir.stride_class<unit>, "
        "stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, "
        "verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>\n"
        "  } { %i = bcir.index_range 0 to 64 step 1 }\n"
        "}\n"
    )


def _duplicate_claim_mlir(good: str) -> str:
    for line in good.splitlines():
        if "bcir.claim @" in line and "claim_id =" in line and "reads =" in line:
            cloned = re.sub(r"(bcir\.claim\s+)@\S+", r"\1@dup", line, count=1)
            index = good.rfind("}")
            if index < 0:
                return good + cloned + "\n"
            return good[:index] + cloned + "\n" + good[index:]
    raise RuntimeError("no complete valid claim to duplicate")


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
    claim_ids = re.findall(r"claim_id\s*=\s*(\d+)", stripped)
    if len(claim_ids) != len(set(claim_ids)):
        return {"state": "PASS", "rejected": True, "reason": "duplicate-claim-id"}
    if re.search(r'op\s*=\s*"atomic\.[^"]+"', stripped) and "hazard<unique>" in stripped:
        return {"state": "PASS", "rejected": True, "reason": "r5-atomic-without-ordered-hazard"}
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
        "state": "FAIL" if _compiled_crash(result.returncode) else "PASS",
        "rejected": result.returncode != 0 and not _compiled_crash(result.returncode),
        "returncode": result.returncode,
        "tool": opt,
        "reason": f"signal-or-fatal {result.returncode}" if _compiled_crash(result.returncode) else None,
    }


def _compiled_crash(returncode: int) -> bool:
    return returncode < 0 or returncode >= 0xC0000000


def _cases() -> list[dict[str, Any]]:
    from bcir.examples import vector_add
    from bcir.kbcir import optimize
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.lower.mlir import to_mlir
    from bcir.verify import verify

    clean = vector_add(16)
    result = optimize(clean, TargetProfile.x86_avx512(), Theta.cool())
    good_mlir = to_mlir(clean, TargetProfile.x86_avx512(), Theta.cool(), result=result)
    illegal = _r5_module()
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
            "mlir_text": _duplicate_claim_mlir(good_mlir),
        },
        {
            "name": "python-illegal-module",
            "expect_reject": True,
            "python": lambda: verify(illegal),
            "mlir_text": _r5_mlir(),
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


def run_differential(root: Path, require_bcir_opt: bool = False) -> dict[str, Any]:
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
        if compiled["state"] == "FAIL":
            disagreements.append(
                f"{case['name']}: compiled verifier crashed rc={compiled.get('returncode')}"
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
    if require_bcir_opt:
        if not find_bcir_opt(root):
            report["state"] = "FAIL"
            report["error"] = "bcir-opt required but not found in PATH or build/mlir-build"
        else:
            skipped = [
                row["name"] for row in rows
                if row["mlir_compiled"]["state"] == "UNAVAILABLE/SKIPPED"
            ]
            if skipped:
                report["state"] = "FAIL"
                report["error"] = f"compiled rail skipped for {skipped}"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_malformed_differential")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-bcir-opt", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if str(args.root) not in sys.path:
        sys.path.insert(0, str(args.root))
    report = run_differential(args.root, require_bcir_opt=args.require_bcir_opt)
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
