#!/usr/bin/env python3
"""Malformed-input differential that follows existing BCIR law rails.

Required rails: Python ``verify`` and an always-on MLIR text parser.
Compiled ``bcir-opt -bcir-verify`` is used only on official witnesses that
the compiled pass already implements (R1 RID uniqueness, isolated R5).
It is never stock ``mlir-opt``. Absence is UNAVAILABLE/SKIPPED — fatal
under ``--require-compiled``, which CI passes in the job that builds it.

This campaign does not add compiled laws. Oracle-only laws such as R1.1
(claim-id uniqueness per Module) stay on the Python rail.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
PYTHON_VERIFY_TIMEOUT = 20.0


class _VerifyHang(Exception):
    """Raised by the watchdog when a python-rail probe exceeds its bound."""


def _bounded_verify(probe: Callable[[], Any]) -> Any:
    """Run one python-rail probe under a wall bound (POSIX main thread),
    matching the compiled rail's 20s subprocess timeout. Where SIGALRM is
    unavailable (Windows, a non-main thread) the call runs unbounded — the
    watchdog is a POSIX rail, like the campaign tools'."""
    if (
        os.name == "nt"
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        return probe()

    def _expired(signum: int, frame: Any) -> None:
        raise _VerifyHang(f"python verifier timed out after {PYTHON_VERIFY_TIMEOUT}s")

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, PYTHON_VERIFY_TIMEOUT)
    try:
        return probe()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _is_bcir_opt_name(name: str) -> bool:
    # Positive identification: the tool must BE a bcir-opt build. A basename
    # test against exactly "mlir-opt" let version-suffixed stock binaries
    # (mlir-opt-22) through, which reject the BCIR dialect and poison the
    # differential with false disagreements.
    return "bcir-opt" in name


def find_bcir_opt(root: Path) -> str | None:
    env = os.environ.get("BCIR_OPT")
    if env:
        path = Path(env)
        if path.is_file() and _is_bcir_opt_name(path.name):
            return str(path)
    which = shutil.which("bcir-opt")
    if which and _is_bcir_opt_name(Path(which).name):
        return which
    for tree in (root / "build" / "mlir-build", root / "build" / "mlir"):
        if not tree.is_dir():
            continue
        # The cmake layout decides where the binary lands (bin/, tools/...);
        # mirror check_passes.sh's own discovery — the first bcir-opt file in
        # the build tree — instead of probing a fixed path that misses it.
        for candidate in sorted(tree.rglob("bcir-opt")):
            if candidate.is_file() and os.access(candidate, os.X_OK):
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


def _r1_duplicate_rid_python():
    # R1's Python enforcement is the CONSTRUCTION guard: add_resource raises
    # on a duplicate RID (Module.resources is keyed by RID, so a regressed
    # guard silently overwrites and verify() can never observe it). Pairing
    # the attempt here turns that regression into a rail disagreement.
    from types import SimpleNamespace
    from bcir.model import Domain, Module, Resource
    module = Module(name="r1")
    module.add_resource(Resource(rid=10, domain=Domain.RAM, shape=(4,)))
    try:
        module.add_resource(Resource(rid=10, domain=Domain.RAM, shape=(4,)))
    except ValueError:
        return [SimpleNamespace(law="R1")]
    return []


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
            # Bytes: the output is never parsed, and a crashing verifier can
            # spray non-UTF-8 that a text=True strict decode would raise on.
            result = subprocess.run(
                [opt, "-bcir-verify", str(path)],
                capture_output=True, check=False, timeout=20,
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
        # Diagnostic bytes, replace-decoded: the law-pairing check below
        # needs to see WHICH law rejected the witness.
        "stderr_tail": result.stderr.decode("utf-8", "replace")[-400:],
    }


def run_differential(root: Path, require_compiled: bool = False) -> dict[str, Any]:
    if require_compiled and find_bcir_opt(root) is None:
        # The llvm-training job builds bcir-opt specifically so the compiled
        # rail can be compared; there, a silently absent binary must fail the
        # job rather than skip every compiled case (mirrors --require-c).
        return {
            "state": "FAIL",
            "cases": [],
            "disagreements": ["compiled rail required but bcir-opt not found"],
            "malformed_rejected": 0,
            "required_rails": ["python-verify", "mlir-text", "mlir-compiled"],
            "error": "compiled rail required but bcir-opt not found",
        }
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
            "python": _r1_duplicate_rid_python,
            "mlir_text": _official_r1_mlir(),
            "compile": True,
            "text_reason": "duplicate-rid",
            "compiled_marker": "R1:",
        },
        {
            "name": "paired-official-r5",
            "expect_reject": True,
            "python": lambda: verify(r5),
            "mlir_text": _official_r5_mlir(),
            "compile": True,
            "text_reason": "r5-atomic-unique",
            "compiled_marker": "R5:",
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
        if case["python"] is None:
            python: dict[str, Any] = {
                "state": "UNAVAILABLE/SKIPPED", "reason": "no python module",
            }
        else:
            try:
                py_diags = _bounded_verify(case["python"])
            except _VerifyHang as exc:
                python = {
                    "state": "FAIL",
                    "rejected": False,
                    "reason": str(exc),
                }
                disagreements.append(f"{case['name']}/python: {python['reason']}")
            except Exception as exc:  # noqa: BLE001 - a crashing verifier IS the finding
                python = {
                    "state": "FAIL",
                    "rejected": False,
                    "reason": f"python verifier crashed: {type(exc).__name__}",
                }
                disagreements.append(f"{case['name']}/python: {python['reason']}")
            else:
                python = {
                    "state": "PASS",
                    "rejected": bool(py_diags),
                    "laws": [diag.law for diag in (py_diags or ())],
                }
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
        # A rejection must hit the law the witness exists to test: a witness
        # that drifts into syntax rot or a different check would otherwise
        # keep the differential green while the target law regresses.
        expected_reason = case.get("text_reason")
        if (
            expected_reason
            and text["state"] == "PASS"
            and text.get("rejected")
            and text.get("reason") != expected_reason
        ):
            disagreements.append(
                f"{case['name']}/mlir_text: rejected for reason "
                f"{text.get('reason')!r}, expected {expected_reason!r}"
            )
        marker = case.get("compiled_marker")
        if (
            marker
            and compiled["state"] == "PASS"
            and compiled.get("rejected")
            and marker not in compiled.get("stderr_tail", "")
        ):
            # The captured diagnostic IS the evidence of which law fired;
            # print it with the disagreement or the failure is undebuggable
            # from a CI log.
            tail = compiled.get("stderr_tail", "").strip().replace("\n", " | ")
            disagreements.append(
                f"{case['name']}/mlir_compiled: rejection diagnostic lacks {marker!r} "
                f"(stderr: {tail[:220]!r})"
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
    parser.add_argument("--require-compiled", action="store_true",
                        help="fail when the compiled bcir-opt rail is unavailable "
                             "(for the CI job that builds it)")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if str(args.root) not in sys.path:
        sys.path.insert(0, str(args.root))
    report = run_differential(args.root, require_compiled=args.require_compiled)
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
