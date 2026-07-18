"""``bcir-model-assess``: payload-free model cost and execution-plan compiler."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from ..._artifact_json import read_bounded_text, strict_json_loads
from .assessment import (HardwareEnvelope, ModelWorkloadSpec, assess_model,
                         select_execution_candidate)
from .execution_plan import lower_model_execution
from .inventory import build_tensor_inventory
from .model_microbench import run_bounded_model_microbench


def _atomic_write(path: str, data: bytes) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
                                     dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _config(path: Path) -> dict:
    value = strict_json_loads(read_bounded_text(str(path), "model config", max_bytes=4 << 20),
                              "model config", max_bytes=4 << 20)
    if not isinstance(value, dict):
        raise ValueError("model config root must be an object")
    return value


def _shards(model_dir: Path, explicit: list[str]) -> list[str]:
    if explicit:
        paths = [Path(value) for value in explicit]
        return [str(path if path.is_absolute() else model_dir / path) for path in paths]
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        doc = strict_json_loads(read_bounded_text(str(index), "Safetensors index", max_bytes=16 << 20),
                                "Safetensors index", max_bytes=16 << 20)
        if not isinstance(doc, dict) or not isinstance(doc.get("weight_map"), dict):
            raise ValueError("Safetensors index requires a weight_map object")
        values = list(doc["weight_map"].values())
        if any(not isinstance(value, str) or not value or value != os.path.basename(value)
               for value in values):
            raise ValueError("Safetensors index shard names must be basenames")
        files = sorted(set(values))
        return [str(model_dir / value) for value in files]
    single = model_dir / "model.safetensors"
    if single.is_file():
        return [str(single)]
    paths = sorted(model_dir.glob("model-*.safetensors"))
    if not paths:
        raise ValueError("no model Safetensors shards found; pass --shard explicitly")
    return [str(path) for path in paths]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bcir-model-assess",
        description="Assess model headers and compile a verified placement plan without loading weights.")
    parser.add_argument("model_dir", help="directory containing config.json and Safetensors shards")
    parser.add_argument("--config", default="config.json", help="config path relative to model_dir")
    parser.add_argument("--shard", action="append", default=[], help="explicit shard (repeatable)")
    parser.add_argument("--hardware", required=True, help="HardwareEnvelope JSON")
    parser.add_argument("--workload", required=True, help="ModelWorkloadSpec JSON")
    parser.add_argument("--inventory-out", help="write canonical TensorInventory JSON")
    parser.add_argument("--report-out", help="write canonical ModelCostReport JSON")
    parser.add_argument("--plan-out", help="write selected content-addressed plan JSON")
    parser.add_argument("--pack-out", help="write selected verified StreamPack binary")
    parser.add_argument("--candidate", help="select a feasible candidate by exact id")
    parser.add_argument("--microbench", action="store_true",
                        help="run bounded local reference prefill/decode calibration")
    parser.add_argument("--microbench-channel", default="host")
    parser.add_argument("--microbench-dimension", type=int, default=24)
    parser.add_argument("--microbench-repeats", type=int, default=5)
    return parser


def _run(args) -> int:
    model_dir = Path(args.model_dir)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = model_dir / config_path
    inventory = build_tensor_inventory(_shards(model_dir, args.shard), _config(config_path))
    hardware = HardwareEnvelope.from_json(read_bounded_text(
        args.hardware, "hardware envelope", max_bytes=64 << 20))
    workload = ModelWorkloadSpec.from_json(read_bounded_text(
        args.workload, "model workload", max_bytes=1 << 20))
    if args.microbench:
        evidence = list(hardware.benchmarks)
        for fmt in workload.formats:
            measured = run_bounded_model_microbench(
                channel=args.microbench_channel, weight_format=fmt,
                dimension=args.microbench_dimension, repeats=args.microbench_repeats)
            keys = {(row.operation, row.channel, row.weight_format) for row in measured}
            evidence = [row for row in evidence
                        if (row.operation, row.channel, row.weight_format) not in keys]
            evidence.extend(measured)
        hardware = HardwareEnvelope(hardware.name, hardware.architecture, hardware.banks,
                                    hardware.links, tuple(evidence))
    report = assess_model(inventory, hardware, workload)
    if args.inventory_out:
        _atomic_write(args.inventory_out, inventory.to_json().encode("utf-8") + b"\n")
    if args.report_out:
        _atomic_write(args.report_out, report.to_json().encode("utf-8") + b"\n")

    selected = None
    if args.candidate:
        selected = next((row for row in report.candidates
                         if row.candidate_id == args.candidate), None)
        if selected is None:
            raise ValueError(f"candidate {args.candidate!r} is absent from the report")
        if not selected.feasible:
            raise ValueError(f"candidate {args.candidate!r} is infeasible: "
                             + "; ".join(selected.blockers))
    else:
        try:
            selected = select_execution_candidate(report)
        except ValueError:
            selected = None
    lowered = None
    if selected is not None and (args.plan_out or args.pack_out):
        lowered = lower_model_execution(report, selected, inventory, hardware, workload)
        if args.plan_out:
            _atomic_write(args.plan_out, lowered.artifact.to_json().encode("utf-8") + b"\n")
        if args.pack_out:
            _atomic_write(args.pack_out, lowered.streampack_data)
    summary = {
        "inventory_digest": inventory.header_digest,
        "report_digest": report.digest,
        "parameter_count": inventory.parameter_count,
        "payload_bytes_read": 0,
        "selected_candidate": selected.candidate_id if selected else None,
        "plan_digest": lowered.artifact.digest if lowered else None,
        "feasible_candidates": sum(row.feasible for row in report.candidates),
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if selected is not None else 1


def main(argv=None) -> int:
    try:
        return _run(_parser().parse_args(argv))
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"bcir-model-assess: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
