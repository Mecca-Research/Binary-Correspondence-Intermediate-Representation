"""``bcir-ham-plan``: compile a metadata-only HAM workload to verified StreamPack."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .._artifact_json import read_bounded_text
from ..frontends.models.assessment import HardwareEnvelope
from .ham import HAMWorkload, lower_ham_workload


def _atomic_write(path: str, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bcir-ham-plan",
        description=(
            "Plan semantic resource movement from metadata only; direct routes "
            "exist only when declared by the hardware envelope."
        ),
    )
    parser.add_argument("--hardware", required=True, help="HardwareEnvelope JSON")
    parser.add_argument("--workload", required=True, help="HAMWorkload JSON")
    parser.add_argument("--plan-out", required=True, help="canonical HAMExecutionPlan JSON")
    parser.add_argument("--pack-out", help="verified StreamPack binary")
    return parser


def _run(args) -> int:
    hardware = HardwareEnvelope.from_json(
        read_bounded_text(args.hardware, "hardware envelope", max_bytes=64 << 20)
    )
    workload = HAMWorkload.from_json(
        read_bounded_text(args.workload, "HAM workload", max_bytes=32 << 20)
    )
    lowered = lower_ham_workload(workload, hardware)
    _atomic_write(args.plan_out, lowered.artifact.to_json().encode("utf-8") + b"\n")
    if args.pack_out:
        _atomic_write(args.pack_out, lowered.streampack_data)
    summary = {
        "hardware_digest": hardware.digest,
        "workload_digest": workload.digest,
        "plan_digest": lowered.artifact.digest,
        "payload_bytes_read": 0,
        "actions": len(lowered.artifact.actions),
        "logical_transfer_bytes": lowered.artifact.logical_transfer_bytes,
        "wire_transfer_bytes": lowered.artifact.wire_transfer_bytes,
        "direct_peer_bytes": lowered.artifact.direct_peer_bytes,
        "staged_transfer_bytes": lowered.artifact.staged_transfer_bytes,
        "host_bounce_bytes": lowered.artifact.host_bounce_bytes,
        "predicted_transfer_ns": lowered.artifact.predicted_transfer_ns,
        "streampack_bytes": lowered.artifact.streampack_bytes,
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


def main(argv=None) -> int:
    try:
        return _run(_parser().parse_args(argv))
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"bcir-ham-plan: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
