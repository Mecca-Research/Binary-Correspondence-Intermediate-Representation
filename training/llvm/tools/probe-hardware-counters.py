#!/usr/bin/env python3
"""Ask the host whether hardware performance counters are readable, and record
the answer -- including "no" -- as evidence.

Counter harnesses fail in a characteristic way: they run, every counter comes
back 0, and 0 is written into a report as a measurement. A cache-miss rate of
zero then becomes a finding. This probe exists so that the unreadable case has
a representation that is not a number:

    "instructions": null,      not      "instructions": 0

The probe itself is the real test. Reading /proc/sys/kernel/perf_event_paranoid
tells you about permission; it does not tell you whether the counters exist. A
virtualized host commonly grants full privilege over a PMU it does not have, and
answers ENOENT to the syscall regardless. Privilege is not capability, so this
tool opens a counter and reports what the kernel said.

    python3 training/llvm/tools/probe-hardware-counters.py
    python3 training/llvm/tools/probe-hardware-counters.py --format json
    python3 training/llvm/tools/probe-hardware-counters.py --require-counters

Exit status: 0 when the probe completed (whether or not counters were
available), 1 when --require-counters was passed and they were not, 2 on an
internal error. The default is 0 because "counters are unavailable here" is a
successful measurement of the host.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import errno
import json
import os
import platform
import shutil
import struct
import sys
from pathlib import Path

# perf_event_attr.type
PERF_TYPE_HARDWARE = 0
# perf_event_attr.config
HARDWARE_COUNTERS = {
    "cpu_cycles": 0,
    "instructions": 1,
    "cache_references": 2,
    "cache_misses": 3,
    "branch_instructions": 4,
    "branch_misses": 5,
}

# perf_event_attr flag bits used here.
FLAG_DISABLED = 1 << 0
FLAG_EXCLUDE_KERNEL = 1 << 5
FLAG_EXCLUDE_HV = 1 << 6

PERF_EVENT_ATTR_SIZE = 128

# __NR_perf_event_open, per architecture.
SYSCALL_NUMBERS = {
    "x86_64": 298,
    "aarch64": 241,
    "armv7l": 364,
    "ppc64le": 319,
    "s390x": 331,
    "riscv64": 241,
}


def build_attr(config: int) -> ctypes.Array:
    """A zero-filled perf_event_attr with only the fields we set."""
    raw = bytearray(PERF_EVENT_ATTR_SIZE)
    struct.pack_into("<I", raw, 0, PERF_TYPE_HARDWARE)  # type
    struct.pack_into("<I", raw, 4, PERF_EVENT_ATTR_SIZE)  # size
    struct.pack_into("<Q", raw, 8, config)  # config
    struct.pack_into("<Q", raw, 40, FLAG_DISABLED | FLAG_EXCLUDE_KERNEL | FLAG_EXCLUDE_HV)
    return (ctypes.c_char * PERF_EVENT_ATTR_SIZE).from_buffer(raw)


def try_open_counter(name: str, config: int) -> tuple[bool, str | None]:
    """Open one hardware counter. Returns (available, refusal reason)."""
    machine = platform.machine()
    syscall_number = SYSCALL_NUMBERS.get(machine)
    if syscall_number is None:
        return False, f"unknown perf_event_open syscall number for {machine!r}"

    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.syscall.argtypes = [
        ctypes.c_long,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_ulong,
    ]

    attr = build_attr(config)
    ctypes.set_errno(0)
    fd = libc.syscall(
        syscall_number,
        ctypes.cast(attr, ctypes.c_void_p),
        0,  # pid: this process
        -1,  # cpu: any
        -1,  # group_fd
        0,  # flags
    )
    if fd < 0:
        code = ctypes.get_errno()
        return False, f"{errno.errorcode.get(code, code)}: {os.strerror(code)}"

    os.close(fd)
    return True, None


def read_paranoid() -> int | None:
    path = Path("/proc/sys/kernel/perf_event_paranoid")
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def detect_virtualization() -> str | None:
    """Best-effort. A positive answer is informative; a negative one is not."""
    hypervisor_type = Path("/sys/hypervisor/type")
    try:
        return hypervisor_type.read_text().strip()
    except OSError:
        pass
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        if "hypervisor" in cpuinfo:
            return "hypervisor flag present in /proc/cpuinfo"
    except OSError:
        pass
    if shutil.which("systemd-detect-virt"):
        return "systemd-detect-virt available; not run (it may block)"
    return None


def pmu_devices() -> list[str]:
    root = Path("/sys/bus/event_source/devices")
    try:
        return sorted(p.name for p in root.iterdir())
    except OSError:
        return []


def probe() -> dict:
    record: dict = {
        "schema": "training/llvm/hardware-counter-probe/v1",
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "euid": os.geteuid() if hasattr(os, "geteuid") else None,
        },
        "capability": {
            "perf_event_paranoid": read_paranoid(),
            "pmu_devices": pmu_devices(),
            "perf_binary": shutil.which("perf"),
            "virtualization_hint": detect_virtualization(),
        },
        # Every counter is null until the kernel hands one over. This is the
        # whole point of the file: the unreadable case is representable.
        "counters": {name: None for name in HARDWARE_COUNTERS},
        "status": "unknown",
        "refusals": {},
    }

    if platform.system() != "Linux":
        record["status"] = "unsupported-platform"
        record["refusals"]["*"] = (
            f"perf_event_open is Linux-only; this host reports {platform.system()}"
        )
        return record

    available = 0
    for name, config in HARDWARE_COUNTERS.items():
        try:
            ok, reason = try_open_counter(name, config)
        except OSError as exc:  # pragma: no cover - defensive
            ok, reason = False, f"probe error: {exc}"
        if ok:
            available += 1
            # The counter opened. Reading a MEANINGFUL value needs a workload
            # under it; this probe deliberately measures capability only, so
            # the value stays null rather than becoming a zero-length sample.
            record["refusals"][name] = (
                "counter is openable; no workload was run, so no value is reported"
            )
        else:
            record["refusals"][name] = reason

    if available == len(HARDWARE_COUNTERS):
        record["status"] = "available"
    elif available:
        record["status"] = "partial"
    else:
        record["status"] = "unavailable"

    return record


def render_text(record: dict) -> str:
    lines = [
        f"host          : {record['host']['system']} {record['host']['machine']} "
        f"{record['host']['release']}",
        f"euid          : {record['host']['euid']}",
        f"paranoid level: {record['capability']['perf_event_paranoid']}",
        f"PMU devices   : {', '.join(record['capability']['pmu_devices']) or '(none)'}",
        f"perf binary   : {record['capability']['perf_binary'] or '(absent)'}",
        f"virtualization: {record['capability']['virtualization_hint'] or '(no hint)'}",
        "",
        f"{'counter':<22} {'value':>8}  refusal / note",
    ]
    for name in HARDWARE_COUNTERS:
        value = record["counters"][name]
        rendered = "null" if value is None else str(value)
        lines.append(f"{name:<22} {rendered:>8}  {record['refusals'].get(name, '')}")
    lines.append("")
    lines.append(f"STATUS: {record['status']}")
    if record["status"] == "unavailable":
        lines.append(
            "  No hardware counter could be opened on this host. Note that a "
            "privilege level of 0 or -1 does not change this: on a virtualized "
            "host the PMU is commonly absent regardless of permission. Record "
            "counters as null, never as 0, and do not report a modelled number "
            "as a measured one."
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="write the evidence record to this path")
    parser.add_argument(
        "--require-counters",
        action="store_true",
        help="exit nonzero when counters are unavailable (for a rig that is supposed to have them)",
    )
    args = parser.parse_args()

    try:
        record = probe()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"hardware counter probe: FAILED ({exc})", file=sys.stderr)
        return 2

    rendered = (
        json.dumps(record, indent=2, sort_keys=True)
        if args.format == "json"
        else render_text(record)
    )
    print(rendered)

    if args.output:
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if args.require_counters and record["status"] != "available":
        print(
            f"hardware counter probe: counters required but status is '{record['status']}'",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
