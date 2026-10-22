#!/usr/bin/env python3
"""OT/BG (2026-10-21): the backend-optimality experiments' measurement contract (BG0), the sixteen
experiments as a registry, and the eight entry gates as predicates over evidence.

PR #821 proposes them (`docs/research/BCIR_BACKEND_OPTIMALITY_AND_KERNEL_GATES_2026-10-20.md` on its
branch, at c01a53c8): sixteen optimality experiments, OT-01 to OT-16 -- semantic graphs, compilation
and native kernels (5.1), GPU code generation and resident execution (5.2), movement, telemetry,
distribution and OS services (5.3) -- and eight gates, BG0 to BG7, from the measurement contract to
integrated promotion. The proposal introduces no executable test; this module is where they start:

* the contract's vocabularies -- the comparator ROUTES (section 3), the GAP kinds a result reports
  (section 2), the host CAPABILITIES an experiment needs -- and the RECORD every experiment writes,
  held by `validate_record` to one shape (BG0: route, semantic and target IDs, an independent
  reference, raw records, memory scopes, unsupported and fallback labels), every other shape
  refused by its own rule;
* `EXPERIMENTS`: the sixteen, each with the routes it compares, the gaps it can report, the
  capabilities it needs and the gates its evidence feeds (section 9's integration order);
* `GATES`: the eight, each a list of requirements over the evidence -- an experiment's record, a
  GEM+ row at its bound, a milestone of the driver ladder attested by its report, a host
  capability, another gate -- and `evaluate_gates`, which states each gate PASS only when every
  requirement is met by evidence, UNAVAILABLE when one needs a capability this host lacks (never a
  pass, never simulated), and OPEN otherwise, naming what is missing.

`host_capabilities` probes the host; `tools/perf/ot_run.py` runs the experiments this host can run
and writes the others' records as unavailable or blocked, each with its reason. Correctness,
refusal, trace identity and bounded accounting are deterministic gates; timings are this host's
evidence with an interval, never a universal threshold (section 10).

    python tools/perf/ot_contract.py --list               # the sixteen and the eight
    python tools/perf/ot_contract.py --gates RECORDS.json # each gate's state over the records
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass

#: The proposal the registry transcribes: its branch head when this module was written.
PROPOSAL = ("PR #821, docs/research/BCIR_BACKEND_OPTIMALITY_AND_KERNEL_GATES_2026-10-20.md at "
            "c01a53c8d577af2bde6ff182e15fff67782eb4e2")  # fmt: skip

#: Section 3: the comparator routes, each the question it isolates.
ROUTES = {
    "P-EAGER": "PyTorch eager tensor/module/autograd: cost and coverage of ordinary eager execution",
    "P-FX": "capture through Dynamo with an eager backend: capture, guards and graph breaks",
    "P-AOT": "AOTAutograd capture with an eager backend: functionalization and partitioning",
    "P-IND": "torch.compile with Inductor: the compiled baseline, its graph breaks and fallbacks",
    "P-TUNED": "a budgeted Inductor tuning configuration: the best admitted compiled result",
    "B-DIRECT": "BCIR's native C tensor and training entries: library quality and entry overhead",
    "B-PLAN": "BCIR's program and pack execution (PX-2a/PX-2b/PX-3/PX-3b): plan and dispatch effects",
    "B-MLIR": "BCIR-generated MLIR lowered to an executable, on its supported subset",
    "T-DSL": "Triton's tiled DSL on the same target and math contract",
    "T-GLUON": "Triton Gluon's explicit layouts on the same target and math contract",
    "B-DEVICE": "BCIR dispatch of the same device image a direct resident provider runs",
    "S-DEVICE": "a qualified SYCL device path under the same operation and device contract",
    "OS-DIRECT": "RuntimeChannel operations through the direct adapter",
    "OS-LINUX": "RuntimeChannel operations through a Linux-hosted adapter",
    "OS-NATIVE": "RuntimeChannel operations through a native adapter",
}

#: Section 2: the kinds of optimality gap a result reports (never one completion percentage).
GAPS = {
    "capability": "a required operation, gradient, shape, layout, effect or device path unsupported",
    "realization": "a better legal kernel, partition, packing, layout or schedule absent",
    "selection": "a better feasible candidate in the portfolio, another chosen",
    "integration": "the selected plan differs from the executed calls, storage or synchronization",
    "competitor": "T_BCIR / T_reference - 1 on matched hardware, work and semantics",
    "bound-slack": "(T_BCIR - L) / L for a justified positive lower bound L",
    "compilation": "cold compile, specialization, cache and search costs",
    "system": "submission, mapping, copies, queueing, isolation and event or service overhead",
}

#: The host facts an experiment needs; `host_capabilities` probes the ones a host can show.
CAPABILITIES = {
    "cpu": "a CPU running BCIR's native C rail (every host)",
    "numpy": "NumPy, for an independent float64 reference",
    "torch": "PyTorch, for the P-* routes",
    "triton": "Triton, for the T-* routes",
    "gpu": "a qualified GPU with a supported runtime (CUDA, HIP or SYCL)",
    "multi-device": "a qualified multi-device host",
    "pmu": "hardware performance counters readable by this process",
    "qemu": "QEMU, for the functional device/OS slice",
    "physical-device": "a qualified physical device rig (never inferred: declared by its operator)",
}


@dataclass(frozen=True)
class Experiment:
    """One of the sixteen (section 5): what it compares, what it can report, what it needs."""

    id: str
    title: str
    section: str
    routes: tuple[str, ...]
    gaps: tuple[str, ...]
    needs: tuple[str, ...]  # capabilities every run needs
    feeds: tuple[str, ...]  # the gates its evidence feeds (section 9)
    partial: str = ""  # what of it runs on fewer capabilities, where something does


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("OT-01", "common operation/gradient contract", "5.1", ("B-DIRECT", "P-EAGER"),
               ("capability", "competitor"), ("cpu", "numpy"), ("BG0", "BG1"),
               "the first common block (U = X Wg, V = X Wu, Y = silu(U) * V, and its backward) on "
               "BCIR's kernels against an independent float64 reference; P-EAGER needs PyTorch"),
    Experiment("OT-02", "IR/capture cost and coverage", "5.1",
               ("P-EAGER", "P-FX", "P-AOT", "P-IND", "B-PLAN"), ("capability", "compilation"),
               ("cpu", "torch"), ("BG0",)),
    Experiment("OT-03", "backward and optimizer identity", "5.1", ("B-DIRECT", "B-PLAN", "P-EAGER"),
               ("integration", "competitor"), ("cpu",), ("BG0", "BG1"),
               "the direct C step against the plan-executed step, bit for bit; the PyTorch "
               "comparison needs PyTorch"),
    Experiment("OT-04", "plan intervention", "5.1", ("B-DIRECT", "B-PLAN"),
               ("integration", "selection"), ("cpu",), ("BG0", "BG1")),
    Experiment("OT-05", "kernel/block portfolio", "5.1",
               ("B-DIRECT", "B-MLIR", "P-EAGER", "P-IND", "T-DSL", "T-GLUON"),
               ("realization", "competitor"), ("cpu", "torch"), ("BG0",)),
    Experiment("OT-06", "layout/packing/arena frontier", "5.1", ("B-DIRECT", "B-PLAN"),
               ("realization", "bound-slack"), ("cpu",), ("BG0", "BG1")),
    Experiment("OT-07", "mapping and conversion", "5.2", ("T-DSL", "T-GLUON", "B-DEVICE"),
               ("realization",), ("gpu", "triton"), ("BG0", "BG2")),
    Experiment("OT-08", "pipeline and synchronization", "5.2", ("T-DSL", "T-GLUON", "B-DEVICE"),
               ("realization", "system"), ("gpu", "triton"), ("BG0", "BG2")),
    Experiment("OT-09", "same image, different dispatcher", "5.2", ("B-DEVICE",),
               ("system",), ("gpu",), ("BG0", "BG2")),
    Experiment("OT-10", "complete training frontier", "5.2",
               ("B-DIRECT", "B-PLAN", "P-EAGER", "P-IND", "P-TUNED"), ("competitor", "bound-slack"),
               ("cpu", "torch"), ("BG0", "BG7"),
               "the fixed recipe against P-EAGER and P-IND, then the best-admitted-recipe study "
               "(P-TUNED: a budgeted Inductor tuning); every comparator needs PyTorch. BCIR's "
               "direct step and its plan-executed step are one state bit for bit (OT-03), so a "
               "learning curve on B-DIRECT is B-PLAN's too; their time differs by OT-04's ratio"),
    Experiment("OT-11", "specialization/cache lifecycle", "5.2",
               ("P-IND", "P-TUNED", "T-DSL", "B-PLAN"),
               ("compilation",), ("cpu", "torch"), ("BG0",),
               "P-IND's and B-PLAN's lifecycle; T-DSL's needs Triton with a usable backend (the "
               "experimental CPU build, or a GPU's)"),
    Experiment("OT-12", "data/residency path", "5.3", ("B-DEVICE", "S-DEVICE"),
               ("system",), ("gpu",), ("BG0", "BG2")),
    Experiment("OT-13", "collectives/checkpoint", "5.3", ("P-EAGER", "B-PLAN"),
               ("system",), ("multi-device",), ("BG0",)),
    Experiment("OT-14", "telemetry/TMOG observer and feedback", "5.3", ("B-DIRECT", "B-PLAN"),
               ("system",), ("cpu",), ("BG0", "BG4"),
               "the trace observer's cost on the native step, off against on, and its coverage; "
               "device counters need a PMU"),
    Experiment("OT-15", "IPC decomposition", "5.3", ("OS-DIRECT", "OS-LINUX", "OS-NATIVE"),
               ("system",), ("cpu",), ("BG0", "BG4")),
    Experiment("OT-16", "device/OS vertical slice", "5.3", ("OS-DIRECT", "OS-LINUX", "OS-NATIVE"),
               ("system", "capability"), ("qemu", "physical-device"), ("BG0", "BG4", "BG5")),
)  # fmt: skip

#: The gates' requirement kinds: an experiment's record measured (or, for BG0, any valid record),
#: a GEM+ row at its bound, a milestone of the driver ladder attested by a report the repository
#: holds, a host capability, and another gate.
REQUIREMENT_KINDS = ("record", "any-record", "row", "milestone", "gate", "capability")

#: The driver ladder's milestones (docs/kernel/BCIR_DRIVER_KERNEL_ROADMAP.md section 5.4) the gates
#: name, and the first half of M2 that BG3 needs alone: none is a fact a host can probe, so each is
#: met only by a report its evidence names.
MILESTONES = {
    "UART-D2": (
        "M2's first half: a D0-D2 UART package beside the RuntimeChannel loopback, with "
        "telemetry and bounded lifecycle tests"
    ),
    "M2": "UART proof: D0-D2, then Linux-hosted D3, with direct/Linux parity",
    "M4": "virtio queue proof: virtio-console and virtio-blk through D3",
    "M7": "targeted kernel research: measured patches against a stock baseline",
    "M8": "native IPC/kernel service proof: direct/Linux/native parity across UART and virtio-blk",
}


@dataclass(frozen=True)
class Requirement:
    kind: str
    key: str
    why: str


@dataclass(frozen=True)
class Gate:
    """One of the eight (section 8): its exit condition as requirements over the evidence."""

    id: str
    title: str
    requirements: tuple[Requirement, ...]


def _r(kind: str, key: str, why: str) -> Requirement:
    return Requirement(kind, key, why)


GATES: tuple[Gate, ...] = (
    Gate("BG0", "measurement contract", tuple(
        _r("any-record", e.id, f"{e.id} written under the contract (measured, or unavailable with "
                                "its missing capabilities, or blocked with its prerequisite)")
        for e in EXPERIMENTS)),
    Gate("BG1", "trace-to-execution", (
        _r("gate", "BG0", "the measurement contract"),
        _r("row", "train.plan.executed.mismatch", "PX-1: the launches held to their program"),
        _r("row", "train.plan.program.mismatch", "PX-2a: the step from its program, bit for bit"),
        _r("row", "train.plan.pack.mismatch", "PX-2b: the step from its pack, bit for bit"),
        _r("row", "train.plan.choose.mismatch", "PX-3: every choice the C step's bits"),
        _r("record", "OT-03", "the direct step and the plan-executed step one state"),
        _r("record", "OT-04", "one causal plan intervention on a common block"),
    )),
    Gate("BG2", "hosted accelerator slice", (
        _r("gate", "BG0", "the measurement contract"),
        _r("capability", "gpu", "one real target with a resident provider"),
        *(_r("record", x, f"{x} on that target") for x in ("OT-07", "OT-08", "OT-09", "OT-12")),
    )),
    Gate("BG3", "direct driver slice", (
        _r("gate", "BG0", "the measurement contract"),
        _r("milestone", "UART-D2", "a D0-D2 UART package beside the RuntimeChannel loopback, "
                                   "with telemetry and bounded lifecycle tests"),
    )),
    Gate("BG4", "Linux queue/transport proof", (
        _r("gate", "BG3", "the direct driver slice"),
        _r("milestone", "M2", "UART through Linux-hosted D3, direct/Linux parity"),
        _r("milestone", "M4", "virtio console and block, direct/Linux parity"),
        *(_r("record", x, f"{x} measured") for x in ("OT-14", "OT-15", "OT-16")),
    )),
    Gate("BG5", "native service/boot proof", (
        _r("gate", "BG4", "the Linux queue/transport proof"),
        _r("capability", "qemu", "a UEFI/reset boot proof"),
        _r("milestone", "M8", "the native service: direct/Linux/native replay"),
        _r("record", "OT-16", "direct/Linux/native replay for the supported devices"),
    )),
    Gate("BG6", "targeted kernel policy/fast path", (
        _r("gate", "BG5", "the native service/boot proof"),
        _r("capability", "physical-device", "a profile on the declared workload's hardware"),
        _r("milestone", "M7", "an adapter or kernel change that improves the declared workload"),
    )),
    Gate("BG7", "integrated promotion", (
        *(_r("gate", g, f"{g} passed") for g in ("BG1", "BG2", "BG4", "BG5", "BG6")),
        _r("record", "OT-10", "a reproducible complete-workload benefit"),
    )),
)  # fmt: skip

#: The record's statuses: measured here; unavailable (a capability this host lacks, named);
#: blocked (a prerequisite the repository lacks, named) -- the last two are never a pass.
STATUSES = ("measured", "unavailable", "blocked")
RECORD_KEYS = frozenset({"experiment", "status", "routes", "semantic", "target", "versions",
                         "reference", "results", "raw", "memory_scopes", "labels", "reason",
                         "missing", "gaps"})  # fmt: skip
_ID = re.compile(r"[a-z0-9][a-z0-9._+-]*\Z", re.ASCII)


def _refuse(rule: str) -> ValueError:
    return ValueError(f"an OT record {rule}")


def validate_record(record) -> dict:
    """`record` held to the contract (BG0), returned as given; every other shape refused by its
    own rule (ValueError):

    * a JSON object of the contract's keys alone, naming one of the sixteen experiments and one of
      the statuses;
    * measured: the routes it compared (each one of the experiment's), a semantic ID and an
      independent reference (lower-case ASCII identifiers), the target (its CPU model and threads),
      the versions it ran (a non-empty object of strings), its results and raw records (each a
      non-empty list), its memory scopes (an object, each a non-negative integer of bytes), its
      unsupported and fallback labels (two lists of strings), the gaps it reports (each one of the
      experiment's), and no reason or missing capabilities;
    * unavailable: a reason and the capabilities this host lacks (each one the experiment needs);
    * blocked: a reason (the prerequisite the repository lacks)."""
    if not isinstance(record, dict) or not record:
        raise _refuse("is a JSON object")
    extra = set(record) - RECORD_KEYS
    if extra:
        raise _refuse(f"has no key {sorted(extra)[0]!r}")
    experiment = next((e for e in EXPERIMENTS if e.id == record.get("experiment")), None)
    if experiment is None:
        raise _refuse(f"names one of the sixteen experiments, not {record.get('experiment')!r}")
    status = record.get("status")
    if status not in STATUSES:
        raise _refuse(f"has a status of {STATUSES}, not {status!r}")
    if status == "measured":
        need = {"routes", "semantic", "target", "versions", "reference", "results", "raw",
                "memory_scopes", "labels", "gaps"}  # fmt: skip
        missing = sorted(need - set(record))
        if missing:
            raise _refuse(f"measured names its {missing[0]}")
        if "reason" in record or "missing" in record:
            raise _refuse("measured carries no reason and no missing capabilities")
        routes = record["routes"]
        if (not isinstance(routes, list) or not routes or len(set(routes)) != len(routes)
                or any(r not in experiment.routes for r in routes)):  # fmt: skip
            raise _refuse(f"names routes of {experiment.id}'s ({', '.join(experiment.routes)})")
        for key in ("semantic", "reference"):
            if type(record[key]) is not str or not _ID.match(record[key]):
                raise _refuse(f"names its {key} as a lower-case ASCII identifier")
        target = record["target"]
        if (not isinstance(target, dict) or type(target.get("cpu")) is not str
                or not target["cpu"] or type(target.get("threads")) is not int
                or target["threads"] < 1):  # fmt: skip
            raise _refuse("names its target's cpu and threads")
        versions = record["versions"]
        if (not isinstance(versions, dict) or not versions
                or any(type(k) is not str or type(v) is not str or not v
                       for k, v in versions.items())):  # fmt: skip
            raise _refuse("names the versions it ran as strings")
        for key in ("results", "raw"):
            if not isinstance(record[key], list) or not record[key]:
                raise _refuse(f"carries its {key} (a non-empty list)")
        scopes = record["memory_scopes"]
        if not isinstance(scopes, dict) or any(
            type(k) is not str or type(v) is not int or v < 0 for k, v in scopes.items()
        ):
            raise _refuse("states each memory scope in bytes (a non-negative integer)")
        labels = record["labels"]
        if (not isinstance(labels, dict) or set(labels) != {"unsupported", "fallback"}
                or any(not isinstance(v, list) or any(type(x) is not str for x in v)
                       for v in labels.values())):  # fmt: skip
            raise _refuse("labels what was unsupported and what fell back (two lists of strings)")
        gaps = record["gaps"]
        if (not isinstance(gaps, list) or not gaps
                or any(g not in experiment.gaps for g in gaps)):  # fmt: skip
            raise _refuse(f"reports gaps of {experiment.id}'s ({', '.join(experiment.gaps)})")
        return record
    if type(record.get("reason")) is not str or not record["reason"].strip():
        raise _refuse(f"{status} names its reason")
    if status == "unavailable":
        lacking = record.get("missing")
        if (not isinstance(lacking, list) or not lacking
                or any(c not in experiment.needs for c in lacking)):  # fmt: skip
            raise _refuse(f"unavailable names the capabilities it lacks, of {experiment.id}'s "
                          f"({', '.join(experiment.needs)})")  # fmt: skip
    elif "missing" in record:
        raise _refuse("blocked names a prerequisite, not missing capabilities")
    if any(k in record for k in ("results", "raw")):
        raise _refuse(f"{status} carries no results")
    return record


def host_capabilities() -> set[str]:
    """The capabilities this host shows: the CPU always; NumPy or PyTorch when they import, Triton
    when it imports with a usable backend; QEMU when a system emulator is on the path; a PMU when
    `perf_event_open` admits this process. A GPU, a multi-device host and a physical device rig are
    never inferred: their operator declares them (`BCIR_OT_CAPABILITIES`, comma-separated), as
    section 10 requires their qualification."""
    import importlib.util

    caps = {"cpu"}
    for name in ("numpy", "torch"):
        if importlib.util.find_spec(name) is not None:
            caps.add(name)
    if _triton_usable():
        caps.add("triton")
    if shutil.which("qemu-system-x86_64") or shutil.which("qemu-system-aarch64"):
        caps.add("qemu")
    if _pmu_readable():
        caps.add("pmu")
    declared = os.environ.get("BCIR_OT_CAPABILITIES", "")
    for cap in filter(None, (c.strip() for c in declared.split(","))):
        if cap not in CAPABILITIES:
            raise ValueError(f"BCIR_OT_CAPABILITIES names no capability {cap!r}")
        caps.add(cap)
    return caps


def _triton_usable() -> bool:
    """Whether Triton imports and names an active backend -- a GPU's driver, or the CPU backend of
    the experimental triton-cpu build: a package without a target runs no kernel."""
    import importlib.util

    if importlib.util.find_spec("triton") is None:
        return False
    try:
        import triton

        return triton.runtime.driver.active is not None
    except Exception:  # no usable backend: a GPU build without its device, a broken install
        return False


def _pmu_readable() -> bool:
    """Whether perf_event_open admits a hardware cycle counter for this process (Linux only)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        attr = (ctypes.c_uint8 * 128)()
        attr[4] = 128  # size
        # type PERF_TYPE_HARDWARE (0), config PERF_COUNT_HW_CPU_CYCLES (0); exclude kernel (bit 5)
        attr[40] = 1 << 5 | 1 << 6  # exclude_kernel, exclude_hv
        nr = {"x86_64": 298, "aarch64": 241}.get(os.uname().machine)
        if nr is None:
            return False
        fd = libc.syscall(nr, attr, 0, -1, -1, 0)
        if fd < 0:
            return False
        os.close(fd)
        return True
    except OSError:
        return False


def _attested(report, root: str) -> bool:
    """Whether `report` names a file the repository holds: a relative POSIX path of plain parts
    (no `..`, no absolute or drive-led path), existing under `root`."""
    if type(report) is not str or not report or report.startswith("/") or "\\" in report:
        return False
    parts = report.split("/")
    if any(part in ("", ".", "..") or ":" in part for part in parts):
        return False
    return os.path.isfile(os.path.join(root, *parts))


def evaluate_gates(
    records,
    rows: dict | None = None,
    *,
    capabilities=None,
    milestones: dict | None = None,
    root: str | None = None,
):
    """Each gate's state over the evidence, as `{gate: (state, [unmet requirement, ...])}`.

    `records` are the experiments' records (each validated: a record the contract refuses is no
    evidence, and the evaluation refuses it), `rows` the GEM+ rows as measured (a row is met at its
    bound, 0), `capabilities` the host's (`host_capabilities()` by default), `milestones` the
    driver-ladder milestones attested, each by the path of its report in the repository (a
    milestone is met only where that report exists), `root` the repository.
    A gate is PASS when every requirement is met; UNAVAILABLE when one is a capability the host
    lacks or an experiment's record unavailable here -- never a pass; OPEN otherwise. A gate
    requiring another is met only by that gate's PASS, so a gate's state never outruns its
    prerequisite's."""
    root = root or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    caps = host_capabilities() if capabilities is None else set(capabilities)
    rows = dict(rows or {})
    milestones = dict(milestones or {})
    for key in milestones:
        if key not in MILESTONES:
            raise ValueError(f"no milestone {key!r} a gate names ({', '.join(MILESTONES)})")
    by_id: dict[str, dict] = {}
    for record in records:
        validate_record(record)
        if record["experiment"] in by_id:
            raise ValueError(f"two records of {record['experiment']}")
        by_id[record["experiment"]] = record
    out: dict[str, tuple[str, list[str]]] = {}
    for gate in GATES:
        unmet, unavailable = [], False
        for req in gate.requirements:
            met = False
            if req.kind == "any-record":
                met = req.key in by_id
            elif req.kind == "record":
                rec = by_id.get(req.key)
                met = rec is not None and rec["status"] == "measured"
                unavailable |= rec is not None and rec["status"] == "unavailable"
            elif req.kind == "row":
                met = rows.get(req.key) == 0
            elif req.kind == "milestone":
                met = _attested(milestones.get(req.key), root)
            elif req.kind == "capability":
                met = req.key in caps
                unavailable |= not met
            elif req.kind == "gate":
                met = out[req.key][0] == "PASS"
                unavailable |= out[req.key][0] == "UNAVAILABLE"
            else:  # pragma: no cover - the registry's own vocabulary
                raise ValueError(f"no requirement kind {req.kind!r}")
            if not met:
                unmet.append(f"{req.kind} {req.key}: {req.why}")
        state = "PASS" if not unmet else "UNAVAILABLE" if unavailable else "OPEN"
        out[gate.id] = (state, unmet)
    return out


def _unique_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"the key {key!r} twice in one object")
        out[key] = value
    return out


def load_json(path) -> object:
    """A JSON file the contract reads (records, rows, milestones): UTF-8, every object's keys
    distinct -- a key spelled twice is refused, never resolved to its last value."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_unique_keys)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true", help="the sixteen experiments and the gates")
    p.add_argument("--gates", type=str, help="a records JSON file (tools/perf/ot_run.py's)")
    a = p.parse_args(argv)
    if a.list or not a.gates:
        for e in EXPERIMENTS:
            print(f"{e.id}  {e.title}  [{', '.join(e.routes)}]  needs {', '.join(e.needs)}  "
                  f"feeds {', '.join(e.feeds)}")  # fmt: skip
        for g in GATES:
            print(f"{g.id}  {g.title}: {len(g.requirements)} requirement(s)")
        return 0
    try:
        doc = load_json(a.gates)
        if not isinstance(doc, dict) or not isinstance(doc.get("records"), list):
            raise ValueError("a records file is an object holding a list of records")
        states = evaluate_gates(doc["records"], doc.get("rows", {}),
                                milestones=doc.get("milestones", {}))  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ot_contract: INVALID: {exc}")
        return 2
    for gate, (state, unmet) in states.items():
        print(f"{gate} {state}" + "".join(f"\n    - {u}" for u in unmet))
    return 0


if __name__ == "__main__":
    sys.exit(main())
