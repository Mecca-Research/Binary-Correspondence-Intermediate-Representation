"""Physics-anchored cost calibration: measure, quantize, freeze, generation-tag.

The L1 learning law (LangRef Sec. 13): learning and measurement may only enter
the planner as **frozen, generation-tagged, integer tables** -- never as
plan-time inference. This module is the measurement half of that law:

  1. **Measure** (`run_microbench`): host microbenchmarks time four access
     regimes over a buffer that defeats the cache -- streaming, strided,
     random (gather), and pure compute. Index orders are seeded/deterministic;
     only the timing varies. Medians-of-repeats reject noise.
  2. **Quantize** (`calibrate_from_raw`, a pure function): raw timings become
     Q8 ratios against the streaming baseline (stream == 256 by definition),
     then the cost-model constants the seeds guessed:
         gather_penalty = random_q8 / 256      (whole-multiple per-access penalty)
         base_overhead  = 4 * strided_q8 / 256 (unit-stride access-op cost scale)
         mem_unit       = 1                    (the bandwidth unit, definitional)
     A ratio-1 measurement reproduces the seeded constants exactly (the
     degenerate case) -- the checked-in reference table is exactly that.
  3. **Freeze** (`CalibratedProfile`): the table is an immutable dataclass with
     a JSON wire form (`tables/*.json`), applied to a `TargetProfile` by
     `apply()` -- a deterministic data substitution, no code change.
  4. **Generation-tag** (`cal_gen`): every table carries a generation and
     provenance (host fingerprint, sample count); profiles record the
     generation they were calibrated under (R11-style freshness for tables).

Fidelity note: the stdlib harness measures through the interpreter, so absolute
numbers are conservative; the *ratios* still order the regimes correctly, and
the table schema is the contract -- a native (C-runtime) backend can fill the
same tables with silicon numbers without touching the planner.

The native rig's evidence (S0-F / GEM+ G7). `runtime/c/bcir_microbench.c` prints the
same table plus a `NativeEvidence` object: the census of unique elements each regime
touched (the old strided walk `(k * 16) % n` visited n/16 of a power-of-two buffer and
called it cache-defeating), the raw per-regime samples with their min/median/max/MAD,
and an attestation of the host -- hypervisor flag, DMI, WSL, container, PMU event
source, perf_event_paranoid, governor, RAPL, clocksource, timer quantum -- from which
the rig derives its TENANCY. "bare-metal" is a verdict the evidence has to earn (no
virtualization or container signal and an exposed hardware PMU); the old rig printed it
unconditionally. `CalibratedProfile.from_json` re-derives the Q8 ratios from the sample
medians it is handed and refuses a table whose summary disagrees with its evidence, and
`calibrate_native(require_baremetal=True)` refuses any tenancy but the proved one.
"""

from __future__ import annotations

import json
import os
import platform
import time
from array import array
from dataclasses import asdict, dataclass, replace

from .._artifact_json import read_bounded_text, strict_json_loads
from .cost import TargetProfile

Q8 = 256  # 1.0 in Q8 fixed point (the streaming baseline by definition)
TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")

#: The tenancies the native rig can attest. Only "bare-metal" supports a silicon claim;
#: it needs no virtualization or container signal AND an exposed hardware PMU.
TENANCIES = ("bare-metal", "virtualized", "containerized", "unproven")
_MAX_REPEATS = 128
_EVIDENCE_STR_MAX = 256


def _bounded_str(value, what: str, limit: int = _EVIDENCE_STR_MAX) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > limit
        or any(ord(ch) < 0x20 for ch in value)
    ):
        raise ValueError(f"{what} must be a bounded string")
    return value


def _u63(value, what: str, lo: int = 0, hi: int = (1 << 63) - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < lo or value > hi:
        raise ValueError(f"{what} must be an integer in [{lo}, {hi}]")
    return value


def _median_of(xs) -> int:
    ordered = sorted(xs)
    return ordered[len(ordered) // 2]


# --- the host attestation: ONE predicate, mirrored by runtime/c/bcir_microbench.c ---------
#
# The rig decides its tenancy from files under /proc and /sys. The reader below applies the
# same rules to the same files, so a test can hold the C rig to the Python twin field by
# field (the GitHub aarch64 runner is a DMI-attested VM that also exposes a PMU: a reader
# that checked fewer signals than the rig called it bare metal). Keep the two in lockstep:
# a signal added on one side is a drift the test reports as a disagreement.

_DMI_VIRTUAL = (
    "kvm",
    "qemu",
    "vmware",
    "virtualbox",
    "innotek",
    "xen",
    "bochs",
    "parallels",
    "hyper-v",
    "virtual machine",
    "amazon ec2",
    "google compute engine",
    "openstack",
    "bhyve",
    "cloud hypervisor",
    "firecracker",
)
_PMU_SOURCES = ("cpu", "cpu_core", "cpu_atom", "armv8_pmuv3", "armv8_pmuv3_0")
_CONTAINER_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "lxc", "libpod")


def _readable(path: str) -> bool:
    """fopen(path, "r") succeeds: the rig's existence test."""
    try:
        with open(path, "rb"):
            return True
    except IsADirectoryError:
        return True
    except OSError:
        return False


def _first_line(path: str) -> str:
    """The first line of `path` without its newline, or "unavailable" (the rig's rule)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            line = fh.readline()
    except OSError:
        return "unavailable"
    line = line.rstrip("\r\n")
    return line if line else "unavailable"


def _file_contains(path: str, needle: str) -> bool:
    """Whether any line of `path`, lower-cased, contains `needle` (the rig's rule)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return any(needle in line.lower() for line in fh)
    except OSError:
        return False


def host_attestation() -> dict:
    """The tenancy verdict of this host and the signals it rests on, by the rig's rules.

    Mirrors `attest()` in `runtime/c/bcir_microbench.c` exactly: a hypervisor flag in
    /proc/cpuinfo, a hypervisor node, a DMI vendor/product naming a hypervisor, or WSL is
    "virtualized"; a container marker alone is "containerized"; an exposed hardware PMU
    with no such signal is "bare-metal"; anything else is "unproven". Returns the fields
    the rig prints under the same names.
    """
    hypervisor_flag = _file_contains("/proc/cpuinfo", " hypervisor")
    wsl = _file_contains("/proc/version", "microsoft") or _readable(
        "/proc/sys/fs/binfmt_misc/WSLInterop"
    )
    container = (
        _readable("/.dockerenv")
        or _readable("/run/.containerenv")
        or any(_file_contains("/proc/1/cgroup", marker) for marker in _CONTAINER_CGROUP_MARKERS)
    )
    hypervisor_node = (
        _readable("/sys/hypervisor/type")
        or _readable("/proc/device-tree/hypervisor/compatible")
        or _readable("/proc/xen/capabilities")
    )
    dmi_vendor = _first_line("/sys/class/dmi/id/sys_vendor")
    dmi_product = _first_line("/sys/class/dmi/id/product_name")
    dmi_virtual = any(
        key in dmi_vendor.lower() or key in dmi_product.lower() for key in _DMI_VIRTUAL
    )
    pmu_source = next(
        (s for s in _PMU_SOURCES if _readable(f"/sys/bus/event_source/devices/{s}/type")),
        "none",
    )
    hardware_pmu = pmu_source != "none"
    signals = [
        name
        for flag, name in (
            (hypervisor_flag, "hypervisor-flag"),
            (hypervisor_node, "hypervisor-node"),
            (dmi_virtual, "dmi-virtual"),
            (wsl, "wsl"),
            (container, "container"),
        )
        if flag
    ]
    if hypervisor_flag or hypervisor_node or dmi_virtual or wsl:
        tenancy = "virtualized"
    elif container:
        tenancy = "containerized"
    elif hardware_pmu:
        tenancy = "bare-metal"
    else:
        tenancy = "unproven"
    if signals:
        signal_text = ",".join(signals)
    else:
        signal_text = f"pmu={pmu_source}" if hardware_pmu else "no PMU exposed"
    return {
        "tenancy": tenancy,
        "signals": signal_text,
        "hardware_pmu": hardware_pmu,
        "pmu_source": pmu_source,
        "dmi_vendor": dmi_vendor,
        "dmi_product": dmi_product,
    }


@dataclass(frozen=True)
class NativeEvidence:
    """What the native rig measured and where: the summary a reader sees is derived
    from evidence it can check. `tenancy` is the rig's verdict and `signals` the
    evidence it rests on; the `*_ns` tuples are the raw samples (one per repeat, nothing
    discarded) and the `*_stat` tuples their (min, median, max, MAD). `unique_*` is the
    census of elements each regime touched -- counted, not assumed."""

    tenancy: str
    signals: str
    hardware_pmu: bool
    pmu_source: str
    perf_event_paranoid: int
    cpufreq_governor: str
    rapl: bool
    clocksource: str
    timer_quantum_ns: int
    dmi_vendor: str
    dmi_product: str
    os: str
    arch: str
    compiler: str
    n: int
    repeats: int
    stride: int
    unique_stream: int
    unique_strided: int
    unique_random: int
    working_set_bytes: int
    outlier_policy: str
    stream_ns: tuple
    strided_ns: tuple
    random_ns: tuple
    compute_ns: tuple
    stream_stat: tuple
    strided_stat: tuple
    random_stat: tuple
    compute_stat: tuple

    @property
    def silicon(self) -> bool:
        """True only for the tenancy the evidence proved: bare metal with a PMU."""
        return self.tenancy == "bare-metal"

    @staticmethod
    def from_dict(d) -> "NativeEvidence":
        """Validate the rig's evidence object: a closed field set, bounded strings, i63
        integers, one sample per repeat, and statistics that are re-derived from the
        samples rather than trusted (a summary that disagrees with its evidence is
        refused, not seated)."""
        fields = set(NativeEvidence.__dataclass_fields__)
        if not isinstance(d, dict) or set(d) != fields:
            raise ValueError(f"native evidence fields must be exactly {sorted(fields)}")
        out: dict = {}
        for key in (
            "tenancy",
            "signals",
            "pmu_source",
            "cpufreq_governor",
            "clocksource",
            "dmi_vendor",
            "dmi_product",
            "os",
            "arch",
            "compiler",
            "outlier_policy",
        ):
            out[key] = _bounded_str(d[key], f"native evidence {key}")
        if out["tenancy"] not in TENANCIES:
            raise ValueError(f"native evidence tenancy must be one of {TENANCIES}")
        for key in ("hardware_pmu", "rapl"):
            if not isinstance(d[key], bool):
                raise ValueError(f"native evidence {key} must be a boolean")
            out[key] = d[key]
        if out["tenancy"] == "bare-metal" and not out["hardware_pmu"]:
            raise ValueError("native evidence claims bare-metal without a hardware PMU")
        out["perf_event_paranoid"] = _u63(d["perf_event_paranoid"], "perf_event_paranoid", -99, 99)
        out["timer_quantum_ns"] = _u63(d["timer_quantum_ns"], "timer_quantum_ns")
        out["n"] = _u63(d["n"], "native evidence n", 2, 1 << 24)
        out["repeats"] = _u63(d["repeats"], "native evidence repeats", 1, _MAX_REPEATS)
        out["stride"] = _u63(d["stride"], "native evidence stride", 1)
        for key in ("unique_stream", "unique_strided", "unique_random"):
            out[key] = _u63(d[key], f"native evidence {key}", 0, out["n"])
        out["working_set_bytes"] = _u63(d["working_set_bytes"], "working_set_bytes")
        if out["working_set_bytes"] != 8 * out["unique_strided"]:
            raise ValueError("native evidence working_set_bytes is not 8 * unique_strided")
        for regime in ("stream", "strided", "random", "compute"):
            samples = d[f"{regime}_ns"]
            if not isinstance(samples, list) or len(samples) != out["repeats"]:
                raise ValueError(f"native evidence {regime}_ns must carry one sample per repeat")
            samples = tuple(_u63(x, f"native evidence {regime}_ns sample", 1) for x in samples)
            stat = d[f"{regime}_stat"]
            if not isinstance(stat, list) or len(stat) != 4:
                raise ValueError(f"native evidence {regime}_stat must be (min, median, max, mad)")
            stat = tuple(_u63(x, f"native evidence {regime}_stat") for x in stat)
            median = _median_of(samples)
            derived = (
                min(samples),
                median,
                max(samples),
                _median_of(abs(x - median) for x in samples),
            )
            if stat != derived:
                raise ValueError(
                    f"native evidence {regime}_stat {stat} is not the statistic of its samples "
                    f"{derived}"
                )
            out[f"{regime}_ns"] = samples
            out[f"{regime}_stat"] = stat
        return NativeEvidence(**out)

    def q8_ratios(self) -> tuple:
        """The (strided, random, compute) Q8 ratios the sample medians imply."""
        base = max(1, self.stream_stat[1])
        return tuple(
            max(Q8, (stat[1] * Q8) // base)
            for stat in (self.strided_stat, self.random_stat, self.compute_stat)
        )


@dataclass(frozen=True)
class MicrobenchRaw:
    """Raw median timings (ns) for the four access regimes. Inputs to the pure
    quantizer; never consumed by the planner directly."""

    stream_ns: int
    strided_ns: int
    random_ns: int
    compute_ns: int
    n: int = 0
    repeats: int = 0


@dataclass(frozen=True)
class CalibratedProfile:
    """A frozen, generation-tagged Q8 cost table for one target (the L1 artifact)."""

    name: str
    cal_gen: int
    samples: int
    provenance: str
    stream_q8: int = Q8  # == 256 by definition (verified, law R8)
    strided_q8: int = Q8
    random_q8: int = 32 * Q8
    compute_q8: int = Q8
    #: The native rig's evidence (S0-F); None for the interpreter harness, a Bayesian
    #: point, or the checked-in reference table. Serialized only when present, so a
    #: table without it keeps its byte-identical JSON form.
    evidence: NativeEvidence | None = None

    @property
    def gather_penalty(self) -> int:
        return max(1, self.random_q8 // Q8)

    @property
    def base_overhead(self) -> int:
        return max(1, (4 * self.strided_q8) // Q8)

    @property
    def mem_unit(self) -> int:
        return 1

    def apply(self, h: TargetProfile) -> TargetProfile:
        """Substitute the measured constants into a target profile (frozen data
        in, frozen data out -- the planner never sees the measurement)."""
        return replace(
            h,
            gather_penalty=self.gather_penalty,
            base_overhead=self.base_overhead,
            mem_unit=self.mem_unit,
            cal_gen=self.cal_gen,
        )

    def to_json(self) -> str:
        d = asdict(self)
        if d.get("evidence") is None:
            del d["evidence"]
        else:
            d["evidence"] = {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in d["evidence"].items()
            }
        return json.dumps(d, indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "CalibratedProfile":
        d = strict_json_loads(text, "calibrated profile")
        fields = {
            "name",
            "cal_gen",
            "samples",
            "provenance",
            "stream_q8",
            "strided_q8",
            "random_q8",
            "compute_q8",
        }
        if not isinstance(d, dict) or set(d) - {"evidence"} != fields:
            raise ValueError(f"calibrated profile fields must be exactly {sorted(fields)}")
        # An absent or null `evidence` is a table without one (the interpreter harness, a
        # Bayesian point, the reference table); anything else must be the rig's object.
        evidence = d.pop("evidence", None)
        if evidence is not None:
            evidence = NativeEvidence.from_dict(evidence)
        for key in ("name", "provenance"):
            value = d[key]
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 4096
                or any(ord(ch) < 0x20 for ch in value)
            ):
                raise ValueError(f"calibrated profile {key} must be a bounded string")
        for key in ("cal_gen", "samples", "stream_q8", "strided_q8", "random_q8", "compute_q8"):
            value = d[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > (1 << 63) - 1
            ):
                raise ValueError(f"calibrated profile {key} must be a non-negative i63")
        if d["stream_q8"] != Q8:
            raise ValueError(f"calibrated profile stream_q8 must be {Q8}")
        if any(d[key] < Q8 for key in ("strided_q8", "random_q8", "compute_q8")):
            raise ValueError("calibrated profile Q8 ratios must not be below the stream baseline")
        if evidence is not None:
            # The table is the summary of its evidence: the ratios must be the ones the
            # sample medians imply, `samples` the number of repeats the rig ran, and the
            # provenance may say "bare-metal" only where the evidence attests it.
            if d["samples"] != evidence.repeats:
                raise ValueError(
                    f"calibrated profile samples {d['samples']} != the evidence's "
                    f"{evidence.repeats} repeats"
                )
            implied = evidence.q8_ratios()
            stated = (d["strided_q8"], d["random_q8"], d["compute_q8"])
            if stated != implied:
                raise ValueError(
                    f"calibrated profile Q8 ratios {stated} are not the ratios its evidence "
                    f"implies {implied}"
                )
            if ("bare-metal" in d["provenance"]) != evidence.silicon:
                raise ValueError(
                    "calibrated profile provenance claims a tenancy the evidence does not attest"
                )
        return CalibratedProfile(**d, evidence=evidence)


def load_table(path: str) -> CalibratedProfile:
    return CalibratedProfile.from_json(read_bounded_text(path, "calibrated profile"))


def reference_table() -> CalibratedProfile:
    """The checked-in ratio-1 reference table (reproduces the seeded constants)."""
    return load_table(os.path.join(TABLES_DIR, "x86_64_reference.json"))


# --- the measurement half (host microbenchmarks; deterministic access orders) ---


def _median(xs: list[int]) -> int:
    s = sorted(xs)
    return s[len(s) // 2]


def strided_order(n: int, stride: int) -> list[int]:
    """The full-cycle strided walk: every element exactly once, at `stride`.

    `[(k * stride) % n for k in range(n)]` visits only n / gcd(n, stride) distinct elements
    (n/16 of a power-of-two buffer at the default stride), then repeats that cycle -- the
    G7 finding. The g = gcd(stride, n) cosets {c, c+stride, c+2*stride, ...} of <stride>
    in Z_n partition Z_n, each of length n/g, so walking them in turn touches each
    element once at the declared stride. `runtime/c/bcir_microbench.c::pass_strided` is
    the C twin of this order, and its census counts the result rather than assuming it.
    """
    if n < 1 or stride < 1:
        raise ValueError("strided_order needs n >= 1 and stride >= 1")
    from math import gcd

    step = stride % n
    g = gcd(stride, n)
    cycle = n // g
    order = []
    for coset in range(g):
        idx = coset
        for _ in range(cycle):
            order.append(idx)
            idx += step
            if idx >= n:
                idx -= n
    return order


def _shuffled(n: int) -> list[int]:
    """A seeded LCG permutation: deterministic across hosts (no `random` import
    state); only the *timing* of the run varies."""
    idx = list(range(n))
    state = 0xBC12
    for i in range(n - 1, 0, -1):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        j = state % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def _time_pass(buf, idx) -> int:
    t0 = time.perf_counter_ns()
    s = 0.0
    for j in idx:
        s += buf[j]
    t1 = time.perf_counter_ns()
    return max(1, t1 - t0) if s >= 0 else 1  # keep `s` live


def run_microbench(n: int = 1 << 18, repeats: int = 5, stride: int = 16) -> MicrobenchRaw:
    """Time the four regimes. All passes touch exactly n indices through the
    same indexed loop, so the indirection cost cancels and the ratio isolates
    the access order."""
    buf = array("d", (float(i % 97) for i in range(n)))
    seq = list(range(n))
    strided = strided_order(n, stride)  # every element once (not n / gcd(n, stride))
    rand = _shuffled(n)

    t_seq = _median([_time_pass(buf, seq) for _ in range(repeats)])
    t_str = _median([_time_pass(buf, strided) for _ in range(repeats)])
    t_rnd = _median([_time_pass(buf, rand) for _ in range(repeats)])

    def _compute_pass() -> int:
        t0 = time.perf_counter_ns()
        s = 1.0
        for _ in range(n):
            s = s * 1.0000001 + 0.5
        t1 = time.perf_counter_ns()
        return max(1, t1 - t0) if s >= 0 else 1

    t_cmp = _median([_compute_pass() for _ in range(repeats)])
    return MicrobenchRaw(
        stream_ns=t_seq, strided_ns=t_str, random_ns=t_rnd, compute_ns=t_cmp, n=n, repeats=repeats
    )


def calibrate_from_raw(raw: MicrobenchRaw, h: TargetProfile, cal_gen: int = 1) -> CalibratedProfile:
    """Quantize raw timings into a frozen Q8 table (pure, deterministic).

    Ratios are floored at 1.0 (Q8 256): an access regime is never cheaper than
    streaming, and a ratio-1 measurement reproduces the seeded constants.
    """
    base = max(1, raw.stream_ns)

    def q8(ns: int) -> int:
        return max(Q8, (ns * Q8) // base)

    return CalibratedProfile(
        name=h.name,
        cal_gen=max(1, cal_gen),
        samples=raw.repeats,
        provenance=f"microbench host={platform.machine()} n={raw.n} repeats={raw.repeats}",
        stream_q8=Q8,
        strided_q8=q8(raw.strided_ns),
        random_q8=q8(raw.random_ns),
        compute_q8=q8(raw.compute_ns),
    )


def calibrate_profile(
    h: TargetProfile, n: int = 1 << 18, repeats: int = 5, cal_gen: int = 1
) -> CalibratedProfile:
    """Measure this host and freeze the table (measure -> quantize -> freeze)."""
    return calibrate_from_raw(run_microbench(n=n, repeats=repeats), h, cal_gen)


# Path to the native C microbench (the measurement half on real silicon -- when its
# attestation says so).
_NATIVE_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "runtime", "c", "bcir_microbench.c")
)


def native_available() -> bool:
    """True iff a C compiler and the native microbench source are both present."""
    from shutil import which

    return os.path.exists(_NATIVE_SRC) and bool(which("clang") or which("cc") or which("gcc"))


def calibrate_native(
    h: TargetProfile,
    n: int = 1 << 22,
    repeats: int = 5,
    cal_gen: int = 1,
    cc: str = None,
    require_baremetal: bool = False,
) -> CalibratedProfile:
    """Measure this host with the **native C microbench** (cache effects are real,
    unlike the interpreter harness) and freeze the Q8 table for `h`.

    Compiles `runtime/c/bcir_microbench.c`, runs it, and parses its JSON (the
    `CalibratedProfile` schema plus the rig's `NativeEvidence`). The table's provenance
    carries the tenancy the rig attested; with `require_baremetal=True` any tenancy the
    evidence did not prove ("virtualized", "containerized", "unproven") is refused with
    the signals that decided it, so a silicon claim cannot rest on a hypervisor's clock.
    Raises `RuntimeError` if no compiler/source. Offline (L2/L3) only -- the frozen
    table is the artifact the planner consumes.
    """
    import subprocess
    import tempfile
    from shutil import which

    cc = cc or which("clang") or which("cc") or which("gcc")
    if cc is None or not os.path.exists(_NATIVE_SRC):
        raise RuntimeError("native microbench needs a C compiler + runtime/c/bcir_microbench.c")
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "mb")
        build = subprocess.run(
            [cc, "-O2", "-std=c11", _NATIVE_SRC, "-o", exe], capture_output=True, text=True
        )
        if build.returncode != 0:
            raise RuntimeError("native microbench build failed:\n" + build.stderr)
        run = subprocess.run(
            [exe, str(n), str(repeats), str(cal_gen)], capture_output=True, text=True
        )
        if run.returncode != 0:
            raise RuntimeError("native microbench run failed:\n" + run.stderr)
        raw = CalibratedProfile.from_json(run.stdout)
    if raw.evidence is None:
        raise RuntimeError("native microbench printed a table without its evidence")
    if require_baremetal and not raw.evidence.silicon:
        raise RuntimeError(
            f"native microbench attests tenancy {raw.evidence.tenancy!r} "
            f"({raw.evidence.signals}): not a bare-metal measurement, refusing the claim"
        )
    # adopt the target's name; keep the measured ratios, evidence + generation.
    return replace(raw, name=h.name, cal_gen=max(1, cal_gen))


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .cost import TARGETS

    p = argparse.ArgumentParser(
        prog="bcir.kbcir.microbench", description="Measure, quantize, and freeze a cost table"
    )
    p.add_argument("--target", default="x86_avx512", choices=sorted(TARGETS))
    p.add_argument("--out", default=None, help="write the frozen table (JSON) here")
    p.add_argument("--n", type=int, default=1 << 18)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--cal-gen", type=int, default=1)
    p.add_argument(
        "--bayes",
        action="store_true",
        help="Bayesian table: conjugate posterior + conformal +/- delta "
        "over --bayes-runs repeated measurements",
    )
    p.add_argument("--bayes-runs", type=int, default=8)
    p.add_argument("--coverage", type=float, default=0.9)
    args = p.parse_args(argv)

    h = TARGETS[args.target]
    if args.bayes:
        from .bayescal import bayes_calibrate

        raws = [run_microbench(n=args.n, repeats=args.repeats) for _ in range(args.bayes_runs)]
        bt = bayes_calibrate(h, raws, coverage=args.coverage, cal_gen=args.cal_gen)
        text = bt.to_json()
        lo, hi = bt.gather_penalty_interval()
        summary = (
            f"[bayescal] cal_gen={bt.cal_gen} gather_penalty={bt.gather_penalty} "
            f"in [{lo},{hi}] @ {args.coverage:.0%} (delta_q8={bt.random_delta_q8})"
        )
    else:
        bt = calibrate_profile(h, n=args.n, repeats=args.repeats, cal_gen=args.cal_gen)
        text = bt.to_json()
        summary = f"[microbench] cal_gen={bt.cal_gen} gather_penalty={bt.gather_penalty}"

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"{summary} -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
