"""The CT4 measured-replan rig runbook -- end-to-end, honest in a sandbox.

Roadmap (BCIR_MASTER_ROADMAP.md §5.4 / §6): the measured real-silicon replan win is
**rig-gated** -- it needs a bare-metal host with a hardware PMU + a cpufreq userspace
governor + RAPL energy (docs/kernel/HARDWARE_VALIDATION.md). It cannot be produced in CI or this
sandbox, so the honest deliverable is to make the path **push-button and CI-exercised in
degrade mode**: `tools/silicon/measure_replan.sh` runs the full
read-real-signals → fold-Theta → replan → certify loop, prints a MEASURED win on a rig
(provenance=real), and degrades honestly otherwise (provenance=synthetic, win 0, no
fabricated number). These tests gate that the path never silently rots.
"""

import os
import subprocess
import sys

from bcir.kbcir import TARGETS
from bcir.kbcir.calibloop import measured_replan
from bcir.examples import vector_add
from bcir.silicon import cpufreq_info, perf_counters_available, rapl_available
from bcir.toolchain import host_bash

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNBOOK = os.path.join(_ROOT, "tools", "silicon", "measure_replan.sh")
AVX = TARGETS["x86_avx512"]


def _runbook(*args: str):
    """Run with a real shell. On Windows, avoid the legacy WSL launcher and use
    Git for Windows' Bash; hosts with no Bash capability skip this shell-only gate."""
    bash = host_bash()
    if bash is None:
        return None
    env = os.environ.copy()
    env["BCIR_PYTHON"] = sys.executable
    return subprocess.run(
        [bash, _RUNBOOK, *args], capture_output=True, text=True, cwd=_ROOT, env=env
    )


def _rig_ready() -> bool:
    """The rig contract: the measured win is live iff the host exposes ALL THREE real
    signals -- a hardware PMU, RAPL energy, and a cpufreq userspace governor."""
    return (
        perf_counters_available()
        and rapl_available()
        and bool(getattr(cpufreq_info(), "actuatable", False))
    )


def test_runbook_runs_end_to_end_and_is_honest():
    """The runbook completes; on a rig it reports a MEASURED win, otherwise it degrades to
    a synthetic run (never a fabricated number) and exits 0."""
    r = _runbook()
    if r is None:
        return
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "capability probe" in out and "measured replan" in out and "verdict" in out
    if perf_counters_available():
        assert "MEASURED replan win" in out and "provenance=real" in out
    else:
        # the sandbox/CI case: honest degrade, no measured number claimed.
        assert "degraded (synthetic)" in out and "no MEASURED win" in out


def test_runbook_probe_enumerates_the_three_rig_signals():
    """The capability probe names each of the three gating signals (PMU + RAPL + userspace
    governor) and prints an explicit rig-ready verdict -- so the requirement to fire the
    measured win is crisply specified, never implicit."""
    r = _runbook()
    if r is None:
        return
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "PMU (perf_event_open)" in out
    assert "RAPL energy" in out
    assert "DVFS governor" in out and "userspace" in out
    assert "rig-ready:" in out
    if _rig_ready():
        assert "rig-ready: YES" in out
    else:
        # sandbox/CI: rig-ready NO, and every absent signal is named as missing.
        assert "rig-ready: NO" in out and "missing:" in out


def test_rig_fires_exactly_when_all_three_signals_are_present():
    """`--require-real` is the gate: it passes iff the host is rig-ready (all three signals),
    and fails honestly otherwise -- the measured win fires the moment, and only the moment, a
    bare-metal host with PMU + RAPL + a userspace governor runs the runbook."""
    r = _runbook("--require-real")
    if r is None:
        return
    if _rig_ready():
        assert r.returncode == 0 and "MEASURED replan win" in r.stdout
    else:
        assert r.returncode == 1 and "no real signals" in r.stdout


def test_runbook_require_real_fails_without_a_rig():
    """`--require-real` (used on the rig) must FAIL where the real signals are absent --
    so a CI/sandbox run can never masquerade as a measured result."""
    if perf_counters_available():
        return  # on a real rig this would pass; not assertable here
    r = _runbook("--require-real")
    if r is None:
        return
    assert r.returncode == 1 and "no real signals" in r.stdout


def test_certificate_provenance_is_tagged_honestly():
    """The replan certificate records whether the telemetry was real or synthetic -- the
    integrity tag the runbook keys on. In a sandbox it is synthetic; admissible holds."""
    cert = measured_replan(vector_add(), AVX)
    assert cert.cert.admissible
    if not perf_counters_available():
        assert "synthetic" in cert.provenance and not cert.measured
        assert cert.win == 0  # honest: no measured value without the rig
