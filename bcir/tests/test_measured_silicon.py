"""Real-silicon calibration: RAPL energy + on-die thermal probes and the measured
replan certificate. The sandbox exposes no PMU/RAPL/thermal, so these tests (a)
confirm the honest degrade path and (b) FAKE the sysfs reads to exercise the
real-signal code path end to end -- the same path that lights up on a bare-metal
rig with coretemp + intel-rapl."""

import bcir.silicon as silicon
from bcir.examples import vector_add
from bcir.kbcir import TARGETS
from bcir.kbcir.calibloop import measured_replan

AVX = TARGETS["x86_avx512"]


class _FakeSilicon:
    """Patch silicon's sysfs reads to simulate a host with RAPL + a thermal zone at
    `temp_c`; energy advances `energy_step_uj` per read (a positive draw)."""

    def __init__(self, temp_c=85, energy_step_uj=100_000):
        self.temp_milli = str(int(temp_c * 1000))
        self.step = energy_step_uj
        self.energy = 1_000_000

    def __enter__(self):
        self._read, self._listdir = silicon._read, silicon.os.listdir
        silicon._read = self._fake_read
        silicon.os.listdir = self._fake_listdir
        return self

    def __exit__(self, *exc):
        silicon._read = self._read
        silicon.os.listdir = self._listdir

    def _fake_read(self, path):
        if path.endswith("max_energy_range_uj"):
            return "262143328850"
        if path.endswith("energy_uj"):
            self.energy += self.step
            return str(self.energy)
        if "/thermal_zone" in path and path.endswith("/temp"):
            return self.temp_milli
        return None

    def _fake_listdir(self, base):
        if base == silicon._RAPL_BASE:
            return ["intel-rapl:0"]
        if base == silicon._THERMAL_BASE:
            return ["thermal_zone0"]
        raise FileNotFoundError(base)


# --- the honest degrade path (this sandbox) --------------------------------------


def test_sandbox_reports_unavailable_not_faked():
    # no real signals here: probes return None, the certificate is synthetic.
    assert silicon.read_rapl_uj() is None and silicon.thermal_pressure() is None
    mc = measured_replan(vector_add(1024), AVX, samples=4)
    assert not mc.measured and mc.provenance == ("synthetic",) and mc.win == 0
    assert mc.cert.admissible  # win >= 0 by construction


# --- the parsing/sampling logic (faked sysfs) ------------------------------------


def test_thermal_pressure_maps_temperature_to_0_100():
    with _FakeSilicon(temp_c=40):
        assert silicon.thermal_pressure() == 0  # floor
    with _FakeSilicon(temp_c=95):
        assert silicon.thermal_pressure() == 100  # ceil
    with _FakeSilicon(temp_c=67.5):
        assert silicon.thermal_pressure() == 50  # midpoint


def test_rapl_sampler_measures_a_positive_energy_delta():
    with _FakeSilicon(energy_step_uj=250_000):
        assert silicon.rapl_available()
        s = silicon.RaplSampler()
        assert s.available
        assert s.lap_uj() == 250_000  # one step between construction and lap


def test_rapl_wraparound_is_handled():
    with _FakeSilicon() as fk:
        s = silicon.RaplSampler()
        fk.energy = 10  # next read wraps below the start
        delta = s.lap_uj()
        assert delta is not None and delta >= 0  # wrap corrected via max_energy_range


def test_silicon_dna_records_real_signal_provenance():
    with _FakeSilicon(temp_c=85):
        dna, prov = silicon.silicon_dna(lambda: sum(range(2000)), claim_id=7)
        assert "thermal" in prov and "rapl" in prov
        assert dna.thermal == 82  # round((85-40)*100/55) = 82


# --- the measured replan win (faked hot silicon) ---------------------------------


def test_measured_replan_on_hot_silicon_flips_the_plan():
    # a real thermal signal (85 C -> pressure 81) drives the recalibrated planner to
    # the narrower lane on a small kernel; the win is the heat the stale wide plan
    # would have wasted, and the certificate is tagged as genuinely measured.
    with _FakeSilicon(temp_c=85):
        mc = measured_replan(vector_add(1024), AVX, samples=8)
    assert mc.measured and "thermal" in mc.provenance
    assert mc.win >= 0 and mc.cert.admissible
    assert mc.replanned and mc.cert.calibrated_widths != mc.cert.seeded_widths


def test_measured_replan_on_cool_silicon_holds_the_plan():
    with _FakeSilicon(temp_c=42):  # ~3/100 pressure: no replan
        mc = measured_replan(vector_add(1024), AVX, samples=6)
    assert mc.measured and mc.win == 0 and not mc.replanned


def test_summary_reports_the_real_unavailable_split():
    s = silicon.summary()
    assert s["rapl_energy"] is False and s["thermal_zone"] is False  # sandbox
    with _FakeSilicon():
        s2 = silicon.summary()
        assert s2["rapl_energy"] is True and s2["thermal_zone"] is True
