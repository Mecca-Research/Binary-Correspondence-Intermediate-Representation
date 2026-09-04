"""Active regret sensing: high-resolution telemetry only where the model is unsure
(high variance), so overhead is minimized while learning is maximized. Integer,
deterministic."""

from bcir.kbcir.sensing import (
    DEFAULT_CV_THRESHOLD_MILLI,
    PathSamples,
    RegretSensor,
    full_resolution_overhead,
    total_overhead,
)


def test_variance_and_cv_are_integer_and_deterministic():
    s = PathSamples(path="p")
    for c in (100, 100, 100, 100):
        s.observe(c)
    assert s.variance == 0 and s.cv_milli == 0  # perfectly stable -> certain
    n = PathSamples(path="q")
    for c in (10, 200, 10, 200):
        n.observe(c)
    assert n.variance > 0 and n.cv_milli > DEFAULT_CV_THRESHOLD_MILLI  # noisy -> uncertain


def test_uncertain_paths_get_high_resolution_stable_ones_go_quiet():
    sensor = RegretSensor()
    for _ in range(4):
        sensor.observe("stable", 1000)  # zero variance
    for c in (100, 900, 120, 880, 90):
        sensor.observe("noisy", c)  # high variance
    by_path = {d.path: d for d in sensor.sense()}
    assert by_path["noisy"].resolution == "high"  # sense the uncertain path
    assert by_path["stable"].resolution in ("low", "off")  # quiet on the confident one
    assert by_path["noisy"].overhead > by_path["stable"].overhead


def test_hires_budget_caps_overhead_to_the_most_uncertain():
    sensor = RegretSensor()
    # five noisy paths, but only a budget of 2 may run high-resolution.
    for i, spread in enumerate([(10, 500), (10, 480), (10, 460), (10, 440), (10, 420)]):
        for _ in range(3):
            sensor.observe(f"p{i}", spread[0])
            sensor.observe(f"p{i}", spread[1])
    decisions = sensor.sense(budget=2)
    highs = [d for d in decisions if d.resolution == "high"]
    assert len(highs) == 2  # capped at budget
    # the two highest-CV paths win the budget (p0/p1 have the widest spread)
    assert {d.path for d in highs} == {"p0", "p1"}


def test_active_gating_beats_logging_everything():
    sensor = RegretSensor()
    for _ in range(4):
        sensor.observe("a", 1000)
        sensor.observe("b", 1000)  # stable
    for c in (50, 950, 40, 960):
        sensor.observe("c", c)  # uncertain
    decisions = sensor.sense()
    assert total_overhead(decisions) < full_resolution_overhead(decisions)


def test_too_few_samples_stays_off():
    sensor = RegretSensor()
    sensor.observe("new", 5)  # a single sample: cannot judge
    d = sensor.sense()[0]
    assert d.resolution == "off" and d.samples == 1


def test_sense_is_deterministic():
    def build():
        s = RegretSensor()
        for c in (10, 300, 12, 280):
            s.observe("x", c)
        for _ in range(3):
            s.observe("y", 500)
        return s.sense()

    assert build() == build()
