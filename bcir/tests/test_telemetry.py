"""CT4: data-DNA telemetry sinks + calibrator + adaptive policy + rehydrate."""

import json
import os
import tempfile

from bcir.examples import vector_add
from bcir.kbcir.calibrate import (
    EwmaCalibrator,
    LinearCalibrator,
    adaptive_policy,
    calibrate_and_replan,
    rehydrate_decide,
)
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.telemetry import DataDNA, FileSink, KafkaSink, ListSink


def _events(thermal, n=3):
    return [DataDNA(segment_id=f"seg{i}", claim_id=1000, cycles=100, bytes=4096,
                    misses=10, thermal=thermal, voltage=0, utilization=70) for i in range(n)]


def test_list_and_file_sinks():
    ls = ListSink()
    for e in _events(50):
        ls.emit(e)
    assert len(ls.events) == 3

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dna.jsonl")
        fs = FileSink(path)
        for e in _events(50):
            fs.emit(e)
        rows = [json.loads(line) for line in open(path)]
        assert len(rows) == 3 and rows[0]["thermal"] == 50 and rows[0]["claim_id"] == 1000


def test_ewma_calibrator_raises_theta():
    cal = EwmaCalibrator(alpha=128)  # 0.5
    t = cal.update(Theta.cool(), _events(80))
    assert t.thermal == 40  # 0.5*80 + 0.5*0


def test_adaptive_policy_switches_with_theta():
    assert adaptive_policy(Theta.cool()).name == "latency"
    assert adaptive_policy(Theta(thermal=80)).name == "energy"
    assert adaptive_policy(Theta(mem_pressure=80)).name == "throughput"


def test_hot_telemetry_replans_to_narrower_lane():
    # Full-trust calibration of hot telemetry flips the adaptive policy to energy,
    # which replans vector_add from vec16 to vec8 on AVX-512.
    h = TargetProfile.x86_avx512()
    theta, policy, res = calibrate_and_replan(
        vector_add(1024), h, _events(100), base_theta=Theta.cool(),
        calibrator=EwmaCalibrator(alpha=256))
    assert theta.thermal == 100
    assert policy.name == "energy"
    assert res.by_claim()[1000].width == 8


def test_linear_calibrator_learns_thermal_model():
    # Ground truth: thermal == utilization. The online linear model should learn it
    # and generalize to an unseen feature vector (prediction error shrinks).
    cal = LinearCalibrator(lr=0.5)
    data = [DataDNA(segment_id="s", claim_id=1, thermal=u, utilization=u) for u in (20, 40, 60, 80)]
    probe = DataDNA(segment_id="p", claim_id=1, thermal=0, utilization=80)
    before = abs(cal.predict(probe) - 80)
    for _ in range(300):
        cal.partial_fit(data)
    after = abs(cal.predict(probe) - 80)
    assert after < before        # it learned
    assert after < 8             # converged near the true thermal=80


def test_linear_calibrator_drives_replan():
    # A trained model fed hot telemetry projects a high Θ.thermal -> energy policy
    # -> vec8 (same interface as EwmaCalibrator, via duck typing).
    cal = LinearCalibrator(lr=0.5)
    hot = _events(95)
    for _ in range(200):
        cal.partial_fit(hot)
    theta, policy, res = calibrate_and_replan(
        vector_add(1024), TargetProfile.x86_avx512(), hot, calibrator=cal)
    assert theta.thermal >= 60 and policy.name == "energy"
    assert res.by_claim()[1000].width == 8


def test_kafka_sink_serializes_to_producer():
    class FakeProducer:
        def __init__(self):
            self.sent = []
            self.flushed = False

        def send(self, topic, value=None):
            self.sent.append((topic, value))

        def flush(self):
            self.flushed = True

    fp = FakeProducer()
    sink = KafkaSink(fp, topic="bcir.dna")
    sink.emit(DataDNA(segment_id="s0", claim_id=7, thermal=50, voltage=10))
    sink.flush()
    assert len(fp.sent) == 1 and fp.sent[0][0] == "bcir.dna"
    payload = json.loads(fp.sent[0][1].decode())
    assert payload["claim_id"] == 7 and payload["thermal"] == 50
    assert fp.flushed


def test_rehydrate_decision():
    cool = Theta.cool()
    assert rehydrate_decide(1, 4, 1, 5, cool) == "replan"   # data_gen drift
    assert rehydrate_decide(1, 4, 2, 4, cool) == "repack"   # map_gen drift
    assert rehydrate_decide(1, 4, 1, 4, Theta(thermal=80)) == "replan"  # thermal drift >= 40
    assert rehydrate_decide(1, 4, 1, 4, Theta(thermal=25)) == "patch"   # 20 <= drift < 40
    assert rehydrate_decide(1, 4, 1, 4, cool) == "keep"     # in sync
