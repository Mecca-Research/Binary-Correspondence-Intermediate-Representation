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
from bcir.telemetry import (DataDNA, DurableLog, FileSink, KafkaSink, ListSink,
                            TelemetryIntegrityError)


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


def test_file_sinks_refuse_symlink_and_replacement_windows():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "target")
        link = os.path.join(d, "link")
        with open(target, "w", encoding="utf-8") as f:
            f.write("sentinel")
        try:
            os.symlink(target, link)
        except (AttributeError, NotImplementedError, OSError):
            pass  # Windows may not grant symlink creation to the test account.
        else:
            try:
                FileSink(link).emit(DataDNA("s", 1))
                raise AssertionError("FileSink must not follow a telemetry symlink")
            except (OSError, TelemetryIntegrityError):
                pass
            assert open(target, encoding="utf-8").read() == "sentinel"

        path = os.path.join(d, "durable.jsonl")
        displaced = os.path.join(d, "displaced.jsonl")
        log = DurableLog(path)
        os.replace(path, displaced)
        with open(path, "w", encoding="utf-8") as f:
            f.write("replacement")
        try:
            log.emit(DataDNA("s", 1))
            raise AssertionError("DurableLog must detect path replacement")
        except TelemetryIntegrityError:
            pass
        assert open(path, encoding="utf-8").read() == "replacement"

        try:
            DurableLog(path)
            raise AssertionError("DurableLog must create a fresh file exclusively")
        except FileExistsError:
            pass


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


def test_durable_log_round_trips_behind_the_live_broker():
    """Release rung 0.3 (durable telemetry): a fake producer publishes through the LIVE
    Broker to BOTH a small TelemetryRing (backpressure: the overflow is COUNTED, never
    silent) and the schema-tagged DurableLog; the log replays every record bit-for-bit
    under the header's declared schema."""
    import os
    import tempfile
    from bcir.telemetry import Broker, DataDNA, DurableLog, TelemetryRing, load_durable_log
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "run.jsonl")
        broker = Broker()
        ring = broker.subscribe(TelemetryRing(capacity=8))
        broker.subscribe(DurableLog(path))
        events = [DataDNA(f"seg{i}", i, cycles=i * 10, thermal=i % 100,
                          provenance=f"p{i}") for i in range(24)]
        for e in events:                                     # the fake producer
            broker.emit(e)
        back = load_durable_log(path)
        assert [b.to_dict() for b in back] == [e.to_dict() for e in events]   # durable, exact
        st = ring.stats
        assert st.written == 24 and st.dropped == 16         # backpressure counted, not hidden


def test_durable_log_refuses_foreign_and_newer_streams():
    import json
    import os
    import tempfile
    from bcir.telemetry import (TELEMETRY_LOG_KIND, DataDNA, DurableLog, load_durable_log)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.jsonl")
        DurableLog(path).emit(DataDNA("s", 1))
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        head = json.loads(lines[0])
        assert head["kind"] == TELEMETRY_LOG_KIND and head["schema"] == 1
        for bad_head, msg in ((dict(head, schema=99), "newer"),
                              (dict(head, kind="not.telemetry"), TELEMETRY_LOG_KIND),
                              (dict(head, fields=["wrong"]), "lies")):
            bad = os.path.join(d, "bad.jsonl")
            with open(bad, "w", encoding="utf-8") as f:
                f.write(json.dumps(bad_head) + "\n" + "\n".join(lines[1:]))
            try:
                load_durable_log(bad)
                raise AssertionError(f"must refuse: {msg}")
            except ValueError as e:
                assert msg in str(e)


def test_durable_log_rejects_ambiguous_poisoned_and_oversized_records():
    import json
    import os
    import tempfile
    from bcir.telemetry import DataDNA, DurableLog, load_durable_log

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.jsonl")
        log = DurableLog(path)
        log.emit(DataDNA("s", 1))
        with open(path, encoding="utf-8") as f:
            header, record = f.read().splitlines()

        bad_records = [
            record.replace('"claim_id": 1', '"claim_id": 1, "claim_id": 2', 1),
            json.dumps({**json.loads(record), "thermal": 10000}),
            json.dumps({**json.loads(record), "unknown": 1}),
        ]
        for bad_record in bad_records:
            with open(path, "w", encoding="utf-8") as f:
                f.write(header + "\n" + bad_record + "\n")
            try:
                load_durable_log(path)
                raise AssertionError("ambiguous/poisoned telemetry record must be rejected")
            except ValueError:
                pass

        with open(path, "wb") as f:
            f.write(header.encode() + b"\n" + b" " * ((1 << 20) + 1) + b"\n")
        try:
            load_durable_log(path)
            raise AssertionError("oversized telemetry lines must be rejected")
        except ValueError as e:
            assert "line exceeds" in str(e)

        try:
            log.emit(DataDNA("bad\nsegment", 2))
            raise AssertionError("the durable sink must reject invalid text fields")
        except ValueError:
            pass
