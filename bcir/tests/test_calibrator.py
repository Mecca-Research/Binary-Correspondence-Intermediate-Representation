"""Trained calibrator + live broker: the calibration loop's last open half. A
LinearCalibrator trained offline and frozen to deterministic Q8 integers (the
Sec. 13 freeze), and a pub/sub Broker that fans telemetry to subscribers."""

from bcir.examples import vector_add
from bcir.kbcir import TARGETS
from bcir.kbcir.calibloop import close_loop
from bcir.kbcir.calibrate import FrozenCalibrator, train_calibrator
from bcir.telemetry import Broker, DataDNA, ListSink


def _dataset():
    # thermal is perfectly correlated with utilization -> a learnable signal.
    return [
        DataDNA(segment_id="s", claim_id=0, utilization=u, thermal=u) for u in range(0, 100, 10)
    ] * 12


# --- the trained, frozen calibrator ----------------------------------------------


def test_trained_calibrator_learns_the_signal():
    fc = train_calibrator(_dataset(), epochs=400, gen=2)
    assert fc.gen == 2 and fc.samples == len(_dataset())
    # it learned thermal ~ utilization: a high-utilization event predicts hot.
    assert fc.predict(DataDNA(segment_id="s", claim_id=0, utilization=90)) >= 60
    assert fc.predict(DataDNA(segment_id="s", claim_id=0, utilization=10)) <= 40


def test_frozen_calibrator_is_deterministic_and_serializable():
    a = train_calibrator(_dataset(), epochs=300, gen=1)
    b = train_calibrator(_dataset(), epochs=300, gen=1)
    assert a.w_q8 == b.w_q8  # frozen: bit-identical
    assert FrozenCalibrator.from_json(a.to_json()) == a


def test_frozen_predict_is_integer_and_clamped():
    fc = train_calibrator(_dataset(), epochs=200)
    v = fc.predict(DataDNA(segment_id="s", claim_id=0, utilization=100, voltage=100))
    assert isinstance(v, int) and 0 <= v <= 100


# --- the live broker (pub/sub fan-out) -------------------------------------------


def test_broker_fans_out_to_every_subscriber():
    b = Broker()
    s1 = b.subscribe(ListSink())
    s2 = b.subscribe(ListSink())
    for e in _dataset()[:5]:
        b.emit(e)
    assert len(s1.events) == 5 and len(s2.events) == 5
    assert s1.events[0] is s2.events[0]  # same event delivered to both


def test_broker_with_no_subscribers_is_a_noop():
    Broker().emit(DataDNA(segment_id="s", claim_id=0))  # does not raise


def test_broker_bridges_to_kafka_and_local_sinks():
    # The production wiring: the runtime publishes once to the broker, which fans
    # out to a local sink (the calibrator's feed) AND a KafkaSink (the production
    # transport) simultaneously. Tested with a duck-typed producer (no real broker).
    import json

    from bcir.telemetry import KafkaSink

    class FakeProducer:
        def __init__(self):
            self.sent = []

        def send(self, topic, value=None):
            self.sent.append((topic, value))

        def flush(self):
            pass

    fp = FakeProducer()
    b = Broker()
    local = b.subscribe(ListSink())  # local calibrator feed
    b.subscribe(KafkaSink(fp, topic="bcir.dna"))  # production bridge
    b.emit(DataDNA(segment_id="s", claim_id=9, thermal=55))
    b.flush()
    assert len(local.events) == 1  # local subscriber got it
    assert len(fp.sent) == 1 and fp.sent[0][0] == "bcir.dna"
    assert json.loads(fp.sent[0][1].decode())["claim_id"] == 9  # serialized to Kafka


# --- the loop driven by the trained calibrator (closing the last half) -----------


def test_frozen_calibrator_drives_the_loop():
    fc = train_calibrator(_dataset(), epochs=400)
    hot = [DataDNA(segment_id="s", claim_id=0, utilization=95, voltage=80, thermal=95)] * 4
    cert = close_loop(vector_add(1024), TARGETS["x86_avx512"], events=hot, calibrator=fc)
    assert cert.measured_thermal >= 60  # learned hot from telemetry
    assert cert.replanned and cert.win > 0  # vec16 -> vec8, a measured win
