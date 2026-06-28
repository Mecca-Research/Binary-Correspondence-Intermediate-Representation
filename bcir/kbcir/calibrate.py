"""CT4: ML-guided cost calibration, adaptive policy, and the rehydrate loop.

Observed data-DNA telemetry is folded back into the runtime state Theta (an EWMA
calibrator stands in for a pluggable ML model), an adaptive policy is chosen from
Theta, and a rehydrate decision (keep/patch/repack/replan) is made from generation
drift and thermal drift. This is the AI-guided half of `K_BCIR`: the optimizer's
inputs adapt to observed reality.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Module
from ..telemetry import DataDNA, TelemetryIntegrity, sanitize_events
from .cost import HProfile, Theta
from .realize import RealizationResult, optimize
from .weights import ENERGY, PERF, THROUGHPUT, Policy


def _avg(events: list[DataDNA], attr: str) -> int:
    if not events:
        return 0
    return sum(getattr(e, attr) for e in events) // len(events)


@dataclass(frozen=True)
class EwmaCalibrator:
    """Exponentially-weighted calibrator: new = alpha*observed + (1-alpha)*old.

    `alpha` is in Q8 (256 == 1.0). Stands in for a trained ML cost model.
    """

    alpha: int = 128  # 0.5 in Q8

    def _ewma(self, old: int, observed: int) -> int:
        return (self.alpha * observed + (256 - self.alpha) * old) >> 8

    def update(self, theta: Theta, events: list[DataDNA]) -> Theta:
        # Ingest gate: drop any record that violates the documented 0..100 schema BEFORE
        # it reaches the EWMA, so an injected out-of-range/NaN value cannot skew Theta or
        # trip the policy's >=60 thresholds. Legitimate (in-range) batches are unchanged.
        events, _ = sanitize_events(events)
        if not events:
            return theta
        return Theta(
            thermal=self._ewma(theta.thermal, _avg(events, "thermal")),
            power=theta.power,
            mem_pressure=self._ewma(theta.mem_pressure, _avg(events, "misses")),
            contention=theta.contention,
            noise=theta.noise,
            wear=theta.wear,
            utilization=self._ewma(theta.utilization, _avg(events, "utilization")),
            voltage=self._ewma(theta.voltage, _avg(events, "voltage")),
        )


class LinearCalibrator:
    """An online linear model (SGD) that *learns* to predict thermal pressure from
    telemetry features (utilization, voltage, misses) -- a real learning algorithm
    behind the same calibrator interface as `EwmaCalibrator`, dependency-free.

    Unlike the fixed EWMA, this fits weights from observed data: each `update`
    `partial_fit`s the model on the batch, then projects Θ from the learned model.
    A trained model generalizes (predicts thermal for unseen feature vectors).
    """

    NFEAT = 4  # utilization, voltage, misses, bias

    def __init__(self, lr: float = 0.2):
        self.lr = lr
        self.w = [0.0] * self.NFEAT
        self.steps = 0

    @staticmethod
    def _features(e: DataDNA) -> list[float]:
        return [e.utilization / 100.0, e.voltage / 100.0, e.misses / 100.0, 1.0]

    def predict(self, e: DataDNA) -> float:
        """Predicted thermal pressure (0..100) for an event's features."""
        return sum(wi * xi for wi, xi in zip(self.w, self._features(e))) * 100.0

    def partial_fit(self, events: list[DataDNA]) -> None:
        """One SGD pass over the batch, minimizing squared thermal-prediction error."""
        for e in events:
            x = self._features(e)
            err = (self.predict(e) - e.thermal) / 100.0  # scaled residual
            for i in range(self.NFEAT):
                self.w[i] -= self.lr * err * x[i]
            self.steps += 1

    def update(self, theta: Theta, events: list[DataDNA]) -> Theta:
        # Ingest gate: a poisoned record (out-of-range thermal / NaN feature) would
        # corrupt the SGD fit and the projected Theta; drop it at the boundary first.
        events, _ = sanitize_events(events)
        if not events:
            return theta
        self.partial_fit(events)
        clamp = lambda v: max(0, min(100, int(round(v))))
        pred = sum(self.predict(e) for e in events) / len(events)
        return Theta(
            thermal=clamp(pred),
            power=theta.power,
            mem_pressure=clamp(_avg(events, "misses")),
            contention=theta.contention,
            noise=theta.noise,
            wear=theta.wear,
            utilization=clamp(_avg(events, "utilization")),
            voltage=clamp(_avg(events, "voltage")),
        )


@dataclass(frozen=True)
class FrozenCalibrator:
    """A trained `LinearCalibrator` frozen to Q8 integer weights -- deterministic
    across hosts (the LangRef Sec. 13 freeze: learning happens offline at L2/L3,
    then anneals to a frozen, generation-tagged integer artifact). Predicts
    thermal pressure from telemetry features and projects Theta, integer-only, so
    it can stand in for `EwmaCalibrator` in the calibration loop on the certified
    rail (`measure_and_close(..., calibrator=frozen)`)."""

    w_q8: tuple[int, ...]      # 4 weights (utilization, voltage, misses, bias) in Q8
    gen: int = 1
    samples: int = 0

    def predict(self, e: DataDNA) -> int:
        """Predicted thermal pressure (0..100), integer (mirrors the trained
        LinearCalibrator's linear model, quantized)."""
        feats = (e.utilization, e.voltage, e.misses, 100)   # 0..100; bias feature == 1.0*100
        acc = sum(wi * fi for wi, fi in zip(self.w_q8, feats)) >> 8
        return max(0, min(100, acc))

    def update(self, theta: Theta, events: list[DataDNA]) -> Theta:
        # Ingest gate: drop out-of-range/non-finite records before they reach the
        # frozen model's prediction average (the certified rail must not fold poison).
        events, _ = sanitize_events(events)
        if not events:
            return theta
        pred = sum(self.predict(e) for e in events) // len(events)
        clamp = lambda v: max(0, min(100, int(v)))
        return Theta(
            thermal=clamp(pred), power=theta.power,
            mem_pressure=clamp(_avg(events, "misses")), contention=theta.contention,
            noise=theta.noise, wear=theta.wear,
            utilization=clamp(_avg(events, "utilization")),
            voltage=clamp(_avg(events, "voltage")),
        )

    def to_json(self) -> str:
        import json
        return json.dumps({"w_q8": list(self.w_q8), "gen": self.gen, "samples": self.samples},
                          sort_keys=True)

    # The frozen model has exactly NFEAT==4 weights (utilization, voltage, misses, bias).
    # A Q8 weight beyond +/-256*W_ABS_MAX is nonsensical for a model whose features are
    # 0..100 and whose output is clamped to 0..100 -- rejecting it stops a forged
    # certificate from installing adversarial (or non-finite) calibrator weights.
    _NWEIGHTS = 4
    _W_Q8_ABS_MAX = 1 << 20            # |w| <= ~4096.0 in float; far past any trained weight

    @staticmethod
    def from_json(text: str) -> "FrozenCalibrator":
        """Parse + VALIDATE a frozen calibrator. A certificate is an installable
        artifact (its weights drive Theta on the certified rail), so a forged or
        malformed one must be REJECTED, not installed: `w_q8` must be exactly
        `_NWEIGHTS` finite integers within `+/-_W_Q8_ABS_MAX`, and `gen` a
        non-negative int. Any violation raises `ValueError` instead of seating
        adversarial weights."""
        import json
        d = json.loads(text)
        if not isinstance(d, dict) or "w_q8" not in d or "gen" not in d:
            raise ValueError("FrozenCalibrator JSON must be an object with w_q8 and gen")
        w = d["w_q8"]
        if not isinstance(w, (list, tuple)):
            raise ValueError("FrozenCalibrator w_q8 must be an array")
        if len(w) != FrozenCalibrator._NWEIGHTS:
            raise ValueError(
                f"FrozenCalibrator w_q8 must have exactly {FrozenCalibrator._NWEIGHTS} "
                f"weights, got {len(w)}")
        wi_out: list[int] = []
        for i, wi in enumerate(w):
            if isinstance(wi, bool) or not isinstance(wi, int):
                raise ValueError(f"FrozenCalibrator w_q8[{i}] must be an int, got {wi!r}")
            if abs(wi) > FrozenCalibrator._W_Q8_ABS_MAX:
                raise ValueError(
                    f"FrozenCalibrator w_q8[{i}]={wi} exceeds |{FrozenCalibrator._W_Q8_ABS_MAX}| "
                    "(out-of-range / adversarial weight)")
            wi_out.append(wi)
        gen = d["gen"]
        if isinstance(gen, bool) or not isinstance(gen, int) or gen < 0:
            raise ValueError(f"FrozenCalibrator gen must be a non-negative int, got {gen!r}")
        samples = d.get("samples", 0)
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
            raise ValueError(f"FrozenCalibrator samples must be a non-negative int, got {samples!r}")
        return FrozenCalibrator(tuple(wi_out), gen, samples)


def train_calibrator(dataset: list[DataDNA], epochs: int = 300, lr: float = 0.2,
                     gen: int = 1) -> FrozenCalibrator:
    """Train a `LinearCalibrator` on a labeled telemetry dataset (each event's
    `thermal` is the target), then **freeze** the learned weights to Q8 integers.
    Offline (L2/L3, floats during training); the returned artifact is deterministic
    integer and generation-tagged -- the only form that touches the certified rail."""
    model = LinearCalibrator(lr=lr)
    for _ in range(max(1, epochs)):
        model.partial_fit(dataset)
    w_q8 = tuple(int(round(wi * 256)) for wi in model.w)
    return FrozenCalibrator(w_q8=w_q8, gen=max(1, gen), samples=len(dataset))


def adaptive_policy(theta: Theta) -> Policy:
    """Pick a scalarization policy from the live runtime state."""
    if theta.thermal >= 60 or theta.power >= 60 or theta.voltage >= 60:
        return ENERGY          # hot/constrained -> minimize heat/power
    if theta.mem_pressure >= 60 or theta.contention >= 60:
        return THROUGHPUT      # bandwidth-starved -> weight memory/contention
    return PERF                # nominal -> latency


def rehydrate_decide(
    pack_map_gen: int,
    pack_data_gen: int,
    runtime_map_gen: int,
    runtime_data_gen: int,
    theta: Theta,
    baseline_thermal: int = 0,
) -> str:
    """Decide keep / patch / repack / replan for a StreamPack vs runtime state."""
    if pack_data_gen != runtime_data_gen:
        return "replan"        # data changed -> the plan may be invalid
    if pack_map_gen != runtime_map_gen:
        return "repack"        # mapping changed -> re-hydrate the same plan
    drift = abs(theta.thermal - baseline_thermal)
    if drift >= 40:
        return "replan"        # large thermal drift -> a cheaper lane may win
    if drift >= 20:
        return "patch"         # minor drift -> tweak in place
    return "keep"


def calibrate_and_replan(
    module: Module,
    h: HProfile,
    events: list[DataDNA],
    base_theta: Theta = Theta.cool(),
    calibrator: EwmaCalibrator = EwmaCalibrator(),
) -> tuple[Theta, Policy, RealizationResult]:
    """Calibrate Theta from telemetry, pick an adaptive policy, and re-optimize.

    Telemetry is validated at ingest (see `EwmaCalibrator.update`): an out-of-range or
    non-finite record is dropped before it can move Theta or the policy, so the replan
    decision reflects only genuine in-range samples."""
    theta = calibrator.update(base_theta, events)
    policy = adaptive_policy(theta)
    result = optimize(module, h, theta, policy)
    return theta, policy, result


def calibrate_with_witness(
    theta: Theta,
    events: list[DataDNA],
    calibrator: EwmaCalibrator = EwmaCalibrator(),
    *,
    ring_dropped: int = 0,
) -> tuple[Theta, TelemetryIntegrity]:
    """Calibrate Theta AND return the `TelemetryIntegrity` witness for the batch, so a
    suppression/injection attack is observable rather than a silent no-op.

    The plain `calibrator.update` swallows an empty/all-rejected stream (returns Theta
    unchanged) -- correct numerically, but a *blind* calibrator and a *nominal* one are
    indistinguishable to the caller. This surfaces `witness.blind` (no valid record
    survived), `witness.rejected` (records dropped for a schema violation), and
    `witness.dropped` (records the upstream ring evicted, via `ring_dropped`), so the
    decision to NOT throttle is a visible signal. Theta itself is identical to
    `calibrator.update(theta, events)` for any in-range batch."""
    clean, witness = sanitize_events(events, dropped=ring_dropped)
    new_theta = calibrator.update(theta, clean)
    return new_theta, witness
