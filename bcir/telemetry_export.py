"""T4 — telemetry EXPORT ADAPTERS: the external boundary of the telemetry pipeline.

This is the export half of the BCIR telemetry pipeline (the research's ``.PROBE`` /
aggregation layer, ``docs/kernel/TELEMETRY_PIPELINE_RESEARCH.md`` §6 row T4). It takes BCIR's
telemetry — the **T1** signal registry's readings (``bcir.signal_registry``) plus the
**T3** derived metrics / sensitivity ranking (``bcir.kbcir.telemetry_metrics``) — and
re-expresses it in the two industry-standard shapes plus an out-of-band read:

  1. **Prometheus text exposition shape** (PULL / scrape) — the ``# HELP`` /
     ``# TYPE`` / ``name{labels} value`` line protocol a Prometheus server scrapes
     (:func:`to_prometheus`, :func:`scrape`).
  2. **OpenTelemetry (OTLP) data model** (PUSH) — metric points with an *explicit*
     ``Gauge`` vs monotonic ``Sum`` kind, *explicit* delta-vs-cumulative temporality +
     monotonicity, a ``Resource`` (the channel identity), and labels — serialized as the
     OTLP-shaped JSON/dict the protobuf exporter would carry (:func:`to_otlp`,
     :func:`otlp_to_json`, :func:`export_push`).
  3. **DMTF Redfish** (out-of-band PULL) — produce a Redfish ``MetricReport`` JSON from the
     registry snapshot + the ``MetricDefinition`` resources (the 4-resource split), and
     parse a BMC's ``MetricReport`` back into BCIR :class:`~bcir.signal_registry.Reading`s
     (:func:`to_redfish_metric_report`, :func:`metric_definitions`,
     :func:`parse_redfish_metric_report`).

**The two-truth note (the governing invariant, applied to export).** Export is *read-only
egress of the already-graded L2/L3 signal*. Nothing here is — or can become — an R-law
legality verdict: it reads the T1 registry / T3 metrics and emits text/JSON, never a
:class:`~bcir.verify.Diagnostic`, never touching ``bcir/verify`` or the cost-vector
``DIMS``. The witness (:class:`~bcir.telemetry.TelemetryIntegrity`) is exported as an
record and frame-continuity counters plus ``blind`` so suppression, injection, and
transport anomalies stay *observable downstream* (a Prometheus alert / an OTLP gauge) rather
than vanishing silently at the boundary.

**Honest depth.** There is NO live OTLP collector, NO running Prometheus server, and NO
BMC here. These functions *produce / parse the exposition bytes + JSON* — which IS the
testable contract (the wire shape a scrape returns, the bytes a push POSTs, the JSON a
``MetricReport`` carries). The actual transport — an HTTP ``/metrics`` handler, an OTLP
gRPC/HTTP POST with real protobuf, a Redfish REST client — is a thin, documented adapter
on top of these pure functions, NOT built here (it would add a protobuf / requests
dependency and need a live endpoint). Dependency-free: stdlib ``json`` + ``re`` only.

Built ON (never duplicating or modifying): :mod:`bcir.signal_registry` (T1) for the
definitions → metadata and readings → values, with honest ``None`` → an omitted value
line + a separate availability series; :mod:`bcir.kbcir.telemetry_metrics` (T3) for the
optional derived-metric + sensitivity folding; :mod:`bcir.telemetry` for the integrity
witness. The Prometheus charset, label set, and OTLP temporality/monotonicity rules follow
the conventions in the research §2/§3.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .signal_registry import (
    MetricKind,
    MetricDefinition,
    Reading,
    SignalRegistry,
    Unit,
)

# === shared helpers ==================================================================

# A reading namespace prefix, so every BCIR series is greppable on a shared scrape target.
_PREFIX = "bcir_"

# The Prometheus metric-name charset: [a-zA-Z_:][a-zA-Z0-9_:]*  (label names: no ':').
_PROM_NAME_RE = re.compile(r"[^a-zA-Z0-9_:]")
_PROM_FIRST_RE = re.compile(r"^[^a-zA-Z_:]")

def _is_counter(definition: MetricDefinition) -> bool:
    """Use the definition's declared semantics; exporters never infer from names/units."""
    return definition.metric_kind == MetricKind.COUNTER


def sanitize_metric_name(name: str) -> str:
    """Map a BCIR dotted metric id (``"thermal.pressure"``) to a valid Prometheus metric
    name (``"bcir_thermal_pressure"``): prefix with ``bcir_``, replace every char outside
    ``[a-zA-Z0-9_:]`` with ``_``, and guarantee a legal first char. Deterministic + total
    (never raises). The ``bcir_`` prefix makes the leading-char rule moot, but the regex is
    applied anyway so a hand-passed already-prefixed name is still sanitized."""
    safe = _PROM_NAME_RE.sub("_", name)
    if _PROM_FIRST_RE.match(safe):
        safe = "_" + safe
    return _PREFIX + safe


def _sanitize_label_value(value: str) -> str:
    """Escape a label VALUE per the Prometheus text format: backslash, double-quote, and
    newline are escaped; everything else is allowed in a value (only the metric/label
    *names* are charset-restricted)."""
    return (str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n"))


def _as_snapshot(registry_or_snapshot) -> dict[str, Reading | None]:
    """Accept either a :class:`SignalRegistry` (call ``.snapshot()``) or an already-taken
    snapshot dict ``{name: Reading|None}``. Taking a snapshot once and passing the dict
    keeps an export deterministic (the same readings feed every shape). Deterministic key
    order: a registry snapshot is already name-sorted; a passed dict is sorted here."""
    if isinstance(registry_or_snapshot, SignalRegistry):
        return registry_or_snapshot.snapshot()
    return {k: registry_or_snapshot[k] for k in sorted(registry_or_snapshot)}


def _fmt_value(value) -> str:
    """Render a numeric value for a text/JSON exposition: an ``int`` stays an int (no
    trailing ``.0``), a ``float`` uses ``repr`` (round-trippable). Deterministic."""
    if isinstance(value, bool):           # defensive: a bool is an int in Python
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


# The label set carried on EVERY exported series (the research §2 taxonomy: cost_dim +
# provenance + unit + the channel identity). Order is fixed for deterministic output.
def _labels_for(definition: MetricDefinition, reading: Reading | None, channel: str) -> dict[str, str]:
    """The deterministic label set for a metric: channel identity + the T1 taxonomy
    (cost_dim / provenance / unit). Provenance prefers the READING's actual provenance
    (how THIS sample was obtained) and falls back to the definition's nominal provenance
    when there is no reading (the availability series)."""
    return {
        "channel": channel,
        "cost_dim": definition.cost_dim or "none",
        "provenance": reading.provenance if reading is not None else definition.provenance,
        "unit": definition.unit,
    }


def _render_labels(labels: dict[str, str]) -> str:
    """Render a label dict as Prometheus ``{k="v",...}`` (deterministic key order)."""
    inner = ",".join(f'{k}="{_sanitize_label_value(v)}"' for k, v in labels.items())
    return "{" + inner + "}"


# === 1. Prometheus text exposition shape (PULL) ======================================


def to_prometheus(
    registry_or_snapshot,
    *,
    channel: str = "host",
    derived=None,
    sensitivity=None,
    witness=None,
) -> str:
    """Render the registry snapshot in the Prometheus text exposition shape.

    For each metric, emit a ``# HELP`` line (the definition's description), a ``# TYPE``
    line (``counter`` for a monotonic-counter source — RAPL energy_uj, cycles — else
    ``gauge``), and:

      * an AVAILABLE reading → a ``bcir_<name>{channel,cost_dim,provenance,unit} <value>``
        series with the value;
      * an UNAVAILABLE signal → NO value line (honest: we never fabricate a value), but it
        STILL appears in a ``bcir_signal_up{...} 0`` availability series so the real /
        unavailable split is observable downstream. An available signal contributes
        ``bcir_signal_up{...} 1``.

    Optional folding (read-only, never required):
      * ``derived`` — a ``{field: FieldMetrics}`` map (T3 :func:`derive_all`): each
        non-``None`` figure-of-merit (count/min/max/avg/rms) becomes a
        ``bcir_derived_<field>_<stat>`` gauge.
      * ``sensitivity`` — a list of T3 ``SignalSensitivity`` rows: each row's ``abs_delta``
        becomes a ``bcir_signal_sensitivity{signal=...} <abs_delta>`` gauge (the
        sampling-budget ranking, exported so an operator can see WHERE the budget should go).
      * ``witness`` — a :class:`~bcir.telemetry.TelemetryIntegrity`: exported as the
        record counters, frame rejection/continuity counters, and ``_blind`` gauge so
        suppression/injection/transport anomalies are alertable.

    Metric names are sanitized to the Prometheus charset (:func:`sanitize_metric_name`).
    Output is fully deterministic (sorted metric order, fixed label order), so two scrapes
    of the same snapshot are byte-identical."""
    snap = _as_snapshot(registry_or_snapshot)
    lines: list[str] = []

    # --- the availability series (one type block, every signal up=0|1) ---
    up_name = _PREFIX + "signal_up"
    lines.append(f"# HELP {up_name} 1 if the telemetry source is available on this host, else 0 (honest real/unavailable split).")
    lines.append(f"# TYPE {up_name} gauge")
    for name in snap:
        reading = snap[name]
        # The definition is on the reading when available; reconstruct labels either way.
        if reading is not None:
            definition = reading.definition
        else:
            definition = _definition_only_labels(name)
        labels = _labels_for(definition, reading, channel) if definition else {"channel": channel}
        up = 1 if reading is not None else 0
        lines.append(f"{up_name}{_render_labels(labels)} {up}")

    # --- one value series per AVAILABLE reading ---
    for name in snap:
        reading = snap[name]
        if reading is None:
            continue                                      # unavailable: no value line (honest)
        definition = reading.definition
        metric = sanitize_metric_name(name)
        mtype = "counter" if _is_counter(definition) else "gauge"
        desc = definition.description or definition.name
        labels = _labels_for(definition, reading, channel)
        lines.append(f"# HELP {metric} {_help_text(desc)}")
        lines.append(f"# TYPE {metric} {mtype}")
        lines.append(f"{metric}{_render_labels(labels)} {_fmt_value(reading.value)}")

    # --- optional: T3 derived metrics ---
    if derived:
        for fld in sorted(derived):
            fm = derived[fld]
            stats = fm.as_dict() if hasattr(fm, "as_dict") else dict(fm)
            for stat in ("count", "min", "max", "avg", "rms"):
                v = stats.get(stat)
                if v is None:
                    continue
                metric = sanitize_metric_name(f"derived.{fld}.{stat}")
                lines.append(f"# HELP {metric} T3 derived .MEASURE {stat} of DataDNA field {fld}.")
                lines.append(f"# TYPE {metric} gauge")
                lines.append(f"{metric}{{field=\"{_sanitize_label_value(fld)}\"}} {_fmt_value(v)}")

    # --- optional: T3 sensitivity ranking ---
    if sensitivity:
        sens_name = _PREFIX + "signal_sensitivity"
        lines.append(f"# HELP {sens_name} T3 .SENS |Delta score| of a telemetry signal on the optimized plan cost (sampling-budget rank).")
        lines.append(f"# TYPE {sens_name} gauge")
        for row in sensitivity:
            sig = getattr(row, "signal", None)
            ad = getattr(row, "abs_delta", None)
            if sig is None or ad is None:
                continue
            lines.append(f"{sens_name}{{signal=\"{_sanitize_label_value(sig)}\"}} {_fmt_value(ad)}")

    # --- optional: the integrity witness (suppression observable downstream) ---
    if witness is not None:
        lines.extend(_witness_lines(witness, channel))

    return "\n".join(lines) + "\n"


def _help_text(desc: str) -> str:
    """A ``# HELP`` description is a single line: collapse newlines (the only char that would
    break the line protocol). Backslashes/quotes are legal in HELP text."""
    return str(desc).replace("\n", " ").strip()


def _witness_lines(witness, channel: str) -> list[str]:
    """Render a :class:`~bcir.telemetry.TelemetryIntegrity` as a Prometheus block: the
    record and frame-continuity counters plus ``blind``/frame-order gauges. Tolerant of
    any object exposing those attributes."""
    out: list[str] = []
    lab = f'{{channel="{_sanitize_label_value(channel)}"}}'
    record_fields = (
        ("telemetry_accepted", "accepted", "Telemetry records that passed the documented RT3 schema (the stream up-count)."),
        ("telemetry_rejected", "rejected", "Telemetry records dropped for a schema violation (value injection)."),
        ("telemetry_dropped", "dropped", "Telemetry records evicted upstream before read (ring overwrite / suppression)."),
    )
    frame_fields = (
        ("telemetry_frames_accepted", "frames_accepted", "BTLM telemetry frames that passed flags/version/length/CRC validation."),
        ("telemetry_frames_rejected", "frames_rejected", "Magic-aligned telemetry candidates rejected for flags/version/length/CRC."),
        ("telemetry_frames_missing", "frames_missing", "Telemetry frame sequence numbers missing between recovered frames."),
        ("telemetry_frames_reordered", "frames_reordered", "Telemetry frames received behind the sequence watermark."),
        ("telemetry_frames_duplicated", "frames_duplicated", "Telemetry frames repeating the current sequence number."),
    )
    has_frame_evidence = any(getattr(witness, attr, 0) for _, attr, _ in frame_fields)
    fields = record_fields + (frame_fields if has_frame_evidence else ())
    for suffix, attr, help_ in fields:
        val = getattr(witness, attr, None)
        if val is None:
            continue
        metric = _PREFIX + suffix
        out.append(f"# HELP {metric} {help_}")
        out.append(f"# TYPE {metric} counter")
        out.append(f"{metric}{lab} {_fmt_value(val)}")
    frame_monotonic = getattr(witness, "frame_monotonic", None)
    if has_frame_evidence and frame_monotonic is not None:
        metric = _PREFIX + "telemetry_frame_monotonic"
        out.append(f"# HELP {metric} 1 if recovered frame sequence order has no duplicate/reorder, else 0.")
        out.append(f"# TYPE {metric} gauge")
        out.append(f"{metric}{lab} {1 if frame_monotonic else 0}")
    blind = getattr(witness, "blind", None)
    if blind is not None:
        metric = _PREFIX + "telemetry_blind"
        out.append(f"# HELP {metric} 1 if no valid telemetry survived (suppression / all-rejected / all-dropped), else 0.")
        out.append(f"# TYPE {metric} gauge")
        out.append(f"{metric}{lab} {1 if blind else 0}")
    return out


# The registry snapshot does NOT carry a definition for an unavailable (None) signal; to
# label the `up 0` series correctly we look the definition up from the default registry's
# provider map by name. This keeps the availability series fully labelled even for
# absent sources, without re-reading the source.
_DEF_CACHE: dict[str, MetricDefinition] = {}


def _definition_only_labels(name: str) -> MetricDefinition | None:
    """Resolve a metric definition by name for an UNAVAILABLE signal (so its ``up 0`` series
    still carries cost_dim/provenance/unit labels). Looks the provider up in a default
    registry once (cached). Returns ``None`` only for a name no provider declares."""
    if not _DEF_CACHE:
        from .signal_registry import default_registry
        for p in default_registry().providers():
            _DEF_CACHE[p.definition.name] = p.definition
    return _DEF_CACHE.get(name)


def scrape(registry_or_snapshot, **kwargs) -> str:
    """The PULL convenience: render the Prometheus text a scrape would return.
    Thin alias for :func:`to_prometheus` — the function a ``GET /metrics`` HTTP handler
    would call and write to the response body (the handler itself is the unbuilt thin
    adapter; this produces its body, which IS the contract)."""
    return to_prometheus(registry_or_snapshot, **kwargs)


# === 2. OpenTelemetry (OTLP) data model (PUSH) =======================================

# OTLP metric kinds + temporality enums (the data-model vocabulary, research §3 #5).
class OtlpKind:
    GAUGE = "gauge"      # an instantaneous level; non-monotonic; no temporality
    SUM = "sum"          # an accumulating sum; carries temporality + monotonicity


class OtlpTemporality:
    # OTLP's AggregationTemporality. A counter is reported CUMULATIVE (running total since
    # start); a gauge has NO temporality (it is a level, not an aggregate over an interval).
    DELTA = "delta"
    CUMULATIVE = "cumulative"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class OtlpMetric:
    """One OTLP metric (the data-model shape, NOT a protobuf message). Declares the kind +
    temporality + monotonicity EXPLICITLY — the research's "these break silently if
    implicit": a consumer must never have to GUESS whether a number is a level or a running
    total, or whether it resets.

    Rules (decided once, from the T1 definition):
      * a monotonic-counter source (RAPL energy_uj, cycles) → ``kind=sum``,
        ``temporality=cumulative``, ``monotonic=True``;
      * everything else (0..100 pressure, capacity, level, flag) → ``kind=gauge``,
        ``temporality=unspecified``, ``monotonic=False``.

    ``points`` is a list of ``(labels: dict, value)`` — the OTLP NumberDataPoints, each with
    its attributes. ``resource`` is the channel identity (the OTLP ``Resource``)."""

    name: str
    unit: str
    kind: str
    temporality: str
    monotonic: bool
    points: list = field(default_factory=list)        # [(labels: dict, value)]
    resource: str = "host"
    description: str = ""

    def to_dict(self) -> dict:
        """The OTLP-JSON shape for this one metric (the ``Metric`` message body): a
        ``gauge`` or ``sum`` object carrying ``dataPoints`` (each ``{attributes, asInt|
        asDouble}``); a ``sum`` additionally carries ``aggregationTemporality`` +
        ``isMonotonic`` (a gauge carries neither — that is the point)."""
        data_points = [
            {
                "attributes": [{"key": k, "value": {"stringValue": str(v)}}
                               for k, v in labels.items()],
                **_otlp_number(value),
            }
            for labels, value in self.points
        ]
        body: dict = {"dataPoints": data_points}
        if self.kind == OtlpKind.SUM:
            body["aggregationTemporality"] = self.temporality
            body["isMonotonic"] = self.monotonic
        return {
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
            self.kind: body,
        }


def _otlp_number(value) -> dict:
    """An OTLP NumberDataPoint value: ``asInt`` (string per the JSON mapping) for an int,
    ``asDouble`` for a float — the explicit int/double split the OTLP/JSON mapping requires."""
    if isinstance(value, bool):
        return {"asInt": "1" if value else "0"}
    if isinstance(value, int):
        return {"asInt": str(value)}
    return {"asDouble": float(value)}


def _otlp_metric_for(definition: MetricDefinition, reading: Reading, channel: str) -> OtlpMetric:
    """Build the :class:`OtlpMetric` for one available reading, deciding kind/temporality/
    monotonicity from the T1 definition (the same counter/gauge split as Prometheus)."""
    counter = _is_counter(definition)
    labels = _labels_for(definition, reading, channel)
    return OtlpMetric(
        name=sanitize_metric_name(definition.name),
        unit=definition.unit,
        kind=OtlpKind.SUM if counter else OtlpKind.GAUGE,
        temporality=definition.temporality if counter else OtlpTemporality.UNSPECIFIED,
        monotonic=definition.monotonic if counter else False,
        points=[(labels, reading.value)],
        resource=channel,
        description=definition.description or definition.name,
    )


def to_otlp(
    registry_or_snapshot,
    *,
    channel: str = "host",
    derived=None,
    sensitivity=None,
    witness=None,
) -> dict:
    """Build the OTLP-shaped dict (a ``MetricsData`` / ``ResourceMetrics`` structure) for
    the PUSH model — the JSON the protobuf exporter would marshal and POST.

    Shape (the OTLP data model, JSON mapping): a single ``resourceMetrics[0]`` carrying a
    ``resource`` (the channel identity as a Resource attribute) and a ``scopeMetrics[0]``
    whose ``metrics[]`` are one :class:`OtlpMetric`-per-available-reading. Each metric
    declares its kind + temporality + monotonicity explicitly (counters →
    sum/cumulative/monotonic; gauges → gauge/unspecified/non-monotonic). Unavailable
    signals contribute no metric point but DO contribute to a ``bcir_signal_up`` gauge so
    the availability split survives the push too.

    Optional ``derived`` / ``sensitivity`` / ``witness`` fold in as gauges (and the witness
    counters as cumulative sums), mirroring :func:`to_prometheus`. Deterministic order."""
    snap = _as_snapshot(registry_or_snapshot)
    metrics: list[OtlpMetric] = []

    # availability gauge: one metric, one point per signal.
    up_points = []
    for name in snap:
        reading = snap[name]
        definition = reading.definition if reading is not None else _definition_only_labels(name)
        labels = _labels_for(definition, reading, channel) if definition else {"channel": channel}
        up_points.append((labels, 1 if reading is not None else 0))
    metrics.append(OtlpMetric(
        name=_PREFIX + "signal_up", unit=Unit.COUNT, kind=OtlpKind.GAUGE,
        temporality=OtlpTemporality.UNSPECIFIED, monotonic=False, points=up_points,
        resource=channel,
        description="1 if the telemetry source is available on this host, else 0."))

    for name in snap:
        reading = snap[name]
        if reading is None:
            continue
        metrics.append(_otlp_metric_for(reading.definition, reading, channel))

    if derived:
        for fld in sorted(derived):
            fm = derived[fld]
            stats = fm.as_dict() if hasattr(fm, "as_dict") else dict(fm)
            pts = [({"field": fld, "stat": stat}, stats[stat])
                   for stat in ("count", "min", "max", "avg", "rms")
                   if stats.get(stat) is not None]
            if pts:
                metrics.append(OtlpMetric(
                    name=sanitize_metric_name(f"derived.{fld}"), unit=Unit.NONE,
                    kind=OtlpKind.GAUGE, temporality=OtlpTemporality.UNSPECIFIED,
                    monotonic=False, points=pts, resource=channel,
                    description=f"T3 derived .MEASURE figures-of-merit for DataDNA field {fld}."))

    if sensitivity:
        pts = [({"signal": getattr(r, "signal", "")}, getattr(r, "abs_delta", 0))
               for r in sensitivity
               if getattr(r, "signal", None) is not None and getattr(r, "abs_delta", None) is not None]
        if pts:
            metrics.append(OtlpMetric(
                name=_PREFIX + "signal_sensitivity", unit=Unit.NONE, kind=OtlpKind.GAUGE,
                temporality=OtlpTemporality.UNSPECIFIED, monotonic=False, points=pts,
                resource=channel,
                description="T3 .SENS |Delta score| per signal on the optimized plan cost."))

    if witness is not None:
        metrics.extend(_otlp_witness_metrics(witness, channel))

    return {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "bcir.telemetry"}},
                        {"key": "bcir.channel", "value": {"stringValue": channel}},
                    ]
                },
                "scopeMetrics": [
                    {
                        "scope": {"name": "bcir.telemetry_export", "version": "t4"},
                        "metrics": [m.to_dict() for m in metrics],
                    }
                ],
            }
        ]
    }


def _otlp_witness_metrics(witness, channel: str) -> list[OtlpMetric]:
    """The integrity witness as OTLP record/frame sums plus order/blind gauges."""
    out: list[OtlpMetric] = []
    lbl = {"channel": channel}
    record_attrs = ("accepted", "rejected", "dropped")
    frame_attrs = ("frames_accepted", "frames_rejected", "frames_missing", "frames_reordered",
                   "frames_duplicated")
    has_frame_evidence = any(getattr(witness, attr, 0) for attr in frame_attrs)
    for attr in record_attrs + (frame_attrs if has_frame_evidence else ()):
        val = getattr(witness, attr, None)
        if val is None:
            continue
        out.append(OtlpMetric(
            name=_PREFIX + f"telemetry_{attr}", unit=Unit.COUNT, kind=OtlpKind.SUM,
            temporality=OtlpTemporality.CUMULATIVE, monotonic=True,
            points=[(lbl, val)], resource=channel,
            description=f"Telemetry integrity counter: {attr}."))
    frame_monotonic = getattr(witness, "frame_monotonic", None)
    if has_frame_evidence and frame_monotonic is not None:
        out.append(OtlpMetric(
            name=_PREFIX + "telemetry_frame_monotonic", unit=Unit.COUNT,
            kind=OtlpKind.GAUGE, temporality=OtlpTemporality.UNSPECIFIED,
            monotonic=False, points=[(lbl, 1 if frame_monotonic else 0)],
            resource=channel,
            description="1 if recovered frame order has no duplicate/reorder, else 0."))
    blind = getattr(witness, "blind", None)
    if blind is not None:
        out.append(OtlpMetric(
            name=_PREFIX + "telemetry_blind", unit=Unit.COUNT, kind=OtlpKind.GAUGE,
            temporality=OtlpTemporality.UNSPECIFIED, monotonic=False,
            points=[(lbl, 1 if blind else 0)], resource=channel,
            description="1 if no valid telemetry survived (suppression), else 0."))
    return out


def otlp_to_json(registry_or_snapshot_or_dict, *, channel: str = "host", **kwargs) -> bytes:
    """Serialize the OTLP data model to JSON bytes — the PUSH payload a real OTLP/HTTP
    exporter would POST to a collector (``Content-Type: application/json``; the protobuf
    transport is the unbuilt thin adapter).

    Accepts either a registry/snapshot (builds the OTLP dict via :func:`to_otlp`) or an
    already-built OTLP dict (serializes it directly). Deterministic: ``sort_keys=False``
    preserves the data-model order, ``separators`` are compact + stable, so two pushes of
    the same snapshot are byte-identical."""
    if isinstance(registry_or_snapshot_or_dict, dict) and "resourceMetrics" in registry_or_snapshot_or_dict:
        payload = registry_or_snapshot_or_dict
    else:
        payload = to_otlp(registry_or_snapshot_or_dict, channel=channel, **kwargs)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def export_push(registry_or_snapshot, *, channel: str = "host", **kwargs) -> bytes:
    """The PUSH convenience: the OTLP JSON bytes a push exporter would POST. Thin alias for
    :func:`otlp_to_json` — pairs with :func:`scrape` so the "both pull and push" boundary
    is explicit (pull → Prometheus text; push → OTLP bytes)."""
    return otlp_to_json(registry_or_snapshot, channel=channel, **kwargs)


# === 3. DMTF Redfish read adapter (out-of-band PULL) =================================
# The Redfish 4-resource split (research §3 #4): MetricDefinition[T1] (units + provenance
# live HERE) / MetricReportDefinition (cadence + cohort) / MetricReport (the readings) /
# Triggers (thresholds). This module produces the MetricDefinition + MetricReport, and
# parses a foreign MetricReport back into BCIR Readings.

_REDFISH_TYPE_FOR_UNIT = {
    Unit.PERCENT: "Percent",
    Unit.MILLICELSIUS: "Temperature",     # m°C; the value carries the scale
    Unit.MICROJOULE: "EnergyJoules",      # a microjoule energy counter
    Unit.MILLIWATT: "PowerWatts",
    Unit.MICROWATT: "PowerWatts",
    Unit.KHZ: "Frequency",
    Unit.BYTES: "Count",
    Unit.BYTES_PER_SECOND: "Rate",
    Unit.BITMASK: "Count",
    Unit.COUNT: "Count",
    Unit.RATIO_MILLI: "Count",
    Unit.NONE: "Count",
}

# A fixed, deterministic timestamp default for a snapshot-derived report. Redfish wants an
# ISO-8601 Timestamp; an out-of-band export of a point-in-time snapshot uses the caller's
# stamp, defaulting to a sentinel that signals "snapshot, no wall-clock" rather than a
# fabricated time (honest: we do not invent a clock we did not read).
_DEFAULT_TIMESTAMP = "1970-01-01T00:00:00Z"


def metric_definitions(registry_or_snapshot) -> list[dict]:
    """The Redfish ``MetricDefinition`` resources (the 4-split's definition tier) — one per
    metric, carrying the units + provenance + cost_dim + sampling model ON THE DEFINITION
    (Redfish puts the taxonomy here, not on every sample). Built from the T1 registry's
    definitions; deterministic order.

    Accepts a registry (uses every provider's definition, available or not) or a snapshot
    (uses each available reading's definition). The definition tier is the complete
    namespace, so a registry is preferred."""
    if isinstance(registry_or_snapshot, SignalRegistry):
        definitions = [p.definition for p in registry_or_snapshot.providers()]
    else:
        snap = _as_snapshot(registry_or_snapshot)
        definitions = []
        for name in snap:
            r = snap[name]
            d = r.definition if r is not None else _definition_only_labels(name)
            if d is not None:
                definitions.append(d)
    out: list[dict] = []
    for d in definitions:
        out.append({
            "@odata.type": "#MetricDefinition.v1_0_0.MetricDefinition",
            "Id": sanitize_metric_name(d.name),
            "Name": d.name,
            "MetricType": "Counter" if _is_counter(d) else "Gauge",
            "Units": d.unit,
            "MetricDataType": "Integer",
            "Description": _help_text(d.description or d.name),
            # provenance + the cost-dim taxonomy live HERE, on the definition (the 4-split).
            "Oem": {"BCIR": {
                "Provenance": d.provenance,
                "CostDim": d.cost_dim or "none",
                "SamplingModel": d.sampling_model,
                "SignalId": d.signal_id,
                "MetricKind": d.metric_kind,
                "Temporality": d.temporality,
                "Monotonic": d.monotonic,
                "MinIntervalNs": d.min_interval_ns,
            }},
        })
    return out


def to_redfish_metric_report(
    registry_or_snapshot,
    *,
    report_id: str = "BCIRTelemetry",
    timestamp: str | None = None,
) -> dict:
    """Produce a DMTF Redfish ``MetricReport`` (JSON) from the registry snapshot — the
    out-of-band export a BMC would expose at ``/redfish/v1/TelemetryService/MetricReports/``.

    Each AVAILABLE reading becomes a ``MetricValues[]`` entry with ``MetricId`` (the
    sanitized metric name), ``MetricValue`` (a STRING, as Redfish requires), ``Timestamp``,
    and an ``Oem.BCIR`` carrying the provenance + cost_dim (so a downstream consumer keeps
    the taxonomy even off the MetricDefinition). An unavailable signal contributes NO
    MetricValue (honest); its absence is itself the availability signal in the Redfish
    model. Deterministic order. The MetricReportDefinition reference + the MetricDefinition
    set live in the sibling resources (:func:`metric_definitions`); this is the report
    tier."""
    snap = _as_snapshot(registry_or_snapshot)
    ts = timestamp if timestamp is not None else _DEFAULT_TIMESTAMP
    values = []
    for name in snap:
        reading = snap[name]
        if reading is None:
            continue
        d = reading.definition
        values.append({
            "MetricId": sanitize_metric_name(name),
            "MetricProperty": d.name,
            "MetricValue": _fmt_value(reading.value),       # Redfish MetricValue is a string
            "Timestamp": ts,
            "Oem": {"BCIR": {
                "Provenance": reading.provenance,
                "CostDim": d.cost_dim or "none",
                "Unit": d.unit,
            }},
        })
    return {
        "@odata.type": "#MetricReport.v1_4_0.MetricReport",
        "@odata.id": f"/redfish/v1/TelemetryService/MetricReports/{report_id}",
        "Id": report_id,
        "Name": "BCIR Telemetry Metric Report",
        "MetricReportDefinition": {
            "@odata.id": f"/redfish/v1/TelemetryService/MetricReportDefinitions/{report_id}"
        },
        "Timestamp": ts,
        "MetricValues": values,
    }


def parse_redfish_metric_report(report_json) -> list[Reading]:
    """The OUT-OF-BAND READ: parse a Redfish ``MetricReport`` (a BMC's JSON, as a dict or a
    JSON string/bytes) back into BCIR :class:`~bcir.signal_registry.Reading`s. Tolerant of
    missing / extra / foreign fields — a real BMC's report has many keys we ignore, and may
    omit any of the optional ones.

    For each ``MetricValues[]`` entry it recovers: the metric id (``MetricProperty`` — the
    BCIR dotted name — preferred, else ``MetricId``), the numeric value (parsed from the
    string ``MetricValue``), and the provenance/cost_dim/unit from ``Oem.BCIR`` when
    present (a foreign BMC without the Oem still parses — provenance defaults to
    ``measured`` and the unit to a best-effort from any ``MetricDefinition`` lookup, else
    ``none``). An entry whose ``MetricValue`` is missing or non-numeric is skipped (honest:
    we do not fabricate a reading from an unparseable cell). Never raises on a well-formed-
    but-minimal or foreign report; raises only on input that is not JSON at all."""
    report = _coerce_json(report_json)
    if not isinstance(report, dict):
        return []
    raw_values = report.get("MetricValues")
    if not isinstance(raw_values, list):
        return []

    out: list[Reading] = []
    for entry in raw_values:
        if not isinstance(entry, dict):
            continue
        # value: required + numeric, else skip (no fabrication).
        value = _parse_redfish_value(entry.get("MetricValue"))
        if value is None:
            continue
        name = entry.get("MetricProperty") or entry.get("MetricId")
        if not name:
            continue
        oem = (entry.get("Oem") or {}).get("BCIR") or {}
        provenance = oem.get("Provenance", "measured")
        cost_dim = oem.get("CostDim")
        unit = oem.get("Unit")
        # Recover units/cost_dim from the known T1 definition if the report omitted them;
        # else synthesize a tolerant definition so a foreign metric still becomes a Reading.
        known = _definition_only_labels(str(name))
        if known is not None:
            definition = known
        else:
            definition = MetricDefinition(
                name=str(name),
                unit=unit if unit in Unit.ALL else Unit.NONE,
                cost_dim=cost_dim if cost_dim and cost_dim != "none" else None,
                provenance=provenance if provenance in {"measured", "modeled", "simulated"} else "measured",
                description="parsed from a Redfish MetricReport (foreign source).",
            )
        prov = provenance if provenance in {"measured", "modeled", "simulated"} else "measured"
        out.append(Reading(definition=definition, value=value, provenance=prov))
    return out


def _coerce_json(report_json):
    """Accept a dict, a JSON string, or JSON bytes; return the parsed object. A non-JSON
    string raises ``json.JSONDecodeError`` (the one honest hard failure — it is not a
    report at all)."""
    if isinstance(report_json, (str, bytes, bytearray)):
        return json.loads(report_json)
    return report_json


def _parse_redfish_value(raw):
    """Parse a Redfish ``MetricValue`` (a string) into an int or float, or ``None`` if it is
    missing / non-numeric. Prefers int when the string is an integer literal (so a counter
    round-trips as an int, not ``1.0``)."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return None
