"""T4 — telemetry EXPORT ADAPTERS (Prometheus / OTLP / Redfish), the external boundary.

Asserts the three export shapes against the testable wire contract (the exposition bytes /
JSON they produce + parse — there is no live collector / BMC here, per the honest-depth
note in `bcir.telemetry_export`):

  * **Prometheus/OpenMetrics**: the text parses to valid `# TYPE` / series lines; an
    available reading appears with the right value + the cost_dim/provenance/unit/channel
    labels; an unavailable signal has NO value line but DOES emit `bcir_signal_up{...} 0`;
    metric names match the Prometheus charset regex; output is deterministic; a counter is
    `# TYPE ... counter`, a gauge `gauge`.
  * **OTLP**: `to_otlp` declares the right kind/temporality/monotonicity (counter →
    sum+cumulative+monotonic; gauge → gauge+unspecified+non-monotonic); `otlp_to_json`
    round-trips via `json.loads`; the channel Resource is carried.
  * **Redfish**: `to_redfish_metric_report` covers the available readings with provenance
    in `Oem`; `parse_redfish_metric_report` round-trips encode→decode AND tolerates a
    minimal / foreign report without crashing; `metric_definitions` carries unit+provenance
    on the definition (the 4-split).
  * **Witness export**: a `TelemetryIntegrity` with rejected/blind surfaces as up/accepted/
    rejected series.
  * **Two-truth**: the export path emits no Diagnostic and touches no verifier — it only
    reads the registry/metrics.

Built so it never assumes a specific host signal is present: the tests synthesize a small
in-memory registry with a known-available counter + gauge and a known-unavailable signal,
so they are deterministic on any host (CI has no RAPL/thermal).
"""

import json
import re

from bcir.signal_registry import (
    MetricDefinition,
    Provenance,
    Reading,
    SamplingModel,
    SignalProvider,
    SignalRegistry,
    Unit,
    default_registry,
)
from bcir.telemetry import TelemetryIntegrity
from bcir.telemetry_export import (
    OtlpKind,
    OtlpMetric,
    OtlpTemporality,
    export_push,
    metric_definitions,
    otlp_to_json,
    parse_redfish_metric_report,
    sanitize_metric_name,
    scrape,
    to_otlp,
    to_prometheus,
    to_redfish_metric_report,
)

# the Prometheus metric-name charset, the contract sanitize_metric_name must satisfy.
_PROM_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")


# === a deterministic synthetic registry (host-independent) ===========================
# A counter (energy_uj), a gauge (thermal pressure), and a definitely-unavailable signal.
# These match the real T1 definition shapes (so sanitization / labels are exercised), but
# are forced available/unavailable so the test is identical on every host.

_COUNTER_DEF = MetricDefinition(
    "power.rapl_energy_uj", Unit.MICROJOULE, "power", Provenance.MEASURED,
    SamplingModel.POLLED, "RAPL package energy counter, microjoules (monotonic).")
_GAUGE_DEF = MetricDefinition(
    "thermal.pressure", Unit.PERCENT, "thermal", Provenance.MEASURED,
    SamplingModel.POLLED, "On-die temperature mapped to 0..100 thermal pressure.")
_GAP_DEF = MetricDefinition(
    "fabric.interconnect_bytes", Unit.BYTES, "fabric", Provenance.MEASURED,
    SamplingModel.STREAMED, "Interconnect bytes (gap: needs a fabric backend).")


class _FixedProvider(SignalProvider):
    """A provider returning a fixed Reading (or None when value is None) — deterministic on
    any host so the export contract is tested without depending on real silicon."""

    def __init__(self, definition: MetricDefinition, value, provenance=Provenance.MEASURED):
        self._def = definition
        self._value = value
        self._prov = provenance

    @property
    def definition(self) -> MetricDefinition:
        return self._def

    def available(self) -> bool:
        return self._value is not None

    def read(self):
        return None if self._value is None else Reading(self._def, self._value, self._prov)


def _synthetic_registry() -> SignalRegistry:
    reg = SignalRegistry()
    reg.register(_FixedProvider(_COUNTER_DEF, 123456))     # available counter
    reg.register(_FixedProvider(_GAUGE_DEF, 42))           # available gauge
    reg.register(_FixedProvider(_GAP_DEF, None))           # unavailable gap signal
    return reg


# === Prometheus =====================================================================


def test_prometheus_has_valid_type_and_series_lines():
    text = to_prometheus(_synthetic_registry(), channel="ch0")
    lines = text.splitlines()
    # every TYPE line names a metric + a valid type.
    type_lines = [ln for ln in lines if ln.startswith("# TYPE ")]
    assert type_lines
    for ln in type_lines:
        _, _, metric, mtype = ln.split(" ", 3)
        assert _PROM_NAME.match(metric), metric
        assert mtype in {"counter", "gauge"}
    # every non-comment, non-blank line is a `name{labels} value` series.
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)$", ln)
        assert m, f"not a valid Prometheus series line: {ln!r}"
        assert _PROM_NAME.match(m.group(1))


def test_prometheus_available_reading_has_value_and_labels():
    text = to_prometheus(_synthetic_registry(), channel="ch0")
    # the available gauge appears with its value + the channel/cost_dim/provenance/unit labels.
    line = next(ln for ln in text.splitlines()
                if ln.startswith("bcir_thermal_pressure{"))
    assert ' 42' in line
    assert 'channel="ch0"' in line
    assert 'cost_dim="thermal"' in line
    assert 'provenance="measured"' in line
    assert 'unit="percent"' in line
    # the available counter appears with its value.
    cline = next(ln for ln in text.splitlines()
                 if ln.startswith("bcir_power_rapl_energy_uj{"))
    assert ' 123456' in cline
    assert 'cost_dim="power"' in cline


def test_prometheus_unavailable_signal_has_no_value_but_up_zero():
    text = to_prometheus(_synthetic_registry(), channel="ch0")
    lines = text.splitlines()
    # NO value series for the unavailable gap signal...
    assert not any(ln.startswith("bcir_fabric_interconnect_bytes{") for ln in lines), \
        "an unavailable signal must NOT emit a value line"
    # ...but it DOES appear in the availability series as up=0.
    up_lines = [ln for ln in lines if ln.startswith("bcir_signal_up{")]
    gap_up = next(ln for ln in up_lines if 'cost_dim="fabric"' in ln)
    assert gap_up.endswith(" 0"), gap_up
    # and an available one is up=1.
    therm_up = next(ln for ln in up_lines if 'cost_dim="thermal"' in ln)
    assert therm_up.endswith(" 1")


def test_prometheus_counter_is_counter_gauge_is_gauge():
    text = to_prometheus(_synthetic_registry())
    assert "# TYPE bcir_power_rapl_energy_uj counter" in text
    assert "# TYPE bcir_thermal_pressure gauge" in text


def test_prometheus_metric_names_match_charset():
    text = to_prometheus(_synthetic_registry())
    for ln in text.splitlines():
        if ln.startswith("# TYPE ") or ln.startswith("# HELP "):
            metric = ln.split(" ", 3)[2]
            assert _PROM_NAME.match(metric), metric
        elif ln and not ln.startswith("#"):
            metric = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)", ln).group(1)
            assert _PROM_NAME.match(metric), metric


def test_prometheus_is_deterministic_across_runs():
    snap = _synthetic_registry().snapshot()
    a = to_prometheus(snap, channel="ch0")
    b = to_prometheus(snap, channel="ch0")
    assert a == b


def test_sanitize_metric_name_is_valid_and_prefixed():
    assert sanitize_metric_name("thermal.pressure") == "bcir_thermal_pressure"
    assert sanitize_metric_name("a.b-c/d") == "bcir_a_b_c_d"
    for raw in ("thermal.pressure", "9bad", "weird name!", "x"):
        assert _PROM_NAME.match(sanitize_metric_name(raw)), raw


def test_prometheus_folds_in_derived_and_sensitivity():
    class _FM:
        def as_dict(self):
            return {"count": 3, "min": 1, "max": 9, "avg": 4, "rms": 5}

    class _Sens:
        def __init__(self, signal, ad):
            self.signal, self.abs_delta = signal, ad

    text = to_prometheus(
        _synthetic_registry(),
        derived={"thermal": _FM()},
        sensitivity=[_Sens("thermal.pressure", 7), _Sens("power.rapl_energy_uj", 0)],
    )
    assert "bcir_derived_thermal_max{field=\"thermal\"} 9" in text
    assert "# TYPE bcir_signal_sensitivity gauge" in text
    assert "bcir_signal_sensitivity{signal=\"thermal.pressure\"} 7" in text


def test_scrape_is_to_prometheus():
    snap = _synthetic_registry().snapshot()
    assert scrape(snap, channel="ch0") == to_prometheus(snap, channel="ch0")


# === OTLP ===========================================================================


def test_otlp_counter_is_sum_cumulative_monotonic():
    otlp = to_otlp(_synthetic_registry(), channel="ch0")
    metrics = otlp["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    counter = next(m for m in metrics if m["name"] == "bcir_power_rapl_energy_uj")
    assert "sum" in counter and "gauge" not in counter
    assert counter["sum"]["aggregationTemporality"] == OtlpTemporality.CUMULATIVE
    assert counter["sum"]["isMonotonic"] is True
    # the value is carried as an OTLP NumberDataPoint asInt (string per the JSON mapping).
    dp = counter["sum"]["dataPoints"][0]
    assert dp["asInt"] == "123456"


def test_otlp_gauge_is_gauge_nonmonotonic_no_temporality():
    otlp = to_otlp(_synthetic_registry(), channel="ch0")
    metrics = otlp["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    gauge = next(m for m in metrics if m["name"] == "bcir_thermal_pressure")
    assert "gauge" in gauge and "sum" not in gauge
    # a gauge carries NEITHER temporality NOR monotonicity (the explicit "level, not sum").
    assert "aggregationTemporality" not in gauge["gauge"]
    assert "isMonotonic" not in gauge["gauge"]
    assert gauge["gauge"]["dataPoints"][0]["asInt"] == "42"


def test_otlp_metric_dataclass_rules():
    # the frozen OtlpMetric declares kind/temporality/monotonicity explicitly.
    counter = OtlpMetric("c", Unit.MICROJOULE, OtlpKind.SUM, OtlpTemporality.CUMULATIVE,
                         True, points=[({"k": "v"}, 5)], resource="ch0")
    d = counter.to_dict()
    assert d["sum"]["isMonotonic"] is True
    assert d["sum"]["aggregationTemporality"] == "cumulative"
    gauge = OtlpMetric("g", Unit.PERCENT, OtlpKind.GAUGE, OtlpTemporality.UNSPECIFIED,
                       False, points=[({"k": "v"}, 3)])
    dg = gauge.to_dict()
    assert "isMonotonic" not in dg["gauge"] and "aggregationTemporality" not in dg["gauge"]


def test_otlp_carries_the_channel_resource():
    otlp = to_otlp(_synthetic_registry(), channel="gpu0")
    res_attrs = otlp["resourceMetrics"][0]["resource"]["attributes"]
    chan = next(a for a in res_attrs if a["key"] == "bcir.channel")
    assert chan["value"]["stringValue"] == "gpu0"


def test_otlp_to_json_round_trips():
    snap = _synthetic_registry().snapshot()
    raw = otlp_to_json(snap, channel="ch0")
    assert isinstance(raw, (bytes, bytearray))
    parsed = json.loads(raw)
    assert parsed == to_otlp(snap, channel="ch0")
    # deterministic bytes across runs.
    assert otlp_to_json(snap, channel="ch0") == raw


def test_export_push_is_otlp_json():
    snap = _synthetic_registry().snapshot()
    assert export_push(snap, channel="ch0") == otlp_to_json(snap, channel="ch0")


def test_otlp_availability_gauge_covers_unavailable_signal():
    otlp = to_otlp(_synthetic_registry(), channel="ch0")
    metrics = otlp["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    up = next(m for m in metrics if m["name"] == "bcir_signal_up")
    # one data point per signal (3), the fabric gap one is 0.
    assert len(up["gauge"]["dataPoints"]) == 3
    vals = {tuple(sorted((a["key"], a["value"]["stringValue"]) for a in dp["attributes"])):
            dp["asInt"] for dp in up["gauge"]["dataPoints"]}
    fabric = next(v for k, v in vals.items() if ("cost_dim", "fabric") in k)
    assert fabric == "0"


# === Redfish ========================================================================


def test_redfish_metric_report_covers_available_with_provenance_oem():
    report = to_redfish_metric_report(_synthetic_registry(), report_id="R1")
    assert report["Id"] == "R1"
    ids = {mv["MetricId"] for mv in report["MetricValues"]}
    # the two available signals are present; the unavailable gap is NOT.
    assert "bcir_thermal_pressure" in ids and "bcir_power_rapl_energy_uj" in ids
    assert "bcir_fabric_interconnect_bytes" not in ids
    therm = next(mv for mv in report["MetricValues"] if mv["MetricId"] == "bcir_thermal_pressure")
    assert therm["MetricValue"] == "42"                 # Redfish MetricValue is a string
    assert therm["Oem"]["BCIR"]["Provenance"] == "measured"
    assert therm["Oem"]["BCIR"]["CostDim"] == "thermal"


def test_redfish_round_trip_encode_decode():
    reg = _synthetic_registry()
    report = to_redfish_metric_report(reg)
    readings = parse_redfish_metric_report(report)
    by_name = {r.name: r for r in readings}
    # the two available readings round-trip back, values + provenance preserved.
    assert by_name["thermal.pressure"].value == 42
    assert by_name["thermal.pressure"].provenance == "measured"
    assert by_name["thermal.pressure"].unit == "percent"
    assert by_name["power.rapl_energy_uj"].value == 123456
    assert "fabric.interconnect_bytes" not in by_name      # unavailable: never in the report


def test_redfish_round_trip_via_json_string():
    report = to_redfish_metric_report(_synthetic_registry())
    readings = parse_redfish_metric_report(json.dumps(report))   # accept a JSON string
    assert {r.name for r in readings} == {"thermal.pressure", "power.rapl_energy_uj"}


def test_redfish_parse_tolerates_minimal_foreign_report():
    # A hand-written minimal/foreign BMC report: missing Oem, extra unknown keys, a bad cell.
    foreign = {
        "@odata.type": "#MetricReport.v1_4_0.MetricReport",
        "SomeVendorField": {"nested": True},
        "MetricValues": [
            {"MetricId": "Fan1", "MetricValue": "3200"},                 # no Oem, no property
            {"MetricId": "Volt", "MetricValue": "not_a_number"},         # unparseable -> skipped
            {"MetricId": "NoValue"},                                     # no MetricValue -> skipped
            "a string entry, not a dict",                                # junk -> skipped
            {"MetricProperty": "thermal.pressure", "MetricValue": "55",
             "Oem": {"BCIR": {"Provenance": "measured", "CostDim": "thermal", "Unit": "percent"}}},
        ],
    }
    readings = parse_redfish_metric_report(foreign)
    by_name = {r.name: r for r in readings}
    assert "Fan1" in by_name and by_name["Fan1"].value == 3200       # foreign metric still parsed
    assert by_name["Fan1"].unit == "none"                            # no Oem -> tolerant default
    assert "Volt" not in by_name and "NoValue" not in by_name        # bad cells skipped, no crash
    assert by_name["thermal.pressure"].value == 55


def test_redfish_parse_empty_and_garbage_does_not_crash():
    assert parse_redfish_metric_report({}) == []
    assert parse_redfish_metric_report({"MetricValues": "not a list"}) == []
    assert parse_redfish_metric_report([1, 2, 3]) == []               # not a report dict


def test_metric_definitions_carry_unit_and_provenance_on_definition():
    defs = metric_definitions(_synthetic_registry())
    by_id = {d["Id"]: d for d in defs}
    # the 4-split: units + provenance + cost_dim live on the DEFINITION resource.
    counter = by_id["bcir_power_rapl_energy_uj"]
    assert counter["Units"] == "microjoule"
    assert counter["MetricType"] == "Counter"
    assert counter["Oem"]["BCIR"]["Provenance"] == "measured"
    assert counter["Oem"]["BCIR"]["CostDim"] == "power"
    assert counter["Oem"]["BCIR"]["SamplingModel"] == "polled"
    gauge = by_id["bcir_thermal_pressure"]
    assert gauge["MetricType"] == "Gauge" and gauge["Units"] == "percent"
    # the definition tier is the COMPLETE namespace (includes the unavailable gap signal).
    assert "bcir_fabric_interconnect_bytes" in by_id


# === witness export =================================================================


def test_witness_export_surfaces_up_accepted_rejected():
    # a stream with a rejected record but some accepted -> not blind.
    witness = TelemetryIntegrity(accepted=5, rejected=2, dropped=1)
    text = to_prometheus(_synthetic_registry(), witness=witness)
    assert "bcir_telemetry_accepted{channel=\"host\"} 5" in text
    assert "bcir_telemetry_rejected{channel=\"host\"} 2" in text
    assert "bcir_telemetry_dropped{channel=\"host\"} 1" in text
    assert "bcir_telemetry_blind{channel=\"host\"} 0" in text


def test_witness_blind_surfaces_when_all_rejected():
    blind_witness = TelemetryIntegrity(accepted=0, rejected=3)
    assert blind_witness.blind is True
    text = to_prometheus(_synthetic_registry(), witness=blind_witness)
    assert "bcir_telemetry_blind{channel=\"host\"} 1" in text
    assert "bcir_telemetry_accepted{channel=\"host\"} 0" in text
    # and in OTLP the witness counters are cumulative monotonic sums.
    otlp = to_otlp(_synthetic_registry(), witness=blind_witness)
    metrics = otlp["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    rej = next(m for m in metrics if m["name"] == "bcir_telemetry_rejected")
    assert rej["sum"]["isMonotonic"] is True
    assert rej["sum"]["aggregationTemporality"] == "cumulative"


# === two-truth boundary =============================================================


def test_export_emits_no_diagnostic_and_touches_no_verifier():
    # The export path only READS the registry/metrics; it never produces a Diagnostic and
    # never imports/invokes bcir.verify. The outputs are plain str / dict / bytes / Reading.
    import sys

    text = to_prometheus(_synthetic_registry())
    otlp = to_otlp(_synthetic_registry())
    report = to_redfish_metric_report(_synthetic_registry())
    readings = parse_redfish_metric_report(report)
    assert isinstance(text, str)
    assert isinstance(otlp, dict)
    assert isinstance(report, dict)
    assert all(isinstance(r, Reading) for r in readings)
    # no Diagnostic-shaped object anywhere (a Reading has no `.code` field).
    for r in readings:
        assert not hasattr(r, "code")
    # importing the export module must not drag in the legality verifier.
    import bcir.telemetry_export  # noqa: F401
    # (verify may be loaded by an *earlier* test in the same process; the contract here is
    # that telemetry_export itself never IMPORTS or INVOKES it — checked from the parsed AST
    # so a mention in the docstring/comments does not false-positive.)
    imports, calls = _module_imports_and_calls("bcir.telemetry_export")
    assert not any("verify" in name for name in imports), (
        f"export module must not import the legality verifier: {sorted(imports)}")
    assert "verify" not in calls, "export module must not invoke verify()"


def _module_imports_and_calls(modname: str):
    """Return (imported-module-names, called-function-names) parsed from a module's AST."""
    import ast
    import importlib
    import inspect
    tree = ast.parse(inspect.getsource(importlib.import_module(modname)))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
            imports.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                calls.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                calls.add(fn.attr)
    return imports, calls


def test_export_works_on_the_real_default_registry():
    # Smoke test on the REAL default registry (host-dependent availability): it must produce
    # all three shapes without crashing and stay self-consistent, whatever this host exposes.
    reg = default_registry()
    text = to_prometheus(reg, channel="host")
    otlp = to_otlp(reg, channel="host")
    report = to_redfish_metric_report(reg)
    # every available signal in the snapshot appears as a value line; every signal (avail or
    # not) appears in the up series.
    snap = reg.snapshot()
    n_up = sum(1 for ln in text.splitlines() if ln.startswith("bcir_signal_up{"))
    assert n_up == len(snap)
    # the report's MetricValues count == the available-reading count.
    avail = sum(1 for r in snap.values() if r is not None)
    assert len(report["MetricValues"]) == avail
    # a round-trip of the real report parses back to exactly the available readings.
    parsed = parse_redfish_metric_report(report)
    assert len(parsed) == avail
    assert json.loads(otlp_to_json(otlp)) == otlp


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys

    sys.exit(_run_all())
