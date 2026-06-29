"""T1 — the vendor-neutral telemetry signal-provider registry (the PAPI-component model).

Asserts the typed-definition/sample split, the honest real/unavailable split (adapting to
THIS host — never assuming a specific signal is present), the plugin seam (register a
custom provider), the channel↔provider power mapping, and the two-truth boundary (the
registry is off the legality path: only Readings/None, never a verdict/Diagnostic)."""

from bcir.channels import host_channel
from bcir.kbcir.cost import DIMS, MemTier
from bcir.signal_registry import (
    BmcPowerProvider,
    CacheCapacityProvider,
    FabricBytesProvider,
    GpuPowerProvider,
    MemBandwidthProvider,
    MetricDefinition,
    Provenance,
    Reading,
    ReliabilityProvider,
    SamplingModel,
    SignalProvider,
    SignalRegistry,
    ThrottleStateProvider,
    Unit,
    default_registry,
    power_provider_for_energy_source,
    registry_for_channel,
    theta_pressures,
)

# the gap providers that must be honest-unavailable on any host without a vendor backend.
_GAP_PROVIDERS = (GpuPowerProvider, BmcPowerProvider, MemBandwidthProvider,
                  FabricBytesProvider, ThrottleStateProvider, ReliabilityProvider)


# --- every definition is valid + the namespace is unique -----------------------------

def test_default_registry_builds_with_valid_unique_definitions():
    reg = default_registry()
    names = reg.names()
    assert names and len(names) == len(set(names))         # unique PAPI namespace
    for p in reg.providers():
        d = p.definition
        assert isinstance(d, MetricDefinition)
        assert d.validate() == []                          # schema-valid
        assert d.unit in Unit.ALL
        assert d.cost_dim is None or d.cost_dim in DIMS    # maps to a real dim or None
        assert d.provenance in Provenance.ALL
        assert d.sampling_model in SamplingModel.ALL


# --- available() and read() ALWAYS agree (the honest contract) -----------------------

def test_available_and_read_always_agree_and_readings_are_well_typed():
    reg = default_registry()
    for p in reg.providers():
        avail = p.available()
        r = p.read()
        # the core honesty invariant: a Reading iff available, None iff not (no fabrication).
        assert (r is None) == (not avail), (p.name, avail, r)
        if r is not None:
            assert isinstance(r, Reading)
            assert r.unit == p.definition.unit             # carried via the definition
            assert r.cost_dim == p.definition.cost_dim
            assert r.name == p.definition.name
            assert isinstance(r.value, (int, float))
            if r.unit == Unit.PERCENT:                     # in-range when present
                assert 0 <= r.value <= 100
            if r.unit in (Unit.BYTES, Unit.KHZ, Unit.COUNT):
                assert r.value >= 0


# --- the gap providers NEVER fabricate (unavailable here) ----------------------------

def test_gap_providers_are_honest_unavailable_and_never_fabricate():
    for cls in _GAP_PROVIDERS:
        p = cls()
        assert p.available() is False
        assert p.read() is None                            # no fabricated value, ever
        assert p.definition.validate() == []
        assert p.definition.cost_dim in DIMS


# --- snapshot() / availability() are one-per-provider and consistent -----------------

def test_snapshot_and_availability_are_per_provider_and_consistent():
    reg = default_registry()
    snap = reg.snapshot()
    avail = reg.availability()
    assert set(snap) == set(reg.names()) == set(avail)     # one entry per provider
    for name in reg.names():
        r = snap[name]
        assert (r is None) == (not avail[name])            # snapshot agrees with availability
        if r is not None:
            # provenance is carried on the definition (Redfish split), surfaced on the read.
            assert r.provenance in Provenance.ALL
            assert r.definition is reg.get(name).definition


# --- dim mapping: providers_for_dim returns the right providers ----------------------

def test_providers_for_dim_maps_correctly():
    reg = default_registry()
    power = {p.name for p in reg.providers_for_dim("power")}
    # RAPL (default in-band) + the GPU/BMC gap power providers all map to `power`.
    assert "power.rapl_energy_uj" in power
    assert "power.gpu_energy_uj" in power
    assert all(reg.get(n).definition.cost_dim == "power" for n in power)
    # a thermal provider is under `thermal`, not `power`.
    thermal = {p.name for p in reg.providers_for_dim("thermal")}
    assert "thermal.pressure" in thermal
    assert thermal.isdisjoint(power)


# --- the plugin seam: reject duplicates, accept a new custom provider ----------------

class _FakeProvider(SignalProvider):
    """A tiny custom provider exercising the plugin seam (joins WITHOUT editing core)."""

    _DEF = MetricDefinition("custom.fake_metric", Unit.COUNT, "accuracy",
                            Provenance.SIMULATED, SamplingModel.POLLED, "a test provider")

    @property
    def definition(self):
        return self._DEF

    def available(self):
        return True

    def read(self):
        return Reading(self._DEF, 7, Provenance.SIMULATED)


def test_register_rejects_duplicate_and_accepts_a_custom_provider():
    reg = default_registry()
    # duplicate name is rejected.
    try:
        reg.register(GpuPowerProvider())   # power.gpu_energy_uj already registered
        raise AssertionError("expected duplicate-name rejection")
    except ValueError:
        pass
    # a fresh custom provider joins and shows up in the snapshot/availability.
    reg.register(_FakeProvider())
    assert "custom.fake_metric" in reg.names()
    r = reg.snapshot()["custom.fake_metric"]
    assert r is not None and r.value == 7 and r.provenance == Provenance.SIMULATED
    assert reg.availability()["custom.fake_metric"] is True


def test_register_rejects_invalid_definition():
    reg = SignalRegistry()

    class _BadUnit(SignalProvider):
        _DEF = MetricDefinition("bad.metric", "not_a_unit", "compute")

        @property
        def definition(self):
            return self._DEF

        def available(self):
            return False

        def read(self):
            return None

    try:
        reg.register(_BadUnit())
        raise AssertionError("expected invalid-definition rejection")
    except ValueError:
        pass


# --- registry_for_channel selects the power provider by energy_source ----------------

def test_registry_for_channel_selects_power_by_energy_source():
    hc = host_channel()
    reg = registry_for_channel(hc)
    src = hc.runtime.energy_source
    expected = power_provider_for_energy_source(src)
    power_names = {p.name for p in reg.providers_for_dim("power")}
    if expected is None:                                   # energy_source == "none"
        # no dedicated in-band power provider; the gap power providers may still be present.
        assert "power.rapl_energy_uj" not in power_names
    else:
        assert expected in power_names                     # the channel's source is registered
    # an explicit x86/rapl channel selects the RAPL provider as its power source.
    from bcir.channels import channel
    rapl_reg = registry_for_channel(channel("x86_avx512"))
    assert channel("x86_avx512").runtime.energy_source == "rapl"
    assert "power.rapl_energy_uj" in {p.name for p in rapl_reg.providers_for_dim("power")}
    # an nvml channel selects the GPU power provider as its source.
    nvml_reg = registry_for_channel(channel("nvidia_ptx"))
    assert channel("nvidia_ptx").runtime.energy_source == "nvml"
    assert nvml_reg.get("power.gpu_energy_uj") is not None


# --- the cache providers expose the real /sys topology (or honest-absent) ------------

def test_cache_capacity_providers_match_silicon_topology():
    from bcir import silicon
    caps = silicon.tier_capacities() or {}
    for tier, name in ((MemTier.L1, "memory.cache_l1_bytes"),
                       (MemTier.L2, "memory.cache_l2_bytes"),
                       (MemTier.L3, "memory.cache_l3_bytes")):
        p = CacheCapacityProvider(tier)
        if tier in caps:
            r = p.read()
            assert r is not None and r.value == caps[tier] and r.unit == Unit.BYTES
        else:
            assert p.available() is False and p.read() is None


# --- the two-truth boundary: off the legality path -----------------------------------

def test_registry_is_off_the_legality_path():
    """Building + reading the registry produces NO Diagnostic and never touches the
    verifier: a provider returns only a Reading-or-None, never a legality object. The
    registry has no verify/R-law surface (the two-truth quarantine, applied to telemetry)."""
    reg = default_registry()
    # no legality / verify surface on the registry or its providers.
    for surface in ("verify", "diagnostics", "legality", "verdict", "check"):
        assert not hasattr(reg, surface), surface
    for p in reg.providers():
        for surface in ("verify", "diagnostics", "legality", "verdict"):
            assert not hasattr(p, surface), (p.name, surface)
    # every read is a Reading or None — never a Diagnostic-like object.
    for r in reg.snapshot().values():
        assert r is None or isinstance(r, Reading)
    # the graded-signal helper feeds theta WITHOUT touching the legality path: pure dict.
    pressures = theta_pressures(reg)
    assert isinstance(pressures, dict)
    for dim, val in pressures.items():
        assert dim in DIMS and 0 <= val <= 100             # 0..100 graded pressure


def test_theta_pressures_only_surfaces_present_percent_signals():
    reg = default_registry()
    pressures = theta_pressures(reg)
    # only PERCENT-typed, present signals are surfaced; absent ones are omitted (honest).
    thermal = reg.get("thermal.pressure")
    if thermal.available():
        assert pressures.get("thermal") == thermal.read().value
    else:
        assert "thermal" not in pressures                  # absent -> omitted, not defaulted


# --- a host-adaptive sanity check on the real/unavailable split ----------------------

def test_host_split_is_honest_and_self_adapting():
    reg = default_registry()
    avail = reg.availability()
    # OS/cache-derived topology is essentially always readable in this sandbox.
    # (cache may be absent on an exotic host; assert the *agreement*, not a fixed value.)
    for name in ("memory.cache_l1_bytes", "compute.cpu_nominal_khz"):
        p = reg.get(name)
        assert (p.read() is None) == (not p.available())
    # the gap providers are unavailable here regardless of host.
    for name in ("power.gpu_energy_uj", "power.bmc_node_milliwatt",
                 "memory.bandwidth_bytes_per_s", "fabric.interconnect_bytes",
                 "contention.throttle_state", "reliability.ecc_rul"):
        assert avail[name] is False


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
