"""The hardware-channel plugin boundary (`bcir.channel_plugin`): the stable, declarative format a
backend (FPGA / NVMe / HBM-PIM / a new accelerator) registers through WITHOUT editing the core.

Pins: every built-in round-trips through the manifest (schema is complete), validation rejects
malformed plugins, an external `channel.json` loads + registers + plans through the tower with
capability-based routing, and the built-ins' routing is unchanged.
"""

import json
import os
import tempfile

from bcir import channel_plugin as P
from bcir.channels import (
    CHANNELS,
    _claim_suits_channel,
    channel_suits,
    host_channel,
)
from bcir.kbcir.cost import MemoryHierarchy, TargetProfile, Tier

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _snapshot():
    return set(CHANNELS)


def _restore(before):
    for k in set(CHANNELS) - before:
        CHANNELS.pop(k, None)


def _example_profile() -> TargetProfile:
    return TargetProfile(
        name="tpu", triple="tpu", lane_widths=(1, 128), gather_penalty=512, mem_unit=1,
        base_overhead=24, isa_features=frozenset({"systolic"}), affinity_domains=128,
        mem_channels=24, mem=MemoryHierarchy(tiers=(Tier("sram", 3, 128, 24, 0),)))


def _example_manifest() -> P.ChannelManifest:
    return P.ChannelManifest(
        name="tpu_test", kind="accelerator", profile=_example_profile(),
        capabilities=frozenset({"matmul", "tile", "data_parallel"}),
        calibration=P.CalibrationArtifact(ref="calib/tpu.json", digest="sha256:x",
                                          provenance="modeled"),
        provenance="simulated", modeled=True)


# --- the schema is complete: built-ins round-trip ------------------------------------------------

def test_every_builtin_round_trips_through_the_manifest():
    for name, ch in CHANNELS.items():
        m = P.manifest_from_channel(ch)
        assert not m.validate(), f"{name}: {m.validate()}"
        ch2 = P.ChannelManifest.from_dict(json.loads(json.dumps(m.to_dict()))).to_channel()
        assert ch2.profile == ch.profile, f"{name}: profile not preserved"
        assert (ch2.kind, ch2.llvm_triple, ch2.e_machine, ch2.runtime) == \
               (ch.kind, ch.llvm_triple, ch.e_machine, ch.runtime), f"{name}: identity not preserved"


# --- validation rejects malformed plugins --------------------------------------------------------

def test_validation_catches_each_kind_of_malformed_manifest():
    good = _example_manifest()
    assert good.validate() == []

    from dataclasses import replace
    assert any("kind" in e for e in replace(good, kind="quantum").validate())
    assert any("capab" in e for e in replace(good, capabilities=frozenset({"telepathy"})).validate())
    assert any("format_version" in e for e in replace(good, format_version=999).validate())
    assert any("lane_widths" in e for e in
               replace(good, profile=replace(good.profile, lane_widths=(2, 4))).validate())
    assert any("provenance" in e for e in replace(good, provenance="vibes").validate())
    # e_machine only makes sense on a cpu (host ELF)
    assert any("e_machine" in e for e in replace(good, e_machine=183).validate())
    # 'real' provenance must not also be modeled
    assert any("real" in e for e in replace(good, provenance="real", modeled=True).validate())


def test_register_from_manifest_rejects_invalid_and_leaves_registry_clean():
    before = _snapshot()
    from dataclasses import replace
    bad = replace(_example_manifest(), kind="quantum")
    try:
        try:
            P.register_from_manifest(bad)
        except ValueError as e:
            assert "invalid channel manifest" in str(e)
        else:
            raise AssertionError("invalid manifest should raise")
        assert set(CHANNELS) == before, "a rejected plugin must not enter the registry"
    finally:
        _restore(before)


def test_channel_manifest_loader_rejects_ambiguous_oversized_and_malformed_documents():
    """A channel file is a third-party optimizer/runtime input, not permissive config.

    Duplicate keys, unknown fields, type coercions, invalid arithmetic constants, and
    unbounded input must fail before a HardwareChannel can be constructed or registered.
    """
    good = _example_manifest().to_dict()
    before = _snapshot()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hostile.channel.json")

        encoded = json.dumps(good)
        duplicate = encoded.replace('"format_version": 1',
                                    '"format_version": 1, "format_version": 1', 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(duplicate)
        try:
            P.load_manifest(path)
            raise AssertionError("duplicate JSON keys must be rejected")
        except ValueError as e:
            assert "duplicate JSON key" in str(e)

        with open(path, "wb") as f:
            f.write(b" " * (P._MAX_MANIFEST_BYTES + 1))
        try:
            P.load_manifest(path)
            raise AssertionError("oversized manifests must be rejected")
        except ValueError as e:
            assert "exceeds" in str(e)

        for mutate in (
            lambda d: d.update({"typo_field": 1}),
            lambda d: d["profile"].update({"mem_unit": 0}),
            lambda d: d["profile"].update({"cacheline": True}),
            lambda d: d["profile"].update({"lane_widths": [1, 8, 8]}),
        ):
            doc = json.loads(json.dumps(good))
            mutate(doc)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            try:
                P.register_from_manifest(path)
                raise AssertionError("malformed channel manifest must be rejected")
            except ValueError:
                pass
            assert set(CHANNELS) == before


def test_calibration_generation_must_match_the_profile_generation():
    from dataclasses import replace

    good = _example_manifest()
    forged = replace(good, calibration=replace(good.calibration, cal_gen=7))
    assert any("calibration.cal_gen" in e for e in forged.validate())
    try:
        P.register_from_manifest(forged)
        raise AssertionError("mismatched calibration generation must be rejected")
    except ValueError:
        pass


# --- an external plugin loads + registers + plans + routes ---------------------------------------

def test_register_from_manifest_adds_a_working_channel():
    before = _snapshot()
    try:
        ch = P.register_from_manifest(_example_manifest())
        assert ch.name in CHANNELS and CHANNELS[ch.name] is ch
        assert ch.capabilities == frozenset({"matmul", "tile", "data_parallel"})
        # it plans through the same unified optimizer as any built-in.
        from bcir.kbcir.realize import optimize
        from bcir.kbcir.weights import PERF
        from bcir.model import Module, Phase
        from bcir.kbcir.cost import Theta
        # a trivial module just needs to plan without error on the plugin's profile.
        prog = __import__("bcir.bench", fromlist=["PROGRAMS"]).PROGRAMS["vector_add"]()
        plan = optimize(prog, ch.profile, Theta.cool(), PERF)
        assert plan.steps, "plugin channel produced no plan"
    finally:
        _restore(before)


def test_channel_json_round_trips_on_disk_and_discovers():
    before = _snapshot()
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tpu_test.channel.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_example_manifest().to_dict(), f)
            registered = P.discover_plugins(dirs=[d], entry_points=False)
            assert any(c.name == "tpu_test" for c in registered)
            assert "tpu_test" in CHANNELS
    finally:
        _restore(before)


def test_the_shipped_example_plugin_is_valid():
    path = os.path.join(_REPO, "channels", "example_tpu.channel.json")
    assert os.path.exists(path), "the example channel plugin should ship in channels/"
    m = P.load_manifest(path)
    assert m.validate() == [], m.validate()
    assert m.kind == "accelerator" and "matmul" in m.capabilities


# --- routing: declared capabilities for plugins, legacy rule for built-ins -----------------------

def test_capability_routing_is_identical_to_legacy_for_builtins():
    # built-ins declare no capabilities -> channel_suits must equal the legacy per-kind rule.
    from bcir.bench import PROGRAMS
    from bcir.kbcir.realize import _flatten
    claims = [c for prog in ("vector_add", "gather_reduce", "histogram_gather", "matmul_tiled",
                             "saxpy_strided", "scan")
              for _ph, c in _flatten(PROGRAMS[prog]())]
    for ch in CHANNELS.values():
        assert not ch.capabilities                       # built-ins are kind-routed
        for claim in claims:
            assert channel_suits(claim, ch) == _claim_suits_channel(claim, ch)


def test_declared_capabilities_route_by_tag():
    before = _snapshot()
    try:
        ch = P.register_from_manifest(_example_manifest())  # caps: matmul, tile, data_parallel
        from bcir.channels import claim_required_caps

        class _Claim:
            def __init__(self, op, sc=None):
                self.op, self.stride_class = op, sc
        from bcir.model import StrideClass
        matmul = _Claim("matmul.acc", StrideClass.TILE)
        gather = _Claim("gather.load", StrideClass.RANDOM)
        assert channel_suits(matmul, ch)                 # declares matmul + tile
        assert not channel_suits(gather, ch)             # does not declare gather
        assert "matmul" in claim_required_caps(matmul)
    finally:
        _restore(before)
