"""Hardware-channel plugin boundary — the stable, declarative format a backend registers through.

A :class:`~bcir.channels.HardwareChannel` already isolates everything hardware-specific about a
backend. This module turns that isolation into a *real plugin boundary*: an FPGA / NVMe / HBM-PIM
(or any future accelerator) extension can be added **without editing the core** by shipping a
``channel.json`` manifest (or registering a Python entry point). The manifest is a versioned,
round-trippable schema with seven explicit sections — exactly the contract a third-party backend
must satisfy:

  1. **identity**        — ``name`` + ``kind`` (cpu/gpu/fpga/accelerator/storage/memory).
  2. **target profile**  — the full K_BCIR cost model H (lane widths, penalties, memory hierarchy):
                            the *only* part the optimizer reasons over (`profile` section).
  3. **codegen identity**— the real LLVM triple + ELF e_machine (`codegen` section).
  4. **runtime signals** — the host-runtime signal-provider contract: the ``perf_event_open`` ABI
                            number, the energy source, the thermal zones (`runtime` section).
  5. **capability**      — the StreamPack-execution capability set (which GEM work it runs), the
                            data-driven routing contract (`capabilities`).
  6. **calibration**     — a calibration-artifact reference + content digest + provenance
                            (`calibration` section), so a measured cost profile is auditable.
  7. **provenance**      — ``provenance`` (real | modeled | simulated) + the ``modeled`` flag, so the
                            tower never mistakes a simulator's numbers for measured silicon.

The built-in channels round-trip through this format byte-for-byte (a test), so the schema is known
complete. Loading is dependency-free: ``channel.json`` files on a search path (or ``$BCIR_CHANNEL_PATH``)
and, when the package is installed, Python entry points in the ``bcir.channels`` group.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

from .channels import (
    CAPABILITY_VOCAB,
    EM_NONE,
    CHANNELS,
    HardwareChannel,
    RuntimeChannel,
    register_channel,
)
from .kbcir.cost import MemoryHierarchy, TargetProfile, Tier

PLUGIN_FORMAT_VERSION = 1
KINDS = frozenset({"cpu", "gpu", "fpga", "accelerator", "storage", "memory"})
PROVENANCE = frozenset({"real", "modeled", "simulated"})


# --- (2) target profile schema: a faithful, declarative serialization of TargetProfile ----------
def _tier_to_schema(t: Tier) -> dict:
    return {"name": t.name, "latency_cyc": t.latency_cyc, "bw_factor": t.bw_factor,
            "lat_factor": t.lat_factor, "capacity": t.capacity}


def _tier_from_schema(d: dict) -> Tier:
    return Tier(name=d["name"], latency_cyc=d["latency_cyc"], bw_factor=d["bw_factor"],
                lat_factor=d["lat_factor"], capacity=d.get("capacity", 0))


def profile_to_schema(p: TargetProfile) -> dict:
    """The cost model H as JSON — every field the optimizer prices, so a plugin fully declares its
    cost identity (no hidden reference into the built-in registry)."""
    return {
        "name": p.name, "triple": p.triple, "cacheline": p.cacheline, "elem_bytes": p.elem_bytes,
        "lane_widths": list(p.lane_widths), "warp": p.warp, "scalable": p.scalable,
        "gather_penalty": p.gather_penalty, "mem_unit": p.mem_unit,
        "base_overhead": p.base_overhead, "thermal_density": p.thermal_density,
        "power_density": p.power_density, "per_op_heat": p.per_op_heat, "fma": p.fma,
        "isa_features": sorted(p.isa_features), "affinity_domains": p.affinity_domains,
        "mem_channels": p.mem_channels, "cal_gen": p.cal_gen,
        "mem_tiers": [_tier_to_schema(t) for t in p.mem.tiers],
    }


def schema_to_profile(d: dict) -> TargetProfile:
    return TargetProfile(
        name=d["name"], triple=d["triple"], cacheline=d.get("cacheline", 64),
        elem_bytes=d.get("elem_bytes", 4), lane_widths=tuple(d["lane_widths"]),
        warp=d.get("warp", 0), scalable=d.get("scalable", False),
        gather_penalty=d["gather_penalty"], mem_unit=d["mem_unit"],
        base_overhead=d["base_overhead"], thermal_density=d.get("thermal_density", 0),
        power_density=d.get("power_density", 0), per_op_heat=d.get("per_op_heat", 0),
        fma=d.get("fma", True), isa_features=frozenset(d.get("isa_features", ())),
        affinity_domains=d.get("affinity_domains", 1), mem_channels=d.get("mem_channels", 1),
        cal_gen=d.get("cal_gen", 0),
        mem=MemoryHierarchy(tiers=tuple(_tier_from_schema(t) for t in d.get("mem_tiers", ()))),
    )


# --- (6) calibration artifact: the auditable provenance of a measured cost profile --------------
@dataclass(frozen=True)
class CalibrationArtifact:
    """A reference to the calibration data behind a channel's cost profile. ``provenance`` says
    whether the numbers were *measured* on the part, *modeled*, or absent."""
    ref: str = ""                                    # path / URI to the calibration record
    digest: str = ""                                 # content digest (tamper / drift check)
    cal_gen: int = 0                                 # the calibration generation baked into profile
    provenance: str = "none"                         # measured | modeled | none

    def to_dict(self) -> dict:
        return {"ref": self.ref, "digest": self.digest, "cal_gen": self.cal_gen,
                "provenance": self.provenance}

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationArtifact":
        d = d or {}
        return cls(ref=d.get("ref", ""), digest=d.get("digest", ""),
                   cal_gen=d.get("cal_gen", 0), provenance=d.get("provenance", "none"))


# --- the manifest: the stable plugin format ------------------------------------------------------
@dataclass(frozen=True)
class ChannelManifest:
    name: str
    kind: str
    profile: TargetProfile
    llvm_triple: str = ""
    e_machine: int = EM_NONE
    runtime: RuntimeChannel = field(default_factory=RuntimeChannel)
    capabilities: frozenset = frozenset()
    calibration: CalibrationArtifact = field(default_factory=CalibrationArtifact)
    provenance: str = "modeled"
    modeled: bool = True
    arch_match: tuple = ()
    format_version: int = PLUGIN_FORMAT_VERSION

    # --- (1) channel.json round-trip ---
    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "name": self.name,
            "kind": self.kind,
            "provenance": self.provenance,
            "modeled": self.modeled,
            "arch_match": list(self.arch_match),
            "capabilities": sorted(self.capabilities),
            "codegen": {"llvm_triple": self.llvm_triple, "e_machine": self.e_machine},
            "runtime": {"perf_syscall_nr": self.runtime.perf_syscall_nr,
                        "energy_source": self.runtime.energy_source,
                        "thermal_zone_types": list(self.runtime.thermal_zone_types)},
            "calibration": self.calibration.to_dict(),
            "profile": profile_to_schema(self.profile),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelManifest":
        cg = d.get("codegen", {})
        rt = d.get("runtime", {})
        return cls(
            name=d["name"], kind=d["kind"],
            profile=schema_to_profile(d["profile"]),
            llvm_triple=cg.get("llvm_triple", ""), e_machine=cg.get("e_machine", EM_NONE),
            runtime=RuntimeChannel(
                perf_syscall_nr=rt.get("perf_syscall_nr", 298),
                energy_source=rt.get("energy_source", "none"),
                thermal_zone_types=tuple(rt.get("thermal_zone_types", ("cpu", "soc")))),
            capabilities=frozenset(d.get("capabilities", ())),
            calibration=CalibrationArtifact.from_dict(d.get("calibration", {})),
            provenance=d.get("provenance", "modeled"), modeled=d.get("modeled", True),
            arch_match=tuple(d.get("arch_match", ())),
            format_version=d.get("format_version", PLUGIN_FORMAT_VERSION))

    # --- schema validation ---
    def validate(self) -> list[str]:
        """Return a list of schema errors ([] == valid). Checked before a plugin joins the tower."""
        errs: list[str] = []
        if self.format_version > PLUGIN_FORMAT_VERSION:
            errs.append(f"format_version {self.format_version} > supported {PLUGIN_FORMAT_VERSION}")
        if not self.name:
            errs.append("name is required")
        if self.kind not in KINDS:
            errs.append(f"kind {self.kind!r} not in {sorted(KINDS)}")
        if self.provenance not in PROVENANCE:
            errs.append(f"provenance {self.provenance!r} not in {sorted(PROVENANCE)}")
        bad_caps = self.capabilities - CAPABILITY_VOCAB
        if bad_caps:
            errs.append(f"unknown capabilities {sorted(bad_caps)} (vocab: {sorted(CAPABILITY_VOCAB)})")
        if not self.profile.lane_widths or self.profile.lane_widths[0] != 1:
            errs.append("profile.lane_widths must be non-empty and start at the scalar width 1")
        if self.e_machine != EM_NONE and self.kind != "cpu":
            errs.append(f"e_machine set on a non-cpu kind {self.kind!r} (only cpu makes a host ELF)")
        if self.provenance == "real" and self.modeled:
            errs.append("provenance 'real' contradicts modeled=True")
        return errs

    # --- (build) a live channel ---
    def to_channel(self) -> HardwareChannel:
        return HardwareChannel(
            name=self.name, kind=self.kind, profile=self.profile, llvm_triple=self.llvm_triple,
            e_machine=self.e_machine, runtime=self.runtime, arch_match=self.arch_match,
            modeled=self.modeled, capabilities=self.capabilities)


def manifest_from_channel(ch: HardwareChannel, *, calibration: CalibrationArtifact | None = None,
                          provenance: str | None = None) -> ChannelManifest:
    """Express a live channel as a manifest (the built-ins round-trip through this, proving the
    schema is complete). ``provenance`` defaults from the channel's ``modeled`` flag."""
    prov = provenance or ("modeled" if ch.modeled else "real")
    cal = calibration or CalibrationArtifact(cal_gen=ch.profile.cal_gen,
                                             provenance="modeled" if ch.modeled else "measured")
    return ChannelManifest(
        name=ch.name, kind=ch.kind, profile=ch.profile, llvm_triple=ch.llvm_triple,
        e_machine=ch.e_machine, runtime=ch.runtime, capabilities=ch.capabilities,
        calibration=cal, provenance=prov, modeled=ch.modeled, arch_match=ch.arch_match)


# --- loading + registration ----------------------------------------------------------------------
def load_manifest(path: str) -> ChannelManifest:
    with open(path, encoding="utf-8") as f:
        return ChannelManifest.from_dict(json.load(f))


def register_from_manifest(src) -> HardwareChannel:
    """Validate + build + register a channel from a manifest, a dict, or a ``channel.json`` path.
    Raises ``ValueError`` on a schema violation, so a broken plugin never silently joins the tower."""
    if isinstance(src, ChannelManifest):
        manifest = src
    elif isinstance(src, dict):
        manifest = ChannelManifest.from_dict(src)
    else:
        manifest = load_manifest(src)
    errs = manifest.validate()
    if errs:
        raise ValueError(f"invalid channel manifest {manifest.name!r}: {'; '.join(errs)}")
    ch = manifest.to_channel()
    register_channel(ch)
    return ch


def discover_plugins(dirs=None, *, entry_points: bool = True) -> list[HardwareChannel]:
    """Find + register channel plugins, without the core knowing them in advance:

      * every ``*.channel.json`` under ``dirs`` (default: ``$BCIR_CHANNEL_PATH`` colon-list, then a
        repo-local ``channels/`` dir), and
      * Python entry points in the ``bcir.channels`` group (only when the package is installed).

    Returns the channels registered. Invalid manifests are skipped (collected via ``validate``)."""
    found: list[HardwareChannel] = []
    search: list[str] = list(dirs or [])
    if not dirs:
        search += [p for p in os.environ.get("BCIR_CHANNEL_PATH", "").split(os.pathsep) if p]
        repo_channels = os.path.join(os.path.dirname(os.path.dirname(__file__)), "channels")
        search.append(repo_channels)
    for d in search:
        for path in sorted(glob.glob(os.path.join(d, "*.channel.json"))):
            manifest = load_manifest(path)
            if not manifest.validate():
                found.append(register_from_manifest(manifest))
    if entry_points:
        try:
            from importlib.metadata import entry_points as _eps  # noqa: PLC0415
            eps = _eps()
            group = eps.select(group="bcir.channels") if hasattr(eps, "select") \
                else eps.get("bcir.channels", [])
            for ep in group:
                obj = ep.load()
                manifest = obj() if callable(obj) else obj
                if isinstance(manifest, ChannelManifest) and not manifest.validate():
                    found.append(register_from_manifest(manifest))
        except Exception:  # noqa: BLE001 -- entry-point discovery is best-effort (not installed, etc.)
            pass
    return found
