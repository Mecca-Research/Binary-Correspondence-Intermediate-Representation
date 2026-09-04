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
_CALIBRATION_PROVENANCE = frozenset({"measured", "modeled", "none"})
_ENERGY_SOURCES = frozenset({"rapl", "nvml", "hwmon", "none"})
_MAX_MANIFEST_BYTES = 1 << 20
_MAX_STRING = 4096
_MAX_LIST = 256
_MAX_INT = (1 << 63) - 1


def _reject_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value!r}")


def _object(value, where: str, *, fields: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ValueError(f"{where} has " + "; ".join(details))
    return value


def _string(value, where: str, *, allow_empty: bool = True) -> str:
    if (
        not isinstance(value, str)
        or len(value) > _MAX_STRING
        or any(ord(ch) < 0x20 for ch in value)
        or (not allow_empty and not value)
    ):
        qualifier = "a non-empty" if not allow_empty else "a"
        raise ValueError(f"{where} must be {qualifier} bounded string")
    return value


def _integer(value, where: str, *, minimum: int = 0, maximum: int = _MAX_INT) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{where} must be an integer in [{minimum}, {maximum}]")
    return value


def _boolean(value, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be a boolean")
    return value


def _string_list(value, where: str, *, max_items: int = _MAX_LIST) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > max_items:
        raise ValueError(f"{where} must be an array of at most {max_items} strings")
    out = tuple(_string(v, f"{where}[{i}]", allow_empty=False) for i, v in enumerate(value))
    if len(set(out)) != len(out):
        raise ValueError(f"{where} must not contain duplicates")
    return out


def _parse_manifest_document(d: object) -> dict:
    """Validate the v1 wire schema before constructing optimizer/runtime objects.

    Channel manifests are third-party inputs.  Rejecting coercions, duplicate/unknown
    fields, oversized collections, and invalid arithmetic constants here prevents a
    malformed plugin from reaching cost calculations or the global channel registry.
    """
    top = _object(
        d,
        "channel manifest",
        fields={
            "format_version",
            "name",
            "kind",
            "provenance",
            "modeled",
            "arch_match",
            "capabilities",
            "codegen",
            "runtime",
            "calibration",
            "profile",
        },
    )
    _integer(top["format_version"], "format_version", minimum=1, maximum=PLUGIN_FORMAT_VERSION)
    _string(top["name"], "name", allow_empty=False)
    _string(top["kind"], "kind", allow_empty=False)
    _string(top["provenance"], "provenance", allow_empty=False)
    _boolean(top["modeled"], "modeled")
    _string_list(top["arch_match"], "arch_match", max_items=64)
    _string_list(top["capabilities"], "capabilities", max_items=len(CAPABILITY_VOCAB) + 1)

    codegen = _object(top["codegen"], "codegen", fields={"llvm_triple", "e_machine"})
    _string(codegen["llvm_triple"], "codegen.llvm_triple")
    _integer(codegen["e_machine"], "codegen.e_machine", maximum=0xFFFF)

    runtime = _object(
        top["runtime"],
        "runtime",
        fields={
            "perf_syscall_nr",
            "energy_source",
            "thermal_zone_types",
        },
    )
    _integer(runtime["perf_syscall_nr"], "runtime.perf_syscall_nr", maximum=1 << 20)
    energy = _string(runtime["energy_source"], "runtime.energy_source", allow_empty=False)
    if energy not in _ENERGY_SOURCES:
        raise ValueError(f"runtime.energy_source {energy!r} is not supported")
    _string_list(runtime["thermal_zone_types"], "runtime.thermal_zone_types", max_items=64)

    calibration = _object(
        top["calibration"],
        "calibration",
        fields={
            "ref",
            "digest",
            "cal_gen",
            "provenance",
        },
    )
    _string(calibration["ref"], "calibration.ref")
    _string(calibration["digest"], "calibration.digest")
    _integer(calibration["cal_gen"], "calibration.cal_gen")
    cal_prov = _string(calibration["provenance"], "calibration.provenance", allow_empty=False)
    if cal_prov not in _CALIBRATION_PROVENANCE:
        raise ValueError(f"calibration.provenance {cal_prov!r} is not supported")

    profile = _object(
        top["profile"],
        "profile",
        fields={
            "name",
            "triple",
            "cacheline",
            "elem_bytes",
            "lane_widths",
            "warp",
            "scalable",
            "gather_penalty",
            "mem_unit",
            "base_overhead",
            "thermal_density",
            "power_density",
            "per_op_heat",
            "fma",
            "isa_features",
            "affinity_domains",
            "mem_channels",
            "cal_gen",
            "mem_tiers",
        },
    )
    _string(profile["name"], "profile.name", allow_empty=False)
    _string(profile["triple"], "profile.triple", allow_empty=False)
    for key in (
        "cacheline",
        "elem_bytes",
        "gather_penalty",
        "mem_unit",
        "base_overhead",
        "affinity_domains",
        "mem_channels",
    ):
        _integer(profile[key], f"profile.{key}", minimum=1)
    for key in ("warp", "thermal_density", "power_density", "per_op_heat", "cal_gen"):
        _integer(profile[key], f"profile.{key}")
    _boolean(profile["scalable"], "profile.scalable")
    _boolean(profile["fma"], "profile.fma")
    widths = profile["lane_widths"]
    if not isinstance(widths, list) or not widths or len(widths) > 64:
        raise ValueError("profile.lane_widths must be a non-empty array of at most 64 integers")
    parsed_widths = tuple(
        _integer(v, f"profile.lane_widths[{i}]", minimum=1) for i, v in enumerate(widths)
    )
    if parsed_widths[0] != 1 or tuple(sorted(set(parsed_widths))) != parsed_widths:
        raise ValueError("profile.lane_widths must start at 1 and be strictly increasing")
    _string_list(profile["isa_features"], "profile.isa_features")
    tiers = profile["mem_tiers"]
    if not isinstance(tiers, list) or len(tiers) > 64:
        raise ValueError("profile.mem_tiers must be an array of at most 64 tiers")
    tier_names = []
    for i, raw in enumerate(tiers):
        tier = _object(
            raw,
            f"profile.mem_tiers[{i}]",
            fields={
                "name",
                "latency_cyc",
                "bw_factor",
                "lat_factor",
                "capacity",
            },
        )
        tier_names.append(_string(tier["name"], f"profile.mem_tiers[{i}].name", allow_empty=False))
        for key in ("latency_cyc", "bw_factor", "lat_factor"):
            _integer(tier[key], f"profile.mem_tiers[{i}].{key}", minimum=1)
        _integer(tier["capacity"], f"profile.mem_tiers[{i}].capacity")
    if len(set(tier_names)) != len(tier_names):
        raise ValueError("profile.mem_tiers must not contain duplicate names")
    return top


# --- (2) target profile schema: a faithful, declarative serialization of TargetProfile ----------
def _tier_to_schema(t: Tier) -> dict:
    return {
        "name": t.name,
        "latency_cyc": t.latency_cyc,
        "bw_factor": t.bw_factor,
        "lat_factor": t.lat_factor,
        "capacity": t.capacity,
    }


def _tier_from_schema(d: dict) -> Tier:
    return Tier(
        name=d["name"],
        latency_cyc=d["latency_cyc"],
        bw_factor=d["bw_factor"],
        lat_factor=d["lat_factor"],
        capacity=d.get("capacity", 0),
    )


def profile_to_schema(p: TargetProfile) -> dict:
    """The cost model H as JSON — every field the optimizer prices, so a plugin fully declares its
    cost identity (no hidden reference into the built-in registry)."""
    return {
        "name": p.name,
        "triple": p.triple,
        "cacheline": p.cacheline,
        "elem_bytes": p.elem_bytes,
        "lane_widths": list(p.lane_widths),
        "warp": p.warp,
        "scalable": p.scalable,
        "gather_penalty": p.gather_penalty,
        "mem_unit": p.mem_unit,
        "base_overhead": p.base_overhead,
        "thermal_density": p.thermal_density,
        "power_density": p.power_density,
        "per_op_heat": p.per_op_heat,
        "fma": p.fma,
        "isa_features": sorted(p.isa_features),
        "affinity_domains": p.affinity_domains,
        "mem_channels": p.mem_channels,
        "cal_gen": p.cal_gen,
        "mem_tiers": [_tier_to_schema(t) for t in p.mem.tiers],
    }


def schema_to_profile(d: dict) -> TargetProfile:
    return TargetProfile(
        name=d["name"],
        triple=d["triple"],
        cacheline=d.get("cacheline", 64),
        elem_bytes=d.get("elem_bytes", 4),
        lane_widths=tuple(d["lane_widths"]),
        warp=d.get("warp", 0),
        scalable=d.get("scalable", False),
        gather_penalty=d["gather_penalty"],
        mem_unit=d["mem_unit"],
        base_overhead=d["base_overhead"],
        thermal_density=d.get("thermal_density", 0),
        power_density=d.get("power_density", 0),
        per_op_heat=d.get("per_op_heat", 0),
        fma=d.get("fma", True),
        isa_features=frozenset(d.get("isa_features", ())),
        affinity_domains=d.get("affinity_domains", 1),
        mem_channels=d.get("mem_channels", 1),
        cal_gen=d.get("cal_gen", 0),
        mem=MemoryHierarchy(tiers=tuple(_tier_from_schema(t) for t in d.get("mem_tiers", ()))),
    )


# --- (6) calibration artifact: the auditable provenance of a measured cost profile --------------
@dataclass(frozen=True)
class CalibrationArtifact:
    """A reference to the calibration data behind a channel's cost profile. ``provenance`` says
    whether the numbers were *measured* on the part, *modeled*, or absent."""

    ref: str = ""  # path / URI to the calibration record
    digest: str = ""  # content digest (tamper / drift check)
    cal_gen: int = 0  # the calibration generation baked into profile
    provenance: str = "none"  # measured | modeled | none

    def to_dict(self) -> dict:
        return {
            "ref": self.ref,
            "digest": self.digest,
            "cal_gen": self.cal_gen,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationArtifact":
        d = d or {}
        return cls(
            ref=d.get("ref", ""),
            digest=d.get("digest", ""),
            cal_gen=d.get("cal_gen", 0),
            provenance=d.get("provenance", "none"),
        )


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
            "runtime": {
                "perf_syscall_nr": self.runtime.perf_syscall_nr,
                "energy_source": self.runtime.energy_source,
                "thermal_zone_types": list(self.runtime.thermal_zone_types),
            },
            "calibration": self.calibration.to_dict(),
            "profile": profile_to_schema(self.profile),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelManifest":
        d = _parse_manifest_document(d)
        cg = d["codegen"]
        rt = d["runtime"]
        return cls(
            name=d["name"],
            kind=d["kind"],
            profile=schema_to_profile(d["profile"]),
            llvm_triple=cg["llvm_triple"],
            e_machine=cg["e_machine"],
            runtime=RuntimeChannel(
                perf_syscall_nr=rt["perf_syscall_nr"],
                energy_source=rt["energy_source"],
                thermal_zone_types=tuple(rt["thermal_zone_types"]),
            ),
            capabilities=frozenset(d["capabilities"]),
            calibration=CalibrationArtifact.from_dict(d["calibration"]),
            provenance=d["provenance"],
            modeled=d["modeled"],
            arch_match=tuple(d["arch_match"]),
            format_version=d["format_version"],
        )

    # --- schema validation ---
    def validate(self) -> list[str]:
        """Return a list of schema errors ([] == valid). Checked before a plugin joins the tower."""
        errs: list[str] = []
        try:
            _parse_manifest_document(self.to_dict())
        except (TypeError, ValueError) as exc:
            errs.append(str(exc))
            return errs
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
            errs.append(
                f"unknown capabilities {sorted(bad_caps)} (vocab: {sorted(CAPABILITY_VOCAB)})"
            )
        if not self.profile.lane_widths or self.profile.lane_widths[0] != 1:
            errs.append("profile.lane_widths must be non-empty and start at the scalar width 1")
        if self.e_machine != EM_NONE and self.kind != "cpu":
            errs.append(
                f"e_machine set on a non-cpu kind {self.kind!r} (only cpu makes a host ELF)"
            )
        if self.provenance == "real" and self.modeled:
            errs.append("provenance 'real' contradicts modeled=True")
        if self.calibration.cal_gen != self.profile.cal_gen:
            errs.append("calibration.cal_gen must match profile.cal_gen")
        return errs

    # --- (build) a live channel ---
    def to_channel(self) -> HardwareChannel:
        return HardwareChannel(
            name=self.name,
            kind=self.kind,
            profile=self.profile,
            llvm_triple=self.llvm_triple,
            e_machine=self.e_machine,
            runtime=self.runtime,
            arch_match=self.arch_match,
            modeled=self.modeled,
            capabilities=self.capabilities,
        )


def manifest_from_channel(
    ch: HardwareChannel,
    *,
    calibration: CalibrationArtifact | None = None,
    provenance: str | None = None,
) -> ChannelManifest:
    """Express a live channel as a manifest (the built-ins round-trip through this, proving the
    schema is complete). ``provenance`` defaults from the channel's ``modeled`` flag."""
    prov = provenance or ("modeled" if ch.modeled else "real")
    cal = calibration or CalibrationArtifact(
        cal_gen=ch.profile.cal_gen, provenance="modeled" if ch.modeled else "measured"
    )
    return ChannelManifest(
        name=ch.name,
        kind=ch.kind,
        profile=ch.profile,
        llvm_triple=ch.llvm_triple,
        e_machine=ch.e_machine,
        runtime=ch.runtime,
        capabilities=ch.capabilities,
        calibration=cal,
        provenance=prov,
        modeled=ch.modeled,
        arch_match=ch.arch_match,
    )


# --- loading + registration ----------------------------------------------------------------------
def load_manifest(path: str) -> ChannelManifest:
    try:
        with open(path, "rb") as f:
            raw = f.read(_MAX_MANIFEST_BYTES + 1)
        if len(raw) > _MAX_MANIFEST_BYTES:
            raise ValueError(f"channel manifest exceeds {_MAX_MANIFEST_BYTES} bytes")
        doc = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        return ChannelManifest.from_dict(doc)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        RecursionError,
    ) as exc:
        raise ValueError(f"invalid channel manifest {path!r}: {exc}") from exc


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
            group = (
                eps.select(group="bcir.channels")
                if hasattr(eps, "select")
                else eps.get("bcir.channels", [])
            )
            for ep in group:
                obj = ep.load()
                manifest = obj() if callable(obj) else obj
                if isinstance(manifest, ChannelManifest) and not manifest.validate():
                    found.append(register_from_manifest(manifest))
        except Exception:  # noqa: BLE001 -- entry-point discovery is best-effort (not installed, etc.)
            pass
    return found
