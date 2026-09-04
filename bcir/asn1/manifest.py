"""J4 part 3 — schema-bound manifests, and §5.4's two sinks.

`channel_plugin.py` already loads `channel.json`, and it validates carefully: duplicate
keys, unknown fields, oversized collections, coerced types and out-of-range arithmetic are
all refused before a manifest reaches the channel registry. This module does not replace
that. It puts an **ASN.1 schema** next to it and asks whether the two agree — which is the
same move `jer_plan.py` made against the type model, and it found things there too.

**What a schema buys over the hand-written loader.** The loader's field list, its bounds and
its enumerated vocabularies live in imperative code, so they are correct only for as long as
somebody maintains them alongside the writer. A schema is the *same* description the encoder
uses, so a member the writer emits and the schema does not name is a refusal rather than a
silent extra key. It is also transferable: the schema encodes to DER or COER unchanged, so a
manifest can cross a boundary that speaks neither JSON nor Python.

**What it does not buy, and this matters.** A schema gives well-formedness, never
correctness. `profile.lane_widths must start at 1 and be strictly increasing` is a *semantic*
rule about a cost model, not a shape, and X.680 constraints cannot express "strictly
increasing". Those checks stay exactly where they are. This module is deliberately the
narrower thing: it says what a manifest *is*, and leaves what a manifest *means* to the code
that already knows.

§5.4's requirement is the other half:

    JER -> typed value -> claims  ==  JER -> direct claim builder

**Both sinks consume ONE event walk.** That is the design, not an implementation detail. Two
independent readers of the same document would be two parsers, free to disagree about what
the document says — and then "the paths commute" would be testing that two parsers agree
rather than that two *builders* do. With one walk feeding both, a difference in the result
is a difference in the builders, which is what the law is about.

The direct builder never materializes a `ChannelManifest`. It writes constructor arguments
as members arrive and hands back a `HardwareChannel`. On this rail the value graph still
exists behind the walk, because `json.loads` builds it — J1 recorded that honestly and it is
J3 streaming work to remove, not something this module can claim to have done. What IS
structurally true here, and what the commutation test pins, is that the direct path builds
no intermediate typed record.
"""

from __future__ import annotations

from dataclasses import dataclass

from .jer import JerRules
from .jer_bounded import JerLimits, decode_bounded
from .schema import Component, Module, Primitive, Sequence, SequenceOf
from .tags import Asn1Error, Universal

#: One arc per manifest family, under the same private-enterprise root the StreamPack,
#: artifact-bundle and dialect modules use.
MANIFEST_MODULE_OID = (1, 3, 6, 1, 4, 1, 62596, 33)

_INT = Primitive(Universal.INTEGER, "INTEGER")
_UTF8 = Primitive(Universal.UTF8_STRING, "UTF8String")
_BOOL = Primitive(Universal.BOOLEAN, "BOOLEAN")
_INTS = SequenceOf(_INT, "SEQUENCE OF INTEGER")
_STRINGS = SequenceOf(_UTF8, "SEQUENCE OF UTF8String")


# --- channel.json ---------------------------------------------------------------------------
#
# The member names are the JSON names `ChannelManifest.to_dict` already writes, not
# ASN.1-idiomatic ones. That is deliberate: an encoding instruction could rename them
# (X.697 clause 16's NAME), but then the schema would describe a document `channel_plugin`
# does not produce, and the two rails could not be compared on one file. The point of this
# module is to sit beside the existing loader, so it speaks the existing loader's document.

TIER_TYPE = Sequence(
    (
        Component("name", _UTF8),
        Component("latency_cyc", _INT),
        Component("bw_factor", _INT),
        Component("lat_factor", _INT),
        # REQUIRED, not DEFAULT 0 — and the first draft of this schema got it wrong.
        #
        # `Tier.capacity` defaults to 0 in the dataclass and `_tier_from_schema` reads it with
        # `.get("capacity", 0)`, which makes "DEFAULT 0" look like the faithful ASN.1 spelling.
        # It is not, because X.690 §11.5 (via X.697 §21.2) makes a canonical encoder OMIT a
        # component equal to its default — and `channel_plugin._parse_manifest_document`
        # requires the key to be present. So the canonical JER for a zero-capacity tier was a
        # document the repository's own loader refused as `missing ['capacity']`.
        #
        # The schema was the wrong rail. `profile_to_schema` always writes `capacity`, so the
        # document format genuinely carries it and a DEFAULT was an ASN.1 idiom imposed on a
        # file that does not have that shape — exactly what this module's docstring warns
        # against. The loader is unchanged: loosening a third-party-input validator to match a
        # schema I wrote would be fixing the wrong side.
        Component("capacity", _INT),
    ),
    name="MemoryTier",
)

PROFILE_TYPE = Sequence(
    (
        Component("name", _UTF8),
        Component("triple", _UTF8),
        Component("cacheline", _INT),
        Component("elem_bytes", _INT),
        Component("lane_widths", _INTS),
        Component("warp", _INT),
        Component("scalable", _BOOL),
        Component("gather_penalty", _INT),
        Component("mem_unit", _INT),
        Component("base_overhead", _INT),
        Component("thermal_density", _INT),
        Component("power_density", _INT),
        Component("per_op_heat", _INT),
        Component("fma", _BOOL),
        Component("isa_features", _STRINGS),
        Component("affinity_domains", _INT),
        Component("mem_channels", _INT),
        Component("cal_gen", _INT),
        Component("mem_tiers", SequenceOf(TIER_TYPE, "SEQUENCE OF MemoryTier")),
    ),
    name="TargetProfile",
)

CODEGEN_TYPE = Sequence(
    (
        Component("llvm_triple", _UTF8),
        Component("e_machine", _INT),
    ),
    name="Codegen",
)

RUNTIME_TYPE = Sequence(
    (
        Component("perf_syscall_nr", _INT),
        Component("energy_source", _UTF8),
        Component("thermal_zone_types", _STRINGS),
    ),
    name="RuntimeSignals",
)

CALIBRATION_TYPE = Sequence(
    (
        Component("ref", _UTF8),
        Component("digest", _UTF8),
        Component("cal_gen", _INT),
        Component("provenance", _UTF8),
    ),
    name="Calibration",
)

CHANNEL_MANIFEST = Sequence(
    (
        Component("format_version", _INT),
        Component("name", _UTF8),
        Component("kind", _UTF8),
        Component("provenance", _UTF8),
        Component("modeled", _BOOL),
        Component("arch_match", _STRINGS),
        Component("capabilities", _STRINGS),
        Component("codegen", CODEGEN_TYPE),
        Component("runtime", RUNTIME_TYPE),
        Component("calibration", CALIBRATION_TYPE),
        Component("profile", PROFILE_TYPE),
    ),
    name="ChannelManifest",
)


# --- the device manifest (kbcir/device_manifest.py) ------------------------------------------
#
# A bank is written as a five-element array on the JSON rail -- `[name, domain, capacity,
# tile, align]` -- and that shape is preserved rather than "improved" into a SEQUENCE of
# named members. A schema that described a nicer document than the one on disk would be a
# schema for a file this repository does not write.

BANK_TYPE = Sequence(
    (
        Component("name", _UTF8),
        Component("domain", _INT),
        Component("capacity_bytes", _INT),
        Component("native_tile", _INT),
        Component("align", _INT),
    ),
    name="MemoryBank",
)

DEVICE_MANIFEST = Sequence(
    (
        Component("kind", _UTF8),
        Component("schema", _INT),
        Component("device", _UTF8),
        Component("target", _UTF8),
        Component("cal_gen", _INT),
        Component("banks", SequenceOf(BANK_TYPE, "SEQUENCE OF MemoryBank")),
        #: The Q8 interconnect distance matrix, row-major. SEQUENCE OF SEQUENCE OF INTEGER
        #: rather than a flattened array with a stride: a matrix whose row length is implied by
        #: a separate field is a matrix a truncated document can misread as a different shape.
        Component("distance", SequenceOf(_INTS, "SEQUENCE OF SEQUENCE OF INTEGER")),
        Component("digest", _UTF8),
    ),
    name="DeviceManifest",
)


# --- the §6.2 selection envelope --------------------------------------------------------------
#
# §6.2 asks a selection certificate to record "legality independently of costs" -- the
# two-truth law, in a document. The schema enforces the separation structurally: `legal` and
# `refusal` are one member group, the measurements another, and a consumer reading only the
# first has read the whole verdict.

MEASUREMENT_TYPE = Sequence(
    (
        Component("candidate", _UTF8),
        Component("legal", _BOOL),
        #: Absent when the candidate was refused. OPTIONAL rather than DEFAULT 0, because "no
        #: encoding exists" and "the encoding is zero octets long" are different facts and a
        #: certificate that spelled them the same would be lying about one of them.
        Component("octets", _INT, optional=True),
        Component("refusal", _UTF8, optional=True),
        #: Graded truth, kept in its own members and named `_ns` so nobody mistakes an oracle
        #: timing for the calibrated cost table phase H actually selects on.
        Component("encode_ns", _INT, default=0),
        Component("decode_ns", _INT, default=0),
    ),
    name="Measurement",
)

SELECTION_ENVELOPE = Sequence(
    (
        Component("version", _INT),
        Component("objective", _UTF8),
        Component("typeName", _UTF8),
        #: The winner, or absent when nothing was legal. A certificate that named a winner it
        #: had not measured, or omitted the field to mean "the first one", would be exactly the
        #: kind of implicit that §6.2 exists to remove.
        Component("selected", _UTF8, optional=True),
        Component("measurements", SequenceOf(MEASUREMENT_TYPE, "SEQUENCE OF Measurement")),
    ),
    name="SelectionEnvelope",
)

SELECTION_ENVELOPE_VERSION = 1

MODULE = Module(
    "BCIR-Manifests",
    MANIFEST_MODULE_OID,
    {
        "ChannelManifest": CHANNEL_MANIFEST,
        "DeviceManifest": DEVICE_MANIFEST,
        "SelectionEnvelope": SELECTION_ENVELOPE,
    },
)


# --- §5.4's two sinks ---------------------------------------------------------------------------


class ManifestSink:
    """What a schema-directed walk hands its consumer.

    Deliberately the same shape as `bcir_jer_sink` in `runtime/c/bcir_jer.h`: a member name
    with the path that reached it, and a scalar. A sink written against this interface is
    one a C rail could drive with the same events, which is what keeps the direct builder
    from being a Python-only convenience.

    `begin` is told the container's SHAPE rather than left to infer it from the path. That
    is not a convenience: a member named `"0"` and the first element of a list produce the
    same path step, so a sink guessing from the name would build a list where the schema
    said a record. The walk knows the type; it says so.
    """

    def begin(self, path: tuple[str, ...], shape: str) -> None:  # pragma: no cover - no-op
        """A container of `shape` ("record" or "list") opens at `path`."""

    def end(self, path: tuple[str, ...]) -> None:  # pragma: no cover - default no-op
        """The container at `path` closes."""

    def member(self, path: tuple[str, ...], value) -> None:
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError


def walk(kind, value, sink: ManifestSink, path: tuple[str, ...] = ()) -> None:
    """Drive `sink` over `value` in schema order.

    Schema order, not document order, and the difference is the point: the walk visits the
    members the TYPE declares, so a sink sees the same sequence whatever order a
    non-canonical sender wrote them in. That is what lets a generated fixed-schema consumer
    exist at all — §5.2's "same event trace as the table-driven implementation".
    """
    if isinstance(kind, Sequence):
        sink.begin(path, "record")
        for component in kind.components:
            if component.name not in value:
                continue
            walk(component.type, value[component.name], sink, path + (component.name,))
        sink.end(path)
        return
    if isinstance(kind, SequenceOf):
        sink.begin(path, "list")
        for index, item in enumerate(value):
            walk(kind.element, item, sink, path + (str(index),))
        sink.end(path)
        return
    sink.member(path, value)


class TypedValueSink(ManifestSink):
    """§5.4's first sink: rebuild the typed value, for diagnostics and round trips.

    It reconstructs the nested document the encoder started from, which makes it the
    natural input to the *existing* programmatic constructor — `ChannelManifest.from_dict`
    and the loader's own validation. Its job is to be the boring path the direct builder is
    checked against, and to demonstrate the event stream carries enough to rebuild the
    document at all (§5.2's contract for a generated consumer).
    """

    def __init__(self, kind=None) -> None:
        self._kind = kind
        self._root = None

    def _navigate(self, path: tuple[str, ...]):
        node = self._root
        for step in path:
            node = node[int(step)] if isinstance(node, list) else node[step]
        return node

    def _place(self, path: tuple[str, ...], value) -> None:
        if not path:
            self._root = value
            return
        parent = self._navigate(path[:-1])
        if isinstance(parent, list):
            index = int(path[-1])
            while len(parent) <= index:
                parent.append(None)
            parent[index] = value
        else:
            parent[path[-1]] = value

    def begin(self, path: tuple[str, ...], shape: str) -> None:
        self._place(path, [] if shape == "list" else {})

    def member(self, path: tuple[str, ...], value) -> None:
        self._place(path, value)

    def finish(self):
        return self._root


@dataclass
class _ChannelUnderConstruction:
    """The direct builder's accumulator — constructor arguments, not a typed record."""

    name: str = ""
    kind: str = ""
    llvm_triple: str = ""
    e_machine: int = 0
    perf_syscall_nr: int = 0
    energy_source: str = "none"
    thermal_zone_types: tuple = ()
    arch_match: tuple = ()
    modeled: bool = True
    capabilities: frozenset = frozenset()
    profile: dict = None  # type: ignore[assignment]


class DirectChannelSink(ManifestSink):
    """§5.4's second sink: a `HardwareChannel` without an intermediate typed record.

    The distinction from `TypedValueSink` is structural and is what the commutation test
    pins: **no `ChannelManifest` is ever constructed**. Members are written into
    constructor arguments as they arrive, and `finish` builds the channel.

    The one place it *must* reuse existing code is `schema_to_profile`, and that is a
    deliberate limit rather than a shortcut. `TargetProfile` is the K_BCIR cost model; a
    second construction of it here would be a second definition of what a cost model is,
    free to drift from the one the optimizer reads. §5.4 asks the two paths to agree on the
    claims they build, not to reimplement the objects those claims are made of.
    """

    def __init__(self) -> None:
        self._state = _ChannelUnderConstruction()
        self._profile: dict = {}
        self._tiers: list[dict] = []

    def member(self, path: tuple[str, ...], value) -> None:
        head = path[0]
        if head == "profile":
            self._profile_member(path[1:], value)
            return
        if head == "codegen":
            if path[1] == "llvm_triple":
                self._state.llvm_triple = value
            else:
                self._state.e_machine = value
            return
        if head == "runtime":
            if path[1] == "perf_syscall_nr":
                self._state.perf_syscall_nr = value
            elif path[1] == "energy_source":
                self._state.energy_source = value
            else:
                self._state.thermal_zone_types += (value,)
            return
        if head == "calibration":
            # The calibration record is provenance for the profile, not part of the channel
            # the registry holds. Dropped here rather than carried and ignored.
            return
        if head == "arch_match":
            self._state.arch_match += (value,)
            return
        if head == "capabilities":
            self._state.capabilities |= {value}
            return
        if head == "name":
            self._state.name = value
        elif head == "kind":
            self._state.kind = value
        elif head == "modeled":
            self._state.modeled = value

    def _profile_member(self, path: tuple[str, ...], value) -> None:
        if not path:
            return
        if path[0] == "mem_tiers":
            index = int(path[1])
            while len(self._tiers) <= index:
                self._tiers.append({})
            self._tiers[index][path[2]] = value
            return
        if path[0] in ("lane_widths", "isa_features"):
            self._profile.setdefault(path[0], []).append(value)
            return
        self._profile[path[0]] = value

    def finish(self):
        from ..channel_plugin import schema_to_profile
        from ..channels import HardwareChannel, RuntimeChannel

        self._profile["mem_tiers"] = self._tiers
        return HardwareChannel(
            name=self._state.name,
            kind=self._state.kind,
            profile=schema_to_profile(self._profile),
            llvm_triple=self._state.llvm_triple,
            e_machine=self._state.e_machine,
            runtime=RuntimeChannel(
                perf_syscall_nr=self._state.perf_syscall_nr,
                energy_source=self._state.energy_source,
                thermal_zone_types=self._state.thermal_zone_types,
            ),
            arch_match=self._state.arch_match,
            modeled=self._state.modeled,
            capabilities=frozenset(self._state.capabilities),
        )


# --- the two paths §5.4 names --------------------------------------------------------------------


def channel_to_jer(manifest, *, rules: JerRules = JerRules.CANONICAL) -> bytes:
    """Encode a `ChannelManifest` under the schema, from the dict it already writes."""
    from .jer import encode_jer

    return encode_jer(CHANNEL_MANIFEST, manifest.to_dict(), rules=rules)


def channel_via_typed_value(
    data: bytes, *, limits: JerLimits = JerLimits(), rules: JerRules = JerRules.CANONICAL
):
    """`JER -> typed value -> claims`: through the existing programmatic constructor."""
    from ..channel_plugin import ChannelManifest

    value = decode_bounded(data, CHANNEL_MANIFEST, rules=rules, limits=limits)
    sink = TypedValueSink(CHANNEL_MANIFEST)
    walk(CHANNEL_MANIFEST, value, sink)
    return ChannelManifest.from_dict(sink.finish()).to_channel()


def channel_direct(
    data: bytes, *, limits: JerLimits = JerLimits(), rules: JerRules = JerRules.CANONICAL
):
    """`JER -> direct claim builder`: no intermediate `ChannelManifest`."""
    value = decode_bounded(data, CHANNEL_MANIFEST, rules=rules, limits=limits)
    sink = DirectChannelSink()
    walk(CHANNEL_MANIFEST, value, sink)
    return sink.finish()


# --- the device manifest and the selection envelope ----------------------------------------------


def device_manifest_to_value(man) -> dict:
    """The same document `save_device_manifest` writes, as an ASN.1 value."""
    from ..kbcir.device_manifest import MANIFEST_KIND, MANIFEST_SCHEMA

    return {
        "kind": MANIFEST_KIND,
        "schema": MANIFEST_SCHEMA,
        "device": man.device,
        "target": man.target,
        "cal_gen": int(man.cal_gen),
        "banks": tuple(
            {
                "name": b.name,
                "domain": int(b.domain),
                "capacity_bytes": b.capacity_bytes,
                "native_tile": b.native_tile,
                "align": b.align,
            }
            for b in man.banks
        ),
        "distance": tuple(tuple(row) for row in man.distance),
        "digest": man.digest,
    }


def value_to_device_manifest(value: dict):
    from ..kbcir.device_manifest import DeviceManifest, MemoryBank
    from ..model import Domain

    return DeviceManifest(
        device=value["device"],
        target=value["target"],
        cal_gen=value["cal_gen"],
        banks=tuple(
            MemoryBank(
                name=b["name"],
                domain=Domain(b["domain"]),
                capacity_bytes=b["capacity_bytes"],
                native_tile=b["native_tile"],
                align=b["align"],
            )
            for b in value["banks"]
        ),
        distance=tuple(tuple(row) for row in value["distance"]),
    )


def selection_envelope(
    measurements, *, objective: str, type_name: str, selected: str | None
) -> dict:
    """§6.2's certificate as an ASN.1 value.

    Legality is recorded independently of cost, which is the two-truth law made structural:
    a consumer that reads only `legal` and `refusal` has read the whole verdict, and no
    amount of timing can turn a refusal into a candidate.
    """
    rows = []
    for m in measurements:
        row = {
            "candidate": m.candidate,
            "legal": m.legal,
            "encode_ns": m.encode_ns,
            "decode_ns": m.decode_ns,
        }
        if m.octets is not None:
            row["octets"] = m.octets
        if getattr(m, "refusal", ""):
            row["refusal"] = m.refusal
        rows.append(row)
    out = {
        "version": SELECTION_ENVELOPE_VERSION,
        "objective": objective,
        "typeName": type_name,
        "measurements": tuple(rows),
    }
    if selected is not None:
        out["selected"] = selected
    return out


__all__ = [
    "BANK_TYPE",
    "CALIBRATION_TYPE",
    "CHANNEL_MANIFEST",
    "CODEGEN_TYPE",
    "DEVICE_MANIFEST",
    "MANIFEST_MODULE_OID",
    "MEASUREMENT_TYPE",
    "MODULE",
    "PROFILE_TYPE",
    "RUNTIME_TYPE",
    "SELECTION_ENVELOPE",
    "SELECTION_ENVELOPE_VERSION",
    "TIER_TYPE",
    "DirectChannelSink",
    "ManifestSink",
    "TypedValueSink",
    "channel_direct",
    "channel_to_jer",
    "channel_via_typed_value",
    "device_manifest_to_value",
    "selection_envelope",
    "value_to_device_manifest",
    "walk",
]
