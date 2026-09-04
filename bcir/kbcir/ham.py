"""Verified Hierarchical Access Memory (HAM) planning.

This module is the provider-neutral control plane for semantic resources moving across
declared memory banks.  It does not issue DMA, configure CXL, or claim that a host can
perform peer-to-peer transfers.  A route exists only when every directed link is present
in the supplied hardware envelope.  The resulting actions are independently replayed,
then lowered through the existing BCIR claim graph and StreamPack rails.

Learned policy identity may be attached to a workload, but learning never establishes
residency, route, capacity, freshness, or legality.  Those properties are checked from
the immutable resource/access inventory and explicit hardware links.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import asdict, dataclass, replace

from .._artifact_json import strict_json_loads
from ..abi.streampack_abi import encode
from ..gem.streampack import StreamPack, hydrate_pipelined
from ..model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from ..verify import verify_all, verify_smart_lowering
from .cost import TargetProfile, Theta
from .device_manifest import (
    DeviceManifest,
    MemoryBank,
    check_bank_moves,
    check_device_manifest,
    move_claim,
)
from .provenance import hash_module
from .realize import RealizationResult, optimize

_MAX_JSON = 32 * 1024 * 1024
_MAX_U63 = (1 << 63) - 1
_MAX_RESOURCES = 4096
_MAX_ACCESSES = 65536
_MAX_ACTIONS = 1_000_000
_MAX_ROUTE_HOPS = 16
_WORKLOAD_SCHEMA = "bcir.ham_workload.v1"
_PLAN_SCHEMA = "bcir.ham_execution_plan.v1"
_RESOURCE_KINDS = frozenset(
    ("tensor", "model", "adapter", "telemetry", "index", "program", "other")
)
_ACTION_KINDS = frozenset(("transfer", "prefetch", "barrier", "access", "invalidate", "evict"))
_TRANSFER_INTENTS = frozenset(("demand", "prefetch", "writeback"))
_ROUTE_CLASSES = frozenset(("direct-peer", "direct-dma", "staged"))


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _int(value, field: str, *, minimum: int = 0, maximum: int = _MAX_U63) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _name(value, field: str, *, choices=None, empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or (not empty and not value)
        or len(value.encode("utf-8")) > 4096
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValueError(f"{field} must be a bounded, control-free string")
    if choices is not None and value not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}")
    return value


def _digest(value, field: str, *, empty: bool = False) -> str:
    if empty and value == "":
        return value
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _checked_add(left: int, right: int, field: str) -> int:
    if left < 0 or right < 0 or left > _MAX_U63 - right:
        raise ValueError(f"{field} exceeds signed 64-bit accounting")
    return left + right


def _checked_mul(left: int, right: int, field: str) -> int:
    if left < 0 or right < 0 or (right and left > _MAX_U63 // right):
        raise ValueError(f"{field} exceeds signed 64-bit accounting")
    return left * right


@dataclass(frozen=True)
class HAMResource:
    """One immutable identity in the semantic resource hierarchy.

    ``content_sha256`` identifies the initial payload without loading it.  Mutable
    accesses advance a separate deterministic *version lineage*; they do not pretend
    that BCIR can derive the bytes' cryptographic hash without observing the payload.
    The home copy is pinned and is the write-back destination for mutable resources.
    """

    resource_id: int
    name: str
    kind: str
    nbytes: int
    home_bank: str
    content_sha256: str
    dependencies: tuple[int, ...] = ()
    mutable: bool = False
    priority: int = 0

    def __post_init__(self) -> None:
        _int(self.resource_id, "HAM resource_id", minimum=1)
        _name(self.name, "HAM resource name")
        _name(self.kind, "HAM resource kind", choices=_RESOURCE_KINDS)
        _int(self.nbytes, "HAM resource nbytes", minimum=1)
        _name(self.home_bank, "HAM resource home_bank")
        _digest(self.content_sha256, "HAM resource content_sha256")
        if type(self.mutable) is not bool:
            raise ValueError("HAM resource mutable must be Boolean")
        _int(self.priority, "HAM resource priority", maximum=255)
        if not isinstance(self.dependencies, tuple) or any(
            type(value) is not int or value < 1 for value in self.dependencies
        ):
            raise ValueError("HAM resource dependencies must be positive integer IDs")
        if (
            tuple(sorted(self.dependencies)) != self.dependencies
            or len(set(self.dependencies)) != len(self.dependencies)
            or self.resource_id in self.dependencies
        ):
            raise ValueError("HAM resource dependencies must be sorted, unique, and non-self")


@dataclass(frozen=True)
class HAMAccess:
    sequence: int
    resource_id: int
    bank: str
    mode: str
    operation: str

    def __post_init__(self) -> None:
        _int(self.sequence, "HAM access sequence", minimum=1)
        _int(self.resource_id, "HAM access resource_id", minimum=1)
        _name(self.bank, "HAM access bank")
        _name(self.mode, "HAM access mode", choices={"read", "write"})
        _name(self.operation, "HAM access operation")


@dataclass(frozen=True)
class HAMWorkload:
    hardware_digest: str
    resources: tuple[HAMResource, ...]
    accesses: tuple[HAMAccess, ...]
    prefetch_distance: int = 1
    max_route_hops: int = 4
    policy_digest: str = ""
    schema: str = _WORKLOAD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _WORKLOAD_SCHEMA:
            raise ValueError("unsupported HAM workload schema")
        _digest(self.hardware_digest, "HAM workload hardware_digest")
        _digest(self.policy_digest, "HAM workload policy_digest", empty=True)
        _int(self.prefetch_distance, "HAM prefetch_distance", maximum=64)
        _int(self.max_route_hops, "HAM max_route_hops", minimum=1, maximum=_MAX_ROUTE_HOPS)
        if (
            not isinstance(self.resources, tuple)
            or not self.resources
            or len(self.resources) > _MAX_RESOURCES
            or any(not isinstance(row, HAMResource) for row in self.resources)
        ):
            raise ValueError("HAM workload needs a bounded typed resource inventory")
        if (
            not isinstance(self.accesses, tuple)
            or not self.accesses
            or len(self.accesses) > _MAX_ACCESSES
            or any(not isinstance(row, HAMAccess) for row in self.accesses)
        ):
            raise ValueError("HAM workload needs a bounded typed access trace")
        ids = tuple(row.resource_id for row in self.resources)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise ValueError("HAM resources must use unique canonical resource_id order")
        if tuple(row.sequence for row in self.accesses) != tuple(range(1, len(self.accesses) + 1)):
            raise ValueError("HAM accesses must use contiguous canonical sequence numbers")
        known = set(ids)
        if any(row.resource_id not in known for row in self.accesses):
            raise ValueError("HAM access references an unknown resource")
        if any(
            dependency not in known for row in self.resources for dependency in row.dependencies
        ):
            raise ValueError("HAM dependency references an unknown resource")
        graph = {row.resource_id: row.dependencies for row in self.resources}
        done: set[int] = set()
        active: set[int] = set()

        def visit(resource_id: int) -> None:
            if resource_id in done:
                return
            if resource_id in active:
                raise ValueError("HAM resource dependency graph contains a cycle")
            active.add(resource_id)
            for dependency in graph[resource_id]:
                visit(dependency)
            active.remove(resource_id)
            done.add(resource_id)

        for resource_id in ids:
            visit(resource_id)

    def resource(self, resource_id: int) -> HAMResource:
        for row in self.resources:
            if row.resource_id == resource_id:
                return row
        raise KeyError(resource_id)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "hardware_digest": self.hardware_digest,
            "resources": [
                {**asdict(row), "dependencies": list(row.dependencies)} for row in self.resources
            ],
            "accesses": [asdict(row) for row in self.accesses],
            "prefetch_distance": self.prefetch_distance,
            "max_route_hops": self.max_route_hops,
            "policy_digest": self.policy_digest,
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "HAMWorkload":
        doc = strict_json_loads(text, "HAM workload", max_bytes=_MAX_JSON)
        fields = {
            "schema",
            "hardware_digest",
            "resources",
            "accesses",
            "prefetch_distance",
            "max_route_hops",
            "policy_digest",
        }
        if not isinstance(doc, dict) or set(doc) != fields:
            raise ValueError("HAM workload has missing or unknown fields")
        if (
            not isinstance(doc["resources"], list)
            or not doc["resources"]
            or len(doc["resources"]) > _MAX_RESOURCES
            or not isinstance(doc["accesses"], list)
            or not doc["accesses"]
            or len(doc["accesses"]) > _MAX_ACCESSES
        ):
            raise ValueError("HAM workload arrays are empty, malformed, or unbounded")
        try:
            resources = []
            for row in doc.pop("resources"):
                if not isinstance(row, dict):
                    raise ValueError("HAM resource entries must be objects")
                value = dict(row)
                value["dependencies"] = tuple(value["dependencies"])
                resources.append(HAMResource(**value))
            accesses = tuple(HAMAccess(**row) for row in doc.pop("accesses"))
            return cls(**doc, resources=tuple(resources), accesses=accesses)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed HAM workload: {exc}") from exc


@dataclass(frozen=True)
class HAMAction:
    order: int
    sequence: int
    kind: str
    resource_id: int
    bank: str
    route: tuple[str, ...]
    intent: str
    mode: str
    nbytes: int
    generation_before: int
    generation_after: int
    version_sha256: str
    predicted_ns: int
    route_class: str
    host_bounce: bool
    reason: str

    def __post_init__(self) -> None:
        _int(self.order, "HAM action order", minimum=1)
        _int(self.sequence, "HAM action sequence", minimum=1, maximum=_MAX_ACCESSES + 1)
        _name(self.kind, "HAM action kind", choices=_ACTION_KINDS)
        _int(self.resource_id, "HAM action resource_id", minimum=1)
        _name(self.bank, "HAM action bank", empty=True)
        if (
            not isinstance(self.route, tuple)
            or any(not isinstance(value, str) or not value for value in self.route)
            or len(self.route) > _MAX_ROUTE_HOPS + 1
        ):
            raise ValueError("HAM action route is malformed or unbounded")
        _name(self.intent, "HAM action intent", empty=True)
        _name(self.mode, "HAM action mode", empty=True)
        _int(self.nbytes, "HAM action nbytes")
        _int(self.generation_before, "HAM action generation_before")
        _int(self.generation_after, "HAM action generation_after")
        _digest(self.version_sha256, "HAM action version_sha256")
        _int(self.predicted_ns, "HAM action predicted_ns")
        _name(self.route_class, "HAM action route_class", empty=True)
        if type(self.host_bounce) is not bool:
            raise ValueError("HAM action host_bounce must be Boolean")
        _name(self.reason, "HAM action reason")
        if self.kind == "transfer":
            if (
                len(self.route) < 2
                or self.bank != self.route[-1]
                or self.intent not in _TRANSFER_INTENTS
                or self.mode
                or self.nbytes < 1
                or self.generation_before != self.generation_after
                or self.predicted_ns < 1
                or self.route_class not in _ROUTE_CLASSES
            ):
                raise ValueError("HAM transfer action fields are inconsistent")
        else:
            if self.route or self.route_class or self.host_bounce or self.predicted_ns:
                raise ValueError("only HAM transfer actions may carry route metadata")
            if self.kind == "access":
                if (
                    self.mode not in {"read", "write"}
                    or self.intent
                    or self.nbytes < 1
                    or (self.mode == "read" and self.generation_before != self.generation_after)
                    or (
                        self.mode == "write" and self.generation_after != self.generation_before + 1
                    )
                ):
                    raise ValueError("HAM access action fields are inconsistent")
            elif self.kind == "prefetch":
                if (
                    self.intent != "prefetch"
                    or self.mode
                    or self.nbytes < 1
                    or self.generation_before != self.generation_after
                ):
                    raise ValueError("HAM prefetch action fields are inconsistent")
            elif self.kind == "barrier":
                if self.intent or self.mode or self.nbytes:
                    raise ValueError("HAM barrier action fields are inconsistent")
            elif self.kind in {"invalidate", "evict"}:
                if (
                    self.intent
                    or self.mode
                    or self.nbytes < 1
                    or self.generation_after < self.generation_before
                ):
                    raise ValueError("HAM removal action fields are inconsistent")


@dataclass(frozen=True)
class HAMBankPeak:
    bank: str
    peak_bytes: int
    allocatable_bytes: int

    def __post_init__(self) -> None:
        _name(self.bank, "HAM peak bank")
        _int(self.peak_bytes, "HAM peak bytes")
        _int(self.allocatable_bytes, "HAM peak allocatable_bytes", minimum=1)
        if self.peak_bytes > self.allocatable_bytes:
            raise ValueError("HAM peak exceeds allocatable bank capacity")


@dataclass(frozen=True)
class HAMExecutionPlan:
    hardware_digest: str
    workload_digest: str
    policy_digest: str
    actions: tuple[HAMAction, ...]
    peaks: tuple[HAMBankPeak, ...]
    logical_transfer_bytes: int
    wire_transfer_bytes: int
    direct_peer_bytes: int
    staged_transfer_bytes: int
    host_bounce_bytes: int
    predicted_transfer_ns: int
    module_digest: str
    streampack_sha256: str
    streampack_bytes: int
    schema: str = _PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _PLAN_SCHEMA:
            raise ValueError("unsupported HAM execution-plan schema")
        for value, field, empty in (
            (self.hardware_digest, "hardware_digest", False),
            (self.workload_digest, "workload_digest", False),
            (self.policy_digest, "policy_digest", True),
            (self.module_digest, "module_digest", False),
            (self.streampack_sha256, "streampack_sha256", False),
        ):
            _digest(value, f"HAM plan {field}", empty=empty)
        if (
            not isinstance(self.actions, tuple)
            or not self.actions
            or len(self.actions) > _MAX_ACTIONS
            or any(not isinstance(row, HAMAction) for row in self.actions)
        ):
            raise ValueError("HAM plan needs a bounded typed action sequence")
        if tuple(row.order for row in self.actions) != tuple(range(1, len(self.actions) + 1)):
            raise ValueError("HAM plan action order must be contiguous and canonical")
        if any(
            right.sequence < left.sequence for left, right in zip(self.actions, self.actions[1:])
        ):
            raise ValueError("HAM plan action sequences must be nondecreasing")
        if (
            not isinstance(self.peaks, tuple)
            or not self.peaks
            or len(self.peaks) > 32
            or any(not isinstance(row, HAMBankPeak) for row in self.peaks)
            or tuple(sorted(self.peaks, key=lambda row: row.bank)) != self.peaks
            or len({row.bank for row in self.peaks}) != len(self.peaks)
        ):
            raise ValueError("HAM plan bank peaks must be unique and canonically sorted")
        for field in (
            "logical_transfer_bytes",
            "wire_transfer_bytes",
            "direct_peer_bytes",
            "staged_transfer_bytes",
            "host_bounce_bytes",
            "predicted_transfer_ns",
        ):
            _int(getattr(self, field), f"HAM plan {field}")
        if self.wire_transfer_bytes < self.logical_transfer_bytes:
            raise ValueError("HAM wire transfer bytes cannot be below logical bytes")
        if (
            self.direct_peer_bytes > self.logical_transfer_bytes
            or self.staged_transfer_bytes > self.logical_transfer_bytes
            or self.host_bounce_bytes > self.staged_transfer_bytes
        ):
            raise ValueError("HAM transfer classifications exceed the transfer inventory")
        _int(self.streampack_bytes, "HAM plan streampack_bytes", minimum=1)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "hardware_digest": self.hardware_digest,
            "workload_digest": self.workload_digest,
            "policy_digest": self.policy_digest,
            "actions": [{**asdict(row), "route": list(row.route)} for row in self.actions],
            "peaks": [asdict(row) for row in self.peaks],
            "logical_transfer_bytes": self.logical_transfer_bytes,
            "wire_transfer_bytes": self.wire_transfer_bytes,
            "direct_peer_bytes": self.direct_peer_bytes,
            "staged_transfer_bytes": self.staged_transfer_bytes,
            "host_bounce_bytes": self.host_bounce_bytes,
            "predicted_transfer_ns": self.predicted_transfer_ns,
            "module_digest": self.module_digest,
            "streampack_sha256": self.streampack_sha256,
            "streampack_bytes": self.streampack_bytes,
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "HAMExecutionPlan":
        doc = strict_json_loads(text, "HAM execution plan", max_bytes=_MAX_JSON)
        fields = {
            "schema",
            "hardware_digest",
            "workload_digest",
            "policy_digest",
            "actions",
            "peaks",
            "logical_transfer_bytes",
            "wire_transfer_bytes",
            "direct_peer_bytes",
            "staged_transfer_bytes",
            "host_bounce_bytes",
            "predicted_transfer_ns",
            "module_digest",
            "streampack_sha256",
            "streampack_bytes",
        }
        if not isinstance(doc, dict) or set(doc) != fields:
            raise ValueError("HAM execution plan has missing or unknown fields")
        if (
            not isinstance(doc["actions"], list)
            or not doc["actions"]
            or len(doc["actions"]) > _MAX_ACTIONS
            or not isinstance(doc["peaks"], list)
            or not doc["peaks"]
            or len(doc["peaks"]) > 32
        ):
            raise ValueError("HAM execution-plan arrays are empty, malformed, or unbounded")
        try:
            actions = []
            for row in doc.pop("actions"):
                if not isinstance(row, dict):
                    raise ValueError("HAM action entries must be objects")
                value = dict(row)
                value["route"] = tuple(value["route"])
                actions.append(HAMAction(**value))
            peaks = tuple(HAMBankPeak(**row) for row in doc.pop("peaks"))
            return cls(**doc, actions=tuple(actions), peaks=peaks)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed HAM execution plan: {exc}") from exc


@dataclass(frozen=True)
class LoweredHAMExecution:
    artifact: HAMExecutionPlan
    module: Module
    realization: RealizationResult
    streampack: StreamPack
    streampack_data: bytes


def _hardware_inventory(hardware, workload: HAMWorkload) -> tuple[dict, dict, dict]:
    if getattr(hardware, "digest", None) != workload.hardware_digest:
        raise ValueError("HAM workload hardware digest does not match the hardware envelope")
    banks = tuple(getattr(hardware, "banks", ()))
    links = tuple(getattr(hardware, "links", ()))
    if not banks or len(banks) > 32:
        raise ValueError("HAM planning requires a bounded hardware bank inventory")
    by_name = {getattr(row, "name", None): row for row in banks}
    if len(by_name) != len(banks) or None in by_name:
        raise ValueError("HAM hardware banks must have unique names")
    link_map = {}
    adjacency = {name: [] for name in by_name}
    for link in links:
        key = (getattr(link, "source", None), getattr(link, "destination", None))
        if key in link_map or key[0] not in by_name or key[1] not in by_name:
            raise ValueError("HAM hardware links are duplicate or reference unknown banks")
        link_map[key] = link
        adjacency[key[0]].append(key[1])
    for values in adjacency.values():
        values.sort()
    for resource in workload.resources:
        if resource.home_bank not in by_name:
            raise ValueError(
                f"HAM resource {resource.resource_id} names unknown home bank "
                f"{resource.home_bank!r}"
            )
        if getattr(by_name[resource.home_bank], "domain", "") == "MMIO":
            raise ValueError("HAM payload resources cannot use an MMIO home bank")
    for access in workload.accesses:
        if access.bank not in by_name:
            raise ValueError(f"HAM access {access.sequence} names unknown bank {access.bank!r}")
        if getattr(by_name[access.bank], "domain", "") == "MMIO":
            raise ValueError("HAM payload accesses cannot target MMIO")
        if access.mode == "write" and not workload.resource(access.resource_id).mutable:
            raise ValueError(
                f"HAM access {access.sequence} writes immutable resource {access.resource_id}"
            )
    return by_name, link_map, adjacency


def _link_ns(link, nbytes: int) -> int:
    bandwidth = _int(getattr(link, "bandwidth_bps", None), "HAM link bandwidth", minimum=1)
    latency = _int(getattr(link, "latency_ns", None), "HAM link latency")
    transfer = (
        _checked_mul(nbytes, 1_000_000_000, "HAM link transfer") + bandwidth - 1
    ) // bandwidth
    return _checked_add(latency, max(1, transfer), "HAM link latency")


def _shortest_route(
    source: str, destination: str, nbytes: int, *, link_map: dict, adjacency: dict, max_hops: int
) -> tuple[tuple[str, ...], int]:
    if source == destination:
        return (source,), 0
    queue = [(0, 0, (source,), source)]
    best: dict[tuple[str, int], tuple[int, tuple[str, ...]]] = {}
    while queue:
        cost, hops, path, node = heapq.heappop(queue)
        if node == destination:
            return path, cost
        if hops >= max_hops:
            continue
        key = (node, hops)
        prior = best.get(key)
        if prior is not None and prior <= (cost, path):
            continue
        best[key] = (cost, path)
        for target in adjacency[node]:
            if target in path:
                continue
            edge_cost = _link_ns(link_map[(node, target)], nbytes)
            heapq.heappush(
                queue,
                (
                    _checked_add(cost, edge_cost, "HAM route cost"),
                    hops + 1,
                    path + (target,),
                    target,
                ),
            )
    raise ValueError(
        f"no declared HAM route from {source!r} to {destination!r} within {max_hops} hops"
    )


def _closure(workload: HAMWorkload, resource_id: int) -> tuple[int, ...]:
    result = []
    seen = set()

    def visit(value: int) -> None:
        if value in seen:
            return
        for dependency in workload.resource(value).dependencies:
            visit(dependency)
        seen.add(value)
        result.append(value)

    visit(resource_id)
    return tuple(result)


class _Planner:
    def __init__(self, workload: HAMWorkload, hardware) -> None:
        self.workload = workload
        self.hardware = hardware
        self.banks, self.links, self.adjacency = _hardware_inventory(hardware, workload)
        self.resources = {row.resource_id: row for row in workload.resources}
        self.required = {
            row.sequence: _closure(workload, row.resource_id) for row in workload.accesses
        }
        self.generations = {row.resource_id: 0 for row in workload.resources}
        self.versions = {row.resource_id: row.content_sha256 for row in workload.resources}
        self.copies = {
            row.resource_id: {row.home_bank: (0, row.content_sha256)} for row in workload.resources
        }
        self.used = {name: 0 for name in self.banks}
        for row in workload.resources:
            self.used[row.home_bank] = _checked_add(
                self.used[row.home_bank], row.nbytes, "HAM initial occupancy"
            )
        self.peaks = dict(self.used)
        for name, value in self.used.items():
            if value > self.banks[name].allocatable_bytes:
                raise ValueError(
                    f"HAM pinned home resources need {value} bytes in {name!r}, "
                    f"above its {self.banks[name].allocatable_bytes}-byte envelope"
                )
        self.actions: list[HAMAction] = []
        self.logical_transfer_bytes = 0
        self.wire_transfer_bytes = 0
        self.direct_peer_bytes = 0
        self.staged_transfer_bytes = 0
        self.host_bounce_bytes = 0
        self.predicted_transfer_ns = 0

    def add(
        self,
        sequence: int,
        kind: str,
        resource_id: int,
        bank: str,
        *,
        route=(),
        intent="",
        mode="",
        nbytes=0,
        before=None,
        after=None,
        version=None,
        predicted_ns=0,
        route_class="",
        host_bounce=False,
        reason: str,
    ) -> None:
        generation = self.generations[resource_id]
        self.actions.append(
            HAMAction(
                len(self.actions) + 1,
                sequence,
                kind,
                resource_id,
                bank,
                tuple(route),
                intent,
                mode,
                nbytes,
                generation if before is None else before,
                generation if after is None else after,
                self.versions[resource_id] if version is None else version,
                predicted_ns,
                route_class,
                host_bounce,
                reason,
            )
        )
        if len(self.actions) > _MAX_ACTIONS:
            raise ValueError("HAM plan exceeds the bounded action inventory")

    def _next_use(self, resource_id: int, bank: str, sequence: int) -> int | None:
        for access in self.workload.accesses[sequence:]:
            if access.bank == bank and resource_id in self.required[access.sequence]:
                return access.sequence
        return None

    def evict(
        self, resource_id: int, bank: str, sequence: int, reason: str, *, invalidate: bool = False
    ) -> None:
        resource = self.resources[resource_id]
        if bank not in self.copies[resource_id]:
            raise ValueError(f"HAM cannot remove absent resource {resource_id} from {bank!r}")
        if not invalidate and bank == resource.home_bank:
            raise ValueError("HAM cannot evict a pinned home copy")
        generation, version = self.copies[resource_id].pop(bank)
        self.used[bank] -= resource.nbytes
        self.add(
            sequence,
            "invalidate" if invalidate else "evict",
            resource_id,
            bank,
            nbytes=resource.nbytes,
            before=generation,
            after=self.generations[resource_id],
            version=version,
            reason=reason,
        )

    def make_space(
        self, bank: str, nbytes: int, sequence: int, protect: frozenset[tuple[int, str]]
    ) -> None:
        capacity = self.banks[bank].allocatable_bytes
        while self.used[bank] + nbytes > capacity:
            candidates = []
            for resource_id, copies in self.copies.items():
                resource = self.resources[resource_id]
                if (
                    bank not in copies
                    or bank == resource.home_bank
                    or (resource_id, bank) in protect
                ):
                    continue
                next_use = self._next_use(resource_id, bank, sequence)
                candidates.append(
                    (
                        next_use is None,
                        next_use or 0,
                        -resource.priority,
                        resource.nbytes,
                        -resource_id,
                        resource_id,
                    )
                )
            if not candidates:
                raise ValueError(
                    f"HAM bank {bank!r} cannot make {nbytes} bytes of space "
                    "without evicting pinned/current data"
                )
            resource_id = max(candidates)[-1]
            self.evict(resource_id, bank, sequence, "capacity:farthest-next-use")

    def _route_from_current(self, resource_id: int, target: str) -> tuple[tuple[str, ...], int]:
        generation = self.generations[resource_id]
        candidates = []
        for source, (copy_generation, _version) in self.copies[resource_id].items():
            if copy_generation != generation:
                continue
            try:
                path, cost = _shortest_route(
                    source,
                    target,
                    self.resources[resource_id].nbytes,
                    link_map=self.links,
                    adjacency=self.adjacency,
                    max_hops=self.workload.max_route_hops,
                )
            except ValueError:
                continue
            candidates.append((cost, len(path), path))
        if not candidates:
            raise ValueError(f"no current HAM copy of resource {resource_id} can reach {target!r}")
        _cost, _length, path = min(candidates)
        return path, _cost

    def ensure(
        self,
        resource_id: int,
        target: str,
        sequence: int,
        intent: str,
        *,
        protect_extra: frozenset[tuple[int, str]] = frozenset(),
    ) -> bool:
        generation = self.generations[resource_id]
        current = self.copies[resource_id].get(target)
        if current is not None and current[0] == generation:
            return False
        resource = self.resources[resource_id]
        route, predicted_ns = self._route_from_current(resource_id, target)
        protect = frozenset((resource_id, bank) for bank in route) | protect_extra
        for intermediate in route[1:-1]:
            self.make_space(intermediate, resource.nbytes, sequence, protect)
            transient = self.used[intermediate] + resource.nbytes
            self.peaks[intermediate] = max(self.peaks[intermediate], transient)
        self.make_space(target, resource.nbytes, sequence, protect)
        host_bounce = any(self.banks[name].channel.lower() == "host" for name in route[1:-1])
        if len(route) > 2:
            route_class = "staged"
        elif (
            self.banks[route[0]].channel.lower() != "host"
            and self.banks[route[-1]].channel.lower() != "host"
        ):
            route_class = "direct-peer"
        else:
            route_class = "direct-dma"
        self.add(
            sequence,
            "transfer",
            resource_id,
            target,
            route=route,
            intent=intent,
            nbytes=resource.nbytes,
            predicted_ns=predicted_ns,
            route_class=route_class,
            host_bounce=host_bounce,
            reason=f"{intent}:{route[0]}->{target}",
        )
        self.copies[resource_id][target] = (generation, self.versions[resource_id])
        self.used[target] = _checked_add(
            self.used[target], resource.nbytes, "HAM destination occupancy"
        )
        self.peaks[target] = max(self.peaks[target], self.used[target])
        self.logical_transfer_bytes = _checked_add(
            self.logical_transfer_bytes, resource.nbytes, "HAM logical transfers"
        )
        wire = _checked_mul(resource.nbytes, len(route) - 1, "HAM wire transfers")
        self.wire_transfer_bytes = _checked_add(
            self.wire_transfer_bytes, wire, "HAM wire transfers"
        )
        self.predicted_transfer_ns = _checked_add(
            self.predicted_transfer_ns, predicted_ns, "HAM transfer latency"
        )
        if route_class == "direct-peer":
            self.direct_peer_bytes = _checked_add(
                self.direct_peer_bytes, resource.nbytes, "HAM direct-peer transfers"
            )
        if route_class == "staged":
            self.staged_transfer_bytes = _checked_add(
                self.staged_transfer_bytes, resource.nbytes, "HAM staged transfers"
            )
            if host_bounce:
                self.host_bounce_bytes = _checked_add(
                    self.host_bounce_bytes, resource.nbytes, "HAM host-bounce transfers"
                )
        if intent == "prefetch":
            self.add(
                sequence,
                "prefetch",
                resource_id,
                target,
                intent="prefetch",
                nbytes=resource.nbytes,
                reason="lookahead:resident-before-use",
            )
        return True

    def _snapshot(self):
        return (
            {resource_id: dict(copies) for resource_id, copies in self.copies.items()},
            dict(self.used),
            dict(self.peaks),
            len(self.actions),
            self.logical_transfer_bytes,
            self.wire_transfer_bytes,
            self.direct_peer_bytes,
            self.staged_transfer_bytes,
            self.host_bounce_bytes,
            self.predicted_transfer_ns,
        )

    def _restore(self, snapshot) -> None:
        (
            self.copies,
            self.used,
            self.peaks,
            length,
            self.logical_transfer_bytes,
            self.wire_transfer_bytes,
            self.direct_peer_bytes,
            self.staged_transfer_bytes,
            self.host_bounce_bytes,
            self.predicted_transfer_ns,
        ) = snapshot
        del self.actions[length:]

    def _prefetch(self, index: int) -> None:
        if self.workload.prefetch_distance == 0:
            return
        stop = min(len(self.workload.accesses), index + 1 + self.workload.prefetch_distance)
        protected: set[tuple[int, str]] = set()
        for future in self.workload.accesses[index + 1 : stop]:
            for resource_id in self.required[future.sequence]:
                current = self.copies[resource_id].get(future.bank)
                if current is not None and current[0] == self.generations[resource_id]:
                    protected.add((resource_id, future.bank))
                    continue
                snapshot = self._snapshot()
                try:
                    self.ensure(
                        resource_id,
                        future.bank,
                        self.workload.accesses[index].sequence,
                        "prefetch",
                        protect_extra=frozenset(protected),
                    )
                    protected.add((resource_id, future.bank))
                except ValueError:
                    self._restore(snapshot)  # prefetch is optional; demand remains fail-closed

    def _cleanup_dead(self, sequence: int) -> None:
        for resource_id in sorted(self.copies):
            resource = self.resources[resource_id]
            for bank in sorted(tuple(self.copies[resource_id])):
                if bank == resource.home_bank:
                    continue
                if self._next_use(resource_id, bank, sequence) is None:
                    self.evict(resource_id, bank, sequence, "lifetime:last-use")

    def run(self) -> None:
        for index, access in enumerate(self.workload.accesses):
            for resource_id in self.required[access.sequence]:
                self.ensure(resource_id, access.bank, access.sequence, "demand")
            resource = self.resources[access.resource_id]
            before = self.generations[access.resource_id]
            if access.mode == "write":
                after = before + 1
                version = _sha(
                    {
                        "previous_version": self.versions[access.resource_id],
                        "sequence": access.sequence,
                        "operation": access.operation,
                    }
                )
            else:
                after = before
                version = self.versions[access.resource_id]
            self.add(
                access.sequence,
                "barrier",
                access.resource_id,
                access.bank,
                before=before,
                after=before,
                version=self.versions[access.resource_id],
                reason="residency:complete-before-access",
            )
            self.add(
                access.sequence,
                "access",
                access.resource_id,
                access.bank,
                mode=access.mode,
                nbytes=resource.nbytes,
                before=before,
                after=after,
                version=version,
                reason=access.operation,
            )
            if access.mode == "write":
                self.generations[access.resource_id] = after
                self.versions[access.resource_id] = version
                self.copies[access.resource_id][access.bank] = (after, version)
                for bank in sorted(tuple(self.copies[access.resource_id])):
                    if bank != access.bank:
                        self.evict(
                            access.resource_id,
                            bank,
                            access.sequence,
                            "write:invalidate-stale-copy",
                            invalidate=True,
                        )
                if access.bank != resource.home_bank:
                    self.ensure(
                        access.resource_id, resource.home_bank, access.sequence, "writeback"
                    )
                    self.add(
                        access.sequence,
                        "barrier",
                        access.resource_id,
                        resource.home_bank,
                        reason="writeback:quiescent-home-copy",
                    )
            self._cleanup_dead(access.sequence)
            self._prefetch(index)
        final_sequence = len(self.workload.accesses) + 1
        for resource_id in sorted(self.copies):
            resource = self.resources[resource_id]
            for bank in sorted(tuple(self.copies[resource_id])):
                if bank != resource.home_bank:
                    self.evict(resource_id, bank, final_sequence, "plan:final-cleanup")


def _route_properties(
    route: tuple[str, ...], nbytes: int, banks: dict, links: dict
) -> tuple[int, str, bool]:
    predicted = 0
    for source, destination in zip(route, route[1:]):
        link = links.get((source, destination))
        if link is None:
            raise ValueError(f"undeclared HAM route edge {source!r}->{destination!r}")
        predicted = _checked_add(predicted, _link_ns(link, nbytes), "HAM route latency")
    host_bounce = any(banks[name].channel.lower() == "host" for name in route[1:-1])
    if len(route) > 2:
        route_class = "staged"
    elif banks[route[0]].channel.lower() != "host" and banks[route[-1]].channel.lower() != "host":
        route_class = "direct-peer"
    else:
        route_class = "direct-dma"
    return predicted, route_class, host_bounce


def _replay(
    actions: tuple[HAMAction, ...],
    peaks: tuple[HAMBankPeak, ...],
    summary: tuple[int, int, int, int, int, int],
    workload: HAMWorkload,
    hardware,
) -> list[str]:
    errors: list[str] = []
    try:
        banks, links, _adjacency = _hardware_inventory(hardware, workload)
    except ValueError as exc:
        return [str(exc)]
    resources = {row.resource_id: row for row in workload.resources}
    generations = {row.resource_id: 0 for row in workload.resources}
    versions = {row.resource_id: row.content_sha256 for row in workload.resources}
    copies = {
        row.resource_id: {row.home_bank: (0, row.content_sha256)} for row in workload.resources
    }
    used = {name: 0 for name in banks}
    observed_peaks = dict(used)
    for row in workload.resources:
        used[row.home_bank] += row.nbytes
        observed_peaks[row.home_bank] = max(observed_peaks[row.home_bank], used[row.home_bank])
    access_index = 0
    logical = wire = direct = staged = bounced = predicted_total = 0
    required = {row.sequence: _closure(workload, row.resource_id) for row in workload.accesses}
    pending_access_barrier = None
    pending_transfer_marker = None
    for action in actions:
        if pending_access_barrier is not None and action.kind != "access":
            errors.append(
                f"action {action.order}: residency barrier is not followed "
                "immediately by its access"
            )
            pending_access_barrier = None
        if pending_transfer_marker is not None:
            expected_kind, expected_resource, expected_bank, expected_sequence = (
                pending_transfer_marker
            )
            if (action.kind, action.resource_id, action.bank, action.sequence) != (
                expected_kind,
                expected_resource,
                expected_bank,
                expected_sequence,
            ):
                errors.append(
                    f"action {action.order}: transfer is missing its "
                    f"{expected_kind} completion marker"
                )
            pending_transfer_marker = None
        resource = resources.get(action.resource_id)
        if resource is None:
            errors.append(f"action {action.order}: unknown resource {action.resource_id}")
            continue
        if action.kind == "transfer":
            if action.nbytes != resource.nbytes:
                errors.append(f"action {action.order}: transfer size does not match resource")
            if len(action.route) - 1 > workload.max_route_hops:
                errors.append(f"action {action.order}: route exceeds workload hop bound")
            if action.route[0] not in copies[action.resource_id]:
                errors.append(f"action {action.order}: transfer source copy is absent")
                continue
            source_generation, source_version = copies[action.resource_id][action.route[0]]
            if (
                source_generation != generations[action.resource_id]
                or source_generation != action.generation_before
                or source_version != action.version_sha256
            ):
                errors.append(f"action {action.order}: transfer source is stale")
            try:
                expected_ns, route_class, host_bounce = _route_properties(
                    action.route, resource.nbytes, banks, links
                )
            except ValueError as exc:
                errors.append(f"action {action.order}: {exc}")
                continue
            if (
                expected_ns != action.predicted_ns
                or route_class != action.route_class
                or host_bounce != action.host_bounce
            ):
                errors.append(f"action {action.order}: route metadata does not re-derive")
            for intermediate in action.route[1:-1]:
                transient = used[intermediate] + resource.nbytes
                observed_peaks[intermediate] = max(observed_peaks[intermediate], transient)
                if transient > banks[intermediate].allocatable_bytes:
                    errors.append(
                        f"action {action.order}: transient capacity exceeded in {intermediate!r}"
                    )
            destination = action.route[-1]
            if destination not in copies[action.resource_id]:
                used[destination] += resource.nbytes
            copies[action.resource_id][destination] = (
                generations[action.resource_id],
                versions[action.resource_id],
            )
            observed_peaks[destination] = max(observed_peaks[destination], used[destination])
            logical += resource.nbytes
            wire += resource.nbytes * (len(action.route) - 1)
            predicted_total += expected_ns
            if route_class == "direct-peer":
                direct += resource.nbytes
            if route_class == "staged":
                staged += resource.nbytes
                if host_bounce:
                    bounced += resource.nbytes
            if action.intent == "prefetch":
                pending_transfer_marker = (
                    "prefetch",
                    action.resource_id,
                    destination,
                    action.sequence,
                )
            elif action.intent == "writeback":
                pending_transfer_marker = (
                    "barrier",
                    action.resource_id,
                    destination,
                    action.sequence,
                )
        elif action.kind == "prefetch":
            if action.nbytes != resource.nbytes:
                errors.append(f"action {action.order}: prefetch size does not match resource")
            if copies[action.resource_id].get(action.bank) != (
                generations[action.resource_id],
                versions[action.resource_id],
            ):
                errors.append(f"action {action.order}: prefetch target is not current")
            if (
                action.generation_before != generations[action.resource_id]
                or action.generation_after != generations[action.resource_id]
                or action.version_sha256 != versions[action.resource_id]
            ):
                errors.append(f"action {action.order}: prefetch lineage is stale")
        elif action.kind == "barrier":
            if copies[action.resource_id].get(action.bank) != (
                generations[action.resource_id],
                versions[action.resource_id],
            ):
                errors.append(f"action {action.order}: barrier target is not current")
            if (
                action.generation_before != generations[action.resource_id]
                or action.generation_after != generations[action.resource_id]
                or action.version_sha256 != versions[action.resource_id]
            ):
                errors.append(f"action {action.order}: barrier lineage is stale")
            if action.reason == "residency:complete-before-access":
                pending_access_barrier = (action.sequence, action.resource_id, action.bank)
            elif action.reason == "writeback:quiescent-home-copy":
                if action.bank != resource.home_bank:
                    errors.append(f"action {action.order}: writeback barrier is not at home")
            else:
                errors.append(f"action {action.order}: unknown barrier purpose")
        elif action.kind == "access":
            if action.nbytes != resource.nbytes:
                errors.append(f"action {action.order}: access size does not match resource")
            if access_index >= len(workload.accesses):
                errors.append(f"action {action.order}: extra access action")
                continue
            expected = workload.accesses[access_index]
            access_index += 1
            if pending_access_barrier != (expected.sequence, expected.resource_id, expected.bank):
                errors.append(f"action {action.order}: access lacks its residency barrier")
            pending_access_barrier = None
            if (action.sequence, action.resource_id, action.bank, action.mode, action.reason) != (
                expected.sequence,
                expected.resource_id,
                expected.bank,
                expected.mode,
                expected.operation,
            ):
                errors.append(f"action {action.order}: access does not match workload event")
            for dependency in required[expected.sequence]:
                if copies[dependency].get(expected.bank) != (
                    generations[dependency],
                    versions[dependency],
                ):
                    errors.append(
                        f"action {action.order}: resource {dependency} is not current "
                        f"in {expected.bank!r}"
                    )
            if action.generation_before != generations[action.resource_id]:
                errors.append(f"action {action.order}: access generation is stale")
            if action.mode == "write":
                expected_version = _sha(
                    {
                        "previous_version": versions[action.resource_id],
                        "sequence": action.sequence,
                        "operation": action.reason,
                    }
                )
                if (
                    action.generation_after != generations[action.resource_id] + 1
                    or action.version_sha256 != expected_version
                ):
                    errors.append(f"action {action.order}: write lineage does not re-derive")
                generations[action.resource_id] += 1
                versions[action.resource_id] = expected_version
                copies[action.resource_id][action.bank] = (
                    generations[action.resource_id],
                    expected_version,
                )
            elif (
                action.generation_after != generations[action.resource_id]
                or action.version_sha256 != versions[action.resource_id]
            ):
                errors.append(f"action {action.order}: read lineage is inconsistent")
        elif action.kind in {"invalidate", "evict"}:
            if action.nbytes != resource.nbytes:
                errors.append(f"action {action.order}: removal size does not match resource")
            copy = copies[action.resource_id].get(action.bank)
            if copy is None:
                errors.append(f"action {action.order}: removal target is absent")
            else:
                copy_generation, copy_version = copy
                if (
                    copy_generation != action.generation_before
                    or copy_version != action.version_sha256
                ):
                    errors.append(f"action {action.order}: removal metadata is stale")
                if action.generation_after != generations[action.resource_id]:
                    errors.append(f"action {action.order}: removal target generation is stale")
                if action.kind == "evict" and copy_generation != generations[action.resource_id]:
                    errors.append(f"action {action.order}: eviction removes a stale copy")
                if (
                    action.kind == "invalidate"
                    and copy_generation == generations[action.resource_id]
                ):
                    errors.append(f"action {action.order}: invalidation removes a current copy")
                if action.kind == "evict" and action.bank == resource.home_bank:
                    errors.append(f"action {action.order}: pinned home copy was evicted")
                copies[action.resource_id].pop(action.bank)
                used[action.bank] -= resource.nbytes
        for name, value in used.items():
            observed_peaks[name] = max(observed_peaks[name], value)
            if not 0 <= value <= banks[name].allocatable_bytes:
                errors.append(f"action {action.order}: occupancy {value} is invalid in {name!r}")
    if access_index != len(workload.accesses):
        errors.append("HAM plan does not execute every workload access")
    if pending_access_barrier is not None:
        errors.append("HAM plan ends after a residency barrier without its access")
    if pending_transfer_marker is not None:
        errors.append("HAM plan ends before a transfer completion marker")
    for resource in workload.resources:
        expected = (generations[resource.resource_id], versions[resource.resource_id])
        if copies[resource.resource_id] != {resource.home_bank: expected}:
            errors.append(
                f"HAM plan does not finish with one current home copy of resource "
                f"{resource.resource_id}"
            )
    expected_peaks = tuple(
        HAMBankPeak(name, observed_peaks[name], banks[name].allocatable_bytes)
        for name in sorted(banks)
    )
    if peaks != expected_peaks:
        errors.append("HAM plan peak inventory does not re-derive")
    if summary != (logical, wire, direct, staged, bounced, predicted_total):
        errors.append("HAM plan transfer summary does not re-derive")
    return errors


def verify_ham_plan(plan: HAMExecutionPlan, workload: HAMWorkload, hardware) -> tuple[str, ...]:
    """Independently replay residency, generations, routes, and capacities."""
    if not isinstance(plan, HAMExecutionPlan) or not isinstance(workload, HAMWorkload):
        return ("HAM verification requires typed plan and workload artifacts",)
    errors = []
    if plan.hardware_digest != workload.hardware_digest or plan.hardware_digest != getattr(
        hardware, "digest", None
    ):
        errors.append("HAM plan hardware digest mismatch")
    if plan.workload_digest != workload.digest:
        errors.append("HAM plan workload digest mismatch")
    if plan.policy_digest != workload.policy_digest:
        errors.append("HAM plan policy digest mismatch")
    errors += _replay(
        plan.actions,
        plan.peaks,
        (
            plan.logical_transfer_bytes,
            plan.wire_transfer_bytes,
            plan.direct_peer_bytes,
            plan.staged_transfer_bytes,
            plan.host_bounce_bytes,
            plan.predicted_transfer_ns,
        ),
        workload,
        hardware,
    )
    return tuple(errors)


def _device_manifest(hardware) -> DeviceManifest:
    banks = tuple(
        MemoryBank(row.name, Domain[row.domain], row.capacity_bytes, row.native_tile, row.alignment)
        for row in hardware.banks
    )
    distance = tuple(
        tuple(0 if left == right else 256 for right in range(len(banks)))
        for left in range(len(banks))
    )
    manifest = DeviceManifest(hardware.name, banks, distance, hardware.architecture, 0)
    errors = check_device_manifest(manifest)
    if errors:
        raise ValueError("invalid HAM device projection: " + "; ".join(errors))
    return manifest


def _target(hardware, channels: set[str]) -> TargetProfile:
    lowered = {value.lower() for value in channels}
    architecture = hardware.architecture.lower()
    if any("nvidia" in value or "ptx" in value or value == "gpu" for value in lowered):
        return TargetProfile.nvidia_ptx()
    if "aarch64" in architecture or "arm64" in architecture:
        return TargetProfile.arm64_neon()
    if "riscv" in architecture:
        return TargetProfile.riscv_rvv()
    return TargetProfile.x86_avx2()


def _lower(actions: tuple[HAMAction, ...], workload: HAMWorkload, hardware):
    manifest = _device_manifest(hardware)
    resources = {row.resource_id: row for row in workload.resources}
    pairs = {(row.resource_id, row.home_bank) for row in workload.resources}
    for action in actions:
        if action.bank:
            pairs.add((action.resource_id, action.bank))
        for bank in action.route:
            pairs.add((action.resource_id, bank))
    pair_rids = {pair: index + 1 for index, pair in enumerate(sorted(pairs))}
    module = Module(name=f"ham:{workload.digest[:16]}", target=hardware.architecture)
    for (resource_id, bank_name), rid in pair_rids.items():
        semantic = resources[resource_id]
        bank = hardware.bank(bank_name)
        module.add_resource(
            Resource(
                rid,
                Domain[bank.domain],
                1,
                (semantic.nbytes,),
                align=bank.alignment,
                access="ham",
                priority=semantic.priority,
                name=f"ham:{resource_id}:{bank_name}",
            )
        )
    previous_phase = 0
    phase_id = 0
    claim_id = 0
    claim_channels: dict[int, str] = {}

    def emit(claim: Claim, channel: str) -> None:
        nonlocal previous_phase, phase_id
        phase_id += 1
        module.add_phase(Phase(phase_id, (previous_phase,) if previous_phase else (), [claim]))
        previous_phase = phase_id
        claim_channels[claim.id] = channel

    def cid() -> int:
        nonlocal claim_id
        claim_id += 1
        return claim_id

    for action in actions:
        if action.kind == "transfer":
            for source, destination in zip(action.route, action.route[1:]):
                claim = move_claim(
                    manifest,
                    source,
                    destination,
                    pair_rids[(action.resource_id, source)],
                    pair_rids[(action.resource_id, destination)],
                    action.nbytes,
                    cid(),
                )
                claim.op += f":{action.intent}"
                emit(claim, hardware.bank(destination).channel)
        elif action.kind == "prefetch":
            rid = pair_rids[(action.resource_id, action.bank)]
            bank = hardware.bank(action.bank)
            emit(
                Claim(
                    cid(),
                    Opcode.LOAD,
                    Lane.U,
                    StrideClass.UNIT,
                    count=action.nbytes,
                    rd=(rid,),
                    domain=Domain[bank.domain],
                    bounds="assumed_safe",
                    op="ham.prefetch",
                    cost_class="bandwidth",
                ),
                bank.channel,
            )
        elif action.kind == "barrier":
            bank = hardware.bank(action.bank)
            emit(
                Claim(
                    cid(),
                    Opcode.BARRIER,
                    Lane.H,
                    StrideClass.SCALAR,
                    hazard="barriered",
                    domain=Domain[bank.domain],
                    bounds="assumed_safe",
                    op="ham.barrier",
                    cost_class="latency",
                ),
                bank.channel,
            )
        elif action.kind == "access":
            rid = pair_rids[(action.resource_id, action.bank)]
            bank = hardware.bank(action.bank)
            opcode = Opcode.LOAD if action.mode == "read" else Opcode.STORE
            emit(
                Claim(
                    cid(),
                    opcode,
                    Lane.U,
                    StrideClass.UNIT,
                    count=action.nbytes,
                    rd=(rid,),
                    wr=() if action.mode == "read" else (rid,),
                    hazard="barriered" if action.mode == "write" else "unique",
                    domain=Domain[bank.domain],
                    bounds="assumed_safe",
                    op=f"ham.{action.mode}:{action.reason}",
                    cost_class="bandwidth",
                ),
                bank.channel,
            )
        else:  # explicit eviction / stale-copy invalidation
            rid = pair_rids[(action.resource_id, action.bank)]
            bank = hardware.bank(action.bank)
            emit(
                Claim(
                    cid(),
                    Opcode.NOP,
                    Lane.H,
                    StrideClass.SCALAR,
                    rd=(rid,),
                    domain=Domain[bank.domain],
                    bounds="assumed_safe",
                    op=f"ham.{action.kind}",
                    cost_class="bandwidth",
                ),
                bank.channel,
            )
    target = _target(hardware, set(claim_channels.values()))
    result = optimize(module, target, Theta.cool())
    pack = hydrate_pipelined(module, result, plan=f"ham:{workload.digest[:16]}", depth=2)
    pack.segments = [
        replace(segment, channel=claim_channels[segment.claim_id]) for segment in pack.segments
    ]
    diagnostics = verify_all(module, result=result, pack=pack, h=target, theta=Theta.cool())
    diagnostics += verify_smart_lowering(module, pack=pack)
    bank_errors = check_bank_moves(module)
    if diagnostics or bank_errors:
        messages = [f"{row.law}: {row.message}" for row in diagnostics] + bank_errors
        raise ValueError("HAM lowering failed verification: " + "; ".join(messages))
    return module, result, pack, encode(pack)


def lower_ham_workload(workload: HAMWorkload, hardware) -> LoweredHAMExecution:
    """Plan, replay, lower, and verify one metadata-only HAM workload."""
    if not isinstance(workload, HAMWorkload):
        raise ValueError("HAM lowering requires a HAMWorkload")
    planner = _Planner(workload, hardware)
    planner.run()
    peaks = tuple(
        HAMBankPeak(name, planner.peaks[name], planner.banks[name].allocatable_bytes)
        for name in sorted(planner.banks)
    )
    summary = (
        planner.logical_transfer_bytes,
        planner.wire_transfer_bytes,
        planner.direct_peer_bytes,
        planner.staged_transfer_bytes,
        planner.host_bounce_bytes,
        planner.predicted_transfer_ns,
    )
    replay_errors = _replay(tuple(planner.actions), peaks, summary, workload, hardware)
    if replay_errors:
        raise ValueError("HAM planner produced an invalid draft: " + "; ".join(replay_errors))
    module, result, pack, pack_data = _lower(tuple(planner.actions), workload, hardware)
    action_digest = _sha([{**asdict(row), "route": list(row.route)} for row in planner.actions])
    module_digest = hashlib.sha256(f"{hash_module(module)}:{action_digest}".encode()).hexdigest()
    artifact = HAMExecutionPlan(
        workload.hardware_digest,
        workload.digest,
        workload.policy_digest,
        tuple(planner.actions),
        peaks,
        *summary,
        module_digest,
        hashlib.sha256(pack_data).hexdigest(),
        len(pack_data),
    )
    errors = verify_ham_plan(artifact, workload, hardware)
    if errors:
        raise ValueError("HAM execution plan failed replay: " + "; ".join(errors))
    return LoweredHAMExecution(artifact, module, result, pack, pack_data)


__all__ = [
    "HAMAccess",
    "HAMAction",
    "HAMBankPeak",
    "HAMExecutionPlan",
    "HAMResource",
    "HAMWorkload",
    "LoweredHAMExecution",
    "lower_ham_workload",
    "verify_ham_plan",
]
