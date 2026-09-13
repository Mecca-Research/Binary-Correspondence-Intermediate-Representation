"""The workload component `W` (G13 / S2-E): a declared model of the work a plan is for.

The scope table (roadmap section 1) marked `W` "not modelled", and best-fit dispatch cannot
fit work it cannot see: the same program under an interactive latency budget and under a
batch of eight at full concurrency is two different problems, and a plan certified for one
must not be able to carry its certificate to the other. `Workload` is the declaration --
shapes, batch, concurrency, the service-level requirement, the horizon and the input
distribution -- as integers and names only, so it can enter `ExecutionScopeV1`
(`kbcir.scope`, which refuses floats) and be digested; `workload_for` derives the shapes from
the module the plan is for; `verify_workload` holds a declaration to the module it describes.

A workload is a *declared input*, never an inference: nothing here reads telemetry or a
corpus. What the declared workload buys is (1) a scope digest that separates two workloads
on one program, so certificates cannot be carried across them, (2) a class the L2 portfolio's
plan-time table can key on next to the runtime class (`kbcir.portfolio`), and (3) the key
under which measured evidence is filed and replayed (`kbcir.measured`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

#: Bump when the serialization changes shape; a workload digest is comparable only within a
#: version, exactly as a scope digest is.
WORKLOAD_VERSION = "WorkloadV1"

#: The service-level requirement a workload declares. `latency` names a per-invocation budget
#: in nanoseconds; `throughput` a sustained rate per second; `best-effort` names neither.
SERVICE_LEVELS = ("best-effort", "latency", "throughput")

#: The input distribution: `static` prices every dynamic claim at its declared upper bound;
#: `dynamic` declares the expected count of each dynamic claim under this workload.
DISTRIBUTIONS = ("static", "dynamic")

#: The classes the L2 plan-time table keys on (next to the runtime class of Theta).
WORKLOAD_CLASSES = ("nominal", "interactive", "batch")


def _int(value, what: str, *, lo: int = 0) -> int:
    if type(value) is not int or value < lo:
        raise ValueError(f"{what} must be an integer >= {lo}")
    return value


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Workload:
    """The declared workload: what the plan is for, as a digested identity."""

    #: (rid, shape) for every resource the program declares, in rid order.
    shapes: tuple[tuple[int, tuple[int, ...]], ...] = ()
    #: Invocations per call (a batch of one is a single invocation).
    batch: int = 1
    #: Concurrent instances of the program sharing the target.
    concurrency: int = 1
    service_level: str = "best-effort"
    #: The latency budget per invocation, nanoseconds; nonzero iff `service_level` is `latency`.
    latency_ns: int = 0
    #: The sustained rate, invocations per second; nonzero iff `service_level` is `throughput`.
    throughput_per_s: int = 0
    #: The number of invocations the plan is for (the length of the episode).
    horizon: int = 1
    distribution: str = "static"
    #: (claim id, expected count) for dynamic claims, in claim-id order; nonempty iff `dynamic`.
    expected_counts: tuple[tuple[int, int], ...] = ()
    version: str = WORKLOAD_VERSION

    def __post_init__(self) -> None:
        if self.version != WORKLOAD_VERSION:
            raise ValueError(f"unsupported workload version {self.version!r}")
        shapes = []
        seen: set[int] = set()
        for entry in self.shapes:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("a workload shape is a (rid, dims) pair")
            rid, dims = entry
            _int(rid, "resource id")
            if rid in seen:
                raise ValueError(f"resource {rid} is declared twice")
            seen.add(rid)
            if not isinstance(dims, (tuple, list)):
                raise ValueError("a shape is a tuple of extents")
            shapes.append((rid, tuple(_int(dim, "shape extent", lo=1) for dim in dims)))
        if [rid for rid, _ in shapes] != sorted(seen):
            raise ValueError("workload shapes must be declared in resource-id order")
        object.__setattr__(self, "shapes", tuple(shapes))
        _int(self.batch, "batch", lo=1)
        _int(self.concurrency, "concurrency", lo=1)
        _int(self.horizon, "horizon", lo=1)
        if self.service_level not in SERVICE_LEVELS:
            raise ValueError(f"service level must be one of {SERVICE_LEVELS}")
        _int(self.latency_ns, "latency_ns")
        _int(self.throughput_per_s, "throughput_per_s")
        if (self.service_level == "latency") != (self.latency_ns > 0):
            raise ValueError("a latency service level names a positive budget, and only it does")
        if (self.service_level == "throughput") != (self.throughput_per_s > 0):
            raise ValueError("a throughput service level names a positive rate, and only it does")
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {DISTRIBUTIONS}")
        counts = []
        claims: set[int] = set()
        for entry in self.expected_counts:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("an expected count is a (claim id, count) pair")
            cid, count = entry
            _int(cid, "claim id")
            if cid in claims:
                raise ValueError(f"claim {cid} has two expected counts")
            claims.add(cid)
            counts.append((cid, _int(count, "expected count")))
        if [cid for cid, _ in counts] != sorted(claims):
            raise ValueError("expected counts must be declared in claim-id order")
        object.__setattr__(self, "expected_counts", tuple(counts))
        if (self.distribution == "dynamic") != bool(counts):
            raise ValueError("a dynamic distribution declares expected counts, and only it does")

    def component(self) -> dict:
        """The `W` component as the scope serializes it (integers, names, lists)."""
        return {
            "version": self.version,
            "shapes": [[rid, list(dims)] for rid, dims in self.shapes],
            "batch": self.batch,
            "concurrency": self.concurrency,
            "service_level": self.service_level,
            "latency_ns": self.latency_ns,
            "throughput_per_s": self.throughput_per_s,
            "horizon": self.horizon,
            "distribution": self.distribution,
            "expected_counts": [[cid, count] for cid, count in self.expected_counts],
        }

    def to_json(self) -> str:
        return _canonical(self.component())

    def digest(self) -> str:
        """The workload's content address: SHA-256 over the canonical component."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def classify(self) -> str:
        """The plan-time class: `interactive` under a latency budget, `batch` under a
        throughput target or any batching or concurrency, `nominal` otherwise."""
        if self.service_level == "latency":
            return "interactive"
        if self.service_level == "throughput" or self.batch > 1 or self.concurrency > 1:
            return "batch"
        return "nominal"

    @staticmethod
    def from_dict(value) -> "Workload":
        expected = {
            "version",
            "shapes",
            "batch",
            "concurrency",
            "service_level",
            "latency_ns",
            "throughput_per_s",
            "horizon",
            "distribution",
            "expected_counts",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("workload has missing or unknown fields")
        try:
            return Workload(
                shapes=tuple((rid, tuple(dims)) for rid, dims in value["shapes"]),
                batch=value["batch"],
                concurrency=value["concurrency"],
                service_level=value["service_level"],
                latency_ns=value["latency_ns"],
                throughput_per_s=value["throughput_per_s"],
                horizon=value["horizon"],
                distribution=value["distribution"],
                expected_counts=tuple((cid, n) for cid, n in value["expected_counts"]),
                version=value["version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed workload: {exc}") from exc


def workload_for(
    module,
    *,
    batch: int = 1,
    concurrency: int = 1,
    service_level: str = "best-effort",
    latency_ns: int = 0,
    throughput_per_s: int = 0,
    horizon: int = 1,
    expected_counts: dict[int, int] | None = None,
) -> Workload:
    """A workload over `module`'s declared shapes with the rest declared by the caller. An
    `expected_counts` mapping (claim id -> count) declares a dynamic distribution."""
    shapes = tuple((rid, tuple(module.resources[rid].shape)) for rid in sorted(module.resources))
    counts = tuple(sorted((cid, n) for cid, n in (expected_counts or {}).items()))
    return Workload(
        shapes=shapes,
        batch=batch,
        concurrency=concurrency,
        service_level=service_level,
        latency_ns=latency_ns,
        throughput_per_s=throughput_per_s,
        horizon=horizon,
        distribution="dynamic" if counts else "static",
        expected_counts=counts,
    )


def verify_workload(workload: Workload, module) -> list[str]:
    """The declaration against the module it claims to describe: every declared shape is a
    resource of the module with that shape and no resource is missing; every expected count
    names a dynamic claim and stays within its declared bound. Returns the problems."""
    problems: list[str] = []
    declared = dict(workload.shapes)
    for rid in sorted(module.resources):
        shape = tuple(module.resources[rid].shape)
        if rid not in declared:
            problems.append(f"resource {rid} has no declared shape")
        elif declared[rid] != shape:
            problems.append(f"resource {rid} is declared {declared[rid]} but has shape {shape}")
    for rid in sorted(declared):
        if rid not in module.resources:
            problems.append(f"declared resource {rid} is not in the module")
    claims = {claim.id: claim for phase in module.phases for claim in phase.claims}
    for cid, count in workload.expected_counts:
        claim = claims.get(cid)
        if claim is None:
            problems.append(f"expected count names claim {cid}, which is not in the module")
        elif not claim.dynamic:
            problems.append(f"claim {cid} is not dynamic: its count is static")
        elif count > claim.count:
            problems.append(f"claim {cid} expects {count} above its declared bound {claim.count}")
    return problems


__all__ = [
    "DISTRIBUTIONS",
    "SERVICE_LEVELS",
    "WORKLOAD_CLASSES",
    "WORKLOAD_VERSION",
    "Workload",
    "verify_workload",
    "workload_for",
]
