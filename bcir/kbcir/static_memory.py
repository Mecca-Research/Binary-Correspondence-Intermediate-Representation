"""Verified static tensor-address planning for BCIR claim graphs.

The planner assigns every touched resource a byte offset inside one declared hardware
bank. Resources with disjoint phase lifetimes may reuse the same address range;
simultaneously-live resources may not. This is a deterministic compiler artifact, not a
runtime allocator and not a learned legality decision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .._artifact_json import strict_json_loads
from ..model import Module
from .allocator import live_intervals
from .provenance import hash_module

_SCHEMA = "bcir.static_memory_plan.v1"
_MAX_JSON = 16 * 1024 * 1024
# The model-plan lowerer admits at most 4096 layers.  A two-times margin covers
# its per-layer resources while keeping the deterministic reference allocator
# safely bounded on malformed or synthetic inputs.
_MAX_RESOURCES = 8192
_MAX_BANKS = 32
_MAX_U63 = (1 << 63) - 1


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _module_digest(module: Module) -> str:
    return hashlib.sha256(str(hash_module(module)).encode("ascii")).hexdigest()


def _integer(value, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_U63:
        raise ValueError(f"{field} must be an integer in [{minimum}, {_MAX_U63}]")
    return value


def _name(value, field: str) -> str:
    if (not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4096
            or any(ord(character) < 0x20 for character in value)):
        raise ValueError(f"{field} must be a bounded, control-free string")
    return value


def _digest(value, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _checked_add(left: int, right: int) -> int:
    if left < 0 or right < 0 or left > _MAX_U63 - right:
        raise ValueError("static memory accounting exceeds signed 64-bit range")
    return left + right


def _align(value: int, alignment: int) -> int:
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError("static memory alignment must be a positive power of two")
    return _checked_add(value, alignment - 1) & ~(alignment - 1)


@dataclass(frozen=True)
class ResourceBankBinding:
    rid: int
    bank: str

    def __post_init__(self) -> None:
        _integer(self.rid, "binding rid", minimum=1)
        _name(self.bank, "binding bank")


@dataclass(frozen=True)
class StaticAllocation:
    rid: int
    bank: str
    offset: int
    size_bytes: int
    alignment: int
    first_phase: int
    last_phase: int

    def __post_init__(self) -> None:
        _integer(self.rid, "allocation rid", minimum=1)
        _name(self.bank, "allocation bank")
        for field in ("offset", "size_bytes", "alignment", "first_phase", "last_phase"):
            _integer(getattr(self, field), f"allocation {field}",
                     minimum=1 if field in ("size_bytes", "alignment") else 0)
        if self.alignment & (self.alignment - 1):
            raise ValueError("allocation alignment must be a power of two")
        if self.offset % self.alignment:
            raise ValueError("allocation offset violates alignment")
        if self.last_phase < self.first_phase:
            raise ValueError("allocation lifetime is reversed")
        _checked_add(self.offset, self.size_bytes)

    @property
    def end(self) -> int:
        return self.offset + self.size_bytes


@dataclass(frozen=True)
class BankMemoryPlan:
    bank: str
    extent_bytes: int
    naive_bytes: int  # aligned sequential layout without lifetime reuse
    capacity_bytes: int

    def __post_init__(self) -> None:
        _name(self.bank, "bank plan name")
        for field in ("extent_bytes", "naive_bytes", "capacity_bytes"):
            _integer(getattr(self, field), f"bank plan {field}")
        if self.extent_bytes > self.capacity_bytes:
            raise ValueError(f"static address extent exceeds bank {self.bank!r} capacity")
        if self.extent_bytes > self.naive_bytes:
            raise ValueError("static address plan cannot exceed its naive footprint")

    @property
    def saved_bytes(self) -> int:
        return self.naive_bytes - self.extent_bytes


@dataclass(frozen=True)
class StaticMemoryPlan:
    module_digest: str
    hardware_digest: str
    allocations: tuple[StaticAllocation, ...]
    banks: tuple[BankMemoryPlan, ...]
    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("unsupported static memory plan schema")
        _digest(self.module_digest, "static plan module_digest")
        _digest(self.hardware_digest, "static plan hardware_digest")
        if (not isinstance(self.allocations, tuple) or not self.allocations
                or len(self.allocations) > _MAX_RESOURCES
                or any(not isinstance(row, StaticAllocation) for row in self.allocations)):
            raise ValueError("static memory plan needs a bounded allocation tuple")
        if (not isinstance(self.banks, tuple) or not self.banks
                or len(self.banks) > _MAX_BANKS
                or any(not isinstance(row, BankMemoryPlan) for row in self.banks)):
            raise ValueError("static memory plan needs typed bank summaries")
        if len({row.rid for row in self.allocations}) != len(self.allocations):
            raise ValueError("static memory plan has duplicate resource allocations")
        if len({row.bank for row in self.banks}) != len(self.banks):
            raise ValueError("static memory plan has duplicate bank summaries")
        if tuple(sorted(self.allocations, key=lambda row: row.rid)) != self.allocations:
            raise ValueError("static memory allocations must be sorted by rid")
        if tuple(sorted(self.banks, key=lambda row: row.bank)) != self.banks:
            raise ValueError("static memory bank summaries must be sorted by name")

    def to_dict(self) -> dict:
        return {"schema": self.schema, "module_digest": self.module_digest,
                "hardware_digest": self.hardware_digest,
                "allocations": [asdict(row) for row in self.allocations],
                "banks": [asdict(row) for row in self.banks]}

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "StaticMemoryPlan":
        doc = strict_json_loads(text, "static memory plan", max_bytes=_MAX_JSON)
        expected = {"schema", "module_digest", "hardware_digest", "allocations", "banks"}
        if not isinstance(doc, dict) or set(doc) != expected:
            raise ValueError("static memory plan has missing or unknown fields")
        if (not isinstance(doc["allocations"], list) or not doc["allocations"]
                or len(doc["allocations"]) > _MAX_RESOURCES
                or not isinstance(doc["banks"], list) or not doc["banks"]
                or len(doc["banks"]) > _MAX_BANKS):
            raise ValueError("static memory plan arrays are empty, malformed, or unbounded")
        try:
            allocations = tuple(StaticAllocation(**row) for row in doc.pop("allocations"))
            banks = tuple(BankMemoryPlan(**row) for row in doc.pop("banks"))
            return cls(**doc, allocations=allocations, banks=banks)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed static memory plan: {exc}") from exc


def _bindings(value) -> dict[int, str]:
    if isinstance(value, dict):
        rows = tuple(ResourceBankBinding(rid, bank) for rid, bank in value.items())
    elif isinstance(value, (tuple, list)):
        rows = tuple(value)
        if any(not isinstance(row, ResourceBankBinding) for row in rows):
            raise ValueError("resource bank bindings must be typed rows")
    else:
        raise ValueError("resource bank bindings must be a mapping or sequence")
    if len(rows) > _MAX_RESOURCES or len({row.rid for row in rows}) != len(rows):
        raise ValueError("resource bank bindings are duplicate or unbounded")
    return {row.rid: row.bank for row in rows}


class _RangeMaximum:
    """Range-add/range-maximum tree used by the independent alias verifier."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.maximum = [0] * (4 * size)
        self.lazy = [0] * (4 * size)

    def add(self, left: int, right: int, delta: int, node: int = 1,
            low: int = 0, high: int | None = None) -> None:
        if high is None:
            high = self.size - 1
        if right < low or high < left:
            return
        if left <= low and high <= right:
            self.maximum[node] += delta
            self.lazy[node] += delta
            return
        middle = (low + high) // 2
        self.add(left, right, delta, node * 2, low, middle)
        self.add(left, right, delta, node * 2 + 1, middle + 1, high)
        self.maximum[node] = self.lazy[node] + max(
            self.maximum[node * 2], self.maximum[node * 2 + 1])

    def query(self, left: int, right: int, node: int = 1,
              low: int = 0, high: int | None = None) -> int:
        if high is None:
            high = self.size - 1
        if right < low or high < left:
            return 0
        if left <= low and high <= right:
            return self.maximum[node]
        middle = (low + high) // 2
        return self.lazy[node] + max(
            self.query(left, right, node * 2, low, middle),
            self.query(left, right, node * 2 + 1, middle + 1, high))


def _has_live_alias(rows: list[StaticAllocation]) -> bool:
    """Detect overlapping address/lifetime rectangles in O(n log n)."""
    if len(rows) < 2:
        return False
    import heapq

    phases = sorted({value for row in rows for value in (
        row.first_phase, row.last_phase)})
    phase_index = {value: index for index, value in enumerate(phases)}
    occupancy = _RangeMaximum(len(phases))
    active: list[tuple[int, int, int, int]] = []
    serial = 0
    for row in sorted(rows, key=lambda value: (value.offset, value.end, value.rid)):
        while active and active[0][0] <= row.offset:
            _end, _serial, prior_left, prior_right = heapq.heappop(active)
            occupancy.add(prior_left, prior_right, -1)
        left = phase_index[row.first_phase]
        right = phase_index[row.last_phase]
        if occupancy.query(left, right):
            return True
        occupancy.add(left, right, 1)
        serial += 1
        heapq.heappush(active, (row.end, serial, left, right))
    return False


def plan_static_memory(module: Module, resource_banks, hardware) -> StaticMemoryPlan:
    """Assign aligned per-bank offsets, reusing storage only after a resource dies."""
    intervals = live_intervals(module)
    if not intervals:
        raise ValueError("static memory planning requires a touched resource")
    if len(intervals) > _MAX_RESOURCES:
        raise ValueError("static memory planning exceeds the resource bound")
    bindings = _bindings(resource_banks)
    if set(bindings) != set(intervals):
        raise ValueError("resource bank bindings must cover every touched resource exactly")

    by_bank: dict[str, list[int]] = {}
    for rid, bank_name in bindings.items():
        try:
            bank = hardware.bank(bank_name)
        except KeyError as exc:
            raise ValueError(f"resource {rid} names unknown bank {bank_name!r}") from exc
        resource = module.resource(rid)
        if resource is None:
            raise ValueError(f"binding references unknown resource {rid}")
        if resource.domain.name != bank.domain:
            raise ValueError(f"resource {rid} domain does not match bank {bank_name!r}")
        by_bank.setdefault(bank_name, []).append(rid)

    allocations: list[StaticAllocation] = []
    summaries: list[BankMemoryPlan] = []
    for bank_name in sorted(by_bank):
        bank = hardware.bank(bank_name)
        placed: list[StaticAllocation] = []
        naive = 0
        for rid in sorted(by_bank[bank_name], key=lambda value: (intervals[value][0], value)):
            resource = module.resource(rid)
            size = resource.count * resource.elem_bytes
            if not 1 <= size <= _MAX_U63:
                raise ValueError(f"resource {rid} has an invalid static byte size")
            lo, hi = intervals[rid]
            alignment = max(resource.align, bank.alignment)
            if alignment & (alignment - 1):
                raise ValueError(f"resource {rid} or bank {bank_name!r} has non-power-of-two alignment")
            naive = _checked_add(_align(naive, alignment), size)
            active = sorted((row for row in placed
                             if not (row.last_phase < lo or hi < row.first_phase)),
                            key=lambda row: (row.offset, row.rid))
            offset = 0
            for row in active:
                offset = _align(offset, alignment)
                if _checked_add(offset, size) <= row.offset:
                    break
                offset = max(offset, row.end)
            offset = _align(offset, alignment)
            if _checked_add(offset, size) > bank.allocatable_bytes:
                raise ValueError(f"static address plan exceeds allocatable bytes in bank {bank_name!r}")
            placed.append(StaticAllocation(rid, bank_name, offset, size, alignment, lo, hi))
        extent = max(row.end for row in placed)
        summaries.append(BankMemoryPlan(bank_name, extent, naive, bank.allocatable_bytes))
        allocations.extend(placed)
    plan = StaticMemoryPlan(_module_digest(module), hardware.digest,
                            tuple(sorted(allocations, key=lambda row: row.rid)),
                            tuple(summaries))
    errors = verify_static_memory_plan(plan, module, resource_banks, hardware)
    if errors:
        raise ValueError("static memory plan failed verification: " + "; ".join(errors))
    return plan


def verify_static_memory_plan(plan: StaticMemoryPlan, module: Module,
                              resource_banks, hardware) -> tuple[str, ...]:
    """Independently check identity, capacity, alignment, lifetime, and alias safety."""
    errors: list[str] = []
    intervals = live_intervals(module)
    try:
        bindings = _bindings(resource_banks)
    except ValueError as exc:
        return (str(exc),)
    if plan.module_digest != _module_digest(module):
        errors.append("module digest mismatch")
    if plan.hardware_digest != hardware.digest:
        errors.append("hardware digest mismatch")
    rows = {row.rid: row for row in plan.allocations}
    if set(rows) != set(intervals) or set(bindings) != set(intervals):
        errors.append("allocation/binding census mismatch")
    for rid in sorted(set(rows) & set(intervals) & set(bindings)):
        row = rows[rid]; resource = module.resource(rid)
        if resource is None:
            errors.append(f"resource {rid} is absent from the module registry")
            continue
        try:
            bank = hardware.bank(bindings[rid])
        except KeyError:
            errors.append(f"resource {rid} names an unknown bank")
            continue
        if resource.domain.name != bank.domain:
            errors.append(f"resource {rid} domain does not match its bank")
        expected_size = resource.count * resource.elem_bytes
        expected_alignment = max(resource.align, bank.alignment)
        if row.bank != bindings[rid]: errors.append(f"resource {rid} bank mismatch")
        if row.size_bytes != expected_size: errors.append(f"resource {rid} size mismatch")
        if row.alignment != expected_alignment or row.offset % expected_alignment:
            errors.append(f"resource {rid} alignment mismatch")
        if (row.first_phase, row.last_phase) != intervals[rid]:
            errors.append(f"resource {rid} lifetime mismatch")
        if row.end > bank.allocatable_bytes:
            errors.append(f"resource {rid} exceeds bank capacity")
    for bank_name in sorted({row.bank for row in rows.values()}):
        if _has_live_alias([row for row in rows.values() if row.bank == bank_name]):
            errors.append(f"live resources in bank {bank_name!r} alias")
    summary = {row.bank: row for row in plan.banks}
    for bank_name in sorted({row.bank for row in rows.values()}):
        bank_rows = [row for row in rows.values() if row.bank == bank_name]
        if bank_name not in summary:
            errors.append(f"bank {bank_name!r} lacks a summary")
            continue
        try:
            bank = hardware.bank(bank_name)
        except KeyError:
            errors.append(f"bank summary names unknown bank {bank_name!r}")
            continue
        row = summary[bank_name]
        if row.extent_bytes != max(item.end for item in bank_rows):
            errors.append(f"bank {bank_name!r} extent mismatch")
        try:
            naive = 0
            for item in sorted(bank_rows, key=lambda value: (value.first_phase, value.rid)):
                naive = _checked_add(_align(naive, item.alignment), item.size_bytes)
            if row.naive_bytes != naive:
                errors.append(f"bank {bank_name!r} naive footprint mismatch")
        except ValueError:
            errors.append(f"bank {bank_name!r} naive footprint overflows")
        if row.capacity_bytes != bank.allocatable_bytes:
            errors.append(f"bank {bank_name!r} capacity mismatch")
    if set(summary) != {row.bank for row in rows.values()}:
        errors.append("bank summary census mismatch")
    return tuple(errors)


__all__ = ["BankMemoryPlan", "ResourceBankBinding", "StaticAllocation",
           "StaticMemoryPlan", "plan_static_memory", "verify_static_memory_plan"]
