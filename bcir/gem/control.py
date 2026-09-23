"""ControlRecordV1 -- the control plane as bytes (G14, staged plan S3-A).

Lease, generation, quiescence, activation, rollback and cancellation were prose, an
`admit(map_gen, data_gen)` argument and two methods that raise: a stale generation was refused
only where a caller remembered to pass the right number. This module holds the abstract values
the `ControlRecordV1` wire format carries (`bcir/abi/control_abi.py`,
`docs/kernel/BCIR_CONTROL_PLANE_ABI.md`, `runtime/c/bcir_control_plane.h`) and the **resident
plane** that decides them by their bytes:

* the six record kinds, their fixed bodies, their capability bits and their closed reason sets;
* `ControlPlane` -- one handle's resident control state (generation, live artifact and its
  rollback target, the installed registry, the boundary counter, the in-flight count, the drain,
  the lease table and the one pending switch) and the decisions: `submit` a record (applied,
  deferred or refused, in the specification's order), `enter` / `leave` / `advance` across
  phase and event boundaries (a deferred switch applies exactly once, at a boundary), and
  `admit_pack` / `admit_plan` (the data plane, by bytes, against the registry records installed);
* `is_stale` -- the one staleness predicate the plane, the trusted loader
  (`asn1.staged.TrustedLoader.install`) and context-shard activation
  (`kbcir.context_shard.certify_context_activation`) all express;
* `registry_digest` / `rollback_token` / `lease_key` -- the digests and the key derivation,
  byte-identical to the C twin's.

The plane is the authority's resident state: it holds the root key the way the C twin's
`bcir_ctl_state` does. An issuer holds only its lease key; the proposing program holds nothing.
Nothing here is a legality verdict -- a record carries no cost and no diagnostic.

The codec imports lazily (this module is imported through `bcir.gem` and `bcir.abi`, and must
stay off the cold organs; `tools/perf/import_graph.py`).
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass, field

#: The six kinds, in wire-code order (1..6).
CONTROL_KINDS = ("lease", "generation", "quiesce", "activate", "rollback", "cancel")
#: The kinds that move the resident generation. The others act without moving it.
SWITCH_KINDS = frozenset({"generation", "activate", "rollback"})
#: What a record's `subject` names (wire codes 0..4).
CONTROL_SCOPES = ("module", "resource", "mapping", "session", "channel")
#: The capability a record exercises: exactly its kind's bit.
CAPABILITY = {kind: 1 << index for index, kind in enumerate(CONTROL_KINDS)}
#: Every v1 capability, and the ones a lease may grant (no delegation: never `lease`).
CAP_ALL = 0x3F
CAP_GRANTABLE = CAP_ALL & ~CAPABILITY["lease"]
#: Why a record was issued -- a closed set per kind, wire code = index.
REASONS = {
    "lease": ("none",),
    "generation": ("none", "remap", "rewrite", "topology"),
    "quiesce": ("none", "activation", "rollback", "teardown"),
    "activate": ("none", "promotion", "repair"),
    "rollback": ("none", "correctness", "health", "policy"),
    "cancel": ("none", "withdrawn", "superseded"),
}
#: How an operation ended (wire code = index; `none` = succeeded with no record decided).
VERDICTS = ("applied", "deferred", "refused", "none")
#: Why the plane refused (wire code = index) -- the C twin's `bcir_ctl_refusal`.
REFUSALS = (
    "none",
    "malformed",
    "lease",
    "mac",
    "subject",
    "expired",
    "capability",
    "replay",
    "stale",
    "ahead",
    "early",
    "pending",
    "mismatch",
    "nothing",
    "full",
    "duplicate",
    "draining",
    "busy",
    "idle",
    "exhausted",
)
#: The lease table is bounded, and the C twin's is a fixed array of this size.
LEASE_CAPACITY = 8
#: The root key a plane holds: one HMAC block at most, 128 bits at least.
KEY_MIN, KEY_MAX = 16, 64

ZERO32 = bytes(32)
_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1
_LEASE_TAG = b"BCTL/lease/v1\x00"
_REGISTRY_TAG = b"BCTL/registry/v1\x00"
_TOKEN_TAG = b"BCTL/token/v1\x00"
_STATE_TAG = b"BCTL/state/v1\x00"


# --- the bodies (fixed-width; the wire layout is in bcir/abi/control_abi.py) -----------------


@dataclass(frozen=True)
class LeaseGrant:
    """kind 1: the root grants lease `lease_id` to principal `holder`, with the capability mask
    `granted`, valid for the boundaries [issued_epoch, expiry_epoch)."""

    lease_id: int
    granted: int
    issued_epoch: int
    expiry_epoch: int
    holder: int


@dataclass(frozen=True)
class GenerationSwitch:
    """kind 2: install a registry state -- the per-resource vector's maxima, the topology
    generation and `registry_digest` (the digest of the vector itself)."""

    map_gen: int
    data_gen: int
    topo_gen: int
    registry_digest: bytes


@dataclass(frozen=True)
class Quiesce:
    """kind 3: drain -- new work is refused until a switch lands or `drain_deadline` passes."""

    drain_deadline: int


@dataclass(frozen=True)
class Activate:
    """kind 4: install `artifact_sha256` in place of `previous_sha256` (all-zero = none)."""

    artifact_sha256: bytes
    previous_sha256: bytes = ZERO32


@dataclass(frozen=True)
class Rollback:
    """kind 5: restore `restore_sha256`, presenting the token of the activation it undoes."""

    restore_sha256: bytes
    rollback_token: bytes


@dataclass(frozen=True)
class Cancel:
    """kind 6: withdraw the issuer's pending switch whose sequence lies in [first, last]."""

    first_sequence: int
    last_sequence: int


BODY_TYPES = {
    "lease": LeaseGrant,
    "generation": GenerationSwitch,
    "quiesce": Quiesce,
    "activate": Activate,
    "rollback": Rollback,
    "cancel": Cancel,
}


@dataclass(frozen=True)
class ControlRecord:
    """One control record (the abstract value; `abi.control_abi` is its bytes).

    `generation` is the generation the record asserts -- the new one for a switch -- and
    `expect` the resident generation the issuer witnessed: the compare-and-swap witness that
    makes a stale record refusable by its own bytes. `capability` and the body length are
    determined by `kind` and are therefore not fields here; `mac` is the 32-byte HMAC once the
    record is signed (`abi.control_abi.sign_control`)."""

    kind: str
    scope: str
    subject: int
    generation: int
    expect: int
    boundary: int
    sequence: int
    lease: int
    body: object
    reason: str = "none"
    mac: bytes = b""

    @property
    def capability(self) -> int:
        return CAPABILITY[self.kind]

    @property
    def is_switch(self) -> bool:
        return self.kind in SWITCH_KINDS


@dataclass(frozen=True)
class ControlOutcome:
    """How one plane operation ended. `generation` is the resident generation afterwards;
    `status` names the codec's (or the admitted artifact's) refusal when there is one."""

    verdict: str
    refusal: str = "none"
    kind: str = ""
    sequence: int = 0
    generation: int = 0
    status: str = "BCIR_OK"

    @property
    def applied(self) -> bool:
        return self.verdict == "applied"


# --- the shared predicates and digests ---------------------------------------------------------


def is_stale(witnessed: int, resident: int) -> bool:
    """A record, artifact or activation minted against the resident generation `witnessed` is
    stale once the resident has moved past it. The one predicate the plane (a record's
    `expect`), the trusted loader (an artifact admitted against `generation - 1`) and
    context-shard activation (an activation against `activation_generation - 1`) express."""
    return witnessed < resident


def lease_key(root_key: bytes, lease_id: int) -> bytes:
    """The capability-scoped key a lease holder receives: HMAC-SHA256(root, tag || u64le(id))."""
    return hmac.new(
        bytes(root_key), _LEASE_TAG + struct.pack("<Q", lease_id), hashlib.sha256
    ).digest()


def registry_digest(generations) -> bytes:
    """SHA-256 over the per-resource generation vector's wire bytes (rid, map_gen, data_gen per
    resource, RIDs strictly ascending) -- the StreamPack v4 / ExecutionPlanV1 tail, so the C twin
    hashes an artifact's own bytes in place."""
    h = hashlib.sha256(_REGISTRY_TAG)
    previous = -1
    for g in generations:
        if g.rid <= previous:
            raise ValueError(
                f"a generation vector's RIDs must be strictly ascending ({g.rid} after {previous})"
            )
        previous = g.rid
        h.update(struct.pack("<III", g.rid, g.map_gen, g.data_gen))
    return h.digest()


def rollback_token(signed: bytes) -> bytes:
    """The token an activation mints: SHA-256(tag || header || body) of the activate record. A
    public binding -- which activation a rollback undoes -- not a secret."""
    return hashlib.sha256(_TOKEN_TAG + bytes(signed)).digest()


# --- the resident plane ------------------------------------------------------------------------


@dataclass
class _Lease:
    lease_id: int
    granted: int
    issued: int
    expiry: int
    holder: int
    last_sequence: int = 0
    # lease_key(root, lease_id), derived once when the grant is applied (G15/S3-B): a leased
    # record's MAC is checked against it instead of re-deriving it per record. Not state -- the
    # state digest does not cover it -- and it leaves with the lease.
    key: bytes = field(default=b"", compare=False, repr=False)

    def valid_at(self, boundary: int) -> bool:
        return self.issued <= boundary < self.expiry


class ControlPlane:
    """One handle's resident control state, deciding records by their bytes.

    The mirror of `bcir_ctl_state` / `bcir_ctl_submit` / `bcir_ctl_enter` / `bcir_ctl_leave` /
    `bcir_ctl_advance` / `bcir_ctl_admit_pack` / `bcir_ctl_admit_plan`, decision for decision:
    the two rails' traces (every outcome and the state digest after every operation) are held
    identical by the G14 parity gate. A refused record changes nothing."""

    def __init__(self, root_key: bytes, scope: str = "module", subject: int = 0) -> None:
        from ..abi.control_abi import ControlError

        if not isinstance(root_key, (bytes, bytearray)) or not KEY_MIN <= len(root_key) <= KEY_MAX:
            raise ControlError(
                "BCIR_ERR_CONTROL", f"a plane's root key must be {KEY_MIN}..{KEY_MAX} bytes"
            )
        if scope not in CONTROL_SCOPES:
            raise ControlError("BCIR_ERR_CONTROL", f"unknown scope {scope!r}")
        if (
            not isinstance(subject, int)
            or isinstance(subject, bool)
            or not 0 <= subject <= _U64_MAX
        ):
            raise ControlError("BCIR_ERR_CONTROL", f"subject must be a u64, got {subject!r}")
        self._key = bytes(root_key)
        self.scope = scope
        self.subject = subject
        self.generation = 0
        self.artifact = ZERO32
        self.previous = ZERO32
        self.token = ZERO32
        self.registry: tuple[int, int, int, bytes] | None = None
        self.boundary = 0
        self.in_flight = 0
        self.draining = False
        self.drain_deadline = 0
        self.root_sequence = 0
        self.last_lease_id = 0
        self.leases: list[_Lease] = []
        self.pending: bytes | None = None
        self._pending_record: ControlRecord | None = None

    # --- reading the state ---------------------------------------------------------------

    def lease(self, lease_id: int) -> _Lease | None:
        for entry in self.leases:
            if entry.lease_id == lease_id:
                return entry
        return None

    @property
    def pending_record(self) -> ControlRecord | None:
        return self._pending_record

    def state_bytes(self) -> bytes:
        """The canonical serialization of the resident state (the key excluded)."""
        reg = self.registry
        parts = [
            _STATE_TAG,
            struct.pack(
                "<IQQBQ",
                self.generation,
                self.boundary,
                self.in_flight,
                int(self.draining),
                self.drain_deadline,
            ),
            self.artifact,
            self.previous,
            self.token,
            struct.pack("<B", 0 if reg is None else 1),
            struct.pack("<III", *(reg[:3] if reg is not None else (0, 0, 0))),
            reg[3] if reg is not None else ZERO32,
            struct.pack(
                "<QQBQ",
                self.root_sequence,
                self.last_lease_id,
                CONTROL_SCOPES.index(self.scope),
                self.subject,
            ),
            struct.pack("<I", len(self.leases)),
        ]
        for e in self.leases:
            parts.append(
                struct.pack(
                    "<QQQQQQ", e.lease_id, e.granted, e.issued, e.expiry, e.holder, e.last_sequence
                )
            )
        pending = self.pending or b""
        parts.append(struct.pack("<I", len(pending)))
        parts.append(pending)
        return b"".join(parts)

    def state_digest(self) -> bytes:
        return hashlib.sha256(self.state_bytes()).digest()

    # --- outcomes -------------------------------------------------------------------------

    def _outcome(
        self,
        verdict: str,
        refusal: str = "none",
        record: ControlRecord | None = None,
        status: str = "BCIR_OK",
    ) -> ControlOutcome:
        return ControlOutcome(
            verdict,
            refusal,
            record.kind if record is not None else "",
            record.sequence if record is not None else 0,
            self.generation,
            status,
        )

    # --- deciding a record ----------------------------------------------------------------

    def submit(self, data: bytes) -> ControlOutcome:
        """Decide one record: applied, deferred or refused, the checks in the specification's
        order (docs/kernel/BCIR_CONTROL_PLANE_ABI.md, "Deciding a record")."""
        from ..abi.control_abi import ControlError, check_control_mac, decode_control

        try:
            rec = decode_control(data)
        except ControlError as exc:
            return self._outcome("refused", "malformed", status=exc.status)
        entry = None
        if rec.lease:
            entry = self.lease(rec.lease)
            if entry is None:
                return self._outcome("refused", "lease", rec)
            key = entry.key
        else:
            key = self._key
        if not check_control_mac(data, key):
            return self._outcome("refused", "mac", rec, "BCIR_ERR_MAC")
        if rec.scope != self.scope or rec.subject != self.subject:
            return self._outcome("refused", "subject", rec)
        if entry is not None:
            if not entry.valid_at(self.boundary):
                return self._outcome("refused", "expired", rec)
            if not entry.granted & rec.capability:
                return self._outcome("refused", "capability", rec)
        last = entry.last_sequence if entry is not None else self.root_sequence
        if rec.sequence <= last:
            return self._outcome("refused", "replay", rec)
        if is_stale(rec.expect, self.generation):
            return self._outcome("refused", "stale", rec)
        if rec.expect > self.generation:
            return self._outcome("refused", "ahead", rec)
        refusal = self._kind_law(rec)
        if refusal:
            return self._outcome("refused", refusal, rec)
        if not rec.is_switch:
            if self.boundary < rec.boundary:
                return self._outcome("refused", "early", rec)
        elif self.in_flight or self.boundary < rec.boundary:
            self._consume(rec, entry)
            self.pending = bytes(data)
            self._pending_record = rec
            return self._outcome("deferred", "none", rec)
        self._consume(rec, entry)
        self._apply(rec, bytes(data))
        return self._outcome("applied", "none", rec)

    def _consume(self, rec: ControlRecord, entry: _Lease | None) -> None:
        if entry is not None:
            entry.last_sequence = rec.sequence
        else:
            self.root_sequence = rec.sequence

    def _live_leases(self) -> list[_Lease]:
        """The table once lapsed leases are dropped (lease ids never recur, so a dropped lease's
        sequence space cannot come back)."""
        return [e for e in self.leases if e.expiry > self.boundary]

    def _kind_law(self, rec: ControlRecord) -> str:
        body = rec.body
        if rec.kind == "lease":
            if body.lease_id <= self.last_lease_id:
                return "duplicate"
            if body.expiry_epoch <= self.boundary:
                return "expired"
            if len(self._live_leases()) >= LEASE_CAPACITY:
                return "full"
            return ""
        if rec.kind == "quiesce":
            return "expired" if body.drain_deadline < self.boundary else ""
        if rec.kind == "cancel":
            p = self._pending_record
            if (
                p is None
                or p.lease != rec.lease
                or not body.first_sequence <= p.sequence <= body.last_sequence
            ):
                return "nothing"
            return ""
        if self._pending_record is not None:
            return "pending"
        return self._switch_law(rec)

    def _switch_law(self, rec: ControlRecord) -> str:
        """The state laws a switch must meet -- at submission, and again (totally) when a
        deferred switch comes due."""
        body = rec.body
        if rec.kind == "activate":
            return "mismatch" if body.previous_sha256 != self.artifact else ""
        if rec.kind == "rollback":
            if self.previous == ZERO32:
                return "nothing"
            if body.restore_sha256 != self.previous or body.rollback_token != self.token:
                return "mismatch"
            return ""
        installed = (body.map_gen, body.data_gen, body.topo_gen, body.registry_digest)
        return "mismatch" if self.registry == installed else ""

    def _apply(self, rec: ControlRecord, data: bytes) -> None:
        body = rec.body
        if rec.kind == "lease":
            self.leases = self._live_leases()
            self.leases.append(
                _Lease(
                    body.lease_id,
                    body.granted,
                    body.issued_epoch,
                    body.expiry_epoch,
                    body.holder,
                    key=lease_key(self._key, body.lease_id),
                )
            )
            self.last_lease_id = body.lease_id
            return
        if rec.kind == "quiesce":
            self.draining = True
            self.drain_deadline = body.drain_deadline
            return
        if rec.kind == "cancel":
            self.pending = None
            self._pending_record = None
            return
        if rec.kind == "generation":
            self.registry = (body.map_gen, body.data_gen, body.topo_gen, body.registry_digest)
        elif rec.kind == "activate":
            from ..abi.control_abi import signed_part

            self.previous = self.artifact
            self.artifact = body.artifact_sha256
            self.token = rollback_token(signed_part(data))
        else:  # rollback
            self.artifact = self.previous
            self.previous = ZERO32
            self.token = ZERO32
        self.generation = rec.generation
        self.draining = False  # the switch a drain prepared has landed

    # --- boundaries -----------------------------------------------------------------------

    def enter(self) -> ControlOutcome:
        """A phase begins. Refused while draining; counted, never assumed."""
        if self.draining:
            return self._outcome("refused", "draining")
        if self.in_flight == _U64_MAX:
            return self._outcome("refused", "exhausted")
        self.in_flight += 1
        return self._outcome("none")

    def leave(self) -> ControlOutcome:
        """A phase ends; the leave that returns the plane to quiescence crosses a boundary."""
        if self.in_flight == 0:
            return self._outcome("refused", "idle")
        if self.in_flight == 1 and self.boundary == _U64_MAX:
            return self._outcome("refused", "exhausted")
        self.in_flight -= 1
        if self.in_flight:
            return self._outcome("none")
        return self._cross()

    def advance(self) -> ControlOutcome:
        """An event boundary while nothing is in flight."""
        if self.in_flight:
            return self._outcome("refused", "busy")
        if self.boundary == _U64_MAX:
            return self._outcome("refused", "exhausted")
        return self._cross()

    def _cross(self) -> ControlOutcome:
        self.boundary += 1
        if self.draining and self.boundary > self.drain_deadline:
            self.draining = False
        rec = self._pending_record
        if rec is None or self.boundary < rec.boundary:
            return self._outcome("none")
        data = self.pending
        self.pending = None
        self._pending_record = None
        entry = self.lease(rec.lease)
        if entry is None or not entry.valid_at(self.boundary):
            return self._outcome("refused", "expired", rec)
        if rec.expect != self.generation:  # held true by the one-slot rule; checked totally
            return self._outcome("refused", "stale", rec)
        refusal = self._switch_law(rec)
        if refusal:
            return self._outcome("refused", refusal, rec)
        self._apply(rec, data)
        return self._outcome("applied", "none", rec)

    # --- the data plane, by bytes -----------------------------------------------------------

    def admit_pack(self, data: bytes) -> ControlOutcome:
        """Admit a StreamPack only if it carries exactly the installed registry: the header
        maxima and topo_gen, and a generation vector that digests to `registry_digest`. A plane
        with no registry admits nothing (there is nothing to prove the pack current against)."""
        from ..abi.streampack_abi import AbiError, decode

        if self.registry is None:
            return self._outcome("refused", "stale", status="BCIR_ERR_STALE")
        try:
            pack = decode(data)
        except AbiError as exc:
            return self._outcome("refused", "malformed", status=f"malformed: {exc}")
        if not pack.provenance_ok():
            return self._outcome("refused", "malformed", status="BCIR_ERR_PROVENANCE")
        map_gen, data_gen, topo_gen, digest = self.registry
        if (pack.map_gen, pack.data_gen, pack.topo_gen) != (map_gen, data_gen, topo_gen):
            return self._outcome("refused", "stale", status="BCIR_ERR_STALE")
        if registry_digest(pack.generations) != digest:
            return self._outcome("refused", "stale", status="BCIR_ERR_STALE")
        return self._outcome("applied")

    def admit_plan(self, data: bytes) -> ControlOutcome:
        """Admit an ExecutionPlanV1 only if its generation vector digests to the installed
        registry's."""
        from ..abi.execution_plan_abi import decode_plan
        from ..abi.streampack_abi import AbiError

        if self.registry is None:
            return self._outcome("refused", "stale", status="BCIR_ERR_STALE")
        try:
            plan = decode_plan(data)
        except AbiError as exc:
            return self._outcome("refused", "malformed", status=f"malformed: {exc}")
        if registry_digest(plan.generations) != self.registry[3]:
            return self._outcome("refused", "stale", status="BCIR_ERR_STALE")
        return self._outcome("applied")


__all__ = [
    "BODY_TYPES",
    "CAPABILITY",
    "CAP_ALL",
    "CAP_GRANTABLE",
    "CONTROL_KINDS",
    "CONTROL_SCOPES",
    "KEY_MAX",
    "KEY_MIN",
    "LEASE_CAPACITY",
    "REASONS",
    "REFUSALS",
    "SWITCH_KINDS",
    "VERDICTS",
    "Activate",
    "Cancel",
    "ControlOutcome",
    "ControlPlane",
    "ControlRecord",
    "GenerationSwitch",
    "LeaseGrant",
    "Quiesce",
    "Rollback",
    "is_stale",
    "lease_key",
    "registry_digest",
    "rollback_token",
]
