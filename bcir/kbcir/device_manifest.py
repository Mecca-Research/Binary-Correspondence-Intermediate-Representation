"""The DEVICE MANIFEST: the driver seam's static hardware schema (Phase D hardening).

The hardened-driver principles this module encodes (the driver/kernel roadmap's Part V
analysis; the numbers are its D-R rules):

  * **D-R1, static schema**: a driver never discovers hardware in a way that CHANGES
    execution logic. The `DeviceManifest` is the immutable, digestable type object the
    compiler plans against -- banks, native tiles, and the interconnect distance matrix
    are data, fixed before planning (the `TargetProfile`/`UartPlacement` discipline,
    bundled and attested). `probe_agree` is the runtime's ONLY discovery verb: observed
    facts may VETO a manifest (refuse to run -- the UART blueprint's `caps_mismatch`),
    they may never STEER (reroute, resize, or substitute).
  * **D-R2, bank typing**: SRAM/HBM/DRAM are not "memory"; they are distinct `Domain`s.
    A claim whose operands span two MEMORY tiers is illegal unless it IS an explicit
    move (`mem.move.near` / `mem.move.far`) with exactly one source and one destination
    bank -- `check_bank_moves`. MMIO is EXEMPT by measurement: register I/O mixes
    MMIO/RAM in ordinary load/store claims by design (8 such claims in the GPIO fixture
    alone), and its ordering law is the R3 rail, not a DMA move.
  * **D-R3, distance-aware moves**: the manifest's Q8 distance matrix prices every
    bank pair; `move_cost` is the planner's data-movement penalty and `move_claim`
    mints the near/far op-codes (near == same memory tier, far == a tier crossing).
  * **D-R4, strided views only**: the driver seam allocates nothing from a bare size.
    A `StridedView` carries the FULL dimensional stride vector (one stride per
    dimension, in elements) and must fit its bank; `check_strided_view` also enforces
    the hardware's NATIVE TILE -- a 15x15 view against a 16-native bank refuses at
    plan time (runtime fragmentation is a compile error).

Two-truth: this module imports no verifier -- every check returns caller-passed
messages (the `check_attention` posture); the tests verify. The persisted envelope
rides the 0.4b decision-record pattern, generation-tied to the calibration table
(a stale manifest's plan is an unearned claim -- the tile-prior staleness law)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..model import Claim, Domain, Lane, Opcode, StrideClass

_Q8 = 256

# The MEMORY tiers D-R2 governs. MMIO is exempt (measured: drivers mix MMIO/RAM in
# plain load/store claims by design; the R3 rail owns register-access ordering).
_MEM_TIERS = frozenset((Domain.RAM, Domain.VRAM, Domain.NVM, Domain.CXL, Domain.HBM))


@dataclass(frozen=True)
class MemoryBank:
    """One physical bank: a NAME the plan can attest, its memory tier (`Domain`), its
    capacity, its native tile (1 == untiled scalar bank), and its element alignment."""

    name: str
    domain: Domain
    capacity_bytes: int
    native_tile: int = 1
    align: int = 64


@dataclass(frozen=True)
class DeviceManifest:
    """The immutable hardware schema: banks + the Q8 interconnect distance matrix
    (distance[i][j] scales a byte moved from bank i to bank j; the diagonal is 0 --
    staying put is free; `_Q8` == a plain unit-cost hop). `target`/`cal_gen` tie the
    manifest to the calibrated cost model it was authored against."""

    device: str
    banks: tuple                    # tuple[MemoryBank, ...]
    distance: tuple                 # tuple[tuple[int, ...], ...] -- Q8, row-major
    target: str = ""
    cal_gen: int = 0

    @property
    def digest(self) -> str:
        """The manifest's sha256 over its canonical JSON -- the type-system identity a
        plan/binary attests (the R12/R13 posture)."""
        doc = {"device": self.device, "target": self.target, "cal_gen": self.cal_gen,
               "banks": [[b.name, int(b.domain), b.capacity_bytes, b.native_tile, b.align]
                         for b in self.banks],
               "distance": [list(row) for row in self.distance]}
        return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()

    def bank(self, name: str) -> MemoryBank:
        for b in self.banks:
            if b.name == name:
                return b
        raise KeyError(f"no bank {name!r} in device {self.device!r}")

    def bank_index(self, name: str) -> int:
        for i, b in enumerate(self.banks):
            if b.name == name:
                return i
        raise KeyError(f"no bank {name!r} in device {self.device!r}")


def check_device_manifest(man: DeviceManifest) -> list[str]:
    """Internal consistency (caller-passed messages, [] == well-formed): unique bank
    names, positive capacities/tiles/alignments, and a square, zero-diagonal,
    positive-off-diagonal, SYMMETRIC distance matrix (the interconnect is a metric --
    an asymmetric entry is a typo, not a topology)."""
    msgs: list[str] = []
    if not man.banks:
        msgs.append("manifest: no banks")
    names = [b.name for b in man.banks]
    if len(set(names)) != len(names):
        msgs.append(f"manifest: duplicate bank names {names}")
    for b in man.banks:
        if b.capacity_bytes < 1:
            msgs.append(f"bank {b.name}: capacity must be >= 1; got {b.capacity_bytes}")
        if b.native_tile < 1:
            msgs.append(f"bank {b.name}: native_tile must be >= 1; got {b.native_tile}")
        if b.align < 1:
            msgs.append(f"bank {b.name}: align must be >= 1; got {b.align}")
    n = len(man.banks)
    if len(man.distance) != n or any(len(row) != n for row in man.distance):
        msgs.append(f"manifest: distance matrix is not {n}x{n}")
        return msgs
    for i in range(n):
        if man.distance[i][i] != 0:
            msgs.append(f"distance[{i}][{i}] must be 0 (staying put is free); "
                        f"got {man.distance[i][i]}")
        for j in range(n):
            if i != j and man.distance[i][j] < 1:
                msgs.append(f"distance[{i}][{j}] must be >= 1 Q8; got {man.distance[i][j]}")
            if man.distance[i][j] != man.distance[j][i]:
                msgs.append(f"distance[{i}][{j}] {man.distance[i][j]} != "
                            f"distance[{j}][{i}] {man.distance[j][i]} (asymmetric)")
    return msgs


def move_cost(man: DeviceManifest, src: str, dst: str, nbytes: int) -> int:
    """The D-R3 pricing primitive: bytes x the Q8 pairwise distance. The planner's
    data-movement penalty -- a far move IS more expensive because the manifest says
    so, not because a heuristic guessed."""
    d = man.distance[man.bank_index(src)][man.bank_index(dst)]
    return (max(0, nbytes) * d) >> 8


def move_claim(man: DeviceManifest, src: str, dst: str, rid_src: int, rid_dst: int,
               count: int, cid: int) -> Claim:
    """An EXPLICIT bank move (D-R2's cast): `mem.move.near` when source and destination
    share a memory tier, `mem.move.far` on a tier crossing -- the distance-aware
    op-codes the cost model prices differently. The claim's domain is the DESTINATION
    (where the bytes land)."""
    sb, db = man.bank(src), man.bank(dst)
    kind = "near" if sb.domain == db.domain else "far"
    return Claim(id=cid, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
                 count=max(1, count), rd=(rid_src,), wr=(rid_dst,),
                 op=f"mem.move.{kind}:{src}->{dst}", domain=db.domain,
                 bounds="assumed_safe")


def check_bank_moves(module, *, exempt: frozenset = frozenset((Domain.MMIO,))) -> list[str]:
    """The D-R2 law over a module (caller-passed messages; vacuous when every claim's
    operands share one memory tier -- measured true of the ENTIRE existing corpus): a
    claim whose operand resources span two MEMORY tiers must be an explicit
    `mem.move.*` op with exactly one read and one write; anything else is an implicit
    cross-bank access the driver seam forbids. Domains in `exempt` (MMIO by default)
    never count -- register I/O is the R3 rail's law, not a DMA move."""
    msgs: list[str] = []
    res = module.resources if isinstance(module.resources, dict) else {
        r.rid: r for r in module.resources}
    for ph in module.phases:
        for c in ph.claims:
            doms = {res[r].domain for r in (*c.rd, *c.wr)
                    if r in res and res[r].domain not in exempt}
            tiers = doms & _MEM_TIERS
            if len(tiers) <= 1:
                continue
            if not c.op.startswith("mem.move."):
                msgs.append(f"claim {c.id} ({c.op}): operands span memory tiers "
                            f"{sorted(d.name for d in tiers)} without an explicit "
                            f"mem.move (D-R2: banks are types; crossing needs a cast)")
            elif len(c.rd) != 1 or len(c.wr) != 1:
                msgs.append(f"claim {c.id} ({c.op}): a mem.move must have exactly one "
                            f"source and one destination; got rd={c.rd} wr={c.wr}")
    return msgs


@dataclass(frozen=True)
class StridedView:
    """The ONLY allocation currency at the driver seam (D-R4): a bank, a byte offset,
    the shape, and the FULL per-dimension stride vector in ELEMENTS (innermost last).
    No bare-size malloc exists here by construction -- a request without its stride
    vector cannot even be spelled."""

    bank: str
    offset_bytes: int
    shape: tuple                    # (d0, d1, ...) extents, outermost first
    strides: tuple                  # elements, one per dimension
    elem_bytes: int = 4


def check_strided_view(view: StridedView, man: DeviceManifest) -> list[str]:
    """D-R4 (caller-passed messages, [] == admissible): the stride vector must cover
    EVERY dimension; extents/strides positive; the view's last touched byte must sit
    inside the bank; the offset honors the bank alignment; and the two innermost
    extents honor the bank's NATIVE TILE (a 15x15 submission against a 16-native bank
    refuses at plan time -- the fragmentation law)."""
    msgs: list[str] = []
    try:
        bank = man.bank(view.bank)
    except KeyError as e:
        return [str(e)]
    if not view.shape:
        msgs.append(f"view on {view.bank}: empty shape")
        return msgs
    if len(view.strides) != len(view.shape):
        msgs.append(f"view on {view.bank}: {len(view.shape)} dimensions but "
                    f"{len(view.strides)} strides (the FULL stride vector is required)")
        return msgs
    if any(d < 1 for d in view.shape) or any(s < 1 for s in view.strides):
        msgs.append(f"view on {view.bank}: extents and strides must be >= 1 "
                    f"(shape={view.shape}, strides={view.strides})")
        return msgs
    if view.offset_bytes < 0 or view.offset_bytes % bank.align:
        msgs.append(f"view on {view.bank}: offset {view.offset_bytes} violates the "
                    f"bank alignment {bank.align}")
    last = view.offset_bytes + sum((d - 1) * s for d, s in
                                   zip(view.shape, view.strides)) * view.elem_bytes \
        + view.elem_bytes
    if last > bank.capacity_bytes:
        msgs.append(f"view on {view.bank}: last byte {last} exceeds the bank capacity "
                    f"{bank.capacity_bytes}")
    if bank.native_tile > 1:
        for d in view.shape[-2:]:
            if d % bank.native_tile:
                msgs.append(f"view on {view.bank}: extent {d} is not a multiple of the "
                            f"native tile {bank.native_tile} (runtime fragmentation "
                            f"is a plan-time refusal)")
    return msgs


def probe_agree(man: DeviceManifest, observed: dict) -> list[str]:
    """D-R1's veto-not-steer law: `observed` maps bank name -> {capacity_bytes,
    native_tile} facts a runtime probe measured. Every observation must EQUAL the
    manifest; a disagreement is a message (the runtime REFUSES -- it never reroutes,
    resizes, or substitutes; the UART blueprint's `caps_mismatch`, promoted)."""
    msgs: list[str] = []
    names = {b.name for b in man.banks}
    for name, facts in sorted(observed.items()):
        if name not in names:
            msgs.append(f"probe observed unknown bank {name!r} (not in the manifest: "
                        f"discovery may veto, never steer)")
            continue
        bank = man.bank(name)
        for key in ("capacity_bytes", "native_tile"):
            if key in facts and int(facts[key]) != getattr(bank, key):
                msgs.append(f"bank {name}: observed {key} {facts[key]} != manifest "
                            f"{getattr(bank, key)} -- REFUSE (veto), do not adapt")
    return msgs


# --- the persisted envelope (the 0.4b decision-record pattern) --------------------------------

MANIFEST_KIND = "bcir.device_manifest"
MANIFEST_SCHEMA = 1


def save_device_manifest(path: str, man: DeviceManifest) -> None:
    doc = {"kind": MANIFEST_KIND, "schema": MANIFEST_SCHEMA, "device": man.device,
           "target": man.target, "cal_gen": int(man.cal_gen),
           "banks": [[b.name, int(b.domain), b.capacity_bytes, b.native_tile, b.align]
                     for b in man.banks],
           "distance": [list(row) for row in man.distance], "digest": man.digest}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)


def load_device_manifest(path: str, *, expect_target: str = "",
                         expect_cal_gen: int | None = None) -> DeviceManifest:
    """Load with the house refusals: wrong kind, a NEWER schema, a different target
    ('retrain'), a cal_gen mismatch ('STALE'), or a digest that does not re-derive
    (a tampered manifest is caught, not trusted)."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("kind") != MANIFEST_KIND:
        raise ValueError(f"not a {MANIFEST_KIND} document (kind={doc.get('kind')!r})")
    if int(doc.get("schema", 0)) > MANIFEST_SCHEMA:
        raise ValueError(f"device-manifest schema v{doc['schema']} is newer than this "
                         f"build's v{MANIFEST_SCHEMA}; upgrade BCIR to load this manifest")
    if expect_target and doc.get("target") != expect_target:
        raise ValueError(f"device manifest was authored for target {doc.get('target')!r}, "
                         f"not {expect_target!r} -- retrain")
    if expect_cal_gen is not None and int(doc.get("cal_gen", 0)) != expect_cal_gen:
        raise ValueError(f"device manifest is STALE: authored under cal_gen "
                         f"{doc.get('cal_gen')}, the live table is cal_gen "
                         f"{expect_cal_gen} -- retrain")
    man = DeviceManifest(
        device=doc["device"],
        banks=tuple(MemoryBank(name=b[0], domain=Domain(int(b[1])), capacity_bytes=int(b[2]),
                               native_tile=int(b[3]), align=int(b[4]))
                    for b in doc["banks"]),
        distance=tuple(tuple(int(v) for v in row) for row in doc["distance"]),
        target=doc.get("target", ""), cal_gen=int(doc.get("cal_gen", 0)))
    if doc.get("digest") and doc["digest"] != man.digest:
        raise ValueError("device manifest digest does not re-derive (tampered)")
    return man
