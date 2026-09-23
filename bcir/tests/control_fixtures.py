"""The G14 corpora (S3-A), shared by `test_control_plane.py` and the baseline harness
(`tools/perf/gemplus_baseline.py --group control`).

Four families, each declared here with its expected outcome so a rail is graded against the
specification (docs/kernel/BCIR_CONTROL_PLANE_ABI.md), not against the other rail:

* `record_corpus()` -- valid records of every kind at the edges of every field, for the round
  trip Python encode -> C decode (a field-by-field dump) -> Python re-encode, byte for byte;
* `malformed_variants()` -- one variant per wire law, each built from a valid record by one
  mutation with the CRC recomputed (so it reaches the law it names), each declaring the exact
  `bcir_status` both rails must return;
* scenarios -- sequences of plane operations with the verdict and refusal every operation must
  produce: `transition_scenarios` (each kind's transition), `authority_scenarios` (forgeries),
  `stale_scenarios` (a stale generation at every boundary the tree has), `midphase_scenarios`
  (a switch requested mid-phase is deferred, then applied exactly once at the boundary) and
  `witness_scenarios` (every refusal code, so every plane law has a witness on both rails);
* the Python-only boundaries (`python_stale_boundaries`): the trusted loader, context-shard
  activation and the verifier, which have no C twin.

The C rail runs the same scenarios through `runtime/c/test_control_plane.c` (a script of
operations in, one trace line per operation out); the traces of the two rails -- verdicts,
refusals, statuses and the resident state digest after every operation -- must be identical.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import subprocess
import zlib
from dataclasses import dataclass, field, replace

from ..gem.control import (
    CAP_GRANTABLE,
    CAPABILITY,
    CONTROL_KINDS,
    CONTROL_SCOPES,
    REFUSALS,
    SWITCH_KINDS,
    VERDICTS,
    ZERO32,
    Activate,
    Cancel,
    ControlOutcome,
    ControlPlane,
    ControlRecord,
    GenerationSwitch,
    LeaseGrant,
    Quiesce,
    Rollback,
    lease_key,
    registry_digest,
)

ROOT_KEY = bytes(range(0x10, 0x30))  # 32 bytes: the plane's root key
FOREIGN_KEY = bytes(range(0x80, 0xA0))  # another authority's
SCOPE, SUBJECT = "module", 7  # the plane every scenario addresses
HOLDER = 42
_U32 = (1 << 32) - 1
_U64 = (1 << 64) - 1

OP_SUBMIT, OP_ENTER, OP_LEAVE, OP_ADVANCE, OP_ADMIT_PACK, OP_ADMIT_PLAN = 1, 2, 3, 4, 5, 6
# Harness-only: set the boundary counter / the in-flight count to a u64 (the data), so the
# `exhausted` refusals -- 2^64 boundaries away from any honest scenario -- have a witness on
# both rails. Not plane operations: no API on either rail exposes them.
OP_POKE_BOUNDARY, OP_POKE_IN_FLIGHT = 7, 8


def _codec():
    from ..abi import control_abi

    return control_abi


def digest_of(label: str) -> bytes:
    return hashlib.sha256(label.encode()).digest()


def record(kind: str, body, *, seq: int, expect: int, lease: int, **extra) -> ControlRecord:
    """An unsigned record addressed to the scenario plane (switches assert expect + 1)."""
    generation = expect + 1 if kind in SWITCH_KINDS else expect
    fields = dict(
        kind=kind,
        scope=SCOPE,
        subject=SUBJECT,
        generation=generation,
        expect=expect,
        boundary=0,
        sequence=seq,
        lease=lease,
        body=body,
    )
    fields.update(extra)
    return ControlRecord(**fields)


def issue(rec: ControlRecord, key: bytes | None = None) -> bytes:
    """Sign with the key an honest issuer holds (the root key for a grant, the lease's key
    otherwise) unless `key` overrides it, then encode."""
    if key is None:
        key = ROOT_KEY if rec.lease == 0 else lease_key(ROOT_KEY, rec.lease)
    return _codec().issue_control(rec, key)


def grant(lease_id: int, *, seq: int, expect: int = 0, granted: int = CAP_GRANTABLE,
          issued: int = 0, expiry: int = 1000, **extra) -> bytes:  # fmt: skip
    return issue(
        record(
            "lease",
            LeaseGrant(lease_id, granted, issued, expiry, HOLDER),
            seq=seq,
            expect=expect,
            lease=0,
            **extra,
        )
    )


# --- the data-plane fixtures: a module, its packs and plans across a registry move -----------


@dataclass(frozen=True)
class Artifacts:
    """Packs and plans of one program across registry states: `current` is what a generation
    switch installs; `old` was hydrated before a resource moved (maxima differ); `undermax`
    before a resource moved UNDER unchanged maxima (only the vector sees it); `novector` carries
    no per-resource vector at all (a v1-v3 pack, a plan with an empty vector)."""

    registry: tuple  # (map_gen, data_gen, topo_gen, registry_digest) of the current state
    old_registry: tuple
    pack: bytes
    pack_old: bytes
    pack_undermax: bytes
    pack_novector: bytes
    plan: bytes
    plan_old: bytes
    plan_undermax: bytes
    plan_novector: bytes


_ARTIFACTS: Artifacts | None = None


def artifacts() -> Artifacts:
    """Built once: vector_add(64) placed for AVX-512; resource generations moved by hand."""
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS
    from ..abi import encode, encode_plan
    from ..examples import vector_add
    from ..gem.execution_plan import plan_from_realization
    from ..gem.streampack import generation_vector, hydrate
    from ..kbcir import optimize
    from ..kbcir.cost import TargetProfile, Theta

    module = vector_add(64)
    target = TargetProfile.x86_avx512()
    result = optimize(module, target, Theta.cool())
    rids = sorted(module.resources)
    # Two resources at different map generations, so one can move under the maxima.
    first, second = rids[0], rids[1]
    module.resources[first] = replace(module.resources[first], map_gen=2)
    module.touch()

    def state(m):
        pack = hydrate(m, result)
        plan = plan_from_realization(m, result, target, "eft", plan="plan0")
        reg = (pack.map_gen, pack.data_gen, pack.topo_gen, registry_digest(generation_vector(m)))
        return encode(pack), encode_plan(plan), reg, pack, plan

    base_pack, base_plan, base_reg, _, _ = state(module)
    # move `second` under the maxima (its map_gen 0 -> 1; the maximum stays 2)
    module.resources[second] = replace(module.resources[second], map_gen=1)
    module.touch()
    mid_pack, mid_plan, mid_reg, _, _ = state(module)
    # move `first` past the maxima (map_gen 2 -> 3): the current registry
    module.resources[first] = replace(module.resources[first], map_gen=3)
    module.touch()
    cur_pack, cur_plan, cur_reg, cur_pack_obj, cur_plan_obj = state(module)
    # the undermax pair: resident = mid (maxima 2), artifact = base (maxima 2, vector older)
    assert base_reg[:3] == mid_reg[:3] and base_reg[3] != mid_reg[3]
    # the novector pair: the current artifacts with the vector removed -- nothing else differs
    novector_pack = encode(replace(cur_pack_obj, generations=[]))
    novector_plan = encode_plan(replace(cur_plan_obj, generations=[]))
    _ARTIFACTS = Artifacts(
        registry=cur_reg,
        old_registry=mid_reg,
        pack=cur_pack,
        pack_old=mid_pack,
        pack_undermax=base_pack,
        pack_novector=novector_pack,
        plan=cur_plan,
        plan_old=mid_plan,
        plan_undermax=base_plan,
        plan_novector=novector_plan,
    )
    return _ARTIFACTS


def generation_record(reg: tuple, *, seq: int, expect: int, lease: int = 1, **extra) -> bytes:
    map_gen, data_gen, topo_gen, digest = reg
    return issue(
        record(
            "generation",
            GenerationSwitch(map_gen, data_gen, topo_gen, digest),
            seq=seq,
            expect=expect,
            lease=lease,
            **extra,
        )
    )


# --- the round-trip corpus ----------------------------------------------------------------


def record_corpus() -> list[tuple[str, bytes]]:
    """Valid records of every kind, scope and reason at the edges of their fields."""
    out: list[tuple[str, bytes]] = []
    a, b = digest_of("artifact-a"), digest_of("artifact-b")
    for index, scope in enumerate(CONTROL_SCOPES):
        out.append(
            (
                f"lease/scope-{scope}",
                grant(index + 1, seq=index + 1, scope=scope, subject=index * 1000),
            )
        )
    out.append(("lease/min", grant(1, seq=1, granted=CAPABILITY["quiesce"], issued=0, expiry=1)))
    out.append(
        (
            "lease/max",
            issue(
                record(
                    "lease",
                    LeaseGrant(_U64, CAP_GRANTABLE, _U64 - 1, _U64, _U64),
                    seq=_U64,
                    expect=_U32,
                    lease=0,
                    boundary=_U64,
                    subject=_U64,
                )
            ),
        )
    )
    for index, reason in enumerate(("none", "remap", "rewrite", "topology")):
        out.append(
            (
                f"generation/{reason}",
                generation_record(
                    (index, index * 7, 1, digest_of(f"reg-{reason}")),
                    seq=index + 1,
                    expect=index,
                    reason=reason,
                ),
            )
        )
    out.append(
        (
            "generation/max",
            generation_record(
                (_U32, _U32, _U32, bytes([0xFF]) * 32), seq=_U64, expect=_U32 - 1, lease=_U64
            ),
        )
    )
    for index, reason in enumerate(("none", "activation", "rollback", "teardown")):
        out.append(
            (
                f"quiesce/{reason}",
                issue(
                    record(
                        "quiesce",
                        Quiesce(10 + index),
                        seq=index + 1,
                        expect=index,
                        lease=1,
                        boundary=10,
                        reason=reason,
                    )
                ),
            )
        )
    out.append(
        (
            "quiesce/deadline-is-boundary",
            issue(record("quiesce", Quiesce(_U64), seq=9, expect=3, lease=2, boundary=_U64)),
        )
    )
    for index, reason in enumerate(("none", "promotion", "repair")):
        out.append(
            (
                f"activate/{reason}",
                issue(
                    record(
                        "activate",
                        Activate(a, ZERO32 if index == 0 else b),
                        seq=index + 1,
                        expect=index,
                        lease=1,
                        reason=reason,
                    )
                ),
            )
        )
    out.append(
        (
            "activate/max",
            issue(
                record(
                    "activate",
                    Activate(bytes([0xFF]) * 32, bytes([0xFE]) * 32),
                    seq=_U64,
                    expect=_U32 - 1,
                    lease=_U64,
                    boundary=_U64,
                    subject=_U64,
                    scope="channel",
                )
            ),
        )
    )
    for index, reason in enumerate(("none", "correctness", "health", "policy")):
        out.append(
            (
                f"rollback/{reason}",
                issue(
                    record(
                        "rollback",
                        Rollback(a, digest_of(f"token-{index}")),
                        seq=index + 1,
                        expect=index + 1,
                        lease=3,
                        reason=reason,
                    )
                ),
            )
        )
    for index, reason in enumerate(("none", "withdrawn", "superseded")):
        out.append(
            (
                f"cancel/{reason}",
                issue(
                    record(
                        "cancel",
                        Cancel(1, index + 1),
                        seq=index + 2,
                        expect=index,
                        lease=1,
                        reason=reason,
                    )
                ),
            )
        )
    out.append(
        (
            "cancel/max",
            issue(record("cancel", Cancel(1, _U64 - 1), seq=_U64, expect=_U32, lease=_U64)),
        )
    )
    return out


# --- the malformed variants -----------------------------------------------------------------


def _recrc(data: bytes) -> bytes:
    body = data[:-4]
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def _put(data: bytes, offset: int, fmt: str, value) -> bytes:
    out = bytearray(data)
    struct.pack_into("<" + fmt, out, offset, value)
    return _recrc(bytes(out))


def malformed_variants() -> list[tuple[str, bytes, str]]:
    """(name, bytes, the bcir_status both rails must return): one variant per wire law."""
    a = digest_of("artifact-a")
    lease_rec = grant(5, seq=3)
    gen_rec = generation_record((1, 1, 1, digest_of("reg")), seq=2, expect=1)
    quiesce_rec = issue(record("quiesce", Quiesce(8), seq=2, expect=1, lease=1, boundary=4))
    act = issue(record("activate", Activate(a, digest_of("prev")), seq=4, expect=2, lease=1))
    roll = issue(record("rollback", Rollback(a, digest_of("tok")), seq=5, expect=3, lease=1))
    cancel = issue(record("cancel", Cancel(2, 3), seq=6, expect=4, lease=1))
    v: list[tuple[str, bytes, str]] = [
        ("empty", b"", "BCIR_ERR_TRUNCATED"),
        ("header-63", act[:63], "BCIR_ERR_TRUNCATED"),
        ("below-minimum", act[:99], "BCIR_ERR_TRUNCATED"),
        ("short-by-one", act[:-1], "BCIR_ERR_TRUNCATED"),
        ("trailing-byte", _recrc(act + b"\x00"), "BCIR_ERR_TRAILING"),
        ("bad-magic", _recrc(b"BCTX" + act[4:]), "BCIR_ERR_MAGIC"),
        ("version-0", _put(act, 4, "H", 0), "BCIR_ERR_VERSION"),
        ("version-2", _put(act, 4, "H", 2), "BCIR_ERR_VERSION"),
        ("flags", _put(act, 6, "H", 1), "BCIR_ERR_RESERVED"),
        ("reserved0", _put(act, 11, "B", 1), "BCIR_ERR_RESERVED"),
        ("kind-0", _put(act, 8, "B", 0), "BCIR_ERR_CONTROL"),
        ("kind-7", _put(act, 8, "B", 7), "BCIR_ERR_CONTROL"),
        ("body-len-short", _put(lease_rec, 12, "I", 39), "BCIR_ERR_CONTROL"),
        ("body-len-long", _put(lease_rec, 12, "I", 41), "BCIR_ERR_CONTROL"),
        ("crc", act[:-4] + bytes(b ^ 0xFF for b in act[-4:]), "BCIR_ERR_CRC"),
        ("scope-5", _put(act, 9, "B", 5), "BCIR_ERR_CONTROL"),
        ("reason-outside-set", _put(act, 10, "B", 3), "BCIR_ERR_CONTROL"),
        ("capability-not-kind", _put(act, 24, "Q", CAPABILITY["rollback"]), "BCIR_ERR_CONTROL"),
        ("sequence-0", _put(act, 40, "Q", 0), "BCIR_ERR_CONTROL"),
        ("lease-0-on-switch", _put(act, 48, "Q", 0), "BCIR_ERR_CONTROL"),
        ("lease-on-grant", _put(lease_rec, 48, "Q", 1), "BCIR_ERR_CONTROL"),
        ("switch-not-plus-one", _put(act, 16, "I", 5), "BCIR_ERR_CONTROL"),
        ("switch-wraps", _put(_put(act, 20, "I", _U32), 16, "I", 0), "BCIR_ERR_CONTROL"),
        ("non-switch-moves", _put(quiesce_rec, 16, "I", 2), "BCIR_ERR_CONTROL"),
        ("lease-id-0", _put(lease_rec, 64, "Q", 0), "BCIR_ERR_CONTROL"),
        ("grant-delegates", _put(lease_rec, 72, "Q", CAP_GRANTABLE | 1), "BCIR_ERR_CONTROL"),
        ("grant-empty", _put(lease_rec, 72, "Q", 0), "BCIR_ERR_CONTROL"),
        ("grant-beyond-v1", _put(lease_rec, 72, "Q", 0x40), "BCIR_ERR_CONTROL"),
        ("lease-window-reversed", _put(lease_rec, 88, "Q", 0), "BCIR_ERR_CONTROL"),
        ("lease-no-holder", _put(lease_rec, 96, "Q", 0), "BCIR_ERR_CONTROL"),
        ("generation-reserved", _put(gen_rec, 76, "I", 1), "BCIR_ERR_RESERVED"),
        (
            "generation-zero-digest",
            _recrc(gen_rec[:80] + ZERO32 + gen_rec[112:]),
            "BCIR_ERR_CONTROL",
        ),
        ("drain-before-boundary", _put(quiesce_rec, 64, "Q", 3), "BCIR_ERR_CONTROL"),
        ("activate-zero-artifact", _recrc(act[:64] + ZERO32 + act[96:]), "BCIR_ERR_CONTROL"),
        ("activate-unchanged", _recrc(act[:64] + act[96:128] + act[96:]), "BCIR_ERR_CONTROL"),
        ("rollback-zero-restore", _recrc(roll[:64] + ZERO32 + roll[96:]), "BCIR_ERR_CONTROL"),
        ("rollback-zero-token", _recrc(roll[:96] + ZERO32 + roll[128:]), "BCIR_ERR_CONTROL"),
        ("cancel-first-0", _put(cancel, 64, "Q", 0), "BCIR_ERR_CONTROL"),
        ("cancel-reversed", _put(cancel, 64, "Q", 4), "BCIR_ERR_CONTROL"),
        ("cancel-not-earlier", _put(cancel, 72, "Q", 6), "BCIR_ERR_CONTROL"),
        ("mac-all-zero", _recrc(act[:128] + ZERO32 + act[160:]), "BCIR_ERR_MAC"),
    ]
    return v


# --- scenarios ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One plane operation and the (verdict, refusal) the specification requires of it."""

    op: int
    data: bytes = b""
    verdict: str = "none"
    refusal: str = "none"


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    steps: tuple[Step, ...]
    key: bytes = ROOT_KEY
    scope: str = SCOPE
    subject: int = SUBJECT
    checks: tuple = field(default_factory=tuple)  # (attribute, value) on the final Python plane


def _submit(data: bytes, verdict: str = "applied", refusal: str = "none") -> Step:
    return Step(OP_SUBMIT, data, verdict, refusal)


def _enter(verdict: str = "none", refusal: str = "none") -> Step:
    return Step(OP_ENTER, b"", verdict, refusal)


def _leave(verdict: str = "none", refusal: str = "none") -> Step:
    return Step(OP_LEAVE, b"", verdict, refusal)


def _advance(verdict: str = "none", refusal: str = "none") -> Step:
    return Step(OP_ADVANCE, b"", verdict, refusal)


def _admit_pack(data: bytes, verdict: str = "applied", refusal: str = "none") -> Step:
    return Step(OP_ADMIT_PACK, data, verdict, refusal)


def _admit_plan(data: bytes, verdict: str = "applied", refusal: str = "none") -> Step:
    return Step(OP_ADMIT_PLAN, data, verdict, refusal)


def _poke_boundary(value: int) -> Step:
    return Step(OP_POKE_BOUNDARY, struct.pack("<Q", value))


def _poke_in_flight(value: int) -> Step:
    return Step(OP_POKE_IN_FLIGHT, struct.pack("<Q", value))


A_DIGEST, B_DIGEST = digest_of("artifact-a"), digest_of("artifact-b")


def _bootstrap(*, registry: tuple | None = None, activate: bool = True) -> list[Step]:
    """Grant lease 1 (every grantable capability), install a registry -- the current one unless
    told otherwise -- at generation 1 and, unless told not to, activate artifact A
    (generation 2)."""
    reg = artifacts().registry if registry is None else registry
    steps = [_submit(grant(1, seq=1)), _submit(generation_record(reg, seq=1, expect=0))]
    if activate:
        steps.append(
            _submit(issue(record("activate", Activate(A_DIGEST), seq=2, expect=1, lease=1)))
        )
    return steps


def _token_of(activate_bytes: bytes) -> bytes:
    from ..gem.control import rollback_token

    return rollback_token(_codec().signed_part(activate_bytes))


def transition_scenarios() -> list[Scenario]:
    """Each kind's transition at quiescence, with the state it must leave behind."""
    reg = artifacts().registry
    act_a = issue(record("activate", Activate(A_DIGEST), seq=2, expect=1, lease=1))
    act_b = issue(record("activate", Activate(B_DIGEST, A_DIGEST), seq=3, expect=2, lease=1))
    roll = issue(
        record(
            "rollback",
            Rollback(A_DIGEST, _token_of(act_b)),
            seq=4,
            expect=3,
            lease=1,
            reason="health",
        )
    )
    act_deferred = issue(record("activate", Activate(A_DIGEST), seq=2, expect=1, lease=1))
    return [
        Scenario(
            "transition/lease",
            "transition",
            (_submit(grant(1, seq=1)),),
            checks=(("leases", 1), ("last_lease_id", 1), ("root_sequence", 1)),
        ),
        Scenario(
            "transition/generation",
            "transition",
            (_submit(grant(1, seq=1)), _submit(generation_record(reg, seq=1, expect=0))),
            checks=(("generation", 1), ("registry", reg)),
        ),
        Scenario(
            "transition/quiesce",
            "transition",
            (
                _submit(grant(1, seq=1)),
                _submit(
                    issue(
                        record("quiesce", Quiesce(3), seq=1, expect=0, lease=1, reason="activation")
                    )
                ),
                _enter("refused", "draining"),
                _advance(),
                _advance(),
                _advance(),
                _advance(),
                _enter(),
            ),
            checks=(("draining", False), ("in_flight", 1), ("boundary", 4)),
        ),
        Scenario(
            "transition/activate",
            "transition",
            (
                _submit(grant(1, seq=1)),
                _submit(generation_record(reg, seq=1, expect=0)),
                _submit(act_a),
            ),
            checks=(("generation", 2), ("artifact", A_DIGEST), ("previous", ZERO32)),
        ),
        Scenario(
            "transition/rollback",
            "transition",
            (
                _submit(grant(1, seq=1)),
                _submit(generation_record(reg, seq=1, expect=0)),
                _submit(act_a),
                _submit(act_b),
                _submit(roll),
            ),
            checks=(
                ("generation", 4),
                ("artifact", A_DIGEST),
                ("previous", ZERO32),
                ("token", ZERO32),
            ),
        ),
        Scenario(
            "transition/cancel",
            "transition",
            (
                _submit(grant(1, seq=1)),
                _submit(generation_record(reg, seq=1, expect=0)),
                _enter(),
                _submit(act_deferred, "deferred"),
                _submit(
                    issue(
                        record("cancel", Cancel(2, 2), seq=3, expect=1, lease=1, reason="withdrawn")
                    )
                ),
                _leave(),
            ),
            checks=(("generation", 1), ("pending", None), ("artifact", ZERO32)),
        ),
    ]


def authority_scenarios() -> list[Scenario]:
    """Forgeries: each ends with a record a correct plane must refuse for a named reason."""
    boot = _bootstrap()
    act = record("activate", Activate(B_DIGEST, A_DIGEST), seq=3, expect=2, lease=1)
    return [
        Scenario(
            "authority/foreign-root",
            "authority",
            (
                _submit(
                    issue(
                        record(
                            "lease",
                            LeaseGrant(1, CAP_GRANTABLE, 0, 9, HOLDER),
                            seq=1,
                            expect=0,
                            lease=0,
                        ),
                        FOREIGN_KEY,
                    ),
                    "refused",
                    "mac",
                ),
            ),
        ),
        Scenario(
            "authority/another-lease-key",
            "authority",
            (*boot, _submit(issue(act, lease_key(ROOT_KEY, 2)), "refused", "mac")),
        ),
        Scenario(
            "authority/holder-mints-lease",
            "authority",
            (
                *boot,
                _submit(
                    issue(
                        record(
                            "lease",
                            LeaseGrant(2, CAP_GRANTABLE, 0, 9, HOLDER),
                            seq=2,
                            expect=2,
                            lease=0,
                        ),
                        lease_key(ROOT_KEY, 1),
                    ),
                    "refused",
                    "mac",
                ),
            ),
        ),
        Scenario(
            "authority/capability",
            "authority",
            (
                *boot,
                _submit(grant(2, seq=2, expect=2, granted=CAPABILITY["quiesce"])),
                _submit(issue(replace(act, lease=2, sequence=1)), "refused", "capability"),
            ),
        ),
        Scenario(
            "authority/cross-plane",
            "authority",
            (*boot, _submit(issue(replace(act, subject=SUBJECT + 1)), "refused", "subject")),
        ),
        Scenario(
            "authority/replay",
            "authority",
            (*boot, _submit(issue(act)), _submit(issue(act), "refused", "replay")),
        ),
        Scenario(
            "authority/unknown-lease",
            "authority",
            (*boot, _submit(issue(replace(act, lease=9)), "refused", "lease")),
        ),
        Scenario(
            "authority/expired-lease",
            "authority",
            (
                *boot,
                _submit(grant(3, seq=2, expect=2, expiry=2)),
                _advance(),
                _advance(),
                _submit(issue(replace(act, lease=3, sequence=1)), "refused", "expired"),
            ),
        ),
        Scenario(
            "authority/cancel-foreign",
            "authority",
            (
                *boot,
                _submit(grant(2, seq=2, expect=2, granted=CAPABILITY["cancel"])),
                _enter(),
                _submit(issue(act), "deferred"),
                _submit(
                    issue(record("cancel", Cancel(3, 3), seq=4, expect=2, lease=2)),
                    "refused",
                    "nothing",
                ),
                _leave("applied"),
            ),
        ),
    ]


def stale_record_scenarios() -> list[Scenario]:
    """A record of every kind minted against generation 1 while the plane stands at 2."""
    boot = _bootstrap()
    reg = artifacts().registry
    stale = {
        "lease": grant(2, seq=2, expect=1),
        "generation": generation_record(artifacts().old_registry, seq=3, expect=1),
        "quiesce": issue(record("quiesce", Quiesce(5), seq=3, expect=1, lease=1)),
        "activate": issue(
            record("activate", Activate(B_DIGEST, A_DIGEST), seq=3, expect=1, lease=1)
        ),
        "rollback": issue(
            record("rollback", Rollback(A_DIGEST, digest_of("t")), seq=3, expect=1, lease=1)
        ),
        "cancel": issue(record("cancel", Cancel(1, 2), seq=3, expect=1, lease=1)),
    }
    assert reg  # the bootstrap installed it
    return [
        Scenario(f"stale/{kind}", "stale", (*boot, _submit(data, "refused", "stale")))
        for kind, data in stale.items()
    ]


def stale_artifact_scenarios() -> list[Scenario]:
    """The data plane: after the bootstrap installs the current registry, every artifact from
    an older registry state is refused by bytes; the current pack and plan are admitted."""
    art = artifacts()
    boot = _bootstrap()
    mid = _bootstrap(registry=art.old_registry)  # maxima equal to the undermax artifacts'
    return [
        Scenario(
            "stale/pack-old",
            "stale",
            (*boot, _admit_pack(art.pack), _admit_pack(art.pack_old, "refused", "stale")),
        ),
        Scenario(
            "stale/pack-undermax",
            "stale",
            (*mid, _admit_pack(art.pack_old), _admit_pack(art.pack_undermax, "refused", "stale")),
        ),
        Scenario(
            "stale/pack-novector",
            "stale",
            (*boot, _admit_pack(art.pack_novector, "refused", "stale")),
        ),
        Scenario(
            "stale/plan-old",
            "stale",
            (*boot, _admit_plan(art.plan), _admit_plan(art.plan_old, "refused", "stale")),
        ),
        Scenario(
            "stale/plan-undermax",
            "stale",
            (*mid, _admit_plan(art.plan_old), _admit_plan(art.plan_undermax, "refused", "stale")),
        ),
        Scenario(
            "stale/plan-novector",
            "stale",
            (*boot, _admit_plan(art.plan_novector, "refused", "stale")),
        ),
    ]


def stale_scenarios() -> list[Scenario]:
    return stale_record_scenarios() + stale_artifact_scenarios()


def midphase_scenarios() -> list[Scenario]:
    """A switch requested mid-phase (or before its boundary) is deferred -- the resident
    generation does not move -- and then applied exactly once at the boundary, or refused there
    and reported; never applied early, never applied twice, never lost."""
    art = artifacts()
    boot = _bootstrap()
    act = issue(record("activate", Activate(B_DIGEST, A_DIGEST), seq=3, expect=2, lease=1))
    roll = issue(record("rollback", Rollback(A_DIGEST, _token_of(act)), seq=4, expect=3, lease=1))
    gen = generation_record(art.old_registry, seq=3, expect=2, reason="remap")
    return [
        Scenario(
            "midphase/activate",
            "midphase",
            (*boot, _enter(), _submit(act, "deferred"), _leave("applied")),
            checks=(("generation", 3), ("artifact", B_DIGEST), ("pending", None)),
        ),
        Scenario(
            "midphase/rollback",
            "midphase",
            (*boot, _submit(act), _enter(), _submit(roll, "deferred"), _leave("applied")),
            checks=(("generation", 4), ("artifact", A_DIGEST)),
        ),
        Scenario(
            "midphase/generation",
            "midphase",
            (
                *boot,
                _enter(),
                _submit(gen, "deferred"),
                _admit_pack(art.pack),
                _leave("applied"),
                _admit_pack(art.pack, "refused", "stale"),
                _admit_pack(art.pack_old),
            ),
            checks=(("generation", 3), ("registry", art.old_registry)),
        ),
        Scenario(
            "midphase/future-boundary",
            "midphase",
            (
                *boot,
                _submit(replace_boundary(act, 3), "deferred"),
                _advance(),
                _advance(),
                _advance("applied"),
                _advance(),
            ),
            checks=(("generation", 3), ("boundary", 4)),
        ),
        Scenario(
            "midphase/nested",
            "midphase",
            (*boot, _enter(), _enter(), _submit(act, "deferred"), _leave(), _leave("applied")),
            checks=(("generation", 3), ("in_flight", 0), ("boundary", 1)),
        ),
        Scenario(
            "midphase/drain",
            "midphase",
            (
                *boot,
                _enter(),
                _submit(
                    issue(
                        record(
                            "quiesce", Quiesce(10), seq=3, expect=2, lease=1, reason="activation"
                        )
                    )
                ),
                _submit(
                    issue(
                        record("activate", Activate(B_DIGEST, A_DIGEST), seq=4, expect=2, lease=1)
                    ),
                    "deferred",
                ),
                _enter("refused", "draining"),
                _leave("applied"),
                _enter(),
            ),
            checks=(("generation", 3), ("draining", False)),
        ),
        Scenario(
            "midphase/lease-lapses",
            "midphase",
            (
                *boot,
                _submit(grant(2, seq=2, expect=2, expiry=2)),
                _submit(
                    replace_boundary(
                        record("activate", Activate(B_DIGEST, A_DIGEST), seq=1, expect=2, lease=2),
                        3,
                    ),
                    "deferred",
                ),
                _advance(),
                _advance(),
                _advance("refused", "expired"),
            ),
            checks=(("generation", 2), ("artifact", A_DIGEST), ("pending", None)),
        ),
        Scenario(
            "midphase/cancelled",
            "midphase",
            (
                *boot,
                _enter(),
                _submit(act, "deferred"),
                _submit(
                    issue(
                        record("cancel", Cancel(3, 3), seq=4, expect=2, lease=1, reason="withdrawn")
                    )
                ),
                _leave(),
            ),
            checks=(("generation", 2), ("artifact", A_DIGEST), ("pending", None)),
        ),
    ]


def replace_boundary(data: bytes, boundary: int):
    """Re-issue an encoded record with another `boundary` (returns bytes; accepts a record)."""
    codec = _codec()
    rec = data if isinstance(data, ControlRecord) else codec.decode_control(data)
    return issue(replace(rec, boundary=boundary, mac=b""))


def witness_scenarios() -> list[Scenario]:
    """Every refusal the plane can name, each produced once (L22: every rule admits a witness);
    the families above cover the rest."""
    art = artifacts()
    boot = _bootstrap()
    reg = art.registry
    act_c = record("activate", Activate(B_DIGEST, A_DIGEST), seq=3, expect=2, lease=1)
    grants = [_submit(grant(i, seq=i, expiry=5)) for i in range(1, 9)]
    return [
        Scenario("witness/malformed", "witness", (_submit(b"BCTL", "refused", "malformed"),)),
        Scenario(
            "witness/ahead",
            "witness",
            (*boot, _submit(issue(replace(act_c, expect=3, generation=4)), "refused", "ahead")),
        ),
        Scenario(
            "witness/early",
            "witness",
            (
                *boot,
                _submit(
                    issue(record("quiesce", Quiesce(9), seq=3, expect=2, lease=1, boundary=5)),
                    "refused",
                    "early",
                ),
            ),
        ),
        Scenario(
            "witness/pending",
            "witness",
            (
                *boot,
                _enter(),
                _submit(issue(act_c), "deferred"),
                _submit(issue(replace(act_c, sequence=4)), "refused", "pending"),
            ),
        ),
        Scenario(
            "witness/mismatch-activate",
            "witness",
            (
                *boot,
                _submit(
                    issue(replace(act_c, body=Activate(B_DIGEST, B_DIGEST[::-1]))),
                    "refused",
                    "mismatch",
                ),
            ),
        ),
        Scenario(
            "witness/mismatch-rollback",
            "witness",
            (
                *boot,
                _submit(issue(act_c)),
                _submit(
                    issue(
                        record(
                            "rollback",
                            Rollback(A_DIGEST, digest_of("wrong")),
                            seq=4,
                            expect=3,
                            lease=1,
                        )
                    ),
                    "refused",
                    "mismatch",
                ),
            ),
        ),
        Scenario(
            "witness/mismatch-generation",
            "witness",
            (*boot, _submit(generation_record(reg, seq=3, expect=2), "refused", "mismatch")),
        ),
        Scenario(
            "witness/nothing-to-roll-back",
            "witness",
            (
                *boot,
                _submit(
                    issue(
                        record(
                            "rollback", Rollback(A_DIGEST, digest_of("t")), seq=3, expect=2, lease=1
                        )
                    ),
                    "refused",
                    "nothing",
                ),
            ),
        ),
        Scenario(
            "witness/nothing-to-cancel",
            "witness",
            (
                *boot,
                _submit(
                    issue(record("cancel", Cancel(1, 2), seq=3, expect=2, lease=1)),
                    "refused",
                    "nothing",
                ),
            ),
        ),
        Scenario(
            "witness/full",
            "witness",
            (
                *grants,
                _submit(grant(9, seq=9, expiry=5), "refused", "full"),
                _advance(),
                _advance(),
                _advance(),
                _advance(),
                _advance(),
                _submit(grant(10, seq=10, expiry=9)),
            ),
        ),
        Scenario(
            "witness/duplicate",
            "witness",
            (
                _submit(grant(2, seq=1)),
                _submit(grant(2, seq=2), "refused", "duplicate"),
                _submit(grant(1, seq=3), "refused", "duplicate"),
            ),
        ),
        Scenario(
            "witness/expired-grant-and-drain",
            "witness",
            (
                _submit(grant(1, seq=1)),
                _advance(),
                _advance(),
                _submit(grant(2, seq=2, expiry=2), "refused", "expired"),
                _submit(
                    issue(record("quiesce", Quiesce(1), seq=1, expect=0, lease=1)),
                    "refused",
                    "expired",
                ),
                _submit(grant(3, seq=3, issued=5, expiry=9)),
                _submit(
                    issue(record("quiesce", Quiesce(9), seq=1, expect=0, lease=3)),
                    "refused",
                    "expired",
                ),
            ),
        ),
        Scenario(
            "witness/idle-busy",
            "witness",
            (
                _leave("refused", "idle"),
                _enter(),
                _advance("refused", "busy"),
                _leave(),
                _advance(),
            ),
        ),
        Scenario(
            "witness/exhausted",
            "witness",
            (
                _poke_boundary(_U64 - 1),
                _advance(),
                _advance("refused", "exhausted"),
                _enter(),
                _enter(),
                _leave(),
                _leave("refused", "exhausted"),
                _poke_in_flight(_U64),
                _enter("refused", "exhausted"),
            ),
            checks=(("boundary", _U64), ("in_flight", _U64)),
        ),
        Scenario(
            "witness/admit-without-registry",
            "witness",
            (_admit_pack(art.pack, "refused", "stale"), _admit_plan(art.plan, "refused", "stale")),
        ),
        Scenario(
            "witness/admit-malformed",
            "witness",
            (
                *boot,
                _admit_pack(art.pack[:-1], "refused", "malformed"),
                _admit_plan(art.plan[:-1], "refused", "malformed"),
            ),
        ),
        Scenario(
            "witness/lease-sequence-spaces",
            "witness",
            (
                *boot,
                _submit(grant(2, seq=2, expect=2)),
                _submit(issue(record("quiesce", Quiesce(9), seq=1, expect=2, lease=2))),
                _submit(
                    issue(record("quiesce", Quiesce(9), seq=1, expect=2, lease=2)),
                    "refused",
                    "replay",
                ),
                _submit(grant(3, seq=2, expect=2), "refused", "replay"),
            ),
        ),
    ]


def all_scenarios() -> list[Scenario]:
    return (
        transition_scenarios()
        + authority_scenarios()
        + stale_scenarios()
        + midphase_scenarios()
        + witness_scenarios()
    )


# --- running a scenario on the Python rail --------------------------------------------------------


def trace_line(index: int, op: int, outcome, plane: ControlPlane) -> str:
    """One operation's trace line -- the format the C harness prints."""
    kind = CONTROL_KINDS.index(outcome.kind) + 1 if outcome.kind else 0
    status = outcome.status if op == OP_SUBMIT else "-"
    return (
        f"{index} {VERDICTS.index(outcome.verdict)} {REFUSALS.index(outcome.refusal)} "
        f"k={kind} s={outcome.sequence} g={outcome.generation} {status} "
        f"{plane.state_digest().hex()}"
    )


def run_python(scenario: Scenario) -> tuple[ControlPlane, list, list[str]]:
    plane = ControlPlane(scenario.key, scenario.scope, scenario.subject)
    outcomes, lines = [], []
    for index, step in enumerate(scenario.steps):
        if step.op == OP_SUBMIT:
            outcome = plane.submit(step.data)
        elif step.op == OP_ENTER:
            outcome = plane.enter()
        elif step.op == OP_LEAVE:
            outcome = plane.leave()
        elif step.op == OP_ADVANCE:
            outcome = plane.advance()
        elif step.op == OP_ADMIT_PACK:
            outcome = plane.admit_pack(step.data)
        elif step.op == OP_ADMIT_PLAN:
            outcome = plane.admit_plan(step.data)
        else:
            (value,) = struct.unpack("<Q", step.data)
            setattr(plane, "boundary" if step.op == OP_POKE_BOUNDARY else "in_flight", value)
            outcome = ControlOutcome("none", generation=plane.generation)
        outcomes.append(outcome)
        lines.append(trace_line(index, step.op, outcome, plane))
    return plane, outcomes, lines


def _check_value(plane: ControlPlane, attribute: str, want) -> bool:
    if attribute == "leases":
        return len(plane.leases) == want
    if attribute == "pending":
        return plane.pending == want
    return getattr(plane, attribute) == want


def conforms(scenario: Scenario, outcomes, plane: ControlPlane | None = None) -> bool:
    """Whether a rail's outcomes are the ones the specification requires (and, for the Python
    rail, whether the final state carries the declared values)."""
    if len(outcomes) != len(scenario.steps):
        return False
    for step, got in zip(scenario.steps, outcomes):
        verdict, refusal = got if isinstance(got, tuple) else (got.verdict, got.refusal)
        if (verdict, refusal) != (step.verdict, step.refusal):
            return False
    if plane is not None:
        return all(_check_value(plane, a, w) for a, w in scenario.checks)
    return True


def python_stale_boundaries() -> list[tuple[str, bool]]:
    """(boundary, holds) for the stale-generation boundaries with no C twin: the trusted
    loader, context-shard activation and the verifier's R11 over a generation record. A
    boundary holds only when it admits the current generation AND refuses the stale one for
    staleness -- a boundary that refuses everything proves nothing (L2)."""
    return [
        ("staged.TrustedLoader.install", _loader_holds()),
        ("context_shard.certify_context_activation", _context_holds()),
        ("verify.verify_control_record", _verifier_holds()),
    ]


def _loader_holds() -> bool:
    from ..asn1.staged import Artifact, TrustedLoader
    from ..asn1.tags import Asn1Error

    loader = TrustedLoader(key=ROOT_KEY)
    loader.install(Artifact(1, "p", b"code", loader.sign(b"code", "p", 1)))
    if loader.generation != 1:
        return False
    try:  # an artifact minted against generation 0, presented at generation 1
        loader.install(Artifact(1, "q", b"code2", loader.sign(b"code2", "q", 1)))
    except Asn1Error as exc:
        return "not newer" in str(exc) and loader.generation == 1
    return False


def _context_holds() -> bool:
    from ..kbcir.context_shard import (
        ContextShardCatalog,
        ContextShardEntry,
        ContextShardManifest,
        certify_context_activation,
    )

    def sha(label: str) -> str:
        return hashlib.sha256(label.encode()).hexdigest()

    manifest = ContextShardManifest(
        "bcirq8",
        "bcirq8-v1",
        "quantized",
        sha("payload"),
        7,
        sha("base-model"),
        sha("hardware"),
        sha("selector"),
        sha("provenance"),
        sha("certificate"),
        "control-fixture@1",
        1,
    )
    entry = ContextShardEntry(manifest.selector_sha256, manifest.digest, 10)
    catalog = ContextShardCatalog(manifest.base_model_sha256, sha("hardware"), 2, (entry,))

    def activate(generation: int) -> bool:
        try:
            certify_context_activation(
                catalog,
                manifest,
                manifest.selector_sha256,
                activation_generation=generation,
                quiescent=True,
            )
        except ValueError as exc:
            return "advance" not in str(exc)  # refused, but not for staleness
        return True

    return activate(catalog.generation + 1) and not activate(catalog.generation)


def _verifier_holds() -> bool:
    from ..examples import vector_add
    from ..gem.streampack import generation_vector

    try:
        from ..verify import verify_control_record
    except ImportError:  # the parent tree: no boundary to hold
        return False
    module = vector_add(64)
    vector = generation_vector(module)
    rid = min(module.resources)
    moved = [replace(g, map_gen=g.map_gen + 1) if g.rid == rid else g for g in vector]
    maxima = (max(g.map_gen for g in vector), max(g.data_gen for g in vector))
    current = generation_record((*maxima, 1, registry_digest(vector)), seq=1, expect=0)
    older = generation_record((*maxima, 1, registry_digest(moved)), seq=1, expect=0)
    return not verify_control_record(module, current) and bool(verify_control_record(module, older))


# --- the C rail ----------------------------------------------------------------------------------

C_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "runtime", "c"))
C_SOURCES = ("bcir_control_plane.c", "bcir_sha256.c", "bcir_runtime.c", "test_control_plane.c")


def compiler() -> str | None:
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def build_harness(tmp: str) -> str | None:
    """Compile runtime/c/test_control_plane.c against the freestanding plane. None without a
    compiler (the quick tier hides one on purpose) and in an installed package (the wheel does
    not ship runtime/c); in a source checkout a visible compiler must build it -- a missing
    source there is a failure, never a skip (L21)."""
    from .run_all import _is_source_checkout

    cc = compiler()
    if cc is None:
        return None
    if not os.path.isfile(os.path.join(C_DIR, "test_control_plane.c")):
        if _is_source_checkout():
            raise RuntimeError("runtime/c/test_control_plane.c is missing from the checkout")
        return None
    exe = os.path.join(tmp, "test_control_plane")
    build = subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            C_DIR,
            *[os.path.join(C_DIR, source) for source in C_SOURCES],
            "-o",
            exe,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if build.returncode != 0:
        raise RuntimeError(f"control harness build failed: {build.stderr[-2000:]}")
    return exe


def c_api(exe: str) -> tuple[int, str]:
    """The API's own fail-closed laws on the C rail (`--api`): (exit code, output)."""
    run = subprocess.run([exe, "--api"], capture_output=True, text=True, timeout=120)
    return run.returncode, run.stdout + run.stderr


def _frames(blobs) -> bytes:
    return b"".join(struct.pack("<I", len(b)) + b for b in blobs)


def encode_script(scenarios) -> bytes:
    out = bytearray(b"BCTS" + struct.pack("<I", len(scenarios)))
    for s in scenarios:
        out += struct.pack("<B", len(s.key)) + s.key
        out += struct.pack("<BQI", CONTROL_SCOPES.index(s.scope), s.subject, len(s.steps))
        for step in s.steps:
            out += struct.pack("<BI", step.op, len(step.data)) + step.data
    return bytes(out)


def run_c(exe: str, tmp: str, scenarios) -> list[list[str]]:
    """Run scenarios on the C rail; one list of trace lines per scenario."""
    path = os.path.join(tmp, "script.bin")
    with open(path, "wb") as fh:
        fh.write(encode_script(scenarios))
    run = subprocess.run([exe, "--script", path], capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        raise RuntimeError(f"control harness failed: {run.stderr[-2000:]}")
    traces: list[list[str]] = [[] for _ in scenarios]
    for line in run.stdout.splitlines():
        head, _, rest = line.partition(" ")
        traces[int(head)].append(rest)
    return traces


def c_outcomes(trace: list[str]) -> list[tuple[str, str]]:
    out = []
    for line in trace:
        parts = line.split()
        out.append((VERDICTS[int(parts[1])], REFUSALS[int(parts[2])]))
    return out


def c_dump(exe: str, tmp: str, blobs) -> list[str]:
    """Decode records on the C rail: one line per record (fields, MAC verdict) or a status."""
    path = os.path.join(tmp, "records.bin")
    with open(path, "wb") as fh:
        fh.write(_frames(blobs))
    run = subprocess.run(
        [exe, "--dump", path, "--key", ROOT_KEY.hex()], capture_output=True, text=True, timeout=120
    )
    if run.returncode != 0:
        raise RuntimeError(f"control dump failed: {run.stderr[-2000:]}")
    return run.stdout.splitlines()


def parse_c_dump(line: str) -> tuple[ControlRecord | None, str, bool]:
    """Rebuild the record the C rail decoded, field by field; (record, status, mac_ok)."""
    fields = dict(part.split("=", 1) for part in line.split()[1:])
    if line.startswith("refused"):
        return None, fields["status"], False
    kind = CONTROL_KINDS[int(fields["kind"]) - 1]
    num = lambda name: int(fields[name])  # noqa: E731
    raw = lambda name: bytes.fromhex(fields[name])  # noqa: E731
    if kind == "lease":
        body = LeaseGrant(
            num("lease_id"), num("granted"), num("issued"), num("expiry"), num("holder")
        )
    elif kind == "generation":
        body = GenerationSwitch(num("map_gen"), num("data_gen"), num("topo_gen"), raw("registry"))
    elif kind == "quiesce":
        body = Quiesce(num("deadline"))
    elif kind == "activate":
        body = Activate(raw("artifact"), raw("previous"))
    elif kind == "rollback":
        body = Rollback(raw("restore"), raw("token"))
    else:
        body = Cancel(num("first"), num("last"))
    from ..gem.control import REASONS

    rec = ControlRecord(
        kind=kind,
        scope=CONTROL_SCOPES[num("scope")],
        subject=num("subject"),
        generation=num("generation"),
        expect=num("expect"),
        boundary=num("boundary"),
        sequence=num("sequence"),
        lease=num("lease"),
        body=body,
        reason=REASONS[kind][num("reason")],
        mac=raw("mac"),
    )
    if num("capability") != CAPABILITY[kind]:
        return None, "capability-mismatch", False
    return rec, "BCIR_OK", fields.get("macok") == "1"


# --- the G14 rows (tools/perf/gemplus_baseline.py --group control) --------------------------------

ROWS = (
    "control.abi.mismatches",
    "control.malformed.accepted",
    "control.stale.accepted",
    "control.deferred.lost",
    "control.decisions.nonconforming",
    "control.traces.divergent",
)


def honest_key(data: bytes) -> bytes:
    """The key an honest issuer MACs a framed record under: the root key for a grant, the
    lease's key otherwise (read from the header, so a malformed record still gets one)."""
    (lease,) = struct.unpack_from("<Q", data, 48) if len(data) >= 56 else (0,)
    return ROOT_KEY if lease == 0 else lease_key(ROOT_KEY, lease)


def _parsed(line: str):
    try:
        return parse_c_dump(line)
    except (KeyError, ValueError, IndexError):
        return None, "unparseable", False


def _c_conforms(scenario: Scenario, trace: list[str]) -> bool:
    try:
        return conforms(scenario, c_outcomes(trace))
    except (ValueError, IndexError):
        return False


def measure(exe: str, tmp: str) -> dict[str, float]:
    """The six G14 rows over the declared corpora, on both rails (`exe`: the built harness).

    control.abi.mismatches           corpus records whose Python encode -> C decode -> Python
                                     re-encode is not byte-identical, or whose honest MAC either
                                     rail's keyed check refuses
    control.malformed.accepted       (malformed variant, rail) pairs not refused with the status
                                     the specification declares
    control.stale.accepted           (stale fixture, rail) pairs not refused as specified, plus
                                     the Python-only boundaries that do not hold
    control.deferred.lost            (mid-phase fixture, rail) pairs whose switch is not deferred
                                     and then decided exactly as specified
    control.decisions.nonconforming  (transition / authority / witness scenario, rail) pairs
                                     whose decisions are not the specification's
    control.traces.divergent         scenarios whose two rails' traces differ in any verdict,
                                     refusal, status or resident state digest

    Every path is a count: a rail that crashes, or prints what cannot be parsed, fails every
    fixture it was handed (L1)."""
    from ..abi.control_abi import ControlError, check_control_mac, decode_control, encode_control

    out: dict[str, float] = {}
    corpus = record_corpus()
    try:
        lines = c_dump(exe, tmp, [blob for _, blob in corpus])
    except RuntimeError:
        lines = []
    if len(lines) != len(corpus):
        lines = [""] * len(corpus)
    mismatches = 0
    for (_name, blob), line in zip(corpus, lines):
        try:
            python_ok = encode_control(decode_control(blob)) == blob and check_control_mac(
                blob, honest_key(blob)
            )
        except ControlError:
            python_ok = False
        record, _status, mac_ok = _parsed(line)
        c_ok = record is not None and mac_ok and encode_control(record) == blob
        mismatches += not (python_ok and c_ok)
    out["control.abi.mismatches"] = float(mismatches)

    variants = malformed_variants()
    try:
        lines = c_dump(exe, tmp, [blob for _, blob, _ in variants])
    except RuntimeError:
        lines = []
    if len(lines) != len(variants):
        lines = [""] * len(variants)
    accepted = 0
    for (_name, blob, want), line in zip(variants, lines):
        try:
            decode_control(blob)
            python_status = "BCIR_OK"
        except ControlError as exc:
            python_status = exc.status
        accepted += (python_status != want) + (line != f"refused status={want}")
    out["control.malformed.accepted"] = float(accepted)

    scenarios = all_scenarios()
    try:
        traces = run_c(exe, tmp, scenarios)
    except (RuntimeError, ValueError, IndexError):
        traces = [[] for _ in scenarios]
    rows = {"stale": 0, "midphase": 0, "decisions": 0, "divergent": 0}
    for scenario, trace in zip(scenarios, traces):
        try:
            plane, outcomes, lines = run_python(scenario)
            python_ok = conforms(scenario, outcomes, plane)
        except Exception:  # noqa: BLE001 -- a plane that raises decided nothing
            python_ok, lines = False, None
        wrong = (not python_ok) + (not _c_conforms(scenario, trace))
        family = scenario.family if scenario.family in ("stale", "midphase") else "decisions"
        rows[family] += wrong
        rows["divergent"] += trace != lines
    boundaries = python_stale_boundaries()
    out["control.stale.accepted"] = float(rows["stale"] + sum(not ok for _, ok in boundaries))
    out["control.deferred.lost"] = float(rows["midphase"])
    out["control.decisions.nonconforming"] = float(rows["decisions"])
    out["control.traces.divergent"] = float(rows["divergent"])
    return out
