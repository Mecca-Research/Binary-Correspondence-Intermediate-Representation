# BCIR ControlRecordV1 binary ABI — v1 (frozen, normative)

The control record is **the control plane as bytes** (GEM+ roadmap G14, staged plan S3-A):
lease, generation, quiescence, activation, rollback and cancellation as small, fixed, versioned
records that a resident runtime decides by their bytes. Before this format those transitions were
prose, an `admit(map_gen, data_gen)` argument and two Python methods that raise; a stale
generation was refused only where a caller remembered to pass the right number. The reference
codec is [`bcir/abi/control_abi.py`](../../bcir/abi/control_abi.py); the abstract values and the
resident plane are [`bcir/gem/control.py`](../../bcir/gem/control.py); the C view is
[`runtime/c/bcir_control_plane.h`](../../runtime/c/bcir_control_plane.h) with the freestanding
decoder, verifier and plane in `bcir_control_plane.c` and the shared SHA-256 / HMAC-SHA256 in
`bcir_sha256.c`. The rails must agree record for record and decision for decision (the G14 rows
in `tools/perf/gemplus_baseline.py --group control` pin the round trip, the refusals and the
plane's traces).

This is a BCIR artifact format, frozen the way [`ExecutionPlanV1`](BCIR_EXECUTION_PLAN_ABI.md)
is frozen. It is **not** the BCIR UAPI: the driver roadmap keeps the UAPI unfrozen until the
UART and virtio-blk drivers have demonstrated their lifecycles
([`BCIR_DRIVER_KERNEL_ROADMAP.md`](BCIR_DRIVER_KERNEL_ROADMAP.md) §7.1), and a future `ioctl` or
IPC adapter marshals these values rather than exposing them.

## Conventions

- **Endianness:** little-endian; every field is fixed-width. There are **no variable-length
  fields**: each kind's length is a constant, so a record is bounded before any field is trusted
  and a decision needs one bounded read and no allocation.
- **CRC:** CRC-32 (zlib; `bcir_crc32` on the C rail) of every preceding byte — the cheap
  corruption gate a *keyless* reader can apply.
- **MAC:** HMAC-SHA256 (RFC 2104). The CRC detects corruption; the MAC is authority. The field is
  named `mac`, never `signature`: an HMAC proves possession of a key, not the identity of a signer
  ([`bcir/asn1/staged.py`](../../bcir/asn1/staged.py) states the same limitation for the trusted
  loader, and this format inherits it).

## Envelope

```
ControlRecordV1 := header(64) || body(BODY_BYTES[version][kind]) || mac[32] || crc32(4)

  mac   = HMAC-SHA256(K, header || body)
  crc32 = CRC-32(header || body || mac)
  K     = the root key                      when lease == 0 (a lease grant)
          lease_key(root key, lease)        otherwise
  lease_key(root, id) = HMAC-SHA256(root, "BCTL/lease/v1" || 0x00 || u64le(id))
```

A record is therefore `100 + BODY_BYTES` bytes. The largest v1 record is 164 bytes; the declared
bound `CONTROL_RECORD_MAX_BYTES` is **192**. The live ring (G15) sizes its CONTROL slots from
that bound: 256 bytes, the bound plus the ring's 24-byte slot header rounded up to a cache line,
and it refuses to format a control ring that could not carry a record of the bound.

## Header (64 bytes, one cache line; every `u64` at an 8-aligned offset)

| Offset | Field | Type | Meaning |
|---|---|---|---|
| 0 | `magic` | `u8[4]` | `"BCTL"` |
| 4 | `version` | `u16` | `1` |
| 6 | `flags` | `u16` | reserved (0) |
| 8 | `kind` | `u8` | `1` lease, `2` generation, `3` quiesce, `4` activate, `5` rollback, `6` cancel |
| 9 | `scope` | `u8` | what `subject` names: `0` module, `1` resource, `2` mapping, `3` session, `4` channel |
| 10 | `reason` | `u8` | why the record was issued — a closed per-kind code (below); `0` = none |
| 11 | `reserved0` | `u8` | reserved (0) |
| 12 | `body_len` | `u32` | `BODY_BYTES[version][kind]`, redundant on purpose: a keyless reader bounds the record before it trusts the kind table |
| 16 | `generation` | `u32` | the generation this record asserts — the **new** one for a switch |
| 20 | `expect` | `u32` | the resident generation the issuer witnessed: the compare-and-swap witness |
| 24 | `capability` | `u64` | the capability the record exercises (below) |
| 32 | `boundary` | `u64` | the earliest phase/event boundary at which the record may take effect |
| 40 | `sequence` | `u64` | the issuer's monotone sequence under its lease (the root's own for a grant) |
| 48 | `lease` | `u64` | the lease the record is issued under; `0` only on a lease grant |
| 56 | `subject` | `u64` | the handle the record acts on (`scope` says which kind of handle) |

The C `bcir_ctl_header` is this layout without padding (a static assertion locks it at 64 bytes).
`expect` is what makes a stale generation refusable **by bytes**: the refusal does not depend on
what the receiver remembers about the sender, only on the resident state and the record.

## Kinds, bodies, capabilities and reasons

| kind | body (fixed) | body bytes | record bytes | capability | reasons |
|---|---|---:|---:|---:|---|
| `1` lease | `lease_id:u64 granted:u64 issued_epoch:u64 expiry_epoch:u64 holder:u64` | 40 | 140 | `0x01` | `0` none |
| `2` generation | `map_gen:u32 data_gen:u32 topo_gen:u32 reserved:u32 registry_digest:u8[32]` | 48 | 148 | `0x02` | `0` none, `1` remap, `2` rewrite, `3` topology |
| `3` quiesce | `drain_deadline:u64` | 8 | 108 | `0x04` | `0` none, `1` activation, `2` rollback, `3` teardown |
| `4` activate | `artifact_sha256:u8[32] previous_sha256:u8[32]` | 64 | 164 | `0x08` | `0` none, `1` promotion, `2` repair |
| `5` rollback | `restore_sha256:u8[32] rollback_token:u8[32]` | 64 | 164 | `0x10` | `0` none, `1` correctness, `2` health, `3` policy |
| `6` cancel | `first_sequence:u64 last_sequence:u64` | 16 | 116 | `0x20` | `0` none, `1` withdrawn, `2` superseded |

**Switches** are the kinds that move the resident generation: `generation`, `activate`,
`rollback`. The others — `lease`, `quiesce`, `cancel` — act without moving it.

- **lease** — the root grants lease `lease_id` to principal `holder`, with the capability mask
  `granted`, valid for the boundaries `[issued_epoch, expiry_epoch)`. The holder receives
  `lease_key(root, lease_id)` and nothing else: it can issue records under its own lease and
  cannot mint a lease (a grant is MACed under the root key), widen its mask (the mask lives in
  the resident's table, from the root's record) or speak for another lease (another key).
- **generation** — installs a new registry state: the per-resource generation vector's maxima
  (`map_gen`, `data_gen`), the topology generation, and `registry_digest`, the digest of the
  vector itself (below). Packs and plans are admitted against it by bytes.
- **quiesce** — the plane drains: new work is refused from now until a switch lands or the
  boundary `drain_deadline` passes, whichever is first.
- **activate** — installs artifact `artifact_sha256` in place of `previous_sha256` (all-zero =
  nothing live), a content witness beside `expect`.
- **rollback** — restores the previous artifact `restore_sha256`, presenting the
  `rollback_token` of the activation it undoes. Rollback moves the generation **forward**, as the
  trusted loader's does: two live states never share a tag.
- **cancel** — withdraws the issuer's own pending switch whose sequence lies in
  `[first_sequence, last_sequence]`.

The capability a record exercises is exactly its kind's bit; `0x3F` is every v1 capability.
A lease's `granted` mask may name any of them **except** `0x01`: v1 has no delegation.

## Digests

```
registry_digest(vector) = SHA-256("BCTL/registry/v1" || 0x00 || vector)
    vector = rid:u32 map_gen:u32 data_gen:u32 per resource, RIDs strictly ascending
             -- exactly the StreamPack v4 / ExecutionPlanV1 generation tail, so a C reader
                hashes the artifact's own bytes in place
rollback_token(activate) = SHA-256("BCTL/token/v1" || 0x00 || header || body)
```

The token is public (anyone holding the activate record can compute it): it is a binding —
which activation a rollback undoes — not a secret.

## The wire laws (keyless; both rails, before publication and again on read)

The encoder refuses to emit, and every decoder refuses to read, a record that violates any of
these. They are checked **in this order**, so both rails name the same first violation:

| # | Law | C status |
|---|---|---|
| 1 | at least `64 + 36` bytes | `BCIR_ERR_TRUNCATED` |
| 2 | `magic == "BCTL"` | `BCIR_ERR_MAGIC` |
| 3 | `1 ≤ version ≤ CONTROL_VERSION_MAX` | `BCIR_ERR_VERSION` |
| 4 | `flags == 0`, `reserved0 == 0` | `BCIR_ERR_RESERVED` |
| 5 | `kind ∈ 1..6` | `BCIR_ERR_CONTROL` |
| 6 | `body_len == BODY_BYTES[version][kind]` | `BCIR_ERR_CONTROL` |
| 7 | the buffer is exactly `100 + body_len` bytes: shorter is `BCIR_ERR_TRUNCATED`, longer is `BCIR_ERR_TRAILING` — never an extension point | |
| 8 | the CRC matches | `BCIR_ERR_CRC` |
| 9 | header field laws: `scope ≤ 4`; `reason` in the kind's closed set; `capability` is the kind's bit; `sequence ≥ 1`; `lease == 0` iff `kind == lease`; a switch has `expect < 2³²−1` and `generation == expect + 1` (no wrap: a stale tag cannot come back to life), any other kind `generation == expect` | `BCIR_ERR_CONTROL` |
| 10 | body laws: **lease** — `lease_id ≠ 0`, `granted ≠ 0`, `granted ⊆ 0x3E` (no bits outside v1, no `0x01`), `expiry_epoch > issued_epoch`, `holder ≠ 0`; **generation** — `reserved == 0` (`BCIR_ERR_RESERVED`), a nonzero `registry_digest`; **quiesce** — `drain_deadline ≥ boundary`; **activate** — a nonzero `artifact_sha256` distinct from `previous_sha256`; **rollback** — nonzero `restore_sha256` and `rollback_token`; **cancel** — `1 ≤ first_sequence ≤ last_sequence < sequence` (only earlier records can be cancelled) | `BCIR_ERR_CONTROL` |
| 11 | the `mac` is not all zero | `BCIR_ERR_MAC` |

The Python codec raises `ControlError` (an `AbiError`) carrying the same status name for the same
bytes. A keyed reader then checks the MAC (`check_control_mac` / `bcir_ctl_check_mac`,
`BCIR_ERR_MAC` on a mismatch, compared in constant time).

## The resident plane (the semantic trust boundary)

A plane is one handle's resident control state — `ControlPlane` on the oracle,
`bcir_ctl_state` in C, fixed-size and allocation-free. It is created with the root key
(16–64 bytes) and its own identity `(scope, subject)`, and holds:

| State | Meaning |
|---|---|
| `generation` | the resident generation (starts at 0) |
| `artifact`, `previous`, `token` | the live artifact's digest, the rollback target, the live activation's token (all-zero = none) |
| `registry` | the installed registry state — `map_gen`, `data_gen`, `topo_gen`, `registry_digest` — or none |
| `boundary` | how many phase/event boundaries have been crossed |
| `in_flight` | phases in progress (`enter` / `leave`) |
| `draining`, `drain_deadline` | the quiesce state |
| `root_sequence`, `last_lease_id` | the root's last sequence; the largest lease id ever granted |
| `leases[≤ 8]` | `(lease_id, granted, issued, expiry, holder, last_sequence)`, ascending by id — and each lease's derived `lease_key(root, lease_id)`, computed once at the grant and wiped when the lease leaves the table (a cache, excluded from the state digest) |
| `pending` | at most one deferred switch, held as its own verified bytes |

### Deciding a record (`submit`)

Every record ends **applied**, **deferred** or **refused**, and a refused record changes nothing
— not a sequence counter, not a lease, not the pending slot. The checks run in this order and the
first failure names the refusal (a C plane whose `bcir_ctl_init` was refused, or never ran,
holds no key and refuses every well-formed record as **mac**: no key verifies anything):

1. **malformed** — the wire laws above (the status is carried with the refusal).
2. **lease** — `lease ≠ 0` and no such lease in the table.
3. **mac** — the MAC does not verify under the root key (a grant) or the lease's key.
4. **subject** — `(scope, subject)` is not this plane's: a record minted for another handle is
   not replayable here, even under a valid MAC.
5. **expired** — the lease's window `[issued_epoch, expiry_epoch)` does not contain `boundary`
   (not yet valid, or lapsed).
6. **capability** — the lease's `granted` mask does not include the record's capability.
7. **replay** — `sequence` is not above the lease's (or the root's) last accepted sequence.
8. **stale** — `expect` is below the resident generation; **ahead** — above it. A record minted
   against a state the plane has left, or one it never reached, is refused by its own bytes.
9. **kind laws** —
   - lease grant: `lease_id ≤ last_lease_id` is **duplicate** (ids never recur, so a replayed
     grant cannot resurrect a sequence space); `expiry_epoch ≤ boundary` is **expired**; a table
     still full after dropping lapsed leases is **full**;
   - quiesce: `drain_deadline < boundary` is **expired**;
   - cancel: no pending switch of this lease with a sequence in range is **nothing**;
   - any switch while another is pending is **pending** (one slot: the pending switch's
     compare-and-swap stays true until it is applied or cancelled);
   - activate: `previous_sha256 ≠ artifact` is **mismatch**;
   - rollback: no previous artifact is **nothing**; `restore_sha256 ≠ previous` or
     `rollback_token ≠ token` is **mismatch**;
   - generation: a registry identical to the installed one is **mismatch** (a switch that
     changes nothing would invalidate every admitted pack for nothing).
10. **timing** — a non-switch whose `boundary` is ahead of the plane's is **early**. A switch
    applies only at a boundary: if a phase is in flight, or its `boundary` is ahead, it is
    **deferred** — its sequence is consumed and its bytes held in the pending slot, the resident
    generation does not move.
11. Otherwise the record is **applied**: a grant enters the table; a quiesce starts the drain; a
    cancel empties the pending slot; a generation switch installs the registry; an activate moves
    `artifact` to `previous`, installs the new digest and mints its token; a rollback restores
    `previous` and clears it and the token; a switch releases a drain. Every switch sets the
    resident generation to the record's.

### Crossing a boundary (`enter`, `leave`, `advance`)

`enter` refuses while draining (**draining**) and otherwise counts a phase in. `leave` refuses
when nothing is in flight (**idle**); the `leave` that returns `in_flight` to zero crosses a
boundary, as does `advance` when nothing is in flight (it refuses otherwise: **busy**). Crossing a
boundary: the counter advances (it refuses to wrap); a drain whose deadline has passed lapses; a
pending switch that is due is re-checked against its lease's window and then applied — **once,
at that boundary** — or refused there (its lease lapsed while it waited) and reported; a pending
switch not yet due stays pending. A switch is therefore never applied mid-phase, never applied
twice, and never lost: every deferred record reaches exactly one reported decision.

### Admitting a pack or a plan (the data plane, by bytes)

`admit_pack` / `bcir_ctl_admit_pack` and `admit_plan` / `bcir_ctl_admit_plan` refuse an artifact
unless the plane holds a registry and the artifact carries exactly it: the pack's generation
vector (or the plan's) must digest to `registry_digest`, and a pack's header maxima and `topo_gen`
must equal the installed ones. A pack hydrated before a generation switch, one whose single
resource moved under unchanged maxima, a v1–v3 pack with no vector, and a plan minted under an
older vector are all refused (`BCIR_ERR_STALE`) — against the state records established, not
against a number a caller passed. The artifact's own verification runs first and its status
passes through (`bcir_sp_verify_semantic`, `bcir_ep_verify`; on the oracle the StreamPack and plan
decoders).

### The resident state digest

`state_digest` / `bcir_ctl_state_digest` is SHA-256 over the canonical serialization of every
field above except the key and the cached lease keys — fixed order, little-endian, leases
ascending by id, the pending record's bytes last:

```
SHA-256( "BCTL/state/v1" || 0x00
      || generation:u32 || boundary:u64 || in_flight:u64 || draining:u8 || drain_deadline:u64
      || artifact[32] || previous[32] || token[32]
      || has_registry:u8 || map_gen:u32 || data_gen:u32 || topo_gen:u32 || registry_digest[32]
      || root_sequence:u64 || last_lease_id:u64 || scope:u8 || subject:u64
      || n_leases:u32 || n_leases x (lease_id, granted, issued, expiry, holder, last_sequence : u64)
      || pending_len:u32 || pending[pending_len] )
```

(the registry fields are zero when none is installed). It lets two rails prove they hold the same
resident state after every operation, and it is what the parity traces compare.

## Gates

| Gate | Where |
|---|---|
| every corpus record round-trips Python → C → Python byte for byte, its MAC accepted on both rails | `control.abi.mismatches`; `bcir/tests/test_control_plane.py`; `tools/c/check_runtime.sh` |
| one malformed variant per wire law, refused on both rails with the declared status | `control.malformed.accepted` |
| a stale record, pack or plan refused at every boundary (the plane on both rails, the trusted loader, context-shard activation, `verify_control_record`) | `control.stale.accepted` |
| a mid-phase switch deferred, then decided exactly once at a boundary | `control.deferred.lost` |
| every kind's transition, every forgery and every refusal code decided as specified | `control.decisions.nonconforming` |
| the two rails' traces identical to the resident state digest after every operation | `control.traces.divergent` |
| the API's fail-closed laws (a refused `bcir_ctl_init` leaves no key; a missing key is never the empty key) | `test_control_plane --api` |
| the gate fires: the plane rebuilt with the deferral law removed turns a row red | `tools/c/check_runtime.sh` |
| the decoder and the plane under ASan/UBSan: records sealed under the plane's key, the plane's invariants asserted after every operation | `runtime/c/fuzz_control_plane.c` in `tools/c/fuzz_streampack.sh`; the Python decoder campaign's `control` surface |

The corpora and the expected outcomes live in one place, `bcir/tests/control_fixtures.py`, and the
six rows are measured by one function the harness, the gate and the tests share
(`tools/perf/gemplus_baseline.py --group control`).

## Where the record travels

- **ASN.1**: the `BCIR-ControlPlane` module (OID `{ 1 3 6 1 4 1 62596 4 }`,
  [`docs/BCIR_ASN1_X690_ABI.md`](../BCIR_ASN1_X690_ABI.md) §3c) projects the record under DER,
  canonical OER and JER; the native octets survive the round trip byte for byte.
- **Not BCAB.** A bundle is immutable and content-addressed; a control record is a decision with
  a lifetime. A bundle that carried an activation would assert its own liveness, and a replayed
  bundle would carry a replayed admission — the conflation of integrity and authority the trusted
  loader exists to prevent. If a bundle-resident generation witness is ever needed, the data-plane
  hand-off (G16) is where a real caller would create the requirement.
- **Not MLIR.** The law rail has no control op and no byte decoder; nothing in this format is a
  legality verdict (the plane invariant: no plane carries a verdict except the verifier's own
  output). `verify.verify_control_record` holds a generation record to the module's registry
  (R11) on the oracle.
- **The ring (G15)** carries these records second, after telemetry
  ([`BCIR_LIVE_RING_ABI.md`](BCIR_LIVE_RING_ABI.md)): a CONTROL ring is BACKPRESSURE (a control
  record is never overwritten) with 256-byte slots (never unsendable), and all 52 scenarios
  decide identically through a live control ring as submitted directly, on both rails
  (`ring.control.divergent`; `test_control_plane --via-ring`). **The data-plane hand-off (G16)** owns generation gating at the C++ `admit()` seam and
  the invalidation of channel handles at a switch; this format supplies the resident state they
  will read.

## Versioning (the freeze)

- v1 is **frozen**: the layout and the laws above do not change.
- Growth is **append-only**: a later version may carve bits out of `flags` and `reserved0`, add
  kinds, and append fields to the **tail** of a kind's body, declaring the new length in the
  `(version, kind)` table. `control_version` returns the lowest version that carries a record, so
  a record that needs nothing new stays byte-identical v1.
- A reader refuses a version above the maximum it supports and refuses nonzero reserved bytes:
  reserved and trailing bytes are not implicit ABI.

Writers refuse values that cannot be represented exactly: integers never mask or wrap, and a
digest or MAC is exactly 32 bytes.

## Not claimed

- **Not a signature scheme.** HMAC-SHA256 under a shared root key proves possession of the key.
  There is no asymmetric signature, no key distribution, no revocation and no rotation; a holder
  of the root key can forge anything, and a holder of a lease key anything its lease permits.
- **No capability enforcement beyond the record.** The mask is checked against the lease by
  bytes. There is no OS capability, sandbox or MMU behind it.
- **No transport of its own.** A boundary is a function call on either rail; the live ring (G15)
  carries records between two endpoints of one shared region (threads, or processes on one
  host). No socket or eventfd, and no IPC claim beyond that.
- **No replay protection beyond `sequence`, `expect` and lease ids that never recur.** No nonce,
  no clock, no distributed ordering; one issuer per lease (no MPSC).
- **No performance claim.** Every G14 row is an exact count.
