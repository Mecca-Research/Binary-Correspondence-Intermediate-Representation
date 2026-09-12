# BCIR ExecutionPlanV1 binary ABI — v1 (frozen, normative) + v2 (append-only)

The ExecutionPlan is **the plan as bytes** (GEM+ roadmap G11, staged plan S1-C): the plan a
[StreamPack](BCIR_STREAMPACK_ABI.md) was derived from and what every reader prices. The pack
stays the executable; the plan carries what G1 makes canonical — the K_BCIR realization and
the one placement both executors run and both pricers read — together with the static-memory
planner's lifetimes and addresses, the movement-edge family G8 will fill, and the registry's
per-resource generation vector (R11). The reference encoder/decoder is
[`bcir/abi/execution_plan_abi.py`](../../bcir/abi/execution_plan_abi.py); the C view is
[`runtime/c/bcir_execution_plan.h`](../../runtime/c/bcir_execution_plan.h) with the
freestanding decoder/verifier in `bcir_runtime.c`. The abstract value is
[`bcir/gem/execution_plan.py`](../../bcir/gem/execution_plan.py). The three must agree (the
parity gate `plan.abi.mismatches` pins Python encode → C decode → Python re-encode).

Before this format the canonical plan was Python objects: the C twin, the MLIR rail and a
resident executor could not read it, nothing round-tripped it, and a stale or malformed plan
had no bytes to be refused by. A pack is now bound to its plan by bytes on both rails
(`bcir_ep_check_pack`, `verify_execution_plan(pack=...)`).

## Conventions

The StreamPack ABI's, unchanged:

- **Endianness:** little-endian.
- **`str`** := `u16` byte-length, then that many UTF-8 bytes.
- **`i64`** := a two's-complement 64-bit integer (a step cost is signed).
- The tail stream (the decoupled GGG/random stream, the scheduler's `TAIL_STREAM = -1`) is
  spelled `0xFFFFFFFF` in a `stream` field.

## Header (64 bytes, cache-line aligned; every `u64` at an 8-aligned offset)

| Offset | Field | Type | Meaning |
|---|---|---|---|
| 0 | `magic` | `u8[4]` | `"BPLN"` |
| 4 | `version` | `u16` | `1` |
| 6 | `flags` | `u16` | reserved (0) |
| 8 | `mode` | `u8` | `0` = `eft` (phase-barriered LPT/EFT dispatch), `1` = `tokens` (token-pipelined) |
| 9 | `liveness` | `u8` | **v2** (append-only): the lifetimes' liveness domain — `0` = phase positions, `1` = the placement's own ticks; reserved (0) on v1 |
| 10 | `reserved` | `u8[2]` | pad (0) |
| 12 | `streams` | `u32` | affinity domains the plan was placed on (the tail is extra); ≥ 1 |
| 16 | `knee` | `u32` | the bandwidth knee the dispatch clamped to; `1..streams` |
| 20 | `n_steps` | `u32` | body record counts |
| 24 | `n_lifetimes` | `u32` | |
| 28 | `n_moves` | `u32` | |
| 32 | `n_gens` | `u32` | |
| 36 | `reserved` | `u8[4]` | pad (0) |
| 40 | `makespan` | `u64` | the placement's makespan (the scheduled price) |
| 48 | `module_hash` | `u64` | R13 `hash_module` of the goal graph the plan realizes |
| 56 | `target_hash` | `u64` | R13 `hash_target` of the target it was placed for (0 = none) |

The C `bcir_ep_header` is this layout without padding (a static assertion locks it at 64 bytes).

## Body (sequential, length-prefixed)

1. `source_plan : str` — the plan's name; the pack derived from it carries the same
   `source_plan`.
2. `steps[n_steps]`, one per claim in the realization's own order, each:
   `claim_id:u64  phase_id:u32  candidate:str  lane:u8  width:u32  cost:i64  stream:u32
   start:u64  duration:u64` — the realization (candidate name, lane, width, the plan's own
   step cost) and the placement (`stream`, `start`, `duration == max(0, cost)`; the slot's
   finish is `start + duration`).
3. `lifetimes[n_lifetimes]`, each: `rid:u32  bank:str  offset:u64  size:u64  alignment:u32
   first_phase:u32  last_phase:u32  [v2: first_tick:u64  last_tick:u64]` —
   `kbcir.static_memory.StaticAllocation`, RIDs strictly ascending: the declared phase span
   (closed) and, since v2, the **half-open liveness interval** in the header's liveness domain
   (`[first_phase, last_phase + 1)` under phase liveness; the placement's ticks — from the start
   of the first slot that touches the resource to the finish of the last — under schedule
   liveness). A v1 record reads the phase default.
4. `moves[n_moves]`, each: `rid:u32  src_bank:str  dst_bank:str  offset:u64  size:u64
   route:str  kind:u8  coherence:u8  map_gen:u32  data_gen:u32  after_claim:u64
   before_claim:u64` — a G8 movement edge: `kind` ∈ {`0` direct, `1` peer, `2` staged,
   `3` rematerialized, `4` compressed, `5` evicted}, `coherence` ∈ {`0` none, `1` flush,
   `2` invalidate, `3` writeback}, the generation it moves, and the overlap window (after
   `after_claim` finishes, before `before_claim` starts; `0` = unconstrained). Declared,
   carried and verified now; empty until G8 produces edges.
5. `generations[n_gens]`, each: `rid:u32  map_gen:u32  data_gen:u32` — the StreamPack v4
   record, RIDs strictly ascending: the registry's per-resource generation vector the plan was
   placed under.

## Trailer

- `crc32 : u32` — CRC-32 (zlib) of **every preceding byte**. Decoders reject a mismatch.

## The wire laws (both rails, before publication and again on read)

The encoder refuses to emit, and every decoder refuses to read, a plan that violates any of:

- `mode` legal; `streams ≥ 1`; `1 ≤ knee ≤ streams`;
- per step: a legal lane, a nonzero power-of-two width, `stream < streams` or the tail
  sentinel, `duration == max(0, cost)`, `start + duration ≤ makespan` (no 64-bit overflow);
  claim ids unique;
- lifetimes: RIDs strictly ascending, a non-empty bank, `size ≥ 1`, a power-of-two alignment
  the offset honors, `first_phase ≤ last_phase`, `last_tick > first_tick`, no 64-bit overflow of
  `offset + size`, and **no two lifetimes of one bank live at once at overlapping addresses**
  (the alias law by bytes — `BCIR_ERR_PLAN` / `AbiError`, judged on the half-open ticks);
- moves: non-empty banks, `size ≥ 1`, legal `kind` and `coherence` codes, no overflow;
- generations: RIDs strictly ascending;
- **exact body consumption**: the declared records end exactly at the CRC trailer. CRC-valid
  bytes inserted before a recomputed CRC are refused (`BCIR_ERR_TRAILING`); reserved header
  bytes must be zero (`BCIR_ERR_RESERVED`); a newer version is refused (`BCIR_ERR_VERSION`).

The C rail names them: `BCIR_ERR_PLAN` (a plan law), `BCIR_ERR_LANE` / `BCIR_ERR_WIDTH` (the
range gate), `BCIR_ERR_PROVENANCE` (a duplicated claim), `BCIR_ERR_GENERATION` (an unsorted
vector), `BCIR_ERR_OVERFLOW`, `BCIR_ERR_TRAILING`, `BCIR_ERR_TRUNCATED`, `BCIR_ERR_UTF8`,
`BCIR_ERR_CRC`. The Python codec raises `AbiError` for the same bytes.

## Semantic trust boundary

Three laws need more than the bytes, and both rails hold them:

- **The plan realizes THIS module** (`verify.verify_execution_plan`, R9/R13): every step names
  a claim the module declares, in the phase the module declares it, once, in topological phase
  order, and every claim is stepped; `module_hash` is the module's canonical digest (validated
  through the S1-B identity API, by content) and, with a target, `target_hash` and the stream
  geometry are the target's and the placement is what the canonical dispatch produces from the
  plan's own step costs. The C twin cannot see the module; it holds the structural laws above
  and the two below.
- **R11, the plan against the registry** (`verify_execution_plan`, `bcir_ep_check_generation_vector`):
  every carried entry must match the live registry exactly and every declared resource must
  have an entry. A plan minted under an older vector is stale ("rehydrate: repack" for
  `map_gen`, "replan" for `data_gen`); a resource declared after minting is stale; a plan with
  no vector is stale against any registry that declares resources.
- **The lifetimes cover the schedule** (`verify_execution_plan`, R9; G5): under schedule
  liveness every lifetime's ticks are exactly the plan's own placement's interval for its
  resource, under phase liveness the declared span covers every phase that touches it, and no
  two lifetimes of one bank alias. A plan whose lifetimes do not cover its schedule is refused.
  The static memory planner's own verifier (`verify_static_memory_plan(..., schedule=)`)
  additionally refuses a phase-liveness plan the placement does not refine — the report's
  two-phase alias fixture composed with the token placement.
- **The pack is the lowering of this plan** (`verify_execution_plan(pack=...)`,
  `bcir_ep_check_pack`; R10/R11): the pack's `source_plan` is the plan's; it carries exactly one
  segment per step, in step order, with the step's claim, phase, lane and width
  (`BCIR_ERR_PROVENANCE` otherwise); and its generation vector is the plan's entry for entry —
  **a pack whose plan carries an older (or any different) vector for any resource is refused as
  stale** (`BCIR_ERR_STALE`) on both rails.

## Readers

A reader that holds the bytes does not re-run the dispatch: `gem.execution_plan.schedule_of`
returns the placement as the executors' `GemSchedule` (slots, `slot_of`, affinity, makespan,
knee), `realization_of` returns the realization the StreamPack lowering hydrates and the
schedulers re-place (`durations_from` reads the step costs), and `plan.lifetimes` are the
static-memory planner's rows. The G11 gate `plan.readers.disagreements` holds, over every corpus
program in both placements, that the pricer's makespan and slots, both executors' placements,
the re-placement from the plan's own costs, the static-memory lifetimes and the pack hydrated
from the plan's bytes are identical to their in-memory counterparts.

## Where the plan travels

- **BCAB**: kind `25` (`EXECUTION_PLAN`), format `13` — a portable, non-executable sibling of the
  root StreamPack ([`BCIR_ARTIFACT_BUNDLE_ABI.md`](BCIR_ARTIFACT_BUNDLE_ABI.md) §4). Both readers
  run the complete wire verification before admitting the variant; the root stays a StreamPack.
- **ASN.1**: the `BCIR-ExecutionPlan` module (OID `{ 1 3 6 1 4 1 62596 3 }`,
  [`docs/BCIR_ASN1_X690_ABI.md`](../BCIR_ASN1_X690_ABI.md) §3b; projection version 2 carries
  `liveness` and the ticks) projects the abstract value under DER, OER and JER; the native
  octets — v1 or v2 — survive the round trip byte for byte.
- **MLIR**: `bcir.artifact.variant` accepts `kind = "execution_plan"` with
  `format = "execution_plan"` (the pair is closed). The law rail has no plan op of its own yet;
  the C decoder is the reader every rail can link.

## v2 (append-only): the liveness domain and the lifetime ticks (G5)

v2 is the first worked instance of the append-only evolution, landed with the schedule-aware
static memory planner (staged plan S1-D). It changes **no** v1 field offset:

- **Header** gains `liveness : u8` at offset **9** (carved from the v1 reserved pad; bytes
  10–11 and 36–39 stay reserved and must be zero). Decoders read `0` (phase) on v1 buffers,
  and a v1 buffer with a nonzero byte at 9 is refused as reserved (`BCIR_ERR_RESERVED`).
- **Lifetime records** append `first_tick:u64 last_tick:u64`, the half-open liveness interval
  in the header's domain. Step, movement and generation records are **unchanged**.
- **Encoders emit the lowest carrying version**: a plan under phase liveness whose lifetimes
  carry the phase default (`[first_phase, last_phase + 1)`) is byte-identical frozen v1; a
  plan whose lifetimes came from a schedule-liveness static plan (or carry any other ticks)
  encodes as v2. A v2 header over v1-shaped content is a legal, non-canonical spelling, as it
  is for the StreamPack.
- **The alias law by bytes** applies to every version: the C twin (`bcir_ep_verify`) and the
  Python codec refuse two lifetimes of one bank that are live at once at overlapping addresses.

## Versioning (the freeze)

- v1 is **frozen**: the field layout above does not change.
- New fields are **append-only** (the StreamPack's discipline: header pad first, record tails,
  then trailing record families — v2 above is the worked instance); a v1 reader of a v1 buffer
  is exact and lossless.
- A reader **rejects** a buffer whose `version` exceeds the maximum it supports (v2 today), and
  refuses nonzero reserved bytes: reserved or trailing bytes are not implicit ABI.

Writers reject values that cannot be represented exactly: integer fields never mask or wrap, a
`cost` outside the signed 64-bit range is refused, and a string longer than a `u16` byte length
is refused.
