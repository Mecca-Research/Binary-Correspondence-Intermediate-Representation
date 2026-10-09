# The K_BCIR planner's records — BKPI, BKPR and BKPB (version zero, experimental)

A native planner may own a certificate only after it reproduces the Python plan byte for byte over
a generated corpus — the way the C twins earn their rails (GEM+ roadmap G17, staged plan S4-A).
That needs the planner's whole input, and its whole output, as bytes:

- **BKPI**, the planner input: everything `realize.optimize(module, h, theta, policy)` reads, and
  nothing else;
- **BKPR**, the realization: the plan, one step per claim, and its score;
- **BKPB**, the binding (CXX4): what a StreamPack needs beyond the two -- the RIDs, the
  generation vector and the pack's name -- so the native hydrate writes the oracle's pack.

The executable oracle is [`bcir/abi/planner_abi.py`](../../bcir/abi/planner_abi.py), and the
planner it describes is [`bcir/kbcir/realize.py`](../../bcir/kbcir/realize.py). The native twin is
the freestanding [`runtime/c/bcir_kplan.h`](../../runtime/c/bcir_kplan.h) / `bcir_kplan.c`, with
no heap and no libc. It decodes a BKPI record, plans it, and writes the BKPR record the Python
encoder writes for the same plan. The harness is `runtime/c/test_kplan.c`, and the libFuzzer
target is `runtime/c/fuzz_kplan.c`.

**Version zero.** Neither record carries a compatibility promise until a consumer outside this
repository reads it. A change is a version bump, never a reinterpretation. They are not the BCIR
UAPI.

## Conventions

- Little-endian; every field has a fixed width.
- **CRC:** CRC-32 (zlib; `bcir_crc32` on the C rail) over every byte before the trailer.
- **Statuses** are the runtime's `bcir_status`; Python raises `PlannerAbiError.status` with the
  same names. S4-A appended `BCIR_ERR_PLANNER` (25). Every law other than the framing laws refuses
  with it, except an op that is not UTF-8 (`BCIR_ERR_UTF8`, checked by the runtime's one
  validator, `bcir_utf8_valid`), and a realization value above 2⁶³ − 1 (`BCIR_ERR_OVERFLOW`).

## BKPI — the planner input

```
header (64)  @0 magic "BKPI"  @4 version u16 = 0  @6 flags u16 = 0
             @8 n_phases u32  @12 n_deps u32  @16 n_claims u32  @20 n_refs u32
             @24 n_resources u32  @28 n_ops u32  @32 op_bytes u32  @36 n_widths u32
             @40 reserved[24] = 0
scope (168)  @64  target 8 x u32: cacheline, elem_bytes, gather_penalty, mem_unit,
                  base_overhead, thermal_density, power_density, per_op_heat
             @96  tiers 7 x (bw_factor u32, lat_factor u32), indexed by MemTier (L1 .. SSD)
             @152 theta 8 x u32, in Theta's declaration order
             @184 policy 12 x u32, the base weights in cost.DIMS order
sections     widths    n_widths x u32 -- lane_widths in declaration order
             phases    n_phases x (phase_id u32, n_deps u32, n_claims u32), module order
             deps      n_deps x u32 -- phase ids, each phase's in its declared order
             claims    n_claims x 48, phase by phase
             refs      n_refs x u32 -- operand resource indices, each claim's reads then writes
             resources n_resources x (declared u8, domain u8, access u8, reserved u8)
             op lens   n_ops x u16
             op bytes  op_bytes -- the op strings, concatenated
trailer      crc32 u32
```

A claim is 48 bytes:

```
@0 id u32   @4 opcode u8  @5 lane u8  @6 stride_class u8  @7 domain u8
@8 hazard u8 (unique 0, atomic 1, barriered 2)   @9 verify u8 (none 0, bounds 1, exact 2, hash 3)
@10 flags u8: volatile 1, dynamic 2, callee_sig 4, timing 8, lifetime 16, imm 32,
              tolerance_ulp 64, quantized_bits 128   @11 flags2 u8: precision 1
@12 count u32   @16 op u32 (an index into the op table)   @20 n_rd u16  @22 n_wr u16
@24 stride_k i64   @32 offset i64   @40 primary u32 (a resource index, or 0xFFFFFFFF)
@44 reserved u32 = 0
```

The flags are the facts the CSE exclusions read (`realize.cse_eligible`): a field set, not its
value. The op strings are compared by equality only, so an index into a sorted table carries them
exactly. `reduce.gather` is the one op whose spelling changes the enumeration.

**One spelling per input:**

- Resource indices are assigned in first-reference order: phases in module order, and within each
  claim its reads, then its writes, then its primary resource.
- A resource the module does not declare is an entry with `declared = 0`, and no domain or
  addressing model.
- A claim's primary resource is written only when it is the one the planner reads: the claim has
  no reads, and the resource is declared.
- The op table is strictly ascending and fully referenced.
- The tiers are resolved per normative tier by name, so a hierarchy that lacks one reads DRAM's
  factors (`MemoryHierarchy.by_name`).

`encode_input` decodes its own output before it returns it, so it refuses exactly what the decoder
refuses.

### The declared domain, and why it is exact

| field | bound |
|---|---|
| claims, phases, ops | ≤ 2²⁴ |
| operand references, dependencies, resources | ≤ 2²⁶ |
| operands of one claim (reads + writes) | ≤ 255 |
| lane widths | 1..16 entries, each 1..2¹⁶ |
| target constants, tier factors, policy weights | ≤ 2¹⁶ (tier factors, `elem_bytes`, `mem_unit` ≥ 1; `cacheline` a power of two) |
| Theta fields | ≤ 100 |
| count | u32 |

Inside these bounds every intermediate the planner forms is exact in 128 bits:

- A base component is under 2⁸¹. The access term, ⌈n/w⌉ × streams × `base_overhead` × stride
  penalty × `lat_factor` ≫ 8, is at most 2⁴⁰ × 2¹⁶ × 2¹⁶ × 2¹⁶ ≫ 8.
- A weight is under 2¹⁷.
- So an edge weight is under 12 × 2⁹⁸, and a path weight, over at most 2²⁴ columns, under 2¹²⁶.

Each multiplication multiplies a value under 2⁹⁶ by a factor under 2³², so the C twin's `kp_mul`
never loses a bit. The C planner therefore never wraps. It refuses a plan only when a value the
realization must carry exceeds 2⁶³ − 1 (`BCIR_ERR_OVERFLOW`), which is exactly when the Python
encoder refuses the same plan. Python's integers are unbounded, so the two rails agree on every
input in the domain.

Two fixtures make this observable:

- `planner_fixtures.wide_path_case`: the losing paths weigh more than 2⁶⁴ while the winner fits.
- `planner_fixtures.carry_case`: the one edge crosses 2⁶⁴ only through additions whose terms each
  fit a u64. A planner whose 128-bit addition dropped its carry reports the wrapped sum, which
  fits. `tools/c/check_runtime.sh` injects exactly that defect and requires the gate to fire.

### Wire laws (in this order, both rails)

`decode_input` / `bcir_kp_decode_input` apply them, reading in place with no scratch:

1. shorter than the fixed part (236 bytes) → `BCIR_ERR_TRUNCATED`
2. magic ≠ `BKPI` → `BCIR_ERR_MAGIC`
3. version ≠ 0 → `BCIR_ERR_VERSION`
4. the CRC → `BCIR_ERR_CRC`
5. flags or the header's reserved bytes not zero → `BCIR_ERR_RESERVED`
6. a section count outside its bound (`n_widths` 1..16) → `BCIR_ERR_PLANNER`
7. the size is not the sections' exact sum → `BCIR_ERR_TRUNCATED` (short) or
   `BCIR_ERR_TRAILING` (long)
8. the scope, in order: `cacheline`, then `elem_bytes` and `mem_unit`, the target constants, the
   tier factors, Theta, and the policy → `BCIR_ERR_PLANNER`
9. a lane width outside 1..2¹⁶ → `BCIR_ERR_PLANNER`
10. the phases' dependency or claim counts do not add up to the header's → `BCIR_ERR_PLANNER`
11. each resource: a pad byte → `BCIR_ERR_RESERVED`; a flag, domain or addressing model outside
    its vocabulary, or an undeclared resource carrying either → `BCIR_ERR_PLANNER`
12. the op table: the lengths do not add up → `BCIR_ERR_PLANNER`; an op not UTF-8 →
    `BCIR_ERR_UTF8`; not strictly ascending → `BCIR_ERR_PLANNER`
13. each claim in order:
    - its pad → `BCIR_ERR_RESERVED`;
    - its vocabulary, more than 255 operands, operands past the record's, an operand out of
      first-reference order, a primary resource the planner would not read, or an undeclared
      primary → `BCIR_ERR_PLANNER`
14. an operand reference or a resource never accounted for → `BCIR_ERR_PLANNER`

Three laws need memory proportional to the record, so the decoder does not apply them. They are
`check_input`'s and `bcir_kp_plan`'s, applied before any planning, in this order: a phase id names
one phase, a claim id names one claim, and every op string is referenced (`BCIR_ERR_PLANNER`).
The compact Python planner still plans a module with a repeated claim id: every occurrence plans
with the last one's realizations, as the id-keyed candidate map always made it. Such a module is
outside the native domain.

## BKPR — the realization

```
header (32)  @0 magic "BKPR"  @4 version u16 = 0  @6 flags u16 = 0  @8 n_steps u32
             @12 reserved u32 = 0  @16 score u64  @24 reserved u64 = 0
steps        n_steps x 120: claim_id u32 | phase_id u32 | width u32 | lane u8 | name u8 |
             reserved u16 | cost u64 | base 12 x u64 (cost.DIMS order)
trailer      crc32 u32
```

`name` codes: noop 0, barrier 1, atomic 2, blocked 3, gather 4, scalar 5, vec 6 (`vec<width>`),
strided 7, ux_bucket 8, tile 9. The steps are in planning order. `cost` is the realized cost of
the edge the path took, and `base` is the realization's coupled base cost, after the CSE or
deforestation discount.

Laws (`decode_realization` / `bcir_kp_decode_realization`, in order):

1. the framing laws, as BKPI's
2. the header's reserved fields → `BCIR_ERR_RESERVED`
3. the size is not `36 + 120·n_steps` → `BCIR_ERR_TRUNCATED` or `BCIR_ERR_TRAILING`
4. the score above 2⁶³ − 1 → `BCIR_ERR_OVERFLOW`
5. each step:
   - its pad → `BCIR_ERR_RESERVED`;
   - its cost or a base component above 2⁶³ − 1 → `BCIR_ERR_OVERFLOW`;
   - a lane, name or zero width outside the vocabulary, a `vec` narrower than 2, or a width-one
     realization (noop, barrier, atomic, blocked, gather, scalar, strided) wider than 1 →
     `BCIR_ERR_PLANNER`
6. the score is not the sum of the step costs → `BCIR_ERR_PLANNER`

## The planner, step by step

`bcir_kp_plan` is the oracle, function for function:

| step | oracle | C |
|---|---|---|
| phase order: dependency-first, roots in declaration order, a missing dependency ignored, a back edge skipped | `model.topological_phase_ids` | `kp_topo` |
| weights: the policy folded with Theta | `weights.weights` | `kp_weights` |
| the lane geometry: ascending unique widths, the cacheline-bucket and tile widths | `realize._geometry` | `kp_geometry` |
| the enumeration of a claim's realizations, with its primary resource's tier and addressing model | `realize._offer_rows` | `kp_specs`, `kp_factors` |
| the base cost | `realize._base_cost` | `kp_base` |
| the intra-phase discount: CSE by value-numbered identity over eligible claims, else deforestation unless a barrier fences it | `realize.fused_offer`, `cse_eligible`, `cse_identity` | `kp_cse_eligible`, the identity table, the version and fence stamps |
| an edge's weight in both path contexts | `realize._edge_cost_pair` | `kp_edge` |
| the min-plus relaxation: the cheapest narrow and wide predecessor per column, the first on a tie | `realize.optimize` | the column loop |

The C planner fuses the offer and the relaxation into one pass. The scratch is the caller's
(`bcir_kp_scratch_size`): little-endian arrays at byte offsets, so it needs no alignment and no
effective type. Every failure zeroes the output.

`bcir_kp_scratch_size` and `bcir_kp_plan` take a view that `bcir_kp_decode_input` accepted. They
re-check one invariant of it, the lane-width count (1..16), because the geometry indexes by it: a
hand-built `bcir_kp_input` with no widths or with more than 16 is refused `BCIR_ERR_PLANNER`
before anything is read. The harness's `--api` mode drives both counts.

## BKPB — the binding, and the native hydrate (CXX4)

BKPI carries a claim's operands as indices into a resource table, and BKPR carries the plan.
What a StreamPack needs beyond them is the **binding**: the RID behind each index, the registry's
generation vector, and the pack's name. With it, `bcir_kp_hydrate` writes the StreamPack that
`streampack_abi.encode(gem.streampack.hydrate(module, result, plan))` writes for the same module
and plan, byte for byte, so the native planner's plan becomes the pack that runs without Python.

```
header (32)  @0 magic "BKPB"  @4 version u16 = 0  @6 flags u16 = 0  @8 n_resources u32
             @12 n_gens u32  @16 topo_gen u32  @20 plan_len u32  @24 reserved u64 = 0
rids         n_resources x u32 -- the RID of each BKPI resource index
gens         n_gens x (rid, map_gen, data_gen) u32 -- the generation vector, RID order
plan         plan_len bytes, UTF-8 -- the pack's source_plan
trailer      crc32 u32
```

`encode_binding(module, plan)` writes it: the RIDs in BKPI's first-reference order
(`resource_rids`), the vector of every resource the module declares, `topo_gen` 1.

Laws (`decode_binding` / `bcir_kp_decode_binding`, in order):

1. the framing laws, as BKPI's; the reserved word → `BCIR_ERR_RESERVED`
2. a count over 2²⁶, or a plan name over 65,535 bytes → `BCIR_ERR_PLANNER`
3. the size is not `36 + 4·n_resources + 12·n_gens + plan_len` → `BCIR_ERR_TRUNCATED` or
   `BCIR_ERR_TRAILING`
4. the plan name is not UTF-8 → `BCIR_ERR_UTF8`
5. the vector's RIDs not strictly ascending → `BCIR_ERR_GENERATION`

`bcir_kp_hydrate` (and `check_binding` with `gem.streampack.hydrate` on the Python rail) then
holds, in order:

6. the binding against the input: another resource count, two indices naming one RID, or a
   resource declared without a generation (or undeclared with one) → `BCIR_ERR_PLANNER`
7. the oracle hydrate's laws over the steps: a step naming no claim or a claim twice, a step whose
   phase is not its claim's, a step out of the canonical phase order, a claim no step plans →
   `BCIR_ERR_PROVENANCE`
8. the encoder's: a width that is not a power of two → `BCIR_ERR_WIDTH`; a negative offset or
   stride (a block field is a u64) → `BCIR_ERR_OVERFLOW`

The pack is the oracle's: v4 with the vector, or the frozen v1 when the module declares no
resource; one segment per step (`seg<n>`, the claim's op, or the realization's name when the op is
empty: `vec<width>`, `scalar`, ...); a `pf<n>` prefetch of the reads for each vector step that
reads; one block and one trace note per step. Every law is checked, and the size counted, before
the first byte is written (`bcir_kp_hydrate_size`). The scratch is the caller's
(`bcir_kp_hydrate_scratch_size`), and every failure zeroes the output.

## Gates

- **Rows**: `tools/perf/gemplus_baseline.py --group kplan`, graded by `tools/c/check_planner.py`
  → `bcir/tests/planner_fixtures.py::measure`. The same function backs `tools/c/check_runtime.sh`
  and the tests.
  - `planner.parity` 8,430 → 0. The corpus is the fixed corpus (the examples, the audit fixture,
    and one coverage module per construct the planner branches on) under every target, Theta and
    policy, the edge scopes, and 240 generated modules under every target. On it, the compact
    planner is the pre-G17 planner (`realize_reference`, kept verbatim), in its plan, its BKPR
    bytes and its ExecutionPlanV1 bytes. The native planner writes the same BKPR bytes, or makes
    the same refusal.
  - `planner.malformed.accepted` 148 → 0: one variant per BKPI and BKPR law (54 + 20), each refused with
    its status on both rails. A test requires every law's message to be reached by some variant.
  - `planner.r9.misjudged` 232 → 0: R9 through the compact offer. The planner's own plan of every
    legal fixed module is accepted. Every forgery of a field a step carries is refused with a
    diagnostic, with the scope and without it.
- **Fuzz:** `fuzz_kplan.c` fuzzes the two decoders over raw bytes, and a BKPI body resealed so a
  mutation reaches the planner. Every plan must decode, cover every claim, and replan to the same
  bytes over dirty, misaligned scratch. The target is registered in `tools/c/fuzz_streampack.sh`,
  and the Python decoders are the decoder campaign's `planner` and `realization` surfaces.
- **Faults:** `tools/testing/faults/planner.json` injects defects into the compact planner, R9,
  the C planner and both codecs. Each defect is caught by its own row.
- **The native hydrate (CXX4)**: `tools/perf/gemplus_baseline.py --group hydrate`, graded by
  `tools/c/check_hydrate.py` → `bcir/tests/hydrate_fixtures.py::measure`, which the `kplan`
  section of `tools/c/check_runtime.sh` also runs at -O2, -O0 and -O3 and against a fault-injected
  build (a v1 pack written as v4).
  - `hydrate.native.parity` 3,375 → 0. The corpus is every case of the planner's corpus whose plan
    the realization record carries, and the hydrate's own cases: each realization name as an
    opcode, generations above zero, v1 packs, modules with no claims, RIDs and generations at the
    top of their range, and plan names of 0, 13 and 65,535 bytes.
  - `hydrate.native.malformed.accepted` 48 → 0: one BKPB variant per law above, one BKPR forgery
    per hydrate law, and the two values a StreamPack cannot carry, on both rails.
  - `hydrate.native.scale4`: the native hydrate's time at 4,096 claims, beside the oracle's.
  - `fuzz_kplan.c` hydrates every plan the planner returns, under a binding consistent with its
    input. The pack must pass the runtime's own reader (`bcir_sp_verify_semantic`) and be
    rewritten identically over dirty scratch, or be refused only as a width or block field no
    StreamPack can carry. A fourth mode fuzzes the binding decoder.
    `tools/testing/faults/hydrate.json` injects 21 defects into both rails.

## Not claimed

- The MLIR `-bcir-plan` pass is unchanged. It reproduces the planner's scores on its frozen
  corpus; this record is not its input.
- The proposal's CXX3 joint solvers (scheduled liveness, bank placement, the exact portfolio)
  remain Python-only.
- No certificate is produced natively. The native planner reproduces the plan; the certificate
  rail reads the Python plan.
- The native wall rows are same-host, single-core, indicative measurements.
- The native hydrate writes the pack `hydrate` writes, not `hydrate_pipelined`'s (no
  double-buffer prefetch, pipeline depth 1), and no G18 delta pack.
