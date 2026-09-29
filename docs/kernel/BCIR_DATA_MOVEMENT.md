# Data movement as a first-class transformation (GEM+ G8)

A claim that reads one memory tier and writes another names two banks, and nothing in the plan
says how the bytes cross between them. Before G8, four example programs did exactly that: nine
of their claims read RAM and HBM at once (the D-R2 violations `device_manifest.check_bank_moves`
reports), the plan's movement-edge family was declared by G11 but always empty, and the only
transfer-aware rule the repository had (`channels.orchestrate`'s site choice) decided where each
claim runs first and priced the transfers afterwards. The roadmap's G8 asks for the opposite:
movement chosen **jointly** with compute, and Semantic Swap kept explicit and correctness-neutral
(GEM+ roadmap G8, staged plan S5-C).

G8 does not add a second scheduler or a second pricer. It chooses, for every claim, the bank its
operands are read and written in -- its **site** -- and **transforms** the goal graph M into M′:
M plus explicit `mem.move.*` claims over per-(resource, bank, episode) copy resources. M′ is an
ordinary module. `optimize` realizes it, `schedule_plan` places it, `plan_static_memory` gives
every copy its own address, and `plan_from_realization` mints the plan, whose movement family
now carries one edge per move. The laws MV1–MV11 then prove, from (M, spec, M′, plan) alone, that
the transform is correctness-neutral and that the plan describes it faithfully.

| Module | Holds |
|---|---|
| [`bcir/kbcir/movement.py`](../../bcir/kbcir/movement.py) | the spec (`MovementSpec`), the transform (`transform`, `_Builder`), the joint planner (`plan_movement`, `price`), the plan producer (`execution_plan_of`, `movement_edges`) and the laws (`verify_movement`, `races`, `replay_safe`, the certificates) |
| [`bcir/verify/__init__.py`](../../bcir/verify/__init__.py) | `verify_execution_plan(..., movement=(source, spec))`: the movement laws beside R9/R11/R13, and MV11 when a plan that moves data arrives without its context |
| [`bcir/abi/execution_plan_abi.py`](../../bcir/abi/execution_plan_abi.py), [`runtime/c/bcir_execution_plan.h`](../../runtime/c/bcir_execution_plan.h) | the v3 wire on both rails: the move tail, the binding trailer and their laws |
| [`bcir/asn1/execution_plan.py`](../../bcir/asn1/execution_plan.py) | the `BCIR-ExecutionPlan` projection, version 3, held to the native wire laws |
| [`bcir/tests/movement_fixtures.py`](../../bcir/tests/movement_fixtures.py), [`tools/perf/check_movement.py`](../../tools/perf/check_movement.py) | the corpus, the law variants, the rows and the gate |

## The spec

`MovementSpec(hardware, homes, classes, codecs, sites, deadline, remat, max_hops)` is everything
the planner may assume and nothing it may not:

- **`hardware`**: the banks (domain, channel, allocatable bytes, alignment) and the **directed**
  links between them. A route is the fewest hops over declared links, then the least summed
  latency, then the names; a route of more than two banks is a **staged** move, one claim per hop.
- **`homes`**: where each resource lives. By default, the one bank of its domain; a domain with
  several banks needs a declared home, and a resource without one is refused.
- **`classes`**: `immutable`, `recomputable` or `mutable` -- **mutable by default**, the
  conservative class.
- **`codecs`**: the lossy transfer codecs a resource admits, as bits per element.
- **`sites`**: optional per-claim site restrictions; `deadline`: a makespan budget in plan ticks.

The spec has a canonical digest, and the plan carries it (below).

## The transform M → M′

The builder walks M's claims in execution order -- topological phase order, and within a phase
ascending claim id, which is the order the canonical scheduler dispatches them in. It tracks, per
logical resource, the version a write last produced, and per (resource, bank) the **episode** that
holds a copy and the version in it. For a claim at site s:

1. Every operand it reads -- and every operand it writes only partly -- needs a fresh copy at s.
   A copy already at s holding the current version is reused. Otherwise a new episode opens,
   filled by a move from the cheapest bank holding the version (a staged route is one move per
   hop), by a **remat**, or by a **compressed** move.
2. The claim is emitted with its operands remapped to the copies at s. It keeps its id, phase,
   opcode and every other field. It declares its own domain while it still touches it; otherwise
   it declares its site's domain, which is the minimal change R3 needs.
3. A write makes the copy at s hold the next version and every other copy of the resource stale.

**Why moves get their own phases.** The scheduler orders a phase's claims by ascending id, so a
move cannot sit between two claims of one phase. Moves and remats therefore go into an **inbound**
phase `in(p)` that runs before the consumer's phase p. The inbound phase depends on the phases
that last wrote its sources, and on the eviction that freed its room when it reloads. Forced
evictions' writebacks go into an **outbound** phase `out(q)` after the copy's last use, spliced
into every dependence chain q was in. Final writebacks go into one closing phase that depends on
all others. As a consequence, a claim that reads, in another bank, a version its own phase
produced cannot be served: the transform refuses it
(`test_a_version_produced_in_the_same_phase_cannot_cross_banks`).

**Isolated resources** (a device's registers) are bound in place and never move. A claim that
touches one is pinned to its bank.

A transform that moves nothing is M itself, and its plan is the parent's plan byte for byte:
v1, with no binding (`movement.identity.drift`).

## Semantic Swap

Each class admits exactly the swaps that cannot change what a program computes:

- **immutable** -- no claim writes it (a claim that does is refused). A copy is always clean, so
  it may be dropped and reloaded freely.
- **recomputable** -- a temporary that is not live-out, so it needs no writeback. A dropped copy
  may be **rematerialized** by replaying its producer at the site, but only when `replay_safe`
  certifies the producer as replayable: no control opcode, no atomic or ordered access, no
  device, no device operand, no in-place update, no move, and a static count. The replay must also
  read, at the site, exactly the versions the producer read. The edge carries a **replay
  certificate**: a digest of the producer, its inputs and versions, the resource and the site.
- **mutable** -- its home must hold its final version when the plan ends. A copy updated away from
  home is written back by a `coherence = writeback` edge carrying that version (generation-checked
  writeback). A writeback is never compressed and never a remat, on any wire version.

**Approximation** needs a certificate. A resource may cross a lossy codec only if the spec
declares it, and only if every claim that reads the compressed copy declares a tolerance
R17's `meets_tolerance` admits at the codec's bits (`codec_admits`). The edge carries an
**accuracy certificate**: a digest of the resource, the bits and the readers it was granted to.
A compressed copy is never moved onward.

**Deadlines and backpressure.** Capacity is priced per phase as the aligned bytes each bank holds.
When a bank overflows, the planner ends the episode that stays idle across the tightest phase and
whose next use is farthest (Belady). It reloads that episode later, and the static memory plan
then gives every copy its address. The **anti-thrash law** (MV9) makes every reload of a version a
bank already held a forced one: keeping the earlier copy across the gap must have overflowed the
bank at some phase of it. A plan past `spec.deadline` is refused (MV10).

## The planner

The objective is the repository's own M(π, Θ): a candidate's key is (scheduled makespan of M′,
serial score, number of sites that moved, remats, compressions), and it is priced by `optimize` and
`price_scheduled` over M′. A candidate that cannot be scheduled or laid out is infeasible, not
cheap.

- **`exact`** enumerates every site choice, remat subset and compression subset. It runs when the
  space fits the budget (`DEFAULT_BUDGET` = 4,096 evaluations), and its answer is the optimum of
  the joint objective: **TMSAO-1**. Every fixture of the corpus is planned this way.
- **`greedy`** runs local search from the best baseline, within the same budget. It makes no
  optimality claim: **TMSAO-4**. `method="exact"` over budget is an error, never a silent
  downgrade.
- **Baselines**, priced through the same transform so that each one is legal:
  - `home` -- every claim at its first operand's home;
  - `sequential` -- sites chosen for compute alone, with movement priced after;
  - `greedy` -- `channels.orchestrate`'s rule applied as a site choice.

  `movement.excess` reads its RED from `greedy`, the best rule the parent had.

## The laws (`verify_movement`)

The copy relation is derived positionally: from an original claim's operands, from a move's edge
and from a remat's producer. The banks are read from the plan's lifetimes, and the versions are
recomputed by replaying both modules. None of the planner's bookkeeping is trusted.

| Law | Holds |
|---|---|
| MV1 declared | Every resource of M′ is a logical resource of M, unchanged, or a copy of one. A copy has its logical's shape, is in a declared bank of its own domain, and is never in an isolated domain. Every operand M′ names is declared. |
| MV2 preserved | Every claim of M survives with its id, phase and fields, and its operands are remapped to copies at **one** site (D-R2 clean, and inside its allowed sites). M's phases keep their dependences. |
| MV3 explicit | Every other claim is a move or a remat. A move is spelled `mem.move.{far\|near}:src->dst`, has one read and one write, crosses one declared link, and never moves an isolated resource. |
| MV4 versions | Every original claim reads, through its copy, exactly the version it reads in M, and no two accesses of one resource race. |
| MV5 writeback | Every mutable resource's home ends at its final version, landed by a writeback edge. |
| MV6 classes | No claim writes an immutable resource. |
| MV7 remat | A remat replays a replay-safe producer that produced exactly that version, reading the versions it read; its certificate re-derives. |
| MV8 accuracy | A compressed edge uses a declared codec; every reader tolerates it; the copy feeds only original claims; the certificate re-derives. |
| MV9 capacity | Every lifetime fits its bank, and every reload was forced (anti-thrash). |
| MV10 deadline | The makespan is within `spec.deadline`. |
| MV11 binding | The plan names M and the spec (`source_hash`, `spec_hash`). It has one edge per move and remat, and each edge's route, size, generations, version, kind and window (the source's writer, the destination's first reader) are the claim's. A plan that moves data and arrives without its context is refused. |

`verify_movement` is **total**. A transform that names an undeclared operand, an edge that names
no source resource, an edge that moves a device register, a copy declared in an isolated domain, a
lifetime in an undeclared bank or the wrong domain, a spec with no home for a resource, or a remat
of a producer that reads an unplaced register is each answered by its law, never by a traceback.
Before this was fixed, four of these crashed the verifier. A copy re-declared in the MMIO domain
passed every law, because the isolation exemption meant for device registers also covered copies.

## The plan as bytes: v3

v3 is append-only. Encoders emit the lowest version that carries the plan, so a plan that moves
nothing is never v3, and every v1/v2 plan is byte-identical to the parent's.

```
move record tail   claim:u64 version:u32 flags:u8 producer:u64 bits:u8 cert:u64
binding trailer    source_hash:u64 spec_hash:u64          (after the generation vector)
flags              HAS_AFTER=1  HAS_BEFORE=2  HAS_PRODUCER=4  HAS_CLAIM=8
```

The flags settle the "claim 0 is a legal claim" ambiguity of v1/v2, where 0 meant "no claim".
The laws are the same predicate on both rails (`validate_plan` / `bcir_ep_verify`), and the
**wire version decides** which laws apply: a v3 wire whose binding and tails are all zero is
refused, not read as the v2 plan it re-encodes to.

- **Every version:** src ≠ dst unless the edge is a remat; a writeback is neither compressed
  nor a remat.
- **v3 only:**
  - the flags stay within the defined set, and a reference the flags call absent is zero;
  - exactly a remat names a producer, and never itself;
  - exactly a compressed edge names 1..31 codec bits;
  - exactly remat and compressed edges carry a certificate;
  - every referenced claim is a step;
  - the window is ordered by the placement;
  - a writeback lands in the bank of its resource's lifetime;
  - both hashes are nonzero.

Every reader of the plan reads v3:

- **The C twin**: decode, walk, the binding (`bcir_ep_binding`), the generation vector and the
  plan/pack binding. The last two locate the vector through one predicate that skips the
  trailer; the pack check had located it at the body's end and called a matching v3 pack stale.
- **The BCAB reader**, on both rails.
- **The control plane's `admit_plan`**, on both rails.
- **The `BCIR-ExecutionPlan` ASN.1 projection**, version 3. Its DER, OER and JER decoders now
  admit exactly the plans the native codec admits.
- **The decoder campaign's new `plan` surface.**
- **The decoder fuzzer**, which is seeded with v1, v2 and v3 plans and repairs each mutant's CRC,
  so mutation reaches the walk.

## What is not claimed

- **The planner is exact only within its budget.** The corpus is sized to fit it. A larger module
  gets the greedy search and a TMSAO-4 label, and there is no lower bound for movement yet.
- **Far and near moves are priced alike.** A move is priced by the K_BCIR arithmetic every rail
  shares; making the price distance-aware is a four-rail change (the Python realize, the native
  planner, the C R9 re-derivation, the MLIR optimizer). It is recorded as a finding, not made
  here. `device_manifest`'s docstring claims distance-aware pricing that `realize` does not do.
- **The transform, the planner and the movement laws are oracle-only.** The C twin holds the v3
  wire laws on bytes, not MV1–MV11, which need the source module and the spec.
- **No measurement on hardware.** The rows are exact counts over the plan model; nothing here was
  timed on a device with the banks the fixtures declare.

## Evidence

The rows are in group `movement` (`tools/perf/gemplus_baseline.py --group movement`). They are
counted by `movement_fixtures.measure`, which the tests and the gate share. RED is the parent,
2221db5e, judged by the same fixtures and, where it had one, by its own verifier, codec, ASN.1
projection and C twin.

| Row | RED | GREEN |
|---|---|---|
| `movement.implicit_cross_tier` (claims of the example programs) | 9 | 0 |
| `movement.excess` (ticks over the joint optimum, 11 fixtures) | 597,840 | 0 |
| `movement.laws.accepted` (30 variants) | 0 (vacuous: see below) | 0 |
| `movement.laws.misattributed` | 30 | 0 |
| `movement.identity.drift` | 0 | 0 |
| `movement.roundtrip.mismatch` | 9 | 0 |
| `movement.plans.unlawful` | 9 | 0 |
| `movement.asn1.accepted` (malformed documents) | 18 | 0 of 87 |
| `movement.parity.mismatch` (needs a C compiler) | 11 | 0 |

Per fixture, in plan ticks (x86_avx512, `Theta()`), the planner's answer against the three rules it
is compared with. `sequential` and `greedy` are the pre-G8 rules: they choose sites first and price
movement after, and they agree on every fixture here. `home` keeps every claim at its first
operand's home; "--" means that choice cannot be laid out or breaks a site restriction.

| Fixture | Space | Optimum (TMSAO-1) | home | sequential / greedy | What the optimum does |
|---|---|---|---|---|---|
| tiled_matmul | 2 | 524,288 | 843,776 | 524,288 | moves the A/B tiles to HBM |
| matmul_tiled | 256 | 309,248 | 337,920 | 309,248 | eight moves |
| reduce6 | 64 | 491,520 | 491,520 | 507,964 | stays home; the pre-G8 rules send all six reductions to HBM |
| amortize | 16 | 335,872 | 335,872 | 581,632 | stays home (the identity) |
| pingpong | 4 | 336,376 | 336,376 | 459,020 | moves the small array, not the large one |
| stream | 1 | 842,752 | -- | 842,752 | six fetches, two forced evictions |
| remat | 2 | 527,360 | -- | 614,400 | rematerializes the temporary in HBM |
| compressed | 2 | 103,424 | -- | 229,376 | an 8-bit transfer to a tolerant reader |
| staged | 1 | 155,136 | -- | 155,136 | NVM → HBM staged through RAM |
| writeback | 4 | 46,720 | 62,464 | 46,720 | updates in HBM, writes back home |
| device | 2 | 28,692 | 42,004 | 28,692 | the register pinned; the buffer moves |

The pre-G8 rules are 597,840 ticks over the optimum in total, 16.2% of it, spread over five
fixtures. The greedy search (`method="greedy"`, TMSAO-4) reaches the optimum on all eleven, but it
claims nothing.

The parent accepted no law variant, but its refusals were vacuous. Its R9 lifetime cover compared
phase ids with topological phase positions, so it refused every plan that moves data, including
the nine legal ones the planner mints. Every refusal of a variant came from R9 or R13, never from
the law the variant breaks. The defect is fixed here: the cover reads positions, and
`test_the_lifetime_cover_law_reads_phase_positions_not_ids` witnesses it with a module whose
phase ids run against their order.

- **The optimum is re-derived.** `movement_fixtures.joint_optimum` enumerates every fixture's space
  on its own, without the planner, and prices every candidate. `movement.excess` and the tests
  measure the planner against it, never against the planner's own answer.
- **Every plan is lawful.** Every plan the planner prices, its baselines included, satisfies
  every movement law, R1–R25, D-R2 and the race law. It round-trips v3 on both rails, and the
  C twin binds it to its pack and live registry.
- **Faults.** [`tools/testing/faults/movement.json`](../../tools/testing/faults/movement.json)
  injects 51 defects into the planner, the transform, the producer, every movement law, the
  verifier's R9, the Python codec, the ASN.1 projection and the C twin, and each is caught by the
  row it names.
