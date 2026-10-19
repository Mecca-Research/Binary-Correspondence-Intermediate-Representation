# Codex's geometric-head research, read against the tree (2026-10-19)

Between 2026-10-13 and 2026-10-18 Codex ran a research program on BCIR's native trainer and wrote
five reports: a training-architecture blueprint (10-13), and four studies of a **global wedge**
token mixer -- a rank-22 Grassmann/Plücker head offered as an alternative to attention in a
525,696-parameter byte decoder (width 128, three layers, tied 260-token embeddings, RMSNorm,
SwiGLU) -- that end with a rerun on merged main after PR #818 (10-17 *Global Wedge Advancement*,
10-18 *Global Integrated Recipe*, 10-18 *Query-Aware Head*, 10-18 *PR818 Geometric Head*). The
reports, their figures and five study bundles (256-410 MB each) live in the requester's "BCIR
Research" folder, not in this repository. This note reads the reports against the tree: what the
tree confirms, what it corrects, which confounds the comparisons carry, and what the repository
adopted -- the slice the last report names, **coupling per-layer caching with packing-aware buffer
planning** (GEO-3, the NDT measure note's [§42](BCIR_NDT_MEASURE_2026-10-12.md), records in
[measurements/2026-10-19/geo3/](measurements/2026-10-19/geo3/)). The bundles were not re-run here:
their training runs, checkpoints and protocols are the next step (GEO-2), and nothing below
restates one of their learning results as this repository's measurement.

## 1. The research, in brief

The head (10-17 §2): a layer's normalized input `h` is reduced to `z = W_reduce h` (width 128 to
22), its strictly earlier prefix averaged, `v_t = sum_{j<t} z_j / sqrt(max(t,1))`, and the plane
`z_t ^ v_t` normalized into 231 packed coordinates `p_t,ij = (z_t,i v_t,j - z_t,j v_t,i) /
max(||z_t ^ v_t||, 1e-6)`, read out by a dense 231-to-128 projection under a learned gate. The
query-aware variant (10-18) adds an interior product `c_j = sum_i q_i P_ij` with `q = u/||u||`
and reads `g = W_p p + W_c c`. The work ran on AMD EPYC 9V74 virtual machines (AVX-512F and
AVX512-BF16, no AMX), with the native tensor unit pinned to `aaa2e6c` and later to merged main
`2376511`, the geometric kernels as a research overlay outside the tree.

| Claim | Where | Strength as reported |
|---|---|---|
| Whole-layer, frame and FF reconstruction take the head's activation/scratch arena from 30.875 to 10.703 MiB at B1/T1024 with identical logits, gradients and states | 10-17 §3 | exact (bitwise checks at small shapes) |
| Variable-length training lifts global T1024 recall 78.91% -> 94.01%; physical suffix trimming makes it 1.60x faster | 10-18 Integrated §2, §4 | three seeds; trimming not bitwise (3.6e-8 relative gradient) |
| At a 32 MiB controlled-buffer budget the global head reserves 11% less than "attention32", attention at micro-batch 2 x accumulation 2 (26.09 vs 29.30 MiB) | 10-18 Integrated §4 | measured storage, similar time |
| The query-aware contraction lifts ordered recall at T2048 to 89.06% (69.27% base, 93.23% attention32) and leaves key/value association at chance | 10-18 Query-Aware §4-5 | three seeds, 64 examples a model at T2048 |
| All 51 FP32 runs reproduce bitwise on merged main; the isolated AdamW update 1.334x; attention-FMA 1.077-1.079x at T256/T512; huge-page advice no win | 10-18 PR818 §3-4 | exact replay; paired timing, one host |
| bf16 GEMMs lift the query head's T2048 recall 89.06% -> 95.31% | 10-18 PR818 §5 | three seeds (two up, one unchanged), 64 examples each |
| The decoder admits "a conservative sum of whole-call packs"; a tighter safe packing bound could make a mixed cached/recomputed plan fit 32 MiB under bf16 (a 286,072-byte shortfall) | 10-18 PR818 §6.3 | inventory estimate, not run |

The reports are careful with their own limits -- controlled storage is not RSS, a timing interval
describes one host, a selector that picks the fastest calibrated plan is "best measured" rather
than optimal, a modeled HAM transfer is a prediction -- and this note keeps those labels.

## 2. What the tree confirms, and what it corrects

### 2.1 The packing figures: the plan, the probe and the exact need

PR818 §4's table reports, for the attention model at B4/T256 under bf16, **1,933,376 bytes
observed against 2,752,768 admitted**. Both figures are the parent commit's (`12aa6dc`,
[parent.txt](measurements/2026-10-19/geo3/parent.txt)): `bcir_decoder_workspace_bytes` summed
every call of a group as if each packed alone, and the harness's probe recorded what each group
packs by the tensor unit's own sharing rule -- but without the panels the SwiGLU block hands its
GEMMs, so a panel operand (a NULL address in the call) read as one operand two calls share.

| Shape (bf16) | Parent's plan | Parent's probe | Exact need, panels | Exact need, no panels | Floor |
|---|---:|---:|---:|---:|---:|
| Codex attention, B4/T256 | 2,752,768 | 1,933,376 | **1,933,376** | 2,490,432 | 131,136 |
| Codex attention, B2/T256 | 1,442,048 | 1,015,872 | **1,015,872** | 1,310,784 | 65,600 |
| 9.84M decoder, B8/T128 | 11,534,592 | 6,553,664 | **4,522,048** | 10,747,968 | 131,136 |

([inventory.txt](measurements/2026-10-19/geo3/inventory.txt),
[inventory-nopanels.txt](measurements/2026-10-19/geo3/inventory-nopanels.txt).) So the
"observed" figure is the exact need at the attention shape by coincidence of which group is the
largest, and **at the 9.84M decoder it is neither**: 2,031,616 bytes above what the step packs
with its panels and 4,194,304 below what it packs without them (a state whose provider replaces
the GEMM, or whose slots are not 64-byte aligned). An observed high-water mark from that probe is
therefore not a bound on either path, which is the report's own caution ("observed high-water
marks alone are not universal bounds") made concrete. WS-PLAN replaces both figures with the
tensor unit's rule applied to the step's groups as the trainer issues them, panels named
(`bcir_decoder_workspace_need`), and the probe now records that, held equal to it.

The report's next claim is right and is now the tree's: below the need, a group runs its calls one
at a time and a call in strips, to the same bits, down to a floor of one 32-row strip of A and one
32-column strip of B at the step's deepest call. The parent admitted nothing below its plan figure;
the trainer now admits any workspace from the floor up and fails the step with
`BCIR_DT_CAPACITY` -- nothing committed -- where a product refuses, instead of ignoring the GEMM's
return (the parent was safe only because its admission demanded the whole conservative sum).

### 2.2 The matched-storage attention reference changes the training it matches

Every 32 MiB study arm of the reports compares the global head at the full micro-batch with
**attention32**: attention at micro-batch 2 with two accumulated micro-batches an update, the
largest that fit the envelope (the Integrated and Query-Aware reports say so; its 29.30 MiB is this
tree's arena for B2, 30,709,760 bytes = 29.29 MiB, to 0.01 MiB). Accumulation is a different
reduction order, so attention32 is not attention64 in less memory: the Integrated report measures
the first update's weights 2.94e-9 apart and the trained models 0.11-0.13 relative L2 apart after
900 updates, with T2048 recall 89.58% against attention64's 100%, and it rightly declines to blame
accumulation from three seeds. The comparison between the global head and its matched-storage
reference therefore mixes a storage change with a trajectory change.

GEO-3 removes the mix. With every layer recomputed (REMAT), attention at the full micro-batch
needs **30,988,288 bytes (29.55 MiB) in binary32 -- 0.27 MiB more than attention32 -- and trains
the attention64 state bit for bit**; under the bf16 GEMMs with the fused attention it needs
32,921,664 bytes (31.40 MiB) at the exact workspace and would need 33,741,056 (32.18 MiB, over the
envelope) at the parent's plan figure, so the fit is the coupling of the two slices, not either
alone ([inventory.txt](measurements/2026-10-19/geo3/inventory.txt)). The measure note's §42 times
the three -- attention64 kept, the planner's attention64 inside the envelope, attention32 -- update
for update on the same data.

### 2.3 What transfers from the AMD hosts and what does not

The bf16 GEMM contract's instances -- portable, AVX-512 and AMX -- compute one result (NDT-BF16,
held by an independent `__float128` reference), so the GEMMs of PR818's bf16 learning runs, which
ran the AVX-512F instance, are the AMX instance's bit for bit: given the same build of the overlay,
its bf16 recall and loss figures are the AMX host's too. Their timing is not: the AVX-512F instance
emulates the tile unit's arithmetic with binary32 FMAs (PR818 §4 measures it 0.877x of binary32 on
the query head), which is why the report's bf16 slowdown says nothing about the AMX path. The
binary32 regression figures (51 runs bitwise on merged main) are consistent with every contract's
thread-count and instance identity in this tree; the isolated AdamW gain (1.334x at 534,144
parameters) is NDT-UPDATE's shared division at another shape (the measure note's §27: the update
15.33 -> 10.06 ms a step at four threads on the 9.84M decoder, x1.52); the attention-FMA gain at T256/T512 is ATTN-FMA's direction (§33); huge-page advice not
winning at the head's T512 is consistent with NDT-PAGES' gain having been measured at the 9.84M
decoder's working set -- PR818 §4 correctly separates accepted advice from pages observed.

### 2.4 Storage accounting

The reports' "controlled-buffer" envelope counts the four parameter-sized arenas, the activation
and scratch arena, the pool's control block, staged inputs and named C records, and excludes code,
stacks, Python objects and allocator retention; it is not RSS, as they say. This tree's figures
count the arenas the trainer is lent (four parameter arenas, the activation arena at
`bcir_decoder_activation_count` floats, scratch and the bf16 workspace) -- the budget
`NativeDecoder(max_memory_bytes=...)` enforces -- and they agree with the reports where both
measured the same thing: attention at B4/T256 kept, 53,008,384 bytes = 50.55 MiB against the
report's 50.56; at B2, 29.29 against 29.30.

## 3. Method notes

- **Small held-out sets.** T2048 points rest on 32-64 examples a seed; one example is 1.6-3.1
  points. The bf16 T2048 gain (+6.25 points) is four examples a model on average, two seeds up and
  one unchanged; the reports call it promising, not established, and so does this note.
- **Three seeds, frozen protocols.** The protocols were frozen before training and checkpoints were
  selected on validation, never test -- the right design; three seeds bound what any difference of
  a few points can mean.
- **Trimming is a different state here too.** The 1.60x suffix trimming changes the rows a GEMM
  sees, and every partition in this trainer is the shape's, so in BCIR it would be a declared
  transformation with its own digest -- admissible only under a certificate that the mixer is
  causal and the suffix unsupervised, as the Integrated report's blueprint row says.
- **The calibrated selector.** The Integrated report's policy picks the fastest calibrated plan per
  budget and finds it unreliable on a shared host (it loses to a fixed heuristic at some budgets).
  GEO-3's planner takes the other road: a declared order backed by a measured frontier, no
  calibration, TMSAO-4 -- see §42.
- **Modeled transfers.** The HAM fast-bank scenario (10-17 §4) is a planner prediction with no
  transfer executed; the exact-layout result (3,682,304 bytes, `stop_reason="optimal"`) is optimal
  for that fixture's lifetimes only. Both are labeled so in the report and are not re-derived here.
- **Stronger baselines.** Both the 10-17 and 10-18 reports note that IO-aware exact attention
  (FlashAttention) stores no quadratic matrix and is the fairer memory baseline; this tree's
  attention still materializes its probabilities (tiled attention, NDT-ATTN, is the training
  roadmap's next item).

## 4. What the tree adopted, and what stays open

Adopted (GEO-3, landed 2026-10-19 with ABI 5): **WS-PLAN** -- the exact bf16 workspace need, the
floor, admission from the floor up, every product's refusal a `BCIR_DT_CAPACITY` step that commits
nothing, the bf16 attention's regions one a worker of the pool the state names; **REMAT** -- a
recompute set (`bcir_decoder_state.recompute`, bit l for layer l) whose layers share one cache slot
after the kept ones, the highest recomputed layer free, the re-run stopping at the hidden
activation, every set the all-kept state bit for bit; **MEM-PLAN** -- `memory_plan` and
`NativeDecoder(recompute="auto", max_memory_bytes=...)`, which choose a workspace tier and a
recompute set inside a budget in a declared order; and the GEM+ step program's parity with the C
count under every set (the new exact row `train.plan.remat.mismatch`, 0).

Open, in order:

1. **GEO-2: rerun the studies on this AMX host** from their bundles (the binary32 regression first,
   then the bf16 arms, then the paired timing), against current main.
2. **The geometric head as a maintained extension** (PR818 §6.6): its C geometry and VJP behind a
   stable runtime interface with its own contract and digests, so tensor-unit work reaches it
   without textual surgery -- and so REMAT and WS-PLAN apply to it as they do to attention.
3. **Write-side binding** (Query-Aware §7.2): a model question -- an explicit key/value write before
   aggregation -- outside the trainer's remit, gated on the reports' own one-pair, four-pair, swap
   and query-change tests.
4. **Causal suffix trimming** behind a causality certificate, as a declared transformation.
