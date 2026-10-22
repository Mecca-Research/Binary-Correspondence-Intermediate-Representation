# Codex's geometric-head research, read against the tree (2026-10-19)

Between 2026-10-13 and 2026-10-18 Codex ran a research program on BCIR's native trainer and wrote
five reports: a training-architecture blueprint (10-13), and four studies of a **global wedge**
token mixer -- a rank-22 Grassmann/Plücker head offered as an alternative to attention in a
525,696-parameter byte decoder (width 128, three layers, tied 260-token embeddings, RMSNorm,
SwiGLU) -- that end with a rerun on merged main after PR #818 (10-17 *Global Wedge Advancement*,
10-18 *Global Integrated Recipe*, 10-18 *Query-Aware Head*, 10-18 *PR818 Geometric Head*). The
reports and their figures live in the requester's "BCIR Research" folder; the five study bundles
were uploaded to main on 2026-10-21 under `docs/research/` (`gp-global`, `wedge-next`,
`wedge-integrated`, `wedge-query-aware`, `wedge-pr818`: 1,370 files, 1.77 GB, each a
self-contained research tree with its frozen protocols, run records and selected checkpoints). This
note reads the reports against the tree: what the tree confirms, what it corrects, which confounds
the comparisons carry, and what the repository adopted -- the slice the last report names,
**coupling per-layer caching with packing-aware buffer planning** (GEO-3, the NDT measure note's
[§42](BCIR_NDT_MEASURE_2026-10-12.md), records in
[measurements/2026-10-19/geo3/](measurements/2026-10-19/geo3/)). Sections 1 to 4 were written before
the bundles were run; section 5 is the rerun (GEO-2, 2026-10-22), and only it states a learning
result as this repository's measurement.

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
alone ([inventory.txt](measurements/2026-10-19/geo3/inventory.txt)). The measure note's §42 times the
three -- attention64 kept, the planner's attention64 inside the envelope, attention32 -- update for
update on the same data, in binary32 at four threads: the planner's attention64 costs x1.158
(1.131-1.188) of an update of attention64 kept, which does not fit, and trains its state bit for
bit; attention32 costs x1.011 (0.977-1.045) and trains another
([scenario-codex-kblock+exp+reduce+attn-t4.txt](measurements/2026-10-19/geo3/scenario-codex-kblock+exp+reduce+attn-t4.txt)).
Holding the trajectory fixed costs about 15% of an update at this shape -- a price a comparison of
heads across the envelope can now choose to pay.

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
`NativeDecoder(recompute="auto", max_memory_bytes=...)`, which choose a recompute set and a
workspace tier inside a budget in the order the measured frontier prices (each set at the need
and then at the panels' need, the floor after every set); and the GEM+ step program's parity with the C
count under every set (the new exact row `train.plan.remat.mismatch`, 0).

Open, in order:

1. **GEO-2's AMX timing.** The rerun itself landed (section 5): every binary32 and bf16 run and
   evaluation of the PR818 study reproduced bit for bit, as uploaded and against current main. What
   it could not do is time the bf16 arms on AMX: the session's host moved to one without it.
2. **The geometric head as a maintained extension** (PR818 §6.6): its C geometry and VJP behind a
   stable runtime interface with its own contract and digests, so tensor-unit work reaches it
   without textual surgery -- and so REMAT and WS-PLAN apply to it as they do to attention.
3. **Write-side binding** (Query-Aware §7.2): a model question -- an explicit key/value write before
   aggregation -- outside the trainer's remit, gated on the reports' own one-pair, four-pair, swap
   and query-change tests.
4. **Causal suffix trimming** behind a causality certificate, as a declared transformation.

## 5. GEO-2: the studies rerun from their bundles (2026-10-22)

GEO-2 reran the last of the five studies -- PR818's: its 51 binary32 runs are the earlier studies'
recipes on merged PR #818, its 15 bf16 runs the new numerical family -- from the uploaded bytes, as
uploaded and against current main, every record held to Codex's field by field with no tolerance.
Records: [measurements/2026-10-22/geo2/](measurements/2026-10-22/geo2/).

**The bundles (GEO-0).** Each bundle's manifest against the bytes main holds
([bundles.json](measurements/2026-10-22/geo2/bundles.json)):

| Bundle | Manifest entries | Byte-identical | Equal but a trailing newline | Absent |
|---|---:|---:|---:|---|
| `wedge-pr818` | 448 | 442 | 3 | `evaluation-protocol.json`, `vendor-new/LICENSE`, `vendor-old/LICENSE` |
| `wedge-query-aware` | 269 | 267 | 2 | none |
| `wedge-integrated` | 205 | 201 | 2 | `protocol.json`, `vendor/LICENSE` |
| `wedge-next` | 326 | 239 | 4 | `protocol.json`, `vendor/LICENSE` and the generated `build/` directory (81 files) |
| `gp-global` | 206 | 205 | 0 | `vendor/LICENSE` |

A trailing newline is what GitHub's web editor adds when it renames a file; the JSON is the same.
Every absent LICENSE is this repository's LICENSE by hash (the work copies restored it from there);
PR818's root `evaluation-protocol.json` is written afresh by its evaluation. The earlier bundles'
gaps -- `wedge-next`'s and `wedge-integrated`'s `protocol.json`, `wedge-next`'s generated `build/`
directory -- touch none of PR818's study, but reproducing those studies themselves would need the
two protocols (they are in the Drive zips).

**The host.** Not the AMX host this note planned on: the session's container moved to an Intel
Xeon (family 6, model 85: AVX-512F and VNNI, no AVX512-BF16, no AMX) @ 2.80 GHz, 4 vCPUs, GCC 13.3.
Its bf16 GEMMs run the AVX-512F instance -- the one Codex's AMD EPYC 9V74 ran -- and the AMX timing
stays open (section 4); the arithmetic does not depend on the instance (NDT-BF16: one result on
every instance).

**As uploaded (GEO-2a).** Codex's own build (`build_upgrade.py`) regenerates all 17 overlay sources
byte for byte, and the frozen protocol's freeze check holds every source it names; the drivers ran
unchanged under the bundle's NumPy 2.3.5 and pyarrow 26.0.0 (a launcher answers the two cgroup-v2
files the protocol's host block reads -- this host mounts v1 -- and that block is discarded under a
frozen protocol). On Intel silicon, against the AMD records:

| | Runs | Updates | Equal to Codex's |
|---|---:|---:|---|
| binary32 training | 51 | 41,400 | 51 |
| binary32 evaluation | 51 | -- | 51: 600 rows, 120 counterfactual groups |
| bf16 training | 15 | 12,600 | 15 |
| bf16 evaluation | 15 | -- | 15: 33 rows, 18 counterfactual groups |

Equal means every field: the selected and final state digests, the input digests, the validation
histories, every update's loss, gradient norm and step, and the budget policy's decisions; for an
evaluation, every row's predictions, labels and metrics and every counterfactual group. The bf16
ordered-recall result the PR818 report calls promising -- 89.06% to 95.31% at T2048 -- reproduces
example for example on another vendor's machine: it is a property of the declared arithmetic, not of
the AMD host. Whether it holds over more seeds and a fresh test set stays the report's own question
(section 3).

**Against current main (GEO-2b).** The overlays are copies of the ABI-3 trainer with the research
head written in; `build_upgrade.py` upgraded them by grafting PR818's blocks at exact anchors. The
port makes the same graft from origin/main (490f5be0; its runtime is 44b16ff7's): main's state
checks (REMAT's recompute-aware arena, WS-PLAN's floor), its workspace functions (the attention model
takes main's exact need; the research heads keep their own group bound, which main's transformer
`ws_need` cannot describe), its GEMM dispatch (WS-PLAN's refusal returns), its attention entry
(BF16-ATTN), the SwiGLU block's forward and backward (BF16-PANELS under WS-PLAN's gate) and its
optimizer and run tail (FACTS-RUN, `run_args`), compiled with main's tensor unit and headers (ABI 5).
Not grafted: the PX-2 program runner, which runs the transformer's launch program -- its records name
the transformer's embedding, attention, loss rows and scatter -- a step the research heads do not
have and the study never calls. Main's products return whether they ran and the research steps
ignore the return, so every refusal is counted where it happens and the host requires the count to be
0 after every call; it stayed 0. Three driver lines change for ABI 5 (the state's trailing
`recompute` word, 0; ABI 5 admitted where they assert 4; the refusal check), and the protocol
carries the port: its eleven changed entries re-hashed, the other 37 held.

| | Runs | Updates | Equal to Codex's |
|---|---:|---:|---|
| binary32 training | 51 | 41,400 | 51 |
| binary32 evaluation | 51 | -- | 51: 600 rows, 120 counterfactual groups |
| bf16 training | 15 | 12,600 | 15 |
| bf16 evaluation | 15 | -- | 15: 33 rows, 18 counterfactual groups |

The policy's decisions are Codex's in every run -- the same variants, micro-batches and accumulation
-- and its byte accounting moves only where main's moved it: the C state record is 8 bytes longer
(ABI 5's `recompute`), and the bf16 attention packs at WS-PLAN's exact need (1,015,872 bytes at
B2/T256, the figure section 2.1 derived, against the conservative 1,442,048), at the same
micro-batches. A first level -- the PR818-grafted overlays on main's tensor unit alone -- was probed
bit for bit (every head under both contracts, six updates each,
[probe.json](measurements/2026-10-22/geo2/probe.json)) and not run in full: level 2 contains it.

So every change main made to the native rail since PR818 -- FACTS-RUN, BF16-ATTN and ATTN-SOFT, GEO-3
and PX-1 to PX-5 -- keeps every bit of Codex's study under both arithmetic families. That is an
independent check, on another program than BCIR's own decoder, of ATTN-SOFT's "every value as it
was" and of WS-PLAN's "the same bits from the floor up".

**Against PyTorch.** [`geo_compare.py`](measurements/2026-10-22/geo2/port/geo_compare.py) ran the
study's three model families at batch 4 on four threads -- the global head (`global`), the
query-aware head (`query_full`) and the attention baseline (`attention`: the bundle's transformer,
width 128, three layers, four heads of 32, RoPE) -- on three routes: B-DIRECT, the level-2 build
through the bundle's own `Native` host; P-EAGER, the bundle's independent Torch equations
(`oracle.torch_model_next`, the reference Codex validated every head against) with
`torch.optim.AdamW` after `clip_grad_norm_`; and P-IND, the same module through `torch.compile`
(Inductor) with the fused AdamW. B-DIRECT ran twice, the second instance an A/A control, and twelve
rounds rotated the arms' order. Each ratio is the geometric mean of the rounds' paired ratios with
its 95% interval; below 1 BCIR is faster. Torch 2.13.0, NumPy 2.2.6
([geo-compare.json](measurements/2026-10-22/geo2/geo-compare.json)). The routes are matched
work, not one arithmetic: the first update's loss agrees to 7.1e-7 and the forward's logits to
3.2e-6, PyTorch's kernels keeping their own summation orders.

Training, one update of 4 x T tokens (forward, backward, clipping, AdamW):

| Model | T | B-DIRECT ms | P-EAGER ms | P-IND ms | B-DIRECT / P-EAGER | B-DIRECT / P-IND | A/A |
|---|---:|---:|---:|---:|---:|---:|---:|
| `global` | 128 | 20.5 | 148.8 | 60.7 | 0.13 [0.13, 0.14] | 0.33 [0.30, 0.35] | 1.00 [0.92, 1.09] |
| `global` | 256 | 36.7 | 203.0 | 82.6 | 0.18 [0.17, 0.19] | 0.43 [0.40, 0.47] | 1.01 [0.95, 1.07] |
| `global` | 512 | 71.2 | 276.3 | 123.9 | 0.24 [0.22, 0.25] | 0.54 [0.49, 0.59] | 0.98 [0.93, 1.04] |
| `query_full` | 128 | 21.6 | 165.5 | 70.8 | 0.13 [0.12, 0.15] | 0.31 [0.29, 0.33] | 1.04 [0.97, 1.12] |
| `query_full` | 256 | 38.8 | 208.9 | 97.4 | 0.19 [0.17, 0.20] | 0.38 [0.36, 0.40] | 1.00 [0.94, 1.06] |
| `query_full` | 512 | 77.6 | 354.0 | 159.7 | 0.22 [0.20, 0.23] | 0.48 [0.45, 0.51] | 0.92 [0.86, 0.99] |
| `attention` | 128 | 16.5 | 86.8 | 52.6 | 0.19 [0.18, 0.20] | 0.31 [0.28, 0.35] | 1.06 [0.97, 1.15] |
| `attention` | 256 | 33.0 | 110.1 | 64.8 | 0.29 [0.27, 0.31] | 0.49 [0.46, 0.53] | 1.04 [0.97, 1.12] |
| `attention` | 512 | 75.3 | 152.1 | 144.0 | 0.48 [0.45, 0.51] | 0.50 [0.47, 0.53] | 0.97 [0.90, 1.06] |

Inference, one forward of 4 x T tokens:

| Model | T | B-DIRECT ms | P-EAGER ms | P-IND ms | B-DIRECT / P-EAGER | B-DIRECT / P-IND | A/A |
|---|---:|---:|---:|---:|---:|---:|---:|
| `global` | 1024 | 38.2 | 131.0 | 137.8 | 0.30 [0.26, 0.34] | 0.28 [0.25, 0.31] | 1.01 [0.96, 1.06] |
| `global` | 2048 | 76.7 | 249.8 | 252.0 | 0.31 [0.28, 0.34] | 0.30 [0.27, 0.33] | 1.01 [0.97, 1.06] |
| `query_full` | 1024 | 38.6 | 158.7 | 175.4 | 0.25 [0.21, 0.29] | 0.24 [0.20, 0.27] | 1.02 [0.94, 1.11] |
| `query_full` | 2048 | 85.0 | 292.4 | 285.0 | 0.29 [0.26, 0.32] | 0.29 [0.27, 0.32] | 1.01 [0.95, 1.07] |
| `attention` | 1024 | 78.9 | 73.4 | 81.0 | 1.07 [0.98, 1.16] | 1.01 [0.93, 1.09] | 1.03 [0.94, 1.12] |
| `attention` | 2048 | 267.0 | 161.9 | 181.0 | 1.74 [1.52, 1.98] | 1.59 [1.36, 1.87] | 1.00 [0.96, 1.04] |

BCIR trains every model faster: 2.1 to 7.5 times P-EAGER and 1.9 to 3.3 times P-IND, the margin
narrowing as the context grows. It runs both geometric heads' inference 3.2 to 4.2 times faster
than either route. The A/A control's interval holds 1 in every case but `query_full` training at
T512 (0.92 [0.86, 0.99]): this host's noise, so a ratio inside about 8% of 1 is not resolved.

The one loss is the attention baseline's inference at long context: at parity at T1024
(1.07 [0.98, 1.16] against P-EAGER), at T2048 1.74 times slower than P-EAGER and 1.59 times slower
than P-IND. The cause is the attention's probabilities. BCIR's attention writes every head's t x t
probabilities to the arena, because the backward reads them, and the forward writes them even when
no backward follows. At B4/T2048 that is 268 MB a layer (the attention model's arena is 1,061 MB,
the global head's 225 MB), written, read by the softmax's passes and read again by the context
product. PyTorch's `scaled_dot_product_attention` -- the oracle's call -- runs a flash-style kernel on this
CPU (`_scaled_dot_product_flash_attention_for_cpu`, the operator the decoder-scale profile names)
that never forms them: a block's scores stay in cache. That sets the next inference update: an attention
forward that keeps no probabilities when no backward reads them, each value as it was.
The same comparison at decoder scale (OT-10: the 9.84M-parameter decoder, 300 updates on four
threads) and against Triton-CPU's kernels (OT-05) is recorded in the
[OT/BG ledger](BCIR_OT_BG_LEDGER.md).
