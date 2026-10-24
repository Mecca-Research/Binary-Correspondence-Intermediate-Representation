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
   it could not do is time the bf16 arms on AMX: the session's host moved to one without it. Timed
   since, on the AMX host the session moved to next (GEO-AMX, section 19).
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

**Correction (COMPILE-LIMIT, 2026-10-24: section 18).** The P-IND column is not Inductor's in seven of these
rows. `geo_compare.py` compiled all fifteen cases in one process through one forward code object: Dynamo compiled
`global` T128 at its own shapes, the next seven cases with dimensions made dynamic, and from the ninth -- `attention`
T512's training and all six inference rows -- nothing: past its recompile limit those rows ran the module eagerly,
so their P-IND is eager PyTorch's time. Rerun with every case compiled at its own fixed shapes, on the session's later
host (section 18's table), BCIR trains the three models in x0.33-0.83 of Inductor's time (`attention` T512 x0.83,
where this table reads 0.50) and runs the geometric heads' inference in x0.77-0.90 of it -- 1.1 to 1.3 times
Inductor's speed, not 3.2 to 4.2; the attention baseline's inference took x1.39 of Inductor's time at T1024, not
parity, and x1.67 at T2048. Every P-EAGER reading stands.

BCIR trains every model faster: 2.1 to 7.5 times P-EAGER, the margin narrowing as the context grows, and runs both
geometric heads' inference 3.2 to 4.2 times faster than P-EAGER (against Inductor, the correction above). The A/A
control's interval holds 1 in every case but `query_full` training at T512 (0.92 [0.86, 0.99]): this host's noise,
so a ratio inside about 8% of 1 is not resolved.

The one loss is the attention baseline's inference at long context: at parity with P-EAGER at T1024
(1.07 [0.98, 1.16]) and 1.74 times its time at T2048 -- against Inductor at fixed shapes x1.39 and x1.67 (the
correction above). The cause is the attention's probabilities. BCIR's attention writes every head's t x t
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

## 6. INFER-ATTN: the inference update the comparison named (2026-10-22)

Section 5's one loss was the attention model's inference at long context, where BCIR wrote every
head's t x t probabilities that no backward would read. INFER-ATTN is the forward without them. Under
`BCIR_TENSOR_ATTENTION_SCRATCH` each (batch, query head) runs its rows a strip at a time -- its
scores, its softmax and its context -- in a slot of whole rows at the start of p, about 64K floats of
scores a strip (256 KB, in the core's cache), at least the bf16 attention's 32-row query block.
Every value is the stored attention's: a score is its channels' products added in order from +0.0, a
context element its keys' in order from +0.0, a row's softmax its own row's, whichever strip holds
the row. `bcir_decoder_forward` passes the bit -- the backward runs its own forward, which keeps its
probabilities -- and a caller of the tensor entry may lend just
`bcir_tensor_attention_scratch_floats(batch, time, heads)` floats of p: 4 MiB at B4/T2048 in place
of the 256 MiB a layer the stored forward writes. Records:
[measurements/2026-10-22/infer-attn/](measurements/2026-10-22/infer-attn/README.json).

The attention model's inference on level 3 -- section 5's level-2 overlay, built against main's
headers and linked with this branch's tensor unit before INFER-ATTN and after it, the overlay's
inference forward passing the bit (`build_level3.py`) -- one forward of 4 x T tokens, 12 rotated
rounds:

| T | before ms | after ms | P-EAGER ms | P-IND ms | after / before | after / P-EAGER | after / P-IND | A/A |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 83.3 | 65.9 | 106.2 | 70.5 | 0.79 [0.71, 0.87] | 0.63 [0.56, 0.71] | 0.91 [0.78, 1.06] | 1.03 [0.96, 1.11] |
| 2048 | 263.7 | 201.2 | 169.0 | 137.3 | 0.75 [0.72, 0.78] | 1.17 [1.09, 1.26] | 1.36 [1.22, 1.51] | 1.01 [0.97, 1.05] |

Codex's trained attention checkpoints give byte-identical logits from both builds at T1024 and
T2048, and the build after INFER-ATTN writes 27.2M floats of its arena in a forward where the build
before writes 49.3M: the scratch path ran. The kernel alone, at the model's shape, runs at
0.68 (0.65-0.71) of the stored forward's time.

So the probabilities were part of the loss, not all of it. Without them BCIR's inference at T1024 is
faster than eager PyTorch's and at parity with Inductor's; at T2048 it is down from 1.56 times eager's
time (this run's build before) to the table's, and still behind both routes. What remains is the
attention's own rate: about 85% of the forward after INFER-ATTN (three layers' 59.7 ms calls of
210 ms), about 72 GFLOP/s on four cores at head dimension 32. PyTorch's flash kernel
keeps a block's scores in cache by an online softmax, whose sum is not this one's: on BCIR's rails
that is a new arithmetic contract with its own bounds, not these bits. Within this contract the next
inference step is the attention kernel's rate at small head dimensions.

What it does not do: the training step still stores P for its backward. The same strips with the
backward recomputing a row's probabilities from its maximum and sum -- the same roundings, so the same
bits -- would take the O(t^2) memory out of training too; that is the next training update for
attention (the tiled attention the training roadmap names), and a larger one, since the arena, the
budget policy and the step program's plan all change with it.

## 7. ATTN-RATE: the attention's own rate (2026-10-22)

Section 6 left the attention's own rate: about 85% of the model's forward at T2048. Timed a phase at a
time on one core (an instrumented copy of the tensor unit, TSC cycles a call at B4/T2048, head
dimension 32), the scratch kernel ran its scores in 169M cycles, its softmax in 196M and its context in
224M. Two causes, each fixed with every value kept:

- **EXP-LANES.** The BCIR exponential's polynomial is a chain of seven dependent multiply-then-add
  steps under `-ffp-contract=off`, and the softmax's loop of it ran at the chain's latency. The
  exponential's own loops now take it over a block of elements a step at a time -- four vectors in a
  row on AVX-512, two on AVX2 -- so the chains overlap whichever compiler builds the unit. That matters
  here: Codex's study builds with GCC, whose unroll pragma leaves the chains one after another, and
  whose AVX2 instance of the old loop was not vectorized at all.
- **CTX-WIDTH.** The AVX-512 context computed a 48-channel block as three 16-lane vectors whatever the
  head's width; at the study's head dimension 32 the third vector's lanes were all masked off, yet
  loaded and multiplied. The block now runs the vectors its channels reach.

On one core the softmax fell to 169M cycles and the context to 178M; the scores were unchanged. The
attention model on level 3, built as section 6 built it (and with GCC, as the study builds it), one
forward of 4 x T tokens, 12 rotated rounds
([measurements/2026-10-22/attn-rate/](measurements/2026-10-22/attn-rate/README.json)):

| T | before ms | after ms | P-EAGER ms | P-IND ms | after / before | after / P-EAGER | after / P-IND | A/A |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 69.8 | 60.7 | 108.5 | 83.0 | 0.88 [0.78, 0.99] | 0.61 [0.55, 0.67] | 0.78 [0.69, 0.88] | 1.01 [0.92, 1.12] |
| 2048 | 209.2 | 193.5 | 189.2 | 170.4 | 0.92 [0.87, 0.97] | 1.05 [0.96, 1.16] | 1.13 [1.08, 1.19] | 1.03 [0.95, 1.11] |

The trained checkpoints' logits are byte-identical from both builds at T1024 and T2048. At T1024 the
model's inference is now faster than eager PyTorch's and than Inductor's; at T2048 it is at parity
with eager's and 1.13 times Inductor's. The rows' absolute times differ from section 6's: the host's
speed moves between sessions, which is why each comparison here is paired within its session.

What remains at T2048 is three phases at about a third each: scores 168M, softmax 169M and context
178M cycles on one core. Each has a candidate within this contract that has not been measured yet: the
softmax's maximum taken where the scores are stored, rather than in its own transposed pass; a head's
key panels packed once for all its strips (each strip repacks every panel up to its last row, 32 times
the stored forward's packs at T2048); and the context's eight broadcasts a key. An online softmax, as
in PyTorch's flash kernel, would be a new arithmetic contract.

## 8. BWD-WIDTH: the training update at long context (2026-10-22)

Sections 6 and 7 measured the model's inference. Its training update at long context, paired with
PyTorch the same way on ATTN-RATE's tree (the study's update -- forward, backward, clipping, AdamW -- of
4 x T tokens, 12 rotated rounds), was faster than both PyTorch routes at T512, faster than eager's and
at parity with Inductor's at T1024, and slower at T2048: BCIR took 94, 281 and 1107 ms an update at
T512, T1024 and T2048 (medians), against eager PyTorch's 219, 364 and 783 and Inductor's 165, 292 and
739 -- paired, x0.994 (0.964-1.025) of Inductor's time at T1024, and x1.43 of eager's and x1.51 of
Inductor's at T2048. BCIR's time grows as T squared there (x3.9 from T1024 to T2048, PyTorch's
x2.2-2.5): the stored attention, whose forward took 66 ms a layer at T2048 on four threads after
ATTN-RATE and whose backward took 206 ms.

The backward's two transposed products -- dV = P^T dY and dK = dS^T Q -- had the waste CTX-WIDTH removed
from the context: a 48-channel block run as three 16-lane vectors whatever the head's width, so at the
study's head dimension 32 a third vector of masked lanes was loaded, multiplied and stored for every
term. **BWD-WIDTH** runs the vectors a block's channels reach, each element's terms the same, in the
same order. The backward fell to x0.850 (0.815-0.888) of its time on one core at B4/T2048, x0.82-0.89 on
four threads under clang and x0.84 under GCC (a second run of 24 rounds, declared before it ran, where
the first run's 12 rounds left the four-thread rows inside their controls' noise); the decoder's
training step at the study's dimensions x0.912 at T2048 under clang and x0.905 under GCC. Every adjoint
is the same: 34 backward cases byte for byte under both compilers, and the model's trained state the
same digest from both builds.

The attention model on level 3, built as sections 6 and 7 built it (and with GCC, as the study builds
it), one training update of 4 x T tokens, 24 rotated rounds of three updates
([measurements/2026-10-22/bwd-width/](measurements/2026-10-22/bwd-width/README.json); the 12-round first
run is recorded beside it):

| T | before ms | after ms | P-EAGER ms | P-IND ms | after / before | after / P-EAGER | after / P-IND | A/A |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 92.3 | 85.8 | 228.6 | 169.3 | 0.91 [0.88, 0.96] | 0.36 [0.35, 0.38] | 0.49 [0.46, 0.51] | 0.98 [0.93, 1.03] |
| 1024 | 283.2 | 262.8 | 335.7 | 282.1 | 0.91 [0.88, 0.95] | 0.77 [0.73, 0.81] | 0.92 [0.88, 0.96] | 1.00 [0.98, 1.03] |
| 2048 | 1054.6 | 980.3 | 720.7 | 689.9 | 0.92 [0.89, 0.95] | 1.35 [1.30, 1.40] | 1.42 [1.36, 1.47] | 1.00 [0.97, 1.03] |

The trained state is the same digest from both builds at every context. At T1024 and T2048 the update is
x0.91 and x0.92 of the build before's time, each interval clear of its control's; at T512 x0.91, its
interval meeting its control's. The first run, 12 rounds, read x0.90, x0.96 and x0.98, none clear of its
control. Against PyTorch the update is now faster than Inductor's at T1024 too (x0.92, where ATTN-RATE's
tree was at parity, x0.994), and at T2048 still x1.35 of eager's time and x1.42 of Inductor's.

One candidate section 7 named was measured and not taken. **ATTN-SOFTMAX** took the softmax's maximum
along the stored rows -- an order-free maximum: NaN is skipped, and since exp1(+-0) = 1 the sign of a
zero maximum changes no value -- then its exponential in row order and one transpose for the in-order
sum. Every value was the same (the 88 attention cases under both compilers), and the attention ran at
parity: the inference attention at B4/T2048 x0.970 (0.915-1.028) of its time on one core, x0.973
(0.929-1.020) on four, x0.950 (0.886-1.018) under GCC. The transposes it removed were spent again in the
pass it added; what remains of the softmax is its exponential and its division, both fixed by the
contract.

What remains at T2048 is mostly the backward's two transposed products. Timed a phase at a time on one
core (an instrumented copy of the unit, TSC cycles a call at B4/T2048), the backward after BWD-WIDTH
spends 564M of its 2026M cycles on dV and 528M on dK, about 8 cycles a vector multiply-add, where dQ --
the same multiply-adds in the context's form -- takes 231M, about 3.4. Each transposed product walks P
down its columns, a row of P a term: at T2048 an 8 KB stride and a new 4 KB page every term. A band of
P's rows at a time, each output row's terms still in ascending order, would read P along its rows; that
is the next training update, to be measured. After it come the dP tiles (300M, and 209M more for their
second evaluation): past T512 they take the GEMM kernel, since the values' transposed panel holds 16384
floats, and past the first 256 columns pass 1 evaluates them again.

## 9. BWD-LONG: the backward at long context (2026-10-22)

Section 8 left the backward's two transposed products at more than half of it: timed a phase at a time
on one core at B4/T2048, dV and dK took 494M and 491M of the backward's 1823M TSC cycles, about 7 cycles
a vector multiply-add, where dQ -- the same multiply-adds in the context's form -- took 3.2. Each walked
its terms down P's columns, a row of P a term: at T2048 an 8 KB stride, a new page every term. And past
T512 the probability-adjoint tiles took the GEMM kernel, because the KV head's transposed values no
longer fit the task's 16384-float panel. Two changes, each with every value kept:

- **BWD-BAND.** Once a row of P spans a 4 KB page, the transposed products run their terms a band of 32
  rows of P at a time: every output block the band reaches takes the band's terms before the next band
  starts, its sums stored between bands exactly, so a band's rows are read along their columns. Each
  element takes its terms in ascending order either way; a shorter context keeps one band, since banding
  there only costs (x1.07 on the 9.84M decoder's attention at T128).
- **DP-PANEL.** Past the whole panel, each tile's values are transposed into the same buffer as the tile
  is evaluated, so the tile runs the register blocks -- the GEMM kernel's arithmetic for these operands,
  the same bits.

On one core at T2048 the backward fell from 1823M to 1077M cycles: dV to 211M, dK to 199M, the tiles to
166M and their second evaluation to 116M. With both builds in one process and 24 rounds a row, the
backward runs at x0.60 of its time on one core and x0.64 on four threads at T2048 (GCC x0.64),
x0.69-0.71 at T1024 and x0.70 at T4096 with heads of 64, unchanged at T512 and at the decoder's T128;
the decoder's training step at the study's dimensions x0.80 at T2048 under clang and GCC. Every value is
the same: 48 backward cases, 7 of them past the threshold, byte for byte under both compilers. The
attention model on level 3, built as the earlier sections built it (and with GCC, as the study builds
it), one training update of 4 x T tokens, 24 rotated rounds of three updates
([measurements/2026-10-22/bwd-long/](measurements/2026-10-22/bwd-long/README.json)):

| T | before ms | after ms | P-EAGER ms | P-IND ms | after / before | after / P-EAGER | after / P-IND | A/A |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 81.8 | 79.9 | 212.7 | 150.7 | 0.98 [0.93, 1.04] | 0.38 [0.37, 0.40] | 0.55 [0.52, 0.58] | 0.97 [0.93, 1.01] |
| 1024 | 253.8 | 234.9 | 334.3 | 272.4 | 0.90 [0.88, 0.93] | 0.68 [0.65, 0.71] | 0.85 [0.81, 0.88] | 1.01 [0.97, 1.05] |
| 2048 | 938.8 | 743.1 | 673.1 | 605.5 | 0.79 [0.77, 0.81] | 1.09 [1.05, 1.13] | 1.23 [1.19, 1.27] | 1.00 [0.98, 1.02] |

The trained state is the same digest from both builds at every context. The update is x0.79 of the build
before's time at T2048 and x0.90 at T1024, unchanged at T512. Against PyTorch it is faster than eager's
and Inductor's at T512 and T1024 and, at T2048, x1.09 of eager's time and x1.23 of Inductor's -- from
x1.35 and x1.42 after BWD-WIDTH, and x1.43 and x1.51 on ATTN-RATE's tree.

One more candidate was measured and not taken. **BWD-FUSE** took each 32-row strip of P through every
pass while it is in cache -- dV's band of P's rows, the tiles and their dots, the score adjoints, dK's
band of the adjoints' rows and dQ's rows -- every value the same, and read x0.93 at T2048 (its interval
holding 1), parity under GCC and at T4096, and x1.09 at T1024: a strip's bands keep all of dV's and dK's
accumulators live, a larger working set than the separate passes'.

What remains at T2048 on one core is 1077M cycles: dQ 215M, dV 211M, dK 199M, the tiles 166M and their
second evaluation 116M, the row dots 110M and the score adjoints 53M. Every product now runs at about
3.1 cycles a vector multiply-add -- two vector operations a term, since the study's contract multiplies
and adds separately. What remains is those products' traffic in P (268 MB a layer at B4/T2048, written
by the forward, then read five times and rewritten once by the backward) and the passes around them:
the tiles' second evaluation (the first 256 columns are kept between the passes, the rest evaluated
twice) and the row dots' transposes.

## 10. ATT-CONTIG: a head's rows its task's own (2026-10-23)

Section 9 left the backward's products at about 3.1 cycles a vector multiply-add and named their
traffic in P as what remains. Four ways at that traffic were built and measured first -- on the
session's earlier host, an Intel Xeon @ 2.80 GHz without AMX and with 1 MiB of L2 a core, where
sections 5 to 9 were measured too -- every value the same in each, and none moved the attention on
four threads at B4/T2048:

- **BWD-RECOMP**: the forward keeps no probabilities and the backward recomputes each 32-row strip of
  P exactly as the forward computed it. The forward fell to x0.680 and the backward rose to x1.192
  (a strip's recomputation is its score product and softmax again): the attention x0.989, its
  interval holding 1.
- **Row fusion**: dQ, dK and dV, alone or together, taken in the strip loop while the strip is in
  cache: x0.968-x1.019 of the backward, none resolved.
- **ATT-PACK**: the context's keys packed in 64 KB chunks and the bands' rows packed: x1.013.
- **ATT-PREFETCH**: P's lines asked for 32, 64 or 128 bytes ahead of the context's and the
  transposed products' walks: no distance resolved a change under either compiler.

What did move it was the layout. On the same buffers, the GEO model's heads laid out contiguous --
B x H batches of one head, where the trainer has B batches of H heads a row -- ran the attention at
x0.845 at T2048 (the backward x0.801), and B1/T4096 with 8 heads of 64 at x0.617. A row of q, k or
v holds every head's channels, so a head's own rows lie heads x dim floats apart: a probe of the
context's walk alone took 2.10-2.24 cycles a vector multiply-add at a 128-float stride (four heads
of 32) and 1.56-1.66 at 32, the head's rows sharing the few cache sets their offset in the row maps
to with every other operand's; and four threads' heads share their output rows.

**ATT-CONTIG** gives each task its head's rows. From `BCIR_ATT_CONTIG_FROM` positions, where a row
holds more than one head, a task lent a region of a caller's workspace copies its head's rows in,
runs the head as a batch of one head of one row stride, and copies its results back: a forward task
its query head's q and its KV head's k and v, then its context out; a backward task its KV head's k,
v, dk and dv, then each query head's q, dy and dq in turn, the group's later heads adding to dk and
dv as they add in place. Values move; none is computed. A region is seven t x d copies, one a worker
of the pool (7.0 MiB at the GEO model's T2048 on four threads); with less, or none, every task runs
on the rows in place. The trainer lends its state's workspace to the binary32 attention under every
contract, its need covering the regions; the launch trace, the program executor and the pack state
the workspace as the attention's operand. Every value is the same: the entries without a workspace
give BWD-LONG's 136 cases byte for byte, and the workspace entries the entries without one in 564
cases, built as committed and built to copy at every context, under clang and GCC.

The threshold is 1024. The design runs on the earlier host chose copies of every operand and a long
context; the records below come from the host the session moved to on 2026-10-23, an Intel Xeon @
2.30 GHz with AMX, 2 MiB of L2 a core and a 260 MiB L3, where copying from T1024 also read faster --
8 heads of 64 x0.842 at B1/T1024 in the run that set the threshold and x0.850 at B2/T1024 on the
committed tree, the GEO shape x0.935 and x0.975 (neither resolved alone; its training step at T1024
x0.953, resolved) -- and copying from T512 did not (x1.009, x1.014 on one thread, x0.983 for 8 heads
of 64, none resolved). The attention, both trees in one process beside the tree after it run in
place, the contiguous layout and an A/A control, 48 rounds a row on four threads
([measurements/2026-10-23/att-contig/](measurements/2026-10-23/att-contig/README.json)):

| shape | after / before | the contiguous layout | in place |
|---|---:|---:|---:|
| GEO, B4/T2048, 4 heads of 32, clang | 0.934 (0.906-0.962) | 0.922 | 1.013 |
| the same, GCC | 0.945 (0.918-0.973) * | 0.912 | 0.991 |
| the same, one thread | 0.984 (0.969-1.000) * | 0.940 | 0.999 |
| GEO, B4/T1024 | 0.975 (0.955-0.995) * | 0.907 | 1.003 |
| B2/T1024, 8 heads of 64 | 0.850 (0.831-0.868) | 0.792 | 1.007 |
| B1/T2048, 8 heads of 64 | 0.716 (0.693-0.740) | 0.683 | 1.009 |
| B1/T4096, 8 heads of 64: clang / GCC | 0.593 / 0.599 | 0.576 / 0.588 | 1.008 / 1.018 |

(* not resolved: its interval holds 1 or meets the A/A control's.) With 8 heads of 64 the copies
take the attention into the contiguous layout's interval at T2048 and T4096, and to x0.850 against
its x0.792 at T1024; with the GEO model's 4 heads of 32 they take the backward to x0.883 and leave
the forward unchanged. On one thread there is little to take: the layout costs x0.940 there,
against x0.922 on four. The training step at the study's dimensions, both trainers in one process,
48 rounds: x0.912 (0.899-0.925) at T2048 and x0.953 (0.934-0.973) at T1024 under the study's
contract, x0.956 at T2048 under v1, unchanged at T512 and on the 9.84M decoder at T128; under GCC
x0.961 at T2048 and x0.968 at T1024, each meeting its control. The attention model on level 3,
built with GCC as the study builds it, its attention lent its regions, one training update of 4 x
T tokens, 32 rotated rounds of three updates:

| T | before ms | after ms | P-EAGER ms | P-IND ms | after / before | after / P-EAGER | after / P-IND | A/A |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 42.7 | 42.6 | 82.2 | 59.0 | 0.99 [0.96, 1.02] | 0.52 [0.51, 0.54] | 0.72 [0.70, 0.75] | 0.98 [0.96, 1.01] |
| 1024 | 115.1 | 113.8 | 136.9 | 120.4 | 0.99 [0.97, 1.02] | 0.85 [0.82, 0.87] | 0.96 [0.93, 1.00] | 1.02 [1.00, 1.05] |
| 2048 | 359.1 | 335.4 | 330.6 | 312.9 | 0.93 [0.92, 0.94] | 1.02 [0.99, 1.04] | 1.08 [1.05, 1.11] | 0.98 [0.97, 1.00] |

The trained state is the same digest from both builds at every context. At T2048 the update is
x0.934 of the build before's time (x0.948 against the control's arm), and at parity with eager
PyTorch (x1.016, from x1.086 before it on this host) and x1.08 of Inductor's (from x1.15); at T1024
and T512 it is faster than both, as before.

What remains: the copies' own cost, the gap between the copies and the contiguous layout (x0.934
against x0.922 for the GEO shape at T2048, x0.850 against x0.792 for 8 heads of 64 at T1024), which
a layout the step keeps -- its projections writing each head's rows contiguous -- would close; the
GEO shape at T1024 and on one thread, where the layout costs 2-9% and the copies take little of
it; and, under GQA, the forward's K and V copied once a query head.

## 11. FWD-STRIP: the stored forward a strip at a time (2026-10-23)

ATT-CONTIG left the GEO attention model's training update at T2048 at parity with eager PyTorch and
x1.08 of Inductor's time. Paired with PyTorch's own attention -- scaled_dot_product_attention,
causal, on the same inputs, the shape the study's model trains at (B4, 4 heads of 32, four threads)
-- the flash kernel took x0.82 of BCIR's attention's time, and the backward was not the reason
([measurements/2026-10-23/fwd-strip/](measurements/2026-10-23/fwd-strip/README.json),
`prototype.json`):

| T2048, 4 threads | forward ms | backward ms | total ms | total / BCIR's |
|---|---:|---:|---:|---:|
| BCIR (the probabilities stored) | 32.5 | 39.5 | 73.4 | 1 |
| PyTorch flash (none stored, recomputed) | 16.6 | 38.0 | 54.9 | 0.819 |
| PyTorch math (the probabilities stored) | 188.9 | 164.5 | 359.0 | 4.876 |

The backward was at parity with the flash kernel's, which recomputes the probabilities it never
stored; the stored forward took twice the flash forward's time. PyTorch's math kernel, which stores
them as BCIR does, took nearly five times BCIR's. Timed a phase at a time, the stored forward's fill
took 15% of it, the scores 29%, the softmax 33% and the context 20% -- each phase taking the head's
whole t x t probabilities, 16 MB at T2048, through the core in turn -- and its per-task cycles were
the same on one thread as on four.

**FWD-STRIP** runs a stored forward the way INFER-ATTN runs a scratch one. From
`BCIR_ATT_STRIP_FROM` positions (1024) each (batch, query head) takes a strip of rows at a time in
place: the strip's rows' future columns zeroed first, then `att_strip` -- its scores, softmax and
context -- on the stored probabilities' own rows while they are in the core's cache. `att_strip`
computes a strip's rows bit for bit as the head-at-a-time forward does, and every future column ends
+0.0 either way, so the stored probabilities, the context and the verdict are as they were: 136
cases byte for byte before and after (and with the strips at every context), 288 stored forwards at
T1016 to T2050 (every float of p after a NaN fill), under clang and GCC.

The attention, both trees in one process, both lent the pool's need, 48 rounds a row:

| shape | forward | backward | total | A/A total |
|---|---:|---:|---:|---:|
| B4/T2048, 4 heads of 32, EXP, clang, 4 threads | 0.868 (0.839-0.898) | 1.035 | 0.962 (0.930-0.994) | 1.007 |
| B4/T2048, 4 heads of 32, v1, clang, 4 threads | 0.910 (0.874-0.948) | 0.990 | 0.947 (0.912-0.984) | 0.993 |
| B4/T2048, 4 heads of 32, EXP, clang, 1 thread | 0.831 (0.814-0.847) | 1.009 | 0.931 (0.912-0.950) | 0.993 |
| B4/T1024, 4 heads of 32, EXP, clang, 4 threads | 0.925 (0.886-0.966) | 0.990 | 0.963 (0.931-0.995) | 0.980 |
| B4/T512, 4 heads of 32, EXP, clang, 4 threads | 1.010 (0.981-1.040) | 1.000 | 1.004 (0.980-1.029) | 0.987 |
| B2/T1024, 8 heads of 64, FUSED, clang, 4 threads | 1.005 (0.971-1.040) | 1.024 | 1.015 (0.984-1.046) | 1.006 |
| B1/T2048, 8 heads of 64, FUSED, clang, 4 threads | 1.020 (0.953-1.093) | 1.023 | 1.021 (0.970-1.075) | 1.011 |
| B1/T4096, 8 heads of 64, FUSED, clang, 4 threads | 0.973 (0.935-1.013) | 1.020 | 0.997 (0.966-1.029) | 1.004 |
| B4/T2048, 4 heads of 32 (2 KV), EXP, clang, 4 threads | 0.923 (0.885-0.963) | 1.032 | 0.988 (0.944-1.033) | 1.008 |
| B4/T2048, 4 heads of 32, EXP, gcc, 4 threads | 0.921 (0.879-0.966) | 1.045 | 0.994 (0.951-1.039) | 1.029 |
| B4/T1024, 4 heads of 32, EXP, gcc, 4 threads | 0.924 (0.898-0.951) | 0.979 | 0.956 (0.935-0.977) | 0.995 |

The forward resolves at T2048 on four threads and one, under GCC and with GQA; the whole attention
resolves on one thread, and on four reads x0.96, inside its control's interval. 8 heads of 64 are at
parity: there the scores' products outweigh the probabilities' traffic. The training step at the
study's dimensions read x0.966-0.977 at T1024 and T2048, none resolved at 48 rounds; the T2048 rows,
run again with 64 rounds as declared before they ran, resolved: x0.949 under clang and x0.951 under
GCC. The attention model on level 3, built with GCC as the study builds it, both builds lent their
regions:

| T | rounds | before ms | after ms | P-EAGER ms | P-IND ms | after / before | after / P-EAGER | after / P-IND | A/A |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 44.1 | 44.1 | 82.2 | 60.0 | 0.997 (0.974-1.021) | 0.538 (0.525-0.552) | 0.744 (0.719-0.768) | 1.001 (0.980-1.023) |
| 1024 | 32 | 115.6 | 113.0 | 151.6 | 123.9 | 0.968 (0.948-0.988) | 0.749 (0.729-0.771) | 0.892 (0.853-0.931) | 1.001 (0.982-1.021) |
| 2048 | 32 | 335.7 | 326.8 | 329.3 | 301.9 | 0.970 (0.955-0.985) | 0.989 (0.966-1.013) | 1.084 (1.063-1.104) | 0.992 (0.974-1.011) |
| 2048 | 64 | 349.7 | 334.7 | 338.0 | 309.5 | 0.951 (0.939-0.963) | 0.990 (0.974-1.006) | 1.086 (1.067-1.105) | 1.004 (0.994-1.015) |

The trained state is the same digest from both builds at every context. At T2048 the update is now
at parity with eager PyTorch and x1.09 of Inductor's time; at T1024 and T512 it is faster than both.

Streaming stores for the zeros, measured against plain stores directly, read x0.959 of the forward
on four threads (inside the control's interval) and x0.930 on one; not taken -- the trainer runs
four threads, and the stores would need a fence and a witness of their own. What remains of the
forward's gap to the flash kernel: its probabilities written out, half of them the future's zeros,
each line read before it is written.

## 12. P-LOWER: the probabilities past the diagonal left unwritten (2026-10-23)

FWD-STRIP's strips still zeroed every row's future columns before its scores: half of a head's
t x t probabilities, 8 MB at T2048, written only to be ignored -- the binary32 backward reads no
probability past a row's diagonal under any contract. Its register blocks read a few columns past it
inside their diagonal tiles and discard them under their masks, and the GEMM kernel's masked
products skip masked terms; the prototype that dropped the fill outright gave every stored forward's
context and verdict and every backward's dq, dk and dv from p NaN-filled first, 176 cases on each of
the AVX-512, AVX2 and portable instances under clang and GCC, as the committed tree did
([measurements/2026-10-23/p-lower/](measurements/2026-10-23/p-lower/README.json), `prototype.json`).

**P-LOWER** makes that a contract. `BCIR_TENSOR_ATTENTION_LOWER` (16) is the caller's word that it
reads each row's probabilities j <= i alone; under it a stored forward may leave p's future columns
as p held them. From `BCIR_ATT_STRIP_FROM` the strips skip their fill (the score blocks still store
+0.0 in the future lanes of the 32-wide key blocks they reach); below it the head-at-a-time forward
writes every column as before. The trainer runs its stored binary32 forward under the bit, and its
program check, the pack's lowering and the launch plan state it; a scratch forward and the bf16
entry ignore it -- the bf16 backward reads the future +0.0s as its tiles' terms. y, the verdict and
every probability j <= i are the same bit for bit: 208 cases of the stored forward under the bit and
the backward from it byte for byte against the tree before, which ignores the bit, under clang and
GCC, and with the strips at every context. A training step from arenas whose every float is a NaN
canary gives the zeroed arenas' loss, gradients and logits bit for bit, and leaves the canary exactly
where the strips leave it: every other float of the arena the step writes.

The attention, both trees in one process, both lent the pool's need, the bit in both arms' word,
48 rounds a row (the last row the 64 declared before they ran):

| shape | rounds | forward | backward | total | A/A total |
|---|---:|---:|---:|---:|---:|
| B4/T2048, 4 heads of 32, EXP, clang, 4 threads | 48 | 0.817 (0.788-0.848) | 0.980 | 0.915 (0.888-0.942) | 0.979 |
| B4/T2048, 4 heads of 32, v1, clang, 4 threads | 48 | 0.859 (0.825-0.894) | 0.996 | 0.925 (0.893-0.958) | 0.987 |
| B4/T2048, 4 heads of 32, EXP, clang, 1 thread | 48 | 0.826 (0.810-0.841) | 1.017 | 0.941 (0.925-0.958) | 1.007 |
| B4/T1024, 4 heads of 32, EXP, clang, 4 threads | 48 | 0.851 (0.809-0.895) | 1.023 | 0.953 (0.912-0.995) | 1.037 |
| B4/T512, 4 heads of 32, EXP, clang, 4 threads | 48 | 0.999 (0.965-1.033) | 1.010 | 1.006 (0.976-1.036) | 0.990 |
| B2/T1024, 8 heads of 64, FUSED, clang, 4 threads | 48 | 0.934 (0.901-0.968) | 0.993 | 0.964 (0.937-0.992) | 1.015 |
| B1/T2048, 8 heads of 64, FUSED, clang, 4 threads | 48 | 0.922 (0.864-0.982) | 0.975 | 0.948 (0.899-1.000) | 0.976 |
| B1/T4096, 8 heads of 64, FUSED, clang, 4 threads | 48 | 0.927 (0.890-0.965) | 0.974 | 0.951 (0.916-0.987) | 0.987 |
| B4/T2048, 4 heads of 32 (2 KV), EXP, clang, 4 threads | 48 | 0.849 (0.822-0.876) | 1.025 | 0.956 (0.920-0.993) | 1.019 |
| B4/T2048, 4 heads of 32, EXP, gcc, 4 threads | 48 | 0.847 (0.823-0.873) | 1.005 | 0.944 (0.914-0.976) | 1.018 |
| B4/T1024, 4 heads of 32, EXP, gcc, 4 threads | 48 | 0.878 (0.835-0.924) | 1.048 | 0.982 (0.932-1.034) | 1.012 |
| B4/T2048, 4 heads of 32, EXP, clang, 4 threads | 64 | 0.844 (0.821-0.869) | 1.019 | 0.949 (0.920-0.979) | 1.010 |

The forward resolves at the GEO shape on four threads and one, under GCC, with GQA and under v1, and
at T1024; T512 is at parity, below the threshold, and 8 heads of 64 gain less, the scores' products
outweighing the probabilities' traffic there. The whole attention resolves on one thread and under
GCC; on four threads under clang its 48-round interval met its control's by 0.001, and the 64 rounds
declared for it resolved, x0.949. The training step at the study's dimensions resolved at T2048
under GCC (x0.937) and under clang's v1 contract (x0.953); under clang's kblock+exp+reduce it read
x0.970 beside a control that drifted to x0.972, and in the 64 declared rounds x0.948 (0.931-0.966).

Against PyTorch's own attention on the same inputs, the attention as the trainer now runs it:

| B4, 4 heads of 32, 4 threads | forward ms | backward ms | total ms | forward / BCIR's | total / BCIR's |
|---|---:|---:|---:|---:|---:|
| T2048: BCIR (the probabilities stored, LOWER) | 21.8 | 38.3 | 60.6 | 1 | 1 |
| T2048: PyTorch flash (none stored, recomputed) | 17.0 | 39.1 | 56.0 | 0.813 (0.758-0.872) | 0.968 (0.903-1.039) |
| T2048: PyTorch math (the probabilities stored) | 183.6 | 154.4 | 340.7 | 8.085 (7.663-8.530) | 5.495 (5.179-5.830) |
| T1024: BCIR (the probabilities stored, LOWER) | 5.8 | 9.8 | 15.7 | 1 | 1 |
| T1024: PyTorch flash (none stored, recomputed) | 5.1 | 11.4 | 16.5 | 0.896 (0.824-0.974) | 1.072 (0.993-1.157) |
| T1024: PyTorch math (the probabilities stored) | 41.6 | 39.8 | 82.0 | 7.062 (6.526-7.642) | 5.084 (4.697-5.503) |

At T2048 the attention is at parity with PyTorch's flash kernel -- which took x0.82 of BCIR's time
before FWD-STRIP -- its backward the faster and its forward still the slower: the probabilities
j <= i written out for the backward to read, where the flash kernel recomputes them. PyTorch's math
kernel, which stores them as BCIR does, takes five and a half times BCIR's time. The attention model
on level 3, built with GCC as the study builds it, both builds lent their regions and stating the bit
on their stored forward:

| T | rounds | before ms | after ms | P-EAGER ms | P-IND ms | after / before | after / P-EAGER | after / P-IND | A/A |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 44.5 | 44.6 | 84.1 | 57.7 | 1.006 (0.975-1.037) | 0.546 (0.528-0.566) | 0.765 (0.737-0.795) | 1.025 (0.993-1.057) |
| 1024 | 32 | 107.8 | 106.7 | 139.0 | 117.0 | 0.981 (0.965-0.998) | 0.761 (0.740-0.783) | 0.899 (0.869-0.929) | 1.003 (0.978-1.030) |
| 2048 | 32 | 324.5 | 311.4 | 321.1 | 292.5 | 0.960 (0.942-0.977) | 0.964 (0.944-0.983) | 1.063 (1.043-1.084) | 1.002 (0.985-1.020) |

The trained state is the same digest from both builds at every context. At T2048 the update now runs
in x0.964 of eager PyTorch's time and x1.063 of Inductor's; at T1024 and T512 it is faster than both.
What remains of the forward's gap to the flash kernel is the stored probabilities themselves; below
the threshold the head-at-a-time forward still writes its future columns, and the bf16 forward writes
them for its backward.

## 13. BWD-KEEP: every probability-adjoint tile evaluated once (2026-10-23)

Timed a part at a time with the trainer's job clock (`step_parts.py`), the GEO step at T2048 -- 297 ms
under kblock+exp+reduce on four threads -- spent 41% of its time in the attention's backward, 23% in its
forward and about a quarter in the GEMMs. The backward evaluates a row block's probability-adjoint tiles
(dP = dY V^T, 32 rows by 64 columns) once for the rows' dots and again for their score adjoints; it kept
the first 256 columns' tiles on its stack (32 KB) and evaluated every tile past them twice -- three
quarters of a row block's tiles at T2048. A prototype that kept every tile on the stack ran the backward
at x0.909 of the tree's on four threads and x0.873 on one, every value the same
([measurements/2026-10-23/bwd-keep/](measurements/2026-10-23/bwd-keep/README.json), `prototype.json`);
a 256 KB frame is no library's to take on a caller's thread.

**BWD-KEEP** keeps the tiles in the workspace the caller already lends: ATT-CONTIG's region of a task,
from `BCIR_ATT_CONTIG_FROM` positions where a row holds more than one head, now also holds 32 rows of the
context's width in whole tiles after the seven copies, and a backward task keeps every tile of a row
block there, so its second pass reads every tile its first evaluated. A kept tile holds the bits its
second evaluation would have, so dq, dk and dv are as they were: 208 cases of the stored forward under
LOWER and the backward from it, and BWD-LONG's 48 backward cases, byte for byte against the tree before
under clang and GCC, and with the tree after built to keep its tiles at every context; the workspace
entries the entries without one in ATT-CONTIG's 564 cases. Without a region -- no workspace, less than
the pool's need, or below the threshold -- a task keeps the stack's first 256 columns and evaluates the
rest twice, as before. The region grows by the kept tiles -- at T2048 and a head width of 32, by a
seventh -- and the trainer's need with it.

The attention, both trees in one process, both lent the larger of their two needs, 48 rounds a row (the
last row the 64 declared before they ran):

| shape | rounds | forward | backward | total | A/A backward | A/A total |
|---|---:|---:|---:|---:|---:|---:|
| B4/T2048, 4 heads of 32, EXP, clang, 4 threads | 48 | 1.034 | 0.905 (0.866-0.946) | 0.953 (0.920-0.987) | 1.027 | 1.022 |
| B4/T2048, 4 heads of 32, v1, clang, 4 threads | 48 | 1.002 | 0.854 (0.813-0.897) | 0.925 (0.895-0.957) | 0.971 | 0.976 |
| B4/T2048, 4 heads of 32, EXP, clang, 1 thread | 48 | 0.994 | 0.874 (0.857-0.891) | 0.916 (0.902-0.931) | 0.986 | 0.988 |
| B4/T1024, 4 heads of 32, EXP, clang, 4 threads | 48 | 0.986 | 0.898 (0.868-0.929) | 0.931 (0.906-0.956) | 0.999 | 0.988 |
| B4/T512, 4 heads of 32, EXP, clang, 4 threads | 48 | 0.958 | 1.002 (0.977-1.029) | 0.983 (0.945-1.022) | 1.002 | 0.982 |
| B2/T1024, 8 heads of 64, FUSED, clang, 4 threads | 48 | 0.992 | 0.902 (0.868-0.937) | 0.942 (0.907-0.979) | 0.958 | 0.979 |
| B1/T2048, 8 heads of 64, FUSED, clang, 4 threads | 48 | 0.987 | 0.874 (0.833-0.915) | 0.929 (0.881-0.979) | 1.013 | 0.996 |
| B1/T4096, 8 heads of 64, FUSED, clang, 4 threads | 48 | 1.000 | 0.875 (0.838-0.913) | 0.931 (0.900-0.963) | 0.988 | 0.988 |
| B4/T2048, 4 heads of 32 (2 KV), EXP, clang, 4 threads | 48 | 1.001 | 0.868 (0.829-0.909) | 0.915 (0.882-0.950) | 0.999 | 0.998 |
| B4/T2048, 4 heads of 32, EXP, gcc, 4 threads | 48 | 1.009 | 0.877 (0.850-0.905) | 0.924 (0.896-0.953) | 0.985 | 0.979 |
| B4/T1024, 4 heads of 32, EXP, gcc, 4 threads | 48 | 0.989 | 0.877 (0.849-0.907) | 0.918 (0.886-0.951) | 0.961 | 0.964 |
| B4/T2048, 4 heads of 32, EXP, clang, 4 threads | 64 | 1.002 | 0.865 (0.843-0.888) | 0.914 (0.896-0.933) | 1.001 | 1.000 |

The backward resolves on every shape from T1024 on: x0.854-x0.905 under clang, x0.877 under GCC at
T1024 and T2048, x0.868 with GQA and x0.874-x0.875 with 8 heads of 64 at T2048 and T4096 (at T1024
x0.902, meeting a control that drifted to x0.958); T512, below the threshold, is at parity. The whole
attention resolves on one thread, with GQA, at T1024 and under GCC; on four threads under clang its
48-round total met its control's interval by 0.006 beside a forward -- which the slice does not touch --
reading x1.034 inside its own, and the 64 rounds declared for it resolved, x0.914 with the forward at
x1.002. The training step at the study's dimensions resolved at T2048 under GCC (x0.944), under clang's
v1 contract (x0.955) and under clang's kblock+exp+reduce (x0.955, clearing its control by 0.0002; in the
64 declared rounds x0.937, 0.919-0.957); T1024 read x0.941 under clang and x0.966 under GCC and T512
x0.967, each meeting its control, and the 9.84M decoder (T128) is at parity.

Against PyTorch's own attention on the same inputs, the attention as the trainer now runs it:

| B4, 4 heads of 32, 4 threads | forward ms | backward ms | total ms | forward / BCIR's | backward / BCIR's | total / BCIR's |
|---|---:|---:|---:|---:|---:|---:|
| T2048: BCIR (the probabilities stored, LOWER, the tiles kept) | 22.3 | 35.3 | 57.5 | 1 | 1 | 1 |
| T2048: PyTorch flash (none stored, recomputed) | 17.2 | 38.0 | 55.4 | 0.796 (0.731-0.867) | 1.116 (0.974-1.279) | 0.994 (0.888-1.113) |
| T2048: PyTorch math (the probabilities stored) | 183.3 | 155.7 | 343.2 | 7.893 (7.234-8.613) | 4.477 (3.898-5.142) | 5.818 (5.212-6.495) |
| T1024: BCIR (the probabilities stored, LOWER, the tiles kept) | 6.0 | 8.8 | 15.0 | 1 | 1 | 1 |
| T1024: PyTorch flash (none stored, recomputed) | 5.7 | 12.9 | 18.8 | 0.993 (0.932-1.058) | 1.364 (1.270-1.465) | 1.222 (1.168-1.279) |
| T1024: PyTorch math (the probabilities stored) | 43.6 | 41.1 | 85.2 | 7.433 (7.087-7.795) | 4.573 (4.213-4.963) | 5.685 (5.359-6.032) |

At T1024 BCIR's attention is now the faster, the flash kernel taking x1.22 of its time; at T2048 the two
are at parity, BCIR's backward the faster and its forward the slower. The attention model on level 3,
built with GCC as the study builds it, both builds lent their regions and stating LOWER on their stored
forward:

| T | rounds | before ms | after ms | P-EAGER ms | P-IND ms | after / before | after / P-EAGER | after / P-IND | A/A |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 45.5 | 45.3 | 85.3 | 61.3 | 1.022 (0.996-1.049) | 0.539 (0.517-0.562) | 0.744 (0.713-0.776) | 1.012 (0.975-1.051) |
| 1024 | 32 | 107.5 | 102.8 | 145.8 | 119.1 | 0.967 (0.946-0.989) | 0.717 (0.688-0.747) | 0.862 (0.830-0.896) | 1.001 (0.975-1.028) |
| 2048 | 32 | 302.2 | 285.3 | 313.5 | 285.3 | 0.946 (0.931-0.961) | 0.911 (0.892-0.930) | 1.005 (0.983-1.029) | 0.987 (0.973-1.002) |

The trained state is the same digest from both builds at every context. At T2048 the update now runs
in x0.911 of eager PyTorch's time and at parity with Inductor's (x1.005, 0.983-1.029), from x1.063; at
T1024 and T512 it is faster than both. What remains of the attention's gap to the flash kernel is the
stored forward's -- the probabilities j <= i written out for the backward to read, x1.26 of the flash
forward's time; below the copies' threshold, or with one head a row, no region is lent and the tiles
past 256 columns are evaluated twice as before.

## 14. ATT-BOUND: the attention and the GEMMs against the host's measured rates (2026-10-23)

After BWD-KEEP the GEO model's training update at T2048 runs at parity with Inductor; its inference does
not. One forward of 4 x 2048 tokens takes 100.2 ms against eager PyTorch's 92.5 and Inductor's 83.2 --
x1.057 (0.969-1.152) of eager's time and x1.190 (1.116-1.269) of Inductor's -- where at T1024 it takes
x0.714 of eager's and is at parity with Inductor's (x0.992). Two thirds of that forward is the attention's,
the same kernels the training forward runs; the step at T2048 now spends 38% of its time in the attention's
backward, 24% in its forward and 26% in the GEMMs
([measurements/2026-10-23/att-bound/](measurements/2026-10-23/att-bound/README.json); records and
instruments, no change to the runtime).

**The forward against the host's rates.** The scratch forward timed a phase at a time in TSC cycles
(`instrument_phases.py`, `phase_bench.c`: B4/T2048, 4 heads of 32, the EXP contract):

| B4/T2048, 4 heads of 32, EXP | scores | softmax | of it: max | exp + sum | division | context |
|---|---:|---:|---:|---:|---:|---:|
| 1 thread, cycles a probability | 2.07 | 2.26 | 0.25 | 1.46 | 0.53 | 1.83 |
| 4 threads, cycles a probability | 2.05 | 2.29 | 0.25 | 1.47 | 0.53 | 1.84 |

A score or a context term is 32 channels' products and sums, under v1 a multiply and an add each. The TSC
counts at 2.3 GHz, but under AVX-512 load this host's cores run faster: `fma_peak.c` sustains 217 GFLOP/s a
core in fused multiply-adds and 102 in v1's multiply-then-add (856 and 399 on four cores, `host-rates.json`)
-- 94 and 44 flops a TSC cycle, a core clock near 3.4 GHz. At v1's measured rate 32 multiply-adds take
1.45 TSC cycles: the scores (with each strip's key panels) run at about 70% of that rate and the context at
about 79%. The softmax's exponential-and-sum pass is mostly BCIR's exponential -- the EXP contract's fixed
sequence, about 38 vector operations for 16 lanes, some 0.9 TSC cycles an element at the measured rate --
and the transposes that carry each row's chains; its division runs at the divider's rate. (ATT-BOUND's first
text took the TSC's 2.3 GHz for the core's clock and called the products' 2.0 cycles their bound; the
measured rates corrected it the same day.) Four ways at the forward were built and
measured against the tree they edit, every value the same in each (`prototypes.json`: the stored forward
under LOWER and the backward from it, ATTN-RATE's 88 forward cases and FWD-STRIP's 288 stored forwards, or
for the scratch forward its 88, under clang and GCC, and again with the copies and strips at every
context), 32 rounds a row beside an A/A control:

| prototype | values | forward, B4/T2048, 4 threads | 1 thread | T1024 | 8 heads of 64, T4096 | attention, B4/T2048, 4 threads | step T2048 / T1024 / T2048 gcc |
|---|---|---:|---:|---:|---:|---:|---:|
| FWD-STREAM (the strip in the region, streamed out) | 12 of 12 equal | 1.190 (1.146-1.235) | 1.035 | 1.253 | 1.017 | 1.121 (1.075-1.170) | 0.991 / 0.988 / 1.009 |
| the strip at a padded stride, copied out | 12 of 12 equal | 1.037 (0.985-1.093) | 1.120 | 1.088 | 1.072 | 1.001 (0.959-1.044) | 1.009 / 0.998 / 1.011 |
| SOFTMAX-DIV (16-lane division) | 12 of 12 equal | 1.017 (0.979-1.057) | 0.964 | 0.965 | 0.966 | 1.018 (0.972-1.067) | 0.976 / 0.977 / 1.000 |
| SCRATCH-PAD (the scratch forward's strips padded) | 4 of 4 equal | 0.983 (0.943-1.024) | 0.980 | 0.993 | -- | -- | -- |

None is taken. FWD-STREAM computes each strip of the stored forward in the region's kept-tiles room and
streams it out to p with non-temporal stores, skipping each line's read before its write; on this host the
260 MB L3 holds a layer's probabilities for the backward, and streaming stores send them to memory instead:
the attention x1.12 at T2048 and x1.15 at T1024, at parity only where the probabilities outgrow the L3 (8
heads of 64 at T4096) and in the step, whose three layers outgrow it too. The padded stride tests the
other suspicion -- a power-of-two context puts a 16-row transpose's rows in one L1 set -- and the copy costs
more than the stride saves; SCRATCH-PAD, which pads the scratch forward's strips with no copy at all, is
at parity on one thread (x0.980) and four, so the sets are not a cost here. SOFTMAX-DIV divides in 16
lanes where the compilers' baseline-ISA loop divides in 4, and no row resolves in two runs: the divider is
the bound, not the vector width. Each prototype's first run, by the drivers `proto_ab.sh` generalises,
read the same (`first_runs`).

**What the fused contract would buy.** The tree already has the attention's fused contract (ATTN-FMA:
every multiply-add of the scores, the context and the backward's products one `fmaf`), which the GEO study
does not use. One library, each arm under its contract word, interleaved beside an A/A control
(`ab_contract.c`, 32 rounds):

| v1 against the fused products (one library) | forward | backward | total | A/A total |
|---|---:|---:|---:|---:|
| B4/T2048, 4 threads | 0.895 (0.846-0.947) | 0.778 (0.735-0.824) | 0.822 (0.781-0.865) | 1.012 |
| B4/T2048, 1 thread | 0.900 (0.877-0.924) | 0.765 (0.738-0.793) | 0.814 (0.792-0.837) | 0.984 |
| B4/T1024, 4 threads | 0.872 (0.843-0.901) | 0.795 (0.770-0.820) | 0.825 (0.803-0.848) | 1.003 |
| B4/T2048, 4 threads, scratch forward | 0.874 (0.838-0.910) | -- | -- | 1.003 |

and the training step at the GEO dimensions under kblock+exp+reduce+attn took 252.5 ms against 290.5
under kblock+exp+reduce (`step-parts.json`, 8 rounds each, not interleaved). The fused contract changes
every attention value -- another arithmetic, declared, not a faster way to the same bits -- so it is the
study's choice to make, not this tree's; under it BCIR's attention at T2048 would take some 47 ms against
the flash kernel's 55.

**The GEMMs.** The step's GEMMs at T2048 run 25.8 GFLOP in 74.3 ms (`gemm.json`): 347 GFLOP/s on four
cores, 41% of the 856 the cores sustain in fused multiply-adds. OT-05 found BCIR's k-blocked GEMMs level
with MKL's on most of these shapes, so MKL leaves the same room; it is the largest the step has left.

What remains, then: the GEMMs at 41% of the host's FMA rate -- a quarter of the step and about a third of
the inference forward -- and the attention's products at 70-79% of v1's rate; beyond the kernels, the fused
contract, which is the study's to choose.

## 15. MM-ROWS: the GEMMs' tasks, chunks, edge blocks and order (2026-10-23)

ATT-BOUND left the GEMMs at 41% of the cores' FMA rate. Timed a shape at a time on one core, the k-blocked kernel
ran the step's products at 46-57% of the core's 217 GFLOP/s and at 85-87% where A's panel fits in L2, so the room
was in where A's rows came from, not in the register block. Four things cost it, each now changed, every value the
same ([measurements/2026-10-23/mm-rows/](measurements/2026-10-23/mm-rows/README.json)):

- **A read again for every column block.** The nest packs one 48-column block's tile and runs every row of its
  range over it before the next block, so each block streams A's rows again; the step's 8192-row operands are
  4 MiB, past a core's 2 MiB of L2, so each block read them from L3. The serial nest now walks 128-row chunks, every
  block of a chunk before the next, where A is stored by rows, its panel 2 MiB or more and a chunk's slice at most
  512 KiB. Where the panel stays in L2 the chunks only pack the tiles again (the 9.84M decoder's 192-row weight
  adjoint read x1.080 with chunks at every depth to 1,024), and so they do for a transposed A, whose chunk is a
  strided slice the kernel already prefetches (its 1,024-row adjoint x1.04-x1.07 with chunks).
- **Three cores on one slice of A.** The pool's task was one column block by 128 rows, the column fastest, so the
  three tasks of a row range ran at once, each core pulling the same rows from L3. A task now spans as many whole
  column blocks as leave its job sixteen tasks -- every column from 2,048 rows -- and reads its rows once for all
  of them. Sixteen keeps a 1,024-row product from becoming eight large tasks one slow core holds up; whole rows from
  1,024 rows and a 32-task rule timed alike at the GEO shapes and within noise at the decoder's.
- **The last block of a 128-wide product.** 128 columns are blocks of 48, 48 and 32, and the kernel computed the
  last as 48: a ninth of its multiply-adds for nothing. An edge block now computes the vectors its columns reach.
- **The weight adjoints last.** In the step's paired and grouped jobs (NDT-SCHED) the input adjoints' tasks came
  first -- 64 small ones at T2048 -- and the weight adjoints' last: 128 or 256 rows over a depth of 8192, three to
  six tasks of some 50 million multiply-adds, run with threads idle around them. A job whose second part's tasks
  are the larger now starts with them; the parts write apart, so any order gives the same values.

Every value is the same: `bitcheck.c` holds every k-blocked entry -- the serial instances, the pool at four and two
threads and through each instance, pair jobs, group jobs whose chains include a call of depth zero -- over 712
cases, 6,081 hashes, before == after under clang and GCC. In one process beside an A/A control (`ab_gemm.c`), on the
committed tree against the one before it:

| GEMM (rows x columns x depth) | four threads | one thread |
|---|---:|---:|
| the step's 8192-row products (seven: nn and nt, 128 to 260 columns) | x0.734-x0.814 | x0.760-x0.887 |
| the weight adjoints (128-260 rows over 8192) | x0.955 (256 x 128), the rest within noise | x0.943-x0.953 |
| a linear's two adjoints as one job (NDT-SCHED-0) | x0.823 | x0.895 |
| three input adjoints chained beside three weight adjoints / two beside two | x0.791 / x0.874 | x0.872 / x0.918 |
| T1024's and T512's products (4096 and 2048 rows) | x0.793-x0.883 | x0.828-x0.920 |

The step, both trainers in one process on three identical states (`step_ab.c`, 48 rounds), and the GEO study's
attention model, built as the study builds it, against PyTorch (32 rounds; inference 16):

| | T512 | T1024 | T2048 |
|---|---:|---:|---:|
| the step at the GEO dimensions, clang | 0.935 (0.907-0.963) | 0.905 (0.880-0.930) | 0.939 (0.925-0.954); 0.930 (0.915-0.946) in 64 declared rounds |
| the step, GCC | -- | 0.935 (0.912-0.958) | 0.938 (0.920-0.955) |
| the step on one thread | -- | -- | 0.952 (0.935-0.968); 0.956 declared |
| the GEO model's training update, after / before | 0.918 (0.894-0.942) | 0.928 (0.909-0.947) | 0.968 (0.952-0.984) |
| the update against Inductor / eager PyTorch | 0.682 / 0.487 | 0.822 / 0.705 | 0.919 / 0.872 |
| the inference forward, after / before | -- | 0.943, holding 1 | 0.945 (0.916-0.974), meeting its control |

The step's GEMM claims, read a part at a time with the trainer's job clock, went from 72.9 to 61.0 ms at T2048
(x0.837: the paired jobs x0.809, the chained x0.914, the input adjoints alone x0.754-x0.771) and from 252.9 to 219.1
on one thread. Under v1, whose GEMMs keep their partition and kernel and take only the order, the step read x0.976,
meeting its control. The 9.84M decoder (1,024 rows, every product at depth 384 or 1,024) is at parity (x0.988 in 64
declared rounds); of its products, three small ones resolved: the head x0.935 and 1,024 x 1,024 x 384 x0.971 faster,
and the 192-row weight adjoint x1.063 slower on four threads -- a path the slice leaves as it was but for where the
code sits. On one thread that adjoint and the 1,024 x 384 x 384 product read x1.03 and x1.02: the kernel's 8 x 48
loops, the same 40 instructions, now start 32 bytes into a cache line where they started on one, and with both
trees' loops at 64-byte boundaries those shapes read x0.999-x1.011.

What remains in the GEMMs: the weight adjoints, at about half the core's rate (their 448-deep tiles live in L2)
and in three to nine tasks. Their next step is a split over the depth -- each K block's product into its own room,
the blocks then folded into C in order, every value the same -- which needs the trainer to lend that room; and the
kernel's loop placement, which moves its rate by a few percent with any edit of the unit.

## 16. SCORES-T: the inference forward's groups with their rows in the lanes (2026-10-23)

After MM-ROWS the GEO model's inference at T2048 still took x1.17 of Inductor's time, two thirds of it the
attention's scratch forward (INFER-ATTN: a forward no backward reads). ATT-BOUND had timed that forward a phase at a
time: the scores 2.07 TSC cycles a probability, the softmax 2.26 (its maximum 0.25, its exponential and sum 1.46,
its division 0.53) and the context 1.83, against v1's measured 1.45 for each product. Part of the softmax's cost
is layout. The scores land with the keys in the lanes, an 8-row x 32-key block at a time, while the softmax needs
each row's chain in a lane of its own, so it transposed every 16 x 16 block in registers: once for the maximum, and
twice for the exponential and sum (in and back out). The score blocks also packed each 32-key panel of keys afresh
for every 32-row strip. Now, on the AVX-512 instance under the exp contract, a scratch strip runs a 16-row group at a
time with the group's rows in the lanes from the start, every value as it was ([measurements/2026-10-23/scores-t/](measurements/2026-10-23/scores-t/README.json)):

- **The scores**: the group's queries transposed once into a d x 16 panel; each 16 x 16 block takes one
  accumulator a key and broadcasts the key's channel, a score its channels' products in order from +0.0, as
  before.
- **The softmax** on the blocks as they stand: the maximum, the exponential and sum, and the division, each lane
  taking a column where the key is at most its row, with no transpose.
- **The context** reads a key's probabilities for the block's eight rows side by side (64 bytes) where it read
  eight rows a row of p apart, otherwise att512_ctx_block's terms in its order.

A head of up to 256 channels and a strip ending on a whole group take the lanes; the rest, other instances and the
libm softmax keep the row path, with the same values. The prototype's phases at B4/T2048 on four threads (TSC cycles
a probability, summed over the threads): the scores 1.96 -> 1.62, the softmax 2.37 -> 1.82 (the maximum 0.25 ->
0.18, the exponential and sum 1.54 -> 1.13, the division 0.54 -> 0.44) and the context 1.70 -> 1.56.

Every value is the same: `scratch_hash.c` (the scratch forward's context and verdict, and the stored forward's every
byte of p, under LOWER too: contexts 16 to 2050, heads of 4 to 256 channels, the four contracts, a pool of four, the
workspace lent, an infinite key), with the tree's fwd, lower, attn and band hashes, before == after under clang and
GCC, and clang's lines GCC's. Measured on the committed tree against the one before it, in one process beside an A/A
control:

| | after / before |
|---|---:|
| the scratch forward, B4/T2048, 4 heads of 32, four threads | 0.927 (0.908-0.947); GCC 0.917 (0.898-0.936) |
| the same on one thread | 0.920 (0.910-0.930) |
| at T512, T256, T128 | 0.884 (0.859-0.910), 0.848 (0.839-0.857), 0.891 (0.854-0.930) |
| at T1024 | 0.911 (0.856-0.970), meeting its control's interval |
| under the fused attention contract (ATTN-FMA), T2048 | 0.859 (0.836-0.882) |
| 8 heads of 64, B1/T2048 / 4 heads of 128, B1/T1024 | 0.889 (0.868-0.910) / 1.023, holding 1 |
| the 9.84M decoder's attention (B8/T128, 8 heads of 48, ATTN-FMA) | 0.883 (0.861-0.906) |
| `bcir_decoder_forward` at the GEO dimensions, T512 / T1024 / T2048 | 0.951, meeting its control / 0.927 (0.901-0.954) / 0.906 (0.882-0.932) |
| the same at T2048 on one thread / under GCC | 0.927 (0.917-0.937) / 0.937 (0.917-0.957) |
| the 9.84M decoder's forward (T128, its GEMMs most of it) | 0.999, holding 1 |
| the stored forward (a control: it keeps the row path) | 1.004, holding 1 |

The GEO study's attention model, built as the study builds it (level 3, GCC, 24 rounds), its logits the same bytes:
its inference x0.945 at T512, x0.916 at T1024 and x0.924 (0.896-0.952) at T2048. At T2048 that's x1.056 of Inductor's
time (0.991-1.126: at parity, where it was x1.17) and x0.977 of eager PyTorch's (0.942-1.013); at T1024 x0.839 of
Inductor's (0.772-0.912) and x0.717 of eager's.

Measured and not taken: the stored training forward on the same path, its groups' probabilities transposed out to
p's rows for the backward (a block's future lanes +0.0 where the row path stores them; every byte of p the same).
Its transposes out cost 0.79 cycles a probability, which the row path spends inside its other phases. At B4/T2048 the
forward read x1.014 (0.972-1.057) on four threads and x1.016 on one, and x0.940 at T1024, meeting its control.

The backward a phase at a time (B4/T2048, four threads, lent the pool's workspace as the trainer lends it; 36.2 ms
a layer), in TSC cycles a probability against v1's 1.45 a product: dV = P^T dY 1.92 (75%), the dP tiles 2.02 (72%),
the rows' dots 0.68, the score adjoints 0.46, dQ = dS K 1.55 (93%), dK = dS^T Q 1.76 (82%), and the tasks' copies in
and out with the rest 1.15 (12% of the call). The copies prefetched 16 rows ahead read 0.78, the call's time within
noise. The backward's room is in its dP tiles and its transposed products: dP's per-tile panels transpose each
tile's values afresh for every row block, and the dots transpose P and dP for each 16 x 16 block, as the softmax did.

## 17. SOFTMAX-T: the exponential's single scaling, the softmaxes in one pass (2026-10-23)

After SCORES-T the scratch forward's softmax still took 1.82 TSC cycles a probability at B4/T2048, 1.13 of them the
exponential and sum and 0.44 the division, a pass of its own over the group's blocks. The AVX-512 exponential and the
v2 SiLU forms were exp1's and the sigmoid's scalar code compiled for sixteen lanes: 2^k two exponent-field scalings,
each vector's chain after the last. ATTN-RATE's block forms took clang's SiLU forms to x0.67-x0.91 and left GCC's at
x0.97-x1.02. Now, on the AVX-512 instance, every value as it was
([measurements/2026-10-23/softmax-t/](measurements/2026-10-23/softmax-t/README.json)):

- **One scaling**: exp1 applies 2^k as (p * 2^k1) * 2^k2 with k1 = k/2. The first product is exact (|k1| <= 76 keeps
  it normal), so the pair rounds once -- only where the result is subnormal -- and vscalefps(p, k) rounds p * 2^k
  once, to the same float. The AVX-512 exponential is exp1's operations in sixteen lanes with that one instruction for
  the two scalings.
- **Four chains a step**: the exponential, SiLU, SiLU times u and the SwiGLU VJP run as AVX-512 instances of
  intrinsics, each step over four vectors before the next (EXP-LANES' order), so clang and GCC both interleave the
  chains.
- **One pass**: the scratch forward's lanes softmax and the stored forward's transposed blocks scale, subtract the
  maximum, exponentiate, mask and sum in one pass, four columns at a time, each row's sum in column order.
- **The division inside the context**: the lanes path's context divides each key's sixteen probabilities by their
  rows' sums in its first channel pass -- rows 0-7's pass the keys to R+7, rows 8-15's the eight after -- once,
  before it reads them, the divider beside the products.

Every value is the same: over every binary32, the exponential and the three SiLU forms (apart, in place and
overlapping) against their scalar definitions under clang and GCC, 0 mismatches, and before against after byte for
byte, 0 apart; `scratch_hash.c` and the tree's fwd, lower, attn, band and elementwise hashes before == after under
clang and GCC, clang's lines GCC's. At a NaN input the SiLU forms return NaNs, as before. Under GCC three of them for
each NaN input carry another sign or payload than before, under clang none: the compiler orders a product's operands,
and which NaN is not part of the v2 contract (the header now says so; the exponential returns its input NaN on every
instance). Measured on the committed tree against the one before it, in one process beside an A/A control:

| | after / before |
|---|---:|
| the exponential, one thread, 16,384 / 2^20 elements | clang 0.733 / 0.769; GCC 0.678 / 0.745 |
| SiLU times u, one thread, 16,384 / 2^20 | clang 0.905 / 0.882; GCC 0.695 / 0.748 |
| the SwiGLU VJP, one thread, 16,384 / 2^20 | clang 0.860 / 0.940; GCC 0.654 / 0.782 |
| the AVX2 exponential (a control: the slice leaves it) | clang 1.003; GCC 1.008 |
| the scratch forward, B4/T2048, 4 heads of 32, four threads | 0.914 (0.875-0.954); GCC 0.909 (0.882-0.937) |
| the same on one thread / at T512 | 0.938 (0.921-0.956) / 0.927 (0.903-0.952) |
| under the fused attention contract (ATTN-FMA), T2048 | 0.899 (0.866-0.933) |
| at T1024 / T256 / T128 | 0.947, meeting its control / 0.937, holding 1 / 0.986 |
| 8 heads of 64 / 4 heads of 128 / the 9.84M decoder's attention | 0.994 / 1.001 / 0.950, meeting its control |
| the stored forward (EXP\|LOWER), T2048 | 0.965 (0.937-0.994), meeting its control |
| `bcir_decoder_forward` at the GEO dimensions, T2048, GCC | 0.908 (0.882-0.935) |
| the same under clang, 64 declared rounds | 0.965 (0.946-0.983), at its control's edge |
| the training step at the GEO dimensions | 0.971-0.998; 0.974 (0.954-0.995) at T512 |

After the slice the SiLU forms take 0.53-0.77 ns an element under GCC and 0.51-0.78 under clang: GCC's forms at
clang's rate, the gap ATTN-RATE's block forms left. Heads of 64 and 128 channels read at parity: their products take
more of the time, the softmax's share falling with the head's width.

The GEO study's attention model, built as the study builds it (level 3, GCC), its logits and trained state the same
bytes: its inference x0.931 (0.908-0.954) at T2048 in 48 declared rounds (x0.926 in 24), x0.960 at T1024, meeting its
control, and x0.976 at T512. At T2048 that's x1.005 (0.968-1.044) of Inductor's time, at parity, and x0.899 of eager
PyTorch's; at T1024 x0.752 of Inductor's. Its training update is at parity at every context: x0.982 (0.964-1.000) of
Inductor's time at T2048, x0.811 at T1024.

The design runs took the slice a level at a time against the tree before (the scratch forward at B4/T2048, four
threads, 24 rounds): the single scaling with the four-chain forms x0.942 under clang and x0.937 under GCC; with the
one-pass softmaxes x0.994 and x0.922; with the division inside the context x0.953 and x0.870. In TSC cycles a
probability the softmax went 1.80 -> 1.15 (its exponential and sum 1.12 -> 0.93), the division's 0.44 became 0.28
inside the context (1.56 -> 1.84), and the call 5.44 -> 4.96. A first draft wrote each vector's chain as one
expression, which clang scheduled a chain at a time: its SiLU ran x1.06 slower than the tree's.

The new harness case, `exp_scalef`, holds every binary32 where the two forms could part -- the subnormal results
[-104, -87] and the overflow edge [88, 89], 2,359,298 inputs -- through every instance, and a sample of them through
the SiLU forms. Of `softmax-t.json`'s sixteen faults it catches nine of the ten in the exponential and the SiLU forms
on its own; with it out of the harness all sixteen are still caught by the exponential's contract case (every 256th
binary32 of [-104, 89]) and the attention cases.

What remains of the scratch forward at B4/T2048, in TSC cycles a probability: the scores 1.58 and the context 1.84,
about 1.56 of it the products, against v1's measured 1.45 a product (ATT-BOUND), and the softmax 1.15, most of it the
exponential's polynomial and the sum (0.93).

## 18. COMPILE-LIMIT: Inductor at each case's own shapes (2026-10-24)

Every comparison with `torch.compile` in sections 5-17 ran its cases in one process, and each case's P-IND module --
`oracle.torch_model_next`'s, a new class each time -- shares one forward code object with the others. Dynamo keeps a
code object's compiles, and two of its rules then apply. Automatic dynamic shapes: a recompile whose input differs in
a dimension from an earlier compile's makes that dimension dynamic, so every case after a process's first compiled
with the tokens' time dynamic (section 5's comparison, past its first model, the parameters' differing dimensions
too). The recompile limit: past eight compiles of one code object Dynamo compiles nothing and runs the frame eagerly,
warning once. Section 5's comparison compiled fifteen cases in one process; its ninth -- the attention model's
training at T512 -- and all six inference cases ran P-IND eagerly, so those seven readings are eager PyTorch's time,
not Inductor's. Of the 64 P-IND cases in this branch's records 20 compiled at their own shapes, 37 with the time
dimension dynamic and those 7 eagerly ([audit.json](measurements/2026-10-24/compile-limit/audit.json)). `witness.py`
shows the mechanism in one process: the second compile creates a symbol for `L['ids'].size()[1]`, the ninth compiles
nothing and the code object stays eager, and from a fresh state with Dynamo's `fail_on_recompile_limit_hit` the
ninth call raises. The OT ledger's routes were not affected: `tools/perf/ot_torch.py` already compiled each shape
from a fresh Dynamo state and ran OT-02, OT-10 and OT-11 a process a route.

The comparisons now compile every case from a fresh Dynamo state (`torch._dynamo.reset()`) at its own fixed shapes
(`dynamic=False`); a recompile past the limit is a hard failure, never a timing; and each compile's shape symbols and
graphs are recorded and checked -- none, and at least one graph
([measurements/2026-10-24/compile-limit/](measurements/2026-10-24/compile-limit/README.json)). Records and docs only:
no runtime change.

**The dynamic rows.** The GEO attention model on SOFTMAX-T's level-3 build, against P-IND compiled at the case's
fixed shapes and P-IND-DYN, the same module with the time dimension dynamic -- what sections 6-17 compiled for every
case after a process's first -- in one process, 24 rotated rounds (B / X below 1: BCIR faster):

| | T | B ms | P-EAGER ms | P-IND ms | P-IND-DYN ms | B / P-EAGER | B / P-IND | P-IND-DYN / P-IND | A/A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| training | 512 | 40.9 | 87.1 | 68.5 | 67.7 | 0.476 | 0.608 (0.573-0.645) | 0.998 (0.937-1.064) | 0.989 |
| training | 1024 | 94.7 | 141.9 | 119.4 | 117.2 | 0.667 | 0.795 (0.762-0.829) | 0.975 (0.920-1.034) | 1.023 |
| training | 2048 | 271.1 | 330.3 | 321.8 | 314.3 | 0.827 | 0.838 (0.812-0.866) | 0.969 (0.927-1.013) | 0.994 |
| the same, declared, 32 rounds | 2048 | 275.3 | 327.4 | 308.1 | 304.6 | 0.835 | 0.893 (0.873-0.913) | 0.991 (0.966-1.018) | 1.007 |
| inference | 512 | 10.8 | 28.4 | 18.4 | 19.2 | 0.400 | 0.593 (0.560-0.629) | 1.016 (0.941-1.097) | 1.022 |
| inference | 1024 | 27.6 | 37.8 | 31.4 | 31.8 | 0.710 | 0.853 (0.795-0.914) | 1.003 (0.914-1.100) | 1.001 |
| inference | 2048 | 78.4 | 87.4 | 77.8 | 77.5 | 0.868 | 0.995 (0.905-1.094) | 0.992 (0.908-1.085) | 1.022 |

The dynamic compile runs at the fixed-shape compile's rate at every context -- each interval holds 1 -- so the 37
dynamic rows of sections 5-17 read Inductor's rate as their tables state, and their conclusions stand. The first update
loss is the same on every route to five places. Between processes PyTorch's time moves by more than a run's interval:
the training update at T2048 reads x0.838 and, in the declared confirmation, x0.893 of Inductor's time, where
SOFTMAX-T's record read x0.982 -- BCIR's update 271-275 ms against the record's 265, PyTorch's 308-322 against 270.
Read the level-3 rows against Inductor to about 10%: the update faster than it below T2048 and from parity to x0.84
at T2048, the inference faster below T2048 and at parity at T2048.

**Section 5 rerun.** `geo_compare_static.py` is section 5's `geo_compare.py` with only P-IND's compile changed, on
the session's later host -- the AMX Xeon @ 2.30 GHz, where section 5's ran on the 2.80 GHz Xeon without it -- level
2's B-DIRECT twice, 12 rounds ([geo2-rerun.json](measurements/2026-10-24/compile-limit/geo2-rerun.json)):

| Model | T | B-DIRECT ms | P-EAGER ms | P-IND ms | B-DIRECT / P-EAGER | B-DIRECT / P-IND | A/A | section 5's P-IND |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| training `global` | 128 | 11.0 | 73.9 | 30.9 | 0.153 | 0.355 (0.337-0.374) | 1.005 | static, 0.33 |
| training `global` | 256 | 21.0 | 93.4 | 45.4 | 0.226 | 0.463 (0.443-0.484) | 1.018 | dynamic, 0.43 |
| training `global` | 512 | 41.3 | 151.1 | 75.3 | 0.277 | 0.540 (0.501-0.582) | 0.974 | dynamic, 0.54 |
| training `query_full` | 128 | 11.3 | 77.5 | 34.4 | 0.152 | 0.330 (0.314-0.346) | 1.044 | dynamic, 0.31 |
| training `query_full` | 256 | 20.8 | 105.9 | 57.5 | 0.193 | 0.363 (0.343-0.384) | 0.966 | dynamic, 0.38 |
| training `query_full` | 512 | 41.7 | 178.2 | 88.8 | 0.239 | 0.474 (0.450-0.499) | 1.029 | dynamic, 0.48 |
| training `attention` | 128 | 8.6 | 37.2 | 21.3 | 0.235 | 0.396 (0.359-0.436) | 1.014 | dynamic, 0.31 |
| training `attention` | 256 | 17.8 | 51.5 | 31.7 | 0.352 | 0.555 (0.516-0.597) | 0.987 | dynamic, 0.49 |
| training `attention` | 512 | 46.6 | 78.2 | 57.2 | 0.610 | 0.831 (0.774-0.892) | 1.007 | **eager**, 0.50 |
| inference `global` | 1024 | 22.5 | 61.1 | 24.8 | 0.359 | 0.883 (0.854-0.913) | 0.995 | **eager**, 0.28 |
| inference `global` | 2048 | 46.3 | 129.3 | 48.6 | 0.356 | 0.899 (0.809-0.999) | 0.993 | **eager**, 0.30 |
| inference `query_full` | 1024 | 26.8 | 80.9 | 34.0 | 0.320 | 0.772 (0.722-0.826) | 1.019 | **eager**, 0.24 |
| inference `query_full` | 2048 | 49.1 | 146.6 | 62.3 | 0.324 | 0.765 (0.735-0.796) | 0.947 | **eager**, 0.29 |
| inference `attention` | 1024 | 47.3 | 41.0 | 32.5 | 1.165 | 1.386 (1.287-1.492) | 1.027 | **eager**, 1.01 |
| inference `attention` | 2048 | 148.1 | 86.3 | 85.6 | 1.688 | 1.667 (1.482-1.875) | 1.011 | **eager**, 1.59 |

Level 2 trains the three models in x0.33-0.83 of Inductor's time and x0.15-0.61 of eager PyTorch's, and runs the
geometric heads' inference in x0.77-0.90 of Inductor's time (`global` at T2048 not resolved: x0.899, its interval
meeting the A/A control's) and x0.32-0.36 of eager's. Section 5's 3.2-4.2 times faster than P-IND was against eager
PyTorch; against Inductor the margin is 1.1 to 1.3. The attention baseline's inference was the larger loss against
Inductor that section 5 could not see: x1.39 of its time at T1024, x1.67 at T2048. INFER-ATTN and the slices after it
took that to x0.853 and x0.995 (the dynamic rows' table, SOFTMAX-T's tree).

## 19. GEO-AMX: the study's bf16 arm on AMX (2026-10-24)

GEO-2 reran every bf16 run of the PR818 study bit for bit but could not time the arm where it was meant to run: the
session's host then had no AMX (section 4, open item 1). On the host since -- the Xeon @ 2.30 GHz with AVX-512,
AVX512-BF16 and AMX -- the study's three models, level 2's overlays linked with this branch's tensor unit (every
tensor entry they call has the same prototype in both headers, `build_amx.py` checks), train under the study's
binary32 contract (14), its bf16 contract (26: the GEMMs on bfloat16 operands, here through the AMX instance, which
`bcir_tensor_mm_bf16_variants` reports) and, for the attention model, the bf16 attention too (90), beside an A/A arm,
24 rotated rounds of 3 updates ([measurements/2026-10-24/geo-amx/](measurements/2026-10-24/geo-amx/README.json)):

| Model | T | binary32 ms | bf16 GEMMs / binary32 | + bf16 attention / binary32 | A/A |
|---|---:|---:|---:|---:|---:|
| `attention` | 256 | 16.0 | 0.676 (0.652-0.700) | 0.585 (0.570-0.600) | 1.011 |
| `attention` | 512 | 40.8 | 0.706 (0.680-0.732) | 0.503 (0.484-0.523) | 0.959 |
| `attention` | 1024 | 106.9 | 0.778 (0.757-0.801) | 0.532 (0.516-0.548) | 0.977 |
| `attention` | 2048 | 329.1 | 0.888 (0.872-0.905) | 0.564 (0.556-0.573) | 0.983 |
| `global` | 256 | 19.6 | 0.756 (0.717-0.797) | | 1.075 |
| `global` | 512 | 37.8 | 0.719 (0.681-0.759) | | 0.974 |
| `query_full` | 256 | 19.1 | 0.784 (0.754-0.814) | | 1.018 |
| `query_full` | 512 | 36.5 | 0.741 (0.726-0.757) | | 1.004 |

Every effect resolved. Where Codex timed the same pairs -- an AMD EPYC 9V74 without AMX, the bf16 GEMMs on the
AVX-512F instance -- the bf16 arm took x1.14 of binary32's time for `query_full` at T256 and x1.04 for the attention
model; on AMX it takes x0.78 and x0.68. The bf16 GEMMs' gain shrinks as the context grows (x0.89 for the attention
model at T2048, where its binary32 attention is most of the update); with the attention on bfloat16 too the update
takes x0.50-0.59 of binary32's at every context. Each arm is deterministic -- its state the same digest every round,
the two binary32 instances' the same -- and the first update's loss differs from binary32's by at most 1.3e-4:
bf16 is another arithmetic, by design (NDT-BF16: one result on every instance).

**Against PyTorch's bf16.** PyTorch's own mixed precision -- `torch.autocast` to bfloat16 around the forward and the
loss, the weights, gradients and AdamW in binary32, the bf16 matmuls oneDNN's AMX kernels on this host -- eager
(P-EAGER-BF16) and compiled at each case's fixed shapes (P-IND-BF16, section 18's protocol), with compiled binary32
(P-IND), in one process with the BCIR arms, 16 rounds:

| Model | T | best BCIR arm, ms | P-EAGER-BF16 ms | P-IND-BF16 ms | P-IND ms | best / P-IND-BF16 | best / P-IND |
|---|---:|---|---:|---:|---:|---:|---:|
| `attention` | 256 | bf16 attention, 9.9 | 67.1 | 40.4 | 38.1 | 0.244 (0.233-0.257) | 0.254 |
| `attention` | 512 | bf16 attention, 22.2 | 87.0 | 64.5 | 61.1 | 0.350 (0.328-0.373) | 0.357 |
| `attention` | 1024 | bf16 attention, 60.3 | 132.0 | 108.4 | 127.0 | 0.555 (0.527-0.584) | 0.481 |
| `attention` | 2048 | bf16 attention, 181.7 | 279.1 | 244.7 | 314.1 | 0.727 (0.698-0.757) | 0.577 |
| `global` | 256 | bf16 GEMMs, 15.3 | 97.0 | 46.3 | 47.6 | 0.320 (0.308-0.332) | 0.315 |
| `global` | 512 | bf16 GEMMs, 27.7 | 144.7 | 72.9 | 74.3 | 0.373 (0.357-0.390) | 0.368 |
| `query_full` | 256 | bf16 GEMMs, 14.6 | 107.8 | 54.9 | 54.9 | 0.267 (0.256-0.278) | 0.262 |
| `query_full` | 512 | bf16 GEMMs, 27.3 | 194.6 | 91.8 | 90.3 | 0.304 (0.284-0.325) | 0.304 |

BCIR's bf16 trains every model faster than PyTorch's, eager or compiled: the attention model in x0.24-0.73 of
Inductor's bf16 time, the geometric heads in x0.27-0.37. Autocast buys PyTorch the attention model at long context
-- Inductor's bf16 update x0.78 of its binary32 one at T2048 and x0.85 at T1024 -- and nothing at T256 and T512
(x1.06) or on the geometric heads (x0.97-1.02). With the attention left in binary32, BCIR's bf16 GEMMs alone take
x1.14 of P-IND-BF16's time at T2048, where PyTorch's attention runs in bfloat16 too: the bf16 attention is what keeps
BCIR ahead there. The routes are matched work: the first update's loss under PyTorch's bf16 is BCIR's bf16 loss to
4e-5. Every compiled arm compiled one graph at fixed shapes, with no recompile-limit warning.

## 20. BWD-T-0: the backward's gap placed; BWD-VT measured and not taken (2026-10-24)

After SOFTMAX-T the GEO step at T2048 spends 117 ms of its 280 ms of jobs in the attention backward (41.9%), 69 ms
in the attention forward (24.6%) and about 65 in the GEMMs (`step_parts.py`). The backward a phase at a time on
SOFTMAX-T's tree, B4/T2048, four threads, four readings in three runs (`instrument_bwd3.py`'s copy of the unit, TSC cycles a
probability summed over the threads; [measurements/2026-10-24/bwd-vt/](measurements/2026-10-24/bwd-vt/README.json)):

| Phase | cycles a probability | of v1's measured rate |
|---|---:|---:|
| dV (P transposed times dY) | 1.97-2.35 | 61-73% |
| the dP tiles (dY times V transposed) | 1.98-2.08 | 69-73% |
| the rows' dots | 0.72-0.93 | |
| the score adjoints | 0.47-0.50 | |
| dQ | 1.53-1.68 | 86-94% |
| dK | 1.76-1.91 | 76-82% |
| the copies and the rest | 1.04-1.14 | |

One thread reads the same (dV 2.34, 62%; dQ 1.57, 92%), so the gap is not the pool's; at heads of 48 the products
run at 76-95%. The two kernels under dV and the dP tiles, copied from the tree and run on cache-resident operands
(`tr_micro.c`, `dp_micro.c`, eight interleaved repeats under clang and GCC), run at v1's rate: the transposed
products' band at medians of 98-109% at row strides of 64, 1024, 2048 and 2064 floats, the probability-adjoint tile
at 95-108% with its values' panel at row stride 64, 2048 or 2064, heads of 32 and 64 (single repeats as low as 53%:
this host's neighbours). What dV and the dP tiles lose in the backward is the movement of their operands, not their
arithmetic.

**BWD-VT**, measured and not taken. In the backward's contiguous task V's slot holds the KV head's values transposed
-- the dP tiles are V's only reader -- so each tile reads the whole panel and none transposes its values again a row
block at a time (`proto_bwdvt.py`; every hash the same under clang and GCC, with the contiguous tasks at every context
too). At B4/T2048 on four threads the backward x0.946 (0.903-0.990), its interval meeting the A/A control's (0.988,
0.958-1.019); on one thread x0.983, meeting its control; at T1024, for heads of 64 at T2048 and T4096, under the fused
contract and under GCC at parity. The per-tile transpose was not what the dP tiles cost.

## 21. GEO-NOW: the study's models on this branch's tensor unit; three levers measured and not taken (2026-10-24)

The study's three models were last timed on level 2: main's tensor unit with the overlays grafted on (sections 5
and 18). On this branch's unit -- the same overlays and headers, GEO-AMX's builds -- in one process with level 2,
the trained states and the logits the same bytes, 16 rounds
([measurements/2026-10-24/geo-now/](measurements/2026-10-24/geo-now/README.json)):

| this branch / main | training T128 | T256 | T512 | inference T1024 | T2048 |
|---|---:|---:|---:|---:|---:|
| `global` | 0.944 | 0.963, holding 1 | 0.917 | 0.845 | 0.821 |
| `query_full` | 0.887, meeting its control | 0.861 | 0.851 | 0.814 | 0.846 |
| `attention` (level 2's overlay) | 0.885, meeting its control | 0.848 | 0.867 | 0.820 | 0.783 |

Against PyTorch at fixed shapes (section 18's script, 12 rounds), the geometric heads now train in x0.31-0.54 of
Inductor's time (level 2: x0.33-0.54) and run inference in x0.67-0.75 of it at T1024 and for `query_full` at T2048
(level 2: x0.77-0.88); `global` at T2048 is at parity (x0.899, its interval holding 1). The attention baseline here
is its level-2 overlay, which takes neither INFER-ATTN's scratch forward nor ATT-CONTIG's region: x1.04 of
Inductor's time at T1024 (holding 1) and x1.35 at T2048 in inference, where the level-3 build -- the overlay with
both -- is at parity (section 18).

**Three levers measured and not taken.**

- **THP-0**: the attention's buffers on transparent huge pages (the host's mode is madvise; 305 MB of huge pages, every
  value the same). The backward read x0.985 (0.947-1.024) at B4/T2048 on four threads in 48 rounds, and x0.913 on one
  thread, meeting its control. Page walks are not what dV and the dP tiles lose (section 20).
- **WADJ-ROWS-0**: the weight adjoints' rows in ranges small enough for sixteen tasks, where 128-260 rows over a depth
  of 8192 ran three to nine. They read x1.19-x1.56 slower (the 260-row one at parity): each task packs the deep
  operand's tiles again. The step
  pairs each weight adjoint with its input adjoint (NDT-SCHED), and the pair already runs at 520 GFLOP/s, x0.95 of the
  forward products' rate, so their few tasks cost the step little.
- **KBK-0**: the k-blocked kernel's 8 x 48 block on resident operands runs at 188-214 GFLOP/s at depth 128 (87-99% of
  the core's 217). With sixteen row blocks to a 448-term tile, as the nest runs them, it runs at 188-200; walking the
  448 terms in L1-sized sub-ranges, with the accumulators stored and loaded between (the same bits), read
  x0.974-x1.017. Only one block called again and again reads 115-120 at depth 448: the tile's 86 KiB past L1 shows
  only when no other block reuses it. Within the nest the forward products run at 145 GFLOP/s on one core, so what
  the GEMMs lose is outside the block's loop.

## 22. GEMM-NEST-0: where the k-blocked nest loses its block's rate (2026-10-24)

KBK-0 left the GEMMs' gap outside the kernel's block (section 21). Timed against the product's size on one core, the
nest runs M x 128 x 128 at 174-178 GFLOP/s while A and C fit in L2 (to 1,024 rows) and at 141-142 from 4,096 rows:
the gap is A's and C's movement from L3, not the block (on four threads, 548-554 at 8,192 rows, the same rate a
core) ([measurements/2026-10-24/gemm-nest/](measurements/2026-10-24/gemm-nest/README.json)). Two ways at it were
prototyped, every value the same (`bitcheck.c`'s 712 cases, clang and GCC), and neither taken:

- **NOPACK-0**: the kernel reads an untransposed B's rows in place, 48 columns a depth through masked loads, where B's
  rows are whole cache lines apart, instead of packing a tile a chunk. At parity on the step's products, and
  x1.49-1.54 slower where B's rows are 4 KiB apart (1,024 x 1,024 x 384): the packed tile is what keeps those rows from
  meeting in L1's sets.
- **GEMM-PF-0**: software prefetches of the next block's lines of A and of the block's lines of C, with write intent,
  ahead of its chain. x0.94-0.95 on one thread at 8,192 rows, but x1.27-1.30 slower for the 1,024 x 384 x 384 product on
  four threads, the 9.84M decoder's: the hardware's prefetchers already cover what a fixed distance can.

Under the study's binary32 contract the step is now balanced: the attention backward (42% of its jobs at T2048), the
attention forward (25%) and the GEMMs (23%) each with the levers above measured at parity. The larger steps left are
contract choices, each a different arithmetic: the bf16 contract on AMX (GEO-AMX, section 19) and the fused attention
(ATT-BOUND, section 14).

## 23. P-BANDS: the stored probabilities in 16-row bands, the rows in the lanes (2026-10-24)

Section 22's reading of a balanced step held only while the stored probabilities sat in the rows' layout. Timed a
phase at a time on the tree before (thread-local TSC counters around each phase, B4/T2048, four threads, two runs),
the backward took 10.1 TSC cycles a probability: dV 2.30 (v1's rate would take 1.44), the probability-adjoint tiles
2.16, the rows' dots 0.87, the score adjoints 0.48, dQ 1.61, dK 1.92 and the copies 0.63
([measurements/2026-10-24/p-bands/](measurements/2026-10-24/p-bands/README.json), `phases.json`). Its transposed
products (dV = P^T dY, dK = dS^T Q) read P down its columns -- a register block of eight keys takes term i from row i
of P, so a band of 32 terms walks 32 rows of P at once, half a line of each a block -- and the dots and score adjoints
transposed every 16 x 16 block. Eight band and prefetch variants of that walk moved the whole backward within its runs'
spread at four threads (33.2-38.5 ms against 34.7) and to at most x0.92 on one (`row-walk-variants.json`). On its own,
over one head's P at T2048 (`placement.json`, eight interleaved passes), the walk took 1.78 TSC cycles a probability
with P resident and 2.17 with P streamed from 16 buffers in turn; the same product over P laid out in 16-row bands
took 1.51 and 1.84, every result's bits the walk's. That layout, not the walk, was what the variants could not change.
This host has no PMU, so which part of the memory system the row walk costs is not measured.

**P-BANDS** makes the layout a contract bit. `BCIR_TENSOR_ATTENTION_BANDS` (32) lays a stored forward's
probabilities out in 16-row bands with the rows in the vector lanes: row i's probability of key j at
`(i/16)*16*T + j*16 + i%16` of its head's T x T region, so a band's column of 16 rows is one 64-byte line. A band holds
keys 0 to R+15 -- its rows' lower triangle and diagonal block, that block's future lanes +0.0 -- and its keys past R+15
+0.0, or, under LOWER, as p held them. The bit takes effect where the host's widest instance of the products' family is
AVX-512, the context a multiple of 16 and the head at most 256 wide: a function of the host and the shapes, never of
the instance a call names, so a forward and the backward that reads its probabilities agree on the layout whichever
instances a program picks for them (PX-3b's per-site choices). On AVX-512 the stored forward computes SCORES-T's
16-row groups in the band itself -- the scores, the softmax (the BCIR exponential's one pass, or libm's with a vector
maximum: the prototype's scalar maximum ran v1's forward x1.24 slower) and the context with its division -- so the
probabilities stay where the group computed them. The backward walks dV and dK over the bands, a block's 32 terms two
runs of eight lines in order; then, a 16-row group at a time, it evaluates its probability adjoints with the rows in
the lanes (its dY transposed once), takes the rows' dots on the way with no transpose, writes the score adjoints with
masked stores where p held the probabilities and takes dQ from the band while it is in the core's cache. The portable
and AVX2 instances run the same layout through scalar paths. Every accumulator takes the rows' layout's terms in their
order, so every value is as it was; only where each probability sits differs. A scratch forward, the bf16 entries, a
portable build and an AVX2-capped build ignore the bit. The trainer's stored binary32 forward runs under LOWER|BANDS
and its backward under BANDS; the pack's lowering, the step program's words and the Python launch plan state the bit.

Every value is the same. The row paths, the calls without the bit -- the scratch forward's 1,248 cases, FWD-STRIP's
288 stored forwards, P-LOWER's 208, ATTN-RATE's 88 forwards and BWD-LONG's 48 backwards -- hash before == after under
clang 18 and GCC 13, clang's lines GCC's. `bands_check.c` runs the tree before's unit and the tree after's under the bit
side by side, every probability at its band address, every score adjoint, y, dq, dk and dv byte for byte: 6,220 cases
in 13 runs over every pair of forward and backward instances and across the two compilers, 0 failed. The timed runs
below compare every arm's outputs in every round, and the trainer's state is the same digest with the bit on and off.

The attention, both trees in one process on one set of buffers, both lent the larger of their two needs, 48 rounds a
row (the trainer's word, EXP+LOWER+BANDS, which the tree before reads as EXP+LOWER):

| shape, contract, threads | forward | backward | total | A/A total |
|---|---:|---:|---:|---:|
| B4/T2048, 4 heads of 32, EXP, clang, 4 threads | 0.901 | 0.807 (0.782-0.833) | 0.841 (0.817-0.866) | 1.001 |
| the same under GCC | 0.843 | 0.830 | 0.834 (0.811-0.858) | 0.989 |
| the same on one thread (24 rounds) | 0.914 | 0.838 | 0.866 (0.847-0.886) | 0.998 |
| T1024 / T512 / T256 / T128 | 0.843 / 0.773 / 0.800 / 0.821 | 0.807 / 0.741 / 0.816 / 0.829 | 0.821 / 0.755 / 0.808 / 0.829 | 1.014 / 0.967 / 0.984 / 0.969 |
| B4/T2048, FUSED / v1's exponential / FUSED with v1's exponential | 0.801 / 0.988 / 0.926 | 0.851 / 0.816 / 0.852 | 0.831 / 0.904 / 0.895 | 0.994 / 0.980 / 0.998 |
| B2/T1024, 8 heads (2 KV) of 48 | 0.897 | 0.862 | 0.875 (0.836-0.916) | 0.988 |
| B1/T2048, 8 heads of 64 | 0.932 | 0.867 | 0.890 (0.857-0.925) | 1.016 |
| B1/T4096, 8 heads of 64; 64 rounds declared before they ran | 0.913; 0.949 | 0.935; 0.912 | 0.926; 0.925 (0.900-0.949) | 1.020; 0.991 |
| B1/T1024, 4 heads of 128; 64 rounds declared before they ran | 0.973; 0.973 | 0.903; 0.937 (0.906-0.969) | 0.926; 0.950 (meets its control by 0.005) | 0.972; 1.009 |
| B8/T128, 8 heads (4 KV) of 48, EXP+FUSED | 0.790 | 0.868 | 0.838 (0.809-0.869) | 1.004 |
| B4/T1000 (no multiple of 16: the bit takes no effect) | 0.971 | 0.973 | 0.973 (meets its control) | 0.972 |

The backward resolves at x0.74-x0.87 wherever the bit takes effect with heads of 32 to 64 and at x0.90-x0.94 with
heads of 64 at T4096 and of 128; the forward at x0.77-x0.95 under the BCIR exponential, x0.93 with the fused products
and v1's exponential, and at parity under v1's alone (its libm calls dominate). The whole attention resolves at
x0.75-x0.93 on every shape but heads of 128, whose products' arithmetic outweighs the layout (x0.95, its interval
meeting its control's by 0.005), and the control where the bit cannot take effect reads the same in both arms. A phase at a time the backward's task went from 10.1 to 8.1 TSC cycles a
probability: dV 2.30 to 1.90, the tiles, dots, score adjoints and dQ together 5.12 to 3.73, dK 1.92 to 1.88.

The training step at the study's dimensions, both trainers in one process, every round's weights the same bytes, 48
rounds: x0.892 (0.870-0.915) at T2048 under clang, x0.887 under GCC, x0.917 on one thread, x0.911 at T1024 and x0.861
at T512; under v1's contract x0.949, and the 9.84M decoder (T128, the fused attention) x0.962 -- every row resolved. A
part at a time (the trainer's job clock, 8 rounds) the attention's backward is now about 39% of the T2048 step, its
forward 24% and the GEMMs 26%. The GEO attention model on level 3, built with GCC as the study builds it, both builds
stating the bit (the tree before ignoring it), its trained state and logits the same: training x0.808 (0.788-0.828) at
T512, x0.917 at T1024 and x0.905 (0.876-0.935) at T2048; inference at parity (the scratch forward ignores the bit).
Against PyTorch at fixed shapes (COMPILE-LIMIT's protocol, 24 rounds):

| T | BCIR ms | P-EAGER ms | P-IND ms | BCIR / P-EAGER | BCIR / P-IND | A/A |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 32.9 | 83.2 | 64.8 | 0.407 (0.389-0.426) | 0.537 (0.511-0.564) | 1.016 |
| 1024 | 85.3 | 135.7 | 115.2 | 0.629 (0.605-0.654) | 0.727 (0.696-0.759) | 0.988 |
| 2048 | 245.2 | 310.7 | 306.9 | 0.780 (0.762-0.799) | 0.797 (0.775-0.821) | 1.003 |
| 2048 (32 rounds) | 242.8 | 314.0 | 298.6 | 0.780 (0.766-0.795) | 0.824 (0.806-0.842) | 0.993 |

The update at T2048 now takes x0.80-x0.82 of Inductor's time, from x0.84-x0.89 (section 18). Against PyTorch's
`scaled_dot_product_attention` on the same inputs (B4, 4 heads of 32, four threads, 24 rounds) the attention as the
trainer runs it takes 48.7 ms at T2048 -- its forward 19.1, its backward 29.6 -- against the flash kernel's 56.6 (17.0
and 38.2): flash x1.188 (1.128-1.251) of BCIR's time, from parity (section 13), its forward at parity with BCIR's
(x0.966, 0.906-1.029) where it was x0.80; at T1024 flash x1.422 (1.320-1.532). The math kernel, which stores the
probabilities as BCIR does, takes x6.9.

The prototype first: a copy of the unit with the bands (proto_bands.py, AVX-512 only, the forward's bit a static
global) read x0.851 for the whole attention at B4/T2048 on four threads and x0.831 on one, every value the tree's over
529 cases; the tree's implementation reads x0.841 and x0.866. What remains is the arithmetic: under the BCIR
exponential the backward's walks run at 76-77% of v1's rate and its groups at 3.7 TSC cycles a probability against two
products' 2.9; heads of 128 gain least. The larger steps left are still contract choices: under the fused attention
contract with the bands the attention at B4/T2048 takes 40.0 ms against the BCIR exponential's 50.1 (about x0.80), every
attention value different, and the bf16 contract on AMX (section 19).
