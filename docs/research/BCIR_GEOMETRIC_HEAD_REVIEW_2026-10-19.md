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

## 24. POOL-AFFINITY: the pool's tasks in per-worker blocks (2026-10-24)

Section 23 left the attention about 63% of the GEO step at T2048, but at the study's own training contexts the step is
mostly the GEMMs' and the elementwise jobs'. Timed a part at a time after P-BANDS with the trainer's job clock (NEXT-0,
[measurements/2026-10-24/pool-affinity/](measurements/2026-10-24/pool-affinity/README.json), `survey.json`), the GEMMs
took 54% of the step at T256 and the attention 21%; 46% and 34% at T512; 37% and 48% at T1024. On one thread the T512
step took 109.3 ms and on four 36.5, where a quarter would be 27.3 (NEXT-0's first reading: 106.0 and 31.4). The
elementwise jobs scaled worst -- the RMSNorm ran x1.37 its one-thread speed, its VJP with the residual add x1.68 and
SwiGLU's VJP x2.13, against the attention backward's x3.18 -- and each reads rows the job before it wrote. On four
threads an RMSNorm over the step's 2,048 rows took 45 µs back to back, 56 after a pool job wrote its operands and 161
cold. The pool's one counter sent each task to whichever thread asked next, so a job's rows ran on another core than
the job that wrote them. This host has no PMU, so which level of the memory system those reads waited on is not
measured.

Six levers inside the kernels were measured first, on the tree before, and not taken (`survey.json`). EXP8 -- the
banded softmax evaluating eight columns of a block at once where the tree takes four -- takes the resident softmax
from 1.10 to 1.02 TSC cycles a probability, about 2% of a group's three phases; it is measured only on its kernel and
left for a slice of its own. FWD-BANDW-0 (the next band prefetched for writing, or each group streamed out of a
resident scratch) read the stored forward x1.022 and x1.009; BWD-SWEEP-0 (dV and dK inside the groups' sweep) and
BWD-PF-0 (P's lines prefetched in the banded walks) read the backward x0.966 and x0.963, none of them resolved, the
sweep's phases summing to the tree's (8.15 TSC cycles a probability against 8.12). WADJ-TILE-0 (the weight adjoints'
rows in ranges of 64 for more tasks) took a lone 1,024-row weight adjoint to x0.774 but the chains to x1.085-x1.207 and
the 8,192-row pair to x1.081, each resolved. POOL-YIELDS-0 (the workers yielding 200,000 rounds before they sleep,
where the tree yields 512) read the step x0.962 (0.929-0.996) at T256 and x0.997 at T512 in alternating processes; it
keeps every idle worker yielding the processor for 200,000 rounds after each job, a cost to anything else the host
runs, so it is not taken.

**POOL-AFFINITY** deals a job's tasks in contiguous blocks, one a worker. Worker w's block of a job's n tasks over its
W workers starts at `n/W*w + min(w, n%W)` -- n/W a block, one more for each of the first n%W -- so the blocks cover the
tasks in order, worker by worker, and consecutive jobs whose partitions follow the same rows (a product's rows of C,
then the norm over them) run those rows on the same thread, its caches holding what the job before wrote. Where the
compiler has atomics each block has a counter of its own on a 64-byte line of its own: a worker claims its block's tasks
one at a time in ascending order and then the other blocks' leftovers, from the next worker's on, so no worker idles
while a task remains and a thread that wakes late takes fewer. A job's `head` leading tasks are claimed in turn from the
shared counter before the blocks: the pair and group jobs whose large weight adjoints lead (MM-ROWS's order) name them,
so those long tasks start together on as many threads, and every task of a job with no more tasks than workers is
claimed so. Without atomics a worker runs the head tasks w, w+W, ... and then its whole block. The blocks' counters
live in the block the caller lends the pool (`bcir_tensor_pool_size` counts a line of room and a block a thread), so the
unit still allocates nothing.

Every value is the same: which thread runs a task never reaches the arithmetic, since every task writes outputs no
other task of its job writes and every partition is the shapes' alone, as NDT-THREADS made them. `bitcheck.c`'s 6,081
k-blocked GEMM cases and the attention's five hash programs (the scratch forward's 1,248 cases, FWD-STRIP's 288,
P-LOWER's 208, ATTN-RATE's 88, BWD-LONG's 48) print the same lines before and after under clang 18 and GCC 13, clang's
lines GCC's; every timed arm below holds its weights, logits or outputs the same bytes in every round, and the study's
models keep their trained states and logits.

The mechanism, alone (`elementwise.json`, `ew_cold.c`, four threads, 2,048 rows): the RMSNorm takes 26.9 µs back to back
where it took 43.5 -- back-to-back runs of one job had landed its rows on different cores too -- 29.8 after a pool job
wrote its operands where it took 55.2, and 127.8 cold where it took 149.3; the residual add 10.7 where it took 22.1,
15.3 where it took 30.0; the RMSNorm's VJP 135.3 after a pool job where it took 174.8. On one thread, with no pool, every
row reads as before.

The training step at the study's dimensions, both trainers in one process beside an A/A control (48 rounds, 24 at
T2048), each state on its own pool, and alone in its own process, steps back to back on one pool as training runs them
(`steady.json`: the libraries alternating run by run, six runs each, the ratios paired by run):

| | one process | one library a process |
|---|---:|---:|
| T128 | 0.918 (0.867-0.972; meets its control) | 0.824 (0.778-0.872) |
| T256 | 0.949 (0.893-1.007) | 0.944 (0.898-0.991) |
| T256 under GCC / on two threads / v1's contract | 0.920 / 0.919 / 0.893, each resolved | |
| T512 | 0.942 (0.912-0.972; meets its control) | 0.932 (0.865-1.004) |
| T512 under GCC | 0.953 (meets its control) | |
| T1024 / T2048 | 0.986 / 1.004 | 1.000 (T1024) |
| the 9.84M decoder (T128, the fused attention) | 0.962 (0.946-0.980) | |

A part at a time at T256 the job clock's total went from 15.11 to 13.94 ms (x0.922): SwiGLU's VJP x0.54, the residual
add x0.56, the lone nt products x0.77, SiLU-times-up x0.79, RoPE x0.84; at T512 two runs of the tree before differ by
19% (31.8 and 37.9 ms), so its parts say little beyond the RMSNorm's x0.87. The inference forward, in one process
(`forward.json`): x0.874 (0.807-0.947) at B1/T128, x0.919 at B4/T256 and x0.959 at B4/T1024 meeting their controls,
parity at B4/T2048; alone in its own process (`steady.json`): x0.837 (0.761-0.920) at B4/T256, x0.894 (0.846-0.944) at
B1/T128, x0.906 (0.884-0.929) at B1/T256 and x0.972 at B4/T1024. The attention alone (B4, 4 heads of 32) reads x0.918
(0.893-0.944) at T256 -- forward x0.911, backward x0.922 -- where a head's probabilities fit in a core's L2, and parity
at T2048, where they do not. The gain is where a worker's share of consecutive jobs' rows fits in its 2 MiB of L2: T128
to T512 and the small batches, not T1024 and T2048.

The study's three models (level 2, `build_amx.py`'s builds of each unit, GCC as the study builds them, one process,
16 rounds; `study-ab.json`), every trained state and logit the same:

| model | training T128 / T256 / T512 | inference T256 / T1024 / T2048 |
|---|---|---|
| global | 0.701 / 0.658 / 0.707 | 0.801 / 0.830 / 0.851 |
| query_full | 0.766 / 0.765 / 0.766 | 0.803 / 0.826 / 0.812 |
| attention | 0.872 / 0.917 / 0.944 | 0.910 / 0.976 / 0.969 |

every geometric-head row resolved, the attention model's but T512's training and T1024's and T2048's inference,
which meet their controls or hold 1. The geometric
heads' many short jobs over the same rows gain the most: their training takes x0.66-x0.77 of its time. Against PyTorch
at fixed shapes (COMPILE-LIMIT's protocol, 12 rounds; `study-vs-pytorch.json`) their training now takes x0.22-x0.35 of
Inductor's time and their inference x0.52-x0.67; the attention model's training x0.32 at T128 to x0.65 at T512. The
level-2 overlays' attention model is slower than PyTorch in inference (x1.09 of Inductor's time at T1024, x1.33 at
T2048), as it was in section 21; the attention model built as P-BANDS builds it (level 3, `level3-ab.json`) is at
parity before and after -- training x0.995, x1.005 and x0.972 at T512, T1024 and T2048, inference x1.006 and x0.939,
none resolved -- and against PyTorch takes x0.513 (0.487-0.541), x0.665 and x0.793 (0.761-0.827) of Inductor's time in
training at T512, T1024 and T2048, x0.736 and x0.964 in inference, as in section 23.

The design a part at a time, on the tree after against itself (`design.json`, `steady.json`; proto_aff_variants.py):
without stealing -- each worker its own block alone -- the step takes x1.092, x1.104 (1.057-1.153) and x1.050 of the
committed pool's time at T128, T256 and T512 in one process, and the pair and chain jobs x1.06-x1.40; stealing is the
balance that blocks need. Without the led head the step is at parity in one process (x0.996-x1.009) and x0.946
(0.909-0.984) at T256 alone in its own process, parity at the other contexts; alone, four of the pair and group jobs
at 2,048 and 8,192 rows take x1.14-x1.20 of the led pool's time, resolved, the rest within their controls. The head stays: it costs the step nothing measurable in one
process and keeps the weight adjoints from finishing last. How it trades against the input adjoints' rows -- a worker
held in a long head task has its block's rows taken by the others -- is left open.

The rigs, read for this slice: `ab_gemm.c` and `ab_bands.c` give their arms one set of buffers, and the second
library's arm always follows an arm on the first library's pool. Under POOL-AFFINITY each pool's workers keep their own
rows of those buffers in their cores' caches, so the second arm reads them from other cores: the tree after against a
byte copy of itself (`rig-control.json`) reads the lone nt products x1.395 and x1.618 slower at 1,024 and 2,048 rows,
with one code on both sides, the pairs and chains within 0.96-1.04. Single GEMM jobs in that rig (`gemm.json`) say
nothing of this slice; the step, the forward and the study, each arm on its own state and pool, do. `ab_bands.c` also
made every arm's pool with the first library's entries, which holds only while the two units lay their pools out
alike: the tree after's attention ran on the tree before's pool and waited on reports its workers never made, every
thread asleep. `ab_bands_own.c` makes each arm's pool with its own library; P-BANDS's records, whose two units share the
layout, stand.

Witnesses: the C harness's `pool_blocks` checks every task once at each cap (0 to 4) of a four-thread pool over 60
jobs, the pool inside the bytes it asks for from every `max_align_t` offset of a line, and a held first task's block
taken by the other workers; built with `BCIR_TENSOR_POOL_PROBE`, with stealing off each worker's own block is one
contiguous ascending run of its share's length, with stealing on the last worker takes every other block's rest, and
led jobs at caps 0, 2 and 3 and heads from 0 past the job's tasks run every task once -- under ThreadSanitizer too.
`pool-affinity.json` PA1-PA11, each caught by its own check; NDT-THREADS' NT2 and NT11 anchored on the claim this slice
rewrote and are re-anchored on the block claim, caught again.

Not claimed: hosts with more cores or another cache topology (this one has 4 vCPUs, 2 MiB of L2 a core); the bf16
contract's jobs, dealt in blocks too but not timed here; counter-based figures (no PMU on this VM).

## 25. LOSS-ONCE: the loss's exponentials once; the study's gate a span at a time (2026-10-24)

After POOL-AFFINITY the step's large parts sit near their rates on this host, measured a part at a time on the tree
before this slice (NEXT-1, [measurements/2026-10-24/loss-once/](measurements/2026-10-24/loss-once/README.json),
`survey.json`). Back to back on four threads the k-blocked GEMM runs the T256 step's products at 480-613 GFLOP/s,
x2.7-x3.7 their one-thread rates (the 1,024 x 128 x 128 product at 480.1, where NEXT-0 read it at 355.6 before
POOL-AFFINITY), and its weight adjoints over 1,024 rows, three and six tasks, at 399 and 457. Timed a worker at a
time with TSC counters, the workers spend 88-93% of the 1,024 x 128 x 128 A Bᵀ product's job inside its tasks, and
the caller is back 3.4-5.3% of the job after the last worker leaves; the plain product's job reads 75-91%, the
2,048-row one's 80-88%, and the weight adjoint's three tasks leave one of the four workers idle (59-64%). After a 1
ms pause, the workers asleep, the same products run at 298-452 GFLOP/s, 39-57 µs a job slower; in the step the
workers sleep 2.2, 3.9 and 8.6 times a step at T128, T256 and T512 (84 jobs a step; a counter in a copy of the
unit), and the step spends 1.9% of its time outside its jobs. With no PMU, a sampling profiler (SIGPROF on the
process's CPU time, each sample's instruction pointer resolved against the libraries' symbols offline) placed what
is left: in BCIR's own step at T256 the GEMMs 50% of the samples, the attention 20%, and the loss's task and libm
5.7% between them; in the study's geometric heads' inference at T1024 the GEMMs 56%, and the gate 16.6-19.2% --
`wedge_gate_fwd_task` and `bcir_tensor_expf`, the tensor unit's scalar exponential called once an element -- with
the heads' own pair kernels at 5.8% (`global`) and 11.1% (`query_full`, with its query contraction).

**LOSS-ONCE.** The trainer's loss called libm's `exp` twice a logit, for the row's sum and again for each adjoint. A
row now keeps its first 1,024 exponentials in double on its task's stack and its adjoints divide the kept values; a
row wider than the buffer evaluates its tail again, as before. The same call on the same argument returns the same
double, and the sum, the term and both divisions are the loop's operations in its order, so every adjoint, term and
loss is the two-evaluation loop's bit for bit, at every thread count and under every contract.

**The study's builds.** The PR818 study's three native libraries are forks of BCIR's trainer, with the same loss.
`once_level3.py` makes the same edit to each fork and, in the geometric heads, gives the gate's forward its
exponential a task's 512 elements at a time through the tensor unit's span entry, `bcir_tensor_exp` -- the widest
instance, whose every instance `exp_span_check.c` holds to `bcir_tensor_expf` on all 2^32 inputs -- where the fork
calls `bcir_tensor_expf` an element. Every trained state and logit is the fork's.

In the step on four threads, by the trainer's job clock, the loss job takes x0.618-x0.645 of its time over two runs:
0.67-0.70 ms to 0.44-0.45 at T256 and 1.33-1.38 to 0.84-0.85 at T512. The loss task alone, on one thread over 1,024
to 8,192 rows of 260 logits, goes from 9.6-10.1 ns a logit to 6.4-6.5 under clang (x0.650-x0.672) and from 9.0-9.4
to 6.6-6.7 under GCC (x0.712-x0.742). With 512 rows of 1,500 logits, where the tail past the buffer is still
evaluated twice, it reads x0.797 and x0.840. Every adjoint and term is the same bytes. The step itself, alone in its
own process with the libraries alternating, reads x1.008, x0.977, x1.002 and x0.976 from T128 to T1024 and x0.988
for the 9.84M decoder, each holding 1: the 1.6-1.9% of the step the loss job saves is below this rig's resolution.
Every run's weights are the same digest on both trees.

The study's models with the overlay against without it, in one process, 16 rounds, every trained state and logit the
same (`study-ab.json`, `level3-ab.json`):

| model (build) | training T128 / T256 / T512 | inference T1024 / T2048 |
|---|---|---|
| `global` (level 2) | 0.918 / 0.918 / 0.943 | 0.859 / 0.849 |
| `query_full` (level 2) | 0.901 / 0.944 / 0.922 | 0.869 / 0.863 |
| `attention` (level 3: the loss alone; training T512 / T1024 / T2048) | 0.996 / 0.999 / 0.995 | 1.011 / 0.929 |

The geometric heads' inference is resolved at both contexts, and their training for `global` at T512 (x0.943,
0.904-0.984). The other training rows meet their controls (the A/A arms read 0.92-1.03 here) or hold 1. The attention
model's rows hold 1 or meet theirs; its inference runs no loss. Against PyTorch at fixed shapes (section 18's
script, 12 rounds; `study-vs-pytorch.json`), the geometric heads now run inference in x0.452-x0.605 of Inductor's
time (x0.52-0.67 in section 24's records) and x0.16-0.26 of eager's, and train in x0.219-x0.327 of Inductor's. The
attention model at level 3 (`level3-vs-pytorch-*.json`, 24 rounds) trains in x0.529, x0.697 and x0.824 of Inductor's
time at T512, T1024 and T2048, and runs inference in x0.746 of it at T1024 and at parity at T2048 (x1.013,
0.953-1.077).

Witnesses: the C harness's `loss_once`, through `bcir_decoder_loss_probe` (BCIR_DT_LOSS_PROBE, the harness's builds
only), holds every adjoint, row term and their sum byte for byte to the two-evaluation loop written apart, at nine
vocabularies from one logit to 2,049, five row counts about a task's sixteen, four kinds of row and targets in the
buffer and past it, on no pool and pools of two and four (135 runs). `loss-once.json` LO1-LO7, each caught by its
own check; NDT-REDUCE's NR4 and NR9, anchored beside the new task, caught again.

Not claimed: vocabularies past the buffer gain only on their first 1,024 logits; the study's other kernels (the
pair kernels' double-precision chains, the scans after the gate) are the study's own, measured here, not changed;
the survey's readings vary run to run on this VM (a first exploration read the 1,024-row product at 529.7 GFLOP/s
and under two sleeps a step at T256), and counter-based figures are out of reach (no PMU).

## 26. GP-LANES: the geometric heads a row a lane; NEXT-2's survey (2026-10-24)

After LOSS-ONCE, NEXT-2 asked how close the GEO step now sits to what this host can do
([measurements/2026-10-24/gp-lanes/](measurements/2026-10-24/gp-lanes/README.json), `survey.json`). The cores
sustain 221-230 GFLOP/s of FMA a thread with every operand in registers, on one thread as on four (one four-thread
reading 198): no clock drop under load. Four threads each running the k-blocked GEMM on operands of its own reach
x0.86-x0.98 of four times one thread on the T256 step's shapes (three readings lower, x0.69-x0.81). The pool's one
GEMM over four threads reaches x0.85-x1.16 of that on the forward and input-adjoint shapes and x0.65-x0.72 on a
weight adjoint alone, whose three to nine tasks each run the whole depth -- but the step never runs one alone. As
the trainer issues them, the step's GEMM jobs take x0.86-x1.20 of the time four threads take each running the whole
job on operands of its own: the forward's groups faster than the copies, the lone products 1-20% slower (the 1,024 x
128 x 128 one the most) and the pairs and chains 3-17%. A job of trivial tasks costs the pool 1.2-4.8 µs back to
back, and 34-35 µs once its workers have gone to sleep. Timed a worker at a time through the whole step
(`probe_jobs.py`), the workers enter every job within 0.9 µs at T256, except the step's first after the host's pause
between steps (48-54 µs), and are 66-97% busy inside it; the GEMMs take 56% of the pool's time at T256, 48% at T512
and 37% at T1024, the attention 21%, 34% and 50%. What the elementwise jobs lose is inside their tasks: RoPE's job,
for one, takes 34 µs at T256, its cosines and sines from libm every call.

Measured and not taken: GROUP-ROWS-0 (the forward's group jobs dealt a row range at a time, so a worker's block of
them covers the rows its blocks of the norm before and the RoPE after cover) reads x0.965, x1.015 and x0.978 from
T128 to T512, each holding 1, every run's weights the same digest: the rows that move between cores cost the step
nothing this rig resolves. COMMIT-BLOCKS-0 (the update's commit in chunks of 16,384, four times as many tasks) holds
1 at T128 and T256; the commit streams some 14.7 MB a step, and its job read 275-302 µs against 322-385 in this
survey's two runs a chunking but at parity in the exploration's. ROPE-TABLE-0 is the clearest of what the step has
left: the rotation reading a table of the same cosines and sines takes x0.22-x0.31 of `bcir_tensor_rope_pool`'s time
under clang and x0.33-x0.48 under GCC from T256 to T2048, every value the same -- some 2% of the step at T256 and
T512 -- but the table needs a span of its own in the plan, on both rails, so it is left for its own slice.

**The study's geometric heads.** NEXT-1's profiles placed the heads' own pair kernels at 5.8-11.1% of their
inference and 6.7-12.0% of their training (§25, `../loss-once/survey.json`). The forks run `gp_pair_forward` and
`gp_pair_backward` (GP_RANK 22: 231 pairs a row) and query_full's `gp_query_contract_pair` and
`gp_query_contract_vjp` a row at a time, a task a row; each row's norm and contractions are double-precision chains,
the norm 231 dependent additions, so one row at a time they run at the adder's latency.

**GP-LANES.** `gp_lanes_overlay.py` runs them sixteen rows at a time, a row a vector lane: every per-row float a
16-lane vector, every per-row double two 8-lane halves, each scalar operation the one intrinsic that performs it --
a rounded product, sum, difference, quotient or square root, a widening or a rounding conversion -- in the fork's
order, no product fused. Each lane therefore performs its row's IEEE operations in the scalar kernel's order, and
every feature and adjoint is the fork's bit for bit. A row the fork sends to its FP64 fallback
(`wedge_ill_conditioned`) is flagged in its lane and recomputed by the fork's own kernel. Where the CPU has AVX-512F
the forks' jobs take a task a group of sixteen rows, each prefetching the next group's rows (GP-PF-0: on outputs
made cold, x0.68-x0.95 of the lanes' time without it); elsewhere, and for the rows past the last whole group, the
forks' tasks as they were.

In isolation the lane kernels take x0.54-x0.80 (the forward), x0.26-x0.28 (the backward), x0.41-x0.49 (the query
contraction) and x0.44-x0.72 (its adjoint) of the fork's time a row, clang and GCC, operands L2-resident and
streaming (`gp-lanes.json`). The forward is the one bound by memory: it writes 924 bytes of features a row, and on
one L1-resident group computes them in 121-131 TSC cycles a row against the scalar kernel's 518-539. In the models,
48 rounds in one process with every trained state and logit the same (`study-ab.json`), against once_level3.py's
builds:

| model | training T128 / T256 / T512 | inference T1024 / T2048 |
|---|---|---|
| `global` | 0.929 / 0.924 / 0.972 | 0.990 / 0.958 |
| `query_full` | 0.947 / 0.915 / 0.940 | 0.953 / 0.938 |

Resolved: `global`'s training at T256 (0.893-0.956) and inference at T2048 (0.940-0.976), and `query_full`'s
training at T256 and T512 and inference at T2048 (0.924-0.953). The T128 rows and `query_full`'s inference at T1024
meet their A/A controls; `global`'s training at T512 and inference at T1024 hold 1. In the sampled profiles the
heads' GP kernels fall from 8.5% to 3.9% of `global`'s training at T256 and from 13.9% to 6.5% of `query_full`'s at
T512, and from 6.8% to 5.4% and 11.2% to 6.7% of their inference at T1024 (`profiles.json`). Against PyTorch at
fixed shapes (`study-vs-pytorch.json`), the geometric heads with the lanes train in x0.194-x0.325 of Inductor's time
(x0.22-0.33 with §25's overlay alone) and x0.09-0.16 of eager's, and run inference in x0.41-0.57 of Inductor's
(x0.45-0.61) and x0.21-0.24 of eager's.

Witnesses: `gp_lanes_check.c` holds the four lane kernels to the fork's, which `gp_extract.py` copies verbatim from
the level-2 build, on every output byte for byte, over six kinds of row -- uniform, near parallel (8,189 of 24,576
rows through the fallback), tiny, zero, large and sparse -- under clang and GCC. `gp_tasks_check.c` loads both
builds of each fork side by side and calls their exported entries on row counts 1, 15, 16, 17, 66, 111, 512 and
1,024, on one thread and a pool of four: 48 and 80 outputs, none apart. `red_lanes.py` injects five faults (the
forward normalised in float, a fallback row kept from the lanes, the backward's tail a row late, the contraction
rounded with a term too many, an adjoint's first term scaled by 1 + 2^-23), each caught. Three mutations it records
change no output on any input, and say what the witnesses cannot see: the norm's square of a widened float is exact
in double, so fusing it with the sum rounds the same; two updates of distinct accumulators commute; and a lane whose
query norm is at most the epsilon never reads the dot it skips.

Not claimed: the overlay edits the study's code, not BCIR's runtime -- the forks are the study's, their kernels its
contract, and the lanes are offered to it as a measurement; a host without AVX-512F runs the forks' tasks; random
rows cannot see a perturbation of a double intermediate below float's rounding, so the bit-identity rests on the
statement-for-statement correspondence, which the witnesses check wherever an output can show it; where two NaN
operands meet, which payload survives is the compiler's choice in either form (every row the witnesses run is
finite); counter-based figures (no PMU on this VM).

## 27. ROPE-TABLE and ROPE-BOUND: the rotation's cosines and sines once a forward, its loop at its rate (2026-10-24)

NEXT-2's per-job probe put each of RoPE's jobs at 34-36 µs at T256, twelve a step at the GEO dimensions: the
forward's queries and keys and the backward's adjoints of both, in each of three layers (§26). Each rotation
evaluated libm's `cos` and `sin` for every position and pair of the head -- 8,192 calls at T256 -- on the same
arguments in every rotation of the step, and turned each of the batch's heads with them. ROPE-TABLE evaluates them
once a forward ([measurements/2026-10-24/rope-table/](measurements/2026-10-24/rope-table/README.json)).

`bcir_tensor_rope_table_pool` fills a table of time x head-dim floats, each position's half of cosines then its
half of sines, with the expressions the rotation evaluates: each pair's frequency the same `pow`, each angle the same
product in double, each entry the same libm call rounded to float. `bcir_tensor_rope_lent_pool` rotates with the
table's entries -- the sine times -1 under inverse, as the per-call rotation rounds it, then the same two rounded
products and one rounded sum or difference an element -- so every rotated value is `bcir_tensor_rope_pool`'s bit for
bit. The per-call entry stays, and the PR818 study's attention model calls it. The table is the plan's: the scratch
arena's last span, which `bcir_decoder_make_plan` and the step program (`native_program.py`'s buffer `rope`) both
count, so `train.plan.mismatch` holds the two together (32 KiB at the GEO dimensions at T256). Each forward fills
it after the embedding, before its first layer, and every rotation of the step reads it: the forward's, the
backward's inverse ones, and a recomputed layer's. The backward's workspace ends where the table begins, and nothing
else writes it. On the program rail a ROPE record reads the table as its second operand, and a new record,
ROPE_TABLE (kind 24, the pack's claim `train.rope.table`), writes it at the plan's base; the checker holds the
rotation's table to time x dim floats it may read and the fill's to time x dim floats it may write, an even dim and
no unused word.

**ROPE-BOUND.** Measured on its own (`design.json`), ROPE-TABLE's first commit read its table rotation at
x0.47-0.65 of the per-call one's time under clang, where ROPE-TABLE-0's prototype (§26) read x0.22-0.31. The two
loops perform the same operations; only their bound differs. The prototype's inner loop ran to the head's half --
the distance between the two halves the loop pairs -- and the commit's to the range's count, min(half - k0, 64), the
same count for every head of at most 64 pairs. `bound_cause.c` times five forms that differ only in that expression.
Under clang 18 and clang 23, the two bounded by the halves' distance take 18.3-19.1 µs at T256 on one thread, and the
three bounded otherwise (the range's count, half - k0, or min(half, 64) without the ranges) 54.4-56.4. Under GCC all
five run alike: 20.8-26.6 µs, the high reading the first form's, which in six runs more took 22.1-23.2 µs beside the
others' 21.0-22.9 (`recheck.sh`). Clang enters the loop's vector code past one overlap test for all of a batch row's
heads, whose halves interleave from head to head, so the test fails and every head turns in the scalar loop; bounded
by the halves' distance, the loop needs no test between them. With one head a row the five forms run alike under
clang, 6.4-7.8 µs, and with two or four the three bounded otherwise take three times the others' time
(`bound_heads.sh`). The per-call rotation (`rope_rows`) had the same loop and, under clang, the same third of the
rate. Both rotations now turn a position's pairs through `rope_turn`, called with the head's half wherever the head
fits one range of 64 pairs -- every head up to 128 wide, so every model here -- and with the range's count otherwise.
The operations and their order are the same, and the C harness's output is the first commit's byte for byte.

Measured on the committed tree against the tree before the slice (`rope.json`, `step-parts.json`, `step.json`,
`forward.json`, `steady.json`):

| | after / before |
|---|---|
| the table's rotation / the per-call one, clang, one thread / four, T256 to T2048 | 0.281-0.288 / 0.288-0.478 |
| the same, GCC | 0.312-0.386 / 0.376-0.709 |
| the per-call rotation after ROPE-BOUND / before it, clang, one thread / four | 0.574-0.810 / 0.436-0.885 |
| the rotations and the fill in the step, a worker at a time: T256 / T512 / T1024 (µs a step) | 447 -> 190 / 815 -> 336 / 1,596 -> 764 |
| the training step in one process: T128 / T256 / T512 / T1024 / T2048 | 0.970 / 0.958 / 0.973 / 0.999 / 0.973 |
| the same, GCC T256 / T512; the 9.84M decoder | 0.992 / 0.993; 1.006 |
| the step alone in its own process: T128 / T256 | 0.948 (0.905-0.992, resolved) / 1.016 |
| the inference forward: B4 T1024 / B4 T2048 / B1 T256 | 0.988 / 1.003 / 0.930 |

The fill costs x0.69-0.76 of one per-call rotation on one thread under clang, in place of twelve. At T256 the
rotations' saving is some 1.9% of the step; in one process the step's interval there (0.931-0.985) meets its A/A
control's (0.980-1.027), and every other row of the step and the forward holds 1 or meets its control, as an effect
of 1-2% does on these rigs. Under GCC on four threads the records read the table's rotation at T256 at 8.94 µs before
ROPE-BOUND and 13.04 after. Run six times more on each unit (`recheck.sh`), it takes 8.35-9.91 µs in some runs and
15.35-16.70 in others, on either unit, and in every run of `bound_cause.c` (one thread or four, T256 or T1024) its
five forms lie within x1.11 of one another: that reading is a run's level, not ROPE-BOUND's. Every round's weights
and logits are the same bytes on both trees.

**RMS-LANES-0**, measured and not taken. The RMSNorm's jobs were the probe's next elementwise jobs.
`proto_rms_lanes.py` runs their per-row passes sixteen rows at a time, a row a lane -- the forward's scale and its
scan, the VJP's dx, the gamma adjoint's columns -- every operation the unit's in its order. Alone, with their
operands in cache, the jobs take x0.66-0.80 of their time (`design.json`); in the step, one library a process, they
read x0.953-x1.023 from T128 to T1024, each holding 1, every run's weights the same digest. The norms' jobs wait on
where their operands sit, not on their arithmetic. On non-finite input one to three of the variant's elements also
carry another NaN than the unit's: a second spelling the trainer's scans refuse either way, but not every value as
it was.

Witnesses: the C harness's `rope_table` holds the table's 201,372 entries each to the per-pair reference's own libm
call, and 504 rotations reading it byte for byte to the reference's and `bcir_tensor_rope_pool`'s -- seven shapes
(heads of 2 to 260, heads shorter than a range of 64 pairs and wider than one, positions 1 to 70 about the
16-position task), four bases, forward and inverse, the fill and the rotation each serial and on pools of two and
four -- and sees a table of another base and one read a position late. Every other line of the harness is the tree
before's, under clang 18 and GCC 13, and under ThreadSanitizer. The program checker's six new refusals: a rotation
without its table, a table of another time, a table inside the x it turns, an odd-width fill, a fill into the
weights, a fill with an unused dimension. `rope-table.json` RT1-RT16 inject the table's and the rotation's faults
into the unit and the trainer, and `rope-table-px.json` RTX1-RTX16 into the plan, the checker, the executor, the
trace, the lowering, the launch plan, the step program and the pack, each caught by its own check. One fault, as
first written, removed the table from the step program's forward rotations alone; the backward's still read it, so
its lifetime -- the one thing the step program's reads decide -- was unchanged, and no check could see it. Aimed at
the backward's rotations, its life ends before the loss, where the step program's bound peaks, and the bench row's
bound falls from 755,712 to 753,664 bytes: caught. PX-DISPATCH's PY27 is re-anchored on the new run count, and
NDT-ROPE's NO1-NO7 (NO4 re-anchored in `rope_turn`, NO6 and NO7 new) and NDT-THREADS' NT6 are caught by the harness.

Not claimed: heads wider than 128 keep the ranges' loop, at clang's slower rate; the branch is a rate, not a value --
sending every head down the ranges' path changes no output, so only the timings witness it; the v1 contract's step
at x0.935 at T256, resolved by the rule but more than the rotations' whole time a step (0.45 ms of 25) beside an A/A
control of 1.025, and the first commit's B1/T256 forward at x0.734 beside one of 1.105, the rig's own bias at those
sizes; counter-based figures (no PMU on this VM).

## 28. KSPLIT: the blocked contract's weight adjoints split over their depth (2026-10-25)

NEXT-3 re-probed the GEO step after ROPE-TABLE (§27): at T256 the GEMMs take 60% of the pool's time and the
attention 22%, and from T256 to T1024 the GEMM jobs' rate falls from 454 to 389 GFLOP/s while the attention's
share grows to 48%. A job at a time, the weight adjoints -- dW = dY^T X, a call with A transposed whose depth is
the batch's rows -- were the GEMM form furthest from MKL's: at a depth of 4,096 on four threads BCIR's took
x1.16-2.26 of `torch.mm`'s time (`design.json`, `tn_bench.py`), where the forward form, its depth the width, ran
at x0.76-1.07 of it. Under MM-ROWS (§15) a blocked call's task is a range of whole column blocks over all of the depth, so
every task walks all of A and its part of B; once the depth passes a few K blocks of 448, A outgrows a core's L2 and
each task reads it from L3 again. KSPLIT-0, the prototype, gave such a call a task a K block: the GEO step
x0.928-0.972 at T1024 and x0.945-0.981 at T2048 in its six runs, every byte equal. KSPLIT lands it
([measurements/2026-10-25/ksplit/](measurements/2026-10-25/ksplit/README.json)).

**The split.** A blocked-contract call with A transposed and B not, at least four K blocks deep
(`bcir_tensor_kblock_split`'s blocks; a build may set `BCIR_MM_KSPLIT_BLOCKS`), runs a pool task a K block where
the room it is lent holds its blocks' partial products, blocks x m x n floats. A task computes its block's
product over all of C with the instance's own register kernel, the accumulators stored as they are -- a new
epilogue of each blocked kernel (`BCIR_MM_RAW`: no alpha, no beta, C never read; masked stores at the portable,
AVX2 and AVX-512 instances' edges) -- into its slice of the room. A fold job, sixteen rows of C a task, then gives
every element its blocks in order with the blocked kernels' epilogue operations: the first block's beta scaling
(NDT-BETA: +0.0 under beta 0, else the rounded product) and c + alpha*P, a rounded product and a rounded sum, then
c + alpha*P for each later block. Each block's partial is the nest's -- its fmaf chain from zero in ascending
depth -- and the fold performs the nest's epilogue in the nest's order, so every element of C is the nest's bit
for bit; only where each block's product waits differs. A job's calls take the room in call order while their
partials fit, a group's independent calls and never its chain (a chain's calls share one tiling), and the rest
run as before; an entry lent no room, or one that cannot hold floats, runs every call as before. The pair entry
gains its room (`bcir_tensor_mm_pair_ws_variant_pool`); the single and group entries take the workspace they
already took.

**The room.** The trainer lends the state's workspace -- under a binary32 contract the attention's copies
(ATT-CONTIG), idle while a GEMM job runs -- to every blocked-contract GEMM job with a weight adjoint, and the unit
splits the calls whose partials fit what is left of it. No plan grows: where the state lends no workspace (under
the blocked contract, contexts below ATT-CONTIG's 1,024 positions) nothing changes. The Python trainer lends the
exact need at its thread count (`bcir_decoder_workspace_need`): at the GEO dimensions on four threads 4 MiB at
B4/T1024 and 8 MiB at T2048, where the largest group's partials take 2.5 and 4.75 MiB, so all 22 weight adjoints a
step split; the 9.84M decoder's 5.75 MiB at B8/T1024 holds one weight adjoint a layer (the keys', 5.3 MiB) of
groups whose partials take up to 57 MiB. A caller lending the workspace for any pool
(`bcir_decoder_workspace_bytes`: 16, 32 and 92 MiB there), as `step_ab.c` does, has every one split (`sizes.json`).
On the program rail every record of such a job states the workspace (slot 3, the state's whole workspace), the
checker requires it there and nowhere else under the contract, the executor lends it by the same rule
(`ks_lends`), the trace records it, the launch plan states it, and the pack's claims carry the workspace the job
writes and its lowering gives it back.

Measured on the committed tree against the tree before the slice (`gemm.json`, `step-parts.json`, `step.json`,
`forward.json`):

| | after / before |
|---|---|
| the weight adjoint alone, GEO shapes, 4,096 and 8,192 rows (10 and 19 K blocks), four threads | x0.56-0.78, every one resolved |
| the same at 2,048 rows (5 K blocks), four threads | x0.68-0.96, three of eight holding 1 |
| the same, 4,096 and 8,192 rows, one thread | x0.66-0.88 |
| the 9.84M decoder's 192- and 384-wide weight adjoints, 4,096 and 8,192 rows, four threads | x0.66-0.91 (one thread x0.69-0.91) |
| its 1,024-wide one (the down projection's), four threads | x0.905-0.98 |
| its 1,024-wide one at 8,192 rows, one thread | x0.88-1.14: x1.00-1.14 in the recheck's six runs, five resolved slower |
| at 1,024 rows (three K blocks: not split, the same path either way) | x0.94-1.03, one interval of twelve excluding 1 |
| the GEO step's split GEMM jobs a worker at a time (the Python trainer), T1024 / T2048 | 21.3 -> 17.9 ms a step (x0.84) / 46.0 -> 36.9 ms (x0.80) |
| the training step, GEO T1024, clang / gcc | x0.950 (0.930-0.970), resolved / x0.941 (0.913-0.970), meeting its control |
| the same at the Python trainer's exact need | x0.968 (0.938-0.998), meeting its control |
| GEO T2048, five 64-round runs (clang and gcc twice; clang at the exact need) | x0.959-0.971, each excluding 1 and meeting its control |
| the 9.84M decoder, B8/T1024, every weight adjoint lent room / at the exact need | x0.938 (0.912-0.966), resolved / x1.008, holding 1 |
| one thread: GEO T1024 / the 9.84M decoder T1024, every weight adjoint lent room | x0.967-x0.988 / x0.975 (clang), x0.957 (gcc, resolved) |
| where nothing splits: GEO T512, the v1 contract, the 9.84M decoder T128 and T512; the forward | x0.975-x0.997 and x0.996-x1.014, each holding 1 |

The gain is where the probe placed it. Alone, the GEO step's weight adjoints over 4,096 and 8,192 rows take
x0.56-0.78 of their time on four threads, at 316-448 GFLOP/s where the nest ran 198-298, and against `torch.mm`
beside them x0.74-1.31 of MKL's time at 4,096 rows and x0.82-1.11 at 8,192, where the nest took x0.98-2.03 and
x1.44-1.87: at MKL's rate, near enough, from twice its time. A worker at a time, the GEO step's two GEMM claims
that split -- the backward's groups and pairs, each split job now followed by its fold -- take 3.4 ms less a step at
T1024 and 9.1 ms less at T2048, the GEMM claims that do not split as they were; the step in one process reads
x0.950 at T1024, resolved, and the five 64-round runs at T2048 x0.959-0.971, the 3.8% the probe's 9.1 ms predicts,
each meeting its control's interval, so not resolved by the rule. The first T2048 runs, 24 rounds a compiler,
read x1.005 and x0.9995, their rounds swinging by a tenth and clang's control excluding 1; the 64-round runs were
made for them, and pooled with them each compiler's 152 rounds read x0.970 and x0.971, still meeting their
controls. The 9.84M decoder's step at T1024 reads x0.938 where every weight adjoint has its room -- 89 ms a step --
and x1.008 at its exact need, where one a layer has.

What remains is made of two things. The width: the split's rate falls as the weight adjoint widens. At 8,192 rows
and n 384 (`sweep_m.sh`) its time is x1.11-1.24 of MKL's from m 256 to 640 on four threads and x1.68-1.80 at 1,024
(one thread x1.11-1.30 and x1.91-2.10), and on one thread its gain over the nest fades from x0.62-0.84 at m 512
and below to x0.92-0.96 at 768 and x0.98-1.11 at 1,024: a K block's panel of A, 448 x m floats, grows past what a
core's L2 holds beside B's (1.75 MiB at m 1,024, against 2 MiB), so on one thread the split reads A from L3 as the
nest does and adds the partials' traffic, while four threads still gain from reading each block once in all. That
is the one-thread regression's whole cause: the 9.84M decoder's 1,024-wide weight adjoint at 8,192 rows. The
trainer lends it no room on one thread (1.44 MiB against its 28.5 MiB of partials), and where a caller does, the
one-thread step still reads x0.957 and x0.975. A K block's task over a range of rows where its panel would not
stay in L2 is the next lever (KSPLIT-ROWS). The room: where the state lends none -- the GEO step below 1,024
positions, whose 2,048-row weight adjoints run x0.68-0.96 of their time split alone, and the 9.84M decoder at its
need -- nothing splits; a room planned for the partials, or a window of blocks a fold at a time, would take them
(KSPLIT-ROOM). After the split the GEMM claims are 24% of the GEO step's pool time at T2048 and the attention's two
claims 67%.

Witnesses: the C harness's `gemm_kblock_split` holds 990 weight adjoints through the single, pair and group entries
byte for byte to the K-blocked fmaf chains -- three to ten K blocks across kblock_next's tail rule, rows across
each instance's row blocks and columns across its edge widths, beta 0 and 0.5, alpha 1 and -0.75, every instance,
serially and on pools of two and four -- 432 of them split, the room written exactly where a call qualifies and
its partials fit (the whole room, one call's partials) and untouched otherwise (none, a float short, misaligned,
three blocks deep), a group's second weight adjoint from the next 64-byte offset, and never a byte past what was
lent. The harness's `-contig` build splits every blocked weight adjoint whatever its depth
(`-DBCIR_MM_KSPLIT_BLOCKS=1`, 2,726 splits), so each of its trainer witnesses runs split, and its trainer digests
are the default build's; the ThreadSanitizer build splits them too. `test_a_blocked_weight_adjoint_job_is_lent_the_workspace`
runs a long-context case (B2/T1024, five K blocks) through the C step's trace, every program run and every pack run
of the planner's choices, and a weight adjoint run as a group's chain of one, each the C step's state, and the state
a build that never splits trains; `test_the_plan_lends_a_blocked_weight_adjoint_job_the_workspace` holds the step
program's loan under each job shape and contract; `test_a_blocked_job_states_its_workspace_or_is_refused` holds the
checker's five new refusals, and a short context's blocked program, stating no workspace, runs. `ksplit.json`
KS1-KS19 inject the unit's faults and `ksplit-px.json` KX1-KX14 the trainer's, the checker's, the executor's, the
trace's, the launch plan's and the pack's, each caught by its own check; KS9, KS16 and KX7 as first written stopped
the build and were rewritten to build. GEO-3's GW8, MM-ROWS' MR5, NDT-KBLOCK's NK5 and NDT-THREADS' NT14 are
re-anchored on the new entries and caught again. Clang 23's analyzer found the executor's pair reading its second
call's flag at a fixed count of two; it takes the job's count, which the checker holds to two (KX5 re-anchored,
caught). In every timed arm the split's bytes are the nest's, and every round's weights and logits the same bytes on
both trees.

Not claimed: the split where the state lends no workspace -- the blocked contract's shorter contexts and the 9.84M
decoder's wider weight adjoints at its need -- whose room a later slice could plan; the wide weight adjoints' rate
(KSPLIT-ROWS); T2048's gain by the rule, every run of it meeting its control; the fuzz harness's state lends no
workspace under the blocked contract, so the checker's new rule is the witnesses' and not the decoder campaign's;
counter-based figures (no PMU on this VM).

## 29. KSPLIT-ROOM: the room the weight adjoints split in, planned (2026-10-25)

KSPLIT (§28) split a blocked weight adjoint over its depth only where the state happened to lend a workspace -- under
a binary32 contract the attention's copies, from 1,024 positions -- so the blocked contract's shorter contexts ran
every weight adjoint as the nest, and the 9.84M decoder's 5.75 MiB at T1024 held one weight adjoint a layer of the
57 MiB its largest group's partials take. NEXT-4 re-probed after KSPLIT on the tree before
([measurements/2026-10-25/ksplit-room/](measurements/2026-10-25/ksplit-room/README.json), `explore.json`). At three K
blocks -- GEO T256's 1,024 rows -- the split alone takes x1.01-1.69 of the nest's time on four threads, so the
threshold of four stands. Lent the room, the same trainer's step read x0.917-0.922 at the corpus recipe's decoder
(BCIR-Corpus-860K's B8/T256, the train rail's shape), x0.932-0.937 at the GEO dimensions' B4/T512 and x0.925-0.939
at the 9.84M decoder's T512, every byte equal: the room, not the rule, was the step's lever. KSPLIT-ROWS, a K block's
task over a range of rows (for the widest weight adjoints' panels and the five-block calls' four workers), took some
weight adjoints alone to x0.62-0.89 of the nest's time but read x0.933-0.951 in the step against the committed
split's x0.917-0.937, and x0.994 at the 9.84M decoder's T1024 with both lent every room: measured, not taken.

**The room.** The tensor unit states the room a call takes by the rule its entries take a slice by
(`bcir_tensor_mm_ksplit_bytes`: the call's K blocks' partials, blocks x m x n floats, rounded to the next 64 bytes,
from `BCIR_MM_KSPLIT_BLOCKS` blocks; 0 for a call that does not split or a figure past `SIZE_MAX`). One step,
`mm_ksplit_step`, serves both, so the plan cannot reserve a room the split would not take. The trainer's need under
the blocked contract is the most of the attention's copies and the backward's largest GEMM job's room, the sum of its
weight adjoints' slices in the order the job's calls take them: the head's pair, the gate and up group, or the q, k
and v group (the down projection's pair takes a slice of the gate's shape and the output projection's one of q's, so
neither is ever the most). `bcir_decoder_workspace_unsplit` is the need without the room -- the copies, and under
bf16 the need, whose GEMMs do not split. `native.memory_plan` offers each recompute set at the need and then without
the room before the next set, so a budget gives the room back before it re-runs a layer.

| shape (blocked contract) | the need on four threads, before -> after | weight adjoints split a step |
|---|---|---|
| corpus decoder B8/T256 (2,048 rows, five K blocks) | none -> 2.50 MiB (the head's pair) | 0 -> 29 |
| GEO B4/T512 (2,048 rows) | none -> 1.25 MiB (the gate and up group) | 0 -> 22 |
| GEO B4/T1024 and T2048 | 4.00 and 8.00 MiB, unchanged (the copies hold the room) | 22 and 22 |
| 9.84M decoder B8/T512 (4,096 rows, ten K blocks) | none -> 30.00 MiB | 0 -> 43 |
| 9.84M decoder B8/T1024 (8,192 rows) | 5.75 -> 57.00 MiB | 6 -> 43 |
| 9.84M decoder B8/T128 (three K blocks) | none, unchanged | 0 |

Measured on the committed tree (`sizes.json`, `step.json`). The slice's reading is the room alone: one library,
the state lent the need without the room against the need, every instruction the same.

| the step, the room alone (one library) | clang | gcc |
|---|---|---|
| corpus decoder B8/T256, four threads | x0.906 (0.878-0.935), resolved | x0.923 (0.894-0.954), meeting its control |
| GEO B4/T512, four threads | x0.922 (0.898-0.946), resolved | x0.923 (0.901-0.945), resolved |
| 9.84M decoder B8/T512, four threads | x0.941 (0.918-0.964), resolved | x0.939 (0.916-0.963), resolved |
| 9.84M decoder B8/T1024, four threads | x0.928 (0.907-0.949), resolved | x0.949 (0.933-0.965), meeting its control |
| GEO T1024 and T2048, the two figures equal | x0.998, x0.997, holding 1 | x1.040, x0.967, holding 1 |
| one thread: the corpus decoder | x0.947 (0.927-0.966), resolved | x0.972 (0.952-0.993), meeting its control |
| one thread: GEO T1024, the 9.84M decoder T512 | x0.945, x0.984, holding 1 | x0.979, x0.979, holding 1 |

The two trees against each other read the same way where the room applies -- x0.918-0.953 on four threads,
x0.952-0.974 on one -- but that run's own controls, GEO T1024 and T2048, where both trees lend the same workspace
and split the same calls, read x0.952 and x1.003 at T1024 and x0.905 and x0.938 at T2048, the last two excluding 1
by the rule. Run again with the trees' places swapped they read x0.973-1.023, every one holding 1: the run's swing at
T2048 (KSPLIT's records saw it swing by a tenth round to round at 24 rounds), not a difference between the builds --
and the reason the one-library run, whose controls hold 1, carries the slice's reading. Every round's weights are the
same bytes on both trees and both arms.

The gain is what NEXT-4 placed: some 5-9% of the step on four threads wherever the room makes the weight adjoints
split, at a memory cost of the largest job's partials -- 2.5 MiB of the corpus decoder's 140 MiB plan, 1.25 of GEO
T512's 118, and 30 and 57 MiB of the 9.84M decoder's 1.2 and 3.0 GiB at T512 and T1024: 1.1-2.5% of each plan, which
the budget's tier gives back before it re-runs a layer. On one thread the gain is smaller and mostly within the
controls, as KSPLIT's one-thread rows were.

Witnesses: the C harness's `ksplit_room` holds the unit's rule to a statement written apart from it at five depths
across `kblock_next`'s three-to-four-block edge and its refusals (no depth, no rows, a product past `size_t`); the
need, the need without the room and every other contract's need at three specs, each of the three jobs that can be
the largest once (the gate and up group at the fixture's dimensions, the head's pair at a vocabulary of 512, the q,
k and v group at a hidden width of 8 under a model width of 32 with two KV heads of four); and a step lent the need,
a byte short of the room and none, the same state bytes, splitting 15, 13 and 0 weight adjoints under the probe
(`bcir_tensor_ksplit_probe`, the instance probe's builds). The Python witness,
`test_the_blocked_contract_plans_the_room_its_weight_adjoints_split_in`, holds the Python trainer's loan at 1,408
rows (four K blocks, no copies) at one thread and four and for any pool, every contract's need and need without the
room, `memory_plan`'s order at a long context whose copies fall short of
the room (a vocabulary of 2,048 at T1024: the need, the copies, nothing, a set at a time), the budget that fits only
the copies training on them, and one state under every loan -- the need, the room less a byte, less than any weight
adjoint's room, none -- at one thread and four, the probe counting every weight adjoint, all but one a layer, none and
none. The KSPLIT launch witness gains the short context: its trace, every program and pack run of the planner's
choices and a never-splitting build, each the C step's state. `ksplit-room.json` KR1-KR15 inject the plan's, the
unit's, the probe's and `memory_plan`'s faults, each caught by its own check; KR8 as first written stopped the build
and was rewritten to build. KSPLIT's KS12 and KS13, ATT-CONTIG's AC20 and NDT-BF16's NB14 are re-anchored on the
changed lines and caught again.

Not claimed: a gain below four K blocks, where the split does not pay; the widest weight adjoints' rate
(KSPLIT-ROWS, measured and not taken); a room under the bf16 contract, whose GEMMs do not split; any figure on
another host; counter-based figures (no PMU on this VM).

## 30. NEXT-5 and Q8-LANES: the train rail's step at its bounds; the C decoder's matrix kernels in vector lanes (2026-10-25)

NEXT-5 re-probed the train rail's binary32 step after KSPLIT-ROOM on the tree before
([measurements/2026-10-25/q8-lanes/](measurements/2026-10-25/q8-lanes/README.json), `explore.json`). The step is
the corpus recipe's decoder (BCIR-Corpus-860K, B8/T256, the blocked contract, four threads): 37.2 ms of pool jobs.
Each part was read against its own bound:

- **The GEMMs**, 22.0 ms (59%). Split by task count, every large GEMM job is 94-98% busy; the 81% the first probe
  showed for the chain jobs was a mean over their small fold jobs. Alone, the step's forward and input-adjoint
  shapes run at 474-613 GFLOP/s on four threads (the 64-wide projection 333) and 137-157 on one, 63-72% of a core's
  217 GFLOP/s FMA rate (ATT-BOUND's); its weight adjoints at 357-404 and 108-132. The k-blocked kernel's inner loop
  alone in L1 runs at 87-90% of its FMAs in registers (`kloop.c`, clang).
- **The attention**, 6.7 ms (18%). A probability costs within 9% of the long context's at T256 on one thread. In
  the step it runs at 87-92% of four times its one-thread rate, the forward at 91-95% and the backward at 80-87% of
  v1's arithmetic bound (the scores and the context 1.44 TSC cycles a probability each).
- **The loss**, 3.3 ms (9%). libm's `exp` takes 8-10 TSC cycles a call, some 70% of the job, and the row maximum
  about 15% (a timing build without it read x0.853). LOSS-MAX, the maximum in vector lanes with every value as it
  was, took the loss job to x0.889 with eight lanes (AVX2) but x1.066 with sixteen (AVX-512, its bursts among the
  scalar libm calls): about 1% of the step, measured and not taken.
- **The rest.** The elementwise jobs take 5.1 ms, and 0.6 ms of the 37.9 a step is spent outside the pool
  (`wall_gap.py`). The attention's thresholds at 256 positions (`BCIR_ATT_STRIP_FROM`, `_BAND_FROM`,
  `_CONTIG_FROM`) hold 1 at the corpus decoder and at GEO B4/T512 (x0.982-1.017). Under the bf16 contract the step
  is 25.7-27.5 ms, the loss 14-15% and the attention 25% of it. The stored forward's collapse after a 128 MiB
  eviction (`att_cold*.c`) is the instrument's: the step's attention runs at its warm rate.

So no bit-identical lever above about 1.5% remains in the binary32 step at this shape. The model the rail trains is
qualified on the standalone C decoder (`bcir_llama.c`, through `bcir_ai_kernels.c`), and there the qualification's
own figures put the BCIRQ8 matvec and row dots at 95% of the instrumented request: they ran a scalar double term
at a time.

**Q8-LANES.** `bcir_ai_kernels.c`'s BCIRQ8 matvec and row dots now run on three instances chosen at run time on x86-64
Linux under GCC or Clang (`__builtin_cpu_supports`): 1 the portable loops, as they were; 2 AVX2; 4 AVX-512. Each
output's chain is the portable loop's own -- from +0.0, a term an input coordinate in increasing order, the input
times the code times its group's scale (a code times a power of two: exact), rounded as a product and then as a sum,
never fused (every build of the unit passes `-ffp-contract=off`, and the intrinsics spell the multiplies and the add
apart). The vector instances only run several outputs' chains side by side:

- **the matvec** puts a row's columns in the lanes, a run of one group at a time (`q8_run_avx2`, `q8_run_avx512`).
  On AVX-512, where the output count is a multiple of 8 and the group a power of two of at least 8,
  `q8_matvec_block512` keeps 64 columns' sums in registers over every input row, a vector's eight columns always in
  one group and the scale computed where the row's group changes -- with no local array: GCC 13's AddressSanitizer
  fake stack gives a 64-byte-aligned local array 32 bytes, and the realigned frame's `vmovapd` faulted;
- **the head's row dots** take eight rows a vector: their codes transposed eight by eight (`q8_transpose8`), a
  run of coordinates every row keeps in its present group at a time, a short block's lanes masked at the store; a
  non-finite total zeroes the output, as the portable loop leaves it;
- **a group's scale** is ldexp's value, built from its bits where 2^e is a normal double and ldexp's own elsewhere.

`bcir_ai_q8_instances()` is the instances the host admits, as bits (1 always), `BCIR_AI_MAX_INSTANCE` caps them and
`BCIR_AI_NO_DISPATCH` compiles them out; `bcir_ai_q8_matvec_f64_instance` and `bcir_ai_q8_rows_dot_f64_instance` run a
named one (0: the widest) and refuse one the host does not admit, writing nothing; the prevalidated entries -- the
decoder's -- run the widest. The Python binding's `q8_matvec` / `q8_rows_dot` take `instance=`. Valgrind hides
AVX-512 from its guest, so the AVX2 instance is the one memcheck and callgrind run.

**The portable instance.** The records' first kernel A/B (`kernels-b444807.json`: the tree before's unit, its entry
points renamed, against the tree after's in one process) read instance 1 at x1.17-1.28 of the tree before's time
under clang. Inside the dispatching entry the vector instances' code took the portable loop's registers, and clang's
matvec reloaded three of its bases from the stack every term. The decode A/B's portable arm hid it
(`decode-b444807.json`: a build capped at 1, where the compiler drops the vector code, holding 1), so only a default
x86-64 Linux build running instance 1 -- a host without AVX2, or `instance=1` -- paid it. The second commit makes the
portable loops the tree before's prevalidated entries byte for byte, as static functions never inlined; their
instructions are now the tree before's under both compilers, every one but GCC's `endbr64` and alignment padding
(`portable.json`: 168 and 152 under clang, 187 and 151 under GCC, none differing). Clang's portable instance holds 1
or better (x0.91-1.03). GCC's matvec reads x1.05-1.23 at its default loop alignment and x0.93-1.06 with loops
aligned to 32 bytes, in either link order: the same instructions placed differently.

Measured on the committed tree against the tree before (one host; `kernels.json`, `decode.json`, `ops.json`,
`bits.json`, `qualification.json`):

| kernel, x of the tree before's time | clang AVX-512 | clang AVX2 | gcc AVX-512 | gcc AVX2 |
|---|---|---|---|---|
| matvec 128 x 128 (q, o) | x0.175 | x0.210 | x0.176 | x0.246 |
| matvec 128 x 64 (k, v) | x0.172 | x0.206 | x0.179 | x0.247 |
| matvec 128 x 344 (gate, up) | x0.177 | x0.214 | x0.211 | x0.265 |
| matvec 344 x 128 (down) | x0.169 | x0.211 | x0.170 | x0.242 |
| row dots 1,024 x 128 (the tied head) | x0.221 | x0.248 | x0.283 | x0.330 |
| matvec 2,048 x 256 (TinyLlama's k, v) | x0.160 | x0.205 | x0.160 | x0.225 |
| matvec 2,048 x 2,048 to 5,632 x 2,048 | x0.272-0.347 | x0.225-0.277 | x0.310-0.373 | x0.259-0.328 |
| row dots 32,000 x 2,048 | x0.330 | x0.283 | x0.376 | x0.433 |

Each row resolved against its A/A control (`kernels.json`: one process, every output the tree before's bytes); the
portable instance holds 1 under clang (x0.91-1.03) and reads as above under GCC.

| the standalone C decoder, 200 tokens of the lab model | clang | gcc |
|---|---|---|
| a generated token, before -> after | 1.048 -> 0.298 ms | 0.980 -> 0.302 ms |
| a generated token, x of before | x0.280 (0.271-0.288) | x0.316 (0.297-0.336) |
| the 200-token request, its load included | x0.313 (0.305-0.320) | x0.372 (0.355-0.390) |
| capped at AVX2 | x0.314 (0.296-0.334) | x0.353 (0.333-0.374) |
| capped at the portable loops | x1.031 (0.972-1.095), holding 1 | x0.981 (0.944-1.019), holding 1 |
| the A/A control | x0.991 | x1.023 |

Over those 200 tokens (`ops.json`, clang) the decoder's operations take 208.4 ms before and 60.8 after: the matvec
160.3 -> 31.5 ms, the head 27.9 -> 7.1, the attention 13.1 -> 14.4 (now 24% of them), the rotation 2.7 -> 3.1, the
SwiGLU 2.3 and the RMSNorm 1.3 -> 1.5. Every token and final logit of 7 builds of the tree after -- clang and gcc at
each instance cap, clang with the dispatch compiled out -- is the tree before's bytes over 2 models, 3 prompts and 4
lengths (`bits.json`).

The qualification (`qualification.json`, each tree's own `run_qualification.py --require-callgrind` on the lab
model): monolithic 52.10 -> 12.81 ms (x0.246), GEM 55.97 -> 17.11 ms (x0.306), a generated token 0.972 -> 0.234 ms
(791 -> 2,828 tokens/s), the first token 14.7 -> 5.8 ms; the logits one SHA-256 on both trees (`2a18b8613b6b60e4...`),
0 from the oracle's, and 23 of 23 checks on each. The matrix kernels were 95.3% of the instrumented request's
operations and are 78.9%. Under callgrind (the AVX2 instance) a GEM request takes 922 M instructions before and 231 M
after, 252 M and 64 M data references, with the same D1 misses. GEM/monolithic rises 1.074 -> 1.336 under its 1.5
band: GEM's own overhead (GEM less monolithic) is 3.9 ms before and 4.3 ms after, its trust boundary 3.1 and 2.9 ms.

Witnesses: the kernel harness's `q8_lanes_test` holds every admitted instance to the portable loops over 22 shapes --
the decoder's projections and head, the AVX-512 block's with its tail blocks, groups straddling rows and vectors, a
group not a power of two, groups under 8 -- at four exponent ranges (the format's, a narrow one, ldexp's subnormal
scales, past the largest double), the code 0x80 among the codes: 176 comparisons a vector instance, 352 on this
host. Non-finite outputs in a block's last lane and in a block's middle row must zero every output, refused instances
and inputs must write nothing, and the prevalidated entries must run the widest instance (`bcir_ai_q8_instance_probe`
reads the path the values cannot show). `test_native_ai_c_harness_c11_and_c23` runs it under C11, C23, C23 capped
at AVX2 and with the dispatch compiled out; `test_native_q8_instances_match_the_increasing_coordinate_oracle` holds
each admitted instance to the Python oracle's loop over nine shapes, with its refusals. `q8-lanes.json` QL1-QL22
inject the instances' faults, each caught by its own check (QL2 and QL11 rewritten to build; QL21's first form, a
term's product rounded before its power-of-two scale, differs only under double rounding below a subnormal grid, so
it was replaced). Swept again on the witnesses there were before Q8-LANES -- the harness's cases without `q8_lanes_test`, one
build, and the matrix kernels' oracle test (`without_the_cases.py`) -- 5 of the 22 are caught (QL1, QL3, QL11, QL12,
QL22: the AVX-512 paths the decoder's own shapes reach), and the other 17 only by the new witnesses.

Not claimed: a gain past the matrix kernels (the attention, now about a quarter of a 200-token request's operations,
and the rotation are the C decoder's next costs: DECODE-ROW); threads; the larger shapes' best instance -- at
TinyLlama's sizes (2,048 x 2,048 and past) the AVX-512 block's walk down 64 columns of every row strides 2-5.6 KB a
row and reads x0.27-0.37 against AVX2's runs at x0.23-0.33, which a block over rows as well (their codes held in L2
across the column blocks) would take back; any other host's figures; counter-based figures (no PMU on this VM). The
GEM ratio row rises because GEM's trust boundary is a fixed cost the faster kernels no longer hide: one GEM load
computes the 610 KB pack's CRC-32 four times, a byte at a time (`bcir_sp_validate`, `bcir_sp_verify_semantic`, the
segment and block walks), and the BCIRQ8 loader computes its body's and every tensor's bit by bit -- NEXT-6's.
