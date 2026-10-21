from experiment import *
S=json.loads((ROOT/'results/summary.json').read_text());Q=S['quality'];T=S['timing'];K=S['kernel']
NAMES={'transformer':'Native attention','grassmann':'Local Plücker','symmetric':'Local symmetric',
       'global':'Global Plücker','global_symmetric':'Global symmetric','projector':'Global ridge projector'}
ORDER=list(NAMES)
v=S['validation'];g1=T['1']['1024']['global'];a1=T['1']['1024']['transformer'];l1=T['1']['1024']['grassmann']
ab=S.get('position_ablation');ab_intro='';precision_note=''
if ab:
    ab_intro=f"An exploratory follow-up removed only rotary position encoding from the global path. Wedge recall was 100% through test length 512 and {ab['length']['1024']['global']['mean']*100:.2f}% at length 1,024, versus {S['recall_length']['1024']['global']['mean']*100:.2f}% with RoPE. This materially improves the length finding but remains a simple synthetic memory task; the no-RoPE variant was not tested on language modeling.\n"
if 'common_mode' in S:
    cm=next(row for row in S['common_mode'] if row['kind']=='global' and row['coefficient']==1000000.)
    precision_note=f"A separate common-mode stress probe tests the exact identity `wedge(u,v+c*u)=wedge(u,v)`. At c=1,000,000, native FP32 product/subtraction had maximum Plücker residual **{cm['native_max_plucker_residual']:.3g}**, versus **{cm['same_input_fp64_max_plucker_residual']:.3g}** when the same FP32 input values were used with FP64 pair arithmetic. Input quantization itself still changed the FP64 feature by L2 **{cm['same_input_fp64_feature_change_l2']:.3g}** from the unperturbed reference. This is an extreme conditioning diagnostic, not a measured frequency in the training runs. It exposes a real finite-precision limit: matching a Torch implementation of the same FP32 formula does not by itself guarantee preservation of the geometric laws. A conditioned higher-precision or orthogonalized fallback should be evaluated before claiming robust geometry on near-collinear inputs.\n"
speed_intro=f"At sequence length 1,024, batch 1, one pinned CPU core, the global model used **{g1['median_ms']:.2f} ms/update**, local Plücker **{l1['median_ms']:.2f}**, and attention **{a1['median_ms']:.2f}**. Its activation-plus-scratch arena was **{g1['transient_bytes']/2**20:.3f} MiB**, versus attention's **{a1['transient_bytes']/2**20:.3f} MiB** (**{(1-g1['transient_bytes']/a1['transient_bytes'])*100:.1f}% less**)."
if ab and 'timing' in ab:
    new=ab['timing']['1']['global_norope'];att=ab['timing']['1']['transformer']
    speed_intro=f"In the final alternating timing run at sequence length 1,024, batch 1, one pinned CPU core, the no-RoPE global wedge used **{new['median_ms']:.2f} ms/update**, versus native attention **{att['median_ms']:.2f}** ({att['median_ms']/new['median_ms']:.2f}× throughput for this fixed-update workload). Its activation-plus-scratch arena was **{new['transient_bytes']/2**20:.3f} MiB**, versus attention's **{att['transient_bytes']/2**20:.3f} MiB** (**{(1-new['transient_bytes']/att['transient_bytes'])*100:.1f}% less**)."
parts=[f'''# BCIR: native global Grassmann/Plücker experiment

2026-10-17. Reproducible research extension, not a merged BCIR feature.

The global oriented-wedge model recovered long-range recall while retaining linear training arenas. Its mean recall accuracy was **{Q['global']['recall_accuracy']['mean']*100:.2f}%**, versus **{Q['grassmann']['recall_accuracy']['mean']*100:.2f}%** for the local pilot and **{Q['transformer']['recall_accuracy']['mean']*100:.2f}%** for native attention. The matched global symmetric model reached **{Q['global_symmetric']['recall_accuracy']['mean']*100:.2f}%**. This is evidence worth investigating for this recipe and task, not a universal expressivity result.

Language modeling went the other way: global Plücker cross-entropy was **{Q['global']['lm_ce']['mean']:.4f}**, local Plücker **{Q['grassmann']['lm_ce']['mean']:.4f}**, and attention **{Q['transformer']['lm_ce']['mean']:.4f}** nats on the same new test samples. This is consistent with losing useful local detail in a single prefix summary; training and initialization choices may also contribute.

{ab_intro}

{speed_intro} These are native decoder comparisons. PyTorch is the independent correctness oracle, not the timed engine. Fixed-update timing does not establish time to a common quality target.

## What was implemented

All new model training runs through native C forward/backward kernels and BCIR's native AdamW, pool, grouped matrix operations, rotary features, RMSNorm and SwiGLU. Python constructs recipes and inputs, invokes C and evaluates logits. No model training forward/backward step runs through PyTorch.

The source is based on BCIR commit `aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d`, the unmerged PR #818 head used in the pilot. The checkout remains unchanged. `global_core.h` implements three pair features, prefix/reverse scans and explicit vector-Jacobian products. `build_native.py` produces reviewable C translation units and bounded caller-owned arena plans. The new ops have not been registered in BCIR's IR/MLIR or made to execute from a GEM+ physical plan.
''',r'''
For each token, let $z_t\in\mathbb{R}^{22}$ be its learned, rotary-positioned projection. We form a strictly earlier prefix summary

\[
s_t=\sum_{j<t}z_j,\qquad v_t=s_t/\sqrt{\max(t,1)}.
\]

The scale keeps prefix magnitudes better behaved; it cancels from a nondegenerate normalized wedge. The feature is

\[
p_t=\frac{z_t\wedge v_t}{\max(\|z_t\wedge v_t\|_2,10^{-6})}.
\]

Its 231 independent coordinates represent an oriented, decomposable 2-vector whenever nonzero. At the first token it is zero, not a valid two-plane. A shared learned projection and sigmoid gate mix it back into the residual stream. The symmetric control replaces the wedge subtraction with addition and changes nothing else.

The third mixer uses $U_t=[z_t,v_t]$ and a smooth regularized projector:

\[
\widetilde P_t=U_t(U_t^TU_t+10^{-4}I)^{-1}U_t^T/\sqrt2.
\]

It stores 253 upper-triangle entries. Its two-by-two inverse is expanded into scalars; no general matrix inverse is called. The unscaled exact full-rank projector at zero ridge is basis-independent and idempotent. Positive ridge keeps degenerate gradients finite but makes these properties approximate under general nonorthogonal basis changes. Orthogonal basis changes remain invariant in exact arithmetic. The ridge version is not falsely labeled an exact Grassmann geodesic or orthogonal retraction.

For the projector VJP, set $A=(U^TU+\epsilon I)^{-1}$, let $S$ be the symmetric matrix adjoint (half the packed off-diagonal adjoint), and account for the constant output scale. Then

\[
\bar U=2(SUA-UA(U^TSU)A).
\]

Only rank-two intermediates are needed; the C implementation does not materialize a dense projector adjoint. Wedge backward similarly applies the derivative of the clamped norm and the antisymmetric bilinear map directly.

## An important correction to the pilot report

I previously used an average of unrelated wedges to illustrate that averaging can destroy decomposability, then applied that concern too broadly to the local mixer. Its per-token wedges share the same first vector. In fact,

\[
\frac1m\sum_j\frac{u\wedge v_j}{\max(\|u\wedge v_j\|,\epsilon)}
=u\wedge\left[\frac1m\sum_j\frac{v_j}{\max(\|u\wedge v_j\|,\epsilon)}\right].
\]

Thus every nonzero local feature remains decomposable in exact arithmetic, even after the individual clamped normalizations. The old generic counterexample did not show a defect in this construction. A zero feature is degenerate, and the later arbitrary learned projection is not itself a Grassmann representation. The original pilot's measured quality and finite receptive field findings are unaffected.
''',f'''
`common_factor_check.py` checked **128 native local features**, all **7,315 four-index Plücker relations per feature**, and the factorization above. Maximum native relation residual was **{S['common_factor_erratum']['native_local_max_plucker_residual']:.3g}**; maximum difference from the factored expression was **{S['common_factor_erratum']['native_vs_common_factor_max_error']:.3g}**.

## Optimization from equations to C

A direct implementation constructs every prefix by rereading earlier vectors and scatters each prefix adjoint into all earlier inputs. That costs quadratic work in sequence length. The optimized implementation builds prefixes once, evaluates independent pair features in parallel, evaluates independent pair VJPs in parallel, then reverse-scans the prefix adjoints. If the pair VJP returns direct input gradient $a_t$ and prefix-vector gradient $b_t$, the reverse carry contributes

`dz[t] = a[t] + carry; if (t > 0) carry += b[t] / sqrt(t)`

after writing `dz[t]`, so the current token never enters its own earlier-prefix summary. Time is **O(B T r²)** for the mixer and training storage is **O(B T r²)** at fixed rank. Prefix and reverse scans themselves cost O(B T r). Pair tasks own output rows; scans own batches. No atomics, shared scatter races, heap allocation in the primitive, or attention probability matrix are required. Double scan accumulators, cached FP32 prefixes and fixed scan order are explicit numerical choices.

The direct and scan implementations share the identical pair algebra. On one pinned core, their full feature-forward-plus-backward timings were:

| Mixer | Context | Direct prefixes (ms) | Scans (ms) | Primitive speedup |
|---|---:|---:|---:|---:|
''']
for kind in ('global','global_symmetric','projector'):
    for c in ('128','1024','4096'):
        k=K[kind][c];parts.append(f"| {NAMES[kind]} | {c} | {k['direct']:.3f} | {k['scan']:.3f} | {k['speedup']:.2f}× |\n")
parts.append('''
These speedups isolate an intentionally direct mathematical translation versus its scan optimization. They are not whole-model speedups and do not establish superiority over established scan libraries.

`streaming_probe.c` adds a feature-only incremental API. Full-prefix output was bitwise identical to chunk sizes 1, 7, 64 and 513 for all three mixers. The retained per-stream state is 22 FP64 sums plus a uint64 counter: **184 bytes per layer per stream**, independent of sequence length. This excludes weights, the rest of the decoder and training arenas. Full incremental language-model decoding and streaming training are not implemented by this probe.

## Learning protocol and results

Each model has width 128, three layers, four attention heads where applicable, byte vocabulary 260, tied embeddings and the same optimizer. Wedge and symmetric variants have exactly 525,696 parameters. The projector uses FF width 250 and 526,080 parameters, **0.073% more**, rather than quietly granting it a larger model budget.

Three seeds are 20261017, 20261018 and 20261019. WikiText-2 byte training uses 600 updates, batch 4, context 64 (153,600 input tokens per run). Recall uses 400 updates, batch 8, context 128 (3,200 examples per run), with loss only on the final query. Chance recall is 6.25%. Native AdamW uses peak learning rate .001, 30-update warmup, cosine decay to .0001, betas .9/.95, epsilon 1e-8, weight decay .1 and gradient clipping 1.0. Validation at 100-update intervals chooses the checkpoint. No architecture-specific tuning was performed.

The 18 new runs comprise **9,000 native training updates**. The three prior baselines reuse their validation-selected checkpoints; their state digests were verified before evaluation. All six architectures use identical training-batch hashes per seed/task. New test sample seeds 20261020/20261021 were fixed in advance and applied equally to all models: 16,384 language-model targets and 2,048 recall examples per seed. These are new samples from the same public test split, not an independent new corpus. Baseline training was not repeated.

| Model | Parameters | Byte CE ↓ | Byte accuracy ↑ | Recall accuracy ↑ | Recall by seed |
|---|---:|---:|---:|---:|---|
''')
for kind in ORDER:
    q=Q[kind];parts.append(f"| {NAMES[kind]} | {q['parameters']:,} | {q['lm_ce']['mean']:.4f} | {q['lm_accuracy']['mean']*100:.2f}% | {q['recall_accuracy']['mean']*100:.2f}% | {', '.join(f'{v*100:.2f}%' for v in q['recall_accuracy']['values'])} |\n")
if 'recall_length' in S:
    parts.append('''
The subsequent length stress test evaluates the already-selected recall checkpoints without fitting or reselection: 128 new examples per length, shared across recipes and seeds. All were trained only at length 128. Mean three-seed accuracy:

| Test length | Attention | Global Plücker | Global symmetric | Global projector |
|---|---:|---:|---:|---:|
''')
    for c,cell in S['recall_length'].items():
        parts.append(f"| {c} | "+' | '.join(f"{cell[k]['mean']*100:.2f}%" for k in ('transformer','global','global_symmetric','projector'))+' |\n')
    parts.append('\nThis is a small out-of-training-length diagnostic, not evidence of matched language-model quality at long context. Recall loss/calibration and per-seed values are retained in the raw file.\n')
if ab:
    parts.append(f'''
## Exploratory follow-up: position encoding is part of the architecture

The main length probe motivated a separate, predeclared two-recipe ablation: remove both forward and backward rotary maps from the global input. Keep every parameter, initial weight, training batch, optimizer setting, update budget and validation rule identical. Six additional native runs add **2,400 updates**, bringing new training to **11,400 updates across 24 runs**. Independent PyTorch forward/gradient checks and exact one/four-thread state/resume checks passed for both identity-position paths. Its own protocol and source hashes are in `position-ablation/`.

At the original length 128, mean test recall is **{ab['primary_recall']['global']['mean']*100:.2f}%** for wedge without RoPE and **{ab['primary_recall']['global_symmetric']['mean']*100:.2f}%** for the matched symmetric control. Length stress results:

| Test length | Wedge + RoPE | Wedge, no RoPE | Symmetric, no RoPE | Native attention |
|---|---:|---:|---:|---:|
''')
    for c in ('64','128','256','512','1024'):
        parts.append(f"| {c} | {S['recall_length'][c]['global']['mean']*100:.2f}% | {ab['length'][c]['global']['mean']*100:.2f}% | {ab['length'][c]['global_symmetric']['mean']*100:.2f}% | {S['recall_length'][c]['transformer']['mean']*100:.2f}% |\n")
    parts.append('''
This follow-up reused the already observed recall benchmark samples, so it is exploratory evidence, not a fresh confirmatory holdout. The task places its single value in tokens 0–15 and distractors in 16–31; content pooling can solve it without general order-sensitive associative memory. No-RoPE language quality was not tested. These results support changing the positional treatment of a compressed global channel; they do not prove a generally better language model or an intrinsic advantage over every alternative recurrent/linear-attention architecture.
''')
    if 'timing' in ab:
        parts.append('\nA separate alternating, six-repetition T1,024/B1 timing run measured the identity-position variants directly (three warmup plus ten timed full updates per repetition):\n\n| Native model | One-core median [min, max] ms | Four-thread median [min, max] ms | Arena MiB |\n|---|---:|---:|---:|\n')
        for lib,label in [('transformer','Attention'),('global','Wedge + RoPE'),('global_norope','Wedge, no RoPE'),('global_symmetric_norope','Symmetric, no RoPE')]:
            one=ab['timing']['1'][lib];four=ab['timing']['4'][lib]
            parts.append(f"| {label} | {one['median_ms']:.2f} [{one['min_ms']:.2f}, {one['max_ms']:.2f}] | {four['median_ms']:.2f} [{four['min_ms']:.2f}, {four['max_ms']:.2f}] | {one['transient_bytes']/2**20:.3f} |\n")
        parts.append('\nThe no-RoPE timing model has the same arena and parameter count. Its full final state was identical across repetitions and thread counts. These timing batches are byte sequences; the numbers measure update cost, not trained long-context language quality.\n')
contrast=S['contrasts']['global_minus_global_symmetric']['lm_ce']
parts.append(f'''
The paired global-wedge minus global-symmetric language-model CE difference is **{contrast['mean']:+.5f} nats**, with a descriptive three-seed t interval **[{contrast['ci95'][0]:+.5f}, {contrast['ci95'][1]:+.5f}]**. Three seeds and this small training budget do not settle general superiority. Recall's large recipe-specific gap deserves replication with more seeds, variable positions, multiple keys and longer contexts. The symmetric/projector failures at this budget do not prove that they cannot learn recall with different optimization. The symmetric control intentionally keeps only off-diagonal coordinates to match the wedge budget; it does not represent every possible symmetric bilinear architecture. Feature scales, projector conditioning and optimizer settings could change comparisons. Optimized linear-attention and state-space models were not benchmarked here.

The projector's basis invariance can discard orientation and coordinate information useful for sequence tasks. This experiment supports treating invariance as a task choice, not an automatic improvement. A separate native-feature retrieval diagnostic uses 32 random two-planes and 3,200 transformed/noisy query bases per noise level. It is deliberately aligned to subspace invariance, so its result is a diagnostic rather than a general ML victory:

| Feature / comparison | Noise 0 | Noise .02 | Noise .10 |
|---|---:|---:|---:|
''')
for kind,sim in [('global','signed inner product'),('global','absolute inner product (orientation-invariant)'),('global_symmetric','signed inner product'),('projector','signed inner product')]:
    values=[next(row['accuracy'] for row in S['subspace'] if row['kind']==kind and row['similarity']==sim and row['noise_std']==noise) for noise in (0.,.02,.1)]
    parts.append(f"| {NAMES[kind]} / {sim} | "+' | '.join(f'{v*100:.2f}%' for v in values)+' |\n')
parts.append('''
Absolute Plücker inner product is the essential control: a sign-invariant comparison can recover orientation invariance without replacing every feature with a projector. It is not the same metric as projector distance for arbitrary pairs of subspaces.

## Whole-model performance and memory

All shapes were rerun in this session. Each repetition creates the same initial model, performs three warmup updates and times ten complete native updates. Six architectures rotate order; raw repetitions and ranges are retained. One-thread trials are pinned to CPU 0. Four-thread trials use the BCIR pool. Jobs are serialized on a shared Intel Xeon Platinum 8573C VM with an eight-core quota and 8 GiB memory limit. No GPU, bare-metal or device-driver claim is made.

| Context, batch 1 | Attention ms | Local Plücker ms | Global Plücker ms | Global projector ms | Attention arena MiB | Global Plücker arena MiB |
|---|---:|---:|---:|---:|---:|---:|
''')
for c in ('64','128','256','512','1024'):
    cells=T['1'][c]
    parts.append(f"| {c} | {cells['transformer']['median_ms']:.2f} | {cells['grassmann']['median_ms']:.2f} | {cells['global']['median_ms']:.2f} | {cells['projector']['median_ms']:.2f} | {cells['transformer']['transient_bytes']/2**20:.3f} | {cells['global']['transient_bytes']/2**20:.3f} |\n")
parts.append('''
Arena figures are exact planned activation-plus-scratch sizes, not peak process RSS. Four FP32 persistent arrays (weights, gradients and two moments) add about 8.02 MiB for the matched models. The attention baseline materializes its probability matrix; tiled/recomputed attention would change its memory comparison. Language modeling is tested at context 64; the separate recall length probe does not establish matched language-model quality at context 1,024.

Four-thread median [min, max] ms/update at context 1,024:

''')
for kind in ORDER:
    cell=T['4']['1024'][kind];parts.append(f"- {NAMES[kind]}: {cell['median_ms']:.2f} [{cell['min_ms']:.2f}, {cell['max_ms']:.2f}]\n")
parts.append(f'''
Do not interpret shared-VM scheduling noise as a hardware limit. Full final model/optimizer state was identical across repeated timing runs and one/four threads in all 30 kind/context cells.

## Correctness and limits

Six full-model masked/unmasked comparisons against independent PyTorch autograd passed. Maximum logit discrepancy was **{max(row['max_logit_error'] for kind in ('global','global_symmetric','projector') for row in v[kind]):.3g}**, maximum gradient discrepancy **{max(row['max_gradient_error'] for kind in ('global','global_symmetric','projector') for row in v[kind]):.3g}**. Primitive tests cover random, zero, collinear and near-collinear inputs. Seventy-two finite-difference probes passed. Prefix causality, exact resume, thread-state identity, all-masked refusal before state mutation, and four complete UBSan training steps for each of wedge/projector passed. Checkpoint hashes and restore digests are retained.

The FP64 geometry control verifies Plücker relations, basis sign behavior and projector invariance. The ridge projector's general-basis discrepancy was **{v['geometry']['ridge_projector_general_basis_change_max_error']:.3g}**, and idempotence discrepancy **{v['geometry']['ridge_projector_idempotence_max_error']:.3g}**, on the recorded well-conditioned sample. These are sample measurements, not worst-case bounds. Fixed rank, finite norm clamps and ridge introduce explicit numerical/modeling limitations.

{precision_note}

Tested toolchain: GCC 13.3, C11, `-O3 -ffp-contract=off -fno-math-errno`, Python 3.12.14 and CPU Torch 2.14.1. The latest-toolchain status log records unavailable LLVM/Clang/MLIR 23 and Python 3.15, plus the installed Node version. The full BCIR CI/default/latest/native-ARM/Windows production matrix was not run for this isolated research extension. It is not qualified for merging or device deployment.

## Recommended next update

1. **Preserve a local channel alongside global geometry.** The global scan solved the architectural reach limit but worsened byte modeling. Test a budget-matched local/global hybrid or multiscale gated prefix memories, with controls changing only wedge versus symmetric algebra. Keep both quality and memory targets. The positional ablation makes a separate position-free global route a concrete candidate; test its language quality and variable-length training before adopting it.
2. **Challenge the recall finding.** Randomize key/query positions, require multiple independent stored values, vary sequence length and distractor distributions, and extend the training budget equally for all controls. A single summed prefix is a compression bottleneck; passing this task does not establish associative-memory capacity.
3. **Make the algebra first-class in BCIR.** Specify typed normalized-wedge, prefix-scan, scan-adjoint and rank-two-projector operations with causality, shape, alias, degeneracy and numerical contracts. Lower them to these C kernels behind verifier and parity gates. The verified physical training plan should select and drive their execution and buffer lifetimes.
4. **Keep alternative representations.** Use orientation-sensitive wedge features where order matters; sign-invariant comparisons or projectors where basis/orientation is nuisance. For larger rank, compare factorized frames against O(r²) packed coordinates rather than assuming coordinate expansion stays cheap.
5. **Optimize the measured bottlenecks and conditioning.** Profile batched feature projection, prefix scans and matrix kernels separately; measure recomputation versus saved-prefix storage. The scan already removes quadratic prefix work. A fused projection may save traffic but can lose GEMM reuse, so select it by measurement. Add conditioning telemetry and benchmark FP64-product or QR-based fallbacks for near-collinear wedge inputs, with forward/VJP/geometry parity. Compare against tiled attention before general memory claims.
6. **Develop real streaming and accelerators next.** The 184-byte mixer-state probe is a valid starting point; it is not a full streaming decoder. Add reset/position/length contracts and incremental decoder parity, then hardware-specific scan/vector/GEMM lowering. SYCL/ARM/GPU support needs its own measured evidence.
7. **Test intrinsic manifold optimization separately.** QR/polar retractions, principal-angle objectives, tangent projection and subspace flows are legitimate further directions. None is implemented or validated by merely naming a feature Grassmannian. Add them only with a task requiring the invariant and a cheaper competing baseline.

## Sources and reproduction

- Zhang Chong, *Attention Is Not What You Need: Grassmann Flows as an Attention-Free Alternative for Sequence Modeling*, arXiv:2512.19428v1: https://arxiv.org/abs/2512.19428 . This experiment is an independent rework; it is not author-code reproduction.
- Edelman, Arias and Smith, *The Geometry of Algorithms with Orthogonality Constraints*, SIAM J. Matrix Anal. Appl. 20(2), 1998: https://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf . Established Grassmann/Stiefel geometry motivates separating a subspace from an arbitrary basis.
- Katharopoulos et al., *Transformers are RNNs*, ICML 2020: https://proceedings.mlr.press/v119/katharopoulos20a.html . Global linear-memory sequence processing and recurrent scans are established ideas; we make no priority claim for them or the independent experimental combination here.
- BCIR repository and pinned source: https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/tree/aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d . Preserve its noncommercial license in `vendor/LICENSE`.
- WikiText-2 raw splits: https://huggingface.co/datasets/Salesforce/wikitext . Original authors Merity, Xiong, Bradbury and Socher; Wikipedia contributors; dataset card CC BY-SA 3.0/GFDL. Retained source parquet files are unmodified; exact URLs and hashes are in `protocol.json`.

See `README.md` for commands. The archive includes source/generated C, source hashes, protocol, raw results, input parquets and all 42 selected checkpoints (36 from the main comparison plus six positional-ablation checkpoints). Each stage's model implementation and protocol hashes were frozen before its training; later timing/probe/report scripts have separate final manifest hashes. No paper PDF is redistributed.
''')
report=''.join(parts).replace(chr(92)+'[','$$').replace(chr(92)+']','$$')
(ROOT/'REPORT.md').write_text(report)
(ROOT.parent/'BCIR_Grassmann_Global_Report_2026-10-17.md').write_text(report)
print(json.dumps({'words':len(report.split()),'characters':len(report)}))
