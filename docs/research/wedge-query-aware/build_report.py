"""Generate the source-backed report from audited native experiment results."""
from experiment import *

s=json.loads((ROOT/'results/summary.json').read_text())
v=json.loads((ROOT/'results/validation.json').read_text())
probe=json.loads((ROOT/'results/precision-budget-probe.json').read_text())
audit=json.loads((ROOT/'results/final-audit.json').read_text())
protocol=json.loads((ROOT/'protocol.json').read_text())
models=('global','query_full','query_matched','attention')
names={'global':'Full global baseline','query_full':'Full global + contraction, FF257','query_matched':'Matched + contraction, FF250','attention':'Native attention32','hybrid':'Separate language hybrid'}
def pct(x):return f'{100*x:.2f}%'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def curve(task,m,t,pairs=0):return next(r for r in s['curves'] if (r['task'],r['model'],r['context'],r['pairs'])==(task,m,t,pairs))
def cf(task,m,t,mode):return next(r for r in s['counterfactuals'] if (r['task'],r['model'],r['context'],r['mode'])==(task,m,t,mode))
def primary(task,m,key='accuracy'):return s['primary_association'][f'{task}/{m}'][key]

p=[]
p.append('''# BCIR full global head with query-aware geometric contraction

Research implementation and controlled CPU study · 18 October 2026

**The full head and its query-aware contraction are implemented in C and tested. The contraction does not establish useful associative learning at this training budget.** The retained full wedge route is augmented with an explicit query–plane interior product. Both the original-size and nearly parameter-matched versions remain near the16-label chance level on random key/value association; attention learns single-pair value retrieval but also falls short of reliable multi-pair binding.

The result is task-dependent. This experiment tests one precise contraction, one small model scale and a declared data/optimization budget.
''')
p.append(f"**Long ordered recall does improve:** at length2048, the full-size contraction rises to **{pct(curve('ordered','query_full',2048)['accuracy']['mean'])}**, versus **{pct(curve('ordered','global',2048)['accuracy']['mean'])}** baseline; the matched version reaches **{pct(curve('ordered','query_matched',2048)['accuracy']['mean'])}**. At length1024, the matched version reaches100% on all three selected models. This supports an ordered-recall candidate, not a solved associative-memory architecture.\n")
p.append(f"The frozen study completed **{s['count']['training_runs']} from-scratch training runs, {s['count']['quality_training_updates']:,} quality-training updates and {s['count']['selected_checkpoints']} retained selected checkpoints**. Three independently trained seeds are20261018,20261019 and20261020. All checkpoints restore with their recorded model/optimizer digests, and all paired input/source checks pass. Each task trains separate weights; this is not one jointly trained model.\n")
p.append('## 1. What was built\n')
p.append(r'''For the existing full rank22 global branch, let

\[
u_t=z_t,\qquad v_t=\frac{\sum_{j<t}z_j}{\sqrt{\max(t,1)}},\qquad
p_{ij}=\frac{u_i v_j-u_j v_i}{\|u\wedge v\|},\ i<j.
\]

The full231-coordinate dense projection remains. With the numerical safeguards below, the new channel is

\[
q=\frac{u}{\max(\|u\|,10^{-6})},\qquad
P_{ij}=p_{ij},\quad P_{ji}=-p_{ij},\qquad
c_j=\sum_i q_iP_{ij},\qquad
g=W_p p+W_c c.
\]

The sign is **qᵀP**, not Pq. Away from clamping/quantization, this yields the normalized component of v orthogonal to q. It does not restore the parallel component, prefix magnitude or associations that were never encoded in the prefix representation. The current query uses the existing learned reduced frame; this experiment does not test an independent query projector or a key/value matrix memory.

`query_core.h` implements the packed contraction, its direct query VJP, its packed-plane VJP and native pool dispatch without allocation. Backpropagation includes both paths: through p and through q. The VJP adds

\[
\bar p_{ij}\mathrel{+}=q_i\bar c_j-q_j\bar c_i,
\qquad \bar q=P\bar c,
\qquad \bar u\mathrel{+}=\frac{\bar q-\hat u(\hat u^\mathsf T\bar q)}{\|u\|}
\]

above the query floor; below it the denominator is constant. The native backward reconstructs a row's packed feature locally when necessary, so compact execution may reuse the former packed-feature arena. It then combines the geometry adjoints before the existing reverse prefix scan.

`build_query.py` produces compact cached, frame/FF reconstruction and combined layer/frame/FF reconstruction variants. Native plans count the new cached22-vector per row/layer and contraction-gradient scratch. `policy.py` retains one persistent model/optimizer state and reusable workspace, with real resizing and budget admission. All linear, tensor, backward and AdamW arithmetic runs in C. Python generates records, binds plans and records experiments. PyTorch supplies an independent AD reference, not the training-throughput engine.
''')
p.append(table(['Model','FF width','Parameters','Change from baseline'],[[names[m],str(256 if m=='attention' else 250 if m=='query_matched' else 257),f"{protocol['parameters'][m]:,}",f"{100*(protocol['parameters'][m]/protocol['parameters']['global']-1):+.3f}%"] for m in models]))
p.append('''
Width128, depth3, vocabulary260, tied embeddings, RMSNorm and SwiGLU are shared. Global arms retain rank22 and no RoPE. Attention has4 heads and RoPE. Per-parameter name/seed initialization makes shared tensors identical; Wc is initialized at standard deviation1.0, the same unit-feature scale as Wp. No candidate-specific hyperparameter search was performed.

The full-size addition tests the new route with unchanged FF257. The FF250 variant adds only384 total parameters relative to the525696-parameter baseline, so a gain cannot simply be credited to an extra8448 parameters. FF reallocation itself remains a trade-off, rather than an exactly identical architecture.
''')
p.append('## 2. Numerical and implementation gates\n')
p.append('''A repeated-prefix check exposed a pre-existing sensitivity in the original area-clamp convention. Identical leading tokens create an absent plane. FP32 forward outputs can agree closely while very large intermediate derivatives cancel differently through shared weights. In the recorded two-layer baseline example, the C gradient norm is about344205 and its relative disagreement with Torch is1.26. The query head exhibits the same inherited issue. This is a numerical/derivative-conditioning finding, not a diagnosis that the new contraction VJP is wrong.

The already available precision safeguard is therefore common to **both** baseline and contraction study arms: when the frame Gram determinant is at most1e-4 of the squared-norm product, pair arithmetic/VJP use FP64. A precise area at most1e-6 yields zero feature and zero pair VJP. The query has its separate1e-6 norm floor. Ordinary feature/gradient buffers remain FP32. The safeguarded duplicate-prefix baseline's gradient norm is2.81 and C/Torch relative error is4.45e-7.

This explicitly changes the absent-plane numerical convention; it is not a proof of smoothness at the cutoff, a remedy for FP32 input quantization or a universal conditioning guarantee. Guarded near-collinear and tiny-query cases are tested independently. The common guard keeps the association comparison focused on the contraction. All six old ordered/multi-pair baseline checkpoints reproduce the previous study's selected weights and moments bitwise; the language baselines are newly trained under the common guarded convention and are not bitwise continuations of the old unguarded ones.
''')
p.append(f'''Validation passes include:

- Independent full-model forward/backward at width32 and width128, including FF257/250 and duplicate leading tokens, across all three query layouts. Maximum gradient relative L2 error: **{max(r['gradient_relative'] for r in v['models']):.3g}**; maximum logit error: **{max(r['logit_max'] for r in v['models']):.3g}**.
- Packed-contraction AD checks with ordinary, tiny, large and zero queries; full prefix-geometry AD and directional finite differences. Guarded geometry maximum gradient relative error: **{max(r['gradient_relative'] for r in probe['guarded_geometry']):.3g}**.
- Exact causal-prefix output equality when future tokens change; bitwise state equality across shape/layout transitions and one/four threads; exact checkpoint replay.
- Exact-capacity buffer red zones and focused UBSan for compact and combined reconstruction; independent baseline, hybrid and attention controls.
- Microbatch accumulation normalized according to the native SUM-of-means contract: gradient relative difference **{v['state']['normalized_accumulation_gradient_relative']:.3g}**, first updated-weight difference **{v['state']['first_updated_weights_relative']:.3g}**.
- Query-head budget changes96→32→20→64MiB at B4/T1024, fixed microbatch1, preserve persistent storage and produce bitwise-identical optimizer state to the ample cached reference. The20MiB plan actually uses about18.97MiB;8MiB is refused before allocation.
''')
p.append('## 3. Frozen learning protocol\n')
p.append(table(['Task','Updates/run','Data'],[
 ['Single-pair value retrieval','900','B4;1 random key/value triple; balanced lengths64/128/256'],
 ['Multi-pair association','900','B4;2 pairs for300 updates, then4; balanced64/128/256'],
 ['Ordered first-value recall','900','B4; targets and distractors share tokens0–15; balanced64/128/256'],
 ['Raw WikiText-2 byte modeling','600','Lengths32/64/128; batches8/4/2; exactly256 supervised bytes/update']]))
p.append('''
Synthetic associations have unique random keys32–47 and values0–15, explicit triples `(258,key,value)`, filler64–95, and a final query marker/key. Validation always tests the declared task:1 pair for the single-pair arm;4 pairs for the curriculum arm. Data are paired across model candidates. Same optimizer/rates are used: AdamW, peak LR1e-3,30-update warmup, cosine floor1e-4, betas0.9/0.95, epsilon1e-8, decay0.1, norm clip1.0. The lowest aggregate validation CE at100-update intervals selects a checkpoint. Final tests never select checkpoints or revise the recipe.

Every training arm has the same32MiB **controlled-buffer** envelope. Global/query/hybrid retain the full effective microbatch; attention uses the largest fitting microbatch and accumulation, including micro2×2 at T256. This is the stronger matched-storage comparator. It is the retained native probability-materializing attention implementation, not a claim about every optimized attention implementation.

The bound counts four FP32 persistent arrays, activation/scratch workspace, staging, pool control, alignment slack and named C records. It excludes code, thread stacks, Python metadata, data, checkpoint/result copies and allocator retention. It is not process RSS or an OS memory cap. Fresh test inference has a common separate512MiB envelope, and performs no updates.

Primary association evaluation combines T128 and T256:256 examples/length per selected seed checkpoint,512 per seed. Larger-length,1/2/4/8-pair and counterfactual results are secondary. Samples are shared across candidates and training seeds. Tables give the mean of three trained models; seed ranges are descriptive, not confidence intervals or independent pooled-example replication.
''')
p.append('## 4. Association result: the new readout does not solve binding\n')
p.append(table(['Model','Single-pair accuracy','Four-pair accuracy','Four-pair CE'],[[names[m],pct(primary('kv_single',m)['mean']),pct(primary('kv_curriculum',m)['mean']),f"{primary('kv_curriculum',m,'ce_nats')['mean']:.5f}"] for m in models]))
p.append('\nUnconditional guessing over16 labels averages **6.25%**. An oracle that knows the4 candidate values but guesses their assignment averages25%; that is a different reference. Attention’s single-pair success verifies that the objective/data can support value retrieval at this budget, while its four-pair result still does not establish reliable binding.\n')
p.append(table(['Contraction comparison','Accuracy change vs baseline','Three-seed changes'],[[f"{task}: {names[m]}",f"{100*s['primary_differences'][f'{task}/{m}-global']['accuracy_delta']['mean']:+.3f} points",', '.join(f'{100*x:+.3f}' for x in s['primary_differences'][f'{task}/{m}-global']['accuracy_delta']['values'])] for task in ('kv_single','kv_curriculum') for m in ('query_full','query_matched')]))
p.append('\nThe relevant finding is absence of a useful retrieval gain, not whether a small finite-sample fluctuation is exactly zero. Neither additional parameters nor near-matched FF reallocation produces reliable association learning here.\n')
p.append(table(['Four-pair test','Model','Accuracy','Prediction is one of the4 values'],[[f'T{t}',names[m],pct(curve('kv_curriculum',m,t,4)['accuracy']['mean']),pct(curve('kv_curriculum',m,t,4)['prediction_in_candidate_set_rate']['mean'])] for t in (128,256,1024) for m in models]))
p.append('\nThe candidate-set rate distinguishes access to any plausible value from retrieval of the queried assignment. Longer contexts and8-pair extrapolations are retained in `results/summary.json` and per-case predictions, including weak results.\n')
p.append(table(['Counterfactual at T128','Model','Both versions correct','Prediction changes'],[[mode,names[m],pct(cf('kv_curriculum',m,128,mode)['both_correct_rate']['mean']),pct(cf('kv_curriculum',m,128,mode)['prediction_change_rate']['mean'])] for mode in ('value_swap','query_change') for m in models]))
p.append('''
Value swaps preserve the complete token multiset and query key while changing the correct value. Query changes leave the stored triples unchanged and ask for another present key. Both-correct rates require appropriate responses to both versions. The analogous T512 tests are retained. These probes prevent a favorable average candidate-value score from being mistaken for assignment learning.

In the global-family four-pair T128 swap tests, predictions never change, so no example pair has both answers correct. The full-size contraction does increase response sensitivity to query-key changes (31.64% prediction changes, versus19.01% baseline), but both-correct accuracy remains0.13%. Query sensitivity alone is insufficient: the model still fails to map the new key to its value. These are observed argmax behaviors on the finite test, not proofs that every internal state or logit is invariant.

The result is consistent with an unresolved learned write/binding bottleneck. The contraction reads the same compressed plane and current frame. If earlier layers have not encoded which value belongs to which key before aggregation, the readout cannot recover that missing assignment. This is an architectural hypothesis supported by the failure pattern, not a representation-impossibility proof; more training, better optimization or a different learned feature construction could change it.
''')
p.append('## 5. Ordered recall and language controls\n')
p.append(table(['Ordered recall','T128','T256','T512','T1024','T2048'],[[names[m],*[pct(curve('ordered',m,t)['accuracy']['mean']) for t in (128,256,512,1024,2048)]] for m in models]))
p.append('\nOrdered recall uses a genuinely order-sensitive shared target/distractor vocabulary. The tables report fresh data:256 examples/seed at T128/256/512,128 at T1024 and64 at T2048. Training ends at T256, so longer lengths test extrapolation. This control is distinct from keyed association and cannot substitute for it.\n')
p.append(f"The full-size contraction gains **{100*(curve('ordered','query_full',2048)['accuracy']['mean']-curve('ordered','global',2048)['accuracy']['mean']):.2f} percentage points** at T2048; the matched version gains **{100*(curve('ordered','query_matched',2048)['accuracy']['mean']-curve('ordered','global',2048)['accuracy']['mean']):.2f} points**. The full-size variant's three-seed range is87.50–90.63%; the matched variant spans68.75–95.31%, so its longer-context stability is less convincing even though it wins T1024. Attention32 remains ahead in mean T2048 accuracy. These small long-length samples need replication.\n")
p.append(table(['Model','First checked99% validation step','Training time to crossing','T512 swap both-correct'],[[names[m],f"{s['ordered_time_to_quality'][m]['first_checked_step']['mean']:.1f}",f"{s['ordered_time_to_quality'][m]['training_seconds']['mean']:.3f}s",pct(cf('ordered',m,512,'value_swap')['both_correct_rate']['mean'])] for m in models]))
p.append('\nCrossings are observed every100 updates and are not a sustained-accuracy certificate. This is time to a task-specific validation threshold; it must be considered alongside total update time and held-out extrapolation.\n')
p.append(table(['Language model','CE, nats/byte','T512 CE'],[[names[m],f"{s['language'][m]['ce_nats']['mean']:.5f}",f"{curve('lm',m,512)['ce_nats']['mean']:.5f}"] for m in (*models[:3],'hybrid','attention')]))
p.append('''
The language aggregate weights the actual test bytes at T32/64/128 (14336 bytes per seed). Each run trains153600 bytes from the raw train split. The local/global language hybrid remains a separate candidate and checkpoint family; no hybrid recall result is implied. Small CE changes need broader scale/seed replication before promotion.

The contraction improves aggregate byte CE by only about0.002 nats/byte; T512 is essentially unchanged. The separate language hybrid produces a much larger gain and still trails attention. The new query route does not replace that language-focused candidate.
''')
p.append('## 6. Execution and memory cost\n')
p.append(table(['Multi-pair training','Controlled buffers','900-update execution','ms/update'],[[names[m],f"{s['training'][f'kv_curriculum/{m}']['buffers_mib']['mean']:.3f}MiB",f"{s['training'][f'kv_curriculum/{m}']['training_seconds']['mean']:.3f}s",f"{s['training'][f'kv_curriculum/{m}']['ms_per_update']['mean']:.3f}"] for m in models]))
p.append('''
Training execution includes host update preparation/binding/staging and C updates, excludes validation/data generation/startup/builds, and averages three seeds. There is no borrowed timing calibration or claim that a noisy short lookup finds the global optimum. The controller prefers the larger fitting microbatch and cheaper admitted reconstruction tier. All candidates see the same physical positions and supervision within each task.

The contraction consumes additional cache/parameters and small GEMMs/pool barriers. Its execution cost is worth paying only for a measured capability or convergence advantage. The full head’s low-storage property is retained, but comparison to attention must grant attention accumulation too.

This study runs native BCIR-derived attention and global kernels. It does **not** rerun a PyTorch training-throughput benchmark, establish a new speedup over PyTorch or qualify accelerators/bare metal.
''')
p.append('## 7. Recommended next steps\n')
p.append('''1. Keep the full-head contraction as an explicit experimental recipe, with its tested numerical contract and checkpoint identity. Promote it only for tasks whose held-out quality/time-to-quality justify its cost; do not make it the default associative-memory mechanism.
2. Target **write-side binding** next: construct a local triple/adjacency-aware feature before global aggregation, or a small explicit key/value matrix state with learned write/read interfaces. Retain the plane for complementary global context. Compare against the same one-pair, four-pair, value-swap and query-change tests.
3. Separate the hypotheses: independent query projection, area/parallel-coordinate side channels, and a1→2→4→8-pair curriculum should be individually named and parameter/data matched. More updates are a legitimate next study, not an explanation already proven by these900-update failures.
4. Keep the common precision guard and budgeted C execution. For the useful ordered-recall candidate, fuse its feature/VJP work and group compatible readout VJPs to reduce barriers, then recheck gradients and trajectories. These are proposed optimizations; no speed gain from them is claimed here.
5. Retain the separate language hybrid and strong attention32 reference. Before claiming general architecture superiority, expand scale, data, seeds, stronger attention baselines and hardware qualification.

The key distinction is between a correct geometric program and a useful learned memory. The new implementation closes the equations-to-code gap for this hypothesis. The controlled association tests show that this contraction alone does not close the binding gap.
''')
p.append('## 8. Reproduction, provenance and limits\n')
p.append('''The recovered integrated archive was SHA-256 verified (`e3899d3a52ab283e2aa7976fdc7c1e6bb95566563ceb3b97ffa808596caf2731`), and all205 listed source/data/checkpoint entries matched their manifest. This study is self-contained: pinned C/tensor sources, exact ctypes ABI, generated C, build commands, independent oracle, protocols, raw events, test predictions/labels,51 checkpoints and source manifests are included. See `README.md` for clean-run commands.

Native foundations are pinned to BCIR `aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d`; inspected upstream main remains `be6e377a2eb417cb4bfe3de5243ca09961ba1833`. The host is AMD EPYC9V74, eight-core quota/8GiB container; experiments use4 native pool threads. GCC13.3/C11 builds use `-O3`, no fast-math and disabled implicit FP contraction. Python3.12, NumPy2.3.5, PyArrow26.0.0 and CPU Torch2.14.1 support the study. Builds, model jobs and heavy checks were serialized.

The repository's requested newest LLVM/Clang/MLIR23.1.3 and Python3.15 were unavailable; Node24.19 was behind24.22. Full repository CI, native ARM/Windows, accelerators, UEFI and physical hardware-memory movement are unqualified. This is isolated research C and a host controller, not a merged/registered production IR/MLIR operation. No repository source, commit or PR was changed.

Synthetic tasks and a small byte model do not predict large-language-model capability. Three seeds and limited training do not prove representational impossibility, universal superiority or statistical equivalence. Test data are paired and fixed, not thousands of independent training replications. Negative association results, full per-seed curves and the numerical stress findings are retained.

Source references:

- BCIR pinned native foundation: https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/tree/aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d
- Inspected main: https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/tree/be6e377a2eb417cb4bfe3de5243ca09961ba1833
- Dataset split hashes/URLs: `DATA_PROVENANCE.md` and `protocol.json`.
- Numerical evidence: `results/validation.json`, `results/conditioning-check.json`, `results/precision-budget-probe.json`.
- Learning evidence: `results/train-*.json`, `results/eval-*.json`, `results/summary.json`, `results/baseline-continuity.json`.
- Final source/checkpoint/pairing audit: `results/final-audit.json`.

All numeric outcomes in this report derive from the retained runs, not claimed figures in a research paper.
''')
out=ROOT/'BCIR_Query_Aware_Head_Report_2026-10-18.md'
import re
text='\n\n'.join(p)
words='the|at|has|only|over|are|and|about|most|separate|changes|fixed microbatch|width|Width|depth|vocabulary|lengths|Lengths|length|batches|for|then|values|keys|filler|exactly|task|micro|checked|every|trains|these|a|full|cached|same|from|with|than|all|case|Python|NumPy|PyArrow|Torch|GCC|MLIR|Node|EPYC|reaches|is|rank|deviation|norm|betas|epsilon|decay|clip|floor|microbatch|before|after'
text=re.sub(r'\b('+words+r')(?=\d)',r'\1 ',text)
text=re.sub(r'(?<=\d)(MiB|GiB|s)\b',r' \1',text)
text=re.sub(r'(?<=[;:])(?=[0-9A-Za-z])',' ',text)
text=re.sub(r'(?<=,)(?=[A-Za-z]|[0-9]{4,}\b)',' ',text)
text=re.sub(r'\n{3,}','\n\n',text)
out.write_text(text);print(out)
