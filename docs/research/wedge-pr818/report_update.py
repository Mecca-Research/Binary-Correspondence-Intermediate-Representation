"""Source-backed assessment and standalone scientific plot from retained records."""
from study import *
from collections import defaultdict
import re


def read(p):return json.loads((ROOT/p).read_text())
def avg(xs):return float(np.mean(list(xs)))
def pct(x):return f'{100*x:.2f}%'
def ci(r):
    lo,hi=r['a_over_b']['paired_bootstrap_95_percent_interval']
    return f'{lo:.3f}–{hi:.3f}'
def modelname(m):return {'global':'Global','query_full':'Query FF257','query_matched':'Query FF250','attention':'Attention','hybrid':'Hybrid'}[m]
def curve(s,task,m,t):return next(x for x in s['curves'] if x['task']==task and x['model']==m and x['context']==t)


def main():
    freeze();summary=read('results/summary.json');old=read('baseline/results/summary.json');audit=read('results/before-after-audit.json')
    assert audit['all_training_checks_pass'] and audit['all_regression_rows_equal'], 'Review numerical differences before issuing the assessment.'
    timing=read('results/paired-timing.json');changes=read('change-analysis.json');meta=read('pr818-metadata.json');inventory=read('results/buffer-inventory.json')
    bftrain=[json.loads(p.read_text()) for p in sorted((ROOT/'bf16/results').glob('train-*.json'))]
    bfeval=[json.loads(p.read_text()) for p in sorted((ROOT/'bf16/results').glob('eval-*.json'))]
    assert len(bftrain)==len(bfeval)==15
    matched=[];ordered_pairs=[]
    for d in bftrain:
        stem=f"{d['task']}-{d['model']}-{d['seed']}";a=read(f'results/train-{stem}.json')
        keys=('parameters','steps','input_sha256','physical_positions','supervised_tokens')
        checks={k:a[k]==d[k] for k in keys}
        checks['recipe_except_contract']={k:v for k,v in a['recipe'].items() if k!='contract'}=={k:v for k,v in d['recipe'].items() if k!='contract'}
        shape_keys=('context','variant','effective_batch','microbatch','accumulation')
        checks['same_microbatch_reconstruction']=[{k:c[k] for k in shape_keys} for c in a['policy']['choices']]==[{k:c[k] for k in shape_keys} for c in d['policy']['choices']]
        ea=read(f'results/eval-{stem}.json');eb=next(e for e in bfeval if (e['task'],e['model'],e['seed'])==(d['task'],d['model'],d['seed']))
        checks['same_fixed_evaluation_inputs']=all(next(x for x in ea['rows'] if (x['context'],x['pairs'])==(r['context'],r['pairs']))['input_sha256']==r['input_sha256'] for r in eb['rows'])
        assert all(checks.values()),(stem,checks)
        matched.append({'case':stem,'checks':checks})
        if d['task']=='ordered':
            aa=next(x for x in ea['rows'] if x['context']==2048)['test'];bb=next(x for x in eb['rows'] if x['context']==2048)['test']
            ordered_pairs.append({'seed':d['seed'],'FP32_accuracy':aa['accuracy'],'BF16_accuracy':bb['accuracy'],'accuracy_delta':bb['accuracy']-aa['accuracy'],
                                  'examples_per_seed_model':bb['scored_tokens'],'FP32_selected_step':a['selected_step'],'BF16_selected_step':d['selected_step']})
    bg=defaultdict(list)
    for d in bfeval:
        for row in d['rows']:bg[(d['task'],d['model'],row['context'])].append({'seed':d['seed'],**row['test']})
    bfquality={}
    for task,model in sorted(set((d['task'],d['model']) for d in bfeval)):
        values=[]
        for d in bfeval:
            if (d['task'],d['model'])!=(task,model):continue
            count=sum(r['test']['scored_tokens'] for r in d['rows'])
            values.append({'seed':d['seed'],**{k:sum(r['test'][k]*r['test']['scored_tokens'] for r in d['rows'])/count for k in ('accuracy','ce_nats')}})
        ds=[d for d in bftrain if (d['task'],d['model'])==(task,model)]
        bfquality[f'{task}/{model}']={'evaluation_contexts':sorted(t for tt,mm,t in bg if (tt,mm)==(task,model)),
            'aggregation':'Within each seed, supervised-token-weighted over focused contexts; then mean of three seed metrics.',
            'per_seed':values,'accuracy':avg(v['accuracy'] for v in values),'ce_nats':avg(v['ce_nats'] for v in values),
            'buffers_mib':ds[0]['policy']['resident_bound_bytes']/2**20,
            'training_ms_per_update':avg(1000*d['training_seconds']/d['steps'] for d in ds)}
    out={'source':{'PR':818,'merge':meta['merge_commit_sha'],'previous_native_pin':changes['exact_experiment_pin']['base']},
         'audit':{k:v for k,v in audit.items() if k!='rows'},'BF16_quality':bfquality,
         'BF16_updates':sum(d['steps'] for d in bftrain),'BF16_quality_by_length':[
            {'task':task,'model':model,'context':t,'accuracy':avg(x['accuracy'] for x in rows),'ce_nats':avg(x['ce_nats'] for x in rows),'per_seed':rows}
            for (task,model,t),rows in sorted(bg.items())],
         'BF16_matching_audit':matched,'BF16_ordered_T2048_paired':ordered_pairs,
         'mixed_layer_research_estimates':inventory['mixed_layer_research_estimates'],
         'paired_timing_metadata':timing['metadata'],'paired_timing_summary':[{k:v for k,v in r.items() if k!='observations'} for r in timing['rows']]}
    (ROOT/'results/update-assessment.json').write_text(json.dumps(out,indent=2))

    speedrows=[r for r in timing['rows'] if r['label'].startswith('FP32/')]
    speechtab='\n'.join(f"| {r['label'].removeprefix('FP32/')} | {r['a_median_ms_per_update']:.3f} | {r['b_median_ms_per_update']:.3f} | {r['a_over_b']['median']:.3f}× | {ci(r)} |" for r in speedrows)
    association='\n'.join(f"| {modelname(m)} | {pct(old['primary_association']['kv_single/'+m]['accuracy']['mean'])} | {pct(summary['primary_association']['kv_single/'+m]['accuracy']['mean'])} | {pct(old['primary_association']['kv_curriculum/'+m]['accuracy']['mean'])} | {pct(summary['primary_association']['kv_curriculum/'+m]['accuracy']['mean'])} |" for m in MODELS)
    ordered='\n'.join(f"| {modelname(m)} | {pct(curve(old,'ordered',m,1024)['accuracy']['mean'])} | {pct(curve(summary,'ordered',m,1024)['accuracy']['mean'])} | {pct(curve(old,'ordered',m,2048)['accuracy']['mean'])} | {pct(curve(summary,'ordered',m,2048)['accuracy']['mean'])} |" for m in MODELS)
    language='\n'.join(f"| {modelname(m)} | {old['language'][m]['ce_nats']['mean']:.6f} | {summary['language'][m]['ce_nats']['mean']:.6f} |" for m in (*MODELS,'hybrid'))
    bfrows=[]
    for key,b in bfquality.items():
        task,m=key.split('/')
        if task=='ordered':
            baseline=curve(summary,'ordered',m,2048)['accuracy']['mean'];now=avg(x['accuracy'] for x in bg[(task,m,2048)])
            metric='T2048 accuracy';a=pct(baseline);bb=pct(now)
        elif task=='lm':metric='CE, lengths32/64/128';a=f"{summary['language'][m]['ce_nats']['mean']:.6f}";bb=f"{b['ce_nats']:.6f}"
        else:metric='4-pair accuracy, T128/256';a=pct(summary['primary_association'][key]['accuracy']['mean']);bb=pct(b['accuracy'])
        bfrows.append(f'| {modelname(m)} / {task} | {metric} | {a} | {bb} | {b["buffers_mib"]:.3f} |')
    ordered_pair_table='\n'.join(f"| {r['seed']} | {pct(r['FP32_accuracy'])} | {pct(r['BF16_accuracy'])} | {100*r['accuracy_delta']:+.3f} points |" for r in ordered_pairs)
    panelrows=[r for r in timing['rows'] if r['label'].startswith('BF16-panels/')]
    paneltable='\n'.join(f"| {r['label'].removeprefix('BF16-panels/')} | {r['a_median_ms_per_update']:.3f} | {r['b_median_ms_per_update']:.3f} | {r['a_over_b']['median']:.3f}× | {ci(r)} | {r['b_observed_packing_bytes']:,} / {r['b_packing_bytes']:,} |" for r in panelrows)
    placements=[r for r in timing['rows'] if r['label'].startswith('Placement/')]
    pagetable='\n'.join(f"| {r['label'].removeprefix('Placement/')} | {r['a_median_ms_per_update']:.3f} | {r['b_median_ms_per_update']:.3f} | {r['a_over_b']['median']:.3f}× | {ci(r)} | {r['b_huge_pages']['observed_AnonHugePages_kib']:,} |" for r in placements)
    numerics=[r for r in timing['rows'] if r['label'].startswith('Numerics/')]
    numerictable='\n'.join(f"| {r['label'].removeprefix('Numerics/')} | {r['a_median_ms_per_update']:.3f} | {r['b_median_ms_per_update']:.3f} | {r['a_over_b']['median']:.3f}× | {ci(r)} | {r['initial_numerical_comparison']['logits_max_abs_delta']:.3g} |" for r in numerics)
    p=changes['pr818'];e=changes['exact_experiment_pin'];records=p['categories']['measurement_records']
    text=f'''# BCIR PR #818: merged runtime rerun of the full global/query head

Completed {datetime.datetime.now(datetime.timezone.utc).isoformat()} · native C, controlled CPU research

## Result

The merged update has been rebuilt and the geometric-head experiments have been rerun. **All {audit['training_runs']} FP32 runs / {audit['training_updates']:,} updates reproduce the previous selected and final training states exactly.** All {summary['count']['evaluation_rows']} fixed model/context evaluation groups and {summary['count']['counterfactual_pairs_of_batches']} counterfactual test groups were executed again and reproduce the prior predictions and metrics exactly. Each group contains multiple scored examples or tokens; these counts are not independent training replications. These are regression samples reused from the previous study, not a new unseen holdout.

The query-aware contraction retains its ordered-recall benefit and its association-learning limitation at this budget. This update supplies execution mechanisms and leaves the retained head's equations unchanged. The result does not prove that a multilayer prefix model can never learn bindings; it leaves the representation/training question open. Timing changes must be read against the exact previous source pin, which already included much of PR818.

An additional **15 from-scratch BF16 runs / {out['BF16_updates']:,} updates**, matched to the same three seeds and initial states, test the new numerical family. Their results are separated below. The host is AMD EPYC 9V74, with AVX-512F and BF16 CPU features but **no Intel AMX**. This is not a rerun of the upstream Intel-AMX versus compiled-PyTorch throughput comparison.

**The notable new learning result is BF16 ordered recall at length2048: 89.06%→95.31%** for the queryFF257 head, across three matched seeds on64 fixed examples/model. Association remains near chance and language loss is effectively unchanged. The isolated AdamW updater is1.334× faster; full-head FP32 timing changes are mostly unresolved. BF16 adds packing storage and runs slower for the query head on this CPU, so the recall improvement comes with an accuracy/cost trade-off.

![Measured runtime and learning results](BCIR_PR818_Results_2026-10-18.png)

## 1. What changed, and what the previous experiment already had

[PR818]({meta['url']}) merged at `{meta.get('merged_at','2026-10-18T22:10:03Z')}`. Source identities:

| Role | Commit |
| --- | --- |
| Previous experiment's native source | `{e['base']}` |
| PR base | `{p['base']}` |
| PR head | `{p['head']}` |
| Merged main tested here | `{e['head']}` |

The PR changes **{p['files']} files, adds {p['added']:,} lines and deletes {p['deleted']:,}**. Benchmark/measurement records account for **{records['files']} files and {records['added']:,} added lines ({100*records['added']/p['added']:.2f}% of additions)**. The runtime C/header changes are 5files, +{p['categories']['runtime']['added']:,}/−{p['categories']['runtime']['deleted']:,}. This is a substantial documented optimization campaign; the line count is mostly retained evidence rather than 390,000 new lines in the execution engine.

The previous pin is inside PR818's history. From that exact pin to merged main, the diff is {e['files']} files, +{e['added']:,}/−{e['deleted']:,}; runtime is +{e['categories']['runtime']['added']:,}/−{e['categories']['runtime']['deleted']:,}. Crediting all of PR818's advertised gain as an additional gain over our previous study would double-count earlier work.

| Change | Effect on this research |
| --- | --- |
| Parallel scans/reductions, grouped and paired GEMM, fixed-shape FP32 blocks, pool scheduling and earlier SIMD work | Much was already in the old pin; retained in the matched control. |
| AdamW shared-division update | Newly grafted production updater; its bit-preserving claim is checked over all41,400 FP32 updates and in isolated timing. |
| ABI4 and explicit BF16 packing workspace | Host records upgraded from240 to256bytes; packing storage and alignment are included in budget admission. |
| Portable, AVX-512F and Intel-AMX BF16 GEMM instances | Distinct arithmetic contract; this host admits mask{timing['metadata']['bf16_available_variants']} ({', '.join(timing['metadata']['bf16_instances'].values())}). No AMX instance is available here. |
| BF16 SwiGLU panel producers/readers | FF256 probe activates all3 panel paths and agrees bitwise with panels disabled. Original FF257/250 model recipes are not silently changed to fit these panels. |
| Fused attention arithmetic | Separate bit32 numerical option; timed separately from the fixed FP32 learning rerun. |
| Framework huge-page state allocation | Optional 2MiB-aligned common mapping tested separately, with actual `AnonHugePages` evidence rather than treating accepted advice as allocated pages. |
| Duplicate BPE merge fix | First-ranked duplicate is retained. Focused regression passes. Byte-level datasets here have no BPE step, so this cannot explain their learning results. |
| Grassmann/Plücker head | Still the saved standalone C research overlay; no production geometric-head implementation was found in the merged runtime. |
| ASN.1 / GEM+ / signed-plan optimizer | No implementation change relevant to those engines in this PR; this report does not relabel tensor-buffer measurements as new ASN.1 or GEM+ results. |

`change-analysis.json` retains file-by-file diff counts, and `pr818-metadata.json` retains the retrieved PR description. The separate upstream 9,836,928-parameter AMX study reports2.282× compiledPyTorchBF16 on an Intel Xeon, with5paired seeds/1,000steps. Its hardware, model, software versions and numerical contracts differ. That reported ratio is neither reproduced nor contradicted by the AMD geometric-head measurements here.

## 2. Implementation and validation

`build_upgrade.py` ports the production tensor unit, dispatch/group/panel helpers, validity checks, attention entry and updater into the saved research overlays. It retains the full231-coordinate rank22 wedge, query-aware contraction, causal prefix/VJP, precision guard, masked loss, compact arena layouts and rematerialization equations. Old controls compile the original overlays against their original ABI3 tensor/header source. New overlays use ABI4. Everything is built without fast-math or implicit contraction.

Execution uses the native C training rail through the standalone Python host controller. Lowering this custom head through BCIR's general IR/GEM+ planner remains an integration task. Minibatches use ordinary native token arrays; ASN.1 training-data plumbing is not exercised by these measurements. This defines what the rerun attributes to kernels and explicit buffer plans.

The custom ignored-target path is preserved; production dense-target admission cannot simply replace it. Compact FF layouts have deliberate cross aliases that are safe for their elementwise FP32 writers. Their BF16 panel path remains disabled until those tiled writers have a separate alias proof. The cached FF256 variant is directly qualified for panels. This is an integration constraint, not an assumption that every backend switch is interchangeable.

Passed FP32 checks include independent Torch full-model forward/backward, guarded/near-collinear geometry, contraction VJP, exact legacy/new logits and gradients, checkpoint replay, thread/layout/shape transitions, equal-supervision accumulation, exact-capacity red zones, focused UBSan, and96→32→20→64MiB buffer reconfiguration. Separately, all18 BF16 cached-versus-rematerialized cases pass logits, VJP and updated-state parity. All3 FF256 panel paths execute and agree with the non-panel implementation. These BF16 parity checks do not assert that rounding is a differentiable operator; they qualify the declared mixed-precision training algorithm. The upstream native-decoder gate also passes gradients, AdamW, accumulation, checkpoint resume and training-to-Q8-C inference.

This was a focused training/runtime study, not a new repository release qualification. GCC13.3, Python3.12.14, NumPy2.3.5 and Torch2.14.1+cpu were used. LLVM/MLIR23, Python3.15, native ARM64, Windows and AMX were unavailable; their full qualification gates were not run. `toolchain-status.log` retains the environment check.

## 3. Matched FP32 learning rerun

Seeds are 20261018, 20261019 and 20261020. The models use width 128, depth 3, vocabulary 260, tied embeddings, RMSNorm and SwiGLU. Global/query use rank 22 and no RoPE; attention uses 4 heads and RoPE. Parameters: Global 525,696; Query FF257 534,144; Query FF250 526,080; Attention 525,696. Hyperparameters, streams, initialization, validation checkpoint selection and update counts are unchanged. Recall trains 900 updates with batch 4 and balanced lengths 64/128/256; language trains 600 updates with 256 supervised bytes/update and balanced lengths 32/64/128. Arithmetic is contract 14; optimizer is native C AdamW.

Every training run has a common32MiB controlled-array/pool/staging/C-record envelope. Global/query retain batch4; attention uses microbatch2×accumulation2 atT256. Both FP32 and BF16 choose the same microbatch/reconstruction tiers for these shapes. BF16 adds packing storage within the envelope. The bound excludes code, thread stacks, Python objects, datasets, checkpoints and allocator retention; **it is not RSS or process peak memory**. Fixed test inference has a separate common512MiB bound.

### Association

Each accuracy combines256examples atT128 and256atT256, per selected seed. Values are three-seed means. Random guessing over16labels is6.25%; guessing uniformly among the4known candidate values is25%.

| Model | Old single-pair | New single-pair | Old four-pair | New four-pair |
| --- | --- | --- | --- | --- |
{association}

The query heads remain near chance. The value-swap and query-change counterfactual outputs are also unchanged. Faster kernels do not establish a new association-learning capability.

### Ordered recall and language

| Model | Old T1024 | New T1024 | Old T2048 | New T2048 |
| --- | --- | --- | --- | --- |
{ordered}

| Model | Old byte-language CE | New byte-language CE |
| --- | --- | --- |
{language}

Language CE is weighted over test lengths32/64/128, lower is better. The query head's long ordered-recall advantage remains intact, while native attention and the separate language-focused hybrid retain lower language loss. This update does not change that ranking under the fixed FP32 training recipe.

`results/before-after-audit.json` compares every selected and final state digest, input digest, validation record and per-update loss/gradient norm against the prior study; it also compares the rerun predictions, labels, CE and counterfactual outputs exactly. Whole NPZ file hashes are not used as substitutes for model-state equality.

## 4. Same-host paired performance

Historical session wall times are retained but are not used to attribute speed gains. The following tests alternateAB/BA for24paired rounds,3updates/observation, fixed synthetic dense targets, warm pools and an identical state restored outside each timed call. All use the same4logical CPUs and4native threads; no other experiment runs concurrently. Ratios above1 favor the new/right-hand configuration. Intervals bootstrap repeated paired timing observations on this one host; they do not describe hardware-population uncertainty.

Old/new rows use the same cached compact layout and allocation method with an ample arena, rather than the32MiB rematerialization policy. They isolate engine cost at fixed shape. They are full native updates, not scalar math microbenchmarks. AdamW-only includes its norm/clipping and API validation but excludes backward. The ratio is the median of paired ratios; it need not equal the quotient of the two separately reported timing medians.

| Test | Old ms/update | New ms/update | Old/new | Paired95% interval |
| --- | --- | --- | --- | --- |
{speechtab}

The isolated optimizer is1.334× faster (paired interval1.315–1.395). Eleven of the twelve full-model FP32 intervals include1; the globalT256 cell is1.049× (1.017–1.082). These are individual descriptive intervals without a multiple-comparison correction. They support a faster optimizer and mostly unresolved small whole-head changes on this host. The isolated optimizer multiplier must not be presented as the same multiplier for the entire head.

### Panel buffers

Same merged runtime and BF16 contract26, FF256 only; A disables panels and B enables them. Exact updated states agree on every paired trial. All3 panel paths are exercised. Observed workspace high-water marks are retained beside admitted upper bounds; they are not a proof of a globally minimum allocation.

| Test | No panels ms | Panels ms | A/B | Paired95% interval | Observed / admitted packing bytes |
| --- | --- | --- | --- | --- | --- |
{paneltable}

### Huge-page placement

Same merged FP32 model on both sides; A uses the original separate aligned allocations, B the common2MiB-aligned `MADV_HUGEPAGE` arena. Each row restores the same state and agrees bitwise. Accepted advice and observed huge pages are distinct fields in the raw records. The page counts cover overlapping VMAs, including mapping padding. Default NumPy/Linux allocations may already receive huge pages; this is an allocation-strategy comparison, not a controlled small-page-versus-huge-page ablation. Mapping/page-table behavior is host-specific.

| Test | Default ms | Advised arena ms | A/B | Paired95% interval | B observed huge pages, KiB |
| --- | --- | --- | --- | --- | --- |
{pagetable}

No automatic placement win is established. AtT512, the advised common arena is slower in this sample despite real huge-page coverage. Retain a measured choice rather than assuming that page advice accelerates this head.

### New arithmetic options

Attention-FMA uses contract46 against14. BF16 uses26 against14. These intentionally change arithmetic; their bitwise state identity is not required. Initial dense-sample logit differences quantify numerical change but do not certify converged quality. Attention-FMA here is a performance probe; the full51run learning rerun keeps it off.

| Option / test | FP32 ms | Option ms | A/B | Paired95% interval | Initial max logit difference |
| --- | --- | --- | --- | --- | --- |
{numerictable}

Attention-FMA shows about1.077× atT256 and1.079× atT512, with its intervals above1; theT128 probe is unresolved. BF16 queryFF257 is slower on this AVX-512F fallback (paired speed ratio0.877). The fallback in `bcir_tensor.c` expands BF16 operands and uses FP32 vector FMAs with separate even/odd accumulators to reproduce the declared AMX arithmetic. A CPU's AVX-512-BF16 feature flag therefore does not imply that this runtime is using a native BF16 dot-product path. Such a new instance would need its own parity or numerical-contract admission. These results favor testing the attention option further and keeping a hardware-specific arithmetic choice.

## 5. BF16 learning follow-up

The15case protocol was frozen before its first update:3seeds each for four-pair Global/QueryFF257/Attention, ordered QueryFF257 and language QueryFF257. Architecture, data, initialization, rates, validation and selection stay fixed; only GEMM arithmetic changes to26 and packing memory is counted. Attention-FMA remains off. QueryFF257 is preserved rather than silently replaced byFF256 to gain panel eligibility.

| Model/task | Metric | FP32 | BF16 | BF16 controlled buffers, MiB |
| --- | --- | --- | --- | --- |
{chr(10).join(bfrows)}

The BF16 matching audit checks all15 runs for identical model shapes except contract, input hashes, update/token counts, microbatch/reconstruction choices and fixed evaluation inputs. The T2048 ordered result is an observed **+6.25 percentage points** over the corresponding FP32 query head:

| Seed | FP32 T2048 | BF16 T2048 | Paired change |
| --- | --- | --- | --- |
{ordered_pair_table}

All six ordered checkpoints select step900. Each model is evaluated on the same64 fixed examples; three trained seeds do not create192unique test examples. Two seeds improve and one is unchanged. This is a promising long-recall result at one scale and budget. More seeds and a fresh test set are needed to assess how reliably it transfers. BF16 changes the training trajectory; these results alone do not identify why long-context accuracy increased.

BF16 seed-level outcomes and counterfactuals are retained in `bf16/results`; a mean near chance should not be read as a meaningful association breakthrough. Its runtime on this AMD fallback is not an estimate of the Intel-AMX path. BF16 packing raises global/query recall storage by about2.891MiB and attention by about1.375MiB here, while stored parameters, gradients and Adam moments remainFP32. Therefore the production PR's half-RSS comparison to PyTorch is not a claim that our controlled head arena halves simply by enablingBF16.

## 6. What to update next

1. **Separate representation work from kernel work.** Retain the full query head as an ordered-recall candidate. Test an explicit key/value relationship write before aggregation: for example, a matched-capacity key⊗value or per-record geometric memory with a query read. A readout cannot reconstruct information that its write path discarded; whether the existing multilayer model can instead learn a sufficient write is still open. Keep single-pair, four-pair, multiset-preserving swaps and query-only changes as gates.
2. **Qualify an aligned FF256 geometry candidate.** Compare it against its own FP32 and non-panel BF16 controls, with equal data and an explicit parameter-count difference. Rank22 creates231packed coordinates; panel padding and rank alignment are separate costs worth measuring rather than silently changing the model.
3. **Couple packing lifetimes with partial-layer caching.** The decoder admits a conservative sum of whole-call packs; record actual shared-operand group lifetimes and derive a safe smaller bound. The100native plan/packing queries in `results/buffer-inventory.json` expose a specific opportunity: atB4/T256, replacing one queryFF257 frame/FF-recomputed layer by a cached layer adds3,051,520activation bytes, with unchanged total scratch. The FP32 plan has5,796,744bytes of32MiB slack, enough arithmetically for one layer; the present portfolio chooses the same tier for all layers. BF16 has2,765,448bytes of slack, a286,072byte shortfall for that one-layer substitution. A tighter safe packing bound could make that mixed plan feasible. These are inventory-derived estimates, not implemented or timed mixed plans. Derive all mixed scratch/packing lifetimes before admitting one. Make BF16 panel element widths explicit before reclaiming storage in float-sized reservations; observed high-water marks alone are not universal bounds.
4. **Prove panel compatibility for rematerialized layouts.** The cross-alias FF scratch contract needs tiled-writer lifetime/overlap checks and parity/red-zone gates before panel writes can be enabled. Then remeasure whether panels plus rematerialization beat the current full-batch plan at the same32MiB limit.
5. **Promote arithmetic options only after quality gates.** Keep contract14 as the regression reference; gate26 and46 separately with same-seed learning, nearly-collinear/absent-plane safeguards and target-hardware timing. Add Intel-AMX and native ARM64 runs when hardware is available. SIMD eligibility is not a universal throughput guarantee.
6. **Integrate the head through a maintained extension boundary.** This session's source grafts make the experiment concrete, but production adoption should expose its C geometry/VJP and arena requirements through stable IR/runtime interfaces, with source/contract/plan digests. That allows future tensor/updater work to flow into the head without repeated textual surgery.

This evidence supports continued optimization of a working C training engine and a memory-conscious ordered-recall architecture. It does not establish a theoretical maximum, universal dominance over PyTorch/JAX, or solved associative memory. Those require their own matched quality, hardware and workload comparisons.

## Reproduction and retained evidence

The companion ZIP retains both pinned native source sets and licenses, original geometric sources, ABI/generator/controller code, data, frozen protocols, build commands, validation results, paired observations, baseline records, all66selected checkpoints, quality outputs and this report. `MANIFEST_SHA256.json` records per-file hashes; the archive hash is delivered separately. Run these commands from the extracted `wedge-pr818` directory:

```sh
python -m venv --system-site-packages .venv
.venv/bin/pip install numpy==2.3.5 pyarrow==26.0.0 safetensors==0.8.0 matplotlib==3.10.8
.venv/bin/pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python build_upgrade.py
.venv/bin/python validate_update.py
.venv/bin/python study.py
.venv/bin/python benchmark_update.py
.venv/bin/python buffer_inventory.py
.venv/bin/python validate_bf16.py
.venv/bin/python study_bf16.py
.venv/bin/python evaluate_query.py
.venv/bin/python evaluate_bf16.py
.venv/bin/python summarize_query.py
.venv/bin/python audit_update.py
.venv/bin/python report_update.py
```

Run heavy commands sequentially. Existing study outputs are retained by the scripts; use a fresh copy without results/checkpoints for a from-scratch replication. GCC and pthreads are required. Rebuilding on another compiler/host may change binaries or numerical instances; preserve the stored evidence and label the new environment separately. Upstream repository gates additionally require a checkout of the recorded merge; the whole repository is not bundled.
'''
    # Keep programmatic shape names intact while making prose units readable.
    text=re.sub(r'(?<=\d)(?=(?:MiB|bytes|files|runs|run|updates|steps|paired|seeds|heads|layers|examples|supervised|coordinate|selected|native|logical|case|labels|known|model|fixed|parameters|activation)\b)',' ',text)
    text=re.sub(r'\b(All|all|The|with|the|same|every|about|of|over|these|first|from|to|and|is|has|adds|mask|bit|reports|full|rank|common|batch|microbatch|accumulation|lengths|for|rounds|above|include|Paired|contract|against|uses|ratio|gate|GCC|Python|NumPy|Torch|MLIR|combines|interval|creates)(?=\d)',r'\1 ',text)
    text=re.sub(r'\bat(?=T\d)','at ',text)
    for a,b in (('and256atT256','and 256 at T256'),('remainFP32','remain FP32'),('enablingBF16','enabling BF16'),('compiledPyTorchBF16','compiled PyTorch BF16'),('globalT256','global T256'),('The100','The 100'),('a286,072byte','a 286,072-byte')):text=text.replace(a,b)
    (ROOT/'BCIR_PR818_Geometric_Head_Report_2026-10-18.md').write_text(text)
    plot(summary,old,timing,bfquality,bg)
    print(json.dumps({'audit':out['audit'],'BF16_quality':bfquality},indent=2))


def plot(s,old,timing,bfq,bg):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,9));fig.subplots_adjust(top=.88,bottom=.12,hspace=.46,wspace=.30)
    fig.suptitle('BCIR #818 — fixed geometry, newer native engine',x=.065,ha='left',fontsize=20,fontweight='bold')
    fig.text(.065,.91,'AMD EPYC · four native threads · same-host paired timings; learning means over three seeds',fontsize=11,color='#4c5965')
    colors=('#b3bec8','#176f83');names=[modelname(m) for m in MODELS];x=np.arange(4);w=.35
    ax=axes[0,0];rs=[next(r for r in timing['rows'] if r['label']==f'FP32/{m}/B4T256') for m in MODELS]
    ax.bar(x-w/2,[r['a_median_ms_per_update'] for r in rs],w,label='Prior source pin',color=colors[0]);ax.bar(x+w/2,[r['b_median_ms_per_update'] for r in rs],w,label='Merged #818',color=colors[1])
    ax.set_ylim(0,1.27*max(max(r['a_median_ms_per_update'],r['b_median_ms_per_update']) for r in rs))
    ax.set_xticks(x,names);ax.set_ylabel('Median ms / native update');ax.set_title('Fixed FP32 shape: batch4, length256',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=9)
    ax=axes[0,1];a=[100*s['primary_association']['kv_curriculum/'+m]['accuracy']['mean'] for m in MODELS]
    ax.bar(x,a,color=[colors[1]]*3+['#a66b27']);ax.axhline(6.25,linestyle='--',color='#697681',linewidth=1,label='16-label chance:6.25%')
    ax.set_xticks(x,names);ax.set_ylim(0,30);ax.set_ylabel('Four-pair accuracy, %');ax.set_title('Association remains unchanged in FP32',loc='left',fontweight='bold')
    for i,v in enumerate(a):ax.text(i,v+.8,f'{v:.2f}%',ha='center',fontsize=9)
    ax.legend(frameon=False,fontsize=9,loc='upper left')
    ax=axes[1,0];a=[100*curve(s,'ordered',m,2048)['accuracy']['mean'] for m in MODELS]
    ax.bar(x,a,color=[colors[1]]*3+['#a66b27']);ax.set_xticks(x,names);ax.set_ylim(0,110);ax.set_ylabel('Ordered recall accuracy, %');ax.set_title('Ordered recall at length 2,048',loc='left',fontweight='bold')
    for i,v in enumerate(a):ax.text(i,v+1,f'{v:.2f}%',ha='center',fontsize=9)
    bf=100*avg(z['accuracy'] for z in bg[('ordered','query_full',2048)])
    ax.scatter([1],[bf],marker='D',s=65,color='#bd475d',label=f'BF16 Query FF257: {bf:.2f}%',zorder=4);ax.legend(frameon=False,fontsize=9,loc='upper left')
    ax=axes[1,1];pr=[r for r in timing['rows'] if r['label'].startswith('BF16-panels/')];xx=np.arange(len(pr))
    ax.bar(xx-w/2,[r['a_median_ms_per_update'] for r in pr],w,color=colors[0],label='Panels disabled');ax.bar(xx+w/2,[r['b_median_ms_per_update'] for r in pr],w,color=colors[1],label='Panels enabled')
    labels=[r['label'].split('/')[1].replace('query_full','Query')+'\n'+r['label'].split('/')[2] for r in pr]
    ax.set_xticks(xx,labels);ax.set_ylabel('Median ms / native update');ax.set_title('FF256 panel probe, same BF16 arithmetic',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=9)
    fig.text(.065,.047,'FP32:51 runs /41,400 updates; exact before/after learned states and test outputs. BF16:15 runs /12,600 updates.\nTimings use cached layouts; quality training uses a32MiB controlled-buffer budget. No Intel AMX on this host. See report for paired timing intervals.',fontsize=9,color='#4c5965')
    fig.savefig(ROOT/'BCIR_PR818_Results_2026-10-18.png',dpi=170,facecolor='white');plt.close(fig)


if __name__=='__main__':main()
