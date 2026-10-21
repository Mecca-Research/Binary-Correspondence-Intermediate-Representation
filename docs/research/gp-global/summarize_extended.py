from experiment import *
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KINDS=('transformer','grassmann','symmetric','global','global_symmetric','projector')
NAMES={'transformer':'Attention','grassmann':'Local wedge','symmetric':'Local symmetric',
       'global':'Global wedge','global_symmetric':'Global symmetric','projector':'Global projector'}
COLORS=['#3b648c','#8f65a8','#999999','#19866c','#d29b42','#be6072']
SEEDS=(20261017,20261018,20261019)
def stat(a):
    a=np.asarray(a,dtype=float);mean=float(a.mean());sd=float(a.std(ddof=1)) if len(a)>1 else 0
    h=float(student_t.ppf(.975,len(a)-1)*sd/math.sqrt(len(a))) if len(a)>1 else 0
    return {'mean':mean,'sd':sd,'ci95':[mean-h,mean+h],'values':a.tolist()}
quality={};raw={}
for kind in KINDS:
    raw[kind]={}
    for task in ('wikitext2_byte','first_token_recall'):
        raw[kind][task]=[json.loads((ROOT/'results'/f'{task}-{kind}-{seed}.json').read_text()) for seed in SEEDS]
    lm=raw[kind]['wikitext2_byte'];rec=raw[kind]['first_token_recall']
    quality[kind]={'parameters':lm[0]['parameters'],'lm_ce':stat([r['test']['ce_nats'] for r in lm]),
        'lm_accuracy':stat([r['test']['accuracy'] for r in lm]),
        'recall_accuracy':stat([r['test']['accuracy'] for r in rec]),
        'recall_ce':stat([r['test']['ce_nats'] for r in rec])}
for task in ('wikitext2_byte','first_token_recall'):
    for si in range(3):
        assert len(set(raw[k][task][si]['input_sha256'] for k in KINDS))==1
contrasts={}
for a,b in [('global','grassmann'),('global','global_symmetric'),('global','transformer'),('projector','global')]:
    contrasts[f'{a}_minus_{b}']={
        'lm_ce':stat([raw[a]['wikitext2_byte'][i]['test']['ce_nats']-raw[b]['wikitext2_byte'][i]['test']['ce_nats'] for i in range(3)]),
        'recall_accuracy':stat([raw[a]['first_token_recall'][i]['test']['accuracy']-raw[b]['first_token_recall'][i]['test']['accuracy'] for i in range(3)])}
timing={};timing_raw={}
for threads in (1,4):
    path=ROOT/'results'/('scaling.json' if threads==4 else 'scaling-1thread.json')
    rows=json.loads(path.read_text());timing_raw[threads]=rows;timing[str(threads)]={}
    for context in (64,128,256,512,1024):
        timing[str(threads)][str(context)]={}
        for kind in KINDS:
            group=[r for r in rows if r['kind']==kind and r['context']==context]
            assert len(group)>=5
            assert len(set(r['state_sha256'] for r in group))==1
            ms=[r['ms_per_update'] for r in group]
            timing[str(threads)][str(context)][kind]={'median_ms':float(np.median(ms)),
                'min_ms':min(ms),'max_ms':max(ms),'transient_bytes':group[0]['transient_bytes'],
                'total_arena_bytes':group[0]['total_arena_bytes'],'repetitions':len(group)}
for row in timing_raw[1]:
    matches=[r for r in timing_raw[4] if r['kind']==row['kind'] and r['context']==row['context']]
    assert all(r['state_sha256']==row['state_sha256'] for r in matches)
kernel={}
rows=json.loads((ROOT/'results/kernel-1thread.json').read_text())
for kind in ('global','global_symmetric','projector'):
    kernel[kind]={}
    for context in (128,1024,4096):
        med={method:float(np.median([r['forward_backward_ms'] for r in rows if r['kind']==kind and r['context']==context and r['method']==method])) for method in ('scan','direct')}
        kernel[kind][str(context)]=med|{'speedup':med['direct']/med['scan']}
summary={'quality':quality,'contrasts':contrasts,'timing':timing,'kernel':kernel,
    'validation':json.loads((ROOT/'results/validation.json').read_text()),
    'common_factor_erratum':json.loads((ROOT/'results/common-factor-erratum.json').read_text()),
    'subspace':json.loads((ROOT/'results/subspace.json').read_text()),
    'streaming':json.loads((ROOT/'results/streaming.json').read_text()),
    'timing_state_identity':'all 30 kind/context cells exact across repetitions and one/four threads'}
if (ROOT/'results/recall-length.json').exists():
    length_rows=json.loads((ROOT/'results/recall-length.json').read_text())
    summary['recall_length']={str(c):{k:stat([r['test']['accuracy'] for r in length_rows if r['kind']==k and r['context']==c])
        for k in ('transformer','global','global_symmetric','projector')} for c in (64,128,256,512,1024)}
if (ROOT/'position-ablation/recall-length.json').exists():
    abrows=json.loads((ROOT/'position-ablation/recall-length.json').read_text())
    summary['position_ablation']={'primary_recall':{k:stat([json.loads((ROOT/'position-ablation'/f'{k}_norope-{seed}.json').read_text())['test']['accuracy'] for seed in SEEDS])
        for k in ('global','global_symmetric')},
        'length':{str(c):{k:stat([r['test']['accuracy'] for r in abrows if r['kind']==k and r['context']==c])
            for k in ('global','global_symmetric')} for c in (64,128,256,512,1024)}}
    if (ROOT/'position-ablation/timing-1thread.json').exists() and (ROOT/'position-ablation/timing-4thread.json').exists():
        summary['position_ablation']['timing']={}
        states={}
        for threads in (1,4):
            tr=json.loads((ROOT/'position-ablation'/f'timing-{threads}thread.json').read_text())
            summary['position_ablation']['timing'][str(threads)]={}
            for lib in ('transformer','global','global_norope','global_symmetric_norope'):
                group=[r for r in tr if r['library_name']==lib];assert len(group)==6
                hashes=set(r['state_sha256'] for r in group);assert len(hashes)==1
                if lib in states:assert states[lib]==hashes
                else:states[lib]=hashes
                ms=[r['ms_per_update'] for r in group]
                summary['position_ablation']['timing'][str(threads)][lib]={'median_ms':float(np.median(ms)),
                    'min_ms':min(ms),'max_ms':max(ms),'transient_bytes':group[0]['transient_bytes']}
if (ROOT/'results/common-mode.json').exists():summary['common_mode']=json.loads((ROOT/'results/common-mode.json').read_text())
(ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
x=np.arange(6)
for i,k in enumerate(KINDS):
    vals=quality[k]['lm_ce']['values'];axes[0,0].bar(i,np.mean(vals),color=COLORS[i],alpha=.8)
    axes[0,0].scatter([i]*3,vals,color='#182029',s=18,zorder=3)
axes[0,0].set_xticks(x,[NAMES[k].replace(' ','\n') for k in KINDS])
for ax in axes[0]:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0,0].set(title='Byte language modeling: lower is better',ylabel='Test cross-entropy (nats)',ylim=(0,3))
axes[0,1].axhline(6.25,color='#555555',ls=':',lw=1,label='Chance: 6.25%')
cs=np.array([64,128,256,512,1024])
for k,color,style,label in [('transformer',COLORS[0],'-','Attention'),('global',COLORS[3],'--','Global wedge + RoPE')]:
    vals=[summary['recall_length'][str(c)][k]['mean']*100 for c in cs]
    axes[0,1].plot(cs,vals,style+'o',color=color,label=label,ms=4)
if 'position_ablation' in summary:
    for k,color,label in [('global',COLORS[3],'Wedge, no RoPE'),('global_symmetric',COLORS[4],'Symmetric, no RoPE')]:
        axes[0,1].plot(cs,[summary['position_ablation']['length'][str(c)][k]['mean']*100 for c in cs],'-s',color=color,label=label,ms=4)
axes[0,1].axvline(128,color='#999999',ls=':',lw=1)
axes[0,1].set_xscale('log',base=2);axes[0,1].set_xticks(cs,[str(c) for c in cs])
axes[0,1].set(title='Recall length stress test · trained at 128',xlabel='Test sequence length',ylabel='Test accuracy (%)',ylim=(0,107));axes[0,1].legend(loc='center right',fontsize=9)
for i,k in enumerate(KINDS):
    contexts=np.array([64,128,256,512,1024]);cells=[timing['1'][str(c)][k] for c in contexts]
    axes[1,0].plot(contexts,[c['median_ms'] for c in cells],'-o',ms=4,label=NAMES[k],color=COLORS[i])
    axes[1,0].fill_between(contexts,[c['min_ms'] for c in cells],[c['max_ms'] for c in cells],alpha=.08,color=COLORS[i])
    axes[1,1].plot(contexts,[c['transient_bytes']/2**20 for c in cells],'-o',ms=4,label=NAMES[k],color=COLORS[i])
for ax in axes[1]:ax.set_xlabel('Sequence length (batch 1)');ax.grid(alpha=.15)
axes[1,0].set(title='Native full update: pinned CPU core',ylabel='Milliseconds / update');axes[1,0].legend(fontsize=9)
axes[1,1].set(title='Caller-owned activation + scratch arenas',ylabel='MiB (excludes persistent model state)')
fig.suptitle('BCIR native Grassmann extension · three paired seeds · ~526k parameters',fontsize=15)
fig.savefig(ROOT.parent/'BCIR_Grassmann_Global_Comparison_2026-10-17.png',dpi=160)
print(json.dumps({'quality':quality,'contrasts':contrasts,'T1024':timing['1']['1024'],'kernel':kernel},indent=2))
