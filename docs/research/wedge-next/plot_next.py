"""Export a source-backed scientific figure from the retained summary."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'results/summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.titleweight':'bold',
    'axes.labelsize':10,'figure.facecolor':'#f4f6fa','axes.facecolor':'white','axes.spines.top':False,
    'axes.spines.right':False,'grid.color':'#e4e9f0','axes.edgecolor':'#a4aeba','savefig.facecolor':'#f4f6fa'})
fig,axes=plt.subplots(2,2,figsize=(15.8,11.5),gridspec_kw={'hspace':.48,'wspace':.26})
colors={'baseline':'#6e7c91','compact':'#2076ae','frame_ff_compact':'#159d80','layer_compact':'#8b62b3','layer_frame_ff':'#c67b26',
    'global':'#2076ae','hybrid':'#d47a37','attention':'#7655ae','variable':'#159d80'}
ax=axes[0,0]
labels={'baseline':'Original global','compact':'Buffer reuse','frame_ff_compact':'Frame + FF recompute',
    'layer_compact':'Layer recompute','layer_frame_ff':'Layer + frame + FF'}
offset={'baseline':(9,-12),'compact':(9,7),'frame_ff_compact':(7,8),'layer_compact':(6,9),'layer_frame_ff':(7,-17)}
for r in s['timings']['limits_1thread']:
    if r['name'] not in labels:continue
    x,y=r['arena_mib'],r['ms']['median'];c=colors[r['name']]
    ax.errorbar(x,y,yerr=[[y-r['ms']['min']],[r['ms']['max']-y]],fmt='o',ms=8,color=c,capsize=3,lw=1.3)
    ax.annotate(labels[r['name']],(x,y),xytext=offset[r['name']],textcoords='offset points',fontsize=9,
        ha='left',color=c)
ax.set_title('A  Exact execution: memory versus update time',loc='left',pad=25)
ax.text(0,1.025,'Same model and tested training state · AMD EPYC · one pinned CPU',transform=ax.transAxes,fontsize=9,color='#53637b')
ax.set_xlabel('Activation + scratch arena (MiB; excludes 8.02 MiB persistent state)')
ax.set_ylabel('Milliseconds per update at B=1, T=1,024')
ax.set_xlim(5,41);ax.set_ylim(46,79);ax.grid(axis='y');ax.set_axisbelow(True)

ax=axes[0,1];ts=(64,128,256,512,1024)
series=[('recall_fixed','global','Global, fixed length',colors['global']),
        ('recall_variable','global','Global, variable length',colors['variable']),
        ('recall_fixed','attention','Attention, fixed length',colors['attention'])]
for task,n,label,c in series:
    rr=[next(r for r in s['length_stress'] if r['task']==task and r['kind']=='disjoint_vocab' and r['name']==n and r['context']==t) for t in ts]
    ys=[r['accuracy']['mean']*100 for r in rr];lo=[r['accuracy']['min']*100 for r in rr];hi=[r['accuracy']['max']*100 for r in rr]
    ax.plot(ts,ys,'o-',label=label,color=c,lw=2,ms=4);ax.fill_between(ts,lo,hi,color=c,alpha=.09)
rr=[next(r for r in s['followup'] if r['task']=='hybrid_length' and r['context']==t) for t in ts]
ax.plot(ts,[r['accuracy']['mean']*100 for r in rr],'o--',color=colors['hybrid'],label='Equal-blend hybrid, fixed length',lw=2,ms=4)
ax.fill_between(ts,[r['accuracy']['min']*100 for r in rr],[r['accuracy']['max']*100 for r in rr],color=colors['hybrid'],alpha=.09)
ax.set_title('B  Recall extrapolation needs its own quality gate',loc='left',pad=25)
ax.text(0,1.025,'Disjoint-vocabulary first-token task · mean and range of three seeds',transform=ax.transAxes,fontsize=9,color='#53637b')
ax.set_xscale('log',base=2);ax.set_xticks(ts,[str(t) for t in ts]);ax.set_ylim(15,103);ax.set_yticks([25,50,75,100])
ax.set_xlabel('Evaluation sequence length');ax.set_ylabel('Recall accuracy (%)');ax.grid(axis='y');ax.set_axisbelow(True)
ax.legend(loc='lower left',fontsize=8.5,framealpha=.92)

ax=axes[1,0]
paired={r['name']:r for r in s['hybrid_fresh_paired_test'] if r['task']=='byte_lm'}
attention=next(r for r in s['followup'] if r['task']=='byte_lm_hybrid_paired')
vals=[paired['global']['test_ce_nats'],paired['hybrid']['test_ce_nats'],attention['ce_nats']]
ys=np.arange(3);means=[r['mean'] for r in vals]
ax.barh(ys,means,color=[colors[n] for n in ('global','hybrid','attention')],height=.58)
ax.errorbar(means,ys,xerr=[[r['mean']-r['min'] for r in vals],[r['max']-r['mean'] for r in vals]],fmt='none',color='#293c55',capsize=4,lw=1.5)
for y,v in zip(ys,means):ax.text(v+.025,y,f'{v:.3f}',va='center',fontsize=11,fontweight='bold')
ax.set_yticks(ys,['Global','Hybrid','Attention']);ax.invert_yaxis();ax.set_xlim(0,2.85)
ax.set_title('C  A local component improves language loss',loc='left',pad=25)
ax.text(0,1.025,'Fresh paired WikiText-2 byte windows · same training budgets · three seeds',transform=ax.transAxes,fontsize=9,color='#53637b')
ax.set_xlabel('Cross-entropy (nats per byte; lower is better)');ax.grid(axis='x');ax.set_axisbelow(True)

ax=axes[1,1]
names=('baseline','compact','frame_ff_compact','layer_compact')
rr=[next(r for r in s['ham']['scenarios'] if r['variant']==n and r['capacity_mib']==64) for n in names]
xx=np.arange(len(rr));mb=[r['wire_transfer_bytes']/2**20 for r in rr]
ax.bar(xx,mb,color=[colors[n] for n in names],width=.6)
for x,r,m in zip(xx,rr,mb):
    ax.text(x,m+5,f'{m:.1f} MiB\n{r["capacity_evictions"]} capacity evictions',ha='center',va='bottom',fontsize=9)
ax.set_xticks(xx,['Original','Buffer\nreuse','Frame + FF\nrecompute','Layer\nrecompute'])
ax.set_title('D  BCIR HAM: capacity thresholds amplify savings',loc='left',pad=25)
ax.text(0,1.025,'Verified planner scenario · 64 MiB bank · six alternating read-only accesses',transform=ax.transAxes,fontsize=9,color='#53637b')
ax.set_ylabel('Predicted transfer volume (MiB; no hardware DMA executed)');ax.set_ylim(0,290);ax.grid(axis='y');ax.set_axisbelow(True)
fig.suptitle('BCIR global wedge: the next improvement frontier',x=.07,ha='left',y=.98,fontsize=21,fontweight='bold',color='#142a45')
fig.text(.07,.938,'57 native-C learning runs · 28,200 updates · exact execution changes and architectural changes evaluated separately',fontsize=11,color='#53637b')
fig.subplots_adjust(left=.085,right=.965,top=.855,bottom=.11)
fig.text(.085,.035,'2026-10-17  |  Timings: six sequential repetitions, bars show observed ranges. Small-model CPU experiments; no PyTorch speed or hardware-optimality claim.',fontsize=9,color='#53637b')
path=ROOT/'BCIR_Wedge_Memory_Quality_Frontier_2026-10-17.png';fig.savefig(path,dpi=170);plt.close(fig);print(path)
