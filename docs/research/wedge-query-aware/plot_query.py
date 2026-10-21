"""Standalone scientific figure from retained per-seed test measurements."""
from experiment import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

s=json.loads((ROOT/'results/summary.json').read_text())
models=('global','query_full','query_matched','attention');labels=('Full global','+ contraction','Matched + contraction','Attention32')
colors=('#4b647f','#d47b28','#299181','#735ab1')
fig,axes=plt.subplots(2,2,figsize=(12,9.4));fig.subplots_adjust(left=.075,right=.975,bottom=.18,top=.88,wspace=.29,hspace=.70)
fig.suptitle('Query-aware full wedge head: measured learning outcomes',fontsize=17,x=.075,ha='left',y=.965,weight='bold')
fig.text(.075,.918,'3 independent training seeds · 32 MiB controlled training buffers · 900 recall / 600 language updates',fontsize=10,color='#4c5560')
for ax,task,title in zip(axes[0],('kv_single','kv_curriculum'),('Single-pair value retrieval','Four-pair key/value association')):
    vals=[s['primary_association'][f'{task}/{m}']['accuracy'] for m in models]
    means=[v['mean']*100 for v in vals];ax.bar(range(4),means,color=colors,width=.65)
    for i,v in enumerate(vals):
        ax.scatter([i]*3,np.array(v['values'])*100,s=18,color='#15232e',zorder=4)
        ax.annotate(f'{means[i]:.2f}%',(i,max(v['values'])*100),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
    ax.axhline(6.25,color='#9b3434',linestyle='--',linewidth=1,label='16-label chance: 6.25%')
    ax.set_xticks(range(4),labels,rotation=18,ha='right',fontsize=9);ax.set_ylabel('Held-out accuracy (%)');ax.set_title(title,loc='left',fontsize=12,weight='bold')
    ax.set_ylim(0,115 if task=='kv_single' else max(42,max(means)+12));ax.legend(frameon=False,fontsize=8,loc='upper left' if task=='kv_single' else 'upper right')
    ax.text(0,-.37,'Paired T128 + T256; dots show the three seed results.',transform=ax.transAxes,fontsize=8,color='#4c5560')
ax=axes[1,0]
for m,label,color in zip(models,labels,colors):
    rs=[r for r in s['curves'] if r['task']=='ordered' and r['model']==m]
    ax.plot([r['context'] for r in rs],[100*r['accuracy']['mean'] for r in rs],marker='o',color=color,label=label,linewidth=1.8)
ax.set_xscale('log',base=2);ax.set_xticks([128,256,512,1024,2048],['128','256','512','1024','2048']);ax.set_ylim(0,105)
ax.set_ylabel('Held-out accuracy (%)');ax.set_xlabel('Sequence length');ax.set_title('Ordered-recall control',loc='left',fontsize=12,weight='bold');ax.legend(fontsize=8,frameon=False,loc='lower left')
ax=axes[1,1];ms=(*models[:3],'hybrid','attention');ls=(*labels[:3],'Language hybrid','Attention32');cs=(*colors[:3],'#9b8965',colors[3])
means=[s['language'][m]['ce_nats']['mean'] for m in ms];ax.bar(range(5),means,color=cs,width=.65)
for i,m in enumerate(ms):
    ax.scatter([i]*3,s['language'][m]['ce_nats']['values'],s=18,color='#15232e',zorder=4)
    ax.annotate(f'{means[i]:.3f}',(i,means[i]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
ax.set_xticks(range(5),ls,rotation=22,ha='right',fontsize=9);ax.set_ylim(0,max(means)*1.17)
ax.set_ylabel('Cross-entropy (nats / byte; lower is better)');ax.set_title('Byte-language control',loc='left',fontsize=12,weight='bold')
for ax in axes.flat:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.text(.075,.035,'All global arms retain the full 231-coordinate head and common precision guard. Buffers exclude process overhead; no PyTorch throughput claim.',fontsize=8,color='#4c5560')
out=ROOT/'BCIR_Query_Aware_Head_Results_2026-10-18.png';fig.savefig(out,dpi=180,facecolor='white');print(out)
