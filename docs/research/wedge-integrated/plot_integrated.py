"""Scientific figure from audited aggregates; no generated benchmark values."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;s=json.loads((ROOT/'results/summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.titleweight':'bold',
 'figure.facecolor':'#f4f6fa','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#a4aeba','grid.color':'#e4e9f0'})
fig,axes=plt.subplots(2,2,figsize=(16,11.5),gridspec_kw={'hspace':.5,'wspace':.28})
blue,green,purple,orange,gray='#2076ae','#159d80','#7655ae','#d47a37','#6e7c91'
ax=axes[0,0]
tr={r['case']:r for r in s['training']}
vals=[tr['ordered_padded_global']['training_seconds'],tr['ordered_bucket_global']['training_seconds'],s['matched_attention32']['training_seconds']]
means=[r['mean'] for r in vals];ys=np.arange(3)
ax.barh(ys,means,color=[gray,blue,purple],height=.58)
ax.errorbar(means,ys,xerr=[[r['mean']-r['min'] for r in vals],[r['max']-r['mean'] for r in vals]],fmt='none',color='#293c55',capsize=4)
for y,x in zip(ys,means):ax.text(x+.4,y,f'{x:.2f} s',va='center',fontweight='bold')
ax.set_yticks(ys,['Global, padded','Global, trimmed','Attention, accumulated']);ax.invert_yaxis();ax.set_xlim(0,27)
ax.set_xlabel('Training execution seconds, excluding validation and calibration')
ax.set_title('A  Trimming preserves the head and removes work',loc='left',pad=26)
ax.text(0,1.025,'900 updates · same effective batch 4 · 41.7% fewer padded positions',transform=ax.transAxes,fontsize=9,color='#53637b')
ax.grid(axis='x');ax.set_axisbelow(True)

ax=axes[0,1];ts=(256,1024,2048)
for case,label,color in [('ordered_bucket_global','Global, 32 MiB budget',blue),('ordered_bucket_attention','Attention, 64 MiB budget',gray),('ordered_bucket_attention32','Attention, 32 MiB budget',purple)]:
 rr=[next(r for r in s['matched_attention32']['quality'] if r['case']==case and r['context']==t) for t in ts]
 ax.plot(ts,[r['accuracy']['mean']*100 for r in rr],'o-',lw=2,ms=5,color=color,label=label)
 ax.fill_between(ts,[r['accuracy']['min']*100 for r in rr],[r['accuracy']['max']*100 for r in rr],color=color,alpha=.1)
ax.set_xscale('log',base=2);ax.set_xticks(ts,[str(t) for t in ts]);ax.set_ylim(20,103);ax.set_yticks([25,50,75,100])
ax.set_xlabel('Evaluation sequence length');ax.set_ylabel('Ordered-recall accuracy (%)');ax.grid(axis='y');ax.set_axisbelow(True)
ax.set_title('B  A fair memory budget narrows the advantage',loc='left',pad=26)
ax.text(0,1.025,'Fresh paired tests · three seeds · training lengths 64/128/256',transform=ax.transAxes,fontsize=9,color='#53637b')
ax.legend(loc='lower left',fontsize=9,framealpha=.93)

ax=axes[1,0];ts=(1024,2048,4096)
for role,label,color in [('before','Before continuation',gray),('short_control','40 extra updates at T128',green),('long_adaptive','40 updates at T1024, adaptive budget',blue)]:
 rr=[next(r for r in s['continuation_extension'] if r['role']==role and r['context']==t) for t in ts]
 ax.plot(ts,[r['accuracy']['mean']*100 for r in rr],'o-',lw=2,ms=5,color=color,label=label)
 ax.fill_between(ts,[r['accuracy']['min']*100 for r in rr],[r['accuracy']['max']*100 for r in rr],color=color,alpha=.08)
ax.set_xscale('log',base=2);ax.set_xticks(ts,[str(t) for t in ts]);ax.set_ylim(0,103);ax.set_yticks([0,25,50,75,100])
ax.set_xlabel('Evaluation sequence length');ax.set_ylabel('Ordered-recall accuracy (%)');ax.grid(axis='y');ax.set_axisbelow(True)
ax.set_title('C  Additional training helps, with a context trade-off',loc='left',pad=26)
ax.text(0,1.025,'Long tail uses 8× the input positions · 128/64/32 examples per seed',transform=ax.transAxes,fontsize=9,color='#53637b')
ax.legend(loc='lower left',fontsize=8.5,framealpha=.93)

ax=axes[1,1];rr=s['language'];ys=np.arange(3);means=[r['ce_nats']['mean'] for r in rr]
ax.barh(ys,means,color=[blue,orange,purple],height=.58)
ax.errorbar(means,ys,xerr=[[r['ce_nats']['mean']-r['ce_nats']['min'] for r in rr],[r['ce_nats']['max']-r['ce_nats']['mean'] for r in rr]],fmt='none',color='#293c55',capsize=4)
for y,x in zip(ys,means):ax.text(x+.025,y,f'{x:.3f}',va='center',fontweight='bold')
ax.set_yticks(ys,['Global','Language hybrid','Attention']);ax.invert_yaxis();ax.set_xlim(0,2.8)
ax.set_xlabel('Cross-entropy (nats/byte; lower is better)');ax.grid(axis='x');ax.set_axisbelow(True)
ax.set_title('D  Keep the language hybrid as its own candidate',loc='left',pad=26)
ax.text(0,1.025,'Matched 153,600 training bytes · paired test lengths 32/64/128',transform=ax.transAxes,fontsize=9,color='#53637b')
fig.suptitle('BCIR full global recipe: integrated execution and quality',x=.07,ha='left',y=.98,fontsize=20,fontweight='bold',color='#142a45')
fig.text(.07,.939,'33 training runs + 9 continuation arms  |  0.526M parameters  |  Native C CPU execution, full global head retained',fontsize=11,color='#53637b')
fig.subplots_adjust(left=.15,right=.965,top=.855,bottom=.11)
fig.text(.07,.035,'2026-10-18  |  Means and ranges across three seeds. Controlled buffer budgets exclude process overhead; CPU timing is provisional on a shared host.',fontsize=9,color='#53637b')
out=ROOT/'BCIR_Global_Integrated_Recipe_Frontier_2026-10-18.png';fig.savefig(out,dpi=170);plt.close(fig);print(out)
