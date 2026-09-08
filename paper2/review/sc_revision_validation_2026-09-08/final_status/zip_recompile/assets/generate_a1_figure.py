"""SC-only readable vector A1 figure; immutable accepted medians and quartiles."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);args=p.parse_args()
source=args.repo/'experiments/sci34_supplement/results/a1/sci34_f11ccba_20260901_a1/analysis.json'
rows=json.loads(source.read_text())['rows']
plt.rcParams.update({'font.size':12,'axes.labelsize':13,'legend.fontsize':12,'xtick.labelsize':11,'ytick.labelsize':11,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(6.3,4.25))
x=[r['target_length'] for r in rows]
for key,label,color,marker in [('crop_role_joint_ms','Joint crop + role repair','#0072B2','o'),('reprefill_ms','Full re-prefill','#D55E00','s')]:
 s=[r['statistics'][key] for r in rows];assert all(v['n']==50 for v in s)
 med=[v['median'] for v in s]
 ax.errorbar(x,med,yerr=[[v['median']-v['q1'] for v in s],[v['q3']-v['median'] for v in s]],label=label,color=color,marker=marker,linewidth=1.8,markersize=6,capsize=4)
ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(x,[str(v) for v in x]);ax.set_yticks([30,100,300,1000,3000]);ax.yaxis.set_major_formatter(ScalarFormatter())
ax.set_ylim(20,5000);ax.set_xlim(205,10500);ax.set_xlabel('Initial context length (tokens)');ax.set_ylabel('Latency (ms, logarithmic scale)')
ax.grid(axis='y',alpha=.25);ax.legend(loc='upper left',frameon=False)
for r in rows:
 ratio=r['speedup_reprefill_over_joint_median'];assert abs(ratio-r['statistics']['reprefill_ms']['median']/r['statistics']['crop_role_joint_ms']['median'])<1e-12
 ax.annotate(f'{ratio:.2f}×',(r['target_length'],r['statistics']['crop_role_joint_ms']['median']),xytext=(0,13),textcoords='offset points',ha='center',fontsize=11,color='#004A73')
fig.subplots_adjust(left=.14,right=.98,top=.97,bottom=.23)
fig.text(.14,.055,'Bars: interquartile range; 50 repetitions per length.\nLabels: median re-prefill / joint-repair ratio.',fontsize=11,ha='left')
out=Path(__file__).resolve().parents[1]/'figures/Figure_5_cost_microbenchmark.pdf'
fig.savefig(out);plt.close(fig)
print(out)
