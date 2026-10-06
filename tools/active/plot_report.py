"""Static scientific figure from the measured 35-frame work/quality ledger."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT

data=json.loads((ROOT/'reports/active/RESTRICT_WINDOW_QUALITY.json').read_text())
groups=[('StiffGIPC',data['runs'][:3],'#505962'),
        ('TOI + serial restriction',[data['runs'][3],data['runs'][6]],'#c25435'),
        ('TOI + parallel restriction',data['runs'][4:6],'#19779b')]
fig,axes=plt.subplots(1,2,figsize=(10.8,4.3),sharex=True)
for label,runs,color in groups:
    for ax,field in zip(axes,('seconds','pcg_iterations')):
        values=np.array([[f[field] for f in r['frames']] for r in runs]);x=np.arange(1,36)
        ax.plot(x,np.median(values,axis=0),label=label,color=color,lw=1.6)
        ax.fill_between(x,values.min(axis=0),values.max(axis=0),color=color,alpha=.12,lw=0)
for ax in axes:
    ax.set_xlabel('Physical frame');ax.set_xlim(1,35);ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Solver time per frame (s)');axes[0].set_title('Execution cost falls')
axes[1].set_ylabel('PCG iterations per frame');axes[1].set_title('Excess solve work remains')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.98))
fig.text(.5,.025,'35-frame shared-desktop diagnostic. Shading: observed repeat range, not confidence interval.\nTOI quality does not pass the frozen StiffGIPC envelope; this is not same-quality speedup certification.',ha='center',fontsize=8.5)
fig.subplots_adjust(top=.79,bottom=.23,wspace=.29,left=.07,right=.98)
out=ROOT/'reports/active/figures';out.mkdir(exist_ok=True)
for extension in ('png','pdf'):fig.savefig(out/('window_cost_and_work.'+extension),dpi=180)
print(str(out/'window_cost_and_work.png'))
