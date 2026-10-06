"""Reproducible endpoint FEM-J figure; diagnostics are not quality certification."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import ROOT,read

base=read(ROOT/'reports/active/DEFAULTS_COMBINED_GEOMETRY_20261005.json')
native=read(ROOT/'reports/active/DEFAULTS_NATIVE_WINDOW_GEOMETRY_20261005.json')
exit_data=read(ROOT/'reports/active/OUTER_EXIT_GEOMETRY_20261005.json')
fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
baseline=np.array([[f['fem_min_J'] for f in r['frames']] for r in base['runs'][:3]])
x=np.arange(1,36)
for ax in axes:
    ax.fill_between(x,baseline.min(axis=0),baseline.max(axis=0),color='#222222',alpha=.2,label='Stiff repeat envelope')
    ax.plot(x,np.median(baseline,axis=0),color='#222222',lw=2)
    ax.axhline(0,color='#888888',lw=.8,ls=':')
    ax.set_xlabel('Physical frame (dt = 0.01 s)');ax.set_ylabel('Minimum FEM J')
    ax.grid(alpha=.18);ax.set_ylim(-.12,1.03)
for i,r in enumerate(base['runs'][4:7],1):
    axes[0].plot(x,[f['fem_min_J'] for f in r['frames']],color='#0076a8',alpha=.6,lw=1.2,label='Two defaults + explicit warp (3 runs)' if i==1 else None)
r=native['runs'][-1]
axes[0].plot(x,[f['fem_min_J'] for f in r['frames']],color='#d34430',lw=1.7,label='All current defaults: serial (1 run)')
axes[0].set_title('Combined defaults: compression remains')
axes[0].set_xlim(20,35)
for r,label,color,ls in zip(exit_data['runs'][3:],['Native diagnostic','Selected exit probe r1','Selected exit probe r2'],['#0076a8','#d34430','#b17500'],['-','--','--']):
    axes[1].plot(x,[f['fem_min_J'] for f in r['frames']],label=label,color=color,ls=ls,lw=1.6)
axes[1].set_title('Single-outer intervention: no quality pass')
axes[1].set_xlim(32,35);axes[1].set_xticks([32,33,34,35])
for ax in axes:ax.legend(fontsize=8,loc='lower left')
fig.suptitle('RTX 3070 Laptop: continuous 35-frame diagnostics; not controlled performance tests',fontsize=10)
out=ROOT/'reports/active/figures';out.mkdir(exist_ok=True)
for ext in ('png','pdf'):fig.savefig(out/('defaults_fem_j_20261005.'+ext),dpi=180)
print(out/'defaults_fem_j_20261005.png')
