"""Render actual saved meshes and performance/strain plots after timing ends."""
import json
import os
import argparse
import numpy as np
from config import ROOT,read
os.environ['MPLCONFIGDIR']=str(ROOT/'reports/.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ARMS=('base','graph','toi','graph_toi')
LABELS=('base','base+Graph','base+TOI','base+Graph+TOI')
COLORS=('#555555','#0076a8','#bc7319','#a51b45')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plot-only',action='store_true');args=parser.parse_args()
    report=read(ROOT/'reports/active/FOURWAY_CLOTH_RESULTS_20261005.json')
    out=ROOT/'reports/active/figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    visual=[]
    for row,(key,s) in enumerate(report['scenes'].items()):
        med=[s['summary'][a]['median_seconds'] for a in ARMS]
        lo=[med[i]-min(s['summary'][a]['seconds']) for i,a in enumerate(ARMS)]
        hi=[max(s['summary'][a]['seconds'])-med[i] for i,a in enumerate(ARMS)]
        ax=axes[row,0];bars=ax.bar(range(4),med,color=COLORS,yerr=[lo,hi],capsize=3)
        for i,bar in enumerate(bars):ax.text(bar.get_x()+bar.get_width()/2,max(s['summary'][ARMS[i]]['seconds'])+max(med)*.055,f'{med[i]:.2f}s\n{s["summary"][ARMS[i]]["median_paired_speedup_over_base"]:.3f}x',ha='center',fontsize=9)
        ax.set_xticks(range(4),LABELS,rotation=15);ax.set_ylim(0,max(med)*1.3);ax.set_ylabel('100-frame solver seconds')
        ax.set_title(key+' | median with repeat range');ax.grid(axis='y',alpha=.18)
        g=s['geometry'];names={a:[r['name'] for r in s['observations'] if r['arm']==a] for a in ARMS}
        for a,label,color in zip(ARMS,LABELS,COLORS):
            curves=np.array([[f['cloth_max_stretch'] for f in r['frames']] for r in g['runs'] if r['name'] in names[a]])
            x=np.arange(1,101);axes[row,1].plot(x,np.median(curves,axis=0),label=label,color=color,lw=1.4)
            axes[row,1].fill_between(x,curves.min(axis=0),curves.max(axis=0),color=color,alpha=.12)
        axes[row,1].set(title=key+' | accepted endpoint stretch',xlabel='Physical frame (dt=.01s)',ylabel='Maximum cloth edge / rest length')
        axes[row,1].grid(alpha=.18);axes[row,1].legend(fontsize=8)
        if args.plot_only:continue
        # Predeclare repeat 1; select common final and peak frames from real data.
        tasks=[next(o for o in s['observations'] if o['arm']==a and o['repeat']=='r1') for a in ARMS]
        runs=[ROOT/'runs/active'/t['name'] for t in tasks]
        raw=np.fromfile(runs[0]/'trace/topology.bin','<u4');nv,nf,nt=map(int,raw[:3])
        faces=raw[3:3+nf*3].reshape(-1,3);tets=raw[3+nf*3:].reshape(-1,4)
        volume=np.zeros(nv,bool);volume[tets.ravel()]=True;cloth=~volume[faces].any(axis=1)
        edges=np.stack([faces[cloth][:,[0,1]],faces[cloth][:,[1,2]],faces[cloth][:,[2,0]]],axis=1)
        load=lambda p,f:np.fromfile(p/f'trace/state_{f:04d}.bin','<f8').reshape(-1,3)
        lens=lambda x:np.linalg.norm(x[edges[:,:,0]]-x[edges[:,:,1]],axis=2)
        rest=lens(load(runs[0],0));series=[[float((lens(load(p,f))/rest).max()) for f in range(1,101)] for p in runs]
        frames=sorted({1+int(np.argmax(series[0])),1+int(np.argmax(series[3])),100})
        if len(frames)==1:frames=[50,100]
        positions=[load(p,f) for p in runs for f in frames]
        low=np.min([x.min(axis=0) for x in positions],axis=0);high=np.max([x.max(axis=0) for x in positions],axis=0)
        extent=high-low;low-=extent*.04;high+=extent*.04
        norm=Normalize(1,max(map(max,series)));cmap=plt.get_cmap('viridis')
        mesh=plt.figure(figsize=(len(frames)*3.3,12))
        for i,(p,label) in enumerate(zip(runs,LABELS)):
            for j,f in enumerate(frames):
                x=load(p,f);stretch=lens(x)/rest;colors=np.full((nf,4),[.68,.68,.7,1]);colors[cloth]=cmap(norm(stretch.max(axis=1)))
                ax=mesh.add_subplot(4,len(frames),i*len(frames)+j+1,projection='3d')
                ax.add_collection3d(Poly3DCollection(x[:,[0,2,1]][faces],facecolors=colors,edgecolors='none',rasterized=True))
                ax.set(xlim=(low[0],high[0]),ylim=(low[2],high[2]),zlim=(low[1],high[1]));ax.set_box_aspect((high-low)[[0,2,1]])
                ax.set_proj_type('ortho');ax.view_init(elev=26,azim=-65);ax.set_axis_off()
                ax.set_title(f'{label} | frame {f}\nmax cloth stretch {stretch.max():.5f}',fontsize=9)
        mesh.suptitle(key+' | actual meshes, repeat 1, common camera/scale/color',fontsize=11)
        mesh.subplots_adjust(left=.01,right=.9,bottom=.01,top=.93,hspace=.08,wspace=.01)
        bar=mesh.add_axes([.93,.25,.015,.5]);mesh.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=bar,label='cloth edge / rest length')
        target=out/f'fourway_cloth_{key}_meshes_20261005.png';mesh.savefig(target,dpi=140,bbox_inches='tight');plt.close(mesh)
        visual.append({'scene':key,'frames':frames,'runs':[p.name for p in runs],'image':target.relative_to(ROOT).as_posix(),
            'scope':'actual mesh morphology, not independent collision certification'})
    fig.suptitle('RTX 3070 Laptop | three interleaved 100-frame diagnostic repeats',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(out/f'fourway_cloth_timing_quality_20261005.{ext}',dpi=180)
    plt.close(fig)
    if not args.plot_only:
        with (ROOT/'reports/active/FOURWAY_CLOTH_VISUAL_20261005.json').open('x') as f:json.dump(visual,f,indent=2)
    print(json.dumps(visual))
