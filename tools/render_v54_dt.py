"""Common-time actual meshes and shared-time observables for dt refinement."""
import json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];os.environ['MPLCONFIGDIR']=str(ROOT/'reports/.mplconfig')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
q=json.loads((ROOT/'reports/V54_DT_COMPARISON.json').read_text())
for s in q['scenes']:
    selected=[next(r for r in s['runs'] if r['variant']==v and r['div']==d and r['repeat']==1) for v in ['off','guard'] for d in [1,2,4]]
    base=ROOT/f"runs/local/{selected[0]['run']}/trace";raw=np.fromfile(base/'topology.bin',dtype='<u4')
    nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+nf*3].reshape(-1,3);tets=raw[3+nf*3:].reshape(-1,4)
    tet=np.zeros(nv,bool);tet[tets.ravel()]=True;mask=~tet[faces].any(axis=1);cloth=faces[mask]
    edges=np.stack([cloth[:,[0,1]],cloth[:,[1,2]],cloth[:,[2,0]]],axis=1)
    load=lambda r,i:np.fromfile(ROOT/f"runs/local/{r['run']}/trace/state_{i:04d}.bin",dtype='<f8').reshape(-1,3)
    lens=lambda x:np.linalg.norm(x[edges[:,:,0]]-x[edges[:,:,1]],axis=2)
    rest=lens(load(selected[0],0));data=[load(r,50*r['div']) for r in selected]
    lo=np.min([x.min(axis=0) for x in data],axis=0);hi=np.max([x.max(axis=0) for x in data],axis=0);extent=hi-lo;lo-=.03*extent;hi+=.03*extent
    norm=Normalize(1,max((lens(x)/rest).max() for x in data));cmap=plt.get_cmap('viridis');fig=plt.figure(figsize=(12,7))
    for i,(r,x) in enumerate(zip(selected,data)):
        stretch=lens(x)/rest;colors=np.full((nf,4),[.7,.7,.72,1.]);colors[mask]=cmap(norm(stretch.max(axis=1)))
        ax=fig.add_subplot(2,3,i+1,projection='3d');ax.add_collection3d(Poly3DCollection(x[:,[0,2,1]][faces],facecolors=colors,edgecolors='none',rasterized=True))
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[2],hi[2]),zlim=(lo[1],hi[1]));ax.set_box_aspect((hi-lo)[[0,2,1]]);ax.view_init(elev=26,azim=-65);ax.set_proj_type('ortho');ax.set_axis_off()
        ax.set_title(f"{r['variant'].upper()} dt={r['dt']} | max {stretch.max():.5f}",fontsize=10)
    fig.suptitle(s['scene']+' | physical time 0.50s, repetition1',fontsize=12);fig.subplots_adjust(left=.01,right=.9,bottom=.02,top=.9,hspace=.02,wspace=.02)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.93,.25,.015,.5]),label='edge length / rest')
    dest=ROOT/'reports'/f"V54_DT_MESH_{s['scene']}.png";assert not dest.exists();fig.savefig(dest,dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');colors={1:'tab:blue',2:'tab:orange',4:'tab:green'}
    for r in s['runs']:
        col=0 if r['variant']=='off' else 1;color=colors[r['div']];style='-' if r['repeat']==1 else '--'
        label=f"dt={r['dt']} r{r['repeat']}";t=[o['time'] for o in r['common_stretch']]
        axes[0,col].plot(t,[o['max_stretch'] for o in r['common_stretch']],style,color=color,lw=1,label=label)
        axes[1,col].plot(t,[o['cloth_potential']+o['bending_potential'] for o in r['common_energy']],style,color=color,lw=1,label=label)
    for col,v in enumerate(['OFF','GUARD']):
        axes[0,col].set_title(v);axes[0,col].set_ylabel('max edge length / rest');axes[1,col].set_ylabel('cloth + bending potential');axes[1,col].set_xlabel('physical time (s)')
        axes[0,col].legend(fontsize=7);axes[0,col].grid(alpha=.2);axes[1,col].grid(alpha=.2)
    fig.suptitle(s['scene']+' | common .01s sample grid; dashed = repetition2')
    dest=ROOT/'reports'/f"V54_DT_CURVES_{s['scene']}.png";assert not dest.exists();fig.savefig(dest,dpi=140);plt.close(fig)
