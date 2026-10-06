"""Inspect each run at its own maximum-stretch frame, same camera per scene."""
import json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];os.environ['MPLCONFIGDIR']=str(ROOT/'reports/.mplconfig')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
q=json.loads((ROOT/'reports/V54_CLOTH_QUALITY.json').read_text())
for s in q['scenes']:
    runs=[next(r for r in s['runs'] if r['variant']==v and r['repeat']==rep) for rep in [1,2] for v in ['off','legacy','guard']]
    base=ROOT/'runs/local'/runs[0]['name']/'trace';raw=np.fromfile(base/'topology.bin',dtype='<u4')
    nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
    tet=np.zeros(nv,bool);tet[tets.ravel()]=True;mask=~tet[faces].any(axis=1);cloth=faces[mask]
    edges=np.stack([cloth[:,[0,1]],cloth[:,[1,2]],cloth[:,[2,0]]],axis=1)
    lens=lambda x:np.linalg.norm(x[edges[:,:,0]]-x[edges[:,:,1]],axis=2)
    load=lambda r,f:np.fromfile(ROOT/f"runs/local/{r['name']}/trace/state_{f:04d}.bin",dtype='<f8').reshape(-1,3)
    rest=lens(load(runs[0],0));peaks=[max(r['observations'],key=lambda o:o['max_stretch']) for r in runs]
    data=[load(r,p['frame']) for r,p in zip(runs,peaks)]
    lo=np.min([x.min(axis=0) for x in data],axis=0);hi=np.max([x.max(axis=0) for x in data],axis=0)
    extent=hi-lo;lo-=extent*.03;hi+=extent*.03;norm=Normalize(1,max(p['max_stretch'] for p in peaks));cmap=plt.get_cmap('viridis')
    fig=plt.figure(figsize=(12,7))
    for i,(r,p,x) in enumerate(zip(runs,peaks,data)):
        stretch=lens(x)/rest;assert abs(stretch.max()-p['max_stretch'])<1e-12
        ax=fig.add_subplot(2,3,i+1,projection='3d');colors=np.full((nf,4),[.70,.70,.72,1.]);colors[mask]=cmap(norm(stretch.max(axis=1)))
        ax.add_collection3d(Poly3DCollection(x[:,[0,2,1]][faces],facecolors=colors,edgecolors='none',rasterized=True))
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[2],hi[2]),zlim=(lo[1],hi[1]));ax.set_box_aspect((hi-lo)[[0,2,1]])
        ax.view_init(elev=26,azim=-65);ax.set_proj_type('ortho');ax.set_axis_off()
        ax.set_title(f"{r['variant'].upper()} r{r['repeat']} | frame {p['frame']} | max {p['max_stretch']:.5f}",fontsize=10)
    fig.suptitle(s['scene']+' — each run at its own peak stretch (different times)',fontsize=12)
    fig.subplots_adjust(left=.01,right=.9,bottom=.02,top=.9,hspace=.03,wspace=.02)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.93,.25,.015,.5]),label='edge length / rest')
    out=ROOT/'reports'/f"V54_PEAK_{s['scene']}.png";assert not out.exists();fig.savefig(out,dpi=140);plt.close(fig)
