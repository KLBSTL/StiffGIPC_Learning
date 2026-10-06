"""Actual exported cloth meshes, common camera/scale and edge-stretch colors."""
import json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];os.environ['MPLCONFIGDIR']=str(ROOT/'reports/.mplconfig')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
q=json.loads((ROOT/'reports/V54_CLOTH_QUALITY.json').read_text());details=[]
for s in q['scenes']:
    off=next(r for r in s['runs'] if r['switch']=='0' and r['repeat']==1)
    on=next(r for r in s['runs'] if r['variant']=='guard' and r['repeat']==1)
    frames={'cloth_hang_l':[4,50,100],'cloth_sphere7_l':[25,51,100],'cloth_fixed_bunny_l':[23,24,100]}[s['scene']]
    base=ROOT/'runs/local'/off['name']/'trace';raw=np.fromfile(base/'topology.bin',dtype='<u4')
    nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
    in_tet=np.zeros(nv,bool);in_tet[tets.ravel()]=True;cloth_mask=~in_tet[faces].any(axis=1);cloth=faces[cloth_mask]
    load=lambda name,f:np.fromfile(ROOT/f'runs/local/{name}/trace/state_{f:04d}.bin',dtype='<f8').reshape(-1,3)
    x0=load(off['name'],0)
    tri_edges=np.stack([cloth[:,[0,1]],cloth[:,[1,2]],cloth[:,[2,0]]],axis=1)
    lens=lambda x:np.linalg.norm(x[tri_edges[:,:,0]]-x[tri_edges[:,:,1]],axis=2)
    rest=lens(x0);vmax=max(r['max_stretch'] for r in s['runs']);norm=Normalize(1,vmax);cmap=plt.get_cmap('viridis')
    data=[load(r['name'],f) for r in [off,on] for f in frames]
    lo=np.min([x.min(axis=0) for x in data],axis=0);hi=np.max([x.max(axis=0) for x in data],axis=0)
    extent=hi-lo;lo-=extent*.03;hi+=extent*.03
    fig=plt.figure(figsize=(12,7));axes=[]
    for row,r in enumerate([off,on]):
        for col,f in enumerate(frames):
            x=load(r['name'],f);stretch=lens(x)/rest;face_stretch=stretch.max(axis=1)
            ax=fig.add_subplot(2,3,row*3+col+1,projection='3d');axes.append(ax);plotx=x[:,[0,2,1]]
            colors=np.full((nf,4),[.70,.70,.72,1.]);colors[cloth_mask]=cmap(norm(face_stretch))
            ax.add_collection3d(Poly3DCollection(plotx[faces],facecolors=colors,edgecolors='none',rasterized=True))
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[2],hi[2]),zlim=(lo[1],hi[1]))
            ax.set_box_aspect((hi-lo)[[0,2,1]]);ax.view_init(elev=26,azim=-65);ax.set_proj_type('ortho');ax.set_axis_off()
            ax.set_title(f'{"OFF" if row==0 else "GUARD"} | frame {f} | max {stretch.max():.4f}',fontsize=10)
            k=np.unravel_index(np.argmax(stretch),stretch.shape);edge=tri_edges[k[0],k[1]]
            details.append({'scene':s['scene'],'run':r['name'],'frame':f,'edge':edge.tolist(),
                'rest_length_m':float(rest[k]),'length_m':float(lens(x)[k]),'ratio':float(stretch[k])})
    fig.suptitle(s['scene']+' — same initial cloth/material; color = max triangle edge length / rest',fontsize=11)
    fig.subplots_adjust(left=.01,right=.9,bottom=.02,top=.90,hspace=.03,wspace=.02)
    cax=fig.add_axes([.93,.25,.015,.5]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,label='edge length / rest')
    out=ROOT/'reports'/f'V54_CLOTH_VISUAL_{s["scene"]}.png';assert not out.exists()
    fig.savefig(out,dpi=140);plt.close(fig)
target=ROOT/'reports/V54_CLOTH_EDGE_SPOTCHECK.json';assert not target.exists();target.write_text(json.dumps(details,indent=2))
