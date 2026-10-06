"""Actual target-frame cloth meshes and magnitude of native/velocity displacement."""
import json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];os.environ['MPLCONFIGDIR']=str(ROOT/'reports/.mplconfig')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
q=json.loads((ROOT/'reports/V54_EXIT_COMPARISON.json').read_text())
for s in q['scenes']:
    seed=next(r for r in s['runs'] if r['kind']=='seed');native=next(r for r in s['runs'] if r['kind']=='resume' and r['repeat']==1);vel=next(r for r in s['runs'] if r['kind']=='velocity' and r['repeat']==1)
    base=ROOT/f"runs/local/{seed['run']}/trace";raw=np.fromfile(base/'topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+nf*3].reshape(-1,3);tets=raw[3+nf*3:].reshape(-1,4)
    tet=np.zeros(nv,bool);tet[tets.ravel()]=True;mask=~tet[faces].any(axis=1);cloth=faces[mask]
    edges=np.stack([cloth[:,[0,1]],cloth[:,[1,2]],cloth[:,[2,0]]],axis=1);lens=lambda x:np.linalg.norm(x[edges[:,:,0]]-x[edges[:,:,1]],axis=2)
    x0=np.fromfile(base/'state_0000.bin',dtype='<f8').reshape(-1,3);rest=lens(x0)
    data=[np.fromfile(ROOT/f"runs/local/{r['run']}/final_checkpoint/vertices.bin",dtype='<f8').reshape(-1,3) for r in [native,vel]]
    norms=[lens(x)/rest for x in data];displacement=np.linalg.norm(data[0]-data[1],axis=1)*1000
    lo=np.min([x.min(axis=0) for x in data],axis=0);hi=np.max([x.max(axis=0) for x in data],axis=0);extent=hi-lo;lo-=.03*extent;hi+=.03*extent
    stretch_norm=Normalize(1,max(x.max() for x in norms));diff_norm=Normalize(0,max(1e-8,float(displacement.max())))
    fig=plt.figure(figsize=(15,5));axes=[]
    for col,x in enumerate([data[0],data[1],data[1]]):
        ax=fig.add_subplot(1,3,col+1,projection='3d');axes.append(ax);colors=np.full((nf,4),[.70,.70,.72,1.])
        colors[mask]=plt.get_cmap('viridis')(stretch_norm(norms[col].max(axis=1))) if col<2 else plt.get_cmap('magma')(diff_norm(displacement[cloth].max(axis=1)))
        ax.add_collection3d(Poly3DCollection(x[:,[0,2,1]][faces],facecolors=colors,edgecolors='none',rasterized=True))
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[2],hi[2]),zlim=(lo[1],hi[1]));ax.set_box_aspect((hi-lo)[[0,2,1]]);ax.view_init(elev=26,azim=-65);ax.set_proj_type('ortho');ax.set_axis_off()
        ax.set_title(['Native | max stretch '+f'{norms[0].max():.6f}','Velocity only | max stretch '+f'{norms[1].max():.6f}','Position difference | max '+f'{displacement.max():.4g} mm'][col],fontsize=10)
    fig.suptitle(f"{s['scene']} | physical frame {s['target_frame']} | same checkpoint, repetition1",fontsize=12)
    fig.subplots_adjust(left=.01,right=.93,top=.86,bottom=.2,wspace=.02)
    fig.colorbar(plt.cm.ScalarMappable(norm=stretch_norm,cmap='viridis'),cax=fig.add_axes([.1,.12,.45,.025]),orientation='horizontal',label='edge length / rest (first two panels)')
    fig.colorbar(plt.cm.ScalarMappable(norm=diff_norm,cmap='magma'),cax=fig.add_axes([.68,.12,.24,.025]),orientation='horizontal',label='max vertex difference per triangle (mm)')
    out=ROOT/'reports'/f"V54_EXIT_VISUAL_{s['scene']}.png";assert not out.exists();fig.savefig(out,dpi=150);plt.close(fig)
