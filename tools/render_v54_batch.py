"""Fixed-cloth geometry and measured propagation, shared camera and physical frames."""
import json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];os.environ['MPLCONFIGDIR']=str(ROOT/'reports/.mplconfig')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
q=json.loads((ROOT/'reports/V54_BATCH_COMPARISON.json').read_text());p=json.loads((ROOT/'reports/V54_PERTURB_COMPARISON.json').read_text())
configs=[('Native all frames','v54_batch_fixed_window_native_r1',23),('Velocity only all frames','v54_batch_fixed_window_velocity_only_r1',23),('Velocity frame24, then native','v54_perturb_velocity_r1',24)]
base=ROOT/'runs/local/v54_fixed_bunny_guard_r1/trace';raw=np.fromfile(base/'topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+nf*3].reshape(-1,3);tets=raw[3+nf*3:].reshape(-1,4)
tet=np.zeros(nv,bool);tet[tets.ravel()]=True;mask=~tet[faces].any(axis=1);cloth=faces[mask];edges=np.stack([cloth[:,[0,1]],cloth[:,[1,2]],cloth[:,[2,0]]],axis=1)
lens=lambda x:np.linalg.norm(x[edges[:,:,0]]-x[edges[:,:,1]],axis=2)
x0=np.fromfile(base/'state_0000.bin',dtype='<f8').reshape(-1,3);rest=lens(x0)
data=[np.fromfile(ROOT/f'runs/local/{name}/trace/state_{frame-offset:04d}.bin',dtype='<f8').reshape(-1,3) for _,name,offset in configs for frame in [24,27,30]]
lo=np.min([x.min(axis=0) for x in data],axis=0);hi=np.max([x.max(axis=0) for x in data],axis=0);extent=hi-lo;lo-=.03*extent;hi+=.03*extent
norm=Normalize(1,max((lens(x)/rest).max() for x in data));cmap=plt.get_cmap('viridis');fig=plt.figure(figsize=(12,10))
for i,x in enumerate(data):
    row,col=divmod(i,3);ax=fig.add_subplot(3,3,i+1,projection='3d');stretch=lens(x)/rest;colors=np.full((nf,4),[.7,.7,.72,1.]);colors[mask]=cmap(norm(stretch.max(axis=1)))
    ax.add_collection3d(Poly3DCollection(x[:,[0,2,1]][faces],facecolors=colors,edgecolors='none',rasterized=True))
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[2],hi[2]),zlim=(lo[1],hi[1]));ax.set_box_aspect((hi-lo)[[0,2,1]]);ax.view_init(elev=26,azim=-65);ax.set_proj_type('ortho');ax.set_axis_off()
    ax.set_title(f'{configs[row][0]}\nframe {[24,27,30][col]} | max {stretch.max():.5f}',fontsize=9)
fig.suptitle('Fixed bunny cloth | actual exported geometry, common view and color scale',fontsize=12,y=.985);fig.subplots_adjust(left=.01,right=.86,top=.89,bottom=.02,wspace=.02,hspace=.16)
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.90,.3,.015,.4]),label='edge length / rest')
dest=ROOT/'reports/V54_BATCH_FIXED_MESH_R2.png';assert not dest.exists();fig.savefig(dest,dpi=140);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for report,label,color in [(q,'Different rule every frame','tab:blue'),(p,'Different frame24 state; native thereafter','tab:orange')]:
    for i,pair in enumerate(x for x in report['fixed_pairs'] if not x['same_mode'] and x['same_repeat']):
        series=[s for s in pair['series'] if s['physical_frame']>=24];style='-' if i==0 else '--'
        for ax,key in zip(axes,['rms_m','max_vertex_m']):ax.plot([s['physical_frame'] for s in series],[s[key]*1000 for s in series],style,color=color,label=label+f' r{i+1}')
    for i,pair in enumerate(x for x in report['fixed_pairs'] if x['same_mode']):
        for ax,key in zip(axes,['rms_m','max_vertex_m']):ax.plot([s['physical_frame'] for s in pair['series'] if s['physical_frame']>=24],[s[key]*1000 for s in pair['series'] if s['physical_frame']>=24],':',color=color,alpha=.5,label='Within-group repeat' if i==0 else None)
axes[0].set_ylabel('Cloth mass-weighted RMS difference (mm)');axes[1].set_ylabel('Maximum cloth vertex difference (mm)')
for ax in axes:ax.set_xlabel('Physical frame');ax.grid(alpha=.2)
axes[0].legend(fontsize=7);fig.suptitle('Measured trajectory differences, not errors against physical ground truth')
dest=ROOT/'reports/V54_BATCH_PROPAGATION_R2.png';assert not dest.exists();fig.savefig(dest,dpi=150);plt.close(fig)
