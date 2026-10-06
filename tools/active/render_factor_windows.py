"""Common-frame exported mesh views for the finite factor-action comparison."""
import json,os
import numpy as np
from config import ROOT,read
os.environ['MPLCONFIGDIR']=str(ROOT/'reports/.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if __name__=='__main__':
    matrix=read(ROOT/'reports/active/FACTOR_WINDOW_BATCH.json');records=[]
    for key in ('hang','fixed_bunny','mixed'):
        tasks=[next(r for r in matrix['runs'] if r['scene_key']==key and r['variant']==arm and r['repeat']==1)
               for arm in ('stiff','triangular','factor_inverse')]
        runs=[ROOT/'runs/active'/t['name'] for t in tasks];end=tasks[0]['config']['steps']
        assert all(read(r/'result.json')['recorded_frames']==end for r in runs)
        raw=np.fromfile(runs[0]/'trace/topology.bin','<u4');nv,nf,nt=map(int,raw[:3])
        faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
        body=np.zeros(nv,bool);body[tets.ravel()]=True;cloth=~body[faces].any(axis=1)
        edges=np.stack([faces[cloth][:,[0,1]],faces[cloth][:,[1,2]],faces[cloth][:,[2,0]]],axis=1)
        load=lambda r,f:np.fromfile(r/f'trace/state_{f:04d}.bin','<f8').reshape(-1,3)
        lens=lambda x:np.linalg.norm(x[edges[:,:,0]]-x[edges[:,:,1]],axis=2)
        rest=lens(load(runs[0],0));series=[[float((lens(load(r,f))/rest).max()) for f in range(1,end+1)] for r in runs]
        frames=sorted({1+int(np.argmax(series[0])),1+int(np.argmax(series[2])),end})
        if len(frames)<3:frames=sorted(set(frames)|{max(1,end//2)})
        if len(frames)<3:frames=sorted(set(frames)|{1})
        positions=[load(r,f) for r in runs for f in frames]
        lo=np.min([x.min(axis=0) for x in positions],axis=0);hi=np.max([x.max(axis=0) for x in positions],axis=0)
        extent=hi-lo;lo-=extent*.04;hi+=extent*.04
        norm=Normalize(1,max(map(max,series)));cmap=plt.get_cmap('viridis')
        fig=plt.figure(figsize=(12,10))
        for row,(r,label) in enumerate(zip(runs,('StiffGIPC','TOI: triangular + warp restriction','TOI: inverse factor + warp restriction'))):
            for col,f in enumerate(frames):
                x=load(r,f);s=lens(x)/rest;colors=np.full((nf,4),[.7,.7,.72,1.]);colors[cloth]=cmap(norm(s.max(axis=1)))
                ax=fig.add_subplot(3,len(frames),row*len(frames)+col+1,projection='3d')
                ax.add_collection3d(Poly3DCollection(x[:,[0,2,1]][faces],facecolors=colors,edgecolors='none',rasterized=True))
                ax.set(xlim=(lo[0],hi[0]),ylim=(lo[2],hi[2]),zlim=(lo[1],hi[1]));ax.set_box_aspect((hi-lo)[[0,2,1]])
                ax.view_init(elev=26,azim=-65);ax.set_proj_type('ortho');ax.set_axis_off()
                ax.set_title(f'{label}\nframe {f} | cloth max {s.max():.4f}',fontsize=9)
        fig.suptitle(tasks[0]['config']['scene']+' | common frames, scale, camera and cloth color',fontsize=12)
        fig.subplots_adjust(left=.01,right=.9,bottom=.02,top=.91,hspace=.08,wspace=.02)
        bar=fig.add_axes([.93,.25,.014,.5]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=bar,label='cloth edge length / initial length')
        target=ROOT/f'reports/active/figures/factor_window_{key}.png';assert not target.exists()
        fig.savefig(target,dpi=140,bbox_inches='tight');plt.close(fig)
        records.append({'scene':key,'runs':[r.name for r in runs],'frames':frames,'image':str(target),
                        'max_cloth_stretch':list(map(max,series)),'scope':'Actual mesh morphology only; no collision or FEM-quality proof.'})
        print(json.dumps(records[-1]),flush=True)
    with (ROOT/'reports/active/FACTOR_WINDOW_VISUAL.json').open('x') as f:json.dump(records,f,indent=2)
