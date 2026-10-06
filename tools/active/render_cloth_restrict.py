"""Render actual exported meshes at common times, with a common camera and scale."""
import argparse
import json
import os
from config import ROOT, read
from prepare_cloth_benchmark import SCENES
import numpy as np
os.environ['MPLCONFIGDIR']=str(ROOT/'reports/.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replace',action='store_true');args=parser.parse_args()
    folder=ROOT/'reports/active/figures';folder.mkdir(exist_ok=True)
    output=ROOT/'reports/active/CLOTH_RESTRICT_VISUAL.json'
    if output.exists() and not args.replace:raise FileExistsError(output)
    details=[]
    for key,scene in SCENES.items():
        names=[f'cloth_restrict_{key}_{arm}_r1' for arm in ('stiff','toi_serial','toi_warp')]
        labels=['StiffGIPC','TOI + Graph, serial restriction','TOI + Graph, parallel restriction']
        base=ROOT/'runs/active'/names[0]/'trace'
        raw=np.fromfile(base/'topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3])
        faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
        assert len(tets)==nt
        in_tet=np.zeros(nv,bool);in_tet[tets.ravel()]=True
        cloth_mask=~in_tet[faces].any(axis=1);cloth=faces[cloth_mask]
        edges=np.stack([cloth[:,[0,1]],cloth[:,[1,2]],cloth[:,[2,0]]],axis=1)
        load=lambda name,f:np.fromfile(ROOT/f'runs/active/{name}/trace/state_{f:04d}.bin',dtype='<f8').reshape(-1,3)
        lengths=lambda x:np.linalg.norm(x[edges[:,:,0]]-x[edges[:,:,1]],axis=2)
        rest=lengths(load(names[0],0));assert np.min(rest)>0
        series={name:[float((lengths(load(name,f))/rest).max()) for f in range(1,101)] for name in names}
        peaks={name:1+int(np.argmax(series[name])) for name in names}
        frames=sorted({peaks[names[0]],peaks[names[2]],100})
        if len(frames)<3:frames=sorted(set(frames)|{50})
        if len(frames)<3:frames=sorted(set(frames)|{25})
        vmax=max(max(values) for values in series.values());norm=Normalize(1,vmax);cmap=plt.get_cmap('viridis')
        data=[load(name,f) for name in names for f in frames]
        lo=np.min([x.min(axis=0) for x in data],axis=0);hi=np.max([x.max(axis=0) for x in data],axis=0)
        extent=hi-lo;lo-=extent*.04;hi+=extent*.04
        fig=plt.figure(figsize=(12,10))
        for row,(name,label) in enumerate(zip(names,labels)):
            for col,f in enumerate(frames):
                x=load(name,f);stretch=lengths(x)/rest;face_stretch=stretch.max(axis=1)
                ax=fig.add_subplot(3,len(frames),row*len(frames)+col+1,projection='3d')
                colors=np.full((nf,4),[.70,.70,.72,1.]);colors[cloth_mask]=cmap(norm(face_stretch))
                ax.add_collection3d(Poly3DCollection(x[:,[0,2,1]][faces],facecolors=colors,edgecolors='none',rasterized=True))
                ax.set(xlim=(lo[0],hi[0]),ylim=(lo[2],hi[2]),zlim=(lo[1],hi[1]))
                ax.set_box_aspect((hi-lo)[[0,2,1]]);ax.view_init(elev=26,azim=-65)
                ax.set_proj_type('ortho');ax.set_axis_off()
                ax.set_title(f'{label}\nframe {f} | max stretch {stretch.max():.4f}',fontsize=9)
        fig.suptitle(scene+' | actual meshes | common times, camera, scale and colors',fontsize=12)
        fig.subplots_adjust(left=.01,right=.91,bottom=.01,top=.91,hspace=.08,wspace=.02)
        cax=fig.add_axes([.94,.25,.014,.5]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,label='edge length / initial length')
        target=folder/f'cloth_restrict_{key}.png'
        if target.exists() and not args.replace:raise FileExistsError(target)
        fig.savefig(target,dpi=140,bbox_inches='tight');plt.close(fig)
        details.append({'scene':scene,'runs':names,'frames':frames,'peak_frames':peaks,
                        'max_stretch':{name:max(values) for name,values in series.items()},'image':str(target),
                        'scope':'Visual morphology check only; visibility does not prove contact safety.'})
        output.write_text(json.dumps(details,indent=2))
        print(json.dumps(details[-1]),flush=True)


if __name__=='__main__':main()
