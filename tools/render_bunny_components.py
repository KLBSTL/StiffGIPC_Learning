"""Actual endpoint meshes with identical camera, bounds and physical times."""
import os
from run_bunny_components import ROOT
os.environ['MPLCONFIGDIR']=str(ROOT/'reports/.mplconfig')
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
names=[('StiffGIPC','bunny_components_matrix_base_r1'),('IPC + four components','bunny_components_matrix_combined_r1'),('IPC + Cholesky','bunny_components_matrix_cholesky_r1')]
d=ROOT/'runs/local'/names[0][1]/'trace';raw=np.fromfile(d/'topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3])
faces=raw[3:3+nf*3].reshape(-1,3);tets=raw[3+nf*3:].reshape(-1,4)
import json
abd=json.loads((d/'metadata.json').read_text())['abd_point_num'];in_tet=np.zeros(nv,bool);in_tet[tets.ravel()]=True
colors=np.tile(to_rgb('#5fa0ca'),(nf,1));colors[np.all(faces<abd,axis=1)]=to_rgb('#d88a52');colors[~in_tet[faces].any(axis=1)]=to_rgb('#8fc38b')
frames=[0,40,100];data={(r,f):np.fromfile(ROOT/'runs/local'/name/f'trace/state_{f:04d}.bin',dtype='<f8').reshape(-1,3)[:,[0,2,1]] for r,(_,name) in enumerate(names) for f in frames}
lo=np.min([x.min(axis=0) for x in data.values()],axis=0);hi=np.max([x.max(axis=0) for x in data.values()],axis=0)
fig=plt.figure(figsize=(12,10))
for r,(label,name) in enumerate(names):
    for c,f in enumerate(frames):
        x=data[r,f];ax=fig.add_subplot(3,3,3*r+c+1,projection='3d');tri=x[faces]
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-30)
        light=np.array([.3,-.5,1.]);light/=np.linalg.norm(light);shade=.4+.6*np.abs(normals@light)
        ax.add_collection3d(Poly3DCollection(tri,facecolors=colors*shade[:,None],edgecolors='none',rasterized=True))
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]));ax.set_box_aspect(hi-lo)
        ax.view_init(elev=18,azim=-65);ax.set_proj_type('ortho');ax.set_axis_off();ax.set_title(f'{label}\nt={f*.01:.2f} s',fontsize=10)
fig.suptitle('Mixed bunny: same initial state and parameters\nGreen: cloth | Blue: FEM bunny | Orange: ABD bunny',fontsize=12)
fig.tight_layout(rect=(0,0,1,.94));path=ROOT/'reports/BUNNY_COMPONENTS_MESH.png';assert not path.exists();fig.savefig(path,dpi=145);plt.close(fig)
print(path)
