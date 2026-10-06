"""Independent geometric observations and shared-scale rendered trace comparison.
No physical-accuracy gate is inferred from these observations.
"""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--output',type=Path,required=True)
p.add_argument('--render',action='store_true');a=p.parse_args();assert not a.output.exists()
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dirs=[ROOT/'runs/local'/n for n in a.runs];d=dirs[0]/'trace'
raw=np.fromfile(d/'topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3])
faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
assert len(tets)==nt
load=lambda d,i:np.fromfile(d/f'state_{i:04d}.bin',dtype='<f8').reshape(-1,3)
x0=load(d,0);mass=np.fromfile(d/'masses.bin',dtype='<f8');abd=read(d/'metadata.json')['abd_point_num']
boundary=np.fromfile(d/'boundary_types.bin',dtype='<i4');in_tet=np.zeros(nv,bool);in_tet[tets.ravel()]=True
cloth_faces=faces[~in_tet[faces].any(axis=1)];cloth=np.zeros(nv,bool);cloth[np.unique(cloth_faces)]=True
groups={'cloth':cloth,'fem':in_tet & (np.arange(nv)>=abd) & (boundary!=1),
        'abd':(np.arange(nv)<abd) & (boundary!=1)}
edges=np.unique(np.sort(np.concatenate([cloth_faces[:,[0,1]],cloth_faces[:,[1,2]],cloth_faces[:,[2,0]]]),axis=1),axis=0)
lengths=np.linalg.norm(x0[edges[:,0]]-x0[edges[:,1]],axis=1)
fem=tets[np.all(tets>=abd,axis=1)]
def det(x):
    q=x[fem];return np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))
rest=det(x0);assert np.all(rest!=0) and np.all(lengths>0)
scales={k:float(np.linalg.norm(np.ptp(x0[v],axis=0))) for k,v in groups.items()}
rows=[];counts=[]
for run in dirs:
    z=run/'trace';req=read(run/'requested.json');res=read(run/'result.json');n=res['recorded_frames'];counts.append(n)
    for file in ['topology.bin','masses.bin','boundary_types.bin','state_0000.bin','metadata.json']:
        assert (d/file).read_bytes()==(z/file).read_bytes(),(run.name,file)
    assert read(run/'output/scene.json')==read(dirs[0]/'output/scene.json'),run.name
    observations=[]
    for i in range(1,n+1):
        x=load(z,i);assert x.shape==(nv,3) and np.isfinite(x).all()
        J=det(x)/rest;stretch=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/lengths
        observations.append({'frame':i,'fem_min_J':float(J.min()),'nonpositive_tets':int((J<=0).sum()),
            'cloth_max_stretch':float(stretch.max()),'cloth_p95_stretch':float(np.percentile(stretch,95))})
    rows.append({'run':run.name,'result':res,'requested_sha256':sha(run/'requested.json'),
        'observations':observations,'worst_J':min(r['fem_min_J'] for r in observations),
        'max_nonpositive_tets':max(r['nonpositive_tets'] for r in observations),
        'max_cloth_stretch':max(r['cloth_max_stretch'] for r in observations)})
comparisons=[]
for ia in range(len(dirs)):
    for ib in range(ia+1,len(dirs)):
        diffs=[];n=min(counts[ia],counts[ib])
        for i in range(1,n+1):
            delta=load(dirs[ia]/'trace',i)-load(dirs[ib]/'trace',i);dist2=np.sum(delta**2,axis=1)
            r={'frame':i,'whole_mass_rms_over_bbox':float(np.sqrt(np.average(dist2,weights=mass))/np.linalg.norm(np.ptp(x0,axis=0)))}
            for k,mask in groups.items():r[k+'_rms_percent_scale']=float(100*np.sqrt(np.average(dist2[mask],weights=mass[mask]))/scales[k])
            diffs.append(r)
        comparisons.append({'a':dirs[ia].name,'b':dirs[ib].name,'common_frames':n,'frames':diffs,
            'maxima':{k:max(r[k] for r in diffs) for k in diffs[0] if k!='frame'},
            'existing_1e_6_position_gate_passes':max(r['whole_mass_rms_over_bbox'] for r in diffs)<=1e-6})
report={'same_scene_initial_state_topology_masses_boundary':True,'scales_m':scales,'runs':rows,
    'comparisons':comparisons,'physical_quality_certified':False,'performance_certified':False,
    'scope':'Endpoint observations and pairwise trajectory differences; neither Stiff nor TOI is assumed ground truth. Existing PLAN 1e-6 position gate is diagnostic for execution-mode equivalence only.'}
a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({'runs':[{k:v for k,v in r.items() if k not in ['observations','result']} for r in rows],
    'comparisons':[{k:v for k,v in c.items() if k!='frames'} for c in comparisons]}),flush=True)
if a.render:
    os.environ['MPLCONFIGDIR']=str(ROOT/'reports/.mplconfig')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Real mesh surfaces, same camera and physical bounds in every panel.
    keyframes=[0,40,min(counts)];fig=plt.figure(figsize=(12,3.5*len(dirs)))
    face_color=np.full(len(faces),'#5fa0ca',dtype=object)
    face_color[np.all(faces<abd,axis=1)]='#d88a52';face_color[~in_tet[faces].any(axis=1)]='#8fc38b'
    all_key=[load(run/'trace',i) for run in dirs for i in keyframes]
    lo=np.min([x.min(axis=0) for x in all_key],axis=0);hi=np.max([x.max(axis=0) for x in all_key],axis=0)
    for row,run in enumerate(dirs):
        for col,i in enumerate(keyframes):
            x=load(run/'trace',i)[:,[0,2,1]];ax=fig.add_subplot(len(dirs),3,row*3+col+1,projection='3d')
            triangles=x[faces];normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-30)
            light=np.array([.3,-.5,1.]);light/=np.linalg.norm(light)
            shade=.4+.6*np.abs(normals@light)
            colors=np.array([to_rgb(c) for c in face_color])*shade[:,None]
            ax.add_collection3d(Poly3DCollection(triangles,facecolors=colors,edgecolors='none',rasterized=True))
            floor=np.array([[lo[0],lo[2],-1],[hi[0],lo[2],-1],[hi[0],hi[2],-1],[lo[0],hi[2],-1]])
            # Outline only: a filled plane causes mplot3d collection-order
            # occlusion of above-ground cloth; it is not a depth-buffer renderer.
            ax.plot(*np.vstack([floor,floor[0]]).T,color='#bbbbbb',linewidth=.4)
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[2],hi[2]),zlim=(lo[1],hi[1]))
            ax.set_box_aspect((hi-lo)[[0,2,1]]);ax.view_init(elev=18,azim=-65);ax.set_proj_type('ortho')
            ax.set_axis_off();ax.set_title(f'{run.name}\nframe {i}',fontsize=9)
    fig.tight_layout();fig.savefig(a.output.with_suffix('.png'),dpi=135);plt.close(fig)
