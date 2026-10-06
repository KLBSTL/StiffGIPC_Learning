"""CPU physical observables for uninterrupted controls; no invented quality pass gate."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())
rows=[];runs=[]
for mode in ['graph','host']:
    run=ROOT/'runs/local'/('v46_continuous_'+mode);d=run/'trace';req=read(run/'requested.json');res=read(run/'result.json')
    assert not req['checkpoint_load'] and not req['fixed_study'] and req['inner_exit']=='native'
    assert req['pcg_tol']==1e-4 and req['dt']==.01 and req['steps']==40 and req['mas_cholesky']=='1'
    raw=np.fromfile(d/'topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
    x0=np.fromfile(d/'state_0000.bin',dtype='<f8').reshape(-1,3);mass=np.fromfile(d/'masses.bin',dtype='<f8')
    abd=read(d/'metadata.json')['abd_point_num'];fem_tets=tets[np.all(tets>=abd,axis=1)]
    in_tet=np.zeros(nv,bool);in_tet[tets.ravel()]=True;cloth_faces=faces[~in_tet[faces].any(axis=1)]
    cloth=np.zeros(nv,bool);cloth[np.unique(cloth_faces)]=True
    edges=np.unique(np.sort(np.concatenate([cloth_faces[:,[0,1]],cloth_faces[:,[1,2]],cloth_faces[:,[2,0]]]),axis=1),axis=0)
    lengths=np.linalg.norm(x0[edges[:,0]]-x0[edges[:,1]],axis=1);assert np.all(lengths>0)
    def determinants(x):
        q=x[fem_tets];return np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))
    initial_det=determinants(x0);assert np.all(initial_det!=0)
    q={'mode':mode,'status':res['status'],'completed_frames':res.get('recorded_frames',0),'frames':[],'scope':'Endpoint observations; not converged physical error. CCD is checked separately.'}
    paths=sorted(d.glob('state_*.bin'));assert len(paths)==q['completed_frames']+1
    for path in paths[1:]:
        x=np.fromfile(path,dtype='<f8').reshape(-1,3);assert np.isfinite(x).all()
        stretch=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/lengths;J=determinants(x)/initial_det
        q['frames'].append({'frame':int(path.stem.split('_')[1]),'cloth_stretch_max':float(stretch.max()),
            'cloth_stretch_p95':float(np.percentile(stretch,95)),'fem_min_J':float(J.min()),'fem_nonpositive_tets':int(np.sum(J<=0))})
    q['physical_energy']=list(csv.DictReader((d/'physical_energy.csv').open()))
    movable_fem=in_tet & (np.arange(nv)>=abd) & (np.fromfile(d/'boundary_types.bin',dtype='<i4')!=1)
    assert movable_fem.any() and mass[movable_fem].sum()>0
    rows.append(q);runs.append((d,paths,x0,mass,cloth,movable_fem))
a,b=runs;assert (a[0]/'topology.bin').read_bytes()==(b[0]/'topology.bin').read_bytes()
assert np.array_equal(a[2],b[2]) and np.array_equal(a[3],b[3])
differences=[];scale=float(np.linalg.norm(np.ptp(a[2][a[4]],axis=0)))
fem_scale=float(np.linalg.norm(np.ptp(a[2][a[5]],axis=0)))
for p,z in zip(a[1][1:],b[1][1:]):
    assert p.name==z.name
    x=np.fromfile(p,dtype='<f8').reshape(-1,3);y=np.fromfile(z,dtype='<f8').reshape(-1,3)
    rms=float(np.sqrt(np.average(np.sum((x[a[4]]-y[a[4]])**2,axis=1),weights=a[3][a[4]])))
    fem_rms=float(np.sqrt(np.average(np.sum((x[a[5]]-y[a[5]])**2,axis=1),weights=a[3][a[5]])))
    differences.append({'frame':int(p.stem.split('_')[1]),'cloth_mass_rms_m':rms,'percent_initial_cloth_scale':100*rms/scale,
        'movable_fem_mass_rms_m':fem_rms,'percent_initial_fem_scale':100*fem_rms/fem_scale})
report={'runs':rows,'same_initial_positions_masses_topology':True,'host_graph_differences':differences,
    'quality_matched':False,'performance_certified':False,'reason':'No converged common physical reference; single diagnostic run per arm.'}
out=ROOT/'reports/DIRECTION_V46_QUALITY_R1_20261004.json';assert not out.exists();out.write_text(json.dumps(report,indent=2))
print(json.dumps({'runs':[{'mode':r['mode'],'status':r['status'],'completed_frames':r['completed_frames'],
    'last':r['frames'][-1] if r['frames'] else None} for r in rows],
    'max_cloth_rms_percent':max((x['percent_initial_cloth_scale'] for x in differences),default=None),
    'max_movable_fem_rms_percent':max((x['percent_initial_fem_scale'] for x in differences),default=None)}))
