"""Final identity, finite geometry, CCD coverage and peak spot-check assertions."""
import json,hashlib,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads((ROOT/p).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read('manifests/perf_v54_local.json')
for f in m['files']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
for path,e in m['binaries'].items():assert sha(ROOT/path)==e['sha256']
rows=read('reports/V54_DT_MATRIX.json');assert len(rows)==16
v=read('reports/V54_DT_VERIFICATION.json');assert len(v['runs'])==16
q=read('reports/V54_DT_COMPARISON.json');spot=[];paths=0
for r in v['runs']:
    assert r['identity_verified'] and r['result']['status']=='completed' and r['result']['finite']
    assert r['flushed_stats']['limit_hits']==0 and r['flushed_stats']['breakdowns']==0
    c=r['ccd']['report'];assert c['passed'] and c['conservative_collision_flags']==0 and r['ccd_exit_code']==0
    assert c['paths_checked']==r['path_coverage']['accepted_segments']+r['path_coverage']['stationary_bridges']
    paths+=c['paths_checked']
for s in q['scenes']:
    for r in s['runs']:
        assert r['fixed_motion']==0 and r['min_area_ratio']>0
        if r['div']==1:continue
        base=ROOT/f"runs/local/{r['run']}/trace";raw=np.fromfile(base/'topology.bin',dtype='<u4')
        nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+nf*3].reshape(-1,3);tets=raw[3+nf*3:].reshape(-1,4)
        tet=np.zeros(nv,bool);tet[tets.ravel()]=True;faces=faces[~tet[faces].any(axis=1)]
        edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
        x0=np.fromfile(base/'state_0000.bin',dtype='<f8').reshape(-1,3)
        frame=round(r['peak_time']/r['dt']);x=np.fromfile(base/f'state_{frame:04d}.bin',dtype='<f8').reshape(-1,3)
        ratio=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/np.linalg.norm(x0[edges[:,0]]-x0[edges[:,1]],axis=1)
        edge=edges[int(ratio.argmax())];val=math.dist(x[edge[0]],x[edge[1]])/math.dist(x0[edge[0]],x0[edge[1]])
        assert abs(val-r['max_stretch_all_endpoints'])<1e-12
        spot.append({'run':r['run'],'frame':frame,'edge':edge.tolist(),'math_dist_ratio':val})
summary={'new_runs':16,'new_physical_frames':sum(r['result']['recorded_frames'] for r in rows),
    'source_files_verified':len(m['files']),'ccd_paths':paths,'ccd_passed':True,'all_fixed_motion_zero':True,
    'all_endpoint_triangles_positive_area':True,'pcg_limits_or_breakdowns':0,'peak_spotchecks':spot,
    'physical_truth_certified':False,'performance_certified':False}
out=ROOT/'reports/V54_DT_FINAL_CHECK.json';assert not out.exists();out.write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in summary.items() if k!='peak_spotchecks'}))
