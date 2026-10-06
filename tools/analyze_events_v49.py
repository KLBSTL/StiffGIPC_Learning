"""Independent CPU event checks. No GPU solve and no accuracy-gate relaxation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from analyze_perf_v36 import matrix_from_snapshot
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=ROOT/'runs/local'/a.run
req=read(run/'requested.json');res=read(run/'result.json');manifest=read(ROOT/'manifests/perf_v49_local.json')
assert req['source_digest']==manifest['source_digest']
assert req['exe_sha256']==manifest['binaries']['builds/local-v49/Release/gipc.exe']['sha256']
raw=np.fromfile(run/'trace/topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3]);tets=raw[3+3*nf:].reshape(-1,4)
abd=read(run/'trace/metadata.json')['abd_point_num'];tets=tets[np.all(tets>=abd,axis=1)]
x0=np.fromfile(run/'trace/state_0000.bin',dtype='<f8').reshape(-1,3)
fixed=np.fromfile(run/'trace/boundary_types.bin',dtype='<i4')==1
def det(x):
    t=x[tets];return np.einsum('ij,ij->i',t[:,1]-t[:,0],np.cross(t[:,2]-t[:,0],t[:,3]-t[:,0]))
rest=det(x0)
progress=[]
for line in (run/'events/progress.jsonl').read_text().splitlines():
    try:progress.append(json.loads(line))
    except json.JSONDecodeError:pass
rows=[]
for path in sorted((run/'events').glob('*_event.json')):
    event=read(path);prefix=path.parent/path.name.removesuffix('_event.json')
    completed=Path(str(prefix)+'_line_search.json')
    if completed.exists():
        try:event=read(completed)
        except json.JSONDecodeError:pass # Preserve committed pre-line-search metadata on interruption.
    meta,A,b=matrix_from_snapshot(prefix);x=np.fromfile(str(prefix)+'_solution.bin',dtype='<f8')
    vertices=np.fromfile(str(prefix)+'_vertices.bin',dtype='<f8').reshape(-1,3)
    d=np.fromfile(str(prefix)+'_direction.bin',dtype='<f8').reshape(-1,3)
    safe=np.fromfile(str(prefix)+'_safe.bin',dtype='<f8').reshape(-1,3)
    assert vertices.shape==d.shape==safe.shape==(nv,3) and len(x)==len(b)
    assert all(np.isfinite(z).all() for z in [x,b,d,vertices,safe,A.data])
    velocity=float(np.max(np.abs(d))/req['dt']);assert np.isclose(velocity,event['trial_velocity'],rtol=1e-12,atol=1e-12)
    assert event['capture_state_unchanged'] and event['streak'] in [req['event_first'],req['event_second']]
    assert event['frame']>=req['event_from'] and event['trial_velocity']>event['trigger_velocity']
    assert (Path(str(prefix)+'_contacts.bin').stat().st_size)==event['contacts']*event['contact_bytes']
    residual=b-A@x;relative=float(np.linalg.norm(residual)/max(np.linalg.norm(b),1e-300));bx=float(b@x);xax=float(x@(A@x))
    J=det(vertices)/rest if len(rest) else np.empty(0)
    positions=np.argsort(np.max(np.abs(d),axis=1))[-5:][::-1]
    r={'event':event,'cpu_true_relative_residual':relative,'gpu_true_relative_residual':event['pcg'].get('true_relative_residual'),
       'negative_gradient_dot_step':-bx,'x_A_x':xax,'direction_rayleigh_quotient':float(xax/max(x@x,1e-300)),
       'max_trial_safe_distance_m':float(np.linalg.norm(vertices-safe,axis=1).max()),
       'fem_min_J':float(J.min()) if len(J) else None,'fem_nonpositive_tets':int(np.sum(J<=0)),
       'top_direction_vertices':[{'id':int(i),'kind':'ABD' if i<abd else 'FEM','fixed':bool(fixed[i]),'axis_velocity':float(np.max(np.abs(d[i]))/req['dt'])} for i in positions],
       'full_preconditioner_exported':meta['preconditioner_export_complete']}
    gpu=r['gpu_true_relative_residual'];assert gpu is not None and abs(gpu-relative)<=1e-8*max(1.,relative)
    for stage in ['before','after']:
        if 'energy_components_'+stage in event:
            total=sum(event['energy_components_'+stage].values());error=abs(total-event['energy_'+stage])
            assert error<=1e-8*max(1.,abs(event['energy_'+stage]))
            r['energy_sum_error_'+stage]=error
    if event['line_search_completed']:
        assert event['post_probe_state_unchanged']
        alpha=event['line_search_r'];endpoint=np.fromfile(str(prefix)+'_after_vertices.bin',dtype='<f8').reshape(-1,3)
        mask=(np.arange(nv)>=abd)&~fixed
        error=float(np.max(np.abs(endpoint[mask]-(vertices-alpha*d)[mask]))) if mask.any() else 0
        assert error<=1e-10
        r['fem_step_sign_max_abs_error']=error
        r['predicted_quadratic_energy_delta']=-alpha*bx+.5*alpha*alpha*xax
        r['actual_energy_delta']=event['energy_after']-event['energy_before']
        r['energy_component_deltas']={k:event['energy_components_after'][k]-v for k,v in event['energy_components_before'].items()}
    same=[e for e in progress if e['frame']==event['frame'] and e['outer']==event['outer'] and (e['stage']=='outer_end' or e['inner']>=event['inner'])]
    r['outer_completed_later']=any(e['stage']=='outer_end' for e in same)
    r['last_progress_in_outer']=same[-1] if same else None
    r['sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in path.parent.glob(prefix.name+'_*')}
    rows.append(r)
assert 1<=len(rows)<=4
out=ROOT/'reports'/f'EVENTS_{a.run}_CPU.json';assert not out.exists()
out.write_text(json.dumps({'run':a.run,'result':res,'events':rows,'scope':'Independent exported A/b/x, geometry and energy checks; not accurate counterfactual solve, full M replay or restart.'},indent=2))
for r in rows:print(json.dumps({'frame':r['event']['frame'],'outer':r['event']['outer'],'inner':r['event']['inner'],
    'velocity':r['event']['trial_velocity'],'residual':r['cpu_true_relative_residual'],'fem_min_J':r['fem_min_J'],
    'predicted_delta':r.get('predicted_quadratic_energy_delta'),'actual_delta':r.get('actual_energy_delta'),'outer_completed':r['outer_completed_later']}))
