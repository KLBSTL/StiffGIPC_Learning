"""CPU-only reassessment of frozen v45 evidence; does not alter solver gates."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
from analyze_perf_v36 import matrix_from_snapshot
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
systems=[]
for run,name in [('v45r2_fixed_smoke','f3_n1'),('v45r2_branch_default','f35_n1')]:
    prefix=ROOT/'runs/local'/run/'fixed'/name
    meta,A,b=matrix_from_snapshot(prefix);normA=float(np.max(np.asarray(abs(A).sum(axis=1))));normb=float(np.linalg.norm(b))
    rows=[]
    for phase,tol in [(1,1e-4),(3,1e-16)]:
        for mode,label in enumerate(['host','graph']):
            x=np.fromfile(f'{prefix}_p{phase}_m{mode}_x.bin',dtype='<f8');r=b-A@x
            rows.append({'rho_tol':tol,'mode':label,'rhs_l2':normb,'residual_l2':float(np.linalg.norm(r)),
                'relative_l2':float(np.linalg.norm(r)/max(normb,1e-300)),
                'normwise_backward_error_inf':float(np.linalg.norm(r,np.inf)/(normA*np.linalg.norm(x,np.inf)+np.linalg.norm(b,np.inf))),
                'componentwise_backward_error':float(np.max(np.abs(r)/np.maximum(abs(A)@np.abs(x)+np.abs(b),1e-300)))})
    systems.append({'run':run,'dofs':meta['dofs'],'rows':rows,'preconditioner_export_complete':meta['preconditioner_export_complete'],
        'scope':'Backward error is supplementary, not forward solution accuracy or physical quality; original 1e-8 residual gate retained.',
        'snapshot_sha256':{p.name:sha(p) for p in prefix.parent.glob(name+'*') if p.is_file()}})
branches=[]
for name in ['default','strict','velocity']:
    run=ROOT/'runs/local'/('v45r2_branch_'+name);events=[json.loads(s) for s in (run/'stage.jsonl').read_text().splitlines()]
    inners=[r['newton'] for r in events if r['stage']=='inner_end'];pcg=[r['pcg'] for r in events if r['stage']=='pcg_end']
    stats=read(run/'output/stats.json')['frames'] if (run/'output/stats.json').exists() else []
    reasons=collections.Counter(n.get('inner_exit_reason') for f in stats for n in f.get('newton',[]) if n.get('inner_exit_reason'))
    changes=[(n['trial_energy_before']-n['trial_energy_after'])/max(abs(n['trial_energy_before']),1e-300) for n in inners]
    branches.append({'run':run.name,'result':read(run/'result.json'),'completed_pcg':len(pcg),
        'total_pcg_iterations':sum(n['iterations'] for n in pcg),'exit_reasons':dict(reasons),
        'completed_inner':len(inners),'accepted_outer':sum(r['stage']=='outer_end' for r in events),
        'last_velocity':inners[-1]['trial_newton_axis_velocity_m_s'],
        'last_energy_relative_decrease':changes[-1],'last_100_energy_decrease_range':[min(changes[-100:]),max(changes[-100:])],
        'last_line_search_r':inners[-1]['line_search_r']})
out=ROOT/'reports/DIRECTION_V46_CPU_20261004.json';assert not out.exists()
out.write_text(json.dumps({'systems':systems,'branches':branches,'thresholds_changed':False,'gpu_runs_started':False},indent=2))
print(json.dumps({'systems':[{k:s[k] for k in ['run','rows','preconditioner_export_complete']} for s in systems],'branches':branches},indent=2))
