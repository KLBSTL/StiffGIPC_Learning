"""Locate the first recorded nonlinear branch and meaningful trajectory divergence."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
data=ROOT/'downloads/autodl_perf_v36_20261003/runs/autodl'
paths=[data/'autodl_perf_v36_strict_base_toi_r1',data/'autodl_perf_v36_strict_base_toi_r2']
stats=[json.loads((p/'output/stats.json').read_text())['frames'] for p in paths]
raw=np.fromfile(paths[0]/'trace/topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3])
faces=raw[3:3+nf*3].reshape(-1,3);tet=raw[3+nf*3:].reshape(-1,4)
in_tet=np.zeros(nv,dtype=bool);in_tet[tet.ravel()]=True;cloth=np.unique(faces[~in_tet[faces].any(axis=1)])
mass=np.fromfile(paths[0]/'trace/masses.bin',dtype='<f8')[cloth]
x0=np.fromfile(paths[0]/'trace/state_0000.bin',dtype='<f8').reshape(-1,3)[cloth]
scale=np.linalg.norm(np.ptp(x0,axis=0))
trajectory=[];first_branch=None
for i in range(101):
    x,y=[np.fromfile(p/f'trace/state_{i:04d}.bin',dtype='<f8').reshape(-1,3)[cloth] for p in paths]
    rms=float(np.sqrt(np.average(np.sum((x-y)**2,axis=1),weights=mass))/scale)
    trajectory.append({'frame':i,'normalized_cloth_rms':rms})
    if i==0 or first_branch is not None:continue
    a,b=stats[0][i-1],stats[1][i-1]
    an,bn=a['newton'],b['newton']
    for j,(na,nb) in enumerate(zip(an,bn)):
        keys=['toi_outer','toi_inner','line_search_r','inner_exit_reason']
        differences={k:[na.get(k),nb.get(k)] for k in keys if na.get(k)!=nb.get(k)}
        if differences:
            keep=['toi_outer','toi_inner','line_search_r','line_search_backtracks','inner_exit_reason','trial_energy_before','trial_energy_after','trial_newton_axis_velocity_m_s','pcg']
            first_branch={'frame':i,'direction':j+1,'differences':differences,
                          'reference':{k:na[k] for k in keep if k in na},'repeat':{k:nb[k] for k in keep if k in nb}}
            break
    if first_branch is None and len(an)!=len(bn):first_branch={'frame':i,'direction_counts':[len(an),len(bn)]}
first_outer_difference=None
for i,(a,b) in enumerate(zip(*stats),1):
    for j,(ta,tb) in enumerate(zip(a['toi'],b['toi'])):
        diffs={k:[ta.get(k),tb.get(k)] for k in ['active_self','active_ground','active_added','active_removed'] if ta.get(k)!=tb.get(k)}
        if 'alpha' in ta and 'alpha' in tb and abs(ta['alpha']-tb['alpha'])>1e-10:
            diffs['alpha']=[ta['alpha'],tb['alpha']]
        if diffs:first_outer_difference={'frame':i,'outer':j,'differences':diffs};break
    if first_outer_difference:break
out={'scope':'First recorded nonlinear label/outer difference, not the first differing GPU instruction or a causal proof; both full_step and velocity_converged can take the same exit',
     'first_recorded_branch':first_branch,
     'first_outer_difference':first_outer_difference,
     'first_cloth_gate_exceedance':next((r for r in trajectory if r['normalized_cloth_rms']>1e-6),None),
     'first_cloth_rms_above_1e10':next((r for r in trajectory if r['normalized_cloth_rms']>1e-10),None),
     'trajectory':trajectory}
target=ROOT/'reports/AUTODL_PERF_V36_HOST_FIRST_BRANCH_EXTENDED.json';assert not target.exists()
target.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='trajectory'},indent=2))
