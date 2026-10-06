"""Current finite paired observations; never certifies quality or 2x speed."""
import argparse
import itertools
import json
import statistics
import numpy as np
from analyze import analyze
from config import ROOT, read, sha
from validate_run import validate


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    target=ROOT/a.output
    if target.exists():raise FileExistsError(target)
    arms=('stiff','generalized','world')
    folders=[ROOT/'runs/active'/f'world_penalty_mixed_{arm}_r{r}' for arm in arms for r in (1,2,3)]
    geometry=analyze(folders,3)
    by_name={r['name']:r for r in geometry['runs']}
    observations=[]
    for folder in folders:
        arm=folder.name.split('_')[-2]
        req=read(folder/'requested.json');result=read(folder/'result.json')
        if result['status']!='completed' or result['recorded_frames']!=35 or not result.get('finite'):
            raise ValueError('Incomplete run: '+folder.name)
        if arm!='stiff' and not validate(folder)['passed']:raise ValueError('Configuration gate failed')
        f=read(folder/'output/stats.json')['frames']
        rows=by_name[folder.name]['frames']
        calls=sum(r['pcg_calls'] for r in rows);work=sum(r['pcg_iterations'] for r in rows)
        observed={'name':folder.name,'arm':arm,'repeat':int(folder.name[-1]),
            'solver_seconds':result['solver_seconds'],'pcg_calls':calls,'pcg_iterations':work,
            'pcg_per_call':work/calls,'outer_count':sum(len(row.get('toi',[])) for row in f),
            'min_fem_J':min(r['fem_min_J'] for r in rows),
            'max_nonpositive_fem_tets':max(r['fem_nonpositive'] for r in rows),
            'max_negative_fem_volume':max(r['fem_negative_volume'] for r in rows),
            'max_cloth_stretch':max(r['cloth_max_stretch'] for r in rows),
            'min_abd_J':min(r['abd_min_J'] for r in rows),
            'fixed_max_drift':max(r['fixed_max_drift'] for r in rows),
            'world_mu_range': [min(row['toi_world_hessian_diagonal']['world_mu'] for row in f),
                               max(row['toi_world_hessian_diagonal']['world_mu'] for row in f)] if arm=='world' else None,
            'resolved_config_confirmed':arm!='stiff','velocity_frames':len(list((folder/'trace').glob('velocity_*.bin'))),
            'requested_sha256':sha(folder/'requested.json'),'result_sha256':sha(folder/'result.json'),
            'build_manifest_sha256':sha(folder/'build_manifest.json'),'exe_sha256':req['exe_sha256']}
        systems=[(i+1,n) for i,row in enumerate(f) for n in row['newton'] if 'pcg' in n]
        observed['pcg_frames1_23']=sum(n['pcg']['iterations'] for frame,n in systems if frame<=23)
        observed['pcg_frames24_35']=sum(n['pcg']['iterations'] for frame,n in systems if frame>=24)
        observed['pcg_inner_ge1']=sum(n['pcg']['iterations'] for _,n in systems if n.get('toi_inner',0)>=1)
        observed['outer_accept_alpha_below_half']=sum(o['alpha']<.5 for row in f for o in row.get('toi',[]))
        observed['outer_with_one_inner_direction']=sum(o['inner_iterations']==1 for row in f for o in row.get('toi',[]))
        observations.append(observed)
    medians={arm:{key:statistics.median(r[key] for r in observations if r['arm']==arm)
                 for key in ('solver_seconds','pcg_calls','pcg_iterations','pcg_per_call','outer_count')}
             for arm in arms}
    pairs=[]
    for repeat in (1,2,3):
        group={r['arm']:r for r in observations if r['repeat']==repeat}
        pairs.append({'repeat':repeat,'generalized_over_world_seconds':group['generalized']['solver_seconds']/group['world']['solver_seconds'],
            'stiff_over_world_seconds':group['stiff']['solver_seconds']/group['world']['solver_seconds'],
            'pcg_work_reduction_fraction':1-group['world']['pcg_iterations']/group['generalized']['pcg_iterations']})
    # Check when the trajectories first diverge. This does not assume atomics
    # are deterministic, or treat matching inputs as matching physical outputs.
    divergences=[]
    for repeat in (1,2,3):
        runs=[ROOT/'runs/active'/f'world_penalty_mixed_{arm}_r{repeat}' for arm in ('generalized','world')]
        first_bit=None;first_material=None;max_before_contact=0.
        for frame in range(36):
            x=[np.fromfile(run/'trace'/f'state_{frame:04d}.bin','<f8') for run in runs]
            difference=float(np.max(np.abs(x[0]-x[1]),initial=0))
            if frame<24:max_before_contact=max(max_before_contact,difference)
            if first_bit is None and not np.array_equal(*x):first_bit=frame
            if first_material is None and difference>1e-8:first_material=frame
        divergences.append({'repeat':repeat,'first_bitwise_difference_frame':first_bit,
            'first_max_coordinate_difference_gt_1e_8_m_frame':first_material,
            'max_coordinate_difference_before_frame24_m':max_before_contact})
    result={'scope':'Three interleaved 35-frame shared-desktop causal diagnostics, not 100-frame controlled benchmark',
        'performance_certified':False,'quality_certified':False,'candidate_default':'generalized',
        'promotion':'blocked by relative Stiff quality gate; no AutoDL/100/300-frame promotion',
        'script_sha256':sha(__file__),'observations':observations,'medians':medians,'paired_observations':pairs,
        'median_generalized_over_world_ratio':statistics.median(r['generalized_over_world_seconds'] for r in pairs),
        'median_stiff_over_world_ratio':statistics.median(r['stiff_over_world_seconds'] for r in pairs),
        'trajectory_divergences':divergences,'geometry':geometry}
    target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ('medians','paired_observations','median_generalized_over_world_ratio','median_stiff_over_world_ratio','trajectory_divergences')}))
    print(json.dumps({'quality':[{k:r[k] for k in ('name','min_fem_J','max_nonpositive_fem_tets','velocity_frames')} for r in observations]}))

if __name__=='__main__':main()
