import argparse,csv,json
from pathlib import Path
import numpy as np

def states(run,indices):
    files=[run/'trace'/f'state_{i:04d}.bin' for i in indices]
    return np.stack([np.fromfile(p,dtype='<f8').reshape(-1,3) for p in files])
def solver_info(run):
    d=json.loads((run/'output/stats.json').read_text())
    frames=d['frames']
    if isinstance(frames,dict):frames=list(frames.values())
    pcg=[n['pcg'] for f in frames for n in f.get('newton',[]) if 'pcg' in n]
    return {'direction_solves':len(pcg),'pcg_iterations':[p['iterations'] for p in pcg],
            'pcg_limit_hits':sum(p.get('iteration_limit',False) for p in pcg),
            'max_true_relative_residual':max((p.get('true_relative_residual',0) for p in pcg),default=0),
            'graph_captures':max((p.get('graph_captures_total',0) for p in pcg),default=0),
            'graph_cache_hits':sum(p.get('graph_cache_hit',False) for p in pcg),
            'outer_limit_hits':sum(f.get('newton_exit')=='iteration_limit' or f.get('toi_exit')=='iteration_limit' for f in frames)}
def main():
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--graph-equivalence',action='store_true');a=p.parse_args()
    indices=sorted(set(int(p.stem[-4:]) for p in (a.reference/'trace').glob('state_*.bin')) & set(int(p.stem[-4:]) for p in (a.candidate/'trace').glob('state_*.bin')))
    if len(indices)<2:raise ValueError('Need two common state samples')
    x,y=states(a.reference,indices),states(a.candidate,indices)
    if x.shape!=y.shape:raise ValueError('Different state dimensions')
    difference=np.linalg.norm(y-x,axis=2)
    scene_scale=float(np.linalg.norm(np.ptp(x[0],axis=0)))
    mass=np.fromfile(a.reference/'trace/masses.bin',dtype='<f8')
    base_info,candidate_info=solver_info(a.reference),solver_info(a.candidate)
    report={'initial_state_identical':bool(np.array_equal(x[0],y[0])),
        'frames':indices[-1],'state_samples_compared':len(indices),'compared_frame_indices':indices,
        'trajectory_scope':'all_frames' if indices==list(range(indices[-1]+1)) else 'sampled_states',
        'max_position_difference_m':float(difference.max()),
        'max_mass_weighted_rms_m':float(np.sqrt(np.average(difference**2,axis=1,weights=mass)).max()),
        'pcg_iteration_sequence_identical':base_info['pcg_iterations']==candidate_info['pcg_iterations'],
        'reference':base_info,'candidate':candidate_info,'graph_equivalence_requested':a.graph_equivalence}
    if a.graph_equivalence:
        report['graph_position_gate']='PLAN.md section 8.2: mass-weighted trajectory RMS / initial bbox diagonal <= 1e-6; iteration equality is diagnostic'
        report['scene_scale_m']=scene_scale
        report['normalized_max_mass_rms']=report['max_mass_weighted_rms_m']/scene_scale
        report['passed_position_gate']=report['initial_state_identical'] and report['normalized_max_mass_rms']<=1e-6 and candidate_info['pcg_limit_hits']==0 and candidate_info['outer_limit_hits']==0
        report['passed']=report['passed_position_gate'] and report['trajectory_scope']=='all_frames'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['reference','candidate']}))
    if a.graph_equivalence and not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
