"""Summarize completed diagnostics without labeling failed/partial runs passing."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def summarize(name):
    run=ROOT/'runs/local'/name
    requested=json.loads((run/'requested.json').read_text())
    result=json.loads((run/'result.json').read_text())
    stats_path=run/'output/stats.json'
    stats=json.loads(stats_path.read_text()) if stats_path.exists() else {'frames':[]}
    frames=stats['frames']
    newton=[n for f in frames for n in f.get('newton',[])]
    pcg=[n['pcg'] for n in newton if 'pcg' in n]
    accepted=[o for f in frames for o in f.get('toi',[]) if 'alpha' in o]
    diagonals=[f['toi_initial_hessian_diagonal'] for f in frames if 'toi_initial_hessian_diagonal' in f]
    residuals=[p['true_relative_residual'] for p in pcg if 'true_relative_residual' in p]
    directional=[n['total_energy_directional_scaled_error'] for n in newton if 'total_energy_directional_scaled_error' in n]
    answer={'run':name,'result':result,'failure':stats.get('failure'),
        'source_digest':requested['source_digest'],'exe_sha256':requested['exe_sha256'],
        'configuration':{k:requested.get(k) for k in ['pcg_tol','mu_scale','preconditioner','robust_velocity_tol','suite','clamp_trial','no_safe_injectivity','steps']},
        'deferred_stats_available':stats_path.exists(),
        'solves':len(pcg) if stats_path.exists() else None,
        'pcg_caps':sum(p.get('iteration_limit',False) for p in pcg) if stats_path.exists() else None,
        'negative_rho_stop_count':sum(p.get('rho_stop',0)<0 for p in pcg) if stats_path.exists() else None,
        'euclidean_residual_audited_solves':len(residuals),
        'euclidean_residual_max':max(residuals) if residuals else None,
        'energy_directional_audits':len(directional),'energy_directional_scaled_error_max':max(directional) if directional else None,
        'accepted_outer_steps':len(accepted),'min_accepted_alpha':min(o['alpha'] for o in accepted) if accepted else None,
        'tet_bound_smaller_than_ccd_count':sum(o.get('tet_injective_alpha',1)<o.get('full_ccd_alpha',0) for o in accepted),
        'initial_hessian_diagonal_first':diagonals[0] if diagonals else None,
        'initial_hessian_diagonal_last':diagonals[-1] if diagonals else None,
        'quality_gate':'not established; completion and geometric/physical checks remain separate'}
    log=run/'toi_diagnostic.jsonl'
    if log.exists():
        entries=[json.loads(s) for s in log.read_text().splitlines()]
        trials=[e for e in entries if 'trial_max_motion_m' in e]
        answer['trial_max_motion_m']=max((e['trial_max_motion_m'] for e in trials),default=None)
        answer['last_trial']=({k:trials[-1][k] for k in ['frame','outer','mu','trial_max_motion_m','inner_iterations']} if trials else None)
    return answer

def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[summarize(n) for n in a.runs]
    a.output.write_text(json.dumps({'scope':'TOI diagnostics; no formal speedup claim','runs':rows},indent=2),encoding='utf-8')
    print(json.dumps([{'run':r['run'],'status':r['result']['status'],'frames':r['result'].get('recorded_frames'),
        'failure':r['failure'],'pcg_caps':r['pcg_caps'],'negative_rho':r['negative_rho_stop_count'],
        'trial_max_motion_m':r.get('trial_max_motion_m'),'euclidean_residual_max':r['euclidean_residual_max']} for r in rows]))

if __name__=='__main__':main()
