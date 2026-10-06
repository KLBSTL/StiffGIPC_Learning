"""Offline direction-level TOI work audit; observations are not causal intervention."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def summary(values):
    values = sorted(values)
    if not values: return None
    def percentile(p):
        x = (len(values)-1)*p; lo = int(x); hi = min(lo+1, len(values)-1)
        return values[lo] + (values[hi]-values[lo])*(x-lo)
    return {'count': len(values), 'sum': sum(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'p90': percentile(.9), 'p95': percentile(.95),
            'min': values[0], 'max': values[-1]}


def analyze(root):
    analysis_path = root / 'reports/active/AUTODL_FACTOR_WINDOW_ANALYSIS_V2.json'
    quality = read(analysis_path)
    assert quality['all_planned_runs_validated'] and quality['all_scenes_analyzed']
    scene = next(s for s in quality['scenes'] if s['scene_key'] == 'mixed')
    geometry = {r['name']: r for r in scene['runs']}
    output = {'schema': 'toi.pcg.offline_causality.v1', 'root': str(root), 'physical_certified': False,
              'performance_certified': False, 'sources': {str(analysis_path): sha(analysis_path)},
              'formulas': {
                  'work': 'W = sum over physical frames and PCG directions of iterations = N * mean_iterations_per_direction',
                  'paired_frame_increment': 'W_TOI(frame, repeat) - W_IPC_graph(frame, same repeat)',
                  'symmetric_count_contribution': '(N_TOI-N_IPC)*(mean_k_TOI+mean_k_IPC)/2',
                  'symmetric_difficulty_contribution': '(mean_k_TOI-mean_k_IPC)*(N_TOI+N_IPC)/2',
                  'decomposition_scope': 'Algebraic decomposition of counted work; not a controlled causal intervention.',
                  'rho_ratio': 'rho_stop/rho_initial; preconditioned quadratic residual criterion, not true residual.',
                  'rho_average_contraction': '(rho_stop/rho_initial) ** (1/iterations); geometric endpoint average only, not an observed per-iteration trace.',
                  'endpoint_geometry': 'FEM before=previous physical frame endpoint; after=current endpoint, from the v2 geometry analyzer; no per-inner FEM J exists.'},
              'missing_evidence': ['Per-PCG iteration rho/residual trajectories', 'True residual histories and RHS/vector components',
                    'Per-direction A/M eigenvalues or condition estimates', 'Per-contact effective Hessian/penalty weights, gamma/slack values and their changes',
                    'Per-inner FEM J/deformation', 'Same-state IPC-vs-TOI frozen A/b/M comparison'],
              'runs': [], 'paired_repeats': []}
    for role in ('stiff', 'ipc_graph', 'toi_graph'):
        for repeat in (1, 2, 3):
            name = f'autodl_window_mixed_{role}_r{repeat}'; folder = root / 'runs/active' / name
            source = folder / 'output/stats.json'; frames = read(source)['frames']; assert len(frames) == 35
            log = folder / 'run.log'; log_text = log.read_text(errors='replace')
            output['sources'].update({str(source): sha(source), str(log): sha(log)})
            request = read(folder / 'requested.json'); config = request['expanded_config']
            output['sources'][str(folder / 'requested.json')] = sha(folder / 'requested.json')
            directions, outers, physical_frames = [], [], []
            for frame, data in enumerate(frames, 1):
                geometry_frame = geometry[name]['frames'][frame-1]
                outer_by_id = {o['outer']: o for o in data.get('toi', [])}
                per_frame = []
                for index, newton in enumerate(data['newton']):
                    if 'pcg' not in newton: continue
                    pcg = newton['pcg']; outer = outer_by_id.get(newton.get('outer'), {})
                    first, last = pcg.get('rho_initial'), pcg.get('rho_stop')
                    rho_ratio = last/first if first is not None and first > 0 and last is not None else None
                    row = {'frame': frame, 'newton_record_index': index, 'outer': newton.get('outer'),
                           'inner': newton.get('inner'), 'contact_model_id': newton.get('contact_model_id'),
                           'linear_system_id': newton.get('linear_system_id', pcg.get('linear_system_id')),
                           'iterations': pcg['iterations'], 'iteration_limit': pcg.get('iteration_limit', False),
                           'breakdown': pcg.get('breakdown', False), 'rho_initial': first, 'rho_stop': last,
                           'rho_ratio': rho_ratio, 'rho_average_contraction': rho_ratio**(1/pcg['iterations'])
                               if rho_ratio and pcg['iterations'] else None,
                           'logged_active_self': outer.get('active_self'), 'logged_active_ground': outer.get('active_ground'),
                           'mu': outer.get('mu'), 'initial_guess_selection': outer.get('initial_guess_selection', {}).get('selected'),
                           **{k: newton.get(k) for k in ('inner_exit_reason', 'restart_full_step_blocked',
                               'line_search_r', 'line_search_backtracks', 'trial_newton_axis_velocity_m_s',
                               'trial_energy_before', 'trial_energy_after')}}
                    directions.append(row); per_frame.append(row)
                for outer_index, outer in outer_by_id.items():
                    records = [r for r in per_frame if r['outer'] == outer_index]
                    assert len(records) == outer['inner_iterations']
                    outers.append(outer | {'pcg_iterations': sum(r['iterations'] for r in records),
                                          'pcg_direction_iterations': [r['iterations'] for r in records],
                                          'inner_exit_reason': records[-1]['inner_exit_reason']})
                physical_frames.append({'frame': frame, 'directions': len(per_frame),
                    'iterations': sum(r['iterations'] for r in per_frame),
                    'max_direction_iterations': max(r['iterations'] for r in per_frame),
                    'outer_count': len(outer_by_id), 'safe_restarts': sum(o.get('initial_guess_selection',{}).get('selected')=='safe' for o in outer_by_id.values()),
                    'active_added': sum(o['active_added'] for o in outer_by_id.values()),
                    'active_removed': sum(o['active_removed'] for o in outer_by_id.values()),
                    'warm_start': data.get('toi_warm_start_state'), 'contact_geometry': data.get('contact_geometry'),
                    'fem_min_J_before': geometry[name]['frames'][frame-2]['fem_min_J'] if frame > 1 else 1.,
                    'fem_min_J_after': geometry_frame['fem_min_J'], 'fem_nonpositive_after': geometry_frame['fem_nonpositive'],
                    'solver_seconds': geometry_frame['seconds']})
            assert sum(r['iterations'] for r in directions) == geometry[name]['pcg_iterations']
            assert len(directions) == geometry[name]['direction_count']
            def group(selected): return summary([r['iterations'] for r in directions if selected(r)])
            run = {'name': name, 'role': role, 'repeat': repeat, 'pcg_rho_tol': config['pcg_rho_tol'],
                   'iteration_distribution': group(lambda r: True), 'direction_count': len(directions),
                   'limit_or_breakdown': sum(r['iteration_limit'] or r['breakdown'] for r in directions),
                   'windows': {f'{a}-{b}': group(lambda r: a<=r['frame']<=b) for a,b in ((1,23),(24,35),(24,26),(27,32),(33,35))},
                   'top_directions': sorted(directions,key=lambda r:r['iterations'],reverse=True)[:12],
                   'work_share_top_directions': {str(k):sum(r['iterations'] for r in sorted(directions,key=lambda r:r['iterations'],reverse=True)[:k])/sum(r['iterations'] for r in directions) for k in (1,5,10)},
                   'above_iteration_threshold': {str(k):group(lambda r:r['iterations']>=k) for k in (100,200,300,500)},
                   'inner_0': group(lambda r:r['inner']==0), 'inner_ge_1':group(lambda r:r['inner'] is not None and r['inner']>=1),
                   'safe_restart_outer_directions':group(lambda r:r['initial_guess_selection']=='safe'),
                   'other_outer_directions':group(lambda r:r['initial_guess_selection']!='safe'),
                   'guard_blocked_directions':group(lambda r:r['restart_full_step_blocked'] is True),
                   'inner_exits':dict(collections.Counter(r['inner_exit_reason'] for r in directions if r['inner_exit_reason'])),
                   'line_search_values':dict(collections.Counter(str(r['line_search_r']) for r in directions)),
                   'total_backtracks':sum(r['line_search_backtracks'] or 0 for r in directions),
                   'rho_ratio_distribution':summary([r['rho_ratio'] for r in directions if r['rho_ratio'] is not None]),
                   'rho_contraction_distribution':summary([r['rho_average_contraction'] for r in directions if r['rho_average_contraction'] is not None]),
                   'mu_distribution':summary([o['mu'] for o in outers]),
                   'run_log_lines':len(log_text.splitlines()),
                   'run_log_iterative_rho_trace_present':any('rho=' in line or 'rho_initial' in line for line in log_text.splitlines()),
                   'frames':physical_frames, 'outers':outers, 'directions':directions}
            output['runs'].append(run)
    lookup={(r['role'],r['repeat']):r for r in output['runs']}
    for repeat in (1,2,3):
        ipc, toi = lookup['ipc_graph',repeat], lookup['toi_graph',repeat]
        n0,n1=ipc['direction_count'],toi['direction_count'];k0,k1=ipc['iteration_distribution']['mean'],toi['iteration_distribution']['mean']
        delta=toi['iteration_distribution']['sum']-ipc['iteration_distribution']['sum']
        count=(n1-n0)*(k1+k0)/2;difficulty=(k1-k0)*(n1+n0)/2
        assert math.isclose(count+difficulty,delta)
        paired=[{'frame':a['frame'],'ipc_iterations':a['iterations'],'toi_iterations':b['iterations'],
                 'extra_iterations':b['iterations']-a['iterations'],'share_of_total_increment':(b['iterations']-a['iterations'])/delta}
                for a,b in zip(ipc['frames'],toi['frames'])]
        output['paired_repeats'].append({'repeat':repeat,'extra_iterations':delta,
            'direction_count_factor':n1/n0,'mean_iterations_per_direction_factor':k1/k0,
            'total_iteration_factor':toi['iteration_distribution']['sum']/ipc['iteration_distribution']['sum'],
            'symmetric_count_contribution':count,'symmetric_difficulty_contribution':difficulty,
            'symmetric_difficulty_fraction':difficulty/delta,
            'frames':paired,'top_increment_frames':sorted(paired,key=lambda p:p['extra_iterations'],reverse=True)[:8],
            'increment_by_window':{f'{a}-{b}':sum(p['extra_iterations'] for p in paired if a<=p['frame']<=b) for a,b in ((1,23),(24,35),(24,26),(27,32),(33,35))}})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); report=analyze(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'output':str(args.output),'runs':len(report['runs']),'directions':sum(r['direction_count'] for r in report['runs']),
                      'paired_repeats':[{k:v for k,v in r.items() if k not in ('frames','top_increment_frames')} for r in report['paired_repeats']]},indent=2))


if __name__=='__main__':main()
