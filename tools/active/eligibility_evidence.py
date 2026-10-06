"""Detailed offline evidence after each component round, including counterexamples."""
import itertools
import argparse
import json
import math
import statistics
from collections import defaultdict
from config import ROOT, read, sha
from ipc_benchmark import write
from eligibility_round import TAG, pairs
from analyze_ipc_light_cost import analyze
from ipc_benchmark import metrics


def median(values):
    return statistics.median(values) if values else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hang-subset',action='store_true')
    args=parser.parse_args()
    suffix='_hang_subset' if args.hang_subset else ''
    result = {'guard_summary': {}, 'paired_stages': {}, 'profiles': [], 'quality_protocol_available': None,
              'performance_certified': False, 'quality_certified': False, 'default_promoted': False,
              'method': {'performance': 'Paired ratios within each round; exact three-round resampling is diagnostic only.',
                         'profile': 'Actual GPU kernel sums and CPU NVTX wall separate; nested scopes and waits never added.',
                         'quality': 'Whole-segment frozen metrics. Candidate passes do not repair failed baseline calibration.'}}
    guard_path = ROOT / f'reports/active/{TAG}_guards_batch.json'
    for row in read(guard_path)['runs']:
        folder = ROOT/'runs/active'/row['name']
        frames = read(folder/'output/stats.json')['frames']
        counters = row['checks'].get('eligibility_counters')
        if counters is None:
            counters = {kind: {key: (max(f['bvh_eligibility'][kind][key] for f in frames)
                                       if key in ('scratch_bytes_peak','old_peak_capacity','new_peak_capacity')
                                       else sum(f['bvh_eligibility'][kind][key] for f in frames))
                               for key in frames[0]['bvh_eligibility'][kind]}
                        for kind in ('ordinary_vf','ordinary_ee','swept_vf','swept_ee')}
        request = read(folder/'requested.json')
        before = request['gpu_before'][-1]['free_mib']
        samples = row['result']['gpu_samples']
        memory = {'before_free_mib': before, 'budget_mib': request['memory_budget_mib'],
                  'minimum_sampled_free_mib': min(g['free_mib'] for g in samples),
                  'maximum_sampled_global_decrease_mib': before-min(g['free_mib'] for g in samples),
                  'attribution': 'Global GPU free-memory decrease includes this process and other applications.'}
        prefix_checks = {kind: {'all_saved_queries_validated': c['validation_calls']==c['prepare_calls']==c['validation_passed']
                                                         and c['validation_failed']==0,
                                'refresh_coverage': c['enabled_queries']==c['prepare_calls']+c['empty_queries'],
                                'pruned_roots_over_candidate_nodes_tested':
                                    sum(c[k] for k in ('diagnostic_pruned_body','diagnostic_pruned_fixed','diagnostic_pruned_id'))
                                    /c['diagnostic_nodes_tested'] if c['diagnostic_nodes_tested'] else None}
                         for kind,c in counters.items()}
        result['guard_summary'][row['scene_key']] = {'passed': row['checks']['passed'],
              'run_status': row['result']['status'], 'recorded_frames': len(frames), 'requested_frames': row['config']['steps'],
              'counters': counters, 'saved_prefix_checks': prefix_checks, 'memory': memory,
              'prefix_metrics': {k:v for k,v in metrics(folder).items() if k!='frames'},
              'prefix_limit': 'Saved complete frames only; an interrupted following frame is not verified.'}
    for stage in ('screen','full'):
        path = ROOT / f'reports/active/{TAG}_{stage}{suffix if stage=="screen" else ""}_batch.json'
        if not path.exists():
            result['paired_stages'][stage] = {'attempted': False, 'reason': 'Prerequisite or finite selection gate'}
            continue
        batch = read(path)
        paired, groups = pairs(batch)
        summary = {'attempted': True, 'batch_sha256': sha(path), 'paired_speed': paired, 'scenes': {}}
        complete = all(r['complete_pairs'] == 3 for r in paired.values())
        if complete:
            ratios = [[paired[s]['off_over_on'][i] for s in ('hang','fixed_bunny')] for i in range(3)]
            samples = sorted(math.sqrt(math.prod(median([ratios[i][s] for i in sample]) for s in (0,1)))
                             for sample in itertools.product(range(3), repeat=3))
            summary['geometric_mean_of_paired_medians'] = math.sqrt(math.prod(r['median'] for r in paired.values()))
            summary['diagnostic_exact27_bootstrap_lower95'] = samples[1]
            summary['bootstrap_limit'] = 'Only three shared-load rounds; no controlled-performance certification.'
        for scene in ('hang','fixed_bunny'):
            scene_result = {}
            for arm in ('stiff','off','on'):
                rows = list(groups[(scene,arm)].values())
                if not rows: continue
                metric = [r['checks']['metrics'] for r in rows]
                names = [ROOT/'runs/active'/r['name'] for r in rows]
                per_direction_iterations = [[n['pcg']['iterations'] for f in read(p/'output/stats.json')['frames']
                                            for n in f['newton'] if 'pcg' in n] for p in names]
                scene_result[arm] = {
                    'completed': len(rows), 'median_seconds': median([r['result']['solver_seconds'] for r in rows]),
                    'pcg_range': [min(m['pcg'] for m in metric),max(m['pcg'] for m in metric)],
                    'directions_range': [min(m['directions'] for m in metric),max(m['directions'] for m in metric)],
                    'iterations_per_direction_mean': [sum(v)/len(v) for v in per_direction_iterations],
                    'iterations_per_direction_max': [max(v) for v in per_direction_iterations],
                    'phase_ms_medians': {k: median([m['phase_ms'][k] for m in metric]) for k in metric[0]['phase_ms']},
                    'max_stretch_range': [min(m['max_stretch'] for m in metric),max(m['max_stretch'] for m in metric)],
                    'p99_stretch_range': [min(m['p99_stretch'] for m in metric),max(m['p99_stretch'] for m in metric)],
                    'material_passed': sum(r['checks']['material_100f_passed'] is True for r in rows),
                    'material_failed': sum(r['checks']['material_100f_passed'] is False for r in rows),
                    'material_fails': [{'name': r['name'], 'metrics': {k:v for k,v in (r['checks']['material_checks'] or {}).items() if not v['passed']}} for r in rows if r['checks']['material_100f_passed'] is False],
                    'gpu_sm_clock_range': [min(g['sm_mhz'] for r in rows for g in r['result']['gpu_samples']),max(g['sm_mhz'] for r in rows for g in r['result']['gpu_samples'])],
                    'counters': [r['checks']['eligibility_counters'] for r in rows]}
            stiff = groups[(scene,'stiff')]
            if stiff:
                for arm in ('off','on'):
                    values = [stiff[i]['result']['solver_seconds']/r['result']['solver_seconds']
                              for i,r in groups[(scene,arm)].items() if i in stiff]
                    scene_result[arm]['paired_vs_stiff'] = {'values': values, 'median': median(values)}
            summary['scenes'][scene] = scene_result
        result['paired_stages'][stage] = summary
    profile_path = ROOT / f'reports/active/{TAG}_profile{suffix}_batch.json'
    if profile_path.exists():
        for row in read(profile_path)['runs']:
            path = ROOT / 'runs/active' / row['name'] / 'nsight.sqlite'
            if not path.exists(): continue
            d = analyze(path, 0)
            assert d['captured_physical_frames'] == 1
            kernels = d['kernel_hotspots']['rows']
            a = d['aggregate']
            counters = row['checks']['eligibility_per_frame'][-1]
            result['profiles'].append({'name': row['name'], 'scene': row['scene_key'], 'arm': row['arm'],
                'captured_frame': row['config']['steps'], 'sqlite_sha256': sha(path),
                'cpu_frame_wall_ms': a['cpu_physical_frame_wall']['sum_ms'],
                'gpu_union_ms': a['gpu_timeline_clipped']['all']['union_ms'],
                'query_gpu_sum_ms': sum(k['sum_ms'] for k in kernels if '_selfQuery_' in k['name']),
                'eligibility_gpu_sum_ms': sum(k['sum_ms'] for k in kernels if 'eligibility_' in k['name']),
                'query_kernels': sorted([k for k in kernels if '_selfQuery_' in k['name'] or 'eligibility_' in k['name']],key=lambda r:r['sum_ms'],reverse=True),
                'eligibility_stages': [r for r in d['stages_alphabetical'] if 'eligibility' in r['name']],
                'target_frame_counters': counters,
                'target_frame_pcg': sum(n['pcg']['iterations'] for n in read(path.parent/'output/stats.json')['frames'][-1]['newton'] if 'pcg' in n),
                'method': d['method']})
    analysis_path = ROOT / f'reports/active/{TAG}{suffix}_analysis.json'
    result['paired_states'] = read(analysis_path)['paired_states']
    result['analysis_sha256'] = sha(analysis_path)
    full = result['paired_stages']['full']
    if full['attempted']:
        result['quality_protocol_available'] = all(full['scenes'][s]['stiff']['material_passed']==3 for s in ('hang','fixed_bunny'))
    result['independent_accepted_path_CPU_CCD_this_round'] = False
    result['scene_scope']='hang' if args.hang_subset else None
    write(ROOT / f'reports/active/{TAG}{suffix}_evidence.json', result)
    print(json.dumps({'guard_scenes': list(result['guard_summary']), 'profile_count': len(result['profiles']),
                      'paired_stages': {k: {f:v for f,v in r.items() if f in ('attempted','geometric_mean_of_paired_medians','diagnostic_exact27_bootstrap_lower95')} for k,r in result['paired_stages'].items()},
                      'quality_protocol_available': result['quality_protocol_available']}))


if __name__ == '__main__': main()
