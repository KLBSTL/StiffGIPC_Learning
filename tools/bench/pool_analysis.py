"""CPU-only analysis of one immutable stage; no GPU or hidden continuation."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from config import read, sha, expand, digest
from validate_run import validate
from plans import WINDOWS, tasks
from pool_metrics import (metrics, window, pool_evidence, export_evidence,
                          initial_inputs, state_comparison, require)

PROTOCOL = Path(__file__).with_name('quality_protocol.json')

def summarize_run(run, task, manifest):
    row = {k: task[k] for k in ('name', 'scene_key', 'variant', 'stage')}
    row.update(hard_checks_passed=False, cloth_material_support=False, failures=[])
    try:
        req, result = read(run/'requested.json'), read(run/'result.json')
        require(req['expanded_config'] == expand(task['config']), 'Expanded config differs from plan')
        require(req['source_digest'] == manifest['source_digest'] and req['exe_sha256'] == manifest['exe_sha256'], 'Run/build identity mismatch')
        require(result['status'] == 'completed' and result['recorded_frames'] == task['config']['steps'], 'Incomplete run')
        row['configuration'] = validate(run)
        require(row['configuration']['passed'], 'Resolved/observed configuration failed')
        c = req['expanded_config']
        require(read(run/'output/scene.json')['case_id']==c['scene'], 'Actual scene differs from plan')
        resolved = read(run/'resolved_config.json')['report_components']
        require(all(resolved.get(k) == c[k] for k in ('contact_pool', 'contact_pool_validate')), 'Pool requested/resolved mismatch')
        frames = read(run/'output/stats.json')['frames']
        require(len(frames) == c['steps'], 'Incomplete stats')
        with (run/'trace/frames.csv').open() as stream:
            times = list(csv.DictReader(stream))
        require([int(t['frame']) for t in times] == list(range(1, c['steps']+1)), 'Timing frame sequence differs')
        require(all(math.isfinite(float(t['solver_ms'])) and float(t['solver_ms']) > 0 for t in times), 'Invalid timing')
        row['whole'] = window(frames, times, 1, c['steps'])
        row['contact_window'] = window(frames, times, *WINDOWS[task['scene_key']])
        for kind in ('state', 'velocity'):
            row[kind+'_exports'] = export_evidence(run, c['steps'], kind)
            require(row[kind+'_exports']['passed'], 'Incomplete or nonfinite actual '+kind)
        row['pool'] = pool_evidence(frames, c['contact_pool'], c['contact_pool_validate'])
        require(row['pool']['passed'], 'Typed pool/activation/energy guard failed')
        m = metrics(run)
        require(m['finite'] and not m['pcg_failures'], 'Nonfinite or PCG failure')
        require(all(value is None or not isinstance(value,(int,float)) or math.isfinite(value)
                    for entry in m['frames'] for value in entry.values()), 'Nonfinite physical metric')
        require(not any(x['abd_min_J'] is not None and x['abd_min_J'] <= 0 for x in m['frames']), 'ABD inversion')
        require(all(math.isfinite(n['alpha']) for f in frames for n in f['newton'] if 'alpha' in n), 'Nonfinite alpha')
        if task['scene_key'] != 'mixed':
            require(not any(x['fem_nonpositive'] for x in m['frames']), 'Cloth guard unexpected FEM inversion')
        row['material_by_frame'] = m['frames']
        row['material'] = {k: m[k] for k in ('max_stretch', 'p99_stretch', 'fixed_drift_m')}
        bounds = read(PROTOCOL)['scenes'].get({'fixed': 'fixed_bunny'}.get(task['scene_key'],task['scene_key']), {}).get('bounds')
        row['material_bounds'] = bounds
        if bounds:
            row['cloth_material_support'] = all(m[k] <= value for k, value in bounds.items())
            require(row['cloth_material_support'], 'Frozen cloth material bounds failed')
        else:
            row['mixed_quality_status'] = 'pending; no pre-frozen mixed trajectory or FEM tolerance; not a cloth-screen prerequisite'
        if task['scene_key'] == 'mixed' and c['contact_pool']:
            row['mixed_activation_windows'] = {}
            for key, lo, hi in [('middle',24,26),('late',33,35)]:
                selected = frames[lo-1:hi]
                active = any(f.get('contact_geometry',{}).get('native_narrow_self_pairs',0)
                             or f.get('contact_geometry',{}).get('native_narrow_ground_pairs',0)
                             or any(n.get('active_pairs',0) for n in f['newton']) for f in selected)
                reused = sum(f['contact_pool']['reused_queries'] for f in selected)
                row['mixed_activation_windows'][key] = {'active_pairs_observed':bool(active),'reused_queries':reused}
                require(active and reused > 0, 'Mixed '+key+' lacks contact-pool coverage')
        row['hard_checks_passed'] = True
    except (ValueError, KeyError, FileNotFoundError, AssertionError, IndexError) as error:
        row['failures'].append(type(error).__name__+': '+str(error))
    return row

def analyze(session, stage, manifest):
    selected = tasks(stage)
    rows = [summarize_run(session/t['name'],t,manifest) for t in selected]
    result = {'schema':'flat_contact_pool_analysis.v1','stage':stage,'runs':rows,'pairs':{},
              'performance_certified':False,'quality_certified':False,'mixed_quality_certified':False,
              'protocol_sha256':sha(PROTOCOL), 'analysis_sha256':sha(__file__),
              'source_digest':manifest['source_digest'],'exe_sha256':manifest['exe_sha256'],
              'allow_next_round':all(r['hard_checks_passed'] for r in rows),
              'scope':'Bounded screen support only. Actual state/velocity export overhead remains in every arm. No independent CCD certification.'}
    for scene in (('mixed',) if stage=='guards' else ('hang','fixed')):
        pair = {r['variant']:r for r in rows if r['scene_key']==scene}
        if set(pair) != {'on','off'} or not all(r['hard_checks_passed'] for r in pair.values()):
            result['allow_next_round'] = False
            continue
        off,on = (session/pair[v]['name'] for v in ('off','on'))
        p = {'initial_inputs':initial_inputs(off,on)}
        result['pairs'][scene]=p
        result['allow_next_round'] &= p['initial_inputs']['passed']
        if not p['initial_inputs']['passed']:
            continue
        p['state_difference'] = state_comparison(off,on)
        if stage!='guards':
            p['whole_speedup'] = pair['off']['whole']['solver_ms']/pair['on']['whole']['solver_ms']
            p['contact_window_speedup'] = pair['off']['contact_window']['solver_ms']/pair['on']['contact_window']['solver_ms']
        if scene=='mixed':
            p['material_differences'] = []
            for a,b in zip(pair['off']['material_by_frame'],pair['on']['material_by_frame']):
                p['material_differences'].append({'frame':a['frame'],'on_minus_off':{k:b[k]-a[k] if a[k] is not None and b[k] is not None else None
                    for k in ('max_stretch','p99_stretch','fixed_drift_m','fem_min_J','fem_nonpositive','fem_negative_volume','abd_min_J')}})
            p['interpretation']='Pending mixed quality; neither tolerance expansion nor a pass of the old guard. No hard failure observed permits only the declared cloth screen.'
            p['timing_interpretation']='Validation is enabled only in the on guard; guard timings are not speedups and are excluded from screen ratios.'
    return result

def aggregate(session):
    reports=[read(session/(s+'_analysis.json')) for s in ('r1','r2','r3')]
    require(all(r['allow_next_round'] for r in reports), 'An analyzed round failed')
    output={'schema':'flat_contact_pool_screen_summary.v1','scenes':{},'performance_certified':False,
            'quality_certified':False,'eligible_for_cost_review':False,'automatic_full_run':False}
    for scene in ('hang','fixed'):
        pairs=[r['pairs'][scene] for r in reports]
        whole=[p['whole_speedup'] for p in pairs]; contact=[p['contact_window_speedup'] for p in pairs]
        output['scenes'][scene]={'whole_speedups':whole,'contact_window_speedups':contact,
             'median_whole_speedup':statistics.median(whole),'median_contact_window_speedup':statistics.median(contact),
             'screen_threshold_met':statistics.median(whole)>=1.05 and statistics.median(contact)>=1.05}
    output['eligible_for_cost_review']=all(x['screen_threshold_met'] for x in output['scenes'].values())
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--stage',choices=('guards','r1','r2','r3'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=analyze(a.session,a.stage,read(a.manifest))
    with a.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'allow_next_round':report['allow_next_round'],'output':str(a.output)}))
