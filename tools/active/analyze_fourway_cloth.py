"""Analyze the predeclared four-way cloth timing/endpoint comparison."""
import argparse
import itertools
import json
import statistics
from config import ROOT,read,sha,digest,expand
from analyze import analyze as geometry
from validate_run import validate

ARMS=('base','graph','toi','graph_toi')
LABELS={'base':'base','graph':'base+CUDA Graph','toi':'base+TOI','graph_toi':'base+CUDA Graph+TOI'}

def check(task):
    folder=ROOT/'runs/active'/task['name'];q=read(folder/'requested.json');r=read(folder/'result.json')
    assert q['expanded_config']==expand(task['config'])
    assert q['config_sha256']==digest(q['expanded_config'])
    assert task['result']==r
    assert r['status']=='completed' and r['recorded_frames']==100 and r.get('finite')
    assert not r['heavy_diagnostics']
    c=q['expanded_config'];s=read(folder/'output/scene.json')
    for k,target in [('dt','dt'),('ipc_newton_tol','newton_tol'),('pcg_rho_tol','pcg_tol')]:
        assert c[k]==s['effective_run'][target]
    assert not any(c[k] for k in ('refit','batch','reuse'))
    assert c['diagnostics']==[] and c['profile']=='none' and c['trace_velocity']==False
    assert not list((folder/'trace').glob('velocity_*.bin'))
    assert c['mas_restrict']=='serial' and c['full_step_exit_probe'] is None
    assert c['pcg_rho_tol']==1e-4 and c['ipc_newton_tol']==c['toi_remaining_fraction_tol']==.01
    assert c['toi_trial_velocity_tol']==.05
    toi=task['variant'] in ('toi','graph_toi');graph=task['variant'] in ('graph','graph_toi')
    assert c['backend']==('toi_al' if toi else 'ipc')
    assert c['execution']==('conditional_graph' if graph else 'host')
    assert c['mas']==('cholesky' if toi else 'legacy')
    assert c['mas_factor_action']==('factor_inverse' if toi else 'triangular')
    assert c['mu_coordinates']==('world_block' if toi else 'generalized')
    assert q['binary']==task['binary']==('base' if task['variant']=='base' else 'active')
    manifest=read(folder/'build_manifest.json')
    assert sha(folder/'build_manifest.json')==q['manifest_sha256']
    assert manifest['source_digest']==q['source_digest']
    assert q['exe_sha256'] in {v['sha256'] for v in manifest['binaries'].values()}
    frames=read(folder/'output/stats.json')['frames']
    systems=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
    assert systems and len(frames)==100
    assert {p.get('execution','host') for p in systems}=={c['execution']}
    assert not any(p.get('iteration_limit') or p.get('breakdown') for p in systems)
    if task['binary']=='active':
        gate=validate(folder);assert gate['passed']
    else:gate={'passed':True,'resolved_interface':False,'scope':'Frozen manifest/request/effective scene/observed host execution checked'}
    return folder,gate

def analyze():
    path=ROOT/'reports/active/FOURWAY_CLOTH_TIMING_BATCH_20261005.json';batch=read(path)
    assert len(batch['runs'])==24,'Incomplete predeclared matrix; do not invent missing speed ratios'
    result={'performance_certified':False,'quality_certified':False,
        'scope':'RTX 3070 Laptop shared-desktop 100-frame diagnostic, three interleaved repeats',
        'batch_sha256':sha(path),'script_sha256':sha(__file__),'scenes':{}}
    for scene in ('hang','fixed_bunny'):
        tasks=[next(t for t in batch['runs'] if t['scene_key']==scene and t['variant']==arm and t['repeat']=='r'+str(i))
               for arm in ARMS for i in (1,2,3)]
        checked=[check(t) for t in tasks];g=geometry([p for p,_ in checked],3)
        by={r['name']:r for r in g['runs']};obs=[]
        for t,(p,gate) in zip(tasks,checked):
            r=by[t['name']];f=r['frames'];work=sum(x['pcg_iterations'] for x in f);calls=sum(x['pcg_calls'] for x in f)
            obs.append({'name':t['name'],'arm':t['variant'],'repeat':t['repeat'],
                'solver_seconds':r['result']['solver_seconds'],'pcg_iterations':work,'directions':calls,
                'pcg_per_direction':work/calls,'outers':sum(x['outers'] for x in f),
                'cloth_max_stretch':max(x['cloth_max_stretch'] for x in f),
                'fixed_max_drift_m':max(x['fixed_max_drift'] for x in f),
                'min_abd_J':min(x['abd_min_J'] for x in f),'max_fem_nonpositive':max(x['fem_nonpositive'] for x in f),
                'configuration_gate':gate,'exe_sha256':read(p/'requested.json')['exe_sha256'],
                'requested_sha256':sha(p/'requested.json')})
        pairs=[]
        for i in (1,2,3):
            rows={r['arm']:r for r in obs if r['repeat']=='r'+str(i)}
            pairs.append({'repeat':i,'base_over_arm':{a:rows['base']['solver_seconds']/rows[a]['solver_seconds'] for a in ARMS},
                'graph_gain_with_ipc':rows['base']['solver_seconds']/rows['graph']['solver_seconds'],
                'graph_gain_with_toi':rows['toi']['solver_seconds']/rows['graph_toi']['solver_seconds'],
                'toi_gain_with_graph':rows['graph']['solver_seconds']/rows['graph_toi']['solver_seconds']})
        summaries={a:{'label':LABELS[a],
            'seconds':[r['solver_seconds'] for r in obs if r['arm']==a],
            'median_seconds':statistics.median(r['solver_seconds'] for r in obs if r['arm']==a),
            'median_paired_speedup_over_base':statistics.median(p['base_over_arm'][a] for p in pairs),
            'paired_speedups_over_base':[p['base_over_arm'][a] for p in pairs],
            'median_pcg_iterations':statistics.median(r['pcg_iterations'] for r in obs if r['arm']==a),
            'median_directions':statistics.median(r['directions'] for r in obs if r['arm']==a),
            'cloth_max_stretch_range':[min(r['cloth_max_stretch'] for r in obs if r['arm']==a),max(r['cloth_max_stretch'] for r in obs if r['arm']==a)],
            'fixed_max_drift_m':max(r['fixed_max_drift_m'] for r in obs if r['arm']==a)} for a in ARMS}
        own=[p for p in g['comparisons'] if p['a'] in g['baseline_names'] and p['b'] in g['baseline_names']]
        assert len(own)==3
        candidates=[]
        for t in tasks[3:]:
            relevant=[p for p in g['comparisons'] if p['a'] in g['baseline_names'] and p['b']==t['name']]
            assert len(relevant)==3
            outside=[]
            for i in range(100):
                limit=max(p['frames'][i]['position_cloth_rms'] for p in own)
                value=max(p['frames'][i]['position_cloth_rms'] for p in relevant)
                if value>limit:outside.append(i+1)
            endpoint=next(e for e in g['candidate_gates'] if e['run']==t['name'])
            candidates.append({'name':t['name'],'arm':t['variant'],
                'endpoint_envelope_passes':endpoint['endpoint_envelope_passes'],
                'endpoint_violation_count':endpoint['violation_count'],
                'position_outside_base_repeat_frames':outside,
                'max_position_cloth_rms_against_base_m':max(p['maxima']['position_cloth_rms'] for p in relevant),
                'quality_certified':False})
        result['scenes'][scene]={'observations':obs,'summary':summaries,'pairs':pairs,
            'median_graph_gain_with_toi':statistics.median(p['graph_gain_with_toi'] for p in pairs),
            'median_toi_gain_with_graph':statistics.median(p['toi_gain_with_graph'] for p in pairs),
            'base_repeat_max_position_cloth_rms_m':max(p['maxima']['position_cloth_rms'] for p in own),
            'candidate_quality':candidates,'geometry':g}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    data=analyze()
    with (ROOT/a.output).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
    print(json.dumps({key:{'summary':{a:{k:r[k] for k in ('median_seconds','median_paired_speedup_over_base','median_pcg_iterations','median_directions','cloth_max_stretch_range')} for a,r in s['summary'].items()},
        'median_graph_gain_with_toi':s['median_graph_gain_with_toi'],'median_toi_gain_with_graph':s['median_toi_gain_with_graph'],
        'endpoint_pass_counts':{a:sum(q['endpoint_envelope_passes'] for q in s['candidate_quality'] if q['arm']==a) for a in ARMS[1:]}}
        for key,s in data['scenes'].items()}))
