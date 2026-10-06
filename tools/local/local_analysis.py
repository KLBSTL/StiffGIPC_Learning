"""CPU-only interpretation of actual exports using the frozen material protocol."""
import csv
import math
from local_plan import ROOT,tasks,plan
from config import read,expand
from linux_runner import require,sha,verify_files
from pool_metrics import metrics,export_evidence,state_comparison,window,pool_evidence
from quality_analysis import validate_base,input_comparison
from validate_run import validate

PROTOCOL=ROOT/'tools/bench/quality_protocol.json'

def analyze_one(session,t,seal):
    out=session/t['name'];row={k:t[k] for k in ('name','binary','variant','repeat','scene_key')}
    row.update(hard_checks_passed=False,failures=[])
    try:
        req,res=read(out/'requested.json'),read(out/'result.json');c=expand(t['config']);steps=c['steps']
        identity=seal['programs'][t['binary']]
        require(req['binary']==t['binary'] and req['expanded_config']==c,'Requested config/binary differs')
        require(req['exe_sha256']==identity['exe']['sha256'] and req['source_digest']==identity['source_digest'],'Program identity differs')
        require(res['status']=='completed' and res['recorded_frames']==steps,'Incomplete or failed run')
        row['configuration']=validate(out) if t['binary']=='active' else validate_base(out)
        require(row['configuration']['passed'],'Configuration evidence failed')
        row['positions']=export_evidence(out,steps,'state');require(row['positions']['passed'],'Missing/nonfinite actual positions')
        row['velocity']=export_evidence(out,steps,'velocity') if t['binary']=='active' else {'available':False,'reason':'Frozen Stiff has no actual velocity export; no reconstruction.'}
        if t['binary']=='active':require(row['velocity']['passed'],'Missing/nonfinite actual velocity')
        frames=read(out/'output/stats.json')['frames'];require(len(frames)==steps,'Stats length differs')
        require(not any(f.get('newton_exit')=='iteration_limit' for f in frames),'Newton iteration cap reached')
        if t['binary']=='active':
            row['pool']=pool_evidence(frames,c['contact_pool'],False);require(row['pool']['passed'],'Pool execution telemetry failed')
        m=metrics(out);require(m['finite'] and not m['pcg_failures'],'Nonfinite/PCG failure')
        require(all(math.isfinite(n['alpha']) for f in frames for n in f['newton'] if 'alpha' in n),'Nonfinite alpha')
        require(all(v is None or not isinstance(v,(float,int)) or math.isfinite(v) for f in m['frames'] for v in f.values()),'Nonfinite material metric')
        require(not any(f['fem_nonpositive'] or (f['abd_min_J'] is not None and f['abd_min_J']<=0) for f in m['frames']),'Element/ABD inversion')
        key='fixed_bunny' if t['scene_key']=='fixed' else 'hang';bounds=read(PROTOCOL)['scenes'][key]['bounds']
        row['material']={k:m[k] for k in bounds};row['material_by_frame']=m['frames']
        row['original_bounds']={k:{'bound':b,'actual':m[k],'excess':m[k]-b,'within_bound':m[k]<=b,
                'exceeding_frames':[f['frame'] for f in m['frames'] if f[k]>b]} for k,b in bounds.items()}
        row['within_original_bounds']=all(v['within_bound'] for v in row['original_bounds'].values())
        with (out/'trace/frames.csv').open() as stream:timing=list(csv.DictReader(stream))
        require([int(x['frame']) for x in timing]==list(range(1,steps+1)),'Frame timing sequence differs')
        require(all(math.isfinite(float(x['solver_ms'])) and float(x['solver_ms'])>0 for x in timing),'Invalid timing')
        low,high=plan()['contact_windows'][t['scene_key']]
        row['timing_observation']={'prefix':window(frames,timing,1,steps),'contact':window(frames,timing,low,high),
            'scope':'Shared Windows desktop diagnostic; no exclusive-load, speedup or quality certification.'}
        row['hard_checks_passed']=True
    except (ValueError,KeyError,FileNotFoundError,AssertionError,IndexError) as e:row['failures'].append(type(e).__name__+': '+str(e))
    return row

def analyze_round(session,stage,seal,batch):
    ts=tasks(stage);names=[t['name'] for t in ts];recorded=[r['name'] for r in batch['runs']]
    require(batch['stage']==stage and batch['plan']==plan(),'Plan/stage mismatch')
    require(recorded==names[:len(recorded)] and batch['skipped']==names[len(recorded):],'Ledger is not planned prefix')
    rows=[]
    for entry,t in zip(batch['runs'],ts):
        out=session/t['name'];require(sha(out/'evidence.json')==entry['evidence_sha256'],'Run inventory changed')
        verify_files(out,read(out/'evidence.json')['files']);require(read(out/'result.json')==entry['result'],'Result changed')
        require(sha(session/(t['name']+'_check.json'))==entry['check_sha256'],'Per-run analysis changed')
        rows.append(analyze_one(session,t,seal))
    comparisons=[]
    good=[r for r in rows if r['hard_checks_passed']]
    for a,b in zip(good,good[1:]):
        pair={'left':a['name'],'right':b['name'],'initial_inputs':input_comparison(session/a['name'],session/b['name'])}
        if pair['initial_inputs']['passed']:pair['state_difference']=state_comparison(session/a['name'],session/b['name'])
        comparisons.append(pair)
    complete=len(rows)==len(ts) and all(r['hard_checks_passed'] for r in rows) and all(p['initial_inputs']['passed'] for p in comparisons)
    return {'schema':'windows_cloth_round_analysis.v1','stage':stage,'runs':rows,'comparisons':comparisons,
            'diagnosis_complete':complete,'all_original_bounds_satisfied':complete and all(r['within_original_bounds'] for r in rows),
            'quality_protocol_sha256':sha(PROTOCOL),'prior_failed_guard':seal['prior_guard'],
            'old_quality_gate_unchanged':True,'allow_performance_screen':False,'quality_certified':False,
            'performance_certified':False,'automatic_next_round':False,'automatic_long_run':False,
            'limitations':['One on fixed run does not establish repeatability.','Stiff actual velocities and PCG breakdown telemetry are unavailable.',
                           'New observed repeat ranges never replace the frozen quality bounds.']}
