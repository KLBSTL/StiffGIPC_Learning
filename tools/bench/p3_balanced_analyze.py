"""Audit balanced P3 timings and export full per-frame single-pair evidence."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from config import read
from linux_runner import require,write_new
from local_identity import record,verify
from m0_analyze import geometry,compare,csv_write

PHASES=('assembly','pcg','ccd','line_search','state_update')


def summarize_samples(batch,metric):
    rows=batch['samples'];require(len(rows)==32,'Expected 32 balanced timing samples')
    for i,r in enumerate(rows):
        split=((i%4 in (1,2)) != bool((i//4)%2))
        require(r['index']==i and r['block']==i//4 and r['position']==i%4 and
            r['mode']==('W' if split else 'F') and r['counts_equal'],'Invalid balanced schedule/counts')
        require(type(r[metric]) in (int,float) and math.isfinite(r[metric]) and r[metric]>0,'Invalid duration')
    ratios=[];by_first=defaultdict(list)
    for i in range(0,32,2):
        pair={r['mode']:r[metric] for r in rows[i:i+2]};ratio=pair['F']/pair['W']
        ratios.append(ratio);by_first[rows[i]['mode']].append(ratio)
    return dict(paired_geometric_speedup=statistics.geometric_mean(ratios),paired_ratios=ratios,
        first_mode_speedup={k:statistics.geometric_mean(v) for k,v in by_first.items()},
        geometric_ms={k:statistics.geometric_mean(r[metric] for r in rows if r['mode']==k) for k in ('F','W')},
        scope='Within-input descriptive timings; adjacent samples are not independent scene repeats')


def batch_summary(data):
    require(data['schema']=='gipc.p3.ee.balanced.v2','Missing timing v2')
    require(not data['cpu_primary']['gpu_events_injected'] and data['gpu_separate']['gpu_events_injected'],
        'CPU and GPU timing batches mixed')
    for k in ('cpu_primary','gpu_separate'):
        require(data[k]['boundary_typed_checks']==4 and data[k]['all_timed_counts_equal'],'Incomplete correctness boundaries')
    return dict(cpu=summarize_samples(data['cpu_primary'],'complete_cpu_ms'),
        gpu=summarize_samples(data['gpu_separate'],'gpu_interval_ms'))


def analyze(session,output):
    require(not output.exists(),'Fresh analysis required');packet=read(session/'seal.json')
    verify(packet['dependencies']);identity=packet['identity'];verify(identity['sources']+[identity['exe']]+identity['dlls'])
    ledgers={s:read(session/(s+'_ledger.json')) for s in ('smoke','prefix','query','profile','whole')}
    require(all(v['complete'] for v in ledgers.values()),'Incomplete data cannot be presented as complete')
    queries=[];profiles=[];frames_out=[];newton_out=[];runs=[];geometry_out=[];differences=[];telemetry=[]
    input_records=[record(session/'seal.json')]
    for s,ledger in ledgers.items():
        input_records.append(record(session/(s+'_ledger.json')))
        for entry in ledger['runs']:
            t=entry['task'];folder=session/t['name'];result=read(folder/'result.json')
            require(entry['passed'] and result['status']=='completed' and result['cleanup_owned_job_empty'],'Invalid run')
            require(read(folder/'config_validation.json')['passed'],'Configuration mismatch')
            input_records.extend(record(folder/f) for f in ('result.json','output/stats.json','trace/frames.csv','requested.json','resolved_config.json','balanced_validation.json','power_samples.json'))
            stats=read(folder/'output/stats.json')['frames']
            if s=='query':
                study=json.loads((folder/'p3_ee.jsonl').read_text())
                from p3_analyze import inputs
                input_records.extend(inputs(study));input_records.append(record(folder/'p3_ee.jsonl'))
                queries.append(dict(scene=t['scene'],timing=batch_summary(study['batched_timing']),
                    resources=study['resources'],gpu_stage_ms=study['gpu_stage_ms'],workload=study['workload'],
                    boundary_checked=True,per_call_legacy_checks=len(study['pairs'])*2))
            if s=='profile':
                costs=[json.loads(x) for x in (folder/'cost.jsonl').read_text().splitlines()]
                input_records.append(record(folder/'cost.jsonl'))
                scopes=defaultdict(lambda:dict(calls=0,cpu_submit_ms=0.,gpu_interval_ms=0.))
                for r in costs:
                    if r['sample_kind']!='production':continue
                    d=scopes[r['stage']];d['calls']+=1;d['cpu_submit_ms']+=r['cpu_submit_ms'];d['gpu_interval_ms']+=r['gpu_interval_ms'] or 0
                ee=scopes['collision.discrete_ee'];frame=scopes['ipc.physical_frame'];dcd=scopes['collision.discrete_query']
                profiles.append(dict(scene=t['scene'],mode=t['mode'],scopes=dict(scopes),
                    ee_frame_gpu_envelope_fraction=ee['gpu_interval_ms']/frame['gpu_interval_ms'],
                    ee_dcd_gpu_envelope_fraction=ee['gpu_interval_ms']/dcd['gpu_interval_ms'],
                    scope='Instrumented three-frame GPU event envelopes, including host gaps; not formal performance'))
            if s!='whole':continue
            gpu={}
            for key in ('memory.free','temperature.gpu','power.draw','clocks.sm','utilization.gpu'):
                values=[]
                for sample in result['gpu_samples']:
                    try:value=float(sample[key])
                    except (KeyError,TypeError,ValueError):continue
                    if math.isfinite(value):values.append(value)
                gpu[key]=dict(samples=len(values),minimum=min(values),maximum=max(values),median=statistics.median(values)) if values else None
            telemetry.append(dict(run=t['name'],scene=t['scene'],mode=t['mode'],gpu=gpu))
            with (folder/'trace/frames.csv').open(newline='') as f:timings=list(csv.DictReader(f))
            require(len(stats)==len(timings)==100,'Formal run must have exactly 100 complete frames')
            base=dict(run=t['name'],scene=t['scene'],mode=t['mode']);totals=defaultdict(float)
            for i,(f,timing) in enumerate(zip(stats,timings),1):
                require(int(timing['frame'])==i,'Frame identity differs')
                pcg=[n['pcg'] for n in f['newton'] if 'pcg' in n]
                require(not any(p.get('breakdown') or p.get('iteration_limit') for p in pcg),'PCG failure')
                row=dict(base,frame=i,solver_ms=float(timing['solver_ms']),N=len(pcg),P=sum(p['iterations'] for p in pcg),
                    candidate_pairs=int(timing['candidate_pairs']),ground_candidates=int(timing['ground_candidates']),
                    exit=f['newton_exit'],**f['p3_ee'])
                for k in PHASES:
                    row[k+'_ms']=f['recording_phase_ms'][k];row['legacy_'+k+'_ms']=f['phase_ms'][k]
                row.update(accepted_updates=f['ipc_stopping'].get('accepted_updates'),final_beta=f['ipc_stopping'].get('final_beta'),
                    energy_backtracks=sum(n.get('energy_backtracks',0) for n in f['newton']),
                    intersection_backtracks=sum(n.get('intersection_backtracks',0) for n in f['newton']),
                    **f.get('contact_geometry',{}))
                frames_out.append(row)
                for k in ('solver_ms','N','P',*(p+'_ms' for p in PHASES),'energy_backtracks','intersection_backtracks','dispatch_queries','w_queries','complete_fused_fallbacks'):
                    totals[k]+=row[k]
                for ni,n in enumerate(f['newton']):
                    p=n.get('pcg',{});newton_out.append(dict(**base,frame=i,newton_index=ni,
                        pcg_iterations=p.get('iterations'),rho_initial=p.get('rho_initial'),rho_stop=p.get('rho_stop'),
                        rho_tolerance=p.get('rho_tolerance'),graph_capture_host_ms=p.get('graph_capture_instantiate_host_ms'),
                        alpha=n.get('alpha'),beta=n.get('beta'),active_pairs=n.get('active_pairs'),
                        energy_backtracks=n.get('energy_backtracks'),intersection_backtracks=n.get('intersection_backtracks'),
                        **{k+'_ms':n.get('phase_ms',{}).get(k) for k in PHASES}))
            for key in ('N','P','energy_backtracks','intersection_backtracks','dispatch_queries','w_queries','complete_fused_fallbacks'):
                totals[key]=int(totals[key])
            runs.append(dict(**base,frames=100,**totals,raw_peak=max(f['p3_ee']['raw_peak'] for f in stats),
                mean_solver_ms=totals['solver_ms']/100,mean_solves_per_frame=totals['N']/100,
                mean_pcg_iterations_per_solve=totals['P']/totals['N'],wall_seconds=result['wall_seconds']))
            geometry_out.extend(dict(**base,**g) for g in geometry(folder,folder,100))
    pairs=[]
    for scene in ('Boxes','Mixed','Bunny'):
        f,w=[next(r for r in runs if r['scene']==scene and r['mode']==m) for m in ('fused','two_stage')]
        pairs.append(dict(scene=scene,solver_speedup=f['solver_ms']/w['solver_ms'],
            time_reduction_percent=100*(1-w['solver_ms']/f['solver_ms']),
            line_search_speedup=f['line_search_ms']/w['line_search_ms'],
            F=f,W=w,independent_pairs=1,confidence_interval=None))
        a,b=[session/f'whole_{scene.lower()}_{mode}' for mode in ('fused','two_stage')]
        for i in range(1,101):differences.append(dict(scene=scene,frame=i,**compare(a,b,i,i)))
    output.mkdir(parents=True)
    for name,rows in (('runs.csv',runs),('frames.csv',frames_out),('newton.csv',newton_out),('geometry.csv',geometry_out)):
        # Native contact metadata can be absent in a frame; preserve missing values.
        keys=list(dict.fromkeys(k for r in rows for k in r));csv_write(output/name,[{k:r.get(k) for k in keys} for r in rows])
    write_new(output/'state_differences.json',differences);write_new(output/'queries.json',queries);write_new(output/'profiles.json',profiles)
    write_new(output/'gpu_telemetry.json',telemetry)
    summary=dict(schema='p3_balanced_100.v1',formal_runs=6,formal_frames=600,binary=identity['exe'],
        pairs=pairs,queries=queries,profiles=profiles,performance_certified=False,physical_quality_certified=False,
        error_scope='F/W trajectory differences, not errors against a converged solution; native rho is not true linear residual')
    write_new(output/'summary.json',summary);write_new(output/'input_records.json',input_records)
    print(json.dumps(dict(complete=True,frames=600,speedups={p['scene']:p['solver_speedup'] for p in pairs})),flush=True)

