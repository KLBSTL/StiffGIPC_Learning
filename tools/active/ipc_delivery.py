"""Evidence attribution and bounded follow-up plans for the IPC revision."""
import argparse
import csv
import itertools
import json
import re
import collections
import subprocess
from pathlib import Path
import numpy as np
from config import ROOT, read, sha
from ipc_benchmark import TAG, SCENES, task, write, metrics

def observer_provenance():
    from build_support import file_record, digest, read_text, actual_compile_commands, link_evidence
    build=ROOT/'builds/base-observed';path=ROOT/'manifests/base_observed_20261005.json';old=read(path)
    initial=ROOT/f'manifests/{TAG}_observer_initial_manifest.json'
    if not initial.exists():write(initial,old)
    commands=actual_compile_commands(build,build/'unused_history')
    for line in read_text(build/'build_1.log').splitlines():
        if 'nvcc.exe' not in line.lower() or ' --compile ' not in line:continue
        src=re.findall(r'"([^"\r\n]+\.cu)"',line,re.I)
        if not src:continue
        p=Path(src[-1]).resolve();invocation=re.search(r'("[^"\r\n]*nvcc\.exe".*)',line,re.I)
        commands[str(p).lower()]={'command':invocation.group(1),'evidence_log':str(build/'build_1.log'),'kind':'nvcc'}
    units=[]
    for name,c in sorted(commands.items()):
        p=Path(name)
        if not (p.is_relative_to(ROOT/'sources/stiff_base/StiffGIPC') or
                p.is_relative_to(ROOT/'sources/stiff_base_observed/StiffGIPC')):continue
        if c['kind']=='nvcc':
            token=re.search(r'(?<!\S)-o\s+(?:"([^"]+)"|(\S+))',c['command'])
            if token is None:raise ValueError('Missing NVCC object output')
            obj=Path(token.group(1) or token.group(2))
            objects=[obj if obj.is_absolute() else build/'inherited-base'/obj]
            if not objects[0].exists():raise ValueError('Missing compiled observer object: '+str(objects[0]))
        else:objects=list(build.rglob(p.stem+'.obj'))
        if len(objects)!=1:raise ValueError('Ambiguous observer object: '+str(p))
        units.append({'source':file_record(p,ROOT),'object':file_record(objects[0],ROOT),**c})
    assert any(u['source']['path'].endswith('stiff_base_observed/StiffGIPC/app/gl_main.cu') for u in units)
    links=link_evidence(build,ROOT);assert links and all(l['commands'] and l['inputs'] for l in links)
    records=[r for r in old['files'] if r['path'].startswith(('sources/stiff_base/','sources/stiff_base_observed/'))]
    records.append(file_record(ROOT/'sources/stiff_base_observed/CMakeLists.txt',ROOT))
    old.update(implementation_version='official-base-velocity-observer',files=records,source_digest=digest(records),
        compiler_inputs=units,links=links,previous_manifest_sha256=sha(ROOT/f'manifests/{TAG}_observer_initial_manifest.json'))
    # Generated identity metadata; prior per-run manifests and initial manifest are preserved.
    path.write_text(json.dumps(old,indent=2),encoding='utf-8')
    write(ROOT/f'reports/active/{TAG}_observer_build_provenance.json',{'compiler_units':len(units),
        'manifest_sha256':sha(path),'exe_sha256':sha(build/'Release/gipc.exe'),
        'logs':[file_record(build/f'build_{i}.log',ROOT) for i in (0,1)],'manifest':str(path)})
    print(json.dumps({'observer_compiler_units':len(units),'link_records':len(links)}))

def cost():
    scenes={}
    for key in ('hang','fixed'):
        prefix=ROOT/f'reports/active/ipc_{key}_nsys'
        ranges=list(csv.DictReader(Path(str(prefix)+'_nvtx_gpu_proj_sum.csv').open()))
        kernel=list(csv.DictReader(Path(str(prefix)+'_cuda_gpu_kern_sum.csv').open()))
        d={r['Range'].lstrip(':'):float(r['Total Proj Time (ns)'])/1e6 for r in ranges}
        wall=d['ipc.physical_frame'];pcg=d['pcg.production_loop']
        dot=sum(float(r['Total Time (ns)'])/1e6 for r in kernel if 'PCG_vdv_Reduction' in r['Name'])
        family=['ipc.gradient_hessian_assembly','linear.total_including_diagnostics','linear.matrix_convert',
            'mas.prepare','pcg.production_loop','collision.discrete_bvh_build','collision.discrete_query',
            'collision.swept_bvh_build_or_refit','collision.swept_query','collision.self_ccd',
            'collision.ground_ccd','ipc.line_search','ipc.energy_evaluation']
        scenes[key]={'captured_frames':1,'frame_gpu_projected_ms':wall,
            'scopes':{s:{'projected_ms':d.get(s,0),'frame_fraction':d.get(s,0)/wall} for s in family},
            'all_pcg_dot_kernel_ms':dot,'all_dot_fraction_of_pcg':dot/pcg,
            'discrete_build_plus_query_fraction':(d['collision.discrete_bvh_build']+d['collision.discrete_query'])/wall,
        'discrete_build_absolute_upper_fraction':d['collision.discrete_bvh_build']/wall,
            'evidence':[str(Path(str(prefix)+s).relative_to(ROOT)) for s in
                ['_nvtx_gpu_proj_sum.csv','_cuda_gpu_kern_sum.csv','_cuda_api_sum.csv']]}
    report={'schema':1,'scenes':scenes,'diagnostic_only':True,
        'method':'One Nsight frame per scene; no operator probe. Inclusive scopes overlap and are not summed. GPU-projected intervals are diagnostic, not formal wall-clock shares.',
        'candidate_decisions':{
            'mas_rho_fusion':{'status':'not_started','reason':'Even all PCG dot kernels together are below the 10% PCG threshold; the removable final r.z subset is smaller.'},
            'static_dynamic_hessian':{'status':'not_started','reason':'Entire matrix conversion is below 10% of frame; removable symbolic work is a subset.'},
            'ordinary_bvh_refit':{'status':'not_started','reason':'Node trace gives build upper bounds 5.53%/3.34%; selected full-range event traces give 2.88%/2.16% including tracing overhead. Refit must retain bounds updates and query. No supported 5% whole-scene prediction.'}}}
    write(ROOT/f'reports/active/{TAG}_cost_decisions.json',report)
    print(json.dumps(report,indent=2))

def selected_cost():
    # Remove diagnostic descendants from the physical frame and linear scopes.
    out={}
    for key in [*SCENES,'mixed']:
        run=ROOT/f'runs/active/{TAG}_{key}_graph_cost_v2';r=[json.loads(l) for l in (run/'cost.jsonl').read_text().splitlines()]
        m={v['scope_id']:v for v in r};d={}
        for v in r:
            if v['sample_kind']=='production' and v['gpu_interval_ms'] is not None:
                d[v['stage']]=d.get(v['stage'],0)+v['gpu_interval_ms']
        probe=sum(v['gpu_interval_ms'] or 0 for v in r if v['sample_kind']!='production' and
            m.get(v['parent_scope_id'],{}).get('sample_kind')=='production')
        frame=d['ipc.physical_frame']-probe;linear=d['linear.total_including_diagnostics']-probe
        pcg=d['pcg.entry_including_diagnostics']-probe
        out[key]={'frames':sorted({v['frame'] for v in r}),'probe_ms_removed':probe,
            'frame_excluding_probe_ms':frame,'linear_excluding_probe_ms':linear,'pcg_entry_excluding_probe_ms':pcg,
            'inclusive_stages_ms':d,'scope':'Inclusive stages overlap. CUDA intervals include trace and submission gaps; subtraction removes probe work, not trace overhead.'}
    write(ROOT/f'reports/active/{TAG}_selected_cost_corrected.json',out)
    print(json.dumps({k:{s:v[s] for s in ['frame_excluding_probe_ms','linear_excluding_probe_ms','probe_ms_removed']} for k,v in out.items()}))

def state_distance(a,b):
    meta=read(a/'trace/metadata.json');nv=meta['vertex_count'] if 'vertex_count' in meta else np.fromfile(a/'trace/state_0000.bin',dtype='<f8').size//3
    abd=meta['abd_point_num'];raw=np.fromfile(a/'trace/topology.bin',dtype='<u4');nf,nt=map(int,raw[1:3]);tets=raw[3+3*nf:].reshape(-1,4)
    fem=np.zeros(nv,bool);fem[tets.ravel()]=True;fem[:abd]=False
    groups={'cloth':~fem & (np.arange(nv)>=abd),'FEM':fem,'ABD':np.arange(nv)<abd}
    result={}
    for kind,pattern in [('position','state_*.bin'),('velocity','velocity_*.bin')]:
        files=sorted((a/'trace').glob(pattern));acc={k:[] for k,m in groups.items() if m.any()}
        for p in files:
            other=b/'trace'/p.name
            if not other.exists():raise ValueError('Missing paired state: '+str(other))
            delta=(np.fromfile(p,dtype='<f8')-np.fromfile(other,dtype='<f8')).reshape(-1,3)
            for k in acc:
                v=delta[groups[k]];acc[k].append((float(np.sqrt(np.mean(np.sum(v*v,axis=1)))),float(np.linalg.norm(v,axis=1).max())))
        result[kind]={k:{'max_frame_rms':max(v[0] for v in rows),'max_vertex':max(v[1] for v in rows),'frames':len(rows)} for k,rows in acc.items()}
    return result

def quality(stage):
    batch=read(ROOT/f'reports/active/{TAG}_{stage}_batch.json');protocol=read(ROOT/f'reports/active/{TAG}_quality_protocol.json')
    records=[]
    for t in batch['runs']:
        if t['result']['status']!='completed':continue
        r=metrics(ROOT/'runs/active'/t['name']);k=t['scene_key']
        if k not in protocol['scenes']:continue
        bounds=protocol['scenes'][k]['bounds']
        records.append({'name':r['name'],'variant':t['variant'],'scene_key':k,'repeat':t['repeat'],
            'material_passed':r['finite'] and not r['pcg_failures'] and all(r[x]<=y for x,y in bounds.items()),
            'material_values':{x:r[x] for x in bounds},'bounds':bounds,
            'pcg_failures':r['pcg_failures'],'finite':r['finite'],'frames':len(r['frames'])})
    write(ROOT/f'reports/active/{TAG}_{stage}_material_checks.json',{'protocol_sha256':sha(ROOT/f'reports/active/{TAG}_quality_protocol.json'),
        'records':records,'independent_ccd_pending':True,'quality_certified':False})
    print(json.dumps(records))

def final_plan():
    # Predeclared finite local performance matrix, with all heavy diagnostics off.
    tasks=[]
    for repeat in range(1,4):
        arms=['base','reference','combined_host','combined']
        arms=arms[repeat-1:]+arms[:repeat-1]
        for key in SCENES:
            for v in arms:
                if v=='combined_host':
                    t=task(key,v,f'final_r{repeat}',execution='host',refit=True,batch=True,reuse=True,trace_velocity=False)
                else:t=task(key,v,f'final_r{repeat}',trace_velocity=False)
                tasks.append(t)
    write(ROOT/f'configs/active/{TAG}_final.json',{'report':f'reports/active/{TAG}_final_batch.json','stop_on_failure':True,'runs':tasks})
    audits=[]
    for key in SCENES:
        for v in ['observed','combined']:
            audits.append(task(key,v,'accepted_audit',diagnostics=['substeps']))
    audits.append(task('mixed','combined','compatibility',steps=100))
    write(ROOT/f'configs/active/{TAG}_audit.json',{'report':f'reports/active/{TAG}_audit_batch.json','stop_on_failure':True,'runs':audits})
    write(ROOT/f'configs/active/{TAG}_long.json',{'report':f'reports/active/{TAG}_long_batch.json','stop_on_failure':True,
        'runs':[task(k,'combined','long',steps=300,trace_velocity=True) for k in [*SCENES,'mixed']]})
    print(json.dumps({'timing_runs':len(tasks),'audit_runs':len(audits),'long_runs':3}))

def compensation_plan():
    # Two finite from-zero windows, using the natural gate-only veto at frame 30
    # and the independently preselected high line-search-cost window 57--59.
    tasks=[]
    for window,steps in [('28-30',30),('57-59',59)]:
        for v in ['gated','compensated']:
            tasks.append(task('fixed_bunny',v,'window_'+window,steps=steps,
                ipc_termination=v,ipc_residual_shadow=True))
    write(ROOT/f'configs/active/{TAG}_compensation.json',{'report':f'reports/active/{TAG}_compensation_batch.json',
        'stop_on_failure':True,'window_selection':'Natural veto in gated frame30; control is preselected line-search window57-59. No parameter grid.', 'runs':tasks})

def verify_audit():
    output=ROOT/f'reports/active/{TAG}_accepted_ccd.json'
    if output.exists():raise FileExistsError(output)
    matrix=read(ROOT/f'reports/active/{TAG}_audit_batch.json');exe=ROOT/'builds/validator/Release/diagnose_first_path.exe'
    report={'validator_sha256':sha(exe),'runs':[],
        'scope':'Independent CPU CCD on separate diagnostic trajectories only; production timing runs are not audited by inheritance.'}
    for row in matrix['runs']:
        if row['repeat']!='accepted_audit':continue
        run=ROOT/'runs/active'/row['name'];trace=run/'trace';res=read(run/'result.json')
        rec={'run':row['name'],'scene_key':row['scene_key'],'variant':row['variant'],'coverage_passed':False};report['runs'].append(rec)
        if res['status']!='completed' or res['recorded_frames']!=100:
            rec['failure']='Incomplete simulation';continue
        groups=collections.defaultdict(list)
        for p in sorted((trace/'substeps').glob('safe_*.bin')):groups[int(p.stem.split('_')[1])].append(p)
        assert sorted(groups)==list(range(100));segments=0
        for f,paths in sorted(groups.items()):
            assert [int(p.stem.split('_')[2]) for p in paths]==list(range(len(paths)))
            assert paths[0].read_bytes()==(trace/f'state_{f:04d}.bin').read_bytes()
            assert paths[-1].read_bytes()==(trace/f'state_{f+1:04d}.bin').read_bytes()
            segments+=len(paths)-1
        rec.update(coverage_passed=True,accepted_segments=segments,stationary_bridges=99)
        target=ROOT/f'reports/active/{TAG}_CCD_{row["scene_key"]}_{row["variant"]}.json'
        if target.exists():raise FileExistsError(target)
        command=[str(exe),str(trace),str(target),'substeps','--stable-nh1'];rec['command']=command
        try:
            with target.with_suffix('.log').open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            rec['exit_code']=r.returncode;rec['ccd']=read(target);assert rec['ccd']['paths_checked']==segments+99
        except subprocess.TimeoutExpired:rec['failure']='Predeclared 300-second CPU audit timeout'
        output.write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(rec),flush=True)

def trajectories():
    result={}
    for key in SCENES:
        bases=[ROOT/f'runs/active/{TAG}_{key}_observed_r{i}' for i in range(1,4)]
        repeated=[state_distance(a,b) for a,b in itertools.combinations(bases,2)]
        candidate=ROOT/f'runs/active/{TAG}_{key}_combined_accepted_audit'
        comparisons=[state_distance(a,candidate) for a in bases]
        result[key]={'baseline_pairwise':repeated,'candidate_vs_baseline':comparisons,
            'meaning':'Real exported velocity. Trajectory divergence is not a physical truth error or a deformation budget.'}
    write(ROOT/f'reports/active/{TAG}_trajectory_comparison.json',result)
    print(json.dumps(result))

def performance():
    batch=read(ROOT/f'reports/active/{TAG}_final_batch.json');scenes={};joint=[]
    for key in SCENES:
        rows=[r for r in batch['runs'] if r['scene_key']==key]
        times={(r['variant'],r['repeat']):r['result']['solver_seconds'] for r in rows}
        summary={}
        for v in ['base','reference','combined_host','combined']:
            t=[times[v,f'final_r{i}'] for i in range(1,4)]
            against_base=[times['base',f'final_r{i}']/times[v,f'final_r{i}'] for i in range(1,4)]
            against_graph=[times['reference',f'final_r{i}']/times[v,f'final_r{i}'] for i in range(1,4)]
            summary[v]={'median_seconds':float(np.median(t)),'seconds':t,
                'paired_base_speedup':float(np.median(against_base)),
                'paired_frozen_graph_speedup':float(np.median(against_graph)),
                'paired_frozen_graph_ratios':against_graph}
        ratios=summary['combined']['paired_frozen_graph_ratios'];joint.append(ratios)
        summary['graph_in_new_combination']={'paired_speedup':float(np.median([times['combined_host',f'final_r{i}']/times['combined',f'final_r{i}'] for i in range(1,4)]))}
        scenes[key]=summary
    joint=np.array(joint);point=float(np.exp(np.mean(np.log(np.median(joint,axis=1)))))
    rng=np.random.default_rng(20261005);indices=rng.integers(0,3,(50000,3))
    bootstrap=np.exp(np.mean(np.log(np.median(joint[:,indices],axis=2)),axis=0));lower=float(np.quantile(bootstrap,.05))
    report={'scenes':scenes,'combined_vs_frozen_graph_geometric_mean':point,'diagnostic_one_sided_95_bootstrap_lower':lower,
        'seed':20261005,'bootstrap_samples':50000,'paired_repeats':3,
        'numerical_stage_target_met':point>=1.10 and float(np.median(joint,axis=1).min())>=1/1.03 and lower>1,
        'performance_certified':False,'reason':'Shared RTX3070 Laptop desktop; only 3 local pairs. AutoDL 7 pairs and exclusive-load quality acceptance deferred by user.'}
    write(ROOT/f'reports/active/{TAG}_performance.json',report);print(json.dumps(report))

def residual_evidence():
    batch=read(ROOT/f'reports/active/{TAG}_residual_batch.json');report=[]
    for t in batch['runs']:
        folder=ROOT/'runs/active'/t['name'];frames=read(folder/'output/stats.json')['frames'];rows=[];movement_reject=0
        for i,f in enumerate(frames,1):
            observations=[n['ipc_residual'] for n in f['newton'] if 'ipc_residual' in n]
            if f.get('newton_exit')=='movement' and observations:
                last=observations[-1]
                movement_reject+=int(last['active'] and last['relative'] is not None and last['relative']>.03)
            for o in observations:
                rows.append({'frame':i,**o})
        identity=[]
        for o in rows:
            if o['active'] and o['valid']:
                weight=t['config'].get('ipc_cumulative_tol',.01)/.03
                identity.append(abs(o['budget']-(o['beta']+weight*(o['unit_budget']-o['beta']))))
        terminal=[o for o in rows if o['kind']=='terminal_diagnostic_assembly']
        report.append({'run':t['name'],'variant':t['variant'],'scene_key':t['scene_key'],
            'observations':len(rows),'invalid_budgets':sum(not r['valid'] for r in rows),
            'budget_identity_max_abs':max(identity,default=0),'observe_host_ms':sum(r['observe_host_ms'] for r in rows),
            'objective_epochs':sorted({r['objective_epoch'] for r in rows}),
            'movement_exit_above_residual_tol':movement_reject,
            'history_vetoes':[r for r in rows if r['history_veto']], 'terminal':terminal,
            'terminal_assembly_host_ms':sum(f.get('ipc_terminal_assembly_host_ms',0) for f in frames)})
    comp=read(ROOT/f'reports/active/{TAG}_compensation_batch.json');windows=[]
    for t in comp['runs']:
        run=ROOT/'runs/active'/t['name'];frames=read(run/'output/stats.json')['frames'];lo,hi=map(int,t['repeat'].removeprefix('window_').split('-'))
        selected=frames[lo-1:hi];pcg=[n['pcg'] for f in selected for n in f['newton'] if 'pcg' in n]
        r=metrics(run)
        windows.append({'run':t['name'],'variant':t['variant'],'window':[lo,hi],
            'directions':len(pcg),'pcg':sum(p['iterations'] for p in pcg),
            'window_max_stretch':max(f['max_stretch'] for f in r['frames'][lo-1:hi]),
            'window_p99_stretch':max(f['p99_stretch'] for f in r['frames'][lo-1:hi]),
            'history_vetoes':sum(n.get('ipc_residual',{}).get('history_veto',False) for f in selected for n in f['newton']),
            'from_zero_solver_seconds':r['seconds']})
    write(ROOT/f'reports/active/{TAG}_residual_evidence.json',{'runs':report,'compensation_windows':windows,
        'decision':'Gates remain experimental. Current gate increases work and fixed-bunny material failures persist. Two compensation windows cannot prove independent repeatable quality benefit.',
        'movement_exit_is_residual_certificate':False})
    print(json.dumps({'windows':windows,'invalid_budgets':sum(r['invalid_budgets'] for r in report)}))

def register():
    # Apply only this revision's evidence-backed dispositions; keep old entries.
    index_path=ROOT/'reports/active/EXPERIMENT_INDEX.json';decision_path=ROOT/'reports/active/EXPERIMENT_DECISIONS.json'
    index=read(index_path);decisions=read(decision_path);updates={}
    for stage in ['components','residual','final','audit']:
        data=read(ROOT/f'reports/active/{TAG}_{stage}_material_checks.json')
        for r in data['records']:
            if not r['material_passed']:
                updates[r['name']]={'status':'rejected' if r['variant'] not in ('base','observed','reference') else 'pending',
                    'status_scope':'Frozen whole-window material protocol only',
                    'reason':'Material bound exceeded. Baseline failures make protocol uncertifiable; no post-hoc widening.',
                    'performance_certified':False,'quality_certified':False,
                    'decision_evidence':f'reports/active/{TAG}_{stage}_material_checks.json'}
    for stage in ['residual','compensation']:
        for t in read(ROOT/f'reports/active/{TAG}_{stage}_batch.json')['runs']:
            if t['variant'] in ('gated','gate001','compensated'):
                updates.setdefault(t['name'],{'status':'rejected','status_scope':'Promotion to fastest/default combination',
                    'reason':'No repeatable quality-cost advantage demonstrated within the predeclared finite experiment.',
                    'performance_certified':False,'quality_certified':False,
                    'decision_evidence':f'reports/active/{TAG}_residual_evidence.json'})
    decisions.update(updates);decision_path.write_text(json.dumps(decisions,indent=2))
    for row in index['entries']:
        if row['id'] in updates:
            row.update(updates[row['id']]);proof=decision_path.relative_to(ROOT).as_posix()
            if proof not in row['evidence']:row['evidence'].append(proof)
    aggregate={'id':TAG+'_local_conclusion','origin':'curated_aggregate','status':'rejected',
        'status_scope':'Stage target / default promotion only; completed implementation preserved',
        'reason':'Geometric mean 1.099423 <1.10; fixed-bunny baseline and candidate material protocol failures. 300frames and AutoDL deferred.',
        'performance_certified':False,'quality_certified':False,
        'evidence':['reports/active/IPC_EXECUTION_RESULTS_20261005.md',f'reports/active/{TAG}_performance.json',f'reports/active/{TAG}_final_verification.json']}
    if aggregate['id'] not in {r['id'] for r in index['entries']}:index['entries'].append(aggregate)
    index_path.write_text(json.dumps(index,indent=2));print(json.dumps({'current_decisions':len(updates),'index_entries':len(index['entries'])}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['cost','quality','prepare-final','observer-provenance','prepare-compensation','audit','trajectories','performance','selected-cost','residual-evidence','register']);p.add_argument('--stage',default='components');a=p.parse_args()
    if a.action=='cost':cost()
    elif a.action=='quality':quality(a.stage)
    elif a.action=='observer-provenance':observer_provenance()
    elif a.action=='prepare-compensation':compensation_plan()
    elif a.action=='audit':verify_audit()
    elif a.action=='trajectories':trajectories()
    elif a.action=='performance':performance()
    elif a.action=='selected-cost':selected_cost()
    elif a.action=='residual-evidence':residual_evidence()
    elif a.action=='register':register()
    else:final_plan()
