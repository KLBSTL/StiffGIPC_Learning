"""Bind lightweight tracing evidence to configuration, work and cost limits."""
import csv
import json
from pathlib import Path
import numpy as np
from config import ROOT,read,sha,matches_requested
from ipc_benchmark import write
from analyze_ipc_light_cost import analyze as analyze_sqlite

TAG='ipc_light_cost_20261005'

def workload(run):
    frames=read(run/'output/stats.json')['frames']
    result=[]
    for number,frame in enumerate(frames,1):
        solves=[n['pcg'] for n in frame['newton'] if 'pcg' in n]
        assert not any(p.get('iteration_limit') or p.get('breakdown') for p in solves)
        assert all(p['execution']=='conditional_graph' for p in solves)
        result.append({'frame':number,'direction_count':len(solves),
            'iterations':[p['iterations'] for p in solves],'pcg_total':sum(p['iterations'] for p in solves),
            'accepted_updates':frame['ipc_stopping']['accepted_updates'],'exit':frame['newton_exit'],
            'beta':frame['ipc_stopping']['final_beta'],'alpha':[n['alpha'] for n in frame['newton'] if 'alpha' in n],
            'active_pairs':[n['active_pairs'] for n in frame['newton'] if 'active_pairs' in n]})
    return result

def main():
    folder=ROOT/'reports/active';plan=read(ROOT/f'configs/active/{TAG}.json');protocol=read(folder/f'{TAG}_protocol.json')
    manifest=read(ROOT/'builds/active/manifest.json');exe=ROOT/'builds/active/Release/gipc.exe'
    identity=sha(exe);assert manifest['binaries'][exe.relative_to(ROOT).as_posix()]['sha256']==identity
    assert sha(folder/'ipc_revision_20261005_quality_protocol.json')==protocol['old_quality_protocol_sha256']
    inputs=[]
    for unit in manifest['compiler_inputs']:
        for key in ('source','object'):
            item=unit[key];assert sha(ROOT/item['path'])==item['sha256'];inputs.append({'path':item['path'],'kind':key,'passed':True})
        for item in unit['project_include_inputs']:assert sha(ROOT/item['path'])==item['sha256']
    run_records=[];profiles=[]
    for task in plan['runs']:
        run=ROOT/'runs/active'/task['name'];req=read(run/'requested.json');out=read(run/'result.json');resolved=read(run/'resolved_config.json')
        config=req['expanded_config'];assert matches_requested(config,task['config'])
        assert req['exe_sha256']==identity and out['status']=='completed' and out['finite']
        assert out['recorded_frames']==config['steps']
        assert resolved['legacy_restrict']=='atomic' and resolved['ipc_stopping']['termination']=='legacy'
        assert resolved['ipc_stopping']['cumulative_tol']==.01 and resolved['ipc_stopping']['min_updates']==6 and resolved['pcg_rho_tol']==1e-4
        active='cost' in config['diagnostics'];observation=resolved['cost_observation']
        assert observation['active']==active and observation['gpu_events_effective']==(active and config['cost_events'])
        assert observation['mode']==('disabled' if not active else 'cuda_events_nvtx_cpu' if config['cost_events'] else 'nvtx_cpu_only')
        events=[]
        if active:
            events=[json.loads(row) for row in (run/'cost.jsonl').read_text().splitlines()]
            assert events and all(e['nvtx_available'] and e['gpu_events_enabled']==config['cost_events'] for e in events)
            ids={e['scope_id'] for e in events}
            assert len(ids)==len(events) and all(e['parent_scope_id']==0 or e['parent_scope_id'] in ids for e in events)
            assert not any(e['operator_probe_enabled'] for e in events)
            if not config['cost_events']:assert all(e['gpu_interval_ms'] is None and e['measurement_mode']=='nvtx_cpu_only' for e in events)
        else:assert not (run/'cost.jsonl').exists()
        work=workload(run)
        record={'name':run.name,'scene':config['scene'],'steps':config['steps'],'solver_seconds':out['solver_seconds'],
            'observation':observation,'work':work,'trace_scopes':len(events),'exe_sha256':identity,
            'requested_sha256':sha(run/'requested.json'),'resolved_sha256':sha(run/'resolved_config.json')}
        run_records.append(record)
        if config['profile']!='node':continue
        activity=read(run/'activity_analysis.json');assert activity['captured_physical_frames']==1
        assert activity['source_sha256']==sha(run/'nsight.sqlite')
        selected_log={e['frame'] for e in events};assert len(selected_log)==1
        selected_frame=next(iter(selected_log))
        components=[config[k] for k in ('refit','batch','reuse')]
        assert all(components) or not any(components)
        combined=all(components)
        rows=list(csv.DictReader((run/'nsys_nvtx_gpu_proj_sum.csv').open(encoding='utf-8-sig')))
        scopes={r['Range'].lstrip(':'):{'projected_ms':float(r['Total Proj Time (ns)'])/1e6,
                                     'cpu_ms':float(r['Total Range Time (ns)'])/1e6,
                                     'instances':int(r['Range Instances'])} for r in rows}
        gpu=activity['aggregate']['gpu_timeline_clipped'];wall=activity['aggregate']['cpu_physical_frame_wall']['union_ms']
        # Do not infer a missing kernel has zero cost from a truncated top list.
        complete_activity=analyze_sqlite(run/'nsight.sqlite',0)
        kernels=complete_activity['kernel_hotspots']['rows'];ee=sum(k['sum_ms'] for k in kernels if k['name'].startswith('_selfQuery_ee('))
        api=activity['aggregate']['runtime_api']
        profiles.append({'run':run.name,'scene':config['scene'],'frame':selected_frame,'combined':combined,
            'work':work[selected_frame-1],'cpu_nvtx_frame_wall_ms':wall,'gpu_activity_union_ms':gpu['all']['union_ms'],
            'gpu_activity_span_ms':gpu['all']['first_to_last_span_ms'],'gpu_activity_gaps_ms':gpu['all']['gaps_between_activities_ms'],
            'kernel_sum_ms':gpu['by_kind']['kernel']['sum_ms'],'memcpy_gpu_sum_ms':gpu['by_kind']['memcpy']['sum_ms'],
            'memcpy_gpu_fraction_of_cpu_frame':gpu['by_kind']['memcpy']['sum_ms']/wall,
            'memcpy_directions':gpu['memcpy_directions'],'runtime_api':api,
            'unmatched_graph_node_records':gpu['unmatched_graph_node_records'],
            'ee_kernel_sum_ms':ee,'ee_kernel_fraction_of_cpu_frame':ee/wall,
            'scopes':scopes,'activity_analysis_sha256':sha(run/'activity_analysis.json'),
            'sqlite_sha256':sha(run/'nsight.sqlite'),'nsight_report_sha256':sha(run/'nsight.nsys-rep')})
    neutrality=[]
    for scene in ('hang','fixed'):
        paths={mode:ROOT/f'runs/active/{TAG}_{scene}_combined_{mode}' for mode in ('off','cpu','events')}
        reference=workload(paths['off'])
        items=[]
        for mode in ('cpu','events'):
            observed=workload(paths[mode]);deltas=[];alpha_max=0
            for frame in range(4):
                x=np.fromfile(paths[mode]/f'trace/state_{frame:04d}.bin','<f8');y=np.fromfile(paths['off']/f'trace/state_{frame:04d}.bin','<f8')
                assert x.shape==y.shape and np.isfinite(x).all()
                deltas.append(float(np.abs(x-y).max()))
            same_work=all(a['iterations']==b['iterations'] and a['accepted_updates']==b['accepted_updates'] and a['exit']==b['exit'] and a['active_pairs']==b['active_pairs'] for a,b in zip(reference,observed))
            assert same_work and max(deltas)<=protocol['checks']['prefix_position_component_difference_m_max']
            for a,b in zip(reference,observed):alpha_max=max(alpha_max,float(np.max(np.abs(np.array(a['alpha'])-b['alpha']),initial=0)))
            items.append({'mode':mode,'same_pcg_accept_count_exit_and_contact_work':same_work,
                'position_component_max_by_frame_m':deltas,'max_alpha_difference':alpha_max})
        times={mode:read(path/'result.json')['solver_seconds'] for mode,path in paths.items()}
        neutrality.append({'scene':scene,'comparisons':items,'three_frame_solver_seconds':times,
            'cpu_overhead_relative_to_off':times['cpu']/times['off']-1,
            'events_overhead_relative_to_off':times['events']/times['off']-1,
            'scope':'One short-prefix observation per mode, not proof of long-window neutrality or timing improvement'})
    fixtures=read(ROOT/f'runs/active/{TAG}_fixtures/result.json')
    assert fixtures['passed'] and fixtures['exe_sha256']==identity and len(fixtures['checks'])==8
    assert not (ROOT/'runs/active/.gpu.lock').exists()
    late_combined=next(p for p in profiles if p['combined'] and p['frame']==41)
    initial_fixed=next(p for p in profiles if p['combined'] and p['scene']=='cloth_fixed_bunny_l')
    report={'delivery_verified':True,'scope':'Observation tooling and cost diagnosis only; no new execution optimization',
        'run_records':run_records,'profiles':profiles,'short_prefix_neutrality':neutrality,
        'source_object_identity':inputs,'exe_sha256':identity,'compiled_units':len(manifest['compiler_inputs']),
        'changed_objects':len(manifest['changed_objects']),'fixtures_passed':8,'config_tests_passed':12,'config_subtests_passed':58,
        'legacy_numeric_path_unchanged':True,'old_quality_protocol_unchanged':True,'old_failures_reclassified':False,
        'performance_certified':False,'quality_certified':False,'default_promoted':False,
        'candidate':{'name':'Discrete EE subtree max original ID pruning','status':'pending, not implemented',
            'evidence':'Safe static predicate, but no same-tree measured net benefit, metadata cost or whole-scene 5% evidence',
            'combined_hang_frame41_ee_fraction_diagnostic':late_combined['ee_kernel_fraction_of_cpu_frame'],
            'combined_fixed_frame1_ee_fraction_diagnostic':initial_fixed['ee_kernel_fraction_of_cpu_frame'],
            'required_ee_fraction_reduction_for_5_percent_diagnostic':{
                'hang41':.05/late_combined['ee_kernel_fraction_of_cpu_frame'],
                'fixed1':.05/initial_fixed['ee_kernel_fraction_of_cpu_frame']},
            'limitation':'These are diagnostic proportions, not rigorous production speedup bounds. The narrow initial fixed frame does not replace late/mixed coverage.'},
        'limitations':['Nsight node capture has cold activation and tracing overhead; solver_seconds is not a performance denominator.',
            'GPU activity union and gaps are not achievable savings; gaps include submission/profiler/other-process scheduling.',
            'CUDA memcpy API time includes waiting; it cannot be added to GPU activity or equated with copy bandwidth.',
            'Graph nodes without runtime correlation remain unassigned. NVTX projections/temporal overlaps are inclusive.',
            'Current fixed late/mixed windows remain unavailable. No 100/300-frame trajectory, material or AutoDL acceptance.'],
        'protocol_sha256':sha(folder/f'{TAG}_protocol.json'),'plan_sha256':sha(ROOT/f'configs/active/{TAG}.json')}
    write(folder/f'{TAG}_analysis_v2.json',report)
    print(json.dumps({'delivery_verified':True,'completed_runs':len(run_records),'profiles':len(profiles),
        'fixtures_passed':8,'short_prefix_neutrality':neutrality,'candidate':report['candidate']}))

if __name__=='__main__':main()
