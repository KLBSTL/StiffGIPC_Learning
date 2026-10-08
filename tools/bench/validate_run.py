"""Check requested/resolved configuration and observed execution, not physics quality."""
import argparse
import json
from pathlib import Path
from config import ROOT, read

def validate(run):
    req=read(run/'requested.json'); c=req['expanded_config']; result=read(run/'result.json')
    checks=[]
    def check(name,actual,expected):
        checks.append({'check':name,'actual':actual,'expected':expected,'passed':actual==expected})
    check('run_completion',result['status'],'completed')
    check('frame_count',result['recorded_frames'],c['steps'])
    if req['binary']!='active':
        return {'passed':False,'scope':'configuration only','reason':'Frozen program has no resolved config contract'}
    r=read(run/'resolved_config.json')
    chunk=c.get('pcg_graph_chunk',1)
    # Old saved requests/results predate the field and represent K=1. New
    # requests must receive explicit native confirmation, even at the default.
    chunk_contract='pcg_graph_chunk' in c or 'configured_pcg_graph_chunk' in r
    check('pcg_graph_chunk.valid',type(chunk) is int and chunk in (1,4),True)
    check('pcg_graph_chunk.execution',chunk!=4 or c['execution']=='conditional_graph',True)
    if chunk_contract:
        actual=r.get('configured_pcg_graph_chunk')
        check('pcg_graph_chunk.resolved_type',type(actual) is int,True)
        check('pcg_graph_chunk.resolved',actual,chunk)
    if 'fixed_graph_chunk_study' in c or 'fixed_graph_chunk_study' in r:
        actual=r.get('fixed_graph_chunk_study')
        check('fixed_graph_chunk_study.resolved_type',type(actual) is bool,True)
        check('fixed_graph_chunk_study.resolved',actual,c.get('fixed_graph_chunk_study',False))
    if 'edge_query_order' in r:
        e=r['edge_query_order']
        check('edge_query_order.requested',e['requested'],c.get('edge_query_order','raw'))
        check('edge_query_order.effective',e['effective'],c.get('edge_query_order','raw'))
        probe=c.get('edge_order_probe_frames')
        check('edge_query_order.probe_frames',e['probe_frames'],[] if probe is None else list(map(int,probe.split(','))))
        check('edge_query_order.probe_file',e['probe_file'],'' if probe is None else str(run/'edge_order_probe.jsonl'))
    elif c.get('edge_query_order','raw')!='raw' or c.get('edge_order_probe_frames') is not None:
        check('edge_query_order_supported',False,True)
    structure_requested='structure_probe' in c.get('diagnostics',[])
    if 'linear_structure_probe' in r:
        probe=r['linear_structure_probe']
        for key in ('requested','effective'):
            check('linear_structure_probe.'+key,probe.get(key),structure_requested)
        check('linear_structure_probe.gpu_events',probe.get('gpu_events'),structure_requested and c.get('cost_events',True))
        check('linear_structure_probe.include_converted_structure',probe.get('include_converted_structure'),True)
        check('linear_structure_probe.numerical_path',probe.get('numerical_path'),'unchanged_full_conversion')
    elif structure_requested:
        check('linear_structure_probe_supported',False,True)
    query_requested='bvh_query_probe' in c.get('diagnostics',[])
    if 'bvh_query_probe' in r:
        probe=r['bvh_query_probe']
        check('bvh_query_probe.enabled',probe.get('enabled'),query_requested)
        check('bvh_query_probe.file',probe.get('file'),str(run/'bvh_query_probe.jsonl') if query_requested else '')
        check('bvh_query_probe.production_outputs_untouched',probe.get('production_outputs_untouched'),True)
        if query_requested:
            from config import environment
            expected=list(map(int,environment(c,run)['GIPC_BVH_QUERY_PROBE_FRAMES'].split(',')))
            check('bvh_query_probe.frames',probe.get('frames'),expected)
    elif query_requested:
        check('bvh_query_probe_supported',False,True)
    if 'ipc_stopping' in r:
        for key,target in [('ipc_termination','termination'),('ipc_cumulative_tol','cumulative_tol'),
            ('ipc_min_updates','min_updates'),('ipc_residual_rel_tol','residual_rel_tol'),
            ('ipc_residual_floor','residual_floor'),('ipc_residual_shadow','residual_shadow'),
            ('ipc_terminal_audit_frame','terminal_audit_frame'),
            ('ipc_residual_cpu_audit','residual_cpu_audit')]:
            if key in c:check(key,r['ipc_stopping'][target],c[key])
    for source,target in [('backend','contact_backend'),('dt','dt'),('ipc_newton_tol','ipc_newton_tol'),
                          ('pcg_rho_tol','pcg_rho_tol'),('execution','configured_pcg_execution')]:
        check(source,r[target],c[source])
    check('mas.cholesky',r['mas']['cholesky'],c['mas']=='cholesky')
    check('mas.inverse64',r['mas']['inverse64'],False)
    check('mas.wide_apply',r['mas']['wide_apply'],c['mas']=='cholesky')
    if 'restriction_mode' in r['mas']:
        check('mas.restriction_mode',r['mas']['restriction_mode'],c.get('mas_restrict','serial') if c['mas']=='cholesky' else 'inactive')
    if 'factor_action' in r['mas']:
        check('mas.factor_action',r['mas']['factor_action'],c.get('mas_factor_action','triangular') if c['mas']=='cholesky' else 'inactive')
    elif c.get('mas_factor_action','triangular')!='triangular':
        check('mas.factor_action_supported',False,True)
    for source,target in [('refit','GIPC_CCD_BVH_REFIT'),('batch','GIPC_BATCHED_ENERGY'),('reuse','GIPC_ENERGY_REUSE')]:
        check(source,r['acceleration_features'][target],c[source])
    if 'report_components' in r:
        for source,target in [('mas_fused_dot','mas_fused_dot_requested'),('fixed_mas_dot_study','fixed_mas_dot_study'),
            ('discrete_bvh_refit','discrete_bvh_refit'),('discrete_bvh_rebuild_interval','discrete_bvh_rebuild_interval'),
            ('discrete_bvh_validate','discrete_bvh_validate')]:
            if source in c:check(source,r['report_components'][target],c[source])
        for source,target in [('spmv_fused_quadratic','spmv_fused_quadratic_requested'),
                              ('fixed_spmv_quadratic_study','fixed_spmv_quadratic_study')]:
            if target in r['report_components'] and source in c:check(source,r['report_components'][target],c[source])
            elif c.get(source,False):check(target+'_supported',False,True)
        for key in ('bvh_eligibility','bvh_eligibility_validate','bounded_ccd','bounded_ccd_validate','contact_pool','contact_pool_validate'):
            if key in r['report_components'] and key in c:
                check(key,r['report_components'][key],c[key])
            elif c.get(key,False):check(key+'_supported',False,True)
    elif any(c.get(key,False) for key in ('mas_fused_dot','fixed_mas_dot_study','discrete_bvh_refit','discrete_bvh_validate')):
        check('report_components_supported',False,True)
    if c['backend']=='toi_al':
        for source,target in [('toi_remaining_fraction_tol','remaining_fraction_tol'),
                              ('toi_trial_velocity_tol','trial_velocity_tol_m_s'),
                              ('choose_start','choose_start'),('restart_guard','restart_full_step_guard'),('mu_mode','mu_mode')]:
            check(source,r['toi'][target],c[source])
        for target in ('contacts_retained','lambda_retained','gamma_retained'):
            check(target,r['toi'][target],True)
        if 'mu_coordinates' in r['toi']:
            check('mu_coordinates',r['toi']['mu_coordinates'],c.get('mu_coordinates','generalized'))
        elif c.get('mu_coordinates','generalized')!='generalized':
            check('mu_coordinates_supported',False,True)
        if 'full_step_exit_probe' in r['toi']:
            target=c.get('full_step_exit_probe')
            expected=None
            if target is not None:
                f,o=map(int,target.split(':'));expected={'frame':f,'outer':o,'max_inner':8}
            check('full_step_exit_probe',r['toi']['full_step_exit_probe'],expected)
        elif c.get('full_step_exit_probe') is not None:
            check('full_step_exit_probe_supported',False,True)
    records=[n['pcg'] for f in read(run/'output/stats.json')['frames'] for n in f['newton'] if 'pcg' in n]
    check('observed_execution_modes',sorted({p.get('execution','host') for p in records}),[c['execution']])
    # Host PCG has no graph body or chunk observation. Its K1-only contract
    # is checked through the requested/resolved configuration above.
    graph_records=[p for p in records if p.get('execution','host')=='conditional_graph']
    if ((chunk_contract and c['execution']=='conditional_graph')
        or (graph_records and (chunk_contract or any('graph_chunk_iterations' in p for p in graph_records)))):
        check('pcg_graph_chunk.observed',bool(graph_records) and all(
            type(p.get('graph_chunk_iterations')) is int and p['graph_chunk_iterations']==chunk
            for p in graph_records),True)
    check('observed_fused_diag_disabled',any(p.get('fused_diag_update',False) for p in records),False)
    check('iteration_limit_or_breakdown',any(p.get('iteration_limit',False) or p.get('breakdown',False) for p in records),False)
    if 'report_components' in r:
        check('mas_fused_dot_requested_per_pcg',all(p.get('mas_fused_dot_requested')==c['mas_fused_dot'] for p in records),True)
        if c['mas_fused_dot']:
            check('mas_fused_dot_effective_per_pcg',bool(records) and all(p.get('mas_fused_dot_effective') for p in records),True)
        if 'spmv_fused_quadratic_requested' in r['report_components']:
            check('spmv_fused_quadratic_requested_per_pcg',all(p.get('spmv_fused_quadratic_requested')==c['spmv_fused_quadratic'] for p in records),True)
            if c['spmv_fused_quadratic']:
                check('spmv_fused_quadratic_effective_per_pcg',bool(records) and all(p.get('spmv_fused_quadratic_effective') for p in records),True)
        if 'bounded_ccd' in r['report_components']:
            frame_rows=read(run/'output/stats.json')['frames']
            check('bounded_ccd_requested_per_frame',all(f.get('bounded_ccd',{}).get('requested')==c['bounded_ccd'] for f in frame_rows),True)
            check('bounded_ccd_validation_failed',sum(f.get('bounded_ccd',{}).get('validation_failed',0) for f in frame_rows),0)
    if c.get('full_step_exit_probe') is not None:
        f,o=map(int,c['full_step_exit_probe'].split(':'))
        selected=[n for frame in read(run/'output/stats.json')['frames'] for n in frame['newton'] if n.get('full_step_exit_probe_active')]
        check('exit_probe.observed_target',sorted({(n['frame'],n['outer']) for n in selected}),[(f,o)])
        check('exit_probe.within_budget',0<len(selected)<=8,True)
        check('exit_probe.velocity_converged',selected[-1].get('inner_exit_reason') if selected else None,'velocity_converged')
    probes=[p['cost_operator_probe'] for p in records if 'cost_operator_probe' in p]
    for target in ('preserved_main_solution','preserved_rhs','preserved_operator_inputs','mas_persistent_buffers_audited'):
        if probes: check('probe.'+target,all(p[target] for p in probes),True)
    return {'passed':all(x['passed'] for x in checks),'scope':'resolved configuration and execution only; no physical/performance certification',
            'run':str(run),'pcg_records':len(records),'fixed_operator_probes':len(probes),'checks':checks}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    rows=[validate(ROOT/n) for n in a.runs];result={'passed':all(r['passed'] for r in rows),'runs':rows}
    with (ROOT/a.output).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'passed':result['passed'],'failed_checks':[(r['run'],c) for r in rows for c in r.get('checks',[]) if not c['passed']]}))
    raise SystemExit(int(not result['passed']))
