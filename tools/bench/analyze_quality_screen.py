"""Audit all four native policies against original CSR and a shared CPU reference."""
import json
from pathlib import Path
import statistics
from m3_numeric import ROOT,FrozenSystem,need,rel,residual_metrics
import numpy as np
from cgstats_numeric import statistical_gram,coefficients
from qrfree_numeric import CoordinateBalanced,scaled_coefficients
from cpu_fixed_reference import decimal_residual
from local_identity import record,verify,tree
from linux_runner import write_new
from analyze_native_balanced import csv_new

RAW=ROOT/'runs/innovation_p5_quality_screen_20261016'
REPORT=ROOT/'reports/innovation_p5_quality_screen_20261016'
CASES=(('legacy_smoke','Bunny',1),('smoke','Bunny',1),
       ('coarse_rejection','Bunny',14),('after_apply_failure','Bunny',14),
       ('bunny','Bunny',14),('mixed','Mixed',15))
ORDER=((0,1,2,3),(3,2,1,0),(2,3,0,1))

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def select(rows,pair,augmented,screened):
    return next(r for r in rows if r['pair']==pair and r['requested_balanced']==augmented and r['screened']==screened)

def schedule(row,study):
    result=row['result'];trace=result['trace'];screened=row['screened']
    need(result['screened']==screened and result['residual_restarts']==0,'Policy changed/restarted')
    need(result['converged'] and len(trace)==result['iterations'] and trace[-1]['quality_checked'],
         'Target recurrence incomplete')
    active=False;last=0;screens=0;checks=0;skips=0;forced=0
    limit=min(study['max_iter']-1,2000);zero=result.get('zero_rho_true_quality_checked',False)
    for i,point in enumerate(trace,1):
        need(point['iteration']==i and point['p_Ap']>0 and point['pre_update_rho']>0 and point['alpha']>0
             and np.isfinite([point['p_Ap'],point['pre_update_rho'],point['alpha']]).all(),
             'Invalid recurrence sequence/scalars')
        active=active or point['pre_update_rho']<=1e-4*result['rho_initial']
        due=active;force=False
        if active and screened:
            value=point['recursive_relative_l2']
            need(value is not None and np.isfinite(value) and value>=0,'Missing recursive screen')
            screens+=1;force=last==0 or i==limit or i-last>=8
            due=force or value<=.012
        else:need(point['recursive_relative_l2'] is None,'Unexpected extra recursive screen')
        override=zero and i==len(trace)
        need(point['quality_checked']==(due or override),'Full check schedule differs from protocol')
        need(point['screened_out']==(active and screened and not due and not override),'Wrong skip flag')
        need(point['forced_quality_check']==(force or override),'Wrong forced check flag')
        if point['quality_checked']:
            need(np.isfinite([point['true_relative_l2'],point['common_native_MAS_relative']]).all(),
                 'Checked quality is not finite')
            checks+=1;last=i;forced+=int(point['forced_quality_check'])
        else:
            need(point['true_relative_l2'] is None and point['common_native_MAS_relative'] is None,
                 'Unchecked quality must remain unset')
        skips+=int(point['screened_out'])
    need(screens==result['screen_checks'] and checks<=result['quality_checks']<=checks+int(zero),
         'Native screen/check count disagrees with trace')
    need(result['last_quality_iteration']==last and result['screen_host_ms']>=0 and result['quality_host_ms']>=0,
         'Check metadata invalid')
    need(result['preconditioner_calls']==result['iterations']+int(zero),'Recurrence MAS call count differs')
    need(screened or (screens==0 and result['screen_host_ms']==0),'Dense path acquired extra reduction')
    need(max(result['true_relative_l2'],result['common_native_MAS_relative'])<=.0095,'Final quality loosened')
    return dict(screened_updates= screens,skipped_full_checks=skips,forced_full_checks=forced,
                zero_rho_override=zero,schedule_passed=True)

def main():
    need(all(read(RAW/(s+'_ledger.json'))['complete'] for s in ('small','faults','representative')),'Stages incomplete')
    out=RAW/'analysis';need(not out.exists(),'Fresh quality analysis required');out.mkdir()
    tables=REPORT/'tables';need(not tables.exists(),'Fresh quality tables required');tables.mkdir()
    inputs=tree(RAW);sources=[];targets=[];timings=[];curves=[];coarse=[];comparisons=[];gates=[]
    for label,scene,frame in CASES:
        prefix=RAW/label/'fixed'/f'f{frame}_n1';src=FrozenSystem(prefix)
        ss=read(str(prefix)+'_native_balanced_study.json')
        tp=RAW/label/'fixed'/f'f{frame}_n2';system=FrozenSystem(tp)
        ts=read(str(tp)+'_native_balanced_study.json')
        injected=label=='after_apply_failure'
        need(ss['passed'] and np.array_equal(src.active,system.active),'Source/target layout invalid')
        need(ts['passed'] or (injected and ts['error']=='Native balanced injected failure after apply'),
             'Unexpected target failure')
        need(ts['quality_frequency_study']==(label!='legacy_smoke'),'Wrong quality selector')
        capture=next(r for r in ss['runs'] if r['capture']);steps=capture['iterations'];width=capture['history_width']
        full_n=len(src.full_b);ids=np.fromfile(capture['indices_file'],dtype='<i4')
        history=np.fromfile(capture['history_file'],dtype='<f8').reshape(full_n,width,order='F')
        scalar=np.fromfile(capture['scalars_file'],dtype='<f8').reshape(steps,3)
        final=np.fromfile(capture['solution_file'],dtype='<f8')
        need(np.array_equal(ids,np.r_[np.arange(min(16,steps)),np.arange(max(16,steps-112),steps)])
             and np.count_nonzero(history[:,0])==0,'History selection/initial state changed')
        need(np.all(scalar[:-1,1]>1e-4*scalar[0,1]) and scalar[-1,1]<=1e-4*scalar[0,1],'Source stop changed')
        H,_=statistical_gram(scalar[:,0],scalar[:,1],ids,curvature=scalar[:,2])
        S=final[:,None]-history;G=S.T@(src.full_A@S);G=.5*(G+G.T)
        source_info=dict(case=label,scene=scene,frame=frame,steps=steps,width=width,
            Gram_relative=rel(H,G),source_space_built=ss['source_space_built'])
        Zfull=None
        if ss['source_space_built']:
            Zfull=np.fromfile(ss['source_Z_file'],dtype='<f8').reshape(full_n,8,order='F')
            u,info=coefficients(H,8);need(info['actual_rank']==8,'Source reference rank mismatch')
            expected=S@scaled_coefficients(u,info['selected_eigenvalues'])
            for j in range(8):
                if expected[:,j]@Zfull[:,j]<0:expected[:,j]*=-1
            source_info['Z_relative']=rel(Zfull,expected)
            need(source_info['Z_relative']<1e-7,'GPU source projection differs from reference coefficients')
        else:need(width<8 or coefficients(H,8)[1]['actual_rank']<8,'Unexpected source rejection')
        if ts['coarse_ready']:
            def matrix(name,shape):return np.fromfile(ts[f'target_{name}_file'],dtype='<f8').reshape(shape,order='F')
            Z=matrix('Z',(len(system.full_b),8));W=matrix('W',Z.shape)
            E=matrix('E',(8,8));L=matrix('L',(8,8))
            need(np.array_equal(Z,Zfull),'Target basis differs from source basis')
            werr=rel(W,system.full_A@Z);eerr=rel(E,Z.T@W);lerr=rel(L@L.T,.5*(E+E.T))
            need(max(werr,eerr,lerr)<1e-9,'Coarse matrix/factor original-CSR audit failed')
            need(ts['coarse_restored_bitwise'] and len(ts['coarse_warm_calibration'])==3,'Warm restoration incomplete')
            for warm in ts['coarse_warm_calibration']:
                need(0<=max(warm['W_relative'],warm['E_relative'])<=1e-10 and warm['host_ms']>0,
                     'Warm repeat changed coarse operator')
                coarse.append(dict(case=label,scene=scene,kind='warm',**warm))
            coarse.append(dict(case=label,scene=scene,kind='first',repeat=0,host_ms=ts['target_coarse_host_ms'],
                W_relative=werr,E_relative=eerr,factor_relative=lerr,restored_bitwise=ts['coarse_restored_bitwise']))
            if not injected:
                action=CoordinateBalanced(system.A,system.apply,Z[system.active])
                native_base=np.fromfile(ts['MAS_b_action_file'],dtype='<f8')[system.active]
                native_balanced=np.fromfile(ts['balanced_b_action_file'],dtype='<f8')[system.active]
                source_info['native_MAS_b_vs_fixed_reference_relative']=rel(native_base,system.apply(system.b))
                source_info['native_balanced_b_vs_fixed_reference_relative']=rel(native_balanced,action(system.b))
                source_info['action_calibration_passed']=source_info['native_balanced_b_vs_fixed_reference_relative']<=1e-3
        sources.append(source_info)
        for pair in (1,2,3):
            rows=sorted((r for r in ts['runs'] if r['pair']==pair),key=lambda r:r['order'])
            variants=[2*int(r['screened'])+int(r['requested_balanced']) for r in rows]
            expected=list(ORDER[pair-1]) if ts['quality_frequency_study'] else [(pair-1+i)%2 for i in range(2)]
            if injected:expected=[0] if pair==1 else []
            need(variants==expected,'Predeclared order changed')
        if injected:
            failed=ts['failed_target_candidate']
            need(failed['augmented'] and failed['preconditioner_calls']==1 and not failed['trace'],
                 'Injected exception did not follow the first actual balanced action')
        for row in ts['runs']:
            result=row['result'];policy=schedule(row,ts)
            full_x=np.fromfile(row['solution_file'],dtype='<f8')
            need(full_x.shape==system.full_b.shape and np.isfinite(full_x).all(),'Invalid target solution')
            metrics,_=residual_metrics(system.A,system.b,full_x[system.active],system.apply)
            full_res=float(np.linalg.norm(system.full_b-system.full_A@full_x)/np.linalg.norm(system.full_b))
            need(abs(full_res-result['true_relative_l2'])<1e-8,'Original-CSR residual differs from native result')
            detail=dict(case=label,scene=scene,pair=row['pair'],requested_balanced=row['requested_balanced'],
                used_balanced=row['used_balanced'],screened=row['screened'],fallback=row['fallback'],
                fallback_reason=row['fallback_reason'],iterations=result['iterations'],**metrics,**policy,
                native_true_l2=result['true_relative_l2'],native_MAS_relative=result['common_native_MAS_relative'],
                common_CPU_quality_passed=max(metrics.values())<=.01,injected_failure_case=injected)
            if label in ('bunny','mixed') and row['pair']==1:
                _,dec=decimal_residual(system.full_A,system.full_b,full_x)
                need(abs(dec['relative_l2']-full_res)<1e-9,'Decimal50 residual mismatch')
                detail['Decimal50_relative_l2']=dec['relative_l2']
            targets.append(detail)
            for point in result['trace']:
                curves.append(dict(case=label,scene=scene,pair=row['pair'],requested_balanced=row['requested_balanced'],
                    screened=row['screened'],used_balanced=row['used_balanced'],**point))
            timings.append(dict(case=label,scene=scene,pair=row['pair'],order=row['order'],
                requested_balanced=row['requested_balanced'],used_balanced=row['used_balanced'],
                screened=row['screened'],fallback=row['fallback'],iterations=result['iterations'],
                solve_host_ms=row['solve_host_ms'],solve_event_ms=row['solve_event_ms'],reset_host_ms=row['reset_host_ms'],
                quality_host_ms=result['quality_host_ms'],quality_checks=result['quality_checks'],
                screen_host_ms=result['screen_host_ms'],screen_checks=result['screen_checks'],
                remaining_host_ms=row['solve_host_ms']-result['quality_host_ms']-result['screen_host_ms'],
                recurrence_preconditioner_calls=result['preconditioner_calls'],
                successful_MAS_calls=result['preconditioner_calls']+result['quality_checks'],
                successful_spmv=result['iterations']+result['quality_checks'],
                solution_export_host_ms=row['solution_export_host_ms'],
                failed_attempt=bool(row['failed_attempt'])))
        if label not in ('bunny','mixed'):continue
        rows=[r for r in targets if r['case']==label];times=[r for r in timings if r['case']==label]
        for pair in (1,2,3):
            for augmented in (False,True):
                dense,screen=select(rows,pair,augmented,False),select(rows,pair,augmented,True)
                td,tscreen=select(times,pair,augmented,False),select(times,pair,augmented,True)
                comparisons.append(dict(scene=scene,pair=pair,requested_balanced=augmented,
                    dense_iterations=dense['iterations'],screened_iterations=screen['iterations'],
                    iteration_ratio=screen['iterations']/dense['iterations'],
                    dense_quality_checks=td['quality_checks'],screened_quality_checks=tscreen['quality_checks'],
                    screen_checks=tscreen['screen_checks'],dense_solve_ms=td['solve_host_ms'],screened_solve_ms=tscreen['solve_host_ms'],
                    solve_delta_ms=tscreen['solve_host_ms']-td['solve_host_ms'],
                    quality_delta_ms=tscreen['quality_host_ms']-td['quality_host_ms'],
                    extra_screen_ms=tscreen['screen_host_ms'],
                    remaining_delta_ms=tscreen['remaining_host_ms']-td['remaining_host_ms'],
                    dense_used_balanced=dense['used_balanced'],screened_used_balanced=screen['used_balanced']))
            d=select(rows,pair,True,False);s=select(rows,pair,True,True)
            gates.append(dict(scene=scene,pair=pair,primary=pair==1,
                all_four_CPU_quality_passed=all(r['common_CPU_quality_passed'] for r in rows if r['pair']==pair),
                screened_balanced_used=s['used_balanced'],dense_balanced_used=d['used_balanced'],
                iteration_budget_passed=s['iterations']<=1.10*d['iterations'],
                action_calibration_passed=source_info.get('action_calibration_passed',False)))
    for name,rows in (('source_and_action_audit',sources),('target_solutions',targets),('native_timings',timings),
                      ('native_curves',curves),('coarse_first_warm',coarse),('screen_vs_dense',comparisons),('gates',gates)):
        csv_new(tables/(name+'.csv'),rows)
    verify(inputs)
    primary=[g for g in gates if g['primary']]
    finite=all(r['common_CPU_quality_passed'] for r in targets) and all(
        g['screened_balanced_used'] and g['dense_balanced_used'] and g['iteration_budget_passed']
        and g['action_calibration_passed'] for g in primary)
    means=[]
    for scene in ('Bunny','Mixed'):
        for augmented in (False,True):
            rows=[r for r in comparisons if r['scene']==scene and r['requested_balanced']==augmented]
            keys=('solve_delta_ms','quality_delta_ms','extra_screen_ms','remaining_delta_ms')
            row=dict(scene=scene,requested_balanced=augmented,**{k:statistics.mean(r[k] for r in rows) for k in keys})
            need(abs(row['solve_delta_ms']-row['quality_delta_ms']-row['extra_screen_ms']-row['remaining_delta_ms'])<1e-9,
                 'Additive cost accounting failed')
            means.append(row)
    write_new(out/'summary.json',dict(implementation_checks_passed=True,source=record(Path(__file__)),inputs=inputs,
        sources=sources,targets=targets,timings=timings,coarse=coarse,comparisons=comparisons,gates=gates,
        additive_pair_means=means,finite_gate_passed=finite,
        production_solver_modified=False,performance_certified=False))
    print(json.dumps(dict(implementation_checks_passed=True,finite_gate_passed=finite,gates=gates,
        additive_pair_means=means)),flush=True)

if __name__=='__main__':main()
