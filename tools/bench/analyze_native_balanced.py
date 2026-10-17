"""Independent original-A audit of actual native balanced PCG; no timing certification."""
import json
from pathlib import Path
from m3_numeric import ROOT,FrozenSystem,need,rel,pcg,residual_metrics,orthogonal
import numpy as np
from cgstats_numeric import statistical_gram,coefficients
from qrfree_numeric import CoordinateBalanced,scaled_coefficients
from cpu_fixed_reference import decimal_residual
from local_identity import record,verify,tree
from linux_runner import write_new

RAW=ROOT/'runs/innovation_p5_native_balanced_20261016'
REPORT=ROOT/'reports/innovation_p5_native_balanced_20261016'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def csv_new(path,rows):
    import csv
    need(rows,'No table rows');names=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)

def main():
    need(all(read(RAW/(s+'_ledger.json'))['complete'] for s in ('small','faults','representative')),'Native stages incomplete')
    out=RAW/'analysis';need(not out.exists(),'Fresh balanced analysis required');out.mkdir()
    report_tables=REPORT/'tables';need(not report_tables.exists(),'Fresh report tables required');report_tables.mkdir()
    inputs=tree(RAW);sources=[];targets=[];timings=[];curves=[];shadow=[];gates=[]
    for label,scene,frame in (('smoke','Bunny',1),('coarse_rejection','Bunny',14),
                              ('bunny','Bunny',14),('mixed','Mixed',15)):
        folder=RAW/label;prefix=folder/'fixed'/f'f{frame}_n1'
        src=FrozenSystem(prefix);ss=read(str(prefix)+'_native_balanced_study.json')
        target_prefix=folder/'fixed'/f'f{frame}_n2';system=FrozenSystem(target_prefix)
        ts=read(str(target_prefix)+'_native_balanced_study.json')
        need(np.array_equal(src.active,system.active),'Changed active coordinates')
        source_row=next(r for r in ss['runs'] if r['capture'])
        steps=source_row['iterations'];width=source_row['history_width'];full_n=len(src.full_b)
        ids=np.fromfile(source_row['indices_file'],dtype='<i4')
        history=np.fromfile(source_row['history_file'],dtype='<f8').reshape(full_n,width,order='F')
        scalar=np.fromfile(source_row['scalars_file'],dtype='<f8').reshape(steps,3)
        final=np.fromfile(source_row['solution_file'],dtype='<f8')
        need(np.array_equal(ids,np.r_[np.arange(min(16,steps)),np.arange(max(16,steps-112),steps)])
             and np.count_nonzero(history[:,0])==0,'Source indices/initial iterate invalid')
        need(np.all(scalar[:-1,1]>1e-4*scalar[0,1]) and scalar[-1,1]<=1e-4*scalar[0,1],
             'Source native stopping changed')
        H,_=statistical_gram(scalar[:,0],scalar[:,1],ids,curvature=scalar[:,2])
        S=final[:,None]-history;G=S.T@(src.full_A@S);G=.5*(G+G.T)
        source_info=dict(case=label,scene=scene,frame=frame,steps=steps,width=width,
            Gram_relative=rel(H,G),source_space_built=ss['source_space_built'])
        Zfull=None
        if ss['source_space_built']:
            Zfull=np.fromfile(ss['source_Z_file'],dtype='<f8').reshape(full_n,8,order='F')
            u,info=coefficients(H,8);need(info['actual_rank']==8,'Source CPU rank mismatch')
            expected=S@scaled_coefficients(u,info['selected_eigenvalues'])
            for j in range(8):
                if expected[:,j]@Zfull[:,j]<0:expected[:,j]*=-1
            source_info['Z_relative']=rel(Zfull,expected)
            need(source_info['Z_relative']<1e-7,'GPU source projection differs from independent CPU coefficients')
        else:
            need(width<8 or coefficients(H,8)[1]['actual_rank']<8,'Unexpected source rank rejection')
        sources.append(source_info)
        candidate=None
        if ts['coarse_ready']:
            Z=np.fromfile(ts['target_Z_file'],dtype='<f8').reshape(len(system.full_b),8,order='F')
            W=np.fromfile(ts['target_W_file'],dtype='<f8').reshape(len(system.full_b),8,order='F')
            E=np.fromfile(ts['target_E_file'],dtype='<f8').reshape(8,8,order='F')
            L=np.fromfile(ts['target_L_file'],dtype='<f8').reshape(8,8,order='F')
            need(np.array_equal(Z,Zfull),'Target basis changed from committed source')
            need(rel(W,system.full_A@Z)<1e-9 and rel(E,Z.T@W)<1e-9
                 and rel(L@L.T,.5*(E+E.T))<1e-9,'Native W/E/factor audit failed')
            candidate=CoordinateBalanced(system.A,system.apply,Z[system.active])
            native_base=np.fromfile(ts['MAS_b_action_file'],dtype='<f8')[system.active]
            native_balanced=np.fromfile(ts['balanced_b_action_file'],dtype='<f8')[system.active]
            source_info['native_MAS_b_vs_fixed_reference_relative']=rel(native_base,system.apply(system.b))
            source_info['native_balanced_b_vs_fixed_reference_relative']=rel(native_balanced,candidate(system.b))
            source_info['action_calibration_passed']=source_info['native_balanced_b_vs_fixed_reference_relative']<=1e-3
        for row in ts['runs']:
            full_x=np.fromfile(row['solution_file'],dtype='<f8');need(full_x.shape==system.full_b.shape,'Target x size')
            need(np.isfinite(full_x).all(),'Target x nonfinite');x=full_x[system.active]
            metrics,_=residual_metrics(system.A,system.b,x,system.apply)
            full_res=np.linalg.norm(system.full_b-system.full_A@full_x)/np.linalg.norm(system.full_b)
            result=row['result'];trace=result['trace']
            need(result['converged'] and len(trace)==result['iterations'] and trace[-1]['quality_checked'],
                 'Incomplete native target recurrence')
            need(abs(full_res-result['true_relative_l2'])<1e-8,'Native final residual differs from full original CSR')
            need(max(result['true_relative_l2'],result['common_native_MAS_relative'])<=.0095,
                 'Native stopping metadata invalid')
            for point in trace:
                need(point['p_Ap']>0 and point['pre_update_rho']>0 and point['alpha']>0
                     and np.isfinite([point['p_Ap'],point['pre_update_rho'],point['alpha']]).all(),
                     'Invalid native recurrence scalar')
                curves.append(dict(case=label,scene=scene,pair=row['pair'],requested_balanced=row['requested_balanced'],
                                   used_balanced=row['used_balanced'],**point))
            detail=dict(case=label,scene=scene,pair=row['pair'],requested_balanced=row['requested_balanced'],
                used_balanced=row['used_balanced'],fallback=row['fallback'],fallback_reason=row['fallback_reason'],
                iterations=result['iterations'],**metrics,common_CPU_quality_passed=max(metrics.values())<=.01)
            if label in ('bunny','mixed') and row['pair']==1:
                _,dec=decimal_residual(system.full_A,system.full_b,full_x)
                need(abs(dec['relative_l2']-full_res)<1e-9,'Native target Decimal50 residual mismatch')
                detail['Decimal50_relative_l2']=dec['relative_l2']
            targets.append(detail)
            timings.append(dict(case=label,scene=scene,pair=row['pair'],order=row['order'],
                requested_balanced=row['requested_balanced'],used_balanced=row['used_balanced'],
                fallback=row['fallback'],iterations=result['iterations'],solve_host_ms=row['solve_host_ms'],
                solve_event_ms=row['solve_event_ms'],reset_host_ms=row['reset_host_ms'],
                quality_host_ms=result['quality_host_ms'],quality_checks=result['quality_checks'],
                recurrence_preconditioner_calls=result['preconditioner_calls'],
                total_success_MAS_calls=result['preconditioner_calls']+result['quality_checks'],
                total_success_spmv=result['iterations']+result['quality_checks'],
                solution_export_host_ms=row['solution_export_host_ms']))
        if label not in ('bunny','mixed'):continue
        for method,action in (('MAS_reference',system.apply),('balanced_reference_same_native_Z',candidate)):
            if action is None:continue
            result,x,_=pcg(system.A,system.b,action,system.apply,tol=.0095)
            result.pop('trace');np.save(out/f'{scene}_{method}.npy',x,allow_pickle=False)
            shadow.append(dict(scene=scene,method=method,**result))
        rows=[r for r in targets if r['case']==label]
        for pair in (1,2,3):
            arm=next(r for r in rows if r['pair']==pair and r['requested_balanced'])
            base_row=next(r for r in rows if r['pair']==pair and not r['requested_balanced'])
            gates.append(dict(scene=scene,pair=pair,primary=pair==1,quality_passed=arm['common_CPU_quality_passed']
                and base_row['common_CPU_quality_passed'],used_balanced=arm['used_balanced'],
                base_iterations=base_row['iterations'],balanced_iterations=arm['iterations'],
                reduced_iterations=arm['iterations']<base_row['iterations'],
                action_calibration_passed=source_info.get('action_calibration_passed',False)))
    for name,rows in (('source_and_action_audit',sources),('target_solutions',targets),('native_timings',timings),
                       ('native_curves',curves),('CPU_shadow',shadow),('gates',gates)):
        csv_new(report_tables/(name+'.csv'),rows)
    verify(inputs)
    primary=[g for g in gates if g['primary']]
    write_new(out/'summary.json',dict(implementation_checks_passed=True,source=record(Path(__file__)),inputs=inputs,
        sources=sources,targets=targets,timings=timings,CPU_shadow=shadow,gates=gates,
        primary_numerical_gate_passed=all(g['quality_passed'] and g['used_balanced'] and g['action_calibration_passed'] for g in primary),
        primary_iteration_reduction_passed=all(g['reduced_iterations'] and g['used_balanced'] for g in primary),
        production_solver_modified=False,performance_certified=False))
    print(json.dumps(dict(implementation_checks_passed=True,gates=gates,source_audits=sources,CPU_shadow=shadow)),flush=True)

if __name__=='__main__':main()
