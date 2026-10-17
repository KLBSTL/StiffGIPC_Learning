"""Original-CSR and CPU-reference audit of local projection fusion."""
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
from analyze_quality_screen import schedule

RAW=ROOT/'runs/innovation_p5_fused_apply_20261016'
REPORT=ROOT/'reports/innovation_p5_fused_apply_20261016'
CASES=(('legacy_smoke','Bunny',1),('smoke','Bunny',1),('coarse_rejection','Bunny',14),
       ('after_fused_failure','Bunny',14),('bunny','Bunny',14),('mixed','Mixed',15))
ORDER=((0,1,2),(2,1,0),(1,2,0))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def variant(row):return 2 if row['requested_fused'] else int(row['requested_balanced'])
def choose(rows,pair,kind):return next(r for r in rows if r['pair']==pair and r['variant']==kind)

def main():
    need(all(read(RAW/(s+'_ledger.json'))['complete'] for s in ('small','faults','representative')),'Stages incomplete')
    out=RAW/'analysis';need(not out.exists(),'Fresh analysis required');out.mkdir()
    tables=REPORT/'tables';need(not tables.exists(),'Fresh tables required');tables.mkdir()
    inputs=tree(RAW);sources=[];targets=[];timings=[];curves=[];batches=[];pairs=[];gates=[];coarse=[]
    for label,scene,frame in CASES:
        prefix=RAW/label/'fixed'/f'f{frame}_n1';src=FrozenSystem(prefix);ss=read(str(prefix)+'_native_balanced_study.json')
        tp=RAW/label/'fixed'/f'f{frame}_n2';system=FrozenSystem(tp);ts=read(str(tp)+'_native_balanced_study.json')
        injected=label=='after_fused_failure';enabled=label!='legacy_smoke'
        need(ss['passed'] and np.array_equal(src.active,system.active),'Source layout/capture failed')
        need(ts['passed'] or (injected and ts['error']=='Native balanced injected failure after fused apply'),'Unexpected failure')
        need(ts['fused_apply_study']==enabled and ts['quality_frequency_study']==enabled,'Selector mismatch')
        capture=next(r for r in ss['runs'] if r['capture']);steps=capture['iterations'];width=capture['history_width'];n=len(src.full_b)
        ids=np.fromfile(capture['indices_file'],dtype='<i4')
        history=np.fromfile(capture['history_file'],dtype='<f8').reshape(n,width,order='F')
        scalar=np.fromfile(capture['scalars_file'],dtype='<f8').reshape(steps,3)
        final=np.fromfile(capture['solution_file'],dtype='<f8')
        need(np.array_equal(ids,np.r_[np.arange(min(16,steps)),np.arange(max(16,steps-112),steps)])
             and np.count_nonzero(history[:,0])==0,'History selection/initial state changed')
        need(np.all(scalar[:-1,1]>1e-4*scalar[0,1]) and scalar[-1,1]<=1e-4*scalar[0,1],'Source stopping changed')
        H,_=statistical_gram(scalar[:,0],scalar[:,1],ids,curvature=scalar[:,2])
        S=final[:,None]-history;G=S.T@(src.full_A@S);G=.5*(G+G.T)
        source=dict(case=label,scene=scene,steps=steps,width=width,Gram_relative=rel(H,G),source_space_built=ss['source_space_built'])
        Zfull=None
        if ss['source_space_built']:
            Zfull=np.fromfile(ss['source_Z_file'],dtype='<f8').reshape(n,8,order='F')
            u,info=coefficients(H,8);need(info['actual_rank']==8,'Independent source rank differs')
            expected=S@scaled_coefficients(u,info['selected_eigenvalues'])
            for j in range(8):
                if expected[:,j]@Zfull[:,j]<0:expected[:,j]*=-1
            source['Z_relative']=rel(Zfull,expected);need(source['Z_relative']<1e-7,'Source coefficients/projection differ')
        else:need(width<8 or coefficients(H,8)[1]['actual_rank']<8,'Unexpected source rejection')
        if ts['coarse_ready']:
            def matrix(key,shape):return np.fromfile(ts[f'target_{key}_file'],dtype='<f8').reshape(shape,order='F')
            Z=matrix('Z',(len(system.full_b),8));W=matrix('W',Z.shape);E=matrix('E',(8,8));L=matrix('L',(8,8))
            need(np.array_equal(Z,Zfull),'Transferred Z differs')
            need(max(rel(W,system.full_A@Z),rel(E,Z.T@W),rel(L@L.T,.5*(E+E.T)))<1e-9,'Coarse CSR/factor audit failed')
            need(ts['coarse_restored_bitwise'] and len(ts['coarse_warm_calibration'])==3,'Warm state not restored')
            coarse.append(dict(case=label,scene=scene,kind='first',repeat=0,host_ms=ts['target_coarse_host_ms']))
            for row in ts['coarse_warm_calibration']:
                need(max(row['W_relative'],row['E_relative'])<=1e-10,'Warm coarse changed')
                coarse.append(dict(case=label,scene=scene,kind='warm',**row))
            if not injected:
                action=CoordinateBalanced(system.A,system.apply,Z[system.active])
                base=np.fromfile(ts['MAS_b_action_file'],dtype='<f8')[system.active]
                reference_full=np.fromfile(ts['balanced_b_action_file'],dtype='<f8')
                fused_full=np.fromfile(ts['fused_b_action_file'],dtype='<f8')
                reference=reference_full[system.active];fused=fused_full[system.active];expected=action(system.b)
                source.update(native_MAS_b_CPU_relative=rel(base,system.apply(system.b)),
                    reference_b_CPU_relative=rel(reference,expected),fused_b_CPU_relative=rel(fused,expected),
                    fused_b_reference_relative=rel(fused_full,reference_full))
                source['action_calibration_passed']=max(source[k] for k in
                    ('native_MAS_b_CPU_relative','reference_b_CPU_relative','fused_b_CPU_relative','fused_b_reference_relative'))<=1e-3
                need(ts['apply_batch_inputs_unchanged'] and len(ts['apply_batches'])==12,'Batch state/count failed')
                for pair in range(1,7):
                    rows=sorted((r for r in ts['apply_batches'] if r['pair']==pair),key=lambda r:r['order'])
                    need([r['fused'] for r in rows]==[bool((pair-1+i)%2) for i in range(2)],'Batch AB/BA order changed')
                    for row in rows:
                        value=np.fromfile(row['output_file'],dtype='<f8')
                        need(value.shape==reference_full.shape and np.isfinite(value).all(),'Invalid batch output')
                        delta=rel(value,reference_full);cpu=rel(value[system.active],expected)
                        need(row['calls']==16 and abs(delta-row['action_relative'])<1e-8 and delta<=1e-3 and cpu<=1e-3,
                             'Batch action independent audit failed')
                        need(min(row['host_ms'],row['event_ms'])>0,'Invalid batch timing')
                        batches.append(dict(case=label,scene=scene,**row,CPU_action_relative=cpu,
                            per_apply_host_ms=row['host_ms']/16,per_apply_event_ms=row['event_ms']/16))
        sources.append(source)
        for pair in (1,2,3):
            rows=sorted((r for r in ts['runs'] if r['pair']==pair),key=lambda r:r['order'])
            expected=list(ORDER[pair-1]) if enabled else [(pair-1+i)%2 for i in range(2)]
            if injected:expected=[0,1] if pair==1 else []
            need([variant(r) for r in rows]==expected,'Three-arm order changed')
        if injected:
            fail=ts['failed_target_candidate']
            need(fail['fused_apply'] and fail['augmented'] and fail['preconditioner_calls']==1 and not fail['trace'],
                 'Injected failure did not follow actual fused action')
        for row in ts['runs']:
            result=row['result'];policy=schedule(row,ts);kind=variant(row)
            need(row['used_fused']==(row['used_balanced'] and row['requested_fused'])
                and result['fused_apply']==row['used_fused'],'Requested/actual fused mismatch')
            need(row['screened']==enabled,'Quality policy differs between arms')
            x=np.fromfile(row['solution_file'],dtype='<f8')
            need(x.shape==system.full_b.shape and np.isfinite(x).all(),'Target solution shape/finiteness')
            metrics,_=residual_metrics(system.A,system.b,x[system.active],system.apply)
            full=float(np.linalg.norm(system.full_b-system.full_A@x)/np.linalg.norm(system.full_b))
            need(abs(full-result['true_relative_l2'])<1e-8,'Native residual differs from full original CSR')
            detail=dict(case=label,scene=scene,pair=row['pair'],variant=kind,used_balanced=row['used_balanced'],
                used_fused=row['used_fused'],fallback=row['fallback'],iterations=result['iterations'],**metrics,**policy,
                common_CPU_quality_passed=max(metrics.values())<=.01)
            if label in ('bunny','mixed') and row['pair']==1:
                _,decimal=decimal_residual(system.full_A,system.full_b,x)
                need(abs(decimal['relative_l2']-full)<1e-9,'Decimal50 residual differs')
                detail['Decimal50_relative_l2']=decimal['relative_l2']
            targets.append(detail)
            for point in result['trace']:curves.append(dict(case=label,scene=scene,pair=row['pair'],variant=kind,**point))
            timings.append(dict(case=label,scene=scene,pair=row['pair'],order=row['order'],variant=kind,
                used_balanced=row['used_balanced'],used_fused=row['used_fused'],fallback=row['fallback'],iterations=result['iterations'],
                solve_host_ms=row['solve_host_ms'],solve_event_ms=row['solve_event_ms'],reset_host_ms=row['reset_host_ms'],
                quality_host_ms=result['quality_host_ms'],quality_checks=result['quality_checks'],
                screen_host_ms=result['screen_host_ms'],screen_checks=result['screen_checks'],
                remaining_host_ms=row['solve_host_ms']-result['quality_host_ms']-result['screen_host_ms'],
                recurrence_MAS_calls=result['preconditioner_calls'],failed_attempt=bool(row['failed_attempt']),
                solution_export_host_ms=row['solution_export_host_ms']))
        if label not in ('bunny','mixed'):continue
        rows=[r for r in targets if r['case']==label];times=[r for r in timings if r['case']==label]
        for pair in (1,2,3):
            ref,fused=choose(rows,pair,1),choose(rows,pair,2);tr,tf=choose(times,pair,1),choose(times,pair,2)
            pairs.append(dict(scene=scene,scope='PCG',pair=pair,reference_ms=tr['solve_host_ms'],fused_ms=tf['solve_host_ms'],
                ratio=tf['solve_host_ms']/tr['solve_host_ms'],delta_ms=tf['solve_host_ms']-tr['solve_host_ms'],
                quality_delta_ms=tf['quality_host_ms']-tr['quality_host_ms'],screen_delta_ms=tf['screen_host_ms']-tr['screen_host_ms'],
                remaining_delta_ms=tf['remaining_host_ms']-tr['remaining_host_ms']))
            gates.append(dict(scene=scene,pair=pair,primary=pair==1,used_fused=fused['used_fused'],
                used_reference=ref['used_balanced'] and not ref['used_fused'],
                iteration_budget_passed=fused['iterations']<=1.10*ref['iterations'],
                action_calibration_passed=source['action_calibration_passed']))
        bs=[r for r in batches if r['case']==label]
        for pair in range(1,7):
            ref=next(r for r in bs if r['pair']==pair and not r['fused']);fused=next(r for r in bs if r['pair']==pair and r['fused'])
            pairs.append(dict(scene=scene,scope='apply16',pair=pair,reference_ms=ref['per_apply_host_ms'],
                fused_ms=fused['per_apply_host_ms'],ratio=fused['host_ms']/ref['host_ms'],
                delta_ms=fused['per_apply_host_ms']-ref['per_apply_host_ms']))
    for name,rows in (('source_action_audit',sources),('target_solutions',targets),('native_timings',timings),
                      ('native_curves',curves),('apply_batches',batches),('paired_cost',pairs),('gates',gates),('coarse',coarse)):
        csv_new(tables/(name+'.csv'),rows)
    primary=[g for g in gates if g['primary']]
    numeric=all(r['common_CPU_quality_passed'] for r in targets) and all(
        g['used_fused'] and g['used_reference'] and g['iteration_budget_passed'] and g['action_calibration_passed'] for g in primary)
    costs=[]
    for scene in ('Bunny','Mixed'):
        pcg=[r for r in pairs if r['scene']==scene and r['scope']=='PCG'];apps=[r for r in pairs if r['scene']==scene and r['scope']=='apply16']
        pr=statistics.median(r['ratio'] for r in pcg);ar=statistics.median(r['ratio'] for r in apps)
        costs.append(dict(scene=scene,PCG_pair_ratio_median=pr,apply_pair_ratio_median=ar,
            exploratory_cost_gate_passed=pr<=1.03 and ar<=.90,
            **{key:statistics.mean(r[key] for r in pcg) for key in ('delta_ms','quality_delta_ms','screen_delta_ms','remaining_delta_ms')}))
    verify(inputs)
    write_new(out/'summary.json',dict(implementation_checks_passed=True,source=record(Path(__file__)),inputs=inputs,
        sources=sources,targets=targets,timings=timings,coarse=coarse,batches=batches,pairs=pairs,gates=gates,costs=costs,
        numerical_gate_passed=numeric,exploratory_cost_gate_passed=all(c['exploratory_cost_gate_passed'] for c in costs),
        production_solver_modified=False,performance_certified=False))
    print(json.dumps(dict(implementation_checks_passed=True,numerical_gate_passed=numeric,costs=costs)),flush=True)

if __name__=='__main__':main()
