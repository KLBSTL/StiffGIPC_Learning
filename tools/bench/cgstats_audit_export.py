"""Independent source-history/final-residual recheck and complete evidence tables."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import csv
import json
from pathlib import Path

import numpy as np

from cgstats_numeric import statistical_gram
from m3_numeric import ROOT,FrozenSystem,pcg,residual_metrics,rel,need
from local_identity import record,verify
from linux_runner import write_new
from m0_analyze import csv_write

REPORT=ROOT/'reports/innovation_p5_cgstats_20261015'
RAW=ROOT/'runs/innovation_p5_cgstats_20261015'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    out=REPORT/'tables';need(not out.exists(),'Fresh evidence tables');out.mkdir()
    summary=read(RAW/'summary.json');need(summary['complete'],'CPU gate incomplete');verify(summary['cases'])
    source_rows=[];method_rows=[];sources_verified=0;solutions_verified=0;decimal_checks=0
    for record_case in summary['cases']:
        path=Path(record_case['path']);case=read(path);verify(case['input_records']);folder=path.parent
        label=case['label'];kind=label.split('_')[0];scene=case['scene'];frame=case['source_context']['frame']
        source=FrozenSystem(ROOT/f'runs/innovation_m3_20261015/{kind}_{scene.lower()}/fixed/f{frame}_n1')
        r,x,history=pcg(source.A,source.b,source.apply,source.apply,collect=True)
        old=ROOT/f'reports/innovation_m3_20261015/analysis_v2/{scene}/{label}/baseline_x.npy'
        need(np.array_equal(x,np.load(old,allow_pickle=False)),'Independent source solve differs')
        S=np.column_stack([x-h[3] for h in history]);need(np.array_equal(S,np.load(folder/'S.npy',allow_pickle=False)),
                                                     'Independent source history differs')
        with np.load(folder/'statistics.npz',allow_pickle=False) as data:
            alpha=data['alpha'];rho=data['rho'];ids=data['indices']
            need(np.array_equal(alpha,np.array([t['alpha'] for t in r['trace']])) and
                 np.array_equal(rho,np.array([t['rho'] for t in r['trace']])) and
                 np.array_equal(ids,np.array([h[0] for h in history])),'Source statistics mismatch')
            H,_=statistical_gram(alpha,rho,ids,curvature=data['curvature'])
        need(np.array_equal(H,np.load(folder/'statistics_Gram.npy',allow_pickle=False)),'Statistics Gram changed')
        G=S.T@(source.A@S);G=.5*(G+G.T)
        need(rel(G,np.load(folder/'direct_Gram.npy',allow_pickle=False))<1e-12,'Direct Gram changed')
        U=np.load(folder/'coefficients.npy',allow_pickle=False);ev=np.linalg.eigvalsh(G)
        for rank in (8,16):
            if U.shape[1]<rank:continue
            retained=float(np.trace(U[:,:rank].T@G@U[:,:rank])/ev[-rank:].sum())
            need(0<=retained<=1+1e-10,'Invalid true covariance energy ratio')
            source_rows.append(dict(scene=scene,label=label,rank=rank,Gram_relative=rel(H,G),
                direct_POD_covariance_energy_ratio=retained,history_bitwise_equal=True,source_solution_bitwise_equal=True))
        sources_verified+=1
        if kind=='adjacent':
            target=FrozenSystem(ROOT/f'runs/innovation_m3_20261015/adjacent_{scene.lower()}/fixed/f{frame}_n2')
            with (folder/'residual_curves.csv').open(encoding='utf-8-sig') as f:curves=list(csv.DictReader(f))
            for m in case['methods']:
                need(m['converged'],'Retained target failure must be reported before claiming pass');verify([m['solution']])
                value=np.load(m['solution']['path'],allow_pickle=False)
                metrics,_=residual_metrics(target.A,target.b,value,target.apply)
                need(max(metrics.values())<=.01 and max(abs(metrics[k]-m[k]) for k in metrics)<1e-12,'Target residual mismatch')
                trace=[t for t in curves if t['method']==m['method']]
                need(len(trace)==m['iterations'] and int(trace[-1]['iteration'])==m['iterations'],'Iteration curve count mismatch')
                if m['method'].startswith('CG_statistics'):
                    need(m['Decimal50_relative_l2']<=.01 and abs(m['Decimal50_relative_l2']-metrics['relative_l2'])<1e-9,'Independent scalar residual mismatch')
                    decimal_checks+=1
                method_rows.append({k:m[k] for k in ('scene','method','requested_rank','actual_rank','baseline_iterations','iterations','relative_l2','common_MAS_relative')})
                solutions_verified+=1
    GPU=[];component=[];pairs=[];budgets=[];apply=[];QA=[];launch_rows=[]
    for name in ('fixture_positive','fixture_negative','mixed_cost','fixture_memcheck'):
        launch=read(RAW/name/'launch.json');need(launch['passed'] and launch['cleanup_owned_job_empty'] and launch['wall_seconds']<180,'GPU job incomplete')
        need(all(x['power_guid']==launch['power_guid'] for x in launch['samples']),'Power drift')
        launch_rows.append(dict(name=name,status=launch['status'],exit_code=launch['exit_code'],wall_seconds=launch['wall_seconds'],
            power_guid=launch['power_guid'],GPU_utilization_before=launch['gpu_before']['utilization.gpu'],
            observed_memory_drop_mib=launch['gpu_before']['memory.free']-min(x['memory.free'] for x in launch['samples'])))
        if name=='fixture_negative':continue
        validation=read(RAW/name/'validation.json');need(validation['passed'],'Independent GPU validation failed')
        verify(validation['inputs']+validation['results'])
        if name!='mixed_cost':need(not validation['conditional_budgets'],'A tiny fixture has a scene speed budget');continue
        run=read(RAW/name/'data/result.json')
        for c in run['ranks']:
            for t in c['pipeline_samples']:GPU.append(dict(rank=c['rank'],family=c['family'],**t))
            for stage,t in c['component_times'].items():component.append(dict(rank=c['rank'],family=c['family'],stage=stage,**t))
            for t in c['apply_samples']:apply.append(dict(rank=c['rank'],family=c['family'],**t))
        pairs=validation['setup_pairs'];budgets=[{k:v for k,v in b.items() if k not in ('paired_added_us','scope')} for b in validation['conditional_budgets']]
        QA=[{k:(max(v) if isinstance(v,list) else v) for k,v in q.items()} for q in validation['checks']]
    sanitizer=(RAW/'fixture_memcheck/run.log').read_text(encoding='utf-8-sig',errors='replace')
    need('ERROR SUMMARY: 0 errors' in sanitizer,'Memcheck did not pass')
    for name,rows in (('source_quality',source_rows),('CPU_methods',method_rows),('setup_samples',GPU),('components',component),
                      ('matched_setup_pairs',pairs),('conditional_budget',budgets),('apply_pairs',apply),('GPU_numerical_checks',QA),('launches',launch_rows)):
        csv_write(out/(name+'.csv'),rows)
    need(sources_verified==6 and solutions_verified==12 and decimal_checks==6,'Unexpected evidence coverage')
    write_new(REPORT/'AUDIT.json',dict(passed=True,source=record(Path(__file__)),source_histories_bitwise_rechecked=sources_verified,
        target_solutions_residual_rechecked=solutions_verified,Decimal50_checks=decimal_checks,
        GPU_jobs=4,expected_rejections=1,memcheck_errors=0,production_solver_modified=False,performance_certified=False))
    print(json.dumps(dict(passed=True,sources=sources_verified,target_solutions=solutions_verified,memcheck_errors=0)),flush=True)


if __name__=='__main__':main()
