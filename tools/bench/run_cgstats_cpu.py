"""Finite CPU gate on unchanged ordinary sources and three real adjacent pairs."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import json
from pathlib import Path
import time

import numpy as np

from cgstats_numeric import statistical_gram,coefficients
from m3_numeric import ROOT,FrozenSystem,pcg,Balanced,orthogonal,rel,allow_transfer,need
from cpu_fixed_reference import decimal_residual
from local_identity import record,verify
from linux_runner import write_new
from m0_analyze import csv_write

REPORT=ROOT/'reports/innovation_p5_cgstats_20261015'
RAW=ROOT/'runs/innovation_p5_cgstats_20261015'
M3=ROOT/'reports/innovation_m3_20261015/analysis_v2'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    need(not RAW.exists(),'Fresh CG statistics run required');RAW.mkdir()
    parent=ROOT/'reports/innovation_p5_gpu_20261015/DELIVERY_SEAL.json';verify(read(parent)['files'])
    dependencies=[record(Path(__file__)),record(ROOT/'tools/bench/cgstats_numeric.py'),record(ROOT/'tools/bench/m3_numeric.py')]
    write_new(RAW/'plan.json',dict(parent=record(parent),dependencies=dependencies,states=['Mixed15','Bunny14','Boxes17'],
        ordinary_sources=True,ranks=[8,16],common_residual_tolerance=.01,
        Mixed_iteration_gate=1.10,solver_changed=False,target_solution_used_for_basis=False))
    cases=[];all_methods=[];matrix_rows=[];gate=[]
    for scene,frame in (('Mixed',15),('Bunny',14),('Boxes',17)):
        for kind,physical_frame in (('ordinary',1),('adjacent',frame)):
            start=time.perf_counter();label=f'{kind}_f{physical_frame}_n1';out=RAW/scene/label;out.mkdir(parents=True)
            parent_case=read(M3/scene/label/'result.json');verify(parent_case['input_records'])
            prefix=ROOT/f'runs/innovation_m3_20261015/{kind}_{scene.lower()}/fixed/f{physical_frame}_n1'
            source=FrozenSystem(prefix)
            result,x,history=pcg(source.A,source.b,source.apply,source.apply,collect=True)
            need(result['converged'] and result['residual_restarts']==0,'Source PCG does not fit fixed recurrence model')
            old=np.load(parent_case['methods'][0]['solution']['path'],allow_pickle=False)
            need(np.array_equal(x,old),'Source differs from sealed ordinary-accuracy solution')
            packet=read(parent_case['source_packet']['path']);verify([packet['bases']])
            indices=np.array([h[0] for h in history],dtype=np.int32);S=np.column_stack([x-h[3] for h in history])
            trace=result['trace'];alpha=np.array([t['alpha'] for t in trace]);rho=np.array([t['rho'] for t in trace])
            curvature=np.array([t['p_Ap'] for t in trace]);stats,info=statistical_gram(alpha,rho,indices,
                restarts=result['residual_restarts'],curvature=curvature)
            clock=time.perf_counter();direct=S.T@(source.A@S);direct=.5*(direct+direct.T);direct_seconds=time.perf_counter()-clock
            gram_error=rel(stats,direct);U,ci=coefficients(stats,16);Ud,di=coefficients(direct,16)
            np.save(out/'S.npy',S,allow_pickle=False);np.save(out/'direct_Gram.npy',direct,allow_pickle=False)
            np.save(out/'statistics_Gram.npy',stats,allow_pickle=False);np.save(out/'coefficients.npy',U,allow_pickle=False)
            np.savez(out/'statistics.npz',alpha=alpha,rho=rho,curvature=curvature,indices=indices)
            row=dict(scene=scene,label=label,source_iterations=result['iterations'],history_width=len(history),
                Gram_relative_error=gram_error,exact_identity_tolerance_passed=gram_error<=1e-8,
                statistics_seconds=info['seconds'],CPU_eigensolve_seconds=ci['seconds'],
                direct_Gram_seconds=direct_seconds,statistics_rank=ci['actual_rank'],direct_rank=di['actual_rank'])
            matrix_rows.append(row);methods=[];curves=[];spaces=[]
            with np.load(packet['bases']['path'],allow_pickle=False) as archive:old_pod=archive['pod'].copy()
            if kind=='adjacent':
                target_case=read(M3/scene/f'adjacent_f{frame}_n2/result.json');verify(target_case['input_records'])
                target=FrozenSystem(ROOT/f'runs/innovation_m3_20261015/adjacent_{scene.lower()}/fixed/f{frame}_n2')
                allow_transfer(source.meta,target.meta)
                need(np.array_equal(source.active,target.active) and np.array_equal(source.soft,target.soft) and
                     np.array_equal(source.abd,target.abd),'Global coordinate layout differs')
            for rank in (8,16):
                if U.shape[1]<rank or Ud.shape[1]<rank or old_pod.shape[1]<rank:
                    spaces.append(dict(rank=rank,status='insufficient_rank',available=U.shape[1]));continue
                clock=time.perf_counter();vectors=S@U[:,:rank];basis_seconds=time.perf_counter()-clock
                z,_=orthogonal(vectors);zd,_=orthogonal(S@Ud[:,:rank]);zo,_=orthogonal(old_pod[:,:rank])
                need(rel(zd,zo@(zo.T@zd))<1e-8,'Rebuilt direct POD disagrees with the sealed source space')
                np.save(out/f'basis_{rank}.npy',vectors,allow_pickle=False)
                spaces.append(dict(rank=rank,status='available',projection_error_vs_direct=rel(z,zd@(zd.T@z)),
                    CPU_extract_seconds=basis_seconds,actual_rank=z.shape[1],source_Gram_energy=np.trace(z.T@(source.A@z))))
                if kind!='adjacent':continue
                for family,raw in (('direct_POD',old_pod[:,:rank]),('CG_statistics',vectors)):
                    name=family+'_'+str(rank)
                    try:
                        op=Balanced(target.A,target.apply,raw);r,solution,_=pcg(target.A,target.b,op,target.apply)
                        curves.extend(dict(method=name,**t) for t in r.pop('trace'))
                        need(r['converged'],'Target residual target not met')
                        path=out/(name+'_x.npy');np.save(path,solution,allow_pickle=False)
                        r.update(scene=scene,method=name,requested_rank=rank,actual_rank=op.Z.shape[1],
                            baseline_iterations=target_case['methods'][0]['iterations'],setup_seconds=op.setup_seconds,
                            solution=record(path),target_solution_used_for_basis=False)
                        if family=='CG_statistics':
                            _,independent=decimal_residual(target.A,target.b,solution)
                            need(independent['relative_l2']<=.01 and abs(independent['relative_l2']-r['relative_l2'])<1e-9,
                                 'Independent Decimal50 residual failed')
                            r['Decimal50_relative_l2']=independent['relative_l2']
                    except Exception as error:r=dict(scene=scene,method=name,converged=False,error=str(error),requested_rank=rank)
                    methods.append(r);all_methods.append(r)
            if kind=='adjacent':
                csv_write(out/'residual_curves.csv',curves)
                for rank in (8,16):
                    selected=[r for r in methods if r['requested_rank']==rank]
                    ok=len(selected)==2 and all(r.get('converged') and r['actual_rank']==rank for r in selected)
                    direct_result=next((r for r in selected if r['method'].startswith('direct')),None)
                    stats_result=next((r for r in selected if r['method'].startswith('CG_statistics')),None)
                    acceptable=ok and stats_result['iterations']<=1.10*direct_result['iterations']
                    gate.append(dict(scene=scene,rank=rank,passed=bool(acceptable),
                        direct_iterations=direct_result.get('iterations') if direct_result else None,
                        statistics_iterations=stats_result.get('iterations') if stats_result else None))
            write_new(out/'result.json',dict(**row,spaces=spaces,methods=methods,seconds=time.perf_counter()-start,
                source_bitwise_equal=True,source_context={k:source.meta[k] for k in ('frame','inner','linear_system_id')},
                source_tolerance=.01,approximate_Gram=True,source_packet=parent_case['source_packet'],
                input_records=parent_case['input_records']+(target_case['input_records'] if kind=='adjacent' else []),
                performance_certified=False))
            cases.append(record(out/'result.json'));print(json.dumps(dict(scene=scene,kind=kind,
                Gram_relative=gram_error,iterations=[(r['method'],r.get('iterations')) for r in methods])),flush=True)
    verify(dependencies);csv_write(REPORT/'Gram_comparison.csv',matrix_rows);csv_write(REPORT/'CPU_gate.csv',gate)
    fields=sorted({k for r in all_methods for k in r if k not in ('solution',)})
    csv_write(REPORT/'CPU_methods.csv',[{k:r.get(k) for k in fields} for r in all_methods])
    write_new(RAW/'summary.json',dict(complete=True,cases=cases,Mixed_primary_gate=next(g for g in gate if g['scene']=='Mixed' and g['rank']==8),
        gates=gate,approximate_Gram=True,native_solver_modified=False,GPU_performance_certified=False))


if __name__=='__main__':main()
