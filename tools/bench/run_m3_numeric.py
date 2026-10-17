"""Finite, separately guarded offline experiments on nine frozen native systems."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from m3_numeric import (ROOT,FrozenSystem,Balanced,pcg,ritz_basis,pod_basis,harmonic_abd,
    allow_transfer,residual_metrics,rel,need)
from cpu_fixed_reference import decimal_residual
from local_identity import record,verify
from linux_runner import write_new
from m0_analyze import csv_write

CAPTURE=ROOT/'runs/innovation_m3_20261015'
OUTPUT=ROOT/'reports/innovation_m3_20261015/analysis_v2'
FRAMES=dict(Boxes=17,Mixed=15,Bunny=14)
RANKS=(8,16,32)


def native_calibration(system,folder,ordinary):
    records=[]
    if ordinary:
        audit=json.loads((folder/'operator_initial_audit.json').read_text())
        for probe in audit['probes']:
            i=probe['index'];base=folder/'operator_initial'
            v=np.fromfile(str(base)+f'_v{i}.bin',dtype='<f8');av=np.fromfile(str(base)+f'_Av{i}.bin',dtype='<f8')
            mv=np.fromfile(str(base)+f'_Mv{i}.bin',dtype='<f8')
            aerr=rel(system.full_A@v,av);old=system.original_action(v);linear=system.full_apply(v)
            records.append(dict(index=i,input=probe['input'],A_relative=aerr,
                FP32_surrogate_relative=rel(old,mv),linear_FP64_relative=rel(linear,mv)))
        need(max(r['A_relative'] for r in records)<1e-10,'Native A serialization/operator mismatch')
        need(max(r['linear_FP64_relative'] for r in records)<1e-4,'Linear reference/native MAS calibration failed')
    else:
        study=json.loads(Path(str(system.prefix)+'_study.json').read_text())
        for phase,tol in ((1,1e-4),(3,1e-16)):
            for mode in (0,1):
                x=np.fromfile(str(system.prefix)+f'_p{phase}_m{mode}_x.bin',dtype='<f8')
                measure,_=residual_metrics(system.A,system.b,x[system.active],system.apply)
                raw=next(r for r in study['runs'] if r['mode']==('host' if mode==0 else 'graph')
                    and r['rho_tolerance']==tol and r['repeat']==1)
                need(abs(measure['relative_l2']-raw['true_relative_residual'])<1e-8,'Saved native residual mismatch')
                records.append(dict(mode=raw['mode'],rho_tolerance=tol,iterations=raw['iterations'],
                    native_relative_l2=raw['true_relative_residual'],**measure))
    return records


def worker(scene,directory):
    need(not directory.exists(),'Fresh CPU scene directory required');directory.mkdir(parents=True)
    tasks=[('ordinary',1,1),('adjacent',FRAMES[scene],1),('adjacent',FRAMES[scene],2)]
    cases=[];previous=None
    for kind,frame,direction in tasks:
        label=f'{kind}_f{frame}_n{direction}';case_dir=directory/label;case_dir.mkdir()
        folder=CAPTURE/f'{kind}_{scene.lower()}';prefix=folder/'fixed'/f'f{frame}_n{direction}'
        started=time.perf_counter();system=FrozenSystem(prefix);A,b,B0=system.A,system.b,system.apply
        context={k:system.meta[k] for k in ('frame','inner','linear_system_id')}
        calibration=native_calibration(system,folder,kind=='ordinary')
        study=json.loads(Path(str(prefix)+'_study.json').read_text())
        results=[];curves=[]
        def save_solution(name,x):
            path=case_dir/(name+'_x.npy');np.save(path,x,allow_pickle=False);return record(path)
        base,xbase,history=pcg(A,b,B0,B0,collect=True)
        need(base['converged'],'Common-tolerance baseline did not converge')
        base['method']='MAS_linear_reference';base['actual_rank']=0;base['solution']=save_solution('baseline',xbase)
        curves.extend(dict(method=base['method'],**r) for r in base.pop('trace'));results.append(base)
        # Freeze transferable bases from the ordinary-accuracy source solve,
        # never from the extra high-accuracy oracle solve performed below.
        source_ritz,source_ritz_info=ritz_basis(A,history)
        source_pod,source_pod_info=pod_basis(A,history,xbase)
        packet_path=case_dir/'source_bases.npz'
        np.savez(packet_path,ritz=source_ritz,pod=source_pod)
        packet=dict(context=context,source_tolerance=.01,bases=record(packet_path),
            source_ritz_info=source_ritz_info,source_pod_info=source_pod_info,
            source_solve_iterations=base['iterations'],target_solution_used=False,
            layout=dict(dofs=len(b),soft_dofs=int(system.soft.sum()),abd_dofs=int(system.abd.sum())))
        write_new(case_dir/'source_packet.json',packet)
        current_packet=record(case_dir/'source_packet.json')

        def evaluate(name,vectors,construction=None,source=None,requested_rank=None):
            try:
                op=Balanced(A,B0,vectors);answer,x,_=pcg(A,b,op,B0)
                answer.update(method=name,actual_rank=op.Z.shape[1],requested_rank=requested_rank,
                    setup_seconds=op.setup_seconds,gram_asymmetry=op.gram_asymmetry,
                    Z_W_bytes=op.Z.nbytes+op.W.nbytes,construction=construction,source=source,
                    solution=save_solution(name,x))
                curves.extend(dict(method=name,**r) for r in answer.pop('trace'))
                # Diagnostic CPU action timings include no GPU implementation.
                clock=time.perf_counter()
                for _ in range(4):op(b)
                answer['CPU_apply_mean_ms']=250*(time.perf_counter()-clock)
            except Exception as error:
                answer=dict(method=name,converged=False,error=type(error).__name__+': '+str(error),requested_rank=requested_rank,
                    construction=construction,source=source)
            results.append(answer)

        # This block runs before the target high-accuracy oracle solve. It
        # cannot consume the target solution even accidentally.
        if kind=='adjacent' and direction==2:
            need(previous is not None,'Missing immediately preceding source packet')
            verify([previous]);old=json.loads(Path(previous['path']).read_text());verify([old['bases']])
            allow_transfer(old['context'],context);need(old['layout']==packet['layout'],'Global DOF layout changed')
            need(old['source_tolerance']==.01 and not old['target_solution_used'],'Illegal oracle source')
            with np.load(old['bases']['path'],allow_pickle=False) as data:
                for family in ('ritz','pod'):
                    z=data[family].copy()
                    for rank in RANKS:
                        evaluate('transfer_'+family+f'_{rank}',z[:,:rank],source=previous,requested_rank=rank)
                    soft=z.copy();soft[~system.soft]=0
                    for rank in RANKS:
                        evaluate('transfer_soft_'+family+f'_{rank}',soft[:,:rank],source=previous,requested_rank=rank)
        for rank in RANKS:
            evaluate(f'local_rhs_translation_{rank}',system.local_basis(rank),
                construction=dict(source='Current RHS and original fine MAS patch map; no solution'),requested_rank=rank)
        try:
            z,harmonic=harmonic_abd(system)
            if harmonic['applicable']:evaluate(harmonic['kind'],z,construction=harmonic,requested_rank=12)
        except Exception as error:
            results.append(dict(method='harmonic_ABD12',converged=False,error=type(error).__name__+': '+str(error)))

        reference,xref,oracle_history=pcg(A,b,B0,B0,tol=1e-8,collect=True)
        need(reference['converged'],'High-accuracy oracle solve did not converge')
        reference['solution']=save_solution('reference_1e_8',xref)
        curves.extend(dict(method='reference_1e_8',**r) for r in reference.pop('trace'))
        rz,ritz_info=ritz_basis(A,oracle_history)
        for rank in RANKS:
            evaluate(f'oracle_ritz_{rank}',rz[:,:rank],construction=ritz_info,requested_rank=rank)
            soft=rz[:,:rank].copy();soft[~system.soft]=0
            evaluate(f'oracle_soft_ritz_{rank}',soft,construction=ritz_info,requested_rank=rank)
        # Independent Decimal-50 scalar residual on each baseline and, for
        # each difficult first system, the best accepted candidate as well.
        decimal_checks=[]
        candidates=[base]
        if kind=='adjacent' and direction==1:
            candidates.append(min((r for r in results if r.get('converged')),key=lambda r:r['iterations']))
        for r in candidates:
            x=np.load(r['solution']['path'],allow_pickle=False);_,measure=decimal_residual(A,b,x)
            need(measure['relative_l2']<=.01,'Independent scalar residual failed')
            need(abs(measure['relative_l2']-r['relative_l2'])<1e-9,'Scalar/CSR residual disagreement')
            decimal_checks.append(dict(method=r['method'],**measure))
        csv_write(case_dir/'residual_curves.csv',curves)
        native_costs=study.get('operator_mean_us')
        native_default=[r for r in study['runs'] if r['mode']=='graph' and r['rho_tolerance']==1e-4]
        if native_default:
            native_iteration_ms=float(np.median([r['solve_ms']/r['iterations'] for r in native_default]))
            for r in results:
                if not r.get('converged'):continue
                saved=(base['iterations']-r['iterations'])*native_iteration_ms
                r['conditional_saved_GPU_ms_before_setup']=saved
                r['conditional_coarse_budget_us_per_iteration']=1000*saved/max(r['iterations'],1)
                r['cost_scope']='Offline iteration ratios with assumed native per-iteration cost; NOT measured GPU acceleration'
        else:native_iteration_ms=None
        case=dict(scene=scene,label=label,kind=kind,context=context,matrix=system.details,
            local_reference_checks=system.local_checks,native_calibration=calibration,
            input_records=list(system.evidence.values())+system.extra+[record(Path(str(prefix)+'_study.json'))],
            reference=reference,ritz_info=ritz_info,source_packet=current_packet,methods=results,
            independent_scalar_residuals=decimal_checks,native_operator_us=native_costs,
            native_default_iteration_ms=native_iteration_ms,seconds=time.perf_counter()-started,
            original_A_readonly=not A.data.flags.writeable,performance_certified=False)
        write_new(case_dir/'result.json',case);cases.append(record(case_dir/'result.json'))
        print(json.dumps(dict(scene=scene,case=label,base_iterations=base['iterations'],
            oracle_reference_iterations=reference['iterations'],methods=len(results),
            failures=[r['method'] for r in results if not r.get('converged')],seconds=case['seconds'])),flush=True)
        if kind=='adjacent' and direction==1:previous=current_packet
    write_new(directory/'scene_result.json',dict(scene=scene,complete=True,cases=cases,
        no_target_solution_in_transfer=True,performance_certified=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',choices=tuple(FRAMES))
    parser.add_argument('--case-output',type=Path);a=parser.parse_args()
    if a.worker:
        worker(a.worker,a.case_output);return
    import psutil
    need(not OUTPUT.exists(),'Fresh offline output required');OUTPUT.mkdir(parents=True)
    ledger=[];dependencies=[record(Path(__file__)),record(ROOT/'tools/bench/m3_numeric.py')]
    write_new(OUTPUT/'plan.json',dict(scenes=['Bunny','Mixed','Boxes'],source=record(CAPTURE/'ledger.json'),
        dependencies=dependencies,timeout_seconds_per_scene=600,rss_budget_mib=4096,
        ranks=RANKS,common_relative_l2_and_MAS_tolerance=.01,oracle_tolerance=1e-8))
    for scene in ('Bunny','Mixed','Boxes'):
        path=OUTPUT/scene;start=time.monotonic();peak=0;guard=None
        with (OUTPUT/(scene+'_worker.log')).open('x',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--worker',scene,
                '--case-output',str(path)],stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                try:peak=max(peak,psutil.Process(proc.pid).memory_info().rss)
                except psutil.NoSuchProcess:pass
                if time.monotonic()-start>600:guard='timeout'
                elif peak>4096*2**20:guard='rss_budget'
                if guard:proc.kill();proc.wait();break
                time.sleep(.25)
        row=dict(scene=scene,exit_code=proc.returncode,guard=guard,seconds=time.monotonic()-start,peak_rss_mib=peak/2**20)
        ledger.append(row);print(json.dumps(row),flush=True)
        if proc.returncode or guard:break
    verify(dependencies)
    write_new(OUTPUT/'ledger.json',dict(complete=len(ledger)==3 and all(r['exit_code']==0 and not r['guard'] for r in ledger),runs=ledger))
    need(len(ledger)==3 and all(r['exit_code']==0 and not r['guard'] for r in ledger),'Offline run incomplete; preserve failures')


if __name__=='__main__':main()
