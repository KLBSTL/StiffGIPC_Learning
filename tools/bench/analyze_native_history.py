"""Native scalar/history audit and source-only rank-eight transfer; CPU targets."""
import json
from pathlib import Path
import numpy as np
from m3_numeric import ROOT,FrozenSystem,pcg,residual_metrics,orthogonal,rel,need
from cgstats_numeric import statistical_gram,coefficients
from qrfree_numeric import CoordinateBalanced,scaled_coefficients
from cpu_fixed_reference import decimal_residual
from local_identity import record,verify
from linux_runner import write_new
from m0_analyze import csv_write

RAW=ROOT/'runs/innovation_p5_native_history_20261016'
REPORT=ROOT/'reports/innovation_p5_native_history_20261016'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    need(read(RAW/'representative_ledger.json')['complete'],'Representative captures required')
    out=RAW/'analysis';need(not out.exists(),'Fresh native analysis');out.mkdir()
    quality=[];methods=[];curves=[];comparisons=[];inputs=[];checks=[]
    for label,scene,frame in (('bunny','Bunny',14),('mixed','Mixed',15)):
        folder=RAW/label;systems={};banks={}
        for direction in (1,2):
            prefix=folder/'fixed'/f'f{frame}_n{direction}';study=read(str(prefix)+'_native_history_study.json')
            system=FrozenSystem(prefix);systems[direction]=system
            banks[direction]=[];n=len(system.full_b)
            for row in study['runs']:
                xfull=np.fromfile(row['solution_file'],dtype='<f8');need(xfull.shape==(n,),'Native solution size')
                x=xfull[system.active];metrics,_=residual_metrics(system.A,system.b,x,system.apply)
                full_residual=np.linalg.norm(system.full_b-system.full_A@xfull)/np.linalg.norm(system.full_b)
                need(abs(full_residual-row['true_relative_residual'])<1e-8,'CPU original-A residual differs from native SpMV')
                checks.append(dict(scene=scene,direction=direction,pair=row['pair'],capture=row['capture'],
                    iterations=row['iterations'],native_relative_l2=row['true_relative_residual'],
                    CPU_full_relative_l2=float(full_residual),**metrics))
                inputs.append(record(row['solution_file']))
                if not row['capture']:continue
                steps=row['iterations'];w=row['history_width'];indices=np.fromfile(row['indices_file'],dtype='<i4')
                history=np.fromfile(row['history_file'],dtype='<f8').reshape(n,w,order='F')
                scalar=np.fromfile(row['scalars_file'],dtype='<f8').reshape(steps,3)
                need(len(indices)==w and indices[0]==0 and indices[-1]==steps-1 and np.all(np.diff(indices)>0),
                     'Invalid native history indices')
                need(w==min(steps,128) and np.array_equal(history[:,0],np.zeros(n)),'Native bank/initial x invalid')
                expected=np.r_[np.arange(min(16,steps)),np.arange(max(16,steps-112),steps)]
                need(np.array_equal(indices,expected),'Native first/tail bank differs from protocol')
                rho=scalar[:,1];threshold=1e-4*rho[0]
                if not row['pcg'].get('zero_residual',False):
                    need(np.all(rho[:-1]>threshold) and rho[-1]<=threshold,'Native previous-rho stopping changed')
                S=(xfull[:,None]-history)[system.active]
                H,_=statistical_gram(scalar[:,0],rho,indices,curvature=scalar[:,2])
                direct=S.T@(system.A@S);direct=.5*(direct+direct.T)
                u,info=coefficients(H,8);ud,di=coefficients(direct,8)
                qrow=dict(scene=scene,direction=direction,pair=row['pair'],native_iterations=steps,width=w,
                    Gram_relative=rel(H,direct),stat_rank=info['actual_rank'],direct_rank=di['actual_rank'])
                if min(info['actual_rank'],di['actual_rank'])>=8:
                    zs=S@scaled_coefficients(u,info['selected_eigenvalues'])
                    zd=S@scaled_coefficients(ud,di['selected_eigenvalues']);qs,_=orthogonal(zs);qd,_=orthogonal(zd)
                    qrow['projection_relative']=rel(qs,qd@(qd.T@qs))
                    banks[direction].append((row['pair'],zs,zd))
                quality.append(qrow)
                for key in ('history_file','indices_file','scalars_file'):inputs.append(record(row[key]))
            controls=[r for r in study['runs'] if not r['capture']]
            captures=[r for r in study['runs'] if r['capture']]
            base=np.fromfile(controls[0]['solution_file'],dtype='<f8')
            comparisons.append(dict(scene=scene,direction=direction,control_iterations=[r['iterations'] for r in controls],
                capture_iterations=[r['iterations'] for r in captures],
                control_repeat_relative=[rel(np.fromfile(r['solution_file'],dtype='<f8'),base) for r in controls],
                capture_vs_first_control_relative=[rel(np.fromfile(r['solution_file'],dtype='<f8'),base) for r in captures]))
        target=systems[2];source=systems[1]
        need(np.array_equal(source.active,target.active),'Native global coordinate map changed')
        for pair,zs,zd in banks[1]:
            for method,z in (('direct_native_history',zd),('statistics_native_history',zs)):
                try:action=CoordinateBalanced(target.A,target.apply,z)
                except ValueError as error:
                    if 'coarse rank/conditioning rejected' not in str(error):raise
                    methods.append(dict(scene=scene,pair=pair,method=method,status='coarse_rank_rejected',
                        reason=str(error),converged=False,target_solution_used_for_basis=False))
                    continue
                r,x,_=pcg(target.A,target.b,action,target.apply)
                curves.extend(dict(scene=scene,pair=pair,method=method,**t) for t in r.pop('trace'))
                metrics,_=residual_metrics(target.A,target.b,x,target.apply)
                need(r['converged'] and max(metrics.values())<=.01,'Native-history transfer common residual failed')
                if pair==1:
                    _,dec=decimal_residual(target.A,target.b,x)
                    need(abs(dec['relative_l2']-metrics['relative_l2'])<1e-9,'Transfer Decimal50 residual failed')
                    r['Decimal50_relative_l2']=dec['relative_l2']
                dest=out/f'{scene}_p{pair}_{method}_x.npy';np.save(dest,x,allow_pickle=False)
                methods.append(dict(scene=scene,pair=pair,method=method,**r,coarse_condition=float(action.eigenvalues[-1]/action.eigenvalues[0]),
                    solution=record(dest),target_solution_used_for_basis=False))
        r,x,_=pcg(target.A,target.b,target.apply,target.apply)
        curves.extend(dict(scene=scene,pair=0,method='MAS_reference',**t) for t in r.pop('trace'))
        need(r['converged'],'Target MAS reference did not converge');dest=out/f'{scene}_MAS_reference_x.npy';np.save(dest,x,allow_pickle=False)
        methods.append(dict(scene=scene,pair=0,method='MAS_reference',**r,solution=record(dest)))
    gates=[]
    for scene in ('Bunny','Mixed'):
        for pair in (1,2,3):
            rows=[r for r in methods if r['scene']==scene and r['pair']==pair]
            direct=next((r for r in rows if r['method']=='direct_native_history'),None)
            stats=next((r for r in rows if r['method']=='statistics_native_history'),None)
            gates.append(dict(scene=scene,pair=pair,primary=pair==1,direct_iterations=direct.get('iterations') if direct else None,
                statistics_iterations=stats.get('iterations') if stats else None,
                passed=bool(direct and stats and direct.get('converged') and stats.get('converged')
                    and stats['iterations']<=1.10*direct['iterations'])))
    for name,rows in (('source_quality',quality),('native_solutions',checks),('CPU_methods',methods),('CPU_curves',curves),('iteration_gate',gates)):
        csv_write(out/(name+'.csv'),rows)
    verify(inputs);write_new(out/'summary.json',dict(implementation_checks_passed=True,
        primary_iteration_gate_passed=all(g['passed'] for g in gates if g['primary']),gates=gates,source_quality=quality,
        native_repeat_comparison=comparisons,methods=methods,inputs=inputs,analyzer=record(Path(__file__)),
        target_solver='fixed FP64 coefficient reference; not native balanced PCG',native_graph_history_captured=False,
        performance_certified=False))
    print(json.dumps(dict(implementation_checks_passed=True,gates=gates,source_quality=quality)),flush=True)


if __name__=='__main__':main()
