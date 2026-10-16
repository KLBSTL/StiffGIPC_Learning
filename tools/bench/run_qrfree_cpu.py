"""Three preselected source-to-target coordinate checks; historical bytes immutable."""
import json
from pathlib import Path
import numpy as np
from m3_numeric import ROOT,FrozenSystem,Balanced,pcg,rel,need,residual_metrics
from cgstats_numeric import coefficients
from qrfree_numeric import CoordinateBalanced,scaled_coefficients
from cpu_fixed_reference import decimal_residual
from local_identity import record,verify
from linux_runner import write_new

RAW=ROOT/'runs/innovation_p5_qrfree_20261016'
OLD=ROOT/'runs/innovation_p5_cgstats_20261015'
REPORT=ROOT/'reports/innovation_p5_qrfree_20261016'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    need(not RAW.exists(),'Fresh batch required');verify(read(ROOT/'reports/innovation_p5_cgstats_20261015/DELIVERY_SEAL.json')['files'])
    RAW.mkdir();write_new(RAW/'plan.json',dict(plan=record(REPORT/'PLAN.md'),rank=8,
        dependencies=[record(Path(__file__)),record(ROOT/'tools/bench/qrfree_numeric.py')],
        native_history_available=False,production_modified=False))
    results=[]
    for scene,frame in (('Mixed',15),('Bunny',14),('Boxes',17)):
        folder=RAW/scene;folder.mkdir();old=OLD/scene/f'adjacent_f{frame}_n1'
        S=np.load(old/'S.npy',allow_pickle=False);H=np.load(old/'statistics_Gram.npy',allow_pickle=False)
        u,info=coefficients(H,8);need(info['actual_rank']==8,'No rank completion')
        vectors=S@scaled_coefficients(u,info['selected_eigenvalues'])
        target=FrozenSystem(ROOT/f'runs/innovation_m3_20261015/adjacent_{scene.lower()}/fixed/f{frame}_n2')
        qr=Balanced(target.A,target.apply,S@u);op=CoordinateBalanced(target.A,target.apply,vectors)
        rng=np.random.default_rng(20261016);probes=np.column_stack([target.b,rng.normal(size=len(target.b)),rng.normal(size=len(target.b))])
        action_errors=[rel(op(v),qr(v)) for v in probes.T];need(max(action_errors)<1e-8,'Coordinate action changed')
        methods=[];curves=[]
        for name,action in (('QR',qr),('QR_free',op)):
            r,x,_=pcg(target.A,target.b,action,target.apply);curves.extend(dict(method=name,**t) for t in r.pop('trace'))
            metrics,_=residual_metrics(target.A,target.b,x,target.apply)
            need(r['converged'] and max(metrics.values())<=.01,'Common residual failed')
            _,dec=decimal_residual(target.A,target.b,x)
            need(abs(dec['relative_l2']-metrics['relative_l2'])<1e-9,'Independent scalar residual failed')
            path=folder/(name+'_x.npy');np.save(path,x,allow_pickle=False);r.update(method=name,solution=record(path),Decimal50_relative_l2=dec['relative_l2'])
            methods.append(r)
        previous=next(x for x in read(old/'result.json')['methods'] if x['method']=='CG_statistics_8')
        need(methods[0]['iterations']==previous['iterations'],'QR strong control changed')
        gate=methods[1]['iterations']<=1.10*methods[0]['iterations'];need(gate,'Iteration gate failed')
        np.save(folder/'scaled_basis.npy',vectors,allow_pickle=False)
        write_new(folder/'curves.json',curves)
        result=dict(scene=scene,frame=frame,rank=8,action_relative_errors=action_errors,
            coarse_eigenvalues=op.eigenvalues.tolist(),coarse_condition=float(op.eigenvalues[-1]/op.eigenvalues[0]),
            coarse_asymmetry=op.asymmetry,methods=methods,iteration_gate_passed=gate,
            inputs=[record(old/f) for f in ('S.npy','statistics_Gram.npy','result.json')])
        write_new(folder/'result.json',result);results.append(result)
        print(json.dumps(dict(scene=scene,iterations=[r['iterations'] for r in methods],condition=result['coarse_condition'],action_error=max(action_errors))),flush=True)
    write_new(RAW/'CPU_summary.json',dict(passed=True,cases=results,native_PCG=False))


if __name__=='__main__':main()
