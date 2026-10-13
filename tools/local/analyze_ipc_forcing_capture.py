"""Independent saved-A residual/descent and dynamic Graph-control checks."""
from pathlib import Path
import argparse
import json
import os
import statistics
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
for folder in ('local','bench'):
    sys.path.insert(0,str(ROOT/'tools'/folder))
from cpu_fixed_reference import load_snapshot
from audit_recycling_reference import preconditioner
from local_identity import record,verify
from config import read
from linux_runner import require,write_new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    require(not args.output.exists(),'Fresh report required')
    cases=[]
    for file in sorted(args.capture.glob('*/fixed/*_study.json')):
        prefix=Path(str(file)[:-11])
        study=read(file)
        require(study['eta_study'] and study['system_unchanged'] and study['primary_restored_bitwise'],
            'Operator/primary solution was not restored')
        matrix,rhs,active,zero,evidence,detail=load_snapshot(prefix)
        apply,operator_evidence=preconditioner(prefix,read(Path(str(prefix)+'_meta.json')))
        reference=float(rhs@apply(rhs));require(reference>0,'Nonpositive common baseline rho')
        solutions={}
        rows=[]
        for phase,eta in enumerate((.01,.03,.1,.01)):
            rho_tolerance=(1e-4,9e-4,1e-2,1e-4)[phase]
            for mode,name in ((0,'host'),(1,'graph')):
                path=Path(str(prefix)+f'_p{phase}_m{mode}_x.bin')
                x=np.fromfile(path,dtype='<f8')
                require(x.shape==rhs.shape and np.isfinite(x).all(),'Invalid saved native solution')
                true=rhs-matrix@x
                relative=float(np.linalg.norm(true)/np.linalg.norm(rhs))
                descent=float(rhs@x)
                require(descent>0,'Non-descent native solution')
                native=[r for r in study['runs'] if r['mode']==name and r['rho_tolerance']==rho_tolerance]
                # eta=.01 appears twice; preserve phase ordering of the same tolerance.
                if phase in (0,3):native=native[:2] if phase==0 else native[-2:]
                require(len(native)==2 and all(not r['limit'] and not r['error'] for r in native),
                    'Missing or failed native runs')
                require(abs(relative-native[0]['true_relative_residual'])<1e-8,
                    'Independent full-A residual disagrees with native diagnostic')
                if name=='graph' and phase>0:
                    require(all(r['graph_cache_hit'] for r in native),
                        'Tolerance-value change recaptured Graph')
                solutions[phase,mode]=x
                rows.append(dict(phase=phase,eta=eta,rho_tolerance=rho_tolerance,mode=name,
                    iterations=[r['iterations'] for r in native],
                    median_native_solve_ms=statistics.median(r['solve_ms'] for r in native),
                    true_relative_residual_FP64=relative,rhs_dot_solution=descent,
                    true_common_weighted_rho_CPU=float(true@apply(true))/reference,
                    graph_cache_hits=[r.get('graph_cache_hit') for r in native],solution=record(path),
                    relative_to_baseline_solution=float(np.linalg.norm(x-solutions[0,mode])/
                        np.linalg.norm(solutions[0,mode]))))
        graph=[r for r in rows if r['mode']=='graph']
        require(graph[1]['iterations'][0]<graph[0]['iterations'][0] and
            graph[2]['iterations'][0]<graph[0]['iterations'][0], 'Loose Graph tolerance did not change stopping')
        baseline_repeat_difference=float(np.linalg.norm(solutions[3,1]-solutions[0,1])/
            np.linalg.norm(solutions[0,1]))
        case=dict(prefix=str(prefix),system=detail,evidence=evidence,operator_evidence=operator_evidence,
            studies=record(file),rows=rows,baseline_graph_repeat_difference=baseline_repeat_difference,
            independent_checks_passed=True,operator_unchanged=True,primary_restored_bitwise=True)
        cases.append(case)
        print(json.dumps(dict(prefix=prefix.name,graph=graph,baseline_graph_repeat_difference=baseline_repeat_difference)),flush=True)
    require(len(cases)==6,'Expected six selected IPC matrices')
    write_new(args.output,dict(controller=record(__file__),cases=cases,passed=True,
        performance_certified=False,physical_quality_certified=False,
        weighted_residual_scope='CPU replay of saved FP32 MAS coefficients; rounding order differs from GPU',
        stopping_scope='Preserved previous-rho check; current true residual need not satisfy requested eta'))


if __name__=='__main__':
    main()
