import json
from pathlib import Path
import numpy as np
from analyze_perf_v36 import matrix_from_snapshot
from analyze_perf_v37 import difference
ROOT=Path(__file__).resolve().parents[1];data=ROOT/'downloads/replay_v42_strict_20261003';main=ROOT/'downloads/replay_v42_20261003';rows=[]
for case,version,name in [('v39_default',39,'wide_bunny'),('v39_strict',39,'wide_bunny_strict'),('v41_graph',41,'double_bunny'),('v41_host',41,'double_bunny_host'),('v41_wide',41,'wide_guard')]:
    run=data/'strict16'/case;raw=json.loads((run/'result.json').read_text())
    prefix=ROOT/f'downloads/autodl_perf_v{version}_20261003/runs/autodl'/f'autodl_perf_v{version}_{name}'/'mas_audit_failure'
    _,A,b=matrix_from_snapshot(prefix);solves=[];first=None
    for solve in raw['solves']:
        x=np.fromfile(run/solve['file'],'<f8');res=float(np.linalg.norm(b-A@x)/np.linalg.norm(b))
        if first is None:first=x
        solves.append({**{k:v for k,v in solve.items() if k!='history'},'cpu_true_residual':res,
            'cpu_gpu_residual_difference':abs(res-solve['true_relative_residual']),'solution_repeat':difference(first,x),
            'max_sampled_residual_drift':max(s['residual_drift_over_b'] for s in solve['history'])})
    control=main/'runs'/f'{case}_m7'
    row={'case':case,'solves':solves,'factor_bitwise_same_as_main':(run/'factors.bin').read_bytes()==(control/'factors.bin').read_bytes(),
         'all_probes_bitwise_same_as_main':all((run/f'Mv{i}.bin').read_bytes()==(control/f'Mv{i}.bin').read_bytes() for i in range(10)),
         'operator_repeat_passed':raw['operator_repeat_passed'],'all_rho_stop':all(s['reason']=='rho_stop' for s in solves),
         'max_cpu_true_residual':max(s['cpu_true_residual'] for s in solves),'max_repeat_relative':max(s['solution_repeat']['relative'] for s in solves)}
    row['strict_residual_passed']=row['max_cpu_true_residual']<=1e-8;row['repeat_passed']=row['max_repeat_relative']<=1e-6
    rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='solves'}),flush=True)
target=ROOT/'reports/REPLAY_V42_STRICT_CPU.json';assert not target.exists();target.write_text(json.dumps(rows,indent=2,allow_nan=False))
