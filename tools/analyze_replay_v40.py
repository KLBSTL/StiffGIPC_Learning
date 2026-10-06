"""Independent symmetric sparse A residual verification for all v40 fixed solves."""
import json
from pathlib import Path
import numpy as np
from analyze_perf_v36 import matrix_from_snapshot
from analyze_perf_v37 import difference

ROOT=Path(__file__).resolve().parents[1]
data=ROOT/'downloads/replay_v40_20261003'
old=ROOT/'downloads/autodl_perf_v39_20261003/runs/autodl'
records=[]
for case,name in [('default','autodl_perf_v39_wide_bunny'),('strict','autodl_perf_v39_wide_bunny_strict')]:
    prefix=old/name/'mas_audit_failure';meta,A,b=matrix_from_snapshot(prefix)
    for mode in [0,4,5]:
        run=data/'runs'/f'{case}_m{mode}';raw=json.loads((run/'result.json').read_text())
        row={'case':case,'mode':raw['mode'],'inverse32_bitwise':raw['native_inverse_recomputed_bitwise'],'solves':[]}
        first={}
        for solve in raw['solves']:
            x=np.fromfile(run/solve['file'],'<f8');res=float(np.linalg.norm(b-A@x)/np.linalg.norm(b))
            first.setdefault(solve['tolerance'],x)
            history=solve['history']
            entry={k:v for k,v in solve.items() if k!='history'}
            entry.update(cpu_true_residual=res,cpu_gpu_residual_difference=abs(res-solve['true_relative_residual']),
                         solution_repeat=difference(first[solve['tolerance']],x),history_samples=len(history),
                         max_sampled_residual_drift=max((h['residual_drift_over_b'] for h in history),default=None),
                         final_sample=history[-1] if history else None)
            row['solves'].append(entry)
        records.append(row)
        print(json.dumps({'case':case,'mode':raw['mode'],'reasons':[s['reason'] for s in row['solves']],
            'max_cpu_gpu_residual_difference':max(s['cpu_gpu_residual_difference'] for s in row['solves']),
            'strict_max_residual':max(s['cpu_true_residual'] for s in row['solves'] if s['tolerance']==1e-14),
            'strict_repeat_relative':max(s['solution_repeat']['relative'] for s in row['solves'] if s['tolerance']==1e-14)}),flush=True)
out=ROOT/'reports/PCG_V40_REPLAY_CPU.json';assert not out.exists();out.write_text(json.dumps(records,indent=2,allow_nan=False))
assert max(s['cpu_gpu_residual_difference'] for r in records for s in r['solves'])<1e-6
