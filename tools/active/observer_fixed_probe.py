"""Independent CPU references for the two same-process repeat diagnostics."""
import json
import sys
from pathlib import Path
import numpy as np
from config import ROOT,sha,read
from ipc_benchmark import write
from verify_systems import accurate_reference,matrix_from_snapshot

def main():
    rows=[]
    for frame in [2,25]:
        run=ROOT/f'runs/active/ipc_observer_fixed_probe_20261005_f{frame}'
        prefix=run/f'fixed/f{frame}_n1';study=read(Path(str(prefix)+'_study.json'))
        meta,a,b=matrix_from_snapshot(prefix);reference,ref=accurate_reference(a,b)
        solutions=[]
        for phase in [1,3]:
            for mode in [0,1]:
                path=Path(str(prefix)+f'_p{phase}_m{mode}_x.bin');x=np.fromfile(path,dtype='<f8')
                solutions.append({'phase':phase,'mode':'host' if mode==0 else 'graph',
                    'rho_tolerance':1e-4 if phase==1 else 1e-16,
                    'cpu_true_relative_residual':float(np.linalg.norm(b-a@x)/np.linalg.norm(b)),
                    'relative_solution_error_to_cpu':float(np.linalg.norm(x-reference)/np.linalg.norm(reference)),
                    'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
        inp=[Path(str(prefix)+s) for s in ['_meta.json','_rows.bin','_cols.bin','_values.bin','_rhs.bin']]
        rows.append({'frame':frame,'direction':1,'dofs':meta['dofs'],'cpu_reference':ref,
            'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inp},
            'preconditioner_export_complete':study['system']['preconditioner_export_complete'],
            'system_unchanged':study['system_unchanged'],'primary_restored_bitwise':study['primary_restored_bitwise'],
            'operator_repeats':study['operator_repeats'],'replayed_solves':study['runs'],'cpu_solution_checks':solutions})
    report={'schema':1,'systems':rows,'passed':all(r['cpu_reference']['passed'] and r['system_unchanged'] and
        r['primary_restored_bitwise'] for r in rows),
        'scope':'CPU accurate linear reference and protected fixed-operator repeatability, not trajectory/physical certification.',
        'conclusion':'Legacy preconditioner action varies on the same serialized A/b/M. This directly identifies a numerical repeatability mechanism; it does not prove it explains all long-window divergence.',
        'production_rho_tolerance_changed':False,'default_rho_is_true_residual_threshold':False}
    write(ROOT/'reports/active/ipc_observer_fixed_probe_20261005_analysis.json',report)
    print(json.dumps({'passed':report['passed'],'systems':[{'frame':r['frame'],'cpu_reference':r['cpu_reference'],
        'cpu_solution_checks':r['cpu_solution_checks']} for r in rows]}))

if __name__=='__main__':main()
