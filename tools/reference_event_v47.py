"""One bounded CPU sparse-direct counterfactual on an exported event A/b.

Run this worker under a parent timeout. It does not step the simulation or change M.
"""
import argparse,json,time,warnings
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import spsolve,MatrixRankWarning
from analyze_perf_v36 import matrix_from_snapshot
p=argparse.ArgumentParser();p.add_argument('prefix',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();meta,A,b=matrix_from_snapshot(a.prefix);original=np.fromfile(str(a.prefix)+'_solution.bin',dtype='<f8')
event=json.loads(Path(str(a.prefix)+'_event.json').read_text());start=time.monotonic()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',MatrixRankWarning)
    ref=spsolve(A.tocsc(),b)
finite=bool(np.isfinite(ref).all());res=float(np.linalg.norm(b-A@ref)/max(np.linalg.norm(b),1e-300)) if finite else None
record={'method':'CPU SciPy sparse direct LU; projected exported A, not nonlinear physical truth',
    'seconds':time.monotonic()-start,'finite':finite,'relative_residual':res,'residual_gate':1e-8,
    'passed':finite and res<=1e-8,'warnings':[str(w.message) for w in caught]}
if record['passed']:
    physical=np.fromfile(str(a.prefix)+'_direction.bin',dtype='<f8').reshape(-1,3)
    run=a.prefix.parent.parent;abd=json.loads((run/'trace/metadata.json').read_text())['abd_point_num']
    offset=len(b)-3*(len(physical)-abd)
    original_fem=original[offset:].reshape(-1,3);ref_fem=ref[offset:].reshape(-1,3)
    assert np.allclose(original_fem,physical[abd:],rtol=1e-12,atol=1e-12)
    record.update(relative_solution_difference=float(np.linalg.norm(ref-original)/max(np.linalg.norm(ref),1e-300)),
        original_fem_axis_velocity=float(np.max(np.abs(original_fem))/event['dt']),
        reference_fem_axis_velocity=float(np.max(np.abs(ref_fem))/event['dt']),
        reference_negative_gradient_dot_step=float(-b@ref),reference_xAx=float(ref@(A@ref)),
        physical_trial_velocity=event['trial_velocity'])
    ref.tofile(a.output.with_suffix('.bin'))
a.output.write_text(json.dumps(record,indent=2,allow_nan=False));print(json.dumps(record))
