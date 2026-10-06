"""Independent CPU reference for saved frame30 systems; no new GPU trajectory."""
import json
import numpy as np
from scipy.sparse.linalg import splu
from run_v54_exit import ROOT,read,difference
from analyze_perf_v36 import matrix_from_snapshot
names=['v54_batch_sphere_seed_native_r1','v54_batch_sphere_resume_native_r1','v54_batch_sphere_resume_native_r2']
rows=[];solutions=[];defaults=[]
for name in names:
    p=ROOT/f'runs/local/{name}/fixed/f30_n1';meta,A,b=matrix_from_snapshot(p);A=A.tocsc();A.sum_duplicates()
    lu=splu(A);x=lu.solve(b)
    for _ in range(2):
        if np.linalg.norm(b-A@x)/np.linalg.norm(b)<=1e-10:break
        x+=lu.solve(b-A@x)
    residual=float(np.linalg.norm(b-A@x)/np.linalg.norm(b));assert np.isfinite(x).all() and residual<=1e-10
    solutions.append(x);comparisons=[]
    for phase,tol in [(1,1e-4),(3,1e-16)]:
        for mode in [0,1,2]:
            z=np.fromfile(f'{p}_p{phase}_m{mode}_x.bin',dtype='<f8')
            comparisons.append({'rho_tolerance':tol,'mode':['host','graph','graph_fused'][mode],
                'cpu_true_residual':float(np.linalg.norm(b-A@z)/np.linalg.norm(b)),
                'relative_error_to_cpu':float(np.linalg.norm(z-x)/np.linalg.norm(x)),
                'cloth_axis_velocity_m_s':float(np.max(np.abs(z[12:]))/.01)})
            if phase==1 and mode==1:defaults.append(z)
    # These scenes have one fixed ABD (12 dofs), all remaining DOFs are cloth.
    assert meta['dofs']==7815
    study=read(ROOT/f'runs/local/{name}/fixed/f30_n1_study.json')
    diag=np.fromfile(f'{p}_diag_inverse.bin',dtype='<f8')
    rows.append({'run':name,'dofs':meta['dofs'],'cpu_residual':residual,'cpu_cloth_axis_velocity_m_s':float(np.max(np.abs(x[12:]))/.01),
        'comparisons':comparisons,'operator_repeats':study['operator_repeats'],
        'default_graph_repeat_relative':[r['repeat_difference']['relative'] for r in study['runs'] if r['mode']=='graph' and r['rho_tolerance']==1e-4],
        'diag_norm':float(np.linalg.norm(diag))})
pairs=[]
for i in range(3):
    for j in range(i+1,3):pairs.append({'a':names[i],'b':names[j],
        'cpu_solution_difference':difference(solutions[i],solutions[j]),'default_graph_solution_difference':difference(defaults[i],defaults[j])})
out={'scope':'Saved first frame30 A/b only. CPU factorization accuracy verified by residual; not a nonlinear or physical reference.',
    'systems':rows,'pairs':pairs,'accurate_fixed_system_reference_passed':True,'physical_truth_certified':False}
target=ROOT/'reports/V54_BATCH_SPHERE_SYSTEM.json';assert not target.exists();target.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out,indent=2))
