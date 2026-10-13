"""Screen unchanged captured IPC A,b,M; CPU counts are not native speed evidence."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
for folder in ('local', 'bench'):
    sys.path.insert(0, str(ROOT / 'tools' / folder))
from cpu_fixed_reference import load_snapshot
from audit_recycling_reference import preconditioner
from config import read
from linux_runner import require, write_new
from local_identity import record


def solve(matrix, rhs, apply, eta):
    start = time.monotonic()
    x = np.zeros_like(rhs)
    r = rhs.copy()
    z = apply(r)
    p = z.copy()
    rho = reference = float(r @ z)
    require(reference > 0 and np.isfinite(reference), 'Invalid initial rho')
    limit = 2000
    for iteration in range(1, limit):
        ap = matrix @ p
        curvature = float(p @ ap)
        require(curvature > 0 and np.isfinite(curvature), 'Nonpositive curvature')
        alpha = rho / curvature
        x += alpha * p
        r -= alpha * ap
        # Original host checks PREVIOUS rho after updating x/r.
        if abs(rho) <= eta * eta * reference:
            break
        z = apply(r)
        next_rho = float(r @ z)
        require(next_rho > 0 and np.isfinite(next_rho), 'Invalid residual rho')
        p = z + (next_rho / rho) * p
        rho = next_rho
    else:
        raise RuntimeError('PCG iteration cap reached')
    true = rhs - matrix @ x
    current_ratio = float(true @ apply(true)) / reference
    descent = float(rhs @ x)  # update is x_state - solution
    require(np.isfinite(x).all() and descent > 0, 'Invalid descent direction')
    return x, dict(eta=eta, rho_tolerance=eta*eta, iterations=iteration,
        previous_rho_ratio=rho/reference, true_common_rho_ratio=current_ratio,
        true_relative_residual=float(np.linalg.norm(true)/np.linalg.norm(rhs)),
        rhs_dot_solution=descent, CPU_seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    require(args.session.replace('_', '').isalnum(), 'Invalid session')
    out = ROOT / 'runs' / args.session
    require(not out.exists(), 'Fresh output required')
    out.mkdir(parents=True)
    (out/'controller_at_run.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    for case in ('fixed_ipc', 'mixed_ipc'):
        folder = ROOT/'runs/audit_v3_adjacent_systems'/case/'fixed'
        for meta_path in sorted(folder.glob('*_meta.json')):
            prefix = Path(str(meta_path)[:-10])
            matrix, rhs, active, zero, evidence, detail = load_snapshot(prefix)
            apply, operator_evidence = preconditioner(prefix, read(meta_path))
            require(np.all(rhs[~active] == 0), 'Zero rows have nonzero RHS')
            solutions = []
            cases = []
            for eta in (.01, .03, .1):
                solution, result = solve(matrix, rhs, apply, eta)
                if solutions:
                    result['relative_to_baseline_solution'] = float(
                        np.linalg.norm(solution-solutions[0])/np.linalg.norm(solutions[0]))
                else:
                    result['relative_to_baseline_solution'] = 0.0
                solutions.append(solution)
                cases.append(result)
            row = dict(case=case, prefix=str(prefix), system=detail, cases=cases,
                evidence=evidence, operator_evidence=operator_evidence,
                scope='Existing v3 saved coefficients/hierarchy; no state or quality inference.')
            rows.append(row)
            write_new(out/(case+'_'+prefix.name+'.json'), row)
            print(json.dumps(dict(case=case, prefix=prefix.name, cases=cases)), flush=True)
    write_new(out/'summary.json', dict(controller=record(__file__), cases=rows,
        native_performance_certified=False, physical_quality_certified=False))


if __name__ == '__main__':
    main()
