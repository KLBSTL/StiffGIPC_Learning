"""Measure RHS excitation of verified spaces; not a condition-number test."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.linalg import cho_solve
from threadpoolctl import threadpool_info

from m3_numeric import ROOT, FrozenSystem, Balanced, need, residual_metrics
from local_identity import record, verify
from linux_runner import write_new
from m0_analyze import csv_write

REPORT = ROOT / 'reports/innovation_m3_20261015'
OUT = REPORT / 'rhs_excitation'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    need(not OUT.exists(), 'Fresh RHS diagnostic directory required')
    OUT.mkdir()
    rows = []
    for scene, frame in (('Bunny',14), ('Mixed',15), ('Boxes',17)):
        spectral = read(REPORT / 'arnoldi_control' / scene / 'result.json')
        verify([spectral['basis']])
        low = np.load(spectral['basis']['path'], allow_pickle=False)
        packet = read(REPORT / 'analysis_v2' / scene / f'adjacent_f{frame}_n1/source_packet.json')
        verify([packet['bases']])
        with np.load(packet['bases']['path'],allow_pickle=False) as history:
            pod = history['pod'].copy()
        for direction in (1,2):
            label = f'adjacent_f{frame}_n{direction}'
            data = read(REPORT / 'analysis_v2' / scene / label / 'result.json')
            system = FrozenSystem(ROOT / f'runs/innovation_m3_20261015/adjacent_{scene.lower()}/fixed/f{frame}_n{direction}')
            verify([data['reference']['solution']])
            xref = np.load(data['reference']['solution']['path'],allow_pickle=False)
            reference_energy = float(system.b @ xref)
            need(reference_energy > 0, 'Invalid positive reference energy')
            for family, vectors in (('Arnoldi_low_modes',low),('previous_common_POD',pod)):
                if family == 'previous_common_POD' and direction == 1:
                    continue # This is not a transferable source for itself.
                for rank in (8,16,32):
                    op = Balanced(system.A,system.apply,vectors[:,:rank])
                    q = op.Z @ cho_solve(op.factor,op.Z.T@system.b)
                    fraction = float(system.b@q)/reference_energy
                    need(-1e-8 <= fraction <= 1+1e-6, 'Invalid Galerkin energy fraction')
                    metrics, _ = residual_metrics(system.A,system.b,q,system.apply)
                    rows.append(dict(scene=scene,label=label,family=family,requested_rank=rank,
                        actual_rank=op.Z.shape[1],coarse_energy_fraction=fraction,
                        relative_l2_after_coarse=metrics['relative_l2'],
                        common_MAS_relative_after_coarse=metrics['common_MAS_relative'],
                        rhs_ABD_relative_l2=float(np.linalg.norm(system.b[system.abd])/np.linalg.norm(system.b)),
                        scope='Galerkin correction energy / high-accuracy reference energy; not forward error or spectral optimality'))
    csv_write(OUT/'spaces.csv',rows)
    write_new(OUT/'environment.json',dict(python=sys.version,python_binary=record(Path(sys.executable)),
        numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
        thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        threadpools=threadpool_info()))
    write_new(OUT/'validation.json',dict(passed=True,rows=len(rows),source=record(Path(__file__)),
        dependencies=[record(ROOT/'tools/bench/m3_numeric.py')],performance_certified=False))
    print(json.dumps([r for r in rows if r['requested_rank']==32]),flush=True)


if __name__ == '__main__':
    main()
