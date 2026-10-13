"""CPU surrogate: test previous-solve subspaces on adjacent captured systems.

Never use the current system's saved solution as its own initial guess. Legacy
float restriction/reduction order differs from CUDA, so this is an iteration
hypothesis screen, not equivalence, timing, or physical-quality certification.
"""
from pathlib import Path
import argparse
import json
import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/local'))
sys.path.insert(0, str(ROOT/'tools/bench'))
from cpu_fixed_reference import load_snapshot, fnv1a64
from config import read
from linux_runner import require, write_new
from local_identity import record


def preconditioner(prefix, meta, with_data=False):
    """Reconstruct the stored hierarchy and inverse; no refactorization of A."""
    operators = []
    evidence = []
    for local in meta['local_preconditioners']:
        offset = 3*local['offset']
        if local['kind'] == 'abd':
            path = Path(str(prefix)+'_'+local['buffer']+'.bin')
            raw = path.read_bytes()
            expected = meta['buffers'][local['buffer']]
            require(len(raw)==expected['bytes'] and fnv1a64(raw)==expected['fnv1a64'], 'ABD snapshot differs')
            inverse = np.frombuffer(raw, dtype='<f8').reshape(local['count'],12,12).transpose(0,2,1)
            evidence.append(record(path))
            operators.append(('abd', offset, inverse))
            continue
        require(local['kind']=='MAS_full_owned_buffers' and not local['cholesky']
            and not local['wide_apply'] and not local['inverse64'], 'Requires legacy float MAS')
        nodes, mapped, levels, _, clusters, *_ = local['dimensions']
        def buffer(name, dtype):
            path = Path(str(prefix)+'_mas_'+name+'.bin')
            raw = path.read_bytes()
            expected = local['buffers'][name]
            require(len(raw)==expected['count']*expected['item_bytes']
                and fnv1a64(raw)==expected['fnv1a64'], 'MAS snapshot differs: '+name)
            evidence.append(record(path))
            return np.frombuffer(raw, dtype=dtype)
        part = buffer('d_partId_map_real', '<i4')
        fine = buffer('d_real_map_partId', '<i4')
        coarse = buffer('d_coarseTable', '<i4').reshape(nodes,6)[:,:levels-1]
        packed = buffer('d_precondMatMas', '<f4').reshape(-1,136,3,3).transpose(0,1,3,2)
        require(len(part)==mapped and clusters%16==0 and len(packed)==clusters//16, 'MAS ABI differs')
        valid = part>=0
        require(np.all(part[valid]<nodes) and np.all((fine>=0)&(fine<mapped))
            and np.all((coarse>=0)&(coarse<clusters)), 'Hierarchy index outside domain')
        row = np.r_[np.flatnonzero(valid),coarse.ravel()]
        col = np.r_[part[valid],np.repeat(np.arange(nodes),levels-1)]
        restriction = coo_matrix((np.ones(len(row),np.float32),(row,col)),shape=(clusters,nodes)).tocsr()
        dense = np.zeros((len(packed),48,48),np.float32)
        for i in range(16):
            for j in range(i,16):
                tile = packed[:,16*i-i*(i+1)//2+j]
                dense[:,3*i:3*i+3,3*j:3*j+3] = tile
                if i!=j:
                    dense[:,3*j:3*j+3,3*i:3*i+3] = tile.transpose(0,2,1)
        operators.append(('mas',offset,(nodes,restriction,dense,fine,coarse)))
    def apply(vector):
        output = vector.copy()
        for kind,offset,data in operators:
            if kind=='abd':
                count = len(data)*12
                output[offset:offset+count] = np.einsum('bij,bj->bi',data,
                    vector[offset:offset+count].reshape(-1,12)).ravel()
            else:
                nodes,restriction,dense,fine,coarse = data
                restricted = restriction@vector[offset:offset+3*nodes].reshape(nodes,3).astype(np.float32)
                local = np.einsum('bij,bj->bi',dense,restricted.reshape(-1,48)).reshape(-1,3)
                result = local[fine].copy()
                for level in range(coarse.shape[1]):
                    result += local[coarse[:,level]]
                output[offset:offset+3*nodes] = result.ravel()
        require(np.isfinite(output).all(), 'Nonfinite surrogate preconditioner')
        return output
    return (apply,evidence,operators) if with_data else (apply,evidence)


def pcg(matrix,rhs,apply,initial):
    # Keep the original-b reference rho; retain the previous-rho extra update.
    reference = float(rhs@apply(rhs))
    x = initial.copy()
    residual = rhs-matrix@x
    z = apply(residual)
    rho = float(residual@z)
    if reference==0 and not np.any(rhs):
        return dict(iterations=0,true_relative_residual=0,rho_reference=0),x
    require(reference>0 and rho>=0, 'Nonpositive surrogate rho')
    if rho==0:
        require(not np.any(residual), 'Zero surrogate rho with nonzero residual')
        return dict(iterations=0,true_relative_residual=0,rho_reference=reference,rho_stop=0),x
    p = z.copy()
    for iteration in range(1,501):
        ap = matrix@p
        curvature = float(p@ap)
        if rho==0:
            break
        require(curvature>0 and np.isfinite(curvature), 'Nonpositive curvature')
        alpha = rho/curvature
        x += alpha*p
        residual -= alpha*ap
        if abs(rho)<=1e-4*reference:
            break
        z = apply(residual)
        new = float(residual@z)
        require(new>=0 and np.isfinite(new), 'Nonpositive surrogate rho')
        p = z+(new/rho)*p
        rho = new
    else:
        raise ValueError('Surrogate PCG cap')
    return dict(iterations=iteration,true_relative_residual=float(np.linalg.norm(rhs-matrix@x)/np.linalg.norm(rhs)),
        rho_reference=reference,rho_stop=rho),x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = dict(scope=__doc__,cases=[],controller=record(__file__),performance_certified=False,quality_certified=False)
    for folder in sorted(args.capture.iterdir()):
        if not (folder/'fixed').is_dir():
            continue
        prior = []
        for metadata in sorted((folder/'fixed').glob('*_meta.json')):
            prefix = Path(str(metadata)[:-len('_meta.json')])
            matrix,rhs,_,_,evidence,details = load_snapshot(prefix)
            meta = read(metadata)
            study = read(Path(str(prefix)+'_study.json'))
            require(study['system_unchanged'] and study['primary_restored_bitwise'], 'Unprotected captured system')
            apply,extra = preconditioner(prefix,meta)
            zero,_ = pcg(matrix,rhs,apply,np.zeros_like(rhs))
            initial = np.zeros_like(rhs)
            rank = 0
            if prior:
                vectors = np.column_stack(prior[-2:])
                vectors /= np.linalg.norm(vectors,axis=0)
                av = matrix@vectors
                gram = vectors.T@av
                gram = .5*(gram+gram.T)
                eigen = np.linalg.eigvalsh(gram)
                # Reject nearly dependent/indefinite small projections outright.
                if eigen[0]>1e-10*eigen[-1] and eigen[-1]>0:
                    initial = vectors@np.linalg.solve(gram,vectors.T@rhs)
                    rank = len(eigen)
            augmented,_ = pcg(matrix,rhs,apply,initial)
            native = [r for r in study['runs'] if r['phase']=='rho_tolerance'
                and r['rho_tolerance']==1e-4 and r['mode']=='graph']
            report['cases'].append(dict(case=folder.name,prefix=str(prefix),source=evidence,
                preconditioner_inputs=extra,matrix=details,zero=zero,recycled=augmented,projection_rank=rank,
                additional_projection_spmv=rank,additional_projection_dots=rank*(rank+3)//2,
                native_default_iterations=[r['iterations'] for r in native]))
            # Only solutions of earlier real systems enter later subspaces.
            solution_path = Path(str(prefix)+'_p1_m1_x.bin')
            prior.append(np.fromfile(solution_path,dtype='<f8'))
            require(prior[-1].shape==rhs.shape, 'Previous global DOF layout differs')
    write_new(args.output,report)
    print(json.dumps([{k:r[k] for k in ('case','prefix','zero','recycled','projection_rank','native_default_iterations')}
        for r in report['cases']]))


if __name__=='__main__':
    main()
