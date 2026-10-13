"""Measure the 12 affine/FEM coupling modes of a saved coefficient system.

This uses a symmetrized FP64 CPU coefficient matrix and exact-block sparse LU,
not the GPU MAS operator. It is a numerical coupling diagnostic, not a full
preconditioned condition-number, performance, or physical-quality certificate.
"""
from pathlib import Path
import argparse
import os
import sys
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import psutil
ROOT=Path(__file__).resolve().parents[2]
for folder in ('local','bench'):sys.path.insert(0,str(ROOT/'tools'/folder))
from cpu_fixed_reference import load_snapshot
from local_identity import record
from linux_runner import require,write_new
from config import read
from audit_contact_reference import cg
from audit_recycling_reference import preconditioner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--session',required=True)
    args=p.parse_args();require(args.session.replace('_','').isalnum(),'Invalid session')
    out=ROOT/'runs'/args.session;require(not out.exists(),'Fresh output required');out.mkdir(parents=True)
    (out/'controller_at_run.py').write_bytes(Path(__file__).read_bytes())
    require(psutil.virtual_memory().available>3*1024**3,'CPU memory reserve insufficient')
    start=time.monotonic();matrix,rhs,active,zero,evidence,detail=load_snapshot(args.prefix)
    require(active.all(),'Need explicit zero-row elimination')
    symmetric=(matrix+matrix.T)*.5
    abd=symmetric[:12,:12].toarray();cross=symmetric[:12,12:];fem=symmetric[12:,12:]
    scaling=1/np.sqrt(fem.diagonal());require(np.isfinite(scaling).all(),'Invalid FEM diagonal')
    scale=diags(scaling);normalized=(scale@fem@scale).tocsc()
    factor=splu(normalized,permc_spec='MMD_AT_PLUS_A')
    target=cross.T.toarray();solution=scaling[:,None]*factor.solve(scaling[:,None]*target)
    residual=fem@solution-target
    denominator=np.linalg.norm(target,axis=0)
    relative=np.linalg.norm(residual,axis=0)/np.maximum(denominator,1e-30)
    require(relative.max()<1e-7,'Inaccurate FEM block solve; coupling spectrum not accepted')
    response=cross@solution
    eta=eigh((response+response.T)*.5,abd,eigvals_only=True)
    require(eta.min()>-1e-8 and eta.max()<1,'Coupling outside SPD interval')
    sigma=float(np.sqrt(max(eta.max(),0)))
    meta=read(Path(str(args.prefix)+'_meta.json'))
    native,operator_evidence=preconditioner(args.prefix,meta)
    inv_abd=np.linalg.inv(abd)
    def exact_block(vector):
        result=np.empty_like(vector);result[:12]=inv_abd@vector[:12]
        result[12:]=scaling*factor.solve(scaling*vector[12:])
        return result
    ideal=np.vstack((np.eye(12),-solution))
    approximate=np.empty_like(ideal);approximate[:12]=np.eye(12)
    for col in range(12):
        vector=np.r_[np.zeros(12),target[:,col]]
        approximate[12:,col]=-native(vector)[12:]
    studies=[]
    definitions=[('native_CPU_MAS',native),('ideal_exact_block',exact_block)]
    coarse=[]
    for label,basis in (('ideal_harmonic',ideal),('MAS_harmonic',approximate)):
        action=matrix@basis;gram=basis.T@action;gram=(gram+gram.T)*.5
        values=np.linalg.eigvalsh(gram);require(values.min()>0,'Nonpositive coarse matrix')
        inverse=np.linalg.inv(gram)
        def additive(vector,u=basis,inv=inverse):
            return native(vector)+u@(inv@(u.T@vector))
        def balanced(vector,u=basis,inv=inverse,au=action):
            coefficients=inv@(u.T@vector)
            z=native(vector-au@coefficients)
            return z+u@(coefficients-inv@(au.T@z))
        definitions.extend([(label+'_additive',additive),(label+'_balanced',balanced)])
        coarse.append(dict(name=label,rank=12,gram_eigenvalues=values.tolist()))
    for label,apply in definitions:
        result=cg(matrix,rhs,apply,stop_apply=native)
        result['name']=label;studies.append(result)
        print({k:v for k,v in result.items() if k!='history'},flush=True)
    report=dict(controller=record(__file__),evidence=evidence,system=detail,scope=__doc__,
        coupling_eta=eta.tolist(),max_coupling_sigma=sigma,
        ideal_exact_block_preconditioned_condition_estimate=(1+sigma)/(1-sigma),
        fem_block_solve_relative_residuals=relative.tolist(),
        symmetrization_absolute_max=detail['max_absolute_asymmetry']/2,
        LU_nonzeros=int(factor.L.nnz+factor.U.nnz),
        PCG_studies=studies,coarse=coarse,operator_evidence=operator_evidence,
        stopping_norm='Original captured MAS r^T M r, common original-b reference, previous-rho timing',
        setup_warning='Exact FEM factorization is diagnostic only; no native speed inference. Approximate harmonic basis requires 12 M applications, 12 SpMV, coarse products and factorization per new system.',
        elapsed_seconds=time.monotonic()-start,process_rss_bytes=psutil.Process().memory_info().rss,
        quality_certified=False,performance_certified=False)
    write_new(out/'analysis.json',report)
    print({k:v for k,v in report.items() if k not in
        ('evidence','controller','scope','system','PCG_studies','operator_evidence','coarse')},flush=True)


if __name__=='__main__':main()
