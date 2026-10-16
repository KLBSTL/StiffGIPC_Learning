"""CPU diagnostic algebra. Original A is retained; no production solver changes."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
import json
import math
import sys
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve,eigh
from scipy.sparse.linalg import cg,LinearOperator
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from cpu_fixed_reference import load_snapshot,fnv1a64
from audit_recycling_reference import preconditioner
from local_identity import record


def need(ok,message):
    if not ok:raise ValueError(message)


def rel(value,reference):
    return float(np.linalg.norm(value-reference)/max(np.linalg.norm(reference),1e-300))


class FrozenSystem:
    def __init__(self,prefix):
        self.prefix=Path(prefix);self.meta=json.loads(Path(str(prefix)+'_meta.json').read_text())
        need(self.meta['preconditioner_export_complete'],'Incomplete MAS export')
        self.full_A,self.full_b,self.active,_,self.evidence,self.details=load_snapshot(prefix)
        self.original_action,self.extra,self.original_ops=preconditioner(prefix,self.meta,True)
        self.A=self.full_A[self.active][:,self.active].tocsr();self.b=self.full_b[self.active].copy()
        self.soft=np.zeros(len(self.full_b),bool);self.abd=np.zeros(len(self.full_b),bool)
        self.ops=[];self.local_checks=[];self.patches=[];coverage=np.zeros(len(self.full_b),int)
        for kind,offset,data in self.original_ops:
            dense=data if kind=='abd' else data[2].astype(np.float64)
            symmetric=.5*(dense+dense.transpose(0,2,1))
            eigen=np.linalg.eigvalsh(symmetric)
            need(np.isfinite(eigen).all() and np.all(eigen[:,0]>0),'Stored local inverse is not numerically SPD')
            np.linalg.cholesky(symmetric) # separate factorization check; no clipping
            self.local_checks.append(dict(kind=kind,blocks=len(dense),minimum_eigenvalue=float(eigen.min()),
                minimum_relative_eigenvalue=float(np.min(eigen[:,0]/eigen[:,-1])),
                explicit_symmetrization_relative=rel(symmetric,dense),negative_eigenvalues_clipped=False))
            if kind=='abd':
                size=12*len(dense);self.abd[offset:offset+size]=True
                self.ops.append((kind,offset,symmetric))
            else:
                nodes,restriction,_,fine,coarse=data;size=3*nodes
                fine_rows=restriction[fine].tocsr()
                need(np.all(np.diff(fine_rows.indptr)==1) and np.array_equal(fine_rows.indices,np.arange(nodes))
                    and np.all(fine_rows.data==1),'Fine prolongation/restriction is not an injective identity mapping')
                self.soft[offset:offset+size]=True
                self.ops.append((kind,offset,(nodes,restriction.astype(np.float64),symmetric)))
                part=np.fromfile(str(prefix)+'_mas_d_partId_map_real.bin',dtype='<i4')
                for i in range(0,len(part),16):
                    ids=part[i:i+16];ids=ids[ids>=0]
                    if len(ids):self.patches.append((offset,ids.copy()))
            need(offset>=0 and offset+size<=len(coverage),'Local preconditioner range invalid')
            coverage[offset:offset+size]+=1
        need(np.max(coverage,initial=0)<=1,'Overlapping overwrite ranges need explicit native ordering')
        need('diag_inverse' not in self.meta['buffers'],'Global diagonal needs a separate calibrated reconstruction')
        self.soft=self.soft[self.active];self.abd=self.abd[self.active]
        self.inverse_active=np.full(len(self.full_b),-1,int);self.inverse_active[self.active]=np.arange(len(self.b))
        asym=self.A-self.A.T
        self.details['relative_frobenius_asymmetry']=float(np.linalg.norm(asym.data)/max(np.linalg.norm(self.A.data),1e-300))
        need(self.details['relative_frobenius_asymmetry']<=1e-12,'A is not numerically symmetric; no silent repair')
        self.details['A_symmetrized']=False
        for array in (self.A.data,self.A.indices,self.A.indptr,self.b):array.flags.writeable=False

    def full_apply(self,vector):
        out=np.asarray(vector,dtype=np.float64).copy()
        for kind,offset,data in self.ops:
            if kind=='abd':
                size=12*len(data);out[offset:offset+size]=np.einsum('bij,bj->bi',data,vector[offset:offset+size].reshape(-1,12)).ravel()
            else:
                nodes,restriction,dense=data
                restricted=restriction@vector[offset:offset+3*nodes].reshape(nodes,3)
                local=np.einsum('bij,bj->bi',dense,restricted.reshape(-1,48)).reshape(-1,3)
                out[offset:offset+3*nodes]=(restriction.T@local).ravel()
        need(np.isfinite(out).all(),'Nonfinite linear reference action');return out

    def apply(self,vector):
        full=np.zeros(len(self.full_b));full[self.active]=vector
        return self.full_apply(full)[self.active]

    def local_basis(self,rank):
        choices=[]
        for index,(offset,ids) in enumerate(self.patches):
            full=(offset+3*ids[:,None]+np.arange(3)).ravel();active=self.inverse_active[full];active=active[active>=0]
            choices.append((-float(self.b[active]@self.b[active]),index,offset,ids))
        vectors=[]
        for _,_,offset,ids in sorted(choices,key=lambda v:(v[0],v[1])):
            for axis in range(3):
                indices=self.inverse_active[offset+3*ids+axis];indices=indices[indices>=0]
                if not len(indices):continue
                v=np.zeros(len(self.b));v[indices]=1/math.sqrt(len(indices));vectors.append(v)
                if len(vectors)==rank:return np.column_stack(vectors)
        return np.column_stack(vectors) if vectors else np.empty((len(self.b),0))


def residual_metrics(A,b,x,Bref,normalizers=None):
    r=b-A@x;rz=float(r@Bref(r))
    if normalizers is None:normalizers=(float(np.linalg.norm(b)),float(b@Bref(b)))
    nb,bb=normalizers;need(rz>=0 and bb>0 and np.isfinite(rz),'Invalid common MAS quadratic')
    return dict(relative_l2=float(np.linalg.norm(r)/max(nb,1e-300)),common_MAS_relative=math.sqrt(rz/bb)),r


def pcg(A,b,apply,Bref,tol=.01,maxiter=2000,collect=False):
    started=time.perf_counter();x=np.zeros_like(b);norms=(float(np.linalg.norm(b)),float(b@Bref(b)))
    if not norms[0]:return dict(converged=True,iterations=0,relative_l2=0.,common_MAS_relative=0.,seconds=0.,trace=[]),x,[]
    need(norms[1]>0,'Invalid fixed reference norm');r=b.copy();z=apply(r);rho=float(r@z);p=z.copy()
    trace=[];samples=[];rng=np.random.default_rng(20261015);seen=0;restarts=0
    for iteration in range(1,maxiter+1):
        need(rho>0 and np.isfinite(rho),'PCG nonpositive preconditioned curvature')
        if collect:
            sample=(iteration-1,r.copy()/math.sqrt(rho),z.copy()/math.sqrt(rho),x.copy())
            # Keep the first 16, then an unbiased deterministic reservoir of 112.
            if len(samples)<128:samples.append(sample)
            else:
                j=int(rng.integers(0,seen-16+1))
                if j<112:samples[16+j]=sample
            seen+=1
        ap=A@p;curvature=float(p@ap);need(curvature>0 and np.isfinite(curvature),'PCG nonpositive A curvature')
        alpha=rho/curvature;x+=alpha*p;r-=alpha*ap
        metrics,explicit=residual_metrics(A,b,x,Bref,norms)
        trace.append(dict(iteration=iteration,**metrics,rho=rho,p_Ap=curvature,alpha=alpha))
        if max(metrics.values())<=tol:
            return dict(converged=True,iterations=iteration,**metrics,seconds=time.perf_counter()-started,
                residual_restarts=restarts,validation_spmv=iteration,trace=trace),x,sorted(samples,key=lambda v:v[0])
        if np.linalg.norm(r)<=tol*norms[0] and np.linalg.norm(explicit)>tol*norms[0]:
            r=explicit;z=apply(r);rho=float(r@z);p=z.copy();restarts+=1;continue
        newz=apply(r);newrho=float(r@newz);need(newrho>0 and np.isfinite(newrho),'PCG invalid new rho')
        p=newz+(newrho/rho)*p;z=newz;rho=newrho
    return dict(converged=False,iterations=maxiter,**metrics,seconds=time.perf_counter()-started,
        residual_restarts=restarts,validation_spmv=maxiter,trace=trace),x,sorted(samples,key=lambda v:v[0])


def orthogonal(vectors,maximum=32):
    need(np.isfinite(vectors).all(),'Nonfinite basis');accepted=[]
    # Preserve slow-mode priority; never invent a QR completion for an early
    # dependent column. Two reorthogonalizations control numerical dependence.
    for column in vectors.T:
        norm=np.linalg.norm(column)
        if not norm:continue
        v=column/norm
        if accepted:
            q=np.column_stack(accepted)
            for _ in range(2):v-=q@(q.T@v)
        norm=np.linalg.norm(v)
        if norm>1e-8:accepted.append(v/norm)
        if len(accepted)==maximum:break
    q=np.column_stack(accepted) if accepted else np.empty((vectors.shape[0],0))
    return q,dict(input_rank=int(vectors.shape[1]),rank=len(accepted))


class Balanced:
    def __init__(self,A,B0,vectors):
        started=time.perf_counter();self.Z,self.rank_info=orthogonal(vectors,32);self.B0=B0
        self.W=A@self.Z;E=self.Z.T@self.W;self.gram_asymmetry=rel(E,E.T) if E.size else 0.
        need(self.gram_asymmetry<1e-8,'Coarse matrix is not numerically symmetric')
        self.E=.5*(E+E.T);self.factor=cho_factor(self.E,lower=True) if len(E) else None
        self.setup_seconds=time.perf_counter()-started
    def __call__(self,r):
        if self.factor is None:return self.B0(r)
        a=cho_solve(self.factor,self.Z.T@r);u=self.B0(r-self.W@a)
        # Transpose on W is intentional: Q + P^T B0 P stays symmetric even
        # when native A has tiny diagonal-block roundoff asymmetry.
        return u+self.Z@(a-cho_solve(self.factor,self.W.T@u))


def ritz_basis(A,samples):
    started=time.perf_counter()
    if not samples:return np.empty((A.shape[0],0)),dict(rank=0,seconds=0.)
    R=np.column_stack([s[1] for s in samples]);Z=np.column_stack([s[2] for s in samples])
    G=R.T@Z;G=.5*(G+G.T);ge,gu=eigh(G);keep=ge>1e-10*max(float(ge[-1]),1e-300)
    need(np.any(keep),'No independent sampled Krylov vectors')
    V=(Z@gu[:,keep])/np.sqrt(ge[keep]);H=V.T@(A@V);he,hu=eigh(.5*(H+H.T))
    need(he[0]>0,'Nonpositive sampled preconditioned Ritz value')
    basis=V@hu[:,:min(32,len(he))]
    return basis,dict(rank=basis.shape[1],sampled_dimension=len(samples),retained_metric_rank=int(keep.sum()),
        lowest_ritz=float(he[0]),largest_sampled_ritz=float(he[-1]),seconds=time.perf_counter()-started,
        scope='Sampled current-solve Ritz estimates, not a global spectral optimum or condition certificate')


def pod_basis(A,samples,solution):
    started=time.perf_counter()
    if not samples:return np.empty((len(solution),0)),dict(rank=0,seconds=0.)
    S=np.column_stack([solution-s[3] for s in samples]);G=S.T@(A@S);ev,u=eigh(.5*(G+G.T))
    order=np.flatnonzero(ev>1e-10*max(float(ev[-1]),1e-300))[::-1][:32]
    return S@u[:,order],dict(rank=len(order),seconds=time.perf_counter()-started,
        scope='Source-solve A-energy POD of final approximate solution minus sampled iterates')


def harmonic_abd(system):
    if not system.abd.any():return np.empty((len(system.b),0)),dict(applicable=False,reason='No retained ABD DOFs')
    started=time.perf_counter();ai=np.flatnonzero(system.abd);fi=np.flatnonzero(~system.abd)
    need(len(ai)%12==0,'ABD layout not in complete 12-DOF bodies')
    bodies=len(ai)//12;V=np.tile(np.eye(12),(bodies,1))/math.sqrt(bodies)
    Z=np.zeros((len(system.b),12));Z[ai]=V
    iterations=[];maximum_column_residual=0.;kind='zero_coupling_exact'
    if len(fi):
        C=system.A[fi][:,ai];rhs=-(C@V)
        # Fixed ABD often has exactly zero coupling. Do not factor a large
        # soft block merely to solve its zero RHS.
        if np.any(rhs):
            kind='bounded_PCG_extension';F=system.A[fi][:,fi].tocsr()
            def apply_soft(r):
                full=np.zeros(len(system.b));full[fi]=r;return system.apply(full)[fi]
            M=LinearOperator(F.shape,matvec=apply_soft,dtype=np.float64)
            for column in range(12):
                if not np.any(rhs[:,column]):iterations.append(0);continue
                count=[0];column_start=time.perf_counter()
                def callback(_):
                    count[0]+=1
                    if time.perf_counter()-column_start>30 or time.perf_counter()-started>90:
                        raise TimeoutError('Bounded harmonic construction budget reached')
                solution,info=cg(F,rhs[:,column],M=M,rtol=1e-9,atol=0.,maxiter=1600,callback=callback)
                error=rel(F@solution,rhs[:,column]);maximum_column_residual=max(maximum_column_residual,error)
                need(info==0 and error<1e-7,'Harmonic extension failed explicit per-column residual')
                Z[fi,column]=solution;iterations.append(count[0])
    return Z,dict(applicable=True,bodies=bodies,rank=12,seconds=time.perf_counter()-started,
        extension_relative_residual=maximum_column_residual,extension_solver=kind,extension_iterations=iterations,
        kind='single_ABD_harmonic12' if bodies==1 else 'collective_ABD_affine12',
        scope='Current soft-block harmonic equation; exact zero shortcut or bounded iterative residual <=1e-7; not a GPU implementation')


def allow_transfer(source,target):
    need(source['frame']==target['frame'] and source['linear_system_id']+1==target['linear_system_id']
        and source['inner']+1==target['inner'],'Transfer requires an immediately preceding real system')
