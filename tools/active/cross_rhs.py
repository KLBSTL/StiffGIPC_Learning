"""Bounded CPU same-A/M cross-RHS test: separates operator from load difficulty."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
from scipy import sparse
from config import ROOT,read,sha
sys.path.insert(0,str(ROOT/'tools'))
from analyze_perf_v36 import matrix_from_snapshot

def operator(prefix):
    meta,A,b=matrix_from_snapshot(prefix)
    locals=[]
    def raw(s,dtype='<f8'):return np.fromfile(str(prefix)+s,dtype)
    for m in meta['local_preconditioners']:
        offset=3*m['offset']
        if m['kind']=='abd':
            inv=raw('_'+m['buffer']+'.bin').reshape(-1,12,12).transpose(0,2,1)
            locals.append(('abd',offset,inv));continue
        assert m['kind']=='MAS_full_owned_buffers' and m['cholesky']
        nodes,mapped,levels,_,clusters,*_=m['dimensions']
        starts=raw('_mas_d_restriction_starts.bin','<i4');ids=raw('_mas_d_restriction_nodes.bin','<i4')
        R=sparse.csr_matrix((np.ones(len(ids)),ids,starts),shape=(clusters,nodes))
        L=raw('_mas_d_cholesky.bin').reshape(-1,48,48)
        # Production prolongation order: fine row, then increasing hierarchy.
        real=raw('_mas_d_real_map_partId.bin','<i4')[:nodes]
        coarse=raw('_mas_d_coarseTable.bin','<i4').reshape(-1,6)[:nodes,:levels-1]
        locals.append(('mas',offset,(nodes,clusters,R,L,real,coarse)))
    def apply(v):
        result=v.copy()
        for kind,offset,data in locals:
            if kind=='abd':
                n=len(data)*12;result[offset:offset+n]=np.einsum('nij,nj->ni',data,v[offset:offset+n].reshape(-1,12)).ravel()
            else:
                nodes,clusters,R,L,real,coarse=data
                w=(R@v[offset:offset+nodes*3].reshape(nodes,3)).reshape(-1,48)
                for j in range(48):
                    w[:,j]/=L[:,j,j];w[:,j+1:]-=L[:,j+1:,j]*w[:,j,None]
                for j in range(47,-1,-1):
                    w[:,j]/=L[:,j,j];w[:,:j]-=L[:,j,:j]*w[:,j,None]
                z=w.reshape(clusters,3);out=z[real].copy()
                for level in range(coarse.shape[1]):out+=z[coarse[:,level]]
                result[offset:offset+nodes*3]=out.ravel()
        return result
    return A,b,apply

def pcg(A,b,M):
    start=time.monotonic();x=np.zeros_like(b);r=b.copy();z=M(r);p=z.copy();rho=float(r@z);initial=rho;ledger=[]
    if rho<=0:raise ValueError('Nonpositive initial rho')
    for i in range(1,1001):
        q=A@p;curvature=float(p@q)
        if curvature<=0 or not np.isfinite(curvature):raise ValueError('Nonpositive curvature')
        alpha=rho/curvature;x+=alpha*p;r-=alpha*q;z=M(r);newrho=float(r@z)
        ledger.append({'iteration':i,'rho_relative':newrho/initial,'curvature':curvature})
        if newrho<0 or not np.isfinite(newrho):raise ValueError('Bad rho')
        if newrho<=initial*1e-4:break
        if time.monotonic()-start>90:break
        p=z+(newrho/rho)*p;rho=newrho
    return {'iterations':i,'rho_relative':newrho/initial,'rho_converged':newrho<=initial*1e-4,
            'true_relative_residual':float(np.linalg.norm(b-A@x)/np.linalg.norm(b)),
            'wall_seconds':time.monotonic()-start,'ledger':ledger}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prefixes',nargs=2);p.add_argument('--output',required=True);a=p.parse_args()
    target=ROOT/a.output
    if target.exists():raise FileExistsError(target)
    systems=[operator(ROOT/v) for v in a.prefixes]
    report={'scope':'CPU diagnostic, same fixed A/M with swapped RHS; not performance or physics acceptance',
            'rho_tol':1e-4,'max_iterations':1000,'max_seconds_per_case':90,'script_sha256':sha(__file__),'runs':[]}
    for i,(A,b,M) in enumerate(systems):
        for j,(_,rhs,_) in enumerate(systems):
            result=pcg(A,rhs,M);row={'operator':a.prefixes[i],'rhs':a.prefixes[j],**result}
            report['runs'].append(row);target.write_text(json.dumps(report,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!='ledger'}),flush=True)
