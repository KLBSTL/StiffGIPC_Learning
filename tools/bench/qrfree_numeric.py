"""Coordinate-only balanced action; no QR and no native solver equivalence claim."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve,eigvalsh
from m3_numeric import need,rel


class CoordinateBalanced:
    def __init__(self,A,B0,vectors,minimum_relative_eigenvalue=1e-12):
        self.Z=np.asarray(vectors,dtype=np.float64).copy();self.B0=B0
        need(self.Z.ndim==2 and self.Z.shape[0]==A.shape[0] and self.Z.shape[1]>0,
             'Invalid unorthogonalized basis shape')
        need(np.isfinite(self.Z).all(),'Nonfinite unorthogonalized basis')
        self.W=A@self.Z;raw=self.Z.T@self.W
        self.asymmetry=rel(raw,raw.T);need(self.asymmetry<1e-8,'Coarse matrix is not numerically symmetric')
        self.E=.5*(raw+raw.T);ev=eigvalsh(self.E)
        need(np.isfinite(ev).all() and ev[-1]>0 and ev[0]>minimum_relative_eigenvalue*ev[-1],
             'QR-free coarse rank/conditioning rejected')
        self.eigenvalues=ev;self.factor=cho_factor(self.E,lower=True,check_finite=True)

    def __call__(self,r):
        a=cho_solve(self.factor,self.Z.T@r)
        u=self.B0(r-self.W@a)
        return u+self.Z@(a-cho_solve(self.factor,self.W.T@u))


def scaled_coefficients(u,selected_eigenvalues):
    u=np.asarray(u,dtype=np.float64);ev=np.asarray(selected_eigenvalues,dtype=np.float64)
    need(u.ndim==2 and ev.shape==(u.shape[1],) and u.shape[1]>0,'Coefficient/eigenvalue shape mismatch')
    need(np.isfinite(u).all() and np.isfinite(ev).all() and np.all(ev>0),'Invalid energy scaling')
    return u/np.sqrt(ev)[None,:]
