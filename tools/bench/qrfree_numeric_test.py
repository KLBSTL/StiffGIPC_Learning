import unittest
import numpy as np
from scipy.sparse import csr_matrix
from m3_numeric import Balanced,rel
from qrfree_numeric import CoordinateBalanced,scaled_coefficients


class QRFreeTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(8816);h=rng.normal(size=(30,30))
        self.A=csr_matrix(h.T@h+np.eye(30));h=rng.normal(size=(30,30));self.B=h.T@h+np.eye(30)
        self.Z=rng.normal(size=(30,8));self.r=rng.normal(size=30)

    def test_invertible_change_of_coordinates(self):
        rng=np.random.default_rng(4);T=rng.normal(size=(8,8))+3*np.eye(8)
        a=CoordinateBalanced(self.A,lambda v:self.B@v,self.Z)
        b=CoordinateBalanced(self.A,lambda v:self.B@v,self.Z@T)
        self.assertLess(rel(a(self.r),b(self.r)),1e-11)
        self.assertLess(rel(a(self.r),Balanced(self.A,lambda v:self.B@v,self.Z)(self.r)),1e-11)

    def test_adjoint_formula_with_roundoff_asymmetry(self):
        A=self.A.toarray();A[0,1]+=1e-12
        a=CoordinateBalanced(csr_matrix(A),lambda v:self.B@v,self.Z)
        q=self.Z@np.linalg.solve(a.E,self.Z.T);p=np.eye(30)-A@q
        expected=q@self.r+p.T@self.B@p@self.r
        self.assertLess(rel(a(self.r),expected),1e-12)

    def test_rank_deficient_rejected(self):
        z=self.Z.copy();z[:,7]=z[:,0]
        with self.assertRaisesRegex(ValueError,'rank/conditioning'):
            CoordinateBalanced(self.A,lambda v:v,z)

    def test_ill_conditioned_rejected_without_jitter(self):
        z=self.Z.copy();z[:,7]*=1e-9
        with self.assertRaisesRegex(ValueError,'rank/conditioning'):
            CoordinateBalanced(self.A,lambda v:v,z)

    def test_energy_scaling_and_invalid_inputs(self):
        np.testing.assert_array_equal(scaled_coefficients(np.ones((3,2)),[1.,4.]),[[1.,.5]]*3)
        for ev in ([0.,1.],[-1.,1.],[np.nan,1.],[1.]):
            with self.assertRaises(ValueError):scaled_coefficients(np.ones((3,2)),ev)

    def test_nonfinite_and_nonsymmetric_rejected(self):
        z=self.Z.copy();z[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):CoordinateBalanced(self.A,lambda v:v,z)
        A=self.A.toarray();A[0,1]+=10
        with self.assertRaisesRegex(ValueError,'symmetric'):CoordinateBalanced(csr_matrix(A),lambda v:v,self.Z)


if __name__=='__main__':unittest.main()
