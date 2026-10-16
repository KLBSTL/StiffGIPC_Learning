import unittest
import numpy as np
from scipy.sparse import csr_matrix
from m3_numeric import Balanced,pcg,orthogonal,ritz_basis,pod_basis,allow_transfer,residual_metrics,harmonic_abd
from unittest.mock import patch


class M3Algebra(unittest.TestCase):
    def system(self,n=24):
        rng=np.random.default_rng(29);q,_=np.linalg.qr(rng.normal(size=(n,n)))
        a=(q*np.geomspace(1.,1e4,n))@q.T;b=rng.normal(size=n)
        return a,b
    def test_balanced_matches_dense_symmetric_positive_operator(self):
        a,b=self.system();rng=np.random.default_rng(91);z=rng.normal(size=(24,8));d=np.diag(np.linspace(.1,2.,24))
        op=Balanced(csr_matrix(a),lambda v:d@v,z);Z=op.Z;Q=Z@np.linalg.solve(Z.T@a@Z,Z.T);P=np.eye(24)-a@Q
        expected=Q+P.T@d@P;actual=np.column_stack([op(v) for v in np.eye(24)])
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(actual).min(),0)
        np.testing.assert_allclose(actual,actual.T,rtol=1e-10,atol=1e-10)
    def test_full_space_recovers_exact_inverse_and_common_residual(self):
        a,b=self.system(12);A=csr_matrix(a);op=Balanced(A,lambda v:v,np.eye(12))
        result,x,_=pcg(A,b,op,lambda v:v,tol=1e-9)
        self.assertTrue(result['converged']);self.assertLessEqual(result['iterations'],2)
        self.assertLess(np.linalg.norm(b-a@x)/np.linalg.norm(b),1e-9)
    def test_dependent_columns_do_not_create_new_physical_modes(self):
        z=np.array([[1.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
        q,info=orthogonal(z);self.assertEqual(info['rank'],2)
        np.testing.assert_allclose(q[2:],0,atol=1e-14);np.testing.assert_allclose(q.T@q,np.eye(2),atol=1e-13)
    def test_current_system_is_forbidden_as_its_own_online_source(self):
        s=dict(frame=14,inner=0,linear_system_id=80)
        allow_transfer(s,dict(frame=14,inner=1,linear_system_id=81))
        for bad in (s,dict(frame=15,inner=1,linear_system_id=81),dict(frame=14,inner=2,linear_system_id=82)):
            with self.assertRaises(ValueError):allow_transfer(s,bad)
    def test_ritz_and_pod_are_valid_subspaces_of_recorded_source_history(self):
        a,b=self.system();A=csr_matrix(a);r,x,history=pcg(A,b,lambda v:v,lambda v:v,tol=1e-8,collect=True)
        self.assertTrue(r['converged']);z,info=ritz_basis(A,history);pod,pinfo=pod_basis(A,history,x)
        self.assertGreater(info['rank'],0);self.assertGreater(pinfo['rank'],0)
        self.assertTrue(np.isfinite(z).all() and np.isfinite(pod).all())
    def test_explicit_residual_uses_common_reference_not_candidate(self):
        A=csr_matrix(np.diag([1.,100.]));b=np.ones(2);x=np.array([1.,0.])
        m,_=residual_metrics(A,b,x,lambda v:v)
        self.assertAlmostEqual(m['relative_l2'],2**-.5);self.assertAlmostEqual(m['common_MAS_relative'],2**-.5)
        with self.assertRaises(ValueError):pcg(A,b,lambda v:-v,lambda v:v)
    def test_nonsymmetric_roundoff_is_not_silently_removed_from_A(self):
        a,b=self.system(12);a[0,1]+=1e-12;before=a.copy();A=csr_matrix(a)
        op=Balanced(A,lambda v:v,np.eye(12)[:,:3]);op(b)
        np.testing.assert_array_equal(A.toarray(),before)
    def test_zero_coupling_avoids_any_soft_solve(self):
        class System:pass
        s=System();s.A=csr_matrix(np.eye(15));s.b=np.ones(15);s.abd=np.arange(15)<12
        with patch('m3_numeric.cg',side_effect=AssertionError('Unexpected solve')):
            z,info=harmonic_abd(s)
        np.testing.assert_array_equal(z[:12],np.eye(12));np.testing.assert_array_equal(z[12:],0)
        self.assertEqual(info['extension_solver'],'zero_coupling_exact')
    def test_nonzero_harmonic_columns_are_verified_against_original_block(self):
        class System:pass
        s=System();rng=np.random.default_rng(12);c=.01*rng.normal(size=(3,12));a=np.eye(15)
        a[12:,:12]=c;a[:12,12:]=c.T;s.A=csr_matrix(a);s.b=np.ones(15)
        s.abd=np.arange(15)<12;s.apply=lambda r:r
        z,info=harmonic_abd(s)
        np.testing.assert_allclose(a[12:]@z,0,atol=1e-12)
        self.assertLess(info['extension_relative_residual'],1e-7)


if __name__=='__main__':unittest.main()
