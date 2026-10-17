import unittest
import numpy as np
from scipy.sparse import csr_matrix
from m3_numeric import pcg,Balanced
from m3_lanczos import full_cg_basis


class FullCGOracle(unittest.TestCase):
    def test_nonidentity_preconditioned_recurrence_replays_and_recovers_small_space(self):
        rng=np.random.default_rng(55);q,_=np.linalg.qr(rng.normal(size=(12,12)))
        A=csr_matrix((q*np.geomspace(1,1000,12))@q.T);d=np.linspace(.1,2,12);b=rng.normal(size=12)
        B=lambda r:d*r;result,x,_=pcg(A,b,B,B,tol=1e-9)
        z,info=full_cg_basis(A,b,B,result['trace'],x)
        self.assertTrue(info['replay_solution_bitwise_equal']);self.assertEqual(z.shape[1],12)
        test,_,_=pcg(A,b,Balanced(A,B,z),B,tol=1e-8);self.assertLessEqual(test['iterations'],2)
    def test_wrong_saved_solution_or_recurrence_is_rejected(self):
        A=csr_matrix(np.diag([1.,2.,3.]));b=np.ones(3);B=lambda r:r
        result,x,_=pcg(A,b,B,B,tol=1e-10)
        with self.assertRaises(ValueError):full_cg_basis(A,b,B,result['trace'],x+1)
        bad=[dict(r) for r in result['trace']];bad[0]['rho']=-1
        with self.assertRaises(ValueError):full_cg_basis(A,b,B,bad,x)


if __name__=='__main__':unittest.main()
