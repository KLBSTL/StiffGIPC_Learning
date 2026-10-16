import unittest

import numpy as np

from cgstats_numeric import statistical_gram,coefficients


class CGStatistics(unittest.TestCase):
    def test_exact_A_conjugate_increments_and_unsorted_bank(self):
        rng=np.random.default_rng(53);q,_=np.linalg.qr(rng.normal(size=(24,24)))
        diag=np.geomspace(.5,100.,24);A=(q*diag)@q.T
        D=q/np.sqrt(diag);D*=np.linspace(.2,2.,24)
        weights=np.einsum('ij,ij->j',D,A@D);ids=np.array([23,0,8,4,19])
        S=np.column_stack([D[:,i:].sum(axis=1) for i in ids])
        actual,_=statistical_gram(np.ones(24),weights,ids)
        np.testing.assert_allclose(actual,S.T@A@S,rtol=1e-12,atol=1e-12)

    def test_last_stored_pre_update_iterate_includes_last_step(self):
        g,_=statistical_gram([1.,2.,3.],[4.,5.,6.],np.array([2,0]))
        np.testing.assert_array_equal(g,[[18.,18.],[18.,32.]])

    def test_loss_of_conjugacy_is_not_an_exact_Gram_identity(self):
        D=np.array([[1.,1.],[0.,0.]]);ids=np.array([0,1])
        S=np.column_stack([D[:,i:].sum(axis=1) for i in ids])
        approximate,info=statistical_gram([1.,1.],[1.,1.],ids)
        self.assertTrue(info['approximate']);self.assertGreater(np.linalg.norm(approximate-S.T@S),1.)

    def test_invalid_statistics_and_restart_are_rejected(self):
        for alpha,rho,ids,kw in [([1.],[1.],[0],dict(restarts=1)),([0.],[1.],[0],{}),
            ([1.],[np.nan],[0],{}),([1.],[1.],[1],{}),([1.,1.],[1.,1.],[0,0],{}),
            ([1.],[1.],[0.],{}),([1.],[1.],[0],dict(curvature=[2.]))]:
            with self.assertRaises(ValueError):statistical_gram(alpha,rho,np.asarray(ids),**kw)

    def test_coefficients_keep_largest_energy_and_do_not_invent_rank(self):
        G=np.diag([0.,1.,4.]);u,info=coefficients(G,8)
        self.assertEqual(info['actual_rank'],2)
        np.testing.assert_array_equal(u.T@G@u,np.diag([4.,1.]))
        with self.assertRaises(ValueError):coefficients(np.diag([-1.,4.]),1)

    def test_formula_uses_alpha_times_rho_from_same_iteration(self):
        alpha=np.array([.1,.2,.3]);curvature=np.array([7.,5.,2.]);rho=alpha*curvature
        g,_=statistical_gram(alpha,rho,np.arange(3),curvature=curvature)
        expected=(alpha*alpha*curvature).sum();self.assertAlmostEqual(g[0,0],expected)


if __name__=='__main__':unittest.main()
