import unittest

import numpy as np
from scipy.sparse import csr_matrix

from run_m3_arnoldi import spectral_space


class M3Arnoldi(unittest.TestCase):
    def test_general_preconditioned_operator_has_verified_low_modes(self):
        # B A is similar to SPD, but is not Euclidean symmetric. This catches
        # incorrectly replacing the general eigenproblem with symmetric eigh.
        rng = np.random.default_rng(63)
        q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
        a = (q * np.linspace(1., 30., 20)) @ q.T
        b = np.diag(np.linspace(.2, 3., 20))
        z, info = spectral_space(csr_matrix(a), lambda v: b @ v, k=3)
        self.assertEqual(info['status'], 'complete')
        self.assertEqual(info['qualified'], 3)
        expected = np.sort(np.linalg.eigvals(b @ a).real)[:3]
        actual = [r['value'] for r in info['eigenpairs']]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(z.T @ z, np.eye(3), atol=1e-12)
        self.assertTrue(all(r['relative_eigen_residual'] < 1e-4 for r in info['eigenpairs']))

    def test_spectral_budget_failure_does_not_invent_modes(self):
        z, info = spectral_space(csr_matrix(np.eye(20)), lambda v: v, seconds=-1, k=3)
        self.assertEqual(info['status'], 'timeout')
        self.assertEqual(z.shape, (20, 0))
        self.assertEqual(info['qualified'], 0)


if __name__ == '__main__':
    unittest.main()
