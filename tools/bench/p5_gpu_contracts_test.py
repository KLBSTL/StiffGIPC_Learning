import unittest
from unittest.mock import patch
import subprocess

import numpy as np
from scipy.sparse import csr_matrix

from m3_numeric import pod_basis, Balanced
from p5_validate import mixed_budget_source


class P5GPUContracts(unittest.TestCase):
    def test_A_energy_POD_agrees_with_independent_weighted_SVD(self):
        rng = np.random.default_rng(140)
        q, _ = np.linalg.qr(rng.normal(size=(40,40)))
        values = np.geomspace(.2,200.,40)
        a = (q*values)@q.T
        root = (q*np.sqrt(values))@q.T
        final = rng.normal(size=40)
        x = rng.normal(size=(40,20))
        history = [(j,None,None,x[:,j]) for j in range(20)]
        pod, _ = pod_basis(csr_matrix(a),history,final)
        s = final[:,None]-x
        _, _, vt = np.linalg.svd(root@s,full_matrices=False)
        for rank in (8,16):
            actual, _ = np.linalg.qr(pod[:,:rank])
            expected, _ = np.linalg.qr(s@vt[:rank].T)
            np.testing.assert_allclose(actual@actual.T,expected@expected.T,rtol=1e-10,atol=1e-10)

    def test_balanced_action_is_invariant_under_full_rank_basis_change(self):
        rng = np.random.default_rng(721)
        u = rng.normal(size=(25,25));a=u.T@u+np.eye(25)
        z = rng.normal(size=(25,8));change=rng.normal(size=(8,8))+5*np.eye(8)
        op = Balanced(csr_matrix(a),lambda v:v,z)
        other = Balanced(csr_matrix(a),lambda v:v,z@change)
        probes=rng.normal(size=(25,5))
        for p in probes.T:
            np.testing.assert_allclose(op(p),other(p),rtol=1e-10,atol=1e-10)

    def test_rank_deficiency_cannot_be_hidden_by_QR_completion(self):
        z=np.zeros((20,8));z[:,:2]=np.arange(20)[:,None]
        op=Balanced(csr_matrix(np.eye(20)),lambda v:v,z)
        self.assertEqual(op.Z.shape[1],1)

    def test_fixture_never_receives_a_real_scene_performance_budget(self):
        self.assertIsNone(mixed_budget_source(dict(fixture=True,n=60)))

    def test_budget_rejects_wrong_real_system_identity(self):
        with self.assertRaises(ValueError):
            mixed_budget_source(dict(n=60858,source_context=dict(frame=15,inner=0,linear_system_id=1),
                target_context=dict(frame=16,inner=1,linear_system_id=2)))

    def test_final_deadline_check_rejects_last_poll_race(self):
        from run_p5_probe import remaining
        with patch('windows_runner.time.monotonic',return_value=179.9):
            self.assertGreater(remaining(180.,180),0)
        for now in (180.,180.1):
            with patch('windows_runner.time.monotonic',return_value=now):
                with self.assertRaises(subprocess.TimeoutExpired):
                    remaining(180.,180)


if __name__ == '__main__':
    unittest.main()
