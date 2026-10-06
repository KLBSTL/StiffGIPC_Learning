"""CPU contracts for the finite four-arm experiment; no GPU calls."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import bvh_rounds as b


class Contracts(unittest.TestCase):
    def test_only_two_component_fields_change(self):
        for scene in ('fixed', 'hang'):
            for repeat in (1, 2, 3):
                rows = b.tasks(scene, repeat)
                ref = next(r['config'] for r in rows if r['variant'] == 'D1S1')
                for row in rows:
                    self.assertLessEqual({k for k in ref if ref[k] != row['config'][k]}, {'discrete_bvh_refit', 'refit'})
                    self.assertEqual(row['config']['contact_pool'], False)
                    self.assertEqual((row['config']['ipc_cumulative_tol'], row['config']['ipc_min_updates'], row['config']['pcg_rho_tol']), (.01, 6, 1e-4))
                    self.assertEqual(row['config']['steps'], 59 if scene == 'fixed' else 51)

    def test_finite_orders(self):
        rows = [r for scene in ('fixed', 'hang') for repeat in (1, 2, 3) for r in b.tasks(scene, repeat)]
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({r['name'] for r in rows}), 24)
        self.assertEqual(tuple(reversed(b.ORDERS[1])), b.ORDERS[2])
        with self.assertRaises(ValueError): b.tasks('fixed', 4)

    def test_no_retry_or_missing_predecessor(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp)
            self.assertEqual(b.check_previous(session, 'fixed', 1, 'seal', {}, None), 1)
            with self.assertRaises(ValueError): b.check_previous(session, 'fixed', 2, 'seal', {}, None)
            (session / 'fixed_r1').mkdir()
            with self.assertRaises(ValueError): b.check_previous(session, 'fixed', 1, 'seal', {}, None)

    def test_cross_scene_requires_latest_review(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp); folder = session / 'fixed_r1'; folder.mkdir()
            b.write_new(folder / 'receipt.json', {})
            b.write_new(folder / 'analysis.json', {'diagnosis_complete': True})
            with patch.object(b, 'verify_stage', return_value={'sequence': 1}):
                with self.assertRaises(ValueError): b.check_previous(session, 'hang', 1, 'seal', {}, None)
                self.assertEqual(b.check_previous(session, 'hang', 1, 'seal', {}, b.sha(folder / 'analysis.json')), 2)

    def test_failed_receipt_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'fixed_r1'; folder.mkdir()
            b.write_new(folder / 'receipt.json', {'seal_sha256': 'wrong', 'tools': {}})
            b.write_new(folder / 'batch.json', {})
            b.write_new(folder / 'analysis.json', {})
            with self.assertRaises(ValueError): b.verify_stage(folder, 'expected', {})

    def test_compact_excludes_large_arrays(self):
        row = {'name': 'fixed_D1S1_r1', 'variant': 'D1S1', 'hard_checks_passed': True, 'failures': [],
               'material': {}, 'within_original_bounds': False,
               'timing_observation': {'prefix': {'solver_ms': 1, 'directions': 6, 'accepted_alphas_by_frame': [[.5]]}}}
        compact = b.compact(copy.deepcopy(row))
        self.assertNotIn('accepted_alphas_by_frame', compact['timing']['prefix'])
        self.assertFalse(compact['within_original_bounds'])


if __name__ == '__main__': unittest.main()
