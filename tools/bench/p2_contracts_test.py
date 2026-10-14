import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from config import expand,environment,matches_requested
from run_innovation_p2a import tasks,ARMS
import run_innovation_p2a as driver

class Contracts(unittest.TestCase):
    def test_default_and_combined_do_not_enable_p2(self):
        for preset in ('host','combined'):
            c=expand(dict(preset=preset))
            self.assertFalse(any(c[k] for k in ('p2_lazy_scene','p2_merge_abd_energy','p2_audit')))
    def test_environment_does_not_inherit_p2(self):
        before=os.environ.get('GIPC_P2_LAZY_SCENE')
        try:
            os.environ['GIPC_P2_LAZY_SCENE']='1'
            self.assertEqual(environment(expand({}),ROOT/'build/test')['GIPC_P2_LAZY_SCENE'],'0')
        finally:
            if before is None:os.environ.pop('GIPC_P2_LAZY_SCENE',None)
            else:os.environ['GIPC_P2_LAZY_SCENE']=before
    def test_reject_nonboolean_and_wrong_backend(self):
        with self.assertRaises(ValueError):expand(dict(p2_lazy_scene=1))
        with self.assertRaises(ValueError):expand(dict(backend='toi_al',p2_merge_abd_energy=True))
    def test_historical_request_preserved(self):
        old=expand(dict(preset='combined'))
        for k in ('p2_lazy_scene','p2_merge_abd_energy','p2_audit'):old.pop(k)
        self.assertTrue(matches_requested(old,dict(preset='combined')))
    def test_relative_session_normalized_before_read(self):
        class StopRead(Exception):pass
        with patch.object(driver,'read',side_effect=StopRead) as read:
            with self.assertRaises(StopRead):driver.run_stage(Path('runs/relative_contract'),'smoke')
        self.assertTrue(read.call_args.args[0].is_absolute())
    def test_finite_matched_and_timing_contracts(self):
        self.assertEqual([len(tasks(s)) for s in ('smoke','prefix','cost','audit','matched','whole')],
            [4,3,12,6,36,18])
        matched=tasks('matched');self.assertEqual(len(matched),36)
        for row in matched:
            c=row['config'];self.assertEqual(c['steps'],1)
            self.assertEqual((c['p2_lazy_scene'],c['p2_merge_abd_energy']),ARMS[row['arm']])
            self.assertEqual((c['dt'],c['pcg_rho_tol'],c['ipc_cumulative_tol'],c['ipc_min_updates']),(.04,1e-4,.01,6))
            self.assertFalse(c['p2_audit']);self.assertEqual(c['diagnostics'],[])
            self.assertIn('checkpoint_from',row)
        self.assertEqual(len(tasks('whole')),18)

if __name__=='__main__':unittest.main()
