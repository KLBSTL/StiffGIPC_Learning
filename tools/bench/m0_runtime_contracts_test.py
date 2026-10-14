"""CPU-only M0 launcher declarations and checkpoint rejection contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('run_m0',ROOT/'tools/local/run_innovation_m0.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class RuntimeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch=ROOT/'build/m0_test_scratch';cls.scratch.mkdir(parents=True,exist_ok=True)

    def test_bounded_declared_tasks_and_original_identity(self):
        self.assertEqual([len(m.tasks(s)) for s in ('smoke','prefix','matched','reference')],[2,9,30,8])
        all_tasks=sum((m.tasks(s) for s in ('smoke','prefix','matched','reference')),[])
        self.assertEqual(len({t['name'] for t in all_tasks}),49)
        for t in all_tasks:
            c=t['config'];self.assertEqual(c['dt'],.04)
            self.assertLessEqual(c['steps'],20);self.assertEqual(c['timeout_seconds'],120)
            if t['binary']=='base':
                self.assertEqual(c['execution'],'host');self.assertEqual(c['ipc_termination'],'legacy')
                self.assertFalse(c['trace_velocity']);self.assertFalse(c['discrete_bvh_refit'])

    def test_reference_does_not_bypass_outer_checkpoint_guard(self):
        for t in m.tasks('reference'):
            self.assertNotIn('checkpoint_from',t)
        for t in m.tasks('matched'):
            self.assertEqual(t['config']['ipc_newton_tol'],.01)
            self.assertEqual(t['target_frame'],m.SCENES[t['scene']][1])

    def test_restore_rejects_wrong_binary_before_claiming_success(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
            root=Path(temp);cp=root/'cp';out=root/'out'
            cp.mkdir();(out/'output').mkdir(parents=True)
            (cp/'checkpoint.json').write_text(json.dumps(dict(binary_id='wrong',next_physical_frame=14,buffers={})))
            (out/'output/checkpoint_restore.json').write_text('{}')
            with self.assertRaises(ValueError):m.validate_restore(out,cp,dict(exe=dict(sha256='correct')),14)

    def test_ambient_experiment_flags_removed(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ,{'GIPC_UNDECLARED_FLAG':'bad'}):
            env=m.environment(m.configuration('Bunny',1),ROOT/'runs/test_contract')
        self.assertNotIn('GIPC_UNDECLARED_FLAG',env)

    def test_requested_origin_follows_actual_checkpoint_environment(self):
        self.assertTrue(m.windows_runner.run_origin({})['from_zero'])
        self.assertTrue(m.windows_runner.run_origin({'GIPC_CHECKPOINT_LOAD':''})['from_zero'])
        origin=m.windows_runner.run_origin({'GIPC_CHECKPOINT_LOAD':'verified/checkpoint'})
        self.assertFalse(origin['from_zero'])
        self.assertEqual(origin['checkpoint_load'],'verified/checkpoint')


if __name__=='__main__':unittest.main()
