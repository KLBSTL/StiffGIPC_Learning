"""CPU contracts for the explicitly requested 21-run diagnostic suite."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).with_name('run.py')

class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.is_file(), 'Seven-scene controller missing')
        spec = importlib.util.spec_from_file_location('seven_suite', PATH)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_matrix_has_exactly_seven_scenes_three_arms_120_frames(self):
        tasks = self.module.tasks()
        self.assertEqual(len(tasks), 21)
        self.assertEqual(len({t['config']['scene'] for t in tasks}), 7)
        for scene in self.module.SCENES:
            group = [t for t in tasks if t['config']['scene'] == scene]
            self.assertEqual({t['arm'] for t in group}, {'base','graph','all'})
            for t in group:
                c = self.module.expand(t['config'])
                self.assertEqual((c['steps'],c['dt'],c['pcg_rho_tol']), (120,.01,1e-4))
                self.assertEqual((c['ipc_termination'],c['ipc_cumulative_tol'],c['ipc_min_updates']), ('legacy',.01,6))
                self.assertEqual(c['timeout_seconds'],180)
                self.assertEqual(c['pcg_graph_chunk'],1)
                self.assertFalse(c['diagnostics'])
                self.assertFalse(c['contact_pool'])

    def test_all_means_retained_execution_components_only(self):
        for t in self.module.tasks():
            c = self.module.expand(t['config'])
            for key in ('refit','batch','reuse','discrete_bvh_refit'):
                self.assertEqual(c[key], t['arm']=='all')
            self.assertEqual(c['execution'], 'host' if t['arm']=='base' else 'conditional_graph')
            for key in ('mas_fused_dot','spmv_fused_quadratic','bounded_ccd','bvh_eligibility','ipc_residual_shadow'):
                self.assertFalse(c[key])

    def test_ratios_require_all_complete_and_equal_inputs(self):
        rows = [{'arm':arm,'hard_checks_passed':True,'timing':{'solver_seconds':sec}}
                for arm,sec in [('base',12),('graph',8),('all',6)]]
        ratios = self.module.ratios(rows, True)
        self.assertEqual(ratios, {'base':1.0,'graph':1.5,'all':2.0,'other_components':8/6})
        self.assertIsNone(self.module.ratios(rows,False))
        rows[1]['hard_checks_passed']=False
        self.assertIsNone(self.module.ratios(rows,True))
        self.assertIsNone(self.module.ratios(rows[:2],True))

    def test_batch_stops_resource_or_configuration_failures(self):
        batch_path=PATH.with_name('batch.py')
        self.assertTrue(batch_path.is_file(),'Automatic batch controller missing')
        spec=importlib.util.spec_from_file_location('seven_batch',batch_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertTrue(module.must_stop({'status':'memory_budget'},[]))
        self.assertTrue(module.must_stop({'status':'configuration_failed'},[]))
        self.assertFalse(module.must_stop({'status':'completed'},['Reported PCG limit/breakdown']))

    def test_missing_start_receipt_does_not_fabricate_supervisor_time(self):
        deployment=PATH.with_name('deploy.py')
        spec=importlib.util.spec_from_file_location('seven_deployment',deployment)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertTrue(hasattr(module,'configure_time_bound'),'Explicit timestamp recovery absent')
        result=module.configure_time_bound('1970-01-01 00:00:02.123456789 +0000',1000000000)
        self.assertFalse(result['actual_supervisor_start_available'])
        self.assertAlmostEqual(result['time_unix'],2.123455,places=6)
        self.assertEqual(result['timestamp_source'],'Filesystem birth time of first configure log; conservative lower bound')
        with self.assertRaises(ValueError):module.configure_time_bound('-',1000000000)
        with self.assertRaises(ValueError):module.configure_time_bound('1970-01-01 00:00:00.000000000 +0000',1000000000)

if __name__ == '__main__': unittest.main()
