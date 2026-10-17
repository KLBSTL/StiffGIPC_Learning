import os
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from config import expand,environment,matches_requested
from contracts_test import make_graph_config_run,dump
from validate_run import validate

class P3Contracts(unittest.TestCase):
    def observed(self,mode,dispatch,w,missing_frames=False):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);c=make_graph_config_run(base);c['p3_ee_mode']=mode
            dump(base/'requested.json',dict(binary='active',expanded_config=c))
            import json
            resolved=json.loads((base/'resolved_config.json').read_text())
            resolved['p3_ee']=dict(mode=mode,queue_capacity=1048576,study_frames=[],study_file='')
            dump(base/'resolved_config.json',resolved)
            stats=json.loads((base/'output/stats.json').read_text())
            usage=dict(mode=mode,w_queries=w)
            if dispatch is not None:usage['dispatch_queries']=dispatch
            stats['frames'][0]['p3_ee']=usage
            if missing_frames:stats['frames']=[]
            dump(base/'output/stats.json',stats)
            return validate(base)['passed']
    def test_no_eligible_query_is_valid(self):
        self.assertTrue(self.observed('two_stage',0,0))
        self.assertTrue(self.observed('fused',0,0))
    def test_missing_frame_is_not_a_valid_empty_query(self):
        self.assertFalse(self.observed('two_stage',0,0,missing_frames=True))
    def test_every_eligible_query_must_dispatch_w(self):
        self.assertTrue(self.observed('two_stage',3,3))
        self.assertFalse(self.observed('two_stage',3,2))
        self.assertFalse(self.observed('two_stage',0,1))
    def test_counter_contract_rejects_invalid_or_mixed_modes(self):
        for dispatch,w in ((True,True),(-1,0),(2,-1),(2,True)):
            self.assertFalse(self.observed('two_stage',dispatch,w))
        self.assertFalse(self.observed('fused',1,1))
        self.assertTrue(self.observed('fused',3,0))
    def test_historical_usage_is_not_reinterpreted(self):
        self.assertTrue(self.observed('two_stage',None,1))
        self.assertFalse(self.observed('two_stage',None,0))
    def test_ratio_sample_count_and_finiteness(self):
        from p3_analyze import ratios
        for values in ([],[1],[1,1],[1]*4,[1,0,1],[1,float('inf'),1],[1,float('nan'),1]):
            with self.assertRaises(ValueError):ratios(values)
        self.assertEqual(ratios([1,1,1])['geometric_mean'],1)
    def test_presets_keep_fused(self):
        for preset in ('base','combined','graph'):
            c=expand(dict(preset=preset));self.assertEqual(c['p3_ee_mode'],'fused')
            self.assertIsNone(c['p3_ee_study_frames'])
    def test_isolated_module(self):
        for other in ({'backend':'toi_al'},{'p2_lazy_scene':True},{'p2_merge_abd_energy':True},{'p2_audit':True},{'contact_pool':True}):
            with self.assertRaises(ValueError):expand(dict(p3_ee_mode='two_stage')|other)
    def test_capacity_contract(self):
        for cap in (True,0,-1,8388609,1.5,'100'):
            with self.assertRaises(ValueError):expand(dict(p3_ee_queue_capacity=cap))
        self.assertEqual(expand(dict(p3_ee_queue_capacity=8388608))['p3_ee_queue_capacity'],8388608)
    def test_explicit_diagnostic_selection(self):
        for frames in ('','0','1,1','1,','-1','1-3',True):
            with self.assertRaises(ValueError):expand(dict(p3_ee_study_frames=frames,diagnostics=['p3_ee']))
        with self.assertRaises(ValueError):expand(dict(p3_ee_study_frames='17'))
        with self.assertRaises(ValueError):expand(dict(diagnostics=['p3_ee']))
        c=expand(dict(p3_ee_study_frames='17',diagnostics=['p3_ee'],steps=1))
        self.assertEqual(c['p3_ee_study_frames'],'17') # cold replay uses physical frame
    def test_no_ambient_p3(self):
        prior=os.environ.get('GIPC_P3_EE_STUDY_FRAMES')
        try:
            os.environ['GIPC_P3_EE_STUDY_FRAMES']='17'
            env=environment(expand({}),ROOT/'build/test')
            self.assertNotIn('GIPC_P3_EE_STUDY_FRAMES',env)
            self.assertEqual(env['GIPC_P3_EE_MODE'],'fused')
        finally:
            if prior is None:os.environ.pop('GIPC_P3_EE_STUDY_FRAMES',None)
            else:os.environ['GIPC_P3_EE_STUDY_FRAMES']=prior
    def test_old_requests_keep_original_defaults(self):
        old=expand(dict(preset='combined'))
        for key in ('p3_ee_mode','p3_ee_queue_capacity','p3_ee_study_frames'):old.pop(key)
        self.assertTrue(matches_requested(old,dict(preset='combined')))
    def test_finite_execution_plan(self):
        sys.path.insert(0,str(ROOT/'tools/local'))
        from run_innovation_p3 import tasks
        self.assertEqual([len(tasks(s)) for s in ('smoke','prefix','query','matched','fallback','whole')],[2,3,6,18,3,18])
        for stage in ('smoke','prefix','query','matched','fallback','whole'):
            for task in tasks(stage):
                c=task['config'];self.assertEqual((c['dt'],c['pcg_rho_tol'],c['ipc_cumulative_tol'],c['ipc_min_updates']),(.04,1e-4,.01,6))
                self.assertFalse(any(c[k] for k in ('p2_lazy_scene','p2_merge_abd_energy','p2_audit')))
                if stage in ('matched','whole'):self.assertEqual(c['diagnostics'],[])
                self.assertLessEqual(c['steps'],17)

if __name__=='__main__':unittest.main()
