import unittest
from unittest.mock import patch
import report_rounds as r
import quality_analysis
from config import digest,expand
from profiler_windows import derive_config

class ReportRoundsTests(unittest.TestCase):
    def test_observed_capability_does_not_accept_resource_or_partial_failure(self):
        from pathlib import Path
        c={'steps':100,'dt':.01,'scene':'cloth_sphere7_l','ipc_newton_tol':.01,'pcg_rho_tol':1e-4}
        def check(observed,status,frames=100,exit_code=0,finite=True,capability_only=False):
            values={'requested.json':{'expanded_config':c,'binary':'base','capabilities':{'actual_velocity':observed}},
                'result.json':dict(status=status,recorded_frames=frames,exit_code=exit_code,finite=finite),
                'scene.json':{'case_id':c['scene'],'effective_run':{'dt':.01,'newton_tol':.01,'pcg_tol':1e-4},'effective_scalar_fields':{'preconditioner_type':1}},
                'stats.json':{'frames':[{'newton':[{'pcg':{'iteration_limit':False}}]}]}}
            with patch.object(quality_analysis,'read',side_effect=lambda p:values[p.name]),patch.object(quality_analysis,'export_evidence',return_value={'passed':True}):
                result=quality_analysis.validate_base(Path('nonexistent_fixture_root'))
                return result['actual_velocity_available'] if capability_only else result['passed']
        self.assertTrue(check(True,'configuration_failed'))
        self.assertTrue(check(False,'completed'))
        self.assertFalse(check(False,'configuration_failed'))
        for status in ('memory_budget','timeout','failed'):
            self.assertFalse(check(True,status))
        self.assertFalse(check(True,'configuration_failed',frames=99))
        self.assertFalse(check(True,'configuration_failed',exit_code=1))
        self.assertFalse(check(True,'configuration_failed',finite=False))
        self.assertTrue(check(True,'configuration_failed',capability_only=True))
        self.assertFalse(check(False,'completed',capability_only=True))

    def test_finite_stages_and_adjacent_pairs(self):
        self.assertEqual([len(r.tasks(s)) for s in ('reference','screen','ablation_sphere','ablation_fixed')],[10,12,24,24])
        for stage in ('screen','ablation_sphere','ablation_fixed'):
            ts=r.tasks(stage)
            for a,b in zip(ts[::2],ts[1::2]):
                self.assertEqual((a['scene_key'],a['repeat']),(b['scene_key'],b['repeat']))
                self.assertNotEqual(a['arm'],b['arm'])
        for stage in ('reference','screen','ablation_sphere','ablation_fixed'):
            for t in r.tasks(stage):
                c=t['config'];self.assertEqual((c['steps'],c['dt'],c['pcg_rho_tol'],c['ipc_cumulative_tol'],c['ipc_min_updates'],c['pcg_graph_chunk']),(100,.01,1e-4,.01,6,1))
                self.assertEqual(c,expand(c));self.assertEqual(t['expanded_config_sha256'],digest(c))
                self.assertFalse(c['contact_pool']);self.assertFalse(c['diagnostics'])

    def test_one_component_ablation(self):
        for flag,key in r.FLAGS.items():
            full=r.task('sphere','full_'+flag,1,'ablation')['config']
            minus=r.task('sphere','minus_'+flag,1,'ablation')['config']
            self.assertEqual({k for k in full if full[k]!=minus[k]},{key})

    def test_no_ablation_from_incomplete_or_insufficient_samples(self):
        def rows(n,seconds):
            return [dict(phase='screen',scene_key='sphere',repeat=i,arm=a,name=f'{i}{a}',timing={'solver_seconds':v})
                    for i in range(1,n+1) for a,v in (('graph',10.),('combined',seconds))]
        with patch.object(r,'input_comparison',return_value={'passed':True}):
            self.assertFalse(r.screen_summary(rows(2,8),'sphere')['allow_ablation'])
            self.assertFalse(r.screen_summary(rows(3,9.6),'sphere')['allow_ablation'])
            self.assertTrue(r.screen_summary(rows(3,9.4),'sphere')['allow_ablation'])

    def test_profile_window_does_not_change_physics(self):
        for scene in ('sphere','fixed'):
            c=r.task(scene,'combined',1,'screen')['config']
            req={'expanded_config':c,'config_sha256':digest(c),'binary':'active'}
            res={'status':'completed','recorded_frames':100,'finite':True}
            observed=derive_config(req,res,'node',24)
            self.assertEqual({k for k in c if c[k]!=observed[k]}, {'diagnostics','cost_frames','cost_events','profile'})
            for frame in (0,101,1.5,True):
                with self.assertRaises(ValueError):derive_config(req,res,'node',frame)

if __name__=='__main__':unittest.main()
