"""Behavioral tests for identity-sensitive configuration and acceptance math."""
import os
import sys
import unittest
from pathlib import Path
from config import ROOT, PRESETS, expand, environment, matches_requested, read, digest
sys.path.insert(0,str(ROOT/'tools'))
from run_bunny_components import PRESETS as FROZEN_PRESETS

class Contracts(unittest.TestCase):
    def test_contact_pool_is_explicit_and_history_is_preserved(self):
        c=expand({'preset':'combined'});old=dict(c)
        self.assertFalse(c['contact_pool']);self.assertFalse(c['contact_pool_validate'])
        old.pop('contact_pool');old.pop('contact_pool_validate');identity=digest(old)
        self.assertTrue(matches_requested(old,{'preset':'combined'}))
        self.assertEqual(digest(old),identity)
        self.assertFalse(matches_requested(old,{'preset':'combined','contact_pool':True}))
        env=environment(expand({'contact_pool':True,'contact_pool_validate':True}),Path('unused'))
        self.assertEqual(env['GIPC_CONTACT_POOL'],'1')
        self.assertEqual(env['GIPC_CONTACT_POOL_VALIDATE'],'1')
        for changed in ({'contact_pool':1},{'contact_pool_validate':'0'},
                        {'contact_pool_validate':True},{'preset':'toi','contact_pool':True}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):expand(changed)

    def test_bounded_ccd_is_explicit_and_does_not_rewrite_history(self):
        c=expand({'preset':'combined'});old=dict(c)
        self.assertFalse(c['bounded_ccd']);self.assertFalse(c['bounded_ccd_validate'])
        old.pop('bounded_ccd');old.pop('bounded_ccd_validate');identity=digest(old)
        self.assertTrue(matches_requested(old,{'preset':'combined'}))
        self.assertEqual(digest(old),identity)
        self.assertFalse(matches_requested(old,{'preset':'combined','bounded_ccd':True}))
        env=environment(expand({'bounded_ccd':True,'bounded_ccd_validate':True}),Path('unused'))
        self.assertEqual(env['GIPC_BOUNDED_CCD'],'1')
        self.assertEqual(env['GIPC_BOUNDED_CCD_VALIDATE'],'1')
        for changed in ({'bounded_ccd':1},{'bounded_ccd_validate':'0'},
                        {'bounded_ccd_validate':True},{'preset':'toi','bounded_ccd':True}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):expand(changed)

    def test_bvh_eligibility_is_explicit_and_history_is_preserved(self):
        c=expand({'preset':'combined'});old=dict(c)
        self.assertFalse(c['bvh_eligibility']);self.assertFalse(c['bvh_eligibility_validate'])
        old.pop('bvh_eligibility');old.pop('bvh_eligibility_validate')
        identity=digest(old)
        self.assertTrue(matches_requested(old,{'preset':'combined'}))
        self.assertEqual(digest(old),identity)
        self.assertFalse(matches_requested(old,{'preset':'combined','bvh_eligibility':True}))
        env=environment(expand({'bvh_eligibility':True,'bvh_eligibility_validate':True}),Path('unused'))
        self.assertEqual(env['GIPC_BVH_ELIGIBILITY'],'1')
        self.assertEqual(env['GIPC_BVH_ELIGIBILITY_VALIDATE'],'1')
        for changed in ({'bvh_eligibility':1},{'bvh_eligibility_validate':'0'},
                        {'bvh_eligibility_validate':True},{'preset':'toi','bvh_eligibility':True}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):expand(changed)

    def test_compensated_entrypoint_is_explicit_and_does_not_enable_execution_candidates(self):
        c=expand({'preset':'toi_compensated'})
        self.assertEqual(c['backend'],'ipc');self.assertEqual(c['ipc_termination'],'compensated')
        self.assertEqual(c['execution'],'conditional_graph');self.assertEqual(c['mas'],'legacy')
        self.assertEqual(c['ipc_cumulative_tol'],.001);self.assertEqual(c['ipc_residual_rel_tol'],.03)
        self.assertEqual(c['ipc_newton_tol'],.01);self.assertEqual(c['pcg_rho_tol'],1e-4)
        self.assertEqual(c['ipc_min_updates'],6)
        for key in ('refit','batch','reuse','mas_fused_dot','discrete_bvh_refit','spmv_fused_quadratic'):
            with self.subTest(default=key):self.assertFalse(c[key])
        self.assertEqual(expand({'preset':'toi_compensated','execution':'host'})['execution'],'host')
        self.assertEqual(expand({})['ipc_termination'],'legacy')
        self.assertEqual(expand({})['ipc_cumulative_tol'],.01)
        for changed in ({'ipc_min_updates':True},{'ipc_terminal_audit_frame':False},
                        {'ipc_residual_shadow':1},{'ipc_residual_cpu_audit':'0'},
                        {'preset':'toi_compensated','ipc_termination':'legacy'},
                        {'preset':'toi_compensated','backend':'toi_al'},
                        {'preset':'toi_compensated','ipc_residual_rel_tol':.0001}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):expand(changed)

    def test_spmv_quadratic_is_opt_in_and_fixed_study_is_isolated(self):
        c=expand({});old=dict(c)
        old.pop('spmv_fused_quadratic');old.pop('fixed_spmv_quadratic_study')
        self.assertTrue(matches_requested(old,{}))
        self.assertFalse(c['spmv_fused_quadratic'])
        env=environment(expand({'preset':'graph','spmv_fused_quadratic':True}),Path('unused'))
        self.assertEqual(env['GIPC_SPMV_FUSED_QUADRATIC'],'1')
        self.assertNotIn('GIPC_FIXED_SPMV_QUADRATIC_STUDY',env)
        study={'steps':2,'diagnostics':['fixed'],'fixed_frames':'2','fixed_directions':'1','fixed_spmv_quadratic_study':True}
        self.assertEqual(environment(expand(study),Path('unused'))['GIPC_FIXED_SPMV_QUADRATIC_STUDY'],'1')
        for changed in ({'spmv_fused_quadratic':1},{'fixed_spmv_quadratic_study':'1'},
                        {'preset':'toi','spmv_fused_quadratic':True},{'fixed_spmv_quadratic_study':True},
                        study|{'fixed_directions':'1,2'},study|{'fixed_mas_dot_study':True},
                        study|{'diagnostics':['fixed','cost']}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):expand(changed)

    def test_report_components_are_independent_explicit_and_historical(self):
        keys=('mas_fused_dot','fixed_mas_dot_study','discrete_bvh_refit','discrete_bvh_rebuild_interval','discrete_bvh_validate')
        c=expand({'preset':'combined'});old=dict(c)
        for key in keys:old.pop(key)
        self.assertTrue(matches_requested(old,{'preset':'combined'}))
        self.assertFalse(c['mas_fused_dot']);self.assertFalse(c['discrete_bvh_refit'])
        candidate=expand({'preset':'combined','mas_fused_dot':True,'discrete_bvh_refit':True})
        env=environment(candidate,Path('unused'))
        self.assertEqual(env['GIPC_MAS_FUSED_DOT'],'1')
        self.assertEqual(env['GIPC_DISCRETE_BVH_REFIT'],'1')
        self.assertEqual(env['GIPC_DISCRETE_BVH_REBUILD_INTERVAL'],'8')
        self.assertEqual(env['GIPC_DISCRETE_BVH_VALIDATE'],'0')
        self.assertNotIn('GIPC_FIXED_MAS_DOT_STUDY',env)
        study=expand({'steps':2,'diagnostics':['fixed'],'fixed_frames':'2','fixed_directions':'1','fixed_mas_dot_study':True})
        self.assertEqual(environment(study,Path('unused'))['GIPC_FIXED_MAS_DOT_STUDY'],'1')
        invalid=[{'mas_fused_dot':1},{'discrete_bvh_refit':'1'},{'discrete_bvh_validate':True},
                 {'discrete_bvh_rebuild_interval':True},{'discrete_bvh_rebuild_interval':0},
                 {'discrete_bvh_rebuild_interval':1025},{'preset':'toi','mas_fused_dot':True},
                 {'mas':'cholesky','mas_fused_dot':True},{'fixed_mas_dot_study':True},
                 {'steps':2,'diagnostics':['fixed'],'fixed_frames':'2','fixed_mas_dot_study':True,'fixed_mas_stage_study':True}]
        for value in invalid:
            with self.subTest(value=value),self.assertRaises(ValueError):expand(value)

    def test_light_cost_trace_preserves_default_and_is_explicit(self):
        old=expand({});old.pop('cost_events')
        self.assertTrue(matches_requested(old,{}))
        self.assertTrue(expand({})['cost_events'])
        request={'diagnostics':['cost'],'profile':'node','cost_events':False}
        c=expand(request);env=environment(c,Path('unused'))
        self.assertEqual(env['GIPC_COST_EVENTS'],'0')
        self.assertEqual(env['GIPC_COST_OPERATOR_PROBE'],'0')
        self.assertNotIn('GIPC_COST_EVENTS',environment(expand({}),Path('unused')))
        for changed in ({'diagnostics':[]},{'cost_events':'0'},{'cost_events':0}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):expand(request|changed)

    def test_ordered_legacy_candidate_is_isolated_and_bounded(self):
        old=expand({});old.pop('legacy_restrict');old.pop('fixed_legacy_restrict_study')
        self.assertTrue(matches_requested(old,{}))
        self.assertEqual(expand({})['legacy_restrict'],'atomic')
        self.assertFalse(expand({})['fixed_legacy_restrict_study'])
        request={'steps':2,'diagnostics':['fixed'],'fixed_frames':'2',
                 'fixed_directions':'1','fixed_legacy_restrict_study':True}
        env=environment(expand(request),Path('unused'))
        self.assertEqual(env['GIPC_FIXED_LEGACY_RESTRICT_STUDY'],'1')
        self.assertEqual(env['GIPC_LEGACY_RESTRICT'],'atomic')
        for changed in ({'diagnostics':[]},{'diagnostics':['fixed','cost']},
                        {'mas':'cholesky'},{'backend':'toi_al'},
                        {'fixed_frames':'1,2'},{'fixed_directions':'1,2'},
                        {'legacy_restrict':'ordered'},{'fixed_mas_stage_study':True}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):expand(request|changed)
        for changed in ({'legacy_restrict':'typo'}, {'legacy_restrict':'ordered','mas':'cholesky'},
                        {'legacy_restrict':'ordered','backend':'toi_al'}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):expand(changed)

    def test_mas_stage_probe_is_opt_in_single_system_and_legacy_only(self):
        old=expand({});old.pop('fixed_mas_stage_study')
        self.assertTrue(matches_requested(old,{}))
        self.assertFalse(expand({})['fixed_mas_stage_study'])
        request={'steps':2,'diagnostics':['fixed'],'fixed_frames':'2',
                 'fixed_directions':'1','fixed_mas_stage_study':True}
        c=expand(request)
        self.assertEqual(environment(c,Path('unused'))['GIPC_FIXED_MAS_STAGE_STUDY'],'1')
        for changed in ({'diagnostics':[]},{'diagnostics':['fixed','cost']},
                        {'mas':'cholesky'},{'backend':'toi_al'},
                        {'fixed_frames':'1,2'},{'fixed_directions':'1,2'},
                        {'fixed_frames':'1'},{'fixed_factor_study':True}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):expand(request|changed)

    def test_ipc_stopping_namespace_and_legacy_alias(self):
        c=expand({'ipc_newton_tol':.007})
        self.assertEqual(c['ipc_cumulative_tol'],.007)
        self.assertEqual(c['ipc_termination'],'legacy')
        self.assertFalse(c['ipc_residual_shadow'])
        self.assertTrue(matches_requested({k:v for k,v in c.items() if not k.startswith('ipc_') or k=='ipc_newton_tol'}, {'ipc_newton_tol':.007}))
        for cfg in ({'ipc_termination':'typo'},{'ipc_cumulative_tol':0},
                    {'ipc_residual_rel_tol':float('nan')},{'ipc_min_updates':0},
                    {'steps':43,'ipc_terminal_audit_frame':42},
                    {'preset':'toi','ipc_termination':'gated'}):
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):expand(cfg)
    def test_frozen_preset_compatibility(self):
        for label,old in FROZEN_PRESETS.items():
            with self.subTest(label=label):
                c=expand({'preset':label});env=environment(c,Path('unused'))
                self.assertEqual(env['GIPC_PCG_EXECUTION'],'conditional_graph' if old.get('graph') else 'host')
                self.assertEqual(env['GIPC_CONTACT_BACKEND'],'toi_al' if old.get('toi') else 'ipc')
                for flag,key in [('cholesky','GIPC_MAS_CHOLESKY'),('refit','GIPC_CCD_BVH_REFIT'),
                                 ('batch','GIPC_BATCHED_ENERGY'),('reuse','GIPC_ENERGY_REUSE')]:
                    self.assertEqual(env[key],str(int(old.get(flag,False))))
                self.assertEqual(c['ipc_newton_tol'],.01)
                self.assertEqual(c['toi_remaining_fraction_tol'],.01)
                self.assertEqual(c['toi_trial_velocity_tol'],.05)
                self.assertEqual(c['pcg_rho_tol'],1e-4)

    def test_independent_tolerances_and_environment_isolation(self):
        os.environ['GIPC_TEST_POLLUTION']='1'
        try:
            env=environment(expand({'toi_remaining_fraction_tol':.003}),Path('unused'))
            self.assertNotIn('GIPC_TEST_POLLUTION',env)
            self.assertEqual(env['GIPC_NEWTON_TOL'],'0.01')
            self.assertEqual(env['GIPC_TOI_REMAINING_FRACTION_TOL'],'0.003')
        finally:del os.environ['GIPC_TEST_POLLUTION']

    def test_unknown_or_unbounded_configuration_is_rejected(self):
        for cfg in ({'steps':301},{'timeout_seconds':601},{'pcg_rho_tol':0},
                    {'tol':.1},{'diagnostics':['typo']},{'diagnostics':['operator_probe']},
                    {'mas_restrict':'typo'},{'mas_restrict':'warp'},{'fixed_restrict_study':True},
                    {'mas_factor_action':'factor_inverse'},{'mas_factor_action':'inverse'},
                    {'fixed_factor_study':True},
                    {'preset':'toi','diagnostics':['fixed'],'fixed_factor_study':True,'fixed_restrict_study':True}):
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):expand(cfg)

    def test_restriction_candidate_is_explicit_and_inactive_by_default(self):
        self.assertEqual(expand({'preset':'toi'})['mas_restrict'],'serial')
        c=expand({'preset':'toi','mas_restrict':'warp','diagnostics':['fixed'],'fixed_restrict_study':True})
        env=environment(c,Path('unused'))
        self.assertEqual(env['GIPC_MAS_RESTRICT_MODE'],'warp')
        self.assertEqual(env['GIPC_FIXED_RESTRICT_STUDY'],'1')

    def test_factor_action_defaults_on_in_stable_path_and_can_be_disabled(self):
        self.assertEqual(expand({'preset':'toi'})['mas_factor_action'],'factor_inverse')
        self.assertEqual(expand({'preset':'host'})['mas_factor_action'],'triangular')
        self.assertEqual(expand({'preset':'toi','mas_factor_action':'triangular'})['mas_factor_action'],'triangular')
        c=expand({'preset':'toi','mas_factor_action':'factor_inverse','diagnostics':['fixed'],'fixed_factor_study':True})
        env=environment(c,Path('unused'))
        self.assertEqual(env['GIPC_MAS_FACTOR_ACTION'],'factor_inverse')
        self.assertEqual(env['GIPC_FIXED_FACTOR_STUDY'],'1')

    def test_world_penalty_defaults_on_in_toi_diagonal_and_can_be_disabled(self):
        self.assertEqual(expand({'preset':'toi'})['mu_coordinates'],'world_block')
        self.assertEqual(expand({'preset':'toi','mu_coordinates':'generalized'})['mu_coordinates'],'generalized')
        self.assertEqual(expand({'preset':'toi','mu_mode':'mass'})['mu_coordinates'],'generalized')
        c=expand({'preset':'toi','mu_coordinates':'world_block'})
        self.assertEqual(environment(c,Path('unused'))['GIPC_TOI_MU_COORDINATES'],'world_block')
        for cfg in ({'mu_coordinates':'world_block'},
                    {'preset':'toi','mu_coordinates':'world_block','mu_mode':'mass'},
                    {'preset':'toi','mu_coordinates':'typo'}):
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):expand(cfg)

    def test_historical_penalty_config_keeps_identity_and_cannot_claim_world(self):
        request=read(ROOT/'runs/active/factor_window_mixed_triangular_r1/requested.json')
        saved=request['expanded_config'];before=digest(saved)
        self.assertNotIn('mu_coordinates',saved)
        self.assertTrue(matches_requested(saved,request['requested_config']))
        self.assertFalse(matches_requested(saved,request['requested_config']|{'mu_coordinates':'world_block'}))
        self.assertEqual(digest(saved),before)
        self.assertEqual(digest(saved),request['config_sha256'])

    def test_exit_probe_is_bounded_observed_and_off_by_default(self):
        self.assertIsNone(expand({'preset':'toi'})['full_step_exit_probe'])
        c=expand({'preset':'toi','diagnostics':['outer_probe'],'full_step_exit_probe':'34:2'})
        self.assertEqual(environment(c,Path('unused'))['GIPC_TOI_FULL_STEP_EXIT_PROBE'],'34:2')
        for data in ({'full_step_exit_probe':'34:2'},
                     {'preset':'toi','full_step_exit_probe':'34:2'},
                     {'preset':'toi','diagnostics':['outer_probe'],'full_step_exit_probe':'36:2'},
                     {'preset':'toi','diagnostics':['outer_probe'],'full_step_exit_probe':'34:-1'},
                     {'preset':'toi','diagnostics':['outer_probe'],'full_step_exit_probe':'34:2:3'}):
            with self.subTest(data=data),self.assertRaises(ValueError):expand(data)

if __name__=='__main__':unittest.main()
