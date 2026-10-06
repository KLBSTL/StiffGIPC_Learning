"""Finite contact-pool study. No native defaults are changed here."""
from config import expand, digest

SCENES = {'hang': 'cloth_hang_l', 'fixed': 'cloth_fixed_bunny_l', 'mixed': 'bunny_cloth_bunny_l'}
STEPS = {'hang': 51, 'fixed': 59, 'mixed': 35}
WINDOWS = {'hang': [41, 50], 'fixed': [22, 59], 'mixed': [24, 35]}
STAGES = ('guards', 'r1', 'r2', 'r3')

def task(scene, variant, stage):
    c = {'scene': SCENES[scene], 'preset': 'combined', 'backend': 'ipc',
         'execution': 'conditional_graph', 'mas': 'legacy', 'dt': .01,
         'steps': STEPS[scene], 'timeout_seconds': 120, 'trace_velocity': True,
         'diagnostics': [], 'profile': 'none', 'discrete_bvh_refit': True,
         'bounded_ccd': False, 'bounded_ccd_validate': False,
         'bvh_eligibility': False, 'bvh_eligibility_validate': False,
         'contact_pool': variant == 'on', 'contact_pool_validate': stage == 'guards' and variant == 'on',
         'ipc_termination': 'legacy', 'ipc_min_updates': 6,
         'ipc_newton_tol': .01, 'ipc_cumulative_tol': .01, 'pcg_rho_tol': 1e-4}
    return {'name': f'{stage}_{scene}_{variant}', 'stage': stage, 'repeat': stage,
            'scene_key': scene, 'variant': variant, 'binary': 'active', 'config': c,
            'expanded_config_sha256': digest(expand(c))}

def tasks(stage):
    if stage not in STAGES:
        raise ValueError('Unknown finite stage')
    if stage == 'guards':
        return [task('hang', 'on', stage), task('fixed', 'on', stage),
                task('mixed', 'off', stage), task('mixed', 'on', stage)]
    return [task(s, v, stage) for s in (('fixed', 'hang') if stage == 'r2' else ('hang', 'fixed'))
            for v in (('on', 'off') if stage == 'r2' else ('off', 'on'))]

def protocol():
    return {'schema': 'flat_contact_pool_plan.v1', 'stages': {s: tasks(s) for s in STAGES},
            'contact_windows': WINDOWS, 'finite_run_budget': 16, 'from_zero': True,
            'predecessors': {'guards': None, 'r1': 'guards', 'r2': 'r1', 'r3': 'r2'},
            'guard_gate': 'All-scene hard checks and typed/energy pool validation; frozen cloth material bounds. Mixed differences remain pending, not a cloth-screen prerequisite.',
            'round_gate': 'Analyze completed round before next launch; hard checks, cloth bounds and initial-state matching required. No candidate-based tolerance expansion.',
            'screen_threshold': {'paired_median_whole': 1.05, 'paired_median_contact_window': 1.05, 'required_pairs_per_scene': 3},
            'after_screen': 'No automatic long run. Cost-package evidence and independent quality review still required.',
            'performance_certified': False, 'quality_certified': False}
