"""Predeclared full evaluation; the plan never launches a process."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from config import expand, digest

SCENES = {'hang': 'cloth_hang_l', 'fixed': 'cloth_fixed_bunny_l', 'mixed': 'bunny_cloth_bunny_l'}
ARMS = ('original_stiff', 'ipc_host', 'ipc_graph', 'combined_graph')


def task(scene, arm, phase, repeat):
    c = {'scene': SCENES[scene], 'preset': 'combined' if arm == 'combined_graph' else 'base',
         'backend': 'ipc', 'execution': 'conditional_graph' if arm in ('ipc_graph', 'combined_graph') else 'host',
         'mas': 'legacy', 'steps': 300 if phase == 'stability' else 100, 'dt': .01,
         'timeout_seconds': 600 if phase == 'stability' else 120,
         'ipc_termination': 'legacy', 'ipc_newton_tol': .01,
         'ipc_cumulative_tol': .01, 'ipc_min_updates': 6, 'pcg_rho_tol': 1e-4,
         'pcg_graph_chunk': 1, 'fixed_graph_chunk_study': False,
         'discrete_bvh_refit': arm == 'combined_graph', 'discrete_bvh_rebuild_interval': 8,
         'contact_pool': False, 'contact_pool_validate': False, 'edge_query_order': 'raw',
         'trace_velocity': arm != 'original_stiff', 'diagnostics': [], 'profile': 'none'}
    prefix = 'p' if phase == 'paired' else phase
    return {'name': f'{prefix}{repeat}_{scene}_{arm}', 'scene_key': scene, 'arm': arm,
            'variant': arm, 'phase': phase, 'repeat': repeat,
            'pair': repeat if phase == 'paired' else None,
            'calibration_index': repeat if phase == 'calibration' else None,
            'verification_index': repeat if phase == 'verification' else None,
            'binary': 'base' if arm == 'original_stiff' else 'active',
            'config': c, 'expanded_config_sha256': digest(expand(c))}


def tasks():
    rows = []
    # Every reference envelope and its holdout evidence precedes every candidate.
    for scene in SCENES:
        rows += [task(scene, 'original_stiff', 'calibration', r) for r in range(1, 4)]
        rows += [task(scene, 'original_stiff', 'verification', r) for r in range(1, 3)]
    for pair in range(1, 8):
        scenes = tuple(SCENES) if pair % 2 else tuple(reversed(SCENES))
        for scene in scenes:
            # Keep each principal pair adjacent. Reverse both comparisons on
            # alternate rounds; calibration runs never enter paired ratios.
            arms = ('original_stiff', 'combined_graph') if pair % 2 else ('combined_graph', 'original_stiff')
            if pair <= 3:
                arms += ('ipc_host', 'ipc_graph') if pair % 2 else ('ipc_graph', 'ipc_host')
            rows += [task(scene, arm, 'paired', pair) for arm in arms]
    rows += [task(scene, 'combined_graph', 'stability', 1) for scene in SCENES]
    return [dict(row, index=i) for i, row in enumerate(rows, 1)]


def plan():
    return {'schema': 'full_eval_plan.v1', 'tasks': tasks(), 'scenes': SCENES,
            'main_pairs_per_scene': 7, 'component_pairs_per_scene': 3,
            'calibration_per_scene': 3, 'holdout_per_scene': 2,
            'full_runs': 75, 'conditional_stability_runs': 3, 'maximum_gpu_runs': 78,
            'from_zero': True, 'automatic_retry': False, 'one_task_per_invocation': True,
            'heavy_diagnostics': False, 'graph_chunk': 1,
            'quality_certified': False, 'performance_certified': False,
            'baseline_capabilities': {'resolved_config': False, 'actual_velocity': False},
            'quality_policy': 'Freeze three original-Stiff repetitions, check two holdouts before candidates. Quality differences remain diagnostic; never expand the reference using candidates.',
            'hard_failure_policy': 'Stop this scene/arm after numerical, configuration or resource failure. Retain failures and skip its later declared runs. Analysis exceptions stop all advancement.',
            'stability_condition': 'All 25 declared 100-frame runs for that scene completed with hard checks passed, and the CPU analysis frozen-material gate is eligible.',
            'resources': {'timeout_seconds_by_steps': {'100': 120, '300': 600},
                          'gpu_memory_budget': 'min(0.75*free, free-1536MiB)',
                          'minimum_start_disk_gib': 4, 'minimum_running_disk_gib': 1,
                          'minimum_running_gpu_free_mib': 768, 'gpu_serial_lock': 'runs/.gpu.lock'}}
