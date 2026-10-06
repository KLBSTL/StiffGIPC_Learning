"""Predeclared seven-run quality diagnosis, independent of the failed screen."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from config import expand, digest
from plans import task as screen_task

ORDER=(('off',1),('base',1),('base',2),('off',2),('on',1),('off',3),('base',3))

def tasks():
    rows=[]
    for variant,repeat in ORDER:
        c=screen_task('fixed','on' if variant=='on' else 'off','r1')['config']
        if variant=='base':
            c.update(preset='base',execution='host',discrete_bvh_refit=False,trace_velocity=False)
        c.update(steps=59,timeout_seconds=120,contact_pool_validate=False,diagnostics=[])
        rows.append({'name':f'fixed_{variant}_r{repeat}','variant':variant,'repeat':repeat,
                     'binary':'base' if variant=='base' else 'active','scene_key':'fixed','stage':'quality_diagnosis',
                     'config':c,'expanded_config_sha256':digest(expand(c))})
    return rows

def plan():
    return {'schema':'fixed_quality_diagnosis_plan.v1','runs':tasks(),'finite_run_budget':7,
            'scene':'cloth_fixed_bunny_l','steps':59,'from_zero':True,'contact_window':[22,59],
            'purpose':'Diagnose whether active-off and official baseline also exceed the original fixed-cloth p99 bound.',
            'original_guard_remains_failed':True,'thresholds_unchanged':True,
            'stop_on_hard_failure':True,'stop_on_frozen_bound_exceedance':False,
            'performance_certified':False,'quality_certified':False,
            'allow_performance_screen':False,'automatic_long_run':False,
            'baseline_capabilities':{'resolved_config':False,'actual_velocity':False},
            'interpretation':'Bounds are diagnostic outcomes, never widened. A single on run cannot establish on repeatability. Timings are observations, not speedup certification.'}
