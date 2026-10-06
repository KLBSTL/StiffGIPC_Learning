"""Finite Windows diagnostic rounds; launching the next round is always explicit."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/bench'),str(ROOT/'tools/diagnostic')]
from config import expand,digest
from plans import task as cloth_task

ROUNDS={'fixed_r1':(('off',1),('base',1)), 'fixed_r2':(('base',2),('off',2)),
        'fixed_r3':(('on',1),('off',3),('base',3)),
        'hang_r1':(('off',1),('on',1)), 'hang_r2':(('on',2),('off',2)),
        'hang_r3':(('off',3),('on',3))}

def tasks(stage):
    if stage not in ROUNDS:raise ValueError('Unknown finite diagnostic round')
    scene=stage.split('_')[0];rows=[]
    for variant,repeat in ROUNDS[stage]:
        c=cloth_task(scene,'on' if variant=='on' else 'off','r1')['config']
        if variant=='base':c.update(preset='base',execution='host',discrete_bvh_refit=False,trace_velocity=False)
        c.update(diagnostics=[],profile='none',contact_pool_validate=False,timeout_seconds=120)
        rows.append({'name':f'{scene}_{variant}_r{repeat}','scene_key':scene,'variant':variant,
                     'repeat':repeat,'binary':'base' if variant=='base' else 'active','config':c,
                     'expanded_config_sha256':digest(expand(c))})
    return rows

def predecessor(stage):
    scene,repeat=stage.split('_r');return None if repeat=='1' else scene+'_r'+str(int(repeat)-1)

def plan():
    return {'schema':'windows_cloth_diagnosis.v1','rounds':{s:tasks(s) for s in ROUNDS},
            'finite_run_budget':13,'from_zero':True,'contact_windows':{'fixed':[22,59],'hang':[41,50]},
            'prefixes':{'fixed':[1,59],'hang':[1,51]},'old_quality_gate_unchanged':True,
            'fixed_bound_excess_is_diagnostic_outcome':True,'automatic_next_round':False,
            'quality_certified':False,'performance_certified':False,'automatic_long_run':False,
            'next_round':'Explicit CLI plus previous analysis SHA after review; hard/configuration/resource failures block continuation. No diagnostic receipt grants a performance-screen gate.'}
