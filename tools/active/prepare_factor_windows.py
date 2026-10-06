"""Generate the finite local window comparison after fixed-system gates pass."""
import json
from config import ROOT

if __name__=='__main__':
    scenes=[('hang','cloth_hang_l',21),('fixed_bunny','cloth_fixed_bunny_l',45),('mixed','bunny_cloth_bunny_l',35)]
    arms=['stiff','triangular','factor_inverse'];runs=[]
    for repeat in range(1,4):
        for key,scene,steps in scenes:
            order=arms[repeat-1:]+arms[:repeat-1]
            for arm in order:
                config=({'preset':'base'} if arm=='stiff' else
                        {'preset':'toi','mas_restrict':'warp','mas_factor_action':arm})
                config.update(scene=scene,steps=steps,timeout_seconds=120,trace_velocity=False)
                runs.append({'name':f'factor_window_{key}_{arm}_r{repeat}','arm':key+'_'+arm,
                             'scene_key':key,'variant':arm,'repeat':repeat,
                             'binary':'base' if arm=='stiff' else 'active','config':config})
    target=ROOT/'configs/active/factor_windows.json'
    with target.open('x') as f:json.dump({'report':'reports/active/FACTOR_WINDOW_BATCH.json',
                                        'stop_on_failure':False,'runs':runs},f,indent=2)
    print(json.dumps({'plan':str(target),'runs':len(runs),'maximum_run_seconds':120}))
