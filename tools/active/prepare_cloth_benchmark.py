"""Generate the predeclared cloth benchmark; does not run GPU work."""
import json
from config import ROOT

SCENES={'hang':'cloth_hang_l','sphere':'cloth_sphere7_l','bunny':'cloth_fixed_bunny_l'}
ARMS={
    'stiff':({'preset':'base'},'base'),
    'toi_serial':({'preset':'toi'},'active'),
    'toi_warp':({'preset':'toi','mas_restrict':'warp'},'active'),
    'ipc_warp':({'preset':'toi','backend':'ipc','mas_restrict':'warp'},'active'),
}

def task(scene,key,arm,repeat,steps=100):
    cfg,binary=ARMS[arm]
    return {'name':f'cloth_restrict_{key}_{arm}_{repeat}','arm':key+'_'+arm,'binary':binary,
            'config':cfg|{'scene':scene,'steps':steps,'timeout_seconds':120 if steps==100 else 30,
                          'trace_velocity':False},'scene_key':key,'variant':arm,'repeat':repeat}

if __name__=='__main__':
    smoke=[];runs=[]
    for key,scene in SCENES.items():
        for arm in ('stiff','toi_warp'):smoke.append(task(scene,key,arm,'smoke',3))
    order=['stiff','toi_serial','toi_warp','ipc_warp']
    for repeat in range(1,4):
        for key,scene in SCENES.items():
            rotation=order[repeat-1:]+order[:repeat-1]
            for arm in rotation:runs.append(task(scene,key,arm,'r'+str(repeat)))
    for label,items,stop in [('smoke',smoke,True),('timing',runs,False)]:
        out=ROOT/f'configs/active/cloth_restrict_{label}.json'
        with out.open('x') as f:json.dump({'report':f'reports/active/CLOTH_RESTRICT_{label.upper()}_BATCH.json',
                                        'stop_on_failure':stop,'runs':items},f,indent=2)
    print(json.dumps({'smoke_runs':len(smoke),'timing_runs':len(runs),'frames':100,'dt':.01,'repeats':3}))
