"""Predeclare the requested two-scene, four-configuration comparison."""
import json
from config import ROOT,expand

SCENES={'hang':'cloth_hang_l','fixed_bunny':'cloth_fixed_bunny_l'}
ARMS={
    'base':({'preset':'base'},'base'),
    'graph':({'preset':'graph'},'active'),
    'toi':({'preset':'toi','execution':'host'},'active'),
    'graph_toi':({'preset':'toi'},'active'),
}
def task(key,arm,repeat,steps):
    cfg,binary=ARMS[arm]
    cfg=cfg|{'scene':SCENES[key],'steps':steps,'dt':.01,
        'timeout_seconds':120 if steps==100 else 20,'trace_velocity':False,
        'refit':False,'batch':False,'reuse':False,'mas_restrict':'serial'}
    expanded=expand(cfg)
    assert expanded['refit']==expanded['batch']==expanded['reuse']==False
    return {'name':f'fourway_cloth_20261005_{key}_{arm}_{repeat}',
        'arm':key+'_'+arm,'scene_key':key,'variant':arm,'repeat':repeat,
        'binary':binary,'config':cfg}

if __name__=='__main__':
    smoke=[task(key,arm,'smoke',2) for key in SCENES for arm in ARMS]
    timing=[]
    orders=[['base','graph','toi','graph_toi'],['graph_toi','toi','graph','base'],
            ['graph','graph_toi','base','toi']]
    for repeat,order in enumerate(orders,1):
        keys=list(SCENES) if repeat!=2 else list(reversed(SCENES))
        for key in keys:
            for arm in order:timing.append(task(key,arm,'r'+str(repeat),100))
    for kind,runs in [('smoke',smoke),('timing',timing)]:
        with (ROOT/f'configs/active/fourway_cloth_{kind}_20261005.json').open('x') as f:
            json.dump({'report':f'reports/active/FOURWAY_CLOTH_{kind.upper()}_BATCH_20261005.json',
                'stop_on_failure':kind=='smoke','runs':runs},f,indent=2)
    print(json.dumps({'scenes':SCENES,'smokes':len(smoke),'timing_runs':len(timing),'frames':100,'repeats':3}))
