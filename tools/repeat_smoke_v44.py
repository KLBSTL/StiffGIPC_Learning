"""Preserve failed bitwise gate; measure ordinary run-to-run variability separately."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
original=json.loads((ROOT/'reports/V44_smoke.json').read_text())
for name,stage in [('off_repeat1',0),('off_repeat2',0),('on_repeat1',1)]:
    command=original[stage]['command'].copy()
    command[command.index('--name')+1]='autodl_perf_v44_'+name
    with (ROOT/'reports'/f'{name}.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=280)
import numpy as np
runs={name:ROOT/'runs/autodl'/('autodl_perf_v44_'+name) for name in ['off','on','off_repeat1','off_repeat2','on_repeat1']}
arrays={name:np.fromfile(path/'final.bin',dtype='<f8') for name,path in runs.items()}
comparisons=[]
for i,(name,x) in enumerate(arrays.items()):
    for other,y in list(arrays.items())[i+1:]:
        comparisons.append({'a':name,'b':other,'bitwise_equal':bool(np.array_equal(x.view('u8'),y.view('u8'))),
            'max_abs':float(np.max(np.abs(x-y))),'relative_l2':float(np.linalg.norm(x-y)/np.linalg.norm(x))})
summaries=[]
for name,path in runs.items():
    stats=json.loads((path/'output/stats.json').read_text())
    solves=[n['pcg'] for f in stats['frames'] for n in f.get('newton',[]) if 'pcg' in n]
    summaries.append({'name':name,'pcg_iterations':[x['iterations'] for x in solves],
        'limits':sum(bool(x.get('iteration_limit')) for x in solves),'breakdowns':sum(bool(x.get('breakdown')) for x in solves)})
report={'original_bitwise_gate_passed':False,'comparisons':comparisons,'runs':summaries}
(ROOT/'reports/SMOKE_REPEAT_V44.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
