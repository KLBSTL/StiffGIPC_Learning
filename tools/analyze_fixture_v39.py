"""Independent CPU P^T B P check of the integrated MAS fixture."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_perf_v37 import unpack, difference

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
task=Path(__file__).resolve().parents[1]
old=task/'downloads/autodl_perf_v37_20261003/runs/autodl'
prefixes={'smoke':old/'autodl_perf_v37_mas_smoke/mas_audit_initial',
          'initial':old/'autodl_perf_v37_bunny_mas/mas_audit_initial',
          'failure':old/'autodl_perf_v37_bunny_mas/mas_audit_failure'}
records=[]
for case,prefix in prefixes.items():
    raw=json.loads((a.root/case/'fixture.json').read_text())
    meta=json.loads(Path(str(prefix)+'_meta.json').read_text())
    local=next(x for x in meta['local_preconditioners'] if x['kind']=='MAS_full_owned_buffers')
    nodes,mapped,levels,_,clusters,*_=local['dimensions'];offset=3*local['offset']
    count=local['buffers']['d_inverseMatMas']['count'];bank=clusters//count
    read=lambda name,dtype:np.fromfile(str(prefix)+'_mas_'+name+'.bin',dtype)
    part=read('d_partId_map_real','<i4')[:mapped]
    real=read('d_real_map_partId','<i4')[:nodes]
    coarse=read('d_coarseTable','<i4').reshape(-1,6)[:nodes,:levels-1]
    stored=unpack(str(prefix)+'_mas_d_precondMatMas.bin','<f4',count,bank)
    checks=[]
    for k in range(10):
        v=np.fromfile(str(prefix)+f'_v{k}.bin','<f8')[offset:].reshape(nodes,3)
        lr=np.zeros((clusters,3));valid=part>=0;lr[np.flatnonzero(valid)]=v[part[valid]]
        for level in range(levels-1):np.add.at(lr,coarse[:,level],v)
        lz=np.einsum('nij,nj->ni',stored,lr.reshape(count,bank*3)).reshape(clusters,3)
        expected=lz[real].copy()
        for level in range(levels-1):expected+=lz[coarse[:,level]]
        wide=np.fromfile(a.root/case/f'p1_v{k}.bin','<f8')
        native=np.fromfile(a.root/case/f'p0_v{k}.bin','<f8')
        restored=np.fromfile(a.root/case/f'p2_v{k}.bin','<f8')
        row={'index':k,'cpu_hierarchy':difference(expected.ravel(),wide),
             'off_on_off':difference(native,restored)}
        replay=task/'downloads/mas_replay_v38/runs'/f'{case}_m4'/f'Mv{k}.bin'
        if replay.exists():row['v38_m4']=difference(np.fromfile(replay,'<f8')[offset:],wide)
        checks.append(row)
    worst=max(x['cpu_hierarchy']['relative'] for x in checks)
    passed=worst<1e-10 and all(raw[k] for k in ['growth_changes_graph_signature','precision_changes_graph_signature','off_signature_restored'])
    records.append({'case':case,'passed':passed,'max_cpu_difference':worst,'fixture':raw,'checks':checks})
    print(json.dumps({'case':case,'passed':passed,'max_cpu_difference':worst}),flush=True)
a.output.write_text(json.dumps(records,indent=2,allow_nan=False))
assert all(x['passed'] for x in records)
