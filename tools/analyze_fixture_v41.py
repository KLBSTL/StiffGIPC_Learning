"""CPU hierarchy action using the exact exported v41 GPU double inverse."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_perf_v37 import unpack,difference

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();task=Path(__file__).resolve().parents[1]
prefixes={'smoke':task/'downloads/autodl_perf_v37_20261003/runs/autodl/autodl_perf_v37_mas_smoke/mas_audit_initial',
          'default':task/'downloads/autodl_perf_v39_20261003/runs/autodl/autodl_perf_v39_wide_bunny/mas_audit_failure',
          'strict':task/'downloads/autodl_perf_v39_20261003/runs/autodl/autodl_perf_v39_wide_bunny_strict/mas_audit_failure'}
records=[]
for case,prefix in prefixes.items():
    folder=a.root/case;raw=json.loads((folder/'fixture.json').read_text())
    meta=json.loads(Path(str(prefix)+'_meta.json').read_text())
    local=next(x for x in meta['local_preconditioners'] if x['kind']=='MAS_full_owned_buffers')
    nodes,mapped,levels,_,clusters,*_=local['dimensions'];offset=3*local['offset'];count=local['buffers']['d_inverseMatMas']['count'];bank=clusters//count
    read=lambda name,dtype:np.fromfile(str(prefix)+'_mas_'+name+'.bin',dtype)
    part=read('d_partId_map_real','<i4')[:mapped];real=read('d_real_map_partId','<i4')[:nodes]
    coarse=read('d_coarseTable','<i4').reshape(-1,6)[:nodes,:levels-1]
    blocks=unpack(folder/'inverse64.bin','<f8',count,bank)
    eig=np.linalg.eigvalsh((blocks+blocks.transpose(0,2,1))/2)
    checks=[]
    for k in range(10):
        v=np.fromfile(str(prefix)+f'_v{k}.bin','<f8')[offset:].reshape(nodes,3)
        lr=np.zeros((clusters,3));valid=part>=0;lr[np.flatnonzero(valid)]=v[part[valid]]
        for level in range(levels-1):np.add.at(lr,coarse[:,level],v)
        lz=np.einsum('nij,nj->ni',blocks,lr.reshape(count,bank*3)).reshape(clusters,3)
        expected=lz[real].copy()
        for level in range(levels-1):expected+=lz[coarse[:,level]]
        actual=np.fromfile(folder/f'p2_v{k}.bin','<f8')
        entry={'index':k,'cpu_hierarchy':difference(expected.ravel(),actual),
               'native_off_restore':difference(np.fromfile(folder/f'p0_v{k}.bin','<f8'),np.fromfile(folder/f'p3_v{k}.bin','<f8'))}
        previous=task/'downloads/replay_v40_20261003/runs'/f'{case}_m5'/f'Mv{k}.bin'
        if previous.exists():entry['v40_full64']=difference(np.fromfile(previous,'<f8')[offset:],actual)
        checks.append(entry)
    maximum=max(x['cpu_hierarchy']['relative'] for x in checks)
    row={'case':case,'fixture':raw,'checks':checks,'max_cpu_relative':maximum,
         'inverse_min_eigenvalue':float(eig.min()),'inverse_nonpositive_blocks':int(np.count_nonzero(eig[:,0]<=0)),
         'cpu_operator_1e10_passed':maximum<=1e-10}
    records.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('fixture','checks')}),flush=True)
a.output.write_text(json.dumps(records,indent=2,allow_nan=False))
