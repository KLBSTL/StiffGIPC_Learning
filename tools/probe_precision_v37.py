"""Separate CPU FP64 inversion, float32 storage rounding, and captured GPU inverse."""
import json
from pathlib import Path
import numpy as np
from analyze_perf_v37 import unpack

root=Path(__file__).resolve().parents[1]
prefix=root/'downloads/autodl_perf_v37_20261003/runs/autodl/autodl_perf_v37_bunny_mas/mas_audit_failure'
meta=json.loads(Path(str(prefix)+'_meta.json').read_text())
local=next(x for x in meta['local_preconditioners'] if x['kind']=='MAS_full_owned_buffers')
count=local['buffers']['d_inverseMatMas']['count'];bank=local['dimensions'][4]//count
h=unpack(str(prefix)+'_mas_d_inverseMatMas.bin','<f8',count,bank)
m=unpack(str(prefix)+'_mas_d_precondMatMas.bin','<f4',count,bank)
width=3*bank;d=np.arange(width)
h[:,d,d]=np.where(h[:,d,d]==0,1,h[:,d,d])
identity=np.eye(width)
error=lambda a,b:np.linalg.norm(a@b-identity,axis=(1,2))/np.sqrt(width)
gpu_error=error(h,m)
selected=sorted(set(np.argsort(gpu_error)[-10:].tolist()+[1276]))
records=[]
for i in selected:
    exact=np.linalg.inv((h[i]+h[i].T)/2)
    rounded=exact.astype(np.float32).astype(np.float64)
    records.append({'block':i,'local_condition_2':float(np.linalg.cond(h[i])),
        'inverse_product_error_fp64':float(error(h[i:i+1],exact[None])[0]),
        'inverse_product_error_fp32_rounded':float(error(h[i:i+1],rounded[None])[0]),
        'inverse_product_error_captured_gpu':float(gpu_error[i]),
        'gpu_inverse_relative_vs_cpu_fp64':float(np.linalg.norm(m[i]-exact)/np.linalg.norm(exact)),
        'gpu_inverse_relative_vs_cpu_rounded':float(np.linalg.norm(m[i]-rounded)/np.linalg.norm(exact))})
target=root/'reports/AUTODL_PERF_V37_PRECISION_PROBE.json'
assert not target.exists()
target.write_text(json.dumps({'scope':'CPU local-block diagnostic only; no proof of global PCG failure causality or GPU speed.',
                             'blocks':records},indent=2))
print(json.dumps(records,indent=2))
