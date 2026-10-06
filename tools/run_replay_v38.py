"""Frozen-input serial GPU experiments; each invocation has a hard timeout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--case',choices=['smoke','initial','failure'],required=True)
p.add_argument('--modes',default='0,1,2,3,4,5')
a=p.parse_args()
r=Path(__file__).resolve().parents[1]
previous=r.parent/'v37_20261003/runs/autodl'
prefixes={'smoke':previous/'autodl_perf_v37_mas_smoke/mas_audit_initial',
          'initial':previous/'autodl_perf_v37_bunny_mas/mas_audit_initial',
          'failure':previous/'autodl_perf_v37_bunny_mas/mas_audit_failure'}
prefix=prefixes[a.case]
manifest=json.loads((r/'manifests/mas_replay_v38_repaired.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(r/f['path'])==f['sha256'] for f in manifest['files'])
binary=r/'builds/replay/mas_replay'
inputs={x.name:sha(x) for x in prefix.parent.glob(prefix.name+'*.bin')}
for mode in map(int,a.modes.split(',')):
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    assert not active,active
    out=r/f'runs/{a.case}_m{mode}';assert not out.exists();out.mkdir()
    command=[str(binary),str(prefix),str(out),str(mode)]
    req={'command':command,'binary_sha256':sha(binary),'input_sha256':inputs,
         'manifest_sha256':sha(r/'manifests/mas_replay_v38_repaired.json'),'runner_sha256':sha(Path(__file__)),
         'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.free','--format=csv,noheader'],text=True).strip()}
    (out/'requested.json').write_text(json.dumps(req,indent=2))
    start=time.monotonic()
    with (out/'run.log').open('wb') as log:
        try:status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
        except subprocess.TimeoutExpired:status='timeout'
    (out/'completion.json').write_text(json.dumps({'exit_code':status,'wall_seconds':time.monotonic()-start},indent=2))
    assert status==0,(out,status)
    result=json.loads((out/'result.json').read_text())
    print(json.dumps({'case':a.case,'mode':result['mode'],'inverse32_identical':result['native_inverse_recomputed_bitwise'],
          'max_reference_difference':max(v['versus_v37']['relative'] for v in result['operators']),
          'max_repeat_difference':max(d['relative'] for v in result['operators'] for d in v['repeats']),
          'solves':[{k:s[k] for k in ['tolerance','repeat','iterations','reason','true_relative_residual']} for s in result['solves']]}),flush=True)
    assert all(sha(prefix.parent/name)==digest for name,digest in inputs.items())
