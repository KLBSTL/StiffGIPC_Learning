"""Six captured systems, including historic indefinite inverse failures."""
import json,os,subprocess
from pathlib import Path
from run_bunny_components import ROOT,read,sha
cases=[('smoke',37,'mas_smoke','initial'),('default',39,'wide_bunny','failure'),('strict',39,'wide_bunny_strict','failure'),
       ('v41_graph',41,'double_bunny','failure'),('v41_host',41,'double_bunny_host','failure'),('v41_wide',41,'wide_guard','failure')]
target=ROOT/'reports/V56_FIXTURES.json';assert not target.exists();rows=[]
exe=ROOT/'builds/local-v56/Release/gipc.exe'
for case,v,name,kind in cases:
    prefix=ROOT/f'downloads/autodl_perf_v{v}_20261003/runs/autodl/autodl_perf_v{v}_{name}/mas_audit_{kind}'
    assert Path(str(prefix)+'_meta.json').exists(),prefix
    out=ROOT/'runs/fixtures_v56'/case;assert not out.exists();out.parent.mkdir(exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
    env.update(GIPC_MAS_CHOLESKY_FIXTURE=str(prefix),GIPC_MAS_FIXTURE_OUTPUT=str(out))
    with (ROOT/f'reports/V56_fixture_{case}.log').open('x') as log:r=subprocess.run([str(exe)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    result=read(out/'fixture.json') if (out/'fixture.json').exists() else {}
    rows.append({'case':case,'prefix':str(prefix),'output':str(out),'exit_code':r.returncode,'result':result})
    target.write_text(json.dumps({'exe_sha256':sha(exe),'cases':rows},indent=2));print(json.dumps(rows[-1]),flush=True)
    assert r.returncode==0 and result['passed']
