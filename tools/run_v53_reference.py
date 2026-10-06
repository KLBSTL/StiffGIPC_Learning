"""Run the preserved instrumented Stiff base with verified relevant provenance."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--name',choices=['v53_base_reference','v53_base_repeat'],default='v53_base_reference');a=p.parse_args()
m=json.loads((ROOT/'manifests/perf_v34.json').read_text())
rows=[r for r in m['files'] if r['path'].startswith('sources/stiff_base/')]
assert len(rows)==274
for r in rows:assert sha(ROOT/r['path'])==r['sha256'],r['path']
exe='builds/local-base/Release/gipc.exe'
assert sha(ROOT/exe)==m['binaries'][exe]['sha256']
out=ROOT/'reports'/f'V53_REFERENCE_IDENTITY_{a.name}.json';assert not out.exists()
out.write_text(json.dumps({'base_commit':m['base_commit'],'verified_source_files':len(rows),
    'exe_sha256':sha(ROOT/exe),'provenance_manifest':'manifests/perf_v34.json',
    'scope':'Only base source and base binary verified; fused v34 is not used.'},indent=2))
command=[sys.executable,str(ROOT/'tools/run_local.py'),'--arm','base','--scene','bunny_cloth_bunny_l',
    '--steps','100','--timeout','600','--name',a.name,'--dt','.01','--pcg-tol','1e-4',
    '--tol','.01','--trace','--substeps','--physics','--audit-pcg','--quality-only',
    '--ccd-pair-limit','10000000','--manifest',str(ROOT/'manifests/perf_v34.json')]
env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
with (ROOT/'reports'/f'V53_{a.name}_runner.log').open('x') as log:
    r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=690)
assert sha(ROOT/exe)==m['binaries'][exe]['sha256']
print((ROOT/'runs/local'/a.name/'result.json').read_text(),flush=True)
raise SystemExit(r.returncode)
