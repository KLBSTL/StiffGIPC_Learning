"""Verify frozen identities and remove only uploaded/downloaded duplicate v39 archives."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/root/autodl-tmp/stiff_toi_cudagraph_20260929').resolve()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=ROOT/'v39_20261003/reports/final_remote_verification.json'
assert not report.exists()
result={'frozen':[],'removed_duplicate_archives':[],'free_before':shutil.disk_usage(ROOT).free}
for version in [36,37,39]:
    root=ROOT/f'v{version}_20261003'
    manifest=json.loads((root/f'manifests/perf_v{version}_autodl.json').read_text())
    for f in manifest['files']:assert sha(root/f['path'])==f['sha256'],f['path']
    for name,entry in manifest['binaries'].items():assert sha(root/name)==entry['sha256'],name
    result['frozen'].append({'version':version,'files':len(manifest['files']),'binaries':manifest['binaries']})
for name,expected in {
    'autodl_perf_v39_20261003.tar.gz':'a8f466ead979738a73e40200c03106d232d252f7391c8f4203683fbfcd382c43',
    'v39_20261003/fixtures_v39.tar.gz':'8fceb32279d062063d49605a8142eb6f48bb3720780593959a269bc5675bab16',
    'v39_20261003/results_v39.tar.gz':'a16a5d2b5caa2039a7b37e1d021f468da0759a65d6d177462ecdc19c22e4682f',
}.items():
    p=ROOT/name
    assert ROOT in p.resolve(strict=True).parents and not p.is_symlink()
    assert sha(p)==expected
    result['removed_duplicate_archives'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':expected,
        'retained':'Verified complete local archive and remote extracted source/results'})
    p.unlink()
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
result['free_after']=shutil.disk_usage(ROOT).free
report.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
