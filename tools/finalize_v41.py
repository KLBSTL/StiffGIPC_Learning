"""Run only after local archive SHA checks; retain extracted originals and local archives."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/root/autodl-tmp/stiff_toi_cudagraph_20260929').resolve()
target=ROOT/'v41_20261003/reports/final_verification.json';assert not target.exists()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
result={'identities':[],'removed_verified_duplicates':[],'free_before':shutil.disk_usage(ROOT).free}
for version in [37,39,41]:
    root=ROOT/f'v{version}_20261003';m=json.loads((root/f'manifests/perf_v{version}_autodl.json').read_text())
    for f in m['files']:assert sha(root/f['path'])==f['sha256'],f['path']
    for name,f in m['binaries'].items():assert sha(root/name)==f['sha256'],name
    result['identities'].append({'version':version,'files':len(m['files']),'binaries':m['binaries']})
for name,expected in {
    'fixtures_v41.tar.gz':'8b0e0ddeac5d7d913aadef35d81ece3925a1c598549cf483668fa19f0a4001cd',
    'results_v41.tar.gz':'f8091546ea2b0d4ddcc5244a099fd20675fa91dfb2453d056d92b954407d1360',
}.items():
    path=ROOT/'v41_20261003'/name
    assert ROOT in path.resolve(strict=True).parents and not path.is_symlink() and path.is_file()
    assert sha(path)==expected
    result['removed_verified_duplicates'].append({'path':str(path),'bytes':path.stat().st_size,'sha256':expected,
        'retained':'Local SHA-verified archive and remote extracted originals'})
    path.unlink()
result['free_after']=shutil.disk_usage(ROOT).free
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
target.write_text(json.dumps(result,indent=2));print(json.dumps(result))
