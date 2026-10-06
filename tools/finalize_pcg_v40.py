import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/root/autodl-tmp/stiff_toi_cudagraph_20260929').resolve()
report=ROOT/'v40_20261003/reports/final_verification.json';assert not report.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'identities':[],'removed_verified_duplicates':[],'free_before':shutil.disk_usage(ROOT).free}
for version in [37,39]:
    root=ROOT/f'v{version}_20261003';m=json.loads((root/f'manifests/perf_v{version}_autodl.json').read_text())
    for f in m['files']:assert sha(root/f['path'])==f['sha256']
    for name,f in m['binaries'].items():assert sha(root/name)==f['sha256']
    result['identities'].append({'version':version,'files':len(m['files']),'binaries':m['binaries']})
root=ROOT/'v40_20261003';m=json.loads((root/'manifests/mas_replay_v40.json').read_text())
for f in m['files']:assert sha(root/f['path'])==f['sha256']
result['replay_binary_sha256']=sha(root/'builds/replay/mas_replay')
for relative,expected in {
    'mas_replay_v40.tar.gz':'5d9e17b0add748765f66bae1ee935914e52be27a7729946ca36bc2bc3add4234',
    'v39_20261003/controls_v40.tar.gz':'398491e7704457a6ac648c3b8311da8a8a95e5460038bbcac028a87fd691dfe9',
    'v40_20261003/replay_v40_results.tar.gz':'92884d37c8fa5e4174339e269037dbf9d67b6a4356b9ec12fbfcc78340a12878',
}.items():
    path=ROOT/relative;assert ROOT in path.resolve(strict=True).parents and not path.is_symlink()
    assert sha(path)==expected
    result['removed_verified_duplicates'].append({'path':str(path),'bytes':path.stat().st_size,'sha256':expected,
        'retained':'Local verified archive and remote extracted original data'})
    path.unlink()
result['free_after']=shutil.disk_usage(ROOT).free
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
report.write_text(json.dumps(result,indent=2));print(json.dumps(result))
