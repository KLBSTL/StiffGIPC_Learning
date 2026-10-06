import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path('/root/autodl-tmp/stiff_toi_cudagraph_20260929').resolve()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'identities':[],'removed_verified_duplicates':[],'free_before':shutil.disk_usage(ROOT).free}
for folder,manifest in [('v42_20261003','mas_replay_v42'),('v42a_20261003','mas_replay_v42a')]:
    root=ROOT/folder;m=json.loads((root/f'manifests/{manifest}.json').read_text())
    for f in m['files']:assert sha(root/f['path'])==f['sha256']
    result['identities'].append({'folder':folder,'files':len(m['files']),'binary_sha256':sha(root/'builds/replay/mas_replay')})
for name,expected in {'replay_v42_results.tar.gz':'3142dfb407466ae51a2f5f1dd09d31b46f10064c9d70d41bf4002862024e0a74','replay_v42_strict.tar.gz':'e29ef869437edb8e4e07a9f53acd15550be7cba4df0823751aa6dedac70ceefd','initial_v42.tar.gz':'f935d99c889144a296b9b1900466ecdfaf7445703279c6b4ae7bf5e42fd9ab83'}.items():
    path=ROOT/'v42a_20261003'/name;assert ROOT in path.resolve(strict=True).parents and not path.is_symlink()
    assert sha(path)==expected;result['removed_verified_duplicates'].append({'path':str(path),'bytes':path.stat().st_size,'sha256':expected,'retained':'Verified local archive and remote extracted originals'});path.unlink()
result['free_after']=shutil.disk_usage(ROOT).free
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
target=ROOT/'v42a_20261003/reports/final_verification.json';assert not target.exists();target.write_text(json.dumps(result,indent=2));print(json.dumps(result))
