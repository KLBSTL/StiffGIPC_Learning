"""Verify frozen identities; remove verified transfer duplicates and deduplicate immutable factors."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path('/root/autodl-tmp/stiff_toi_cudagraph_20260929').resolve();r=ROOT/'v43_20261003'
target=r/'reports/final_verification.json';assert not target.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'identities':[],'removed_verified_duplicates':[],'factor_hardlinks':[],'free_before':shutil.disk_usage(ROOT).free}
for version in [37,39,41,43]:
    folder=ROOT/f'v{version}_20261003';m=json.loads((folder/f'manifests/perf_v{version}_autodl.json').read_text())
    for f in m['files']:assert sha(folder/f['path'])==f['sha256'],f['path']
    for p,entry in m['binaries'].items():assert sha(folder/p)==entry['sha256'],p
    result['identities'].append({'version':version,'files':len(m['files']),'binaries':m['binaries']})
for name,expected in {'fixtures_v43.tar.gz':'0ce010a469e94e43a7f39dba165d8be0b355be79fe6307b5654ced84c284935f','results_v43.tar.gz':'30536c0bc406388e48f64296a6d5d28ce79da22776fa86a327ef515e0c107e0d'}.items():
    p=r/name;assert ROOT in p.resolve(strict=True).parents and not p.is_symlink() and sha(p)==expected
    result['removed_verified_duplicates'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':expected,'retained':'Local SHA-verified archive and remote extracted originals'});p.unlink()
for folder,case,oldcase in [('fixtures','default','v39_default'),('fixtures','strict','v39_strict'),('fixtures_corrected','v41_graph','v41_graph'),('fixtures_corrected','v41_host','v41_host'),('fixtures_corrected','v41_wide','v41_wide')]:
    p=r/'runs'/folder/case/'factors.bin';original=ROOT/'v42a_20261003/runs'/f'{oldcase}_m6/factors.bin'
    assert ROOT in p.resolve(strict=True).parents and ROOT in original.resolve(strict=True).parents and not p.is_symlink() and not original.is_symlink()
    digest=sha(original)
    if sha(p)!=digest:result['factor_hardlinks'].append({'path':str(p),'skipped':'Different bytes; original retained'});continue
    temp=p.with_name('factors.verified-hardlink.tmp');assert not temp.exists();os.link(original,temp);os.replace(temp,p);assert sha(p)==digest
    result['factor_hardlinks'].append({'path':str(p),'shared_with':str(original),'sha256':digest,'bytes':p.stat().st_size})
result['free_after']=shutil.disk_usage(ROOT).free
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
target.write_text(json.dumps(result,indent=2));print(json.dumps(result))
