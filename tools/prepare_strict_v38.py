"""Preserve the primary binary; build a tolerance-only follow-up identity."""
import hashlib,json,shutil,tarfile
from pathlib import Path
r=Path(__file__).resolve().parents[1]
src=r/'sources/mas_replay_v38';dst=r/'sources/mas_replay_v38_strict'
assert not dst.exists();dst.mkdir()
for name in ['native_kernels.cuh','CMakeLists.txt']:shutil.copyfile(src/name,dst/name)
text=(src/'replay.cu').read_text()
needle='for(double tol:{1e-4,1e-8,1e-12})'
assert text.count(needle)==1
(dst/'replay.cu').write_text(text.replace(needle,'for(double tol:{1e-14,1e-16})'))
manifest=json.loads((r/'manifests/mas_replay_v38_repaired.json').read_text())
files=list(dst.iterdir())+[Path(__file__),r/'tools/run_strict_v38.py']
manifest['scope']='Tolerance-only follow-up, original v38 binary preserved'
manifest['files'] += [{'path':p.relative_to(r).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
m=r/'manifests/mas_replay_v38_strict.json';assert not m.exists();m.write_text(json.dumps(manifest,indent=2))
archive=r/'bundles/mas_replay_v38_strict.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for p in files+[m]:t.add(p,arcname=p.relative_to(r).as_posix())
print(hashlib.sha256(archive.read_bytes()).hexdigest())
