from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
src=ROOT.parent/'cross_method_benchmark_20260927/build-ipc/_deps/tight_inclusion-src'
dst=ROOT/'references/tight_inclusion'
dst.mkdir(parents=True,exist_ok=True)
manifest={'origin':str(src),'files':[]}
for p in [*sorted((src/'tight_inclusion').glob('*.hpp')),*sorted((src/'tight_inclusion').glob('*.cpp')),*sorted(src.glob('LICENSE*'))]:
    relative=p.relative_to(src);out=dst/relative;out.parent.mkdir(parents=True,exist_ok=True)
    data=p.read_bytes();out.write_bytes(data)
    manifest['files'].append({'path':str(relative),'sha256':hashlib.sha256(data).hexdigest()})
(dst/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'frozen_files':len(manifest['files'])}))
