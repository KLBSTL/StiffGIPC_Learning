import hashlib, json, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SHA='389f8a52a29606ccaf5640607dd1e38987500988'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7892','http':'http://127.0.0.1:7892'}))
def get(url):
    return opener.open(urllib.request.Request(url,headers={'User-Agent':'Stiff-TOI-integration'}),timeout=90).read()
tree=json.loads(get(f'https://api.github.com/repos/wiso-enoji/libuipc/git/trees/{SHA}?recursive=1'))
paths=[t['path'] for t in tree['tree'] if t['type']=='blob' and
       (t['path']=='LICENSE' or '/al_' in t['path'].lower() or
        ('AL' in t['path'] and t['path'].endswith('README.md')))]
out=ROOT/'references'/'paper_al';out.mkdir(exist_ok=True)
(out/'tree.json').write_text(json.dumps(tree),encoding='utf-8')
manifest={'repository':'https://github.com/wiso-enoji/libuipc','commit':SHA,'files':[]}
for path in paths:
    data=get(f'https://raw.githubusercontent.com/wiso-enoji/libuipc/{SHA}/{path}')
    target=out/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    manifest['files'].append({'path':path,'sha256':hashlib.sha256(data).hexdigest()})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'selected_paths':paths}))
