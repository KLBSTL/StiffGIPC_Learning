"""Preservation/provenance check for official base, v31 and new port."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=read(ROOT/'IMPLEMENTATION_MANIFEST.json')
    new=read(ROOT/'manifests/robust_port_v32.json')
    for m in [old,new]:
        for f in m['files']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
    previous={'builds/local-base/Release/gipc.exe':'1c175da5a664bbc98657ddb6690a9902d821732c19e9b18f1f3ce0e868e05205',
        'builds/local-fused-v31/Release/gipc.exe':'7ff7e39ce10b3efa3f229ac6c2af707bb0f11da12321100df69301f297dcd9eb'}
    for path,expected in previous.items():assert sha(ROOT/path)==expected,path
    for path,identity in new['binaries'].items():assert sha(ROOT/path)==identity['sha256'],path
    report={'old_files_verified':len(old['files']),'new_files_verified':len(new['files']),
        'old_source_digest':old['source_digest'],'new_source_digest':new['source_digest'],
        'old_binaries_unchanged':True,'new_binaries':new['binaries']}
    (ROOT/'reports/ROBUST_V32_FREEZE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='new_binaries'}))
if __name__=='__main__':main()
