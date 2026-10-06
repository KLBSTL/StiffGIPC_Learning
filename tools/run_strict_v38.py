import hashlib,json,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=r/'manifests/mas_replay_v38_strict.json';manifest=json.loads(m.read_text())
assert all(sha(r/f['path'])==f['sha256'] for f in manifest['files'])
prefix=r.parent/'v37_20261003/runs/autodl/autodl_perf_v37_bunny_mas/mas_audit_failure'
inputs={p.name:sha(p) for p in prefix.parent.glob(prefix.name+'*.bin')}
for mode in [4,5]:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
 out=r/f'runs/strict_m{mode}';assert not out.exists();out.mkdir()
 binary=r/'builds/replay_strict/mas_replay'
 cmd=[str(binary),str(prefix),str(out),str(mode)]
 (out/'requested.json').write_text(json.dumps({'command':cmd,'binary_sha256':sha(binary),'manifest_sha256':sha(m),'inputs':inputs},indent=2))
 start=time.monotonic()
 with (out/'run.log').open('wb') as log:
  try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
  except subprocess.TimeoutExpired:code='timeout'
 (out/'completion.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.monotonic()-start},indent=2))
 assert code==0
 print((out/'result.json').read_text(),flush=True)
 assert all(sha(prefix.parent/name)==h for name,h in inputs.items())
