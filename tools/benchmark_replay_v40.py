import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'manifests/mas_replay_v40.json').read_text())
for f in manifest['files']:assert hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()==f['sha256']
exe=ROOT/'builds/replay/mas_replay';binary_sha=hashlib.sha256(exe.read_bytes()).hexdigest()
report=ROOT/'reports/replay_matrix.json';assert not report.exists()
records=[]
for case,run in [('default','autodl_perf_v39_wide_bunny'),('strict','autodl_perf_v39_wide_bunny_strict')]:
    prefix=ROOT.parent/'v39_20261003/runs/autodl'/run/'mas_audit_failure'
    for mode in [0,4,5]:
        out=ROOT/'runs'/f'{case}_m{mode}';assert not out.exists()
        command=[str(exe),str(prefix),str(out),str(mode)]
        with (ROOT/'reports'/f'{case}_m{mode}.log').open('w') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        records.append({'case':case,'mode':mode,'command':command,'exit_code':proc.returncode,'binary_sha256':binary_sha})
        report.write_text(json.dumps(records,indent=2))
        assert proc.returncode==0
        raw=json.loads((out/'result.json').read_text())
        print(json.dumps({'case':case,'mode':mode,'solves':[{k:s[k] for k in ['tolerance','repeat','iterations','reason','true_relative_residual']} for s in raw['solves']]}),flush=True)
    wide=json.loads((ROOT/'runs'/f'{case}_m4/result.json').read_text())
    if any(s['reason']=='iteration_limit' for s in wide['solves']):
        out=ROOT/'runs'/f'{case}_m4_budget06';assert not out.exists()
        command=[str(exe),str(prefix),str(out),'4','1e-4','.6','1']
        with (ROOT/'reports'/f'{case}_m4_budget06.log').open('w') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        records.append({'case':case,'mode':4,'budget_ablation':True,'command':command,'exit_code':proc.returncode,'binary_sha256':binary_sha})
        report.write_text(json.dumps(records,indent=2));assert proc.returncode==0
