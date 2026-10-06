"""Bounded integrated MAS validation; timings are diagnostic only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--phase', required=True, choices=['fixtures', 'smoke', 'paths', 'bunny', 'strict'])
a = p.parse_args()
report = ROOT / 'reports' / ('v39_' + a.phase + '.json')
assert not report.exists(), 'Preserve previous result'
records = []
if a.phase == 'fixtures':
    old = ROOT.parent / 'v37_20261003/runs/autodl'
    for case, prefix in [('smoke', old/'autodl_perf_v37_mas_smoke/mas_audit_initial'),
                         ('initial', old/'autodl_perf_v37_bunny_mas/mas_audit_initial'),
                         ('failure', old/'autodl_perf_v37_bunny_mas/mas_audit_failure')]:
        out = ROOT/'runs/fixtures'/case
        assert not out.exists()
        env = {k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
        env.update(GIPC_MAS_REPLAY_FIXTURE=str(prefix), GIPC_MAS_FIXTURE_OUTPUT=str(out))
        command = [str(ROOT/'builds/autodl-stiff_perf_v39/gipc')]
        with (ROOT/'reports'/('fixture_'+case+'.log')).open('w') as log:
            run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        records.append({'case':case,'command':command,'fixture_input':str(prefix),'code':run.returncode})
        report.write_text(json.dumps(records,indent=2))
        print(json.dumps(records[-1]),flush=True)
        if run.returncode:sys.exit(run.returncode)
else:
    if a.phase == 'smoke':
        tasks = [('native_smoke','cloth_hang_l','mas','0',2,[]),
                 ('wide_smoke','cloth_hang_l','mas','1',2,['--mas-audit']),
                 ('diag_control','cloth_sphere7_l','diag','1',2,[])]
    elif a.phase == 'paths':
        tasks = [('wide_hang_paths','cloth_hang_l','mas','1',100,['--substeps','--physics'])]
    elif a.phase == 'bunny':
        tasks = [('wide_bunny','bunny_cloth_bunny_l','mas','1',100,['--mas-audit','--substeps','--physics'])]
    else:
        tasks = [('wide_bunny_strict','bunny_cloth_bunny_l','mas','1',100,
                  ['--pcg-tol','1e-14','--mas-audit','--substeps','--physics'])]
    for name,scene,prec,wide,steps,extra in tasks:
        command = [sys.executable,str(ROOT/'tools/run_perf_v39.py'),'--platform','autodl',
                   '--arm','base_toi_graph','--scene',scene,'--preconditioner',prec,
                   '--mas-wide',wide,'--steps',str(steps),'--name','autodl_perf_v39_'+name,
                   '--suite','1','--dt','.01','--robust-velocity-tol','0.05','--trace','--audit-pcg',
                   '--failure-system','--diagnostic-toi','--quality-only','--timeout','240']+extra
        with (ROOT/'reports'/('runner_'+name+'.log')).open('w') as log:
            run = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=280)
        records.append({'name':name,'command':command,'code':run.returncode})
        report.write_text(json.dumps(records,indent=2))
        print(json.dumps(records[-1]),flush=True)
        if run.returncode and a.phase in ('smoke','paths'):sys.exit(run.returncode)
