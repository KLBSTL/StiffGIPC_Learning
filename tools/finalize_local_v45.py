"""Freeze local run identities and summarize acceptance without hiding failed gates."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read(ROOT/'manifests/perf_v45_local.json')
for f in m['files']:assert sha(ROOT/f['path'])==f['sha256']
for p,x in m['binaries'].items():assert sha(ROOT/p)==x['sha256']
rows=[]
for run in sorted((ROOT/'runs/local').glob('v45r2_*')):
    if not (run/'result.json').exists():continue
    req=read(run/'requested.json');res=read(run/'result.json')
    assert req['source_digest']==m['source_digest'] and req['runner_sha256']==sha(ROOT/'tools/run_perf_v45.py')
    assert req['exe_sha256']==m['binaries']['builds/local-v45/Release/gipc.exe']['sha256']
    rows.append({'run':run.name,'requested':req,'result':res,'requested_sha256':sha(run/'requested.json'),
        'result_sha256':sha(run/'result.json'),'log_sha256':sha(run/'run.log')})
assert len(rows)==12
branches=[r for r in rows if 'branch_' in r['run']]
assert len(branches)==3 and len({r['requested']['checkpoint_load'] for r in branches})==1
cp=Path(branches[0]['requested']['checkpoint_load']);checkpoint=read(cp/'checkpoint.json')
assert checkpoint['completed_physical_frames']==34
component=read(ROOT/'reports/V45R2_components.json')
checks=[x for v in component.values() if isinstance(v,list) for x in v]
assert len(checks)==37 and all(x['passed'] for x in checks)
for name,text in [('reject_corrupt','Checkpoint member integrity mismatch'),('reject_physics','Checkpoint scene/physics context mismatch')]:
    run=ROOT/'runs/local'/('v45r2_'+name)
    assert read(run/'result.json')['exit_code']==4 and text in (run/'run.log').read_text(errors='replace')
reports=['VERIFICATION_V45R2_RESTART.json','VERIFICATION_V45R2_REACH35.json','VERIFICATION_V45R2_DEFAULT.json',
    'VERIFICATION_V45R2_STRICT.json','VERIFICATION_V45R2_VELOCITY.json']
verified=[x for p in reports for x in read(ROOT/'reports'/p)['runs']]
ccd=[x['ccd']['report'] for x in verified if 'ccd' in x]
strict_fixed=[s for x in verified for s in x['fixed_systems']]
result={'source_files_verified':len(m['files']),'source_digest':m['source_digest'],'binaries':m['binaries'],'runs':rows,
    'same_frame35_checkpoint':{'metadata_sha256':sha(cp/'checkpoint.json'),
        'members_sha256':{p.name:sha(p) for p in cp.glob('*.bin')},'device_restore_verified':all(read(ROOT/'runs/local'/r['run']/'output/checkpoint_restore.json')['device_bytes_verified'] for r in branches)},
    'components_passed':len(checks),'negative_checkpoint_checks_passed':True,
    'hang_continuation':read(ROOT/'reports/V45R2_CONTINUE_hang.json'),
    'contact_continuation':read(ROOT/'reports/V45R2_CONTINUE_contact.json'),
    'contact_repeat':read(ROOT/'reports/V45R2_CONTACT_REPEAT.json'),
    'strict_fixed_system_gates':[{'path':s['path'],'passed':s['strict_gate_passed']} for s in strict_fixed],
    'cpu_ccd':{'runs':len(ccd),'paths_checked':sum(x['paths_checked'] for x in ccd),
        'flags':sum(x['conservative_collision_flags'] for x in ccd),'all_exported_coverage_passed':all(x['passed'] for x in ccd),
        'fem_inversion_path_observations':sum(x['fem_tet_inversions'] for x in ccd),'scope':'Exported accepted substeps only; not all 30 seed frames or unaccepted trial paths'},
    'reports_sha256':{p:sha(ROOT/'reports'/p) for p in reports},
    'archives_sha256':{p.name:sha(p) for p in (ROOT/'archives').glob('v45*local*failure.zip')},
    'tools_sha256':{p.name:sha(p) for p in (ROOT/'tools').glob('*v45.py')},
    'final_gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.free,memory.total','--format=csv,noheader'],text=True).strip(),
    'full_trajectory_repair_passed':False,'performance_certified':False}
out=ROOT/'reports/VERIFICATION_V45_LOCAL_FINAL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ['source_files_verified','components_passed','cpu_ccd','strict_fixed_system_gates','final_gpu']}))
