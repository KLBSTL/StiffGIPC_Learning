"""Validate base provenance, accepted-path coverage and independent CCD."""
import collections,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];name=sys.argv[1];run=ROOT/'runs/local'/name
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
target=ROOT/'reports'/f'VERIFY_{name}.json';assert not target.exists()
identity=read(ROOT/'reports'/f'V53_REFERENCE_IDENTITY_{name}.json');req=read(run/'requested.json');res=read(run/'result.json')
assert req['exe_sha256']==identity['exe_sha256']==sha(ROOT/'builds/local-base/Release/gipc.exe')
assert req['runner_sha256']==sha(ROOT/'tools/run_local.py')
m=read(ROOT/'manifests/perf_v34.json');rows=[r for r in m['files'] if r['path'].startswith('sources/stiff_base/')]
for r in rows:assert sha(ROOT/r['path'])==r['sha256']
assert req['source_digest']==m['source_digest'] and req['dt']==.01 and req['steps']==100
ref=ROOT/'runs/local/v53_bunny100_completion'
assert read(run/'output/scene.json')==read(ref/'output/scene.json')
for file in ['topology.bin','state_0000.bin','masses.bin','boundary_types.bin','metadata.json']:
    assert (run/'trace'/file).read_bytes()==(ref/'trace'/file).read_bytes(),file
frames=read(run/'output/stats.json')['frames'];groups=collections.defaultdict(list)
pcg=[n['pcg'] for f in frames for n in f.get('newton',[]) if 'pcg' in n]
for p in sorted((run/'trace/substeps').glob('safe_*.bin')):groups[int(p.stem.split('_')[1])].append(p)
n=res['recorded_frames'];assert len(frames)==n and len(groups)>=n
segments=0
for f,paths in sorted(groups.items()):
    assert [int(p.stem.split('_')[2]) for p in paths]==list(range(len(paths)))
    assert paths[0].read_bytes()==(run/f'trace/state_{f:04d}.bin').read_bytes()
    if f<n:assert paths[-1].read_bytes()==(run/f'trace/state_{f+1:04d}.bin').read_bytes()
    segments+=len(paths)-1
ccd=ROOT/'reports'/f'CCD_{name}.json';assert not ccd.exists()
exe=ROOT/'builds/validator/Release/diagnose_first_path.exe'
cmd=[str(exe),str(run/'trace'),str(ccd),'substeps','--stable-nh1']
with ccd.with_suffix('.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
c=read(ccd);assert c['paths_checked']==segments+len(groups)-1
report={'run':name,'result':res,'identity_verified':True,'base_source_files_verified':len(rows),
    'scene_and_initial_state_equal_v53':True,'accepted_segments':segments,'bridges':len(groups)-1,
    'pcg_calls':len(pcg),'pcg_limit_hits':sum(bool(p.get('iteration_limit')) for p in pcg),
    'pcg_max_iterations':max((p['iterations'] for p in pcg),default=None),
    'ccd_exit_code':r.returncode,'ccd':c,'ccd_command':cmd,'validator_sha256':sha(exe)}
target.write_text(json.dumps(report,indent=2));print(json.dumps(report))
assert r.returncode==0 and c['passed']
