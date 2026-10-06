"""Audit candidate accepted paths, source identity and independent continuous collision."""
import collections,json,subprocess
from run_bunny_components import ROOT,read,sha
target=ROOT/'reports/BUNNY_COMPONENTS_VERIFICATION.json';assert not target.exists()
rows=[]
for record in read(ROOT/'reports/BUNNY_COMPONENTS_QUALITY.json')['runs']:
    name=record['name'];run=ROOT/'runs/local'/name;result=read(run/'result.json');req=read(run/'requested.json')
    assert result['status']=='completed' and result['recorded_frames']==100 and req['audit']
    frames=read(run/'output/stats.json')['frames'];assert len(frames)==100
    pcg=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
    assert not any(p.get('iteration_limit') or p.get('breakdown') for p in pcg)
    if record['label']=='combined':
        assert all(f['bvh_refit_audit']['identical'] for f in frames)
        assert all(f['energy_batch_audit']['max_relative_error']<=1e-10 for f in frames)
    groups=collections.defaultdict(list)
    for path in sorted((run/'trace/substeps').glob('safe_*.bin')):groups[int(path.stem.split('_')[1])].append(path)
    assert len(groups)==100
    accepted=0
    for f,paths in sorted(groups.items()):
        assert [int(p.stem.split('_')[2]) for p in paths]==list(range(len(paths)))
        assert paths[0].read_bytes()==(run/f'trace/state_{f:04d}.bin').read_bytes()
        assert paths[-1].read_bytes()==(run/f'trace/state_{f+1:04d}.bin').read_bytes()
        accepted+=len(paths)-1
    ccd=ROOT/f'reports/CCD_{name}.json';assert not ccd.exists()
    exe=ROOT/'builds/validator/Release/diagnose_first_path.exe'
    cmd=[str(exe),str(run/'trace'),str(ccd),'substeps','--stable-nh1']
    with ccd.with_suffix('.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    c=read(ccd);assert c['paths_checked']==accepted+len(groups)-1
    rows.append({'name':name,'pcg_calls':len(pcg),'pcg_limit_hits':0,'accepted_segments':accepted,'bridges':len(groups)-1,
        'ccd':c,'command':cmd,'exit_code':r.returncode,'validator_sha256':sha(exe),
        'refit_audited_frames':sum('bvh_refit_audit' in f for f in frames),
        'energy_batch_audited_frames':sum('energy_batch_audit' in f for f in frames)})
    target.write_text(json.dumps({'runs':rows},indent=2));print(json.dumps(rows[-1]),flush=True)
    assert r.returncode==0 and c['passed']
