"""Compact evidence from preserved v54 results; no simulator changes."""
import json, math, statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text())
def load(name,frame):return np.fromfile(ROOT/f'runs/local/{name}/trace/state_{frame:04d}.bin',dtype='<f8').reshape(-1,3)
out={}
c=read('reports/V54_components.json');g=read('reports/V54_guard.json')
assert c['passed'] and g['passed']
out['component_cases']=sum(len(v) for v in c.values() if isinstance(v,list))
out['guard_cases']=len(g['cases'])
out['smoke_max_coordinate_difference_m']=max(float(np.max(np.abs(load('v54_smoke_legacy_r1',i)-load('v54_smoke_guard_r1',i)))) for i in range(4))
for name in ['v54_smoke_legacy_r1','v54_smoke_guard_r1']:
    frames=read(f'runs/local/{name}/output/stats.json')['frames']
    assert not any(t.get('initial_guess_selection',{}).get('selected')=='safe' for f in frames for t in f.get('toi',[]))
out['no_restart_smoke_passed']=out['smoke_max_coordinate_difference_m']<1e-10
out['ccd']={}
for label in ['SCREEN','CLOTH']:
    path=f'reports/V54_{label}_VERIFICATION.json'
    if not (ROOT/path).exists():continue
    v=read(path);paths=0
    assert len(v['runs'])=={'SCREEN':5,'CLOTH':18}[label]
    for run in v['runs']:
        ccd=run['ccd']['report']
        assert ccd['passed'] and ccd['conservative_collision_flags']==0 and run.get('ccd_exit_code',0)==0
        assert run['identity_verified'] and run['flushed_stats']['limit_hits']==0 and run['flushed_stats']['breakdowns']==0
        paths+=ccd['paths_checked']
    out['ccd'][label]={'runs':len(v['runs']),'paths':paths,'passed':True,'source_files_verified':v['source_files_verified']}
out['scenes']=[]
if (ROOT/'reports/V54_CLOTH_QUALITY.json').exists():
    q=read('reports/V54_CLOTH_QUALITY.json')
    for s in q['scenes']:
        variants={}
        for variant in ['off','legacy','guard']:
            runs=[r for r in s['runs'] if r['variant']==variant]
            assert len(runs)==2 and all(r['native_audit_passed'] for r in runs)
            variants[variant]={'max_stretch':[r['max_stretch'] for r in runs],
                'solver_seconds':[r['result']['solver_seconds'] for r in runs],
                'median_solver_seconds':statistics.median(r['result']['solver_seconds'] for r in runs),
                'safe_selections':[r['safe_selections'] for r in runs],
                'blocked_full_steps':[r['restart_full_steps_blocked'] for r in runs],
                'pcg_iterations':[r['pcg_iterations'] for r in runs],
                'peak_frames':[max(r['observations'],key=lambda o:o['max_stretch'])['frame'] for r in runs]}
        out['scenes'].append({'scene':s['scene'],'variants':variants,
            'diagnostic_off_over_guard':variants['off']['median_solver_seconds']/variants['guard']['median_solver_seconds'],
            'repeat_rms_percent':[{k:p[k] for k in ['a','b','max_cloth_rms_percent_scale']} for p in s['pairs'] if p['same_switch']]})
    out['fixed_bunny_frame24']=[]
    for r in next(s for s in q['scenes'] if s['scene']=='cloth_fixed_bunny_l')['runs']:
        x=load(r['name'],24);x0=load(r['name'],0);edge=[23850,23857]
        ratio=math.dist(x[edge[0]],x[edge[1]])/math.dist(x0[edge[0]],x0[edge[1]])
        assert math.isfinite(ratio)
        f=read(f"runs/local/{r['name']}/output/stats.json")['frames'][23]
        out['fixed_bunny_frame24'].append({'run':r['name'],'same_edge':edge,'same_edge_ratio':ratio,
            'global_max_stretch':r['observations'][23]['max_stretch'],
            'newton':len(f['newton']),'outer':len(f['toi']),
            'last_exit':f['newton'][-1].get('inner_exit_reason'),
            'last_raw_velocity':f['newton'][-1].get('trial_newton_axis_velocity_m_s')})
target=ROOT/'reports/V54_FINAL_CHECK.json';assert not target.exists()
target.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out,indent=2))
