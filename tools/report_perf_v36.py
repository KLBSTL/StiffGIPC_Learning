"""Verify downloaded identities, timing, physical-time-aligned quality, and profiles."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from report_local_perf_v33 import work
from report_v32_autodl import quality

ROOT=Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def aligned_quality(run, reference):
    req,ref_req=read(run/'requested.json'),read(reference/'requested.json')
    tr,rr=run/'trace',reference/'trace'
    raw=np.fromfile(tr/'topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3])
    assert (tr/'topology.bin').read_bytes()==(rr/'topology.bin').read_bytes()
    faces=raw[3:3+nf*3].reshape(-1,3);tets=raw[3+nf*3:].reshape(-1,4)
    in_tet=np.zeros(nv,dtype=bool);in_tet[tets.ravel()]=True
    cloth=np.unique(faces[~in_tet[faces].any(axis=1)])
    mass=np.fromfile(tr/'masses.bin',dtype='<f8')[cloth]
    initial=np.fromfile(tr/'state_0000.bin',dtype='<f8').reshape(-1,3)
    assert np.array_equal(initial,np.fromfile(rr/'state_0000.bin',dtype='<f8').reshape(-1,3))
    scale=float(np.linalg.norm(np.ptp(initial[cloth],axis=0)))
    rms=[];vel=[];previous=None
    for i in range(101):
        t=i*.01
        ia,ib=round(t/req['dt']),round(t/ref_req['dt'])
        x=np.fromfile(tr/f'state_{ia:04d}.bin',dtype='<f8').reshape(-1,3)[cloth]
        y=np.fromfile(rr/f'state_{ib:04d}.bin',dtype='<f8').reshape(-1,3)[cloth]
        diff=x-y
        rms.append(float(np.sqrt(np.average(np.sum(diff**2,axis=1),weights=mass))))
        if previous is not None:
            v=(diff-previous)/.01
            vel.append(float(np.sqrt(np.average(np.sum(v**2,axis=1),weights=mass))))
        previous=diff
    return {'max_rms_percent':100*max(rms)/scale,'final_rms_percent':100*rms[-1]/scale,
            'max_sampled_velocity_difference_m_s':max(vel),
            'note':'Cloth mass weighted, normalized by initial cloth bounding-box diagonal; sampled at common 0.01 s times. Trajectory differences, not certified error.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();data=args.data.resolve();assert not args.output.exists()
    manifest=read(data/'manifests/perf_v36_autodl.json')
    for f in manifest['files']:
        assert hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()==f['sha256'],f['path']
    for name,value in manifest['binaries'].items():
        assert hashlib.sha256((data/name).read_bytes()).hexdigest()==value['sha256'],name
    index=read(data/'reports/AUTODL_PERF_V36_EXPORT_FILES.json')
    for f in index:
        assert hashlib.sha256((data/f['path']).read_bytes()).hexdigest()==f['sha256'],f['path']
    runs={};matrix=read(data/'reports/AUTODL_PERF_V36_MATRIX.json')['runs']
    for run in sorted((data/'runs/autodl').iterdir()):
        if not run.is_dir() or not (run/'result.json').exists():continue
        req,res=read(run/'requested.json'),read(run/'result.json')
        assert req['source_digest']==manifest['source_digest']
        assert req['runner_sha256']==hashlib.sha256((data/'tools/run_perf_v36.py').read_bytes()).hexdigest()
        assert req['exe_sha256'] in [v['sha256'] for v in manifest['binaries'].values()]
        csv_rows=list(csv.DictReader((run/'trace/frames.csv').open()))
        seconds=sum(float(r['solver_ms']) for r in csv_rows)/1000
        assert abs(seconds-res['solver_seconds'])<1e-8
        frames,counts=work(run)
        if res['status']=='completed':
            assert len(frames)==req['steps']==len(csv_rows)
            assert res['finite'] and not any(counts[k] for k in ['pcg_limit_hits','outer_limit_hits','unsafe_native_steps','unfinished_outer'])
            if req['execution']=='conditional_graph':assert counts['execution']=={'conditional_graph':counts['directions']}
        directions=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
        audit={k:max((n[k] for n in directions if k in n),default=None)
               for k in ['same_system_relative_solution_difference','same_system_host_repeat_relative_difference','true_relative_residual']}
        runs[run.name]={'requested':req,'result':res,'work':counts,'audit':audit}
    performance=[]
    for scene in ['cloth_sphere7_l','cloth_hang_l']:
        selected=[r for r in matrix if r['scene']==scene]
        times={arm:[r['solver_seconds'] for r in selected if arm in r['label']] for arm in ['base','toi_suite']}
        candidates=[r for r in selected if 'toi_suite' in r['label']]
        base_runs=[r for r in selected if '_base_' in r['label']]
        q=[quality(data/c['run'],data/b['run']) for c,b in zip(candidates,base_runs)]
        repeats=[quality(data/c['run'],data/candidates[0]['run']) for c in candidates]
        performance.append({'scene':scene,'base_median_seconds':statistics.median(times['base']),
                            'toi_suite_median_seconds':statistics.median(times['toi_suite']),
                            'raw_speed_ratio':statistics.median(times['base'])/statistics.median(times['toi_suite']),
                            'paired_ratios':[a/b for a,b in zip(times['base'],times['toi_suite'])],
                            'candidate_cv_percent':100*statistics.stdev(times['toi_suite'])/statistics.mean(times['toi_suite']),
                            'max_cloth_rms_vs_base_percent':max(v['max_cloth_mass_rms_vs_base_percent'] for v in q),
                            'max_cloth_repeat_rms_percent':max(v['max_cloth_mass_rms_vs_base_percent'] for v in repeats),
                            'quality_matched':False})
    profile=[]
    for r in read(data/'reports/AUTODL_PERF_V36_PROFILE.json')['runs']:
        frames=read(data/r['run']/'output/stats.json')['frames']
        outer=[t for f in frames for t in f['toi'] if 'alpha' in t]
        detail={k:sum(t['safe_detail_ms'][k] for t in outer)/1000 for k in outer[0]['safe_detail_ms'] if k!='attempts'}
        profile.append({'label':r['label'],'accepted_outer':len(outer),'detail_seconds':detail,
                        'detail_ms_per_outer':{k:v*1000/len(outer) for k,v in detail.items()}})
    root=data/'runs/autodl'
    ref=lambda n:root/f'autodl_perf_v36_reference_sphere_base_strict_dt{n}'
    precision=lambda n:root/f'autodl_perf_v36_precision_{n}'
    convergence={'dt_to_dt2':aligned_quality(ref(1),ref(2)),
                 'dt2_to_dt4':aligned_quality(ref(2),ref(4)),
                 'dt4_to_dt8':aligned_quality(ref(4),precision('base_dt8')),
                 'dt4_repeat':aligned_quality(ref(4),precision('base_dt4_repeat'))}
    precision_comparison=[]
    for tol in ['1e-8','1e-12']:
        rr=[precision(f'toi_{tol}_r{i}') for i in range(1,4)]
        precision_comparison.append({'rho_tolerance':tol,
            'median_seconds':statistics.median(runs[r.name]['result']['solver_seconds'] for r in rr),
            'repeat_max_rms_percent':max(aligned_quality(r,rr[0])['max_rms_percent'] for r in rr),
            'reference_dt8_max_rms_percent':max(aligned_quality(r,precision('base_dt8'))['max_rms_percent'] for r in rr)})
    nonlinear=aligned_quality(precision('toi_nonlinear_tight'),precision('base_dt8'))
    output={'source_files_verified':len(manifest['files']),'archive_files_verified':len(index),
            'runs':runs,'performance':performance,'safe_profile':profile,'reference_convergence':convergence,
            'precision_comparison':precision_comparison,'nonlinear_tight_vs_dt8':nonlinear,
            'formal_quality_matched_speedup':None}
    args.output.write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k not in ['runs']},indent=2))


if __name__=='__main__':main()
