"""Isolated local smoke/performance runs; no writes to other task folders."""
import argparse, csv, hashlib, json, os, subprocess, time, math, struct, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    'base': ('local-base', 'ipc', 'host'),
    'base_graph': ('local-fused-v34', 'ipc', 'conditional_graph'),
    'base_toi': ('local-fused-v34', 'toi_al', 'host'),
    'base_toi_graph': ('local-fused-v34', 'toi_al', 'conditional_graph'),
    'fused_host': ('local-fused-v34', 'ipc', 'host'),
}

def gpu_snapshot():
    r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    name, total, free = r.stdout.strip().split(',')
    return {'name': name, 'total_mib': int(total), 'free_mib': int(free)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--toi-policy',choices=['paper','robust'],default='robust')
    p.add_argument('--case', type=int, default=4)
    p.add_argument('--scene')
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--name', required=True)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--trace', action='store_true')
    p.add_argument('--suite', choices=['0','1'], default='0')
    p.add_argument('--tol', type=float, default=.01)
    p.add_argument('--pcg-tol', type=float, default=1e-4)
    p.add_argument('--dt', type=float)
    p.add_argument('--friction', type=float)
    p.add_argument('--audit-pcg', action='store_true')
    p.add_argument('--pcg-replay-from-frame',type=int,help='Diagnostic fixed-system repeat of the first linear solve per frame, from this 1-based frame')
    p.add_argument('--substeps', action='store_true')
    p.add_argument('--audit-graph', action='store_true')
    p.add_argument('--audit-friction-snapshot',action='store_true',help='Read-only checks of frame-lagged friction buffers after every outer solve')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--platform', choices=['local','autodl'], default='local')
    p.add_argument('--paper-stop', action='store_true',help='Disable contact-free movement exit for a diagnostic ablation')
    p.add_argument('--trace-stride',type=int,default=1)
    p.add_argument('--audit-suite',action='store_true')
    p.add_argument('--quality-only',action='store_true',help='Allow overlapping external GPU work only for correctness; timing is invalid')
    p.add_argument('--physics',action='store_true')
    p.add_argument('--filter-ratio',type=float,default=.2)
    p.add_argument('--clamp-trial',action='store_true',help='Diagnostic ablation of the v4 inner injectivity clamp')
    p.add_argument('--no-initial-reuse',action='store_true')
    p.add_argument('--manifest',default=str(ROOT/'manifests/perf_v39_autodl.json'),help='Frozen v39 identity')
    p.add_argument('--fixed-study',action='store_true')
    p.add_argument('--fixed-study-frames',default='1,22,40,70,100')
    p.add_argument('--fixed-study-directions',default='1,8')
    p.add_argument('--failure-system',action='store_true')
    p.add_argument('--state-window',action='store_true')
    p.add_argument('--mas-audit',action='store_true')
    p.add_argument('--mas-wide',choices=['0','1'],default='0')
    p.add_argument('--fused-diag-update', choices=['0','1'], default='0')
    p.add_argument('--mu-scale',type=float,default=1.)
    p.add_argument('--mu-mode',choices=['diagonal','mass'],default='diagonal',help='Diagnostic public Robust per-contact mass stiffness branch')
    p.add_argument('--mu-scope',choices=['full','movable'],help='Override contact-free stiffness norm scope; robust defaults to movable DOFs')
    p.add_argument('--ccd-pair-limit',type=int,default=0)
    p.add_argument('--diagnostic-toi',action='store_true')
    p.add_argument('--curvature-from-frame',type=int,help='Diagnostic-only second directional energy difference from this 1-based frame')
    p.add_argument('--triangle-audit',action='store_true',help='Record per-triangle cloth energy concentration for sampled TOI directions')
    p.add_argument('--state-audit-from-frame',type=int,help='Diagnostic unchanged-state reassembly and exact energy-probe snapshot restoration, from this 1-based frame')
    p.add_argument('--blocker-audit-from-frame',type=int,help='Read-only diagnostic of contacts limiting safe CCD alpha, from this 1-based frame')
    p.add_argument('--blocker-audit-to-frame',type=int,help='Last 1-based frame of the safe-alpha blocker diagnostic')
    p.add_argument('--watch-contact',help='Diagnostic PT contact as four comma-separated vertex IDs')
    p.add_argument('--stall-window',type=int,help='Diagnostic number of tiny safe steps before doubling TOI stiffness')
    p.add_argument('--stall-hold-delta',action='store_true',help='Diagnostic stiffness escalation without reducing contact target distance')
    p.add_argument('--persist-contacts',action='store_true',help='Diagnostic Robust-style cross-frame active set and friction warm start')
    p.add_argument('--warm-contacts',choices=['reset','keep'],help='Independent frame-start contact identity history ablation')
    p.add_argument('--warm-lambda',choices=['reset','keep'],help='Independent frame-start AL multiplier history ablation')
    p.add_argument('--warm-gamma',choices=['reset','keep'],help='Independent frame-start AL gamma history ablation')
    p.add_argument('--friction-history',choices=['coupled','independent'],default='coupled',help='Default AL-derived friction, or a diagnostic separately updated force proxy')
    p.add_argument('--warm-friction',choices=['reset','keep'],help='Frame-start history for the independent friction proxy')
    volume_group=p.add_mutually_exclusive_group()
    volume_group.add_argument('--no-safe-injectivity',action='store_true',help='Disable extra volume constraint; Stable NH1/base default policy')
    volume_group.add_argument('--safe-injectivity',action='store_true',help='Require positive volume as a separately labeled constraint ablation')
    p.add_argument('--robust-velocity-tol',type=float,help='Public Robust trial displacement safeguard, in m/s; separate implementation ablation')
    p.add_argument('--robust-velocity-stop',action='store_true',help='Allow an accepted energy-decreasing trial to exit its inner solve when the Robust displacement tolerance is reached')
    p.add_argument('--preconditioner',choices=['mas','diag'],help='Fused binary diagnostic preconditioner override')
    p.add_argument('--legacy-volume-bound',action='store_true',help='Unsafe historical bound, for a labeled regression diagnostic only')
    a = p.parse_args()
    split_history=any(getattr(a,k) is not None for k in ['warm_contacts','warm_lambda','warm_gamma','warm_friction']) or a.friction_history=='independent'
    if split_history and a.arm not in ('base_toi','base_toi_graph'):
        p.error('Independent warm history requires a TOI arm')
    if split_history and a.persist_contacts:
        p.error('Use --persist-contacts or independent warm options, not both')
    if a.warm_friction is not None and a.friction_history!='independent':
        p.error('--warm-friction requires --friction-history independent')
    if any(getattr(a,k)=='keep' for k in ['warm_lambda','warm_gamma','warm_friction']) and a.warm_contacts!='keep':
        p.error('Keeping lambda/gamma/friction requires --warm-contacts keep')
    if a.pcg_replay_from_frame is not None and (a.pcg_replay_from_frame<1 or a.arm=='base'):
        p.error('--pcg-replay-from-frame requires a positive frame number and a fused-binary arm')
    if a.robust_velocity_stop and a.robust_velocity_tol is None:
        p.error('--robust-velocity-stop requires --robust-velocity-tol')
    if a.curvature_from_frame is not None and (not a.diagnostic_toi or a.curvature_from_frame<1):
        p.error('--curvature-from-frame requires --diagnostic-toi and a positive frame number')
    if a.triangle_audit and a.curvature_from_frame is None:
        p.error('--triangle-audit requires --curvature-from-frame')
    if a.state_audit_from_frame is not None and (a.state_audit_from_frame<1 or a.arm not in ('base_toi','base_toi_graph')):
        p.error('--state-audit-from-frame requires a positive frame and a TOI arm')
    if a.blocker_audit_from_frame is not None and (a.blocker_audit_from_frame<1 or a.arm not in ('base_toi','base_toi_graph')):
        p.error('--blocker-audit-from-frame requires a positive frame and a TOI arm')
    if a.blocker_audit_to_frame is not None and (a.blocker_audit_from_frame is None or a.blocker_audit_to_frame<a.blocker_audit_from_frame):
        p.error('--blocker-audit-to-frame requires an ordered blocker audit window')
    out = ROOT / 'runs' / a.platform / a.name
    build, backend, execution = ARMS[a.arm]
    exe = ROOT / 'builds' / build / 'Release' / 'gipc.exe'
    if a.platform=='autodl':
        exe=ROOT/'builds'/('autodl-stiff_base' if build=='local-base' else 'autodl-stiff_perf_v39')/'gipc'
    if not exe.is_file(): raise FileNotFoundError(exe)
    manifest=json.loads(Path(a.manifest).read_text(encoding='utf-8-sig'))
    if manifest['implementation_version']!='v39-mas-wide-integrated':
        raise ValueError('This runner requires the separate v39 manifest')
    for entry in manifest['files']:
        source=ROOT/entry['path']
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError(f'Frozen source changed: {source}')
    exe_sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    identity=manifest['binaries'][str(exe.relative_to(ROOT).as_posix())]
    if exe_sha!=identity['sha256']:
        raise ValueError(f'Frozen binary changed: {exe}')
    gpu = gpu_snapshot()
    budget = min(.75 * gpu['free_mib'], gpu['free_mib'] - 1536)
    # Official case4: 4225 cloth vertices. This is a coarse feasibility gate;
    # peak memory is measured below before scaling or extended runs.
    if budget < 1024: raise RuntimeError(f'Insufficient free GPU memory: {gpu}')
    preflight=None
    if a.scene:
        from preflight import estimate_scene
        preflight=estimate_scene(ROOT/'sources/stiff_base/Assets/benchmark_scenes'/f'{a.scene}.json',budget)
        if preflight['estimated_reserve_mib']>budget:raise RuntimeError(f'Scene reserve estimate exceeds GPU budget: {preflight}')
    if a.platform=='autodl':
        others=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
        if others and not a.quality_only:raise RuntimeError(f'GPU already has compute processes; refusing overlapping measurement: {others}')
    out.mkdir(parents=True,exist_ok=False)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('GIPC_'):env.pop(key)
    env['GIPC_TOI_POLICY']=a.toi_policy
    env['GIPC_MAS_WIDE_APPLY']=a.mas_wide
    env['GIPC_PCG_FUSED_DIAG_UPDATE']=a.fused_diag_update
    env['GIPC_TOI_FRAME_FRICTION_AUDIT']='1' if a.audit_friction_snapshot else '0'
    env.update(GIPC_CASE=str(a.case), GIPC_STEPS=str(a.steps),
               GIPC_CONTACT_BACKEND=backend, GIPC_PCG_EXECUTION=execution,
               GIPC_DUMP_STATE=str(out/'final.bin'), GIPC_OUTPUT_PATH=str(out/'output'))
    env.update(GIPC_ACCEL_SUITE=a.suite,GIPC_NEWTON_TOL=str(a.tol),GIPC_PCG_TOL=str(a.pcg_tol),
               GIPC_AUDIT_PCG='1' if a.audit_pcg else '0')
    if a.pcg_replay_from_frame is not None:env['GIPC_PCG_REPLAY_FROM_FRAME']=str(a.pcg_replay_from_frame)
    env.update(GIPC_AUDIT_GRAPH='1' if a.audit_graph else '0',GIPC_PROFILE='1' if a.profile else '0',
               GIPC_TOI_EARLY_TERMINATION='0' if a.paper_stop else '1')
    env.update(GIPC_AUDIT_REFIT='1' if a.audit_suite else '0',GIPC_AUDIT_ENERGY='1' if a.audit_suite else '0',
               GIPC_DEFER_STATS='1',GIPC_TRACE_STRIDE=str(a.trace_stride))
    env['GIPC_TRACE_PHYSICS']='1' if a.physics else '0'
    env['GIPC_TOI_FILTER_RATIO']=str(a.filter_ratio)
    env['GIPC_TOI_TRIAL_INJECTIVITY']='1' if a.clamp_trial else '0'
    env['GIPC_TOI_REUSE_INITIAL_ASSEMBLY']='0' if a.no_initial_reuse else '1'
    env['GIPC_TOI_MU_SCALE']=str(a.mu_scale)
    env['GIPC_TOI_MU_MODE']=a.mu_mode
    if a.mu_scope is not None:env['GIPC_TOI_MU_SCOPE']=a.mu_scope
    env['GIPC_TOI_SAFE_INJECTIVITY']='1' if a.safe_injectivity else '0' if a.no_safe_injectivity else 'auto'
    env['GIPC_TOI_VOLUME_BOUND']='legacy' if a.legacy_volume_bound else 'normalized'
    if a.robust_velocity_tol is not None:env['GIPC_TOI_ROBUST_VELOCITY_TOL']=str(a.robust_velocity_tol)
    if a.robust_velocity_stop:env['GIPC_TOI_ROBUST_VELOCITY_STOP']='1'
    if a.preconditioner is not None:
        if build=='local-base':raise ValueError('Preconditioner override requires the fused binary')
        env['GIPC_PCG_PRECONDITIONER']=a.preconditioner
    if a.ccd_pair_limit:env['GIPC_CCD_PAIR_LIMIT']=str(a.ccd_pair_limit)
    if a.diagnostic_toi:env['GIPC_TOI_DIAGNOSTIC_LOG']=str(out/'toi_diagnostic.jsonl')
    if a.curvature_from_frame is not None:env['GIPC_TOI_CURVATURE_FROM_FRAME']=str(a.curvature_from_frame)
    if a.triangle_audit:env['GIPC_TOI_TRIANGLE_AUDIT']='1'
    if a.state_audit_from_frame is not None:env['GIPC_TOI_STATE_AUDIT_FROM_FRAME']=str(a.state_audit_from_frame)
    env['GIPC_TOI_BLOCKER_AUDIT_FROM']=str(a.blocker_audit_from_frame or 0)
    env['GIPC_TOI_BLOCKER_AUDIT_TO']=str(a.blocker_audit_to_frame or 2147483647)
    if a.watch_contact:env['GIPC_TOI_WATCH_CONTACT']=a.watch_contact
    if a.stall_window is not None:env['GIPC_TOI_STALL_WINDOW']=str(a.stall_window)
    if a.stall_hold_delta:env['GIPC_TOI_STALL_HOLD_DELTA']='1'
    env['GIPC_TOI_PERSIST_CONTACTS']='1' if a.persist_contacts else '0'
    env['GIPC_TOI_INDEPENDENT_FRICTION_HISTORY']='1' if a.friction_history=='independent' else '0'
    for name,flag in [('warm_contacts','GIPC_TOI_WARM_CONTACTS'),('warm_lambda','GIPC_TOI_WARM_LAMBDA'),
                      ('warm_gamma','GIPC_TOI_WARM_GAMMA'),('warm_friction','GIPC_TOI_WARM_FRICTION')]:
        value=getattr(a,name)
        if value is None:env.pop(flag,None)
        else:env[flag]='1' if value=='keep' else '0'
    if a.scene: env['GIPC_SCENE']=a.scene
    if a.dt is not None:env['GIPC_DT']=str(a.dt)
    if a.friction is not None:env['GIPC_FRICTION']=str(a.friction)
    (out/'output').mkdir()
    if a.fixed_study:
        if a.arm=='base':raise ValueError('Fixed-system study requires v39 binary')
        (out/'fixed').mkdir()
        env['GIPC_FIXED_STUDY_DIR']=str(out/'fixed')
        env['GIPC_FIXED_STUDY_FRAMES']=a.fixed_study_frames
        env['GIPC_FIXED_STUDY_DIRECTIONS']=a.fixed_study_directions
    if a.failure_system:env['GIPC_FAILURE_SYSTEM']=str(out/'failure_system')
    if a.state_window:
        (out/'state_window').mkdir()
        env['GIPC_STATE_WINDOW_DIR']=str(out/'state_window')
    if a.mas_audit:
        env['GIPC_MAS_SNAPSHOT']='1'
        env['GIPC_MAS_AUDIT']=str(out/'mas_audit')
    if a.trace: env['GIPC_TRACE_DIR'] = str(out/'trace')
    if a.substeps: env['GIPC_TRACE_SUBSTEPS']=str(out/'trace/substeps')
    requested = vars(a) | {'gpu_before': gpu, 'memory_budget_mib': budget,
                            'scene_preflight':preflight,
                            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'source_digest':manifest['source_digest'],
                            'exe_sha256':exe_sha,
                            'backend': backend, 'execution': execution}
    (out/'requested.json').write_text(json.dumps(requested, indent=2), encoding='utf-8')
    started = time.monotonic(); peak = 0; status = 'running'
    with (out/'run.log').open('wb') as log:
        command=[str(exe)] if a.platform=='local' else ['xvfb-run','-a','-s','-screen 0 640x480x24',str(exe)]
        proc = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=a.platform=='autodl')
        (out/'process.json').write_text(json.dumps({'pid':proc.pid,'command':command,'started_unix':time.time()}),encoding='utf-8')
        while proc.poll() is None:
            if shutil.disk_usage(out).free < 768*1024**2:
                if a.platform=='autodl':
                    import signal;os.killpg(proc.pid,signal.SIGTERM)
                else:proc.terminate()
                status='disk_reserve';break
            peak = max(peak, gpu['free_mib'] - gpu_snapshot()['free_mib'])
            if time.monotonic() - started > a.timeout:
                if a.platform=='autodl':
                    import signal;os.killpg(proc.pid,signal.SIGTERM)
                else:proc.terminate()
                status = 'timeout'; break
            if peak > budget:
                if a.platform=='autodl':
                    import signal;os.killpg(proc.pid,signal.SIGTERM)
                else:proc.terminate()
                status = 'memory_budget'; break
            time.sleep(1)
        proc.wait()
    status = ('completed' if proc.returncode == 0 else 'failed') if status == 'running' else status
    result = {'status': status, 'exit_code': proc.returncode,
              'wall_seconds': time.monotonic()-started, 'wall_time_monitor_uncertainty_seconds':1,
              'observed_gpu_increment_peak_mib': peak,'quality_only':a.quality_only,
              'timing_is_diagnostic':a.state_window or a.mas_audit or a.fixed_study or a.failure_system or a.quality_only or a.audit_graph or a.audit_friction_snapshot or a.audit_pcg or a.audit_suite or a.profile or a.substeps or a.diagnostic_toi or a.pcg_replay_from_frame is not None or a.state_audit_from_frame is not None or a.blocker_audit_from_frame is not None}
    final = out/'final.bin'
    if final.exists():
        raw=final.read_bytes()
        if len(raw)%24:raise ValueError('Invalid final state byte count')
        result.update(vertices=len(raw)//24,finite=all(math.isfinite(v[0]) for v in struct.iter_unpack('<d',raw)))
    frames = out/'trace'/'frames.csv'
    if frames.exists():
        with frames.open() as f: rows = list(csv.DictReader(f))
        result['recorded_frames'] = len(rows)
        result['solver_seconds'] = sum(float(r['solver_ms']) for r in rows)/1000
    if backend=='toi_al' and a.toi_policy=='robust' and status=='completed':
        records=json.loads((out/'output/stats.json').read_text())['frames']
        captures=[f.get('toi_frame_friction_snapshot',{}) for f in records]
        valid=len(records)==a.steps and all(s.get('captured_this_solve')
            and s.get('physical_frame')==i for i,s in enumerate(captures))
        result['robust_frame_protocol_verified']=valid
        if not valid:
            status='protocol_failed';result['status']=status
    (out/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    if status != 'completed':
        print('\n'.join((out/'run.log').read_text(errors='replace').splitlines()[-18:]))
        raise SystemExit(1)

if __name__ == '__main__': main()
