"""CPU-only extraction of complete selective-precision controls, without reruns."""
from pathlib import Path
import argparse
import json
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
for folder in ('bench','diagnostic','local'):
    sys.path.insert(0,str(ROOT/'tools'/folder))
from config import read
from linux_runner import require,write_new
from local_identity import record,verify
from pool_metrics import state_comparison
from quality_analysis import input_comparison


def analyze(session,output):
    ledger=read(session/'ledger.json');identity=read(session/'identity.json')
    rows=ledger['runs'];expected=['legacy','wide','inverse64','inverse64','wide','legacy']
    require(not ledger['skipped'] and [r['task']['arm'] for r in rows]==expected,'Incomplete control sequence')
    require(ledger['native_source_unchanged'],'Native source changed during controls')
    verify(identity['native_source_before']+[identity['controller']])
    program=identity['program']
    verify(program['build_evidence']+program.get('objects',[])+[program['exe']]+program['dlls'])
    runner=record(ROOT/'tools/local/windows_runner.py')
    clean=[];artifacts=[record(session/'ledger.json'),record(session/'identity.json')]
    for row in rows:
        path=session/row['task']['name'];result=row['result'];summary=row['summary']
        requested=read(path/'requested.json');config=read(path/'resolved_config.json')
        validation=read(path/'config_validation.json')
        require(result['status']=='completed' and result['recorded_frames']==120,'Incomplete native run')
        require(result['cleanup_owned_job_empty'] and result['exit_code']==0,'Owned process cleanup failed')
        require(validation['passed'] and row['initial_input']['passed'] and summary['hard_checks_passed'],'Contract failure')
        require(requested['runner_sha256']==runner['sha256'],'Runner changed during/after controls')
        require(requested['exe_sha256']==program['exe']['sha256'],'Different executable')
        require(requested['expanded_config']==rows[0]['task']['config'],'Config mismatch')
        require(config['contact_backend']=='ipc' and config['ipc_pcg_forcing']['mode']=='off','Wrong solver mode')
        require(config['pcg_rho_tol']==1e-4 and config['ipc_stopping']['cumulative_tol']==.01
                and config['ipc_stopping']['min_updates']==6,'Stopping requirement changed')
        require(not result['heavy_diagnostics'],'Instrumented native run')
        frames=read(path/'output/stats.json')['frames']
        pcg=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
        require(len(frames)==120 and len(pcg)==summary['directions'],'Missing PCG/frame records')
        require(all(p['rho_tolerance']==1e-4 and not p.get('iteration_limit') and not p.get('breakdown')
                    and p['execution']=='conditional_graph' for p in pcg),'PCG contract differs')
        samples=result['gpu_samples']
        require(samples and not any(s['foreign_compute_pids'] for s in samples),'Foreign compute evidence')
        clean.append(dict(name=row['task']['name'],arm=row['task']['arm'],summary=summary,
            effective_precision=row['effective_precision'],initial_input=row['initial_input'],
            trajectory_vs_first_legacy=row['trajectory'],wall_seconds=result['wall_seconds'],
            minimum_free_mib=min(s['memory.free'] for s in samples),
            resource_status=result['status'],owned_job_empty=result['cleanup_owned_job_empty'],
            newton_exits=sorted({f.get('newton_exit') for f in frames}),
            pcg_stage_us_per_count=summary['phase_ms']['pcg']*1000/summary['pcg']))
        artifacts.extend(record(path/name) for name in
            ('requested.json','resolved_config.json','result.json','config_validation.json','output/stats.json'))
    aggregate={}
    for arm in ('legacy','wide','inverse64'):
        group=[r for r in clean if r['arm']==arm]
        aggregate[arm]={key:statistics.median(r['summary'][key] for r in group) for key in
            ('solver_seconds','pcg','directions','pcg_per_direction','max_stretch','p99_stretch','graph_captures','graph_capture_ms')}
        aggregate[arm]['phase_seconds']={key:statistics.median(r['summary']['phase_ms'][key]/1000 for r in group)
            for key in group[0]['summary']['phase_ms']}
        aggregate[arm]['solver_range_seconds']=[min(r['summary']['solver_seconds'] for r in group),
            max(r['summary']['solver_seconds'] for r in group)]
        aggregate[arm]['pcg_stage_us_per_count']=statistics.median(r['pcg_stage_us_per_count'] for r in group)
    base=aggregate['legacy'];pairs=[]
    for a,b in ((0,1),(0,2),(5,4),(5,3)):
        reference,candidate=clean[a],clean[b]
        match=input_comparison(session/reference['name'],session/candidate['name'])
        require(match['passed'],'Pair initial input mismatch')
        pairs.append(dict(reference=reference['name'],candidate=candidate['name'],initial_input=match,
            speedup=reference['summary']['solver_seconds']/candidate['summary']['solver_seconds'],
            total_time_increase=candidate['summary']['solver_seconds']/reference['summary']['solver_seconds']-1,
            pcg_count_change=candidate['summary']['pcg']/reference['summary']['pcg']-1,
            pcg_phase_time_change=candidate['summary']['phase_ms']['pcg']/reference['summary']['phase_ms']['pcg']-1,
            trajectory=state_comparison(session/reference['name'],session/candidate['name'])))
    for arm in ('wide','inverse64'):
        c=aggregate[arm]
        c.update(speedup=base['solver_seconds']/c['solver_seconds'],
            total_time_increase=c['solver_seconds']/base['solver_seconds']-1,
            pcg_count_change=c['pcg']/base['pcg']-1,
            pcg_phase_time_change=c['phase_seconds']['pcg']/base['phase_seconds']['pcg']-1,
            pcg_stage_us_per_count_change=c['pcg_stage_us_per_count']/base['pcg_stage_us_per_count']-1)
    report=dict(session=str(session),scene='cloth_fixed_bunny_l',order=expected,frames_per_run=120,total_frames=720,
        executable=program['exe'],runner=runner,controller=record(__file__),artifacts=artifacts,
        native_source_file_count=len(identity['native_source_before']),native_source_unchanged=True,
        runs=clean,aggregates=aggregate,pairs=pairs,
        legacy_repeat_trajectory=state_comparison(session/clean[0]['name'],session/clean[5]['name']),
        optimistic_zero_pcg_speedup=base['solver_seconds']/(base['solver_seconds']-base['phase_seconds']['pcg']),
        scope='Same sealed v5 IPC binary and physical inputs, forcing off, existing precision flags only. '
              'Shared WDDM desktop diagnostic, two observations per arm. Net solver time sums frame solver_ms; '
              'not process startup/export wall time. PCG phase/count includes preparation and Graph work, '
              'so is an amortized stage cost, not a measured single-iteration kernel duration. '
              'M-weighted native stopping norms and trajectories differ; common-norm fixed-system CPU evidence is separate. '
              'No original-Stiff or new TOI speedup certificate.',
        quality_certified=False,performance_certified=False,
        decision='Keep legacy precision. Neither existing high-precision flag improves net solver time in either pair. '
                 'No native defaults or key code changes; do not infer all possible mixed-precision designs are disproved.')
    write_new(output,report)
    print(json.dumps(dict(aggregates=aggregate,pairs=[{k:v for k,v in p.items() if k not in ('initial_input','trajectory')} for p in pairs],
        legacy_repeat_trajectory=report['legacy_repeat_trajectory'],optimistic_zero_pcg_speedup=report['optimistic_zero_pcg_speedup'],
        native_source_file_count=report['native_source_file_count']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();analyze(args.session,args.output)
