"""CPU-only final analysis; absent legacy counters remain unavailable."""
from pathlib import Path
import hashlib
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
SESSION = ROOT / 'runs/two_contact_scenes_20261008'
sys.path[:0] = [str(ROOT / 'tools' / p) for p in ('bench','diagnostic')]
from linux_runner import verify_files, require
from quality_analysis import input_comparison
from config import read, digest

ARMS = ('base','graph','all')
SCENES = ('cloth_table_l','cloth_stack10_l')


def record(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def observed(rows, key):
    values = [r[key] for r in rows if key in r]
    return dict(observed_records=len(values), expected_records=len(rows),
        complete=len(values)==len(rows) and bool(rows), total=sum(values) if values else None)


def main():
    ledger = read(SESSION / 'LEDGER.json')
    seal = read(SESSION / 'IDENTITY.json')
    require(ledger['status'] in ('completed','failed') and len(ledger['rows']) in (17,18),
            'Insufficient completed stage: no final speed report')
    require([r['name'] for r in ledger['rows']]==[t['name'] for t in seal['tasks'][:len(ledger['rows'])]], 'Task order differs')
    failures = []
    if ledger['status']=='failed':
        require(len(ledger['rows'])==17 and len(ledger['failures'])==1 and
                ledger['failures'][0]['name']==seal['tasks'][-1]['name'], 'Unexpected failure or incomplete cohort')
        folder = SESSION / ledger['failures'][0]['name']
        verify_files(folder, read(folder/'evidence.json')['files'])
        result, req = read(folder/'result.json'), read(folder/'requested.json')
        require(result['status']=='memory_budget' and result['recorded_frames']==47 and
                result['cleanup_owned_job_empty'] is True and req['expanded_config']==seal['tasks'][-1]['config'],
                'Different failed attempt; do not reinterpret')
        failures.append(dict(name=folder.name, status=result['status'], recorded_frames=47,
            included_in_speed=False, memory_budget_mib=req['memory_budget_mib'],
            memory_free_before_mib=req['gpu_before'][-1]['memory.free'],
            memory_free_at_stop_mib=result['gpu_samples'][-1]['memory.free'],
            cleanup_owned_job_empty=True, evidence_sha256=record(folder/'evidence.json')['sha256'],
            scope='Total-device allocation delta exceeded guard; no inference that a numerical solver failure or a specific foreign process caused it.'))
    else:
        require(len(ledger['rows'])==18 and not ledger['failures'], 'Incomplete successful stage')
    rows = []
    for old, task in zip(ledger['rows'], seal['tasks']):
        folder = SESSION / old['name']
        files = read(folder / 'evidence.json')['files']
        verify_files(folder, files)
        req, result = read(folder / 'requested.json'), read(folder / 'result.json')
        identity = seal['programs'][task['binary']]
        require(result['status']=='completed' and result['recorded_frames']==120 and
                result['finite'] is True and result['cleanup_owned_job_empty'] is True and
                req['exe_sha256']==identity['exe']['sha256'] and req['source_digest']==identity['source_digest'] and
                req['expanded_config']==task['config'] and digest(req['expanded_config'])==task['config_sha256'],
                'Result/identity/config changed')
        frames = read(folder / 'output/stats.json')['frames']
        ns = [n for f in frames for n in f['newton']]
        updates = [n for n in ns if 'pcg' in n]
        pcgs = [n['pcg'] for n in ns if 'pcg' in n]
        require(len(frames)==120 and pcgs and not any(p.get('iteration_limit') or p.get('breakdown') for p in pcgs)
                and not any(f.get('newton_exit')=='iteration_limit' for f in frames)
                and not any(n.get('line_search_failure') for n in ns) and not read(folder/'output/stats.json').get('failure'),
                'Native reported failure')
        raw = (folder / 'trace/topology.bin').read_bytes()
        import struct
        vertices, faces, tets = struct.unpack_from('<III', raw)
        geometry = [f['contact_geometry'] for f in frames if 'contact_geometry' in f]
        row = {k:old[k] for k in ('name','scene','arm','repeat','solver_seconds','wall_seconds','material','exits')}
        row.update(vertices=vertices, faces=faces, tets=tets,
            phases_ms=old['summary']['recorded_phase_ms'], directions=len(pcgs),
            pcg_iterations=observed(pcgs,'iterations'), energy_evaluations=observed(updates,'energy_evaluations'),
            energy_backtracks=observed(updates,'energy_backtracks'),
            intersection_backtracks=observed(updates,'intersection_backtracks'),
            energy_counter_scope='PCG direction/update rows; standalone movement exit rows do not execute a line search.',
            active_pair_observed_records=sum('active_pairs' in n for n in ns),
            active_pair_peak=max((n['active_pairs'] for n in ns if 'active_pairs' in n), default=None),
            active_pair_frames=sum(any(n.get('active_pairs', 0)>0 for n in f['newton']) for f in frames)
                if any('active_pairs' in n for n in ns) else None,
            native_geometry_frames=len(geometry),
            native_self_pair_peak=max((g['native_narrow_self_pairs'] for g in geometry), default=None),
            actual_velocity_available=old['arm']!='base', hard_checks_passed=True,
            breakdown_telemetry_complete=all('breakdown' in p for p in pcgs),
            evidence_sha256=record(folder/'evidence.json')['sha256'])
        rows.append(row)
    scenes = []
    for scene in SCENES:
        rr = [r for r in rows if r['scene']==scene]
        complete_repeats = [i for i in (1,2,3) if
            {r['arm'] for r in rr if r['repeat']==i}==set(ARMS)]
        require(len(complete_repeats)>=2, 'Need at least two complete three-arm pairs')
        groups = {a:[r for r in rr if r['arm']==a and r['repeat'] in complete_repeats] for a in ARMS}
        pairs, checks = [], []
        for repeat in complete_repeats:
            arms = {r['arm']:r for r in rr if r['repeat']==repeat}
            for left, right in (('base','graph'),('base','all'),('graph','all')):
                c = input_comparison(SESSION/arms[left]['name'], SESSION/arms[right]['name'])
                require(c['passed'], 'Physical inputs differ')
                checks.append(dict(repeat=repeat, left=left, right=right, **c))
            t = {a:r['solver_seconds'] for a,r in arms.items()}
            pairs.append(dict(repeat=repeat, stiff_over_graph=t['base']/t['graph'],
                stiff_over_all=t['base']/t['all'], graph_over_all=t['graph']/t['all']))
        medians = {a:statistics.median(r['solver_seconds'] for r in groups[a]) for a in ARMS}
        scenes.append(dict(scene=scene, complete_primary_repeats=complete_repeats,
            completed_per_arm={a:sum(r['arm']==a for r in rr) for a in ARMS},
            excluded_complete_rows=[r['name'] for r in rr if r['repeat'] not in complete_repeats],
            median_solver_seconds=medians,
            paired_ratios=pairs, paired_medians={k:statistics.median(r[k] for r in pairs)
                for k in ('stiff_over_graph','stiff_over_all','graph_over_all')},
            solver_seconds_range={a:[min(r['solver_seconds'] for r in rs),max(r['solver_seconds'] for r in rs)] for a,rs in groups.items()},
            phase_ms_medians={a:{p:statistics.median(r['phases_ms'][p] for r in rs)
                for p in ('assembly','pcg','ccd','line_search','state_update')} for a,rs in groups.items()},
            workload_medians={a:{k:statistics.median(r[k] if k=='directions' else r[k]['total'] for r in rs)
                if all(k=='directions' or r[k]['complete'] for r in rs) else None
                for k in ('directions','pcg_iterations','energy_evaluations','energy_backtracks')} for a,rs in groups.items()},
            material_ranges={a:{k:[min(r['material'][k] for r in rs),max(r['material'][k] for r in rs)]
                for k in ('max_stretch','p99_stretch','fixed_drift_m')} for a,rs in groups.items()},
            physical_inputs=checks))
    result = dict(schema='two_contact_scenes.analysis.v1', status='completed' if not failures else 'partial_stage_with_two_valid_scenes',
        planned_runs=18, completed_runs=len(rows), failures=failures,
        scenes=scenes, rows=rows, performance_certified=False, quality_certified=False,
        scope='Shared WDDM diagnostic, same numeric configuration, three rotated pairs. Missing base energy/breakdown/velocity remain unobserved; no independent CCD or held-out material certification.',
        program_identity={k:dict(exe=v['exe'], source_digest=v['source_digest'], capabilities=v['capabilities'])
            for k,v in seal['programs'].items()},
        original_seal=record(SESSION/'IDENTITY.json'), resume_seal=record(SESSION/'RESUME_IDENTITY.json'),
        derived_runtime_inputs=read(SESSION/'RESUME_IDENTITY.json')['cache_inputs'])
    (REPORT/'ANALYSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'completed':len(rows), 'failures':failures, 'scenes':[{k:s[k] for k in
        ('scene','complete_primary_repeats','median_solver_seconds','paired_medians','workload_medians','phase_ms_medians','material_ranges')} for s in scenes]},indent=2),flush=True)


if __name__=='__main__':
    main()
