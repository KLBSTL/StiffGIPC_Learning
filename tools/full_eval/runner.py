"""One explicit Linux full-evaluation step; no build, SSH, retry or cleanup."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'tools/bench'), str(ROOT/'tools/diagnostic')]
from plan import plan, tasks
from config import read, digest, expand
from linux_runner import (require, child, sha, inventory, verify_files, write_new,
                          verify_manifest, gpu_lock, execute as execute_active)
from fixed_quality import base_seal, compiler_identity, execute_base


def analyzer():
    # Local import keeps plan/identity/CPU contracts usable before the independent
    # analyzer is delivered. The seal includes this module before real execution.
    import analysis
    require(Path(analysis.__file__).resolve() == HERE/'analysis.py', 'Wrong full-evaluation analyzer import')
    return analysis


def controller_files(root):
    paths = []
    for name in ('tools/full_eval', 'tools/diagnostic'):
        paths += [p for p in child(root, name).rglob('*')
                  if p.is_file() and p.suffix in ('.py', '.json', '.md') and '__pycache__' not in p.parts]
    require(child(root, 'tools/full_eval/analysis.py').is_file(), 'Independent CPU analyzer is required before sealing')
    return inventory(root, paths)


def match_build_programs(packet, active, baseline):
    for kind, manifest in (('active', active), ('base', baseline)):
        exe = packet['builds'][kind]['exe']
        require(exe['path'] == manifest['exe_path'] and exe['sha256'] == manifest['exe_sha256'],
                'Actual object/link build attachment and executable manifest differ: '+kind)


def verify_build_attachment(root, attachment, active, baseline):
    import build_identity
    packet = build_identity.verify_identity(root, child(root, attachment['path']), attachment['sha256'])
    match_build_programs(packet, active, baseline)
    return packet


def baseline_identity(root, build_log, packet):
    """Reviewed adapter for this plan's eight unused baseline-only caches.

    Call only after verify_identity has validated the actual build attachment.
    This explicit branch does not catch or bypass errors from the old seal.
    Every extra asset remains in the baseline source SHA inventory.
    """
    import build_identity
    assets = build_identity.assets_equivalence(root, packet['sources'])
    require(assets == packet['assets_equivalence'], 'Build attachment Assets proof differs')
    if assets['roots_identical']:
        return base_seal(root, build_log)
    require({r['path'] for r in assets['ignored_baseline_only']} == build_identity.UNUSED_BASE_CACHES,
            'Baseline adapter permits exactly the eight reviewed unused caches')
    scenes = {'cloth_hang_l','cloth_fixed_bunny_l','bunny_cloth_bunny_l'}
    require(set(plan()['scenes'].values()) == scenes
            and {s['scene'] for s in assets['selected_scene_references']} == scenes,
            'Baseline cache exception is limited to the three reviewed scenes')
    base = child(root, 'baseline'); build = child(root, 'build/base')
    cache = build/'CMakeCache.txt'
    home = next((line.split('=',1)[1] for line in cache.read_text().splitlines()
                 if line.startswith('CMAKE_HOME_DIRECTORY:')), None)
    require(home and Path(home).resolve() == base, 'Baseline build root differs')
    native = {p.resolve() for p in (base/'StiffGIPC').rglob('*') if p.suffix in ('.cu','.cpp')}
    observed = packet['builds']['base']
    require(observed['counts'] == {'tu':35,'cuda':31,'cxx':4}, 'Expected 35 baseline translation units')
    actual = set()
    for row in observed['objects']:
        source = child(root, row['source']).resolve()
        require(source in native and source not in actual, 'Baseline actual compile coverage differs')
        definitions = [arg.split('=',1)[1].strip('"') for arg in row['actual_argv']
                       if arg.startswith('-DGIPC_ASSETS_DIR=')]
        require(len(definitions) == 1 and Path(definitions[0]).resolve() == base/'Assets',
                'Baseline actual compile lacks the exact baseline Assets definition')
        actual.add(source)
    require(actual == native and len(actual) == 35, 'Baseline compile source coverage mismatch')
    stopping = base/'StiffGIPC/core/GIPC.cu'
    text = stopping.read_text()
    require(re.search(r'Kmin\s*=\s*6\s*;', text)
            and re.search(r'beta\s*<=\s*Newton_solver_threshold', text),
            'Baseline source stopping semantics not identified')
    require(all(expand(t['config'])['ipc_newton_tol'] == .01
                and expand(t['config'])['ipc_min_updates'] == 6
                for t in tasks() if t['binary'] == 'base'), 'Baseline runtime stopping request changed')
    require(child(root, build_log) == child(root, observed['build_logs']['build']['path']),
            'Baseline build log differs from actual build attachment')
    source_paths = [child(root,r['path']) for r in packet['sources'] if r['path'].startswith('baseline/')]
    sources = inventory(root, source_paths)
    expected = [{k:r[k] for k in ('path','bytes','sha256')} for r in packet['sources'] if r['path'].startswith('baseline/')]
    require(sources == sorted(expected,key=lambda r:r['path']), 'Baseline source/input differs from actual build attachment')
    evidence_paths = [child(root,r['path']) for r in observed['build_evidence']]
    evidence_paths += [child(root,r['path']) for r in observed['build_logs'].values()]
    exe = child(root,observed['exe']['path'])
    require(exe == build/'gipc' and sha(exe) == observed['exe']['sha256'], 'Baseline executable differs')
    return {'schema':'baseline_build_identity.v1','sources':sources,'source_digest':digest(sources),
            'exe_path':exe.relative_to(root).as_posix(),'exe_sha256':sha(exe),'compile_source_count':35,
            'build_evidence':inventory(root,evidence_paths),'missing_build_evidence':[],
            'compiler_identity':compiler_identity(cache),'assets_equivalence':assets,
            'stopping_source_evidence':{'path':stopping.relative_to(root).as_posix(),'sha256':sha(stopping),
                'min_updates':6,'cumulative_tolerance':'Newton_solver_threshold','requested_tolerance':.01,
                'runtime_resolved_available':False},
            'capabilities':{'resolved_config':False,'actual_velocity':False},
            'adapter':'Reviewed three-scene, eight unreferenced-cache exception; all baseline extras retain source SHA.',
            'scope':'Actual object/link attachment plus full baseline source/input identity and unchanged runtime stop request.'}


def seal(root, active_manifest, base_build_log, output, build_identity_path):
    root = Path(root).resolve()
    active_path = child(root, active_manifest)
    active = verify_manifest(root, active_path)
    attachment = {'path': build_identity_path, 'sha256': sha(child(root, build_identity_path))}
    import build_identity
    packet = build_identity.verify_identity(root, child(root, build_identity_path), attachment['sha256'])
    baseline = baseline_identity(root, base_build_log, packet)
    match_build_programs(packet, active, baseline)
    files = controller_files(root)
    value = {'schema': 'full_eval_seal.v1', 'plan': plan(), 'plan_sha256': digest(plan()),
             'active_manifest_path': active_manifest, 'active_manifest_sha256': sha(active_path),
             'active_manifest': active, 'base_manifest': baseline,
             'build_identity_attachment': attachment,
             'controller_files': files, 'controller_sha256': digest(files),
             'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'prior_failed_guard_is_not_a_precondition': True}
    write_new(child(root, output), value)
    return value


def verify_seal(root, seal_path):
    value = read(seal_path)
    require(value['schema'] == 'full_eval_seal.v1', 'Wrong seal schema')
    require(value['plan'] == plan() and value['plan_sha256'] == digest(plan()), 'Full evaluation plan changed')
    require(controller_files(root) == value['controller_files']
            and digest(value['controller_files']) == value['controller_sha256'], 'Controller/analyzer identity changed')
    active_path = child(root, value['active_manifest_path'])
    require(sha(active_path) == value['active_manifest_sha256'], 'Active build manifest changed')
    require(verify_manifest(root, active_path) == value['active_manifest'], 'Active build identity differs')
    baseline = value['base_manifest']
    require(digest(baseline['sources']) == baseline['source_digest'], 'Baseline source digest differs')
    verify_files(root, baseline['sources']); verify_files(root, baseline['build_evidence'])
    require(sha(child(root, baseline['exe_path'])) == baseline['exe_sha256'], 'Baseline executable changed')
    for row in baseline['compiler_identity']['files']:
        require(sha(Path(row['resolved_path'])) == row['sha256'], 'Baseline compiler changed')
    verify_build_attachment(root, value['build_identity_attachment'], value['active_manifest'], baseline)
    return value


def session_path(root, relative):
    path = child(root, relative)
    require(path.is_relative_to(Path(root).resolve()/'runs') and path != Path(root).resolve()/'runs',
            'Use a new explicit session below runs/')
    return path


def initialize(root, seal_name, session_name):
    seal_path = child(root, seal_name)
    value = verify_seal(root, seal_path)
    session = session_path(root, session_name)
    require(not session.exists(), 'Session exists; initialization never overwrites it')
    session.mkdir(parents=True)
    write_new(session/'session.json', {'schema': 'full_eval_session.v1', 'seal_path': seal_name,
              'seal_sha256': sha(seal_path), 'plan_sha256': value['plan_sha256']})
    write_new(session/'plan.json', value['plan'])
    return {'session': str(session), 'planned_runs': 75, 'maximum_gpu_runs': 78}


def verify_retained_evidence(root, run, evidence_sha256):
    evidence = read(run/'evidence.json')['files']
    require(len({r['path'] for r in evidence}) == len(evidence), 'Duplicate evidence paths')
    missing = []; raw = {}
    for record in evidence:
        path = child(run, record['path'])
        is_raw = path.suffix == '.bin' and (record['path'].startswith('trace/') or record['path'] == 'final.bin')
        if is_raw:
            raw[record['path']] = record
            if not path.is_file(): missing.append(record['path'])
            else: require(path.stat().st_size == record['bytes'], 'Raw evidence size changed')
        else:
            verify_files(run, [record])
    receipt_path = child(run, 'archive_receipt.json')
    if not receipt_path.exists():
        require(not missing, 'Raw evidence missing without a durable archive receipt')
        # Without a durable original archive, size alone cannot detect changed
        # same-sized raw. Verify all remaining original files byte-for-byte.
        verify_files(run, evidence)
        return
    require(receipt_path.is_file(), 'Raw evidence missing without a durable archive receipt')
    receipt = read(receipt_path)
    require(receipt.get('schema') == 'full_eval_raw_archive.v1'
            and receipt.get('download_verified') is True
            and receipt.get('evidence_sha256') == evidence_sha256, 'Archive receipt/evidence identity invalid')
    removed = receipt.get('removed_raw_paths')
    require(isinstance(removed, list) and all(isinstance(p, str) for p in removed)
            and len(set(removed)) == len(removed) and set(missing) <= set(removed) <= raw.keys(),
            'Archive removal list is not bound to original raw evidence')
    local_name = receipt.get('local_archive_name')
    require(isinstance(local_name, str) and local_name and '/' not in local_name and '\\' not in local_name,
            'Local archive receipt must identify a filename')
    archive = child(root, receipt['archive_path'])
    require(archive.is_file() and archive.stat().st_size == receipt.get('archive_bytes')
            and sha(archive) == receipt.get('archive_sha256'), 'Durable archive bytes/hash differ')
    # A verified archive supplies the preserved original raw identity. This is
    # not a claim that raw copies still on disk were rehashed by this branch.


def read_entries(session, root):
    """Verify the immutable metadata chain; raw payload may be archived later.

    The per-run analyzer already verified exports before this receipt was saved.
    Original evidence.json is retained and hash-bound, never rewritten to hide
    removed raw files. Remote transfer/pruning requires its own durable receipt.
    """
    paths = sorted(child(session, 'ledger').glob('*.json'))
    declared = tasks(); entries = []; previous = None
    for index, path in enumerate(paths, 1):
        require(path.name == f'{index:04d}.json' and not path.is_symlink(), 'Ledger gap or linked entry')
        row = read(path)
        require(row['task'] == declared[index-1] and row['index'] == index, 'Ledger task/order changed')
        require(row['predecessor_sha256'] == previous, 'Ledger predecessor changed')
        for key in ('analysis', 'summary'):
            artifact = child(session, row[key+'_path'])
            require(sha(artifact) == row[key+'_sha256'], key+' artifact changed')
        require(read(child(session, row['analysis_path'])) == row['analysis'], 'Embedded analysis differs')
        if row['run_created']:
            run = child(session, row['task']['name'])
            require(sha(run/'evidence.json') == row['evidence_sha256'], 'Original evidence list changed')
            require(sha(run/'result.json') == row['result_sha256'] and read(run/'result.json') == row['result'], 'Run result changed')
            verify_retained_evidence(root, run, row['evidence_sha256'])
        previous = sha(path); entries.append(row)
    return entries


def blocked_configs(entries):
    return {(r['task']['scene_key'], r['task']['variant']) for r in entries if r['status'] == 'hard_failed'}


def skip_reason(task, entries, summary):
    if (task['scene_key'], task['variant']) in blocked_configs(entries):
        return 'This scene/arm had a prior hard failure; no repeated long run.'
    if task['phase'] != 'stability':
        return None
    expected = [t for t in tasks() if t['scene_key'] == task['scene_key'] and t['phase'] != 'stability']
    rows = {r['task']['name']: r for r in entries}
    if len(expected) != 25 or any(t['name'] not in rows or rows[t['name']]['status'] != 'completed'
        or rows[t['name']]['analysis'].get('hard_checks_passed') is not True for t in expected):
        return 'Not all 25 declared 100-frame runs completed with hard checks passed.'
    gate = summary.get('long_run_gates', {}).get(task['scene_key'], {})
    if gate.get('eligible') is not True:
        return 'Frozen material/holdout gate not satisfied: '+str(gate.get('reasons', ['missing gate']))
    return None


def advance(root, seal_name, session_name, gpu=0):
    root = Path(root).resolve(); seal_path = child(root, seal_name)
    value = verify_seal(root, seal_path)
    session = session_path(root, session_name)
    header = read(child(session, 'session.json'))
    require(header['seal_path'] == seal_name and header['seal_sha256'] == sha(seal_path)
            and header['plan_sha256'] == digest(plan()) and read(child(session, 'plan.json')) == plan(), 'Session seal/plan differs')
    with gpu_lock(root):
        entries = read_entries(session, root)
        require(not any(r.get('analysis_exception') for r in entries), 'Prior CPU analysis/controller exception; advancement stopped without retry')
        if len(entries) == len(tasks()):
            return {'status': 'plan_finished', 'processed_tasks': len(entries), 'gpu_launched': False}
        task = tasks()[len(entries)]; index = task['index']
        out = child(session, task['name'])
        require(not out.exists(), 'Run exists without a finalized ledger; retain it and investigate, never retry')
        previous_summary = read(child(session, entries[-1]['summary_path'])) if entries else {}
        reason = skip_reason(task, entries, previous_summary)
        row = {'schema': 'full_eval_ledger.v1', 'index': index, 'task': task,
               'predecessor_sha256': sha(session/'ledger'/f'{index-1:04d}.json') if entries else None,
               'seal_sha256': header['seal_sha256'], 'plan_sha256': header['plan_sha256'],
               'run_created': False, 'gpu_launched': False, 'analysis_exception': False}
        if reason:
            result = {'status': 'skipped', 'reason': reason}
            check = {'name': task['name'], 'hard_checks_passed': False, 'quality_status': 'not_run', 'failures': [reason]}
            row['status'] = 'skipped'
            row['skip_reason'] = reason
        else:
            try:
                manifest = value['base_manifest' if task['binary'] == 'base' else 'active_manifest']
                result = (execute_base if task['binary'] == 'base' else execute_active)(root, session, task, manifest, gpu)
                row['run_created'] = True
                row['gpu_launched'] = (out/'process.json').is_file()
                row['evidence_sha256'] = sha(out/'evidence.json'); row['result_sha256'] = sha(out/'result.json')
                check = analyzer().analyze_run(session, task, value)
                require(type(check.get('hard_checks_passed')) is bool, 'CPU analyzer did not give a boolean hard gate')
                if result['status'] == 'completed':
                    wall = result.get('wall_seconds')
                    if type(wall) not in (int, float) or not math.isfinite(wall) or not 0 <= wall <= expand(task['config'])['timeout_seconds']:
                        check = dict(check, hard_checks_passed=False,
                                     failures=list(check.get('failures', []))+['Controller absolute wall-time limit failed.'])
                row['status'] = 'completed' if result['status'] == 'completed' and check['hard_checks_passed'] else 'hard_failed'
            except Exception as exc:
                result = read(out/'result.json') if (out/'result.json').is_file() else {'status': 'controller_failed'}
                check = {'name': task['name'], 'hard_checks_passed': False, 'quality_status': 'analysis_failed',
                         'failures': [type(exc).__name__+': '+str(exc)]}
                row.update(status='hard_failed', analysis_exception=True)
                # Preserve crash artifacts, even when the underlying runner did
                # not finish its evidence list. A later invocation cannot retry.
                row['run_created'] = (out/'evidence.json').is_file() and (out/'result.json').is_file()
                if row['run_created']:
                    row['evidence_sha256'] = sha(out/'evidence.json'); row['result_sha256'] = sha(out/'result.json')
        analysis_path = f'analysis/{task["name"]}.json'
        write_new(child(session, analysis_path), check)
        row.update(result=result, analysis=check, analysis_path=analysis_path,
                   analysis_sha256=sha(child(session, analysis_path)))
        try:
            summary = analyzer().analyze_session(session, value, plan(), entries+[row])
        except Exception as exc:
            row['analysis_exception'] = True
            summary = {'schema': 'full_eval_analysis_error.v1', 'quality_certified': False,
                       'performance_certified': False, 'long_run_gates': {},
                       'error': type(exc).__name__+': '+str(exc)}
        summary_path = f'summary/{index:04d}.json'; write_new(child(session, summary_path), summary)
        row.update(summary_path=summary_path, summary_sha256=sha(child(session, summary_path)))
        write_new(child(session, f'ledger/{index:04d}.json'), row)
        return {'status': row['status'], 'task': task['name'], 'index': index,
                'gpu_launched': row['gpu_launched'], 'analysis_exception': row['analysis_exception'],
                'quality_status': check.get('quality_status'), 'ledger': f'ledger/{index:04d}.json'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('seal'); s.add_argument('--active-manifest', required=True)
    s.add_argument('--base-build-log', required=True); s.add_argument('--output', required=True)
    s.add_argument('--build-identity', required=True)
    for command in ('init', 'run-next'):
        q = sub.add_parser(command); q.add_argument('--seal', required=True); q.add_argument('--session', required=True)
        if command == 'run-next': q.add_argument('--gpu', type=int, default=0)
    a = p.parse_args(); root = a.root.resolve()
    if a.command == 'seal': result = seal(root, a.active_manifest, a.base_build_log, a.output, a.build_identity)
    elif a.command == 'init': result = initialize(root, a.seal, a.session)
    else: result = advance(root, a.seal, a.session, a.gpu)
    import json
    print(json.dumps({k: result[k] for k in ('status','task','index','gpu_launched','analysis_exception','quality_status','session','maximum_gpu_runs') if k in result}))
    return 2 if result.get('analysis_exception') else 0


if __name__ == '__main__':
    raise SystemExit(main())
