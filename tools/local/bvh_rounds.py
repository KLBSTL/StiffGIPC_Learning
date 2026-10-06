"""One explicit four-arm BVH round; reuse the sealed Windows runner unchanged."""
import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
from pathlib import Path
from local_plan import ROOT, tasks as reference_tasks
from config import read, expand, digest
from linux_runner import require, child, sha, write_new, verify_files
from local_identity import verify_seal
from windows_runner import execute, gpu_lock
from local_analysis import analyze_one
from quality_analysis import input_comparison
from pool_metrics import state_comparison

ORDERS = {1: ('D1S1', 'D0S1', 'D0S0', 'D1S0'),
          2: ('D1S0', 'D0S0', 'D0S1', 'D1S1'),
          3: ('D1S0', 'D1S1', 'D0S1', 'D0S0')}
V1_COMMIT = 'd155290f75409c300f20c65864888d47b59b76c7'
V1_TOOLS = {'bvh_rounds.py': '6d751076d97be771131a1c96efdf7748ff141a5b0e1089ff725edae55228ae9c',
            'bvh_rounds_test.py': 'da5a51539f8b9e50807ed976bd07b1f00d90670f223421afb7bf23f364d0d29e'}


def plan():
    return {'schema': 'bvh_four_arm_rounds.v1', 'scenes': {'fixed': 59, 'hang': 51},
            'orders': {str(k): list(v) for k, v in ORDERS.items()}, 'max_runs': 24,
            'from_zero': True, 'timeout_seconds': 120, 'pool': False,
            'automatic_next_round': False, 'performance_certified': False,
            'quality_certified': False, 'net_saving_threshold': .05,
            'material_misses_are_diagnostic_not_promotion': True}


def tasks(scene, repeat):
    require(scene in ('fixed', 'hang') and repeat in ORDERS, 'Invalid finite round')
    c = expand(reference_tasks(scene + '_r1')[0]['config'])
    c.update(contact_pool=False, contact_pool_validate=False, refit=True,
             batch=True, reuse=True, discrete_bvh_refit=True, trace_velocity=True)
    rows = []
    for arm in ORDERS[repeat]:
        cfg = c | {'discrete_bvh_refit': arm[1] == '1', 'refit': arm[3] == '1'}
        rows.append({'name': f'{scene}_{arm}_r{repeat}', 'variant': arm,
                     'scene_key': scene, 'repeat': repeat, 'binary': 'active', 'config': cfg})
    return rows


def code_identity():
    return {n: sha(Path(__file__).parent / n) for n in ('bvh_rounds.py', 'bvh_rounds_test.py')}


def verify_tools(recorded, current):
    if recorded == current: return
    # Only the initial two rounds may retain the exact archived controller.
    # Do not rewrite their receipts or silently accept arbitrary older tooling.
    require(recorded == V1_TOOLS, 'Previous controller identity is unsupported')
    for name, expected in recorded.items():
        result = subprocess.run(['git', '-c', 'safe.directory=' + ROOT.as_posix(), 'show',
                                 V1_COMMIT + ':tools/local/' + name], cwd=ROOT,
                                capture_output=True, timeout=10, check=True)
        require(hashlib.sha256(result.stdout).hexdigest() == expected, 'Archived controller differs')


def within_time_budget(result):
    wall = result.get('wall_seconds')
    return type(wall) in (float, int) and math.isfinite(wall) and 0 < wall <= 120


def time_guard(row, result):
    if not within_time_budget(result):
        row['hard_checks_passed'] = False
        row['failures'].append('Absolute elapsed deadline invalid/exceeded; raw result preserved')
    return row


def verify_stage(folder, seal_sha, tools):
    receipt, batch, analysis = (read(folder / n) for n in ('receipt.json', 'batch.json', 'analysis.json'))
    require(receipt['seal_sha256'] == seal_sha, 'Seal changed')
    verify_tools(receipt['tools'], tools)
    require(receipt['batch_sha256'] == sha(folder / 'batch.json') and
            receipt['analysis_sha256'] == sha(folder / 'analysis.json'), 'Receipt changed')
    require(batch['plan'] == plan() and analysis['diagnosis_complete'], 'Previous failed/incomplete round')
    expected = tasks(batch['scene'], batch['repeat'])
    require(analysis['plan'] == plan() and analysis['scene'] == batch['scene'] and
            analysis['repeat'] == batch['repeat'] and
            [r['name'] for r in analysis['runs']] == [t['name'] for t in expected] and
            all(r['hard_checks_passed'] for r in analysis['runs']), 'Analysis identity/ledger differs')
    require(folder.name == f"{batch['scene']}_r{batch['repeat']}" and not batch['skipped'] and
            [r['name'] for r in batch['runs']] == [t['name'] for t in expected], 'Previous ledger differs')
    for row in batch['runs']:
        out = folder / row['name']
        require(sha(out / 'evidence.json') == row['evidence_sha256'], 'Run evidence changed')
        verify_files(out, read(out / 'evidence.json')['files'])
        require(read(out / 'result.json') == row['result'], 'Run result changed')
        require(within_time_budget(row['result']), 'Previous run exceeded elapsed deadline')
        check = folder / (row['name'] + '_check.json')
        if receipt['tools'] != V1_TOOLS:
            require(sha(check) == row['check_sha256'], 'Per-run interpretation changed')
        observed = next(r for r in analysis['runs'] if r['name'] == row['name'])
        require(compact(read(check)) == observed, 'Per-run interpretation/summary differs')
    return receipt


def check_previous(session, scene, repeat, seal_sha, tools, reviewed):
    folders = list(session.iterdir()) if session.exists() else []
    receipts = []
    for folder in folders:
        require(folder.is_dir() and (folder / 'receipt.json').is_file(), 'Preserve incomplete/unknown session')
        receipts.append((verify_stage(folder, seal_sha, tools), folder))
    require(len(receipts) < 6, 'Six finite rounds already used')
    require(not (session / f'{scene}_r{repeat}').exists(), 'Round exists; no retry/resume')
    same = sorted(int(f.name.rsplit('_r', 1)[1]) for _, f in receipts if f.name.startswith(scene + '_r'))
    require(same == list(range(1, repeat)), 'Missing/repeated scene predecessor')
    sequence = sorted(r['sequence'] for r, _ in receipts)
    require(sequence == list(range(1, len(receipts) + 1)), 'Invalid session sequence')
    if receipts:
        latest = max(receipts, key=lambda x: x[0]['sequence'])[1]
        require(reviewed == sha(latest / 'analysis.json'), 'Supply latest actual analysis SHA after review')
    else:
        require(reviewed is None, 'First round has no predecessor')
    return len(receipts) + 1


def compact(row):
    result = {k: row[k] for k in ('name', 'variant', 'hard_checks_passed', 'failures')}
    if row['hard_checks_passed']:
        result.update(material=row['material'], within_original_bounds=row['within_original_bounds'])
        result['timing'] = {kind: {k: v for k, v in values.items() if
                           k in ('frames', 'solver_ms', 'phase_ms', 'exit_assembly_ms', 'unclassified_ms',
                                 'unclassified_includes_unreported_exit_assembly', 'directions',
                                 'pcg_iterations', 'newton_exits')}
                           for kind, values in row['timing_observation'].items() if isinstance(values, dict)}
    return result


def run(root, seal_path, session, scene, repeat, reviewed, gpu=0):
    require(os.name == 'nt' and gpu >= 0, 'Windows/nonnegative GPU required')
    tools = code_identity()
    with gpu_lock(root):
        seal = verify_seal(root, seal_path)
        seq = check_previous(session, scene, repeat, sha(seal_path), tools, reviewed)
        folder = session / f'{scene}_r{repeat}'
        folder.mkdir(parents=True, exist_ok=False)
        ts = tasks(scene, repeat)
        batch = {'plan': plan(), 'scene': scene, 'repeat': repeat, 'runs': [], 'skipped': []}
        rows = []
        for t in ts:
            result = execute(folder, t, seal['programs']['active'], gpu)
            row = time_guard(analyze_one(folder, t, seal), result)
            write_new(folder / (t['name'] + '_check.json'), row)
            batch['runs'].append({'name': t['name'], 'result': result,
                                 'evidence_sha256': sha(folder / t['name'] / 'evidence.json'),
                                 'check_sha256': sha(folder / (t['name'] + '_check.json'))})
            rows.append(row)
            if not row['hard_checks_passed']:
                break
        batch['skipped'] = [t['name'] for t in ts[len(rows):]]
        comparisons = []
        for a, b in itertools.combinations(rows, 2):
            if not (a['hard_checks_passed'] and b['hard_checks_passed']): continue
            inputs = input_comparison(folder / a['name'], folder / b['name'])
            pair = {'left': a['name'], 'right': b['name'], 'initial_inputs': inputs}
            if inputs['passed']: pair['state_difference'] = state_comparison(folder / a['name'], folder / b['name'])
            comparisons.append(pair)
        complete = (len(rows) == 4 and all(r['hard_checks_passed'] for r in rows) and
                    len(comparisons) == 6 and all(p['initial_inputs']['passed'] for p in comparisons))
        verify_seal(root, seal_path)
        require(tools == code_identity(), 'Own tools changed while running')
        analysis = {'plan': plan(), 'scene': scene, 'repeat': repeat, 'runs': [compact(r) for r in rows],
                    'comparisons': comparisons, 'diagnosis_complete': complete,
                    'all_original_bounds_satisfied': complete and all(r['within_original_bounds'] for r in rows),
                    'quality_certified': False, 'performance_certified': False, 'automatic_next_round': False}
        write_new(folder / 'batch.json', batch)
        write_new(folder / 'analysis.json', analysis)
        write_new(folder / 'receipt.json', {'seal_sha256': sha(seal_path), 'tools': tools, 'sequence': seq,
                  'batch_sha256': sha(folder / 'batch.json'), 'analysis_sha256': sha(folder / 'analysis.json')})
        print(json.dumps({'stage': folder.name, 'diagnosis_complete': complete,
                          'material_passed': analysis['all_original_bounds_satisfied'],
                          'analysis_sha256': sha(folder / 'analysis.json'),
                          'runs': analysis['runs']}, ensure_ascii=False), flush=True)
        return analysis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('plan', 'run'))
    p.add_argument('--seal', default='build/local_step3_20261006/local_diagnostic_seal.json')
    p.add_argument('--session', required=True)
    p.add_argument('--scene', choices=('fixed', 'hang'), required=True)
    p.add_argument('--round', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--reviewed-previous-sha')
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()
    if a.action == 'plan':
        print(json.dumps({'plan': plan(), 'tasks': tasks(a.scene, a.round)}, indent=2)); return 0
    session = child(ROOT, a.session)
    require(session.is_relative_to(ROOT / 'runs') and session != ROOT / 'runs', 'Session must be below runs/')
    result = run(ROOT, child(ROOT, a.seal), session, a.scene, a.round, a.reviewed_previous_sha, a.gpu)
    return int(not result['diagnosis_complete'])


if __name__ == '__main__': raise SystemExit(main())
