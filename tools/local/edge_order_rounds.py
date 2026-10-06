"""Two manually reviewed, same-process edge-order probes; no automatic next run."""
import argparse
import json
import statistics
from pathlib import Path
import bvh_rounds
from local_plan import ROOT
from config import read, expand
from linux_runner import child, require, sha, write_new, verify_files
from local_identity import verify_seal
from local_analysis import analyze_one
from windows_runner import execute, gpu_lock


def task(scene):
    c = bvh_rounds.tasks(scene, 1)[0]['config'].copy()
    c.update(discrete_bvh_refit=True, refit=True, edge_query_order='raw',
             diagnostics=['edge_order'], edge_order_probe_frames='1,57' if scene == 'fixed' else '1,41')
    return {'name': scene + '_probe', 'binary': 'active', 'variant': 'raw_with_private_probe',
            'scene_key': scene, 'repeat': 1, 'config': expand(c)}


def code_identity():
    return {n: sha(Path(__file__).parent / n) for n in ('edge_order_rounds.py', 'edge_order_test.py')}


def probe_analysis(out, t):
    p = out / 'edge_order_probe.jsonl'
    rows = [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
    expected = list(map(int, t['config']['edge_order_probe_frames'].split(',')))
    require([r['frame'] for r in rows] == expected, 'Missing/repeated/out-of-order probe frame')
    result = []
    for r in rows:
        require(r['passed'] and r['production_order'] == 'raw' and r['first_query_only'] and
                r['inputs_read_only'] and r['production_scratch_untouched'], 'Probe contract failed')
        require(r.get('leaf_permutation_valid') and r['per_face_mismatches'] == 0 and
                r['invalid_or_unwritten_faces'] == 0 and r['compared_faces'] == r['face_count'] and
                r['raw_any'] == r['leaf_any'] == r['raw_diagnostic_any'], 'Probe coverage mismatch')
        require(r['timed_pairs'] == 7 and len(r['pairs']) == 7 and
                r['warmups_per_mode'] == 2 and r['speed_evidence'], 'Invalid microbenchmark budget')
        pairs = r['pairs']
        require(all(x['index'] == i and x['first'] == ('raw' if i % 2 == 0 else 'leaf')
                    for i, x in enumerate(pairs)), 'Microbenchmark is not interleaved')
        require(all(0 < x[k] < float('inf') for x in pairs
                    for k in ('raw_kernel_ms', 'leaf_kernel_ms')), 'Invalid event time')
        ratios = [x['raw_kernel_ms'] / x['leaf_kernel_ms'] for x in pairs]
        median = statistics.median(ratios)
        result.append({'frame': r['frame'], 'faces': r['face_count'], 'edges': r['edge_count'],
                       'hit_faces': r['per_face_hit_count'], 'coverage_passed': True,
                       'paired_speedups': ratios, 'median_speedup': median,
                       'median_kernel_time_saving': 1 - 1 / median,
                       'raw_median_ms': statistics.median(x['raw_kernel_ms'] for x in pairs),
                       'leaf_median_ms': statistics.median(x['leaf_kernel_ms'] for x in pairs),
                       'host_readback_ms': r['host_readback_ms'],
                       'diagnostic_host_ms': r['diagnostic_host_ms']})
    return result


def previous(session, scene, seal_sha, tools, reviewed):
    folders = set(session.iterdir()) if session.exists() else set()
    if scene == 'fixed':
        require(not folders and reviewed is None, 'First probe requires a fresh session'); return
    prior = session / 'fixed'
    require(folders == {prior}, 'Probe prefix incomplete/unknown; no retry or resume')
    r = read(prior / 'receipt.json')
    require(r['seal_sha256'] == seal_sha and r['tools'] == tools and
            r['analysis_sha256'] == sha(prior / 'analysis.json') == reviewed,
            'Supply actual completed predecessor SHA after review')
    a = read(prior / 'analysis.json')
    require(a['diagnosis_complete'], 'Failed predecessor blocks next probe')
    out = prior / 'fixed_probe'
    require(r['evidence_sha256'] == sha(out / 'evidence.json'), 'Predecessor evidence changed')
    verify_files(out, read(out / 'evidence.json')['files'])


def run(seal_path, session, scene, reviewed_bvh, reviewed):
    bvh_path = ROOT / 'reports/local_rounds_20261006/BVH_RESULTS.json'
    require(sha(bvh_path) == reviewed_bvh and read(bvh_path)['complete_runs'] == 24,
            'Explicit completed BVH review SHA required')
    tools = code_identity()
    with gpu_lock(ROOT):
        seal = verify_seal(ROOT, seal_path)
        previous(session, scene, sha(seal_path), tools, reviewed)
        folder = session / scene
        folder.mkdir(parents=True, exist_ok=False)
        t = task(scene)
        write_new(folder / 'started.json', {'task': t, 'seal_sha256': sha(seal_path),
                  'reviewed_bvh_sha256': reviewed_bvh, 'reviewed_previous_sha256': reviewed, 'tools': tools})
        res = execute(folder, t, seal['programs']['active'], 0)
        row = bvh_rounds.time_guard(analyze_one(folder, t, seal), res)
        write_new(folder / 'check.json', row)
        analysis = {'scene': scene, 'diagnosis_complete': False, 'errors': list(row['failures']),
                    'full_scene_timing_usable': False, 'quality_certified': False,
                    'performance_certified': False, 'automatic_next_round': False}
        if row['hard_checks_passed']:
            try:
                analysis['probes'] = probe_analysis(folder / t['name'], t)
                analysis['material'] = row['material']
                analysis['within_original_bounds'] = row['within_original_bounds']
                analysis['diagnosis_complete'] = True
            except (ValueError, KeyError, FileNotFoundError) as e:
                analysis['errors'].append(type(e).__name__ + ': ' + str(e))
        verify_seal(ROOT, seal_path)
        require(tools == code_identity(), 'Probe controller changed during execution')
        write_new(folder / 'analysis.json', analysis)
        write_new(folder / 'receipt.json', {'seal_sha256': sha(seal_path), 'tools': tools,
                  'analysis_sha256': sha(folder / 'analysis.json'),
                  'evidence_sha256': sha(folder / t['name'] / 'evidence.json')})
        print(json.dumps(analysis | {'analysis_sha256': sha(folder / 'analysis.json')}, ensure_ascii=False))
        return int(not analysis['diagnosis_complete'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seal', required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--scene', choices=('fixed', 'hang'), required=True)
    p.add_argument('--reviewed-bvh-sha', required=True)
    p.add_argument('--reviewed-previous-sha')
    a = p.parse_args()
    session = child(ROOT, a.session)
    require(session.is_relative_to(ROOT / 'runs') and session != ROOT / 'runs', 'Session below runs/ required')
    return run(child(ROOT, a.seal), session, a.scene, a.reviewed_bvh_sha, a.reviewed_previous_sha)


if __name__ == '__main__': raise SystemExit(main())
