"""CPU-only contracts for frozen ranges, pairing, missing data and archival."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis import (MATERIAL_DIRECTIONS, analyze_session, compare_material,
                      freeze_bounds, paired_statistics, previous_comparisons,
                      read_timecost, timing_summary)
from config import digest, sha
from plan import plan


def material(stretch=1.1):
    return {'max_stretch': stretch, 'p99_stretch': 1.05, 'fixed_drift_m': 0.,
            'fem_min_J': None, 'fem_nonpositive_peak': 0, 'fem_nonpositive_frame_sum': 0,
            'fem_negative_volume_peak': 0., 'abd_min_J': 1., 'abd_nonpositive_frames': 0}


def make_row(task):
    seconds = {'original_stiff': 20., 'combined_graph': 10., 'ipc_host': 15., 'ipc_graph': 12.}[task['arm']]
    if task['phase'] in ('calibration', 'verification'):
        seconds = 999.  # Must never contaminate actual paired denominators.
    row = {key: task.get(key) for key in ('index', 'name', 'phase', 'repeat', 'pair',
           'calibration_index', 'verification_index', 'scene_key', 'arm', 'binary')}
    row.update(hard_checks_passed=True, quality_status='material_pending', failures=[],
               gpu_uuid='same-gpu', material=material(),
               timing={'solver_seconds': seconds, 'process_wall_seconds': seconds + 1},
               work={'linear_directions': 6, 'pcg_iterations': 60},
               input_identity={'topology': 'identical', 'initial_actual_velocity': None},
               paired_comparisons=[])
    return row


def retained_entries(session, declaration, edits=None):
    (session / 'analysis').mkdir(exist_ok=True)
    entries = []
    for task in declaration['tasks']:
        row = make_row(task)
        if edits:
            edits(task, row)
        result = {'status': 'completed', 'recorded_frames': 100}
        row['result_identity'] = digest(result)
        path = session / 'analysis' / (task['name'] + '.json')
        path.write_text(json.dumps(row, indent=2), encoding='utf-8')
        entries.append({'task': task, 'result': result, 'status': 'completed',
                        'analysis': row, 'analysis_path': str(path.relative_to(session)),
                        'analysis_sha256': sha(path), 'evidence_sha256': 'raw-inventory-was-verified'})
    return entries


class FullAnalysisContracts(unittest.TestCase):
    def test_frozen_range_uses_calibration_only_and_signed_direction(self):
        rows = [dict(name=f'cal{i}', calibration_index=i, hard_checks_passed=True,
                     material=material(1.1 + i*.001), ledger_analysis_sha256=str(i)) for i in (1, 2, 3)]
        frozen = freeze_bounds(rows);before = copy.deepcopy(frozen)
        check = compare_material(material(1.2), frozen)
        self.assertFalse(check['passed'])
        self.assertAlmostEqual(check['metrics']['max_stretch']['signed_worsening_gap'], .097)
        self.assertEqual(frozen, before)
        lower = material();lower['abd_min_J'] = .9
        self.assertAlmostEqual(compare_material(lower, frozen)['metrics']['abd_min_J']['signed_worsening_gap'], .1)
        # Absolute comparison tolerance is fixed, never derived from candidate noise.
        within = material(1.103 + 5e-13)
        self.assertTrue(compare_material(within, frozen)['passed'])

    def test_paired_bootstrap_and_incomplete_no_ci(self):
        result = paired_statistics([2.] * 7, 7)
        self.assertEqual(result['paired_median'], 2.)
        self.assertEqual(result['one_sided_95_percent_lower'], 2.)
        self.assertTrue(result['statistical_2x_threshold_met'])
        self.assertFalse(result['quality_certified'])
        self.assertIsNone(paired_statistics([4., 4.], 7)['one_sided_95_percent_lower'])
        with self.assertRaises(ValueError): paired_statistics([float('nan')], 7)

    def test_full_plan_real_pair_denominators_and_eligible_stability(self):
        declaration = plan()
        declaration['tasks'] = [t for t in declaration['tasks'] if t['phase'] != 'stability']
        with tempfile.TemporaryDirectory() as folder:
            session = Path(folder);entries = retained_entries(session, declaration)
            report = analyze_session(session, {}, declaration, entries)
            for scene in ('hang', 'fixed', 'mixed'):
                result = report['scenes'][scene]
                self.assertEqual(result['main_original_to_combined']['solver']['paired_median'], 2.)
                self.assertEqual(result['main_original_to_combined']['solver']['pairs_observed'], 7)
                self.assertEqual(result['graph_component']['solver']['paired_median'], 1.25)
                self.assertEqual(result['combined_execution_component']['solver']['paired_median'], 1.2)
                self.assertTrue(report['long_run_gates'][scene]['eligible'])
            self.assertFalse(report['quality_certified'])
            self.assertFalse(report['performance_certified'])
            # No raw run directory was created: retained summaries suffice.
            self.assertFalse((session / declaration['tasks'][0]['name']).exists())

    def test_holdout_cannot_expand_bounds_or_authorize_long(self):
        declaration = plan();declaration['tasks'] = declaration['tasks'][:5]
        def edit(task, row):
            if task['phase'] == 'verification': row['material'] = material(1.2)
        with tempfile.TemporaryDirectory() as folder:
            session = Path(folder);entries = retained_entries(session, declaration, edit)
            report = analyze_session(session, {}, declaration, entries)
            result = report['scenes']['hang']
            self.assertEqual(result['frozen_material_bounds']['bounds']['max_stretch']['max'], 1.1)
            self.assertFalse(result['baseline_verification_passed'])
            self.assertFalse(report['long_run_gates']['hang']['eligible'])

    def test_mutated_ledger_analysis_or_task_is_rejected(self):
        declaration = plan();declaration['tasks'] = declaration['tasks'][:1]
        with tempfile.TemporaryDirectory() as folder:
            session = Path(folder);entries = retained_entries(session, declaration)
            entries[0]['analysis']['material']['max_stretch'] = 99
            with self.assertRaisesRegex(ValueError, 'embedded analysis'): analyze_session(session, {}, declaration, entries)
        with tempfile.TemporaryDirectory() as folder:
            session = Path(folder);entries = retained_entries(session, declaration)
            entries[0]['task'] = dict(entries[0]['task'], pair=5)
            with self.assertRaisesRegex(ValueError, 'planned task prefix'): analyze_session(session, {}, declaration, entries)

    def test_minimal_skipped_report_is_preserved(self):
        declaration = plan();declaration['tasks'] = declaration['tasks'][:1]
        with tempfile.TemporaryDirectory() as folder:
            session = Path(folder);entries = retained_entries(session, declaration)
            e = entries[0]
            e['analysis'] = {'name': e['task']['name'], 'hard_checks_passed': False,
                             'quality_status': 'not_run', 'failures': ['prior resource failure']}
            p = session / e['analysis_path'];p.write_text(json.dumps(e['analysis']), encoding='utf-8')
            e['analysis_sha256'] = sha(p);e['status'] = 'skipped'
            report = analyze_session(session, {}, declaration, entries)
            self.assertEqual(report['runs'][0]['quality_status'], 'not_run')
            self.assertFalse(report['long_run_gates']['hang']['eligible'])

    def test_phase_and_export_cost_missing_not_zero_or_cuda_event(self):
        rows = [{'frame': 1, 'solver_ms': 10}]
        result = timing_summary([{'phase_ms': {'pcg': 2}}], rows,
                                {'solver_seconds': .01, 'wall_seconds': .1})
        self.assertIsNone(result['phase_ms']['assembly'])
        self.assertIsNone(result['load_seconds']);self.assertIsNone(result['export_seconds'])
        self.assertIn('CPU steady_clock', result['solver_time_scope'])
        self.assertAlmostEqual(result['wall_minus_solver_seconds'], .09)

    def test_native_timecost_seconds_and_iteration_count_are_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'timeCost.txt'
            self.assertFalse(read_timecost(path)['available'])
            path.write_text('time0: 1.25\ntime1: 2.5\ntotalTime: 4.125\ntotalCgTime: 19324\nframes: 100\n', encoding='utf-8')
            actual = read_timecost(path)
            self.assertEqual(actual['core_cuda_event_seconds'], 4.125)
            self.assertEqual(actual['cumulative_phase_event_seconds']['pcg'], 2.5)
            self.assertEqual(actual['native_counters']['totalCgTime'], 19324)
            self.assertIsNone(actual['cumulative_phase_event_seconds']['ccd'])
            path.write_text('totalTime: nan\n', encoding='utf-8')
            with self.assertRaises(ValueError): read_timecost(path)

    def test_pair_cache_survives_raw_archival_and_never_infers_velocity(self):
        with tempfile.TemporaryDirectory() as folder:
            session = Path(folder);(session / 'analysis').mkdir()
            names = ['p1_hang_original_stiff', 'p1_hang_combined_graph']
            positions = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
            for name in names:
                run = session / name;(run / 'trace').mkdir(parents=True)
                np.array([3, 1, 0, 0, 1, 2], dtype='<u4').tofile(run / 'trace/topology.bin')
                (run / 'trace/metadata.json').write_text('{"abd_point_num":0}', encoding='utf-8')
                for i in range(2): positions.astype('<f8').tofile(run / f'trace/state_{i:04d}.bin')
                if 'combined' in name:
                    for i in range(2): np.zeros((3, 3), dtype='<f8').tofile(run / f'trace/velocity_{i:04d}.bin')
                files = [{'path': str(p.relative_to(run)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in (run / 'trace').iterdir()]
                (run / 'evidence.json').write_text(json.dumps({'files': files}), encoding='utf-8')
            prior = {'hard_checks_passed': True, 'input_identity': {'initial': 'same', 'initial_actual_velocity': None},
                     'positions': {'expected_frames': 2}, 'actual_velocity': {'available': False}}
            (session / 'analysis' / (names[0] + '.json')).write_text(json.dumps(prior), encoding='utf-8')
            task = {'name': names[1], 'phase': 'paired', 'arm': 'combined_graph', 'pair': 1, 'scene_key': 'hang'}
            current = dict(prior, actual_velocity={'available': True})
            first = previous_comparisons(session, task, current)
            self.assertFalse(first[0]['state_difference']['velocity']['available'])
            self.assertEqual(first[0]['state_difference']['position']['cloth']['max_vertex'], 0)
            for name in names:
                for p in (session / name / 'trace').glob('*.bin'): p.unlink()
            self.assertEqual(previous_comparisons(session, task, current), first)


if __name__ == '__main__':
    unittest.main()
