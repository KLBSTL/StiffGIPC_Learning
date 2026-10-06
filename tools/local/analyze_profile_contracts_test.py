"""Small synthetic CPU-only contracts for the selected-frame profile analyzer."""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import analyze_profile as profile


MS = 1_000_000


def make_fixture(root, include_frame=True):
    database = root / 'trace.sqlite'
    cost = root / 'cost.jsonl'
    connection = sqlite3.connect(database)
    connection.executescript('''
        CREATE TABLE StringIds(id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, demangledName INTEGER,
            graphNodeId INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, name INTEGER,
            globalTid INTEGER, correlationId INTEGER);
    ''')
    connection.executemany('INSERT INTO StringIds VALUES (?,?)', [
        (1, 'void cholesky_action(double*, double*)'),
        (2, 'void warp_reduce_sym_spmv_kernel<double>(double*)'),
        (3, 'void future_unclassified_kernel(double*)'),
        (4, 'cudaStreamSynchronize'), (5, 'cudaGraphLaunch'),
        (6, 'ipc.physical_frame'),
    ])
    ranges = [(10, 90, 'linear.total', None, 100), (20, 60, 'mas.apply', None, 100)]
    if include_frame:
        # Exact and nested duplicate frame ranges must not inflate any GPU or CPU total.
        ranges.extend([(0, 100, None, 6, 100), (0, 100, 'ipc.physical_frame', None, 100),
                       (5, 95, 'ipc.physical_frame', None, 100)])
    connection.executemany('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)',
                           [(a*MS, b*MS, c, d, e) for a,b,c,d,e in ranges])
    connection.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)', [
        (20*MS, 40*MS, 1, 1, 111), (30*MS, 50*MS, 2, 2, 112),
        (70*MS, 80*MS, 3, 3, 113), (110*MS, 120*MS, 3, 4, 114),
    ])
    # Only one graph node has a unique replay-launch correlation. Even that correlation
    # does not identify the internal graph operator's captured NVTX cost scope.
    connection.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)', [
        (30*MS, 60*MS, 4, 100, 900), (21*MS, 22*MS, 5, 100, 111),
    ])
    connection.commit()
    connection.close()
    row = {'stage': 'ipc.physical_frame', 'frame': 57, 'parent_scope_id': 0,
           'gpu_events_enabled': False, 'gpu_interval_ms': None, 'cpu_submit_ms': 99.9,
           'measurement_mode': 'nvtx_cpu_only', 'nvtx_available': True, 'sample_kind': 'production'}
    cost.write_text(json.dumps(row)+'\n', encoding='utf-8')
    return database, cost


class ProfileContracts(unittest.TestCase):
    def test_nested_windows_and_union(self):
        self.assertEqual(profile.clip(20, 80, [(0, 100), (10, 90), (30, 40)]), [(20, 80)])
        self.assertEqual(profile.merge([(20, 40), (30, 50), (70, 80)]), [(20, 50), (70, 80)])
        self.assertEqual(profile.intersection([(0, 100), (10, 90)], [(20, 40), (30, 50)]), [(20, 50)])
        accounting = profile.timing([(20*MS, 40*MS), (30*MS, 50*MS), (70*MS, 80*MS)])
        self.assertEqual(accounting['sum_ms'], 50)
        self.assertEqual(accounting['union_ms'], 40)

    def test_nested_nvtx_cpu_wait_and_gpu_accounting_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            database, cost = make_fixture(Path(temporary))
            original = [profile.file_hash(p) for p in (database, cost)]
            result = profile.analyze(database, cost)
            self.assertEqual(original, [profile.file_hash(p) for p in (database, cost)])
        self.assertEqual(result['raw_physical_nvtx_ranges'], 3)
        self.assertEqual(result['canonical_physical_nvtx_ranges'], 1)
        self.assertEqual(result['cpu']['physical_frame_nvtx_wall_ms'], 100)
        self.assertEqual(result['cpu']['explicit_wait_api_union_ms'], 30)
        self.assertEqual(result['gpu']['all_gpu_activity']['sum_ms'], 50)
        self.assertEqual(result['gpu']['all_gpu_activity']['union_ms'], 40)
        self.assertEqual(result['gpu']['cross_category_union_overlap_ms'], 10)
        self.assertEqual(result['activity_records_outside_selected_frame'], 1)
        frame_stage = next(x for x in result['cpu']['nvtx_envelopes'] if x['stage']=='ipc.physical_frame')
        self.assertEqual(frame_stage['cpu_union_ms'], 100)
        self.assertFalse(result['performance_certified'])

    def test_unknown_symbols_and_graph_internal_unattributed_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = profile.analyze(*make_fixture(Path(temporary)))
        unknown = next(x for x in result['kernel_symbols'] if x['category']=='unknown_kernel')
        self.assertIn('future_unclassified_kernel', unknown['symbol'])
        self.assertEqual(unknown['union_ms'], 10)
        self.assertEqual(result['unknown_kernel']['union_ms'], 10)
        self.assertEqual(result['graph']['node_without_unique_runtime']['union_ms'], 30)
        self.assertEqual(result['graph']['node_without_runtime_launch_owner']['union_ms'], 30)
        self.assertEqual(result['graph']['node_internal_nvtx_operator_unattributed']['union_ms'], 40)
        self.assertEqual(result['graph']['node_unknown_kernel_symbol']['union_ms'], 10)
        self.assertEqual(result['graph']['node_span_uncovered_by_captured_gpu_ms'], 20)

    def test_missing_nvtx_frame_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'completed ipc.physical_frame NVTX range'):
                profile.analyze(*make_fixture(Path(temporary), include_frame=False))

    def test_mixed_cost_frames_and_non_native_measurements_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            database, cost = make_fixture(Path(temporary))
            row = json.loads(cost.read_text(encoding='utf-8'))
            cost.write_text(json.dumps(row)+'\n'+json.dumps(row|{'frame': 58})+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'only the selected physical frame'):
                profile.analyze(database, cost)
            cost.write_text(json.dumps(row|{'gpu_events_enabled': True})+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'CPU/NVTX-only'):
                profile.analyze(database, cost)

    def test_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database, cost = make_fixture(root)
            output = root / 'analysis.json'
            args = ['analyze_profile.py', '--sqlite', str(database), '--cost-jsonl', str(cost), '--output', str(output)]
            with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                profile.main()
            previous = output.read_bytes()
            with patch('sys.argv', args), self.assertRaisesRegex(ValueError, 'Output already exists'):
                profile.main()
            self.assertEqual(output.read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()
