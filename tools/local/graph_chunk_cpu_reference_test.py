"""Small CPU-only serialization, completeness, reference and owned-worker tests."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cpu_fixed_reference import expand_blocks, fnv1a64
from graph_chunk_cpu_reference import (RESTORATION_KEYS, bounded_worker,
                                       load_study, solve, write_new)


def fixture(directory):
    """Synthetic 6-DOF fixture, not a saved GPU experiment."""
    prefix = Path(directory) / 'fixed' / 'f2_n1'
    prefix.parent.mkdir()
    rows = np.array([0, 0, 1], dtype='<i4')
    cols = np.array([0, 1, 1], dtype='<i4')
    blocks = np.array([[[4., 1., 0.], [1., 5., .2], [0., .2, 6.]],
                       .1*np.eye(3), np.diag([7., 8., 9.])])
    matrix = expand_blocks(rows, cols, blocks, 6)
    exact = np.array([.5, -.25, .75, 1., -1.25, 1.5])
    rhs = matrix @ exact
    meta = {'dofs': 6, 'blocks': 3,
            'matrix_format': 'symmetric upper block COO, int32 indices, float64 column-major 3x3',
            'preconditioner_export_complete': True,
            'local_preconditioners': [{'kind': 'MAS_full_owned_buffers', 'offset': 0}],
            'buffers': {}}
    arrays = {'rows': rows.tobytes(), 'cols': cols.tobytes(),
              'values': blocks.transpose(0, 2, 1).astype('<f8').tobytes(),
              'rhs': rhs.astype('<f8').tobytes()}
    for name, raw in arrays.items():
        Path(str(prefix) + '_' + name + '.bin').write_bytes(raw)
        meta['buffers'][name] = {'bytes': len(raw), 'fnv1a64': fnv1a64(raw)}
    Path(str(prefix) + '_meta.json').write_text(json.dumps(meta), encoding='utf-8')
    study = {'schema': 'fixed_graph_chunk_v1', 'passed': True, 'mas': 'legacy',
             'zero_initial_guess': True, 'rho_tolerance': 1e-4, 'max_iter': 30,
             'system': meta, 'system_after': copy.deepcopy(meta), 'restoration_error': '',
             'warmup_pairs': 2, 'measured_pairs': 7, 'runs': [],
             **{key: True for key in RESTORATION_KEYS}}
    uses = {1: 0, 4: 0}
    for warm, count in ((True, 2), (False, 7)):
        for pair in range(1, count + 1):
            order = (1, 4) if pair % 2 else (4, 1)
            for index, chunk in enumerate(order, 1):
                uses[chunk] += 1
                x = exact + (pair + (chunk == 4)*.5)*1e-3
                raw = x.astype('<f8').tobytes()
                row = {'warmup': warm, 'pair': pair, 'chunk': chunk, 'order': index,
                       'passed': True, 'finite': True, 'cache_contract_passed': True,
                       'iterations': 6, 'pcg': {}, 'arm_use': uses[chunk],
                       'true_relative_residual': float(np.linalg.norm(matrix @ x - rhs)/np.linalg.norm(rhs)),
                       'solution_identity': {'bytes': len(raw), 'fnv1a64': fnv1a64(raw)}}
                if not warm:
                    path = Path(str(prefix) + f'_k{chunk}_pair{pair}_x.bin')
                    path.write_bytes(raw)
                    row['solution_file'] = str(path.resolve())
                study['runs'].append(row)
    save_study(prefix, study)
    return prefix, study


def save_study(prefix, study):
    Path(str(prefix) + '_graph_chunk_study.json').write_text(json.dumps(study), encoding='utf-8')


class ReferenceContracts(unittest.TestCase):
    def test_all14_solutions_reference_pass_is_not_default_pcg_accuracy_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix, _ = fixture(folder)
            result = solve(prefix)
            self.assertTrue(result['passed'])
            self.assertEqual(len(result['saved_solutions']), 14)
            self.assertLess(result['reference']['relative_l2'], 1e-8)
            self.assertTrue(all(not row['residual_at_most_1e_minus_8'] for row in result['saved_solutions']))
            self.assertTrue(all(row['relative_error_to_cpu_reference'] > 0 for row in result['saved_solutions']))
            self.assertFalse(result['physics_certified'])
            self.assertFalse(result['performance_certified'])

    def test_missing_duplicate_or_unrestored_study_rejected(self):
        for change in ('missing', 'duplicate', 'unrestored'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                prefix, study = fixture(folder)
                if change == 'missing': study['runs'].pop()
                if change == 'duplicate': study['runs'][-1] = copy.deepcopy(study['runs'][-2])
                if change == 'unrestored': study['workspace_restored'] = False
                save_study(prefix, study)
                with self.assertRaises(ValueError): load_study(prefix)

    def test_metadata_and_solution_fingerprints_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix, study = fixture(folder)
            study['system_after']['dofs'] = 9
            save_study(prefix, study)
            with self.assertRaises(ValueError): load_study(prefix)
        with tempfile.TemporaryDirectory() as folder:
            prefix, study = fixture(folder)
            path = Path(study['runs'][-1]['solution_file'])
            raw = bytearray(path.read_bytes());raw[-1] ^= 1;path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): solve(prefix)

    def test_solution_path_escape_rejected_and_output_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix, study = fixture(folder)
            study['runs'][-1]['solution_file'] = str(Path(folder) / 'elsewhere.bin')
            save_study(prefix, study)
            with self.assertRaises(ValueError): load_study(prefix)
            output = Path(folder) / 'report.json'
            write_new(output, {'original': True})
            before = output.read_bytes()
            with self.assertRaises(FileExistsError): write_new(output, {'replacement': True})
            self.assertEqual(before, output.read_bytes())

    def test_resource_budget_cannot_expand(self):
        for seconds, memory in ((61, 2048), (60, 2049), (0, 2048)):
            with self.assertRaises(ValueError): bounded_worker(Path('unused'), seconds, memory)

    @unittest.skipUnless(os.name == 'nt', 'Windows owned process test')
    def test_real_bounded_cpu_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix, _ = fixture(folder)
            result = bounded_worker(prefix, seconds=20)
            self.assertEqual(result['status'], 'completed', result)
            self.assertTrue(result['passed'])
            self.assertEqual(len(result['saved_solutions']), 14)
            guard = result['resource_guard']
            self.assertEqual(guard['launch_lifecycle'], ['created_suspended', 'assigned_to_job', 'resumed'])
            self.assertTrue(all(member['exit_code'] is not None for member in guard['members']))

    @unittest.skipUnless(os.name == 'nt', 'Windows owned process test')
    def test_owned_worker_timeout_does_not_claim_partial_success(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix, _ = fixture(folder)
            result = bounded_worker(prefix, seconds=.001)
            self.assertEqual(result['status'], 'timeout')
            self.assertFalse(result['passed'])
            self.assertEqual(result['saved_solutions'], [])
            self.assertIsNotNone(result['resource_guard']['worker_exit_code'])

    @unittest.skipUnless(os.name == 'nt', 'Windows owned process test')
    def test_owned_worker_memory_budget_does_not_claim_partial_success(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix, _ = fixture(folder)
            result = bounded_worker(prefix, seconds=10, memory_mib=1)
            self.assertEqual(result['status'], 'memory_budget')
            self.assertFalse(result['passed'])
            self.assertEqual(result['saved_solutions'], [])
            self.assertIsNotNone(result['resource_guard']['worker_exit_code'])


if __name__ == '__main__':
    unittest.main()
