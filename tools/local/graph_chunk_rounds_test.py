"""Probe evidence and finite continuation contracts (CPU only)."""
import tempfile
from pathlib import Path
import numpy as np
import unittest
import graph_chunk_rounds as g
from cpu_fixed_reference import fnv1a64


def evidence():
    s = {'schema': 'fixed_graph_chunk_v1', 'passed': True, 'restoration_error': '',
         'warmup_pairs': 2, 'measured_pairs': 7, 'runs': []}
    s.update({k: True for k in g.RESTORED})
    uses = {1: 0, 4: 0}
    for warm, count in ((True, 2), (False, 7)):
        for i in range(1, count + 1):
            for order, chunk in enumerate((1, 4) if i % 2 else (4, 1), 1):
                uses[chunk] += 1
                s['runs'].append({'warmup': warm, 'pair': i, 'order': order, 'chunk': chunk,
                                 'arm_use': uses[chunk], 'cache_hit': uses[chunk] != 1,
                                 'passed': True, 'finite': True, 'iterations': 7,
                                 'solve_host_ms': 10 if chunk == 1 else 8,
                                 'solve_event_ms': 9 if chunk == 1 else 7,
                                 'true_relative_residual': .02,
                                 'pcg': {'graph_inactive_tail_steps': 0 if chunk == 1 else 1}})
    return s


class Contracts(unittest.TestCase):
    def test_separate_cold_cache_and_exact_pairs(self):
        r = g.analyze(evidence())
        self.assertEqual(len(r['pairs']), 7)
        self.assertAlmostEqual(r['median_complete_solve_host_speedup'], 1.25)
        self.assertTrue(r['local_performance_gate'])
        self.assertFalse(r['quality_certified'])

    def test_missing_reordered_or_recaptured_pair_fails(self):
        for change in ('missing', 'order', 'cache'):
            s = evidence()
            if change == 'missing': s['runs'].pop()
            elif change == 'order': s['runs'][4]['order'] = 2
            else: s['runs'][4]['cache_hit'] = False
            with self.assertRaises(ValueError): g.analyze(s)

    def test_failed_restore_or_nonfinite_time_fails(self):
        for change in ('restore', 'time', 'limit'):
            s = evidence()
            if change == 'restore': s['workspace_restored'] = False
            elif change == 'time': s['runs'][4]['solve_host_ms'] = float('nan')
            else: s['runs'][4]['pcg']['iteration_limit'] = True
            with self.assertRaises(ValueError): g.analyze(s)

    def test_iteration_change_cannot_claim_execution_gain(self):
        s = evidence()
        s['runs'][5]['iterations'] = 6
        self.assertFalse(g.analyze(s)['local_performance_gate'])

    def test_insufficient_or_negative_gain_stops(self):
        for time in (9, 11):
            s = evidence()
            for r in s['runs']:
                if r['chunk'] == 4: r['solve_host_ms'] = time
            self.assertFalse(g.analyze(s)['local_performance_gate'])

    def test_fixed_task_has_one_terminal_system(self):
        for stage, (scene, frame) in g.STAGES.items():
            c = g.task(stage)['config']
            self.assertEqual(c['steps'], frame)
            self.assertEqual(c['fixed_frames'], str(frame))
            self.assertEqual(c['fixed_directions'], '1')
            self.assertEqual(c['pcg_graph_chunk'], 1)
            self.assertTrue(c['fixed_graph_chunk_study'])
            self.assertEqual(c['diagnostics'], ['fixed'])
            self.assertEqual(c['pcg_rho_tol'], 1e-4)

    def test_changed_or_external_solution_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            raw = np.array([1., 2., 3.], dtype='<f8').tobytes()
            p = out / 'x.bin'
            p.write_bytes(raw)
            s = {'system': {'dofs': 3}, 'runs': [
                {'warmup': False, 'chunk': 1, 'pair': 1, 'solution_file': str(p),
                 'solution_identity': {'bytes': len(raw), 'fnv1a64': fnv1a64(raw)}}]}
            p.write_bytes(raw[:-1] + b'\x01')
            with self.assertRaises(ValueError): g.numerical_comparison(s, out)
            p.write_bytes(raw)
            with self.assertRaises(ValueError): g.numerical_comparison(s, out / 'other')

    def test_numerical_divergence_is_not_hidden_by_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            s = {'system': {'dofs': 3}, 'runs': []}
            for chunk in (1, 4):
                for pair in range(1, 8):
                    raw = np.array([1., 2., 3. if chunk == 1 else 3.1], dtype='<f8').tobytes()
                    p = out / f'{chunk}_{pair}.bin'
                    p.write_bytes(raw)
                    s['runs'].append({'warmup': False, 'chunk': chunk, 'pair': pair,
                                      'solution_file': str(p), 'solution_identity':
                                      {'bytes': len(raw), 'fnv1a64': fnv1a64(raw)}})
            r = g.numerical_comparison(s, out)
            self.assertEqual(r['K1_observed_max_relative_repeat_difference'], 0)
            self.assertFalse(r['within_observed_K1_range_plus_roundoff'])

    def test_resource_recovery_requires_review_and_cannot_retry_algorithm_failure(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            session=Path(directory)
            out=session/'long'/'long_probe'
            out.mkdir(parents=True)
            a=session/'long'/'analysis.json'
            a.write_text(json.dumps({'diagnosis_complete':False}),encoding='utf-8')
            r=out/'result.json'
            r.write_text(json.dumps({'status':'memory_budget'}),encoding='utf-8')
            with self.assertRaises(ValueError):g.review_resource_failure(session,None)
            g.review_resource_failure(session,g.sha(a))
            r.write_text(json.dumps({'status':'failed'}),encoding='utf-8')
            with self.assertRaises(ValueError):g.review_resource_failure(session,g.sha(a))


if __name__ == '__main__':
    unittest.main()
