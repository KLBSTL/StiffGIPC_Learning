"""CPU-only planning/gating contract tests; no CUDA, subprocess, or archive creation."""
import collections
import ast
import json
from pathlib import Path
import tempfile
import unittest

from config import digest, expand, sha
from autodl_prepare import make_plan, portable_tools, ROOT as TASK_ROOT
import autodl_linux


class AutoDLContracts(unittest.TestCase):
    def setUp(self):
        self.candidate = {'preset': 'toi', 'mas_restrict': 'warp'}

    def test_finite_stages(self):
        for stage, count in [('smoke', 15), ('window', 45), ('pilot', 21), ('audit', 6), ('paired', 105)]:
            plan = make_plan(self.candidate, stage)
            self.assertEqual(len(plan['runs']), count)
            self.assertEqual(len({r['name'] for r in plan['runs']}), count)
            self.assertTrue(all(1 <= r['config']['timeout_seconds'] <= 120 for r in plan['runs']))
            self.assertTrue(all(r['config']['dt'] == .01 for r in plan['runs']))
            self.assertFalse(plan['performance_certified'])

    def test_one_variable_four_way(self):
        plan = make_plan(self.candidate, 'pilot')
        rows = {r['variant']: r for r in plan['runs'] if r['scene_key'] == 'hang' and r['repeat'] == 1}
        for left, right, expected in [('ipc_host', 'toi_host', {'backend'}),
                                      ('ipc_host', 'ipc_graph', {'execution'}),
                                      ('toi_host', 'toi_graph', {'execution'})]:
            a, b = rows[left]['config'], rows[right]['config']
            self.assertEqual({k for k in a if a[k] != b[k]}, expected)
        self.assertEqual(rows['stiff']['binary'], 'base')
        self.assertEqual(rows['stiff']['config']['mas'], 'legacy')

    def test_diagnostics_and_baseline_repeats(self):
        pilot = make_plan(self.candidate, 'pilot')['runs']
        self.assertEqual(collections.Counter(r['scene_key'] for r in pilot if r['variant'] == 'stiff'),
                         {'mixed': 3, 'hang': 3, 'fixed_bunny': 3})
        for stage in ('smoke', 'pilot', 'paired'):
            self.assertTrue(all(not r['config']['diagnostics'] and not r['config']['trace_velocity']
                                for r in make_plan(self.candidate, stage)['runs']))
        self.assertTrue(all(r['config']['diagnostics'] == ['substeps', 'physics']
                            and r['config']['trace_velocity'] for r in make_plan(self.candidate, 'audit')['runs']))

    def test_rotation_and_stopping_parameters(self):
        candidate = self.candidate | {'ipc_newton_tol': .012, 'pcg_rho_tol': 2e-4}
        plan = make_plan(candidate, 'paired')
        first = [r['variant'] for r in plan['runs'] if r['scene_key'] == 'mixed' and r['repeat'] == 1]
        second = [r['variant'] for r in plan['runs'] if r['scene_key'] == 'mixed' and r['repeat'] == 2]
        self.assertEqual(first[1:] + first[:1], second)
        self.assertTrue(all(r['config']['ipc_newton_tol'] == .012 and r['config']['pcg_rho_tol'] == 2e-4
                            for r in plan['runs']))

    def test_gate_requires_identity_and_evidence(self):
        original = autodl_linux.ROOT
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); autodl_linux.ROOT = root
            try:
                evidence = root / 'evidence.json'; evidence.write_text('{}')
                plan = make_plan(self.candidate, 'paired')
                manifest = {'source_digest': 'source'}
                record = {'quality_passed': True, 'baseline_repeat_range_frozen': True, 'load_controlled': True,
                          'reviewed_stage': 'pilot_audit',
                          'evidence': [{'path': 'evidence.json', 'sha256': sha(evidence)}]}
                gate = {'source_digest': 'source', 'candidate_sha256': plan['candidate_sha256'],
                        'scenes': {'hang': record, 'mixed': record | {'quality_passed': False}}}
                path = root / 'gate.json'; path.write_text(json.dumps(gate))
                self.assertEqual(autodl_linux.accepted_gate(path, plan, manifest), {'hang'})
                evidence.write_text('{"changed":true}')
                with self.assertRaises(ValueError):
                    autodl_linux.accepted_gate(path, plan, manifest)
                gate['source_digest'] = 'other'; path.write_text(json.dumps(gate))
                with self.assertRaises(ValueError):
                    autodl_linux.accepted_gate(path, plan, manifest)
            finally:
                autodl_linux.ROOT = original

    def test_explicit_factor_candidate_reference(self):
        candidate = self.candidate | {'mas_factor_action': 'factor_inverse'}
        plan = make_plan(candidate, 'pilot', triangular_reference=True)
        self.assertEqual(len(plan['runs']), 24)
        for row in plan['runs']:
            expected = 'triangular' if row['variant'] in ('stiff', 'toi_graph_triangular') else 'factor_inverse'
            self.assertEqual(row['config']['mas_factor_action'], expected)
            self.assertFalse(row['config']['fixed_factor_study'])
            if row['variant'] == 'toi_graph_triangular':
                self.assertEqual(row['config']['execution'], 'conditional_graph')
                self.assertEqual(row['config']['mas_restrict'], 'warp')
        with self.assertRaises(ValueError):
            make_plan(self.candidate, 'pilot', triangular_reference=True)

    def test_windows_before_full_runs(self):
        plan = make_plan(self.candidate, 'window')
        self.assertEqual(len(plan['runs']), 45)
        for row in plan['runs']:
            self.assertEqual(row['config']['steps'], {'mixed': 35, 'hang': 21, 'fixed_bunny': 45}[row['scene_key']])
            self.assertEqual(row['config']['diagnostics'], ['substeps', 'physics'])
            self.assertTrue(row['config']['trace_velocity'])
        self.assertTrue(all(count == 3 for count in collections.Counter(row['arm'] for row in plan['runs']).values()))
        for stage in ('smoke', 'window'):
            self.assertEqual(autodl_linux.gated_scenes(stage, None, make_plan(self.candidate, stage), {}),
                             {'mixed', 'hang', 'fixed_bunny'})
        for stage in ('pilot', 'audit', 'paired'):
            with self.assertRaises(ValueError):
                autodl_linux.gated_scenes(stage, None, make_plan(self.candidate, stage), {})

    def test_window_quality_gate_does_not_authorize_paired(self):
        original = autodl_linux.ROOT
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); autodl_linux.ROOT = root
            try:
                evidence = root / 'review.json'; evidence.write_text('{"reviewed":true}')
                pilot = make_plan(self.candidate, 'pilot')
                record = {'reviewed_stage': 'window', 'quality_passed': True,
                          'baseline_repeat_range_frozen': True, 'load_controlled': False,
                          'evidence': [{'path': 'review.json', 'sha256': sha(evidence)}]}
                gate = {'source_digest': 'source', 'candidate_sha256': pilot['candidate_sha256'],
                        'scenes': {'hang': record, 'mixed': record | {'quality_passed': False}}}
                path = root / 'gate.json'; path.write_text(json.dumps(gate))
                manifest = {'source_digest': 'source'}
                for stage in ('pilot', 'audit'):
                    self.assertEqual(autodl_linux.gated_scenes(stage, path, make_plan(self.candidate, stage), manifest), {'hang'})
                with self.assertRaises(ValueError):
                    autodl_linux.gated_scenes('paired', path, make_plan(self.candidate, 'paired'), manifest)
                record.update(reviewed_stage='pilot_audit', load_controlled=True)
                path.write_text(json.dumps(gate))
                self.assertEqual(autodl_linux.gated_scenes('paired', path, make_plan(self.candidate, 'paired'), manifest), {'hang'})
            finally:
                autodl_linux.ROOT = original

    def test_portable_tool_import_closure(self):
        included = {path.resolve() for path in portable_tools()}
        self.assertTrue(all(path.is_file() for path in included))
        for path in included:
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                names = ([node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module else
                         [alias.name.split('.')[0] for alias in node.names] if isinstance(node, ast.Import) else [])
                for name in names:
                    for base in (TASK_ROOT / 'tools/active', TASK_ROOT / 'tools'):
                        dependency = base / (name + '.py')
                        if dependency.is_file():
                            self.assertIn(dependency.resolve(), included, (path.name, name))
        for name in ('verify_systems.py', 'fixtures.py', 'test_contracts.py', 'build.py', 'run.py',
                     'audit_cloth_restrict.py', 'analyze_cloth_restrict.py'):
            self.assertNotIn((TASK_ROOT / 'tools/active' / name).resolve(), included)

    def test_linux_layout_contract(self):
        active = (TASK_ROOT / 'sources/stiff_active/CMakeLists.txt').read_text()
        self.assertIn('RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"', active)
        self.assertIn('"${CMAKE_CURRENT_BINARY_DIR}/inherited-v50"', active)
        for tree in ('stiff_base', 'stiff_perf_v50'):
            base = TASK_ROOT / 'sources' / tree
            cmake = (base / 'CMakeLists.txt').read_text()
            self.assertIn('GIPC_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets/"', cmake)
            self.assertTrue((base / 'Assets/scene/parameterSetting.txt').is_file())
        validator = (TASK_ROOT / 'tools/validator/CMakeLists.txt').read_text()
        self.assertIn('add_executable(validate_path ', validator)
        self.assertIn('add_executable(diagnose_first_path ', validator)
        self.assertIn('set(TI ../../references/tight_inclusion)', validator)
        self.assertTrue((TASK_ROOT / 'references/tight_inclusion/tight_inclusion/inclusion_ccd.cpp').is_file())


if __name__ == '__main__':
    unittest.main()
