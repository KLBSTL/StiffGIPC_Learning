"""Synthetic evidence only. Native CCD, configuration and provenance are not simulated as production passes."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import reaggregate_autodl_audit as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture_record(root, directory, validator, index=0, extra_failure=None):
    name = f'autodl_window_synthetic_{index:02d}'
    run = root / 'runs/active' / name; trace = run / 'trace'
    (trace / 'substeps').mkdir(parents=True)
    (run / 'output').mkdir()
    np.array([3, 1, 0, 0, 1, 2], dtype='<u4').tofile(trace / 'topology.bin')
    np.zeros(3, dtype='<i4').tofile(trace / 'boundary_types.bin')
    np.zeros(3, dtype='<i4').tofile(trace / 'body_ids.bin')
    np.ones(3, dtype='<f8').tofile(trace / 'masses.bin')
    write(trace / 'metadata.json', {'abd_point_num': 0, 'ground_normal': [0, 1, 0], 'ground_offset': -1})
    write(run / 'output/scene.json', {'objects': [{'dimension': 2, 'body_type': 'FEM'}]})
    x0 = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    x1 = x0.copy(); x1[:, 1] += .01
    x0.tofile(trace / 'state_0000.bin'); x1.tofile(trace / 'state_0001.bin')
    x0.tofile(trace / 'substeps/safe_0000_0000.bin'); x1.tofile(trace / 'substeps/safe_0000_0001.bin')
    inventory = module.original_wrapper.inspect_trace(run, 1)
    identity_path = directory / (name + '.identity.json'); write(identity_path, inventory)
    stdout = directory / (name + '.stdout.log'); stdout.write_text('synthetic native stdout\n')
    ccd_path = directory / (name + '.ccd.json')
    ccd = {'scope': 'accepted substeps', 'paths_checked': 1, 'stopped_after_first_collision': False,
           'passed': True, 'finite': True, 'conservative_collision_flags': 0, 'abd_tet_inversions': 0}
    write(ccd_path, ccd)
    task = {'name': name, 'scene_key': 'synthetic', 'role': 'ipc_host', 'repeat': 1,
            'expanded_config': {'steps': 1}}
    record = {'run': name, 'scene_key': 'synthetic', 'role': 'ipc_host', 'repeat': 1,
              'configuration_passed': False, 'accepted_path_audit_passed': False,
              'failures': [module.CONFIG_PREFIX + 'Unconfirmed MAS action/restriction: ' + name],
              'stdout': str(stdout), 'stdout_sha256': module.sha(stdout),
              'identity_json': str(identity_path), 'identity_sha256': module.sha(identity_path),
              'native_ccd_json': str(ccd_path), 'native_ccd_sha256': module.sha(ccd_path), 'ccd': ccd,
              'validator_invoked': True, 'validator_exit_code': 0, 'validator_coverage_passed': True,
              'ccd_observation_passed': True, 'same_scene_initial_identity': True,
              'command': [str(validator), str(trace), str(ccd_path), 'substeps', '--stable-nh1']}
    record.update({key: inventory[key] for key in module.INVENTORY_FIELDS})
    if extra_failure: record['failures'].append(extra_failure)
    write(directory / (name + '.record.json'), record)
    return task, record


class ReaggregateAudit(unittest.TestCase):
    def test_configuration_repair_preserves_unrelated_failure_and_no_native_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); directory = root / 'audit'; directory.mkdir()
            validator = root / 'validator'; validator.write_bytes(b'no execution')
            task, old = fixture_record(root, directory, validator, extra_failure='Existing nonconfiguration observation')
            with patch.object(module.analyzer, 'validate_run', return_value={'synthetic_check': True}), \
                 patch.object(module.original_wrapper.subprocess, 'run', side_effect=AssertionError('Must not invoke native code')):
                new = module.recheck_record(root, directory, old, task, {}, {}, {}, {}, validator, 60, {})
            self.assertTrue(new['configuration_passed']); self.assertTrue(new['reaggregation_evidence_verified'])
            self.assertFalse(new['accepted_path_audit_passed'])
            self.assertEqual(new['failures'], ['Existing nonconfiguration observation'])
            self.assertEqual(new['original_failures'], old['failures'])
            self.assertEqual(len(new['cleared_configuration_failures']), 1)
            with patch.object(module.analyzer, 'validate_run', side_effect=ValueError('Different real configuration problem')):
                bad = module.recheck_record(root, directory, old, task, {}, {}, {}, {}, validator, 60, {})
            self.assertFalse(bad['configuration_passed']); self.assertEqual(bad['cleared_configuration_failures'], [])
            self.assertTrue(any('Different real configuration problem' in value for value in bad['failures']))

    def test_native_json_stdout_or_retained_trace_tamper_cannot_be_reused(self):
        for kind in ('ccd', 'stdout', 'trace'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve(); directory = root / 'audit'; directory.mkdir()
                validator = root / 'validator'; validator.write_bytes(b'no execution')
                task, old = fixture_record(root, directory, validator)
                if kind == 'ccd': write(directory / (task['name'] + '.ccd.json'), old['ccd'] | {'passed': False})
                elif kind == 'stdout': (directory / (task['name'] + '.stdout.log')).write_text('changed')
                else:
                    path = root / 'runs/active' / task['name'] / 'trace/substeps/safe_0000_0001.bin'
                    positions = np.fromfile(path, dtype='<f8'); positions[0] += .2; positions.tofile(path)
                with patch.object(module.analyzer, 'validate_run', return_value={'synthetic_check': True}) as check:
                    new = module.recheck_record(root, directory, old, task, {}, {}, {}, {}, validator, 60, {})
                self.assertFalse(new['reaggregation_evidence_verified'])
                self.assertFalse(new['accepted_path_audit_passed']); self.assertEqual(check.call_count, 0)
                self.assertEqual(module.summarize([new])['actual_validator_reports'], 0)

    def test_full_54_record_reaggregation_is_new_output_and_binds_original_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); directory = root / 'reports/active/AUTODL_FACTOR_CCD'; directory.mkdir(parents=True)
            tools = root / 'tools/active'; tools.mkdir(parents=True)
            shutil.copyfile(module.original_wrapper.__file__, tools / 'audit_autodl_factor.py')
            shutil.copyfile(module.analyzer.__file__, tools / 'analyze_autodl_factor_v2.py')
            source = root / 'tools/validator/diagnose_first_path.cpp'; source.parent.mkdir(); source.write_text('synthetic source')
            validator = root / 'builds/validator'; validator.parent.mkdir(); validator.write_bytes(b'never executed')
            tasks, records = [], []
            for index in range(54):
                task, record = fixture_record(root, directory, validator, index)
                tasks.append(task); records.append(record)
            plan_path = root / 'configs/active/autodl_window.json'; write(plan_path, {'synthetic': True})
            matrix_path = root / 'reports/active/AUTODL_WINDOW_BATCH.json'; write(matrix_path, {'runs': [{'name': t['name']} for t in tasks]})
            bundle_path = root / 'autodl_bundle.json'; write(bundle_path, {'synthetic': True})
            original = {'schema_version': 1, 'planned_runs': 54, 'recorded_runs': 54, 'runs': records,
                        'download_root': str(root), 'wrapper_sha256': module.sha(tools / 'audit_autodl_factor.py'),
                        'validator': str(validator), 'validator_sha256': module.sha(validator),
                        'validator_source_sha256': module.sha(source), 'plan_sha256': module.sha(plan_path),
                        'matrix_sha256': module.sha(matrix_path), 'bundle_sha256': module.sha(bundle_path),
                        'all_54_accepted_path_audits_passed': False, 'actual_validator_reports': 54,
                        'actual_validator_paths_checked': 54, 'actual_collision_flags_from_available_reports': 0,
                        'collision_flag_total_covers_all_54_full_windows': True}
            write(directory / 'audit.json', original); before = module.sha(directory / 'audit.json')
            output = root / 'reports/active/AUTODL_FACTOR_CCD_REAGGREGATED.json'
            with patch.object(module.analyzer, 'prepare_tasks', return_value=tasks), \
                 patch.object(module.analyzer, 'verify_download_provenance', return_value={'passed': True}), \
                 patch.object(module.analyzer, 'validate_run', return_value={'synthetic_check': True}), \
                 patch.object(module.original_wrapper.subprocess, 'run', side_effect=AssertionError('No native execution')), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = module.reaggregate(root, directory, output)
            self.assertTrue(result['all_54_accepted_path_audits_passed'])
            self.assertEqual(result['reanalysis']['cleared_configuration_failure_count'], 54)
            self.assertFalse(result['physical_certified']); self.assertFalse(result['performance_certified'])
            self.assertFalse(result['reanalysis']['validator_invoked_this_reanalysis'])
            self.assertEqual(module.sha(directory / 'audit.json'), before)
            with self.assertRaises(ValueError): module.reaggregate(root, directory, output)
            validator.write_bytes(b'changed executable')
            with self.assertRaisesRegex(ValueError, 'validator bytes changed'):
                module.reaggregate(root, directory, root / 'reports/active/second.json')


if __name__ == '__main__':
    unittest.main()
