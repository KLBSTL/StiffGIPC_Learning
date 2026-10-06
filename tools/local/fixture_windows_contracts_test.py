"""CPU-only launcher contracts: every GPU/subprocess interaction is simulated."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock
import fixture_windows as f


def gpu(free=4096):
    return {'uuid': 'fake-gpu', 'memory.free': free, 'utilization.gpu': 0, 'driver_model.current': 'WDDM'}


def empty_processes(*args, **kwargs):
    return f.process_rows('', None, 'WDDM')


class Contracts(unittest.TestCase):
    @unittest.skipUnless(f.os.name == 'nt', 'Requires actual Windows byte locking')
    def test_fixture_and_round_share_actual_gpu_lock(self):
        import windows_runner as rounds
        with tempfile.TemporaryDirectory() as folder, patch.object(f, 'ROOT', Path(folder)):
            with rounds.gpu_lock(Path(folder)):
                with self.assertRaises(OSError):
                    with f.gpu_lock():
                        self.fail('Fixture acquired a lock while a round held it')
            with f.gpu_lock():
                with self.assertRaises(OSError):
                    with rounds.gpu_lock(Path(folder)):
                        self.fail('Round acquired a lock while a fixture held it')
            # Both wrappers must release the same byte after leaving the context.
            with rounds.gpu_lock(Path(folder)):
                pass
            with f.gpu_lock():
                pass

    def test_clean_environment_and_native_contracts(self):
        env, path = f.fixture_environment('pcg_chunk', Path('new'))
        self.assertEqual(env['GIPC_PCG_CHUNK_GUARD_FIXTURE'], str(Path('new/fixture.json')))
        self.assertEqual(path, Path('new/fixture.json'))
        with patch.dict(f.os.environ, {'GIPC_STEPS': '999', 'gipc_retired': '1', 'SAFE': 'yes'}, clear=True):
            env, path = f.fixture_environment('contact_pool', Path('new'))
        self.assertEqual({k for k in env if k.upper().startswith('GIPC_')}, {'GIPC_CONTACT_POOL_FIXTURE'})
        self.assertEqual(env['SAFE'], 'yes')
        self.assertEqual(path, Path('new/fixture.json'))
        with self.assertRaises(ValueError):
            f.fixture_environment('fixed', Path('new'))
        with self.assertRaises(ValueError):
            f.fixture_environment('component', Path('new'), Path('wrong'))
        env, _ = f.fixture_environment('fixed', Path('new'), scene='bunny_cloth_bunny_l', mas='cholesky')
        for key, expected in {'GIPC_STEPS': '2', 'GIPC_FIXED_STUDY_FRAMES': '2', 'GIPC_IPC_MIN_UPDATES': '6',
                              'GIPC_IPC_TERMINATION': 'legacy', 'GIPC_MAS_FACTOR_ACTION': 'factor_inverse'}.items():
            self.assertEqual(env[key], expected)

    def test_memory_and_disk_gates_stop_only_owned(self):
        for free, disk, wanted in ((700, 8*1024**3, 'memory_budget'),
                                    (2000, 8*1024**3, 'memory_budget'), (4096, 1, 'disk_reserve')):
            child = Mock(returncode=None)
            child.poll.return_value = None
            with patch.object(f.time, 'monotonic', return_value=0), patch.object(f, 'gpu_query', return_value=gpu(free)), \
                 patch.object(f, 'gpu_processes', side_effect=empty_processes), \
                 patch.object(f.shutil, 'disk_usage', return_value=SimpleNamespace(free=disk)):
                self.assertEqual(f.monitor(child, Path('.'), 0, gpu(), 1500, 0, []), wanted)
            f.stop_owned(child)
            child.terminate.assert_called_once()
            child.wait.assert_called_once_with(timeout=2)
            child.kill.assert_not_called()

    def test_deadline_does_not_start_another_gpu_query(self):
        child = Mock()
        child.poll.return_value = None
        with patch.object(f.time, 'monotonic', return_value=120), patch.object(f, 'gpu_query') as query:
            self.assertEqual(f.monitor(child, Path('.'), 0, gpu(), 2000, 0, []), 'timeout')
            query.assert_not_called()

    def test_only_owned_kill_after_terminate_timeout(self):
        child = Mock()
        child.poll.return_value = None
        child.wait.side_effect = [f.subprocess.TimeoutExpired('fake', 2), 0]
        f.stop_owned(child)
        child.terminate.assert_called_once()
        child.kill.assert_called_once()

    def test_fixed_requires_all_actual_runs_not_empty_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            runs = [{'mode': mode, 'rho_tolerance': tol, 'repeat': repeat, 'error': '', 'limit': False,
                     'true_relative_residual': .0002} for mode in ('host', 'graph') for tol in (1e-4, 1e-16) for repeat in (1, 2)]
            value = {'system_unchanged': True, 'primary_restored_bitwise': True, 'runs': runs}
            path.write_text(json.dumps(value))
            result = f.report_evidence('fixed', path)
            self.assertTrue(result['native_passed'])
            self.assertFalse(result['numerical_acceptance_certified'])
            runs[-1]['error'] = 'breakdown'
            path.write_text(json.dumps(value))
            self.assertFalse(f.report_evidence('fixed', path)['native_passed'])
            value['runs'] = []
            path.write_text(json.dumps(value))
            self.assertFalse(f.report_evidence('fixed', path)['native_passed'])
            path.write_text('{"pass": true}')
            self.assertTrue(f.report_evidence('contact_pool', path)['native_passed'])
            self.assertFalse(f.report_evidence('component', path)['native_passed'])

    def test_execute_mock_subprocess_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as folder:
            exe = Path(folder) / 'fake.exe'
            exe.write_bytes(b'not executable; mock only')
            out = Path(folder) / 'fresh'
            args = SimpleNamespace(exe=exe, output=out, kind='pcg_guard', prefix=None, scene=None,
                                   mas='legacy', restrict='serial', factor_action=None, gpu=0,
                                   diagnostic_under_desktop_load=True)
            ident = {'sources': [], 'exe': {'sha256': 'fake'}, 'dlls': [], 'inputs': []}
            child = Mock(pid=123, returncode=0)
            child.poll.return_value = 0
            child.wait.return_value = 0
            def start(*unused, **kw):
                self.assertEqual(kw['creationflags'], f.NO_WINDOW)
                self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'], 'fake-gpu')
                (out / 'fixture.json').write_text('{"passed":true}')
                return child
            with patch.object(f, 'identity', return_value=ident), patch.object(f, 'gpu_query', return_value=gpu() | {'utilization.gpu': 39}), \
                 patch.object(f, 'gpu_processes', side_effect=empty_processes), \
                 patch.object(f.time, 'sleep'), patch.object(f.shutil, 'disk_usage', return_value=SimpleNamespace(free=8*1024**3)), \
                 patch.object(f.subprocess, 'Popen', side_effect=start) as popen:
                result = f.execute(args)
                self.assertEqual(result['status'], 'completed')
                self.assertTrue(result['identity_unchanged'])
                self.assertFalse(result['load_controlled'])
                self.assertTrue(result['diagnostic_only'])
                self.assertTrue(result['desktop_load_gate_override'])
                self.assertTrue((out / 'requested.json').is_file())
                self.assertTrue((out / 'result.json').is_file())
                with self.assertRaises(ValueError):
                    f.execute(args)
                self.assertEqual(popen.call_count, 1)

    def test_low_initial_memory_never_starts_process(self):
        with tempfile.TemporaryDirectory() as folder:
            exe = Path(folder) / 'fake.exe'
            exe.write_bytes(b'fake')
            args = SimpleNamespace(exe=exe, output=Path(folder)/'fresh', kind='component', prefix=None,
                                   scene=None, mas='legacy', restrict='serial', factor_action=None, gpu=0)
            with patch.object(f, 'identity', return_value={'sources': [], 'exe': {'sha256': 'fake'}}), \
                 patch.object(f, 'gpu_query', return_value=gpu(2000)), patch.object(f.time, 'sleep'), \
                 patch.object(f, 'gpu_processes', side_effect=empty_processes), \
                 patch.object(f.subprocess, 'Popen') as popen:
                result = f.execute(args)
                self.assertEqual(result['status'], 'launcher_or_monitor_failed')
                self.assertEqual(result['gpu_memory_budget_mib'], 464)
                popen.assert_not_called()

    def test_high_load_requires_flag_and_two_wddm_samples(self):
        before = [gpu() | {'utilization.gpu': 39}, gpu() | {'utilization.gpu': 42}]
        with self.assertRaises(ValueError):
            f.initial_load_gate(before, False)
        result = f.initial_load_gate(before, True)
        self.assertFalse(result['load_controlled'])
        self.assertTrue(result['diagnostic_only'])
        self.assertTrue(result['desktop_load_gate_override'])
        before[1]['driver_model.current'] = 'TCC'
        with self.assertRaises(ValueError):
            f.initial_load_gate(before, True)
        before[0]['driver_model.current'] = 'TCC'
        with self.assertRaises(ValueError):
            f.initial_load_gate(before, True)

    def test_desktop_unknown_is_not_invented_compute_but_solver_still_blocks(self):
        raw = ('100, C:/Windows/explorer.exe, N/A\n101, C:/app/gui.exe, [Insufficient Permissions]\n'
               '200, C:/repo/gipc.exe, N/A\n201, C:/ml/python.exe, 512\n300, C:/repo/gipc.exe, 1024\n')
        parsed = f.process_rows(raw, 300, 'WDDM')
        self.assertEqual(parsed['blocking_foreign_pids'], [200, 201])
        self.assertTrue(parsed['unknown_load'])
        self.assertIsNotNone(parsed['capability_gap'])
        self.assertEqual(f.process_rows(raw, 300, 'TCC')['blocking_foreign_pids'], [100, 101, 200, 201])
        with self.assertRaises(ValueError):
            f.process_rows('1, app.exe, nan\n', None, 'WDDM')

    def test_known_foreign_during_run_stops_guard(self):
        child = Mock(pid=300, returncode=None)
        child.poll.return_value = None
        with patch.object(f.time, 'monotonic', return_value=0), patch.object(f, 'gpu_query', return_value=gpu()), \
             patch.object(f, 'gpu_processes', return_value=f.process_rows('200, gipc.exe, N/A', 300, 'WDDM')), \
             patch.object(f.shutil, 'disk_usage', return_value=SimpleNamespace(free=8*1024**3)):
            self.assertEqual(f.monitor(child, Path('.'), 0, gpu(), 2000, 0, []), 'foreign_gpu_load')


if __name__ == '__main__':
    unittest.main()
