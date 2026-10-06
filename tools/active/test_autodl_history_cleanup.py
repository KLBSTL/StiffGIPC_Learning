"""Bounded cleanup safety checks; apply mutations are mocked, never executed."""
import json
import gzip
import os
from contextlib import ExitStack
from pathlib import Path
import stat
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

import autodl_history_cleanup as cleanup


class CleanupTests(unittest.TestCase):
    def fixture(self):
        base = Path(__file__).resolve().parents[2] / 'runs' / ('history_cleanup_selftest_' + uuid.uuid4().hex)
        root = base / 'old'; root.mkdir(parents=True)
        for name, contents in {'reports/summary.md': 'result', 'runs/run/stats.json': '{}',
                               'sources/readme.md': 'dependency', 'runs/run/trace/state.bin': 'raw',
                               'sources/keep/fixture.bin': 'important', 'downloads/source.zip': 'archive'}.items():
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(contents)
        (root / 'empty_keep').mkdir()
        (root / 'runs/run/stdout.log.gz').write_bytes(gzip.compress(b'compressed experiment log\n'))
        policy = {'schema': 1, 'reports_root': str(base / 'saved'), 'protected_paths': [],
                  'trees': [{'path': str(root), 'mode': 'important',
                             'keep_prefixes': ['sources/keep', 'empty_keep'], 'note': ''}]}
        policy_path = base / 'policy.json'; policy_path.write_bytes(cleanup.encoded(policy))
        plan_path = base / 'plan.json'; receipt_path = base / 'receipt.json'
        with mock.patch.object(cleanup, 'assert_linux'), mock.patch.object(cleanup, 'validate_policy', side_effect=lambda v: v):
            cleanup.make_plan(policy_path, plan_path)
            cleanup.preserve(plan_path, receipt_path)
        return base, root, policy, plan_path, receipt_path

    def test_path_and_classification_contract(self):
        for bad in ['/root', '/root/autodl-tmp', '/etc/project', '/root/autodl-tmp/../x',
                    '/root/autodl-tmp/autodl_factor_20261004_4090', '/root/autodl-tmp/x/autodl_factor_new']:
            with self.assertRaises(ValueError): cleanup.validate_tree_text(bad)
        self.assertEqual(cleanup.validate_tree_text('/root/autodl-tmp/old_project'), '/root/autodl-tmp/old_project')
        tree = {'keep_prefixes': ['runs/important']}
        for name in ['reports/a.md', 'reports/figure.png', 'runs/r/stats.json', 'manifests/build.json',
                     'runs/r/trace/frames.csv', 'runs/r/trace/metadata.json', 'reports/docs/analysis.pdf',
                     'runs/r/stdout.log.gz']:
            self.assertEqual(cleanup.classify(name, tree), ('retain', 'experimental_report_or_metadata', True))
        for name in ['src/a.json', 'vendor/readme.md', 'builds/notes.txt', 'runs/r/trace/state.json',
                     'README.md', 'CMakeLists.txt', 'downloads/source.tar.gz', 'assets/mesh.zip',
                     'tools/fixture.json', 'scripts/input.json', 'references/source.md', 'externals/config.json',
                     'toolsrc/config.json', 'StiffGIPC/test.json', 'MeshProcess/info.txt']:
            self.assertEqual(cleanup.classify(name, tree)[0], 'delete', name)
        self.assertEqual(cleanup.classify('run_results.tar.gz', tree)[1], 'archive_external_report_gate')
        self.assertEqual(cleanup.classify('builds/old_results.tar.gz', tree)[1], 'archive_external_report_gate')
        self.assertEqual(cleanup.classify('runs/important/trace/state.bin', tree)[1], 'important_exact_prefix')
        for root in cleanup.DEPENDENCY_ONLY_ROOTS:
            dependency = {'path': root, 'keep_prefixes': []}
            for name in ['versions/registry.json', 'reports/readme.md', 'package.tar.gz']:
                self.assertEqual(cleanup.classify(name, dependency), ('delete', 'explicit_dependency_only_tree', False))
            dependency['keep_prefixes'] = ['important']
            self.assertEqual(cleanup.classify('important/fixture.json', dependency), ('retain', 'important_exact_prefix', False))
        important_source = {'keep_prefixes': ['tools']}
        self.assertEqual(cleanup.classify('tools/fixture.json', important_source), ('retain', 'important_exact_prefix', False))

    def test_preserved_reports_and_mocked_apply(self):
        base, root, _, plan_path, receipt_path = self.fixture()
        plan = cleanup.read(plan_path)
        cleanup.verify_preservation(plan_path, plan, receipt_path)
        with mock.patch.object(cleanup, 'assert_linux'), mock.patch.object(cleanup, 'validate_policy', side_effect=lambda v: v), \
             mock.patch.object(cleanup, 'active_process_conflicts', return_value=[]), \
             mock.patch.object(cleanup.os, 'unlink') as unlink, mock.patch.object(cleanup.os, 'rmdir') as rmdir, \
             mock.patch.object(cleanup, 'unlink_planned', side_effect=lambda path, row, own: cleanup.os.unlink(path)):
            result = cleanup.apply(plan_path, receipt_path, cleanup.sha(plan_path), base / 'apply.json')
        self.assertTrue(result['completed'])
        self.assertEqual(unlink.call_count, 3)
        deleted = {str(call.args[0]) for call in unlink.call_args_list}
        self.assertNotIn(str(root / 'sources/keep/fixture.bin'), deleted)
        self.assertNotIn(str(root / 'empty_keep'), {str(call.args[0]) for call in rmdir.call_args_list})
        self.assertTrue((root / 'runs/run/trace/state.bin').exists(), 'Test must never actually delete payload')

    def test_actual_temporary_hardlinks_keep_outside_link(self):
        base = Path(__file__).resolve().parents[2] / 'runs' / ('history_cleanup_hardlinks_' + uuid.uuid4().hex)
        selected = base / 'selected'; protected = base / 'protected'
        selected.mkdir(parents=True); protected.mkdir()
        retained = protected / 'retained.bin'; retained.write_bytes(b'protected original contents')
        paths = [selected / 'a.bin', selected / 'b.bin']
        for path in paths: os.link(retained, path)
        rows = [cleanup.identity(path) for path in paths]
        original_hash = cleanup.sha(retained); observed_ctimes = {}
        # Windows cannot unlink an os.open handle without FILE_SHARE_DELETE.
        # Emulate Linux O_PATH handles with the real surviving same-inode link;
        # unlink and nlink/fstat metadata below remain real filesystem actions.
        with ExitStack() as stack:
            if os.name != 'posix':
                for patcher in [mock.patch.object(cleanup.os, 'O_PATH', 0, create=True),
                                mock.patch.object(cleanup.os, 'O_NOFOLLOW', 0, create=True),
                                mock.patch.object(cleanup.os, 'open', return_value=123456),
                                mock.patch.object(cleanup.os, 'fstat', side_effect=lambda fd: retained.stat()),
                                mock.patch.object(cleanup.os, 'close')]:
                    stack.enter_context(patcher)
            for path, row in zip(paths, rows): cleanup.unlink_planned(path, row, observed_ctimes)
        self.assertTrue(all(not path.exists() for path in paths))
        self.assertEqual(retained.stat().st_nlink, 1)
        self.assertEqual(cleanup.sha(retained), original_hash)

    def test_only_own_unlink_ctime_transitions_are_accepted(self):
        def status(ctime, nlink):
            return SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_dev=1, st_ino=2, st_size=5,
                                   st_mtime_ns=100, st_ctime_ns=ctime, st_nlink=nlink)
        first, second, third = status(10, 3), status(20, 2), status(30, 1)
        row = cleanup.stat_identity(first); own = {}
        with mock.patch.object(cleanup.os, 'O_PATH', 0, create=True), \
             mock.patch.object(cleanup.os, 'O_NOFOLLOW', 0, create=True), \
             mock.patch.object(cleanup.os, 'open', return_value=123456), \
             mock.patch.object(cleanup.os, 'fstat', side_effect=[first, second, second, third]), \
             mock.patch.object(cleanup.os, 'close'), mock.patch.object(cleanup.os, 'unlink') as unlink, \
             mock.patch.object(cleanup, 'identity', side_effect=[row, row, cleanup.stat_identity(second), cleanup.stat_identity(second)]):
            cleanup.unlink_planned(Path('a'), row, own)
            cleanup.unlink_planned(Path('b'), row, own)
            self.assertEqual(unlink.call_count, 2)
        self.assertEqual(own[(1, 2)], 30)
        with mock.patch.object(cleanup, 'identity', return_value=cleanup.stat_identity(status(40, 1))), \
             mock.patch.object(cleanup.os, 'unlink') as unlink:
            with self.assertRaisesRegex(ValueError, 'File changed after preflight'):
                cleanup.unlink_planned(Path('c'), row, own)
            unlink.assert_not_called()

    def test_changed_tree_and_active_process_refuse_before_unlink(self):
        base, root, _, plan_path, receipt_path = self.fixture()
        patches = (mock.patch.object(cleanup, 'assert_linux'),
                   mock.patch.object(cleanup, 'validate_policy', side_effect=lambda v: v))
        with patches[0], patches[1], mock.patch.object(cleanup.os, 'unlink') as unlink, \
             mock.patch.object(cleanup, 'active_process_conflicts', return_value=[{'pid': '123', 'path': str(root)}]):
            with self.assertRaisesRegex(ValueError, 'Active process'):
                cleanup.apply(plan_path, receipt_path, cleanup.sha(plan_path), base / 'active_apply.json')
            unlink.assert_not_called()
        (root / 'new.bin').write_text('new')
        with mock.patch.object(cleanup, 'assert_linux'), mock.patch.object(cleanup, 'validate_policy', side_effect=lambda v: v), \
             mock.patch.object(cleanup.os, 'unlink') as unlink:
            with self.assertRaisesRegex(ValueError, 'file set or file identities changed'):
                cleanup.apply(plan_path, receipt_path, cleanup.sha(plan_path), base / 'changed_apply.json')
            unlink.assert_not_called()

    def test_backup_corruption_and_directory_escape_refused(self):
        base, _, _, plan_path, receipt_path = self.fixture()
        plan = cleanup.read(plan_path); receipt = cleanup.read(receipt_path)
        Path(receipt['reports'][0]['backup']).write_text('corruption')
        with self.assertRaisesRegex(ValueError, 'Report backup changed'):
            cleanup.verify_preservation(plan_path, plan, receipt_path)
        plan['directories'][0]['path'] = str(base.parent)
        changed = base / 'tampered.json'; changed.write_bytes(cleanup.encoded(plan))
        with mock.patch.object(cleanup, 'validate_policy', side_effect=lambda v: v):
            with self.assertRaisesRegex(ValueError, 'Planned directory is outside'):
                cleanup.load_plan(changed)


if __name__ == '__main__':
    unittest.main()
