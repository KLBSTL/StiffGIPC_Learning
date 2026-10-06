"""Tiny local synthetic archives only; no access to real/remote archives."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from autodl_archive_reports import ArchivePolicyError, extract_reports, file_identity, load_reviewed_plan


def fixture(path, members):
    with tarfile.open(path, 'w' if str(path).endswith('.tar') else 'w:gz') as archive:
        for name, content, kind in members:
            entry = tarfile.TarInfo(name)
            if kind == 'link':
                entry.type = tarfile.SYMTYPE; entry.linkname = '../../outside'
                archive.addfile(entry)
            else:
                entry.size = len(content); archive.addfile(entry, io.BytesIO(content))


class ArchiveReports(unittest.TestCase):
    def test_benign_policy_hashes_and_no_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); archive = root / 'old.tar.gz'; destination = root / 'reports'
            fixture(archive, [('./reports/summary.md', b'report', 'file'),
                ('runs/test/output/stats.json', b'{"frames":[]}', 'file'),
                ('runs/test/trace/frames.csv', b'frame,ms\n1,3\n', 'file'),
                ('reports/figures/plot.png', b'png', 'file'),
                ('reports/direct_plot.png', b'direct png', 'file'),
                ('sources/x/README.md', b'code docs', 'file'),
                ('builds/build.log', b'build noise', 'file'),
                ('runs/test/trace/state_0001.bin', b'raw', 'file'),
                ('runs/test/frame.png', b'mesh render', 'file'),
                ('tools/code.py', b'code', 'file'),
                ('./sources/skip_link', b'', 'link')])
            result = extract_reports([archive], destination)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(len(result['selected_files']), 5)
            self.assertEqual(len(result['archives'][0]['ignored_links']), 1)
            self.assertEqual(result['archives'][0]['archive_sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
            for row in result['selected_files']:
                data = (destination / row['output_path']).read_bytes()
                self.assertEqual(row['sha256'], hashlib.sha256(data).hexdigest())
            self.assertFalse((destination / archive.name / 'sources').exists())
            with self.assertRaises(FileExistsError):
                extract_reports([archive], destination)

    def test_traversal_fails_closed_even_if_not_selected(self):
        for name, kind in [('../outside.md', 'file'), ('/absolute.json', 'file'),
                           ('C:/absolute.json', 'file'), ('./sources/../../outside', 'link')]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); archive = root / 'evil.tar.gz'; destination = root / 'reports'
                fixture(archive, [('reports/ok.md', b'ok', 'file'), (name, b'bad', kind)])
                with self.assertRaises(ArchivePolicyError):
                    extract_reports([archive], destination)
                self.assertEqual(json.loads((destination / 'manifest.json').read_text())['status'], 'failed')
                self.assertFalse((root / 'outside.md').exists())

    def test_oversize_no_truncation_duplicate_and_corrupt_gzip(self):
        for mode in ('oversize', 'duplicate', 'truncated'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); archive = root / 'old.tar.gz'; destination = root / 'reports'
                members = [('reports/report.md', b'1234567890', 'file')]
                if mode == 'duplicate':
                    members *= 2
                fixture(archive, members)
                if mode == 'truncated':
                    archive.write_bytes(archive.read_bytes()[:-5])
                with self.assertRaises((ArchivePolicyError, OSError, EOFError)):
                    extract_reports([archive], destination, max_entry_bytes=5 if mode == 'oversize' else 64)
                result = json.loads((destination / 'manifest.json').read_text())
                self.assertEqual(result['status'], 'failed')
                if mode == 'oversize':
                    self.assertFalse((destination / archive.name / 'reports/report.md').exists())

    def test_plan_duplicate_basenames_tar_zip_compressed_log_and_empty_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); a = root / 'a'; b = root / 'b'; a.mkdir(); b.mkdir()
            first = a / 'same.tar.gz'; second = b / 'same.tar.gz'
            plain = root / 'plain.tar'; zipped = root / 'source.zip'
            fixture(first, [('reports/summary.md', b'a report', 'file'),
                            ('runs/a/run.log.gz', b'compressed-log-bytes', 'file')])
            fixture(second, [('reports/summary.md', b'b report', 'file')])
            fixture(plain, [('reports/data.json', b'{}', 'file')])
            with zipfile.ZipFile(zipped, 'w') as stream:
                stream.writestr('sources/main.py', b'code only, no report')
            archives = [first, second, plain, zipped]
            context = {'source_plan': {'sha256': 'a' * 64}, 'excluded_compressed_logs': [],
                       'rows': [{'path': str(path.resolve()), **file_identity(path.stat())} for path in archives]}
            destination = root / 'reports'
            result = extract_reports(archives, destination, plan_context=context)
            self.assertEqual(result['schema'], 'autodl.archive_reports.v2')
            self.assertEqual(len(result['selected_files']), 4)
            self.assertEqual(result['archives'][-1]['selected_count'], 0)
            self.assertEqual(len({v['archive_label'] for v in result['archives']}), 4)
            for i, row in enumerate(result['archives']):
                expected = f'{i:03d}_{hashlib.sha256(str(archives[i].resolve()).encode()).hexdigest()[:12]}_{archives[i].name}'
                self.assertEqual(row['archive_label'], expected)
                self.assertEqual(row['archive_sha256'], hashlib.sha256(archives[i].read_bytes()).hexdigest())
                self.assertEqual(row['plan_identity'], file_identity(archives[i].stat()))
            for row in result['selected_files']:
                self.assertEqual(Path(row['output_path']).parts[0], row['archive_label'])
            legacy = extract_reports([first], root / 'legacy')
            self.assertEqual(legacy['schema'], 'autodl.archive_reports.v1')
            self.assertEqual(len(legacy['selected_files']), 1)
            self.assertNotIn('archive_label', legacy['archives'][0])

    def test_plan_sha_binding_selection_and_identity_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); archive = root / 'old.tar'; fixture(archive, [('reports/a.md', b'ok', 'file')])
            row = {'path': '/root/autodl-tmp/old/a.tar', 'tree': '/root/autodl-tmp/old',
                   'relative_path': 'a.tar', 'reason': 'archive_external_report_gate',
                   'action': 'retain', 'preserve_report': False, **file_identity(archive.stat())}
            plan = {'schema': 1, 'plan_id': 'history_' + 'b' * 32, 'policy_sha256': 'c' * 64,
                    'tool_sha256': 'd' * 64, 'files': [row, row | {'path': '/root/autodl-tmp/old/run.log.gz',
                    'relative_path': 'run.log.gz'}]}
            path = root / 'plan.json'; path.write_text(json.dumps(plan))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            context = load_reviewed_plan(path, digest)
            self.assertEqual(len(context['rows']), 1)
            self.assertEqual(context['excluded_compressed_logs'], ['/root/autodl-tmp/old/run.log.gz'])
            with self.assertRaises(ArchivePolicyError):
                load_reviewed_plan(path, '0' * 64)
            native = {'source_plan': context['source_plan'], 'excluded_compressed_logs': [],
                      'rows': [row | {'path': str(archive.resolve()), 'mtime_ns': row['mtime_ns'] - 1}]}
            with self.assertRaises(ArchivePolicyError):
                extract_reports([archive], root / 'output', plan_context=native)
            self.assertFalse((root / 'output').exists())
            plan['files'][0]['path'] = '/root/autodl-tmp/old/../outside.tar'
            path.write_text(json.dumps(plan)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ArchivePolicyError):
                load_reviewed_plan(path, digest)

    def test_zip_report_traversal_and_selected_crc_failure(self):
        for mode in ('benign', 'traversal', 'crc'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); archive = root / 'old.zip'; destination = root / 'reports'
                with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as stream:
                    stream.writestr('reports/report.md', b'UNIQUE_REPORT_CONTENT')
                    if mode == 'traversal':
                        stream.writestr('../escape.md', b'bad')
                if mode == 'crc':
                    archive.write_bytes(archive.read_bytes().replace(b'UNIQUE_REPORT_CONTENT', b'XNIQUE_REPORT_CONTENT'))
                if mode == 'benign':
                    result = extract_reports([archive], destination)
                    self.assertEqual(len(result['selected_files']), 1)
                else:
                    with self.assertRaises((ArchivePolicyError, zipfile.BadZipFile)):
                        extract_reports([archive], destination)
                    self.assertEqual(json.loads((destination / 'manifest.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
