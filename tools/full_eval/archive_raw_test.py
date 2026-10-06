import json
from pathlib import Path
import tempfile
import unittest
import archive_raw as a


class ArchiveTests(unittest.TestCase):
    def setup_run(self, root):
        run = root / 'runs/session/p1_fixed_combined_graph'
        (run / 'trace').mkdir(parents=True)
        (run / 'trace/000001.bin').write_bytes(b'full raw state')
        (run / 'final.bin').write_bytes(b'final state')
        (run / 'run.log').write_text('preserved log')
        (run / 'result.json').write_text(json.dumps({'status': 'completed'}))
        rows = [{'path': p.relative_to(run).as_posix(), 'bytes': p.stat().st_size, 'sha256': a.sha(p)}
                for p in sorted(run.rglob('*')) if p.is_file()]
        (run / 'evidence.json').write_text(json.dumps({'files': rows}))
        return run, run.relative_to(root).as_posix(), 'runs/session/archives/' + run.name + '.tar.gz'

    def test_archive_download_proof_and_exact_pruning(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); run, name, archive = self.setup_run(root)
            ready = a.prepare(root, name, archive)
            self.assertTrue((run / 'final.bin').is_file())
            receipt = a.persist(root, name, ready['archive_sha256'], run.name + '.tar.gz')
            self.assertTrue(receipt['download_verified'])
            self.assertFalse((run / 'final.bin').exists())
            self.assertTrue((run / 'run.log').exists())
            self.assertTrue((root / archive).exists())
            with self.assertRaises(ValueError):
                a.persist(root, name, ready['archive_sha256'], run.name + '.tar.gz')

    def test_wrong_download_leaves_all_raw(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); run, name, archive = self.setup_run(root)
            a.prepare(root, name, archive)
            with self.assertRaises(ValueError): a.persist(root, name, 'wrong', run.name + '.tar.gz')
            self.assertTrue((run / 'final.bin').exists())
            self.assertFalse((run / 'archive_receipt.json').exists())

    def test_unsafe_evidence_path_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); run, name, archive = self.setup_run(root)
            d = json.loads((run / 'evidence.json').read_text())
            d['files'].append({'path': 'trace/../../outside.bin', 'bytes': 1, 'sha256': 'bad'})
            (run / 'evidence.json').write_text(json.dumps(d))
            with self.assertRaises(ValueError): a.prepare(root, name, archive)

    def test_same_size_tamper_rejected_before_pruning(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); run, name, archive = self.setup_run(root)
            ready = a.prepare(root, name, archive)
            (run / 'final.bin').write_bytes(b'other state')
            with self.assertRaises(ValueError):
                a.persist(root, name, ready['archive_sha256'], run.name + '.tar.gz')
            self.assertTrue((run / 'trace/000001.bin').exists())


if __name__ == '__main__':
    unittest.main()
