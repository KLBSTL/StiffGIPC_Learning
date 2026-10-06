"""Small synthetic transfer safety checks; no GPU, SSH or real trace deletion."""
import io
import json
from pathlib import Path
import tarfile
import unittest
from unittest.mock import patch
import uuid

import autodl_transfer as transfer


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.area = transfer.ROOT / 'runs' / ('transfer_selftest_' + uuid.uuid4().hex)
        cls.remote = cls.area / 'remote'
        cls.local = cls.area / 'local'
        cls.remote.mkdir(parents=True)
        transfer.write_new(cls.remote / 'autodl_bundle.json', {'synthetic': True})
        binary = cls.remote / 'builds/autodl-active/active/gipc'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'not an executable; synthetic bytes only')
        transfer.write_new(cls.remote / 'builds/autodl-active/manifest.json',
                           {'binaries': {'active': {'path': binary.relative_to(cls.remote).as_posix(),
                                                    'sha256': transfer.sha(binary)}}})
        cls.plans = {}
        for stage, repeats in [('smoke', [1]), ('window', [1, 2])]:
            tasks = []
            for repeat in repeats:
                name = f'autodl_{stage}_unit_r{repeat}'
                task = {'name': name, 'repeat': repeat, 'binary': 'active', 'config': {}}
                tasks.append(task)
                result = {'status': 'completed', 'recorded_frames': 1}
                transfer.write_new(cls.remote / 'runs/active' / name / 'result.json', result)
                trace = cls.remote / 'runs/active' / name / 'trace'
                trace.mkdir()
                (trace / 'state_0000.bin').write_bytes(b'synthetic trace ' + str(repeat).encode())
                (trace.parent / 'final.bin').write_bytes(b'preserve this')
            plan_path = cls.remote / f'configs/active/autodl_{stage}.json'
            transfer.write_new(plan_path, {'stage': stage, 'runs': tasks})
            ledger = {'plan_sha256': transfer.sha(plan_path), 'source_digest': 'synthetic-source',
                      'candidate_sha256': 'synthetic-candidate',
                      'runs': [task | {'result': transfer.read(cls.remote / 'runs/active' / task['name'] / 'result.json')}
                               for task in tasks]}
            transfer.write_new(cls.remote / f'reports/active/AUTODL_{stage.upper()}_BATCH.json', ledger)
            cls.plans[stage] = ledger

    def test_transfer_and_receipt_protection(self):
        archive = self.remote / 'transfers/smoke.tar.gz'
        exported = transfer.export(self.remote, 'smoke', None, archive)
        received = transfer.receive(archive, exported['archive_sha256'], self.local)
        receipt = Path(received['receipt'])
        self.assertTrue(receipt.is_file())
        run = self.remote / 'runs/active/autodl_smoke_unit_r1'
        before = (run / 'trace/state_0000.bin').read_bytes()
        with patch.object(transfer.platform, 'system', return_value='Linux'), patch.object(transfer.shutil, 'rmtree') as remove:
            pruned = transfer.prune(self.remote, receipt)
            remove.assert_called_once_with((run / 'trace').resolve())
            self.assertEqual(len(pruned['removed_trace_directories']), 1)
        self.assertEqual((run / 'trace/state_0000.bin').read_bytes(), before)
        self.assertTrue((run / 'final.bin').is_file() and archive.is_file())
        with self.assertRaises(ValueError):
            transfer.receive(archive, exported['archive_sha256'], self.local)
        with self.assertRaises(ValueError):
            transfer.receive(archive, '0' * 64, self.area / 'bad_hash')
        changed = transfer.read(receipt)
        changed['archive_sha256'] = '0' * 64
        bad_receipt = self.area / 'bad_receipt.json'
        transfer.write_new(bad_receipt, changed)
        with patch.object(transfer.platform, 'system', return_value='Linux'), patch.object(transfer.shutil, 'rmtree') as remove:
            with self.assertRaises(ValueError):
                transfer.prune(self.remote, bad_receipt)
            remove.assert_not_called()

    def test_incremental_ledger_and_no_run_overwrite(self):
        ledger_path = self.remote / 'reports/active/AUTODL_WINDOW_BATCH.json'
        complete = self.plans['window']
        ledger_path.write_bytes(transfer.encoded(complete | {'runs': complete['runs'][:1]}))
        first = transfer.export(self.remote, 'window', 1, self.remote / 'transfers/window1.tar.gz')
        transfer.receive(Path(first['archive']), first['archive_sha256'], self.area / 'rounds')
        ledger_path.write_bytes(transfer.encoded(complete))
        second = transfer.export(self.remote, 'window', 2, self.remote / 'transfers/window2.tar.gz')
        transfer.receive(Path(second['archive']), second['archive_sha256'], self.area / 'rounds')
        self.assertEqual(transfer.read(self.area / 'rounds/reports/active/AUTODL_WINDOW_BATCH.json'), complete)
        repeated = transfer.export(self.remote, 'window', 1, self.remote / 'transfers/window1_again.tar.gz')
        with self.assertRaises(ValueError):
            transfer.receive(Path(repeated['archive']), repeated['archive_sha256'], self.area / 'rounds')

    def test_hostile_tar_members_rejected_before_extraction(self):
        for index, name in enumerate(('../escape', '/absolute', 'C:/drive', 'valid/symlink')):
            archive_path = self.area / f'hostile{index}.tar.gz'
            with tarfile.open(archive_path, 'w:gz') as archive:
                member = tarfile.TarInfo(name)
                if index == 3:
                    member.type = tarfile.SYMTYPE; member.linkname = '../escape'
                    archive.addfile(member)
                else:
                    member.size = 1; archive.addfile(member, io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                transfer.receive(archive_path, transfer.sha(archive_path), self.area / f'hostile_dest{index}')
            self.assertFalse((self.area / f'hostile_dest{index}').exists())

    def test_payload_hash_mismatch_before_extraction(self):
        archive_path = self.area / 'corrupt_payload.tar.gz'
        control = transfer.encoded({'files': [{'path': 'runs/active/unit/trace/data.bin',
                                               'bytes': 3, 'sha256': '0' * 64}]})
        with tarfile.open(archive_path, 'w:gz') as archive:
            for name, data in [(transfer.CONTROL, control), ('runs/active/unit/trace/data.bin', b'bad')]:
                member = tarfile.TarInfo(name); member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        target = self.area / 'corrupt_payload_dest'
        with self.assertRaises(ValueError):
            transfer.receive(archive_path, transfer.sha(archive_path), target)
        self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
