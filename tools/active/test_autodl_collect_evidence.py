"""Tiny 72-run evidence fixture; no solver/GPU, remote access or deletions."""
import io
import json
from pathlib import Path
import tarfile
import unittest
import uuid

import autodl_collect_evidence as collector


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        base = Path(__file__).resolve().parents[2] / 'runs' / ('evidence_selftest_' + uuid.uuid4().hex)
        root = base / 'remote'; root.mkdir(parents=True)
        def put(name, value):
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value if isinstance(value, bytes) else collector.encoded(value))
        for name in collector.CONFIGS: put(f'configs/active/autodl_{name}.json', {})
        put('autodl_bundle.json', {}); put('builds/autodl-active/manifest.json', {})
        for name in ('components.json', 'guards.json', 'active_configure.log', 'active_build.log',
                     'base_configure.log', 'base_build.log', 'validator_configure.log', 'validator_build.log',
                     'validator_selftest.log', 'components.json.log', 'guards.json.log'):
            put('builds/autodl-active/' + name, b'{}')
        for name in collector.TOOLS: put('tools/active/' + name, b'# test identity\n')
        put('tools/active/autodl_collect_evidence.py', Path(collector.__file__).read_bytes())
        put('tools/validator/diagnose_first_path.cpp', b'// fixture\n')
        rows = []
        roles = ('stiff', 'ipc_host', 'ipc_graph', 'toi_host', 'toi_graph', 'toi_graph_triangular')
        for stage, repeats in [('smoke', (1,)), ('window', (1, 2, 3))]:
            planned, recorded = [], []
            for repeat in repeats:
                for scene in ('mixed', 'hang', 'fixed_bunny'):
                    for role in roles:
                        name = f'autodl_{stage}_{scene}_{role}_r{repeat}'
                        task = {'name': name, 'scene_key': scene, 'repeat': repeat,
                                'binary': 'base' if role == 'stiff' else 'active', 'config': {'steps': 2}}
                        result = {'status': 'completed', 'recorded_frames': 2}
                        planned.append(task); recorded.append(task | {'result': result}); rows.append(task)
                        for filename in collector.RUN_FILES: put(f'runs/active/{name}/' + filename, b'{}')
                        put(f'runs/active/{name}/result.json', result)
                        if task['binary'] == 'active': put(f'runs/active/{name}/resolved_config.json', {})
            plan_path = f'configs/active/autodl_{stage}.json'
            put(plan_path, {'stage': stage, 'runs': planned})
            put(f'reports/active/AUTODL_{stage.upper()}_BATCH.json', {'plan_sha256': collector.sha(root / plan_path), 'runs': recorded})
        for name in ('AUTODL_FACTOR_WINDOW_ANALYSIS.json', 'AUTODL_FACTOR_WINDOW_ANALYSIS_V2.json',
                     'AUTODL_FACTOR_CCD_REAGGREGATED.json'): put('reports/active/' + name, {})
        audited = []
        for row in rows:
            if not row['name'].startswith('autodl_window_'): continue
            prefix = 'reports/active/AUTODL_FACTOR_CCD/' + row['name']
            for suffix in ('.record.json', '.identity.json', '.stdout.log', '.ccd.json'): put(prefix + suffix, {})
            audited.append({'run': row['name'], 'native_ccd_sha256': collector.sha(root / (prefix + '.ccd.json'))})
        put('reports/active/AUTODL_FACTOR_CCD/audit.json', {'runs': audited})
        scenes = []
        for scene in ('mixed', 'hang', 'fixed_bunny'):
            panels = []; initial = {}
            for repeat in (1, 2, 3):
                for role in ('stiff', 'toi_graph', 'toi_graph_triangular'):
                    name = f'autodl_window_{scene}_{role}_r{repeat}'
                    for filename in ('topology.bin', 'state_0000.bin', 'boundary_types.bin', 'masses.bin', 'body_ids.bin', 'metadata.json', 'scene.json'):
                        relative = f'runs/active/{name}/' + ('output/' if filename == 'scene.json' else 'trace/') + filename
                        put(relative, b'{}'); initial[filename] = collector.sha(root / relative)
                    final = f'runs/active/{name}/trace/state_0002.bin'; put(final, b'final state')
                    put(f'runs/active/{name}/trace/state_0001.bin', b'NEVER EXPORT RAW MIDDLE FRAME')
                    panels.append({'run': name, 'frame': 2, 'state_sha256': collector.sha(root / final)})
            image = f'reports/active/AUTODL_FACTOR_FIGURES/autodl_factor_{scene}_final.png'; put(image, b'fake fixture PNG')
            scenes.append({'scene_key': scene, 'frame': 2, 'image': str(root / image),
                           'image_sha256': collector.sha(root / image), 'initial_identity_sha256': initial, 'panels': panels})
        put('reports/active/AUTODL_FACTOR_FIGURES/render_manifest.json', {'all_three_scenes_rendered': True, 'scenes': scenes})
        put('reports/active/AUTODL_FACTOR_FIGURES/.mplconfig/cache.json', b'NEVER EXPORT CACHE')
        put('cleanup/backups/old.json', b'NEVER EXPORT HISTORY')
        for name in ('build', 'smoke', 'window1', 'window2', 'window3', 'analysis'):
            put(f'jobs/{name}.json', {'status': 'completed', 'ended_utc': '2026-10-05T00:00:00Z'})
            put(f'jobs/{name}.log', b'finalized log')
        return base, root

    def test_export_verify_receive_exact_render_inputs(self):
        base, root = self.fixture()
        result = collector.export(root, base / 'export', True)
        manifest = collector.verify(Path(result['archive']), result['archive_sha256'])
        self.assertEqual(manifest['selection']['planned_runs'], 72)
        self.assertEqual(len(manifest['selection']['render_panels']), 27)
        names = {row['path'] for row in manifest['files']}
        self.assertFalse(any('state_0001.bin' in name or '.mplconfig' in name or name.startswith('cleanup/') for name in names))
        self.assertEqual(sum(name.endswith('trace/state_0002.bin') for name in names), 27)
        received = collector.receive(Path(result['archive']), result['archive_sha256'], base / 'local')
        self.assertTrue(received['verified'])
        self.assertEqual(received['received_files'], len(names))
        bad = base / 'conflict'; (bad / 'tools/active').mkdir(parents=True)
        (bad / 'tools/active/config.py').write_text('local user edit')
        with self.assertRaisesRegex(ValueError, 'Existing local file conflicts'):
            collector.receive(Path(result['archive']), result['archive_sha256'], bad)
        self.assertFalse((bad / 'autodl_bundle.json').exists(), 'Conflicts must fail before any extraction')
        for index, filename in enumerate(('runs', 'evidence_receipts')):
            blocked = base / f'blocked_parent_{index}'; blocked.mkdir(); (blocked / filename).write_text('local file')
            with self.assertRaisesRegex(ValueError, 'Existing parent is not a directory'):
                collector.receive(Path(result['archive']), result['archive_sha256'], blocked)
            self.assertFalse((blocked / 'autodl_bundle.json').exists(), 'Parent conflicts must fail before writing')

    def test_ledger_and_pending_job_rejected(self):
        base, root = self.fixture()
        job = root / 'jobs/analysis.json'; job.write_bytes(collector.encoded({'status': 'running'}))
        with self.assertRaisesRegex(ValueError, 'Job has not finalized'):
            collector.export(root, base / 'bad')
        self.assertFalse((base / 'bad').exists())
        job.write_bytes(collector.encoded({'status': 'completed', 'ended_utc': 'now'}))
        ledger = root / 'reports/active/AUTODL_WINDOW_BATCH.json'; data = collector.read(ledger)
        data['runs'].pop(); ledger.write_bytes(collector.encoded(data))
        with self.assertRaisesRegex(ValueError, 'Incomplete or duplicate'):
            collector.export(root, base / 'bad2')

    def test_malicious_members_and_hash_mismatch_rejected(self):
        base = Path(__file__).resolve().parents[2] / 'runs' / ('evidence_bad_selftest_' + uuid.uuid4().hex); base.mkdir(parents=True)
        for i, name in enumerate(('../escape', '/absolute', 'C:/drive', 'safe.txt')):
            archive = base / f'bad{i}.tar.gz'
            content = b'payload'; record = {'path': name, 'bytes': len(content), 'sha256': '0' * 64}
            manifest = {'schema': 'autodl.evidence.v1', 'export_id': uuid.uuid4().hex, 'files': [record], 'payload_bytes': len(content)}
            with tarfile.open(archive, 'w:gz') as tar:
                blob = collector.encoded(manifest); info = tarfile.TarInfo(collector.CONTROL); info.size = len(blob); tar.addfile(info, io.BytesIO(blob))
                info = tarfile.TarInfo(name); info.size = len(content); tar.addfile(info, io.BytesIO(content))
            with self.assertRaises(ValueError): collector.verify(archive, collector.sha(archive))


if __name__ == '__main__': unittest.main()
