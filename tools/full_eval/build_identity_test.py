"""CPU contracts with fake compilers, objects and mocked Ninja command output."""
import copy
import json
import os
from pathlib import Path
import shlex
import tempfile
import time
import unittest

import build_identity as audit


class BuildIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='build identity ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        for folder in audit.SOURCE_DIRS:
            (self.root / folder).mkdir(parents=True)
        for name in audit.SOURCE_FILES:
            (self.root / name).write_text('project(test)\n')
        for prefix in ('', 'baseline/'):
            (self.root / prefix / 'Assets/input.bin').write_bytes(b'identical asset')
            (self.root / prefix / 'MeshProcess/source.c').write_text('int dependency;')
            assets = self.root / prefix / 'Assets'
            for folder in ('benchmark_scenes', 'benchmark_meshes', 'sorted_mesh'):
                (assets / folder).mkdir()
            mesh = assets / 'benchmark_meshes/fixture.obj'
            mesh.write_bytes(b'mesh fixture')
            for name in ('fixture_sorted.16.obj', 'fixture_sorted.16.part'):
                (assets / 'sorted_mesh' / name).write_bytes(b'cache fixture')
            for case in audit.SELECTED_SCENES:
                (assets / 'benchmark_scenes' / (case + '.json')).write_text(json.dumps({
                    'case_id': case, 'objects': [{'stiff_mesh': 'benchmark_meshes/fixture.obj',
                                                'stiff_mesh_sha256': audit.sha(mesh)}]}))
        for kind, counts in audit.EXPECTED.items():
            source = self.root if kind == 'active' else self.root / 'baseline'
            for suffix, number in (('.cu', counts['cuda']), ('.cpp', counts['cxx'])):
                for i in range(number):
                    (source / 'StiffGIPC' / f'unit_{i}{suffix}').write_text(f'// {kind} {i}\n')
        self.before = self.root / 'before.json'
        self.before.write_text(json.dumps(audit.snapshot(self.root)))
        # Synthetic timestamps are controlled, not derived from filesystem write
        # time. Linux overlay mtime may lag time.time() by a clock tick. Keep
        # each ordering boundary >= 2 seconds apart, with no actual sleeping.
        self.start_ns = (time.time_ns() // 1_000_000_000 - 30) * 1_000_000_000
        self.stamp(self.before, self.start_ns - 2_000_000_000)
        (self.root / 'build_start.json').write_text(json.dumps({
            'time_unix': self.start_ns / 1_000_000_000, 'pid': 123}))
        self.dumps = {}
        for kind in audit.EXPECTED:
            self.make_build(kind)

    def make_build(self, kind):
        source = self.root if kind == 'active' else self.root / 'baseline'
        build = self.root / 'build' / kind
        (build / 'CMakeFiles/gipc.dir').mkdir(parents=True)
        tools = self.root / 'toolchain'
        tools.mkdir(exist_ok=True)
        for name in ('nvcc', 'cxx', 'ninja'):
            (tools / name).write_bytes(b'fake CPU contract compiler ' + name.encode())
        cache = {'CMAKE_HOME_DIRECTORY': str(source), 'CMAKE_GENERATOR': 'Ninja',
                 'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_CUDA_ARCHITECTURES': '89',
                 'CMAKE_CUDA_COMPILER': str(tools / 'nvcc'),
                 'CMAKE_CXX_COMPILER': str(tools / 'cxx'), 'CMAKE_MAKE_PROGRAM': str(tools / 'ninja')}
        (build / 'CMakeCache.txt').write_text('\n'.join(k + ':STRING=' + v for k, v in cache.items()))
        (build / 'build.ninja').write_text('# fake generated build graph\n')
        (build / 'CMakeFiles/rules.ninja').write_text('# fake generated rules\n')
        commands, actual, objects, log = [], [], [], ['# ninja log v5']
        for src in sorted((source / 'StiffGIPC').glob('*')):
            obj = 'CMakeFiles/gipc.dir/StiffGIPC/' + src.name + '.o'
            (build / obj).parent.mkdir(parents=True, exist_ok=True)
            (build / obj).write_bytes(('object ' + src.name).encode())
            compiler = tools / ('nvcc' if src.suffix == '.cu' else 'cxx')
            argv = ([str(compiler), '-O3', '-x', 'cu', '-dc', str(src), '-o', obj]
                    if src.suffix == '.cu' else [str(compiler), '-O3', '-o', obj, '-c', str(src)])
            commands.append({'directory': str(build), 'file': str(src), 'command': shlex.join(argv)})
            actual.append(shlex.join(argv + ['-MD', '-MT', obj, '-MF', obj + '.d']))
            objects.append(obj)
            log.append('0\t10\t0\t' + obj + '\tdeadbeef')
        dlink = 'CMakeFiles/gipc.dir/cmake_device_link.o'
        (build / dlink).write_bytes(b'device link object')
        actual.append(shlex.join([str(tools / 'nvcc'), '-dlink', *objects, '-o', dlink]))
        actual.append(': && ' + shlex.join([str(tools / 'cxx'), *objects, dlink, '-o', 'gipc', '-lcudart']) + ' && :')
        log += ['11\t12\t0\t' + dlink + '\tdeadbeef', '13\t14\t0\tgipc\tdeadbeef']
        (build / 'gipc').write_bytes(b'not executable; CPU fixture only')
        (build / 'compile_commands.json').write_text(json.dumps(commands))
        (build / '.ninja_log').write_text('\n'.join(log) + '\n')
        self.dumps[kind] = '\n'.join(actual)
        logs = self.root / 'build_logs' / kind
        logs.mkdir(parents=True)
        configure = ['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
                     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=89',
                     '-DCMAKE_CUDA_COMPILER=' + str(tools / 'nvcc')]
        (logs / 'configure.log').write_text('ARGV ' + json.dumps(configure) + '\nconfigured\n')
        (logs / 'build.log').write_text('ARGV ' + json.dumps(['cmake', '--build', str(build), '--parallel', '2']) + '\nlinked\n')
        object_ns = self.start_ns + (4 if kind == 'active' else 10) * 1_000_000_000
        for name in (*objects, dlink, 'gipc'):
            self.stamp(build / name, object_ns)
        for name in ('configure.log', 'build.log'):
            self.stamp(logs / name, object_ns + 2_000_000_000)

    @staticmethod
    def stamp(path, value_ns):
        os.utime(path, ns=(value_ns, value_ns))

    def reader(self, program, build):
        return {'argv': [str(program), '-C', str(build), '-t', 'commands', 'gipc'],
                'stdout': self.dumps[build.name], 'stderr': '', 'returncode': 0}

    def verify(self):
        return audit.verify(self.root, self.before, self.reader)

    def test_complete_object_and_link_identity(self):
        result = self.verify()
        self.assertTrue(result['passed'])
        self.assertTrue(result['sources_unchanged'])
        self.assertTrue(result['fresh_build_directory_prestate_recorded'])
        temporal = result['temporal_freshness']
        self.assertEqual(self.start_ns - temporal['before_file_mtime_ns'], 2_000_000_000)
        self.assertTrue(all(o['mtime_ns'] - self.start_ns >= 4_000_000_000
                            for b in temporal['builds'].values() for o in b['objects']))
        for kind, expected in audit.EXPECTED.items():
            b = result['builds'][kind]
            self.assertEqual(b['counts'], expected)
            self.assertEqual(len(b['objects']), expected['tu'])
            self.assertEqual(b['device_link']['input_object_count'], expected['tu'])

    def test_changed_source_or_asset_rejected(self):
        (self.root / 'Assets/input.bin').write_bytes(b'changed after compile')
        with self.assertRaisesRegex(ValueError, 'Prebuild/postbuild'):
            self.verify()

    def test_missing_real_object_rejected(self):
        (self.root / 'build/active/CMakeFiles/gipc.dir/StiffGIPC/unit_0.cu.o').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing evidence file'):
            self.verify()

    def test_duplicate_object_rejected(self):
        path = self.root / 'build/active/compile_commands.json'
        rows = json.loads(path.read_text())
        rows.append(copy.deepcopy(rows[0]))
        path.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, 'Duplicate TU or object'):
            self.verify()

    def test_host_link_missing_object_rejected(self):
        rows = self.dumps['active'].splitlines()
        rows[-1] = rows[-1].replace('CMakeFiles/gipc.dir/StiffGIPC/unit_0.cu.o ', '', 1)
        self.dumps['active'] = '\n'.join(rows)
        with self.assertRaisesRegex(ValueError, 'Host link object set'):
            self.verify()

    def test_device_link_missing_cuda_object_rejected(self):
        rows = self.dumps['active'].splitlines()
        rows[-2] = rows[-2].replace('CMakeFiles/gipc.dir/StiffGIPC/unit_0.cu.o ', '', 1)
        self.dumps['active'] = '\n'.join(rows)
        with self.assertRaisesRegex(ValueError, 'Device link misses CUDA'):
            self.verify()

    def test_compile_flags_differ_rejected(self):
        self.dumps['active'] = self.dumps['active'].replace('-O3', '-O0', 1)
        with self.assertRaisesRegex(ValueError, 'actual compile command differs'):
            self.verify()

    def test_cuda_device_c_long_alias_matches_actual_commands(self):
        path = self.root / 'build/active/compile_commands.json'
        rows = json.loads(path.read_text())
        for row in rows:
            row['command'] = row['command'].replace(' -dc ', ' --device-c ')
        path.write_text(json.dumps(rows))
        self.dumps['active'] = self.dumps['active'].replace(' -dc ', ' --device-c ')
        self.assertTrue(self.verify()['passed'])

    def test_cuda_device_c_wrong_source_rejected(self):
        path = self.root / 'build/active/compile_commands.json'
        rows = json.loads(path.read_text())
        row = next(r for r in rows if r['file'].endswith('.cu'))
        argv = shlex.split(row['command'])
        argv[argv.index('-dc') + 1] = str(self.root / 'StiffGIPC/wrong.cu')
        row['command'] = shlex.join(argv)
        path.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, 'Compile command source argument differs'):
            self.verify()

    def test_duplicate_or_missing_compile_markers_rejected(self):
        src = self.root / 'StiffGIPC/unit_0.cu'
        for markers in (('-dc', '-dc'), ('-dc', '-c'), ('-dc', '--device-c'), ()):
            with self.subTest(markers=markers):
                argv = ['nvcc'] + [word for flag in markers for word in (flag, str(src))]
                with self.assertRaisesRegex(ValueError, 'exactly one explicit compile source marker'):
                    audit.require_compile_source(argv, self.root, src)

    def test_device_compile_marker_on_cpp_rejected(self):
        src = self.root / 'StiffGIPC/unit_0.cpp'
        with self.assertRaisesRegex(ValueError, 'non-CUDA source'):
            audit.require_compile_source(['nvcc', '-dc', str(src)], self.root, src)

    def test_missing_actual_execution_rejected(self):
        path = self.root / 'build/active/.ninja_log'
        path.write_text('\n'.join(line for line in path.read_text().splitlines() if 'unit_0.cu.o' not in line))
        with self.assertRaisesRegex(ValueError, 'actual Ninja command/execution'):
            self.verify()

    def test_common_source_record_format_and_missing_root(self):
        packet = json.loads(self.before.read_text())
        rows = [{'relative_path': r['path'], 'size': r['bytes'], 'sha256': r['sha256']}
                for r in packet['sources']]
        self.before.write_text(json.dumps({'records': rows}))
        self.stamp(self.before, self.start_ns - 2_000_000_000)
        self.assertFalse(self.verify()['fresh_build_directory_prestate_recorded'])
        self.before.write_text(json.dumps({'records': [r for r in rows if not r['relative_path'].startswith('Assets/')]}))
        with self.assertRaisesRegex(ValueError, 'Prebuild/postbuild'):
            self.verify()

    def test_snapshot_rejects_existing_build(self):
        with self.assertRaisesRegex(ValueError, 'precede creation'):
            audit.snapshot(self.root)

    def test_missing_response_file_fail_closed(self):
        rows = self.dumps['active'].splitlines()
        rows[-1] = rows[-1].replace('-lcudart', '@missing.rsp')
        self.dumps['active'] = '\n'.join(rows)
        with self.assertRaisesRegex(ValueError, 'Missing response file'):
            self.verify()

    def test_wrong_count_not_silently_skipped(self):
        (self.root / 'StiffGIPC/extra.cu').write_text('// added before snapshot')
        packet = json.loads(self.before.read_text())
        packet['sources'] = audit.source_inventory(self.root)
        self.before.write_text(json.dumps(packet))
        with self.assertRaisesRegex(ValueError, 'source counts differ'):
            self.verify()

    def test_attachment_recheck_rejects_object_change(self):
        packet = self.verify()
        path = self.root / 'identity.json'
        path.write_text(json.dumps(packet))
        bound = audit.sha(path)
        self.assertTrue(audit.verify_identity(self.root, path, bound)['passed'])
        (self.root / packet['builds']['base']['objects'][0]['object']['path']).write_bytes(b'new object')
        with self.assertRaisesRegex(ValueError, 'Sealed build identity changed'):
            audit.verify_identity(self.root, path, bound)

    def test_source_changed_during_ninja_inspection_rejected(self):
        def reader(program, build):
            if build.name == 'active':
                (self.root / 'StiffGIPC/unit_0.cu').write_text('// changed while auditing')
            return self.reader(program, build)
        with self.assertRaisesRegex(ValueError, 'Source changed during verification'):
            audit.verify(self.root, self.before, reader)

    def test_build_log_wrong_directory_rejected(self):
        log = self.root / 'build_logs/active/build.log'
        log.write_text('ARGV ' + json.dumps(['cmake', '--build', str(self.root / 'build/base')]) + '\n')
        with self.assertRaisesRegex(ValueError, 'Build log targets a different'):
            self.verify()

    def test_exact_eight_unused_baseline_caches_recorded(self):
        for name in audit.UNUSED_BASE_CACHES:
            (self.root / 'baseline/Assets' / name).write_bytes(b'historical unused cache')
        result = audit.assets_equivalence(self.root, audit.source_inventory(self.root))
        self.assertTrue(result['passed'])
        self.assertFalse(result['roots_identical'])
        self.assertEqual({r['path'] for r in result['ignored_baseline_only']}, audit.UNUSED_BASE_CACHES)
        self.assertEqual(len(result['selected_scene_references']), 3)

    def test_unapproved_baseline_cache_rejected(self):
        (self.root / 'baseline/Assets/sorted_mesh/unknown.part').write_bytes(b'unknown')
        with self.assertRaisesRegex(ValueError, 'Unapproved baseline-only'):
            audit.assets_equivalence(self.root, audit.source_inventory(self.root))

    def test_whitelisted_but_selected_cache_not_ignored(self):
        for prefix in ('', 'baseline/'):
            assets = self.root / prefix / 'Assets'
            mesh = assets / 'benchmark_meshes/cube.msh'
            mesh.write_bytes(b'selected cube')
            for name in ('cube_sorted.16.msh', 'cube_sorted.16.part'):
                if prefix:
                    (assets / 'sorted_mesh' / name).write_bytes(b'now referenced cache')
            scene = assets / 'benchmark_scenes' / (audit.SELECTED_SCENES[0] + '.json')
            scene.write_text(json.dumps({'case_id': audit.SELECTED_SCENES[0], 'objects': [{
                'stiff_mesh': 'benchmark_meshes/cube.msh', 'stiff_mesh_sha256': audit.sha(mesh)}]}))
        with self.assertRaisesRegex(ValueError, 'Selected sorted-mesh/cache missing'):
            audit.assets_equivalence(self.root, audit.source_inventory(self.root))

    def test_common_assets_byte_difference_rejected(self):
        (self.root / 'baseline/Assets/input.bin').write_bytes(b'different')
        with self.assertRaisesRegex(ValueError, 'Common Assets differ'):
            audit.assets_equivalence(self.root, audit.source_inventory(self.root))

    def test_postbuild_snapshot_timestamp_rejected(self):
        self.stamp(self.before, self.start_ns + 20_000_000_000)
        with self.assertRaisesRegex(ValueError, 'not written before build supervisor'):
            self.verify()

    def test_stale_actual_object_timestamp_rejected(self):
        self.stamp(self.root / 'build/active/CMakeFiles/gipc.dir/StiffGIPC/unit_0.cu.o',
                   self.start_ns - 2_000_000_000)
        with self.assertRaisesRegex(ValueError, 'Actual object predates current build supervisor'):
            self.verify()

    def test_stale_build_log_timestamp_rejected(self):
        self.stamp(self.root / 'build_logs/base/build.log', self.start_ns - 2_000_000_000)
        with self.assertRaisesRegex(ValueError, 'Build command log predates supervisor start'):
            self.verify()

    def test_missing_absolute_build_start_rejected(self):
        (self.root / 'build_start.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing evidence file'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
