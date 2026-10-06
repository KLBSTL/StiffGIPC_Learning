"""Small CPU contracts; no solver, compiler, SSH, GPU, or real-run changes."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import runner
from plan import plan, tasks
from config import expand, environment


class ControllerContracts(unittest.TestCase):
    def test_finite_order_inputs_and_pair_counts(self):
        declared = tasks()
        self.assertEqual(len(declared), 78)
        self.assertEqual(len({t['name'] for t in declared}), 78)
        self.assertTrue(all(t['phase'] in ('calibration','verification') for t in declared[:15]))
        self.assertTrue(all(t['phase']=='stability' for t in declared[75:]))
        for scene in ('hang','fixed','mixed'):
            selected=[t for t in declared if t['scene_key']==scene]
            self.assertEqual(len(selected),26)
            self.assertEqual(sum(t['phase']=='calibration' for t in selected),3)
            self.assertEqual(sum(t['phase']=='verification' for t in selected),2)
            for arm, count in [('original_stiff',7),('combined_graph',7),('ipc_host',3),('ipc_graph',3)]:
                self.assertEqual(sum(t['phase']=='paired' and t['arm']==arm for t in selected),count)
        for t in declared:
            c=expand(t['config']); env=environment(c,Path('fixture'))
            self.assertEqual((c['dt'],c['ipc_newton_tol'],c['ipc_min_updates'],c['pcg_rho_tol']),(.01,.01,6,1e-4))
            self.assertEqual(c['timeout_seconds'],600 if t['phase']=='stability' else 120)
            self.assertEqual(c['pcg_graph_chunk'],1)
            self.assertEqual(c['diagnostics'],[]);self.assertEqual(c['profile'],'none')
            self.assertFalse(c['contact_pool']);self.assertEqual(c['edge_query_order'],'raw')
            self.assertNotIn('GIPC_COST_TRACE',env);self.assertNotIn('GIPC_FIXED_STUDY_DIR',env)
            self.assertEqual(c['steps'],300 if t['phase']=='stability' else 100)
            combined=t['arm']=='combined_graph'
            self.assertEqual((c['refit'],c['batch'],c['reuse'],c['discrete_bvh_refit']),(combined,)*4)

    def test_pair_order_alternates_without_calibration_pollution(self):
        for pair in range(1,8):
            arms=[t['arm'] for t in tasks() if t['scene_key']=='hang' and t['pair']==pair]
            main=['original_stiff','combined_graph'] if pair%2 else ['combined_graph','original_stiff']
            self.assertEqual(arms[:2],main)
            self.assertEqual(arms[2:],(['ipc_host','ipc_graph'] if pair%2 else ['ipc_graph','ipc_host']) if pair<=3 else [])

    def test_object_link_attachment_must_match_both_executables(self):
        manifests={kind:{'exe_path':f'build/{kind}/gipc','exe_sha256':kind+'sha'} for kind in ('active','base')}
        packet={'builds':{kind:{'exe':{'path':m['exe_path'],'sha256':m['exe_sha256']}} for kind,m in manifests.items()}}
        verifier=SimpleNamespace(verify_identity=lambda *_:packet)
        with tempfile.TemporaryDirectory() as name,patch.dict('sys.modules',{'build_identity':verifier}):
            root=Path(name);attachment={'path':'identity.json','sha256':'boundsha'}
            runner.verify_build_attachment(root,attachment,manifests['active'],manifests['base'])
            packet['builds']['base']['exe']['sha256']='wrong'
            with self.assertRaises(ValueError):
                runner.verify_build_attachment(root,attachment,manifests['active'],manifests['base'])

    def baseline_packet(self,root):
        import build_identity as audit
        for folder in audit.SOURCE_DIRS:(root/folder).mkdir(parents=True)
        for file in audit.SOURCE_FILES:(root/file).write_text('project(fixture)')
        for prefix in ('','baseline/'):
            assets=root/prefix/'Assets'
            for folder in ('benchmark_scenes','benchmark_meshes','sorted_mesh'):(assets/folder).mkdir()
            mesh=assets/'benchmark_meshes/current.obj';mesh.write_bytes(b'common mesh')
            for suffix in ('obj','part'):(assets/f'sorted_mesh/current_sorted.16.{suffix}').write_bytes(b'common cache')
            for case in audit.SELECTED_SCENES:
                runner.write_new(assets/f'benchmark_scenes/{case}.json',{'case_id':case,'objects':[
                    {'stiff_mesh':'benchmark_meshes/current.obj','stiff_mesh_sha256':runner.sha(mesh)}]})
        for path in audit.UNUSED_BASE_CACHES:(root/'baseline/Assets'/path).write_bytes(b'unused historical cache')
        source=root/'baseline/StiffGIPC';(source/'core').mkdir()
        (source/'core/GIPC.cu').write_text('int Kmin = 6; if(beta <= Newton_solver_threshold) {}')
        for i in range(30):(source/f'unit{i}.cu').write_text('// CUDA fixture')
        for i in range(4):(source/f'unit{i}.cpp').write_text('// C++ fixture')
        build=root/'build/base';build.mkdir(parents=True)
        (root/'compiler').write_bytes(b'CPU fixture compiler')
        (build/'CMakeCache.txt').write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={root/"baseline"}\n'
            f'CMAKE_CXX_COMPILER:FILEPATH={root/"compiler"}\nCMAKE_CUDA_COMPILER:FILEPATH={root/"compiler"}\n')
        (build/'compile_commands.json').write_text('[]');(build/'gipc').write_bytes(b'CPU fixture executable')
        log=root/'build_logs/base/build.log';log.parent.mkdir(parents=True);log.write_text('actual build fixture')
        objects=[{'source':p.relative_to(root).as_posix(),'actual_argv':['nvcc',
            '-DGIPC_ASSETS_DIR="'+str(root/'baseline/Assets')+'/"']} for p in source.rglob('*') if p.suffix in ('.cu','.cpp')]
        sources=audit.source_inventory(root)
        return {'sources':sources,'assets_equivalence':audit.assets_equivalence(root,sources),
            'builds':{'base':{'counts':{'tu':35,'cuda':31,'cxx':4},'objects':objects,
                'exe':audit.file_record(root,build/'gipc'),
                'build_evidence':runner.inventory(root,[build/'CMakeCache.txt',build/'compile_commands.json']),
                'build_logs':{'build':audit.file_record(root,log)}}}}

    def test_baseline_adapter_retains_exact_eight_extra_identities(self):
        import build_identity as audit
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);packet=self.baseline_packet(root)
            baseline=runner.baseline_identity(root,'build_logs/base/build.log',packet)
            self.assertEqual(baseline['compile_source_count'],35)
            paths={r['path'] for r in baseline['sources']}
            self.assertTrue({'baseline/Assets/'+p for p in audit.UNUSED_BASE_CACHES} <= paths)
            self.assertEqual(baseline['stopping_source_evidence']['requested_tolerance'],.01)
            self.assertTrue(baseline['compiler_identity']['complete'])
            packet['builds']['base']['objects'][0]['actual_argv']=['nvcc','-DGIPC_ASSETS_DIR="'+str(root/'Assets')+'"']
            with self.assertRaises(ValueError):runner.baseline_identity(root,'build_logs/base/build.log',packet)

    def test_baseline_adapter_rejects_ninth_extra_or_changed_common(self):
        import build_identity as audit
        for mode in ('ninth','common'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as name:
                root=Path(name);packet=self.baseline_packet(root)
                if mode=='ninth':(root/'baseline/Assets/sorted_mesh/unknown.obj').write_bytes(b'not approved')
                else:(root/'baseline/Assets/benchmark_meshes/current.obj').write_bytes(b'changed common mesh')
                packet['sources']=audit.source_inventory(root)
                with self.assertRaises(ValueError):runner.baseline_identity(root,'build_logs/base/build.log',packet)

    def prepare(self, root):
        (root/'seal.json').write_text('{}')
        value={'plan':plan(),'plan_sha256':runner.digest(plan()),'active_manifest':{},'base_manifest':{}}
        with patch.object(runner,'verify_seal',return_value=value):runner.initialize(root,'seal.json','runs/test')
        return value

    @staticmethod
    def execute(root, session, task, manifest, gpu):
        run=session/task['name'];run.mkdir()
        result={'status':'completed','recorded_frames':expand(task['config'])['steps'],'wall_seconds':1.}
        runner.write_new(run/'result.json',result)
        runner.write_new(run/'process.json',{'pid':123})
        runner.write_new(run/'evidence.json',{'files':runner.inventory(run,[run/'result.json',run/'process.json'])})
        return result

    def advance(self,root,value,check=None,execute=None,summary=None):
        fake=SimpleNamespace(analyze_run=lambda *_:check or {'hard_checks_passed':True,'quality_status':'outside_frozen_range','failures':[]},
                             analyze_session=lambda *_:summary or {'quality_certified':False,'long_run_gates':{}})
        with patch.object(runner,'verify_seal',return_value=value),patch.object(runner,'gpu_lock',return_value=nullcontext()),\
             patch.object(runner,'execute_base',side_effect=execute or self.execute),\
             patch.object(runner,'execute_active',side_effect=execute or self.execute),patch.object(runner,'analyzer',return_value=fake):
            return runner.advance(root,'seal.json','runs/test')

    def test_one_step_quality_difference_continues_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);value=self.prepare(root)
            first=self.advance(root,value);second=self.advance(root,value)
            self.assertEqual((first['index'],second['index']),(1,2))
            self.assertEqual(first['status'],'completed')
            entries=runner.read_entries(root/'runs/test',root)
            self.assertEqual(entries[0]['analysis']['quality_status'],'outside_frozen_range')
            with patch.object(runner,'verify_seal',return_value=value),self.assertRaises(ValueError):
                runner.initialize(root,'seal.json','runs/test')
            (root/'runs/test'/tasks()[2]['name']).mkdir()
            with self.assertRaises(ValueError):self.advance(root,value)

    def test_hard_failure_skips_same_config_and_preserves_analysis(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);value=self.prepare(root)
            failed=self.advance(root,value,{'hard_checks_passed':False,'quality_status':'failed','failures':['PCG breakdown']})
            self.assertEqual(failed['status'],'hard_failed')
            with patch.object(self,'execute',side_effect=AssertionError('Must not launch blocked config')):
                skipped=self.advance(root,value)
            self.assertEqual(skipped['status'],'skipped');self.assertFalse(skipped['gpu_launched'])
            self.assertEqual(len(runner.read_entries(root/'runs/test',root)),2)

    def test_absolute_deadline_and_analysis_exception_stop(self):
        def late(*args):
            result=self.execute(*args);result['wall_seconds']=120.000001
            (args[1]/args[2]['name']/'result.json').write_text(__import__('json').dumps(result))
            run=args[1]/args[2]['name'];(run/'evidence.json').write_text(__import__('json').dumps({'files':runner.inventory(run,[run/'result.json',run/'process.json'])}))
            return result
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);value=self.prepare(root)
            self.assertEqual(self.advance(root,value,execute=late)['status'],'hard_failed')
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);value=self.prepare(root)
            self.assertTrue(self.advance(root,value,check={'hard_checks_passed':'false'})['analysis_exception'])
            with self.assertRaises(ValueError):self.advance(root,value)

    def test_metadata_chain_tamper_rejected_before_next_launch(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);value=self.prepare(root);self.advance(root,value)
            entries=runner.read_entries(root/'runs/test',root)
            (root/'runs/test'/entries[0]['analysis_path']).write_text('{}')
            with self.assertRaises(ValueError):self.advance(root,value)

    def test_real_analyzer_accepts_failed_run_prefix_without_new_gpu(self):
        # The synthetic runner intentionally lacks native request/state exports.
        # Real CPU analysis must record a hard failure and produce its summary,
        # not raise an interface error or invent a passing physical result.
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);value=self.prepare(root)
            with patch.object(runner,'verify_seal',return_value=value),\
                 patch.object(runner,'gpu_lock',return_value=nullcontext()),\
                 patch.object(runner,'execute_base',side_effect=self.execute):
                result=runner.advance(root,'seal.json','runs/test')
            self.assertEqual(result['status'],'hard_failed')
            self.assertFalse(result['analysis_exception'])
            entry=runner.read_entries(root/'runs/test',root)[0]
            self.assertFalse(entry['analysis']['hard_checks_passed'])
            summary=runner.read(root/'runs/test'/entry['summary_path'])
            self.assertEqual(summary['recorded_tasks'],1)
            self.assertFalse(summary['quality_certified'])

    def test_stability_requires_all_25_and_explicit_quality_gate(self):
        stability=next(t for t in tasks() if t['scene_key']=='hang' and t['phase']=='stability')
        complete=[{'task':t,'status':'completed','analysis':{'hard_checks_passed':True}}
                  for t in tasks() if t['scene_key']=='hang' and t['phase']!='stability']
        self.assertIsNotNone(runner.skip_reason(stability,complete,{}))
        gate={'long_run_gates':{'hang':{'eligible':True,'reasons':[]}}}
        self.assertIsNone(runner.skip_reason(stability,complete,gate))
        self.assertIsNotNone(runner.skip_reason(stability,complete[:-1],gate))
        complete[0]['status']='hard_failed'
        self.assertIsNotNone(runner.skip_reason(stability,complete,gate))

    def test_missing_raw_requires_bound_verified_archive_and_metadata(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);run=root/'runs/test/run';(run/'trace').mkdir(parents=True)
            (run/'trace/state_0000.bin').write_bytes(b'raw');runner.write_new(run/'result.json',{'status':'completed'})
            runner.write_new(run/'evidence.json',{'files':runner.inventory(run,[run/'trace/state_0000.bin',run/'result.json'])})
            evidence_sha=runner.sha(run/'evidence.json');(run/'trace/state_0000.bin').unlink()
            with self.assertRaises(ValueError):runner.verify_retained_evidence(root,run,evidence_sha)

            archive=root/'archive.tar.gz';archive.write_bytes(b'archivefixture')
            receipt={'schema':'full_eval_raw_archive.v1','archive_path':'archive.tar.gz','archive_sha256':runner.sha(archive),
                     'archive_bytes':archive.stat().st_size,'evidence_sha256':evidence_sha,'download_verified':True,
                     'local_archive_name':'archive.tar.gz','removed_raw_paths':['trace/state_0000.bin']}
            runner.write_new(run/'archive_receipt.json',receipt)
            runner.verify_retained_evidence(root,run,evidence_sha)
            for override in ({'download_verified':False},{'archive_sha256':'bad'},
                             {'removed_raw_paths':['../outside.bin']},{'archive_path':'../outside.tar.gz'}):
                (run/'archive_receipt.json').write_text(__import__('json').dumps(receipt|override))
                with self.assertRaises(ValueError):runner.verify_retained_evidence(root,run,evidence_sha)
            (run/'archive_receipt.json').write_text(__import__('json').dumps(receipt))
            (run/'result.json').unlink()
            with self.assertRaises(ValueError):runner.verify_retained_evidence(root,run,evidence_sha)

    def test_unarchived_same_size_raw_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);run=root/'runs/test/run';(run/'trace').mkdir(parents=True)
            raw=run/'trace/state_0000.bin';raw.write_bytes(b'first')
            runner.write_new(run/'evidence.json',{'files':runner.inventory(run,[raw])})
            evidence_sha=runner.sha(run/'evidence.json')
            runner.verify_retained_evidence(root,run,evidence_sha)
            raw.write_bytes(b'other')
            with self.assertRaises(ValueError):runner.verify_retained_evidence(root,run,evidence_sha)


if __name__=='__main__':unittest.main()
