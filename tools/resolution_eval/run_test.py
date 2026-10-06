"""Small CPU-only controller/input contracts. Never invokes a native program."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run


class ResolutionContracts(unittest.TestCase):
    def test_offline_report_does_not_change_controller_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);folder=root/'tools/resolution_eval';folder.mkdir(parents=True)
            for file in ('run.py','run_test.py'):(folder/file).write_text('sealed controller')
            before=run.tools_identity(root)
            (folder/'report.py').write_text('new offline report');(folder/'report_test.py').write_text('offline tests')
            self.assertEqual(run.tools_identity(root),before)
            (folder/'run.py').write_text('changed execution controller')
            self.assertNotEqual(run.tools_identity(root),before)

    def test_exact_sixteen_inputs_and_alternating_order(self):
        rows=run.tasks();self.assertEqual(len(rows),16)
        self.assertEqual(len({r['name'] for r in rows}),16)
        for i,key in enumerate(run.CASES):
            selected=[t for t in rows if t['scene_key']==key]
            self.assertEqual([t['arm'] for t in selected],list(run.ARMS if i%2==0 else reversed(run.ARMS)))
            for t in selected:
                c=run.expand(t['config']);self.assertEqual(t['name'],f'p1_{key}_{t["arm"]}')
                self.assertEqual((c['steps'],c['timeout_seconds'],c['dt'],c['ipc_newton_tol'],c['ipc_min_updates'],c['pcg_rho_tol']),
                                 (100,120,.01,.01,6,1e-4))
                self.assertEqual(c['pcg_graph_chunk'],1);self.assertEqual(c['diagnostics'],[])
                self.assertEqual(c['profile'],'none');self.assertFalse(c['contact_pool'])
                self.assertEqual(c['edge_query_order'],'raw');self.assertEqual(c['trace_velocity'],t['binary']=='active')
                combined=t['arm']=='combined_graph'
                self.assertEqual((c['refit'],c['batch'],c['reuse'],c['discrete_bvh_refit']),(combined,)*4)

    def assets(self,root):
        for relative in run.builds.SOURCE_DIRS:(root/relative).mkdir(parents=True)
        for relative in run.builds.SOURCE_FILES:(root/relative).write_text('unchanged source')
        for prefix in ('','baseline/'):
            assets=root/prefix/'Assets'
            for folder in ('benchmark_scenes','benchmark_meshes','sorted_mesh'):(assets/folder).mkdir()
            for level in ('l','m'):
                mesh=assets/f'benchmark_meshes/cloth_{level}.obj';mesh.write_bytes(('original '+level).encode())
                if level=='l':
                    for suffix in ('obj','part'):(assets/f'sorted_mesh/cloth_l_sorted.16.{suffix}').write_bytes(b'existing cache')
            for key,case in run.CASES.items():
                name=f'benchmark_meshes/cloth_{key[-1]}.obj'
                run.write_new(assets/f'benchmark_scenes/{case}.json',{'case_id':case,'objects':[{'dimension':2,'body_type':'FEM','stiff_mesh':name,'stiff_mesh_sha256':run.sha(assets/name)}]})
        for relative in run.builds.UNUSED_BASE_CACHES:(root/'baseline/Assets'/relative).write_bytes(b'old unused baseline cache')
        old=run.builds.source_inventory(root)
        for prefix in ('','baseline/'):
            for suffix in ('obj','part'):(root/prefix/f'Assets/sorted_mesh/cloth_m_sorted.16.{suffix}').write_bytes(b'new shared runtime cache')
        return old

    def test_additive_inputs_pass_without_rewriting_old_records(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);old=self.assets(root);saved=copy.deepcopy(old)
            proof=run.asset_proof(root,old,run.builds.source_inventory(root))
            self.assertTrue(proof['passed']);self.assertEqual(old,saved)
            self.assertEqual(len(proof['added_files']),4)
            self.assertEqual(len(proof['selected_scene_references']),4)

    def test_mutation_new_source_and_asymmetric_cache_rejected(self):
        for fault in ('old_source','unexpected_cache','new_source','asymmetric','missing_cache'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as name:
                root=Path(name);old=self.assets(root)
                if fault=='old_source':(root/'CMakeLists.txt').write_text('changed existing source')
                elif fault=='new_source':(root/'StiffGIPC/new.cu').write_text('new source forbidden')
                elif fault=='unexpected_cache':(root/'Assets/sorted_mesh/unreferenced.part').write_bytes(b'extra')
                elif fault=='asymmetric':(root/'baseline/Assets/sorted_mesh/cloth_m_sorted.16.part').write_bytes(b'different')
                else:(root/'baseline/Assets/sorted_mesh/cloth_m_sorted.16.part').unlink()
                with self.assertRaises(ValueError):run.asset_proof(root,old,run.builds.source_inventory(root))

    def test_excluded_baseline_cache_cannot_become_selected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.assets(root)
            # Construct an unchanged-parent fixture referencing a reviewed unused
            # cache name. The new four-scene plan must still reject that use.
            for prefix in ('','baseline/'):
                assets=root/prefix/'Assets';mesh=assets/'benchmark_meshes/cipc_table.obj';mesh.write_bytes(b'mesh')
                for suffix in ('obj','part'):(assets/f'sorted_mesh/cipc_table_sorted.16.{suffix}').write_bytes(b'same cache')
                (assets/'benchmark_scenes/cloth_hang_l.json').write_text(json.dumps({'case_id':'cloth_hang_l','objects':[
                    {'dimension':2,'body_type':'FEM','stiff_mesh':'benchmark_meshes/cipc_table.obj','stiff_mesh_sha256':run.sha(mesh)}]}))
            current=run.builds.source_inventory(root)
            with self.assertRaises(ValueError):run.asset_proof(root,current,current)

    def test_3d_abd_needs_no_cache_but_fem_does(self):
        for body in ('ABD','FEM'):
            with self.subTest(body=body),tempfile.TemporaryDirectory() as name:
                root=Path(name);self.assets(root)
                for prefix in ('','baseline/'):
                    assets=root/prefix/'Assets';mesh=assets/'benchmark_meshes/ball.msh';mesh.write_bytes(b'identical tetrahedral mesh')
                    scene_path=assets/'benchmark_scenes/cloth_sphere7_l.json';scene=run.read(scene_path)
                    scene['objects'].append({'dimension':3,'body_type':body,'stiff_mesh':'benchmark_meshes/ball.msh',
                                             'stiff_mesh_sha256':run.sha(mesh)})
                    scene_path.write_text(json.dumps(scene))
                current=run.builds.source_inventory(root)
                if body=='FEM':
                    with self.assertRaises(ValueError):run.asset_proof(root,current,current)
                else:
                    proof=run.asset_proof(root,current,current)
                    sphere=next(s for s in proof['selected_scene_references'] if s['scene']=='cloth_sphere7_l')
                    self.assertEqual(sphere['meshes'][-1]['caches'],[])
                    self.assertFalse(sphere['meshes'][-1]['cache_required'])

    def test_parent_build_proof_reused_and_only_sources_digest_extended(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);old=self.assets(root)
            active_old=[r for r in old if not r['path'].startswith('baseline/')]
            base_old=[r for r in old if r['path'].startswith('baseline/')]
            manifests={}
            for kind,sources in (('active',active_old),('base',base_old)):
                exe=root/f'build/{kind}/gipc';exe.parent.mkdir(parents=True);exe.write_bytes(kind.encode())
                manifests[kind+'_manifest']={'sources':sources,'source_digest':run.digest(sources),
                    'build_evidence':[],'exe_path':exe.relative_to(root).as_posix(),'exe_sha256':run.sha(exe)}
            packet={'schema':'full_eval_linux_build_identity.v1','passed':True,'sources_unchanged':True,
                    'temporal_freshness':{'passed':True},'assets_equivalence':{'passed':True},'sources':old}
            run.write_new(root/'build.json',packet);run.write_new(root/'active.json',manifests['active_manifest'])
            parent={'schema':'full_eval_seal.v1','plan':{},'plan_sha256':run.digest({}),
                    'controller_files':[],'controller_sha256':run.digest([]),
                    'active_manifest_path':'active.json','active_manifest_sha256':run.sha(root/'active.json'),
                    'build_identity_attachment':{'path':'build.json','sha256':run.sha(root/'build.json')},**manifests}
            run.write_new(root/'parent.json',parent)
            active_current=[r for r in run.builds.source_inventory(root) if not r['path'].startswith('baseline/')]
            with patch.object(run.old_runner,'controller_files',return_value=[]),\
                 patch.object(run.builds,'_verify_identity_files') as verify_old,\
                 patch.object(run.old_runner,'match_build_programs'),\
                 patch.object(run.linux,'source_inventory',return_value=active_current):
                extended,proof=run.parent_inputs(root,root/'parent.json')
                verify_old.assert_called_once_with(root,packet)
            self.assertEqual(len(proof['added_files']),4)
            for key,original in manifests.items():
                new=extended[key]
                self.assertEqual(new['exe_sha256'],original['exe_sha256'])
                self.assertEqual(len(new['sources']),len(original['sources'])+2)
                self.assertNotEqual(new['source_digest'],original['source_digest'])
                self.assertIs(new['runtime_input_supplement']['recompiled'],False)

    def prepare(self,root):
        (root/'seal.json').write_text('{}')
        value={'plan':run.plan(),'active_manifest':{},'base_manifest':{}}
        with patch.object(run,'verify_seal',return_value=value):run.initialize(root,'seal.json','runs/resolution')
        return value

    @staticmethod
    def execute(root,session,task,manifest,gpu):
        out=session/task['name'];out.mkdir()
        result={'status':'completed','wall_seconds':1.,'recorded_frames':100}
        run.write_new(out/'result.json',result);run.write_new(out/'process.json',{'pid':111})
        run.write_new(out/'evidence.json',{'files':run.inventory(out,[out/'result.json',out/'process.json'])})
        return result

    def advance(self,root,value,hard=True,execute=None):
        with patch.object(run,'verify_seal',return_value=value),patch.object(run.linux,'gpu_lock',return_value=nullcontext()),\
             patch.object(run,'execute_base',side_effect=execute or self.execute),\
             patch.object(run.linux,'execute',side_effect=execute or self.execute),\
             patch.object(run.metrics,'analyze_run',return_value={'hard_checks_passed':hard,'quality_status':'single_sample','failures':[]}),\
             patch.object(run,'summarize',return_value={'quality_certified':False,'performance_certified':False}):
            return run.advance(root,'seal.json','runs/resolution')

    def test_one_run_then_hard_failure_skips_only_same_case(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);value=self.prepare(root)
            first=self.advance(root,value,hard=False)
            self.assertEqual(first['status'],'hard_failed')
            for index in (2,3,4):
                skipped=self.advance(root,value);self.assertEqual(skipped['index'],index)
                self.assertEqual(skipped['status'],'skipped');self.assertFalse(skipped['gpu_launched'])
            self.assertEqual(self.advance(root,value)['status'],'completed')
            self.assertEqual(len(run.entries(root,root/'runs/resolution')),5)

    def test_no_retry_metadata_tamper_and_existing_session(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);value=self.prepare(root);self.advance(root,value)
            first=run.entries(root,root/'runs/resolution')[0]
            (root/'runs/resolution'/first['analysis_path']).write_text('{}')
            with self.assertRaises(ValueError):self.advance(root,value)
            with patch.object(run,'verify_seal',return_value=value),self.assertRaises(ValueError):
                run.initialize(root,'seal.json','runs/resolution')
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);value=self.prepare(root)
            (root/'runs/resolution'/run.tasks()[0]['name']).mkdir()
            with self.assertRaises(ValueError):self.advance(root,value)

    def test_completed_beyond_deadline_is_hard_failure(self):
        def late(*args):
            result=self.execute(*args);result['wall_seconds']=120.01
            out=args[1]/args[2]['name'];(out/'result.json').write_text(json.dumps(result))
            (out/'evidence.json').write_text(json.dumps({'files':run.inventory(out,[out/'result.json',out/'process.json'])}))
            return result
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);value=self.prepare(root)
            self.assertEqual(self.advance(root,value,execute=late)['status'],'hard_failed')

    def test_extra_original_host_state_edge_records_unavailable_truthfully(self):
        with tempfile.TemporaryDirectory() as name:
            session=Path(name);task=next(t for t in run.tasks() if t['name']=='p1_hang_l_ipc_host')
            for arm in ('original_stiff','ipc_host'):
                folder=session/f'p1_hang_l_{arm}';folder.mkdir()
                run.write_new(folder/'evidence.json',{'files':[]})
            run.write_new(session/'analysis/p1_hang_l_original_stiff.json',
                          {'hard_checks_passed':True,'input_identity':{},'actual_velocity':{'available':False}})
            row={'hard_checks_passed':True,'input_identity':{},'actual_velocity':{'available':True},'paired_comparisons':[]}
            with patch.object(run.metrics,'compare_inputs',return_value={'passed':True}),\
                 patch.object(run.metrics,'state_comparison',side_effect=AssertionError('No raw states available')):
                run.stiff_host_comparison(session,task,row)
            pair=row['paired_comparisons'][0]
            self.assertEqual((pair['left'],pair['right']),('original_stiff','ipc_host'))
            self.assertIs(pair['state_difference']['available'],False)
            self.assertEqual(run.sha(session/pair['pair_cache_path']),pair['pair_cache_sha256'])


if __name__=='__main__':unittest.main()
