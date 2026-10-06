"""Offline synthetic contracts. Does not launch a solver, GPU, compiler or SSH."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from local_plan import ROOT,tasks,plan,predecessor
from local_identity import project_map,active_identity,record,verify,assets_equivalence,UNUSED_BASE_CACHES
from local_analysis import analyze_one,analyze_round
import windows_runner as runner
from quality_contracts_test import fixture as synthetic_run
from config import expand
from linux_runner import sha,inventory,write_new

def dump(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf-8')

class Contracts(unittest.TestCase):
    def test_plan_finite_and_rounds(self):
        self.assertEqual(sum(len(x) for x in plan()['rounds'].values()),13)
        fixed=[x['variant'] for s in ('fixed_r1','fixed_r2','fixed_r3') for x in tasks(s)]
        self.assertEqual(fixed,['off','base','base','off','on','off','base'])
        self.assertEqual([x['variant'] for x in tasks('hang_r2')],['on','off'])
        for group in plan()['rounds'].values():
            for t in group:
                c=expand(t['config']);self.assertEqual((c['dt'],c['ipc_min_updates'],c['pcg_rho_tol'],c['timeout_seconds']),(.01,6,1e-4,120))
                self.assertFalse(c['contact_pool_validate']);self.assertEqual(c['diagnostics'],[])
        self.assertFalse(plan()['automatic_next_round']);self.assertFalse(plan()['quality_certified'])
        self.assertEqual(predecessor('hang_r3'),'hang_r2')

    def test_deadline_is_absolute(self):
        with patch.object(runner.time,'monotonic',return_value=100):
            self.assertEqual(runner.remaining(101),1)
            with self.assertRaises(runner.subprocess.TimeoutExpired):runner.remaining(100)

    def test_only_eight_unreferenced_baseline_caches_allowed(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);a=root/'active';b=root/'base'
            for folder in (a,b):
                (folder/'benchmark_meshes').mkdir(parents=True);(folder/'sorted_mesh').mkdir()
                (folder/'benchmark_meshes/current.obj').write_bytes(b'current mesh')
                for suffix in ('obj','part'):(folder/('sorted_mesh/current_sorted.16.'+suffix)).write_bytes(b'cache')
                for case in ('cloth_hang_l','cloth_fixed_bunny_l'):
                    dump(folder/('benchmark_scenes/'+case+'.json'),{'case_id':case,'objects':[{'stiff_mesh':'benchmark_meshes/current.obj','stiff_mesh_sha256':sha(folder/'benchmark_meshes/current.obj')}]})
            for relative in UNUSED_BASE_CACHES:(b/relative).write_bytes(b'unused historical cache')
            result=assets_equivalence(a,b);self.assertEqual(len(result['ignored_baseline_only']),8)
            self.assertTrue(all(r['sha256']==sha(b/r['path']) for r in result['ignored_baseline_only']))
            (b/'sorted_mesh/other.part').write_bytes(b'unknown')
            with self.assertRaises(ValueError):assets_equivalence(a,b)
            (b/'sorted_mesh/other.part').unlink();(a/'new.txt').write_bytes(b'active only')
            with self.assertRaises(ValueError):assets_equivalence(a,b)
            (a/'new.txt').unlink();(b/'benchmark_meshes/current.obj').write_bytes(b'changed common')
            with self.assertRaises(ValueError):assets_equivalence(a,b)
            (b/'benchmark_meshes/current.obj').write_bytes(b'current mesh')
            for folder in (a,b):
                (folder/'benchmark_meshes/cipc_table.obj').write_bytes(b'referenced mesh')
                dump(folder/'benchmark_scenes/cloth_hang_l.json',{'case_id':'cloth_hang_l','objects':[{'stiff_mesh':'benchmark_meshes/cipc_table.obj','stiff_mesh_sha256':sha(folder/'benchmark_meshes/cipc_table.obj')}]})
            with self.assertRaises(ValueError):assets_equivalence(a,b)

    def test_wddm_unknown_desktop_is_not_fabricated_compute_evidence(self):
        raw='100, C:/Windows/explorer.exe, N/A\n101, C:/app/gui.exe, [Insufficient Permissions]\n200, C:/repo/gipc.exe, N/A\n201, C:/ml/python.exe, 512\n300, C:/repo/gipc.exe, 1000\n'
        value=runner.process_rows(raw,300,'WDDM')
        self.assertEqual(value['blocking_foreign_pids'],[200,201]);self.assertTrue(value['unknown_load'])
        self.assertIsNone(value['rows'][0]['used_gpu_memory_mib'])
        self.assertEqual(runner.process_rows(raw,300,'TCC')['blocking_foreign_pids'],[100,101,200,201])

    def test_xml_relative_sources_last_release_property(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'StiffGIPC';source.mkdir();build=root/'build/a';build.mkdir(parents=True)
            (source/'PCG_SOLVER.cu').write_text('source');(source/'other.cpp').write_text('source')
            xml='''<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003"><PropertyGroup>
              <IntDir Condition="'$(Configuration)|$(Platform)'=='Release|x64'">gipc.dir/Release/</IntDir></PropertyGroup><ItemGroup>
              <CudaCompile Include="../../StiffGIPC/PCG_SOLVER.cu"><CompileOut>$(IntDir)%(Filename).obj</CompileOut><CompileOut>$(IntDir)solver_a.obj</CompileOut></CudaCompile>
              <ClCompile Include="../../StiffGIPC/other.cpp"><ObjectFileName>$(IntDir)other_b.obj</ObjectFileName></ClCompile>
              </ItemGroup></Project>'''
            (build/'gipc.vcxproj').write_text(xml)
            rows=project_map(root,build);self.assertEqual(len(rows),2);self.assertEqual(Path(rows[0]['object']).name,'solver_a.obj')
            (build/'gipc.vcxproj').write_text(xml.replace('other_b.obj','SOLVER_A.obj'))
            with self.assertRaises(ValueError):project_map(root,build)

    def test_compile_object_link_and_before_source_contract(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);build=root/'build/a';build.mkdir(parents=True)
            for f in ('StiffGIPC','Assets','MeshProcess'):(root/f).mkdir()
            src=root/'StiffGIPC/solver.cu';src.write_text('source');(root/'CMakeLists.txt').write_text('cmake')
            tracking=build/'gipc.dir/Release/gipc.tlog';tracking.mkdir(parents=True)
            obj=tracking.parent/'solver_a.obj';obj.write_bytes(b'object');exe=build/'Release/gipc.exe';exe.parent.mkdir();exe.write_bytes(b'exe')
            (build/'gipc.vcxproj').write_text('''<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003"><PropertyGroup><IntDir Condition="'$(Configuration)|$(Platform)'=='Release|x64'">gipc.dir/Release/</IntDir></PropertyGroup><ItemGroup><CudaCompile Include="../../StiffGIPC/solver.cu"><CompileOut>$(IntDir)solver_a.obj</CompileOut></CudaCompile></ItemGroup></Project>''')
            (build/'CMakeCache.txt').write_text('CMAKE_HOME_DIRECTORY:INTERNAL='+str(root))
            (tracking/'CL.command.1.tlog').write_text('empty fixture cpp set');(tracking/'link.read.1.tlog').write_text(str(obj))
            log=build/'build.log';log.write_text('"C:/CUDA/nvcc.exe" --compile "'+str(src)+'" -o "'+str(obj)+'"')
            dump(build/'build_inputs_before.json',[record(src),record(root/'CMakeLists.txt')])
            with patch('local_identity.compiler_records',return_value=[]):
                identity=active_identity(root,build,exe,log);self.assertEqual(len(identity['objects']),1)
                (tracking/'link.read.1.tlog').write_text('missing object')
                with self.assertRaises(ValueError):active_identity(root,build,exe,log)
                (tracking/'link.read.1.tlog').write_text(str(obj));log.write_text('nvcc.exe --device-link '+str(src))
                with self.assertRaises(ValueError):active_identity(root,build,exe,log)
                log.write_text('nvcc.exe -x cu "'+str(src)+'" -o "'+str(obj)+'"')
                self.assertEqual(len(active_identity(root,build,exe,log)['cuda_commands']),1)
                src.write_text('changed')
                with self.assertRaises(ValueError):active_identity(root,build,exe,log)

    def make_round(self,session,stage):
        ids={'active':{'source_digest':'active','exe':{'sha256':'a'}},'base':{'source_digest':'base','exe':{'sha256':'b'}}}
        seal={'programs':ids,'prior_guard':{'sha256':'failed guard'}}
        old={'active_manifest':{'source_digest':'active','exe_sha256':'a'},'base_manifest':{'source_digest':'base','exe_sha256':'b'},'original_guard':{'sha256':'failed guard'}}
        rows=[]
        for t in tasks(stage):
            entry=synthetic_run(session,t,old);check=analyze_one(session,t,seal)
            dump(session/(t['name']+'_check.json'),check);entry['check_sha256']=sha(session/(t['name']+'_check.json'));rows.append(entry)
        batch={'stage':stage,'plan':plan(),'runs':rows,'skipped':[]}
        return seal,batch,analyze_round(session,stage,seal,batch)

    def test_real_metrics_baseline_missing_velocity_and_frozen_bound(self):
        with tempfile.TemporaryDirectory() as name:
            session=Path(name);seal,batch,report=self.make_round(session,'fixed_r1')
            self.assertTrue(report['diagnosis_complete'],report)
            self.assertFalse(report['all_original_bounds_satisfied']);self.assertFalse(report['allow_performance_screen'])
            self.assertFalse(report['comparisons'][0]['state_difference']['velocity']['available'])
            self.assertTrue(report['old_quality_gate_unchanged'])
            bad=copy.deepcopy(batch);bad['runs'].reverse()
            with self.assertRaises(ValueError):analyze_round(session,'fixed_r1',seal,bad)
            (session/'fixed_base_r1/trace/state_0001.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError):analyze_round(session,'fixed_r1',seal,batch)

    def test_hang_51_and_explicit_reviewed_predecessor(self):
        with tempfile.TemporaryDirectory() as name:
            session=Path(name);folder=session/'hang_r1';folder.mkdir();seal,batch,report=self.make_round(folder,'hang_r1')
            self.assertTrue(report['diagnosis_complete'],report)
            dump(folder/'batch.json',batch);dump(folder/'analysis.json',report)
            dump(folder/'receipt.json',{'seal_sha256':'seal','analysis_sha256':sha(folder/'analysis.json'),'batch_sha256':sha(folder/'batch.json')})
            with self.assertRaises(ValueError):runner.check_previous(session,'hang_r2','seal',None)
            self.assertEqual(len(runner.check_previous(session,'hang_r2','seal',sha(folder/'analysis.json'))),1)
            with self.assertRaises(ValueError):runner.check_previous(session,'hang_r2','changed seal',sha(folder/'analysis.json'))

if __name__=='__main__':unittest.main()
