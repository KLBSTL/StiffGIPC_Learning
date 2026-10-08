"""Offline synthetic contracts. Does not launch a solver, GPU, compiler or SSH."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from local_plan import ROOT,tasks,plan,predecessor
from local_identity import (project_map,active_identity,record,verify,assets_equivalence,UNUSED_BASE_CACHES,
                            freeze_active_identity,verify_active_sources,_git_source_bytes,
                            create_execution_observation_seal,verify_seal)
from local_analysis import analyze_one,analyze_round
import windows_runner as runner
from quality_contracts_test import fixture as synthetic_run
from config import expand,digest
from linux_runner import sha,inventory,write_new

def dump(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf-8')

class Contracts(unittest.TestCase):
    def test_fixed_export_directory_created_before_launch_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'fixed'
            runner.prepare_run_output(out,{'diagnostics':['fixed']})
            self.assertTrue((out/'output').is_dir())
            self.assertTrue((out/'fixed').is_dir())
            with self.assertRaises(ValueError):runner.prepare_run_output(out,{'diagnostics':['fixed']})
            normal=Path(directory)/'normal'
            runner.prepare_run_output(normal,{'diagnostics':[]})
            self.assertFalse((normal/'fixed').exists())

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

    def test_configured_180_and_600_second_budgets(self):
        for steps,budget in ((120,180),(300,600)):
            self.assertEqual(expand({'steps':steps,'timeout_seconds':budget})['timeout_seconds'],budget)
            self.assertEqual(runner.final_status('running',0,budget-.001,budget),'completed')
            self.assertEqual(runner.final_status('running',0,budget+.001,budget),'timeout')
            with patch.object(runner.time,'monotonic',return_value=100):
                with self.assertRaises(runner.subprocess.TimeoutExpired) as error:runner.remaining(100,budget)
                self.assertEqual(error.exception.timeout,budget)
        self.assertEqual(runner.final_status('running',0,120.001),'timeout')

    def test_native_launcher_monitor_failure_closes_only_owned_job(self):
        class CpuJob:
            pid=123; lifecycle=['created_suspended','assigned_to_job','resumed']
            stopped=False;closed=False;launched=False
            def launch(self,command,cwd,env,log):self.launched=True
            def finished(self):return self.stopped
            def poll(self):return 1 if self.stopped else None
            def observe(self):return [{'pid':123,'image':'synthetic.exe','exit_code':self.poll()}]
            def owns_pid(self,pid):return pid==123
            def terminate(self):self.stopped=True
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as name:
            session=Path(name);job=CpuJob()
            task={'name':'cpu_only','binary':'active','config':{'steps':120,'timeout_seconds':180}}
            identity={'sources':[],'dlls':[],'exe':{'path':'synthetic.exe','sha256':'a'*64},
                      'source_digest':'cpu-fixture','capabilities':{}}
            sample={'memory.free':6000,'utilization.gpu':0,'uuid':'cpu-fixture'}
            with patch.object(runner,'OwnedJob',return_value=job),patch.object(runner,'verify'), \
                 patch.object(runner,'driver_model',return_value='WDDM'), \
                 patch.object(runner,'gpu_query',side_effect=[sample,sample,RuntimeError('injected monitor failure')]), \
                 patch.object(runner,'gpu_processes',return_value={'rows':[],'blocking_foreign_pids':[]}), \
                 patch.object(runner.shutil,'disk_usage',return_value=type('Disk',(),{'free':5*1024**3})()), \
                 patch.object(runner.time,'sleep'):
                result=runner.execute(session,task,identity,0)
            self.assertEqual(result['status'],'launcher_or_monitor_failed');self.assertEqual(result['timeout_seconds'],180)
            self.assertTrue(job.launched and job.closed and job.stopped);self.assertTrue(result['cleanup_owned_job_empty'])
            requested=json.loads((session/'cpu_only/requested.json').read_text())
            self.assertEqual(requested['resource_policy']['absolute_timeout_seconds'],180)
            self.assertEqual(requested['memory_budget_mib'],min(.75*6000,6000-1536))
            self.assertEqual(requested['resource_policy']['memory_free_min_mib'],768)

    def freeze_fixture(self,root):
        build=root/'build/a';build.mkdir(parents=True)
        for folder in ('StiffGIPC','Assets','MeshProcess'):(root/folder).mkdir()
        source=root/'StiffGIPC/solver.cu';source.write_bytes(b'compiled source\r\n')
        (root/'MeshProcess/helper.h').write_bytes(b'header\r\n')
        (root/'Assets/mesh.bin').write_bytes(b'runtime input')
        (root/'CMakeLists.txt').write_bytes(b'cmake\r\n')
        originals=[record(p) for p in (source,root/'MeshProcess/helper.h',root/'Assets/mesh.bin',root/'CMakeLists.txt')]
        tracking=build/'gipc.dir/Release/gipc.tlog';tracking.mkdir(parents=True)
        obj=tracking.parent/'solver.obj';obj.write_bytes(b'object')
        exe=build/'Release/gipc.exe';exe.parent.mkdir();exe.write_bytes(b'exe')
        log=build/'build.log';log.write_bytes(b'actual build evidence')
        (build/'gipc.vcxproj').write_bytes(b'actual source/object XML')
        link=tracking/'link.read.1.tlog';link.write_text(str(obj),encoding='utf-8')
        dump(build/'build_inputs_before.json',originals)
        dump(build/'windows_manifest.json',{'kind':'active','status':'completed','source_root':str(root),
              'build_dir':str(build),'exe':record(exe),'source_inventory':originals})
        identity={'kind':'active','source_root':str(root),'sources':originals,'source_digest':digest(originals),
            'exe':record(exe),'dlls':[],'objects':[record(obj)],'compilers':[],'scope':'CPU fixture',
            'source_object_map':[{'source':str(source),'object':str(obj),'kind':'CudaCompile'}],
            'build_evidence':[record(p) for p in (log,link,build/'gipc.vcxproj',build/'build_inputs_before.json',build/'windows_manifest.json')]}
        prior=root/'prior_identity.json';dump(prior,identity)
        return build,exe,log,source,prior,identity

    def test_verified_snapshot_recovers_exact_blob_and_keeps_runtime_assets_live(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);build,exe,log,source,prior,identity=self.freeze_fixture(root)
            source.write_bytes(b'new probe source\n');(root/'StiffGIPC/new_probe.cuh').write_bytes(b'new')
            snapshot=build/'frozen_native_inputs'
            with patch('local_identity._git_source_bytes',return_value=(b'compiled source\r\n',
                       {'kind':'git_blob','commit':'a'*40,'path':'StiffGIPC/solver.cu'})) as fetch:
                frozen=freeze_active_identity(root,build,exe,log,snapshot,prior,'3663d22')
            self.assertEqual(fetch.call_count,1);self.assertEqual(frozen['source_digest'],identity['source_digest'])
            self.assertEqual((snapshot/'StiffGIPC/solver.cu').read_bytes(),b'compiled source\r\n')
            self.assertFalse((snapshot/'StiffGIPC/new_probe.cuh').exists());self.assertFalse((snapshot/'Assets').exists())
            verify_active_sources(frozen)
            source.write_bytes(b'further active changes');verify_active_sources(frozen)
            (root/'Assets/mesh.bin').write_bytes(b'changed runtime input')
            with self.assertRaises(ValueError):verify_active_sources(frozen)
            (root/'Assets/mesh.bin').write_bytes(b'runtime input')
            (snapshot/'StiffGIPC/solver.cu').write_bytes(b'tampered snapshot')
            with self.assertRaises(ValueError):verify_active_sources(frozen)

    def test_frozen_snapshot_never_normalizes_blob_or_accepts_mismatched_build(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);build,exe,log,source,prior,identity=self.freeze_fixture(root)
            source.write_bytes(b'changed')
            with patch('local_identity._git_source_bytes',return_value=(b'compiled source\n',{'kind':'git_blob'})):
                with self.assertRaisesRegex(ValueError,'exactly match compiled bytes'):
                    freeze_active_identity(root,build,exe,log,build/'wrong_newlines',prior,'3663d22')
            manifest=build/'windows_manifest.json';value=json.loads(manifest.read_text());value['source_inventory'][0]['sha256']='0'*64;dump(manifest,value)
            with self.assertRaises(ValueError):freeze_active_identity(root,build,exe,log,build/'wrong_manifest',prior,'3663d22')
            self.assertFalse((build/'wrong_manifest').exists())

    def test_git_source_reads_raw_bytes_without_shell_or_text_conversion(self):
        completed=runner.subprocess.CompletedProcess([],0,stdout=b'a'*40+b'\n')
        blob=runner.subprocess.CompletedProcess([],0,stdout=b'raw\r\nbytes\xff')
        with patch('local_identity.subprocess.run',side_effect=[completed,blob]) as run:
            raw,origin=_git_source_bytes(Path('repo'),'3663d22',Path('StiffGIPC/solver.cu'))
        self.assertEqual(raw,b'raw\r\nbytes\xff');self.assertEqual(origin['commit'],'a'*40)
        self.assertEqual(run.call_args_list[1].args[0][-2:],['blob','a'*40+':StiffGIPC/solver.cu'])
        self.assertNotIn('text',run.call_args_list[1].kwargs);self.assertNotIn('shell',run.call_args_list[1].kwargs)

    def test_observation_seal_checks_fresh_program_tools_plan_guard_and_source(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);build,exe,log,source,prior,identity=self.freeze_fixture(root)
            plan_path=root/'PLAN.md';plan_path.write_bytes(b'actual finite observation plan\r\n')
            guard=root/'prior_guard.json';dump(guard,{'allow_next_round':False,'old_failure':'retained'})
            tool=root/'profiler.py';tool.write_bytes(b'actual tool')
            seal_path=root/'observation.json'
            with patch('local_identity.execution_observation_tools',side_effect=lambda _: [record(tool)]):
                seal=create_execution_observation_seal(root,identity,plan_path,guard,seal_path)
                self.assertEqual(verify_seal(root,seal_path),seal)
                self.assertNotIn('plan',seal);self.assertEqual(seal['profile_attempts_per_program'],2)
                for path,changed in ((tool,b'changed tool'),(plan_path,b'changed plan'),
                                     (guard,b'{"allow_next_round": true}'),(source,b'changed source')):
                    original=path.read_bytes();path.write_bytes(changed)
                    with self.subTest(path=path),self.assertRaises(ValueError):verify_seal(root,seal_path)
                    path.write_bytes(original)
                for key,value in (('profile_attempts_per_program',3),('quality_certified',True),('extra',False)):
                    bad=dict(seal);bad[key]=value;dump(root/'bad.json',bad)
                    with self.subTest(key=key),self.assertRaises(ValueError):verify_seal(root,root/'bad.json')

    def test_observation_seal_accepts_verified_frozen_program_and_rejects_object_drift(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);build,exe,log,source,prior,identity=self.freeze_fixture(root)
            source.write_bytes(b'new active code')
            with patch('local_identity._git_source_bytes',return_value=(b'compiled source\r\n',
                       {'kind':'git_blob','commit':'a'*40,'path':'StiffGIPC/solver.cu'})):
                frozen=freeze_active_identity(root,build,exe,log,build/'snapshot',prior,'3663d22')
            plan_path=root/'PLAN.md';plan_path.write_bytes(b'actual observation plan')
            guard=root/'guard.json';dump(guard,{'allow_next_round':False})
            with patch('local_identity.execution_observation_tools',return_value=[]):
                seal_path=root/'observation.json'
                create_execution_observation_seal(root,frozen,plan_path,guard,seal_path)
                verify_seal(root,seal_path)
                Path(frozen['objects'][0]['path']).write_bytes(b'changed object')
                with self.assertRaises(ValueError):verify_seal(root,seal_path)

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
