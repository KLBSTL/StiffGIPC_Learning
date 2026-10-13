"""Offline protocol, metric and failure tests; synthetic fixtures are not results."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext
import numpy as np
from protocol import ROOT,plan,config,SCENES,write,require
from analyze import position_error,observation,physical_inputs_equal,tables,checkpoint_gate
import run_suite


def fixture(path):
    (path/'trace').mkdir(parents=True);(path/'output').mkdir()
    np.array([3,1,0,0,1,2],dtype='<u4').tofile(path/'trace/topology.bin')
    x=np.array([[0,0,0],[1,0,0],[0,0,1]],dtype='<f8')
    for f in (0,1):
        x.tofile(path/f'trace/state_{f:04d}.bin');np.zeros_like(x).tofile(path/f'trace/velocity_{f:04d}.bin')
    for name,values,dtype in [('boundary_types',[0,0,0],'<i4'),('masses',[1,1,1],'<f8'),('body_ids',[-1,-1,-1],'<i4')]:
        np.array(values,dtype=dtype).tofile(path/f'trace/{name}.bin')
    write(path/'trace/metadata.json',dict(abd_point_num=0))
    write(path/'output/scene.json',dict(case_id='synthetic',objects=[dict(dimension=2,body_type='FEM')],effective_run={'dt':.01}))
    write(path/'output/stats.json',dict(frames=[dict(newton=[dict(pcg=dict(iterations=7,iteration_limit=False))],
        phase_ms=dict(assembly=100,pcg=200,ccd=50,line_search=50,state_update=20),newton_exit='movement')]))
    with (path/'trace/frames.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['frame','solver_ms']);w.writerow([1,500])
    write(path/'requested.json',dict(expanded_config=dict(scene='synthetic'),binary='active'))
    write(path/'result.json',dict(status='completed',recorded_frames=1))


class Contracts(unittest.TestCase):
    def setUp(self):
        temp=ROOT/'runs/paper_suite_cpu_tmp';temp.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=temp);self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_manifest_order_is_platform_independent(self):
        paths=[self.root/'baseline/a.cpp',self.root/'CMakeLists.txt',self.root/'baseline/Z.cpp']
        for p in paths:p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture',encoding='utf8')
        expected=['CMakeLists.txt','baseline/Z.cpp','baseline/a.cpp']
        self.assertEqual([r['path'] for r in run_suite.records(paths,root=self.root)],expected)
    def test_exact_seven_by_five_by_three(self):
        tasks=plan()['stages']['performance'];self.assertEqual(len(tasks),105)
        self.assertEqual(len({t['name'] for t in tasks}),105)
        self.assertEqual({t['config']['steps'] for t in tasks},{100})
        for s in SCENES:
            for arm in ('stiff','host','graph','pure','ours'):
                self.assertEqual([t['repeat'] for t in tasks if t['scene_key']==s and t['arm']==arm],[1,2,3])
    def test_graph_changes_only_execution(self):
        for s in SCENES:
            a=config(s,'host');b=config(s,'graph')
            self.assertEqual([k for k in a if a[k]!=b[k]],['execution'])
    def test_physical_dt_and_precision_preserved(self):
        self.assertEqual(config('Hang','ours')['dt'],.04);self.assertEqual(config('Stack','ours')['dt'],.02)
        for t in plan()['stages']['performance']:
            self.assertEqual(t['config']['pcg_rho_tol'],1e-4);self.assertEqual(t['config']['mas'],'legacy')
            self.assertEqual(t['extra_environment']['GIPC_IPC_PCG_FORCING'],'off')
    def test_checkpoint_rules_and_reference(self):
        p=plan();self.assertEqual(len(p['stages']['matched']),24)
        for t in p['stages']['matched']:
            self.assertEqual(t['config']['steps'],1);self.assertEqual(t['config']['execution'],'conditional_graph')
            self.assertEqual(t['target_frame'],28 if t['scene_key']=='Table' else 46)
            if t['arm']=='strict':self.assertEqual(t['config']['pcg_rho_tol'],1e-6)
        self.assertTrue(all(t['config']['ipc_termination']=='compensated' for t in p['stages']['performance'] if t['arm']=='ours'))
    def test_rms_uses_vertex_norm_and_mm(self):
        a=self.root/'a';b=self.root/'b'
        np.array([[0,3,4],[0,0,0]],dtype='<f8').tofile(a);np.zeros((2,3),dtype='<f8').tofile(b)
        e=position_error(a,b);self.assertAlmostEqual(e['rms_mm'],5000/np.sqrt(2));self.assertEqual(e['max_mm'],5000)
    def test_metrics_counts_and_misc_reconcile(self):
        p=self.root/'run';fixture(p)
        r=observation(p,dict(config=dict(steps=1)),dict(status='completed'))
        self.assertEqual((r['P_bar'],r['N_bar'],r['P_tot']),(7,1,7));self.assertAlmostEqual(r['Misc'],.1)
        self.assertAlmostEqual(sum(r[k] for k in ('Hess','LSolver','CCD','LineS','Misc')),r['TimeTot'])
    def test_partial_failure_and_nonfinite_rejected(self):
        p=self.root/'run';fixture(p)
        with self.assertRaises(ValueError):observation(p,dict(config=dict(steps=100)),dict(status='completed'))
        with self.assertRaises(ValueError):observation(p,dict(config=dict(steps=1)),dict(status='memory_budget'))
        np.array([np.nan]*9,dtype='<f8').tofile(p/'trace/state_0001.bin')
        with self.assertRaises(ValueError):observation(p,dict(config=dict(steps=1)),dict(status='completed'))
    def test_physical_inputs_not_solver_tolerance(self):
        a=self.root/'a';b=self.root/'b';fixture(a);fixture(b)
        s=json.loads((b/'output/scene.json').read_text());s['effective_run']['pcg_tol']=1e-6
        (b/'output/scene.json').write_text(json.dumps(s))
        self.assertTrue(physical_inputs_equal(a,b)['passed'])
        np.array([2,1,1],dtype='<f8').tofile(b/'trace/masses.bin')
        self.assertFalse(physical_inputs_equal(a,b)['passed'])
    def test_pending_data_produces_headers_without_example_numbers(self):
        session=self.root/'empty';session.mkdir();out=self.root/'out';tables(session,out)
        with (out/'phase_table.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),0)
        self.assertFalse(read_summary(out)['performance_certified'])
    def test_immutable_outputs(self):
        p=self.root/'file.json';write(p,{'a':1})
        with self.assertRaises(FileExistsError):write(p,{'a':2})
    def test_foreign_compute_prevents_launch_and_is_never_stopped(self):
        t=plan()['stages']['smoke'][0]
        gpu={'uuid':'synthetic','utilization.gpu':0,'memory.free':8000}
        with patch.object(run_suite,'gpu_query',return_value=gpu),patch.object(run_suite,'compute_pids',return_value=[12345]),\
             patch.object(run_suite.subprocess,'Popen') as launch,patch.object(run_suite,'stop_owned') as stop,\
             patch.object(run_suite.time,'sleep'):
            r=run_suite.execute(self.root,t,{},0)
        self.assertEqual(r['status'],'launcher_or_monitor_failed');launch.assert_not_called()
        stop.assert_called_once_with(None)
    def test_preflight_waits_for_two_idle_samples_without_lowering_threshold(self):
        samples=[{'uuid':'same','utilization.gpu':v,'memory.free':8000} for v in (40,0,0)]
        report=self.root/'preflight.json'
        with patch.object(run_suite,'gpu_query',side_effect=samples),patch.object(run_suite,'compute_pids',return_value=[]),\
             patch.object(run_suite.time,'sleep'):
            before=run_suite.wait_idle_gpu(0,report)
        self.assertEqual([s['utilization.gpu'] for s in before],[0,0])
        recorded=json.loads(report.read_text());self.assertTrue(recorded['passed']);self.assertEqual(len(recorded['samples']),3)
        self.assertEqual(recorded['utilization_max'],5)
    def test_preflight_deadline_is_finite_and_records_failure(self):
        report=self.root/'expired.json'
        with patch.object(run_suite.time,'monotonic',side_effect=[0,2]),patch.object(run_suite,'gpu_query') as query:
            with self.assertRaises(ValueError):run_suite.wait_idle_gpu(0,report,timeout_seconds=1)
        query.assert_not_called();self.assertFalse(json.loads(report.read_text())['passed'])
    def test_failure_skips_same_scene_without_retry_and_keeps_other_scene(self):
        tasks=plan()['stages']['smoke'][:3]  # Sphere stiff/ours, Bunny stiff
        manifest=self.root/'manifest.json';write(manifest,{'synthetic':True})
        session=self.root/'serial';responses=[dict(status='memory_budget',recorded_frames=1),dict(status='completed',recorded_frames=2)]
        small=plan();small['stages']['smoke']=tasks
        with patch.object(run_suite.sys,'platform','linux'),patch.object(run_suite.os,'environ',{'DISPLAY':':synthetic'}),\
             patch.object(run_suite,'verify'),patch.object(run_suite,'gpu_lock',return_value=nullcontext()),\
             patch.object(run_suite,'plan',return_value=small),patch.object(run_suite,'execute',side_effect=responses) as execute:
            code=run_suite.run(manifest,session,'smoke',0)
        packet=json.loads((session/'smoke_ledger.json').read_text())
        self.assertEqual(code,2);self.assertEqual(execute.call_count,2)
        self.assertEqual(len(packet['skipped']),1);self.assertEqual(packet['runs'][1]['task']['scene_key'],'Bunny')
    def test_exact_checkpoint_input_is_separate_from_endpoint_determinism(self):
        session=self.root/'checkpoint';session.mkdir();tasks=plan()['stages']['matched'];rows=[]
        binary='a'*64;x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,1.]],dtype='<f8')
        for scene,frame in [('Table',28),('Bunny',46)]:
            seed=session/f'seed_{scene.lower()}_original_r1';cp=seed/'checkpoint';cp.mkdir(parents=True);(seed/'trace').mkdir()
            x.tofile(cp/'vertices.bin');np.zeros_like(x).tofile(cp/'velocities.bin')
            x.tofile(seed/f'trace/state_{frame:04d}.bin');np.zeros_like(x).tofile(seed/f'trace/velocity_{frame:04d}.bin')
            write(cp/'checkpoint.json',dict(binary_id=binary,completed_physical_frames=frame-1,buffers={'vertices':{},'velocities':{}}))
        for t in tasks:
            out=session/t['name'];(out/'trace').mkdir(parents=True);(out/'output').mkdir()
            x.tofile(out/'trace/state_0000.bin');np.zeros_like(x).tofile(out/'trace/velocity_0000.bin')
            (x+1e-5).tofile(out/'trace/state_0001.bin');np.zeros_like(x).tofile(out/'trace/velocity_0001.bin')
            write(out/'requested.json',dict(exe_sha256=binary))
            write(out/'output/checkpoint_restore.json',dict(binary_id=binary,completed_physical_frames=t['target_frame']-1,
                device_bytes_verified=True,context_verified=True,restored_buffers=2))
            rows.append(dict(task=t,result={'status':'completed'}))
        ledger=dict(complete=True,runs=rows);gate=checkpoint_gate(session,ledger)
        self.assertTrue(gate['passed']);self.assertFalse(gate['hot_cold_1e8_passed'])
        np.zeros_like(x).tofile(session/tasks[0]['name']/'trace/state_0000.bin')
        self.assertFalse(checkpoint_gate(session,ledger)['passed'])


def read_summary(out):return json.loads((out/'summary.json').read_text())
if __name__=='__main__':unittest.main()
