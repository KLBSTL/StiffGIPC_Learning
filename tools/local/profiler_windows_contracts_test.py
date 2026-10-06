"""CPU-only contracts, including real suspended Windows parent/child jobs.

No nvidia-smi, Nsight capture, GIPC launch, native build or GPU work is invoked.
"""
import csv
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from profiler_windows import (ROOT, capture_evidence, derive_config, digest, expand, execute,
    nsys_command, owned_process_rows, remaining, reserve_attempt, resource_reason)
from windows_owned_job import OwnedJob


class ConfigurationTests(unittest.TestCase):
    def reference(self):
        c = expand({'scene':'cloth_fixed_bunny_l', 'preset':'combined', 'steps':59,
                    'diagnostics':[], 'profile':'none', 'timeout_seconds':120,
                    'discrete_bvh_refit':True})
        return {'binary':'active', 'expanded_config':c, 'config_sha256':digest(c)}, {
            'status':'completed', 'recorded_frames':59, 'finite':True}

    def test_only_allowed_diagnostic_fields_change(self):
        req, res = self.reference()
        for mode in ('node','graph'):
            c = derive_config(req,res,mode)
            changed = {k for k in c if c[k] != req['expanded_config'][k]}
            self.assertEqual(changed, {'diagnostics','cost_frames','cost_events','profile'})
            self.assertEqual((c['steps'],c['timeout_seconds'],c['pcg_rho_tol']), (59,120,1e-4))
        res['status']='memory_budget'
        with self.assertRaises(ValueError): derive_config(req,res,'node')

    def test_no_silent_protocol_or_config_change(self):
        for key, value in [('steps',58),('ipc_min_updates',5),('pcg_rho_tol',1e-8),('dt',.02),('contact_pool_validate',True)]:
            req,res=self.reference();req['expanded_config'][key]=value;req['config_sha256']=digest(req['expanded_config'])
            with self.assertRaises(ValueError): derive_config(req,res,'node')
        req,res=self.reference();req['config_sha256']='changed'
        with self.assertRaises(ValueError): derive_config(req,res,'node')

    def test_nsys_contract_and_absolute_deadline(self):
        command=nsys_command(Path('nsys.exe'),Path('gipc.exe'),Path('out'),'node')
        self.assertIn('--capture-range-end=stop',command)
        self.assertIn('--nvtx-capture=ipc.physical_frame',command)
        self.assertNotIn('--duration=120',command)
        with self.assertRaises(subprocess.TimeoutExpired): remaining(time.monotonic()-1)
        self.assertLessEqual(remaining(time.monotonic()+1),1)

    def test_two_immutable_attempts_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in (1,2): self.assertEqual(Path(reserve_attempt(root,'a'*64,root/f'out{i}','node')['path']).name,f'attempt{i}.json')
            first=(root/'runs/local_profile_attempts'/('a'*64)/'attempt1.json').read_bytes()
            with self.assertRaises(ValueError): reserve_attempt(root,'a'*64,root/'out3','graph')
            self.assertEqual(first,(root/'runs/local_profile_attempts'/('a'*64)/'attempt1.json').read_bytes())

    def test_owned_descendants_and_foreign_gpu_rules(self):
        raw='11, C:\\test\\nsys.exe, [N/A]\n12, C:\\test\\gipc.exe, [N/A]\n13, C:\\other\\gipc.exe, [N/A]\n14, desktop.exe, [N/A]\n15, compute.exe, 1\n'
        r=owned_process_rows(raw,'WDDM',lambda pid: pid in (11,12))
        self.assertEqual(r['blocking_foreign_pids'],[13,15])
        self.assertTrue(r['unknown_load'])
        # Stale historical PID sets are not an ownership authority.
        r=owned_process_rows(raw,'WDDM',lambda pid: False)
        self.assertEqual(r['blocking_foreign_pids'],[12,13,15])

    def test_original_resource_thresholds(self):
        self.assertIsNone(resource_reason({'memory.free':2000},3500,1900,2**30,[]))
        self.assertEqual(resource_reason({'memory.free':1500},3500,1900,2**30,[]),'memory_budget')
        self.assertEqual(resource_reason({'memory.free':767},2000,1500,2**30,[]),'memory_budget')
        self.assertEqual(resource_reason({'memory.free':2000},3500,1900,2**30,[5]),'foreign_gpu_load')
        self.assertEqual(resource_reason({'memory.free':2000},3500,1900,2**30-1,[]),'disk_reserve')

    def fixture(self, out):
        (out/'trace').mkdir();(out/'output').mkdir()
        with (out/'trace/frames.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['frame','solver_ms']);writer.writeheader()
            writer.writerows({'frame':i,'solver_ms':1} for i in range(1,60))
        (out/'output/stats.json').write_text(json.dumps({'frames':[{} for _ in range(59)]}))
        (out/'final.bin').write_bytes(b'\0'*8)
        row={'schema':'gipc.cost.v1','frame':57,'nvtx_available':True,'gpu_events_enabled':False,
             'gpu_interval_ms':None,'operator_probe_enabled':False,'sample_kind':'production',
             'measurement_mode':'nvtx_cpu_only','cpu_submit_ms':1.,'scope_id':1,
             'parent_scope_id':0,'stage':'ipc.physical_frame'}
        (out/'cost.jsonl').write_text(json.dumps(row)+'\n')
        (out/'nsight.nsys-rep').write_bytes(b'synthetic CPU contract only')
        return row

    def test_capture_fail_closed(self):
        for key,value in [('nvtx_available',False),('frame',56),('gpu_events_enabled',True),('operator_probe_enabled',True)]:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);row=self.fixture(out)
                self.assertTrue(capture_evidence(out)['passed'])
                row[key]=value;(out/'cost.jsonl').write_text(json.dumps(row)+'\n')
                with self.assertRaises(ValueError):capture_evidence(out)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);self.fixture(out);(out/'nsight.nsys-rep').unlink()
            with self.assertRaises(ValueError):capture_evidence(out)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);self.fixture(out);(out/'output/stats.json').write_text('{"frames":[]}')
            with self.assertRaises(ValueError):capture_evidence(out)


@unittest.skipUnless(os.name=='nt','Real Windows Job tests')
class RealJobTests(unittest.TestCase):
    def wait_file(self,path):
        deadline=time.monotonic()+6
        while not path.exists():
            if time.monotonic()>deadline:self.fail('CPU child marker timeout')
            time.sleep(.02)
        return json.loads(path.read_text())

    def test_parent_child_grandchild_termination_and_foreign_job_survival(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);helper=root/'cpu_tree.py'
            helper.write_text('''import json,os,pathlib,subprocess,sys,time
root=pathlib.Path(sys.argv[1]); level=int(sys.argv[2])
(root/(str(level)+'.json')).write_text(json.dumps({'pid':os.getpid()}))
if level<2:subprocess.Popen([sys.executable,__file__,str(root),str(level+1)])
time.sleep(40)
''')
            with (root/'own.log').open('wb') as log,(root/'foreign.log').open('wb') as log2,OwnedJob() as job,OwnedJob() as foreign:
                foreign.launch([sys.executable,'-c','import time;time.sleep(40)'],root,os.environ.copy(),log2)
                job.launch([sys.executable,str(helper),str(root),'0'],root,os.environ.copy(),log)
                child_ids={self.wait_file(root/f'{i}.json')['pid'] for i in range(3)}
                self.assertEqual(job.lifecycle,['created_suspended','assigned_to_job','resumed'])
                self.assertTrue(child_ids<=job.pids())
                self.assertTrue(all(job.owns_pid(pid) for pid in child_ids))
                self.assertFalse(job.owns_pid(foreign.pid))
                job.observe();job.terminate()
                self.assertFalse(job.pids())
                self.assertTrue(all(p['exit_code'] is not None for p in job.member_evidence()))
                self.assertIsNone(foreign.poll())
                self.assertIn(foreign.pid,foreign.pids())

    def test_assignment_failure_never_resumes_or_leaks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);marker=root/'should_not_run.txt'
            with (root/'log').open('wb') as log,OwnedJob() as job:
                with patch.object(job,'_assign',side_effect=OSError('injected assignment failure')):
                    with self.assertRaises(OSError):
                        job.launch([sys.executable,'-c',f'from pathlib import Path;Path({str(marker)!r}).write_text("bad")'],root,os.environ.copy(),log)
                self.assertEqual(job.lifecycle,['created_suspended'])
                self.assertFalse(marker.exists());self.assertFalse(job.pids())

    def test_exception_cleanup_kills_owned_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            job=OwnedJob()
            with (root/'log').open('wb') as log:
                with self.assertRaisesRegex(RuntimeError,'injected monitor failure'):
                    with job:
                        job.launch([sys.executable,'-c','import time;time.sleep(40)'],root,os.environ.copy(),log)
                        job.observe()
                        held=job.api.OpenProcess(0x1000 | 0x100000,False,job.pid)
                        self.assertTrue(held)
                        raise RuntimeError('injected monitor failure')
                self.assertIsNone(job.job);self.assertIsNone(job.process)
                try:self.assertIsNotNone(job.exit_code(held))
                finally:job.api.CloseHandle(held)

    def test_root_exit_is_not_tree_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);marker=root/'child.json'
            code='import os,json,time;from pathlib import Path;Path('+repr(str(marker))+').write_text(json.dumps({"pid":os.getpid()}));time.sleep(40)'
            parent='import subprocess,sys;subprocess.Popen([sys.executable,"-c",'+repr(code)+'])'
            with (root/'log').open('wb') as log,OwnedJob() as job:
                job.launch([sys.executable,'-c',parent],root,os.environ.copy(),log)
                pid=self.wait_file(marker)['pid']
                until=time.monotonic()+5
                while job.poll() is None and time.monotonic()<until:time.sleep(.02)
                self.assertEqual(job.poll(),0)
                self.assertIn(pid,job.pids());self.assertFalse(job.finished())
                job.observe();job.terminate();self.assertTrue(job.finished())

    def test_launcher_monitor_failure_records_and_kills_tree(self):
        # Exercise the actual launcher error/finally path with CPU subprocesses.
        # All GPU/profiler/identity inputs are explicitly replaced by test data.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'runs/cpu_only';seal=root/'seal.json';seal.write_text('{}')
            marker=root/'child.json';held=[];api=OwnedJob()
            childcode='import os,json,time;from pathlib import Path;Path('+repr(str(marker))+').write_text(json.dumps({"pid":os.getpid()}));time.sleep(40)'
            parent='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",'+repr(childcode)+']);time.sleep(40)'
            identity={'exe':{'path':sys.executable,'sha256':'cpu-test'},'source_digest':'cpu-test'}
            c=ConfigurationTests().reference()[0]['expanded_config']
            calls=[]
            def monitor(*args):
                calls.append(1)
                if len(calls)==1:return {'blocking_foreign_pids':[]}
                pid=self.wait_file(marker)['pid']
                handle=api.api.OpenProcess(0x1000 | 0x100000,False,pid)
                self.assertTrue(handle);held.append(handle)
                raise RuntimeError('injected telemetry failure')
            try:
                with patch('profiler_windows.gpu_lock',return_value=nullcontext()), \
                     patch('profiler_windows.prepare',return_value=({},identity,c,{},[],[])), \
                     patch('profiler_windows.driver_model',return_value='WDDM'), \
                     patch('profiler_windows.gpu_query',return_value={'memory.free':6000,'utilization.gpu':20,'uuid':'cpu-only'}), \
                     patch('profiler_windows.processes',side_effect=monitor), \
                     patch('profiler_windows.environment',return_value=os.environ.copy()), \
                     patch('profiler_windows.nsys_command',return_value=[sys.executable,'-c',parent]):
                    result=execute(root,seal,root/'reference',out,'node',Path(sys.executable),0)
                self.assertEqual(result['status'],'launcher_or_monitor_failed')
                self.assertEqual(result['raw_status'],'launched')
                self.assertTrue(result['cleanup_owned_job_empty'])
                self.assertFalse(result['capture_verified'])
                self.assertTrue((out/'raw_result.json').is_file())
                self.assertTrue((out/'evidence.json').is_file())
                self.assertTrue(held)
                self.assertTrue(all(api.exit_code(h) is not None for h in held))
            finally:
                for h in held:api.api.CloseHandle(h)
                api.close()


if __name__=='__main__':unittest.main()
