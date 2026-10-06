"""Synthetic CPU contract checks. No GPU, SSH, build or real-run mutation."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from quality_plan import tasks,plan
import fixed_quality as driver
import quality_analysis as analysis
from config import expand
from contracts_test import make_run as synthetic_run
from linux_runner import inventory,sha

def dump(p,value):p.write_text(json.dumps(value),encoding='utf-8')

def fixture(session,t,seal):
    identity=seal['base_manifest' if t['binary']=='base' else 'active_manifest']
    run=synthetic_run(session,t,identity);c=expand(t['config'])
    req=driver.read(run/'requested.json');req['binary']=t['binary'];dump(run/'requested.json',req)
    scene=driver.read(run/'output/scene.json')
    scene.update(effective_run={'dt':.01,'newton_tol':.01,'pcg_tol':1e-4,'friction':.2},effective_scalar_fields={'preconditioner_type':1})
    dump(run/'output/scene.json',scene)
    stats=driver.read(run/'output/stats.json')
    if t['binary']=='base':
        (run/'resolved_config.json').unlink()
        for p in (run/'trace').glob('velocity_*.bin'):p.unlink()
        for f in stats['frames']:
            f.pop('contact_pool')
            for n in f['newton']:n['pcg']['iteration_limit']=False
    else:
        resolved={'contact_backend':'ipc','dt':.01,'ipc_newton_tol':.01,'pcg_rho_tol':1e-4,'configured_pcg_execution':c['execution'],
            'configured_pcg_graph_chunk':c['pcg_graph_chunk'],'fixed_graph_chunk_study':c['fixed_graph_chunk_study'],
            'mas':{'cholesky':False,'inverse64':False,'wide_apply':False,'restriction_mode':'inactive','factor_action':'inactive'},
            'acceleration_features':{'GIPC_CCD_BVH_REFIT':True,'GIPC_BATCHED_ENERGY':True,'GIPC_ENERGY_REUSE':True},
            'report_components':{'mas_fused_dot_requested':False,'fixed_mas_dot_study':False,'discrete_bvh_refit':True,
                'discrete_bvh_rebuild_interval':8,'discrete_bvh_validate':False,'contact_pool':c['contact_pool'],'contact_pool_validate':False}}
        dump(run/'resolved_config.json',resolved)
        for f in stats['frames']:
            for n in f['newton']:
                p=n['pcg'];p.update(execution=c['execution'],mas_fused_dot_requested=False)
                if c['execution']=='conditional_graph':p['graph_chunk_iterations']=c['pcg_graph_chunk']
                else:p.pop('graph_chunk_iterations',None)
    dump(run/'output/stats.json',stats)
    # Deliberate frozen-bound excess; finite and no element inversion.
    for p in (run/'trace').glob('state_*.bin'):
        if p.name!='state_0000.bin':
            x=np.fromfile(p,dtype='<f8');(x*1.02).tofile(p)
    dump(run/'evidence.json',{'files':inventory(run,[p for p in run.rglob('*') if p.is_file()])})
    check=analysis.analyze_run(session,t,seal);dump(session/(t['name']+'_check.json'),check)
    return {'name':t['name'],'result':driver.read(run/'result.json'),'evidence_sha256':sha(run/'evidence.json'),
            'check_sha256':sha(session/(t['name']+'_check.json')),'hard_checks_passed':check['hard_checks_passed']}

class QualityContracts(unittest.TestCase):
    def test_plan_seven_fixed_inputs_no_screen_gate(self):
        ts=tasks();self.assertEqual([t['variant'] for t in ts],['off','base','base','off','on','off','base'])
        self.assertEqual([t['name'] for t in ts],['fixed_off_r1','fixed_base_r1','fixed_base_r2','fixed_off_r2','fixed_on_r1','fixed_off_r3','fixed_base_r3'])
        for t in ts:
            c=expand(t['config']);self.assertEqual((c['steps'],c['dt'],c['ipc_min_updates'],c['pcg_rho_tol'],c['timeout_seconds']),(59,.01,6,1e-4,120))
            self.assertEqual(c['diagnostics'],[]);self.assertFalse(c['contact_pool_validate'])
        self.assertFalse(plan()['allow_performance_screen']);self.assertFalse(plan()['stop_on_frozen_bound_exceedance'])

    def test_baseline_environment_not_active_identity(self):
        c=expand(next(t['config'] for t in tasks() if t['binary']=='base'))
        with patch.dict(os.environ,{'GIPC_CONTACT_POOL':'1','GIPC_RESOLVED_CONFIG':'wrong','GIPC_TRACE_VELOCITY':'1'}):
            env=driver.baseline_environment(c,Path('/tmp/fixture'))
        self.assertNotIn('GIPC_CONTACT_POOL',env);self.assertNotIn('GIPC_RESOLVED_CONFIG',env);self.assertNotIn('GIPC_TRACE_VELOCITY',env)
        self.assertEqual(env['GIPC_STEPS'],'59');self.assertEqual(env['GIPC_PCG_TOL'],'0.0001')

    def test_full_analysis_preserves_original_failure_and_absent_velocity(self):
        seal={'active_manifest':{'source_digest':'active','exe_sha256':'a'},'base_manifest':{'source_digest':'base','exe_sha256':'b'},'original_guard':{'sha256':'g'}}
        with tempfile.TemporaryDirectory() as name:
            session=Path(name);rows=[fixture(session,t,seal) for t in tasks()]
            report=analysis.analyze(session,seal,{'plan':plan(),'runs':rows,'skipped':[]})
            self.assertTrue(report['diagnosis_complete'],[(r['name'],r['failures']) for r in report['runs']])
            self.assertTrue(report['original_guard_remains_failed']);self.assertFalse(report['quality_certified']);self.assertFalse(report['allow_performance_screen'])
            self.assertEqual(len(report['observed_ranges']['base']['bounds']['p99_stretch']['runs_exceeding_old_bound']),3)
            for r in report['runs']:
                self.assertTrue(r['hard_checks_passed']);self.assertFalse(r['all_original_bounds_satisfied'])
                if r['binary']=='base':self.assertFalse(r['actual_velocity']['available'])
            pair=next(x for x in report['comparisons'] if x['right']=='fixed_base_r1')
            self.assertTrue(pair['initial_inputs']['passed']);self.assertFalse(pair['initial_inputs']['initial_actual_velocity_comparable'])
            self.assertFalse(pair['state_difference']['velocity']['available'])
            bad=dict(rows[0]);bad['name']='fixed_on_r1'
            with self.assertRaises(ValueError):analysis.analyze(session,seal,{'plan':plan(),'runs':[bad],'skipped':[t['name'] for t in tasks()[1:]]})
            (session/'fixed_base_r1/output/stats.json').write_text('{}')
            with self.assertRaises(ValueError):analysis.analyze(session,seal,{'plan':plan(),'runs':rows,'skipped':[]})

    def test_baseline_config_failure_and_no_velocity_fabrication(self):
        seal={'base_manifest':{'source_digest':'base','exe_sha256':'b'},'active_manifest':{},'original_guard':{'sha256':'g'}}
        with tempfile.TemporaryDirectory() as name:
            t=next(t for t in tasks() if t['binary']=='base');session=Path(name);fixture(session,t,seal);run=session/t['name']
            self.assertTrue(analysis.validate_base(run)['passed'])
            stats=driver.read(run/'output/stats.json')
            for f in stats['frames']:
                for n in f['newton']:n['pcg'].pop('breakdown',None)
            dump(run/'output/stats.json',stats)
            self.assertFalse(analysis.validate_base(run)['pcg_breakdown_telemetry_available'])
            stats['frames'][0]['newton_exit']='iteration_limit';dump(run/'output/stats.json',stats)
            self.assertFalse(analysis.analyze_run(session,t,seal)['hard_checks_passed'])
            dump(run/'resolved_config.json',{'pretend':'active'})
            self.assertFalse(analysis.validate_base(run)['passed'])
            (run/'resolved_config.json').unlink();scene=driver.read(run/'output/scene.json');scene['effective_run']['pcg_tol']=1e-3;dump(run/'output/scene.json',scene)
            self.assertFalse(analysis.validate_base(run)['passed'])

    def test_independent_baseline_source_seal(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);base=root/'baseline';build=root/'build/base'
            for folder in ('StiffGIPC/core','Assets','MeshProcess'):(base/folder).mkdir(parents=True)
            (root/'Assets').mkdir();(base/'Assets/scene.json').write_text('{}');(root/'Assets/scene.json').write_text('{}')
            build.mkdir(parents=True);(root/'compiler').write_text('compiler fixture')
            (base/'StiffGIPC/core/GIPC.cu').write_text('int Kmin = 6; if(beta <= Newton_solver_threshold) return;')
            (base/'CMakeLists.txt').write_text('fixture');(build/'gipc').write_text('binary');(build/'build.log').write_text('build evidence')
            (build/'CMakeCache.txt').write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={base}\nCMAKE_CXX_COMPILER:FILEPATH={root/"compiler"}\n')
            dump(build/'compile_commands.json',[{'file':str(base/'StiffGIPC/core/GIPC.cu'),'directory':str(build),'command':'nvcc -o CMakeFiles/gipc.dir/a.o -DGIPC_ASSETS_DIR='+str(base/'Assets')}])
            identity=driver.base_seal(root,'build/base/build.log')
            self.assertEqual(identity['exe_path'],'build/base/gipc');self.assertIn('CMAKE_CUDA_COMPILER',identity['compiler_identity']['missing'])
            self.assertFalse(identity['capabilities']['actual_velocity'])
            self.assertTrue(identity['assets_equivalence']['passed'])
            (root/'Assets/scene.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):driver.base_seal(root,'build/base/build.log')
            (root/'Assets/scene.json').write_text('{}')
            (build/'CMakeCache.txt').write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={root}\n')
            with self.assertRaises(ValueError):driver.base_seal(root,'build/base/build.log')

    def test_original_failed_guard_receipt_and_raw_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);session=root/'runs/old';session.mkdir(parents=True)
            protocol=root/'tools/bench/quality_protocol.json';protocol.parent.mkdir(parents=True);dump(protocol,{})
            names=['guards_hang_on','guards_fixed_on','guards_mixed_off','guards_mixed_on'];rows=[];entries=[]
            for n in names:
                run=session/n;run.mkdir();(run/'raw.bin').write_bytes(b'original raw evidence')
                dump(run/'evidence.json',{'files':inventory(run,[run/'raw.bin'])})
                fixed=n=='guards_fixed_on'
                rows.append({'name':n,'configuration':{'passed':True},'pool':{'passed':True},'hard_checks_passed':not fixed,
                             'failures':['ValueError: Frozen cloth material bounds failed'] if fixed else [],
                             'material':{'p99_stretch':1.02},'material_bounds':{'p99_stretch':1.01}})
                entries.append({'name':n,'result':{'status':'completed'},'evidence_sha256':sha(run/'evidence.json')})
            guard=session/'guards_analysis.json';batch=session/'guards_batch.json'
            value={'stage':'guards','allow_next_round':False,'source_digest':'s','exe_sha256':'e','protocol_sha256':sha(protocol),'runs':rows}
            dump(guard,value);dump(batch,{'runs':entries,'skipped':[]})
            dump(session/'guards_receipt.json',{'analysis_sha256':sha(guard),'batch_sha256':sha(batch)})
            active={'source_digest':'s','exe_sha256':'e'}
            self.assertTrue(driver.original_guard_evidence(root,guard,active)['remains_failed'])
            (session/names[0]/'raw.bin').write_bytes(b'tampered')
            with self.assertRaises(ValueError):driver.original_guard_evidence(root,guard,active)
            value['allow_next_round']=True;dump(guard,value)
            with self.assertRaises(ValueError):driver.original_guard_evidence(root,guard,active)

if __name__=='__main__':unittest.main()
