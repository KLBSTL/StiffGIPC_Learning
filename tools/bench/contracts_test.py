"""CPU synthetic contracts only. No build, SSH, GPU, or native executable."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import linux_runner as runner
import pool_analysis as analysis
from config import expand, digest, environment, matches_requested
from validate_run import validate
from plans import tasks, protocol
from pool_metrics import pool_evidence, POOL_SUM_FIELDS, POOL_DETAIL_FIELDS

def dump(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf-8')

def make_run(base,t,manifest):
    d=base/t['name'];(d/'trace').mkdir(parents=True);(d/'output').mkdir()
    c=expand(t['config']);steps=c['steps'];on=c['contact_pool'];guard=c['contact_pool_validate']
    dump(d/'requested.json',{'expanded_config':c,'source_digest':manifest['source_digest'],'exe_sha256':manifest['exe_sha256']})
    dump(d/'result.json',{'status':'completed','recorded_frames':steps})
    dump(d/'resolved_config.json',{'report_components':{'contact_pool':on,'contact_pool_validate':guard}})
    np.array([3,1,0,0,1,2],dtype='<u4').tofile(d/'trace/topology.bin')
    np.array([1.,1.,1.],dtype='<f8').tofile(d/'trace/masses.bin')
    np.zeros(3,dtype='<i4').tofile(d/'trace/boundary_types.bin')
    np.full(3,-1,dtype='<i4').tofile(d/'trace/body_ids.bin')
    dump(d/'trace/metadata.json',{'abd_point_num':0})
    dump(d/'output/scene.json',{'case_id':c['scene'],'objects':[{'dimension':2,'body_type':'FEM'}]})
    for i in range(steps+1):
        np.array([[0,0,0],[1,0,0],[0,1,0]],dtype='<f8').tofile(d/f'trace/state_{i:04d}.bin')
        np.zeros((3,3),dtype='<f8').tofile(d/f'trace/velocity_{i:04d}.bin')
    counters={k:0 for k in (*POOL_SUM_FIELDS,*POOL_DETAIL_FIELDS)}
    counters.update(requested=on,validate=guard,fallback_reasons={},pool_bytes_peak=64 if on else 0)
    if on:counters.update(attempts=1,reused_queries=1,prepare_calls=1,capture_passes=1,generations=1,pool_pairs=1,vf_pairs=1)
    else:counters['old_queries']=1
    if guard:
        counters.update(validation_calls=1,validation_passed=1,pairs_compared=1,nonempty_validation_calls=1,
                        energy_audit={'calls':1,'failed':0,'max_relative_error':0.,'max_barrier_relative_error':0.})
    energy={'alpha':1.,'candidate_energy':1.,'legacy_energy':1.,'relative_error':0.,'barrier_relative_error':0.,
            'armijo_branch_equal':True,'production_restored':True,'passed':True}
    frame={'newton':[{'pcg':{'iterations':1},'alpha':1.,'active_pairs':1}],
           'newton_exit':'movement','phase_ms':dict.fromkeys(('assembly','pcg','ccd','line_search','state_update'),1.),
           'contact_pool':counters,'contact_geometry':{'native_narrow_self_pairs':1}}
    if guard:frame['newton'][0]['contact_pool_energy']=[energy]
    dump(d/'output/stats.json',{'frames':[copy.deepcopy(frame) for _ in range(steps)]})
    (d/'trace/frames.csv').write_text('frame,solver_ms\n'+''.join(f'{i},{5 if on else 6}\n' for i in range(1,steps+1)))
    return d

class Contracts(unittest.TestCase):
    def test_finite_plan_and_ablation(self):
        p=protocol();self.assertEqual(sum(map(len,p['stages'].values())),16)
        self.assertEqual([x['name'] for x in tasks('r2')],['r2_fixed_on','r2_fixed_off','r2_hang_on','r2_hang_off'])
        for stage in ('guards','r1','r2','r3'):
            for t in tasks(stage):
                c=expand(t['config']);self.assertEqual(c['diagnostics'],[]);self.assertTrue(c['trace_velocity'])
                self.assertEqual(c['timeout_seconds'],120);self.assertEqual(c['backend'],'ipc')
                self.assertFalse(c['bounded_ccd']);self.assertFalse(c['bvh_eligibility'])
        for scene in ('hang','fixed'):
            a,b=[expand(t['config']) for t in tasks('r1') if t['scene_key']==scene]
            self.assertEqual({k for k in a if a[k]!=b[k]},{'contact_pool'})

    def test_pool_guard_missing_and_mutated_energy(self):
        manifest={'source_digest':'s','exe_sha256':'e'}
        with tempfile.TemporaryDirectory() as name:
            d=make_run(Path(name),tasks('guards')[0],manifest)
            frames=runner.read(d/'output/stats.json')['frames']
            self.assertTrue(pool_evidence(frames,True,True)['passed'])
            frames[0]['newton'][0]['contact_pool_energy'][0]['production_restored']=False
            self.assertFalse(pool_evidence(frames,True,True)['passed'])
            del frames[0]['contact_pool']['validation_passed']
            self.assertFalse(pool_evidence(frames,True,True)['passed'])

    def test_analysis_actual_velocity_and_input_gate(self):
        manifest={'source_digest':'s','exe_sha256':'e'}
        with tempfile.TemporaryDirectory() as name,patch.object(analysis,'validate',return_value={'passed':True}):
            base=Path(name)
            for t in tasks('r1'):make_run(base,t,manifest)
            report=analysis.analyze(base,'r1',manifest)
            self.assertTrue(report['allow_next_round']);self.assertFalse(report['quality_certified'])
            self.assertAlmostEqual(report['pairs']['hang']['whole_speedup'],1.2)
            (base/'r1_hang_on/trace/velocity_0001.bin').unlink()
            self.assertFalse(analysis.analyze(base,'r1',manifest)['allow_next_round'])

    def test_mixed_quality_pending_but_hard_failure_blocks(self):
        manifest={'source_digest':'s','exe_sha256':'e'}
        with tempfile.TemporaryDirectory() as name,patch.object(analysis,'validate',return_value={'passed':True}):
            base=Path(name)
            for t in tasks('guards'):make_run(base,t,manifest)
            report=analysis.analyze(base,'guards',manifest)
            self.assertTrue(report['allow_next_round']);self.assertFalse(report['mixed_quality_certified'])
            self.assertNotIn('whole_speedup',report['pairs']['mixed'])
            d=base/'guards_mixed_on';stats=runner.read(d/'output/stats.json');stats['frames'][0]['newton'][0]['pcg']['breakdown']=True
            dump(d/'output/stats.json',stats)
            self.assertFalse(analysis.analyze(base,'guards',manifest)['allow_next_round'])

    def test_safe_path_and_identity_tamper(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for unsafe in ('../other',str(root/'outside')):
                with self.assertRaises(ValueError):runner.child(root,unsafe)
            f=root/'file';f.write_text('first');rows=runner.inventory(root,[f]);runner.verify_files(root,rows)
            f.write_text('other')
            with self.assertRaises(ValueError):runner.verify_files(root,rows)

    def test_build_seal_exact_source_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for folder in ('StiffGIPC','MeshProcess','Assets','tools/bench','tests','build'):(root/folder).mkdir(parents=True)
            for file in ('StiffGIPC/a.cu','CMakeLists.txt','FLATTEN_SOURCE_MAP.json','build/gipc','build/CMakeCache.txt','build/build.log'):
                (root/file).write_text('fixture')
            dump(root/'build/compile_commands.json',[{'file':str(root/'StiffGIPC/a.cu'),'directory':str(root/'build'),'command':'nvcc -o CMakeFiles/gipc.dir/a.cu.o'}])
            runner.seal_build(root,'build','build/gipc','build/build.log','build/manifest.json')
            runner.verify_manifest(root,root/'build/manifest.json')
            (root/'StiffGIPC/a.cu').write_text('changed')
            with self.assertRaises(ValueError):runner.verify_manifest(root,root/'build/manifest.json')

    def test_gpu_queries_receive_bounded_timeout(self):
        from types import SimpleNamespace
        with patch.object(runner.subprocess,'run',return_value=SimpleNamespace(stdout='GPU-x,4090,580,0,24000,30,10,100,8.9\n')) as query:
            runner.gpu_query(0,timeout=.025)
            self.assertEqual(query.call_args.kwargs['timeout'],.025)
        with patch.object(runner.subprocess,'run',return_value=SimpleNamespace(stdout='')) as query:
            self.assertEqual(runner.compute_pids(0,timeout=.01),[])
            self.assertEqual(query.call_args.kwargs['timeout'],.01)

    def test_prior_gate_ledger_and_payload_tamper(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);rows=[]
            for t in tasks('guards'):
                d=root/t['name'];dump(d/'result.json',{'status':'completed'})
                dump(d/'evidence.json',{'files':runner.inventory(d,[d/'result.json'])})
                rows.append({'name':t['name'],'result':{'status':'completed'},'evidence_sha256':runner.sha(d/'evidence.json')})
            dump(root/'guards_batch.json',{'tasks':tasks('guards'),'manifest_sha256':'m','runs':rows})
            dump(root/'guards_analysis.json',{'stage':'guards','allow_next_round':True})
            dump(root/'guards_receipt.json',{'batch_sha256':runner.sha(root/'guards_batch.json'),'analysis_sha256':runner.sha(root/'guards_analysis.json'),
                 'manifest_sha256':'m','plan_sha256':digest(protocol())})
            runner.verify_stage(root,'guards','m')
            dump(root/'guards_hang_on/result.json',{'status':'failed'})
            with self.assertRaises(ValueError):runner.verify_stage(root,'guards','m')


def make_graph_config_run(base,chunk=1):
    """Only the resolved/observed contract; no native execution or quality claim."""
    c=expand({'execution':'conditional_graph','pcg_graph_chunk':chunk,'steps':1})
    dump(base/'requested.json',{'binary':'active','expanded_config':c})
    dump(base/'result.json',{'status':'completed','recorded_frames':1})
    dump(base/'resolved_config.json',{
        'contact_backend':c['backend'],'dt':c['dt'],'ipc_newton_tol':c['ipc_newton_tol'],
        'pcg_rho_tol':c['pcg_rho_tol'],'configured_pcg_execution':c['execution'],
        'configured_pcg_graph_chunk':chunk,
        'fixed_graph_chunk_study':False,
        'mas':{'cholesky':False,'inverse64':False,'wide_apply':False},
        'acceleration_features':{'GIPC_CCD_BVH_REFIT':False,'GIPC_BATCHED_ENERGY':False,'GIPC_ENERGY_REUSE':False}})
    dump(base/'output/stats.json',{'frames':[{'newton':[{'pcg':{
        'execution':'conditional_graph','graph_chunk_iterations':chunk,
        'fused_diag_update':False,'iterations':4}}]}]})
    return c


class GraphConfigContracts(unittest.TestCase):
    def test_default_and_explicit_graph_chunk(self):
        c=expand({});self.assertEqual(c['pcg_graph_chunk'],1)
        self.assertEqual(environment(c,Path('fixture'))['GIPC_PCG_GRAPH_CHUNK'],'1')
        c=expand({'preset':'graph','pcg_graph_chunk':4})
        self.assertEqual(c['pcg_graph_chunk'],4)
        env=environment(c,Path('fixture'))
        self.assertEqual(env['GIPC_PCG_GRAPH_CHUNK'],'4')
        self.assertEqual(env['GIPC_PCG_FUSED_DIAG_UPDATE'],'0')

    def test_chunk_type_range_and_execution_rejected(self):
        for value in (0,2,8,-1,True,False,1.0,4.0,'1','4',None,[],{}):
            with self.subTest(value=value),self.assertRaises(ValueError):
                expand({'preset':'graph','pcg_graph_chunk':value})
        with self.assertRaises(ValueError):expand({'pcg_graph_chunk':4})
        # Fused diagonal execution remains outside this portable experiment
        # contract; do not introduce that retired option while adding K=4.
        with self.assertRaises(ValueError):
            expand({'preset':'graph','pcg_graph_chunk':4,'fused_diag_update':True})

    def test_ambient_chunk_and_fusion_cannot_leak(self):
        with patch.dict('os.environ',{'GIPC_PCG_GRAPH_CHUNK':'4','GIPC_PCG_FUSED_DIAG_UPDATE':'1'}):
            env=environment(expand({}),Path('fixture'))
        self.assertEqual(env['GIPC_PCG_GRAPH_CHUNK'],'1')
        self.assertEqual(env['GIPC_PCG_FUSED_DIAG_UPDATE'],'0')

    def test_historical_request_compatibility_does_not_hide_k4(self):
        old=expand({'preset':'graph'});old.pop('pcg_graph_chunk');old.pop('fixed_graph_chunk_study')
        self.assertTrue(matches_requested(old,{'preset':'graph'}))
        self.assertFalse(matches_requested(old,{'preset':'graph','pcg_graph_chunk':4}))
        current=expand({'preset':'graph','pcg_graph_chunk':4})
        self.assertFalse(matches_requested(current,{'preset':'graph','pcg_graph_chunk':1}))

    def test_resolved_and_observed_chunk_agree(self):
        for chunk in (1,4):
            with self.subTest(chunk=chunk),tempfile.TemporaryDirectory() as name:
                base=Path(name);make_graph_config_run(base,chunk)
                self.assertTrue(validate(base)['passed'])

    def test_host_pcg_has_no_graph_observation(self):
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);make_graph_config_run(base)
            requested=runner.read(base/'requested.json')
            requested['expanded_config']=expand({'execution':'host','steps':1})
            dump(base/'requested.json',requested)
            resolved=runner.read(base/'resolved_config.json')
            resolved['configured_pcg_execution']='host'
            dump(base/'resolved_config.json',resolved)
            stats=runner.read(base/'output/stats.json')
            record=stats['frames'][0]['newton'][0]['pcg']
            record['execution']='host';record.pop('graph_chunk_iterations')
            dump(base/'output/stats.json',stats)
            result=validate(base)
            self.assertTrue(result['passed'])
            self.assertNotIn('pcg_graph_chunk.observed',{c['check'] for c in result['checks']})
            # A forged host/K4 request must still fail without inventing a host
            # graph statistic; the execution/configuration restriction suffices.
            requested['expanded_config']['pcg_graph_chunk']=4
            resolved['configured_pcg_graph_chunk']=4
            dump(base/'requested.json',requested);dump(base/'resolved_config.json',resolved)
            self.assertFalse(validate(base)['passed'])

    def test_missing_or_mistyped_resolved_chunk_rejected_for_new_request(self):
        for value in (None,1,True,4.0,'4'):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as name:
                base=Path(name);make_graph_config_run(base,4)
                r=runner.read(base/'resolved_config.json')
                if value is None:r.pop('configured_pcg_graph_chunk')
                else:r['configured_pcg_graph_chunk']=value
                dump(base/'resolved_config.json',r)
                self.assertFalse(validate(base)['passed'])
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);make_graph_config_run(base,1)
            r=runner.read(base/'resolved_config.json');r.pop('configured_pcg_graph_chunk')
            dump(base/'resolved_config.json',r)
            self.assertFalse(validate(base)['passed'])

    def test_observed_chunk_and_fusion_cannot_silently_change(self):
        for value in (None,1,True,4.0,'4'):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as name:
                base=Path(name);make_graph_config_run(base,4)
                stats=runner.read(base/'output/stats.json');p=stats['frames'][0]['newton'][0]['pcg']
                if value is None:p.pop('graph_chunk_iterations')
                else:p['graph_chunk_iterations']=value
                dump(base/'output/stats.json',stats)
                self.assertFalse(validate(base)['passed'])
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);make_graph_config_run(base,4)
            stats=runner.read(base/'output/stats.json')
            stats['frames'][0]['newton'][0]['pcg']['fused_diag_update']=True
            dump(base/'output/stats.json',stats)
            self.assertFalse(validate(base)['passed'])

    def test_old_k1_result_not_rewritten_or_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);make_graph_config_run(base,1)
            request=runner.read(base/'requested.json');request['expanded_config'].pop('pcg_graph_chunk')
            request['expanded_config'].pop('fixed_graph_chunk_study')
            resolved=runner.read(base/'resolved_config.json');resolved.pop('configured_pcg_graph_chunk')
            resolved.pop('fixed_graph_chunk_study')
            stats=runner.read(base/'output/stats.json');stats['frames'][0]['newton'][0]['pcg'].pop('graph_chunk_iterations')
            dump(base/'requested.json',request);dump(base/'resolved_config.json',resolved);dump(base/'output/stats.json',stats)
            before={p:runner.sha(base/p) for p in ('requested.json','resolved_config.json','output/stats.json')}
            self.assertTrue(validate(base)['passed'])
            self.assertEqual(before,{p:runner.sha(base/p) for p in before})

    def test_fixed_graph_chunk_study_selector_and_environment(self):
        for frame in ('2','57'):
            c=expand({'preset':'graph','diagnostics':['fixed'],'steps':59,
                      'fixed_frames':frame,'fixed_directions':'1','fixed_graph_chunk_study':True})
            env=environment(c,Path('fixture'))
            self.assertEqual(env['GIPC_FIXED_GRAPH_CHUNK_STUDY'],'1')
            self.assertEqual(env['GIPC_FIXED_STUDY_FRAMES'],frame)
            self.assertEqual(env['GIPC_FIXED_STUDY_DIRECTIONS'],'1')
        c=expand({});self.assertIs(c['fixed_graph_chunk_study'],False)
        self.assertEqual(environment(c,Path('fixture'))['GIPC_FIXED_GRAPH_CHUNK_STUDY'],'0')

    def test_fixed_graph_chunk_study_scope_and_types_rejected(self):
        base={'preset':'graph','diagnostics':['fixed'],'steps':2,
              'fixed_frames':'2','fixed_directions':'1','fixed_graph_chunk_study':True}
        for override in ({'diagnostics':[]},{'execution':'host'},{'pcg_graph_chunk':4},
                         {'mas':'cholesky'},{'profile':'node','diagnostics':['fixed','cost']},
                         {'fixed_graph_chunk_study':1},{'fixed_graph_chunk_study':'true'}):
            with self.subTest(override=override),self.assertRaises(ValueError):expand(base|override)
        for key in ('fixed_restrict_study','fixed_factor_study','fixed_mas_stage_study',
                    'fixed_legacy_restrict_study','fixed_mas_dot_study','fixed_spmv_quadratic_study'):
            with self.subTest(study=key),self.assertRaises(ValueError):expand(base|{key:True})

    def test_fixed_graph_chunk_study_resolved_flag_is_explicit(self):
        for actual in (None,False,1,'true',True):
            with self.subTest(actual=actual),tempfile.TemporaryDirectory() as name:
                base=Path(name);make_graph_config_run(base)
                req=runner.read(base/'requested.json')
                req['expanded_config']=expand({'preset':'graph','steps':1,'diagnostics':['fixed'],
                    'fixed_frames':'1','fixed_directions':'1','fixed_graph_chunk_study':True})
                dump(base/'requested.json',req)
                resolved=runner.read(base/'resolved_config.json')
                if actual is None:resolved.pop('fixed_graph_chunk_study')
                else:resolved['fixed_graph_chunk_study']=actual
                dump(base/'resolved_config.json',resolved)
                self.assertEqual(validate(base)['passed'],actual is True)

if __name__=='__main__':unittest.main()
