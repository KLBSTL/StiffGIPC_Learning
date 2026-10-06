"""CPU-only report contracts; synthetic metadata, no solver or GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import report as r


def rows():
    result=[]
    for case in r.CASES:
        for arm,seconds in zip(r.ARMS,(10.,12.,8.,6.)):
            result.append({'index':len(result)+1,'name':f'p1_{case}_{arm}','scene_key':case,'arm':arm,
                'binary':'base' if arm=='original_stiff' else 'active','phase':'paired','pair':1,
                'hard_checks_passed':True,'complete_hard_pass':True,'ledger_status':'completed',
                'gpu_uuid':'fixturegpu','input_identity':{'topology':'same','initial_actual_velocity':None},
                'timing':{'solver_seconds':seconds,'core_cuda_event_seconds':seconds-.1,'process_wall_seconds':seconds+1,
                    'phase_ms':{k:100 for k in r.PHASES},'exit_assembly_ms':None},
                'material':{'max_stretch':1.01,'fem_min_J':None},
                'work':{'linear_directions':10,'pcg_iterations':100},'paired_comparisons':[]})
    return result


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.session=self.root/'session';self.session.mkdir()

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8');return path

    def fixture(self):
        data=rows();tasks=[];previous=None
        for row in data:
            task={k:row[k] for k in ('index','name','scene_key','arm','binary','phase','pair')}
            task['config']={'scene':r.CASES[row['scene_key']],'steps':100};tasks.append(task)
            run=self.session/row['name'];run.mkdir()
            result={'status':'completed','recorded_frames':100,'exit_code':0}
            frames=[{'contact_geometry':{'native_narrow_self_pairs':5 if i==40 else 0,
                'native_narrow_ground_pairs':0,'geometric_contact':False},'newton':[]} for i in range(100)]
            files=[self.write(run/'result.json',result),self.write(run/'requested.json',{'expanded_config':{}}),
                   self.write(run/'output/stats.json',{'frames':frames})]
            if row['binary']=='active':
                files.append(self.write(run/'resolved_config.json',{'acceleration_features':{'GIPC_MAS_STATIC_TOPOLOGY':False}}))
            inv=self.write(run/'evidence.json',{'files':[{'path':p.relative_to(run).as_posix(),
                        'bytes':p.stat().st_size,'sha256':r.sha(p)} for p in files]})
            analysis={k:v for k,v in row.items() if k not in ('complete_hard_pass','ledger_status')}
            ap=self.write(self.session/'analysis'/f"{row['name']}.json",analysis)
            sp=self.write(self.session/'summary'/f"{row['index']:04d}.json",{})
            lp=self.write(self.session/'ledger'/f"{row['index']:04d}.json",{'task':task,'index':row['index'],
                'predecessor_sha256':previous,'status':'completed','run_created':True,'analysis_exception':False,
                'analysis':analysis,'analysis_path':str(ap.relative_to(self.session)), 'analysis_sha256':r.sha(ap),
                'summary_path':str(sp.relative_to(self.session)),'summary_sha256':r.sha(sp),
                'result':result,'evidence_sha256':r.sha(inv)})
            previous=r.sha(lp)
        self.write(self.session/'plan.json',{'tasks':tasks})
        stat=lambda n:{'solver':{'paired_median':1.25,'pairs_observed':n,'pairs_required':n,'complete':True}}
        self.prior=self.write(self.root/'prior.json',{'scene_statistics':{c:{'main_original_to_combined':stat(7),
            'graph_component':stat(3),'combined_execution_component':stat(3)} for c in ('hang','fixed','mixed')}})

    def test_four_edges_and_regression_not_hidden(self):
        cases=r.case_summary(rows());v=cases['hang_l']['comparisons']['original_stiff/ipc_host']
        self.assertAlmostEqual(v['solver']['ratio'],10/12)
        self.assertAlmostEqual(v['solver']['saved_fraction'],-.2)
        self.assertEqual(v['solver']['saved_seconds'],-2)
        self.assertIsNone(v['confidence_interval']);self.assertFalse(v['quality_certified'])
        self.assertEqual(len(cases['sphere_m']['comparisons']),4)

    def test_failed_case_no_partial_ratios(self):
        data=rows();data[3]['complete_hard_pass']=False
        self.assertTrue(all(not x['solver']['available'] for x in r.case_summary(data)['hang_l']['comparisons'].values()))
        data=rows();data[1]['gpu_uuid']='different'
        self.assertFalse(r.case_summary(data)['hang_l']['comparisons']['original_stiff/ipc_host']['comparable'])

    def test_narrow_pairs_not_tight_geometric_or_missing_zero(self):
        frames=[{'contact_geometry':{'native_narrow_self_pairs':5,'native_narrow_ground_pairs':0,'geometric_contact':False}}]
        c=r.contact_summary(frames,1)
        self.assertEqual(c['native_narrow_nonzero_frame_count'],1);self.assertEqual(c['geometric_contact_frame_count'],0)
        missing=r.contact_summary(frames,2)
        self.assertIsNone(missing['native_narrow_nonzero_frame_count']);self.assertFalse(missing['complete'])
        self.assertEqual(missing['native_narrow_first_nonzero_frame'],1)

    def test_material_is_signed_observation_not_acceptance(self):
        d=r.compare_material({'max_stretch':1.01,'fem_min_J':None},{'max_stretch':1.05,'fem_min_J':None})
        self.assertAlmostEqual(d['metrics']['max_stretch']['signed_worsening_gap'],.04)
        self.assertFalse(d['metrics']['fem_min_J']['available']);self.assertIsNone(d['acceptance_threshold'])
        self.assertNotIn('passed',d)

    def test_real_loader_outputs_and_no_overwrite(self):
        self.fixture();d=r.analyze(self.session,self.prior)
        self.assertEqual(d['completed_hard_pass'],16);self.assertIsNone(d['confidence_intervals'])
        self.assertEqual(d['runs'][0]['contact']['native_narrow_first_nonzero_frame'],41)
        self.assertEqual(d['prior_experiment']['cases']['fixed']['Stiff/combined']['pairs_observed'],7)
        text=r.markdown(d);self.assertIn('MAS 静态拓扑复用关闭',text);self.assertIn('无置信区间',text)
        dest=self.root/'reports';r.write_reports(d,dest)
        self.assertEqual(len((dest/'RUN_INDEX.csv').read_text(encoding='utf-8-sig').splitlines()),17)
        with self.assertRaisesRegex(ValueError,'no overwrite'):r.write_reports(d,dest)

    def test_changed_stats_rejected(self):
        self.fixture();path=self.session/'p1_hang_l_original_stiff/output/stats.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Run metadata changed'):r.analyze(self.session,self.prior)

    def test_changed_embedded_analysis_rejected(self):
        self.fixture();path=self.session/'ledger/0001.json';p=r.read(path)
        p['analysis']['timing']['solver_seconds']=100;self.write(path,p)
        with self.assertRaisesRegex(ValueError,'Embedded analysis differs'):r.analyze(self.session,self.prior)

    def test_retired_fallback_reasons_remain_explicit(self):
        frames=[{'newton':[{'pcg':{'execution':'host','graph_fallback_reason':'capture_error',
            'mas_fused_dot_fallback_reason':'retired_component'}}]}]
        d=r.execution_summary(frames,{'environment':{'GIPC_ACCEL_SUITE':'0'}},
            {'acceleration_features':{'GIPC_MAS_STATIC_TOPOLOGY':False}})
        self.assertEqual(d['observed_pcg_execution'],{'host':1});self.assertEqual(len(d['fallback_reason_counts']),2)
        self.assertIs(d['mas_static_topology_effective'],False)

    def test_unexpected_mas_on_is_not_reported_off(self):
        self.fixture();run=self.session/'p1_hang_l_ipc_host'
        path=self.write(run/'resolved_config.json',{'acceleration_features':{'GIPC_MAS_STATIC_TOPOLOGY':True}})
        inv=r.read(run/'evidence.json')
        for entry in inv['files']:
            if entry['path']=='resolved_config.json':entry.update(bytes=path.stat().st_size,sha256=r.sha(path))
        self.write(run/'evidence.json',inv)
        previous=None
        for path in sorted((self.session/'ledger').glob('*.json')):
            entry=r.read(path);entry['predecessor_sha256']=previous
            if entry['task']['name']==run.name:entry['evidence_sha256']=r.sha(run/'evidence.json')
            self.write(path,entry);previous=r.sha(path)
        with self.assertRaisesRegex(ValueError,'Unexpected MAS static topology'):r.analyze(self.session,self.prior)


if __name__=='__main__':unittest.main()
