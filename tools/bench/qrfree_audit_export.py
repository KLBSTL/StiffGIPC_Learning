"""Audit saved solutions and export complete paired costs without scene certification."""
import json
from pathlib import Path
import numpy as np
from m3_numeric import ROOT,FrozenSystem,residual_metrics,need
from cpu_fixed_reference import decimal_residual
from local_identity import record,verify
from linux_runner import write_new
from m0_analyze import csv_write

RAW=ROOT/'runs/innovation_p5_qrfree_20261016'
REPORT=ROOT/'reports/innovation_p5_qrfree_20261016'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    need(read(RAW/'CPU_summary.json')['passed'],'CPU summary failed')
    out=REPORT/'tables_final';need(not out.exists(),'Fresh tables required');out.mkdir()
    final=[];cpu=[];curves=0
    for scene,frame in (('Mixed',15),('Bunny',14),('Boxes',17)):
        folder=RAW/scene;result=read(folder/'result.json');verify(result['inputs'])
        system=FrozenSystem(ROOT/f'runs/innovation_m3_20261015/adjacent_{scene.lower()}/fixed/f{frame}_n2')
        trace=read(folder/'curves.json');curves+=len(trace)
        for method in result['methods']:
            verify([method['solution']]);x=np.load(method['solution']['path'],allow_pickle=False)
            metrics,_=residual_metrics(system.A,system.b,x,system.apply);_,dec=decimal_residual(system.A,system.b,x)
            steps=[t for t in trace if t['method']==method['method']]
            need(len(steps)==method['iterations'] and [t['iteration'] for t in steps]==list(range(1,len(steps)+1)),
                 'Incomplete per-iteration curve')
            need(max(metrics.values())<=.01 and abs(metrics['relative_l2']-dec['relative_l2'])<1e-9,'Final scalar residual failed')
            for key,value in metrics.items():
                need(abs(value-method[key])<1e-12 and abs(value-steps[-1][key])<1e-12,'Final saved metrics mismatch')
            row=dict(scene=scene,method=method['method'],iterations=method['iterations'],**metrics,
                Decimal50_relative_l2=dec['relative_l2'],coarse_condition=result['coarse_condition'],
                maximum_action_error=max(result['action_relative_errors']))
            final.append(row);cpu.extend(dict(scene=scene,**t) for t in steps)
    launches=[]
    for name in ('fixture_positive','fixture_negative','mixed_cost','fixture_memcheck_v2'):
        r=read(RAW/name/'launch.json');need(r['passed'] and r['cleanup_owned_job_empty'],'GPU Job not complete')
        verify([r['runner']]+r['identity']['sources']+r['identity']['evidence']+[r['identity']['exe']])
        before=r['gpu_before'];samples=r['samples'];need(samples and all(s['power_guid']==r['power_guid'] for s in samples),'Power changed')
        launches.append(dict(name=name,status=r['status'],exit_code=r['exit_code'],wall_seconds=r['wall_seconds'],
            power_guid=r['power_guid'],prelaunch_GPU_utilization=before['utilization.gpu'],
            memory_drop_mib=before['memory.free']-min(s['memory.free'] for s in samples),
            desktop_load_uncontrolled=r['desktop_load_uncontrolled']))
        if name!='fixture_negative':
            v=read(RAW/name/'validation.json');need(v['passed'],'GPU validation failed');verify(v['inputs']+v['results']+[v['validator']])
    failed=read(RAW/'fixture_memcheck/launch.json')
    need(not failed['passed'] and failed['cleanup_owned_job_empty'],'Failed sandbox Job not retained/cleaned')
    failed_log=(RAW/'fixture_memcheck/run.log').read_text(encoding='utf-8-sig',errors='replace')
    need('No attachable process found' in failed_log,'Unexpected first sanitizer failure')
    text=(RAW/'fixture_memcheck_v2/run.log').read_text(encoding='utf-8-sig',errors='replace')
    need('ERROR SUMMARY: 0 errors' in text,'No memcheck success')
    validation=read(RAW/'mixed_cost/validation.json');run=read(RAW/'mixed_cost/data/result.json')
    components=[];apply=[];setup=[]
    for case in run['cases']:
        tag=case['method']
        components.extend(dict(method=tag,stage=k,**v) for k,v in case['components'].items())
        apply.extend(dict(method=tag,**v) for v in case['apply_samples'])
        setup.extend(dict(method=tag,**v) for v in case['pipeline_samples'])
    for name,rows in (('CPU_final',final),('CPU_curves',cpu),('launches',launches),('components',components),
                      ('apply',apply),('setup',setup),('setup_pairs',validation['setup_pairs']),
                      ('conditional_budget',validation['conditional']),('GPU_checks',validation['checks'])):
        csv_write(out/(name+'.csv'),rows)
    tests=(ROOT/'build/qrfree_all_python_tests_v2.log').read_text(encoding='utf-8-sig')
    need('Ran 90 tests' in tests and tests.rstrip().endswith('OK'),'90 tests not passed')
    write_new(REPORT/'AUDIT.json',dict(passed=True,source=record(Path(__file__)),solutions_rechecked=len(final),
        Decimal50_checks=len(final),curve_rows=curves,GPU_jobs=5,successful_GPU_jobs=4,failed_sandbox_GPU_jobs=1,
        expected_rejections=1,memcheck_errors=0,Python_tests=90,
        native_PCG=False,production_modified=False,performance_certified=False))
    print(json.dumps(dict(passed=True,solutions=len(final),curve_rows=curves,Python_tests=90,memcheck_errors=0)),flush=True)


if __name__=='__main__':main()
