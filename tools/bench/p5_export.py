"""Export finite probe evidence. No inferred GPU/scene speedup is certified."""
import json
from pathlib import Path

from m3_numeric import ROOT,need
from local_identity import record,verify
from linux_runner import write_new
from m0_analyze import csv_write

REPORT=ROOT/'reports/innovation_p5_gpu_20261015'
RAW=ROOT/'runs/innovation_p5_gpu_20261015'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    output=REPORT/'tables';need(not output.exists(),'Fresh table output');output.mkdir()
    review=read(REPORT/'REVIEW.json');need(review['passed'],'M3 review incomplete')
    launches=[]
    for name in ('fixture_positive','fixture_deficient','mixed_cost','fixture_memcheck'):
        row=read(RAW/name/'launch.json');need(row['passed'] and row['cleanup_owned_job_empty'],'Run incomplete')
        samples=row['samples'];launches.append(dict(name=name,status=row['status'],exit_code=row['exit_code'],
            wall_seconds=row['wall_seconds'],power_guid=row['power_guid'],
            prelaunch_GPU_utilization=row['gpu_before']['utilization.gpu'],
            maximum_free_memory_drop_mib=row['gpu_before']['memory.free']-min(x['memory.free'] for x in samples),
            native_timing_certified=False))
    log=(RAW/'fixture_memcheck/run.log').read_text(encoding='utf-8-sig',errors='replace')
    need('ERROR SUMMARY: 0 errors' in log,'Compute Sanitizer result missing')
    validation_paths=[RAW/'fixture_positive/validation_v2.json',RAW/'mixed_cost/validation.json',RAW/'fixture_memcheck/validation.json']
    for path in validation_paths:
        v=read(path);need(v['passed'],'Independent check failed');verify(v['inputs']+v['results'])
    v=read(validation_paths[1]);run=read(RAW/'mixed_cost/data/result.json')
    need(read(validation_paths[0])['costs']==[] and read(validation_paths[2])['costs']==[],
         'Tiny fixture received a real scene cost budget')
    setup=[];components=[];apply=[];cold=[]
    for entry in run['ranks']:
        rank=entry['rank']
        for i,x in enumerate(entry['pipeline_samples']):setup.append(dict(rank=rank,sample=i,**x))
        for stage,x in entry['component_times'].items():components.append(dict(rank=rank,stage=stage,**x))
        for x in entry['apply_samples']:apply.append(dict(rank=rank,**x))
        cold.append(dict(rank=rank,first_pipeline_GPU_ms=entry['first_pipeline']['gpu_ms'],
            first_pipeline_wall_ms=entry['first_pipeline']['wall_ms'],
            MAS_graph_capture_ms=entry['graph_capture_ms']['base'],balanced_graph_capture_ms=entry['graph_capture_ms']['balanced']))
    costs=[{k:x for k,x in row.items() if k not in ('paired_samples','interpretation')} for row in v['costs']]
    correct=[{k:(max(x) if isinstance(x,list) else x) for k,x in row.items()} for row in v['checks']]
    for name,rows in (('launches',launches),('pipeline',setup),('components',components),
                      ('apply_pairs',apply),('conditional_budget',costs),('numerical_checks',correct),('cold_cost',cold)):
        csv_write(output/(name+'.csv'),rows)
    write_new(output/'validation.json',dict(passed=True,source=record(Path(__file__)),
        positive_independent_validations=[record(p) for p in validation_paths],
        review=record(REPORT/'REVIEW.json'),GPU_jobs=4,expected_rejections=1,memcheck_errors=0,
        native_solver_modified=False,native_PCG_implemented=False,performance_certified=False))
    print(json.dumps(dict(passed=True,tables=7,GPU_jobs=4,memcheck_errors=0)),flush=True)


if __name__=='__main__':main()
